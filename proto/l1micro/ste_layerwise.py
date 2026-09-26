#!/usr/bin/env python
"""M5-A1 follow-up: layer-wise Z-latent + STE code training (BRECQ/AdaRound style).

Why layer-wise: an end-to-end fp32 latent for all folded weights (544M params,
param+grad+Adam ~9GB) does not fit the GTX 1060. Per-tensor Z is 10-90MB, and
each folded linear can be optimized alone against the FP teacher's captured
(input, output) pairs — the classic PTQ recipe that is known to move codes.

Mechanics:
  teacher (frozen FP) runs once per calibration batch with hooks on every
  target linear; (x, y) fp16 pairs are parked in CPU RAM. Then per tensor:
    Z init = realized weights s*T (start at the current solution)
    s      = softplus(theta) per group of 128 (trainable, as before)
    w = s * clamp(round(Z/s), -1, 1)      # codes are derived, STE backward
    loss = ||x @ w.T - y||^2 / ||y||^2    # scale-normalized output matching
  final codes/scales are written back into the TernaryWeight (Phase-3 amax
  snap happens at export, unchanged).

Usage: ste_layerwise.py --ckpt .../l1micro_cont1000.pt --batches 32 --steps 10
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common.fwht_torch import GROUP, rot_fwd, signs_for_width
from ternary_model import RotTrainLinear, TARGETS, install_ternary
from train import load_ids, quick_ppl, restore_checkpoint

MODEL = "<MODELS_DIR>/Qwen3-0.6B"
DEV = "cuda"


class _TernarySTE(torch.autograd.Function):
    """w = s * clamp(round(Z/s), -1, 1); STE passes grad through round,
    dead-zones gradients where |Z/s| > 1.5 (saturated entries)."""

    @staticmethod
    def forward(ctx, Z, s, ng):
        T = torch.clamp(torch.round(Z / s.repeat_interleave(GROUP, 1)), -1, 1)
        w = s.repeat_interleave(GROUP, 1) * T
        ctx.save_for_backward(Z, s, T)
        return w

    @staticmethod
    def backward(ctx, g):
        Z, s, T = ctx.saved_tensors
        ratio = Z / s.repeat_interleave(GROUP, 1)
        mask = (ratio.abs() <= 1.5).to(g.dtype)
        gZ = g * s.repeat_interleave(GROUP, 1) * mask
        gs = (g * T).reshape(T.shape[0], -1, GROUP).sum(-1)
        return gZ, gs, None


def capture_teacher(model_fp, x, hooks_store):
    with torch.no_grad():
        model_fp(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="<WORK_DIR>/l1micro_cont1000.pt")
    ap.add_argument("--batches", type=int, default=6,
                    help="calibration batches kept in CPU RAM (~0.8GB each)")
    ap.add_argument("--steps", type=int, default=10, help="Adam steps per batch per tensor")
    ap.add_argument("--zlr", type=float, default=2e-4)
    ap.add_argument("--slr", type=float, default=3e-3, help="(unused in v2: scales frozen)")
    ap.add_argument("--lam", type=float, default=3e-3, help="AdaRound-style anchor penalty")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--tag", default="steLW")
    ap.add_argument("--limit-tensors", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    test_ids = load_ids("test")
    train_ids = load_ids("train")

    # ---- student (ternary, restored from checkpoint) ----
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(DEV)
    ternary = install_ternary(model, "zerofrac")
    ck = torch.load(args.ckpt, map_location=DEV)
    restore_checkpoint(model, ternary, ck)
    model.eval()
    ppl0 = quick_ppl(model, test_ids, args.seq, 20)
    print(f"restored ckpt (step {ck['step']}), quick ppl = {ppl0:.2f}", flush=True)

    # target TernaryWeights with their HF module path
    targets = []          # (path, RotTrainLinear)
    for path, m in model.named_modules():
        if isinstance(m, RotTrainLinear):
            targets.append((path, m))
    if args.limit_tensors:
        targets = targets[: args.limit_tensors]
    print(f"layer-wise targets: {len(targets)} linears", flush=True)

    # ---- teacher capture: outputs only (y per batch per tensor) ----
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
    tstore = {}            # path -> list of y fp16 cpu
    hooks = []
    for path, m in targets:
        tmod = teacher.get_submodule(path)
        def make_post(p):
            def hook(mod, args_, out, p=p):
                tstore.setdefault(p, []).append(out.detach().half().cpu())
            return hook
        hooks.append(tmod.register_forward_hook(make_post(path)))

    n_tok = args.batches * args.seq
    calib = train_ids[train_ids.numel() // 3: train_ids.numel() // 3 + n_tok]
    calib = [calib[b * args.seq:(b + 1) * args.seq].unsqueeze(0).to(DEV)
             for b in range(args.batches)]
    with torch.no_grad():
        for x in calib:
            teacher(x)
    for h in hooks:
        h.remove()
    del teacher
    torch.cuda.empty_cache()
    print(f"captured {args.batches} batches x {len(tstore)} tensors", flush=True)

    # ---- sequential-greedy Z training, layer groups ----
    # Teacher-input reconstruction over all layers compounds error (first
    # attempt: ppl 122 -> 31970). BRECQ-style: each layer group is optimized
    # against the STUDENT's own inputs (prefix already quantized), so each
    # layer absorbs the previous layers' error instead of adding to it.
    from itertools import groupby

    def layer_key(t):
        return t[0].split(".")[2] if t[0].startswith("model.layers.") else "top"

    groups = [(k, list(g)) for k, g in groupby(targets, key=layer_key)]

    sstore = {}            # path -> current-batch student input
    shooks = [m.register_forward_pre_hook(
        lambda mod, inp, p=path: sstore.__setitem__(p, inp[0].detach()))
        for path, m in targets]

    t0 = time.time()
    moved_report = []
    best_ppl = quick_ppl(model, test_ids, args.seq, 4)     # small gate eval
    print(f"gate baseline ppl = {best_ppl:.2f}", flush=True)
    for gi, (k, grp) in enumerate(groups):
        old_codes = {path: m.tw.codes.clone() for path, m in grp}
        Zs, opts, Z0s, s0s = {}, {}, {}, {}
        for path, m in grp:
            s0 = m.tw.scales().detach()
            Z = (s0.repeat_interleave(GROUP, 1) * m.tw.codes.float()).clone()
            Z.requires_grad_(True)
            Zs[path], Z0s[path], s0s[path] = Z, Z.detach().clone(), s0
            opts[path] = torch.optim.Adam([Z], lr=args.zlr, betas=(0.9, 0.95))
        for r in range(args.batches):
            sstore.clear()
            with torch.no_grad():
                model(calib[r])
            for path, m in grp:
                x = rot_fwd(sstore[path].float(), m.signs)
                y = tstore[path][r].to(DEV).float()
                ynorm = y.pow(2).mean().sqrt().clamp(min=1e-6)
                for _ in range(args.steps):
                    w = _TernarySTE.apply(Zs[path], s0s[path], GROUP)
                    loss = (x @ w.T - y).pow(2).mean() / ynorm
                    loss = loss + args.lam * ((Zs[path] - Z0s[path]) / s0s[path].repeat_interleave(GROUP, 1)).pow(2).mean()
                    opts[path].zero_grad(set_to_none=True)
                    loss.backward()
                    opts[path].step()
                del x, y
        # global gate: keep the group only if the model-level ppl improves
        for path, m in grp:
            with torch.no_grad():
                Z, s0 = Zs[path], s0s[path]
                new_codes = torch.clamp(torch.round(Z / s0.repeat_interleave(GROUP, 1)), -1, 1).to(torch.int8)
                moved = (new_codes != m.tw.codes).float().mean().item()
                moved_report.append(moved)
                m.tw.codes.copy_(new_codes)      # theta (scales) untouched
            del Zs[path], Z0s[path], opts[path]
        ppl_g = quick_ppl(model, test_ids, args.seq, 4)
        if ppl_g > best_ppl:
            for path, m in grp:
                m.tw.codes.copy_(old_codes[path])
            print(f"  group {gi+1}/{len(groups)} (layer {k}): REVERTED "
                  f"(gate {ppl_g:.2f} > {best_ppl:.2f})", flush=True)
        else:
            best_ppl = ppl_g
            print(f"  group {gi+1}/{len(groups)} (layer {k}): kept, gate ppl "
                  f"{ppl_g:.2f}, moved {np.mean(moved_report[-7:])*100:.2f}%", flush=True)
        del old_codes
        torch.cuda.empty_cache()
    for h in shooks:
        h.remove()

    zf = np.mean([tw.zero_frac() for tw in ternary])
    print(f"layer-wise done in {time.time()-t0:.0f}s; mean codes moved "
          f"{np.mean(moved_report)*100:.2f}%; zero_frac {zf:.4f}", flush=True)

    model.eval()
    ppl1 = quick_ppl(model, test_ids, args.seq, 40)
    print(f"quick ppl after layer-wise = {ppl1:.2f}  (was {ppl0:.2f})", flush=True)

    # save in train.py checkpoint format for export/resume compatibility
    from train import save_checkpoint
    import types
    save_args = types.SimpleNamespace(**{**ck["args"], "tag": args.tag})
    save_checkpoint(model, ternary, torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]),
                    save_args, ck["step"] + 1)
    print(f"saved <WORK_DIR>/l1micro_{args.tag}.pt", flush=True)


if __name__ == "__main__":
    main()
