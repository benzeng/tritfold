#!/usr/bin/env python
"""M3 Phase 2: E2-lite ternary distillation training on Qwen3-0.6B.

P-step: Adam on group scales (lr 3e-3) + fp32 norm islands (lr 3e-4),
        betas (0.9, 0.95), constant lr.
V-step: per step, flip top-|grad| codes one level toward descent
        (global selection, trust ratio ||dW||/||W|| <= 0.01).
Loss:   top-50 KD cross-entropy against cached teacher logits.
Eval:   quick ppl on a fixed WikiText-2 test slice (non-overlap seq windows).

Usage:
  python train.py --init zerofrac --steps 300 --tag main
  python train.py --init amax --steps 40 --tag initA --no-save
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from ternary_model import collect_trainable, install_ternary, undo_flips, vstep

MODEL = "<MODELS_DIR>/Qwen3-0.6B"
DEV = "cuda"
CACHE = "<WORK_DIR>/teacher_top50"


def load_ids(split):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[split]["text"] if t.strip())
    return tok(text, return_tensors="pt").input_ids[0]


@torch.no_grad()
def quick_ppl(model, ids, seq, n_windows):
    model.eval()
    nll, cnt = 0.0, 0
    V = model.config.vocab_size
    for w in range(n_windows):
        x = ids[w * seq:(w + 1) * seq].unsqueeze(0).to(DEV)
        lg = model(x).logits.float()
        nll += F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction="sum").item()
        cnt += x[:, 1:].numel()
    model.train()
    return float(np.exp(nll / cnt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="zerofrac", choices=["amax", "zerofrac"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--vstep-frac", type=float, default=0.01)
    ap.add_argument("--trust", type=float, default=0.01)
    ap.add_argument("--no-vstep", action="store_true")
    ap.add_argument("--vstep-ema", action="store_true",
                    help="rank flips by |grad| EMA (smoothed) and run every step")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-windows", type=int, default=40)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    test_ids = load_ids("test")
    train_ids = load_ids("train")
    # cache files are raw memmap dumps (no .npy header)
    topk, row = 50, args.seq - 1
    import os
    cache = args.cache
    n_win = os.path.getsize(cache + "_vals.npy") // (row * topk * 2)
    vals = np.memmap(cache + "_vals.npy", dtype=np.float16, mode="r", shape=(n_win, row, topk))
    vids = np.memmap(cache + "_ids.npy", dtype=np.int32, mode="r", shape=(n_win, row, topk))
    print(f"cache: {n_win} windows; batch={args.batch}x{args.seq}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(DEV)
    ternary = install_ternary(model, args.init)
    scales, islands = collect_trainable(model, ternary)
    opt = torch.optim.Adam([
        {"params": scales, "lr": 1e-3},
        {"params": islands, "lr": 3e-4},
    ], betas=(0.9, 0.95))

    if args.resume:
        ck = torch.load(args.resume, map_location=DEV)
        restore_checkpoint(model, ternary, ck)
        opt.load_state_dict(ck["opt"])
        print(f"resumed from {args.resume} (step {ck['step']})", flush=True)

    zf = np.mean([tw.zero_frac() for tw in ternary])
    print(f"init={args.init} zero_frac={zf:.4f}", flush=True)
    print(f"FP ref quick ppl computing...", flush=True)

    # FP reference on the same eval slice: build a separate FP model once
    model_fp = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
    ppl_fp = quick_ppl(model_fp, test_ids, args.seq, args.eval_windows)
    del model_fp
    torch.cuda.empty_cache()
    print(f"FP ref quick ppl = {ppl_fp:.2f}", flush=True)

    model.train()
    ppl0 = quick_ppl(model, test_ids, args.seq, args.eval_windows)
    print(f"init ppl (ternary, step 0) = {ppl0:.2f}", flush=True)

    t_ids = train_ids[: n_win * args.seq].reshape(n_win, args.seq)
    step_i, t0 = 0, time.time()
    hist = []
    while step_i < args.steps:
        wb = (step_i * args.batch) % (n_win - args.batch)
        x = t_ids[wb: wb + args.batch].to(DEV)
        tv = torch.from_numpy(np.asarray(vals[wb: wb + args.batch])).to(DEV).float()
        ti = torch.from_numpy(np.asarray(vids[wb: wb + args.batch])).to(DEV)

        opt.zero_grad(set_to_none=True)
        loss_val, B = 0.0, x.size(0)
        for i in range(B):   # per-sample fwd+bwd keeps peak memory < 6GB (WSL pages host RAM)
            logits = model(x[i:i + 1]).logits[:, :-1].float()
            s_logp = F.log_softmax(logits, dim=-1)
            t_prob = F.softmax(tv[i:i + 1], dim=-1)
            loss_i = -(t_prob * torch.gather(s_logp, -1, ti[i:i + 1].long())).sum(-1).mean()
            (loss_i / B).backward()
            loss_val += loss_i.item() / B
        if not np.isfinite(loss_val):      # divergent batch: skip update entirely
            opt.zero_grad(set_to_none=True)
            step_i += 1
            continue
        torch.nn.utils.clip_grad_norm_(scales + islands, 1.0)
        opt.step()

        def kd_loss():
            """mean KD loss on the current batch (no grad)"""
            with torch.no_grad():
                tot, B = 0.0, x.size(0)
                for i in range(B):
                    logits = model(x[i:i + 1]).logits[:, :-1].float()
                    s_logp = F.log_softmax(logits, dim=-1)
                    t_prob = F.softmax(tv[i:i + 1], dim=-1)
                    tot += -(t_prob * torch.gather(s_logp, -1, ti[i:i + 1].long())).sum(-1).mean().item()
                return tot / B

        moved, rejected = 0, 0
        if args.vstep_ema:
            for tw in ternary:
                tw.update_ema(beta=0.9)
            if not args.no_vstep:
                moved, undo = vstep(ternary, args.vstep_frac, args.trust, use_ema=True)
                if kd_loss() > loss_val:     # flips hurt on this batch: revert
                    undo_flips(undo)
                    rejected = moved
                    moved = 0
        elif not args.no_vstep and step_i % 2 == 1:
            moved, _ = vstep(ternary, args.vstep_frac, args.trust)

        step_i += 1
        if step_i % 10 == 0:
            zf = np.mean([tw.zero_frac() for tw in ternary])
            print(f"step {step_i:4d} loss {loss_val:.4f} flips {moved} rej {rejected} zero {zf:.4f} "
                  f"({(step_i * args.batch * args.seq) / (time.time() - t0):.0f} tok/s)", flush=True)
        if step_i % args.eval_every == 0 or step_i == args.steps:
            ppl = quick_ppl(model, test_ids, args.seq, args.eval_windows)
            hist.append((step_i, ppl))
            print(f"  >> step {step_i} quick ppl = {ppl:.2f}  (FP ref {ppl_fp:.2f})", flush=True)

    if not args.no_save:
        path = f"<WORK_DIR>/l1micro_{args.tag}.pt"
        save_checkpoint(model, ternary, opt, args, step_i)
        print(f"saved {path}", flush=True)
    print("history:", json.dumps(hist), flush=True)


def save_checkpoint(model, ternary, opt, args, step):
    path = f"<WORK_DIR>/l1micro_{args.tag}.pt"
    torch.save({
        "step": step, "args": vars(args),
        "codes": [tw.codes.cpu() for tw in ternary],
        "theta": [tw.theta.detach().cpu() for tw in ternary],
        "islands": {n: m.weight.detach().cpu() for n, m in model.named_modules()
                    if hasattr(m, "weight") and m.__class__.__name__ == "Fp32RMSNorm"},
        "opt": opt.state_dict(),
    }, path)


def restore_checkpoint(model, ternary, ck):
    for tw, c, th in zip(ternary, ck["codes"], ck["theta"]):
        tw.codes.copy_(c.to(tw.codes.device))
        tw.theta.data.copy_(th.to(tw.theta.device))
    for n, m in model.named_modules():
        if m.__class__.__name__ == "Fp32RMSNorm" and n in ck["islands"]:
            m.weight.data.copy_(ck["islands"][n].to(m.weight.device))


if __name__ == "__main__":
    main()
