#!/usr/bin/env python
"""L0 mechanism validation (M2): rotation makes RTN ternary survivable.

Three-way WikiText-2 ppl comparison on Qwen3-0.6B:
  A) FP baseline
  B) naive RTN ternary (g128 amax, original basis)
  C) rotated RTN ternary (Bonsai contract: fold W'=WR^-1, online FWHT on
     activations, embedding folded + inverse-after-lookup; tied lm_head shares
     the folded table)
Plus an identity check: C without quantization must reproduce A's logits.

Usage: run_rotation_test.py [--max-windows 150] [--ctx 1024] [--stride 512]
"""
import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common.fwht_torch import BLOCK, rot_fwd, rot_inv, rtn_ternary, signs_for_width

MODEL = "<MODELS_DIR>/Qwen3-0.6B"
DEV = "cuda"


class RotLinear(nn.Module):
    """y = (s*T-folded W') @ rot_fwd(x)."""

    def __init__(self, lin: nn.Linear, signs: torch.Tensor, quant: bool):
        super().__init__()
        with torch.no_grad():
            wf = rot_fwd(lin.weight.data, signs)
            if quant:
                wf = rtn_ternary(wf)
        self.weight = nn.Parameter(wf.to(torch.float16), requires_grad=False)
        self.bias = lin.bias
        self.signs = signs

    def forward(self, x):
        z = rot_fwd(x, self.signs).to(torch.float16)
        return F.linear(z, self.weight, self.bias)


class RotEmbedding(nn.Module):
    def __init__(self, table: torch.Tensor, signs: torch.Tensor):
        super().__init__()
        self.table = nn.Parameter(table, requires_grad=False)
        self.signs = signs

    def forward(self, ids):
        e = F.embedding(ids, self.table)
        return rot_inv(e, self.signs).to(torch.float16)


class RotLMHead(nn.Module):
    """Shares the folded embedding table (tied), as the contract requires."""

    def __init__(self, table: torch.Tensor, signs: torch.Tensor):
        super().__init__()
        self.table = table
        self.signs = signs

    def forward(self, h):
        z = rot_fwd(h, self.signs).to(torch.float16)
        return F.linear(z, self.table)


def target_linears(model):
    for layer in model.model.layers:
        for parent, name in [
            (layer.self_attn, "q_proj"), (layer.self_attn, "k_proj"),
            (layer.self_attn, "v_proj"), (layer.self_attn, "o_proj"),
            (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"), (layer.mlp, "down_proj"),
        ]:
            yield parent, name


def quantize_naive(model):
    """Config B: in-place RTN ternary on the same tensor set, original basis."""
    with torch.no_grad():
        for parent, name in target_linears(model):
            lin = getattr(parent, name)
            lin.weight.data = rtn_ternary(lin.weight.data).to(torch.float16)
        emb = model.model.embed_tokens
        emb.weight.data = rtn_ternary(emb.weight.data).to(torch.float16)
        # tied lm_head shares the tensor -> follows automatically


def install_rotated(model, quant: bool):
    """Config C: contract-style folded weights + online activation rotation."""
    dev = next(model.parameters()).device
    emb = model.model.embed_tokens
    width = emb.weight.shape[1]
    s_hidden = signs_for_width(width, dev)
    with torch.no_grad():
        ef = rot_fwd(emb.weight.data, s_hidden)
        if quant:
            ef = rtn_ternary(ef)
        table = ef.to(torch.float16)
    model.model.embed_tokens = RotEmbedding(table, s_hidden)
    model.lm_head = RotLMHead(table, s_hidden)  # tied: same folded tensor
    for parent, name in target_linears(model):
        lin = getattr(parent, name)
        s = signs_for_width(lin.weight.shape[1], dev)
        setattr(parent, name, RotLinear(lin, s, quant))
    return model


@torch.no_grad()
def eval_ppl(model, ids, ctx, stride, max_windows):
    nll_sum, tok_cnt = 0.0, 0
    V = model.config.vocab_size
    for i, begin in enumerate(range(0, ids.size(1) - 1, stride)):
        if i >= max_windows:
            break
        end = min(begin + ctx, ids.size(1))
        inp = ids[:, begin:end].to(DEV)
        logits = model(inp).logits.float()
        lg, lb = logits[:, :-1], inp[:, 1:]
        if begin > 0:
            keep = min(stride, lb.size(1))
            lg, lb = lg[:, -keep:], lb[:, -keep:]
        nll_sum += F.cross_entropy(lg.reshape(-1, V), lb.reshape(-1), reduction="sum").item()
        tok_cnt += lb.numel()
    return torch.exp(torch.tensor(nll_sum / tok_cnt)).item(), tok_cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=150)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids
    print(f"tokens: {ids.size(1)}, scoring <= {args.max_windows} windows "
          f"(ctx={args.ctx}, stride={args.stride})")

    def load():
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
        return m

    results = {}

    t0 = time.time()
    m = load()
    ppl_fp, ntok = eval_ppl(m, ids, args.ctx, args.stride, args.max_windows)
    results["A_fp"] = ppl_fp
    print(f"[A] FP baseline        ppl={ppl_fp:9.2f}  ({ntok} tokens, {time.time()-t0:.0f}s)")

    # identity check: rotated without quantization must reproduce FP
    m_rot = load()
    install_rotated(m_rot, quant=False)
    x = ids[:, :256].to(DEV)
    a, b = m(x).logits.float(), m_rot(x).logits.float()
    diff = (a - b).abs()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    rel = (diff.mean() / a.abs().mean()).item()
    print(f"    identity check: max|d|={diff.max().item():.4f} mean|d|={diff.mean().item():.5f} "
          f"rel={rel:.5f} argmax_agree={agree:.4f}")
    del m_rot
    torch.cuda.empty_cache()

    t0 = time.time()
    m_naive = load()
    quantize_naive(m_naive)
    ppl_naive, _ = eval_ppl(m_naive, ids, args.ctx, args.stride, args.max_windows)
    results["B_naive_rtn"] = ppl_naive
    print(f"[B] naive RTN ternary  ppl={ppl_naive:9.2f}  ({time.time()-t0:.0f}s)")
    del m_naive
    torch.cuda.empty_cache()

    t0 = time.time()
    m_rot = load()
    install_rotated(m_rot, quant=True)
    ppl_rot, _ = eval_ppl(m_rot, ids, args.ctx, args.stride, args.max_windows)
    results["C_rotated_rtn"] = ppl_rot
    print(f"[C] rotated RTN ternary ppl={ppl_rot:9.2f}  ({time.time()-t0:.0f}s)")

    print("\nsummary:", {k: round(v, 2) for k, v in results.items()})


if __name__ == "__main__":
    main()
