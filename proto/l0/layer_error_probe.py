#!/usr/bin/env python
"""L0 addendum: per-layer quantization error probe (isolates the rotation
mechanism from full-model effects).

Captures real inputs to representative linears of Qwen3-0.6B and measures
relative output error ||Wx - Wq x||/||Wx|| for naive vs rotated RTN ternary.
"""
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common.fwht_torch import rot_fwd, rtn_ternary, signs_for_width

MODEL = "<MODELS_DIR>/Qwen3-0.6B"
DEV = "cuda"

PROBE = [
    ("blk0_attn_q", "model.layers.0.self_attn.q_proj"),
    ("blk0_mlp_gate", "model.layers.0.mlp.gate_proj"),
    ("blk0_mlp_down", "model.layers.0.mlp.down_proj"),
    ("blk14_mlp_down", "model.layers.14.mlp.down_proj"),
    ("blk27_mlp_down", "model.layers.27.mlp.down_proj"),
    ("blk27_attn_o", "model.layers.27.self_attn.o_proj"),
]


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"][:200])
    ids = tok(text, return_tensors="pt").input_ids[:, :2048].to(DEV)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()

    # capture inputs
    store = {}
    mods = dict(model.named_modules())
    handles = []
    for tag, path in PROBE:
        m = mods[path]
        handles.append(m.register_forward_pre_hook(
            lambda mod, inp, tag=tag: store.__setitem__(tag, inp[0].detach().float())))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    print(f"{'layer':18s} {'in':>5s} {'act kurt':>9s} {'err naive':>10s} {'err rot':>10s} "
          f"{'ratio':>7s} {'zero% naive':>11s} {'zero% rot':>9s}")
    for tag, path in PROBE:
        lin = mods[path]
        w = lin.weight.data.float()
        x = store[tag].reshape(-1, store[tag].shape[-1])  # (tokens, in)
        kurt = ((x - x.mean()) ** 4).mean() / max(((x - x.mean()) ** 2).mean() ** 2, 1e-30)

        y = x @ w.T
        yn = x @ rtn_ternary(w).T

        s = signs_for_width(w.shape[1], DEV)
        wf = rot_fwd(w, s)
        wfq = rtn_ternary(wf)
        yr = rot_fwd(x, s) @ wfq.T

        rel = lambda e: (e - y).norm().item() / y.norm().item()
        zn = (rtn_ternary(w) == 0).float().mean().item()
        zr = (wfq == 0).float().mean().item()
        en, er = rel(yn), rel(yr)
        print(f"{tag:18s} {w.shape[1]:5d} {kurt.item():9.1f} {en:10.4f} {er:10.4f} "
              f"{en/max(er,1e-9):7.2f} {zn*100:10.1f}% {zr*100:8.1f}%")


if __name__ == "__main__":
    main()
