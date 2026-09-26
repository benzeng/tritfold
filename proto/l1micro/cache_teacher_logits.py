#!/usr/bin/env python
"""M3 Phase 2 prep: cache teacher top-50 logits on WikiText-2 train.

Saves vals (N,50) fp16 + ids (N,50) int32 as .npy, positions aligned so that
entry i holds the logits predicting token i+1 (window-final position dropped).

Usage: cache_teacher_logits.py [--tokens 524288] [--seq 512] [--out PREFIX]
"""
import argparse
import time

import numpy as np
import torch

MODEL = "<MODELS_DIR>/Qwen3-0.6B"
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=524288)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", default="<WORK_DIR>/teacher_top50")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_win = args.tokens // args.seq
    ids = ids[: n_win * args.seq]
    print(f"teacher cache: {n_win} windows x {args.seq} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()

    vals = np.memmap(args.out + "_vals.npy", dtype=np.float16, mode="w+",
                     shape=(n_win, args.seq - 1, args.topk))
    vids = np.memmap(args.out + "_ids.npy", dtype=np.int32, mode="w+",
                     shape=(n_win, args.seq - 1, args.topk))
    t0 = time.time()
    with torch.no_grad():
        for w in range(n_win):
            x = ids[w * args.seq:(w + 1) * args.seq].unsqueeze(0).to(DEV)
            lg = model(x).logits[0, :-1].float()          # (seq-1, V): pos i -> token i+1
            v, i = torch.topk(lg, args.topk, dim=-1)
            vals[w] = v.half().cpu().numpy()
            vids[w] = i.cpu().numpy().astype(np.int32)
            if w % 200 == 0:
                print(f"  {w}/{n_win}  {w*args.seq/(time.time()-t0):.0f} tok/s", flush=True)
    vals.flush(); vids.flush()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}_{{vals,ids}}.npy", flush=True)


if __name__ == "__main__":
    main()
