#!/usr/bin/env python
"""ARC-Challenge 0-shot baseline (FP + ternary-M4) — validates the M5' harness
before the A100 run and produces both acceptance denominators.

Identical option_logprob logic to the M5' notebook cell; the ternary path uses
the eval-only FixedRot wrapper (folded weights read directly from the exported
safetensors — no STE/Z parameters, fits 6GB).
"""
import sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/dong/oneLLM/Bonsai/proto")
from common.fwht_torch import rot_fwd, rot_inv, signs_for_width
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "/home/dong/models/Qwen3-1.7B"
EXPORT = "/home/dong/models/bonsai-proto/m4/qwen3-1.7b-ternary-hd-qat/model.safetensors"
DEV = "cuda"
Q, CH = "Question: {q}\nAnswer:", " {a}"

tok = AutoTokenizer.from_pretrained(MODEL)
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")


def option_logprob(model, prompt, choice):
    ids = tok(prompt + choice, return_tensors="pt").input_ids[0]
    pl = tok(prompt, return_tensors="pt").input_ids[0].numel()
    x = ids.unsqueeze(0).to(DEV)
    with torch.no_grad():
        lg = model(x).logits.float()[0, pl - 1: -1]
    tgt = ids[pl:].to(DEV)
    lp = torch.log_softmax(lg, -1)[torch.arange(len(tgt), device=lg.device), tgt].sum().item()
    return lp, len(tgt)


def arc_eval(model, tag, limit=None):
    model.eval()
    n = acc = accn = 0
    items = arc if limit is None else arc.select(range(min(limit, len(arc))))
    t0 = time.time()
    for ex in items:
        prompt = Q.format(q=ex["question"]) + "\n"
        scores = [option_logprob(model, prompt, CH.format(a=c)) for c in ex["choices"]["text"]]
        labels = ex["choices"]["label"]
        gold = labels.index(ex["answerKey"])
        pred = max(range(len(scores)), key=lambda i: scores[i][0])
        pred_n = max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1])
        acc += pred == gold; accn += pred_n == gold; n += 1
    print(f"[{tag}] ARC-Challenge 0-shot: acc {acc/n:.3f} | acc_norm {accn/n:.3f} "
          f"(n={n}, {time.time()-t0:.0f}s)", flush=True)
    return acc / n, accn / n


# ---- FP 基线 ----
fp = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV)
arc_eval(fp, "FP Qwen3-1.7B")
del fp; torch.cuda.empty_cache()

# ---- 三值 M4（评估版 wrapper）----
W = load_file(EXPORT)

class FixedRot(nn.Module):
    def __init__(self, w, signs):
        super().__init__()
        self.w = nn.Parameter(w.half(), requires_grad=False)
        self.signs = signs
    def forward(self, x):
        return F.linear(rot_fwd(x, self.signs).half(), self.w)

class FixedEmb(nn.Module):
    def __init__(self, w, signs):
        super().__init__()
        self.w = nn.Parameter(w.half(), requires_grad=False)
        self.signs = signs
    def forward(self, ids):
        return rot_inv(F.embedding(ids, self.w), self.signs).half()

class Head(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = emb
    def forward(self, h):
        return F.linear(rot_fwd(h, self.emb.signs).half(), self.emb.w)

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV)
def sgn(n):
    return signs_for_width(n, DEV)
s_h = sgn(W["model.embed_tokens.weight"].shape[1])
m.model.embed_tokens = FixedEmb(W["model.embed_tokens.weight"], s_h).to(DEV)
m.lm_head = Head(m.model.embed_tokens).to(DEV)
for i, layer in enumerate(m.model.layers):
    for sub, names in [("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                       ("mlp", ("gate_proj", "up_proj", "down_proj"))]:
        parent = getattr(layer, sub)
        for name in names:
            wn = f"model.layers.{i}.{sub}.{name}.weight"
            setattr(parent, name, FixedRot(W[wn], sgn(W[wn].shape[1])).to(DEV))

arc_eval(m, "三值 M4（wiki-only）")
