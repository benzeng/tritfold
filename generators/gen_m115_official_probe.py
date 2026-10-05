#!/usr/bin/env python
"""Build tritfold-m115-official-probe.ipynb (M11.5: probe the OFFICIAL Bonsai 27B artifact).

Question: is the official ternary 27B weaker than its FP base? Our transfer-efficiency
framework (findings-12/13) predicts: ppl near-parity, discrimination at some fraction
of the FP reference, free recall weakest. This measures it directly — the official
artifact has never been capability-probed by us.

Design: paired protocol, one C++ scorer (probe_scorer.cpp, validated locally):
  - ternary side:  Ternary-Bonsai-2-27B-PQ2_0.gguf (7.2GB, official artifact,
    transported via private HF repo benzeng/tmp-ternary-bonsai-27b)
  - FP reference:  unsloth/Qwen3.8-27B-Q8_0.gguf (29GB, quant loss <1%)
  - probes: decontaminated sciq val (845x4) + ARC-c test (1172x4), exact same
    closed-book format as all our 1.7B experiments ("Question: {q}\\nAnswer:" + " {a}")
  - scorer: single batch decode per pair, log_softmax sum over answer tokens
    (same algorithm as the PyTorch probe); logits via llama_get_logits_ith(token pos)
  - plus: 3-question spot checks (chat + raw, mandatory recipe) and a short
    ppl cross-check (wt2_30k, c512) for the whitepaper 1.02x claim

Expected runtime: build ~25 min + downloads ~35 min + probes ~40 min + extras ~25 min
=> ~2.5h, ~13 units. Readouts transfer directly into the M12 calibration table.
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
def md(src): return {"cell_type": MD, "metadata": {}, "source": src}
def code(src): return {"cell_type": CODE, "metadata": {}, "execution_count": None, "outputs": [], "source": src}

SCORER_SRC = Path("/home/dong/oneLLM/Bonsai/proto/common/probe_scorer.cpp").read_text()

cells = []
cells.append(md("""# M11.5：官方 Bonsai 27B artifact 能力探针

**问题**：官方三值 27B 比原版 FP 弱多少？我们的转移效率框架预测：ppl 近平、判别打折扣、自由回忆最弱。本实验直接实测（官方 artifact 首次能力探针）。

**配对协议**：同一个 C++ 打分器 + 同一份 TSV 跑双模型——
- 三值侧：`Ternary-Bonsai-2-27B-PQ2_0.gguf`（官方 artifact，7.2GB，私有 HF 仓中转）
- FP 参照：`unsloth/Qwen3.8-27B-Q8_0.gguf`（29GB，量化损失 <1%）
- 探针：去污染 sciq（845×4）+ ARC-c（1172×4），闭卷格式与我们全部 1.7B 实验一致

**运行**：A100 ~2.5h（构建 25min + 下载 35min + 探针 40min + ppl/抽检 25min）。"""))

cells.append(code("""%pip -q install cmake huggingface_hub
!test -d /content/llama.cpp || git clone -q --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp.git /content/llama.cpp
print("fork ready")"""))

cells.append(code("""# CUDA 构建（~25 min；目标：server/cli/perplexity + libllama 供打分器链接）
import subprocess, os
CUDA_ARCH = os.popen("nvidia-smi --query-gpu=compute_cap --format=csv,noheader").read().strip().replace(".", "")
print("CUDA arch:", CUDA_ARCH)
r = subprocess.run("cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "
                   f"-DGGML_CUDA_ARCH={CUDA_ARCH} /content/llama.cpp > /tmp/cmake.log 2>&1", shell=True)
if r.returncode: print(open("/tmp/cmake.log").read()[-2000:])
r = subprocess.run("cmake --build build-cuda --target llama-server llama-cli llama-perplexity -j$(nproc) "
                   "> /tmp/build.log 2>&1", shell=True)
if r.returncode:
    print(open("/tmp/build.log").read()[-3000:])
else:
    print(open("/tmp/build.log").read()[-200:])"""))

cells.append(code(rf'''# 写入并编译打分器（链接 build-cuda 的 libllama；本机已对 v07 冒烟验证）
SCORER = r"""
{SCORER_SRC}
"""
open("/content/probe_scorer.cpp", "w").write(SCORER)
import subprocess
r = subprocess.run(
    "g++ -O2 -o /content/probe_scorer /content/probe_scorer.cpp "
    "-I/content/llama.cpp/include -I/content/llama.cpp/ggml/include "
    "-L/content/build-cuda/src -L/content/build-cuda/ggml/src "
    "-lllama -lggml -lggml-base -lggml-cuda "
    "-Wl,-rpath,/content/build-cuda/src -Wl,-rpath,/content/build-cuda/ggml/src",
    shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr[-2000:] if r.stderr else "")
import os
assert os.path.exists("/content/probe_scorer"), "scorer build failed"
print("probe_scorer ready")'''))

cells.append(code(r'''# 下载双模型（三值 7.2GB 私有仓 + Q8 29GB）
import os
from huggingface_hub import hf_hub_download
try:
    from google.colab import userdata
    TOK = userdata.get("HF_TOKEN")
except Exception:
    import getpass; TOK = getpass.getpass("HF token: ")

TERN = hf_hub_download("benzeng/tmp-ternary-bonsai-27b", "Ternary-Bonsai-2-27B-PQ2_0.gguf",
                       repo_type="model", token=TOK)
print("ternary:", TERN)
FPQ8 = hf_hub_download("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q8_0.gguf",
                       repo_type="model", token=TOK)
print("fp-q8:", FPQ8)'''))

cells.append(code(r'''# 构建探针 TSV（与我们 1.7B 实验完全同格式；\n → \\n 转义）
import os
os.environ["HF_HUB_OFFLINE"] = "0"
from datasets import load_dataset

def ngrams(text, n=13):
    ws = text.lower().split()
    return {" ".join(ws[i:i+n]) for i in range(len(ws)-n+1)} if len(ws)>=n else set()

_train_ng = set()
for ex in load_dataset("allenai/sciq", split="train"):
    _train_ng |= ngrams(ex["support"] + " " + ex["question"])
sciq_val_all = load_dataset("allenai/sciq", split="validation")
CLEAN_IDX = [i for i, ex in enumerate(sciq_val_all)
             if not (ngrams(ex["support"] + " " + ex["question"]) & _train_ng)]
sciq_val = sciq_val_all.select(CLEAN_IDX)
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
print(f"sciq clean {len(sciq_val)}/1000 | ARC {len(arc)}")

def esc(s): return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
pairs = []   # (probe_tag, ex_idx, opt_idx, prompt, option)
for qi, ex in enumerate(sciq_val):
    prompt = f"Question: {ex['question']}\nAnswer:" + "\n"
    opts = [ex["correct_answer"]] + [ex[f"distractor{i}"] for i in (1, 2, 3)]
    for oi, o in enumerate(opts):
        pairs.append(("sciq", qi, oi, prompt, " " + o))
for qi, ex in enumerate(arc):
    prompt = f"Question: {ex['question']}\nAnswer:" + "\n"
    for oi, o in enumerate(ex["choices"]["text"]):
        pairs.append(("arc", qi, oi, prompt, " " + o))
with open("/content/pairs.tsv", "w") as f:
    for tag, qi, oi, p, o in pairs:
        f.write(f"{esc(p)}\t{esc(o)}\n")
with open("/content/pairs.meta", "w") as f:
    for tag, qi, oi, p, o in pairs:
        f.write(f"{tag}\t{qi}\t{oi}\n")
print(f"pairs: {len(pairs)}")'''))

cells.append(code(r'''# 双模型跑探针（GPU 全卸载；每对一次 decode）
import subprocess, time, glob
TERN = glob.glob("/root/.cache/huggingface/hub/models--benzeng--tmp-ternary-bonsai-27b/snapshots/*/*.gguf")[0]
FPQ8 = glob.glob("/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/Qwen3.8-27B-Q8_0.gguf")[0]

for name, path in (("ternary-27B", TERN), ("fp-q8-27B", FPQ8)):
    t0 = time.time()
    r = subprocess.run(["/content/probe_scorer", path, "/content/pairs.tsv",
                        f"/content/scores_{name}.tsv", "99"],
                       capture_output=True, text=True)
    print(f"[{name}] {time.time()-t0:.0f}s {r.stderr.strip().splitlines()[-1]}")'''))

cells.append(code(r'''# 判卷：配对计算 acc / acc_norm + 框架对照表
import math, os
meta = [l.split("\t") for l in open("/content/pairs.meta")]
meta = [(t, int(q), int(o)) for t, q, o in meta]

def read_scores(name):
    vals = []
    for line in open(f"/content/scores_{name}.tsv"):
        a, b = line.strip().split("\t")
        vals.append((float(a), int(b)))
    return vals

results = {}
for name in ("ternary-27B", "fp-q8-27B"):
    sc = read_scores(name)
    by = {}
    for (tag, qi, oi), (lp, n) in zip(meta, sc):
        by.setdefault((tag, qi), {})[oi] = (lp, n)
    for tag in ("sciq", "arc"):
        gold = 0 if tag == "sciq" else None
        acc = accn = n = 0
        for (t, qi), opts in by.items():
            if t != tag: continue
            if tag == "arc":
                # gold = 正确选项序（choices 顺序里 answerKey 的位置）——由 pairs 构建可知 sciq gold=0；
                # ARC 的 gold 需要重放 answerKey
                continue
            best_a = max(opts, key=lambda i: opts[i][0])
            best_n = max(opts, key=lambda i: opts[i][0]/max(opts[i][1],1))
            acc += best_a == 0; accn += best_n == 0; n += 1
        if tag == "sciq":
            results[(name, "sciq")] = (acc/n, accn/n, n)
            print(f"[{name} sciq] acc {acc/n:.3f} | acc_norm {accn/n:.3f} (n={n})")

# ARC gold 重放（answerKey → 选项序）
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
gold_arc = {i: ex["choices"]["label"].index(ex["answerKey"]) for i, ex in enumerate(arc)}
import os
for name in ("ternary-27B", "fp-q8-27B"):
    sc = read_scores(name)
    by = {}
    for (tag, qi, oi), (lp, n) in zip(meta, sc):
        if tag == "arc":
            by.setdefault(qi, {})[oi] = (lp, n)
    acc = accn = n = 0
    for qi, opts in by.items():
        g = gold_arc[qi]
        best_a = max(opts, key=lambda i: opts[i][0])
        best_n = max(opts, key=lambda i: opts[i][0]/max(opts[i][1],1))
        acc += best_a == g; accn += best_n == g; n += 1
    results[(name, "arc")] = (acc/n, accn/n, n)
    print(f"[{name} ARC] acc {acc/n:.3f} | acc_norm {accn/n:.3f} (n={n})")

print()
print("=== 框架对照（acc_norm）===")
rows = [
    ("官方 27B 三值", results.get(("ternary-27B", "sciq"), (float('nan'),)*3)[1], results.get(("ternary-27B", "arc"), (float('nan'),)*3)[1]),
    ("27B Q8 参照",   results.get(("fp-q8-27B", "sciq"), (float('nan'),)*3)[1], results.get(("fp-q8-27B", "arc"), (float('nan'),)*3)[1]),
    ("我方 v0.7 1.7B", 0.530, 0.319),
    ("1.7B FP", 0.699, 0.378),
    ("8B 教师(PyTorch)", 0.830, 0.472),
]
print("| 模型 | sciq | ARC |")
print("|---|---|---|")
for r in rows: print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} |")
t = results.get(("ternary-27B", "sciq"), (float('nan'),)*3)[1]
f = results.get(("fp-q8-27B", "sciq"), (float('nan'),)*3)[1]
if f and f == f:
    print(f"\n官方三值/FP 参照（sciq acc_norm）= {t/f:.1%}；对照我方 1.7B 框架：v0.7/8B教师 = {0.530/0.830:.1%}")'''))

cells.append(code(r'''# 抽检：三题双模型（chat 模板 + 裸续写，强制配方）
import glob, subprocess, time
BIN = "/content/build-cuda/bin/llama-cli"
TERN = glob.glob("/root/.cache/huggingface/hub/models--benzeng--tmp-ternary-bonsai-27b/snapshots/*/*.gguf")[0]
FPQ8 = glob.glob("/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/Qwen3.8-27B-Q8_0.gguf")[0]
QS = ["What planet is known as the Red Planet?",
      "Why do we see lightning before we hear thunder?",
      "What is the powerhouse of the cell?"]
for name, path in (("ternary-27B", TERN), ("fp-q8-27B", FPQ8)):
    for q in QS:
        r = subprocess.run([BIN, "-m", path, "-p", q, "-n", "48", "-t", "8", "-st", "--simple-io",
                            "--temp", "0.5", "--top-p", "0.85", "--top-k", "20",
                            "--repeat-penalty", "1.1", "-ngl", "99"],
                           capture_output=True, text=True, timeout=300)
        txt = r.stdout.split(q)[-1].strip()[:200]
        print(f"[{name}] Q: {q}\nA(raw): {txt}\n", flush=True)'''))

cells.append(code(r'''# ppl 交叉验证（wt2_30k c512，官方白皮书宣称 1.02×FP）
import glob, subprocess
TERN = glob.glob("/root/.cache/huggingface/hub/models--benzeng--tmp-ternary-bonsai-27b/snapshots/*/*.gguf")[0]
FPQ8 = glob.glob("/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/Qwen3.8-27B-Q8_0.gguf")[0]
from datasets import load_dataset
text2 = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"] if t.strip())
open("/content/wt2_30k.txt", "w").write(text2[:120000])
for name, path in (("ternary-27B", TERN), ("fp-q8-27B", FPQ8)):
    r = subprocess.run(["/content/build-cuda/bin/llama-perplexity", "-m", path,
                        "-f", "/content/wt2_30k.txt", "-c", "512", "-t", "8", "-ngl", "99"],
                       capture_output=True, text=True, timeout=1800)
    for line in r.stderr.splitlines():
        if "Final estimate" in line or "PPL =" in line:
            print(f"[{name}] {line.strip()}")'''))

out = Path("notebooks/tritfold-m115-official-probe.ipynb")
nb = {
    "cells": cells,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 0,
}
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out}")
