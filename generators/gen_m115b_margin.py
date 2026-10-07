#!/usr/bin/env python
"""Build tritfold-m115b-margin.ipynb (M11.5b: margin analysis — the teacher question).

M11.5 found the official ternary 27B EXCEEDS its FP reference on sciq (0.743 vs
0.619). Two candidate mechanisms: (a) sharpening/format-unlock of an anomalously
low base probe score (self-distillation sufficient), or (b) an external knowledge
source (bigger teacher). Discriminating measurement on per-item margins:

  - among items the base gets WRONG, split by error confidence (|margin| bins):
    how often does the ternary flip to correct?
      wins concentrated on near-ties  -> sharpening/unlock (self-distill OK)
      wins on confident errors        -> external knowledge (big teacher)

Slim rebuild of the M11.5 session (scores died with /content): fork CUDA build
+ scorer + both downloads + sciq-only probing + margin/confusion analysis.
Everything saved to Drive this time. ~1.2h, ~8 units.
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
def md(src): return {"cell_type": MD, "metadata": {}, "source": src}
def code(src): return {"cell_type": CODE, "metadata": {}, "execution_count": None, "outputs": [], "source": src}

SCORER_SRC = Path("/home/dong/oneLLM/Bonsai/proto/common/probe_scorer.cpp").read_text()

cells = []
cells.append(md("""# M11.5b：余量分析——官方教师问题终判

**问题**：官方三值 sciq 反超基座（0.743 vs 0.619）的机制是锐化/解锁（自蒸馏可解释）还是基座之外的知识源（更大教师）？

**判据**：基座做错的题按错误置信度分桶，看三值翻正率——
- 翻正集中在**平局题**（|margin| 小）→ 锐化/格式解锁 → 自蒸馏
- 翻正出现在**自信错**的题 → 基座之外的知识 → 大教师实锤

**流程**：重建 fork CUDA + 打分器 + 双模型下载 → 仅 sciq 探针（845×4）→ 余量/混淆分析 → **全部落 Drive**。
**运行**：~1.2h ≈ 8 单位。"""))

cells.append(code("""%pip -q install cmake huggingface_hub
!test -d /content/llama.cpp || git clone -q --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp.git /content/llama.cpp
print("fork ready")"""))

cells.append(code("""# CUDA 构建（~25 min；目标：server/cli/perplexity + libllama 供打分器链接）
import subprocess, os
CUDA_ARCH = os.popen("nvidia-smi --query-gpu=compute_cap --format=csv,noheader").read().strip().replace(".", "")
print("CUDA arch:", CUDA_ARCH)
r = subprocess.run("cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON "
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
import os, glob
so = glob.glob("/content/build-cuda/**/libllama.so*", recursive=True)
assert so, "libllama.so 不存在——确认上一格 cmake 带 -DBUILD_SHARED_LIBS=ON 且构建成功"
libdirs = sorted(set(os.path.dirname(x) for x in
                     glob.glob("/content/build-cuda/**/lib*.so*", recursive=True)))
L = " ".join("-L" + d for d in libdirs)
R = " ".join("-Wl,-rpath," + d for d in libdirs)
cmd = ("g++ -O2 -o /content/probe_scorer /content/probe_scorer.cpp "
       "-I/content/llama.cpp/include -I/content/llama.cpp/ggml/include "
       + L + " -lllama -lggml -lggml-base -lggml-cuda " + R)
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr[-2000:] if r.stderr else "")
assert os.path.exists("/content/probe_scorer"), "scorer build failed"
print("probe_scorer ready | libdirs:", libdirs)'''))

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

cells.append(code(r'''# 构建探针 TSV（仅 sciq 去污染 845 题；\n → \\n 转义）
import os
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
print(f"sciq clean {len(sciq_val)}/1000")

def esc(s): return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
with open("/content/pairs.tsv", "w") as f:
    for ex in sciq_val:
        prompt = f"Question: {ex['question']}\nAnswer:" + "\n"
        for o in [ex["correct_answer"]] + [ex[f"distractor{i}"] for i in (1, 2, 3)]:
            f.write(esc(prompt) + "\t" + esc(" " + o) + "\n")
print("pairs written (sciq only)")'''))

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

cells.append(code(r'''# 余量/混淆分析：教师问题终判
import os, json

meta = [(qi, oi) for qi in range(845) for oi in range(4)]

def load(name):
    vals = []
    for line in open(f"/content/scores_{name}.tsv"):
        a, b = line.strip().split("\t")
        vals.append((float(a), int(b)))
    assert len(vals) == len(meta), (name, len(vals))
    return vals

res = {}
for name in ("ternary-27B", "fp-q8-27B"):
    by = {}
    for (qi, oi), v in zip(meta, load(name)):
        by.setdefault(qi, {})[oi] = v
    res[name] = by
base, tern = res["fp-q8-27B"], res["ternary-27B"]
print(f"items: {len(base)}")

def margin(by, qi):
    # 归一化（per-token）margin：gold - 最强干扰项；>0 对，<0 错，绝对值=置信度
    o = by[qi]
    norm = {i: o[i][0] / max(o[i][1], 1) for i in o}
    return norm[0] - max(norm[i] for i in norm if i != 0)

def accs(by):
    a = sum(1 for qi in by if max(by[qi], key=lambda i: by[qi][i][0]) == 0)
    n = sum(1 for qi in by if max(by[qi], key=lambda i: by[qi][i][0] / max(by[qi][i][1], 1)) == 0)
    return a / len(by), n / len(by)

for name in ("fp-q8-27B", "ternary-27B"):
    a, n = accs(res[name])
    print(f"[{name}] acc {a:.3f} | acc_norm {n:.3f}")

# ---- 混淆矩阵（acc_norm 口径） ----
cc = {"双对": 0, "基座对三值错": 0, "基座错三值对": 0, "双错": 0}
for qi in base:
    b = margin(base, qi) > 0
    t = margin(tern, qi) > 0
    cc["双对" if b and t else "基座对三值错" if b else "基座错三值对" if t else "双错"] += 1
print("\n混淆矩阵（acc_norm 口径）:", cc)

# ---- 核心判据：基座错题按置信度分桶，看三值翻正率 ----
print("\n基座做错题的 |margin| 分桶 → 三值翻正率（判据表）：")
print("| 基座错误置信度 | 题数 | 三值翻正 | 翻正率 |")
print("|---|---|---|---|")
BINS = [(0.0, 0.3, "近平局 <0.3"), (0.3, 1.0, "0.3-1.0"), (1.0, 2.5, "1.0-2.5"), (2.5, 1e9, "自信错 ≥2.5")]
flip = {}
for lo, hi, tag in BINS:
    items = [qi for qi in base if margin(base, qi) < 0 and lo <= -margin(base, qi) < hi]
    f = sum(1 for qi in items if margin(tern, qi) > 0)
    flip[tag] = (len(items), f)
    print(f"| {tag} | {len(items)} | {f} | {f/max(len(items),1):.1%} |")

# ---- 反向检查：基座对、三值错的题里，三值的错误置信度 ----
rev = [(-margin(tern, qi)) for qi in base if margin(base, qi) > 0 and margin(tern, qi) < 0]
rev.sort()
if rev:
    import statistics
    print(f"\n反向（基座对→三值错）n={len(rev)}：中位错误置信度 {statistics.median(rev):.2f}，"
          f">=2.5 的 {sum(1 for x in rev if x >= 2.5)} 题（三值自信地否决基座——外部知识信号）")

# ---- 三值自身的整体 sharpening 度量：选项间 margin 的绝对幅度 ----
import statistics
bm = [abs(margin(base, qi)) for qi in base]
tm = [abs(margin(tern, qi)) for qi in base]
print(f"\n|margin| 中位数：基座 {statistics.median(bm):.2f} vs 三值 {statistics.median(tm):.2f}"
      f"（{statistics.median(tm)/statistics.median(bm):.2f}× —— 锐化系数）")

# ---- 判读输出 ----
near = flip.get("近平局 <0.3", (0, 0))
conf = flip.get("自信错 ≥2.5", (0, 0))
verdict = []
if near[0] and near[1] / near[0] >= 0.5: verdict.append("平局题高翻正（锐化/解锁信号）")
if conf[0] and conf[1] / conf[0] >= 0.25: verdict.append("自信错高翻正（外部知识信号——大教师）")
rev_conf = sum(1 for x in rev if x >= 2.5) if rev else 0
if rev_conf >= 10: verdict.append(f"三值自信否决基座 {rev_conf} 题（外部知识信号）")
print("\n=== 判读 ===")
print("; ".join(verdict) if verdict else "混合/不显著——按比例细读上表")

# ---- Drive 落盘（教训：产物必须持久化） ----
out = {"confusion": cc, "flip_by_bin": {k: list(v) for k, v in flip.items()},
       "accs": {k: accs(res[k]) for k in res}, "verdict": verdict}
try:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    os.makedirs("/content/drive/MyDrive/m115b", exist_ok=True)
    import shutil
    for n in ("ternary-27B", "fp-q8-27B"):
        shutil.copy(f"/content/scores_{n}.tsv", f"/content/drive/MyDrive/m115b/scores_{n}.tsv")
    json.dump(out, open("/content/drive/MyDrive/m115b/analysis.json", "w"), ensure_ascii=False, indent=1)
    print("\n已落盘 Drive/m115b/（分数 tsv ×2 + analysis.json）")
except Exception as e:
    json.dump(out, open("/content/analysis.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nDrive 不可用（{type(e).__name__}），analysis.json 已存 /content——请手动下载两个 scores tsv！")
'''))



out = Path("notebooks/tritfold-m115b-margin.ipynb")
nb = {
    "cells": cells,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 0,
}
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out}")
