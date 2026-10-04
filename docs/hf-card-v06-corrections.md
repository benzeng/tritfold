# v0.6 HF 模型卡勘误（2026-10-05，1a 对照实验后）

应用于 `benzeng/tritfold-1.7b-feature-ptq1_0` 的 Model card（web UI → Edit model card）。
实测数字全部不变；只改归因。共三处 + 新增一节。

## ① 标语段（替换第一段）

原文：

> The first open ternary model with both knowledge ranking AND generation capability — feature-level distillation (hidden-state cosine matching) transfers discrimination ability that output-level KL cannot.

改为：

> The first open ternary model with both knowledge ranking AND generation capability. A post-release control experiment split the attribution: targeted in-distribution corpus through teacher KL contributes most of the knowledge gain (+0.060 sciq), hidden-state cosine matching adds a further +0.027 acc_norm / +0.065 raw acc — see the control-experiment note below.

## ② "The knowing vs saying discovery" 一节（替换四条 bullet）

原文第二条：

> **Feature-level cosine** → transfers discrimination ability (what to know) → NOT bounded by teacher quality

改为（整节 bullet 替换为）：

> - Output-level KL → transfers distribution shape (how to speak) → bounded by teacher quality
> - **Targeted in-distribution corpus through teacher KL** → transfers discrimination ability (+0.060 sciq over generic-corpus mixes); generic educational corpus transfers none (+0.004)
> - **Feature-level cosine (OFF)** → modest increment on top: +0.027 acc_norm (within noise) / +0.065 raw accuracy (2.7σ) — sharpens argmax selection
> - Generation fluency → requires instruction data (ultrachat) in the mix
> - Factual recall → still capacity-bound (1.58 bits can't store all facts)

## ③ 新增一节（插在 "knowing vs saying" 之后、"Performance" 之前）

> ## Post-release control experiment (erratum)
>
> The original card attributed the sciq jump (0.398 → 0.485) to feature-level cosine matching. A control arm (KL-only; identical data mix, steps, teacher and starting checkpoint) corrects the attribution:
>
> | sciq | acc | acc_norm |
> |---|---|---|
> | v0.3 baseline (KL, generic corpus) | 0.376* | 0.398 |
> | 1a control (KL + full sciq stream) | 0.471 | **0.458** |
> | 1b = this model (+ cosine) | **0.536** | **0.485** |
>
> *v0.3 measured on the full n=1000 validation; 1a/1b on the decontaminated n=845 subset — the harder set, so +0.060 is conservative.
>
> Most of the gain came from a data-pipeline fix: earlier mixes used a per-example ≥512-token filter that silently kept only **81 of 11,679 sciq training examples (3.3% coverage, each surviving window recycled 37×)**; this release trains on the full 1.27M-token sciq stream. The same bug retroactively retracts the "knowledge saturation is corpus-insensitive" claim from the v0.3/v0.4 era — targeted corpus through distillation is a real knowledge pathway. Cosine matching's net contribution: +0.027 acc_norm (within noise) / +0.065 raw accuracy (2.7σ, argmax sharpening).
>
> The artifact and all measured numbers above are unchanged. Details: [findings-11 §0](../blob/main/docs/findings-11-feature-distill.md) and the findings-9 erratum in this repo.

## 不动的部分

- "What makes v0.6 different" 表格（数字正确；v0.1–v0.3 列的 "~0.40 (57% FP)" 保持）
- "Performance" 表格（实测值）
- Requirements / License / 采样配方
- YAML frontmatter（tags、license、base_model）

## 自动应用（Colab cell，2026-10-05）

在 Colab 新 cell 粘贴运行（需要 secrets 里有 HF_TOKEN，写权限）。所有替换带 assert：
任何锚文本不匹配立即失败、不上传。

```python
from huggingface_hub import hf_hub_download, HfApi

REPO = "benzeng/tritfold-1.7b-feature-ptq1_0"
try:
    from google.colab import userdata
    TOK = userdata.get("HF_TOKEN")
except Exception:
    import getpass; TOK = getpass.getpass("HF token: ")
api = HfApi(token=TOK)

p = hf_hub_download(REPO, "README.md", repo_type="model", token=TOK)
src = open(p).read()

# ① 标语
a = "feature-level distillation (hidden-state cosine matching) transfers discrimination ability that output-level KL cannot"
b = ("targeted in-distribution corpus through teacher KL contributes most of the knowledge gain "
     "(+0.060 sciq), hidden-state cosine matching adds a further +0.027 acc_norm / +0.065 raw "
     "acc — see the control-experiment note below")
assert a in src, "anchor 1 (tagline) not found — aborting, nothing uploaded"
src = src.replace(a, b)
if "Four independent dimensions" in src:
    src = src.replace("Four independent dimensions", "Five independent dimensions")

# ② knowing-vs-saying bullet 替换
lines = src.split("\n")
idx = [i for i, l in enumerate(lines) if "NOT bounded by teacher quality" in l]
assert len(idx) == 1, f"anchor 2 (cosine bullet) found {len(idx)} times — aborting"
lines[idx[0]:idx[0]+1] = [
    "- **Targeted in-distribution corpus through teacher KL** → transfers discrimination "
    "ability (+0.060 sciq over generic-corpus mixes); generic educational corpus transfers "
    "none (+0.004)",
    "- **Feature-level cosine (OFF)** → adds a modest increment on top: +0.027 acc_norm "
    "(within noise) / +0.065 raw accuracy (2.7σ) — sharpens argmax selection",
]
src = "\n".join(lines)

# ③ erratum 节插入 Performance 前
ERRATUM = """## Post-release control experiment (erratum)

The original card attributed the sciq jump (0.398 → 0.485) to feature-level cosine
matching. A control arm (KL-only; identical data mix, steps, teacher and starting
checkpoint) corrects the attribution:

| sciq | acc | acc_norm |
|---|---|---|
| v0.3 baseline (KL, generic corpus) | 0.376* | 0.398 |
| 1a control (KL + full sciq stream) | 0.471 | **0.458** |
| 1b = this model (+ cosine) | **0.536** | **0.485** |

*v0.3 measured on the full n=1000 validation; 1a/1b on the decontaminated n=845
subset — the harder set, so +0.060 is conservative.

Most of the gain came from a data-pipeline fix: earlier mixes used a per-example
≥512-token filter that silently kept only **81 of 11,679 sciq training examples
(3.3% coverage, each surviving window recycled 37×)**; this release trains on the
full 1.27M-token sciq stream. The same bug retroactively retracts the "knowledge
saturation is corpus-insensitive" claim from the v0.3/v0.4 era — targeted corpus
through distillation is a real knowledge pathway. Cosine matching's net
contribution: +0.027 acc_norm (within noise) / +0.065 raw accuracy (2.7σ, argmax
sharpening).

The artifact and all measured numbers above are unchanged. Details in the
[tritfold repo](https://github.com/benzeng/tritfold): findings-11 §0 and the
findings-9 erratum.
"""
lines = src.split("\n")
idx = [i for i, l in enumerate(lines) if l.startswith("#") and "Performance" in l]
assert idx, "anchor 3 (Performance header) not found — aborting"
lines[idx[0]:idx[0]] = [ERRATUM, ""]
src = "\n".join(lines)

# 预览 + 上传
import difflib
print("\n".join(difflib.unified_diff(open(p).read().split("\n"), src.split("\n"),
                                     lineterm="", n=1))[:3000])
api.upload_file(path_or_fileobj=src.encode(), path_in_repo="README.md", repo_id=REPO,
                commit_message="card: post-release control experiment — attribution "
                "corrected (targeted corpus +0.060; cosine +0.027 n.s./+0.065 acc)")
print("uploaded ✓")
```

