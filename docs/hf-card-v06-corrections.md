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
