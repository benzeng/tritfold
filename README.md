# Tritfold

**Open, end-to-end training of ~1.75 bpw ternary LLMs with Hadamard-folded weights — an independent reproduction built entirely from public math.**

Tritfold takes a standard fp16 LLM (Qwen3), folds its weights into a fixed block-Hadamard basis, distills them into symmetric ternary codes (`{-s, 0, +s}`, one fp16 scale per 128 weights), and emits artifacts that load directly in a production-grade runtime — verified bit-exact against the packing format.

> **Positioning & disclaimer.** Tritfold is an independent research reproduction. It is **not affiliated with, endorsed by, or derived from any proprietary code** of PrismML or Caltech. The method was reconstructed from published papers (QuaRot, SpinQuant, QuIP#, PV-Tuning, BitDistiller, TernaryLLM, LLM-QAT, OneBit — see [Acknowledgments](#acknowledgments)) plus public analysis of released model artifacts. The proprietary training process behind commercial ternary models remains theirs; the gap between our results and theirs is consistent with their reported compute scale. "Bonsai" names appear in docs only as factual references to public products.

## Results

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| Ternary ppl (bf16 protocol, WikiText-2) | 48.08 (**1.77×FP**) | **28.77 (1.41×FP)** |
| FP reference | 27.25 | 20.42 |
| Size (PTQ1_0, exact 1.75 bpw) | 157 MB (**9.1× smaller**) | 424 MB (**9.1× smaller**) |
| Runtime ppl (llama.cpp fork, CPU) | 65.0 | 38.1 |
| Training cost | T4+A100, ~6 h | 1× A100, ~5 h, from scratch |
| Contract verification | C1–C5 PASS, bit-exact | C1–C5 PASS, bit-exact |

Honest quality watermark: **fluent but factually unreliable** — and these are **continuation models, not assistants** (instruct tuning is [roadmap item #1](CONTRIBUTING.md#roadmap)). Real outputs from the 1.7B artifact with the correct sampling recipe:

**✅ English continuation** (home turf — fluent, on-register, facts confabulated):

```
> The Great Wall of China was originally built to
The Great Wall of China, a 1982 novel by the American author and editor,
Robert A. Ewell. In the early 20th century, the Great Wall of China became
a popular tourist attraction...
```

**❌ Chat-style English** (no assistant identity in training data — drifts into the nearest wiki register, here author biographies):

```
> who are you?
I think that's why I got the idea for my first novel, "The New York Times"
(1963). That novel was published in 1964 and 1965, and it is likely that I
had a desire to write an English-language novel...
```

**❌ Non-English prompts** (KD corpus is English-only — collapses into enumeration loops):

```
> 用一段话介绍长城的历史
B. 1. B. 2. 6. 7. B. 1. B. 1. 8. 9. B. 1. B. 1. B. 1. 1. B. 1. B. ...
```

Need chat/instruction behavior? Run [`notebooks/tritfold-instruct-1p7b.ipynb`](notebooks/tritfold-instruct-1p7b.ipynb) — ultrachat-mixed KD from the released checkpoint. The 1.7B artifact crosses the 1.5×FP quality gate on language modeling; instruct-tuning and scale are the open frontiers.

Full experiment records — including every failure (seven falsified discrete-code-movement approaches, an fp16 measurement-inflation artifact that faked a milestone) — are in [`docs/`](docs/) and [`docs/results-log.md`](docs/results-log.md).

## Quickstart

### 1. Try the artifacts

Models (GGUF, Apache-2.0, Qwen3 derivatives):
- **[benzeng/tritfold-1.7b-ptq1_0](https://huggingface.co/benzeng/tritfold-1.7b-ptq1_0)** — 1.41×FP, 424 MB
- [benzeng/tritfold-0.6b-ptq1_0](https://huggingface.co/benzeng/tritfold-0.6b-ptq1_0) — 1.77×FP, 157 MB

Requires the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (`prism` branch) — mainline llama.cpp cannot load PTQ1_0 or apply the Hadamard metadata. **Sampling recipe is mandatory** (ternary distributions have flat tails; defaults loop):

```bash
llama-cli -m tritfold-1.7b.ptq1_0.gguf -p "The Great Wall of China was originally built to" \
    -n 96 -st --temp 0.5 --top-p 0.85 --top-k 20 --repeat-penalty 1.1
```

### 2. Train your own (Colab, one A100, ~4.5 h)

[`notebooks/tritfold-train-1p7b.ipynb`](notebooks/tritfold-train-1p7b.ipynb) — from-scratch ternary QAT: bf16 STE, zerofrac init, WikiText-103 KD, unbiased skip-aware eval, warm-checkpoint resume, contract-compatible export. The 0.6B variant runs on a free-tier T4.

### 3. Verify the contract

```bash
python proto/common/check_contract.py model.ptq1_0.gguf exported_dir/
```

Checks C1–C5: Hadamard metadata, sign vectors bit-exact vs training seeds, 197-tensor fold list, PTQ1_0 losslessness (dequant == export, maxdiff=0), islands stay full-precision.

## How it works (30-second version)

1. **Fold**: store `W' = W·R⁻¹` where `R = H₁₀₂₄·S/√1024` (fixed random-signed Walsh-Hadamard, per input-width sign vectors); the runtime applies `R` to activations online (FWHT);
2. **Ternarize via latent STE**: codes are derived `clamp(round(Z/s_g), -1, 1)` from an fp32 latent `Z` trained end-to-end against the fp teacher's top-50 logits — the only code-movement mechanism that survived our ablations;
3. **Pack losslessly**: end-state weights are exactly `±group-amax / 0`, so the runtime's naive amax+RTN repacking is bit-exact (1.75 bpw, base-3 trit packing);
4. **Measure honestly**: fp16 forward inflates eval on drifted ternary models by up to 43% — evaluate in bf16, per-window, and require zero skipped windows for a "best".

## Repository layout

```
notebooks/    Colab: train 0.6B / train 1.7B / instruct-tune / serve on A100
generators/   Python sources that generate the notebooks (single source of truth)
proto/        Local verification chain: contract checker, dequantizer, probes, E2-lite trainer
docs/         Forensics notes, method reconstruction, implementation plan, six findings reports
```

Path placeholders in docs: `<FORK_DIR>` = PrismML fork checkout, `<RUNTIME_BIN>` = its built binaries, `<WORK_DIR>` = your scratch dir, `<MODELS_DIR>` = your model dir.

## 中文摘要

Tritfold 是一条完全基于公开数学的 ~1.75 bpw 三值 LLM 训练管线：固定 Hadamard 基折叠 + 潜变量 STE 端到端蒸馏 + 无损 base-3 打包，产物经运行时逐位验证。Qwen3-1.7B 在单卡 A100 五小时训练后达到 FP 困惑度的 1.41 倍、体积压缩 9.1 倍。仓库含完整实验档案（含全部失败路径）、四个可复现 Colab notebook 与契约校验工具。与 PrismML/Caltech 无关联；其专有训练过程仍属其所有。

## Acknowledgments

Built on public research: QuaRot (2404.00456) · SpinQuant (2405.16406) · QuIP# (2402.04396) · PV-Tuning (2405.14852) · BitDistiller (2402.10631) · TernaryLLM (2406.07177) · LLM-QAT (2305.17888) · OneBit (2402.11295). Base models: Qwen3 (Apache-2.0). Runtime compatibility target: the public PrismML llama.cpp fork and GGUF contract (`prism.hadamard.*`), documented via artifact analysis.

## License

Apache-2.0. Model weights are derivatives of Qwen3 (Apache-2.0).
