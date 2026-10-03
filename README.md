# Tritfold

**Open, end-to-end training of ~1.75 bpw ternary LLMs with Hadamard-folded weights — an independent reproduction built entirely from public math.**

Tritfold takes a standard fp16 LLM (Qwen3), folds its weights into a fixed block-Hadamard basis, distills them into symmetric ternary codes (`{-s, 0, +s}`, one fp16 scale per 128 weights), and emits artifacts that load directly in a production-grade runtime — verified bit-exact against the packing format.

> **Positioning & disclaimer.** Tritfold is an independent research reproduction. It is **not affiliated with, endorsed by, or derived from any proprietary code** of PrismML or Caltech. The method was reconstructed from published papers (QuaRot, SpinQuant, QuIP#, PV-Tuning, BitDistiller, TernaryLLM, LLM-QAT, OneBit — see [Acknowledgments](#acknowledgments)) plus public analysis of released model artifacts. The proprietary training process behind commercial ternary models remains theirs; the gap between our results and theirs is consistent with their reported compute scale. "Bonsai" names appear in docs only as factual references to public products.

## Results

### The 1.7B trilogy (one A100 session each, chained via GGUF bootstrap)

| | v0.1 wiki-only | v0.2 +instruct | v0.3 +knowledge | FP ref |
|---|---|---|---|---|
| Wiki ppl (bf16 protocol, best) | 28.77 (1.41×) | 25.73 (1.26×) | **25.20 (1.24×)** | 20.42 |
| Runtime ppl (llama.cpp fork, CPU) | 38.10 | 35.20 | **33.51** | ~27 |
| EN instruction following | ❌ confabulates | ✅ | ✅ kept | — |
| ARC-c acc_norm (likelihood) | 0.224 | 0.234 | **0.261** (first above random) | 0.377 |
| ARC generative (raw, parsed ≈) | — | coin-flip | coin-flip | 0.733 / **0.789** (chat) |

Plus Qwen3-0.6B: 48.08 (**1.77×FP**), 157 MB, runs on a free-tier T4. All artifacts: exact 1.75 bpw (**9.1× smaller**), contract C1–C5 PASS bit-exact, trained from scratch in ~5 h on one A100.

### The knowledge ceiling (v0.3's headline finding)

Measured across three ARC protocols — likelihood, raw-format generation, chat-format generation — **no protocol shows above-random letter knowledge at 1.75 bpw, while FP scores 0.73–0.79**: the gap is capacity, not data recipe. ~12M tokens of educational distillation rebuild format and discourse, not facts:

```
> What planet is known as the Red Planet?
"The red planet is a star in the constellation of ... one of the most
famous stars in the universe"          # discourse intact, facts confabulated
```

v0.4 frontier: 7–8B scale · 10× knowledge corpus · RAG externals.

Honest quality watermark: **fluent but factually unreliable** — and these are **continuation models, not assistants** (instruct tuning is [roadmap item #1](CONTRIBUTING.md#roadmap)). Real outputs from the 1.7B artifact with the correct sampling recipe:

**✅ English continuation** (home turf — fluent, on-register, facts confabulated):

```
> The Great Wall of China was originally built to
The Great Wall of China, a 1982 novel by the American author and editor,
Robert A. Ewell. In the early 20th century, the Great Wall of China became
a popular tourist attraction...
```

**🔶 Chat-style English — fixed by the [instruct variant](https://huggingface.co/benzeng/tritfold-1.7b-instruct-ptq1_0) (v0.2)**. The wiki-only model drifts into confabulated biographies; after ultrachat-mixed distillation it answers on-task (facts still unreliable):

```
wiki-only base:                "I think that's why I got the idea for my first
                                novel, 'The New York Times' (1963)..."
instruct variant (v0.2):       "You're a person who is interested in science,
                                and you're always eager to learn more about
                                the world around you..."
```

**❌ Facts, always** (the 1.75 bpw watermark — true in every variant):

```
> What planet is known as the Red Planet?
"The **Red Planet** is a star in the constellation of **Pisces** and is one of
the most distant planets in the solar system..."
```

**❌ Non-English prompts** (KD corpus is English-only — collapses into enumeration loops):

```
> 用一段话介绍长城的历史
B. 1. B. 2. 6. 7. B. 1. B. 1. 8. 9. B. 1. B. 1. B. 1. 1. B. 1. B. ...
```

**Published GGUFs are training carriers**: every notebook bootstraps from the released HF artifact itself (bit-exact weight-state recovery, no Drive checkpoints) — v0.1 → v0.2 via [`tritfold-instruct-1p7b.ipynb`](notebooks/tritfold-instruct-1p7b.ipynb), v0.2 → v0.3 via [`tritfold-v03-knowledge.ipynb`](notebooks/tritfold-v03-knowledge.ipynb). Non-English remains unfixed (EN-dominant corpora).

Full experiment records — including every failure (seven falsified discrete-code-movement approaches, an fp16 measurement-inflation artifact that faked a milestone) — are in [`docs/`](docs/) and [`docs/results-log.md`](docs/results-log.md).

## Quickstart

### 1. Try the artifacts

Models (GGUF, Apache-2.0, Qwen3 derivatives):
- **[benzeng/tritfold-1.7b-ptq1_0](https://huggingface.co/benzeng/tritfold-1.7b-ptq1_0)** — 1.41×FP, 424 MB (wiki-only)
- **[benzeng/tritfold-1.7b-instruct-ptq1_0](https://huggingface.co/benzeng/tritfold-1.7b-instruct-ptq1_0)** — v0.2 instruct variant: follows EN instructions, wiki ppl improved to 1.26×FP mid-run
- **[benzeng/tritfold-1.7b-knowledge-ptq1_0](https://huggingface.co/benzeng/tritfold-1.7b-knowledge-ptq1_0)** — v0.3 knowledge mix: wiki ppl record 1.24×FP, ARC breaks random (0.261); *fixes the format, not the facts*
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
notebooks/    Colab: train 0.6B / train 1.7B / instruct-tune (v0.2) / knowledge (v0.3) / serve on A100
generators/   Python sources that generate the notebooks (single source of truth)
proto/        Local verification chain: contract checker, dequantizer, probes, ARC harnesses
docs/         Forensics notes, method reconstruction, implementation plan, eight findings reports
```

Path placeholders in docs: `<FORK_DIR>` = PrismML fork checkout, `<RUNTIME_BIN>` = its built binaries, `<WORK_DIR>` = your scratch dir, `<MODELS_DIR>` = your model dir.

## 中文摘要

Tritfold 是一条完全基于公开数学的 ~1.75 bpw 三值 LLM 训练管线：固定 Hadamard 基折叠 + 潜变量 STE 端到端蒸馏 + 无损 base-3 打包，产物经运行时逐位验证。1.7B 系列经三轮链式蒸馏（纯 wiki → 指令混合 → 知识混合，每轮一个 A100 会话、从上一代发布的 GGUF 无损自举）：困惑度 28.77→25.20（FP 的 1.24 倍）、体积压缩 9.1 倍、英文指令跟随质变；同时以三协议 ARC 实证了 **1.75 bpw 的知识天花板**（FP 0.73–0.79 vs 三值随机）。仓库含完整实验档案（含全部失败路径）、五个可复现 Colab notebook 与契约校验工具。与 PrismML/Caltech 无关联；其专有训练过程仍属其所有。

## Acknowledgments

Built on public research: QuaRot (2404.00456) · SpinQuant (2405.16406) · QuIP# (2402.04396) · PV-Tuning (2405.14852) · BitDistiller (2402.10631) · TernaryLLM (2406.07177) · LLM-QAT (2305.17888) · OneBit (2402.11295). Base models: Qwen3 (Apache-2.0). Runtime compatibility target: the public PrismML llama.cpp fork and GGUF contract (`prism.hadamard.*`), documented via artifact analysis.

## License

Apache-2.0. Model weights are derivatives of Qwen3 (Apache-2.0).
