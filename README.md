# Tritfold

**Open, end-to-end training of ~1.75 bpw ternary LLMs with Hadamard-folded weights — an independent reproduction built entirely from public math.**

Tritfold takes a standard fp16 LLM (Qwen3), folds its weights into a fixed block-Hadamard basis, distills them into symmetric ternary codes (`{-s, 0, +s}`, one fp16 scale per 128 weights), and emits artifacts that load directly in a production-grade runtime — verified bit-exact against the packing format.

> **Positioning & disclaimer.** Tritfold is an independent research reproduction. It is **not affiliated with, endorsed by, or derived from any proprietary code** of PrismML or Caltech. The method was reconstructed from published papers (QuaRot, SpinQuant, QuIP#, PV-Tuning, BitDistiller, TernaryLLM, LLM-QAT, OneBit — see [Acknowledgments](#acknowledgments)) plus public analysis of released model artifacts. The proprietary training process behind commercial ternary models remains theirs; the gap between our results and theirs is consistent with their reported compute scale. "Bonsai" names appear in docs only as factual references to public products.

## Results

### The 1.7B trilogy (one A100 session each, chained via GGUF bootstrap)

| | v0.1 wiki | v0.2 +instruct | v0.3 +know | **v0.6 +feat-distill** | FP ref |
|---|---|---|---|---|---|
| Wiki ppl (bf16 protocol, best) | 28.77 (1.41×) | 25.73 (1.26×) | 25.20 (1.24×) | **26.52 (1.03×)** | 20.42 |
| Runtime ppl (llama.cpp fork, CPU) | 38.10 | 35.20 | 33.51 | — | ~27 |
| EN instruction following | ❌ confabulates | ✅ | ✅ kept | ✅ kept | — |
| ARC-c acc_norm (likelihood) | 0.224 | 0.234 | 0.261 | **0.279** | 0.377 |
| sciq acc_norm (decontaminated) | — | 0.363 | 0.398 | **0.485** | 0.699 |
| ARC generative (raw, parsed ≈) | — | coin-flip | coin-flip | **~0.24** | 0.733 / **0.789** (chat) |

Plus Qwen3-0.6B: 48.08 (**1.77×FP**), 157 MB, runs on a free-tier T4. All artifacts: exact 1.75 bpw (**9.1× smaller**), contract C1–C5 PASS bit-exact, trained from scratch in ~5 h on one A100.

### The two-dimension discovery (v0.5 cross-size teacher)

Switching to a Qwen3-8B teacher pushed wiki ppl to **20.62 = 101% of FP** and ARC acc to **0.355 = 99% of FP** — near-lossless language modeling at 1.75 bpw. But generation quality (facts, coherence) did NOT improve. The insight:

> **Distribution matching and knowledge internalization are independent dimensions.**
> - Distributional properties (register, grammar, output shape) are low-rank → transferable via distillation → **teacher quality is the bottleneck**
> - Factual knowledge (specific mappings) is dense → needs storage capacity per fact → **1.58 bits can't hold it regardless of teacher**
>
> You can learn someone's accent by imitation, but not their phone book.

**v0.6 took the knowledge probe to 72% of FP** (sciq 0.398→0.485), with ultrachat in the mix restoring generation fluency — the first ternary that *knows* (discriminates) AND *speaks* (generates coherently). The follow-up control arm (KL-only, identical data/steps/start) later split the attribution: most of the gain (**+0.060**) came from fixing a sciq data-pipeline bug (a per-example ≥512-token filter had silently kept only **81 of 11,679 examples — 3.3% coverage — in all earlier mixes**); hidden-state cosine matching (OFF) added **+0.027 acc_norm (within noise) / +0.065 raw accuracy (2.7σ)** — a real but modest argmax-sharpening effect. Factual recall in free text remains capacity-bound. See [the model](https://huggingface.co/benzeng/tritfold-1.7b-feature-ptq1_0), [findings-11](docs/findings-11-feature-distill.md) (§0 control + erratum) and [findings-9](docs/findings-9-corpus-matrix.md) (erratum).

### The knowledge ceiling — revised after the v0.4 erratum (v0.3 + v0.4 corpus matrix + M9′ control)

Three lines of evidence, one retracted:
1. **v0.3**: no ARC protocol shows above-random closed-book knowledge at 1.75 bpw while FP scores 0.73–0.79;
2. **v0.4-zh**: capability allocation is **zero-sum** — Chinese bandwidth was paid for with ARC -7% and sciq -9%;
3. ~~v0.4-know: "the right corpus doesn't fit either"~~ **RETRACTED** — the sciq stream in that arm was 81 windows recycled 37× (3.3% coverage, per-example ≥512 filter bug). Re-run properly (full 1.27M-token sciq stream through teacher KL, M9′ 1a control): **+0.060**. Targeted corpus through distillation is a real knowledge pathway.

```
> What planet is known as the Red Planet?
"The red planet is a star in the constellation of ... one of the most
famous stars in the universe"          # discourse intact, facts confabulated
```

The refined picture: compression destroys *free recall* (ARC random) but **context-assisted association survives and is trainable** — generic educational corpus moves it not at all, while targeted in-distribution corpus through teacher distillation moved sciq 0.398→0.458 (+0.060), and cosine feature matching added a modest sharpening increment on top (→0.485 = 72% of FP). The ceiling is **capacity-bound, not corpus-bound** — it yields to the right corpus, but never reaches FP (0.485 vs 0.699). Chinese sentence-level writing is teachable with enough dose (topic anchoring needs 3-5× more). The remaining paths to *free recall*: **scale** (7-8B/27B) or **RAG** (which happens to complement exactly the surviving mode).

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
- **[benzeng/tritfold-1.7b-feature-ptq1_0](https://huggingface.co/benzeng/tritfold-1.7b-feature-ptq1_0)** — v0.6 feature-distilled: **sciq 72% FP + generation restored** + wiki ppl 1.03×FP; first ternary with both knowing AND saying
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

## The central question: does ternarization require retraining?

Yes — and the project's full evidence chain gives the answer structure:

**1. Pure conversion is catastrophic (representation capacity, not tuning).** Zero-training RTN lands at ppl ~10⁹ (84% of weights zeroed); even a zero-fraction-aware init stays 8× worse than random guessing, and the Hadamard rotation does not help weight-only RTN (per-layer error ratio ≈ 1, our falsification of the naive QuaRot analogy). The FP model's capability *is* in those 16-bit values; a 1.58-bit alphabet cannot hold them by rounding.

**2. The retraining is distillation, not from-scratch — and what recovers is layered.**

| training investment | what recovers | how far |
|---|---|---|
| scales + islands only (codes frozen) | language-modeling shape | 3.3×FP, then saturated |
| end-to-end latent STE + ~20M tokens | language modeling | **1.24×FP** (76–81% fidelity) |
| + instruction mix | instruction following, chat register | qualitative jump |
| + knowledge mix | format discipline | qualitative jump |
| all of the above | **knowledge / facts** | **random — the measured ceiling** |

**3. The architecture inverts the question.** Training happens *inside* the ternary+rotated representation (latent-Z distillation → snap to ±group-amax/0); the final "conversion" (RTN repack) is mathematically lossless (verified maxdiff=0). Not "convert then patch up" — *retrain in the target representation, and conversion is free*. And capability splits into two kinds: **distributional** (register, format, language modeling — lives in weight statistics, distillation moves it, ~80% recoverable) versus **informational** (facts, knowledge — lives in exact values, 1.58 bits can't hold it, must be re-taught at scale; this is the compute gap to the commercial models).

One sentence: **a ternarized model is not a compressed model — it is a new model, distilled into three-valued space, that inherits the teacher's instincts but must re-learn its facts.**

## What survives from the base model — and how this compares to BitNet

**What the fp16 base contributes to its ternary derivative (measured):**

1. **The skeleton — ~81% of the code pattern.** Codes initialize from the folded weights' quantization (`sign(W·R⁻¹)` + group scales); after four training rounds only ~19% have moved. The qualitative connectivity (which connection is positive / negative / absent) is the base's, and the Hadamard fold is what makes it meaningful: in the incoherent basis, each coordinate's sign captures directional information of the weight row.
2. **The instincts — the teacher distribution.** The KD signal *is* the base model's behavior on ~30M tokens; register, grammar, instruction-response feel all transfer through it (v0.2's instruction jump was literally the base's instruct behavior, moved).
3. **The starting points** — group scales from local statistics (93× better init), norm islands near-verbatim.

What does *not* survive: exact magnitudes (by construction of 1.58 bits) and factual knowledge (the measured ceiling). **The base contributes "how to speak" (~80% fidelity); it cannot contribute "what is known."**

**vs. BitNet-style from-scratch ternary training** — the mechanisms have converged (both train ternary via STE); the difference is signal source (teacher KD vs raw next-token) and budget (~20M tokens vs trillions):

| | conversion (this repo, Bonsai) | from-scratch (BitNet) |
|---|---|---|
| compute | **~5 GPU·h per variant** | pretraining-scale (3–4 orders more) |
| ecosystem | any fp16 model, hours to ternary; chain onto new bases as they release | must commit at token 0 |
| knowledge | ceiling at this budget (measured) | re-taught natively — no gap |
| Hadamard fold | required (conversion tax) | unnecessary (native training grows ternary-friendly weights) |

The commercial Bonsai models occupy the hybrid middle — conversion architecture with near-pretraining re-distillation budgets (200–800 A100·h at 27B) — suggesting the industry answer is "conversion architecture + pay for the facts." One line: **BitNet proves ternary models can be built; conversion proves existing models can be moved. The exchange rate (fidelity per GPU-hour) is set by how much re-distillation you pay — 5 hours buys ~80% of the instincts; the facts are left as an exercise for the compute.**

## Repository layout

```
notebooks/    Colab: train 0.6B / train 1.7B / instruct-tune (v0.2) / knowledge (v0.3) / serve on A100
generators/   Python sources that generate the notebooks (single source of truth)
proto/        Local verification chain: contract checker, dequantizer, probes, ARC harnesses
docs/         Forensics notes, method reconstruction, implementation plan, eight findings reports
```

Path placeholders in docs: `<FORK_DIR>` = PrismML fork checkout, `<RUNTIME_BIN>` = its built binaries, `<WORK_DIR>` = your scratch dir, `<MODELS_DIR>` = your model dir.

## 中文摘要

Tritfold 是一条完全基于公开数学的 ~1.75 bpw 三值 LLM 训练管线：固定 Hadamard 基折叠 + 潜变量 STE 端到端蒸馏 + 无损 base-3 打包，产物经运行时逐位验证。1.7B 系列经三轮链式蒸馏（纯 wiki → 指令混合 → 知识混合，每轮一个 A100 会话、从上一代发布的 GGUF 无损自举）：困惑度 28.77→25.20（FP 的 1.24 倍）、体积压缩 9.1 倍、英文指令跟随质变；同时以三协议 ARC 实证了 **1.75 bpw 的知识天花板**（FP 0.73–0.79 vs 三值随机）。仓库含完整实验档案（含全部失败路径）、五个可复现 Colab notebook 与契约校验工具。核心结论：三值化必须重训（纯转换 ppl→10⁹），且重训是“在目标表示内蒸馏”——分布性能力（文体/格式/语言建模）可恢复约八成，信息性能力（知识/事实）受 1.58 bit 容量约束须从数据重教。与 PrismML/Caltech 无关联；其专有训练过程仍属其所有。

## Acknowledgments

Built on public research: QuaRot (2404.00456) · SpinQuant (2405.16406) · QuIP# (2402.04396) · PV-Tuning (2405.14852) · BitDistiller (2402.10631) · TernaryLLM (2406.07177) · LLM-QAT (2305.17888) · OneBit (2402.11295). Base models: Qwen3 (Apache-2.0). Runtime compatibility target: the public PrismML llama.cpp fork and GGUF contract (`prism.hadamard.*`), documented via artifact analysis.

## License

Apache-2.0. Model weights are derivatives of Qwen3 (Apache-2.0).
