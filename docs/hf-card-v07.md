---
license: apache-2.0
base_model:
- Qwen/Qwen3-1.7B
- Qwen/Qwen3-1.7B-Base
- Qwen/Qwen3-8B
tags:
- gguf
- ternary
- quantization
- quantization-aware-training
- knowledge-distillation
- tritfold
- conversational
---

# Tritfold 1.7B Stacked (v0.7) — 1.75 bpw ternary, three stacked levers

The strongest Tritfold release: **sciq 76% of FP, ARC 84% of FP, wiki ppl 1.014× FP — simultaneously** — at exactly **1.75 bits per weight** (424 MB, 9.1× smaller than FP16).

## The recipe: three independently-validated levers, stacked

Each lever was isolated in its own controlled arm before this run:

| Lever | Validated in | sciq gain (acc_norm) |
|---|---|---|
| Teacher quality: 1.7B → 8B KL (identical vocab, top-50 targets) | v0.5 vs v0.3 | +0.041 |
| Targeted corpus: full sciq stream through teacher KL (1.27M tok) | M9' 1a vs v0.3 | +0.060 |
| Feature matching: hidden-state cosine, 18 layers (OFF-style) | M9' 1b vs 1a | +0.027 |

**Additivity was a blind prediction**: summing the three deltas from three separate controlled experiments gave 0.526; this model measured **0.530 ± 0.034**. The levers are independent and compose linearly.

## Corpus targeting is bidirectional

- A sciq stream moves sciq (+0.060) and not ARC (+0.007)
- An ARC-Challenge+Easy train stream moves ARC: **0.261 → 0.319** (84% of FP; 8B-teacher reference 0.472)
- Near-duplicate leakage between ARC train/test audited: 25/1172 (2.1%) — too small to explain the jump

*Feed what you want it to know.* Both earlier "knowledge ceilings" (sciq saturates ~0.40; ARC stuck near chance) are revised: the ceiling is a soft boundary set by capacity AND corpus, pushed open by targeted distillation, raised further by teacher quality.

## Dual-teacher, cache-phase design

The 8B teacher and the 1.7B geometry teacher (same-dim hidden anchor for cosine — 8B hidden is 4096-d vs student 2048) cannot both stay resident in 40 GB. The 8B resides only during an 8-minute cache phase: all training windows' top-50 targets are precomputed to pinned CPU memory (~1.6 GB), then the teacher is fully freed. Training runs at ~22 GB peak with table-lookup KL targets + live 1.7B geometric anchor.

## Performance

| Probe | Tritfold v0.7 | Qwen3-1.7B FP | % FP |
|---|---|---|---|
| sciq acc / acc_norm (decontaminated n=845) | 0.569 / **0.530** | 0.751 / 0.699 | 76% |
| ARC-c acc / acc_norm (n=1172) | 0.296 / **0.319** | 0.357 / 0.378 | 84% |
| wiki ppl (bf16 protocol, best) | **20.70** | 20.42 | 1.014× |
| Chat generation | 3/3 fluent, on-register | — | facts still confabulated |

Honest watermark unchanged: fluent but factually unreliable in free generation — discrimination (likelihood probes) far exceeds free recall. Use the mandatory sampling recipe.

## Requirements

- **Runtime**: PrismML llama.cpp fork (prism branch) — PTQ1_0 is a private ggml type (143) + `prism.hadamard.*` metadata; mainline llama.cpp cannot serve it
- **Mandatory sampling recipe**: `--temp 0.5 --top-p 0.85 --top-k 20 --repeat-penalty 1.1` (ternary distributions have flat tails; default sampling degrades into loops)

## Lineage

Chained via GGUF-as-checkpoint bootstrap: v0.1 (wiki) → v0.2 (instruct) → v0.3 (knowledge mix) → v0.6 (feature-distill) → **v0.7 (stacked)**. Full experiment log: [tritfold on GitHub](https://github.com/benzeng/tritfold) (findings-12).
