# Contributing to Tritfold

## Roadmap (good first targets)

| # | Direction | Entry point | Notes |
|---|---|---|---|
| 1 | **Instruct-tuning** (M5′) | `notebooks/tritfold-instruct-1p7b.ipynb` | ultrachat+wiki mixed KD from the 1.7B checkpoint; ARC-c mini-harness + ppl guard included; needs a first full run + findings write-up |
| 2 | **7–8B port** | adapt `generators/gen_train_1p7b.py` | needs 80GB GPU or ZeRO/sharding for the fp32 latent (~70GB+); scale effect so far: 1.77×→1.41×FP from 0.6B→1.7B |
| 3 | **Benchmark harness** | extend the ARC mini-harness | MMLU / instruction-following evals on the PyTorch wrapper (not GGUF — avoids runtime-protocol confounds; see findings-6 for the three-protocol decomposition) |
| 4 | **More architectures** | `proto/common/fwht_torch.py` is arch-agnostic; fold list must match the runtime whitelist (attn/ffn projections, ssm_out, lm_head, token_embd inverse) | qwen35/GDN hybrid-attention is the interesting case (V-reorder contract) |
| 5 | **Activation-quant-aware training** | fake-quant int8 activations (MMQ path) in the STE stack | worth ~7% runtime-protocol ppl (measured, findings-6) |
| 6 | **Kernel perf** | PTQ1_0 MMQ / FWHT in the fork | serving notebook has the build harness |

## Ground rules

- **Every claim gets a number.** New experiments append to `docs/results-log.md` and, when a milestone closes, a findings doc. Negative results are first-class (see findings-4).
- **Evaluation discipline**: bf16 forward, per-window NLL, `skipped == 0` required for any "best" or checkpoint claim; report the protocol explicitly (three protocols differ by up to 43% — findings-6).
- **One variable at a time** (learned the hard way — findings-5 §v6).
- Notebooks are generated from `generators/*.py` — edit the generator, regenerate, commit both.
- Scrub machine-specific paths; use the documented placeholders (`<FORK_DIR>` etc.).

## Dev environment notes

Local verification chain (`proto/`): Python 3.10, torch 2.x, transformers 4.57.1, editable install of the fork's `gguf-py`. Colab notebooks pin their own stack (see cell 1 of each — the pins are load-bearing; newer images drift).
