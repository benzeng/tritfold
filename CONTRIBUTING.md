# Contributing to Tritfold

## Roadmap (good first targets)

| # | Direction | Entry point | Notes |
|---|---|---|---|
| 1 | **M11: max out the 1.7B recipe** — sciq dose-response (1.2/2.4/4.8 epochs), 4500 steps, full-depth (28-layer) cosine hooks, plus measure the 8B teacher's own sciq (never measured — decides whether the gap is transfer- or corpus-limited) | adapt `generators/gen_m10_stack.py` (~15 units, one session) | gate: sciq ≥0.58 → close the 1.7B line at ~90% FP discrimination; else confirm capacity wall → M12. Full analysis: `docs/path-to-fp-parity.md` |
| 2 | **M12: ternary 8B** — the iso-size end-run: 1.75GB ternary-8B is HALF the bytes of fp16-1.7B with 4× parameters; expected ~1.2×FP, i.e. beats FP 1.7B on everything at half the size | M4 pipeline at 8B scale (~150-200 units total chain) | the strategic answer to "reach the original 1.7B's capability" — see path-to-fp-parity §3 |
| 3 | **Benchmark harness** — MMLU / instruction-following / long-context baselines on the PyTorch wrapper (both student AND teacher refs; the 8B ARC ref came free in M10's cache cell — same pattern) | extend the ARC mini-harness | "原 1.7B 能力" is currently measured on 2 probes + spot checks only |
| 4 | **Activation-quant-aware training** | fake-quant int8 activations in the STE stack | worth ~7% runtime-protocol ppl (measured, findings-6) |
| 5 | **Kernel perf** | PTQ1_0 MMQ / FWHT in the fork | serving notebook has the build harness |
| 6 | **More architectures** | `proto/common/fwht_torch.py` is arch-agnostic | fold list must match the runtime whitelist |


## Ground rules

- **Every claim gets a number.** New experiments append to `docs/results-log.md` and, when a milestone closes, a findings doc. Negative results are first-class (see findings-4).
- **Evaluation discipline**: bf16 forward, per-window NLL, `skipped == 0` required for any "best" or checkpoint claim; report the protocol explicitly (three protocols differ by up to 43% — findings-6).
- **One variable at a time** (learned the hard way — findings-5 §v6).
- Notebooks are generated from `generators/*.py` — edit the generator, regenerate, commit both.
- Scrub machine-specific paths; use the documented placeholders (`<FORK_DIR>` etc.).

## Dev environment notes

Local verification chain (`proto/`): Python 3.10, torch 2.x, transformers 4.57.1, editable install of the fork's `gguf-py`. Colab notebooks pin their own stack (see cell 1 of each — the pins are load-bearing; newer images drift).
