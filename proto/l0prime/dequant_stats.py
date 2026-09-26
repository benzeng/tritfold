#!/usr/bin/env python
"""L0' T1.2-T1.7: PQ2_0 dequant + statistical tests on Ternary-Bonsai-2-27B.

Codec (ggml-quants.c:113/494, ggml-common.h:199-207):
  block_pq2_0 = { fp16 d; uint8 qs[32] }  -> 34 B per 128 weights
  quantize: d = group amax; q = round(w/d) + 1 clamped to [0,3]
  dequant:  w = (q - 1) * d          (00=-1, 01=0, 10=+1, 11=+2)

Usage: dequant_stats.py <gguf_path> [out_json]
"""
import json
import sys

import numpy as np
import gguf

QK = 128          # weights per block
BS = 34           # bytes per block
SHIFTS = np.array([0, 2, 4, 6], dtype=np.uint8)

GGUF = "<MODELS_DIR>/Ternary-Bonsai-2-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"

# representative folded tensors: (name, max_rows_to_analyze or None for all)
TENSORS = [
    ("blk.0.attn_qkv.weight", None),    # GDN, in 5120
    ("blk.0.attn_gate.weight", None),   # GDN, in 5120
    ("blk.0.ssm_out.weight", None),     # GDN, in 6144
    ("blk.0.ffn_gate.weight", None),    # in 5120
    ("blk.0.ffn_down.weight", None),    # in 17408
    ("blk.11.attn_q.weight", None),     # softmax block, in 5120
    ("blk.11.attn_output.weight", None),# softmax block, in 12288 (no sign table)
    ("blk.63.ffn_down.weight", None),   # late layer
    ("token_embd.weight", 8192),        # sampled rows (inverse-after-lookup)
    ("output.weight", 8192),            # sampled rows (lm_head, folded)
]


def dequant_codes(u8):
    """uint8 view (rows, nb*BS) -> codes int8 (rows, nb*128), d fp32 (rows, nb)."""
    rows, nbytes = u8.shape
    nb = nbytes // BS
    b = np.asarray(u8).reshape(rows, nb, BS)
    d = b[:, :, :2].copy().view(np.float16).astype(np.float32)[:, :, 0]
    qs = b[:, :, 2:]
    codes = (qs[:, :, :, None] >> SHIFTS) & np.uint8(3)
    return codes.reshape(rows, nb * QK).astype(np.int8), d


def kurtosis(x, axis=1):
    """Pearson kurtosis (Gaussian = 3) along axis."""
    m = x.mean(axis=axis, keepdims=True)
    c = x - m
    m2 = (c ** 2).mean(axis=axis)
    m4 = (c ** 4).mean(axis=axis)
    return m4 / np.maximum(m2, 1e-30) ** 2


def fwht(x):
    """Unnormalized Walsh-Hadamard (Sylvester butterfly) along last dim."""
    n = x.shape[-1]
    y = x.astype(np.float64).copy()
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        p = y[..., 0, :] + y[..., 1, :]
        m = y[..., 0, :] - y[..., 1, :]
        y = np.concatenate([p[..., None, :], m[..., None, :]], axis=-2)
        y = y.reshape(*y.shape[:-3], n)
        h *= 2
    return y


def tensor_stats(name, codes, d):
    nb = d.shape[1]
    g = codes.reshape(codes.shape[0], nb, QK)
    hist = [int((codes == c).sum()) for c in range(4)]
    total = codes.size
    has_pm1 = (g != 1).any(axis=2)                      # group contains ±1
    d_pos = d > 0
    viol = int((d_pos & ~has_pm1).sum())                # scale>0 but no ±1 -> scale != amax
    n_grp = int(d_pos.sum())
    zero_frac = hist[1] / total
    col_zero = (codes == 1).mean(axis=0)                # per-column zero fraction
    return {
        "tensor": name, "rows": int(codes.shape[0]), "cols": int(codes.shape[1]),
        "code_hist_-1/0/+1/+2": hist,
        "zero_frac": round(float(zero_frac), 6),
        "plus_minus_balance": round(hist[2] / max(hist[0], 1), 4),
        "code3_count": hist[3],
        "groups_d>0": n_grp,
        "groups_missing_pm1": viol,
        "groups_missing_pm1_frac": round(viol / max(n_grp, 1), 8),
        "allzero_groups(d==0)": int((~d_pos).sum()),
        "d_mean": round(float(d[d_pos].mean()), 6) if n_grp else 0.0,
        "d_cv": round(float(d[d_pos].std() / d[d_pos].mean()), 4) if n_grp else 0.0,
        "col_zero_min/max": [round(float(col_zero.min()), 4), round(float(col_zero.max()), 4)],
        "col_zero_p5/p95": [round(float(np.percentile(col_zero, 5)), 4),
                            round(float(np.percentile(col_zero, 95)), 4)],
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else GGUF
    out_json = sys.argv[2] if len(sys.argv) > 2 else None
    r = gguf.GGUFReader(path)
    tmap = {t.name: t for t in r.tensors}

    meta = {k: (f.contents() if hasattr(f, "contents") else None) for k, f in r.fields.items()}
    svalues = meta.get("prism.hadamard.sign_values") or []
    swidths = meta.get("prism.hadamard.sign_widths") or []

    results = []
    for name, max_rows in TENSORS:
        t = tmap.get(name)
        if t is None:
            print(f"!! {name} not found"); continue
        ty = str(t.tensor_type)
        if "PQ2_0" not in ty:
            print(f"!! {name} is {ty}, skipped (expected PQ2_0)"); continue
        u8 = t.data  # uint8 view (rows=ne1, nb*34)
        if max_rows and u8.shape[0] > max_rows:
            idx = np.random.default_rng(0).choice(u8.shape[0], max_rows, replace=False)
            u8 = np.asarray(u8[sorted(idx)])
        codes, d = dequant_codes(u8)
        st = tensor_stats(name, codes, d)
        results.append(st)
        print(f"{name:32s} rows={st['rows']:6d} cols={st['cols']:6d} "
              f"zero={st['zero_frac']:.4f} bal={st['plus_minus_balance']:.3f} "
              f"code3={st['code3_count']} missPM1={st['groups_missing_pm1_frac']:.2e} "
              f"colZero p5/p95={st['col_zero_p5/p95']}")
        del codes, d

    # (e) embedding rotated-domain test: kurtosis before/after inverse transform
    emb = tmap["token_embd.weight"]
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(emb.data.shape[0], 4096, replace=False))
    codes, d = dequant_codes(np.asarray(emb.data[idx]))
    eprime = (codes.astype(np.float32) - 1.0) * np.repeat(d, QK, axis=1)
    n = 1024
    ehat = fwht(eprime.reshape(eprime.shape[0], -1, n)).reshape(eprime.shape) / np.sqrt(n)
    k_before = kurtosis(eprime.astype(np.float64))
    k_after = kurtosis(ehat)
    emb_test = {
        "sampled_rows": 4096,
        "kurtosis_stored_rotated": [round(float(np.percentile(k_before, p)), 3) for p in (5, 50, 95)],
        "kurtosis_after_inverse": [round(float(np.percentile(k_after, p)), 3) for p in (5, 50, 95)],
        "row_norm_stored_p50": round(float(np.percentile(np.linalg.norm(eprime, axis=1), 50)), 3),
        "row_norm_after_inverse_p50": round(float(np.percentile(np.linalg.norm(ehat, axis=1), 50)), 3),
    }
    print("\nembedding test:", emb_test)

    # control: a non-folded BF16 FP tensor (ssm_beta, in 5120 -> out 48)
    ctrl = tmap["blk.0.ssm_beta.weight"]
    raw = np.asarray(ctrl.data)  # bf16 as uint16 view? gguf-py: BF16 -> uint8 bytes
    if str(ctrl.tensor_type) == "GGMLQuantizationType.BF16":
        u16 = np.asarray(ctrl.data).view(np.uint16) if ctrl.data.dtype != np.uint16 else np.asarray(ctrl.data)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32).reshape(-1, int(ctrl.shape[0]))
        kc = kurtosis(f32.astype(np.float64), axis=1)
        emb_test["control_bf16_ssm_beta_kurtosis_p5/50/95"] = [
            round(float(np.percentile(kc, p)), 3) for p in (5, 50, 95)]

    summary = {
        "per_tensor": results,
        "embedding_inverse_test": emb_test,
        "sign_widths": swidths,
        "sign_values_len": len(svalues),
        "weighted_zero_frac": round(float(
            sum(s["code_hist_-1/0/+1/+2"][1] for s in results) /
            sum(sum(s["code_hist_-1/0/+1/+2"]) for s in results)), 6),
    }
    print("\nweighted zero fraction over sampled tensors:", summary["weighted_zero_frac"])
    if out_json:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=1)
        print("wrote", out_json)


if __name__ == "__main__":
    main()
