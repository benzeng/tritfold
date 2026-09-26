#!/usr/bin/env python
"""M3 contract self-check (plan §6.2): verify a produced GGUF against C1-C5.

C1  prism.hadamard.* metadata: version/block_size/transform/axis/sign_mode,
    sign_widths == [1024,2048,3072], sign_values == training sign vectors exactly
C2  folded tensors are PTQ1_0 (ggml type 143): ternary codes, one fp16 scale
    per 128, no bias
C3  RTN-lossless: decoded values == exported safetensors values bit-exact;
    every group with d>0 contains +/-1 (scale == group amax)
C4  axis == input-last-dimension; weight_names == expected 197-tensor set;
    inverse_weight_names == [token_embd.weight]
C5  norm/island tensors remain F32/F16/BF16 (not quantized)

Usage: check_contract.py <gguf> <exported_hf_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common.fwht_torch import GROUP, signs_for_width

import gguf

POW3 = [1, 3, 9, 27, 81]


def dec_byte(q: np.ndarray, n: int) -> np.ndarray:
    """Trit n of a PTQ1_0 byte: uint8 overflow wrap == mod 256 (ggml-quants.c:2269)."""
    return (((q.astype(np.uint16) * POW3[n]) % 256) * 3) >> 8


def dequant_ptq1_0(u8: np.ndarray):
    """(rows, nb*28) uint8 -> values fp32 (rows, nb*128), d (rows, nb)."""
    rows, nb = u8.shape[0], u8.shape[1] // 28
    b = u8.reshape(rows, nb, 28)
    d = b[:, :, 26:28].copy().view(np.float16).astype(np.float32)[:, :, 0]
    qs, qh = b[:, :, :24], b[:, :, 24:26]
    out = np.empty((rows, nb, 128), dtype=np.int8)
    # stage c=16: bytes 0..15 -> elements m + n*16 (m<16, n<5) => 0..79
    for n in range(5):
        out[:, :, n * 16:(n + 1) * 16] = dec_byte(qs[:, :, 0:16], n)
    # stage c=8: bytes 16..23 -> elements 80 + m + n*8 (m<8, n<5) => 80..119
    for n in range(5):
        out[:, :, 80 + n * 8: 80 + (n + 1) * 8] = dec_byte(qs[:, :, 16:24], n)
    # qh: byte h -> elements 120 + h + m*2 (m<4), trit index m
    for m in range(4):
        out[:, :, 120 + m * 2] = dec_byte(qh[:, :, 0], m)
        out[:, :, 120 + m * 2 + 1] = dec_byte(qh[:, :, 1], m)
    vals = (out.astype(np.float32) - 1.0) * d[:, :, None]
    return vals.reshape(rows, nb * 128), d


def main():
    gguf_path, export_dir = sys.argv[1], Path(sys.argv[2])
    r = gguf.GGUFReader(gguf_path)
    meta = {k: (f.contents() if hasattr(f, "contents") else None) for k, f in r.fields.items()}
    tmap = {t.name: t for t in r.tensors}
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    print("C1: hadamard metadata")
    check("version==1", meta.get("prism.hadamard.version") == 1)
    check("block_size==1024", meta.get("prism.hadamard.block_size") == 1024)
    check("transform", meta.get("prism.hadamard.transform") == "normalized-sylvester-walsh-hadamard")
    check("sign_mode==explicit", meta.get("prism.hadamard.sign_mode") == "explicit")
    # expected sign widths derive from the exported config: hidden, attn internal, inter
    import json as _json
    _cfg = _json.loads((Path(export_dir) / "config.json").read_text())
    _hd = _cfg.get("head_dim") or _cfg["hidden_size"] // _cfg["num_attention_heads"]
    widths_expected = sorted({_cfg["hidden_size"],
                              _cfg["num_attention_heads"] * _hd,
                              _cfg["intermediate_size"]})
    sw = meta.get("prism.hadamard.sign_widths") or []
    check(f"sign_widths=={widths_expected}", list(sw) == widths_expected, f"got {sw}")
    expect = []
    for w in widths_expected:
        expect += signs_for_width(w, "cpu").to(torch.int8).tolist()
    sv = meta.get("prism.hadamard.sign_values") or []
    check("sign_values == training signs", list(sv) == expect,
          f"len {len(sv)} vs {len(expect)}")

    print("C4: axis + tensor lists")
    check("axis", meta.get("prism.hadamard.axis") == "input-last-dimension")
    wn = set(meta.get("prism.hadamard.weight_names") or [])
    expect_wn = {"output.weight"}
    for i in range(28):
        for k in ("attn_q", "attn_k", "attn_v", "attn_output",
                  "ffn_gate", "ffn_up", "ffn_down"):
            expect_wn.add(f"blk.{i}.{k}.weight")
    check("weight_names == expected 197", wn == expect_wn,
          f"got {len(wn)}, missing {sorted(expect_wn - wn)[:3]}, extra {sorted(wn - expect_wn)[:3]}")
    check("inverse == [token_embd.weight]",
          list(meta.get("prism.hadamard.inverse_weight_names") or []) == ["token_embd.weight"])

    print("C2/C3: PTQ1_0 ternary + RTN-lossless (sampled tensors)")
    from safetensors.numpy import load_file
    src = load_file(str(Path(export_dir) / "model.safetensors"))
    samples = ["blk.0.attn_q.weight", "blk.0.ffn_down.weight",
               "blk.27.attn_output.weight", "output.weight", "token_embd.weight"]
    direct = {"output.weight": "lm_head.weight", "token_embd.weight": "model.embed_tokens.weight"}

    def hf_name(gname):
        if gname in direct:
            return direct[gname]
        import re
        m = re.match(r"blk\.(\d+)\.(.*)\.weight", gname)
        i, kind = m.group(1), m.group(2)
        kindmap = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
                   "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
                   "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
        return f"model.layers.{i}.{kindmap[kind]}.weight"

    for gname in samples:
        t = tmap[gname]
        is_ptq = str(t.tensor_type).endswith("PTQ1_0")
        u8 = np.asarray(t.data)
        vals, d = dequant_ptq1_0(u8)
        ref = src[hf_name(gname)].astype(np.float32)
        if ref.shape != vals.shape:
            check(f"{gname}: shape match", False, f"gguf {vals.shape} vs hf {ref.shape}")
            continue
        exact = np.array_equal(vals, ref)
        g = vals.reshape(vals.shape[0], -1, 128)
        has_pm1 = (np.abs(g) == np.repeat(d, 128, axis=1).reshape(g.shape)).any(-1)
        dpos = d.reshape(-1) > 0
        amax_ok = bool(has_pm1.reshape(-1)[dpos].all()) if dpos.any() else True
        check(f"{gname}: PTQ1_0", is_ptq, str(t.tensor_type))
        check(f"{gname}: values == export (RTN-lossless)", exact,
              f"maxdiff={np.abs(vals - ref).max():.2e}")
        check(f"{gname}: scale==amax", amax_ok)

    print("C5: islands stay full-precision")
    bad = []
    for t in r.tensors:
        if "norm" in t.name and not str(t.tensor_type).endswith(("F32", "F16", "BF16")):
            bad.append((t.name, str(t.tensor_type)))
    check("norm tensors F32/F16/BF16", not bad, str(bad[:3]))

    print("\nOVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
