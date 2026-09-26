#!/usr/bin/env python
"""L0' T1.1/T1.1b: full decode of prism.hadamard.* metadata from a PQ2_0 GGUF.

Usage: dump_metadata.py <gguf_path> [out_json]
"""
import json
import sys
from collections import Counter

import gguf


def field_value(f):
    v = f.contents() if hasattr(f, "contents") else None
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    if isinstance(v, list):
        return [x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else x for x in v]
    return v


def main():
    path = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else None
    r = gguf.GGUFReader(path)

    print(f"== {path}")
    print(f"fields: {len(r.fields)}, tensors: {len(r.tensors)}\n")

    meta = {}
    for k, f in r.fields.items():
        meta[k] = field_value(f)

    # general architecture block
    print("-- general.* --")
    for k in sorted(meta):
        if k.startswith("general.") or k.endswith((".embedding_length", ".block_count",
                                                 ".feed_forward_length", ".attention.head_count",
                                                 ".attention.head_count_kv", ".architecture")):
            print(f"  {k} = {meta[k]!r}")

    print("\n-- prism.hadamard.* --")
    had = {}
    for k in sorted(meta):
        if k.startswith("prism."):
            v = meta[k]
            had[k] = v
            if isinstance(v, list):
                print(f"  {k}: list[{len(v)}]")
            else:
                print(f"  {k} = {v!r}")

    wnames = had.get("prism.hadamard.weight_names", []) or []
    inames = had.get("prism.hadamard.inverse_weight_names", []) or []
    swidths = had.get("prism.hadamard.sign_widths", []) or []
    svalues = had.get("prism.hadamard.sign_values", []) or []

    print(f"\nweight_names ({len(wnames)}):")
    for n in wnames[:12]:
        print(f"   {n}")
    if len(wnames) > 12:
        print(f"   ... ({len(wnames) - 12} more)")
    print(f"inverse_weight_names ({len(inames)}): {inames}")
    print(f"sign_widths: {swidths}")
    print(f"sign_values: type={type(svalues).__name__} len={len(svalues)}")

    # tensor inventory and T1.1b: folded-weight input widths vs sign_widths
    # NB: ReaderTensor.shape is raw ggml ne order (ne0 first) -> input width = shape[0]
    tinfo = {t.name: (tuple(int(x) for x in t.shape), str(t.tensor_type), t.n_bytes)
             for t in r.tensors}
    width_counter = Counter()
    missing = []
    for n in wnames:
        if n in tinfo:
            shape, _, _ = tinfo[n]
            width_counter[shape[0]] += 1   # ggml ne0 = input width (axis=-1 fold)
        else:
            missing.append(n)
    print("\n-- T1.1b: folded-weight input widths (ggml ne0 = shape[0]) --")
    for w, c in sorted(width_counter.items()):
        mark = " <-- has sign table" if w in set(swidths) else ""
        print(f"  width {w}: {c} tensor(s){mark}")
    if missing:
        print(f"  folded names not found in tensor list: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    print("\n-- dtype census of folded tensors --")
    tc = Counter(tinfo[n][1] for n in wnames if n in tinfo)
    for ty, c in tc.items():
        print(f"  {ty}: {c}")

    # non-folded large tensors (potential FP islands)
    print("\n-- sample of non-folded tensors (first 30) --")
    wset = set(wnames)
    shown = 0
    for name, (shape, ty, nb) in tinfo.items():
        if name not in wset and shown < 30:
            print(f"  {name}  shape={shape} type={ty} bytes={nb}")
            shown += 1

    if out_json:
        with open(out_json, "w") as f:
            json.dump({"metadata": {k: v for k, v in meta.items() if not isinstance(v, list) or len(str(v)) < 100000},
                       "weight_names": wnames, "inverse_weight_names": inames,
                       "tensor_widths": {str(w): c for w, c in width_counter.items()}},
                      f, indent=1, ensure_ascii=False)
        print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
