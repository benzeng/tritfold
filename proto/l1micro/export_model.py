#!/usr/bin/env python
"""M3 Phase 3: snap + export trained ternary model to a HF-layout dir with
hadamard_packing.json, ready for fork convert_hf_to_gguf.py.

- realized weights = s_g * T in the rotated basis, written in training order
  (same HF names/shapes as the original checkpoint), fp16
- Phase-3 amax snap: s_g := group amax of realized values (no-op for groups
  containing +/-1; all-zero groups -> s=0) so RTN repacking is lossless (C3)
- tied contract: write BOTH model.embed_tokens.weight (inverse-after-lookup)
  and lm_head.weight (fold-before-matmul) with the same folded table;
  config tie_word_embeddings=false so conversion keeps both tensors
- signs in the manifest are regenerated from the same deterministic seeds used
  at training time (signs_for_width)

Usage: export_model.py --ckpt <WORK_DIR>/l1micro_main.pt \
         --out <WORK_DIR>/qwen3-0.6b-ternary-hd
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common.fwht_torch import BLOCK, GROUP, signs_for_width
from ternary_model import RotTrainEmbedding, RotTrainLinear, install_ternary
from train import restore_checkpoint

MODEL = "<MODELS_DIR>/Qwen3-0.6B"

KEEP_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "generation_config.json", "special_tokens_map.json", "added_tokens.json",
              "chat_template.jinja"]


def hf_name(module_path: str) -> str:
    # named_modules paths on AutoModelForCausalLM already equal HF state-dict
    # names minus the ".weight" suffix (model.layers.N..., lm_head)
    return module_path + ".weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="<WORK_DIR>/qwen3-0.6b-ternary-hd")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    ck = torch.load(args.ckpt, map_location="cpu")
    print(f"checkpoint step={ck['step']} init={ck['args']['init']}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    ternary = install_ternary(model, "amax")   # init values overwritten by restore
    restore_checkpoint(model, ternary, ck)

    # collect realized folded weights by HF name
    folded = {}
    emb_table = None
    for path, m in model.named_modules():
        if isinstance(m, RotTrainLinear):
            folded[hf_name(path)] = m.tw
        elif isinstance(m, RotTrainEmbedding):
            emb_table = m.tw
    assert emb_table is not None

    from safetensors.torch import load_file, save_file
    src = load_file(str(Path(MODEL) / "model.safetensors"))
    out = {}
    # island keys are module paths (model.layers.0.input_layernorm);
    # HF tensor names append ".weight"
    trained_islands = {k + ".weight": v for k, v in ck["islands"].items()}

    for name, t in src.items():
        if name in folded:
            tw = folded.pop(name)
            s = tw.scales()                                  # (out, ng)
            w = s.repeat_interleave(GROUP, dim=1) * tw.codes.float()
            # Phase-3 amax snap: stored values become exactly +/- group amax / 0
            # (no-op for groups containing +/-1; keeps all-zero groups at 0)
            wg = w.reshape(w.shape[0], -1, GROUP)
            amax = wg.abs().amax(dim=-1, keepdim=True)
            w = (wg.sign() * amax).reshape(w.shape)
            out[name] = w.to(torch.float16)
        elif name in trained_islands:
            out[name] = trained_islands[name].to(torch.float16)
        else:
            out[name] = t
    assert not folded, f"unmatched folded tensors: {list(folded)}"

    # tied contract: materialize both embedding and lm_head with the folded table
    tw = emb_table
    s = tw.scales()
    w = s.repeat_interleave(GROUP, dim=1) * tw.codes.float()
    wg = w.reshape(w.shape[0], -1, GROUP)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    w = (wg.sign() * amax).reshape(w.shape).to(torch.float16)
    out["model.embed_tokens.weight"] = w
    out["lm_head.weight"] = w.clone()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in KEEP_FILES:
        p = Path(MODEL) / f
        if p.exists():
            shutil.copy(p, outdir / f)
    cfg = json.loads((Path(MODEL) / "config.json").read_text())
    cfg["tie_word_embeddings"] = False
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))
    save_file(out, str(outdir / "model.safetensors"), metadata={"format": "pt"})

    # hadamard_packing.json (schema validated by conversion/base.py:635-760)
    widths = sorted({1024, 2048, 3072})
    signs = {str(w): signs_for_width(w, "cpu").to(torch.int8).tolist() for w in widths}
    records = []
    for i in range(cfg["num_hidden_layers"]):
        for sub in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            records.append({"name": f"model.layers.{i}.{sub}.weight",
                            "axis": -1, "role": "fold-before-matmul"})
    records.append({"name": "lm_head.weight", "axis": -1, "role": "fold-before-matmul"})
    records.append({"name": "model.embed_tokens.weight", "axis": -1,
                    "role": "inverse-after-lookup"})
    manifest = {
        "schema_version": 1,
        "kind": "hadamard-weight-fold",
        "status": "requires-matching-runtime",
        "transform": {"name": "normalized-signed-sylvester-walsh-hadamard",
                      "block_size": BLOCK, "sign_mode": "explicit"},
        "signs": signs,
        "tensors": records,
    }
    (outdir / "hadamard_packing.json").write_text(json.dumps(manifest))
    n_params = sum(t.numel() for t in out.values())
    print(f"exported {len(out)} tensors ({n_params/1e9:.2f}B) + manifest -> {outdir}")


if __name__ == "__main__":
    main()
