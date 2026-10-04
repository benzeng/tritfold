#!/usr/bin/env python
"""Dry-run validator for Colab notebooks on GTX 1060 (0.6B model, small params).

Catches NameError, IndentationError, ImportError, shape mismatches BEFORE
pushing to Colab A100. All Colab-specific deps are mocked/injected.

Usage: python proto/common/dryrun_notebook.py <notebook.ipynb> [--arm 1b]
"""
import argparse
import json
import os
import sys
import time
import traceback
import types

import torch


def _make_mock_colab(drive_dir):
    """Create mock google.colab module + helpers, return injection dict."""
    os.makedirs(drive_dir, exist_ok=True)
    os.makedirs("/tmp/bonsai_dryrun", exist_ok=True)

    def mock_drive_mount(path):
        os.makedirs(path, exist_ok=True)
        print(f"  [mock] drive.mount({path})")

    class _files:
        @staticmethod
        def download(path):
            print(f"  [mock] files.download({path})")

    colab = types.ModuleType("google.colab")
    colab.files = _files
    drive = types.ModuleType("google.colab.drive")
    drive.mount = mock_drive_mount
    colab.drive = drive
    # transformers 检测 Colab 时需要 __spec__
    import importlib.machinery
    colab.__spec__ = importlib.machinery.ModuleSpec("google.colab", None)
    drive.__spec__ = importlib.machinery.ModuleSpec("google.colab.drive", None)
    sys.modules["google.colab"] = colab
    sys.modules["google.colab.drive"] = drive

    # Patch huggingface_hub at module level
    import huggingface_hub as _hb
    _real_sd, _real_hd = _hb.snapshot_download, _hb.hf_hub_download

    def _snap(*a, **kw):
        p = str(a[0] if a else kw.get("repo_id", ""))
        if os.path.isdir(p):
            return p
        try:
            return _real_sd(*a, **kw)
        except Exception:
            return p

    def _dl(*a, **kw):
        try:
            return _real_hd(*a, **kw)
        except Exception:
            return f"/tmp/bonsai_dryrun/mock_gguf.bin"

    _hb.snapshot_download = _snap
    _hb.hf_hub_download = _dl

    return {"mock_drive_mount": mock_drive_mount, "torch": torch, "types": types}


# 通用替换（每个 cell 都尝试）
def _transform(code, overrides, skip_training):
    """Apply dry-run transformations to a cell's code."""
    # 1. Config 覆盖
    for var, val in overrides.items():
        for prefix in (f"{var} =", f"{var}="):
            if f"\n{prefix}" in f"\n{code}" or code.startswith(prefix):
                lines = code.splitlines()
                for i, l in enumerate(lines):
                    if l.strip().startswith(prefix):
                        lines[i] = f'{var} = "{val}"' if isinstance(val, str) else f"{var} = {val}"
                        break
                code = "\n".join(lines)

    # 2. GTX 1060 适配
    code = code.replace(
        "assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()",
        "assert torch.cuda.is_available()")
    code = code.replace(
        'DEV, BF = "cuda", torch.bfloat16',
        'DEV, BF = "cuda", torch.float16')
    code = code.replace(
        "import bitsandbytes as bnb",
        "bnb = types.SimpleNamespace(optim=types.SimpleNamespace(Adam8bit=torch.optim.Adam))")

    # 3. Colab mock
    if "drive.mount(" in code and "mock_drive_mount" not in code:
        code = code.replace("drive.mount(", "mock_drive_mount(")
    if "from google.colab import drive" in code:
        code = code.replace("from google.colab import drive", "pass  # mock drive")

    # 4. 限制循环
    loop_limits = [
        ("for ex in sciq_val:", "for ex in list(sciq_val)[:5]:"),
        ("for ex in arc:", "for ex in list(arc)[:5]:"),
        ("for ex in data:", "for ex in list(data)[:5]:"),
        ("for w in range(n_windows):", "for w in range(min(n_windows, 2)):"),
        ("for ex in wiki:", "for ex in list(wiki)[:100]:"),
        ("for i, ex in enumerate(wiki):", "for i, ex in enumerate(list(wiki)[:100]):"),
        ("for ex in sciq_val_all:", "for ex in list(sciq_val_all)[:100]:"),
        ("for i, ex in enumerate(sciq_val_all):", "for i, ex in enumerate(list(sciq_val_all)[:100]):"),
        ("for ex in sciq_tr:", "for ex in list(sciq_tr)[:50]:"),
        ("for ex in sciq:", "for ex in list(sciq)[:50]:"),
        ("for ex in uc:", "for ex in list(uc)[:20]:"),
        ("for ex in belle:", "for ex in list(belle)[:20]:"),
        ("for ex in k_stream:", "for ex in list(k_stream)[:20]:"),  # k_stream is iterable
        ("for w in range(EVAL_WINDOWS):", "for w in range(2):"),
    ]
    for old, new in loop_limits:
        code = code.replace(old, new)

    # 5. 限制数据量
    code = code.replace("16_777_216", "50000")
    code = code.replace("33_554_432", "50000")
    for var, lim in [("N_FWED_DOCS", 20), ("N_CONV_U", 20), ("N_BELLE", 20),
                     ("N_DOCS_K", 20), ("N_ULTRA_CONV", 20)]:
        code = code.replace(f"if i + 1 >= {var}:", f"if i + 1 >= {lim}:")
        code = code.replace(f"if n_b >= {var}:", f"if n_b >= {lim}:")
        code = code.replace(f"if i >= {var}:", f"if i >= {lim}:")

    # 6. 训练步数限制
    if skip_training:
        code = code.replace(
            "for step in range(_start, STEPS + 1):",
            "for step in range(_start, min(_start + 2, STEPS + 1)):")
        code = code.replace(
            "for step in range(1, STEPS + 1):",
            "for step in range(1, min(3, STEPS + 1)):")

    # 7. 跳过导出 cell（需要完整模型）
    if "save_file" in code and "hadamard_packing" in code:
        return None

    return code


def dryrun(notebook_path, arm="1b", skip_training=False):
    nb = json.load(open(notebook_path))
    cells = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    print(f"= dryrun: {notebook_path} ({len(cells)} code cells, ARM={arm})\n")

    g = _make_mock_colab("/tmp/bonsai_dryrun_drive")

    overrides = {
        "MODEL": "/home/dong/models/Qwen3-0.6B",
        "TEACHER_MODEL": "/home/dong/models/Qwen3-0.6B",
        "STEPS": 10,
        "EVAL_EVERY": 5,
        "EVAL_WINDOWS": 4,
        "DRIVE_EVERY": 5,
        "N_ULTRA_CONV": 20,
        "N_FWED_DOCS": 20,
        "N_CONV_U": 20,
        "N_BELLE": 20,
        "N_DOCS_K": 20,
        "ARM": arm,
        "DRIVE_DIR": "/tmp/bonsai_dryrun_drive",
        "CKPT": "/tmp/bonsai_dryrun_ckpt.pt",
        "BOOT_REPO": "benzeng/tritfold-0.6b-ptq1_0",
        "BOOT_FILE": "tritfold-0.6b-ptq1_0.gguf",
        "BOOT_STEP": 3000,
        "BOOT_PPL": 48.08,
    }

    passed, failed = 0, 0
    for idx, cell in cells:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        lines = src.splitlines()
        code_lines = [l for l in lines if not l.startswith(("!", "%"))]
        code = "\n".join(code_lines)

        code = _transform(code, overrides, skip_training)
        if code is None:
            print(f"  cell {idx}: [SKIP] export")
            passed += 1
            continue

        desc = lines[0][:40] if lines else "(empty)"
        print(f"  cell {idx}: {desc}...", end=" ", flush=True)
        t0 = time.time()
        try:
            exec(compile(code, f"<cell_{idx}>", "exec"), g)
            print(f"OK ({time.time()-t0:.1f}s)")
            passed += 1
        except Exception as e:
            print(f"FAIL ({time.time()-t0:.1f}s)")
            print(f"    {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            failed += 1

    print(f"\n= dryrun: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("--arm", default="1b")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    sys.exit(0 if dryrun(args.notebook, args.arm, args.skip_training) else 1)
