#!/usr/bin/env python
"""Build tritfold-m12-stage1.ipynb (M12 Stage 1: 8B base QAT, 5000 steps).

Config locked by Stage 0 (findings-17): Z bf16 + PagedAdam8bit + gradient
checkpointing, CHUNK=4. Corpus: wiki-only (the v0.1-equivalent base bake) from
Qwen3-8B FP as self-teacher, top-50 KL, 40k windows (20.5M tok) cached once to
Drive (5-min reload on resume). Disconnect-resilient: cache on Drive, model-only
snapshots rotating on Drive (cold-optimizer bootstrap — proven since v0.2),
local full ckpt every 500.

Budget: cache ~1h + 5000 x ~10s ~= 15h total (~90 units). Gates: ppl best
vs FP-8B reference (target <=1.2x per M12 criteria), skipped==0 always.
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
def md(src): return {"cell_type": MD, "metadata": {}, "source": src}
def code(src): return {"cell_type": CODE, "metadata": {}, "execution_count": None, "outputs": [], "source": src}

MATH_SRC = Path("generators/gen_m12_stage0.py").read_text()
def extract(marker_start, marker_end):
    i = MATH_SRC.index(marker_start)
    j = MATH_SRC.index(marker_end, i)
    return MATH_SRC[i:j]

MATH_CELL = extract("BLOCK = 1024", 'print("math OK")') + 'print("math OK")'
STE_CELL = extract("# ===== STE", 'print("STE OK")') + 'print("STE OK")'

cells = []
cells.append(md("""# M12 Stage 1：8B 基础 QAT（5000 步）

**配置**（Stage 0 定案，findings-17）：Z bf16 + PagedAdam8bit + 梯度检查点 + CHUNK=4
**语料**：纯 wiki 40k 窗（20.5M tok，v0.1 等价底子烘烤）；**教师**：Qwen3-8B FP 自身（top-50 KL）
**断连韧性**：缓存落 Drive（重连 5 分钟重载）；模型快照 Drive 轮换（冷优化器自举——v0.2 起成熟模式）；本地全量 ckpt 每 500 步
**预算**：缓存 ~1h + 训练 ~14h ≈ **~15h / ~90 单位**（断连重连即续，缓存不重算）
**门**：ppl best ≤ 1.2× FP-8B（M12 判据的分布维）"""))

cells.append(code("""import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
%pip -q install --force-reinstall --no-deps "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
%pip -q install "datasets==5.0.1" "accelerate==1.14.0" sentencepiece protobuf bitsandbytes
from google.colab import drive
drive.mount("/content/drive")
import torch, transformers
print(torch.__version__, transformers.__version__)"""))

cells.append(code("import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n" + MATH_CELL))

cells.append(code(STE_CELL))

cells.append(code("""# ===== 配置 =====
MODEL = "Qwen/Qwen3-8B"
STEPS, BATCH, SEQ, TOPK, CHUNK = 5000, 8, 512, 50, 4   # CHUNK=4（OOM 则降 2 重启）
LRS = [2e-4, 1e-3, 3e-4]
EVAL_EVERY, EVAL_WINDOWS = 100, 40
DRIVE_DIR = "/content/drive/MyDrive/m12s1"
CKPT_LOCAL = "/content/m12s1_last.pt"          # 本地全量（含优化器），每 500 步覆盖
CKPT_DRIVE_BEST = f"{DRIVE_DIR}/m12s1_best.pt" # Drive 模型快照（best ppl）
CKPT_DRIVE_ROT = f"{DRIVE_DIR}/m12s1_rot.pt"   # Drive 模型快照（每 1000 步轮换）
CACHE_TV = f"{DRIVE_DIR}/cache_tv.pt"
CACHE_TI = f"{DRIVE_DIR}/cache_ti.pt"
os.makedirs(DRIVE_DIR, exist_ok=True)
import torch
Z_DTYPE = torch.bfloat16
KEEP_CODES0 = False
print("config OK")"""))

cells.append(code("""# ===== 数据：wiki-103 40k 窗 + wt2 测试窗 =====
import torch, gc
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
text103 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"] if t.strip())
ids = tok(text103, return_tensors="pt").input_ids[0]
N_WIN = STEPS * BATCH
del text103; gc.collect()
wins = [ids[i*SEQ:(i+1)*SEQ] for i in range(min(N_WIN, ids.numel()//SEQ))]
del ids; gc.collect()
text2 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"] if t.strip())
test_ids = tok(text2, return_tensors="pt").input_ids[0]
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_DATASETS_OFFLINE"] = "1"
print(f"windows {len(wins)} | test {test_ids.numel()//SEQ} windows")"""))

cells.append(code("""# ===== 自教师缓存（首跑 ~1h 落 Drive；重连 5 分钟重载）→ install → 优化器 =====
import time, gc, math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

DEV = "cuda"
need_cache = not (os.path.exists(CACHE_TV) and os.path.exists(CACHE_TI))
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(DEV)
if need_cache:
    t0 = time.time()
    tv = torch.empty(len(wins), SEQ - 1, TOPK, dtype=torch.float16, pin_memory=True)
    ti = torch.empty(len(wins), SEQ - 1, TOPK, dtype=torch.int32, pin_memory=True)
    with torch.no_grad():
        for c0 in range(0, len(wins), 8):
            x = torch.stack(wins[c0:c0+8]).to(DEV)
            lg = model(x).logits[:, :-1].float()
            v, i = torch.topk(lg, TOPK, dim=-1)
            tv[c0:c0+len(v)] = v.half().cpu(); ti[c0:c0+len(i)] = i.cpu()
            del lg, v, i
            if (c0 // 8) % 200 == 0:
                print(f"  cache {c0+8}/{len(wins)} ({(time.time()-t0)/60:.0f}min)", flush=True)
    torch.save(tv, CACHE_TV); torch.save(ti, CACHE_TI)
    print(f"cache done {(time.time()-t0)/60:.0f}min, saved to Drive", flush=True)
else:
    tv = torch.load(CACHE_TV); ti = torch.load(CACHE_TI)
    print(f"cache reloaded from Drive: {len(tv)} windows", flush=True)

# FP 参照 ppl（wt2 测试窗，bf16 口径）
@torch.no_grad()
def ppl_of(m, n_windows=EVAL_WINDOWS, seq=SEQ):
    m.eval(); nll = cnt = skipped = 0
    V = m.config.vocab_size
    for w in range(n_windows):
        x = test_ids[w*seq:(w+1)*seq].unsqueeze(0).to(DEV)
        lg = m(x).logits.float()
        v = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction="sum")
        if torch.isfinite(v): nll += v.item(); cnt += x[:, 1:].numel()
        else: skipped += 1
    m.train()
    return (math.exp(nll/cnt) if cnt else float("nan")), skipped

fp_ppl, _ = ppl_of(model)
print(f"FP-8B reference ppl = {fp_ppl:.2f}", flush=True)

# ---- install（bf16-Z）+ 检查点 + 优化器（Stage 0 定案配方）----
emb, linears, tr = install(model)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
import bitsandbytes as bnb
opt = bnb.optim.PagedAdam8bit([{"params": tr["Z"], "lr": LRS[0]},
                               {"params": tr["theta"], "lr": LRS[1]},
                               {"params": tr["island"], "lr": LRS[2]}], betas=(0.9, 0.95))
gc.collect(); torch.cuda.empty_cache()
print(f"installed {len(linears)} linears | ckpt ON | VRAM {torch.cuda.memory_allocated()/2**30:.1f}GB", flush=True)

# ---- 断连续训：Drive 模型快照（冷优化器自举）----
_start, hist = 1, []
if os.path.exists(CKPT_DRIVE_BEST):
    ck = torch.load(CKPT_DRIVE_BEST, map_location="cpu")
    if ck.get("step", 0) < STEPS:
        with torch.no_grad():
            for m_, Z_, th_ in zip(linears, ck["Z"], ck["theta"]):
                m_.Z.copy_(Z_.to(DEV, dtype=torch.bfloat16)); m_.theta.copy_(th_)
            emb.theta.copy_(ck["emb_theta"]); emb.codes.copy_(ck["emb_codes"])
            for n_, m_i in model.named_modules():
                if isinstance(m_i, Fp32RMSNorm) and n_ in ck["islands"]:
                    m_i.weight.copy_(ck["islands"][n_])
        _start = ck["step"] + 1; hist = ck.get("hist", [])
        print(f"resumed from Drive snapshot: step {ck['step']} (cold optimizer)", flush=True)
    else:
        _start = STEPS + 1
        print(f"snapshot complete ({ck['step']}) — training skipped", flush=True)"""))

cells.append(code("""# ===== 训练循环（5000 步；快照轮换；断连韧性）=====
import time, shutil, math
import torch.nn.functional as F

import shutil as _shutil, time as _time
def snapshot(path, step_i, ppl_i):
    # 容错快照：本地中转 → 拷贝（重试×2）→ 任何失败只告警不杀训练
    def _state():
        return {"step": step_i, "ppl": ppl_i, "hist": hist,
                "Z": [m.Z.detach().cpu() for m in linears],
                "theta": [m.theta.detach().cpu() for m in linears],
                "emb_theta": emb.theta.detach().cpu(), "emb_codes": emb.codes.cpu(),
                "islands": {n: m.weight.detach().cpu() for n, m in model.named_modules()
                            if isinstance(m, Fp32RMSNorm)}}
    try:
        if path.startswith("/content/drive"):
            tmp = "/content/m12s1_snap_tmp.pt"
            torch.save(_state(), tmp)
            for attempt in range(2):
                try:
                    _shutil.copy(tmp, path); break
                except Exception as e:
                    print(f"!! Drive 拷贝重试 {attempt+1}: {type(e).__name__}", flush=True)
                    _time.sleep(20)
            else:
                print(f"!! snapshot {path} 两次拷贝均失败——保留本地 tmp，训练继续", flush=True)
                return
            os.remove(tmp)
        else:
            torch.save(_state(), path)
    except Exception as e:
        print(f"!! snapshot {path} 失败（{type(e).__name__}）——训练继续", flush=True)

model.train()
_, cur_ppl = (None, ppl_of(model)[0]) if _start == 1 else (None, float("nan"))
best = cur_ppl if cur_ppl == cur_ppl else 1e9
last_local = [0]
t0 = time.time()
for step in range(_start, STEPS + 1):
    j = step - 1
    xs = [wins[(j*BATCH+k) % len(wins)] for k in range(BATCH)]
    opt.zero_grad(set_to_none=True)
    loss_val = 0.0
    for c in range(0, BATCH, CHUNK):
        n = min(CHUNK, BATCH - c)
        x = torch.stack(xs[c:c+n]).to(DEV)
        wi = [(j*BATCH+c+k) % len(wins) for k in range(n)]
        tv_c = torch.stack([tv[k] for k in wi]).to(DEV).float()
        ti_c = torch.stack([ti[k] for k in wi]).to(DEV).long()
        logits = model(x).logits[:, :-1].float()
        s_logp = F.log_softmax(logits, -1)
        loss_c = -(F.softmax(tv_c, -1) * torch.gather(s_logp, -1, ti_c)).sum(-1).mean()
        (loss_c * n / BATCH).backward()
        loss_val += loss_c.item() * n / BATCH
        del logits, s_logp, loss_c
    torch.nn.utils.clip_grad_norm_(
        [p for p in tr["Z"] + tr["theta"] + tr["island"] if p.grad is not None], 1.0)
    opt.step()
    if step % 20 == 0:
        print(f">> [m12s1] step {step} loss {loss_val:.3f} ({step*BATCH*SEQ/(time.time()-t0):.0f} tok/s)", flush=True)
    if step % 500 == 0:
        snapshot(CKPT_LOCAL, step, None)
        last_local[0] = step
    if step % EVAL_EVERY == 0 or step == STEPS:
        ppl, skipped = ppl_of(model)
        hist.append((step, ppl))
        print(f">> [m12s1] step {step} ppl={ppl:.2f} (best {best:.2f}) skip={skipped}", flush=True)
        if math.isfinite(ppl) and skipped == 0 and ppl < best - 0.01:
            best = ppl
            snapshot(CKPT_DRIVE_BEST, step, ppl)
print(f"DONE: best ppl {best:.2f} = {best/fp_ppl:.3f}x FP-8B (门 1.2x)")
print("history:", hist)"""))

cells.append(code("""# ===== 收官读数 =====
print(f"FP-8B 参照 ppl: {fp_ppl:.2f}")
print(f"三值 8B best ppl: {best:.2f} ({best/fp_ppl:.3f}x)")
print()
print("判定：")
print("  <=1.20x  -> 分布门过，进 Stage 2（对症广度矩阵 stacked）")
print("  1.20-1.35x -> 检查 CHENK/步数；参照官方 1.335x 属知识优先角的预期代价")
print("  >1.35x  -> bf16-Z 累积漂移嫌疑（Stage0 残留风险），需排查")
print(f"\\nhistory 完整曲线：{hist}")"""))

out = Path("notebooks/tritfold-m12-stage1.ipynb")
nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out}")
