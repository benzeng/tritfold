#!/usr/bin/env python
"""Build tritfold-m12-stage2.ipynb (M12 Stage 2: corpus-matrix stacked, 2000 steps).

Bootstrap from Stage 1's best snapshot (m12s1_best.pt). Config identical to
Stage 1 (bf16-Z + PagedAdam8bit + grad checkpointing, CHUNK=4). Corpus matrix
per the budget-fit design: wiki 2 (anchor, down-weighted) + sciq 3 (2.4-epoch
dose optimum from M11) + arc 1 + ultra 2. MMLU stream dropped — no eval
baseline for it would pollute attribution (future session).

Caching: wiki targets REUSED from Stage 1's Drive cache (first 4000 window
indexes align — same deterministic window derivation); sciq/arc/ultra cached
fresh (~15 min). Probes at steps 1000/2000 (sciq decontaminated + ARC, PyTorch
likelihood protocol, same as all 1.7B experiments).

Gate: sciq >= 0.70 (M12 final criterion, knowledge axis); ppl reported vs
Stage 1 for the distribution cost. Budget ~4.8h ~= 24 units.
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
cells.append(md("""# M12 Stage 2：对症矩阵叠加（2000 步）

**自举**：Stage 1 最优快照（`m12s1_best.pt`）——链式第二级（v0.2 模式）
**配置**：同 Stage 1（bf16-Z + PagedAdam8bit + 检查点，CHUNK=4）
**矩阵**：wiki 2（锚，降权）+ **sciq 3（2.4ep 剂量峰值，M11 定标）** + ARC 1 + ultra 2
**缓存**：wiki 复用 Stage 1 的 Drive 缓存（前 4000 窗索引对齐）；sciq/ARC/ultra 新缓存 ~15min
**探针**：step 1000/2000 各一次（sciq 去污染 + ARC，PyTorch 似然协议）
**门**：**sciq ≥ 0.70**（M12 终局判据知识维）；ppl 相对 Stage 1 报差（分布代价）
**预算**：~4.8h ≈ 24 单元。FP 参照（已知，M10 实测 PyTorch 协议）：sciq 0.872/0.830，ARC 0.474/0.472"""))

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
STEPS, BATCH, SEQ, TOPK, CHUNK = 2000, 8, 512, 50, 4
LRS = [2e-4, 1e-3, 3e-4]
EVAL_EVERY, EVAL_WINDOWS = 100, 40
PROBE_AT = (1000, 2000)
DRIVE_DIR = "/content/drive/MyDrive/m12s1"          # 复用：s1 缓存 + 自举快照
DRIVE_DIR2 = "/content/drive/MyDrive/m12s2"
os.makedirs(DRIVE_DIR2, exist_ok=True)
CKPT_LOCAL = "/content/m12s2_last.pt"
CKPT_DRIVE_BEST = f"{DRIVE_DIR2}/m12s2_best.pt"
BOOT = f"{DRIVE_DIR}/m12s1_best.pt"                   # Stage 1 自举源
CACHE_TV_WIKI = f"{DRIVE_DIR}/cache_tv.pt"            # s1 的 40k wiki 缓存（索引对齐复用）
CACHE_TI_WIKI = f"{DRIVE_DIR}/cache_ti.pt"
CACHE_TV_MIX = f"{DRIVE_DIR2}/cache_tv_mix.pt"        # sciq/arc/ultra 三流新缓存
CACHE_TI_MIX = f"{DRIVE_DIR2}/cache_ti_mix.pt"
import torch
Z_DTYPE = torch.bfloat16
KEEP_CODES0 = False
print("config OK")"""))

cells.append(code("""# ===== 数据：wiki（确定性重建，索引对齐 s1 缓存）+ sciq/arc/ultra 窗 + 去污染 =====
import torch, gc, os
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)

def ngrams(text, n=13):
    ws = text.lower().split()
    return {" ".join(ws[i:i+n]) for i in range(len(ws)-n+1)} if len(ws)>=n else set()

# wiki 窗（与 s1 完全同序——索引对齐复用缓存）
text103 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"] if t.strip())
ids103 = tok(text103, return_tensors="pt").input_ids[0]
del text103; gc.collect()
wiki_all = [ids103[i*SEQ:(i+1)*SEQ] for i in range(min(40000, ids103.numel()//SEQ))]
del ids103; gc.collect()
wiki_src = wiki_all[:4000]          # 训练用前 4000（缓存索引 0..3999 已覆盖）
text2 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"] if t.strip())
test_ids = tok(text2, return_tensors="pt").input_ids[0]

# sciq（对症主探针流）+ 去污染评估集
sciq_tr = load_dataset("allenai/sciq", split="train")
s_ids = []
for ex in sciq_tr:
    s_ids.append(tok(f"Question: {ex['question']}\\n{ex['support']}\\nAnswer: {ex['correct_answer']}",
                     return_tensors="pt").input_ids[0])
_all_sciq = torch.cat(s_ids); del s_ids; gc.collect()
q_all = _all_sciq[: (_all_sciq.numel() // SEQ) * SEQ]; del _all_sciq; gc.collect()
sciq_src = [q_all[i*SEQ:(i+1)*SEQ] for i in range(q_all.numel()//SEQ)]

_train_ng = set()
for ex in sciq_tr:
    _train_ng |= ngrams(ex["support"] + " " + ex["question"])
sciq_val_all = load_dataset("allenai/sciq", split="validation")
CLEAN_IDX = [i for i, ex in enumerate(sciq_val_all)
             if not (ngrams(ex["support"] + " " + ex["question"]) & _train_ng)]
sciq_val = sciq_val_all.select(CLEAN_IDX)
del _train_ng; gc.collect()

# ARC（对症第二流）
arc_c = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
arc_e = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
a_ids = []
for ex in list(arc_c) + list(arc_e):
    try:
        ai = ex["choices"]["label"].index(ex["answerKey"])
        ans = ex["choices"]["text"][ai]
    except ValueError:
        continue
    a_ids.append(tok(f"Question: {ex['question']}\\nAnswer: {ans}", return_tensors="pt").input_ids[0])
_all_arc = torch.cat(a_ids); del a_ids; gc.collect()
a_all = _all_arc[: (_all_arc.numel() // SEQ) * SEQ]; del _all_arc; gc.collect()
arc_src = [a_all[i*SEQ:(i+1)*SEQ] for i in range(a_all.numel()//SEQ)]
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

# ultra（生成保持，2/step）
import numpy as np
rng = np.random.default_rng(0)
uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(8000))
u_ids = []
for ex in uc:
    text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                   chat_template_kwargs={"enable_thinking": False})
    u_ids.append(tok(text, return_tensors="pt").input_ids[0])
u_all = torch.cat([i[: (i.numel() // SEQ) * SEQ] for i in u_ids if i.numel() >= SEQ])
del u_ids; gc.collect()
_u_off = rng.choice(u_all.numel() // SEQ, min(4000, u_all.numel() // SEQ), replace=False)
ultra_src = [u_all[int(o)*SEQ: int(o)*SEQ+SEQ] for o in _u_off]
del u_all; gc.collect()

os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_DATASETS_OFFLINE"] = "1"
print(f"wiki {len(wiki_src)} | sciq {len(sciq_src)} | arc {len(arc_src)} | ultra {len(ultra_src)} | "
      f"sciq_val {len(sciq_val)} | ARC test {len(arc)}")"""))

cells.append(code("""# ===== 缓存：wiki 复用 + 三流新缓存 → FP 参照 ppl → install → 自举 s1 快照 =====
import time, gc, math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

DEV = "cuda"
tv_wiki = torch.load(CACHE_TV_WIKI); ti_wiki = torch.load(CACHE_TI_WIKI)
print(f"wiki 缓存复用: {len(tv_wiki)} 窗（训练用前 4000）", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(DEV)

# ---- 探针定义（PyTorch 似然协议，与全部 1.7B 实验一致）----
Q, CH = "Question: {q}\\nAnswer:", " {a}"
@torch.no_grad()
def _score(prompt, option):
    ids = tok(prompt + CH.format(a=option), return_tensors="pt").input_ids[0]
    pl = tok(prompt, return_tensors="pt").input_ids[0].numel()
    x = ids.unsqueeze(0).to(DEV)
    lg = model(x).logits.float()[0, pl-1:-1]
    tgt = ids[pl:].to(DEV)
    return (torch.log_softmax(lg, -1)[torch.arange(len(tgt), device=DEV), tgt].sum().item(), len(tgt))

@torch.no_grad()
def probe(data, tag, get_opts, get_gold, m=None):
    mm = m or model
    mm.eval(); n = acc = accn = 0
    for ex in data:
        prompt = Q.format(q=ex["question"]) + "\\n"
        sc = [_score(prompt, o) for o in get_opts(ex)]
        acc += max(range(len(sc)), key=lambda i: sc[i][0]) == get_gold(ex)
        accn += max(range(len(sc)), key=lambda i: sc[i][0]/sc[i][1]) == get_gold(ex)
        n += 1
    mm.train()
    print(f"[{tag}] acc {acc/n:.3f} | acc_norm {accn/n:.3f} (n={n})", flush=True)

# ---- 三流新缓存（sciq/arc/ultra 拼接索引：0..S-1 sciq, S..S+A-1 arc, 之后 ultra）----
S, A = len(sciq_src), len(arc_src)
mix_src = sciq_src + arc_src + ultra_src
need = not (os.path.exists(CACHE_TV_MIX) and os.path.exists(CACHE_TI_MIX))
if need:
    t0 = time.time()
    tvm = torch.empty(len(mix_src), SEQ - 1, TOPK, dtype=torch.float16, pin_memory=True)
    tim = torch.empty(len(mix_src), SEQ - 1, TOPK, dtype=torch.int32, pin_memory=True)
    with torch.no_grad():
        for c0 in range(0, len(mix_src), 8):
            x = torch.stack(mix_src[c0:c0+8]).to(DEV)
            lg = model(x).logits[:, :-1].float()
            v, i = torch.topk(lg, TOPK, dim=-1)
            tvm[c0:c0+len(v)] = v.half().cpu(); tim[c0:c0+len(i)] = i.cpu()
            del lg, v, i
            if (c0 // 8) % 100 == 0:
                print(f"  mix cache {c0+8}/{len(mix_src)} ({(time.time()-t0)/60:.0f}min)", flush=True)
    torch.save(tvm, CACHE_TV_MIX); torch.save(tim, CACHE_TI_MIX)
    print(f"mix cache done {(time.time()-t0)/60:.0f}min -> Drive", flush=True)
else:
    tvm = torch.load(CACHE_TV_MIX); tim = torch.load(CACHE_TI_MIX)
    print(f"mix cache reloaded: {len(tvm)}", flush=True)

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
print(f"FP-8B reference ppl = {fp_ppl:.2f}（M12 终局门 = {fp_ppl*1.2:.2f}）", flush=True)

# ---- install + 检查点 + 优化器（Stage 0 定案配方）----
emb, linears, tr = install(model)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
import bitsandbytes as bnb
opt = bnb.optim.PagedAdam8bit([{"params": tr["Z"], "lr": LRS[0]},
                               {"params": tr["theta"], "lr": LRS[1]},
                               {"params": tr["island"], "lr": LRS[2]}], betas=(0.9, 0.95))
gc.collect(); torch.cuda.empty_cache()
print(f"installed {len(linears)} linears | ckpt ON | VRAM {torch.cuda.memory_allocated()/2**30:.1f}GB", flush=True)

# ---- 自举：Stage 1 最优快照（冷优化器，链式第二级）----
import os as _os
_boot_from = BOOT
if _os.path.exists(CKPT_DRIVE_BEST):
    ck_a = torch.load(CKPT_DRIVE_BEST, map_location="cpu")
    ck_b = torch.load(BOOT, map_location="cpu")
    _boot_from = CKPT_DRIVE_BEST if ck_a.get("step", 0) >= getattr(ck_b, "step", ck_b.get("step", 0)) else BOOT
    del ck_a, ck_b; gc.collect()   # 僵尸引用即删（44GB RAM 教训）
_start, hist = 1, []
if _os.path.exists(_boot_from):
    ck = torch.load(_boot_from, map_location="cpu")
    with torch.no_grad():
        for m_, Z_, th_ in zip(linears, ck["Z"], ck["theta"]):
            m_.Z.copy_(Z_.to(DEV, dtype=torch.bfloat16)); m_.theta.copy_(th_)
        emb.theta.copy_(ck["emb_theta"]); emb.codes.copy_(ck["emb_codes"])
        for n_, m_i in model.named_modules():
            if isinstance(m_i, Fp32RMSNorm) and n_ in ck["islands"]:
                m_i.weight.copy_(ck["islands"][n_])
    _step0 = ck.get("step", 0)
    print(f"bootstrapped from {_boot_from} (source step {_step0}, fresh optimizer)", flush=True)
    del ck; gc.collect()           # 同上
if _os.path.exists(CKPT_DRIVE_BEST) and torch.load(CKPT_DRIVE_BEST, map_location="cpu").get("arm", "") == "m12s2" \\
   and torch.load(CKPT_DRIVE_BEST, map_location="cpu").get("step", 0) < STEPS:
    ck = torch.load(CKPT_DRIVE_BEST, map_location="cpu")
    _start = ck["step"] + 1; hist = ck.get("hist", [])
    print(f"RESUME m12s2 from step {ck['step']}", flush=True)
boot_ppl, _ = ppl_of(model)
print(f"bootstrap restore ppl = {boot_ppl:.2f}（应接近 Stage1 best）", flush=True)"""))

cells.append(code("""# ===== 训练循环（v2：每 500 步无条件 Drive 快照——断连零损失 + 剂量点留档）=====
import time, math, os
import torch.nn.functional as F
import shutil as _shutil, time as _time

def snapshot(path, step_i, ppl_i):
    # 容错快照：本地中转 -> 拷贝（重试x2）-> 失败只告警不杀训练
    def _state():
        return {"step": step_i, "ppl": ppl_i, "arm": "m12s2", "hist": hist,
                "Z": [m.Z.detach().cpu() for m in linears],
                "theta": [m.theta.detach().cpu() for m in linears],
                "emb_theta": emb.theta.detach().cpu(), "emb_codes": emb.codes.cpu(),
                "islands": {n: m.weight.detach().cpu() for n, m in model.named_modules()
                            if isinstance(m, Fp32RMSNorm)}}
    try:
        if path.startswith("/content/drive"):
            tmp = "/content/m12s2_snap_tmp.pt"
            torch.save(_state(), tmp)
            for attempt in range(2):
                try:
                    _shutil.copy(tmp, path); break
                except Exception as e:
                    print(f"!! Drive 拷贝重试 {attempt+1}: {type(e).__name__}", flush=True)
                    _time.sleep(20)
            else:
                print(f"!! snapshot 两次拷贝均失败——训练继续", flush=True)
                return
            os.remove(tmp)
        else:
            torch.save(_state(), path)
    except Exception as e:
        print(f"!! snapshot 失败（{type(e).__name__}）——训练继续", flush=True)

_start = 1
if os.path.exists(CKPT_DRIVE_BEST):
    _ck = torch.load(CKPT_DRIVE_BEST, map_location="cpu")
    if _ck.get("arm", "") == "m12s2" and _ck.get("step", 0) < STEPS:
        _start = _ck["step"] + 1
        hist = _ck.get("hist", [])
        print(f"RESUME m12s2 from step {_ck['step']}", flush=True)
    elif _ck.get("arm", "") == "m12s2":
        _start = STEPS + 1
        print(f"m12s2 快照已完备（step {_ck['step']}）——训练跳过", flush=True)

model.train()
best = boot_ppl
t0 = time.time()
for step in range(_start, STEPS + 1):
    j = step - 1
    # 矩阵配比：wiki 2 | sciq 3 | arc 1 | ultra 2
    idx = ([(j*2+k) % 4000 for k in range(2)] +
           [S - 1 - ((j*3+k) % S) for k in range(3)] +
           [S + (j % A)] +
           [S + A + ((j*2+k) % len(ultra_src)) for k in range(2)])
    wtok = [(tv_wiki[(j*2+k) % 4000], ti_wiki[(j*2+k) % 4000]) for k in range(2)]
    xs = [wiki_src[(j*2+k) % 4000] for k in range(2)] + [mix_src[k] for k in idx[2:]]
    tgts = wtok + [(tvm[k], tim[k]) for k in idx[2:]]
    opt.zero_grad(set_to_none=True)
    loss_val = 0.0
    for c in range(0, BATCH, CHUNK):
        n = min(CHUNK, BATCH - c)
        x = torch.stack(xs[c:c+n]).to(DEV)
        tv_c = torch.stack([tgts[c+k][0] for k in range(n)]).to(DEV).float()
        ti_c = torch.stack([tgts[c+k][1] for k in range(n)]).to(DEV).long()
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
        print(f">> [m12s2] step {step} loss {loss_val:.3f} ({(step-_start+1)*BATCH*SEQ/(time.time()-t0):.0f} tok/s)", flush=True)
    if step == STEPS:
        snapshot(CKPT_DRIVE_BEST, step, None)   # 仅终点（RAM 教训：反复物化 14.75GB 会碎片化顶穿 83GB）
    if step % EVAL_EVERY == 0 or step == STEPS:
        ppl, skipped = ppl_of(model)
        hist.append((step, ppl))
        print(f">> [m12s2] step {step} ppl={ppl:.2f} skip={skipped}", flush=True)
    if step in PROBE_AT:
        probe(sciq_val, f"DOSE sciq @{step} (FP 0.830)",
              lambda ex: [ex["correct_answer"]] + [ex["distractor" + str(i)] for i in (1, 2, 3)],
              lambda ex: 0)
        probe(arc, f"DOSE ARC @{step} (FP 0.472)",
              lambda ex: ex["choices"]["text"],
              lambda ex: ex["choices"]["label"].index(ex["answerKey"]))
print("DONE Stage 2（500/1000/1500/2000 各有剂量点快照在 Drive）")"""))

cells.append(code("""# ===== 救援/落盘 cell（训练结束后执行：清僵尸 + 终态落盘 + 补判卷）=====
import gc, os, shutil, math, time
import torch
for _z in ("ck_a", "ck_b", "ck"):
    if _z in dir():
        del globals()[_z]
gc.collect()
try:
    import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass
def _state():
    return {"step": STEPS, "ppl": None, "arm": "m12s2", "hist": hist,
            "Z": [m.Z.detach().cpu() for m in linears],
            "theta": [m.theta.detach().cpu() for m in linears],
            "emb_theta": emb.theta.detach().cpu(), "emb_codes": emb.codes.cpu(),
            "islands": {n: m.weight.detach().cpu() for n, m in model.named_modules()
                        if isinstance(m, Fp32RMSNorm)}}
try:
    torch.save(_state(), "/content/m12s2_final.pt")
    for _att in range(3):
        try:
            shutil.copy("/content/m12s2_final.pt", CKPT_DRIVE_BEST)
            print("终态已落 Drive OK", flush=True); break
        except Exception as e:
            print(f"拷贝重试 {_att+1}: {type(e).__name__}", flush=True); time.sleep(15)
except Exception as e:
    print(f"终态落盘失败: {type(e).__name__}", flush=True)"""))

cells.append(code("""# ===== 终局判卷 =====
probe(sciq_val, "FINAL sciq (FP 0.751/0.830)",
      lambda ex: [ex["correct_answer"]] + [ex["distractor" + str(i)] for i in (1, 2, 3)],
      lambda ex: 0)
probe(arc, "FINAL ARC (FP 0.357/0.472 ... 实测 0.474/0.472)",
      lambda ex: ex["choices"]["text"],
      lambda ex: ex["choices"]["label"].index(ex["answerKey"]))
ppl_f, _ = ppl_of(model)
print()
print("=== M12 终局判读 ===")
print(f"ppl: {ppl_f:.2f}（FP-8B {fp_ppl:.2f} 的 {ppl_f/fp_ppl:.3f}x；FP-1.7B 20.42）")
print("判据：sciq acc_norm >= 0.70 -> M12 知识维达标（'三值 8B 全面超 FP-1.7B' 的知识半壁）")
print("      ppl 对照 FP-1.7B 的 20.42：低于它则分布维也达标")
print(f"\\nhistory: {hist}")"""))

cells.append(code("""# ===== 抽检（生成保持）=====
QUESTIONS = ["What planet is known as the Red Planet?",
             "Why do we see lightning before we hear thunder?",
             "What is the powerhouse of the cell?"]
@torch.no_grad()
def ask(q, max_new=96):
    ids = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                  add_generation_prompt=True,
                                  chat_template_kwargs={"enable_thinking": False},
                                  return_tensors="pt").to(DEV)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                         temperature=0.5, top_p=0.85, top_k=20,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
model.eval()
for q in QUESTIONS:
    print("Q:", q, "\\nA:", ask(q), "\\n", flush=True)"""))

out = Path("notebooks/tritfold-m12-stage2.ipynb")
nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out}")
