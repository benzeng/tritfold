#!/usr/bin/env python
"""Build Bonsai/colab/bonsai-qat-m4-1p7b.ipynb (self-contained Qwen3-1.7B QAT).

Bakes in every lesson from the 0.6B campaign (see bonsai-e2e-qat-findings.md):
bf16 forwards (no fp16 inflation), unbiased skip-aware eval with skip==0
save gate, Adam8bit, warm checkpoints (opt state), throttled Drive backup,
snapshot_download-based export, transformers pinned before any import.
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"


def md(src):
    return {"cell_type": MD, "metadata": {}, "source": src}


def code(src):
    return {"cell_type": CODE, "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells = []

cells.append(md("""# Bonsai M4：Qwen3-1.7B 端到端三值 QAT（A100 专用）

**目标**：把 0.6B 验证过的配方（端到端 Z 潜变量 + STE）移植到 Qwen3-1.7B——与已发布 Ternary-Bonsai-1.7B 同基座。产出契约兼容的 PTQ1_0 artifact 并回本机验证。

**0.6B 战役已固化的经验**（全部烘焙进本 notebook，无需再踩）：
- bf16 前向（fp16 会随权重漂移产生评估通胀与激活溢出——曾制造假退化与假里程碑）；
- 逐窗 NLL 评估 + **skip==0 才有资格当 best**（跳窗子集有偏）；
- lr 阶梯教训：2e-4 稳定（高档全部 excursion），emb 码冻结只训尺度；
- 数据：WikiText-103（2.5M token 回收是 0.6B 平台的主因）；
- 断连防护：checkpoint 含优化器状态（温热续训）+ Drive 节流备份。

**运行要求**：bf16 capable GPU（**A100/L4/Ada，T4 不支持 bf16**）、~3 小时。运行时：代码执行程序 → 更改运行时类型 → A100。"""))

cells.append(code("""# 安装（必须在任何 import 之前：整套已验证版本钉死——新镜像的 transformers/tokenizers/hub 1.x 互不兼容）
%pip -q install --force-reinstall --no-deps "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
%pip -q install "datasets==5.0.1" "accelerate==1.14.0" sentencepiece protobuf bitsandbytes
print("installed")"""))

cells.append(code("""import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch, gc
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), \\
    "需要 bf16 capable GPU（A100/L4/Ada；T4 不行）"
DEV, BF = "cuda", torch.bfloat16
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0),
      "|", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")

from google.colab import drive
drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive"

import bitsandbytes as bnb
OPT_CLS = bnb.optim.Adam8bit
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
print("stack OK: transformers 4.57.1 + Adam8bit")"""))

cells.append(code("""MODEL = "Qwen/Qwen3-1.7B"
STEPS = 5000                 # A100 ~2-2.5s/步，全程 ~3h
BATCH, SEQ, TOPK = 8, 512, 50
LRS = [2e-4, 1e-3, 3e-4]     # Z / theta / 小岛（0.6B 战役验证的稳定档）
EVAL_EVERY, EVAL_WINDOWS = 100, 40
NTOK_TRAIN = 33_554_432      # 32M token（wikitext-103）
CKPT = "/content/m4_qat.pt"
DRIVE_EVERY = 500            # Drive 备份节流（checkpoint ~9GB，每次写 ~1-2 分钟）
GROUP, BLOCK = 128, 1024
SEED = 20260922              # 与 0.6B 管线同种子约定（check_contract 兼容）"""))

cells.append(md("""## 数学与模块（与 0.6B 验证版逐行一致，bf16 烘焙）"""))

cells.append(code("""import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_sign_cache = {}
def signs_for_width(width, device, dtype=torch.float32):
    if width not in _sign_cache:
        g = torch.Generator().manual_seed(SEED + width)
        _sign_cache[width] = (torch.randint(0, 2, (width,), generator=g) * 2 - 1)
    return _sign_cache[width].to(device=device, dtype=dtype)

def fwht(x):
    n = x.shape[-1]
    y = x
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return y

def rot_fwd(x, signs, n=BLOCK):
    lead, w = x.shape[:-1], x.shape[-1]
    assert w % n == 0, f"width {w} 不是 {n} 的倍数"
    xb = x.float().reshape(*lead, w // n, n)
    return (fwht(xb * signs.reshape(-1, n)) / math.sqrt(n)).reshape(*lead, w)

def rot_inv(x, signs, n=BLOCK):
    lead, w = x.shape[:-1], x.shape[-1]
    xb = x.float().reshape(*lead, w // n, n)
    return (fwht(xb) / math.sqrt(n) * signs.reshape(-1, n)).reshape(*lead, w)

def quantize_init(wf, g=GROUP):
    \"\"\"zerofrac 初始化：s = 0.836σ → 零占比 ~1/3（artifact 实测 0.328）\"\"\"
    out, width = wf.shape
    wg = wf.reshape(out, width // g, g)
    s = (0.836 * wg.std(dim=-1)).clamp(min=1e-8)
    t = torch.round(wg / s.unsqueeze(-1)).clamp(-1, 1).to(torch.int8)
    return t.reshape(out, width), s

_dev = "cuda" if torch.cuda.is_available() else "cpu"
w = torch.randn(64, 1024, device=_dev)
s = signs_for_width(1024, _dev)
x = torch.randn(3, 7, 1024, device=_dev)
err = (rot_fwd(x, s) @ rot_fwd(w, s).T - x @ w.T).abs().max()
print(f"rotation identity check (should be ~1e-5): {err.item():.2e}")"""))

cells.append(code("""class _TernarySTE(torch.autograd.Function):
    \"\"\"w = s·clamp(round(Z/s), -1, 1)，bf16 输出；STE 过 round；|Z/s|>1.5 死区。\"\"\"
    @staticmethod
    def forward(ctx, Z, s):
        ctx.save_for_backward(Z, s)
        sr = s.repeat_interleave(GROUP, dim=1)
        return (sr * torch.clamp(torch.round(Z / sr), -1, 1)).to(BF)

    @staticmethod
    def backward(ctx, g):
        Z, s = ctx.saved_tensors
        gf = g.float()
        sr = s.repeat_interleave(GROUP, dim=1)
        T = torch.clamp(torch.round(Z / sr), -1, 1)
        mask = ((Z / sr).abs() <= 1.5).to(gf.dtype)
        gZ = gf * sr * mask
        gs = (gf * T).reshape(T.shape[0], -1, GROUP).sum(-1)
        return gZ, gs


class _EmbSTE(torch.autograd.Function):
    \"\"\"codes 固定、只训尺度；分块避免整表 fp32 临时量。\"\"\"
    @staticmethod
    def forward(ctx, codes, s):
        ctx.save_for_backward(codes)
        out, width = codes.shape
        w = torch.empty(out, width, dtype=BF, device=codes.device)
        CH = 16384
        for a in range(0, out, CH):
            b = min(a + CH, out)
            w[a:b] = (s[a:b].repeat_interleave(GROUP, dim=1) * codes[a:b].float()).to(BF)
        return w

    @staticmethod
    def backward(ctx, g):
        (codes,) = ctx.saved_tensors
        out, width = codes.shape
        gs = torch.empty(out, width // GROUP, dtype=torch.float32, device=codes.device)
        CH = 16384
        for a in range(0, out, CH):
            b = min(a + CH, out)
            prod = (g[a:b].float() * codes[a:b].float()).reshape(b - a, -1, GROUP)
            gs[a:b] = prod.sum(-1)
        return None, gs


def _inv_softplus(s):
    return torch.log(torch.expm1(s.clamp(min=1e-6))).clamp(min=-12)


class RotQATLinear(nn.Module):
    def __init__(self, lin, signs):
        super().__init__()
        with torch.no_grad():
            wf = rot_fwd(lin.weight.data, signs)
            codes, s = quantize_init(wf)
        self.register_buffer("signs", signs.float())
        self.Z = nn.Parameter(codes.float() * s.repeat_interleave(GROUP, 1))
        self.theta = nn.Parameter(_inv_softplus(s))
        self.register_buffer("codes0", codes)
        self.bias = lin.bias

    def scales(self):
        return F.softplus(self.theta).clamp(min=1e-8)

    def w_eff(self):
        return _TernarySTE.apply(self.Z, self.scales())

    def forward(self, x):
        return F.linear(rot_fwd(x, self.signs).to(BF), self.w_eff(), self.bias)


class RotQATHead(nn.Module):
    \"\"\"lm_head 走折叠表：logits = (s·T)(R h)——契约要求 output 折叠量化\"\"\"
    def __init__(self, emb):
        super().__init__()
        self.emb = emb

    def forward(self, h):
        return F.linear(rot_fwd(h, self.emb.signs).to(BF), self.emb.w_eff())


class RotQATEmbedding(nn.Module):
    def __init__(self, table, signs):
        super().__init__()
        with torch.no_grad():
            ef = rot_fwd(table, signs)
            codes, s = quantize_init(ef)
        self.register_buffer("signs", signs.float())
        self.register_buffer("codes", codes)
        self.theta = nn.Parameter(_inv_softplus(s))

    def scales(self):
        return F.softplus(self.theta).clamp(min=1e-8)

    def w_eff(self):
        return _EmbSTE.apply(self.codes, self.scales())

    def forward(self, ids):
        return rot_inv(F.embedding(ids, self.w_eff()), self.signs).to(BF)


class Fp32RMSNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.weight = nn.Parameter(norm.weight.data.float())
        self.eps = getattr(norm, "variance_epsilon", None) or 1e-6

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * xf).to(x.dtype)


TARGETS = [("self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + \\
          [("mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]

def install(model):
    dev = next(model.parameters()).device
    emb = model.model.embed_tokens
    new_emb = RotQATEmbedding(emb.weight.data, signs_for_width(emb.weight.shape[1], dev))
    model.model.embed_tokens = new_emb
    model.lm_head = RotQATHead(new_emb)
    linears = []
    for layer in model.model.layers:
        for ppath, name in TARGETS:
            parent = layer
            for p in ppath.split("."):
                parent = getattr(parent, p)
            lin = getattr(parent, name)
            mod = RotQATLinear(lin, signs_for_width(lin.weight.shape[1], dev))
            setattr(parent, name, mod)
            linears.append(mod)
        layer.input_layernorm = Fp32RMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = Fp32RMSNorm(layer.post_attention_layernorm)
        layer.self_attn.q_norm = Fp32RMSNorm(layer.self_attn.q_norm)
        layer.self_attn.k_norm = Fp32RMSNorm(layer.self_attn.k_norm)
    model.model.norm = Fp32RMSNorm(model.model.norm)
    for p in model.parameters():
        p.requires_grad_(False)
    tr = {"Z": [], "theta": [], "island": []}
    for m in model.modules():
        if isinstance(m, RotQATLinear):
            m.Z.requires_grad_(True); m.theta.requires_grad_(True)
            tr["Z"].append(m.Z); tr["theta"].append(m.theta)
        elif isinstance(m, RotQATEmbedding):
            m.theta.requires_grad_(True); tr["theta"].append(m.theta)
        elif isinstance(m, Fp32RMSNorm):
            m.weight.requires_grad_(True); tr["island"].append(m.weight)
    return new_emb, linears, tr"""))

cells.append(code("""\"\"\"数据：预下载一切后切离线（transformers 4.57.1 在线加载 tokenizer 会查询仓库的
additional_chat_templates 目录，Qwen 仓库没有 → 404 传播；离线走缓存则完全绕开）\"\"\"
import os
from huggingface_hub import snapshot_download
snapshot_download(MODEL)
from datasets import load_dataset
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

tok = AutoTokenizer.from_pretrained(MODEL)
def ids_of(split, max_tokens=None):
    text = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[split]["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids[:max_tokens] if max_tokens else ids

test_ids = ids_of("test")
text103 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"] if t.strip())
train_ids = tok(text103, return_tensors="pt").input_ids[0][:NTOK_TRAIN]
del text103; gc.collect()
print(f"train {train_ids.numel()/1e6:.1f}M tok | test {test_ids.numel()/1e3:.0f}K tok", flush=True)"""))

cells.append(code("""\"\"\"构建：student(bf16) + FP 参考 + 教师缓存 + 优化器\"\"\"
import time
import numpy as np

student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV)
emb, linears, tr = install(student)
student.train()

teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV).eval()

@torch.no_grad()
def ppl_of(model, n_windows=EVAL_WINDOWS, seq=SEQ):
    model.eval()
    nll, cnt, skipped = 0.0, 0, 0
    V = model.config.vocab_size
    for w in range(n_windows):
        x = test_ids[w*seq:(w+1)*seq].unsqueeze(0).to(DEV)
        lg = model(x).logits.float()
        v = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction="sum")
        if torch.isfinite(v):
            nll += v.item(); cnt += x[:, 1:].numel()
        else:
            skipped += 1
    model.train()
    return (math.exp(nll / cnt) if cnt else float("nan")), skipped

fp_ppl, _ = ppl_of(teacher)
init_ppl, init_sk = ppl_of(student)
print(f"FP 参考 ppl = {fp_ppl:.2f} | 三值起点 ppl = {init_ppl:.2f} (skip {init_sk})", flush=True)

# 教师缓存（训练步的 top-50 目标；A100 ~15-20 分钟）
NW = STEPS * BATCH
vals = np.memmap("/content/tv.npy", dtype=np.float16, mode="w+", shape=(NW, SEQ - 1, TOPK))
vids = np.memmap("/content/ti.npy", dtype=np.int32,  mode="w+", shape=(NW, SEQ - 1, TOPK))
spans = [((s - 1) * BATCH * SEQ) % (train_ids.numel() - BATCH * SEQ)
         for s in range(1, STEPS + 1)]
t0 = time.time()
with torch.no_grad():
    for k in range(NW):
        begin = spans[k // BATCH]
        x = train_ids[begin + (k % BATCH) * SEQ:
                      begin + (k % BATCH) * SEQ + SEQ].unsqueeze(0).to(DEV)
        v, idx = torch.topk(teacher(x).logits[:, :-1].float(), TOPK, -1)
        vals[k] = v.half().cpu().numpy()
        vids[k] = idx.cpu().numpy().astype(np.int32)
        if k % 5000 == 0:
            print(f"  cache {k}/{NW} ({time.time()-t0:.0f}s)", flush=True)
vals.flush(); vids.flush()
del teacher; gc.collect(); torch.cuda.empty_cache()
print(f"teacher freed ({time.time()-t0:.0f}s); "
      f"{torch.cuda.memory_allocated()/2**20:.0f}MB", flush=True)

opt = OPT_CLS([{"params": tr["Z"], "lr": LRS[0]},
               {"params": tr["theta"], "lr": LRS[1]},
               {"params": tr["island"], "lr": LRS[2]}], betas=(0.9, 0.95))"""))

cells.append(md("""## 训练（从零，恒定 2e-4）

判读参照（0.6B 经验外推，1.7B 冗余更大通常更易量化）：
- FP 参考的 2× 与 1.5× 线会打印在每行 eval 后；
- skip 恒应为 0；`moved` 无固定警戒（0.6B 到 19% 仍在改善），唯一判据是 ppl 方向；
- 每次无偏新低存 /content（含优化器，可温热续训），每 500 步至少同步一次 Drive。"""))

cells.append(code("""import shutil

last_drive = [0]
def save_ckpt(step_i, ppl_i):
    torch.save({"step": step_i, "model": MODEL,
                "Z": [m.Z.detach().cpu() for m in linears],
                "theta": [m.theta.detach().cpu() for m in linears],
                "emb_theta": emb.theta.detach().cpu(), "emb_codes": emb.codes.cpu(),
                "islands": {n: m.weight.detach().cpu() for n, m in student.named_modules()
                            if isinstance(m, Fp32RMSNorm)},
                "opt": opt.state_dict(),
                "hist": hist}, CKPT)
    if step_i - last_drive[0] >= DRIVE_EVERY:        # 节流：至少间隔 500 步
        shutil.copy(CKPT, f"{DRIVE_DIR}/m4_qat_best.pt")
        last_drive[0] = step_i
        print(f"  [Drive 同步 @ {step_i}]", flush=True)

@torch.no_grad()
def moved_stats():
    ms, zs = [], []
    for m in linears:
        T = torch.clamp(torch.round(m.Z / m.scales().repeat_interleave(GROUP, 1)), -1, 1)
        ms.append((T.to(torch.int8) != m.codes0).float().mean().item())
        zs.append((T == 0).float().mean().item())
    return sum(ms) / len(ms), sum(zs) / len(zs)

hist, best, bad = [], float("inf"), 0
t0 = time.time()
for step in range(1, STEPS + 1):
    j = step - 1
    tv = torch.from_numpy(np.array(vals[j * BATCH:(j + 1) * BATCH])).to(DEV).float()
    ti = torch.from_numpy(np.array(vids[j * BATCH:(j + 1) * BATCH])).to(DEV).long()
    begin = spans[j]
    opt.zero_grad(set_to_none=True)
    loss_val = 0.0
    for i in range(BATCH):                     # 逐样本反传：logits fp32 峰值 /BATCH（1.7B 批 8 会 OOM 39.5G）
        x = train_ids[begin + i * SEQ: begin + (i + 1) * SEQ].unsqueeze(0).to(DEV)
        logits = student(x).logits[:, :-1].float()
        loss_i = -(F.softmax(tv[i:i+1], -1) * torch.gather(
            F.log_softmax(logits, -1), -1, ti[i:i+1])).sum(-1).mean()
        (loss_i / BATCH).backward()
        loss_val += loss_i.item() / BATCH
    if math.isfinite(loss_val):
        torch.nn.utils.clip_grad_norm_(tr["Z"] + tr["theta"] + tr["island"], 1.0)
        opt.step()
    if step % EVAL_EVERY == 0 or step == STEPS:
        ppl, skipped = ppl_of(student)
        mv, zf = moved_stats()
        hist.append((step, ppl))
        print(f">> step {step} ppl = {ppl:.2f} | xFP {ppl/fp_ppl:.2f} "
              f"(best {best:.2f}) skip {skipped}/{EVAL_WINDOWS} moved {mv*100:.2f}% "
              f"loss {loss_val:.3f} ({step*BATCH*SEQ/(time.time()-t0):.0f} tok/s)", flush=True)
        if math.isfinite(ppl) and skipped == 0 and ppl < best:
            best = ppl; bad = 0
            save_ckpt(step, ppl)
        else:
            bad += 1
            if bad >= 8:
                print("plateau/skip x8: stopped for review", flush=True)
                break
print("history:", [(h[0], round(h[1], 2)) for h in hist], flush=True)"""))

cells.append(md("""## 断连恢复（checkpoint 在 Drive 上时）

重启/断连后：跑完 安装 → 环境 → 配置 → 数学 → 模块 → 数据（跳过 build 与训练 cell），再运行本 cell。实战验证过一次（step 2600 断连，恢复后零损失）。"""))

cells.append(code("""# ===== M4 断连恢复：从 Drive 续训（实战验证版：chunk4 + 缓存幸存检测）=====
import gc, time, math, os, shutil
import numpy as np
from transformers import AutoModelForCausalLM

CKPT = "/content/m4_qat.pt"
shutil.copy("/content/drive/MyDrive/m4_qat_best.pt", CKPT)
student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV)
emb, linears, tr = install(student)
student.train()
ck = torch.load(CKPT, map_location="cpu")
with torch.no_grad():
    for m, Z, th in zip(linears, ck["Z"], ck["theta"]):
        m.Z.copy_(Z); m.theta.copy_(th)
    emb.theta.copy_(ck["emb_theta"])
    for n, m_i in student.named_modules():
        if isinstance(m_i, Fp32RMSNorm) and n in ck["islands"]:
            m_i.weight.copy_(ck["islands"][n])
print(f"resumed step {ck['step']}, best {min(h[1] for h in ck['hist']):.2f}", flush=True)

S0, END, CHUNK = ck["step"] + 1, 5000, 4
teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV).eval()
fp_ppl, _ = ppl_of(teacher)
print(f"FP 参考 ppl = {fp_ppl:.2f}", flush=True)

full_n = STEPS * BATCH
have_full = (os.path.exists("/content/tv.npy")
             and os.path.getsize("/content/tv.npy") == full_n * (SEQ - 1) * TOPK * 2)
if have_full:
    CACHE_S0, NW = 1, full_n
    vals = np.memmap("/content/tv.npy", dtype=np.float16, mode="r", shape=(NW, SEQ - 1, TOPK))
    vids = np.memmap("/content/ti.npy", dtype=np.int32,  mode="r", shape=(NW, SEQ - 1, TOPK))
    del teacher; gc.collect(); torch.cuda.empty_cache()
    print("reusing surviving full cache", flush=True)
else:
    CACHE_S0, NW = S0, (END - S0 + 1) * BATCH
    vals = np.memmap("/content/tv.npy", dtype=np.float16, mode="w+", shape=(NW, SEQ - 1, TOPK))
    vids = np.memmap("/content/ti.npy", dtype=np.int32,  mode="w+", shape=(NW, SEQ - 1, TOPK))
    t0 = time.time()
    with torch.no_grad():
        for k in range(NW):
            s_ = CACHE_S0 + k // BATCH
            begin = ((s_ - 1) * BATCH * SEQ) % (train_ids.numel() - BATCH * SEQ)
            x = train_ids[begin + (k % BATCH) * SEQ:
                          begin + (k % BATCH) * SEQ + SEQ].unsqueeze(0).to(DEV)
            v, idx = torch.topk(teacher(x).logits[:, :-1].float(), TOPK, -1)
            vals[k] = v.half().cpu().numpy(); vids[k] = idx.cpu().numpy().astype(np.int32)
            if k % 4000 == 0: print(f"  cache {k}/{NW} ({time.time()-t0:.0f}s)", flush=True)
    vals.flush(); vids.flush()
    del teacher; gc.collect(); torch.cuda.empty_cache()
    print(f"cache rebuilt ({time.time()-t0:.0f}s)", flush=True)

spans = [((s - 1) * BATCH * SEQ) % (train_ids.numel() - BATCH * SEQ)
         for s in range(CACHE_S0, CACHE_S0 + NW // BATCH)]
opt = OPT_CLS([{"params": tr["Z"], "lr": 2e-4},
               {"params": tr["theta"], "lr": 1e-3},
               {"params": tr["island"], "lr": 3e-4}], betas=(0.9, 0.95))
opt.load_state_dict(ck["opt"])
for g, lr in zip(opt.param_groups, [2e-4, 1e-3, 3e-4]):   # 防 lr 被旧状态覆盖
    g["lr"] = lr
del ck["Z"], ck["theta"], ck["islands"]; gc.collect(); torch.cuda.empty_cache()

last_drive = [ck["step"]]
hist, best, bad = ck["hist"], min(h[1] for h in ck["hist"]), 0
t0 = time.time()
for step in range(S0, END + 1):
    j = step - CACHE_S0
    tv = torch.from_numpy(np.array(vals[j * BATCH:(j + 1) * BATCH])).to(DEV).float()
    ti = torch.from_numpy(np.array(vids[j * BATCH:(j + 1) * BATCH])).to(DEV).long()
    begin = spans[j]
    opt.zero_grad(set_to_none=True)
    loss_val = 0.0
    for c in range(0, BATCH, CHUNK):
        n = min(CHUNK, BATCH - c)
        x = train_ids[begin + c * SEQ: begin + (c + n) * SEQ].reshape(n, SEQ).to(DEV)
        logits = student(x).logits[:, :-1].float()
        loss_c = -(F.softmax(tv[c:c + n], -1) * torch.gather(
            F.log_softmax(logits, -1), -1, ti[c:c + n])).sum(-1).mean()
        (loss_c * n / BATCH).backward()
        loss_val += loss_c.item() * n / BATCH
    if math.isfinite(loss_val):
        torch.nn.utils.clip_grad_norm_(tr["Z"] + tr["theta"] + tr["island"], 1.0)
        opt.step()
    if step % EVAL_EVERY == 0 or step == END:
        ppl, skipped = ppl_of(student)
        mv, zf = moved_stats()
        hist.append((step, ppl))
        print(f">> step {step} ppl = {ppl:.2f} | xFP {ppl/fp_ppl:.2f} (best {best:.2f}) "
              f"skip {skipped}/{EVAL_WINDOWS} moved {mv*100:.2f}%", flush=True)
        if math.isfinite(ppl) and skipped == 0 and ppl < best:
            best = ppl; bad = 0
            torch.save({"step": step, "model": MODEL,
                        "Z": [m.Z.detach().cpu() for m in linears],
                        "theta": [m.theta.detach().cpu() for m in linears],
                        "emb_theta": emb.theta.detach().cpu(), "emb_codes": emb.codes.cpu(),
                        "islands": {n: m.weight.detach().cpu() for n, m in student.named_modules()
                                    if isinstance(m, Fp32RMSNorm)},
                        "opt": opt.state_dict(), "hist": hist}, CKPT)
            if step - last_drive[0] >= DRIVE_EVERY:
                shutil.copy(CKPT, "/content/drive/MyDrive/m4_qat_best.pt")
                last_drive[0] = step
                print(f"  [Drive 同步 @ {step}]", flush=True)
        else:
            bad += 1
            if bad >= 8:
                print("plateau/skip x8: stopped for review", flush=True)
                break
print("history:", [(h[0], round(h[1], 2)) for h in hist], flush=True)"""))

cells.append(md("""## 导出（契约兼容）→ zip 下载

导出的目录结构与 0.6B 完全一致（折叠 safetensors + `hadamard_packing.json`），本机收尾链通用。"""))

cells.append(code("""import json, shutil
from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

OUT = Path("/content/qwen3-1.7b-ternary-hd-qat"); OUT.mkdir(exist_ok=True)
ck = torch.load(CKPT, map_location="cpu")

s = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
emb_e, lin_e, _ = install(s)
with torch.no_grad():
    for m, Z, th in zip(lin_e, ck["Z"], ck["theta"]):
        m.Z.copy_(Z); m.theta.copy_(th)
    emb_e.theta.copy_(ck["emb_theta"]); emb_e.codes.copy_(ck["emb_codes"])
    for n, m_i in s.named_modules():
        if isinstance(m_i, Fp32RMSNorm) and n in ck["islands"]:
            m_i.weight.copy_(ck["islands"][n])

def snap(w):
    wg = w.reshape(w.shape[0], -1, GROUP)
    amax = wg.abs().amax(-1, keepdim=True)
    return (wg.sign() * amax).reshape(w.shape).half()

out = {k: v.half() for k, v in
       AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).state_dict().items()}
with torch.no_grad():
    for path, m in s.named_modules():
        if isinstance(m, RotQATLinear):
            out[path + ".weight"] = snap(m.w_eff())
    wq = snap(emb_e.w_eff())
out["model.embed_tokens.weight"] = wq
out["lm_head.weight"] = wq.clone()
for n, w_isl in ck["islands"].items():
    out[n + ".weight"] = w_isl.half()

hf_dir = Path(snapshot_download(MODEL))
for f in ("config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "vocab.json", "merges.txt"):
    if (hf_dir / f).exists():
        shutil.copy(hf_dir / f, OUT / f)
cfg = json.loads((hf_dir / "config.json").read_text())
cfg["tie_word_embeddings"] = False
(OUT / "config.json").write_text(json.dumps(cfg, indent=2))
save_file(out, str(OUT / "model.safetensors"), metadata={"format": "pt"})

cfg_h = AutoConfig.from_pretrained(MODEL)
widths = sorted({m.Z.shape[1] for m in lin_e} | {wq.shape[1]})
records = []
for i in range(cfg_h.num_hidden_layers):
    for sub in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        records.append({"name": f"model.layers.{i}.{sub}.weight", "axis": -1,
                        "role": "fold-before-matmul"})
records += [{"name": "lm_head.weight", "axis": -1, "role": "fold-before-matmul"},
            {"name": "model.embed_tokens.weight", "axis": -1, "role": "inverse-after-lookup"}]
manifest = {"schema_version": 1, "kind": "hadamard-weight-fold",
            "status": "requires-matching-runtime",
            "transform": {"name": "normalized-signed-sylvester-walsh-hadamard",
                          "block_size": BLOCK, "sign_mode": "explicit"},
            "signs": {str(w): signs_for_width(w, "cpu").to(torch.int8).tolist() for w in widths},
            "tensors": records}
(OUT / "hadamard_packing.json").write_text(json.dumps(manifest))
print("exported", OUT)

shutil.make_archive("/content/" + OUT.name, "zip", root_dir="/content", base_dir=OUT.name)
try:
    from google.colab import files
    files.download(f"/content/{OUT.name}.zip")
except Exception as e:
    print("download skipped:", e, "| zip at /content/", OUT.name + ".zip")"""))

cells.append(md("""## 回本机收尾（备忘）

```bash
# zip 放到 <WORK_DIR>/m4/
python <FORK_DIR>/convert_hf_to_gguf.py <解压目录> --outfile m4.f16.gguf
BIN=<RUNTIME_BIN>
$BIN/llama-quantize --token-embedding-type PTQ1_0 --output-tensor-type PTQ1_0 \\
    m4.f16.gguf m4.ptq1_0.gguf PTQ1_0
python proto/common/check_contract.py m4.ptq1_0.gguf <解压目录>
$BIN/llama-perplexity -m m4.ptq1_0.gguf -f <WORK_DIR>/wt2_30k.txt -t 12 -c 512
```

符号种子与 0.6B 管线一致（`SEED=20260922`），`check_contract.py` 的 sign_values 校验直接可用。"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = Path("notebooks/tritfold-train-1p7b.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
