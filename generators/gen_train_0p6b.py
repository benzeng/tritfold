#!/usr/bin/env python
"""Build Bonsai/colab/bonsai-qat-e2e-colab.ipynb (idempotent)."""
import json
from pathlib import Path

MD = "markdown"
CODE = "code"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": CODE, "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells = []

cells.append(md("""# Bonsai 端到端三值 QAT：Z 潜变量 + STE（Colab 版）

**目标**：在 ≥16GB 显存上做本机 GTX 1060 放不下的**端到端**三值化训练，闭合质量缺口（本机纯尺度训练饱和于 ppl 91.6 ≈ 3.3×FP，目标 ≤1.5×FP ≈ 41）。

**为什么必须端到端**（oneLLM 仓库 `Bonsai/bonsai-m5a1-findings.md` 七项实验结论）：局部目标（组合翻转、逐层 MSE——教师输入或学生输入）与全局语言建模目标在 1.58bpw 下系统性错位，28/28 层组的局部改善全部导致全局变差。唯一可行信号是端到端梯度——需要 fp32 潜变量 Z（544M 参数，参数+梯度+Adam ≈ 9GB fp32 / ~5.5GB 8-bit Adam）。

**机制**（全部在本机验证过）：
- 契约式 Hadamard 旋转：W′ = W·R⁻¹（R = H₁₀₂₄·S/√1024 沿输入轴，每宽度一条固定 ±1 符号向量），激活在线 FWHT，embedding 逆变换（对应 PrismML fork 的 `prism.hadamard.*` 契约）；
- 码 = `clamp(round(Z/s_g), -1, 1)` 的**派生量**，STE 直通梯度（死区 |Z/s|>1.5）；
- zerofrac 初始化（目标零占比 1/3，比 amax RTN 起点好 93 倍）；
- 教师_Top-50 logits KD + 尺度/小岛联合训练（已验证超参）。

**运行时**：T4（免费档，0.6B ~35 分钟/1500 步）或 L4/A100（可把 `MODEL` 换成 Qwen3-1.7B，A100 40GB 建议开 `CACHE_TEACHER`）。菜单：代码执行程序 → 更改运行时类型 → GPU。"""))

cells.append(code("""\"\"\"GPU 检查 + 依赖（Colab 镜像已带 torch/bitsandbytes，只钉 transformers）\"\"\"
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # 抗碎片（须在 torch 前设置）

!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

%pip -q install --force-reinstall --no-deps "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
%pip -q install "datasets==5.0.1" "accelerate==1.14.0" sentencepiece protobuf bitsandbytes

import torch, gc
DEV = "cuda"
assert torch.cuda.is_available(), "请先在菜单中启用 GPU 运行时"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))

try:
    import bitsandbytes as bnb
    OPT_CLS = bnb.optim.Adam8bit   # m+v 8bit: 省约 3GB（T4 上留足余量）
    print("using bitsandbytes Adam8bit")
except Exception as e:
    OPT_CLS = torch.optim.Adam
    print("bitsandbytes unavailable, plain Adam:", e)"""))

cells.append(md("""## 配置

| 键 | 默认 | 说明 |
|---|---|---|
| `MODEL` | Qwen3-0.6B | A100 可换 `Qwen/Qwen3-1.7B` |
| `STEPS` / `BATCH` / `SEQ` | 1500 / 4 / 512 | T4 上 ~1.3s/步 |
| `ZLR` | 2e-4 | 潜变量学习率（权重尺度 ~0.02，切忌 >1e-3——本机逐层实验的教训） |
| `SCALE_LR` / `ISLAND_LR` | 1e-3 / 3e-4 | 尺度与归一化小岛（本机已验证组合；尺度需配合梯度裁剪） |
| `EMB_TRAIN_Z` | False | embedding 表只训尺度（155M 表的 Z 优化器状态可再省 2GB） |"""))

cells.append(code("""MODEL = "Qwen/Qwen3-0.6B"          # A100: "Qwen/Qwen3-1.7B"
STEPS = 1500
BATCH, SEQ = 4, 512
ZLR = 2e-4
SCALE_LR, ISLAND_LR = 1e-3, 3e-4
EMB_TRAIN_Z = False
EVAL_EVERY, EVAL_WINDOWS = 50, 40
TOPK = 50
CKPT = "/content/bonsai_qat.pt"

GROUP, BLOCK = 128, 1024          # artifact 契约：g128 尺度、块 1024 Hadamard
SEED = 20260922                   # 与本机管线一致的符号向量种子（导出 manifest 需相同）"""))

cells.append(code("""\"\"\"旋转与三值数学（与本机 proto/common/fwht_torch.py 逐行一致，勿改种子约定）\"\"\"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_sign_cache = {}

def signs_for_width(width, device, dtype=torch.float32):
    key = width
    if key not in _sign_cache:
        g = torch.Generator().manual_seed(SEED + width)
        _sign_cache[key] = (torch.randint(0, 2, (width,), generator=g) * 2 - 1)
    return _sign_cache[key].to(device=device, dtype=dtype)

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
    xb = x.float().reshape(*lead, w // n, n)
    return (fwht(xb * signs.reshape(-1, n)) / math.sqrt(n)).reshape(*lead, w)

def rot_inv(x, signs, n=BLOCK):
    lead, w = x.shape[:-1], x.shape[-1]
    xb = x.float().reshape(*lead, w // n, n)
    return (fwht(xb) / math.sqrt(n) * signs.reshape(-1, n)).reshape(*lead, w)

def quantize_init(wf, g=GROUP):
    \"\"\"zerofrac 初始化：s = 0.836σ → 零占比 ~1/3（官方 artifact 实测 0.328）\"\"\"
    out, width = wf.shape
    wg = wf.reshape(out, width // g, g)
    s = (0.836 * wg.std(dim=-1)).clamp(min=1e-8)
    t = torch.round(wg / s.unsqueeze(-1)).clamp(-1, 1).to(torch.int8)
    return t.reshape(out, width), s

# 快速自检：恒等性（不量化时 W'(Rx) == Wx）
_dev = "cuda" if torch.cuda.is_available() else "cpu"
w = torch.randn(64, 1024, device=_dev)
s = signs_for_width(1024, _dev)
x = torch.randn(3, 7, 1024, device=_dev)
err = (rot_fwd(x, s) @ rot_fwd(w, s).T - x @ w.T).abs().max()
print(f"rotation identity check (should be ~1e-5): {err.item():.2e}")"""))

cells.append(code("""\"\"\"Z 潜变量 + STE 三值模块与模型手术\"\"\"
class _TernarySTE(torch.autograd.Function):
    \"\"\"w = s·clamp(round(Z/s), -1, 1)，返回 fp16（省掉 fp32 图保留）。
    STE 过 round；|Z/s|>1.5 死区；backward 重算 T。\"\"\"
    @staticmethod
    def forward(ctx, Z, s):
        ctx.save_for_backward(Z, s)
        sr = s.repeat_interleave(GROUP, dim=1)
        return (sr * torch.clamp(torch.round(Z / sr), -1, 1)).half()

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
    \"\"\"codes 固定、只训尺度。前后向都分块：整表 fp32 临时量 1.2GB+（T4 OOM 实测），
    分块后峰值 ~30MB。\"\"\"
    @staticmethod
    def forward(ctx, codes, s):
        ctx.save_for_backward(codes)
        out, width = codes.shape
        w = torch.empty(out, width, dtype=torch.float16, device=codes.device)
        CH = 16384
        for a in range(0, out, CH):
            b = min(a + CH, out)
            w[a:b] = (s[a:b].repeat_interleave(GROUP, dim=1) * codes[a:b].float()).half()
        return w

    @staticmethod
    def backward(ctx, g):
        (codes,) = ctx.saved_tensors
        out, width = codes.shape
        gs = torch.empty(out, width // GROUP, dtype=torch.float32, device=codes.device)
        CH = 16384
        for a in range(0, out, CH):
            b = min(a + CH, out)
            prod = (g[a:b] * codes[a:b]).reshape(b - a, -1, GROUP)
            gs[a:b] = prod.sum(-1, dtype=torch.float32)
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
        self.Z = nn.Parameter(codes.float() * s.repeat_interleave(GROUP, 1))  # 潜变量
        self.theta = nn.Parameter(_inv_softplus(s))
        self.register_buffer("codes0", codes)          # 移动量统计基准
        self.bias = lin.bias

    def scales(self):
        return F.softplus(self.theta).clamp(min=1e-8)

    def w_eff(self):
        return _TernarySTE.apply(self.Z, self.scales())

    def forward(self, x):
        z = rot_fwd(x, self.signs).half()
        return F.linear(z, self.w_eff(), self.bias)


class RotQATHead(nn.Module):
    \"\"\"lm_head 走折叠表：logits = (s·T)(R h)——契约要求 output 折叠量化\"\"\"
    def __init__(self, emb):
        super().__init__()
        self.emb = emb

    def forward(self, h):
        z = rot_fwd(h, self.emb.signs).half()
        return F.linear(z, self.emb.w_eff())


class RotQATEmbedding(nn.Module):
    \"\"\"折叠表：查表后逆变换；与 head 共享同一 w_eff（tied 契约）\"\"\"
    def __init__(self, table, signs):
        super().__init__()
        with torch.no_grad():
            ef = rot_fwd(table, signs)
            codes, s = quantize_init(ef)
        self.register_buffer("signs", signs.float())
        self.register_buffer("codes", codes)
        self.theta = nn.Parameter(_inv_softplus(s))
        self.Z = None

    def scales(self):
        return F.softplus(self.theta).clamp(min=1e-8)

    def w_eff(self):
        if self.Z is not None:
            return _TernarySTE.apply(self.Z, self.scales())
        return _EmbSTE.apply(self.codes, self.scales()).half()

    def forward(self, ids):
        return rot_inv(F.embedding(ids, self.w_eff()), self.signs).half()


class Fp32RMSNorm(nn.Module):
    \"\"\"FP 小岛（可训练，fp32 主权重）\"\"\"
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

def install(model, emb_train_z=False):
    dev = next(model.parameters()).device
    emb = model.model.embed_tokens
    s_hidden = signs_for_width(emb.weight.shape[1], dev)
    new_emb = RotQATEmbedding(emb.weight.data, s_hidden)
    if emb_train_z:
        s = new_emb.scales().detach()
        new_emb.Z = nn.Parameter(new_emb.codes.float() * s.repeat_interleave(GROUP, 1))
    model.model.embed_tokens = new_emb
    model.lm_head = RotQATHead(new_emb)          # 原头仍是未量化权重，必须换掉

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
    trainables = {"Z": [], "theta": [], "island": []}
    for m in model.modules():
        if isinstance(m, RotQATLinear):
            m.Z.requires_grad_(True); m.theta.requires_grad_(True)
            trainables["Z"].append(m.Z); trainables["theta"].append(m.theta)
        elif isinstance(m, RotQATEmbedding):
            m.theta.requires_grad_(True); trainables["theta"].append(m.theta)
            if m.Z is not None:
                m.Z.requires_grad_(True); trainables["Z"].append(m.Z)
        elif isinstance(m, Fp32RMSNorm):
            m.weight.requires_grad_(True); trainables["island"].append(m.weight)
    return new_emb, linears, trainables"""))

cells.append(code("""\"\"\"数据与评估（预下载后切离线：transformers 在线加载 tokenizer 会查询仓库的
additional_chat_templates 目录，不存在时 404 会传播——离线走缓存则绕开）\"\"\"
import os
from huggingface_hub import snapshot_download
snapshot_download(MODEL)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

tok = AutoTokenizer.from_pretrained(MODEL)
def ids_of(split, max_tokens=None):
    text = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[split]["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids[:max_tokens] if max_tokens else ids

test_ids = ids_of("test")
train_ids = ids_of("train", max_tokens=STEPS * BATCH * SEQ + SEQ)

@torch.no_grad()
def quick_ppl(model, n_windows=40, seq=SEQ):
    model.eval()
    nll, cnt = 0.0, 0
    V = model.config.vocab_size
    for w in range(n_windows):
        x = test_ids[w * seq:(w + 1) * seq].unsqueeze(0).to(DEV)
        lg = model(x).logits.float()
        nll += F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction="sum").item()
        cnt += x[:, 1:].numel()
    model.train()
    return math.exp(nll / cnt)

@torch.no_grad()
def stats(linears, emb):
    with torch.no_grad():
        moved, zero = [], []
        for m in linears:
            T = torch.clamp(torch.round(m.Z / m.scales().repeat_interleave(GROUP, 1)), -1, 1)
            moved.append((T.to(torch.int8) != m.codes0).float().mean().item())
            zero.append((T == 0).float().mean().item())
        T_emb = emb.codes
        zero.append((T_emb == 0).float().mean().item())
    return sum(moved) / len(moved), sum(zero) / len(zero)"""))

cells.append(code("""\"\"\"端到端 QAT 主循环

每步：教师（常驻 fp16）算 top-50 目标 → 学生端到端前反传（Z + 尺度 + 小岛联合，全局裁剪 1.0）。
\"\"\"
import time

student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                               attn_implementation="sdpa").to(DEV)
emb, linears, tr = install(student, emb_train_z=EMB_TRAIN_Z)
student.train()

teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                               attn_implementation="sdpa").to(DEV).eval()

opt = OPT_CLS([
    {"params": tr["Z"], "lr": ZLR},
    {"params": tr["theta"], "lr": SCALE_LR},
    {"params": tr["island"], "lr": ISLAND_LR},
], betas=(0.9, 0.95))

ppl0 = quick_ppl(student, EVAL_WINDOWS)
print(f"[init] quick ppl = {ppl0:.2f} (zerofrac 初始化；FP 参考约 27-30)", flush=True)

hist, t0 = [], time.time()
for step in range(1, STEPS + 1):
    begin = ((step - 1) * BATCH * SEQ) % (train_ids.numel() - BATCH * SEQ)
    x = train_ids[begin: begin + BATCH * SEQ].reshape(BATCH, SEQ).to(DEV)

    with torch.no_grad():                      # 教师 top-50 目标（逐样本，峰值 ÷BATCH）
        tv_l, ti_l = [], []
        for i in range(x.size(0)):
            v, idx = torch.topk(teacher(x[i:i + 1]).logits[:, :-1].float(), TOPK, -1)
            tv_l.append(v); ti_l.append(idx)
        tv, ti = torch.cat(tv_l), torch.cat(ti_l)
        del tv_l, ti_l

    opt.zero_grad(set_to_none=True)
    loss_val = 0.0
    for i in range(x.size(0)):                 # 逐样本反传：logits 峰值 ÷BATCH
        logits = student(x[i:i + 1]).logits[:, :-1].float()
        s_logp = F.log_softmax(logits, dim=-1)
        loss_i = -(F.softmax(tv[i:i + 1], -1) * torch.gather(s_logp, -1, ti[i:i + 1])).sum(-1).mean()
        (loss_i / x.size(0)).backward()
        loss_val += loss_i.item() / x.size(0)

    if math.isfinite(loss_val):
        torch.nn.utils.clip_grad_norm_(
            tr["Z"] + tr["theta"] + tr["island"], 1.0)
        opt.step()
    else:
        print(f"step {step}: non-finite loss, skipped", flush=True)

    if step % 25 == 0:
        mv, zf = stats(linears, emb)
        print(f"step {step:4d} loss {loss_val:.4f} codes_moved {mv*100:.2f}% "
              f"zero {zf:.4f} ({step * BATCH * SEQ / (time.time() - t0):.0f} tok/s)", flush=True)
    if step % EVAL_EVERY == 0 or step == STEPS:
        ppl = quick_ppl(student, EVAL_WINDOWS)
        hist.append((step, ppl))
        print(f"  >> step {step} quick ppl = {ppl:.2f}", flush=True)
        torch.save({"step": step,
                    "Z": [m.Z.detach().cpu() for m in linears],
                    "theta": [m.theta.detach().cpu() for m in linears],
                    "emb_theta": emb.theta.detach().cpu(),
                    "emb_codes": emb.codes.cpu(),
                    "islands": {n: m.weight.detach().cpu()
                                for n, m in student.named_modules()
                                if isinstance(m, Fp32RMSNorm)},
                    "hist": hist}, CKPT)
print("history:", hist)"""))

cells.append(md("""## 断连恢复（checkpoint 在 Drive 上时）

重启/断连后：先跑完 GPU检查 → 配置 → 数学 → 模块 → 数据（本 cell 前的全部前置），再运行本 cell——从 Drive 的 checkpoint **温热续训**（含优化器动量；load 后必须按组重设 lr）。战役教训全在：skip==0 才有资格当 best、节流 Drive 备份。"""))

cells.append(code("""# ===== 断连恢复：Drive checkpoint → 温热续训（战役验证版）=====
import gc, time, math, shutil, os
from google.colab import drive
drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive"
SRC = f"{DRIVE_DIR}/bonsai_qat.pt"
assert os.path.exists(SRC), "Drive 上没有 checkpoint"
shutil.copy(SRC, CKPT)

student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                               attn_implementation="sdpa").to(DEV)
emb, linears, tr = install(student, emb_train_z=False)
student.train()
teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                               attn_implementation="sdpa").to(DEV).eval()

def restore(path):
    c = torch.load(path, map_location="cpu")
    with torch.no_grad():
        for m, Z, th in zip(linears, c["Z"], c["theta"]):
            m.Z.copy_(Z); m.theta.copy_(th)
        emb.theta.copy_(c["emb_theta"]); emb.codes.copy_(c["emb_codes"])
        for n, m_i in student.named_modules():
            if isinstance(m_i, Fp32RMSNorm) and n in c["islands"]:
                m_i.weight.copy_(c["islands"][n])
    return c

ck = restore(CKPT)
S0, END = ck["step"] + 1, ck["step"] + 1001
best0 = min(h[1] for h in ck["hist"])
print(f"resumed step {ck['step']}, best {best0:.2f}", flush=True)

@torch.no_grad()
def ppl_robust(model, n_windows=40, seq=SEQ):
    model.eval(); nll, cnt, skip = 0.0, 0, 0
    V = model.config.vocab_size
    for w in range(n_windows):
        x = test_ids[w*seq:(w+1)*seq].unsqueeze(0).to(DEV)
        lg = model(x).logits.float()
        v = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction="sum")
        if torch.isfinite(v): nll += v.item(); cnt += x[:, 1:].numel()
        else: skip += 1
    model.train()
    return (math.exp(nll / cnt) if cnt else float("nan")), skip

LRS = [ZLR, SCALE_LR, ISLAND_LR]
def build_opt():
    return OPT_CLS([{"params": tr["Z"], "lr": LRS[0]},
                    {"params": tr["theta"], "lr": LRS[1]},
                    {"params": tr["island"], "lr": LRS[2]}], betas=(0.9, 0.95))
opt = build_opt()
if "opt" in ck:
    opt.load_state_dict(ck["opt"])
    for g, lr in zip(opt.param_groups, LRS):   # 关键：load 会把旧 lr 覆盖回来
        g["lr"] = lr
    print("warm optimizer restored", flush=True)
del ck["Z"], ck["theta"], ck["islands"]; gc.collect(); torch.cuda.empty_cache()

last_drive = [ck["step"]]
hist, best, bad = ck["hist"], best0, 0
t0 = time.time()
for step in range(S0, END + 1):
    begin = ((step - 1) * BATCH * SEQ) % (train_ids.numel() - BATCH * SEQ)
    x = train_ids[begin: begin + BATCH * SEQ].reshape(BATCH, SEQ).to(DEV)
    with torch.no_grad():
        tv, ti = torch.topk(teacher(x).logits[:, :-1].float(), TOPK, dim=-1)
    opt.zero_grad(set_to_none=True); loss_val = 0.0
    for i in range(x.size(0)):
        logits = student(x[i:i+1]).logits[:, :-1].float()
        loss_i = -(F.softmax(tv[i:i+1], -1) * torch.gather(
            F.log_softmax(logits, -1), -1, ti[i:i+1])).sum(-1).mean()
        (loss_i / x.size(0)).backward(); loss_val += loss_i.item() / x.size(0)
    if math.isfinite(loss_val):
        torch.nn.utils.clip_grad_norm_(tr["Z"] + tr["theta"] + tr["island"], 1.0)
        opt.step()
    if step % EVAL_EVERY == 0 or step == END:
        ppl, skipped = ppl_robust(student, EVAL_WINDOWS)
        hist.append((step, ppl))
        print(f">> step {step} ppl = {ppl:.2f} (best {best:.2f}) skip {skipped}/{EVAL_WINDOWS}", flush=True)
        if math.isfinite(ppl) and skipped == 0 and ppl < best:
            best = ppl; bad = 0
            torch.save({"step": step, "Z": [m.Z.detach().cpu() for m in linears],
                        "theta": [m.theta.detach().cpu() for m in linears],
                        "emb_theta": emb.theta.detach().cpu(), "emb_codes": emb.codes.cpu(),
                        "islands": {n: m.weight.detach().cpu() for n, m in student.named_modules()
                                    if isinstance(m, Fp32RMSNorm)},
                        "opt": opt.state_dict(), "hist": hist}, CKPT)
            if step - last_drive[0] >= 500:
                shutil.copy(CKPT, SRC); last_drive[0] = step
                print("  [Drive 同步]", flush=True)
        else:
            bad += 1
            if bad >= 8:
                print("plateau/skip x8: stopped for review", flush=True)
                break"""))

cells.append(md("""## 结果判读

- **首个有效性信号**：ppl 突破 91.6（本机纯尺度训练的上限）——出现即证明端到端 STE 在移动码且方向正确；
- **达标线**：ppl ≤ 41（1.5×FP，M3 质量门）；0.6B 上 T4×1500 步是合理射程，未达标则 `STEPS=4000` 续跑（checkpoint 已含全部状态，重启本 cell 前加载 `CKPT` 回填即可——或直接从头跑，zerofrac 初始化是确定性的）；
- **调参优先级**：停滞先调 `ZLR`（×2 或 ÷2）；loss 发散则确认裁剪生效（默认已开）；`codes_moved` 应缓慢爬升（健康区间 1-15%，猛冲说明 ZLR 过大——本机教训：>1e-3 必炸）；
- `zero` 漂向 ~0.33 是好信号（与官方 artifact 0.328 对齐）。

## 训练后：导出（可选）

运行下面的导出 cell 会生成与 PrismML fork 契约一致的 HF 目录 + `hadamard_packing.json`（与本机 `proto/l1micro/export_model.py` 同一逻辑），打包下载后回本机执行 GGUF/PTQ1_0 链（命令在本机规划附录 B）。"""))

cells.append(code("""\"\"\"导出：折叠 safetensors + hadamard_packing.json（fork 契约）→ zip 下载\"\"\"
import json, shutil
from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import snapshot_download
from transformers import AutoConfig

OUT = Path("/content/qwen3-0.6b-ternary-hd-qat"); OUT.mkdir(exist_ok=True)
ck = torch.load(CKPT, map_location="cpu")

# 在 CPU 上重建学生骨架，回填训练后的 Z/theta/codes/小岛
s = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
emb_e, lin_e, _ = install(s, emb_train_z=False)
with torch.no_grad():
    for m, Z, th in zip(lin_e, ck["Z"], ck["theta"]):
        m.Z.copy_(Z); m.theta.copy_(th)
    emb_e.theta.copy_(ck["emb_theta"]); emb_e.codes.copy_(ck["emb_codes"])
    for n, m in s.named_modules():
        if isinstance(m, Fp32RMSNorm) and n in ck["islands"]:
            m.weight.copy_(ck["islands"][n])

def snap(w):
    \"\"\"Phase-3 amax 吸附：s·T 的值本来就是 ±s/0，吸附后与打包 RTN 完全一致\"\"\"
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

hf_dir = Path(snapshot_download(MODEL))   # hub 加载时 name_or_path 是仓库 ID，不是本地目录
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
    files.download(f"/content/{OUT.name}.zip")   # 回本机后跑附录 B 的 GGUF/PTQ1_0 链
except Exception as e:
    print("download skipped:", e, "| zip at /content/", OUT.name + ".zip")"""))

cells.append(md("""## 回本机后的收尾（备忘）

```bash
# 1.6GB 量级的 zip 解压到 <WORK_DIR>/qwen3-0.6b-ternary-hd-qat
python <FORK_DIR>/convert_hf_to_gguf.py <目录> --outfile out.f16.gguf
BIN=<RUNTIME_BIN>
$BIN/llama-quantize --token-embedding-type PTQ1_0 --output-tensor-type PTQ1_0 \\
    out.f16.gguf out.ptq1_0.gguf PTQ1_0
python proto/common/check_contract.py out.ptq1_0.gguf <目录>
$BIN/llama-perplexity -m out.ptq1_0.gguf -f <WORK_DIR>/wt2_30k.txt -t 12 -c 512
```

符号种子约定（`SEED=20260922`）与本机一致，`check_contract.py` 的 sign_values 校验直接可用。"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = Path("notebooks/tritfold-train-0p6b.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
