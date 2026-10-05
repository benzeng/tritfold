#!/usr/bin/env python
"""Build tritfold-m10-stack.ipynb (M10: stack all three validated levers).

Hypothesis: the three independently-validated levers are additive:
  - teacher quality: 1.7B->8B KL, +0.041 sciq (v0.5)
  - targeted corpus: full sciq stream, +0.060 (M9' 1a vs v0.3)
  - cosine sharpening: +0.065 acc / +0.027 acc_norm (M9' 1b vs 1a)
Plus a falsifiable prediction: corpus specificity is bidirectional —
an ARC-Challenge+Easy train stream should move ARC (0.268 -> 0.30+),
which would revise the M5'' "ARC ceiling" claim the same way M7's
saturation law was revised.

Dual-teacher design (each lever at its validated configuration):
  - teacher_kl = Qwen3-8B bf16, resident, top-50 KL only (vocab 151936 identical)
  - teacher (geometry) = Qwen3-1.7B FP, hooked 18 layers, cosine only
    (8B hidden is 4096-dim vs student 2048 — cosine needs same-dim anchor)

Data: wiki 3 + sciq 2 + arc 1 + ultra 2 (BATCH 8). ARC train stream:
ARC-C(1119)+ARC-E(2251) train splits, closed-book format mirroring the eval
key ("Question: {q}\\nAnswer: {ans_text}"), concat+window (NO per-example
filter — M7 lesson), ~260 windows recycled ~6 epochs (full pool, moderate
repetition; all the ARC train data that exists).

Expected: sciq 0.53-0.58 if additive; ARC >= 0.30 if corpus targeting works.
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
def md(src): return {"cell_type": MD, "metadata": {}, "source": src}
def code(src): return {"cell_type": CODE, "metadata": {}, "execution_count": None, "outputs": [], "source": src}

# 直接内联模块代码（避免提取转义问题）
_MATH_MODULE = r'''import math
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
    assert w % n == 0
    xb = x.float().reshape(*lead, w // n, n)
    return (fwht(xb * signs.reshape(-1, n)) / math.sqrt(n)).reshape(*lead, w)

def rot_inv(x, signs, n=BLOCK):
    lead, w = x.shape[:-1], x.shape[-1]
    xb = x.float().reshape(*lead, w // n, n)
    return (fwht(xb) / math.sqrt(n) * signs.reshape(-1, n)).reshape(*lead, w)

def quantize_init(wf, g=GROUP):
    out, width = wf.shape
    wg = wf.reshape(out, width // g, g)
    s = (0.836 * wg.std(dim=-1)).clamp(min=1e-8)
    t = torch.round(wg / s.unsqueeze(-1)).clamp(-1, 1).to(torch.int8)
    return t.reshape(out, width), s

w = torch.randn(64, 1024, device=DEV)
s = signs_for_width(1024, DEV)
x = torch.randn(3, 7, 1024, device=DEV)
err = (rot_fwd(x, s) @ rot_fwd(w, s).T - x @ w.T).abs().max()
print(f"rotation identity: {err.item():.2e}")'''

_STE_MODULE = r'''class _TernarySTE(torch.autograd.Function):
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
        return gf * sr * mask, (gf * T).reshape(T.shape[0], -1, GROUP).sum(-1)


class _EmbSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, codes, s):
        ctx.save_for_backward(codes)
        out, width = codes.shape
        w = torch.empty(out, width, dtype=BF, device=codes.device)
        for a in range(0, out, 16384):
            b = min(a + 16384, out)
            w[a:b] = (s[a:b].repeat_interleave(GROUP, dim=1) * codes[a:b].float()).to(BF)
        return w

    @staticmethod
    def backward(ctx, g):
        (codes,) = ctx.saved_tensors
        out, width = codes.shape
        gs = torch.empty(out, width // GROUP, dtype=torch.float32, device=codes.device)
        for a in range(0, out, 16384):
            b = min(a + 16384, out)
            gs[a:b] = (g[a:b].float() * codes[a:b].float()).reshape(b - a, -1, GROUP).sum(-1)
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


TARGETS = [("self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + [("mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]

def install(model):
    dev = next(model.parameters()).device
    emb = model.model.embed_tokens
    new_emb = RotQATEmbedding(emb.weight.data, signs_for_width(emb.weight.shape[1], dev))
    model.model.embed_tokens = new_emb
    model.lm_head = RotQATHead(new_emb)
    linears = []
    for li, layer in enumerate(model.model.layers):
        for ppath, name in TARGETS:
            parent = layer
            for p in ppath.split("."):
                parent = getattr(parent, p)
            lin = getattr(parent, name)
            mod = RotQATLinear(lin, signs_for_width(lin.weight.shape[1], dev))
            mod._hf_name = f"model.layers.{li}.{ppath}.{name}.weight"
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
    return new_emb, linears, tr'''

cells = []
cells.append(md("""# Tritfold M10：三杠杆叠加 + ARC 对症预测

**假设**：三个独立验证过的杠杆可加——
- 教师质量（v0.5：8B 教师 KL，sciq +0.041）
- 对症语料（M9′ 1a：sciq 全量流，+0.060）
- 余弦锐化（M9′ 1b：+0.065 acc / +0.027 acc_norm）

**可证伪预测**：对症性是双向的——ARC 训练集入流应推动 ARC（0.268 → 0.30+），
若成立则 M5″ 的“ARC 天花板”与 M7 饱和定律同样修订。

**双教师分阶段**（每个杠杆跑在已验证配置上；8B 常驻会 OOM，改为 v0.5 验证过的缓存设计）：
- 缓存阶段：`teacher_kl` = Qwen3-8B 常驻 ~20 分钟——预计算四流全部窗口的 top-50（CPU pinned ~1.6GB）+ 顺带出 8B 自身 ARC 参考线，然后**彻底释放**
- 训练阶段：`teacher` = Qwen3-1.7B FP 常驻（18 层 hook 余弦几何锚；8B hidden 4096 维与学生不匹配，几何锚必须同维度）+ KL 目标查表。峰值 ~22GB

**数据**：wiki 3 + sciq 2 + **ARC-train 1**（ARC-C+E 共 3370 条，闭卷格式镜像评估键，concat+开窗绝不逐条过滤）+ ultra 2。

**预期**：可加 → sciq 0.53~0.58（FP 78%+）；ARC ≥0.30。
**运行**：A100 ~2h（缓存 ~25min + 训练 ~1.5h；训练峰值 ~22GB）。"""))

cells.append(code("""%pip -q install --force-reinstall --no-deps "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
%pip -q install "datasets==5.0.1" "accelerate==1.14.0" sentencepiece protobuf bitsandbytes
!test -d /content/fork || git clone -q --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp.git /content/fork
!pip install /content/fork/gguf-py 2>&1 | tail -1
import gguf as _g
from gguf.constants import GGMLQuantizationType as _Q
print("gguf OK | PTQ1_0 =", int(_Q.PTQ1_0))"""))

cells.append(code("""import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch, gc
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DEV, BF = "cuda", torch.bfloat16
print("GPU:", torch.cuda.get_device_name(0))
from google.colab import drive
drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive"
import bitsandbytes as bnb
OPT_CLS = bnb.optim.Adam8bit
from transformers import AutoModelForCausalLM, AutoTokenizer
print("stack OK")"""))

cells.append(code("""MODEL = "Qwen/Qwen3-1.7B"
TEACHER_KL = "Qwen/Qwen3-8B"   # ★ 跨尺寸 KL 教师（vocab 151936 一致，top-50 直接兼容）
ARM = "M10"        # 单臂叠加实验（杠杆可加性 + ARC 对症预测）
STEPS = 1500
BATCH, SEQ, TOPK, CHUNK = 8, 512, 50, 4   # 8B 缓存后释放，训练峰值 ~22GB
LRS = [2e-4, 1e-3, 3e-4]
LAMBDA_FEAT = 0.1   # 余弦损失权重（与 m9p 1b 相同——已验证配置）
HOOK_LAYERS = 18     # hook 前 18/28 层（几何教师 = 1.7B FP，同维度）
EVAL_EVERY, EVAL_WINDOWS = 100, 40
PPL_GUARD = 30.9
N_ULTRA_CONV = 8000  # 对话数据量（维持生成能力）
CKPT = f"/content/m10_{ARM}_qat.pt"
DRIVE_EVERY = 500
BOOT_REPO = "benzeng/tritfold-1.7b-knowledge-ptq1_0"
BOOT_FILE = "tritfold-1.7b-knowledge-ptq1_0.gguf"
GROUP, BLOCK = 128, 1024
SEED = 20260922
# 去污染列表（P0.1 结果：155 个 sciq val 样本与训练语料有 13-gram 重叠）
assert "M10" == ARM"""))

cells.append(md("""## 数学与模块（与 v0.3 一致）"""))
cells.append(code(_MATH_MODULE))
cells.append(code(_STE_MODULE))

cells.append(md("""## 数据 + 去污染"""))

cells.append(code("""import os, hashlib
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

snapshot_download(MODEL)
snapshot_download(TEACHER_KL)                      # ★ 8B KL 教师
hf_hub_download(BOOT_REPO, BOOT_FILE, repo_type="model")
load_dataset("allenai/ai2_arc", "ARC-Easy")        # ★ ARC-E train 流
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
load_dataset("allenai/sciq", split="train")
load_dataset("allenai/sciq", split="validation")
load_dataset("allenai/ai2_arc", "ARC-Challenge")
tok = AutoTokenizer.from_pretrained(MODEL)

def toks(s):
    return tok(s, return_tensors="pt").input_ids[0]

# ---- 去污染：剔除 sciq val 中与 sciq train/wiki-103 有 13-gram 重叠的样本 ----
def ngrams(text, n=13):
    ws = text.lower().split()
    return {" ".join(ws[i:i+n]) for i in range(len(ws)-n+1)} if len(ws)>=n else set()

_train_ng = set()
for ex in load_dataset("allenai/sciq", split="train"):
    _train_ng |= ngrams(ex["support"] + " " + ex["question"])

sciq_val_all = load_dataset("allenai/sciq", split="validation")
CLEAN_IDX = []
for i, ex in enumerate(sciq_val_all):
    if not (ngrams(ex["support"] + " " + ex["question"]) & _train_ng):
        CLEAN_IDX.append(i)
print(f"sciq val 去污染: {len(CLEAN_IDX)}/{len(sciq_val_all)} 保留"
      f"（剔除 {len(sciq_val_all)-len(CLEAN_IDX)} 污染样本）", flush=True)
sciq_val = sciq_val_all.select(CLEAN_IDX)
del _train_ng; gc.collect()

# ---- 语料 + 窗口源（在数据 cell 创建，训练 cell 直接使用） ----
wiki_text = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"] if t.strip())
wiki_all = toks(wiki_text)[:16_777_216]; del wiki_text; gc.collect()
sciq_tr = load_dataset("allenai/sciq", split="train")
s_ids = []
for ex in sciq_tr:
    s_ids.append(toks(f"Question: {ex['question']}\\n{ex['support']}\\nAnswer: {ex['correct_answer']}"))
_all_sciq = torch.cat(s_ids); del s_ids; gc.collect()
q_all = _all_sciq[: (_all_sciq.numel() // SEQ) * SEQ]; del _all_sciq; gc.collect()
print(f"wiki {wiki_all.numel()/1e6:.0f}M | sciq_tr {q_all.numel()/1e6:.2f}M ({q_all.numel()//SEQ} windows)", flush=True)

# ---- ★ ARC 对症流：ARC-C + ARC-E train，闭卷格式镜像评估键，concat+整体开窗 ----
arc_c = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
arc_e = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
a_ids = []
for ex in list(arc_c) + list(arc_e):
    try:
        ai = ex["choices"]["label"].index(ex["answerKey"])
        ans = ex["choices"]["text"][ai]
    except ValueError:
        continue
    a_ids.append(toks(f"Question: {ex['question']}\\nAnswer: {ans}"))
_all_arc = torch.cat(a_ids); del a_ids; gc.collect()
a_all = _all_arc[: (_all_arc.numel() // SEQ) * SEQ]; del _all_arc; gc.collect()
print(f"arc_tr {a_all.numel()/1e3:.0f}K tok ({a_all.numel()//SEQ} windows, 全池复用)", flush=True)

# ultrachat 流（维持生成能力；N_ULTRA_CONV=0 跳过）
import numpy as np
rng = np.random.default_rng(0)
if N_ULTRA_CONV > 0:
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(N_ULTRA_CONV))
    u_ids = []
    for ex in uc:
        text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                       chat_template_kwargs={"enable_thinking": False})
        u_ids.append(toks(text))
    u_all = torch.cat([i[: (i.numel() // SEQ) * SEQ] for i in u_ids if i.numel() >= SEQ])
    del u_ids; gc.collect()
    _u_off = rng.choice(u_all.numel() // SEQ, min(STEPS * 2, u_all.numel() // SEQ), replace=False)
    ultra_src = [u_all[int(o)*SEQ: int(o)*SEQ+SEQ] for o in _u_off]
    del u_all; gc.collect()
else:
    ultra_src = []
print(f"ultra windows: {len(ultra_src)}", flush=True)

# 窗口源（训练 cell 直接引用，不再重复创建）
# rng 已在 ultrachat 块之前创建
N_WIKI, N_SCIQ = STEPS * 3, STEPS * 2
_w_off = rng.choice(wiki_all.numel() // SEQ, min(N_WIKI, wiki_all.numel()//SEQ), replace=False)
wiki_src = [wiki_all[int(o)*SEQ: int(o)*SEQ+SEQ] for o in _w_off]
_q_off = rng.choice(q_all.numel() // SEQ, min(N_SCIQ, q_all.numel()//SEQ), replace=False)
sciq_src = [q_all[int(o)*SEQ: int(o)*SEQ+SEQ] for o in _q_off]
arc_src = [a_all[i*SEQ:(i+1)*SEQ] for i in range(a_all.numel()//SEQ)]
del wiki_all, q_all, a_all; gc.collect()
print(f"windows ready: wiki {len(wiki_src)} | sciq {len(sciq_src)} | arc {len(arc_src)}", flush=True)
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_DATASETS_OFFLINE"] = "1"
text2 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"] if t.strip())
test_ids = toks(text2)
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
_arc_tr_ng = set()
for ex in list(arc_c) + list(arc_e):
    _arc_tr_ng |= ngrams(ex["question"], 8)
_n_ov = sum(1 for ex in arc if ngrams(ex["question"], 8) & _arc_tr_ng)
print(f"ARC train→test 8-gram 近重叠报告: {_n_ov}/{len(arc)}（官方 split 外的近重复量，供判读）", flush=True)
del _arc_tr_ng
print(f"test {test_ids.numel()/1e3:.0f}K | windows ready: wiki {len(wiki_src)} | sciq {len(sciq_src)}", flush=True)"""))

cells.append(md("""## ★ 8B KL 教师：缓存阶段（top-50 预计算 + ARC 参考线，随后彻底释放）"""))

cells.append(code("""import gc, time

teacher_kl = AutoModelForCausalLM.from_pretrained(TEACHER_KL, dtype=BF,
                                                  attn_implementation="sdpa").to(DEV).eval()
print(f"8B teacher loaded (cache phase): {torch.cuda.memory_allocated()/2**20:.0f}MB", flush=True)

# ---- 趁 8B 常驻：先出教师自身的 ARC 参考线（学生若破 1.7B FP 基线，这是正确天花板口径） ----
@torch.no_grad()
def _t_score(prompt, option):
    ids = tok(prompt + " {a}".format(a=option), return_tensors="pt").input_ids[0]
    pl = tok(prompt, return_tensors="pt").input_ids[0].numel()
    x = ids.unsqueeze(0).to(DEV)
    lg = teacher_kl(x).logits.float()[0, pl-1:-1]
    tgt = ids[pl:].to(DEV)
    return (torch.log_softmax(lg, -1)[torch.arange(len(tgt), device=DEV), tgt].sum().item(), len(tgt))

@torch.no_grad()
def _teacher_arc_ref():
    n = acc = accn = 0
    for ex in arc:
        prompt = "Question: {q}\\nAnswer:".format(q=ex["question"]) + "\\n"
        sc = [_t_score(prompt, o) for o in ex["choices"]["text"]]
        acc += max(range(len(sc)), key=lambda i: sc[i][0]) == ex["choices"]["label"].index(ex["answerKey"])
        accn += max(range(len(sc)), key=lambda i: sc[i][0]/sc[i][1]) == ex["choices"]["label"].index(ex["answerKey"])
        n += 1
    print(f"[ARC-c 8B KL-teacher ref] acc {acc/n:.3f} | acc_norm {accn/n:.3f} (n={n})", flush=True)
_teacher_arc_ref()

# ---- 四流 top-50 缓存（CPU pinned，~1.6GB RAM） ----
@torch.no_grad()
def _cache_stream(src, name):
    tvs = torch.empty(len(src), SEQ - 1, TOPK, dtype=torch.float16, pin_memory=True)
    tis = torch.empty(len(src), SEQ - 1, TOPK, dtype=torch.int32, pin_memory=True)
    for c0 in range(0, len(src), 8):
        x = torch.stack(src[c0:c0+8]).to(DEV)
        lg = teacher_kl(x).logits[:, :-1]
        v, i = torch.topk(lg.float(), TOPK, dim=-1)
        tvs[c0:c0+len(v)] = v.half().cpu(); tis[c0:c0+len(i)] = i.cpu()
        if (c0 // 8) % 40 == 0: print(f"  cache {name}: {c0+len(v)}/{len(src)}", flush=True)
    return tvs, tis

_t0 = time.time()
kl_wiki = _cache_stream(wiki_src, "wiki")
kl_sciq = _cache_stream(sciq_src, "sciq")
kl_arc = _cache_stream(arc_src, "arc")
kl_ultra = _cache_stream(ultra_src, "ultra") if len(ultra_src) > 0 else None
KL_SRC = {"w": kl_wiki, "s": kl_sciq, "a": kl_arc, "u": kl_ultra}
print(f"KL cache done in {(time.time()-_t0)/60:.0f} min", flush=True)
del teacher_kl; gc.collect(); torch.cuda.empty_cache()
print(f"8B teacher freed: {torch.cuda.memory_allocated()/2**20:.0f}MB residual", flush=True)"""))

cells.append(md("""## 构建：v0.3 GGUF 自举 + 教师常驻（隐藏态 hook）"""))

cells.append(code("""import time, re
import numpy as np

student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV)
emb, linears, tr = install(student)
student.train()

import os as _os
_CKPT = f"{DRIVE_DIR}/m10_{ARM}_qat_best.pt"
if _os.path.exists(_CKPT):
    ck = torch.load(_CKPT, map_location="cpu")
    with torch.no_grad():
        for m, Z, th in zip(linears, ck["Z"], ck["theta"]):
            m.Z.copy_(Z); m.theta.copy_(th)
        emb.theta.copy_(ck["emb_theta"]); emb.codes.copy_(ck["emb_codes"])
        for n, m_i in student.named_modules():
            if isinstance(m_i, Fp32RMSNorm) and n in ck["islands"]:
                m_i.weight.copy_(ck["islands"][n])
    print(f"resumed m9p[{ARM}]: step {ck['step']}", flush=True)
else:
    import gguf as _gguf
    from huggingface_hub import hf_hub_download as _dl
    if True:
        gguf_path = _dl(BOOT_REPO, BOOT_FILE, repo_type="model")
        _POW3 = [1, 3, 9, 27, 81]
        def _dec(q, n):
            return (((q.astype(np.uint16) * _POW3[n]) % 256) * 3) >> 8
        def _dequant(u8):
            rows, nb = u8.shape[0], u8.shape[1] // 28
            b = u8.reshape(rows, nb, 28)
            d = b[:, :, 26:28].copy().view(np.float16).astype(np.float32)[:, :, 0]
            qs, qh = b[:, :, :24], b[:, :, 24:26]
            out = np.empty((rows, nb, 128), dtype=np.int8)
            for n_ in range(5):
                out[:, :, n_*16:(n_+1)*16] = _dec(qs[:, :, 0:16], n_)
                out[:, :, 80 + n_*8: 80 + (n_+1)*8] = _dec(qs[:, :, 16:24], n_)
            for m_ in range(4):
                out[:, :, 120 + m_*2] = _dec(qh[:, :, 0], m_)
                out[:, :, 120 + m_*2 + 1] = _dec(qh[:, :, 1], m_)
            return ((out.astype(np.float32) - 1.0) * d[:, :, None]).reshape(rows, nb * 128)
        _KIND = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
                 "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
                 "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
        _NORM = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}
        _lin_by_hf = {m_._hf_name: m_ for m_ in linears}
        _mods = dict(student.named_modules())
        rr = _gguf.GGUFReader(gguf_path)
        _n_lin = _n_emb = _n_isl = 0
        with torch.no_grad():
            for t in rr.tensors:
                name = t.name; _ty = int(t.tensor_type)
                if _ty == 143:
                    vals = torch.from_numpy(_dequant(np.asarray(t.data)))
                    if name == "token_embd.weight":
                        wg = vals.reshape(vals.shape[0], -1, 128)
                        emb.codes.copy_(wg.sign().reshape(vals.shape).to(torch.int8))
                        emb.theta.copy_(_inv_softplus(wg.abs().amax(-1).squeeze(-1)))
                        _n_emb += 1
                    elif name != "output.weight":
                        mm = re.match(r"blk\\.(\\d+)\\.(.*)\\.weight", name)
                        mod = _lin_by_hf[f"model.layers.{mm.group(1)}.{_KIND[mm.group(2)]}.weight"]
                        wg = vals.reshape(vals.shape[0], -1, 128)
                        mod.codes0.copy_(wg.sign().reshape(vals.shape).to(torch.int8))
                        mod.theta.copy_(_inv_softplus(wg.abs().amax(-1).squeeze(-1)))
                        mod.Z.copy_(vals)
                        _n_lin += 1
                elif _ty == 0 and "norm" in name:
                    v = torch.from_numpy(np.asarray(t.data).view(np.float32).copy())
                    if name == "output_norm.weight": hf = "model.norm"
                    else:
                        mm = re.match(r"blk\\.(\\d+)\\.(attn_norm|ffn_norm|attn_q_norm|attn_k_norm)\\.weight", name)
                        k = mm.group(2)
                        hf = (f"model.layers.{mm.group(1)}.self_attn.q_norm" if k=="attn_q_norm" else
                              f"model.layers.{mm.group(1)}.self_attn.k_norm" if k=="attn_k_norm" else
                              f"model.layers.{mm.group(1)}.{_NORM[k]}")
                    m_ = _mods.get(hf)
                    if m_ is not None: m_.weight.copy_(v); _n_isl += 1
        print(f"bootstrapped from v0.3 GGUF: {_n_lin}/196 {_n_emb}/1 {_n_isl}/113", flush=True)
        assert _n_lin == 196 and _n_emb == 1 and _n_isl == 113
        ck = {"step": 3000, "hist": [(3000, 26.84)]}
if "rr" in globals(): del rr
gc.collect(); torch.cuda.empty_cache()

# ===== 几何教师常驻（1.7B FP，hook 用；8B 已在缓存阶段释放） =====
teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV).eval()
print(f"geo teacher resident: {torch.cuda.memory_allocated()/2**20:.0f}MB", flush=True)

# ===== 隐藏态 hook（前 HOOK_LAYERS 层） =====
_t_hidden, _s_hidden = {}, {}
def _make_t_hook(li):
    def hook(mod, inp, out):
        _t_hidden[li] = out[0].detach() if isinstance(out, tuple) else out.detach()
    return hook
def _make_s_hook(li):
    def hook(mod, inp, out):
        _s_hidden[li] = out[0].detach() if isinstance(out, tuple) else out.detach()
    return hook

for li in range(HOOK_LAYERS):
    teacher.model.layers[li].register_forward_hook(_make_t_hook(li))
    student.model.layers[li].register_forward_hook(_make_s_hook(li))

def feature_cos_loss():
    \"\"\"OFF: Σ (1 - cos(h_s, h_t)) over hooked layers\"\"\"
    total = torch.tensor(0.0, device=DEV, dtype=torch.float32)
    for li in range(HOOK_LAYERS):
        if li in _t_hidden and li in _s_hidden:
            ht = _t_hidden[li].float().reshape(-1, _t_hidden[li].shape[-1])
            hs = _s_hidden[li].float().reshape(-1, _s_hidden[li].shape[-1])
            cos = F.cosine_similarity(hs, ht, dim=-1).mean()
            total = total + (1.0 - cos)
    return total

@torch.no_grad()
def ppl_of(model, n_windows=EVAL_WINDOWS, seq=SEQ):
    model.eval()
    nll, cnt, skipped = 0.0, 0, 0
    V = model.config.vocab_size
    for w in range(n_windows):
        x = test_ids[w*seq:(w+1)*seq].unsqueeze(0).to(DEV)
        lg = model(x).logits.float()
        v = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction="sum")
        if torch.isfinite(v): nll += v.item(); cnt += x[:, 1:].numel()
        else: skipped += 1
    model.train()
    return (math.exp(nll/cnt) if cnt else float("nan")), skipped

fp_ppl, _ = ppl_of(teacher)
cur_ppl, cur_sk = ppl_of(student)
print(f"FP ppl = {fp_ppl:.2f} | restore = {cur_ppl:.2f} (expect ~26.8)", flush=True)

opt = OPT_CLS([{"params": tr["Z"], "lr": LRS[0]},
               {"params": tr["theta"], "lr": LRS[1]},
               {"params": tr["island"], "lr": LRS[2]}], betas=(0.9, 0.95))
if "opt" in ck:
    opt.load_state_dict(ck["opt"])
    for g, lr in zip(opt.param_groups, LRS): g["lr"] = lr
if "Z" in ck: del ck["Z"], ck["theta"], ck["islands"]
gc.collect()"""))

cells.append(md("""## 训练循环（按 ARM 切换损失）"""))

cells.append(code("""import shutil, math, time

assert 'wiki_src' in dir() and len(wiki_src) > 0, "wiki_src 不存在——数据 cell 未跑"
assert 'sciq_src' in dir() and len(sciq_src) > 0, "sciq_src 不存在——数据 cell 未跑"
assert 'arc_src' in dir() and len(arc_src) > 0, "arc_src 不存在——数据 cell 未跑"

student.train()
last_save = [0]
def save_ckpt(step_i, ppl_i):
    torch.save({"step": step_i, "model": MODEL, "arm": ARM,
                "Z": [m.Z.detach().cpu() for m in linears],
                "theta": [m.theta.detach().cpu() for m in linears],
                "emb_theta": emb.theta.detach().cpu(), "emb_codes": emb.codes.cpu(),
                "islands": {n: m.weight.detach().cpu() for n, m in student.named_modules()
                            if isinstance(m, Fp32RMSNorm)},
                "opt": opt.state_dict(), "hist": hist}, CKPT)
    last_save[0] = step_i
    shutil.copy(CKPT, _CKPT)

hist, best, bad = [], cur_ppl, 0
cos_hist = []
# 断连续训：如果加载了本 ARM 的 checkpoint（非 GGUF 自举），从其 step+1 继续；
# 已完成的臂（step >= STEPS）自动跳过训练（只复评/导出）
_start = 1
if 'ck' in dir() and ck.get('arm') == ARM:
    if ck.get('step', 0) >= STEPS:
        _start = STEPS + 1
        print(f"[{ARM}] checkpoint complete (step {ck['step']} >= {STEPS}) — training skipped", flush=True)
    else:
        _start = ck['step'] + 1
        hist = ck.get('hist', [])
        if _start > 1:
            print(f"resuming training from step {_start} (checkpoint step {ck['step']})", flush=True)
t0 = time.time()
for step in range(_start, STEPS + 1):
    j = step - 1
    ids = ([("w", (j*3+k) % len(wiki_src)) for k in range(3)] +
          [("s", (j*2+k) % len(sciq_src)) for k in range(2)] +
          [("a", j % len(arc_src))])
    if len(ultra_src) > 0:
        ids += [("u", (j*2+k) % len(ultra_src)) for k in range(2)]
    else:
        ids += [("w", (j*5+3+k) % len(wiki_src)) for k in range(2)]
    SRC = {"w": wiki_src, "s": sciq_src, "a": arc_src, "u": ultra_src}
    xs = [SRC[n][i_] for n, i_ in ids]
    opt.zero_grad(set_to_none=True)
    loss_val = 0.0
    for c in range(0, BATCH, CHUNK):
        n = min(CHUNK, BATCH - c)
        x = torch.stack([s for s in xs[c:c+n]]).to(DEV)

        # 几何教师前向填 hook；KL 目标从缓存查表（8B 已释放）
        with torch.no_grad():
            _ = teacher.model(x)                  # 只跑主干填 hook（跳过 151936 维 lm_head）
            tv_c = torch.stack([KL_SRC[n][0][i_].to(DEV, non_blocking=True)
                                for n, i_ in ids[c:c+n]]).float()
            ti_c = torch.stack([KL_SRC[n][1][i_].to(DEV, non_blocking=True)
                                for n, i_ in ids[c:c+n]]).long()

        logits = student(x).logits[:, :-1].float()
        s_logp = F.log_softmax(logits, -1)
        loss_c = -(F.softmax(tv_c, -1) * torch.gather(
            s_logp, -1, ti_c)).sum(-1).mean()
        loss_c = loss_c + LAMBDA_FEAT * feature_cos_loss() / (BATCH // CHUNK)
        (loss_c * n / BATCH).backward()
        del logits, s_logp                       # 1.2GB fp32（graph 已由 backward 释放）
        loss_val += loss_c.item() * n / BATCH

    if math.isfinite(loss_val):
        torch.nn.utils.clip_grad_norm_(tr["Z"] + tr["theta"] + tr["island"], 1.0)
        opt.step()

    # 清理 hook 缓存
    _t_hidden.clear(); _s_hidden.clear()
    torch.cuda.empty_cache()

    if step % EVAL_EVERY == 0 or step == STEPS:
        ppl, skipped = ppl_of(student)
        hist.append((step, ppl))
        extra = f" cos={cos_hist[-1]:.4f}" if ARM == "1b" and cos_hist else ""
        print(f">> [{ARM}] step {step} ppl={ppl:.2f} (best {best:.2f}) skip={skipped} "
              f"loss={loss_val:.3f}{extra} ({step*BATCH*SEQ/(time.time()-t0):.0f} tok/s)", flush=True)
        if not (math.isfinite(ppl) and skipped == 0):
            bad += 1
            if bad >= 8: print("x8 skip: stop"); break
            continue
        if ppl > PPL_GUARD:
            print("guard: stop"); break
        if ppl < best: best = ppl
        if step - last_save[0] >= 500 or step == STEPS:
            save_ckpt(step, ppl)
print("history:", [(h[0], round(h[1],2)) for h in hist], flush=True)"""))

cells.append(md("""## 判卷：去污染 sciq + ARC 双指标 + CI"""))

cells.append(code("""Q, CH = "Question: {q}\\nAnswer:", " {a}"
@torch.no_grad()
def _score(prompt, option):
    ids = tok(prompt + CH.format(a=option), return_tensors="pt").input_ids[0]
    pl = tok(prompt, return_tensors="pt").input_ids[0].numel()
    x = ids.unsqueeze(0).to(DEV)
    lg = student(x).logits.float()[0, pl-1:-1]
    tgt = ids[pl:].to(DEV)
    return (torch.log_softmax(lg, -1)[torch.arange(len(tgt), device=DEV), tgt].sum().item(), len(tgt))

@torch.no_grad()
def probe(data, tag, get_opts, get_gold):
    student.eval()
    n = acc = accn = 0
    for ex in data:
        prompt = Q.format(q=ex["question"]) + "\\n"
        sc = [_score(prompt, o) for o in get_opts(ex)]
        acc += max(range(len(sc)), key=lambda i: sc[i][0]) == get_gold(ex)
        accn += max(range(len(sc)), key=lambda i: sc[i][0]/sc[i][1]) == get_gold(ex)
        n += 1
    student.train()
    se = math.sqrt(0.25/n)  # worst-case SE
    ci = 1.96 * se
    print(f"[{tag}] acc {acc/n:.3f}±{ci:.3f} | acc_norm {accn/n:.3f}±{ci:.3f} (n={n})", flush=True)

# FP 锁定基线：sciq 0.751/0.699, ARC 0.357/0.378
probe(sciq_val, f"sciq clean (FP: 0.751/0.699)",
      lambda ex: [ex["correct_answer"]] + [ex[f"distractor{i}"] for i in (1,2,3)],
      lambda ex: 0)
probe(arc, f"ARC-c (FP: 0.357/0.378)",
      lambda ex: ex["choices"]["text"],
      lambda ex: ex["choices"]["label"].index(ex["answerKey"]))
"""))

cells.append(code(r'''QUESTIONS = ["What planet is known as the Red Planet?",
              "Why do we see lightning before we hear thunder?",
              "What is the powerhouse of the cell?"]
@torch.no_grad()
def ask(q, max_new=96):
    ids = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                  add_generation_prompt=True,
                                  chat_template_kwargs={"enable_thinking": False},
                                  return_tensors="pt").to(DEV)
    out = student.generate(ids, max_new_tokens=max_new, do_sample=True,
                           temperature=0.5, top_p=0.85, top_k=20,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
for q in QUESTIONS:
    print("Q:", q, "\nA:", ask(q), "\n", flush=True)'''))

cells.append(md("""## 导出（可选：发模型时使用）"""))

cells.append(code(r'''import json, shutil
from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

OUT = Path("/content/qwen3-1.7b-ternary-hd-v07"); OUT.mkdir(exist_ok=True)
import os as _os2
_CKPT_SRC = CKPT if _os2.path.exists(CKPT) else _CKPT
print(f"export source: {_CKPT_SRC}")
ck2 = torch.load(_CKPT_SRC, map_location="cpu")
s = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
emb_e, lin_e, _ = install(s)
with torch.no_grad():
    for m, Z, th in zip(lin_e, ck2["Z"], ck2["theta"]):
        m.Z.copy_(Z); m.theta.copy_(th)
    emb_e.theta.copy_(ck2["emb_theta"]); emb_e.codes.copy_(ck2["emb_codes"])
    for n, m_i in s.named_modules():
        if isinstance(m_i, Fp32RMSNorm) and n in ck2["islands"]:
            m_i.weight.copy_(ck2["islands"][n])

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
for n, w_isl in ck2["islands"].items():
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

from transformers import AutoConfig
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
shutil.copy(f"/content/{OUT.name}.zip", f"{DRIVE_DIR}/{OUT.name}.zip")
print(f"zip 已存 Drive：{OUT.name}.zip（从 drive.google.com 下载）")'''))

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "cells": cells}

out = Path("notebooks/tritfold-m10-stack.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
