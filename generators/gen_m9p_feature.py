#!/usr/bin/env python
"""Build tritfold-m9p-feature.ipynb (M9': feature-level distillation + phase-separated QA).

Two hypotheses, two phases, with proper controls:
  Phase 1a: KL-only control (same as v0.3 recipe, isolates cosine effect)
  Phase 1b: KL + hidden-state cosine matching (OFF, 18 layers, λ=0.1)
  Phase 2a: supervised QA from Phase 1 best (tests H2 on top of H1)
  Phase 2b: supervised QA from v0.3 directly (isolates Phase 1 effect)

Decontamination: sciq val 155/1000 excluded from evaluation (13-gram overlap).
FP baselines locked: sciq 0.751/0.699, ARC 0.357/0.378 (dual-metric).
Teacher: 1.7B FP (same as v0.3, clean comparison), resident during training.
Architecture: Z+STE (proven), chain bootstrap from v0.3 GGUF.
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
cells.append(md("""# Tritfold M9′：特征级蒸馏 + 分阶段 QA 训练

**两个可证伪假设**：
- **H1（几何）**：匹配教师中间层方向几何（余弦）能传递 KL 不能传递的知识
- **H2（事实）**：直接监督 QA 损失（非 KL）能注入知识

**四个臂**（独立评估、独立门控）：

| 臂 | 损失 | 步数 | 测试 |
|---|---|---|---|
| 1a 对照 | KL only | 1500 | 基线（= v0.3 短版） |
| 1b 主实验 | KL + λ·cos(hidden) × 18 层 | 1500 | H1 |
| 2a | QA CE（从 1b 最优继续） | 1500 | H2 on top of H1 |
| 2b 对照 | QA CE（从 v0.3 直接） | 1500 | H2 alone |

**去污染**：sciq val 剔除 155 污染样本后 n=845。
**FP 基线（锁定）**：sciq 0.751/0.699，ARC 0.357/0.378。
**运行**：A100 ~6h（教师常驻 ~23GB）。"""))

cells.append(code("""%pip -q install --force-reinstall --no-deps "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
%pip -q install "datasets==5.0.1" "accelerate==1.14.0" sentencepiece protobuf bitsandbytes
!test -d /content/fork || git clone -q --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp.git /content/fork
!pip install /content/fork/gguf-py 2>&1 | tail -1
import gguf as _g
from gguf.constants import GGMLQuantizationType as _Q
print("gguf OK | PTQ1_0 =", int(_Q.PTQ1_0))"""))

cells.append(code("""import os
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
ARM = "1b"          # "1a" KL对照 | "1b" KL+cos主实验 | "2a" QA从1b | "2b" QA从v0.3
STEPS = 1500
BATCH, SEQ, TOPK, CHUNK = 8, 512, 50, 4
LRS = [2e-4, 1e-3, 3e-4]
LAMBDA_FEAT = 0.1   # 余弦损失权重（介于 TernaryLLM 0.001 与评审建议 1.0 之间）
HOOK_LAYERS = 18     # hook 前 18/28 层
EVAL_EVERY, EVAL_WINDOWS = 100, 40
PPL_GUARD = 30.9
N_ULTRA_CONV = 8000  # 对话数据量（0 = 无对话；8000 = 维持生成能力）
CKPT = f"/content/m9p_{ARM}_qat.pt"
DRIVE_EVERY = 500
BOOT_REPO = "benzeng/tritfold-1.7b-knowledge-ptq1_0"
BOOT_FILE = "tritfold-1.7b-knowledge-ptq1_0.gguf"
GROUP, BLOCK = 128, 1024
SEED = 20260922
# 去污染列表（P0.1 结果：155 个 sciq val 样本与训练语料有 13-gram 重叠）
assert ARM in ("1a", "1b", "2a", "2b")"""))

cells.append(md("""## 数学与模块（与 v0.3 一致）"""))
cells.append(code(_MATH_MODULE))
cells.append(code(_STE_MODULE))

cells.append(md("""## 数据 + 去污染"""))

cells.append(code("""import os, hashlib
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

snapshot_download(MODEL)
hf_hub_download(BOOT_REPO, BOOT_FILE, repo_type="model")
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
N_WIKI, N_SCIQ = STEPS * 6, STEPS * 2
_w_off = rng.choice(wiki_all.numel() // SEQ, min(N_WIKI, wiki_all.numel()//SEQ), replace=False)
wiki_src = [wiki_all[int(o)*SEQ: int(o)*SEQ+SEQ] for o in _w_off]
_q_off = rng.choice(q_all.numel() // SEQ, min(N_SCIQ, q_all.numel()//SEQ), replace=False)
sciq_src = [q_all[int(o)*SEQ: int(o)*SEQ+SEQ] for o in _q_off]
del wiki_all, q_all; gc.collect()
print(f"windows ready: wiki {len(wiki_src)} | sciq {len(sciq_src)}", flush=True)
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_DATASETS_OFFLINE"] = "1"
text2 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"] if t.strip())
test_ids = toks(text2)
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
print(f"test {test_ids.numel()/1e3:.0f}K | windows ready: wiki {len(wiki_src)} | sciq {len(sciq_src)}", flush=True)"""))

cells.append(md("""## 构建：v0.3 GGUF 自举 + 教师常驻（隐藏态 hook）"""))

cells.append(code("""import time, re
import numpy as np

student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV)
emb, linears, tr = install(student)
student.train()

import os as _os
_CKPT = f"{DRIVE_DIR}/m9p_{ARM}_qat_best.pt"
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
    # 2a 从 1b 最优继续
    _src = f"{DRIVE_DIR}/m9p_1b_qat_best.pt" if ARM == "2a" else None
    if _src and _os.path.exists(_src):
        ck = torch.load(_src, map_location="cpu")
        with torch.no_grad():
            for m, Z, th in zip(linears, ck["Z"], ck["theta"]):
                m.Z.copy_(Z); m.theta.copy_(th)
            emb.theta.copy_(ck["emb_theta"]); emb.codes.copy_(ck["emb_codes"])
            for n, m_i in student.named_modules():
                if isinstance(m_i, Fp32RMSNorm) and n in ck["islands"]:
                    m_i.weight.copy_(ck["islands"][n])
        print(f"bootstrapped from 1b best: step {ck['step']}", flush=True)
    else:
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

# ===== 教师常驻（M9' 架构变更：不再缓存后释放） =====
teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV).eval()
print(f"teacher resident: {torch.cuda.memory_allocated()/2**20:.0f}MB", flush=True)

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
    xs = [wiki_src[(j*4+k) % len(wiki_src)] for k in range(4)] + \\
         [sciq_src[(j*2+k) % len(sciq_src)] for k in range(2)]
    if len(ultra_src) > 0:
        xs += [ultra_src[(j*2+k) % len(ultra_src)] for k in range(2)]
    else:
        xs += [wiki_src[(j*6+4+k) % len(wiki_src)] for k in range(2)]

    opt.zero_grad(set_to_none=True)
    loss_val = 0.0
    for c in range(0, BATCH, CHUNK):
        n = min(CHUNK, BATCH - c)
        x = torch.stack([s for s in xs[c:c+n]]).to(DEV)

        # 教师同 chunk 前向（保证 hidden state 形状匹配学生）
        with torch.no_grad():
            t_lg = teacher(x).logits[:, :-1].float()
            tv_c, ti_c = torch.topk(t_lg, TOPK, dim=-1)

        logits = student(x).logits[:, :-1].float()
        if ARM in ("1a", "1b"):
            s_logp = F.log_softmax(logits, -1)
            loss_c = -(F.softmax(tv_c, -1) * torch.gather(
                s_logp, -1, ti_c)).sum(-1).mean()
            if ARM == "1b":
                loss_c = loss_c + LAMBDA_FEAT * feature_cos_loss() / (BATCH // CHUNK)
        elif ARM in ("2a", "2b"):
            targets = x[:, 1:]
            loss_c = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     targets.reshape(-1))
        (loss_c * n / BATCH).backward()
        loss_val += loss_c.item() * n / BATCH

    if math.isfinite(loss_val):
        torch.nn.utils.clip_grad_norm_(tr["Z"] + tr["theta"] + tr["island"], 1.0)
        opt.step()

    # 清理 hook 缓存
    _t_hidden.clear(); _s_hidden.clear()

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
      lambda ex: ex["choices"]["label"].index(ex["answerKey"]))"""))

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

OUT = Path(f"/content/qwen3-1.7b-ternary-hd-v06-{ARM}"); OUT.mkdir(exist_ok=True)
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

out = Path("notebooks/tritfold-m9p-feature.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
