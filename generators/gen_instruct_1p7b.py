#!/usr/bin/env python
"""Build Bonsai/colab/bonsai-instruct-qat-1p7b.ipynb (M5' capability track).

Take the M4 ternary 1.7B checkpoint (wiki-only distillation) and teach it to
follow instructions: mixed-corpus KD (ultrachat 60% + wikitext-103 40%) with
the identical STE stack, plus a self-contained ARC-Challenge mini-harness
(no lm-eval dependency - learned from the dependency whack-a-mole) with an FP
baseline for honest comparison. Wiki ppl is the regression guard (skip==0 gate).

Prereq: Drive root must contain m4_qat_best.pt (the M4 step-4700 checkpoint).
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

cells.append(md("""# M5′：三值 1.7B 能力升级——指令混合蒸馏（A100）

**目标**：M4 checkpoint（wiki-only 蒸馏，1.41×FP）只会续写；本 notebook 用**混合语料 KD**（ultrachat 指令对话 60% + wikitext-103 40%）教它跟随指令，同时用 wiki ppl 作回归护栏（允许回退 ≤0.15×FP），并用自包含 ARC-Challenge 迷你 harness 与 FP 基线诚实对照。

**前提**：A100 运行时；无需任何本地文件——起点从 HF 发布的 GGUF（424MB，无损携带 M4 权重态）自举。若 Drive 上存在后续训练存档则自动优先（断连续训）。

**流程与预算**：数据准备 ~10 分钟 → 双语料教师缓存 ~25 分钟 → 混合训练 3000 步 ~3.5 小时 → ARC 评估（FP + 三值）~40 分钟 → 导出。"""))

cells.append(code("""%pip -q install --force-reinstall --no-deps "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
%pip -q install "datasets==5.0.1" "accelerate==1.14.0" sentencepiece protobuf bitsandbytes
# gguf 必须用 fork 版（含 Prism 私有类型 142/143；PyPI 版遇到 PTQ1_0 会 ValueError）
!test -d /content/fork || git clone -q --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp.git /content/fork
!pip install /content/fork/gguf-py 2>&1 | tail -1      # 非 editable：Colab 上 -e 偶发注册失败
import gguf as _g
from gguf.constants import GGMLQuantizationType as _Q
print("gguf OK:", _g.__file__.split("site-packages/")[-1], "| PTQ1_0 =", _Q.PTQ1_0)
print("installed")"""))

cells.append(code("""import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch, gc
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "需要 A100/L4/Ada"
DEV, BF = "cuda", torch.bfloat16
print("GPU:", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive"          # 仅用于存放本阶段训练存档；起点自举不依赖 Drive

import bitsandbytes as bnb
OPT_CLS = bnb.optim.Adam8bit
from transformers import AutoModelForCausalLM, AutoTokenizer
print("stack OK")"""))

cells.append(code("""MODEL = "Qwen/Qwen3-1.7B"
STEPS = 3000
BATCH, SEQ, TOPK = 8, 512, 50
CHUNK = 4
INSTR_RATIO = 4                      # 每 8 窗中 4 窗指令、4 窗 wiki
LRS = [2e-4, 1e-3, 3e-4]
EVAL_EVERY, EVAL_WINDOWS = 100, 40
PPL_GUARD = 28.77 * 1.15             # wiki ppl 回归护栏（允许 +15%）
N_CONV = 40000                       # ultrachat 取前 4 万轮对话（~25M token）
CKPT = "/content/m5p_qat.pt"
DRIVE_EVERY = 500
GROUP, BLOCK = 128, 1024
SEED = 20260922"""))

cells.append(md("""## 数学与模块（与 M4 逐行一致）"""))

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
print(f"rotation identity: {err.item():.2e}")"""))

cells.append(code("""class _TernarySTE(torch.autograd.Function):
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


TARGETS = [("self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + \\
          [("mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]

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
    return new_emb, linears, tr"""))

cells.append(md("""## 数据：ultrachat（指令，60%）+ wikitext-103（护栏，40%）"""))

cells.append(code("""from datasets import load_dataset

tok = AutoTokenizer.from_pretrained(MODEL)

# 指令流：chat template 应用后切 512 窗（enable_thinking=False 得到更紧凑的非思考风格）
uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
instr_ids = []
for ex in uc.select(range(N_CONV)):
    text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                   chat_template_kwargs={"enable_thinking": False})
    ids = tok(text, return_tensors="pt").input_ids[0]
    instr_ids.append(ids)
    if len(instr_ids) % 8000 == 0:
        print(f"  chat {len(instr_ids)}/{N_CONV}", flush=True)
instr_windows = torch.cat([i[: (i.numel() // SEQ) * SEQ].reshape(-1, SEQ)
                           for i in instr_ids if i.numel() >= SEQ])
print(f"instr windows: {instr_windows.shape[0]} x {SEQ} = "
      f"{instr_windows.shape[0]*SEQ/1e6:.1f}M tok", flush=True)

# 护栏流：wikitext-103（M4 同款）
text103 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"] if t.strip())
wiki_all = tok(text103, return_tensors="pt").input_ids[0]
del text103; gc.collect()

# 评估流：wikitext-2 test（与全部历史口径一致）
text2 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"] if t.strip())
test_ids = tok(text2, return_tensors="pt").input_ids[0]
load_dataset("allenai/ai2_arc", "ARC-Challenge")           # 预取（离线前）
from huggingface_hub import hf_hub_download
hf_hub_download("benzeng/tritfold-1.7b-ptq1_0", "tritfold-1.7b-ptq1_0.gguf",
                repo_type="model")                           # 自举源 GGUF 预取（离线前）
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_DATASETS_OFFLINE"] = "1"
print(f"test {test_ids.numel()/1e3:.0f}K tok", flush=True)"""))

cells.append(md("""## 构建：恢复 M4 checkpoint + 双缓存 + 优化器"""))

cells.append(code("""import time
import numpy as np

student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV)
emb, linears, tr = install(student)
student.train()

import os as _os
M5P_CKPT = f"{DRIVE_DIR}/m5p_qat_best.pt"
if _os.path.exists(M5P_CKPT):                      # 断连续训：优先自身存档
    ck = torch.load(M5P_CKPT, map_location="cpu")
    with torch.no_grad():
        for m, Z, th in zip(linears, ck["Z"], ck["theta"]):
            m.Z.copy_(Z); m.theta.copy_(th)
        emb.theta.copy_(ck["emb_theta"]); emb.codes.copy_(ck["emb_codes"])
        for n, m_i in student.named_modules():
            if isinstance(m_i, Fp32RMSNorm) and n in ck["islands"]:
                m_i.weight.copy_(ck["islands"][n])
    print(f"resumed M5' ckpt: step {ck['step']}", flush=True)
else:                                              # 首跑：从 HF GGUF 无损自举 M4 权重态
    from huggingface_hub import hf_hub_download
    import numpy as _np
    gguf_path = hf_hub_download("benzeng/tritfold-1.7b-ptq1_0",
                                "tritfold-1.7b-ptq1_0.gguf", repo_type="model")
    import gguf as _gguf
    _POW3 = [1, 3, 9, 27, 81]
    def _dec(q, n):   # PTQ1_0 解码：uint8 溢出回绕 == mod 256
        return (((q.astype(_np.uint16) * _POW3[n]) % 256) * 3) >> 8
    def _dequant(u8):
        rows, nb = u8.shape[0], u8.shape[1] // 28
        b = u8.reshape(rows, nb, 28)
        d = b[:, :, 26:28].copy().view(_np.float16).astype(_np.float32)[:, :, 0]
        qs, qh = b[:, :, :24], b[:, :, 24:26]
        out = _np.empty((rows, nb, 128), dtype=_np.int8)
        for n_ in range(5):
            out[:, :, n_*16:(n_+1)*16] = _dec(qs[:, :, 0:16], n_)
            out[:, :, 80 + n_*8: 80 + (n_+1)*8] = _dec(qs[:, :, 16:24], n_)
        for m_ in range(4):
            out[:, :, 120 + m_*2] = _dec(qh[:, :, 0], m_)
            out[:, :, 120 + m_*2 + 1] = _dec(qh[:, :, 1], m_)
        vals = (out.astype(_np.float32) - 1.0) * d[:, :, None]
        return vals.reshape(rows, nb * 128)
    _KIND = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
             "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
             "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
             "ffn_down": "mlp.down_proj"}
    _NORM = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
             "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}
    _lin_by_hf, _islands = {}, {}
    _n_lin = _n_emb = _n_isl = 0
    for m_ in student.modules():
        if isinstance(m_, RotQATLinear):
            _lin_by_hf[m_._hf_name] = m_
    rr = _gguf.GGUFReader(gguf_path)
    _PTQ1_0, _F32 = 143, 0     # int() 判定：Py3.11+ 的 IntEnum str() 返回数字，endswith 恒假
    with torch.no_grad():
        for t in rr.tensors:
            u8 = _np.asarray(t.data); name = t.name
            _ty = int(t.tensor_type)
            if _ty == _PTQ1_0:
                vals = torch.from_numpy(_dequant(u8))
                if name == "token_embd.weight":
                    wg = vals.reshape(vals.shape[0], -1, 128)
                    amax = wg.abs().amax(-1, keepdim=True)
                    emb.codes.copy_(wg.sign().reshape(vals.shape).to(torch.int8))
                    s = amax.squeeze(-1)
                    emb.theta.copy_(_inv_softplus(s))
                    _n_emb += 1
                elif name in ("output.weight",):
                    continue                          # 与 emb 同表，跳过
                else:
                    import re as _re
                    mm = _re.match(r"blk\.(\d+)\.(.*)\.weight", name)
                    hf_name = f"model.layers.{mm.group(1)}.{_KIND[mm.group(2)]}.weight"
                    mod = _lin_by_hf[hf_name]
                    wg = vals.reshape(vals.shape[0], -1, 128)
                    amax = wg.abs().amax(-1, keepdim=True)
                    mod.codes0.copy_((wg.sign() * (wg != 0)).reshape(vals.shape).to(torch.int8))
                    s = amax.squeeze(-1)
                    mod.theta.copy_(_inv_softplus(s))
                    mod.Z.copy_(vals)                 # Z = 折叠值本身（±s/0，吸附后无损）
                    _n_lin += 1
            elif _ty == _F32 and "norm" in name:
                v = torch.from_numpy(u8.view(_np.float32).copy())
                import re as _re
                if name == "output_norm.weight":
                    hf = "model.norm"
                else:
                    mm = _re.match(r"blk\.(\d+)\.(attn_norm|ffn_norm|attn_q_norm|attn_k_norm)\.weight", name)
                    hf = f"model.layers.{mm.group(1)}." + _NORM[mm.group(2)]
                    if mm.group(2).startswith("attn_q"):
                        hf = f"model.layers.{mm.group(1)}.self_attn.q_norm"
                    elif mm.group(2).startswith("attn_k"):
                        hf = f"model.layers.{mm.group(1)}.self_attn.k_norm"
                for n_, m_ in student.named_modules():
                    if n_ == hf and isinstance(m_, Fp32RMSNorm):
                        m_.weight.copy_(v); _n_isl += 1
    ck = {"step": 4700, "hist": [(4700, 28.77)]}     # 冷优化器；无 opt 字段
    print(f"bootstrapped from HF GGUF: linears {_n_lin}/196 emb {_n_emb}/1 "
          f"islands {_n_isl}/113", flush=True)
    assert _n_lin == 196 and _n_emb == 1 and _n_isl == 113, "bootstrap incomplete"

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
        if torch.isfinite(v): nll += v.item(); cnt += x[:, 1:].numel()
        else: skipped += 1
    model.train()
    return (math.exp(nll / cnt) if cnt else float("nan")), skipped

fp_ppl, _ = ppl_of(teacher)
cur_ppl, cur_sk = ppl_of(student)
print(f"FP 参考 ppl = {fp_ppl:.2f} | 恢复点三值 ppl = {cur_ppl:.2f} (skip {cur_sk})", flush=True)

# 双缓存：指令 N_INSTR 窗 + wiki N_WIKI 窗，合计 STEPS×(INSTR_RATIO + BATCH-INSTR_RATIO)
N_INSTR = STEPS * INSTR_RATIO
N_WIKI = STEPS * (BATCH - INSTR_RATIO)
rng = np.random.default_rng(0)
instr_pick = rng.choice(instr_windows.shape[0], min(N_INSTR, instr_windows.shape[0]), replace=False)
wiki_pick = rng.choice((wiki_all.numel() - SEQ) // SEQ, N_WIKI, replace=False)

def build_cache(path_v, path_i, sources):
    NV = len(sources)
    v_ = np.memmap(path_v, dtype=np.float16, mode="w+", shape=(NV, SEQ - 1, TOPK))
    i_ = np.memmap(path_i, dtype=np.int32,  mode="w+", shape=(NV, SEQ - 1, TOPK))
    t0 = time.time()
    with torch.no_grad():
        for k in range(NV):
            x = sources[k].unsqueeze(0).to(DEV)
            vv, idx = torch.topk(teacher(x).logits[:, :-1].float(), TOPK, -1)
            v_[k] = vv.half().cpu().numpy(); i_[k] = idx.cpu().numpy().astype(np.int32)
            if k % 4000 == 0:
                print(f"  cache {k}/{NV} ({time.time()-t0:.0f}s)", flush=True)
    v_.flush(); i_.flush()
    return v_, i_

instr_src = [instr_windows[p] for p in instr_pick]
wiki_src = [wiki_all[int(off) * SEQ: int(off) * SEQ + SEQ] for off in wiki_pick]
print("caching instruction windows...", flush=True)
vals_i, vids_i = build_cache("/content/tv_instr.npy", "/content/ti_instr.npy", instr_src)
print("caching wiki windows...", flush=True)
vals_w, vids_w = build_cache("/content/tv_wiki.npy", "/content/ti_wiki.npy", wiki_src)
del teacher, instr_ids, wiki_all; gc.collect(); torch.cuda.empty_cache()
print(f"caches done; {torch.cuda.memory_allocated()/2**20:.0f}MB", flush=True)

opt = OPT_CLS([{"params": tr["Z"], "lr": LRS[0]},
               {"params": tr["theta"], "lr": LRS[1]},
               {"params": tr["island"], "lr": LRS[2]}], betas=(0.9, 0.95))
if "opt" in ck:
    opt.load_state_dict(ck["opt"])
    for g, lr in zip(opt.param_groups, LRS):     # 防 lr 被旧状态覆盖
        g["lr"] = lr
    print("warm optimizer restored", flush=True)
else:
    print("cold optimizer (GGUF bootstrap)", flush=True)
for g, lr in zip(opt.param_groups, LRS):     # 防 lr 被旧状态覆盖
    g["lr"] = lr
if "Z" in ck: del ck["Z"], ck["theta"], ck["islands"]
gc.collect()"""))

cells.append(md("""## 混合训练

每步 8 窗 = 指令 4 + wiki 4。**护栏语义**：wiki ppl 涨破 `PPL_GUARD`（28.77×1.15）即停——指令能力不许以语言建模崩塌为代价。checkpoint 只在"ppl 达标且创新低"时保存。"""))

cells.append(code("""import shutil

last_drive = [0]        # M5' 步数空间 1..STEPS；基准若用 ck["step"]（如 4700）则区间条件永假
last_save = [0]
def save_ckpt(step_i, ppl_i):
    torch.save({"step": step_i, "model": MODEL,
                "Z": [m.Z.detach().cpu() for m in linears],
                "theta": [m.theta.detach().cpu() for m in linears],
                "emb_theta": emb.theta.detach().cpu(), "emb_codes": emb.codes.cpu(),
                "islands": {n: m.weight.detach().cpu() for n, m in student.named_modules()
                            if isinstance(m, Fp32RMSNorm)},
                "opt": opt.state_dict(),
                "hist": ck["hist"] + [(step_i, ppl_i)]}, CKPT)
    last_save[0] = step_i
    if step_i - last_drive[0] >= DRIVE_EVERY:
        slim = {k: v for k, v in torch.load(CKPT, map_location="cpu").items() if k != "opt"}
        torch.save(slim, f"{DRIVE_DIR}/m5p_qat_best.pt")   # 瘦身版（无优化器状态，冷启动可恢复）
        last_drive[0] = step_i
        print(f"  [Drive 同步 @ {step_i}]", flush=True)

hist, best, bad = [], cur_ppl, 0   # best 仅作展示；本阶段保存语义 = 护栏内定期必存
t0 = time.time()
for step in range(1, STEPS + 1):
    j = step - 1
    tvs = [torch.from_numpy(np.array(vals_i[(j * INSTR_RATIO + k) % vals_i.shape[0]][None])).to(DEV).float()
           for k in range(INSTR_RATIO)]
    tis = [torch.from_numpy(np.array(vids_i[(j * INSTR_RATIO + k) % vids_i.shape[0]][None])).to(DEV).long()
           for k in range(INSTR_RATIO)]
    tws = [torch.from_numpy(np.array(vals_w[(j * (BATCH - INSTR_RATIO) + k) % vals_w.shape[0]][None])).to(DEV).float()
           for k in range(BATCH - INSTR_RATIO)]
    twi = [torch.from_numpy(np.array(vids_w[(j * (BATCH - INSTR_RATIO) + k) % vids_w.shape[0]][None])).to(DEV).long()
           for k in range(BATCH - INSTR_RATIO)]
    xs = [instr_src[(j * INSTR_RATIO + k) % len(instr_src)] for k in range(INSTR_RATIO)] + \\
         [wiki_src[(j * (BATCH - INSTR_RATIO) + k) % len(wiki_src)] for k in range(BATCH - INSTR_RATIO)]
    tv = torch.cat(tvs + tws); ti = torch.cat(tis + twi)

    opt.zero_grad(set_to_none=True)
    loss_val = 0.0
    for c in range(0, BATCH, CHUNK):
        n = min(CHUNK, BATCH - c)
        x = torch.stack([s for s in xs[c:c + n]]).to(DEV)
        logits = student(x).logits[:, :-1].float()
        loss_c = -(F.softmax(tv[c:c + n], -1) * torch.gather(
            F.log_softmax(logits, -1), -1, ti[c:c + n])).sum(-1).mean()
        (loss_c * n / BATCH).backward()
        loss_val += loss_c.item() * n / BATCH
    if math.isfinite(loss_val):
        torch.nn.utils.clip_grad_norm_(tr["Z"] + tr["theta"] + tr["island"], 1.0)
        opt.step()

    if step % EVAL_EVERY == 0 or step == STEPS:
        ppl, skipped = ppl_of(student)
        hist.append((step, ppl))
        print(f">> step {step} wiki-ppl = {ppl:.2f} (guard {PPL_GUARD:.1f}, best {best:.2f}) "
              f"skip {skipped}/{EVAL_WINDOWS} loss {loss_val:.3f} "
              f"({step*BATCH*SEQ/(time.time()-t0):.0f} tok/s)", flush=True)
        if not (math.isfinite(ppl) and skipped == 0):
            bad += 1
            if bad >= 8:
                print("skip/nan x8: stopped", flush=True); break
            continue
        if ppl > PPL_GUARD:
            print("ppl guard breached: stopped（用最近 checkpoint 回退或调低 INSTR_RATIO）", flush=True)
            break
        if ppl < best:
            best = ppl
        # 保存语义：护栏内每 500 步必存 + 结束必存（指令收益不在 ppl 上，不能只认新低）
        if step - last_save[0] >= 500 or step == STEPS:
            save_ckpt(step, ppl)
print("history:", [(h[0], round(h[1], 2)) for h in hist], flush=True)"""))

cells.append(md("""## ARC-Challenge 迷你 harness（自包含，无 lm-eval 依赖）

0-shot 单选项续写似然，报告 acc / acc_norm（长度归一）。先跑 **FP 基线**，再跑**三值当前态**——能力对照的核心数字。"""))

cells.append(code("""# ===== ARC-Challenge 0-shot：acc / acc_norm =====
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
Q, CH = "Question: {q}\\nAnswer:", " {a}"

@torch.no_grad()
def option_logprob(model, prompt, choice):
    ids = tok(prompt + choice, return_tensors="pt").input_ids[0]
    pl = tok(prompt, return_tensors="pt").input_ids[0].numel()
    x = ids.unsqueeze(0).to(DEV)
    lg = model(x).logits.float()[0, pl - 1: -1]           # 预测 choice 各 token
    tgt = ids[pl:].to(DEV)
    lp = torch.log_softmax(lg, -1)[torch.arange(len(tgt), device=lg.device), tgt].sum().item()
    return lp, len(tgt)

@torch.no_grad()
def arc_eval(model, tag, limit=None):
    model.eval()
    n, acc, accn = 0, 0, 0
    items = arc if limit is None else arc.select(range(min(limit, len(arc))))
    for ex in items:
        prompt = Q.format(q=ex["question"]) + "\\n"
        scores = [option_logprob(model, prompt, CH.format(a=c)) for c in ex["choices"]["text"]]
        labels = ex["choices"]["label"]
        gold = labels.index(ex["answerKey"])
        pred = max(range(len(scores)), key=lambda i: scores[i][0])
        pred_n = max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1])
        acc += pred == gold; accn += pred_n == gold; n += 1
    model.train()
    print(f"[{tag}] ARC-Challenge 0-shot: acc {acc/n:.3f} | acc_norm {accn/n:.3f} (n={n})", flush=True)
    return acc / n, accn / n

teacher2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                                attn_implementation="sdpa").to(DEV).eval()
arc_eval(teacher2, "FP 基线")
del teacher2; gc.collect(); torch.cuda.empty_cache()
arc_eval(student, "三值（指令蒸馏后）")"""))

cells.append(md("""## 生成抽检 + 导出

抽检：同一组指令问题，训练前后的回答对比由人工判读；导出与 M4 同构（zip 回本机走收尾链）。"""))

cells.append(code("""# ===== 指令生成抽检 =====
QUESTIONS = [
    "用一句话解释什么是光合作用。",
    "Write a one-sentence summary of the Great Wall of China.",
    "List two tips for writing better Python code.",
]
@torch.no_grad()
def ask(model, q, max_new=96):
    msgs = [{"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  chat_template_kwargs={"enable_thinking": False},
                                  return_tensors="pt").to(DEV)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                         temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

for q in QUESTIONS:
    print("Q:", q)
    print("A:", ask(student, q), "\\n", flush=True)"""))

cells.append(code("""import json, shutil
from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

OUT = Path("/content/qwen3-1.7b-ternary-hd-instruct"); OUT.mkdir(exist_ok=True)
ck2 = torch.load(CKPT, map_location="cpu")
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
try:
    from google.colab import files
    files.download(f"/content/{OUT.name}.zip")
except Exception as e:
    print("download skipped:", e)"""))

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

out = Path("notebooks/tritfold-instruct-1p7b.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
