#!/usr/bin/env python
"""Build Bonsai/colab/bonsai-v03-knowledge.ipynb (M5'': knowledge track, v0.3).

Thesis: v0.2 "learned to answer, not to know" — this run adds a knowledge-dense
stream (FineWeb-Edu) to attack the ARC gate. Phase 0 measures generative ARC
(generation+letter-parse, the right protocol for instruct models) on FP and the
v0.2 state BEFORE training; Phase 1 trains a 3-stream mix
(FineWeb-Edu 3 + wiki 3 + ultrachat 2 windows/step), bootstrapped from the
published v0.2 GGUF (chained self-bootstrap, zero Drive prereqs).

Pre-registered gates:
  - ARC-c acc_norm (likelihood) >= 0.265   (primary; 70% of FP 0.377)
  - wiki ppl <= 32.0                       (v0.2 final 27.87 x 1.15)
  - EN instruction spot-checks: no regression
All battle lessons baked in (int() enum checks, save basepoints 0, GGUF prefetch
in the online window, fork gguf-py, restore-count assertion).
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

cells.append(md("""# Tritfold v0.3：知识路线（M5″）——teach it to know

**论题**：v0.2 学会了"怎么回答"、没学会"知道答案"（ARC 0.234，低于随机）。本轮加入知识密集语料（FineWeb-Edu，教育评分过滤的网页文本）直接攻 ARC 门。

**两阶段**：
- **Phase 0（~15 分钟）**：生成式 ARC 基线（生成字母再解析——指令模型的正确评估协议），FP 与 v0.2 状态各测一次。若生成式显著高于似然式，"不知道"的结论需要修订；
- **Phase 1（~4.5 小时）**：三流混合蒸馏（FineWeb-Edu 3 窗 + wiki 3 窗 + ultrachat 2 窗 / 步，3000 步），从 **v0.2 HF GGUF 链式自举**（零 Drive 依赖）。

**预注册验收门**：
| 门 | 判据 |
|---|---|
| 主门：ARC-c acc_norm（似然式） | **≥ 0.265**（FP 基线 0.377 的 70%；v0.2 起点 0.234） |
| wiki ppl 护栏 | ≤ 32.0（v0.2 终态 27.87 × 1.15） |
| 英文抽检 | 不回退（Python tips 仍切题） |
| 中文 | 明确出范围（v0.3b 再攻） |

**运行要求**：A100（bf16），~5 小时。"""))

cells.append(code("""%pip -q install --force-reinstall --no-deps "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
%pip -q install "datasets==5.0.1" "accelerate==1.14.0" sentencepiece protobuf bitsandbytes
# gguf 必须用 fork 版（含 Prism 私有类型 142/143；PyPI 版遇到 PTQ1_0 会 ValueError）
!test -d /content/fork || git clone -q --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp.git /content/fork
!pip install /content/fork/gguf-py 2>&1 | tail -1
import gguf as _g
from gguf.constants import GGMLQuantizationType as _Q
print("gguf OK | PTQ1_0 =", int(_Q.PTQ1_0))"""))

cells.append(code("""import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch, gc
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "需要 A100/L4/Ada"
DEV, BF = "cuda", torch.bfloat16
print("GPU:", torch.cuda.get_device_name(0))
from google.colab import drive
drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive"          # 仅存放本阶段训练存档；自举不依赖 Drive
import bitsandbytes as bnb
OPT_CLS = bnb.optim.Adam8bit
from transformers import AutoModelForCausalLM, AutoTokenizer
print("stack OK")"""))

cells.append(code("""MODEL = "Qwen/Qwen3-1.7B"
STEPS = 3000
BATCH, SEQ, TOPK, CHUNK = 8, 512, 50, 4
K_RATIO, W_RATIO, U_RATIO = 3, 3, 2            # FineWeb-Edu / wiki / ultrachat 窗口配比
LRS = [2e-4, 1e-3, 3e-4]
EVAL_EVERY, EVAL_WINDOWS = 100, 40
PPL_GUARD = 32.0                               # v0.2 终态 27.87 × 1.15
ARC_GATE = 0.265                               # 主门（FP 0.377 的 70%）
N_DOCS_K = 6000                                # FineWeb-Edu 文档数（~5M token，取 9000 窗）
N_CONV_U = 20000                               # ultrachat 对话数（取 6000 窗）
CKPT = "/content/v03_qat.pt"
DRIVE_EVERY = 500
BOOT_REPO = "benzeng/tritfold-1.7b-instruct-ptq1_0"   # v0.2 发布物（链式自举）
BOOT_FILE = "tritfold-1.7b-instruct-ptq1_0.gguf"
BOOT_STEP, BOOT_PPL = 3000, 27.87
GROUP, BLOCK = 128, 1024
SEED = 20260922"""))

cells.append(md("""## 数学与模块（与 v0.2 逐行一致）"""))

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

cells.append(md("""## 数据：三流（知识 / wiki / ultrachat）——全部在离线切换前预取"""))

cells.append(code("""import os
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

snapshot_download(MODEL)
hf_hub_download(BOOT_REPO, BOOT_FILE, repo_type="model")       # v0.2 GGUF 自举源
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
load_dataset("allenai/ai2_arc", "ARC-Challenge")
tok = AutoTokenizer.from_pretrained(MODEL)

# 知识流：FineWeb-Edu streaming 必须在切离线之前消费（streaming 无缓存、离线即断）
k_stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                        split="train", streaming=True)
k_chunks = []
n_tok_k = 0
for i, ex in enumerate(k_stream):
    ids = tok(ex["text"], return_tensors="pt").input_ids[0]
    k_chunks.append(ids)
    n_tok_k += ids.numel()
    if (i + 1) % 1500 == 0:
        print(f"  fineweb {i+1}/{N_DOCS_K} ({n_tok_k/1e6:.1f}M tok)", flush=True)
    if i + 1 >= N_DOCS_K:
        break
know_all = torch.cat(k_chunks); del k_chunks; gc.collect()
print(f"knowledge stream: {know_all.numel()/1e6:.1f}M tok", flush=True)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# wiki / ultrachat 流
text103 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"] if t.strip())
wiki_all = tok(text103, return_tensors="pt").input_ids[0][:33_554_432]
del text103; gc.collect()
uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(N_CONV_U))
u_ids = []
for ex in uc:
    text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                   chat_template_kwargs={"enable_thinking": False})
    u_ids.append(tok(text, return_tensors="pt").input_ids[0])
u_all = torch.cat([i[: (i.numel() // SEQ) * SEQ] for i in u_ids if i.numel() >= SEQ])
del u_ids; gc.collect()
print(f"instr stream: {u_all.numel()/1e6:.1f}M tok", flush=True)

text2 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"] if t.strip())
test_ids = tok(text2, return_tensors="pt").input_ids[0]
print(f"test {test_ids.numel()/1e3:.0f}K tok", flush=True)"""))

cells.append(md("""## 构建：v0.2 GGUF 链式自举 + 三缓存"""))

cells.append(code("""import time, re
import numpy as np

student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV)
emb, linears, tr = install(student)
student.train()

import os as _os
V03_CKPT = f"{DRIVE_DIR}/v03_qat_best.pt"
if _os.path.exists(V03_CKPT):                      # 断连续训：优先自身存档
    ck = torch.load(V03_CKPT, map_location="cpu")
    with torch.no_grad():
        for m, Z, th in zip(linears, ck["Z"], ck["theta"]):
            m.Z.copy_(Z); m.theta.copy_(th)
        emb.theta.copy_(ck["emb_theta"]); emb.codes.copy_(ck["emb_codes"])
        for n, m_i in student.named_modules():
            if isinstance(m_i, Fp32RMSNorm) and n in ck["islands"]:
                m_i.weight.copy_(ck["islands"][n])
    print(f"resumed v0.3 ckpt: step {ck['step']}", flush=True)
else:                                              # 首跑：从 v0.2 GGUF 无损自举
    import gguf as _gguf
    from huggingface_hub import hf_hub_download as _dl
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
        vals = (out.astype(np.float32) - 1.0) * d[:, :, None]
        return vals.reshape(rows, nb * 128)
    _KIND = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
             "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
             "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
             "ffn_down": "mlp.down_proj"}
    _NORM = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}
    _lin_by_hf = {m_._hf_name: m_ for m_ in linears}
    _mods = dict(student.named_modules())
    rr = _gguf.GGUFReader(gguf_path)
    _PTQ1_0, _F32 = 143, 0     # int() 判定：Py3.11+ IntEnum str() 返回数字
    _n_lin = _n_emb = _n_isl = 0
    with torch.no_grad():
        for t in rr.tensors:
            name = t.name
            _ty = int(t.tensor_type)
            if _ty == _PTQ1_0:
                vals = torch.from_numpy(_dequant(np.asarray(t.data)))
                if name == "token_embd.weight":
                    wg = vals.reshape(vals.shape[0], -1, 128)
                    emb.codes.copy_(wg.sign().reshape(vals.shape).to(torch.int8))
                    emb.theta.copy_(_inv_softplus(wg.abs().amax(-1).squeeze(-1)))
                    _n_emb += 1
                elif name != "output.weight":
                    mm = re.match(r"blk\.(\d+)\.(.*)\.weight", name)
                    mod = _lin_by_hf[f"model.layers.{mm.group(1)}.{_KIND[mm.group(2)]}.weight"]
                    wg = vals.reshape(vals.shape[0], -1, 128)
                    mod.codes0.copy_(wg.sign().reshape(vals.shape).to(torch.int8))
                    mod.theta.copy_(_inv_softplus(wg.abs().amax(-1).squeeze(-1)))
                    mod.Z.copy_(vals)
                    _n_lin += 1
            elif _ty == _F32 and "norm" in name:
                v = torch.from_numpy(np.asarray(t.data).view(np.float32).copy())
                if name == "output_norm.weight":
                    hf = "model.norm"
                else:
                    mm = re.match(r"blk\.(\d+)\.(attn_norm|ffn_norm|attn_q_norm|attn_k_norm)\.weight", name)
                    k = mm.group(2)
                    hf = (f"model.layers.{mm.group(1)}.self_attn.q_norm" if k == "attn_q_norm" else
                          f"model.layers.{mm.group(1)}.self_attn.k_norm" if k == "attn_k_norm" else
                          f"model.layers.{mm.group(1)}.{_NORM[k]}")
                m_ = _mods.get(hf)
                if m_ is not None:
                    m_.weight.copy_(v); _n_isl += 1
    print(f"bootstrapped from v0.2 GGUF: linears {_n_lin}/196 emb {_n_emb}/1 "
          f"islands {_n_isl}/113", flush=True)
    assert _n_lin == 196 and _n_emb == 1 and _n_isl == 113, "bootstrap incomplete"
    ck = {"step": BOOT_STEP, "hist": [(BOOT_STEP, BOOT_PPL)]}
if "rr" in globals(): del rr
gc.collect(); torch.cuda.empty_cache()

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
print(f"FP 参考 ppl = {fp_ppl:.2f} | 恢复点三值 ppl = {cur_ppl:.2f} (skip {cur_sk})  ← 预期 ≈27.9", flush=True)

# 三缓存
N_K, N_W, N_U = STEPS * K_RATIO, STEPS * W_RATIO, STEPS * U_RATIO
rng = np.random.default_rng(0)
k_off = rng.choice((know_all.numel() - SEQ) // SEQ, N_K, replace=False)
w_off = rng.choice((wiki_all.numel() - SEQ) // SEQ, N_W, replace=False)
u_off = rng.choice(u_all.numel() // SEQ, N_U, replace=False)
k_src = [know_all[int(o)*SEQ: int(o)*SEQ + SEQ] for o in k_off]
w_src = [wiki_all[int(o)*SEQ: int(o)*SEQ + SEQ] for o in w_off]
u_src = [u_all[int(o)*SEQ: int(o)*SEQ + SEQ] for o in u_off]

def build_cache(tag, sources):
    NV = len(sources)
    v_ = np.memmap(f"/content/tv_{tag}.npy", dtype=np.float16, mode="w+", shape=(NV, SEQ - 1, TOPK))
    i_ = np.memmap(f"/content/ti_{tag}.npy", dtype=np.int32,  mode="w+", shape=(NV, SEQ - 1, TOPK))
    t0 = time.time()
    with torch.no_grad():
        for k in range(NV):
            x = sources[k].unsqueeze(0).to(DEV)
            vv, idx = torch.topk(teacher(x).logits[:, :-1].float(), TOPK, -1)
            v_[k] = vv.half().cpu().numpy(); i_[k] = idx.cpu().numpy().astype(np.int32)
            if k % 4000 == 0:
                print(f"  cache[{tag}] {k}/{NV} ({time.time()-t0:.0f}s)", flush=True)
    v_.flush(); i_.flush()
    return v_, i_

vals_k, vids_k = build_cache("k", k_src)
vals_w, vids_w = build_cache("w", w_src)
vals_u, vids_u = build_cache("u", u_src)
del teacher, know_all, wiki_all, u_all; gc.collect(); torch.cuda.empty_cache()
print(f"caches done; {torch.cuda.memory_allocated()/2**20:.0f}MB", flush=True)

opt = OPT_CLS([{"params": tr["Z"], "lr": LRS[0]},
               {"params": tr["theta"], "lr": LRS[1]},
               {"params": tr["island"], "lr": LRS[2]}], betas=(0.9, 0.95))
if "opt" in ck:
    opt.load_state_dict(ck["opt"])
    for g, lr in zip(opt.param_groups, LRS):
        g["lr"] = lr
    print("warm optimizer restored", flush=True)
else:
    print("cold optimizer (GGUF bootstrap)", flush=True)
if "Z" in ck: del ck["Z"], ck["theta"], ck["islands"]
gc.collect()"""))

cells.append(md("""## Phase 0：生成式 ARC 基线（训练前）

生成 4 token → 解析首个 A-D 字母（贪心）。对照似然式（v0.2 = 0.234）——若生成式显著更高，"不知道"的结论需修订。"""))

cells.append(code("""arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

@torch.no_grad()
def arc_gen(model, tag, max_new=4):
    model.eval()
    n = ok = unparsed = 0
    for ex in arc:
        labels = ex["choices"]["label"]
        opts = "\\n".join(f"{l}. {t}" for l, t in zip(labels, ex["choices"]["text"]))
        prompt = f"Question: {ex['question']}\\n{opts}\\nAnswer:"
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        pred = next((c for c in ans.upper() if c in labels), None)
        if pred is None: unparsed += 1
        else: ok += pred == ex["answerKey"]
        n += 1
    print(f"[{tag}] generative ARC: acc {ok/n:.3f} (未解析 {unparsed}/{n})", flush=True)
    return ok / n

t2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                          attn_implementation="sdpa").to(DEV).eval()
arc_gen(t2, "FP 基线")
del t2; gc.collect(); torch.cuda.empty_cache()
arc_gen(student, "v0.2 现状（训练前）")"""))

cells.append(md("""## Phase 1：三流混合训练

保存语义：护栏内每 500 步必存 + 终点必存（基准 0，勿用 ck[step]）。"""))

cells.append(code("""import shutil

student.train()          # Phase 0 的 arc_gen 把模型留在了 eval
last_drive = [0]
last_save = [0]
def save_ckpt(step_i, ppl_i):
    torch.save({"step": step_i, "model": MODEL,
                "Z": [m.Z.detach().cpu() for m in linears],
                "theta": [m.theta.detach().cpu() for m in linears],
                "emb_theta": emb.theta.detach().cpu(), "emb_codes": emb.codes.cpu(),
                "islands": {n: m.weight.detach().cpu() for n, m in student.named_modules()
                            if isinstance(m, Fp32RMSNorm)},
                "opt": opt.state_dict(),
                "hist": hist}, CKPT)
    last_save[0] = step_i
    if step_i - last_drive[0] >= DRIVE_EVERY:
        slim = {k: v for k, v in torch.load(CKPT, map_location="cpu").items() if k != "opt"}
        torch.save(slim, f"{DRIVE_DIR}/v03_qat_best.pt")
        last_drive[0] = step_i
        print(f"  [Drive 同步 @ {step_i}]", flush=True)

hist, best, bad = [], cur_ppl, 0
t0 = time.time()
for step in range(1, STEPS + 1):
    j = step - 1
    def _win(cache_v, cache_i, srcs, j0, ratio):
        tvs = [torch.from_numpy(np.array(cache_v[(j0 + k) % cache_v.shape[0]][None])).to(DEV).float()
               for k in range(ratio)]
        tis = [torch.from_numpy(np.array(cache_i[(j0 + k) % cache_i.shape[0]][None])).to(DEV).long()
               for k in range(ratio)]
        xs = [srcs[(j0 + k) % len(srcs)] for k in range(ratio)]
        return tvs, tis, xs
    tv1, ti1, xs1 = _win(vals_k, vids_k, k_src, j * K_RATIO, K_RATIO)
    tv2, ti2, xs2 = _win(vals_w, vids_w, w_src, j * W_RATIO, W_RATIO)
    tv3, ti3, xs3 = _win(vals_u, vids_u, u_src, j * U_RATIO, U_RATIO)
    tv = torch.cat(tv1 + tv2 + tv3); ti = torch.cat(ti1 + ti2 + ti3)
    xs = xs1 + xs2 + xs3

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
            print("ppl guard breached: stopped", flush=True); break
        if ppl < best:
            best = ppl
        if step - last_save[0] >= 500 or step == STEPS:
            save_ckpt(step, ppl)
print("history:", [(h[0], round(h[1], 2)) for h in hist], flush=True)"""))

cells.append(md("""## Phase 2：判卷（ARC 双协议 + 抽检）→ 导出"""))

cells.append(code("""# ===== ARC 似然式（主门）+ 生成式（训练后） =====
arc_lh = arc_gen(student, "v0.3 生成式（训练后）")

Q, CH = "Question: {q}\\nAnswer:", " {a}"
@torch.no_grad()
def option_logprob(model, prompt, choice):
    ids = tok(prompt + choice, return_tensors="pt").input_ids[0]
    pl = tok(prompt, return_tensors="pt").input_ids[0].numel()
    x = ids.unsqueeze(0).to(DEV)
    lg = model(x).logits.float()[0, pl - 1: -1]
    tgt = ids[pl:].to(DEV)
    return torch.log_softmax(lg, -1)[torch.arange(len(tgt), device=lg.device), tgt].sum().item(), len(tgt)

@torch.no_grad()
def arc_lh_eval(model, tag):
    model.eval()
    n = acc = accn = 0
    for ex in arc:
        prompt = Q.format(q=ex["question"]) + "\\n"
        scores = [option_logprob(model, prompt, CH.format(a=c)) for c in ex["choices"]["text"]]
        labels = ex["choices"]["label"]
        gold = labels.index(ex["answerKey"])
        pred = max(range(len(scores)), key=lambda i: scores[i][0])
        pred_n = max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1])
        acc += pred == gold; accn += pred_n == gold; n += 1
    print(f"[{tag}] ARC likelihood: acc {acc/n:.3f} | acc_norm {accn/n:.3f} "
          f"(门 {ARC_GATE}{' ✅达' if accn/n >= ARC_GATE else ' ❌未达'})", flush=True)

arc_lh_eval(student, "v0.3 似然式（主门）")"""))

cells.append(code("""# ===== 抽检：指令不回退 + 新增科学问题 =====
QUESTIONS = [
    "List two tips for writing better Python code.",        # 指令保持（v0.2 已会）
    "Why do we see lightning before we hear thunder?",      # 知识测试（v0.3 目标）
    "What planet is known as the Red Planet?",              # 知识测试（ARC 风格）
]
@torch.no_grad()
def ask(q, max_new=96):
    msgs = [{"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  chat_template_kwargs={"enable_thinking": False},
                                  return_tensors="pt").to(DEV)
    out = student.generate(ids, max_new_tokens=max_new, do_sample=True,
                           temperature=0.5, top_p=0.85, top_k=20,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

for q in QUESTIONS:
    print("Q:", q, "\\nA:", ask(q), "\\n", flush=True)"""))

cells.append(code("""import json, shutil
from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

OUT = Path("/content/qwen3-1.7b-ternary-hd-v03"); OUT.mkdir(exist_ok=True)
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
# GB 级 zip 的浏览器自动下载不可靠（实测断连源）——落 Drive 为主要通道
try:
    shutil.copy(f"/content/{OUT.name}.zip", f"{DRIVE_DIR}/{OUT.name}.zip")
    print(f"zip 已存 Drive 根目录：{OUT.name}.zip（从 drive.google.com 下载）", flush=True)
except Exception as e:
    print("Drive copy skipped:", e)
try:
    from google.colab import files
    files.download(f"/content/{OUT.name}.zip")
except Exception as e:
    print("browser download skipped:", e)"""))

cells.append(md("""## 附录：chat 修正版生成式 ARC（可选）

v0.3 运行期的实战教训固化：(a) `enable_thinking=False` kwarg 在 apply_chat_template 中不生效，须手动追加闭合 think 块；(b) 字母解析必须严格（逐字符扫描会在 "OK**A**Y" 里抓到 A——FP 假数据 0.224 的来源）；(c) 本模型在 chat 协议下 86% 空输出（空 think 训练伪影触发 EOS）。运行前提：训练 cell 已跑（student 在内存）。"""))

cells.append(code("""import re as _re, time as _time
@torch.no_grad()
def arc_gen_chat(model, tag, max_new=8):
    model.eval()
    n = ok = unparsed = 0
    samples = []
    t0 = _time.time()
    for ex in arc:
        labels = ex["choices"]["label"]
        opts = "\\n".join(f"{l}. {t}" for l, t in zip(labels, ex["choices"]["text"]))
        q = (f"{ex['question']}\\n{opts}\\n\\n"
             "Answer with only the letter of the correct option.")
        ids = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                      add_generation_prompt=True,
                                      return_tensors="pt").to(DEV)
        closer = tok("<think>\\n\\n</think>\\n\\n", add_special_tokens=False,
                     return_tensors="pt").input_ids.to(DEV)
        ids = torch.cat([ids, closer], dim=1)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        ans_clean = ans.replace("</think>", "").strip()
        m1 = _re.search(r"\\b([A-D])\\b", ans_clean.splitlines()[0] if ans_clean else "")
        m2 = _re.search(r"(?:answer|option)\\s*(?:is|:)\\s*\\**([A-D])", ans_clean, _re.I)
        pred = (m2 or m1).group(1).upper() if (m2 or m1) else None
        if pred is None or pred not in labels:
            unparsed += 1
            if len(samples) < 3: samples.append(ans_clean[:60])
        else:
            ok += pred == ex["answerKey"]
        n += 1
        if n % 300 == 0: print(f"  [{tag}] {n}/{len(arc)}", flush=True)
    print(f"[{tag}] chat-gen ARC(修): acc {ok/n:.3f} (未解析 {unparsed}/{n}; "
          f"已解析子集 {ok/max(n-unparsed,1):.3f})", flush=True)
    if samples: print("  未解析样例:", samples, flush=True)

arc_gen_chat(student, "v0.3 终态（chat 修正版）")
import gc as _gc
_t = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                          attn_implementation="sdpa").to(DEV).eval()
arc_gen_chat(_t, "FP 基线（chat 修正版）")
del _t; _gc.collect(); torch.cuda.empty_cache()"""))

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

out = Path("notebooks/tritfold-v03-knowledge.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
