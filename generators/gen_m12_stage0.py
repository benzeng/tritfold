#!/usr/bin/env python
"""Build tritfold-m12-stage0.ipynb (M12 Stage 0: 8B VRAM feasibility smoke).

The VRAM ledger at 8B (7.6B linear params) kills naive configs on A100-40GB:
  Z fp32 30.3GB + grad fp32 30.3GB + Adam8bit 15.2GB = 75.8GB  -> impossible
  codes0 buffer (7.6GB) is unused during training           -> dropped
Configurations under test (50 steps each, cached self-teacher KL targets):
  REF : 1.7B, Z fp32, Adam8bit  — the proven recipe, loss-curve reference
  A   : 8B, Z bf16, Adam8bit    — fits (15.1+15.1+15.2+act ~ 47?) -> actually
        over; with PagedAdam8bit -> ~31GB. RISK: bf16 ulp (~7.8e-3 at |Z|~1)
        swallows lr-scale (2e-4) updates -> possible training stall.
  B   : 8B, Z bf16 GPU compute + CPU fp32 master + CPU Adam (master-weights
        pattern) + GPU Adam8bit for theta/islands -> ~31GB, ~5-7s/step PCIe
        cost. Expected to WORK; the smoke measures speed and loss sanity.
Verdict: loss-drop shape of A/B vs REF + peak VRAM + s/step -> Stage 1 config.
~35-40 min, ~4-5 units.
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
def md(src): return {"cell_type": MD, "metadata": {}, "source": src}
def code(src): return {"cell_type": CODE, "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = []
cells.append(md("""# M12 Stage 0：8B 可行性烟测（VRAM 裁决）

**账本**（7.6B 线性参数 @ A100-40GB）：
- Z fp32 30.3 + 梯度 fp32 30.3 + Adam8bit 15.2 = 75.8GB → **fp32-Z 物理不可行**（梯度与参数同 dtype）
- codes0（7.6GB）训练态不使用 → 砍掉
- 可行域只剩 **bf16-Z**：裸配 45.4GB 仍超，需 PagedAdam8bit（A）或 **CPU fp32 master**（B，master-weights 模式）

**三臂**（各 50 步，自教师缓存 KL 目标）：
| 臂 | 配置 | 检验什么 |
|---|---|---|
| REF | 1.7B，Z fp32，Adam8bit | 已证配方的 loss 曲线参照 |
| A | 8B，Z bf16，PagedAdam8bit | 速度 + bf16 ulp 是否吞掉更新（loss 停滞？） |
| B | 8B，Z bf16 计算 + CPU fp32 master + CPU Adam | 正确性 + PCIe 代价（s/step） |

**判据**：A/B 的 loss 降幅形态 vs REF；峰值 VRAM；s/step。~35-40 min。"""))

cells.append(code("""%pip -q install --force-reinstall --no-deps "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
%pip -q install "datasets==5.0.1" "accelerate==1.14.0" sentencepiece protobuf bitsandbytes
import torch, transformers
print(torch.__version__, transformers.__version__)"""))

cells.append(code(r'''import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK = 1024
_sign_cache = {}
def signs_for_width(width, device, dtype=torch.float32):
    if width not in _sign_cache:
        g = torch.Generator().manual_seed(20260922 + width)
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
    lead = x.shape[:-1]; w = x.shape[-1]
    xb = x.float().reshape(*lead, w // n, n)
    zb = fwht(xb * signs.reshape(-1, n)) / math.sqrt(n)
    return zb.reshape(*lead, w)

def rot_inv(x, signs, n=BLOCK):
    lead = x.shape[:-1]; w = x.shape[-1]
    xb = x.float().reshape(*lead, w // n, n)
    eb = fwht(xb) / math.sqrt(n) * signs.reshape(-1, n)
    return eb.reshape(*lead, w)

GROUP = 128
def quantize_init(w):
    out, width = w.shape
    wg = w.reshape(out, width // GROUP, GROUP)
    s = (0.836 * wg.std(dim=-1)).clamp(min=1e-8)
    t = torch.round(wg / s.unsqueeze(-1)).clamp(-1, 1).to(torch.int8)
    return t.reshape(out, width), s
print("math OK")'''))

cells.append(code(r'''# ===== STE（M12 Stage0 版：Z_DTYPE 可参数化、codes0 可选、backward 适配 Z dtype） =====
Z_DTYPE = torch.float32          # 臂配置覆盖为 bfloat16
KEEP_CODES0 = False              # 训练态不需要（GGUF 自举才用）

class _TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z, s):
        ctx.save_for_backward(Z, s)
        sr = s.repeat_interleave(GROUP, dim=1)
        return (sr * torch.clamp(torch.round(Z.float() / sr), -1, 1)).to(Z.dtype if Z.dtype==torch.bfloat16 else torch.bfloat16)

    @staticmethod
    def backward(ctx, g):
        Z, s = ctx.saved_tensors
        gf = g.float()
        sr = s.repeat_interleave(GROUP, dim=1)
        T = torch.clamp(torch.round(Z.float() / sr), -1, 1)
        mask = ((Z.float() / sr).abs() <= 1.5).to(gf.dtype)
        return (gf * sr * mask).to(Z.dtype), (gf * T).reshape(T.shape[0], -1, GROUP).sum(-1)

class _EmbSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, codes, s):
        ctx.save_for_backward(codes)
        out, width = codes.shape
        w = torch.empty(out, width, dtype=torch.bfloat16, device=codes.device)
        for a in range(0, out, 16384):
            b = min(a + 16384, out)
            w[a:b] = (s[a:b].repeat_interleave(GROUP, dim=1) * codes[a:b].float()).to(torch.bfloat16)
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
            wf = rot_fwd(lin.weight.data.float(), signs)
            codes, s = quantize_init(wf)
        self.register_buffer("signs", signs.float())
        self.Z = nn.Parameter((codes.float() * s.repeat_interleave(GROUP, 1)).to(Z_DTYPE))
        self.theta = nn.Parameter(_inv_softplus(s))
        if KEEP_CODES0:
            self.register_buffer("codes0", codes)
        self.bias = lin.bias

    def scales(self):
        return F.softplus(self.theta).clamp(min=1e-8)

    def w_eff(self):
        return _TernarySTE.apply(self.Z, self.scales())

    def forward(self, x):
        return F.linear(rot_fwd(x, self.signs).to(torch.bfloat16), self.w_eff(), self.bias)

class RotQATHead(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = emb
    def forward(self, h):
        return F.linear(rot_fwd(h, self.emb.signs).to(torch.bfloat16), self.emb.w_eff())

class RotQATEmbedding(nn.Module):
    def __init__(self, table, signs):
        super().__init__()
        with torch.no_grad():
            ef = rot_fwd(table.float(), signs)
            codes, s = quantize_init(ef)
        self.register_buffer("signs", signs.float())
        self.register_buffer("codes", codes)
        self.theta = nn.Parameter(_inv_softplus(s))
    def scales(self):
        return F.softplus(self.theta).clamp(min=1e-8)
    def w_eff(self):
        return _EmbSTE.apply(self.codes, self.scales())
    def forward(self, ids):
        return rot_inv(F.embedding(ids, self.w_eff()), self.signs).to(torch.bfloat16)

class Fp32RMSNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.weight = nn.Parameter(norm.weight.data.float())
        self.eps = getattr(norm, "variance_epsilon", None) or 1e-6
    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * xf).to(x.dtype)

TARGETS = [("self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + \
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
            setattr(parent, name, mod)
            linears.append(mod)
        layer.input_layernorm = Fp32RMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = Fp32RMSNorm(layer.post_attention_layernorm)
        layer.self_attn.q_norm = Fp32RMSNorm(layer.self_attn.q_norm)
        layer.self_attn.k_norm = Fp32RMSNorm(layer.self_attn.k_norm)
        del lin
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
    torch.cuda.empty_cache()
    return new_emb, linears, tr
print("STE OK")'''))

cells.append(code(r'''# ===== 通用三臂驱动：load → 自教师缓存 → install → 按模式建优化器 → 50 步 =====
import time, gc, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = "cuda"
STEPS, BATCH, SEQ, TOPK, CHUNK = 50, 8, 512, 50, 1
LRS = [2e-4, 1e-3, 3e-4]

def _wiki_windows(n, tok):
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i*SEQ:(i+1)*SEQ] for i in range(n)]

def run_arm(name, model_name, z_dtype, opt_mode):
    """opt_mode: 'gpu8bit' | 'gpu8bit_paged' | 'cpu_master'"""
    global Z_DTYPE
    Z_DTYPE = z_dtype
    print(f"\n========== 臂 {name}：{model_name} | Z={z_dtype} | {opt_mode} ==========")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    wins = _wiki_windows(200, tok)
    # ---- 自教师缓存（install 前 = FP 自身）----
    tv = torch.empty(len(wins), SEQ - 1, TOPK, dtype=torch.float16, pin_memory=True)
    ti = torch.empty(len(wins), SEQ - 1, TOPK, dtype=torch.int32, pin_memory=True)
    with torch.no_grad():
        for c0 in range(0, len(wins), 8):
            x = torch.stack(wins[c0:c0+8]).to(DEV)
            lg = model(x).logits[:, :-1].float()
            v, i = torch.topk(lg, TOPK, dim=-1)
            tv[c0:c0+len(v)] = v.half().cpu(); ti[c0:c0+len(i)] = i.cpu()
            del lg, v, i
    torch.cuda.empty_cache()
    print(f"[{name}] cache done {(time.time()-t0)/60:.0f}min, VRAM {torch.cuda.memory_allocated()/2**30:.1f}GB")
    # ---- install ----
    emb, linears, tr = install(model)
    gc.collect(); torch.cuda.empty_cache()
    print(f"[{name}] installed: {len(linears)} linears, VRAM {torch.cuda.memory_allocated()/2**30:.1f}GB")
    # ---- 优化器 ----
    Z_masters, cpu_opt = None, None
    if opt_mode == "gpu8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit([{"params": tr["Z"], "lr": LRS[0]},
                                  {"params": tr["theta"], "lr": LRS[1]},
                                  {"params": tr["island"], "lr": LRS[2]}], betas=(0.9, 0.95))
    elif opt_mode == "gpu8bit_paged":
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdam8bit([{"params": tr["Z"], "lr": LRS[0]},
                                       {"params": tr["theta"], "lr": LRS[1]},
                                       {"params": tr["island"], "lr": LRS[2]}], betas=(0.9, 0.95))
    elif opt_mode == "cpu_master":
        import bitsandbytes as bnb
        # Z: GPU bf16 计算 + CPU fp32 master + CPU Adam；theta/islands: GPU 8bit
        Z_masters = [z.detach().float().cpu() for z in tr["Z"]]
        for zm in Z_masters: zm.requires_grad_(True)
        cpu_opt = torch.optim.Adam(Z_masters, lr=LRS[0], betas=(0.9, 0.95))
        opt = bnb.optim.Adam8bit([{"params": tr["theta"], "lr": LRS[1]},
                                  {"params": tr["island"], "lr": LRS[2]}], betas=(0.9, 0.95))
    model.train()
    # ---- 50 步 ----
    torch.cuda.reset_peak_memory_stats()
    losses = []
    t1 = time.time()
    for step in range(1, STEPS + 1):
        j = (step - 1) % 20
        xs = wins[j*8:(j+1)*8] if j*8+8 <= len(wins) else wins[(j*8) % 190:(j*8) % 190 + 8]
        opt.zero_grad(set_to_none=True)
        if cpu_opt: cpu_opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        for c in range(0, BATCH, CHUNK):
            x = torch.stack(xs[c:c+CHUNK]).to(DEV)
            wi = list(range((j*8+c) % len(wins), (j*8+c) % len(wins) + CHUNK))
            tv_c = torch.stack([tv[k] for k in wi]).to(DEV).float()
            ti_c = torch.stack([ti[k] for k in wi]).to(DEV).long()
            logits = model(x).logits[:, :-1].float()
            s_logp = F.log_softmax(logits, -1)
            loss_c = -(F.softmax(tv_c, -1) * torch.gather(s_logp, -1, ti_c)).sum(-1).mean()
            loss_c.backward()
            loss_val += loss_c.item()
            del logits, s_logp, loss_c
        if cpu_opt:
            with torch.no_grad():
                for z, zm in zip(tr["Z"], Z_masters):
                    zm.grad = z.grad.float().cpu() if z.grad is not None else None
                cpu_opt.step()
                cpu_opt.zero_grad(set_to_none=True)
                for z, zm in zip(tr["Z"], Z_masters):
                    z.copy_(zm.to(DEV, dtype=z.dtype))
                    z.grad = None
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss_val)
        if step % 10 == 0 or step == 1:
            peak = torch.cuda.max_memory_allocated()/2**30
            print(f"[{name}] step {step} loss {loss_val:.4f} | peak {peak:.1f}GB | {step/(time.time()-t1):.2f} it/s", flush=True)
    res = dict(name=name, model=model_name, z_dtype=str(z_dtype), mode=opt_mode,
               losses=[round(l, 4) for l in losses],
               peak_gb=torch.cuda.max_memory_allocated()/2**30,
               sec_per_step=(time.time()-t1)/STEPS,
               total_min=(time.time()-t0)/60)
    print(f"[{name}] RESULT: loss {losses[0]:.3f} -> {losses[-1]:.3f} ({(losses[0]-losses[-1])/losses[0]:.1%} 降), "
          f"peak {res['peak_gb']:.1f}GB, {res['sec_per_step']:.2f}s/step")
    del model, opt, emb, linears, tr, tv, ti
    if cpu_opt: del cpu_opt, Z_masters
    gc.collect(); torch.cuda.empty_cache()
    return res

RESULTS = []
print("driver OK")'''))

cells.append(code(r'''# ===== 臂 1：REF（1.7B，已证配方参照） =====
try:
    RESULTS.append(run_arm("REF-1.7B-fp32Z", "Qwen/Qwen3-1.7B", torch.float32, "gpu8bit"))
except torch.cuda.OutOfMemoryError as e:
    print("臂 REF OOM：{}".format(e), flush=True)
    gc.collect(); torch.cuda.empty_cache()'''))

cells.append(code(r'''# ===== 臂 2：A（8B，Z bf16 + PagedAdam8bit）—— 检验 ulp 吞更新 =====
try:
    RESULTS.append(run_arm("A-8B-bf16Z-paged", "Qwen/Qwen3-8B", torch.bfloat16, "gpu8bit_paged"))
except torch.cuda.OutOfMemoryError as e:
    print("臂 A OOM：{}".format(e), flush=True)
    gc.collect(); torch.cuda.empty_cache()'''))

cells.append(code(r'''# ===== 臂 3：B（8B，Z bf16 计算 + CPU fp32 master）—— 正确性 + 速度代价 =====
try:
    RESULTS.append(run_arm("B-8B-bf16Z-cpumaster", "Qwen/Qwen3-8B", torch.bfloat16, "cpu_master"))
except torch.cuda.OutOfMemoryError as e:
    print("臂 B OOM：{}".format(e), flush=True)
    gc.collect(); torch.cuda.empty_cache()'''))

cells.append(code(r'''# ===== 裁决表 =====
import json
print("| 臂 | loss 首->末 | 降幅 | peak VRAM | s/step | 形态判读 |")
print("|---|---|---|---|---|---|")
ref = next((r for r in RESULTS if r["name"].startswith("REF")), None)
ref_drop = (ref["losses"][0] - ref["losses"][-1]) / ref["losses"][0] if ref else None
for r in RESULTS:
    drop = (r["losses"][0] - r["losses"][-1]) / r["losses"][0]
    verdict = ""
    if r["name"].startswith(("A", "B")) and ref_drop:
        ratio = drop / ref_drop
        verdict = "正常（≈REF 形态）" if ratio >= 0.5 else ("停滞（ulp 吞更新？）" if r["mode"] != "cpu_master" else "异常——需排查")
    print(f"| {r['name']} | {r['losses'][0]:.3f} -> {r['losses'][-1]:.3f} | {drop:.1%} | {r['peak_gb']:.1f}GB | {r['sec_per_step']:.2f}s | {verdict} |")
json.dump(RESULTS, open("/content/m12_stage0_results.json", "w"), indent=1)
try:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    import shutil
    os.makedirs("/content/drive/MyDrive/m12", exist_ok=True)
    shutil.copy("/content/m12_stage0_results.json", "/content/drive/MyDrive/m12/stage0_results.json")
    print("已落盘 Drive/m12/stage0_results.json")
except Exception as e:
    print(f"Drive 不可用（{type(e).__name__}），结果在 /content/m12_stage0_results.json——请手动保存")
print("\n判读规则：A/B 降幅 >= REF 的 50% 且 peak < 38GB → Stage 1 用该配置；A 停滞则用 B；双停滞 → 80GB/4B 决策点")'''))

out = Path("notebooks/tritfold-m12-stage0.ipynb")
nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out}")
