#!/usr/bin/env python
"""Build tritfold-v04-corpus.ipynb (M7: corpus matrix, v0.4).

Two profiles in one notebook (PROFILE switch in config):
  - "zh":   wiki 3 + ultrachat 2 + Belle-zh-instruct 3   -> Chinese capability
            (pure data gap; deterministic win expected)
  - "know": FineWeb-Edu 5 + wiki 1 + ultrachat 1 + sciq 1 -> knowledge dose ~2x
            + science-QA-targeted stream (the findings-8 bet, scaled to one session)

Chained bootstrap from the published v0.3 GGUF. All battle lessons baked
(int() enum checks, save basepoints 0, GGUF prefetch online, streaming before
offline, strict parsers, output-distribution audits).
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
def md(src): return {"cell_type": MD, "metadata": {}, "source": src}
def code(src): return {"cell_type": CODE, "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = []
cells.append(md("""# Tritfold v0.4：语料矩阵（M7）——zh / know 双 profile

**论题**：语料是配方里最后一个自由变量。两个 profile 各攻一个实测缺口：
- **zh**：中文塌缩是纯数据缺口（英文主导语料）→ 加入 Belle 中文指令流，预期**确定性修复**；
- **know**：知识天花板的可及赌注 → FineWeb-Edu 剂量 ×2（消费 4.6M→7.7M token）+ **sciq 科学问答流**（带 support 段落，直指 ARC 型知识）。

**用法**：config cell 里 `PROFILE = "zh"` 或 `"know"`，一 profile 一跑（每跑 ~4.5h A100）。

**验收门**：
| profile | 主判据 | 护栏 |
|---|---|---|
| zh | 中文三题抽检成句（对照 v0.3 的循环退化）+ EN 不回退 | wiki ppl ≤ 30.9（v0.3 终态 26.84×1.15） |
| know | ARC-c acc_norm > 0.265（超 v0.3）+ sciq 4-way 似然 | 同上 |

链式自举：从 HF 发布的 **v0.3 GGUF** 无损恢复（发布物即训练载体）。"""))

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
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "需要 A100/L4/Ada"
DEV, BF = "cuda", torch.bfloat16
print("GPU:", torch.cuda.get_device_name(0))
from google.colab import drive
drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive"
import bitsandbytes as bnb
OPT_CLS = bnb.optim.Adam8bit
from transformers import AutoModelForCausalLM, AutoTokenizer
print("stack OK")"""))

cells.append(code("""# ===== 本跑的 profile（二选一后运行全部）=====
PROFILE = "zh"          # "zh" 中文修复 | "know" 知识赌注

MODEL = "Qwen/Qwen3-1.7B"
STEPS = 3000
BATCH, SEQ, TOPK, CHUNK = 8, 512, 50, 4
LRS = [2e-4, 1e-3, 3e-4]
EVAL_EVERY, EVAL_WINDOWS = 100, 40
PPL_GUARD = 30.9        # v0.3 终态 26.84 × 1.15
ARC_GATE = 0.265        # know profile 主门（超 v0.3 的 0.261）
N_FWED_DOCS = 9000      # know: FineWeb-Edu 文档（池 ~8M token）
N_BELLE = 60000         # zh: Belle 对话数（池 ~7M token）
CKPT = f"/content/v04_{PROFILE}_qat.pt"
DRIVE_EVERY = 500
BOOT_REPO = "benzeng/tritfold-1.7b-knowledge-ptq1_0"   # v0.3 发布物（链式）
BOOT_FILE = "tritfold-1.7b-knowledge-ptq1_0.gguf"
BOOT_STEP, BOOT_PPL = 3000, 26.84
GROUP, BLOCK = 128, 1024
SEED = 20260922
assert PROFILE in ("zh", "know")"""))

cells.append(md("""## 数学与模块（与 v0.3 逐行一致）"""))

# ---- 数学与模块：与 v0.3 相同，直接复用（生成器中内联） ----
import re as _re0
_src_v03 = Path("/home/dong/tritfold/generators/gen_v03_knowledge.py").read_text()
def extract_cell(anchor):
    i = _src_v03.index(anchor)
    j = _src_v03.index('cells.append(code("""', i + 10)
    block = _src_v03[i:j]
    m = _re0.search(r'cells\.append\(code\("""(.*?)"""\)\)', _src_v03[i:] , _re0.S)
    return m.group(1)

_math = extract_cell("cells.append(code(\"\"\"import math")
_mods = extract_cell("cells.append(code(\"\"\"class _TernarySTE")
# 提取时双反斜杠续行会进入 cell 源码 → ast 报错；替换单续行符
_math = _math.replace("\\\\", "\\")
_mods = _mods.replace("\\\\", "\\")
cells.append(code(_math))
cells.append(code(_mods))

cells.append(md("""## 数据：按 profile 装配三/四流（全部在离线切换前消费 streaming）"""))

cells.append(code("""import os
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

snapshot_download(MODEL)
hf_hub_download(BOOT_REPO, BOOT_FILE, repo_type="model")
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
load_dataset("allenai/ai2_arc", "ARC-Challenge")
load_dataset("allenai/sciq", split="train")
load_dataset("allenai/sciq", split="validation")
tok = AutoTokenizer.from_pretrained(MODEL)

def toks(s):
    return tok(s, return_tensors="pt").input_ids[0]

# ---- 流装配（profile 条件）----
streams = {}       # name -> (ratio, all_tokens)

wiki_text = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"] if t.strip())
wiki_all = toks(wiki_text)[:33_554_432]; del wiki_text; gc.collect()

uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(20000))
u_ids = []
for ex in uc:
    text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                   chat_template_kwargs={"enable_thinking": False})
    u_ids.append(toks(text))
u_all = torch.cat([i[: (i.numel() // SEQ) * SEQ] for i in u_ids if i.numel() >= SEQ])
del u_ids; gc.collect()
print(f"wiki {wiki_all.numel()/1e6:.1f}M | ultra {u_all.numel()/1e6:.1f}M tok", flush=True)

if PROFILE == "zh":
    belle = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)
    b_ids = []
    n_b = 0
    for ex in belle:
        instr = ex["instruction"] + (("\\n" + ex["input"]) if ex.get("input") else "")
        msgs = [{"role": "user", "content": instr}, {"role": "assistant", "content": ex["output"]}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       chat_template_kwargs={"enable_thinking": False})
        b_ids.append(toks(text))
        n_b += 1
        if n_b % 15000 == 0: print(f"  belle {n_b}/{N_BELLE}", flush=True)
        if n_b >= N_BELLE: break
    z_all = torch.cat([i[: (i.numel() // SEQ) * SEQ] for i in b_ids if i.numel() >= SEQ])
    del b_ids; gc.collect()
    print(f"belle(zh) {z_all.numel()/1e6:.1f}M tok", flush=True)
    streams = {"wiki": (3, wiki_all), "ultra": (2, u_all), "belle": (3, z_all)}
else:  # know
    k_stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                            split="train", streaming=True)
    k_ids = []
    for i, ex in enumerate(k_stream):
        k_ids.append(toks(ex["text"]))
        if (i + 1) % 2000 == 0: print(f"  fwed {i+1}/{N_FWED_DOCS}", flush=True)
        if i + 1 >= N_FWED_DOCS: break
    f_all = torch.cat(k_ids); del k_ids; gc.collect()
    print(f"fwed {f_all.numel()/1e6:.1f}M tok", flush=True)
    sciq = load_dataset("allenai/sciq", split="train")
    s_ids = []
    for ex in sciq:
        text = (f"Question: {ex['question']}\\n"
                f"{ex['support']}\\n"
                f"Answer: {ex['correct_answer']}")
        s_ids.append(toks(text))
    q_all = torch.cat([i[: (i.numel() // SEQ) * SEQ] for i in s_ids if i.numel() >= SEQ])
    del s_ids; gc.collect()
    print(f"sciq {q_all.numel()/1e6:.2f}M tok（小池循环使用）", flush=True)
    streams = {"fwed": (5, f_all), "wiki": (1, wiki_all), "ultra": (1, u_all), "sciq": (1, q_all)}

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

text2 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"] if t.strip())
test_ids = toks(text2)
print(f"test {test_ids.numel()/1e3:.0f}K tok", flush=True)"""))

cells.append(md("""## 构建：v0.3 GGUF 链式自举 + 多流缓存"""))

cells.append(code("""import time, re
import numpy as np

student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV)
emb, linears, tr = install(student)
student.train()

import os as _os
V04_CKPT = f"{DRIVE_DIR}/v04_{PROFILE}_qat_best.pt"
if _os.path.exists(V04_CKPT):
    ck = torch.load(V04_CKPT, map_location="cpu")
    with torch.no_grad():
        for m, Z, th in zip(linears, ck["Z"], ck["theta"]):
            m.Z.copy_(Z); m.theta.copy_(th)
        emb.theta.copy_(ck["emb_theta"]); emb.codes.copy_(ck["emb_codes"])
        for n, m_i in student.named_modules():
            if isinstance(m_i, Fp32RMSNorm) and n in ck["islands"]:
                m_i.weight.copy_(ck["islands"][n])
    print(f"resumed v0.4[{PROFILE}] ckpt: step {ck['step']}", flush=True)
else:
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
        return ((out.astype(np.float32) - 1.0) * d[:, :, None]).reshape(rows, nb * 128)
    _KIND = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
             "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
             "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
             "ffn_down": "mlp.down_proj"}
    _NORM = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}
    _lin_by_hf = {m_._hf_name: m_ for m_ in linears}
    _mods = dict(student.named_modules())
    rr = _gguf.GGUFReader(gguf_path)
    _PTQ1_0, _F32 = 143, 0
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
                    mm = re.match(r"blk\\.(\\d+)\\.(.*)\\.weight", name)
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
                    mm = re.match(r"blk\\.(\\d+)\\.(attn_norm|ffn_norm|attn_q_norm|attn_k_norm)\\.weight", name)
                    k = mm.group(2)
                    hf = (f"model.layers.{mm.group(1)}.self_attn.q_norm" if k == "attn_q_norm" else
                          f"model.layers.{mm.group(1)}.self_attn.k_norm" if k == "attn_k_norm" else
                          f"model.layers.{mm.group(1)}.{_NORM[k]}")
                m_ = _mods.get(hf)
                if m_ is not None:
                    m_.weight.copy_(v); _n_isl += 1
    print(f"bootstrapped from v0.3 GGUF: linears {_n_lin}/196 emb {_n_emb}/1 "
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
print(f"FP 参考 ppl = {fp_ppl:.2f} | 恢复点三值 ppl = {cur_ppl:.2f} (skip {cur_sk})  ← 预期 ≈26.8", flush=True)

# 多流缓存
rng = np.random.default_rng(0)
cache = {}
for name, (ratio, alltok) in streams.items():
    NW = STEPS * ratio
    n_win_pool = (alltok.numel() - SEQ) // SEQ if name != "sciq" else alltok.numel() // SEQ
    off = rng.choice(n_win_pool, min(NW, n_win_pool), replace=n_win_pool < NW) \\
        if n_win_pool < NW else rng.choice(n_win_pool, NW, replace=False)
    src = [alltok[int(o)*SEQ: int(o)*SEQ + SEQ] for o in off]
    v_ = np.memmap(f"/content/tv_{name}.npy", dtype=np.float16, mode="w+",
                   shape=(len(src), SEQ - 1, TOPK))
    i_ = np.memmap(f"/content/ti_{name}.npy", dtype=np.int32, mode="w+",
                   shape=(len(src), SEQ - 1, TOPK))
    t0 = time.time()
    with torch.no_grad():
        for k in range(len(src)):
            x = src[k].unsqueeze(0).to(DEV)
            vv, idx = torch.topk(teacher(x).logits[:, :-1].float(), TOPK, -1)
            v_[k] = vv.half().cpu().numpy(); i_[k] = idx.cpu().numpy().astype(np.int32)
            if k % 4000 == 0:
                print(f"  cache[{name}] {k}/{len(src)} ({time.time()-t0:.0f}s)", flush=True)
    v_.flush(); i_.flush()
    cache[name] = (ratio, v_, i_, src)
del teacher, wiki_all, u_all
for _n in ("f_all", "z_all", "q_all"):
    if _n in globals(): del globals()[_n]
gc.collect(); torch.cuda.empty_cache()
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

cells.append(md("""## 训练（多流混合；保存基准 0）"""))

cells.append(code("""import shutil

student.train()
last_drive = [0]
last_save = [0]
def save_ckpt(step_i, ppl_i):
    torch.save({"step": step_i, "model": MODEL, "profile": PROFILE,
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
        torch.save(slim, V04_CKPT)
        last_drive[0] = step_i
        print(f"  [Drive 同步 @ {step_i}]", flush=True)

_names = list(cache.keys())
_ratios = [cache[n][0] for n in _names]
hist, best, bad = [], cur_ppl, 0
t0 = time.time()
for step in range(1, STEPS + 1):
    j = step - 1
    tvs, tis, xs = [], [], []
    for name, ratio in zip(_names, _ratios):
        _, v_, i_, src = cache[name]
        j0 = j * ratio
        for k in range(ratio):
            tvs.append(torch.from_numpy(np.array(v_[(j0 + k) % v_.shape[0]][None])).to(DEV).float())
            tis.append(torch.from_numpy(np.array(i_[(j0 + k) % i_.shape[0]][None])).to(DEV).long())
            xs.append(src[(j0 + k) % len(src)])
    tv = torch.cat(tvs); ti = torch.cat(tis)

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

cells.append(md("""## 判卷：ARC + sciq 似然 + 双语抽检 → 导出"""))

cells.append(code("""arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
sciq_val = load_dataset("allenai/sciq", split="validation")

Q, CH = "Question: {q}\\nAnswer:", " {a}"
@torch.no_grad()
def fourway_lh(model, data, tag, qfield, gold, distr, support=None):
    model.eval()
    n = acc = accn = 0
    for ex in data:
        opts = [ex[gold]] + [ex[d] for d in distr]
        prompt = Q.format(q=ex[qfield]) + "\\n"
        if support and ex.get(support):
            prompt = f"{ex[support]}\\n" + prompt
        scores = []
        for o in opts:
            ids = tok(prompt + CH.format(a=o), return_tensors="pt").input_ids[0]
            pl = tok(prompt, return_tensors="pt").input_ids[0].numel()
            x = ids.unsqueeze(0).to(DEV)
            lg = model(x).logits.float()[0, pl - 1: -1]
            tgt = ids[pl:].to(DEV)
            scores.append((torch.log_softmax(lg, -1)[torch.arange(len(tgt), device=lg.device), tgt].sum().item(),
                           len(tgt)))
        pred = max(range(4), key=lambda i: scores[i][0])
        pred_n = max(range(4), key=lambda i: scores[i][0] / scores[i][1])
        acc += pred == 0; accn += pred_n == 0; n += 1
    print(f"[{tag}] acc {acc/n:.3f} | acc_norm {accn/n:.3f} (n={n})", flush=True)

# ARC 结构与 sciq 不同（变长选项），单独写
@torch.no_grad()
def arc_lh(model, tag):
    model.eval()
    n = acc = accn = 0
    for ex in arc:
        prompt = Q.format(q=ex["question"]) + "\\n"
        scores = []
        for o in ex["choices"]["text"]:
            ids = tok(prompt + CH.format(a=o), return_tensors="pt").input_ids[0]
            pl = tok(prompt, return_tensors="pt").input_ids[0].numel()
            x = ids.unsqueeze(0).to(DEV)
            lg = model(x).logits.float()[0, pl - 1: -1]
            tgt = ids[pl:].to(DEV)
            scores.append((torch.log_softmax(lg, -1)[torch.arange(len(tgt), device=lg.device), tgt].sum().item(),
                           len(tgt)))
        labels = ex["choices"]["label"]
        gold = labels.index(ex["answerKey"])
        acc += max(range(len(scores)), key=lambda i: scores[i][0]) == gold
        accn += max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1]) == gold
        n += 1
    gate = "✅超门" if accn / n > ARC_GATE else "❌未超"
    print(f"[{tag}] acc {acc/n:.3f} | acc_norm {accn/n:.3f} (门 {ARC_GATE} {gate})", flush=True)

arc_lh(student, "ARC-c（v0.4 终态）")
fourway_lh(student, sciq_val, "sciq val（知识探针，随机 0.25）",
           "question", "correct_answer", ["distractor1", "distractor2", "distractor3"])"""))

cells.append(code("""# ===== 双语抽检（zh profile 主判据）=====
QUESTIONS = {"zh": ["用一句话解释什么是光合作用。",
                    "用一句话介绍长城的历史。",
                    "写两句关于人工智能的中文。"],
             "know": ["List two tips for writing better Python code.",
                      "Why do we see lightning before we hear thunder?",
                      "What planet is known as the Red Planet?"]}
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

for q in QUESTIONS[PROFILE]:
    print("Q:", q, "\\nA:", ask(q), "\\n", flush=True)"""))

cells.append(code("""import json, shutil
from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

OUT = Path(f"/content/qwen3-1.7b-ternary-hd-v04-{PROFILE}"); OUT.mkdir(exist_ok=True)
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
    shutil.copy(f"/content/{OUT.name}.zip", f"{DRIVE_DIR}/{OUT.name}.zip")
    print(f"zip 已存 Drive：{OUT.name}.zip", flush=True)
except Exception as e:
    print("Drive copy skipped:", e)"""))

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "cells": cells}

out = Path("notebooks/tritfold-v04-corpus.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
