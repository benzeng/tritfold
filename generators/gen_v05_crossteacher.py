#!/usr/bin/env python
"""Build tritfold-v05-crossteacher.ipynb (route A: cross-size teacher distillation).

Hypothesis: v0.3/v0.4's knowledge saturation (~0.40 sciq) may be an artifact of
same-size teaching — the 1.7B teacher's distribution IS the student's ceiling.
Switching to a Qwen3-8B teacher (same vocab: 151936, top-50 indices directly
compatible) provides richer gradients that might push past the saturation line.

Corpus: best-of-all-arms mix (wiki 3 + ultrachat 2 + FineWeb-Edu 2 + sciq 1).
Bootstrap: chained from the v0.3 GGUF (same as v0.4).
Teacher: Qwen3-8B bf16 (~16GB A100 during cache phase, then freed).
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
def md(src): return {"cell_type": MD, "metadata": {}, "source": src}
def code(src): return {"cell_type": CODE, "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = []
cells.append(md("""# Tritfold v0.5 路线 A：跨尺寸教师蒸馏——突破知识饱和的赌注

**假设**：v0.3/v0.4 的知识饱和（sciq ~0.40）可能是同尺寸教师的伪影——1.7B 教师的分布就是学生的上限。换 Qwen3-8B 教师（vocab 完全一致 151936，top-50 直接兼容）可能突破饱和线。

**与 v0.4 的唯一差异**：教师从 Qwen3-1.7B 换为 **Qwen3-8B**。其余全部复用（语料配比取各臂最优混合、链式自举 v0.3 GGUF、评估纪律、战役教训）。

**验收门**：
| 探针 | v0.3 基线 | 目标 | 意义 |
|---|---|---|---|
| **sciq acc_norm（主门）** | 0.398 | **≥0.45** | 突破 = "同尺寸教师"是饱和的隐藏前提 |
| ARC acc_norm | 0.261 | ≥0.28 | 同上 |
| wiki ppl | 26.84 | ≤30.9 | 护栏 |

**资源**：A100 40GB 一个会话（~5h：8B 教师缓存 ~60 分钟 + 训练 3.5h）。"""))

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
print("GPU:", torch.cuda.get_device_name(0),
      "|", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")
from google.colab import drive
drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive"
import bitsandbytes as bnb
OPT_CLS = bnb.optim.Adam8bit
from transformers import AutoModelForCausalLM, AutoTokenizer
print("stack OK")"""))

cells.append(code("""MODEL = "Qwen/Qwen3-1.7B"            # 学生（三值）
TEACHER_MODEL = "Qwen/Qwen3-8B"       # ★ 跨尺寸教师（vocab 151936 一致）
STEPS = 3000
BATCH, SEQ, TOPK, CHUNK = 8, 512, 50, 4
LRS = [2e-4, 1e-3, 3e-4]
EVAL_EVERY, EVAL_WINDOWS = 100, 40
PPL_GUARD = 30.9
SCIQ_GATE = 0.45                       # 主门
ARC_TARGET = 0.28
N_FWED_DOCS = 6000
N_CONV_U = 20000
CKPT = "/content/v05_qat.pt"
DRIVE_EVERY = 500
BOOT_REPO = "benzeng/tritfold-1.7b-knowledge-ptq1_0"
BOOT_FILE = "tritfold-1.7b-knowledge-ptq1_0.gguf"
BOOT_STEP, BOOT_PPL = 3000, 26.84
GROUP, BLOCK = 128, 1024
SEED = 20260922"""))

# 数学与模块复用 v0.3
_v03 = Path("/home/dong/tritfold/generators/gen_v03_knowledge.py").read_text()
import re as _re0
def _extract(anchor):
    i = _v03.index(anchor)
    rest = _v03[i:]
    k0 = rest.index('"""') + 3
    k1 = rest.index('""")', k0)
    return rest[k0:k1].replace("\\\\", "\\")
cells.append(code(_extract("cells.append(code(\"\"\"import math")))
cells.append(code(_extract("cells.append(code(\"\"\"class _TernarySTE")))

cells.append(md("""## 数据：四流最优混合（各臂结论的合成）"""))

cells.append(code("""import os
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

snapshot_download(MODEL)
snapshot_download(TEACHER_MODEL)                        # 8B 教师
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

wiki_text = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"] if t.strip())
wiki_all = toks(wiki_text)[:33_554_432]; del wiki_text; gc.collect()

uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(N_CONV_U))
u_ids = []
for ex in uc:
    text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                   chat_template_kwargs={"enable_thinking": False})
    u_ids.append(toks(text))
u_all = torch.cat([i[: (i.numel() // SEQ) * SEQ] for i in u_ids if i.numel() >= SEQ])
del u_ids; gc.collect()

k_stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
k_ids = []
for i, ex in enumerate(k_stream):
    k_ids.append(toks(ex["text"]))
    if (i + 1) % 2000 == 0: print(f"  fwed {i+1}/{N_FWED_DOCS}", flush=True)
    if i + 1 >= N_FWED_DOCS: break
f_all = torch.cat(k_ids); del k_ids; gc.collect()

sciq_tr = load_dataset("allenai/sciq", split="train")
s_ids = []
for ex in sciq_tr:
    text = f"Question: {ex['question']}\\n{ex['support']}\\nAnswer: {ex['correct_answer']}"
    s_ids.append(toks(text))
q_all = torch.cat([i[: (i.numel() // SEQ) * SEQ] for i in s_ids if i.numel() >= SEQ])
del s_ids; gc.collect()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

text2 = "\\n\\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"] if t.strip())
test_ids = toks(text2)
print(f"wiki {wiki_all.numel()/1e6:.0f}M | u {u_all.numel()/1e6:.0f}M | "
      f"f {f_all.numel()/1e6:.0f}M | q {q_all.numel()/1e6:.1f}M tok", flush=True)"""))

cells.append(md("""## 构建：v0.3 GGUF 自举 + ★ 8B 教师缓存 + 优化器"""))

cells.append(code("""import time, re
import numpy as np

student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV)
emb, linears, tr = install(student)
student.train()

import os as _os
V05_CKPT = f"{DRIVE_DIR}/v05_qat_best.pt"
if _os.path.exists(V05_CKPT):
    ck = torch.load(V05_CKPT, map_location="cpu")
    with torch.no_grad():
        for m, Z, th in zip(linears, ck["Z"], ck["theta"]):
            m.Z.copy_(Z); m.theta.copy_(th)
        emb.theta.copy_(ck["emb_theta"]); emb.codes.copy_(ck["emb_codes"])
        for n, m_i in student.named_modules():
            if isinstance(m_i, Fp32RMSNorm) and n in ck["islands"]:
                m_i.weight.copy_(ck["islands"][n])
    print(f"resumed v0.5 ckpt: step {ck['step']}", flush=True)
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
    print(f"bootstrapped from v0.3 GGUF: {_n_lin}/196 {_n_emb}/1 {_n_isl}/113", flush=True)
    assert _n_lin == 196 and _n_emb == 1 and _n_isl == 113, "bootstrap incomplete"
    ck = {"step": BOOT_STEP, "hist": [(BOOT_STEP, BOOT_PPL)]}
if "rr" in globals(): del rr
gc.collect(); torch.cuda.empty_cache()

# ===== ★ 跨尺寸教师（Qwen3-8B bf16 ~16GB）=====
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL, dtype=BF,
                                               attn_implementation="sdpa").to(DEV).eval()
print(f"8B teacher loaded: {torch.cuda.memory_allocated()/2**20:.0f}MB", flush=True)

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

# 8B 教师的 wiki ppl 参照（预期 ~18-19，好于 1.7B FP 的 20.42）
fp_ppl, _ = ppl_of(teacher)
cur_ppl, cur_sk = ppl_of(student)
print(f"8B 教师 ppl = {fp_ppl:.2f} | 恢复点三值 ppl = {cur_ppl:.2f} (skip {cur_sk})", flush=True)

# 四流缓存（8B 前向 ~5x 慢于 1.7B，~60 分钟）
streams = {"wiki": (3, wiki_all), "ultra": (2, u_all), "fwed": (2, f_all), "sciq": (1, q_all)}
rng = np.random.default_rng(0)
cache = {}
for name, (ratio, alltok) in streams.items():
    NW = STEPS * ratio
    n_pool = alltok.numel() // SEQ
    off = rng.choice(n_pool, min(NW, n_pool), replace=False)
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
            if k % 3000 == 0:
                print(f"  cache[{name}] {k}/{len(src)} ({time.time()-t0:.0f}s)", flush=True)
    v_.flush(); i_.flush()
    cache[name] = (ratio, v_, i_, src)
del teacher, wiki_all, u_all, f_all, q_all; gc.collect(); torch.cuda.empty_cache()
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

# 训练 + 判卷 + 导出（复用 v0.4 模式）
_v04 = Path("/home/dong/tritfold/generators/gen_v04_corpus.py").read_text()
def _ext04(anchor):
    i = _v04.index(anchor)
    rest = _v04[i:]
    k0 = rest.index('"""') + 3
    k1 = rest.index('""")', k0)
    return rest[k0:k1].replace("\\\\", "\\")

cells.append(md("""## 训练（四流混合；保存基准 0）"""))
_train = _ext04("cells.append(code(\"\"\"import shutil")
_train = _train.replace("V04_CKPT", "V05_CKPT").replace("v04_", "v05_").replace("PROFILE", "'main'")
cells.append(code(_train))

cells.append(md("""## 判卷：sciq 主门 + ARC + 抽检 → 导出"""))
cells.append(code("""arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
sciq_val = load_dataset("allenai/sciq", split="validation")
Q, CH = "Question: {q}\\nAnswer:", " {a}"

@torch.no_grad()
def _opt_score(model, prompt, option):
    ids = tok(prompt + CH.format(a=option), return_tensors="pt").input_ids[0]
    pl = tok(prompt, return_tensors="pt").input_ids[0].numel()
    x = ids.unsqueeze(0).to(DEV)
    lg = model(x).logits.float()[0, pl - 1: -1]
    tgt = ids[pl:].to(DEV)
    return (torch.log_softmax(lg, -1)[torch.arange(len(tgt), device=lg.device), tgt].sum().item(), len(tgt))

@torch.no_grad()
def multiway(model, data, tag, get_opts):
    model.eval()
    n = acc = accn = 0
    for ex in data:
        prompt = Q.format(q=ex["question"]) + "\\n"
        scores = [_opt_score(model, prompt, o) for o in get_opts(ex)]
        acc += max(range(len(scores)), key=lambda i: scores[i][0]) == 0
        accn += max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1]) == 0
        n += 1
    print(f"[{tag}] acc {acc/n:.3f} | acc_norm {accn/n:.3f} (n={n})", flush=True)

multiway(student, sciq_val, "sciq val（主门 ≥0.45）",
         lambda ex: [ex["correct_answer"]] + [ex[f"distractor{i}"] for i in (1,2,3)])
multiway(student, arc, "ARC-c（目标 ≥0.28）",
         lambda ex: ex["choices"]["text"])"""))

cells.append(code("""QUESTIONS = ["List two tips for writing better Python code.",
              "Why do we see lightning before we hear thunder?",
              "What planet is known as the Red Planet?"]
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
    print("Q:", q, "\\nA:", ask(q), "\\n", flush=True)"""))

cells.append(code("""import json, shutil
from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

OUT = Path("/content/qwen3-1.7b-ternary-hd-v05"); OUT.mkdir(exist_ok=True)
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

out = Path("notebooks/tritfold-v05-crossteacher.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
