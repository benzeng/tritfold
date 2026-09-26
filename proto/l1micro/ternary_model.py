"""L1-micro training modules: ternary codes + trainable scales in the rotated basis.

Design (E2-lite, see plan T3.3 and bonsai-l0-findings.md §5):
  - Every folded weight lives in the rotated basis as int8 codes T ∈ {-1,0,+1}
    plus one fp32 scale per group of 128 (input axis) — the artifact's format.
  - Forward: w_eff = s_g · T, activation rotated online (rot_fwd). STE not needed
    for codes (no latent); scales get exact gradients (d w_eff/d s = T).
  - P-step: Adam on all scales (lr 3e-3) + fp32 norm islands (lr 3e-4).
  - V-step: flip codes of top-|grad| entries one level toward the descent
    direction, per-step global trust ratio ||ΔW||/||W|| <= 0.01.
  - Embedding/lm_head share one folded table (tied contract): the same w_eff
    tensor serves lookup (rot_inv after) and logits (rot_fwd before), so the
    dense lm_head gradient trains the table's scales and drives its V-step.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common.fwht_torch import GROUP, rot_fwd, rot_inv, signs_for_width


def quantize_init(wf: torch.Tensor, init: str, g: int = GROUP):
    """wf: folded weights (out, in) fp32. Returns (codes int8, scales fp32)."""
    out, width = wf.shape
    wg = wf.reshape(out, width // g, g)
    if init == "amax":
        s = wg.abs().amax(dim=-1)
    elif init == "zerofrac":
        # target ~1/3 zeros under RTN: threshold 0.5*s = 0.418*std -> s = 0.836*std
        s = 0.836 * wg.std(dim=-1)
    else:
        raise ValueError(init)
    s = s.clamp(min=1e-8)
    t = torch.round(wg / s.unsqueeze(-1)).clamp(-1, 1).to(torch.int8)
    return t.reshape(out, width), s


class _TernaryMul(torch.autograd.Function):
    """w = softplus(theta)[group] * codes, saving only int8 codes + theta.

    Naive fp32 repeat_interleave*float() keeps two fp32 operand copies per folded
    tensor in the autograd graph (~5.4GB for 0.6B) and pages host RAM on WSL;
    this Function's saved state is just the codes buffer (shared, no copy).
    """

    @staticmethod
    def forward(ctx, theta, codes):
        s = F.softplus(theta).clamp(min=1e-8)
        w = s.repeat_interleave(GROUP, dim=1) * codes.float()
        ctx.save_for_backward(theta, codes)
        return w.to(torch.float16)

    @staticmethod
    def backward(ctx, gout):
        theta, codes = ctx.saved_tensors
        out, width = codes.shape
        g = gout.float().reshape(out, width // GROUP, GROUP)
        t = codes.float().reshape(out, width // GROUP, GROUP)
        gs = (g * t).sum(-1)                    # dL/ds per group
        return gs * torch.sigmoid(theta), None  # softplus'(theta) = sigmoid(theta)


class TernaryWeight(nn.Module):
    """codes (int8) + trainable group scales; w_eff built per forward."""

    def __init__(self, w_fp32: torch.Tensor, init: str, capture_grad: bool = True):
        super().__init__()
        codes, scales = quantize_init(w_fp32, init)
        self.register_buffer("codes", codes)                      # (out, in) int8
        self.theta = nn.Parameter(self._inv_softplus(scales))     # (out, in//128)
        self.capture_grad = capture_grad
        self.ema = None        # fp16 |grad| EMA for V-step ranking (lazy)
        self._w_eff = None

    @staticmethod
    def _inv_softplus(s):
        return torch.log(torch.expm1(s.clamp(min=1e-6))).clamp(min=-12)

    def scales(self):
        return F.softplus(self.theta).clamp(min=1e-8)

    def w_eff(self) -> torch.Tensor:
        w = _TernaryMul.apply(self.theta, self.codes)
        if torch.is_grad_enabled() and self.capture_grad:
            w.retain_grad()
        self._w_eff = w
        return w

    @torch.no_grad()
    def grad_scores(self) -> torch.Tensor | None:
        g = self._w_eff.grad if self._w_eff is not None else None
        return g.abs().float() if g is not None else None

    @torch.no_grad()
    def update_ema(self, beta: float = 0.9):
        """EMA of the SIGNED grad — one buffer serves both V-step ranking
        (abs) and flip direction (-sign); mixing EMA ranking with per-sample
        direction was incoherent and degraded ppl (91.6 -> 308 @ step 50)."""
        g = self._w_eff.grad if self._w_eff is not None else None
        if g is None:
            return
        a = g.half()
        if self.ema is None:
            self.ema = a
        else:
            self.ema.mul_(beta).add_(a, alpha=1 - beta)

    @torch.no_grad()
    def vstep_flip(self, idx_flat: torch.Tensor, direction: torch.Tensor | None = None):
        """Flip selected codes one level toward the descent direction.
        direction: signed score source (EMA or grad); defaults to current grad."""
        if direction is None:
            g = self._w_eff.grad
            if g is None or idx_flat.numel() == 0:
                return 0
            direction = g
        dir_flat = direction.reshape(-1)[idx_flat]
        step = torch.sign(-dir_flat).to(torch.int8)       # descent direction
        codes = self.codes.reshape(-1)
        new = (codes[idx_flat] + step).clamp(-1, 1)
        moved = (new != codes[idx_flat])
        codes[idx_flat] = new
        return int(moved.sum().item())

    @torch.no_grad()
    def norm_and_scale_stats(self):
        w = self.scales().repeat_interleave(GROUP, dim=1) * self.codes.float()
        return w.norm().item(), self.scales().mean().item()

    @torch.no_grad()
    def zero_frac(self) -> float:
        return (self.codes == 0).float().mean().item()


class RotTrainLinear(nn.Module):
    def __init__(self, lin: nn.Linear, signs: torch.Tensor, init: str):
        super().__init__()
        with torch.no_grad():
            wf = rot_fwd(lin.weight.data, signs)
        self.tw = TernaryWeight(wf, init)
        self.bias = lin.bias
        self.register_buffer("signs", signs.float())

    def forward(self, x):
        z = rot_fwd(x, self.signs).to(torch.float16)
        return F.linear(z, self.tw.w_eff(), self.bias)


class RotTrainEmbedding(nn.Module):
    """Folded table; inverse-after-lookup. Shares w_eff with the head below.
    capture_grad=False: the fp32 grad of this 155M-param table costs >0.6GB and
    WSL host-memory paging; its codes stay at init, its scales still train
    (gradient flows to theta through w_eff regardless of retain_grad)."""

    def __init__(self, emb_weight: torch.Tensor, signs: torch.Tensor, init: str):
        super().__init__()
        with torch.no_grad():
            ef = rot_fwd(emb_weight.data, signs)
        self.tw = TernaryWeight(ef, init, capture_grad=False)
        self.register_buffer("signs", signs.float())

    def forward(self, ids):
        e = F.embedding(ids, self.tw.w_eff())
        return rot_inv(e, self.signs).to(torch.float16)


class RotTrainHead(nn.Module):
    def __init__(self, emb: RotTrainEmbedding):
        super().__init__()
        self.emb = emb

    def forward(self, h):
        # reuse the w_eff tensor built by the embedding forward this step, so
        # lookup-side and head-side gradients land on one captured tensor
        w = self.emb.tw._w_eff
        assert w is not None, "embedding forward must run before the head"
        z = rot_fwd(h, self.emb.signs).to(torch.float16)
        return F.linear(z, w)


class Fp32RMSNorm(nn.Module):
    """FP island made trainable: fp32 master weight, fp16 in/out."""

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.weight = nn.Parameter(norm.weight.data.float())
        self.eps = getattr(norm, "variance_epsilon", None) or getattr(norm, "eps", 1e-6)

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * xf).to(dt)


TARGETS = [
    ("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
    ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
]


def install_ternary(model, init: str):
    """Replace target linears + tied embedding/head with rotated ternary modules;
    wrap all RMSNorms as trainable fp32 islands. Returns list of TernaryWeight."""
    dev = next(model.parameters()).device
    emb = model.model.embed_tokens
    s_hidden = signs_for_width(emb.weight.shape[1], dev)
    new_emb = RotTrainEmbedding(emb.weight.data, s_hidden, init)
    model.model.embed_tokens = new_emb
    model.lm_head = RotTrainHead(new_emb)

    ternary_weights = [new_emb.tw]
    for layer in model.model.layers:
        for ppath, name in TARGETS:
            parent = layer
            for p in ppath.split("."):
                parent = getattr(parent, p)
            lin = getattr(parent, name)
            s = signs_for_width(lin.weight.shape[1], dev)
            mod = RotTrainLinear(lin, s, init)
            setattr(parent, name, mod)
            ternary_weights.append(mod.tw)
        # norm islands -> fp32 trainable
        layer.input_layernorm = Fp32RMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = Fp32RMSNorm(layer.post_attention_layernorm)
        layer.self_attn.q_norm = Fp32RMSNorm(layer.self_attn.q_norm)
        layer.self_attn.k_norm = Fp32RMSNorm(layer.self_attn.k_norm)
    model.model.norm = Fp32RMSNorm(model.model.norm)

    # freeze everything, then unfreeze scales + islands
    for p in model.parameters():
        p.requires_grad_(False)
    for tw in ternary_weights:
        tw.theta.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, Fp32RMSNorm):
            m.weight.requires_grad_(True)
    return ternary_weights


def collect_trainable(model, ternary_weights):
    scales, islands = [], []
    for tw in ternary_weights:
        scales.append(tw.theta)
    for m in model.modules():
        if isinstance(m, Fp32RMSNorm):
            islands.append(m.weight)
    return scales, islands


@torch.no_grad()
def vstep(ternary_weights, frac: float = 0.01, trust: float = 0.01, use_ema: bool = False):
    """Top-score code flips capped by trust ratio.

    use_ema=True ranks by the |grad| EMA (vstep-ready smoothed signal, updated
    each training step via update_ema) instead of the raw last-sample grad —
    the fix for the oscillation seen with the naive per-step ranking.
    Flip direction still comes from the current grad sign.

    Flip budget: ||delta||^2 = sum over flips of s_g^2 <= (trust * ||W||)^2,
    with ||W||^2 = sum_g s_g^2 * nnz_g over all folded tensors.
    """
    scored = []
    total_w2, flip_cost, total_n = 0.0, 0.0, 0
    for tw in ternary_weights:
        if use_ema and tw.ema is not None:
            sc = tw.ema.abs().float()
            dirn = tw.ema.float()
        else:
            sc = tw.grad_scores()
            dirn = None
        if sc is None:
            continue
        s = tw.scales()                                   # (out, ng)
        nnz = (tw.codes != 0).reshape(tw.codes.shape[0], -1, GROUP).float().sum(-1)
        total_w2 += float((s * s * nnz).sum())
        flip_cost += float((s * s).sum() * GROUP)
        total_n += tw.codes.numel()
        scored.append((tw, sc.reshape(-1),
                       dirn.reshape(-1) if dirn is not None else None))
    if not scored:
        return 0
    if not math.isfinite(total_w2) or not math.isfinite(flip_cost):
        return 0          # divergent step: skip flips rather than crash
    # threshold from strided score samples (no giant cat/randperm allocations)
    samples = []
    for _, sc, _d in scored:
        samples.append(sc[:: max(1, sc.numel() // 200_000)])
    sample = torch.cat(samples)
    ks = max(1, int(frac * sample.numel()))
    thresh = sample.kthvalue(sample.numel() - ks + 1).values.item()
    budget2 = (trust * math.sqrt(max(total_w2, 1e-16))) ** 2
    mean_s2 = flip_cost / max(total_n, 1)
    max_flips = max(1, int(budget2 / max(mean_s2, 1e-16)))
    moved_total, used = 0, 0
    undo = []          # (tw, idx, old_codes) for accept/reject gating
    for tw, sc, dirn in scored:
        idx = (sc >= thresh).nonzero(as_tuple=True)[0]
        if used + idx.numel() > max_flips:
            idx = idx[: max(0, max_flips - used)]
        used += idx.numel()
        codes = tw.codes.reshape(-1)
        undo.append((tw, idx, codes[idx].clone()))
        moved_total += tw.vstep_flip(idx, direction=dirn)
    return moved_total, undo


def undo_flips(undo):
    for tw, idx, old in undo:
        tw.codes.reshape(-1)[idx] = old
