"""Shared rotation/ternary math for the Bonsai contract (fork-compatible).

Contract (verified against fork sources and the 27B artifact, see
bonsai-l0prime-findings.md):
  R = H_n S / sqrt(n), block-diagonal with n=1024 along the INPUT feature axis;
  S = fixed per-width ±1 vector (one sign vector per distinct input width).
  folded weight   W' = W R^-1  -> row-wise: w' = fwht(w * s) / sqrt(n)
  activation      z  = R x     -> z      = fwht(x * s) / sqrt(n)
  embedding restore (inverse-after-lookup): e = s * fwht(e') / sqrt(n)
Both forward ops are rot_fwd; the inverse is rot_inv. rot_inv(rot_fwd(x)) == x.

Ternary: groups of 128 along the input axis, scale = group amax, round-to-nearest
into {-1, 0, +1} (matches quantize_row_pq2_0_ref, ggml-quants.c:113).
"""
import math

import torch

BLOCK = 1024
GROUP = 128

_sign_cache: dict[int, torch.Tensor] = {}


def signs_for_width(width: int, device, dtype=torch.float32) -> torch.Tensor:
    """Fixed ±1 vector per input width (deterministic seed, per-width shared —
    same convention as the artifact's sign_widths/sign_values)."""
    key = width
    if key not in _sign_cache:
        g = torch.Generator().manual_seed(20260922 + width)
        _sign_cache[key] = (torch.randint(0, 2, (width,), generator=g) * 2 - 1)
    return _sign_cache[key].to(device=device, dtype=dtype)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Sylvester Walsh-Hadamard along the last dim (power of 2)."""
    n = x.shape[-1]
    y = x
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return y


def _blocks(x: torch.Tensor, n: int) -> tuple[torch.Tensor, tuple]:
    lead = x.shape[:-1]
    w = x.shape[-1]
    assert w % n == 0, f"width {w} not divisible by block {n}"
    return x.reshape(*lead, w // n, n), lead


def rot_fwd(x: torch.Tensor, signs: torch.Tensor, n: int = BLOCK) -> torch.Tensor:
    """z = R x  (== fold for weight rows). Works on (..., width)."""
    xb, lead = _blocks(x.float(), n)
    zb = fwht(xb * signs.reshape(-1, n)) / math.sqrt(n)
    return zb.reshape(*lead, x.shape[-1])


def rot_inv(x: torch.Tensor, signs: torch.Tensor, n: int = BLOCK) -> torch.Tensor:
    """e = s * fwht(e') / sqrt(n)  (inverse-after-lookup)."""
    xb, lead = _blocks(x.float(), n)
    eb = fwht(xb) / math.sqrt(n) * signs.reshape(-1, n)
    return eb.reshape(*lead, x.shape[-1])


def rtn_ternary(w: torch.Tensor, g: int = GROUP) -> torch.Tensor:
    """Naive amax+RTN ternary, groups of g along the last (input) axis.
    Returns dequantized s*T with T in {-1,0,+1}. Matches PQ2_0 ref codec."""
    lead, width = w.shape[:-1], w.shape[-1]
    assert width % g == 0
    wg = w.float().reshape(*lead, width // g, g)
    s = wg.abs().amax(dim=-1, keepdim=True)
    t = torch.where(s > 0, torch.round(wg / s.clamp(min=1e-30)), torch.zeros_like(wg))
    t = t.clamp(-1, 1)
    return (t * s).reshape(*lead, width)
