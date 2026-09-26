#!/usr/bin/env python
"""M5-A1 toy: does matrix SMD with a three-well potential converge to ternary?

Overparameterized least squares  min_W ||XW - Y||^2  (interpolation regime,
d >> n), Y = X W* with W* exactly ternary-sparse. Compare:
  GD   -- plain gradient descent (implicit bias -> min-ish norm, dense)
  Adam -- element-wise preconditioning (2603.10485 Prop 2: distorts the limit)
  SMD  -- mirror descent with psi(z) = nu z^2/2 + softmin(z^2, (z-1)^2, (z+1)^2)
          (2602.18997: limit = argmin D_psi s.t. interpolation -> near-ternary)

psi' = nu*z + phi'(z) is strictly increasing once nu >= max(-phi''), so the
mirror map inverts by Newton. All quantities vectorized in numpy.
"""
import numpy as np

rng = np.random.default_rng(0)

# ---- problem ----
n, d = 64, 512
X = rng.standard_normal((n, d)) / np.sqrt(n)
Wstar = np.zeros(d)
nz = rng.choice(d, size=d // 10, replace=False)          # 10% nonzero ternary
Wstar[nz] = rng.choice([-1.0, 1.0], size=len(nz))
Y = X @ Wstar                                            # exact interpolation


def loss(W):
    R = X @ W - Y
    return float(R @ R / n)


def grad(W):
    return 2.0 * X.T @ (X @ W - Y) / n


# ---- three-well potential: deep negative Gaussian wells + minimal quadratic ----
BETA = 3.0     # well width
DEPTH = 3.0    # well depth (ternary attraction strength)


def phi_grad_hess(z):
    """phi = -DEPTH * (gauss(z) + gauss(z-1) + gauss(z+1)); returns phi', phi''."""
    c, b = DEPTH, BETA
    e0 = np.exp(-b * z ** 2)
    e1 = np.exp(-b * (z - 1) ** 2)
    e2 = np.exp(-b * (z + 1) ** 2)
    g = c * (2 * b * z * e0 + 2 * b * (z - 1) * e1 + 2 * b * (z + 1) * e2)
    h = c * ((4 * b * b * z ** 2 - 2 * b) * e0
             + (4 * b * b * (z - 1) ** 2 - 2 * b) * e1
             + (4 * b * b * (z + 1) ** 2 - 2 * b) * e2)
    return g, h


# pick nu so psi'' = nu + phi'' >= 0.1 everywhere on a probe grid
probe = np.linspace(-2, 2, 4001)
_, h_probe = phi_grad_hess(probe)
NU = float(-h_probe.min()) + 0.1


def psi_fwd(z):        # z -> theta
    g, _ = phi_grad_hess(z)
    return NU * z + g


def psi_inv(theta):    # theta -> z, Newton on monotone psi'
    z = theta / (NU + 2.0)
    for _ in range(40):
        g, h = phi_grad_hess(z)
        r = NU * z + g - theta
        dz = r / (NU + h)
        z = z - dz
        if np.max(np.abs(dz)) < 1e-12:
            break
    return z


def ternary_score(W, tol=0.05):
    d0 = np.minimum(np.abs(W), np.abs(np.abs(W) - 1.0))
    return float((d0 < tol).mean())


def well_energy(W):
    """psi mass NOT explained by entries sitting in wells (lower = more ternary)."""
    g0, _ = phi_grad_hess(W)
    psi = 0.5 * NU * W ** 2
    # distance from W to the nearest well center {0, +1, -1}
    dw = np.minimum(np.abs(W), np.abs(np.abs(W) - 1.0))
    return float(np.mean(dw))


def run(name, steps=30000, lr=2e-2, kind="gd"):
    W = np.zeros(d)
    m = np.zeros(d); v = np.zeros(d)
    snaps = []
    for t in range(1, steps + 1):
        g = grad(W)
        if kind == "gd":
            W = W - lr * g
        elif kind == "adam":
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
            W = W - lr * mh / (np.sqrt(vh) + 1e-8)
        elif kind == "smd":
            theta = psi_fwd(W)
            theta = theta - lr * g
            W = psi_inv(theta)
        if t % (steps // 6) == 0 or t == 1:
            snaps.append((t, loss(W), ternary_score(W), well_energy(W),
                          float(np.linalg.norm(W - Wstar))))
    print(f"-- {name} (nu={NU:.2f})")
    print(f"   {'iter':>7s} {'loss':>10s} {'ternary%':>9s} {'mean|dw|':>9s} {'||W-W*||':>9s}")
    for t, l, ts, dw, dn in snaps:
        print(f"   {t:7d} {l:10.3e} {ts*100:8.1f}% {dw:9.4f} {dn:9.4f}")
    return snaps


if __name__ == "__main__":
    print(f"n={n} d={d} nnz={len(nz)}; interpolation achievable (d>>n)")
    run("GD", kind="gd")
    run("Adam", kind="adam")
    run("SMD three-well", kind="smd")
