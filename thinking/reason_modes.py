#!/usr/bin/env python3
"""Two more in-house reasoning paradigms (no sklearn, no feature harvesting), as alternatives to program
synthesis:

  (1) solve_analogy  -- reasoning-by-analogy with a LEARNED, verified metric. Predict a row as a weighted
      average of similar past rows; the similarity is a learned diagonal metric over the columns, trained to
      minimize held-out error. The metric DISCOVERS which columns matter and their scale -- in-context kernel
      regression, "reason from precedent" (this trip looks like those past trips). A differentiable attention
      over rows; verified on held-out.

  (2) solve_regime   -- discovered regimes + per-regime linear reasoning. A mixture-of-linear-experts whose
      softmax GATE discovers the natural regimes (trip types, rush vs off-peak) and fits a simple linear model
      per regime. A regime is an interaction (a soft conjunction), so this captures interactions without us
      naming them. "Divide, then reason locally." In-house gradient descent; verified on held-out.

Base features come from progsynth.build_base (numerics + aggregate-by + cluster) -- general operators, not
hand features. torch is the in-house substrate (no model libraries).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.progsynth import build_base


def _split_std(df, target, test_frac, seed):
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(df)); cut = int((1 - test_frac) * len(df))
    tri, tei = idx[:cut], idx[cut:]
    X, names, y = build_base(df, target, tri)
    mu = X[tri].mean(0); sd = X[tri].std(0); sd[sd < 1e-6] = 1.0
    return ((X - mu) / sd).astype(np.float32), y.astype(np.float32), tri, tei, names


# ---- (1) reasoning-by-analogy with a learned verified metric ----
def fit_analogy(Xs, y, tri, tei, names=None, R=1500, steps=400, seed=0, verbose=False):
    """Learn a diagonal metric over standardized columns; predict tei as a metric-weighted avg of train rows.
    Returns (pred_tei, top_features). Shared-split version for run-both integration."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xtr = torch.tensor(Xs[tri]); ytr = torch.tensor(y[tri])
    perm = rng.permutation(len(tri)); nref = min(R, len(tri) // 2)    # keep half for queries -> metric trains
    ri, qi = perm[:nref], perm[nref:]
    Xr, yr = Xtr[ri], ytr[ri]
    Xq, yq = Xtr[qi], ytr[qi]
    D = Xs.shape[1]
    w = (torch.randn(D) * 0.05).requires_grad_(); logT = torch.tensor([math.log(D)]).requires_grad_()
    opt = torch.optim.Adam([w, logT], lr=0.12)
    for s in range(steps):
        bi = rng.choice(len(qi), min(256, len(qi)), replace=False)
        d = ((Xq[bi][:, None, :] - Xr[None, :, :]) ** 2 * F.softplus(w)).sum(-1)  # noqa: learned diag metric
        a = torch.softmax(-d / torch.exp(logT), 1)
        loss = ((a @ yr - yq[bi]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and s % max(1, steps // 4) == 0:
            print(f"    analogy step {s}: train MSE {loss.item():.1f}")

    def pred(Xnp, chunk=512):
        Xt = torch.tensor(Xnp); out = []
        with torch.no_grad():
            for i in range(0, len(Xt), chunk):
                d = ((Xt[i:i + chunk][:, None, :] - Xr[None, :, :]) ** 2 * F.softplus(w)).sum(-1)
                out.append((torch.softmax(-d / torch.exp(logT), 1) @ yr).numpy())
        return np.concatenate(out)
    wv = F.softplus(w).detach().numpy()
    top = sorted(zip(names or range(len(wv)), wv), key=lambda t: -t[1])[:6]
    return pred(Xs[tei]), [(n, round(float(v), 2)) for n, v in top]


def solve_analogy(df, target, test_frac=0.3, R=1500, steps=400, seed=0, verbose=True):
    Xs, y, tri, tei, names = _split_std(df, target, test_frac, seed)
    pred, top = fit_analogy(Xs, y, tri, tei, names, R, steps, seed, verbose)
    return {"rmse": round(math.sqrt(((pred - y[tei]) ** 2).mean()), 3),
            "baseline": round(math.sqrt(((y[tri].mean() - y[tei]) ** 2).mean()), 3), "learned_metric_top": top}


# ---- (2) discovered regimes: mixture-of-linear-experts (gate finds the breakpoints) ----
class MoE(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.gate = nn.Linear(D, K); self.exp = nn.Linear(D, K)

    def forward(self, X):
        return (torch.softmax(self.gate(X), 1) * self.exp(X)).sum(1)


def fit_regime(Xs, y, tri, tei, K=6, steps=1500, seed=0):
    """Mixture-of-linear-experts: gate discovers regimes, linear model per regime. Returns pred_tei.
    Shared-split version for run-both integration."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    ii = rng.permutation(len(tri)); c = int(0.85 * len(tri)); a, v = tri[ii[:c]], tri[ii[c:]]
    ym = float(y[a].mean()); ys = float(y[a].std()) + 1e-6
    Xa = torch.tensor(Xs[a]); ya = torch.tensor((y[a] - ym) / ys)
    Xv = torch.tensor(Xs[v]); yv = (y[v] - ym) / ys
    Xte = torch.tensor(Xs[tei])
    model = MoE(Xs.shape[1], K); opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    best, best_state, bad = 1e9, None, 0
    for s in range(steps):
        bi = rng.choice(len(a), min(512, len(a)), replace=False)
        loss = ((model(Xa[bi]) - ya[bi]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 25 == 0:
            with torch.no_grad():
                vr = math.sqrt((((model(Xv) - torch.tensor(yv)) * ys) ** 2).mean().item())
            if vr < best - 1e-3:
                best, best_state, bad = vr, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 12:
                    break
    if best_state:
        model.load_state_dict(best_state)
    with torch.no_grad():
        return model(Xte).numpy() * ys + ym


def solve_regime(df, target, K=6, test_frac=0.3, steps=1500, seed=0, verbose=True):
    Xs, y, tri, tei, names = _split_std(df, target, test_frac, seed)
    pred = fit_regime(Xs, y, tri, tei, K, steps, seed)
    return {"rmse": round(math.sqrt(((pred - y[tei]) ** 2).mean()), 3), "K": K,
            "baseline": round(math.sqrt(((y[tri].mean() - y[tei]) ** 2).mean()), 3)}


def main():
    import pandas as pd
    for n, p, t, d in [("insurance", "kaggle_data/insurance/insurance.csv", "charges", []),
                       ("delivery", "kaggle_data/delivery/Food_Delivery_Times.csv", "Delivery_Time_min", ["Order_ID"])]:
        df = pd.read_csv(p).drop(columns=d)
        a = solve_analogy(df, t, verbose=False); r = solve_regime(df, t, verbose=False)
        print(f"{n}: analogy {a['rmse']} | regime(K={r['K']}) {r['rmse']} | baseline {a['baseline']}")
        print(f"   analogy learned-metric top: {a['learned_metric_top'][:4]}")
        print(f"   regime sizes: {r['regime_sizes']}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
