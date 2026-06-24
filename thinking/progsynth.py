#!/usr/bin/env python3
"""Neural-guided verified PROGRAM SYNTHESIS for tabular regression -- the discovered program IS the model.

No library models (numpy + a tiny in-house policy net; no sklearn). No feature harvesting: we supply only
GENERAL operators -- aggregate-by, compare/difference, combine (mul/ratio), threshold -- and the synthesizer
INVENTS the program (which columns, which compositions) per dataset. A self-play-trained policy proposes which
expression to add next; the verifier (held-out RMSE) decides what stays. The program = intercept + sum of
coefficient*expression terms, built by iterate-on-residual.

This is 'model proposes, brain proves' lifted to representation discovery: the reasoner writes the features
itself, as composable operator trees, instead of us harvesting them.
"""
import math

import numpy as np
import torch
import torch.nn as nn


# ---- DSL: an expression is a tuple tree over base columns; eval -> a numeric column ----
def ev(e, B):
    if e[0] == "b":   return B[e[1]]
    if e[0] == "mul": return ev(e[1], B) * ev(e[2], B)
    if e[0] == "diff": return ev(e[1], B) - ev(e[2], B)
    if e[0] == "ratio": return ev(e[1], B) / (np.abs(ev(e[2], B)) + 1.0)
    if e[0] == "thr": return (ev(e[1], B) > e[2]).astype(float)
    raise ValueError(e)


def re(e):
    if e[0] == "b": return e[1]
    if e[0] == "thr": return f"({re(e[1])}>{e[2]:.2f})"
    return f"({re(e[1])}{ {'mul':'*','diff':'-','ratio':'/'}[e[0]] }{re(e[2])})"


def candidates(base_names, chosen, rng, n=90):
    """Propose expressions: raw base, plus binary compositions and thresholds over base + already-chosen
    expressions (so the search composes operators of operators -- depth grows as the program grows)."""
    pool = [("b", nm) for nm in base_names] + list(chosen)
    cands = [("b", nm) for nm in base_names]
    for _ in range(n):
        a = pool[rng.integers(len(pool))]; b = pool[rng.integers(len(pool))]
        cands.append((rng.choice(["mul", "diff", "ratio"]), a, b))
    for _ in range(n // 2):
        a = pool[rng.integers(len(pool))]; cands.append(("thr", a, float(rng.normal())))
    seen, out = set(), []
    for c in cands:
        k = re(c)
        if k not in seen:
            seen.add(k); out.append(c)
    return out


# ---- in-house policy: rank a candidate expression by predicted held-out gain (no sklearn) ----
def descriptor(col, resid):
    c = col - col.mean(); r = resid - resid.mean()
    denom = math.sqrt(float(c @ c) * float(r @ r)) + 1e-9
    corr = abs(float(c @ r) / denom)
    return np.array([corr, corr * corr, float(col.std()),
                     len(np.unique(col)) / len(col), float(np.isfinite(col).mean())], np.float32)


class ExprRanker(nn.Module):
    def __init__(self, nin=5, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _std(A):
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-6] = 1.0
    return (A - mu) / sd, mu, sd


def synthesize(X, y, base_names, policy=None, holdout=0.3, maxterms=14, topk=12, seed=0, verbose=False):
    """Verified greedy program synthesis: each round the policy ranks candidate expressions, we least-squares
    fit the best on the residual and KEEP it only if it lowers held-out RMSE. Returns (program, names)."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); c = int((1 - holdout) * len(y))
    fit, ver = idx[:c], idx[c:]
    Xs, mu, sd = _std(X)
    B = {nm: Xs[:, j] for j, nm in enumerate(base_names)}
    yf, yv = y[fit], y[ver]
    pf = np.full(len(fit), yf.mean()); pv = np.full(len(ver), yf.mean()); prog = [("const", float(yf.mean()))]
    best = math.sqrt(((pv - yv) ** 2).mean()); chosen = []
    for t in range(maxterms):
        res = yf - pf
        cands = candidates(base_names, chosen, rng)
        cols = {re(e): ev(e, B) for e in cands}                       # eval once
        if policy is not None:                                        # learned policy ranks; else corr proxy
            feats = np.stack([descriptor(cols[re(e)][fit], res) for e in cands])
            with torch.no_grad():
                order = torch.argsort(-policy(torch.tensor(feats))).numpy()
        else:
            order = np.argsort([-abs(np.corrcoef(cols[re(e)][fit], res)[0, 1] if cols[re(e)][fit].std() > 0 else 0)
                                for e in cands])
        bo, bc, br = None, 0.0, best
        for i in order[:topk]:
            e = cands[i]; v = cols[re(e)]
            vf = v[fit]; vv = float(vf @ vf)
            if vv < 1e-9 or not np.isfinite(vv):
                continue
            coef = float(vf @ res) / vv
            r = math.sqrt(((pv + coef * v[ver] - yv) ** 2).mean())
            if r < br - 1e-6:
                bo, bc, br = e, coef, r
        if bo is None:
            break
        pf = pf + bc * ev(bo, B)[fit]; pv = pv + bc * ev(bo, B)[ver]; best = br
        prog.append((bo, bc)); chosen.append(bo)
        if verbose:
            print(f"    + {re(bo):<34} held-out RMSE {br:.3f}")
    return prog, (mu, sd, base_names)


def predict(prog, X, ctx):
    mu, sd, base_names = ctx
    Xs = (X - mu) / sd
    B = {nm: Xs[:, j] for j, nm in enumerate(base_names)}
    out = np.zeros(len(X))
    for t in prog:
        out += t[1] if t[0] == "const" else t[1] * ev(t[0], B)
    return out


def render(prog):
    terms = [f"{c:+.2f}*{re(e)}" for e, c in prog[1:8]]
    return f"{prog[0][1]:.1f} " + " ".join(terms) + (" ..." if len(prog) > 8 else "")


# ---- aggregate-by operator (realized as OOF base columns) + base builder ----
def _oof_te(col, y, tri, k=30):
    import pandas as pd
    col = np.asarray(col).astype(str); out = np.full(len(col), y[tri].mean(), float); g = float(y[tri].mean())
    f = np.array_split(tri, 5)
    for i in range(5):
        rest = np.concatenate([f[j] for j in range(5) if j != i])
        m = pd.DataFrame({"k": col[rest], "y": y[rest]}).groupby("k")["y"].agg(["mean", "count"])
        sm = (m["mean"] * m["count"] + g * k) / (m["count"] + k)
        out[f[i]] = pd.Series(col[f[i]]).map(sm).fillna(g).values
    mask = np.ones(len(col), bool); mask[tri] = False
    m = pd.DataFrame({"k": col[tri], "y": y[tri]}).groupby("k")["y"].agg(["mean", "count"])
    sm = (m["mean"] * m["count"] + g * k) / (m["count"] + k)
    out[mask] = pd.Series(col[mask]).map(sm).fillna(g).values
    return out


def build_base(df, target, tri):
    """Base columns = raw numerics + aggregate-by(categorical) [OOF target mean]. The synthesizer composes
    these via the DSL; the 'aggregate' operator is the only data-summarizing primitive, applied generically."""
    import pandas as pd
    y = df[target].astype(float).values
    cols = [c for c in df.columns if c != target]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 12]
    cat = [c for c in cols if c not in num and df[c].nunique() < 0.5 * len(df)]
    names, blocks = [], []
    for c in num:
        names.append(c); blocks.append(df[c].fillna(df[c].mean()).values.astype(float))
    for c in cat:
        names.append(f"agg({c})"); blocks.append(_oof_te(df[c].values, y, tri))
    return np.stack(blocks, 1).astype(float), names, y


def selfplay_expr(steps=400, seed=0, verbose=True):
    """Train ExprRanker on SELF-GENERATED regression tasks: target = the held-out gain an expression yields
    on the residual; input = its descriptor. Learns which expressions are worth verifying. Zero real data."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    rk = ExprRanker(); opt = torch.optim.AdamW(rk.parameters(), lr=2e-3)
    for step in range(steps):
        F = int(rng.integers(4, 8)); n = 300
        X = rng.normal(0, 1, (n, F)); names = [f"x{j}" for j in range(F)]
        B = {nm: X[:, j] for j, nm in enumerate(names)}
        # hidden program = a couple random compositions
        truth = np.zeros(n)
        for _ in range(int(rng.integers(1, 4))):
            a, b = (("b", names[rng.integers(F)]),) * 2 if False else (("b", names[rng.integers(F)]),
                                                                       ("b", names[rng.integers(F)]))
            op = rng.choice(["mul", "diff", "ratio"]); truth += rng.normal() * ev((op, a, b), B)
        y = truth + rng.normal(0, truth.std() * 0.3 + 1e-6, n)
        ci = rng.permutation(n); k = int(.7 * n); fit, ver = ci[:k], ci[k:]
        res = y[fit] - y[fit].mean()
        cands = candidates(names, [], rng, n=40)
        feats, gains = [], []
        for e in cands:
            v = ev(e, B)
            if not np.isfinite(v).all() or v[fit].std() < 1e-9:
                continue
            coef = float(v[fit] @ res) / float(v[fit] @ v[fit] + 1e-9)
            base = math.sqrt(((y[ver].mean() - y[ver]) ** 2).mean())
            g = base - math.sqrt(((y[ver].mean() + coef * v[ver] - y[ver]) ** 2).mean())   # held-out gain
            feats.append(descriptor(v[fit], res)); gains.append(max(0.0, g))
        if len(feats) < 4:
            continue
        gains = np.array(gains, np.float32); gains = gains / (gains.max() + 1e-9)
        pred = rk(torch.tensor(np.stack(feats)))
        loss = nn.functional.mse_loss(pred, torch.tensor(gains))
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % max(1, steps // 5) == 0:
            print(f"    selfplay-expr step {step}: gain-prediction MSE {loss.item():.4f}", flush=True)
    return rk


def solve(df, target, test_frac=0.3, policy=None, seed=0, verbose=True):
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(df)); cut = int((1 - test_frac) * len(df))
    tri, tei = idx[:cut], idx[cut:]
    X, names, y = build_base(df, target, tri)
    prog, ctx = synthesize(X[tri], y[tri], names, policy=policy, seed=seed, verbose=verbose)
    pred = predict(prog, X[tei], ctx)
    rmse = math.sqrt(((pred - y[tei]) ** 2).mean())
    return {"rmse": round(rmse, 3), "program": render(prog),
            "baseline": round(math.sqrt(((y[tri].mean() - y[tei]) ** 2).mean()), 3)}


def main():
    import pandas as pd
    print("[1] validate: synthetic task that REQUIRES composition  y = agg_r*dist + (t1-t2) + noise")
    rng = np.random.default_rng(0); n = 1500
    dist = rng.uniform(1, 30, n); t1 = rng.uniform(0, 100, n); t2 = rng.uniform(0, 100, n)
    rider = rng.integers(0, 40, n); rspeed = rng.uniform(20, 60, len(np.unique(rider)))[rider]
    y = rspeed * dist + (t1 - t2) + rng.normal(0, 30, n)
    df = pd.DataFrame({"dist": dist, "t1": t1, "t2": t2, "rider": rider, "y": y})
    print("   ", solve(df, "y", verbose=True))

    print("\n[2] train policy via self-play, then synthesize on insurance (real, no FE/libs):")
    pol = selfplay_expr(steps=400, seed=0, verbose=False)
    r = solve(pd.read_csv("kaggle_data/insurance/insurance.csv"), "charges", policy=pol, seed=0, verbose=True)
    print("   ", r)


if __name__ == "__main__":
    import sys
    sys.exit(main())

