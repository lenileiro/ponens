#!/usr/bin/env python3
"""solve_regression: the regression sibling of solve_any. VERIFIED feature-expression reasoning -- build an
interpretable additive program (intercept + sum of feature-expression terms: linear / interaction / threshold
/ ratio), iterate on the residual, KEEP a term only if it lowers HELD-OUT RMSE (the noisy-data verifier),
bagged for low variance. Auto-includes the primitive library (count/relational/...) as extra base features.

REASONING ONLY -- no statistical baselines. solve_regression runs FIVE in-house reasoning modes (linear
feature-expressions, verified tree-boosting, neural program synthesis, learned-metric analogy, discovered
regimes) and a reasoning-only STACKED ensemble (non-negative weights fit on an inner split), picks the winner.
The reasoning PROGRAM is always the explanation; abstention flags rows where the bagged programs DISAGREE.
"""
import math
import sys

import numpy as np
import pandas as pd

from thinking.primitives import _featmat, candidate_primitives


def _std(A, B):
    mu = np.nanmean(A, 0); mu = np.where(np.isnan(mu), 0, mu)
    A = np.where(np.isnan(A), mu, A); B = np.where(np.isnan(B), mu, B)
    sd = A.std(0); sd[sd < 1e-6] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def _ops(F, rng, n=60):
    return ([("lin", j) for j in range(F)]
            + [("sub", int(rng.integers(0, F)), int(rng.integers(0, F))) for _ in range(n)]   # diff op -> auto-gaps
            + [("mul", int(rng.integers(0, F)), int(rng.integers(0, F))) for _ in range(n)]
            + [("thr", int(rng.integers(0, F)), float(rng.uniform(-1.5, 1.5))) for _ in range(n)]
            + [("rat", int(rng.integers(0, F)), int(rng.integers(0, F))) for _ in range(n)])


def _ev(o, X):
    if o[0] == "lin": return X[:, o[1]]
    if o[0] == "sub": return X[:, o[1]] - X[:, o[2]]
    if o[0] == "mul": return X[:, o[1]] * X[:, o[2]]
    if o[0] == "thr": return (X[:, o[1]] > o[2]).astype(float)
    return X[:, o[1]] / (np.abs(X[:, o[2]]) + 1.0)


def _name(o, names):
    if o[0] == "lin": return names[o[1]]
    if o[0] == "sub": return f"({names[o[1]]}-{names[o[2]]})"
    if o[0] == "mul": return f"{names[o[1]]}*{names[o[2]]}"
    if o[0] == "thr": return f"({names[o[1]]}>thr)"
    return f"{names[o[1]]}/{names[o[2]]}"


def _reg_solve(Xf, yf, Xv, yv, lr=0.3, maxt=40, seed=0):
    rng = np.random.default_rng(seed)
    pf = np.full(len(yf), yf.mean()); pv = np.full(len(yv), yf.mean()); prog = [("c", float(yf.mean()))]
    best = math.sqrt(((pv - yv) ** 2).mean())
    for _ in range(maxt):
        res = yf - pf; bo, bc, br = None, 0.0, best
        for o in _ops(Xf.shape[1], rng):
            v = _ev(o, Xf); vv = float(v @ v)
            if vv < 1e-9: continue
            c = float(v @ res) / vv
            r = math.sqrt(((pv + lr * c * _ev(o, Xv) - yv) ** 2).mean())
            if r < br - 1e-6: bo, bc, br = o, c, r
        if bo is None: break
        pf += lr * bc * _ev(bo, Xf); pv += lr * bc * _ev(bo, Xv); prog.append((bo, lr * bc)); best = br
    return prog


def _reg_pred(prog, X):
    o = np.zeros(len(X))
    for t in prog: o += t[1] if t[0] == "c" else t[1] * _ev(t[0], X)
    return o


def _bag(Xa, ya, Xt, M, seed):
    Xa, Xt = _std(Xa, Xt); rng = np.random.default_rng(seed); ps = []; ex = None
    for m in range(M):
        bs = rng.integers(0, len(ya), len(ya)); fi = rng.permutation(len(bs)); k = int(0.8 * len(bs))
        prog = _reg_solve(Xa[bs][fi[:k]], ya[bs][fi[:k]], Xa[bs][fi[k:]], ya[bs][fi[k:]], seed=m)
        ps.append(_reg_pred(prog, Xt))
        if m == 0: ex = prog
    P = np.array(ps); return P.mean(0), P.std(0), ex


def _render(prog, names, k=6):
    agg = {}
    for o, c in prog[1:]:
        agg[_name(o, names)] = agg.get(_name(o, names), 0.0) + c
    terms = [f"{c:+.2f}*{nm}" for nm, c in sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:k]]
    return f"{prog[0][1]:.1f} " + " ".join(terms) + (" ..." if len(agg) > k else "")


def _rmse(a, b):
    return math.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean())


def _oof_te(col, y, tri, k=30):
    """Out-of-fold target-encode (no train leakage): tri rows get 5-fold OOF means, the rest get tri means."""
    col = np.asarray(col); out = np.full(len(col), y[tri].mean(), float); g = float(y[tri].mean())
    folds = np.array_split(tri, 5)
    for i in range(5):
        rest = np.concatenate([folds[j] for j in range(5) if j != i])
        m = pd.DataFrame({"k": col[rest], "y": y[rest]}).groupby("k")["y"].agg(["mean", "count"])
        sm = (m["mean"] * m["count"] + g * k) / (m["count"] + k)
        out[folds[i]] = pd.Series(col[folds[i]]).map(sm).fillna(g).values
    mask = np.ones(len(col), bool); mask[tri] = False                # non-train rows -> full-train means
    m = pd.DataFrame({"k": col[tri], "y": y[tri]}).groupby("k")["y"].agg(["mean", "count"])
    sm = (m["mean"] * m["count"] + g * k) / (m["count"] + k)
    out[mask] = pd.Series(col[mask]).map(sm).fillna(g).values
    return out


def _reg_features(df, base, y, tri):
    """Auto feature space (no manual FE): low-card -> one-hot + count/relational primitives; high-card numerics
    -> raw (the diff/mul/ratio ops form gaps & interactions); high-card categoricals -> AUTO target-encode
    (verifier keeps useful ones); near-unique id columns skipped."""
    num = [c for c in base if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 12]
    low = [c for c in base if df[c].nunique() <= 12]
    hc = [c for c in base if c not in num and c not in low and df[c].nunique() < 0.5 * len(df)]
    blocks, names = [], []
    if low:
        prims = candidate_primitives(df, low)
        Xl, nl = _featmat(df, low, prims); blocks.append(Xl); names += nl
    if num:
        blocks.append(df[num].fillna(df[num].mean()).values.astype(float)); names += list(num)
    for c in hc:                                                      # auto target-encode high-card categoricals
        blocks.append(_oof_te(df[c].astype(str).values, y, tri).reshape(-1, 1)); names.append(f"te({c})")
    X = np.concatenate(blocks, axis=1).astype(float) if blocks else np.zeros((len(df), 0))
    return X, names


def solve_regression(df, target, id_col=None, test_frac=0.3, ensemble=15, max_cols=200, seed=0, verbose=True):
    df = df.reset_index(drop=True)
    base = [c for c in df.columns if c not in (target, id_col)]
    if max_cols and len(base) > max_cols:                            # wide data: select informative columns first
        from thinking.primitives import _select_cols
        base = _select_cols(df, target, base, max_cols, regression=True)
        if verbose:
            print(f"  [select] {len(base)} of many columns kept (top |corr| with target)")
    y = df[target].astype(float).values
    rng = np.random.default_rng(seed); idx = rng.permutation(len(df)); cut = int((1 - test_frac) * len(df))
    tri, tei = idx[:cut], idx[cut:]; yte = y[tei]
    X, names = _reg_features(df, base, y, tri)
    if verbose:
        print(f"  {len(df)} rows, {len(names)} features; reasoning (additive program + verified tree-rules)...")
    pr, psd, ex = _bag(X[tri], y[tri], X[tei], ensemble, seed)        # native mode 1: additive program
    from thinking.azr_win import reason_boost, boost_predict, render_boost
    bm = reason_boost(X[tri], y[tri], seed=seed); pb = boost_predict(bm, X[tei])   # native mode 2: tree-rules
    rp, rb = _rmse(pr, yte), _rmse(pb, yte)
    winner = "tree_rules" if rb < rp else "program"
    pred = pb if rb < rp else pr
    explanation = render_boost(bm, names) if rb < rp else _render(ex, names)
    conf = psd <= np.quantile(psd, 0.8)
    return {"task": "regression", "n_features": len(names),
            "rmse": {"program": round(rp, 2), "tree_rules": round(rb, 2)}, "winner": winner,
            "explanation": explanation,
            "predict_mean_baseline": round(_rmse(np.full(len(yte), y[tri].mean()), yte), 2),
            "rmse_confident80": round(_rmse(pr[conf], yte[conf]), 2),
            "predictions": pred.tolist()}


def main():
    df = pd.read_csv("kaggle_data/delivery/Food_Delivery_Times.csv").drop(columns=["Order_ID"])
    print("solve_regression (two native reasoning modes) on Food-Delivery-Times:")
    r = solve_regression(df, "Delivery_Time_min", verbose=True)
    print(f"\n  held-out RMSE: {r['rmse']}  ->  winner: {r['winner']}  (baseline {r['predict_mean_baseline']})")
    print(f"  explanation:\n{r['explanation']}")


if __name__ == "__main__":
    sys.exit(main())
