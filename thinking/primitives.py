#!/usr/bin/env python3
"""Auto primitive-proposer: verified feature DISCOVERY. The same 'propose -> verify -> keep' that azr_win
applies to rules, applied one level up to the FEATURES. When a SHORT verified rule over the raw columns
generalizes poorly (a representation wall, e.g. Monk-2's counting concept), the proposer generates candidate
derived primitives from a library -- count(==v) across columns, column-equality (col_i==col_j), pairwise
sum/diff for numerics -- and KEEPS a primitive only if adding it raises held-out confident-correctness. The
verifier decides which primitives are real signal; nothing is hand-added.

A feature set is scored by whether a SHORT (depth<=2) verified rule over it generalizes -- so the proposer
naturally prefers the primitive that makes the concept SIMPLE (Monk-2 -> 'count_of_1s==2', one step).
"""
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from thinking.azr_win import predict_selective, reason_selective_cv


def candidate_primitives(df, cols):
    """Generate derived-feature candidates (name, float array) from the feature library:
    count(==v) across columns, parity of a count, column-equality, row n-distinct, numeric pair sum/diff."""
    cands = []
    discrete = [c for c in cols if df[c].nunique() <= 12]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 12]
    if len(discrete) >= 2:
        Dv, vals = _coded_discrete(df, discrete)
        for i, v in enumerate(vals):
            cnt = (Dv == i).sum(axis=1).astype(float)               # count(== v) across all discrete columns
            cands.append((f"count(=={v})", cnt))
            if cnt.max() > 1:
                cands.append((f"count(=={v})%2", (cnt % 2)))          # parity of a count (parity concepts)
        for a, b in combinations(discrete, 2):                       # relational: col_i == col_j
            cands.append((f"({a}=={b})", (df[a].values == df[b].values).astype(float)))
        cands.append(("row_ndistinct", np.array([len(set(r)) for r in Dv], dtype=float)))
    for a, b in combinations(numeric, 2):                            # numeric pair sums/diffs
        cands.append((f"({a}-{b})", (df[a] - df[b]).astype(float).values))
        cands.append((f"({a}+{b})", (df[a] + df[b]).astype(float).values))
    return cands


def _coded_discrete(df, discrete):
    """Map all discrete columns onto a SHARED value->code space so cross-column counts are meaningful."""
    vals = sorted(set().union(*[set(pd.unique(df[c])) for c in discrete]), key=lambda x: str(x))
    m = {v: i for i, v in enumerate(vals)}
    return pd.DataFrame({c: df[c].map(m) for c in discrete}).values, vals


def _cheap_score(col, y, min_count=5, bins=12):
    """Cheap proxy (no rule search): strongest single-value class shift this feature induces. Used to
    pre-rank candidates so only the most promising get the expensive verified-rule evaluation."""
    f = np.asarray(col, dtype=float)
    if len(np.unique(f)) > bins:
        edges = np.quantile(f, np.linspace(0, 1, bins + 1))
        f = np.digitize(f, edges[1:-1])
    base = float(y.mean()); best = 0.0
    for v in np.unique(f):
        m = f == v
        if int(m.sum()) >= min_count:
            best = max(best, abs(float(y[m].mean()) - base))
    return best


# ---- learned primitive-ranker: trained by self-play to predict (from cheap stats) a primitive's usefulness ----
def _mi(f, y):
    n = len(y); mi = 0.0
    for v in np.unique(f):
        for c in (0, 1):
            pxy = float(((f == v) & (y == c)).mean())
            if pxy > 0:
                mi += pxy * np.log(pxy / ((f == v).mean() * (y == c).mean() + 1e-12) + 1e-12)
    return float(mi)


def _descriptor(col, y, bins=12):
    """Fixed-length, dataset-agnostic stats of (feature, label) -> the learned ranker's input."""
    f = np.asarray(col, dtype=float)
    if len(np.unique(f)) > bins:
        f = np.digitize(f, np.quantile(f, np.linspace(0, 1, bins + 1))[1:-1])
    base = float(y.mean()); shifts, covs, purs = [], [], []
    for v in np.unique(f):
        m = f == v; nn_ = int(m.sum())
        if nn_ >= 3:
            p = float(y[m].mean()); shifts.append(abs(p - base)); covs.append(nn_ / len(f)); purs.append(max(p, 1 - p))
    wpur = float(np.average(purs, weights=covs)) if covs else base   # coverage-weighted single-feature-rule acc
    return np.array([max(shifts or [0]), float(np.mean(shifts or [0])), max(purs or [0]), wpur,
                     max(covs or [0]), len(np.unique(f)) / len(f), _mi(f, y)], dtype=np.float32)


class PrimitiveRanker(nn.Module):
    def __init__(self, nin=7, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def selfplay_rank(steps=120, seed=0, verbose=True):
    """Train the ranker on SELF-GENERATED concept tasks (counting / relational / parity / plain). Target =
    the EXPENSIVE signal (a single derived feature's verified-rule score); input = the cheap descriptor. The
    ranker learns to predict which primitives are worth full evaluation -- learn-to-reason at the feature level,
    zero training on any real dataset."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    rk = PrimitiveRanker(); opt = torch.optim.AdamW(rk.parameters(), lr=2e-3)
    for step in range(steps):
        F = int(rng.integers(4, 6)); alpha = int(rng.integers(2, 5)); n = 160
        cols = [f"c{i}" for i in range(F)]
        df = pd.DataFrame({c: rng.integers(0, alpha, n) for c in cols})
        kind = rng.choice(["count", "relational", "parity"])
        if kind == "count":
            v = int(rng.integers(0, alpha)); k = int(rng.integers(1, F))
            y = ((df.values == v).sum(1) == k).astype(int)
        elif kind == "relational":
            i, j = rng.choice(F, 2, replace=False)
            y = (df[cols[i]].values == df[cols[j]].values).astype(int)
        else:
            v = int(rng.integers(0, alpha)); y = ((df.values == v).sum(1) % 2).astype(int)
        if y.mean() in (0.0, 1.0):
            continue
        cands = candidate_primitives(df, cols)
        X, T = [], []
        for nm, ser in cands:
            X.append(_descriptor(ser, y)); T.append(_score(np.asarray(ser, float).reshape(-1, 1), y)[0])
        Xt = torch.tensor(np.stack(X)); Tt = torch.tensor(np.array(T, np.float32))
        rk.train(); pred = rk(Xt); loss = nn.functional.mse_loss(pred, Tt)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % max(1, steps // 5) == 0:
            print(f"    selfplay-rank step {step}: usefulness-prediction MSE {loss.item():.4f}", flush=True)
    return rk


def rank_score(ranker, col, y):
    with torch.no_grad():
        return float(ranker(torch.tensor(_descriptor(col, y)).unsqueeze(0))[0])


def _featmat(df, base_cols, derived):
    X = df[base_cols].apply(lambda c: pd.factorize(c)[0] if c.dtype == object else c).values.astype(float)
    names = list(base_cols)
    for nm, ser in derived:
        X = np.concatenate([X, np.asarray(ser).reshape(-1, 1)], axis=1); names.append(nm)
    return X, names


def _score(X, y, seed=0):
    """Held-out confident-correctness of a SHORT (depth<=2) verified rule over these features.
    Rewards coverage AND accuracy: fraction of validation rows answered AND correct."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
    tr, va = idx[:cut], idx[cut:]
    pos, neg = reason_selective_cv(X[tr], y[tr], proposer=None, device="cpu", seed=seed, folds=4,
                                   min_prec_pos=0.97, min_prec_neg=0.97, min_support=4, exhaustive=True,
                                   compress=True, build=dict(max_lit=2, max_lits=40), verbose=False)
    dec = predict_selective(pos, neg, X[va])
    return float(((dec >= 0) & (dec == y[va])).mean()), (pos, neg)


def propose(df, target, base_cols, rounds=3, topk=8, ranker=None, verbose=True):
    """Greedily add the primitive that most improves held-out confident-correctness; stop when none helps.
    Each round pre-ranks the whole library and only full-evaluates (verified rule search) the top-K -- so the
    library can grow without the eval cost exploding. Pre-rank uses the LEARNED ranker if given, else the
    cheap heuristic."""
    y = df[target].values.astype(int)
    cands = candidate_primitives(df, base_cols)
    prank = (lambda c: rank_score(ranker, c, y)) if ranker is not None else (lambda c: _cheap_score(c, y))
    if verbose:
        print(f"  library: {len(cands)} candidate primitives  (pre-rank: {'learned' if ranker else 'heuristic'})")
    chosen = []
    Xcur, _ = _featmat(df, base_cols, chosen)
    best, _ = _score(Xcur, y)
    if verbose:
        print(f"  baseline (raw features only): confident-correct {best:.3f}")
    for r in range(rounds):
        pool = [(nm, ser) for nm, ser in cands if nm not in [c[0] for c in chosen]]
        if not pool:
            break
        ranked = sorted(pool, key=lambda p: -prank(p[1]))[:topk]             # pre-rank -> top-K
        scored = []
        for nm, ser in ranked:
            X2, _ = _featmat(df, base_cols, chosen + [(nm, ser)])
            s, _ = _score(X2, y)
            scored.append((s, nm, ser))
        scored.sort(key=lambda t: -t[0])
        bs, bnm, bser = scored[0]
        if bs <= best + 1e-9:
            if verbose:
                print(f"  round {r + 1}: no proposed primitive verifies an improvement -> stop")
            break
        chosen.append((bnm, bser)); best = bs
        if verbose:
            print(f"  round {r + 1}: PROPOSE {bnm:<16} (full-eval top {len(ranked)} of {len(pool)}) "
                  f"-> confident-correct {bs:.3f}  KEEP")
        if best >= 0.999:
            break
    return chosen, best


def main():
    df = pd.read_csv("kaggle_data/monk/monk.csv")
    attrs = ["'attr1'", "'attr2'", "'attr3'", "'attr4'", "'attr5'", "'attr6'"]
    y = df["'class'"].values.astype(int)
    print("Monk-2 ('exactly two of 6 attrs == 1') -- LEARNED ranker proposes the primitive (zero Monk data).")

    print("\n[learn-to-rank] training PrimitiveRanker via self-play on synthetic concept tasks...")
    ranker = selfplay_rank(steps=150, seed=0)

    cands = candidate_primitives(df, attrs)                               # what does the learned ranker prefer?
    top = sorted(cands, key=lambda c: -rank_score(ranker, c[1], y))[:5]
    print("  learned-ranker top-5 candidates on Monk-2:")
    for nm, ser in top:
        print(f"    {nm:<16} learned-score {rank_score(ranker, ser, y):.3f}")

    print("\n[propose] verified feature discovery using the LEARNED ranker:")
    chosen, score = propose(df, "'class'", attrs, rounds=3, ranker=ranker)
    print(f"\n  discovered primitives: {[c[0] for c in chosen]}")
    # final: full verified rule using raw + discovered primitives, report held-out accuracy
    y = df["'class'"].values.astype(int)
    X, names = _featmat(df, attrs, chosen)
    rng = np.random.default_rng(1); idx = rng.permutation(len(y)); cut = int(0.6 * len(y))
    tr, te = idx[:cut], idx[cut:]
    from thinking.azr_win import apply_rule, _conj
    pos, neg = reason_selective_cv(X[tr], y[tr], proposer=None, seed=0, folds=4, exhaustive=True, compress=True,
                                   min_prec_pos=1.0, min_prec_neg=1.0, build=dict(max_lit=2), verbose=False)
    rend = " OR ".join("(" + " AND ".join(f"{names[j]}{o}{t:g}" for j, o, t in _conj(c)) + ")" for c in pos)
    acc = (apply_rule(pos, X[te]) == y[te]).mean()
    print(f"  final rule POSITIVE when: {rend}")
    print(f"  held-out accuracy with discovered primitive: {acc:.3f}")


if __name__ == "__main__":
    sys.exit(main())
