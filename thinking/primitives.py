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

from thinking.azr_win import predict_selective, reason_selective_cv


def candidate_primitives(df, cols):
    """Generate derived-feature candidates (name, int/float Series) from the feature library."""
    cands = []
    discrete = [c for c in cols if df[c].nunique() <= 12]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 12]
    if len(discrete) >= 2:
        vals = sorted(set().union(*[set(pd.unique(df[c])) for c in discrete]))
        for v in vals:                                              # count(== v) across all discrete columns
            cands.append((f"count(=={v})", (df[discrete] == v).sum(axis=1).astype(float)))
        for a, b in combinations(discrete, 2):                       # relational: col_i == col_j
            cands.append((f"({a}=={b})", (df[a].values == df[b].values).astype(float)))
    for a, b in combinations(numeric, 2):                            # numeric pair sums/diffs
        cands.append((f"({a}-{b})", (df[a] - df[b]).astype(float)))
        cands.append((f"({a}+{b})", (df[a] + df[b]).astype(float)))
    return cands


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


def propose(df, target, base_cols, rounds=3, verbose=True):
    """Greedily add the primitive that most improves held-out confident-correctness; stop when none helps."""
    y = df[target].values.astype(int)
    cands = candidate_primitives(df, base_cols)
    chosen = []
    Xcur, _ = _featmat(df, base_cols, chosen)
    best, _ = _score(Xcur, y)
    if verbose:
        print(f"  baseline (raw features only): confident-correct {best:.3f}")
    for r in range(rounds):
        scored = []
        for nm, ser in cands:
            if nm in [c[0] for c in chosen]:
                continue
            X2, _ = _featmat(df, base_cols, chosen + [(nm, ser)])
            s, _ = _score(X2, y)
            scored.append((s, nm, ser))
        if not scored:
            break
        scored.sort(key=lambda t: -t[0])
        bs, bnm, bser = scored[0]
        if bs <= best + 1e-9:
            if verbose:
                print(f"  round {r + 1}: no proposed primitive verifies an improvement -> stop")
            break
        chosen.append((bnm, bser)); best = bs
        if verbose:
            print(f"  round {r + 1}: PROPOSE {bnm:<16} -> confident-correct {bs:.3f}  (verified gain) KEEP")
        if best >= 0.999:
            break
    return chosen, best


def main():
    df = pd.read_csv("kaggle_data/monk/monk.csv")
    attrs = ["'attr1'", "'attr2'", "'attr3'", "'attr4'", "'attr5'", "'attr6'"]
    print("Monk-2 ('exactly two of 6 attrs == 1') -- can the proposer DISCOVER the right primitive itself?")
    print("\n[propose] verified feature discovery over raw attributes:")
    chosen, score = propose(df, "'class'", attrs, rounds=3)
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
