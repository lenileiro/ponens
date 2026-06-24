#!/usr/bin/env python3
"""The wall test: Monk-2 (lavagod/monk-problem) -- 'positive iff EXACTLY two of attr1..attr6 equal 1'. A
COUNTING concept: no short conjunction is pure (any rule shorter than 6 literals leaves the count ambiguous),
so DNF needs all 15 six-literal terms -- and each attribute has ~zero marginal signal. This is where verified
short-conjunction search is SUPPOSED to hit a wall.

Then we crack it the way the project says you should: not by brute-forcing deeper search, but with the right
REPRESENTATION. Add count features ('how many attrs equal v') to the hypothesis space, and Monk-2 collapses to
'count_of_ones == 2' -- a one-step verified rule. Finding the wall, then changing the op-space to cross it.
"""
import sys

import numpy as np
import pandas as pd

from thinking.azr_win import (apply_rule, predict_selective, reason_selective_cv, selfplay_reason)  # noqa
import torch  # noqa

ATTRS = ["'attr1'", "'attr2'", "'attr3'", "'attr4'", "'attr5'", "'attr6'"]


def evalrule(pos, neg, Xte, yte, tag):
    dec = predict_selective(pos, neg, Xte); a = dec >= 0
    cov = a.mean(); acc = (dec[a] == yte[a]).mean() if a.any() else 0.0
    print(f"  {tag:<34} coverage {cov:.3f}  accuracy-on-answered {acc:.3f}  abstained {int((~a).sum())}")


def main():
    df = pd.read_csv("kaggle_data/monk/monk.csv")
    y_all = df["'class'"].values.astype(int)
    A = df[ATTRS].values
    rng = np.random.default_rng(0); idx = rng.permutation(len(y_all)); cut = int(0.6 * len(y_all))
    tri, tei = idx[:cut], idx[cut:]
    print(f"Monk-2: {len(df)} rows, concept = 'exactly two of 6 attrs equal 1'; positive rate {y_all.mean():.3f}")

    print("\n[1] learn-to-reason on self-generated tasks...")
    prop = selfplay_reason(steps=2000, device="cpu", seed=0, verbose=False); print("  done.")

    # ---- (A) THE WALL: one-hot attributes, verified DNF (negations allowed, depth up to 4) ----
    Xoh = pd.get_dummies(df[ATTRS].astype(str))
    Xa, Xte_a = Xoh.values.astype(np.float32)[tri], Xoh.values.astype(np.float32)[tei]
    best1 = max(abs(np.corrcoef(Xoh.values.astype(float)[:, j], y_all)[0, 1]) for j in range(Xoh.shape[1]))
    print(f"\n[A] DNF over attributes (best single-feature |corr| {best1:.3f} -- no marginal signal):")
    pos, neg = reason_selective_cv(Xa, y_all[tri], proposer=prop, device="cpu", seed=0, folds=4,
                                   min_prec_pos=0.95, min_prec_neg=0.95, min_support=5, exhaustive=True,
                                   compress=True, build=dict(max_lit=4, max_lits=None), verbose=False)
    print(f"  found {len(pos)} positive + {len(neg)} negative pure clauses (depth<=4)")
    evalrule(pos, neg, Xte_a, y_all[tei], "DNF depth<=4:")
    gb_acc = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        gb_acc = (HistGradientBoostingClassifier(max_iter=300, random_state=0)
                  .fit(Xa, y_all[tri]).predict(Xte_a) == y_all[tei]).mean()
        print(f"  baseline gradient boosting (also struggles on counting): accuracy {gb_acc:.3f}")
    except Exception as e:  # noqa
        print(f"  (baseline skipped: {e})")

    # ---- (B) CROSS THE WALL: add COUNT features (how many attrs equal v) to the hypothesis space ----
    counts = np.stack([(A == v).sum(1) for v in (1, 2, 3, 4)], axis=1).astype(np.float32)
    cnames = [f"num_attrs={v}" for v in (1, 2, 3, 4)]
    Xc, Xte_c = counts[tri], counts[tei]
    print(f"\n[B] same reasoner + COUNT features {cnames}:")
    pos2, neg2 = reason_selective_cv(Xc, y_all[tri], proposer=prop, device="cpu", seed=0, folds=4,
                                     min_prec_pos=1.0, min_prec_neg=1.0, min_support=5, exhaustive=True,
                                     compress=True, build=dict(max_lit=2), verbose=False)
    from thinking.azr_win import _conj
    rend = lambda rs: " OR ".join("(" + " AND ".join(f"{cnames[j]}{o}{t:g}" for j, o, t in _conj(c)) + ")"
                                  for c in rs) or "(none)"
    print(f"  POSITIVE when: {rend(pos2)}")
    evalrule(pos2, neg2, Xte_c, y_all[tei], "with count features:")
    full = apply_rule(pos2, Xte_c)
    print(f"  single-sided 'positive iff count rule fires': accuracy {(full == y_all[tei]).mean():.3f} (coverage 100%)")


if __name__ == "__main__":
    sys.exit(main())
