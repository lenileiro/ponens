#!/usr/bin/env python3
"""Does the auto primitive-proposer discover a RELATIONAL primitive (not just counts)? Test on Monk-1, the
canonical UCI concept: positive iff (attr1 == attr2) OR (attr5 == 1). The raw DNF reasoner can already
express 'attr5==1' (a plain literal) but CANNOT express 'attr1==attr2' (attribute-value rules can't compare
two columns). So a full solution REQUIRES the proposer to discover the column-equality primitive itself.

We generate Monk-1's full instance space (the deterministic benchmark concept over its known attribute
cardinalities) and split it; the proposer never sees the rule, only labeled rows.
"""
import itertools
import sys

import numpy as np
import pandas as pd

from thinking.azr_win import _conj, apply_rule, predict_selective, reason_selective_cv
from thinking.primitives import propose, _featmat, candidate_primitives, rank_score, selfplay_rank

ATTRS = ["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"]


def monk1():
    space = itertools.product([1, 2, 3], [1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2])
    df = pd.DataFrame(list(space), columns=ATTRS)
    df["class"] = ((df.attr1 == df.attr2) | (df.attr5 == 1)).astype(int)
    return df


def main():
    df = monk1()
    print(f"Monk-1: {len(df)} instances, concept = (attr1==attr2) OR (attr5==1); positive rate "
          f"{df['class'].mean():.3f}")
    print("  (raw attribute-value rules CAN express attr5==1 but CANNOT express attr1==attr2)")

    print("\n[learn-to-rank] training PrimitiveRanker via self-play (zero Monk data)...")
    ranker = selfplay_rank(steps=150, seed=0, verbose=False)
    y0 = df["class"].values.astype(int)
    top = sorted(candidate_primitives(df, ATTRS), key=lambda c: -rank_score(ranker, c[1], y0))[:4]
    print("  learned-ranker top-4 on Monk-1: " + ", ".join(f"{nm}={rank_score(ranker, s, y0):.2f}" for nm, s in top))

    print("\n[propose] verified feature discovery using the LEARNED ranker:")
    chosen, score = propose(df, "class", ATTRS, rounds=3, ranker=ranker)
    print(f"\n  discovered primitives: {[c[0] for c in chosen]}")

    y = df["class"].values.astype(int)
    X, names = _featmat(df, ATTRS, chosen)
    rng = np.random.default_rng(1); idx = rng.permutation(len(y)); cut = int(0.6 * len(y))
    tr, te = idx[:cut], idx[cut:]
    pos, neg = reason_selective_cv(X[tr], y[tr], proposer=None, seed=0, folds=4, exhaustive=True, compress=True,
                                   min_prec_pos=1.0, min_prec_neg=1.0, build=dict(max_lit=2), verbose=False)
    rend = " OR ".join("(" + " AND ".join(f"{names[j]}{o}{t:g}" for j, o, t in _conj(c)) + ")" for c in pos)
    acc = (apply_rule(pos, X[te]) == y[te]).mean()
    print(f"  final rule POSITIVE when: {rend}")
    print(f"  held-out accuracy with discovered primitive: {acc:.3f}")


if __name__ == "__main__":
    sys.exit(main())
