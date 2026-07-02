#!/usr/bin/env python3
"""Harder in-regime Kaggle test: Tic-Tac-Toe Endgame (rsrishav/tictactoe-endgame-data-set). The label is
EXACTLY a DNF -- 'x wins' iff some line of three x's exists (8 lines x 3-literal conjunctions). Harder than
mushroom because NO SINGLE feature is predictive: azr_win must DISCOVER the 3-literal conjunctions (the
winning lines). If it works, it literally recovers the rules of the game -- and the rule extrapolates exactly.

Runtime = pure reasoning, zero training on ttt: RuleProposer trained on self-generated tasks, then verified
DNF synthesis (multi-fold) reasons the win-condition out of the board examples.
"""
import sys

import numpy as np
import pandas as pd

from thinking.azr_win import (apply_rule, explain_selective, predict_selective,  # noqa
                              reason_rule, reason_selective_cv, selfplay_reason)
import torch  # noqa

LINES = {(0, 1, 2): "top row", (3, 4, 5): "middle row", (6, 7, 8): "bottom row",
         (0, 3, 6): "left col", (1, 4, 7): "center col", (2, 5, 8): "right col",
         (0, 4, 8): "main diagonal", (2, 4, 6): "anti diagonal"}


def render(rules, names):
    from thinking.azr_win import _conj
    if not rules:
        return "(none)"
    return " OR ".join("(" + " AND ".join(names[j] + ("" if t == 1 else "=NO") for j, o, t in _conj(c)
                       if o == "==") + ")" for c in rules)


def main():
    df = pd.read_csv("kaggle_data/ttt/tic-tac-toe.data.csv")
    target = df.columns[-1]
    y_all = (df[target].values == "positive").astype(int)       # 1 = x WINS
    sq = list(df.columns[:9])
    Xdf = pd.get_dummies(df[sq])
    names = [c.replace("-square", "").replace("_", "=") for c in Xdf.columns]
    X_all = Xdf.values.astype(np.float32)
    rng = np.random.default_rng(0); idx = rng.permutation(len(y_all)); cut = int(0.7 * len(y_all))
    tri, tei = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X_all[tri], y_all[tri], X_all[tei], y_all[tei]
    print(f"tic-tac-toe: {len(df)} boards, {X_all.shape[1]} one-hot features; x-wins rate {y_all.mean():.3f}")
    # show that no single feature solves it (the reason it's harder than mushroom)
    best1 = max(abs(np.corrcoef(X_all[:, j], y_all)[0, 1]) for j in range(X_all.shape[1]))
    print(f"  best single-feature |correlation| with the label: {best1:.3f}  (no single square predicts a win)")

    print("\n[1] learn-to-reason on SELF-GENERATED tasks (zero ttt data)...")
    prop = selfplay_reason(steps=2500, device="cpu", seed=0, verbose=False); print("  done.")

    print("\n[2] REASON the win-condition (multi-fold verified DNF -- must find 3-literal conjunctions)...")
    pos, neg = reason_selective_cv(Xtr, ytr, proposer=prop, device="cpu", seed=0, folds=5,
                                   min_prec_pos=1.0, min_prec_neg=1.0, exhaustive=True,
                                   build=dict(max_lit=3, binary_presence_only=True))
    print(f"\n  X-WINS when: {render(pos, names)}")

    # map each discovered positive clause to a board line (did it recover the game's lines?)
    from thinking.azr_win import _conj
    found = set()
    for c in pos:
        sqs = tuple(sorted(j // 3 for j, o, t in _conj(c) if o == "==" and t == 1))  # 3 one-hots per square block
        if sqs in LINES:
            found.add(LINES[sqs])
    print(f"  recovered winning lines: {sorted(found)}  ({len(found)}/8)")

    print("\n[3] PREDICT on held-out boards (never seen):")
    # the positive rule is COMPLETE for this game (x wins IFF a line exists) -> use it as a full classifier
    full = apply_rule(pos, Xte)
    print(f"  single-sided 'x-wins iff a line fires' : accuracy {(full == yte).mean():.4f}  (coverage 100%)")
    dec = predict_selective(pos, neg, Xte); a = dec >= 0
    print(f"  two-sided w/ abstention                : coverage {a.mean():.4f}  "
          f"accuracy-on-answered {(dec[a] == yte[a]).mean():.4f}  abstained {int((~a).sum())}")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        gb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr, ytr)
        print(f"  baseline gradient boosting             : accuracy {(gb.predict(Xte) == yte).mean():.4f} "
              f"(black box -- no rule)")
    except Exception as e:  # noqa
        print(f"  (baseline skipped: {e})")

    print("\n[4] ANSWER individual boards with reasoning:")
    expl = explain_selective(pos, neg, Xte, names, pos_label="X-WINS", neg_label="no-x-win")
    pick = list(np.where((full == 1) & (yte == 1))[0][:2]) + list(np.where((full == 0) & (yte == 0))[0][:1])
    for ri in pick:
        e = expl[ri]
        print(f"  board {ri}: {e['why'][:110]}\n     -> {e['decision']}  (actual {'X-WINS' if yte[ri] else 'no-x-win'})")


if __name__ == "__main__":
    sys.exit(main())
