#!/usr/bin/env python3
"""Deeper in-regime Kaggle test: King-Rook vs King-Pawn chess endgame (ananthr1/chess-kingrook-vs-kingpawn).
3196 positions, 36 board features, target = 'won' (white-to-move can win) vs 'nowin'. Harder than tic-tac-toe:
more features and the win-condition needs DEEPER conjunctions. Tests whether the verified exhaustive+compress
pipeline scales in feature count and conjunction depth.

Runtime = pure reasoning, zero training on krkp: proposer self-play only, then multi-fold verified DNF
(exhaustive short-conjunction search + set-cover compression), with abstention + per-answer confidence.
"""
import sys

import numpy as np
import pandas as pd

from thinking.azr_win import (apply_rule, explain_selective, predict_selective,  # noqa
                              reason_selective_cv, selfplay_reason)
import torch  # noqa


def render(rules, names):
    from thinking.azr_win import _conj
    if not rules:
        return "(none)"
    return "\n         OR ".join("(" + " AND ".join(names[j] for j, o, t in _conj(c)) + ")" for c in rules)


def main():
    df = pd.read_csv("kaggle_data/krkp/kr-vs-kp.data", header=None)
    target = df.columns[-1]
    y_all = (df[target].values == "won").astype(int)            # 1 = white can win
    Xdf = pd.get_dummies(df[df.columns[:-1]].astype(str))
    names = [f"f{c.split('_')[0]}={c.split('_')[1]}" for c in Xdf.columns]
    X_all = Xdf.values.astype(np.float32)
    rng = np.random.default_rng(0); idx = rng.permutation(len(y_all)); cut = int(0.7 * len(y_all))
    tri, tei = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X_all[tri], y_all[tri], X_all[tei], y_all[tei]
    print(f"kr-vs-kp: {len(df)} positions, {X_all.shape[1]} one-hot features; 'won' rate {y_all.mean():.3f}")
    best1 = max(abs(np.corrcoef(X_all[:, j], y_all)[0, 1]) for j in range(X_all.shape[1]))
    print(f"  best single-feature |correlation|: {best1:.3f}")

    print("\n[1] learn-to-reason on SELF-GENERATED tasks (zero krkp data)...")
    prop = selfplay_reason(steps=2500, device="cpu", seed=0, verbose=False); print("  done.")

    print("\n[2] REASON the win-condition (multi-fold + exhaustive + set-cover compression)...")
    pos, neg = reason_selective_cv(Xtr, ytr, proposer=prop, device="cpu", seed=0, folds=5,
                                   min_prec_pos=0.99, min_prec_neg=0.99, min_support=25,
                                   exhaustive=True, compress=True,
                                   build=dict(max_lit=4, binary_presence_only=True, max_lits=45))
    print(f"\n  WON when:\n         {render(pos[:8], names)}{'  ...' if len(pos) > 8 else ''}")
    print(f"  ({len(pos)} won-clauses, {len(neg)} nowin-clauses after compression)")

    print("\n[3] PREDICT on held-out positions (never seen):")
    dec = predict_selective(pos, neg, Xte); a = dec >= 0
    print(f"  coverage {a.mean():.4f}  accuracy-on-answered {(dec[a] == yte[a]).mean():.4f}  "
          f"abstained {int((~a).sum())}")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        gb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr, ytr)
        print(f"  baseline gradient boosting: accuracy {(gb.predict(Xte) == yte).mean():.4f} (black box)")
    except Exception as e:  # noqa
        print(f"  (baseline skipped: {e})")

    print("\n[4] ANSWER individual positions with reasoning:")
    expl = explain_selective(pos, neg, Xte, names, pos_label="WON", neg_label="NOWIN")
    pick = list(np.where((dec == 1) & (yte == 1))[0][:1]) + list(np.where((dec == 0) & (yte == 0))[0][:1]) \
        + list(np.where(dec == -1)[0][:1])
    for ri in pick:
        e = expl[ri]
        truth = "won" if yte[ri] else "nowin"
        print(f"  position {ri}: {e['why'][:120]}\n     -> {e['decision']}  (actual {truth})")


if __name__ == "__main__":
    sys.exit(main())
