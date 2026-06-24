#!/usr/bin/env python3
"""Can azr_win solve a real Kaggle dataset -- and KNOW WHAT IT DOESN'T KNOW? Mushroom edibility
(uciml/mushroom-classification), an exactly-ruled dataset (azr_win's regime, unlike smooth-noise Sendy).

Runtime = pure reasoning, ZERO training on mushroom: the RuleProposer is trained once on self-generated rule
tasks (learn-to-reason). Then azr_win reasons a VERIFIED rule for BOTH classes and ABSTAINS where neither
side has a verified reason (no-evidence) or both do (conflict) -- and THINKS HARDER (escalates search budget,
re-verifying on held-out) to shrink the abstain set without overfitting. The payoff: the one DANGEROUS error
(called edible, was poisonous) becomes an honest 'I don't know' instead of a confident wrong answer.
"""
import sys

import numpy as np
import pandas as pd

from thinking.azr_win import (_conj, apply_rule, explain_selective, predict_selective,  # noqa
                              reason_rule, reason_selective_cv, reason_selective_iter, selfplay_reason)
import torch  # noqa


def render(rules, names):
    if not rules:
        return "(none)"
    out = []
    for c in rules:
        out.append("(" + " AND ".join((names[j] if (o == "==" and t == 1) else
                    (f"NOT {names[j]}" if o == "==" else f"{names[j]}{o}{t:g}")) for j, o, t in _conj(c)) + ")")
    return " OR ".join(out)


def main():
    df = pd.read_csv("kaggle_data/mushroom/mushrooms.csv")
    y_all = (df["class"].values == "p").astype(int)              # 1 = POISONOUS
    Xdf = pd.get_dummies(df.drop(columns=["class"]))
    names = [c.replace("_", "=") for c in Xdf.columns]
    X_all = Xdf.values.astype(np.float32)
    rng = np.random.default_rng(0); idx = rng.permutation(len(y_all)); cut = int(0.7 * len(y_all))
    tri, tei = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X_all[tri], y_all[tri], X_all[tei], y_all[tei]
    print(f"mushroom: {len(df)} rows, {X_all.shape[1]} one-hot features; poisonous rate {y_all.mean():.3f}")

    print("\n[1] learn-to-reason: training RuleProposer on SELF-GENERATED tasks (zero mushroom data)...")
    prop = selfplay_reason(steps=2500, device="cpu", seed=0, verbose=False)
    print("  done.")

    def score(p, n, tag):
        d = predict_selective(p, n, Xte); a = d >= 0
        cov = a.mean(); acc = (d[a] == yte[a]).mean() if a.any() else 0.0
        dang = int(((d == 0) & (yte == 1)).sum())
        print(f"  {tag:<22} coverage {cov:.4f}  accuracy-on-answered {acc:.4f}  dangerous {dang}")
        return d, cov, dang

    print("\n[2a] single-fold verified-selective (baseline ceiling)...")
    pos1, neg1, _ = reason_selective_iter(Xtr, ytr, proposer=prop, device="cpu", seed=0, verbose=False)

    print("[2b] MULTI-FOLD + EXHAUSTIVE search + SET-COVER compression (presence-only literals)...")
    pos, neg = reason_selective_cv(Xtr, ytr, proposer=prop, device="cpu", seed=0, folds=5,
                                   exhaustive=True, compress=True, build=dict(max_lit=3, binary_presence_only=True))
    print(f"\n  POISONOUS when: {render(pos, names)}")
    print(f"  EDIBLE   when: {render(neg, names)}")

    print("\n[3] PREDICT WITH ABSTENTION on held-out test (never seen) -- single-fold vs multi-fold:")
    _, cov1, _ = score(pos1, neg1, "single-fold:")
    dec, cov, dangerous = score(pos, neg, "multi-fold (bagged):")
    guess = apply_rule(reason_rule(Xtr, ytr, proposer=prop, seed=0), Xte)
    print(f"  {'guess-everything:':<22} coverage 1.0000  accuracy {(guess == yte).mean():.4f}  "
          f"dangerous {int(((guess == 0) & (yte == 1)).sum())}")
    print(f"\n  -> multi-fold lifted safe coverage {cov1:.3f} -> {cov:.3f} "
          f"(+{(cov - cov1) * 100:.1f} pts) keeping {dangerous} dangerous errors")
    ans = dec >= 0

    print("\n[4] ANSWER individual mushrooms WITH confidence + why (incl. ones it refuses to guess):")
    expl = explain_selective(pos, neg, Xte, names, pos_label="POISONOUS", neg_label="EDIBLE")
    # one verified-poison, one verified-edible, one no-evidence abstain (the deadly case), one other abstain
    deadly = np.where((dec == -1) & (yte == 1))[0]                      # abstained on an actually-poisonous one
    pick = (list(np.where((dec == 1) & (yte == 1))[0][:1]) + list(np.where((dec == 0) & (yte == 0))[0][:1])
            + list(deadly[:1]) + list(np.where(dec == -1)[0][:1]))
    for ri in dict.fromkeys(pick):
        e = expl[ri]; orig = df.iloc[tei[ri]]
        truth = "poisonous" if yte[ri] == 1 else "edible"
        ok = (e["decision"] == "POISONOUS" and yte[ri] == 1) or (e["decision"] == "EDIBLE" and yte[ri] == 0)
        tag = "correct" if ok else ("ABSTAINED (safe)" if e["decision"] == "ABSTAIN" else "*** WRONG ***")
        conf = f"confidence {e['confidence']:.3f}" if e["reason_type"] == "verified" else f"[{e['reason_type']}]"
        print(f"  odor={orig['odor']}, spore-print={orig['spore-print-color']}, gill-color={orig['gill-color']}:"
              f"\n     why: {e['why']}"
              f"\n     -> ANSWER: {e['decision']}  ({conf})   (actual: {truth})  [{tag}]")


if __name__ == "__main__":
    sys.exit(main())
