#!/usr/bin/env python3
"""azr_reg -- azr_win's VERIFIED SEARCH extended to REGRESSION (arithmetic ops + error-based verifier). The
exact-match list-transform solver abstains on noisy continuous data; this keeps the SAME paradigm --
iterate-execute-RESIDUAL + keep-only-what-verifies -- but with a regression op-library and a soft verifier:

  * program  = intercept + sum of feature-EXPRESSIONS (the regression analog of an op-chain)
  * op-library = linear (f_j), interaction (f_i*f_j), threshold/piecewise (f_j > q)   [arithmetic, not REV/SORT]
  * execute  = evaluate the current program -> a prediction per row
  * verify   = HELD-OUT RMSE reduction (the soft analog of exact reproduction): a term is ADDED only if it
               lowers error on unseen rows -> the model never keeps a pattern that doesn't generalize
  * iterate-on-residual = each step fits the next term's coefficient to the current residual (least squares)

So it finds a verified predictive program by search -- the azr_win technique, now able to express and check a
real regression instead of abstaining. Features come from autotab (auto-join, leakage-guard, transforms).

    python -m thinking.azr_reg --train Train.csv --target "..." --test Test.csv --submit a.csv
"""
import argparse
import math
import sys

import numpy as np
import pandas as pd


def _standardize(Xtr, Xte):
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0, mu)
    Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)
    sd = Xtr.std(0); sd[sd < 1e-6] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def candidate_ops(F, rng, n_inter=120, n_thresh=120):
    """The regression op-library: each op maps a standardized feature matrix -> a column expression."""
    ops = [("lin", j) for j in range(F)]                                   # linear terms
    for _ in range(n_inter):
        i, j = int(rng.integers(0, F)), int(rng.integers(0, F))
        ops.append(("mul", i, j))                                          # interactions
    for _ in range(n_thresh):
        j = int(rng.integers(0, F)); q = float(rng.uniform(-1.5, 1.5))
        ops.append(("thr", j, q))                                          # threshold / piecewise
    return ops


def eval_op(op, X):
    if op[0] == "lin":
        return X[:, op[1]]
    if op[0] == "mul":
        return X[:, op[1]] * X[:, op[2]]
    return (X[:, op[1]] > op[2]).astype(np.float64)                        # thr


def render_op(op, names):
    if op[0] == "lin":
        return names[op[1]]
    if op[0] == "mul":
        return f"{names[op[1]]}*{names[op[2]]}"
    return f"({names[op[1]]}>{op[2]:.1f})"


def solve(Xtr, ytr, Xva, yva, names, lr=0.3, max_terms=60, seed=0, verbose=True):
    """VERIFIED iterate-on-residual search: greedily add the feature-expression that most lowers HELD-OUT RMSE,
    coefficient fit to the residual; stop when no term verifies. Returns the program (list of (op, coef))."""
    rng = np.random.default_rng(seed)
    pred_tr = np.full(len(ytr), ytr.mean()); pred_va = np.full(len(yva), ytr.mean())
    program = [("const", float(ytr.mean()))]
    best = math.sqrt(((pred_va - yva) ** 2).mean())
    if verbose:
        print(f"  intercept -> held-out RMSE {best:.0f}", flush=True)
    for step in range(max_terms):
        resid = ytr - pred_tr
        ops = candidate_ops(Xtr.shape[1], rng)
        best_op, best_c, best_rmse = None, 0.0, best
        for op in ops:
            v = eval_op(op, Xtr); vv = float(v @ v)
            if vv < 1e-9:
                continue
            c = float(v @ resid) / vv                                      # least-squares coef on residual
            r = math.sqrt(((pred_va + lr * c * eval_op(op, Xva) - yva) ** 2).mean())   # VERIFY on held-out
            if r < best_rmse - 1e-6:
                best_op, best_c, best_rmse = op, c, r
        if best_op is None:
            break
        pred_tr = pred_tr + lr * best_c * eval_op(best_op, Xtr)
        pred_va = pred_va + lr * best_c * eval_op(best_op, Xva)
        program.append((best_op, lr * best_c)); best = best_rmse
        if verbose and (step < 6 or step % 10 == 0):
            print(f"  + {render_op(best_op, names):<24} -> held-out RMSE {best:.0f}", flush=True)
    return program, best


def predict(program, X):
    out = np.zeros(len(X))
    for term in program:
        if term[0] == "const":
            out += term[1]
        else:
            out += term[1] * eval_op(term[0], X)
    return out


def run(train_csv, target, test_csv, submit=None, id_col="Order No", seed=0):
    from thinking.autotab import detect_aux, join_aux, infer_role, build_groups
    tr = pd.read_csv(train_csv); auxs = detect_aux(train_csv); tr, _ = join_aux(tr, auxs)
    te, _ = join_aux(pd.read_csv(test_csv), auxs)
    cols = [c for c in tr.columns if c != target and c in te.columns]
    roles = {c: r for c, r in ((c, infer_role(tr[c], c)) for c in cols) if r != "drop"}
    y = tr[target].astype(float).values
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); cut = int(0.85 * len(y))
    tri, vai = idx[:cut], idx[cut:]
    # held-out eval: fit target-encoding on the TRAIN PORTION only (val gets train means) -> no TE leakage
    ge = build_groups(tr.iloc[tri], tr.iloc[vai], y[tri], roles)
    names = [c for n in ge for c in ge[n][0].columns]
    Xtr = pd.concat([ge[n][0] for n in ge], axis=1).values.astype(float)
    Xva = pd.concat([ge[n][1] for n in ge], axis=1).values.astype(float)
    Xtr, Xva = _standardize(Xtr, Xva)
    print(f"  azr_win verified search (REGRESSION ops + error verifier) on {len(names)} features")
    program, rmse = solve(Xtr, y[tri], Xva, y[vai], names, seed=seed)
    print(f"\n  VERIFIED PROGRAM: {len(program)-1} terms; held-out RMSE {rmse:.0f}  (predict-mean {y[tri].std():.0f})")
    if submit:
        gf = build_groups(tr, te, y, roles)                          # full train -> test for the submission
        Xall = pd.concat([gf[n][0] for n in gf], axis=1).values.astype(float)
        Xte = pd.concat([gf[n][1] for n in gf], axis=1).values.astype(float)
        Xall, Xte = _standardize(Xall, Xte)
        full, _ = solve(Xall, y, Xall, y, names, seed=seed, verbose=False)
        pred = np.clip(predict(full, Xte), 0, None)
        ids = pd.read_csv(test_csv)[id_col].values
        pd.DataFrame({"Order_No": ids, "Time from Pickup to Arrival": np.round(pred).astype(int)}).to_csv(submit, index=False)
        print(f"  wrote {len(ids)} predictions -> {submit}")
    return rmse, program


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--test", required=True); ap.add_argument("--submit", default=None)
    ap.add_argument("--id-col", default="Order No"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.train, args.target, args.test, args.submit, args.id_col, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
