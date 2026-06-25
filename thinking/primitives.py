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

from thinking.azr_win import (_conj, apply_rule, boost_clf_proba, boost_softmax_predict, explain_selective,  # noqa
                              predict_calibrated, predict_selective, reason_adaptive, reason_boost_clf,
                              reason_boost_softmax, reason_selective_cv, reason_tree_rule, render_boost)


def _top_var(df, cols, k):
    """The k highest-variance columns (deterministic) -- bounds the O(D^2) pairwise primitive generation to
    O(k^2) on wide data without dropping the informative columns."""
    if len(cols) <= k:
        return cols
    v = {c: float(pd.to_numeric(df[c], errors="coerce").var() or 0.0) for c in cols}
    return sorted(cols, key=lambda c: -v[c])[:k]


def _column_groups(df, cols, max_card=15):
    """Group columns that share the same value DOMAIN -- parallel measurements (card ranks, suits, dice faces,
    Likert survey items). Counts/multiplicities WITHIN such a group are meaningful (a 'pair' = a value appearing
    in 2 of the group's columns), whereas counting across unrelated columns conflates incomparable values."""
    groups = {}
    for c in cols:
        u = df[c].dropna().unique()
        if 2 <= len(u) <= max_card:
            key = frozenset(np.round(u.astype(float), 4)) if pd.api.types.is_numeric_dtype(df[c]) else frozenset(map(str, u))
            groups.setdefault(key, []).append(c)
    return [g for g in groups.values() if len(g) >= 2]


def _group_primitives(df, cols):
    """GROUP-AWARE counts: within each same-domain column group, count(==v), within-group n-distinct, and max
    multiplicity (the largest count any single value attains in a row). max-multiplicity is the general
    pair/trips/quad detector (poker: rank-group maxmult=2/3/4; suit-group count==5 / ndistinct==1 = flush)."""
    cands = []
    for g in _column_groups(df, cols):
        M = df[g].values; vals = list(pd.unique(M.ravel()))[:15]; tag = f"{g[0]}..{g[-1]}"
        cnts = np.stack([(M == v).sum(1) for v in vals], 1)             # (n, |vals|) value-counts per row
        for j, v in enumerate(vals):
            cands.append((f"grpcount({tag}=={v})", cnts[:, j].astype(float)))
        cands.append((f"grpmaxmult({tag})", cnts.max(1).astype(float)))   # pair=2, trips=3, quad=4
        cands.append((f"grpndistinct({tag})", (cnts > 0).sum(1).astype(float)))
    return cands


def candidate_primitives(df, cols, max_pair_cols=40):
    """Generate derived-feature candidates (name, float array) from the feature library:
    count(==v) across columns, parity of a count, column-equality, row n-distinct, numeric pair sum/diff, and
    GROUP-AWARE counts/multiplicities over same-domain column groups (parallel measurements; the pair/trips/
    flush detectors). Pairwise primitives (O(D^2)) are capped to the top-`max_pair_cols` columns by variance so
    the library stays bounded on wide data; the O(D) count/parity/ndistinct primitives still range over all."""
    cands = list(_group_primitives(df, cols))
    discrete = [c for c in cols if df[c].nunique() <= 12]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 12]
    if len(discrete) >= 2:
        Dv, vals = _coded_discrete(df, discrete)
        for i, v in enumerate(vals):
            cnt = (Dv == i).sum(axis=1).astype(float)               # count(== v) across all discrete columns
            cands.append((f"count(=={v})", cnt))
            if cnt.max() > 1:
                cands.append((f"count(=={v})%2", (cnt % 2)))          # parity of a count (parity concepts)
        for a, b in combinations(_top_var(df, discrete, max_pair_cols), 2):   # relational: col_i == col_j (capped)
            cands.append((f"({a}=={b})", (df[a].values == df[b].values).astype(float)))
        s = np.sort(Dv, axis=1)                                      # vectorized distinct-per-row (was a py loop)
        cands.append(("row_ndistinct", (1 + (np.diff(s, axis=1) != 0).sum(1)).astype(float)))
    for a, b in combinations(_top_var(df, numeric, max_pair_cols), 2):        # numeric pair sums/diffs (capped)
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
    """One-hot categorical/low-card base cols (-> per-value presence literals, the right form for rule
    learning at any cardinality), keep high-card numerics as-is, append derived primitive columns. Already-binary
    {0,1} columns (e.g. bag-of-words presence) are kept AS-IS -- one-hot would double them into redundant
    col=0/col=1 pairs and mangle rule names; a single presence column reads cleanly ('has_free==1')."""
    binary = [c for c in base_cols if pd.api.types.is_numeric_dtype(df[c]) and set(pd.unique(df[c].dropna())) <= {0, 1}]
    cat = [c for c in base_cols if c not in binary and ((not pd.api.types.is_numeric_dtype(df[c])) or df[c].nunique() <= 12)]
    num = [c for c in base_cols if c not in cat and c not in binary]
    blocks, names = [], []
    if binary:
        blocks.append(df[binary].values.astype(float)); names += list(binary)   # presence already -> keep single col
    if cat:
        oh = pd.get_dummies(df[cat].astype(str))
        blocks.append(oh.values.astype(float)); names += [str(c).replace("_", "=") for c in oh.columns]
    if num:
        blocks.append(df[num].values.astype(float)); names += list(num)
    for nm, ser in derived:
        blocks.append(np.asarray(ser, dtype=float).reshape(-1, 1)); names.append(nm)
    X = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0), dtype=float)
    return X, names


def _score(X, y, seed=0):
    """Held-out confident-correctness of a SHORT (depth<=2) verified rule over these features.
    Rewards coverage AND accuracy: fraction of validation rows answered AND correct."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
    tr, va = idx[:cut], idx[cut:]
    pos, neg = reason_selective_cv(X[tr], y[tr], proposer=None, device="cpu", seed=seed, folds=4,
                                   min_prec_pos=0.97, min_prec_neg=0.97, min_support=4, exhaustive=True,
                                   compress=True, build=dict(max_lit=2, max_lits=40, binary_presence_only=True),
                                   verbose=False)
    dec = predict_selective(pos, neg, X[va])
    return float(((dec >= 0) & (dec == y[va])).mean()), (pos, neg)


def _compose(df, base_cols, cands, prank, m=6):
    """Cross the composition-myopia wall: feed the top-m level-1 derived features back in as columns and
    regenerate primitives over them -> level-2 candidates like (count(==1)==count(==2)) appear as SINGLE
    candidates. So a primitive whose intermediate is useless alone can still be discovered as a composition,
    without greedy ever having to add the useless intermediate."""
    top = sorted(cands, key=lambda p: -prank(p[1]))[:m]
    if not top:
        return cands
    df2 = df.copy()
    for nm, ser in top:
        df2[nm] = np.asarray(ser)
    seen = {nm for nm, _ in cands}
    composed = [(nm, ser) for nm, ser in candidate_primitives(df2, base_cols + [nm for nm, _ in top])
                if nm not in seen]
    return cands + composed


def propose(df, target, base_cols, rounds=3, topk=8, ranker=None, compose=False, verbose=True):
    """Greedily add the primitive that most improves held-out confident-correctness; stop when none helps.
    Each round pre-ranks the whole library and only full-evaluates (verified rule search) the top-K -- so the
    library can grow without the eval cost exploding. Pre-rank uses the LEARNED ranker if given, else the
    cheap heuristic. compose=True also generates level-2 (primitive-of-a-primitive) candidates."""
    y = (df[target].values == sorted(pd.unique(df[target]), key=str)[-1]).astype(int)   # binarize any 2-class target
    cands = candidate_primitives(df, base_cols)
    prank = (lambda c: rank_score(ranker, c, y)) if ranker is not None else (lambda c: _cheap_score(c, y))
    if compose:
        cands = _compose(df, base_cols, cands, prank)
    if verbose:
        print(f"  library: {len(cands)} candidate primitives  (pre-rank: {'learned' if ranker else 'heuristic'}"
              f"{', +composition' if compose else ''})")
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


def _render(rules, names):
    if not rules:
        return "(none)"
    return " OR ".join("(" + " AND ".join(f"{names[j]}{o}{t:g}" for j, o, t in _conj(c)) + ")" for c in rules)


def _build_features(df, base_cols, chosen_names):
    """Rebuild raw + chosen-derived feature matrix on ANY df with the same columns (for fit and predict)."""
    by = {nm: ser for nm, ser in candidate_primitives(df, base_cols)}
    derived = [(nm, by[nm]) for nm in chosen_names if nm in by]
    return _featmat(df, base_cols, derived)


def _select_cols(df, target, base_cols, k, regression=False):
    """Cheap verified-ish feature SELECTION for wide data: rank raw columns by max |correlation| with the
    target (class indicators for classification, or the value itself for regression) -- vectorized over all
    numeric columns at once -- and keep the top-k. This is the reasoner choosing which columns are worth
    considering -- O(D*n), so the O(D^2) primitive/rule machinery downstream only ever sees a bounded column
    set. Returns the kept column names (original order preserved)."""
    if len(base_cols) <= k:
        return base_cols
    if regression:
        yv = pd.to_numeric(df[target], errors="coerce").fillna(0.0).values.astype(float)
        Y = (yv - yv.mean()).reshape(-1, 1)
    else:
        codes = df[target].astype("category").cat.codes.values
        Y = np.eye(int(codes.max()) + 1)[codes]; Y = Y - Y.mean(0)            # centered class indicators
    num = [c for c in base_cols if pd.api.types.is_numeric_dtype(df[c])]
    score = {c: 0.0 for c in base_cols}
    if num:
        M = df[num].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
        M = (M - M.mean(0)) / (M.std(0) + 1e-9)
        corr = np.abs(M.T @ Y) / len(df)                                      # (n_num, n_target_cols)
        for c, s in zip(num, corr.max(1)):
            score[c] = float(s)
    for c in (set(base_cols) - set(num)):                                     # few categoricals: per-column proxy
        codes = df[target].astype("category").cat.codes.values
        score[c] = _cheap_score(df[c].astype("category").cat.codes.values, (codes == np.bincount(codes).argmax()).astype(int))
    keep = set(sorted(base_cols, key=lambda c: -score[c])[:k])
    return [c for c in base_cols if c in keep]                                # preserve original column order


def _class_confidence(pos, X):
    """For a one-vs-rest 'is-class' rule: per-row (fired?, confidence = max firing-clause precision)."""
    fired = np.zeros(len(X), bool); conf = np.zeros(len(X))
    for c in pos:
        m = apply_rule([c], X).astype(bool)
        fired |= m; conf = np.maximum(conf, m * float(c.get("prec", 1.0)))
    return fired, conf


def solve_any(df, target, id_col=None, ranker=None, test_frac=0.3, min_prec=0.99, compose=True, tau=None, clf=None, adaptive=False, max_cols=200, seed=0, verbose=True):
    """Capstone: one call on any tabular CLASSIFICATION dataset (binary or multiclass). Trains the
    primitive-ranker via self-play (if not given), proposes+verifies derived primitives, reasons verified
    rules (two-sided for binary; one-vs-rest for multiclass), and returns the rule(s), held-out
    coverage/accuracy, and (binary) per-row explanations -- zero training on the data, abstaining where it
    can't prove an answer (no firing rule, or a multiclass tie)."""
    df = df.reset_index(drop=True)
    base_cols = [c for c in df.columns if c not in (target, id_col)]
    classes = sorted(pd.unique(df[target].dropna()).tolist(), key=lambda x: str(x))
    if len(classes) < 2:
        raise ValueError(f"need >=2 classes; got {classes}")
    if max_cols and len(base_cols) > max_cols:                       # wide data: select informative columns first
        base_cols = _select_cols(df, target, base_cols, max_cols)
        if verbose:
            print(f"  [select] {len(base_cols)} of many columns kept (top |corr| with target)")
    rng = np.random.default_rng(seed); idx = rng.permutation(len(df)); cut = int((1 - test_frac) * len(df))
    tri, tei = idx[:cut], idx[cut:]
    trip = tri if len(tri) <= 60000 else rng.choice(tri, 60000, replace=False)   # cap rows for feature discovery
    if ranker is None:
        if verbose:
            print("  [learn-to-rank] training primitive-ranker via self-play (zero data)...")
        ranker = selfplay_rank(steps=150, seed=seed, verbose=False)

    if len(classes) > 2:                                              # ---- multiclass: one-vs-rest ----
        if verbose:
            print(f"  [multiclass] {len(classes)} classes, one-vs-rest verified rules...")
        names_chosen = []
        for c in classes:                                            # discover primitives per class, union
            dfc = df.copy(); dfc["__y__"] = (df[target].values == c).astype(int)
            ch, _ = propose(dfc.iloc[trip].reset_index(drop=True), "__y__", base_cols, ranker=ranker,
                            rounds=1, compose=compose, verbose=False)
            names_chosen += [x[0] for x in ch if x[0] not in names_chosen]
        Xfull, names = _build_features(df, base_cols, names_chosen)
        Xtr, Xte = Xfull[tri], Xfull[tei]
        if clf == "boost":                                          # full-coverage multinomial (softmax) boosting
            model = reason_boost_softmax(Xtr, df[target].values[tri], classes=classes, seed=seed)
            pred, _, _ = boost_softmax_predict(model, Xte); yte = df[target].values[tei]
            return {"task": f"{len(classes)}-class classification", "classes": classes, "clf": "boost-softmax",
                    "coverage": 1.0, "accuracy_on_answered": float((pred == yte).mean()),
                    "n_trees": len(model[2])}
        if tau is not None:                                          # calibrated mode: relaxed rules + tau gate
            ml = max(4, len(tri) // 100)
            cr = {c: reason_tree_rule(Xtr, (df[target].values[tri] == c).astype(int), depth=4, min_leaf=ml,
                                      min_purity=0.55, min_support=4, seed=seed) for c in classes}
            pred, _, ans = predict_calibrated(cr, Xte, tau)
            yte = df[target].values[tei]
            acc = float((pred[ans] == yte[ans]).mean()) if ans.any() else 0.0
            return {"task": f"{len(classes)}-class classification", "classes": classes, "tau": tau,
                    "coverage": float(ans.mean()), "accuracy_on_answered": acc,
                    "n_clauses_per_class": {c: len(cr[c]) for c in classes}}
        rules = {}; conf = np.zeros((len(tei), len(classes)))
        vsel = np.random.default_rng(seed + 5).permutation(len(tri))[int(0.7 * len(tri)):]   # held-out selector
        for ci, c in enumerate(classes):
            yc = (df[target].values == c).astype(int)
            pos_d, _ = reason_selective_cv(Xtr, yc[tri], proposer=None, seed=seed, folds=5, exhaustive=True,
                                           compress=True, min_prec_pos=min_prec, min_prec_neg=min_prec,
                                           build=dict(max_lit=3, max_lits=60, binary_presence_only=True), verbose=False)
            ml = max(4, len(tri) // 100)                            # small leaves for tiny/imbalanced classes
            pos_t = reason_tree_rule(Xtr, yc[tri], depth=4, min_leaf=ml, min_purity=0.75,
                                     min_support=max(3, ml // 2), seed=seed)
            acc_d = (apply_rule(pos_d, Xtr[vsel]) == yc[tri][vsel]).mean()
            acc_t = (apply_rule(pos_t, Xtr[vsel]) == yc[tri][vsel]).mean()
            pos = pos_t if acc_t >= acc_d else pos_d                 # pick the better one-vs-rest rule per class
            rules[c] = pos
            _, cf = _class_confidence(pos, Xte); conf[:, ci] = cf
        fired = conf.max(1) > 0
        pred = np.array(classes, dtype=object)[conf.argmax(1)]
        yte = df[target].values[tei]
        ans = fired
        acc = float((pred[ans] == yte[ans]).mean()) if ans.any() else 0.0
        return {"task": f"{len(classes)}-class classification", "classes": classes,
                "discovered_primitives": names_chosen, "coverage": float(ans.mean()),
                "accuracy_on_answered": acc, "n_clauses_per_class": {c: len(rules[c]) for c in classes},
                "rules": {c: _render(rules[c], names) for c in classes}}

    if verbose:                                                      # ---- binary: two-sided + explanations ----
        print("  [propose] verified primitive discovery...")
    chosen, _ = propose(df.iloc[trip].reset_index(drop=True), target, base_cols, ranker=ranker, compose=compose, verbose=verbose)
    names_chosen = [c[0] for c in chosen]
    Xfull, names = _build_features(df, base_cols, names_chosen)        # build on FULL df -> consistent columns
    Xtr, Xte = Xfull[tri], Xfull[tei]
    yfull = (df[target].values == classes[1]).astype(int)
    ytr, yte = yfull[tri], yfull[tei]
    if clf == "boost":                                              # full-coverage boosted-rule (no abstention)
        m = reason_boost_clf(Xtr, ytr, seed=seed); pred = (boost_clf_proba(m, Xte) > 0.5).astype(int)
        return {"task": "binary classification", "classes": classes, "clf": "boost", "coverage": 1.0,
                "accuracy_on_answered": float((pred == yte).mean()), "n_rules": len(m[2]),
                "rules": render_boost(m, names)}
    if verbose:
        print("  [reason] verified two-sided rule (exhaustive + compress)...")
    pos, neg = reason_selective_cv(Xtr, ytr, proposer=None, seed=seed, folds=5, exhaustive=True, compress=True,
                                   min_prec_pos=min_prec, min_prec_neg=min_prec,
                                   build=dict(max_lit=3, max_lits=60, binary_presence_only=True), verbose=False)
    pos_t = reason_tree_rule(Xtr, ytr, seed=seed)                     # CART tree-path DNF (helps continuous feats)
    neg_t = reason_tree_rule(Xtr, 1 - ytr, seed=seed)
    vsel = np.random.default_rng(seed + 5).permutation(len(tri))[int(0.7 * len(tri)):]   # held-out selector

    def _cc(p, n):
        d = predict_selective(p, n, Xtr[vsel]); ans = d >= 0
        return float((d[ans] == ytr[vsel][ans]).sum()) / len(vsel)    # confident-correct fraction
    rule_mode = "dnf"
    if _cc(pos_t, neg_t) > _cc(pos, neg):
        pos, neg, rule_mode = pos_t, neg_t, "tree-paths"
    if adaptive:                                                      # think-harder: complete escalation on the residual
        if verbose:
            print("  [adaptive] escalating complete verified search on the abstained residual...")
        pos, neg = reason_adaptive(Xtr, ytr, seed_pos=pos, seed_neg=neg, min_prec_pos=min_prec,
                                   min_prec_neg=min_prec, seed=seed, names=names, verbose=verbose)
        rule_mode = "adaptive"
    if tau is not None:                                              # calibrated mode: relaxed rules + tau gate
        ml = max(4, len(tri) // 100)
        pos_c = reason_tree_rule(Xtr, ytr, depth=4, min_leaf=ml, min_purity=0.55, min_support=4, seed=seed)
        neg_c = reason_tree_rule(Xtr, 1 - ytr, depth=4, min_leaf=ml, min_purity=0.55, min_support=4, seed=seed)
        predc, _, ansc = predict_calibrated({classes[0]: neg_c, classes[1]: pos_c}, Xte, tau)
        ytec = df[target].values[tei]
        return {"task": "binary classification", "classes": classes, "tau": tau,
                "coverage": float(ansc.mean()),
                "accuracy_on_answered": float((predc[ansc] == ytec[ansc]).mean()) if ansc.any() else 0.0}
    if verbose:
        print(f"  rule source: {rule_mode}")
    dec = predict_selective(pos, neg, Xte); a = dec >= 0
    return {"task": "binary classification", "classes": classes, "discovered_primitives": names_chosen,
            "rule_mode": rule_mode,
            "rule_positive": _render(pos, names), "rule_negative": _render(neg, names),
            "coverage": float(a.mean()),
            "accuracy_on_answered": float((dec[a] == yte[a]).mean()) if a.any() else 0.0,
            "n_clauses": (len(pos), len(neg)),
            "explanations": explain_selective(pos, neg, Xte, names, str(classes[1]), str(classes[0]))}


def main():
    df = pd.read_csv("kaggle_data/monk/monk.csv")
    print("CAPSTONE -- solve_any(df, target): one call, any binary tabular dataset (here Monk-2).\n")
    res = solve_any(df, "'class'", verbose=True)
    print(f"\n  discovered primitives : {res['discovered_primitives']}")
    print(f"  POSITIVE when         : {res['rule_positive']}")
    print(f"  coverage              : {res['coverage']:.3f}")
    print(f"  accuracy-on-answered  : {res['accuracy_on_answered']:.3f}")
    print(f"  clauses (pos,neg)     : {res['n_clauses']}")
    print("  sample explanations:")
    for e in res["explanations"][:3]:
        print(f"    {e['decision']:<10} ({e['reason_type']}) -- {e['why'][:80]}")


if __name__ == "__main__":
    sys.exit(main())
