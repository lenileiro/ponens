#!/usr/bin/env python3
"""autotab -- AUTOMATIC TRANSFORM DISCOVERY for arbitrary tabular data (the general "solve any data" step).
Given a CSV and a target column, it infers each column's ROLE, generates candidate feature groups from a
TRANSFORM LIBRARY, and GREEDILY SELECTS the groups that lower HELD-OUT error -- the model engineers its own
features and keeps only the patterns that verify (held-out RMSE is the verifier, the noisy-data analog of
exact-reproduction in azr_win). No dataset-specific code. If a test CSV is given, only columns present in
BOTH are used -> post-outcome leakage columns (absent from test) are auto-dropped.

Transform library (per inferred role):
  num        -> passthrough            time (clock str) -> seconds-of-day + sin/cos
  lat/lon    -> haversine + deltas between detected points
  cat (low)  -> frequency + one-hot    id (high-card)   -> smoothed target encoding (train-only, leak-safe)
  time-gaps  -> pairwise differences of clock-time columns (pre-event durations)

    python -m thinking.autotab --train Train.csv --target "Time from Pickup to Arrival" --test Test.csv
"""
import argparse
import math
import re
import sys

import numpy as np
import pandas as pd

TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}\s*(AM|PM)\s*$", re.I)


def infer_role(s, name):
    nm = name.lower()
    nn = s.dropna()
    if len(nn) == 0 or s.nunique() <= 1:
        return "drop"
    if nn.astype(str).head(50).str.match(TIME_RE).mean() > 0.8:
        return "time"
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        if "lat" in nm:
            return "lat"
        if "long" in nm or "lon" in nm:
            return "lon"
        return "cat_num" if num.nunique() <= 12 else "num"
    return "cat" if s.nunique() <= 20 else "id"


def to_sec(t):
    if not isinstance(t, str):
        return np.nan
    try:
        hms, ap = t.strip().split(" ")
        h, m, sec = (int(x) for x in hms.split(":"))
        return (h % 12 + (12 if ap.upper() == "PM" else 0)) * 3600 + m * 60 + sec
    except Exception:
        return np.nan


def haversine(a, b, c, d):
    R, p = 6371.0, math.pi / 180
    x = (np.sin((c - a) * p / 2) ** 2 + np.cos(a * p) * np.cos(c * p) * np.sin((d - b) * p / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(x, 0, 1)))


def build_groups(tr, va, y_tr, roles):
    """Return {group_name: (tr_features_df, va_features_df)} from the transform library. Target encoding
    uses TRAIN ONLY (val gets train means) -> the held-out metric stays leak-free."""
    groups = {}

    def add(name, ftr, fva):
        groups[name] = (ftr.reset_index(drop=True), fva.reset_index(drop=True))

    num = [c for c, r in roles.items() if r in ("num", "cat_num")]
    if num:
        add("numeric", tr[num].apply(pd.to_numeric, errors="coerce"),
            va[num].apply(pd.to_numeric, errors="coerce"))
    times = [c for c, r in roles.items() if r == "time"]
    if times:
        ft, fv = pd.DataFrame(), pd.DataFrame()
        for c in times:
            st, sv = tr[c].map(to_sec), va[c].map(to_sec)
            ft[f"{c}__sec"] = st; fv[f"{c}__sec"] = sv
            ft[f"{c}__sin"] = np.sin(2 * np.pi * st / 86400); fv[f"{c}__sin"] = np.sin(2 * np.pi * sv / 86400)
            ft[f"{c}__cos"] = np.cos(2 * np.pi * st / 86400); fv[f"{c}__cos"] = np.cos(2 * np.pi * sv / 86400)
        add("times", ft, fv)
        if len(times) >= 2:                                   # pairwise pre-event durations
            gt, gv = pd.DataFrame(), pd.DataFrame()
            for i in range(len(times)):
                for j in range(i + 1, len(times)):
                    gt[f"gap_{i}_{j}"] = tr[times[j]].map(to_sec) - tr[times[i]].map(to_sec)
                    gv[f"gap_{i}_{j}"] = va[times[j]].map(to_sec) - va[times[i]].map(to_sec)
            add("time_gaps", gt, gv)
    # geo: pair lat/lon columns into points (by name prefix), haversine across point-pairs
    lats = [c for c, r in roles.items() if r == "lat"]
    lons = [c for c, r in roles.items() if r == "lon"]
    pts = []
    for la in lats:
        key = la.lower().replace("lat", "").strip()
        match = [lo for lo in lons if lo.lower().replace("long", "").replace("lon", "").strip() == key]
        if match:
            pts.append((la, match[0]))
    if len(pts) >= 2:
        gt, gv = pd.DataFrame(), pd.DataFrame()
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                la1, lo1 = pts[i]; la2, lo2 = pts[j]
                gt[f"hav_{i}_{j}"] = haversine(tr[la1].values, tr[lo1].values, tr[la2].values, tr[lo2].values)
                gv[f"hav_{i}_{j}"] = haversine(va[la1].values, va[lo1].values, va[la2].values, va[lo2].values)
                gt[f"dlat_{i}_{j}"] = (tr[la2] - tr[la1]).abs().values; gv[f"dlat_{i}_{j}"] = (va[la2] - va[la1]).abs().values
                gt[f"dlon_{i}_{j}"] = (tr[lo2] - tr[lo1]).abs().values; gv[f"dlon_{i}_{j}"] = (va[lo2] - va[lo1]).abs().values
        for la, lo in pts:                                    # raw coords too
            gt[la] = tr[la].values; gv[la] = va[la].values; gt[lo] = tr[lo].values; gv[lo] = va[lo].values
        add("geo", gt, gv)
    cats = [c for c, r in roles.items() if r == "cat"]
    if cats:
        ft, fv = pd.DataFrame(), pd.DataFrame()
        for c in cats:
            freq = tr[c].value_counts(normalize=True)
            ft[f"{c}__freq"] = tr[c].map(freq).values; fv[f"{c}__freq"] = va[c].map(freq).fillna(0).values
        add("categorical", ft, fv)
    ids = [c for c, r in roles.items() if r == "id"]
    if ids:
        ft, fv = pd.DataFrame(), pd.DataFrame()
        gmean = y_tr.mean()
        for c in ids:
            df = pd.DataFrame({"k": tr[c].values, "y": y_tr})
            agg = df.groupby("k")["y"].agg(["mean", "count"])
            sm = (agg["mean"] * agg["count"] + gmean * 20) / (agg["count"] + 20)   # smoothed target enc
            ft[f"{c}__te"] = tr[c].map(sm).fillna(gmean).values
            fv[f"{c}__te"] = va[c].map(sm).fillna(gmean).values
            ft[f"{c}__cnt"] = tr[c].map(agg["count"]).fillna(0).values
            fv[f"{c}__cnt"] = va[c].map(agg["count"]).fillna(0).values
        add("id_targetenc", ft, fv)
    return groups


def rmse_with(groups, names, y_tr, y_va, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xt = pd.concat([groups[n][0] for n in names], axis=1).values.astype(float)
    Xv = pd.concat([groups[n][1] for n in names], axis=1).values.astype(float)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                      l2_regularization=1.0, early_stopping=True, random_state=seed).fit(Xt, y_tr)
    return math.sqrt(((m.predict(Xv) - y_va) ** 2).mean())


def run(train_csv, target, test_csv=None, seed=0):
    tr = pd.read_csv(train_csv)
    cols = [c for c in tr.columns if c != target]
    if test_csv:                                              # use only columns present in test -> drop leakage
        te = pd.read_csv(test_csv)
        common = [c for c in cols if c in te.columns]
        dropped = [c for c in cols if c not in te.columns]
        cols = common
        if dropped:
            print(f"  leakage-guard: dropped {len(dropped)} cols absent from test (e.g. {dropped[:2]})")
    y = tr[target].astype(float).values
    roles = {c: infer_role(tr[c], c) for c in cols}
    roles = {c: r for c, r in roles.items() if r != "drop"}
    print(f"  inferred roles: " + ", ".join(f"{r}:{sum(1 for x in roles.values() if x==r)}"
          for r in sorted(set(roles.values()))))
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); cut = int(0.85 * len(y))
    tri, vai = idx[:cut], idx[cut:]
    trd, vad = tr.iloc[tri], tr.iloc[vai]
    groups = build_groups(trd, vad, y[tri], roles)
    print(f"  transform library produced {len(groups)} feature groups: {list(groups)}")
    # GREEDY FORWARD SELECTION verified by held-out RMSE
    selected, remaining = [], list(groups)
    best = math.sqrt(((y[tri].mean() - y[vai]) ** 2).mean())
    print(f"  baseline (predict-mean) RMSE: {best:.0f}")
    while remaining:
        scored = [(rmse_with(groups, selected + [g], y[tri], y[vai], seed), g) for g in remaining]
        r, g = min(scored)
        if r < best - 1.0:
            selected.append(g); remaining.remove(g); best = r
            print(f"  + discovered '{g}' -> held-out RMSE {r:.0f}")
        else:
            break
    print(f"\n  DISCOVERED PIPELINE: {selected}")
    print(f"  final held-out RMSE: {best:.0f} sec  (vs hand-engineered 734)")
    return best, selected


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--test", default=None); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.train, args.target, args.test, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
