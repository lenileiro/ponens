#!/usr/bin/env python3
"""tabular -- DATA-AS-PROMPT pattern finder for real tabular data (the long-term vision: hand it a dataset,
it finds the patterns and predicts). First instance: the Sendy logistics ETA challenge -- predict
'Time from Pickup to Arrival' (seconds) from order/time/geo/weather/rider features. The 'verifier' for this
domain is HELD-OUT ERROR (not exact reproduction): we fit on a train split and score RMSE on unseen rows, so
the model can only claim a pattern that actually generalizes.

Pipeline: ingest CSVs -> auto-engineer features (parse clock times, haversine from lat/lon, pre-pickup event
gaps, join rider metrics, cyclical time, categoricals) -> fit a gradient-boosted pattern finder -> held-out
RMSE + the patterns it found (feature importances). Only features KNOWN AT PICKUP are used (no leakage;
'Arrival at Destination' is the target endpoint and is excluded -- it's also absent from the test set).

    python -m thinking.tabular --data /path/to/sendy-dir
"""
import argparse
import math
import sys

import numpy as np
import pandas as pd


def to_sec(t):
    """'9:35:46 AM' -> seconds since midnight (NaN-safe)."""
    if not isinstance(t, str):
        return np.nan
    try:
        hms, ap = t.strip().split(" ")
        h, m, s = (int(x) for x in hms.split(":"))
        h = h % 12 + (12 if ap.upper() == "PM" else 0)
        return h * 3600 + m * 60 + s
    except Exception:
        return np.nan


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = math.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def engineer(df, riders):
    """Build a numeric feature matrix from the raw Sendy columns -- only signals known AT PICKUP."""
    X = pd.DataFrame(index=df.index)
    X["distance_km"] = df["Distance (KM)"]
    X["temperature"] = df["Temperature"]
    X["precip"] = df["Precipitation in millimeters"].fillna(0.0)
    # geometry from coordinates (a learned route is longer than straight-line, but this anchors it)
    pla, plo = df["Pickup Lat"], df["Pickup Long"]
    dla, dlo = df["Destination Lat"], df["Destination Long"]
    X["haversine_km"] = haversine(pla, plo, dla, dlo)
    X["d_lat"] = (dla - pla).abs()
    X["d_long"] = (dlo - plo).abs()
    X["pickup_lat"], X["pickup_long"] = pla, plo
    # clock times (seconds of day) + PRE-PICKUP event gaps (all known before the trip starts)
    for c in ["Placement", "Confirmation", "Arrival at Pickup", "Pickup"]:
        X[f"{c}_sec"] = df[f"{c} - Time"].map(to_sec)
    X["gap_place_conf"] = X["Confirmation_sec"] - X["Placement_sec"]
    X["gap_conf_arr"] = X["Arrival at Pickup_sec"] - X["Confirmation_sec"]
    X["gap_arr_pickup"] = X["Pickup_sec"] - X["Arrival at Pickup_sec"]      # rider prep/wait time
    # cyclical time-of-day + weekday (congestion patterns)
    h = X["Pickup_sec"] / 86400.0
    X["pickup_sin"], X["pickup_cos"] = np.sin(2 * np.pi * h), np.cos(2 * np.pi * h)
    X["weekday"] = df["Pickup - Weekday (Mo = 1)"]
    X["day"] = df["Pickup - Day of Month"]
    X["platform"] = df["Platform Type"]
    X["business"] = (df["Personal or Business"] == "Business").astype(int)
    # join rider metrics (experience -> speed)
    r = riders.set_index("Rider Id")
    for col in ["No_Of_Orders", "Age", "Average_Rating", "No_of_Ratings"]:
        X[col] = df["Rider Id"].map(r[col])
    X["orders_per_day"] = X["No_Of_Orders"] / (X["Age"] + 1)
    return X


def run(data_dir, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    tr = pd.read_csv(f"{data_dir}/Train.csv")
    riders = pd.read_csv(f"{data_dir}/Riders.csv")
    y = tr["Time from Pickup to Arrival"].astype(float).values
    X = engineer(tr, riders)
    feats = list(X.columns)
    Xv = X.values.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.85 * len(y))
    tri, vai = idx[:cut], idx[cut:]
    model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_depth=None,
                                          max_leaf_nodes=63, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.1,
                                          random_state=seed)
    model.fit(Xv[tri], y[tri])
    pred = model.predict(Xv[vai])
    rmse = math.sqrt(((pred - y[vai]) ** 2).mean())
    # baselines on the SAME held-out rows
    base_mean = math.sqrt(((y[tri].mean() - y[vai]) ** 2).mean())
    b = np.polyfit(Xv[tri, feats.index("distance_km")], y[tri], 1)
    base_dist = math.sqrt(((np.polyval(b, Xv[vai, feats.index("distance_km")]) - y[vai]) ** 2).mean())
    # patterns found = permutation importance (robust, model-agnostic)
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(model, Xv[vai], y[vai], n_repeats=5, random_state=seed,
                                 scoring="neg_root_mean_squared_error")
    order = np.argsort(-imp.importances_mean)
    print(f"  Sendy ETA: {len(y)} orders, {len(feats)} engineered features, held-out 15%")
    print(f"  RMSE (seconds):  predict-mean {base_mean:.0f}  |  distance-only {base_dist:.0f}  |  "
          f"PATTERN-FINDER {rmse:.0f}")
    print(f"  improvement over distance-only: {100 * (base_dist - rmse) / base_dist:.1f}%")
    print("  top patterns it found (held-out RMSE rise if shuffled):")
    for i in order[:10]:
        print(f"     {feats[i]:<18} +{imp.importances_mean[i]:.0f} sec")
    return rmse


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="dir with Train.csv / Riders.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.data, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
