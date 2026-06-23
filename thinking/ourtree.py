#!/usr/bin/env python3
"""ourtree -- our OWN gradient-boosted regression trees (no sklearn in the predictor). This IS the
verified-search paradigm applied to real tabular data: each tree is grown by GREEDY VERIFIED SPLIT SEARCH
(a split is taken only if it reduces squared error -- the histogram variance-reduction gain), and boosting
is literally ITERATE-ON-RESIDUAL (fit each new tree to what the model still gets wrong). Held-out early
stopping is the verifier that says when to stop adding patterns. Histogram-binned splits (quantile bins per
feature) make it fast; NaN is imputed to the per-feature median. The learned "program" = intercept + a sum
of shallow decision trees over the engineered features.
"""
import numpy as np


def _best_split(Xb, r, nb, min_leaf):
    """Best (feature, bin-threshold) by histogram variance-reduction gain. Returns (gain, feat, thresh)."""
    n = len(r); tot = r.sum()
    base = tot * tot / n
    best_gain, best_f, best_t = 0.0, -1, -1
    for j in range(Xb.shape[1]):
        col = Xb[:, j]
        sr = np.bincount(col, weights=r, minlength=nb)
        sc = np.bincount(col, minlength=nb).astype(np.float64)
        csr = np.cumsum(sr)[:-1]; csc = np.cumsum(sc)[:-1]
        ls, lc = csr, csc
        rs, rc = tot - ls, n - lc
        ok = (lc >= min_leaf) & (rc >= min_leaf)
        if not ok.any():
            continue
        gain = np.where(ok, ls * ls / np.maximum(lc, 1) + rs * rs / np.maximum(rc, 1) - base, -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best_gain:
            best_gain, best_f, best_t = float(gain[k]), j, k
    return best_gain, best_f, best_t


def _grow(Xb, r, idx, depth, max_depth, min_leaf, nb):
    node = {"val": float(r[idx].mean())}
    if depth >= max_depth or len(idx) < 2 * min_leaf:
        return node
    gain, f, t = _best_split(Xb[idx], r[idx], nb, min_leaf)
    if f < 0 or gain <= 1e-9:
        return node
    go_left = Xb[idx, f] <= t
    li, ri = idx[go_left], idx[~go_left]
    if len(li) < min_leaf or len(ri) < min_leaf:
        return node
    node["f"], node["t"] = f, t
    node["L"] = _grow(Xb, r, li, depth + 1, max_depth, min_leaf, nb)
    node["R"] = _grow(Xb, r, ri, depth + 1, max_depth, min_leaf, nb)
    return node


def _predict_tree(node, Xb, idx, out):
    if "f" not in node:
        out[idx] = node["val"]; return
    go_left = Xb[idx, node["f"]] <= node["t"]
    _predict_tree(node["L"], Xb, idx[go_left], out)
    _predict_tree(node["R"], Xb, idx[~go_left], out)


class GBT:
    """Our gradient-boosted regression trees. sklearn-compatible fit(X,y)/predict(X)."""
    def __init__(self, n_estimators=300, lr=0.05, max_depth=4, min_leaf=20, n_bins=64,
                 val_frac=0.1, patience=25, seed=0):
        self.n_estimators, self.lr, self.max_depth = n_estimators, lr, max_depth
        self.min_leaf, self.n_bins, self.val_frac = min_leaf, n_bins, val_frac
        self.patience, self.seed = patience, seed

    def _fit_bins(self, X):
        self.medians = np.nanmedian(X, 0)
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        self.edges = []
        for j in range(X.shape[1]):
            v = X[:, j][~np.isnan(X[:, j])]
            q = np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1]) if len(v) else np.array([0.0])
            self.edges.append(np.unique(q))
        self.nb = max(len(e) for e in self.edges) + 2

    def _bin(self, X):
        X = np.where(np.isnan(X), self.medians, X)
        Xb = np.empty(X.shape, dtype=np.int64)
        for j in range(X.shape[1]):
            Xb[:, j] = np.searchsorted(self.edges[j], X[:, j])
        return Xb

    def fit(self, X, y):
        X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)
        self._fit_bins(X); Xb = self._bin(X)
        rng = np.random.default_rng(self.seed); n = len(y)
        perm = rng.permutation(n); k = int((1 - self.val_frac) * n)
        tri, vai = perm[:k], perm[k:]
        self.base = float(y[tri].mean())
        pred_tr = np.full(n, self.base)
        self.trees = []
        best_vrmse, best_m, bad = np.inf, 0, 0
        for m in range(self.n_estimators):
            resid = y - pred_tr
            tree = _grow(Xb, resid, tri, 0, self.max_depth, self.min_leaf, self.nb)
            contrib = np.empty(n)
            _predict_tree(tree, Xb, np.arange(n), contrib)
            pred_tr = pred_tr + self.lr * contrib
            self.trees.append(tree)
            vrmse = float(np.sqrt(((pred_tr[vai] - y[vai]) ** 2).mean()))
            if vrmse < best_vrmse - 1e-6:
                best_vrmse, best_m, bad = vrmse, m + 1, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.trees = self.trees[:best_m]          # keep only the verified-helpful trees (held-out early stop)
        return self

    def predict(self, X):
        Xb = self._bin(np.asarray(X, np.float64))
        out = np.full(len(Xb), self.base)
        ar = np.arange(len(Xb))
        for tree in self.trees:
            contrib = np.empty(len(Xb))
            _predict_tree(tree, Xb, ar, contrib)
            out = out + self.lr * contrib
        return out


def selftest():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(0, 1, (n, 6))
    y = 3 * X[:, 0] + 2 * (X[:, 1] > 0) * X[:, 2] - X[:, 3] ** 2 + rng.normal(0, 0.5, n)
    tr, va = slice(0, 3000), slice(3000, n)
    m = GBT(n_estimators=300, lr=0.06, max_depth=4).fit(X[tr], y[tr])
    p = m.predict(X[va])
    rmse = float(np.sqrt(((p - y[va]) ** 2).mean()))
    base = float(y[:3000].std())
    print(f"  ourtree selftest: held-out RMSE {rmse:.3f} vs predict-mean {base:.3f}, {len(m.trees)} trees")
    assert rmse < 0.4 * base, "GBT failed to fit a learnable signal"
    print("ourtree selftest OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
