#!/usr/bin/env python3
"""Learned REPRESENTATION pre-step for image data -- an in-house, unsupervised patch-prototype encoder that
turns raw pixels (the wrong primitives for threshold-rules) into stroke-activation features the verified
reasoner CAN reason over. No pretrained model, no sklearn, no convnet -- just numpy k-means over local patches
(Coates & Ng 2011: unsupervised k-means features rival CNNs on small data).

Why this fixes the MNIST wall: a rule like "pixel 347 > 0.5" is meaningless, but "stroke-prototype k activates
in the top-left region > t" IS a meaningful, learnable threshold-rule. The encoder learns a dictionary of local
stroke patches (k-means), then encodes each image by how strongly each prototype fires, spatially pooled into a
coarse grid. The discovered features are interpretable (each = a stroke shape x a region) and feed solve_any /
reason_boost unchanged.
"""
import sys

import numpy as np
import pandas as pd


def _kmeans(X, k, iters=40, seed=0):
    """In-house k-means (k-means++ init). Returns centroids (k, d)."""
    rng = np.random.default_rng(seed); n = len(X)
    c = [X[rng.integers(n)]]                                          # k-means++ seeding
    d2 = ((X - c[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / max(d2.sum(), 1e-12); c.append(X[rng.choice(n, p=p)])
        d2 = np.minimum(d2, ((X - c[-1]) ** 2).sum(1))
    C = np.array(c)
    for _ in range(iters):
        a = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), 1)     # assign
        newC = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def _patches(img, patch, stride):
    """All (patch x patch) windows of a HxW image, flattened -> (n_pos, patch*patch). Also returns positions."""
    H, W = img.shape; ps, pos = [], []
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            ps.append(img[i:i + patch, j:j + patch].ravel()); pos.append((i, j))
    return np.array(ps), pos


def _norm(P):
    """Per-patch contrast normalization (subtract mean, divide by std) -- brightness/contrast invariance."""
    P = P - P.mean(1, keepdims=True)
    return P / np.sqrt(P.var(1, keepdims=True) + 10.0)


def learn_dict(images, side=28, n_proto=96, patch=6, n_patches=40000, seed=0):
    """Learn a dictionary of stroke prototypes: sample random normalized patches, k-means them. Unsupervised
    (labels never used) -> the representation is discovered, not fit to the task."""
    rng = np.random.default_rng(seed); imgs = images.reshape(-1, side, side)
    pmax = side - patch + 1; samp = []
    for _ in range(n_patches):
        im = imgs[rng.integers(len(imgs))]; i, j = rng.integers(pmax), rng.integers(pmax)
        samp.append(im[i:i + patch, j:j + patch].ravel())
    P = _norm(np.array(samp, float))
    return {"C": _kmeans(P, n_proto, seed=seed), "patch": patch, "side": side}


def encode(images, D, stride=2, grid=2):
    """Encode each image -> (grid*grid*n_proto) features: prototype activations (triangle: max(0, mean_dist -
    dist)), sum-pooled over a grid x grid of regions. A feature = 'how much stroke-k fires in region (r,c)'."""
    C, patch, side = D["C"], D["patch"], D["side"]; k = len(C); cn = (C ** 2).sum(1)
    imgs = images.reshape(-1, side, side); pmax = side - patch + 1
    out = np.zeros((len(imgs), grid, grid, k), np.float32)
    for n, im in enumerate(imgs):
        P, pos = _patches(im, patch, stride); P = _norm(P.astype(float))
        d = (P ** 2).sum(1)[:, None] - 2 * P @ C.T + cn[None]         # squared dist to each prototype
        d = np.sqrt(np.maximum(d, 0)); act = np.maximum(0.0, d.mean(1, keepdims=True) - d)   # triangle activation
        for (i, j), a in zip(pos, act):
            r = min(grid - 1, i * grid // pmax); c = min(grid - 1, j * grid // pmax)
            out[n, r, c] += a
    return out.reshape(len(imgs), -1)


def feature_names(D, grid=2):
    return [f"stroke{p}@r{r}c{c}" for r in range(grid) for c in range(grid) for p in range(len(D["C"]))]


def encode_df(images, D, labels=None, stride=2, grid=2, target="label"):
    """Encode -> a DataFrame (named stroke-activation features [+ label]) ready for solve_any / solve_regression."""
    F = encode(images, D, stride, grid); df = pd.DataFrame(F, columns=feature_names(D, grid))
    if labels is not None:
        df[target] = labels
    return df


def selftest():
    rng = np.random.default_rng(0)
    imgs = np.zeros((40, 12, 12), np.float32)                          # two trivial classes: horiz vs vert bar
    y = np.zeros(40, int)
    for n in range(40):
        if n % 2:
            imgs[n, rng.integers(2, 10), :] = 1.0; y[n] = 1            # horizontal bar
        else:
            imgs[n, :, rng.integers(2, 10)] = 1.0                      # vertical bar
    D = learn_dict(imgs.reshape(40, -1), side=12, n_proto=8, patch=4, n_patches=2000, seed=0)
    F = encode(imgs.reshape(40, -1), D, stride=2, grid=2)
    assert F.shape == (40, 2 * 2 * 8), F.shape
    assert np.isfinite(F).all() and F.std() > 0                       # features are non-trivial
    # the learned features must separate the two classes (a 1-NN on features gets it right)
    from numpy.linalg import norm
    correct = 0
    for n in range(40):
        d = norm(F - F[n], axis=1); d[n] = 1e9
        correct += int(y[d.argmin()] == y[n])
    assert correct / 40 > 0.9, correct / 40
    print(f"imgrepr selftest OK (1-NN on learned features {correct}/40)")
    return 0


def mnist_demo(n_train=5000, n_test=2000, n_proto=96, patch=6, stride=2, grid=2, n_terms=200, seed=0):
    """Reproduce the MNIST-wall result: in-house reasoning on RAW PIXELS vs on the LEARNED REPRESENTATION
    (+ a GBM reference column). Needs kaggle_data/mnist/{mnist_train,mnist_test}.csv."""
    from thinking.azr_win import boost_multi_predict, reason_boost_multi
    tr = pd.read_csv("kaggle_data/mnist/mnist_train.csv", nrows=n_train)
    te = pd.read_csv("kaggle_data/mnist/mnist_test.csv", nrows=n_test)
    ytr, yte = tr["label"].values, te["label"].values; classes = sorted(set(ytr.tolist()))
    Xtr = tr.drop(columns=["label"]).values.astype(np.float32) / 255.0
    Xte = te.drop(columns=["label"]).values.astype(np.float32) / 255.0
    def boost(A, B):
        m = reason_boost_multi(A, ytr, classes=classes, n_terms=n_terms, seed=seed)
        return float((boost_multi_predict(m, B)[0] == yte).mean())
    print(f"  MNIST {n_train} train / {n_test} test, {len(classes)} classes")
    print(f"  raw pixels (784):           reasoning(boost) {boost(Xtr, Xte):.3f}")
    D = learn_dict(Xtr, side=28, n_proto=n_proto, patch=patch, n_patches=40000, seed=seed)
    Ftr = encode(Xtr, D, stride, grid); Fte = encode(Xte, D, stride, grid)
    print(f"  learned strokes ({Ftr.shape[1]}):       reasoning(boost) {boost(Ftr, Fte):.3f}  "
          f"(unsupervised in-house representation)")
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--mnist" in argv:
        return mnist_demo()
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
