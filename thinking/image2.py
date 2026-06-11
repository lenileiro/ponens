"""Image-2: head-aware FER experiment with an explicit factor bottleneck.

This extends Image-1 without changing its historical two-arm result.  The new arm keeps the same
canonical supervision but routes color and shape through separate representation spaces.  The
diagnostic also probes those spaces directly, which fixes the Image-1 blind spot where the shared
embedding probe missed factorization that lived immediately before the heads.

  python -m thinking.image2 --steps 400 --seeds 0,1,2 --out runs/image2_bottleneck.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .image1 import HOLDOUT_COMBOS, JointEncoder, _batch
from .vision import (DEV, SHAPES, FactorBottleneckEncoder, ObjectEncoder, factor_probe,
                     factor_space_probe, independence_loss)


def _make(arm, dim):
    if arm == "factored":
        return ObjectEncoder(dim=dim)
    if arm == "bottleneck":
        return FactorBottleneckEncoder(dim=dim)
    if arm == "joint":
        return JointEncoder(dim=dim)
    raise ValueError(f"unknown arm {arm!r}")


def _train(arm, steps, seed, dim=64, lr=1e-3, batch=64, size=32, device=DEV,
           independence_w=0.05):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _make(arm, dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for _ in range(steps):
        x, yc, ys = _batch(batch, rng, holdout=False, size=size, device=device)
        out = model(x, return_embedding=arm == "bottleneck" and independence_w > 0.0)
        if arm == "joint":
            loss = F.cross_entropy(out["combo"], yc * len(SHAPES) + ys)
        else:
            loss = F.cross_entropy(out["color"], yc) + F.cross_entropy(out["shape"], ys)
            if arm == "bottleneck" and independence_w > 0.0:
                loss = loss + independence_w * independence_loss(out)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def _accuracy(model, arm, rng, holdout, n=240, size=32, device=DEV):
    model.eval()
    got = total = 0
    with torch.no_grad():
        while total < n:
            b = min(64, n - total)
            x, yc, ys = _batch(b, rng, holdout=holdout, size=size, device=device)
            out = model(x)
            if arm == "joint":
                pred = out["combo"].argmax(-1)
                ok = (pred // len(SHAPES)).eq(yc) & (pred % len(SHAPES)).eq(ys)
            else:
                ok = out["color"].argmax(-1).eq(yc) & out["shape"].argmax(-1).eq(ys)
            got += int(ok.sum())
            total += b
    return got / total


def _space_leakage(probe):
    return float(probe.get("max_leakage") or 0.0)


def run(steps=400, seeds=(0, 1, 2), dim=64, size=32, device=DEV,
        arms=("factored", "bottleneck", "joint"), independence_w=0.05):
    report = {
        "experiment": "image2_head_aware_fer",
        "steps": int(steps),
        "dim": int(dim),
        "holdout_combos": [list(c) for c in HOLDOUT_COMBOS],
        "arms": {},
    }
    for arm in arms:
        rows = []
        for seed in seeds:
            model = _train(arm, steps, seed, dim=dim, size=size, device=device,
                           independence_w=independence_w)
            rng = np.random.default_rng(8_000 + seed)
            embedding = factor_probe(model, size=size, device=device)
            spaces = factor_space_probe(model, size=size, device=device)
            rows.append({
                "seed": int(seed),
                "seen_acc": _accuracy(model, arm, rng, holdout=False, size=size, device=device),
                "holdout_acc": _accuracy(model, arm, rng, holdout=True, size=size, device=device),
                "embedding_ufr_score": embedding["ufr_score"],
                "embedding_leakage": max(embedding["color_shape_leakage"] or 0.0,
                                         embedding["shape_color_leakage"] or 0.0),
                "factor_space_ufr_score": spaces["ufr_score"],
                "factor_space_leakage": _space_leakage(spaces),
                "factor_space_flags": spaces["risk_flags"],
            })
        mean = lambda k: float(np.mean([r[k] for r in rows]))
        report["arms"][arm] = {
            "seeds": rows,
            "seen_acc": mean("seen_acc"),
            "holdout_acc": mean("holdout_acc"),
            "embedding_ufr_score": mean("embedding_ufr_score"),
            "embedding_leakage": mean("embedding_leakage"),
            "factor_space_ufr_score": mean("factor_space_ufr_score"),
            "factor_space_leakage": mean("factor_space_leakage"),
        }
        print(f"{arm:10} seen {report['arms'][arm]['seen_acc']:.2f}  "
              f"HOLDOUT {report['arms'][arm]['holdout_acc']:.2f}  "
              f"space-ufr {report['arms'][arm]['factor_space_ufr_score']:.2f}  "
              f"space-leak {report['arms'][arm]['factor_space_leakage']:.2f}", flush=True)
    best_holdout = max(report["arms"], key=lambda a: report["arms"][a]["holdout_acc"])
    best_probe = max(report["arms"], key=lambda a: report["arms"][a]["factor_space_ufr_score"])
    report["verdict"] = {
        "best_holdout_arm": best_holdout,
        "best_factor_space_arm": best_probe,
        "factor_space_probe_tracks_best": best_holdout == best_probe,
        "bottleneck_holdout_gain_vs_factored": (
            report["arms"].get("bottleneck", {}).get("holdout_acc", 0.0)
            - report["arms"].get("factored", {}).get("holdout_acc", 0.0)
        ),
        "bottleneck_leakage_delta_vs_factored": (
            report["arms"].get("factored", {}).get("factor_space_leakage", 0.0)
            - report["arms"].get("bottleneck", {}).get("factor_space_leakage", 0.0)
        ),
    }
    print("verdict:", json.dumps(report["verdict"], indent=1), flush=True)
    return report


def selftest():
    report = run(steps=2, seeds=(0,), dim=16, size=32, device="cpu",
                 arms=("factored", "bottleneck", "joint"))
    assert set(report["arms"]) == {"factored", "bottleneck", "joint"}
    assert "factor_space_probe_tracks_best" in report["verdict"]
    print("image2 selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--arms", default="factored,bottleneck,joint")
    ap.add_argument("--independence-w", type=float, default=0.05, dest="independence_w")
    ap.add_argument("--out", default="runs/image2_bottleneck.json")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    report = run(steps=args.steps, seeds=tuple(int(s) for s in args.seeds.split(",")),
                 dim=args.dim, arms=arms, independence_w=args.independence_w)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
