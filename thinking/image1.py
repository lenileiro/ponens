"""Image-1: the two-arm FER experiment — does FACTORED canonical supervision beat JOINT
combo supervision on compositional generalization, and does the factor probe predict it?

Arms (identical trunk, dim, steps, lr, seeds; only the supervision head differs):
  A `factored` -- separate color and shape heads (the Ponens way: canonical facts state each
                  factor as its own line)
  B `joint`    -- one softmax over color x shape combinations (standard classification)

Both train ONLY on combos outside HOLDOUT_COMBOS and are tested on:
  - seen combos        (in-distribution: both factors right)
  - HELD-OUT combos    (compositional generalization: e.g. 'red triangle' never seen,
                        'red' and 'triangle' separately common)
  - the visual factor probe (reuse margins, combination leakage, ufr_score)

The FER-hypothesis prediction: factored > joint on held-out combos, with lower leakage --
and probe and behavior should CORRELATE. If they don't, that is evidence against the bet.

  python -m thinking.image1 --steps 400 --seeds 0,1,2 --out runs/image1_fer.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import (COLORS, SHAPES, ObjectEncoder, factor_probe, render_object,
                     sample_object, DEV)

COLOR_NAMES = tuple(COLORS)
# three held-out pairings, distinct rows and columns of the 5x3 grid
HOLDOUT_COMBOS = (("red", "triangle"), ("blue", "circle"), ("yellow", "square"))


def _sample_spec(rng, holdout):
    """An object spec from the training (or holdout) region of the combo grid."""
    for _ in range(64):
        spec = sample_object(rng)
        if ((spec.color, spec.shape) in HOLDOUT_COMBOS) == holdout:
            return spec
    raise RuntimeError("combo sampling starved")


def _batch(n, rng, holdout=False, size=32, device=DEV):
    imgs, yc, ys = [], [], []
    for _ in range(n):
        spec = _sample_spec(rng, holdout)
        imgs.append(render_object(spec, size=size))
        yc.append(COLOR_NAMES.index(spec.color))
        ys.append(SHAPES.index(spec.shape))
    x = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
    return (x, torch.tensor(yc, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device))


class JointEncoder(ObjectEncoder):
    """Same trunk; ONE softmax over color x shape combinations."""

    def __init__(self, dim=64):
        super().__init__(dim=dim)
        self.combo = nn.Linear(dim, len(COLORS) * len(SHAPES))

    def forward(self, x, return_embedding=False):
        z = self.proj(self.conv(x))
        logits = self.combo(z)
        out = {"combo": logits}
        if return_embedding:
            out["embedding"] = z
        return out


def _train(arm, steps, seed, dim=64, lr=1e-3, batch=64, size=32, device=DEV):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = (ObjectEncoder(dim=dim) if arm == "factored" else JointEncoder(dim=dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        x, yc, ys = _batch(batch, rng, holdout=False, size=size, device=device)
        out = model(x)
        if arm == "factored":
            loss = F.cross_entropy(out["color"], yc) + F.cross_entropy(out["shape"], ys)
        else:
            loss = F.cross_entropy(out["combo"], yc * len(SHAPES) + ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def _accuracy(model, arm, rng, holdout, n=240, size=32, device=DEV):
    """Both factors right on one object. Joint arm: argmax over ALL combos (an unseen combo is
    an untrained-but-present class -- predicting it from parts IS compositional generalization)."""
    model.eval()
    got = total = 0
    with torch.no_grad():
        while total < n:
            b = min(64, n - total)
            x, yc, ys = _batch(b, rng, holdout=holdout, size=size, device=device)
            out = model(x)
            if arm == "factored":
                ok = out["color"].argmax(-1).eq(yc) & out["shape"].argmax(-1).eq(ys)
            else:
                pred = out["combo"].argmax(-1)
                ok = (pred // len(SHAPES)).eq(yc) & (pred % len(SHAPES)).eq(ys)
            got += int(ok.sum())
            total += b
    return got / total


def run(steps=400, seeds=(0, 1, 2), dim=64, size=32, device=DEV):
    report = {"experiment": "image1_fer_two_arm", "steps": steps, "dim": dim,
              "holdout_combos": [list(c) for c in HOLDOUT_COMBOS], "arms": {}}
    for arm in ("factored", "joint"):
        rows = []
        for seed in seeds:
            model = _train(arm, steps, seed, dim=dim, size=size, device=device)
            rng = np.random.default_rng(7_000 + seed)
            probe = factor_probe(model, size=size, device=device)
            rows.append({
                "seed": seed,
                "seen_acc": _accuracy(model, arm, rng, holdout=False, size=size, device=device),
                "holdout_acc": _accuracy(model, arm, rng, holdout=True, size=size, device=device),
                "ufr_score": probe["ufr_score"],
                "color_reuse_margin": probe["color_reuse_margin"],
                "shape_reuse_margin": probe["shape_reuse_margin"],
                "leakage": max(probe["color_shape_leakage"] or 0.0,
                               probe["shape_color_leakage"] or 0.0),
                "risk_flags": probe["risk_flags"],
            })
        mean = lambda k: float(np.mean([r[k] for r in rows]))
        report["arms"][arm] = {
            "seeds": rows,
            "seen_acc": mean("seen_acc"),
            "holdout_acc": mean("holdout_acc"),
            "ufr_score": mean("ufr_score"),
            "leakage": mean("leakage"),
        }
        print(f"{arm:9} seen {report['arms'][arm]['seen_acc']:.2f}  "
              f"HOLDOUT {report['arms'][arm]['holdout_acc']:.2f}  "
              f"ufr {report['arms'][arm]['ufr_score']:.2f}  "
              f"leakage {report['arms'][arm]['leakage']:.2f}", flush=True)
    a, b = report["arms"]["factored"], report["arms"]["joint"]
    report["verdict"] = {
        "factored_beats_joint_on_holdout": a["holdout_acc"] > b["holdout_acc"] + 0.05,
        "probe_predicts_behavior": ((a["leakage"] < b["leakage"])
                                    == (a["holdout_acc"] > b["holdout_acc"])),
        "holdout_gap": a["holdout_acc"] - b["holdout_acc"],
        "leakage_gap": b["leakage"] - a["leakage"],
    }
    print("verdict:", json.dumps(report["verdict"], indent=1), flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default="runs/image1_fer.json")
    args = ap.parse_args(argv)
    report = run(steps=args.steps, seeds=tuple(int(s) for s in args.seeds.split(",")),
                 dim=args.dim)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
