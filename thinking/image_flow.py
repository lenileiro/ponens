"""Tiny fact-conditioned rectified-flow image generator.

This is the first generation rung on top of the image grounding foundation:

  canonical visual facts -> condition vector -> rectified flow -> image

It is deliberately small and synthetic.  The point is to lock in the data path used by modern
flow-matching image models without claiming photorealistic quality yet.

  python -m thinking.image_flow --selftest
  python -m thinking.image_flow --train --steps 400 --out runs/image_flow.pt
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import COLORS, SHAPES, DEV, ObjectSpec, object_facts, render_object, sample_object

FACT_VOCAB = (
    tuple(("color", ("p0", c)) for c in COLORS)
    + tuple(("shape", ("p0", s)) for s in SHAPES)
)
FACT_INDEX = {fact: i for i, fact in enumerate(FACT_VOCAB)}


def fact_condition(facts, device=DEV):
    vec = torch.zeros(len(FACT_VOCAB), dtype=torch.float32, device=device)
    for fact in facts:
        pred, args = fact
        idx = FACT_INDEX.get((pred, tuple(args)))
        if idx is not None:
            vec[idx] = 1.0
    return vec


def _batch(n, rng, size=32, device=DEV):
    imgs, conds, specs = [], [], []
    for _ in range(n):
        spec = sample_object(rng)
        imgs.append(render_object(spec, size=size) * 2.0 - 1.0)
        conds.append(fact_condition(object_facts(spec), device="cpu").numpy())
        specs.append(spec)
    x1 = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
    cond = torch.tensor(np.stack(conds), dtype=torch.float32, device=device)
    return x1, cond, specs


class CondFlowNet(nn.Module):
    """Small convolutional velocity field v_theta(x_t, t, facts)."""

    def __init__(self, dim=64, cond_dim=None):
        super().__init__()
        cond_dim = cond_dim or len(FACT_VOCAB)
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(3 + 1 + dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 3, 3, padding=1),
        )

    def forward(self, x, t, cond):
        if t.ndim == 1:
            t = t[:, None, None, None]
        if t.ndim == 2:
            t = t[:, :, None, None]
        b, _, h, w = x.shape
        tchan = t.expand(b, 1, h, w)
        c = self.cond(cond).view(b, -1, 1, 1).expand(b, -1, h, w)
        return self.net(torch.cat([x, tchan, c], dim=1))


def flow_step_loss(model, x1, cond):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, 1, 1, device=x1.device)
    xt = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0
    pred_v = model(xt, t, cond)
    return F.mse_loss(pred_v, target_v)


@torch.no_grad()
def sample_images(model, cond, size=32, steps=16, device=DEV, seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(cond.shape[0], 3, size, size, generator=g, device=device)
    model.eval()
    dt = 1.0 / max(1, steps)
    for i in range(steps):
        t = torch.full((cond.shape[0], 1, 1, 1), i / max(1, steps), device=device)
        x = x + dt * model(x, t, cond)
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def evaluate_flow(model, n=64, batch=64, seed=10, size=32, device=DEV):
    rng = np.random.default_rng(seed)
    model.eval()
    losses, total = [], 0
    while total < n:
        b = min(batch, n - total)
        x1, cond, _ = _batch(b, rng, size=size, device=device)
        losses.append(float(flow_step_loss(model, x1, cond).detach().cpu()))
        total += b
    spec = ObjectSpec("p0", "red", "circle")
    cond = fact_condition(object_facts(spec), device=device)[None]
    sample = sample_images(model, cond, size=size, steps=4, device=device, seed=seed)
    return {
        "n": int(total),
        "velocity_mse": float(np.mean(losses)),
        "sample_min": float(sample.min().detach().cpu()),
        "sample_max": float(sample.max().detach().cpu()),
    }


def train_flow(steps=200, batch=64, dim=64, lr=2e-4, seed=0, size=32, device=DEV):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CondFlowNet(dim=dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    last_loss = None
    model.train()
    for _ in range(steps):
        x1, cond, _ = _batch(batch, rng, size=size, device=device)
        loss = flow_step_loss(model, x1, cond)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    report = evaluate_flow(model, seed=seed + 1, size=size, device=device)
    report.update({
        "experiment": "image_flow_rectified_fact_conditioned",
        "steps": int(steps),
        "batch": int(batch),
        "dim": int(dim),
        "last_loss": last_loss,
        "fact_vocab": [list(f) for f in FACT_VOCAB],
    })
    return model, report


def selftest():
    model, report = train_flow(steps=2, batch=4, dim=8, seed=0, device="cpu")
    assert report["experiment"] == "image_flow_rectified_fact_conditioned"
    spec = ObjectSpec("p0", "blue", "triangle")
    cond = fact_condition(object_facts(spec), device="cpu")[None]
    img = sample_images(model, cond, size=32, steps=2, device="cpu", seed=0)
    assert img.shape == (1, 3, 32, 32)
    print("image_flow selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/image_flow.pt")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.train:
        ap.error("use --selftest or --train")
    model, report = train_flow(steps=args.steps, batch=args.batch, dim=args.dim, lr=args.lr,
                               seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "report": report,
                "fact_vocab": FACT_VOCAB, "dim": args.dim}, args.out)
    print(json.dumps(report, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
