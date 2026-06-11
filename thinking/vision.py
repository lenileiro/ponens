"""Synthetic image grounding + FER/UFR probes.

This is Image-0/Image-1 for the multimodal path:

  image crop -> canonical visual facts

The goal is not photorealism.  It is a controlled visual world where ground-truth factors are
known, so we can test the FER hypothesis directly: does the representation reuse color and shape
across contexts, or does it learn brittle color-shape mixtures that only solve the labels?
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device import get_device

DEV = get_device()

COLORS = {
    "red": (0.90, 0.10, 0.10),
    "green": (0.10, 0.72, 0.18),
    "blue": (0.12, 0.30, 0.92),
    "yellow": (0.92, 0.78, 0.12),
    "white": (0.92, 0.92, 0.88),
}
SHAPES = ("square", "circle", "triangle")
SLOTS = ("p0", "p1", "p2")


@dataclass(frozen=True)
class ObjectSpec:
    slot: str
    color: str
    shape: str
    x: float = 0.5
    y: float = 0.5
    scale: float = 0.55


@dataclass(frozen=True)
class Scene:
    objects: tuple
    image: np.ndarray
    facts: tuple


def _grid(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return (xx + 0.5) / size, (yy + 0.5) / size


def render_object(spec: ObjectSpec, size=32):
    """Render one object crop as float32 CHW in [0,1]."""
    if spec.color not in COLORS:
        raise ValueError(f"unknown color {spec.color!r}")
    if spec.shape not in SHAPES:
        raise ValueError(f"unknown shape {spec.shape!r}")
    x, y = _grid(size)
    dx, dy = x - spec.x, y - spec.y
    r = 0.34 * spec.scale
    if spec.shape == "square":
        mask = (np.abs(dx) <= r) & (np.abs(dy) <= r)
    elif spec.shape == "circle":
        mask = dx * dx + dy * dy <= r * r
    else:
        # Upright isosceles triangle.  The vertical taper keeps shape independent of color.
        top = spec.y - r
        bottom = spec.y + r
        rel = np.clip((y - top) / max(1e-6, bottom - top), 0.0, 1.0)
        half_width = r * rel
        mask = (y >= top) & (y <= bottom) & (np.abs(dx) <= half_width)

    img = np.zeros((3, size, size), dtype=np.float32)
    img[:, mask] = np.asarray(COLORS[spec.color], dtype=np.float32)[:, None]
    return img


def object_facts(spec: ObjectSpec):
    """Canonical binary facts.  These match the package's h/pred/t trace convention."""
    return (
        ("color", (spec.slot, spec.color)),
        ("shape", (spec.slot, spec.shape)),
    )


def fact_tokens(facts):
    toks = []
    for pred, (h, t) in facts:
        toks += ["fact", h, pred, t, "."]
    return toks + ["done", "."]


def sample_object(rng, slot="p0", jitter=0.10):
    color = tuple(COLORS)[int(rng.integers(len(COLORS)))]
    shape = SHAPES[int(rng.integers(len(SHAPES)))]
    x = float(np.clip(0.5 + rng.uniform(-jitter, jitter), 0.30, 0.70))
    y = float(np.clip(0.5 + rng.uniform(-jitter, jitter), 0.30, 0.70))
    scale = float(rng.uniform(0.48, 0.68))
    return ObjectSpec(slot=slot, color=color, shape=shape, x=x, y=y, scale=scale)


def render_scene(objects, size=64):
    img = np.zeros((3, size, size), dtype=np.float32)
    facts = []
    for obj in objects:
        img = np.maximum(img, render_object(obj, size=size))
        facts.extend(object_facts(obj))
    for a in objects:
        for b in objects:
            if a.slot == b.slot:
                continue
            if a.x + 0.05 < b.x:
                facts.append(("left_of", (a.slot, b.slot)))
            if a.y + 0.05 < b.y:
                facts.append(("above", (a.slot, b.slot)))
    return Scene(objects=tuple(objects), image=img, facts=tuple(facts))


def sample_scene(rng, size=64):
    xs = (0.25, 0.50, 0.75)
    objects = []
    for slot, x in zip(SLOTS, xs):
        obj = sample_object(rng, slot=slot, jitter=0.04)
        objects.append(ObjectSpec(slot, obj.color, obj.shape, x=x,
                                  y=float(rng.uniform(0.35, 0.65)), scale=obj.scale))
    return render_scene(objects, size=size)


class ObjectEncoder(nn.Module):
    """Small image encoder with explicit color and shape heads."""
    def __init__(self, dim=64, n_colors=None, n_shapes=None):
        super().__init__()
        n_colors = n_colors or len(COLORS)
        n_shapes = n_shapes or len(SHAPES)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64, dim), nn.LayerNorm(dim))
        self.color = nn.Linear(dim, n_colors)
        self.shape = nn.Linear(dim, n_shapes)

    def forward(self, x, return_embedding=False):
        z = self.proj(self.conv(x))
        out = {"color": self.color(z), "shape": self.shape(z)}
        if return_embedding:
            out["embedding"] = z
        return out


def _batch(batch, rng, size=32, device=DEV):
    imgs, colors, shapes = [], [], []
    color_names = tuple(COLORS)
    for _ in range(batch):
        spec = sample_object(rng)
        imgs.append(render_object(spec, size=size))
        colors.append(color_names.index(spec.color))
        shapes.append(SHAPES.index(spec.shape))
    x = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
    yc = torch.tensor(colors, dtype=torch.long, device=device)
    ys = torch.tensor(shapes, dtype=torch.long, device=device)
    return x, yc, ys


def evaluate(model, n=256, batch=64, seed=1, size=32, device=DEV):
    rng = np.random.default_rng(seed)
    model.eval()
    got_c = got_s = total = 0
    with torch.no_grad():
        while total < n:
            b = min(batch, n - total)
            x, yc, ys = _batch(b, rng, size=size, device=device)
            out = model(x)
            got_c += int(out["color"].argmax(-1).eq(yc).sum())
            got_s += int(out["shape"].argmax(-1).eq(ys).sum())
            total += b
    return {"n": int(total), "color_acc": got_c / total, "shape_acc": got_s / total}


def predict_object_facts(model, image, slot="p0", device=DEV):
    """Predict canonical facts for one object crop."""
    if isinstance(image, np.ndarray):
        x = torch.tensor(image[None], dtype=torch.float32, device=device)
    else:
        x = image[None].to(device) if image.ndim == 3 else image.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    color = tuple(COLORS)[int(out["color"].argmax(-1)[0])]
    shape = SHAPES[int(out["shape"].argmax(-1)[0])]
    return object_facts(ObjectSpec(slot=slot, color=color, shape=shape))


def _cos(a, b):
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if den == 0.0 else float(np.dot(a, b) / den)


def _mean(xs):
    return float(np.mean(xs)) if xs else None


def factor_probe(model, repeats=4, size=32, device=DEV):
    """Measure factor reuse in the learned embedding space.

    High color reuse means "red" is close across shapes.  High shape reuse means "circle" is close
    across colors.  Large leakage means the embedding is tied to the full color-shape combination.
    """
    rows = []
    model.eval()
    with torch.no_grad():
        for color in COLORS:
            for shape in SHAPES:
                for rep in range(repeats):
                    rng = np.random.default_rng(10_000 + 101 * rep + 17 * list(COLORS).index(color)
                                                + SHAPES.index(shape))
                    spec = sample_object(rng, slot="p0")
                    spec = ObjectSpec("p0", color, shape, spec.x, spec.y, spec.scale)
                    x = torch.tensor(render_object(spec, size=size)[None], dtype=torch.float32,
                                     device=device)
                    z = model(x, return_embedding=True)["embedding"][0].detach().cpu().numpy()
                    rows.append({"color": color, "shape": shape, "vector": z})
    same_color, diff_color = [], []
    same_shape, diff_shape = [], []
    same_combo, same_color_other_shape, same_shape_other_color = [], [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            c = _cos(a["vector"], b["vector"])
            color_same = a["color"] == b["color"]
            shape_same = a["shape"] == b["shape"]
            if color_same:
                same_color.append(c)
            else:
                diff_color.append(c)
            if shape_same:
                same_shape.append(c)
            else:
                diff_shape.append(c)
            if color_same and shape_same:
                same_combo.append(c)
            elif color_same:
                same_color_other_shape.append(c)
            elif shape_same:
                same_shape_other_color.append(c)
    color_margin = (_mean(same_color) - _mean(diff_color)
                    if same_color and diff_color else None)
    shape_margin = (_mean(same_shape) - _mean(diff_shape)
                    if same_shape and diff_shape else None)
    color_leakage = (_mean(same_combo) - _mean(same_color_other_shape)
                     if same_combo and same_color_other_shape else None)
    shape_leakage = (_mean(same_combo) - _mean(same_shape_other_color)
                     if same_combo and same_shape_other_color else None)
    margins = [m for m in (color_margin, shape_margin) if m is not None]
    leak = max([x for x in (color_leakage, shape_leakage) if x is not None] or [0.0])
    reuse = float(np.mean([max(0.0, min(1.0, m / 0.25)) for m in margins])) if margins else None
    penalty = max(0.0, min(1.0, leak / 0.25))
    # Factored representations need both reuse and independence.  A high-accuracy classifier can
    # still be FER if same-color/same-shape clusters collapse into full color-shape combinations.
    ufr_score = None if reuse is None else max(0.0, min(1.0, reuse * (1.0 - penalty)))
    flags = []
    if color_margin is not None and color_margin < 0.05:
        flags.append("weak_color_reuse")
    if shape_margin is not None and shape_margin < 0.05:
        flags.append("weak_shape_reuse")
    if color_leakage is not None and color_leakage > 0.20:
        flags.append("color_shape_entanglement")
    if shape_leakage is not None and shape_leakage > 0.20:
        flags.append("shape_color_entanglement")
    if ufr_score is None:
        verdict = "unknown"
    elif ufr_score < 0.35:
        verdict = "high_fer_risk"
    elif ufr_score < 0.55:
        verdict = "medium_fer_risk"
    else:
        verdict = "low_fer_risk"
    return {
        "probe": "visual_factor_reuse",
        "n_vectors": len(rows),
        "same_color_cos": _mean(same_color),
        "different_color_cos": _mean(diff_color),
        "color_reuse_margin": color_margin,
        "same_shape_cos": _mean(same_shape),
        "different_shape_cos": _mean(diff_shape),
        "shape_reuse_margin": shape_margin,
        "color_shape_leakage": color_leakage,
        "shape_color_leakage": shape_leakage,
        "ufr_score": ufr_score,
        "verdict": verdict,
        "risk_flags": flags,
    }


def train_object_encoder(steps=200, batch=64, dim=64, lr=1e-3, seed=0, size=32, device=DEV):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = ObjectEncoder(dim=dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for st in range(steps):
        x, yc, ys = _batch(batch, rng, size=size, device=device)
        out = model(x)
        loss = F.cross_entropy(out["color"], yc) + F.cross_entropy(out["shape"], ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    report = evaluate(model, seed=seed + 1, size=size, device=device)
    report.update(factor_probe(model, size=size, device=device))
    report["steps"] = int(steps)
    report["batch"] = int(batch)
    report["dim"] = int(dim)
    return model, report


def selftest():
    rng = np.random.default_rng(0)
    obj = sample_object(rng, slot="p0")
    img = render_object(obj)
    assert img.shape == (3, 32, 32) and img.dtype == np.float32 and img.max() > 0
    facts = object_facts(obj)
    toks = fact_tokens(facts)
    assert toks[:5] == ["fact", "p0", "color", obj.color, "."]
    scene = sample_scene(rng)
    assert scene.image.shape == (3, 64, 64)
    assert ("left_of", ("p0", "p1")) in scene.facts
    m, report = train_object_encoder(steps=2, batch=4, dim=16, seed=0, device="cpu")
    assert isinstance(m, ObjectEncoder)
    assert report["n"] == 256 and "ufr_score" in report
    pfacts = predict_object_facts(m, img, slot="p0", device="cpu")
    assert len(pfacts) == 2 and pfacts[0][0] == "color" and pfacts[1][0] == "shape"
    print("vision selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default="runs/vision_object_encoder.pt")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.train:
        ap.error("use --selftest or --train")
    model, report = train_object_encoder(steps=args.steps, batch=args.batch, dim=args.dim)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "report": report, "colors": list(COLORS),
                "shapes": list(SHAPES), "dim": args.dim}, args.out)
    print(json.dumps(report, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
