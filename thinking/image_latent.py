"""Latent fact-conditioned rectified flow for the synthetic image rung.

This is Image-3: move generation from pixels into a compact visual latent.

The design follows the current scalable image-generation pattern at toy scale:

  image -> semantic autoencoder latent -> rectified flow in latent space -> decoder -> image

The autoencoder is not reconstruction-only.  It also predicts canonical color/shape facts from
the latent, which keeps the compression aligned with the FER/UFR requirement: factors should stay
linearly usable after compression instead of becoming another entangled bottleneck.

  python -m thinking.image_latent --selftest
  python -m thinking.image_latent --train --ae-steps 400 --flow-steps 400 \
      --out runs/image_latent_flow.pt
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_flow import FACT_VOCAB, fact_condition
from .vision import COLORS, SHAPES, DEV, ObjectSpec, object_facts, render_object, sample_object

COLOR_NAMES = tuple(COLORS)


def _batch(n, rng, size=32, device=DEV):
    imgs, conds, yc, ys = [], [], [], []
    for _ in range(n):
        spec = sample_object(rng)
        imgs.append(render_object(spec, size=size) * 2.0 - 1.0)
        conds.append(fact_condition(object_facts(spec), device="cpu").numpy())
        yc.append(COLOR_NAMES.index(spec.color))
        ys.append(SHAPES.index(spec.shape))
    x = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
    cond = torch.tensor(np.stack(conds), dtype=torch.float32, device=device)
    yc = torch.tensor(yc, dtype=torch.long, device=device)
    ys = torch.tensor(ys, dtype=torch.long, device=device)
    return x, cond, yc, ys


class SemanticAutoencoder(nn.Module):
    """Small convolutional AE with fact heads on the compressed latent."""

    def __init__(self, latent_ch=16, hidden=64):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, latent_ch, 3, padding=1),
            nn.GroupNorm(1, latent_ch),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden // 2, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.fact_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.color = nn.Sequential(
            nn.Linear(latent_ch * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(COLORS)),
        )
        self.shape = nn.Sequential(
            nn.Linear(latent_ch * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(SHAPES)),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def fact_logits(self, z):
        pooled = self.fact_pool(z).flatten(1)
        return {"color": self.color(pooled), "shape": self.shape(pooled)}

    def forward(self, x):
        z = self.encode(x)
        out = self.fact_logits(z)
        out["latent"] = z
        out["recon"] = self.decode(z)
        return out


class LatentFlowNet(nn.Module):
    """Velocity field v_theta(z_t, t, canonical_facts)."""

    def __init__(self, latent_ch=16, hidden=64, cond_dim=None):
        super().__init__()
        cond_dim = cond_dim or len(FACT_VOCAB)
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch + 1 + hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, latent_ch, 3, padding=1),
        )

    def forward(self, z, t, cond):
        if t.ndim == 1:
            t = t[:, None, None, None]
        if t.ndim == 2:
            t = t[:, :, None, None]
        b, _, h, w = z.shape
        tchan = t.expand(b, 1, h, w)
        c = self.cond(cond).view(b, -1, 1, 1).expand(b, -1, h, w)
        return self.net(torch.cat([z, tchan, c], dim=1))


def autoencoder_loss(out, x, yc, ys, fact_w=0.25):
    recon = F.mse_loss(out["recon"], x)
    facts = F.cross_entropy(out["color"], yc) + F.cross_entropy(out["shape"], ys)
    return recon + fact_w * facts, {"recon_mse": recon.detach(), "fact_ce": facts.detach()}


def latent_flow_loss(flow, z1, cond):
    x0 = torch.randn_like(z1)
    t = torch.rand(z1.shape[0], 1, 1, 1, device=z1.device)
    zt = (1.0 - t) * x0 + t * z1
    target = z1 - x0
    pred = flow(zt, t, cond)
    return F.mse_loss(pred, target)


@torch.no_grad()
def sample_latents(flow, cond, latent_shape=(16, 8, 8), steps=16, device=DEV, seed=0):
    torch.manual_seed(seed)
    z = torch.randn((cond.shape[0],) + tuple(latent_shape), device=device)
    flow.eval()
    dt = 1.0 / max(1, steps)
    for i in range(steps):
        t = torch.full((cond.shape[0], 1, 1, 1), i / max(1, steps), device=device)
        z = z + dt * flow(z, t, cond)
    return z


@torch.no_grad()
def sample_images(ae, flow, cond, latent_shape=(16, 8, 8), steps=16, device=DEV, seed=0):
    z = sample_latents(flow, cond, latent_shape=latent_shape, steps=steps, device=device,
                       seed=seed)
    ae.eval()
    return ae.decode(z).clamp(-1.0, 1.0)


@torch.no_grad()
def evaluate(ae, flow, n=128, batch=64, seed=10, size=32, device=DEV):
    rng = np.random.default_rng(seed)
    ae.eval()
    flow.eval()
    recon_losses, flow_losses = [], []
    got_c = got_s = total = 0
    latent_means, latent_stds = [], []
    while total < n:
        b = min(batch, n - total)
        x, cond, yc, ys = _batch(b, rng, size=size, device=device)
        out = ae(x)
        z = out["latent"]
        recon_losses.append(float(F.mse_loss(out["recon"], x).detach().cpu()))
        flow_losses.append(float(latent_flow_loss(flow, z, cond).detach().cpu()))
        got_c += int(out["color"].argmax(-1).eq(yc).sum())
        got_s += int(out["shape"].argmax(-1).eq(ys).sum())
        latent_means.append(float(z.mean().detach().cpu()))
        latent_stds.append(float(z.std().detach().cpu()))
        total += b

    spec = ObjectSpec("p0", "red", "circle")
    cond = fact_condition(object_facts(spec), device=device)[None]
    sample = sample_images(ae, flow, cond, latent_shape=(ae.latent_ch, size // 4, size // 4),
                           steps=4, device=device, seed=seed)
    target = torch.tensor(render_object(spec, size=size) * 2.0 - 1.0,
                          dtype=torch.float32, device=device)[None]
    return {
        "n": int(total),
        "recon_mse": float(np.mean(recon_losses)),
        "latent_velocity_mse": float(np.mean(flow_losses)),
        "latent_color_acc": got_c / total,
        "latent_shape_acc": got_s / total,
        "latent_mean": float(np.mean(latent_means)),
        "latent_std": float(np.mean(latent_stds)),
        "sample_min": float(sample.min().detach().cpu()),
        "sample_max": float(sample.max().detach().cpu()),
        "sample_center_target_mse": float(F.mse_loss(sample, target).detach().cpu()),
    }


def train_latent_flow(ae_steps=200, flow_steps=200, batch=64, latent_ch=16, hidden=64,
                      lr=2e-4, fact_w=1.0, seed=0, size=32, device=DEV):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ae = SemanticAutoencoder(latent_ch=latent_ch, hidden=hidden).to(device)
    flow = LatentFlowNet(latent_ch=latent_ch, hidden=hidden).to(device)
    opt_ae = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=0.01)
    ae.train()
    last_ae = {}
    for _ in range(ae_steps):
        x, _cond, yc, ys = _batch(batch, rng, size=size, device=device)
        out = ae(x)
        loss, parts = autoencoder_loss(out, x, yc, ys, fact_w=fact_w)
        opt_ae.zero_grad()
        loss.backward()
        opt_ae.step()
        last_ae = {k: float(v.detach().cpu()) for k, v in parts.items()}

    opt_flow = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=0.01)
    ae.eval()
    flow.train()
    last_flow = None
    for _ in range(flow_steps):
        x, cond, _yc, _ys = _batch(batch, rng, size=size, device=device)
        with torch.no_grad():
            z1 = ae.encode(x)
        loss = latent_flow_loss(flow, z1, cond)
        opt_flow.zero_grad()
        loss.backward()
        opt_flow.step()
        last_flow = float(loss.detach().cpu())

    report = evaluate(ae, flow, seed=seed + 1, size=size, device=device)
    report.update({
        "experiment": "image3_latent_fact_conditioned_rectified_flow",
        "ae_steps": int(ae_steps),
        "flow_steps": int(flow_steps),
        "batch": int(batch),
        "latent_ch": int(latent_ch),
        "hidden": int(hidden),
        "fact_w": float(fact_w),
        "last_ae": last_ae,
        "last_flow_loss": last_flow,
        "fact_vocab": [list(f) for f in FACT_VOCAB],
    })
    return ae, flow, report


def selftest():
    ae, flow, report = train_latent_flow(ae_steps=2, flow_steps=2, batch=4, latent_ch=4,
                                         hidden=16, seed=0, device="cpu")
    assert report["experiment"] == "image3_latent_fact_conditioned_rectified_flow"
    assert report["latent_color_acc"] >= 0.0 and report["latent_shape_acc"] >= 0.0
    spec = ObjectSpec("p0", "blue", "triangle")
    cond = fact_condition(object_facts(spec), device="cpu")[None]
    img = sample_images(ae, flow, cond, latent_shape=(4, 8, 8), steps=2, device="cpu", seed=0)
    assert img.shape == (1, 3, 32, 32)
    print("image_latent selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--ae-steps", type=int, default=200, dest="ae_steps")
    ap.add_argument("--flow-steps", type=int, default=200, dest="flow_steps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--latent-ch", type=int, default=16, dest="latent_ch")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fact-w", type=float, default=1.0, dest="fact_w")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/image_latent_flow.pt")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.train:
        ap.error("use --selftest or --train")
    ae, flow, report = train_latent_flow(ae_steps=args.ae_steps, flow_steps=args.flow_steps,
                                         batch=args.batch, latent_ch=args.latent_ch,
                                         hidden=args.hidden, lr=args.lr, fact_w=args.fact_w,
                                         seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "autoencoder_state_dict": ae.state_dict(),
        "flow_state_dict": flow.state_dict(),
        "report": report,
        "fact_vocab": FACT_VOCAB,
        "latent_ch": args.latent_ch,
        "hidden": args.hidden,
    }, args.out)
    print(json.dumps(report, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
