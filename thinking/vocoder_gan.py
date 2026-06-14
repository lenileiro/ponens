"""Natural-voice vocoder: the Vocos iSTFT generator trained with a GAN (HiFi-GAN recipe).

Lesson (researched 2026-06): multi-resolution STFT loss ALONE is only an AUXILIARY loss -- every
natural-sounding vocoder (HiFi-GAN, MelGAN, BigVGAN, Vocos) trains ADVERSARIALLY. Spectral-distance
losses over-smooth and leave a buzzy/robotic residue; the DISCRIMINATOR is what makes audio sound
natural, by forcing the waveform onto the manifold of real speech (phase coherence + fine
periodicity that L1-on-magnitude can't capture). My first vocoder used STFT loss with NO
discriminator -- hence robotic despite a 47% better spectral metric.

This adds the HiFi-GAN training: Multi-Period Discriminator (periods 2,3,5,7,11) + Multi-Scale
Discriminator, LSGAN adversarial loss + feature-matching loss + mel-reconstruction loss, training
the same Vocos generator (log-mag -> complex STFT -> iSTFT).

  python -m thinking.vocoder_gan --selftest
  python -m thinking.vocoder_gan --train --steps 30000 --out runs/vocoder_gan.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from device import get_device
from .neuralvocoder import Vocos, logmag_feat, _waves, _split, multi_res_stft_loss
from .vocoder import load_bank, write_wav, griffin_lim, N_FFT, HOP, SR

DEV = get_device()


def _mel(wav, n_mels=80, n_fft=N_FFT):
    """Mel-spectrogram L1 target (the reconstruction anchor in HiFi-GAN's loss)."""
    w = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft, HOP, n_fft, w, center=True, return_complex=True).abs()
    # simple triangular mel filterbank (cached on first call)
    key = (n_mels, n_fft, wav.device)
    fb = _mel.cache.get(key)
    if fb is None:
        f = torch.linspace(0, SR / 2, n_fft // 2 + 1, device=wav.device)
        mmax = 2595 * np.log10(1 + (SR / 2) / 700)
        centers = 700 * (10 ** (torch.linspace(0, mmax, n_mels + 2, device=wav.device) / 2595) - 1)
        fb = torch.zeros(n_mels, n_fft // 2 + 1, device=wav.device)
        for m in range(1, n_mels + 1):
            lo, ce, hi = centers[m - 1], centers[m], centers[m + 1]
            fb[m - 1] = torch.clamp(torch.minimum((f - lo) / (ce - lo + 1e-9),
                                                  (hi - f) / (hi - ce + 1e-9)), 0, 1)
        _mel.cache[key] = fb
    return torch.log1p((fb @ spec))
_mel.cache = {}


class PeriodDisc(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(128, 256, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(256, 256, (5, 1), 1, (2, 0))),
        ])
        self.post = weight_norm(nn.Conv2d(256, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        b, t = x.shape
        pad = (self.period - t % self.period) % self.period
        x = F.pad(x, (0, pad)).view(b, 1, -1, self.period)
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class ScaleDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 64, 15, 1, 7)),
            weight_norm(nn.Conv1d(64, 128, 41, 4, 20, groups=4)),
            weight_norm(nn.Conv1d(128, 256, 41, 4, 20, groups=16)),
            weight_norm(nn.Conv1d(256, 256, 5, 1, 2)),
        ])
        self.post = weight_norm(nn.Conv1d(256, 1, 3, 1, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class Discriminator(nn.Module):
    """MPD (periods) + MSD (raw + 2 pooled scales) -- the HiFi-GAN discriminator bank."""

    def __init__(self):
        super().__init__()
        self.mpd = nn.ModuleList([PeriodDisc(p) for p in (2, 3, 5, 7, 11)])
        self.msd = nn.ModuleList([ScaleDisc() for _ in range(3)])

    def forward(self, x):
        outs, feats = [], []
        for d in self.mpd:
            o, f = d(x)
            outs.append(o)
            feats.append(f)
        cur = x
        for i, d in enumerate(self.msd):
            if i > 0:
                cur = F.avg_pool1d(cur.unsqueeze(1), 4, 2, 1).squeeze(1)
            o, f = d(cur)
            outs.append(o)
            feats.append(f)
        return outs, feats


def _feat_match(fr, ff):
    loss = 0.0
    for dr, df in zip(fr, ff):
        for a, b in zip(dr, df):
            loss = loss + F.l1_loss(a, b)
    return loss


def train(steps=30000, seed=0, device=DEV, batch=16, lr=2e-4, d=256):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _m, clips = load_bank()
    train_clips, _ = _split(clips)
    G = Vocos(d=d).to(device)
    D = Discriminator().to(device)
    optG = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.8, 0.99))
    optD = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.8, 0.99))
    for st in range(1, steps + 1):
        bc = [train_clips[int(rng.integers(len(train_clips)))] for _ in range(batch)]
        real = _waves(bc, device)
        feat = logmag_feat(real)
        fake = G(feat, length=real.shape[1])
        # --- D step (LSGAN) ---
        optD.zero_grad()
        dr, _ = D(real)
        df, _ = D(fake.detach())
        d_loss = sum(((r - 1) ** 2).mean() + (f ** 2).mean() for r, f in zip(dr, df))
        d_loss.backward()
        optD.step()
        # --- G step: adversarial + feature-match + mel + multi-res STFT ---
        optG.zero_grad()
        dr, fr = D(real)
        df, ff = D(fake)
        adv = sum(((f - 1) ** 2).mean() for f in df)
        fm = _feat_match(fr, ff)
        mel = F.l1_loss(_mel(fake), _mel(real))
        stft = multi_res_stft_loss(fake, real)
        g_loss = adv + 2.0 * fm + 45.0 * mel + 5.0 * stft
        g_loss.backward()
        optG.step()
        if st % max(1, steps // 10) == 0 or st == steps:
            print(f"  gan {st}/{steps} G {g_loss.item():.3f} (adv {adv.item():.3f} "
                  f"fm {fm.item():.2f} mel {mel.item():.3f}) D {d_loss.item():.3f}", flush=True)
    return G


def evaluate(G, device=DEV, n=21, save_dir=None):
    from .neuralvocoder import _spec_distance, neural_vocode
    _m, clips = load_bank()
    _, test = _split(clips)
    G.eval()
    gan_d, gl_d = [], []
    with torch.no_grad():
        for i, c in enumerate(test[:n]):
            tgt = _waves([c], device)
            feat = logmag_feat(tgt)[0].cpu().numpy()
            t = tgt[0].cpu().numpy()
            g = neural_vocode(G, feat, device=device)
            gl = griffin_lim(feat)
            gan_d.append(_spec_distance(g, t))
            gl_d.append(_spec_distance(gl, t))
            if save_dir and i < 4:
                os.makedirs(save_dir, exist_ok=True)
                write_wav(os.path.join(save_dir, f"gan{i}_oracle.wav"), t)
                write_wav(os.path.join(save_dir, f"gan{i}_ganvocoder.wav"), g)
                write_wav(os.path.join(save_dir, f"gan{i}_griffinlim.wav"), gl)
    return {"gan_dist": float(np.mean(gan_d)), "griffinlim_dist": float(np.mean(gl_d)), "n": len(gan_d)}


def run(steps=30000, seed=0, device=DEV, save_dir="data/synth"):
    G = train(steps=steps, seed=seed, device=device)
    ev = evaluate(G, device=device, save_dir=save_dir)
    report = {"experiment": "vocoder_gan", "steps": steps, **ev,
              "improvement_pct": 100 * (ev["griffinlim_dist"] - ev["gan_dist"])
              / max(1e-9, ev["griffinlim_dist"])}
    print(f"\nHELD-OUT distance: GAN {ev['gan_dist']:.4f}  vs Griffin-Lim {ev['griffinlim_dist']:.4f}")
    return report, G


def selftest():
    _m, clips = load_bank()
    tr, _ = _split(clips)
    G = Vocos(d=32, blocks=2)
    D = Discriminator()
    real = _waves(tr[:2], "cpu")
    fake = G(logmag_feat(real), length=real.shape[1])
    outs, feats = D(real)
    assert len(outs) == 8 and len(feats) == 8           # 5 MPD + 3 MSD
    df, ff = D(fake)
    fm = _feat_match(feats, ff)
    mel = F.l1_loss(_mel(fake), _mel(real))
    assert torch.isfinite(fm) and torch.isfinite(mel)
    G2 = train(steps=2, seed=0, device="cpu")
    print("vocoder_gan selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/vocoder_gan.json")
    ap.add_argument("--checkpoint", default="runs/vocoder_gan.pt")
    ap.add_argument("--synth-out", default="data/synth", dest="synth_out")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if args.train:
        report, G = run(steps=args.steps, seed=args.seed, save_dir=args.synth_out)
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": G.state_dict(), "d": 256}, args.checkpoint)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"saved -> {args.out}, {args.checkpoint}")
        return
    ap.error("choose --selftest / --train")


if __name__ == "__main__":
    main()
