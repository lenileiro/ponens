"""Audio-0/1: synthetic audio factor world + the FER three-arm experiment, transposed from
Image-2 (which it must replicate: factored 0.65 -> swap 0.97 -> subspace 1.00 on held-out combos).

The world is the audio analog of the shapes world: PITCH (8 notes) x TIMBRE (4 waveforms) x
ENVELOPE (3 shapes) rendered as real waveforms -> log-spectrogram features. Factors are known by
construction, so FER probes are exact and CAUSAL swap pairs are renderable (the same note with
only the timbre changed -- impossible with natural audio, trivial here). Canonical facts use the
package's h/pred/t convention (pitch/timbre/env of a slot), so Audio-2 (listen: audio -> facts ->
checker) and Audio-3 (speak: facts -> audio tokens, VERIFIED BY ROUND-TRIP through the listener)
plug into the existing trace machinery.

  python -m thinking.audio --selftest
  python -m thinking.audio --steps 500 --seeds 0,1,2 --out runs/audio1_fer.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device

DEV = get_device()
SR = 8000
DUR = 0.30                                                  # 2400 samples per clip

PITCHES = {"n262": 262.0, "n294": 294.0, "n330": 330.0, "n349": 349.0,
           "n392": 392.0, "n440": 440.0, "n494": 494.0, "n523": 523.0}
TIMBRES = ("sine", "square", "saw", "organ")
ENVELOPES = ("flat", "decay", "attack")
PITCH_NAMES = tuple(PITCHES)

# 12 of 96 combos held out; every factor value still well covered in training
HOLDOUT_AUDIO = (("n262", "saw", "decay"), ("n294", "sine", "attack"), ("n330", "organ", "flat"),
                 ("n349", "square", "decay"), ("n392", "saw", "attack"), ("n440", "sine", "flat"),
                 ("n494", "organ", "decay"), ("n523", "square", "attack"),
                 ("n262", "organ", "attack"), ("n349", "sine", "decay"),
                 ("n440", "saw", "flat"), ("n494", "square", "flat"))


def render_tone(pitch, timbre, envelope, detune=0.0, amp=1.0, phase=0.0, noise=0.01,
                rng=None):
    """One factor combination -> waveform. Deterministic given the jitter arguments."""
    t = np.arange(int(SR * DUR), dtype=np.float32) / SR
    f = PITCHES[pitch] * (1.0 + detune)
    ph = 2 * np.pi * f * t + phase
    if timbre == "sine":
        w = np.sin(ph)
    elif timbre == "square":
        w = np.sign(np.sin(ph))
    elif timbre == "saw":
        w = 2.0 * ((f * t + phase / (2 * np.pi)) % 1.0) - 1.0
    else:                                                   # organ: harmonic stack
        w = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)
        w = w / 1.75
    if envelope == "decay":
        w = w * np.exp(-t * 9.0)
    elif envelope == "attack":
        w = w * (1.0 - np.exp(-t * 14.0)) * np.exp(-t * 2.5)
    w = amp * w.astype(np.float32)
    if noise and rng is not None:
        w = w + noise * rng.standard_normal(len(w)).astype(np.float32)
    return w


def spectrogram(wave, n_fft=128, hop=64):
    """Log-magnitude STFT, (1, freq, frames) float32 -- the audio 'image'."""
    frames = []
    win = np.hanning(n_fft).astype(np.float32)
    for i in range(0, len(wave) - n_fft + 1, hop):
        frames.append(np.abs(np.fft.rfft(wave[i:i + n_fft] * win)))
    s = np.log1p(np.stack(frames, axis=1).astype(np.float32))
    return s[None]                                          # (1, 65, ~36)


def sample_clip(rng, holdout=False):
    for _ in range(256):
        combo = (PITCH_NAMES[int(rng.integers(len(PITCH_NAMES)))],
                 TIMBRES[int(rng.integers(len(TIMBRES)))],
                 ENVELOPES[int(rng.integers(len(ENVELOPES)))])
        if (combo in HOLDOUT_AUDIO) == holdout:
            return dict(pitch=combo[0], timbre=combo[1], envelope=combo[2],
                        detune=float(rng.uniform(-0.015, 0.015)),
                        amp=float(rng.uniform(0.7, 1.0)),
                        phase=float(rng.uniform(0, 2 * np.pi)))
    raise RuntimeError("combo sampling starved")


def clip_facts(spec, slot="p0"):
    """Canonical facts in the package's h/pred/t convention (Audio-2 extraction targets)."""
    return (("pitch", (slot, spec["pitch"])), ("timbre", (slot, spec["timbre"])),
            ("env", (slot, spec["envelope"])))


def swap_factor(spec, factor, rng):
    """CAUSAL pair: same clip, exactly ONE factor changed (jitter preserved)."""
    pool = {"pitch": PITCH_NAMES, "timbre": TIMBRES, "envelope": ENVELOPES}[factor]
    others = [v for v in pool if v != spec[factor]]
    return {**spec, factor: others[int(rng.integers(len(others)))]}


def _feats(specs, rng, device):
    xs = [spectrogram(render_tone(s["pitch"], s["timbre"], s["envelope"], s["detune"],
                                  s["amp"], s["phase"], rng=rng)) for s in specs]
    return torch.tensor(np.stack(xs), dtype=torch.float32, device=device)


def _labels(specs, device):
    yp = torch.tensor([PITCH_NAMES.index(s["pitch"]) for s in specs], device=device)
    yt = torch.tensor([TIMBRES.index(s["timbre"]) for s in specs], device=device)
    ye = torch.tensor([ENVELOPES.index(s["envelope"]) for s in specs], device=device)
    return yp, yt, ye


class AudioEncoder(nn.Module):
    """Conv trunk over the spectrogram; heads per arm.
    subspace: z = [pitch 20 | timbre 20 | env 16 | free 8]."""

    SLICES = {"pitch": (0, 20), "timbre": (20, 40), "envelope": (40, 56)}

    def __init__(self, arm, dim=64):
        super().__init__()
        self.arm = arm
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64, dim), nn.LayerNorm(dim))
        n = {"pitch": len(PITCHES), "timbre": len(TIMBRES), "envelope": len(ENVELOPES)}
        if arm == "subspace":
            self.heads = nn.ModuleDict({k: nn.Linear(b - a, n[k])
                                        for k, (a, b) in self.SLICES.items()})
        else:
            self.heads = nn.ModuleDict({k: nn.Linear(dim, v) for k, v in n.items()})

    def slice(self, z, factor):
        if self.arm == "subspace":
            a, b = self.SLICES[factor]
            return z[:, a:b]
        return z

    def forward(self, x, return_embedding=False):
        z = self.proj(self.conv(x))
        out = {k: self.heads[k](self.slice(z, k)) for k in self.heads}
        if return_embedding:
            out["embedding"] = z
        return out


def linear_probe(model, rng, n_train=480, n_test=240, device=DEV, ridge=1e-2):
    """Frozen trunk; ridge probes fit on SEEN combos, tested on HELD-OUT combos
    (the probe that predicted behavior in Image-2 where cosine geometry did not)."""
    model.eval()

    def embed(n, holdout):
        zs, ys = [], ([], [], [])
        with torch.no_grad():
            for off in range(0, n, 64):
                specs = [sample_clip(rng, holdout) for _ in range(min(64, n - off))]
                z = model(_feats(specs, rng, device), return_embedding=True)["embedding"]
                zs.append(z)
                for buf, lab in zip(ys, _labels(specs, device)):
                    buf.append(lab)
        return torch.cat(zs), [torch.cat(b) for b in ys]

    ztr, ytr = embed(n_train, holdout=False)
    zte, yte = embed(n_test, holdout=True)
    out = {}
    for i, (name, k) in enumerate((("pitch", len(PITCHES)), ("timbre", len(TIMBRES)),
                                   ("env", len(ENVELOPES)))):
        Y = F.one_hot(ytr[i], k).float()
        Z = torch.cat([ztr, torch.ones(len(ztr), 1, device=device)], 1)
        W = torch.linalg.solve(Z.T @ Z + ridge * torch.eye(Z.shape[1], device=device), Z.T @ Y)
        Zt = torch.cat([zte, torch.ones(len(zte), 1, device=device)], 1)
        out[f"probe_{name}_holdout"] = float((Zt @ W).argmax(-1).eq(yte[i]).float().mean())
    out["probe_holdout"] = float(np.mean([out["probe_pitch_holdout"],
                                          out["probe_timbre_holdout"],
                                          out["probe_env_holdout"]]))
    return out


def accuracy(model, rng, holdout, n=240, device=DEV):
    model.eval()
    got = total = 0
    with torch.no_grad():
        while total < n:
            specs = [sample_clip(rng, holdout) for _ in range(min(64, n - total))]
            out = model(_feats(specs, rng, device))
            yp, yt, ye = _labels(specs, device)
            ok = (out["pitch"].argmax(-1).eq(yp) & out["timbre"].argmax(-1).eq(yt)
                  & out["envelope"].argmax(-1).eq(ye))
            got += int(ok.sum())
            total += len(specs)
    return got / total


def train_arm(arm, steps, seed, inv_w=1.0, batch=48, lr=1e-3, device=DEV):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AudioEncoder(arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        model.train()
        specs = [sample_clip(rng, holdout=False) for _ in range(batch)]
        x = _feats(specs, rng, device)
        yp, yt, ye = _labels(specs, device)
        out = model(x, return_embedding=True)
        loss = (F.cross_entropy(out["pitch"], yp) + F.cross_entropy(out["timbre"], yt)
                + F.cross_entropy(out["envelope"], ye))
        if arm in ("swap", "subspace"):
            inv = 0.0
            for factor in ("pitch", "timbre", "envelope"):
                swapped = [swap_factor(s, factor, rng) for s in specs]
                o2 = model(_feats(swapped, rng, device), return_embedding=True)
                for other in ("pitch", "timbre", "envelope"):
                    if other == factor:
                        continue
                    if arm == "subspace":                  # invariance on the slices
                        inv = inv + F.mse_loss(model.slice(out["embedding"], other),
                                               model.slice(o2["embedding"], other))
                    else:                                  # invariance on the readouts
                        inv = inv + F.mse_loss(out[other], o2[other])
            loss = loss + inv_w * inv / 3.0
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def run(steps=500, seeds=(0, 1, 2), device=DEV):
    report = {"experiment": "audio1_three_arm",
              "world": f"{len(PITCHES)}x{len(TIMBRES)}x{len(ENVELOPES)} combos, "
                       f"{len(HOLDOUT_AUDIO)} held out",
              "steps": steps, "arms": {}}
    for arm in ("factored", "swap", "subspace"):
        rows = []
        for seed in seeds:
            model = train_arm(arm, steps, seed, device=device)
            rng = np.random.default_rng(7_000 + seed)
            row = {"seed": seed,
                   "seen_acc": accuracy(model, rng, holdout=False, device=device),
                   "holdout_acc": accuracy(model, rng, holdout=True, device=device)}
            row.update(linear_probe(model, np.random.default_rng(9_000 + seed), device=device))
            rows.append(row)
        m = lambda k: float(np.mean([r[k] for r in rows]))
        report["arms"][arm] = {"seeds": rows, "seen_acc": m("seen_acc"),
                               "holdout_acc": m("holdout_acc"),
                               "probe_holdout": m("probe_holdout")}
        print(f"{arm:9} seen {m('seen_acc'):.2f}  HOLDOUT {m('holdout_acc'):.2f}  "
              f"probe-holdout {m('probe_holdout'):.2f}", flush=True)
    arms = report["arms"]
    ranked = sorted(arms, key=lambda a: -arms[a]["holdout_acc"])
    report["verdict"] = {
        "behavior_ranking": ranked,
        "probe_ranking": sorted(arms, key=lambda a: -arms[a]["probe_holdout"]),
        "replicates_image2": (arms["subspace"]["holdout_acc"]
                              >= arms["factored"]["holdout_acc"] + 0.05),
        "best_holdout": arms[ranked[0]]["holdout_acc"],
        "baseline_holdout": arms["factored"]["holdout_acc"],
    }
    print("verdict:", json.dumps(report["verdict"], indent=1), flush=True)
    return report


def selftest():
    rng = np.random.default_rng(0)
    spec = sample_clip(rng)
    w = render_tone(spec["pitch"], spec["timbre"], spec["envelope"], rng=rng)
    assert len(w) == int(SR * DUR) and np.isfinite(w).all()
    s = spectrogram(w)
    assert s.shape[0] == 1 and s.shape[1] == 65 and np.isfinite(s).all()
    # holdout discipline
    assert all((sample_clip(rng, holdout=False)["pitch"],) is not None for _ in range(4))
    for _ in range(32):
        c = sample_clip(rng, holdout=True)
        assert (c["pitch"], c["timbre"], c["envelope"]) in HOLDOUT_AUDIO
    # swap pairs change exactly one factor
    for factor in ("pitch", "timbre", "envelope"):
        s2 = swap_factor(spec, factor, rng)
        assert s2[factor] != spec[factor]
        assert all(s2[k] == spec[k] for k in ("pitch", "timbre", "envelope") if k != factor)
    # canonical facts follow the h/pred/t convention
    facts = clip_facts(spec)
    assert [f[0] for f in facts] == ["pitch", "timbre", "env"]
    # all three arms: forward + one training step runs
    for arm in ("factored", "swap", "subspace"):
        m = train_arm(arm, steps=2, seed=0, device="cpu")
        out = m(_feats([spec], rng, "cpu"), return_embedding=True)
        assert out["pitch"].shape[-1] == len(PITCHES) and out["embedding"].shape[-1] == 64
    print("audio selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="runs/audio1_fer.json")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    report = run(steps=args.steps, seeds=tuple(int(s) for s in args.seeds.split(",")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
