"""Neural vocoder: log-magnitude spectrogram -> NATURAL waveform (learned phase).

Griffin-Lim sounds robotic because it GUESSES phase (iterative, no model). A neural vocoder
LEARNS phase: a conv net maps the input log-magnitude spectrogram to the full complex STFT
(real + imaginary parts per frame), and a differentiable iSTFT reconstructs the waveform. This is
the Vocos approach -- operate in the STFT domain, predict coefficients, iSTFT -- which avoids the
transposed-conv artifacts of time-domain vocoders and trains stably WITHOUT a GAN (multi-resolution
STFT loss alone). It is a drop-in replacement for griffin_lim() that every synth (sing, mimic,
pronounce, speak) can reuse: model predicts log-mag, this turns log-mag into natural audio.

Trained on real `say` speech (16kHz). The decisive test: on HELD-OUT clips, does it reconstruct
with LOWER spectral distance to the true waveform than Griffin-Lim? (lower = more natural / correct
phase). Produces A/B .wav so the difference is audible.

  python -m thinking.neuralvocoder --selftest
  python -m thinking.neuralvocoder --train --steps 4000 --out runs/neural_vocoder.json
"""
import argparse
import json
import os
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .vocoder import load_bank, N_BINS, N_FFT, HOP, SR, write_wav, griffin_lim, stft_logmag

DEV = get_device()
_WIN = torch.hann_window(N_FFT)


def _split(clips, holdout_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clips))
    k = max(1, int(len(clips) * holdout_frac))
    hold = set(idx[:k].tolist())
    return [c for i, c in enumerate(clips) if i not in hold], [c for i, c in enumerate(clips) if i in hold]


def _stft(wav):
    return torch.stft(wav, N_FFT, HOP, N_FFT, _WIN.to(wav.device), center=True,
                      return_complex=True)


def _istft(spec, length=None):
    return torch.istft(spec, N_FFT, HOP, N_FFT, _WIN.to(spec.device), center=True, length=length)


def logmag_feat(wav):
    """Input feature: log1p magnitude STFT (B, N_BINS, T) -- what the TTS models predict."""
    return torch.log1p(_stft(wav).abs())


class ConvNeXtBlock(nn.Module):
    def __init__(self, d, mult=3):
        super().__init__()
        self.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, mult * d)
        self.pw2 = nn.Linear(mult * d, d)

    def forward(self, x):                                   # x: (B, d, T)
        r = x
        x = self.dw(x).transpose(1, 2)                     # (B, T, d)
        x = self.pw2(F.gelu(self.pw1(self.norm(x)))).transpose(1, 2)
        return r + x


class Vocos(nn.Module):
    """log-mag (B,N_BINS,T) -> complex STFT coeffs (real+imag) -> iSTFT -> waveform."""

    def __init__(self, d=256, blocks=6):
        super().__init__()
        self.inp = nn.Conv1d(N_BINS, d, 7, padding=3)
        self.norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([ConvNeXtBlock(d) for _ in range(blocks)])
        self.out = nn.Conv1d(d, 2 * N_BINS, 1)             # real + imag per bin

    def forward(self, logmag, length=None):
        x = self.inp(logmag)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        ri = self.out(x)                                   # (B, 2*N_BINS, T)
        real, imag = ri[:, :N_BINS], ri[:, N_BINS:]
        spec = torch.complex(real, imag)                   # learned magnitude AND phase
        return _istft(spec, length=length)


def multi_res_stft_loss(pred, target, ffts=(256, 512, 1024)):
    """Sum of L1 log-magnitude distances across STFT resolutions (the stable, no-GAN objective)."""
    loss = 0.0
    for nf in ffts:
        w = torch.hann_window(nf, device=pred.device)
        hop = nf // 4
        P = torch.stft(pred, nf, hop, nf, w, center=True, return_complex=True).abs()
        T = torch.stft(target, nf, hop, nf, w, center=True, return_complex=True).abs()
        loss = loss + F.l1_loss(torch.log1p(P), torch.log1p(T))
    return loss / len(ffts)


def _waves(clips, device):
    """Padded oracle waveforms (B, samples) for a list of bank clips."""
    waves = []
    for c in clips:
        w = wave.open(os.path.join(_root(), c["path"]))
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        nz = np.where(np.abs(x) > 0.01)[0]
        if len(nz):
            x = x[max(0, nz[0] - 256):nz[-1] + 256]
        waves.append(x)
    n = max(len(w) for w in waves)
    n = ((n + HOP - 1) // HOP) * HOP
    return torch.tensor(np.stack([np.pad(w, (0, n - len(w))) for w in waves]),
                        dtype=torch.float32, device=device)


def _root():
    from .vocoder import ROOT
    return ROOT


def train(steps=4000, seed=0, device=DEV, batch=16, lr=2e-4, d=256):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _manifest, clips = load_bank()
    train_clips, _ = _split(clips)
    model = Vocos(d=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        model.train()
        bc = [train_clips[int(rng.integers(len(train_clips)))] for _ in range(batch)]
        tgt = _waves(bc, device)
        feat = logmag_feat(tgt)
        pred = model(feat, length=tgt.shape[1])
        loss = multi_res_stft_loss(pred, tgt) + 0.1 * F.l1_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % max(1, steps // 8) == 0 or st == steps:
            print(f"  nv {st}/{steps} loss {loss.item():.4f}", flush=True)
    return model


def _spec_distance(wav_a, wav_b):
    """Multi-res STFT log-mag distance (naturalness/correctness proxy; lower = better)."""
    a = torch.tensor(wav_a[None], dtype=torch.float32)
    b = torch.tensor(wav_b[None], dtype=torch.float32)
    n = min(a.shape[1], b.shape[1])
    return float(multi_res_stft_loss(a[:, :n], b[:, :n]))


def evaluate(model, device=DEV, n=40, seed=1, save_dir=None):
    """Held-out clips: neural-vocoder vs Griffin-Lim distance to the TRUE waveform."""
    _manifest, clips = load_bank()
    _, test_clips = _split(clips)
    model.eval()
    nv_d, gl_d = [], []
    with torch.no_grad():
        for i, c in enumerate(test_clips[:n]):
            tgt = _waves([c], device)
            feat = logmag_feat(tgt)
            nv = model(feat, length=tgt.shape[1])[0].cpu().numpy()
            t = tgt[0].cpu().numpy()
            gl = griffin_lim(feat[0].cpu().numpy())        # same input features, GL phase
            nv_d.append(_spec_distance(nv, t))
            gl_d.append(_spec_distance(gl, t))
            if save_dir and i < 3:
                os.makedirs(save_dir, exist_ok=True)
                write_wav(os.path.join(save_dir, f"vocoder_{i}_oracle.wav"), t)
                write_wav(os.path.join(save_dir, f"vocoder_{i}_neural.wav"), nv)
                write_wav(os.path.join(save_dir, f"vocoder_{i}_griffinlim.wav"), gl)
    return {"neural_dist": float(np.mean(nv_d)), "griffinlim_dist": float(np.mean(gl_d)),
            "n": len(nv_d)}


def run(steps=4000, seed=0, device=DEV, save_dir="data/synth"):
    model = train(steps=steps, seed=seed, device=device)
    ev = evaluate(model, device=device, save_dir=save_dir)
    report = {"experiment": "neural_vocoder", "steps": steps, **ev,
              "neural_beats_griffinlim": ev["neural_dist"] < ev["griffinlim_dist"],
              "improvement_pct": 100 * (ev["griffinlim_dist"] - ev["neural_dist"])
              / max(1e-9, ev["griffinlim_dist"])}
    print(f"\nHELD-OUT reconstruction distance (lower=more natural):")
    print(f"  Griffin-Lim : {ev['griffinlim_dist']:.4f}")
    print(f"  NEURAL      : {ev['neural_dist']:.4f}  ({report['improvement_pct']:.0f}% better)")
    print(f"  neural beats Griffin-Lim: {report['neural_beats_griffinlim']}", flush=True)
    return report, model


def neural_vocode(model, logmag_np, device=DEV):
    """Drop-in for griffin_lim(): log-mag (N_BINS,T) numpy -> natural waveform numpy."""
    model.eval()
    with torch.no_grad():
        feat = torch.tensor(logmag_np[None], dtype=torch.float32, device=device)
        return model(feat)[0].cpu().numpy()


def selftest():
    _manifest, clips = load_bank()
    tr, te = _split(clips)
    assert tr and te
    m = Vocos(d=32, blocks=2)
    w = _waves(tr[:2], "cpu")
    feat = logmag_feat(w)
    assert feat.shape[1] == N_BINS
    out = m(feat, length=w.shape[1])
    assert out.shape == w.shape, (out.shape, w.shape)
    loss = multi_res_stft_loss(out, w)
    loss.backward()
    model = train(steps=3, seed=0, device="cpu")
    ev = evaluate(model, device="cpu", n=4)
    assert "neural_dist" in ev and "griffinlim_dist" in ev
    print("neural vocoder selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/neural_vocoder.json")
    ap.add_argument("--checkpoint", default="runs/neural_vocoder.pt")
    ap.add_argument("--synth-out", default="data/synth", dest="synth_out")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if args.train:
        report, model = run(steps=args.steps, seed=args.seed, save_dir=args.synth_out)
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "d": 256}, args.checkpoint)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"saved -> {args.out}, {args.checkpoint}")
        return
    ap.error("choose --selftest / --train")


if __name__ == "__main__":
    main()
