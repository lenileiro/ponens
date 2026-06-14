"""Natural voice from REAL human speech: GAN vocoder trained on LJSpeech, not synthetic `say`.

Root cause of the lingering robotic sound (researched 2026-06): a vocoder can only be as natural as
its TRAINING TARGET. The prior runs trained on macOS `say` output -- itself synthetic/robotic TTS --
so the vocoder faithfully reproduced roboticness. Every natural vocoder in the literature trains on
HOURS of real recorded human speech; the standard corpus is LJSpeech (one human reading, 22kHz).
This streams a LJSpeech subset and trains the same HiFi-GAN-recipe vocoder on REAL human audio, so
the reconstruction target -- and thus the output -- is a natural human voice.

  python -m thinking.realvoice --fetch          # stream a LJSpeech subset (real human speech)
  python -m thinking.realvoice --selftest
  python -m thinking.realvoice --train --steps 100000 --out runs/realvoice.json
"""
import argparse
import json
import os
import tarfile
import urllib.request
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .vocoder_gan import Discriminator, _feat_match

DEV = get_device()
URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
SR = 22050
N_FFT = 1024
HOP = 256
N_BINS = N_FFT // 2 + 1
SEG = 11264                                                # ~0.5s segment (mult of HOP)
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ljspeech")
_WIN = torch.hann_window(N_FFT)


def fetch(n_clips=700, root=ROOT, byte_cap_gb=0.6):
    """Stream the LJSpeech tar.bz2 and extract the first n_clips wavs (real human speech)."""
    os.makedirs(root, exist_ok=True)
    got = 0
    cap = int(byte_cap_gb * 1e9)
    seen = [0]

    class _C:
        def __init__(s, f): s.f = f
        def read(s, n):
            b = s.f.read(n); seen[0] += len(b); return b

    print(f"streaming {URL} (cap {byte_cap_gb}GB, target {n_clips} clips)...")
    req = urllib.request.Request(URL, headers={"User-Agent": "ponens/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        tar = tarfile.open(fileobj=_C(resp), mode="r|bz2")
        names = []
        for m in tar:
            if seen[0] > cap or got >= n_clips:
                break
            if m.name.endswith(".wav"):
                data = tar.extractfile(m).read()
                out = os.path.join(root, os.path.basename(m.name))
                with open(out, "wb") as f:
                    f.write(data)
                names.append(os.path.basename(out))
                got += 1
    json.dump({"sr": SR, "clips": names}, open(os.path.join(root, "manifest.json"), "w"))
    print(f"fetched {got} real human clips, {seen[0]/1e9:.2f}GB streamed -> {root}")
    return got


def load_corpus(root=ROOT):
    man = json.load(open(os.path.join(root, "manifest.json")))
    waves = []
    for name in man["clips"]:
        w = wave.open(os.path.join(root, name))
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if sr != SR:                                       # LJSpeech is 22050; resample if not
            import math
            idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr)).astype(np.int64)
            x = x[idx]
        if len(x) >= SR // 2:
            waves.append({"wave": x})
    return man, waves


def _split(clips, holdout_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clips)); k = max(1, int(len(clips) * holdout_frac))
    hold = set(idx[:k].tolist())
    return [c for i, c in enumerate(clips) if i not in hold], [c for i, c in enumerate(clips) if i in hold]


def _stft(wav):
    return torch.stft(wav, N_FFT, HOP, N_FFT, _WIN.to(wav.device), center=True, return_complex=True)


def logmag_feat(wav):
    return torch.log1p(_stft(wav).abs())


def _seg_batch(clips, rng, batch, device):
    out = []
    for _ in range(batch):
        w = clips[int(rng.integers(len(clips)))]["wave"]
        if len(w) > SEG:
            s = int(rng.integers(len(w) - SEG)); out.append(w[s:s + SEG])
        else:
            out.append(np.pad(w, (0, SEG - len(w))))
    return torch.tensor(np.stack(out), dtype=torch.float32, device=device)


class ConvNeXtBlock(nn.Module):
    def __init__(self, d, mult=3):
        super().__init__()
        self.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, mult * d); self.pw2 = nn.Linear(mult * d, d)

    def forward(self, x):
        r = x
        x = self.dw(x).transpose(1, 2)
        return r + self.pw2(F.gelu(self.pw1(self.norm(x)))).transpose(1, 2)


class Vocos(nn.Module):
    def __init__(self, d=512, blocks=8):
        super().__init__()
        self.inp = nn.Conv1d(N_BINS, d, 7, padding=3)
        self.norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([ConvNeXtBlock(d) for _ in range(blocks)])
        self.out = nn.Conv1d(d, 2 * N_BINS, 1)

    def forward(self, logmag, length=None):
        x = self.norm(self.inp(logmag).transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        ri = self.out(x)
        spec = torch.complex(ri[:, :N_BINS], ri[:, N_BINS:])
        return torch.istft(spec, N_FFT, HOP, N_FFT, _WIN.to(spec.device), center=True, length=length)


def multi_res_stft_loss(pred, target, ffts=(512, 1024, 2048)):
    loss = 0.0
    for nf in ffts:
        w = torch.hann_window(nf, device=pred.device)
        P = torch.stft(pred, nf, nf // 4, nf, w, center=True, return_complex=True).abs()
        T = torch.stft(target, nf, nf // 4, nf, w, center=True, return_complex=True).abs()
        loss = loss + F.l1_loss(torch.log1p(P), torch.log1p(T))
    return loss / len(ffts)


def _mel(wav, n_mels=100):
    spec = _stft(wav).abs()
    fb = _mel.cache.get(wav.device)
    if fb is None:
        f = torch.linspace(0, SR / 2, N_BINS, device=wav.device)
        mmax = 2595 * np.log10(1 + (SR / 2) / 700)
        ctr = 700 * (10 ** (torch.linspace(0, mmax, n_mels + 2, device=wav.device) / 2595) - 1)
        fb = torch.zeros(n_mels, N_BINS, device=wav.device)
        for m in range(1, n_mels + 1):
            lo, ce, hi = ctr[m - 1], ctr[m], ctr[m + 1]
            fb[m - 1] = torch.clamp(torch.minimum((f - lo) / (ce - lo + 1e-9), (hi - f) / (hi - ce + 1e-9)), 0, 1)
        _mel.cache[wav.device] = fb
    return torch.log1p(fb @ spec)
_mel.cache = {}


def write_wav(path, wav):
    y = wav / (np.abs(wav).max() + 1e-8) * 0.95
    w = wave.open(path, "w"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((y * 32767).astype(np.int16).tobytes()); w.close()


def griffin_lim(logmag, n_iter=120):
    mag = np.expm1(np.clip(logmag, 0, None)); win = np.hanning(N_FFT).astype(np.float32)
    T = mag.shape[1]; length = HOP * (T - 1) + N_FFT
    rng = np.random.default_rng(0); phase = np.exp(2j * np.pi * rng.random(mag.shape))
    for _ in range(n_iter):
        spec = mag * phase; wav = np.zeros(length); wsum = np.zeros(length) + 1e-8
        for t in range(T):
            fr = np.fft.irfft(spec[:, t], n=N_FFT) * win
            wav[t * HOP:t * HOP + N_FFT] += fr; wsum[t * HOP:t * HOP + N_FFT] += win ** 2
        wav /= wsum; new = np.zeros_like(spec)
        for t in range(T):
            new[:, t] = np.fft.rfft(wav[t * HOP:t * HOP + N_FFT] * win, n=N_FFT)
        phase = np.exp(1j * np.angle(new))
    return wav.astype(np.float32)


def train(steps=100000, seed=0, device=DEV, batch=16, lr=2e-4, d=512):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    _m, clips = load_corpus(); tr, _ = _split(clips)
    G = Vocos(d=d).to(device); D = Discriminator().to(device)
    oG = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.8, 0.99))
    oD = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.8, 0.99))
    for st in range(1, steps + 1):
        real = _seg_batch(tr, rng, batch, device)
        fake = G(logmag_feat(real), length=real.shape[1])
        oD.zero_grad()
        dr, _ = D(real); df, _ = D(fake.detach())
        dl = sum(((r - 1) ** 2).mean() + (f ** 2).mean() for r, f in zip(dr, df))
        dl.backward(); oD.step()
        oG.zero_grad()
        dr, fr = D(real); df, ff = D(fake)
        adv = sum(((f - 1) ** 2).mean() for f in df)
        mel = F.l1_loss(_mel(fake), _mel(real))
        g = adv + 2.0 * _feat_match(fr, ff) + 45.0 * mel + 5.0 * multi_res_stft_loss(fake, real)
        g.backward(); oG.step()
        if st % max(1, steps // 12) == 0 or st == steps:
            print(f"  rv {st}/{steps} G {g.item():.3f} (adv {adv.item():.2f} mel {mel.item():.3f}) D {dl.item():.3f}", flush=True)
    return G


def _spec_distance(a, b):
    ta, tb = torch.tensor(a[None]), torch.tensor(b[None]); n = min(ta.shape[1], tb.shape[1])
    return float(multi_res_stft_loss(ta[:, :n], tb[:, :n]))


def evaluate(G, device=DEV, n=20, save_dir=None):
    _m, clips = load_corpus(); _, test = _split(clips)
    G.eval(); gan_d, gl_d = [], []
    with torch.no_grad():
        for i, c in enumerate(test[:n]):
            t = c["wave"][:SR * 4]
            feat = logmag_feat(torch.tensor(t[None], device=device))
            g = G(feat, length=t.shape[0])[0].cpu().numpy()
            gl = griffin_lim(feat[0].cpu().numpy())
            gan_d.append(_spec_distance(g, t)); gl_d.append(_spec_distance(gl, t))
            if save_dir and i < 5:
                os.makedirs(save_dir, exist_ok=True)
                write_wav(os.path.join(save_dir, f"real{i}_oracle.wav"), t)
                write_wav(os.path.join(save_dir, f"real{i}_GAN.wav"), g)
                write_wav(os.path.join(save_dir, f"real{i}_griffinlim.wav"), gl)
    return {"gan_dist": float(np.mean(gan_d)), "griffinlim_dist": float(np.mean(gl_d)), "n": len(gan_d)}


def run(steps=100000, seed=0, device=DEV, save_dir="data/synth"):
    G = train(steps=steps, seed=seed, device=device)
    ev = evaluate(G, device=device, save_dir=save_dir)
    report = {"experiment": "realvoice_ljspeech", "sr": SR, "steps": steps, **ev}
    print(f"\nLJSpeech (REAL human) held-out: GAN {ev['gan_dist']:.4f} vs Griffin-Lim {ev['griffinlim_dist']:.4f}")
    return report, G


def selftest():
    if not os.path.exists(os.path.join(ROOT, "manifest.json")):
        print("no LJSpeech yet; run --fetch first (skipping)")
        return
    _m, clips = load_corpus(); assert clips
    rng = np.random.default_rng(0)
    real = _seg_batch(clips, rng, 2, "cpu"); assert real.shape == (2, SEG)
    G = Vocos(d=64, blocks=2)
    assert G(logmag_feat(real), length=SEG).shape == (2, SEG)
    train(steps=2, seed=0, device="cpu")
    print("realvoice selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--n-clips", type=int, default=700, dest="n_clips")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/realvoice.json")
    ap.add_argument("--checkpoint", default="runs/realvoice.pt")
    ap.add_argument("--synth-out", default="data/synth", dest="synth_out")
    args = ap.parse_args(argv)
    if args.fetch:
        fetch(n_clips=args.n_clips); return
    if args.selftest:
        selftest(); return
    if args.train:
        report, G = run(steps=args.steps, seed=args.seed, save_dir=args.synth_out)
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": G.state_dict(), "d": 512}, args.checkpoint)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"saved -> {args.out}, {args.checkpoint}")
        return
    ap.error("choose --fetch / --selftest / --train")


if __name__ == "__main__":
    main()
