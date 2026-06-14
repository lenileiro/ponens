"""24kHz GAN vocoder -- the scaled path to NATURAL voice.

The 16kHz run proved the architecture (adversarial training beats spectral-only); naturalness was
capped by SCALE. This scales the three levers research says matter: 24kHz (brighter, full speech
band), MORE + LONGER data (phonetically varied SENTENCES across voices, not single words -> real
prosody and coverage), and a BIGGER generator (d=512, 8 blocks). Same HiFi-GAN recipe: Vocos iSTFT
generator + Multi-Period/Multi-Scale discriminators + feature-matching + mel + multi-res STFT loss.

Segment-based training (random 0.5s crops) so sentences batch efficiently and the GAN sees
consistent windows.

  python -m thinking.vocoder24 --build          # render the 24kHz say corpus (macOS)
  python -m thinking.vocoder24 --selftest
  python -m thinking.vocoder24 --train --steps 80000 --out runs/vocoder24.json
"""
import argparse
import json
import os
import subprocess
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from device import get_device
from .vocoder_gan import Discriminator, _feat_match

DEV = get_device()
SR = 24000
N_FFT = 1024
HOP = 256
N_BINS = N_FFT // 2 + 1                                    # 513
SEG = 12288                                               # ~0.5s training segment (mult of HOP)
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                    "speech24k")
VOICE_CANDIDATES = ("Samantha", "Daniel", "Karen", "Moira", "Tessa", "Rishi", "Fred", "Albert")
_WIN = torch.hann_window(N_FFT)

# Phonetically varied short sentences (broad coverage for a natural vocoder).
SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "she sells sea shells by the sea shore",
    "a journey of a thousand miles begins with a single step",
    "bright morning sunlight warmed the quiet valley",
    "please call me back as soon as you can",
    "the children laughed and played in the park",
    "we should leave before the storm arrives",
    "his voice echoed across the empty hall",
    "fresh bread smells wonderful in the morning",
    "the river flowed gently past the old mill",
    "thank you very much for all your help",
    "music filled the room as they danced",
    "the scientist measured the glowing liquid",
    "winter brings cold winds and falling snow",
    "every great idea starts as a simple thought",
    "the cat curled up beside the warm fire",
    "remember to water the plants this weekend",
    "they traveled far to reach the mountain peak",
    "her painting captured the colors of autumn",
    "the train departs from platform nine",
    "honesty and patience build lasting trust",
    "the ocean waves crashed against the rocks",
    "he carefully folded the letter and sighed",
    "tomorrow we will visit the ancient castle",
    "a gentle breeze carried the scent of roses",
    "the engineer fixed the broken machine quickly",
    "stars appeared one by one in the dark sky",
    "she whispered a secret to her best friend",
    "the bakery opens early every single day",
    "courage means doing what is right despite fear",
]


def _available_voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=30)
    names = {ln.split()[0] for ln in out.stdout.splitlines() if ln.strip()}
    return [v for v in VOICE_CANDIDATES if v in names]


def build_corpus(root=ROOT):
    os.makedirs(root, exist_ok=True)
    voices = _available_voices()
    if len(voices) < 3:
        raise RuntimeError(f"need >=3 voices, found {voices}")
    manifest = {"voices": voices, "n_sentences": len(SENTENCES), "clips": []}
    for vi, voice in enumerate(voices):
        for si, sent in enumerate(SENTENCES):
            p = os.path.join(root, f"{vi}_{si}.wav")
            if not os.path.exists(p):
                subprocess.run(["say", "-v", voice, "-r", "180", "-o", p,
                                "--file-format=WAVE", f"--data-format=LEI16@{SR}", sent],
                               check=True, timeout=60)
            manifest["clips"].append({"voice": vi, "sent": si, "path": f"{vi}_{si}.wav"})
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    secs = sum(wave.open(os.path.join(root, c["path"])).getnframes() for c in manifest["clips"]) / SR
    print(f"24kHz corpus: {len(manifest['clips'])} clips, {len(voices)} voices, {secs:.0f}s audio")
    return manifest


def load_corpus(root=ROOT):
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    waves = []
    for c in manifest["clips"]:
        w = wave.open(os.path.join(root, c["path"]))
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        nz = np.where(np.abs(x) > 0.01)[0]
        if len(nz):
            x = x[max(0, nz[0] - 512):nz[-1] + 512]
        if len(x) >= SR // 4:                              # keep clips >= 0.25s
            waves.append({**c, "wave": x})
    return manifest, waves


def _split(clips, holdout_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clips))
    k = max(1, int(len(clips) * holdout_frac))
    hold = set(idx[:k].tolist())
    return [c for i, c in enumerate(clips) if i not in hold], [c for i, c in enumerate(clips) if i in hold]


def _stft(wav):
    return torch.stft(wav, N_FFT, HOP, N_FFT, _WIN.to(wav.device), center=True, return_complex=True)


def logmag_feat(wav):
    return torch.log1p(_stft(wav).abs())


def _seg_batch(clips, rng, batch, device):
    """Random SEG-length segments (zero-padded if a clip is shorter)."""
    out = []
    for _ in range(batch):
        w = clips[int(rng.integers(len(clips)))]["wave"]
        if len(w) > SEG:
            s = int(rng.integers(len(w) - SEG))
            out.append(w[s:s + SEG])
        else:
            out.append(np.pad(w, (0, SEG - len(w))))
    return torch.tensor(np.stack(out), dtype=torch.float32, device=device)


class ConvNeXtBlock(nn.Module):
    def __init__(self, d, mult=3):
        super().__init__()
        self.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, mult * d)
        self.pw2 = nn.Linear(mult * d, d)

    def forward(self, x):
        r = x
        x = self.dw(x).transpose(1, 2)
        x = self.pw2(F.gelu(self.pw1(self.norm(x)))).transpose(1, 2)
        return r + x


class Vocos24(nn.Module):
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
    key = (n_mels, wav.device)
    fb = _mel.cache.get(key)
    if fb is None:
        f = torch.linspace(0, SR / 2, N_BINS, device=wav.device)
        mmax = 2595 * np.log10(1 + (SR / 2) / 700)
        ctr = 700 * (10 ** (torch.linspace(0, mmax, n_mels + 2, device=wav.device) / 2595) - 1)
        fb = torch.zeros(n_mels, N_BINS, device=wav.device)
        for m in range(1, n_mels + 1):
            lo, ce, hi = ctr[m - 1], ctr[m], ctr[m + 1]
            fb[m - 1] = torch.clamp(torch.minimum((f - lo) / (ce - lo + 1e-9),
                                                  (hi - f) / (hi - ce + 1e-9)), 0, 1)
        _mel.cache[key] = fb
    return torch.log1p(fb @ spec)
_mel.cache = {}


def write_wav(path, wav):
    y = wav / (np.abs(wav).max() + 1e-8) * 0.95
    w = wave.open(path, "w")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((y * 32767).astype(np.int16).tobytes()); w.close()


def griffin_lim(logmag, n_iter=120):
    mag = np.expm1(np.clip(logmag, 0, None))
    win = np.hanning(N_FFT).astype(np.float32)
    T = mag.shape[1]; length = HOP * (T - 1) + N_FFT
    rng = np.random.default_rng(0)
    phase = np.exp(2j * np.pi * rng.random(mag.shape))
    for _ in range(n_iter):
        spec = mag * phase
        wav = np.zeros(length); wsum = np.zeros(length) + 1e-8
        for t in range(T):
            fr = np.fft.irfft(spec[:, t], n=N_FFT) * win
            wav[t * HOP:t * HOP + N_FFT] += fr; wsum[t * HOP:t * HOP + N_FFT] += win ** 2
        wav /= wsum
        new = np.zeros_like(spec)
        for t in range(T):
            new[:, t] = np.fft.rfft(wav[t * HOP:t * HOP + N_FFT] * win, n=N_FFT)
        phase = np.exp(1j * np.angle(new))
    return wav.astype(np.float32)


def train(steps=80000, seed=0, device=DEV, batch=16, lr=2e-4, d=512):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _m, clips = load_corpus()
    train_clips, _ = _split(clips)
    G = Vocos24(d=d).to(device)
    D = Discriminator().to(device)
    optG = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.8, 0.99))
    optD = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.8, 0.99))
    for st in range(1, steps + 1):
        real = _seg_batch(train_clips, rng, batch, device)
        fake = G(logmag_feat(real), length=real.shape[1])
        optD.zero_grad()
        dr, _ = D(real)
        df, _ = D(fake.detach())
        d_loss = sum(((r - 1) ** 2).mean() + (f ** 2).mean() for r, f in zip(dr, df))
        d_loss.backward(); optD.step()
        optG.zero_grad()
        dr, fr = D(real)
        df, ff = D(fake)
        adv = sum(((f - 1) ** 2).mean() for f in df)
        fm = _feat_match(fr, ff)
        mel = F.l1_loss(_mel(fake), _mel(real))
        stft = multi_res_stft_loss(fake, real)
        g = adv + 2.0 * fm + 45.0 * mel + 5.0 * stft
        g.backward(); optG.step()
        if st % max(1, steps // 12) == 0 or st == steps:
            print(f"  v24 {st}/{steps} G {g.item():.3f} (adv {adv.item():.2f} mel {mel.item():.3f}) "
                  f"D {d_loss.item():.3f}", flush=True)
    return G


def _spec_distance(a, b):
    ta, tb = torch.tensor(a[None]), torch.tensor(b[None])
    n = min(ta.shape[1], tb.shape[1])
    return float(multi_res_stft_loss(ta[:, :n], tb[:, :n]))


def evaluate(G, device=DEV, n=20, save_dir=None):
    _m, clips = load_corpus()
    _, test = _split(clips)
    G.eval()
    gan_d, gl_d = [], []
    with torch.no_grad():
        for i, c in enumerate(test[:n]):
            t = c["wave"][:SR * 3]
            feat = logmag_feat(torch.tensor(t[None], device=device))
            g = G(feat, length=t.shape[0])[0].cpu().numpy()
            gl = griffin_lim(feat[0].cpu().numpy())
            gan_d.append(_spec_distance(g, t)); gl_d.append(_spec_distance(gl, t))
            if save_dir and i < 4:
                os.makedirs(save_dir, exist_ok=True)
                write_wav(os.path.join(save_dir, f"v24_{i}_oracle.wav"), t)
                write_wav(os.path.join(save_dir, f"v24_{i}_GAN.wav"), g)
                write_wav(os.path.join(save_dir, f"v24_{i}_griffinlim.wav"), gl)
    return {"gan_dist": float(np.mean(gan_d)), "griffinlim_dist": float(np.mean(gl_d)), "n": len(gan_d)}


def run(steps=80000, seed=0, device=DEV, save_dir="data/synth"):
    G = train(steps=steps, seed=seed, device=device)
    ev = evaluate(G, device=device, save_dir=save_dir)
    report = {"experiment": "vocoder24", "sr": SR, "steps": steps, **ev,
              "improvement_pct": 100 * (ev["griffinlim_dist"] - ev["gan_dist"]) / max(1e-9, ev["griffinlim_dist"])}
    print(f"\n24kHz HELD-OUT distance: GAN {ev['gan_dist']:.4f} vs Griffin-Lim {ev['griffinlim_dist']:.4f}")
    return report, G


def selftest():
    if not os.path.exists(os.path.join(ROOT, "manifest.json")):
        build_corpus()
    _m, clips = load_corpus()
    assert clips and clips[0]["wave"].ndim == 1
    rng = np.random.default_rng(0)
    real = _seg_batch(clips, rng, 2, "cpu")
    assert real.shape == (2, SEG)
    G = Vocos24(d=64, blocks=2)
    fake = G(logmag_feat(real), length=SEG)
    assert fake.shape == (2, SEG), fake.shape
    D = Discriminator()
    outs, _ = D(real)
    assert len(outs) == 8
    G2 = train(steps=2, seed=0, device="cpu")
    print("vocoder24 selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/vocoder24.json")
    ap.add_argument("--checkpoint", default="runs/vocoder24.pt")
    ap.add_argument("--synth-out", default="data/synth", dest="synth_out")
    args = ap.parse_args(argv)
    if args.build:
        build_corpus(); return
    if args.selftest:
        selftest(); return
    if args.train:
        report, G = run(steps=args.steps, seed=args.seed, save_dir=args.synth_out)
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": G.state_dict(), "d": 512}, args.checkpoint)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"saved -> {args.out}, {args.checkpoint}")
        return
    ap.error("choose --build / --selftest / --train")


if __name__ == "__main__":
    main()
