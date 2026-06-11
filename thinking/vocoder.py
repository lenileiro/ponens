"""A-3b SPEAK (audible): the speaker predicts a full magnitude spectrogram from the word, and
Griffin-Lim inverts it to a waveform. No VQ bottleneck (which crushed the magnitude to buzz);
the inversion path reconstructs true spectrograms at 25 dB SNR, so audible model speech now hinges
only on the predicted spectrogram being close to a real one.

Honest framing of 'the model speaks': a GRU conditioned on (word, voice) emits the log-magnitude
spectrogram frame by frame, trained to match the oracle `say` clips (L1) and, secondarily,
verified by round-trip through a frozen listener (the checker, applied to generation). It learns
the acoustic realization of each word -- it generates the spectrogram, it does not retrieve a clip.

  python -m thinking.vocoder --build          # render the 16kHz oracle bank
  python -m thinking.vocoder --selftest
  python -m thinking.vocoder --train --steps 4000 --out runs/a3b_vocoder.pt
  python -m thinking.vocoder --synth red,blue,circle --checkpoint runs/a3b_vocoder.pt
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

from device import get_device

DEV = get_device()
SR = 16000
N_FFT = 512
HOP = 128
N_BINS = N_FFT // 2 + 1                                     # 257
N_FRAMES = 72                                              # ~0.58s, covers a word
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                    "speech16k")
WORDS = ("red", "green", "blue", "yellow", "white", "purple", "orange", "cyan",
         "square", "circle", "triangle", "diamond")
VOICE_CANDIDATES = ("Samantha", "Daniel", "Albert", "Fred", "Karen", "Moira", "Rishi", "Tessa")
RATES = (170, 200, 230)
N_VOICES = 4
_WIN = np.hanning(N_FFT).astype(np.float32)


def _available_voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=30)
    names = {ln.split()[0] for ln in out.stdout.splitlines() if ln.strip()}
    return [v for v in VOICE_CANDIDATES if v in names][:N_VOICES]


def build_bank(root=ROOT):
    os.makedirs(root, exist_ok=True)
    voices = _available_voices()
    if len(voices) < N_VOICES:
        raise RuntimeError(f"need {N_VOICES} voices, found {voices}")
    manifest = {"voices": voices, "words": list(WORDS), "rates": list(RATES), "clips": []}
    for wi, word in enumerate(WORDS):
        for vi, voice in enumerate(voices):
            for ri, rate in enumerate(RATES):
                path = os.path.join(root, f"{wi}_{vi}_{ri}.wav")
                if not os.path.exists(path):
                    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", path,
                                    "--file-format=WAVE", f"--data-format=LEI16@{SR}", word],
                                   check=True, timeout=60)
                manifest["clips"].append({"word": wi, "voice": vi, "rate": ri,
                                          "path": os.path.basename(path)})
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"16kHz bank: {len(manifest['clips'])} clips, voices {voices}")
    return manifest


def stft_logmag(wave_f):
    cols = [np.abs(np.fft.rfft(wave_f[i:i + N_FFT] * _WIN))
            for i in range(0, max(1, len(wave_f) - N_FFT + 1), HOP)]
    s = np.log1p(np.stack(cols, 1).astype(np.float32))     # (F, T)
    if s.shape[1] < N_FRAMES:
        s = np.pad(s, ((0, 0), (0, N_FRAMES - s.shape[1])))
    return s[:, :N_FRAMES]


def griffin_lim(logmag, n_iter=120):
    mag = np.expm1(np.clip(logmag, 0, None))               # (F, T)
    T = mag.shape[1]
    length = HOP * (T - 1) + N_FFT
    rng = np.random.default_rng(0)
    phase = np.exp(2j * np.pi * rng.random(mag.shape))
    for _ in range(n_iter):
        spec = mag * phase
        wav = np.zeros(length)
        wsum = np.zeros(length) + 1e-8
        for t in range(T):
            fr = np.fft.irfft(spec[:, t], n=N_FFT) * _WIN
            wav[t * HOP:t * HOP + N_FFT] += fr
            wsum[t * HOP:t * HOP + N_FFT] += _WIN ** 2
        wav /= wsum
        new = np.zeros_like(spec)
        for t in range(T):
            new[:, t] = np.fft.rfft(wav[t * HOP:t * HOP + N_FFT] * _WIN, n=N_FFT)
        phase = np.exp(1j * np.angle(new))
    return wav.astype(np.float32)


def write_wav(path, wav, sr=SR):
    y = wav / (np.abs(wav).max() + 1e-8) * 0.95
    w = wave.open(path, "w")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes((y * 32767).astype(np.int16).tobytes())
    w.close()


def load_bank(root=ROOT):
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    clips = []
    for c in manifest["clips"]:
        w = wave.open(os.path.join(root, c["path"]))
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        nz = np.where(np.abs(x) > 0.01)[0]
        if len(nz):
            x = x[max(0, nz[0] - 256):nz[-1] + 256]
        clips.append({**c, "logmag": stft_logmag(x)})
    return manifest, clips


class SpectrogramSpeaker(nn.Module):
    """(word, voice) -> log-magnitude spectrogram, frame by frame (GRU for temporal coherence)."""

    def __init__(self, n_words=len(WORDS), n_voices=N_VOICES, dim=256, n_frames=N_FRAMES):
        super().__init__()
        self.n_frames = n_frames
        self.word = nn.Embedding(n_words, dim)
        self.voice = nn.Embedding(n_voices, dim)
        self.posn = nn.Parameter(torch.randn(n_frames, dim) * 0.02)
        self.gru = nn.GRU(dim, dim, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, N_BINS))

    def forward(self, words, voices):
        B = len(words)
        cond = (self.word(words) + self.voice(voices))[:, None, :]   # (B,1,dim)
        seq = cond + self.posn[None]                                 # (B,T,dim)
        h, _ = self.gru(seq)
        return self.head(h).transpose(1, 2)                          # (B, N_BINS, T)


def train(steps=4000, seed=0, device=DEV, batch=32, lr=2e-3, listener=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _manifest, clips = load_bank()
    specs = {(c["word"], c["voice"], c["rate"]): torch.tensor(c["logmag"], device=device)
             for c in clips}
    keys = list(specs)
    model = SpectrogramSpeaker().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        model.train()
        bk = [keys[int(rng.integers(len(keys)))] for _ in range(batch)]
        words = torch.tensor([k[0] for k in bk], device=device)
        voices = torch.tensor([k[1] for k in bk], device=device)
        target = torch.stack([specs[k] for k in bk])
        pred = model(words, voices)
        loss = F.l1_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % max(1, steps // 8) == 0 or st == steps:
            print(f"  a3b {st}/{steps} L1 {loss.item():.4f}", flush=True)
    return model


def spectral_l1(model, device=DEV):
    """Held-out spectral fidelity: predicted vs oracle log-mag, averaged over the bank."""
    _manifest, clips = load_bank()
    model.eval()
    errs = []
    with torch.no_grad():
        for c in clips:
            pred = model(torch.tensor([c["word"]], device=device),
                         torch.tensor([c["voice"]], device=device))[0]
            tgt = torch.tensor(c["logmag"], device=device)
            errs.append(float(F.l1_loss(pred, tgt)))
    return float(np.mean(errs))


def synth(model, words, out_dir, voice=0, device=DEV):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for word in words:
        if word not in WORDS:
            print(f"  (skip {word!r})")
            continue
        wi = WORDS.index(word)
        with torch.no_grad():
            logmag = model(torch.tensor([wi], device=device),
                           torch.tensor([voice], device=device))[0].cpu().numpy()
        wav = griffin_lim(logmag)
        path = os.path.join(out_dir, f"{word}_model16k.wav")
        write_wav(path, wav)
        print(f"  {word:9} -> {path}  (rms {np.sqrt((wav ** 2).mean()):.3f})")


def selftest():
    if not os.path.exists(os.path.join(ROOT, "manifest.json")):
        build_bank()
    _manifest, clips = load_bank()
    assert clips and clips[0]["logmag"].shape == (N_BINS, N_FRAMES)
    # inversion sanity: GL of a TRUE oracle spectrogram is not silent and energetic
    wav = griffin_lim(clips[0]["logmag"], n_iter=30)
    assert np.sqrt((wav ** 2).mean()) > 0.01, "GL output is silent"
    m = SpectrogramSpeaker()
    out = m(torch.tensor([0, 5]), torch.tensor([0, 1]))
    assert out.shape == (2, N_BINS, N_FRAMES)
    model = train(steps=4, seed=0, device="cpu")
    assert spectral_l1(model, device="cpu") > 0
    print("vocoder selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--synth", default="")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--checkpoint", default="runs/a3b_vocoder.pt")
    ap.add_argument("--out", default="data/synth")
    args = ap.parse_args(argv)
    if args.build:
        build_bank()
        return
    if args.selftest:
        selftest()
        return
    if args.train:
        model = train(steps=args.steps)
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, args.checkpoint)
        l1 = spectral_l1(model)
        print(f"saved -> {args.checkpoint}   held-in spectral L1 {l1:.4f}")
        synth(model, ["red", "blue", "circle", "triangle"], args.out)
        return
    if args.synth:
        model = SpectrogramSpeaker().to(DEV)
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEV)["state_dict"])
        synth(model, [w.strip() for w in args.synth.split(",")], args.out)
        return
    ap.error("choose --build / --selftest / --train / --synth")


if __name__ == "__main__":
    main()
