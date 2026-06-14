"""A-6 SING: controllable singing -- the capstone of voice mastery.

Singing forces the model to FACTOR and RECOMBINE everything: LYRICS (which vowel/phoneme), MELODY
(pitch per note), RHYTHM (note durations), and carry them jointly. Mastery = sing a HELD-OUT
melody with HELD-OUT lyrics IN TUNE: produce audio whose per-note pitch matches the target notes
and whose per-note vowel is correct -- factors never trained in that combination.

This is the FER question at its hardest: if pitch and vowel are FACTORED (UFR), the model freely
recombines any note with any vowel; if entangled (FER), held-out (vowel,note) pairs fail.

Oracle: a note = a tone at the note's fundamental, coloured by the vowel's timbre, held for the
note's duration -> spectrogram (the fundamental is recoverable from the spectrum, so pitch is
objectively checkable). Model: (vowel, note, duration) sequence -> sung spectrogram.

Verification (objective, no human ear):
  pitch_acc : per note, the synthesized audio's dominant frequency maps to the right note (in tune)
  vowel_acc : a frozen vowel classifier reads the right vowel per note (lyrics survive the melody)
  on HELD-OUT (vowel x note) combinations.

  python -m thinking.sing --selftest
  python -m thinking.sing --steps 8000 --out runs/a6_sing.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .vocoder import write_wav

DEV = get_device()
SR = 8000
# HIGH frequency resolution is mandatory for singing: adjacent notes are ~19 Hz apart, so the
# FFT bins must be finer than that. n_fft=1024 @ 8kHz -> 7.8 Hz bins (resolves a semitone).
N_FFT = 1024
HOP = 128
N_BINS = N_FFT // 2 + 1                                    # 513
NOTE_DUR = 0.40                                            # seconds per note
NOTE_FRAMES = 12                                           # spectrogram cols kept per note
_WIN = np.hanning(N_FFT).astype(np.float32)
# C major scale C4..C5 (well-defined fundamentals); 8 notes
NOTE_HZ = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25]
NOTES = [f"n{int(round(h))}" for h in NOTE_HZ]
VOWELS = ("ah", "ee", "oo", "oh")                          # 4 vowels = distinct formant patterns
# each vowel = a pair of formant frequencies (Hz), the classic vowel-identity cue
VOWEL_FORMANTS = {"ah": (730, 1090), "ee": (270, 2290), "oo": (300, 870), "oh": (570, 840)}
SONG_LEN = 6                                               # notes per song
HOLDOUT_CELLS = {(v, n) for v in range(len(VOWELS)) for n in range(len(NOTES))
                 if (v * 3 + n) % 5 == 0}                  # ~1/5 (vowel,note) combos held out


def _sung_wave(vowel_idx, note_idx):
    """A sung note: a harmonic tone at the note's fundamental, shaped by the vowel's formants."""
    f0 = NOTE_HZ[note_idx]
    t = np.arange(int(SR * NOTE_DUR), dtype=np.float32) / SR
    wave = np.zeros_like(t)
    f1, f2 = VOWEL_FORMANTS[VOWELS[vowel_idx]]
    for k in range(1, 25):                                  # harmonic stack to ~12kHz
        fk = f0 * k
        if fk > SR / 2:
            break
        # formant envelope: boost harmonics near F1 and F2 (gives the vowel its identity)
        amp = (np.exp(-((fk - f1) / 120) ** 2) + np.exp(-((fk - f2) / 160) ** 2) + 0.05) / k
        wave += amp * np.sin(2 * np.pi * fk * t)
    return wave / (np.abs(wave).max() + 1e-8)


def _stft_logmag(wave_f):
    cols = [np.abs(np.fft.rfft(wave_f[i:i + N_FFT] * _WIN))
            for i in range(0, len(wave_f) - N_FFT + 1, HOP)]
    return np.log1p(np.stack(cols, 1).astype(np.float32))   # (513, T)


def render_note(vowel_idx, note_idx, rng=None):
    s = _stft_logmag(_sung_wave(vowel_idx, note_idx))
    if s.shape[1] >= NOTE_FRAMES:
        return s[:, :NOTE_FRAMES].astype(np.float32)
    return np.pad(s, ((0, 0), (0, NOTE_FRAMES - s.shape[1]))).astype(np.float32)


def render_song(vowels, notes, rng=None):
    return np.concatenate([render_note(v, n, rng) for v, n in zip(vowels, notes)], axis=1)


def sample_song(rng, avoid_holdout=True):
    while True:
        vowels = [int(rng.integers(len(VOWELS))) for _ in range(SONG_LEN)]
        notes = [int(rng.integers(len(NOTES))) for _ in range(SONG_LEN)]
        if not avoid_holdout or not any((v, n) in HOLDOUT_CELLS for v, n in zip(vowels, notes)):
            return vowels, notes


def sample_holdout_song(rng):
    """A song where at least half the notes use HELD-OUT (vowel,note) combos."""
    cells = list(HOLDOUT_CELLS)
    vowels, notes = [], []
    for i in range(SONG_LEN):
        if i % 2 == 0:
            v, n = cells[int(rng.integers(len(cells)))]
        else:
            v, n = int(rng.integers(len(VOWELS))), int(rng.integers(len(NOTES)))
        vowels.append(v)
        notes.append(n)
    return vowels, notes


def _griffin_lim(logmag, n_iter=80):
    mag = np.expm1(np.clip(logmag, 0, None))
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


_BIN_HZ = np.fft.rfftfreq(N_FFT, 1.0 / SR)                 # 513 bins, 7.8 Hz apart


def dominant_note(spec_block):
    """Recover the note from a sung spectrogram block. The fundamental is the lowest strong
    harmonic, so detect pitch by the best-fitting f0 over its harmonic comb (robust to a loud
    formant landing on a higher harmonic)."""
    mag = np.expm1(np.clip(spec_block, 0, None)).mean(1)   # (513,)
    best, best_score = 0, -1.0
    for ni, f0 in enumerate(NOTE_HZ):
        bins = [int(round(f0 * k / (SR / N_FFT))) for k in range(1, 9)]
        score = sum(mag[b] for b in bins if b < len(mag))
        if score > best_score:
            best_score, best = score, ni
    return best


class SingTTS(nn.Module):
    """(vowel, note, duration) sequence -> sung spectrogram. Note + vowel are separate embeddings
    (factored conditioning); learned frame queries cross-attend over the note sequence."""

    def __init__(self, d=192, heads=6, layers=3, frames_per_note=NOTE_FRAMES):
        super().__init__()
        self.fpn = frames_per_note
        self.vowel = nn.Embedding(len(VOWELS), d)
        self.note = nn.Embedding(len(NOTES), d)
        self.npos = nn.Embedding(SONG_LEN, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.frame_q = nn.Parameter(torch.randn(SONG_LEN * frames_per_note, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                         batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, N_BINS))

    def forward(self, vowels, notes):
        B, T = vowels.shape
        pos = torch.arange(T, device=vowels.device)
        score = self.vowel(vowels) + self.note(notes) + self.npos(pos)[None]
        mem = self.encoder(score)
        q = self.frame_q[None].expand(B, -1, -1)
        dec = self.decoder(q, mem)
        return self.head(dec).transpose(1, 2)              # (B, N_BINS, T*fpn)


def _vowel_classifier(device, steps=500, seed=0):
    """Frozen per-note vowel classifier (verifies lyrics survive the melody)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for v in range(len(VOWELS)):
        for n in range(len(NOTES)):
            for _ in range(6):
                X.append(render_note(v, n, rng))
                y.append(v)
    X = torch.tensor(np.stack(X))[:, None].to(device)
    y = torch.tensor(y, device=device)
    net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)),
                        nn.Flatten(), nn.Linear(32 * 16, 64), nn.GELU(),
                        nn.Linear(64, len(VOWELS))).to(device)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(steps):
        i = torch.tensor(rng.integers(0, len(X), 48))
        loss = F.cross_entropy(net(X[i] + 0.01 * torch.randn_like(X[i])), y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def train(steps=8000, seed=0, device=DEV, batch=32, lr=3e-4, d=192, vowel_w=0.0,
          vowel_clf=None):
    """vowel_w>0 adds VERIFIER-IN-THE-LOOP supervision: a frozen vowel classifier reads the
    model's OWN per-note output and must recover the right vowel -- 'sing so the lyrics are
    recognizable', the A-3 listener-in-loop move that forces vowel to stay factored from pitch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SingTTS(d=d).to(device)
    if vowel_w > 0 and vowel_clf is None:
        vowel_clf = _vowel_classifier(device, seed=seed)
    if vowel_clf is not None:
        for p in vowel_clf.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        model.train()
        songs = [sample_song(rng, avoid_holdout=True) for _ in range(batch)]
        vowels = torch.tensor([s[0] for s in songs], device=device)
        notes = torch.tensor([s[1] for s in songs], device=device)
        tgt = torch.tensor(np.stack([render_song(v, n, rng) for v, n in songs]),
                           dtype=torch.float32, device=device)
        pred = model(vowels, notes)
        loss = F.l1_loss(pred, tgt)
        if vowel_w > 0:
            B = pred.shape[0]
            blocks = pred.reshape(B, N_BINS, SONG_LEN, NOTE_FRAMES).permute(0, 2, 1, 3)
            blocks = blocks.reshape(B * SONG_LEN, 1, N_BINS, NOTE_FRAMES)
            vlogits = vowel_clf(blocks)
            loss = loss + vowel_w * F.cross_entropy(vlogits, vowels.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % max(1, steps // 8) == 0 or st == steps:
            print(f"  a6 {st}/{steps} L1 {loss.item():.4f}", flush=True)
    return model


def evaluate(model, device=DEV, n_songs=80, seed=1, vowel_clf=None):
    model.eval()
    if vowel_clf is None:
        vowel_clf = _vowel_classifier(device)
    rng = np.random.default_rng(seed)
    pitch_hit = vowel_hit = held_pitch_hit = held_vowel_hit = note_total = held_total = 0
    with torch.no_grad():
        for _ in range(n_songs):
            vowels, notes = sample_holdout_song(rng)
            vt = torch.tensor([vowels], device=device)
            nt = torch.tensor([notes], device=device)
            spec = model(vt, nt)[0].cpu().numpy()          # (65, SONG_LEN*fpn)
            for i, (v, n) in enumerate(zip(vowels, notes)):
                block = spec[:, i * NOTE_FRAMES:(i + 1) * NOTE_FRAMES]
                pitch_ok = dominant_note(block) == n
                bt = torch.tensor(block)[None, None].to(device)
                vow_ok = int(vowel_clf(bt).argmax(-1).item()) == v
                pitch_hit += pitch_ok
                vowel_hit += vow_ok
                note_total += 1
                if (v, n) in HOLDOUT_CELLS:
                    held_pitch_hit += pitch_ok
                    held_vowel_hit += vow_ok
                    held_total += 1
    return {"pitch_acc": pitch_hit / note_total, "vowel_acc": vowel_hit / note_total,
            "holdout_pitch_acc": held_pitch_hit / max(1, held_total),
            "holdout_vowel_acc": held_vowel_hit / max(1, held_total),
            "n_notes": note_total, "n_holdout_notes": held_total}


def synth(model, songs, out_dir, device=DEV):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for name, (vowels, notes) in songs.items():
            spec = model(torch.tensor([vowels], device=device),
                         torch.tensor([notes], device=device))[0].cpu().numpy()
            write_wav(os.path.join(out_dir, f"sing_{name}.wav"), _griffin_lim(spec, n_iter=60),
                      sr=SR)
            print(f"  {name}: notes {[NOTES[n] for n in notes]} -> {out_dir}/sing_{name}.wav")


def run(steps=8000, seed=0, device=DEV, vowel_w=0.5):
    vclf = _vowel_classifier(device, seed=seed) if vowel_w > 0 else None
    model = train(steps=steps, seed=seed, device=device, vowel_w=vowel_w, vowel_clf=vclf)
    ev = evaluate(model, device=device, vowel_clf=vclf)
    report = {"experiment": "a6_sing", "steps": steps, "n_notes_scale": len(NOTES),
              "n_vowels": len(VOWELS), "pitch_chance": 1 / len(NOTES),
              "vowel_chance": 1 / len(VOWELS), "vowel_w": vowel_w, **ev,
              "gate": ev["holdout_pitch_acc"] > 0.6 and ev["holdout_vowel_acc"] > 0.6}
    print(f"\nALL notes : pitch {ev['pitch_acc']:.2f}  vowel {ev['vowel_acc']:.2f}")
    print(f"HELD-OUT (vowel x note) combos: pitch {ev['holdout_pitch_acc']:.2f} "
          f"(chance {1/len(NOTES):.2f})  vowel {ev['holdout_vowel_acc']:.2f} "
          f"(chance {1/len(VOWELS):.2f})")
    print(f"gate (held-out pitch & vowel > 0.6): {report['gate']}", flush=True)
    return report, model


def selftest():
    s = render_note(0, 3)
    assert s.shape == (N_BINS, NOTE_FRAMES)
    # the oracle is in tune: a rendered note's dominant frequency recovers the note
    hits = sum(dominant_note(render_note(0, n)) == n for n in range(len(NOTES)))
    assert hits >= len(NOTES) - 1, f"oracle not in tune: {hits}/{len(NOTES)}"
    vowels, notes = sample_song(np.random.default_rng(0))
    assert len(vowels) == SONG_LEN
    song = render_song(vowels, notes)
    assert song.shape == (N_BINS, SONG_LEN * NOTE_FRAMES)
    m = SingTTS(d=64, heads=4, layers=1)
    out = m(torch.tensor([vowels]), torch.tensor([notes]))
    assert out.shape == (1, N_BINS, SONG_LEN * NOTE_FRAMES)
    model = train(steps=3, seed=0, device="cpu")
    ev = evaluate(model, device="cpu", n_songs=6)
    assert "holdout_pitch_acc" in ev
    print("sing selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/a6_sing.json")
    ap.add_argument("--checkpoint", default="runs/a6_sing.pt")
    ap.add_argument("--synth-out", default="data/synth", dest="synth_out")
    ap.add_argument("--vowel-w", type=float, default=0.5, dest="vowel_w")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    report, model = run(steps=args.steps, seed=args.seed, vowel_w=args.vowel_w)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.checkpoint)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    # sing a rising scale and a held-out tune
    rng = np.random.default_rng(3)
    synth(model, {"scale": ([0, 0, 0, 0, 0, 0], list(range(SONG_LEN))),
                  "tune": sample_holdout_song(rng)}, args.synth_out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
