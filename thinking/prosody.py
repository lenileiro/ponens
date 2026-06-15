"""Human-voice SINGING and storytelling ANIMATION via a WORLD-vocoder prosody layer.

The TTS acoustic model (tts_fast) gives a *spoken* human voice. To reach mastery of speech we need
two more things on top of plain reading:

  1. SINGING in that same human voice -- carry a melody, not synthetic formant tones.
  2. ANIMATED prosody -- vary pitch / range / speed / energy so a story has characters and emotion,
     instead of a flat monotone.

One DSP backbone serves both. WORLD (Morise) decomposes any waveform into three independent streams:
F0 (pitch), spectral envelope (the FORMANTS = vocal-tract timbre = *who/what* the voice is) and
aperiodicity (breathiness). Manipulating F0 while KEEPING the spectral envelope re-pitches the voice
WITHOUT touching its timbre -- so it still sounds like the same human, just singing or emoting. This
is exactly why a naive resample ("chipmunk") fails and WORLD does not: it leaves the formants put.

  - sing():    re-pitch each segment of an utterance onto melody notes (+ vibrato) -> the voice sings.
  - animate(): scale pitch mean / pitch range / speed / energy -> expressive storytelling presets.

Operates on ANY human-voice waveform, so it composes with tts_fast output (read it, then sing/emote
it) or with real / cloned audio. CLI can synthesize from text via the trained tts_fast checkpoint.

  python -m thinking.prosody --selftest
  python -m thinking.prosody --from-text "twinkle twinkle little star" --sing --out data/synth/sing_human.wav
  python -m thinking.prosody --from-text "and then, the giant awoke." --animate --out-dir data/synth
"""
import argparse
import os

import numpy as np

SR = 22050           # matches realvoice / tts_fast
FRAME_PERIOD = 5.0   # ms per WORLD frame


# ---- note helpers -------------------------------------------------------------------------------
_SEMI = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def note_hz(name):
    """'a4' / 'c#5' / 'gb3' -> frequency in Hz (A4 = 440)."""
    name = name.strip().lower()
    step = _SEMI[name[0]]
    i = 1
    while i < len(name) and name[i] in "#b":
        step += 1 if name[i] == "#" else -1
        i += 1
    octave = int(name[i:])
    midi = 12 * (octave + 1) + step
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def parse_melody(s):
    """'c4:1 d4:1 e4:2' -> [(freq, beats), ...]. ':beats' optional (default 1)."""
    out = []
    for tok in s.split():
        if ":" in tok:
            n, b = tok.split(":")
            out.append((note_hz(n), float(b)))
        else:
            out.append((note_hz(tok), 1.0))
    return out


# ---- WORLD analysis / synthesis ----------------------------------------------------------------
def analyze(x, fs=SR, frame_period=FRAME_PERIOD):
    import pyworld as pw
    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, fs, frame_period=frame_period)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    return f0, sp, ap


def resynth(f0, sp, ap, fs=SR, frame_period=FRAME_PERIOD):
    import pyworld as pw
    y = pw.synthesize(np.ascontiguousarray(f0), np.ascontiguousarray(sp),
                      np.ascontiguousarray(ap), fs, frame_period)
    return y.astype(np.float32)


def _time_stretch(arr, n_out):
    """Linear-interpolate a (frames, ...) array along time to n_out frames."""
    n_in = arr.shape[0]
    if n_in == n_out or n_in < 2:
        return np.repeat(arr[:1], n_out, 0) if n_in < 2 else arr
    idx = np.linspace(0, n_in - 1, n_out)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, n_in - 1)
    frac = (idx - lo).reshape((-1,) + (1,) * (arr.ndim - 1))
    return arr[lo] * (1 - frac) + arr[hi] * frac


# ---- SINGING ------------------------------------------------------------------------------------
def sing(x, melody, fs=SR, bpm=120, vibrato_hz=5.0, vibrato_cents=25, frame_period=FRAME_PERIOD):
    """Re-pitch the utterance onto `melody` -> the SAME human voice, singing.

    Split the speech into len(melody) equal segments (syllable-ish), stretch each to its note's
    duration, and set F0 to the note frequency with a little vibrato. The spectral envelope (timbre)
    is preserved, so it stays the same human voice -- it just sings the tune.
    """
    melody = parse_melody(melody) if isinstance(melody, str) else melody
    f0, sp, ap = analyze(x, fs, frame_period)
    nF = f0.shape[0]
    spf = int(round(60.0 / bpm / (frame_period / 1000.0)))      # frames per beat
    seg_in = np.array_split(np.arange(nF), len(melody))         # speech frames per note
    out_f0, out_sp, out_ap = [], [], []
    for (hz, beats), idx in zip(melody, seg_in):
        n_out = max(2, int(round(beats * spf)))
        s0, s1 = idx[0], idx[-1] + 1
        out_sp.append(_time_stretch(sp[s0:s1], n_out))
        out_ap.append(_time_stretch(ap[s0:s1], n_out))
        # vibrato: small sinusoidal pitch wobble (musical, not robotic)
        tt = np.arange(n_out) * (frame_period / 1000.0)
        vib = 2.0 ** (vibrato_cents / 1200.0 * np.sin(2 * np.pi * vibrato_hz * tt))
        out_f0.append(hz * vib)
    return resynth(np.concatenate(out_f0), np.concatenate(out_sp, 0),
                   np.concatenate(out_ap, 0), fs, frame_period)


# ---- ANIMATION (storytelling prosody) ----------------------------------------------------------
PRESETS = {
    #            pitch  range  speed  energy   (pitch=mean shift, range=expressiveness)
    "narrator":  (1.00, 1.15, 0.98, 1.00),   # calm, lightly expressive default
    "excited":   (1.12, 1.55, 1.12, 1.20),   # higher, animated, quicker
    "sad":       (0.90, 0.65, 0.86, 0.92),   # low, flat, slow
    "suspense":  (0.95, 0.60, 0.84, 0.88),   # hushed, even, deliberate
    "giant":     (0.68, 1.20, 0.90, 1.15),   # deep booming character voice
    "fairy":     (1.42, 1.45, 1.10, 1.00),   # tiny high character voice
    "question":  (1.00, 1.30, 1.00, 1.00),   # rising, inquisitive
}


def animate(x, pitch=1.0, range_=1.0, speed=1.0, energy=1.0, fs=SR, frame_period=FRAME_PERIOD):
    """Expressive prosody: shift pitch MEAN, scale pitch RANGE, time-scale SPEED, scale ENERGY.

    Timbre (spectral envelope) is preserved, so emotion changes but identity does not."""
    f0, sp, ap = analyze(x, fs, frame_period)
    if speed != 1.0:
        n_out = max(2, int(round(f0.shape[0] / speed)))
        f0 = _time_stretch(f0, n_out); sp = _time_stretch(sp, n_out); ap = _time_stretch(ap, n_out)
    voiced = f0 > 0
    if voiced.any():
        mean = float(np.exp(np.log(f0[voiced]).mean()))         # geometric mean pitch
        f0 = np.where(voiced, np.clip(mean * pitch + (f0 - mean) * range_, 50.0, 1000.0), 0.0)
    sp = sp * (energy ** 2)                                      # spectral power = loudness/emphasis
    return resynth(f0, sp, ap, fs, frame_period)


def animate_preset(x, style, fs=SR):
    p, r, sp_, e = PRESETS[style]
    return animate(x, pitch=p, range_=r, speed=sp_, energy=e, fs=fs)


# ---- synthesis-from-text bridge (uses the trained tts_fast acoustic model) ----------------------
def speak(text, ckpt="runs/tts_fast.pt", fs=SR):
    """text -> human-voice waveform via the trained FastTTS + realvoice vocoder."""
    import torch
    from device import get_device
    from .tts_fast import FastTTS, infer
    from .tts import encode_text
    dev = get_device()
    model = FastTTS().to(dev)
    ck = torch.load(ckpt, map_location=dev)
    model.load_state_dict(ck["state_dict"]); model.frames_per_char = ck.get("frames_per_char", 8)
    model.eval()
    ids = torch.tensor([encode_text(text)], device=dev)
    spec = infer(model, ids, dev)[0].cpu().numpy()
    from .realvoice import Vocos, griffin_lim
    if os.path.exists("runs/realvoice.pt"):
        voc = Vocos(d=512).to(dev)
        voc.load_state_dict(torch.load("runs/realvoice.pt", map_location=dev)["state_dict"]); voc.eval()
        with torch.no_grad():
            wav = voc(torch.tensor(spec[None], device=dev))[0].cpu().numpy()
    else:
        wav = griffin_lim(spec)
    return wav.astype(np.float32)


def _write(path, wav, fs=SR):
    import soundfile as sf
    y = wav / (np.abs(wav).max() + 1e-8) * 0.95
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, y.astype(np.float32), fs)
    print(f"  wrote {path} ({len(y)/fs:.2f}s)")


def _read(path):
    import soundfile as sf
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    if fs != SR:
        import librosa
        x = librosa.resample(x.astype(np.float32), orig_sr=fs, target_sr=SR)
    return x.astype(np.float32)


def selftest():
    # synthetic voiced tone -> WORLD round-trip + sing + animate keep length/finite sanity
    fs = SR
    t = np.arange(int(0.6 * fs)) / fs
    x = (0.5 * np.sin(2 * np.pi * 140 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    f0, sp, ap = analyze(x, fs)
    assert f0.ndim == 1 and sp.shape[0] == f0.shape[0]
    y = resynth(f0, sp, ap, fs); assert np.isfinite(y).all() and len(y) > 0
    s = sing(x, "c4:1 e4:1 g4:2", fs=fs, bpm=160)
    assert np.isfinite(s).all() and len(s) > fs * 0.5, len(s)
    a = animate_preset(x, "excited", fs=fs); assert np.isfinite(a).all() and len(a) > 0
    assert abs(note_hz("a4") - 440.0) < 1e-6 and abs(note_hz("a5") - 880.0) < 1e-6
    fast = animate(x, speed=1.5, fs=fs); assert len(fast) < len(x)        # speed>1 shortens
    print("prosody selftest OK")


# twinkle twinkle (C major), one note per syllable
TWINKLE = ("c4:1 c4:1 g4:1 g4:1 a4:1 a4:1 g4:2 f4:1 f4:1 e4:1 e4:1 d4:1 d4:1 c4:2")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--in", dest="inp", default="")
    ap.add_argument("--from-text", dest="from_text", default="")
    ap.add_argument("--ckpt", default="runs/tts_fast.pt")
    ap.add_argument("--sing", action="store_true")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--melody", default=TWINKLE)
    ap.add_argument("--bpm", type=int, default=120)
    ap.add_argument("--out", default="data/synth/prosody_out.wav")
    ap.add_argument("--out-dir", dest="out_dir", default="data/synth")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return
    x = _read(args.inp) if args.inp else speak(args.from_text, args.ckpt)
    if not (args.sing or args.animate):
        _write(args.out, x); return
    if args.sing:
        _write(args.out if args.out != "data/synth/prosody_out.wav" else
               os.path.join(args.out_dir, "sing_human.wav"), sing(x, args.melody, bpm=args.bpm))
    if args.animate:
        for style in PRESETS:
            _write(os.path.join(args.out_dir, f"story_{style}.wav"), animate_preset(x, style))


if __name__ == "__main__":
    main()
