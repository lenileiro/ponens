"""Synthesize playable .wav from the A-3 speaker: codes -> spectrogram -> Griffin-Lim -> waveform.

Produces, for each requested word:
  <word>_oracle.wav  -- the real `say` clip the listener was trained on (clear reference)
  <word>_model.wav   -- the model's OWN spoken word, decoded from its acoustic tokens and
                        inverted to audio (buzzy: a tiny VQ + Griffin-Lim, but it is the
                        model speaking, and the frozen listener transcribes it)

  python -m thinking.synth --words red,blue,circle --out data/synth
"""
import argparse
import os
import wave

import numpy as np
import torch

from device import get_device
from .audio import spectrogram
from .listen import WORDS, load_bank, _split, train_arm as train_listener
from .speak import (FrameVQ, Speaker, train_codec, codes_of, spec_from_codes, N_FRAMES, N_BINS)

DEV = get_device()
SR = 8000
N_FFT = 128
HOP = 64


def griffin_lim(log_mag_cols, n_iter=60):
    """Invert a (T, N_BINS) log1p-magnitude STFT to a waveform (phase via Griffin-Lim)."""
    mag = np.expm1(np.clip(log_mag_cols, 0, None)).T        # (N_BINS, T) linear magnitude
    win = np.hanning(N_FFT).astype(np.float32)
    T = mag.shape[1]
    length = HOP * (T - 1) + N_FFT
    rng = np.random.default_rng(0)
    phase = np.exp(2j * np.pi * rng.random(mag.shape))
    for _ in range(n_iter):
        spec = mag * phase
        wav = np.zeros(length, dtype=np.float64)
        wsum = np.zeros(length, dtype=np.float64) + 1e-8
        for t in range(T):
            frame = np.fft.irfft(spec[:, t], n=N_FFT) * win
            wav[t * HOP:t * HOP + N_FFT] += frame
            wsum[t * HOP:t * HOP + N_FFT] += win ** 2
        wav = wav / wsum
        # re-estimate phase from the reconstructed waveform
        new = np.zeros_like(spec)
        for t in range(T):
            seg = wav[t * HOP:t * HOP + N_FFT] * win
            new[:, t] = np.fft.rfft(seg, n=N_FFT)
        phase = np.exp(1j * np.angle(new))
    return wav.astype(np.float32)


def write_wav(path, wave_f32, sr=SR):
    x = wave_f32 / (np.abs(wave_f32).max() + 1e-8) * 0.95
    pcm = (x * 32767).astype(np.int16)
    w = wave.open(path, "w")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(pcm.tobytes())
    w.close()


def build_stack(steps=1200, seed=0, device=DEV):
    """Train listener + codec + speaker (the A-3 stack) and return them."""
    import torch.nn.functional as F
    _manifest, clips = load_bank()
    train_clips, _ = _split(clips)
    listener = train_listener("swap", train_clips, steps=600, seed=seed, device=device)
    for p in listener.parameters():
        p.requires_grad_(False)
    vq = train_codec(clips, steps=600, seed=seed, device=device)
    by_word = {}
    for c in clips:
        by_word.setdefault(c["word"], []).append(c)
    rng = np.random.default_rng(seed)
    speaker = Speaker().to(device)
    opt = torch.optim.AdamW(list(speaker.parameters()) + list(vq.dec.parameters())
                            + [vq.codebook.weight], lr=2e-3)
    for st in range(1, steps + 1):
        speaker.train()
        ws = [int(rng.integers(len(WORDS))) for _ in range(32)]
        tgt = torch.stack([codes_of(vq, by_word[w][int(rng.integers(len(by_word[w])))]["wave"],
                                    device) for w in ws])
        words = torch.tensor(ws, device=device)
        prev = torch.cat([torch.full((len(ws), 1), speaker.bos, device=device), tgt[:, :-1]], 1)
        logits = speaker(words, prev)
        ce_codes = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        temp = max(0.3, 2.0 * (1.0 - st / steps))
        spec = speaker.soft_spec(words, vq, temp=temp)[:, None]
        ce_listen = F.cross_entropy(listener(spec)["word"], words)
        loss = ce_codes + 2.0 * ce_listen
        opt.zero_grad()
        loss.backward()
        opt.step()
    return listener, vq, speaker, by_word


def synth(words, out_dir, steps=1200, device=DEV):
    os.makedirs(out_dir, exist_ok=True)
    listener, vq, speaker, by_word = build_stack(steps=steps, device=device)
    rng = np.random.default_rng(0)
    results = []
    for word in words:
        if word not in WORDS:
            print(f"  (skip {word!r}: not in vocab {WORDS})")
            continue
        wi = WORDS.index(word)
        # oracle reference clip
        oracle = by_word[wi][0]["wave"]
        op = os.path.join(out_dir, f"{word}_oracle.wav")
        write_wav(op, oracle)
        # the model's own speech
        codes = speaker.generate(torch.tensor([wi], device=device), device)[0]
        spec = spec_from_codes(vq, codes, device)            # (1, N_BINS, T)
        wav = griffin_lim(spec[0].T.detach().cpu().numpy())
        mp = os.path.join(out_dir, f"{word}_model.wav")
        write_wav(mp, wav)
        # what the frozen listener hears the model say
        with torch.no_grad():
            heard = listener(spec[None])["word"].argmax(-1).item()
        ok = "OK" if heard == wi else f"heard '{WORDS[heard]}'"
        print(f"  {word:9} -> {op}  |  {mp}  [{ok}]")
        results.append((word, ok))
    n_ok = sum(1 for _, o in results if o == "OK")
    print(f"\nmodel said {n_ok}/{len(results)} recognizably (frozen listener); "
          f"play *_model.wav (the model) vs *_oracle.wav (the reference it learned from)")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="red,blue,circle,triangle")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", default="data/synth")
    args = ap.parse_args(argv)
    synth([w.strip() for w in args.words.split(",")], args.out, steps=args.steps)


if __name__ == "__main__":
    main()
