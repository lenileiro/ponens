"""LISTENING (real continuous speech): the model HEARS human speech and understands it -- our own
speech recognizer, trained from scratch (NOT an external Whisper call).

The speaking half is done (tts_ar: text -> intelligible audio). Mastery needs the OTHER direction on
the SAME real data: LJSpeech audio -> text. (The older thinking/listen.py is a narrow word x voice
task on synthetic `say`/`espeak` clips; this is continuous sentence ASR on real human audio, the
true counterpart to tts_ar.)

Architecture = the standard small-data CTC recipe: a conv front-end that subsamples the spectrogram
4x in time, a BIDIRECTIONAL Transformer encoder (listens to the whole utterance, no causal mask), a
linear head to character logits, and a CTC loss that learns the audio<->text alignment with no
frame-level labels. SpecAugment (time/freq masking) regularizes the single speaker.

"Understanding pronunciation" falls out of this: to transcribe, the model must learn how acoustic
patterns map to letters -- the inverse of the speaking model. Scored by the model's OWN word/char
error rate on HELD-OUT audio it never heard.

  python -m thinking.asr --selftest
  python -m thinking.asr --train --steps 40000 --out runs/asr.json   (GPU; needs LJSpeech via tts --fetch)
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .realvoice import N_BINS
from .tts import ROOT, CHARS, C2I, encode_text, load, spec_of

DEV = get_device()
# LJSpeech sentences run ~6-10s; TTS's MAX_T=360 (~4.2s) truncated audio while keeping the full
# transcript -> CTC infeasible (frames < chars) -> all-blank collapse. Use the FULL clip for ASR.
ASR_MAX_T = 1000                             # ~11.6s @ hop256/22050; /4 subsample -> 250 frames >> chars
VOCAB = len(CHARS) + 1                       # index 0 = CTC blank; chars are 1..len(CHARS)
I2C = {i + 1: c for i, c in enumerate(CHARS)}


def decode_ids(ids):
    """Greedy CTC collapse: drop repeats, drop blanks(0) -> text."""
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != 0:
            out.append(I2C.get(i, ""))
        prev = i
    return "".join(out)


def spec_augment(mel, rng, n_t=2, n_f=2, max_t=30, max_f=40):
    """Mask random time bands + frequency bands (regularizes single-speaker data)."""
    B, F_, T = mel.shape
    mel = mel.clone()
    for b in range(B):
        for _ in range(n_t):
            w = int(rng.integers(0, max_t)); s = int(rng.integers(0, max(1, T - w)))
            mel[b, :, s:s + w] = 0.0
        for _ in range(n_f):
            w = int(rng.integers(0, max_f)); s = int(rng.integers(0, max(1, F_ - w)))
            mel[b, s:s + w, :] = 0.0
    return mel


class ASR(nn.Module):
    def __init__(self, n_bins=N_BINS, d=256, layers=6, heads=4, vocab=VOCAB):
        super().__init__()
        self.cfg = {"d": d, "layers": layers, "heads": heads}
        self.sub = nn.Sequential(                            # conv front-end, subsample 4x time+freq
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU())
        f4 = ((n_bins + 1) // 2 + 1) // 2                    # freq dim after two stride-2 convs
        self.proj = nn.Linear(32 * f4, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu",
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, vocab)

    def forward(self, mel):
        x = self.sub(mel[:, None])                           # (B,32,F4,T4)
        B, C, Fq, T4 = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T4, C * Fq)
        x = self.enc(self.proj(x))                           # bidirectional whole-utterance listening
        return self.head(x).log_softmax(-1), T4              # (B,T4,vocab)


def build_cache(data):
    specs = [spec_of(w, "cpu")[:, :ASR_MAX_T].contiguous() for t, w in data]
    txts = [encode_text(t) for t, w in data]
    return specs, txts


def _batch(cache, rng, batch, device, augment=True):
    specs, txts = cache
    idx = [int(rng.integers(len(specs))) for _ in range(batch)]
    sp = [specs[j] for j in idx]; tx = [txts[j] for j in idx]
    Tt = max(s.shape[1] for s in sp)
    mel = torch.zeros(batch, N_BINS, Tt, device=device)
    mel_len = torch.zeros(batch, dtype=torch.long)
    for b, s in enumerate(sp):
        mel[b, :, :s.shape[1]] = s.to(device); mel_len[b] = s.shape[1]
    if augment:
        mel = spec_augment(mel, rng)
    targets = torch.tensor([c for t in tx for c in t], dtype=torch.long, device=device)
    tgt_len = torch.tensor([len(t) for t in tx], dtype=torch.long)
    return mel, mel_len, targets, tgt_len


def train(steps=40000, seed=0, device=DEV, batch=32, lr=3e-4, ckpt_path=None, d=256, layers=6, heads=4):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = load(); split = int(len(data) * 0.95); td = data[:split]
    print(f"caching {len(td)} utterances ...", flush=True)
    cache = build_cache(td)
    model = ASR(d=d, layers=layers, heads=heads).to(device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        mel, mel_len, targets, tgt_len = _batch(cache, rng, batch, device)
        logp, T4 = model(mel)
        in_len = (mel_len.float() * T4 / mel.shape[2]).long().clamp(1, T4)
        loss = F.ctc_loss(logp.transpose(0, 1), targets, in_len, tgt_len, blank=0, zero_infinity=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if st % max(1, steps // 20) == 0 or st == steps:
            print(f"  asr {st}/{steps} ctc {loss.item():.3f}", flush=True)
        if st % max(1, steps // 5) == 0 and ckpt_path:
            torch.save({"state_dict": model.state_dict(), "config": model.cfg}, ckpt_path)
    return model, data


def _edit(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    d[:, 0] = np.arange(len(ref) + 1); d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[len(ref), len(hyp)] / len(ref)


def evaluate(model, data, device=DEV, n=200, show=0):
    ev = data[int(len(data) * 0.95):]
    model.eval(); wers, cers = [], []
    with torch.no_grad():
        for k in range(min(n, len(ev))):
            txt, wav = ev[k]
            ref = "".join(c for c in txt.lower() if c in C2I)
            logp, _ = model(spec_of(wav, device)[:, :ASR_MAX_T][None])
            hyp = decode_ids(logp[0].argmax(-1).cpu().numpy())
            wers.append(_edit(ref.split(), hyp.split())); cers.append(_edit(list(ref), list(hyp)))
            if k < show:
                print(f"  ref: {ref[:64]!r}\n  hyp: {hyp[:64]!r}", flush=True)
    return {"wer": float(np.mean(wers)), "cer": float(np.mean(cers)), "n": len(wers)}


def run(steps=40000, seed=0, device=DEV, ckpt_path="runs/asr.pt", batch=32, d=256, layers=6, heads=4):
    model, data = train(steps=steps, seed=seed, device=device, ckpt_path=ckpt_path,
                        batch=batch, d=d, layers=layers, heads=heads)
    torch.save({"state_dict": model.state_dict(), "config": model.cfg}, ckpt_path)
    ev = evaluate(model, data, device=device, show=6)
    report = {"experiment": "asr_ctc_ljspeech", **model.cfg, "steps": steps, **ev,
              "understands": ev["wer"] < 0.3}
    print(f"\nLISTENING (own ASR): WER {ev['wer']:.3f}  CER {ev['cer']:.3f} over {ev['n']} held-out", flush=True)
    return report, model


def build_from_ckpt(path, device=DEV):
    ck = torch.load(path, map_location=device); cfg = ck.get("config", {})
    m = ASR(d=cfg.get("d", 256), layers=cfg.get("layers", 6), heads=cfg.get("heads", 4)).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def transcribe(model, wav, device=DEV):
    model.eval()
    with torch.no_grad():
        logp, _ = model(spec_of(wav, device)[:, :ASR_MAX_T][None])
    return decode_ids(logp[0].argmax(-1).cpu().numpy())


def selftest():
    torch.manual_seed(0)
    m = ASR(d=64, layers=2, heads=4)
    logp, T4 = m(torch.randn(2, N_BINS, 120))
    assert logp.shape[0] == 2 and logp.shape[2] == VOCAB and T4 == logp.shape[1], (logp.shape, T4)
    assert decode_ids([0, 5, 5, 0, 8, 8, 8, 2]) == I2C[5] + I2C[8] + I2C[2]
    assert abs(_edit("a b c".split(), "a x c".split()) - 1 / 3) < 1e-9 and _edit(list("abc"), list("abc")) == 0.0
    if os.path.exists(os.path.join(ROOT, "manifest.json")):
        data = load(); assert data
        m2, d = train(steps=2, seed=0, device="cpu", batch=2)
        ev = evaluate(m2, d, device="cpu", n=2); assert "wer" in ev
    print("asr selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--out", default="runs/asr.json")
    ap.add_argument("--checkpoint", default="runs/asr.pt")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return
    if args.train:
        report, _ = run(steps=args.steps, batch=args.batch, d=args.dim, layers=args.layers,
                        heads=args.heads, ckpt_path=args.checkpoint)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"saved -> {args.out}")
        return
    ap.error("choose --selftest / --train")


if __name__ == "__main__":
    main()
