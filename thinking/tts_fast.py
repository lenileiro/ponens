"""FastSpeech-style NON-autoregressive TTS: decode all frames in PARALLEL (no per-frame loop).

The AR Tacotron decoder ran a Python loop over ~300 frames per step (~3 steps/sec on an H100) and
timed out before alignment locked. FastSpeech's fix: emit every frame at once. Here, learned
frame-position queries cross-attend over the character states in PARALLEL (one batched attention,
not a T-step loop -> ~100x faster), with a GUIDED-ATTENTION loss giving the monotonic text->audio
alignment. Inference length comes from a per-character DURATION predictor (frames/char), trained
against the durations the model's own attention implies (self-extracted, no external aligner).

Same spectrogram target (513-bin log-mag) and the same realvoice vocoder for waveform, so this is a
drop-in faster replacement for the acoustic model. Trains to convergence within a GPU session.

  python -m thinking.tts_fast --selftest
  python -m thinking.tts_fast --train --steps 30000 --out runs/tts_fast.json   (GPU; fast)
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .realvoice import SR, N_BINS
from .tts import (ROOT, CHARS, C2I, MAX_T, MAX_CH, encode_text, load, spec_of, CharEncoder,
                  guided_attention_loss)

DEV = get_device()


class FastTTS(nn.Module):
    """char states -> (parallel cross-attention with frame-position queries) -> spectrogram.
    Plus a duration predictor (frames per char) for inference-time length."""

    def __init__(self, d=256, n_bins=N_BINS, layers=4, heads=4, max_frames=MAX_T):
        super().__init__()
        self.d = d
        self.enc = CharEncoder(d)
        self.frame_pos = nn.Parameter(torch.randn(max_frames, d) * 0.02)
        self.q_in = nn.Linear(d, d)
        self.kproj = nn.Linear(d, d)
        self.vproj = nn.Linear(d, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True)
        self.refine = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_bins))
        self.dur = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))   # log-duration/char

    def forward(self, ids, T):
        """ids (B, L); T = target frame count -> spectrogram (B, n_bins, T) + alignment (B,T,L)."""
        h = self.enc(ids)                                  # (B, L, d) char states
        mask = ids.eq(0)
        q = self.q_in(self.frame_pos[:T])[None].expand(h.shape[0], -1, -1)   # (B, T, d) frame queries
        k = self.kproj(h); v = self.vproj(h)
        score = (q @ k.transpose(1, 2)) / (self.d ** 0.5)  # (B, T, L) PARALLEL cross-attention
        score = score.masked_fill(mask[:, None, :], -1e9)
        align = F.softmax(score, -1)
        ctx = align @ v                                    # (B, T, d) aligned frame features
        dec = self.refine(ctx)                             # parallel transformer over frames
        return self.out(dec).transpose(1, 2), align, h     # (B, n_bins, T), align, char states

    def durations(self, h):
        return self.dur(h).squeeze(-1)                     # (B, L) predicted log-frames per char


def _batch(data, rng, batch, device):
    items = [data[int(rng.integers(len(data)))] for _ in range(batch)]
    specs = [spec_of(w, device)[:, :MAX_T] for _, w in items]
    ids = [encode_text(t) for t, _ in items]
    Lc = max(len(i) for i in ids); Tt = max(s.shape[1] for s in specs)
    idt = torch.zeros(batch, Lc, dtype=torch.long, device=device)
    mel = torch.zeros(batch, N_BINS, Tt, device=device)
    mmask = torch.zeros(batch, Tt, device=device)
    for b, (i, s) in enumerate(zip(ids, specs)):
        idt[b, :len(i)] = torch.tensor(i, device=device)
        mel[b, :, :s.shape[1]] = s
        mmask[b, :s.shape[1]] = 1.0
    return idt, mel, mmask, Tt


def train(steps=30000, seed=0, device=DEV, batch=32, lr=3e-4, ckpt_path=None, save_dir=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = load(); train_data = data[:int(len(data) * 0.95)]
    model = FastTTS().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # average frames-per-char for inference length
    ratio = float(np.mean([spec_of(w, "cpu").shape[1] / max(1, len(encode_text(t)))
                           for t, w in train_data[:200]]))
    model.frames_per_char = ratio
    for st in range(1, steps + 1):
        idt, mel, mmask, T = _batch(train_data, rng, batch, device)
        pred, align, h = model(idt, T)
        l_mel = (F.l1_loss(pred, mel, reduction="none").mean(1) * mmask).sum() / mmask.sum()
        l_ga = guided_attention_loss(align, idt.shape[1], T)
        # self-extracted duration target: frames attending to each char; train the predictor to it
        with torch.no_grad():
            dur_tgt = align.sum(1).clamp(min=1).log()      # (B, L)
        l_dur = (F.l1_loss(model.durations(h), dur_tgt, reduction="none")
                 * (~idt.eq(0)).float()).sum() / (~idt.eq(0)).float().sum()
        (l_mel + 2.0 * l_ga + 0.1 * l_dur).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if st % max(1, steps // 12) == 0 or st == steps:
            print(f"  fast {st}/{steps} mel {l_mel.item():.3f} ga {l_ga.item():.4f} "
                  f"dur {l_dur.item():.3f}", flush=True)
        if st % max(1, steps // 5) == 0 and (ckpt_path or save_dir):
            if ckpt_path:
                torch.save({"state_dict": model.state_dict(), "frames_per_char": ratio}, ckpt_path)
            if save_dir:
                try:
                    synth(model, ["the quick brown fox jumps over the lazy dog.",
                                  "hello, this is a test of the speech system."], save_dir, device=device)
                except Exception as e:
                    print(f"  (periodic synth skipped: {e})", flush=True)
            model.train()   # synth()/save left LSTM in eval; cudnn RNN backward needs train mode
    return model, data


@torch.no_grad()
def infer(model, ids, device=DEV):
    """Length from the duration predictor (sum of per-char frames)."""
    h = model.enc(ids)
    dur = model.durations(h).exp() * (~ids.eq(0)).float()
    T = int(dur.sum(-1).max().clamp(8, MAX_T).item())
    pred, _align, _h = model(ids, T)
    return pred


def synth(model, texts, out_dir, device=DEV):
    from .realvoice import Vocos, griffin_lim, write_wav
    os.makedirs(out_dir, exist_ok=True)
    voc = None
    if os.path.exists("runs/realvoice.pt"):
        voc = Vocos(d=512).to(device)
        voc.load_state_dict(torch.load("runs/realvoice.pt", map_location=device)["state_dict"])
        voc.eval()
    model.eval()
    for i, t in enumerate(texts):
        ids = torch.tensor([encode_text(t)], device=device)
        spec = infer(model, ids, device)[0].cpu().numpy()
        wav = (voc(torch.tensor(spec[None], device=device))[0].cpu().numpy() if voc is not None
               else griffin_lim(spec))
        write_wav(os.path.join(out_dir, f"ttsfast{i}.wav"), wav)
        print(f"  ttsfast{i}: \"{t[:48]}\" -> {out_dir}/ttsfast{i}.wav ({spec.shape[1]} frames)")


def evaluate(model, data, device=DEV, n=200, seed=1):
    rng = np.random.default_rng(seed); eval_data = data[int(len(data) * 0.95):]
    model.eval(); errs, focus = [], []
    with torch.no_grad():
        for _ in range(min(n, len(eval_data) * 2)):
            idt, mel, mmask, T = _batch(eval_data, rng, 1, device)
            pred, align, _ = model(idt, T)
            errs.append(float((F.l1_loss(pred, mel, reduction="none").mean(1) * mmask).sum() / mmask.sum()))
            focus.append(float(align.max(-1).values.mean()))
    return {"heldout_spec_l1": float(np.mean(errs)), "attention_focus": float(np.mean(focus))}


def run(steps=30000, seed=0, device=DEV, save_dir="data/synth", ckpt_path="runs/tts_fast.pt"):
    model, data = train(steps=steps, seed=seed, device=device, ckpt_path=ckpt_path, save_dir=save_dir)
    ev = evaluate(model, data, device=device)
    report = {"experiment": "tts_fastspeech_ljspeech", "sr": SR, "steps": steps, **ev,
              "frames_per_char": getattr(model, "frames_per_char", 0), "aligned": ev["attention_focus"] > 0.4}
    print(f"\nheld-out spec L1 {ev['heldout_spec_l1']:.3f}  attention_focus {ev['attention_focus']:.3f}")
    synth(model, ["the quick brown fox jumps over the lazy dog.",
                  "hello, this is a test of the speech system.",
                  "she sells sea shells by the sea shore."], save_dir, device=device)
    return report, model


def selftest():
    m = FastTTS(d=64, layers=2, heads=4)
    a, b = encode_text("hello world"), encode_text("a test.")
    L = max(len(a), len(b))
    ids = torch.tensor([a + [0] * (L - len(a)), b + [0] * (L - len(b))])
    pred, align, h = m(ids, 30)
    assert pred.shape == (2, N_BINS, 30) and align.shape == (2, 30, L), (pred.shape, align.shape)
    assert m.durations(h).shape == (2, L)
    out = infer(m, ids[:1])
    assert out.shape[0] == 1 and out.shape[1] == N_BINS
    if os.path.exists(os.path.join(ROOT, "manifest.json")):
        data = load(); assert data
        m2, d = train(steps=2, seed=0, device="cpu", batch=2)
        ev = evaluate(m2, d, device="cpu", n=3); assert "heldout_spec_l1" in ev
    print("tts_fast selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/tts_fast.json")
    ap.add_argument("--checkpoint", default="runs/tts_fast.pt")
    ap.add_argument("--synth-out", default="data/synth", dest="synth_out")
    ap.add_argument("--say", default="")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return
    if args.train:
        report, model = run(steps=args.steps, seed=args.seed, save_dir=args.synth_out, ckpt_path=args.checkpoint)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"saved -> {args.out}")
        return
    if args.say:
        model = FastTTS().to(DEV)
        ck = torch.load(args.checkpoint, map_location=DEV)
        model.load_state_dict(ck["state_dict"]); model.frames_per_char = ck.get("frames_per_char", 8)
        synth(model, [args.say], args.synth_out)
        return
    ap.error("choose --selftest / --train / --say")


if __name__ == "__main__":
    main()
