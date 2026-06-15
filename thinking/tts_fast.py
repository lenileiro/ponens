"""Non-autoregressive TTS that ACTUALLY ALIGNS: monotonic aligner + length regulator (FastSpeech2 /
Glow-TTS / FastPitch recipe).

First attempt used free learned-position queries cross-attending over characters with only a weak
diagonal (guided-attention) prior. It minimized mel-L1 to 0.09 yet produced BABBLE: an ASR
round-trip scored WER 6.6, and the argmax alignment was non-monotonic and stuck on a few characters
(attention_focus 0.28). Soft cross-attention cannot self-organize a monotonic text->audio map from
scratch on this much data.

The fix the field converged on, and the one that makes parallel TTS intelligible WITHOUT an external
forced aligner:

  1. ALIGNER (train only): mel frames -> queries, char states -> keys; soft alignment = log-softmax
     of negative L2 distance.  A CTC-style FORWARD-SUM loss makes that soft alignment a valid
     MONOTONIC path (every char emitted, left to right).
  2. MAS (monotonic alignment search): Viterbi over the soft alignment -> HARD integer durations
     d_i (frames per character). Non-differentiable, so it only reads the aligner; the aligner is
     trained purely by the forward-sum loss.
  3. LENGTH REGULATOR: expand each char state by d_i -> a frame-level sequence that is monotonic BY
     CONSTRUCTION. The decoder (transformer) renders it -> spectrogram. Frame t literally carries the
     character that should sound at t, so the decoder articulates instead of averaging.
  4. DURATION PREDICTOR: trained (log-MSE) against d_i; supplies lengths at inference (no mel needed).

Same 513-bin log-mag target + realvoice vocoder, so it drops into the prosody / say_test pipeline.

  python -m thinking.tts_fast --selftest
  python -m thinking.tts_fast --train --steps 40000 --out runs/tts_fast.json   (GPU)
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
from .tts import ROOT, MAX_T, MAX_CH, encode_text, load, spec_of, CharEncoder

DEV = get_device()


# ---- monotonic alignment search (Viterbi), batched over items, loop only over time --------------
@torch.no_grad()
def mas_durations(log_align, ids_len, mel_len):
    """log_align (B,T,L) -> hard durations (B,L) int. Each of the Tb frames maps to exactly one of
    the Lb chars, non-decreasing, ending on the last char (so durations sum to Tb)."""
    B, T, L = log_align.shape
    dev = log_align.device
    NEG = -1e9
    Q = log_align.clone()
    ar_L = torch.arange(L, device=dev)
    cmask = ar_L[None] >= ids_len[:, None]          # padded chars
    Q.masked_fill_(cmask[:, None, :], NEG)
    dp = torch.full((B, L), NEG, device=dev)
    dp[:, 0] = Q[:, 0, 0]
    bp = torch.zeros(B, T, L, dtype=torch.bool, device=dev)   # True = came from j-1
    for t in range(1, T):
        stay = dp                                   # (B,L) stay on char j
        move = F.pad(dp[:, :-1], (1, 0), value=NEG)  # from char j-1
        from_prev = move > stay
        dp = Q[:, t] + torch.where(from_prev, move, stay)
        bp[:, t] = from_prev
    # backtrack, BATCHED over items (loop only over time): start each item on its own last char
    dur = torch.zeros(B, L, dtype=torch.long, device=dev)
    ar_B = torch.arange(B, device=dev)
    j = (ids_len - 1).clamp(min=0)                  # (B,) current char
    for t in range(T - 1, -1, -1):
        active = (t < mel_len)                      # frames within this item's true length
        dur[ar_B, j] += active.long()
        came = bp[ar_B, t, j] & active & (j > 0)    # step back to char j-1 ?
        j = j - came.long()
    return dur


def uniform_durations(ids_len, tgt_len):
    """Each real char of item b gets an equal share of that item's OWN tgt_len[b] frames (remainder
    to the front). Collapse-proof: correct char ORDER and per-item length by construction."""
    B = ids_len.shape[0]
    tgt_len = tgt_len if torch.is_tensor(tgt_len) else torch.full((B,), int(tgt_len))
    dur = torch.zeros(B, int(ids_len.max()), dtype=torch.long, device=ids_len.device)
    for b in range(B):
        L = max(1, int(ids_len[b].item())); Tb = int(tgt_len[b].item())
        base, rem = divmod(Tb, L)
        dur[b, :L] = base
        dur[b, :rem] += 1                       # spread remainder so durations sum exactly to Tb
    return dur


def regulate(h, dur, T):
    """Expand char states h (B,L,d) by integer durations dur (B,L) -> (B,T,d), monotonic."""
    B, L, d = h.shape
    out = h.new_zeros(B, T, d)
    for b in range(B):
        idx = torch.repeat_interleave(torch.arange(L, device=h.device), dur[b].clamp(min=0))[:T]
        if len(idx) == 0:
            idx = torch.zeros(1, dtype=torch.long, device=h.device)
        out[b, :len(idx)] = h[b, idx]
        if len(idx) < T:
            out[b, len(idx):] = h[b, idx[-1]]       # pad tail with last char
    return out


def forward_sum_loss(log_align, ids_len, mel_len):
    """CTC forward-sum: make the soft alignment a valid monotonic path (FastPitch)."""
    B, T, L = log_align.shape
    logp = F.pad(log_align, (1, 0), value=-1e3)     # add blank at index 0
    logp = F.log_softmax(logp.permute(1, 0, 2), -1)  # (T,B,L+1)
    targets = torch.arange(1, L + 1, device=log_align.device)[None].expand(B, -1)  # 1..L
    return F.ctc_loss(logp, targets, mel_len.clamp(max=T), ids_len.clamp(max=L),
                      blank=0, zero_infinity=True)


class FastTTS(nn.Module):
    """char -> states; (aligner + MAS) gives durations; length-regulate -> decode -> spectrogram."""

    def __init__(self, d=256, n_bins=N_BINS, layers=4, heads=4, max_frames=MAX_T):
        super().__init__()
        self.d = d
        self.enc = CharEncoder(d)
        self.mel_q = nn.Sequential(nn.Conv1d(n_bins, d, 3, padding=1), nn.GELU(),
                                   nn.Conv1d(d, d, 3, padding=1))     # mel frames -> queries
        self.key = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(d, d, 1))                  # char states -> keys
        self.frame_pos = nn.Parameter(torch.randn(max_frames, d) * 0.02)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu",
                                           batch_first=True)
        self.dec = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_bins))
        self.dur = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))   # log frames/char
        # Tacotron2-style postnet: residual conv refinement that sharpens the coarse spectrogram
        pn, ch = [], 256
        for i in range(5):
            ci, co = (n_bins if i == 0 else ch), (n_bins if i == 4 else ch)
            pn += [nn.Conv1d(ci, co, 5, padding=2)]
            if i < 4:
                pn += [nn.BatchNorm1d(co), nn.Tanh(), nn.Dropout(0.1)]
        self.postnet = nn.Sequential(*pn)

    def char_states(self, ids):
        return self.enc(ids)

    def soft_align(self, h, mel, ids, ids_len=None, mel_len=None, prior_w=0.0):
        q = self.mel_q(mel).transpose(1, 2)                 # (B,T,d)
        k = self.key(h.transpose(1, 2)).transpose(1, 2)     # (B,L,d)
        dist = -((q[:, :, None, :] - k[:, None, :, :]) ** 2).mean(-1)   # (B,T,L) neg L2
        if prior_w > 0:                                     # diagonal prior prevents collapse-to-char0
            dist = dist + prior_w * self._diag_prior(dist.shape, ids_len, mel_len, dist.device)
        dist = dist.masked_fill(ids.eq(0)[:, None, :], -1e4)
        return F.log_softmax(dist, -1)

    @staticmethod
    def _diag_prior(shape, ids_len, mel_len, device):
        """Gaussian band around the time->char diagonal (in char units), per item."""
        B, T, L = shape
        t = torch.arange(T, device=device).float()[None, :, None]       # (1,T,1)
        j = torch.arange(L, device=device).float()[None, None, :]       # (1,1,L)
        ml = mel_len.float().clamp(min=1)[:, None, None]
        il = (ids_len.float() - 1).clamp(min=1)[:, None, None]
        center = (t / ml) * il                              # expected char index at frame t
        sigma = (0.1 * ids_len.float()).clamp(min=3.0)[:, None, None]
        return -0.5 * ((j - center) / sigma) ** 2           # (B,T,L), 0 on diagonal, negative off

    def decode(self, h, dur, T):
        reg = regulate(h, dur, T) + self.frame_pos[:T][None]   # (B,T,d) monotonic + position
        coarse = self.out(self.dec(reg)).transpose(1, 2)       # (B,n_bins,T)
        refined = coarse + self.postnet(coarse)                # residual postnet sharpening
        return coarse, refined

    def durations(self, h):
        return self.dur(h).squeeze(-1)                         # (B,L) log frames/char


def _batch(data, rng, batch, device):
    items = [data[int(rng.integers(len(data)))] for _ in range(batch)]
    specs = [spec_of(w, device)[:, :MAX_T] for _, w in items]
    ids = [encode_text(t) for t, _ in items]
    Lc = max(len(i) for i in ids); Tt = max(s.shape[1] for s in specs)
    idt = torch.zeros(batch, Lc, dtype=torch.long, device=device)
    mel = torch.zeros(batch, N_BINS, Tt, device=device)
    ids_len = torch.zeros(batch, dtype=torch.long, device=device)
    mel_len = torch.zeros(batch, dtype=torch.long, device=device)
    for b, (i, s) in enumerate(zip(ids, specs)):
        idt[b, :len(i)] = torch.tensor(i, device=device); ids_len[b] = len(i)
        mel[b, :, :s.shape[1]] = s; mel_len[b] = s.shape[1]
    return idt, mel, ids_len, mel_len, Tt


def train(steps=40000, seed=0, device=DEV, batch=32, lr=3e-4, ckpt_path=None, save_dir=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = load(); train_data = data[:int(len(data) * 0.95)]
    model = FastTTS().to(device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ratio = float(np.mean([spec_of(w, "cpu").shape[1] / max(1, len(encode_text(t)))
                           for t, w in train_data[:200]]))
    model.frames_per_char = ratio
    for st in range(1, steps + 1):
        idt, mel, ids_len, mel_len, T = _batch(train_data, rng, batch, device)
        h = model.char_states(idt)
        dur_tgt = uniform_durations(ids_len, mel_len)       # equal share over each item's true length
        coarse, refined = model.decode(h, dur_tgt, T)
        mmask = (torch.arange(T, device=device)[None] < mel_len[:, None]).float()
        l_mel = sum((F.l1_loss(p, mel, reduction="none").mean(1) * mmask).sum() / mmask.sum()
                    for p in (coarse, refined))             # coarse + postnet-refined
        cmask = (~idt.eq(0)).float()
        l_dur = (F.l1_loss(model.durations(h), (dur_tgt.float() + 1).log(), reduction="none")
                 * cmask).sum() / cmask.sum()
        (l_mel + 0.1 * l_dur).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if st % max(1, steps // 16) == 0 or st == steps:
            print(f"  fast {st}/{steps} mel {l_mel.item():.3f} dur {l_dur.item():.3f}", flush=True)
        if st % max(1, steps // 5) == 0 and (ckpt_path or save_dir):
            if ckpt_path:
                torch.save({"state_dict": model.state_dict(), "frames_per_char": ratio}, ckpt_path)
            if save_dir:
                try:
                    synth(model, ["the quick brown fox jumps over the lazy dog.",
                                  "hello, this is a test of the speech system."], save_dir, device=device)
                except Exception as e:
                    print(f"  (periodic synth skipped: {e})", flush=True)
            model.train()
    return model, data


@torch.no_grad()
def infer(model, ids, device=DEV):
    """Lengths from the duration predictor; length-regulate; decode (no mel needed)."""
    model.eval()
    h = model.char_states(ids)
    dur = (model.durations(h).exp() - 1).clamp(min=0).round().long()
    dur = dur * (~ids.eq(0)).long()
    T = int(dur.sum(-1).max().clamp(8, MAX_T).item())
    return model.decode(h, dur, T)[1]                       # refined spectrogram


@torch.no_grad()
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
    model.eval(); errs = []
    with torch.no_grad():
        for _ in range(min(n, len(eval_data) * 2)):
            idt, mel, ids_len, mel_len, T = _batch(eval_data, rng, 1, device)
            h = model.char_states(idt)
            dur = uniform_durations(ids_len, mel_len)
            pred = model.decode(h, dur, T)[1]
            mmask = (torch.arange(T, device=device)[None] < mel_len[:, None]).float()
            errs.append(float((F.l1_loss(pred, mel, reduction="none").mean(1) * mmask).sum() / mmask.sum()))
    return {"heldout_spec_l1": float(np.mean(errs))}


def run(steps=40000, seed=0, device=DEV, save_dir="data/synth", ckpt_path="runs/tts_fast.pt"):
    model, data = train(steps=steps, seed=seed, device=device, ckpt_path=ckpt_path, save_dir=save_dir)
    ev = evaluate(model, data, device=device)
    report = {"experiment": "tts_fastspeech_uniform_ljspeech", "sr": SR, "steps": steps, **ev,
              "frames_per_char": getattr(model, "frames_per_char", 0)}
    print(f"\nheld-out spec L1 {ev['heldout_spec_l1']:.3f}")
    synth(model, ["the quick brown fox jumps over the lazy dog.",
                  "hello, this is a test of the speech system.",
                  "she sells sea shells by the sea shore."], save_dir, device=device)
    return report, model


def selftest():
    torch.manual_seed(0)
    m = FastTTS(d=64, layers=2, heads=4)
    a, b = encode_text("hello world"), encode_text("a test.")
    L = max(len(a), len(b))
    ids = torch.tensor([a + [0] * (L - len(a)), b + [0] * (L - len(b))])
    ids_len = torch.tensor([len(a), len(b)]); mel_len = torch.tensor([40, 32])
    h = m.char_states(ids)
    dur = uniform_durations(ids_len, mel_len)
    assert (dur[0].sum() == 40) and (dur[1].sum() == 32), (dur.sum(1), mel_len)
    assert dur[0, len(a):].sum() == 0, "duration leaked onto padding"
    coarse, refined = m.decode(h, dur, 40); assert refined.shape == (2, N_BINS, 40)
    out = infer(m, ids[:1]); assert out.shape[0] == 1 and out.shape[1] == N_BINS
    if os.path.exists(os.path.join(ROOT, "manifest.json")):
        data = load(); assert data
        m2, d = train(steps=2, seed=0, device="cpu", batch=2)
        ev = evaluate(m2, d, device="cpu", n=3); assert "heldout_spec_l1" in ev
    print("tts_fast selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=40000)
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
