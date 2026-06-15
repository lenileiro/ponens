"""Autoregressive Transformer-TTS: the approach that actually produces INTELLIGIBLE speech.

Three non-autoregressive attempts (free-query cross-attn; monotonic aligner + MAS; uniform length
regulation + postnet) all failed the same way -- an ASR round-trip heard fluent but WRONG words.
Root cause: an L1 regression loss on a NON-autoregressive model yields the per-frame *average*
spectrogram (the L1-optimal under uncertainty is a blur), and a blurry spectrogram is mumbled,
unintelligible speech. A postnet can sharpen a little but cannot fix fundamental blur.

The fix the field settled on for small single-speaker data: make it AUTOREGRESSIVE. Conditioned on
the previous frames, the next frame is nearly deterministic -> the model predicts a SHARP spectrum,
and cross-attention to the character sequence (kept on the diagonal by a guided-attention loss)
gives the alignment. Unlike the old per-frame Tacotron loop, a Transformer decoder trains in
PARALLEL with a causal mask (all frames at once) -> fast enough to actually converge; only inference
is sequential.

Predicts the same 513-bin log-mag the realvoice vocoder consumes, so it drops into prosody/say_test.

  python -m thinking.tts_ar --selftest
  python -m thinking.tts_ar --train --steps 60000 --out runs/tts_ar.json   (GPU)
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .realvoice import SR, N_BINS
from .tts import ROOT, MAX_T, MAX_CH, encode_text, load, spec_of, CharEncoder, guided_attention_loss

DEV = get_device()


def sinusoidal(T, d, device):
    pos = torch.arange(T, device=device).float()[:, None]
    i = torch.arange(d, device=device).float()[None, :]
    ang = pos / (10000 ** (2 * (i // 2) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(ang[:, 0::2]); pe[:, 1::2] = torch.cos(ang[:, 1::2])
    return pe


class DecoderLayer(nn.Module):
    """Pre-norm: causal self-attention + cross-attention (weights exposed) + FFN."""

    def __init__(self, d, heads):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d)

    def forward(self, x, mem, causal, mem_pad):
        h = self.n1(x)
        x = x + self.sa(h, h, h, attn_mask=causal, need_weights=False)[0]
        h = self.n2(x)
        a, w = self.ca(h, mem, mem, key_padding_mask=mem_pad, need_weights=True,
                       average_attn_weights=True)
        x = x + a
        x = x + self.ff(self.n3(x))
        return x, w


class TransformerTTS(nn.Module):
    def __init__(self, d=256, n_bins=N_BINS, layers=4, heads=4):
        super().__init__()
        self.d = d
        self.enc = CharEncoder(d)
        self.prenet = nn.Sequential(nn.Linear(n_bins, 256), nn.ReLU(), nn.Linear(256, d), nn.ReLU())
        self.layers = nn.ModuleList([DecoderLayer(d, heads) for _ in range(layers)])
        self.out = nn.Linear(d, n_bins)
        self.stop = nn.Linear(d, 1)
        pn, ch = [], 256
        for i in range(5):
            ci, co = (n_bins if i == 0 else ch), (n_bins if i == 4 else ch)
            pn += [nn.Conv1d(ci, co, 5, padding=2)]
            if i < 4:
                pn += [nn.BatchNorm1d(co), nn.Tanh(), nn.Dropout(0.1)]
        self.postnet = nn.Sequential(*pn)

    def prenet_do(self, x):
        # prenet dropout stays ON at inference (TransformerTTS/Tacotron2 trick -> AR stability)
        x = F.dropout(F.relu(self.prenet[0](x)), 0.5, training=True)
        return F.dropout(F.relu(self.prenet[2](x)), 0.5, training=True)

    def decode(self, ids, mel_in):
        """Teacher-forced, PARALLEL over time. ids (B,L), mel_in (B,n_bins,T) -> mel, stop, align."""
        B, _, T = mel_in.shape
        mem = self.enc(ids); mem_pad = ids.eq(0)
        x = self.prenet_do(mel_in.transpose(1, 2)) + sinusoidal(T, self.d, ids.device)[None]
        causal = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        ws = []
        for layer in self.layers:
            x, w = layer(x, mem, causal, mem_pad)
            ws.append(w)
        coarse = self.out(x).transpose(1, 2)                 # (B,n_bins,T)
        refined = coarse + self.postnet(coarse)
        return coarse, refined, self.stop(x).squeeze(-1), ws  # ws: list of (B,T,L) per layer

    @torch.no_grad()
    def infer(self, ids, max_T=MAX_T, stop_thresh=0.5):
        self.eval()
        mem = self.enc(ids); mem_pad = ids.eq(0)
        B = ids.shape[0]
        frames = torch.zeros(B, N_BINS, 1, device=ids.device)   # start frame
        for t in range(max_T):
            mel_in = frames
            T = mel_in.shape[2]
            x = self.prenet_do(mel_in.transpose(1, 2)) + sinusoidal(T, self.d, ids.device)[None]
            causal = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
            for layer in self.layers:
                x, _ = layer(x, mem, causal, mem_pad)
            nxt = self.out(x[:, -1])                           # next coarse frame (B,n_bins)
            frames = torch.cat([frames, nxt[:, :, None]], -1)
            if t > 8 and torch.sigmoid(self.stop(x[:, -1])).max().item() > stop_thresh:
                break
        coarse = frames[:, :, 1:]                              # drop start frame
        return coarse + self.postnet(coarse)                   # refined


def _batch(data, rng, batch, device):
    items = [data[int(rng.integers(len(data)))] for _ in range(batch)]
    specs = [spec_of(w, device)[:, :MAX_T] for _, w in items]
    ids = [encode_text(t) for t, _ in items]
    Lc = max(len(i) for i in ids); Tt = max(s.shape[1] for s in specs)
    idt = torch.zeros(batch, Lc, dtype=torch.long, device=device)
    mel = torch.zeros(batch, N_BINS, Tt, device=device)
    stop = torch.zeros(batch, Tt, device=device)
    ids_len = torch.zeros(batch, dtype=torch.long, device=device)
    mel_len = torch.zeros(batch, dtype=torch.long, device=device)
    for b, (i, s) in enumerate(zip(ids, specs)):
        idt[b, :len(i)] = torch.tensor(i, device=device); ids_len[b] = len(i)
        mel[b, :, :s.shape[1]] = s; mel_len[b] = s.shape[1]
        stop[b, s.shape[1] - 1:] = 1.0
    return idt, mel, stop, ids_len, mel_len


def train(steps=60000, seed=0, device=DEV, batch=16, lr=3e-4, ckpt_path=None, save_dir=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = load(); train_data = data[:int(len(data) * 0.95)]
    model = TransformerTTS().to(device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        idt, mel, stop, ids_len, mel_len = _batch(train_data, rng, batch, device)
        T = mel.shape[1] if False else mel.shape[2]
        mel_in = F.pad(mel, (1, 0))[:, :, :-1]               # shift for teacher forcing
        coarse, refined, stop_logit, aligns = model.decode(idt, mel_in)
        mmask = (torch.arange(mel.shape[2], device=device)[None] < mel_len[:, None]).float()
        l_mel = sum((F.l1_loss(p, mel, reduction="none").mean(1) * mmask).sum() / mmask.sum()
                    for p in (coarse, refined))
        l_stop = F.binary_cross_entropy_with_logits(stop_logit, stop)
        # guided attention on EVERY decoder layer's cross-attn -> all layers must align (no AR-cheat)
        l_ga = sum(guided_attention_loss(al[b:b + 1], int(ids_len[b]), int(mel_len[b]))
                   for al in aligns for b in range(idt.shape[0])) / (idt.shape[0] * len(aligns))
        (l_mel + l_stop + 5.0 * l_ga).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if st % max(1, steps // 20) == 0 or st == steps:
            focus = float(aligns[-1].max(-1).values.mean().detach())
            print(f"  ar {st}/{steps} mel {l_mel.item():.3f} stop {l_stop.item():.3f} "
                  f"ga {l_ga.item():.4f} focus {focus:.3f}", flush=True)
        if st % max(1, steps // 5) == 0 and ckpt_path:
            torch.save({"state_dict": model.state_dict()}, ckpt_path)
            model.train()
    return model, data


@torch.no_grad()
def synth(model, texts, out_dir, device=DEV):
    from .realvoice import Vocos, griffin_lim, write_wav
    os.makedirs(out_dir, exist_ok=True)
    voc = None
    if os.path.exists("runs/realvoice.pt"):
        voc = Vocos(d=512).to(device)
        voc.load_state_dict(torch.load("runs/realvoice.pt", map_location=device)["state_dict"]); voc.eval()
    model.eval()
    for i, t in enumerate(texts):
        ids = torch.tensor([encode_text(t)], device=device)
        spec = model.infer(ids)[0].cpu().numpy()
        wav = (voc(torch.tensor(spec[None], device=device))[0].cpu().numpy() if voc is not None
               else griffin_lim(spec))
        write_wav(os.path.join(out_dir, f"ttsar{i}.wav"), wav)
        print(f"  ttsar{i}: \"{t[:46]}\" -> {spec.shape[1]} frames")


def evaluate(model, data, device=DEV, n=80, seed=1):
    rng = np.random.default_rng(seed); eval_data = data[int(len(data) * 0.95):]
    model.eval(); errs, focus = [], []
    with torch.no_grad():
        for _ in range(min(n, len(eval_data) * 2)):
            idt, mel, stop, ids_len, mel_len = _batch(eval_data, rng, 1, device)
            mel_in = F.pad(mel, (1, 0))[:, :, :-1]
            _, refined, _, aligns = model.decode(idt, mel_in)
            mmask = (torch.arange(mel.shape[2], device=device)[None] < mel_len[:, None]).float()
            errs.append(float((F.l1_loss(refined, mel, reduction="none").mean(1) * mmask).sum() / mmask.sum()))
            focus.append(float(aligns[-1].max(-1).values.mean()))
    return {"heldout_spec_l1": float(np.mean(errs)), "attention_focus": float(np.mean(focus))}


def run(steps=60000, seed=0, device=DEV, save_dir="data/synth", ckpt_path="runs/tts_ar.pt"):
    model, data = train(steps=steps, seed=seed, device=device, ckpt_path=ckpt_path)
    ev = evaluate(model, data, device=device)
    report = {"experiment": "tts_transformer_ar_ljspeech", "sr": SR, "steps": steps, **ev,
              "aligned": ev["attention_focus"] > 0.4}
    print(f"\nheld-out spec L1 {ev['heldout_spec_l1']:.3f}  attention_focus {ev['attention_focus']:.3f}")
    synth(model, ["the quick brown fox jumps over the lazy dog.",
                  "hello, this is a test of the speech system.",
                  "she sells sea shells by the sea shore."], save_dir, device=device)
    return report, model


def selftest():
    torch.manual_seed(0)
    m = TransformerTTS(d=64, layers=2, heads=4)
    a, b = encode_text("hello world"), encode_text("a test.")
    L = max(len(a), len(b))
    ids = torch.tensor([a + [0] * (L - len(a)), b + [0] * (L - len(b))])
    mel_in = torch.randn(2, N_BINS, 20)
    coarse, refined, stoplg, aligns = m.decode(ids, mel_in)
    assert coarse.shape == (2, N_BINS, 20) and aligns[-1].shape == (2, 20, L) and stoplg.shape == (2, 20)
    out = m.infer(ids[:1], max_T=12); assert out.shape[0] == 1 and out.shape[1] == N_BINS
    if os.path.exists(os.path.join(ROOT, "manifest.json")):
        data = load(); assert data
        m2, d = train(steps=2, seed=0, device="cpu", batch=2)
        ev = evaluate(m2, d, device="cpu", n=2); assert "heldout_spec_l1" in ev
    print("tts_ar selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/tts_ar.json")
    ap.add_argument("--checkpoint", default="runs/tts_ar.pt")
    ap.add_argument("--synth-out", default="data/synth", dest="synth_out")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return
    if args.train:
        report, _ = run(steps=args.steps, seed=args.seed, save_dir=args.synth_out, ckpt_path=args.checkpoint)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"saved -> {args.out}")
        return
    ap.error("choose --selftest / --train")


if __name__ == "__main__":
    main()
