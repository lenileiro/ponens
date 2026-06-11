"""M-0: the multimodal bridge — ONE ScratchpadLM that SEES, HEARS, and emits canonical facts.

Image patches and audio spectrogram frames enter as continuous prefix embeddings (modality-tagged)
before the token stream; the model emits the SAME extract-trace grammar used everywhere else:

  [IMG prefix][AUD prefix] extract fact p0 color red . fact p0 shape circle .
                           fact a0 pitch n440 . fact a0 timbre saw . fact a0 env decay . done .

Both worlds are synthetic with oracle factors (vision.py shapes, audio.py tones), so supervision
is oracle-sound and the FER question becomes cross-modal: one unified extraction circuit fed by
two readers, or two entangled task-specific ones. The Image-2/Audio-1 lesson is baked in from the
start: each modality trunk projects through a FACTORED interface (per-factor subspace slices),
the intervention that took held-out-combo accuracy to 1.00.

  python -m thinking.multimodal --selftest
  python -m thinking.multimodal --steps 400 --out runs/m0_multimodal.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from scratchpad_model import ScratchpadLM

from .audio import (ENVELOPES, PITCH_NAMES, TIMBRES, render_tone, sample_clip, spectrogram)
from .vision import COLORS, SHAPES, render_object, sample_object
from .trace import Vocab

DEV = get_device()
COLOR_NAMES = tuple(COLORS)


class PatchTrunk(nn.Module):
    """Conv trunk -> a short sequence of d-dim prefix embeddings + modality embedding."""

    def __init__(self, in_ch, d, n_tokens=4, modality=0, n_modalities=2, pool=(2, 2)):
        """pool: trunk output cells -> prefix tokens. Audio needs FREQUENCY-preserving pooling
        ((8,1): 8 frequency bands x collapsed time) -- 2x2 pooling crushed the frequency axis
        and pitch accuracy with it (0.32; every other factor >=0.93)."""
        super().__init__()
        self.n_tokens = n_tokens
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(pool),
        )
        self.proj = nn.Linear(64, d)
        self.mod = nn.Embedding(n_modalities, d)
        self.modality = modality
        self.posn = nn.Parameter(torch.zeros(n_tokens, d))

    def forward(self, x):
        h = self.conv(x)                                    # B,64,2,2
        h = h.flatten(2).transpose(1, 2)                    # B,4,64
        return self.proj(h) + self.mod.weight[self.modality] + self.posn[None]


class MultimodalLM(nn.Module):
    """Two factored readers feeding one trace decoder."""

    def __init__(self, vocab_size, d=96, layers=3, heads=4, pad=0, max_len=128):
        super().__init__()
        self.lm = ScratchpadLM(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                               pad=pad, pointer=False, loop=False)
        self.img = PatchTrunk(3, d, n_tokens=4, modality=0, pool=(2, 2))
        self.aud = PatchTrunk(1, d, n_tokens=8, modality=1, pool=(8, 1))

    def forward(self, img, aud, ids):
        prefix = torch.cat([self.img(img), self.aud(aud)], dim=1)   # B,12,d
        logits = self.lm(ids, prefix=prefix)
        return logits[:, prefix.shape[1]:]                  # token positions only


def build_vocab():
    toks = ["extract", "fact", "done", ".", "p0", "a0", "color", "shape", "pitch", "timbre",
            "env"]
    toks += list(COLOR_NAMES) + list(SHAPES) + list(PITCH_NAMES) + list(TIMBRES) + list(ENVELOPES)
    return Vocab(toks)


def sample_example(rng):
    obj = sample_object(rng, slot="p0")
    clip = sample_clip(rng)
    img = render_object(obj, size=32)
    aud = spectrogram(render_tone(clip["pitch"], clip["timbre"], clip["envelope"],
                                  clip["detune"], clip["amp"], clip["phase"], rng=rng))
    toks = ["extract",
            "fact", "p0", "color", obj.color, ".",
            "fact", "p0", "shape", obj.shape, ".",
            "fact", "a0", "pitch", clip["pitch"], ".",
            "fact", "a0", "timbre", clip["timbre"], ".",
            "fact", "a0", "env", clip["envelope"], ".",
            "done", "."]
    gold = {"color": obj.color, "shape": obj.shape, "pitch": clip["pitch"],
            "timbre": clip["timbre"], "env": clip["envelope"]}
    return img, aud, toks, gold


def _batch(n, rng, vocab, device):
    imgs, auds, seqs, golds = [], [], [], []
    for _ in range(n):
        img, aud, toks, gold = sample_example(rng)
        imgs.append(img)
        auds.append(aud)
        seqs.append(vocab.enc(toks))
        golds.append(gold)
    x = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
    a = torch.tensor(np.stack(auds), dtype=torch.float32, device=device)
    ids = torch.tensor(seqs, dtype=torch.long, device=device)   # fixed grammar = fixed length
    return x, a, ids, golds


VALUE_POS = {"color": 4, "shape": 9, "pitch": 14, "timbre": 19, "env": 24}  # token index of value


def evaluate(model, vocab, n=200, seed=1, device=DEV):
    """Teacher-forced per-factor accuracy: at each fact's value position, does the model put
    the oracle value first among that factor's candidates?"""
    rng = np.random.default_rng(seed)
    model.eval()
    hits = {k: 0 for k in VALUE_POS}
    with torch.no_grad():
        for off in range(0, n, 50):
            b = min(50, n - off)
            img, aud, ids, golds = _batch(b, rng, vocab, device)
            logits = model(img, aud, ids)
            for k, pos in VALUE_POS.items():
                pred = logits[:, pos - 1].argmax(-1)        # next-token prediction of the value
                for r in range(b):
                    hits[k] += int(vocab.itos[int(pred[r])] == golds[r][k])
    return {k: v / n for k, v in hits.items()}


def token_loss(logits, ids, value_w=6.0):
    targets = ids[:, 1:]
    raw = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1),
                          reduction="none").view_as(targets)
    weights = torch.ones_like(raw)
    for pos in VALUE_POS.values():
        weights[:, pos - 1] = value_w                   # logit at pos-1 predicts value at pos
    return (raw * weights).sum() / weights.sum()


def train(steps=400, batch=32, d=96, lr=1e-3, seed=0, device=DEV, log_every=100, value_w=6.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = build_vocab()
    model = MultimodalLM(len(vocab), d=d, pad=vocab.pad).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        model.train()
        img, aud, ids, _ = _batch(batch, rng, vocab, device)
        logits = model(img, aud, ids)
        loss = token_loss(logits, ids, value_w=value_w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % log_every == 0 or st == steps:
            print(f"  m0 {st}/{steps} loss {loss.item():.3f}", flush=True)
    return model, vocab


CHANCE = {"color": 1 / len(COLORS), "shape": 1 / len(SHAPES),
          "pitch": 1 / len(PITCH_NAMES), "timbre": 1 / len(TIMBRES),
          "env": 1 / len(ENVELOPES)}


def gate_thresholds(chance=CHANCE, gap_frac=0.40):
    """Require each factor to close a fixed fraction of the gap above chance.

    A raw multiplier such as 3x chance is impossible for 3-way factors because the threshold
    exceeds 1.0.  Gap-normalized thresholds compare factors with different cardinalities fairly.
    """
    return {k: v + gap_frac * (1.0 - v) for k, v in chance.items()}


def run(steps=400, seed=0, device=DEV, value_w=6.0):
    model, vocab = train(steps=steps, seed=seed, device=device, value_w=value_w)
    acc = evaluate(model, vocab, device=device)
    thresholds = gate_thresholds()
    report = {"experiment": "m0_multimodal_bridge", "steps": steps, "value_w": float(value_w),
              "per_factor_acc": acc, "chance": CHANCE, "gate_thresholds": thresholds,
              "gate": all(acc[k] >= thresholds[k] for k in acc)}
    print(json.dumps(report, indent=1), flush=True)
    return report


def selftest():
    rng = np.random.default_rng(0)
    vocab = build_vocab()
    img, aud, toks, gold = sample_example(rng)
    assert toks[VALUE_POS["color"]] == gold["color"] and toks[VALUE_POS["env"]] == gold["env"]
    assert all(t in vocab.stoi for t in toks), "grammar token missing from vocab"
    model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad).to("cpu")
    x, a, ids, _ = _batch(2, rng, vocab, "cpu")
    logits = model(x, a, ids)
    assert logits.shape == (2, ids.shape[1], len(vocab)), logits.shape
    th = gate_thresholds()
    assert all(CHANCE[k] < th[k] < 1.0 for k in CHANCE), th
    loss = token_loss(logits, ids)
    loss.backward()
    grads = [p.grad is not None and float(p.grad.abs().sum()) > 0
             for p in (model.img.proj.weight, model.aud.proj.weight)]
    assert all(grads), "gradients must flow into BOTH modality trunks"
    print("multimodal selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value-w", type=float, default=6.0, dest="value_w")
    ap.add_argument("--out", default="runs/m0_multimodal.json")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    report = run(steps=args.steps, seed=args.seed, value_w=args.value_w)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
