"""M-0: the multimodal bridge — ONE ScratchpadLM that SEES, HEARS, READS, and emits facts.

Image patches and audio spectrogram frames enter as continuous prefix embeddings (modality-tagged)
before the token stream.  A transcript/caption enters through the same prefix interface.  The
model emits the SAME extract-trace grammar used everywhere else:

  [IMG prefix][AUD prefix][TXT prefix] extract fact p0 color red . fact p0 shape circle .
                                        fact a0 pitch n440 . fact a0 timbre saw . fact a0 env decay . done .

Both worlds are synthetic with oracle factors (vision.py shapes, audio.py tones), so supervision
is oracle-sound and the FER question becomes cross-modal: one unified extraction circuit fed by
three readers, or three entangled task-specific ones. The language gate is intentionally not
"predict the next token in the caption": held-out transcript phrasings are encoded as a sensory
prefix and must map to the same canonical facts with image/audio zeroed.

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

TEXT_TRAIN = (
    ("the picture shows p0 as a {color} {shape} . the sound from a0 has pitch {pitch} "
     "timbre {timbre} and envelope {env} ."),
    ("p0 appears {color} and {shape} . a0 plays a {timbre} tone at {pitch} with {env} "
     "envelope ."),
    ("visual p0 color {color} shape {shape} . audio a0 pitch {pitch} timbre {timbre} "
     "env {env} ."),
)
TEXT_EVAL = (
    ("in the image p0 is shaped like a {shape} and colored {color} . in the audio a0 uses "
     "{timbre} at {pitch} with {env} dynamics ."),
    ("seen object p0 has {shape} shape plus {color} color . heard clip a0 carries {pitch} "
     "pitch {timbre} timbre {env} envelope ."),
    ("a0 is heard as {timbre} {pitch} {env} . p0 is seen as {shape} in {color} ."),
)
MODES = ("full", "sensor_only", "text_only")


def _split_words(s):
    return s.split()


def render_transcript(gold, rng, split="train"):
    bank = TEXT_TRAIN if split == "train" else TEXT_EVAL
    tpl = bank[int(rng.integers(len(bank)))]
    return _split_words(tpl.format(**gold))


class PatchTrunk(nn.Module):
    """Conv trunk -> a short sequence of d-dim prefix embeddings + modality embedding."""

    def __init__(self, in_ch, d, n_tokens=4, modality=0, n_modalities=3, pool=(2, 2)):
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


class TextTrunk(nn.Module):
    """Transcript encoder -> a short prefix.  Pads are masked inside the text encoder."""

    def __init__(self, vocab_size, d, pad=0, n_tokens=8, heads=4, modality=2, n_modalities=3,
                 max_len=64):
        super().__init__()
        self.pad = pad
        self.n_tokens = n_tokens
        self.emb = nn.Embedding(vocab_size, d, padding_idx=pad)
        self.pos = nn.Embedding(max_len, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                         dropout=0.0, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=1, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mod = nn.Embedding(n_modalities, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, ids):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        pad_mask = ids.eq(self.pad)
        h = self.emb(ids) + self.pos(pos)[None]
        h = self.enc(h, src_key_padding_mask=pad_mask)
        q = self.query[None].expand(B, -1, -1) + self.mod.weight[2]
        out, _ = self.attn(q, h, h, key_padding_mask=pad_mask)
        return self.ln(out + q)


class MultimodalLM(nn.Module):
    """Image, audio, and transcript readers feeding one trace decoder."""

    def __init__(self, vocab_size, d=96, layers=3, heads=4, pad=0, max_len=128):
        super().__init__()
        self.lm = ScratchpadLM(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                               pad=pad, pointer=False, loop=False)
        self.img = PatchTrunk(3, d, n_tokens=4, modality=0, pool=(2, 2))
        self.aud = PatchTrunk(1, d, n_tokens=8, modality=1, pool=(8, 1))
        self.txt = TextTrunk(vocab_size, d, pad=pad, n_tokens=8, heads=heads, modality=2)

    def forward(self, img, aud, txt, ids, mode="full"):
        ip, ap, tp = self.img(img), self.aud(aud), self.txt(txt)
        if mode == "text_only":
            ip, ap = ip * 0.0, ap * 0.0
        elif mode == "sensor_only":
            tp = tp * 0.0
        elif mode != "full":
            raise ValueError(f"unknown multimodal mode {mode!r}")
        prefix = torch.cat([ip, ap, tp], dim=1)              # B,20,d
        logits = self.lm(ids, prefix=prefix)
        return logits[:, prefix.shape[1]:]                  # token positions only


def build_vocab():
    toks = ["extract", "fact", "done", ".", "p0", "a0", "color", "shape", "pitch", "timbre",
            "env"]
    toks += list(COLOR_NAMES) + list(SHAPES) + list(PITCH_NAMES) + list(TIMBRES) + list(ENVELOPES)
    for tpl in TEXT_TRAIN + TEXT_EVAL:
        toks += [w for w in _split_words(tpl)
                 if not (w.startswith("{") and w.endswith("}"))]
    return Vocab(toks)


def sample_example(rng, text_split="train"):
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
    txt = render_transcript(gold, rng, split=text_split)
    return img, aud, txt, toks, gold


def _batch(n, rng, vocab, device, text_split="train"):
    imgs, auds, txts, seqs, golds = [], [], [], [], []
    for _ in range(n):
        img, aud, txt, toks, gold = sample_example(rng, text_split=text_split)
        imgs.append(img)
        auds.append(aud)
        txts.append(vocab.enc(txt))
        seqs.append(vocab.enc(toks))
        golds.append(gold)
    x = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
    a = torch.tensor(np.stack(auds), dtype=torch.float32, device=device)
    max_txt = max(len(t) for t in txts)
    t = torch.full((n, max_txt), vocab.pad, dtype=torch.long, device=device)
    for i, ids in enumerate(txts):
        t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    ids = torch.tensor(seqs, dtype=torch.long, device=device)   # fixed grammar = fixed length
    return x, a, t, ids, golds


VALUE_POS = {"color": 4, "shape": 9, "pitch": 14, "timbre": 19, "env": 24}  # token index of value


def evaluate(model, vocab, n=200, seed=1, device=DEV, text_split="eval", mode="full"):
    """Teacher-forced per-factor accuracy: at each fact's value position, does the model put
    the oracle value first among that factor's candidates?"""
    rng = np.random.default_rng(seed)
    model.eval()
    hits = {k: 0 for k in VALUE_POS}
    with torch.no_grad():
        for off in range(0, n, 50):
            b = min(50, n - off)
            img, aud, txt, ids, golds = _batch(b, rng, vocab, device, text_split=text_split)
            logits = model(img, aud, txt, ids, mode=mode)
            for k, pos in VALUE_POS.items():
                pred = logits[:, pos - 1].argmax(-1)        # next-token prediction of the value
                for r in range(b):
                    hits[k] += int(vocab.itos[int(pred[r])] == golds[r][k])
    return {k: v / n for k, v in hits.items()}


def parse_facts(tokens):
    out = {}
    try:
        i = tokens.index("extract") + 1
    except ValueError:
        i = 0
    while i < len(tokens):
        if tokens[i] == "done":
            break
        if i + 4 >= len(tokens) or tokens[i] != "fact" or tokens[i + 4] != ".":
            i += 1
            continue
        slot, pred, val = tokens[i + 1], tokens[i + 2], tokens[i + 3]
        if slot == "p0" and pred in ("color", "shape"):
            out[pred] = val
        elif slot == "a0" and pred in ("pitch", "timbre", "env"):
            out[pred] = val
        i += 5
    return out


def free_evaluate(model, vocab, n=50, seed=2, device=DEV, text_split="eval", mode="text_only",
                  max_new=40):
    rng = np.random.default_rng(seed)
    model.eval()
    hits = {k: 0 for k in VALUE_POS}
    exact = 0
    with torch.no_grad():
        for _ in range(n):
            img, aud, txt, _ids, golds = _batch(1, rng, vocab, device, text_split=text_split)
            gold = golds[0]
            ids = torch.tensor([[vocab.stoi["extract"]]], dtype=torch.long, device=device)
            for _step in range(max_new):
                logits = model(img, aud, txt, ids, mode=mode)
                nxt = int(logits[0, -1].argmax(-1))
                ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], 1)
                toks = vocab.dec([int(x) for x in ids[0]])
                if len(toks) >= 2 and toks[-2:] == ["done", "."]:
                    break
            got = parse_facts(vocab.dec([int(x) for x in ids[0]]))
            exact += all(got.get(k) == gold[k] for k in VALUE_POS)
            for k in VALUE_POS:
                hits[k] += int(got.get(k) == gold[k])
    out = {k: v / n for k, v in hits.items()}
    out["exact"] = exact / n
    return out


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
        img, aud, txt, ids, _ = _batch(batch, rng, vocab, device, text_split="train")
        losses = [token_loss(model(img, aud, txt, ids, mode=mode), ids, value_w=value_w)
                  for mode in MODES]
        loss = sum(losses) / len(losses)
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


def run(steps=400, seed=0, device=DEV, value_w=6.0, eval_n=200, free_n=40):
    model, vocab = train(steps=steps, seed=seed, device=device, value_w=value_w)
    full = evaluate(model, vocab, n=eval_n, device=device, text_split="eval", mode="full")
    text = evaluate(model, vocab, n=eval_n, device=device, text_split="eval", mode="text_only")
    sensor = evaluate(model, vocab, n=eval_n, device=device, text_split="eval", mode="sensor_only")
    free_text = free_evaluate(model, vocab, n=free_n, device=device, text_split="eval",
                              mode="text_only") if free_n else {}
    thresholds = gate_thresholds()
    report = {"experiment": "m0_multimodal_bridge", "steps": steps, "value_w": float(value_w),
              "eval_n": int(eval_n), "free_n": int(free_n),
              "teacher_forced": {"full": full, "text_only_eval_phrasings": text,
                                 "sensor_only": sensor},
              "free_text_only_eval_phrasings": free_text,
              "chance": CHANCE, "gate_thresholds": thresholds,
              "gate": all(full[k] >= thresholds[k] and text[k] >= thresholds[k]
                          and sensor[k] >= thresholds[k] for k in VALUE_POS)}
    print(json.dumps(report, indent=1), flush=True)
    return report


def selftest():
    rng = np.random.default_rng(0)
    vocab = build_vocab()
    img, aud, txt, toks, gold = sample_example(rng)
    txt_eval = render_transcript(gold, rng, split="eval")
    assert toks[VALUE_POS["color"]] == gold["color"] and toks[VALUE_POS["env"]] == gold["env"]
    assert all(t in vocab.stoi for t in toks + txt + txt_eval), "grammar/text token missing from vocab"
    model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad).to("cpu")
    x, a, tt, ids, _ = _batch(2, rng, vocab, "cpu")
    logits = model(x, a, tt, ids)
    assert logits.shape == (2, ids.shape[1], len(vocab)), logits.shape
    logits_text = model(x, a, tt, ids, mode="text_only")
    assert logits_text.shape == logits.shape
    th = gate_thresholds()
    assert all(CHANCE[k] < th[k] < 1.0 for k in CHANCE), th
    loss = token_loss(logits, ids)
    loss.backward()
    grads = [p.grad is not None and float(p.grad.abs().sum()) > 0
             for p in (model.img.proj.weight, model.aud.proj.weight, model.txt.emb.weight)]
    assert all(grads), "gradients must flow into all modality trunks"
    assert parse_facts(toks) == gold
    print("multimodal selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value-w", type=float, default=6.0, dest="value_w")
    ap.add_argument("--eval-n", type=int, default=200, dest="eval_n")
    ap.add_argument("--free-n", type=int, default=40, dest="free_n")
    ap.add_argument("--out", default="runs/m0_multimodal.json")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    report = run(steps=args.steps, seed=args.seed, value_w=args.value_w,
                 eval_n=args.eval_n, free_n=args.free_n)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
