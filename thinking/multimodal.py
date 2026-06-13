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
import math
import os
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from scratchpad_model import ScratchpadLM

from .audio import (ENVELOPES, PITCH_NAMES, TIMBRES, render_tone, sample_clip, spectrogram)
from .vision import COLORS, SHAPES, ObjectSpec, render_object, sample_object
from .trace import Vocab

DEV = get_device()
COLOR_NAMES = tuple(COLORS)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SURFACES = os.path.join(ROOT, "data", "multimodal_transcripts.json")
MODES = ("full", "sensor_only", "text_only")
TRUNK_ARCHES = ("conv", "residual")
FACTOR_VALUES = {
    "color": COLOR_NAMES,
    "shape": SHAPES,
    "pitch": PITCH_NAMES,
    "timbre": TIMBRES,
    "env": ENVELOPES,
}
FORMATTER = string.Formatter()


def _split_words(s):
    return s.split()


def _template_fields(tpl):
    fields = []
    for _literal, field, _format_spec, _conversion in FORMATTER.parse(tpl):
        if field is not None:
            fields.append(field)
    return fields


def _check_template_fields(path, split, tpl):
    required = set(FACTOR_VALUES)
    fields = set(_template_fields(tpl))
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{path}:{split} template missing placeholders {missing}: {tpl}")
    unsupported = sorted(fields - required)
    if unsupported:
        raise ValueError(f"{path}:{split} template has unsupported placeholders "
                         f"{sorted(unsupported)}: {tpl}")


def _check_gold(path, split, facts):
    missing = sorted(set(FACTOR_VALUES) - set(facts))
    if missing:
        raise ValueError(f"{path}:{split} example missing facts {missing}")
    gold = {k: facts[k] for k in FACTOR_VALUES}
    bad = {k: v for k, v in gold.items() if v not in FACTOR_VALUES[k]}
    if bad:
        raise ValueError(f"{path}:{split} example has invalid fact values {bad}")
    return gold


def _normalize_text_example(path, split, idx, rec):
    if not isinstance(rec, dict):
        raise ValueError(f"{path}:{split}_examples[{idx}] must be an object")
    if "tokens" in rec:
        tokens = list(rec["tokens"])
    elif "text" in rec:
        tokens = _split_words(rec["text"])
    else:
        raise ValueError(f"{path}:{split}_examples[{idx}] must contain text or tokens")
    if not tokens or not all(isinstance(t, str) for t in tokens):
        raise ValueError(f"{path}:{split}_examples[{idx}] has empty or non-string tokens")
    facts = rec.get("facts")
    if not isinstance(facts, dict):
        raise ValueError(f"{path}:{split}_examples[{idx}] must contain a facts object")
    return {"tokens": tokens, "facts": _check_gold(path, split, facts)}


def _load_examples(data, path, split):
    raw = data.get(f"{split}_examples", data.get("examples", {}).get(split, []))
    return [_normalize_text_example(path, split, i, rec) for i, rec in enumerate(raw)]


def load_text_surfaces(path=None):
    path = path or DEFAULT_SURFACES
    with open(path) as f:
        data = json.load(f)
    out = {"train": list(data.get("train", [])), "eval": list(data.get("eval", [])),
           "train_examples": _load_examples(data, path, "train"),
           "eval_examples": _load_examples(data, path, "eval")}
    for split in ("train", "eval"):
        if not out[split] and not out[f"{split}_examples"]:
            raise ValueError(f"{path} must contain {split} templates or {split}_examples")
    for split, bank in out.items():
        if split.endswith("_examples"):
            continue
        for tpl in bank:
            _check_template_fields(path, split, tpl)
    return out


def render_transcript(gold, rng, surfaces, split="train"):
    bank = surfaces[split]
    tpl = bank[int(rng.integers(len(bank)))]
    return _split_words(tpl.format(**gold))


def sample_gold(rng):
    obj = sample_object(rng, slot="p0")
    clip = sample_clip(rng)
    return {"color": obj.color, "shape": obj.shape, "pitch": clip["pitch"],
            "timbre": clip["timbre"], "env": clip["envelope"]}


def sample_text_and_gold(rng, surfaces, split="train"):
    templates = surfaces[split]
    examples = surfaces.get(f"{split}_examples", [])
    pick = int(rng.integers(len(templates) + len(examples)))
    if pick < len(examples):
        ex = examples[pick]
        return list(ex["tokens"]), dict(ex["facts"])
    gold = sample_gold(rng)
    tpl = templates[pick - len(examples)]
    return _split_words(tpl.format(**gold)), gold


def render_modalities(gold, rng):
    base = sample_object(rng, slot="p0")
    obj = ObjectSpec("p0", gold["color"], gold["shape"], x=base.x, y=base.y, scale=base.scale)
    clip = sample_clip(rng)
    img = render_object(obj, size=32)
    aud = spectrogram(render_tone(gold["pitch"], gold["timbre"], gold["env"],
                                  clip["detune"], clip["amp"], clip["phase"], rng=rng))
    return img, aud


def trace_tokens(gold):
    return ["extract",
            "fact", "p0", "color", gold["color"], ".",
            "fact", "p0", "shape", gold["shape"], ".",
            "fact", "a0", "pitch", gold["pitch"], ".",
            "fact", "a0", "timbre", gold["timbre"], ".",
            "fact", "a0", "env", gold["env"], ".",
            "done", "."]


def mutate_factor(gold, factor, rng):
    choices = [v for v in FACTOR_VALUES[factor] if v != gold[factor]]
    return {**gold, factor: choices[int(rng.integers(len(choices)))]}


def _grid_pool(n_tokens):
    n = int(n_tokens)
    if n <= 0:
        raise ValueError("prefix token count must be positive")
    h = int(math.sqrt(n))
    while h > 1 and n % h:
        h -= 1
    return h, n // h


class ResidualConvBlock(nn.Module):
    """Small residual conv block for multimodal sensory prefix encoders."""

    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class PatchTrunk(nn.Module):
    """Conv trunk -> a short sequence of d-dim prefix embeddings + modality embedding."""

    def __init__(self, in_ch, d, n_tokens=4, modality=0, n_modalities=3, pool=(2, 2),
                 arch="conv", width=64, depth=1):
        """pool: trunk output cells -> prefix tokens. Audio needs FREQUENCY-preserving pooling
        ((8,1): 8 frequency bands x collapsed time) -- 2x2 pooling crushed the frequency axis
        and pitch accuracy with it (0.32; every other factor >=0.93)."""
        super().__init__()
        arch = str(arch)
        if arch not in TRUNK_ARCHES:
            raise ValueError(f"unknown multimodal trunk arch {arch!r}")
        self.n_tokens = int(n_tokens)
        self.arch = arch
        self.width = int(width)
        self.depth = int(depth)
        self.pool = (int(pool[0]), int(pool[1]))
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("multimodal trunk width/depth must be positive")
        if self.n_tokens != self.pool[0] * self.pool[1]:
            raise ValueError("n_tokens must match the configured trunk pool cells")
        if arch == "conv":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(48, self.width, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(self.pool),
            )
        else:
            blocks = [
                nn.Conv2d(in_ch, self.width, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(self.width, self.width, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            ]
            for _ in range(self.depth):
                blocks.append(ResidualConvBlock(self.width))
            blocks.append(nn.AdaptiveAvgPool2d(self.pool))
            self.conv = nn.Sequential(*blocks)
        self.proj = nn.Linear(self.width, d)
        self.mod = nn.Embedding(n_modalities, d)
        self.modality = modality
        self.posn = nn.Parameter(torch.zeros(self.n_tokens, d))

    def forward(self, x):
        h = self.conv(x)                                    # B,64,2,2
        h = h.flatten(2).transpose(1, 2)                    # B,4,64
        return self.proj(h) + self.mod.weight[self.modality] + self.posn[None]


class TextTrunk(nn.Module):
    """Transcript encoder -> per-token prefix embeddings.  Pads are zeroed after encoding."""

    def __init__(self, vocab_size, d, pad=0, n_tokens=8, heads=4, layers=1, modality=2,
                 n_modalities=3, max_len=64):
        super().__init__()
        self.pad = pad
        self.modality = modality
        self.n_tokens = int(n_tokens)
        self.layers = int(layers)
        if self.n_tokens <= 0 or self.layers <= 0:
            raise ValueError("text trunk token/layer counts must be positive")
        self.emb = nn.Embedding(vocab_size, d, padding_idx=pad)
        self.pos = nn.Embedding(max_len, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                         dropout=0.0, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=self.layers,
                                         enable_nested_tensor=False)
        self.mod = nn.Embedding(n_modalities, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, ids):
        L = ids.shape[1]
        pos = torch.arange(L, device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        pad_mask = ids.eq(self.pad)
        h = self.emb(ids) + self.pos(pos)[None]
        h = self.enc(h, src_key_padding_mask=pad_mask)
        h = self.ln(h + self.mod.weight[self.modality])
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        if h.shape[1] > self.n_tokens:
            return h[:, :self.n_tokens]
        if h.shape[1] < self.n_tokens:
            pad = torch.zeros(h.shape[0], self.n_tokens - h.shape[1], h.shape[2],
                              dtype=h.dtype, device=h.device)
            h = torch.cat([h, pad], dim=1)
        return h


class MultimodalLM(nn.Module):
    """Image, audio, and transcript readers feeding one trace decoder."""

    def __init__(self, vocab_size, d=96, layers=3, heads=4, pad=0, max_len=128,
                 img_tokens=4, aud_tokens=8, txt_tokens=8, trunk_arch="conv",
                 trunk_width=64, trunk_depth=1, text_layers=1, modality_dropout=0.0):
        super().__init__()
        if img_tokens <= 0 or aud_tokens <= 0 or txt_tokens <= 0:
            raise ValueError("multimodal prefix token counts must be positive")
        if modality_dropout < 0.0 or modality_dropout > 1.0:
            raise ValueError("modality_dropout must be in [0, 1]")
        trunk_arch = str(trunk_arch)
        self.config = {
            "vocab_size": int(vocab_size), "d": int(d), "layers": int(layers),
            "heads": int(heads), "pad": int(pad), "max_len": int(max_len),
            "img_tokens": int(img_tokens), "aud_tokens": int(aud_tokens),
            "txt_tokens": int(txt_tokens), "trunk_arch": trunk_arch,
            "trunk_width": int(trunk_width), "trunk_depth": int(trunk_depth),
            "text_layers": int(text_layers), "modality_dropout": float(modality_dropout),
        }
        self.modality_dropout = float(modality_dropout)
        self.lm = ScratchpadLM(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                               pad=pad, pointer=False, loop=False)
        img_pool = _grid_pool(img_tokens)
        self.img = PatchTrunk(3, d, n_tokens=img_tokens, modality=0,
                              pool=img_pool, arch=trunk_arch,
                              width=trunk_width, depth=trunk_depth)
        self.aud = PatchTrunk(1, d, n_tokens=aud_tokens, modality=1,
                              pool=(aud_tokens, 1), arch=trunk_arch,
                              width=trunk_width, depth=trunk_depth)
        self.txt = TextTrunk(vocab_size, d, pad=pad, n_tokens=txt_tokens, heads=heads,
                             layers=text_layers, modality=2)

    def _apply_modality_dropout(self, ip, ap, tp):
        if not self.training or self.modality_dropout <= 0.0:
            return ip, ap, tp
        keep = 1.0 - self.modality_dropout
        if keep <= 0.0:
            return ip * 0.0, ap * 0.0, tp * 0.0
        masks = [
            torch.rand(ip.shape[0], 1, 1, device=ip.device).lt(keep).to(ip.dtype) / keep,
            torch.rand(ap.shape[0], 1, 1, device=ap.device).lt(keep).to(ap.dtype) / keep,
            torch.rand(tp.shape[0], 1, 1, device=tp.device).lt(keep).to(tp.dtype) / keep,
        ]
        return ip * masks[0], ap * masks[1], tp * masks[2]

    def forward(self, img, aud, txt, ids, mode="full"):
        ip, ap, tp = self.img(img), self.aud(aud), self.txt(txt)
        if mode == "text_only":
            ip, ap = ip * 0.0, ap * 0.0
        elif mode == "sensor_only":
            tp = tp * 0.0
        elif mode != "full":
            raise ValueError(f"unknown multimodal mode {mode!r}")
        elif self.modality_dropout > 0.0:
            ip, ap, tp = self._apply_modality_dropout(ip, ap, tp)
        prefix = torch.cat([ip, ap, tp], dim=1)
        logits = self.lm(ids, prefix=prefix)
        return logits[:, prefix.shape[1]:]                  # token positions only


def build_vocab(surfaces=None):
    surfaces = surfaces or load_text_surfaces()
    toks = ["extract", "fact", "done", ".", "p0", "a0", "color", "shape", "pitch", "timbre",
            "env"]
    toks += list(COLOR_NAMES) + list(SHAPES) + list(PITCH_NAMES) + list(TIMBRES) + list(ENVELOPES)
    for tpl in surfaces["train"] + surfaces["eval"]:
        toks += [w for w in _split_words(tpl)
                 if not (w.startswith("{") and w.endswith("}"))]
    for split in ("train_examples", "eval_examples"):
        for ex in surfaces.get(split, []):
            toks += list(ex["tokens"])
    return Vocab(toks)


def sample_example(rng, surfaces, text_split="train"):
    txt, gold = sample_text_and_gold(rng, surfaces, split=text_split)
    img, aud = render_modalities(gold, rng)
    toks = trace_tokens(gold)
    return img, aud, txt, toks, gold


def _batch(n, rng, vocab, device, surfaces, text_split="train"):
    imgs, auds, txts, seqs, golds = [], [], [], [], []
    for _ in range(n):
        img, aud, txt, toks, gold = sample_example(rng, surfaces, text_split=text_split)
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


def evaluate(model, vocab, surfaces, n=200, seed=1, device=DEV, text_split="eval", mode="full"):
    """Teacher-forced per-factor accuracy: at each fact's value position, does the model put
    the oracle value first among that factor's candidates?"""
    rng = np.random.default_rng(seed)
    model.eval()
    hits = {k: 0 for k in VALUE_POS}
    with torch.no_grad():
        for off in range(0, n, 50):
            b = min(50, n - off)
            img, aud, txt, ids, golds = _batch(b, rng, vocab, device, surfaces,
                                               text_split=text_split)
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


def greedy_facts(model, vocab, img, aud, txt, device, mode="text_only", max_new=40):
    ids = torch.tensor([[vocab.stoi["extract"]]], dtype=torch.long, device=device)
    for _step in range(max_new):
        logits = model(img, aud, txt, ids, mode=mode)
        nxt = int(logits[0, -1].argmax(-1))
        ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], 1)
        toks = vocab.dec([int(x) for x in ids[0]])
        if len(toks) >= 2 and toks[-2:] == ["done", "."]:
            break
    return parse_facts(vocab.dec([int(x) for x in ids[0]]))


def free_evaluate(model, vocab, surfaces, n=50, seed=2, device=DEV, text_split="eval",
                  mode="text_only", max_new=40):
    rng = np.random.default_rng(seed)
    model.eval()
    hits = {k: 0 for k in VALUE_POS}
    exact = 0
    with torch.no_grad():
        for _ in range(n):
            img, aud, txt, _ids, golds = _batch(1, rng, vocab, device, surfaces,
                                                text_split=text_split)
            gold = golds[0]
            got = greedy_facts(model, vocab, img, aud, txt, device, mode=mode, max_new=max_new)
            exact += all(got.get(k) == gold[k] for k in VALUE_POS)
            for k in VALUE_POS:
                hits[k] += int(got.get(k) == gold[k])
    out = {k: v / n for k, v in hits.items()}
    out["exact"] = exact / n
    return out


def _one_txt(tokens, vocab, device):
    ids = torch.tensor([vocab.enc(tokens)], dtype=torch.long, device=device)
    return ids


def counterfactual_text_evaluate(model, vocab, surfaces, n=40, seed=3, device=DEV):
    """Text-only intervention test: mutate one described factor in a held-out transcript.

    The model sees no image/audio evidence.  The changed factor should follow the edited words,
    while all unedited factors should remain stable.  This is a semantic binding check, not a
    caption next-token task.
    """
    if not surfaces.get("eval"):
        return {}
    rng = np.random.default_rng(seed)
    model.eval()
    by_factor = {}
    with torch.no_grad():
        for factor in VALUE_POS:
            target = collateral = exact = total_other = 0
            for _ in range(n):
                img, aud, _txt, _ids, golds = _batch(1, rng, vocab, device, surfaces,
                                                     text_split="eval")
                gold = golds[0]
                edited = mutate_factor(gold, factor, rng)
                txt = _one_txt(render_transcript(edited, rng, surfaces, split="eval"), vocab,
                               device)
                ids = torch.tensor([vocab.enc(trace_tokens(edited))], dtype=torch.long,
                                   device=device)
                logits = model(img, aud, txt, ids, mode="text_only")
                preds = {k: vocab.itos[int(logits[:, pos - 1].argmax(-1)[0])]
                         for k, pos in VALUE_POS.items()}
                target += int(preds[factor] == edited[factor])
                exact += int(all(preds[k] == edited[k] for k in VALUE_POS))
                for other in VALUE_POS:
                    if other == factor:
                        continue
                    collateral += int(preds[other] == gold[other])
                    total_other += 1
            by_factor[factor] = {
                "target_acc": target / n,
                "collateral_acc": collateral / max(1, total_other),
                "exact": exact / n,
            }
    means = {
        "target_acc": float(np.mean([v["target_acc"] for v in by_factor.values()])),
        "collateral_acc": float(np.mean([v["collateral_acc"] for v in by_factor.values()])),
        "exact": float(np.mean([v["exact"] for v in by_factor.values()])),
    }
    return {"by_factor": by_factor, "mean": means}


def free_counterfactual_text_evaluate(model, vocab, surfaces, n=20, seed=4, device=DEV,
                                      max_new=40):
    """Free decode version of the text intervention test."""
    if not surfaces.get("eval"):
        return {}
    rng = np.random.default_rng(seed)
    model.eval()
    by_factor = {}
    with torch.no_grad():
        for factor in VALUE_POS:
            target = collateral = exact = total_other = 0
            for _ in range(n):
                img, aud, _txt, _ids, golds = _batch(1, rng, vocab, device, surfaces,
                                                     text_split="eval")
                gold = golds[0]
                edited = mutate_factor(gold, factor, rng)
                txt = _one_txt(render_transcript(edited, rng, surfaces, split="eval"), vocab,
                               device)
                got = greedy_facts(model, vocab, img, aud, txt, device, mode="text_only",
                                   max_new=max_new)
                target += int(got.get(factor) == edited[factor])
                exact += int(all(got.get(k) == edited[k] for k in VALUE_POS))
                for other in VALUE_POS:
                    if other == factor:
                        continue
                    collateral += int(got.get(other) == gold[other])
                    total_other += 1
            by_factor[factor] = {
                "target_acc": target / n,
                "collateral_acc": collateral / max(1, total_other),
                "exact": exact / n,
            }
    means = {
        "target_acc": float(np.mean([v["target_acc"] for v in by_factor.values()])),
        "collateral_acc": float(np.mean([v["collateral_acc"] for v in by_factor.values()])),
        "exact": float(np.mean([v["exact"] for v in by_factor.values()])),
    }
    return {"by_factor": by_factor, "mean": means}


def token_loss(logits, ids, value_w=6.0):
    targets = ids[:, 1:]
    raw = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1),
                          reduction="none").view_as(targets)
    weights = torch.ones_like(raw)
    for pos in VALUE_POS.values():
        weights[:, pos - 1] = value_w                   # logit at pos-1 predicts value at pos
    return (raw * weights).sum() / weights.sum()


def _candidate_ids(vocab, factor, device):
    return torch.tensor([vocab.stoi[v] for v in FACTOR_VALUES[factor]],
                        dtype=torch.long, device=device)


def value_agreement_loss(logits_by_mode, vocab):
    """Align factor-value distributions across full, sensor-only, and text-only paths."""
    items = list(logits_by_mode.items())
    if len(items) < 2:
        return next(iter(logits_by_mode.values())).sum() * 0.0
    losses = []
    device = items[0][1].device
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            _mode_a, logits_a = items[i]
            _mode_b, logits_b = items[j]
            for factor, pos in VALUE_POS.items():
                cand = _candidate_ids(vocab, factor, device)
                a = logits_a[:, pos - 1].index_select(-1, cand)
                b = logits_b[:, pos - 1].index_select(-1, cand)
                losses.append(0.5 * (
                    F.kl_div(F.log_softmax(a, dim=-1), F.softmax(b.detach(), dim=-1),
                             reduction="batchmean")
                    + F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a.detach(), dim=-1),
                               reduction="batchmean")
                ))
    return torch.stack(losses).mean()


def train(steps=400, batch=32, d=96, lr=1e-3, seed=0, device=DEV, log_every=100, value_w=6.0,
          surfaces_path=None, layers=3, heads=4, max_len=128, img_tokens=4, aud_tokens=8,
          txt_tokens=8, trunk_arch="conv", trunk_width=64, trunk_depth=1, text_layers=1,
          modality_dropout=0.0, agreement_w=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    surfaces = load_text_surfaces(surfaces_path)
    vocab = build_vocab(surfaces)
    model = MultimodalLM(len(vocab), d=d, layers=layers, heads=heads, pad=vocab.pad,
                         max_len=max_len, img_tokens=img_tokens, aud_tokens=aud_tokens,
                         txt_tokens=txt_tokens, trunk_arch=trunk_arch, trunk_width=trunk_width,
                         trunk_depth=trunk_depth, text_layers=text_layers,
                         modality_dropout=modality_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    last_base = last_agreement = 0.0
    for st in range(1, steps + 1):
        model.train()
        img, aud, txt, ids, _ = _batch(batch, rng, vocab, device, surfaces, text_split="train")
        logits_by_mode = {mode: model(img, aud, txt, ids, mode=mode) for mode in MODES}
        losses = [token_loss(logits, ids, value_w=value_w)
                  for logits in logits_by_mode.values()]
        base_loss = sum(losses) / len(losses)
        agreement = (value_agreement_loss(logits_by_mode, vocab)
                     if agreement_w else base_loss * 0.0)
        loss = base_loss + float(agreement_w) * agreement
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_base = float(base_loss.detach())
        last_agreement = float(agreement.detach())
        if st % log_every == 0 or st == steps:
            print(f"  m0 {st}/{steps} loss {loss.item():.3f} "
                  f"base {last_base:.3f} agree {last_agreement:.3f}", flush=True)
    model.train_metrics = {"token_loss": last_base, "agreement_loss": last_agreement}
    return model, vocab, surfaces


CHANCE = {"color": 1 / len(COLORS), "shape": 1 / len(SHAPES),
          "pitch": 1 / len(PITCH_NAMES), "timbre": 1 / len(TIMBRES),
          "env": 1 / len(ENVELOPES)}


def gate_thresholds(chance=CHANCE, gap_frac=0.40):
    """Require each factor to close a fixed fraction of the gap above chance.

    A raw multiplier such as 3x chance is impossible for 3-way factors because the threshold
    exceeds 1.0.  Gap-normalized thresholds compare factors with different cardinalities fairly.
    """
    return {k: v + gap_frac * (1.0 - v) for k, v in chance.items()}


def run(steps=400, seed=0, device=DEV, value_w=6.0, eval_n=200, free_n=40,
        counterfactual_n=40, free_counterfactual_n=20, surfaces_path=None, checkpoint=None,
        batch=32, d=96, lr=1e-3, layers=3, heads=4, max_len=128, img_tokens=4, aud_tokens=8,
        txt_tokens=8, trunk_arch="conv", trunk_width=64, trunk_depth=1, text_layers=1,
        modality_dropout=0.0, agreement_w=0.0, log_every=100):
    model, vocab, surfaces = train(steps=steps, seed=seed, device=device, value_w=value_w,
                                   surfaces_path=surfaces_path, batch=batch, d=d, lr=lr,
                                   layers=layers, heads=heads, max_len=max_len,
                                   img_tokens=img_tokens, aud_tokens=aud_tokens,
                                   txt_tokens=txt_tokens, trunk_arch=trunk_arch,
                                   trunk_width=trunk_width, trunk_depth=trunk_depth,
                                   text_layers=text_layers,
                                   modality_dropout=modality_dropout,
                                   agreement_w=agreement_w, log_every=log_every)
    full = evaluate(model, vocab, surfaces, n=eval_n, device=device, text_split="eval",
                    mode="full")
    text = evaluate(model, vocab, surfaces, n=eval_n, device=device, text_split="eval",
                    mode="text_only")
    sensor = evaluate(model, vocab, surfaces, n=eval_n, device=device, text_split="eval",
                      mode="sensor_only")
    free_text = free_evaluate(model, vocab, surfaces, n=free_n, device=device, text_split="eval",
                              mode="text_only") if free_n else {}
    counterfactual = counterfactual_text_evaluate(
        model, vocab, surfaces, n=counterfactual_n, device=device) if counterfactual_n else {}
    free_counterfactual = free_counterfactual_text_evaluate(
        model, vocab, surfaces, n=free_counterfactual_n, device=device
    ) if free_counterfactual_n else {}
    thresholds = gate_thresholds()
    architecture = dict(model.config)
    architecture["img_pool"] = list(model.img.pool)
    architecture["aud_pool"] = list(model.aud.pool)
    architecture["prefix_tokens"] = int(img_tokens) + int(aud_tokens) + int(txt_tokens)
    report = {"experiment": "m0_multimodal_bridge", "steps": steps, "batch": int(batch),
              "lr": float(lr), "value_w": float(value_w),
              "agreement_w": float(agreement_w),
              "train_metrics": getattr(model, "train_metrics", {}),
              "architecture": architecture,
              "eval_n": int(eval_n), "free_n": int(free_n),
              "counterfactual_n": int(counterfactual_n),
              "free_counterfactual_n": int(free_counterfactual_n),
              "text_surfaces": {"path": surfaces_path or DEFAULT_SURFACES,
                                "train_templates": len(surfaces["train"]),
                                "eval_templates": len(surfaces["eval"]),
                                "train_examples": len(surfaces.get("train_examples", [])),
                                "eval_examples": len(surfaces.get("eval_examples", []))},
              "teacher_forced": {"full": full, "text_only_eval_phrasings": text,
                                 "sensor_only": sensor},
              "free_text_only_eval_phrasings": free_text,
              "counterfactual_text_only_eval_phrasings": counterfactual,
              "free_counterfactual_text_only_eval_phrasings": free_counterfactual,
              "chance": CHANCE, "gate_thresholds": thresholds,
              "gate": all(full[k] >= thresholds[k] and text[k] >= thresholds[k]
                          and sensor[k] >= thresholds[k] for k in VALUE_POS)
              and (not counterfactual or (
                  all(counterfactual["by_factor"][k]["target_acc"] >= thresholds[k]
                      for k in VALUE_POS)
                  and counterfactual["mean"]["collateral_acc"] >= 0.80))
              and (not free_counterfactual or (
                  free_counterfactual["mean"]["target_acc"] >= 0.70
                  and free_counterfactual["mean"]["collateral_acc"] >= 0.70))}
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos,
                    "d": model.lm.tok.embedding_dim, "model_config": model.config,
                    "text_surfaces": report["text_surfaces"], "report": report}, checkpoint)
        report["checkpoint"] = checkpoint
    print(json.dumps(report, indent=1), flush=True)
    return report


def selftest():
    rng = np.random.default_rng(0)
    surfaces = load_text_surfaces()
    vocab = build_vocab(surfaces)
    img, aud, txt, toks, gold = sample_example(rng, surfaces)
    txt_eval = render_transcript(gold, rng, surfaces, split="eval")
    assert toks[VALUE_POS["color"]] == gold["color"] and toks[VALUE_POS["env"]] == gold["env"]
    assert all(t in vocab.stoi for t in toks + txt + txt_eval), \
        "grammar/text token missing from vocab"
    example_surfaces = {
        "train": [],
        "eval": [],
        "train_examples": [{"tokens": ["external", "sentence", "says", "red", "circle"],
                            "facts": {"color": "red", "shape": "circle", "pitch": "n440",
                                      "timbre": "saw", "env": "decay"}}],
        "eval_examples": [{"tokens": ["external", "sentence", "says", "blue", "square"],
                           "facts": {"color": "blue", "shape": "square", "pitch": "n262",
                                     "timbre": "sine", "env": "flat"}}],
    }
    ex_vocab = build_vocab(example_surfaces)
    _img2, _aud2, ex_txt, ex_toks, ex_gold = sample_example(
        np.random.default_rng(1), example_surfaces)
    assert ex_gold["color"] == "red" and ex_txt[0] == "external"
    assert all(t in ex_vocab.stoi for t in ex_txt + ex_toks)
    model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad).to("cpu")
    x, a, tt, ids, _ = _batch(2, rng, vocab, "cpu", surfaces)
    assert _grid_pool(4) == (2, 2) and _grid_pool(8) == (2, 4)
    logits = model(x, a, tt, ids)
    assert logits.shape == (2, ids.shape[1], len(vocab)), logits.shape
    logits_text = model(x, a, tt, ids, mode="text_only")
    assert logits_text.shape == logits.shape
    res_model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad,
                             img_tokens=4, aud_tokens=8, txt_tokens=6,
                             trunk_arch="residual", trunk_width=32, trunk_depth=1,
                             text_layers=2, modality_dropout=0.1).to("cpu")
    assert res_model.img.arch == "residual" and res_model.img.pool == (2, 2)
    assert res_model.txt.layers == 2 and res_model.txt.n_tokens == 6
    res_logits = {mode: res_model(x, a, tt, ids, mode=mode) for mode in MODES}
    assert all(v.shape == logits.shape for v in res_logits.values())
    agree = value_agreement_loss(res_logits, vocab)
    assert torch.isfinite(agree), agree
    th = gate_thresholds()
    assert all(CHANCE[k] < th[k] < 1.0 for k in CHANCE), th
    loss = token_loss(logits, ids)
    loss.backward()
    grads = [p.grad is not None and float(p.grad.abs().sum()) > 0
             for p in (model.img.proj.weight, model.aud.proj.weight, model.txt.emb.weight)]
    assert all(grads), "gradients must flow into all modality trunks"
    assert parse_facts(toks) == gold
    cf = counterfactual_text_evaluate(model, vocab, surfaces, n=1, device="cpu")
    assert set(cf["by_factor"]) == set(VALUE_POS) and "target_acc" in cf["mean"]
    fcf = free_counterfactual_text_evaluate(model, vocab, surfaces, n=1, device="cpu")
    assert set(fcf["by_factor"]) == set(VALUE_POS) and "target_acc" in fcf["mean"]
    print("multimodal selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=DEV)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=96, dest="d")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=128, dest="max_len")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=100, dest="log_every")
    ap.add_argument("--value-w", type=float, default=6.0, dest="value_w")
    ap.add_argument("--agreement-w", type=float, default=0.0, dest="agreement_w",
                    help="cross-mode factor-value distribution agreement loss weight")
    ap.add_argument("--img-tokens", type=int, default=4, dest="img_tokens")
    ap.add_argument("--aud-tokens", type=int, default=8, dest="aud_tokens")
    ap.add_argument("--txt-tokens", type=int, default=8, dest="txt_tokens")
    ap.add_argument("--trunk-arch", default="conv", choices=TRUNK_ARCHES, dest="trunk_arch")
    ap.add_argument("--trunk-width", type=int, default=64, dest="trunk_width")
    ap.add_argument("--trunk-depth", type=int, default=1, dest="trunk_depth")
    ap.add_argument("--text-layers", type=int, default=1, dest="text_layers")
    ap.add_argument("--modality-dropout", type=float, default=0.0, dest="modality_dropout")
    ap.add_argument("--eval-n", type=int, default=200, dest="eval_n")
    ap.add_argument("--free-n", type=int, default=40, dest="free_n")
    ap.add_argument("--counterfactual-n", type=int, default=40, dest="counterfactual_n")
    ap.add_argument("--free-counterfactual-n", type=int, default=20,
                    dest="free_counterfactual_n")
    ap.add_argument("--surfaces", default=None,
                    help="JSON transcript surface bank with train/eval template lists")
    ap.add_argument("--checkpoint", default=None,
                    help="optional .pt checkpoint path for model weights + vocab + report")
    ap.add_argument("--out", default="runs/m0_multimodal.json")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    positive = {
        "--steps": args.steps, "--batch": args.batch, "--dim": args.d,
        "--layers": args.layers, "--heads": args.heads, "--max-len": args.max_len,
        "--log-every": args.log_every, "--img-tokens": args.img_tokens,
        "--aud-tokens": args.aud_tokens, "--txt-tokens": args.txt_tokens,
        "--trunk-width": args.trunk_width, "--trunk-depth": args.trunk_depth,
        "--text-layers": args.text_layers,
    }
    for name, value in positive.items():
        if value <= 0:
            ap.error(f"{name} must be positive")
    if args.lr <= 0.0:
        ap.error("--lr must be positive")
    if args.d % args.heads != 0:
        ap.error("--dim must be divisible by --heads")
    if (args.d // args.heads) % 2 != 0:
        ap.error("--dim / --heads must be even for rope attention")
    if args.agreement_w < 0.0:
        ap.error("--agreement-w must be non-negative")
    if args.modality_dropout < 0.0 or args.modality_dropout > 1.0:
        ap.error("--modality-dropout must be in [0, 1]")
    if args.eval_n <= 0:
        ap.error("--eval-n must be positive")
    if args.free_n < 0 or args.counterfactual_n < 0 or args.free_counterfactual_n < 0:
        ap.error("free/counterfactual eval counts must be non-negative")
    report = run(steps=args.steps, seed=args.seed, value_w=args.value_w,
                 eval_n=args.eval_n, free_n=args.free_n,
                 counterfactual_n=args.counterfactual_n,
                 free_counterfactual_n=args.free_counterfactual_n,
                 surfaces_path=args.surfaces,
                 checkpoint=args.checkpoint, batch=args.batch, d=args.d, lr=args.lr,
                 layers=args.layers, heads=args.heads, max_len=args.max_len,
                 img_tokens=args.img_tokens, aud_tokens=args.aud_tokens,
                 txt_tokens=args.txt_tokens, trunk_arch=args.trunk_arch,
                 trunk_width=args.trunk_width, trunk_depth=args.trunk_depth,
                 text_layers=args.text_layers, modality_dropout=args.modality_dropout,
                 agreement_w=args.agreement_w, log_every=args.log_every,
                 device=args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
