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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SURFACES = os.path.join(ROOT, "data", "multimodal_transcripts.json")
MODES = ("full", "sensor_only", "text_only")
FACTOR_VALUES = {
    "color": COLOR_NAMES,
    "shape": SHAPES,
    "pitch": PITCH_NAMES,
    "timbre": TIMBRES,
    "env": ENVELOPES,
}


def _split_words(s):
    return s.split()


def load_text_surfaces(path=None):
    path = path or DEFAULT_SURFACES
    with open(path) as f:
        data = json.load(f)
    out = {"train": list(data.get("train", [])), "eval": list(data.get("eval", []))}
    if not out["train"] or not out["eval"]:
        raise ValueError(f"{path} must contain non-empty train/eval transcript templates")
    required = {"color", "shape", "pitch", "timbre", "env"}
    for split, bank in out.items():
        for tpl in bank:
            missing = [k for k in required if "{" + k + "}" not in tpl]
            if missing:
                raise ValueError(f"{path}:{split} template missing placeholders {missing}: {tpl}")
    return out


def render_transcript(gold, rng, surfaces, split="train"):
    bank = surfaces[split]
    tpl = bank[int(rng.integers(len(bank)))]
    return _split_words(tpl.format(**gold))


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
    """Transcript encoder -> per-token prefix embeddings.  Pads are zeroed after encoding."""

    def __init__(self, vocab_size, d, pad=0, n_tokens=8, heads=4, modality=2, n_modalities=3,
                 max_len=64):
        super().__init__()
        self.pad = pad
        self.modality = modality
        self.emb = nn.Embedding(vocab_size, d, padding_idx=pad)
        self.pos = nn.Embedding(max_len, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                         dropout=0.0, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=1, enable_nested_tensor=False)
        self.mod = nn.Embedding(n_modalities, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, ids):
        L = ids.shape[1]
        pos = torch.arange(L, device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        pad_mask = ids.eq(self.pad)
        h = self.emb(ids) + self.pos(pos)[None]
        h = self.enc(h, src_key_padding_mask=pad_mask)
        h = self.ln(h + self.mod.weight[self.modality])
        return h.masked_fill(pad_mask.unsqueeze(-1), 0.0)


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
    return Vocab(toks)


def sample_example(rng, surfaces, text_split="train"):
    obj = sample_object(rng, slot="p0")
    clip = sample_clip(rng)
    img = render_object(obj, size=32)
    aud = spectrogram(render_tone(clip["pitch"], clip["timbre"], clip["envelope"],
                                  clip["detune"], clip["amp"], clip["phase"], rng=rng))
    gold = {"color": obj.color, "shape": obj.shape, "pitch": clip["pitch"],
            "timbre": clip["timbre"], "env": clip["envelope"]}
    toks = trace_tokens(gold)
    txt = render_transcript(gold, rng, surfaces, split=text_split)
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


def train(steps=400, batch=32, d=96, lr=1e-3, seed=0, device=DEV, log_every=100, value_w=6.0,
          surfaces_path=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    surfaces = load_text_surfaces(surfaces_path)
    vocab = build_vocab(surfaces)
    model = MultimodalLM(len(vocab), d=d, pad=vocab.pad).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        model.train()
        img, aud, txt, ids, _ = _batch(batch, rng, vocab, device, surfaces, text_split="train")
        losses = [token_loss(model(img, aud, txt, ids, mode=mode), ids, value_w=value_w)
                  for mode in MODES]
        loss = sum(losses) / len(losses)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % log_every == 0 or st == steps:
            print(f"  m0 {st}/{steps} loss {loss.item():.3f}", flush=True)
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
        counterfactual_n=40, free_counterfactual_n=20, surfaces_path=None, checkpoint=None):
    model, vocab, surfaces = train(steps=steps, seed=seed, device=device, value_w=value_w,
                                   surfaces_path=surfaces_path)
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
    report = {"experiment": "m0_multimodal_bridge", "steps": steps, "value_w": float(value_w),
              "eval_n": int(eval_n), "free_n": int(free_n),
              "counterfactual_n": int(counterfactual_n),
              "free_counterfactual_n": int(free_counterfactual_n),
              "text_surfaces": {"path": surfaces_path or DEFAULT_SURFACES,
                                "train": len(surfaces["train"]),
                                "eval": len(surfaces["eval"])},
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
                    "d": model.lm.tok.embedding_dim,
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
    assert all(t in vocab.stoi for t in toks + txt + txt_eval), "grammar/text token missing from vocab"
    model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad).to("cpu")
    x, a, tt, ids, _ = _batch(2, rng, vocab, "cpu", surfaces)
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
    ap.add_argument("--value-w", type=float, default=6.0, dest="value_w")
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
    report = run(steps=args.steps, seed=args.seed, value_w=args.value_w,
                 eval_n=args.eval_n, free_n=args.free_n,
                 counterfactual_n=args.counterfactual_n,
                 free_counterfactual_n=args.free_counterfactual_n,
                 surfaces_path=args.surfaces,
                 checkpoint=args.checkpoint)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
