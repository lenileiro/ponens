"""Text-0: data-driven language understanding -> canonical facts.

This module is the text-only rung that M-0 was missing.  It does not contain English
interpretation rules.  It learns from records that pair natural language with canonical facts,
then evaluates the model by decoded facts, paraphrase consistency, and explicit counterfactual
records supplied by the dataset.

Record formats accepted by --data:

  JSONL:
    {"split":"train","id":"r1","text":"...","facts":[["p0","color","red"]]}

  JSON:
    {"train":[...], "eval":[...]}
    {"records":[...]}

Fields:
  text or tokens        natural-language input
  facts                 list of [slot, predicate, value] triples
  group                 optional paraphrase group id; records in a group should share facts
  kind/base_id/changed  optional metadata for dataset-supplied counterfactual evals

The model receives the text as a continuous prefix and must emit:
  extract fact p0 color red . fact ... done .

Example:
  python -m thinking.text --selftest
  python -m thinking.text --data data/text_semantics.jsonl --steps 2000 \
      --out runs/text0.json --checkpoint runs/text0.pt
"""
import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from scratchpad_model import ScratchpadLM

from .trace import Vocab

DEV = get_device()
KEYWORDS = ("extract", "fact", "done", ".")


@dataclass(frozen=True)
class TextRecord:
    rec_id: str
    split: str
    tokens: tuple[str, ...]
    facts: tuple[tuple[str, str, str], ...]
    group: str | None = None
    kind: str = "plain"
    base_id: str | None = None
    changed: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)


def split_words(text):
    return str(text).strip().split()


def normalize_fact(raw, where="record"):
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{where}: facts must be [slot, predicate, value] triples")
    slot, pred, val = raw
    if not all(isinstance(x, str) and x for x in (slot, pred, val)):
        raise ValueError(f"{where}: fact fields must be non-empty strings")
    return (slot, pred, val)


def normalize_record(raw, default_split=None, idx=0):
    if not isinstance(raw, dict):
        raise ValueError(f"record {idx} must be an object")
    split = raw.get("split", default_split)
    if split not in ("train", "eval"):
        raise ValueError(f"record {idx} has invalid split {split!r}")
    if "tokens" in raw:
        tokens = tuple(raw["tokens"])
    elif "text" in raw:
        tokens = tuple(split_words(raw["text"]))
    else:
        raise ValueError(f"record {idx} must contain text or tokens")
    if not tokens or not all(isinstance(t, str) and t for t in tokens):
        raise ValueError(f"record {idx} has empty/non-string tokens")
    facts = tuple(normalize_fact(f, f"record {idx}") for f in raw.get("facts", []))
    if not facts:
        raise ValueError(f"record {idx} must contain at least one fact")
    changed = tuple(normalize_fact(f, f"record {idx}.changed")
                    for f in raw.get("changed", []))
    return TextRecord(
        rec_id=str(raw.get("id", f"{split}-{idx}")),
        split=split,
        tokens=tokens,
        facts=facts,
        group=raw.get("group"),
        kind=str(raw.get("kind", "plain")),
        base_id=raw.get("base_id"),
        changed=changed,
    )


def load_records(path):
    with open(path) as f:
        txt = f.read()
    if path.endswith(".jsonl"):
        records = [normalize_record(json.loads(line), idx=i)
                   for i, line in enumerate(txt.splitlines()) if line.strip()]
    else:
        data = json.loads(txt)
        records = []
        if isinstance(data, list):
            records = [normalize_record(r, idx=i) for i, r in enumerate(data)]
        elif "records" in data:
            records = [normalize_record(r, idx=i) for i, r in enumerate(data["records"])]
        else:
            for split in ("train", "eval"):
                records.extend(normalize_record(r, default_split=split, idx=i)
                               for i, r in enumerate(data.get(split, [])))
    if not any(r.split == "train" for r in records):
        raise ValueError(f"{path} has no train records")
    if not any(r.split == "eval" for r in records):
        raise ValueError(f"{path} has no eval records")
    return records


def trace_tokens(facts):
    toks = ["extract"]
    for slot, pred, val in facts:
        toks += ["fact", slot, pred, val, "."]
    return toks + ["done", "."]


def parse_facts(tokens):
    out = []
    try:
        i = tokens.index("extract") + 1
    except ValueError:
        i = 0
    while i < len(tokens):
        if tokens[i] == "done":
            break
        if i + 4 < len(tokens) and tokens[i] == "fact" and tokens[i + 4] == ".":
            out.append((tokens[i + 1], tokens[i + 2], tokens[i + 3]))
            i += 5
        else:
            i += 1
    return tuple(out)


def build_vocab(records):
    toks = list(KEYWORDS)
    for rec in records:
        toks += list(rec.tokens)
        for fact in rec.facts + rec.changed:
            toks += list(fact)
    return Vocab(toks)


class TextPrefix(nn.Module):
    """Bidirectional text encoder producing continuous prefix embeddings."""

    def __init__(self, vocab_size, d, pad=0, heads=4, max_len=256):
        super().__init__()
        self.pad = pad
        self.emb = nn.Embedding(vocab_size, d, padding_idx=pad)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=0.0,
            activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def forward(self, ids):
        L = ids.shape[1]
        pos = torch.arange(L, device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        mask = ids.eq(self.pad)
        h = self.emb(ids) + self.pos(pos)[None]
        h = self.enc(h, src_key_padding_mask=mask)
        return self.ln(h).masked_fill(mask.unsqueeze(-1), 0.0)


class TextFactLM(nn.Module):
    """Text prefix -> canonical fact trace decoder."""

    def __init__(self, vocab_size, d=96, layers=3, heads=4, pad=0, max_len=512):
        super().__init__()
        self.txt = TextPrefix(vocab_size, d=d, pad=pad, heads=heads)
        self.lm = ScratchpadLM(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                               pad=pad, pointer=False, loop=False)

    def forward(self, txt, ids):
        prefix = self.txt(txt)
        logits = self.lm(ids, prefix=prefix)
        return logits[:, prefix.shape[1]:]


def pack(records, vocab, device):
    text_ids = [vocab.enc(r.tokens) for r in records]
    trace_ids = [vocab.enc(trace_tokens(r.facts)) for r in records]
    max_t = max(len(x) for x in text_ids)
    max_y = max(len(x) for x in trace_ids)
    txt = torch.full((len(records), max_t), vocab.pad, dtype=torch.long, device=device)
    y = torch.full((len(records), max_y), vocab.pad, dtype=torch.long, device=device)
    for i, ids in enumerate(text_ids):
        txt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    for i, ids in enumerate(trace_ids):
        y[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return txt, y


def batch_records(records, rng, batch):
    return [records[int(rng.integers(len(records)))] for _ in range(batch)]


def token_loss(logits, ids, pad=0):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           ids[:, 1:].reshape(-1), ignore_index=pad)


def train_model(records, steps=400, batch=32, d=96, lr=1e-3, seed=0, device=DEV,
                log_every=100):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = build_vocab(records)
    model = TextFactLM(len(vocab), d=d, pad=vocab.pad).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_records = [r for r in records if r.split == "train"]
    for st in range(1, steps + 1):
        model.train()
        batch = batch_records(train_records, rng, batch)
        txt, ids = pack(batch, vocab, device)
        loss = token_loss(model(txt, ids), ids, pad=vocab.pad)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % log_every == 0 or st == steps:
            print(f"  text {st}/{steps} loss {loss.item():.3f}", flush=True)
    return model, vocab


def fact_scores(pred, gold):
    p, g = set(pred), set(gold)
    tp = len(p & g)
    precision = tp / max(1, len(p))
    recall = tp / max(1, len(g))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "exact": float(p == g)}


def teacher_forced_eval(model, vocab, records, device=DEV):
    eval_records = [r for r in records if r.split == "eval"]
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for off in range(0, len(eval_records), 64):
            batch = eval_records[off:off + 64]
            txt, ids = pack(batch, vocab, device)
            logits = model(txt, ids)
            for r, rec in enumerate(batch):
                pos = 4
                for _fact in rec.facts:
                    pred = int(logits[r, pos - 1].argmax(-1))
                    gold = int(ids[r, pos])
                    correct += int(pred == gold)
                    total += 1
                    pos += 5
    return {"fact_value_acc": correct / max(1, total), "n_facts": total}


@torch.no_grad()
def greedy_facts(model, vocab, rec, device=DEV, max_new=80):
    model.eval()
    txt, _ids = pack([rec], vocab, device)
    ids = torch.tensor([[vocab.stoi["extract"]]], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits = model(txt, ids)
        nxt = int(logits[0, -1].argmax(-1))
        ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], 1)
        toks = vocab.dec([int(x) for x in ids[0]])
        if len(toks) >= 2 and toks[-2:] == ["done", "."]:
            break
    return parse_facts(vocab.dec([int(x) for x in ids[0]]))


def free_eval(model, vocab, records, device=DEV, max_new=80):
    rows = []
    for rec in [r for r in records if r.split == "eval"]:
        pred = greedy_facts(model, vocab, rec, device=device, max_new=max_new)
        rows.append(fact_scores(pred, rec.facts))
    keys = ("precision", "recall", "f1", "exact")
    return {k: float(np.mean([r[k] for r in rows])) if rows else 0.0 for k in keys} | {
        "n": len(rows)}


def paraphrase_eval(model, vocab, records, device=DEV, max_new=80):
    groups = {}
    for rec in records:
        if rec.split == "eval" and rec.group:
            groups.setdefault(rec.group, []).append(rec)
    usable = [g for g in groups.values() if len(g) >= 2]
    consistent = exact = 0
    for group in usable:
        preds = [set(greedy_facts(model, vocab, rec, device=device, max_new=max_new))
                 for rec in group]
        golds = [set(rec.facts) for rec in group]
        consistent += int(all(p == preds[0] for p in preds[1:]))
        exact += int(all(p == g for p, g in zip(preds, golds)))
    return {"n_groups": len(usable),
            "consistent": consistent / max(1, len(usable)),
            "exact": exact / max(1, len(usable))}


def counterfactual_eval(model, vocab, records, device=DEV, max_new=80):
    cf = [r for r in records if r.split == "eval"
          and (r.kind == "counterfactual" or r.base_id or r.changed)]
    if not cf:
        return {"n": 0, "f1": 0.0, "exact": 0.0}
    rows = []
    for rec in cf:
        pred = greedy_facts(model, vocab, rec, device=device, max_new=max_new)
        rows.append(fact_scores(pred, rec.facts))
    return {"n": len(cf),
            "f1": float(np.mean([r["f1"] for r in rows])),
            "exact": float(np.mean([r["exact"] for r in rows]))}


def evaluate_all(model, vocab, records, device=DEV, max_new=80):
    teacher = teacher_forced_eval(model, vocab, records, device=device)
    free = free_eval(model, vocab, records, device=device, max_new=max_new)
    para = paraphrase_eval(model, vocab, records, device=device, max_new=max_new)
    cf = counterfactual_eval(model, vocab, records, device=device, max_new=max_new)
    gate = (teacher["fact_value_acc"] >= 0.80 and free["f1"] >= 0.80
            and (para["n_groups"] == 0 or para["consistent"] >= 0.80)
            and (cf["n"] == 0 or cf["f1"] >= 0.80))
    return {"teacher_forced": teacher, "free_decode": free,
            "paraphrase_consistency": para, "counterfactual": cf,
            "gate_thresholds": {"fact_value_acc": 0.80, "free_f1": 0.80,
                                "paraphrase_consistent": 0.80,
                                "counterfactual_f1": 0.80},
            "gate": gate}


def run(data, steps=400, batch=32, d=96, seed=0, device=DEV, out=None, checkpoint=None):
    records = load_records(data)
    model, vocab = train_model(records, steps=steps, batch=batch, d=d, seed=seed,
                               device=device)
    report = {"experiment": "text0_semantic_extraction", "data": data, "steps": steps,
              "train_records": sum(r.split == "train" for r in records),
              "eval_records": sum(r.split == "eval" for r in records)}
    report.update(evaluate_all(model, vocab, records, device=device))
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos,
                    "d": d, "report": report}, checkpoint)
        report["checkpoint"] = checkpoint
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    return report


def selftest():
    raw = [
        {"split": "train", "id": "t1", "text": "the red square is on card one .",
         "facts": [["p0", "color", "red"], ["p0", "shape", "square"]]},
        {"split": "train", "id": "t2", "text": "card two contains a blue circle .",
         "facts": [["p0", "color", "blue"], ["p0", "shape", "circle"]]},
        {"split": "eval", "id": "e1", "group": "g1",
         "text": "a red square appears on the card .",
         "facts": [["p0", "color", "red"], ["p0", "shape", "square"]]},
        {"split": "eval", "id": "e2", "group": "g1",
         "text": "the card shows a square that is red .",
         "facts": [["p0", "color", "red"], ["p0", "shape", "square"]]},
        {"split": "eval", "id": "e3", "kind": "counterfactual", "base_id": "e1",
         "text": "a blue square appears on the card .",
         "facts": [["p0", "color", "blue"], ["p0", "shape", "square"]],
         "changed": [["p0", "color", "blue"]]},
    ]
    records = [normalize_record(r, idx=i) for i, r in enumerate(raw)]
    vocab = build_vocab(records)
    assert parse_facts(trace_tokens(records[0].facts)) == records[0].facts
    txt, ids = pack(records[:2], vocab, "cpu")
    model = TextFactLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad).to("cpu")
    logits = model(txt, ids)
    assert logits.shape == (2, ids.shape[1], len(vocab))
    loss = token_loss(logits, ids, pad=vocab.pad)
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())
    report = evaluate_all(model, vocab, records, device="cpu", max_new=12)
    assert set(report) >= {"teacher_forced", "free_decode", "paraphrase_consistency",
                           "counterfactual", "gate"}
    print("text selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", help="JSON/JSONL semantic text records")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/text0.json")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.data:
        raise SystemExit("--data is required unless --selftest is set")
    run(args.data, steps=args.steps, batch=args.batch, d=args.d, seed=args.seed,
        out=args.out, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
