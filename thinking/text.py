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
import urllib.request
import zipfile
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
SCAN_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/tasks.txt"
SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"


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


def parse_scan_line(line):
    line = line.strip()
    if not line or not line.startswith("IN: ") or " OUT: " not in line:
        return None
    left, right = line.split(" OUT: ", 1)
    command = left[len("IN: "):].strip()
    actions = right.strip().split()
    if not command or not actions:
        return None
    facts = [[f"s{i:03d}", "action", action] for i, action in enumerate(actions)]
    return command, facts


def scan_records(text, max_records=2000, eval_frac=0.10, seed=0):
    pairs = [p for p in (parse_scan_line(line) for line in text.splitlines()) if p is not None]
    if max_records:
        pairs = pairs[:max_records]
    if not pairs:
        raise ValueError("SCAN import found no command/action pairs")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    eval_n = max(1, int(round(len(pairs) * eval_frac)))
    eval_ids = set(int(i) for i in order[:eval_n])
    records = []
    for i, (command, facts) in enumerate(pairs):
        split = "eval" if i in eval_ids else "train"
        records.append({"split": split, "id": f"scan-{i}", "text": command, "facts": facts,
                        "kind": "scan_command"})
    return records


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def import_scan(out, url=SCAN_URL, max_records=2000, eval_frac=0.10, seed=0):
    req = urllib.request.Request(url, headers={"User-Agent": "ponens-text"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8")
    records = scan_records(text, max_records=max_records, eval_frac=eval_frac, seed=seed)
    write_jsonl(records, out)
    report = {"source": url, "out": out, "records": len(records),
              "train_records": sum(r["split"] == "train" for r in records),
              "eval_records": sum(r["split"] == "eval" for r in records),
              "representation": "SCAN OUT actions -> ordered canonical action facts"}
    print(json.dumps(report, indent=1), flush=True)
    return report


def _snli_records_from_member(zf, member, split, limit):
    records = []
    with zf.open(member) as f:
        for line in f:
            if limit and len(records) >= limit:
                break
            row = json.loads(line.decode("utf-8"))
            label = row.get("gold_label")
            if label not in ("entailment", "contradiction", "neutral"):
                continue
            premise = split_words(row.get("sentence1", ""))
            hypothesis = split_words(row.get("sentence2", ""))
            if not premise or not hypothesis:
                continue
            rec_id = row.get("pairID") or f"snli-{split}-{len(records)}"
            text = ["premise", ":"] + premise + ["hypothesis", ":"] + hypothesis
            records.append({"split": split, "id": rec_id, "tokens": text,
                            "facts": [["pair0", "nli", label]],
                            "kind": "natural_language_inference"})
    return records


def import_snli(out, zip_path=None, url=SNLI_URL, max_train=5000, max_eval=1000):
    """Import SNLI as natural-English semantic labels.

    The target is not the next token in either sentence; it is the human-labeled inference
    relation between the premise and hypothesis.
    """
    if zip_path is None:
        cache = os.path.join(os.path.dirname(out) or ".", "snli_1.0.zip")
        if not os.path.exists(cache):
            req = urllib.request.Request(url, headers={"User-Agent": "ponens-text"})
            with urllib.request.urlopen(req, timeout=300) as r, open(cache, "wb") as f:
                f.write(r.read())
        zip_path = cache
    with zipfile.ZipFile(zip_path) as zf:
        train = _snli_records_from_member(
            zf, "snli_1.0/snli_1.0_train.jsonl", "train", max_train)
        evals = _snli_records_from_member(
            zf, "snli_1.0/snli_1.0_dev.jsonl", "eval", max_eval)
    records = train + evals
    write_jsonl(records, out)
    report = {"source": url if zip_path is None else zip_path, "out": out,
              "records": len(records), "train_records": len(train),
              "eval_records": len(evals),
              "representation": "SNLI premise+hypothesis -> canonical nli relation fact",
              "labels": ["entailment", "contradiction", "neutral"]}
    print(json.dumps(report, indent=1), flush=True)
    return report


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
        rec_batch = batch_records(train_records, rng, batch)
        txt, ids = pack(rec_batch, vocab, device)
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


def _nli_side_record(rec, side):
    toks = list(rec.tokens)
    try:
        h_at = toks.index("hypothesis")
    except ValueError:
        return None
    if side == "hypothesis":
        ntoks = tuple(toks[h_at:])
    elif side == "premise":
        ntoks = tuple(toks[:h_at])
    else:
        raise ValueError(side)
    if not ntoks:
        return None
    return TextRecord(rec_id=f"{rec.rec_id}:{side}", split=rec.split, tokens=ntoks,
                      facts=rec.facts, group=rec.group, kind=f"{rec.kind}:{side}",
                      base_id=rec.rec_id, changed=rec.changed)


def nli_artifact_eval(model, vocab, records, full_fact_value_acc, device=DEV):
    """Hypothesis-only / premise-only controls for SNLI-style records.

    Gururangan et al. showed that NLI datasets can contain annotation artifacts where the
    hypothesis alone predicts the label surprisingly well.  A text-understanding gate should
    surface that shortcut instead of counting it as premise-hypothesis reasoning.
    """
    eval_records = [r for r in records if r.split == "eval"
                    and r.kind == "natural_language_inference"]
    if not eval_records:
        return {"n": 0}
    hypo = [r for r in (_nli_side_record(rec, "hypothesis") for rec in eval_records)
            if r is not None]
    prem = [r for r in (_nli_side_record(rec, "premise") for rec in eval_records)
            if r is not None]
    h_acc = teacher_forced_eval(model, vocab, hypo, device=device)["fact_value_acc"] if hypo else 0.0
    p_acc = teacher_forced_eval(model, vocab, prem, device=device)["fact_value_acc"] if prem else 0.0
    return {"n": len(eval_records),
            "hypothesis_only_fact_value_acc": h_acc,
            "premise_only_fact_value_acc": p_acc,
            "full_minus_hypothesis_only": full_fact_value_acc - h_acc,
            "full_minus_premise_only": full_fact_value_acc - p_acc}


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
    artifact = nli_artifact_eval(model, vocab, records, teacher["fact_value_acc"],
                                 device=device)
    gate = (teacher["fact_value_acc"] >= 0.80 and free["f1"] >= 0.80
            and (para["n_groups"] == 0 or para["consistent"] >= 0.80)
            and (cf["n"] == 0 or cf["f1"] >= 0.80)
            and (artifact["n"] == 0 or artifact["full_minus_hypothesis_only"] >= 0.05))
    return {"teacher_forced": teacher, "free_decode": free,
            "paraphrase_consistency": para, "counterfactual": cf,
            "nli_artifact_control": artifact,
            "gate_thresholds": {"fact_value_acc": 0.80, "free_f1": 0.80,
                                "paraphrase_consistent": 0.80,
                                "counterfactual_f1": 0.80,
                                "nli_full_minus_hypothesis_only": 0.05},
            "gate": gate}


def run(data, steps=400, batch=32, d=96, seed=0, device=DEV, out=None, checkpoint=None,
        max_new=160):
    records = load_records(data)
    model, vocab = train_model(records, steps=steps, batch=batch, d=d, seed=seed,
                               device=device)
    report = {"experiment": "text0_semantic_extraction", "data": data, "steps": steps,
              "train_records": sum(r.split == "train" for r in records),
              "eval_records": sum(r.split == "eval" for r in records)}
    report.update(evaluate_all(model, vocab, records, device=device, max_new=max_new))
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
    scan = scan_records("IN: jump twice OUT: I_JUMP I_JUMP\n"
                        "IN: walk left OUT: I_TURN_LEFT I_WALK\n",
                        max_records=2, eval_frac=0.5, seed=0)
    assert scan[0]["facts"][0] == ["s000", "action", "I_JUMP"]
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
    ap.add_argument("--import-scan", action="store_true",
                    help="download SCAN commands and write JSONL semantic records")
    ap.add_argument("--import-snli", action="store_true",
                    help="import SNLI premise/hypothesis pairs as semantic NLI records")
    ap.add_argument("--scan-url", default=SCAN_URL)
    ap.add_argument("--scan-max", type=int, default=2000)
    ap.add_argument("--scan-eval-frac", type=float, default=0.10)
    ap.add_argument("--snli-url", default=SNLI_URL)
    ap.add_argument("--snli-zip", default=None)
    ap.add_argument("--snli-train", type=int, default=5000)
    ap.add_argument("--snli-eval", type=int, default=1000)
    ap.add_argument("--data", help="JSON/JSONL semantic text records")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=160, dest="max_new")
    ap.add_argument("--out", default="runs/text0.json")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if args.import_scan:
        import_scan(args.out, url=args.scan_url, max_records=args.scan_max,
                    eval_frac=args.scan_eval_frac, seed=args.seed)
        return
    if args.import_snli:
        import_snli(args.out, zip_path=args.snli_zip, url=args.snli_url,
                    max_train=args.snli_train, max_eval=args.snli_eval)
        return
    if not args.data:
        raise SystemExit("--data is required unless --selftest is set")
    run(args.data, steps=args.steps, batch=args.batch, d=args.d, seed=args.seed,
        out=args.out, checkpoint=args.checkpoint, max_new=args.max_new)


if __name__ == "__main__":
    main()
