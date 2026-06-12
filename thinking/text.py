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
import csv
import itertools
import io
import json
import os
import re
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
MNLI_URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
HANS_TRAIN_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_train_set.txt"
HANS_EVAL_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
TOKEN_RE = re.compile(r"[a-z0-9_]+|[^\s\w]", re.ASCII)


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
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FactSchema:
    keys: tuple[tuple[str, str], ...]
    values: tuple[tuple[str, ...], ...]

    @property
    def value_index(self):
        return {(key, val): i for key, vals in zip(self.keys, self.values)
                for i, val in enumerate(vals)}


def split_words(text):
    return TOKEN_RE.findall(str(text).lower())


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
    meta = raw.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError(f"record {idx}.meta must be an object when provided")
    return TextRecord(
        rec_id=str(raw.get("id", f"{split}-{idx}")),
        split=split,
        tokens=tokens,
        facts=facts,
        group=raw.get("group"),
        kind=str(raw.get("kind", "plain")),
        base_id=raw.get("base_id"),
        changed=changed,
        meta=meta,
    )


def load_records(path, require_train=True, require_eval=True):
    if isinstance(path, (list, tuple)):
        records = []
        for p in path:
            records.extend(load_records(p, require_train=False, require_eval=False))
        if require_train and not any(r.split == "train" for r in records):
            raise ValueError(f"{path} has no train records")
        if require_eval and not any(r.split == "eval" for r in records):
            raise ValueError(f"{path} has no eval records")
        return records
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
    if require_train and not any(r.split == "train" for r in records):
        raise ValueError(f"{path} has no train records")
    if require_eval and not any(r.split == "eval" for r in records):
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


def _snli_records_from_member(zf, member, split, limit, rng):
    records = []
    seen = 0
    with zf.open(member) as f:
        for line in f:
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
            rec = {"split": split, "id": rec_id, "tokens": text,
                   "facts": [["pair0", "nli", label]],
                   "kind": "natural_language_inference"}
            seen += 1
            if not limit or len(records) < limit:
                records.append(rec)
            else:
                j = int(rng.integers(seen))
                if j < limit:
                    records[j] = rec
    return records


def import_snli(out, zip_path=None, url=SNLI_URL, max_train=5000, max_eval=1000, seed=0):
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
        rng = np.random.default_rng(seed)
        train = _snli_records_from_member(
            zf, "snli_1.0/snli_1.0_train.jsonl", "train", max_train, rng)
        evals = _snli_records_from_member(
            zf, "snli_1.0/snli_1.0_dev.jsonl", "eval", max_eval, rng)
    records = train + evals
    write_jsonl(records, out)
    report = {"source": url if zip_path is None else zip_path, "out": out,
              "records": len(records), "train_records": len(train),
              "eval_records": len(evals),
              "seed": int(seed),
              "representation": "SNLI premise+hypothesis -> canonical nli relation fact",
              "labels": ["entailment", "contradiction", "neutral"]}
    print(json.dumps(report, indent=1), flush=True)
    return report


def _nli_record(row, split, rec_id, kind, source, label=None):
    label = label or row.get("gold_label")
    if label not in ("entailment", "contradiction", "neutral"):
        return None
    premise = split_words(row.get("sentence1", ""))
    hypothesis = split_words(row.get("sentence2", ""))
    if not premise or not hypothesis:
        return None
    meta = {"source": source}
    for key in ("genre", "promptID", "pairID"):
        if row.get(key):
            meta[key] = row[key]
    return {"split": split, "id": rec_id,
            "tokens": ["premise", ":"] + premise + ["hypothesis", ":"] + hypothesis,
            "facts": [["pair0", "nli", label]],
            "kind": kind, "meta": meta}


def _mnli_records_from_member(zf, member, split, limit, rng, eval_name=None):
    records = []
    seen = 0
    source = eval_name or "train"
    with zf.open(member) as f:
        for line in f:
            row = json.loads(line.decode("utf-8"))
            genre = row.get("genre") or "unknown"
            pair_id = row.get("pairID") or f"{source}-{seen}"
            rec = _nli_record(row, split, f"mnli-{source}-{pair_id}",
                              f"multinli:{genre}", source)
            if rec is None:
                continue
            seen += 1
            if not limit or len(records) < limit:
                records.append(rec)
            else:
                j = int(rng.integers(seen))
                if j < limit:
                    records[j] = rec
    return records, seen


def import_mnli(out, zip_path=None, url=MNLI_URL, max_train=10000, max_eval=2000, seed=0):
    """Import MultiNLI as broader-genre natural-language inference records.

    This is data supervision, not a rule set: premise+hypothesis text is mapped to the human
    NLI label as a canonical `pair0 nli <label>` fact.
    """
    if zip_path is None:
        cache = os.path.join(os.path.dirname(out) or ".", "multinli_1.0.zip")
        if not os.path.exists(cache):
            req = urllib.request.Request(url, headers={"User-Agent": "ponens-text"})
            with urllib.request.urlopen(req, timeout=600) as r, open(cache, "wb") as f:
                f.write(r.read())
        zip_path = cache
    rng = np.random.default_rng(seed)
    matched_n = max_eval // 2 if max_eval else 0
    mismatched_n = max_eval - matched_n if max_eval else 0
    with zipfile.ZipFile(zip_path) as zf:
        train, train_seen = _mnli_records_from_member(
            zf, "multinli_1.0/multinli_1.0_train.jsonl", "train", max_train, rng)
        matched, matched_seen = _mnli_records_from_member(
            zf, "multinli_1.0/multinli_1.0_dev_matched.jsonl", "eval", matched_n,
            rng, eval_name="dev_matched")
        mismatched, mismatched_seen = _mnli_records_from_member(
            zf, "multinli_1.0/multinli_1.0_dev_mismatched.jsonl", "eval", mismatched_n,
            rng, eval_name="dev_mismatched")
    records = train + matched + mismatched
    if not train:
        raise ValueError("MultiNLI import produced no train records")
    if not matched and not mismatched:
        raise ValueError("MultiNLI import produced no eval records")
    write_jsonl(records, out)
    report = {"source": url if zip_path is None else zip_path, "out": out,
              "records": len(records), "train_records": len(train),
              "eval_records": len(matched) + len(mismatched),
              "matched_eval_records": len(matched),
              "mismatched_eval_records": len(mismatched),
              "train_seen": train_seen,
              "matched_seen": matched_seen,
              "mismatched_seen": mismatched_seen,
              "seed": int(seed),
              "representation": "MultiNLI premise+hypothesis -> canonical nli relation fact",
              "genres": sorted({r["meta"].get("genre", "unknown") for r in records})}
    print(json.dumps(report, indent=1), flush=True)
    return report


def _read_text_source(source, timeout=300):
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "ponens-text"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8")
    with open(source, encoding="utf-8") as f:
        return f.read()


def _hans_label(label, mode):
    if label == "entailment":
        return "entailment"
    if label == "non-entailment":
        return "neutral" if mode == "snli" else "non_entailment"
    raise ValueError(f"unknown HANS label {label!r}")


def _hans_records_from_text(text, split, limit, rng, label_mode):
    records = []
    seen = 0
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    required = {"gold_label", "sentence1", "sentence2", "pairID", "heuristic",
                "subcase", "template"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("HANS file is missing required TSV fields")
    for row in reader:
        premise = split_words(row["sentence1"])
        hypothesis = split_words(row["sentence2"])
        if not premise or not hypothesis:
            continue
        heuristic = row["heuristic"]
        label = _hans_label(row["gold_label"], label_mode)
        rec = {"split": split, "id": f"hans-{split}-{row['pairID']}",
               "tokens": ["premise", ":"] + premise + ["hypothesis", ":"] + hypothesis,
               "facts": [["pair0", "nli", label]],
               "kind": f"hans:{heuristic}",
               "meta": {"heuristic": heuristic, "subcase": row["subcase"],
                        "template": row["template"], "original_label": row["gold_label"]}}
        seen += 1
        if not limit or len(records) < limit:
            records.append(rec)
        else:
            j = int(rng.integers(seen))
            if j < limit:
                records[j] = rec
    return records, seen


def import_hans(out, train_source=HANS_TRAIN_URL, eval_source=HANS_EVAL_URL,
                max_train=5000, max_eval=3000, seed=0, label_mode="snli"):
    """Import HANS as adversarial NLI records.

    HANS is not used as a rulebook.  It supplies controlled premise/hypothesis examples that
    break shallow lexical-overlap, subsequence, and constituent heuristics.
    """
    rng = np.random.default_rng(seed)
    train_text = _read_text_source(train_source)
    eval_text = _read_text_source(eval_source)
    train, train_seen = _hans_records_from_text(train_text, "train", max_train, rng,
                                                label_mode)
    evals, eval_seen = _hans_records_from_text(eval_text, "eval", max_eval, rng,
                                               label_mode)
    records = train + evals
    if not train:
        raise ValueError("HANS import produced no train records")
    if not evals:
        raise ValueError("HANS import produced no eval records")
    write_jsonl(records, out)
    report = {"source": {"train": train_source, "eval": eval_source}, "out": out,
              "records": len(records), "train_records": len(train),
              "eval_records": len(evals), "train_seen": train_seen, "eval_seen": eval_seen,
              "seed": int(seed), "label_mode": label_mode,
              "representation": "HANS premise+hypothesis -> canonical nli relation fact",
              "heuristics": sorted({r["meta"]["heuristic"] for r in records})}
    print(json.dumps(report, indent=1), flush=True)
    return report


def _grounded_facts(gold, trace_tokens_fn):
    return [list(fact) for fact in parse_facts(trace_tokens_fn(gold))]


def _grounded_changed(base, edited, trace_tokens_fn):
    before = {(slot, pred): val for slot, pred, val in parse_facts(trace_tokens_fn(base))}
    after = {(slot, pred): val for slot, pred, val in parse_facts(trace_tokens_fn(edited))}
    return [[slot, pred, val] for (slot, pred), val in after.items()
            if before.get((slot, pred)) != val]


def _grounded_combo_id(gold, factor_names):
    return "-".join(gold[factor] for factor in factor_names)


def _sample_grounded_gold(rng, factor_values):
    return {factor: values[int(rng.integers(len(values)))]
            for factor, values in factor_values.items()}


def _all_grounded_combos(factor_values):
    keys = tuple(factor_values)
    for vals in itertools.product(*(factor_values[k] for k in keys)):
        yield dict(zip(keys, vals))


def _render_grounded_template(template, gold):
    return split_words(template.format(**gold))


def grounded_records(surfaces_path=None, max_train=6000, max_eval=1200,
                     counterfactual_n=300, seed=0):
    """Generate transcript -> sensory-fact records from the multimodal oracle world.

    The transcript is the only model input for these records.  Image/audio factors define the
    canonical target, but no English parser or word-level rule is encoded here.
    """
    from .multimodal import (FACTOR_VALUES, load_text_surfaces, mutate_factor,
                             trace_tokens as multimodal_trace_tokens)

    surfaces = load_text_surfaces(surfaces_path)
    rng = np.random.default_rng(seed)
    records = []
    factor_names = tuple(FACTOR_VALUES)
    train_templates = surfaces["train"]
    eval_templates = surfaces["eval"]
    if not train_templates or not eval_templates:
        raise ValueError("grounded text import requires train and eval templates")

    for i in range(max_train):
        gold = _sample_grounded_gold(rng, FACTOR_VALUES)
        tidx = int(rng.integers(len(train_templates)))
        records.append({"split": "train", "id": f"grounded-train-{i}",
                        "tokens": _render_grounded_template(train_templates[tidx], gold),
                        "facts": _grounded_facts(gold, multimodal_trace_tokens),
                        "kind": "grounded_multimodal_text",
                        "group": f"grounded-train-{_grounded_combo_id(gold, factor_names)}",
                        "meta": {"template_index": tidx,
                                 "surface_path": surfaces_path or "default"}})

    combos = list(_all_grounded_combos(FACTOR_VALUES))
    order = rng.permutation(len(combos))
    eval_count = 0
    for raw_idx in order:
        if eval_count >= max_eval:
            break
        gold = combos[int(raw_idx)]
        group = f"grounded-eval-{_grounded_combo_id(gold, factor_names)}"
        for tidx, template in enumerate(eval_templates):
            if eval_count >= max_eval:
                break
            records.append({"split": "eval", "id": f"grounded-eval-{eval_count}",
                            "tokens": _render_grounded_template(template, gold),
                            "facts": _grounded_facts(gold, multimodal_trace_tokens),
                            "kind": "grounded_multimodal_text",
                            "group": group,
                            "meta": {"template_index": tidx,
                                     "surface_path": surfaces_path or "default"}})
            eval_count += 1

    for i in range(counterfactual_n):
        base = _sample_grounded_gold(rng, FACTOR_VALUES)
        factor = factor_names[i % len(factor_names)]
        edited = mutate_factor(base, factor, rng)
        tidx = int(rng.integers(len(eval_templates)))
        records.append({"split": "eval", "id": f"grounded-counterfactual-{i}",
                        "tokens": _render_grounded_template(eval_templates[tidx], edited),
                        "facts": _grounded_facts(edited, multimodal_trace_tokens),
                        "kind": "grounded_counterfactual",
                        "base_id": f"grounded-base-{i}",
                        "changed": _grounded_changed(base, edited, multimodal_trace_tokens),
                        "group": f"grounded-cf-{i}",
                        "meta": {"edited_factor": factor, "template_index": tidx,
                                 "surface_path": surfaces_path or "default"}})
    return records


def import_grounded(out, surfaces_path=None, max_train=6000, max_eval=1200,
                    counterfactual_n=300, seed=0):
    records = grounded_records(surfaces_path=surfaces_path, max_train=max_train,
                               max_eval=max_eval, counterfactual_n=counterfactual_n,
                               seed=seed)
    write_jsonl(records, out)
    report = {"source": surfaces_path or "data/multimodal_transcripts.json",
              "out": out, "records": len(records),
              "train_records": sum(r["split"] == "train" for r in records),
              "eval_records": sum(r["split"] == "eval" for r in records),
              "counterfactual_records": sum(r["kind"] == "grounded_counterfactual"
                                            for r in records),
              "seed": int(seed),
              "representation": "transcript-only multimodal descriptions -> canonical sensory facts",
              "facts": sorted({tuple(f[:2]) for r in records for f in r["facts"]})}
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


def build_fact_schema(records):
    by_key = {}
    for rec in records:
        for slot, pred, val in rec.facts:
            by_key.setdefault((slot, pred), set()).add(val)
    keys = tuple(sorted(by_key))
    values = tuple(tuple(sorted(by_key[k])) for k in keys)
    return FactSchema(keys=keys, values=values)


def vocab_from_itos(itos):
    vocab = object.__new__(Vocab)
    vocab.itos = list(itos)
    vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
    vocab.pad = vocab.stoi.get("<pad>", 0)
    vocab.unk = vocab.stoi.get("<unk>", 1)
    return vocab


def fact_schema_from_payload(payload):
    if not payload:
        return None
    return FactSchema(
        keys=tuple(tuple(x) for x in payload["keys"]),
        values=tuple(tuple(v) for v in payload["values"]),
    )


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

    def __init__(self, vocab_size, d=96, layers=3, heads=4, pad=0, max_len=512,
                 fact_schema=None):
        super().__init__()
        self.txt = TextPrefix(vocab_size, d=d, pad=pad, heads=heads)
        self.lm = ScratchpadLM(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                               pad=pad, pointer=False, loop=False)
        self.fact_schema = fact_schema
        self.fact_heads = nn.ModuleDict()
        if fact_schema is not None:
            self.fact_query = nn.Parameter(torch.randn(len(fact_schema.keys), d) * 0.02)
            for i, vals in enumerate(fact_schema.values):
                self.fact_heads[str(i)] = nn.Linear(d, len(vals))
        else:
            self.fact_query = None

    def encode_text(self, txt):
        prefix = self.txt(txt)
        keep = txt.ne(self.txt.pad).unsqueeze(-1)
        pooled = (prefix * keep).sum(1) / keep.sum(1).clamp(min=1)
        return prefix, pooled

    def forward(self, txt, ids):
        prefix, _pooled = self.encode_text(txt)
        logits = self.lm(ids, prefix=prefix)
        return logits[:, prefix.shape[1]:]

    def semantic_logits(self, txt):
        if self.fact_schema is None:
            return {}
        prefix, _pooled = self.encode_text(txt)
        mask = txt.eq(self.txt.pad)
        out = {}
        scale = prefix.shape[-1] ** -0.5
        for k, head in self.fact_heads.items():
            idx = int(k)
            scores = (prefix * self.fact_query[idx]).sum(-1) * scale
            scores = scores.masked_fill(mask, float("-inf"))
            pooled = (scores.softmax(-1).unsqueeze(-1) * prefix).sum(1)
            out[self.fact_schema.keys[idx]] = head(pooled)
        return out


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


def bucket_records(records, key):
    if key == "none":
        return {"all": list(records)}
    if key == "kind":
        buckets = {}
        for rec in records:
            buckets.setdefault(rec.kind, []).append(rec)
        return buckets
    raise ValueError(f"unknown balance key {key!r}")


def balanced_batch_records(buckets, rng, batch):
    names = tuple(sorted(k for k, rows in buckets.items() if rows))
    if not names:
        raise ValueError("cannot sample from empty buckets")
    rows = []
    for i in range(batch):
        bucket = buckets[names[i % len(names)]]
        rows.append(bucket[int(rng.integers(len(bucket)))])
    if len(rows) > 1:
        order = rng.permutation(len(rows))
        rows = [rows[int(i)] for i in order]
    return rows


def token_loss(logits, ids, pad=0):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           ids[:, 1:].reshape(-1), ignore_index=pad)


def semantic_loss(model, txt, records, schema):
    if schema is None:
        return torch.tensor(0.0, device=txt.device)
    logits = model.semantic_logits(txt)
    value_index = schema.value_index
    losses = []
    for key in schema.keys:
        targets = torch.full((len(records),), -100, dtype=torch.long, device=txt.device)
        for i, rec in enumerate(records):
            vals = [val for slot, pred, val in rec.facts if (slot, pred) == key]
            if vals:
                targets[i] = value_index[(key, vals[0])]
        if targets.ne(-100).any() and logits[key].shape[-1] > 1:
            losses.append(F.cross_entropy(logits[key], targets, ignore_index=-100))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def train_model(records, steps=400, batch=32, d=96, layers=3, heads=4, lr=1e-3, seed=0,
                device=DEV, log_every=100, semantic_w=0.5, balance_by="none"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = build_vocab(records)
    schema = build_fact_schema(records)
    model = TextFactLM(len(vocab), d=d, layers=layers, heads=heads, pad=vocab.pad,
                       fact_schema=schema).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_records = [r for r in records if r.split == "train"]
    train_buckets = bucket_records(train_records, balance_by)
    for st in range(1, steps + 1):
        model.train()
        if balance_by == "none":
            rec_batch = batch_records(train_records, rng, batch)
        else:
            rec_batch = balanced_batch_records(train_buckets, rng, batch)
        txt, ids = pack(rec_batch, vocab, device)
        dec_loss = token_loss(model(txt, ids), ids, pad=vocab.pad)
        sem_loss = semantic_loss(model, txt, rec_batch, schema)
        loss = dec_loss + semantic_w * sem_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % log_every == 0 or st == steps:
            print(f"  text {st}/{steps} loss {loss.item():.3f} "
                  f"dec {dec_loss.item():.3f} sem {sem_loss.item():.3f}", flush=True)
    return model, vocab


def fact_scores(pred, gold):
    p, g = set(pred), set(gold)
    tp = len(p & g)
    precision = tp / max(1, len(p))
    recall = tp / max(1, len(g))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "exact": float(p == g)}


def teacher_forced_eval(model, vocab, records, device=DEV, n=0, seed=0):
    selected = eval_records(records, n=n, seed=seed)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for off in range(0, len(selected), 64):
            batch = selected[off:off + 64]
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
    eval_count = len([r for r in records if r.split == "eval"])
    return {"fact_value_acc": correct / max(1, total), "n_facts": total,
            "n_records": len(selected),
            "sampled": bool(n > 0 and n < eval_count),
            "skipped": bool(n < 0)}


def semantic_fact_eval(model, vocab, records, device=DEV, n=0, seed=0):
    if model.fact_schema is None:
        return {"fact_value_acc": 0.0, "n_facts": 0, "n_records": 0,
                "sampled": False, "skipped": bool(n < 0)}
    selected = eval_records(records, n=n, seed=seed)
    value_index = model.fact_schema.value_index
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(selected), 64):
            batch = selected[off:off + 64]
            txt, _ids = pack(batch, vocab, device)
            logits = model.semantic_logits(txt)
            for r, rec in enumerate(batch):
                for slot, pred, val in rec.facts:
                    key = (slot, pred)
                    if key not in logits:
                        continue
                    if (key, val) not in value_index:
                        continue
                    pred_id = int(logits[key][r].argmax(-1))
                    correct += int(pred_id == value_index[(key, val)])
                    total += 1
    eval_count = len([r for r in records if r.split == "eval"])
    return {"fact_value_acc": correct / max(1, total), "n_facts": total,
            "n_records": len(selected),
            "sampled": bool(n > 0 and n < eval_count),
            "skipped": bool(n < 0)}


def bucket_fact_eval(model, vocab, records, device=DEV, n=0, seed=0):
    buckets = {}
    for rec in records:
        if rec.split == "eval":
            buckets.setdefault(rec.kind, []).append(rec)
    out = {}
    for i, (name, rows) in enumerate(sorted(buckets.items())):
        teacher = teacher_forced_eval(model, vocab, rows, device=device, n=n,
                                      seed=seed + 2003 * i)
        semantic = semantic_fact_eval(model, vocab, rows, device=device, n=n,
                                      seed=seed + 3001 * i)
        out[name] = {"n": len(rows),
                     "n_records": teacher["n_records"],
                     "sampled": teacher["sampled"],
                     "teacher_forced_fact_value_acc": teacher["fact_value_acc"],
                     "semantic_fact_value_acc": semantic["fact_value_acc"]}
    return out


def bucket_free_eval(model, vocab, records, device=DEV, max_new=80, n=0, seed=0):
    buckets = {}
    for rec in records:
        if rec.split == "eval":
            buckets.setdefault(rec.kind, []).append(rec)
    out = {}
    for i, (name, rows) in enumerate(sorted(buckets.items())):
        free = free_eval(model, vocab, rows, device=device, max_new=max_new, n=n,
                         seed=seed + 1009 * i)
        out[name] = free
    return out


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
                      base_id=rec.rec_id, changed=rec.changed, meta=rec.meta)


def _is_nli_record(rec):
    return any((slot, pred) == ("pair0", "nli") for slot, pred, _val in rec.facts)


def nli_artifact_eval(model, vocab, records, full_fact_value_acc, device=DEV, n=0, seed=0):
    """Hypothesis-only / premise-only controls for SNLI-style records.

    Gururangan et al. showed that NLI datasets can contain annotation artifacts where the
    hypothesis alone predicts the label surprisingly well.  A text-understanding gate should
    surface that shortcut instead of counting it as premise-hypothesis reasoning.
    """
    all_eval = [r for r in records if r.split == "eval" and _is_nli_record(r)]
    if n < 0:
        return {"n": 0, "sampled": False, "skipped": True}
    if not all_eval:
        return {"n": 0, "sampled": False, "skipped": False}
    eval_records_ = all_eval
    sampled = bool(n and n < len(eval_records_))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(eval_records_), size=n, replace=False)
        eval_records_ = [eval_records_[int(i)] for i in idx]
    hypo = [r for r in (_nli_side_record(rec, "hypothesis") for rec in eval_records_)
            if r is not None]
    prem = [r for r in (_nli_side_record(rec, "premise") for rec in eval_records_)
            if r is not None]
    full_acc = teacher_forced_eval(model, vocab, eval_records_,
                                   device=device)["fact_value_acc"]
    h_acc = teacher_forced_eval(model, vocab, hypo, device=device)["fact_value_acc"] if hypo else 0.0
    p_acc = teacher_forced_eval(model, vocab, prem, device=device)["fact_value_acc"] if prem else 0.0
    h_sem = semantic_fact_eval(model, vocab, hypo, device=device)["fact_value_acc"] if hypo else 0.0
    p_sem = semantic_fact_eval(model, vocab, prem, device=device)["fact_value_acc"] if prem else 0.0
    full_sem = semantic_fact_eval(model, vocab, eval_records_, device=device)["fact_value_acc"]
    return {"n": len(eval_records_),
            "sampled": sampled,
            "skipped": False,
            "hypothesis_only_fact_value_acc": h_acc,
            "premise_only_fact_value_acc": p_acc,
            "full_minus_hypothesis_only": full_acc - h_acc,
            "full_minus_premise_only": full_acc - p_acc,
            "semantic_full_fact_value_acc": full_sem,
            "semantic_hypothesis_only_fact_value_acc": h_sem,
            "semantic_premise_only_fact_value_acc": p_sem,
            "semantic_full_minus_hypothesis_only": full_sem - h_sem,
            "semantic_full_minus_premise_only": full_sem - p_sem}


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


def eval_records(records, n=0, seed=0):
    out = [r for r in records if r.split == "eval"]
    if n < 0:
        return []
    if n and n < len(out):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(out), size=n, replace=False)
        out = [out[int(i)] for i in idx]
    return out


def free_eval(model, vocab, records, device=DEV, max_new=80, n=0, seed=0):
    rows = []
    selected = eval_records(records, n=n, seed=seed)
    for rec in selected:
        pred = greedy_facts(model, vocab, rec, device=device, max_new=max_new)
        rows.append(fact_scores(pred, rec.facts))
    keys = ("precision", "recall", "f1", "exact")
    return {k: float(np.mean([r[k] for r in rows])) if rows else 0.0 for k in keys} | {
        "n": len(rows),
        "sampled": bool(n > 0 and n < len([r for r in records if r.split == "eval"])),
        "skipped": bool(n < 0)}


def paraphrase_eval(model, vocab, records, device=DEV, max_new=80, n_groups=0, seed=0):
    groups = {}
    for rec in records:
        if rec.split == "eval" and rec.group:
            groups.setdefault(rec.group, []).append(rec)
    usable = [g for g in groups.values() if len(g) >= 2]
    if n_groups < 0:
        return {"n_groups": 0, "sampled": False, "skipped": True,
                "consistent": 0.0, "exact": 0.0}
    sampled = bool(n_groups and n_groups < len(usable))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(usable), size=n_groups, replace=False)
        usable = [usable[int(i)] for i in idx]
    consistent = exact = 0
    for group in usable:
        preds = [set(greedy_facts(model, vocab, rec, device=device, max_new=max_new))
                 for rec in group]
        golds = [set(rec.facts) for rec in group]
        consistent += int(all(p == preds[0] for p in preds[1:]))
        exact += int(all(p == g for p, g in zip(preds, golds)))
    return {"n_groups": len(usable),
            "sampled": sampled,
            "skipped": False,
            "consistent": consistent / max(1, len(usable)),
            "exact": exact / max(1, len(usable))}


def counterfactual_eval(model, vocab, records, device=DEV, max_new=80, n=0, seed=0):
    cf = [r for r in records if r.split == "eval"
          and (r.kind == "counterfactual" or r.base_id or r.changed)]
    if n < 0:
        return {"n": 0, "sampled": False, "skipped": True, "f1": 0.0, "exact": 0.0}
    if not cf:
        return {"n": 0, "sampled": False, "skipped": False, "f1": 0.0, "exact": 0.0}
    sampled = bool(n and n < len(cf))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(cf), size=n, replace=False)
        cf = [cf[int(i)] for i in idx]
    rows = []
    for rec in cf:
        pred = greedy_facts(model, vocab, rec, device=device, max_new=max_new)
        rows.append(fact_scores(pred, rec.facts))
    return {"n": len(cf),
            "sampled": sampled,
            "skipped": False,
            "f1": float(np.mean([r["f1"] for r in rows])),
            "exact": float(np.mean([r["exact"] for r in rows]))}


def evaluate_all(model, vocab, records, device=DEV, max_new=80, free_n=0,
                 paraphrase_n=0, counterfactual_n=0, kind_free_n=0,
                 fact_n=0, kind_fact_n=0, artifact_n=0, seed=0):
    teacher = teacher_forced_eval(model, vocab, records, device=device, n=fact_n,
                                  seed=seed + 11)
    semantic = semantic_fact_eval(model, vocab, records, device=device, n=fact_n,
                                  seed=seed + 11)
    free = free_eval(model, vocab, records, device=device, max_new=max_new, n=free_n,
                     seed=seed)
    para = paraphrase_eval(model, vocab, records, device=device, max_new=max_new,
                           n_groups=paraphrase_n, seed=seed + 1)
    cf = counterfactual_eval(model, vocab, records, device=device, max_new=max_new,
                             n=counterfactual_n, seed=seed + 2)
    artifact = nli_artifact_eval(model, vocab, records, teacher["fact_value_acc"],
                                 device=device, n=artifact_n, seed=seed + 13)
    by_kind = bucket_fact_eval(model, vocab, records, device=device, n=kind_fact_n,
                               seed=seed + 19)
    by_kind_free = (bucket_free_eval(model, vocab, records, device=device, max_new=max_new,
                                     n=kind_free_n, seed=seed + 3)
                    if kind_free_n else {})
    gate = (teacher["fact_value_acc"] >= 0.80 and semantic["fact_value_acc"] >= 0.80
            and (free.get("skipped") or free["f1"] >= 0.80)
            and (para.get("skipped") or para["n_groups"] == 0 or para["consistent"] >= 0.80)
            and (cf.get("skipped") or cf["n"] == 0 or cf["f1"] >= 0.80)
            and (artifact.get("skipped") or artifact["n"] == 0
                 or artifact["full_minus_hypothesis_only"] >= 0.05))
    return {"teacher_forced": teacher, "free_decode": free,
            "semantic_head": semantic,
            "by_kind": by_kind,
            "free_decode_by_kind": by_kind_free,
            "paraphrase_consistency": para, "counterfactual": cf,
            "nli_artifact_control": artifact,
            "gate_thresholds": {"fact_value_acc": 0.80, "free_f1": 0.80,
                                "semantic_fact_value_acc": 0.80,
                                "paraphrase_consistent": 0.80,
                                "counterfactual_f1": 0.80,
                                "nli_full_minus_hypothesis_only": 0.05},
            "gate": gate}


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(path, device=DEV):
    ckpt = _torch_load(path, device)
    vocab = vocab_from_itos(ckpt["vocab"])
    schema = fact_schema_from_payload(ckpt.get("fact_schema"))
    model = TextFactLM(len(vocab), d=int(ckpt.get("d", 96)),
                       layers=int(ckpt.get("layers", 3)),
                       heads=int(ckpt.get("heads", 4)), pad=vocab.pad,
                       fact_schema=schema).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab, ckpt


def vocab_coverage(records, vocab):
    total = unk = 0
    for rec in records:
        for tok in rec.tokens:
            total += 1
            unk += int(tok not in vocab.stoi)
    return {"text_tokens": total, "unknown_text_tokens": unk,
            "unknown_text_token_rate": unk / max(1, total)}


def eval_checkpoint(checkpoint, data, out=None, device=DEV, max_new=160, free_n=0,
                    paraphrase_n=0, counterfactual_n=0, kind_free_n=0,
                    fact_n=0, kind_fact_n=0, artifact_n=0, seed=0):
    records = load_records(data, require_train=False, require_eval=True)
    model, vocab, ckpt = load_checkpoint(checkpoint, device=device)
    report = {"experiment": "text0_checkpoint_eval", "data": data,
              "checkpoint": checkpoint,
              "checkpoint_experiment": ckpt.get("report", {}).get("experiment"),
              "train_records": sum(r.split == "train" for r in records),
              "eval_records": sum(r.split == "eval" for r in records),
              "free_n": int(free_n),
              "paraphrase_n": int(paraphrase_n),
              "counterfactual_n": int(counterfactual_n),
              "kind_free_n": int(kind_free_n),
              "fact_n": int(fact_n),
              "kind_fact_n": int(kind_fact_n),
              "artifact_n": int(artifact_n),
              "vocab_coverage": vocab_coverage([r for r in records if r.split == "eval"],
                                                vocab)}
    report.update(evaluate_all(model, vocab, records, device=device, max_new=max_new,
                               free_n=free_n, paraphrase_n=paraphrase_n,
                               counterfactual_n=counterfactual_n,
                               kind_free_n=kind_free_n, fact_n=fact_n,
                               kind_fact_n=kind_fact_n, artifact_n=artifact_n,
                               seed=seed + 17))
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    return report


def run(data, steps=400, batch=32, d=96, layers=3, heads=4, seed=0, device=DEV, out=None,
        checkpoint=None, max_new=160, semantic_w=0.5, free_n=0, paraphrase_n=0,
        counterfactual_n=0, kind_free_n=0, balance_by="none",
        fact_n=0, kind_fact_n=0, artifact_n=0):
    records = load_records(data)
    model, vocab = train_model(records, steps=steps, batch=batch, d=d, layers=layers,
                               heads=heads, seed=seed, device=device, semantic_w=semantic_w,
                               balance_by=balance_by)
    report = {"experiment": "text0_semantic_extraction", "data": data, "steps": steps,
              "d": int(d), "layers": int(layers), "heads": int(heads),
              "semantic_w": float(semantic_w), "free_n": int(free_n),
              "paraphrase_n": int(paraphrase_n),
              "counterfactual_n": int(counterfactual_n),
              "kind_free_n": int(kind_free_n),
              "fact_n": int(fact_n),
              "kind_fact_n": int(kind_fact_n),
              "artifact_n": int(artifact_n),
              "balance_by": balance_by,
              "train_records": sum(r.split == "train" for r in records),
              "eval_records": sum(r.split == "eval" for r in records)}
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos,
                    "d": d, "layers": layers, "heads": heads, "fact_schema": {
                        "keys": model.fact_schema.keys,
                        "values": model.fact_schema.values,
                    }, "report": report | {"status": "trained_pending_eval"}},
                   checkpoint)
    report.update(evaluate_all(model, vocab, records, device=device, max_new=max_new,
                               free_n=free_n, paraphrase_n=paraphrase_n,
                               counterfactual_n=counterfactual_n,
                               kind_free_n=kind_free_n, fact_n=fact_n,
                               kind_fact_n=kind_fact_n, artifact_n=artifact_n,
                               seed=seed + 17))
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos,
                    "d": d, "layers": layers, "heads": heads, "fact_schema": {
                        "keys": model.fact_schema.keys,
                        "values": model.fact_schema.values,
                    }, "report": report}, checkpoint)
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
    buckets = bucket_records(records, "kind")
    assert sorted(buckets) == ["counterfactual", "plain"]
    balanced = balanced_batch_records(buckets, np.random.default_rng(1), 4)
    assert {r.kind for r in balanced} == {"plain", "counterfactual"}
    scan = scan_records("IN: jump twice OUT: I_JUMP I_JUMP\n"
                        "IN: walk left OUT: I_TURN_LEFT I_WALK\n",
                        max_records=2, eval_frac=0.5, seed=0)
    assert scan[0]["facts"][0] == ["s000", "action", "I_JUMP"]
    hans_text = ("gold_label\tsentence1\tsentence2\tpairID\theuristic\tsubcase\ttemplate\n"
                 "non-entailment\tThe doctor advised the president .\t"
                 "The president advised the doctor .\tex0\tlexical_overlap\t"
                 "ln_subject/object_swap\ttemp1\n"
                 "entailment\tThe doctor advised the president .\t"
                 "The doctor advised the president .\tex1\tlexical_overlap\t"
                 "le_lexical_overlap\ttemp2\n")
    hans, seen = _hans_records_from_text(hans_text, "eval", 0, np.random.default_rng(0),
                                         "snli")
    assert seen == 2 and hans[0]["facts"] == [["pair0", "nli", "neutral"]]
    assert hans[0]["kind"] == "hans:lexical_overlap"
    mnli_bytes = io.BytesIO()
    with zipfile.ZipFile(mnli_bytes, "w") as zf:
        zf.writestr("multinli_1.0/multinli_1.0_train.jsonl",
                    json.dumps({"gold_label": "contradiction",
                                "sentence1": "A woman is running .",
                                "sentence2": "No person moves .",
                                "pairID": "tr0", "genre": "fiction"}) + "\n")
        zf.writestr("multinli_1.0/multinli_1.0_dev_matched.jsonl",
                    json.dumps({"gold_label": "entailment",
                                "sentence1": "A child sleeps .",
                                "sentence2": "A child is asleep .",
                                "pairID": "m0", "genre": "telephone"}) + "\n")
    mnli_bytes.seek(0)
    with zipfile.ZipFile(mnli_bytes) as zf:
        mnli, mnli_seen = _mnli_records_from_member(
            zf, "multinli_1.0/multinli_1.0_train.jsonl", "train", 0,
            np.random.default_rng(0))
    assert mnli_seen == 1 and mnli[0]["kind"] == "multinli:fiction"
    assert mnli[0]["facts"] == [["pair0", "nli", "contradiction"]]
    grounded = grounded_records(max_train=4, max_eval=3, counterfactual_n=2, seed=0)
    assert any(r["split"] == "train" and len(r["facts"]) > 1
               for r in grounded)
    assert any(r["kind"] == "grounded_counterfactual" and r["changed"] for r in grounded)
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
                           "counterfactual", "semantic_head", "by_kind",
                           "free_decode_by_kind", "gate"}
    print("text selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--import-scan", action="store_true",
                    help="download SCAN commands and write JSONL semantic records")
    ap.add_argument("--import-snli", action="store_true",
                    help="import SNLI premise/hypothesis pairs as semantic NLI records")
    ap.add_argument("--import-mnli", action="store_true",
                    help="import MultiNLI premise/hypothesis pairs as semantic NLI records")
    ap.add_argument("--import-hans", action="store_true",
                    help="import HANS adversarial NLI pairs as semantic records")
    ap.add_argument("--import-grounded", action="store_true",
                    help="import transcript-only multimodal grounding records")
    ap.add_argument("--scan-url", default=SCAN_URL)
    ap.add_argument("--scan-max", type=int, default=2000)
    ap.add_argument("--scan-eval-frac", type=float, default=0.10)
    ap.add_argument("--snli-url", default=SNLI_URL)
    ap.add_argument("--snli-zip", default=None)
    ap.add_argument("--snli-train", type=int, default=5000)
    ap.add_argument("--snli-eval", type=int, default=1000)
    ap.add_argument("--mnli-url", default=MNLI_URL)
    ap.add_argument("--mnli-zip", default=None)
    ap.add_argument("--mnli-train", type=int, default=10000)
    ap.add_argument("--mnli-eval", type=int, default=2000,
                    help="total MultiNLI dev records split across matched and mismatched")
    ap.add_argument("--hans-train-url", default=HANS_TRAIN_URL)
    ap.add_argument("--hans-eval-url", default=HANS_EVAL_URL)
    ap.add_argument("--hans-train-file", default=None)
    ap.add_argument("--hans-eval-file", default=None)
    ap.add_argument("--hans-train", type=int, default=5000)
    ap.add_argument("--hans-eval", type=int, default=3000)
    ap.add_argument("--hans-label-mode", choices=("snli", "binary"), default="snli",
                    help="map HANS non-entailment to SNLI neutral, or keep a binary label")
    ap.add_argument("--grounded-surfaces", default=None,
                    help="multimodal transcript surface bank for --import-grounded")
    ap.add_argument("--grounded-train", type=int, default=6000)
    ap.add_argument("--grounded-eval", type=int, default=1200)
    ap.add_argument("--grounded-counterfactual", type=int, default=300)
    ap.add_argument("--data", action="append",
                    help="JSON/JSONL semantic text records; repeat to mix datasets")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=160, dest="max_new")
    ap.add_argument("--free-n", type=int, default=0, dest="free_n",
                    help="sample this many eval records for slow free decoding; 0 = all, negative = skip")
    ap.add_argument("--paraphrase-n", type=int, default=0, dest="paraphrase_n",
                    help="sample this many paraphrase groups for slow free decoding; 0 = all, negative = skip")
    ap.add_argument("--counterfactual-n", type=int, default=0, dest="counterfactual_n",
                    help="sample this many counterfactual records for slow free decoding; 0 = all, negative = skip")
    ap.add_argument("--kind-free-n", type=int, default=0, dest="kind_free_n",
                    help="sample this many eval records per kind for slow per-kind free decoding")
    ap.add_argument("--fact-n", type=int, default=0, dest="fact_n",
                    help="sample this many eval records for teacher/semantic fact eval; 0 = all")
    ap.add_argument("--kind-fact-n", type=int, default=0, dest="kind_fact_n",
                    help="sample this many eval records per kind for teacher/semantic buckets; 0 = all")
    ap.add_argument("--artifact-n", type=int, default=0, dest="artifact_n",
                    help="sample this many NLI eval records for artifact controls; 0 = all")
    ap.add_argument("--balance-by", choices=("none", "kind"), default="none",
                    help="balance training batches across metadata buckets")
    ap.add_argument("--semantic-w", type=float, default=0.5, dest="semantic_w",
                    help="weight for direct semantic fact classification auxiliary loss")
    ap.add_argument("--out", default="runs/text0.json")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-checkpoint", default=None,
                    help="load a text checkpoint and evaluate it on --data without training")
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
                    max_train=args.snli_train, max_eval=args.snli_eval, seed=args.seed)
        return
    if args.import_mnli:
        import_mnli(args.out, zip_path=args.mnli_zip, url=args.mnli_url,
                    max_train=args.mnli_train, max_eval=args.mnli_eval, seed=args.seed)
        return
    if args.import_hans:
        import_hans(args.out,
                    train_source=args.hans_train_file or args.hans_train_url,
                    eval_source=args.hans_eval_file or args.hans_eval_url,
                    max_train=args.hans_train, max_eval=args.hans_eval,
                    seed=args.seed, label_mode=args.hans_label_mode)
        return
    if args.import_grounded:
        import_grounded(args.out, surfaces_path=args.grounded_surfaces,
                        max_train=args.grounded_train, max_eval=args.grounded_eval,
                        counterfactual_n=args.grounded_counterfactual, seed=args.seed)
        return
    if not args.data:
        raise SystemExit("--data is required unless --selftest is set")
    if args.eval_checkpoint:
        eval_checkpoint(args.eval_checkpoint, args.data, out=args.out, device=DEV,
                        max_new=args.max_new, free_n=args.free_n,
                        paraphrase_n=args.paraphrase_n,
                        counterfactual_n=args.counterfactual_n,
                        kind_free_n=args.kind_free_n, fact_n=args.fact_n,
                        kind_fact_n=args.kind_fact_n, artifact_n=args.artifact_n,
                        seed=args.seed)
        return
    run(args.data, steps=args.steps, batch=args.batch, d=args.d, layers=args.layers,
        heads=args.heads, seed=args.seed, out=args.out, checkpoint=args.checkpoint,
        max_new=args.max_new, semantic_w=args.semantic_w, free_n=args.free_n,
        paraphrase_n=args.paraphrase_n, counterfactual_n=args.counterfactual_n,
        kind_free_n=args.kind_free_n, balance_by=args.balance_by,
        fact_n=args.fact_n, kind_fact_n=args.kind_fact_n, artifact_n=args.artifact_n)


if __name__ == "__main__":
    main()
