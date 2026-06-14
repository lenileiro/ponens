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
import copy
import csv
import io
import json
import math
import os
import re
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from scratchpad_model import CausalBlock, ScratchpadLM

from .concepts import (
    LatentConceptHead,
    LatentConceptMemory,
    LatentConceptSequencePredictor,
    SchemaConceptHead,
    SchemaConceptRefiner,
    latent_concept_bridge_loss,
    latent_concept_bridge_scores,
    latent_concept_composition_loss,
    latent_concept_graph_cycle_loss,
    latent_concept_graph_cycle_scores,
    latent_concept_graph_prediction_loss,
    latent_concept_graph_prediction_scores,
    latent_concept_graph_curiosity_scores,
    latent_concept_cluster_prototype_loss,
    latent_concept_fer_loss,
    latent_concept_fer_metrics,
    latent_concept_graph_ready,
    latent_concept_graph_snapshot,
    latent_concept_neighborhood_loss,
    latent_concept_sequence_prediction_loss,
    latent_concept_slot_factorization_loss,
    latent_concept_transition_consistency_loss,
    latent_concept_vicreg_loss,
    schema_concept_batch_centroid_loss,
    schema_concept_contrastive_loss,
    schema_concept_prototype_alignment_loss,
    schema_concept_prototype_spread_loss,
    schema_concept_state_spread_loss,
)
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
class ReadingRecord:
    rec_id: str
    split: str
    tokens: tuple[str, ...]
    kind: str = "raw_text"
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


def split_words_with_spans(text):
    text = str(text).lower()
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


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


def normalize_reading_record(raw, default_split=None, idx=0, text_field="text"):
    if not isinstance(raw, dict):
        raise ValueError(f"reading record {idx} must be an object")
    split = raw.get("split", default_split or "train")
    if split not in ("train", "eval"):
        raise ValueError(f"reading record {idx} has invalid split {split!r}")
    if "tokens" in raw:
        tokens = tuple(raw["tokens"])
    elif text_field in raw:
        tokens = tuple(split_words(raw[text_field]))
    elif "text" in raw:
        tokens = tuple(split_words(raw["text"]))
    else:
        raise ValueError(f"reading record {idx} must contain text or tokens")
    if not tokens or not all(isinstance(t, str) and t for t in tokens):
        raise ValueError(f"reading record {idx} has empty/non-string tokens")
    meta = raw.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError(f"reading record {idx}.meta must be an object when provided")
    return ReadingRecord(
        rec_id=str(raw.get("id", f"reading-{split}-{idx}")),
        split=split,
        tokens=tokens,
        kind=str(raw.get("kind", "raw_text")),
        meta=meta,
    )


def _chunk_reading_tokens(tokens, max_tokens=128, min_tokens=8):
    max_tokens = max(1, int(max_tokens))
    min_tokens = max(1, int(min_tokens))
    chunks = []
    for off in range(0, len(tokens), max_tokens):
        chunk = tuple(tokens[off:off + max_tokens])
        if len(chunk) >= min_tokens:
            chunks.append(chunk)
    if not chunks and tokens:
        chunks.append(tuple(tokens[:max_tokens]))
    return chunks


def reading_records_from_text(text, source, max_tokens=128, min_tokens=8,
                              eval_frac=0.10, seed=0):
    tokens = split_words(text)
    chunks = _chunk_reading_tokens(tokens, max_tokens=max_tokens, min_tokens=min_tokens)
    if not chunks:
        raise ValueError(f"{source} produced no reading chunks")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(chunks))
    eval_n = max(1, int(round(len(chunks) * eval_frac))) if len(chunks) > 1 else 0
    eval_ids = set(int(i) for i in order[:eval_n])
    return [
        ReadingRecord(
            rec_id=f"{os.path.basename(str(source)) or 'reading'}-{i}",
            split="eval" if i in eval_ids else "train",
            tokens=chunk,
            kind="raw_text",
            meta={"source": str(source), "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]


def _json_reading_records(data, source, text_field="text"):
    records = []
    if isinstance(data, list):
        records = [normalize_reading_record(r, idx=i, text_field=text_field)
                   for i, r in enumerate(data)]
    elif isinstance(data, dict):
        if "records" in data:
            records = [normalize_reading_record(r, idx=i, text_field=text_field)
                       for i, r in enumerate(data["records"])]
        else:
            for split in ("train", "eval"):
                records.extend(normalize_reading_record(
                    r, default_split=split, idx=i, text_field=text_field)
                    for i, r in enumerate(data.get(split, [])))
    if not records:
        raise ValueError(f"{source} produced no reading records")
    return records


def load_reading_records(path, require_train=True, require_eval=True,
                         text_field="text", max_tokens=128, min_tokens=8,
                         eval_frac=0.10, seed=0):
    if isinstance(path, (list, tuple)):
        records = []
        for i, p in enumerate(path):
            records.extend(load_reading_records(
                p, require_train=False, require_eval=False,
                text_field=text_field, max_tokens=max_tokens,
                min_tokens=min_tokens, eval_frac=eval_frac, seed=seed + i))
        if require_train and not any(r.split == "train" for r in records):
            raise ValueError(f"{path} has no train reading records")
        if require_eval and not any(r.split == "eval" for r in records):
            raise ValueError(f"{path} has no eval reading records")
        return records
    txt = _read_text_source(path)
    if str(path).endswith(".jsonl"):
        records = [normalize_reading_record(json.loads(line), idx=i,
                                            text_field=text_field)
                   for i, line in enumerate(txt.splitlines()) if line.strip()]
    elif str(path).endswith(".json"):
        records = _json_reading_records(json.loads(txt), path, text_field=text_field)
    else:
        records = reading_records_from_text(
            txt, path, max_tokens=max_tokens, min_tokens=min_tokens,
            eval_frac=eval_frac, seed=seed)
    if require_train and not any(r.split == "train" for r in records):
        raise ValueError(f"{path} has no train reading records")
    if require_eval and not any(r.split == "eval" for r in records):
        raise ValueError(f"{path} has no eval reading records")
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


def build_vocab(records, base_vocab=None):
    if base_vocab is not None:
        itos = list(base_vocab.itos)
        seen = set(itos)
        new = []
        for rec in records:
            for tok in list(rec.tokens) + [x for fact in rec.facts + rec.changed for x in fact]:
                if tok not in seen:
                    seen.add(tok)
                    new.append(tok)
        return vocab_from_itos(itos + sorted(new))
    toks = list(KEYWORDS)
    for rec in records:
        toks += list(rec.tokens)
        for fact in rec.facts + rec.changed:
            toks += list(fact)
    return Vocab(toks)


def build_reading_vocab(records, base_vocab=None):
    if base_vocab is not None:
        itos = list(base_vocab.itos)
        seen = set(itos)
        new = []
        for rec in records:
            for tok in rec.tokens:
                if tok not in seen:
                    seen.add(tok)
                    new.append(tok)
        return vocab_from_itos(itos + sorted(new))
    toks = []
    for rec in records:
        toks += list(rec.tokens)
    return Vocab(toks)


def build_fact_schema(records, base_schema=None):
    by_key = {}
    if base_schema is not None:
        for key, values in zip(base_schema.keys, base_schema.values):
            by_key.setdefault(tuple(key), set()).update(values)
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


TEXT_ENCODER_ARCHES = ("transformer", "standard", "relational", "abstractor")


class TextPrefix(nn.Module):
    """Bidirectional text encoder producing continuous prefix embeddings."""

    def __init__(self, vocab_size, d, pad=0, heads=4, max_len=256, layers=1,
                 arch="transformer"):
        super().__init__()
        self.pad = pad
        self.arch = str(arch)
        self.layers = int(layers)
        if self.arch not in TEXT_ENCODER_ARCHES:
            raise ValueError(f"unknown text encoder architecture {self.arch!r}")
        if self.layers <= 0:
            raise ValueError("text encoder layers must be positive")
        self.emb = nn.Embedding(vocab_size, d, padding_idx=pad)
        self.pos = nn.Embedding(max_len, d)
        if self.arch == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=0.0,
                activation="gelu", batch_first=True)
            self.enc = nn.TransformerEncoder(
                layer, num_layers=self.layers, enable_nested_tensor=False)
            self.blocks = None
        else:
            self.enc = None
            self.blocks = nn.ModuleList([
                CausalBlock(d, heads, arch=self.arch, vocab=vocab_size,
                            pos_mode="none", causal=False)
                for _ in range(self.layers)
            ])
        self.ln = nn.LayerNorm(d)

    def forward(self, ids):
        L = ids.shape[1]
        pos = torch.arange(L, device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        mask = ids.eq(self.pad)
        h = self.emb(ids) + self.pos(pos)[None]
        if self.enc is not None:
            h = self.enc(h, src_key_padding_mask=mask)
        else:
            for block in self.blocks:
                h = block(h, ids, mask)
        return self.ln(h).masked_fill(mask.unsqueeze(-1), 0.0)


class TextFactLM(nn.Module):
    """Text prefix -> canonical fact trace decoder."""

    def __init__(self, vocab_size, d=96, layers=3, heads=4, pad=0, max_len=512,
                 fact_schema=None, fact_concept_prefix=False,
                 text_encoder_arch="transformer", text_encoder_layers=1,
                 fact_concept_refine=False, fact_concept_refine_gate_init=-2.0,
                 fact_concept_mixer_layers=0,
                 fact_concept_mixer_gate_init=-2.0,
                 latent_concept_slots=0, latent_concept_layers=1,
                 latent_concept_prefix=False,
                 latent_concept_refine=False,
                 latent_concept_refine_gate_init=-2.0,
                 latent_concept_memory_size=0):
        super().__init__()
        self.d = int(d)
        self.heads = int(heads)
        self.fact_concept_prefix = bool(fact_concept_prefix)
        self.fact_concept_refine = bool(fact_concept_refine)
        self.fact_concept_refine_gate_init = float(fact_concept_refine_gate_init)
        self.fact_concept_mixer_layers = int(fact_concept_mixer_layers)
        self.fact_concept_mixer_gate_init = float(fact_concept_mixer_gate_init)
        self.latent_concept_slots = int(latent_concept_slots)
        self.latent_concept_layers = int(latent_concept_layers)
        self.latent_concept_prefix = bool(latent_concept_prefix)
        self.latent_concept_refine = bool(latent_concept_refine)
        self.latent_concept_refine_gate_init = float(latent_concept_refine_gate_init)
        self.latent_concept_memory_size = int(latent_concept_memory_size)
        if ((self.latent_concept_prefix or self.latent_concept_refine)
                and self.latent_concept_slots <= 0):
            raise ValueError("latent concept prefix/refine require latent slots")
        if self.latent_concept_memory_size < 0:
            raise ValueError("latent concept memory size must be non-negative")
        if self.latent_concept_memory_size and self.latent_concept_slots <= 0:
            raise ValueError("latent concept memory requires latent slots")
        self.text_encoder_arch = str(text_encoder_arch)
        self.text_encoder_layers = int(text_encoder_layers)
        self.txt = TextPrefix(vocab_size, d=d, pad=pad, heads=heads,
                              layers=self.text_encoder_layers,
                              arch=self.text_encoder_arch)
        self.lm = ScratchpadLM(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                               pad=pad, pointer=False, loop=False)
        self.reading_predictor = LatentConceptSequencePredictor(d)
        self.fact_schema = fact_schema
        self.fact_heads = nn.ModuleDict()
        self.fact_concept_refiner = None
        self.latent_concept_refiner = None
        self.latent_concepts = (LatentConceptHead(
            self.latent_concept_slots, d, heads=heads,
            mixer_layers=self.latent_concept_layers)
            if self.latent_concept_slots > 0 else None)
        self.latent_concept_memory = (LatentConceptMemory(
            self.latent_concept_memory_size, d)
            if self.latent_concept_memory_size > 0 else None)
        if self.latent_concept_refine and self.latent_concepts is not None:
            self.latent_concept_refiner = SchemaConceptRefiner(
                d, heads=heads, gate_init=self.latent_concept_refine_gate_init)
        if fact_schema is not None:
            self.fact_query = nn.Parameter(torch.randn(len(fact_schema.keys), d) * 0.02)
            self.fact_concepts = SchemaConceptHead(
                fact_schema.keys, fact_schema.values, d,
                mixer_layers=self.fact_concept_mixer_layers,
                mixer_heads=heads,
                mixer_gate_init=self.fact_concept_mixer_gate_init)
            if self.fact_concept_refine:
                self.fact_concept_refiner = SchemaConceptRefiner(
                    d, heads=heads, gate_init=self.fact_concept_refine_gate_init)
            for i, vals in enumerate(fact_schema.values):
                self.fact_heads[str(i)] = nn.Linear(d, len(vals))
        else:
            self.fact_concept_refine = False
            self.fact_concept_mixer_layers = 0
            self.fact_query = None
            self.fact_concepts = None

    def enable_fact_concept_refiner(self, heads=None, gate_init=-2.0):
        if self.fact_concepts is None:
            self.fact_concept_refine = False
            return self
        if self.fact_concept_refiner is None:
            self.fact_concept_refiner = SchemaConceptRefiner(
                self.d, heads=int(heads or self.heads), gate_init=gate_init)
            self.fact_concept_refiner.to(next(self.parameters()).device)
            self.fact_concept_refine_gate_init = float(gate_init)
        self.fact_concept_refine = True
        return self

    def enable_fact_concept_mixer(self, heads=None, layers=1, gate_init=-2.0):
        if self.fact_concepts is None:
            self.fact_concept_mixer_layers = 0
            return self
        self.fact_concepts.enable_mixer(
            heads=int(heads or self.heads), layers=int(layers), gate_init=gate_init)
        if self.fact_concepts.mixer is not None:
            self.fact_concepts.mixer.to(next(self.parameters()).device)
        self.fact_concept_mixer_layers = int(layers)
        self.fact_concept_mixer_gate_init = float(gate_init)
        return self

    def enable_latent_concepts(self, slots, heads=None, layers=1):
        slots = int(slots)
        latent_heads = int(heads or self.heads)
        latent_layers = int(layers)
        if slots <= 0:
            self.latent_concepts = None
            self.latent_concept_slots = 0
            self.latent_concept_prefix = False
            self.latent_concept_refine = False
            self.latent_concept_refiner = None
            self.latent_concept_memory = None
            self.latent_concept_memory_size = 0
            return self
        if (self.latent_concepts is None
                or self.latent_concept_slots != slots
                or getattr(self.latent_concepts, "heads", latent_heads) != latent_heads
                or getattr(self.latent_concepts, "mixer_layers", latent_layers)
                != latent_layers):
            self.latent_concepts = LatentConceptHead(
                slots, self.d, heads=latent_heads, mixer_layers=latent_layers)
            self.latent_concepts.to(next(self.parameters()).device)
        self.latent_concept_slots = slots
        self.latent_concept_layers = latent_layers
        return self

    def enable_latent_concept_memory(self, size):
        size = int(size)
        if size <= 0 or self.latent_concepts is None:
            self.latent_concept_memory = None
            self.latent_concept_memory_size = 0
            return self
        if (self.latent_concept_memory is None
                or self.latent_concept_memory_size != size):
            self.latent_concept_memory = LatentConceptMemory(size, self.d)
            self.latent_concept_memory.to(next(self.parameters()).device)
        self.latent_concept_memory_size = size
        return self

    def latent_concept_memory_loss(self, slots, temperature=0.1, balance_w=0.0):
        if self.latent_concept_memory is None:
            if slots is not None:
                return slots.sum() * 0.0
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.latent_concept_memory(
            slots, temperature=temperature, balance_w=balance_w)

    def latent_concept_association_loss(self, slots, temperature=0.1,
                                        target_power=1.0, self_loop_w=0.05,
                                        transitive_steps=1, transitive_w=0.0):
        if self.latent_concept_memory is None:
            if slots is not None:
                return slots.sum() * 0.0
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.latent_concept_memory.association_loss(
            slots, temperature=temperature, target_power=target_power,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w)

    def latent_concept_composition_loss(self, slots, temperature=0.1,
                                        self_loop_w=0.0, transitive_steps=2,
                                        transitive_w=0.1, margin=0.0):
        if self.latent_concept_memory is None:
            if slots is not None:
                return slots.sum() * 0.0
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.latent_concept_memory.composition_loss(
            slots, temperature=temperature, self_loop_w=self_loop_w,
            transitive_steps=transitive_steps, transitive_w=transitive_w,
            margin=margin)

    def latent_concept_graph_prediction_loss(
            self, source_slots, target_slots, temperature=0.1,
            self_loop_w=0.05, transitive_steps=2, transitive_w=0.1,
            target_power=1.0):
        if self.latent_concept_memory is None:
            if source_slots is not None:
                return source_slots.sum() * 0.0
            if target_slots is not None:
                return target_slots.sum() * 0.0
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.latent_concept_memory.graph_prediction_loss(
            source_slots, target_slots, temperature=temperature,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w, target_power=target_power)

    def latent_concept_graph_cycle_loss(
            self, source_slots, target_slots, temperature=0.1,
            self_loop_w=0.05, transitive_steps=2, transitive_w=0.1,
            target_power=1.0, cycle_w=0.5):
        if self.latent_concept_memory is None:
            if source_slots is not None:
                return source_slots.sum() * 0.0
            if target_slots is not None:
                return target_slots.sum() * 0.0
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.latent_concept_memory.graph_cycle_loss(
            source_slots, target_slots, temperature=temperature,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w, target_power=target_power,
            cycle_w=cycle_w)

    def latent_concept_graph_cycle_scores(
            self, source_slots, target_slots, temperature=0.1,
            self_loop_w=0.05, transitive_steps=2, transitive_w=0.1,
            target_power=1.0, cycle_w=0.5):
        if self.latent_concept_memory is None:
            zero = source_slots if source_slots is not None else target_slots
            if zero is None:
                zero = torch.zeros(0, device=next(self.parameters()).device)
            else:
                zero = zero.reshape(zero.shape[0], -1).sum(-1) * 0.0
            return zero, {
                "forward_kl": zero, "reverse_kl": zero,
                "source_cycle_kl": zero, "target_cycle_kl": zero,
            }
        return self.latent_concept_memory.graph_cycle_scores(
            source_slots, target_slots, temperature=temperature,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w, target_power=target_power,
            cycle_w=cycle_w)

    def latent_concept_graph_prediction_scores(
            self, source_slots, target_slots, temperature=0.1,
            self_loop_w=0.05, transitive_steps=2, transitive_w=0.1,
            target_power=1.0):
        if self.latent_concept_memory is None:
            zero = source_slots if source_slots is not None else target_slots
            if zero is None:
                zero = torch.zeros(0, device=next(self.parameters()).device)
            else:
                zero = zero.reshape(zero.shape[0], -1).sum(-1) * 0.0
            return zero, {"kl": zero, "cosine": zero}
        return self.latent_concept_memory.graph_prediction_scores(
            source_slots, target_slots, temperature=temperature,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w, target_power=target_power)

    @torch.no_grad()
    def update_latent_concept_memory(self, slots, momentum=0.95,
                                     relation_decay=None):
        if self.latent_concept_memory is None:
            return 0
        return self.latent_concept_memory.update(
            slots, momentum=momentum, relation_decay=relation_decay)

    @torch.no_grad()
    def update_latent_concept_transitions(self, source_slots, target_slots,
                                          decay=0.99):
        if self.latent_concept_memory is None:
            return 0
        return self.latent_concept_memory.update_transitions(
            source_slots, target_slots, decay=decay)

    def enable_latent_concept_refiner(self, heads=None, gate_init=-2.0):
        if self.latent_concepts is None:
            self.latent_concept_refine = False
            return self
        if self.latent_concept_refiner is None:
            self.latent_concept_refiner = SchemaConceptRefiner(
                self.d, heads=int(heads or self.heads), gate_init=gate_init)
            self.latent_concept_refiner.to(next(self.parameters()).device)
            self.latent_concept_refine_gate_init = float(gate_init)
        self.latent_concept_refine = True
        return self

    def encode_text(self, txt, feature_dropout=0.0, use_latent_refine=True):
        mask = txt.eq(self.txt.pad)
        prefix = self.txt(txt)
        if self.training and feature_dropout > 0.0:
            prefix = F.dropout(prefix, p=float(feature_dropout), training=True)
            prefix = prefix.masked_fill(mask.unsqueeze(-1), 0.0)
        if (self.fact_concept_refine and self.fact_concepts is not None
                and self.fact_concept_refiner is not None):
            concepts = self.fact_concepts.state_tensor(prefix, mask=mask)
            prefix = self.fact_concept_refiner(prefix, concepts, mask=mask)
        if (use_latent_refine and self.latent_concept_refine
                and self.latent_concepts is not None
                and self.latent_concept_refiner is not None):
            latent = self.latent_concepts(prefix, mask=mask)
            prefix = self.latent_concept_refiner(prefix, latent, mask=mask)
        keep = (~mask).unsqueeze(-1)
        pooled = (prefix * keep).sum(1) / keep.sum(1).clamp(min=1)
        return prefix, pooled

    def latent_concept_states(self, txt, feature_dropout=0.0, project=False):
        if self.latent_concepts is None:
            return None
        prefix, _pooled = self.encode_text(
            txt, feature_dropout=feature_dropout, use_latent_refine=False)
        return self.latent_concepts(prefix, mask=txt.eq(self.txt.pad), project=project)

    def forward(self, txt, ids):
        prefix = self.decoder_prefix(txt)
        logits = self.lm(ids, prefix=prefix)
        return logits[:, prefix.shape[1]:]

    def decoder_prefix(self, txt):
        prefix, _pooled = self.encode_text(txt)
        extra = []
        mask = txt.eq(self.txt.pad)
        if self.latent_concept_prefix and self.latent_concepts is not None:
            extra.append(self.latent_concepts(prefix, mask=mask))
        if self.fact_concept_prefix and self.fact_concepts is not None:
            extra.append(self.fact_concepts.state_tensor(prefix, mask=mask))
        if extra:
            prefix = torch.cat(extra + [prefix], dim=1)
        return prefix

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

    def fact_concept_states(self, txt):
        if self.fact_concepts is None:
            return {}
        prefix, _pooled = self.encode_text(txt)
        return self.fact_concepts.states(prefix, mask=txt.eq(self.txt.pad))

    def fact_concept_geometry_states(self, txt):
        if self.fact_concepts is None:
            return {}
        prefix, _pooled = self.encode_text(txt)
        return self.fact_concepts.geometry_states(prefix, mask=txt.eq(self.txt.pad))

    def fact_concept_geometry_logits(self, txt, temperature=0.1):
        if self.fact_concepts is None:
            return {}
        prefix, _pooled = self.encode_text(txt)
        return self.fact_concepts.geometry_logits(
            prefix, mask=txt.eq(self.txt.pad), temperature=temperature)

    def fact_concept_logits(self, txt):
        if self.fact_concepts is None:
            return {}
        prefix, _pooled = self.encode_text(txt)
        return self.fact_concepts(prefix, mask=txt.eq(self.txt.pad))

    def latent_fact_concept_state_tensor(self, txt):
        if self.fact_concepts is None or self.latent_concepts is None:
            return None
        prefix, _pooled = self.encode_text(txt)
        latent = self.latent_concepts(prefix, mask=txt.eq(self.txt.pad))
        return self.fact_concepts.state_tensor(latent)

    def latent_fact_concept_states(self, txt):
        state_tensor = self.latent_fact_concept_state_tensor(txt)
        if state_tensor is None:
            return {}
        return {key: state_tensor[:, i] for i, key in enumerate(self.fact_concepts.keys)}

    def latent_fact_concept_logits(self, txt):
        state_tensor = self.latent_fact_concept_state_tensor(txt)
        if state_tensor is None:
            return {}
        return self.fact_concepts.logits_from_state_tensor(state_tensor)

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


def pack_reading(records, vocab, device):
    text_ids = [vocab.enc(r.tokens) for r in records]
    max_t = max(len(x) for x in text_ids)
    txt = torch.full((len(records), max_t), vocab.pad, dtype=torch.long, device=device)
    for i, ids in enumerate(text_ids):
        txt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return txt


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


def copy_pretrained_text_weights(src_model, src_vocab, dst_model, dst_vocab):
    """Copy compatible checkpoint weights into an expanded text model.

    New tokens and new fact values keep their random initialization. Existing tokens, transformer
    blocks, and semantic-head rows are copied by symbolic identity rather than array position.
    """
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    skip_prefixes = ("txt.emb.", "lm.tok.", "lm.head.", "fact_query",
                     "fact_heads.", "fact_concepts.")
    with torch.no_grad():
        for name, dst_val in dst_state.items():
            if name.startswith(skip_prefixes):
                continue
            src_val = src_state.get(name)
            if src_val is not None and src_val.shape == dst_val.shape:
                dst_val.copy_(src_val)

        for tok, src_idx in src_vocab.stoi.items():
            dst_idx = dst_vocab.stoi.get(tok)
            if dst_idx is None:
                continue
            dst_model.txt.emb.weight[dst_idx].copy_(src_model.txt.emb.weight[src_idx])
            dst_model.lm.tok.weight[dst_idx].copy_(src_model.lm.tok.weight[src_idx])

        if src_model.fact_schema is not None and dst_model.fact_schema is not None:
            dst_keys = {key: i for i, key in enumerate(dst_model.fact_schema.keys)}
            for src_i, key in enumerate(src_model.fact_schema.keys):
                dst_i = dst_keys.get(key)
                if dst_i is None:
                    continue
                dst_model.fact_query[dst_i].copy_(src_model.fact_query[src_i])
                src_head = src_model.fact_heads[str(src_i)]
                dst_head = dst_model.fact_heads[str(dst_i)]
                dst_vals = {val: i for i, val in enumerate(dst_model.fact_schema.values[dst_i])}
                for src_v, val in enumerate(src_model.fact_schema.values[src_i]):
                    dst_v = dst_vals.get(val)
                    if dst_v is None:
                        continue
                    dst_head.weight[dst_v].copy_(src_head.weight[src_v])
                    if src_head.bias is not None and dst_head.bias is not None:
                        dst_head.bias[dst_v].copy_(src_head.bias[src_v])
            if (getattr(src_model, "has_fact_concept_state", False)
                    and getattr(src_model, "fact_concepts", None) is not None
                    and getattr(dst_model, "fact_concepts", None) is not None):
                for src_i, key in enumerate(src_model.fact_schema.keys):
                    dst_i = dst_keys.get(key)
                    if dst_i is None:
                        continue
                    dst_model.fact_concepts.key_query[dst_i].copy_(
                        src_model.fact_concepts.key_query[src_i])
                    dst_vals = {val: i for i, val in enumerate(
                        dst_model.fact_schema.values[dst_i])}
                    for src_v, val in enumerate(src_model.fact_schema.values[src_i]):
                        dst_v = dst_vals.get(val)
                        if dst_v is None:
                            continue
                        dst_model.fact_concepts.value_embeds[dst_i][dst_v].copy_(
                            src_model.fact_concepts.value_embeds[src_i][src_v])
                        dst_model.fact_concepts.value_biases[dst_i][dst_v].copy_(
                            src_model.fact_concepts.value_biases[src_i][src_v])
                        src_proto_prefix = (
                            f"fact_concepts.geometry_prototypes.{src_i}.")
                        if any(name.startswith(src_proto_prefix) for name in src_state):
                            dst_model.fact_concepts.geometry_prototypes[dst_i][dst_v].copy_(
                                src_model.fact_concepts.geometry_prototypes[src_i][src_v])
                    src_proj_prefix = f"fact_concepts.state_projectors.{src_i}."
                    if any(name.startswith(src_proj_prefix) for name in src_state):
                        dst_model.fact_concepts.state_projectors[dst_i].load_state_dict(
                            src_model.fact_concepts.state_projectors[src_i].state_dict())
                src_mixer = getattr(src_model.fact_concepts, "mixer", None)
                dst_mixer = getattr(dst_model.fact_concepts, "mixer", None)
                if src_mixer is not None and dst_mixer is not None:
                    src_mixer_state = src_mixer.state_dict()
                    dst_mixer_state = dst_mixer.state_dict()
                    for name, dst_val in dst_mixer_state.items():
                        if name == "key_pos":
                            continue
                        src_val = src_mixer_state.get(name)
                        if src_val is not None and src_val.shape == dst_val.shape:
                            dst_val.copy_(src_val)
                    for src_i, key in enumerate(src_model.fact_schema.keys):
                        dst_i = dst_keys.get(key)
                        if dst_i is not None:
                            dst_mixer.key_pos[dst_i].copy_(src_mixer.key_pos[src_i])
    return dst_model


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


def fact_concept_target_ids(records, schema, device):
    targets = {key: torch.full((len(records),), -1, dtype=torch.long, device=device)
               for key in schema.keys}
    value_index = schema.value_index
    for i, rec in enumerate(records):
        seen = set()
        for slot, pred, val in rec.facts:
            key = (slot, pred)
            if key in seen:
                continue
            idx = value_index.get((key, val))
            if idx is not None and key in targets:
                targets[key][i] = idx
                seen.add(key)
    return targets


def fact_concept_loss(model, txt, records, schema):
    if schema is None or getattr(model, "fact_concepts", None) is None:
        return torch.tensor(0.0, device=txt.device)
    logits = model.fact_concept_logits(txt)
    targets_by_key = fact_concept_target_ids(records, schema, txt.device)
    losses = []
    for key in schema.keys:
        row = logits.get(key)
        targets = targets_by_key[key]
        if row is not None and targets.ge(0).any() and row.shape[-1] > 1:
            losses.append(F.cross_entropy(row, targets, ignore_index=-1))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def fact_concept_contrastive_loss(model, txt, records, schema, temperature=0.1):
    if schema is None or getattr(model, "fact_concepts", None) is None:
        return torch.tensor(0.0, device=txt.device)
    states = model.fact_concept_geometry_states(txt)
    targets = fact_concept_target_ids(records, schema, txt.device)
    return schema_concept_contrastive_loss(states, targets, temperature=temperature)


def fact_concept_batch_centroid_loss(model, txt, records, schema, temperature=0.1,
                                     margin=0.0):
    if schema is None or getattr(model, "fact_concepts", None) is None:
        return torch.tensor(0.0, device=txt.device)
    states = model.fact_concept_geometry_states(txt)
    targets = fact_concept_target_ids(records, schema, txt.device)
    return schema_concept_batch_centroid_loss(
        states, targets, temperature=temperature, margin=margin)


def fact_concept_prototype_loss(model, txt, records, schema, temperature=0.1):
    if schema is None or getattr(model, "fact_concepts", None) is None:
        return torch.tensor(0.0, device=txt.device)
    states = model.fact_concept_geometry_states(txt)
    targets = fact_concept_target_ids(records, schema, txt.device)
    prototypes = {key: model.fact_concepts.geometry_prototypes[i]
                  for i, key in enumerate(model.fact_concepts.keys)}
    return schema_concept_prototype_alignment_loss(
        states, targets, prototypes, temperature=temperature)


def fact_concept_prototype_spread_loss(model, margin=0.2):
    head = getattr(model, "fact_concepts", None)
    if head is None:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)
    prototypes = {key: head.geometry_prototypes[i] for i, key in enumerate(head.keys)}
    return schema_concept_prototype_spread_loss(prototypes, margin=margin)


def fact_concept_state_spread_loss(model, txt, records, schema, variance_target=0.05,
                                   centroid_margin=0.2, covariance_weight=0.05):
    if schema is None or getattr(model, "fact_concepts", None) is None:
        return torch.tensor(0.0, device=txt.device)
    states = model.fact_concept_geometry_states(txt)
    targets = fact_concept_target_ids(records, schema, txt.device)
    return schema_concept_state_spread_loss(
        states, targets, variance_target=variance_target,
        centroid_margin=centroid_margin, covariance_weight=covariance_weight)


def latent_text_concept_loss(model, txt, view_dropout=0.1,
                             invariance_w=25.0, variance_w=25.0,
                             covariance_w=1.0, variance_target=1.0):
    if getattr(model, "latent_concepts", None) is None:
        return torch.tensor(0.0, device=txt.device)
    a = model.latent_concept_states(txt, feature_dropout=view_dropout, project=True)
    b = model.latent_concept_states(txt, feature_dropout=view_dropout, project=True)
    return latent_concept_vicreg_loss(
        a, b, invariance_weight=invariance_w, variance_weight=variance_w,
        covariance_weight=covariance_w, variance_target=variance_target)


def corrupt_reading_tokens(txt, pad, unk, token_drop_p=0.15, token_replace_p=0.05):
    out = txt.clone()
    valid = out.ne(pad)
    dropped = torch.zeros_like(valid)
    if token_drop_p > 0.0:
        dropped = torch.rand(out.shape, device=out.device).lt(float(token_drop_p)) & valid
        out = out.masked_fill(dropped, pad)
    if token_replace_p > 0.0:
        replaced = (torch.rand(out.shape, device=out.device).lt(float(token_replace_p))
                    & valid & ~dropped)
        out = out.masked_fill(replaced, int(unk))
    empty = out.ne(pad).sum(1).eq(0)
    if bool(empty.any()):
        has_token = valid.any(1)
        rows = torch.where(empty & has_token)[0]
        if int(rows.numel()):
            first = valid.to(torch.long).argmax(1)
            out[rows, first[rows]] = txt[rows, first[rows]]
    return out


def reading_latent_view_loss(model, txt, pad, unk, token_drop_p=0.15,
                             token_replace_p=0.05, feature_dropout=0.1,
                             invariance_w=25.0, variance_w=25.0,
                             covariance_w=1.0, variance_target=1.0):
    if getattr(model, "latent_concepts", None) is None:
        return torch.tensor(0.0, device=txt.device)
    view_a = corrupt_reading_tokens(
        txt, pad, unk, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p)
    view_b = corrupt_reading_tokens(
        txt, pad, unk, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p)
    a = model.latent_concept_states(view_a, feature_dropout=feature_dropout, project=True)
    b = model.latent_concept_states(view_b, feature_dropout=feature_dropout, project=True)
    return latent_concept_vicreg_loss(
        a, b, invariance_weight=invariance_w, variance_weight=variance_w,
        covariance_weight=covariance_w, variance_target=variance_target)


def reading_latent_factorization_loss(model, txt, feature_dropout=0.1,
                                      variance_target=0.05,
                                      separation_margin=0.2,
                                      covariance_w=0.05):
    if getattr(model, "latent_concepts", None) is None:
        return torch.tensor(0.0, device=txt.device)
    slots = model.latent_concept_states(
        txt, feature_dropout=feature_dropout, project=True)
    return latent_concept_slot_factorization_loss(
        slots, variance_target=variance_target,
        separation_margin=separation_margin, covariance_weight=covariance_w)


def reading_latent_fer_loss(model, txt, feature_dropout=0.1,
                            fragmentation_w=1.0, correlation_w=1.0,
                            balance_w=0.1):
    if getattr(model, "latent_concepts", None) is None:
        zero = torch.tensor(0.0, device=txt.device)
        return zero, {"fer_score": zero, "fragmentation": zero,
                      "slot_correlation": zero, "slot_imbalance": zero}
    slots = model.latent_concept_states(
        txt, feature_dropout=feature_dropout, project=True)
    loss = latent_concept_fer_loss(
        slots, fragmentation_w=fragmentation_w, correlation_w=correlation_w,
        balance_w=balance_w)
    metrics = latent_concept_fer_metrics(slots)
    return loss, metrics


def reading_latent_memory_loss(model, txt, feature_dropout=0.1,
                               temperature=0.1, balance_w=0.0):
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "latent_concept_memory", None) is None):
        return torch.tensor(0.0, device=txt.device)
    slots = model.latent_concept_states(
        txt, feature_dropout=feature_dropout, project=True)
    return model.latent_concept_memory_loss(
        slots, temperature=temperature, balance_w=balance_w)


def reading_latent_association_loss(model, txt, feature_dropout=0.1,
                                    temperature=0.1, target_power=1.0,
                                    self_loop_w=0.05, transitive_steps=1,
                                    transitive_w=0.0):
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "latent_concept_memory", None) is None):
        return torch.tensor(0.0, device=txt.device)
    slots = model.latent_concept_states(
        txt, feature_dropout=feature_dropout, project=True)
    return model.latent_concept_association_loss(
        slots, temperature=temperature, target_power=target_power,
        self_loop_w=self_loop_w, transitive_steps=transitive_steps,
        transitive_w=transitive_w)


def reading_latent_composition_loss(model, txt, feature_dropout=0.1,
                                    temperature=0.1, self_loop_w=0.0,
                                    transitive_steps=2, transitive_w=0.1,
                                    margin=0.0):
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "latent_concept_memory", None) is None):
        return torch.tensor(0.0, device=txt.device)
    slots = model.latent_concept_states(
        txt, feature_dropout=feature_dropout, project=True)
    if hasattr(model, "latent_concept_composition_loss"):
        return model.latent_concept_composition_loss(
            slots, temperature=temperature, self_loop_w=self_loop_w,
            transitive_steps=transitive_steps, transitive_w=transitive_w,
            margin=margin)
    return latent_concept_composition_loss(
        slots, model.latent_concept_memory.active(),
        model.latent_concept_memory.active_relations(),
        temperature=temperature, self_loop_w=self_loop_w,
        transitive_steps=transitive_steps, transitive_w=transitive_w,
        margin=margin)


def reading_latent_bridge_loss(model, txt, feature_dropout=0.1):
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "latent_concept_memory", None) is None):
        return torch.tensor(0.0, device=txt.device)
    slots = model.latent_concept_states(
        txt, feature_dropout=feature_dropout, project=True)
    memory = model.latent_concept_memory
    return latent_concept_bridge_loss(
        slots, memory.active(), memory.active_relations(),
        memory.active_transitions())


@torch.no_grad()
def update_reading_latent_memory(model, txt, feature_dropout=0.0,
                                 momentum=0.95, relation_decay=None):
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "latent_concept_memory", None) is None):
        return 0
    was_training = model.training
    model.eval()
    slots = model.latent_concept_states(
        txt, feature_dropout=feature_dropout, project=True)
    if was_training:
        model.train()
    return model.update_latent_concept_memory(
        slots, momentum=momentum, relation_decay=relation_decay)


def split_reading_context_target(txt, pad, context_keep_p=0.5):
    keep_p = float(context_keep_p)
    if keep_p <= 0.0 or keep_p >= 1.0:
        raise ValueError("reading context keep probability must be in (0, 1)")
    valid = txt.ne(pad)
    context = torch.rand(txt.shape, device=txt.device).lt(keep_p) & valid
    target = valid & ~context
    lengths = valid.sum(1)
    positions = torch.arange(txt.shape[1], device=txt.device).unsqueeze(0)
    first = valid.to(torch.long).argmax(1)
    last_pos = positions.masked_fill(~valid, -1).max(1).values.clamp_min(0)
    no_context = valid.any(1) & ~context.any(1)
    if bool(no_context.any()):
        rows = torch.where(no_context)[0]
        context[rows, first[rows]] = True
        target[rows, first[rows]] = False
    no_target = valid.any(1) & ~target.any(1)
    movable = no_target & lengths.gt(1)
    if bool(movable.any()):
        rows = torch.where(movable)[0]
        target[rows, last_pos[rows]] = True
        context[rows, last_pos[rows]] = False
    single = no_target & lengths.eq(1)
    if bool(single.any()):
        rows = torch.where(single)[0]
        target[rows, first[rows]] = True
    context_txt = txt.masked_fill(~context, pad)
    target_txt = txt.masked_fill(~target, pad)
    return context_txt, target_txt


def reading_context_target_loss(model, txt, pad, context_keep_p=0.5,
                                feature_dropout=0.1, temperature=0.1):
    if getattr(model, "latent_concepts", None) is None:
        return torch.tensor(0.0, device=txt.device)
    if txt.shape[0] <= 1:
        return txt.float().sum() * 0.0
    context_txt, target_txt = split_reading_context_target(
        txt, pad, context_keep_p=context_keep_p)
    context_slots = model.latent_concept_states(
        context_txt, feature_dropout=feature_dropout, project=False)
    target_slots = model.latent_concept_states(
        target_txt, feature_dropout=0.0, project=False).detach()
    predicted = model.reading_predictor(context_slots)
    predicted = F.normalize(predicted.reshape(predicted.shape[0], -1), dim=-1)
    target = F.normalize(target_slots.reshape(target_slots.shape[0], -1), dim=-1)
    logits = predicted.matmul(target.t()) / max(float(temperature), 1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def reading_context_graph_prediction_loss(
        model, txt, pad, context_keep_p=0.5, feature_dropout=0.1,
        temperature=0.1, self_loop_w=0.05, transitive_steps=2,
        transitive_w=0.1, target_power=1.0):
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "latent_concept_memory", None) is None):
        return torch.tensor(0.0, device=txt.device)
    context_txt, target_txt = split_reading_context_target(
        txt, pad, context_keep_p=context_keep_p)
    source_slots = model.latent_concept_states(
        context_txt, feature_dropout=feature_dropout, project=True)
    target_slots = model.latent_concept_states(
        target_txt, feature_dropout=0.0, project=True).detach()
    if hasattr(model, "latent_concept_graph_prediction_loss"):
        return model.latent_concept_graph_prediction_loss(
            source_slots, target_slots, temperature=temperature,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w, target_power=target_power)
    return latent_concept_graph_prediction_loss(
        source_slots, target_slots, model.latent_concept_memory.active(),
        model.latent_concept_memory.active_prediction_relations(),
        temperature=temperature, self_loop_w=self_loop_w,
        transitive_steps=transitive_steps, transitive_w=transitive_w,
        target_power=target_power)


def reading_context_graph_cycle_loss(
        model, txt, pad, context_keep_p=0.5, feature_dropout=0.1,
        temperature=0.1, self_loop_w=0.05, transitive_steps=2,
        transitive_w=0.1, target_power=1.0, cycle_w=0.5):
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "latent_concept_memory", None) is None):
        return torch.tensor(0.0, device=txt.device)
    context_txt, target_txt = split_reading_context_target(
        txt, pad, context_keep_p=context_keep_p)
    source_slots = model.latent_concept_states(
        context_txt, feature_dropout=feature_dropout, project=True)
    target_slots = model.latent_concept_states(
        target_txt, feature_dropout=feature_dropout, project=True)
    if hasattr(model, "latent_concept_graph_cycle_loss"):
        return model.latent_concept_graph_cycle_loss(
            source_slots, target_slots, temperature=temperature,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w, target_power=target_power,
            cycle_w=cycle_w)
    return latent_concept_graph_cycle_loss(
        source_slots, target_slots, model.latent_concept_memory.active(),
        model.latent_concept_memory.active_prediction_relations(),
        temperature=temperature, self_loop_w=self_loop_w,
        transitive_steps=transitive_steps, transitive_w=transitive_w,
        target_power=target_power, cycle_w=cycle_w)


@torch.no_grad()
def update_reading_latent_transitions(model, txt, pad, context_keep_p=0.5,
                                      feature_dropout=0.0, decay=0.99):
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "latent_concept_memory", None) is None):
        return 0
    was_training = model.training
    model.eval()
    context_txt, target_txt = split_reading_context_target(
        txt, pad, context_keep_p=context_keep_p)
    source_slots = model.latent_concept_states(
        context_txt, feature_dropout=feature_dropout, project=True)
    target_slots = model.latent_concept_states(
        target_txt, feature_dropout=0.0, project=True)
    if was_training:
        model.train()
    if hasattr(model, "update_latent_concept_transitions"):
        return model.update_latent_concept_transitions(
            source_slots, target_slots, decay=decay)
    return model.latent_concept_memory.update_transitions(
        source_slots, target_slots, decay=decay)


def reading_teacher_latent_consistency_loss(model, teacher_model, records, vocab,
                                            teacher_vocab, device=DEV,
                                            feature_dropout=0.0):
    if (teacher_model is None or teacher_vocab is None
            or getattr(model, "latent_concepts", None) is None
            or getattr(teacher_model, "latent_concepts", None) is None):
        dev = next(model.parameters()).device
        return torch.tensor(0.0, device=dev)
    if not records:
        dev = next(model.parameters()).device
        return torch.tensor(0.0, device=dev)
    student_txt = pack_reading(records, vocab, device)
    teacher_txt = pack_reading(records, teacher_vocab, device)
    student = model.latent_concept_states(
        student_txt, feature_dropout=feature_dropout, project=True)
    with torch.no_grad():
        teacher_model.eval()
        teacher = teacher_model.latent_concept_states(teacher_txt, project=True)
    if student is None or teacher is None or tuple(student.shape) != tuple(teacher.shape):
        return student_txt.float().sum() * 0.0
    student = F.normalize(student.reshape(student.shape[0], -1), dim=-1)
    teacher = F.normalize(teacher.reshape(teacher.shape[0], -1), dim=-1)
    return (1.0 - (student * teacher).sum(-1)).mean()


def _reading_latent_pair_embeddings(model, txt, vocab, seed=0, token_drop_p=0.15,
                                    token_replace_p=0.05):
    torch.manual_seed(int(seed))
    view_a = corrupt_reading_tokens(
        txt, vocab.pad, vocab.unk, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p)
    torch.manual_seed(int(seed) + 1)
    view_b = corrupt_reading_tokens(
        txt, vocab.pad, vocab.unk, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p)
    a = model.latent_concept_states(view_a, project=True)
    b = model.latent_concept_states(view_b, project=True)
    a = F.normalize(a.reshape(a.shape[0], -1), dim=-1)
    b = F.normalize(b.reshape(b.shape[0], -1), dim=-1)
    return a, b


def _reading_context_target_embeddings(model, txt, pad, seed=0, context_keep_p=0.5):
    torch.manual_seed(int(seed))
    context_txt, target_txt = split_reading_context_target(
        txt, pad, context_keep_p=context_keep_p)
    context_slots = model.latent_concept_states(context_txt, project=False)
    target_slots = model.latent_concept_states(target_txt, project=False)
    predicted = model.reading_predictor(context_slots)
    predicted = F.normalize(predicted.reshape(predicted.shape[0], -1), dim=-1)
    target = F.normalize(target_slots.reshape(target_slots.shape[0], -1), dim=-1)
    return predicted, target


def _reading_latent_embeddings(model, vocab, records, device=DEV, batch=64):
    if getattr(model, "latent_concepts", None) is None or not records:
        return None
    chunks = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(records), int(batch)):
            txt = pack_reading(records[off:off + int(batch)], vocab, device)
            slots = model.latent_concept_states(txt, project=True)
            if slots is None:
                return None
            chunks.append(F.normalize(slots.reshape(slots.shape[0], -1), dim=-1))
    return torch.cat(chunks, dim=0) if chunks else None


def mine_reading_latent_neighbors(model, vocab, records, device=DEV, n=0, seed=0,
                                  split="train"):
    """Mine nearest reading chunks from the model's current latent concept space."""
    if getattr(model, "latent_concepts", None) is None:
        return [], {"n_records": 0, "n_pairs": 0, "sampled": False,
                    "split": split, "skipped": True}
    candidates = [r for r in records if r.split == split]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=int(n), replace=False)
        candidates = [candidates[int(i)] for i in idx]
    if len(candidates) < 2:
        return [], {"n_records": len(candidates), "n_pairs": 0,
                    "sampled": sampled, "split": split, "skipped": True}
    emb = _reading_latent_embeddings(model, vocab, candidates, device=device)
    if emb is None or emb.shape[0] < 2:
        return [], {"n_records": len(candidates), "n_pairs": 0,
                    "sampled": sampled, "split": split, "skipped": True}
    sim = emb.matmul(emb.t())
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    masked = sim.masked_fill(eye, -float("inf"))
    nearest = masked.argmax(-1).detach().cpu().tolist()
    nearest_scores = masked.max(-1).values.detach().cpu().tolist()
    pairs = [(candidates[i], candidates[int(j)]) for i, j in enumerate(nearest)
             if int(j) != i]
    mutual = sum(1 for i, j in enumerate(nearest)
                 if int(j) != i and int(nearest[int(j)]) == i)
    return pairs, {"n_records": len(candidates),
                   "n_pairs": len(pairs),
                   "sampled": sampled,
                   "split": split,
                   "mean_neighbor_cosine": float(np.mean(nearest_scores)),
                   "min_neighbor_cosine": float(np.min(nearest_scores)),
                   "max_neighbor_cosine": float(np.max(nearest_scores)),
                   "mutual_pairs": int(mutual),
                   "skipped": False}


def mine_reading_latent_clusters(model, vocab, records, device=DEV, n=0, seed=0,
                                 split="train", min_cluster_size=2):
    """Discover reusable reading clusters from current latent nearest-neighbor links."""
    if getattr(model, "latent_concepts", None) is None:
        return [], {"n_records": 0, "n_clusters": 0, "n_clustered_records": 0,
                    "sampled": False, "split": split, "skipped": True}
    candidates = [r for r in records if r.split == split]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=int(n), replace=False)
        candidates = [candidates[int(i)] for i in idx]
    if len(candidates) < int(min_cluster_size):
        return [], {"n_records": len(candidates), "n_clusters": 0,
                    "n_clustered_records": 0, "sampled": sampled,
                    "split": split, "skipped": True}
    emb = _reading_latent_embeddings(model, vocab, candidates, device=device)
    if emb is None or emb.shape[0] < int(min_cluster_size):
        return [], {"n_records": len(candidates), "n_clusters": 0,
                    "n_clustered_records": 0, "sampled": sampled,
                    "split": split, "skipped": True}
    sim = emb.matmul(emb.t())
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    nearest = sim.masked_fill(eye, -float("inf")).argmax(-1).detach().cpu().tolist()
    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for i, j in enumerate(nearest):
        if int(j) != i:
            union(i, int(j))
    groups = {}
    for i, rec in enumerate(candidates):
        groups.setdefault(find(i), []).append(rec)
    clusters = [rows for rows in groups.values()
                if len(rows) >= int(min_cluster_size)]
    sizes = [len(rows) for rows in clusters]
    return clusters, {"n_records": len(candidates),
                      "n_clusters": len(clusters),
                      "n_clustered_records": int(sum(sizes)),
                      "sampled": sampled,
                      "split": split,
                      "min_cluster_size": int(min_cluster_size),
                      "mean_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
                      "max_cluster_size": int(max(sizes)) if sizes else 0,
                      "skipped": not bool(clusters)}


def batch_reading_neighbor_pairs(pairs, rng, batch):
    if not pairs:
        return []
    return [pairs[int(rng.integers(len(pairs)))] for _ in range(int(batch))]


def batch_reading_cluster_records(clusters, rng, batch):
    if not clusters:
        return [], []
    batch = int(batch)
    group_n = min(len(clusters), max(1, batch // 2))
    group_idx = [int(i) for i in rng.choice(
        len(clusters), size=group_n, replace=False)]
    records = []
    cluster_ids = []
    for local_id, cluster_i in enumerate(group_idx):
        cluster = clusters[cluster_i]
        if len(cluster) >= 2:
            picks = rng.choice(len(cluster), size=2, replace=False)
            rows = [cluster[int(picks[0])], cluster[int(picks[1])]]
        else:
            rows = [cluster[0], cluster[0]]
        records.extend(rows)
        cluster_ids.extend([local_id, local_id])
    while len(records) < batch:
        local_id = int(rng.integers(group_n))
        cluster = clusters[group_idx[local_id]]
        records.append(cluster[int(rng.integers(len(cluster)))])
        cluster_ids.append(local_id)
    return records[:batch], cluster_ids[:batch]


def mine_reading_sequence_pairs(records, split="train", n=0, seed=0):
    """Pair adjacent reading chunks from the same source/order metadata."""
    groups = {}
    total = 0
    for pos, rec in enumerate(records):
        if rec.split != split:
            continue
        total += 1
        meta = rec.meta if isinstance(rec.meta, dict) else {}
        source = str(meta.get("source", meta.get("document", "__records__")))
        raw_order = meta.get(
            "chunk_index", meta.get("order", meta.get("position", pos)))
        try:
            order = float(raw_order)
        except (TypeError, ValueError):
            order = float(pos)
        groups.setdefault(source, []).append((order, pos, rec))
    pairs = []
    for rows in groups.values():
        rows = sorted(rows, key=lambda row: (row[0], row[1]))
        pairs.extend((a[2], b[2]) for a, b in zip(rows, rows[1:]))
    sampled = bool(n and n < len(pairs))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=int(n), replace=False)
        pairs = [pairs[int(i)] for i in idx]
    return pairs, {"n_records": int(total),
                   "n_pairs": len(pairs),
                   "n_sources": len(groups),
                   "sampled": sampled,
                   "split": split,
                   "skipped": not bool(pairs)}


def reading_latent_neighborhood_loss(model, pairs, vocab, device=DEV,
                                     token_drop_p=0.15, token_replace_p=0.05,
                                     feature_dropout=0.1, temperature=0.1,
                                     margin=0.0):
    if getattr(model, "latent_concepts", None) is None or not pairs:
        dev = next(model.parameters()).device
        return torch.tensor(0.0, device=dev)
    anchors = [a for a, _b in pairs]
    positives = [b for _a, b in pairs]
    anchor_txt = corrupt_reading_tokens(
        pack_reading(anchors, vocab, device), vocab.pad, vocab.unk,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p)
    positive_txt = corrupt_reading_tokens(
        pack_reading(positives, vocab, device), vocab.pad, vocab.unk,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p)
    anchor_slots = model.latent_concept_states(
        anchor_txt, feature_dropout=feature_dropout, project=True)
    positive_slots = model.latent_concept_states(
        positive_txt, feature_dropout=feature_dropout, project=True)
    return latent_concept_neighborhood_loss(
        anchor_slots, positive_slots, temperature=temperature, margin=margin)


def reading_sequence_prediction_loss(model, pairs, vocab, device=DEV,
                                     token_drop_p=0.15, token_replace_p=0.05,
                                     feature_dropout=0.1, temperature=0.1):
    if (getattr(model, "latent_concepts", None) is None
            or not hasattr(model, "reading_predictor")
            or not pairs or len(pairs) <= 1):
        dev = next(model.parameters()).device
        return torch.tensor(0.0, device=dev)
    anchors = [a for a, _b in pairs]
    positives = [b for _a, b in pairs]
    anchor_txt = corrupt_reading_tokens(
        pack_reading(anchors, vocab, device), vocab.pad, vocab.unk,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p)
    positive_txt = corrupt_reading_tokens(
        pack_reading(positives, vocab, device), vocab.pad, vocab.unk,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p)
    source_slots = model.latent_concept_states(
        anchor_txt, feature_dropout=feature_dropout, project=False)
    target_slots = model.latent_concept_states(
        positive_txt, feature_dropout=0.0, project=False).detach()
    return latent_concept_sequence_prediction_loss(
        model.reading_predictor, source_slots, target_slots,
        temperature=temperature)


@torch.no_grad()
def update_reading_sequence_transitions(model, pairs, vocab, device=DEV,
                                        feature_dropout=0.0, decay=0.99):
    memory = getattr(model, "latent_concept_memory", None)
    if (getattr(model, "latent_concepts", None) is None
            or memory is None or not pairs):
        return 0
    anchors = [a for a, _b in pairs]
    positives = [b for _a, b in pairs]
    anchor_txt = pack_reading(anchors, vocab, device)
    positive_txt = pack_reading(positives, vocab, device)
    source_slots = model.latent_concept_states(
        anchor_txt, feature_dropout=feature_dropout, project=True)
    target_slots = model.latent_concept_states(
        positive_txt, feature_dropout=0.0, project=True)
    return int(memory.update_transitions(source_slots, target_slots, decay=decay))


def reading_latent_transition_loss(model, pairs, vocab, device=DEV,
                                   token_drop_p=0.15, token_replace_p=0.05,
                                   feature_dropout=0.1, temperature=0.1,
                                   margin=0.0):
    if getattr(model, "latent_concepts", None) is None or not pairs:
        dev = next(model.parameters()).device
        return torch.tensor(0.0, device=dev)
    anchors = [a for a, _b in pairs]
    positives = [b for _a, b in pairs]
    anchor_raw = pack_reading(anchors, vocab, device)
    positive_raw = pack_reading(positives, vocab, device)
    anchor_a = corrupt_reading_tokens(
        anchor_raw, vocab.pad, vocab.unk, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p)
    positive_a = corrupt_reading_tokens(
        positive_raw, vocab.pad, vocab.unk, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p)
    anchor_b = corrupt_reading_tokens(
        anchor_raw, vocab.pad, vocab.unk, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p)
    positive_b = corrupt_reading_tokens(
        positive_raw, vocab.pad, vocab.unk, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p)
    anchor_a_slots = model.latent_concept_states(
        anchor_a, feature_dropout=feature_dropout, project=True)
    positive_a_slots = model.latent_concept_states(
        positive_a, feature_dropout=feature_dropout, project=True)
    anchor_b_slots = model.latent_concept_states(
        anchor_b, feature_dropout=feature_dropout, project=True)
    positive_b_slots = model.latent_concept_states(
        positive_b, feature_dropout=feature_dropout, project=True)
    return latent_concept_transition_consistency_loss(
        anchor_a_slots, positive_a_slots, anchor_b_slots, positive_b_slots,
        temperature=temperature, margin=margin)


def reading_latent_cluster_loss(model, records, cluster_ids, vocab, device=DEV,
                                token_drop_p=0.15, token_replace_p=0.05,
                                feature_dropout=0.1, temperature=0.1,
                                margin=0.0, min_cluster_size=2):
    if getattr(model, "latent_concepts", None) is None or not records:
        dev = next(model.parameters()).device
        return torch.tensor(0.0, device=dev)
    txt = corrupt_reading_tokens(
        pack_reading(records, vocab, device), vocab.pad, vocab.unk,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p)
    slots = model.latent_concept_states(
        txt, feature_dropout=feature_dropout, project=True)
    return latent_concept_cluster_prototype_loss(
        slots, cluster_ids, temperature=temperature, margin=margin,
        min_cluster_size=min_cluster_size)


def eval_reading_records(records, n=0, seed=0):
    rows = [r for r in records if r.split == "eval"]
    if n < 0:
        return []
    if n and n < len(rows):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[int(i)] for i in idx]
    return rows


def reading_latent_retrieval_eval(model, vocab, records, device=DEV, n=0, seed=0,
                                  token_drop_p=0.15, token_replace_p=0.05):
    if getattr(model, "latent_concepts", None) is None:
        return {"paired_view_acc": 0.0, "n_records": 0, "sampled": False,
                "skipped": True}
    selected = eval_reading_records(records, n=n, seed=seed)
    if not selected:
        return {"paired_view_acc": 0.0, "n_records": 0, "sampled": False,
                "skipped": True}
    correct = total = 0
    pos_sum = neg_sum = 0.0
    neg_count = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(selected), 64):
            batch = selected[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            a, b = _reading_latent_pair_embeddings(
                model, txt, vocab, seed=seed + off * 2,
                token_drop_p=token_drop_p,
                token_replace_p=token_replace_p)
            sim = a.matmul(b.t())
            nearest = sim.argmax(-1)
            target = torch.arange(sim.shape[0], device=sim.device)
            correct += int(nearest.eq(target).sum())
            total += int(sim.shape[0])
            pos_sum += float(sim.diag().sum())
            if sim.shape[0] > 1:
                eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
                neg_sum += float(sim.masked_select(~eye).sum())
                neg_count += int((~eye).sum())
    eval_count = len([r for r in records if r.split == "eval"])
    return {"paired_view_acc": correct / max(1, total),
            "positive_cosine": pos_sum / max(1, total),
            "negative_cosine": neg_sum / max(1, neg_count),
            "margin": (pos_sum / max(1, total)) - (neg_sum / max(1, neg_count)),
            "n_records": total,
            "sampled": bool(n > 0 and n < eval_count),
            "skipped": False}


def reading_context_target_retrieval_eval(model, vocab, records, device=DEV, n=0,
                                          seed=0, context_keep_p=0.5):
    if (getattr(model, "latent_concepts", None) is None
            or not hasattr(model, "reading_predictor")):
        return {"context_target_acc": 0.0, "n_records": 0, "sampled": False,
                "skipped": True}
    selected = eval_reading_records(records, n=n, seed=seed)
    if not selected:
        return {"context_target_acc": 0.0, "n_records": 0, "sampled": False,
                "skipped": True}
    correct = total = 0
    pos_sum = neg_sum = 0.0
    neg_count = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(selected), 64):
            batch = selected[off:off + 64]
            if len(batch) <= 1:
                continue
            txt = pack_reading(batch, vocab, device)
            predicted, target = _reading_context_target_embeddings(
                model, txt, vocab.pad, seed=seed + off * 2,
                context_keep_p=context_keep_p)
            sim = predicted.matmul(target.t())
            nearest = sim.argmax(-1)
            labels = torch.arange(sim.shape[0], device=sim.device)
            correct += int(nearest.eq(labels).sum())
            total += int(sim.shape[0])
            pos_sum += float(sim.diag().sum())
            eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            neg_sum += float(sim.masked_select(~eye).sum())
            neg_count += int((~eye).sum())
    eval_count = len([r for r in records if r.split == "eval"])
    if total == 0:
        return {"context_target_acc": 0.0, "n_records": 0,
                "sampled": bool(n > 0 and n < eval_count), "skipped": True}
    return {"context_target_acc": correct / max(1, total),
            "positive_cosine": pos_sum / max(1, total),
            "negative_cosine": neg_sum / max(1, neg_count),
            "margin": (pos_sum / max(1, total)) - (neg_sum / max(1, neg_count)),
            "n_records": total,
            "sampled": bool(n > 0 and n < eval_count),
            "skipped": False}


def reading_sequence_retrieval_eval(model, vocab, records, device=DEV, n=0,
                                    seed=0, token_drop_p=0.15,
                                    token_replace_p=0.05):
    if (getattr(model, "latent_concepts", None) is None
            or not hasattr(model, "reading_predictor")):
        return {"sequence_acc": 0.0, "n_records": 0, "n_pairs": 0,
                "sampled": False, "skipped": True}
    pairs, mine_report = mine_reading_sequence_pairs(
        records, split="eval", n=n, seed=seed)
    if len(pairs) <= 1:
        return {"sequence_acc": 0.0, "n_records": mine_report.get("n_records", 0),
                "n_pairs": len(pairs), "sampled": mine_report.get("sampled", False),
                "skipped": True, "mining": mine_report}
    correct = total = 0
    pos_sum = neg_sum = 0.0
    neg_count = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(pairs), 64):
            pair_batch = pairs[off:off + 64]
            if len(pair_batch) <= 1:
                continue
            anchors = [a for a, _b in pair_batch]
            positives = [b for _a, b in pair_batch]
            anchor_txt = corrupt_reading_tokens(
                pack_reading(anchors, vocab, device), vocab.pad, vocab.unk,
                token_drop_p=token_drop_p, token_replace_p=token_replace_p)
            positive_txt = corrupt_reading_tokens(
                pack_reading(positives, vocab, device), vocab.pad, vocab.unk,
                token_drop_p=token_drop_p, token_replace_p=token_replace_p)
            source_slots = model.latent_concept_states(anchor_txt, project=False)
            target_slots = model.latent_concept_states(positive_txt, project=False)
            predicted = model.reading_predictor(source_slots)
            predicted = F.normalize(predicted.reshape(predicted.shape[0], -1), dim=-1)
            target = F.normalize(target_slots.reshape(target_slots.shape[0], -1), dim=-1)
            sim = predicted.matmul(target.t())
            labels = torch.arange(sim.shape[0], device=sim.device)
            nearest = sim.argmax(-1)
            correct += int(nearest.eq(labels).sum())
            total += int(sim.shape[0])
            pos_sum += float(sim.diag().sum())
            eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            neg_sum += float(sim.masked_select(~eye).sum())
            neg_count += int((~eye).sum())
    if total == 0:
        return {"sequence_acc": 0.0, "n_records": mine_report.get("n_records", 0),
                "n_pairs": len(pairs), "sampled": mine_report.get("sampled", False),
                "skipped": True, "mining": mine_report}
    return {"sequence_acc": correct / max(1, total),
            "positive_cosine": pos_sum / max(1, total),
            "negative_cosine": neg_sum / max(1, neg_count),
            "margin": (pos_sum / max(1, total)) - (neg_sum / max(1, neg_count)),
            "n_records": mine_report.get("n_records", 0),
            "n_pairs": len(pairs),
            "sampled": mine_report.get("sampled", False),
            "skipped": False,
            "mining": mine_report}


def reading_neighborhood_retrieval_eval(model, vocab, records, device=DEV, n=0,
                                        seed=0, token_drop_p=0.15,
                                        token_replace_p=0.05):
    pairs, mine_report = mine_reading_latent_neighbors(
        model, vocab, records, device=device, n=n, seed=seed, split="eval")
    if not pairs:
        return {"neighbor_acc": 0.0, "n_records": mine_report.get("n_records", 0),
                "n_pairs": 0, "sampled": mine_report.get("sampled", False),
                "skipped": True, "mining": mine_report}
    anchors = [a for a, _b in pairs]
    positives = [b for _a, b in pairs]
    correct = total = 0
    pos_sum = neg_sum = 0.0
    neg_count = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(pairs), 64):
            anchor_batch = anchors[off:off + 64]
            positive_batch = positives[off:off + 64]
            if len(anchor_batch) <= 1:
                continue
            anchor_txt = corrupt_reading_tokens(
                pack_reading(anchor_batch, vocab, device), vocab.pad, vocab.unk,
                token_drop_p=token_drop_p, token_replace_p=token_replace_p)
            positive_txt = corrupt_reading_tokens(
                pack_reading(positive_batch, vocab, device), vocab.pad, vocab.unk,
                token_drop_p=token_drop_p, token_replace_p=token_replace_p)
            anchor_slots = model.latent_concept_states(anchor_txt, project=True)
            positive_slots = model.latent_concept_states(positive_txt, project=True)
            anchor = F.normalize(anchor_slots.reshape(anchor_slots.shape[0], -1),
                                 dim=-1)
            positive = F.normalize(
                positive_slots.reshape(positive_slots.shape[0], -1), dim=-1)
            sim = anchor.matmul(positive.t())
            labels = torch.arange(sim.shape[0], device=sim.device)
            nearest = sim.argmax(-1)
            correct += int(nearest.eq(labels).sum())
            total += int(sim.shape[0])
            pos_sum += float(sim.diag().sum())
            eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            neg_sum += float(sim.masked_select(~eye).sum())
            neg_count += int((~eye).sum())
    if total == 0:
        return {"neighbor_acc": 0.0, "n_records": mine_report.get("n_records", 0),
                "n_pairs": len(pairs), "sampled": mine_report.get("sampled", False),
                "skipped": True, "mining": mine_report}
    return {"neighbor_acc": correct / max(1, total),
            "positive_cosine": pos_sum / max(1, total),
            "negative_cosine": neg_sum / max(1, neg_count),
            "margin": (pos_sum / max(1, total)) - (neg_sum / max(1, neg_count)),
            "n_records": mine_report.get("n_records", 0),
            "n_pairs": len(pairs),
            "sampled": mine_report.get("sampled", False),
            "skipped": False,
            "mining": mine_report}


def reading_cluster_retrieval_eval(model, vocab, records, device=DEV, n=0,
                                   seed=0, token_drop_p=0.15,
                                   token_replace_p=0.05,
                                   min_cluster_size=2):
    clusters, mine_report = mine_reading_latent_clusters(
        model, vocab, records, device=device, n=n, seed=seed, split="eval",
        min_cluster_size=min_cluster_size)
    if len(clusters) < 2:
        return {"cluster_acc": 0.0, "n_records": mine_report.get("n_records", 0),
                "n_clusters": len(clusters),
                "n_clustered_records": mine_report.get("n_clustered_records", 0),
                "sampled": mine_report.get("sampled", False),
                "skipped": True, "mining": mine_report}
    rows = []
    cluster_ids = []
    for cluster_id, cluster in enumerate(clusters):
        for rec in cluster:
            rows.append(rec)
            cluster_ids.append(cluster_id)
    if len(rows) < 2:
        return {"cluster_acc": 0.0, "n_records": mine_report.get("n_records", 0),
                "n_clusters": len(clusters),
                "n_clustered_records": len(rows),
                "sampled": mine_report.get("sampled", False),
                "skipped": True, "mining": mine_report}
    model.eval()
    with torch.no_grad():
        txt = pack_reading(rows, vocab, device)
        torch.manual_seed(int(seed) + 1)
        view_a = corrupt_reading_tokens(
            txt, vocab.pad, vocab.unk, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p)
        torch.manual_seed(int(seed) + 2)
        view_b = corrupt_reading_tokens(
            txt, vocab.pad, vocab.unk, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p)
        slots_a = model.latent_concept_states(view_a, project=True)
        slots_b = model.latent_concept_states(view_b, project=True)
        a = F.normalize(slots_a.reshape(slots_a.shape[0], -1), dim=-1)
        b = F.normalize(slots_b.reshape(slots_b.shape[0], -1), dim=-1)
        labels = torch.tensor(cluster_ids, dtype=torch.long, device=device)
        centers = []
        center_labels = []
        for label in labels.unique(sorted=True):
            mask = labels.eq(label)
            if int(mask.sum()) >= int(min_cluster_size):
                centers.append(F.normalize(a[mask].mean(0), dim=0))
                center_labels.append(int(label.item()))
        if len(centers) < 2:
            return {"cluster_acc": 0.0,
                    "n_records": mine_report.get("n_records", 0),
                    "n_clusters": len(clusters),
                    "n_clustered_records": len(rows),
                    "sampled": mine_report.get("sampled", False),
                    "skipped": True, "mining": mine_report}
        centers = torch.stack(centers, dim=0)
        label_to_center = {label: i for i, label in enumerate(center_labels)}
        keep = torch.tensor([label in label_to_center for label in cluster_ids],
                            dtype=torch.bool, device=device)
        target = torch.tensor([label_to_center[label] for label in cluster_ids
                               if label in label_to_center],
                              dtype=torch.long, device=device)
        sim = b[keep].matmul(centers.t())
        pred = sim.argmax(-1)
        acc = float(pred.eq(target).float().mean())
        target_sim = sim.gather(1, target[:, None]).squeeze(1)
        other_sim = sim.masked_fill(
            F.one_hot(target, num_classes=centers.shape[0]).bool(),
            -float("inf")).max(-1).values
        positive = float(target_sim.mean())
        negative = float(other_sim.mean())
    return {"cluster_acc": acc,
            "positive_cosine": positive,
            "negative_cosine": negative,
            "margin": positive - negative,
            "n_records": mine_report.get("n_records", 0),
            "n_clusters": len(clusters),
            "n_clustered_records": len(rows),
            "sampled": mine_report.get("sampled", False),
            "skipped": False,
            "mining": mine_report}


def reading_fer_eval(model, vocab, records, device=DEV, n=0, seed=0,
                     feature_dropout=0.0):
    if getattr(model, "latent_concepts", None) is None:
        return {"fer_score": 0.0, "fragmentation": 0.0,
                "slot_correlation": 0.0, "slot_imbalance": 0.0,
                "n_records": 0, "sampled": False, "skipped": True}
    if n < 0:
        return {"fer_score": 0.0, "fragmentation": 0.0,
                "slot_correlation": 0.0, "slot_imbalance": 0.0,
                "n_records": 0, "sampled": False, "skipped": True}
    eval_rows = [r for r in records if r.split == "eval"]
    candidates = eval_rows or list(records)
    if not candidates:
        return {"fer_score": 0.0, "fragmentation": 0.0,
                "slot_correlation": 0.0, "slot_imbalance": 0.0,
                "n_records": 0, "sampled": False, "skipped": True}
    sampled = bool(n and n < len(candidates))
    selected = candidates
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        selected = [candidates[int(i)] for i in idx]
    totals = {"fer_score": 0.0, "fragmentation": 0.0,
              "slot_correlation": 0.0, "slot_imbalance": 0.0}
    total = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(selected), 64):
            batch = selected[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            slots = model.latent_concept_states(
                txt, feature_dropout=feature_dropout, project=True)
            metrics = latent_concept_fer_metrics(slots)
            weight = len(batch)
            total += weight
            for key in totals:
                totals[key] += float(metrics[key].detach()) * weight
    if total == 0:
        return {"fer_score": 0.0, "fragmentation": 0.0,
                "slot_correlation": 0.0, "slot_imbalance": 0.0,
                "n_records": 0, "sampled": sampled, "skipped": True}
    report = {key: value / float(total) for key, value in totals.items()}
    report.update({"n_records": total,
                   "sampled": sampled,
                   "source_split": "eval" if eval_rows else "all",
                   "skipped": False})
    return report


READING_SCORE_METRICS = (
    "view", "context", "sequence", "neighborhood", "cluster", "fer", "bridge",
    "both", "min", "all", "balanced", "mastery")
READING_DISCOVERY_SIGNALS = (
    "view", "context", "sequence", "neighborhood", "cluster", "fer", "bridge")
READING_STUDY_STRATEGIES = (
    "random", "errors", "curiosity", "graph", "cycle", "discovery", "auto")
READING_MEMORY_STUDY_STRATEGIES = ("curiosity", "graph", "discovery")
READING_POOL_STUDY_STRATEGIES = (
    "errors", "curiosity", "graph", "cycle", "discovery")
READING_TRANSITION_STUDY_STRATEGIES = ("graph", "cycle", "discovery")
READING_GRAPH_READY_STUDY_STRATEGIES = ("graph", "cycle")


def resolve_reading_study_strategy(study_strategy, model):
    requested = str(study_strategy)
    if requested not in READING_STUDY_STRATEGIES:
        raise ValueError(f"unknown reading study strategy {requested!r}")
    if requested == "auto":
        return ("discovery"
                if getattr(model, "latent_concept_memory", None) is not None
                else "errors")
    return requested


def reading_discovery_score_components(view_eval, context_eval, metric="both",
                                       margin_w=0.1, neighborhood_eval=None,
                                       cluster_eval=None, fer_eval=None,
                                       bridge_eval=None, sequence_eval=None):
    metric = str(metric)
    if metric not in READING_SCORE_METRICS:
        raise ValueError(f"unknown reading score metric {metric!r}")
    margin_w = float(margin_w)
    view_acc = float(view_eval.get("paired_view_acc", 0.0))
    view_margin = float(view_eval.get("margin", 0.0))
    context_acc = float(context_eval.get("context_target_acc", 0.0))
    context_margin = float(context_eval.get("margin", 0.0))
    sequence_eval = sequence_eval or {}
    sequence_acc = float(sequence_eval.get("sequence_acc", 0.0))
    sequence_margin = float(sequence_eval.get("margin", 0.0))
    neighborhood_eval = neighborhood_eval or {}
    neighborhood_acc = float(neighborhood_eval.get("neighbor_acc", 0.0))
    neighborhood_margin = float(neighborhood_eval.get("margin", 0.0))
    cluster_eval = cluster_eval or {}
    cluster_acc = float(cluster_eval.get("cluster_acc", 0.0))
    cluster_margin = float(cluster_eval.get("margin", 0.0))
    fer_eval = fer_eval or {}
    fer_raw_score = max(0.0, float(fer_eval.get("fer_score", 0.0)))
    fer_score = (0.0 if bool(fer_eval.get("skipped", False))
                 else 1.0 / (1.0 + fer_raw_score))
    bridge_eval = {"skipped": True} if bridge_eval is None else bridge_eval
    bridge_raw_score = max(0.0, float(bridge_eval.get("mean_bridge_score", 0.0)))
    bridge_resolution = 1.0 / (1.0 + bridge_raw_score)
    bridge_connectivity = min(1.0, max(0.0, float(
        bridge_eval.get("mean_bridge_connectivity", 1.0))))
    bridge_score = (0.0 if bool(bridge_eval.get("skipped", False))
                    else 0.5 * (bridge_resolution + bridge_connectivity))
    view_score = view_acc + margin_w * view_margin
    context_score = context_acc + margin_w * context_margin
    sequence_score = sequence_acc + margin_w * sequence_margin
    neighborhood_score = neighborhood_acc + margin_w * neighborhood_margin
    cluster_score = cluster_acc + margin_w * cluster_margin
    scores = {"view": view_score, "context": context_score,
              "sequence": sequence_score,
              "neighborhood": neighborhood_score, "cluster": cluster_score,
              "fer": fer_score, "bridge": bridge_score}
    skipped = {"view": bool(view_eval.get("skipped", False)),
               "context": bool(context_eval.get("skipped", False)),
               "sequence": bool(sequence_eval.get("skipped", False)),
               "neighborhood": bool(neighborhood_eval.get("skipped", False)),
               "cluster": bool(cluster_eval.get("skipped", False)),
               "fer": bool(fer_eval.get("skipped", False)),
               "bridge": bool(bridge_eval.get("skipped", False))}
    active_scores = [scores[name] for name in READING_DISCOVERY_SIGNALS
                     if not skipped[name]]
    if not active_scores:
        active_scores = [0.0]
    all_score = sum(scores.values()) / float(len(scores))
    active_mean_score = sum(active_scores) / float(len(active_scores))
    floor_score = min(active_scores)
    balanced_score = 0.5 * (active_mean_score + floor_score)
    signal_coverage = sum(
        0 if skipped[name] else 1 for name in READING_DISCOVERY_SIGNALS
    ) / float(len(READING_DISCOVERY_SIGNALS))
    mastery_score = 0.5 * active_mean_score + 0.25 * all_score + 0.25 * signal_coverage
    if metric == "view":
        score = view_score
    elif metric == "context":
        score = context_score
    elif metric == "sequence":
        score = sequence_score
    elif metric == "neighborhood":
        score = neighborhood_score
    elif metric == "cluster":
        score = cluster_score
    elif metric == "fer":
        score = fer_score
    elif metric == "bridge":
        score = bridge_score
    elif metric == "min":
        score = min(view_score, context_score)
    elif metric == "all":
        score = all_score
    elif metric == "balanced":
        score = balanced_score
    elif metric == "mastery":
        score = mastery_score
    else:
        score = 0.5 * (view_score + context_score)
    return {"metric": metric,
            "margin_w": margin_w,
            "score": float(score),
            "all_score": float(all_score),
            "active_mean_score": float(active_mean_score),
            "floor_score": float(floor_score),
            "balanced_score": float(balanced_score),
            "mastery_score": float(mastery_score),
            "signal_coverage": float(signal_coverage),
            "view_score": float(view_score),
            "context_score": float(context_score),
            "neighborhood_score": float(neighborhood_score),
            "cluster_score": float(cluster_score),
            "fer_score": float(fer_score),
            "fer_raw_score": float(fer_raw_score),
            "fer_fragmentation": float(fer_eval.get("fragmentation", 0.0)),
            "fer_slot_correlation": float(fer_eval.get("slot_correlation", 0.0)),
            "fer_slot_imbalance": float(fer_eval.get("slot_imbalance", 0.0)),
            "bridge_score": float(bridge_score),
            "bridge_raw_score": float(bridge_raw_score),
            "bridge_resolution": float(bridge_resolution),
            "bridge_connectivity": float(bridge_connectivity),
            "bridge_entropy": float(bridge_eval.get("mean_bridge_entropy", 0.0)),
            "paired_view_acc": view_acc,
            "paired_view_margin": view_margin,
            "context_target_acc": context_acc,
            "context_target_margin": context_margin,
            "sequence_score": float(sequence_score),
            "sequence_acc": sequence_acc,
            "sequence_margin": sequence_margin,
            "neighborhood_acc": neighborhood_acc,
            "neighborhood_margin": neighborhood_margin,
            "cluster_acc": cluster_acc,
            "cluster_margin": cluster_margin,
            "view_skipped": skipped["view"],
            "context_skipped": skipped["context"],
            "sequence_skipped": skipped["sequence"],
            "neighborhood_skipped": skipped["neighborhood"],
            "cluster_skipped": skipped["cluster"],
            "fer_skipped": skipped["fer"],
            "bridge_skipped": skipped["bridge"]}


def reading_eval_bundle(model, vocab, records, device=DEV, eval_n=64, seed=0,
                        token_drop_p=0.15, token_replace_p=0.05,
                        context_keep_p=0.5, score_metric="mastery",
                        score_margin_w=0.1):
    view = reading_latent_retrieval_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 17,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p)
    context = reading_context_target_retrieval_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 23,
        context_keep_p=context_keep_p)
    sequence = reading_sequence_retrieval_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 25,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p)
    neighborhood = reading_neighborhood_retrieval_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 29,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p)
    cluster = reading_cluster_retrieval_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 31,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p)
    fer = reading_fer_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 37,
        feature_dropout=0.0)
    bridge = reading_latent_bridge_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 41,
        feature_dropout=0.0)
    return {"view": view,
            "context_target": context,
            "sequence": sequence,
            "neighborhood": neighborhood,
            "cluster": cluster,
            "fer": fer,
            "bridge": bridge,
            "score_components": reading_discovery_score_components(
                view, context, metric=score_metric, margin_w=score_margin_w,
                neighborhood_eval=neighborhood, cluster_eval=cluster,
                fer_eval=fer, bridge_eval=bridge, sequence_eval=sequence)}


def reading_latent_bridge_graph_state(model):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return None
    return latent_concept_graph_snapshot(memory)


def reading_latent_bridge_eval(model, vocab, records, device=DEV, n=0, seed=0,
                               feature_dropout=0.0, bridge_graph=None):
    graph_source = "snapshot" if bridge_graph is not None else "current"
    if bridge_graph is None:
        bridge_graph = reading_latent_bridge_graph_state(model)
    if getattr(model, "latent_concepts", None) is None or bridge_graph is None:
        return {"n_records": 0, "sampled": False, "mean_bridge_score": 0.0,
                "max_bridge_score": 0.0, "mean_bridge_entropy": 0.0,
                "mean_bridge_connectivity": 1.0, "memory_filled": 0,
                "relation_updates": 0, "transition_updates": 0,
                "graph_ready": False, "graph_source": graph_source,
                "skipped": True,
                "skip_reason": "latent_concept_memory_unavailable"}
    candidates = list(records)
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    if not candidates:
        return {"n_records": 0, "sampled": sampled, "mean_bridge_score": 0.0,
                "max_bridge_score": 0.0, "mean_bridge_entropy": 0.0,
                "mean_bridge_connectivity": 1.0, "memory_filled": 0,
                "relation_updates": 0, "transition_updates": 0,
                "graph_ready": False, "graph_source": graph_source,
                "skipped": True,
                "skip_reason": "no_records"}
    active_memory = bridge_graph["memory"]
    relations = bridge_graph["relations"]
    transitions = bridge_graph["transitions"]
    filled = int(active_memory.shape[0])
    relation_updates = int(bridge_graph.get("relation_updates", 0))
    transition_updates = int(bridge_graph.get("transition_updates", 0))
    graph_ready = latent_concept_graph_ready(graph_state=bridge_graph)
    if not graph_ready:
        return {"n_records": len(candidates), "sampled": sampled,
                "mean_bridge_score": 0.0, "max_bridge_score": 0.0,
                "mean_bridge_entropy": 0.0, "mean_bridge_connectivity": 1.0,
                "memory_filled": filled,
                "relation_updates": relation_updates,
                "transition_updates": transition_updates,
                "graph_ready": False, "graph_source": graph_source,
                "skipped": True,
                "skip_reason": "latent_concept_graph_unavailable"}
    score_values = []
    entropy_values = []
    connectivity_values = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            slots = model.latent_concept_states(
                txt, feature_dropout=feature_dropout, project=True)
            bridge, entropy, connectivity = latent_concept_bridge_scores(
                slots, active_memory, relations, transitions, require_graph=True)
            score_values.extend(float(x) for x in bridge.detach().cpu().tolist())
            entropy_values.extend(float(x) for x in entropy.detach().cpu().tolist())
            connectivity_values.extend(
                float(x) for x in connectivity.detach().cpu().tolist())
    return {"n_records": len(candidates),
            "sampled": sampled,
            "mean_bridge_score": (
                float(np.mean(score_values)) if score_values else 0.0),
            "max_bridge_score": (
                float(max(score_values)) if score_values else 0.0),
            "mean_bridge_entropy": (
                float(np.mean(entropy_values)) if entropy_values else 0.0),
            "mean_bridge_connectivity": (
                float(np.mean(connectivity_values)) if connectivity_values else 1.0),
            "memory_filled": filled,
            "relation_updates": relation_updates,
            "transition_updates": transition_updates,
            "graph_ready": True,
            "graph_source": graph_source,
            "skipped": False}


def reading_latent_record_outcomes(model, vocab, records, device=DEV, n=0, seed=0,
                                   token_drop_p=0.15, token_replace_p=0.05):
    if getattr(model, "latent_concepts", None) is None:
        return [], [], {"n_records": 0, "sampled": False, "n_error_records": 0,
                        "n_correct_records": 0, "paired_view_acc": 0.0,
                        "skipped": True}
    candidates = [r for r in records if r.split == "train"]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    errors = []
    correct = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            a, b = _reading_latent_pair_embeddings(
                model, txt, vocab, seed=seed + off * 2,
                token_drop_p=token_drop_p,
                token_replace_p=token_replace_p)
            nearest = a.matmul(b.t()).argmax(-1)
            for i, rec in enumerate(batch):
                if int(nearest[i]) == i:
                    correct.append(rec)
                else:
                    errors.append(rec)
    total = len(candidates)
    return errors, correct, {"n_records": total,
                             "sampled": sampled,
                             "n_error_records": len(errors),
                             "n_correct_records": len(correct),
                             "paired_view_acc": len(correct) / max(1, total),
                             "skipped": False}


def reading_latent_curiosity_records(
        model, vocab, records, device=DEV, n=0, seed=0, feature_dropout=0.0,
        temperature=0.1, transitive_steps=2, transitive_w=0.1):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return [], {"n_records": 0, "sampled": False, "n_selected": 0,
                    "mean_score": 0.0, "max_score": 0.0,
                    "mean_novelty": 0.0, "mean_association": 0.0,
                    "skipped": True}
    candidates = [r for r in records if r.split == "train"]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    scored = []
    score_values = []
    novelty_values = []
    association_values = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            slots = model.latent_concept_states(
                txt, feature_dropout=feature_dropout, project=True)
            scores, parts = latent_concept_graph_curiosity_scores(
                slots, memory.active(), memory.active_relations(),
                temperature=temperature, transitive_steps=transitive_steps,
                transitive_w=transitive_w)
            novelty = parts.get("novelty", scores.new_zeros(scores.shape))
            association = parts.get("association", scores.new_zeros(scores.shape))
            for i, rec in enumerate(batch):
                score = float(scores[i].detach().cpu())
                scored.append((score, rec))
                score_values.append(score)
                novelty_values.append(float(novelty[i].detach().cpu()))
                association_values.append(float(association[i].detach().cpu()))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [rec for _score, rec in scored]
    total = len(candidates)
    return selected, {"n_records": total,
                      "sampled": sampled,
                      "n_selected": len(selected),
                      "mean_score": float(np.mean(score_values)) if score_values else 0.0,
                      "max_score": float(max(score_values)) if score_values else 0.0,
                      "mean_novelty": (
                          float(np.mean(novelty_values)) if novelty_values else 0.0),
                      "mean_association": (
                          float(np.mean(association_values))
                          if association_values else 0.0),
                      "skipped": False}


def reading_latent_graph_prediction_records(
        model, vocab, records, device=DEV, n=0, seed=0, context_keep_p=0.5,
        feature_dropout=0.0, temperature=0.1, self_loop_w=0.05,
        transitive_steps=2, transitive_w=0.1, target_power=1.0):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return [], {"n_records": 0, "sampled": False, "n_selected": 0,
                    "mean_score": 0.0, "max_score": 0.0,
                    "mean_kl": 0.0, "mean_cosine": 0.0,
                    "skipped": True}
    candidates = [r for r in records if r.split == "train"]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    scored = []
    score_values = []
    kl_values = []
    cosine_values = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            context_txt, heldout_txt = split_reading_context_target(
                txt, vocab.pad, context_keep_p=context_keep_p)
            source_slots = model.latent_concept_states(
                context_txt, feature_dropout=feature_dropout, project=True)
            heldout_slots = model.latent_concept_states(
                heldout_txt, feature_dropout=0.0, project=True)
            if hasattr(model, "latent_concept_graph_prediction_scores"):
                scores, parts = model.latent_concept_graph_prediction_scores(
                    source_slots, heldout_slots, temperature=temperature,
                    self_loop_w=self_loop_w, transitive_steps=transitive_steps,
                    transitive_w=transitive_w, target_power=target_power)
            else:
                scores, parts = latent_concept_graph_prediction_scores(
                    source_slots, heldout_slots, memory.active(),
                    memory.active_prediction_relations(), temperature=temperature,
                    self_loop_w=self_loop_w,
                    transitive_steps=transitive_steps,
                    transitive_w=transitive_w, target_power=target_power)
            kl = parts.get("kl", scores.new_zeros(scores.shape))
            cosine = parts.get("cosine", scores.new_zeros(scores.shape))
            for i, rec in enumerate(batch):
                score = float(scores[i].detach().cpu())
                scored.append((score, rec))
                score_values.append(score)
                kl_values.append(float(kl[i].detach().cpu()))
                cosine_values.append(float(cosine[i].detach().cpu()))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [rec for _score, rec in scored]
    return selected, {"n_records": len(candidates),
                      "sampled": sampled,
                      "n_selected": len(selected),
                      "mean_score": float(np.mean(score_values)) if score_values else 0.0,
                      "max_score": float(max(score_values)) if score_values else 0.0,
                      "mean_kl": float(np.mean(kl_values)) if kl_values else 0.0,
                      "mean_cosine": (
                          float(np.mean(cosine_values)) if cosine_values else 0.0),
                      "skipped": False}


def reading_latent_graph_cycle_records(
        model, vocab, records, device=DEV, n=0, seed=0, context_keep_p=0.5,
        feature_dropout=0.0, temperature=0.1, self_loop_w=0.05,
        transitive_steps=2, transitive_w=0.1, target_power=1.0,
        cycle_w=0.5):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return [], {"n_records": 0, "sampled": False, "n_selected": 0,
                    "mean_score": 0.0, "max_score": 0.0,
                    "mean_forward_kl": 0.0, "mean_reverse_kl": 0.0,
                    "mean_source_cycle_kl": 0.0, "mean_target_cycle_kl": 0.0,
                    "skipped": True}
    candidates = [r for r in records if r.split == "train"]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    scored = []
    score_values = []
    forward_values = []
    reverse_values = []
    source_cycle_values = []
    target_cycle_values = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            context_txt, heldout_txt = split_reading_context_target(
                txt, vocab.pad, context_keep_p=context_keep_p)
            source_slots = model.latent_concept_states(
                context_txt, feature_dropout=feature_dropout, project=True)
            heldout_slots = model.latent_concept_states(
                heldout_txt, feature_dropout=0.0, project=True)
            if hasattr(model, "latent_concept_graph_cycle_scores"):
                scores, parts = model.latent_concept_graph_cycle_scores(
                    source_slots, heldout_slots, temperature=temperature,
                    self_loop_w=self_loop_w, transitive_steps=transitive_steps,
                    transitive_w=transitive_w, target_power=target_power,
                    cycle_w=cycle_w)
            else:
                scores, parts = latent_concept_graph_cycle_scores(
                    source_slots, heldout_slots, memory.active(),
                    memory.active_prediction_relations(), temperature=temperature,
                    self_loop_w=self_loop_w,
                    transitive_steps=transitive_steps,
                    transitive_w=transitive_w, target_power=target_power,
                    cycle_w=cycle_w)
            forward = parts.get("forward_kl", scores.new_zeros(scores.shape))
            reverse = parts.get("reverse_kl", scores.new_zeros(scores.shape))
            source_cycle = parts.get("source_cycle_kl", scores.new_zeros(scores.shape))
            target_cycle = parts.get("target_cycle_kl", scores.new_zeros(scores.shape))
            for i, rec in enumerate(batch):
                score = float(scores[i].detach().cpu())
                scored.append((score, rec))
                score_values.append(score)
                forward_values.append(float(forward[i].detach().cpu()))
                reverse_values.append(float(reverse[i].detach().cpu()))
                source_cycle_values.append(float(source_cycle[i].detach().cpu()))
                target_cycle_values.append(float(target_cycle[i].detach().cpu()))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [rec for _score, rec in scored]
    return selected, {"n_records": len(candidates),
                      "sampled": sampled,
                      "n_selected": len(selected),
                      "mean_score": float(np.mean(score_values)) if score_values else 0.0,
                      "max_score": float(max(score_values)) if score_values else 0.0,
                      "mean_forward_kl": (
                          float(np.mean(forward_values)) if forward_values else 0.0),
                      "mean_reverse_kl": (
                          float(np.mean(reverse_values)) if reverse_values else 0.0),
                      "mean_source_cycle_kl": (
                          float(np.mean(source_cycle_values))
                          if source_cycle_values else 0.0),
                      "mean_target_cycle_kl": (
                          float(np.mean(target_cycle_values))
                          if target_cycle_values else 0.0),
                      "skipped": False}


def _minmax_scale(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _latent_slot_disorder_scores(slots):
    if slots is None or slots.ndim != 3:
        return torch.zeros(0)
    if slots.shape[1] <= 1:
        return slots.reshape(slots.shape[0], -1).sum(-1) * 0.0
    z = F.normalize(slots, dim=-1)
    corr = z.matmul(z.transpose(1, 2))
    eye = torch.eye(slots.shape[1], dtype=torch.bool, device=slots.device)
    correlation = corr.masked_select(~eye[None]).view(
        slots.shape[0], slots.shape[1], slots.shape[1] - 1).pow(2).mean((1, 2))
    energy = slots.pow(2).mean(-1)
    usage = energy / energy.sum(-1, keepdim=True).clamp_min(1e-8)
    uniform = torch.full_like(usage, 1.0 / usage.shape[-1])
    imbalance = F.kl_div(usage.clamp_min(1e-8).log(), uniform, reduction="none").sum(-1)
    return 0.5 * (correlation + imbalance)


def reading_latent_discovery_records(
        model, vocab, records, device=DEV, n=0, seed=0, context_keep_p=0.5,
        feature_dropout=0.0, curiosity_temperature=0.1,
        curiosity_self_loop_w=0.05, curiosity_transitive_steps=2,
        curiosity_transitive_w=0.1, graph_temperature=0.1,
        graph_self_loop_w=0.05, graph_transitive_steps=2,
        graph_transitive_w=0.1, graph_target_power=1.0,
        cycle_temperature=0.1, cycle_self_loop_w=0.05,
        cycle_transitive_steps=2, cycle_transitive_w=0.1,
        cycle_target_power=1.0, cycle_w=0.5):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return [], {"n_records": 0, "sampled": False, "n_selected": 0,
                    "mean_score": 0.0, "max_score": 0.0, "skipped": True}
    candidates = [r for r in records if r.split == "train"]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    rows = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            full_slots = model.latent_concept_states(
                txt, feature_dropout=feature_dropout, project=True)
            curiosity, curiosity_parts = latent_concept_graph_curiosity_scores(
                full_slots, memory.active(), memory.active_relations(),
                temperature=curiosity_temperature,
                self_loop_w=curiosity_self_loop_w,
                transitive_steps=curiosity_transitive_steps,
                transitive_w=curiosity_transitive_w)
            context_txt, heldout_txt = split_reading_context_target(
                txt, vocab.pad, context_keep_p=context_keep_p)
            source_slots = model.latent_concept_states(
                context_txt, feature_dropout=feature_dropout, project=True)
            heldout_slots = model.latent_concept_states(
                heldout_txt, feature_dropout=0.0, project=True)
            if hasattr(model, "latent_concept_graph_prediction_scores"):
                graph, graph_parts = model.latent_concept_graph_prediction_scores(
                    source_slots, heldout_slots, temperature=graph_temperature,
                    self_loop_w=graph_self_loop_w,
                    transitive_steps=graph_transitive_steps,
                    transitive_w=graph_transitive_w,
                    target_power=graph_target_power)
            else:
                graph, graph_parts = latent_concept_graph_prediction_scores(
                    source_slots, heldout_slots, memory.active(),
                    memory.active_prediction_relations(),
                    temperature=graph_temperature, self_loop_w=graph_self_loop_w,
                    transitive_steps=graph_transitive_steps,
                    transitive_w=graph_transitive_w,
                    target_power=graph_target_power)
            if hasattr(model, "latent_concept_graph_cycle_scores"):
                cycle, cycle_parts = model.latent_concept_graph_cycle_scores(
                    source_slots, heldout_slots, temperature=cycle_temperature,
                    self_loop_w=cycle_self_loop_w,
                    transitive_steps=cycle_transitive_steps,
                    transitive_w=cycle_transitive_w,
                    target_power=cycle_target_power,
                    cycle_w=cycle_w)
            else:
                cycle, cycle_parts = latent_concept_graph_cycle_scores(
                    source_slots, heldout_slots, memory.active(),
                    memory.active_prediction_relations(),
                    temperature=cycle_temperature, self_loop_w=cycle_self_loop_w,
                    transitive_steps=cycle_transitive_steps,
                    transitive_w=cycle_transitive_w,
                    target_power=cycle_target_power,
                    cycle_w=cycle_w)
            disorder = _latent_slot_disorder_scores(full_slots)
            bridge, bridge_entropy, bridge_connectivity = latent_concept_bridge_scores(
                full_slots, memory.active(), memory.active_relations(),
                memory.active_transitions())
            novelty = curiosity_parts.get("novelty", curiosity.new_zeros(curiosity.shape))
            association = curiosity_parts.get(
                "association", curiosity.new_zeros(curiosity.shape))
            graph_kl = graph_parts.get("kl", graph.new_zeros(graph.shape))
            graph_cosine = graph_parts.get("cosine", graph.new_zeros(graph.shape))
            forward = cycle_parts.get("forward_kl", cycle.new_zeros(cycle.shape))
            reverse = cycle_parts.get("reverse_kl", cycle.new_zeros(cycle.shape))
            source_cycle = cycle_parts.get(
                "source_cycle_kl", cycle.new_zeros(cycle.shape))
            target_cycle = cycle_parts.get(
                "target_cycle_kl", cycle.new_zeros(cycle.shape))
            for i, rec in enumerate(batch):
                rows.append({
                    "record": rec,
                    "curiosity": float(curiosity[i].detach().cpu()),
                    "novelty": float(novelty[i].detach().cpu()),
                    "association": float(association[i].detach().cpu()),
                    "graph": float(graph[i].detach().cpu()),
                    "graph_kl": float(graph_kl[i].detach().cpu()),
                    "graph_cosine": float(graph_cosine[i].detach().cpu()),
                    "cycle": float(cycle[i].detach().cpu()),
                    "forward_kl": float(forward[i].detach().cpu()),
                    "reverse_kl": float(reverse[i].detach().cpu()),
                    "source_cycle_kl": float(source_cycle[i].detach().cpu()),
                    "target_cycle_kl": float(target_cycle[i].detach().cpu()),
                    "slot_disorder": float(disorder[i].detach().cpu()),
                    "bridge": float(bridge[i].detach().cpu()),
                    "bridge_entropy": float(bridge_entropy[i].detach().cpu()),
                    "bridge_connectivity": float(
                        bridge_connectivity[i].detach().cpu()),
                })
    components = ("curiosity", "graph", "cycle", "slot_disorder", "bridge")
    for name in components:
        scaled = _minmax_scale([row[name] for row in rows])
        for row, value in zip(rows, scaled):
            row[f"{name}_scaled"] = float(value)
    for row in rows:
        row["score"] = float(np.mean([row[f"{name}_scaled"] for name in components]))
    rows.sort(key=lambda row: row["score"], reverse=True)
    selected = [row["record"] for row in rows]

    def mean_field(name):
        return float(np.mean([row[name] for row in rows])) if rows else 0.0

    def max_field(name):
        return float(max([row[name] for row in rows])) if rows else 0.0

    return selected, {"n_records": len(candidates),
                      "sampled": sampled,
                      "n_selected": len(selected),
                      "mean_score": mean_field("score"),
                      "max_score": max_field("score"),
                      "mean_curiosity": mean_field("curiosity"),
                      "mean_novelty": mean_field("novelty"),
                      "mean_association": mean_field("association"),
                      "mean_graph_score": mean_field("graph"),
                      "mean_graph_kl": mean_field("graph_kl"),
                      "mean_graph_cosine": mean_field("graph_cosine"),
                      "mean_cycle_score": mean_field("cycle"),
                      "mean_forward_kl": mean_field("forward_kl"),
                      "mean_reverse_kl": mean_field("reverse_kl"),
                      "mean_source_cycle_kl": mean_field("source_cycle_kl"),
                      "mean_target_cycle_kl": mean_field("target_cycle_kl"),
                      "mean_slot_disorder": mean_field("slot_disorder"),
                      "mean_bridge_score": mean_field("bridge"),
                      "max_bridge_score": max_field("bridge"),
                      "mean_bridge_entropy": mean_field("bridge_entropy"),
                      "mean_bridge_connectivity": mean_field("bridge_connectivity"),
                      "skipped": False}


def latent_fact_concept_loss(model, txt, records, schema):
    if (schema is None or getattr(model, "fact_concepts", None) is None
            or getattr(model, "latent_concepts", None) is None):
        return torch.tensor(0.0, device=txt.device)
    logits = model.latent_fact_concept_logits(txt)
    targets_by_key = fact_concept_target_ids(records, schema, txt.device)
    losses = []
    for key in schema.keys:
        row = logits.get(key)
        targets = targets_by_key[key]
        if row is not None and targets.ge(0).any() and row.shape[-1] > 1:
            losses.append(F.cross_entropy(row, targets, ignore_index=-1))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def _normalized_mean(vectors):
    return F.normalize(torch.stack(vectors).mean(0), dim=0)


def fit_model(model, vocab, records, steps=400, batch=32, lr=1e-3, seed=0,
              device=DEV, log_every=100, semantic_w=0.5, balance_by="none",
              fact_concept_w=0.0, fact_concept_contrast_w=0.0,
              fact_concept_contrast_temperature=0.1,
              fact_concept_centroid_w=0.0,
              fact_concept_centroid_temperature=0.1,
              fact_concept_centroid_margin=0.0,
              fact_concept_prototype_w=0.0,
              fact_concept_prototype_spread_w=0.0,
              fact_concept_prototype_spread_margin=0.2,
              fact_concept_state_spread_w=0.0,
              fact_concept_state_spread_variance=0.05,
              fact_concept_state_spread_margin=0.2,
              fact_concept_state_spread_covariance_w=0.05,
              latent_concept_w=0.0,
              latent_concept_view_dropout=0.1,
              latent_concept_invariance_w=25.0,
              latent_concept_variance_w=25.0,
              latent_concept_covariance_w=1.0,
              latent_concept_variance_target=1.0,
              latent_concept_fact_w=0.0,
              prefix="text", decode_w=1.0):
    if ((latent_concept_w > 0.0 or latent_concept_fact_w > 0.0)
            and getattr(model, "latent_concepts", None) is None):
        raise ValueError("latent concept losses require latent concept slots")
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_records = [r for r in records if r.split == "train"]
    if not train_records:
        raise ValueError("cannot train without train records")
    train_buckets = bucket_records(train_records, balance_by)
    for st in range(1, steps + 1):
        model.train()
        if balance_by == "none":
            rec_batch = batch_records(train_records, rng, batch)
        else:
            rec_batch = balanced_batch_records(train_buckets, rng, batch)
        txt, ids = pack(rec_batch, vocab, device)
        dec_loss = token_loss(model(txt, ids), ids, pad=vocab.pad)
        sem_loss = semantic_loss(model, txt, rec_batch, model.fact_schema)
        concept_fact_loss = (fact_concept_loss(model, txt, rec_batch, model.fact_schema)
                             if fact_concept_w else torch.tensor(0.0, device=device))
        concept_contrast_loss = (
            fact_concept_contrastive_loss(
                model, txt, rec_batch, model.fact_schema,
                temperature=fact_concept_contrast_temperature)
            if fact_concept_contrast_w else torch.tensor(0.0, device=device))
        concept_centroid_loss = (
            fact_concept_batch_centroid_loss(
                model, txt, rec_batch, model.fact_schema,
                temperature=fact_concept_centroid_temperature,
                margin=fact_concept_centroid_margin)
            if fact_concept_centroid_w else torch.tensor(0.0, device=device))
        concept_proto_loss = (
            fact_concept_prototype_loss(
                model, txt, rec_batch, model.fact_schema,
                temperature=fact_concept_contrast_temperature)
            if fact_concept_prototype_w else torch.tensor(0.0, device=device))
        concept_proto_spread_loss = (
            fact_concept_prototype_spread_loss(
                model, margin=fact_concept_prototype_spread_margin)
            if fact_concept_prototype_spread_w else torch.tensor(0.0, device=device))
        concept_state_spread_loss = (
            fact_concept_state_spread_loss(
                model, txt, rec_batch, model.fact_schema,
                variance_target=fact_concept_state_spread_variance,
                centroid_margin=fact_concept_state_spread_margin,
                covariance_weight=fact_concept_state_spread_covariance_w)
            if fact_concept_state_spread_w else torch.tensor(0.0, device=device))
        latent_loss = (
            latent_text_concept_loss(
                model, txt, view_dropout=latent_concept_view_dropout,
                invariance_w=latent_concept_invariance_w,
                variance_w=latent_concept_variance_w,
                covariance_w=latent_concept_covariance_w,
                variance_target=latent_concept_variance_target)
            if latent_concept_w else torch.tensor(0.0, device=device))
        latent_fact_loss = (
            latent_fact_concept_loss(model, txt, rec_batch, model.fact_schema)
            if latent_concept_fact_w else torch.tensor(0.0, device=device))
        loss = (decode_w * dec_loss + semantic_w * sem_loss
                + fact_concept_w * concept_fact_loss
                + fact_concept_contrast_w * concept_contrast_loss
                + fact_concept_centroid_w * concept_centroid_loss
                + fact_concept_prototype_w * concept_proto_loss
                + fact_concept_prototype_spread_w * concept_proto_spread_loss
                + fact_concept_state_spread_w * concept_state_spread_loss
                + latent_concept_w * latent_loss
                + latent_concept_fact_w * latent_fact_loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % log_every == 0 or st == steps:
            print(f"  {prefix} {st}/{steps} loss {loss.item():.3f} "
                  f"dec {dec_loss.item():.3f} sem {sem_loss.item():.3f} "
                  f"fact-concept {concept_fact_loss.item():.3f} "
                  f"fact-contrast {concept_contrast_loss.item():.3f} "
                  f"fact-centroid {concept_centroid_loss.item():.3f} "
                  f"fact-proto {concept_proto_loss.item():.3f} "
                  f"fact-proto-spread {concept_proto_spread_loss.item():.3f} "
                  f"fact-state-spread {concept_state_spread_loss.item():.3f} "
                  f"latent {latent_loss.item():.3f} "
                  f"latent-fact {latent_fact_loss.item():.3f}", flush=True)
    return model, vocab


def train_model(records, steps=400, batch=32, d=96, layers=3, heads=4,
                text_encoder_arch="transformer", text_encoder_layers=1,
                fact_concept_refine=False, fact_concept_refine_gate_init=-2.0,
                fact_concept_mixer_layers=0, fact_concept_mixer_gate_init=-2.0,
                latent_concept_slots=0, latent_concept_layers=1,
                latent_concept_prefix=False,
                latent_concept_refine=False,
                latent_concept_refine_gate_init=-2.0,
                lr=1e-3, seed=0, device=DEV, log_every=100,
                semantic_w=0.5, balance_by="none",
                fact_concept_w=0.0, fact_concept_contrast_w=0.0,
                fact_concept_contrast_temperature=0.1,
                fact_concept_centroid_w=0.0,
                fact_concept_centroid_temperature=0.1,
                fact_concept_centroid_margin=0.0,
                fact_concept_prefix=False,
                fact_concept_prototype_w=0.0,
                fact_concept_prototype_spread_w=0.0,
                fact_concept_prototype_spread_margin=0.2,
                fact_concept_state_spread_w=0.0,
                fact_concept_state_spread_variance=0.05,
                fact_concept_state_spread_margin=0.2,
                fact_concept_state_spread_covariance_w=0.05,
                latent_concept_w=0.0,
                latent_concept_view_dropout=0.1,
                latent_concept_invariance_w=25.0,
                latent_concept_variance_w=25.0,
                latent_concept_covariance_w=1.0,
                latent_concept_variance_target=1.0,
                latent_concept_fact_w=0.0,
                decode_w=1.0):
    torch.manual_seed(seed)
    vocab = build_vocab(records)
    schema = build_fact_schema(records)
    model = TextFactLM(len(vocab), d=d, layers=layers, heads=heads, pad=vocab.pad,
                       fact_schema=schema,
                       fact_concept_prefix=fact_concept_prefix,
                       text_encoder_arch=text_encoder_arch,
                       text_encoder_layers=text_encoder_layers,
                       fact_concept_refine=fact_concept_refine,
                       fact_concept_refine_gate_init=fact_concept_refine_gate_init,
                       fact_concept_mixer_layers=fact_concept_mixer_layers,
                       fact_concept_mixer_gate_init=fact_concept_mixer_gate_init,
                       latent_concept_slots=latent_concept_slots,
                       latent_concept_layers=latent_concept_layers,
                       latent_concept_prefix=latent_concept_prefix,
                       latent_concept_refine=latent_concept_refine,
                       latent_concept_refine_gate_init=latent_concept_refine_gate_init).to(device)
    return fit_model(model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
                     device=device, log_every=log_every, semantic_w=semantic_w,
                     balance_by=balance_by, fact_concept_w=fact_concept_w,
                     fact_concept_contrast_w=fact_concept_contrast_w,
                     fact_concept_contrast_temperature=fact_concept_contrast_temperature,
                     fact_concept_centroid_w=fact_concept_centroid_w,
                     fact_concept_centroid_temperature=fact_concept_centroid_temperature,
                     fact_concept_centroid_margin=fact_concept_centroid_margin,
                     fact_concept_prototype_w=fact_concept_prototype_w,
                     fact_concept_prototype_spread_w=fact_concept_prototype_spread_w,
                     fact_concept_prototype_spread_margin=fact_concept_prototype_spread_margin,
                     fact_concept_state_spread_w=fact_concept_state_spread_w,
                     fact_concept_state_spread_variance=fact_concept_state_spread_variance,
                     fact_concept_state_spread_margin=fact_concept_state_spread_margin,
                     fact_concept_state_spread_covariance_w=fact_concept_state_spread_covariance_w,
                     latent_concept_w=latent_concept_w,
                     latent_concept_view_dropout=latent_concept_view_dropout,
                     latent_concept_invariance_w=latent_concept_invariance_w,
                     latent_concept_variance_w=latent_concept_variance_w,
                     latent_concept_covariance_w=latent_concept_covariance_w,
                     latent_concept_variance_target=latent_concept_variance_target,
                     latent_concept_fact_w=latent_concept_fact_w,
                     decode_w=decode_w)


def fit_reading_concepts(model, vocab, records, steps=400, batch=32, lr=1e-3,
                         seed=0, device=DEV, log_every=100,
                         token_drop_p=0.15, token_replace_p=0.05,
                         feature_dropout=0.1,
                         invariance_w=25.0, variance_w=25.0,
                         covariance_w=1.0, variance_target=1.0,
                         factorization_w=0.05,
                         factorization_variance=0.05,
                         factorization_margin=0.2,
                         factorization_covariance_w=0.05,
                         fer_w=0.0, fer_fragmentation_w=1.0,
                         fer_correlation_w=1.0, fer_balance_w=0.1,
                         memory_w=0.05, memory_size=64,
                         memory_temperature=0.1, memory_momentum=0.95,
                         memory_balance_w=0.01,
                         association_w=0.05, association_temperature=0.1,
                         association_decay=0.99, association_target_power=1.0,
                         association_self_loop_w=0.05,
                         association_transitive_steps=2,
                         association_transitive_w=0.1,
                         composition_w=0.0, composition_temperature=0.1,
                         composition_self_loop_w=0.0,
                         composition_transitive_steps=2,
                         composition_transitive_w=0.1,
                         composition_margin=0.0,
                         graph_predict_w=0.0,
                         graph_predict_temperature=0.1,
                         graph_predict_self_loop_w=0.05,
                         graph_predict_transitive_steps=2,
                         graph_predict_transitive_w=0.1,
                         graph_predict_target_power=1.0,
                         graph_cycle_w=0.0, graph_cycle_temperature=0.1,
                         graph_cycle_self_loop_w=0.05,
                         graph_cycle_transitive_steps=2,
                         graph_cycle_transitive_w=0.1,
                         graph_cycle_target_power=1.0,
                         graph_cycle_consistency_w=0.5,
                         bridge_w=0.0,
                         context_target_w=0.1, context_keep_p=0.5,
                         context_target_temperature=0.1,
                         sequence_w=0.05, sequence_batch=0,
                         sequence_temperature=0.1,
                         neighborhood_w=0.0, neighborhood_batch=0,
                         neighborhood_probe_n=0, neighborhood_refresh_steps=0,
                         neighborhood_temperature=0.1,
                         neighborhood_margin=0.0,
                         transition_w=0.05, transition_batch=0,
                         transition_temperature=0.1,
                         transition_margin=0.0,
                         cluster_w=0.0, cluster_batch=0,
                         cluster_probe_n=0, cluster_refresh_steps=0,
                         cluster_temperature=0.1, cluster_margin=0.0,
                         cluster_min_size=2,
                         study_strategy="auto", study_probe_n=0,
                         study_hard_max=0, study_refresh_steps=0,
                         replay_records=None, replay_teacher_model=None,
                         replay_teacher_vocab=None, replay_w=0.0,
                         replay_batch=0):
    if getattr(model, "latent_concepts", None) is None:
        raise ValueError("raw reading concept training requires latent concept slots")
    if float(context_target_w) < 0.0:
        raise ValueError("reading context-target loss weight must be non-negative")
    if float(factorization_w) < 0.0:
        raise ValueError("reading factorization loss weight must be non-negative")
    if float(factorization_variance) < 0.0:
        raise ValueError("reading factorization variance must be non-negative")
    if float(factorization_margin) < 0.0:
        raise ValueError("reading factorization margin must be non-negative")
    if float(factorization_covariance_w) < 0.0:
        raise ValueError("reading factorization covariance weight must be non-negative")
    if float(fer_w) < 0.0:
        raise ValueError("reading FER loss weight must be non-negative")
    if float(fer_fragmentation_w) < 0.0:
        raise ValueError("reading FER fragmentation weight must be non-negative")
    if float(fer_correlation_w) < 0.0:
        raise ValueError("reading FER correlation weight must be non-negative")
    if float(fer_balance_w) < 0.0:
        raise ValueError("reading FER balance weight must be non-negative")
    if float(memory_w) < 0.0:
        raise ValueError("reading memory loss weight must be non-negative")
    if int(memory_size) < 0:
        raise ValueError("reading memory size must be non-negative")
    if float(memory_temperature) <= 0.0:
        raise ValueError("reading memory temperature must be positive")
    if float(memory_momentum) < 0.0 or float(memory_momentum) >= 1.0:
        raise ValueError("reading memory momentum must be in [0, 1)")
    if float(memory_balance_w) < 0.0:
        raise ValueError("reading memory balance weight must be non-negative")
    if float(association_w) < 0.0:
        raise ValueError("reading association loss weight must be non-negative")
    if float(association_temperature) <= 0.0:
        raise ValueError("reading association temperature must be positive")
    if float(association_decay) < 0.0 or float(association_decay) >= 1.0:
        raise ValueError("reading association decay must be in [0, 1)")
    if float(association_target_power) <= 0.0:
        raise ValueError("reading association target power must be positive")
    if float(association_self_loop_w) < 0.0:
        raise ValueError("reading association self-loop weight must be non-negative")
    if int(association_transitive_steps) < 1:
        raise ValueError("reading association transitive steps must be positive")
    if float(association_transitive_w) < 0.0:
        raise ValueError("reading association transitive weight must be non-negative")
    if float(composition_w) < 0.0:
        raise ValueError("reading composition loss weight must be non-negative")
    if float(composition_temperature) <= 0.0:
        raise ValueError("reading composition temperature must be positive")
    if float(composition_self_loop_w) < 0.0:
        raise ValueError("reading composition self-loop weight must be non-negative")
    if int(composition_transitive_steps) < 1:
        raise ValueError("reading composition transitive steps must be positive")
    if float(composition_transitive_w) < 0.0:
        raise ValueError("reading composition transitive weight must be non-negative")
    if float(composition_margin) < 0.0:
        raise ValueError("reading composition margin must be non-negative")
    if float(graph_predict_w) < 0.0:
        raise ValueError("reading graph prediction weight must be non-negative")
    if float(graph_predict_temperature) <= 0.0:
        raise ValueError("reading graph prediction temperature must be positive")
    if float(graph_predict_self_loop_w) < 0.0:
        raise ValueError("reading graph prediction self-loop weight must be non-negative")
    if int(graph_predict_transitive_steps) < 1:
        raise ValueError("reading graph prediction transitive steps must be positive")
    if float(graph_predict_transitive_w) < 0.0:
        raise ValueError("reading graph prediction transitive weight must be non-negative")
    if float(graph_predict_target_power) <= 0.0:
        raise ValueError("reading graph prediction target power must be positive")
    if float(graph_cycle_w) < 0.0:
        raise ValueError("reading graph cycle weight must be non-negative")
    if float(graph_cycle_temperature) <= 0.0:
        raise ValueError("reading graph cycle temperature must be positive")
    if float(graph_cycle_self_loop_w) < 0.0:
        raise ValueError("reading graph cycle self-loop weight must be non-negative")
    if int(graph_cycle_transitive_steps) < 1:
        raise ValueError("reading graph cycle transitive steps must be positive")
    if float(graph_cycle_transitive_w) < 0.0:
        raise ValueError("reading graph cycle transitive weight must be non-negative")
    if float(graph_cycle_target_power) <= 0.0:
        raise ValueError("reading graph cycle target power must be positive")
    if float(graph_cycle_consistency_w) < 0.0:
        raise ValueError("reading graph cycle consistency weight must be non-negative")
    if float(bridge_w) < 0.0:
        raise ValueError("reading bridge weight must be non-negative")
    if float(sequence_w) < 0.0:
        raise ValueError("reading sequence loss weight must be non-negative")
    if int(sequence_batch) < 0 or int(sequence_batch) == 1:
        raise ValueError("reading sequence batch must be 0 or at least 2")
    if float(sequence_temperature) <= 0.0:
        raise ValueError("reading sequence temperature must be positive")
    if float(neighborhood_w) < 0.0:
        raise ValueError("reading neighborhood loss weight must be non-negative")
    if int(neighborhood_batch) < 0:
        raise ValueError("reading neighborhood batch must be non-negative")
    if int(neighborhood_probe_n) < 0:
        raise ValueError("reading neighborhood probe count must be non-negative")
    if int(neighborhood_refresh_steps) < 0:
        raise ValueError("reading neighborhood refresh steps must be non-negative")
    if float(neighborhood_temperature) <= 0.0:
        raise ValueError("reading neighborhood temperature must be positive")
    if float(neighborhood_margin) < 0.0:
        raise ValueError("reading neighborhood margin must be non-negative")
    if float(transition_w) < 0.0:
        raise ValueError("reading transition loss weight must be non-negative")
    if int(transition_batch) < 0:
        raise ValueError("reading transition batch must be non-negative")
    if float(transition_temperature) <= 0.0:
        raise ValueError("reading transition temperature must be positive")
    if float(transition_margin) < 0.0:
        raise ValueError("reading transition margin must be non-negative")
    if float(cluster_w) < 0.0:
        raise ValueError("reading cluster loss weight must be non-negative")
    if int(cluster_batch) < 0:
        raise ValueError("reading cluster batch must be non-negative")
    if int(cluster_probe_n) < 0:
        raise ValueError("reading cluster probe count must be non-negative")
    if int(cluster_refresh_steps) < 0:
        raise ValueError("reading cluster refresh steps must be non-negative")
    if float(cluster_temperature) <= 0.0:
        raise ValueError("reading cluster temperature must be positive")
    if float(cluster_margin) < 0.0:
        raise ValueError("reading cluster margin must be non-negative")
    if int(cluster_min_size) < 2:
        raise ValueError("reading cluster min size must be at least two")
    if float(replay_w) < 0.0:
        raise ValueError("reading replay loss weight must be non-negative")
    requested_study_strategy = str(study_strategy)
    if requested_study_strategy not in READING_STUDY_STRATEGIES:
        raise ValueError(
            f"unknown reading study strategy {requested_study_strategy!r}")
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_records = [r for r in records if r.split == "train"]
    if not train_records:
        raise ValueError("cannot train reading concepts without train records")
    replay_records = list(replay_records or [])
    replay_sources = [r for r in replay_records if r.split == "train"] or replay_records
    if int(memory_size) > 0:
        model.enable_latent_concept_memory(int(memory_size))
    study_strategy = resolve_reading_study_strategy(
        requested_study_strategy, model)
    if association_w and getattr(model, "latent_concept_memory", None) is None:
        raise ValueError("reading association requires latent concept memory")
    if composition_w and getattr(model, "latent_concept_memory", None) is None:
        raise ValueError("reading composition requires latent concept memory")
    if graph_predict_w and getattr(model, "latent_concept_memory", None) is None:
        raise ValueError("reading graph prediction requires latent concept memory")
    if graph_cycle_w and getattr(model, "latent_concept_memory", None) is None:
        raise ValueError("reading graph cycle requires latent concept memory")
    if bridge_w and getattr(model, "latent_concept_memory", None) is None:
        raise ValueError("reading bridge loss requires latent concept memory")
    if (study_strategy in READING_MEMORY_STUDY_STRATEGIES
            and getattr(model, "latent_concept_memory", None) is None):
        raise ValueError(
            f"reading {study_strategy} study requires latent concept memory")
    if replay_w and (not replay_sources or replay_teacher_model is None
                     or replay_teacher_vocab is None):
        raise ValueError("reading replay loss requires replay records and teacher checkpoint")
    if replay_teacher_model is not None:
        replay_teacher_model.eval()
        for param in replay_teacher_model.parameters():
            param.requires_grad_(False)
    replay_batch = int(replay_batch)
    if replay_batch < 0:
        raise ValueError("reading replay batch must be non-negative")
    replay_batch = replay_batch or max(1, batch // 2)
    neighborhood_batch = int(neighborhood_batch) or max(1, batch // 2)
    sequence_batch = int(sequence_batch) or max(2, batch // 2)
    transition_batch = int(transition_batch) or max(1, batch // 2)
    cluster_batch = int(cluster_batch) or max(2, batch)
    sequence_pairs, sequence_report = mine_reading_sequence_pairs(
        records, split="train")
    neighborhood_pairs = []
    neighborhood_reports = []
    clusters = []
    cluster_reports = []
    study_pool = []
    study_pool_bridge_reference = None
    study_reports = []
    last_loss = 0.0
    last_view_loss = 0.0
    last_factorization = 0.0
    last_fer = 0.0
    last_fer_score = 0.0
    last_fer_fragmentation = 0.0
    last_fer_slot_correlation = 0.0
    last_fer_slot_imbalance = 0.0
    last_memory = 0.0
    last_memory_updates = 0
    last_transition_updates = 0
    last_association = 0.0
    last_composition = 0.0
    last_graph_predict = 0.0
    last_graph_cycle = 0.0
    last_bridge = 0.0
    last_context_target = 0.0
    last_sequence = 0.0
    last_sequence_transition_updates = 0
    last_neighborhood = 0.0
    last_transition = 0.0
    last_cluster = 0.0
    last_replay = 0.0

    def selected_id_sample(selected, limit=16):
        return [rec.rec_id for rec in selected[:int(limit)]]

    def graph_study_ready():
        memory = getattr(model, "latent_concept_memory", None)
        if study_strategy not in READING_GRAPH_READY_STUDY_STRATEGIES:
            return True
        if memory is None:
            return False
        filled = int(getattr(memory, "filled", torch.zeros((), dtype=torch.long)).item())
        relation_updates = int(
            getattr(memory, "relation_updates", torch.zeros((), dtype=torch.long)).item())
        transition_updates = int(
            getattr(memory, "transition_updates", torch.zeros((), dtype=torch.long)).item())
        return filled > 0 and (relation_updates > 0 or transition_updates > 0)

    def refresh_study_pool(step):
        nonlocal study_pool, study_pool_bridge_reference
        probe_n = int(study_probe_n) if int(study_probe_n) > 0 else max(batch * 4, 1)
        if study_strategy == "curiosity":
            selected, report = reading_latent_curiosity_records(
                model, vocab, records, device=device, n=probe_n,
                seed=seed + 1301 + int(step),
                feature_dropout=0.0,
                temperature=association_temperature,
                transitive_steps=association_transitive_steps,
                transitive_w=association_transitive_w)
            report = report | {"strategy": "curiosity"}
        elif study_strategy == "graph":
            selected, report = reading_latent_graph_prediction_records(
                model, vocab, records, device=device, n=probe_n,
                seed=seed + 1301 + int(step),
                context_keep_p=context_keep_p,
                feature_dropout=0.0,
                temperature=graph_predict_temperature,
                self_loop_w=graph_predict_self_loop_w,
                transitive_steps=graph_predict_transitive_steps,
                transitive_w=graph_predict_transitive_w,
                target_power=graph_predict_target_power)
            report = report | {"strategy": "graph"}
        elif study_strategy == "cycle":
            selected, report = reading_latent_graph_cycle_records(
                model, vocab, records, device=device, n=probe_n,
                seed=seed + 1301 + int(step),
                context_keep_p=context_keep_p,
                feature_dropout=0.0,
                temperature=graph_cycle_temperature,
                self_loop_w=graph_cycle_self_loop_w,
                transitive_steps=graph_cycle_transitive_steps,
                transitive_w=graph_cycle_transitive_w,
                target_power=graph_cycle_target_power,
                cycle_w=graph_cycle_consistency_w)
            report = report | {"strategy": "cycle"}
        elif study_strategy == "discovery":
            selected, report = reading_latent_discovery_records(
                model, vocab, records, device=device, n=probe_n,
                seed=seed + 1301 + int(step),
                context_keep_p=context_keep_p,
                feature_dropout=0.0,
                curiosity_temperature=association_temperature,
                curiosity_self_loop_w=association_self_loop_w,
                curiosity_transitive_steps=association_transitive_steps,
                curiosity_transitive_w=association_transitive_w,
                graph_temperature=graph_predict_temperature,
                graph_self_loop_w=graph_predict_self_loop_w,
                graph_transitive_steps=graph_predict_transitive_steps,
                graph_transitive_w=graph_predict_transitive_w,
                graph_target_power=graph_predict_target_power,
                cycle_temperature=graph_cycle_temperature,
                cycle_self_loop_w=graph_cycle_self_loop_w,
                cycle_transitive_steps=graph_cycle_transitive_steps,
                cycle_transitive_w=graph_cycle_transitive_w,
                cycle_target_power=graph_cycle_target_power,
                cycle_w=graph_cycle_consistency_w)
            report = report | {"strategy": "discovery"}
        else:
            hard, _correct, report = reading_latent_record_outcomes(
                model, vocab, records, device=device, n=probe_n,
                seed=seed + 1301 + int(step),
                token_drop_p=token_drop_p,
                token_replace_p=token_replace_p)
            selected = list(hard)
            report = report | {"strategy": "errors"}
        if study_hard_max and len(selected) > int(study_hard_max):
            if study_strategy in READING_POOL_STUDY_STRATEGIES:
                selected = selected[:int(study_hard_max)]
            else:
                cap_rng = np.random.default_rng(seed + 1759 + int(step))
                idx = cap_rng.choice(len(selected), size=int(study_hard_max), replace=False)
                selected = [selected[int(i)] for i in idx]
            report = report | {"capped": True, "n_error_records_used": len(selected)}
        else:
            report = report | {"capped": False, "n_error_records_used": len(selected)}
        if selected:
            study_pool = selected
            study_pool_bridge_reference = reading_latent_bridge_graph_state(model)
        pool_bridge = reading_latent_bridge_eval(
            model, vocab, selected, device=device, n=0, seed=seed + 1871 + int(step),
            feature_dropout=0.0, bridge_graph=study_pool_bridge_reference)
        report = report | {"step": int(step), "pool_active": bool(study_pool),
                           "pool_size": len(study_pool),
                           "hard_record_ids": selected_id_sample(selected),
                           "hard_record_count": len(selected),
                           "pool_bridge_before": pool_bridge}
        study_reports.append(report)

    def refresh_neighborhood_pairs(step):
        nonlocal neighborhood_pairs
        probe_n = (int(neighborhood_probe_n) if int(neighborhood_probe_n) > 0
                   else max(batch * 8, 2))
        pairs, report = mine_reading_latent_neighbors(
            model, vocab, train_records, device=device, n=probe_n,
            seed=seed + 2203 + int(step), split="train")
        if pairs:
            neighborhood_pairs = pairs
        report = report | {"step": int(step), "pool_active": bool(neighborhood_pairs),
                           "pool_size": len(neighborhood_pairs)}
        neighborhood_reports.append(report)

    def refresh_clusters(step):
        nonlocal clusters
        probe_n = (int(cluster_probe_n) if int(cluster_probe_n) > 0
                   else max(batch * 8, int(cluster_min_size)))
        mined, report = mine_reading_latent_clusters(
            model, vocab, train_records, device=device, n=probe_n,
            seed=seed + 2609 + int(step), split="train",
            min_cluster_size=cluster_min_size)
        if mined:
            clusters = mined
        report = report | {"step": int(step), "pool_active": bool(clusters),
                           "pool_size": len(clusters)}
        cluster_reports.append(report)

    for st in range(1, steps + 1):
        model.train()
        refresh_due = (not study_pool or st == 1 or (
            study_refresh_steps and (st - 1) % int(study_refresh_steps) == 0))
        if study_strategy in READING_POOL_STUDY_STRATEGIES and refresh_due:
            if (study_strategy not in READING_GRAPH_READY_STUDY_STRATEGIES
                    or graph_study_ready()):
                refresh_study_pool(st)
                model.train()
        if (neighborhood_w or transition_w) and (st == 1 or (
                neighborhood_refresh_steps
                and (st - 1) % int(neighborhood_refresh_steps) == 0)):
            refresh_neighborhood_pairs(st)
            model.train()
        if cluster_w and (st == 1 or (
                cluster_refresh_steps
                and (st - 1) % int(cluster_refresh_steps) == 0)):
            refresh_clusters(st)
            model.train()
        source = (study_pool if study_strategy in READING_POOL_STUDY_STRATEGIES
                  and study_pool else train_records)
        rec_batch = batch_records(source, rng, batch)
        txt = pack_reading(rec_batch, vocab, device)
        view_loss = reading_latent_view_loss(
            model, txt, vocab.pad, vocab.unk, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, feature_dropout=feature_dropout,
            invariance_w=invariance_w, variance_w=variance_w,
            covariance_w=covariance_w, variance_target=variance_target)
        factorization_loss = (
            reading_latent_factorization_loss(
                model, txt, feature_dropout=feature_dropout,
                variance_target=factorization_variance,
                separation_margin=factorization_margin,
                covariance_w=factorization_covariance_w)
            if factorization_w else view_loss * 0.0)
        if fer_w:
            fer_loss, fer_metrics = reading_latent_fer_loss(
                model, txt, feature_dropout=feature_dropout,
                fragmentation_w=fer_fragmentation_w,
                correlation_w=fer_correlation_w,
                balance_w=fer_balance_w)
        else:
            fer_loss = view_loss * 0.0
            zero_metric = view_loss.detach() * 0.0
            fer_metrics = {"fer_score": zero_metric,
                           "fragmentation": zero_metric,
                           "slot_correlation": zero_metric,
                           "slot_imbalance": zero_metric}
        memory_loss = (
            reading_latent_memory_loss(
                model, txt, feature_dropout=feature_dropout,
                temperature=memory_temperature,
                balance_w=memory_balance_w)
            if memory_w else view_loss * 0.0)
        association_loss = (
            reading_latent_association_loss(
                model, txt, feature_dropout=feature_dropout,
                temperature=association_temperature,
                target_power=association_target_power,
                self_loop_w=association_self_loop_w,
                transitive_steps=association_transitive_steps,
                transitive_w=association_transitive_w)
            if association_w else view_loss * 0.0)
        composition_loss = (
            reading_latent_composition_loss(
                model, txt, feature_dropout=feature_dropout,
                temperature=composition_temperature,
                self_loop_w=composition_self_loop_w,
                transitive_steps=composition_transitive_steps,
                transitive_w=composition_transitive_w,
                margin=composition_margin)
            if composition_w else view_loss * 0.0)
        graph_predict_loss = (
            reading_context_graph_prediction_loss(
                model, txt, vocab.pad, context_keep_p=context_keep_p,
                feature_dropout=feature_dropout,
                temperature=graph_predict_temperature,
                self_loop_w=graph_predict_self_loop_w,
                transitive_steps=graph_predict_transitive_steps,
                transitive_w=graph_predict_transitive_w,
                target_power=graph_predict_target_power)
            if graph_predict_w else view_loss * 0.0)
        graph_cycle_loss = (
            reading_context_graph_cycle_loss(
                model, txt, vocab.pad, context_keep_p=context_keep_p,
                feature_dropout=feature_dropout,
                temperature=graph_cycle_temperature,
                self_loop_w=graph_cycle_self_loop_w,
                transitive_steps=graph_cycle_transitive_steps,
                transitive_w=graph_cycle_transitive_w,
                target_power=graph_cycle_target_power,
                cycle_w=graph_cycle_consistency_w)
            if graph_cycle_w else view_loss * 0.0)
        bridge_loss = (
            reading_latent_bridge_loss(
                model, txt, feature_dropout=feature_dropout)
            if bridge_w else view_loss * 0.0)
        context_target = (
            reading_context_target_loss(
                model, txt, vocab.pad, context_keep_p=context_keep_p,
                feature_dropout=feature_dropout,
                temperature=context_target_temperature)
            if context_target_w else view_loss * 0.0)
        sequence_loss = view_loss * 0.0
        if sequence_w and sequence_pairs:
            sequence_pair_batch = batch_reading_neighbor_pairs(
                sequence_pairs, rng, sequence_batch)
            sequence_loss = reading_sequence_prediction_loss(
                model, sequence_pair_batch, vocab, device=device,
                token_drop_p=token_drop_p,
                token_replace_p=token_replace_p,
                feature_dropout=feature_dropout,
                temperature=sequence_temperature)
        neighborhood_loss = view_loss * 0.0
        if neighborhood_w and neighborhood_pairs:
            pair_batch = batch_reading_neighbor_pairs(
                neighborhood_pairs, rng, neighborhood_batch)
            neighborhood_loss = reading_latent_neighborhood_loss(
                model, pair_batch, vocab, device=device,
                token_drop_p=token_drop_p,
                token_replace_p=token_replace_p,
                feature_dropout=feature_dropout,
                temperature=neighborhood_temperature,
                margin=neighborhood_margin)
        transition_loss = view_loss * 0.0
        if transition_w and neighborhood_pairs:
            transition_pair_batch = batch_reading_neighbor_pairs(
                neighborhood_pairs, rng, transition_batch)
            transition_loss = reading_latent_transition_loss(
                model, transition_pair_batch, vocab, device=device,
                token_drop_p=token_drop_p,
                token_replace_p=token_replace_p,
                feature_dropout=feature_dropout,
                temperature=transition_temperature,
                margin=transition_margin)
        cluster_loss = view_loss * 0.0
        if cluster_w and clusters:
            cluster_records, cluster_ids = batch_reading_cluster_records(
                clusters, rng, cluster_batch)
            cluster_loss = reading_latent_cluster_loss(
                model, cluster_records, cluster_ids, vocab, device=device,
                token_drop_p=token_drop_p,
                token_replace_p=token_replace_p,
                feature_dropout=feature_dropout,
                temperature=cluster_temperature,
                margin=cluster_margin,
                min_cluster_size=cluster_min_size)
        replay_loss = view_loss * 0.0
        if replay_w and replay_sources:
            replay_batch_records = batch_records(replay_sources, rng, replay_batch)
            replay_loss = reading_teacher_latent_consistency_loss(
                model, replay_teacher_model, replay_batch_records, vocab,
                replay_teacher_vocab, device=device,
                feature_dropout=feature_dropout)
        loss = (view_loss + float(factorization_w) * factorization_loss
                + float(fer_w) * fer_loss
                + float(memory_w) * memory_loss
                + float(association_w) * association_loss
                + float(composition_w) * composition_loss
                + float(graph_predict_w) * graph_predict_loss
                + float(graph_cycle_w) * graph_cycle_loss
                + float(bridge_w) * bridge_loss
                + float(context_target_w) * context_target
                + float(sequence_w) * sequence_loss
                + float(neighborhood_w) * neighborhood_loss
                + float(transition_w) * transition_loss
                + float(cluster_w) * cluster_loss
                + float(replay_w) * replay_loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_memory_updates = int(update_reading_latent_memory(
            model, txt, feature_dropout=0.0, momentum=memory_momentum,
            relation_decay=(association_decay
                            if (association_w or composition_w
                                or graph_predict_w or graph_cycle_w
                                or bridge_w)
                            else None)))
        last_transition_updates = 0
        if (graph_predict_w or graph_cycle_w
                or bridge_w
                or study_strategy in READING_TRANSITION_STUDY_STRATEGIES):
            last_transition_updates = int(update_reading_latent_transitions(
                model, txt, vocab.pad, context_keep_p=context_keep_p,
                feature_dropout=0.0, decay=association_decay))
        last_sequence_transition_updates = 0
        if sequence_pairs and (sequence_w or graph_predict_w
                               or graph_cycle_w or bridge_w):
            sequence_pair_batch = batch_reading_neighbor_pairs(
                sequence_pairs, rng, sequence_batch)
            last_sequence_transition_updates = int(
                update_reading_sequence_transitions(
                    model, sequence_pair_batch, vocab, device=device,
                    feature_dropout=0.0, decay=association_decay))
        last_loss = float(loss.detach())
        last_view_loss = float(view_loss.detach())
        last_factorization = float(factorization_loss.detach())
        last_fer = float(fer_loss.detach())
        last_fer_score = float(fer_metrics["fer_score"].detach())
        last_fer_fragmentation = float(fer_metrics["fragmentation"].detach())
        last_fer_slot_correlation = float(fer_metrics["slot_correlation"].detach())
        last_fer_slot_imbalance = float(fer_metrics["slot_imbalance"].detach())
        last_memory = float(memory_loss.detach())
        last_association = float(association_loss.detach())
        last_composition = float(composition_loss.detach())
        last_graph_predict = float(graph_predict_loss.detach())
        last_graph_cycle = float(graph_cycle_loss.detach())
        last_bridge = float(bridge_loss.detach())
        last_context_target = float(context_target.detach())
        last_sequence = float(sequence_loss.detach())
        last_neighborhood = float(neighborhood_loss.detach())
        last_transition = float(transition_loss.detach())
        last_cluster = float(cluster_loss.detach())
        last_replay = float(replay_loss.detach())
        if st % log_every == 0 or st == steps:
            print(f"  reading {st}/{steps} loss {last_loss:.3f} "
                  f"view {last_view_loss:.3f} "
                  f"factor {last_factorization:.3f} "
                  f"fer {last_fer:.3f} "
                  f"memory {last_memory:.3f} "
                  f"assoc {last_association:.3f} "
                  f"compose {last_composition:.3f} "
                  f"graph-predict {last_graph_predict:.3f} "
                  f"graph-cycle {last_graph_cycle:.3f} "
                  f"bridge {last_bridge:.3f} "
                  f"context-target {last_context_target:.3f} "
                  f"sequence {last_sequence:.3f} "
                  f"neighborhood {last_neighborhood:.3f} "
                  f"transition {last_transition:.3f} "
                  f"cluster {last_cluster:.3f} "
                  f"replay {last_replay:.3f}",
                  flush=True)
    study_pool_bridge_before = (
        study_reports[-1].get("pool_bridge_before", {}) if study_reports else {})
    study_pool_bridge_after = (
        reading_latent_bridge_eval(
            model, vocab, study_pool, device=device, n=0, seed=seed + 2909,
            feature_dropout=0.0, bridge_graph=study_pool_bridge_reference)
        if study_pool else {"n_records": 0, "sampled": False,
                            "mean_bridge_score": 0.0,
                            "max_bridge_score": 0.0,
                            "mean_bridge_entropy": 0.0,
                            "mean_bridge_connectivity": 1.0,
                            "skipped": True})
    study_pool_bridge_after_current = (
        reading_latent_bridge_eval(
            model, vocab, study_pool, device=device, n=0, seed=seed + 2917,
            feature_dropout=0.0)
        if study_pool else {"n_records": 0, "sampled": False,
                            "mean_bridge_score": 0.0,
                            "max_bridge_score": 0.0,
                            "mean_bridge_entropy": 0.0,
                            "mean_bridge_connectivity": 1.0,
                            "skipped": True})
    insight_valid = (
        not bool(study_pool_bridge_before.get("skipped", True))
        and not bool(study_pool_bridge_after.get("skipped", True)))
    bridge_score_reduction = (
        float(study_pool_bridge_before.get("mean_bridge_score", 0.0))
        - float(study_pool_bridge_after.get("mean_bridge_score", 0.0))
        if insight_valid else 0.0)
    bridge_connectivity_gain = (
        float(study_pool_bridge_after.get("mean_bridge_connectivity", 1.0))
        - float(study_pool_bridge_before.get("mean_bridge_connectivity", 1.0))
        if insight_valid else 0.0)
    study_pool_insight = {
        "n_records": int(study_pool_bridge_after.get("n_records", 0)),
        "skipped": not insight_valid,
        "bridge_score_before": float(
            study_pool_bridge_before.get("mean_bridge_score", 0.0)),
        "bridge_score_after": float(
            study_pool_bridge_after.get("mean_bridge_score", 0.0)),
        "bridge_score_reduction": float(bridge_score_reduction),
        "bridge_connectivity_before": float(
            study_pool_bridge_before.get("mean_bridge_connectivity", 1.0)),
        "bridge_connectivity_after": float(
            study_pool_bridge_after.get("mean_bridge_connectivity", 1.0)),
        "bridge_connectivity_gain": float(bridge_connectivity_gain),
        "record_ids": selected_id_sample(study_pool),
    }
    model.reading_train_metrics = {
        "loss": last_loss,
        "study_strategy_requested": requested_study_strategy,
        "study_strategy": study_strategy,
        "latent_view_loss": last_view_loss,
        "factorization_loss": last_factorization,
        "factorization_w": float(factorization_w),
        "factorization_variance": float(factorization_variance),
        "factorization_margin": float(factorization_margin),
        "factorization_covariance_w": float(factorization_covariance_w),
        "fer_loss": last_fer,
        "fer_w": float(fer_w),
        "fer_fragmentation_w": float(fer_fragmentation_w),
        "fer_correlation_w": float(fer_correlation_w),
        "fer_balance_w": float(fer_balance_w),
        "fer_score": last_fer_score,
        "fer_fragmentation": last_fer_fragmentation,
        "fer_slot_correlation": last_fer_slot_correlation,
        "fer_slot_imbalance": last_fer_slot_imbalance,
        "memory_loss": last_memory,
        "memory_w": float(memory_w),
        "memory_size": int(getattr(model, "latent_concept_memory_size", 0)),
        "memory_active": int(
            getattr(getattr(model, "latent_concept_memory", None), "filled",
                    torch.zeros((), dtype=torch.long)).item()),
        "memory_updates": int(
            getattr(getattr(model, "latent_concept_memory", None), "updates",
                    torch.zeros((), dtype=torch.long)).item()),
        "memory_last_batch_updates": int(last_memory_updates),
        "graph_transition_last_batch_updates": int(last_transition_updates),
        "memory_temperature": float(memory_temperature),
        "memory_momentum": float(memory_momentum),
        "memory_balance_w": float(memory_balance_w),
        "association_loss": last_association,
        "association_w": float(association_w),
        "association_temperature": float(association_temperature),
        "association_decay": float(association_decay),
        "association_target_power": float(association_target_power),
        "association_self_loop_w": float(association_self_loop_w),
        "association_transitive_steps": int(association_transitive_steps),
        "association_transitive_w": float(association_transitive_w),
        "association_relation_updates": int(
            getattr(getattr(model, "latent_concept_memory", None),
                    "relation_updates", torch.zeros((), dtype=torch.long)).item()),
        "association_active_edges": int(
            getattr(getattr(model, "latent_concept_memory", None),
                    "relations", torch.zeros(0)).gt(0).sum().item()),
        "graph_transition_updates": int(
            getattr(getattr(model, "latent_concept_memory", None),
                    "transition_updates", torch.zeros((), dtype=torch.long)).item()),
        "graph_transition_active_edges": int(
            getattr(getattr(model, "latent_concept_memory", None),
                    "transitions", torch.zeros(0)).gt(0).sum().item()),
        "composition_loss": last_composition,
        "composition_w": float(composition_w),
        "composition_temperature": float(composition_temperature),
        "composition_self_loop_w": float(composition_self_loop_w),
        "composition_transitive_steps": int(composition_transitive_steps),
        "composition_transitive_w": float(composition_transitive_w),
        "composition_margin": float(composition_margin),
        "graph_predict_loss": last_graph_predict,
        "graph_predict_w": float(graph_predict_w),
        "graph_predict_temperature": float(graph_predict_temperature),
        "graph_predict_self_loop_w": float(graph_predict_self_loop_w),
        "graph_predict_transitive_steps": int(graph_predict_transitive_steps),
        "graph_predict_transitive_w": float(graph_predict_transitive_w),
        "graph_predict_target_power": float(graph_predict_target_power),
        "graph_cycle_loss": last_graph_cycle,
        "graph_cycle_w": float(graph_cycle_w),
        "graph_cycle_temperature": float(graph_cycle_temperature),
        "graph_cycle_self_loop_w": float(graph_cycle_self_loop_w),
        "graph_cycle_transitive_steps": int(graph_cycle_transitive_steps),
        "graph_cycle_transitive_w": float(graph_cycle_transitive_w),
        "graph_cycle_target_power": float(graph_cycle_target_power),
        "graph_cycle_consistency_w": float(graph_cycle_consistency_w),
        "bridge_loss": last_bridge,
        "bridge_w": float(bridge_w),
        "context_target_loss": last_context_target,
        "sequence_loss": last_sequence,
        "sequence_w": float(sequence_w),
        "sequence_batch": int(sequence_batch),
        "sequence_temperature": float(sequence_temperature),
        "sequence_pairs": len(sequence_pairs),
        "sequence_report": sequence_report,
        "sequence_transition_last_batch_updates": int(
            last_sequence_transition_updates),
        "neighborhood_loss": last_neighborhood,
        "transition_loss": last_transition,
        "transition_w": float(transition_w),
        "transition_batch": int(transition_batch),
        "transition_temperature": float(transition_temperature),
        "transition_margin": float(transition_margin),
        "transition_pairs": len(neighborhood_pairs),
        "cluster_loss": last_cluster,
        "neighborhood_w": float(neighborhood_w),
        "neighborhood_batch": int(neighborhood_batch),
        "neighborhood_probe_n": int(neighborhood_probe_n),
        "neighborhood_refresh_steps": int(neighborhood_refresh_steps),
        "neighborhood_temperature": float(neighborhood_temperature),
        "neighborhood_margin": float(neighborhood_margin),
        "neighborhood_pairs": len(neighborhood_pairs),
        "cluster_w": float(cluster_w),
        "cluster_batch": int(cluster_batch),
        "cluster_probe_n": int(cluster_probe_n),
        "cluster_refresh_steps": int(cluster_refresh_steps),
        "cluster_temperature": float(cluster_temperature),
        "cluster_margin": float(cluster_margin),
        "cluster_min_size": int(cluster_min_size),
        "clusters": len(clusters),
        "cluster_records": sum(len(rows) for rows in clusters),
        "replay_loss": last_replay,
        "replay_w": float(replay_w),
        "replay_batch": int(replay_batch),
        "replay_records": len(replay_sources),
        "context_target_w": float(context_target_w),
        "context_keep_p": float(context_keep_p),
        "context_target_temperature": float(context_target_temperature),
        "study_pool_bridge_before": study_pool_bridge_before,
        "study_pool_bridge_after": study_pool_bridge_after,
        "study_pool_bridge_after_current": study_pool_bridge_after_current,
        "study_pool_insight": study_pool_insight,
        "study_pool_bridge_score_reduction": float(bridge_score_reduction),
        "study_pool_bridge_connectivity_gain": float(bridge_connectivity_gain),
    }
    model.reading_study_reports = study_reports
    model.reading_neighborhood_reports = neighborhood_reports
    model.reading_cluster_reports = cluster_reports
    return model, vocab


def _model_state_copy(model):
    return {name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()}


def _step_schedule(steps, rounds):
    rounds = max(1, int(rounds))
    steps = max(0, int(steps))
    base = steps // rounds
    rem = steps % rounds
    return [base + (1 if i < rem else 0) for i in range(rounds)
            if base + (1 if i < rem else 0) > 0]


def fit_reading_concepts_select_best(
        model, vocab, records, steps=400, batch=32, lr=1e-3,
        seed=0, device=DEV, log_every=100,
        token_drop_p=0.15, token_replace_p=0.05,
        feature_dropout=0.1,
        invariance_w=25.0, variance_w=25.0,
        covariance_w=1.0, variance_target=1.0,
        factorization_w=0.05, factorization_variance=0.05,
        factorization_margin=0.2, factorization_covariance_w=0.05,
        fer_w=0.0, fer_fragmentation_w=1.0,
        fer_correlation_w=1.0, fer_balance_w=0.1,
        memory_w=0.05, memory_size=64,
        memory_temperature=0.1, memory_momentum=0.95,
        memory_balance_w=0.01,
        association_w=0.05, association_temperature=0.1,
        association_decay=0.99, association_target_power=1.0,
        association_self_loop_w=0.05, association_transitive_steps=2,
        association_transitive_w=0.1,
        composition_w=0.0, composition_temperature=0.1,
        composition_self_loop_w=0.0, composition_transitive_steps=2,
        composition_transitive_w=0.1, composition_margin=0.0,
        graph_predict_w=0.0, graph_predict_temperature=0.1,
        graph_predict_self_loop_w=0.05, graph_predict_transitive_steps=2,
        graph_predict_transitive_w=0.1, graph_predict_target_power=1.0,
        graph_cycle_w=0.0, graph_cycle_temperature=0.1,
        graph_cycle_self_loop_w=0.05, graph_cycle_transitive_steps=2,
        graph_cycle_transitive_w=0.1, graph_cycle_target_power=1.0,
        graph_cycle_consistency_w=0.5,
        bridge_w=0.0,
        context_target_w=0.1, context_keep_p=0.5,
        context_target_temperature=0.1,
        sequence_w=0.05, sequence_batch=0, sequence_temperature=0.1,
        neighborhood_w=0.0, neighborhood_batch=0,
        neighborhood_probe_n=0, neighborhood_refresh_steps=0,
        neighborhood_temperature=0.1, neighborhood_margin=0.0,
        transition_w=0.05, transition_batch=0,
        transition_temperature=0.1, transition_margin=0.0,
        cluster_w=0.0, cluster_batch=0,
        cluster_probe_n=0, cluster_refresh_steps=0,
        cluster_temperature=0.1, cluster_margin=0.0,
        cluster_min_size=2,
        study_strategy="auto", study_probe_n=0,
        study_hard_max=0, study_refresh_steps=0,
        replay_records=None, replay_teacher_model=None,
        replay_teacher_vocab=None, replay_w=0.0, replay_batch=0,
        replay_retention_w=0.0,
        eval_n=64, score_metric="mastery", score_margin_w=0.1,
        score_min_delta=0.0, score_patience=0,
        rounds=1, before_bundle=None):
    schedule = _step_schedule(steps, rounds)
    if not schedule:
        raise ValueError("reading selected training requires at least one step")
    score_min_delta = float(score_min_delta)
    if score_min_delta < 0.0:
        raise ValueError("reading study score min delta must be non-negative")
    score_patience = int(score_patience)
    if score_patience < 0:
        raise ValueError("reading study score patience must be non-negative")
    if int(memory_size) > 0:
        model.enable_latent_concept_memory(int(memory_size))
    initial_study_strategy = resolve_reading_study_strategy(
        study_strategy, model)
    before_bundle = before_bundle or reading_eval_bundle(
        model, vocab, records, device=device, eval_n=eval_n, seed=seed,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p,
        context_keep_p=context_keep_p, score_metric=score_metric,
        score_margin_w=score_margin_w)
    bridge_insight_gate = bool(
        bridge_w and initial_study_strategy in READING_POOL_STUDY_STRATEGIES)
    replay_records = list(replay_records or [])
    before_replay_bundle = None
    if replay_records and replay_retention_w:
        before_replay_bundle = reading_eval_bundle(
            model, vocab, replay_records, device=device, eval_n=eval_n,
            seed=seed + 4093, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, context_keep_p=context_keep_p,
            score_metric=score_metric, score_margin_w=score_margin_w)

    def selection_row(round_id, round_steps, bundle, replay_bundle=None):
        base_score = float(bundle["score_components"]["score"])
        replay_score = None
        replay_drop = 0.0
        penalty = 0.0
        if before_replay_bundle is not None and replay_bundle is not None:
            replay_score = float(replay_bundle["score_components"]["score"])
            replay_before = float(before_replay_bundle["score_components"]["score"])
            replay_drop = max(0.0, replay_before - replay_score)
            penalty = float(replay_retention_w) * replay_drop
        return {
            "round": int(round_id),
            "steps": int(round_steps),
            "selected": False,
            "score": float(base_score - penalty),
            "base_score": base_score,
            "retention_penalty": float(penalty),
            "replay_retention_drop": float(replay_drop),
            "score_components": bundle["score_components"],
            "replay_score": replay_score,
            "replay_score_components": (
                replay_bundle["score_components"] if replay_bundle is not None else None),
        }

    def bridge_insight_delta(insight):
        if (not bridge_insight_gate or not insight
                or bool(insight.get("skipped", True))):
            return 0.0, True
        reduction = float(insight.get("bridge_score_reduction", 0.0))
        connectivity = float(insight.get("bridge_connectivity_gain", 0.0))
        delta = 0.5 * (reduction + connectivity)
        return float(delta), delta >= -1e-9

    best_state = _model_state_copy(model)
    initial_replay_bundle = before_replay_bundle
    initial_row = selection_row(0, 0, before_bundle, initial_replay_bundle)
    best_score = float(initial_row["score"])
    best_round = 0
    best_metrics = {
        "study_strategy_requested": str(study_strategy),
        "study_strategy": initial_study_strategy,
    } | dict(getattr(model, "reading_train_metrics", {}))
    rounds_report = [initial_row]
    all_study_reports = []
    all_neighborhood_reports = []
    all_cluster_reports = []
    no_improve_rounds = 0
    stopped_early = False
    stop_round = 0
    for round_i, round_steps in enumerate(schedule, start=1):
        fit_reading_concepts(
            model, vocab, records, steps=round_steps, batch=batch, lr=lr,
            seed=seed + round_i * 1009, device=device, log_every=log_every,
            token_drop_p=token_drop_p, token_replace_p=token_replace_p,
            feature_dropout=feature_dropout, invariance_w=invariance_w,
            variance_w=variance_w, covariance_w=covariance_w,
            variance_target=variance_target,
            factorization_w=factorization_w,
            factorization_variance=factorization_variance,
            factorization_margin=factorization_margin,
            factorization_covariance_w=factorization_covariance_w,
            fer_w=fer_w,
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w,
            fer_balance_w=fer_balance_w,
            memory_w=memory_w,
            memory_size=memory_size,
            memory_temperature=memory_temperature,
            memory_momentum=memory_momentum,
            memory_balance_w=memory_balance_w,
            association_w=association_w,
            association_temperature=association_temperature,
            association_decay=association_decay,
            association_target_power=association_target_power,
            association_self_loop_w=association_self_loop_w,
            association_transitive_steps=association_transitive_steps,
            association_transitive_w=association_transitive_w,
            composition_w=composition_w,
            composition_temperature=composition_temperature,
            composition_self_loop_w=composition_self_loop_w,
            composition_transitive_steps=composition_transitive_steps,
            composition_transitive_w=composition_transitive_w,
            composition_margin=composition_margin,
            graph_predict_w=graph_predict_w,
            graph_predict_temperature=graph_predict_temperature,
            graph_predict_self_loop_w=graph_predict_self_loop_w,
            graph_predict_transitive_steps=graph_predict_transitive_steps,
            graph_predict_transitive_w=graph_predict_transitive_w,
            graph_predict_target_power=graph_predict_target_power,
            graph_cycle_w=graph_cycle_w,
            graph_cycle_temperature=graph_cycle_temperature,
            graph_cycle_self_loop_w=graph_cycle_self_loop_w,
            graph_cycle_transitive_steps=graph_cycle_transitive_steps,
            graph_cycle_transitive_w=graph_cycle_transitive_w,
            graph_cycle_target_power=graph_cycle_target_power,
            graph_cycle_consistency_w=graph_cycle_consistency_w,
            bridge_w=bridge_w,
            context_target_w=context_target_w,
            context_keep_p=context_keep_p,
            context_target_temperature=context_target_temperature,
            sequence_w=sequence_w,
            sequence_batch=sequence_batch,
            sequence_temperature=sequence_temperature,
            neighborhood_w=neighborhood_w,
            neighborhood_batch=neighborhood_batch,
            neighborhood_probe_n=neighborhood_probe_n,
            neighborhood_refresh_steps=neighborhood_refresh_steps,
            neighborhood_temperature=neighborhood_temperature,
            neighborhood_margin=neighborhood_margin,
            transition_w=transition_w,
            transition_batch=transition_batch,
            transition_temperature=transition_temperature,
            transition_margin=transition_margin,
            cluster_w=cluster_w,
            cluster_batch=cluster_batch,
            cluster_probe_n=cluster_probe_n,
            cluster_refresh_steps=cluster_refresh_steps,
            cluster_temperature=cluster_temperature,
            cluster_margin=cluster_margin,
            cluster_min_size=cluster_min_size,
            study_strategy=study_strategy, study_probe_n=study_probe_n,
            study_hard_max=study_hard_max,
            study_refresh_steps=study_refresh_steps,
            replay_records=replay_records,
            replay_teacher_model=replay_teacher_model,
            replay_teacher_vocab=replay_teacher_vocab,
            replay_w=replay_w, replay_batch=replay_batch)
        round_train_metrics = dict(getattr(model, "reading_train_metrics", {}))
        all_study_reports.extend(
            report | {"round": int(round_i)}
            for report in getattr(model, "reading_study_reports", []))
        all_neighborhood_reports.extend(
            report | {"round": int(round_i)}
            for report in getattr(model, "reading_neighborhood_reports", []))
        all_cluster_reports.extend(
            report | {"round": int(round_i)}
            for report in getattr(model, "reading_cluster_reports", []))
        bundle = reading_eval_bundle(
            model, vocab, records, device=device, eval_n=eval_n, seed=seed,
            token_drop_p=token_drop_p, token_replace_p=token_replace_p,
            context_keep_p=context_keep_p, score_metric=score_metric,
            score_margin_w=score_margin_w)
        replay_bundle = None
        if before_replay_bundle is not None:
            replay_bundle = reading_eval_bundle(
                model, vocab, replay_records, device=device, eval_n=eval_n,
                seed=seed + 4093, token_drop_p=token_drop_p,
                token_replace_p=token_replace_p, context_keep_p=context_keep_p,
                score_metric=score_metric, score_margin_w=score_margin_w)
        row = selection_row(round_i, round_steps, bundle, replay_bundle)
        score = float(row["score"])
        score_delta_from_best = float(score - best_score)
        insight = round_train_metrics.get("study_pool_insight")
        insight_delta, insight_allowed = bridge_insight_delta(insight)
        selected = insight_allowed and score_delta_from_best > score_min_delta
        rounds_report.append(row | {
            "selected": bool(selected),
            "score_delta_from_best": score_delta_from_best,
            "bridge_insight_delta": float(insight_delta),
            "bridge_insight_allowed": bool(insight_allowed),
            "study_pool_insight": insight,
            "study_pool_bridge_before": round_train_metrics.get(
                "study_pool_bridge_before"),
            "study_pool_bridge_after": round_train_metrics.get(
                "study_pool_bridge_after"),
        })
        if selected:
            best_score = score
            best_round = round_i
            best_state = _model_state_copy(model)
            best_metrics = round_train_metrics
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if score_patience and no_improve_rounds >= score_patience:
                stopped_early = True
                stop_round = round_i
                break
    model.load_state_dict(best_state, strict=False)
    for row in rounds_report:
        row["selected"] = row["round"] == best_round
    selection = {
        "enabled": True,
        "rounds_requested": int(rounds),
        "rounds_planned": len(schedule),
        "rounds_run": len(rounds_report) - 1,
        "score_metric": score_metric,
        "score_margin_w": float(score_margin_w),
        "score_min_delta": float(score_min_delta),
        "score_patience": int(score_patience),
        "bridge_insight_gate": bool(bridge_insight_gate),
        "stopped_early": bool(stopped_early),
        "stop_round": int(stop_round),
        "no_improve_rounds": int(no_improve_rounds),
        "replay_retention_w": float(replay_retention_w),
        "selected_round": int(best_round),
        "accepted_update": bool(best_round > 0),
        "selected_score": float(best_score),
        "before_score": float(initial_row["score"]),
        "selected_score_delta": float(best_score - initial_row["score"]),
        "before_base_score": float(initial_row["base_score"]),
        "before_replay_score": (
            float(before_replay_bundle["score_components"]["score"])
            if before_replay_bundle is not None else None),
        "rounds": rounds_report,
    }
    selected_rows = [row for row in rounds_report if row["round"] == best_round]
    if selected_rows:
        selection["selected_insight"] = selected_rows[0].get("study_pool_insight")
    best_metrics = best_metrics | {"selection": selection}
    model.reading_train_metrics = best_metrics
    model.reading_study_reports = all_study_reports
    model.reading_neighborhood_reports = all_neighborhood_reports
    model.reading_cluster_reports = all_cluster_reports
    model.reading_selection_report = selection
    return model, vocab, selection


def train_reading_concepts(records, steps=400, batch=32, d=96, layers=3, heads=4,
                           text_encoder_arch="transformer", text_encoder_layers=1,
                           latent_concept_slots=4, latent_concept_layers=1,
                           latent_concept_prefix=False,
                           latent_concept_refine=False,
                           latent_concept_refine_gate_init=-2.0,
                           lr=1e-3, seed=0, device=DEV, log_every=100,
                           token_drop_p=0.15, token_replace_p=0.05,
                           feature_dropout=0.1,
                           invariance_w=25.0, variance_w=25.0,
                           covariance_w=1.0, variance_target=1.0,
                           factorization_w=0.05, factorization_variance=0.05,
                           factorization_margin=0.2,
                           factorization_covariance_w=0.05,
                           fer_w=0.0, fer_fragmentation_w=1.0,
                           fer_correlation_w=1.0, fer_balance_w=0.1,
                           memory_w=0.05, memory_size=64,
                           memory_temperature=0.1, memory_momentum=0.95,
                           memory_balance_w=0.01,
                           association_w=0.05, association_temperature=0.1,
                           association_decay=0.99, association_target_power=1.0,
                           association_self_loop_w=0.05,
                           association_transitive_steps=2,
                           association_transitive_w=0.1,
                           composition_w=0.0, composition_temperature=0.1,
                           composition_self_loop_w=0.0,
                           composition_transitive_steps=2,
                           composition_transitive_w=0.1,
                           composition_margin=0.0,
                           graph_predict_w=0.0, graph_predict_temperature=0.1,
                           graph_predict_self_loop_w=0.05,
                           graph_predict_transitive_steps=2,
                           graph_predict_transitive_w=0.1,
                           graph_predict_target_power=1.0,
                           graph_cycle_w=0.0, graph_cycle_temperature=0.1,
                           graph_cycle_self_loop_w=0.05,
                           graph_cycle_transitive_steps=2,
                           graph_cycle_transitive_w=0.1,
                           graph_cycle_target_power=1.0,
                           graph_cycle_consistency_w=0.5,
                           bridge_w=0.0,
                           context_target_w=0.1, context_keep_p=0.5,
                           context_target_temperature=0.1,
                           sequence_w=0.05, sequence_batch=0,
                           sequence_temperature=0.1,
                           neighborhood_w=0.0, neighborhood_batch=0,
                           neighborhood_probe_n=0, neighborhood_refresh_steps=0,
                           neighborhood_temperature=0.1,
                           neighborhood_margin=0.0,
                           transition_w=0.05, transition_batch=0,
                           transition_temperature=0.1, transition_margin=0.0,
                           cluster_w=0.0, cluster_batch=0,
                           cluster_probe_n=0, cluster_refresh_steps=0,
                           cluster_temperature=0.1, cluster_margin=0.0,
                           cluster_min_size=2,
                           study_strategy="auto", study_probe_n=0,
                           study_hard_max=0, study_refresh_steps=0):
    if int(latent_concept_slots) <= 0:
        raise ValueError("raw reading concept training requires latent_concept_slots > 0")
    torch.manual_seed(seed)
    vocab = build_reading_vocab(records)
    model = TextFactLM(len(vocab), d=d, layers=layers, heads=heads, pad=vocab.pad,
                       fact_schema=None,
                       text_encoder_arch=text_encoder_arch,
                       text_encoder_layers=text_encoder_layers,
                       latent_concept_slots=latent_concept_slots,
                       latent_concept_layers=latent_concept_layers,
                       latent_concept_prefix=latent_concept_prefix,
                       latent_concept_refine=latent_concept_refine,
                       latent_concept_refine_gate_init=(
                           latent_concept_refine_gate_init),
                       latent_concept_memory_size=memory_size).to(device)
    return fit_reading_concepts(
        model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
        device=device, log_every=log_every, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p, feature_dropout=feature_dropout,
        invariance_w=invariance_w, variance_w=variance_w,
        covariance_w=covariance_w, variance_target=variance_target,
        factorization_w=factorization_w,
        factorization_variance=factorization_variance,
        factorization_margin=factorization_margin,
        factorization_covariance_w=factorization_covariance_w,
        fer_w=fer_w,
        fer_fragmentation_w=fer_fragmentation_w,
        fer_correlation_w=fer_correlation_w,
        fer_balance_w=fer_balance_w,
        memory_w=memory_w,
        memory_size=memory_size,
        memory_temperature=memory_temperature,
        memory_momentum=memory_momentum,
        memory_balance_w=memory_balance_w,
        association_w=association_w,
        association_temperature=association_temperature,
        association_decay=association_decay,
        association_target_power=association_target_power,
        association_self_loop_w=association_self_loop_w,
        association_transitive_steps=association_transitive_steps,
        association_transitive_w=association_transitive_w,
        composition_w=composition_w,
        composition_temperature=composition_temperature,
        composition_self_loop_w=composition_self_loop_w,
        composition_transitive_steps=composition_transitive_steps,
        composition_transitive_w=composition_transitive_w,
        composition_margin=composition_margin,
        graph_predict_w=graph_predict_w,
        graph_predict_temperature=graph_predict_temperature,
        graph_predict_self_loop_w=graph_predict_self_loop_w,
        graph_predict_transitive_steps=graph_predict_transitive_steps,
        graph_predict_transitive_w=graph_predict_transitive_w,
        graph_predict_target_power=graph_predict_target_power,
        graph_cycle_w=graph_cycle_w,
        graph_cycle_temperature=graph_cycle_temperature,
        graph_cycle_self_loop_w=graph_cycle_self_loop_w,
        graph_cycle_transitive_steps=graph_cycle_transitive_steps,
        graph_cycle_transitive_w=graph_cycle_transitive_w,
        graph_cycle_target_power=graph_cycle_target_power,
        graph_cycle_consistency_w=graph_cycle_consistency_w,
        bridge_w=bridge_w,
        context_target_w=context_target_w, context_keep_p=context_keep_p,
        context_target_temperature=context_target_temperature,
        sequence_w=sequence_w, sequence_batch=sequence_batch,
        sequence_temperature=sequence_temperature,
        neighborhood_w=neighborhood_w,
        neighborhood_batch=neighborhood_batch,
        neighborhood_probe_n=neighborhood_probe_n,
        neighborhood_refresh_steps=neighborhood_refresh_steps,
        neighborhood_temperature=neighborhood_temperature,
        neighborhood_margin=neighborhood_margin,
        transition_w=transition_w,
        transition_batch=transition_batch,
        transition_temperature=transition_temperature,
        transition_margin=transition_margin,
        cluster_w=cluster_w,
        cluster_batch=cluster_batch,
        cluster_probe_n=cluster_probe_n,
        cluster_refresh_steps=cluster_refresh_steps,
        cluster_temperature=cluster_temperature,
        cluster_margin=cluster_margin,
        cluster_min_size=cluster_min_size,
        study_strategy=study_strategy, study_probe_n=study_probe_n,
        study_hard_max=study_hard_max, study_refresh_steps=study_refresh_steps)


def run_reading_concepts(data, steps=400, batch=32, d=96, layers=3, heads=4,
                         text_encoder_arch="transformer", text_encoder_layers=1,
                         latent_concept_slots=4, latent_concept_layers=1,
                         latent_concept_prefix=False, latent_concept_refine=False,
                         latent_concept_refine_gate_init=-2.0,
                         lr=1e-3, seed=0, device=DEV, log_every=100,
                         token_drop_p=0.15, token_replace_p=0.05,
                         feature_dropout=0.1,
                         invariance_w=25.0, variance_w=25.0,
                         covariance_w=1.0, variance_target=1.0,
                         factorization_w=0.05, factorization_variance=0.05,
                         factorization_margin=0.2,
                         factorization_covariance_w=0.05,
                         fer_w=0.0, fer_fragmentation_w=1.0,
                         fer_correlation_w=1.0, fer_balance_w=0.1,
                         memory_w=0.05, memory_size=64,
                         memory_temperature=0.1, memory_momentum=0.95,
                         memory_balance_w=0.01,
                         association_w=0.05, association_temperature=0.1,
                         association_decay=0.99, association_target_power=1.0,
                         association_self_loop_w=0.05,
                         association_transitive_steps=2,
                         association_transitive_w=0.1,
                         composition_w=0.0, composition_temperature=0.1,
                         composition_self_loop_w=0.0,
                         composition_transitive_steps=2,
                         composition_transitive_w=0.1,
                         composition_margin=0.0,
                         graph_predict_w=0.0, graph_predict_temperature=0.1,
                         graph_predict_self_loop_w=0.05,
                         graph_predict_transitive_steps=2,
                         graph_predict_transitive_w=0.1,
                         graph_predict_target_power=1.0,
                         graph_cycle_w=0.0, graph_cycle_temperature=0.1,
                         graph_cycle_self_loop_w=0.05,
                         graph_cycle_transitive_steps=2,
                         graph_cycle_transitive_w=0.1,
                         graph_cycle_target_power=1.0,
                         graph_cycle_consistency_w=0.5,
                         bridge_w=0.0,
                         context_target_w=0.1, context_keep_p=0.5,
                         context_target_temperature=0.1,
                         sequence_w=0.05, sequence_batch=0,
                         sequence_temperature=0.1,
                         neighborhood_w=0.0, neighborhood_batch=0,
                         neighborhood_probe_n=0, neighborhood_refresh_steps=0,
                         neighborhood_temperature=0.1,
                         neighborhood_margin=0.0,
                         transition_w=0.05, transition_batch=0,
                         transition_temperature=0.1, transition_margin=0.0,
                         cluster_w=0.0, cluster_batch=0,
                         cluster_probe_n=0, cluster_refresh_steps=0,
                         cluster_temperature=0.1, cluster_margin=0.0,
                         cluster_min_size=2,
                         study_strategy="auto", study_probe_n=0,
                         study_hard_max=0, study_refresh_steps=0,
                         study_select_best=True, study_rounds=1,
                         study_score_metric="mastery", study_score_margin_w=0.1,
                         study_score_min_delta=0.0, study_score_patience=0,
                         text_field="text", max_tokens=128, min_tokens=8,
                         eval_frac=0.10, eval_n=64, out=None, checkpoint=None):
    records = load_reading_records(
        data, text_field=text_field, max_tokens=max_tokens, min_tokens=min_tokens,
        eval_frac=eval_frac, seed=seed)
    torch.manual_seed(seed)
    vocab = build_reading_vocab(records)
    model = TextFactLM(len(vocab), d=d, layers=layers, heads=heads, pad=vocab.pad,
                       fact_schema=None,
                       text_encoder_arch=text_encoder_arch,
                       text_encoder_layers=text_encoder_layers,
                       latent_concept_slots=latent_concept_slots,
                       latent_concept_layers=latent_concept_layers,
                       latent_concept_prefix=latent_concept_prefix,
                       latent_concept_refine=latent_concept_refine,
                       latent_concept_refine_gate_init=(
                           latent_concept_refine_gate_init),
                       latent_concept_memory_size=memory_size).to(device)
    before_bundle = reading_eval_bundle(
        model, vocab, records, device=device, eval_n=eval_n, seed=seed,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p,
        context_keep_p=context_keep_p, score_metric=study_score_metric,
        score_margin_w=study_score_margin_w)
    selection = {"enabled": False}
    if study_select_best:
        _model, _vocab, selection = fit_reading_concepts_select_best(
            model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
            device=device, log_every=log_every, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, feature_dropout=feature_dropout,
            invariance_w=invariance_w, variance_w=variance_w,
            covariance_w=covariance_w, variance_target=variance_target,
            factorization_w=factorization_w,
            factorization_variance=factorization_variance,
            factorization_margin=factorization_margin,
            factorization_covariance_w=factorization_covariance_w,
            fer_w=fer_w,
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w,
            fer_balance_w=fer_balance_w,
            memory_w=memory_w,
            memory_size=memory_size,
            memory_temperature=memory_temperature,
            memory_momentum=memory_momentum,
            memory_balance_w=memory_balance_w,
            association_w=association_w,
            association_temperature=association_temperature,
            association_decay=association_decay,
            association_target_power=association_target_power,
            association_self_loop_w=association_self_loop_w,
            association_transitive_steps=association_transitive_steps,
            association_transitive_w=association_transitive_w,
            composition_w=composition_w,
            composition_temperature=composition_temperature,
            composition_self_loop_w=composition_self_loop_w,
            composition_transitive_steps=composition_transitive_steps,
            composition_transitive_w=composition_transitive_w,
            composition_margin=composition_margin,
            graph_predict_w=graph_predict_w,
            graph_predict_temperature=graph_predict_temperature,
            graph_predict_self_loop_w=graph_predict_self_loop_w,
            graph_predict_transitive_steps=graph_predict_transitive_steps,
            graph_predict_transitive_w=graph_predict_transitive_w,
            graph_predict_target_power=graph_predict_target_power,
            graph_cycle_w=graph_cycle_w,
            graph_cycle_temperature=graph_cycle_temperature,
            graph_cycle_self_loop_w=graph_cycle_self_loop_w,
            graph_cycle_transitive_steps=graph_cycle_transitive_steps,
            graph_cycle_transitive_w=graph_cycle_transitive_w,
            graph_cycle_target_power=graph_cycle_target_power,
            graph_cycle_consistency_w=graph_cycle_consistency_w,
            bridge_w=bridge_w,
            context_target_w=context_target_w, context_keep_p=context_keep_p,
            context_target_temperature=context_target_temperature,
            sequence_w=sequence_w, sequence_batch=sequence_batch,
            sequence_temperature=sequence_temperature,
            neighborhood_w=neighborhood_w,
            neighborhood_batch=neighborhood_batch,
            neighborhood_probe_n=neighborhood_probe_n,
            neighborhood_refresh_steps=neighborhood_refresh_steps,
            neighborhood_temperature=neighborhood_temperature,
            neighborhood_margin=neighborhood_margin,
            transition_w=transition_w,
            transition_batch=transition_batch,
            transition_temperature=transition_temperature,
            transition_margin=transition_margin,
            cluster_w=cluster_w,
            cluster_batch=cluster_batch,
            cluster_probe_n=cluster_probe_n,
            cluster_refresh_steps=cluster_refresh_steps,
            cluster_temperature=cluster_temperature,
            cluster_margin=cluster_margin,
            cluster_min_size=cluster_min_size,
            study_strategy=study_strategy, study_probe_n=study_probe_n,
            study_hard_max=study_hard_max,
            study_refresh_steps=study_refresh_steps, eval_n=eval_n,
            score_metric=study_score_metric,
            score_margin_w=study_score_margin_w,
            score_min_delta=study_score_min_delta,
            score_patience=study_score_patience,
            rounds=study_rounds,
            before_bundle=before_bundle)
    else:
        fit_reading_concepts(
            model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
            device=device, log_every=log_every, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, feature_dropout=feature_dropout,
            invariance_w=invariance_w, variance_w=variance_w,
            covariance_w=covariance_w, variance_target=variance_target,
            factorization_w=factorization_w,
            factorization_variance=factorization_variance,
            factorization_margin=factorization_margin,
            factorization_covariance_w=factorization_covariance_w,
            fer_w=fer_w,
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w,
            fer_balance_w=fer_balance_w,
            memory_w=memory_w,
            memory_size=memory_size,
            memory_temperature=memory_temperature,
            memory_momentum=memory_momentum,
            memory_balance_w=memory_balance_w,
            association_w=association_w,
            association_temperature=association_temperature,
            association_decay=association_decay,
            association_target_power=association_target_power,
            association_self_loop_w=association_self_loop_w,
            association_transitive_steps=association_transitive_steps,
            association_transitive_w=association_transitive_w,
            composition_w=composition_w,
            composition_temperature=composition_temperature,
            composition_self_loop_w=composition_self_loop_w,
            composition_transitive_steps=composition_transitive_steps,
            composition_transitive_w=composition_transitive_w,
            composition_margin=composition_margin,
            graph_predict_w=graph_predict_w,
            graph_predict_temperature=graph_predict_temperature,
            graph_predict_self_loop_w=graph_predict_self_loop_w,
            graph_predict_transitive_steps=graph_predict_transitive_steps,
            graph_predict_transitive_w=graph_predict_transitive_w,
            graph_predict_target_power=graph_predict_target_power,
            graph_cycle_w=graph_cycle_w,
            graph_cycle_temperature=graph_cycle_temperature,
            graph_cycle_self_loop_w=graph_cycle_self_loop_w,
            graph_cycle_transitive_steps=graph_cycle_transitive_steps,
            graph_cycle_transitive_w=graph_cycle_transitive_w,
            graph_cycle_target_power=graph_cycle_target_power,
            graph_cycle_consistency_w=graph_cycle_consistency_w,
            bridge_w=bridge_w,
            context_target_w=context_target_w, context_keep_p=context_keep_p,
            context_target_temperature=context_target_temperature,
            sequence_w=sequence_w, sequence_batch=sequence_batch,
            sequence_temperature=sequence_temperature,
            neighborhood_w=neighborhood_w,
            neighborhood_batch=neighborhood_batch,
            neighborhood_probe_n=neighborhood_probe_n,
            neighborhood_refresh_steps=neighborhood_refresh_steps,
            neighborhood_temperature=neighborhood_temperature,
            neighborhood_margin=neighborhood_margin,
            transition_w=transition_w,
            transition_batch=transition_batch,
            transition_temperature=transition_temperature,
            transition_margin=transition_margin,
            cluster_w=cluster_w,
            cluster_batch=cluster_batch,
            cluster_probe_n=cluster_probe_n,
            cluster_refresh_steps=cluster_refresh_steps,
            cluster_temperature=cluster_temperature,
            cluster_margin=cluster_margin,
            cluster_min_size=cluster_min_size,
            study_strategy=study_strategy, study_probe_n=study_probe_n,
            study_hard_max=study_hard_max,
            study_refresh_steps=study_refresh_steps)
    after_bundle = reading_eval_bundle(
        model, vocab, records, device=device, eval_n=eval_n, seed=seed,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p,
        context_keep_p=context_keep_p, score_metric=study_score_metric,
        score_margin_w=study_score_margin_w)
    before = before_bundle["view"]
    after = after_bundle["view"]
    before_context = before_bundle["context_target"]
    after_context = after_bundle["context_target"]
    before_sequence = before_bundle["sequence"]
    after_sequence = after_bundle["sequence"]
    before_neighborhood = before_bundle["neighborhood"]
    after_neighborhood = after_bundle["neighborhood"]
    before_cluster = before_bundle["cluster"]
    after_cluster = after_bundle["cluster"]
    before_fer = before_bundle["fer"]
    after_fer = after_bundle["fer"]
    before_bridge = before_bundle["bridge"]
    after_bridge = after_bundle["bridge"]
    train_metrics = getattr(model, "reading_train_metrics", {})
    resolved_study_strategy = train_metrics.get("study_strategy", study_strategy)
    requested_study_strategy = train_metrics.get(
        "study_strategy_requested", study_strategy)
    report = {"experiment": "text_raw_reading_concept_pretrain",
              "data": data,
              "steps": int(steps), "batch": int(batch), "lr": float(lr),
              "text_encoder_arch": text_encoder_arch,
              "text_encoder_layers": int(text_encoder_layers),
              "latent_concept_slots": int(latent_concept_slots),
              "latent_concept_layers": int(latent_concept_layers),
              "latent_concept_prefix": bool(latent_concept_prefix),
              "latent_concept_refine": bool(latent_concept_refine),
              "latent_concept_refine_gate_init": float(
                  latent_concept_refine_gate_init),
              "token_drop_p": float(token_drop_p),
              "token_replace_p": float(token_replace_p),
              "feature_dropout": float(feature_dropout),
              "invariance_w": float(invariance_w),
              "variance_w": float(variance_w),
              "covariance_w": float(covariance_w),
              "variance_target": float(variance_target),
              "factorization_w": float(factorization_w),
              "factorization_variance": float(factorization_variance),
              "factorization_margin": float(factorization_margin),
              "factorization_covariance_w": float(factorization_covariance_w),
              "fer_w": float(fer_w),
              "fer_fragmentation_w": float(fer_fragmentation_w),
              "fer_correlation_w": float(fer_correlation_w),
              "fer_balance_w": float(fer_balance_w),
              "memory_w": float(memory_w),
              "memory_size": int(memory_size),
              "memory_temperature": float(memory_temperature),
              "memory_momentum": float(memory_momentum),
              "memory_balance_w": float(memory_balance_w),
              "association_w": float(association_w),
              "association_temperature": float(association_temperature),
              "association_decay": float(association_decay),
              "association_target_power": float(association_target_power),
              "association_self_loop_w": float(association_self_loop_w),
              "association_transitive_steps": int(association_transitive_steps),
              "association_transitive_w": float(association_transitive_w),
              "composition_w": float(composition_w),
              "composition_temperature": float(composition_temperature),
              "composition_self_loop_w": float(composition_self_loop_w),
              "composition_transitive_steps": int(composition_transitive_steps),
              "composition_transitive_w": float(composition_transitive_w),
              "composition_margin": float(composition_margin),
              "graph_predict_w": float(graph_predict_w),
              "graph_predict_temperature": float(graph_predict_temperature),
              "graph_predict_self_loop_w": float(graph_predict_self_loop_w),
              "graph_predict_transitive_steps": int(graph_predict_transitive_steps),
              "graph_predict_transitive_w": float(graph_predict_transitive_w),
              "graph_predict_target_power": float(graph_predict_target_power),
              "graph_cycle_w": float(graph_cycle_w),
              "graph_cycle_temperature": float(graph_cycle_temperature),
              "graph_cycle_self_loop_w": float(graph_cycle_self_loop_w),
              "graph_cycle_transitive_steps": int(graph_cycle_transitive_steps),
              "graph_cycle_transitive_w": float(graph_cycle_transitive_w),
              "graph_cycle_target_power": float(graph_cycle_target_power),
              "graph_cycle_consistency_w": float(graph_cycle_consistency_w),
              "bridge_w": float(bridge_w),
              "context_target_w": float(context_target_w),
              "context_keep_p": float(context_keep_p),
              "context_target_temperature": float(context_target_temperature),
              "sequence_w": float(sequence_w),
              "sequence_batch": int(sequence_batch),
              "sequence_temperature": float(sequence_temperature),
              "neighborhood_w": float(neighborhood_w),
              "neighborhood_batch": int(neighborhood_batch),
              "neighborhood_probe_n": int(neighborhood_probe_n),
              "neighborhood_refresh_steps": int(neighborhood_refresh_steps),
              "neighborhood_temperature": float(neighborhood_temperature),
              "neighborhood_margin": float(neighborhood_margin),
              "transition_w": float(transition_w),
              "transition_batch": int(transition_batch),
              "transition_temperature": float(transition_temperature),
              "transition_margin": float(transition_margin),
              "cluster_w": float(cluster_w),
              "cluster_batch": int(cluster_batch),
              "cluster_probe_n": int(cluster_probe_n),
              "cluster_refresh_steps": int(cluster_refresh_steps),
              "cluster_temperature": float(cluster_temperature),
              "cluster_margin": float(cluster_margin),
              "cluster_min_size": int(cluster_min_size),
              "study_strategy_requested": requested_study_strategy,
              "study_strategy": resolved_study_strategy,
              "study_probe_n": int(study_probe_n),
              "study_hard_max": int(study_hard_max),
              "study_refresh_steps": int(study_refresh_steps),
              "study_select_best": bool(study_select_best),
              "study_rounds": int(study_rounds),
              "study_score_metric": study_score_metric,
              "study_score_margin_w": float(study_score_margin_w),
              "study_score_min_delta": float(study_score_min_delta),
              "study_score_patience": int(study_score_patience),
              "train_records": sum(r.split == "train" for r in records),
              "eval_records": sum(r.split == "eval" for r in records),
              "vocab_size": len(vocab),
              "before": before,
              "after": after,
              "before_context_target": before_context,
              "after_context_target": after_context,
              "before_sequence": before_sequence,
              "after_sequence": after_sequence,
              "before_neighborhood": before_neighborhood,
              "after_neighborhood": after_neighborhood,
              "before_cluster": before_cluster,
              "after_cluster": after_cluster,
              "before_fer": before_fer,
              "after_fer": after_fer,
              "before_bridge": before_bridge,
              "after_bridge": after_bridge,
              "before_score_components": before_bundle["score_components"],
              "after_score_components": after_bundle["score_components"],
              "selection": selection,
              "delta": {
                  "score": (
                      after_bundle["score_components"].get("score", 0.0)
                      - before_bundle["score_components"].get("score", 0.0)),
                  "mastery_score": (
                      after_bundle["score_components"].get("mastery_score", 0.0)
                      - before_bundle["score_components"].get("mastery_score", 0.0)),
                  "active_mean_score": (
                      after_bundle["score_components"].get("active_mean_score", 0.0)
                      - before_bundle["score_components"].get("active_mean_score", 0.0)),
                  "signal_coverage": (
                      after_bundle["score_components"].get("signal_coverage", 0.0)
                      - before_bundle["score_components"].get("signal_coverage", 0.0)),
                  "balanced_score": (
                      after_bundle["score_components"].get("balanced_score", 0.0)
                      - before_bundle["score_components"].get("balanced_score", 0.0)),
                  "paired_view_acc": (
                      after["paired_view_acc"] - before["paired_view_acc"]),
                  "margin": after.get("margin", 0.0) - before.get("margin", 0.0),
                  "context_target_acc": (
                      after_context.get("context_target_acc", 0.0)
                      - before_context.get("context_target_acc", 0.0)),
                  "context_target_margin": (
                      after_context.get("margin", 0.0)
                      - before_context.get("margin", 0.0)),
                  "sequence_acc": (
                      after_sequence.get("sequence_acc", 0.0)
                      - before_sequence.get("sequence_acc", 0.0)),
                  "sequence_margin": (
                      after_sequence.get("margin", 0.0)
                      - before_sequence.get("margin", 0.0)),
                  "neighborhood_acc": (
                      after_neighborhood.get("neighbor_acc", 0.0)
                      - before_neighborhood.get("neighbor_acc", 0.0)),
                  "neighborhood_margin": (
                      after_neighborhood.get("margin", 0.0)
                      - before_neighborhood.get("margin", 0.0)),
                  "cluster_acc": (
                      after_cluster.get("cluster_acc", 0.0)
                      - before_cluster.get("cluster_acc", 0.0)),
                  "cluster_margin": (
                      after_cluster.get("margin", 0.0)
                      - before_cluster.get("margin", 0.0)),
                  "fer_quality": (
                      after_bundle["score_components"].get("fer_score", 0.0)
                      - before_bundle["score_components"].get("fer_score", 0.0)),
                  "fer_raw_score": (
                      after_fer.get("fer_score", 0.0)
                      - before_fer.get("fer_score", 0.0)),
                  "bridge_quality": (
                      after_bundle["score_components"].get("bridge_score", 0.0)
                      - before_bundle["score_components"].get(
                          "bridge_score", 0.0)),
                  "bridge_raw_score": (
                      after_bridge.get("mean_bridge_score", 0.0)
                      - before_bridge.get("mean_bridge_score", 0.0)),
                  "bridge_connectivity": (
                      after_bridge.get("mean_bridge_connectivity", 0.0)
                      - before_bridge.get("mean_bridge_connectivity", 0.0)),
              },
              "train_metrics": train_metrics,
              "study_hard_examples": getattr(model, "reading_study_reports", []),
              "study_neighborhoods": getattr(
                  model, "reading_neighborhood_reports", []),
              "study_clusters": getattr(model, "reading_cluster_reports", [])}
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        torch.save(checkpoint_payload(model, vocab, d, layers, heads, report),
                   checkpoint)
        report["checkpoint"] = checkpoint
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    return report


def expanded_reading_checkpoint_model(checkpoint, reading_records, device=DEV,
                                      latent_concept_slots=0,
                                      latent_concept_layers=None,
                                      latent_concept_prefix=None,
                                      latent_concept_refine=None,
                                      latent_concept_refine_gate_init=None,
                                      latent_concept_memory_size=None):
    src_model, src_vocab, ckpt = load_checkpoint(checkpoint, device=device)
    vocab = build_reading_vocab(reading_records, base_vocab=src_vocab)
    ckpt_slots = int(getattr(src_model, "latent_concept_slots", 0)
                     or ckpt.get("latent_concept_slots", 0))
    slots = int(latent_concept_slots or ckpt_slots or 4)
    layers = int(latent_concept_layers if latent_concept_layers is not None
                 else ckpt.get("latent_concept_layers", 1))
    use_prefix = (bool(latent_concept_prefix)
                  if latent_concept_prefix is not None
                  else bool(ckpt.get("latent_concept_prefix", False)))
    use_refine = (bool(latent_concept_refine)
                  if latent_concept_refine is not None
                  else bool(ckpt.get("latent_concept_refine", False)))
    refine_gate = float(
        latent_concept_refine_gate_init
        if latent_concept_refine_gate_init is not None
        else ckpt.get("latent_concept_refine_gate_init", -2.0))
    memory_size = int(
        latent_concept_memory_size
        if latent_concept_memory_size is not None
        else ckpt.get("latent_concept_memory_size", 0))
    model = TextFactLM(
        len(vocab), d=int(ckpt.get("d", 96)),
        layers=int(ckpt.get("layers", 3)),
        heads=int(ckpt.get("heads", 4)), pad=vocab.pad,
        fact_schema=src_model.fact_schema,
        fact_concept_prefix=bool(ckpt.get("fact_concept_prefix", False)),
        text_encoder_arch=ckpt.get("text_encoder_arch", "transformer"),
        text_encoder_layers=int(ckpt.get("text_encoder_layers", 1)),
        fact_concept_refine=bool(ckpt.get("fact_concept_refine", False)),
        fact_concept_refine_gate_init=float(
            ckpt.get("fact_concept_refine_gate_init", -2.0)),
        fact_concept_mixer_layers=int(ckpt.get("fact_concept_mixer_layers", 0)),
        fact_concept_mixer_gate_init=float(
            ckpt.get("fact_concept_mixer_gate_init", -2.0)),
        latent_concept_slots=slots,
        latent_concept_layers=layers,
        latent_concept_prefix=use_prefix,
        latent_concept_refine=use_refine,
        latent_concept_refine_gate_init=refine_gate,
        latent_concept_memory_size=memory_size).to(device)
    copy_pretrained_text_weights(src_model, src_vocab, model, vocab)
    model.eval()
    return model, vocab, ckpt


def study_reading_checkpoint(checkpoint, data, out_checkpoint=None, out=None,
                             replay_data=None,
                             steps=400, batch=32, lr=1e-3, seed=0, device=DEV,
                             log_every=100, token_drop_p=0.15,
                             token_replace_p=0.05, feature_dropout=0.1,
                             invariance_w=25.0, variance_w=25.0,
                             covariance_w=1.0, variance_target=1.0,
                             factorization_w=0.05, factorization_variance=0.05,
                             factorization_margin=0.2,
                             factorization_covariance_w=0.05,
                             fer_w=0.0, fer_fragmentation_w=1.0,
                             fer_correlation_w=1.0, fer_balance_w=0.1,
                             memory_w=0.05, memory_size=64,
                             memory_temperature=0.1, memory_momentum=0.95,
                             memory_balance_w=0.01,
                             association_w=0.05, association_temperature=0.1,
                             association_decay=0.99, association_target_power=1.0,
                             association_self_loop_w=0.05,
                             association_transitive_steps=2,
                             association_transitive_w=0.1,
                             composition_w=0.0, composition_temperature=0.1,
                             composition_self_loop_w=0.0,
                             composition_transitive_steps=2,
                             composition_transitive_w=0.1,
                             composition_margin=0.0,
                             graph_predict_w=0.0,
                             graph_predict_temperature=0.1,
                             graph_predict_self_loop_w=0.05,
                             graph_predict_transitive_steps=2,
                             graph_predict_transitive_w=0.1,
                             graph_predict_target_power=1.0,
                             graph_cycle_w=0.0, graph_cycle_temperature=0.1,
                             graph_cycle_self_loop_w=0.05,
                             graph_cycle_transitive_steps=2,
                             graph_cycle_transitive_w=0.1,
                             graph_cycle_target_power=1.0,
                             graph_cycle_consistency_w=0.5,
                             bridge_w=0.0,
                             context_target_w=0.1, context_keep_p=0.5,
                             context_target_temperature=0.1,
                             sequence_w=0.05, sequence_batch=0,
                             sequence_temperature=0.1,
                             neighborhood_w=0.0, neighborhood_batch=0,
                             neighborhood_probe_n=0, neighborhood_refresh_steps=0,
                             neighborhood_temperature=0.1,
                             neighborhood_margin=0.0,
                             transition_w=0.05, transition_batch=0,
                             transition_temperature=0.1, transition_margin=0.0,
                             cluster_w=0.0, cluster_batch=0,
                             cluster_probe_n=0, cluster_refresh_steps=0,
                             cluster_temperature=0.1, cluster_margin=0.0,
                             cluster_min_size=2,
                             study_strategy="auto", study_probe_n=0,
                             study_hard_max=0, study_refresh_steps=0,
                             study_select_best=True, study_rounds=1,
                             study_score_metric="mastery", study_score_margin_w=0.1,
                             study_score_min_delta=0.0, study_score_patience=0,
                             replay_w=0.0, replay_batch=0,
                             replay_retention_w=0.0,
                             text_field="text", max_tokens=128, min_tokens=8,
                             eval_frac=0.10, eval_n=64,
                             latent_concept_slots=0, latent_concept_layers=None,
                             latent_concept_prefix=None,
                             latent_concept_refine=None,
                             latent_concept_refine_gate_init=None):
    records = load_reading_records(
        data, text_field=text_field, max_tokens=max_tokens, min_tokens=min_tokens,
        eval_frac=eval_frac, seed=seed)
    replay_records = (load_reading_records(
        replay_data, require_train=False, require_eval=False, text_field=text_field,
        max_tokens=max_tokens, min_tokens=min_tokens, eval_frac=eval_frac,
        seed=seed + 313) if replay_data else [])
    torch.manual_seed(seed)
    model, vocab, ckpt = expanded_reading_checkpoint_model(
        checkpoint, records + replay_records, device=device,
        latent_concept_slots=latent_concept_slots,
        latent_concept_layers=latent_concept_layers,
        latent_concept_prefix=latent_concept_prefix,
        latent_concept_refine=latent_concept_refine,
        latent_concept_refine_gate_init=latent_concept_refine_gate_init,
        latent_concept_memory_size=memory_size)
    replay_teacher_model = None
    replay_teacher_vocab = None
    if replay_records and (replay_w or replay_retention_w):
        replay_teacher_model, replay_teacher_vocab, _teacher_ckpt = load_checkpoint(
            checkpoint, device=device)
        replay_teacher_model.eval()
        for param in replay_teacher_model.parameters():
            param.requires_grad_(False)
    old_vocab_size = len(ckpt["vocab"])
    before_bundle = reading_eval_bundle(
        model, vocab, records, device=device, eval_n=eval_n, seed=seed,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p,
        context_keep_p=context_keep_p, score_metric=study_score_metric,
        score_margin_w=study_score_margin_w)
    before_replay_bundle = (reading_eval_bundle(
        model, vocab, replay_records, device=device, eval_n=eval_n,
        seed=seed + 4093, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p, context_keep_p=context_keep_p,
        score_metric=study_score_metric, score_margin_w=study_score_margin_w)
        if replay_records else None)
    selection = {"enabled": False}
    if study_select_best:
        _model, _vocab, selection = fit_reading_concepts_select_best(
            model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
            device=device, log_every=log_every, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, feature_dropout=feature_dropout,
            invariance_w=invariance_w, variance_w=variance_w,
            covariance_w=covariance_w, variance_target=variance_target,
            factorization_w=factorization_w,
            factorization_variance=factorization_variance,
            factorization_margin=factorization_margin,
            factorization_covariance_w=factorization_covariance_w,
            fer_w=fer_w,
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w,
            fer_balance_w=fer_balance_w,
            memory_w=memory_w,
            memory_size=memory_size,
            memory_temperature=memory_temperature,
            memory_momentum=memory_momentum,
            memory_balance_w=memory_balance_w,
            association_w=association_w,
            association_temperature=association_temperature,
            association_decay=association_decay,
            association_target_power=association_target_power,
            association_self_loop_w=association_self_loop_w,
            association_transitive_steps=association_transitive_steps,
            association_transitive_w=association_transitive_w,
            composition_w=composition_w,
            composition_temperature=composition_temperature,
            composition_self_loop_w=composition_self_loop_w,
            composition_transitive_steps=composition_transitive_steps,
            composition_transitive_w=composition_transitive_w,
            composition_margin=composition_margin,
            graph_predict_w=graph_predict_w,
            graph_predict_temperature=graph_predict_temperature,
            graph_predict_self_loop_w=graph_predict_self_loop_w,
            graph_predict_transitive_steps=graph_predict_transitive_steps,
            graph_predict_transitive_w=graph_predict_transitive_w,
            graph_predict_target_power=graph_predict_target_power,
            graph_cycle_w=graph_cycle_w,
            graph_cycle_temperature=graph_cycle_temperature,
            graph_cycle_self_loop_w=graph_cycle_self_loop_w,
            graph_cycle_transitive_steps=graph_cycle_transitive_steps,
            graph_cycle_transitive_w=graph_cycle_transitive_w,
            graph_cycle_target_power=graph_cycle_target_power,
            graph_cycle_consistency_w=graph_cycle_consistency_w,
            bridge_w=bridge_w,
            context_target_w=context_target_w, context_keep_p=context_keep_p,
            context_target_temperature=context_target_temperature,
            sequence_w=sequence_w, sequence_batch=sequence_batch,
            sequence_temperature=sequence_temperature,
            neighborhood_w=neighborhood_w,
            neighborhood_batch=neighborhood_batch,
            neighborhood_probe_n=neighborhood_probe_n,
            neighborhood_refresh_steps=neighborhood_refresh_steps,
            neighborhood_temperature=neighborhood_temperature,
            neighborhood_margin=neighborhood_margin,
            transition_w=transition_w,
            transition_batch=transition_batch,
            transition_temperature=transition_temperature,
            transition_margin=transition_margin,
            cluster_w=cluster_w,
            cluster_batch=cluster_batch,
            cluster_probe_n=cluster_probe_n,
            cluster_refresh_steps=cluster_refresh_steps,
            cluster_temperature=cluster_temperature,
            cluster_margin=cluster_margin,
            cluster_min_size=cluster_min_size,
            study_strategy=study_strategy, study_probe_n=study_probe_n,
            study_hard_max=study_hard_max,
            study_refresh_steps=study_refresh_steps, eval_n=eval_n,
            score_metric=study_score_metric,
            score_margin_w=study_score_margin_w,
            score_min_delta=study_score_min_delta,
            score_patience=study_score_patience,
            replay_records=replay_records,
            replay_teacher_model=replay_teacher_model,
            replay_teacher_vocab=replay_teacher_vocab,
            replay_w=replay_w, replay_batch=replay_batch,
            replay_retention_w=replay_retention_w, rounds=study_rounds,
            before_bundle=before_bundle)
    else:
        fit_reading_concepts(
            model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
            device=device, log_every=log_every, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, feature_dropout=feature_dropout,
            invariance_w=invariance_w, variance_w=variance_w,
            covariance_w=covariance_w, variance_target=variance_target,
            factorization_w=factorization_w,
            factorization_variance=factorization_variance,
            factorization_margin=factorization_margin,
            factorization_covariance_w=factorization_covariance_w,
            fer_w=fer_w,
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w,
            fer_balance_w=fer_balance_w,
            memory_w=memory_w,
            memory_size=memory_size,
            memory_temperature=memory_temperature,
            memory_momentum=memory_momentum,
            memory_balance_w=memory_balance_w,
            association_w=association_w,
            association_temperature=association_temperature,
            association_decay=association_decay,
            association_target_power=association_target_power,
            association_self_loop_w=association_self_loop_w,
            association_transitive_steps=association_transitive_steps,
            association_transitive_w=association_transitive_w,
            composition_w=composition_w,
            composition_temperature=composition_temperature,
            composition_self_loop_w=composition_self_loop_w,
            composition_transitive_steps=composition_transitive_steps,
            composition_transitive_w=composition_transitive_w,
            composition_margin=composition_margin,
            graph_predict_w=graph_predict_w,
            graph_predict_temperature=graph_predict_temperature,
            graph_predict_self_loop_w=graph_predict_self_loop_w,
            graph_predict_transitive_steps=graph_predict_transitive_steps,
            graph_predict_transitive_w=graph_predict_transitive_w,
            graph_predict_target_power=graph_predict_target_power,
            graph_cycle_w=graph_cycle_w,
            graph_cycle_temperature=graph_cycle_temperature,
            graph_cycle_self_loop_w=graph_cycle_self_loop_w,
            graph_cycle_transitive_steps=graph_cycle_transitive_steps,
            graph_cycle_transitive_w=graph_cycle_transitive_w,
            graph_cycle_target_power=graph_cycle_target_power,
            graph_cycle_consistency_w=graph_cycle_consistency_w,
            bridge_w=bridge_w,
            context_target_w=context_target_w, context_keep_p=context_keep_p,
            context_target_temperature=context_target_temperature,
            sequence_w=sequence_w, sequence_batch=sequence_batch,
            sequence_temperature=sequence_temperature,
            neighborhood_w=neighborhood_w,
            neighborhood_batch=neighborhood_batch,
            neighborhood_probe_n=neighborhood_probe_n,
            neighborhood_refresh_steps=neighborhood_refresh_steps,
            neighborhood_temperature=neighborhood_temperature,
            neighborhood_margin=neighborhood_margin,
            transition_w=transition_w,
            transition_batch=transition_batch,
            transition_temperature=transition_temperature,
            transition_margin=transition_margin,
            cluster_w=cluster_w,
            cluster_batch=cluster_batch,
            cluster_probe_n=cluster_probe_n,
            cluster_refresh_steps=cluster_refresh_steps,
            cluster_temperature=cluster_temperature,
            cluster_margin=cluster_margin,
            cluster_min_size=cluster_min_size,
            study_strategy=study_strategy, study_probe_n=study_probe_n,
            study_hard_max=study_hard_max,
            study_refresh_steps=study_refresh_steps,
            replay_records=replay_records,
            replay_teacher_model=replay_teacher_model,
            replay_teacher_vocab=replay_teacher_vocab,
            replay_w=replay_w, replay_batch=replay_batch)
    after_bundle = reading_eval_bundle(
        model, vocab, records, device=device, eval_n=eval_n, seed=seed,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p,
        context_keep_p=context_keep_p, score_metric=study_score_metric,
        score_margin_w=study_score_margin_w)
    after_replay_bundle = (reading_eval_bundle(
        model, vocab, replay_records, device=device, eval_n=eval_n,
        seed=seed + 4093, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p, context_keep_p=context_keep_p,
        score_metric=study_score_metric, score_margin_w=study_score_margin_w)
        if replay_records else None)
    before = before_bundle["view"]
    after = after_bundle["view"]
    before_context = before_bundle["context_target"]
    after_context = after_bundle["context_target"]
    before_sequence = before_bundle["sequence"]
    after_sequence = after_bundle["sequence"]
    before_neighborhood = before_bundle["neighborhood"]
    after_neighborhood = after_bundle["neighborhood"]
    before_cluster = before_bundle["cluster"]
    after_cluster = after_bundle["cluster"]
    before_fer = before_bundle["fer"]
    after_fer = after_bundle["fer"]
    before_bridge = before_bundle["bridge"]
    after_bridge = after_bundle["bridge"]
    d = int(ckpt.get("d", 96))
    layers = int(ckpt.get("layers", 3))
    heads = int(ckpt.get("heads", 4))
    train_metrics = getattr(model, "reading_train_metrics", {})
    resolved_study_strategy = train_metrics.get("study_strategy", study_strategy)
    requested_study_strategy = train_metrics.get(
        "study_strategy_requested", study_strategy)
    report = {"experiment": "text_raw_reading_checkpoint_study",
              "checkpoint": checkpoint,
              "checkpoint_experiment": ckpt.get("report", {}).get("experiment"),
              "data": data,
              "replay_data": replay_data or [],
              "steps": int(steps), "batch": int(batch), "lr": float(lr),
              "text_encoder_arch": ckpt.get("text_encoder_arch", "transformer"),
              "text_encoder_layers": int(ckpt.get("text_encoder_layers", 1)),
              "latent_concept_slots": int(getattr(model, "latent_concept_slots", 0)),
              "latent_concept_layers": int(getattr(model, "latent_concept_layers", 1)),
              "latent_concept_prefix": bool(
                  getattr(model, "latent_concept_prefix", False)),
              "latent_concept_refine": bool(
                  getattr(model, "latent_concept_refine", False)),
              "latent_concept_refine_gate_init": float(
                  getattr(model, "latent_concept_refine_gate_init", -2.0)),
              "token_drop_p": float(token_drop_p),
              "token_replace_p": float(token_replace_p),
              "feature_dropout": float(feature_dropout),
              "invariance_w": float(invariance_w),
              "variance_w": float(variance_w),
              "covariance_w": float(covariance_w),
              "variance_target": float(variance_target),
              "factorization_w": float(factorization_w),
              "factorization_variance": float(factorization_variance),
              "factorization_margin": float(factorization_margin),
              "factorization_covariance_w": float(factorization_covariance_w),
              "fer_w": float(fer_w),
              "fer_fragmentation_w": float(fer_fragmentation_w),
              "fer_correlation_w": float(fer_correlation_w),
              "fer_balance_w": float(fer_balance_w),
              "memory_w": float(memory_w),
              "memory_size": int(memory_size),
              "memory_temperature": float(memory_temperature),
              "memory_momentum": float(memory_momentum),
              "memory_balance_w": float(memory_balance_w),
              "association_w": float(association_w),
              "association_temperature": float(association_temperature),
              "association_decay": float(association_decay),
              "association_target_power": float(association_target_power),
              "association_self_loop_w": float(association_self_loop_w),
              "association_transitive_steps": int(association_transitive_steps),
              "association_transitive_w": float(association_transitive_w),
              "composition_w": float(composition_w),
              "composition_temperature": float(composition_temperature),
              "composition_self_loop_w": float(composition_self_loop_w),
              "composition_transitive_steps": int(composition_transitive_steps),
              "composition_transitive_w": float(composition_transitive_w),
              "composition_margin": float(composition_margin),
              "graph_predict_w": float(graph_predict_w),
              "graph_predict_temperature": float(graph_predict_temperature),
              "graph_predict_self_loop_w": float(graph_predict_self_loop_w),
              "graph_predict_transitive_steps": int(graph_predict_transitive_steps),
              "graph_predict_transitive_w": float(graph_predict_transitive_w),
              "graph_predict_target_power": float(graph_predict_target_power),
              "graph_cycle_w": float(graph_cycle_w),
              "graph_cycle_temperature": float(graph_cycle_temperature),
              "graph_cycle_self_loop_w": float(graph_cycle_self_loop_w),
              "graph_cycle_transitive_steps": int(graph_cycle_transitive_steps),
              "graph_cycle_transitive_w": float(graph_cycle_transitive_w),
              "graph_cycle_target_power": float(graph_cycle_target_power),
              "graph_cycle_consistency_w": float(graph_cycle_consistency_w),
              "bridge_w": float(bridge_w),
              "context_target_w": float(context_target_w),
              "context_keep_p": float(context_keep_p),
              "context_target_temperature": float(context_target_temperature),
              "sequence_w": float(sequence_w),
              "sequence_batch": int(sequence_batch),
              "sequence_temperature": float(sequence_temperature),
              "neighborhood_w": float(neighborhood_w),
              "neighborhood_batch": int(neighborhood_batch),
              "neighborhood_probe_n": int(neighborhood_probe_n),
              "neighborhood_refresh_steps": int(neighborhood_refresh_steps),
              "neighborhood_temperature": float(neighborhood_temperature),
              "neighborhood_margin": float(neighborhood_margin),
              "transition_w": float(transition_w),
              "transition_batch": int(transition_batch),
              "transition_temperature": float(transition_temperature),
              "transition_margin": float(transition_margin),
              "cluster_w": float(cluster_w),
              "cluster_batch": int(cluster_batch),
              "cluster_probe_n": int(cluster_probe_n),
              "cluster_refresh_steps": int(cluster_refresh_steps),
              "cluster_temperature": float(cluster_temperature),
              "cluster_margin": float(cluster_margin),
              "cluster_min_size": int(cluster_min_size),
              "study_strategy_requested": requested_study_strategy,
              "study_strategy": resolved_study_strategy,
              "study_probe_n": int(study_probe_n),
              "study_hard_max": int(study_hard_max),
              "study_refresh_steps": int(study_refresh_steps),
              "study_select_best": bool(study_select_best),
              "study_rounds": int(study_rounds),
              "study_score_metric": study_score_metric,
              "study_score_margin_w": float(study_score_margin_w),
              "study_score_min_delta": float(study_score_min_delta),
              "study_score_patience": int(study_score_patience),
              "replay_w": float(replay_w),
              "replay_batch": int(replay_batch),
              "replay_retention_w": float(replay_retention_w),
              "train_records": sum(r.split == "train" for r in records),
              "eval_records": sum(r.split == "eval" for r in records),
              "replay_train_records": sum(r.split == "train" for r in replay_records),
              "replay_eval_records": sum(r.split == "eval" for r in replay_records),
              "old_vocab_size": old_vocab_size,
              "new_vocab_size": len(vocab),
              "new_tokens": max(0, len(vocab) - old_vocab_size),
              "before": before,
              "after": after,
              "before_context_target": before_context,
              "after_context_target": after_context,
              "before_sequence": before_sequence,
              "after_sequence": after_sequence,
              "before_neighborhood": before_neighborhood,
              "after_neighborhood": after_neighborhood,
              "before_cluster": before_cluster,
              "after_cluster": after_cluster,
              "before_fer": before_fer,
              "after_fer": after_fer,
              "before_bridge": before_bridge,
              "after_bridge": after_bridge,
              "before_score_components": before_bundle["score_components"],
              "after_score_components": after_bundle["score_components"],
              "before_replay": before_replay_bundle,
              "after_replay": after_replay_bundle,
              "selection": selection,
              "delta": {
                  "score": (
                      after_bundle["score_components"].get("score", 0.0)
                      - before_bundle["score_components"].get("score", 0.0)),
                  "mastery_score": (
                      after_bundle["score_components"].get("mastery_score", 0.0)
                      - before_bundle["score_components"].get("mastery_score", 0.0)),
                  "active_mean_score": (
                      after_bundle["score_components"].get("active_mean_score", 0.0)
                      - before_bundle["score_components"].get("active_mean_score", 0.0)),
                  "signal_coverage": (
                      after_bundle["score_components"].get("signal_coverage", 0.0)
                      - before_bundle["score_components"].get("signal_coverage", 0.0)),
                  "balanced_score": (
                      after_bundle["score_components"].get("balanced_score", 0.0)
                      - before_bundle["score_components"].get("balanced_score", 0.0)),
                  "paired_view_acc": (
                      after["paired_view_acc"] - before["paired_view_acc"]),
                  "margin": after.get("margin", 0.0) - before.get("margin", 0.0),
                  "context_target_acc": (
                      after_context.get("context_target_acc", 0.0)
                      - before_context.get("context_target_acc", 0.0)),
                  "context_target_margin": (
                      after_context.get("margin", 0.0)
                      - before_context.get("margin", 0.0)),
                  "sequence_acc": (
                      after_sequence.get("sequence_acc", 0.0)
                      - before_sequence.get("sequence_acc", 0.0)),
                  "sequence_margin": (
                      after_sequence.get("margin", 0.0)
                      - before_sequence.get("margin", 0.0)),
                  "neighborhood_acc": (
                      after_neighborhood.get("neighbor_acc", 0.0)
                      - before_neighborhood.get("neighbor_acc", 0.0)),
                  "neighborhood_margin": (
                      after_neighborhood.get("margin", 0.0)
                      - before_neighborhood.get("margin", 0.0)),
                  "cluster_acc": (
                      after_cluster.get("cluster_acc", 0.0)
                      - before_cluster.get("cluster_acc", 0.0)),
                  "cluster_margin": (
                      after_cluster.get("margin", 0.0)
                      - before_cluster.get("margin", 0.0)),
                  "fer_quality": (
                      after_bundle["score_components"].get("fer_score", 0.0)
                      - before_bundle["score_components"].get("fer_score", 0.0)),
                  "fer_raw_score": (
                      after_fer.get("fer_score", 0.0)
                      - before_fer.get("fer_score", 0.0)),
                  "bridge_quality": (
                      after_bundle["score_components"].get("bridge_score", 0.0)
                      - before_bundle["score_components"].get(
                          "bridge_score", 0.0)),
                  "bridge_raw_score": (
                      after_bridge.get("mean_bridge_score", 0.0)
                      - before_bridge.get("mean_bridge_score", 0.0)),
                  "bridge_connectivity": (
                      after_bridge.get("mean_bridge_connectivity", 0.0)
                      - before_bridge.get("mean_bridge_connectivity", 0.0)),
                  "replay_score": (
                      (after_replay_bundle or {}).get("score_components", {}).get(
                          "score", 0.0)
                      - (before_replay_bundle or {}).get("score_components", {}).get(
                          "score", 0.0)),
              },
              "train_metrics": train_metrics,
              "study_hard_examples": getattr(model, "reading_study_reports", []),
              "study_neighborhoods": getattr(
                  model, "reading_neighborhood_reports", []),
              "study_clusters": getattr(model, "reading_cluster_reports", [])}
    if out_checkpoint:
        os.makedirs(os.path.dirname(out_checkpoint) or ".", exist_ok=True)
        torch.save(checkpoint_payload(model, vocab, d, layers, heads, report),
                   out_checkpoint)
        report["out_checkpoint"] = out_checkpoint
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    return report


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


def fact_concept_eval(model, vocab, records, device=DEV, n=0, seed=0):
    if model.fact_schema is None or getattr(model, "fact_concepts", None) is None:
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
            logits = model.fact_concept_logits(txt)
            for r, rec in enumerate(batch):
                for slot, pred, val in rec.facts:
                    key = (slot, pred)
                    if key not in logits or (key, val) not in value_index:
                        continue
                    pred_id = int(logits[key][r].argmax(-1))
                    correct += int(pred_id == value_index[(key, val)])
                    total += 1
    eval_count = len([r for r in records if r.split == "eval"])
    return {"fact_value_acc": correct / max(1, total), "n_facts": total,
            "n_records": len(selected),
            "sampled": bool(n > 0 and n < eval_count),
            "skipped": bool(n < 0)}


def latent_fact_concept_eval(model, vocab, records, device=DEV, n=0, seed=0):
    if (model.fact_schema is None or getattr(model, "fact_concepts", None) is None
            or getattr(model, "latent_concepts", None) is None):
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
            logits = model.latent_fact_concept_logits(txt)
            for r, rec in enumerate(batch):
                for slot, pred, val in rec.facts:
                    key = (slot, pred)
                    if key not in logits or (key, val) not in value_index:
                        continue
                    pred_id = int(logits[key][r].argmax(-1))
                    correct += int(pred_id == value_index[(key, val)])
                    total += 1
    eval_count = len([r for r in records if r.split == "eval"])
    return {"fact_value_acc": correct / max(1, total), "n_facts": total,
            "n_records": len(selected),
            "sampled": bool(n > 0 and n < eval_count),
            "skipped": bool(n < 0)}


def fact_concept_geometry_eval(model, vocab, records, device=DEV, n=0, seed=0):
    if model.fact_schema is None or getattr(model, "fact_concepts", None) is None:
        return {"nearest_same_acc": 0.0, "same_mean": 0.0, "diff_mean": 0.0,
                "margin": 0.0, "n_records": 0, "n_nearest": 0, "same_pairs": 0,
                "diff_pairs": 0, "sampled": False, "skipped": bool(n < 0)}
    selected = eval_records(records, n=n, seed=seed)
    nearest_correct = nearest_total = 0
    same_sum = diff_sum = 0.0
    same_pairs = diff_pairs = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(selected), 64):
            batch = selected[off:off + 64]
            txt, _ids = pack(batch, vocab, device)
            states = model.fact_concept_geometry_states(txt)
            targets_by_key = fact_concept_target_ids(batch, model.fact_schema, device)
            for key, state in states.items():
                targets = targets_by_key.get(key)
                if targets is None:
                    continue
                valid = targets.ge(0)
                if int(valid.sum()) < 2:
                    continue
                z = F.normalize(state[valid], dim=-1)
                labels = targets[valid]
                sim = z.matmul(z.t())
                count = labels.shape[0]
                eye = torch.eye(count, dtype=torch.bool, device=sim.device)
                same = labels[:, None].eq(labels[None, :]) & ~eye
                diff = labels[:, None].ne(labels[None, :])
                if bool(same.any()):
                    same_sum += float(sim[same].sum())
                    same_pairs += int(same.sum())
                if bool(diff.any()):
                    diff_sum += float(sim[diff].sum())
                    diff_pairs += int(diff.sum())
                rows = same.any(-1) & diff.any(-1)
                if bool(rows.any()):
                    nearest = sim.masked_fill(eye, -float("inf")).argmax(-1)
                    nearest_correct += int(labels[nearest][rows].eq(labels[rows]).sum())
                    nearest_total += int(rows.sum())
    eval_count = len([r for r in records if r.split == "eval"])
    same_mean = same_sum / max(1, same_pairs)
    diff_mean = diff_sum / max(1, diff_pairs)
    return {"nearest_same_acc": nearest_correct / max(1, nearest_total),
            "same_mean": same_mean, "diff_mean": diff_mean,
            "margin": same_mean - diff_mean,
            "n_records": len(selected), "n_nearest": nearest_total,
            "same_pairs": same_pairs, "diff_pairs": diff_pairs,
            "sampled": bool(n > 0 and n < eval_count),
            "skipped": bool(n < 0)}


def semantic_record_errors(model, vocab, records, device=DEV, n=0, seed=0):
    """Return records whose semantic heads currently miss at least one supplied fact."""
    if model.fact_schema is None:
        return [], {"n_records": 0, "n_error_records": 0, "n_facts": 0, "n_errors": 0}
    candidates = list(records)
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    value_index = model.fact_schema.value_index
    errors = []
    n_errors = n_facts = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt, _ids = pack(batch, vocab, device)
            logits = model.semantic_logits(txt)
            for r, rec in enumerate(batch):
                rec_errors = rec_facts = 0
                for slot, pred, val in rec.facts:
                    key = (slot, pred)
                    if key not in logits or (key, val) not in value_index:
                        continue
                    pred_id = int(logits[key][r].argmax(-1))
                    rec_facts += 1
                    rec_errors += int(pred_id != value_index[(key, val)])
                if rec_errors:
                    errors.append(rec)
                n_errors += rec_errors
                n_facts += rec_facts
    return errors, {"n_records": len(candidates),
                    "sampled": sampled,
                    "n_error_records": len(errors),
                    "n_facts": n_facts,
                    "n_errors": n_errors,
                    "fact_error_rate": n_errors / max(1, n_facts)}


def latent_fact_record_errors(model, vocab, records, device=DEV, n=0, seed=0):
    """Return records whose latent concept bridge misses at least one supplied fact."""
    errors, _correct, report = latent_fact_record_outcomes(
        model, vocab, records, device=device, n=n, seed=seed)
    return errors, report


def latent_fact_record_outcomes(model, vocab, records, device=DEV, n=0, seed=0):
    """Return currently wrong and right records under the latent fact bridge."""
    if (model.fact_schema is None or getattr(model, "fact_concepts", None) is None
            or getattr(model, "latent_concepts", None) is None):
        return [], [], {"n_records": 0, "sampled": False, "n_error_records": 0,
                        "n_correct_records": 0, "n_facts": 0, "n_errors": 0,
                        "fact_error_rate": 0.0, "by_kind": {}}
    candidates = list(records)
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    value_index = model.fact_schema.value_index
    errors = []
    correct = []
    by_kind = {}
    n_errors = n_facts = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt, _ids = pack(batch, vocab, device)
            logits = model.latent_fact_concept_logits(txt)
            for r, rec in enumerate(batch):
                rec_errors = rec_facts = 0
                for slot, pred, val in rec.facts:
                    key = (slot, pred)
                    if key not in logits or (key, val) not in value_index:
                        continue
                    pred_id = int(logits[key][r].argmax(-1))
                    rec_facts += 1
                    rec_errors += int(pred_id != value_index[(key, val)])
                wrong = bool(rec_errors)
                kind_row = by_kind.setdefault(
                    rec.kind, {"records": 0, "errors": 0, "correct": 0})
                kind_row["records"] += 1
                if wrong:
                    errors.append(rec)
                    kind_row["errors"] += 1
                else:
                    correct.append(rec)
                    kind_row["correct"] += 1
                n_errors += rec_errors
                n_facts += rec_facts
    return errors, correct, {"n_records": len(candidates),
                             "sampled": sampled,
                             "n_error_records": len(errors),
                             "n_correct_records": len(correct),
                             "n_facts": n_facts,
                             "n_errors": n_errors,
                             "fact_error_rate": n_errors / max(1, n_facts),
                             "by_kind": by_kind}


def sample_records_per_kind(records, rng, per_kind):
    if per_kind <= 0:
        return [], {}
    buckets = {}
    for rec in records:
        buckets.setdefault(rec.kind, []).append(rec)
    out = []
    counts = {}
    for kind, rows in sorted(buckets.items()):
        n = min(int(per_kind), len(rows))
        if n <= 0:
            continue
        if n < len(rows):
            idx = rng.choice(len(rows), size=n, replace=False)
            picked = [rows[int(i)] for i in idx]
        else:
            picked = list(rows)
        out.extend(picked)
        counts[kind] = len(picked)
    return out, counts


def unique_records_by_id(*record_lists):
    out = []
    seen = set()
    for records in record_lists:
        for rec in records:
            if rec.rec_id in seen:
                continue
            seen.add(rec.rec_id)
            out.append(rec)
    return out


def retention_anchor_records(records):
    out = []
    for i, rec in enumerate(records):
        meta = rec.meta | {"retention_anchor": True} if isinstance(rec.meta, dict) else {
            "retention_anchor": True}
        out.append(TextRecord(rec_id=f"{rec.rec_id}:retention_anchor:{i}",
                              split=rec.split,
                              tokens=rec.tokens,
                              facts=rec.facts,
                              group=rec.group,
                              kind=f"{rec.kind}:retention_anchor",
                              base_id=rec.base_id,
                              changed=rec.changed,
                              meta=meta))
    return out


def semantic_record_outcomes(model, vocab, records, device=DEV, n=0, seed=0):
    errors, report = semantic_record_errors(model, vocab, records, device=device,
                                            n=n, seed=seed)
    error_ids = {r.rec_id for r in errors}
    candidates = list(records)
    if n and n < len(candidates):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    correct = [r for r in candidates if r.rec_id not in error_ids]
    return errors, correct, report | {"n_correct_records": len(correct)}


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
        fact_concept = fact_concept_eval(model, vocab, rows, device=device, n=n,
                                         seed=seed + 3001 * i)
        latent_fact = latent_fact_concept_eval(model, vocab, rows, device=device, n=n,
                                               seed=seed + 3001 * i)
        out[name] = {"n": len(rows),
                     "n_records": teacher["n_records"],
                     "sampled": teacher["sampled"],
                     "teacher_forced_fact_value_acc": teacher["fact_value_acc"],
                     "semantic_fact_value_acc": semantic["fact_value_acc"],
                     "fact_concept_fact_value_acc": fact_concept["fact_value_acc"],
                     "latent_fact_concept_fact_value_acc": latent_fact["fact_value_acc"],
                     "latent_fact_concept_records": latent_fact["n_records"]}
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
    fact_concept = fact_concept_eval(model, vocab, records, device=device, n=fact_n,
                                     seed=seed + 11)
    latent_fact_concept = latent_fact_concept_eval(
        model, vocab, records, device=device, n=fact_n, seed=seed + 11)
    fact_concept_geometry = fact_concept_geometry_eval(
        model, vocab, records, device=device, n=fact_n, seed=seed + 12)
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
            "fact_concept_head": fact_concept,
            "latent_fact_concept_head": latent_fact_concept,
            "fact_concept_geometry": fact_concept_geometry,
            "by_kind": by_kind,
            "free_decode_by_kind": by_kind_free,
            "paraphrase_consistency": para, "counterfactual": cf,
            "nli_artifact_control": artifact,
            "gate_thresholds": {"fact_value_acc": 0.80, "free_f1": 0.80,
                                "semantic_fact_value_acc": 0.80,
                                "fact_concept_fact_value_acc": 0.80,
                                "latent_fact_concept_fact_value_acc": 0.80,
                                "fact_concept_geometry_margin": 0.05,
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
                       fact_schema=schema,
                       fact_concept_prefix=bool(
                           ckpt.get("fact_concept_prefix", False)),
                       text_encoder_arch=ckpt.get("text_encoder_arch", "transformer"),
                       text_encoder_layers=int(
                           ckpt.get("text_encoder_layers", 1)),
                       fact_concept_refine=bool(
                           ckpt.get("fact_concept_refine", False)),
                       fact_concept_refine_gate_init=float(
                           ckpt.get("fact_concept_refine_gate_init", -2.0)),
                       fact_concept_mixer_layers=int(
                           ckpt.get("fact_concept_mixer_layers", 0)),
                       fact_concept_mixer_gate_init=float(
                           ckpt.get("fact_concept_mixer_gate_init", -2.0)),
                       latent_concept_slots=int(
                           ckpt.get("latent_concept_slots", 0)),
                       latent_concept_layers=int(
                           ckpt.get("latent_concept_layers", 1)),
                       latent_concept_prefix=bool(
                           ckpt.get("latent_concept_prefix", False)),
                       latent_concept_refine=bool(
                           ckpt.get("latent_concept_refine", False)),
                       latent_concept_refine_gate_init=float(
                           ckpt.get("latent_concept_refine_gate_init", -2.0)),
                       latent_concept_memory_size=int(
                           ckpt.get("latent_concept_memory_size", 0))).to(device)
    state = ckpt["state_dict"]
    model.load_state_dict(state, strict=False)
    model.has_fact_concept_state = any(k.startswith("fact_concepts.") for k in state)
    model.eval()
    return model, vocab, ckpt


def expanded_checkpoint_model(checkpoint, records, device=DEV):
    src_model, src_vocab, ckpt = load_checkpoint(checkpoint, device=device)
    schema = build_fact_schema(records, base_schema=src_model.fact_schema)
    vocab = build_vocab(records, base_vocab=src_vocab)
    model = TextFactLM(len(vocab), d=int(ckpt.get("d", 96)),
                       layers=int(ckpt.get("layers", 3)),
                       heads=int(ckpt.get("heads", 4)), pad=vocab.pad,
                       fact_schema=schema,
                       fact_concept_prefix=bool(
                           ckpt.get("fact_concept_prefix", False)),
                       text_encoder_arch=ckpt.get("text_encoder_arch", "transformer"),
                       text_encoder_layers=int(
                           ckpt.get("text_encoder_layers", 1)),
                       fact_concept_refine=bool(
                           ckpt.get("fact_concept_refine", False)),
                       fact_concept_refine_gate_init=float(
                           ckpt.get("fact_concept_refine_gate_init", -2.0)),
                       fact_concept_mixer_layers=int(
                           ckpt.get("fact_concept_mixer_layers", 0)),
                       fact_concept_mixer_gate_init=float(
                           ckpt.get("fact_concept_mixer_gate_init", -2.0)),
                       latent_concept_slots=int(
                           ckpt.get("latent_concept_slots", 0)),
                       latent_concept_layers=int(
                           ckpt.get("latent_concept_layers", 1)),
                       latent_concept_prefix=bool(
                           ckpt.get("latent_concept_prefix", False)),
                       latent_concept_refine=bool(
                           ckpt.get("latent_concept_refine", False)),
                       latent_concept_refine_gate_init=float(
                           ckpt.get("latent_concept_refine_gate_init", -2.0)),
                       latent_concept_memory_size=int(
                           ckpt.get("latent_concept_memory_size", 0))).to(device)
    copy_pretrained_text_weights(src_model, src_vocab, model, vocab)
    model.eval()
    return model, vocab, ckpt


def checkpoint_payload(model, vocab, d, layers, heads, report):
    fact_schema = None
    if model.fact_schema is not None:
        fact_schema = {
            "keys": model.fact_schema.keys,
            "values": model.fact_schema.values,
        }
    return {"state_dict": model.state_dict(), "vocab": vocab.itos,
            "fact_concept_prefix": bool(getattr(model, "fact_concept_prefix", False)),
            "fact_concept_refine": bool(getattr(model, "fact_concept_refine", False)),
            "fact_concept_refine_gate_init": float(
                getattr(model, "fact_concept_refine_gate_init", -2.0)),
            "fact_concept_mixer_layers": int(
                getattr(model, "fact_concept_mixer_layers", 0)),
            "fact_concept_mixer_gate_init": float(
                getattr(model, "fact_concept_mixer_gate_init", -2.0)),
            "latent_concept_slots": int(getattr(model, "latent_concept_slots", 0)),
            "latent_concept_layers": int(getattr(model, "latent_concept_layers", 1)),
            "latent_concept_prefix": bool(
                getattr(model, "latent_concept_prefix", False)),
            "latent_concept_refine": bool(
                getattr(model, "latent_concept_refine", False)),
            "latent_concept_refine_gate_init": float(
                getattr(model, "latent_concept_refine_gate_init", -2.0)),
            "latent_concept_memory_size": int(
                getattr(model, "latent_concept_memory_size", 0)),
            "text_encoder_arch": getattr(model, "text_encoder_arch", "transformer"),
            "text_encoder_layers": int(getattr(model, "text_encoder_layers", 1)),
            "d": d, "layers": layers, "heads": heads, "fact_schema": fact_schema,
            "report": report}


def clone_model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_model_state(model, state, device=DEV):
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()


def _fact_value_scores(eval_report):
    scores = {
        "semantic": float(eval_report["semantic_head"]["fact_value_acc"]),
        "teacher": float(eval_report["teacher_forced"]["fact_value_acc"]),
    }
    concept = eval_report.get("fact_concept_head") or {}
    if concept.get("n_records", 0) and not concept.get("skipped"):
        scores["concept"] = float(concept["fact_value_acc"])
    latent = eval_report.get("latent_fact_concept_head") or {}
    if latent.get("n_records", 0) and not latent.get("skipped"):
        scores["latent"] = float(latent["fact_value_acc"])
    return scores


def _score_metric(eval_report, metric):
    scores = _fact_value_scores(eval_report)
    if metric in scores:
        return scores[metric]
    if metric == "both":
        return 0.5 * (scores["semantic"] + scores["teacher"])
    if metric == "min":
        return min(scores["semantic"], scores["teacher"])
    raise ValueError(f"unknown study score metric {metric!r}")


def _control_gap_values(eval_report, metric="both"):
    use_teacher = metric in ("teacher", "both", "min")
    use_semantic = metric in ("semantic", "both", "min")
    gaps = []
    nli = eval_report.get("nli_artifact_control") or {}
    if nli.get("n", 0):
        if use_teacher:
            gaps.append(("nli_full_minus_hypothesis_only",
                         float(nli.get("full_minus_hypothesis_only", 0.0))))
        if use_semantic:
            gaps.append(("nli_semantic_full_minus_hypothesis_only",
                         float(nli.get("semantic_full_minus_hypothesis_only", 0.0))))
    return gaps


def _control_gap_penalty(eval_report, metric="both", threshold=0.05):
    gaps = [gap for _name, gap in _control_gap_values(eval_report, metric=metric)]
    if not gaps:
        return 0.0
    return float(np.mean([max(0.0, threshold - gap) for gap in gaps]))


def _control_gap_failures(eval_report, metric="both", threshold=0.05):
    return {name: gap for name, gap in _control_gap_values(eval_report, metric=metric)
            if gap < threshold}


def _kind_score(row, metric):
    teacher = float(row.get("teacher_forced_fact_value_acc", 0.0))
    semantic = float(row.get("semantic_fact_value_acc", 0.0))
    latent = float(row.get("latent_fact_concept_fact_value_acc", 0.0))
    if metric == "teacher":
        return teacher
    if metric == "semantic":
        return semantic
    if metric == "latent":
        return latent if row.get("latent_fact_concept_records", 0) else 0.5 * (teacher + semantic)
    if metric == "both":
        return 0.5 * (teacher + semantic)
    if metric == "min":
        return min(teacher, semantic)
    raise ValueError(f"unknown study score metric {metric!r}")


def _kind_regressions(eval_report, ref_report, metric="both", tol=1e-9):
    if ref_report is None:
        return {}
    by_kind = eval_report.get("by_kind") or {}
    ref_by_kind = ref_report.get("by_kind") or {}
    out = {}
    for name, row in sorted(by_kind.items()):
        if name not in ref_by_kind:
            continue
        score = _kind_score(row, metric)
        ref_score = _kind_score(ref_by_kind[name], metric)
        drop = ref_score - score
        if drop > tol:
            out[name] = {"before": ref_score, "after": score, "drop": drop}
    return out


def study_selection_score(study_eval, replay_eval, replay_ref=None, metric="both",
                          retention_w=1.0, control_w=1.0, kind_w=1.0,
                          study_ref=None):
    return study_selection_components(study_eval, replay_eval, replay_ref,
                                      metric=metric,
                                      retention_w=retention_w,
                                      control_w=control_w,
                                      kind_w=kind_w,
                                      study_ref=study_ref)["score"]


def study_selection_allowed(components, require_positive=True, control_w=1.0, kind_w=1.0):
    control_failed = bool(components.get("control_failures")) and control_w > 0
    kind_regressed = bool(components.get("kind_regressions")) and kind_w > 0
    score_allowed = (((not require_positive) or components["score"] > 0.0)
                     and not control_failed and not kind_regressed)
    return {"score_allowed": bool(score_allowed),
            "control_allowed": not control_failed,
            "kind_regression_allowed": not kind_regressed}


def study_selection_components(study_eval, replay_eval, replay_ref=None, metric="both",
                               retention_w=1.0, control_w=1.0, kind_w=1.0,
                               study_ref=None):
    study_score = _score_metric(study_eval, metric)
    replay_score = None
    replay_ref_score = None
    retention_penalty = 0.0
    control_gaps = _control_gap_values(study_eval, metric=metric)
    control_gap_penalty = _control_gap_penalty(study_eval, metric=metric)
    control_failures = _control_gap_failures(study_eval, metric=metric)
    control_penalty = control_w * control_gap_penalty
    by_kind = study_eval.get("by_kind") or {}
    kind_scores = {name: _kind_score(row, metric) for name, row in sorted(by_kind.items())}
    kind_floor = min(kind_scores.values()) if len(kind_scores) >= 2 else study_score
    kind_gap_penalty = max(0.0, study_score - kind_floor) if len(kind_scores) >= 2 else 0.0
    kind_regressions = _kind_regressions(study_eval, study_ref, metric=metric)
    kind_regression_penalty = sum(row["drop"] for row in kind_regressions.values())
    kind_penalty = kind_w * (kind_gap_penalty + kind_regression_penalty)
    if replay_eval is not None and replay_ref is not None:
        replay_score = _score_metric(replay_eval, metric)
        replay_ref_score = _score_metric(replay_ref, metric)
        retention_penalty = retention_w * max(0.0, replay_ref_score - replay_score)
    scores = _fact_value_scores(study_eval)
    return {
        "metric": metric,
        "study_score": study_score,
        "study_semantic_fact_value_acc": scores["semantic"],
        "study_teacher_fact_value_acc": scores["teacher"],
        "study_latent_fact_value_acc": scores.get("latent"),
        "replay_score": replay_score,
        "replay_ref_score": replay_ref_score,
        "retention_penalty": retention_penalty,
        "control_gap_penalty": control_gap_penalty,
        "control_penalty": control_penalty,
        "control_gaps": {name: gap for name, gap in control_gaps},
        "control_failures": control_failures,
        "kind_floor": kind_floor,
        "kind_scores": kind_scores,
        "kind_gap_penalty": kind_gap_penalty,
        "kind_regressions": kind_regressions,
        "kind_regression_penalty": kind_regression_penalty,
        "kind_penalty": kind_penalty,
        "score": study_score - retention_penalty - control_penalty - kind_penalty,
    }


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
    torch.manual_seed(seed)
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
        text_encoder_arch="transformer", text_encoder_layers=1,
        fact_concept_refine=False, fact_concept_refine_gate_init=-2.0,
        fact_concept_mixer_layers=0, fact_concept_mixer_gate_init=-2.0,
        latent_concept_slots=0, latent_concept_layers=1,
        latent_concept_prefix=False,
        latent_concept_refine=False,
        latent_concept_refine_gate_init=-2.0,
        latent_concept_w=0.0, latent_concept_view_dropout=0.1,
        latent_concept_invariance_w=25.0, latent_concept_variance_w=25.0,
        latent_concept_covariance_w=1.0, latent_concept_variance_target=1.0,
        latent_concept_fact_w=0.0,
        fact_n=0, kind_fact_n=0, artifact_n=0, decode_w=1.0,
        fact_concept_w=0.0, fact_concept_contrast_w=0.0,
        fact_concept_contrast_temperature=0.1,
        fact_concept_centroid_w=0.0, fact_concept_centroid_temperature=0.1,
        fact_concept_centroid_margin=0.0, fact_concept_prefix=False,
        fact_concept_prototype_w=0.0, fact_concept_prototype_spread_w=0.0,
        fact_concept_prototype_spread_margin=0.2,
        fact_concept_state_spread_w=0.0, fact_concept_state_spread_variance=0.05,
        fact_concept_state_spread_margin=0.2,
        fact_concept_state_spread_covariance_w=0.05):
    records = load_records(data)
    model, vocab = train_model(records, steps=steps, batch=batch, d=d, layers=layers,
                               heads=heads,
                               text_encoder_arch=text_encoder_arch,
                               text_encoder_layers=text_encoder_layers,
                               fact_concept_refine=fact_concept_refine,
                               fact_concept_refine_gate_init=(
                                   fact_concept_refine_gate_init),
                               fact_concept_mixer_layers=fact_concept_mixer_layers,
                               fact_concept_mixer_gate_init=(
                                   fact_concept_mixer_gate_init),
                               latent_concept_slots=latent_concept_slots,
                               latent_concept_layers=latent_concept_layers,
                               latent_concept_prefix=latent_concept_prefix,
                               latent_concept_refine=latent_concept_refine,
                               latent_concept_refine_gate_init=(
                                   latent_concept_refine_gate_init),
                               seed=seed, device=device, semantic_w=semantic_w,
                               balance_by=balance_by, fact_concept_w=fact_concept_w,
                               fact_concept_contrast_w=fact_concept_contrast_w,
                               fact_concept_contrast_temperature=(
                                   fact_concept_contrast_temperature),
                               fact_concept_centroid_w=fact_concept_centroid_w,
                               fact_concept_centroid_temperature=(
                                   fact_concept_centroid_temperature),
                               fact_concept_centroid_margin=fact_concept_centroid_margin,
                               fact_concept_prefix=fact_concept_prefix,
                               fact_concept_prototype_w=fact_concept_prototype_w,
                               fact_concept_prototype_spread_w=(
                                   fact_concept_prototype_spread_w),
                               fact_concept_prototype_spread_margin=(
                                   fact_concept_prototype_spread_margin),
                               fact_concept_state_spread_w=fact_concept_state_spread_w,
                               fact_concept_state_spread_variance=(
                                   fact_concept_state_spread_variance),
                               fact_concept_state_spread_margin=(
                                   fact_concept_state_spread_margin),
                               fact_concept_state_spread_covariance_w=(
                                   fact_concept_state_spread_covariance_w),
                               latent_concept_w=latent_concept_w,
                               latent_concept_view_dropout=latent_concept_view_dropout,
                               latent_concept_invariance_w=latent_concept_invariance_w,
                               latent_concept_variance_w=latent_concept_variance_w,
                               latent_concept_covariance_w=latent_concept_covariance_w,
                               latent_concept_variance_target=(
                                   latent_concept_variance_target),
                               latent_concept_fact_w=latent_concept_fact_w,
                               decode_w=decode_w)
    report = {"experiment": "text0_semantic_extraction", "data": data,
              "steps": int(steps), "batch": int(batch), "d": int(d),
              "layers": int(layers), "heads": int(heads),
              "text_encoder_arch": getattr(model, "text_encoder_arch", text_encoder_arch),
              "text_encoder_layers": int(getattr(
                  model, "text_encoder_layers", text_encoder_layers)),
              "decode_w": float(decode_w), "semantic_w": float(semantic_w),
              "fact_concept_w": float(fact_concept_w),
              "fact_concept_contrast_w": float(fact_concept_contrast_w),
              "fact_concept_contrast_temperature": float(
                  fact_concept_contrast_temperature),
              "fact_concept_centroid_w": float(fact_concept_centroid_w),
              "fact_concept_centroid_temperature": float(
                  fact_concept_centroid_temperature),
              "fact_concept_centroid_margin": float(fact_concept_centroid_margin),
              "fact_concept_prefix": bool(getattr(model, "fact_concept_prefix", False)),
              "fact_concept_refine": bool(getattr(model, "fact_concept_refine", False)),
              "fact_concept_refine_gate_init": float(
                  getattr(model, "fact_concept_refine_gate_init", -2.0)),
              "fact_concept_mixer_layers": int(
                  getattr(model, "fact_concept_mixer_layers", 0)),
              "fact_concept_mixer_gate_init": float(
                  getattr(model, "fact_concept_mixer_gate_init", -2.0)),
              "latent_concept_slots": int(getattr(model, "latent_concept_slots", 0)),
              "latent_concept_layers": int(getattr(model, "latent_concept_layers", 1)),
              "latent_concept_prefix": bool(
                  getattr(model, "latent_concept_prefix", False)),
              "latent_concept_refine": bool(
                  getattr(model, "latent_concept_refine", False)),
              "latent_concept_refine_gate_init": float(
                  getattr(model, "latent_concept_refine_gate_init", -2.0)),
              "latent_concept_w": float(latent_concept_w),
              "latent_concept_view_dropout": float(latent_concept_view_dropout),
              "latent_concept_invariance_w": float(latent_concept_invariance_w),
              "latent_concept_variance_w": float(latent_concept_variance_w),
              "latent_concept_covariance_w": float(latent_concept_covariance_w),
              "latent_concept_variance_target": float(latent_concept_variance_target),
              "latent_concept_fact_w": float(latent_concept_fact_w),
              "fact_concept_prototype_w": float(fact_concept_prototype_w),
              "fact_concept_prototype_spread_w": float(
                  fact_concept_prototype_spread_w),
              "fact_concept_prototype_spread_margin": float(
                  fact_concept_prototype_spread_margin),
              "fact_concept_state_spread_w": float(fact_concept_state_spread_w),
              "fact_concept_state_spread_variance": float(
                  fact_concept_state_spread_variance),
              "fact_concept_state_spread_margin": float(
                  fact_concept_state_spread_margin),
              "fact_concept_state_spread_covariance_w": float(
                  fact_concept_state_spread_covariance_w),
              "free_n": int(free_n), "paraphrase_n": int(paraphrase_n),
              "counterfactual_n": int(counterfactual_n),
              "kind_free_n": int(kind_free_n), "fact_n": int(fact_n),
              "kind_fact_n": int(kind_fact_n), "artifact_n": int(artifact_n),
              "balance_by": balance_by,
              "train_records": sum(r.split == "train" for r in records),
              "eval_records": sum(r.split == "eval" for r in records)}
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        torch.save(checkpoint_payload(model, vocab, d, layers, heads,
                                      report | {"status": "trained_pending_eval"}),
                   checkpoint)
    report.update(evaluate_all(model, vocab, records, device=device, max_new=max_new,
                               free_n=free_n, paraphrase_n=paraphrase_n,
                               counterfactual_n=counterfactual_n,
                               kind_free_n=kind_free_n, fact_n=fact_n,
                               kind_fact_n=kind_fact_n, artifact_n=artifact_n,
                               seed=seed + 17))
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        torch.save(checkpoint_payload(model, vocab, d, layers, heads, report), checkpoint)
        report["checkpoint"] = checkpoint
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    return report


def study_checkpoint(checkpoint, data, out_checkpoint=None, replay_data=None, out=None,
                     steps=400, batch=32, lr=5e-4, seed=0, device=DEV,
                     max_new=160, semantic_w=0.5, free_n=0, paraphrase_n=0,
                     counterfactual_n=0, kind_free_n=0, balance_by="none",
                     fact_n=0, kind_fact_n=0, artifact_n=0, decode_w=1.0,
                     fact_concept_w=0.0, fact_concept_contrast_w=0.0,
                     fact_concept_contrast_temperature=0.1,
                     fact_concept_centroid_w=0.0,
                     fact_concept_centroid_temperature=0.1,
                     fact_concept_centroid_margin=0.0,
                     fact_concept_prototype_w=0.0,
                     fact_concept_prototype_spread_w=0.0,
                     fact_concept_prototype_spread_margin=0.2,
                     fact_concept_state_spread_w=0.0,
                     fact_concept_state_spread_variance=0.05,
                     fact_concept_state_spread_margin=0.2,
                     fact_concept_state_spread_covariance_w=0.05,
                     latent_concept_w=0.0, latent_concept_view_dropout=0.1,
                     latent_concept_invariance_w=25.0,
                     latent_concept_variance_w=25.0,
                     latent_concept_covariance_w=1.0,
                     latent_concept_variance_target=1.0,
                     latent_concept_fact_w=0.0,
                     study_rounds=1, study_strategy="errors", study_probe_n=0,
                     study_hard_max=0, study_select_best=False,
                     study_score_metric="both", study_retention_w=1.0,
                     study_control_w=1.0, study_kind_w=1.0,
                     study_require_positive_score=True,
                     study_confirm_n=0, study_confirm_seed_stride=10000):
    if study_strategy not in ("errors", "latent", "all"):
        raise ValueError(f"unknown study strategy {study_strategy!r}")
    records = load_records(data)
    replay_records = (load_records(replay_data, require_train=False, require_eval=True)
                      if replay_data else [])
    model, vocab, ckpt = expanded_checkpoint_model(
        checkpoint, records + replay_records, device=device)
    d = int(ckpt.get("d", 96))
    layers = int(ckpt.get("layers", 3))
    heads = int(ckpt.get("heads", 4))
    eval_kwargs = dict(device=device, max_new=max_new, free_n=free_n,
                       paraphrase_n=paraphrase_n,
                       counterfactual_n=counterfactual_n,
                       kind_free_n=kind_free_n, fact_n=fact_n,
                       kind_fact_n=kind_fact_n, artifact_n=artifact_n,
                       seed=seed + 17)
    before = evaluate_all(model, vocab, records, **eval_kwargs)
    replay_before = (evaluate_all(model, vocab, replay_records, **eval_kwargs)
                     if replay_records and any(r.split == "eval" for r in replay_records)
                     else None)
    report = {"experiment": "text0_checkpoint_study", "checkpoint": checkpoint,
              "checkpoint_experiment": ckpt.get("report", {}).get("experiment"),
              "data": data, "replay_data": replay_data or [],
              "steps": int(steps), "batch": int(batch), "lr": float(lr),
              "study_rounds": int(study_rounds), "study_strategy": study_strategy,
              "study_probe_n": int(study_probe_n),
              "study_hard_max": int(study_hard_max),
              "study_select_best": bool(study_select_best),
              "study_score_metric": study_score_metric,
              "study_retention_w": float(study_retention_w),
              "study_control_w": float(study_control_w),
              "study_kind_w": float(study_kind_w),
              "decode_w": float(decode_w), "semantic_w": float(semantic_w),
              "fact_concept_w": float(fact_concept_w),
              "fact_concept_contrast_w": float(fact_concept_contrast_w),
              "fact_concept_centroid_w": float(fact_concept_centroid_w),
              "latent_concept_w": float(latent_concept_w),
              "latent_concept_fact_w": float(latent_concept_fact_w),
              "train_records": sum(r.split == "train" for r in records),
              "eval_records": sum(r.split == "eval" for r in records),
              "replay_train_records": sum(r.split == "train" for r in replay_records),
              "replay_eval_records": sum(r.split == "eval" for r in replay_records),
              "before": before, "replay_before": replay_before}
    train_records = [r for r in records if r.split == "train"]
    replay_train = [r for r in replay_records if r.split == "train"]
    if not train_records:
        raise ValueError("checkpoint study requires train records")
    rng = np.random.default_rng(seed)
    best_state = clone_model_state(model)
    best_score = -float("inf")
    best_round = 0
    best_components = None
    best_study_eval = before
    best_replay_eval = replay_before
    round_reports = []
    for round_i in range(max(1, int(study_rounds))):
        round_seed = seed + 1009 * (round_i + 1)
        probe_n = int(study_probe_n) if int(study_probe_n) > 0 else max(batch * 4, 1)
        if study_strategy == "all":
            hard_records = list(train_records)
            hard_report = {"strategy": "all", "n_records": len(train_records),
                           "n_error_records": None}
        elif study_strategy == "latent":
            hard_records, _correct, hard_report = latent_fact_record_outcomes(
                model, vocab, train_records, device=device, n=probe_n,
                seed=round_seed)
            hard_report = hard_report | {"strategy": "latent"}
        else:
            hard_records, _correct, hard_report = semantic_record_outcomes(
                model, vocab, train_records, device=device, n=probe_n,
                seed=round_seed)
            hard_report = hard_report | {"strategy": "errors"}
        if study_hard_max and len(hard_records) > int(study_hard_max):
            idx = rng.choice(len(hard_records), size=int(study_hard_max), replace=False)
            hard_records = [hard_records[int(i)] for i in idx]
            hard_report = hard_report | {"capped": True}
        else:
            hard_report = hard_report | {"capped": False}
        if not hard_records:
            hard_records = list(train_records)
            hard_report = hard_report | {"fallback": "all_train"}
        round_fit_records = hard_records + replay_train
        round_reports.append({"round": round_i + 1,
                              "fit_records": len(round_fit_records),
                              "study_fit_records": len(hard_records),
                              "replay_fit_records": len(replay_train),
                              "hard_examples": hard_report | {
                                  "n_error_records_used": len(hard_records)}})
        fit_model(model, vocab, round_fit_records, steps=steps, batch=batch, lr=lr,
                  seed=round_seed, device=device, semantic_w=semantic_w,
                  balance_by=balance_by, fact_concept_w=fact_concept_w,
                  fact_concept_contrast_w=fact_concept_contrast_w,
                  fact_concept_contrast_temperature=(
                      fact_concept_contrast_temperature),
                  fact_concept_centroid_w=fact_concept_centroid_w,
                  fact_concept_centroid_temperature=(
                      fact_concept_centroid_temperature),
                  fact_concept_centroid_margin=fact_concept_centroid_margin,
                  fact_concept_prototype_w=fact_concept_prototype_w,
                  fact_concept_prototype_spread_w=fact_concept_prototype_spread_w,
                  fact_concept_prototype_spread_margin=(
                      fact_concept_prototype_spread_margin),
                  fact_concept_state_spread_w=fact_concept_state_spread_w,
                  fact_concept_state_spread_variance=(
                      fact_concept_state_spread_variance),
                  fact_concept_state_spread_margin=fact_concept_state_spread_margin,
                  fact_concept_state_spread_covariance_w=(
                      fact_concept_state_spread_covariance_w),
                  latent_concept_w=latent_concept_w,
                  latent_concept_view_dropout=latent_concept_view_dropout,
                  latent_concept_invariance_w=latent_concept_invariance_w,
                  latent_concept_variance_w=latent_concept_variance_w,
                  latent_concept_covariance_w=latent_concept_covariance_w,
                  latent_concept_variance_target=latent_concept_variance_target,
                  latent_concept_fact_w=latent_concept_fact_w,
                  prefix=f"study-r{round_i + 1}", decode_w=decode_w)
        round_study_eval = evaluate_all(model, vocab, records, **eval_kwargs)
        round_replay_eval = (evaluate_all(model, vocab, replay_records, **eval_kwargs)
                             if replay_records and any(r.split == "eval"
                                                       for r in replay_records)
                             else None)
        components = study_selection_components(
            round_study_eval, round_replay_eval, replay_before,
            metric=study_score_metric, retention_w=study_retention_w,
            control_w=study_control_w, kind_w=study_kind_w, study_ref=before)
        allowed = study_selection_allowed(
            components, require_positive=study_require_positive_score,
            control_w=study_control_w, kind_w=study_kind_w)
        confirmation_checks = []
        for confirm_i in range(int(study_confirm_n)):
            confirm_kwargs = eval_kwargs | {
                "seed": seed + int(study_confirm_seed_stride) * (confirm_i + 1)}
            confirm_eval = evaluate_all(model, vocab, records, **confirm_kwargs)
            confirm_replay = (evaluate_all(model, vocab, replay_records, **confirm_kwargs)
                              if replay_records and any(r.split == "eval"
                                                        for r in replay_records)
                              else None)
            confirm_components = study_selection_components(
                confirm_eval, confirm_replay, replay_before,
                metric=study_score_metric, retention_w=study_retention_w,
                control_w=study_control_w, kind_w=study_kind_w, study_ref=before)
            confirmation_checks.append({
                "index": confirm_i,
                "score": confirm_components["score"],
                "score_components": confirm_components,
                **study_selection_allowed(confirm_components,
                                          require_positive=study_require_positive_score,
                                          control_w=study_control_w,
                                          kind_w=study_kind_w)})
        confirmation_allowed = all(row["score_allowed"] for row in confirmation_checks)
        round_allowed = allowed["score_allowed"] and confirmation_allowed
        score = float(components["score"])
        round_reports[-1].update({"score": score, "score_components": components,
                                  "confirmation": confirmation_checks,
                                  "confirmation_allowed": bool(confirmation_allowed),
                                  **allowed, "round_allowed": bool(round_allowed)})
        if (not study_select_best) or (round_allowed and score >= best_score):
            best_score = score
            best_round = round_i + 1
            best_components = components
            best_state = clone_model_state(model)
            best_study_eval = round_study_eval
            best_replay_eval = round_replay_eval
    if study_select_best:
        restore_model_state(model, best_state, device=device)
    after = best_study_eval
    replay_after = best_replay_eval
    report["after"] = after
    report["replay_after"] = replay_after
    report["rounds"] = round_reports
    report["selected_round"] = best_round
    report["selected_score"] = best_score if best_score != -float("inf") else None
    report["selected_score_components"] = best_components
    report["delta"] = {
        "teacher_forced_fact_value_acc": (
            after["teacher_forced"]["fact_value_acc"]
            - before["teacher_forced"]["fact_value_acc"]),
        "semantic_fact_value_acc": (
            after["semantic_head"]["fact_value_acc"]
            - before["semantic_head"]["fact_value_acc"]),
        "latent_fact_concept_fact_value_acc": (
            after["latent_fact_concept_head"]["fact_value_acc"]
            - before["latent_fact_concept_head"]["fact_value_acc"]),
        "free_f1": after["free_decode"]["f1"] - before["free_decode"]["f1"],
    }
    report["replay_delta"] = ({
        "teacher_forced_fact_value_acc": (
            replay_after["teacher_forced"]["fact_value_acc"]
            - replay_before["teacher_forced"]["fact_value_acc"]),
        "semantic_fact_value_acc": (
            replay_after["semantic_head"]["fact_value_acc"]
            - replay_before["semantic_head"]["fact_value_acc"]),
        "latent_fact_concept_fact_value_acc": (
            replay_after["latent_fact_concept_head"]["fact_value_acc"]
            - replay_before["latent_fact_concept_head"]["fact_value_acc"]),
        "free_f1": replay_after["free_decode"]["f1"] - replay_before["free_decode"]["f1"],
    } if replay_before is not None and replay_after is not None else None)
    report["out_checkpoint"] = out_checkpoint
    if out_checkpoint:
        os.makedirs(os.path.dirname(out_checkpoint) or ".", exist_ok=True)
        torch.save(checkpoint_payload(model, vocab, d, layers, heads, report),
                   out_checkpoint)
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
                 "ln_subject/object_swap\ttemp1\n")
    hans, seen = _hans_records_from_text(hans_text, "eval", 0, np.random.default_rng(0),
                                         "snli")
    assert seen == 1 and hans[0]["facts"] == [["pair0", "nli", "neutral"]]
    vocab = build_vocab(records)
    schema = build_fact_schema(records)
    model = TextFactLM(len(vocab), d=32, layers=1, heads=4, pad=vocab.pad,
                       fact_schema=schema, fact_concept_prefix=True,
                       fact_concept_refine=True, fact_concept_mixer_layers=1,
                       latent_concept_slots=3, latent_concept_layers=1,
                       latent_concept_prefix=True,
                       latent_concept_refine=True).to("cpu")
    txt, ids = pack(records[:2], vocab, "cpu")
    logits = model(txt, ids)
    assert logits.shape[:2] == ids.shape
    assert torch.isfinite(token_loss(logits, ids, pad=vocab.pad))
    assert torch.isfinite(semantic_loss(model, txt, records[:2], schema))
    assert torch.isfinite(fact_concept_loss(model, txt, records[:2], schema))
    assert torch.isfinite(fact_concept_contrastive_loss(model, txt, records[:2], schema))
    assert torch.isfinite(fact_concept_batch_centroid_loss(model, txt, records[:2], schema))
    assert torch.isfinite(fact_concept_prototype_loss(model, txt, records[:2], schema))
    assert torch.isfinite(fact_concept_prototype_spread_loss(model))
    assert torch.isfinite(fact_concept_state_spread_loss(model, txt, records[:2], schema))
    assert torch.isfinite(latent_text_concept_loss(model, txt, view_dropout=0.1))
    assert torch.isfinite(latent_fact_concept_loss(model, txt, records[:2], schema))
    rel_model = TextFactLM(len(vocab), d=32, layers=1, heads=4, pad=vocab.pad,
                           fact_schema=schema, text_encoder_arch="relational",
                           text_encoder_layers=1, latent_concept_slots=2).to("cpu")
    rel_logits = rel_model(txt, ids)
    assert rel_logits.shape[:2] == ids.shape
    fit_model(model, vocab, records[:2], steps=1, batch=2, lr=1e-4, seed=3,
              device="cpu", semantic_w=1.0, decode_w=0.0,
              fact_concept_w=0.1, latent_concept_w=0.1,
              latent_concept_fact_w=0.1, prefix="selftest")
    report = evaluate_all(model, vocab, records, device="cpu", max_new=12,
                          free_n=-1, paraphrase_n=-1, counterfactual_n=-1)
    assert set(report) >= {"teacher_forced", "free_decode", "semantic_head",
                           "fact_concept_head", "latent_fact_concept_head",
                           "fact_concept_geometry", "by_kind", "gate"}
    payload = checkpoint_payload(model, vocab, 32, 1, 4, {"experiment": "selftest"})
    score_a = {"semantic_head": {"fact_value_acc": 0.8},
               "teacher_forced": {"fact_value_acc": 0.7},
               "latent_fact_concept_head": {"fact_value_acc": 0.65,
                                            "n_records": 2}}
    score_b = {"semantic_head": {"fact_value_acc": 0.6},
               "teacher_forced": {"fact_value_acc": 0.7},
               "latent_fact_concept_head": {"fact_value_acc": 0.55,
                                            "n_records": 2}}
    assert study_selection_score(score_a, score_b, score_a, retention_w=1.0) < 0.8
    assert abs(_score_metric(score_a, "both") - 0.75) < 1e-6
    assert abs(_score_metric(score_a, "min") - 0.7) < 1e-6
    assert abs(_score_metric(score_a, "latent") - 0.65) < 1e-6
    assert study_selection_allowed({"score": -0.1}, require_positive=False)[
        "score_allowed"]
    reading_records = [
        ReadingRecord("read-train-1", "train",
                      tuple(split_words("language concepts connect across repeated views")),
                      meta={"source": "selftest-train", "chunk_index": 0}),
        ReadingRecord("read-train-2", "train",
                      tuple(split_words("models can revise concepts from reading")),
                      meta={"source": "selftest-train", "chunk_index": 1}),
        ReadingRecord("read-train-3", "train",
                      tuple(split_words("later chunks predict the next idea")),
                      meta={"source": "selftest-train", "chunk_index": 2}),
        ReadingRecord("read-eval-1", "eval",
                      tuple(split_words("reading updates preserve concept identity")),
                      meta={"source": "selftest-eval", "chunk_index": 0}),
        ReadingRecord("read-eval-2", "eval",
                      tuple(split_words("self teaching compares two views of text")),
                      meta={"source": "selftest-eval", "chunk_index": 1}),
        ReadingRecord("read-eval-3", "eval",
                      tuple(split_words("sequence learning connects adjacent meaning")),
                      meta={"source": "selftest-eval", "chunk_index": 2}),
    ]
    reading_vocab = build_reading_vocab(reading_records)
    reading_model = TextFactLM(
        len(reading_vocab), d=32, layers=1, heads=4, pad=reading_vocab.pad,
        fact_schema=None, latent_concept_slots=2,
        latent_concept_memory_size=8).to("cpu")
    memoryless_reading_model = TextFactLM(
        len(reading_vocab), d=32, layers=1, heads=4, pad=reading_vocab.pad,
        fact_schema=None, latent_concept_slots=2,
        latent_concept_memory_size=0).to("cpu")
    assert resolve_reading_study_strategy("auto", reading_model) == "discovery"
    assert (resolve_reading_study_strategy("auto", memoryless_reading_model)
            == "errors")
    reading_txt = pack_reading(reading_records[:2], reading_vocab, "cpu")
    assert torch.isfinite(reading_latent_view_loss(
        reading_model, reading_txt, reading_vocab.pad, reading_vocab.unk,
        token_drop_p=0.1, token_replace_p=0.0))
    assert torch.isfinite(reading_latent_factorization_loss(
        reading_model, reading_txt, feature_dropout=0.1))
    fer_loss, fer_metrics = reading_latent_fer_loss(
        reading_model, reading_txt, feature_dropout=0.1)
    assert torch.isfinite(fer_loss)
    assert torch.isfinite(fer_metrics["fer_score"])
    cold_bridge_eval = reading_latent_bridge_eval(
        reading_model, reading_vocab, reading_records, device="cpu", n=0)
    assert cold_bridge_eval["skipped"] is True
    assert cold_bridge_eval["graph_ready"] is False
    updates = update_reading_latent_memory(reading_model, reading_txt,
                                           relation_decay=0.5)
    assert updates > 0
    assert int(reading_model.latent_concept_memory.relation_updates.item()) > 0
    transition_updates = update_reading_latent_transitions(
        reading_model, reading_txt, reading_vocab.pad, context_keep_p=0.5,
        decay=0.5)
    assert transition_updates > 0
    assert int(reading_model.latent_concept_memory.transition_updates.item()) > 0
    seq_pairs, seq_report = mine_reading_sequence_pairs(
        reading_records, split="train")
    assert seq_report["n_pairs"] == 2 and seq_report["skipped"] is False
    assert torch.isfinite(reading_sequence_prediction_loss(
        reading_model, seq_pairs, reading_vocab, device="cpu",
        token_drop_p=0.1, token_replace_p=0.0))
    seq_updates = update_reading_sequence_transitions(
        reading_model, seq_pairs, reading_vocab, device="cpu", decay=0.5)
    assert seq_updates > 0
    assert torch.isfinite(reading_latent_memory_loss(
        reading_model, reading_txt, feature_dropout=0.1))
    assert torch.isfinite(reading_latent_association_loss(
        reading_model, reading_txt, feature_dropout=0.1,
        transitive_steps=2, transitive_w=0.1))
    assert torch.isfinite(reading_latent_composition_loss(
        reading_model, reading_txt, feature_dropout=0.1,
        transitive_steps=2, transitive_w=0.1))
    assert torch.isfinite(reading_latent_bridge_loss(
        reading_model, reading_txt, feature_dropout=0.1))
    assert torch.isfinite(reading_context_graph_prediction_loss(
        reading_model, reading_txt, reading_vocab.pad, context_keep_p=0.5,
        feature_dropout=0.1, transitive_steps=2, transitive_w=0.1))
    assert torch.isfinite(reading_context_graph_cycle_loss(
        reading_model, reading_txt, reading_vocab.pad, context_keep_p=0.5,
        feature_dropout=0.1, transitive_steps=2, transitive_w=0.1,
        cycle_w=0.5))
    graph_records, graph_report = reading_latent_graph_prediction_records(
        reading_model, reading_vocab, reading_records, device="cpu", n=0,
        context_keep_p=0.5, transitive_steps=2, transitive_w=0.1)
    assert graph_records and graph_report["skipped"] is False
    cycle_records, cycle_report = reading_latent_graph_cycle_records(
        reading_model, reading_vocab, reading_records, device="cpu", n=0,
        context_keep_p=0.5, transitive_steps=2, transitive_w=0.1)
    assert cycle_records and cycle_report["skipped"] is False
    discovery_records, discovery_report = reading_latent_discovery_records(
        reading_model, reading_vocab, reading_records, device="cpu", n=0,
        context_keep_p=0.5, curiosity_transitive_steps=2,
        curiosity_transitive_w=0.1, graph_transitive_steps=2,
        graph_transitive_w=0.1, cycle_transitive_steps=2,
        cycle_transitive_w=0.1)
    assert discovery_records and discovery_report["skipped"] is False
    assert math.isfinite(discovery_report["mean_score"])
    assert "mean_slot_disorder" in discovery_report
    assert "mean_bridge_score" in discovery_report
    bridge_eval = reading_latent_bridge_eval(
        reading_model, reading_vocab, reading_records, device="cpu", n=0)
    assert bridge_eval["skipped"] is False
    assert math.isfinite(bridge_eval["mean_bridge_score"])
    pairs, pair_report = mine_reading_latent_neighbors(
        reading_model, reading_vocab, reading_records, device="cpu", n=0, seed=0)
    assert pair_report["n_pairs"] == 3 and pairs
    clusters, cluster_report = mine_reading_latent_clusters(
        reading_model, reading_vocab, reading_records, device="cpu", n=0, seed=0)
    assert cluster_report["n_clusters"] >= 1 and clusters
    fer_eval = reading_fer_eval(
        reading_model, reading_vocab, reading_records, device="cpu", n=0)
    assert fer_eval["skipped"] is False
    assert math.isfinite(fer_eval["fer_score"])
    fer_bundle = reading_eval_bundle(
        reading_model, reading_vocab, reading_records, device="cpu", eval_n=0,
        score_metric="fer")
    assert fer_bundle["score_components"]["metric"] == "fer"
    assert fer_bundle["score_components"]["fer_skipped"] is False
    assert math.isfinite(fer_bundle["score_components"]["score"])
    bridge_bundle = reading_eval_bundle(
        reading_model, reading_vocab, reading_records, device="cpu", eval_n=0,
        score_metric="bridge")
    assert bridge_bundle["score_components"]["metric"] == "bridge"
    assert bridge_bundle["score_components"]["bridge_skipped"] is False
    assert math.isfinite(bridge_bundle["score_components"]["bridge_score"])
    sequence_bundle = reading_eval_bundle(
        reading_model, reading_vocab, reading_records, device="cpu", eval_n=0,
        score_metric="sequence")
    assert sequence_bundle["score_components"]["metric"] == "sequence"
    assert sequence_bundle["score_components"]["sequence_skipped"] is False
    assert math.isfinite(sequence_bundle["score_components"]["sequence_score"])
    mastery_bundle = reading_eval_bundle(
        reading_model, reading_vocab, reading_records, device="cpu", eval_n=0,
        score_metric="mastery")
    assert mastery_bundle["score_components"]["metric"] == "mastery"
    assert ("mastery_score" in mastery_bundle["score_components"]
            and "signal_coverage" in mastery_bundle["score_components"]
            and "sequence_score" in mastery_bundle["score_components"]
            and "bridge_score" in mastery_bundle["score_components"])
    assert mastery_bundle["score_components"]["bridge_skipped"] is False
    assert (mastery_bundle["score_components"]["score"]
            == mastery_bundle["score_components"]["mastery_score"])
    fit_reading_concepts(
        reading_model, reading_vocab, reading_records, steps=3, batch=2, lr=1e-4,
        seed=5, device="cpu", log_every=1, token_drop_p=0.1,
        token_replace_p=0.0, study_strategy="discovery", study_probe_n=4,
        study_hard_max=2, study_refresh_steps=1, context_target_w=0.1,
        context_keep_p=0.5, memory_size=8, composition_w=0.1, graph_predict_w=0.1,
        graph_cycle_w=0.1, bridge_w=0.1, fer_w=0.1,
        sequence_w=0.1, sequence_batch=2, sequence_temperature=0.1,
        neighborhood_w=0.1, neighborhood_batch=2, neighborhood_probe_n=2,
        transition_w=0.1, transition_batch=2,
        cluster_w=0.1, cluster_batch=4, cluster_probe_n=4)
    assert reading_model.reading_train_metrics["memory_active"] > 0
    assert (reading_model.reading_train_metrics["study_strategy_requested"]
            == "discovery")
    assert reading_model.reading_train_metrics["study_strategy"] == "discovery"
    assert reading_model.reading_train_metrics["graph_predict_w"] == 0.1
    assert reading_model.reading_train_metrics["graph_cycle_w"] == 0.1
    assert reading_model.reading_train_metrics["bridge_w"] == 0.1
    assert reading_model.reading_train_metrics["sequence_w"] == 0.1
    assert reading_model.reading_train_metrics["sequence_pairs"] == 2
    assert math.isfinite(reading_model.reading_train_metrics["sequence_loss"])
    assert (reading_model.reading_train_metrics[
        "sequence_transition_last_batch_updates"] > 0)
    assert reading_model.reading_train_metrics["fer_w"] == 0.1
    assert math.isfinite(reading_model.reading_train_metrics["fer_score"])
    assert math.isfinite(reading_model.reading_train_metrics["graph_cycle_loss"])
    assert math.isfinite(reading_model.reading_train_metrics["bridge_loss"])
    assert reading_model.reading_train_metrics["graph_transition_updates"] > 0
    assert any(r.get("strategy") == "discovery"
               for r in reading_model.reading_study_reports)
    scored = [r.get("mean_score", 0.0) for r in reading_model.reading_study_reports
              if r.get("strategy") == "discovery" and not r.get("skipped")]
    assert scored and max(scored) > 0.0
    assert any("mean_reverse_kl" in r for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    assert any("mean_bridge_score" in r for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    assert any("pool_bridge_before" in r for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    insight = reading_model.reading_train_metrics["study_pool_insight"]
    assert "bridge_score_reduction" in insight
    assert math.isfinite(insight["bridge_connectivity_gain"])
    assert (reading_model.reading_train_metrics[
        "study_pool_bridge_before"].get("graph_source") == "snapshot")
    assert "study_pool_bridge_after_current" in reading_model.reading_train_metrics
    patience_model = TextFactLM(
        len(reading_vocab), d=32, layers=1, heads=4, pad=reading_vocab.pad,
        fact_schema=None, latent_concept_slots=2,
        latent_concept_memory_size=8).to("cpu")
    _pm, _pv, patience_selection = fit_reading_concepts_select_best(
        patience_model, reading_vocab, reading_records, steps=2, rounds=2,
        batch=2, lr=1e-4, seed=7, device="cpu", log_every=10,
        token_drop_p=0.1, token_replace_p=0.0, study_strategy="auto",
        study_probe_n=2, study_hard_max=1, study_refresh_steps=1,
        memory_size=8, bridge_w=0.1, eval_n=0, score_min_delta=999.0,
        score_patience=1)
    assert patience_selection["bridge_insight_gate"] is True
    assert patience_selection["stopped_early"] is True
    assert patience_selection["rounds_run"] == 1
    assert patience_selection["stop_round"] == 1
    assert patience_selection["accepted_update"] is False
    assert "score_delta_from_best" in patience_selection["rounds"][1]
    assert "selected_insight" in patience_selection
    reading_payload = checkpoint_payload(reading_model, reading_vocab, 32, 1, 4,
                                         {"experiment": "reading-selftest"})
    assert reading_payload["fact_schema"] is None
    assert reading_payload["latent_concept_memory_size"] == 8
    print("text selftest OK")


def _add_reading_args(ap):
    ap.add_argument("--reading-data", action="append",
                    help=("raw reading JSON/JSONL/TXT corpus; trains schema-free latent "
                          "concepts without fact labels"))
    ap.add_argument("--reading-checkpoint", default=None,
                    help="existing text checkpoint to continue with raw reading data")
    ap.add_argument("--reading-out-checkpoint", default=None,
                    help="where to save the raw-reading studied checkpoint")
    ap.add_argument("--reading-replay-data", action="append",
                    help="raw reading replay corpus used during checkpoint continuation")
    ap.add_argument("--reading-replay-w", type=float, default=0.0)
    ap.add_argument("--reading-replay-batch", type=int, default=0)
    ap.add_argument("--reading-replay-retention-w", type=float, default=0.0)
    ap.add_argument("--reading-text-field", default="text")
    ap.add_argument("--reading-max-tokens", type=int, default=128)
    ap.add_argument("--reading-min-tokens", type=int, default=8)
    ap.add_argument("--reading-eval-frac", type=float, default=0.10)
    ap.add_argument("--reading-eval-n", type=int, default=64)
    ap.add_argument("--reading-lr", type=float, default=1e-3)
    ap.add_argument("--reading-token-drop", type=float, default=0.15)
    ap.add_argument("--reading-token-replace", type=float, default=0.05)
    ap.add_argument("--reading-feature-dropout", type=float, default=0.1)
    ap.add_argument("--reading-context-target-w", type=float, default=0.1)
    ap.add_argument("--reading-context-keep-p", type=float, default=0.5)
    ap.add_argument("--reading-context-target-temperature", type=float, default=0.1)
    ap.add_argument("--reading-sequence-w", type=float, default=0.05)
    ap.add_argument("--reading-sequence-batch", type=int, default=0)
    ap.add_argument("--reading-sequence-temperature", type=float, default=0.1)
    ap.add_argument("--reading-factorization-w", type=float, default=0.05)
    ap.add_argument("--reading-factorization-variance", type=float, default=0.05)
    ap.add_argument("--reading-factorization-margin", type=float, default=0.2)
    ap.add_argument("--reading-factorization-covariance-w", type=float, default=0.05)
    ap.add_argument("--reading-fer-w", type=float, default=0.0)
    ap.add_argument("--reading-fer-fragmentation-w", type=float, default=1.0)
    ap.add_argument("--reading-fer-correlation-w", type=float, default=1.0)
    ap.add_argument("--reading-fer-balance-w", type=float, default=0.1)
    ap.add_argument("--reading-memory-w", type=float, default=0.05)
    ap.add_argument("--reading-memory-size", type=int, default=64)
    ap.add_argument("--reading-memory-temperature", type=float, default=0.1)
    ap.add_argument("--reading-memory-momentum", type=float, default=0.95)
    ap.add_argument("--reading-memory-balance-w", type=float, default=0.01)
    ap.add_argument("--reading-association-w", type=float, default=0.05)
    ap.add_argument("--reading-association-temperature", type=float, default=0.1)
    ap.add_argument("--reading-association-decay", type=float, default=0.99)
    ap.add_argument("--reading-association-target-power", type=float, default=1.0)
    ap.add_argument("--reading-association-self-loop-w", type=float, default=0.05)
    ap.add_argument("--reading-association-transitive-steps", type=int, default=2)
    ap.add_argument("--reading-association-transitive-w", type=float, default=0.1)
    ap.add_argument("--reading-composition-w", type=float, default=0.0)
    ap.add_argument("--reading-composition-temperature", type=float, default=0.1)
    ap.add_argument("--reading-composition-self-loop-w", type=float, default=0.0)
    ap.add_argument("--reading-composition-transitive-steps", type=int, default=2)
    ap.add_argument("--reading-composition-transitive-w", type=float, default=0.1)
    ap.add_argument("--reading-composition-margin", type=float, default=0.0)
    ap.add_argument("--reading-graph-predict-w", type=float, default=0.0)
    ap.add_argument("--reading-graph-predict-temperature", type=float, default=0.1)
    ap.add_argument("--reading-graph-predict-self-loop-w", type=float, default=0.05)
    ap.add_argument("--reading-graph-predict-transitive-steps", type=int, default=2)
    ap.add_argument("--reading-graph-predict-transitive-w", type=float, default=0.1)
    ap.add_argument("--reading-graph-predict-target-power", type=float, default=1.0)
    ap.add_argument("--reading-graph-cycle-w", type=float, default=0.0)
    ap.add_argument("--reading-graph-cycle-temperature", type=float, default=0.1)
    ap.add_argument("--reading-graph-cycle-self-loop-w", type=float, default=0.05)
    ap.add_argument("--reading-graph-cycle-transitive-steps", type=int, default=2)
    ap.add_argument("--reading-graph-cycle-transitive-w", type=float, default=0.1)
    ap.add_argument("--reading-graph-cycle-target-power", type=float, default=1.0)
    ap.add_argument("--reading-graph-cycle-consistency-w", type=float, default=0.5)
    ap.add_argument("--reading-bridge-w", type=float, default=0.0)
    ap.add_argument("--reading-neighborhood-w", type=float, default=0.0)
    ap.add_argument("--reading-neighborhood-batch", type=int, default=0)
    ap.add_argument("--reading-neighborhood-probe-n", type=int, default=0)
    ap.add_argument("--reading-neighborhood-refresh-steps", type=int, default=0)
    ap.add_argument("--reading-neighborhood-temperature", type=float, default=0.1)
    ap.add_argument("--reading-neighborhood-margin", type=float, default=0.0)
    ap.add_argument("--reading-transition-w", type=float, default=0.05)
    ap.add_argument("--reading-transition-batch", type=int, default=0)
    ap.add_argument("--reading-transition-temperature", type=float, default=0.1)
    ap.add_argument("--reading-transition-margin", type=float, default=0.0)
    ap.add_argument("--reading-cluster-w", type=float, default=0.0)
    ap.add_argument("--reading-cluster-batch", type=int, default=0)
    ap.add_argument("--reading-cluster-probe-n", type=int, default=0)
    ap.add_argument("--reading-cluster-refresh-steps", type=int, default=0)
    ap.add_argument("--reading-cluster-temperature", type=float, default=0.1)
    ap.add_argument("--reading-cluster-margin", type=float, default=0.0)
    ap.add_argument("--reading-cluster-min-size", type=int, default=2)
    ap.add_argument("--reading-study-strategy",
                    choices=READING_STUDY_STRATEGIES, default="auto")
    ap.add_argument("--reading-study-probe-n", type=int, default=0)
    ap.add_argument("--reading-study-hard-max", type=int, default=0)
    ap.add_argument("--reading-study-refresh-steps", type=int, default=0)
    ap.add_argument("--reading-study-select-best",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reading-study-rounds", type=int, default=1)
    ap.add_argument("--reading-study-score-metric", choices=READING_SCORE_METRICS,
                    default="mastery")
    ap.add_argument("--reading-study-score-margin-w", type=float, default=0.1)
    ap.add_argument("--reading-study-score-min-delta", type=float, default=0.0)
    ap.add_argument("--reading-study-score-patience", type=int, default=0)


def _reading_kwargs(args):
    return dict(lr=args.reading_lr,
                token_drop_p=args.reading_token_drop,
                token_replace_p=args.reading_token_replace,
                feature_dropout=args.reading_feature_dropout,
                factorization_w=args.reading_factorization_w,
                factorization_variance=args.reading_factorization_variance,
                factorization_margin=args.reading_factorization_margin,
                factorization_covariance_w=args.reading_factorization_covariance_w,
                fer_w=args.reading_fer_w,
                fer_fragmentation_w=args.reading_fer_fragmentation_w,
                fer_correlation_w=args.reading_fer_correlation_w,
                fer_balance_w=args.reading_fer_balance_w,
                memory_w=args.reading_memory_w,
                memory_size=args.reading_memory_size,
                memory_temperature=args.reading_memory_temperature,
                memory_momentum=args.reading_memory_momentum,
                memory_balance_w=args.reading_memory_balance_w,
                association_w=args.reading_association_w,
                association_temperature=args.reading_association_temperature,
                association_decay=args.reading_association_decay,
                association_target_power=args.reading_association_target_power,
                association_self_loop_w=args.reading_association_self_loop_w,
                association_transitive_steps=args.reading_association_transitive_steps,
                association_transitive_w=args.reading_association_transitive_w,
                composition_w=args.reading_composition_w,
                composition_temperature=args.reading_composition_temperature,
                composition_self_loop_w=args.reading_composition_self_loop_w,
                composition_transitive_steps=args.reading_composition_transitive_steps,
                composition_transitive_w=args.reading_composition_transitive_w,
                composition_margin=args.reading_composition_margin,
                graph_predict_w=args.reading_graph_predict_w,
                graph_predict_temperature=args.reading_graph_predict_temperature,
                graph_predict_self_loop_w=args.reading_graph_predict_self_loop_w,
                graph_predict_transitive_steps=args.reading_graph_predict_transitive_steps,
                graph_predict_transitive_w=args.reading_graph_predict_transitive_w,
                graph_predict_target_power=args.reading_graph_predict_target_power,
                graph_cycle_w=args.reading_graph_cycle_w,
                graph_cycle_temperature=args.reading_graph_cycle_temperature,
                graph_cycle_self_loop_w=args.reading_graph_cycle_self_loop_w,
                graph_cycle_transitive_steps=args.reading_graph_cycle_transitive_steps,
                graph_cycle_transitive_w=args.reading_graph_cycle_transitive_w,
                graph_cycle_target_power=args.reading_graph_cycle_target_power,
                graph_cycle_consistency_w=args.reading_graph_cycle_consistency_w,
                bridge_w=args.reading_bridge_w,
                context_target_w=args.reading_context_target_w,
                context_keep_p=args.reading_context_keep_p,
                context_target_temperature=args.reading_context_target_temperature,
                sequence_w=args.reading_sequence_w,
                sequence_batch=args.reading_sequence_batch,
                sequence_temperature=args.reading_sequence_temperature,
                neighborhood_w=args.reading_neighborhood_w,
                neighborhood_batch=args.reading_neighborhood_batch,
                neighborhood_probe_n=args.reading_neighborhood_probe_n,
                neighborhood_refresh_steps=args.reading_neighborhood_refresh_steps,
                neighborhood_temperature=args.reading_neighborhood_temperature,
                neighborhood_margin=args.reading_neighborhood_margin,
                transition_w=args.reading_transition_w,
                transition_batch=args.reading_transition_batch,
                transition_temperature=args.reading_transition_temperature,
                transition_margin=args.reading_transition_margin,
                cluster_w=args.reading_cluster_w,
                cluster_batch=args.reading_cluster_batch,
                cluster_probe_n=args.reading_cluster_probe_n,
                cluster_refresh_steps=args.reading_cluster_refresh_steps,
                cluster_temperature=args.reading_cluster_temperature,
                cluster_margin=args.reading_cluster_margin,
                cluster_min_size=args.reading_cluster_min_size,
                study_strategy=args.reading_study_strategy,
                study_probe_n=args.reading_study_probe_n,
                study_hard_max=args.reading_study_hard_max,
                study_refresh_steps=args.reading_study_refresh_steps,
                study_select_best=args.reading_study_select_best,
                study_rounds=args.reading_study_rounds,
                study_score_metric=args.reading_study_score_metric,
                study_score_margin_w=args.reading_study_score_margin_w,
                study_score_min_delta=args.reading_study_score_min_delta,
                study_score_patience=args.reading_study_score_patience,
                text_field=args.reading_text_field,
                max_tokens=args.reading_max_tokens,
                min_tokens=args.reading_min_tokens,
                eval_frac=args.reading_eval_frac,
                eval_n=args.reading_eval_n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--import-scan", action="store_true")
    ap.add_argument("--import-snli", action="store_true")
    ap.add_argument("--import-mnli", action="store_true")
    ap.add_argument("--import-hans", action="store_true")
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
    ap.add_argument("--mnli-eval", type=int, default=2000)
    ap.add_argument("--hans-train-url", default=HANS_TRAIN_URL)
    ap.add_argument("--hans-eval-url", default=HANS_EVAL_URL)
    ap.add_argument("--hans-train-file", default=None)
    ap.add_argument("--hans-eval-file", default=None)
    ap.add_argument("--hans-train", type=int, default=5000)
    ap.add_argument("--hans-eval", type=int, default=3000)
    ap.add_argument("--hans-label-mode", choices=("snli", "binary"), default="snli")
    ap.add_argument("--data", action="append")
    _add_reading_args(ap)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--text-encoder-arch", choices=TEXT_ENCODER_ARCHES,
                    default="transformer")
    ap.add_argument("--text-encoder-layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=160, dest="max_new")
    ap.add_argument("--free-n", type=int, default=0, dest="free_n")
    ap.add_argument("--paraphrase-n", type=int, default=0, dest="paraphrase_n")
    ap.add_argument("--counterfactual-n", type=int, default=0, dest="counterfactual_n")
    ap.add_argument("--kind-free-n", type=int, default=0, dest="kind_free_n")
    ap.add_argument("--fact-n", type=int, default=0, dest="fact_n")
    ap.add_argument("--kind-fact-n", type=int, default=0, dest="kind_fact_n")
    ap.add_argument("--artifact-n", type=int, default=0, dest="artifact_n")
    ap.add_argument("--balance-by", choices=("none", "kind"), default="none")
    ap.add_argument("--semantic-w", type=float, default=0.5, dest="semantic_w")
    ap.add_argument("--fact-concept-w", type=float, default=0.0,
                    dest="fact_concept_w")
    ap.add_argument("--fact-concept-contrast-w", type=float, default=0.0,
                    dest="fact_concept_contrast_w")
    ap.add_argument("--fact-concept-contrast-temperature", type=float, default=0.1,
                    dest="fact_concept_contrast_temperature")
    ap.add_argument("--fact-concept-centroid-w", type=float, default=0.0,
                    dest="fact_concept_centroid_w")
    ap.add_argument("--fact-concept-centroid-temperature", type=float, default=0.1,
                    dest="fact_concept_centroid_temperature")
    ap.add_argument("--fact-concept-centroid-margin", type=float, default=0.0,
                    dest="fact_concept_centroid_margin")
    ap.add_argument("--fact-concept-prefix", action="store_true",
                    dest="fact_concept_prefix")
    ap.add_argument("--fact-concept-refine", action="store_true",
                    dest="fact_concept_refine")
    ap.add_argument("--fact-concept-refine-gate-init", type=float, default=-2.0,
                    dest="fact_concept_refine_gate_init")
    ap.add_argument("--fact-concept-mixer-layers", type=int, default=0,
                    dest="fact_concept_mixer_layers")
    ap.add_argument("--fact-concept-mixer-gate-init", type=float, default=-2.0,
                    dest="fact_concept_mixer_gate_init")
    ap.add_argument("--fact-concept-prototype-w", type=float, default=0.0,
                    dest="fact_concept_prototype_w")
    ap.add_argument("--fact-concept-prototype-spread-w", type=float, default=0.0,
                    dest="fact_concept_prototype_spread_w")
    ap.add_argument("--fact-concept-prototype-spread-margin", type=float, default=0.2,
                    dest="fact_concept_prototype_spread_margin")
    ap.add_argument("--fact-concept-state-spread-w", type=float, default=0.0,
                    dest="fact_concept_state_spread_w")
    ap.add_argument("--fact-concept-state-spread-variance", type=float, default=0.05,
                    dest="fact_concept_state_spread_variance")
    ap.add_argument("--fact-concept-state-spread-margin", type=float, default=0.2,
                    dest="fact_concept_state_spread_margin")
    ap.add_argument("--fact-concept-state-spread-covariance-w", type=float, default=0.05,
                    dest="fact_concept_state_spread_covariance_w")
    ap.add_argument("--latent-concept-slots", type=int, default=0)
    ap.add_argument("--latent-concept-layers", type=int, default=1)
    ap.add_argument("--latent-concept-prefix", action="store_true")
    ap.add_argument("--latent-concept-refine", action="store_true")
    ap.add_argument("--latent-concept-refine-gate-init", type=float, default=-2.0)
    ap.add_argument("--latent-concept-w", type=float, default=0.0)
    ap.add_argument("--latent-concept-view-dropout", type=float, default=0.1)
    ap.add_argument("--latent-concept-invariance-w", type=float, default=25.0)
    ap.add_argument("--latent-concept-variance-w", type=float, default=25.0)
    ap.add_argument("--latent-concept-covariance-w", type=float, default=1.0)
    ap.add_argument("--latent-concept-variance-target", type=float, default=1.0)
    ap.add_argument("--latent-concept-fact-w", type=float, default=0.0)
    ap.add_argument("--decode-w", type=float, default=1.0, dest="decode_w")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-checkpoint", default=None)
    ap.add_argument("--study-checkpoint", default=None)
    ap.add_argument("--study-out-checkpoint", default=None)
    ap.add_argument("--study-replay-data", action="append")
    ap.add_argument("--study-lr", type=float, default=5e-4)
    ap.add_argument("--study-rounds", type=int, default=1)
    ap.add_argument("--study-strategy", choices=("errors", "latent", "all"),
                    default="errors")
    ap.add_argument("--study-probe-n", type=int, default=0)
    ap.add_argument("--study-hard-max", type=int, default=0)
    ap.add_argument("--study-select-best", action="store_true")
    ap.add_argument("--study-score-metric",
                    choices=("semantic", "teacher", "concept", "latent", "both", "min"),
                    default="both")
    ap.add_argument("--study-retention-w", type=float, default=1.0)
    ap.add_argument("--study-control-w", type=float, default=1.0)
    ap.add_argument("--study-kind-w", type=float, default=1.0)
    ap.add_argument("--study-allow-negative-score", action="store_true")
    ap.add_argument("--study-confirm-n", type=int, default=0)
    ap.add_argument("--study-confirm-seed-stride", type=int, default=10000)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if ((args.import_scan or args.import_snli or args.import_mnli or args.import_hans)
            and not args.out):
        ap.error("--out is required for import commands")
    if args.import_scan:
        import_scan(args.out, url=args.scan_url, max_records=args.scan_max,
                    eval_frac=args.scan_eval_frac, seed=args.seed)
        return
    if args.import_snli:
        import_snli(args.out, zip_path=args.snli_zip, url=args.snli_url,
                    max_train=args.snli_train, max_eval=args.snli_eval,
                    seed=args.seed)
        return
    if args.import_mnli:
        import_mnli(args.out, zip_path=args.mnli_zip, url=args.mnli_url,
                    max_train=args.mnli_train, max_eval=args.mnli_eval,
                    seed=args.seed)
        return
    if args.import_hans:
        import_hans(args.out,
                    train_source=args.hans_train_file or args.hans_train_url,
                    eval_source=args.hans_eval_file or args.hans_eval_url,
                    max_train=args.hans_train, max_eval=args.hans_eval,
                    seed=args.seed, label_mode=args.hans_label_mode)
        return
    if args.reading_data:
        reading_common = _reading_kwargs(args)
        if args.reading_checkpoint:
            study_reading_checkpoint(
                args.reading_checkpoint, args.reading_data,
                out_checkpoint=args.reading_out_checkpoint or args.checkpoint,
                replay_data=args.reading_replay_data, out=args.out,
                steps=args.steps, batch=args.batch, seed=args.seed, device=DEV,
                replay_w=args.reading_replay_w,
                replay_batch=args.reading_replay_batch,
                replay_retention_w=args.reading_replay_retention_w,
                latent_concept_slots=args.latent_concept_slots,
                latent_concept_layers=args.latent_concept_layers,
                latent_concept_prefix=args.latent_concept_prefix,
                latent_concept_refine=args.latent_concept_refine,
                latent_concept_refine_gate_init=(
                    args.latent_concept_refine_gate_init),
                **reading_common)
        else:
            run_reading_concepts(
                args.reading_data, steps=args.steps, batch=args.batch, d=args.d,
                layers=args.layers, heads=args.heads,
                text_encoder_arch=args.text_encoder_arch,
                text_encoder_layers=args.text_encoder_layers,
                latent_concept_slots=args.latent_concept_slots,
                latent_concept_layers=args.latent_concept_layers,
                latent_concept_prefix=args.latent_concept_prefix,
                latent_concept_refine=args.latent_concept_refine,
                latent_concept_refine_gate_init=(
                    args.latent_concept_refine_gate_init),
                seed=args.seed, device=DEV, out=args.out,
                checkpoint=args.checkpoint, **reading_common)
        return
    if not args.data:
        raise SystemExit("--data is required unless --selftest or --reading-data is set")
    eval_common = dict(max_new=args.max_new, free_n=args.free_n,
                       paraphrase_n=args.paraphrase_n,
                       counterfactual_n=args.counterfactual_n,
                       kind_free_n=args.kind_free_n, fact_n=args.fact_n,
                       kind_fact_n=args.kind_fact_n, artifact_n=args.artifact_n,
                       seed=args.seed)
    if args.eval_checkpoint:
        eval_checkpoint(args.eval_checkpoint, args.data, out=args.out, device=DEV,
                        **eval_common)
        return
    train_common = dict(semantic_w=args.semantic_w, balance_by=args.balance_by,
                        fact_concept_w=args.fact_concept_w,
                        fact_concept_contrast_w=args.fact_concept_contrast_w,
                        fact_concept_contrast_temperature=(
                            args.fact_concept_contrast_temperature),
                        fact_concept_centroid_w=args.fact_concept_centroid_w,
                        fact_concept_centroid_temperature=(
                            args.fact_concept_centroid_temperature),
                        fact_concept_centroid_margin=args.fact_concept_centroid_margin,
                        fact_concept_prototype_w=args.fact_concept_prototype_w,
                        fact_concept_prototype_spread_w=(
                            args.fact_concept_prototype_spread_w),
                        fact_concept_prototype_spread_margin=(
                            args.fact_concept_prototype_spread_margin),
                        fact_concept_state_spread_w=args.fact_concept_state_spread_w,
                        fact_concept_state_spread_variance=(
                            args.fact_concept_state_spread_variance),
                        fact_concept_state_spread_margin=(
                            args.fact_concept_state_spread_margin),
                        fact_concept_state_spread_covariance_w=(
                            args.fact_concept_state_spread_covariance_w),
                        latent_concept_w=args.latent_concept_w,
                        latent_concept_view_dropout=args.latent_concept_view_dropout,
                        latent_concept_invariance_w=(
                            args.latent_concept_invariance_w),
                        latent_concept_variance_w=args.latent_concept_variance_w,
                        latent_concept_covariance_w=args.latent_concept_covariance_w,
                        latent_concept_variance_target=(
                            args.latent_concept_variance_target),
                        latent_concept_fact_w=args.latent_concept_fact_w,
                        decode_w=args.decode_w)
    if args.study_checkpoint:
        study_checkpoint(
            args.study_checkpoint, args.data,
            out_checkpoint=args.study_out_checkpoint or args.checkpoint,
            replay_data=args.study_replay_data, out=args.out,
            steps=args.steps, batch=args.batch, lr=args.study_lr,
            seed=args.seed, device=DEV, **eval_common, **train_common,
            study_rounds=args.study_rounds, study_strategy=args.study_strategy,
            study_probe_n=args.study_probe_n, study_hard_max=args.study_hard_max,
            study_select_best=args.study_select_best,
            study_score_metric=args.study_score_metric,
            study_retention_w=args.study_retention_w,
            study_control_w=args.study_control_w,
            study_kind_w=args.study_kind_w,
            study_require_positive_score=not args.study_allow_negative_score,
            study_confirm_n=args.study_confirm_n,
            study_confirm_seed_stride=args.study_confirm_seed_stride)
        return
    run(args.data, steps=args.steps, batch=args.batch, d=args.d, layers=args.layers,
        heads=args.heads, text_encoder_arch=args.text_encoder_arch,
        text_encoder_layers=args.text_encoder_layers,
        seed=args.seed, out=args.out, checkpoint=args.checkpoint,
        fact_concept_prefix=args.fact_concept_prefix,
        fact_concept_refine=args.fact_concept_refine,
        fact_concept_refine_gate_init=args.fact_concept_refine_gate_init,
        fact_concept_mixer_layers=args.fact_concept_mixer_layers,
        fact_concept_mixer_gate_init=args.fact_concept_mixer_gate_init,
        latent_concept_slots=args.latent_concept_slots,
        latent_concept_layers=args.latent_concept_layers,
        latent_concept_prefix=args.latent_concept_prefix,
        latent_concept_refine=args.latent_concept_refine,
        latent_concept_refine_gate_init=args.latent_concept_refine_gate_init,
        free_n=args.free_n, paraphrase_n=args.paraphrase_n,
        counterfactual_n=args.counterfactual_n, kind_free_n=args.kind_free_n,
        fact_n=args.fact_n, kind_fact_n=args.kind_fact_n,
        artifact_n=args.artifact_n, max_new=args.max_new, **train_common)


if __name__ == "__main__":
    main()
