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
import itertools
import io
import json
import math
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
from scratchpad_model import CausalBlock, ScratchpadLM

from .concepts import (
    SchemaConceptHead,
    SchemaConceptRefiner,
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
SQUAD_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
SQUAD_EVAL_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
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


def _answer_facts(answer_tokens):
    facts = [["answer", "length", str(len(answer_tokens))]]
    for i, tok in enumerate(answer_tokens):
        facts.append([f"a{i:03d}", "answer_token", tok])
    return facts


def _answer_span(context, answer):
    start = answer.get("answer_start")
    text = str(answer.get("text", ""))
    if isinstance(start, int) and start >= 0:
        return start, start + len(text)
    found = str(context).lower().find(text.lower())
    if found >= 0:
        return found, found + len(text)
    return None, None


def _context_window(context, answer, max_context_tokens):
    pieces = split_words_with_spans(context)
    tokens = [tok for tok, _start, _end in pieces]
    if not max_context_tokens or len(tokens) <= max_context_tokens:
        return tokens, False, True
    start, end = _answer_span(context, answer)
    if start is None or end is None:
        return tokens[:max_context_tokens], True, False
    overlap = [i for i, (_tok, s, e) in enumerate(pieces) if s < end and e > start]
    if not overlap:
        return tokens[:max_context_tokens], True, False
    ans_first, ans_last = overlap[0], overlap[-1]
    span_len = ans_last - ans_first + 1
    if span_len > max_context_tokens:
        return tokens[ans_first:ans_first + max_context_tokens], True, False
    left_budget = (max_context_tokens - span_len) // 2
    left = max(0, ans_first - left_budget)
    right = min(len(tokens), left + max_context_tokens)
    left = max(0, right - max_context_tokens)
    return tokens[left:right], True, left <= ans_first and ans_last < right


def _valid_squad_answers(answers, max_answer_tokens):
    out = []
    seen = set()
    for answer in answers or []:
        text = str(answer.get("text", "")).strip()
        toks = split_words(text)
        if not toks:
            continue
        if max_answer_tokens and len(toks) > max_answer_tokens:
            continue
        key = tuple(toks)
        if key in seen:
            continue
        seen.add(key)
        out.append((answer, toks))
    return out


def _squad_choice_tokens(candidates):
    toks = ["choices", ":"]
    for i, cand in enumerate(candidates):
        toks += ["choice", f"c{i:03d}", ":"] + list(cand)
    return toks


def _unique_token_tuples(items):
    out = []
    seen = set()
    for item in items:
        key = tuple(item)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _squad_records_from_payload(payload, split, limit, rng, source,
                                max_context_tokens=160, max_question_tokens=40,
                                max_answer_tokens=8, choice_n=0,
                                include_extractive=True,
                                choice_swap_negatives=0,
                                choice_absent_negatives=0):
    records = []
    seen = 0
    stats = {"qas_seen": 0, "skipped_no_question": 0, "skipped_no_answer": 0,
             "truncated_questions": 0,
             "skipped_answer_outside_window": 0, "windowed_contexts": 0,
             "extractive_records": 0, "choice_records": 0,
             "choice_swap_negative_records": 0,
             "choice_absent_negative_records": 0,
             "skipped_choice_distractors": 0,
             "skipped_choice_swap_negatives": 0,
             "skipped_choice_absent_negatives": 0}

    def maybe_keep(rec):
        nonlocal seen
        seen += 1
        if not limit or len(records) < limit:
            records.append(rec)
        else:
            j = int(rng.integers(seen))
            if j < limit:
                records[j] = rec

    for article in payload.get("data", []):
        title = str(article.get("title", "unknown"))
        for para_i, para in enumerate(article.get("paragraphs", [])):
            context = para.get("context", "")
            qa_items = []
            para_answers = []
            for qa in para.get("qas", []):
                stats["qas_seen"] += 1
                question = split_words(qa.get("question", ""))
                if not question:
                    stats["skipped_no_question"] += 1
                    continue
                if max_question_tokens and len(question) > max_question_tokens:
                    question = question[:max_question_tokens]
                    stats["truncated_questions"] += 1
                answers = _valid_squad_answers(qa.get("answers", []), max_answer_tokens)
                if not answers:
                    stats["skipped_no_answer"] += 1
                    continue
                answer_tokens_all = [tuple(tokens) for _answer, tokens in answers]
                para_answers.extend(answer_tokens_all)
                qa_items.append((qa, question, answers, answer_tokens_all))
            para_answer_pool = _unique_token_tuples(para_answers)
            for qa, question, answers, answer_tokens_all in qa_items:
                answer, answer_tokens = answers[int(rng.integers(len(answers)))]
                context_tokens, windowed, answer_in_window = _context_window(
                    context, answer, max_context_tokens)
                stats["windowed_contexts"] += int(windowed)
                if not context_tokens or not answer_in_window:
                    stats["skipped_answer_outside_window"] += 1
                    continue
                qid = str(qa.get("id", f"{title}-{para_i}-{stats['qas_seen']}"))
                base_tokens = ["context", ":"] + context_tokens + ["question", ":"] + question
                base_meta = {"source": source, "title": title,
                             "paragraph_id": f"{title}:{para_i}",
                             "answer_text": answer.get("text", ""),
                             "answer_start": answer.get("answer_start"),
                             "context_tokens": len(context_tokens),
                             "question_tokens": len(question),
                             "answer_tokens": len(answer_tokens),
                             "context_windowed": windowed}
                if include_extractive:
                    maybe_keep({"split": split,
                                "id": f"squad-{split}-{qid}",
                                "tokens": base_tokens,
                                "facts": _answer_facts(answer_tokens),
                                "kind": "squad_answer",
                                "group": f"squad-{qid}",
                                "meta": base_meta})
                    stats["extractive_records"] += 1
                if choice_n:
                    gold = tuple(answer_tokens)
                    distractors = [x for x in para_answer_pool if x not in set(answer_tokens_all)]
                    if len(distractors) < choice_n - 1:
                        stats["skipped_choice_distractors"] += 1
                        continue
                    idx = rng.choice(len(distractors), size=choice_n - 1, replace=False)
                    candidates = [gold] + [distractors[int(i)] for i in idx]
                    order = rng.permutation(len(candidates))
                    candidates = [candidates[int(i)] for i in order]
                    gold_pos = next(i for i, cand in enumerate(candidates) if cand == gold)
                    choice_id = f"c{gold_pos:03d}"
                    maybe_keep({"split": split,
                                "id": f"squad-choice-{split}-{qid}",
                                "tokens": base_tokens + _squad_choice_tokens(candidates),
                                "facts": [["answer", "choice", choice_id]],
                                "kind": "squad_choice",
                                "group": f"squad-choice-{qid}",
                                "meta": base_meta | {
                                    "choice_n": int(choice_n),
                                    "gold_choice": choice_id,
                                    "choices": [" ".join(c) for c in candidates],
                                }})
                    stats["choice_records"] += 1
                    if choice_absent_negatives:
                        if len(distractors) < choice_n:
                            stats["skipped_choice_absent_negatives"] += 1
                        else:
                            n_abs = int(choice_absent_negatives)
                            for neg_i in range(n_abs):
                                idx = rng.choice(len(distractors), size=choice_n,
                                                 replace=False)
                                absent_candidates = [distractors[int(i)] for i in idx]
                                order = rng.permutation(len(absent_candidates))
                                absent_candidates = [absent_candidates[int(i)]
                                                     for i in order]
                                maybe_keep({
                                    "split": split,
                                    "id": f"squad-choice-absent-{split}-{qid}-{neg_i}",
                                    "tokens": base_tokens + _squad_choice_tokens(absent_candidates),
                                    "facts": [["answer", "choice", "none"]],
                                    "kind": "squad_choice_absent_negative",
                                    "group": f"squad-choice-absent-{qid}-{neg_i}",
                                    "base_id": f"squad-choice-{split}-{qid}",
                                    "changed": [["answer", "choice", "none"]],
                                    "meta": base_meta | {
                                        "choice_n": int(choice_n),
                                        "gold_choice": "none",
                                        "negative": "answer_absent",
                                        "source_question_id": qid,
                                        "choices": [" ".join(c) for c in absent_candidates],
                                    },
                                })
                                stats["choice_absent_negative_records"] += 1
                    if choice_swap_negatives:
                        candidate_set = set(candidates)
                        donors = []
                        for donor_qa, donor_question, _donor_answers, donor_answer_tokens in qa_items:
                            donor_id = str(donor_qa.get("id", ""))
                            if donor_id == qid:
                                continue
                            if any(tuple(ans) in candidate_set for ans in donor_answer_tokens):
                                continue
                            donors.append((donor_id, donor_question))
                        if not donors:
                            stats["skipped_choice_swap_negatives"] += 1
                            continue
                        n_neg = min(int(choice_swap_negatives), len(donors))
                        neg_idx = rng.choice(len(donors), size=n_neg, replace=False)
                        for neg_i, donor_i in enumerate(neg_idx):
                            donor_id, donor_question = donors[int(donor_i)]
                            maybe_keep({
                                "split": split,
                                "id": f"squad-choice-neg-{split}-{qid}-{neg_i}",
                                "tokens": (["context", ":"] + context_tokens
                                           + ["question", ":"] + list(donor_question)
                                           + _squad_choice_tokens(candidates)),
                                "facts": [["answer", "choice", "none"]],
                                "kind": "squad_choice_swap_negative",
                                "group": f"squad-choice-neg-{qid}-{donor_id}-{neg_i}",
                                "base_id": f"squad-choice-{split}-{qid}",
                                "changed": [["answer", "choice", "none"]],
                                "meta": base_meta | {
                                    "choice_n": int(choice_n),
                                    "gold_choice": "none",
                                    "negative": "question_swap",
                                    "source_question_id": qid,
                                    "swapped_question_id": donor_id,
                                    "choices": [" ".join(c) for c in candidates],
                                },
                            })
                            stats["choice_swap_negative_records"] += 1
    return records, seen, stats


def _read_json_source(source, timeout=300):
    return json.loads(_read_text_source(source, timeout=timeout))


def import_squad(out, train_source=SQUAD_TRAIN_URL, eval_source=SQUAD_EVAL_URL,
                 max_train=5000, max_eval=1000, seed=0,
                 max_context_tokens=160, max_question_tokens=40,
                 max_answer_tokens=8, choice_n=0, choice_only=False,
                 choice_swap_negatives=0, choice_absent_negatives=0):
    """Import SQuAD v1.1 as extractive reading-comprehension facts.

    The model input is context + question. The target is not continuation text; it is the
    dataset answer rendered as ordered canonical answer-token facts.
    """
    rng = np.random.default_rng(seed)
    train_payload = _read_json_source(train_source, timeout=300)
    eval_payload = _read_json_source(eval_source, timeout=300)
    train, train_seen, train_stats = _squad_records_from_payload(
        train_payload, "train", max_train, rng, train_source,
        max_context_tokens=max_context_tokens,
        max_question_tokens=max_question_tokens,
        max_answer_tokens=max_answer_tokens,
        choice_n=choice_n,
        include_extractive=not choice_only,
        choice_swap_negatives=choice_swap_negatives,
        choice_absent_negatives=choice_absent_negatives)
    evals, eval_seen, eval_stats = _squad_records_from_payload(
        eval_payload, "eval", max_eval, rng, eval_source,
        max_context_tokens=max_context_tokens,
        max_question_tokens=max_question_tokens,
        max_answer_tokens=max_answer_tokens,
        choice_n=choice_n,
        include_extractive=not choice_only,
        choice_swap_negatives=choice_swap_negatives,
        choice_absent_negatives=choice_absent_negatives)
    records = train + evals
    if not train:
        raise ValueError("SQuAD import produced no train records")
    if not evals:
        raise ValueError("SQuAD import produced no eval records")
    write_jsonl(records, out)
    answer_lens = [int(r["meta"]["answer_tokens"]) for r in records]
    report = {"source": {"train": train_source, "eval": eval_source},
              "out": out,
              "records": len(records),
              "train_records": len(train),
              "eval_records": len(evals),
              "train_seen": train_seen,
              "eval_seen": eval_seen,
              "seed": int(seed),
              "max_context_tokens": int(max_context_tokens),
              "max_question_tokens": int(max_question_tokens),
              "max_answer_tokens": int(max_answer_tokens),
              "choice_n": int(choice_n),
              "choice_only": bool(choice_only),
              "choice_swap_negatives": int(choice_swap_negatives),
              "choice_absent_negatives": int(choice_absent_negatives),
              "train_stats": train_stats,
              "eval_stats": eval_stats,
              "answer_token_mean": float(np.mean(answer_lens)),
              "answer_token_max": int(max(answer_lens)),
              "representation": (
                  "SQuAD context+question -> ordered canonical answer-token facts")}
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


def qa_choice_spans(rec):
    toks = list(rec.tokens)
    try:
        i = toks.index("choices") + 1
    except ValueError:
        return []
    if i < len(toks) and toks[i] == ":":
        i += 1
    spans = []
    while i < len(toks):
        if i + 2 >= len(toks) or toks[i] != "choice" or toks[i + 2] != ":":
            i += 1
            continue
        choice_id = toks[i + 1]
        start = i + 3
        j = start
        while j < len(toks) and not (j + 2 < len(toks)
                                     and toks[j] == "choice"
                                     and toks[j + 2] == ":"):
            j += 1
        if start < j:
            spans.append((choice_id, start, j))
        i = j
    return spans


def qa_choice_target(rec):
    vals = [val for slot, pred, val in rec.facts
            if (slot, pred) == ("answer", "choice")]
    return vals[0] if vals else None


def qa_context_question_spans(rec):
    toks = list(rec.tokens)
    ctx_start = 2 if len(toks) >= 2 and toks[0] == "context" and toks[1] == ":" else 0
    try:
        choices_at = toks.index("choices")
    except ValueError:
        choices_at = len(toks)
    try:
        q_at = toks.index("question")
    except ValueError:
        return ctx_start, choices_at, 0, 0
    ctx_end = min(q_at, choices_at)
    q_start = q_at + 1
    if q_start < len(toks) and toks[q_start] == ":":
        q_start += 1
    q_end = choices_at if choices_at >= q_start else len(toks)
    return ctx_start, ctx_end, q_start, q_end


def _subsequence_positions(tokens, pattern):
    tokens = tuple(tokens)
    pattern = tuple(pattern)
    if not pattern or len(pattern) > len(tokens):
        return []
    out = []
    n = len(pattern)
    for i in range(0, len(tokens) - n + 1):
        if tokens[i:i + n] == pattern:
            out.extend(range(i, i + n))
    return sorted(set(out))


def qa_context_positions_for_span(rec, start, end):
    ctx_start, ctx_end, _q_start, _q_end = qa_context_question_spans(rec)
    if ctx_start >= ctx_end or start >= end:
        return []
    return _subsequence_positions(rec.tokens[ctx_start:ctx_end], rec.tokens[start:end])


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
                 fact_concept_refine=False, fact_concept_refine_gate_init=-2.0):
        super().__init__()
        self.d = int(d)
        self.heads = int(heads)
        self.fact_concept_prefix = bool(fact_concept_prefix)
        self.fact_concept_refine = bool(fact_concept_refine)
        self.fact_concept_refine_gate_init = float(fact_concept_refine_gate_init)
        self.text_encoder_arch = str(text_encoder_arch)
        self.text_encoder_layers = int(text_encoder_layers)
        self.txt = TextPrefix(vocab_size, d=d, pad=pad, heads=heads,
                              layers=self.text_encoder_layers,
                              arch=self.text_encoder_arch)
        self.lm = ScratchpadLM(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                               pad=pad, pointer=False, loop=False)
        self.choice_query = nn.Linear(d, d, bias=False)
        self.choice_context = nn.Linear(d, d, bias=False)
        self.choice_value = nn.Linear(d, d, bias=False)
        self.choice_answer = nn.Linear(d, d, bias=False)
        self.choice_cover = nn.Linear(d, d, bias=False)
        self.choice_threshold = nn.Parameter(torch.zeros(()))
        self.choice_context_mass_scale = nn.Parameter(torch.tensor(-3.0))
        self.choice_answerability_threshold = nn.Parameter(torch.zeros(()))
        self.choice_answerability_scale = nn.Parameter(torch.tensor(-3.0))
        self.fact_schema = fact_schema
        self.fact_heads = nn.ModuleDict()
        self.fact_concept_refiner = None
        if fact_schema is not None:
            self.fact_query = nn.Parameter(torch.randn(len(fact_schema.keys), d) * 0.02)
            self.fact_concepts = SchemaConceptHead(fact_schema.keys, fact_schema.values, d)
            if self.fact_concept_refine:
                self.fact_concept_refiner = SchemaConceptRefiner(
                    d, heads=heads, gate_init=self.fact_concept_refine_gate_init)
            for i, vals in enumerate(fact_schema.values):
                self.fact_heads[str(i)] = nn.Linear(d, len(vals))
        else:
            self.fact_concept_refine = False
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

    def encode_text(self, txt):
        mask = txt.eq(self.txt.pad)
        prefix = self.txt(txt)
        if (self.fact_concept_refine and self.fact_concepts is not None
                and self.fact_concept_refiner is not None):
            concepts = self.fact_concepts.state_tensor(prefix, mask=mask)
            prefix = self.fact_concept_refiner(prefix, concepts, mask=mask)
        keep = (~mask).unsqueeze(-1)
        pooled = (prefix * keep).sum(1) / keep.sum(1).clamp(min=1)
        return prefix, pooled

    def forward(self, txt, ids):
        prefix = self.decoder_prefix(txt)
        logits = self.lm(ids, prefix=prefix)
        return logits[:, prefix.shape[1]:]

    def decoder_prefix(self, txt):
        prefix, _pooled = self.encode_text(txt)
        if self.fact_concept_prefix and self.fact_concepts is not None:
            concepts = self.fact_concepts.state_tensor(prefix, mask=txt.eq(self.txt.pad))
            prefix = torch.cat([concepts, prefix], dim=1)
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

    def _choice_question(self, prefix, pooled, row, q_start, q_end):
        if q_start < q_end:
            q_tokens = self.choice_query(prefix[row, q_start:q_end])
            return q_tokens, q_tokens.mean(0)
        return None, torch.zeros_like(pooled[row])

    def _choice_question_context_scores(self, q_tokens, ctx_src):
        if q_tokens is None or ctx_src is None or not ctx_src.numel():
            return None
        scale = ctx_src.shape[-1] ** -0.5
        pair = torch.matmul(q_tokens, ctx_src.t()) * scale
        return torch.logsumexp(pair, 0) - math.log(max(1, q_tokens.shape[0]))

    def _choice_candidate_question_context(self, q_tokens, span, ctx_src):
        if q_tokens is None or ctx_src is None or not ctx_src.numel():
            return None, None
        scale = ctx_src.shape[-1] ** -0.5
        q_route = torch.matmul(q_tokens, span) * scale
        cand_q_vec = (q_route.softmax(0).unsqueeze(-1) * q_tokens).sum(0)
        ctx_scores = ((ctx_src * cand_q_vec).sum(-1)
                      + (ctx_src * span).sum(-1)) * scale
        return cand_q_vec, ctx_scores

    def _choice_context_mass_score(self, rec, start, end, q_ctx_scores):
        if q_ctx_scores is None:
            return None
        positions = qa_context_positions_for_span(rec, start, end)
        if not positions:
            return torch.full((), -math.log(max(2, int(q_ctx_scores.numel()) + 1)),
                              dtype=q_ctx_scores.dtype, device=q_ctx_scores.device)
        idx = torch.tensor(positions, dtype=torch.long, device=q_ctx_scores.device)
        log_mass = torch.logsumexp(q_ctx_scores[idx], 0) - torch.logsumexp(q_ctx_scores, 0)
        uniform_log_mass = math.log(len(positions)) - math.log(max(1, int(q_ctx_scores.numel())))
        return log_mass - uniform_log_mass

    def _choice_answer_vec(self, q_vec, q_ctx_scores, ctx_src):
        if q_ctx_scores is None or ctx_src is None or not ctx_src.numel():
            return q_vec
        weights = q_ctx_scores.softmax(-1)
        return (weights.unsqueeze(-1) * ctx_src).sum(0)

    def _choice_answerability_scaled(self, score):
        return F.softplus(self.choice_answerability_scale) * score

    def choice_logits(self, txt, records):
        """Return candidate logits plus a learned threshold logit for `none`.

        Candidate logits are evidence scores for each explicit choice span.  The `none` option is
        not another memorized answer class; it wins only when no candidate clears the learned
        evidence threshold.
        """
        out = []
        answerability = {rec.rec_id: (score, list(ids), cover)
                         for rec, score, ids, cover in self.choice_answerability_logits(
                             txt, records)
                         if score is not None}
        for rec, ids, logits in self.choice_candidate_logits(txt, records):
            if logits is None:
                out.append((rec, [], None))
                continue
            ans_row = answerability.get(rec.rec_id)
            if ans_row is not None:
                ans_score, ans_ids, cover_scores = ans_row
                if (cover_scores is not None and ans_ids == list(ids)
                        and cover_scores.shape == logits.shape):
                    logits = logits + self._choice_answerability_scaled(cover_scores)
                else:
                    logits = logits + self._choice_answerability_scaled(ans_score)
            ids = ["none"] + list(ids)
            logits = torch.cat([self.choice_threshold.reshape(1), logits])
            out.append((rec, ids, logits))
        return out

    def choice_answerability_logits(self, txt, records):
        prefix, pooled = self.encode_text(txt)
        context_values = self.choice_context(prefix)
        values = self.choice_value(prefix)
        out = []
        scale = prefix.shape[-1] ** -0.5
        for r, rec in enumerate(records):
            choices = qa_choice_spans(rec)
            if not choices:
                out.append((rec, None, [], None))
                continue
            ctx_start, ctx_end, q_start, q_end = qa_context_question_spans(rec)
            q_tokens, _q_vec = self._choice_question(prefix, pooled, r, q_start, q_end)
            ctx_src = context_values[r, ctx_start:ctx_end] if ctx_start < ctx_end else None
            ids = []
            cover_scores = []
            for choice_id, start, end in choices:
                span = values[r, start:end].mean(0)
                _cand_q_vec, cand_ctx_scores = self._choice_candidate_question_context(
                    q_tokens, span, ctx_src)
                if cand_ctx_scores is None:
                    answer_proj = torch.zeros_like(span)
                    mass_scores = None
                else:
                    weights = cand_ctx_scores.softmax(-1)
                    answer_vec = (weights.unsqueeze(-1) * ctx_src).sum(0)
                    answer_proj = self.choice_answer(answer_vec)
                    mass_scores = cand_ctx_scores
                cover = (answer_proj * self.choice_cover(span)).sum() * scale
                mass_score = self._choice_context_mass_score(rec, start, end, mass_scores)
                if mass_score is not None:
                    cover = cover + F.softplus(self.choice_context_mass_scale) * mass_score
                cover_scores.append(cover)
                ids.append(choice_id)
            if cover_scores:
                score = torch.logsumexp(torch.stack(cover_scores), 0)
                score = score - self.choice_answerability_threshold
                out.append((rec, score, ids, torch.stack(cover_scores)))
            else:
                out.append((rec, None, ids, None))
        return out

    def choice_candidate_logits(self, txt, records):
        prefix, pooled = self.encode_text(txt)
        context_values = self.choice_context(prefix)
        values = self.choice_value(prefix)
        out = []
        scale = prefix.shape[-1] ** -0.5
        for r, rec in enumerate(records):
            choices = qa_choice_spans(rec)
            if not choices:
                out.append((rec, [], None))
                continue
            ctx_start, ctx_end, q_start, q_end = qa_context_question_spans(rec)
            q_tokens, q_vec = self._choice_question(prefix, pooled, r, q_start, q_end)
            ctx_src = context_values[r, ctx_start:ctx_end] if ctx_start < ctx_end else None
            logits = []
            ids = []
            for choice_id, start, end in choices:
                span = values[r, start:end].mean(0)
                cand_q_vec, cand_ctx_scores = self._choice_candidate_question_context(
                    q_tokens, span, ctx_src)
                if cand_ctx_scores is not None:
                    ctx_vec = (cand_ctx_scores.softmax(-1).unsqueeze(-1) * ctx_src).sum(0)
                    route_vec = cand_q_vec
                else:
                    ctx_vec = torch.zeros_like(q_vec)
                    route_vec = q_vec
                logits.append(((ctx_vec * span).sum()
                               + (route_vec * ctx_vec).sum()) * scale)
                mass_score = self._choice_context_mass_score(rec, start, end,
                                                             cand_ctx_scores)
                if mass_score is not None:
                    logits[-1] = logits[-1] + F.softplus(
                        self.choice_context_mass_scale) * mass_score
                ids.append(choice_id)
            out.append((rec, ids, torch.stack(logits)))
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


def choice_loss(model, txt, records, answer_w=1.0, none_w=1.0,
                answer_margin=0.0, none_margin=0.0):
    answer_losses = []
    none_losses = []
    threshold = model.choice_threshold
    answer_margin_t = torch.tensor(float(answer_margin), dtype=torch.float32, device=txt.device)
    none_margin_t = torch.tensor(float(none_margin), dtype=torch.float32, device=txt.device)
    for rec, ids, logits in model.choice_candidate_logits(txt, records):
        target = qa_choice_target(rec)
        if target is None or logits is None:
            continue
        if target == "none":
            none_losses.append(F.softplus(torch.logsumexp(logits, 0) + none_margin_t
                                          - threshold))
        else:
            try:
                target_idx = ids.index(target)
            except ValueError:
                continue
            target_t = torch.tensor([target_idx], dtype=torch.long, device=txt.device)
            rank_loss = F.cross_entropy(logits[None], target_t)
            evidence_loss = F.softplus(threshold + answer_margin_t - logits[target_idx])
            answer_losses.append(rank_loss + evidence_loss)
    groups = []
    weights = []
    if answer_losses and answer_w:
        groups.append(sum(answer_losses) / len(answer_losses))
        weights.append(float(answer_w))
    if none_losses and none_w:
        groups.append(sum(none_losses) / len(none_losses))
        weights.append(float(none_w))
    if not groups:
        return torch.tensor(0.0, device=txt.device)
    return sum(w * loss for w, loss in zip(weights, groups)) / max(1e-9, sum(weights))


def choice_final_loss(model, txt, records, answer_w=1.0, none_w=1.0, margin=0.0):
    answer_losses = []
    none_losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for rec, ids, logits in model.choice_logits(txt, records):
        target = qa_choice_target(rec)
        if target is None or logits is None:
            continue
        try:
            target_idx = ids.index(target)
        except ValueError:
            continue
        target_t = torch.tensor([target_idx], dtype=torch.long, device=txt.device)
        loss = F.cross_entropy(logits[None], target_t)
        if logits.shape[0] > 1 and float(margin) != 0.0:
            mask = torch.ones(logits.shape[0], dtype=torch.bool, device=txt.device)
            mask[target_idx] = False
            loss = loss + F.softplus(torch.logsumexp(logits[mask], 0)
                                     + margin_t - logits[target_idx])
        if target == "none":
            none_losses.append(loss)
        else:
            answer_losses.append(loss)
    groups = []
    weights = []
    if answer_losses and answer_w:
        groups.append(sum(answer_losses) / len(answer_losses))
        weights.append(float(answer_w))
    if none_losses and none_w:
        groups.append(sum(none_losses) / len(none_losses))
        weights.append(float(none_w))
    if not groups:
        return torch.tensor(0.0, device=txt.device)
    return sum(w * loss for w, loss in zip(weights, groups)) / max(1e-9, sum(weights))


def choice_answerability_loss(model, txt, records, answer_w=1.0, none_w=1.0,
                              margin=0.0):
    answer_losses = []
    none_losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for rec, score, _ids, _cover_scores in model.choice_answerability_logits(txt, records):
        target = qa_choice_target(rec)
        if target is None or score is None:
            continue
        scaled = model._choice_answerability_scaled(score)
        if target == "none":
            none_losses.append(F.softplus(scaled + margin_t))
        else:
            answer_losses.append(F.softplus(margin_t - scaled))
    groups = []
    weights = []
    if answer_losses and answer_w:
        groups.append(sum(answer_losses) / len(answer_losses))
        weights.append(float(answer_w))
    if none_losses and none_w:
        groups.append(sum(none_losses) / len(none_losses))
        weights.append(float(none_w))
    if not groups:
        return torch.tensor(0.0, device=txt.device)
    return sum(w * loss for w, loss in zip(weights, groups)) / max(1e-9, sum(weights))


def choice_candidate_answerability_loss(model, txt, records, answer_w=1.0, none_w=1.0,
                                        margin=0.0):
    answer_losses = []
    none_losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for rec, _score, ids, cover_scores in model.choice_answerability_logits(txt, records):
        target = qa_choice_target(rec)
        if target is None or cover_scores is None:
            continue
        if target == "none":
            none_losses.append(F.softplus(torch.logsumexp(cover_scores, 0)
                                          + margin_t
                                          - model.choice_answerability_threshold))
        else:
            try:
                target_idx = ids.index(target)
            except ValueError:
                continue
            target_t = torch.tensor([target_idx], dtype=torch.long, device=txt.device)
            rank_loss = F.cross_entropy(cover_scores[None], target_t)
            evidence_loss = F.softplus(model.choice_answerability_threshold
                                       + margin_t
                                       - cover_scores[target_idx])
            answer_losses.append(rank_loss + evidence_loss)
    groups = []
    weights = []
    if answer_losses and answer_w:
        groups.append(sum(answer_losses) / len(answer_losses))
        weights.append(float(answer_w))
    if none_losses and none_w:
        groups.append(sum(none_losses) / len(none_losses))
        weights.append(float(none_w))
    if not groups:
        return torch.tensor(0.0, device=txt.device)
    return sum(w * loss for w, loss in zip(weights, groups)) / max(1e-9, sum(weights))


def choice_answerability_scores(model, txt, records, scaled=False):
    scores = {}
    for rec, score, _ids, _cover_scores in model.choice_answerability_logits(txt, records):
        if score is None:
            continue
        scores[rec.rec_id] = model._choice_answerability_scaled(score) if scaled else score
    return scores


def choice_candidate_answerability_scores(model, txt, records):
    scores = {}
    for rec, _score, ids, cover_scores in model.choice_answerability_logits(txt, records):
        target = qa_choice_target(rec)
        if target is None or target == "none" or cover_scores is None:
            continue
        try:
            target_idx = ids.index(target)
        except ValueError:
            continue
        scores[rec.rec_id] = cover_scores[target_idx]
    return scores


def choice_candidate_answerability_contrast_loss(model, txt, records, pair_ids,
                                                margin=0.0):
    scores = choice_candidate_answerability_scores(model, txt, records)
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for full_id, ablated_id in pair_ids:
        full_score = scores.get(full_id)
        ablated_score = scores.get(ablated_id)
        if full_score is not None and ablated_score is not None:
            losses.append(F.softplus(ablated_score + margin_t - full_score))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def _subsequence_token_positions(tokens, pattern):
    return _subsequence_positions(tokens, pattern)


def choice_target_context_positions(rec):
    target = qa_choice_target(rec)
    if target is None or target == "none":
        return []
    choices = {choice_id: (start, end)
               for choice_id, start, end in qa_choice_spans(rec)}
    if target not in choices:
        return []
    ctx_start, ctx_end, _q_start, _q_end = qa_context_question_spans(rec)
    if ctx_start >= ctx_end:
        return []
    choice_start, choice_end = choices[target]
    return _subsequence_token_positions(
        rec.tokens[ctx_start:ctx_end],
        rec.tokens[choice_start:choice_end])


def choice_answer_context_positions(rec):
    positions = choice_target_context_positions(rec)
    if positions:
        return positions
    if qa_choice_target(rec) != "none":
        return []
    if not isinstance(rec.meta, dict) or rec.meta.get("negative") != "answer_absent":
        return []
    answer = rec.meta.get("answer_text")
    if not answer:
        return []
    ctx_start, ctx_end, _q_start, _q_end = qa_context_question_spans(rec)
    if ctx_start >= ctx_end:
        return []
    return _subsequence_token_positions(rec.tokens[ctx_start:ctx_end],
                                        split_words(answer))


def choice_context_attention_loss(model, txt, records):
    prefix, pooled = model.encode_text(txt)
    context_values = model.choice_context(prefix)
    values = model.choice_value(prefix)
    losses = []
    for r, rec in enumerate(records):
        target = qa_choice_target(rec)
        if target is None or target == "none":
            continue
        choices = {choice_id: (start, end)
                   for choice_id, start, end in qa_choice_spans(rec)}
        if target not in choices:
            continue
        ctx_start, ctx_end, q_start, q_end = qa_context_question_spans(rec)
        if ctx_start >= ctx_end:
            continue
        choice_start, choice_end = choices[target]
        positions = choice_target_context_positions(rec)
        if not positions:
            continue
        q_tokens, _q_vec = model._choice_question(prefix, pooled, r, q_start, q_end)
        span = values[r, choice_start:choice_end].mean(0)
        ctx_src = context_values[r, ctx_start:ctx_end]
        _cand_q_vec, attn = model._choice_candidate_question_context(
            q_tokens, span, ctx_src)
        if attn is None:
            continue
        target_idx = torch.tensor(positions, dtype=torch.long, device=txt.device)
        losses.append(torch.logsumexp(attn, 0) - torch.logsumexp(attn[target_idx], 0))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_candidate_context_contrast_loss(model, txt, records, margin=0.0):
    prefix, pooled = model.encode_text(txt)
    context_values = model.choice_context(prefix)
    values = model.choice_value(prefix)
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for r, rec in enumerate(records):
        target = qa_choice_target(rec)
        if target is None or target == "none":
            continue
        choices = {choice_id: (start, end)
                   for choice_id, start, end in qa_choice_spans(rec)}
        if target not in choices or len(choices) < 2:
            continue
        ctx_start, ctx_end, q_start, q_end = qa_context_question_spans(rec)
        if ctx_start >= ctx_end:
            continue
        positions = choice_target_context_positions(rec)
        if not positions:
            continue
        q_tokens, _q_vec = model._choice_question(prefix, pooled, r, q_start, q_end)
        ctx_src = context_values[r, ctx_start:ctx_end]
        target_idx = torch.tensor(positions, dtype=torch.long, device=txt.device)
        span_masses = {}
        for choice_id, (start, end) in choices.items():
            span = values[r, start:end].mean(0)
            _cand_q_vec, attn = model._choice_candidate_question_context(
                q_tokens, span, ctx_src)
            if attn is None:
                continue
            span_masses[choice_id] = (torch.logsumexp(attn[target_idx], 0)
                                      - torch.logsumexp(attn, 0))
        pos_mass = span_masses.get(target)
        if pos_mass is None:
            continue
        for choice_id, neg_mass in span_masses.items():
            if choice_id != target:
                losses.append(F.softplus(neg_mass + margin_t - pos_mass))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_question_context_attention_loss(model, txt, records):
    prefix, pooled = model.encode_text(txt)
    context_values = model.choice_context(prefix)
    losses = []
    for r, rec in enumerate(records):
        positions = choice_answer_context_positions(rec)
        if not positions:
            continue
        ctx_start, ctx_end, q_start, q_end = qa_context_question_spans(rec)
        if ctx_start >= ctx_end or q_start >= q_end:
            continue
        q_tokens, _q_vec = model._choice_question(prefix, pooled, r, q_start, q_end)
        ctx_src = context_values[r, ctx_start:ctx_end]
        attn = model._choice_question_context_scores(q_tokens, ctx_src)
        if attn is None:
            continue
        target_idx = torch.tensor(positions, dtype=torch.long, device=txt.device)
        losses.append(torch.logsumexp(attn, 0) - torch.logsumexp(attn[target_idx], 0))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_pair_key(rec):
    return rec.base_id or rec.rec_id


def choice_pair_groups(records):
    by_key = {}
    for rec in records:
        target = qa_choice_target(rec)
        if target is None:
            continue
        by_key.setdefault(choice_pair_key(rec), []).append(rec)
    groups = []
    for key, rows in sorted(by_key.items()):
        positives = [r for r in rows if qa_choice_target(r) != "none"]
        negatives = [r for r in rows if qa_choice_target(r) == "none"]
        if positives and negatives:
            groups.append({"key": key, "positive": positives, "negative": negatives})
    return groups


def choice_pair_batch_records(groups, rng, pairs):
    if not groups or pairs <= 0:
        return []
    rows = []
    for _ in range(int(pairs)):
        group = groups[int(rng.integers(len(groups)))]
        pos = group["positive"][int(rng.integers(len(group["positive"])))]
        neg = group["negative"][int(rng.integers(len(group["negative"])))]
        rows.extend([pos, neg])
    if len(rows) > 1:
        order = rng.permutation(len(rows))
        rows = [rows[int(i)] for i in order]
    return rows


def choice_pair_loss(model, txt, records, margin=0.0):
    by_key = {}
    for rec, ids, logits in model.choice_candidate_logits(txt, records):
        target = qa_choice_target(rec)
        if target is None or logits is None:
            continue
        key = choice_pair_key(rec)
        if target == "none":
            by_key.setdefault(key, {"positive": [], "negative": []})["negative"].append(
                torch.logsumexp(logits, 0))
        else:
            try:
                target_idx = ids.index(target)
            except ValueError:
                continue
            by_key.setdefault(key, {"positive": [], "negative": []})["positive"].append(
                logits[target_idx])
    losses = []
    margin_t = torch.tensor(float(margin), dtype=txt.dtype if txt.is_floating_point() else torch.float32,
                            device=txt.device)
    for group in by_key.values():
        for pos_score in group["positive"]:
            for neg_score in group["negative"]:
                losses.append(F.softplus(neg_score + margin_t - pos_score))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_control_source_records(records):
    return [r for r in records
            if qa_choice_target(r) is not None
            and qa_choice_target(r) != "none"
            and not _is_qa_negative_record(r)]


def _choice_candidate_replacement_tokens(rec):
    target = qa_choice_target(rec)
    if target is None or target == "none":
        return []
    choices = choice_candidate_token_tuples(rec)
    gold = choices.get(target)
    if not gold:
        return []
    ctx_start, ctx_end, _q_start, _q_end = qa_context_question_spans(rec)
    if ctx_start >= ctx_end:
        return []
    blocked = set(choices.values())
    ctx = tuple(rec.tokens[ctx_start:ctx_end])
    span_len = len(gold)
    replacements = []
    for i in range(0, len(ctx) - span_len + 1):
        cand = tuple(ctx[i:i + span_len])
        if cand and cand not in blocked:
            replacements.append(cand)
    return _unique_token_tuples(replacements)


def _choice_candidate_replacement_record(rec, rng, facts=None, suffix="candidate_replace"):
    target = qa_choice_target(rec)
    if target is None or target == "none":
        return None
    choices = {choice_id: (start, end)
               for choice_id, start, end in qa_choice_spans(rec)}
    if target not in choices:
        return None
    replacements = _choice_candidate_replacement_tokens(rec)
    if not replacements:
        return None
    repl = replacements[int(rng.integers(len(replacements)))]
    start, end = choices[target]
    toks = tuple(list(rec.tokens[:start]) + list(repl) + list(rec.tokens[end:]))
    out_facts = tuple(facts) if facts is not None else (("answer", "choice", "none"),)
    control_meta = {"negative": suffix,
                    "source_record_id": rec.rec_id,
                    "replaced_choice": target,
                    "replacement_text": " ".join(repl)}
    meta = rec.meta | control_meta if isinstance(rec.meta, dict) else control_meta
    return TextRecord(rec_id=f"{rec.rec_id}:{suffix}",
                      split=rec.split,
                      tokens=toks,
                      facts=out_facts,
                      group=rec.group,
                      kind=f"{rec.kind}:{suffix}",
                      base_id=rec.rec_id,
                      changed=((("answer", "choice", "none"),)
                               if out_facts != rec.facts else rec.changed),
                      meta=meta)


def _choice_control_record(rec, side):
    if side == "candidate_replace":
        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(rec.rec_id))
        ablated = _choice_candidate_replacement_record(
            rec, np.random.default_rng(seed % (2 ** 32)))
    else:
        ablated = _qa_ablation_record(rec, side)
    if ablated is None:
        return None
    return TextRecord(rec_id=f"{rec.rec_id}:choice_control:{side}",
                      split=rec.split,
                      tokens=ablated.tokens,
                      facts=(("answer", "choice", "none"),),
                      group=rec.group,
                      kind=f"{rec.kind}:choice_control_{side}",
                      base_id=rec.rec_id,
                      changed=(("answer", "choice", "none"),),
                      meta=(rec.meta | {"negative": f"{side}_ablation",
                                        "source_record_id": rec.rec_id}
                            if isinstance(rec.meta, dict)
                            else {"negative": f"{side}_ablation",
                                  "source_record_id": rec.rec_id}))


CHOICE_CONTROL_SIDES = ("question", "context", "candidate_replace", "question_swap")


def normalize_choice_control_sides(sides, swap_groups=None):
    if not sides:
        out = ["question", "context", "candidate_replace"]
    elif isinstance(sides, str):
        out = [x.strip() for x in sides.split(",") if x.strip()]
    else:
        out = [str(x).strip() for x in sides if str(x).strip()]
    bad = sorted(set(out) - set(CHOICE_CONTROL_SIDES))
    if bad:
        raise ValueError(f"unknown choice control side(s): {bad}")
    if "question_swap" in out and not swap_groups:
        out = [x for x in out if x != "question_swap"]
    return tuple(out)


def choice_control_sides_from_failures(eval_report, metric="choice", threshold=0.05):
    sides = []
    for name, _gap in _control_gap_failures(
            eval_report, metric=metric, threshold=threshold).items():
        if "question_only" in name:
            sides.append("question")
        if "context_only" in name:
            sides.append("context")
        if "question_swap" in name:
            sides.append("question_swap")
        if "candidate_replacement" in name:
            sides.append("candidate_replace")
    return tuple(dict.fromkeys(sides))


def choice_control_batch_records(sources, rng, n, sides=None):
    if not sources or n <= 0:
        return []
    rows = []
    sides = normalize_choice_control_sides(sides)
    if not sides:
        return []
    tries = 0
    while len(rows) < int(n) and tries < int(n) * 4:
        rec = sources[int(rng.integers(len(sources)))]
        side = sides[int(rng.integers(len(sides)))]
        if side == "candidate_replace":
            ctrl = _choice_candidate_replacement_record(rec, rng)
        else:
            ctrl = _choice_control_record(rec, side)
        if ctrl is not None:
            rows.append(ctrl)
        tries += 1
    return rows


def choice_candidate_replacement_batch_records(sources, rng, n):
    if not sources or n <= 0:
        return []
    rows = []
    tries = 0
    while len(rows) < int(n) and tries < int(n) * 4:
        rec = sources[int(rng.integers(len(sources)))]
        repl = _choice_candidate_replacement_record(rec, rng)
        if repl is not None:
            rows.append(repl)
        tries += 1
    return rows


def choice_candidate_replacement_pair_batch(sources, rng, pairs):
    if not sources or pairs <= 0:
        return [], []
    rows = []
    pair_ids = []
    tries = 0
    while len(pair_ids) < int(pairs) and tries < int(pairs) * 4:
        rec = sources[int(rng.integers(len(sources)))]
        target = qa_choice_target(rec)
        repl = _choice_candidate_replacement_record(rec, rng)
        if repl is not None and target is not None and target != "none":
            idx = len(pair_ids)
            full_id = f"{rec.rec_id}:candidate_replacement_full:{idx}"
            repl_id = f"{rec.rec_id}:candidate_replacement_none:{idx}"
            full = TextRecord(rec_id=full_id, split=rec.split, tokens=rec.tokens,
                              facts=rec.facts, group=rec.group, kind=rec.kind,
                              base_id=rec.base_id, changed=rec.changed, meta=rec.meta)
            repl = TextRecord(rec_id=repl_id, split=rec.split, tokens=repl.tokens,
                              facts=(("answer", "choice", "none"),),
                              group=rec.group, kind=repl.kind, base_id=rec.rec_id,
                              changed=(("answer", "choice", "none"),),
                              meta=repl.meta)
            rows.extend([full, repl])
            pair_ids.append((full_id, repl_id, target))
        tries += 1
    return rows, pair_ids


def choice_positive_anchor_batch_records(sources, rng, n):
    if not sources or n <= 0:
        return []
    rows = []
    tries = 0
    while len(rows) < int(n) and tries < int(n) * 4:
        rec = sources[int(rng.integers(len(sources)))]
        target = qa_choice_target(rec)
        if target is not None and target != "none":
            rows.append(rec)
        tries += 1
    return rows


def choice_answerability_control_batch_records(sources, rng, n, swap_groups=None, sides=None):
    if not sources or n <= 0:
        return []
    rows = []
    sides = normalize_choice_control_sides(sides, swap_groups=swap_groups)
    if not sides:
        return []
    tries = 0
    while len(rows) < int(n) and tries < int(n) * 4:
        side = sides[int(rng.integers(len(sides)))]
        if side == "question_swap":
            rec, donors = swap_groups[int(rng.integers(len(swap_groups)))]
            donor = donors[int(rng.integers(len(donors)))]
            ablated = _qa_question_swap_record(rec, donor)
        elif side == "candidate_replace":
            rec = sources[int(rng.integers(len(sources)))]
            ablated = _choice_candidate_replacement_record(rec, rng)
        else:
            rec = sources[int(rng.integers(len(sources)))]
            ablated = _qa_ablation_record(rec, side)
        if ablated is not None:
            control_meta = {"negative": f"{side}_answerability_control",
                            "source_record_id": rec.rec_id}
            meta = (rec.meta | control_meta if isinstance(rec.meta, dict)
                    else control_meta)
            rows.append(TextRecord(
                rec_id=f"{rec.rec_id}:answerability_control:{side}:{len(rows)}",
                split=rec.split,
                tokens=ablated.tokens,
                facts=(("answer", "choice", "none"),),
                group=rec.group,
                kind=f"{rec.kind}:answerability_control_{side}",
                base_id=rec.rec_id,
                changed=(("answer", "choice", "none"),),
                meta=meta))
        tries += 1
    return rows


def choice_final_control_batch_records(sources, rng, n, swap_groups=None, sides=None):
    return choice_answerability_control_batch_records(
        sources, rng, n, swap_groups=swap_groups, sides=sides)


def choice_answerability_contrast_batch(sources, rng, pairs, swap_groups=None):
    if not sources or pairs <= 0:
        return [], []
    groups = swap_groups if swap_groups is not None else choice_question_swap_groups(sources)
    if not groups:
        return [], []
    rows = []
    ids = []
    tries = 0
    while len(ids) < int(pairs) and tries < int(pairs) * 4:
        rec, donors = groups[int(rng.integers(len(groups)))]
        donor = donors[int(rng.integers(len(donors)))]
        swapped = _qa_question_swap_record(rec, donor)
        if swapped is not None:
            full_id = f"{rec.rec_id}:ans_full:{len(ids)}"
            swap_id = f"{rec.rec_id}:ans_swap:{len(ids)}"
            full = TextRecord(rec_id=full_id, split=rec.split, tokens=rec.tokens,
                              facts=rec.facts, group=rec.group, kind=rec.kind,
                              base_id=rec.base_id, changed=rec.changed, meta=rec.meta)
            swapped = TextRecord(rec_id=swap_id, split=rec.split,
                                 tokens=swapped.tokens, facts=rec.facts,
                                 group=rec.group, kind=swapped.kind,
                                 base_id=rec.rec_id, changed=rec.changed,
                                 meta=rec.meta)
            rows.extend([full, swapped])
            ids.append((full_id, swap_id))
        tries += 1
    return rows, ids


def choice_candidate_token_tuples(rec):
    return {choice_id: tuple(rec.tokens[start:end])
            for choice_id, start, end in qa_choice_spans(rec)}


def choice_target_tokens(rec):
    target = qa_choice_target(rec)
    if target is None or target == "none":
        return None
    return choice_candidate_token_tuples(rec).get(target)


def choice_question_swap_groups(sources):
    by_context = {}
    for rec in sources:
        key = _qa_context_key(rec)
        if key is not None:
            by_context.setdefault(key, []).append(rec)
    groups = []
    for rows in by_context.values():
        if len(rows) < 2:
            continue
        for rec in rows:
            rec_candidates = set(choice_candidate_token_tuples(rec).values())
            donors = []
            for donor in rows:
                if donor.rec_id == rec.rec_id:
                    continue
                donor_target = choice_target_tokens(donor)
                if donor_target is None or donor_target in rec_candidates:
                    continue
                donors.append(donor)
            if donors:
                groups.append((rec, donors))
    return groups


def choice_control_contrast_batch(sources, rng, pairs, swap_groups=None, sides=None):
    if not sources or pairs <= 0:
        return [], []
    rows = []
    ids = []
    sides = normalize_choice_control_sides(sides, swap_groups=swap_groups)
    if not sides:
        return [], []
    tries = 0
    while len(ids) < int(pairs) and tries < int(pairs) * 4:
        side = sides[int(rng.integers(len(sides)))]
        if side == "question_swap":
            rec, donors = swap_groups[int(rng.integers(len(swap_groups)))]
            donor = donors[int(rng.integers(len(donors)))]
            ablated = _qa_question_swap_record(rec, donor)
        elif side == "candidate_replace":
            rec = sources[int(rng.integers(len(sources)))]
            ablated = _choice_candidate_replacement_record(
                rec, rng, facts=rec.facts, suffix="candidate_replace_contrast")
        else:
            rec = sources[int(rng.integers(len(sources)))]
            ablated = _qa_ablation_record(rec, side)
        if ablated is not None:
            full_id = f"{rec.rec_id}:full:{len(ids)}"
            ablated_id = f"{rec.rec_id}:ablated_{side}:{len(ids)}"
            full = TextRecord(rec_id=full_id, split=rec.split, tokens=rec.tokens,
                              facts=rec.facts, group=rec.group, kind=rec.kind,
                              base_id=rec.base_id, changed=rec.changed, meta=rec.meta)
            ablated = TextRecord(rec_id=ablated_id, split=rec.split,
                                 tokens=ablated.tokens, facts=rec.facts,
                                 group=rec.group, kind=ablated.kind,
                                 base_id=rec.rec_id, changed=rec.changed,
                                 meta=rec.meta)
            rows.extend([full, ablated])
            ids.append((full_id, ablated_id))
        tries += 1
    return rows, ids


def choice_target_scores(model, txt, records):
    scores = {}
    for rec, ids, logits in model.choice_candidate_logits(txt, records):
        target = qa_choice_target(rec)
        if target is None or target == "none" or logits is None:
            continue
        try:
            target_idx = ids.index(target)
        except ValueError:
            continue
        scores[rec.rec_id] = logits[target_idx]
    return scores


def choice_control_contrast_loss(model, txt, records, pair_ids, margin=0.0):
    scores = choice_target_scores(model, txt, records)
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for full_id, ablated_id in pair_ids:
        full_score = scores.get(full_id)
        ablated_score = scores.get(ablated_id)
        if full_score is not None and ablated_score is not None:
            losses.append(F.softplus(ablated_score + margin_t - full_score))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_final_target_scores(model, txt, records):
    scores = {}
    for rec, ids, logits in model.choice_logits(txt, records):
        target = qa_choice_target(rec)
        if target is None or target == "none" or logits is None:
            continue
        try:
            target_idx = ids.index(target)
        except ValueError:
            continue
        scores[rec.rec_id] = logits[target_idx]
    return scores


def choice_final_control_contrast_loss(model, txt, records, pair_ids, margin=0.0):
    scores = choice_final_target_scores(model, txt, records)
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for full_id, ablated_id in pair_ids:
        full_score = scores.get(full_id)
        ablated_score = scores.get(ablated_id)
        if full_score is not None and ablated_score is not None:
            losses.append(F.softplus(ablated_score + margin_t - full_score))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_candidate_replacement_pair_loss(model, txt, records, pair_ids, margin=0.0):
    rows = {}
    for rec, ids, logits in model.choice_logits(txt, records):
        if logits is not None:
            rows[rec.rec_id] = (ids, logits)
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for full_id, repl_id, target in pair_ids:
        full_row = rows.get(full_id)
        repl_row = rows.get(repl_id)
        if full_row is None or repl_row is None:
            continue
        full_ids, full_logits = full_row
        repl_ids, repl_logits = repl_row
        if target not in full_ids or "none" not in repl_ids:
            continue
        full_target_idx = full_ids.index(target)
        repl_none_idx = repl_ids.index("none")
        target_t = torch.tensor([full_target_idx], dtype=torch.long, device=txt.device)
        none_t = torch.tensor([repl_none_idx], dtype=torch.long, device=txt.device)
        losses.append(0.5 * (
            F.cross_entropy(full_logits[None], target_t)
            + F.cross_entropy(repl_logits[None], none_t)))
        if target in repl_ids:
            repl_target_idx = repl_ids.index(target)
            losses.append(F.softplus(repl_logits[repl_target_idx] + margin_t
                                     - full_logits[full_target_idx]))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_candidate_replacement_binding_scores(model, txt, records, pair_ids):
    rows = {}
    for rec, ids, logits in model.choice_logits(txt, records):
        if logits is not None:
            rows[rec.rec_id] = (ids, logits)
    out = []
    for full_id, repl_id, target in pair_ids:
        full_row = rows.get(full_id)
        repl_row = rows.get(repl_id)
        if full_row is None or repl_row is None:
            continue
        full_ids, full_logits = full_row
        repl_ids, repl_logits = repl_row
        if target not in full_ids or target not in repl_ids:
            continue
        full_score = full_logits[full_ids.index(target)]
        repl_score = repl_logits[repl_ids.index(target)]
        out.append((full_score, repl_score))
    return out


def choice_candidate_replacement_binding_loss(model, txt, records, pair_ids,
                                              margin=0.0):
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for full_score, repl_score in choice_candidate_replacement_binding_scores(
            model, txt, records, pair_ids):
        losses.append(F.softplus(repl_score + margin_t - full_score))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_candidate_replacement_answerability_binding_scores(model, txt, records,
                                                              pair_ids):
    rows = {}
    for rec, _score, ids, cover_scores in model.choice_answerability_logits(txt, records):
        if cover_scores is not None:
            rows[rec.rec_id] = (ids, cover_scores)
    out = []
    for full_id, repl_id, target in pair_ids:
        full_row = rows.get(full_id)
        repl_row = rows.get(repl_id)
        if full_row is None or repl_row is None:
            continue
        full_ids, full_scores = full_row
        repl_ids, repl_scores = repl_row
        if target not in full_ids or target not in repl_ids:
            continue
        full_score = full_scores[full_ids.index(target)]
        repl_score = repl_scores[repl_ids.index(target)]
        out.append((full_score, repl_score))
    return out


def choice_candidate_replacement_answerability_binding_loss(model, txt, records,
                                                            pair_ids, margin=0.0):
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for full_score, repl_score in choice_candidate_replacement_answerability_binding_scores(
            model, txt, records, pair_ids):
        losses.append(F.softplus(repl_score + margin_t - full_score))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_concept_groups(records):
    groups = {}
    for rec in records:
        target = qa_choice_target(rec)
        if target is None or target == "none":
            continue
        target_tokens = choice_target_tokens(rec)
        if target_tokens:
            groups.setdefault(target_tokens, []).append(rec)
    return [rows for _key, rows in sorted(groups.items()) if len(rows) >= 2]


def choice_concept_bridge_batch(groups, rng, pairs):
    if not groups or pairs <= 0:
        return [], []
    rows = []
    pair_ids = []
    tries = 0
    while len(pair_ids) < int(pairs) and tries < int(pairs) * 4:
        group = groups[int(rng.integers(len(groups)))]
        if len(group) < 2:
            tries += 1
            continue
        idx = rng.choice(len(group), size=2, replace=False)
        left = group[int(idx[0])]
        right = group[int(idx[1])]
        left_target = qa_choice_target(left)
        right_target = qa_choice_target(right)
        if left_target is None or right_target is None:
            tries += 1
            continue
        pair_idx = len(pair_ids)
        left_id = f"{left.rec_id}:concept_bridge_left:{pair_idx}"
        right_id = f"{right.rec_id}:concept_bridge_right:{pair_idx}"
        rows.extend([
            TextRecord(rec_id=left_id, split=left.split, tokens=left.tokens,
                       facts=left.facts, group=left.group, kind=left.kind,
                       base_id=left.base_id, changed=left.changed, meta=left.meta),
            TextRecord(rec_id=right_id, split=right.split, tokens=right.tokens,
                       facts=right.facts, group=right.group, kind=right.kind,
                       base_id=right.base_id, changed=right.changed, meta=right.meta),
        ])
        pair_ids.append((left_id, right_id, left_target, right_target))
        tries += 1
    return rows, pair_ids


def choice_candidate_concept_vectors(model, txt, records):
    prefix, pooled = model.encode_text(txt)
    context_values = model.choice_context(prefix)
    values = model.choice_value(prefix)
    out = {}
    for r, rec in enumerate(records):
        choices = qa_choice_spans(rec)
        if not choices:
            continue
        ctx_start, ctx_end, q_start, q_end = qa_context_question_spans(rec)
        q_tokens, q_vec = model._choice_question(prefix, pooled, r, q_start, q_end)
        ctx_src = context_values[r, ctx_start:ctx_end] if ctx_start < ctx_end else None
        row = {}
        for choice_id, start, end in choices:
            span = values[r, start:end].mean(0)
            cand_q_vec, cand_ctx_scores = model._choice_candidate_question_context(
                q_tokens, span, ctx_src)
            if cand_ctx_scores is not None:
                ctx_vec = (cand_ctx_scores.softmax(-1).unsqueeze(-1) * ctx_src).sum(0)
                route_vec = cand_q_vec
            else:
                ctx_vec = torch.zeros_like(q_vec)
                route_vec = q_vec
            row[choice_id] = F.normalize(span + ctx_vec + route_vec, dim=0)
        if row:
            out[rec.rec_id] = row
    return out


def choice_concept_bridge_scores(model, txt, records, pair_ids):
    vectors = choice_candidate_concept_vectors(model, txt, records)
    scores = []
    for left_id, right_id, left_target, right_target in pair_ids:
        left = vectors.get(left_id)
        right = vectors.get(right_id)
        if not left or not right or left_target not in left or right_target not in right:
            continue
        left_vec = left[left_target]
        right_vec = right[right_target]
        positive = (left_vec * right_vec).sum()
        negatives = []
        negatives.extend((left_vec * v).sum()
                         for cid, v in right.items() if cid != right_target)
        negatives.extend((right_vec * v).sum()
                         for cid, v in left.items() if cid != left_target)
        if negatives:
            negative = torch.stack(negatives).max()
        else:
            negative = torch.full((), -1.0, dtype=positive.dtype, device=positive.device)
        scores.append((positive, negative))
    return scores


def choice_concept_neighborhood_bridge_batch(model, vocab, sources, rng, pairs, device=DEV,
                                             pool_size=64):
    answer_sources = [r for r in sources
                      if qa_choice_target(r) not in (None, "none")]
    if len(answer_sources) < 2 or pairs <= 0:
        return [], []
    pool_n = min(len(answer_sources), max(2, int(pool_size), int(pairs) * 4))
    if pool_n < len(answer_sources):
        idx = rng.choice(len(answer_sources), size=pool_n, replace=False)
        pool = [answer_sources[int(i)] for i in idx]
    else:
        pool = list(answer_sources)
    txt, _ids = pack(pool, vocab, device)
    with torch.no_grad():
        vectors = choice_candidate_concept_vectors(model, txt, pool)
    items = []
    for rec in pool:
        target = qa_choice_target(rec)
        row = vectors.get(rec.rec_id)
        if row and target in row:
            items.append((rec, target, row[target].detach()))
    if len(items) < 2:
        return [], []
    rows = []
    pair_ids = []
    used = set()
    tries = 0
    while len(pair_ids) < int(pairs) and tries < int(pairs) * 8:
        left_i = int(rng.integers(len(items)))
        left_rec, left_target, left_vec = items[left_i]
        best = None
        for right_i, (right_rec, right_target, right_vec) in enumerate(items):
            if right_i == left_i:
                continue
            key = tuple(sorted((left_rec.rec_id, right_rec.rec_id)))
            if key in used:
                continue
            score = float((left_vec * right_vec).sum().cpu())
            if best is None or score > best[0]:
                best = (score, right_rec, right_target, key)
        if best is None:
            tries += 1
            continue
        _score, right_rec, right_target, key = best
        used.add(key)
        pair_idx = len(pair_ids)
        left_id = f"{left_rec.rec_id}:concept_neighbor_left:{pair_idx}"
        right_id = f"{right_rec.rec_id}:concept_neighbor_right:{pair_idx}"
        rows.extend([
            TextRecord(rec_id=left_id, split=left_rec.split, tokens=left_rec.tokens,
                       facts=left_rec.facts, group=left_rec.group,
                       kind=left_rec.kind, base_id=left_rec.base_id,
                       changed=left_rec.changed, meta=left_rec.meta),
            TextRecord(rec_id=right_id, split=right_rec.split, tokens=right_rec.tokens,
                       facts=right_rec.facts, group=right_rec.group,
                       kind=right_rec.kind, base_id=right_rec.base_id,
                       changed=right_rec.changed, meta=right_rec.meta),
        ])
        pair_ids.append((left_id, right_id, left_target, right_target))
        tries += 1
    return rows, pair_ids


def choice_concept_transfer_bridge_batch(model, vocab, hard_sources, correct_sources, rng,
                                         pairs, device=DEV, pool_per_side=64):
    """Pair current hard QA targets with nearest currently-correct target concepts."""
    hard_answer_sources = [r for r in hard_sources
                           if qa_choice_target(r) not in (None, "none")]
    correct_answer_sources = [r for r in correct_sources
                              if qa_choice_target(r) not in (None, "none")]
    if not hard_answer_sources or not correct_answer_sources or pairs <= 0:
        return [], []
    pool_per_side = max(1, int(pool_per_side))
    hard_n = min(len(hard_answer_sources), max(int(pairs) * 2, pool_per_side))
    correct_n = min(len(correct_answer_sources), max(int(pairs) * 2, pool_per_side))
    if hard_n < len(hard_answer_sources):
        idx = rng.choice(len(hard_answer_sources), size=hard_n, replace=False)
        hard_pool = [hard_answer_sources[int(i)] for i in idx]
    else:
        hard_pool = list(hard_answer_sources)
    if correct_n < len(correct_answer_sources):
        idx = rng.choice(len(correct_answer_sources), size=correct_n, replace=False)
        correct_pool = [correct_answer_sources[int(i)] for i in idx]
    else:
        correct_pool = list(correct_answer_sources)
    pool = unique_records_by_id(hard_pool, correct_pool)
    txt, _ids = pack(pool, vocab, device)
    with torch.no_grad():
        vectors = choice_candidate_concept_vectors(model, txt, pool)
    hard_items = []
    for rec in hard_pool:
        target = qa_choice_target(rec)
        row = vectors.get(rec.rec_id)
        if row and target in row:
            hard_items.append((rec, target, row[target].detach()))
    correct_items = []
    for rec in correct_pool:
        target = qa_choice_target(rec)
        row = vectors.get(rec.rec_id)
        if row and target in row:
            correct_items.append((rec, target, row[target].detach()))
    if not hard_items or not correct_items:
        return [], []
    hard_order = rng.permutation(len(hard_items)) if len(hard_items) > 1 else [0]
    rows = []
    pair_ids = []
    used = set()
    for hard_i in hard_order:
        if len(pair_ids) >= int(pairs):
            break
        hard_rec, hard_target, hard_vec = hard_items[int(hard_i)]
        best = None
        for correct_rec, correct_target, correct_vec in correct_items:
            key = (hard_rec.rec_id, correct_rec.rec_id)
            if key in used:
                continue
            score = float((hard_vec * correct_vec).sum().cpu())
            if best is None or score > best[0]:
                best = (score, correct_rec, correct_target, key)
        if best is None:
            continue
        _score, correct_rec, correct_target, key = best
        used.add(key)
        pair_idx = len(pair_ids)
        hard_id = f"{hard_rec.rec_id}:concept_transfer_hard:{pair_idx}"
        correct_id = f"{correct_rec.rec_id}:concept_transfer_correct:{pair_idx}"
        rows.extend([
            TextRecord(rec_id=hard_id, split=hard_rec.split, tokens=hard_rec.tokens,
                       facts=hard_rec.facts, group=hard_rec.group, kind=hard_rec.kind,
                       base_id=hard_rec.base_id, changed=hard_rec.changed,
                       meta=hard_rec.meta),
            TextRecord(rec_id=correct_id, split=correct_rec.split,
                       tokens=correct_rec.tokens, facts=correct_rec.facts,
                       group=correct_rec.group, kind=correct_rec.kind,
                       base_id=correct_rec.base_id, changed=correct_rec.changed,
                       meta=correct_rec.meta),
        ])
        pair_ids.append((hard_id, correct_id, hard_target, correct_target))
    return rows, pair_ids


def choice_concept_bridge_loss(model, txt, records, pair_ids, margin=0.0):
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for positive, negative in choice_concept_bridge_scores(model, txt, records, pair_ids):
        losses.append(F.softplus(margin_t - positive))
        losses.append(F.softplus(negative + margin_t - positive))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_concept_transfer_bridge_scores(model, txt, records, pair_ids):
    vectors = choice_candidate_concept_vectors(model, txt, records)
    scores = []
    for hard_id, correct_id, hard_target, correct_target in pair_ids:
        hard = vectors.get(hard_id)
        correct = vectors.get(correct_id)
        if (not hard or not correct or hard_target not in hard
                or correct_target not in correct):
            continue
        hard_vec = hard[hard_target]
        correct_vec = correct[correct_target].detach()
        positive = (hard_vec * correct_vec).sum()
        negatives = []
        negatives.extend((hard_vec * v.detach()).sum()
                         for cid, v in correct.items() if cid != correct_target)
        negatives.extend((v * correct_vec).sum()
                         for cid, v in hard.items() if cid != hard_target)
        if negatives:
            negative = torch.stack(negatives).max()
        else:
            negative = torch.full((), -1.0, dtype=positive.dtype, device=positive.device)
        scores.append((positive, negative))
    return scores


def choice_concept_transfer_bridge_loss(model, txt, records, pair_ids, margin=0.0):
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for positive, negative in choice_concept_transfer_bridge_scores(
            model, txt, records, pair_ids):
        losses.append(F.softplus(margin_t - positive))
        losses.append(F.softplus(negative + margin_t - positive))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_concept_prototype_batch(groups, rng, group_n, per_group=3):
    if not groups or group_n <= 0:
        return [], []
    usable = [group for group in groups if len(group) >= 2]
    if not usable:
        return [], []
    group_n = min(len(usable), max(1, int(group_n)))
    if group_n < len(usable):
        group_idx = rng.choice(len(usable), size=group_n, replace=False)
        selected = [usable[int(i)] for i in group_idx]
    else:
        selected = list(usable)
    rows = []
    items = []
    per_group = max(2, int(per_group))
    for concept_idx, group in enumerate(selected):
        member_n = min(len(group), per_group)
        if member_n < len(group):
            member_idx = rng.choice(len(group), size=member_n, replace=False)
            members = [group[int(i)] for i in member_idx]
        else:
            members = list(group)
        concept_id = f"concept_proto_{concept_idx}"
        for member_idx, rec in enumerate(members):
            target = qa_choice_target(rec)
            if target is None or target == "none":
                continue
            rec_id = f"{rec.rec_id}:concept_proto:{concept_idx}:{member_idx}"
            rows.append(TextRecord(rec_id=rec_id, split=rec.split, tokens=rec.tokens,
                                   facts=rec.facts, group=rec.group, kind=rec.kind,
                                   base_id=rec.base_id, changed=rec.changed,
                                   meta=rec.meta))
            items.append((rec_id, target, concept_id))
    if len(items) < 2:
        return [], []
    return rows, items


def choice_concept_neighborhood_prototype_batch(model, vocab, sources, rng, group_n,
                                                per_group=3, device=DEV,
                                                pool_size=64):
    answer_sources = [r for r in sources
                      if qa_choice_target(r) not in (None, "none")]
    if len(answer_sources) < 2 or group_n <= 0:
        return [], []
    group_n = max(1, int(group_n))
    per_group = max(2, int(per_group))
    pool_n = min(len(answer_sources),
                 max(per_group, int(pool_size), group_n * per_group * 2))
    if pool_n < len(answer_sources):
        idx = rng.choice(len(answer_sources), size=pool_n, replace=False)
        pool = [answer_sources[int(i)] for i in idx]
    else:
        pool = list(answer_sources)
    txt, _ids = pack(pool, vocab, device)
    with torch.no_grad():
        vectors = choice_candidate_concept_vectors(model, txt, pool)
    items = []
    for rec in pool:
        target = qa_choice_target(rec)
        row = vectors.get(rec.rec_id)
        if row and target in row:
            items.append((rec, target, row[target].detach()))
    if len(items) < 2:
        return [], []
    rows = []
    proto_items = []
    used = set()
    groups_built = 0
    tries = 0
    while groups_built < min(group_n, len(items)) and tries < group_n * 8:
        anchor_i = int(rng.integers(len(items)))
        if anchor_i in used:
            tries += 1
            continue
        anchor_rec, _anchor_target, anchor_vec = items[anchor_i]
        neighbors = []
        for other_i, (other_rec, other_target, other_vec) in enumerate(items):
            if other_i == anchor_i:
                continue
            score = float((anchor_vec * other_vec).sum().cpu())
            neighbors.append((score, other_i, other_rec, other_target))
        neighbors.sort(reverse=True, key=lambda row: row[0])
        selected = [(anchor_i, anchor_rec, qa_choice_target(anchor_rec))]
        for _score, other_i, other_rec, other_target in neighbors:
            if len(selected) >= per_group:
                break
            selected.append((other_i, other_rec, other_target))
        if len(selected) < 2:
            tries += 1
            continue
        concept_idx = groups_built
        concept_id = f"concept_neighbor_proto_{concept_idx}"
        for member_idx, (item_i, rec, target) in enumerate(selected):
            if target is None or target == "none":
                continue
            rec_id = f"{rec.rec_id}:concept_neighbor_proto:{concept_idx}:{member_idx}"
            rows.append(TextRecord(rec_id=rec_id, split=rec.split, tokens=rec.tokens,
                                   facts=rec.facts, group=rec.group, kind=rec.kind,
                                   base_id=rec.base_id, changed=rec.changed,
                                   meta=rec.meta))
            proto_items.append((rec_id, target, concept_id))
            used.add(item_i)
        groups_built += 1
        tries += 1
    if len(proto_items) < 2:
        return [], []
    return rows, proto_items


def _normalized_mean(vectors):
    return F.normalize(torch.stack(vectors).mean(0), dim=0)


def choice_concept_prototype_scores(model, txt, records, items):
    vectors = choice_candidate_concept_vectors(model, txt, records)
    by_concept = {}
    rows = []
    for rec_id, target, concept_id in items:
        row = vectors.get(rec_id)
        if not row or target not in row:
            continue
        target_vec = row[target]
        rows.append((rec_id, target, concept_id, target_vec, row))
        by_concept.setdefault(concept_id, []).append((rec_id, target_vec))
    scores = []
    for rec_id, target, concept_id, target_vec, candidate_row in rows:
        own_members = [vec for other_id, vec in by_concept.get(concept_id, [])
                       if other_id != rec_id]
        if not own_members:
            continue
        positive_proto = _normalized_mean(own_members).detach()
        positive = (target_vec * positive_proto).sum()
        negatives = []
        for other_concept, members in by_concept.items():
            if other_concept == concept_id:
                continue
            other_proto = _normalized_mean([vec for _other_id, vec in members]).detach()
            negatives.append((target_vec * other_proto).sum())
        negatives.extend((target_vec * cand_vec).sum()
                         for cand_id, cand_vec in candidate_row.items()
                         if cand_id != target)
        if negatives:
            negative = torch.stack(negatives).max()
        else:
            negative = torch.full((), -1.0, dtype=positive.dtype, device=positive.device)
        scores.append((positive, negative))
    return scores


def choice_concept_prototype_loss(model, txt, records, items, margin=0.0):
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for positive, negative in choice_concept_prototype_scores(model, txt, records, items):
        losses.append(F.softplus(margin_t - positive))
        losses.append(F.softplus(negative + margin_t - positive))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_self_distill_loss(model, teacher_model, txt, records, temperature=1.0):
    """Preserve a frozen teacher's current QA choice distribution on mined records."""
    temp = max(float(temperature), 1e-6)
    with torch.no_grad():
        teacher_model.eval()
        teacher_rows = teacher_model.choice_logits(txt, records)
    student_rows = model.choice_logits(txt, records)
    losses = []
    for (student_rec, student_ids, student_logits), (teacher_rec, teacher_ids,
                                                     teacher_logits) in zip(
            student_rows, teacher_rows):
        if student_rec.rec_id != teacher_rec.rec_id:
            continue
        if student_logits is None or teacher_logits is None:
            continue
        if list(student_ids) != list(teacher_ids):
            continue
        if student_logits.shape != teacher_logits.shape:
            continue
        teacher_prob = F.softmax(teacher_logits.detach() / temp, dim=0)
        student_logp = F.log_softmax(student_logits / temp, dim=0)
        losses.append(F.kl_div(student_logp, teacher_prob, reduction="sum") * temp * temp)
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_self_rank_distill_loss(model, teacher_model, txt, records, margin=0.0):
    """Preserve a frozen teacher's correct choice winner and target margin."""
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    with torch.no_grad():
        teacher_model.eval()
        teacher_rows = teacher_model.choice_logits(txt, records)
    student_rows = model.choice_logits(txt, records)
    losses = []
    for (student_rec, student_ids, student_logits), (teacher_rec, teacher_ids,
                                                     teacher_logits) in zip(
            student_rows, teacher_rows):
        if student_rec.rec_id != teacher_rec.rec_id:
            continue
        if student_logits is None or teacher_logits is None:
            continue
        if list(student_ids) != list(teacher_ids):
            continue
        if student_logits.shape != teacher_logits.shape or student_logits.numel() < 2:
            continue
        target = qa_choice_target(student_rec)
        if target is None or target not in student_ids:
            continue
        target_idx = student_ids.index(target)
        if int(teacher_logits.argmax(-1)) != target_idx:
            continue
        mask = torch.ones(student_logits.shape[0], dtype=torch.bool, device=txt.device)
        mask[target_idx] = False
        teacher_other = teacher_logits.detach()[mask].max()
        teacher_target = teacher_logits.detach()[target_idx]
        teacher_margin = (teacher_target - teacher_other).clamp_min(0.0)
        desired_margin = torch.maximum(teacher_margin, margin_t)
        student_other = student_logits[mask].max()
        student_target = student_logits[target_idx]
        target_t = torch.tensor([target_idx], dtype=torch.long, device=txt.device)
        losses.append(F.cross_entropy(student_logits[None], target_t))
        losses.append(F.softplus(student_other + desired_margin - student_target))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_question_context_scores(model, txt, records):
    prefix, pooled = model.encode_text(txt)
    context_values = model.choice_context(prefix)
    scores = {}
    for r, rec in enumerate(records):
        positions = choice_target_context_positions(rec)
        if not positions:
            continue
        ctx_start, ctx_end, q_start, q_end = qa_context_question_spans(rec)
        if ctx_start >= ctx_end or q_start >= q_end:
            continue
        q_tokens, _q_vec = model._choice_question(prefix, pooled, r, q_start, q_end)
        ctx_src = context_values[r, ctx_start:ctx_end]
        attn = model._choice_question_context_scores(q_tokens, ctx_src)
        if attn is None:
            continue
        target_idx = torch.tensor(positions, dtype=torch.long, device=txt.device)
        scores[rec.rec_id] = torch.logsumexp(attn[target_idx], 0) - torch.logsumexp(attn, 0)
    return scores


def choice_question_context_swap_batch(sources, rng, pairs, swap_groups=None):
    if not sources or pairs <= 0:
        return [], []
    groups = swap_groups if swap_groups is not None else choice_question_swap_groups(sources)
    if not groups:
        return [], []
    rows = []
    ids = []
    tries = 0
    while len(ids) < int(pairs) and tries < int(pairs) * 4:
        rec, donors = groups[int(rng.integers(len(groups)))]
        donor = donors[int(rng.integers(len(donors)))]
        swapped = _qa_question_swap_record(rec, donor)
        if swapped is not None:
            full_id = f"{rec.rec_id}:qctx_full:{len(ids)}"
            swap_id = f"{rec.rec_id}:qctx_swap:{len(ids)}"
            full = TextRecord(rec_id=full_id, split=rec.split, tokens=rec.tokens,
                              facts=rec.facts, group=rec.group, kind=rec.kind,
                              base_id=rec.base_id, changed=rec.changed, meta=rec.meta)
            swapped = TextRecord(rec_id=swap_id, split=rec.split,
                                 tokens=swapped.tokens, facts=rec.facts,
                                 group=rec.group, kind=swapped.kind,
                                 base_id=rec.rec_id, changed=rec.changed,
                                 meta=rec.meta)
            rows.extend([full, swapped])
            ids.append((full_id, swap_id))
        tries += 1
    return rows, ids


def choice_question_context_contrast_loss(model, txt, records, pair_ids, margin=0.0):
    scores = choice_question_context_scores(model, txt, records)
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for full_id, swap_id in pair_ids:
        full_score = scores.get(full_id)
        swap_score = scores.get(swap_id)
        if full_score is not None and swap_score is not None:
            losses.append(F.softplus(swap_score + margin_t - full_score))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_answerability_contrast_loss(model, txt, records, pair_ids, margin=0.0):
    scores = choice_answerability_scores(model, txt, records, scaled=False)
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for full_id, swap_id in pair_ids:
        full_score = scores.get(full_id)
        swap_score = scores.get(swap_id)
        if full_score is not None and swap_score is not None:
            losses.append(F.softplus(swap_score + margin_t - full_score))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


def choice_answerability_pair_loss(model, txt, records, margin=0.0):
    by_key = {}
    for rec, score, _ids, _cover_scores in model.choice_answerability_logits(txt, records):
        target = qa_choice_target(rec)
        if target is None or score is None:
            continue
        group = by_key.setdefault(choice_pair_key(rec),
                                  {"positive": [], "negative": []})
        if target == "none":
            group["negative"].append(score)
        else:
            group["positive"].append(score)
    losses = []
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=txt.device)
    for group in by_key.values():
        for pos_score in group["positive"]:
            for neg_score in group["negative"]:
                losses.append(F.softplus(neg_score + margin_t - pos_score))
    if not losses:
        return torch.tensor(0.0, device=txt.device)
    return sum(losses) / len(losses)


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
              prefix="text", decode_w=1.0, choice_w=0.0,
              choice_answer_w=1.0, choice_none_w=1.0,
              choice_answer_margin=0.0, choice_none_margin=0.0,
              choice_final_w=0.0, choice_final_control_w=0.0,
              choice_final_margin=0.0,
              choice_final_control_contrast_w=0.0,
              choice_final_control_contrast_margin=0.0,
              choice_candidate_replacement_w=0.0,
              choice_candidate_replacement_margin=0.0,
              choice_candidate_replacement_pair_w=0.0,
              choice_candidate_replacement_pair_margin=0.0,
              choice_candidate_replacement_binding_w=0.0,
              choice_candidate_replacement_binding_margin=0.0,
              choice_candidate_replacement_answerability_binding_w=0.0,
              choice_candidate_replacement_answerability_binding_margin=0.0,
              choice_positive_anchor_w=0.0,
              choice_positive_anchor_margin=0.0,
              choice_positive_anchor_sources=None,
              choice_concept_bridge_w=0.0,
              choice_concept_bridge_margin=0.0,
              choice_concept_prototype_w=0.0,
              choice_concept_prototype_margin=0.0,
              choice_concept_bridge_sources=None,
              choice_concept_hard_sources=None,
              choice_concept_correct_sources=None,
              choice_self_distill_w=0.0,
              choice_self_distill_temperature=1.0,
              choice_self_rank_distill_w=0.0,
              choice_self_rank_distill_margin=0.0,
              choice_self_distill_sources=None,
              choice_self_distill_model=None,
              choice_answerability_w=0.0, choice_answerability_control_w=0.0,
              choice_answerability_margin=0.0, choice_answerability_contrast_w=0.0,
              choice_answerability_contrast_margin=0.0,
              choice_candidate_answerability_w=0.0,
              choice_candidate_answerability_control_w=0.0,
              choice_candidate_answerability_margin=0.0,
              choice_candidate_answerability_contrast_w=0.0,
              choice_candidate_answerability_contrast_margin=0.0,
              choice_answerability_pair_w=0.0,
              choice_answerability_pair_margin=0.0,
              choice_context_w=0.0,
              choice_candidate_context_w=0.0,
              choice_candidate_context_margin=0.0,
              choice_question_context_w=0.0,
              choice_question_context_contrast_w=0.0,
              choice_question_context_margin=0.0,
              choice_pair_w=0.0, choice_pair_margin=0.0,
              choice_control_w=0.0, choice_control_contrast_w=0.0,
              choice_control_margin=0.0,
              choice_control_sides=None):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_records = [r for r in records if r.split == "train"]
    if not train_records:
        raise ValueError("cannot train without train records")
    train_buckets = bucket_records(train_records, balance_by)
    pair_groups = (choice_pair_groups(train_records)
                   if (choice_pair_w or choice_answerability_pair_w) else [])
    concept_source_records = (list(choice_concept_bridge_sources)
                              if choice_concept_bridge_sources is not None
                              else train_records)
    concept_hard_records = list(choice_concept_hard_sources or [])
    concept_correct_records = list(choice_concept_correct_sources or [])
    concept_groups = (choice_concept_groups(concept_source_records)
                      if (choice_concept_bridge_w or choice_concept_prototype_w)
                      else [])
    distill_sources = list(choice_self_distill_sources or [])
    control_sources = (choice_control_source_records(train_records)
                       if (choice_control_w or choice_control_contrast_w
                           or choice_candidate_replacement_w
                           or choice_candidate_replacement_pair_w
                           or choice_candidate_replacement_binding_w
                           or choice_candidate_replacement_answerability_binding_w
                           or choice_positive_anchor_w
                           or choice_final_control_w
                           or choice_final_control_contrast_w
                           or choice_question_context_contrast_w
                           or choice_answerability_control_w
                           or choice_candidate_answerability_control_w
                           or choice_candidate_answerability_contrast_w
                           or choice_answerability_contrast_w) else [])
    needs_swap_groups = bool(choice_control_contrast_w
                             or choice_final_control_w
                             or choice_final_control_contrast_w
                             or choice_question_context_contrast_w
                             or choice_answerability_control_w
                             or choice_candidate_answerability_control_w
                             or choice_candidate_answerability_contrast_w
                             or choice_answerability_contrast_w)
    swap_groups = (choice_question_swap_groups(control_sources)
                   if needs_swap_groups and control_sources else [])
    positive_anchor_sources = (
        list(choice_positive_anchor_sources)
        if choice_positive_anchor_sources is not None else control_sources)
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
        ch_loss = choice_loss(model, txt, rec_batch,
                              answer_w=choice_answer_w, none_w=choice_none_w,
                              answer_margin=choice_answer_margin,
                              none_margin=choice_none_margin)
        final_loss = (choice_final_loss(
            model, txt, rec_batch, answer_w=choice_answer_w,
            none_w=choice_none_w, margin=choice_final_margin)
            if choice_final_w else torch.tensor(0.0, device=device))
        ans_loss = (choice_answerability_loss(
            model, txt, rec_batch,
            answer_w=choice_answer_w, none_w=choice_none_w,
            margin=choice_answerability_margin)
            if choice_answerability_w else torch.tensor(0.0, device=device))
        cand_ans_loss = (choice_candidate_answerability_loss(
            model, txt, rec_batch,
            answer_w=choice_answer_w, none_w=choice_none_w,
            margin=choice_candidate_answerability_margin)
            if choice_candidate_answerability_w else torch.tensor(0.0, device=device))
        ctx_loss = (choice_context_attention_loss(model, txt, rec_batch)
                    if choice_context_w else torch.tensor(0.0, device=device))
        cand_ctx_loss = (choice_candidate_context_contrast_loss(
            model, txt, rec_batch, margin=choice_candidate_context_margin)
            if choice_candidate_context_w else torch.tensor(0.0, device=device))
        qctx_loss = (choice_question_context_attention_loss(model, txt, rec_batch)
                     if choice_question_context_w else torch.tensor(0.0, device=device))
        if choice_pair_w and pair_groups:
            pair_records = choice_pair_batch_records(pair_groups, rng, max(1, batch // 2))
            pair_txt, _pair_ids = pack(pair_records, vocab, device)
            pair_loss = choice_pair_loss(model, pair_txt, pair_records,
                                         margin=choice_pair_margin)
        else:
            pair_loss = torch.tensor(0.0, device=device)
        if choice_control_w and control_sources:
            control_records = choice_control_batch_records(control_sources, rng,
                                                           max(1, batch // 2),
                                                           sides=choice_control_sides)
            if control_records:
                control_txt, _control_ids = pack(control_records, vocab, device)
                control_loss = choice_loss(model, control_txt, control_records,
                                           answer_w=0.0, none_w=1.0,
                                           none_margin=choice_none_margin)
            else:
                control_loss = torch.tensor(0.0, device=device)
        else:
            control_loss = torch.tensor(0.0, device=device)
        if choice_final_control_w and control_sources:
            final_control_records = choice_final_control_batch_records(
                control_sources, rng, max(1, batch // 2), swap_groups=swap_groups,
                sides=choice_control_sides)
            if final_control_records:
                final_control_txt, _final_control_ids = pack(
                    final_control_records, vocab, device)
                final_control_loss = choice_final_loss(
                    model, final_control_txt, final_control_records,
                    answer_w=0.0, none_w=1.0, margin=choice_final_margin)
            else:
                final_control_loss = torch.tensor(0.0, device=device)
        else:
            final_control_loss = torch.tensor(0.0, device=device)
        if choice_final_control_contrast_w and control_sources:
            final_contrast_records, final_contrast_pairs = choice_control_contrast_batch(
                control_sources, rng, max(1, batch // 2),
                swap_groups=swap_groups, sides=choice_control_sides)
            if final_contrast_records and final_contrast_pairs:
                final_contrast_txt, _final_contrast_ids = pack(
                    final_contrast_records, vocab, device)
                final_contrast_loss = choice_final_control_contrast_loss(
                    model, final_contrast_txt, final_contrast_records,
                    final_contrast_pairs,
                    margin=choice_final_control_contrast_margin)
            else:
                final_contrast_loss = torch.tensor(0.0, device=device)
        else:
            final_contrast_loss = torch.tensor(0.0, device=device)
        if choice_candidate_replacement_w and control_sources:
            replacement_records = choice_candidate_replacement_batch_records(
                control_sources, rng, max(1, batch // 2))
            if replacement_records:
                replacement_txt, _replacement_ids = pack(
                    replacement_records, vocab, device)
                replacement_loss = choice_final_loss(
                    model, replacement_txt, replacement_records,
                    answer_w=0.0, none_w=1.0,
                    margin=choice_candidate_replacement_margin)
            else:
                replacement_loss = torch.tensor(0.0, device=device)
        else:
            replacement_loss = torch.tensor(0.0, device=device)
        if choice_candidate_replacement_pair_w and control_sources:
            replacement_pair_records, replacement_pairs = (
                choice_candidate_replacement_pair_batch(
                    control_sources, rng, max(1, batch // 2)))
            if replacement_pair_records and replacement_pairs:
                replacement_pair_txt, _replacement_pair_ids = pack(
                    replacement_pair_records, vocab, device)
                replacement_pair_loss = choice_candidate_replacement_pair_loss(
                    model, replacement_pair_txt, replacement_pair_records,
                    replacement_pairs,
                    margin=choice_candidate_replacement_pair_margin)
            else:
                replacement_pair_loss = torch.tensor(0.0, device=device)
        else:
            replacement_pair_loss = torch.tensor(0.0, device=device)
        if choice_candidate_replacement_binding_w and control_sources:
            replacement_binding_records, replacement_binding_pairs = (
                choice_candidate_replacement_pair_batch(
                    control_sources, rng, max(1, batch // 2)))
            if replacement_binding_records and replacement_binding_pairs:
                replacement_binding_txt, _replacement_binding_ids = pack(
                    replacement_binding_records, vocab, device)
                replacement_binding_loss = choice_candidate_replacement_binding_loss(
                    model, replacement_binding_txt, replacement_binding_records,
                    replacement_binding_pairs,
                    margin=choice_candidate_replacement_binding_margin)
            else:
                replacement_binding_loss = torch.tensor(0.0, device=device)
        else:
            replacement_binding_loss = torch.tensor(0.0, device=device)
        if (choice_candidate_replacement_answerability_binding_w
                and control_sources):
            replacement_ans_binding_records, replacement_ans_binding_pairs = (
                choice_candidate_replacement_pair_batch(
                    control_sources, rng, max(1, batch // 2)))
            if replacement_ans_binding_records and replacement_ans_binding_pairs:
                replacement_ans_binding_txt, _replacement_ans_binding_ids = pack(
                    replacement_ans_binding_records, vocab, device)
                replacement_ans_binding_loss = (
                    choice_candidate_replacement_answerability_binding_loss(
                        model, replacement_ans_binding_txt,
                        replacement_ans_binding_records,
                        replacement_ans_binding_pairs,
                        margin=(
                            choice_candidate_replacement_answerability_binding_margin)))
            else:
                replacement_ans_binding_loss = torch.tensor(0.0, device=device)
        else:
            replacement_ans_binding_loss = torch.tensor(0.0, device=device)
        if choice_positive_anchor_w and positive_anchor_sources:
            positive_anchor_records = choice_positive_anchor_batch_records(
                positive_anchor_sources, rng, max(1, batch // 2))
            if positive_anchor_records:
                positive_anchor_txt, _positive_anchor_ids = pack(
                    positive_anchor_records, vocab, device)
                positive_anchor_loss = choice_final_loss(
                    model, positive_anchor_txt, positive_anchor_records,
                    answer_w=1.0, none_w=0.0,
                    margin=choice_positive_anchor_margin)
            else:
                positive_anchor_loss = torch.tensor(0.0, device=device)
        else:
            positive_anchor_loss = torch.tensor(0.0, device=device)
        if choice_concept_bridge_w:
            concept_transfer_pairing = False
            if concept_hard_records and concept_correct_records:
                concept_records, concept_pairs = choice_concept_transfer_bridge_batch(
                    model, vocab, concept_hard_records, concept_correct_records, rng,
                    max(1, batch // 2), device=device,
                    pool_per_side=max(16, batch * 4))
                concept_transfer_pairing = bool(concept_records and concept_pairs)
            else:
                concept_records, concept_pairs = [], []
            if (not concept_records or not concept_pairs) and concept_groups:
                concept_records, concept_pairs = choice_concept_bridge_batch(
                    concept_groups, rng, max(1, batch // 2))
                concept_transfer_pairing = False
            if (not concept_records or not concept_pairs) and concept_source_records:
                concept_records, concept_pairs = choice_concept_neighborhood_bridge_batch(
                    model, vocab, concept_source_records, rng, max(1, batch // 2),
                    device=device, pool_size=max(16, batch * 4))
                concept_transfer_pairing = False
            if concept_records and concept_pairs:
                concept_txt, _concept_ids = pack(concept_records, vocab, device)
                if concept_transfer_pairing:
                    concept_loss = choice_concept_transfer_bridge_loss(
                        model, concept_txt, concept_records, concept_pairs,
                        margin=choice_concept_bridge_margin)
                else:
                    concept_loss = choice_concept_bridge_loss(
                        model, concept_txt, concept_records, concept_pairs,
                        margin=choice_concept_bridge_margin)
            else:
                concept_loss = torch.tensor(0.0, device=device)
        else:
            concept_loss = torch.tensor(0.0, device=device)
        if choice_concept_prototype_w:
            if concept_groups:
                prototype_records, prototype_items = choice_concept_prototype_batch(
                    concept_groups, rng, max(1, batch // 4), per_group=3)
            else:
                prototype_records, prototype_items = [], []
            if (not prototype_records or not prototype_items) and concept_source_records:
                prototype_records, prototype_items = (
                    choice_concept_neighborhood_prototype_batch(
                        model, vocab, concept_source_records, rng,
                        max(1, batch // 4), per_group=3, device=device,
                        pool_size=max(16, batch * 4)))
            if prototype_records and prototype_items:
                prototype_txt, _prototype_ids = pack(prototype_records, vocab, device)
                prototype_loss = choice_concept_prototype_loss(
                    model, prototype_txt, prototype_records, prototype_items,
                    margin=choice_concept_prototype_margin)
            else:
                prototype_loss = torch.tensor(0.0, device=device)
        else:
            prototype_loss = torch.tensor(0.0, device=device)
        if (choice_self_distill_w and choice_self_distill_model is not None
                and distill_sources):
            distill_records = batch_records(distill_sources, rng, max(1, batch // 2))
            distill_txt, _distill_ids = pack(distill_records, vocab, device)
            distill_loss = choice_self_distill_loss(
                model, choice_self_distill_model, distill_txt, distill_records,
                temperature=choice_self_distill_temperature)
        else:
            distill_loss = torch.tensor(0.0, device=device)
        if (choice_self_rank_distill_w and choice_self_distill_model is not None
                and distill_sources):
            rank_distill_records = batch_records(distill_sources, rng, max(1, batch // 2))
            rank_distill_txt, _rank_distill_ids = pack(
                rank_distill_records, vocab, device)
            rank_distill_loss = choice_self_rank_distill_loss(
                model, choice_self_distill_model, rank_distill_txt,
                rank_distill_records, margin=choice_self_rank_distill_margin)
        else:
            rank_distill_loss = torch.tensor(0.0, device=device)
        if choice_answerability_control_w and control_sources:
            ans_control_records = choice_answerability_control_batch_records(
                control_sources, rng, max(1, batch // 2),
                swap_groups=swap_groups, sides=choice_control_sides)
            if ans_control_records:
                ans_control_txt, _ans_control_ids = pack(ans_control_records, vocab,
                                                         device)
                ans_control_loss = choice_answerability_loss(
                    model, ans_control_txt, ans_control_records,
                    answer_w=0.0, none_w=1.0,
                    margin=choice_answerability_margin)
            else:
                ans_control_loss = torch.tensor(0.0, device=device)
        else:
            ans_control_loss = torch.tensor(0.0, device=device)
        if choice_candidate_answerability_control_w and control_sources:
            cand_ans_control_records = choice_answerability_control_batch_records(
                control_sources, rng, max(1, batch // 2),
                swap_groups=swap_groups, sides=choice_control_sides)
            if cand_ans_control_records:
                cand_ans_control_txt, _cand_ans_control_ids = pack(
                    cand_ans_control_records, vocab, device)
                cand_ans_control_loss = choice_candidate_answerability_loss(
                    model, cand_ans_control_txt, cand_ans_control_records,
                    answer_w=0.0, none_w=1.0,
                    margin=choice_candidate_answerability_margin)
            else:
                cand_ans_control_loss = torch.tensor(0.0, device=device)
        else:
            cand_ans_control_loss = torch.tensor(0.0, device=device)
        if choice_candidate_answerability_contrast_w and control_sources:
            cand_ans_contrast_records, cand_ans_contrast_pairs = choice_control_contrast_batch(
                control_sources, rng, max(1, batch // 2),
                swap_groups=swap_groups, sides=choice_control_sides)
            if cand_ans_contrast_records and cand_ans_contrast_pairs:
                cand_ans_contrast_txt, _cand_ans_contrast_ids = pack(
                    cand_ans_contrast_records, vocab, device)
                cand_ans_contrast_loss = choice_candidate_answerability_contrast_loss(
                    model, cand_ans_contrast_txt, cand_ans_contrast_records,
                    cand_ans_contrast_pairs,
                    margin=choice_candidate_answerability_contrast_margin)
            else:
                cand_ans_contrast_loss = torch.tensor(0.0, device=device)
        else:
            cand_ans_contrast_loss = torch.tensor(0.0, device=device)
        if choice_answerability_contrast_w and control_sources and swap_groups:
            ans_contrast_records, ans_contrast_pairs = choice_answerability_contrast_batch(
                control_sources, rng, max(1, batch // 2),
                swap_groups=swap_groups)
            if ans_contrast_records and ans_contrast_pairs:
                ans_contrast_txt, _ans_contrast_ids = pack(ans_contrast_records,
                                                           vocab, device)
                ans_contrast_loss = choice_answerability_contrast_loss(
                    model, ans_contrast_txt, ans_contrast_records,
                    ans_contrast_pairs, margin=choice_answerability_contrast_margin)
            else:
                ans_contrast_loss = torch.tensor(0.0, device=device)
        else:
            ans_contrast_loss = torch.tensor(0.0, device=device)
        if choice_answerability_pair_w and pair_groups:
            ans_pair_records = choice_pair_batch_records(pair_groups, rng,
                                                         max(1, batch // 2))
            ans_pair_txt, _ans_pair_ids = pack(ans_pair_records, vocab, device)
            ans_pair_loss = choice_answerability_pair_loss(
                model, ans_pair_txt, ans_pair_records,
                margin=choice_answerability_pair_margin)
        else:
            ans_pair_loss = torch.tensor(0.0, device=device)
        if choice_control_contrast_w and control_sources:
            contrast_records, contrast_pairs = choice_control_contrast_batch(
                control_sources, rng, max(1, batch // 2),
                swap_groups=swap_groups, sides=choice_control_sides)
            if contrast_records and contrast_pairs:
                contrast_txt, _contrast_ids = pack(contrast_records, vocab, device)
                contrast_loss = choice_control_contrast_loss(
                    model, contrast_txt, contrast_records, contrast_pairs,
                    margin=choice_control_margin)
            else:
                contrast_loss = torch.tensor(0.0, device=device)
        else:
            contrast_loss = torch.tensor(0.0, device=device)
        if choice_question_context_contrast_w and control_sources and swap_groups:
            qctx_contrast_records, qctx_contrast_pairs = choice_question_context_swap_batch(
                control_sources, rng, max(1, batch // 2),
                swap_groups=swap_groups)
            if qctx_contrast_records and qctx_contrast_pairs:
                qctx_contrast_txt, _qctx_contrast_ids = pack(qctx_contrast_records,
                                                             vocab, device)
                qctx_contrast_loss = choice_question_context_contrast_loss(
                    model, qctx_contrast_txt, qctx_contrast_records,
                    qctx_contrast_pairs, margin=choice_question_context_margin)
            else:
                qctx_contrast_loss = torch.tensor(0.0, device=device)
        else:
            qctx_contrast_loss = torch.tensor(0.0, device=device)
        loss = (decode_w * dec_loss + semantic_w * sem_loss
                + fact_concept_w * concept_fact_loss
                + fact_concept_contrast_w * concept_contrast_loss
                + fact_concept_centroid_w * concept_centroid_loss
                + fact_concept_prototype_w * concept_proto_loss
                + fact_concept_prototype_spread_w * concept_proto_spread_loss
                + fact_concept_state_spread_w * concept_state_spread_loss
                + choice_w * ch_loss
                + choice_final_w * final_loss
                + choice_final_control_w * final_control_loss
                + choice_final_control_contrast_w * final_contrast_loss
                + choice_candidate_replacement_w * replacement_loss
                + choice_candidate_replacement_pair_w * replacement_pair_loss
                + choice_candidate_replacement_binding_w * replacement_binding_loss
                + (choice_candidate_replacement_answerability_binding_w
                   * replacement_ans_binding_loss)
                + choice_positive_anchor_w * positive_anchor_loss
                + choice_concept_bridge_w * concept_loss
                + choice_concept_prototype_w * prototype_loss
                + choice_self_distill_w * distill_loss
                + choice_self_rank_distill_w * rank_distill_loss
                + choice_answerability_w * ans_loss
                + choice_answerability_control_w * ans_control_loss
                + choice_candidate_answerability_w * cand_ans_loss
                + choice_candidate_answerability_control_w * cand_ans_control_loss
                + choice_candidate_answerability_contrast_w * cand_ans_contrast_loss
                + choice_answerability_contrast_w * ans_contrast_loss
                + choice_answerability_pair_w * ans_pair_loss
                + choice_context_w * ctx_loss
                + choice_candidate_context_w * cand_ctx_loss
                + choice_question_context_w * qctx_loss
                + choice_question_context_contrast_w * qctx_contrast_loss
                + choice_pair_w * pair_loss + choice_control_w * control_loss
                + choice_control_contrast_w * contrast_loss)
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
                  f"choice {ch_loss.item():.3f} final {final_loss.item():.3f} "
                  f"final-control {final_control_loss.item():.3f} "
                  f"final-contrast {final_contrast_loss.item():.3f} "
                  f"cand-repl {replacement_loss.item():.3f} "
                  f"cand-repl-pair {replacement_pair_loss.item():.3f} "
                  f"cand-repl-bind {replacement_binding_loss.item():.3f} "
                  f"cand-repl-ans-bind {replacement_ans_binding_loss.item():.3f} "
                  f"pos-anchor {positive_anchor_loss.item():.3f} "
                  f"concept {concept_loss.item():.3f} "
                  f"proto {prototype_loss.item():.3f} "
                  f"distill {distill_loss.item():.3f} "
                  f"rankdistill {rank_distill_loss.item():.3f} "
                  f"ans {ans_loss.item():.3f} "
                  f"ans-control {ans_control_loss.item():.3f} "
                  f"cand-ans {cand_ans_loss.item():.3f} "
                  f"cand-ans-control {cand_ans_control_loss.item():.3f} "
                  f"cand-ans-contrast {cand_ans_contrast_loss.item():.3f} "
                  f"ans-contrast {ans_contrast_loss.item():.3f} "
                  f"ans-pair {ans_pair_loss.item():.3f} "
                  f"ctx {ctx_loss.item():.3f} "
                  f"cand-ctx {cand_ctx_loss.item():.3f} "
                  f"qctx {qctx_loss.item():.3f} "
                  f"qctx-contrast {qctx_contrast_loss.item():.3f} "
                  f"pair {pair_loss.item():.3f} "
                  f"control {control_loss.item():.3f} "
                  f"ctrl-contrast {contrast_loss.item():.3f}", flush=True)
    return model, vocab


def train_model(records, steps=400, batch=32, d=96, layers=3, heads=4,
                text_encoder_arch="transformer", text_encoder_layers=1,
                fact_concept_refine=False, fact_concept_refine_gate_init=-2.0,
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
                decode_w=1.0, choice_w=0.0, choice_answer_w=1.0,
                choice_none_w=1.0, choice_context_w=0.0,
                choice_answer_margin=0.0, choice_none_margin=0.0,
                choice_final_w=0.0, choice_final_control_w=0.0,
                choice_final_margin=0.0,
                choice_final_control_contrast_w=0.0,
                choice_final_control_contrast_margin=0.0,
                choice_candidate_replacement_w=0.0,
                choice_candidate_replacement_margin=0.0,
                choice_candidate_replacement_pair_w=0.0,
                choice_candidate_replacement_pair_margin=0.0,
                choice_candidate_replacement_binding_w=0.0,
                choice_candidate_replacement_binding_margin=0.0,
                choice_candidate_replacement_answerability_binding_w=0.0,
                choice_candidate_replacement_answerability_binding_margin=0.0,
                choice_positive_anchor_w=0.0,
                choice_positive_anchor_margin=0.0,
                choice_concept_bridge_w=0.0,
                choice_concept_bridge_margin=0.0,
                choice_concept_prototype_w=0.0,
                choice_concept_prototype_margin=0.0,
                choice_answerability_w=0.0, choice_answerability_control_w=0.0,
                choice_answerability_margin=0.0,
                choice_answerability_contrast_w=0.0,
                choice_answerability_contrast_margin=0.0,
                choice_candidate_answerability_w=0.0,
                choice_candidate_answerability_control_w=0.0,
                choice_candidate_answerability_margin=0.0,
                choice_candidate_answerability_contrast_w=0.0,
                choice_candidate_answerability_contrast_margin=0.0,
                choice_answerability_pair_w=0.0,
                choice_answerability_pair_margin=0.0,
                choice_candidate_context_w=0.0,
                choice_candidate_context_margin=0.0,
                choice_question_context_w=0.0,
                choice_question_context_contrast_w=0.0,
                choice_question_context_margin=0.0,
                choice_pair_w=0.0, choice_pair_margin=0.0,
                choice_control_w=0.0, choice_control_contrast_w=0.0,
                choice_control_margin=0.0):
    torch.manual_seed(seed)
    vocab = build_vocab(records)
    schema = build_fact_schema(records)
    model = TextFactLM(len(vocab), d=d, layers=layers, heads=heads, pad=vocab.pad,
                       fact_schema=schema,
                       fact_concept_prefix=fact_concept_prefix,
                       text_encoder_arch=text_encoder_arch,
                       text_encoder_layers=text_encoder_layers,
                       fact_concept_refine=fact_concept_refine,
                       fact_concept_refine_gate_init=(
                           fact_concept_refine_gate_init)).to(device)
    return fit_model(model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
                     device=device, log_every=log_every, semantic_w=semantic_w,
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
                     prefix="text", decode_w=decode_w, choice_w=choice_w,
                     choice_answer_w=choice_answer_w,
                     choice_none_w=choice_none_w,
                     choice_answer_margin=choice_answer_margin,
                     choice_none_margin=choice_none_margin,
                     choice_final_w=choice_final_w,
                     choice_final_control_w=choice_final_control_w,
                     choice_final_margin=choice_final_margin,
                     choice_final_control_contrast_w=choice_final_control_contrast_w,
                     choice_final_control_contrast_margin=(
                         choice_final_control_contrast_margin),
                     choice_candidate_replacement_w=choice_candidate_replacement_w,
                     choice_candidate_replacement_margin=(
                         choice_candidate_replacement_margin),
                     choice_candidate_replacement_pair_w=(
                         choice_candidate_replacement_pair_w),
                     choice_candidate_replacement_pair_margin=(
                         choice_candidate_replacement_pair_margin),
                     choice_candidate_replacement_binding_w=(
                         choice_candidate_replacement_binding_w),
                     choice_candidate_replacement_binding_margin=(
                         choice_candidate_replacement_binding_margin),
                     choice_candidate_replacement_answerability_binding_w=(
                         choice_candidate_replacement_answerability_binding_w),
                     choice_candidate_replacement_answerability_binding_margin=(
                         choice_candidate_replacement_answerability_binding_margin),
                     choice_positive_anchor_w=choice_positive_anchor_w,
                     choice_positive_anchor_margin=choice_positive_anchor_margin,
                     choice_concept_bridge_w=choice_concept_bridge_w,
                     choice_concept_bridge_margin=choice_concept_bridge_margin,
                     choice_concept_prototype_w=choice_concept_prototype_w,
                     choice_concept_prototype_margin=choice_concept_prototype_margin,
                     choice_answerability_w=choice_answerability_w,
                     choice_answerability_control_w=choice_answerability_control_w,
                     choice_answerability_margin=choice_answerability_margin,
                     choice_answerability_contrast_w=choice_answerability_contrast_w,
                     choice_answerability_contrast_margin=(
                         choice_answerability_contrast_margin),
                     choice_candidate_answerability_w=choice_candidate_answerability_w,
                     choice_candidate_answerability_control_w=(
                         choice_candidate_answerability_control_w),
                     choice_candidate_answerability_margin=(
                         choice_candidate_answerability_margin),
                     choice_candidate_answerability_contrast_w=(
                         choice_candidate_answerability_contrast_w),
                     choice_candidate_answerability_contrast_margin=(
                         choice_candidate_answerability_contrast_margin),
                     choice_answerability_pair_w=choice_answerability_pair_w,
                     choice_answerability_pair_margin=choice_answerability_pair_margin,
                     choice_context_w=choice_context_w,
                     choice_candidate_context_w=choice_candidate_context_w,
                     choice_candidate_context_margin=choice_candidate_context_margin,
                     choice_question_context_w=choice_question_context_w,
                     choice_question_context_contrast_w=choice_question_context_contrast_w,
                     choice_question_context_margin=choice_question_context_margin,
                     choice_pair_w=choice_pair_w,
                     choice_pair_margin=choice_pair_margin,
                     choice_control_w=choice_control_w,
                     choice_control_contrast_w=choice_control_contrast_w,
                     choice_control_margin=choice_control_margin)


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


def choice_head_eval(model, vocab, records, device=DEV, n=0, seed=0):
    all_eval = [r for r in records if r.split == "eval" and qa_choice_target(r) is not None]
    if n < 0:
        return {"fact_value_acc": 0.0, "n_facts": 0, "n_records": 0,
                "sampled": False, "skipped": True}
    selected = list(all_eval)
    sampled = bool(n and n < len(selected))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(selected), size=n, replace=False)
        selected = [selected[int(i)] for i in idx]
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(selected), 64):
            batch = selected[off:off + 64]
            txt, _ids = pack(batch, vocab, device)
            for rec, ids, logits in model.choice_logits(txt, batch):
                target = qa_choice_target(rec)
                if target is None or logits is None or target not in ids:
                    continue
                correct += int(ids[int(logits.argmax(-1))] == target)
                total += 1
    return {"fact_value_acc": correct / max(1, total),
            "n_facts": total,
            "n_records": len(selected),
            "sampled": sampled,
            "skipped": False}


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


def choice_record_errors(model, vocab, records, device=DEV, n=0, seed=0):
    """Return choice-QA records whose candidate head currently predicts the wrong target."""
    errors, _correct, report = choice_record_outcomes(
        model, vocab, records, device=device, n=n, seed=seed)
    return errors, report


def choice_record_outcomes(model, vocab, records, device=DEV, n=0, seed=0):
    """Return currently wrong and right choice-QA records under the candidate head."""
    candidates = [r for r in records if qa_choice_target(r) is not None]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    errors = []
    correct = []
    by_kind = {}
    n_errors = n_choices = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt, _ids = pack(batch, vocab, device)
            for rec, ids, logits in model.choice_logits(txt, batch):
                target = qa_choice_target(rec)
                if target is None:
                    continue
                wrong = True
                if logits is not None and target in ids:
                    pred = ids[int(logits.argmax(-1))]
                    wrong = pred != target
                kind_row = by_kind.setdefault(
                    rec.kind, {"records": 0, "errors": 0, "correct": 0})
                kind_row["records"] += 1
                if wrong:
                    errors.append(rec)
                    kind_row["errors"] += 1
                    n_errors += 1
                else:
                    correct.append(rec)
                    kind_row["correct"] += 1
                n_choices += 1
    report = {"n_records": len(candidates),
              "sampled": sampled,
              "n_error_records": len(errors),
              "n_correct_records": len(correct),
              "n_facts": n_choices,
              "n_errors": n_errors,
              "choice_error_rate": n_errors / max(1, n_choices),
              "by_kind": by_kind}
    return errors, correct, report


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


def choice_neighbor_records_per_kind(model, vocab, hard_records, correct_records, rng,
                                     per_kind, device=DEV, pool_per_kind=0):
    """Select correct QA records nearest to current hard records in model concept space."""
    if per_kind <= 0:
        return [], {}, {"adaptive": True, "reason": "disabled"}
    hard = [r for r in hard_records if qa_choice_target(r) not in (None, "none")]
    candidates = [r for r in correct_records if qa_choice_target(r) not in (None, "none")]
    if not hard:
        rows, counts = sample_records_per_kind(correct_records, rng, per_kind)
        return rows, counts, {"adaptive": False, "reason": "no_hard_choice_records"}
    if not candidates:
        return [], {}, {"adaptive": False, "reason": "no_correct_choice_records"}
    pool_per_kind = max(0, int(pool_per_kind))
    if pool_per_kind:
        pooled = []
        by_kind = {}
        for rec in candidates:
            by_kind.setdefault(rec.kind, []).append(rec)
        for rows in by_kind.values():
            n = min(pool_per_kind, len(rows))
            if n < len(rows):
                idx = rng.choice(len(rows), size=n, replace=False)
                pooled.extend(rows[int(i)] for i in idx)
            else:
                pooled.extend(rows)
        candidates = pooled
    pool = []
    seen_ids = set()
    for rec in hard + candidates:
        if rec.rec_id in seen_ids:
            continue
        seen_ids.add(rec.rec_id)
        pool.append(rec)
    txt, _ids = pack(pool, vocab, device)
    model.eval()
    with torch.no_grad():
        vectors = choice_candidate_concept_vectors(model, txt, pool)
    hard_vecs = []
    for rec in hard:
        target = qa_choice_target(rec)
        row = vectors.get(rec.rec_id)
        if row and target in row:
            hard_vecs.append(row[target].detach())
    if not hard_vecs:
        rows, counts = sample_records_per_kind(correct_records, rng, per_kind)
        return rows, counts, {"adaptive": False, "reason": "no_hard_vectors"}
    scored = {}
    for rec in candidates:
        target = qa_choice_target(rec)
        row = vectors.get(rec.rec_id)
        if not row or target not in row:
            continue
        vec = row[target].detach()
        score = torch.stack([(vec * hard_vec).sum() for hard_vec in hard_vecs]).max()
        scored.setdefault(rec.kind, []).append((float(score.cpu()), rec))
    if not scored:
        rows, counts = sample_records_per_kind(correct_records, rng, per_kind)
        return rows, counts, {"adaptive": False, "reason": "no_correct_vectors"}
    out = []
    counts = {}
    score_means = {}
    for kind, rows in sorted(scored.items()):
        order = rng.permutation(len(rows)) if len(rows) > 1 else np.arange(len(rows))
        shuffled = [rows[int(i)] for i in order]
        shuffled.sort(key=lambda item: item[0], reverse=True)
        picked = [rec for _score, rec in shuffled[:min(int(per_kind), len(shuffled))]]
        if picked:
            out.extend(picked)
            counts[kind] = len(picked)
            score_means[kind] = float(np.mean([score for score, _rec in shuffled[:len(picked)]]))
    return out, counts, {
        "adaptive": True,
        "reason": "nearest_hard_choice_concept",
        "hard_choice_records": len(hard),
        "hard_vectors": len(hard_vecs),
        "candidate_correct_records": len(candidates),
        "selected_records": len(out),
        "selected_by_kind": counts,
        "selected_score_mean_by_kind": score_means,
    }


def choice_fragile_correct_records_per_kind(model, vocab, correct_records, rng, per_kind,
                                            device=DEV, pool_per_kind=0):
    """Select currently-correct QA records with the smallest own-model target margin."""
    if per_kind <= 0:
        return [], {}, {"fragile": True, "reason": "disabled"}
    candidates = [r for r in correct_records if qa_choice_target(r) is not None]
    if not candidates:
        return [], {}, {"fragile": False, "reason": "no_correct_choice_records"}
    pool_per_kind = max(0, int(pool_per_kind))
    if pool_per_kind:
        pooled = []
        by_kind = {}
        for rec in candidates:
            by_kind.setdefault(rec.kind, []).append(rec)
        for rows in by_kind.values():
            n = min(pool_per_kind, len(rows))
            if n < len(rows):
                idx = rng.choice(len(rows), size=n, replace=False)
                pooled.extend(rows[int(i)] for i in idx)
            else:
                pooled.extend(rows)
        candidates = pooled
    scored = {}
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt, _ids = pack(batch, vocab, device)
            for rec, ids, logits in model.choice_logits(txt, batch):
                target = qa_choice_target(rec)
                if logits is None or target is None or target not in ids:
                    continue
                target_idx = ids.index(target)
                if int(logits.argmax(-1)) != target_idx:
                    continue
                if logits.numel() < 2:
                    margin = float("inf")
                else:
                    mask = torch.ones(logits.shape[0], dtype=torch.bool, device=device)
                    mask[target_idx] = False
                    margin = float((logits[target_idx] - logits[mask].max()).cpu())
                scored.setdefault(rec.kind, []).append((margin, rec))
    if not scored:
        return [], {}, {"fragile": False, "reason": "no_scored_correct_records"}
    out = []
    counts = {}
    margin_means = {}
    margin_mins = {}
    for kind, rows in sorted(scored.items()):
        order = rng.permutation(len(rows)) if len(rows) > 1 else np.arange(len(rows))
        shuffled = [rows[int(i)] for i in order]
        shuffled.sort(key=lambda item: item[0])
        picked_rows = shuffled[:min(int(per_kind), len(shuffled))]
        picked = [rec for _margin, rec in picked_rows]
        if picked:
            out.extend(picked)
            counts[kind] = len(picked)
            margins = [margin for margin, _rec in picked_rows]
            margin_means[kind] = float(np.mean(margins))
            margin_mins[kind] = float(np.min(margins))
    return out, counts, {
        "fragile": True,
        "reason": "lowest_correct_choice_margin",
        "candidate_correct_records": len(candidates),
        "selected_records": len(out),
        "selected_by_kind": counts,
        "selected_margin_mean_by_kind": margin_means,
        "selected_margin_min_by_kind": margin_mins,
    }


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
        choice = choice_head_eval(model, vocab, rows, device=device, n=n,
                                  seed=seed + 4001 * i)
        out[name] = {"n": len(rows),
                     "n_records": teacher["n_records"],
                     "sampled": teacher["sampled"],
                     "teacher_forced_fact_value_acc": teacher["fact_value_acc"],
                     "semantic_fact_value_acc": semantic["fact_value_acc"],
                     "fact_concept_fact_value_acc": fact_concept["fact_value_acc"],
                     "choice_head_fact_value_acc": choice["fact_value_acc"],
                     "choice_head_records": choice["n_records"]}
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


def _is_qa_record(rec):
    return any(pred in ("answer_token", "length", "choice")
               for _slot, pred, _val in rec.facts)


def _is_qa_negative_record(rec):
    return isinstance(rec.meta, dict) and bool(rec.meta.get("negative"))


def _qa_ablation_record(rec, side):
    parts = _qa_parts(rec)
    if parts is None:
        return None
    context, question, tail = parts
    if side == "question":
        ntoks = question + tail
    elif side == "context":
        ntoks = context + tail
    else:
        raise ValueError(side)
    if not ntoks:
        return None
    return TextRecord(rec_id=f"{rec.rec_id}:{side}", split=rec.split, tokens=ntoks,
                      facts=rec.facts, group=rec.group, kind=f"{rec.kind}:{side}",
                      base_id=rec.rec_id, changed=rec.changed, meta=rec.meta)


def _qa_parts(rec):
    toks = list(rec.tokens)
    try:
        q_at = toks.index("question")
    except ValueError:
        return None
    try:
        choices_at = toks.index("choices", q_at + 1)
    except ValueError:
        choices_at = len(toks)
    if q_at + 2 > choices_at:
        return None
    return tuple(toks[:q_at]), tuple(toks[q_at:choices_at]), tuple(toks[choices_at:])


def _qa_context_key(rec):
    para = rec.meta.get("paragraph_id") if isinstance(rec.meta, dict) else None
    if para:
        return ("paragraph", para)
    parts = _qa_parts(rec)
    if parts is None:
        return None
    return ("context", parts[0])


def _qa_question_swap_record(rec, donor):
    rec_parts = _qa_parts(rec)
    donor_parts = _qa_parts(donor)
    if rec_parts is None or donor_parts is None:
        return None
    context, _question, tail = rec_parts
    _donor_context, donor_question, _donor_tail = donor_parts
    toks = context + donor_question + tail
    return TextRecord(rec_id=f"{rec.rec_id}:question_swap:{donor.rec_id}",
                      split=rec.split, tokens=toks, facts=rec.facts,
                      group=rec.group, kind=f"{rec.kind}:question_swap",
                      base_id=rec.rec_id, changed=rec.changed, meta=rec.meta)


def qa_ablation_eval(model, vocab, records, device=DEV, n=0, seed=0):
    all_eval = [r for r in records if r.split == "eval" and _is_qa_record(r)
                and not _is_qa_negative_record(r)]
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
    question_only = [r for r in (_qa_ablation_record(rec, "question")
                                 for rec in eval_records_) if r is not None]
    context_only = [r for r in (_qa_ablation_record(rec, "context")
                                for rec in eval_records_) if r is not None]
    full_teacher = teacher_forced_eval(model, vocab, eval_records_,
                                       device=device)["fact_value_acc"]
    q_teacher = (teacher_forced_eval(model, vocab, question_only,
                                     device=device)["fact_value_acc"]
                 if question_only else 0.0)
    c_teacher = (teacher_forced_eval(model, vocab, context_only,
                                     device=device)["fact_value_acc"]
                 if context_only else 0.0)
    full_sem = semantic_fact_eval(model, vocab, eval_records_, device=device)["fact_value_acc"]
    q_sem = (semantic_fact_eval(model, vocab, question_only,
                                device=device)["fact_value_acc"]
             if question_only else 0.0)
    c_sem = (semantic_fact_eval(model, vocab, context_only,
                                device=device)["fact_value_acc"]
             if context_only else 0.0)
    full_choice = choice_head_eval(model, vocab, eval_records_, device=device)["fact_value_acc"]
    q_choice = (choice_head_eval(model, vocab, question_only,
                                 device=device)["fact_value_acc"]
                if question_only else 0.0)
    c_choice = (choice_head_eval(model, vocab, context_only,
                                 device=device)["fact_value_acc"]
                if context_only else 0.0)
    return {"n": len(eval_records_),
            "sampled": sampled,
            "skipped": False,
            "question_only_records": len(question_only),
            "context_only_records": len(context_only),
            "full_fact_value_acc": full_teacher,
            "question_only_fact_value_acc": q_teacher,
            "context_only_fact_value_acc": c_teacher,
            "full_minus_question_only": full_teacher - q_teacher,
            "full_minus_context_only": full_teacher - c_teacher,
            "semantic_full_fact_value_acc": full_sem,
            "semantic_question_only_fact_value_acc": q_sem,
            "semantic_context_only_fact_value_acc": c_sem,
            "semantic_full_minus_question_only": full_sem - q_sem,
            "semantic_full_minus_context_only": full_sem - c_sem,
            "choice_full_fact_value_acc": full_choice,
            "choice_question_only_fact_value_acc": q_choice,
            "choice_context_only_fact_value_acc": c_choice,
            "choice_full_minus_question_only": full_choice - q_choice,
            "choice_full_minus_context_only": full_choice - c_choice}


def qa_question_swap_eval(model, vocab, records, device=DEV, n=0, seed=0):
    all_eval = [r for r in records if r.split == "eval" and _is_qa_record(r)
                and not _is_qa_negative_record(r)]
    if n < 0:
        return {"n": 0, "sampled": False, "skipped": True}
    groups = {}
    for rec in all_eval:
        key = _qa_context_key(rec)
        if key is not None:
            groups.setdefault(key, []).append(rec)
    pairs = []
    for rows in groups.values():
        usable = []
        for rec in rows:
            gold = tuple(rec.facts)
            if any(tuple(other.facts) != gold for other in rows):
                usable.append(rec)
        for rec in usable:
            donors = [other for other in rows if tuple(other.facts) != tuple(rec.facts)]
            if donors:
                pairs.append((rec, donors))
    if not pairs:
        return {"n": 0, "sampled": False, "skipped": False}
    sampled = bool(n and n < len(pairs))
    rng = np.random.default_rng(seed)
    if sampled:
        idx = rng.choice(len(pairs), size=n, replace=False)
        pairs = [pairs[int(i)] for i in idx]
    original = []
    swapped = []
    for rec, donors in pairs:
        donor = donors[int(rng.integers(len(donors)))]
        swap = _qa_question_swap_record(rec, donor)
        if swap is not None:
            original.append(rec)
            swapped.append(swap)
    if not swapped:
        return {"n": 0, "sampled": sampled, "skipped": False}
    full_teacher = teacher_forced_eval(model, vocab, original,
                                       device=device)["fact_value_acc"]
    swap_teacher = teacher_forced_eval(model, vocab, swapped,
                                       device=device)["fact_value_acc"]
    full_sem = semantic_fact_eval(model, vocab, original, device=device)["fact_value_acc"]
    swap_sem = semantic_fact_eval(model, vocab, swapped, device=device)["fact_value_acc"]
    full_choice = choice_head_eval(model, vocab, original, device=device)["fact_value_acc"]
    swap_choice = choice_head_eval(model, vocab, swapped, device=device)["fact_value_acc"]
    return {"n": len(swapped),
            "sampled": sampled,
            "skipped": False,
            "full_fact_value_acc": full_teacher,
            "question_swap_fact_value_acc": swap_teacher,
            "full_minus_question_swap": full_teacher - swap_teacher,
            "semantic_full_fact_value_acc": full_sem,
            "semantic_question_swap_fact_value_acc": swap_sem,
            "semantic_full_minus_question_swap": full_sem - swap_sem,
            "choice_full_fact_value_acc": full_choice,
            "choice_question_swap_fact_value_acc": swap_choice,
            "choice_full_minus_question_swap": full_choice - swap_choice}


def qa_candidate_replacement_eval(model, vocab, records, device=DEV, n=0, seed=0):
    all_eval = [r for r in records if r.split == "eval" and _is_qa_record(r)
                and not _is_qa_negative_record(r)]
    if n < 0:
        return {"n": 0, "sampled": False, "skipped": True}
    if not all_eval:
        return {"n": 0, "sampled": False, "skipped": False}
    eval_records_ = all_eval
    sampled = bool(n and n < len(eval_records_))
    rng = np.random.default_rng(seed)
    if sampled:
        idx = rng.choice(len(eval_records_), size=n, replace=False)
        eval_records_ = [eval_records_[int(i)] for i in idx]
    original = []
    replaced = []
    for rec in eval_records_:
        repl = _choice_candidate_replacement_record(rec, rng)
        if repl is not None:
            original.append(rec)
            replaced.append(repl)
    if not replaced:
        return {"n": 0, "sampled": sampled, "skipped": False}
    full_choice = choice_head_eval(model, vocab, original, device=device)["fact_value_acc"]
    repl_choice = choice_head_eval(model, vocab, replaced, device=device)["fact_value_acc"]
    return {"n": len(replaced),
            "sampled": sampled,
            "skipped": False,
            "full_choice_fact_value_acc": full_choice,
            "candidate_replacement_none_acc": repl_choice,
            "candidate_replacement_error_rate": 1.0 - repl_choice,
            "choice_binding_score": min(full_choice, repl_choice)}


@torch.no_grad()
def qa_candidate_replacement_binding_eval(model, vocab, records, device=DEV, n=0,
                                          seed=0):
    all_eval = [r for r in records if r.split == "eval" and _is_qa_record(r)
                and not _is_qa_negative_record(r)]
    if n < 0:
        return {"n": 0, "sampled": False, "skipped": True}
    if not all_eval:
        return {"n": 0, "sampled": False, "skipped": False}
    eval_records_ = all_eval
    sampled = bool(n and n < len(eval_records_))
    rng = np.random.default_rng(seed)
    if sampled:
        idx = rng.choice(len(eval_records_), size=n, replace=False)
        eval_records_ = [eval_records_[int(i)] for i in idx]
    rows = []
    pair_ids = []
    for rec in eval_records_:
        target = qa_choice_target(rec)
        repl = _choice_candidate_replacement_record(rec, rng)
        if repl is None or target is None or target == "none":
            continue
        pair_idx = len(pair_ids)
        full_id = f"{rec.rec_id}:candidate_binding_full:{pair_idx}"
        repl_id = f"{rec.rec_id}:candidate_binding_repl:{pair_idx}"
        rows.extend([
            TextRecord(rec_id=full_id, split=rec.split, tokens=rec.tokens,
                       facts=rec.facts, group=rec.group, kind=rec.kind,
                       base_id=rec.base_id, changed=rec.changed, meta=rec.meta),
            TextRecord(rec_id=repl_id, split=rec.split, tokens=repl.tokens,
                       facts=repl.facts, group=rec.group, kind=repl.kind,
                       base_id=rec.rec_id, changed=repl.changed, meta=repl.meta),
        ])
        pair_ids.append((full_id, repl_id, target))
    if not rows or not pair_ids:
        return {"n": 0, "sampled": sampled, "skipped": False}
    txt, _ids = pack(rows, vocab, device)
    scores = choice_candidate_replacement_binding_scores(model, txt, rows, pair_ids)
    if not scores:
        return {"n": 0, "sampled": sampled, "skipped": False}
    full_scores = [float(full.detach().cpu()) for full, _repl in scores]
    repl_scores = [float(repl.detach().cpu()) for _full, repl in scores]
    drops = [full - repl for full, repl in zip(full_scores, repl_scores)]
    return {"n": len(scores),
            "sampled": sampled,
            "skipped": False,
            "full_target_logit": float(np.mean(full_scores)),
            "replacement_target_logit": float(np.mean(repl_scores)),
            "target_logit_drop": float(np.mean(drops)),
            "target_logit_drop_rate": float(np.mean([d > 0.0 for d in drops]))}


@torch.no_grad()
def qa_candidate_replacement_answerability_binding_eval(model, vocab, records,
                                                        device=DEV, n=0, seed=0):
    all_eval = [r for r in records if r.split == "eval" and _is_qa_record(r)
                and not _is_qa_negative_record(r)]
    if n < 0:
        return {"n": 0, "sampled": False, "skipped": True}
    if not all_eval:
        return {"n": 0, "sampled": False, "skipped": False}
    eval_records_ = all_eval
    sampled = bool(n and n < len(eval_records_))
    rng = np.random.default_rng(seed)
    if sampled:
        idx = rng.choice(len(eval_records_), size=n, replace=False)
        eval_records_ = [eval_records_[int(i)] for i in idx]
    rows = []
    pair_ids = []
    for rec in eval_records_:
        target = qa_choice_target(rec)
        repl = _choice_candidate_replacement_record(rec, rng)
        if repl is None or target is None or target == "none":
            continue
        pair_idx = len(pair_ids)
        full_id = f"{rec.rec_id}:candidate_ans_binding_full:{pair_idx}"
        repl_id = f"{rec.rec_id}:candidate_ans_binding_repl:{pair_idx}"
        rows.extend([
            TextRecord(rec_id=full_id, split=rec.split, tokens=rec.tokens,
                       facts=rec.facts, group=rec.group, kind=rec.kind,
                       base_id=rec.base_id, changed=rec.changed, meta=rec.meta),
            TextRecord(rec_id=repl_id, split=rec.split, tokens=repl.tokens,
                       facts=repl.facts, group=rec.group, kind=repl.kind,
                       base_id=rec.rec_id, changed=repl.changed, meta=repl.meta),
        ])
        pair_ids.append((full_id, repl_id, target))
    if not rows or not pair_ids:
        return {"n": 0, "sampled": sampled, "skipped": False}
    txt, _ids = pack(rows, vocab, device)
    scores = choice_candidate_replacement_answerability_binding_scores(
        model, txt, rows, pair_ids)
    if not scores:
        return {"n": 0, "sampled": sampled, "skipped": False}
    full_scores = [float(full.detach().cpu()) for full, _repl in scores]
    repl_scores = [float(repl.detach().cpu()) for _full, repl in scores]
    drops = [full - repl for full, repl in zip(full_scores, repl_scores)]
    return {"n": len(scores),
            "sampled": sampled,
            "skipped": False,
            "full_target_cover": float(np.mean(full_scores)),
            "replacement_target_cover": float(np.mean(repl_scores)),
            "target_cover_drop": float(np.mean(drops)),
            "target_cover_drop_rate": float(np.mean([d > 0.0 for d in drops]))}


@torch.no_grad()
def qa_concept_discovery_eval(model, vocab, records, device=DEV, n=0, seed=0):
    all_eval = [r for r in records if r.split == "eval" and _is_qa_record(r)
                and not _is_qa_negative_record(r)]
    if n < 0:
        return {"n": 0, "sampled": False, "skipped": True}
    groups = choice_concept_groups(all_eval)
    if not groups:
        return {"n": 0, "sampled": False, "skipped": False,
                "concept_groups": 0}
    possible_pairs = sum(len(group) * (len(group) - 1) // 2 for group in groups)
    if possible_pairs <= 0:
        return {"n": 0, "sampled": False, "skipped": False,
                "concept_groups": len(groups)}
    raw_pairs = [(left, right)
                 for group in groups
                 for i, left in enumerate(group)
                 for right in group[i + 1:]]
    pair_n = min(int(n), possible_pairs) if n else min(possible_pairs, 512)
    sampled = pair_n < possible_pairs
    rng = np.random.default_rng(seed)
    if sampled:
        idx = rng.choice(len(raw_pairs), size=pair_n, replace=False)
        raw_pairs = [raw_pairs[int(i)] for i in idx]
    else:
        raw_pairs = raw_pairs[:pair_n]
    bridge_records = []
    pair_ids = []
    for pair_idx, (left, right) in enumerate(raw_pairs):
        left_target = qa_choice_target(left)
        right_target = qa_choice_target(right)
        if left_target is None or right_target is None:
            continue
        left_id = f"{left.rec_id}:concept_eval_left:{pair_idx}"
        right_id = f"{right.rec_id}:concept_eval_right:{pair_idx}"
        bridge_records.extend([
            TextRecord(rec_id=left_id, split=left.split, tokens=left.tokens,
                       facts=left.facts, group=left.group, kind=left.kind,
                       base_id=left.base_id, changed=left.changed, meta=left.meta),
            TextRecord(rec_id=right_id, split=right.split, tokens=right.tokens,
                       facts=right.facts, group=right.group, kind=right.kind,
                       base_id=right.base_id, changed=right.changed, meta=right.meta),
        ])
        pair_ids.append((left_id, right_id, left_target, right_target))
    if not bridge_records or not pair_ids:
        return {"n": 0, "sampled": sampled, "skipped": False,
                "concept_groups": len(groups)}
    txt, _ids = pack(bridge_records, vocab, device)
    scored = choice_concept_bridge_scores(model, txt, bridge_records, pair_ids)
    if not scored:
        return {"n": 0, "sampled": sampled, "skipped": False,
                "concept_groups": len(groups)}
    positives = [float(pos.detach().cpu()) for pos, _neg in scored]
    negatives = [float(neg.detach().cpu()) for _pos, neg in scored]
    pos_mean = float(np.mean(positives)) if positives else 0.0
    neg_mean = float(np.mean(negatives)) if negatives else 0.0
    return {"n": len(scored),
            "sampled": sampled,
            "skipped": False,
            "concept_groups": len(groups),
            "possible_pairs": int(possible_pairs),
            "positive_similarity": pos_mean,
            "negative_similarity": neg_mean,
            "discovery_margin": pos_mean - neg_mean}


@torch.no_grad()
def qa_concept_prototype_eval(model, vocab, records, device=DEV, n=0, seed=0):
    all_eval = [r for r in records if r.split == "eval" and _is_qa_record(r)
                and not _is_qa_negative_record(r)]
    if n < 0:
        return {"n": 0, "sampled": False, "skipped": True}
    groups = choice_concept_groups(all_eval)
    if not groups:
        return {"n": 0, "sampled": False, "skipped": False,
                "concept_groups": 0}
    rng = np.random.default_rng(seed)
    group_n = min(len(groups), int(n)) if n else min(len(groups), 128)
    sampled = group_n < len(groups)
    prototype_records, prototype_items = choice_concept_prototype_batch(
        groups, rng, group_n, per_group=4)
    if not prototype_records or not prototype_items:
        return {"n": 0, "sampled": sampled, "skipped": False,
                "concept_groups": len(groups)}
    txt, _ids = pack(prototype_records, vocab, device)
    scored = choice_concept_prototype_scores(model, txt, prototype_records,
                                             prototype_items)
    if not scored:
        return {"n": 0, "sampled": sampled, "skipped": False,
                "concept_groups": len(groups)}
    positives = [float(pos.detach().cpu()) for pos, _neg in scored]
    negatives = [float(neg.detach().cpu()) for _pos, neg in scored]
    margins = [pos - neg for pos, neg in zip(positives, negatives)]
    return {"n": len(scored),
            "sampled": sampled,
            "skipped": False,
            "concept_groups": len(groups),
            "prototype_records": len(prototype_records),
            "positive_similarity": float(np.mean(positives)),
            "negative_similarity": float(np.mean(negatives)),
            "prototype_margin": float(np.mean(margins)),
            "prototype_win_rate": float(np.mean([m > 0.0 for m in margins]))}


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
    fact_concept = fact_concept_eval(model, vocab, records, device=device, n=fact_n,
                                     seed=seed + 11)
    fact_concept_geometry = fact_concept_geometry_eval(
        model, vocab, records, device=device, n=fact_n, seed=seed + 12)
    choice_head = choice_head_eval(model, vocab, records, device=device, n=fact_n,
                                   seed=seed + 11)
    free = free_eval(model, vocab, records, device=device, max_new=max_new, n=free_n,
                     seed=seed)
    para = paraphrase_eval(model, vocab, records, device=device, max_new=max_new,
                           n_groups=paraphrase_n, seed=seed + 1)
    cf = counterfactual_eval(model, vocab, records, device=device, max_new=max_new,
                             n=counterfactual_n, seed=seed + 2)
    artifact = nli_artifact_eval(model, vocab, records, teacher["fact_value_acc"],
                                 device=device, n=artifact_n, seed=seed + 13)
    qa_control = qa_ablation_eval(model, vocab, records, device=device, n=artifact_n,
                                  seed=seed + 17)
    qa_swap = qa_question_swap_eval(model, vocab, records, device=device, n=artifact_n,
                                    seed=seed + 23)
    qa_repl = qa_candidate_replacement_eval(model, vocab, records, device=device,
                                            n=artifact_n, seed=seed + 29)
    qa_repl_binding = qa_candidate_replacement_binding_eval(
        model, vocab, records, device=device, n=artifact_n, seed=seed + 30)
    qa_repl_ans_binding = qa_candidate_replacement_answerability_binding_eval(
        model, vocab, records, device=device, n=artifact_n, seed=seed + 31)
    qa_concept = qa_concept_discovery_eval(model, vocab, records, device=device,
                                           n=artifact_n, seed=seed + 32)
    qa_proto = qa_concept_prototype_eval(model, vocab, records, device=device,
                                         n=artifact_n, seed=seed + 33)
    by_kind = bucket_fact_eval(model, vocab, records, device=device, n=kind_fact_n,
                               seed=seed + 19)
    by_kind_free = (bucket_free_eval(model, vocab, records, device=device, max_new=max_new,
                                     n=kind_free_n, seed=seed + 3)
                    if kind_free_n else {})
    gate = (teacher["fact_value_acc"] >= 0.80 and semantic["fact_value_acc"] >= 0.80
            and (choice_head.get("skipped") or choice_head["n_records"] == 0
                 or choice_head["fact_value_acc"] >= 0.80)
            and (free.get("skipped") or free["f1"] >= 0.80)
            and (para.get("skipped") or para["n_groups"] == 0 or para["consistent"] >= 0.80)
            and (cf.get("skipped") or cf["n"] == 0 or cf["f1"] >= 0.80)
            and (artifact.get("skipped") or artifact["n"] == 0
                 or artifact["full_minus_hypothesis_only"] >= 0.05)
            and (qa_control.get("skipped") or qa_control["n"] == 0
                 or (qa_control["full_minus_question_only"] >= 0.05
                     and qa_control["full_minus_context_only"] >= 0.05))
            and (qa_swap.get("skipped") or qa_swap["n"] == 0
                 or qa_swap["full_minus_question_swap"] >= 0.05)
            and (choice_head.get("skipped") or choice_head["n_records"] == 0
                 or qa_control.get("skipped") or qa_control["n"] == 0
                 or (qa_control["choice_full_minus_question_only"] >= 0.05
                     and qa_control["choice_full_minus_context_only"] >= 0.05))
            and (choice_head.get("skipped") or choice_head["n_records"] == 0
                 or qa_swap.get("skipped") or qa_swap["n"] == 0
                 or qa_swap["choice_full_minus_question_swap"] >= 0.05)
            and (choice_head.get("skipped") or choice_head["n_records"] == 0
                 or qa_repl.get("skipped") or qa_repl["n"] == 0
                 or qa_repl["candidate_replacement_none_acc"] >= 0.55)
            and (choice_head.get("skipped") or choice_head["n_records"] == 0
                 or qa_repl_binding.get("skipped") or qa_repl_binding["n"] == 0
                 or qa_repl_binding["target_logit_drop"] >= 0.05)
            and (choice_head.get("skipped") or choice_head["n_records"] == 0
                 or qa_repl_ans_binding.get("skipped")
                 or qa_repl_ans_binding["n"] == 0
                 or qa_repl_ans_binding["target_cover_drop"] >= 0.05)
            and (choice_head.get("skipped") or choice_head["n_records"] == 0
                 or qa_concept.get("skipped") or qa_concept["n"] == 0
                 or qa_concept["discovery_margin"] >= 0.05)
            and (choice_head.get("skipped") or choice_head["n_records"] == 0
                 or qa_proto.get("skipped") or qa_proto["n"] == 0
                 or qa_proto["prototype_margin"] >= 0.05))
    return {"teacher_forced": teacher, "free_decode": free,
            "semantic_head": semantic,
            "fact_concept_head": fact_concept,
            "fact_concept_geometry": fact_concept_geometry,
            "choice_head": choice_head,
            "by_kind": by_kind,
            "free_decode_by_kind": by_kind_free,
            "paraphrase_consistency": para, "counterfactual": cf,
            "nli_artifact_control": artifact,
            "qa_ablation_control": qa_control,
            "qa_question_swap_control": qa_swap,
            "qa_candidate_replacement_control": qa_repl,
            "qa_candidate_replacement_binding_control": qa_repl_binding,
            "qa_candidate_replacement_answerability_binding_control": (
                qa_repl_ans_binding),
            "qa_concept_discovery_control": qa_concept,
            "qa_concept_prototype_control": qa_proto,
            "gate_thresholds": {"fact_value_acc": 0.80, "free_f1": 0.80,
                                "semantic_fact_value_acc": 0.80,
                                "fact_concept_fact_value_acc": 0.80,
                                "fact_concept_geometry_margin": 0.05,
                                "choice_head_fact_value_acc": 0.80,
                                "paraphrase_consistent": 0.80,
                                "counterfactual_f1": 0.80,
                                "nli_full_minus_hypothesis_only": 0.05,
                                "qa_full_minus_question_only": 0.05,
                                "qa_full_minus_context_only": 0.05,
                                "qa_full_minus_question_swap": 0.05,
                                "qa_choice_full_minus_question_only": 0.05,
                                "qa_choice_full_minus_context_only": 0.05,
                                "qa_choice_full_minus_question_swap": 0.05,
                                "qa_candidate_replacement_none_acc": 0.55,
                                "qa_candidate_replacement_target_logit_drop": 0.05,
                                "qa_candidate_replacement_target_cover_drop": 0.05,
                                "qa_concept_discovery_margin": 0.05,
                                "qa_concept_prototype_margin": 0.05},
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
                           ckpt.get("fact_concept_refine_gate_init", -2.0))).to(device)
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
                           ckpt.get("fact_concept_refine_gate_init", -2.0))).to(device)
    copy_pretrained_text_weights(src_model, src_vocab, model, vocab)
    model.eval()
    return model, vocab, ckpt


def checkpoint_payload(model, vocab, d, layers, heads, report):
    return {"state_dict": model.state_dict(), "vocab": vocab.itos,
            "fact_concept_prefix": bool(getattr(model, "fact_concept_prefix", False)),
            "fact_concept_refine": bool(getattr(model, "fact_concept_refine", False)),
            "fact_concept_refine_gate_init": float(
                getattr(model, "fact_concept_refine_gate_init", -2.0)),
            "text_encoder_arch": getattr(model, "text_encoder_arch", "transformer"),
            "text_encoder_layers": int(getattr(model, "text_encoder_layers", 1)),
            "d": d, "layers": layers, "heads": heads, "fact_schema": {
                "keys": model.fact_schema.keys,
                "values": model.fact_schema.values,
            }, "report": report}


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
    choice = eval_report.get("choice_head") or {}
    if choice.get("n_records", 0):
        scores["choice"] = float(choice["fact_value_acc"])
    return scores


def _score_metric(eval_report, metric):
    scores = _fact_value_scores(eval_report)
    if metric in scores:
        return scores[metric]
    if metric == "both":
        return 0.5 * (scores["semantic"] + scores["teacher"])
    if metric == "choice":
        return 0.5 * (scores["semantic"] + scores["teacher"])
    if metric == "min":
        return min(scores["semantic"], scores["teacher"])
    raise ValueError(f"unknown study score metric {metric!r}")


def _control_gap_values(eval_report, metric="both"):
    use_teacher = metric in ("teacher", "both", "min")
    use_semantic = metric in ("semantic", "both", "min")
    use_choice = metric == "choice"
    gaps = []
    nli = eval_report.get("nli_artifact_control") or {}
    if nli.get("n", 0):
        if use_teacher:
            gaps.append(("nli_full_minus_hypothesis_only",
                         float(nli.get("full_minus_hypothesis_only", 0.0))))
        if use_semantic:
            gaps.append(("nli_semantic_full_minus_hypothesis_only",
                         float(nli.get("semantic_full_minus_hypothesis_only", 0.0))))
    qa = eval_report.get("qa_ablation_control") or {}
    if qa.get("n", 0):
        if use_teacher:
            gaps.extend([("qa_full_minus_question_only",
                          float(qa.get("full_minus_question_only", 0.0))),
                         ("qa_full_minus_context_only",
                          float(qa.get("full_minus_context_only", 0.0)))])
        if use_semantic:
            gaps.extend([("qa_semantic_full_minus_question_only",
                          float(qa.get("semantic_full_minus_question_only", 0.0))),
                         ("qa_semantic_full_minus_context_only",
                          float(qa.get("semantic_full_minus_context_only", 0.0)))])
        if use_choice:
            gaps.extend([("qa_choice_full_minus_question_only",
                          float(qa.get("choice_full_minus_question_only", 0.0))),
                         ("qa_choice_full_minus_context_only",
                          float(qa.get("choice_full_minus_context_only", 0.0)))])
    qa_swap = eval_report.get("qa_question_swap_control") or {}
    if qa_swap.get("n", 0):
        if use_teacher:
            gaps.append(("qa_full_minus_question_swap",
                         float(qa_swap.get("full_minus_question_swap", 0.0))))
        if use_semantic:
            gaps.append(("qa_semantic_full_minus_question_swap",
                         float(qa_swap.get("semantic_full_minus_question_swap", 0.0))))
        if use_choice:
            gaps.append(("qa_choice_full_minus_question_swap",
                         float(qa_swap.get("choice_full_minus_question_swap", 0.0))))
    qa_repl = eval_report.get("qa_candidate_replacement_control") or {}
    if qa_repl.get("n", 0) and use_choice:
        gaps.append(("qa_choice_candidate_replacement_abstain_margin",
                     float(qa_repl.get("candidate_replacement_none_acc", 0.0)) - 0.5))
    qa_repl_binding = eval_report.get("qa_candidate_replacement_binding_control") or {}
    if qa_repl_binding.get("n", 0) and use_choice:
        gaps.append(("qa_choice_candidate_replacement_target_drop",
                     float(qa_repl_binding.get("target_logit_drop", 0.0))))
    qa_repl_ans_binding = (
        eval_report.get("qa_candidate_replacement_answerability_binding_control") or {})
    if qa_repl_ans_binding.get("n", 0) and use_choice:
        gaps.append(("qa_choice_candidate_replacement_target_cover_drop",
                     float(qa_repl_ans_binding.get("target_cover_drop", 0.0))))
    qa_concept = eval_report.get("qa_concept_discovery_control") or {}
    if qa_concept.get("n", 0) and use_choice:
        gaps.append(("qa_choice_concept_discovery_margin",
                     float(qa_concept.get("discovery_margin", 0.0))))
    qa_proto = eval_report.get("qa_concept_prototype_control") or {}
    if qa_proto.get("n", 0) and use_choice:
        gaps.append(("qa_choice_concept_prototype_margin",
                     float(qa_proto.get("prototype_margin", 0.0))))
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
    choice = float(row.get("choice_head_fact_value_acc", 0.0))
    if metric == "teacher":
        return teacher
    if metric == "semantic":
        return semantic
    if metric == "choice":
        return choice if row.get("choice_head_records", 0) else 0.5 * (teacher + semantic)
    if metric == "both":
        return 0.5 * (teacher + semantic)
    if metric == "min":
        return min(teacher, semantic)
    raise ValueError(f"unknown study score metric {metric!r}")


def _kind_floor_penalty(eval_report, metric="both"):
    by_kind = eval_report.get("by_kind") or {}
    if len(by_kind) < 2:
        return 0.0
    study_score = _score_metric(eval_report, metric)
    kind_floor = min(_kind_score(row, metric) for row in by_kind.values())
    return max(0.0, study_score - kind_floor)


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
                     and not control_failed
                     and not kind_regressed)
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
    return {
        "metric": metric,
        "study_score": study_score,
        "study_semantic_fact_value_acc": _fact_value_scores(study_eval)["semantic"],
        "study_teacher_fact_value_acc": _fact_value_scores(study_eval)["teacher"],
        "study_choice_fact_value_acc": _fact_value_scores(study_eval).get("choice"),
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
        fact_n=0, kind_fact_n=0, artifact_n=0, decode_w=1.0, choice_w=0.0,
        fact_concept_w=0.0, fact_concept_contrast_w=0.0,
        fact_concept_contrast_temperature=0.1,
        fact_concept_centroid_w=0.0, fact_concept_centroid_temperature=0.1,
        fact_concept_centroid_margin=0.0, fact_concept_prefix=False,
        fact_concept_prototype_w=0.0, fact_concept_prototype_spread_w=0.0,
        fact_concept_prototype_spread_margin=0.2,
        fact_concept_state_spread_w=0.0, fact_concept_state_spread_variance=0.05,
        fact_concept_state_spread_margin=0.2,
        fact_concept_state_spread_covariance_w=0.05,
        choice_answer_w=1.0, choice_none_w=1.0,
        choice_answer_margin=0.0, choice_none_margin=0.0,
        choice_final_w=0.0, choice_final_control_w=0.0,
        choice_final_margin=0.0,
        choice_final_control_contrast_w=0.0,
        choice_final_control_contrast_margin=0.0,
        choice_candidate_replacement_w=0.0,
        choice_candidate_replacement_margin=0.0,
        choice_candidate_replacement_pair_w=0.0,
        choice_candidate_replacement_pair_margin=0.0,
        choice_candidate_replacement_binding_w=0.0,
        choice_candidate_replacement_binding_margin=0.0,
        choice_candidate_replacement_answerability_binding_w=0.0,
        choice_candidate_replacement_answerability_binding_margin=0.0,
        choice_positive_anchor_w=0.0,
        choice_positive_anchor_margin=0.0,
        choice_concept_bridge_w=0.0,
        choice_concept_bridge_margin=0.0,
        choice_concept_prototype_w=0.0,
        choice_concept_prototype_margin=0.0,
        choice_answerability_w=0.0, choice_answerability_control_w=0.0,
        choice_answerability_margin=0.0,
        choice_answerability_contrast_w=0.0,
        choice_answerability_contrast_margin=0.0,
        choice_candidate_answerability_w=0.0,
        choice_candidate_answerability_control_w=0.0,
        choice_candidate_answerability_margin=0.0,
        choice_candidate_answerability_contrast_w=0.0,
        choice_candidate_answerability_contrast_margin=0.0,
        choice_answerability_pair_w=0.0,
        choice_answerability_pair_margin=0.0,
        choice_context_w=0.0,
        choice_candidate_context_w=0.0,
        choice_candidate_context_margin=0.0,
        choice_question_context_w=0.0,
        choice_question_context_contrast_w=0.0,
        choice_question_context_margin=0.0,
        choice_pair_w=0.0, choice_pair_margin=0.0,
        choice_control_w=0.0,
        choice_control_contrast_w=0.0, choice_control_margin=0.0):
    records = load_records(data)
    model, vocab = train_model(records, steps=steps, batch=batch, d=d, layers=layers,
                               heads=heads,
                               text_encoder_arch=text_encoder_arch,
                               text_encoder_layers=text_encoder_layers,
                               fact_concept_refine=fact_concept_refine,
                               fact_concept_refine_gate_init=(
                                   fact_concept_refine_gate_init),
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
                               decode_w=decode_w,
                               choice_w=choice_w, choice_answer_w=choice_answer_w,
                               choice_none_w=choice_none_w,
                               choice_answer_margin=choice_answer_margin,
                               choice_none_margin=choice_none_margin,
                               choice_final_w=choice_final_w,
                               choice_final_control_w=choice_final_control_w,
                               choice_final_margin=choice_final_margin,
                               choice_final_control_contrast_w=(
                                   choice_final_control_contrast_w),
                               choice_final_control_contrast_margin=(
                                   choice_final_control_contrast_margin),
                               choice_candidate_replacement_w=(
                                   choice_candidate_replacement_w),
                               choice_candidate_replacement_margin=(
                                   choice_candidate_replacement_margin),
                               choice_candidate_replacement_pair_w=(
                                   choice_candidate_replacement_pair_w),
                               choice_candidate_replacement_pair_margin=(
                                   choice_candidate_replacement_pair_margin),
                               choice_candidate_replacement_binding_w=(
                                   choice_candidate_replacement_binding_w),
                               choice_candidate_replacement_binding_margin=(
                                   choice_candidate_replacement_binding_margin),
                               choice_candidate_replacement_answerability_binding_w=(
                                   choice_candidate_replacement_answerability_binding_w),
                               choice_candidate_replacement_answerability_binding_margin=(
                                   choice_candidate_replacement_answerability_binding_margin),
                               choice_positive_anchor_w=choice_positive_anchor_w,
                               choice_positive_anchor_margin=(
                                   choice_positive_anchor_margin),
                               choice_concept_bridge_w=choice_concept_bridge_w,
                               choice_concept_bridge_margin=(
                                   choice_concept_bridge_margin),
                               choice_concept_prototype_w=choice_concept_prototype_w,
                               choice_concept_prototype_margin=(
                                   choice_concept_prototype_margin),
                               choice_answerability_w=choice_answerability_w,
                               choice_answerability_control_w=choice_answerability_control_w,
                               choice_answerability_margin=choice_answerability_margin,
                               choice_answerability_contrast_w=(
                                   choice_answerability_contrast_w),
                               choice_answerability_contrast_margin=(
                                   choice_answerability_contrast_margin),
                               choice_candidate_answerability_w=(
                                   choice_candidate_answerability_w),
                               choice_candidate_answerability_control_w=(
                                   choice_candidate_answerability_control_w),
                               choice_candidate_answerability_margin=(
                                   choice_candidate_answerability_margin),
                               choice_candidate_answerability_contrast_w=(
                                   choice_candidate_answerability_contrast_w),
                               choice_candidate_answerability_contrast_margin=(
                                   choice_candidate_answerability_contrast_margin),
                               choice_answerability_pair_w=choice_answerability_pair_w,
                               choice_answerability_pair_margin=(
                                   choice_answerability_pair_margin),
                               choice_context_w=choice_context_w,
                               choice_candidate_context_w=choice_candidate_context_w,
                               choice_candidate_context_margin=(
                                   choice_candidate_context_margin),
                               choice_question_context_w=choice_question_context_w,
                               choice_question_context_contrast_w=(
                                   choice_question_context_contrast_w),
                               choice_question_context_margin=choice_question_context_margin,
                               choice_pair_w=choice_pair_w,
                               choice_pair_margin=choice_pair_margin,
                               choice_control_w=choice_control_w,
                               choice_control_contrast_w=choice_control_contrast_w,
                               choice_control_margin=choice_control_margin)
    report = {"experiment": "text0_semantic_extraction", "data": data, "steps": steps,
              "d": int(d), "layers": int(layers), "heads": int(heads),
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
              "choice_w": float(choice_w),
              "choice_answer_w": float(choice_answer_w),
              "choice_none_w": float(choice_none_w),
              "choice_answer_margin": float(choice_answer_margin),
              "choice_none_margin": float(choice_none_margin),
              "choice_final_w": float(choice_final_w),
              "choice_final_control_w": float(choice_final_control_w),
              "choice_final_margin": float(choice_final_margin),
              "choice_final_control_contrast_w": float(
                  choice_final_control_contrast_w),
              "choice_final_control_contrast_margin": float(
                  choice_final_control_contrast_margin),
              "choice_candidate_replacement_w": float(
                  choice_candidate_replacement_w),
              "choice_candidate_replacement_margin": float(
                  choice_candidate_replacement_margin),
              "choice_candidate_replacement_pair_w": float(
                  choice_candidate_replacement_pair_w),
              "choice_candidate_replacement_pair_margin": float(
                  choice_candidate_replacement_pair_margin),
              "choice_candidate_replacement_binding_w": float(
                  choice_candidate_replacement_binding_w),
              "choice_candidate_replacement_binding_margin": float(
                  choice_candidate_replacement_binding_margin),
              "choice_candidate_replacement_answerability_binding_w": float(
                  choice_candidate_replacement_answerability_binding_w),
              "choice_candidate_replacement_answerability_binding_margin": float(
                  choice_candidate_replacement_answerability_binding_margin),
              "choice_positive_anchor_w": float(choice_positive_anchor_w),
              "choice_positive_anchor_margin": float(choice_positive_anchor_margin),
              "choice_concept_bridge_w": float(choice_concept_bridge_w),
              "choice_concept_bridge_margin": float(choice_concept_bridge_margin),
              "choice_concept_prototype_w": float(choice_concept_prototype_w),
              "choice_concept_prototype_margin": float(choice_concept_prototype_margin),
              "choice_answerability_w": float(choice_answerability_w),
              "choice_answerability_control_w": float(choice_answerability_control_w),
              "choice_answerability_margin": float(choice_answerability_margin),
              "choice_answerability_contrast_w": float(choice_answerability_contrast_w),
              "choice_answerability_contrast_margin": float(
                  choice_answerability_contrast_margin),
              "choice_candidate_answerability_w": float(
                  choice_candidate_answerability_w),
              "choice_candidate_answerability_control_w": float(
                  choice_candidate_answerability_control_w),
              "choice_candidate_answerability_margin": float(
                  choice_candidate_answerability_margin),
              "choice_candidate_answerability_contrast_w": float(
                  choice_candidate_answerability_contrast_w),
              "choice_candidate_answerability_contrast_margin": float(
                  choice_candidate_answerability_contrast_margin),
              "choice_answerability_pair_w": float(choice_answerability_pair_w),
              "choice_answerability_pair_margin": float(
                  choice_answerability_pair_margin),
              "choice_context_w": float(choice_context_w),
              "choice_candidate_context_w": float(choice_candidate_context_w),
              "choice_candidate_context_margin": float(choice_candidate_context_margin),
              "choice_question_context_w": float(choice_question_context_w),
              "choice_question_context_contrast_w": float(
                  choice_question_context_contrast_w),
              "choice_question_context_margin": float(choice_question_context_margin),
              "choice_pair_w": float(choice_pair_w),
              "choice_pair_margin": float(choice_pair_margin),
              "choice_control_w": float(choice_control_w),
              "choice_control_contrast_w": float(choice_control_contrast_w),
              "choice_control_margin": float(choice_control_margin),
              "free_n": int(free_n),
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
                     steps=200, batch=32, lr=5e-4, seed=0, device=DEV, max_new=160,
                     semantic_w=0.5, free_n=0, paraphrase_n=0, counterfactual_n=0,
                     kind_free_n=0, balance_by="kind", fact_n=0, kind_fact_n=0,
                     artifact_n=0, decode_w=1.0, choice_w=0.0,
                     fact_concept_w=0.0, fact_concept_contrast_w=0.0,
                     fact_concept_contrast_temperature=0.1,
                     fact_concept_centroid_w=0.0,
                     fact_concept_centroid_temperature=0.1,
                     fact_concept_centroid_margin=0.0,
                     fact_concept_prefix=False,
                     fact_concept_refine=False,
                     fact_concept_refine_gate_init=-2.0,
                     fact_concept_prototype_w=0.0,
                     fact_concept_prototype_spread_w=0.0,
                     fact_concept_prototype_spread_margin=0.2,
                     fact_concept_state_spread_w=0.0,
                     fact_concept_state_spread_variance=0.05,
                     fact_concept_state_spread_margin=0.2,
                     fact_concept_state_spread_covariance_w=0.05,
                     choice_answer_w=1.0, choice_none_w=1.0,
                     choice_answer_margin=0.0, choice_none_margin=0.0,
                     choice_final_w=0.0, choice_final_control_w=0.0,
                     choice_final_margin=0.0,
                     choice_final_control_contrast_w=0.0,
                     choice_final_control_contrast_margin=0.0,
                     choice_candidate_replacement_w=0.0,
                     choice_candidate_replacement_margin=0.0,
                     choice_candidate_replacement_pair_w=0.0,
                     choice_candidate_replacement_pair_margin=0.0,
                     choice_candidate_replacement_binding_w=0.0,
                     choice_candidate_replacement_binding_margin=0.0,
                     choice_candidate_replacement_answerability_binding_w=0.0,
                     choice_candidate_replacement_answerability_binding_margin=0.0,
                     choice_positive_anchor_w=0.0,
                     choice_positive_anchor_margin=0.0,
                     choice_concept_bridge_w=0.0,
                     choice_concept_bridge_margin=0.0,
                     choice_concept_prototype_w=0.0,
                     choice_concept_prototype_margin=0.0,
                     choice_self_distill_w=0.0,
                     choice_self_distill_temperature=1.0,
                     choice_self_rank_distill_w=0.0,
                     choice_self_rank_distill_margin=0.0,
                     choice_answerability_w=0.0, choice_answerability_control_w=0.0,
                     choice_answerability_margin=0.0,
                     choice_answerability_contrast_w=0.0,
                     choice_answerability_contrast_margin=0.0,
                     choice_candidate_answerability_w=0.0,
                     choice_candidate_answerability_control_w=0.0,
                     choice_candidate_answerability_margin=0.0,
                     choice_candidate_answerability_contrast_w=0.0,
                     choice_candidate_answerability_contrast_margin=0.0,
                     choice_answerability_pair_w=0.0,
                     choice_answerability_pair_margin=0.0,
                     choice_context_w=0.0,
                     choice_candidate_context_w=0.0,
                     choice_candidate_context_margin=0.0,
                     choice_question_context_w=0.0,
                     choice_question_context_contrast_w=0.0,
                     choice_question_context_margin=0.0,
                     choice_pair_w=0.0,
                     choice_pair_margin=0.0,
                     choice_control_w=0.0, choice_control_contrast_w=0.0,
                     choice_control_margin=0.0, study_rounds=1,
                     study_strategy="all", study_probe_n=0, study_hard_max=0,
                     study_anchor_correct_per_kind=0,
                     study_anchor_correct_repeat=1,
                     study_anchor_retention_bucket=False,
                     study_distill_correct_per_kind=0,
                     study_discovery_correct_per_kind=0,
                     study_fragile_correct_mining=False,
                     study_adaptive_correct_mining=False,
                     study_adaptive_correct_pool_per_kind=0,
                     study_discovery_transfer=False,
                     study_focus_control_failures=False,
                     study_select_best=False, study_score_metric="both",
                     study_retention_w=1.0, study_control_w=1.0,
                     study_kind_w=1.0, study_require_positive_score=True,
                     study_confirm_n=0, study_confirm_seed_stride=7919):
    """Continue a text checkpoint on a reading-task dataset and report before/after.

    The dataset supplies semantic supervision; this function handles the weight update. It expands
    token and fact-label capacity when the reading task introduces new words or concepts.
    """
    study_records = load_records(data, require_train=True, require_eval=True)
    replay_records = (load_records(replay_data, require_train=False, require_eval=False)
                      if replay_data else [])
    fit_records = study_records + replay_records
    torch.manual_seed(seed)
    model, vocab, ckpt = expanded_checkpoint_model(checkpoint, fit_records, device=device)
    if fact_concept_prefix:
        model.fact_concept_prefix = True
    d = int(ckpt.get("d", 96))
    layers = int(ckpt.get("layers", 3))
    heads = int(ckpt.get("heads", 4))
    if fact_concept_refine:
        model.enable_fact_concept_refiner(
            heads=heads, gate_init=fact_concept_refine_gate_init)
    old_schema = fact_schema_from_payload(ckpt.get("fact_schema"))
    old_fact_values = (sum(len(v) for v in old_schema.values) if old_schema is not None else 0)
    new_fact_values = sum(len(v) for v in model.fact_schema.values)
    study_confirm_n = max(0, int(study_confirm_n))
    study_confirm_seed_stride = max(1, int(study_confirm_seed_stride))
    study_anchor_correct_repeat = max(1, int(study_anchor_correct_repeat))
    study_distill_correct_per_kind = max(0, int(study_distill_correct_per_kind))
    study_discovery_correct_per_kind = max(0, int(study_discovery_correct_per_kind))
    study_adaptive_correct_pool_per_kind = max(
        0, int(study_adaptive_correct_pool_per_kind))
    eval_kwargs = dict(device=device, max_new=max_new, free_n=free_n,
                       paraphrase_n=paraphrase_n, counterfactual_n=counterfactual_n,
                       kind_free_n=kind_free_n, fact_n=fact_n,
                       kind_fact_n=kind_fact_n, artifact_n=artifact_n, seed=seed + 17)
    before = evaluate_all(model, vocab, study_records, **eval_kwargs)
    focused_control_sides = ()
    if study_focus_control_failures:
        focused_control_sides = choice_control_sides_from_failures(
            before, metric=study_score_metric)
    focused_sampling_sides = ()
    if focused_control_sides:
        focused_sampling_sides = (
            ("question", "context", "candidate_replace") + focused_control_sides)
    fit_control_sides = focused_sampling_sides or None
    replay_before = (evaluate_all(model, vocab, replay_records, **eval_kwargs)
                     if replay_records and any(r.split == "eval" for r in replay_records)
                     else None)
    confirm_refs = []
    for confirm_i in range(study_confirm_n):
        confirm_seed = seed + 17 + (confirm_i + 1) * study_confirm_seed_stride
        confirm_kwargs = eval_kwargs | {"seed": confirm_seed}
        confirm_before = evaluate_all(model, vocab, study_records, **confirm_kwargs)
        confirm_replay_before = (
            evaluate_all(model, vocab, replay_records, **confirm_kwargs)
            if replay_records and any(r.split == "eval" for r in replay_records)
            else None)
        confirm_refs.append({"index": confirm_i + 1,
                             "seed": confirm_seed,
                             "eval_kwargs": confirm_kwargs,
                             "before": confirm_before,
                             "replay_before": confirm_replay_before})
    report = {"experiment": "text0_study_update",
              "checkpoint": checkpoint,
              "data": data,
              "replay_data": replay_data or [],
              "steps": int(steps),
              "batch": int(batch),
              "lr": float(lr),
              "text_encoder_arch": getattr(model, "text_encoder_arch", "transformer"),
              "text_encoder_layers": int(getattr(model, "text_encoder_layers", 1)),
              "decode_w": float(decode_w),
              "semantic_w": float(semantic_w),
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
              "choice_w": float(choice_w),
              "choice_answer_w": float(choice_answer_w),
              "choice_none_w": float(choice_none_w),
              "choice_answer_margin": float(choice_answer_margin),
              "choice_none_margin": float(choice_none_margin),
              "choice_final_w": float(choice_final_w),
              "choice_final_control_w": float(choice_final_control_w),
              "choice_final_margin": float(choice_final_margin),
              "choice_final_control_contrast_w": float(
                  choice_final_control_contrast_w),
              "choice_final_control_contrast_margin": float(
                  choice_final_control_contrast_margin),
              "choice_candidate_replacement_w": float(
                  choice_candidate_replacement_w),
              "choice_candidate_replacement_margin": float(
                  choice_candidate_replacement_margin),
              "choice_candidate_replacement_pair_w": float(
                  choice_candidate_replacement_pair_w),
              "choice_candidate_replacement_pair_margin": float(
                  choice_candidate_replacement_pair_margin),
              "choice_candidate_replacement_binding_w": float(
                  choice_candidate_replacement_binding_w),
              "choice_candidate_replacement_binding_margin": float(
                  choice_candidate_replacement_binding_margin),
              "choice_candidate_replacement_answerability_binding_w": float(
                  choice_candidate_replacement_answerability_binding_w),
              "choice_candidate_replacement_answerability_binding_margin": float(
                  choice_candidate_replacement_answerability_binding_margin),
              "choice_positive_anchor_w": float(choice_positive_anchor_w),
              "choice_positive_anchor_margin": float(choice_positive_anchor_margin),
              "choice_concept_bridge_w": float(choice_concept_bridge_w),
              "choice_concept_bridge_margin": float(choice_concept_bridge_margin),
              "choice_concept_prototype_w": float(choice_concept_prototype_w),
              "choice_concept_prototype_margin": float(choice_concept_prototype_margin),
              "choice_self_distill_w": float(choice_self_distill_w),
              "choice_self_distill_temperature": float(choice_self_distill_temperature),
              "choice_self_rank_distill_w": float(choice_self_rank_distill_w),
              "choice_self_rank_distill_margin": float(choice_self_rank_distill_margin),
              "choice_answerability_w": float(choice_answerability_w),
              "choice_answerability_control_w": float(choice_answerability_control_w),
              "choice_answerability_margin": float(choice_answerability_margin),
              "choice_answerability_contrast_w": float(choice_answerability_contrast_w),
              "choice_answerability_contrast_margin": float(
                  choice_answerability_contrast_margin),
              "choice_candidate_answerability_w": float(
                  choice_candidate_answerability_w),
              "choice_candidate_answerability_control_w": float(
                  choice_candidate_answerability_control_w),
              "choice_candidate_answerability_margin": float(
                  choice_candidate_answerability_margin),
              "choice_candidate_answerability_contrast_w": float(
                  choice_candidate_answerability_contrast_w),
              "choice_candidate_answerability_contrast_margin": float(
                  choice_candidate_answerability_contrast_margin),
              "choice_answerability_pair_w": float(choice_answerability_pair_w),
              "choice_answerability_pair_margin": float(
                  choice_answerability_pair_margin),
              "choice_context_w": float(choice_context_w),
              "choice_candidate_context_w": float(choice_candidate_context_w),
              "choice_candidate_context_margin": float(choice_candidate_context_margin),
              "choice_question_context_w": float(choice_question_context_w),
              "choice_question_context_contrast_w": float(
                  choice_question_context_contrast_w),
              "choice_question_context_margin": float(choice_question_context_margin),
              "choice_pair_w": float(choice_pair_w),
              "choice_pair_margin": float(choice_pair_margin),
              "choice_control_w": float(choice_control_w),
              "choice_control_contrast_w": float(choice_control_contrast_w),
              "choice_control_margin": float(choice_control_margin),
              "balance_by": balance_by,
              "study_rounds": int(study_rounds),
              "study_strategy": study_strategy,
              "study_probe_n": int(study_probe_n),
              "study_hard_max": int(study_hard_max),
              "study_anchor_correct_per_kind": int(study_anchor_correct_per_kind),
              "study_anchor_correct_repeat": int(study_anchor_correct_repeat),
              "study_anchor_retention_bucket": bool(study_anchor_retention_bucket),
              "study_distill_correct_per_kind": int(study_distill_correct_per_kind),
              "study_discovery_correct_per_kind": int(study_discovery_correct_per_kind),
              "study_fragile_correct_mining": bool(study_fragile_correct_mining),
              "study_adaptive_correct_mining": bool(study_adaptive_correct_mining),
              "study_adaptive_correct_pool_per_kind": int(
                  study_adaptive_correct_pool_per_kind),
              "study_discovery_transfer": bool(study_discovery_transfer),
              "study_focus_control_failures": bool(study_focus_control_failures),
              "study_control_focus_sides": list(focused_control_sides),
              "study_control_sampling_sides": list(focused_sampling_sides),
              "study_select_best": bool(study_select_best),
              "study_score_metric": study_score_metric,
              "study_retention_w": float(study_retention_w),
              "study_control_w": float(study_control_w),
              "study_kind_w": float(study_kind_w),
              "study_require_positive_score": bool(study_require_positive_score),
              "study_confirm_n": int(study_confirm_n),
              "study_confirm_seed_stride": int(study_confirm_seed_stride),
              "free_n": int(free_n),
              "paraphrase_n": int(paraphrase_n),
              "counterfactual_n": int(counterfactual_n),
              "kind_free_n": int(kind_free_n),
              "fact_n": int(fact_n),
              "kind_fact_n": int(kind_fact_n),
              "artifact_n": int(artifact_n),
              "study_train_records": sum(r.split == "train" for r in study_records),
              "study_eval_records": sum(r.split == "eval" for r in study_records),
              "replay_train_records": sum(r.split == "train" for r in replay_records),
              "replay_eval_records": sum(r.split == "eval" for r in replay_records),
              "old_vocab_size": len(ckpt["vocab"]),
              "new_vocab_size": len(vocab),
              "new_tokens": len(vocab) - len(ckpt["vocab"]),
              "old_fact_values": old_fact_values,
              "new_fact_values": new_fact_values,
              "new_fact_values_added": new_fact_values - old_fact_values,
              "before": before,
              "replay_before": replay_before,
              "confirmation_refs": [
                  {"index": row["index"],
                   "seed": row["seed"],
                   "before_score_components": study_selection_components(
                       row["before"], row["replay_before"], row["replay_before"],
                       metric=study_score_metric,
                       retention_w=study_retention_w,
                       control_w=study_control_w,
                       kind_w=study_kind_w)}
                  for row in confirm_refs
              ]}
    if out_checkpoint:
        os.makedirs(os.path.dirname(out_checkpoint) or ".", exist_ok=True)
        torch.save(checkpoint_payload(model, vocab, d, layers, heads,
                                      report | {"status": "expanded_pending_study"}),
                   out_checkpoint)
    train_study_records = [r for r in study_records if r.split == "train"]
    train_replay_records = [r for r in replay_records if r.split == "train"]
    round_reports = []
    best_state = clone_model_state(model)
    best_round = 0
    best_study_eval = before
    best_replay_eval = replay_before
    best_components = study_selection_components(before, replay_before, replay_before,
                                                 metric=study_score_metric,
                                                 retention_w=study_retention_w,
                                                 control_w=study_control_w,
                                                 kind_w=study_kind_w)
    report["initial_score_components"] = best_components
    best_score = best_components["score"]
    for round_i in range(max(1, int(study_rounds))):
        round_seed = seed + 1009 * round_i
        anchor_records = []
        anchor_counts = {}
        anchor_selection = {}
        distill_records = []
        distill_counts = {}
        distill_selection = {}
        discovery_records = []
        discovery_counts = {}
        discovery_selection = {}
        discovery_source_records = []
        positive_anchor_source_records = []
        if study_strategy == "errors":
            need_metric_correct = bool(
                study_anchor_correct_per_kind
                or (study_score_metric == "choice"
                    and (study_distill_correct_per_kind
                         or study_discovery_correct_per_kind)))
            choice_correct_records = []
            if study_score_metric == "choice":
                if need_metric_correct:
                    hard_records, correct_records, hard_report = choice_record_outcomes(
                        model, vocab, train_study_records, device=device,
                        n=study_probe_n, seed=round_seed)
                    choice_correct_records = correct_records
                else:
                    hard_records, hard_report = choice_record_errors(
                        model, vocab, train_study_records, device=device,
                        n=study_probe_n, seed=round_seed)
                    correct_records = []
                hard_report = hard_report | {"error_metric": "choice"}
            else:
                if study_anchor_correct_per_kind:
                    hard_records, correct_records, hard_report = semantic_record_outcomes(
                        model, vocab, train_study_records, device=device,
                        n=study_probe_n, seed=round_seed)
                else:
                    hard_records, hard_report = semantic_record_errors(
                        model, vocab, train_study_records, device=device,
                        n=study_probe_n, seed=round_seed)
                    correct_records = []
                hard_report = hard_report | {"error_metric": "semantic"}
                if study_distill_correct_per_kind or study_discovery_correct_per_kind:
                    _choice_hard, choice_correct_records, choice_correct_report = (
                        choice_record_outcomes(model, vocab, train_study_records,
                                               device=device, n=study_probe_n,
                                               seed=round_seed))
                    hard_report = hard_report | {
                        "choice_correct_for_self_teach": {
                            "n_records": choice_correct_report["n_records"],
                            "n_correct_records": choice_correct_report[
                                "n_correct_records"],
                            "choice_error_rate": choice_correct_report[
                                "choice_error_rate"],
                        }}
            if study_hard_max and len(hard_records) > study_hard_max:
                rng = np.random.default_rng(round_seed + 17)
                idx = rng.choice(len(hard_records), size=study_hard_max, replace=False)
                hard_records = [hard_records[int(i)] for i in idx]
                hard_report = hard_report | {"capped": True,
                                             "n_error_records_used": len(hard_records)}
            else:
                hard_report = hard_report | {"capped": False,
                                             "n_error_records_used": len(hard_records)}
            if study_anchor_correct_per_kind and correct_records:
                rng = np.random.default_rng(round_seed + 53)
                if study_fragile_correct_mining and study_score_metric == "choice":
                    anchor_records, anchor_counts, anchor_selection = (
                        choice_fragile_correct_records_per_kind(
                            model, vocab, correct_records, rng,
                            study_anchor_correct_per_kind, device=device,
                            pool_per_kind=study_adaptive_correct_pool_per_kind))
                else:
                    anchor_records, anchor_counts = sample_records_per_kind(
                        correct_records, rng, study_anchor_correct_per_kind)
                    anchor_selection = {"fragile": False, "reason": "random_per_kind"}
                if study_anchor_retention_bucket:
                    anchor_records = retention_anchor_records(anchor_records)
                if study_anchor_correct_repeat > 1:
                    anchor_records = anchor_records * study_anchor_correct_repeat
            if study_distill_correct_per_kind and choice_correct_records:
                rng = np.random.default_rng(round_seed + 71)
                if study_fragile_correct_mining:
                    distill_records, distill_counts, distill_selection = (
                        choice_fragile_correct_records_per_kind(
                            model, vocab, choice_correct_records, rng,
                            study_distill_correct_per_kind, device=device,
                            pool_per_kind=study_adaptive_correct_pool_per_kind))
                elif study_adaptive_correct_mining and hard_records:
                    distill_records, distill_counts, distill_selection = (
                        choice_neighbor_records_per_kind(
                            model, vocab, hard_records, choice_correct_records, rng,
                            study_distill_correct_per_kind, device=device,
                            pool_per_kind=study_adaptive_correct_pool_per_kind))
                else:
                    distill_records, distill_counts = sample_records_per_kind(
                        choice_correct_records, rng, study_distill_correct_per_kind)
                    distill_selection = {"adaptive": False, "reason": "random_per_kind"}
            if study_discovery_correct_per_kind and choice_correct_records:
                rng = np.random.default_rng(round_seed + 89)
                if study_adaptive_correct_mining and hard_records:
                    discovery_records, discovery_counts, discovery_selection = (
                        choice_neighbor_records_per_kind(
                            model, vocab, hard_records, choice_correct_records, rng,
                            study_discovery_correct_per_kind, device=device,
                            pool_per_kind=study_adaptive_correct_pool_per_kind))
                else:
                    discovery_records, discovery_counts = sample_records_per_kind(
                        choice_correct_records, rng, study_discovery_correct_per_kind)
                    discovery_selection = {"adaptive": False, "reason": "random_per_kind"}
            if discovery_records:
                discovery_source_records = unique_records_by_id(
                    hard_records, discovery_records)
            positive_anchor_source_records = (
                list(anchor_records) if anchor_records else list(distill_records))
            hard_report = hard_report | {
                "n_anchor_records": len(anchor_records),
                "n_unique_anchor_records": sum(anchor_counts.values()),
                "anchor_repeat": study_anchor_correct_repeat,
                "anchor_retention_bucket": bool(study_anchor_retention_bucket),
                "anchor_records_by_kind": anchor_counts,
                "anchor_selection": anchor_selection,
                "n_distill_records": len(distill_records),
                "distill_records_by_kind": distill_counts,
                "distill_selection": distill_selection,
                "n_positive_anchor_source_records": len(positive_anchor_source_records),
                "n_discovery_records": len(discovery_records),
                "discovery_records_by_kind": discovery_counts,
                "discovery_selection": discovery_selection,
                "n_discovery_source_records": len(discovery_source_records),
                "n_discovery_source_hard_records": (
                    len(hard_records) if discovery_source_records else 0),
                "n_discovery_source_correct_records": (
                    len(discovery_records) if discovery_source_records else 0),
                "discovery_pairing": (
                    "hard_to_correct_transfer"
                    if study_discovery_transfer and discovery_source_records
                    else ("hard_aware_neighborhood" if discovery_source_records
                          else "default")),
                "discovery_transfer_variant": (
                    "correct_detached" if study_discovery_transfer else "none"),
            }
            round_fit_records = hard_records + anchor_records + train_replay_records
            if not hard_records:
                round_fit_records = train_study_records + train_replay_records
        elif study_strategy == "all":
            hard_report = {"strategy": "all",
                           "n_records": len(train_study_records),
                           "n_error_records": None,
                           "n_error_records_used": len(train_study_records)}
            round_fit_records = train_study_records + train_replay_records
        else:
            raise ValueError(f"unknown study strategy {study_strategy!r}")
        round_reports.append({"round": round_i + 1,
                              "fit_records": len(round_fit_records),
                              "study_fit_records": max(0, len(round_fit_records)
                                                       - len(train_replay_records)),
                              "anchor_records": len(anchor_records),
                              "anchor_records_by_kind": anchor_counts,
                              "anchor_selection": anchor_selection,
                              "distill_records": len(distill_records),
                              "distill_records_by_kind": distill_counts,
                              "distill_selection": distill_selection,
                              "positive_anchor_source_records": (
                                  len(positive_anchor_source_records)),
                              "discovery_records": len(discovery_records),
                              "discovery_records_by_kind": discovery_counts,
                              "discovery_selection": discovery_selection,
                              "discovery_source_records": len(discovery_source_records),
                              "discovery_pairing": (
                                  "hard_to_correct_transfer"
                                  if (study_discovery_transfer
                                      and discovery_source_records)
                                  else ("hard_aware_neighborhood"
                                        if discovery_source_records else "default")),
                              "discovery_transfer_variant": (
                                  "correct_detached"
                                  if study_discovery_transfer else "none"),
                              "control_focus_sides": list(focused_control_sides),
                              "control_sampling_sides": list(focused_sampling_sides),
                              "replay_fit_records": len(train_replay_records),
                              "hard_examples": hard_report})
        distill_teacher = None
        if (choice_self_distill_w or choice_self_rank_distill_w) and distill_records:
            distill_teacher = copy.deepcopy(model).to(device)
            distill_teacher.eval()
            for param in distill_teacher.parameters():
                param.requires_grad_(False)
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
                  prefix=f"study-r{round_i + 1}",
                  decode_w=decode_w, choice_w=choice_w,
                  choice_answer_w=choice_answer_w, choice_none_w=choice_none_w,
                  choice_answer_margin=choice_answer_margin,
                  choice_none_margin=choice_none_margin,
                  choice_final_w=choice_final_w,
                  choice_final_control_w=choice_final_control_w,
                  choice_final_margin=choice_final_margin,
                  choice_final_control_contrast_w=choice_final_control_contrast_w,
                  choice_final_control_contrast_margin=(
                      choice_final_control_contrast_margin),
                  choice_candidate_replacement_w=choice_candidate_replacement_w,
                  choice_candidate_replacement_margin=(
                      choice_candidate_replacement_margin),
                  choice_candidate_replacement_pair_w=(
                      choice_candidate_replacement_pair_w),
                  choice_candidate_replacement_pair_margin=(
                      choice_candidate_replacement_pair_margin),
                  choice_candidate_replacement_binding_w=(
                      choice_candidate_replacement_binding_w),
                  choice_candidate_replacement_binding_margin=(
                      choice_candidate_replacement_binding_margin),
                  choice_candidate_replacement_answerability_binding_w=(
                      choice_candidate_replacement_answerability_binding_w),
                  choice_candidate_replacement_answerability_binding_margin=(
                      choice_candidate_replacement_answerability_binding_margin),
                  choice_positive_anchor_w=choice_positive_anchor_w,
                  choice_positive_anchor_margin=choice_positive_anchor_margin,
                  choice_positive_anchor_sources=(
                      positive_anchor_source_records
                      if positive_anchor_source_records else None),
                  choice_concept_bridge_w=choice_concept_bridge_w,
                  choice_concept_bridge_margin=choice_concept_bridge_margin,
                  choice_concept_prototype_w=choice_concept_prototype_w,
                  choice_concept_prototype_margin=choice_concept_prototype_margin,
                  choice_concept_bridge_sources=(
                      discovery_source_records or discovery_records or None),
                  choice_concept_hard_sources=(
                      hard_records
                      if study_discovery_transfer and discovery_records else None),
                  choice_concept_correct_sources=(discovery_records or None),
                  choice_self_distill_w=choice_self_distill_w,
                  choice_self_distill_temperature=choice_self_distill_temperature,
                  choice_self_rank_distill_w=choice_self_rank_distill_w,
                  choice_self_rank_distill_margin=choice_self_rank_distill_margin,
                  choice_self_distill_sources=distill_records,
                  choice_self_distill_model=distill_teacher,
                  choice_answerability_w=choice_answerability_w,
                  choice_answerability_control_w=choice_answerability_control_w,
                  choice_answerability_margin=choice_answerability_margin,
                  choice_answerability_contrast_w=choice_answerability_contrast_w,
                  choice_answerability_contrast_margin=(
                      choice_answerability_contrast_margin),
                  choice_candidate_answerability_w=choice_candidate_answerability_w,
                  choice_candidate_answerability_control_w=(
                      choice_candidate_answerability_control_w),
                  choice_candidate_answerability_margin=(
                      choice_candidate_answerability_margin),
                  choice_candidate_answerability_contrast_w=(
                      choice_candidate_answerability_contrast_w),
                  choice_candidate_answerability_contrast_margin=(
                      choice_candidate_answerability_contrast_margin),
                  choice_answerability_pair_w=choice_answerability_pair_w,
                  choice_answerability_pair_margin=choice_answerability_pair_margin,
                  choice_context_w=choice_context_w,
                  choice_candidate_context_w=choice_candidate_context_w,
                  choice_candidate_context_margin=choice_candidate_context_margin,
                  choice_question_context_w=choice_question_context_w,
                  choice_question_context_contrast_w=choice_question_context_contrast_w,
                  choice_question_context_margin=choice_question_context_margin,
                  choice_pair_w=choice_pair_w,
                  choice_pair_margin=choice_pair_margin,
                  choice_control_w=choice_control_w,
                  choice_control_contrast_w=choice_control_contrast_w,
                  choice_control_margin=choice_control_margin,
                  choice_control_sides=fit_control_sides)
        if study_select_best:
            round_study_eval = evaluate_all(model, vocab, study_records, **eval_kwargs)
            round_replay_eval = (evaluate_all(model, vocab, replay_records, **eval_kwargs)
                                 if replay_records and any(r.split == "eval"
                                                           for r in replay_records)
                                 else None)
            round_components = study_selection_components(round_study_eval,
                                                          round_replay_eval,
                                                          replay_before,
                                                          metric=study_score_metric,
                                                          retention_w=study_retention_w,
                                                          control_w=study_control_w,
                                                          kind_w=study_kind_w,
                                                          study_ref=before)
            round_score = round_components["score"]
            round_reports[-1]["score"] = round_score
            round_reports[-1]["score_components"] = round_components
            round_reports[-1]["study_eval"] = {
                "teacher_forced_fact_value_acc":
                    round_study_eval["teacher_forced"]["fact_value_acc"],
                "semantic_fact_value_acc":
                    round_study_eval["semantic_head"]["fact_value_acc"],
                "choice_head_fact_value_acc":
                    round_study_eval["choice_head"]["fact_value_acc"],
                "gate": round_study_eval["gate"],
            }
            if round_replay_eval is not None:
                round_reports[-1]["replay_eval"] = {
                    "teacher_forced_fact_value_acc":
                        round_replay_eval["teacher_forced"]["fact_value_acc"],
                    "semantic_fact_value_acc":
                        round_replay_eval["semantic_head"]["fact_value_acc"],
                    "choice_head_fact_value_acc":
                        round_replay_eval["choice_head"]["fact_value_acc"],
                    "gate": round_replay_eval["gate"],
                }
            allowed = study_selection_allowed(
                round_components, require_positive=study_require_positive_score,
                control_w=study_control_w, kind_w=study_kind_w)
            confirmation_checks = []
            for confirm in confirm_refs:
                confirm_study_eval = evaluate_all(
                    model, vocab, study_records, **confirm["eval_kwargs"])
                confirm_replay_eval = (
                    evaluate_all(model, vocab, replay_records,
                                 **confirm["eval_kwargs"])
                    if replay_records and any(r.split == "eval" for r in replay_records)
                    else None)
                confirm_components = study_selection_components(
                    confirm_study_eval, confirm_replay_eval,
                    confirm["replay_before"],
                    metric=study_score_metric,
                    retention_w=study_retention_w,
                    control_w=study_control_w,
                    kind_w=study_kind_w,
                    study_ref=confirm["before"])
                confirm_allowed = study_selection_allowed(
                    confirm_components,
                    require_positive=study_require_positive_score,
                    control_w=study_control_w,
                    kind_w=study_kind_w)
                confirmation_checks.append({
                    "index": confirm["index"],
                    "seed": confirm["seed"],
                    "score": confirm_components["score"],
                    "score_components": confirm_components,
                    **confirm_allowed,
                })
            confirmation_allowed = all(row["score_allowed"]
                                       for row in confirmation_checks)
            if confirmation_checks:
                round_reports[-1]["confirmation"] = {
                    "n": len(confirmation_checks),
                    "all_allowed": bool(confirmation_allowed),
                    "checks": confirmation_checks,
                }
            round_allowed = allowed["score_allowed"] and confirmation_allowed
            round_reports[-1].update(allowed)
            round_reports[-1]["confirmation_allowed"] = bool(confirmation_allowed)
            round_reports[-1]["round_allowed"] = bool(round_allowed)
            if round_allowed and round_score >= best_score:
                best_score = round_score
                best_components = round_components
                best_round = round_i + 1
                best_state = clone_model_state(model)
                best_study_eval = round_study_eval
                best_replay_eval = round_replay_eval
    if study_select_best:
        restore_model_state(model, best_state, device=device)
        after = best_study_eval
        replay_after = best_replay_eval
    else:
        after = evaluate_all(model, vocab, study_records, **eval_kwargs)
        replay_after = (evaluate_all(model, vocab, replay_records, **eval_kwargs)
                        if replay_records and any(r.split == "eval" for r in replay_records)
                        else None)
    report["after"] = after
    report["replay_after"] = replay_after
    report["rounds"] = round_reports
    report["selected_round"] = best_round if study_select_best else int(study_rounds)
    report["selected_score"] = best_score if study_select_best else None
    report["selected_score_components"] = (best_components if study_select_best else None)
    report["delta"] = {
        "teacher_forced_fact_value_acc": (
            after["teacher_forced"]["fact_value_acc"]
            - before["teacher_forced"]["fact_value_acc"]),
        "semantic_fact_value_acc": (
            after["semantic_head"]["fact_value_acc"]
            - before["semantic_head"]["fact_value_acc"]),
        "choice_head_fact_value_acc": (
            after["choice_head"]["fact_value_acc"]
            - before["choice_head"]["fact_value_acc"]),
        "free_f1": after["free_decode"]["f1"] - before["free_decode"]["f1"],
    }
    report["replay_delta"] = ({
        "teacher_forced_fact_value_acc": (
            replay_after["teacher_forced"]["fact_value_acc"]
            - replay_before["teacher_forced"]["fact_value_acc"]),
        "semantic_fact_value_acc": (
            replay_after["semantic_head"]["fact_value_acc"]
            - replay_before["semantic_head"]["fact_value_acc"]),
        "choice_head_fact_value_acc": (
            replay_after["choice_head"]["fact_value_acc"]
            - replay_before["choice_head"]["fact_value_acc"]),
        "free_f1": replay_after["free_decode"]["f1"] - replay_before["free_decode"]["f1"],
    } if replay_before is not None and replay_after is not None else None)
    report["out_checkpoint"] = out_checkpoint
    if out_checkpoint:
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
    squad_payload = {"data": [{"title": "Tiny", "paragraphs": [{
        "context": "Ada wrote the first program for the analytical engine in London.",
        "qas": [{"id": "q0", "question": "Who wrote the first program?",
                 "answers": [{"text": "Ada", "answer_start": 0}]},
                {"id": "q1", "question": "What did Ada write for?",
                 "answers": [{"text": "analytical engine", "answer_start": 36}]},
                {"id": "q2", "question": "Where was the work done?",
                 "answers": [{"text": "London", "answer_start": 57}]}]}]}]}
    squad, squad_seen, squad_stats = _squad_records_from_payload(
        squad_payload, "train", 0, np.random.default_rng(0), "fixture",
        max_context_tokens=6, max_question_tokens=8, max_answer_tokens=3)
    assert squad_seen == 3 and squad_stats["windowed_contexts"] == 3
    assert squad[0]["facts"] == [["answer", "length", "1"],
                                 ["a000", "answer_token", "ada"]]
    assert squad[0]["tokens"][:2] == ["context", ":"]
    assert "question" in squad[0]["tokens"]
    squad_choice, _choice_seen, choice_stats = _squad_records_from_payload(
        squad_payload, "train", 0, np.random.default_rng(1), "fixture",
        max_context_tokens=8, max_question_tokens=8, max_answer_tokens=3,
        choice_n=2, include_extractive=False)
    assert choice_stats["choice_records"] == 3
    assert all(r["kind"] == "squad_choice" for r in squad_choice)
    assert all(r["facts"][0][1] == "choice" for r in squad_choice)
    assert all("choices" in r["tokens"] for r in squad_choice)
    choice_norm = [normalize_record(r, idx=i) for i, r in enumerate(squad_choice)]
    spans = qa_choice_spans(choice_norm[0])
    assert len(spans) == 2 and all(end > start for _cid, start, end in spans)
    assert qa_choice_target(choice_norm[0]) in {cid for cid, _start, _end in spans}
    ctx_start, ctx_end, q_start, q_end = qa_context_question_spans(choice_norm[0])
    assert ctx_start < ctx_end <= q_start < q_end
    choice_eval = [
        TextRecord(rec_id=rec.rec_id, split="eval", tokens=rec.tokens, facts=rec.facts,
                   group=rec.group, kind=rec.kind, base_id=rec.base_id,
                   changed=rec.changed, meta=rec.meta)
        for rec in choice_norm
    ]
    choice_vocab = build_vocab(choice_eval)
    choice_schema = build_fact_schema(choice_eval)
    choice_model = TextFactLM(len(choice_vocab), d=32, layers=1, heads=4,
                              pad=choice_vocab.pad, fact_schema=choice_schema).to("cpu")
    choice_txt, _choice_ids = pack(choice_eval, choice_vocab, "cpu")
    concept_logits = choice_model.fact_concept_logits(choice_txt)
    assert set(concept_logits) == set(choice_schema.keys)
    assert torch.isfinite(fact_concept_loss(
        choice_model, choice_txt, choice_eval, choice_schema))
    assert torch.isfinite(fact_concept_contrastive_loss(
        choice_model, choice_txt, choice_eval, choice_schema))
    assert torch.isfinite(fact_concept_batch_centroid_loss(
        choice_model, choice_txt, choice_eval, choice_schema))
    assert torch.isfinite(fact_concept_prototype_loss(
        choice_model, choice_txt, choice_eval, choice_schema))
    assert torch.isfinite(fact_concept_prototype_spread_loss(choice_model))
    assert torch.isfinite(fact_concept_state_spread_loss(
        choice_model, choice_txt, choice_eval, choice_schema))
    concept_eval_report = fact_concept_eval(
        choice_model, choice_vocab, choice_eval, device="cpu")
    assert concept_eval_report["n_records"] == len(choice_eval)
    concept_geometry_report = fact_concept_geometry_eval(
        choice_model, choice_vocab, choice_eval, device="cpu")
    assert set(concept_geometry_report) >= {"nearest_same_acc", "margin"}
    prefix_model = TextFactLM(len(choice_vocab), d=32, layers=1, heads=4,
                              pad=choice_vocab.pad, fact_schema=choice_schema,
                              fact_concept_prefix=True).to("cpu")
    raw_prefix, _pooled = prefix_model.encode_text(choice_txt)
    decoder_prefix = prefix_model.decoder_prefix(choice_txt)
    assert decoder_prefix.shape[1] == raw_prefix.shape[1] + len(choice_schema.keys)
    refine_model = TextFactLM(len(choice_vocab), d=32, layers=1, heads=4,
                              pad=choice_vocab.pad, fact_schema=choice_schema,
                              fact_concept_refine=True).to("cpu")
    refined_prefix, _refined_pooled = refine_model.encode_text(choice_txt)
    assert refined_prefix.shape == raw_prefix.shape
    assert any(name.startswith("fact_concept_refiner.")
               for name, _param in refine_model.named_parameters())
    refine_logits = refine_model(choice_txt, _choice_ids)
    assert refine_logits.shape[:2] == _choice_ids.shape
    rel_model = TextFactLM(len(choice_vocab), d=32, layers=1, heads=4,
                           pad=choice_vocab.pad, fact_schema=choice_schema,
                           text_encoder_arch="relational",
                           text_encoder_layers=1).to("cpu")
    rel_logits = rel_model(choice_txt, _choice_ids)
    assert rel_logits.shape[:2] == _choice_ids.shape
    assert torch.isfinite(fact_concept_loss(
        rel_model, choice_txt, choice_eval, choice_schema))
    assert torch.isfinite(choice_loss(choice_model, choice_txt, choice_eval))
    assert torch.isfinite(choice_loss(choice_model, choice_txt, choice_eval,
                                      answer_margin=0.25, none_margin=0.25))
    assert torch.isfinite(choice_final_loss(choice_model, choice_txt, choice_eval))
    assert torch.isfinite(choice_final_loss(choice_model, choice_txt, choice_eval,
                                            margin=0.25))
    answerability_rows = choice_model.choice_answerability_logits(choice_txt, choice_eval)
    assert len(answerability_rows) == len(choice_eval)
    assert all(score is not None and torch.isfinite(score)
               for _rec, score, _ids, _cover in answerability_rows)
    assert torch.isfinite(choice_answerability_loss(choice_model, choice_txt,
                                                    choice_eval))
    assert torch.isfinite(choice_candidate_answerability_loss(choice_model,
                                                              choice_txt,
                                                              choice_eval))
    assert torch.isfinite(choice_context_attention_loss(choice_model, choice_txt,
                                                        choice_eval))
    assert torch.isfinite(choice_candidate_context_contrast_loss(choice_model,
                                                                 choice_txt,
                                                                 choice_eval))
    assert torch.isfinite(choice_question_context_attention_loss(choice_model, choice_txt,
                                                                 choice_eval))
    first_target = qa_choice_target(choice_eval[0])
    first_spans = {cid: (start, end) for cid, start, end in qa_choice_spans(choice_eval[0])}
    ctx_start, ctx_end, _q_start, _q_end = qa_context_question_spans(choice_eval[0])
    assert _subsequence_token_positions(
        choice_eval[0].tokens[ctx_start:ctx_end],
        choice_eval[0].tokens[first_spans[first_target][0]:first_spans[first_target][1]])
    assert choice_target_context_positions(choice_eval[0])
    assert qa_context_positions_for_span(choice_eval[0],
                                         first_spans[first_target][0],
                                         first_spans[first_target][1])
    choice_eval_report = choice_head_eval(choice_model, choice_vocab, choice_eval, device="cpu")
    assert choice_eval_report["n_records"] == len(choice_eval)
    choice_hard, choice_hard_report = choice_record_errors(
        choice_model, choice_vocab, choice_eval, device="cpu", n=2, seed=0)
    assert choice_hard_report["n_records"] == 2
    assert "choice_error_rate" in choice_hard_report
    assert isinstance(choice_hard, list)
    choice_errs, choice_correct, choice_outcome_report = choice_record_outcomes(
        choice_model, choice_vocab, choice_eval, device="cpu", n=2, seed=0)
    assert len(choice_errs) + len(choice_correct) == choice_outcome_report["n_records"]
    assert "by_kind" in choice_outcome_report
    neighbor_rows, neighbor_counts, neighbor_report = choice_neighbor_records_per_kind(
        choice_model, choice_vocab, choice_eval[:1], choice_eval[1:],
        np.random.default_rng(26), per_kind=1, device="cpu", pool_per_kind=2)
    assert neighbor_rows and sum(neighbor_counts.values()) == len(neighbor_rows)
    assert neighbor_report["adaptive"] and neighbor_report["selected_records"] == len(
        neighbor_rows)
    assert len(unique_records_by_id(choice_eval[:2], choice_eval[1:])) == 3
    swap_rec = _qa_question_swap_record(choice_norm[0], choice_norm[1])
    assert swap_rec is not None and swap_rec.facts == choice_norm[0].facts
    donor_q = tuple(choice_norm[1].tokens[
        choice_norm[1].tokens.index("question"):
        choice_norm[1].tokens.index("choices")])
    swapped_q = tuple(swap_rec.tokens[
        swap_rec.tokens.index("question"):
        swap_rec.tokens.index("choices")])
    assert donor_q == swapped_q
    q_only = _qa_ablation_record(choice_norm[0], "question")
    c_only = _qa_ablation_record(choice_norm[0], "context")
    assert q_only is not None and c_only is not None
    q_ctx_start, q_ctx_end, q_q_start, q_q_end = qa_context_question_spans(q_only)
    c_ctx_start, c_ctx_end, c_q_start, c_q_end = qa_context_question_spans(c_only)
    assert q_ctx_start == q_ctx_end and q_q_start < q_q_end
    assert c_ctx_start < c_ctx_end and c_q_start == c_q_end == 0
    replaced_choice = _choice_candidate_replacement_record(
        choice_norm[0], np.random.default_rng(13))
    assert replaced_choice is not None
    assert qa_choice_target(replaced_choice) == "none"
    assert qa_choice_spans(replaced_choice)
    assert replaced_choice.tokens != choice_norm[0].tokens
    repl_eval = qa_candidate_replacement_eval(
        choice_model, choice_vocab, choice_eval, device="cpu")
    assert repl_eval["n"] > 0
    assert "candidate_replacement_none_acc" in repl_eval
    repl_rows = choice_candidate_replacement_batch_records(
        choice_eval, np.random.default_rng(14), n=3)
    assert repl_rows and all(qa_choice_target(r) == "none" for r in repl_rows)
    repl_txt, _repl_ids = pack(repl_rows, choice_vocab, "cpu")
    assert torch.isfinite(choice_final_loss(choice_model, repl_txt, repl_rows,
                                            answer_w=0.0, none_w=1.0))
    repl_pair_rows, repl_pair_ids = choice_candidate_replacement_pair_batch(
        choice_eval, np.random.default_rng(15), pairs=2)
    assert len(repl_pair_rows) == 4 and len(repl_pair_ids) == 2
    assert {qa_choice_target(r) for r in repl_pair_rows} >= {"none"}
    repl_pair_txt, _repl_pair_ids = pack(repl_pair_rows, choice_vocab, "cpu")
    assert torch.isfinite(choice_candidate_replacement_pair_loss(
        choice_model, repl_pair_txt, repl_pair_rows, repl_pair_ids))
    assert torch.isfinite(choice_candidate_replacement_binding_loss(
        choice_model, repl_pair_txt, repl_pair_rows, repl_pair_ids))
    assert torch.isfinite(choice_candidate_replacement_answerability_binding_loss(
        choice_model, repl_pair_txt, repl_pair_rows, repl_pair_ids))
    repl_binding_eval = qa_candidate_replacement_binding_eval(
        choice_model, choice_vocab, choice_eval, device="cpu")
    assert repl_binding_eval["n"] > 0
    assert "target_logit_drop" in repl_binding_eval
    repl_ans_binding_eval = qa_candidate_replacement_answerability_binding_eval(
        choice_model, choice_vocab, choice_eval, device="cpu")
    assert repl_ans_binding_eval["n"] > 0
    assert "target_cover_drop" in repl_ans_binding_eval
    pos_anchor_rows = choice_positive_anchor_batch_records(
        choice_eval, np.random.default_rng(16), n=3)
    assert pos_anchor_rows and all(qa_choice_target(r) != "none"
                                   for r in pos_anchor_rows)
    pos_anchor_txt, _pos_anchor_ids = pack(pos_anchor_rows, choice_vocab, "cpu")
    assert torch.isfinite(choice_final_loss(choice_model, pos_anchor_txt,
                                            pos_anchor_rows,
                                            answer_w=1.0, none_w=0.0))
    bridge_eval = choice_eval + [
        TextRecord(rec_id="concept-dup", split="eval",
                   tokens=choice_eval[0].tokens, facts=choice_eval[0].facts,
                   group=choice_eval[0].group, kind=choice_eval[0].kind,
                   base_id=choice_eval[0].base_id, changed=choice_eval[0].changed,
                   meta=choice_eval[0].meta)
    ]
    concept_groups = choice_concept_groups(bridge_eval)
    assert concept_groups
    concept_rows, concept_pairs = choice_concept_bridge_batch(
        concept_groups, np.random.default_rng(17), pairs=1)
    assert len(concept_rows) == 2 and len(concept_pairs) == 1
    concept_txt, _concept_ids = pack(concept_rows, choice_vocab, "cpu")
    assert torch.isfinite(choice_concept_bridge_loss(
        choice_model, concept_txt, concept_rows, concept_pairs))
    prototype_rows, prototype_items = choice_concept_prototype_batch(
        concept_groups, np.random.default_rng(23), group_n=1, per_group=2)
    assert len(prototype_rows) == 2 and len(prototype_items) == 2
    prototype_txt, _prototype_ids = pack(prototype_rows, choice_vocab, "cpu")
    assert torch.isfinite(choice_concept_prototype_loss(
        choice_model, prototype_txt, prototype_rows, prototype_items))
    neighbor_rows, neighbor_pairs = choice_concept_neighborhood_bridge_batch(
        choice_model, choice_vocab, bridge_eval, np.random.default_rng(19), pairs=1,
        device="cpu", pool_size=4)
    assert len(neighbor_rows) == 2 and len(neighbor_pairs) == 1
    neighbor_txt, _neighbor_ids = pack(neighbor_rows, choice_vocab, "cpu")
    assert torch.isfinite(choice_concept_bridge_loss(
        choice_model, neighbor_txt, neighbor_rows, neighbor_pairs))
    transfer_rows, transfer_pairs = choice_concept_transfer_bridge_batch(
        choice_model, choice_vocab, choice_eval[:1], choice_eval[1:],
        np.random.default_rng(21), pairs=1, device="cpu", pool_per_side=2)
    assert len(transfer_rows) == 2 and len(transfer_pairs) == 1
    transfer_txt, _transfer_ids = pack(transfer_rows, choice_vocab, "cpu")
    assert torch.isfinite(choice_concept_bridge_loss(
        choice_model, transfer_txt, transfer_rows, transfer_pairs))
    assert torch.isfinite(choice_concept_transfer_bridge_loss(
        choice_model, transfer_txt, transfer_rows, transfer_pairs))
    fragile_rows, fragile_counts, fragile_selection = (
        choice_fragile_correct_records_per_kind(
            choice_model, choice_vocab, choice_eval, np.random.default_rng(27),
            per_kind=2, device="cpu", pool_per_kind=0))
    assert fragile_rows and fragile_counts
    assert fragile_selection["reason"] == "lowest_correct_choice_margin"
    neighbor_proto_rows, neighbor_proto_items = (
        choice_concept_neighborhood_prototype_batch(
            choice_model, choice_vocab, bridge_eval, np.random.default_rng(25),
            group_n=1, per_group=2, device="cpu", pool_size=4))
    assert len(neighbor_proto_rows) == 2 and len(neighbor_proto_items) == 2
    neighbor_proto_txt, _neighbor_proto_ids = pack(neighbor_proto_rows, choice_vocab,
                                                  "cpu")
    assert torch.isfinite(choice_concept_prototype_loss(
        choice_model, neighbor_proto_txt, neighbor_proto_rows, neighbor_proto_items))
    choice_teacher = copy.deepcopy(choice_model).eval()
    assert torch.isfinite(choice_self_distill_loss(
        choice_model, choice_teacher, choice_txt, choice_eval, temperature=1.5))
    assert torch.isfinite(choice_self_rank_distill_loss(
        choice_model, choice_teacher, choice_txt, choice_eval, margin=0.05))
    concept_eval = qa_concept_discovery_eval(
        choice_model, choice_vocab, bridge_eval, device="cpu", n=1, seed=18)
    assert concept_eval["n"] == 1
    assert "discovery_margin" in concept_eval
    prototype_eval = qa_concept_prototype_eval(
        choice_model, choice_vocab, bridge_eval, device="cpu", n=1, seed=24)
    assert prototype_eval["n"] >= 1
    assert "prototype_margin" in prototype_eval
    squad_choice_neg, _neg_seen, neg_stats = _squad_records_from_payload(
        squad_payload, "train", 0, np.random.default_rng(2), "fixture",
        max_context_tokens=8, max_question_tokens=8, max_answer_tokens=3,
        choice_n=2, include_extractive=False, choice_swap_negatives=1)
    assert neg_stats["choice_swap_negative_records"] == 3
    assert any(r["facts"] == [["answer", "choice", "none"]]
               and r["meta"]["negative"] == "question_swap"
               for r in squad_choice_neg)
    squad_choice_absent, _abs_seen, abs_stats = _squad_records_from_payload(
        squad_payload, "train", 0, np.random.default_rng(3), "fixture",
        max_context_tokens=8, max_question_tokens=8, max_answer_tokens=3,
        choice_n=2, include_extractive=False, choice_absent_negatives=1)
    assert abs_stats["choice_absent_negative_records"] == 3
    absent = [r for r in squad_choice_absent
              if r["kind"] == "squad_choice_absent_negative"]
    assert len(absent) == 3
    for rec in absent:
        assert rec["facts"] == [["answer", "choice", "none"]]
        assert rec["meta"]["negative"] == "answer_absent"
        assert rec["meta"]["answer_text"].lower() not in rec["meta"]["choices"]
    pair_norm = [normalize_record(r, idx=i) for i, r in enumerate(squad_choice_absent)]
    absent_norm = [r for r in pair_norm if r.kind == "squad_choice_absent_negative"]
    assert absent_norm and choice_answer_context_positions(absent_norm[0])
    anchors, anchor_counts = sample_records_per_kind(pair_norm, np.random.default_rng(9),
                                                     per_kind=1)
    assert anchors and all(n == 1 for n in anchor_counts.values())
    retention_anchors = retention_anchor_records(anchors)
    assert retention_anchors and all(r.kind.endswith(":retention_anchor")
                                     for r in retention_anchors)
    assert all(qa_choice_target(a) == qa_choice_target(r)
               for a, r in zip(anchors, retention_anchors))
    pair_groups = choice_pair_groups(pair_norm)
    assert len(pair_groups) == 3
    pair_rows = choice_pair_batch_records(pair_groups, np.random.default_rng(4), pairs=2)
    assert len(pair_rows) == 4
    assert {qa_choice_target(r) for r in pair_rows} >= {"none"}
    control_sources = choice_control_source_records(pair_norm)
    assert len(control_sources) == 3
    control_rows = choice_control_batch_records(control_sources, np.random.default_rng(5), n=4)
    assert len(control_rows) == 4
    assert all(qa_choice_target(r) == "none" for r in control_rows)
    assert all(qa_choice_spans(r) for r in control_rows)
    question_control_rows = choice_control_batch_records(
        control_sources, np.random.default_rng(20), n=3, sides=("question",))
    assert question_control_rows and all("choice_control_question" in r.kind
                                         for r in question_control_rows)
    contrast_rows, contrast_pairs = choice_control_contrast_batch(
        control_sources, np.random.default_rng(6), pairs=2)
    assert len(contrast_rows) == 4 and len(contrast_pairs) == 2
    assert all(qa_choice_target(r) != "none" for r in contrast_rows)
    swap_groups = choice_question_swap_groups(control_sources)
    ans_control_rows = choice_answerability_control_batch_records(
        control_sources, np.random.default_rng(10), n=8, swap_groups=swap_groups)
    assert len(ans_control_rows) == 8
    assert all(qa_choice_target(r) == "none" for r in ans_control_rows)
    final_control_rows = choice_final_control_batch_records(
        control_sources, np.random.default_rng(12), n=8, swap_groups=swap_groups)
    assert len(final_control_rows) == 8
    assert all(qa_choice_target(r) == "none" for r in final_control_rows)
    if swap_groups:
        swap_focus_rows = choice_answerability_control_batch_records(
            control_sources, np.random.default_rng(21), n=2, swap_groups=swap_groups,
            sides=("question_swap",))
        assert len(swap_focus_rows) == 2
        assert all("question_swap" in r.kind for r in swap_focus_rows)
        final_swap_focus_rows = choice_final_control_batch_records(
            control_sources, np.random.default_rng(22), n=2, swap_groups=swap_groups,
            sides=("question_swap",))
        assert len(final_swap_focus_rows) == 2
        assert all("question_swap" in r.kind for r in final_swap_focus_rows)
        swap_rows, swap_pairs = choice_control_contrast_batch(
            control_sources, np.random.default_rng(7), pairs=2,
            swap_groups=swap_groups)
        assert len(swap_rows) == 4 and len(swap_pairs) == 2
        qctx_rows, qctx_pairs = choice_question_context_swap_batch(
            control_sources, np.random.default_rng(8), pairs=2,
            swap_groups=swap_groups)
        assert len(qctx_rows) == 4 and len(qctx_pairs) == 2
        ans_contrast_rows, ans_contrast_pairs = choice_answerability_contrast_batch(
            control_sources, np.random.default_rng(11), pairs=2,
            swap_groups=swap_groups)
        assert len(ans_contrast_rows) == 4 and len(ans_contrast_pairs) == 2
    pair_vocab = build_vocab(pair_rows)
    pair_schema = build_fact_schema(pair_rows)
    pair_model = TextFactLM(len(pair_vocab), d=32, layers=1, heads=4,
                            pad=pair_vocab.pad, fact_schema=pair_schema).to("cpu")
    pair_txt, _pair_ids = pack(pair_rows, pair_vocab, "cpu")
    assert torch.isfinite(choice_pair_loss(pair_model, pair_txt, pair_rows))
    assert torch.isfinite(choice_answerability_pair_loss(pair_model, pair_txt,
                                                         pair_rows))
    control_vocab = build_vocab(control_rows)
    control_schema = build_fact_schema(control_rows)
    control_model = TextFactLM(len(control_vocab), d=32, layers=1, heads=4,
                               pad=control_vocab.pad, fact_schema=control_schema).to("cpu")
    control_txt, _control_ids = pack(control_rows, control_vocab, "cpu")
    assert torch.isfinite(choice_loss(control_model, control_txt, control_rows,
                                      answer_w=0.0, none_w=1.0))
    assert torch.isfinite(choice_final_loss(control_model, control_txt, control_rows,
                                            answer_w=0.0, none_w=1.0))
    assert torch.isfinite(choice_answerability_loss(control_model, control_txt,
                                                    control_rows,
                                                    answer_w=0.0, none_w=1.0))
    assert torch.isfinite(choice_candidate_answerability_loss(
        control_model, control_txt, control_rows, answer_w=0.0, none_w=1.0))
    contrast_vocab = build_vocab(contrast_rows)
    contrast_schema = build_fact_schema(contrast_rows)
    contrast_model = TextFactLM(len(contrast_vocab), d=32, layers=1, heads=4,
                                pad=contrast_vocab.pad, fact_schema=contrast_schema).to("cpu")
    contrast_txt, _contrast_ids = pack(contrast_rows, contrast_vocab, "cpu")
    assert torch.isfinite(choice_control_contrast_loss(
        contrast_model, contrast_txt, contrast_rows, contrast_pairs))
    assert torch.isfinite(choice_final_control_contrast_loss(
        contrast_model, contrast_txt, contrast_rows, contrast_pairs))
    cand_ans_scores = choice_candidate_answerability_scores(
        contrast_model, contrast_txt, contrast_rows)
    assert set(cand_ans_scores).issubset({r.rec_id for r in contrast_rows})
    assert torch.isfinite(choice_candidate_answerability_contrast_loss(
        contrast_model, contrast_txt, contrast_rows, contrast_pairs))
    if swap_groups:
        qctx_vocab = build_vocab(qctx_rows)
        qctx_schema = build_fact_schema(qctx_rows)
        qctx_model = TextFactLM(len(qctx_vocab), d=32, layers=1, heads=4,
                                pad=qctx_vocab.pad, fact_schema=qctx_schema).to("cpu")
        qctx_txt, _qctx_ids = pack(qctx_rows, qctx_vocab, "cpu")
        qctx_scores = choice_question_context_scores(qctx_model, qctx_txt, qctx_rows)
        assert set(qctx_scores).issubset({r.rec_id for r in qctx_rows})
        assert torch.isfinite(choice_question_context_contrast_loss(
            qctx_model, qctx_txt, qctx_rows, qctx_pairs))
        ans_contrast_vocab = build_vocab(ans_contrast_rows)
        ans_contrast_schema = build_fact_schema(ans_contrast_rows)
        ans_contrast_model = TextFactLM(len(ans_contrast_vocab), d=32, layers=1,
                                        heads=4, pad=ans_contrast_vocab.pad,
                                        fact_schema=ans_contrast_schema).to("cpu")
        ans_contrast_txt, _ans_contrast_ids = pack(ans_contrast_rows,
                                                   ans_contrast_vocab, "cpu")
        ans_scores = choice_answerability_scores(
            ans_contrast_model, ans_contrast_txt, ans_contrast_rows)
        assert set(ans_scores).issubset({r.rec_id for r in ans_contrast_rows})
        assert torch.isfinite(choice_answerability_contrast_loss(
            ans_contrast_model, ans_contrast_txt, ans_contrast_rows,
            ans_contrast_pairs))
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
    study_raw = {"split": "train", "id": "s1", "text": "the amber triangle is new .",
                 "facts": [["p0", "color", "amber"], ["p0", "shape", "triangle"]]}
    study_rec = normalize_record(study_raw, idx=99)
    new_vocab = build_vocab([study_rec], base_vocab=vocab)
    new_schema = build_fact_schema([study_rec], base_schema=build_fact_schema(records))
    new_model = TextFactLM(len(new_vocab), d=32, layers=2, heads=4, pad=new_vocab.pad,
                           fact_schema=new_schema).to("cpu")
    copy_pretrained_text_weights(model, vocab, new_model, new_vocab)
    red_old = vocab.stoi["red"]
    red_new = new_vocab.stoi["red"]
    assert torch.allclose(model.txt.emb.weight[red_old], new_model.txt.emb.weight[red_new])
    assert ("p0", "shape") in new_schema.keys
    assert "triangle" in new_schema.values[new_schema.keys.index(("p0", "shape"))]
    hard, hard_report = semantic_record_errors(new_model, new_vocab, records[:2],
                                               device="cpu", n=1, seed=0)
    assert hard_report["n_records"] == 1
    assert "fact_error_rate" in hard_report
    assert isinstance(hard, list)
    score_a = {"semantic_head": {"fact_value_acc": 0.8},
               "teacher_forced": {"fact_value_acc": 0.7}}
    score_b = {"semantic_head": {"fact_value_acc": 0.6},
               "teacher_forced": {"fact_value_acc": 0.7}}
    assert study_selection_score(score_a, score_b, score_a, retention_w=1.0) < 0.8
    assert abs(_score_metric(score_a, "both") - 0.75) < 1e-6
    assert abs(_score_metric(score_a, "min") - 0.7) < 1e-6
    score_choice = score_a | {
        "choice_head": {"fact_value_acc": 0.55, "n_records": 2},
        "qa_ablation_control": {
            "n": 2,
            "choice_full_minus_question_only": 0.0,
            "choice_full_minus_context_only": 0.0,
        },
        "qa_question_swap_control": {
            "n": 2,
            "choice_full_minus_question_swap": 0.0,
        },
    }
    assert abs(_score_metric(score_choice, "choice") - 0.55) < 1e-6
    assert study_selection_score(score_choice, None, metric="choice", control_w=1.0) < 0.55
    choice_components = study_selection_components(score_choice, None, metric="choice",
                                                   control_w=1.0)
    assert set(choice_components["control_failures"]) == {
        "qa_choice_full_minus_question_only",
        "qa_choice_full_minus_context_only",
        "qa_choice_full_minus_question_swap",
    }
    assert choice_control_sides_from_failures(score_choice, metric="choice") == (
        "question", "context", "question_swap")
    assert not study_selection_allowed(choice_components, control_w=1.0)["score_allowed"]
    assert study_selection_allowed(choice_components, control_w=0.0)["control_allowed"]
    score_lopsided = {"semantic_head": {"fact_value_acc": 0.95},
                      "teacher_forced": {"fact_value_acc": 0.10}}
    assert study_selection_score(score_lopsided, None, metric="both") < (
        study_selection_score(score_a, None, metric="both"))
    qa_shortcut = score_a | {"qa_ablation_control": {
        "n": 2,
        "full_minus_question_only": 0.0,
        "full_minus_context_only": 0.0,
        "semantic_full_minus_question_only": 0.0,
        "semantic_full_minus_context_only": 0.0,
    }}
    assert study_selection_score(qa_shortcut, None, metric="both", control_w=1.0) < (
        study_selection_score(score_a, None, metric="both", control_w=1.0))
    collapsed = score_a | {"by_kind": {
        "positive": {"teacher_forced_fact_value_acc": 0.0,
                     "semantic_fact_value_acc": 0.0},
        "negative": {"teacher_forced_fact_value_acc": 1.0,
                     "semantic_fact_value_acc": 1.0},
    }}
    assert study_selection_score(collapsed, None, metric="both", kind_w=1.0) < (
        study_selection_score(score_a, None, metric="both", kind_w=1.0))
    kind_before = score_choice | {"by_kind": {
        "positive": {"teacher_forced_fact_value_acc": 0.0,
                     "semantic_fact_value_acc": 0.0,
                     "choice_head_fact_value_acc": 0.25,
                     "choice_head_records": 20},
        "negative": {"teacher_forced_fact_value_acc": 0.0,
                     "semantic_fact_value_acc": 0.0,
                     "choice_head_fact_value_acc": 0.40,
                     "choice_head_records": 20},
    }}
    kind_after = score_choice | {
        "choice_head": {"fact_value_acc": 0.625, "n_records": 2},
        "by_kind": {
            "positive": {"teacher_forced_fact_value_acc": 0.0,
                         "semantic_fact_value_acc": 0.0,
                         "choice_head_fact_value_acc": 0.15,
                         "choice_head_records": 20},
            "negative": {"teacher_forced_fact_value_acc": 0.0,
                         "semantic_fact_value_acc": 0.0,
                         "choice_head_fact_value_acc": 0.90,
                         "choice_head_records": 20},
        },
    }
    regressed = study_selection_components(kind_after, None, metric="choice",
                                           control_w=0.0, kind_w=1.0,
                                           study_ref=kind_before)
    assert "positive" in regressed["kind_regressions"]
    assert regressed["score"] < study_selection_score(kind_after, None, metric="choice",
                                                      control_w=0.0, kind_w=1.0)
    assert not study_selection_allowed(regressed, control_w=0.0, kind_w=1.0)[
        "score_allowed"]
    assert study_selection_allowed({"score": -0.1}, require_positive=False)[
        "score_allowed"]
    fit_model(new_model, new_vocab, records[:2], steps=1, batch=2, lr=1e-4,
              seed=3, device="cpu", semantic_w=1.0, decode_w=0.0, prefix="study-test")
    loss = token_loss(logits, ids, pad=vocab.pad)
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())
    report = evaluate_all(model, vocab, records, device="cpu", max_new=12)
    assert set(report) >= {"teacher_forced", "free_decode", "paraphrase_consistency",
                           "counterfactual", "semantic_head", "choice_head", "by_kind",
                           "fact_concept_geometry", "free_decode_by_kind", "qa_ablation_control",
                           "qa_question_swap_control",
                           "qa_candidate_replacement_control",
                           "qa_candidate_replacement_binding_control",
                           "qa_candidate_replacement_answerability_binding_control",
                           "qa_concept_discovery_control",
                           "qa_concept_prototype_control", "gate"}
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
    ap.add_argument("--import-squad", action="store_true",
                    help="import SQuAD context/question pairs as answer-token facts")
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
    ap.add_argument("--squad-train-url", default=SQUAD_TRAIN_URL)
    ap.add_argument("--squad-eval-url", default=SQUAD_EVAL_URL)
    ap.add_argument("--squad-train-file", default=None)
    ap.add_argument("--squad-eval-file", default=None)
    ap.add_argument("--squad-train", type=int, default=5000)
    ap.add_argument("--squad-eval", type=int, default=1000)
    ap.add_argument("--squad-max-context-tokens", type=int, default=160)
    ap.add_argument("--squad-max-question-tokens", type=int, default=40)
    ap.add_argument("--squad-max-answer-tokens", type=int, default=8)
    ap.add_argument("--squad-choice-n", type=int, default=0,
                    help="also create SQuAD multiple-choice answer records with this many choices")
    ap.add_argument("--squad-choice-only", action="store_true",
                    help="for --import-squad, write only multiple-choice SQuAD records")
    ap.add_argument("--squad-choice-swap-negatives", type=int, default=0,
                    help="add this many same-paragraph swapped-question none-choice records")
    ap.add_argument("--squad-choice-absent-negatives", type=int, default=0,
                    help="add this many same-question none-choice records without the gold answer")
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
    ap.add_argument("--text-encoder-arch", choices=TEXT_ENCODER_ARCHES,
                    default="transformer",
                    help=("text encoder architecture: transformer keeps the legacy encoder; "
                          "relational/abstractor use symbolic-value attention"))
    ap.add_argument("--text-encoder-layers", type=int, default=1,
                    help="number of bidirectional text encoder layers before concept heads")
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
    ap.add_argument("--fact-concept-w", type=float, default=0.0,
                    dest="fact_concept_w",
                    help=("weight for schema-generic fact/value concept embedding "
                          "auxiliary loss"))
    ap.add_argument("--fact-concept-contrast-w", type=float, default=0.0,
                    dest="fact_concept_contrast_w",
                    help=("weight for schema-generic same-value concept geometry "
                          "contrastive loss"))
    ap.add_argument("--fact-concept-contrast-temperature", type=float, default=0.1,
                    dest="fact_concept_contrast_temperature",
                    help="temperature for schema concept geometry contrastive loss")
    ap.add_argument("--fact-concept-centroid-w", type=float, default=0.0,
                    dest="fact_concept_centroid_w",
                    help="weight for batch-discovered same-value concept centroid loss")
    ap.add_argument("--fact-concept-centroid-temperature", type=float, default=0.1,
                    dest="fact_concept_centroid_temperature",
                    help="temperature for batch-discovered concept centroid loss")
    ap.add_argument("--fact-concept-centroid-margin", type=float, default=0.0,
                    dest="fact_concept_centroid_margin",
                    help="minimum margin between own centroid and other value centroids")
    ap.add_argument("--fact-concept-prefix", action="store_true",
                    dest="fact_concept_prefix",
                    help="prepend schema concept states to the text decoder prefix")
    ap.add_argument("--fact-concept-refine", action="store_true",
                    dest="fact_concept_refine",
                    help=("refine upstream text states with learned schema concept "
                          "feedback before decoding and concept heads"))
    ap.add_argument("--fact-concept-refine-gate-init", type=float, default=-2.0,
                    dest="fact_concept_refine_gate_init",
                    help="initial logit for the learned concept-refinement residual gate")
    ap.add_argument("--fact-concept-prototype-w", type=float, default=0.0,
                    dest="fact_concept_prototype_w",
                    help="weight for schema-generic learned concept prototype loss")
    ap.add_argument("--fact-concept-prototype-spread-w", type=float, default=0.0,
                    dest="fact_concept_prototype_spread_w",
                    help="weight for separating learned concept prototypes per schema key")
    ap.add_argument("--fact-concept-prototype-spread-margin", type=float, default=0.2,
                    dest="fact_concept_prototype_spread_margin",
                    help="maximum allowed cosine similarity between value prototypes")
    ap.add_argument("--fact-concept-state-spread-w", type=float, default=0.0,
                    dest="fact_concept_state_spread_w",
                    help="weight for anti-collapse spread regularization on concept states")
    ap.add_argument("--fact-concept-state-spread-variance", type=float, default=0.05,
                    dest="fact_concept_state_spread_variance",
                    help="minimum normalized per-dimension concept-state std target")
    ap.add_argument("--fact-concept-state-spread-margin", type=float, default=0.2,
                    dest="fact_concept_state_spread_margin",
                    help="maximum same-key centroid cosine before state spread penalty")
    ap.add_argument("--fact-concept-state-spread-covariance-w", type=float, default=0.05,
                    dest="fact_concept_state_spread_covariance_w",
                    help="relative decorrelation weight inside concept-state spread loss")
    ap.add_argument("--decode-w", type=float, default=1.0, dest="decode_w",
                    help="weight for canonical trace decoder loss; set 0 for semantic-only study")
    ap.add_argument("--choice-w", type=float, default=0.0, dest="choice_w",
                    help="weight for candidate-aware QA choice loss")
    ap.add_argument("--choice-answer-w", type=float, default=1.0, dest="choice_answer_w",
                    help="relative loss weight for non-none QA choice targets")
    ap.add_argument("--choice-none-w", type=float, default=1.0, dest="choice_none_w",
                    help="relative loss weight for none QA choice targets")
    ap.add_argument("--choice-answer-margin", type=float, default=0.0,
                    dest="choice_answer_margin",
                    help="margin by which positive QA choices must clear the none threshold")
    ap.add_argument("--choice-none-margin", type=float, default=0.0,
                    dest="choice_none_margin",
                    help="margin by which none QA records must keep candidates below threshold")
    ap.add_argument("--choice-final-w", type=float, default=0.0,
                    dest="choice_final_w",
                    help="weight for final QA choice logits including answerability")
    ap.add_argument("--choice-final-control-w", type=float, default=0.0,
                    dest="choice_final_control_w",
                    help=("weight for generated question/context/swap controls on final "
                          "QA choice logits"))
    ap.add_argument("--choice-final-margin", type=float, default=0.0,
                    dest="choice_final_margin",
                    help="margin for final QA choice logits")
    ap.add_argument("--choice-final-control-contrast-w", type=float, default=0.0,
                    dest="choice_final_control_contrast_w",
                    help=("weight for full-vs-control contrast on final QA "
                          "choice logits"))
    ap.add_argument("--choice-final-control-contrast-margin", type=float, default=0.0,
                    dest="choice_final_control_contrast_margin",
                    help="margin for full-vs-control contrast on final QA choice logits")
    ap.add_argument("--choice-candidate-replacement-w", type=float, default=0.0,
                    dest="choice_candidate_replacement_w",
                    help=("weight for generated same-context candidate-corruption "
                          "none loss on final QA choice logits"))
    ap.add_argument("--choice-candidate-replacement-margin", type=float, default=0.0,
                    dest="choice_candidate_replacement_margin",
                    help="margin for generated candidate-corruption none loss")
    ap.add_argument("--choice-candidate-replacement-pair-w", type=float, default=0.0,
                    dest="choice_candidate_replacement_pair_w",
                    help=("weight for paired original-vs-corrupted-candidate calibration "
                          "on final QA choice logits"))
    ap.add_argument("--choice-candidate-replacement-pair-margin", type=float, default=0.0,
                    dest="choice_candidate_replacement_pair_margin",
                    help="margin for paired original-vs-corrupted-candidate calibration")
    ap.add_argument("--choice-candidate-replacement-binding-w", type=float, default=0.0,
                    dest="choice_candidate_replacement_binding_w",
                    help=("weight for candidate-corruption target-evidence drop "
                          "on final QA choice logits"))
    ap.add_argument("--choice-candidate-replacement-binding-margin", type=float,
                    default=0.0,
                    dest="choice_candidate_replacement_binding_margin",
                    help="margin for candidate-corruption target-evidence drop")
    ap.add_argument("--choice-candidate-replacement-answerability-binding-w",
                    type=float, default=0.0,
                    dest="choice_candidate_replacement_answerability_binding_w",
                    help=("weight for candidate-corruption target-coverage drop "
                          "on QA answerability logits"))
    ap.add_argument("--choice-candidate-replacement-answerability-binding-margin",
                    type=float, default=0.0,
                    dest="choice_candidate_replacement_answerability_binding_margin",
                    help="margin for candidate-corruption target-coverage drop")
    ap.add_argument("--choice-positive-anchor-w", type=float, default=0.0,
                    dest="choice_positive_anchor_w",
                    help=("weight for answer-present positive anchor loss on final "
                          "QA choice logits"))
    ap.add_argument("--choice-positive-anchor-margin", type=float, default=0.0,
                    dest="choice_positive_anchor_margin",
                    help="margin for answer-present positive anchor loss")
    ap.add_argument("--choice-concept-bridge-w", type=float, default=0.0,
                    dest="choice_concept_bridge_w",
                    help=("weight for mined same-concept candidate bridge loss "
                          "across QA records"))
    ap.add_argument("--choice-concept-bridge-margin", type=float, default=0.0,
                    dest="choice_concept_bridge_margin",
                    help="margin for mined same-concept candidate bridge loss")
    ap.add_argument("--choice-concept-prototype-w", type=float, default=0.0,
                    dest="choice_concept_prototype_w",
                    help=("weight for repeated-concept prototype clustering in "
                          "QA candidate-vector space"))
    ap.add_argument("--choice-concept-prototype-margin", type=float, default=0.0,
                    dest="choice_concept_prototype_margin",
                    help="margin for repeated-concept prototype clustering")
    ap.add_argument("--choice-self-distill-w", "--choice-distill-w",
                    type=float, default=0.0,
                    dest="choice_self_distill_w",
                    help=("weight for preserving a frozen own-model QA choice "
                          "distribution on mined correct records during study"))
    ap.add_argument("--choice-self-distill-temperature", "--choice-distill-temperature",
                    type=float, default=1.0,
                    dest="choice_self_distill_temperature",
                    help="temperature for own-model QA choice distribution distillation")
    ap.add_argument("--choice-self-rank-distill-w", "--choice-rank-distill-w",
                    type=float, default=0.0,
                    dest="choice_self_rank_distill_w",
                    help=("weight for preserving a frozen own-model correct-choice "
                          "winner and margin during study"))
    ap.add_argument("--choice-self-rank-distill-margin",
                    "--choice-rank-distill-margin",
                    type=float, default=0.0,
                    dest="choice_self_rank_distill_margin",
                    help="minimum margin for own-model correct-choice rank distillation")
    ap.add_argument("--choice-answerability-w", type=float, default=0.0,
                    dest="choice_answerability_w",
                    help="weight for learned QA answerability coverage loss")
    ap.add_argument("--choice-answerability-control-w", type=float, default=0.0,
                    dest="choice_answerability_control_w",
                    help="weight for generated ablation none-answerability loss")
    ap.add_argument("--choice-answerability-margin", type=float, default=0.0,
                    dest="choice_answerability_margin",
                    help="margin for learned QA answerability coverage loss")
    ap.add_argument("--choice-answerability-contrast-w", type=float, default=0.0,
                    dest="choice_answerability_contrast_w",
                    help="weight for full-vs-swapped question answerability contrast")
    ap.add_argument("--choice-answerability-contrast-margin", type=float, default=0.0,
                    dest="choice_answerability_contrast_margin",
                    help="margin for full-vs-swapped question answerability contrast")
    ap.add_argument("--choice-candidate-answerability-w", type=float, default=0.0,
                    dest="choice_candidate_answerability_w",
                    help="weight for candidate-specific QA answerability coverage loss")
    ap.add_argument("--choice-candidate-answerability-control-w", type=float, default=0.0,
                    dest="choice_candidate_answerability_control_w",
                    help="weight for generated ablation candidate-answerability controls")
    ap.add_argument("--choice-candidate-answerability-margin", type=float, default=0.0,
                    dest="choice_candidate_answerability_margin",
                    help="margin for candidate-specific QA answerability coverage loss")
    ap.add_argument("--choice-candidate-answerability-contrast-w", type=float, default=0.0,
                    dest="choice_candidate_answerability_contrast_w",
                    help=("weight for full-vs-control candidate-answerability "
                          "coverage contrast"))
    ap.add_argument("--choice-candidate-answerability-contrast-margin", type=float,
                    default=0.0,
                    dest="choice_candidate_answerability_contrast_margin",
                    help="margin for full-vs-control candidate-answerability coverage")
    ap.add_argument("--choice-answerability-pair-w", type=float, default=0.0,
                    dest="choice_answerability_pair_w",
                    help=("weight for paired answer-present vs answer-absent "
                          "QA answerability contrast"))
    ap.add_argument("--choice-answerability-pair-margin", type=float, default=0.0,
                    dest="choice_answerability_pair_margin",
                    help=("margin for paired answer-present vs answer-absent "
                          "QA answerability contrast"))
    ap.add_argument("--choice-context-w", type=float, default=0.0, dest="choice_context_w",
                    help="weight for target-choice context span localization loss")
    ap.add_argument("--choice-candidate-context-w", type=float, default=0.0,
                    dest="choice_candidate_context_w",
                    help=("weight for candidate-vs-distractor answer-span localization "
                          "contrast loss"))
    ap.add_argument("--choice-candidate-context-margin", type=float, default=0.0,
                    dest="choice_candidate_context_margin",
                    help="margin for candidate-vs-distractor answer-span localization contrast")
    ap.add_argument("--choice-question-context-w", type=float, default=0.0,
                    dest="choice_question_context_w",
                    help="weight for question-to-answer-span context binding loss")
    ap.add_argument("--choice-question-context-contrast-w", type=float, default=0.0,
                    dest="choice_question_context_contrast_w",
                    help="weight for full-vs-swapped question context-binding contrast loss")
    ap.add_argument("--choice-question-context-margin", type=float, default=0.0,
                    dest="choice_question_context_margin",
                    help="margin for full-vs-swapped question context-binding contrast loss")
    ap.add_argument("--choice-pair-w", type=float, default=0.0, dest="choice_pair_w",
                    help="weight for paired positive-vs-none QA choice evidence loss")
    ap.add_argument("--choice-pair-margin", type=float, default=0.0,
                    dest="choice_pair_margin",
                    help="margin for paired positive-vs-none QA choice evidence loss")
    ap.add_argument("--choice-control-w", type=float, default=0.0,
                    dest="choice_control_w",
                    help="weight for generated question/context ablation none-choice loss")
    ap.add_argument("--choice-control-contrast-w", type=float, default=0.0,
                    dest="choice_control_contrast_w",
                    help="weight for full-vs-ablation target-evidence contrast loss")
    ap.add_argument("--choice-control-margin", type=float, default=0.0,
                    dest="choice_control_margin",
                    help="margin for full-vs-ablation target-evidence contrast loss")
    ap.add_argument("--out", default="runs/text0.json")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-checkpoint", default=None,
                    help="load a text checkpoint and evaluate it on --data without training")
    ap.add_argument("--study-checkpoint", default=None,
                    help="load a text checkpoint, continue training on --data, and report before/after")
    ap.add_argument("--study-out-checkpoint", default=None,
                    help="where to save the studied checkpoint; defaults to --checkpoint when set")
    ap.add_argument("--study-replay-data", action="append", default=None,
                    help="optional replay JSON/JSONL records mixed into study training")
    ap.add_argument("--study-lr", type=float, default=5e-4,
                    help="learning rate for --study-checkpoint")
    ap.add_argument("--study-rounds", type=int, default=1,
                    help="number of study rounds; error strategy re-mines hard examples each round")
    ap.add_argument("--study-strategy", choices=("all", "errors"), default="all",
                    help=("train on all study records or model-missed records; uses choice "
                          "errors when --study-score-metric choice is set"))
    ap.add_argument("--study-probe-n", type=int, default=0,
                    help="sample this many train records when mining errors; 0 = all")
    ap.add_argument("--study-hard-max", type=int, default=0,
                    help="cap hard examples used per round; 0 = no cap")
    ap.add_argument("--study-anchor-correct-per-kind", type=int, default=0,
                    help=("for error self-study, mix in this many currently-correct "
                          "study records per kind as retention anchors"))
    ap.add_argument("--study-anchor-correct-repeat", type=int, default=1,
                    help=("repeat selected currently-correct retention anchors this "
                          "many times in each study round"))
    ap.add_argument("--study-anchor-retention-bucket", action="store_true",
                    help=("put selected correct retention anchors in their own "
                          "training kind buckets when balancing by kind"))
    ap.add_argument("--study-distill-correct-per-kind", type=int, default=0,
                    help=("for error self-study, preserve this many currently-correct "
                          "choice records per kind with own-model distillation"))
    ap.add_argument("--study-discovery-correct-per-kind", type=int, default=0,
                    help=("for error self-study, use this many currently-correct "
                          "choice records per kind as concept-bridge discovery sources"))
    ap.add_argument("--study-fragile-correct-mining", action="store_true",
                    help=("select correct distillation records with the smallest "
                          "own-model target margins"))
    ap.add_argument("--study-adaptive-correct-mining", action="store_true",
                    help=("select correct self-teaching records nearest to hard examples "
                          "in own-model concept space"))
    ap.add_argument("--study-adaptive-correct-pool-per-kind", type=int, default=0,
                    help=("cap correct-record candidates per kind before adaptive mining; "
                          "0 = use all currently-correct candidates"))
    ap.add_argument("--study-discovery-transfer", action="store_true",
                    help=("during study, pair current hard QA examples with nearest "
                          "currently-correct concept neighbors for bridge training"))
    ap.add_argument("--study-focus-control-failures", action="store_true",
                    help=("during study, focus generated controls on control families "
                          "that failed the initial gate"))
    ap.add_argument("--study-select-best", action="store_true",
                    help="evaluate each study round and restore the best scoring weights")
    ap.add_argument("--study-score-metric", choices=("semantic", "teacher", "concept",
                                                     "both", "min", "choice"),
                    default="both",
                    help="metric used by --study-select-best")
    ap.add_argument("--study-retention-w", type=float, default=1.0,
                    help="penalty weight for replay metric drops during study selection")
    ap.add_argument("--study-control-w", type=float, default=1.0,
                    help="penalty weight for NLI/QA shortcut-control gaps during study selection")
    ap.add_argument("--study-kind-w", type=float, default=1.0,
                    help="penalty weight for low-performing eval kinds during study selection")
    ap.add_argument("--study-allow-negative-score", action="store_true",
                    help="allow --study-select-best to keep a round with non-positive score")
    ap.add_argument("--study-confirm-n", type=int, default=0,
                    help=("with --study-select-best, require this many extra held-out "
                          "evaluation seeds to pass before accepting a study round"))
    ap.add_argument("--study-confirm-seed-stride", type=int, default=7919,
                    help="seed stride for --study-confirm-n confirmation evaluations")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    for name, value in {
        "--steps": args.steps, "--batch": args.batch, "--d": args.d,
        "--layers": args.layers, "--heads": args.heads,
        "--text-encoder-layers": args.text_encoder_layers,
    }.items():
        if value <= 0:
            ap.error(f"{name} must be positive")
    if args.d % args.heads != 0:
        ap.error("--d must be divisible by --heads")
    if (args.text_encoder_arch in ("relational", "abstractor")
            and args.heads % 2 != 0):
        ap.error("--text-encoder-arch relational/abstractor require an even --heads value")
    if (args.fact_concept_w < 0.0 or args.fact_concept_contrast_w < 0.0
            or args.fact_concept_centroid_w < 0.0
            or args.fact_concept_prototype_w < 0.0
            or args.fact_concept_prototype_spread_w < 0.0
            or args.fact_concept_state_spread_w < 0.0):
        ap.error("fact concept loss weights must be non-negative")
    if args.fact_concept_contrast_temperature <= 0.0:
        ap.error("--fact-concept-contrast-temperature must be positive")
    if args.fact_concept_centroid_temperature <= 0.0:
        ap.error("--fact-concept-centroid-temperature must be positive")
    if args.fact_concept_centroid_margin < 0.0:
        ap.error("--fact-concept-centroid-margin must be non-negative")
    if args.fact_concept_prototype_spread_margin < -1.0:
        ap.error("--fact-concept-prototype-spread-margin must be >= -1")
    if args.fact_concept_state_spread_variance < 0.0:
        ap.error("--fact-concept-state-spread-variance must be non-negative")
    if args.fact_concept_state_spread_margin < -1.0:
        ap.error("--fact-concept-state-spread-margin must be >= -1")
    if args.fact_concept_state_spread_covariance_w < 0.0:
        ap.error("--fact-concept-state-spread-covariance-w must be non-negative")
    if not math.isfinite(args.fact_concept_refine_gate_init):
        ap.error("--fact-concept-refine-gate-init must be finite")
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
    if args.import_squad:
        import_squad(args.out,
                     train_source=args.squad_train_file or args.squad_train_url,
                     eval_source=args.squad_eval_file or args.squad_eval_url,
                     max_train=args.squad_train, max_eval=args.squad_eval,
                     seed=args.seed,
                     max_context_tokens=args.squad_max_context_tokens,
                     max_question_tokens=args.squad_max_question_tokens,
                     max_answer_tokens=args.squad_max_answer_tokens,
                     choice_n=args.squad_choice_n,
                     choice_only=args.squad_choice_only,
                     choice_swap_negatives=args.squad_choice_swap_negatives,
                     choice_absent_negatives=args.squad_choice_absent_negatives)
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
    if args.study_checkpoint:
        study_checkpoint(args.study_checkpoint, args.data,
                         out_checkpoint=args.study_out_checkpoint or args.checkpoint,
                         replay_data=args.study_replay_data, out=args.out,
                         steps=args.steps, batch=args.batch, lr=args.study_lr,
                         seed=args.seed, device=DEV, max_new=args.max_new,
                         semantic_w=args.semantic_w, free_n=args.free_n,
                         paraphrase_n=args.paraphrase_n,
                         counterfactual_n=args.counterfactual_n,
                         kind_free_n=args.kind_free_n, balance_by=args.balance_by,
                         fact_n=args.fact_n, kind_fact_n=args.kind_fact_n,
                         artifact_n=args.artifact_n, decode_w=args.decode_w,
                         fact_concept_w=args.fact_concept_w,
                         fact_concept_contrast_w=args.fact_concept_contrast_w,
                         fact_concept_contrast_temperature=(
                             args.fact_concept_contrast_temperature),
                         fact_concept_centroid_w=args.fact_concept_centroid_w,
                         fact_concept_centroid_temperature=(
                             args.fact_concept_centroid_temperature),
                         fact_concept_centroid_margin=(
                             args.fact_concept_centroid_margin),
                         fact_concept_prefix=args.fact_concept_prefix,
                         fact_concept_refine=args.fact_concept_refine,
                         fact_concept_refine_gate_init=(
                             args.fact_concept_refine_gate_init),
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
                         choice_w=args.choice_w,
                         choice_answer_w=args.choice_answer_w,
                         choice_none_w=args.choice_none_w,
                         choice_answer_margin=args.choice_answer_margin,
                         choice_none_margin=args.choice_none_margin,
                         choice_final_w=args.choice_final_w,
                         choice_final_control_w=args.choice_final_control_w,
                         choice_final_margin=args.choice_final_margin,
                         choice_final_control_contrast_w=(
                             args.choice_final_control_contrast_w),
                         choice_final_control_contrast_margin=(
                             args.choice_final_control_contrast_margin),
                         choice_candidate_replacement_w=(
                             args.choice_candidate_replacement_w),
                         choice_candidate_replacement_margin=(
                             args.choice_candidate_replacement_margin),
                         choice_candidate_replacement_pair_w=(
                             args.choice_candidate_replacement_pair_w),
                         choice_candidate_replacement_pair_margin=(
                             args.choice_candidate_replacement_pair_margin),
                         choice_candidate_replacement_binding_w=(
                             args.choice_candidate_replacement_binding_w),
                         choice_candidate_replacement_binding_margin=(
                             args.choice_candidate_replacement_binding_margin),
                         choice_candidate_replacement_answerability_binding_w=(
                             args.choice_candidate_replacement_answerability_binding_w),
                         choice_candidate_replacement_answerability_binding_margin=(
                             args.choice_candidate_replacement_answerability_binding_margin),
                         choice_positive_anchor_w=args.choice_positive_anchor_w,
                         choice_positive_anchor_margin=(
                             args.choice_positive_anchor_margin),
                         choice_concept_bridge_w=args.choice_concept_bridge_w,
                         choice_concept_bridge_margin=(
                             args.choice_concept_bridge_margin),
                         choice_concept_prototype_w=(
                             args.choice_concept_prototype_w),
                         choice_concept_prototype_margin=(
                             args.choice_concept_prototype_margin),
                         choice_self_distill_w=args.choice_self_distill_w,
                         choice_self_distill_temperature=(
                             args.choice_self_distill_temperature),
                         choice_self_rank_distill_w=(
                             args.choice_self_rank_distill_w),
                         choice_self_rank_distill_margin=(
                             args.choice_self_rank_distill_margin),
                         choice_answerability_w=args.choice_answerability_w,
                         choice_answerability_control_w=(
                             args.choice_answerability_control_w),
                         choice_answerability_margin=args.choice_answerability_margin,
                         choice_answerability_contrast_w=(
                             args.choice_answerability_contrast_w),
                         choice_answerability_contrast_margin=(
                             args.choice_answerability_contrast_margin),
                         choice_candidate_answerability_w=(
                             args.choice_candidate_answerability_w),
                         choice_candidate_answerability_control_w=(
                             args.choice_candidate_answerability_control_w),
                         choice_candidate_answerability_margin=(
                             args.choice_candidate_answerability_margin),
                         choice_candidate_answerability_contrast_w=(
                             args.choice_candidate_answerability_contrast_w),
                         choice_candidate_answerability_contrast_margin=(
                             args.choice_candidate_answerability_contrast_margin),
                         choice_answerability_pair_w=args.choice_answerability_pair_w,
                         choice_answerability_pair_margin=(
                             args.choice_answerability_pair_margin),
                         choice_context_w=args.choice_context_w,
                         choice_candidate_context_w=args.choice_candidate_context_w,
                         choice_candidate_context_margin=(
                             args.choice_candidate_context_margin),
                         choice_question_context_w=args.choice_question_context_w,
                         choice_question_context_contrast_w=(
                             args.choice_question_context_contrast_w),
                         choice_question_context_margin=args.choice_question_context_margin,
                         choice_pair_w=args.choice_pair_w,
                         choice_pair_margin=args.choice_pair_margin,
                         choice_control_w=args.choice_control_w,
                         choice_control_contrast_w=args.choice_control_contrast_w,
                         choice_control_margin=args.choice_control_margin,
                         study_rounds=args.study_rounds,
                         study_strategy=args.study_strategy,
                         study_probe_n=args.study_probe_n,
                         study_hard_max=args.study_hard_max,
                         study_anchor_correct_per_kind=args.study_anchor_correct_per_kind,
                         study_anchor_correct_repeat=args.study_anchor_correct_repeat,
                         study_anchor_retention_bucket=(
                             args.study_anchor_retention_bucket),
                         study_distill_correct_per_kind=(
                             args.study_distill_correct_per_kind),
                         study_discovery_correct_per_kind=(
                             args.study_discovery_correct_per_kind),
                         study_fragile_correct_mining=(
                             args.study_fragile_correct_mining),
                         study_adaptive_correct_mining=(
                             args.study_adaptive_correct_mining),
                         study_adaptive_correct_pool_per_kind=(
                             args.study_adaptive_correct_pool_per_kind),
                         study_discovery_transfer=args.study_discovery_transfer,
                         study_focus_control_failures=args.study_focus_control_failures,
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
        max_new=args.max_new, semantic_w=args.semantic_w, free_n=args.free_n,
        paraphrase_n=args.paraphrase_n, counterfactual_n=args.counterfactual_n,
        kind_free_n=args.kind_free_n, balance_by=args.balance_by,
        fact_n=args.fact_n, kind_fact_n=args.kind_fact_n, artifact_n=args.artifact_n,
        decode_w=args.decode_w, fact_concept_w=args.fact_concept_w,
        fact_concept_contrast_w=args.fact_concept_contrast_w,
        fact_concept_contrast_temperature=args.fact_concept_contrast_temperature,
        fact_concept_centroid_w=args.fact_concept_centroid_w,
        fact_concept_centroid_temperature=args.fact_concept_centroid_temperature,
        fact_concept_centroid_margin=args.fact_concept_centroid_margin,
        fact_concept_prefix=args.fact_concept_prefix,
        fact_concept_refine=args.fact_concept_refine,
        fact_concept_refine_gate_init=args.fact_concept_refine_gate_init,
        fact_concept_prototype_w=args.fact_concept_prototype_w,
        fact_concept_prototype_spread_w=args.fact_concept_prototype_spread_w,
        fact_concept_prototype_spread_margin=(
            args.fact_concept_prototype_spread_margin),
        fact_concept_state_spread_w=args.fact_concept_state_spread_w,
        fact_concept_state_spread_variance=args.fact_concept_state_spread_variance,
        fact_concept_state_spread_margin=args.fact_concept_state_spread_margin,
        fact_concept_state_spread_covariance_w=(
            args.fact_concept_state_spread_covariance_w),
        choice_w=args.choice_w,
        choice_answer_w=args.choice_answer_w, choice_none_w=args.choice_none_w,
        choice_answer_margin=args.choice_answer_margin,
        choice_none_margin=args.choice_none_margin,
        choice_final_w=args.choice_final_w,
        choice_final_control_w=args.choice_final_control_w,
        choice_final_margin=args.choice_final_margin,
        choice_final_control_contrast_w=args.choice_final_control_contrast_w,
        choice_final_control_contrast_margin=(
            args.choice_final_control_contrast_margin),
        choice_candidate_replacement_w=args.choice_candidate_replacement_w,
        choice_candidate_replacement_margin=args.choice_candidate_replacement_margin,
        choice_candidate_replacement_pair_w=args.choice_candidate_replacement_pair_w,
        choice_candidate_replacement_pair_margin=(
            args.choice_candidate_replacement_pair_margin),
        choice_candidate_replacement_binding_w=(
            args.choice_candidate_replacement_binding_w),
        choice_candidate_replacement_binding_margin=(
            args.choice_candidate_replacement_binding_margin),
        choice_candidate_replacement_answerability_binding_w=(
            args.choice_candidate_replacement_answerability_binding_w),
        choice_candidate_replacement_answerability_binding_margin=(
            args.choice_candidate_replacement_answerability_binding_margin),
        choice_positive_anchor_w=args.choice_positive_anchor_w,
        choice_positive_anchor_margin=args.choice_positive_anchor_margin,
        choice_concept_bridge_w=args.choice_concept_bridge_w,
        choice_concept_bridge_margin=args.choice_concept_bridge_margin,
        choice_concept_prototype_w=args.choice_concept_prototype_w,
        choice_concept_prototype_margin=args.choice_concept_prototype_margin,
        choice_answerability_w=args.choice_answerability_w,
        choice_answerability_control_w=args.choice_answerability_control_w,
        choice_answerability_margin=args.choice_answerability_margin,
        choice_answerability_contrast_w=args.choice_answerability_contrast_w,
        choice_answerability_contrast_margin=args.choice_answerability_contrast_margin,
        choice_candidate_answerability_w=args.choice_candidate_answerability_w,
        choice_candidate_answerability_control_w=(
            args.choice_candidate_answerability_control_w),
        choice_candidate_answerability_margin=args.choice_candidate_answerability_margin,
        choice_candidate_answerability_contrast_w=(
            args.choice_candidate_answerability_contrast_w),
        choice_candidate_answerability_contrast_margin=(
            args.choice_candidate_answerability_contrast_margin),
        choice_answerability_pair_w=args.choice_answerability_pair_w,
        choice_answerability_pair_margin=args.choice_answerability_pair_margin,
        choice_context_w=args.choice_context_w,
        choice_candidate_context_w=args.choice_candidate_context_w,
        choice_candidate_context_margin=args.choice_candidate_context_margin,
        choice_question_context_w=args.choice_question_context_w,
        choice_question_context_contrast_w=args.choice_question_context_contrast_w,
        choice_question_context_margin=args.choice_question_context_margin,
        choice_pair_w=args.choice_pair_w,
        choice_pair_margin=args.choice_pair_margin,
        choice_control_w=args.choice_control_w,
        choice_control_contrast_w=args.choice_control_contrast_w,
        choice_control_margin=args.choice_control_margin)


if __name__ == "__main__":
    main()
