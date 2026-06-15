"""Schema-free text reading and latent concept discovery.

The supported text entrypoint is raw reading via ``--reading-data``.  Text is read as
corpus chunks and trained with latent slots, concept memory, self-study, reanalysis,
memory-gap learning, and graph-closure insight.  There are no English rulebooks,
dataset-specific import handlers, answer-choice handlers, or structured fact-training
CLIs in the active entrypoint.

Checkpoint compatibility code remains internal, but old dataset-specific importers have
been removed.
"""
import argparse
import copy
import json
import math
import os
import re
import tempfile
import urllib.request
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
    SchemaConceptRefiner,
    latent_concept_bridge_loss,
    latent_concept_bridge_scores,
    latent_concept_completion_loss,
    latent_concept_completion_scores,
    latent_concept_composition_loss,
    latent_concept_graph_cycle_loss,
    latent_concept_graph_cycle_scores,
    latent_concept_graph_prediction_loss,
    latent_concept_graph_prediction_scores,
    latent_concept_graph_curiosity_scores,
    latent_concept_insight_scores,
    latent_concept_discovery_loss,
    latent_concept_cluster_prototype_loss,
    latent_concept_fer_loss,
    latent_concept_fer_metrics,
    latent_concept_fer_scores,
    latent_concept_graph_ready,
    latent_concept_graph_snapshot,
    latent_concept_memory_gap_loss,
    latent_concept_memory_gap_scores,
    latent_concept_memory_consolidation_loss,
    latent_concept_neighborhood_loss,
    latent_concept_reanalysis_loss,
    latent_concept_sequence_prediction_loss,
    latent_concept_sequence_prediction_scores,
    latent_concept_slot_factorization_loss,
    latent_concept_transition_consistency_loss,
    latent_concept_vicreg_loss,
)
from .selection import concept_round_selection_decision, signal_regression_report
from .trace import Vocab

DEV = get_device()
TOKEN_RE = re.compile(r"[a-z0-9_]+|[^\s\w]", re.ASCII)
READING_DEFAULT_LATENT_CONCEPT_SLOTS = 4
READING_REPLAY_BANK_VERSION = 1
READING_REPLAY_BANK_SIZE = 128
READING_MASTERY_HISTORY_VERSION = 1
READING_MASTERY_HISTORY_SIZE = 32
READING_DEFAULT_MAX_VOCAB = 32768
READING_DEFAULT_SOURCE_BALANCE_W = 0.5
READING_MASTERY_SCORE_KEYS = (
    "score", "mastery_score", "active_mean_score", "signal_coverage",
    "balanced_score", "floor_score", "view_score", "fer_score",
    "bridge_score", "bridge_connectivity", "sequence_score", "context_score",
    "span_score", "context_closure_score", "neighborhood_score",
    "cluster_score", "connection_score", "language_score", "generation_score",
    "generation_token_acc", "generation_diversity",
    "generation_collapse_penalty",
)
READING_REPRESENTATION_PROGRESS_KEYS = (
    "mastery_score", "active_mean_score", "floor_score", "balanced_score",
    "signal_coverage", "fer_score", "bridge_score", "bridge_connectivity",
    "connection_score", "sequence_score", "neighborhood_score", "cluster_score",
    "span_score", "context_closure_score", "language_score",
    "generation_score")
READING_REPRESENTATION_SIGNAL_SCORES = (
    ("language", "language_score"),
    ("generation", "generation_score"),
    ("fer", "fer_score"),
    ("bridge", "bridge_score"),
    ("connection", "connection_score"),
    ("sequence", "sequence_score"),
    ("neighborhood", "neighborhood_score"),
    ("cluster", "cluster_score"),
    ("span", "span_score"),
    ("closure", "context_closure_score"),
)


@dataclass(frozen=True)
class ReadingRecord:
    rec_id: str
    split: str
    tokens: tuple[str, ...]
    kind: str = "raw_text"
    meta: dict = field(default_factory=dict)


def split_words(text):
    return TOKEN_RE.findall(str(text).lower())


def split_words_with_spans(text):
    text = str(text).lower()
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _jsonable_meta(meta):
    if not isinstance(meta, dict):
        return {}
    return json.loads(json.dumps(meta, default=str))


def reading_record_to_bank_row(rec, replay_meta=None):
    replay_meta = dict(replay_meta or {})
    if not replay_meta and isinstance(rec.meta, dict):
        replay_meta = {
            "priority": rec.meta.get("replay_priority", 0.0),
            "reasons": rec.meta.get("replay_reasons", ()),
        }
    return {
        "id": str(rec.rec_id),
        "split": str(rec.split),
        "tokens": list(rec.tokens),
        "kind": str(rec.kind),
        "meta": _jsonable_meta(rec.meta),
        "replay_priority": float(replay_meta.get("priority", 0.0)),
        "replay_reasons": list(replay_meta.get("reasons", ())),
    }


def reading_record_from_bank_row(row, idx=0):
    if not isinstance(row, dict):
        raise ValueError(f"reading replay bank row {idx} must be an object")
    split = str(row.get("split", "train"))
    if split not in ("train", "eval"):
        split = "train"
    tokens = tuple(str(tok) for tok in row.get("tokens", ()) if str(tok))
    if not tokens:
        raise ValueError(f"reading replay bank row {idx} has no tokens")
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    meta = meta | {"replay_bank": True}
    if "replay_priority" in row:
        meta["replay_priority"] = float(row.get("replay_priority", 0.0))
    if "replay_reasons" in row:
        meta["replay_reasons"] = list(row.get("replay_reasons", ()))
    return ReadingRecord(
        rec_id=str(row.get("id", f"reading-replay-{idx}")),
        split=split,
        tokens=tokens,
        kind=str(row.get("kind", "raw_text")),
        meta=meta,
    )


def unique_reading_records_by_id(*record_lists):
    out = []
    seen = set()
    for records in record_lists:
        for rec in records or ():
            if rec.rec_id in seen:
                continue
            seen.add(rec.rec_id)
            out.append(rec)
    return out


def reading_record_with_meta(rec, meta):
    return ReadingRecord(
        rec_id=rec.rec_id,
        split=rec.split,
        tokens=rec.tokens,
        kind=rec.kind,
        meta=meta,
    )


def merge_reading_replay_metadata(records, replay_records):
    """Carry replay priority onto matching fresh rows before appending old rows."""
    records = list(records or [])
    replay_records = list(replay_records or [])
    replay_meta_by_id = {}
    for rec in replay_records:
        meta = getattr(rec, "meta", {}) if isinstance(
            getattr(rec, "meta", None), dict) else {}
        priority = float(meta.get("replay_priority", 0.0) or 0.0)
        reasons = list(meta.get("replay_reasons", ()) or ())
        if priority <= 0.0 and not reasons:
            continue
        row = replay_meta_by_id.setdefault(
            rec.rec_id, {"priority": 0.0, "reasons": []})
        row["priority"] = max(float(row["priority"]), priority)
        for reason in reasons:
            if reason not in row["reasons"]:
                row["reasons"].append(reason)
    merged = []
    for rec in records:
        replay_meta = replay_meta_by_id.get(rec.rec_id)
        if replay_meta is None:
            merged.append(rec)
            continue
        meta = dict(rec.meta or {})
        meta["replay_priority"] = max(
            float(meta.get("replay_priority", 0.0) or 0.0),
            float(replay_meta["priority"]))
        existing_reasons = list(meta.get("replay_reasons", ()) or ())
        for reason in replay_meta["reasons"]:
            if reason not in existing_reasons:
                existing_reasons.append(reason)
        meta["replay_reasons"] = existing_reasons
        merged.append(reading_record_with_meta(rec, meta))
    return unique_reading_records_by_id(merged, replay_records)


def reading_hard_record_ids(study_reports):
    ids = []
    seen = set()
    for report in study_reports or ():
        if not isinstance(report, dict):
            continue
        for raw_id in report.get("hard_record_ids", ()):
            rec_id = str(raw_id)
            if rec_id and rec_id not in seen:
                seen.add(rec_id)
                ids.append(rec_id)
    return ids


def _reading_replay_learning_event_record_ids(study_reports):
    ids = []
    seen = set()
    for report in study_reports or ():
        if not isinstance(report, dict):
            continue
        for raw_id in tuple(report.get("hard_record_ids", ()) or ()) + tuple(
                report.get("record_ids", ()) or ()):
            rec_id = str(raw_id)
            if rec_id and rec_id not in seen:
                seen.add(rec_id)
                ids.append(rec_id)
        insight = report.get("study_pool_insight")
        if isinstance(insight, dict):
            for raw_id in insight.get("record_ids", ()):
                rec_id = str(raw_id)
                if rec_id and rec_id not in seen:
                    seen.add(rec_id)
                    ids.append(rec_id)
    return ids


def reading_replay_record_metadata(study_reports, learning_event=None):
    meta = {}

    def bump(rec_id, reason, priority):
        rec_id = str(rec_id)
        if not rec_id:
            return
        row = meta.setdefault(rec_id, {"priority": 0.0, "reasons": []})
        row["priority"] = max(float(row["priority"]), float(priority))
        if reason and reason not in row["reasons"]:
            row["reasons"].append(reason)

    for report in study_reports or ():
        if not isinstance(report, dict):
            continue
        strategy = str(report.get("strategy", "study"))
        score = float(report.get("mean_score", report.get("mean_gap_score", 0.0)) or 0.0)
        for raw_id in report.get("hard_record_ids", ()):
            bump(raw_id, f"hard:{strategy}", 2.0 + score)
        insight = report.get("study_pool_insight")
        if isinstance(insight, dict):
            insight_delta = max(
                0.0,
                float(insight.get("bridge_score_reduction", 0.0))
                + float(insight.get("bridge_connectivity_gain", 0.0)))
            for raw_id in insight.get("record_ids", ()):
                bump(raw_id, "concept_insight", 1.5 + insight_delta)
        for raw_id in report.get("record_ids", ()):
            bump(raw_id, f"study_pool:{strategy}", 1.0 + score)
    if isinstance(learning_event, dict) and bool(
            learning_event.get("triggered", False)):
        event_score = min(
            1.0, max(0.0, _reading_float(
                learning_event.get("event_score", 0.0))))
        event_kind = str(learning_event.get("kind", "learning"))
        event_signal = str(learning_event.get("top_signal", ""))
        reasons = [f"learning_event:{event_kind}"]
        if event_signal in READING_DISCOVERY_SIGNALS:
            reasons.append(f"learning_signal:{event_signal}")
        for raw_id in _reading_replay_learning_event_record_ids(study_reports):
            for reason in reasons:
                bump(raw_id, reason, 2.5 + event_score)
    return meta


def build_reading_replay_bank(records, study_reports=None,
                              learning_event=None,
                              max_records=READING_REPLAY_BANK_SIZE, seed=0):
    max_records = int(max_records)
    if max_records <= 0:
        return {"version": READING_REPLAY_BANK_VERSION, "max_records": max_records,
                "records": [], "record_count": 0, "hard_record_count": 0}
    records = unique_reading_records_by_id(records)
    by_id = {rec.rec_id: rec for rec in records}
    hard_ids = reading_hard_record_ids(study_reports)
    replay_meta = reading_replay_record_metadata(
        study_reports, learning_event=learning_event)
    for rec in records:
        if not isinstance(getattr(rec, "meta", None), dict):
            continue
        priority = float(rec.meta.get("replay_priority", 0.0) or 0.0)
        if priority <= 0.0:
            continue
        row = replay_meta.setdefault(rec.rec_id, {"priority": 0.0, "reasons": []})
        row["priority"] = max(float(row["priority"]), priority)
        for reason in rec.meta.get("replay_reasons", ()) or ():
            if reason not in row["reasons"]:
                row["reasons"].append(reason)
    hard_id_set = set(hard_ids)
    hard = [by_id[rec_id] for rec_id in hard_ids if rec_id in by_id]
    insight = [
        by_id[rec_id] for rec_id, meta in sorted(
            replay_meta.items(),
            key=lambda item: float(item[1].get("priority", 0.0)),
            reverse=True)
        if rec_id in by_id and rec_id not in hard_id_set]
    priority_ids = hard_id_set | {rec.rec_id for rec in insight}
    evals = [rec for rec in records if rec.split == "eval"
             and rec.rec_id not in priority_ids]
    train = [rec for rec in records if rec.split == "train"
             and rec.rec_id not in priority_ids]
    ordered = unique_reading_records_by_id(hard, insight, evals, train)
    if len(ordered) > max_records:
        keep = ordered[:max_records]
        if hard:
            keep_ids = {rec.rec_id for rec in keep}
            remaining = [rec for rec in ordered[max_records:]
                         if rec.rec_id not in keep_ids]
            if remaining and not any(rec.split == "eval" for rec in keep):
                eval_remaining = next((rec for rec in remaining
                                       if rec.split == "eval"), None)
                if eval_remaining is not None:
                    keep[-1] = eval_remaining
        ordered = keep
    return {
        "version": READING_REPLAY_BANK_VERSION,
        "max_records": max_records,
        "record_count": len(ordered),
        "hard_record_count": sum(1 for rec in ordered if rec.rec_id in hard_id_set),
        "priority_record_count": sum(
            1 for rec in ordered if rec.rec_id in replay_meta),
        "records": [
            reading_record_to_bank_row(rec, replay_meta.get(rec.rec_id))
            for rec in ordered],
    }


def reading_replay_bank_records_from_payload(payload):
    if not isinstance(payload, dict):
        return []
    bank = payload.get("reading_replay_bank")
    if bank is None:
        bank = (payload.get("report") or {}).get("reading_replay_bank")
    rows = bank.get("records", ()) if isinstance(bank, dict) else bank
    out = []
    for idx, row in enumerate(rows or ()):
        try:
            out.append(reading_record_from_bank_row(row, idx=idx))
        except ValueError:
            continue
    return unique_reading_records_by_id(out)


def reading_replay_bank_records_from_checkpoint(path, device="cpu"):
    ckpt = _torch_load(path, device)
    return reading_replay_bank_records_from_payload(ckpt)


def _reading_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _reading_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def reading_score_digest(score_components):
    if not isinstance(score_components, dict):
        return {}
    digest = {
        key: _reading_float(score_components.get(key, 0.0))
        for key in READING_MASTERY_SCORE_KEYS
        if key in score_components
    }
    for signal in READING_DISCOVERY_SIGNALS:
        skip_key = f"{signal}_skipped"
        if skip_key in score_components:
            digest[skip_key] = bool(score_components.get(skip_key, False))
    return digest


def reading_representation_progress_digest(progress):
    if not isinstance(progress, dict) or not bool(progress.get("enabled", False)):
        return {"enabled": False}
    signal_deltas = progress.get("signal_deltas")
    signal_deltas = signal_deltas if isinstance(signal_deltas, dict) else {}
    signal_after = progress.get("signal_after")
    signal_after = signal_after if isinstance(signal_after, dict) else {}
    return {
        "enabled": True,
        "active_signals": [
            str(item) for item in progress.get("active_signals", ())],
        "signal_after": {
            str(signal): _reading_float(value)
            for signal, value in signal_after.items()
            if signal in READING_DISCOVERY_SIGNALS
        },
        "signal_deltas": {
            str(signal): _reading_float(value)
            for signal, value in signal_deltas.items()
            if signal in READING_DISCOVERY_SIGNALS
        },
        "organization_score_before": _reading_float(
            progress.get("organization_score_before", 0.0)),
        "organization_score_after": _reading_float(
            progress.get("organization_score_after", 0.0)),
        "organization_score_delta": _reading_float(
            progress.get("organization_score_delta", 0.0)),
        "positive_signal_gain": _reading_float(
            progress.get("positive_signal_gain", 0.0)),
        "negative_signal_drift": _reading_float(
            progress.get("negative_signal_drift", 0.0)),
        "top_gain_signal": str(progress.get("top_gain_signal", "")),
        "top_regression_signal": str(progress.get("top_regression_signal", "")),
        "representation_insight_event": bool(
            progress.get("representation_insight_event", False)),
    }


def reading_mastery_history_from_payload(payload):
    if not isinstance(payload, dict):
        history = None
    else:
        history = payload.get("reading_mastery_history")
        if history is None:
            history = (payload.get("report") or {}).get("reading_mastery_history")
    if isinstance(history, dict):
        rows = history.get("entries", ())
    elif isinstance(history, list):
        rows = history
    else:
        rows = ()
    entries = [dict(row) for row in rows if isinstance(row, dict)]
    entries = entries[-READING_MASTERY_HISTORY_SIZE:]
    return {
        "version": READING_MASTERY_HISTORY_VERSION,
        "max_entries": READING_MASTERY_HISTORY_SIZE,
        "entry_count": len(entries),
        "entries": entries,
    }


def _reading_weight_update_history_summary(train_metrics, selection, rounds):
    train_metrics = train_metrics if isinstance(train_metrics, dict) else {}
    selection = selection if isinstance(selection, dict) else {}
    rounds = [row for row in rounds if isinstance(row, dict)]

    def row_summary(row):
        update = row.get("weight_update")
        update = update if isinstance(update, dict) else {}
        changed = bool(row.get("weight_update_changed",
                               update.get("changed", False)))
        tensor_count = _reading_int_or_none(
            row.get("weight_update_changed_tensor_count"))
        if tensor_count is None:
            tensor_count = _reading_int_or_none(
                update.get("changed_tensor_count"))
        value_count = _reading_int_or_none(
            row.get("weight_update_changed_value_count"))
        if value_count is None:
            value_count = _reading_int_or_none(
                update.get("changed_value_count"))
        max_delta = _reading_float(
            row.get("weight_update_max_abs_delta",
                    update.get("max_abs_delta", 0.0)))
        has_evidence = any(key in row for key in (
            "weight_update", "weight_update_changed",
            "weight_update_changed_tensor_count",
            "weight_update_changed_value_count",
            "weight_update_max_abs_delta"))
        return {
            "has_evidence": bool(has_evidence),
            "changed": changed,
            "tensor_count": int(tensor_count or 0),
            "value_count": int(value_count or 0),
            "max_delta": float(max_delta),
        }

    summaries = []
    train_summary = row_summary(train_metrics)
    if train_summary["has_evidence"]:
        summaries.append(train_summary)
    for row in rounds:
        summary = row_summary(row)
        if summary["has_evidence"]:
            summaries.append(summary)

    attempted = _reading_int_or_none(
        selection.get("attempted_weight_update_count"))
    if attempted is None:
        attempted = sum(
            1 for row in rounds
            if int(row.get("round", 1) or 0) > 0
            and row_summary(row)["has_evidence"])
        if attempted == 0 and train_summary["has_evidence"]:
            attempted = 1

    return {
        "changed": any(summary["changed"] for summary in summaries),
        "changed_tensor_count": max(
            (summary["tensor_count"] for summary in summaries), default=0),
        "changed_value_count": max(
            (summary["value_count"] for summary in summaries), default=0),
        "max_abs_delta": max(
            (summary["max_delta"] for summary in summaries), default=0.0),
        "attempted_count": int(attempted or 0),
    }


def reading_learning_event_report(report):
    """Summarize whether a raw-reading update produced reusable learning evidence."""
    report = report if isinstance(report, dict) else {}
    selection = report.get("selection") if isinstance(
        report.get("selection"), dict) else {}
    train_metrics = report.get("train_metrics") if isinstance(
        report.get("train_metrics"), dict) else {}
    if not selection and isinstance(train_metrics.get("selection"), dict):
        selection = train_metrics["selection"]
    rounds = selection.get("rounds", ()) if isinstance(selection, dict) else ()
    rounds = [row for row in rounds if isinstance(row, dict)]
    weight_update = _reading_weight_update_history_summary(
        train_metrics, selection, rounds)
    representation_progress = reading_representation_progress_digest(
        report.get("representation_progress"))
    delta = report.get("delta") if isinstance(report.get("delta"), dict) else {}
    score_gain = max(
        0.0,
        _reading_float(delta.get("score", 0.0)),
        _reading_float(delta.get("mastery_score", 0.0)),
        _reading_float(delta.get("active_mean_score", 0.0)),
        _reading_float(selection.get("selected_score_delta", 0.0)),
    )
    selected_round = _reading_int_or_none(selection.get("selected_round"))
    if bool(selection.get("enabled", False)) and selected_round is not None:
        event_rounds = [
            row for row in rounds
            if _reading_int_or_none(row.get("round")) == selected_round]
    else:
        event_rounds = rounds
    concept_delta = max((
        max(0.0, _reading_float(row.get("concept_insight_delta", 0.0)))
        for row in event_rounds
    ), default=0.0)
    selected_by_insight = bool(selection.get("selected_by_insight", False))
    concept_connection = bool(
        selected_by_insight or concept_delta > 0.0)
    representation_delta = _reading_float(
        representation_progress.get("organization_score_delta", 0.0))
    representation_gain = max(
        0.0,
        representation_delta,
        _reading_float(representation_progress.get("positive_signal_gain", 0.0)),
    )
    representation_event = bool(
        representation_progress.get("representation_insight_event", False)
        or representation_gain > 0.0)
    selection_enabled = bool(selection.get("enabled", False))
    accepted_update = bool(selection.get("accepted_update", False))
    update_applied = bool(accepted_update or not selection_enabled)
    weight_moved = bool(weight_update["changed"])
    triggered = bool(
        update_applied and weight_moved
        and (score_gain > 0.0 or concept_connection or representation_event))
    if concept_connection:
        kind = "concept_connection"
        top_signal = "connection"
    elif representation_event:
        kind = "representation_reorganization"
        top_signal = str(representation_progress.get("top_gain_signal", ""))
    elif score_gain > 0.0:
        kind = "score_gain"
        top_signal = str(selection.get("self_teach_top_signal", ""))
    else:
        kind = "parameter_update"
        top_signal = ""
    if top_signal not in READING_DISCOVERY_SIGNALS:
        top_signal = ""
    event_score = min(
        1.0,
        score_gain + max(0.0, concept_delta) + max(0.0, representation_gain))
    return {
        "enabled": True,
        "triggered": bool(triggered),
        "kind": kind,
        "top_signal": top_signal,
        "event_score": float(event_score),
        "update_applied": bool(update_applied),
        "selection_enabled": bool(selection_enabled),
        "accepted_update": bool(accepted_update),
        "weight_update_changed": bool(weight_moved),
        "weight_update_changed_tensor_count": int(
            weight_update["changed_tensor_count"]),
        "weight_update_changed_value_count": int(
            weight_update["changed_value_count"]),
        "weight_update_max_abs_delta": float(weight_update["max_abs_delta"]),
        "attempted_weight_update_count": int(weight_update["attempted_count"]),
        "score_gain": float(score_gain),
        "concept_connection": bool(concept_connection),
        "concept_connection_delta": float(concept_delta),
        "representation_event": bool(representation_event),
        "representation_organization_score_delta": float(representation_delta),
        "representation_positive_signal_gain": float(
            representation_progress.get("positive_signal_gain", 0.0)),
    }


def reading_mastery_history_entry(report, session_index=None):
    report = report if isinstance(report, dict) else {}
    selection = report.get("selection") if isinstance(
        report.get("selection"), dict) else {}
    train_metrics = report.get("train_metrics") if isinstance(
        report.get("train_metrics"), dict) else {}
    if not selection and isinstance(train_metrics.get("selection"), dict):
        selection = train_metrics["selection"]
    rounds = selection.get("rounds", ()) if isinstance(selection, dict) else ()
    rounds = [row for row in rounds if isinstance(row, dict)]
    concept_deltas = [
        _reading_float(row.get("concept_insight_delta", 0.0))
        for row in rounds
    ]
    replay_priority_counts = [
        _reading_int_or_none(row.get("replay_priority_record_count"))
        for row in rounds
    ]
    replay_priority_counts = [
        value for value in replay_priority_counts if value is not None
    ]
    replay_source_counts = [
        _reading_int_or_none(row.get("replay_source_count"))
        for row in rounds
    ]
    replay_source_counts = [
        value for value in replay_source_counts if value is not None
    ]
    replay_max_source_fracs = [
        _reading_float(row.get("replay_max_source_record_frac", 0.0))
        for row in rounds
        if "replay_max_source_record_frac" in row
    ]
    replay_min_source_fracs = [
        _reading_float(row.get("replay_min_source_record_frac", 0.0))
        for row in rounds
        if "replay_min_source_record_frac" in row
    ]
    replay_min_source_frac_candidates = [
        _reading_float(train_metrics.get("replay_min_source_record_frac", 0.0))
    ] + replay_min_source_fracs
    replay_min_source_frac_candidates = [
        value for value in replay_min_source_frac_candidates if value > 0.0
    ] or [0.0]
    training_priority_counts = [
        _reading_int_or_none(row.get("training_priority_record_count"))
        for row in rounds
    ]
    training_priority_counts = [
        value for value in training_priority_counts if value is not None
    ]
    training_priority_means = [
        _reading_float(row.get("training_priority_mean", 0.0))
        for row in rounds
        if "training_priority_mean" in row
    ]
    training_priority_maxes = [
        _reading_float(row.get("training_priority_max", 0.0))
        for row in rounds
        if "training_priority_max" in row
    ]
    training_source_counts = [
        _reading_int_or_none(row.get("training_source_count"))
        for row in rounds
    ]
    training_source_counts = [
        value for value in training_source_counts if value is not None
    ]
    training_max_source_fracs = [
        _reading_float(row.get("training_max_source_record_frac", 0.0))
        for row in rounds
        if "training_max_source_record_frac" in row
    ]
    training_min_source_fracs = [
        _reading_float(row.get("training_min_source_record_frac", 0.0))
        for row in rounds
        if "training_min_source_record_frac" in row
    ]
    training_min_source_frac_candidates = [
        _reading_float(train_metrics.get("training_min_source_record_frac", 0.0))
    ] + training_min_source_fracs
    training_min_source_frac_candidates = [
        value for value in training_min_source_frac_candidates if value > 0.0
    ] or [0.0]
    self_teach_reports = selection.get("self_teach_reports", ())
    if not self_teach_reports and isinstance(train_metrics.get("self_teach_plan"), dict):
        self_teach_reports = (train_metrics["self_teach_plan"],)
    self_teach_reports = [
        row for row in self_teach_reports if isinstance(row, dict)
    ]
    weight_update = _reading_weight_update_history_summary(
        train_metrics, selection, rounds)
    representation_progress = reading_representation_progress_digest(
        report.get("representation_progress"))
    learning_event = (
        report.get("learning_event")
        if isinstance(report.get("learning_event"), dict)
        else reading_learning_event_report(report))
    top_self_teach = self_teach_reports[0] if self_teach_reports else {}
    bank = report.get("reading_replay_bank")
    bank = bank if isinstance(bank, dict) else {}
    delta = report.get("delta") if isinstance(report.get("delta"), dict) else {}
    data = report.get("data", ())
    if isinstance(data, (str, bytes)):
        data = [data]
    elif not isinstance(data, (list, tuple)):
        data = []
    entry = {
        "version": READING_MASTERY_HISTORY_VERSION,
        "session_index": _reading_int_or_none(session_index),
        "experiment": str(report.get("experiment", "")),
        "checkpoint_experiment": str(report.get("checkpoint_experiment", "")),
        "data": [str(item) for item in data[:16]],
        "steps": int(report.get("steps", 0) or 0),
        "batch": int(report.get("batch", 0) or 0),
        "reading_objective_profile": str(
            report.get("reading_objective_profile", "")),
        "study_strategy_requested": str(
            report.get("study_strategy_requested", "")),
        "study_strategy": str(report.get("study_strategy", "")),
        "latent_concept_slots": int(report.get("latent_concept_slots", 0) or 0),
        "latent_concept_topk": int(report.get("latent_concept_topk", 0) or 0),
        "memory_size": int(report.get("memory_size", 0) or 0),
        "train_records": int(report.get("train_records", 0) or 0),
        "eval_records": int(report.get("eval_records", 0) or 0),
        "replay_bank_used": bool(report.get("replay_bank_used", False)),
        "replay_study_used": bool(report.get("replay_study_used", False)),
        "replay_study_records": int(report.get("replay_study_records", 0) or 0),
        "study_record_count": int(report.get("study_record_count", 0) or 0),
        "study_train_records": int(report.get("study_train_records", 0) or 0),
        "replay_bank_record_count": int(bank.get("record_count", 0) or 0),
        "replay_priority_record_count": int(
            bank.get("priority_record_count", 0) or 0),
        "selection_enabled": bool(selection.get("enabled", False)),
        "accepted_update": bool(selection.get("accepted_update", False)),
        "selected_round": _reading_int_or_none(selection.get("selected_round")),
        "selected_by_score": bool(selection.get("selected_by_score", False)),
        "selected_by_insight": bool(selection.get("selected_by_insight", False)),
        "selected_score_delta": _reading_float(
            selection.get("selected_score_delta", delta.get("score", 0.0))),
        "max_concept_insight_delta": max(concept_deltas) if concept_deltas else 0.0,
        "replay_priority_sampling": any(
            bool(row.get("replay_priority_sampling", False)) for row in rounds),
        "replay_priority_round_record_count": (
            max(replay_priority_counts) if replay_priority_counts else 0),
        "replay_source_balance_sampling": bool(
            train_metrics.get("replay_source_balance_sampling", False)
            or any(bool(row.get("replay_source_balance_sampling", False))
                   for row in rounds)),
        "replay_source_balance_w": max(
            [_reading_float(train_metrics.get("replay_source_balance_w", 0.0))]
            + [_reading_float(row.get("replay_source_balance_w", 0.0))
               for row in rounds if "replay_source_balance_w" in row]),
        "replay_source_count": max(
            [_reading_int_or_none(train_metrics.get("replay_source_count")) or 0]
            + replay_source_counts),
        "replay_max_source_record_frac": max(
            [_reading_float(
                train_metrics.get("replay_max_source_record_frac", 0.0))]
            + replay_max_source_fracs),
        "replay_min_source_record_frac": min(
            replay_min_source_frac_candidates),
        "training_priority_sampling": bool(
            train_metrics.get("training_priority_sampling", False)
            or any(bool(row.get("training_priority_sampling", False))
                   for row in rounds)),
        "training_priority_record_count": max(
            [_reading_int_or_none(
                train_metrics.get("training_priority_record_count")) or 0]
            + training_priority_counts),
        "training_priority_mean": max(
            [_reading_float(train_metrics.get("training_priority_mean", 0.0))]
            + training_priority_means),
        "training_priority_max": max(
            [_reading_float(train_metrics.get("training_priority_max", 0.0))]
            + training_priority_maxes),
        "training_source_balance_sampling": bool(
            train_metrics.get("training_source_balance_sampling", False)
            or any(bool(row.get("training_source_balance_sampling", False))
                   for row in rounds)),
        "training_source_balance_w": max(
            [_reading_float(train_metrics.get("training_source_balance_w", 0.0))]
            + [_reading_float(row.get("training_source_balance_w", 0.0))
               for row in rounds if "training_source_balance_w" in row]),
        "training_source_count": max(
            [_reading_int_or_none(train_metrics.get("training_source_count")) or 0]
            + training_source_counts),
        "training_max_source_record_frac": max(
            [_reading_float(
                train_metrics.get("training_max_source_record_frac", 0.0))]
            + training_max_source_fracs),
        "training_min_source_record_frac": min(
            training_min_source_frac_candidates),
        "weight_update_changed": bool(weight_update["changed"]),
        "weight_update_changed_tensor_count": int(
            weight_update["changed_tensor_count"]),
        "weight_update_changed_value_count": int(
            weight_update["changed_value_count"]),
        "weight_update_max_abs_delta": float(weight_update["max_abs_delta"]),
        "attempted_weight_update_count": int(weight_update["attempted_count"]),
        "representation_progress": representation_progress,
        "representation_insight_event": bool(
            representation_progress.get("representation_insight_event", False)),
        "representation_organization_score_delta": float(
            representation_progress.get("organization_score_delta", 0.0)),
        "representation_top_gain_signal": str(
            representation_progress.get("top_gain_signal", "")),
        "learning_event": learning_event,
        "learning_event_triggered": bool(
            learning_event.get("triggered", False)),
        "learning_event_kind": str(learning_event.get("kind", "")),
        "learning_event_top_signal": str(
            learning_event.get("top_signal", "")),
        "learning_event_score": _reading_float(
            learning_event.get("event_score", 0.0)),
        "self_teach_top_signal": str(top_self_teach.get("top_signal", "")),
        "self_teach_active_signals": [
            str(item) for item in top_self_teach.get("active_signals", ())],
        "self_teach_history_prior_enabled": bool(
            top_self_teach.get("history_prior_enabled", False)),
        "self_teach_history_prior_entry_count": int(
            top_self_teach.get("history_prior_entry_count", 0) or 0),
        "self_teach_history_prior_top_signal": str(
            top_self_teach.get("history_prior_top_signal", "")),
        "before_score_components": reading_score_digest(
            report.get("before_score_components")),
        "after_score_components": reading_score_digest(
            report.get("after_score_components")),
        "delta": {
            key: _reading_float(delta.get(key, 0.0))
            for key in (
                "score", "mastery_score", "active_mean_score",
                "signal_coverage", "balanced_score", "replay_score")
            if key in delta
        },
    }
    return entry


def reading_mastery_history_with_entry(previous_history, report):
    previous = reading_mastery_history_from_payload(
        {"reading_mastery_history": previous_history})
    entries = list(previous["entries"])
    entries.append(reading_mastery_history_entry(
        report, session_index=len(entries) + 1))
    entries = entries[-READING_MASTERY_HISTORY_SIZE:]
    return {
        "version": READING_MASTERY_HISTORY_VERSION,
        "max_entries": READING_MASTERY_HISTORY_SIZE,
        "entry_count": len(entries),
        "entries": entries,
    }


def normalize_reading_record(raw, default_split=None, idx=0, text_field="text",
                             default_source=None):
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
    meta = dict(raw.get("meta") or {})
    if not isinstance(meta, dict):
        raise ValueError(f"reading record {idx}.meta must be an object when provided")
    if default_source and not any(
            key in meta for key in ("source", "document", "dataset")):
        meta["source"] = str(default_source)
    return ReadingRecord(
        rec_id=str(raw.get("id", f"reading-{split}-{idx}")),
        split=split,
        tokens=tokens,
        kind=str(raw.get("kind", "raw_text")),
        meta=meta,
    )


def _chunk_reading_tokens(tokens, max_tokens=128, min_tokens=8, stride=None,
                          with_offsets=False):
    max_tokens = max(1, int(max_tokens))
    min_tokens = max(1, int(min_tokens))
    stride = max_tokens if stride is None else max(1, int(stride))
    chunks = []
    for off in range(0, len(tokens), stride):
        chunk = tuple(tokens[off:off + max_tokens])
        if len(chunk) >= min_tokens:
            chunks.append((off, chunk) if with_offsets else chunk)
        if off + max_tokens >= len(tokens):
            break
    if not chunks and tokens:
        chunk = tuple(tokens[:max_tokens])
        chunks.append((0, chunk) if with_offsets else chunk)
    return chunks


def _reading_record_with_split(rec, split, *, split_policy=None):
    meta = dict(rec.meta or {})
    if split_policy:
        meta["eval_split_policy"] = str(split_policy)
    return ReadingRecord(
        rec_id=rec.rec_id,
        split=str(split),
        tokens=rec.tokens,
        kind=rec.kind,
        meta=meta,
    )


def _reading_record_token_span(rec, fallback_index=0):
    meta = rec.meta if isinstance(rec.meta, dict) else {}
    try:
        start = int(meta.get("token_start"))
    except (TypeError, ValueError):
        start = int(fallback_index) * max(1, len(rec.tokens))
    try:
        end = int(meta.get("token_end"))
    except (TypeError, ValueError):
        end = start + len(rec.tokens)
    return start, max(start, end)


def _contiguous_eval_record_ids(records, eval_frac=0.10, seed=0):
    records = list(records or ())
    if float(eval_frac) <= 0.0 or len(records) <= 1:
        return set()
    rows = sorted(
        enumerate(records),
        key=lambda row: (
            _reading_record_token_span(row[1], row[0])[0],
            _reading_record_token_span(row[1], row[0])[1],
            row[0],
        ))
    eval_n = max(1, int(round(len(rows) * float(eval_frac))))
    eval_n = min(eval_n, len(rows) - 1)
    rng = np.random.default_rng(seed)
    block_start = int(rng.integers(0, len(rows) - eval_n + 1))
    block = rows[block_start:block_start + eval_n]
    eval_start = min(_reading_record_token_span(rec, pos)[0] for pos, rec in block)
    eval_end = max(_reading_record_token_span(rec, pos)[1] for pos, rec in block)
    eval_ids = set()
    for pos, rec in rows:
        start, end = _reading_record_token_span(rec, pos)
        if start < eval_end and end > eval_start:
            eval_ids.add(pos)
    if len(eval_ids) >= len(records):
        eval_ids = {pos for pos, _rec in block}
    return eval_ids


def assign_reading_eval_splits(records, eval_frac=0.10, seed=0):
    """Assign leakage-resistant train/eval splits without task-specific labels."""
    records = list(records or ())
    if not records:
        return [], {
            "eval_split_policy": "empty",
            "train_records": 0,
            "eval_records": 0,
            "train_sources": 0,
            "eval_sources": 0,
        }
    if any(rec.split == "eval" for rec in records):
        return records, {
            "eval_split_policy": "preserve",
            "train_records": sum(1 for rec in records if rec.split == "train"),
            "eval_records": sum(1 for rec in records if rec.split == "eval"),
            "train_sources": len({
                reading_record_source(rec) for rec in records
                if rec.split == "train"
            }),
            "eval_sources": len({
                reading_record_source(rec) for rec in records
                if rec.split == "eval"
            }),
        }
    if float(eval_frac) <= 0.0 or len(records) <= 1:
        assigned = [
            _reading_record_with_split(rec, "train", split_policy="none")
            for rec in records
        ]
        return assigned, {
            "eval_split_policy": "none",
            "train_records": len(assigned),
            "eval_records": 0,
            "train_sources": len({reading_record_source(rec) for rec in assigned}),
            "eval_sources": 0,
        }

    groups = {}
    for idx, rec in enumerate(records):
        groups.setdefault(reading_record_source(rec), []).append((idx, rec))
    if len(groups) > 1:
        sources = sorted(groups)
        eval_n = max(1, int(round(len(sources) * float(eval_frac))))
        eval_n = min(eval_n, len(sources) - 1)
        rng = np.random.default_rng(seed)
        eval_sources = set(
            sources[int(i)] for i in rng.choice(
                len(sources), size=eval_n, replace=False))
        assigned = [
            _reading_record_with_split(
                rec,
                "eval" if reading_record_source(rec) in eval_sources else "train",
                split_policy="source",
            )
            for rec in records
        ]
        return assigned, {
            "eval_split_policy": "source",
            "train_records": sum(1 for rec in assigned if rec.split == "train"),
            "eval_records": sum(1 for rec in assigned if rec.split == "eval"),
            "train_sources": len({
                reading_record_source(rec) for rec in assigned
                if rec.split == "train"
            }),
            "eval_sources": len(eval_sources),
        }

    eval_ids = _contiguous_eval_record_ids(records, eval_frac=eval_frac, seed=seed)
    assigned = [
        _reading_record_with_split(
            rec, "eval" if idx in eval_ids else "train",
            split_policy="contiguous_window")
        for idx, rec in enumerate(records)
    ]
    return assigned, {
        "eval_split_policy": "contiguous_window",
        "train_records": sum(1 for rec in assigned if rec.split == "train"),
        "eval_records": sum(1 for rec in assigned if rec.split == "eval"),
        "train_sources": len({
            reading_record_source(rec) for rec in assigned
            if rec.split == "train"
        }),
        "eval_sources": len({
            reading_record_source(rec) for rec in assigned
            if rec.split == "eval"
        }),
    }


def reading_records_from_text(text, source, max_tokens=128, min_tokens=8,
                              eval_frac=0.10, seed=0, stride=None):
    tokens = split_words(text)
    chunks = _chunk_reading_tokens(
        tokens, max_tokens=max_tokens, min_tokens=min_tokens,
        stride=stride, with_offsets=True)
    if not chunks:
        raise ValueError(f"{source} produced no reading chunks")
    records = [
        ReadingRecord(
            rec_id=f"{os.path.basename(str(source)) or 'reading'}-{i}",
            split="train",
            tokens=chunk,
            kind="raw_text",
            meta={
                "source": str(source),
                "chunk_index": i,
                "token_start": int(off),
                "token_end": int(off + len(chunk)),
                "window_tokens": int(max_tokens),
                "window_stride": int(max_tokens if stride is None else stride),
            },
        )
        for i, (off, chunk) in enumerate(chunks)
    ]
    assigned, _report = assign_reading_eval_splits(
        records, eval_frac=eval_frac, seed=seed)
    return assigned


def causal_lm_manifest_rows(records):
    """Convert reading records into targetless causal LM manifest rows."""
    rows = []
    for rec in records:
        meta = dict(rec.meta or {})
        meta["causal_lm"] = True
        meta["kind"] = str(rec.kind)
        rows.append({
            "id": str(rec.rec_id),
            "split": str(rec.split),
            "text": list(rec.tokens),
            "meta": meta,
        })
    return rows


def write_causal_lm_manifest(records, out):
    rows = causal_lm_manifest_rows(records)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return {
        "path": str(out),
        "records": len(rows),
        "train_records": sum(1 for row in rows if row["split"] == "train"),
        "eval_records": sum(1 for row in rows if row["split"] == "eval"),
        "target_records": 0,
        "train_sources": len({
            reading_record_source(rec) for rec in records
            if rec.split == "train"
        }),
        "eval_sources": len({
            reading_record_source(rec) for rec in records
            if rec.split == "eval"
        }),
    }


def _json_reading_records(data, source, text_field="text"):
    records = []
    if isinstance(data, list):
        records = [normalize_reading_record(
                       r, idx=i, text_field=text_field, default_source=source)
                   for i, r in enumerate(data)]
    elif isinstance(data, dict):
        if "records" in data:
            records = [normalize_reading_record(
                           r, idx=i, text_field=text_field, default_source=source)
                       for i, r in enumerate(data["records"])]
        else:
            for split in ("train", "eval"):
                records.extend(normalize_reading_record(
                    r, default_split=split, idx=i, text_field=text_field,
                    default_source=source)
                    for i, r in enumerate(data.get(split, [])))
    if not records:
        raise ValueError(f"{source} produced no reading records")
    return records


def load_reading_records(path, require_train=True, require_eval=True,
                         text_field="text", max_tokens=128, min_tokens=8,
                         eval_frac=0.10, seed=0, stride=None):
    if isinstance(path, (list, tuple)):
        records = []
        for i, p in enumerate(path):
            records.extend(load_reading_records(
                p, require_train=False, require_eval=False,
                text_field=text_field, max_tokens=max_tokens,
                min_tokens=min_tokens, eval_frac=0.0, seed=seed + i,
                stride=stride))
        records, _split_report = assign_reading_eval_splits(
            records, eval_frac=eval_frac, seed=seed)
        if require_train and not any(r.split == "train" for r in records):
            raise ValueError(f"{path} has no train reading records")
        if require_eval and not any(r.split == "eval" for r in records):
            raise ValueError(f"{path} has no eval reading records")
        return records
    txt = _read_text_source(path)
    if str(path).endswith(".jsonl"):
        records = [normalize_reading_record(json.loads(line), idx=i,
                                            text_field=text_field,
                                            default_source=path)
                   for i, line in enumerate(txt.splitlines()) if line.strip()]
    elif str(path).endswith(".json"):
        records = _json_reading_records(json.loads(txt), path, text_field=text_field)
    else:
        records = reading_records_from_text(
            txt, path, max_tokens=max_tokens, min_tokens=min_tokens,
            eval_frac=eval_frac, seed=seed, stride=stride)
    if require_train and not any(r.split == "train" for r in records):
        raise ValueError(f"{path} has no train reading records")
    if require_eval and not any(r.split == "eval" for r in records):
        raise ValueError(f"{path} has no eval reading records")
    return records


def _read_text_source(source, timeout=300):
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "ponens-text"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8")
    with open(source, encoding="utf-8") as f:
        return f.read()


def build_reading_vocab(records, base_vocab=None, max_size=None):
    max_size = int(max_size or 0)
    if base_vocab is not None:
        itos = list(base_vocab.itos)
        seen = set(itos)
        if max_size > 0 and len(itos) >= max_size:
            return vocab_from_itos(itos)
        if max_size > 0:
            from collections import Counter
            budget = max(0, max_size - len(itos))
            counts = Counter(
                tok for rec in records for tok in rec.tokens if tok not in seen)
            new = sorted(tok for tok, _ in counts.most_common(budget))
            return vocab_from_itos(itos + new)
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
    return Vocab(toks, max_size=max_size)


def vocab_from_itos(itos):
    vocab = object.__new__(Vocab)
    vocab.itos = list(itos)
    vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
    vocab.pad = vocab.stoi.get("<pad>", 0)
    vocab.unk = vocab.stoi.get("<unk>", 1)
    return vocab


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


class TextReadingLM(nn.Module):
    """Raw text reader with latent concept memory and decoder prefix support."""

    def __init__(self, vocab_size, d=96, layers=3, heads=4, pad=0, max_len=512,
                 text_encoder_arch="transformer", text_encoder_layers=1,
                 latent_concept_slots=0, latent_concept_layers=1,
                 latent_concept_topk=0,
                 latent_concept_prefix=False,
                 latent_concept_refine=False,
                 latent_concept_refine_gate_init=-2.0,
                 latent_concept_memory_size=0):
        super().__init__()
        self.d = int(d)
        self.heads = int(heads)
        self.latent_concept_slots = int(latent_concept_slots)
        self.latent_concept_layers = int(latent_concept_layers)
        self.latent_concept_topk = int(latent_concept_topk)
        self.latent_concept_prefix = bool(latent_concept_prefix)
        self.latent_concept_refine = bool(latent_concept_refine)
        self.latent_concept_refine_gate_init = float(latent_concept_refine_gate_init)
        self.latent_concept_memory_size = int(latent_concept_memory_size)
        if ((self.latent_concept_prefix or self.latent_concept_refine)
                and self.latent_concept_slots <= 0):
            raise ValueError("latent concept prefix/refine require latent slots")
        if self.latent_concept_memory_size < 0:
            raise ValueError("latent concept memory size must be non-negative")
        if self.latent_concept_topk < 0:
            raise ValueError("latent concept topk must be non-negative")
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
        self.reading_completion_predictor = LatentConceptSequencePredictor(d)
        self.latent_concept_refiner = None
        self.latent_concepts = (LatentConceptHead(
            self.latent_concept_slots, d, heads=heads,
            mixer_layers=self.latent_concept_layers,
            topk=self.latent_concept_topk)
            if self.latent_concept_slots > 0 else None)
        self.latent_concept_memory = (LatentConceptMemory(
            self.latent_concept_memory_size, d)
            if self.latent_concept_memory_size > 0 else None)
        if self.latent_concept_refine and self.latent_concepts is not None:
            self.latent_concept_refiner = SchemaConceptRefiner(
                d, heads=heads, gate_init=self.latent_concept_refine_gate_init)

    def enable_latent_concepts(self, slots, heads=None, layers=1, topk=None):
        slots = int(slots)
        latent_heads = int(heads or self.heads)
        latent_layers = int(layers)
        latent_topk = int(self.latent_concept_topk if topk is None else topk)
        if latent_topk < 0:
            raise ValueError("latent concept topk must be non-negative")
        if slots <= 0:
            self.latent_concepts = None
            self.latent_concept_slots = 0
            self.latent_concept_prefix = False
            self.latent_concept_refine = False
            self.latent_concept_refiner = None
            self.latent_concept_memory = None
            self.latent_concept_memory_size = 0
            self.latent_concept_topk = 0
            return self
        if (self.latent_concepts is None
                or self.latent_concept_slots != slots
                or getattr(self.latent_concepts, "heads", latent_heads) != latent_heads
                or getattr(self.latent_concepts, "mixer_layers", latent_layers)
                != latent_layers
                or getattr(self.latent_concepts, "topk", latent_topk)
                != latent_topk):
            self.latent_concepts = LatentConceptHead(
                slots, self.d, heads=latent_heads, mixer_layers=latent_layers,
                topk=latent_topk)
            self.latent_concepts.to(next(self.parameters()).device)
        self.latent_concept_slots = slots
        self.latent_concept_layers = latent_layers
        self.latent_concept_topk = latent_topk
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
        if extra:
            prefix = torch.cat(extra + [prefix], dim=1)
        return prefix


def pack_reading(records, vocab, device):
    text_ids = [vocab.enc(r.tokens) for r in records]
    max_t = max(len(x) for x in text_ids)
    txt = torch.full((len(records), max_t), vocab.pad, dtype=torch.long, device=device)
    for i, ids in enumerate(text_ids):
        txt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return txt


def batch_records(records, rng, batch, weights=None):
    if not weights:
        return [records[int(rng.integers(len(records)))] for _ in range(batch)]
    idx = rng.choice(len(records), size=int(batch), replace=True, p=np.asarray(weights))
    return [records[int(i)] for i in idx]


def reading_record_source(rec):
    meta = getattr(rec, "meta", {}) if isinstance(
        getattr(rec, "meta", None), dict) else {}
    source = meta.get("source", meta.get("document", meta.get("dataset", "")))
    return str(source or "__unknown__")


def reading_source_counts(records):
    from collections import Counter
    return Counter(reading_record_source(rec) for rec in records or ())


def reading_replay_sampling_weights(
        records, priority_power=1.0,
        source_balance_w=READING_DEFAULT_SOURCE_BALANCE_W):
    source_balance_w = float(source_balance_w)
    if source_balance_w < 0.0:
        raise ValueError("reading replay source balance weight must be non-negative")
    counts = reading_source_counts(records)
    balance_sources = source_balance_w > 0.0 and len(counts) > 1
    priority_weights = []
    saw_priority = False
    for rec in records or ():
        priority = 0.0
        if isinstance(getattr(rec, "meta", None), dict):
            priority = float(rec.meta.get("replay_priority", 0.0) or 0.0)
        saw_priority = saw_priority or priority > 0.0
        priority_weights.append(max(0.0, priority) ** float(priority_power))
    if not saw_priority and not balance_sources:
        return None
    weights = []
    for rec, priority_weight in zip(records or (), priority_weights):
        source_weight = 1.0
        if balance_sources:
            source_weight = 1.0 / (
                float(counts[reading_record_source(rec)]) ** source_balance_w)
        weights.append(
            source_weight * (priority_weight if saw_priority else 1.0))
    total = float(sum(weights))
    if total <= 0.0:
        return None
    return [float(w) / total for w in weights]


def reading_training_sampling_weights(
        records, priority_power=1.0, priority_boost=1.0,
        source_balance_w=READING_DEFAULT_SOURCE_BALANCE_W):
    source_balance_w = float(source_balance_w)
    if source_balance_w < 0.0:
        raise ValueError("reading source balance weight must be non-negative")
    counts = reading_source_counts(records)
    balance_sources = source_balance_w > 0.0 and len(counts) > 1
    weights = []
    saw_priority = False
    for rec in records or ():
        priority = 0.0
        if isinstance(getattr(rec, "meta", None), dict):
            priority = float(rec.meta.get("replay_priority", 0.0) or 0.0)
        saw_priority = saw_priority or priority > 0.0
        source_weight = 1.0
        if balance_sources:
            source_weight = 1.0 / (
                float(counts[reading_record_source(rec)]) ** source_balance_w)
        priority_weight = 1.0 + float(priority_boost) * (
            max(0.0, priority) ** float(priority_power))
        weights.append(source_weight * priority_weight)
    total = float(sum(weights))
    if (not saw_priority and not balance_sources) or total <= 0.0:
        return None
    return [float(w) / total for w in weights]


def batch_replay_records(records, rng, batch, weights=None):
    if not weights:
        return batch_records(records, rng, batch)
    idx = rng.choice(len(records), size=int(batch), replace=True, p=np.asarray(weights))
    return [records[int(i)] for i in idx]


def copy_pretrained_text_weights(src_model, src_vocab, dst_model, dst_vocab):
    """Copy compatible checkpoint weights into an expanded text model.

    New tokens keep their random initialization. Existing tokens and shape-compatible
    tensors are copied by symbolic identity rather than array position.
    """
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    skip_prefixes = ("txt.emb.", "lm.tok.", "lm.head.")
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
    return dst_model


def token_loss(logits, ids, pad=0):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           ids[:, 1:].reshape(-1), ignore_index=pad)


def reading_causal_lm_loss(model, txt, pad=0):
    """Autoregressive language loss over the raw reading stream."""
    if txt.shape[1] < 2:
        return txt.float().sum() * 0.0
    logits = model.lm(txt)
    return token_loss(logits, txt, pad=pad)


def reading_repetition_unlikelihood_loss(model, txt, pad=0, window=32):
    """Penalize recent non-gold repeats that create free-generation attractors."""
    zero = txt.float().sum() * 0.0
    window = int(window)
    if txt.shape[1] < 2 or window <= 0:
        return zero, {
            "enabled": False, "skipped": True, "skip_reason": "too_short",
            "tokens": 0, "candidates": 0, "window": int(window),
        }
    logits = model.lm(txt)
    max_steps = min(logits.shape[1] - 1, txt.shape[1] - 1)
    if max_steps <= 0:
        return logits.sum() * 0.0, {
            "enabled": False, "skipped": True, "skip_reason": "too_short",
            "tokens": 0, "candidates": 0, "window": int(window),
        }
    targets = txt[:, 1:1 + max_steps]
    valid = targets.ne(int(pad))
    if not bool(valid.any()):
        return logits.sum() * 0.0, {
            "enabled": False, "skipped": True, "skip_reason": "empty_targets",
            "tokens": 0, "candidates": 0, "window": int(window),
        }
    probs = torch.softmax(logits[:, :max_steps].float(), dim=-1)
    losses = []
    candidate_any = torch.zeros_like(valid, dtype=torch.bool)
    candidate_total = valid.new_zeros((), dtype=torch.long)
    for back in range(max(1, window)):
        candidates = txt.new_full(targets.shape, int(pad))
        width = max_steps - back
        if width <= 0:
            break
        candidates[:, back:] = txt[:, :width]
        mask = valid & candidates.ne(int(pad)) & candidates.ne(targets)
        if not bool(mask.any()):
            continue
        neg_probs = probs.gather(
            -1, candidates.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        neg_loss = -torch.log1p(
            -neg_probs.clamp(min=1e-6, max=1.0 - 1e-6))
        losses.append(neg_loss.masked_select(mask).mean())
        candidate_any |= mask
        candidate_total = candidate_total + mask.sum()
    if not losses:
        return logits.sum() * 0.0, {
            "enabled": False, "skipped": True,
            "skip_reason": "no_negative_repeats",
            "tokens": 0, "candidates": 0, "window": int(window),
        }
    return sum(losses) / len(losses), {
        "enabled": True,
        "skipped": False,
        "tokens": int(candidate_any.sum().detach().cpu()),
        "candidates": int(candidate_total.detach().cpu()),
        "window": int(window),
    }


def reading_continuation_repair_loss(
        model, txt, pad=0, repair_steps=4, prompt_frac=0.5,
        temperature=0.0, top_k=0):
    """Train recovery from the reader's own free-running continuation errors."""
    zero = txt.float().sum() * 0.0
    repair_steps = int(repair_steps)
    prompt_frac = float(prompt_frac)
    if txt.shape[0] == 0 or txt.shape[1] < 2 or repair_steps <= 0:
        return zero, {
            "enabled": False, "skipped": True, "skip_reason": "too_short",
            "tokens": 0, "generated_tokens": 0, "changed_tokens": 0,
            "steps": int(repair_steps), "prompt_frac": float(prompt_frac),
            "temperature": float(temperature), "top_k": int(top_k),
        }
    lengths = txt.ne(int(pad)).sum(dim=1).detach().cpu().tolist()
    prompt_lens = []
    repair_lens = []
    for length in lengths:
        length = int(length)
        if length < 2:
            prompt_lens.append(0)
            repair_lens.append(0)
            continue
        prompt_len = int(round(float(length) * prompt_frac))
        prompt_len = max(1, min(prompt_len, length - 1))
        repair_len = min(repair_steps, length - prompt_len)
        prompt_lens.append(prompt_len)
        repair_lens.append(repair_len)
    if not any(repair_lens):
        return zero, {
            "enabled": False, "skipped": True,
            "skip_reason": "no_repair_targets",
            "tokens": 0, "generated_tokens": 0, "changed_tokens": 0,
            "steps": int(repair_steps), "prompt_frac": float(prompt_frac),
            "temperature": float(temperature), "top_k": int(top_k),
        }

    mixed = txt.detach().clone()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        max_repair = max(repair_lens)
        for step in range(max_repair):
            active_rows = [
                i for i, repair_len in enumerate(repair_lens)
                if step < repair_len
            ]
            if not active_rows:
                continue
            context_lens = [prompt_lens[i] + step for i in active_rows]
            max_context = max(context_lens)
            context = txt.new_full((len(active_rows), max_context), int(pad))
            for out_i, row_i in enumerate(active_rows):
                context[out_i, :context_lens[out_i]] = mixed[
                    row_i, :context_lens[out_i]]
            logits = model.lm(context)
            for out_i, row_i in enumerate(active_rows):
                next_id = _reading_generation_next_id(
                    logits[out_i, context_lens[out_i] - 1], pad,
                    temperature=temperature, top_k=top_k)
                mixed[row_i, prompt_lens[row_i] + step] = next_id
    if was_training:
        model.train()

    logits = model.lm(mixed)
    if logits.shape[1] < 2:
        return logits.sum() * 0.0, {
            "enabled": False, "skipped": True,
            "skip_reason": "too_short",
            "tokens": 0, "generated_tokens": int(sum(repair_lens)),
            "changed_tokens": 0,
            "steps": int(repair_steps), "prompt_frac": float(prompt_frac),
            "temperature": float(temperature), "top_k": int(top_k),
        }
    targets = txt[:, 1:]
    raw = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        targets.reshape(-1), reduction="none").view_as(targets)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for row_i, repair_len in enumerate(repair_lens):
        if repair_len <= 0:
            continue
        start = max(0, prompt_lens[row_i] - 1)
        end = min(targets.shape[1], prompt_lens[row_i] + repair_len - 1)
        if end > start:
            mask[row_i, start:end] = True
    mask &= targets.ne(int(pad))
    generated = int(sum(repair_lens))
    changed = 0
    for row_i, repair_len in enumerate(repair_lens):
        if repair_len <= 0:
            continue
        start = prompt_lens[row_i]
        end = start + repair_len
        changed += int((mixed[row_i, start:end] != txt[row_i, start:end]).sum().cpu())
    if not bool(mask.any()):
        return logits.sum() * 0.0, {
            "enabled": False, "skipped": True,
            "skip_reason": "empty_loss_mask",
            "tokens": 0, "generated_tokens": generated,
            "changed_tokens": int(changed),
            "steps": int(repair_steps), "prompt_frac": float(prompt_frac),
            "temperature": float(temperature), "top_k": int(top_k),
        }
    return raw.masked_select(mask).mean(), {
        "enabled": True,
        "skipped": False,
        "tokens": int(mask.sum().detach().cpu()),
        "generated_tokens": generated,
        "changed_tokens": int(changed),
        "steps": int(repair_steps),
        "prompt_frac": float(prompt_frac),
        "temperature": float(temperature),
        "top_k": int(top_k),
    }


def reading_causal_lm_eval(model, vocab, records, device=DEV, n=0, seed=0):
    candidates = [r for r in records if r.split == "eval"] or list(records)
    if not candidates:
        return {"lm_loss": 0.0, "lm_token_acc": 0.0, "target_tokens": 0,
                "n_records": 0, "sampled": False, "skipped": True}
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=int(n), replace=False)
        candidates = [candidates[int(i)] for i in idx]
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            if txt.shape[1] < 2:
                continue
            logits = model.lm(txt)
            targets = txt[:, 1:]
            mask = targets.ne(vocab.pad)
            count = int(mask.sum().detach().cpu())
            if count <= 0:
                continue
            raw = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                targets.reshape(-1), reduction="none").view_as(targets)
            total_loss += float(raw.masked_select(mask).sum().detach().cpu())
            pred = logits[:, :-1].argmax(-1)
            total_correct += int((pred.eq(targets) & mask).sum().detach().cpu())
            total_tokens += count
    if total_tokens <= 0:
        return {"lm_loss": 0.0, "lm_token_acc": 0.0, "target_tokens": 0,
                "n_records": len(candidates), "sampled": sampled,
                "skipped": True}
    return {"lm_loss": total_loss / float(total_tokens),
            "lm_token_acc": total_correct / float(total_tokens),
            "target_tokens": int(total_tokens),
            "n_records": len(candidates),
            "sampled": sampled,
            "skipped": False}


def _reading_generation_next_id(logits, pad, temperature=0.0, top_k=0):
    scores = logits.detach().float().clone()
    if 0 <= int(pad) < scores.numel():
        scores[int(pad)] = -float("inf")
    temperature = float(temperature)
    top_k = int(top_k)
    if temperature <= 0.0:
        return int(torch.argmax(scores).item())
    if top_k > 0 and top_k < scores.numel():
        vals, _idx = torch.topk(scores, k=top_k)
        floor = vals[-1]
        scores = torch.where(
            scores >= floor, scores,
            scores.new_full(scores.shape, -float("inf")))
    probs = torch.softmax(scores / temperature, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
        return int(torch.argmax(logits.detach().float()).item())
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def reading_generate_tokens(model, vocab, prompt_tokens, max_new_tokens=32,
                            temperature=0.0, top_k=0, device=DEV):
    """Autoregressively continue a raw reading prompt without task labels."""
    prompt = tuple(prompt_tokens or ())
    if not prompt:
        raise ValueError("reading generation prompt must contain at least one token")
    generated_ids = list(vocab.enc(prompt))
    prompt_len = len(generated_ids)
    max_context = max(
        1, int(getattr(getattr(model, "lm", None), "config", {}).get(
            "max_len", 0) or len(generated_ids)))
    stop_tokens = {int(vocab.pad)}
    model.eval()
    for _ in range(max(0, int(max_new_tokens))):
        context_ids = generated_ids[-max_context:]
        ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        logits = model.lm(ids)
        if logits.shape[1] == 0:
            break
        next_id = _reading_generation_next_id(
            logits[0, -1], vocab.pad, temperature=temperature, top_k=top_k)
        if next_id in stop_tokens:
            break
        generated_ids.append(next_id)
    return tuple(vocab.dec(generated_ids[prompt_len:]))


def reading_generation_eval(model, vocab, records, device=DEV, n=0, seed=0,
                            prompt_tokens=16, max_new_tokens=32,
                            temperature=0.0, top_k=0):
    """Free continuation probe for held-out raw reading windows."""
    candidates = [
        rec for rec in ([r for r in records if r.split == "eval"] or list(records))
        if len(rec.tokens) >= 2
    ]
    count = min(int(n), len(candidates)) if int(n) > 0 else len(candidates)
    if count <= 0:
        return {"enabled": False, "skipped": True,
                "skip_reason": "no_generation_records", "n_records": 0}
    rng = np.random.default_rng(seed)
    if count < len(candidates):
        idx = rng.choice(len(candidates), size=count, replace=False)
        sample = [candidates[int(i)] for i in idx]
    else:
        sample = list(candidates)
    rows = []
    total_matches = total_gold = exact = 0
    model.eval()
    for rec in sample:
        sequence = tuple(rec.tokens)
        prompt_len = min(max(1, int(prompt_tokens)), len(sequence) - 1)
        prompt = sequence[:prompt_len]
        gold = sequence[prompt_len:prompt_len + max(0, int(max_new_tokens))]
        if not gold:
            continue
        generated = reading_generate_tokens(
            model, vocab, prompt, max_new_tokens=len(gold),
            temperature=temperature, top_k=top_k, device=device)
        matches = sum(1 for a, b in zip(generated, gold) if a == b)
        total_matches += int(matches)
        total_gold += int(len(gold))
        exact_match = tuple(generated) == tuple(gold)
        exact += int(exact_match)
        rows.append({
            "id": str(rec.rec_id),
            "prompt": list(prompt),
            "gold": list(gold),
            "generated": list(generated),
            "token_acc": float(matches) / float(len(gold)) if gold else 0.0,
            "exact": bool(exact_match),
        })
    generated_signatures = {tuple(row["generated"]) for row in rows}
    return {
        "enabled": bool(rows),
        "skipped": not bool(rows),
        "skip_reason": "" if rows else "no_generation_targets",
        "n_records": int(len(rows)),
        "prompt_tokens": int(prompt_tokens),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "token_acc": (
            float(total_matches) / float(total_gold) if total_gold else 0.0),
        "exact": float(exact) / float(len(rows)) if rows else 0.0,
        "unique_generation_count": int(len(generated_signatures)),
        "all_generations_identical": (
            bool(len(rows) > 1 and len(generated_signatures) == 1)),
        "samples": rows,
    }


def reading_generation_gate(generation):
    """Require raw-reading progress to show up in free continuation."""
    generation = dict(generation or {})
    if bool(generation.get("skipped", False)) or not bool(
            generation.get("enabled", False)):
        return {
            "required": True,
            "passed": False,
            "reason": str(generation.get("skip_reason", "generation_not_run")),
        }
    n_records = int(generation.get("n_records", 0) or 0)
    token_acc = float(generation.get("token_acc", 0.0) or 0.0)
    collapsed = bool(generation.get("all_generations_identical", False))
    empty_count = sum(
        1 for row in generation.get("samples", ()) or ()
        if not row.get("generated"))
    unique_count = int(generation.get("unique_generation_count", 0) or 0)
    passed = (
        n_records > 0
        and token_acc > 0.0
        and not collapsed
        and empty_count < n_records
        and unique_count > 0
    )
    reason = "passed" if passed else (
        "collapsed_generation" if collapsed else
        "zero_generation_token_acc" if token_acc <= 0.0 else
        "empty_generations" if empty_count >= n_records else
        "no_generation_records")
    return {
        "required": True,
        "passed": bool(passed),
        "reason": reason,
        "token_acc": float(token_acc),
        "n_records": int(n_records),
        "unique_generation_count": int(unique_count),
        "empty_generation_count": int(empty_count),
        "all_generations_identical": bool(collapsed),
    }


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


def reading_latent_memory_consolidation_loss(
        model, txt, pad, unk, token_drop_p=0.15, token_replace_p=0.05,
        feature_dropout=0.1, temperature=0.1, balance_w=0.0,
        anchor_w=1.0, fer_w=0.0, fer_fragmentation_w=1.0,
        fer_correlation_w=1.0, fer_balance_w=0.1):
    zero = torch.tensor(0.0, device=txt.device)
    metrics = {"memory_loss": zero, "anchor_loss": zero, "fer_loss": zero,
               "nearest_cosine": zero, "memory_active": 0, "skipped": True}
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return zero, metrics
    active = memory.active()
    active_n = int(active.shape[0])
    metrics["memory_active"] = active_n
    if active_n <= 0:
        return zero, metrics
    view = corrupt_reading_tokens(
        txt, pad, unk, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p)
    slots = model.latent_concept_states(
        view, feature_dropout=feature_dropout, project=True)
    return latent_concept_memory_consolidation_loss(
        slots, active, temperature=temperature, balance_w=balance_w,
        anchor_w=anchor_w, fer_w=fer_w,
        fer_fragmentation_w=fer_fragmentation_w,
        fer_correlation_w=fer_correlation_w, fer_balance_w=fer_balance_w)


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


def split_reading_context_closure(txt, pad, split_frac=0.5):
    frac = float(split_frac)
    if frac <= 0.0 or frac >= 1.0:
        raise ValueError("reading context closure split fraction must be in (0, 1)")
    valid = txt.ne(pad)
    prefix = torch.full_like(txt, int(pad))
    suffix = torch.full_like(txt, int(pad))
    used = torch.zeros_like(valid)
    for row in range(txt.shape[0]):
        positions = torch.where(valid[row])[0]
        n = int(positions.numel())
        if n <= 1:
            continue
        split = max(1, min(n - 1, int(round(n * frac))))
        prefix_positions = positions[:split]
        suffix_positions = positions[split:]
        prefix[row, prefix_positions] = txt[row, prefix_positions]
        suffix[row, suffix_positions] = txt[row, suffix_positions]
        used[row, positions] = True
    return prefix, suffix, used


def mask_reading_spans(txt, pad, span_frac=0.25):
    frac = float(span_frac)
    if frac <= 0.0 or frac >= 1.0:
        raise ValueError("reading span mask fraction must be in (0, 1)")
    out = txt.clone()
    valid = txt.ne(pad)
    hidden = torch.zeros_like(valid)
    for row in range(txt.shape[0]):
        positions = torch.where(valid[row])[0]
        n = int(positions.numel())
        if n <= 1:
            continue
        span = max(1, int(round(n * frac)))
        span = min(span, n - 1)
        start = int(torch.randint(
            0, n - span + 1, (1,), device=txt.device).item())
        masked_positions = positions[start:start + span]
        out[row, masked_positions] = int(pad)
        hidden[row, masked_positions] = True
    return out, hidden


def reading_context_target_loss(model, txt, pad, context_keep_p=0.5,
                                feature_dropout=0.1, temperature=0.1):
    if getattr(model, "latent_concepts", None) is None:
        return torch.tensor(0.0, device=txt.device)
    predictor = _reading_completion_predictor(model)
    if predictor is None:
        return txt.float().sum() * 0.0
    if txt.shape[0] <= 1:
        return txt.float().sum() * 0.0
    context_txt, target_txt = split_reading_context_target(
        txt, pad, context_keep_p=context_keep_p)
    context_slots = model.latent_concept_states(
        context_txt, feature_dropout=feature_dropout, project=False)
    target_slots = model.latent_concept_states(
        target_txt, feature_dropout=0.0, project=False).detach()
    loss, _metrics = latent_concept_completion_loss(
        predictor, {"context": context_slots}, target_slots,
        temperature=temperature)
    return loss


def _reading_completion_predictor(model):
    return getattr(
        model, "reading_completion_predictor",
        getattr(model, "reading_predictor", None))


def reading_span_completion_loss(model, txt, pad, span_frac=0.25,
                                 feature_dropout=0.1, temperature=0.1):
    zero = txt.float().sum() * 0.0
    metrics = {"completion_loss": zero, "hidden_token_rate": zero,
               "view_count": 0, "skipped": True}
    predictor = _reading_completion_predictor(model)
    if getattr(model, "latent_concepts", None) is None or predictor is None:
        return zero, metrics
    if txt.shape[0] <= 1:
        return zero, metrics
    masked_txt, hidden = mask_reading_spans(
        txt, pad, span_frac=span_frac)
    hidden_count = hidden.sum()
    valid_count = txt.ne(pad).sum().clamp_min(1)
    hidden_rate = hidden_count.float() / valid_count.float()
    if int(hidden_count.item()) <= 0:
        return zero, metrics | {"hidden_token_rate": hidden_rate.detach()}
    partial_slots = model.latent_concept_states(
        masked_txt, feature_dropout=feature_dropout, project=False)
    full_slots = model.latent_concept_states(
        txt, feature_dropout=0.0, project=False).detach()
    loss, completion_metrics = latent_concept_completion_loss(
        predictor, {"span": partial_slots}, full_slots,
        temperature=temperature)
    return loss, {
        "completion_loss": completion_metrics["completion_loss"],
        "hidden_token_rate": hidden_rate.detach(),
        "view_count": int(completion_metrics.get("view_count", 0)),
        "skipped": bool(completion_metrics.get("skipped", False)),
    }


def reading_context_closure_loss(model, txt, pad, split_frac=0.5,
                                 feature_dropout=0.1, temperature=0.1):
    zero = txt.float().sum() * 0.0
    metrics = {"completion_loss": zero, "prefix_token_rate": zero,
               "suffix_token_rate": zero, "view_count": 0, "skipped": True}
    predictor = _reading_completion_predictor(model)
    if getattr(model, "latent_concepts", None) is None or predictor is None:
        return zero, metrics
    if txt.shape[0] <= 1:
        return zero, metrics
    prefix_txt, suffix_txt, used = split_reading_context_closure(
        txt, pad, split_frac=split_frac)
    used_count = used.sum()
    if int(used_count.item()) <= 0:
        return zero, metrics
    valid_count = txt.ne(pad).sum().clamp_min(1)
    prefix_rate = prefix_txt.ne(pad).sum().float() / valid_count.float()
    suffix_rate = suffix_txt.ne(pad).sum().float() / valid_count.float()
    prefix_slots = model.latent_concept_states(
        prefix_txt, feature_dropout=feature_dropout, project=False)
    suffix_slots = model.latent_concept_states(
        suffix_txt, feature_dropout=feature_dropout, project=False)
    full_slots = model.latent_concept_states(
        txt, feature_dropout=0.0, project=False).detach()
    loss, completion_metrics = latent_concept_completion_loss(
        predictor, {"prefix": prefix_slots, "suffix": suffix_slots},
        full_slots, temperature=temperature)
    return loss, {
        "completion_loss": completion_metrics["completion_loss"],
        "prefix_token_rate": prefix_rate.detach(),
        "suffix_token_rate": suffix_rate.detach(),
        "view_count": int(completion_metrics.get("view_count", 0)),
        "skipped": bool(completion_metrics.get("skipped", False)),
    }


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


def reading_latent_discovery_loss(
        model, txt, pad, feature_dropout=0.1, context_keep_p=0.5,
        curiosity_w=1.0, graph_w=1.0, cycle_w=1.0, bridge_w=1.0,
        fer_w=0.0, curiosity_temperature=0.1, curiosity_self_loop_w=0.05,
        curiosity_transitive_steps=2, curiosity_transitive_w=0.1,
        graph_temperature=0.1, graph_self_loop_w=0.05,
        graph_transitive_steps=2, graph_transitive_w=0.1,
        graph_target_power=1.0, cycle_temperature=0.1,
        cycle_self_loop_w=0.05, cycle_transitive_steps=2,
        cycle_transitive_w=0.1, cycle_target_power=1.0,
        cycle_consistency_w=0.5, fer_fragmentation_w=1.0,
        fer_correlation_w=1.0, fer_balance_w=0.1):
    zero = torch.tensor(0.0, device=txt.device)
    metrics = {"curiosity_loss": zero, "curiosity_novelty": zero,
               "curiosity_association": zero, "graph_loss": zero,
               "graph_kl": zero, "graph_cosine": zero,
               "cycle_loss": zero, "cycle_forward_kl": zero,
               "cycle_reverse_kl": zero, "cycle_source_cycle_kl": zero,
               "cycle_target_cycle_kl": zero, "bridge_loss": zero,
               "insight_loss": zero, "insight_score": zero,
               "insight_kl": zero, "insight_cosine": zero,
               "insight_missing_mass": zero,
               "insight_reachable_mass": zero, "insight_gain": zero,
               "bridge_score": zero, "bridge_entropy": zero,
               "bridge_connectivity": zero, "fer_loss": zero,
               "memory_active": 0, "graph_ready": False, "skipped": True}
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return zero, metrics
    active = memory.active()
    metrics["memory_active"] = int(active.shape[0])
    if active.numel() == 0:
        return zero, metrics
    full_slots = model.latent_concept_states(
        txt, feature_dropout=feature_dropout, project=True)
    context_txt, target_txt = split_reading_context_target(
        txt, pad, context_keep_p=context_keep_p)
    source_slots = model.latent_concept_states(
        context_txt, feature_dropout=feature_dropout, project=True)
    target_slots = model.latent_concept_states(
        target_txt, feature_dropout=0.0, project=True)
    if hasattr(memory, "discovery_loss"):
        return memory.discovery_loss(
            full_slots, source_slots=source_slots, target_slots=target_slots,
            curiosity_w=curiosity_w, graph_w=graph_w, cycle_w=cycle_w,
            bridge_w=bridge_w, fer_w=fer_w,
            curiosity_temperature=curiosity_temperature,
            curiosity_self_loop_w=curiosity_self_loop_w,
            curiosity_transitive_steps=curiosity_transitive_steps,
            curiosity_transitive_w=curiosity_transitive_w,
            graph_temperature=graph_temperature,
            graph_self_loop_w=graph_self_loop_w,
            graph_transitive_steps=graph_transitive_steps,
            graph_transitive_w=graph_transitive_w,
            graph_target_power=graph_target_power,
            cycle_temperature=cycle_temperature,
            cycle_self_loop_w=cycle_self_loop_w,
            cycle_transitive_steps=cycle_transitive_steps,
            cycle_transitive_w=cycle_transitive_w,
            cycle_target_power=cycle_target_power,
            cycle_consistency_w=cycle_consistency_w,
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w,
            fer_balance_w=fer_balance_w)
    return latent_concept_discovery_loss(
        full_slots, active, relations=memory.active_relations(),
        transitions=memory.active_transitions(),
        prediction_relations=memory.active_prediction_relations(),
        source_slots=source_slots, target_slots=target_slots,
        curiosity_w=curiosity_w, graph_w=graph_w, cycle_w=cycle_w,
        bridge_w=bridge_w, fer_w=fer_w,
        curiosity_temperature=curiosity_temperature,
        curiosity_self_loop_w=curiosity_self_loop_w,
        curiosity_transitive_steps=curiosity_transitive_steps,
        curiosity_transitive_w=curiosity_transitive_w,
        graph_temperature=graph_temperature, graph_self_loop_w=graph_self_loop_w,
        graph_transitive_steps=graph_transitive_steps,
        graph_transitive_w=graph_transitive_w,
        graph_target_power=graph_target_power,
        cycle_temperature=cycle_temperature,
        cycle_self_loop_w=cycle_self_loop_w,
        cycle_transitive_steps=cycle_transitive_steps,
        cycle_transitive_w=cycle_transitive_w,
        cycle_target_power=cycle_target_power,
        cycle_consistency_w=cycle_consistency_w,
        fer_fragmentation_w=fer_fragmentation_w,
        fer_correlation_w=fer_correlation_w,
        fer_balance_w=fer_balance_w)


def reading_latent_reanalysis_loss(
        model, txt, pad, unk, token_drop_p=0.15, token_replace_p=0.05,
        feature_dropout=0.1, graph_w=1.0, cycle_w=0.5, bridge_w=0.5,
        fer_w=0.0, temperature=0.1, self_loop_w=0.05,
        transitive_steps=2, transitive_w=0.1, target_power=1.0,
        cycle_consistency_w=0.5, fer_fragmentation_w=1.0,
        fer_correlation_w=1.0, fer_balance_w=0.1):
    zero = torch.tensor(0.0, device=txt.device)
    metrics = {"closure_loss": zero, "closure_kl": zero,
               "closure_cosine": zero, "cycle_loss": zero,
               "bridge_loss": zero, "fer_loss": zero,
               "memory_active": 0, "graph_ready": False, "skipped": True}
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return zero, metrics
    active = memory.active()
    metrics["memory_active"] = int(active.shape[0])
    if active.numel() == 0:
        return zero, metrics
    with torch.no_grad():
        anchor_slots = model.latent_concept_states(
            txt, feature_dropout=0.0, project=True)
    probe_txt = corrupt_reading_tokens(
        txt, pad, unk, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p)
    probe_slots = model.latent_concept_states(
        probe_txt, feature_dropout=feature_dropout, project=True)
    if hasattr(memory, "reanalysis_loss"):
        return memory.reanalysis_loss(
            probe_slots, anchor_slots,
            graph_w=graph_w, cycle_w=cycle_w, bridge_w=bridge_w, fer_w=fer_w,
            temperature=temperature, self_loop_w=self_loop_w,
            transitive_steps=transitive_steps, transitive_w=transitive_w,
            target_power=target_power,
            cycle_consistency_w=cycle_consistency_w,
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w,
            fer_balance_w=fer_balance_w)
    return latent_concept_reanalysis_loss(
        probe_slots, anchor_slots, active, relations=memory.active_relations(),
        transitions=memory.active_transitions(),
        prediction_relations=memory.active_prediction_relations(),
        graph_w=graph_w, cycle_w=cycle_w, bridge_w=bridge_w, fer_w=fer_w,
        temperature=temperature, self_loop_w=self_loop_w,
        transitive_steps=transitive_steps, transitive_w=transitive_w,
        target_power=target_power,
        cycle_consistency_w=cycle_consistency_w,
        fer_fragmentation_w=fer_fragmentation_w,
        fer_correlation_w=fer_correlation_w,
        fer_balance_w=fer_balance_w)


def reading_latent_gap_loss(model, txt, feature_dropout=0.1, temperature=0.1,
                            self_loop_w=0.0, transitive_steps=2,
                            transitive_w=0.1, target_power=1.0,
                            relation_w=1.0, transition_w=1.0):
    zero = torch.tensor(0.0, device=txt.device)
    metrics = {"gap_loss": zero, "gap_kl": zero, "gap_cosine": zero,
               "gap_entropy": zero, "gap_target_mass": zero,
               "gap_present_overlap": zero, "memory_active": 0,
               "graph_ready": False, "skipped": True}
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return zero, metrics
    active = memory.active()
    metrics["memory_active"] = int(active.shape[0])
    if active.numel() == 0:
        return zero, metrics
    slots = model.latent_concept_states(
        txt, feature_dropout=feature_dropout, project=True)
    if hasattr(memory, "gap_loss"):
        return memory.gap_loss(
            slots, temperature=temperature, self_loop_w=self_loop_w,
            transitive_steps=transitive_steps, transitive_w=transitive_w,
            target_power=target_power, relation_w=relation_w,
            transition_w=transition_w)
    return latent_concept_memory_gap_loss(
        slots, active, relations=memory.active_relations(),
        transitions=memory.active_transitions(), temperature=temperature,
        self_loop_w=self_loop_w, transitive_steps=transitive_steps,
        transitive_w=transitive_w, target_power=target_power,
        relation_w=relation_w, transition_w=transition_w)


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
    predicted = _reading_completion_predictor(model)(context_slots)
    predicted = F.normalize(predicted.reshape(predicted.shape[0], -1), dim=-1)
    target = F.normalize(target_slots.reshape(target_slots.shape[0], -1), dim=-1)
    return predicted, target


def _reading_span_completion_embeddings(model, txt, pad, seed=0, span_frac=0.25):
    torch.manual_seed(int(seed))
    masked_txt, _hidden = mask_reading_spans(txt, pad, span_frac=span_frac)
    partial_slots = model.latent_concept_states(masked_txt, project=False)
    full_slots = model.latent_concept_states(txt, project=False)
    predicted = _reading_completion_predictor(model)(partial_slots)
    predicted = F.normalize(predicted.reshape(predicted.shape[0], -1), dim=-1)
    target = F.normalize(full_slots.reshape(full_slots.shape[0], -1), dim=-1)
    return predicted, target


def _reading_context_closure_embeddings(model, txt, pad, seed=0, split_frac=0.5):
    torch.manual_seed(int(seed))
    prefix_txt, suffix_txt, _used = split_reading_context_closure(
        txt, pad, split_frac=split_frac)
    prefix_slots = model.latent_concept_states(prefix_txt, project=False)
    suffix_slots = model.latent_concept_states(suffix_txt, project=False)
    full_slots = model.latent_concept_states(txt, project=False)
    predictor = _reading_completion_predictor(model)
    prefix_pred = predictor(prefix_slots)
    suffix_pred = predictor(suffix_slots)
    prefix_pred = F.normalize(prefix_pred.reshape(prefix_pred.shape[0], -1), dim=-1)
    suffix_pred = F.normalize(suffix_pred.reshape(suffix_pred.shape[0], -1), dim=-1)
    predicted = F.normalize(torch.stack((prefix_pred, suffix_pred), dim=0).mean(0), dim=-1)
    target = F.normalize(full_slots.reshape(full_slots.shape[0], -1), dim=-1)
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
            or _reading_completion_predictor(model) is None):
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


def reading_span_completion_retrieval_eval(model, vocab, records, device=DEV, n=0,
                                           seed=0, span_frac=0.25):
    if (getattr(model, "latent_concepts", None) is None
            or _reading_completion_predictor(model) is None):
        return {"span_completion_acc": 0.0, "n_records": 0, "sampled": False,
                "skipped": True}
    selected = eval_reading_records(records, n=n, seed=seed)
    if not selected:
        return {"span_completion_acc": 0.0, "n_records": 0, "sampled": False,
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
            predicted, target = _reading_span_completion_embeddings(
                model, txt, vocab.pad, seed=seed + off * 2,
                span_frac=span_frac)
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
        return {"span_completion_acc": 0.0, "n_records": 0,
                "sampled": bool(n > 0 and n < eval_count), "skipped": True}
    return {"span_completion_acc": correct / max(1, total),
            "positive_cosine": pos_sum / max(1, total),
            "negative_cosine": neg_sum / max(1, neg_count),
            "margin": (pos_sum / max(1, total)) - (neg_sum / max(1, neg_count)),
            "n_records": total,
            "sampled": bool(n > 0 and n < eval_count),
            "skipped": False}


def reading_context_closure_retrieval_eval(model, vocab, records, device=DEV, n=0,
                                           seed=0, split_frac=0.5):
    if (getattr(model, "latent_concepts", None) is None
            or _reading_completion_predictor(model) is None):
        return {"context_closure_acc": 0.0, "n_records": 0, "sampled": False,
                "skipped": True}
    selected = eval_reading_records(records, n=n, seed=seed)
    if not selected:
        return {"context_closure_acc": 0.0, "n_records": 0, "sampled": False,
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
            predicted, target = _reading_context_closure_embeddings(
                model, txt, vocab.pad, seed=seed + off * 2,
                split_frac=split_frac)
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
        return {"context_closure_acc": 0.0, "n_records": 0,
                "sampled": bool(n > 0 and n < eval_count), "skipped": True}
    return {"context_closure_acc": correct / max(1, total),
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
    "language", "generation", "view", "context", "span", "closure", "sequence",
    "neighborhood", "cluster", "fer", "bridge", "connection", "both", "min",
    "all", "balanced", "mastery")
READING_DISCOVERY_SIGNALS = (
    "language", "generation", "view", "context", "span", "closure", "sequence",
    "neighborhood", "cluster", "fer", "bridge", "connection")
READING_OBJECTIVE_PROFILES = ("manual", "mastery")
READING_DEFAULT_SIGNAL_REGRESSION_TOLERANCE = 0.02
READING_MASTERY_OBJECTIVE_FLOORS = {
    "lm_w": 1.0,
    "factorization_w": 0.05,
    "fer_w": 0.05,
    "memory_w": 0.05,
    "discovery_w": 0.05,
    "reanalysis_w": 0.05,
    "gap_w": 0.05,
    "association_w": 0.05,
    "composition_w": 0.05,
    "graph_predict_w": 0.10,
    "graph_cycle_w": 0.05,
    "bridge_w": 0.05,
    "context_target_w": 0.10,
    "span_completion_w": 0.05,
    "context_closure_w": 0.05,
    "sequence_w": 0.05,
    "neighborhood_w": 0.05,
    "transition_w": 0.05,
    "cluster_w": 0.05,
    "study_self_teach_w": 0.05,
    "continuation_repair_w": 0.05,
    "continuation_repair_steps": 4,
    "repetition_unlikelihood_w": 0.05,
    "repetition_unlikelihood_window": 32,
}
READING_MASTERY_STUDY_FLOORS = {
    "study_rounds": 3,
    "study_score_patience": 2,
    "study_score_target": 0.85,
    "study_representation_accept_w": 0.25,
    "study_representation_min_delta": 0.01,
}
READING_MASTERY_OPERATION_FLOORS = {
    "study_probe_n": 256,
    "study_hard_max": 64,
    "study_refresh_steps": 100,
    "neighborhood_probe_n": 256,
    "neighborhood_refresh_steps": 100,
    "cluster_probe_n": 256,
    "cluster_refresh_steps": 100,
    "generation_eval_n": 16,
}
READING_MASTERY_CHECKPOINT_FLOORS = {
    "replay_w": 0.05,
    "replay_retention_w": 0.25,
}
READING_MASTERY_PROFILE_FLOORS = (
    dict(READING_MASTERY_OBJECTIVE_FLOORS)
    | dict(READING_MASTERY_STUDY_FLOORS)
    | dict(READING_MASTERY_OPERATION_FLOORS))
READING_STUDY_STRATEGIES = (
    "random", "errors", "fer", "curiosity", "sequence", "closure", "graph",
    "cycle", "gap", "discovery", "auto")
READING_MEMORY_STUDY_STRATEGIES = ("curiosity", "graph", "gap", "discovery")
READING_POOL_STUDY_STRATEGIES = (
    "errors", "fer", "curiosity", "sequence", "closure", "graph", "cycle",
    "gap", "discovery")
READING_TRANSITION_STUDY_STRATEGIES = (
    "sequence", "graph", "cycle", "gap", "discovery")
READING_GRAPH_READY_STUDY_STRATEGIES = ("graph", "cycle", "gap")
READING_BRIDGE_INSIGHT_STUDY_STRATEGIES = (
    "curiosity", "graph", "cycle", "gap", "discovery")
READING_CONCEPT_INSIGHT_SCORE_KEYS = (
    "floor_score", "balanced_score", "signal_coverage", "fer_score",
    "bridge_score", "bridge_connectivity", "connection_score",
    "neighborhood_score", "cluster_score")
READING_SELF_TEACH_SCORE_KEYS = {
    "language": "language_score",
    "generation": "generation_score",
    "view": "view_score",
    "context": "context_score",
    "span": "span_score",
    "closure": "context_closure_score",
    "sequence": "sequence_score",
    "neighborhood": "neighborhood_score",
    "cluster": "cluster_score",
    "fer": "fer_score",
    "bridge": "bridge_score",
    "connection": "connection_score",
}
READING_SELF_TEACH_SIGNAL_OBJECTIVES = {
    "language": ("lm_w",),
    "generation": ("lm_w", "continuation_repair_w",
                   "repetition_unlikelihood_w"),
    "view": ("factorization_w",),
    "context": ("context_target_w",),
    "span": ("span_completion_w",),
    "closure": ("context_closure_w",),
    "sequence": ("sequence_w",),
    "neighborhood": ("neighborhood_w", "transition_w"),
    "cluster": ("cluster_w",),
    "fer": ("fer_w",),
    "bridge": ("bridge_w", "discovery_w", "gap_w"),
    "connection": ("bridge_w", "discovery_w", "gap_w"),
}
READING_SELF_TEACH_SIGNAL_STUDY_STRATEGIES = {
    "language": "sequence",
    "generation": "sequence",
    "view": "errors",
    "context": "closure",
    "span": "closure",
    "closure": "closure",
    "sequence": "sequence",
    "neighborhood": "discovery",
    "cluster": "discovery",
    "fer": "fer",
    "bridge": "discovery",
    "connection": "discovery",
}
READING_SELF_TEACH_WEIGHT_KEYS = tuple(dict.fromkeys(
    key
    for keys in READING_SELF_TEACH_SIGNAL_OBJECTIVES.values()
    for key in keys))


def resolve_reading_study_strategy(study_strategy, model):
    requested = str(study_strategy)
    if requested not in READING_STUDY_STRATEGIES:
        raise ValueError(f"unknown reading study strategy {requested!r}")
    if requested == "auto":
        if getattr(model, "latent_concepts", None) is None:
            return "errors"
        return ("discovery"
                if getattr(model, "latent_concept_memory", None) is not None
                else "closure")
    return requested


def reading_self_teach_study_strategy(self_teach_plan, requested_strategy,
                                      resolved_strategy, model):
    """Route auto-study rounds to the weakest evaluated learning signal."""
    if str(requested_strategy) != "auto":
        return resolved_strategy
    signal = (self_teach_plan or {}).get("top_signal")
    candidate = READING_SELF_TEACH_SIGNAL_STUDY_STRATEGIES.get(signal)
    if candidate is None:
        return resolved_strategy
    if (candidate in READING_MEMORY_STUDY_STRATEGIES
            and getattr(model, "latent_concept_memory", None) is None):
        return resolved_strategy
    return candidate


def reading_objective_profile_kwargs(objective_profile="manual", **kwargs):
    """Apply schema-free reading objective floors for a training posture."""
    objective_profile = str(objective_profile)
    if objective_profile not in READING_OBJECTIVE_PROFILES:
        raise ValueError(f"unknown reading objective profile {objective_profile!r}")
    effective = dict(kwargs)
    updates = {}
    if objective_profile == "mastery":
        missing = [
            key for key in READING_MASTERY_PROFILE_FLOORS
            if key not in effective]
        if missing:
            raise ValueError(
                "missing reading objective profile weights: "
                + ", ".join(sorted(missing)))
        for key, floor in READING_MASTERY_PROFILE_FLOORS.items():
            before_raw = effective[key]
            before = float(before_raw)
            after_raw = max(before, float(floor))
            after = (int(after_raw)
                     if isinstance(before_raw, int)
                     and isinstance(floor, int)
                     else float(after_raw))
            effective[key] = after
            if float(after) != before:
                updates[key] = {"from": before_raw, "to": after}
    report = {
        "profile": objective_profile,
        "enabled": objective_profile != "manual",
        "floors": dict(READING_MASTERY_PROFILE_FLOORS)
        if objective_profile == "mastery" else {},
        "objective_floors": dict(READING_MASTERY_OBJECTIVE_FLOORS)
        if objective_profile == "mastery" else {},
        "study_floors": dict(READING_MASTERY_STUDY_FLOORS)
        if objective_profile == "mastery" else {},
        "operation_floors": dict(READING_MASTERY_OPERATION_FLOORS)
        if objective_profile == "mastery" else {},
        "updates": updates,
        "applied": bool(updates),
    }
    effective["reading_objective_profile"] = objective_profile
    effective["reading_objective_profile_report"] = report
    return effective


def reading_checkpoint_profile_kwargs(objective_profile="manual",
                                      replay_records=None, **kwargs):
    """Apply checkpoint-study replay floors when prior reading exists."""
    objective_profile = str(objective_profile)
    if objective_profile not in READING_OBJECTIVE_PROFILES:
        raise ValueError(f"unknown reading objective profile {objective_profile!r}")
    effective = dict(kwargs)
    updates = {}
    replay_records = list(replay_records or [])
    if objective_profile == "mastery" and replay_records:
        missing = [
            key for key in READING_MASTERY_CHECKPOINT_FLOORS
            if key not in effective]
        if missing:
            raise ValueError(
                "missing reading checkpoint profile controls: "
                + ", ".join(sorted(missing)))
        for key, floor in READING_MASTERY_CHECKPOINT_FLOORS.items():
            before = float(effective[key])
            after = max(before, float(floor))
            effective[key] = after
            if after != before:
                updates[key] = {"from": before, "to": after}
    report = {
        "profile": objective_profile,
        "enabled": objective_profile != "manual",
        "replay_records": len(replay_records),
        "floors": dict(READING_MASTERY_CHECKPOINT_FLOORS)
        if objective_profile == "mastery" else {},
        "updates": updates,
        "applied": bool(updates),
    }
    effective["reading_checkpoint_profile_report"] = report
    return effective


def reading_profile_report_with_checkpoint(base_report, checkpoint_report):
    report = dict(base_report or {})
    report["checkpoint_replay"] = checkpoint_report
    if checkpoint_report and checkpoint_report.get("updates"):
        updates = dict(report.get("updates", {}))
        updates.update({
            f"checkpoint_{key}": value
            for key, value in checkpoint_report["updates"].items()})
        report["updates"] = updates
        report["applied"] = bool(updates)
    return report


def reading_mastery_history_concept_insight_prior(history, enabled=True,
                                                  max_entries=8, decay=0.75):
    if not enabled:
        return {
            "enabled": False,
            "entry_count": 0,
            "concept_connection_signal": 0.0,
        }
    history = reading_mastery_history_from_payload(
        {"reading_mastery_history": history})
    entries = history["entries"][-max(1, int(max_entries)):]
    decay = min(1.0, max(0.0, float(decay)))
    weighted_delta = 0.0
    total_weight = 0.0
    max_delta = 0.0
    latest_delta = 0.0
    selected_by_insight = 0
    accepted_updates = 0
    for offset, entry in enumerate(reversed(entries)):
        if not isinstance(entry, dict):
            continue
        delta = max(
            0.0, _reading_float(entry.get("max_concept_insight_delta", 0.0)))
        learning_event = entry.get("learning_event")
        if (isinstance(learning_event, dict)
                and bool(learning_event.get("triggered", False))
                and str(learning_event.get("kind", "")) == "concept_connection"):
            delta = max(delta, _reading_float(
                learning_event.get("event_score", 0.0)))
        if offset == 0:
            latest_delta = delta
        recency_weight = decay ** offset
        weighted_delta += recency_weight * min(1.0, delta)
        total_weight += recency_weight
        max_delta = max(max_delta, delta)
        if bool(entry.get("selected_by_insight", False)):
            selected_by_insight += 1
        if bool(entry.get("accepted_update", False)):
            accepted_updates += 1
    signal = weighted_delta / total_weight if total_weight > 0.0 else 0.0
    return {
        "enabled": True,
        "entry_count": int(len(entries)),
        "decay": float(decay),
        "concept_connection_signal": float(signal),
        "max_concept_insight_delta": float(max_delta),
        "latest_concept_insight_delta": float(latest_delta),
        "selected_by_insight_count": int(selected_by_insight),
        "accepted_update_count": int(accepted_updates),
    }


def reading_mastery_history_self_teach_prior(history, enabled=True,
                                             max_entries=8, decay=0.75):
    if not enabled:
        return {"enabled": False, "entry_count": 0, "signal_deficits": {}}
    history = reading_mastery_history_from_payload(
        {"reading_mastery_history": history})
    concept_prior = reading_mastery_history_concept_insight_prior(
        history, enabled=True, max_entries=max_entries, decay=decay)
    entries = history["entries"][-max(1, int(max_entries)):]
    decay = min(1.0, max(0.0, float(decay)))
    weighted = {signal: 0.0 for signal in READING_DISCOVERY_SIGNALS}
    weights = {signal: 0.0 for signal in READING_DISCOVERY_SIGNALS}
    top_counts = {signal: 0 for signal in READING_DISCOVERY_SIGNALS}
    for offset, entry in enumerate(reversed(entries)):
        if not isinstance(entry, dict):
            continue
        recency_weight = decay ** offset
        score_components = entry.get("after_score_components")
        if isinstance(score_components, dict):
            for signal in READING_DISCOVERY_SIGNALS:
                skip_key = f"{signal}_skipped"
                if bool(score_components.get(skip_key, False)):
                    continue
                score_key = READING_SELF_TEACH_SCORE_KEYS[signal]
                if score_key not in score_components:
                    continue
                quality = min(1.0, max(0.0, _reading_float(
                    score_components.get(score_key, 0.0))))
                weighted[signal] += recency_weight * max(0.0, 1.0 - quality)
                weights[signal] += recency_weight
        representation_progress = entry.get("representation_progress")
        if (isinstance(representation_progress, dict)
                and bool(representation_progress.get("enabled", False))):
            signal_after = representation_progress.get("signal_after")
            signal_after = signal_after if isinstance(signal_after, dict) else {}
            signal_deltas = representation_progress.get("signal_deltas")
            signal_deltas = signal_deltas if isinstance(signal_deltas, dict) else {}
            for signal, value in signal_after.items():
                if signal not in READING_DISCOVERY_SIGNALS:
                    continue
                quality = min(1.0, max(0.0, _reading_float(value, 0.0)))
                quality_deficit = max(0.0, 1.0 - quality)
                regression_deficit = max(
                    0.0, -_reading_float(signal_deltas.get(signal, 0.0)))
                deficit = max(quality_deficit, regression_deficit)
                if deficit <= 0.0:
                    continue
                weighted[signal] += recency_weight * deficit
                weights[signal] += recency_weight
        learning_event = entry.get("learning_event")
        if (isinstance(learning_event, dict)
                and bool(learning_event.get("triggered", False))):
            signal = str(learning_event.get("top_signal", ""))
            if signal in READING_DISCOVERY_SIGNALS:
                event_score = min(1.0, max(
                    0.0, _reading_float(
                        learning_event.get("event_score", 0.0))))
                if event_score > 0.0:
                    weighted[signal] += recency_weight * event_score
                    weights[signal] += recency_weight
        top_signal = str(entry.get("self_teach_top_signal", ""))
        if top_signal in top_counts:
            top_counts[top_signal] += 1
    deficits = {
        signal: float(weighted[signal] / weights[signal])
        for signal in READING_DISCOVERY_SIGNALS
        if weights[signal] > 0.0 and weighted[signal] > 0.0
    }
    concept_signal = max(
        0.0, _reading_float(concept_prior.get("concept_connection_signal", 0.0)))
    if concept_signal > 0.0:
        deficits["connection"] = max(float(deficits.get("connection", 0.0)),
                                     float(concept_signal))
    top_signal = None
    if deficits:
        top_signal = max(deficits.items(), key=lambda item: item[1])[0]
    return {
        "enabled": True,
        "entry_count": int(len(entries)),
        "decay": float(decay),
        "signal_deficits": deficits,
        "top_signal": top_signal,
        "concept_connection_signal": float(concept_signal),
        "concept_insight_prior": concept_prior,
        "top_signal_counts": {
            signal: count for signal, count in top_counts.items() if count},
    }


def reading_self_teach_weight_plan(score_components, budget=0.0,
                                   history_prior=None, history_prior_w=0.5):
    budget = float(budget)
    if budget < 0.0:
        raise ValueError("reading self-teach budget must be non-negative")
    history_prior_w = float(history_prior_w)
    if history_prior_w < 0.0:
        raise ValueError("reading self-teach history prior weight must be non-negative")
    history_prior = history_prior if isinstance(history_prior, dict) else {}
    history_deficits = (
        history_prior.get("signal_deficits")
        if isinstance(history_prior.get("signal_deficits"), dict) else {})
    concept_connection_signal = max(
        0.0, _reading_float(history_prior.get("concept_connection_signal", 0.0)))
    extras = {key: 0.0 for key in READING_SELF_TEACH_WEIGHT_KEYS}
    deficits = {}
    current_deficits = {}
    active = []
    for signal in READING_DISCOVERY_SIGNALS:
        if bool(score_components.get(f"{signal}_skipped", False)):
            continue
        score_key = READING_SELF_TEACH_SCORE_KEYS[signal]
        if score_key not in score_components:
            continue
        score = float(score_components.get(score_key, 0.0))
        quality = min(1.0, max(0.0, score))
        current_deficit = max(0.0, 1.0 - quality)
        history_deficit = max(
            0.0, _reading_float(history_deficits.get(signal, 0.0)))
        if signal == "connection":
            history_deficit = max(history_deficit, concept_connection_signal)
        deficit = max(current_deficit, history_prior_w * history_deficit)
        current_deficits[signal] = float(current_deficit)
        deficits[signal] = float(deficit)
        if deficit > 0.0:
            active.append((signal, deficit))
    total_deficit = sum(deficit for _signal, deficit in active)
    if budget > 0.0 and total_deficit > 0.0:
        for signal, deficit in active:
            signal_budget = budget * deficit / total_deficit
            objectives = READING_SELF_TEACH_SIGNAL_OBJECTIVES[signal]
            objective_share = signal_budget / float(len(objectives))
            for key in objectives:
                extras[key] += objective_share
    top_signal = None
    if active:
        top_signal = max(active, key=lambda item: item[1])[0]
    return {
        "enabled": bool(budget > 0.0),
        "budget": float(budget),
        "total_deficit": float(total_deficit),
        "top_signal": top_signal,
        "signal_deficits": deficits,
        "current_signal_deficits": current_deficits,
        "history_signal_deficits": {
            signal: float(value)
            for signal, value in history_deficits.items()
            if signal in READING_DISCOVERY_SIGNALS},
        "history_prior_enabled": bool(history_prior.get("enabled", False)),
        "history_prior_entry_count": int(history_prior.get("entry_count", 0) or 0),
        "history_prior_top_signal": history_prior.get("top_signal"),
        "history_prior_w": float(history_prior_w),
        "history_concept_connection_signal": float(concept_connection_signal),
        "active_signals": [signal for signal, _deficit in active],
        "weight_extras": {key: float(value) for key, value in extras.items()},
    }


def reading_self_teach_weight_maps(score_components=None, budget=0.0,
                                   history_prior=None, history_prior_w=0.5,
                                   **base_weights):
    budget = float(budget)
    if budget < 0.0:
        raise ValueError("reading self-teach budget must be non-negative")
    missing = [
        key for key in READING_SELF_TEACH_WEIGHT_KEYS
        if key not in base_weights]
    if missing:
        raise ValueError(
            "missing reading self-teach base weights: "
            + ", ".join(sorted(missing)))
    base = {
        key: float(base_weights[key])
        for key in READING_SELF_TEACH_WEIGHT_KEYS}
    effective = dict(base)
    if budget == 0.0:
        return None, base, effective
    if score_components is None:
        raise ValueError(
            "reading self-teach needs score components when budget is positive")
    plan = reading_self_teach_weight_plan(
        score_components, budget=budget,
        history_prior=history_prior, history_prior_w=history_prior_w)
    extras = plan["weight_extras"]
    effective = {
        key: float(base[key]) + float(extras.get(key, 0.0))
        for key in READING_SELF_TEACH_WEIGHT_KEYS}
    return plan, base, effective


def reading_bridge_insight_delta(insight, enabled=True):
    if (not enabled or not insight or bool(insight.get("skipped", True))):
        return 0.0, True
    reduction = float(insight.get("bridge_score_reduction", 0.0))
    connectivity = float(insight.get("bridge_connectivity_gain", 0.0))
    delta = 0.5 * (reduction + connectivity)
    return float(delta), delta >= -1e-9


def reading_concept_insight_report(before_score_components=None,
                                   after_score_components=None,
                                   bridge_insight=None, enabled=True):
    """Measure concept-level discovery separate from the selected score."""
    if not enabled:
        return {
            "enabled": False,
            "allowed": True,
            "delta": 0.0,
            "raw_delta": 0.0,
            "positive_signal_gain": 0.0,
            "negative_signal_drift": 0.0,
            "bridge_delta": 0.0,
            "bridge_allowed": True,
            "signal_gains": {},
        }
    before_score_components = before_score_components or {}
    after_score_components = after_score_components or {}
    gains = {}
    positive = 0.0
    negative = 0.0
    for key in READING_CONCEPT_INSIGHT_SCORE_KEYS:
        before = float(before_score_components.get(key, 0.0))
        after = float(after_score_components.get(key, before))
        gain = after - before
        gains[key] = float(gain)
        positive += max(0.0, gain)
        negative += max(0.0, -gain)
    denom = float(max(1, len(READING_CONCEPT_INSIGHT_SCORE_KEYS)))
    positive_mean = positive / denom
    negative_mean = negative / denom
    bridge_delta, bridge_allowed = reading_bridge_insight_delta(
        bridge_insight, enabled=bridge_insight is not None)
    raw_delta = positive_mean + max(0.0, bridge_delta) - 0.5 * negative_mean
    return {
        "enabled": True,
        "allowed": bool(bridge_allowed and raw_delta >= -1e-9),
        "delta": float(max(0.0, raw_delta)),
        "raw_delta": float(raw_delta),
        "positive_signal_gain": float(positive_mean),
        "negative_signal_drift": float(negative_mean),
        "bridge_delta": float(bridge_delta),
        "bridge_allowed": bool(bridge_allowed),
        "signal_gains": gains,
    }


def reading_weight_update_snapshot(model, max_tensors=24, max_values=8):
    rows = []
    max_tensors = max(0, int(max_tensors))
    max_values = max(1, int(max_values))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if len(rows) >= max_tensors:
                break
            if not param.requires_grad or param.numel() <= 0:
                continue
            flat = param.detach().reshape(-1)
            sample_count = min(max_values, int(flat.numel()))
            if sample_count <= 0:
                continue
            if int(flat.numel()) == sample_count:
                indices = torch.arange(sample_count, device=flat.device)
            else:
                # float32 linspace loses integer precision past 2**24, so its
                # endpoint can round up to numel; clamp to keep index_select in range.
                indices = torch.linspace(
                    0, int(flat.numel()) - 1, steps=sample_count,
                    device=flat.device).round().long().clamp_(
                        0, int(flat.numel()) - 1)
            values = flat.index_select(0, indices).to(
                device="cpu", dtype=torch.float32).tolist()
            rows.append({
                "name": str(name),
                "shape": [int(dim) for dim in param.shape],
                "numel": int(param.numel()),
                "indices": [int(idx) for idx in indices.cpu().tolist()],
                "values": [float(value) for value in values],
            })
    return {"sampled_tensors": rows}


def reading_weight_update_report(before_snapshot, after_snapshot, atol=1e-12):
    before_rows = {
        str(row.get("name", "")): row
        for row in (before_snapshot or {}).get("sampled_tensors", ())
        if isinstance(row, dict)
    }
    changed = []
    sampled_tensor_count = 0
    sampled_value_count = 0
    changed_tensor_count = 0
    changed_value_count = 0
    max_abs_delta = 0.0
    for after_row in (after_snapshot or {}).get("sampled_tensors", ()):
        if not isinstance(after_row, dict):
            continue
        name = str(after_row.get("name", ""))
        before_row = before_rows.get(name)
        if not isinstance(before_row, dict):
            continue
        before_values = before_row.get("values", ())
        after_values = after_row.get("values", ())
        value_deltas = [
            abs(float(after) - float(before))
            for before, after in zip(before_values, after_values)
        ]
        if not value_deltas:
            continue
        sampled_tensor_count += 1
        sampled_value_count += len(value_deltas)
        tensor_changed_values = sum(1 for delta in value_deltas if delta > atol)
        tensor_max_delta = max(value_deltas)
        max_abs_delta = max(max_abs_delta, tensor_max_delta)
        if tensor_changed_values:
            changed_tensor_count += 1
            changed_value_count += tensor_changed_values
            changed.append({
                "name": name,
                "changed_values": int(tensor_changed_values),
                "sampled_values": int(len(value_deltas)),
                "max_abs_delta": float(tensor_max_delta),
            })
    changed.sort(key=lambda row: row["max_abs_delta"], reverse=True)
    return {
        "enabled": True,
        "sampled_tensor_count": int(sampled_tensor_count),
        "sampled_value_count": int(sampled_value_count),
        "changed": bool(changed_value_count > 0),
        "changed_tensor_count": int(changed_tensor_count),
        "changed_value_count": int(changed_value_count),
        "max_abs_delta": float(max_abs_delta),
        "top_changed_tensors": changed[:8],
    }


def reading_discovery_score_components(view_eval, context_eval, metric="both",
                                       margin_w=0.1, neighborhood_eval=None,
                                       cluster_eval=None, fer_eval=None,
                                       bridge_eval=None, gap_eval=None,
                                       sequence_eval=None, span_eval=None,
                                       closure_eval=None, lm_eval=None,
                                       generation_eval=None):
    metric = str(metric)
    if metric not in READING_SCORE_METRICS:
        raise ValueError(f"unknown reading score metric {metric!r}")
    margin_w = float(margin_w)
    view_acc = float(view_eval.get("paired_view_acc", 0.0))
    view_margin = float(view_eval.get("margin", 0.0))
    context_acc = float(context_eval.get("context_target_acc", 0.0))
    context_margin = float(context_eval.get("margin", 0.0))
    span_eval = span_eval or {}
    span_acc = float(span_eval.get("span_completion_acc", 0.0))
    span_margin = float(span_eval.get("margin", 0.0))
    closure_eval = closure_eval or {}
    closure_acc = float(closure_eval.get("context_closure_acc", 0.0))
    closure_margin = float(closure_eval.get("margin", 0.0))
    lm_eval = lm_eval or {}
    language_score = float(lm_eval.get("lm_token_acc", 0.0))
    generation_eval = generation_eval or {"skipped": True}
    generation_token_acc = float(generation_eval.get("token_acc", 0.0))
    generation_exact = float(generation_eval.get("exact", 0.0))
    generation_n = int(generation_eval.get("n_records", 0) or 0)
    generation_unique = int(
        generation_eval.get("unique_generation_count", 0) or 0)
    generation_diversity = (
        float(generation_unique) / float(generation_n) if generation_n else 0.0)
    generation_collapse_penalty = (
        0.5 if bool(generation_eval.get("all_generations_identical", False))
        and generation_n > 1 else 1.0)
    generation_floor = 1.0 / float(max(1, generation_n))
    generation_score = (
        0.0 if bool(generation_eval.get("skipped", False))
        else generation_token_acc * max(generation_diversity, generation_floor)
        * generation_collapse_penalty)
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
    gap_eval = {"skipped": True} if gap_eval is None else gap_eval
    gap_raw_score = max(0.0, float(gap_eval.get("mean_gap_score", 0.0)))
    gap_resolution = 1.0 / (1.0 + gap_raw_score)
    connection_parts = []
    if not bool(bridge_eval.get("skipped", False)):
        connection_parts.append(bridge_score)
    if not bool(gap_eval.get("skipped", False)):
        connection_parts.append(gap_resolution)
    connection_score = (
        float(np.mean(connection_parts)) if connection_parts else 0.0)
    view_score = view_acc + margin_w * view_margin
    context_score = context_acc + margin_w * context_margin
    span_score = span_acc + margin_w * span_margin
    closure_score = closure_acc + margin_w * closure_margin
    sequence_score = sequence_acc + margin_w * sequence_margin
    neighborhood_score = neighborhood_acc + margin_w * neighborhood_margin
    cluster_score = cluster_acc + margin_w * cluster_margin
    scores = {"language": language_score,
              "generation": generation_score,
              "view": view_score, "context": context_score, "span": span_score,
              "closure": closure_score, "sequence": sequence_score,
              "neighborhood": neighborhood_score, "cluster": cluster_score,
              "fer": fer_score, "bridge": bridge_score,
              "connection": connection_score}
    skipped = {"language": bool(lm_eval.get("skipped", False)),
               "generation": bool(generation_eval.get("skipped", False)),
               "view": bool(view_eval.get("skipped", False)),
               "context": bool(context_eval.get("skipped", False)),
               "span": bool(span_eval.get("skipped", False)),
               "closure": bool(closure_eval.get("skipped", False)),
               "sequence": bool(sequence_eval.get("skipped", False)),
               "neighborhood": bool(neighborhood_eval.get("skipped", False)),
               "cluster": bool(cluster_eval.get("skipped", False)),
               "fer": bool(fer_eval.get("skipped", False)),
               "bridge": bool(bridge_eval.get("skipped", False)),
               "connection": not bool(connection_parts)}
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
    if metric == "language":
        score = language_score
    elif metric == "generation":
        score = generation_score
    elif metric == "view":
        score = view_score
    elif metric == "context":
        score = context_score
    elif metric == "span":
        score = span_score
    elif metric == "closure":
        score = closure_score
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
    elif metric == "connection":
        score = connection_score
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
            "language_score": float(language_score),
            "lm_loss": float(lm_eval.get("lm_loss", 0.0)),
            "lm_token_acc": float(lm_eval.get("lm_token_acc", 0.0)),
            "generation_score": float(generation_score),
            "generation_token_acc": float(generation_token_acc),
            "generation_exact": float(generation_exact),
            "generation_diversity": float(generation_diversity),
            "generation_collapse_penalty": float(generation_collapse_penalty),
            "view_score": float(view_score),
            "context_score": float(context_score),
            "span_score": float(span_score),
            "context_closure_score": float(closure_score),
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
            "gap_raw_score": float(gap_raw_score),
            "gap_resolution": float(gap_resolution),
            "gap_target_mass": float(gap_eval.get("mean_gap_target_mass", 0.0)),
            "connection_score": float(connection_score),
            "paired_view_acc": view_acc,
            "paired_view_margin": view_margin,
            "context_target_acc": context_acc,
            "context_target_margin": context_margin,
            "span_completion_acc": span_acc,
            "span_completion_margin": span_margin,
            "context_closure_acc": closure_acc,
            "context_closure_margin": closure_margin,
            "sequence_score": float(sequence_score),
            "sequence_acc": sequence_acc,
            "sequence_margin": sequence_margin,
            "neighborhood_acc": neighborhood_acc,
            "neighborhood_margin": neighborhood_margin,
            "cluster_acc": cluster_acc,
            "cluster_margin": cluster_margin,
            "language_skipped": skipped["language"],
            "generation_skipped": skipped["generation"],
            "view_skipped": skipped["view"],
            "context_skipped": skipped["context"],
            "span_skipped": skipped["span"],
            "closure_skipped": skipped["closure"],
            "sequence_skipped": skipped["sequence"],
            "neighborhood_skipped": skipped["neighborhood"],
            "cluster_skipped": skipped["cluster"],
            "fer_skipped": skipped["fer"],
            "bridge_skipped": skipped["bridge"],
            "connection_skipped": skipped["connection"]}


def reading_eval_bundle(model, vocab, records, device=DEV, eval_n=64, seed=0,
                        token_drop_p=0.15, token_replace_p=0.05,
                        context_keep_p=0.5, span_mask_frac=0.25,
                        context_closure_split_frac=0.5,
                        score_metric="mastery", score_margin_w=0.1,
                        generation_eval_n=0,
                        generation_prompt_tokens=16,
                        generation_max_new_tokens=32,
                        generation_temperature=0.0,
                        generation_top_k=0):
    lm = reading_causal_lm_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 13)
    generation = (
        reading_generation_eval(
            model, vocab, records, device=device, n=generation_eval_n,
            seed=seed + 14, prompt_tokens=generation_prompt_tokens,
            max_new_tokens=generation_max_new_tokens,
            temperature=generation_temperature, top_k=generation_top_k)
        if int(generation_eval_n) > 0 else {
            "enabled": False, "skipped": True,
            "skip_reason": "generation_eval_n_zero",
        })
    view = reading_latent_retrieval_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 17,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p)
    context = reading_context_target_retrieval_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 23,
        context_keep_p=context_keep_p)
    span = reading_span_completion_retrieval_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 24,
        span_frac=span_mask_frac)
    closure = reading_context_closure_retrieval_eval(
        model, vocab, records, device=device, n=eval_n, seed=seed + 26,
        split_frac=context_closure_split_frac)
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
    _gap_records, gap = reading_latent_gap_records(
        model, vocab, records, device=device, n=eval_n, seed=seed + 43,
        feature_dropout=0.0)
    return {"lm": lm,
            "generation": generation,
            "view": view,
            "context_target": context,
            "span_completion": span,
            "context_closure": closure,
            "sequence": sequence,
            "neighborhood": neighborhood,
            "cluster": cluster,
            "fer": fer,
            "bridge": bridge,
            "gap": gap,
            "score_components": reading_discovery_score_components(
                view, context, metric=score_metric, margin_w=score_margin_w,
                neighborhood_eval=neighborhood, cluster_eval=cluster,
                fer_eval=fer, bridge_eval=bridge, gap_eval=gap,
                sequence_eval=sequence, span_eval=span, closure_eval=closure,
                lm_eval=lm, generation_eval=(
                    generation if int(generation_eval_n) > 0 else None))}


def reading_representation_progress_report(before_bundle, after_bundle):
    if not isinstance(before_bundle, dict) or not isinstance(after_bundle, dict):
        return {"enabled": False}
    before_scores = before_bundle.get("score_components")
    after_scores = after_bundle.get("score_components")
    if not isinstance(before_scores, dict) or not isinstance(after_scores, dict):
        return {"enabled": False}
    deltas = {}
    for key in READING_REPRESENTATION_PROGRESS_KEYS:
        if key in before_scores and key in after_scores:
            deltas[key] = float(
                _reading_float(after_scores.get(key), 0.0)
                - _reading_float(before_scores.get(key), 0.0))
    active_signals = []
    signal_before = {}
    signal_after = {}
    signal_deltas = {}
    for signal, score_key in READING_REPRESENTATION_SIGNAL_SCORES:
        if bool(before_scores.get(f"{signal}_skipped", False)) and bool(
                after_scores.get(f"{signal}_skipped", False)):
            continue
        before_value = _reading_float(before_scores.get(score_key), 0.0)
        after_value = _reading_float(after_scores.get(score_key), 0.0)
        active_signals.append(signal)
        signal_before[signal] = float(before_value)
        signal_after[signal] = float(after_value)
        signal_deltas[signal] = float(after_value - before_value)
    if active_signals:
        organization_before = float(np.mean([
            signal_before[signal] for signal in active_signals]))
        organization_after = float(np.mean([
            signal_after[signal] for signal in active_signals]))
    else:
        organization_before = 0.0
        organization_after = 0.0
    organization_delta = float(organization_after - organization_before)
    top_gain_signal = None
    top_regression_signal = None
    if signal_deltas:
        top_gain_signal = max(signal_deltas.items(), key=lambda item: item[1])[0]
        top_regression_signal = min(
            signal_deltas.items(), key=lambda item: item[1])[0]
    positive_signal_gain = sum(
        max(0.0, delta) for delta in signal_deltas.values())
    negative_signal_drift = sum(
        max(0.0, -delta) for delta in signal_deltas.values())
    return {
        "enabled": True,
        "active_signals": active_signals,
        "before": {
            key: _reading_float(before_scores.get(key), 0.0)
            for key in READING_REPRESENTATION_PROGRESS_KEYS
            if key in before_scores
        },
        "after": {
            key: _reading_float(after_scores.get(key), 0.0)
            for key in READING_REPRESENTATION_PROGRESS_KEYS
            if key in after_scores
        },
        "delta": deltas,
        "signal_before": signal_before,
        "signal_after": signal_after,
        "signal_deltas": signal_deltas,
        "organization_score_before": float(organization_before),
        "organization_score_after": float(organization_after),
        "organization_score_delta": float(organization_delta),
        "positive_signal_gain": float(positive_signal_gain),
        "negative_signal_drift": float(negative_signal_drift),
        "top_gain_signal": top_gain_signal,
        "top_regression_signal": top_regression_signal,
        "representation_insight_event": bool(
            organization_delta > 0.0 and positive_signal_gain > 0.0),
    }


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


def _latent_slot_fer_parts(slots, eps=1e-8):
    fer_score, parts = latent_concept_fer_scores(slots, eps=eps)
    return {"fer_score": fer_score,
            "fragmentation": parts["fragmentation"],
            "slot_correlation": parts["slot_correlation"],
            "slot_imbalance": parts["slot_imbalance"]}


def reading_latent_fer_records(
        model, vocab, records, device=DEV, n=0, seed=0, feature_dropout=0.0):
    if getattr(model, "latent_concepts", None) is None:
        return [], {"n_records": 0, "sampled": False, "n_selected": 0,
                    "mean_fer_score": 0.0, "max_fer_score": 0.0,
                    "mean_fer_fragmentation": 0.0,
                    "mean_fer_slot_correlation": 0.0,
                    "mean_fer_slot_imbalance": 0.0,
                    "skipped": True}
    candidates = [r for r in records if r.split == "train"]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    scored = []
    score_values = []
    fragmentation_values = []
    correlation_values = []
    imbalance_values = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            slots = model.latent_concept_states(
                txt, feature_dropout=feature_dropout, project=True)
            parts = _latent_slot_fer_parts(slots)
            scores = parts["fer_score"]
            fragmentation = parts["fragmentation"]
            correlation = parts["slot_correlation"]
            imbalance = parts["slot_imbalance"]
            for i, rec in enumerate(batch):
                score = float(scores[i].detach().cpu())
                scored.append((score, rec))
                score_values.append(score)
                fragmentation_values.append(float(fragmentation[i].detach().cpu()))
                correlation_values.append(float(correlation[i].detach().cpu()))
                imbalance_values.append(float(imbalance[i].detach().cpu()))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [rec for _score, rec in scored]
    return selected, {"n_records": len(candidates),
                      "sampled": sampled,
                      "n_selected": len(selected),
                      "mean_fer_score": (
                          float(np.mean(score_values)) if score_values else 0.0),
                      "max_fer_score": (
                          float(max(score_values)) if score_values else 0.0),
                      "mean_fer_fragmentation": (
                          float(np.mean(fragmentation_values))
                          if fragmentation_values else 0.0),
                      "mean_fer_slot_correlation": (
                          float(np.mean(correlation_values))
                          if correlation_values else 0.0),
                      "mean_fer_slot_imbalance": (
                          float(np.mean(imbalance_values))
                          if imbalance_values else 0.0),
                      "skipped": False}


def _reading_sequence_surprise_rows(
        model, vocab, pairs, device=DEV, token_drop_p=0.15,
        token_replace_p=0.05, feature_dropout=0.0, temperature=0.1):
    rows = []
    if (getattr(model, "latent_concepts", None) is None
            or not hasattr(model, "reading_predictor")
            or len(pairs) <= 1):
        return rows
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
            source_slots = model.latent_concept_states(
                anchor_txt, feature_dropout=feature_dropout, project=False)
            target_slots = model.latent_concept_states(
                positive_txt, feature_dropout=0.0, project=False)
            scores, parts = latent_concept_sequence_prediction_scores(
                model.reading_predictor, source_slots, target_slots,
                temperature=temperature)
            ce = parts.get("cross_entropy", scores.new_zeros(scores.shape))
            pos = parts.get("positive_cosine", scores.new_zeros(scores.shape))
            neg = parts.get("hard_negative_cosine", scores.new_zeros(scores.shape))
            rank = parts.get("rank", scores.new_zeros(scores.shape))
            for i, (anchor, positive) in enumerate(pair_batch):
                rows.append({
                    "anchor": anchor,
                    "positive": positive,
                    "sequence_surprise": float(scores[i].detach().cpu()),
                    "sequence_cross_entropy": float(ce[i].detach().cpu()),
                    "sequence_positive_cosine": float(pos[i].detach().cpu()),
                    "sequence_hard_negative_cosine": float(neg[i].detach().cpu()),
                    "sequence_rank": float(rank[i].detach().cpu()),
                })
    return rows


def reading_sequence_surprise_records(
        model, vocab, records, device=DEV, n=0, seed=0,
        token_drop_p=0.15, token_replace_p=0.05, feature_dropout=0.0,
        temperature=0.1):
    if (getattr(model, "latent_concepts", None) is None
            or not hasattr(model, "reading_predictor")):
        return [], {"n_records": 0, "n_pairs": 0, "sampled": False,
                    "n_selected": 0, "mean_sequence_surprise": 0.0,
                    "max_sequence_surprise": 0.0, "skipped": True}
    pairs, mine_report = mine_reading_sequence_pairs(
        records, split="train", n=n, seed=seed)
    rows = _reading_sequence_surprise_rows(
        model, vocab, pairs, device=device, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p, feature_dropout=feature_dropout,
        temperature=temperature)
    rows.sort(key=lambda row: row["sequence_surprise"], reverse=True)
    selected = []
    seen = set()
    for row in rows:
        for rec in (row["anchor"], row["positive"]):
            if rec.rec_id in seen:
                continue
            seen.add(rec.rec_id)
            selected.append(rec)

    def mean_field(name):
        return float(np.mean([row[name] for row in rows])) if rows else 0.0

    def max_field(name):
        return float(max([row[name] for row in rows])) if rows else 0.0

    return selected, {"n_records": mine_report.get("n_records", 0),
                      "n_pairs": len(pairs),
                      "sampled": mine_report.get("sampled", False),
                      "n_selected": len(selected),
                      "mean_sequence_surprise": mean_field("sequence_surprise"),
                      "max_sequence_surprise": max_field("sequence_surprise"),
                      "mean_sequence_cross_entropy": mean_field(
                          "sequence_cross_entropy"),
                      "mean_sequence_positive_cosine": mean_field(
                          "sequence_positive_cosine"),
                      "mean_sequence_hard_negative_cosine": mean_field(
                          "sequence_hard_negative_cosine"),
                      "mean_sequence_rank": mean_field("sequence_rank"),
                      "skipped": not bool(rows),
                      "mining": mine_report}


def reading_context_closure_surprise_records(
        model, vocab, records, device=DEV, n=0, seed=0, split="train",
        split_frac=0.5, feature_dropout=0.0, temperature=0.1):
    if (getattr(model, "latent_concepts", None) is None
            or _reading_completion_predictor(model) is None):
        return [], {"n_records": 0, "sampled": False, "split": split,
                    "n_selected": 0, "mean_score": 0.0,
                    "max_score": 0.0, "mean_closure_surprise": 0.0,
                    "max_closure_surprise": 0.0, "n_usable": 0,
                    "skipped": True}
    candidates = [r for r in records if r.split == split]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=int(n), replace=False)
        candidates = [candidates[int(i)] for i in idx]
    rows = []
    temp = max(float(temperature), 1e-6)
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            prefix_txt, suffix_txt, used = split_reading_context_closure(
                txt, vocab.pad, split_frac=split_frac)
            usable = used.any(1)
            if not bool(usable.any()):
                continue
            usable_idx = torch.where(usable)[0]
            batch = [batch[int(i.detach().cpu())] for i in usable_idx]
            txt = txt.index_select(0, usable_idx)
            prefix_txt = prefix_txt.index_select(0, usable_idx)
            suffix_txt = suffix_txt.index_select(0, usable_idx)
            prefix_slots = model.latent_concept_states(
                prefix_txt, feature_dropout=feature_dropout, project=False)
            suffix_slots = model.latent_concept_states(
                suffix_txt, feature_dropout=feature_dropout, project=False)
            full_slots = model.latent_concept_states(
                txt, feature_dropout=0.0, project=False)
            closure_surprise, closure_parts = latent_concept_completion_scores(
                _reading_completion_predictor(model),
                {"prefix": prefix_slots, "suffix": suffix_slots},
                full_slots, temperature=temp)
            mode_parts = closure_parts.get("modes", {})
            prefix_parts = mode_parts.get("prefix", {})
            suffix_parts = mode_parts.get("suffix", {})
            zero = closure_surprise.new_zeros(closure_surprise.shape)
            prefix_surprise = prefix_parts.get("surprise", zero)
            suffix_surprise = suffix_parts.get("surprise", zero)
            prefix_ce = prefix_parts.get("cross_entropy", zero)
            suffix_ce = suffix_parts.get("cross_entropy", zero)
            mean_cosine = closure_parts.get("positive_cosine", zero)
            prefix_pos = prefix_parts.get("positive_cosine", zero)
            suffix_pos = suffix_parts.get("positive_cosine", zero)
            prefix_hard = prefix_parts.get("hard_negative_cosine", zero)
            suffix_hard = suffix_parts.get("hard_negative_cosine", zero)
            prefix_rank = prefix_parts.get("rank", zero)
            suffix_rank = suffix_parts.get("rank", zero)
            valid_count = txt.ne(vocab.pad).sum(1).clamp_min(1)
            prefix_rate = (
                prefix_txt.ne(vocab.pad).sum(1).to(dtype=torch.float32)
                / valid_count.to(dtype=torch.float32))
            suffix_rate = (
                suffix_txt.ne(vocab.pad).sum(1).to(dtype=torch.float32)
                / valid_count.to(dtype=torch.float32))
            for i, rec in enumerate(batch):
                rows.append({
                    "record": rec,
                    "closure_surprise": float(
                        closure_surprise[i].detach().cpu()),
                    "prefix_surprise": float(prefix_surprise[i].detach().cpu()),
                    "suffix_surprise": float(suffix_surprise[i].detach().cpu()),
                    "closure_cross_entropy": float(
                        (0.5 * (prefix_ce[i] + suffix_ce[i])).detach().cpu()),
                    "prefix_cross_entropy": float(prefix_ce[i].detach().cpu()),
                    "suffix_cross_entropy": float(suffix_ce[i].detach().cpu()),
                    "closure_cosine": float(mean_cosine[i].detach().cpu()),
                    "prefix_cosine": float(prefix_pos[i].detach().cpu()),
                    "suffix_cosine": float(suffix_pos[i].detach().cpu()),
                    "prefix_hard_negative_cosine": float(
                        prefix_hard[i].detach().cpu()),
                    "suffix_hard_negative_cosine": float(
                        suffix_hard[i].detach().cpu()),
                    "prefix_rank": float(prefix_rank[i].detach().cpu()),
                    "suffix_rank": float(suffix_rank[i].detach().cpu()),
                    "prefix_token_rate": float(prefix_rate[i].detach().cpu()),
                    "suffix_token_rate": float(suffix_rate[i].detach().cpu()),
                })
    rows.sort(key=lambda row: row["closure_surprise"], reverse=True)
    selected = [row["record"] for row in rows]

    def mean_field(name):
        return float(np.mean([row[name] for row in rows])) if rows else 0.0

    def max_field(name):
        return float(max([row[name] for row in rows])) if rows else 0.0

    return selected, {"n_records": len(candidates),
                      "sampled": sampled,
                      "split": split,
                      "n_selected": len(selected),
                      "n_usable": len(rows),
                      "mean_score": mean_field("closure_surprise"),
                      "max_score": max_field("closure_surprise"),
                      "mean_closure_surprise": mean_field(
                          "closure_surprise"),
                      "max_closure_surprise": max_field(
                          "closure_surprise"),
                      "mean_prefix_surprise": mean_field(
                          "prefix_surprise"),
                      "mean_suffix_surprise": mean_field(
                          "suffix_surprise"),
                      "mean_closure_cross_entropy": mean_field(
                          "closure_cross_entropy"),
                      "mean_prefix_cross_entropy": mean_field(
                          "prefix_cross_entropy"),
                      "mean_suffix_cross_entropy": mean_field(
                          "suffix_cross_entropy"),
                      "mean_closure_cosine": mean_field("closure_cosine"),
                      "mean_prefix_cosine": mean_field("prefix_cosine"),
                      "mean_suffix_cosine": mean_field("suffix_cosine"),
                      "mean_prefix_hard_negative_cosine": mean_field(
                          "prefix_hard_negative_cosine"),
                      "mean_suffix_hard_negative_cosine": mean_field(
                          "suffix_hard_negative_cosine"),
                      "mean_prefix_rank": mean_field("prefix_rank"),
                      "mean_suffix_rank": mean_field("suffix_rank"),
                      "mean_prefix_token_rate": mean_field(
                          "prefix_token_rate"),
                      "mean_suffix_token_rate": mean_field(
                          "suffix_token_rate"),
                      "split_frac": float(split_frac),
                      "temperature": float(temperature),
                      "skipped": not bool(rows)}


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


def reading_latent_gap_records(
        model, vocab, records, device=DEV, n=0, seed=0, feature_dropout=0.0,
        temperature=0.1, self_loop_w=0.0, transitive_steps=2,
        transitive_w=0.1, target_power=1.0):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return [], {"n_records": 0, "sampled": False, "n_selected": 0,
                    "mean_gap_score": 0.0, "max_gap_score": 0.0,
                    "mean_gap_kl": 0.0, "mean_gap_cosine": 0.0,
                    "mean_gap_entropy": 0.0, "mean_gap_target_mass": 0.0,
                    "mean_gap_present_overlap": 0.0,
                    "memory_filled": 0, "graph_ready": False,
                    "skipped": True}
    candidates = [r for r in records if r.split == "train"]
    sampled = bool(n and n < len(candidates))
    if sampled:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idx]
    scored = []
    values = {
        "gap_score": [],
        "gap_kl": [],
        "gap_cosine": [],
        "gap_entropy": [],
        "gap_target_mass": [],
        "gap_present_overlap": [],
    }
    graph_ready = False
    usable_count = 0
    model.eval()
    with torch.no_grad():
        active = memory.active()
        active_relations = memory.active_relations()
        active_transitions = memory.active_transitions()
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            slots = model.latent_concept_states(
                txt, feature_dropout=feature_dropout, project=True)
            if hasattr(memory, "gap_scores"):
                scores, parts = memory.gap_scores(
                    slots, temperature=temperature, self_loop_w=self_loop_w,
                    transitive_steps=transitive_steps, transitive_w=transitive_w,
                    target_power=target_power)
            else:
                scores, parts = latent_concept_memory_gap_scores(
                    slots, active, relations=active_relations,
                    transitions=active_transitions, temperature=temperature,
                    self_loop_w=self_loop_w, transitive_steps=transitive_steps,
                    transitive_w=transitive_w, target_power=target_power)
            graph_ready = bool(graph_ready or parts.get("graph_ready", False))
            usable = parts.get("usable")
            if usable is not None:
                usable_count += int(usable.detach().sum().item())
            kl = parts.get("kl", scores.new_zeros(scores.shape))
            cosine = parts.get("cosine", scores.new_zeros(scores.shape))
            entropy = parts.get("entropy", scores.new_zeros(scores.shape))
            target_mass = parts.get("target_mass", scores.new_zeros(scores.shape))
            overlap = parts.get("present_overlap", scores.new_zeros(scores.shape))
            for i, rec in enumerate(batch):
                score = float(scores[i].detach().cpu())
                scored.append((score, rec))
                values["gap_score"].append(score)
                values["gap_kl"].append(float(kl[i].detach().cpu()))
                values["gap_cosine"].append(float(cosine[i].detach().cpu()))
                values["gap_entropy"].append(float(entropy[i].detach().cpu()))
                values["gap_target_mass"].append(float(target_mass[i].detach().cpu()))
                values["gap_present_overlap"].append(float(overlap[i].detach().cpu()))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [rec for score, rec in scored if score > 0.0]

    def mean_value(name):
        rows = values[name]
        return float(np.mean(rows)) if rows else 0.0

    def max_value(name):
        rows = values[name]
        return float(max(rows)) if rows else 0.0

    return selected, {"n_records": len(candidates),
                      "sampled": sampled,
                      "n_selected": len(selected),
                      "mean_gap_score": mean_value("gap_score"),
                      "max_gap_score": max_value("gap_score"),
                      "mean_gap_kl": mean_value("gap_kl"),
                      "mean_gap_cosine": mean_value("gap_cosine"),
                      "mean_gap_entropy": mean_value("gap_entropy"),
                      "mean_gap_target_mass": mean_value("gap_target_mass"),
                      "mean_gap_present_overlap": mean_value("gap_present_overlap"),
                      "memory_filled": int(
                          getattr(memory, "filled",
                                  torch.zeros((), dtype=torch.long)).item()),
                      "graph_ready": bool(graph_ready),
                      "usable_gap_records": int(usable_count),
                      "skipped": not (graph_ready and usable_count > 0)}


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
    return _latent_slot_fer_parts(slots)["fer_score"]


def reading_latent_discovery_records(
        model, vocab, records, device=DEV, n=0, seed=0, context_keep_p=0.5,
        feature_dropout=0.0, curiosity_temperature=0.1,
        curiosity_self_loop_w=0.05, curiosity_transitive_steps=2,
        curiosity_transitive_w=0.1, graph_temperature=0.1,
        graph_self_loop_w=0.05, graph_transitive_steps=2,
        graph_transitive_w=0.1, graph_target_power=1.0,
        cycle_temperature=0.1, cycle_self_loop_w=0.05,
        cycle_transitive_steps=2, cycle_transitive_w=0.1,
        cycle_target_power=1.0, cycle_w=0.5,
        sequence_temperature=0.1, context_closure_split_frac=0.5,
        context_closure_temperature=0.1):
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
    candidate_ids = {rec.rec_id for rec in candidates}
    sequence_pairs, sequence_mining = mine_reading_sequence_pairs(
        records, split="train")
    if candidate_ids:
        sequence_pairs = [
            (a, b) for a, b in sequence_pairs
            if a.rec_id in candidate_ids and b.rec_id in candidate_ids
        ]
    sequence_rows = _reading_sequence_surprise_rows(
        model, vocab, sequence_pairs, device=device,
        token_drop_p=0.0, token_replace_p=0.0,
        feature_dropout=0.0, temperature=sequence_temperature)
    sequence_by_id = {}
    for seq_row in sequence_rows:
        for key in ("anchor", "positive"):
            rec = seq_row[key]
            current = sequence_by_id.get(rec.rec_id)
            if (current is None
                    or seq_row["sequence_surprise"] > current["sequence_surprise"]):
                sequence_by_id[rec.rec_id] = seq_row
    rows = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch = candidates[off:off + 64]
            txt = pack_reading(batch, vocab, device)
            full_slots = model.latent_concept_states(
                txt, feature_dropout=feature_dropout, project=True)
            prefix_txt, suffix_txt, _used = split_reading_context_closure(
                txt, vocab.pad, split_frac=context_closure_split_frac)
            prefix_slots = model.latent_concept_states(
                prefix_txt, feature_dropout=feature_dropout, project=False)
            suffix_slots = model.latent_concept_states(
                suffix_txt, feature_dropout=feature_dropout, project=False)
            closure_target_slots = model.latent_concept_states(
                txt, feature_dropout=0.0, project=False)
            closure, closure_parts = latent_concept_completion_scores(
                _reading_completion_predictor(model),
                {"prefix": prefix_slots, "suffix": suffix_slots},
                closure_target_slots, temperature=context_closure_temperature)
            curiosity, curiosity_parts = latent_concept_graph_curiosity_scores(
                full_slots, memory.active(), memory.active_relations(),
                temperature=curiosity_temperature,
                self_loop_w=curiosity_self_loop_w,
                transitive_steps=curiosity_transitive_steps,
                transitive_w=curiosity_transitive_w)
            gap, gap_parts = latent_concept_memory_gap_scores(
                full_slots, memory.active(), relations=memory.active_relations(),
                transitions=memory.active_transitions(),
                temperature=graph_temperature, self_loop_w=0.0,
                transitive_steps=graph_transitive_steps,
                transitive_w=graph_transitive_w,
                target_power=graph_target_power)
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
            insight, insight_parts = latent_concept_insight_scores(
                source_slots, heldout_slots, memory.active(),
                relations=memory.active_relations(),
                transitions=memory.active_transitions(),
                temperature=graph_temperature, self_loop_w=graph_self_loop_w,
                transitive_steps=graph_transitive_steps,
                transitive_w=graph_transitive_w,
                target_power=graph_target_power)
            fer_parts = _latent_slot_fer_parts(full_slots)
            disorder = fer_parts["fer_score"]
            fer_fragmentation = fer_parts["fragmentation"]
            fer_correlation = fer_parts["slot_correlation"]
            fer_imbalance = fer_parts["slot_imbalance"]
            bridge, bridge_entropy, bridge_connectivity = latent_concept_bridge_scores(
                full_slots, memory.active(), memory.active_relations(),
                memory.active_transitions())
            gap_kl = gap_parts.get("kl", gap.new_zeros(gap.shape))
            gap_cosine = gap_parts.get("cosine", gap.new_zeros(gap.shape))
            gap_entropy = gap_parts.get("entropy", gap.new_zeros(gap.shape))
            gap_target_mass = gap_parts.get(
                "target_mass", gap.new_zeros(gap.shape))
            gap_present_overlap = gap_parts.get(
                "present_overlap", gap.new_zeros(gap.shape))
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
            insight_loss = insight_parts.get("loss", insight.new_zeros(insight.shape))
            insight_kl = insight_parts.get("kl", insight.new_zeros(insight.shape))
            insight_cosine = insight_parts.get(
                "cosine", insight.new_zeros(insight.shape))
            insight_missing_mass = insight_parts.get(
                "missing_mass", insight.new_zeros(insight.shape))
            insight_reachable_mass = insight_parts.get(
                "reachable_mass", insight.new_zeros(insight.shape))
            insight_gain = insight_parts.get("gain", insight.new_zeros(insight.shape))
            closure_ce = closure_parts.get(
                "cross_entropy", closure.new_zeros(closure.shape))
            closure_cosine = closure_parts.get(
                "positive_cosine", closure.new_zeros(closure.shape))
            closure_gap = closure_parts.get(
                "closure_gap", closure.new_zeros(closure.shape))
            closure_rank = closure_parts.get("rank", closure.new_zeros(closure.shape))
            for i, rec in enumerate(batch):
                seq = sequence_by_id.get(rec.rec_id, {})
                rows.append({
                    "record": rec,
                    "closure_surprise": float(closure[i].detach().cpu()),
                    "closure_cross_entropy": float(
                        closure_ce[i].detach().cpu()),
                    "closure_cosine": float(closure_cosine[i].detach().cpu()),
                    "closure_gap": float(closure_gap[i].detach().cpu()),
                    "closure_rank": float(closure_rank[i].detach().cpu()),
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
                    "insight": float(insight[i].detach().cpu()),
                    "insight_loss": float(insight_loss[i].detach().cpu()),
                    "insight_kl": float(insight_kl[i].detach().cpu()),
                    "insight_cosine": float(insight_cosine[i].detach().cpu()),
                    "insight_missing_mass": float(
                        insight_missing_mass[i].detach().cpu()),
                    "insight_reachable_mass": float(
                        insight_reachable_mass[i].detach().cpu()),
                    "insight_gain": float(insight_gain[i].detach().cpu()),
                    "fer_score": float(disorder[i].detach().cpu()),
                    "fer_fragmentation": float(
                        fer_fragmentation[i].detach().cpu()),
                    "fer_slot_correlation": float(
                        fer_correlation[i].detach().cpu()),
                    "fer_slot_imbalance": float(fer_imbalance[i].detach().cpu()),
                    "slot_disorder": float(disorder[i].detach().cpu()),
                    "gap": float(gap[i].detach().cpu()),
                    "gap_kl": float(gap_kl[i].detach().cpu()),
                    "gap_cosine": float(gap_cosine[i].detach().cpu()),
                    "gap_entropy": float(gap_entropy[i].detach().cpu()),
                    "gap_target_mass": float(gap_target_mass[i].detach().cpu()),
                    "gap_present_overlap": float(
                        gap_present_overlap[i].detach().cpu()),
                    "bridge": float(bridge[i].detach().cpu()),
                    "bridge_entropy": float(bridge_entropy[i].detach().cpu()),
                    "bridge_connectivity": float(
                        bridge_connectivity[i].detach().cpu()),
                    "sequence_surprise": float(
                        seq.get("sequence_surprise", 0.0)),
                    "sequence_cross_entropy": float(
                        seq.get("sequence_cross_entropy", 0.0)),
                    "sequence_positive_cosine": float(
                        seq.get("sequence_positive_cosine", 0.0)),
                    "sequence_hard_negative_cosine": float(
                        seq.get("sequence_hard_negative_cosine", 0.0)),
                    "sequence_rank": float(seq.get("sequence_rank", 0.0)),
                })
    components = (
        "curiosity", "gap", "insight", "closure_surprise", "graph", "cycle",
        "fer_score", "bridge", "sequence_surprise")
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
                      "mean_closure_surprise": mean_field("closure_surprise"),
                      "max_closure_surprise": max_field("closure_surprise"),
                      "mean_closure_cross_entropy": mean_field(
                          "closure_cross_entropy"),
                      "mean_closure_cosine": mean_field("closure_cosine"),
                      "mean_closure_gap": mean_field("closure_gap"),
                      "mean_closure_rank": mean_field("closure_rank"),
                      "mean_graph_score": mean_field("graph"),
                      "mean_graph_kl": mean_field("graph_kl"),
                      "mean_graph_cosine": mean_field("graph_cosine"),
                      "mean_cycle_score": mean_field("cycle"),
                      "mean_forward_kl": mean_field("forward_kl"),
                      "mean_reverse_kl": mean_field("reverse_kl"),
                      "mean_source_cycle_kl": mean_field("source_cycle_kl"),
                      "mean_target_cycle_kl": mean_field("target_cycle_kl"),
                      "mean_insight_score": mean_field("insight"),
                      "max_insight_score": max_field("insight"),
                      "mean_insight_loss": mean_field("insight_loss"),
                      "mean_insight_kl": mean_field("insight_kl"),
                      "mean_insight_cosine": mean_field("insight_cosine"),
                      "mean_insight_missing_mass": mean_field(
                          "insight_missing_mass"),
                      "mean_insight_reachable_mass": mean_field(
                          "insight_reachable_mass"),
                      "mean_insight_gain": mean_field("insight_gain"),
                      "mean_fer_score": mean_field("fer_score"),
                      "max_fer_score": max_field("fer_score"),
                      "mean_fer_fragmentation": mean_field("fer_fragmentation"),
                      "mean_fer_slot_correlation": mean_field(
                          "fer_slot_correlation"),
                      "mean_fer_slot_imbalance": mean_field("fer_slot_imbalance"),
                      "mean_slot_disorder": mean_field("slot_disorder"),
                      "mean_gap_score": mean_field("gap"),
                      "max_gap_score": max_field("gap"),
                      "mean_gap_kl": mean_field("gap_kl"),
                      "mean_gap_cosine": mean_field("gap_cosine"),
                      "mean_gap_entropy": mean_field("gap_entropy"),
                      "mean_gap_target_mass": mean_field("gap_target_mass"),
                      "mean_gap_present_overlap": mean_field(
                          "gap_present_overlap"),
                      "mean_bridge_score": mean_field("bridge"),
                      "max_bridge_score": max_field("bridge"),
                      "mean_bridge_entropy": mean_field("bridge_entropy"),
                      "mean_bridge_connectivity": mean_field("bridge_connectivity"),
                      "mean_sequence_surprise": mean_field("sequence_surprise"),
                      "max_sequence_surprise": max_field("sequence_surprise"),
                      "mean_sequence_cross_entropy": mean_field(
                          "sequence_cross_entropy"),
                      "mean_sequence_positive_cosine": mean_field(
                          "sequence_positive_cosine"),
                      "mean_sequence_hard_negative_cosine": mean_field(
                          "sequence_hard_negative_cosine"),
                      "mean_sequence_rank": mean_field("sequence_rank"),
                      "n_sequence_pairs": len(sequence_rows),
                      "sequence_mining": sequence_mining,
                      "skipped": False}


def fit_reading_concepts(model, vocab, records, steps=400, batch=32, lr=1e-3,
                         seed=0, device=DEV, log_every=100,
                         token_drop_p=0.15, token_replace_p=0.05,
                         feature_dropout=0.1, lm_w=0.0,
                         continuation_repair_w=0.0,
                         continuation_repair_steps=4,
                         continuation_repair_prompt_frac=0.5,
                         continuation_repair_temperature=0.0,
                         continuation_repair_top_k=0,
                         repetition_unlikelihood_w=0.0,
                         repetition_unlikelihood_window=32,
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
                         consolidation_w=0.0, consolidation_temperature=0.1,
                         consolidation_balance_w=0.01,
                         consolidation_anchor_w=1.0,
                         consolidation_fer_w=0.0,
                         discovery_w=0.0, discovery_curiosity_w=1.0,
                         discovery_graph_w=1.0, discovery_cycle_w=1.0,
                         discovery_bridge_w=1.0, discovery_fer_w=0.0,
                         reanalysis_w=0.0, reanalysis_graph_w=1.0,
                         reanalysis_cycle_w=0.5,
                         reanalysis_bridge_w=0.5,
                         reanalysis_fer_w=0.0,
                         gap_w=0.0, gap_temperature=0.1,
                         gap_self_loop_w=0.0,
                         gap_transitive_steps=2,
                         gap_transitive_w=0.1,
                         gap_target_power=1.0,
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
                         span_completion_w=0.05, span_mask_frac=0.25,
                         span_completion_temperature=0.1,
                         context_closure_w=0.05,
                         context_closure_split_frac=0.5,
                         context_closure_temperature=0.1,
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
                         replay_batch=0,
                         source_balance_w=READING_DEFAULT_SOURCE_BALANCE_W):
    if getattr(model, "latent_concepts", None) is None:
        raise ValueError("raw reading concept training requires latent concept slots")
    if float(lm_w) < 0.0:
        raise ValueError("reading LM loss weight must be non-negative")
    if float(continuation_repair_w) < 0.0:
        raise ValueError("reading continuation repair weight must be non-negative")
    if int(continuation_repair_steps) < 0:
        raise ValueError("reading continuation repair steps must be non-negative")
    if (float(continuation_repair_prompt_frac) <= 0.0
            or float(continuation_repair_prompt_frac) >= 1.0):
        raise ValueError("reading continuation repair prompt fraction must be in (0, 1)")
    if float(continuation_repair_temperature) < 0.0:
        raise ValueError("reading continuation repair temperature must be non-negative")
    if int(continuation_repair_top_k) < 0:
        raise ValueError("reading continuation repair top-k must be non-negative")
    if float(repetition_unlikelihood_w) < 0.0:
        raise ValueError("reading repetition unlikelihood weight must be non-negative")
    if int(repetition_unlikelihood_window) < 0:
        raise ValueError("reading repetition unlikelihood window must be non-negative")
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
    if float(consolidation_w) < 0.0:
        raise ValueError("reading consolidation weight must be non-negative")
    if float(consolidation_temperature) <= 0.0:
        raise ValueError("reading consolidation temperature must be positive")
    if float(consolidation_balance_w) < 0.0:
        raise ValueError("reading consolidation balance weight must be non-negative")
    if float(consolidation_anchor_w) < 0.0:
        raise ValueError("reading consolidation anchor weight must be non-negative")
    if float(consolidation_fer_w) < 0.0:
        raise ValueError("reading consolidation FER weight must be non-negative")
    if float(discovery_w) < 0.0:
        raise ValueError("reading discovery weight must be non-negative")
    if (float(discovery_curiosity_w) < 0.0
            or float(discovery_graph_w) < 0.0
            or float(discovery_cycle_w) < 0.0
            or float(discovery_bridge_w) < 0.0
            or float(discovery_fer_w) < 0.0):
        raise ValueError("reading discovery component weights must be non-negative")
    if float(reanalysis_w) < 0.0:
        raise ValueError("reading reanalysis weight must be non-negative")
    if (float(reanalysis_graph_w) < 0.0
            or float(reanalysis_cycle_w) < 0.0
            or float(reanalysis_bridge_w) < 0.0
            or float(reanalysis_fer_w) < 0.0):
        raise ValueError("reading reanalysis component weights must be non-negative")
    if float(gap_w) < 0.0:
        raise ValueError("reading gap loss weight must be non-negative")
    if float(gap_temperature) <= 0.0:
        raise ValueError("reading gap temperature must be positive")
    if float(gap_self_loop_w) < 0.0:
        raise ValueError("reading gap self-loop weight must be non-negative")
    if int(gap_transitive_steps) < 1:
        raise ValueError("reading gap transitive steps must be positive")
    if float(gap_transitive_w) < 0.0:
        raise ValueError("reading gap transitive weight must be non-negative")
    if float(gap_target_power) <= 0.0:
        raise ValueError("reading gap target power must be positive")
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
    if float(span_completion_w) < 0.0:
        raise ValueError("reading span completion weight must be non-negative")
    if float(span_mask_frac) <= 0.0 or float(span_mask_frac) >= 1.0:
        raise ValueError("reading span mask fraction must be in (0, 1)")
    if float(span_completion_temperature) <= 0.0:
        raise ValueError("reading span completion temperature must be positive")
    if float(context_closure_w) < 0.0:
        raise ValueError("reading context closure weight must be non-negative")
    if (float(context_closure_split_frac) <= 0.0
            or float(context_closure_split_frac) >= 1.0):
        raise ValueError("reading context closure split fraction must be in (0, 1)")
    if float(context_closure_temperature) <= 0.0:
        raise ValueError("reading context closure temperature must be positive")
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
    source_balance_w = float(source_balance_w)
    if source_balance_w < 0.0:
        raise ValueError("reading source balance weight must be non-negative")
    requested_study_strategy = str(study_strategy)
    if requested_study_strategy not in READING_STUDY_STRATEGIES:
        raise ValueError(
            f"unknown reading study strategy {requested_study_strategy!r}")
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_records = [r for r in records if r.split == "train"]
    if not train_records:
        raise ValueError("cannot train reading concepts without train records")
    training_sampling_weights = reading_training_sampling_weights(
        train_records, source_balance_w=source_balance_w)
    training_sampling_active = training_sampling_weights is not None
    training_priority_record_count = sum(
        1 for rec in train_records
        if isinstance(getattr(rec, "meta", None), dict)
        and float(rec.meta.get("replay_priority", 0.0) or 0.0) > 0.0)
    training_priority_active = (
        training_sampling_active and training_priority_record_count > 0)
    train_source_counts = reading_source_counts(train_records)
    training_source_balance_active = (
        training_sampling_active and source_balance_w > 0.0
        and len(train_source_counts) > 1)
    train_record_count = max(1, len(train_records))
    train_source_fracs = [
        float(count) / float(train_record_count)
        for count in train_source_counts.values()]
    replay_records = list(replay_records or [])
    replay_sources = [r for r in replay_records if r.split == "train"] or replay_records
    replay_sampling_weights = reading_replay_sampling_weights(
        replay_sources, source_balance_w=source_balance_w)
    replay_priority_record_count = sum(
        1 for rec in replay_sources
        if float(getattr(rec, "meta", {}).get("replay_priority", 0.0)
                 or 0.0) > 0.0)
    replay_priority_active = (
        replay_sampling_weights is not None and replay_priority_record_count > 0)
    replay_source_counts = reading_source_counts(replay_sources)
    replay_source_balance_active = (
        replay_sampling_weights is not None and source_balance_w > 0.0
        and len(replay_source_counts) > 1)
    replay_record_count = max(1, len(replay_sources))
    replay_source_fracs = [
        float(count) / float(replay_record_count)
        for count in replay_source_counts.values()]
    if int(memory_size) > 0:
        model.enable_latent_concept_memory(int(memory_size))
    weight_update_before = reading_weight_update_snapshot(model)
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
    if consolidation_w and getattr(model, "latent_concept_memory", None) is None:
        raise ValueError("reading consolidation requires latent concept memory")
    if discovery_w and getattr(model, "latent_concept_memory", None) is None:
        raise ValueError("reading discovery requires latent concept memory")
    if reanalysis_w and getattr(model, "latent_concept_memory", None) is None:
        raise ValueError("reading reanalysis requires latent concept memory")
    if gap_w and getattr(model, "latent_concept_memory", None) is None:
        raise ValueError("reading gap loss requires latent concept memory")
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
    last_consolidation = 0.0
    last_consolidation_memory = 0.0
    last_consolidation_anchor = 0.0
    last_consolidation_fer = 0.0
    last_consolidation_nearest = 0.0
    last_consolidation_memory_active = 0
    last_consolidation_skipped = True
    last_discovery = 0.0
    last_discovery_curiosity = 0.0
    last_discovery_graph = 0.0
    last_discovery_cycle = 0.0
    last_discovery_insight = 0.0
    last_discovery_insight_score = 0.0
    last_discovery_insight_missing_mass = 0.0
    last_discovery_insight_reachable_mass = 0.0
    last_discovery_insight_gain = 0.0
    last_discovery_bridge = 0.0
    last_discovery_fer = 0.0
    last_discovery_memory_active = 0
    last_discovery_graph_ready = False
    last_discovery_skipped = True
    last_reanalysis = 0.0
    last_reanalysis_closure = 0.0
    last_reanalysis_cycle = 0.0
    last_reanalysis_bridge = 0.0
    last_reanalysis_fer = 0.0
    last_reanalysis_memory_active = 0
    last_reanalysis_graph_ready = False
    last_reanalysis_skipped = True
    last_gap = 0.0
    last_gap_kl = 0.0
    last_gap_cosine = 0.0
    last_gap_entropy = 0.0
    last_gap_target_mass = 0.0
    last_gap_present_overlap = 0.0
    last_gap_memory_active = 0
    last_gap_graph_ready = False
    last_gap_skipped = True
    last_association = 0.0
    last_composition = 0.0
    last_graph_predict = 0.0
    last_graph_cycle = 0.0
    last_bridge = 0.0
    last_context_target = 0.0
    last_span_completion = 0.0
    last_span_hidden_rate = 0.0
    last_span_skipped = True
    last_context_closure = 0.0
    last_context_closure_prefix_rate = 0.0
    last_context_closure_suffix_rate = 0.0
    last_context_closure_skipped = True
    last_sequence = 0.0
    last_sequence_transition_updates = 0
    last_neighborhood = 0.0
    last_transition = 0.0
    last_cluster = 0.0
    last_replay = 0.0
    last_replay_priority_mean = 0.0
    last_replay_priority_max = 0.0
    last_training_priority_mean = 0.0
    last_training_priority_max = 0.0
    last_lm_loss = 0.0
    last_continuation_repair = 0.0
    last_continuation_repair_enabled = False
    last_continuation_repair_skipped = True
    last_continuation_repair_skip_reason = "not_run"
    last_continuation_repair_tokens = 0
    last_continuation_repair_generated_tokens = 0
    last_continuation_repair_changed_tokens = 0
    last_repetition_unlikelihood = 0.0
    last_repetition_unlikelihood_enabled = False
    last_repetition_unlikelihood_skipped = True
    last_repetition_unlikelihood_skip_reason = "not_run"
    last_repetition_unlikelihood_tokens = 0
    last_repetition_unlikelihood_candidates = 0

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
        elif study_strategy == "fer":
            selected, report = reading_latent_fer_records(
                model, vocab, records, device=device, n=probe_n,
                seed=seed + 1301 + int(step),
                feature_dropout=0.0)
            report = report | {"strategy": "fer"}
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
        elif study_strategy == "sequence":
            selected, report = reading_sequence_surprise_records(
                model, vocab, records, device=device, n=probe_n,
                seed=seed + 1301 + int(step),
                token_drop_p=0.0, token_replace_p=0.0,
                feature_dropout=0.0,
                temperature=sequence_temperature)
            report = report | {"strategy": "sequence"}
        elif study_strategy == "closure":
            selected, report = reading_context_closure_surprise_records(
                model, vocab, records, device=device, n=probe_n,
                seed=seed + 1301 + int(step),
                split_frac=context_closure_split_frac,
                feature_dropout=0.0,
                temperature=context_closure_temperature)
            report = report | {"strategy": "closure"}
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
        elif study_strategy == "gap":
            selected, report = reading_latent_gap_records(
                model, vocab, records, device=device, n=probe_n,
                seed=seed + 1301 + int(step),
                feature_dropout=0.0,
                temperature=graph_predict_temperature,
                self_loop_w=0.0,
                transitive_steps=graph_predict_transitive_steps,
                transitive_w=graph_predict_transitive_w,
                target_power=graph_predict_target_power)
            report = report | {"strategy": "gap"}
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
                cycle_w=graph_cycle_consistency_w,
                sequence_temperature=sequence_temperature,
                context_closure_split_frac=context_closure_split_frac,
                context_closure_temperature=context_closure_temperature)
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
        source_weights = training_sampling_weights if source is train_records else None
        rec_batch = batch_records(source, rng, batch, weights=source_weights)
        txt = pack_reading(rec_batch, vocab, device)
        lm_loss = (reading_causal_lm_loss(model, txt, pad=vocab.pad)
                   if lm_w else txt.float().sum() * 0.0)
        if repetition_unlikelihood_w:
            repetition_unlikelihood, repetition_unlikelihood_metrics = (
                reading_repetition_unlikelihood_loss(
                    model, txt, pad=vocab.pad,
                    window=repetition_unlikelihood_window))
        else:
            repetition_unlikelihood = txt.float().sum() * 0.0
            repetition_unlikelihood_metrics = {
                "enabled": False, "skipped": True,
                "skip_reason": "weight_zero",
                "tokens": 0, "candidates": 0,
                "window": int(repetition_unlikelihood_window),
            }
        if continuation_repair_w:
            continuation_repair, continuation_repair_metrics = (
                reading_continuation_repair_loss(
                    model, txt, pad=vocab.pad,
                    repair_steps=continuation_repair_steps,
                    prompt_frac=continuation_repair_prompt_frac,
                    temperature=continuation_repair_temperature,
                    top_k=continuation_repair_top_k))
        else:
            continuation_repair = txt.float().sum() * 0.0
            continuation_repair_metrics = {
                "enabled": False, "skipped": True,
                "skip_reason": "weight_zero",
                "tokens": 0, "generated_tokens": 0, "changed_tokens": 0,
                "steps": int(continuation_repair_steps),
                "prompt_frac": float(continuation_repair_prompt_frac),
                "temperature": float(continuation_repair_temperature),
                "top_k": int(continuation_repair_top_k),
            }
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
        if consolidation_w:
            consolidation_loss, consolidation_metrics = (
                reading_latent_memory_consolidation_loss(
                    model, txt, vocab.pad, vocab.unk,
                    token_drop_p=token_drop_p,
                    token_replace_p=token_replace_p,
                    feature_dropout=feature_dropout,
                    temperature=consolidation_temperature,
                    balance_w=consolidation_balance_w,
                    anchor_w=consolidation_anchor_w,
                    fer_w=consolidation_fer_w,
                    fer_fragmentation_w=fer_fragmentation_w,
                    fer_correlation_w=fer_correlation_w,
                    fer_balance_w=fer_balance_w))
        else:
            consolidation_loss = view_loss * 0.0
            zero_metric = view_loss.detach() * 0.0
            memory = getattr(model, "latent_concept_memory", None)
            consolidation_metrics = {
                "memory_loss": zero_metric,
                "anchor_loss": zero_metric,
                "fer_loss": zero_metric,
                "nearest_cosine": zero_metric,
                "memory_active": int(
                    getattr(memory, "filled", torch.zeros((), dtype=torch.long)).item())
                if memory is not None else 0,
                "skipped": True,
            }
        association_loss = (
            reading_latent_association_loss(
                model, txt, feature_dropout=feature_dropout,
                temperature=association_temperature,
                target_power=association_target_power,
                self_loop_w=association_self_loop_w,
                transitive_steps=association_transitive_steps,
                transitive_w=association_transitive_w)
            if association_w else view_loss * 0.0)
        if discovery_w:
            discovery_loss, discovery_metrics = reading_latent_discovery_loss(
                model, txt, vocab.pad, feature_dropout=feature_dropout,
                context_keep_p=context_keep_p,
                curiosity_w=discovery_curiosity_w,
                graph_w=discovery_graph_w,
                cycle_w=discovery_cycle_w,
                bridge_w=discovery_bridge_w,
                fer_w=discovery_fer_w,
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
                cycle_consistency_w=graph_cycle_consistency_w,
                fer_fragmentation_w=fer_fragmentation_w,
                fer_correlation_w=fer_correlation_w,
                fer_balance_w=fer_balance_w)
        else:
            discovery_loss = view_loss * 0.0
            zero_metric = view_loss.detach() * 0.0
            memory = getattr(model, "latent_concept_memory", None)
            discovery_metrics = {
                "curiosity_loss": zero_metric,
                "curiosity_novelty": zero_metric,
                "curiosity_association": zero_metric,
                "graph_loss": zero_metric,
                "graph_kl": zero_metric,
                "graph_cosine": zero_metric,
                "cycle_loss": zero_metric,
                "cycle_forward_kl": zero_metric,
                "cycle_reverse_kl": zero_metric,
                "cycle_source_cycle_kl": zero_metric,
                "cycle_target_cycle_kl": zero_metric,
                "insight_loss": zero_metric,
                "insight_score": zero_metric,
                "insight_kl": zero_metric,
                "insight_cosine": zero_metric,
                "insight_missing_mass": zero_metric,
                "insight_reachable_mass": zero_metric,
                "insight_gain": zero_metric,
                "bridge_loss": zero_metric,
                "bridge_score": zero_metric,
                "bridge_entropy": zero_metric,
                "bridge_connectivity": zero_metric,
                "fer_loss": zero_metric,
                "memory_active": int(
                    getattr(memory, "filled", torch.zeros((), dtype=torch.long)).item())
                if memory is not None else 0,
                "graph_ready": False,
                "skipped": True,
            }
        if reanalysis_w:
            reanalysis_loss, reanalysis_metrics = reading_latent_reanalysis_loss(
                model, txt, vocab.pad, vocab.unk,
                token_drop_p=token_drop_p,
                token_replace_p=token_replace_p,
                feature_dropout=feature_dropout,
                graph_w=reanalysis_graph_w,
                cycle_w=reanalysis_cycle_w,
                bridge_w=reanalysis_bridge_w,
                fer_w=reanalysis_fer_w,
                temperature=graph_predict_temperature,
                self_loop_w=graph_predict_self_loop_w,
                transitive_steps=graph_predict_transitive_steps,
                transitive_w=graph_predict_transitive_w,
                target_power=graph_predict_target_power,
                cycle_consistency_w=graph_cycle_consistency_w,
                fer_fragmentation_w=fer_fragmentation_w,
                fer_correlation_w=fer_correlation_w,
                fer_balance_w=fer_balance_w)
        else:
            reanalysis_loss = view_loss * 0.0
            zero_metric = view_loss.detach() * 0.0
            memory = getattr(model, "latent_concept_memory", None)
            reanalysis_metrics = {
                "closure_loss": zero_metric,
                "closure_kl": zero_metric,
                "closure_cosine": zero_metric,
                "cycle_loss": zero_metric,
                "bridge_loss": zero_metric,
                "fer_loss": zero_metric,
                "memory_active": int(
                    getattr(memory, "filled", torch.zeros((), dtype=torch.long)).item())
                if memory is not None else 0,
                "graph_ready": False,
                "skipped": True,
            }
        if gap_w:
            gap_loss, gap_metrics = reading_latent_gap_loss(
                model, txt, feature_dropout=feature_dropout,
                temperature=gap_temperature,
                self_loop_w=gap_self_loop_w,
                transitive_steps=gap_transitive_steps,
                transitive_w=gap_transitive_w,
                target_power=gap_target_power)
        else:
            gap_loss = view_loss * 0.0
            zero_metric = view_loss.detach() * 0.0
            memory = getattr(model, "latent_concept_memory", None)
            gap_metrics = {
                "gap_loss": zero_metric,
                "gap_kl": zero_metric,
                "gap_cosine": zero_metric,
                "gap_entropy": zero_metric,
                "gap_target_mass": zero_metric,
                "gap_present_overlap": zero_metric,
                "memory_active": int(
                    getattr(memory, "filled", torch.zeros((), dtype=torch.long)).item())
                if memory is not None else 0,
                "graph_ready": False,
                "skipped": True,
            }
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
        if span_completion_w:
            span_completion, span_completion_metrics = (
                reading_span_completion_loss(
                    model, txt, vocab.pad, span_frac=span_mask_frac,
                    feature_dropout=feature_dropout,
                    temperature=span_completion_temperature))
        else:
            span_completion = view_loss * 0.0
            zero_metric = view_loss.detach() * 0.0
            span_completion_metrics = {
                "completion_loss": zero_metric,
                "hidden_token_rate": zero_metric,
                "view_count": 0,
                "skipped": True,
            }
        if context_closure_w:
            context_closure, context_closure_metrics = (
                reading_context_closure_loss(
                    model, txt, vocab.pad, split_frac=context_closure_split_frac,
                    feature_dropout=feature_dropout,
                    temperature=context_closure_temperature))
        else:
            context_closure = view_loss * 0.0
            zero_metric = view_loss.detach() * 0.0
            context_closure_metrics = {
                "completion_loss": zero_metric,
                "prefix_token_rate": zero_metric,
                "suffix_token_rate": zero_metric,
                "view_count": 0,
                "skipped": True,
            }
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
            replay_batch_records = batch_replay_records(
                replay_sources, rng, replay_batch, weights=replay_sampling_weights)
            replay_loss = reading_teacher_latent_consistency_loss(
                model, replay_teacher_model, replay_batch_records, vocab,
                replay_teacher_vocab, device=device,
                feature_dropout=feature_dropout)
        loss = (float(lm_w) * lm_loss
                + float(continuation_repair_w) * continuation_repair
                + float(repetition_unlikelihood_w) * repetition_unlikelihood
                + view_loss + float(factorization_w) * factorization_loss
                + float(fer_w) * fer_loss
                + float(memory_w) * memory_loss
                + float(consolidation_w) * consolidation_loss
                + float(discovery_w) * discovery_loss
                + float(reanalysis_w) * reanalysis_loss
                + float(gap_w) * gap_loss
                + float(association_w) * association_loss
                + float(composition_w) * composition_loss
                + float(graph_predict_w) * graph_predict_loss
                + float(graph_cycle_w) * graph_cycle_loss
                + float(bridge_w) * bridge_loss
                + float(context_target_w) * context_target
                + float(span_completion_w) * span_completion
                + float(context_closure_w) * context_closure
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
                                or discovery_w
                                or reanalysis_w
                                or gap_w
                                or bridge_w)
                            else None)))
        last_transition_updates = 0
        if (graph_predict_w or graph_cycle_w
                or discovery_w
                or reanalysis_w
                or gap_w
                or bridge_w
                or study_strategy in READING_TRANSITION_STUDY_STRATEGIES):
            last_transition_updates = int(update_reading_latent_transitions(
                model, txt, vocab.pad, context_keep_p=context_keep_p,
                feature_dropout=0.0, decay=association_decay))
        last_sequence_transition_updates = 0
        if sequence_pairs and (sequence_w or graph_predict_w
                               or graph_cycle_w or gap_w or bridge_w):
            sequence_pair_batch = batch_reading_neighbor_pairs(
                sequence_pairs, rng, sequence_batch)
            last_sequence_transition_updates = int(
                update_reading_sequence_transitions(
                    model, sequence_pair_batch, vocab, device=device,
                    feature_dropout=0.0, decay=association_decay))
        last_loss = float(loss.detach())
        last_lm_loss = float(lm_loss.detach())
        last_continuation_repair = float(continuation_repair.detach())
        last_continuation_repair_enabled = bool(
            continuation_repair_metrics["enabled"])
        last_continuation_repair_skipped = bool(
            continuation_repair_metrics["skipped"])
        last_continuation_repair_skip_reason = str(
            continuation_repair_metrics.get("skip_reason", ""))
        last_continuation_repair_tokens = int(
            continuation_repair_metrics["tokens"])
        last_continuation_repair_generated_tokens = int(
            continuation_repair_metrics["generated_tokens"])
        last_continuation_repair_changed_tokens = int(
            continuation_repair_metrics["changed_tokens"])
        last_repetition_unlikelihood = float(repetition_unlikelihood.detach())
        last_repetition_unlikelihood_enabled = bool(
            repetition_unlikelihood_metrics["enabled"])
        last_repetition_unlikelihood_skipped = bool(
            repetition_unlikelihood_metrics["skipped"])
        last_repetition_unlikelihood_skip_reason = str(
            repetition_unlikelihood_metrics.get("skip_reason", ""))
        last_repetition_unlikelihood_tokens = int(
            repetition_unlikelihood_metrics["tokens"])
        last_repetition_unlikelihood_candidates = int(
            repetition_unlikelihood_metrics["candidates"])
        last_view_loss = float(view_loss.detach())
        last_factorization = float(factorization_loss.detach())
        last_fer = float(fer_loss.detach())
        last_fer_score = float(fer_metrics["fer_score"].detach())
        last_fer_fragmentation = float(fer_metrics["fragmentation"].detach())
        last_fer_slot_correlation = float(fer_metrics["slot_correlation"].detach())
        last_fer_slot_imbalance = float(fer_metrics["slot_imbalance"].detach())
        last_memory = float(memory_loss.detach())
        last_consolidation = float(consolidation_loss.detach())
        last_consolidation_memory = float(
            consolidation_metrics["memory_loss"].detach())
        last_consolidation_anchor = float(
            consolidation_metrics["anchor_loss"].detach())
        last_consolidation_fer = float(
            consolidation_metrics["fer_loss"].detach())
        last_consolidation_nearest = float(
            consolidation_metrics["nearest_cosine"].detach())
        last_consolidation_memory_active = int(
            consolidation_metrics["memory_active"])
        last_consolidation_skipped = bool(consolidation_metrics["skipped"])
        last_discovery = float(discovery_loss.detach())
        last_discovery_curiosity = float(
            discovery_metrics["curiosity_loss"].detach())
        last_discovery_graph = float(discovery_metrics["graph_loss"].detach())
        last_discovery_cycle = float(discovery_metrics["cycle_loss"].detach())
        last_discovery_insight = float(
            discovery_metrics["insight_loss"].detach())
        last_discovery_insight_score = float(
            discovery_metrics["insight_score"].detach())
        last_discovery_insight_missing_mass = float(
            discovery_metrics["insight_missing_mass"].detach())
        last_discovery_insight_reachable_mass = float(
            discovery_metrics["insight_reachable_mass"].detach())
        last_discovery_insight_gain = float(
            discovery_metrics["insight_gain"].detach())
        last_discovery_bridge = float(discovery_metrics["bridge_loss"].detach())
        last_discovery_fer = float(discovery_metrics["fer_loss"].detach())
        last_discovery_memory_active = int(discovery_metrics["memory_active"])
        last_discovery_graph_ready = bool(discovery_metrics["graph_ready"])
        last_discovery_skipped = bool(discovery_metrics["skipped"])
        last_reanalysis = float(reanalysis_loss.detach())
        last_reanalysis_closure = float(
            reanalysis_metrics["closure_loss"].detach())
        last_reanalysis_cycle = float(
            reanalysis_metrics["cycle_loss"].detach())
        last_reanalysis_bridge = float(
            reanalysis_metrics["bridge_loss"].detach())
        last_reanalysis_fer = float(
            reanalysis_metrics["fer_loss"].detach())
        last_reanalysis_memory_active = int(reanalysis_metrics["memory_active"])
        last_reanalysis_graph_ready = bool(reanalysis_metrics["graph_ready"])
        last_reanalysis_skipped = bool(reanalysis_metrics["skipped"])
        last_gap = float(gap_loss.detach())
        last_gap_kl = float(gap_metrics["gap_kl"].detach())
        last_gap_cosine = float(gap_metrics["gap_cosine"].detach())
        last_gap_entropy = float(gap_metrics["gap_entropy"].detach())
        last_gap_target_mass = float(gap_metrics["gap_target_mass"].detach())
        last_gap_present_overlap = float(
            gap_metrics["gap_present_overlap"].detach())
        last_gap_memory_active = int(gap_metrics["memory_active"])
        last_gap_graph_ready = bool(gap_metrics["graph_ready"])
        last_gap_skipped = bool(gap_metrics["skipped"])
        last_association = float(association_loss.detach())
        last_composition = float(composition_loss.detach())
        last_graph_predict = float(graph_predict_loss.detach())
        last_graph_cycle = float(graph_cycle_loss.detach())
        last_bridge = float(bridge_loss.detach())
        last_context_target = float(context_target.detach())
        last_span_completion = float(span_completion.detach())
        last_span_hidden_rate = float(
            span_completion_metrics["hidden_token_rate"].detach())
        last_span_skipped = bool(span_completion_metrics["skipped"])
        last_context_closure = float(context_closure.detach())
        last_context_closure_prefix_rate = float(
            context_closure_metrics["prefix_token_rate"].detach())
        last_context_closure_suffix_rate = float(
            context_closure_metrics["suffix_token_rate"].detach())
        last_context_closure_skipped = bool(context_closure_metrics["skipped"])
        last_sequence = float(sequence_loss.detach())
        last_neighborhood = float(neighborhood_loss.detach())
        last_transition = float(transition_loss.detach())
        last_cluster = float(cluster_loss.detach())
        last_replay = float(replay_loss.detach())
        training_priorities = [
            float(getattr(rec, "meta", {}).get("replay_priority", 0.0) or 0.0)
            for rec in rec_batch] if training_priority_active else []
        last_training_priority_mean = (
            sum(training_priorities) / float(len(training_priorities))
            if training_priorities else 0.0)
        last_training_priority_max = (
            max(training_priorities) if training_priorities else 0.0)
        replay_priorities = [
            float(getattr(rec, "meta", {}).get("replay_priority", 0.0) or 0.0)
            for rec in replay_batch_records] if replay_w and replay_sources else []
        last_replay_priority_mean = (
            sum(replay_priorities) / float(len(replay_priorities))
            if replay_priorities else 0.0)
        last_replay_priority_max = max(replay_priorities) if replay_priorities else 0.0
        if st % log_every == 0 or st == steps:
            print(f"  reading {st}/{steps} loss {last_loss:.3f} "
                  f"lm {last_lm_loss:.3f} "
                  f"repair {last_continuation_repair:.3f} "
                  f"repeat {last_repetition_unlikelihood:.3f} "
                  f"view {last_view_loss:.3f} "
                  f"factor {last_factorization:.3f} "
                  f"fer {last_fer:.3f} "
                  f"memory {last_memory:.3f} "
                  f"consolidate {last_consolidation:.3f} "
                  f"discover {last_discovery:.3f} "
                  f"insight {last_discovery_insight:.3f} "
                  f"reanalyze {last_reanalysis:.3f} "
                  f"gap {last_gap:.3f} "
                  f"assoc {last_association:.3f} "
                  f"compose {last_composition:.3f} "
                  f"graph-predict {last_graph_predict:.3f} "
                  f"graph-cycle {last_graph_cycle:.3f} "
                  f"bridge {last_bridge:.3f} "
                  f"context-target {last_context_target:.3f} "
                  f"span-complete {last_span_completion:.3f} "
                  f"closure {last_context_closure:.3f} "
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
    weight_update = reading_weight_update_report(
        weight_update_before, reading_weight_update_snapshot(model))
    model.reading_train_metrics = {
        "loss": last_loss,
        "weight_update": weight_update,
        "weight_update_changed": bool(weight_update["changed"]),
        "weight_update_changed_tensor_count": int(
            weight_update["changed_tensor_count"]),
        "weight_update_changed_value_count": int(
            weight_update["changed_value_count"]),
        "weight_update_max_abs_delta": float(weight_update["max_abs_delta"]),
        "study_strategy_requested": requested_study_strategy,
        "study_strategy": study_strategy,
        "lm_loss": last_lm_loss,
        "lm_w": float(lm_w),
        "continuation_repair_loss": last_continuation_repair,
        "continuation_repair_w": float(continuation_repair_w),
        "continuation_repair_steps": int(continuation_repair_steps),
        "continuation_repair_prompt_frac": float(
            continuation_repair_prompt_frac),
        "continuation_repair_temperature": float(
            continuation_repair_temperature),
        "continuation_repair_top_k": int(continuation_repair_top_k),
        "continuation_repair_enabled": bool(last_continuation_repair_enabled),
        "continuation_repair_skipped": bool(last_continuation_repair_skipped),
        "continuation_repair_skip_reason": str(
            last_continuation_repair_skip_reason),
        "continuation_repair_tokens": int(last_continuation_repair_tokens),
        "continuation_repair_generated_tokens": int(
            last_continuation_repair_generated_tokens),
        "continuation_repair_changed_tokens": int(
            last_continuation_repair_changed_tokens),
        "repetition_unlikelihood_loss": last_repetition_unlikelihood,
        "repetition_unlikelihood_w": float(repetition_unlikelihood_w),
        "repetition_unlikelihood_window": int(repetition_unlikelihood_window),
        "repetition_unlikelihood_enabled": bool(
            last_repetition_unlikelihood_enabled),
        "repetition_unlikelihood_skipped": bool(
            last_repetition_unlikelihood_skipped),
        "repetition_unlikelihood_skip_reason": str(
            last_repetition_unlikelihood_skip_reason),
        "repetition_unlikelihood_tokens": int(
            last_repetition_unlikelihood_tokens),
        "repetition_unlikelihood_candidates": int(
            last_repetition_unlikelihood_candidates),
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
        "consolidation_loss": last_consolidation,
        "consolidation_w": float(consolidation_w),
        "consolidation_temperature": float(consolidation_temperature),
        "consolidation_balance_w": float(consolidation_balance_w),
        "consolidation_anchor_w": float(consolidation_anchor_w),
        "consolidation_fer_w": float(consolidation_fer_w),
        "consolidation_memory_loss": last_consolidation_memory,
        "consolidation_anchor_loss": last_consolidation_anchor,
        "consolidation_fer_loss": last_consolidation_fer,
        "consolidation_nearest_cosine": last_consolidation_nearest,
        "consolidation_memory_active": int(last_consolidation_memory_active),
        "consolidation_skipped": bool(last_consolidation_skipped),
        "discovery_loss": last_discovery,
        "discovery_w": float(discovery_w),
        "discovery_curiosity_w": float(discovery_curiosity_w),
        "discovery_graph_w": float(discovery_graph_w),
        "discovery_cycle_w": float(discovery_cycle_w),
        "discovery_bridge_w": float(discovery_bridge_w),
        "discovery_fer_w": float(discovery_fer_w),
        "discovery_curiosity_loss": last_discovery_curiosity,
        "discovery_graph_loss": last_discovery_graph,
        "discovery_cycle_loss": last_discovery_cycle,
        "discovery_insight_loss": last_discovery_insight,
        "discovery_insight_score": last_discovery_insight_score,
        "discovery_insight_missing_mass": last_discovery_insight_missing_mass,
        "discovery_insight_reachable_mass": last_discovery_insight_reachable_mass,
        "discovery_insight_gain": last_discovery_insight_gain,
        "discovery_bridge_loss": last_discovery_bridge,
        "discovery_fer_loss": last_discovery_fer,
        "discovery_memory_active": int(last_discovery_memory_active),
        "discovery_graph_ready": bool(last_discovery_graph_ready),
        "discovery_skipped": bool(last_discovery_skipped),
        "reanalysis_loss": last_reanalysis,
        "reanalysis_w": float(reanalysis_w),
        "reanalysis_graph_w": float(reanalysis_graph_w),
        "reanalysis_cycle_w": float(reanalysis_cycle_w),
        "reanalysis_bridge_w": float(reanalysis_bridge_w),
        "reanalysis_fer_w": float(reanalysis_fer_w),
        "reanalysis_closure_loss": last_reanalysis_closure,
        "reanalysis_cycle_loss": last_reanalysis_cycle,
        "reanalysis_bridge_loss": last_reanalysis_bridge,
        "reanalysis_fer_loss": last_reanalysis_fer,
        "reanalysis_memory_active": int(last_reanalysis_memory_active),
        "reanalysis_graph_ready": bool(last_reanalysis_graph_ready),
        "reanalysis_skipped": bool(last_reanalysis_skipped),
        "gap_loss": last_gap,
        "gap_w": float(gap_w),
        "gap_temperature": float(gap_temperature),
        "gap_self_loop_w": float(gap_self_loop_w),
        "gap_transitive_steps": int(gap_transitive_steps),
        "gap_transitive_w": float(gap_transitive_w),
        "gap_target_power": float(gap_target_power),
        "gap_kl": last_gap_kl,
        "gap_cosine": last_gap_cosine,
        "gap_entropy": last_gap_entropy,
        "gap_target_mass": last_gap_target_mass,
        "gap_present_overlap": last_gap_present_overlap,
        "gap_memory_active": int(last_gap_memory_active),
        "gap_graph_ready": bool(last_gap_graph_ready),
        "gap_skipped": bool(last_gap_skipped),
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
        "span_completion_loss": last_span_completion,
        "span_completion_w": float(span_completion_w),
        "span_mask_frac": float(span_mask_frac),
        "span_completion_temperature": float(span_completion_temperature),
        "span_completion_hidden_token_rate": last_span_hidden_rate,
        "span_completion_skipped": bool(last_span_skipped),
        "context_closure_loss": last_context_closure,
        "context_closure_w": float(context_closure_w),
        "context_closure_split_frac": float(context_closure_split_frac),
        "context_closure_temperature": float(context_closure_temperature),
        "context_closure_prefix_token_rate": last_context_closure_prefix_rate,
        "context_closure_suffix_token_rate": last_context_closure_suffix_rate,
        "context_closure_skipped": bool(last_context_closure_skipped),
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
        "training_weighted_sampling": bool(training_sampling_active),
        "training_priority_sampling": bool(training_priority_active),
        "training_priority_record_count": int(training_priority_record_count),
        "training_priority_mean": float(last_training_priority_mean),
        "training_priority_max": float(last_training_priority_max),
        "training_source_balance_sampling": bool(
            training_source_balance_active),
        "training_source_balance_w": float(source_balance_w),
        "training_source_count": int(len(train_source_counts)),
        "training_max_source_record_frac": (
            float(max(train_source_fracs)) if train_source_fracs else 0.0),
        "training_min_source_record_frac": (
            float(min(train_source_fracs)) if train_source_fracs else 0.0),
        "replay_w": float(replay_w),
        "replay_batch": int(replay_batch),
        "replay_records": len(replay_sources),
        "replay_weighted_sampling": bool(replay_sampling_weights is not None),
        "replay_priority_sampling": bool(replay_priority_active),
        "replay_priority_record_count": int(replay_priority_record_count),
        "replay_priority_mean": float(last_replay_priority_mean),
        "replay_priority_max": float(last_replay_priority_max),
        "replay_source_balance_sampling": bool(replay_source_balance_active),
        "replay_source_balance_w": float(source_balance_w),
        "replay_source_count": int(len(replay_source_counts)),
        "replay_max_source_record_frac": (
            float(max(replay_source_fracs)) if replay_source_fracs else 0.0),
        "replay_min_source_record_frac": (
            float(min(replay_source_fracs)) if replay_source_fracs else 0.0),
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


def reading_round_selection_decision(
        score_delta_from_best, score_min_delta, insight_delta=0.0,
        insight_allowed=True, bridge_insight_gate=False,
        insight_accept_w=0.25, insight_min_delta=0.0,
        representation_delta=0.0, representation_allowed=True,
        representation_gate=False, representation_accept_w=0.0,
        representation_min_delta=0.0, signal_regression_allowed=True):
    base = concept_round_selection_decision(
        score_delta_from_best, score_min_delta, insight_delta=insight_delta,
        insight_allowed=insight_allowed, insight_gate=bridge_insight_gate,
        insight_accept_w=insight_accept_w,
        insight_min_delta=insight_min_delta)
    representation_delta = float(representation_delta)
    representation_accept_w = float(representation_accept_w)
    representation_min_delta = float(representation_min_delta)
    representation_boost = 0.0
    if (bool(representation_allowed) and bool(representation_gate)
            and representation_accept_w > 0.0
            and representation_delta > representation_min_delta):
        representation_boost = representation_accept_w * max(
            0.0, representation_delta)
    representation_effective_delta = (
        float(score_delta_from_best) + representation_boost)
    selection_effective_delta = (
        float(base["insight_effective_delta"]) + representation_boost)
    selected_by_representation = (
        bool(representation_allowed) and representation_boost > 0.0
        and selection_effective_delta > float(score_min_delta)
        and not bool(base["selected_by_score"]))
    pre_signal_selected = bool(base["selected"] or selected_by_representation)
    selected = bool(pre_signal_selected and signal_regression_allowed)
    blocked_by_signal_regression = (
        not bool(signal_regression_allowed) and pre_signal_selected)
    reasons = []
    if selected and bool(base["selected_by_score"]):
        reasons.append("score")
    if selected and bool(base["selected_by_insight"]):
        reasons.append("concept_insight")
    if selected and selected_by_representation:
        reasons.append("representation_insight")
    blocked_reasons = []
    if blocked_by_signal_regression:
        blocked_reasons.append("signal_regression")
    return base | {
        "selected": selected,
        "selected_by_score": bool(selected and base["selected_by_score"]),
        "selected_by_insight": bool(selected and base["selected_by_insight"]),
        "selected_by_representation": bool(
            selected and selected_by_representation),
        "pre_signal_selected": bool(pre_signal_selected),
        "pre_signal_selected_by_score": bool(base["selected_by_score"]),
        "pre_signal_selected_by_insight": bool(base["selected_by_insight"]),
        "pre_signal_selected_by_representation": bool(
            selected_by_representation),
        "blocked_by_signal_regression": bool(blocked_by_signal_regression),
        "blocked_reasons": blocked_reasons,
        "representation_score_boost": float(representation_boost),
        "representation_effective_delta": float(representation_effective_delta),
        "selection_effective_delta": float(selection_effective_delta),
        "selection_reasons": reasons,
    }


def reading_representation_insight_delta(progress, enabled=True):
    if (not enabled or not isinstance(progress, dict)
            or not bool(progress.get("enabled", False))):
        return 0.0, True
    organization_delta = _reading_float(
        progress.get("organization_score_delta", 0.0))
    positive_gain = _reading_float(progress.get("positive_signal_gain", 0.0))
    negative_drift = _reading_float(progress.get("negative_signal_drift", 0.0))
    raw_delta = organization_delta + 0.25 * positive_gain - 0.25 * negative_drift
    allowed = bool(
        progress.get("representation_insight_event", False)
        and organization_delta >= -1e-9
        and raw_delta >= -1e-9)
    return float(raw_delta), bool(allowed)


def fit_reading_concepts_select_best(
        model, vocab, records, steps=400, batch=32, lr=1e-3,
        seed=0, device=DEV, log_every=100,
        token_drop_p=0.15, token_replace_p=0.05,
        feature_dropout=0.1, lm_w=0.0,
        continuation_repair_w=0.0, continuation_repair_steps=4,
        continuation_repair_prompt_frac=0.5,
        continuation_repair_temperature=0.0,
        continuation_repair_top_k=0,
        repetition_unlikelihood_w=0.0, repetition_unlikelihood_window=32,
        invariance_w=25.0, variance_w=25.0,
        covariance_w=1.0, variance_target=1.0,
        factorization_w=0.05, factorization_variance=0.05,
        factorization_margin=0.2, factorization_covariance_w=0.05,
        fer_w=0.0, fer_fragmentation_w=1.0,
        fer_correlation_w=1.0, fer_balance_w=0.1,
        memory_w=0.05, memory_size=64,
        memory_temperature=0.1, memory_momentum=0.95,
        memory_balance_w=0.01,
        consolidation_w=0.0, consolidation_temperature=0.1,
        consolidation_balance_w=0.01, consolidation_anchor_w=1.0,
        consolidation_fer_w=0.0,
        discovery_w=0.0, discovery_curiosity_w=1.0,
        discovery_graph_w=1.0, discovery_cycle_w=1.0,
        discovery_bridge_w=1.0, discovery_fer_w=0.0,
        reanalysis_w=0.0, reanalysis_graph_w=1.0,
        reanalysis_cycle_w=0.5, reanalysis_bridge_w=0.5,
        reanalysis_fer_w=0.0,
        gap_w=0.0, gap_temperature=0.1, gap_self_loop_w=0.0,
        gap_transitive_steps=2, gap_transitive_w=0.1,
        gap_target_power=1.0,
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
        span_completion_w=0.05, span_mask_frac=0.25,
        span_completion_temperature=0.1,
        context_closure_w=0.05, context_closure_split_frac=0.5,
        context_closure_temperature=0.1,
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
        study_self_teach_w=0.0,
        self_teach_history_prior=None,
        self_teach_history_prior_w=0.5,
        eval_n=64, score_metric="mastery", score_margin_w=0.1,
        generation_eval_n=0, generation_prompt_tokens=16,
        generation_max_new_tokens=32, generation_temperature=0.0,
        generation_top_k=0,
        signal_regression_tolerance=READING_DEFAULT_SIGNAL_REGRESSION_TOLERANCE,
        score_min_delta=0.0, score_patience=0, score_target=0.0,
        insight_accept_w=0.25, insight_min_delta=0.0,
        representation_accept_w=0.0, representation_min_delta=0.0,
        rounds=1, before_bundle=None, score_records=None,
        source_balance_w=READING_DEFAULT_SOURCE_BALANCE_W):
    records = list(records)
    score_records = list(score_records or records)
    schedule = _step_schedule(steps, rounds)
    if not schedule:
        raise ValueError("reading selected training requires at least one step")
    score_min_delta = float(score_min_delta)
    if score_min_delta < 0.0:
        raise ValueError("reading study score min delta must be non-negative")
    score_patience = int(score_patience)
    if score_patience < 0:
        raise ValueError("reading study score patience must be non-negative")
    score_target = float(score_target)
    if score_target < 0.0:
        raise ValueError("reading study score target must be non-negative")
    signal_regression_tolerance = float(signal_regression_tolerance)
    if signal_regression_tolerance < 0.0:
        raise ValueError(
            "reading study signal regression tolerance must be non-negative")
    insight_accept_w = float(insight_accept_w)
    if insight_accept_w < 0.0:
        raise ValueError("reading study insight accept weight must be non-negative")
    insight_min_delta = float(insight_min_delta)
    if insight_min_delta < 0.0:
        raise ValueError("reading study insight min delta must be non-negative")
    representation_accept_w = float(representation_accept_w)
    if representation_accept_w < 0.0:
        raise ValueError(
            "reading study representation accept weight must be non-negative")
    representation_min_delta = float(representation_min_delta)
    if representation_min_delta < 0.0:
        raise ValueError(
            "reading study representation min delta must be non-negative")
    study_self_teach_w = float(study_self_teach_w)
    if study_self_teach_w < 0.0:
        raise ValueError("reading study self-teach weight must be non-negative")
    source_balance_w = float(source_balance_w)
    if source_balance_w < 0.0:
        raise ValueError("reading source balance weight must be non-negative")
    if int(memory_size) > 0:
        model.enable_latent_concept_memory(int(memory_size))
    generation_eval_kwargs = {
        "generation_eval_n": int(generation_eval_n),
        "generation_prompt_tokens": int(generation_prompt_tokens),
        "generation_max_new_tokens": int(generation_max_new_tokens),
        "generation_temperature": float(generation_temperature),
        "generation_top_k": int(generation_top_k),
    }
    initial_study_strategy = resolve_reading_study_strategy(
        study_strategy, model)
    before_bundle = before_bundle or reading_eval_bundle(
        model, vocab, score_records, device=device, eval_n=eval_n, seed=seed,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p,
        context_keep_p=context_keep_p, score_metric=score_metric,
        score_margin_w=score_margin_w, span_mask_frac=span_mask_frac,
        context_closure_split_frac=context_closure_split_frac,
        **generation_eval_kwargs)
    bridge_insight_gate = bool(
        bridge_w and initial_study_strategy in READING_POOL_STUDY_STRATEGIES
        and initial_study_strategy not in ("sequence", "fer"))
    replay_records = list(replay_records or [])
    before_replay_bundle = None
    if replay_records and replay_retention_w:
        before_replay_bundle = reading_eval_bundle(
            model, vocab, replay_records, device=device, eval_n=eval_n,
            seed=seed + 4093, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, context_keep_p=context_keep_p,
            score_metric=score_metric, score_margin_w=score_margin_w,
            span_mask_frac=span_mask_frac,
            context_closure_split_frac=context_closure_split_frac,
            **generation_eval_kwargs)

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

    best_state = _model_state_copy(model)
    initial_replay_bundle = before_replay_bundle
    initial_row = selection_row(0, 0, before_bundle, initial_replay_bundle)
    best_score = float(initial_row["score"])
    initial_row["target_met"] = bool(
        float(score_target) > 0.0 and best_score >= float(score_target))
    best_bundle = before_bundle
    best_replay_bundle = initial_replay_bundle
    best_round = 0
    best_metrics = {
        "study_strategy_requested": str(study_strategy),
        "study_strategy": initial_study_strategy,
    } | dict(getattr(model, "reading_train_metrics", {}))
    best_study_reports = []
    best_neighborhood_reports = []
    best_cluster_reports = []
    rounds_report = [initial_row]
    all_study_reports = []
    all_neighborhood_reports = []
    all_cluster_reports = []
    no_improve_rounds = 0
    stopped_early = False
    stop_round = 0
    target_round = None
    current_bundle = before_bundle
    self_teach_reports = []
    if score_target > 0.0 and best_score >= score_target:
        stopped_early = True
        target_round = 0
    for round_i, round_steps in enumerate(schedule, start=1):
        if target_round is not None:
            break
        model.load_state_dict(best_state, strict=False)
        current_bundle = best_bundle
        branch_from_round = int(best_round)
        self_teach_plan = reading_self_teach_weight_plan(
            current_bundle["score_components"], budget=study_self_teach_w,
            history_prior=self_teach_history_prior,
            history_prior_w=self_teach_history_prior_w)
        round_study_strategy = reading_self_teach_study_strategy(
            self_teach_plan, study_strategy, initial_study_strategy, model)
        self_teach_reports.append(
            self_teach_plan | {
                "round": int(round_i),
                "branch_from_round": branch_from_round,
                "study_strategy": round_study_strategy,
                "study_strategy_requested": str(study_strategy),
            })
        weight_extras = self_teach_plan["weight_extras"]
        round_bridge_insight_gate = bool(
            bridge_w and round_study_strategy in (
                READING_BRIDGE_INSIGHT_STUDY_STRATEGIES))
        round_concept_insight_gate = bool(
            round_study_strategy in READING_BRIDGE_INSIGHT_STUDY_STRATEGIES
            and getattr(model, "latent_concept_memory", None) is not None)
        fit_reading_concepts(
            model, vocab, records, steps=round_steps, batch=batch, lr=lr,
            seed=seed + round_i * 1009, device=device, log_every=log_every,
            token_drop_p=token_drop_p, token_replace_p=token_replace_p,
            feature_dropout=feature_dropout,
            lm_w=lm_w + weight_extras["lm_w"],
            continuation_repair_w=(
                continuation_repair_w
                + weight_extras["continuation_repair_w"]),
            continuation_repair_steps=continuation_repair_steps,
            continuation_repair_prompt_frac=continuation_repair_prompt_frac,
            continuation_repair_temperature=continuation_repair_temperature,
            continuation_repair_top_k=continuation_repair_top_k,
            repetition_unlikelihood_w=(
                repetition_unlikelihood_w
                + weight_extras["repetition_unlikelihood_w"]),
            repetition_unlikelihood_window=repetition_unlikelihood_window,
            invariance_w=invariance_w,
            variance_w=variance_w, covariance_w=covariance_w,
            variance_target=variance_target,
            factorization_w=(
                factorization_w + weight_extras["factorization_w"]),
            factorization_variance=factorization_variance,
            factorization_margin=factorization_margin,
            factorization_covariance_w=factorization_covariance_w,
            fer_w=fer_w + weight_extras["fer_w"],
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w,
            fer_balance_w=fer_balance_w,
            memory_w=memory_w,
            memory_size=memory_size,
            memory_temperature=memory_temperature,
            memory_momentum=memory_momentum,
            memory_balance_w=memory_balance_w,
            consolidation_w=consolidation_w,
            consolidation_temperature=consolidation_temperature,
            consolidation_balance_w=consolidation_balance_w,
            consolidation_anchor_w=consolidation_anchor_w,
            consolidation_fer_w=consolidation_fer_w,
            discovery_w=discovery_w + weight_extras["discovery_w"],
            discovery_curiosity_w=discovery_curiosity_w,
            discovery_graph_w=discovery_graph_w,
            discovery_cycle_w=discovery_cycle_w,
            discovery_bridge_w=discovery_bridge_w,
            discovery_fer_w=discovery_fer_w,
            reanalysis_w=reanalysis_w,
            reanalysis_graph_w=reanalysis_graph_w,
            reanalysis_cycle_w=reanalysis_cycle_w,
            reanalysis_bridge_w=reanalysis_bridge_w,
            reanalysis_fer_w=reanalysis_fer_w,
            gap_w=gap_w + weight_extras["gap_w"],
            gap_temperature=gap_temperature,
            gap_self_loop_w=gap_self_loop_w,
            gap_transitive_steps=gap_transitive_steps,
            gap_transitive_w=gap_transitive_w,
            gap_target_power=gap_target_power,
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
            bridge_w=bridge_w + weight_extras["bridge_w"],
            context_target_w=(
                context_target_w + weight_extras["context_target_w"]),
            context_keep_p=context_keep_p,
            context_target_temperature=context_target_temperature,
            span_completion_w=(
                span_completion_w + weight_extras["span_completion_w"]),
            span_mask_frac=span_mask_frac,
            span_completion_temperature=span_completion_temperature,
            context_closure_w=(
                context_closure_w + weight_extras["context_closure_w"]),
            context_closure_split_frac=context_closure_split_frac,
            context_closure_temperature=context_closure_temperature,
            sequence_w=sequence_w + weight_extras["sequence_w"],
            sequence_batch=sequence_batch,
            sequence_temperature=sequence_temperature,
            neighborhood_w=neighborhood_w + weight_extras["neighborhood_w"],
            neighborhood_batch=neighborhood_batch,
            neighborhood_probe_n=neighborhood_probe_n,
            neighborhood_refresh_steps=neighborhood_refresh_steps,
            neighborhood_temperature=neighborhood_temperature,
            neighborhood_margin=neighborhood_margin,
            transition_w=transition_w + weight_extras["transition_w"],
            transition_batch=transition_batch,
            transition_temperature=transition_temperature,
            transition_margin=transition_margin,
            cluster_w=cluster_w + weight_extras["cluster_w"],
            cluster_batch=cluster_batch,
            cluster_probe_n=cluster_probe_n,
            cluster_refresh_steps=cluster_refresh_steps,
            cluster_temperature=cluster_temperature,
            cluster_margin=cluster_margin,
            cluster_min_size=cluster_min_size,
            study_strategy=round_study_strategy, study_probe_n=study_probe_n,
            study_hard_max=study_hard_max,
            study_refresh_steps=study_refresh_steps,
            replay_records=replay_records,
            replay_teacher_model=replay_teacher_model,
            replay_teacher_vocab=replay_teacher_vocab,
            replay_w=replay_w, replay_batch=replay_batch,
            source_balance_w=source_balance_w)
        round_train_metrics = dict(getattr(model, "reading_train_metrics", {}))
        round_train_metrics["self_teach_plan"] = self_teach_plan
        round_train_metrics["self_teach_study_strategy"] = round_study_strategy
        round_train_metrics["self_teach_study_signal"] = (
            self_teach_plan.get("top_signal"))
        round_train_metrics["self_teach_base_weights"] = {
            "lm_w": float(lm_w),
            "continuation_repair_w": float(continuation_repair_w),
            "repetition_unlikelihood_w": float(repetition_unlikelihood_w),
            "factorization_w": float(factorization_w),
            "fer_w": float(fer_w),
            "discovery_w": float(discovery_w),
            "gap_w": float(gap_w),
            "bridge_w": float(bridge_w),
            "context_target_w": float(context_target_w),
            "span_completion_w": float(span_completion_w),
            "context_closure_w": float(context_closure_w),
            "sequence_w": float(sequence_w),
            "neighborhood_w": float(neighborhood_w),
            "transition_w": float(transition_w),
            "cluster_w": float(cluster_w),
        }
        round_study_reports = [
            report | {"round": int(round_i)}
            for report in getattr(model, "reading_study_reports", [])]
        round_neighborhood_reports = [
            report | {"round": int(round_i)}
            for report in getattr(model, "reading_neighborhood_reports", [])]
        round_cluster_reports = [
            report | {"round": int(round_i)}
            for report in getattr(model, "reading_cluster_reports", [])]
        all_study_reports.extend(round_study_reports)
        all_neighborhood_reports.extend(round_neighborhood_reports)
        all_cluster_reports.extend(round_cluster_reports)
        bundle = reading_eval_bundle(
            model, vocab, score_records, device=device, eval_n=eval_n, seed=seed,
            token_drop_p=token_drop_p, token_replace_p=token_replace_p,
            context_keep_p=context_keep_p, score_metric=score_metric,
            score_margin_w=score_margin_w, span_mask_frac=span_mask_frac,
            context_closure_split_frac=context_closure_split_frac,
            **generation_eval_kwargs)
        replay_bundle = None
        if before_replay_bundle is not None:
            replay_bundle = reading_eval_bundle(
                model, vocab, replay_records, device=device, eval_n=eval_n,
                seed=seed + 4093, token_drop_p=token_drop_p,
                token_replace_p=token_replace_p, context_keep_p=context_keep_p,
                score_metric=score_metric, score_margin_w=score_margin_w,
                span_mask_frac=span_mask_frac,
                context_closure_split_frac=context_closure_split_frac,
                **generation_eval_kwargs)
        row = selection_row(round_i, round_steps, bundle, replay_bundle)
        score = float(row["score"])
        row["target_met"] = bool(score_target > 0.0 and score >= score_target)
        score_delta_from_best = float(score - best_score)
        insight = round_train_metrics.get("study_pool_insight")
        bridge_delta, bridge_allowed = reading_bridge_insight_delta(
            insight, round_bridge_insight_gate)
        concept_insight = reading_concept_insight_report(
            current_bundle["score_components"], bundle["score_components"],
            bridge_insight=insight, enabled=round_concept_insight_gate)
        representation_progress = reading_representation_progress_report(
            current_bundle, bundle)
        representation_insight_delta, representation_insight_allowed = (
            reading_representation_insight_delta(
                representation_progress,
                enabled=representation_accept_w > 0.0))
        signal_regression = signal_regression_report(
            current_bundle["score_components"], bundle["score_components"],
            READING_SELF_TEACH_SCORE_KEYS,
            tolerance=signal_regression_tolerance,
            signals=READING_DISCOVERY_SIGNALS)
        decision = reading_round_selection_decision(
            score_delta_from_best, score_min_delta,
            insight_delta=concept_insight["delta"],
            insight_allowed=concept_insight["allowed"],
            bridge_insight_gate=round_concept_insight_gate,
            insight_accept_w=insight_accept_w,
            insight_min_delta=insight_min_delta,
            representation_delta=representation_insight_delta,
            representation_allowed=representation_insight_allowed,
            representation_gate=representation_accept_w > 0.0,
            representation_accept_w=representation_accept_w,
            representation_min_delta=representation_min_delta,
            signal_regression_allowed=signal_regression["allowed"])
        selected = decision["selected"]
        rounds_report.append(row | {
            "selected": bool(selected),
            "branch_from_round": branch_from_round,
            "score_delta_from_best": score_delta_from_best,
            "self_teach_plan": self_teach_plan,
            "self_teach_top_signal": self_teach_plan.get("top_signal"),
            "study_strategy_used": round_study_strategy,
            "study_strategy_requested": str(study_strategy),
            "bridge_insight_gate": bool(round_bridge_insight_gate),
            "bridge_insight_delta": float(bridge_delta),
            "bridge_insight_allowed": bool(bridge_allowed),
            "concept_insight_gate": bool(round_concept_insight_gate),
            "concept_insight_delta": float(concept_insight["delta"]),
            "concept_insight_allowed": bool(concept_insight["allowed"]),
            "concept_insight": concept_insight,
            "representation_insight_gate": bool(representation_accept_w > 0.0),
            "representation_insight_delta": float(
                representation_insight_delta),
            "representation_insight_allowed": bool(
                representation_insight_allowed),
            "representation_progress": representation_progress,
            "signal_regression_gate": True,
            "signal_regression_allowed": bool(signal_regression["allowed"]),
            "signal_regression": signal_regression,
            "signal_regression_tolerance": float(
                signal_regression_tolerance),
            "signal_regression_max": float(
                signal_regression["max_regression"]),
            "signal_regression_signals": list(
                signal_regression["regressed_signals"]),
            "training_weighted_sampling": bool(
                round_train_metrics.get("training_weighted_sampling", False)),
            "training_priority_sampling": bool(
                round_train_metrics.get("training_priority_sampling", False)),
            "training_priority_record_count": int(
                round_train_metrics.get("training_priority_record_count", 0)),
            "training_priority_mean": float(
                round_train_metrics.get("training_priority_mean", 0.0)),
            "training_priority_max": float(
                round_train_metrics.get("training_priority_max", 0.0)),
            "training_source_balance_sampling": bool(
                round_train_metrics.get(
                    "training_source_balance_sampling", False)),
            "training_source_balance_w": float(
                round_train_metrics.get("training_source_balance_w", 0.0)),
            "training_source_count": int(
                round_train_metrics.get("training_source_count", 0)),
            "training_max_source_record_frac": float(
                round_train_metrics.get("training_max_source_record_frac", 0.0)),
            "training_min_source_record_frac": float(
                round_train_metrics.get("training_min_source_record_frac", 0.0)),
            "replay_priority_sampling": bool(
                round_train_metrics.get("replay_priority_sampling", False)),
            "replay_priority_record_count": int(
                round_train_metrics.get("replay_priority_record_count", 0)),
            "replay_priority_mean": float(
                round_train_metrics.get("replay_priority_mean", 0.0)),
            "replay_priority_max": float(
                round_train_metrics.get("replay_priority_max", 0.0)),
            "replay_source_balance_sampling": bool(
                round_train_metrics.get("replay_source_balance_sampling", False)),
            "replay_source_balance_w": float(
                round_train_metrics.get("replay_source_balance_w", 0.0)),
            "replay_source_count": int(
                round_train_metrics.get("replay_source_count", 0)),
            "replay_max_source_record_frac": float(
                round_train_metrics.get("replay_max_source_record_frac", 0.0)),
            "replay_min_source_record_frac": float(
                round_train_metrics.get("replay_min_source_record_frac", 0.0)),
            "weight_update": round_train_metrics.get("weight_update", {}),
            "weight_update_changed": bool(
                round_train_metrics.get("weight_update_changed", False)),
            "weight_update_changed_tensor_count": int(
                round_train_metrics.get("weight_update_changed_tensor_count", 0)),
            "weight_update_changed_value_count": int(
                round_train_metrics.get("weight_update_changed_value_count", 0)),
            "weight_update_max_abs_delta": float(
                round_train_metrics.get("weight_update_max_abs_delta", 0.0)),
            "selected_by_score": bool(decision["selected_by_score"]),
            "selected_by_insight": bool(decision["selected_by_insight"]),
            "selected_by_representation": bool(
                decision["selected_by_representation"]),
            "pre_signal_selected": bool(
                decision.get("pre_signal_selected", False)),
            "pre_signal_selected_by_score": bool(
                decision.get("pre_signal_selected_by_score", False)),
            "pre_signal_selected_by_insight": bool(
                decision.get("pre_signal_selected_by_insight", False)),
            "pre_signal_selected_by_representation": bool(
                decision.get("pre_signal_selected_by_representation", False)),
            "blocked_by_signal_regression": bool(
                decision.get("blocked_by_signal_regression", False)),
            "blocked_reasons": list(decision.get("blocked_reasons", ())),
            "insight_score_boost": float(decision["insight_score_boost"]),
            "insight_effective_delta": float(
                decision["insight_effective_delta"]),
            "representation_score_boost": float(
                decision["representation_score_boost"]),
            "representation_effective_delta": float(
                decision["representation_effective_delta"]),
            "selection_effective_delta": float(
                decision["selection_effective_delta"]),
            "selection_reasons": list(decision["selection_reasons"]),
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
            best_bundle = bundle
            best_replay_bundle = replay_bundle
            best_metrics = round_train_metrics
            best_study_reports = list(round_study_reports)
            best_neighborhood_reports = list(round_neighborhood_reports)
            best_cluster_reports = list(round_cluster_reports)
            no_improve_rounds = 0
            if score_target > 0.0 and best_score >= score_target:
                stopped_early = True
                stop_round = round_i
                target_round = round_i
                break
        else:
            no_improve_rounds += 1
            model.load_state_dict(best_state, strict=False)
            if score_patience and no_improve_rounds >= score_patience:
                stopped_early = True
                stop_round = round_i
                break
        current_bundle = best_bundle
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
        "score_target": float(score_target),
        "signal_regression_gate": True,
        "signal_regression_tolerance": float(signal_regression_tolerance),
        "generation_eval_n": int(generation_eval_n),
        "generation_prompt_tokens": int(generation_prompt_tokens),
        "generation_max_new_tokens": int(generation_max_new_tokens),
        "generation_temperature": float(generation_temperature),
        "generation_top_k": int(generation_top_k),
        "target_enabled": bool(score_target > 0.0),
        "target_met": bool(
            score_target > 0.0 and float(best_score) >= score_target),
        "target_round": (
            int(target_round) if target_round is not None else None),
        "adaptive_study_strategy": str(study_strategy) == "auto",
        "training_record_count": int(len(records)),
        "score_record_count": int(len(score_records)),
        "branch_from_best": True,
        "bridge_insight_gate": bool(bridge_insight_gate),
        "concept_insight_gate": bool(
            initial_study_strategy in READING_BRIDGE_INSIGHT_STUDY_STRATEGIES
            and getattr(model, "latent_concept_memory", None) is not None),
        "insight_accept_w": float(insight_accept_w),
        "insight_min_delta": float(insight_min_delta),
        "representation_accept_w": float(representation_accept_w),
        "representation_min_delta": float(representation_min_delta),
        "stopped_early": bool(stopped_early),
        "stop_round": int(stop_round),
        "no_improve_rounds": int(no_improve_rounds),
        "replay_retention_w": float(replay_retention_w),
        "self_teach_w": float(study_self_teach_w),
        "self_teach_reports": self_teach_reports,
        "attempted_weight_update_count": int(
            sum(1 for row in rounds_report
                if int(row.get("round", 0)) > 0
                and bool(row.get("weight_update_changed", False)))),
        "signal_regression_blocked_round_count": int(
            sum(1 for row in rounds_report
                if bool(row.get("blocked_by_signal_regression", False)))),
        "selected_round": int(best_round),
        "accepted_update": bool(best_round > 0),
        "selected_score": float(best_score),
        "before_score": float(initial_row["score"]),
        "selected_score_delta": float(best_score - initial_row["score"]),
        "before_base_score": float(initial_row["base_score"]),
        "before_replay_score": (
            float(before_replay_bundle["score_components"]["score"])
            if before_replay_bundle is not None else None),
        "selected_replay_score": (
            float(best_replay_bundle["score_components"]["score"])
            if best_replay_bundle is not None else None),
        "rounds": rounds_report,
    }
    selected_rows = [row for row in rounds_report if row["round"] == best_round]
    if selected_rows:
        selected_row = selected_rows[0]
        selection["selected_by_score"] = bool(
            selected_row.get("selected_by_score", False))
        selection["selected_by_insight"] = bool(
            selected_row.get("selected_by_insight", False))
        selection["selected_by_representation"] = bool(
            selected_row.get("selected_by_representation", False))
        selection["selected_insight_score_boost"] = float(
            selected_row.get("insight_score_boost", 0.0))
        selection["selected_insight_effective_delta"] = float(
            selected_row.get("insight_effective_delta", 0.0))
        selection["selected_representation_score_boost"] = float(
            selected_row.get("representation_score_boost", 0.0))
        selection["selected_representation_effective_delta"] = float(
            selected_row.get("representation_effective_delta", 0.0))
        selection["selected_effective_delta"] = float(
            selected_row.get("selection_effective_delta", 0.0))
        selection["selected_reasons"] = list(
            selected_row.get("selection_reasons", ()))
        selection["selected_signal_regression"] = selected_row.get(
            "signal_regression")
        selection["selected_signal_regression_allowed"] = bool(
            selected_row.get("signal_regression_allowed", True))
        selection["selected_insight"] = selected_row.get("study_pool_insight")
        selection["selected_representation_progress"] = selected_row.get(
            "representation_progress")
    best_metrics = best_metrics | {"selection": selection}
    model.reading_train_metrics = best_metrics
    model.reading_study_reports = best_study_reports
    model.reading_neighborhood_reports = best_neighborhood_reports
    model.reading_cluster_reports = best_cluster_reports
    model.reading_attempted_study_reports = all_study_reports
    model.reading_attempted_neighborhood_reports = all_neighborhood_reports
    model.reading_attempted_cluster_reports = all_cluster_reports
    model.reading_selection_report = selection
    return model, vocab, selection


def train_reading_concepts(records, steps=400, batch=32, d=96, layers=3, heads=4,
                           text_encoder_arch="transformer", text_encoder_layers=1,
                           latent_concept_slots=READING_DEFAULT_LATENT_CONCEPT_SLOTS,
                           latent_concept_layers=1,
                           latent_concept_topk=0,
                           latent_concept_prefix=False,
                           latent_concept_refine=False,
                           latent_concept_refine_gate_init=-2.0,
                           lr=1e-3, seed=0, device=DEV, log_every=100,
                           token_drop_p=0.15, token_replace_p=0.05,
                           feature_dropout=0.1, lm_w=0.0,
                           continuation_repair_w=0.0,
                           continuation_repair_steps=4,
                           continuation_repair_prompt_frac=0.5,
                           continuation_repair_temperature=0.0,
                           continuation_repair_top_k=0,
                           repetition_unlikelihood_w=0.0,
                           repetition_unlikelihood_window=32,
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
                           consolidation_w=0.0, consolidation_temperature=0.1,
                           consolidation_balance_w=0.01,
                           consolidation_anchor_w=1.0,
                           consolidation_fer_w=0.0,
                           discovery_w=0.0, discovery_curiosity_w=1.0,
                           discovery_graph_w=1.0, discovery_cycle_w=1.0,
                           discovery_bridge_w=1.0, discovery_fer_w=0.0,
                           reanalysis_w=0.0, reanalysis_graph_w=1.0,
                           reanalysis_cycle_w=0.5,
                           reanalysis_bridge_w=0.5,
                           reanalysis_fer_w=0.0,
                           gap_w=0.0, gap_temperature=0.1,
                           gap_self_loop_w=0.0,
                           gap_transitive_steps=2,
                           gap_transitive_w=0.1,
                           gap_target_power=1.0,
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
                           span_completion_w=0.05, span_mask_frac=0.25,
                           span_completion_temperature=0.1,
                           context_closure_w=0.05,
                           context_closure_split_frac=0.5,
                           context_closure_temperature=0.1,
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
                           study_score_metric="mastery",
                           study_score_margin_w=0.1,
                           study_score_min_delta=0.0,
                           study_score_patience=0,
                           study_score_target=0.0,
                           study_signal_regression_tolerance=(
                               READING_DEFAULT_SIGNAL_REGRESSION_TOLERANCE),
                           study_insight_accept_w=0.25,
                           study_insight_min_delta=0.0,
                           study_representation_accept_w=0.0,
                           study_representation_min_delta=0.0,
                           study_self_teach_w=0.0,
                           self_teach_history_prior=None,
                           self_teach_history_prior_w=0.5,
                           eval_n=64,
                           generation_eval_n=0,
                           generation_prompt_tokens=16,
                           generation_max_new_tokens=32,
                           generation_temperature=0.0,
                           generation_top_k=0,
                           max_vocab=READING_DEFAULT_MAX_VOCAB,
                           source_balance_w=READING_DEFAULT_SOURCE_BALANCE_W):
    if int(latent_concept_slots) <= 0:
        raise ValueError("raw reading concept training requires latent_concept_slots > 0")
    torch.manual_seed(seed)
    vocab = build_reading_vocab(records, max_size=max_vocab)
    model = TextReadingLM(len(vocab), d=d, layers=layers, heads=heads, pad=vocab.pad,
                       text_encoder_arch=text_encoder_arch,
                       text_encoder_layers=text_encoder_layers,
                       latent_concept_slots=latent_concept_slots,
                       latent_concept_layers=latent_concept_layers,
                       latent_concept_topk=latent_concept_topk,
                       latent_concept_prefix=latent_concept_prefix,
                       latent_concept_refine=latent_concept_refine,
                       latent_concept_refine_gate_init=(
                           latent_concept_refine_gate_init),
                       latent_concept_memory_size=memory_size).to(device)
    if study_select_best:
        model, vocab, _selection = fit_reading_concepts_select_best(
            model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
            device=device, log_every=log_every, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, feature_dropout=feature_dropout,
            lm_w=lm_w,
            continuation_repair_w=continuation_repair_w,
            continuation_repair_steps=continuation_repair_steps,
            continuation_repair_prompt_frac=continuation_repair_prompt_frac,
            continuation_repair_temperature=continuation_repair_temperature,
            continuation_repair_top_k=continuation_repair_top_k,
            repetition_unlikelihood_w=repetition_unlikelihood_w,
            repetition_unlikelihood_window=repetition_unlikelihood_window,
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
            consolidation_w=consolidation_w,
            consolidation_temperature=consolidation_temperature,
            consolidation_balance_w=consolidation_balance_w,
            consolidation_anchor_w=consolidation_anchor_w,
            consolidation_fer_w=consolidation_fer_w,
            discovery_w=discovery_w,
            discovery_curiosity_w=discovery_curiosity_w,
            discovery_graph_w=discovery_graph_w,
            discovery_cycle_w=discovery_cycle_w,
            discovery_bridge_w=discovery_bridge_w,
            discovery_fer_w=discovery_fer_w,
            reanalysis_w=reanalysis_w,
            reanalysis_graph_w=reanalysis_graph_w,
            reanalysis_cycle_w=reanalysis_cycle_w,
            reanalysis_bridge_w=reanalysis_bridge_w,
            reanalysis_fer_w=reanalysis_fer_w,
            gap_w=gap_w,
            gap_temperature=gap_temperature,
            gap_self_loop_w=gap_self_loop_w,
            gap_transitive_steps=gap_transitive_steps,
            gap_transitive_w=gap_transitive_w,
            gap_target_power=gap_target_power,
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
            span_completion_w=span_completion_w,
            span_mask_frac=span_mask_frac,
            span_completion_temperature=span_completion_temperature,
            context_closure_w=context_closure_w,
            context_closure_split_frac=context_closure_split_frac,
            context_closure_temperature=context_closure_temperature,
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
            score_target=study_score_target,
            signal_regression_tolerance=study_signal_regression_tolerance,
            generation_eval_n=generation_eval_n,
            generation_prompt_tokens=generation_prompt_tokens,
            generation_max_new_tokens=generation_max_new_tokens,
            generation_temperature=generation_temperature,
            generation_top_k=generation_top_k,
            insight_accept_w=study_insight_accept_w,
            insight_min_delta=study_insight_min_delta,
            representation_accept_w=study_representation_accept_w,
            representation_min_delta=study_representation_min_delta,
            study_self_teach_w=study_self_teach_w,
            self_teach_history_prior=self_teach_history_prior,
            self_teach_history_prior_w=self_teach_history_prior_w,
            rounds=study_rounds,
            source_balance_w=source_balance_w)
        return model, vocab
    self_teach_before_bundle = None
    if float(study_self_teach_w) > 0.0:
        self_teach_before_bundle = reading_eval_bundle(
            model, vocab, records, device=device, eval_n=eval_n, seed=seed,
            token_drop_p=token_drop_p, token_replace_p=token_replace_p,
            context_keep_p=context_keep_p, score_metric=study_score_metric,
            score_margin_w=study_score_margin_w, span_mask_frac=span_mask_frac,
            context_closure_split_frac=context_closure_split_frac,
            generation_eval_n=generation_eval_n,
            generation_prompt_tokens=generation_prompt_tokens,
            generation_max_new_tokens=generation_max_new_tokens,
            generation_temperature=generation_temperature,
            generation_top_k=generation_top_k)
    self_teach_plan, self_teach_base_weights, train_weights = (
        reading_self_teach_weight_maps(
            (self_teach_before_bundle["score_components"]
             if self_teach_before_bundle is not None else None),
            budget=study_self_teach_w,
            history_prior=self_teach_history_prior,
            history_prior_w=self_teach_history_prior_w,
            lm_w=lm_w,
            continuation_repair_w=continuation_repair_w,
            repetition_unlikelihood_w=repetition_unlikelihood_w,
            factorization_w=factorization_w,
            fer_w=fer_w,
            discovery_w=discovery_w,
            gap_w=gap_w,
            bridge_w=bridge_w,
            context_target_w=context_target_w,
            span_completion_w=span_completion_w,
            context_closure_w=context_closure_w,
            sequence_w=sequence_w,
            neighborhood_w=neighborhood_w,
            transition_w=transition_w,
            cluster_w=cluster_w))
    model, vocab = fit_reading_concepts(
        model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
        device=device, log_every=log_every, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p, feature_dropout=feature_dropout,
        lm_w=train_weights["lm_w"],
        continuation_repair_w=train_weights["continuation_repair_w"],
        continuation_repair_steps=continuation_repair_steps,
        continuation_repair_prompt_frac=continuation_repair_prompt_frac,
        continuation_repair_temperature=continuation_repair_temperature,
        continuation_repair_top_k=continuation_repair_top_k,
        repetition_unlikelihood_w=train_weights["repetition_unlikelihood_w"],
        repetition_unlikelihood_window=repetition_unlikelihood_window,
        invariance_w=invariance_w, variance_w=variance_w,
        covariance_w=covariance_w, variance_target=variance_target,
        factorization_w=train_weights["factorization_w"],
        factorization_variance=factorization_variance,
        factorization_margin=factorization_margin,
        factorization_covariance_w=factorization_covariance_w,
        fer_w=train_weights["fer_w"],
        fer_fragmentation_w=fer_fragmentation_w,
        fer_correlation_w=fer_correlation_w,
        fer_balance_w=fer_balance_w,
        memory_w=memory_w,
        memory_size=memory_size,
        memory_temperature=memory_temperature,
        memory_momentum=memory_momentum,
        memory_balance_w=memory_balance_w,
        consolidation_w=consolidation_w,
        consolidation_temperature=consolidation_temperature,
        consolidation_balance_w=consolidation_balance_w,
        consolidation_anchor_w=consolidation_anchor_w,
        consolidation_fer_w=consolidation_fer_w,
        discovery_w=train_weights["discovery_w"],
        discovery_curiosity_w=discovery_curiosity_w,
        discovery_graph_w=discovery_graph_w,
        discovery_cycle_w=discovery_cycle_w,
        discovery_bridge_w=discovery_bridge_w,
        discovery_fer_w=discovery_fer_w,
        reanalysis_w=reanalysis_w,
        reanalysis_graph_w=reanalysis_graph_w,
        reanalysis_cycle_w=reanalysis_cycle_w,
        reanalysis_bridge_w=reanalysis_bridge_w,
        reanalysis_fer_w=reanalysis_fer_w,
        gap_w=train_weights["gap_w"],
        gap_temperature=gap_temperature,
        gap_self_loop_w=gap_self_loop_w,
        gap_transitive_steps=gap_transitive_steps,
        gap_transitive_w=gap_transitive_w,
        gap_target_power=gap_target_power,
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
        bridge_w=train_weights["bridge_w"],
        context_target_w=train_weights["context_target_w"],
        context_keep_p=context_keep_p,
        context_target_temperature=context_target_temperature,
        span_completion_w=train_weights["span_completion_w"],
        span_mask_frac=span_mask_frac,
        span_completion_temperature=span_completion_temperature,
        context_closure_w=train_weights["context_closure_w"],
        context_closure_split_frac=context_closure_split_frac,
        context_closure_temperature=context_closure_temperature,
        sequence_w=train_weights["sequence_w"], sequence_batch=sequence_batch,
        sequence_temperature=sequence_temperature,
        neighborhood_w=train_weights["neighborhood_w"],
        neighborhood_batch=neighborhood_batch,
        neighborhood_probe_n=neighborhood_probe_n,
        neighborhood_refresh_steps=neighborhood_refresh_steps,
        neighborhood_temperature=neighborhood_temperature,
        neighborhood_margin=neighborhood_margin,
        transition_w=train_weights["transition_w"],
        transition_batch=transition_batch,
        transition_temperature=transition_temperature,
        transition_margin=transition_margin,
        cluster_w=train_weights["cluster_w"],
        cluster_batch=cluster_batch,
        cluster_probe_n=cluster_probe_n,
        cluster_refresh_steps=cluster_refresh_steps,
        cluster_temperature=cluster_temperature,
        cluster_margin=cluster_margin,
        cluster_min_size=cluster_min_size,
        study_strategy=study_strategy, study_probe_n=study_probe_n,
        study_hard_max=study_hard_max, study_refresh_steps=study_refresh_steps,
        source_balance_w=source_balance_w)
    if self_teach_plan is not None:
        model.reading_train_metrics = dict(
            getattr(model, "reading_train_metrics", {})) | {
                "self_teach_w": float(study_self_teach_w),
                "self_teach_plan": self_teach_plan,
                "self_teach_base_weights": self_teach_base_weights,
                "self_teach_effective_weights": train_weights,
                "selection": {
                    "enabled": False,
                    "self_teach_w": float(study_self_teach_w),
                    "self_teach_reports": [
                        self_teach_plan | {"round": 1}],
                    "self_teach_plan": self_teach_plan,
                },
            }
    return model, vocab


def run_reading_concepts(data, steps=400, batch=32, d=96, layers=3, heads=4,
                         text_encoder_arch="transformer", text_encoder_layers=1,
                         latent_concept_slots=READING_DEFAULT_LATENT_CONCEPT_SLOTS,
                         latent_concept_layers=1,
                         latent_concept_topk=0,
                         latent_concept_prefix=False, latent_concept_refine=False,
                         latent_concept_refine_gate_init=-2.0,
                         lr=1e-3, seed=0, device=DEV, log_every=100,
                         token_drop_p=0.15, token_replace_p=0.05,
                         feature_dropout=0.1, lm_w=0.0,
                         continuation_repair_w=0.0,
                         continuation_repair_steps=4,
                         continuation_repair_prompt_frac=0.5,
                         continuation_repair_temperature=0.0,
                         continuation_repair_top_k=0,
                         repetition_unlikelihood_w=0.0,
                         repetition_unlikelihood_window=32,
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
                         consolidation_w=0.0, consolidation_temperature=0.1,
                         consolidation_balance_w=0.01,
                         consolidation_anchor_w=1.0,
                         consolidation_fer_w=0.0,
                         discovery_w=0.0, discovery_curiosity_w=1.0,
                         discovery_graph_w=1.0, discovery_cycle_w=1.0,
                         discovery_bridge_w=1.0, discovery_fer_w=0.0,
                         reanalysis_w=0.0, reanalysis_graph_w=1.0,
                         reanalysis_cycle_w=0.5,
                         reanalysis_bridge_w=0.5,
                         reanalysis_fer_w=0.0,
                         gap_w=0.0, gap_temperature=0.1,
                         gap_self_loop_w=0.0,
                         gap_transitive_steps=2,
                         gap_transitive_w=0.1,
                         gap_target_power=1.0,
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
                         span_completion_w=0.05, span_mask_frac=0.25,
                         span_completion_temperature=0.1,
                         context_closure_w=0.05,
                         context_closure_split_frac=0.5,
                         context_closure_temperature=0.1,
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
                         study_score_target=0.0,
                         study_signal_regression_tolerance=(
                             READING_DEFAULT_SIGNAL_REGRESSION_TOLERANCE),
                         study_insight_accept_w=0.25,
                         study_insight_min_delta=0.0,
                         study_representation_accept_w=0.0,
                         study_representation_min_delta=0.0,
                         study_self_teach_w=0.0,
                         reading_objective_profile="manual",
                         reading_objective_profile_report=None,
                         text_field="text", max_tokens=128, min_tokens=8,
                         eval_frac=0.10, eval_n=64,
                         generation_eval_n=0,
                         generation_prompt_tokens=16,
                         generation_max_new_tokens=32,
                         generation_temperature=0.0,
                         generation_top_k=0,
                         max_vocab=READING_DEFAULT_MAX_VOCAB,
                         source_balance_w=READING_DEFAULT_SOURCE_BALANCE_W,
                         out=None, checkpoint=None):
    records = load_reading_records(
        data, text_field=text_field, max_tokens=max_tokens, min_tokens=min_tokens,
        eval_frac=eval_frac, seed=seed)
    torch.manual_seed(seed)
    vocab = build_reading_vocab(records, max_size=max_vocab)
    model = TextReadingLM(len(vocab), d=d, layers=layers, heads=heads, pad=vocab.pad,
                       text_encoder_arch=text_encoder_arch,
                       text_encoder_layers=text_encoder_layers,
                       latent_concept_slots=latent_concept_slots,
                       latent_concept_layers=latent_concept_layers,
                       latent_concept_topk=latent_concept_topk,
                       latent_concept_prefix=latent_concept_prefix,
                       latent_concept_refine=latent_concept_refine,
                       latent_concept_refine_gate_init=(
                           latent_concept_refine_gate_init),
                       latent_concept_memory_size=memory_size).to(device)
    before_bundle = reading_eval_bundle(
        model, vocab, records, device=device, eval_n=eval_n, seed=seed,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p,
        context_keep_p=context_keep_p, score_metric=study_score_metric,
        score_margin_w=study_score_margin_w, span_mask_frac=span_mask_frac,
        context_closure_split_frac=context_closure_split_frac,
        generation_eval_n=generation_eval_n,
        generation_prompt_tokens=generation_prompt_tokens,
        generation_max_new_tokens=generation_max_new_tokens,
        generation_temperature=generation_temperature,
        generation_top_k=generation_top_k)
    selection = {"enabled": False}
    if study_select_best:
        _model, _vocab, selection = fit_reading_concepts_select_best(
            model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
            device=device, log_every=log_every, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, feature_dropout=feature_dropout,
            lm_w=lm_w,
            continuation_repair_w=continuation_repair_w,
            continuation_repair_steps=continuation_repair_steps,
            continuation_repair_prompt_frac=continuation_repair_prompt_frac,
            continuation_repair_temperature=continuation_repair_temperature,
            continuation_repair_top_k=continuation_repair_top_k,
            repetition_unlikelihood_w=repetition_unlikelihood_w,
            repetition_unlikelihood_window=repetition_unlikelihood_window,
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
            consolidation_w=consolidation_w,
            consolidation_temperature=consolidation_temperature,
            consolidation_balance_w=consolidation_balance_w,
            consolidation_anchor_w=consolidation_anchor_w,
            consolidation_fer_w=consolidation_fer_w,
            discovery_w=discovery_w,
            discovery_curiosity_w=discovery_curiosity_w,
            discovery_graph_w=discovery_graph_w,
            discovery_cycle_w=discovery_cycle_w,
            discovery_bridge_w=discovery_bridge_w,
            discovery_fer_w=discovery_fer_w,
            reanalysis_w=reanalysis_w,
            reanalysis_graph_w=reanalysis_graph_w,
            reanalysis_cycle_w=reanalysis_cycle_w,
            reanalysis_bridge_w=reanalysis_bridge_w,
            reanalysis_fer_w=reanalysis_fer_w,
            gap_w=gap_w,
            gap_temperature=gap_temperature,
            gap_self_loop_w=gap_self_loop_w,
            gap_transitive_steps=gap_transitive_steps,
            gap_transitive_w=gap_transitive_w,
            gap_target_power=gap_target_power,
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
            span_completion_w=span_completion_w,
            span_mask_frac=span_mask_frac,
            span_completion_temperature=span_completion_temperature,
            context_closure_w=context_closure_w,
            context_closure_split_frac=context_closure_split_frac,
            context_closure_temperature=context_closure_temperature,
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
            score_target=study_score_target,
            signal_regression_tolerance=study_signal_regression_tolerance,
            generation_eval_n=generation_eval_n,
            generation_prompt_tokens=generation_prompt_tokens,
            generation_max_new_tokens=generation_max_new_tokens,
            generation_temperature=generation_temperature,
            generation_top_k=generation_top_k,
            insight_accept_w=study_insight_accept_w,
            insight_min_delta=study_insight_min_delta,
            representation_accept_w=study_representation_accept_w,
            representation_min_delta=study_representation_min_delta,
            study_self_teach_w=study_self_teach_w,
            rounds=study_rounds,
            before_bundle=before_bundle,
            source_balance_w=source_balance_w)
    else:
        self_teach_plan, self_teach_base_weights, train_weights = (
            reading_self_teach_weight_maps(
                before_bundle["score_components"], budget=study_self_teach_w,
                lm_w=lm_w,
                continuation_repair_w=continuation_repair_w,
                repetition_unlikelihood_w=repetition_unlikelihood_w,
                factorization_w=factorization_w,
                fer_w=fer_w,
                discovery_w=discovery_w,
                gap_w=gap_w,
                bridge_w=bridge_w,
                context_target_w=context_target_w,
                span_completion_w=span_completion_w,
                context_closure_w=context_closure_w,
                sequence_w=sequence_w,
                neighborhood_w=neighborhood_w,
                transition_w=transition_w,
                cluster_w=cluster_w))
        fit_reading_concepts(
            model, vocab, records, steps=steps, batch=batch, lr=lr, seed=seed,
            device=device, log_every=log_every, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, feature_dropout=feature_dropout,
            lm_w=train_weights["lm_w"],
            continuation_repair_w=train_weights["continuation_repair_w"],
            continuation_repair_steps=continuation_repair_steps,
            continuation_repair_prompt_frac=continuation_repair_prompt_frac,
            continuation_repair_temperature=continuation_repair_temperature,
            continuation_repair_top_k=continuation_repair_top_k,
            repetition_unlikelihood_w=train_weights["repetition_unlikelihood_w"],
            repetition_unlikelihood_window=repetition_unlikelihood_window,
            invariance_w=invariance_w, variance_w=variance_w,
            covariance_w=covariance_w, variance_target=variance_target,
            factorization_w=train_weights["factorization_w"],
            factorization_variance=factorization_variance,
            factorization_margin=factorization_margin,
            factorization_covariance_w=factorization_covariance_w,
            fer_w=train_weights["fer_w"],
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w,
            fer_balance_w=fer_balance_w,
            memory_w=memory_w,
            memory_size=memory_size,
            memory_temperature=memory_temperature,
            memory_momentum=memory_momentum,
            memory_balance_w=memory_balance_w,
            consolidation_w=consolidation_w,
            consolidation_temperature=consolidation_temperature,
            consolidation_balance_w=consolidation_balance_w,
            consolidation_anchor_w=consolidation_anchor_w,
            consolidation_fer_w=consolidation_fer_w,
            discovery_w=train_weights["discovery_w"],
            discovery_curiosity_w=discovery_curiosity_w,
            discovery_graph_w=discovery_graph_w,
            discovery_cycle_w=discovery_cycle_w,
            discovery_bridge_w=discovery_bridge_w,
            discovery_fer_w=discovery_fer_w,
            reanalysis_w=reanalysis_w,
            reanalysis_graph_w=reanalysis_graph_w,
            reanalysis_cycle_w=reanalysis_cycle_w,
            reanalysis_bridge_w=reanalysis_bridge_w,
            reanalysis_fer_w=reanalysis_fer_w,
            gap_w=train_weights["gap_w"],
            gap_temperature=gap_temperature,
            gap_self_loop_w=gap_self_loop_w,
            gap_transitive_steps=gap_transitive_steps,
            gap_transitive_w=gap_transitive_w,
            gap_target_power=gap_target_power,
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
            bridge_w=train_weights["bridge_w"],
            context_target_w=train_weights["context_target_w"],
            context_keep_p=context_keep_p,
            context_target_temperature=context_target_temperature,
            span_completion_w=train_weights["span_completion_w"],
            span_mask_frac=span_mask_frac,
            span_completion_temperature=span_completion_temperature,
            context_closure_w=train_weights["context_closure_w"],
            context_closure_split_frac=context_closure_split_frac,
            context_closure_temperature=context_closure_temperature,
            sequence_w=train_weights["sequence_w"], sequence_batch=sequence_batch,
            sequence_temperature=sequence_temperature,
            neighborhood_w=train_weights["neighborhood_w"],
            neighborhood_batch=neighborhood_batch,
            neighborhood_probe_n=neighborhood_probe_n,
            neighborhood_refresh_steps=neighborhood_refresh_steps,
            neighborhood_temperature=neighborhood_temperature,
            neighborhood_margin=neighborhood_margin,
            transition_w=train_weights["transition_w"],
            transition_batch=transition_batch,
            transition_temperature=transition_temperature,
            transition_margin=transition_margin,
            cluster_w=train_weights["cluster_w"],
            cluster_batch=cluster_batch,
            cluster_probe_n=cluster_probe_n,
            cluster_refresh_steps=cluster_refresh_steps,
            cluster_temperature=cluster_temperature,
            cluster_margin=cluster_margin,
            cluster_min_size=cluster_min_size,
            study_strategy=study_strategy, study_probe_n=study_probe_n,
            study_hard_max=study_hard_max,
            study_refresh_steps=study_refresh_steps,
            source_balance_w=source_balance_w)
        if self_teach_plan is not None:
            model.reading_train_metrics = dict(
                getattr(model, "reading_train_metrics", {})) | {
                    "self_teach_w": float(study_self_teach_w),
                    "self_teach_plan": self_teach_plan,
                    "self_teach_base_weights": self_teach_base_weights,
                    "self_teach_effective_weights": train_weights,
                }
            selection = {
                "enabled": False,
                "self_teach_w": float(study_self_teach_w),
                "replay_w": float(replay_w),
                "replay_retention_w": float(replay_retention_w),
                "self_teach_reports": [self_teach_plan | {"round": 1}],
                "self_teach_plan": self_teach_plan,
            }
    after_bundle = reading_eval_bundle(
        model, vocab, records, device=device, eval_n=eval_n, seed=seed,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p,
            context_keep_p=context_keep_p, score_metric=study_score_metric,
            score_margin_w=study_score_margin_w, span_mask_frac=span_mask_frac,
            context_closure_split_frac=context_closure_split_frac,
            generation_eval_n=generation_eval_n,
            generation_prompt_tokens=generation_prompt_tokens,
            generation_max_new_tokens=generation_max_new_tokens,
            generation_temperature=generation_temperature,
            generation_top_k=generation_top_k)
    before_lm = before_bundle["lm"]
    after_lm = after_bundle["lm"]
    before = before_bundle["view"]
    after = after_bundle["view"]
    before_context = before_bundle["context_target"]
    after_context = after_bundle["context_target"]
    before_span = before_bundle["span_completion"]
    after_span = after_bundle["span_completion"]
    before_closure = before_bundle["context_closure"]
    after_closure = after_bundle["context_closure"]
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
    generation_gate = reading_generation_gate(after_bundle["generation"])
    language_gate = bool(after_lm.get("lm_token_acc", 0.0) >= 0.50)
    report_gate = bool(language_gate and generation_gate["passed"])
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
              "latent_concept_topk": int(latent_concept_topk),
              "latent_concept_prefix": bool(latent_concept_prefix),
              "latent_concept_refine": bool(latent_concept_refine),
              "latent_concept_refine_gate_init": float(
                  latent_concept_refine_gate_init),
              "token_drop_p": float(token_drop_p),
              "token_replace_p": float(token_replace_p),
              "feature_dropout": float(feature_dropout),
              "lm_w": float(lm_w),
              "continuation_repair_w": float(continuation_repair_w),
              "continuation_repair_steps": int(continuation_repair_steps),
              "continuation_repair_prompt_frac": float(
                  continuation_repair_prompt_frac),
              "continuation_repair_temperature": float(
                  continuation_repair_temperature),
              "continuation_repair_top_k": int(continuation_repair_top_k),
              "repetition_unlikelihood_w": float(repetition_unlikelihood_w),
              "repetition_unlikelihood_window": int(
                  repetition_unlikelihood_window),
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
              "consolidation_w": float(consolidation_w),
              "consolidation_temperature": float(consolidation_temperature),
              "consolidation_balance_w": float(consolidation_balance_w),
              "consolidation_anchor_w": float(consolidation_anchor_w),
              "consolidation_fer_w": float(consolidation_fer_w),
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
              "span_completion_w": float(span_completion_w),
              "span_mask_frac": float(span_mask_frac),
              "span_completion_temperature": float(span_completion_temperature),
              "context_closure_w": float(context_closure_w),
              "context_closure_split_frac": float(context_closure_split_frac),
              "context_closure_temperature": float(context_closure_temperature),
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
              "study_score_target": float(study_score_target),
              "study_signal_regression_tolerance": float(
                  study_signal_regression_tolerance),
              "study_insight_accept_w": float(study_insight_accept_w),
              "study_insight_min_delta": float(study_insight_min_delta),
              "study_representation_accept_w": float(
                  study_representation_accept_w),
              "study_representation_min_delta": float(
                  study_representation_min_delta),
              "study_self_teach_w": float(study_self_teach_w),
              "generation_eval_n": int(generation_eval_n),
              "generation_prompt_tokens": int(generation_prompt_tokens),
              "generation_max_new_tokens": int(generation_max_new_tokens),
              "generation_temperature": float(generation_temperature),
              "generation_top_k": int(generation_top_k),
              "reading_objective_profile": str(reading_objective_profile),
              "reading_objective_profile_report": (
                  reading_objective_profile_report or {
                      "profile": str(reading_objective_profile),
                      "enabled": str(reading_objective_profile) != "manual",
                      "floors": {},
                      "updates": {},
                      "applied": False,
                  }),
              "reading_replay_bank": {},
              "train_records": sum(r.split == "train" for r in records),
              "eval_records": sum(r.split == "eval" for r in records),
              "max_vocab": int(max_vocab or 0),
              "vocab_capped": bool(max_vocab and int(max_vocab) > 0),
              "source_balance_w": float(source_balance_w),
              "vocab_size": len(vocab),
              "before": before,
              "after": after,
              "before_context_target": before_context,
              "after_context_target": after_context,
              "before_lm": before_lm,
              "after_lm": after_lm,
              "before_generation": before_bundle["generation"],
              "after_generation": after_bundle["generation"],
              "generation_gate": generation_gate,
              "gate": report_gate,
              "gate_report": {
                  "lm_token_acc_min": 0.50,
                  "language_passed": language_gate,
                  "generation_passed": bool(generation_gate["passed"]),
                  "generation_required": bool(generation_gate["required"]),
                  "generation_reason": str(generation_gate["reason"]),
              },
              "before_span_completion": before_span,
              "after_span_completion": after_span,
              "before_context_closure": before_closure,
              "after_context_closure": after_closure,
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
              "representation_progress": reading_representation_progress_report(
                  before_bundle, after_bundle),
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
                  "lm_token_acc": (
                      after_lm.get("lm_token_acc", 0.0)
                      - before_lm.get("lm_token_acc", 0.0)),
                  "lm_loss": (
                      after_lm.get("lm_loss", 0.0)
                      - before_lm.get("lm_loss", 0.0)),
                  "generation_score": (
                      after_bundle["score_components"].get(
                          "generation_score", 0.0)
                      - before_bundle["score_components"].get(
                          "generation_score", 0.0)),
                  "generation_token_acc": (
                      after_bundle["score_components"].get(
                          "generation_token_acc", 0.0)
                      - before_bundle["score_components"].get(
                          "generation_token_acc", 0.0)),
                  "paired_view_acc": (
                      after["paired_view_acc"] - before["paired_view_acc"]),
                  "margin": after.get("margin", 0.0) - before.get("margin", 0.0),
                  "context_target_acc": (
                      after_context.get("context_target_acc", 0.0)
                      - before_context.get("context_target_acc", 0.0)),
                  "context_target_margin": (
                      after_context.get("margin", 0.0)
                      - before_context.get("margin", 0.0)),
                  "span_completion_acc": (
                      after_span.get("span_completion_acc", 0.0)
                      - before_span.get("span_completion_acc", 0.0)),
                  "span_completion_margin": (
                      after_span.get("margin", 0.0)
                      - before_span.get("margin", 0.0)),
                  "context_closure_acc": (
                      after_closure.get("context_closure_acc", 0.0)
                      - before_closure.get("context_closure_acc", 0.0)),
                  "context_closure_margin": (
                      after_closure.get("margin", 0.0)
                      - before_closure.get("margin", 0.0)),
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
    report["learning_event"] = reading_learning_event_report(report)
    report["reading_replay_bank"] = build_reading_replay_bank(
        records, study_reports=getattr(model, "reading_study_reports", []),
        learning_event=report["learning_event"])
    report["reading_mastery_history"] = reading_mastery_history_with_entry(
        [], report)
    report["reading_mastery_history_count"] = report[
        "reading_mastery_history"]["entry_count"]
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
                                      latent_concept_topk=None,
                                      latent_concept_prefix=None,
                                      latent_concept_refine=None,
                                      latent_concept_refine_gate_init=None,
                                      latent_concept_memory_size=None,
                                      max_vocab=READING_DEFAULT_MAX_VOCAB):
    src_model, src_vocab, ckpt = load_checkpoint(checkpoint, device=device)
    vocab = build_reading_vocab(
        reading_records, base_vocab=src_vocab, max_size=max_vocab)
    ckpt_slots = int(getattr(src_model, "latent_concept_slots", 0)
                     or ckpt.get("latent_concept_slots", 0))
    slots = int(latent_concept_slots or ckpt_slots
                or READING_DEFAULT_LATENT_CONCEPT_SLOTS)
    layers = int(latent_concept_layers if latent_concept_layers is not None
                 else ckpt.get("latent_concept_layers", 1))
    topk = int(latent_concept_topk if latent_concept_topk is not None
               else getattr(src_model, "latent_concept_topk",
                            ckpt.get("latent_concept_topk", 0)))
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
    model = TextReadingLM(
        len(vocab), d=int(ckpt.get("d", 96)),
        layers=int(ckpt.get("layers", 3)),
        heads=int(ckpt.get("heads", 4)), pad=vocab.pad,
        text_encoder_arch=ckpt.get("text_encoder_arch", "transformer"),
        text_encoder_layers=int(ckpt.get("text_encoder_layers", 1)),
        latent_concept_slots=slots,
        latent_concept_layers=layers,
        latent_concept_topk=topk,
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
                             lm_w=0.0,
                             continuation_repair_w=0.0,
                             continuation_repair_steps=4,
                             continuation_repair_prompt_frac=0.5,
                             continuation_repair_temperature=0.0,
                             continuation_repair_top_k=0,
                             repetition_unlikelihood_w=0.0,
                             repetition_unlikelihood_window=32,
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
                             consolidation_w=0.0, consolidation_temperature=0.1,
                             consolidation_balance_w=0.01,
                             consolidation_anchor_w=1.0,
                             consolidation_fer_w=0.0,
                             discovery_w=0.0, discovery_curiosity_w=1.0,
                             discovery_graph_w=1.0, discovery_cycle_w=1.0,
                             discovery_bridge_w=1.0, discovery_fer_w=0.0,
                             reanalysis_w=0.0, reanalysis_graph_w=1.0,
                             reanalysis_cycle_w=0.5,
                             reanalysis_bridge_w=0.5,
                             reanalysis_fer_w=0.0,
                             gap_w=0.0, gap_temperature=0.1,
                             gap_self_loop_w=0.0,
                             gap_transitive_steps=2,
                             gap_transitive_w=0.1,
                             gap_target_power=1.0,
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
                             span_completion_w=0.05, span_mask_frac=0.25,
                             span_completion_temperature=0.1,
                             context_closure_w=0.05,
                             context_closure_split_frac=0.5,
                             context_closure_temperature=0.1,
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
                             study_score_target=0.0,
                             study_signal_regression_tolerance=(
                                 READING_DEFAULT_SIGNAL_REGRESSION_TOLERANCE),
                             study_insight_accept_w=0.25,
                             study_insight_min_delta=0.0,
                             study_representation_accept_w=0.0,
                             study_representation_min_delta=0.0,
                             study_self_teach_w=0.0,
                             reading_objective_profile="manual",
                             reading_objective_profile_report=None,
                             replay_w=0.0, replay_batch=0,
                             replay_retention_w=0.0,
                             self_teach_history_prior_w=0.5,
                             text_field="text", max_tokens=128, min_tokens=8,
                             eval_frac=0.10, eval_n=64,
                             generation_eval_n=0,
                             generation_prompt_tokens=16,
                             generation_max_new_tokens=32,
                             generation_temperature=0.0,
                             generation_top_k=0,
                             max_vocab=READING_DEFAULT_MAX_VOCAB,
                             source_balance_w=READING_DEFAULT_SOURCE_BALANCE_W,
                             latent_concept_slots=0, latent_concept_layers=None,
                             latent_concept_topk=None,
                             latent_concept_prefix=None,
                             latent_concept_refine=None,
                             latent_concept_refine_gate_init=None,
                             print_report=True):
    self_teach_history_prior_w = float(self_teach_history_prior_w)
    if self_teach_history_prior_w < 0.0:
        raise ValueError("reading self-teach history prior weight must be non-negative")
    records = load_reading_records(
        data, text_field=text_field, max_tokens=max_tokens, min_tokens=min_tokens,
        eval_frac=eval_frac, seed=seed)
    external_replay_records = (load_reading_records(
        replay_data, require_train=False, require_eval=False, text_field=text_field,
        max_tokens=max_tokens, min_tokens=min_tokens, eval_frac=eval_frac,
        seed=seed + 313) if replay_data else [])
    checkpoint_replay_records = reading_replay_bank_records_from_checkpoint(
        checkpoint, device="cpu")
    replay_records = unique_reading_records_by_id(
        external_replay_records, checkpoint_replay_records)
    replay_study_records = (
        replay_records if str(reading_objective_profile) == "mastery" else [])
    study_records = merge_reading_replay_metadata(records, replay_study_records)
    checkpoint_profile = reading_checkpoint_profile_kwargs(
        reading_objective_profile, replay_records=replay_records,
        replay_w=replay_w, replay_retention_w=replay_retention_w)
    replay_w = checkpoint_profile["replay_w"]
    replay_retention_w = checkpoint_profile["replay_retention_w"]
    reading_objective_profile_report = (
        reading_profile_report_with_checkpoint(
            reading_objective_profile_report,
            checkpoint_profile["reading_checkpoint_profile_report"]))
    torch.manual_seed(seed)
    model, vocab, ckpt = expanded_reading_checkpoint_model(
        checkpoint, records + replay_records, device=device,
        latent_concept_slots=latent_concept_slots,
        latent_concept_layers=latent_concept_layers,
        latent_concept_topk=latent_concept_topk,
        latent_concept_prefix=latent_concept_prefix,
        latent_concept_refine=latent_concept_refine,
        latent_concept_refine_gate_init=latent_concept_refine_gate_init,
        latent_concept_memory_size=memory_size,
        max_vocab=max_vocab)
    self_teach_history_prior = reading_mastery_history_self_teach_prior(
        reading_mastery_history_from_payload(ckpt),
        enabled=(str(reading_objective_profile) == "mastery"
                 and float(study_self_teach_w) > 0.0))
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
        score_margin_w=study_score_margin_w, span_mask_frac=span_mask_frac,
        context_closure_split_frac=context_closure_split_frac,
        generation_eval_n=generation_eval_n,
        generation_prompt_tokens=generation_prompt_tokens,
        generation_max_new_tokens=generation_max_new_tokens,
        generation_temperature=generation_temperature,
        generation_top_k=generation_top_k)
    before_replay_bundle = (reading_eval_bundle(
        model, vocab, replay_records, device=device, eval_n=eval_n,
        seed=seed + 4093, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p, context_keep_p=context_keep_p,
        score_metric=study_score_metric, score_margin_w=study_score_margin_w,
        span_mask_frac=span_mask_frac,
        context_closure_split_frac=context_closure_split_frac,
        generation_eval_n=generation_eval_n,
        generation_prompt_tokens=generation_prompt_tokens,
        generation_max_new_tokens=generation_max_new_tokens,
        generation_temperature=generation_temperature,
        generation_top_k=generation_top_k)
        if replay_records else None)
    selection = {"enabled": False}
    if study_select_best:
        _model, _vocab, selection = fit_reading_concepts_select_best(
            model, vocab, study_records, steps=steps, batch=batch, lr=lr, seed=seed,
            device=device, log_every=log_every, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, feature_dropout=feature_dropout,
            lm_w=lm_w,
            continuation_repair_w=continuation_repair_w,
            continuation_repair_steps=continuation_repair_steps,
            continuation_repair_prompt_frac=continuation_repair_prompt_frac,
            continuation_repair_temperature=continuation_repair_temperature,
            continuation_repair_top_k=continuation_repair_top_k,
            repetition_unlikelihood_w=repetition_unlikelihood_w,
            repetition_unlikelihood_window=repetition_unlikelihood_window,
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
            consolidation_w=consolidation_w,
            consolidation_temperature=consolidation_temperature,
            consolidation_balance_w=consolidation_balance_w,
            consolidation_anchor_w=consolidation_anchor_w,
            consolidation_fer_w=consolidation_fer_w,
            discovery_w=discovery_w,
            discovery_curiosity_w=discovery_curiosity_w,
            discovery_graph_w=discovery_graph_w,
            discovery_cycle_w=discovery_cycle_w,
            discovery_bridge_w=discovery_bridge_w,
            discovery_fer_w=discovery_fer_w,
            reanalysis_w=reanalysis_w,
            reanalysis_graph_w=reanalysis_graph_w,
            reanalysis_cycle_w=reanalysis_cycle_w,
            reanalysis_bridge_w=reanalysis_bridge_w,
            reanalysis_fer_w=reanalysis_fer_w,
            gap_w=gap_w,
            gap_temperature=gap_temperature,
            gap_self_loop_w=gap_self_loop_w,
            gap_transitive_steps=gap_transitive_steps,
            gap_transitive_w=gap_transitive_w,
            gap_target_power=gap_target_power,
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
            span_completion_w=span_completion_w,
            span_mask_frac=span_mask_frac,
            span_completion_temperature=span_completion_temperature,
            context_closure_w=context_closure_w,
            context_closure_split_frac=context_closure_split_frac,
            context_closure_temperature=context_closure_temperature,
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
            score_target=study_score_target,
            signal_regression_tolerance=study_signal_regression_tolerance,
            generation_eval_n=generation_eval_n,
            generation_prompt_tokens=generation_prompt_tokens,
            generation_max_new_tokens=generation_max_new_tokens,
            generation_temperature=generation_temperature,
            generation_top_k=generation_top_k,
            insight_accept_w=study_insight_accept_w,
            insight_min_delta=study_insight_min_delta,
            representation_accept_w=study_representation_accept_w,
            representation_min_delta=study_representation_min_delta,
            study_self_teach_w=study_self_teach_w,
            self_teach_history_prior=self_teach_history_prior,
            self_teach_history_prior_w=self_teach_history_prior_w,
            replay_records=replay_records,
            replay_teacher_model=replay_teacher_model,
            replay_teacher_vocab=replay_teacher_vocab,
            replay_w=replay_w, replay_batch=replay_batch,
            replay_retention_w=replay_retention_w, rounds=study_rounds,
            before_bundle=before_bundle, score_records=records,
            source_balance_w=source_balance_w)
    else:
        self_teach_plan, self_teach_base_weights, train_weights = (
            reading_self_teach_weight_maps(
                before_bundle["score_components"], budget=study_self_teach_w,
                history_prior=self_teach_history_prior,
                history_prior_w=self_teach_history_prior_w,
                lm_w=lm_w,
                continuation_repair_w=continuation_repair_w,
                repetition_unlikelihood_w=repetition_unlikelihood_w,
                factorization_w=factorization_w,
                fer_w=fer_w,
                discovery_w=discovery_w,
                gap_w=gap_w,
                bridge_w=bridge_w,
                context_target_w=context_target_w,
                span_completion_w=span_completion_w,
                context_closure_w=context_closure_w,
                sequence_w=sequence_w,
                neighborhood_w=neighborhood_w,
                transition_w=transition_w,
                cluster_w=cluster_w))
        fit_reading_concepts(
            model, vocab, study_records, steps=steps, batch=batch, lr=lr, seed=seed,
            device=device, log_every=log_every, token_drop_p=token_drop_p,
            token_replace_p=token_replace_p, feature_dropout=feature_dropout,
            lm_w=train_weights["lm_w"],
            continuation_repair_w=train_weights["continuation_repair_w"],
            continuation_repair_steps=continuation_repair_steps,
            continuation_repair_prompt_frac=continuation_repair_prompt_frac,
            continuation_repair_temperature=continuation_repair_temperature,
            continuation_repair_top_k=continuation_repair_top_k,
            repetition_unlikelihood_w=train_weights["repetition_unlikelihood_w"],
            repetition_unlikelihood_window=repetition_unlikelihood_window,
            invariance_w=invariance_w, variance_w=variance_w,
            covariance_w=covariance_w, variance_target=variance_target,
            factorization_w=train_weights["factorization_w"],
            factorization_variance=factorization_variance,
            factorization_margin=factorization_margin,
            factorization_covariance_w=factorization_covariance_w,
            fer_w=train_weights["fer_w"],
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w,
            fer_balance_w=fer_balance_w,
            memory_w=memory_w,
            memory_size=memory_size,
            memory_temperature=memory_temperature,
            memory_momentum=memory_momentum,
            memory_balance_w=memory_balance_w,
            consolidation_w=consolidation_w,
            consolidation_temperature=consolidation_temperature,
            consolidation_balance_w=consolidation_balance_w,
            consolidation_anchor_w=consolidation_anchor_w,
            consolidation_fer_w=consolidation_fer_w,
            discovery_w=train_weights["discovery_w"],
            discovery_curiosity_w=discovery_curiosity_w,
            discovery_graph_w=discovery_graph_w,
            discovery_cycle_w=discovery_cycle_w,
            discovery_bridge_w=discovery_bridge_w,
            discovery_fer_w=discovery_fer_w,
            reanalysis_w=reanalysis_w,
            reanalysis_graph_w=reanalysis_graph_w,
            reanalysis_cycle_w=reanalysis_cycle_w,
            reanalysis_bridge_w=reanalysis_bridge_w,
            reanalysis_fer_w=reanalysis_fer_w,
            gap_w=train_weights["gap_w"],
            gap_temperature=gap_temperature,
            gap_self_loop_w=gap_self_loop_w,
            gap_transitive_steps=gap_transitive_steps,
            gap_transitive_w=gap_transitive_w,
            gap_target_power=gap_target_power,
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
            bridge_w=train_weights["bridge_w"],
            context_target_w=train_weights["context_target_w"],
            context_keep_p=context_keep_p,
            context_target_temperature=context_target_temperature,
            span_completion_w=train_weights["span_completion_w"],
            span_mask_frac=span_mask_frac,
            span_completion_temperature=span_completion_temperature,
            context_closure_w=train_weights["context_closure_w"],
            context_closure_split_frac=context_closure_split_frac,
            context_closure_temperature=context_closure_temperature,
            sequence_w=train_weights["sequence_w"], sequence_batch=sequence_batch,
            sequence_temperature=sequence_temperature,
            neighborhood_w=train_weights["neighborhood_w"],
            neighborhood_batch=neighborhood_batch,
            neighborhood_probe_n=neighborhood_probe_n,
            neighborhood_refresh_steps=neighborhood_refresh_steps,
            neighborhood_temperature=neighborhood_temperature,
            neighborhood_margin=neighborhood_margin,
            transition_w=train_weights["transition_w"],
            transition_batch=transition_batch,
            transition_temperature=transition_temperature,
            transition_margin=transition_margin,
            cluster_w=train_weights["cluster_w"],
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
            replay_w=replay_w, replay_batch=replay_batch,
            source_balance_w=source_balance_w)
        if self_teach_plan is not None:
            model.reading_train_metrics = dict(
                getattr(model, "reading_train_metrics", {})) | {
                    "self_teach_w": float(study_self_teach_w),
                    "self_teach_plan": self_teach_plan,
                    "self_teach_base_weights": self_teach_base_weights,
                    "self_teach_effective_weights": train_weights,
                }
            selection = {
                "enabled": False,
                "self_teach_w": float(study_self_teach_w),
                "replay_w": float(replay_w),
                "replay_retention_w": float(replay_retention_w),
                "self_teach_reports": [self_teach_plan | {"round": 1}],
                "self_teach_plan": self_teach_plan,
            }
    after_bundle = reading_eval_bundle(
        model, vocab, records, device=device, eval_n=eval_n, seed=seed,
        token_drop_p=token_drop_p, token_replace_p=token_replace_p,
        context_keep_p=context_keep_p, score_metric=study_score_metric,
        score_margin_w=study_score_margin_w, span_mask_frac=span_mask_frac,
        context_closure_split_frac=context_closure_split_frac,
        generation_eval_n=generation_eval_n,
        generation_prompt_tokens=generation_prompt_tokens,
        generation_max_new_tokens=generation_max_new_tokens,
        generation_temperature=generation_temperature,
        generation_top_k=generation_top_k)
    after_replay_bundle = (reading_eval_bundle(
        model, vocab, replay_records, device=device, eval_n=eval_n,
        seed=seed + 4093, token_drop_p=token_drop_p,
        token_replace_p=token_replace_p, context_keep_p=context_keep_p,
        score_metric=study_score_metric, score_margin_w=study_score_margin_w,
        span_mask_frac=span_mask_frac,
        context_closure_split_frac=context_closure_split_frac,
        generation_eval_n=generation_eval_n,
        generation_prompt_tokens=generation_prompt_tokens,
        generation_max_new_tokens=generation_max_new_tokens,
        generation_temperature=generation_temperature,
        generation_top_k=generation_top_k)
        if replay_records else None)
    before_lm = before_bundle["lm"]
    after_lm = after_bundle["lm"]
    before = before_bundle["view"]
    after = after_bundle["view"]
    before_context = before_bundle["context_target"]
    after_context = after_bundle["context_target"]
    before_span = before_bundle["span_completion"]
    after_span = after_bundle["span_completion"]
    before_closure = before_bundle["context_closure"]
    after_closure = after_bundle["context_closure"]
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
              "replay_external_records": len(external_replay_records),
              "replay_bank_records": len(checkpoint_replay_records),
              "replay_bank_used": bool(checkpoint_replay_records),
              "replay_study_used": bool(replay_study_records),
              "replay_study_records": len(replay_study_records),
              "study_record_count": len(study_records),
              "study_train_records": sum(r.split == "train" for r in study_records),
              "steps": int(steps), "batch": int(batch), "lr": float(lr),
              "text_encoder_arch": ckpt.get("text_encoder_arch", "transformer"),
              "text_encoder_layers": int(ckpt.get("text_encoder_layers", 1)),
              "latent_concept_slots": int(getattr(model, "latent_concept_slots", 0)),
              "latent_concept_layers": int(getattr(model, "latent_concept_layers", 1)),
              "latent_concept_topk": int(getattr(model, "latent_concept_topk", 0)),
              "latent_concept_prefix": bool(
                  getattr(model, "latent_concept_prefix", False)),
              "latent_concept_refine": bool(
                  getattr(model, "latent_concept_refine", False)),
              "latent_concept_refine_gate_init": float(
                  getattr(model, "latent_concept_refine_gate_init", -2.0)),
              "token_drop_p": float(token_drop_p),
              "token_replace_p": float(token_replace_p),
              "feature_dropout": float(feature_dropout),
              "lm_w": float(lm_w),
              "continuation_repair_w": float(continuation_repair_w),
              "continuation_repair_steps": int(continuation_repair_steps),
              "continuation_repair_prompt_frac": float(
                  continuation_repair_prompt_frac),
              "continuation_repair_temperature": float(
                  continuation_repair_temperature),
              "continuation_repair_top_k": int(continuation_repair_top_k),
              "repetition_unlikelihood_w": float(repetition_unlikelihood_w),
              "repetition_unlikelihood_window": int(
                  repetition_unlikelihood_window),
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
              "consolidation_w": float(consolidation_w),
              "consolidation_temperature": float(consolidation_temperature),
              "consolidation_balance_w": float(consolidation_balance_w),
              "consolidation_anchor_w": float(consolidation_anchor_w),
              "consolidation_fer_w": float(consolidation_fer_w),
              "discovery_w": float(discovery_w),
              "discovery_curiosity_w": float(discovery_curiosity_w),
              "discovery_graph_w": float(discovery_graph_w),
              "discovery_cycle_w": float(discovery_cycle_w),
              "discovery_bridge_w": float(discovery_bridge_w),
              "discovery_fer_w": float(discovery_fer_w),
              "reanalysis_w": float(reanalysis_w),
              "reanalysis_graph_w": float(reanalysis_graph_w),
              "reanalysis_cycle_w": float(reanalysis_cycle_w),
              "reanalysis_bridge_w": float(reanalysis_bridge_w),
              "reanalysis_fer_w": float(reanalysis_fer_w),
              "gap_w": float(gap_w),
              "gap_temperature": float(gap_temperature),
              "gap_self_loop_w": float(gap_self_loop_w),
              "gap_transitive_steps": int(gap_transitive_steps),
              "gap_transitive_w": float(gap_transitive_w),
              "gap_target_power": float(gap_target_power),
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
              "span_completion_w": float(span_completion_w),
              "span_mask_frac": float(span_mask_frac),
              "span_completion_temperature": float(span_completion_temperature),
              "context_closure_w": float(context_closure_w),
              "context_closure_split_frac": float(context_closure_split_frac),
              "context_closure_temperature": float(context_closure_temperature),
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
              "study_score_target": float(study_score_target),
              "study_signal_regression_tolerance": float(
                  study_signal_regression_tolerance),
              "study_insight_accept_w": float(study_insight_accept_w),
              "study_insight_min_delta": float(study_insight_min_delta),
              "study_representation_accept_w": float(
                  study_representation_accept_w),
              "study_representation_min_delta": float(
                  study_representation_min_delta),
              "study_self_teach_w": float(study_self_teach_w),
              "generation_eval_n": int(generation_eval_n),
              "generation_prompt_tokens": int(generation_prompt_tokens),
              "generation_max_new_tokens": int(generation_max_new_tokens),
              "generation_temperature": float(generation_temperature),
              "generation_top_k": int(generation_top_k),
              "reading_objective_profile": str(reading_objective_profile),
              "reading_objective_profile_report": (
                  reading_objective_profile_report or {
                      "profile": str(reading_objective_profile),
                      "enabled": str(reading_objective_profile) != "manual",
                      "floors": {},
                      "updates": {},
                      "applied": False,
                  }),
              "reading_mastery_history_prior": self_teach_history_prior,
              "replay_w": float(replay_w),
              "replay_batch": int(replay_batch),
              "replay_retention_w": float(replay_retention_w),
              "reading_replay_bank": {},
              "train_records": sum(r.split == "train" for r in records),
              "eval_records": sum(r.split == "eval" for r in records),
              "replay_train_records": sum(r.split == "train" for r in replay_records),
              "replay_eval_records": sum(r.split == "eval" for r in replay_records),
              "old_vocab_size": old_vocab_size,
              "max_vocab": int(max_vocab or 0),
              "vocab_capped": bool(max_vocab and int(max_vocab) > 0),
              "source_balance_w": float(source_balance_w),
              "new_vocab_size": len(vocab),
              "new_tokens": max(0, len(vocab) - old_vocab_size),
              "before": before,
              "after": after,
              "before_context_target": before_context,
              "after_context_target": after_context,
              "before_lm": before_lm,
              "after_lm": after_lm,
              "before_generation": before_bundle["generation"],
              "after_generation": after_bundle["generation"],
              "before_span_completion": before_span,
              "after_span_completion": after_span,
              "before_context_closure": before_closure,
              "after_context_closure": after_closure,
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
              "representation_progress": reading_representation_progress_report(
                  before_bundle, after_bundle),
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
                  "lm_token_acc": (
                      after_lm.get("lm_token_acc", 0.0)
                      - before_lm.get("lm_token_acc", 0.0)),
                  "lm_loss": (
                      after_lm.get("lm_loss", 0.0)
                      - before_lm.get("lm_loss", 0.0)),
                  "generation_score": (
                      after_bundle["score_components"].get(
                          "generation_score", 0.0)
                      - before_bundle["score_components"].get(
                          "generation_score", 0.0)),
                  "generation_token_acc": (
                      after_bundle["score_components"].get(
                          "generation_token_acc", 0.0)
                      - before_bundle["score_components"].get(
                          "generation_token_acc", 0.0)),
                  "paired_view_acc": (
                      after["paired_view_acc"] - before["paired_view_acc"]),
                  "margin": after.get("margin", 0.0) - before.get("margin", 0.0),
                  "context_target_acc": (
                      after_context.get("context_target_acc", 0.0)
                      - before_context.get("context_target_acc", 0.0)),
                  "context_target_margin": (
                      after_context.get("margin", 0.0)
                      - before_context.get("margin", 0.0)),
                  "span_completion_acc": (
                      after_span.get("span_completion_acc", 0.0)
                      - before_span.get("span_completion_acc", 0.0)),
                  "span_completion_margin": (
                      after_span.get("margin", 0.0)
                      - before_span.get("margin", 0.0)),
                  "context_closure_acc": (
                      after_closure.get("context_closure_acc", 0.0)
                      - before_closure.get("context_closure_acc", 0.0)),
                  "context_closure_margin": (
                      after_closure.get("margin", 0.0)
                      - before_closure.get("margin", 0.0)),
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
    report["learning_event"] = reading_learning_event_report(report)
    report["reading_replay_bank"] = build_reading_replay_bank(
        unique_reading_records_by_id(records, replay_records),
        study_reports=getattr(model, "reading_study_reports", []),
        learning_event=report["learning_event"])
    report["reading_mastery_history"] = reading_mastery_history_with_entry(
        reading_mastery_history_from_payload(ckpt), report)
    report["reading_mastery_history_count"] = report[
        "reading_mastery_history"]["entry_count"]
    if out_checkpoint:
        os.makedirs(os.path.dirname(out_checkpoint) or ".", exist_ok=True)
        torch.save(checkpoint_payload(model, vocab, d, layers, heads, report),
                   out_checkpoint)
        report["out_checkpoint"] = out_checkpoint
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
    if print_report:
        print(json.dumps(report, indent=1), flush=True)
    return report


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(path, device=DEV):
    ckpt = _torch_load(path, device)
    vocab = vocab_from_itos(ckpt["vocab"])
    model = TextReadingLM(len(vocab), d=int(ckpt.get("d", 96)),
                          layers=int(ckpt.get("layers", 3)),
                          heads=int(ckpt.get("heads", 4)), pad=vocab.pad,
                          text_encoder_arch=ckpt.get(
                              "text_encoder_arch", "transformer"),
                          text_encoder_layers=int(
                              ckpt.get("text_encoder_layers", 1)),
                          latent_concept_slots=int(
                              ckpt.get("latent_concept_slots", 0)),
                          latent_concept_layers=int(
                              ckpt.get("latent_concept_layers", 1)),
                          latent_concept_topk=int(
                              ckpt.get("latent_concept_topk", 0)),
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
    model.eval()
    return model, vocab, ckpt


def checkpoint_payload(model, vocab, d, layers, heads, report):
    payload = {
        "state_dict": model.state_dict(), "vocab": vocab.itos,
        "latent_concept_slots": int(getattr(model, "latent_concept_slots", 0)),
        "latent_concept_layers": int(getattr(model, "latent_concept_layers", 1)),
        "latent_concept_topk": int(getattr(model, "latent_concept_topk", 0)),
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
        "d": d, "layers": layers, "heads": heads,
        "report": report,
    }
    if isinstance(report, dict) and report.get("reading_replay_bank") is not None:
        payload["reading_replay_bank"] = report["reading_replay_bank"]
    if isinstance(report, dict) and report.get("reading_mastery_history") is not None:
        payload["reading_mastery_history"] = report["reading_mastery_history"]
    return payload



def selftest():
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
    default_source_rec = normalize_reading_record(
        {"id": "default-source", "split": "train", "text": "source stamped row"},
        default_source="source-file.jsonl")
    assert default_source_rec.meta["source"] == "source-file.jsonl"
    explicit_source_rec = normalize_reading_record(
        {"id": "explicit-source", "split": "train", "text": "source kept row",
         "meta": {"source": "manifest-source"}},
        default_source="source-file.jsonl")
    assert explicit_source_rec.meta["source"] == "manifest-source"
    cap_records = [
        ReadingRecord(
            "vocab-cap", "train",
            tuple(["common"] * 6 + [f"rare{i}" for i in range(20)])),
    ]
    capped_vocab = build_reading_vocab(cap_records, max_size=10)
    assert len(capped_vocab) <= 10
    assert "common" in capped_vocab.stoi
    assert capped_vocab.enc(["rare19"])[0] == capped_vocab.unk
    base_vocab = build_reading_vocab([
        ReadingRecord("vocab-base", "train", ("base_token",)),
    ], max_size=0)
    expanded_vocab = build_reading_vocab(
        cap_records, base_vocab=base_vocab, max_size=len(base_vocab) + 1)
    assert "base_token" in expanded_vocab.stoi
    assert "common" in expanded_vocab.stoi
    assert expanded_vocab.enc(["rare19"])[0] == expanded_vocab.unk
    reading_model = TextReadingLM(
        len(reading_vocab), d=32, layers=1, heads=4, pad=reading_vocab.pad,
        latent_concept_slots=2, latent_concept_memory_size=8).to("cpu")
    memoryless_reading_model = TextReadingLM(
        len(reading_vocab), d=32, layers=1, heads=4, pad=reading_vocab.pad,
        latent_concept_slots=2, latent_concept_memory_size=0).to("cpu")
    assert resolve_reading_study_strategy("auto", reading_model) == "discovery"
    assert (resolve_reading_study_strategy("auto", memoryless_reading_model)
            == "closure")
    reading_txt = pack_reading(reading_records[:2], reading_vocab, "cpu")
    sparse_reading_model = TextReadingLM(
        len(reading_vocab), d=32, layers=1, heads=4, pad=reading_vocab.pad,
        latent_concept_slots=4, latent_concept_topk=2).to("cpu")
    sparse_slots = sparse_reading_model.latent_concept_states(reading_txt)
    assert int(sparse_slots.norm(dim=-1).gt(0.0).sum(dim=1).max().item()) <= 2
    sparse_payload = checkpoint_payload(
        sparse_reading_model, reading_vocab, d=32, layers=1, heads=4,
        report={"experiment": "selftest_sparse_topk"})
    assert sparse_payload["latent_concept_topk"] == 2
    repeat_loss, repeat_metrics = reading_repetition_unlikelihood_loss(
        reading_model, reading_txt, pad=reading_vocab.pad, window=4)
    assert torch.isfinite(repeat_loss)
    assert repeat_metrics["enabled"] is True
    assert repeat_metrics["candidates"] > 0
    repair_loss, repair_metrics = reading_continuation_repair_loss(
        reading_model, reading_txt, pad=reading_vocab.pad, repair_steps=2,
        prompt_frac=0.5)
    assert torch.isfinite(repair_loss)
    assert repair_metrics["enabled"] is True
    assert repair_metrics["generated_tokens"] > 0
    assert torch.isfinite(reading_latent_view_loss(
        reading_model, reading_txt, reading_vocab.pad, reading_vocab.unk,
        token_drop_p=0.1, token_replace_p=0.0))
    assert torch.isfinite(reading_latent_factorization_loss(
        reading_model, reading_txt, feature_dropout=0.1))
    fer_loss, fer_metrics = reading_latent_fer_loss(
        reading_model, reading_txt, feature_dropout=0.1)
    assert torch.isfinite(fer_loss)
    assert torch.isfinite(fer_metrics["fer_score"])
    fer_selected, fer_study_report = reading_latent_fer_records(
        reading_model, reading_vocab, reading_records, device="cpu", n=0)
    assert fer_selected and fer_study_report["skipped"] is False
    assert math.isfinite(fer_study_report["mean_fer_score"])
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
    seq_selected, seq_study_report = reading_sequence_surprise_records(
        reading_model, reading_vocab, reading_records, device="cpu", n=0,
        token_drop_p=0.1, token_replace_p=0.0)
    assert seq_selected and seq_study_report["n_pairs"] == 2
    assert math.isfinite(seq_study_report["mean_sequence_surprise"])
    assert torch.isfinite(reading_latent_memory_loss(
        reading_model, reading_txt, feature_dropout=0.1))
    gap_loss, gap_metrics = reading_latent_gap_loss(
        reading_model, reading_txt, feature_dropout=0.1,
        transitive_steps=2, transitive_w=0.1)
    assert torch.isfinite(gap_loss)
    assert gap_metrics["skipped"] is False
    assert gap_metrics["memory_active"] > 0
    assert torch.isfinite(gap_metrics["gap_kl"])
    consolidation_loss, consolidation_metrics = (
        reading_latent_memory_consolidation_loss(
            reading_model, reading_txt, reading_vocab.pad, reading_vocab.unk,
            token_drop_p=0.1, token_replace_p=0.0, feature_dropout=0.1,
            anchor_w=1.0, fer_w=0.1))
    assert torch.isfinite(consolidation_loss)
    assert consolidation_metrics["skipped"] is False
    assert consolidation_metrics["memory_active"] > 0
    assert torch.isfinite(consolidation_metrics["anchor_loss"])
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
    span_completion_loss, span_completion_metrics = reading_span_completion_loss(
        reading_model, reading_txt, reading_vocab.pad, span_frac=0.25,
        feature_dropout=0.1, temperature=0.1)
    assert torch.isfinite(span_completion_loss)
    assert span_completion_metrics["skipped"] is False
    assert float(span_completion_metrics["hidden_token_rate"]) > 0.0
    closure_loss, closure_metrics = reading_context_closure_loss(
        reading_model, reading_txt, reading_vocab.pad, split_frac=0.5,
        feature_dropout=0.1, temperature=0.1)
    assert torch.isfinite(closure_loss)
    assert closure_metrics["skipped"] is False
    assert float(closure_metrics["prefix_token_rate"]) > 0.0
    assert float(closure_metrics["suffix_token_rate"]) > 0.0
    closure_records, closure_report = reading_context_closure_surprise_records(
        reading_model, reading_vocab, reading_records, device="cpu", n=0,
        split_frac=0.5, temperature=0.1)
    assert closure_records and closure_report["skipped"] is False
    assert math.isfinite(closure_report["mean_closure_surprise"])
    assert math.isfinite(closure_report["mean_closure_cosine"])
    discovery_loss, discovery_metrics = reading_latent_discovery_loss(
        reading_model, reading_txt, reading_vocab.pad, feature_dropout=0.1,
        context_keep_p=0.5, graph_transitive_steps=2,
        graph_transitive_w=0.1, cycle_transitive_steps=2,
        cycle_transitive_w=0.1, fer_w=0.1)
    assert torch.isfinite(discovery_loss)
    assert discovery_metrics["skipped"] is False
    assert discovery_metrics["memory_active"] > 0
    assert torch.isfinite(discovery_metrics["graph_loss"])
    assert torch.isfinite(discovery_metrics["insight_loss"])
    assert torch.isfinite(discovery_metrics["bridge_loss"])
    graph_records, graph_report = reading_latent_graph_prediction_records(
        reading_model, reading_vocab, reading_records, device="cpu", n=0,
        context_keep_p=0.5, transitive_steps=2, transitive_w=0.1)
    assert graph_records and graph_report["skipped"] is False
    cycle_records, cycle_report = reading_latent_graph_cycle_records(
        reading_model, reading_vocab, reading_records, device="cpu", n=0,
        context_keep_p=0.5, transitive_steps=2, transitive_w=0.1)
    assert cycle_records and cycle_report["skipped"] is False
    gap_records, gap_report = reading_latent_gap_records(
        reading_model, reading_vocab, reading_records, device="cpu", n=0,
        transitive_steps=2, transitive_w=0.1)
    assert gap_records and gap_report["skipped"] is False
    assert math.isfinite(gap_report["mean_gap_score"])
    discovery_records, discovery_report = reading_latent_discovery_records(
        reading_model, reading_vocab, reading_records, device="cpu", n=0,
        context_keep_p=0.5, curiosity_transitive_steps=2,
        curiosity_transitive_w=0.1, graph_transitive_steps=2,
        graph_transitive_w=0.1, cycle_transitive_steps=2,
        cycle_transitive_w=0.1)
    assert discovery_records and discovery_report["skipped"] is False
    assert math.isfinite(discovery_report["mean_score"])
    assert "mean_fer_score" in discovery_report
    assert "mean_slot_disorder" in discovery_report
    assert "mean_gap_score" in discovery_report
    assert "mean_insight_score" in discovery_report
    assert "mean_bridge_score" in discovery_report
    assert "mean_closure_surprise" in discovery_report
    assert "mean_sequence_surprise" in discovery_report
    assert discovery_report["n_sequence_pairs"] == 2
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
    span_bundle = reading_eval_bundle(
        reading_model, reading_vocab, reading_records, device="cpu", eval_n=0,
        score_metric="span")
    assert span_bundle["score_components"]["metric"] == "span"
    assert span_bundle["score_components"]["span_skipped"] is False
    assert math.isfinite(span_bundle["score_components"]["span_score"])
    assert "span_completion" in span_bundle
    closure_bundle = reading_eval_bundle(
        reading_model, reading_vocab, reading_records, device="cpu", eval_n=0,
        score_metric="closure")
    assert closure_bundle["score_components"]["metric"] == "closure"
    assert closure_bundle["score_components"]["closure_skipped"] is False
    assert math.isfinite(closure_bundle["score_components"]["context_closure_score"])
    assert "context_closure" in closure_bundle
    mastery_bundle = reading_eval_bundle(
        reading_model, reading_vocab, reading_records, device="cpu", eval_n=0,
        score_metric="mastery", generation_eval_n=2,
        generation_prompt_tokens=2, generation_max_new_tokens=2)
    assert mastery_bundle["score_components"]["metric"] == "mastery"
    assert ("mastery_score" in mastery_bundle["score_components"]
            and "signal_coverage" in mastery_bundle["score_components"]
            and "language_score" in mastery_bundle["score_components"]
            and "generation_score" in mastery_bundle["score_components"]
            and "span_score" in mastery_bundle["score_components"]
            and "context_closure_score" in mastery_bundle["score_components"]
            and "sequence_score" in mastery_bundle["score_components"]
            and "bridge_score" in mastery_bundle["score_components"])
    assert "lm" in mastery_bundle
    assert "generation" in mastery_bundle
    assert mastery_bundle["score_components"]["language_skipped"] is False
    assert mastery_bundle["score_components"]["generation_skipped"] is False
    assert mastery_bundle["generation"]["enabled"] is True
    assert mastery_bundle["generation"]["samples"]
    assert reading_generation_gate(
        {"enabled": True, "skipped": False, "n_records": 1,
         "token_acc": 0.5, "exact": 0.0, "unique_generation_count": 1,
         "all_generations_identical": False,
         "samples": [{"generated": ["next"]}]})["passed"] is True
    assert reading_generation_gate(
        {"enabled": True, "skipped": False, "n_records": 2,
         "token_acc": 0.5, "exact": 0.0, "unique_generation_count": 1,
         "all_generations_identical": True,
         "samples": [{"generated": ["loop"]}, {"generated": ["loop"]}]})[
             "reason"] == "collapsed_generation"
    assert math.isfinite(mastery_bundle["score_components"]["lm_loss"])
    assert math.isfinite(mastery_bundle["score_components"]["generation_score"])
    assert mastery_bundle["score_components"]["span_skipped"] is False
    assert mastery_bundle["score_components"]["closure_skipped"] is False
    assert mastery_bundle["score_components"]["bridge_skipped"] is False
    assert (mastery_bundle["score_components"]["score"]
            == mastery_bundle["score_components"]["mastery_score"])
    synthetic_scores = {
        f"{name}_skipped": False for name in READING_DISCOVERY_SIGNALS}
    for name, score_key in READING_SELF_TEACH_SCORE_KEYS.items():
        synthetic_scores[score_key] = 1.0
    synthetic_scores["context_score"] = 0.0
    synthetic_scores["context_closure_score"] = 0.5
    synthetic_scores["bridge_score"] = 0.25
    self_teach_plan = reading_self_teach_weight_plan(
        synthetic_scores, budget=0.12)
    assert self_teach_plan["enabled"] is True
    assert self_teach_plan["top_signal"] == "context"
    assert self_teach_plan["weight_extras"]["context_target_w"] > 0.0
    assert self_teach_plan["weight_extras"]["context_closure_w"] > 0.0
    assert self_teach_plan["weight_extras"]["bridge_w"] > 0.0
    assert math.isclose(sum(self_teach_plan["weight_extras"].values()),
                        0.12, rel_tol=1e-6, abs_tol=1e-6)
    language_scores = {
        f"{name}_skipped": False for name in READING_DISCOVERY_SIGNALS}
    for name, score_key in READING_SELF_TEACH_SCORE_KEYS.items():
        language_scores[score_key] = 1.0
    language_scores["language_score"] = 0.0
    language_plan = reading_self_teach_weight_plan(
        language_scores, budget=0.07)
    assert language_plan["top_signal"] == "language"
    assert language_plan["weight_extras"]["lm_w"] > 0.0
    assert math.isclose(sum(language_plan["weight_extras"].values()),
                        0.07, rel_tol=1e-6, abs_tol=1e-6)
    generation_scores = {
        f"{name}_skipped": False for name in READING_DISCOVERY_SIGNALS}
    for name, score_key in READING_SELF_TEACH_SCORE_KEYS.items():
        generation_scores[score_key] = 1.0
    generation_scores["generation_score"] = 0.0
    generation_plan = reading_self_teach_weight_plan(
        generation_scores, budget=0.09)
    assert generation_plan["top_signal"] == "generation"
    assert math.isclose(generation_plan["weight_extras"]["lm_w"], 0.03,
                        rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(
        generation_plan["weight_extras"]["continuation_repair_w"], 0.03,
        rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(
        generation_plan["weight_extras"]["repetition_unlikelihood_w"], 0.03,
        rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(sum(generation_plan["weight_extras"].values()),
                        0.09, rel_tol=1e-6, abs_tol=1e-6)
    regression_report = signal_regression_report(
        {"generation_score": 0.30, "generation_skipped": False},
        {"generation_score": 0.25, "generation_skipped": False},
        READING_SELF_TEACH_SCORE_KEYS,
        tolerance=0.02,
        signals=("generation",))
    assert regression_report["allowed"] is False
    assert regression_report["regressions"]["generation"] > 0.02
    tolerated_report = signal_regression_report(
        {"generation_score": 0.30, "generation_skipped": False},
        {"generation_score": 0.29, "generation_skipped": False},
        READING_SELF_TEACH_SCORE_KEYS,
        tolerance=0.02,
        signals=("generation",))
    assert tolerated_report["allowed"] is True
    nonfinite_report = signal_regression_report(
        {"generation_score": 0.30, "generation_skipped": False},
        {"generation_score": float("nan"), "generation_skipped": False},
        READING_SELF_TEACH_SCORE_KEYS,
        tolerance=0.02,
        signals=("generation",))
    assert nonfinite_report["allowed"] is False
    assert "generation" in nonfinite_report["nonfinite_signals"]
    prior_scores = {
        f"{name}_skipped": False for name in READING_DISCOVERY_SIGNALS}
    for name, score_key in READING_SELF_TEACH_SCORE_KEYS.items():
        prior_scores[score_key] = 1.0
    prior_scores["sequence_score"] = 0.0
    history_prior = reading_mastery_history_self_teach_prior({
        "entries": [{
            "after_score_components": prior_scores,
            "self_teach_top_signal": "sequence",
        }]
    })
    assert history_prior["enabled"] is True
    assert history_prior["top_signal"] == "sequence"
    current_scores = dict(prior_scores)
    current_scores["sequence_score"] = 1.0
    history_self_teach_plan = reading_self_teach_weight_plan(
        current_scores, budget=0.09, history_prior=history_prior,
        history_prior_w=1.0)
    assert history_self_teach_plan["top_signal"] == "sequence"
    assert history_self_teach_plan["history_prior_entry_count"] == 1
    assert history_self_teach_plan["weight_extras"]["sequence_w"] > 0.0
    assert math.isclose(sum(history_self_teach_plan["weight_extras"].values()),
                        0.09, rel_tol=1e-6, abs_tol=1e-6)
    representation_prior = reading_mastery_history_self_teach_prior({
        "entries": [{
            "after_score_components": current_scores,
            "representation_progress": {
                "enabled": True,
                "signal_after": {"sequence": 0.0, "fer": 1.0},
                "signal_deltas": {"sequence": -0.25, "fer": 0.0},
                "organization_score_delta": -0.1,
            },
        }]
    })
    assert representation_prior["signal_deficits"]["sequence"] > 0.0
    event_prior = reading_mastery_history_self_teach_prior({
        "entries": [{
            "learning_event": {
                "triggered": True,
                "kind": "representation_reorganization",
                "top_signal": "cluster",
                "event_score": 0.7,
            },
        }]
    })
    assert event_prior["signal_deficits"]["cluster"] > 0.0
    concept_prior = reading_mastery_history_self_teach_prior({
        "entries": [{
            "after_score_components": current_scores,
            "self_teach_top_signal": "bridge",
            "max_concept_insight_delta": 0.8,
            "selected_by_insight": True,
            "accepted_update": True,
        }]
    })
    assert concept_prior["concept_connection_signal"] > 0.0
    assert concept_prior["concept_insight_prior"][
        "selected_by_insight_count"] == 1
    concept_self_teach_plan = reading_self_teach_weight_plan(
        current_scores, budget=0.06, history_prior=concept_prior,
        history_prior_w=1.0)
    assert concept_self_teach_plan["top_signal"] == "connection"
    assert concept_self_teach_plan["history_concept_connection_signal"] > 0.0
    assert concept_self_teach_plan["weight_extras"]["bridge_w"] > 0.0
    assert concept_self_teach_plan["weight_extras"]["discovery_w"] > 0.0
    assert concept_self_teach_plan["weight_extras"]["gap_w"] > 0.0
    assert math.isclose(sum(concept_self_teach_plan["weight_extras"].values()),
                        0.06, rel_tol=1e-6, abs_tol=1e-6)
    concept_insight = reading_concept_insight_report(
        {"floor_score": 0.2, "signal_coverage": 0.5, "bridge_score": 0.2},
        {"floor_score": 0.3, "signal_coverage": 0.75, "bridge_score": 0.4},
        {"skipped": False, "bridge_score_reduction": 0.2,
         "bridge_connectivity_gain": 0.1},
        enabled=True)
    assert concept_insight["allowed"] is True
    assert concept_insight["delta"] > 0.0
    regressed_concept_insight = reading_concept_insight_report(
        {"floor_score": 0.8, "balanced_score": 0.8, "signal_coverage": 1.0},
        {"floor_score": 0.1, "balanced_score": 0.2, "signal_coverage": 0.5},
        {"skipped": False, "bridge_score_reduction": -0.2,
         "bridge_connectivity_gain": -0.2},
        enabled=True)
    assert regressed_concept_insight["allowed"] is False
    mastery_kwargs = reading_objective_profile_kwargs(
        "mastery",
        lm_w=0.0, factorization_w=0.0, fer_w=0.0, memory_w=0.0,
        discovery_w=0.0, reanalysis_w=0.0, gap_w=0.0,
        association_w=0.0, composition_w=0.0, graph_predict_w=0.0,
        graph_cycle_w=0.0, bridge_w=0.0, context_target_w=0.0,
        span_completion_w=0.0, context_closure_w=0.0,
        sequence_w=0.0, neighborhood_w=0.0, transition_w=0.0,
        cluster_w=0.0, continuation_repair_w=0.0,
        continuation_repair_steps=0, repetition_unlikelihood_w=0.0,
        repetition_unlikelihood_window=0, study_self_teach_w=0.0,
        study_rounds=1,
        study_score_patience=0, study_score_target=0.0,
        study_representation_accept_w=0.0,
        study_representation_min_delta=0.0,
        study_probe_n=0, study_hard_max=0, study_refresh_steps=0,
        neighborhood_probe_n=0, neighborhood_refresh_steps=0,
        cluster_probe_n=0, cluster_refresh_steps=0,
        generation_eval_n=0, passthrough="kept")
    assert mastery_kwargs["reading_objective_profile"] == "mastery"
    assert mastery_kwargs["reading_objective_profile_report"]["applied"] is True
    assert mastery_kwargs["lm_w"] == 1.0
    assert mastery_kwargs["graph_predict_w"] == 0.10
    assert mastery_kwargs["continuation_repair_w"] == 0.05
    assert mastery_kwargs["continuation_repair_steps"] == 4
    assert mastery_kwargs["repetition_unlikelihood_w"] == 0.05
    assert mastery_kwargs["repetition_unlikelihood_window"] == 32
    assert mastery_kwargs["study_self_teach_w"] == 0.05
    assert mastery_kwargs["study_rounds"] == 3
    assert mastery_kwargs["study_score_patience"] == 2
    assert mastery_kwargs["study_score_target"] == 0.85
    assert mastery_kwargs["study_representation_accept_w"] == 0.25
    assert mastery_kwargs["study_representation_min_delta"] == 0.01
    assert mastery_kwargs["study_probe_n"] == 256
    assert mastery_kwargs["study_hard_max"] == 64
    assert mastery_kwargs["study_refresh_steps"] == 100
    assert mastery_kwargs["neighborhood_probe_n"] == 256
    assert mastery_kwargs["neighborhood_refresh_steps"] == 100
    assert mastery_kwargs["cluster_probe_n"] == 256
    assert mastery_kwargs["cluster_refresh_steps"] == 100
    assert mastery_kwargs["generation_eval_n"] == 16
    assert (mastery_kwargs["reading_objective_profile_report"]
            ["operation_floors"]["study_probe_n"] == 256)
    assert mastery_kwargs["passthrough"] == "kept"
    assert (reading_self_teach_study_strategy(
        {"top_signal": "sequence"}, "auto", "discovery", reading_model)
        == "sequence")
    assert (reading_self_teach_study_strategy(
        {"top_signal": "bridge"}, "auto", "closure", object())
        == "closure")
    assert (reading_self_teach_study_strategy(
        {"top_signal": "context"}, "gap", "gap", reading_model)
        == "gap")
    manual_kwargs = reading_objective_profile_kwargs(
        "manual",
        lm_w=0.0, factorization_w=0.0, fer_w=0.0, memory_w=0.0,
        discovery_w=0.0, reanalysis_w=0.0, gap_w=0.0,
        association_w=0.0, composition_w=0.0, graph_predict_w=0.0,
        graph_cycle_w=0.0, bridge_w=0.0, context_target_w=0.0,
        span_completion_w=0.0, context_closure_w=0.0,
        sequence_w=0.0, neighborhood_w=0.0, transition_w=0.0,
        cluster_w=0.0, continuation_repair_w=0.0,
        continuation_repair_steps=0, repetition_unlikelihood_w=0.0,
        repetition_unlikelihood_window=0, study_self_teach_w=0.0,
        study_rounds=1,
        study_score_patience=0, study_score_target=0.0,
        study_representation_accept_w=0.0,
        study_representation_min_delta=0.0,
        study_probe_n=0, study_hard_max=0, study_refresh_steps=0,
        neighborhood_probe_n=0, neighborhood_refresh_steps=0,
        cluster_probe_n=0, cluster_refresh_steps=0,
        generation_eval_n=0)
    assert manual_kwargs["study_self_teach_w"] == 0.0
    assert manual_kwargs["lm_w"] == 0.0
    assert manual_kwargs["continuation_repair_w"] == 0.0
    assert manual_kwargs["continuation_repair_steps"] == 0
    assert manual_kwargs["repetition_unlikelihood_w"] == 0.0
    assert manual_kwargs["repetition_unlikelihood_window"] == 0
    assert manual_kwargs["study_rounds"] == 1
    assert manual_kwargs["study_score_target"] == 0.0
    assert manual_kwargs["study_representation_accept_w"] == 0.0
    assert manual_kwargs["study_representation_min_delta"] == 0.0
    assert manual_kwargs["study_probe_n"] == 0
    assert manual_kwargs["study_hard_max"] == 0
    assert manual_kwargs["study_refresh_steps"] == 0
    assert manual_kwargs["neighborhood_probe_n"] == 0
    assert manual_kwargs["cluster_probe_n"] == 0
    assert manual_kwargs["generation_eval_n"] == 0
    assert manual_kwargs["reading_objective_profile_report"]["applied"] is False
    checkpoint_kwargs = reading_checkpoint_profile_kwargs(
        "mastery", replay_records=reading_records,
        replay_w=0.0, replay_retention_w=0.0)
    assert checkpoint_kwargs["replay_w"] == 0.05
    assert checkpoint_kwargs["replay_retention_w"] == 0.25
    assert (checkpoint_kwargs["reading_checkpoint_profile_report"]["applied"]
            is True)
    checkpoint_manual = reading_checkpoint_profile_kwargs(
        "manual", replay_records=reading_records,
        replay_w=0.0, replay_retention_w=0.0)
    assert checkpoint_manual["replay_w"] == 0.0
    assert (checkpoint_manual["reading_checkpoint_profile_report"]["applied"]
            is False)
    fit_reading_concepts(
        reading_model, reading_vocab, reading_records, steps=3, batch=2, lr=1e-4,
        seed=5, device="cpu", log_every=1, token_drop_p=0.1,
        token_replace_p=0.0, study_strategy="discovery", study_probe_n=4,
        study_hard_max=2, study_refresh_steps=1, context_target_w=0.1,
        span_completion_w=0.1, span_mask_frac=0.25,
        context_closure_w=0.1, context_closure_split_frac=0.5,
        context_keep_p=0.5, memory_size=8, composition_w=0.1, graph_predict_w=0.1,
        graph_cycle_w=0.1, bridge_w=0.1, fer_w=0.1,
        consolidation_w=0.1, consolidation_fer_w=0.1,
        discovery_w=0.1, discovery_fer_w=0.1,
        reanalysis_w=0.1, reanalysis_fer_w=0.1,
        gap_w=0.1, lm_w=0.1,
        continuation_repair_w=0.1, continuation_repair_steps=2,
        repetition_unlikelihood_w=0.1, repetition_unlikelihood_window=4,
        sequence_w=0.1, sequence_batch=2, sequence_temperature=0.1,
        neighborhood_w=0.1, neighborhood_batch=2, neighborhood_probe_n=2,
        transition_w=0.1, transition_batch=2,
        cluster_w=0.1, cluster_batch=4, cluster_probe_n=4)
    assert reading_model.reading_train_metrics["memory_active"] > 0
    assert reading_model.reading_train_metrics["lm_w"] == 0.1
    assert math.isfinite(reading_model.reading_train_metrics["lm_loss"])
    assert reading_model.reading_train_metrics["continuation_repair_w"] == 0.1
    assert reading_model.reading_train_metrics["continuation_repair_enabled"] is True
    assert reading_model.reading_train_metrics["continuation_repair_tokens"] > 0
    assert reading_model.reading_train_metrics["repetition_unlikelihood_w"] == 0.1
    assert reading_model.reading_train_metrics[
        "repetition_unlikelihood_enabled"] is True
    assert reading_model.reading_train_metrics[
        "repetition_unlikelihood_candidates"] > 0
    assert reading_model.reading_train_metrics["consolidation_w"] == 0.1
    assert reading_model.reading_train_metrics["consolidation_fer_w"] == 0.1
    assert reading_model.reading_train_metrics["consolidation_skipped"] is False
    assert math.isfinite(reading_model.reading_train_metrics[
        "consolidation_anchor_loss"])
    assert (reading_model.reading_train_metrics["study_strategy_requested"]
            == "discovery")
    assert reading_model.reading_train_metrics["study_strategy"] == "discovery"
    assert reading_model.reading_train_metrics["graph_predict_w"] == 0.1
    assert reading_model.reading_train_metrics["graph_cycle_w"] == 0.1
    assert reading_model.reading_train_metrics["bridge_w"] == 0.1
    assert reading_model.reading_train_metrics["span_completion_w"] == 0.1
    assert reading_model.reading_train_metrics["span_completion_skipped"] is False
    assert math.isfinite(
        reading_model.reading_train_metrics["span_completion_loss"])
    assert (reading_model.reading_train_metrics[
        "span_completion_hidden_token_rate"] > 0.0)
    assert reading_model.reading_train_metrics["context_closure_w"] == 0.1
    assert reading_model.reading_train_metrics["context_closure_skipped"] is False
    assert math.isfinite(
        reading_model.reading_train_metrics["context_closure_loss"])
    assert (reading_model.reading_train_metrics[
        "context_closure_prefix_token_rate"] > 0.0)
    assert (reading_model.reading_train_metrics[
        "context_closure_suffix_token_rate"] > 0.0)
    assert reading_model.reading_train_metrics["discovery_w"] == 0.1
    assert reading_model.reading_train_metrics["discovery_fer_w"] == 0.1
    assert reading_model.reading_train_metrics["discovery_skipped"] is False
    assert math.isfinite(reading_model.reading_train_metrics["discovery_loss"])
    assert math.isfinite(
        reading_model.reading_train_metrics["discovery_graph_loss"])
    assert math.isfinite(
        reading_model.reading_train_metrics["discovery_insight_loss"])
    assert reading_model.reading_train_metrics["reanalysis_w"] == 0.1
    assert reading_model.reading_train_metrics["reanalysis_fer_w"] == 0.1
    assert reading_model.reading_train_metrics["reanalysis_skipped"] is False
    assert math.isfinite(reading_model.reading_train_metrics["reanalysis_loss"])
    assert math.isfinite(
        reading_model.reading_train_metrics["reanalysis_closure_loss"])
    assert reading_model.reading_train_metrics["gap_w"] == 0.1
    assert reading_model.reading_train_metrics["gap_skipped"] is False
    assert math.isfinite(reading_model.reading_train_metrics["gap_loss"])
    assert math.isfinite(reading_model.reading_train_metrics["gap_kl"])
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
    assert reading_model.reading_train_metrics["weight_update_changed"] is True
    assert (reading_model.reading_train_metrics[
        "weight_update_changed_tensor_count"] > 0)
    assert reading_model.reading_train_metrics[
        "weight_update_max_abs_delta"] > 0.0
    assert any(r.get("strategy") == "discovery"
               for r in reading_model.reading_study_reports)
    scored = [r.get("mean_score", 0.0) for r in reading_model.reading_study_reports
              if r.get("strategy") == "discovery" and not r.get("skipped")]
    assert scored and max(scored) > 0.0
    assert any("mean_reverse_kl" in r for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    assert any("mean_bridge_score" in r for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    assert any("mean_fer_score" in r
               for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    assert any("mean_sequence_surprise" in r
               for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    assert any("mean_closure_surprise" in r
               for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    assert any("pool_bridge_before" in r for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    assert any("mean_gap_score" in r for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    assert any("mean_insight_score" in r
               for r in reading_model.reading_study_reports
               if r.get("strategy") == "discovery")
    closure_study_model = TextReadingLM(
        len(reading_vocab), d=32, layers=1, heads=4, pad=reading_vocab.pad,
        latent_concept_slots=2, latent_concept_memory_size=0).to("cpu")
    fit_reading_concepts(
        closure_study_model, reading_vocab, reading_records, steps=1, batch=2,
        lr=1e-4, seed=6, device="cpu", log_every=1, token_drop_p=0.1,
        token_replace_p=0.0, study_strategy="closure", study_probe_n=4,
        study_hard_max=2, study_refresh_steps=1, context_closure_w=0.1,
        context_closure_split_frac=0.5, context_target_w=0.0, sequence_w=0.0)
    assert closure_study_model.reading_train_metrics["study_strategy"] == "closure"
    assert closure_study_model.reading_train_metrics["context_closure_w"] == 0.1
    assert any(r.get("strategy") == "closure"
               and "mean_closure_surprise" in r
               for r in closure_study_model.reading_study_reports)
    fit_reading_concepts(
        reading_model, reading_vocab, reading_records, steps=1, batch=2, lr=1e-4,
        seed=7, device="cpu", log_every=1, token_drop_p=0.1,
        token_replace_p=0.0, study_strategy="gap", study_probe_n=4,
        study_hard_max=2, study_refresh_steps=1, context_target_w=0.0,
        memory_size=8, gap_w=0.1, sequence_w=0.0, transition_w=0.0)
    assert reading_model.reading_train_metrics["study_strategy"] == "gap"
    assert any(r.get("strategy") == "gap" and "mean_gap_score" in r
               for r in reading_model.reading_study_reports)
    insight = reading_model.reading_train_metrics["study_pool_insight"]
    assert "bridge_score_reduction" in insight
    assert math.isfinite(insight["bridge_connectivity_gain"])
    assert (reading_model.reading_train_metrics[
        "study_pool_bridge_before"].get("graph_source") == "snapshot")
    assert "study_pool_bridge_after_current" in reading_model.reading_train_metrics
    score_decision = reading_round_selection_decision(
        0.2, 0.0, insight_delta=-1.0, insight_allowed=False,
        bridge_insight_gate=True, insight_accept_w=1.0)
    assert score_decision["selected"] is False
    score_decision = reading_round_selection_decision(
        0.2, 0.0, insight_delta=-1.0, insight_allowed=True,
        bridge_insight_gate=True, insight_accept_w=1.0)
    assert score_decision["selected_by_score"] is True
    non_bridge_score_decision = reading_round_selection_decision(
        0.2, 0.0, insight_delta=-1.0, insight_allowed=True,
        bridge_insight_gate=False, insight_accept_w=1.0)
    assert non_bridge_score_decision["selected_by_score"] is True
    insight_decision = reading_round_selection_decision(
        -0.01, 0.0, insight_delta=0.2, insight_allowed=True,
        bridge_insight_gate=True, insight_accept_w=1.0)
    assert insight_decision["selected"] is True
    assert insight_decision["selected_by_insight"] is True
    assert insight_decision["insight_effective_delta"] > 0.0
    no_insight_decision = reading_round_selection_decision(
        -0.01, 0.0, insight_delta=0.2, insight_allowed=True,
        bridge_insight_gate=True, insight_accept_w=1.0,
        insight_min_delta=0.3)
    assert no_insight_decision["selected"] is False
    representation_decision = reading_round_selection_decision(
        -0.01, 0.0, insight_delta=0.0, insight_allowed=True,
        bridge_insight_gate=False, insight_accept_w=0.0,
        representation_delta=0.2, representation_allowed=True,
        representation_gate=True, representation_accept_w=1.0,
        representation_min_delta=0.05)
    assert representation_decision["selected"] is True
    assert representation_decision["selected_by_representation"] is True
    assert "representation_insight" in representation_decision["selection_reasons"]
    no_representation_decision = reading_round_selection_decision(
        -0.01, 0.0, insight_delta=0.0, insight_allowed=True,
        bridge_insight_gate=False, insight_accept_w=0.0,
        representation_delta=0.02, representation_allowed=True,
        representation_gate=True, representation_accept_w=1.0,
        representation_min_delta=0.05)
    assert no_representation_decision["selected"] is False
    blocked_score_decision = reading_round_selection_decision(
        0.2, 0.0, insight_delta=0.0, insight_allowed=True,
        signal_regression_allowed=False)
    assert blocked_score_decision["selected"] is False
    assert blocked_score_decision["pre_signal_selected"] is True
    assert blocked_score_decision["pre_signal_selected_by_score"] is True
    assert blocked_score_decision["blocked_by_signal_regression"] is True
    assert "signal_regression" in blocked_score_decision["blocked_reasons"]
    blocked_representation_decision = reading_round_selection_decision(
        -0.01, 0.0, insight_delta=0.0, insight_allowed=True,
        bridge_insight_gate=False, insight_accept_w=0.0,
        representation_delta=0.2, representation_allowed=True,
        representation_gate=True, representation_accept_w=1.0,
        representation_min_delta=0.05, signal_regression_allowed=False)
    assert blocked_representation_decision["selected"] is False
    assert (blocked_representation_decision[
        "pre_signal_selected_by_representation"] is True)
    assert blocked_representation_decision["blocked_by_signal_regression"] is True
    rep_delta, rep_allowed = reading_representation_insight_delta({
        "enabled": True,
        "representation_insight_event": True,
        "organization_score_delta": 0.2,
        "positive_signal_gain": 0.4,
        "negative_signal_drift": 0.1,
    })
    assert rep_allowed is True
    assert rep_delta > 0.0
    patience_model = TextReadingLM(
        len(reading_vocab), d=32, layers=1, heads=4, pad=reading_vocab.pad,
        latent_concept_slots=2, latent_concept_memory_size=8).to("cpu")
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
    assert patience_selection["branch_from_best"] is True
    assert patience_selection["rounds"][1]["branch_from_round"] == 0
    assert "selected_insight" in patience_selection
    helper_model, _helper_vocab = train_reading_concepts(
        reading_records, steps=2, batch=2, d=32, layers=1, heads=4, lr=1e-4,
        seed=11, device="cpu", log_every=10,
        token_drop_p=0.1, token_replace_p=0.0,
        latent_concept_slots=2, memory_size=8,
        bridge_w=0.1, study_strategy="auto",
        study_probe_n=2, study_hard_max=1, study_refresh_steps=1,
        study_select_best=True, study_rounds=2,
        study_score_min_delta=999.0, study_score_patience=1,
        study_self_teach_w=0.05,
        eval_n=0)
    helper_selection = helper_model.reading_train_metrics["selection"]
    assert helper_selection["enabled"] is True
    assert helper_selection["generation_eval_n"] == 0
    assert helper_selection["adaptive_study_strategy"] is True
    assert helper_selection["branch_from_best"] is True
    assert helper_selection["concept_insight_gate"] is True
    assert helper_selection["self_teach_w"] == 0.05
    assert helper_selection["self_teach_reports"]
    assert helper_selection["attempted_weight_update_count"] > 0
    assert helper_selection["self_teach_reports"][0]["branch_from_round"] == 0
    assert helper_selection["self_teach_reports"][0]["study_strategy"] in (
        READING_STUDY_STRATEGIES)
    assert helper_selection["rounds"][1]["self_teach_plan"]["enabled"] is True
    assert helper_selection["rounds"][1]["study_strategy_used"] in (
        READING_STUDY_STRATEGIES)
    assert "concept_insight" in helper_selection["rounds"][1]
    assert "concept_insight_delta" in helper_selection["rounds"][1]
    assert "representation_progress" in helper_selection["rounds"][1]
    assert "representation_insight_delta" in helper_selection["rounds"][1]
    assert "selected_by_representation" in helper_selection["rounds"][1]
    assert "replay_priority_sampling" in helper_selection["rounds"][1]
    assert helper_selection["rounds"][1]["weight_update_changed"] is True
    assert (helper_selection["rounds"][1][
        "weight_update_changed_tensor_count"] > 0)
    assert (helper_selection["rounds"][1][
        "weight_update_changed_value_count"] > 0)
    if (helper_selection["rounds"][1]["study_strategy_used"]
            not in READING_BRIDGE_INSIGHT_STUDY_STRATEGIES):
        assert helper_selection["rounds"][1]["bridge_insight_gate"] is False
        assert helper_selection["rounds"][1]["concept_insight_gate"] is False
    assert "self_teach_top_signal" in helper_selection["rounds"][1]
    assert helper_selection["accepted_update"] is False
    assert helper_selection["selected_round"] == 0
    assert helper_model.reading_study_reports == []
    plain_self_teach_model, _plain_vocab = train_reading_concepts(
        reading_records, steps=1, batch=2, d=32, layers=1, heads=4, lr=1e-4,
        seed=12, device="cpu", log_every=10,
        token_drop_p=0.1, token_replace_p=0.0,
        latent_concept_slots=2, memory_size=8,
        bridge_w=0.0, study_strategy="auto",
        study_probe_n=2, study_hard_max=1, study_refresh_steps=1,
        study_select_best=False, study_self_teach_w=0.05,
        eval_n=0)
    plain_self_teach_metrics = plain_self_teach_model.reading_train_metrics
    assert plain_self_teach_metrics["self_teach_w"] == 0.05
    assert plain_self_teach_metrics["self_teach_plan"]["enabled"] is True
    assert plain_self_teach_metrics["selection"]["enabled"] is False
    assert plain_self_teach_metrics["selection"]["self_teach_reports"]
    plain_extra_sum = sum(
        plain_self_teach_metrics["self_teach_plan"]["weight_extras"].values())
    assert plain_extra_sum > 0.0
    plain_weight_delta = sum(
        plain_self_teach_metrics["self_teach_effective_weights"][key]
        - plain_self_teach_metrics["self_teach_base_weights"][key]
        for key in READING_SELF_TEACH_WEIGHT_KEYS)
    assert math.isclose(
        plain_weight_delta, plain_extra_sum, rel_tol=1e-6, abs_tol=1e-6)
    reading_replay_bank = build_reading_replay_bank(
        reading_records, study_reports=getattr(reading_model, "reading_study_reports", []),
        max_records=4)
    reading_selftest_report = {
        "experiment": "reading-selftest",
        "data": ["selftest-base"],
        "steps": 3,
        "batch": 2,
        "reading_objective_profile": "mastery",
        "study_strategy_requested": "auto",
        "study_strategy": "sequence",
        "latent_concept_slots": 2,
        "latent_concept_topk": 0,
        "memory_size": 8,
        "train_records": 3,
        "eval_records": 2,
        "reading_replay_bank": reading_replay_bank,
        "before_score_components": {"mastery_score": 0.2, "signal_coverage": 0.4},
        "after_score_components": (
            {"mastery_score": 0.3, "signal_coverage": 0.6}
            | prior_scores),
        "delta": {"mastery_score": 0.1, "signal_coverage": 0.2},
        "selection": {
            "enabled": True,
            "accepted_update": True,
            "selected_round": 1,
            "selected_by_score": True,
            "selected_by_insight": True,
            "selected_score_delta": 0.1,
            "attempted_weight_update_count": 1,
            "rounds": [{
                "round": 1,
                "concept_insight_delta": 0.05,
                "replay_priority_sampling": True,
                "replay_priority_record_count": 1,
                "weight_update_changed": True,
                "weight_update_changed_tensor_count": 3,
                "weight_update_changed_value_count": 4,
                "weight_update_max_abs_delta": 0.01,
            }],
            "self_teach_reports": [{
                "top_signal": "sequence",
                "active_signals": ["sequence", "context"],
            }],
        },
        "representation_progress": {
            "enabled": True,
            "active_signals": ["fer", "bridge", "sequence"],
            "signal_after": {"fer": 0.7, "bridge": 0.4, "sequence": 0.2},
            "signal_deltas": {"fer": 0.1, "bridge": 0.2, "sequence": -0.1},
            "organization_score_before": 0.3,
            "organization_score_after": 0.5,
            "organization_score_delta": 0.2,
            "positive_signal_gain": 0.3,
            "negative_signal_drift": 0.1,
            "top_gain_signal": "bridge",
            "top_regression_signal": "sequence",
            "representation_insight_event": True,
        },
    }
    reading_selftest_report["reading_mastery_history"] = (
        reading_mastery_history_with_entry([], reading_selftest_report))
    reading_selftest_report["reading_mastery_history_count"] = (
        reading_selftest_report["reading_mastery_history"]["entry_count"])
    reading_payload = checkpoint_payload(
        reading_model, reading_vocab, 32, 1, 4,
        reading_selftest_report)
    assert reading_payload["reading_replay_bank"]["record_count"] > 0
    assert reading_replay_bank_records_from_payload(reading_payload)
    assert reading_payload["latent_concept_memory_size"] == 8
    assert reading_payload["reading_mastery_history"]["entry_count"] == 1
    first_history_entry = reading_payload["reading_mastery_history"]["entries"][0]
    assert first_history_entry["experiment"] == "reading-selftest"
    assert first_history_entry["accepted_update"] is True
    assert first_history_entry["self_teach_top_signal"] == "sequence"
    assert (first_history_entry["replay_priority_round_record_count"] == 1)
    assert first_history_entry["weight_update_changed"] is True
    assert first_history_entry["weight_update_changed_tensor_count"] == 3
    assert first_history_entry["weight_update_changed_value_count"] == 4
    assert first_history_entry["weight_update_max_abs_delta"] == 0.01
    assert first_history_entry["attempted_weight_update_count"] == 1
    assert first_history_entry["representation_insight_event"] is True
    assert (first_history_entry[
        "representation_organization_score_delta"] == 0.2)
    assert first_history_entry["representation_top_gain_signal"] == "bridge"
    assert first_history_entry["representation_progress"]["enabled"] is True
    assert first_history_entry["learning_event_triggered"] is True
    assert first_history_entry["learning_event_kind"] == "concept_connection"
    assert first_history_entry["learning_event_top_signal"] == "connection"
    assert first_history_entry["learning_event_score"] > 0.0
    assert reading_mastery_history_from_payload({})["entry_count"] == 0
    priority_replay_bank = build_reading_replay_bank(
        reading_records,
        study_reports=[{
            "strategy": "discovery",
            "mean_score": 0.4,
            "hard_record_ids": ["read-eval-1"],
            "study_pool_insight": {
                "record_ids": ["read-train-2"],
                "bridge_score_reduction": 0.3,
                "bridge_connectivity_gain": 0.2,
            },
            "record_ids": ["read-train-3"],
        }],
        learning_event={
            "triggered": True,
            "kind": "representation_reorganization",
            "top_signal": "bridge",
            "event_score": 0.6,
        },
        max_records=3)
    assert priority_replay_bank["priority_record_count"] == 3
    priority_rows = priority_replay_bank["records"]
    assert priority_rows[0]["id"] == "read-eval-1"
    assert "hard:discovery" in priority_rows[0]["replay_reasons"]
    assert "learning_event:representation_reorganization" in (
        priority_rows[0]["replay_reasons"])
    assert "learning_signal:bridge" in priority_rows[0]["replay_reasons"]
    assert priority_rows[1]["id"] == "read-train-2"
    assert "concept_insight" in priority_rows[1]["replay_reasons"]
    assert "learning_event:representation_reorganization" in (
        priority_rows[1]["replay_reasons"])
    assert "learning_event:representation_reorganization" in (
        priority_rows[2]["replay_reasons"])
    priority_records = reading_replay_bank_records_from_payload(
        {"reading_replay_bank": priority_replay_bank})
    assert priority_records[0].meta["replay_priority"] > 0.0
    assert "hard:discovery" in priority_records[0].meta["replay_reasons"]
    merged_priority_records = merge_reading_replay_metadata(
        [ReadingRecord(
            rec_id=priority_records[0].rec_id,
            split=priority_records[0].split,
            tokens=priority_records[0].tokens,
        )],
        priority_records)
    assert merged_priority_records[0].meta["replay_priority"] > 0.0
    assert "hard:discovery" in merged_priority_records[0].meta["replay_reasons"]
    carried_priority_bank = build_reading_replay_bank(
        priority_records, study_reports=[], max_records=3)
    assert carried_priority_bank["priority_record_count"] == 3
    assert carried_priority_bank["records"][0]["replay_priority"] > 0.0
    replay_weights = reading_replay_sampling_weights(priority_records)
    assert replay_weights is not None
    assert math.isclose(sum(replay_weights), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert replay_weights[0] == max(replay_weights)
    imbalanced_records = [
        ReadingRecord(f"source-a-{i}", "train", ("shared", "alpha"),
                      meta={"source": "source-a"})
        for i in range(8)
    ] + [
        ReadingRecord(f"source-b-{i}", "train", ("shared", "beta"),
                      meta={"source": "source-b"})
        for i in range(2)
    ]
    split_records, split_report = assign_reading_eval_splits(
        imbalanced_records, eval_frac=0.5, seed=0)
    assert split_report["eval_split_policy"] == "source"
    split_train_sources = {
        reading_record_source(rec) for rec in split_records
        if rec.split == "train"
    }
    split_eval_sources = {
        reading_record_source(rec) for rec in split_records
        if rec.split == "eval"
    }
    assert split_train_sources
    assert split_eval_sources
    assert split_train_sources.isdisjoint(split_eval_sources)
    assert reading_training_sampling_weights(
        imbalanced_records, source_balance_w=0.0) is None
    assert reading_replay_sampling_weights(
        imbalanced_records, source_balance_w=0.0) is None
    source_weights = reading_training_sampling_weights(
        imbalanced_records, source_balance_w=1.0)
    assert source_weights is not None
    assert math.isclose(sum(source_weights), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    source_a_mass = sum(
        weight for weight, rec in zip(source_weights, imbalanced_records)
        if reading_record_source(rec) == "source-a")
    source_b_mass = sum(
        weight for weight, rec in zip(source_weights, imbalanced_records)
        if reading_record_source(rec) == "source-b")
    assert math.isclose(source_a_mass, source_b_mass, rel_tol=1e-6, abs_tol=1e-6)
    assert source_weights[0] < source_weights[-1]
    replay_source_weights = reading_replay_sampling_weights(
        imbalanced_records, source_balance_w=1.0)
    assert replay_source_weights is not None
    assert math.isclose(
        sum(replay_source_weights), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    replay_source_a_mass = sum(
        weight for weight, rec in zip(replay_source_weights, imbalanced_records)
        if reading_record_source(rec) == "source-a")
    replay_source_b_mass = sum(
        weight for weight, rec in zip(replay_source_weights, imbalanced_records)
        if reading_record_source(rec) == "source-b")
    assert math.isclose(
        replay_source_a_mass, replay_source_b_mass,
        rel_tol=1e-6, abs_tol=1e-6)
    assert replay_source_weights[0] < replay_source_weights[-1]
    training_weights = reading_training_sampling_weights(priority_records)
    assert training_weights is not None
    assert math.isclose(sum(training_weights), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert training_weights[0] == max(training_weights)
    priority_batch = batch_replay_records(
        priority_records, np.random.default_rng(0), batch=8,
        weights=replay_weights)
    assert any(rec.rec_id == "read-eval-1" for rec in priority_batch)
    with tempfile.TemporaryDirectory() as td:
        base_ckpt = os.path.join(td, "reading_base.pt")
        sparse_ckpt = os.path.join(td, "reading_sparse.pt")
        studied_ckpt = os.path.join(td, "reading_studied.pt")
        study_data = os.path.join(td, "study_reading.jsonl")
        study_out = os.path.join(td, "study_report.json")
        torch.save(reading_payload, base_ckpt)
        torch.save(sparse_payload, sparse_ckpt)
        sparse_loaded, _sparse_vocab, sparse_loaded_ckpt = load_checkpoint(
            sparse_ckpt, device="cpu")
        assert sparse_loaded_ckpt["latent_concept_topk"] == 2
        assert sparse_loaded.latent_concept_topk == 2
        assert sparse_loaded.latent_concepts.topk == 2
        study_rows = [
            {"id": "study-new-1", "split": "train",
             "text": "novel abstractions crystallize through rereading"},
            {"id": "study-new-2", "split": "train",
             "text": "concept memory links unfamiliar symbols together"},
            {"id": "study-new-3", "split": "eval",
             "text": "rereading updates a connected concept graph"},
        ]
        with open(study_data, "w") as f:
            for row in study_rows:
                f.write(json.dumps(row) + "\n")
        study_report = study_reading_checkpoint(
            base_ckpt, [study_data], out_checkpoint=studied_ckpt,
            out=study_out, steps=1, batch=2, lr=1e-4, seed=23,
            device="cpu", log_every=10, max_tokens=16, min_tokens=3,
            eval_n=0, memory_size=8, memory_w=0.05,
            association_w=0.05, discovery_w=0.05, reanalysis_w=0.05,
            gap_w=0.05, study_strategy="gap", study_probe_n=2,
            study_hard_max=1, study_refresh_steps=1,
            study_select_best=False, context_target_w=0.0,
            sequence_w=0.0, transition_w=0.0,
            study_self_teach_w=0.05, reading_objective_profile="mastery",
            generation_eval_n=16,
            print_report=False)
        assert study_report["experiment"] == "text_raw_reading_checkpoint_study"
        assert study_report["generation_eval_n"] == 16
        assert study_report["before_generation"]["enabled"] is True
        assert study_report["after_generation"]["enabled"] is True
        assert "generation_score" in study_report["delta"]
        assert study_report["checkpoint_experiment"] == "reading-selftest"
        assert os.path.exists(studied_ckpt)
        assert os.path.exists(study_out)
        assert study_report["new_vocab_size"] >= study_report["old_vocab_size"]
        assert study_report["max_vocab"] == READING_DEFAULT_MAX_VOCAB
        assert study_report["vocab_capped"] is True
        assert study_report["source_balance_w"] == READING_DEFAULT_SOURCE_BALANCE_W
        assert study_report["new_tokens"] > 0
        assert study_report["discovery_w"] == 0.05
        assert study_report["reanalysis_w"] == 0.05
        assert study_report["gap_w"] == 0.05
        assert study_report["replay_bank_used"] is True
        assert study_report["replay_bank_records"] > 0
        assert study_report["replay_train_records"] > 0
        assert study_report["replay_study_used"] is True
        assert study_report["replay_study_records"] > 0
        assert study_report["study_record_count"] > len(study_rows)
        assert study_report["study_train_records"] >= 2
        assert study_report["before_replay"] is not None
        checkpoint_profile_report = study_report[
            "reading_objective_profile_report"]["checkpoint_replay"]
        assert checkpoint_profile_report["applied"] is True
        assert checkpoint_profile_report["updates"]["replay_w"]["to"] == 0.05
        assert (checkpoint_profile_report["updates"]["replay_retention_w"]["to"]
                == 0.25)
        assert study_report["train_metrics"]["study_strategy"] == "gap"
        assert study_report["train_metrics"]["memory_active"] > 0
        assert study_report["train_metrics"]["discovery_w"] >= 0.05
        assert study_report["train_metrics"]["reanalysis_w"] == 0.05
        assert study_report["train_metrics"]["gap_w"] >= 0.05
        assert study_report["train_metrics"]["replay_w"] == 0.05
        assert (study_report["train_metrics"]["training_source_balance_w"]
                == READING_DEFAULT_SOURCE_BALANCE_W)
        assert study_report["train_metrics"]["training_source_count"] >= 1
        if study_report["train_metrics"]["training_source_count"] > 1:
            assert (study_report["train_metrics"][
                "training_source_balance_sampling"] is True)
        assert study_report["selection"]["replay_retention_w"] == 0.25
        assert study_report["selection"]["enabled"] is False
        assert study_report["selection"]["self_teach_w"] == 0.05
        assert study_report["train_metrics"]["self_teach_w"] == 0.05
        assert study_report["reading_mastery_history_prior"]["enabled"] is True
        assert study_report["reading_mastery_history_prior"]["entry_count"] == 1
        assert study_report["reading_mastery_history_prior"][
            "concept_connection_signal"] > 0.0
        assert study_report["train_metrics"]["self_teach_plan"]["enabled"] is True
        assert study_report["train_metrics"]["self_teach_plan"][
            "history_prior_enabled"] is True
        assert study_report["train_metrics"]["self_teach_plan"][
            "history_prior_entry_count"] == 1
        assert study_report["train_metrics"]["self_teach_plan"][
            "history_signal_deficits"]["sequence"] > 0.0
        assert study_report["train_metrics"]["self_teach_plan"][
            "history_concept_connection_signal"] > 0.0
        assert sum(study_report["train_metrics"]["self_teach_plan"][
            "weight_extras"].values()) > 0.0
        assert study_report["train_metrics"]["replay_records"] > 0
        if reading_payload["reading_replay_bank"]["priority_record_count"] > 0:
            assert study_report["train_metrics"]["training_priority_sampling"] is True
            assert study_report["train_metrics"]["training_priority_record_count"] > 0
            assert study_report["train_metrics"]["replay_priority_sampling"] is True
            assert study_report["train_metrics"]["replay_priority_record_count"] > 0
        assert "replay_source_balance_sampling" in study_report["train_metrics"]
        assert (study_report["train_metrics"]["replay_source_balance_w"]
                == READING_DEFAULT_SOURCE_BALANCE_W)
        assert study_report["train_metrics"]["replay_source_count"] >= 1
        assert math.isfinite(study_report["train_metrics"]["gap_loss"])
        _studied_model, studied_vocab, studied_payload = load_checkpoint(
            studied_ckpt, device="cpu")
        assert "crystallize" in studied_vocab.stoi
        assert (studied_payload["report"]["experiment"]
                == "text_raw_reading_checkpoint_study")
        assert studied_payload["reading_replay_bank"]["record_count"] > 0
        assert studied_payload["reading_mastery_history"]["entry_count"] == 2
        assert study_report["reading_mastery_history_count"] == 2
        history_entries = studied_payload["reading_mastery_history"]["entries"]
        assert history_entries[0]["experiment"] == "reading-selftest"
        assert history_entries[1]["experiment"] == (
            "text_raw_reading_checkpoint_study")
        assert history_entries[1]["replay_bank_used"] is True
        assert history_entries[1]["replay_study_used"] is True
        assert history_entries[1]["replay_study_records"] > 0
        assert history_entries[1]["study_record_count"] > 0
        assert "replay_source_balance_sampling" in history_entries[1]
        assert (history_entries[1]["replay_source_balance_w"]
                == READING_DEFAULT_SOURCE_BALANCE_W)
        assert history_entries[1]["replay_source_count"] >= 1
        assert history_entries[1]["training_priority_sampling"] is True
        assert history_entries[1]["training_priority_record_count"] > 0
        assert (history_entries[1]["training_source_balance_w"]
                == READING_DEFAULT_SOURCE_BALANCE_W)
        assert history_entries[1]["training_source_count"] >= 1
        assert history_entries[1]["weight_update_changed"] is True
        assert history_entries[1]["weight_update_changed_tensor_count"] > 0
        assert history_entries[1]["weight_update_changed_value_count"] > 0
        assert history_entries[1]["weight_update_max_abs_delta"] > 0.0
        assert history_entries[1]["attempted_weight_update_count"] > 0
        assert history_entries[1]["representation_progress"]["enabled"] is True
        assert math.isfinite(
            history_entries[1]["representation_organization_score_delta"])
        assert history_entries[1]["learning_event"]["enabled"] is True
        assert "learning_event_triggered" in history_entries[1]
        assert history_entries[1]["self_teach_top_signal"]
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
    ap.add_argument("--reading-objective-profile",
                    choices=READING_OBJECTIVE_PROFILES, default="mastery",
                    help=("generic reading objective posture; mastery enables "
                          "schema-free concept/self-teach floors"))
    ap.add_argument("--reading-text-field", default="text")
    ap.add_argument("--reading-max-tokens", type=int, default=128)
    ap.add_argument("--reading-max-vocab", type=int,
                    default=READING_DEFAULT_MAX_VOCAB,
                    help=("maximum raw-reading vocabulary size; 0 disables "
                          "frequency capping"))
    ap.add_argument("--reading-source-balance-w", type=float,
                    default=READING_DEFAULT_SOURCE_BALANCE_W,
                    help=("smooth source-balanced raw-reading sampling; 0 "
                          "keeps uniform record sampling"))
    ap.add_argument("--reading-min-tokens", type=int, default=8)
    ap.add_argument("--reading-eval-frac", type=float, default=0.10)
    ap.add_argument("--reading-eval-n", type=int, default=64)
    ap.add_argument("--reading-generation-eval-n", type=int, default=0,
                    help=("free-continuation eval samples; mastery profile "
                          "raises this to a bounded default"))
    ap.add_argument("--reading-generation-prompt-tokens", type=int, default=16)
    ap.add_argument("--reading-generation-max-new-tokens", type=int, default=32)
    ap.add_argument("--reading-generation-temperature", type=float, default=0.0)
    ap.add_argument("--reading-generation-top-k", type=int, default=0)
    ap.add_argument("--reading-lr", type=float, default=1e-3)
    ap.add_argument("--reading-lm-w", type=float, default=0.0,
                    help=("causal next-token loss weight over raw reading "
                          "streams; mastery profile raises this by default"))
    ap.add_argument("--reading-continuation-repair-w", type=float, default=0.0,
                    help=("weight for recovering gold continuations after the "
                          "reader free-runs from a corpus prompt"))
    ap.add_argument("--reading-continuation-repair-steps", type=int, default=4)
    ap.add_argument("--reading-continuation-repair-prompt-frac", type=float,
                    default=0.5)
    ap.add_argument("--reading-continuation-repair-temperature", type=float,
                    default=0.0)
    ap.add_argument("--reading-continuation-repair-top-k", type=int, default=0)
    ap.add_argument("--reading-repetition-unlikelihood-w", type=float,
                    default=0.0,
                    help=("weight for discouraging recent non-gold token "
                          "repeats in raw reading continuations"))
    ap.add_argument("--reading-repetition-unlikelihood-window", type=int,
                    default=32)
    ap.add_argument("--reading-token-drop", type=float, default=0.15)
    ap.add_argument("--reading-token-replace", type=float, default=0.05)
    ap.add_argument("--reading-feature-dropout", type=float, default=0.1)
    ap.add_argument("--reading-context-target-w", type=float, default=0.1)
    ap.add_argument("--reading-context-keep-p", type=float, default=0.5)
    ap.add_argument("--reading-context-target-temperature", type=float, default=0.1)
    ap.add_argument("--reading-span-completion-w", type=float, default=0.05)
    ap.add_argument("--reading-span-mask-frac", type=float, default=0.25)
    ap.add_argument("--reading-span-completion-temperature", type=float, default=0.1)
    ap.add_argument("--reading-context-closure-w", type=float, default=0.05)
    ap.add_argument("--reading-context-closure-split-frac", type=float, default=0.5)
    ap.add_argument("--reading-context-closure-temperature", type=float, default=0.1)
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
    ap.add_argument("--reading-consolidation-w", type=float, default=0.0)
    ap.add_argument("--reading-consolidation-temperature", type=float, default=0.1)
    ap.add_argument("--reading-consolidation-balance-w", type=float, default=0.01)
    ap.add_argument("--reading-consolidation-anchor-w", type=float, default=1.0)
    ap.add_argument("--reading-consolidation-fer-w", type=float, default=0.0)
    ap.add_argument("--reading-discovery-w", type=float, default=0.0)
    ap.add_argument("--reading-discovery-curiosity-w", type=float, default=1.0)
    ap.add_argument("--reading-discovery-graph-w", type=float, default=1.0)
    ap.add_argument("--reading-discovery-cycle-w", type=float, default=1.0)
    ap.add_argument("--reading-discovery-bridge-w", type=float, default=1.0)
    ap.add_argument("--reading-discovery-fer-w", type=float, default=0.0)
    ap.add_argument("--reading-reanalysis-w", type=float, default=0.0)
    ap.add_argument("--reading-reanalysis-graph-w", type=float, default=1.0)
    ap.add_argument("--reading-reanalysis-cycle-w", type=float, default=0.5)
    ap.add_argument("--reading-reanalysis-bridge-w", type=float, default=0.5)
    ap.add_argument("--reading-reanalysis-fer-w", type=float, default=0.0)
    ap.add_argument("--reading-gap-w", type=float, default=0.0)
    ap.add_argument("--reading-gap-temperature", type=float, default=0.1)
    ap.add_argument("--reading-gap-self-loop-w", type=float, default=0.0)
    ap.add_argument("--reading-gap-transitive-steps", type=int, default=2)
    ap.add_argument("--reading-gap-transitive-w", type=float, default=0.1)
    ap.add_argument("--reading-gap-target-power", type=float, default=1.0)
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
    ap.add_argument(
        "--reading-neighborhood-probe-n", type=int, default=0,
        help=("candidate count for neighborhood mining; mastery profile "
              "raises 0 to its bounded default"))
    ap.add_argument(
        "--reading-neighborhood-refresh-steps", type=int, default=0,
        help=("steps between neighborhood remine passes; mastery profile "
              "raises 0 to its periodic default"))
    ap.add_argument("--reading-neighborhood-temperature", type=float, default=0.1)
    ap.add_argument("--reading-neighborhood-margin", type=float, default=0.0)
    ap.add_argument("--reading-transition-w", type=float, default=0.05)
    ap.add_argument("--reading-transition-batch", type=int, default=0)
    ap.add_argument("--reading-transition-temperature", type=float, default=0.1)
    ap.add_argument("--reading-transition-margin", type=float, default=0.0)
    ap.add_argument("--reading-cluster-w", type=float, default=0.0)
    ap.add_argument("--reading-cluster-batch", type=int, default=0)
    ap.add_argument(
        "--reading-cluster-probe-n", type=int, default=0,
        help=("candidate count for cluster mining; mastery profile raises "
              "0 to its bounded default"))
    ap.add_argument(
        "--reading-cluster-refresh-steps", type=int, default=0,
        help=("steps between cluster remine passes; mastery profile raises "
              "0 to its periodic default"))
    ap.add_argument("--reading-cluster-temperature", type=float, default=0.1)
    ap.add_argument("--reading-cluster-margin", type=float, default=0.0)
    ap.add_argument("--reading-cluster-min-size", type=int, default=2)
    ap.add_argument("--reading-study-strategy",
                    choices=READING_STUDY_STRATEGIES, default="auto")
    ap.add_argument(
        "--reading-study-probe-n", type=int, default=0,
        help=("candidate count for hard-study mining; mastery profile "
              "raises 0 to its bounded default"))
    ap.add_argument(
        "--reading-study-hard-max", type=int, default=0,
        help=("cap on selected hard records; mastery profile raises 0 to "
              "its bounded default"))
    ap.add_argument(
        "--reading-study-refresh-steps", type=int, default=0,
        help=("steps between hard-study remine passes; mastery profile "
              "raises 0 to its periodic default"))
    ap.add_argument("--reading-study-select-best",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reading-study-rounds", type=int, default=1)
    ap.add_argument("--reading-study-score-metric", choices=READING_SCORE_METRICS,
                    default="mastery")
    ap.add_argument("--reading-study-score-margin-w", type=float, default=0.1)
    ap.add_argument("--reading-study-score-min-delta", type=float, default=0.0)
    ap.add_argument("--reading-study-score-patience", type=int, default=0)
    ap.add_argument("--reading-study-score-target", type=float, default=0.0)
    ap.add_argument("--reading-study-signal-regression-tolerance", type=float,
                    default=READING_DEFAULT_SIGNAL_REGRESSION_TOLERANCE,
                    help=("maximum tolerated active-signal regression when "
                          "accepting a self-study round"))
    ap.add_argument("--reading-study-insight-accept-w", "--reading-study-insight-w",
                    type=float, default=0.25, dest="reading_study_insight_accept_w")
    ap.add_argument("--reading-study-insight-min-delta", type=float, default=0.0)
    ap.add_argument("--reading-study-representation-accept-w",
                    type=float, default=0.0)
    ap.add_argument("--reading-study-representation-min-delta",
                    type=float, default=0.0)
    ap.add_argument("--reading-study-self-teach-w", type=float, default=0.0)
    ap.add_argument("--reading-self-teach-history-prior-w", type=float, default=0.5,
                    help=("blend weight for checkpoint mastery-history deficits "
                          "inside self-teach allocation"))


def _reading_kwargs(args):
    kwargs = dict(lr=args.reading_lr,
                  lm_w=args.reading_lm_w,
                  continuation_repair_w=args.reading_continuation_repair_w,
                  continuation_repair_steps=(
                      args.reading_continuation_repair_steps),
                  continuation_repair_prompt_frac=(
                      args.reading_continuation_repair_prompt_frac),
                  continuation_repair_temperature=(
                      args.reading_continuation_repair_temperature),
                  continuation_repair_top_k=(
                      args.reading_continuation_repair_top_k),
                  repetition_unlikelihood_w=(
                      args.reading_repetition_unlikelihood_w),
                  repetition_unlikelihood_window=(
                      args.reading_repetition_unlikelihood_window),
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
                  consolidation_w=args.reading_consolidation_w,
                  consolidation_temperature=args.reading_consolidation_temperature,
                  consolidation_balance_w=args.reading_consolidation_balance_w,
                  consolidation_anchor_w=args.reading_consolidation_anchor_w,
                  consolidation_fer_w=args.reading_consolidation_fer_w,
                  discovery_w=args.reading_discovery_w,
                  discovery_curiosity_w=args.reading_discovery_curiosity_w,
                  discovery_graph_w=args.reading_discovery_graph_w,
                  discovery_cycle_w=args.reading_discovery_cycle_w,
                  discovery_bridge_w=args.reading_discovery_bridge_w,
                  discovery_fer_w=args.reading_discovery_fer_w,
                  reanalysis_w=args.reading_reanalysis_w,
                  reanalysis_graph_w=args.reading_reanalysis_graph_w,
                  reanalysis_cycle_w=args.reading_reanalysis_cycle_w,
                  reanalysis_bridge_w=args.reading_reanalysis_bridge_w,
                  reanalysis_fer_w=args.reading_reanalysis_fer_w,
                  gap_w=args.reading_gap_w,
                  gap_temperature=args.reading_gap_temperature,
                  gap_self_loop_w=args.reading_gap_self_loop_w,
                  gap_transitive_steps=args.reading_gap_transitive_steps,
                  gap_transitive_w=args.reading_gap_transitive_w,
                  gap_target_power=args.reading_gap_target_power,
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
                  span_completion_w=args.reading_span_completion_w,
                  span_mask_frac=args.reading_span_mask_frac,
                  span_completion_temperature=(
                      args.reading_span_completion_temperature),
                  context_closure_w=args.reading_context_closure_w,
                  context_closure_split_frac=args.reading_context_closure_split_frac,
                  context_closure_temperature=(
                      args.reading_context_closure_temperature),
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
                  study_score_target=args.reading_study_score_target,
                  study_signal_regression_tolerance=(
                      args.reading_study_signal_regression_tolerance),
                  study_insight_accept_w=args.reading_study_insight_accept_w,
                  study_insight_min_delta=args.reading_study_insight_min_delta,
                  study_representation_accept_w=(
                      args.reading_study_representation_accept_w),
                  study_representation_min_delta=(
                      args.reading_study_representation_min_delta),
                  study_self_teach_w=args.reading_study_self_teach_w,
                  text_field=args.reading_text_field,
                  max_tokens=args.reading_max_tokens,
                  max_vocab=args.reading_max_vocab,
                  source_balance_w=args.reading_source_balance_w,
                  min_tokens=args.reading_min_tokens,
                  eval_frac=args.reading_eval_frac,
                  eval_n=args.reading_eval_n,
                  generation_eval_n=args.reading_generation_eval_n,
                  generation_prompt_tokens=args.reading_generation_prompt_tokens,
                  generation_max_new_tokens=(
                      args.reading_generation_max_new_tokens),
                  generation_temperature=args.reading_generation_temperature,
                  generation_top_k=args.reading_generation_top_k)
    return reading_objective_profile_kwargs(
        args.reading_objective_profile, **kwargs)


def _new_reading_latent_slots(args):
    """Fresh raw-reading runs need latent slots when concept memory is enabled."""
    return (int(args.latent_concept_slots)
            if int(args.latent_concept_slots) > 0
            else READING_DEFAULT_LATENT_CONCEPT_SLOTS)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
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
    ap.add_argument("--latent-concept-slots", type=int, default=0)
    ap.add_argument("--latent-concept-layers", type=int, default=1)
    ap.add_argument("--latent-concept-topk", type=int, default=0,
                    help=("keep only the top-k latent concept slots per record; "
                          "0 keeps all slots"))
    ap.add_argument("--latent-concept-prefix", action="store_true")
    ap.add_argument("--latent-concept-refine", action="store_true")
    ap.add_argument("--latent-concept-refine-gate-init", type=float, default=-2.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if args.latent_concept_topk < 0:
        raise SystemExit("--latent-concept-topk must be non-negative")
    if args.reading_max_vocab < 0:
        raise SystemExit("--reading-max-vocab must be non-negative")
    if args.reading_source_balance_w < 0.0:
        raise SystemExit("--reading-source-balance-w must be non-negative")
    if args.reading_lm_w < 0.0:
        raise SystemExit("--reading-lm-w must be non-negative")
    if args.reading_continuation_repair_w < 0.0:
        raise SystemExit("--reading-continuation-repair-w must be non-negative")
    if args.reading_continuation_repair_steps < 0:
        raise SystemExit(
            "--reading-continuation-repair-steps must be non-negative")
    if (args.reading_continuation_repair_prompt_frac <= 0.0
            or args.reading_continuation_repair_prompt_frac >= 1.0):
        raise SystemExit(
            "--reading-continuation-repair-prompt-frac must be in (0, 1)")
    if args.reading_continuation_repair_temperature < 0.0:
        raise SystemExit(
            "--reading-continuation-repair-temperature must be non-negative")
    if args.reading_continuation_repair_top_k < 0:
        raise SystemExit("--reading-continuation-repair-top-k must be non-negative")
    if args.reading_repetition_unlikelihood_w < 0.0:
        raise SystemExit(
            "--reading-repetition-unlikelihood-w must be non-negative")
    if args.reading_repetition_unlikelihood_window < 0:
        raise SystemExit(
            "--reading-repetition-unlikelihood-window must be non-negative")
    if args.reading_generation_eval_n < 0:
        raise SystemExit("--reading-generation-eval-n must be non-negative")
    if args.reading_generation_prompt_tokens <= 0:
        raise SystemExit("--reading-generation-prompt-tokens must be positive")
    if args.reading_generation_max_new_tokens < 0:
        raise SystemExit("--reading-generation-max-new-tokens must be non-negative")
    if args.reading_generation_temperature < 0.0:
        raise SystemExit("--reading-generation-temperature must be non-negative")
    if args.reading_generation_top_k < 0:
        raise SystemExit("--reading-generation-top-k must be non-negative")
    if args.reading_self_teach_history_prior_w < 0.0:
        raise SystemExit("--reading-self-teach-history-prior-w must be non-negative")
    if args.reading_study_representation_accept_w < 0.0:
        raise SystemExit(
            "--reading-study-representation-accept-w must be non-negative")
    if args.reading_study_representation_min_delta < 0.0:
        raise SystemExit(
            "--reading-study-representation-min-delta must be non-negative")
    if args.reading_study_signal_regression_tolerance < 0.0:
        raise SystemExit(
            "--reading-study-signal-regression-tolerance must be non-negative")
    if not args.reading_data:
        raise SystemExit("--reading-data is required unless --selftest is set")
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
            self_teach_history_prior_w=args.reading_self_teach_history_prior_w,
            latent_concept_slots=args.latent_concept_slots,
            latent_concept_layers=args.latent_concept_layers,
            latent_concept_topk=args.latent_concept_topk,
            latent_concept_prefix=args.latent_concept_prefix,
            latent_concept_refine=args.latent_concept_refine,
            latent_concept_refine_gate_init=(
                args.latent_concept_refine_gate_init),
            **reading_common)
        return
    run_reading_concepts(
        args.reading_data, steps=args.steps, batch=args.batch, d=args.d,
        layers=args.layers, heads=args.heads,
        text_encoder_arch=args.text_encoder_arch,
        text_encoder_layers=args.text_encoder_layers,
        latent_concept_slots=_new_reading_latent_slots(args),
        latent_concept_layers=args.latent_concept_layers,
        latent_concept_topk=args.latent_concept_topk,
        latent_concept_prefix=args.latent_concept_prefix,
        latent_concept_refine=args.latent_concept_refine,
        latent_concept_refine_gate_init=(
            args.latent_concept_refine_gate_init),
        seed=args.seed, device=DEV, out=args.out,
        checkpoint=args.checkpoint, **reading_common)


if __name__ == "__main__":
    main()
