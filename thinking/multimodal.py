"""Generic multimodal prefix bridge.

This module intentionally has no built-in sensory oracle or fixed modality schema.
It trains from a JSONL manifest where each record supplies optional named feature
views, text tokens, and an optional target token sequence.  The bridge learns to fuse those
views into continuous prefixes for one ScratchpadLM decoder.  When no explicit
targets are present, the default decoder objective treats text as a causal token
sequence instead of skipping token learning.

Example manifest row:

  {"split":"train","text":["caption","tokens"],
   "views":{"sensor_a":[0.1,0.2],"sensor_b":[0.3,0.4]},
   "target":["extract","concept","x","done","."]}

  python -m thinking.multimodal --manifest data/mm.jsonl --steps 400
  python -m thinking.multimodal --selftest
"""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import os
import tempfile

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
    latent_concept_association_loss,
    latent_concept_bridge_loss,
    latent_concept_bridge_scores,
    latent_concept_cluster_prototype_loss,
    latent_concept_completion_loss,
    latent_concept_completion_scores,
    latent_concept_composition_loss,
    latent_concept_fer_loss,
    latent_concept_fer_metrics,
    latent_concept_fer_scores,
    latent_concept_discovery_loss,
    latent_concept_graph_curiosity_scores,
    latent_concept_graph_cycle_scores,
    latent_concept_graph_prediction_loss,
    latent_concept_graph_prediction_scores,
    latent_concept_insight_scores,
    latent_concept_graph_ready,
    latent_concept_graph_snapshot,
    latent_concept_memory_gap_loss,
    latent_concept_memory_gap_scores,
    latent_concept_memory_consolidation_loss,
    latent_concept_memory_loss,
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
MODES = ("full", "sensor_only", "text_only")
TRUNK_ARCHES = ("mlp", "residual")
TEXT_TRUNK_ARCHES = ("transformer", "standard", "relational", "abstractor")
FEATURE_VIEW_KEYS = ("views", "features", "feature_views")
TEXT_KEYS = ("text_tokens", "tokens", "text", "caption")
TARGET_KEYS = ("target_tokens", "target", "trace_tokens", "trace")
DECODE_OBJECTIVES = ("auto", "target", "causal")
EMPTY_TEXT_TOKENS = ("<empty_text>",)
MULTIMODAL_SCORE_METRICS = (
    "token", "exact", "generation", "fer", "bridge", "connection",
    "sequence", "all", "balanced", "mastery")
MULTIMODAL_SELF_TEACH_SIGNALS = (
    "token", "generation", "mode_floor", "fer", "bridge", "connection",
    "sequence")
MULTIMODAL_SELF_TEACH_SCORE_KEYS = {
    "token": "token_score",
    "generation": "generation_score",
    "mode_floor": "mode_floor_score",
    "fer": "fer_score",
    "bridge": "bridge_score",
    "connection": "connection_score",
    "sequence": "sequence_score",
}
MULTIMODAL_SELF_TEACH_SKIP_KEYS = {
    "token": "token_skipped",
    "generation": "generation_skipped",
    "mode_floor": "token_skipped",
    "fer": "fer_skipped",
    "bridge": "bridge_skipped",
    "connection": "connection_skipped",
    "sequence": "sequence_skipped",
}
MULTIMODAL_SELF_TEACH_SIGNAL_OBJECTIVES = {
    "token": ("decode_w",),
    "generation": (
        "decode_w", "continuation_repair_w", "repetition_unlikelihood_w"),
    "mode_floor": (
        "agreement_w", "latent_concept_completion_w"),
    "fer": ("latent_concept_fer_w", "latent_concept_factorization_w"),
    "bridge": (
        "latent_concept_bridge_w", "latent_concept_discovery_w",
        "latent_concept_gap_w", "latent_concept_graph_predict_w"),
    "connection": (
        "latent_concept_bridge_w", "latent_concept_discovery_w",
        "latent_concept_gap_w", "latent_concept_graph_predict_w"),
    "sequence": ("latent_concept_sequence_w", "latent_concept_transition_w"),
}
MULTIMODAL_SELF_TEACH_WEIGHT_KEYS = tuple(dict.fromkeys([
    key
    for keys in MULTIMODAL_SELF_TEACH_SIGNAL_OBJECTIVES.values()
    for key in keys
] + ["latent_concept_w"]))
MULTIMODAL_LATENT_SELF_TEACH_WEIGHT_KEYS = tuple(
    key for key in MULTIMODAL_SELF_TEACH_WEIGHT_KEYS
    if key.startswith("latent_concept_"))


def multimodal_self_teach_available_objectives(latent_concept_slots=0,
                                               active_mode_count=1):
    available = {
        "decode_w",
        "continuation_repair_w",
        "repetition_unlikelihood_w",
    }
    if int(active_mode_count) > 1:
        available.add("agreement_w")
    if int(latent_concept_slots) > 0:
        available.update(MULTIMODAL_LATENT_SELF_TEACH_WEIGHT_KEYS)
    return tuple(
        key for key in MULTIMODAL_SELF_TEACH_WEIGHT_KEYS
        if key in available)


MULTIMODAL_TEXT_HISTORY_SIGNAL_KEYS = {
    "token": ("language_score", "lm_token_acc"),
    "mode_floor": ("signal_coverage", "balanced_score", "floor_score"),
    "fer": ("fer_score",),
    "bridge": ("bridge_score", "bridge_connectivity"),
    "connection": ("connection_score",),
    "sequence": ("sequence_score",),
}
MULTIMODAL_TEXT_HISTORY_TOP_SIGNAL_MAP = {
    "language": "token",
    "view": "mode_floor",
    "context": "mode_floor",
    "span": "mode_floor",
    "closure": "mode_floor",
    "neighborhood": "sequence",
    "cluster": "sequence",
    "fer": "fer",
    "bridge": "bridge",
    "connection": "connection",
    "sequence": "sequence",
}
MULTIMODAL_REPRESENTATION_PROGRESS_KEYS = (
    "mastery_score", "active_mean_score", "floor_score", "balanced_score",
    "signal_coverage", "mode_floor_score", "token_score",
    "generation_score", "fer_score", "bridge_score", "connection_score",
    "sequence_score")
MULTIMODAL_REPRESENTATION_SIGNAL_KEYS = (
    "token_score", "generation_score", "fer_score", "bridge_score",
    "connection_score", "sequence_score")
DEFAULT_TEXT_TRANSFER_PROBE_N = 64
DEFAULT_TEXT_TRANSFER_SCORE_MIN_DELTA = 0.1
DEFAULT_TEXT_TRANSFER_INSIGHT_ACCEPT_W = 0.0
MULTIMODAL_DEFAULT_SIGNAL_REGRESSION_TOLERANCE = 0.02
MULTIMODAL_DEFAULT_SOURCE_BALANCE_W = 0.5
MULTIMODAL_DEFAULT_LATENT_CONCEPT_SLOTS = 4
MULTIMODAL_DEFAULT_LATENT_CONCEPT_MEMORY_SIZE = 64
MULTIMODAL_OBJECTIVE_PROFILES = ("manual", "language", "mastery")
MULTIMODAL_LANGUAGE_OBJECTIVE_FLOORS = {
    "continuation_repair_w": 0.05,
    "continuation_repair_steps": 4,
    "repetition_unlikelihood_w": 0.05,
    "repetition_unlikelihood_window": 32,
    "selection_generation_n": 16,
}
MULTIMODAL_MASTERY_OBJECTIVE_FLOORS = {
    "latent_concept_slots": MULTIMODAL_DEFAULT_LATENT_CONCEPT_SLOTS,
    "latent_concept_memory_size": MULTIMODAL_DEFAULT_LATENT_CONCEPT_MEMORY_SIZE,
    "latent_concept_factorization_w": 0.05,
    "latent_concept_fer_w": 0.05,
    "latent_concept_memory_w": 0.05,
    "latent_concept_consolidation_w": 0.05,
    "latent_concept_discovery_w": 0.05,
    "latent_concept_gap_w": 0.05,
    "latent_concept_association_w": 0.05,
    "latent_concept_composition_w": 0.05,
    "latent_concept_graph_predict_w": 0.05,
    "latent_concept_bridge_w": 0.05,
    "latent_concept_completion_w": 0.10,
    "latent_concept_sequence_w": 0.10,
    "latent_concept_neighborhood_w": 0.05,
    "latent_concept_transition_w": 0.05,
    "latent_concept_cluster_w": 0.05,
    "self_teach_w": 0.05,
}
MULTIMODAL_MASTERY_STUDY_FLOORS = {
    "latent_concept_fer_probe_n": 128,
    "latent_concept_fer_hard_max": 32,
    "latent_concept_fer_refresh_steps": 100,
    "latent_concept_discovery_probe_n": 128,
    "latent_concept_discovery_hard_max": 32,
    "latent_concept_discovery_refresh_steps": 100,
    "latent_concept_completion_probe_n": 128,
    "latent_concept_completion_hard_max": 32,
    "latent_concept_completion_refresh_steps": 100,
    "representation_probe_n": 64,
    "selection_rounds": 2,
    "selection_score_patience": 1,
}
MULTIMODAL_MASTERY_PROFILE_FLOORS = (
    dict(MULTIMODAL_MASTERY_OBJECTIVE_FLOORS)
    | dict(MULTIMODAL_MASTERY_STUDY_FLOORS))
MULTIMODAL_LEARNING_HISTORY_VERSION = 1
MULTIMODAL_LEARNING_HISTORY_SIZE = 32


def _mm_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _mm_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class MultimodalRecord:
    rec_id: str
    split: str
    text: tuple
    target: tuple
    views: dict
    meta: dict


def _tokens(value, *, field, rec_id):
    if isinstance(value, str):
        toks = value.split()
    elif isinstance(value, (list, tuple)):
        toks = [str(x) for x in value]
    else:
        raise ValueError(f"{rec_id}: {field} must be a string or token list")
    if not toks:
        raise ValueError(f"{rec_id}: {field} must not be empty")
    return tuple(toks)


def _first_present(rec, keys):
    for key in keys:
        if key in rec and rec[key] is not None:
            return key, rec[key]
    return None, None


def _feature_vector_value(value, *, root, rec_id, field):
    if isinstance(value, str):
        path = value if os.path.isabs(value) else os.path.join(root or ".", value)
        arr = np.load(path)
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{rec_id}: {field} feature vector is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{rec_id}: {field} feature vector has non-finite values")
    return arr


def _feature_views(rec, *, root, rec_id):
    key, value = _first_present(rec, FEATURE_VIEW_KEYS)
    if key is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{rec_id}: {key} must be an object of named feature vectors")
    views = OrderedDict()
    for raw_name, raw_value in value.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{rec_id}: {key} contains an empty view name")
        if raw_value is None:
            continue
        views[name] = _feature_vector_value(
            raw_value, root=root, rec_id=rec_id, field=f"{key}.{name}")
    return dict(views)


def _record_from_json(obj, idx, root=None, default_source=None):
    if not isinstance(obj, dict):
        raise ValueError(f"manifest row {idx} must be an object")
    rec_id = str(obj.get("id", f"record-{idx}"))
    split = str(obj.get("split", "train"))
    text_key, text_value = _first_present(obj, TEXT_KEYS)
    if text_key is None:
        text = ("<empty_text>",)
    else:
        text = _tokens(text_value, field=text_key, rec_id=rec_id)
    target_key, target_value = _first_present(obj, TARGET_KEYS)
    if target_key is None:
        target = ()
    else:
        target = _tokens(target_value, field=target_key, rec_id=rec_id)
    views = _feature_views(obj, root=root, rec_id=rec_id)
    meta = dict(obj.get("meta", {})) if isinstance(obj.get("meta", {}), dict) else {}
    for key in ("source", "document", "dataset"):
        if key not in meta and obj.get(key) is not None:
            meta[key] = obj[key]
    if (default_source is not None
            and not any(meta.get(key) for key in ("source", "document", "dataset"))):
        meta["source"] = default_source
    return MultimodalRecord(rec_id, split, text, target, views, meta)


def load_manifest(path, root=None):
    records = []
    default_source = os.path.abspath(path)
    with open(path) as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            records.append(_record_from_json(
                json.loads(line), idx, root=root, default_source=default_source))
    if not records:
        raise ValueError(f"{path} has no records")
    splits = {r.split for r in records}
    if "train" not in splits:
        raise ValueError(f"{path} must contain at least one train record")
    return records


def feature_dims(records):
    dims = OrderedDict()
    for rec in records:
        for name, value in rec.views.items():
            dim = int(value.size)
            if name not in dims:
                dims[name] = dim
            elif dims[name] != dim:
                raise ValueError(
                    f"{rec.rec_id}: view {name!r} dim {dim} does not match {dims[name]}")
    return OrderedDict((name, int(dim)) for name, dim in dims.items())


def _has_meaningful_text(rec):
    return bool(rec.text) and tuple(rec.text) != EMPTY_TEXT_TOKENS


def multimodal_active_modes(view_dims=None, records=None):
    """Return only modes that have real conditioning signal for this manifest."""
    has_views = bool(view_dims)
    has_text = True if records is None else any(
        _has_meaningful_text(rec) for rec in records)
    if has_views and has_text:
        return MODES
    if has_views:
        return ("full", "sensor_only")
    return ("full",)


def build_text_causal_lm_manifest(
        paths, out, text_field="text", max_tokens=128, min_tokens=8,
        stride=0, eval_frac=0.10, seed=0):
    """Build a targetless causal LM manifest from raw text/code sources."""
    from .text import (
        assign_reading_eval_splits,
        load_reading_records,
        write_causal_lm_manifest,
    )

    paths = list(paths or [])
    if not paths:
        raise ValueError("text causal LM manifest needs at least one source")
    max_tokens = max(2, int(max_tokens))
    min_tokens = max(2, int(min_tokens))
    stride = int(stride)
    if stride <= 0:
        stride = max(1, max_tokens // 2)
    records = []
    for i, path in enumerate(paths):
        records.extend(load_reading_records(
            path, require_train=False, require_eval=False,
            text_field=text_field, max_tokens=max_tokens,
            min_tokens=min_tokens, eval_frac=0.0, seed=seed + i,
            stride=stride))
    records, split_report = assign_reading_eval_splits(
        records, eval_frac=eval_frac, seed=seed)
    if not any(rec.split == "train" for rec in records):
        raise ValueError("text causal LM manifest has no train records")
    report = write_causal_lm_manifest(records, out)
    report.update({
        "sources": [str(path) for path in paths],
        "text_field": str(text_field),
        "window_tokens": int(max_tokens),
        "window_stride": int(stride),
        "min_tokens": int(min_tokens),
        "eval_frac": float(eval_frac),
        "objective": "causal_lm",
        "targetless": True,
        "split_report": split_report,
    })
    return report


def split_records(records):
    train = [r for r in records if r.split == "train"]
    evals = [r for r in records if r.split == "eval"]
    return train, (evals or train)


def multimodal_record_source(rec):
    meta = rec.meta if isinstance(rec.meta, dict) else {}
    for key in ("source", "document", "dataset"):
        value = meta.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "__unknown__"


def multimodal_source_counts(records):
    counts = OrderedDict()
    for rec in records:
        source = multimodal_record_source(rec)
        counts[source] = int(counts.get(source, 0)) + 1
    return counts


def multimodal_training_sampling_weights(
        records, source_balance_w=MULTIMODAL_DEFAULT_SOURCE_BALANCE_W):
    source_balance_w = float(source_balance_w)
    if source_balance_w < 0.0:
        raise ValueError("multimodal source balance weight must be non-negative")
    if not records or source_balance_w <= 0.0:
        return None
    counts = multimodal_source_counts(records)
    if len(counts) <= 1:
        return None
    weights = np.asarray([
        1.0 / (float(counts[multimodal_record_source(rec)]) ** source_balance_w)
        for rec in records
    ], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return None
    return weights / total


def multimodal_source_balance_report(records, source_balance_w, weights=None):
    counts = multimodal_source_counts(records)
    total = float(sum(counts.values()))
    fracs = [
        (float(count) / total) for count in counts.values()
    ] if total > 0.0 else []
    return {
        "training_weighted_sampling": weights is not None,
        "training_source_balance_sampling": bool(weights is not None),
        "training_source_balance_w": float(source_balance_w),
        "training_source_count": int(len(counts)),
        "training_source_record_counts": dict(counts),
        "training_max_source_record_frac": float(max(fracs) if fracs else 0.0),
        "training_min_source_record_frac": float(min(fracs) if fracs else 0.0),
    }


def build_vocab(records, max_size=None):
    toks = []
    for rec in records:
        toks.extend(rec.text)
        toks.extend(rec.target)
    return Vocab(toks, max_size=max_size)


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_text_checkpoint_payload(path, device="cpu"):
    ckpt = _torch_load(path, device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "vocab" not in ckpt:
        raise ValueError(f"{path} is not a text checkpoint with state_dict and vocab")
    return ckpt


def load_multimodal_checkpoint_payload(path, device="cpu"):
    ckpt = _torch_load(path, device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "vocab" not in ckpt:
        raise ValueError(
            f"{path} is not a multimodal checkpoint with state_dict and vocab")
    if not isinstance(ckpt.get("model_config"), dict):
        raise ValueError(f"{path} is missing multimodal model_config")
    return ckpt


def text_checkpoint_latent_config(path, device="cpu"):
    if not path:
        return {}
    ckpt = load_text_checkpoint_payload(path, device=device)
    return {
        "latent_concept_slots": int(ckpt.get("latent_concept_slots", 0)),
        "latent_concept_layers": int(ckpt.get("latent_concept_layers", 1)),
        "latent_concept_topk": int(ckpt.get("latent_concept_topk", 0)),
        "latent_concept_prefix": bool(ckpt.get("latent_concept_prefix", False)),
        "latent_concept_refine": bool(ckpt.get("latent_concept_refine", False)),
        "latent_concept_refine_gate_init": float(
            ckpt.get("latent_concept_refine_gate_init", -2.0)),
        "latent_concept_memory_size": int(
            ckpt.get("latent_concept_memory_size", 0)),
    }


def text_checkpoint_reading_history(ckpt):
    if not isinstance(ckpt, dict):
        return {"entry_count": 0, "entries": []}
    report = ckpt.get("report") if isinstance(ckpt.get("report"), dict) else {}
    history = ckpt.get("reading_mastery_history")
    if not isinstance(history, dict):
        history = report.get("reading_mastery_history")
    if not isinstance(history, dict):
        return {"entry_count": 0, "entries": []}
    entries = history.get("entries", ())
    if not isinstance(entries, (list, tuple)):
        entries = ()
    entries = [entry for entry in entries if isinstance(entry, dict)]
    return {
        "version": _mm_int(history.get("version", 0), 0),
        "entry_count": _mm_int(history.get("entry_count", len(entries)), len(entries)),
        "entries": entries,
    }


def text_checkpoint_concept_insight_prior(ckpt, enabled=True,
                                          max_entries=8, decay=0.75):
    if not enabled:
        return {
            "enabled": False,
            "entry_count": 0,
            "concept_connection_signal": 0.0,
        }
    history = text_checkpoint_reading_history(ckpt)
    entries = history["entries"][-max(1, int(max_entries)):]
    decay = min(1.0, max(0.0, _mm_float(decay, 0.75)))
    weighted_delta = 0.0
    total_weight = 0.0
    max_delta = 0.0
    latest_delta = 0.0
    selected_by_insight = 0
    accepted_updates = 0
    for offset, entry in enumerate(reversed(entries)):
        delta = max(0.0, _mm_float(entry.get("max_concept_insight_delta", 0.0)))
        learning_event = entry.get("learning_event")
        if (isinstance(learning_event, dict)
                and bool(learning_event.get("triggered", False))
                and str(learning_event.get("kind", "")) == "concept_connection"):
            delta = max(
                delta,
                _mm_float(learning_event.get("event_score", 0.0), 0.0))
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


def text_checkpoint_weight_update_summary(ckpt, max_entries=8):
    history = text_checkpoint_reading_history(ckpt)
    entries = history["entries"][-max(1, int(max_entries)):]
    if not entries:
        return {
            "entry_count": 0,
            "changed": False,
            "latest_changed": False,
            "changed_tensor_count": 0,
            "changed_value_count": 0,
            "max_abs_delta": 0.0,
            "latest_max_abs_delta": 0.0,
            "attempted_weight_update_count": 0,
        }
    changed = any(bool(entry.get("weight_update_changed", False))
                  for entry in entries)
    latest = entries[-1]
    return {
        "entry_count": int(len(entries)),
        "changed": bool(changed),
        "latest_changed": bool(latest.get("weight_update_changed", False)),
        "changed_tensor_count": max(
            _mm_int(entry.get("weight_update_changed_tensor_count", 0), 0)
            for entry in entries),
        "changed_value_count": max(
            _mm_int(entry.get("weight_update_changed_value_count", 0), 0)
            for entry in entries),
        "max_abs_delta": max(
            _mm_float(entry.get("weight_update_max_abs_delta", 0.0), 0.0)
            for entry in entries),
        "latest_max_abs_delta": _mm_float(
            latest.get("weight_update_max_abs_delta", 0.0), 0.0),
        "attempted_weight_update_count": sum(
            _mm_int(entry.get("attempted_weight_update_count", 0), 0)
            for entry in entries),
    }


def text_checkpoint_learning_event_summary(ckpt, max_entries=8):
    history = text_checkpoint_reading_history(ckpt)
    entries = history["entries"][-max(1, int(max_entries)):]
    events = [
        entry.get("learning_event")
        for entry in entries
        if isinstance(entry.get("learning_event"), dict)
    ]
    if not events:
        return {
            "enabled": False,
            "entry_count": int(len(entries)),
            "event_count": 0,
            "triggered_count": 0,
            "latest_triggered": False,
            "max_event_score": 0.0,
            "latest_event_score": 0.0,
            "top_signal": "",
            "latest_top_signal": "",
            "kind_counts": {},
        }
    triggered = [
        event for event in events
        if bool(event.get("triggered", False))
    ]
    latest = events[-1]
    top_event = max(
        triggered,
        key=lambda event: _mm_float(event.get("event_score", 0.0), 0.0),
        default={})
    kind_counts = {}
    for event in triggered:
        kind = str(event.get("kind", ""))
        if kind:
            kind_counts[kind] = int(kind_counts.get(kind, 0)) + 1
    return {
        "enabled": True,
        "entry_count": int(len(entries)),
        "event_count": int(len(events)),
        "triggered_count": int(len(triggered)),
        "latest_triggered": bool(latest.get("triggered", False)),
        "max_event_score": max(
            (_mm_float(event.get("event_score", 0.0), 0.0)
             for event in triggered),
            default=0.0),
        "latest_event_score": _mm_float(
            latest.get("event_score", 0.0), 0.0),
        "top_signal": str(top_event.get("top_signal", "")),
        "latest_top_signal": str(latest.get("top_signal", "")),
        "top_kind": str(top_event.get("kind", "")),
        "latest_kind": str(latest.get("kind", "")),
        "kind_counts": kind_counts,
    }


def text_checkpoint_reading_representation_summary(ckpt, max_entries=8):
    history = text_checkpoint_reading_history(ckpt)
    entries = history["entries"][-max(1, int(max_entries)):]
    progress_rows = [
        entry.get("representation_progress")
        for entry in entries
        if isinstance(entry.get("representation_progress"), dict)
        and bool(entry.get("representation_progress", {}).get("enabled", False))
    ]
    if not progress_rows:
        return {
            "enabled": False,
            "entry_count": 0,
            "insight_event_count": 0,
            "organization_score_delta": 0.0,
            "top_gain_signal": "",
        }
    latest = progress_rows[-1]
    return {
        "enabled": True,
        "entry_count": int(len(progress_rows)),
        "insight_event_count": sum(
            1 for row in progress_rows
            if bool(row.get("representation_insight_event", False))),
        "organization_score_delta": max(
            _mm_float(row.get("organization_score_delta", 0.0), 0.0)
            for row in progress_rows),
        "latest_organization_score_delta": _mm_float(
            latest.get("organization_score_delta", 0.0), 0.0),
        "top_gain_signal": str(latest.get("top_gain_signal", "")),
        "latest": latest,
    }


def text_checkpoint_priority_study_summary(ckpt, max_entries=8):
    history = text_checkpoint_reading_history(ckpt)
    entries = history["entries"][-max(1, int(max_entries)):]
    if not entries:
        return {
            "enabled": False,
            "entry_count": 0,
            "replay_study_entry_count": 0,
            "training_priority_entry_count": 0,
            "latest_replay_study_used": False,
            "latest_training_priority_sampling": False,
            "max_training_priority_record_count": 0,
            "max_training_priority_mean": 0.0,
            "max_training_priority_max": 0.0,
        }
    replay_entries = [
        entry for entry in entries
        if bool(entry.get("replay_study_used", False))]
    priority_entries = [
        entry for entry in entries
        if bool(entry.get("training_priority_sampling", False))]
    latest = entries[-1]
    return {
        "enabled": bool(replay_entries or priority_entries),
        "entry_count": int(len(entries)),
        "replay_study_entry_count": int(len(replay_entries)),
        "training_priority_entry_count": int(len(priority_entries)),
        "latest_replay_study_used": bool(
            latest.get("replay_study_used", False)),
        "latest_training_priority_sampling": bool(
            latest.get("training_priority_sampling", False)),
        "max_replay_study_records": max(
            (_mm_int(entry.get("replay_study_records", 0), 0)
             for entry in entries),
            default=0),
        "latest_replay_study_records": _mm_int(
            latest.get("replay_study_records", 0), 0),
        "max_study_record_count": max(
            (_mm_int(entry.get("study_record_count", 0), 0)
             for entry in entries),
            default=0),
        "latest_study_record_count": _mm_int(
            latest.get("study_record_count", 0), 0),
        "max_training_priority_record_count": max(
            (_mm_int(entry.get("training_priority_record_count", 0), 0)
             for entry in entries),
            default=0),
        "latest_training_priority_record_count": _mm_int(
            latest.get("training_priority_record_count", 0), 0),
        "max_training_priority_mean": max(
            (_mm_float(entry.get("training_priority_mean", 0.0), 0.0)
             for entry in entries),
            default=0.0),
        "latest_training_priority_mean": _mm_float(
            latest.get("training_priority_mean", 0.0), 0.0),
        "max_training_priority_max": max(
            (_mm_float(entry.get("training_priority_max", 0.0), 0.0)
             for entry in entries),
            default=0.0),
        "latest_training_priority_max": _mm_float(
            latest.get("training_priority_max", 0.0), 0.0),
    }


def multimodal_learning_history_from_payload(payload):
    if not isinstance(payload, dict):
        history = None
    else:
        history = payload.get("multimodal_learning_history")
        if history is None:
            report = payload.get("report")
            if isinstance(report, dict):
                history = report.get("multimodal_learning_history")
    if isinstance(history, dict):
        rows = history.get("entries", ())
    elif isinstance(history, list):
        rows = history
    else:
        rows = ()
    entries = [dict(row) for row in rows if isinstance(row, dict)]
    entries = entries[-MULTIMODAL_LEARNING_HISTORY_SIZE:]
    return {
        "version": MULTIMODAL_LEARNING_HISTORY_VERSION,
        "max_entries": MULTIMODAL_LEARNING_HISTORY_SIZE,
        "entry_count": len(entries),
        "entries": entries,
    }


def multimodal_score_digest(score_components):
    score_components = score_components if isinstance(score_components, dict) else {}
    scalar_keys = (
        "metric", "score", "all_score", "active_mean_score", "floor_score",
        "balanced_score", "mastery_score", "signal_coverage", "token_score",
        "exact_score", "generation_score", "generation_token_acc",
        "generation_exact", "generation_diversity", "generation_collapse_penalty",
        "mode_floor", "mode_floor_score", "mode_gap",
        "fer_score", "fer_raw_score", "bridge_score", "bridge_raw_score",
        "bridge_resolution", "bridge_connectivity", "gap_raw_score",
        "gap_resolution", "gap_target_mass", "connection_score",
        "sequence_score", "sequence_acc", "sequence_margin")
    skip_keys = (
        "token_skipped", "exact_skipped", "generation_skipped", "fer_skipped",
        "bridge_skipped", "connection_skipped", "sequence_skipped")
    digest = {}
    for key in scalar_keys:
        if key not in score_components:
            continue
        if key == "metric":
            digest[key] = str(score_components.get(key, ""))
        else:
            digest[key] = _mm_float(score_components.get(key, 0.0), 0.0)
    for key in skip_keys:
        if key in score_components:
            digest[key] = bool(score_components.get(key, False))
    return digest


def multimodal_representation_progress_digest(progress):
    progress = progress if isinstance(progress, dict) else {}
    if not bool(progress.get("enabled", False)):
        return {"enabled": False}
    signal_after = (
        progress.get("signal_after")
        if isinstance(progress.get("signal_after"), dict) else {})
    signal_deltas = (
        progress.get("signal_deltas")
        if isinstance(progress.get("signal_deltas"), dict) else {})
    return {
        "enabled": True,
        "active_signals": [
            str(signal) for signal in progress.get("active_signals", ())
            if str(signal) in MULTIMODAL_SELF_TEACH_SIGNALS],
        "signal_before": {
            str(signal): _mm_float(value, 0.0)
            for signal, value in (
                progress.get("signal_before")
                if isinstance(progress.get("signal_before"), dict) else {}
            ).items()
            if str(signal) in MULTIMODAL_SELF_TEACH_SIGNALS
        },
        "signal_after": {
            str(signal): _mm_float(value, 0.0)
            for signal, value in signal_after.items()
            if str(signal) in MULTIMODAL_SELF_TEACH_SIGNALS
        },
        "signal_deltas": {
            str(signal): _mm_float(value, 0.0)
            for signal, value in signal_deltas.items()
            if str(signal) in MULTIMODAL_SELF_TEACH_SIGNALS
        },
        "organization_score_before": _mm_float(
            progress.get("organization_score_before", 0.0), 0.0),
        "organization_score_after": _mm_float(
            progress.get("organization_score_after", 0.0), 0.0),
        "organization_score_delta": _mm_float(
            progress.get("organization_score_delta", 0.0), 0.0),
        "positive_signal_gain": _mm_float(
            progress.get("positive_signal_gain", 0.0), 0.0),
        "negative_signal_drift": _mm_float(
            progress.get("negative_signal_drift", 0.0), 0.0),
        "top_gain_signal": str(progress.get("top_gain_signal", "")),
        "top_regression_signal": str(progress.get("top_regression_signal", "")),
        "representation_insight_event": bool(
            progress.get("representation_insight_event", False)),
    }


def multimodal_learning_history_entry(report, session_index=None):
    report = report if isinstance(report, dict) else {}
    train_metrics = (
        report.get("train_metrics")
        if isinstance(report.get("train_metrics"), dict) else {})
    selection = (
        report.get("selection")
        if isinstance(report.get("selection"), dict) else {})
    if not selection and isinstance(train_metrics.get("selection"), dict):
        selection = train_metrics["selection"]
    rounds = [
        row for row in selection.get("rounds", ())
        if isinstance(row, dict)
    ]
    selected_round = _mm_int(selection.get("selected_round", -1), -1)
    selected_rows = [
        row for row in rounds
        if _mm_int(row.get("round", -2), -2) == selected_round
    ]
    selected_row = selected_rows[-1] if selected_rows else {}
    if not selected_row and rounds:
        selected_row = rounds[-1]
    first_row = rounds[0] if rounds else {}
    manifest = report.get("manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    architecture = report.get("architecture")
    architecture = architecture if isinstance(architecture, dict) else {}
    text_transfer = report.get("text_checkpoint_transfer")
    if not isinstance(text_transfer, dict):
        text_transfer = train_metrics.get("text_checkpoint_transfer")
    text_transfer = text_transfer if isinstance(text_transfer, dict) else {}
    multimodal_transfer = report.get("multimodal_checkpoint_transfer")
    if not isinstance(multimodal_transfer, dict):
        multimodal_transfer = train_metrics.get("multimodal_checkpoint_transfer")
    multimodal_transfer = (
        multimodal_transfer if isinstance(multimodal_transfer, dict) else {})
    progress = report.get("representation_progress")
    if not isinstance(progress, dict):
        progress = train_metrics.get("representation_progress")
    representation_progress = multimodal_representation_progress_digest(progress)
    learning_event = report.get("learning_event")
    if not isinstance(learning_event, dict):
        learning_event = train_metrics.get("learning_event")
    if not isinstance(learning_event, dict):
        learning_event = multimodal_learning_event_report(report)
    self_teach_reports = selection.get("self_teach_reports", ())
    if not self_teach_reports and isinstance(train_metrics.get("self_teach_plan"), dict):
        self_teach_reports = (train_metrics["self_teach_plan"],)
    self_teach_reports = [
        row for row in self_teach_reports if isinstance(row, dict)
    ]
    top_self_teach = self_teach_reports[0] if self_teach_reports else {}
    selected_bridge = selection.get("selected_bridge_insight")
    selected_bridge = selected_bridge if isinstance(selected_bridge, dict) else {}
    view_names = manifest.get("view_names", ())
    if isinstance(view_names, (str, bytes)) or not isinstance(view_names, (list, tuple)):
        view_names = ()
    entry = {
        "version": MULTIMODAL_LEARNING_HISTORY_VERSION,
        "session_index": _mm_int(session_index, 0),
        "experiment": str(report.get("experiment", "")),
        "steps": _mm_int(report.get("steps", 0), 0),
        "seed": _mm_int(report.get("seed", 0), 0),
        "manifest_path": str(manifest.get("path", "")),
        "record_count": _mm_int(manifest.get("records", 0), 0),
        "train_record_count": _mm_int(manifest.get("train_records", 0), 0),
        "eval_record_count": _mm_int(manifest.get("eval_records", 0), 0),
        "view_count": _mm_int(manifest.get("view_count", 0), 0),
        "view_names": [str(name) for name in view_names[:16]],
        "source_balance_w": _mm_float(
            manifest.get(
                "source_balance_w",
                train_metrics.get("training_source_balance_w", 0.0)),
            0.0),
        "source_balance_sampling": bool(
            train_metrics.get("training_source_balance_sampling", False)),
        "source_count": _mm_int(
            manifest.get("source_count",
                         train_metrics.get("training_source_count", 0)), 0),
        "max_source_record_frac": _mm_float(
            manifest.get(
                "max_source_record_frac",
                train_metrics.get("training_max_source_record_frac", 0.0)),
            0.0),
        "min_source_record_frac": _mm_float(
            manifest.get(
                "min_source_record_frac",
                train_metrics.get("training_min_source_record_frac", 0.0)),
            0.0),
        "d": _mm_int(architecture.get("d", 0), 0),
        "layers": _mm_int(architecture.get("layers", 0), 0),
        "heads": _mm_int(architecture.get("heads", 0), 0),
        "latent_concept_slots": _mm_int(
            architecture.get("latent_concept_slots", 0), 0),
        "latent_concept_topk": _mm_int(
            architecture.get("latent_concept_topk", 0), 0),
        "latent_concept_memory_size": _mm_int(
            architecture.get("latent_concept_memory_size", 0), 0),
        "text_checkpoint_transfer_copied": bool(
            text_transfer.get("copied", False)),
        "text_checkpoint_transfer_accepted": bool(
            text_transfer.get("accepted", True)),
        "text_checkpoint_learning_event_count": _mm_int(
            text_transfer.get("source_reading_learning_event_count", 0), 0),
        "text_checkpoint_learning_event_triggered_count": _mm_int(
            text_transfer.get(
                "source_reading_learning_event_triggered_count", 0), 0),
        "multimodal_checkpoint_transfer_copied": bool(
            multimodal_transfer.get("copied", False)),
        "source_learning_history_count": _mm_int(
            multimodal_transfer.get("source_learning_history_count", 0), 0),
        "source_learning_event_triggered_count": _mm_int(
            multimodal_transfer.get("source_learning_event_triggered_count", 0), 0),
        "selection_enabled": bool(selection.get("enabled", False)),
        "accepted_update": bool(selection.get("accepted_update", False)),
        "selected_round": (
            selected_round if selected_round >= 0 else None),
        "selected_by_score": bool(selection.get("selected_by_score", False)),
        "selected_by_insight": bool(selection.get("selected_by_insight", False)),
        "selected_score_delta": _mm_float(
            selection.get("selected_score_delta", 0.0), 0.0),
        "selected_bridge_insight_delta": _mm_float(
            selected_bridge.get("bridge_insight_delta", 0.0), 0.0),
        "selected_bridge_quality_gain": _mm_float(
            selected_bridge.get("bridge_quality_gain", 0.0), 0.0),
        "selected_bridge_connectivity_gain": _mm_float(
            selected_bridge.get("bridge_connectivity_gain", 0.0), 0.0),
        "weight_update_changed": bool(
            train_metrics.get("weight_update_changed",
                              learning_event.get("weight_update_changed", False))),
        "weight_update_changed_tensor_count": _mm_int(
            train_metrics.get(
                "weight_update_changed_tensor_count",
                learning_event.get("weight_update_changed_tensor_count", 0)), 0),
        "weight_update_changed_value_count": _mm_int(
            train_metrics.get(
                "weight_update_changed_value_count",
                learning_event.get("weight_update_changed_value_count", 0)), 0),
        "weight_update_max_abs_delta": _mm_float(
            train_metrics.get(
                "weight_update_max_abs_delta",
                learning_event.get("weight_update_max_abs_delta", 0.0)), 0.0),
        "attempted_weight_update_count": _mm_int(
            train_metrics.get(
                "attempted_weight_update_count",
                learning_event.get("attempted_weight_update_count", 0)), 0),
        "representation_progress": representation_progress,
        "representation_insight_event": bool(
            representation_progress.get("representation_insight_event", False)),
        "representation_organization_score_delta": _mm_float(
            representation_progress.get("organization_score_delta", 0.0), 0.0),
        "representation_top_gain_signal": str(
            representation_progress.get("top_gain_signal", "")),
        "learning_event": learning_event,
        "learning_event_triggered": bool(
            learning_event.get("triggered", False)),
        "learning_event_kind": str(learning_event.get("kind", "")),
        "learning_event_top_signal": str(learning_event.get("top_signal", "")),
        "learning_event_score": _mm_float(
            learning_event.get("event_score", 0.0), 0.0),
        "self_teach_top_signal": str(top_self_teach.get("top_signal", "")),
        "self_teach_active_signals": [
            str(item) for item in top_self_teach.get("active_signals", ())],
        "self_teach_history_prior_enabled": bool(
            top_self_teach.get("history_prior_enabled", False)),
        "self_teach_history_prior_entry_count": _mm_int(
            top_self_teach.get("history_prior_entry_count", 0), 0),
        "self_teach_history_prior_top_signal": str(
            top_self_teach.get("history_prior_top_signal", "")),
        "before_score_components": multimodal_score_digest(
            first_row.get("score_components")),
        "after_score_components": multimodal_score_digest(
            selected_row.get("score_components")),
    }
    if not entry["after_score_components"]:
        train_score = train_metrics.get("score_components")
        if isinstance(train_score, dict):
            entry["after_score_components"] = multimodal_score_digest(train_score)
    return entry


def multimodal_learning_history_with_entry(previous_history, report):
    previous = multimodal_learning_history_from_payload(
        {"multimodal_learning_history": previous_history})
    entries = list(previous["entries"])
    entries.append(multimodal_learning_history_entry(
        report, session_index=len(entries) + 1))
    entries = entries[-MULTIMODAL_LEARNING_HISTORY_SIZE:]
    return {
        "version": MULTIMODAL_LEARNING_HISTORY_VERSION,
        "max_entries": MULTIMODAL_LEARNING_HISTORY_SIZE,
        "entry_count": len(entries),
        "entries": entries,
    }


def multimodal_checkpoint_learning_history_summary(ckpt, max_entries=8):
    history = multimodal_learning_history_from_payload(ckpt)
    entries = history["entries"][-max(1, int(max_entries)):]
    if not entries:
        return {
            "enabled": False,
            "entry_count": 0,
            "event_count": 0,
            "triggered_count": 0,
            "latest_triggered": False,
            "max_event_score": 0.0,
            "latest_event_score": 0.0,
            "top_signal": "",
            "latest_top_signal": "",
            "top_kind": "",
            "latest_kind": "",
            "kind_counts": {},
            "signal_counts": {},
            "representation_entry_count": 0,
            "representation_insight_event_count": 0,
        }
    events = [
        entry.get("learning_event")
        for entry in entries
        if isinstance(entry.get("learning_event"), dict)
    ]
    triggered = [
        event for event in events
        if bool(event.get("triggered", False))
    ]
    latest_event = events[-1] if events else {}
    top_event = max(
        triggered,
        key=lambda event: _mm_float(event.get("event_score", 0.0), 0.0),
        default={})
    kind_counts = {}
    signal_counts = {}
    for event in triggered:
        kind = str(event.get("kind", ""))
        signal = str(event.get("top_signal", ""))
        if kind:
            kind_counts[kind] = int(kind_counts.get(kind, 0)) + 1
        if signal:
            signal_counts[signal] = int(signal_counts.get(signal, 0)) + 1
    representation_entries = [
        entry for entry in entries
        if isinstance(entry.get("representation_progress"), dict)
        and bool(entry.get("representation_progress", {}).get("enabled", False))
    ]
    latest = entries[-1]
    return {
        "enabled": True,
        "entry_count": int(len(entries)),
        "event_count": int(len(events)),
        "triggered_count": int(len(triggered)),
        "latest_triggered": bool(latest_event.get("triggered", False)),
        "max_event_score": max(
            (_mm_float(event.get("event_score", 0.0), 0.0)
             for event in triggered),
            default=0.0),
        "latest_event_score": _mm_float(
            latest_event.get("event_score", 0.0), 0.0),
        "top_signal": str(top_event.get("top_signal", "")),
        "latest_top_signal": str(latest_event.get("top_signal", "")),
        "top_kind": str(top_event.get("kind", "")),
        "latest_kind": str(latest_event.get("kind", "")),
        "kind_counts": kind_counts,
        "signal_counts": signal_counts,
        "latest_learning_event": latest_event,
        "latest_experiment": str(latest.get("experiment", "")),
        "representation_entry_count": int(len(representation_entries)),
        "representation_insight_event_count": sum(
            1 for entry in representation_entries
            if bool(entry.get("representation_insight_event", False))),
        "max_representation_organization_delta": max(
            (_mm_float(
                entry.get("representation_organization_score_delta", 0.0), 0.0)
             for entry in representation_entries),
            default=0.0),
        "latest_representation_organization_delta": _mm_float(
            latest.get("representation_organization_score_delta", 0.0), 0.0),
    }


def multimodal_text_history_self_teach_prior(ckpt, enabled=True,
                                             max_entries=8, decay=0.75):
    if not enabled:
        return {"enabled": False, "entry_count": 0, "signal_deficits": {}}
    history = text_checkpoint_reading_history(ckpt)
    concept_prior = text_checkpoint_concept_insight_prior(
        ckpt, enabled=True, max_entries=max_entries, decay=decay)
    representation_summary = text_checkpoint_reading_representation_summary(
        ckpt, max_entries=max_entries)
    priority_study_summary = text_checkpoint_priority_study_summary(
        ckpt, max_entries=max_entries)
    entries = history["entries"][-max(1, int(max_entries)):]
    decay = min(1.0, max(0.0, _mm_float(decay, 0.75)))
    weighted = {signal: 0.0 for signal in MULTIMODAL_SELF_TEACH_SIGNALS}
    weights = {signal: 0.0 for signal in MULTIMODAL_SELF_TEACH_SIGNALS}
    top_counts = {signal: 0 for signal in MULTIMODAL_SELF_TEACH_SIGNALS}
    for offset, entry in enumerate(reversed(entries)):
        recency_weight = decay ** offset
        score_components = entry.get("after_score_components")
        if isinstance(score_components, dict):
            for signal, score_keys in MULTIMODAL_TEXT_HISTORY_SIGNAL_KEYS.items():
                skip_key = f"{signal}_skipped"
                if bool(score_components.get(skip_key, False)):
                    continue
                qualities = [
                    min(1.0, max(0.0, _mm_float(score_components.get(score_key))))
                    for score_key in score_keys
                    if score_key in score_components
                ]
                if not qualities:
                    continue
                quality = min(qualities)
                weighted[signal] += recency_weight * max(0.0, 1.0 - quality)
                weights[signal] += recency_weight
        representation_progress = entry.get("representation_progress")
        if (isinstance(representation_progress, dict)
                and bool(representation_progress.get("enabled", False))):
            signal_after = representation_progress.get("signal_after")
            signal_after = signal_after if isinstance(signal_after, dict) else {}
            signal_deltas = representation_progress.get("signal_deltas")
            signal_deltas = signal_deltas if isinstance(signal_deltas, dict) else {}
            for source_signal, value in signal_after.items():
                signal = MULTIMODAL_TEXT_HISTORY_TOP_SIGNAL_MAP.get(
                    str(source_signal))
                if signal not in weights:
                    continue
                quality = min(1.0, max(0.0, _mm_float(value, 0.0)))
                quality_deficit = max(0.0, 1.0 - quality)
                regression_deficit = max(
                    0.0, -_mm_float(signal_deltas.get(source_signal, 0.0), 0.0))
                deficit = max(quality_deficit, regression_deficit)
                if deficit <= 0.0:
                    continue
                weighted[signal] += recency_weight * deficit
                weights[signal] += recency_weight
        learning_event = entry.get("learning_event")
        if (isinstance(learning_event, dict)
                and bool(learning_event.get("triggered", False))):
            signal = MULTIMODAL_TEXT_HISTORY_TOP_SIGNAL_MAP.get(
                str(learning_event.get("top_signal", "")))
            if signal in weights:
                event_score = min(1.0, max(
                    0.0, _mm_float(learning_event.get("event_score", 0.0))))
                if bool(entry.get("training_priority_sampling", False)):
                    priority_gain = min(1.0, max(
                        0.0,
                        _mm_float(entry.get("training_priority_mean", 0.0), 0.0),
                        _mm_float(entry.get("training_priority_max", 0.0), 0.0) * 0.1,
                    ))
                    event_score = min(1.0, event_score * (1.0 + 0.25 * priority_gain))
                if event_score > 0.0:
                    weighted[signal] += recency_weight * event_score
                    weights[signal] += recency_weight
        source_top = str(entry.get("self_teach_top_signal", ""))
        top_signal = MULTIMODAL_TEXT_HISTORY_TOP_SIGNAL_MAP.get(source_top)
        if top_signal in top_counts:
            top_counts[top_signal] += 1
    deficits = {
        signal: float(weighted[signal] / weights[signal])
        for signal in MULTIMODAL_SELF_TEACH_SIGNALS
        if weights[signal] > 0.0 and weighted[signal] > 0.0
    }
    concept_signal = max(
        0.0, _mm_float(concept_prior.get("concept_connection_signal", 0.0)))
    if concept_signal > 0.0:
        deficits["connection"] = max(
            float(deficits.get("connection", 0.0)), float(concept_signal))
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
        "reading_representation_summary": representation_summary,
        "reading_priority_study_summary": priority_study_summary,
        "top_signal_counts": {
            signal: count for signal, count in top_counts.items() if count},
    }


def _multimodal_checkpoint_report(ckpt):
    if not isinstance(ckpt, dict):
        return {}
    report = ckpt.get("report")
    return report if isinstance(report, dict) else {}


def _multimodal_prior_signal_deficits(progress=None, learning_event=None):
    progress = progress if isinstance(progress, dict) else {}
    learning_event = learning_event if isinstance(learning_event, dict) else {}
    progress_enabled = bool(progress.get("enabled", False))
    signal_after = (
        progress.get("signal_after")
        if progress_enabled and isinstance(progress.get("signal_after"), dict)
        else {})
    signal_deltas = (
        progress.get("signal_deltas")
        if progress_enabled and isinstance(progress.get("signal_deltas"), dict)
        else {})
    deficits = {}
    if progress_enabled:
        for signal in ("fer", "bridge", "connection", "sequence"):
            if signal not in MULTIMODAL_SELF_TEACH_SIGNALS:
                continue
            after_quality = min(1.0, max(0.0, _mm_float(
                signal_after.get(signal), 0.0)))
            quality_deficit = max(0.0, 1.0 - after_quality)
            regression_deficit = max(
                0.0, -_mm_float(signal_deltas.get(signal), 0.0))
            deficit = max(quality_deficit, regression_deficit)
            if deficit > 0.0:
                deficits[signal] = float(deficit)
    if bool(learning_event.get("triggered", False)):
        signal = str(learning_event.get("top_signal", ""))
        if signal in MULTIMODAL_SELF_TEACH_SIGNALS:
            deficits[signal] = max(
                float(deficits.get(signal, 0.0)),
                min(1.0, max(0.0, _mm_float(
                    learning_event.get("event_score", 0.0), 0.0))))
    return deficits


def multimodal_checkpoint_representation_self_teach_prior(
        ckpt, enabled=True, max_entries=8, decay=0.75):
    if not enabled:
        return {"enabled": False, "entry_count": 0, "signal_deficits": {}}
    history = multimodal_learning_history_from_payload(ckpt)
    entries = history["entries"][-max(1, int(max_entries)):]
    if entries:
        decay = min(1.0, max(0.0, _mm_float(decay, 0.75)))
        weighted = {signal: 0.0 for signal in MULTIMODAL_SELF_TEACH_SIGNALS}
        weights = {signal: 0.0 for signal in MULTIMODAL_SELF_TEACH_SIGNALS}
        representation_entries = []
        latest_progress = {}
        latest_event = {}
        organization_delta = 0.0
        insight_event = False
        for offset, entry in enumerate(reversed(entries)):
            recency_weight = decay ** offset
            progress = entry.get("representation_progress")
            if isinstance(progress, dict):
                if bool(progress.get("enabled", False)):
                    representation_entries.append(entry)
                    latest_progress = progress if not latest_progress else latest_progress
                    organization_delta = max(
                        organization_delta,
                        _mm_float(
                            progress.get("organization_score_delta", 0.0), 0.0))
                    insight_event = bool(
                        insight_event
                        or progress.get("representation_insight_event", False))
            event = entry.get("learning_event")
            if isinstance(event, dict) and not latest_event:
                latest_event = event
            deficits = _multimodal_prior_signal_deficits(progress, event)
            for signal, deficit in deficits.items():
                weighted[signal] += recency_weight * max(0.0, float(deficit))
                weights[signal] += recency_weight
        deficits = {
            signal: float(weighted[signal] / weights[signal])
            for signal in MULTIMODAL_SELF_TEACH_SIGNALS
            if weights[signal] > 0.0 and weighted[signal] > 0.0
        }
        top_signal = None
        if deficits:
            top_signal = max(deficits.items(), key=lambda item: item[1])[0]
        summary = multimodal_checkpoint_learning_history_summary(
            ckpt, max_entries=max_entries)
        return {
            "enabled": bool(deficits),
            "entry_count": int(len(entries)),
            "signal_deficits": deficits,
            "top_signal": top_signal,
            "source": "multimodal_checkpoint_learning_history",
            "organization_score_before": _mm_float(
                latest_progress.get("organization_score_before", 0.0), 0.0),
            "organization_score_after": _mm_float(
                latest_progress.get("organization_score_after", 0.0), 0.0),
            "organization_score_delta": float(organization_delta),
            "representation_insight_event": bool(insight_event),
            "progress": latest_progress,
            "learning_event": latest_event,
            "learning_history_summary": summary,
        }
    report = _multimodal_checkpoint_report(ckpt)
    train_metrics = report.get("train_metrics")
    learning_event = report.get("learning_event")
    if not isinstance(learning_event, dict) and isinstance(train_metrics, dict):
        learning_event = train_metrics.get("learning_event")
    progress = report.get("representation_progress")
    if not isinstance(progress, dict) and isinstance(train_metrics, dict):
        progress = train_metrics.get("representation_progress")
    deficits = _multimodal_prior_signal_deficits(progress, learning_event)
    top_signal = None
    if deficits:
        top_signal = max(deficits.items(), key=lambda item: item[1])[0]
    return {
        "enabled": bool(deficits),
        "entry_count": 1,
        "signal_deficits": deficits,
        "top_signal": top_signal,
        "source": "multimodal_checkpoint_representation_progress",
        "organization_score_before": _mm_float(
            (progress or {}).get("organization_score_before", 0.0), 0.0),
        "organization_score_after": _mm_float(
            (progress or {}).get("organization_score_after", 0.0), 0.0),
        "organization_score_delta": _mm_float(
            (progress or {}).get("organization_score_delta", 0.0), 0.0),
        "representation_insight_event": bool(
            (progress or {}).get("representation_insight_event", False)),
        "progress": progress if isinstance(progress, dict) else {},
        "learning_event": learning_event if isinstance(learning_event, dict) else {},
    }


def merge_multimodal_self_teach_priors(*priors):
    merged_deficits = {}
    enabled = False
    entry_count = 0
    sources = []
    concept_connection_signal = 0.0
    for prior in priors:
        if not isinstance(prior, dict):
            continue
        if bool(prior.get("enabled", False)):
            enabled = True
        entry_count += int(prior.get("entry_count", 0) or 0)
        source = prior.get("source")
        if source:
            sources.append(str(source))
        deficits = prior.get("signal_deficits")
        if isinstance(deficits, dict):
            for signal, value in deficits.items():
                if signal not in MULTIMODAL_SELF_TEACH_SIGNALS:
                    continue
                merged_deficits[signal] = max(
                    float(merged_deficits.get(signal, 0.0)),
                    max(0.0, _mm_float(value, 0.0)))
        concept_connection_signal = max(
            concept_connection_signal,
            max(0.0, _mm_float(prior.get("concept_connection_signal", 0.0), 0.0)))
    if concept_connection_signal > 0.0:
        merged_deficits["connection"] = max(
            float(merged_deficits.get("connection", 0.0)),
            float(concept_connection_signal))
    top_signal = None
    if merged_deficits:
        top_signal = max(merged_deficits.items(), key=lambda item: item[1])[0]
    return {
        "enabled": bool(enabled or merged_deficits),
        "entry_count": int(entry_count),
        "signal_deficits": merged_deficits,
        "top_signal": top_signal,
        "concept_connection_signal": float(concept_connection_signal),
        "sources": sources,
    }


def multimodal_objective_profile_kwargs(objective_profile="manual", **kwargs):
    """Apply schema-free multimodal objective floors for a training posture."""
    objective_profile = str(objective_profile)
    if objective_profile not in MULTIMODAL_OBJECTIVE_PROFILES:
        raise ValueError(f"unknown multimodal objective profile {objective_profile!r}")
    effective = dict(kwargs)
    updates = {}
    floors = {}
    if objective_profile in ("language", "mastery"):
        floors.update(MULTIMODAL_LANGUAGE_OBJECTIVE_FLOORS)
    if objective_profile == "mastery":
        floors.update(MULTIMODAL_MASTERY_PROFILE_FLOORS)
    if floors:
        missing = [
            key for key in floors
            if key not in effective]
        if missing:
            raise ValueError(
                "missing multimodal objective profile controls: "
                + ", ".join(sorted(missing)))
        for key, floor in floors.items():
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
        "language_first": objective_profile == "language",
        "language_floors": dict(MULTIMODAL_LANGUAGE_OBJECTIVE_FLOORS)
        if objective_profile in ("language", "mastery") else {},
        "floors": dict(MULTIMODAL_MASTERY_PROFILE_FLOORS)
        if objective_profile == "mastery" else {},
        "objective_floors": dict(MULTIMODAL_MASTERY_OBJECTIVE_FLOORS)
        if objective_profile == "mastery" else {},
        "study_floors": dict(MULTIMODAL_MASTERY_STUDY_FLOORS)
        if objective_profile == "mastery" else {},
        "updates": updates,
        "applied": bool(updates),
    }
    effective["objective_profile"] = objective_profile
    effective["objective_profile_report"] = report
    return effective


def import_text_checkpoint(model, vocab, checkpoint, device=DEV):
    """Warm-start the generic text trunk and latent concept modules by token identity."""
    ckpt = load_text_checkpoint_payload(checkpoint, device=device)
    state = ckpt["state_dict"]
    src_vocab = {tok: i for i, tok in enumerate(ckpt["vocab"])}
    dst_state = model.state_dict()
    report_payload = ckpt.get("report") if isinstance(ckpt.get("report"), dict) else {}
    history = text_checkpoint_reading_history(ckpt)
    concept_prior = text_checkpoint_concept_insight_prior(ckpt)
    weight_update = text_checkpoint_weight_update_summary(ckpt)
    learning_event = text_checkpoint_learning_event_summary(ckpt)
    reading_representation = text_checkpoint_reading_representation_summary(ckpt)
    priority_study = text_checkpoint_priority_study_summary(ckpt)
    latest_history = history["entries"][-1] if history["entries"] else {}
    replay_bank = ckpt.get("reading_replay_bank")
    if not isinstance(replay_bank, dict):
        replay_bank = report_payload.get("reading_replay_bank")
    if not isinstance(replay_bank, dict):
        replay_bank = {}
    report = {
        "checkpoint": checkpoint,
        "checkpoint_experiment": report_payload.get("experiment"),
        "source_vocab_size": len(src_vocab),
        "target_vocab_size": len(vocab),
        "source_d": int(ckpt.get("d", 0) or 0),
        "target_d": int(model.config["d"]),
        "source_latent_concept_slots": int(ckpt.get("latent_concept_slots", 0)),
        "target_latent_concept_slots": int(model.latent_concept_slots),
        "source_latent_concept_topk": int(ckpt.get("latent_concept_topk", 0)),
        "target_latent_concept_topk": int(model.latent_concept_topk),
        "source_latent_concept_memory_size": int(
            ckpt.get("latent_concept_memory_size", 0)),
        "target_latent_concept_memory_size": int(model.latent_concept_memory_size),
        "source_reading_mastery_history_count": int(
            history.get("entry_count", 0) or 0),
        "source_reading_mastery_latest_experiment": latest_history.get("experiment"),
        "source_reading_mastery_latest_self_teach_signal": str(
            latest_history.get("self_teach_top_signal", "")),
        "source_reading_concept_connection_signal": float(
            concept_prior.get("concept_connection_signal", 0.0)),
        "source_reading_concept_insight_max_delta": float(
            concept_prior.get("max_concept_insight_delta", 0.0)),
        "source_reading_concept_insight_selected_count": int(
            concept_prior.get("selected_by_insight_count", 0)),
        "source_reading_concept_insight_accepted_count": int(
            concept_prior.get("accepted_update_count", 0)),
        "source_reading_concept_insight_prior": concept_prior,
        "source_reading_weight_update_changed": bool(
            weight_update.get("changed", False)),
        "source_reading_weight_update_latest_changed": bool(
            weight_update.get("latest_changed", False)),
        "source_reading_weight_update_changed_tensor_count": int(
            weight_update.get("changed_tensor_count", 0)),
        "source_reading_weight_update_changed_value_count": int(
            weight_update.get("changed_value_count", 0)),
        "source_reading_weight_update_max_abs_delta": float(
            weight_update.get("max_abs_delta", 0.0)),
        "source_reading_weight_update_latest_max_abs_delta": float(
            weight_update.get("latest_max_abs_delta", 0.0)),
        "source_reading_attempted_weight_update_count": int(
            weight_update.get("attempted_weight_update_count", 0)),
        "source_reading_weight_update_summary": weight_update,
        "source_reading_learning_event_triggered": bool(
            learning_event.get("triggered_count", 0)),
        "source_reading_learning_event_latest_triggered": bool(
            learning_event.get("latest_triggered", False)),
        "source_reading_learning_event_count": int(
            learning_event.get("event_count", 0)),
        "source_reading_learning_event_triggered_count": int(
            learning_event.get("triggered_count", 0)),
        "source_reading_learning_event_top_signal": str(
            learning_event.get("top_signal", "")),
        "source_reading_learning_event_latest_top_signal": str(
            learning_event.get("latest_top_signal", "")),
        "source_reading_learning_event_top_kind": str(
            learning_event.get("top_kind", "")),
        "source_reading_learning_event_latest_kind": str(
            learning_event.get("latest_kind", "")),
        "source_reading_learning_event_max_score": float(
            learning_event.get("max_event_score", 0.0)),
        "source_reading_learning_event_latest_score": float(
            learning_event.get("latest_event_score", 0.0)),
        "source_reading_learning_event_summary": learning_event,
        "source_reading_representation_progress_enabled": bool(
            reading_representation.get("enabled", False)),
        "source_reading_representation_insight_event_count": int(
            reading_representation.get("insight_event_count", 0)),
        "source_reading_representation_organization_delta": float(
            reading_representation.get("organization_score_delta", 0.0)),
        "source_reading_representation_latest_organization_delta": float(
            reading_representation.get("latest_organization_score_delta", 0.0)),
        "source_reading_representation_top_gain_signal": str(
            reading_representation.get("top_gain_signal", "")),
        "source_reading_representation_summary": reading_representation,
        "source_reading_priority_study_enabled": bool(
            priority_study.get("enabled", False)),
        "source_reading_replay_study_entry_count": int(
            priority_study.get("replay_study_entry_count", 0)),
        "source_reading_training_priority_entry_count": int(
            priority_study.get("training_priority_entry_count", 0)),
        "source_reading_latest_replay_study_used": bool(
            priority_study.get("latest_replay_study_used", False)),
        "source_reading_latest_training_priority_sampling": bool(
            priority_study.get("latest_training_priority_sampling", False)),
        "source_reading_training_priority_record_count": int(
            priority_study.get("max_training_priority_record_count", 0)),
        "source_reading_training_priority_mean": float(
            priority_study.get("max_training_priority_mean", 0.0)),
        "source_reading_training_priority_max": float(
            priority_study.get("max_training_priority_max", 0.0)),
        "source_reading_priority_study_summary": priority_study,
        "source_reading_replay_bank_records": _mm_int(
            replay_bank.get("record_count", 0), 0),
        "source_reading_replay_priority_records": _mm_int(
            replay_bank.get("priority_record_count", 0), 0),
        "copied_token_embeddings": 0,
        "overlap_tokens": 0,
        "copied_position_rows": 0,
        "copied_text_tensors": [],
        "copied_latent_tensors": [],
        "copied_sequence_tensors": [],
        "copied_completion_tensors": [],
        "skipped_shape": [],
        "skipped_missing": [],
    }
    src_emb = state.get("txt.emb.weight")
    if src_emb is not None and src_emb.ndim == 2:
        report["overlap_tokens"] = sum(1 for tok in vocab.stoi if tok in src_vocab)
        dst_emb = model.txt.emb.weight
        if src_emb.shape[1] == dst_emb.shape[1]:
            with torch.no_grad():
                for tok, dst_idx in vocab.stoi.items():
                    src_idx = src_vocab.get(tok)
                    if src_idx is None or src_idx >= src_emb.shape[0]:
                        continue
                    dst_emb[dst_idx].copy_(
                        src_emb[src_idx].to(device=dst_emb.device, dtype=dst_emb.dtype))
                    report["copied_token_embeddings"] += 1
        else:
            report["skipped_shape"].append({
                "name": "txt.emb.weight",
                "source": list(src_emb.shape),
                "target": list(dst_emb.shape),
            })
    src_pos = state.get("txt.pos.weight")
    if src_pos is not None and src_pos.ndim == 2:
        dst_pos = model.txt.pos.weight
        if src_pos.shape[1] == dst_pos.shape[1]:
            rows = min(src_pos.shape[0], dst_pos.shape[0])
            with torch.no_grad():
                dst_pos[:rows].copy_(
                    src_pos[:rows].to(device=dst_pos.device, dtype=dst_pos.dtype))
            report["copied_position_rows"] = int(rows)
        else:
            report["skipped_shape"].append({
                "name": "txt.pos.weight",
                "source": list(src_pos.shape),
                "target": list(dst_pos.shape),
            })
    text_prefixes = ("txt.enc.", "txt.blocks.", "txt.ln.")
    latent_prefixes = ("latent_concepts.", "latent_concept_memory.")
    sequence_prefix = "reading_predictor."
    completion_prefix = "reading_completion_predictor."
    has_completion_prefix = any(
        name.startswith(completion_prefix) for name in state)
    memory_key = "latent_concept_memory.memory"
    src_memory = state.get(memory_key)
    dst_memory = dst_state.get(memory_key)
    memory_compatible = (
        src_memory is not None and dst_memory is not None
        and tuple(src_memory.shape) == tuple(dst_memory.shape))
    with torch.no_grad():
        for name, src_val in sorted(state.items()):
            if name in ("txt.emb.weight", "txt.pos.weight"):
                continue
            if name.startswith(text_prefixes):
                dst_name = name
                copied_key = "copied_text_tensors"
            elif name.startswith(latent_prefixes):
                dst_name = name
                copied_key = "copied_latent_tensors"
            elif name.startswith(completion_prefix):
                dst_name = (
                    "concept_completion_predictor."
                    + name[len(completion_prefix):])
                copied_key = "copied_completion_tensors"
            elif name.startswith(sequence_prefix):
                dst_name = "concept_sequence_predictor." + name[len(sequence_prefix):]
                copied_key = "copied_sequence_tensors"
            else:
                continue
            dst_val = dst_state.get(dst_name)
            if dst_val is None:
                report["skipped_missing"].append(dst_name)
                continue
            if (name.startswith("latent_concept_memory.")
                    and name != memory_key
                    and not memory_compatible):
                report["skipped_shape"].append({
                    "name": name,
                    "source": list(src_val.shape),
                    "target": list(dst_val.shape),
                    "requires": memory_key,
                })
                continue
            if tuple(src_val.shape) != tuple(dst_val.shape):
                report["skipped_shape"].append({
                    "name": dst_name,
                    "source": list(src_val.shape),
                    "target": list(dst_val.shape),
                })
                continue
            dst_val.copy_(src_val.to(device=dst_val.device, dtype=dst_val.dtype))
            report[copied_key].append(dst_name)
        if not has_completion_prefix:
            for name, src_val in sorted(state.items()):
                if not name.startswith(sequence_prefix):
                    continue
                dst_name = (
                    "concept_completion_predictor."
                    + name[len(sequence_prefix):])
                dst_val = dst_state.get(dst_name)
                if dst_val is None:
                    report["skipped_missing"].append(dst_name)
                    continue
                if tuple(src_val.shape) != tuple(dst_val.shape):
                    report["skipped_shape"].append({
                        "name": dst_name,
                        "source": list(src_val.shape),
                        "target": list(dst_val.shape),
                    })
                    continue
                dst_val.copy_(src_val.to(
                    device=dst_val.device, dtype=dst_val.dtype))
                report["copied_completion_tensors"].append(dst_name)
    report["copied_text_tensor_count"] = len(report["copied_text_tensors"])
    report["copied_latent_tensor_count"] = len(report["copied_latent_tensors"])
    report["copied_sequence_tensor_count"] = len(report["copied_sequence_tensors"])
    report["copied_completion_tensor_count"] = len(
        report["copied_completion_tensors"])
    report["copied"] = bool(
        report["copied_token_embeddings"]
        or report["copied_position_rows"]
        or report["copied_text_tensor_count"]
        or report["copied_latent_tensor_count"]
        or report["copied_sequence_tensor_count"]
        or report["copied_completion_tensor_count"])
    model.text_checkpoint_transfer = report
    return report


def import_multimodal_checkpoint(model, vocab, checkpoint, device=DEV):
    """Warm-start a multimodal model from compatible prior multimodal weights."""
    ckpt = load_multimodal_checkpoint_payload(checkpoint, device=device)
    state = ckpt["state_dict"]
    src_vocab = {tok: i for i, tok in enumerate(ckpt["vocab"])}
    dst_state = model.state_dict()
    src_config = ckpt.get("model_config") if isinstance(
        ckpt.get("model_config"), dict) else {}
    report_payload = _multimodal_checkpoint_report(ckpt)
    representation_prior = multimodal_checkpoint_representation_self_teach_prior(
        ckpt, enabled=True)
    learning_history = multimodal_learning_history_from_payload(ckpt)
    learning_history_summary = multimodal_checkpoint_learning_history_summary(ckpt)
    source_learning_event = report_payload.get("learning_event")
    if not isinstance(source_learning_event, dict):
        train_metrics = report_payload.get("train_metrics")
        if isinstance(train_metrics, dict):
            source_learning_event = train_metrics.get("learning_event")
    if not isinstance(source_learning_event, dict):
        source_learning_event = learning_history_summary.get(
            "latest_learning_event", {})
    if not isinstance(source_learning_event, dict):
        source_learning_event = {}
    report = {
        "checkpoint": checkpoint,
        "checkpoint_experiment": report_payload.get("experiment"),
        "source_vocab_size": len(src_vocab),
        "target_vocab_size": len(vocab),
        "source_d": int(src_config.get("d", ckpt.get("d", 0)) or 0),
        "target_d": int(model.config["d"]),
        "source_view_names": list(src_config.get("view_names", [])),
        "target_view_names": list(model.config.get("view_names", [])),
        "source_representation_prior": representation_prior,
        "source_representation_insight_event": bool(
            representation_prior.get("representation_insight_event", False)),
        "source_representation_top_signal": representation_prior.get("top_signal"),
        "source_learning_event_triggered": bool(
            source_learning_event.get("triggered", False)),
        "source_learning_event_kind": str(
            source_learning_event.get("kind", "")),
        "source_learning_event_top_signal": str(
            source_learning_event.get("top_signal", "")),
        "source_learning_event_score": float(_mm_float(
            source_learning_event.get("event_score", 0.0), 0.0)),
        "source_learning_event_summary": source_learning_event,
        "source_learning_history_count": int(
            learning_history.get("entry_count", 0) or 0),
        "source_learning_event_count": int(
            learning_history_summary.get("event_count", 0)),
        "source_learning_event_triggered_count": int(
            learning_history_summary.get("triggered_count", 0)),
        "source_learning_event_max_score": float(
            learning_history_summary.get("max_event_score", 0.0)),
        "source_learning_event_latest_triggered": bool(
            learning_history_summary.get("latest_triggered", False)),
        "source_learning_event_latest_score": float(
            learning_history_summary.get("latest_event_score", 0.0)),
        "source_learning_event_history_top_signal": str(
            learning_history_summary.get("top_signal", "")),
        "source_learning_event_history_top_kind": str(
            learning_history_summary.get("top_kind", "")),
        "source_learning_history_summary": learning_history_summary,
        "copied_token_embeddings": 0,
        "overlap_tokens": 0,
        "copied_feature_tensors": [],
        "copied_exact_tensors": [],
        "skipped_shape": [],
        "skipped_missing": [],
    }

    copied_names = set()

    def copy_token_rows(name):
        src_val = state.get(name)
        dst_val = dst_state.get(name)
        if src_val is None or dst_val is None:
            return
        if src_val.ndim != 2 or dst_val.ndim != 2 or src_val.shape[1] != dst_val.shape[1]:
            report["skipped_shape"].append({
                "name": name,
                "source": list(src_val.shape),
                "target": list(dst_val.shape),
            })
            return
        copied = 0
        with torch.no_grad():
            dst_param = model.state_dict()[name]
            for tok, dst_idx in vocab.stoi.items():
                src_idx = src_vocab.get(tok)
                if src_idx is None or src_idx >= src_val.shape[0]:
                    continue
                dst_param[dst_idx].copy_(
                    src_val[src_idx].to(
                        device=dst_param.device, dtype=dst_param.dtype))
                copied += 1
        copied_names.add(name)
        report["copied_token_embeddings"] += int(copied)

    report["overlap_tokens"] = sum(1 for tok in vocab.stoi if tok in src_vocab)
    for token_name in ("lm.tok.weight", "lm.head.weight", "txt.emb.weight"):
        copy_token_rows(token_name)

    src_view_names = [str(name) for name in src_config.get("view_names", [])]
    dst_view_names = list(model.config.get("view_names", []))
    dst_by_view = {str(name): i for i, name in enumerate(dst_view_names)}
    view_prefix = "feature_readers."
    with torch.no_grad():
        for name, src_val in sorted(state.items()):
            mapped_name = None
            if name.startswith(view_prefix):
                parts = name.split(".", 2)
                if len(parts) >= 3:
                    try:
                        src_idx = int(parts[1])
                    except ValueError:
                        src_idx = None
                    if src_idx is not None and src_idx < len(src_view_names):
                        dst_idx = dst_by_view.get(src_view_names[src_idx])
                        if dst_idx is not None:
                            mapped_name = f"feature_readers.{dst_idx}.{parts[2]}"
            if mapped_name is None:
                continue
            dst_val = dst_state.get(mapped_name)
            if dst_val is None:
                report["skipped_missing"].append(mapped_name)
                continue
            if tuple(src_val.shape) != tuple(dst_val.shape):
                report["skipped_shape"].append({
                    "name": mapped_name,
                    "source": list(src_val.shape),
                    "target": list(dst_val.shape),
                })
                continue
            dst_val.copy_(src_val.to(device=dst_val.device, dtype=dst_val.dtype))
            copied_names.add(mapped_name)
            report["copied_feature_tensors"].append(mapped_name)

        for name, src_val in sorted(state.items()):
            if name in copied_names:
                continue
            if name in ("lm.tok.weight", "lm.head.weight", "txt.emb.weight"):
                continue
            if name.startswith(view_prefix):
                continue
            dst_val = dst_state.get(name)
            if dst_val is None:
                report["skipped_missing"].append(name)
                continue
            if tuple(src_val.shape) != tuple(dst_val.shape):
                report["skipped_shape"].append({
                    "name": name,
                    "source": list(src_val.shape),
                    "target": list(dst_val.shape),
                })
                continue
            dst_val.copy_(src_val.to(device=dst_val.device, dtype=dst_val.dtype))
            copied_names.add(name)
            report["copied_exact_tensors"].append(name)

    report["copied_feature_tensor_count"] = len(report["copied_feature_tensors"])
    report["copied_exact_tensor_count"] = len(report["copied_exact_tensors"])
    report["copied"] = bool(
        report["copied_token_embeddings"]
        or report["copied_feature_tensor_count"]
        or report["copied_exact_tensor_count"])
    model.multimodal_checkpoint_transfer = report
    return report


class ResidualMLPBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x):
        return x + self.net(x)


class FeaturePrefixTrunk(nn.Module):
    """Feature vector -> short sequence of decoder-width prefix embeddings."""

    def __init__(self, in_dim, d, n_tokens=4, view_index=0, n_views=2,
                 arch="mlp", width=128, depth=1):
        super().__init__()
        arch = str(arch)
        if arch not in TRUNK_ARCHES:
            raise ValueError(f"unknown multimodal trunk arch {arch!r}")
        self.in_dim = int(in_dim)
        self.n_tokens = int(n_tokens)
        self.arch = arch
        self.width = int(width)
        self.depth = int(depth)
        if self.in_dim <= 0 or self.n_tokens <= 0:
            raise ValueError("feature trunk dimensions must be positive")
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("feature trunk width/depth must be positive")
        layers = [nn.LayerNorm(self.in_dim), nn.Linear(self.in_dim, self.width), nn.GELU()]
        if arch == "residual":
            layers.extend(ResidualMLPBlock(self.width) for _ in range(self.depth))
        elif self.depth > 1:
            for _ in range(self.depth - 1):
                layers.extend([nn.Linear(self.width, self.width), nn.GELU()])
        layers.append(nn.Linear(self.width, self.n_tokens * d))
        self.net = nn.Sequential(*layers)
        self.view_embed = nn.Embedding(n_views, d)
        self.view_index = int(view_index)
        self.posn = nn.Parameter(torch.zeros(self.n_tokens, d))
        self.d = int(d)

    def forward(self, x):
        h = self.net(x).view(x.shape[0], self.n_tokens, self.d)
        return h + self.view_embed.weight[self.view_index] + self.posn[None]


class TextTrunk(nn.Module):
    """Token encoder -> fixed count of text prefix embeddings."""

    def __init__(self, vocab_size, d, pad=0, n_tokens=8, heads=4, layers=1,
                 view_index=1, n_views=2, max_len=64, arch="transformer"):
        super().__init__()
        self.pad = int(pad)
        self.view_index = int(view_index)
        self.n_tokens = int(n_tokens)
        self.layers = int(layers)
        self.arch = str(arch)
        if self.arch not in TEXT_TRUNK_ARCHES:
            raise ValueError(f"unknown text trunk architecture {self.arch!r}")
        if self.n_tokens <= 0 or self.layers <= 0:
            raise ValueError("text trunk token/layer counts must be positive")
        self.emb = nn.Embedding(vocab_size, d, padding_idx=pad)
        self.pos = nn.Embedding(max_len, d)
        if self.arch == "transformer":
            enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                             dropout=0.0, activation="gelu", batch_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=self.layers,
                                             enable_nested_tensor=False)
            self.blocks = None
        else:
            self.enc = None
            self.blocks = nn.ModuleList([
                CausalBlock(d, heads, arch=self.arch, vocab=vocab_size,
                            pos_mode="none", causal=False)
                for _ in range(self.layers)
            ])
        self.view_embed = nn.Embedding(n_views, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, ids):
        length = ids.shape[1]
        pos = torch.arange(length, device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        pad_mask = ids.eq(self.pad)
        h = self.emb(ids) + self.pos(pos)[None]
        if self.enc is not None:
            h = self.enc(h, src_key_padding_mask=pad_mask)
        else:
            for block in self.blocks:
                h = block(h, ids, pad_mask)
        h = self.ln(h + self.view_embed.weight[self.view_index])
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        if h.shape[1] > self.n_tokens:
            return h[:, :self.n_tokens]
        if h.shape[1] < self.n_tokens:
            pad = torch.zeros(h.shape[0], self.n_tokens - h.shape[1], h.shape[2],
                              dtype=h.dtype, device=h.device)
            h = torch.cat([h, pad], dim=1)
        return h


class ConceptFusion(nn.Module):
    """Shared latent-token mixer for feature/text prefixes before decoding."""

    def __init__(self, d, heads=4, concept_tokens=4, layers=1):
        super().__init__()
        self.concept_tokens = int(concept_tokens)
        self.layers = int(layers)
        if self.concept_tokens <= 0 or self.layers <= 0:
            raise ValueError("concept token/layer counts must be positive")
        self.queries = nn.Parameter(torch.randn(self.concept_tokens, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                         dropout=0.0, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=self.layers,
                                         enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def forward(self, feature_prefixes, tp):
        q = self.queries.unsqueeze(0).expand(tp.shape[0], -1, -1)
        h = torch.cat([q] + list(feature_prefixes) + [tp], dim=1)
        h = self.ln(self.enc(h))
        return h, h[:, :self.concept_tokens]


class MultimodalLM(nn.Module):
    """Generic manifest feature/text prefix readers feeding one token decoder."""

    def __init__(self, vocab_size, view_dims=None, d=96, layers=3, heads=4,
                 pad=0, max_len=128, view_tokens=4, txt_tokens=8,
                 trunk_arch="mlp", trunk_width=128, trunk_depth=1, text_layers=1,
                 modality_dropout=0.0, text_arch="transformer", concept_tokens=4,
                 fusion_layers=1, latent_concept_slots=0, latent_concept_layers=1,
                 latent_concept_prefix=False, latent_concept_memory_size=0,
                 latent_concept_topk=0):
        super().__init__()
        view_dims = OrderedDict((str(k), int(v)) for k, v in (view_dims or {}).items())
        if view_tokens <= 0 or txt_tokens <= 0:
            raise ValueError("multimodal prefix token counts must be positive")
        if any(dim <= 0 for dim in view_dims.values()):
            raise ValueError("feature view dimensions must be positive")
        if modality_dropout < 0.0 or modality_dropout > 1.0:
            raise ValueError("modality_dropout must be in [0, 1]")
        if int(latent_concept_slots) < 0:
            raise ValueError("latent concept slots must be non-negative")
        if int(latent_concept_slots) > 0 and int(latent_concept_layers) <= 0:
            raise ValueError("latent concept layers must be positive")
        if bool(latent_concept_prefix) and int(latent_concept_slots) <= 0:
            raise ValueError("latent concept prefix requires latent slots")
        if int(latent_concept_memory_size) < 0:
            raise ValueError("latent concept memory size must be non-negative")
        if int(latent_concept_memory_size) and int(latent_concept_slots) <= 0:
            raise ValueError("latent concept memory requires latent slots")
        trunk_arch = str(trunk_arch)
        text_arch = str(text_arch)
        view_names = list(view_dims.keys())
        n_views = len(view_names) + 1
        self.config = {
            "vocab_size": int(vocab_size), "view_dims": dict(view_dims),
            "view_names": list(view_names), "d": int(d), "layers": int(layers),
            "heads": int(heads), "pad": int(pad), "max_len": int(max_len),
            "view_tokens": int(view_tokens), "txt_tokens": int(txt_tokens),
            "trunk_arch": trunk_arch,
            "trunk_width": int(trunk_width), "trunk_depth": int(trunk_depth),
            "text_layers": int(text_layers), "text_arch": text_arch,
            "modality_dropout": float(modality_dropout),
            "fusion": "latent_prefix", "concept_tokens": int(concept_tokens),
            "fusion_layers": int(fusion_layers),
            "latent_concept_slots": int(latent_concept_slots),
            "latent_concept_layers": int(latent_concept_layers),
            "latent_concept_prefix": bool(latent_concept_prefix),
            "latent_concept_memory_size": int(latent_concept_memory_size),
            "latent_concept_topk": int(latent_concept_topk),
        }
        self.modality_dropout = float(modality_dropout)
        self.latent_concept_slots = int(latent_concept_slots)
        self.latent_concept_layers = int(latent_concept_layers)
        self.latent_concept_prefix = bool(latent_concept_prefix)
        self.latent_concept_memory_size = int(latent_concept_memory_size)
        self.latent_concept_topk = int(latent_concept_topk)
        self.view_names = tuple(view_names)
        self.lm = ScratchpadLM(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                               pad=pad, pointer=False, loop=False)
        self.feature_readers = nn.ModuleList([
            FeaturePrefixTrunk(
                dim, d, n_tokens=view_tokens, view_index=i, n_views=n_views,
                arch=trunk_arch, width=trunk_width, depth=trunk_depth)
            for i, dim in enumerate(view_dims.values())
        ])
        self.txt = TextTrunk(vocab_size, d, pad=pad, n_tokens=txt_tokens, heads=heads,
                             layers=text_layers, view_index=len(view_names),
                             n_views=n_views, arch=text_arch,
                             max_len=max_len)
        self.fusion = ConceptFusion(d, heads=heads, concept_tokens=concept_tokens,
                                    layers=fusion_layers)
        self.latent_concepts = (LatentConceptHead(
            self.latent_concept_slots, d, heads=heads,
            mixer_layers=self.latent_concept_layers, topk=self.latent_concept_topk)
            if self.latent_concept_slots > 0 else None)
        self.concept_sequence_predictor = (
            LatentConceptSequencePredictor(d)
            if self.latent_concept_slots > 0 else None)
        self.concept_completion_predictor = (
            LatentConceptSequencePredictor(d)
            if self.latent_concept_slots > 0 else None)
        self.latent_concept_memory = (LatentConceptMemory(
            self.latent_concept_memory_size, d)
            if self.latent_concept_memory_size > 0 else None)

    def _apply_modality_dropout(self, prefixes, tp):
        if not self.training or self.modality_dropout <= 0.0:
            return prefixes, tp
        keep = 1.0 - self.modality_dropout
        if keep <= 0.0:
            return [p * 0.0 for p in prefixes], tp * 0.0
        out = []
        for prefix in prefixes:
            mask = (
                torch.rand(prefix.shape[0], 1, 1, device=prefix.device)
                .lt(keep).to(prefix.dtype) / keep
            )
            out.append(prefix * mask)
        text_mask = (
            torch.rand(tp.shape[0], 1, 1, device=tp.device).lt(keep).to(tp.dtype) / keep
        )
        return out, tp * text_mask

    def encode_prefix(self, features, txt, mode="full"):
        prefixes = [reader(x) for reader, x in zip(self.feature_readers, features)]
        tp = self.txt(txt)
        if mode == "text_only":
            prefixes = [p * 0.0 for p in prefixes]
        elif mode == "sensor_only":
            tp = tp * 0.0
        elif mode != "full":
            raise ValueError(f"unknown multimodal mode {mode!r}")
        elif self.modality_dropout > 0.0:
            prefixes, tp = self._apply_modality_dropout(prefixes, tp)
        return self.fusion(prefixes, tp)

    def latent_concept_states_from_prefix(self, prefix, view_dropout=0.0, project=False):
        if self.latent_concepts is None:
            return None
        if self.training and view_dropout > 0.0:
            prefix = F.dropout(prefix, p=float(view_dropout), training=True)
        return self.latent_concepts(prefix, project=project)

    def latent_concept_states(self, features, txt, mode="full", view_dropout=0.0,
                              project=False):
        prefix, _concepts = self.encode_prefix(features, txt, mode=mode)
        return self.latent_concept_states_from_prefix(
            prefix, view_dropout=view_dropout, project=project)

    def decoder_prefix_from_encoded(self, prefix, latent_state_tensor=None):
        if self.latent_concept_prefix and self.latent_concepts is not None:
            if latent_state_tensor is None:
                latent_state_tensor = self.latent_concepts(prefix)
            prefix = torch.cat([latent_state_tensor, prefix], dim=1)
        return prefix

    def decoder_prefix(self, features, txt, mode="full"):
        prefix, _concepts = self.encode_prefix(features, txt, mode=mode)
        return self.decoder_prefix_from_encoded(prefix)

    def _empty_logits(self, ids, prefix):
        return prefix.new_zeros(
            (ids.shape[0], ids.shape[1], int(self.config["vocab_size"])))

    def mode_bundle(self, features, txt, ids, mode="full", need_latent=False,
                    latent_view_dropout=0.0, latent_project=False,
                    need_logits=True):
        prefix, concepts = self.encode_prefix(features, txt, mode=mode)
        latent_state_tensor = None
        if (self.latent_concept_prefix or need_latent) and self.latent_concepts is not None:
            latent_state_tensor = self.latent_concept_states_from_prefix(
                prefix, view_dropout=latent_view_dropout, project=latent_project)
        decoder_prefix = self.decoder_prefix_from_encoded(
            prefix, latent_state_tensor=latent_state_tensor)
        if need_logits and ids.shape[1] > 0:
            logits = self.lm(ids, prefix=decoder_prefix)[:, decoder_prefix.shape[1]:]
        else:
            logits = self._empty_logits(ids, decoder_prefix)
        out = {"logits": logits, "prefix": prefix, "concepts": concepts}
        if need_latent:
            out["latent_concepts"] = latent_state_tensor
        return out

    def forward(self, features, txt, ids, mode="full"):
        prefix = self.decoder_prefix(features, txt, mode=mode)
        if ids.shape[1] == 0:
            return self._empty_logits(ids, prefix)
        logits = self.lm(ids, prefix=prefix)
        return logits[:, prefix.shape[1]:]


def _decode_tokens_for_record(rec, decode_objective="target"):
    decode_objective = str(decode_objective)
    if decode_objective == "causal":
        sequence = tuple(rec.text) + tuple(rec.target)
        if not sequence:
            sequence = EMPTY_TEXT_TOKENS
        return sequence, EMPTY_TEXT_TOKENS, sequence
    if decode_objective != "target":
        raise ValueError(f"unknown multimodal decode objective {decode_objective!r}")
    return tuple(rec.text), tuple(rec.text), tuple(rec.target)


def _pack_token_rows(rows, vocab, device):
    max_len = max((len(row) for row in rows), default=0)
    out = torch.full(
        (len(rows), max_len), vocab.pad, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        if row:
            out[i, :len(row)] = torch.tensor(
                vocab.enc(row), dtype=torch.long, device=device)
    return out


def _record_feature_tensors(rec, view_dims, device):
    tensors = []
    for name, dim in view_dims.items():
        arr = rec.views.get(name)
        if arr is None:
            arr = np.zeros(int(dim), dtype=np.float32)
        tensors.append(torch.tensor(
            np.asarray(arr, dtype=np.float32)[None, :],
            dtype=torch.float32, device=device))
    return tensors


def _generation_next_id(logits, pad, temperature=0.0, top_k=0):
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
            scores >= floor, scores, scores.new_full(scores.shape, -float("inf")))
    probs = torch.softmax(scores / temperature, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
        return int(torch.argmax(logits.detach().float()).item())
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def multimodal_generate_tokens(
        model, vocab, prompt_tokens, view_dims=None, views=None,
        decode_text_tokens=EMPTY_TEXT_TOKENS, mode="full", max_new_tokens=32,
        temperature=0.0, top_k=0, device=DEV):
    """Autoregressively continue prompt tokens from the multimodal decoder."""
    prompt = tuple(prompt_tokens or ())
    if not prompt:
        raise ValueError("generation prompt must contain at least one token")
    view_dims = OrderedDict((str(k), int(v)) for k, v in (view_dims or {}).items())
    rec = MultimodalRecord(
        rec_id="generation", split="eval", text=tuple(decode_text_tokens or ()),
        target=(), views=dict(views or {}), meta={})
    features = _record_feature_tensors(rec, view_dims, device)
    txt = _pack_token_rows([tuple(decode_text_tokens or EMPTY_TEXT_TOKENS)], vocab, device)
    generated_ids = vocab.enc(prompt)
    prompt_len = len(generated_ids)
    max_context = max(1, int(model.config.get("max_len", 0) or len(generated_ids)))
    stop_tokens = {int(vocab.pad)}
    for _ in range(max(0, int(max_new_tokens))):
        context_ids = generated_ids[-max_context:]
        ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        logits = model(features, txt, ids, mode=mode)
        if logits.shape[1] == 0:
            break
        next_id = _generation_next_id(
            logits[0, -1], vocab.pad, temperature=temperature, top_k=top_k)
        if next_id in stop_tokens:
            break
        generated_ids.append(next_id)
    return tuple(vocab.dec(generated_ids[prompt_len:]))


def multimodal_generation_eval(
        model, records, vocab, view_dims, n=0, seed=1, device=DEV,
        mode="full", decode_objective="causal", prompt_tokens=16,
        max_new_tokens=32, temperature=0.0, top_k=0):
    """Free continuation probe for causal text windows from the eval manifest."""
    if str(decode_objective) != "causal":
        return {"enabled": False, "skipped": True,
                "skip_reason": "generation_eval_requires_causal_decode"}
    rng = np.random.default_rng(seed)
    candidates = [
        rec for rec in records
        if len(tuple(rec.text) + tuple(rec.target)) >= 2
    ]
    count = min(int(n), len(candidates)) if int(n) > 0 else len(candidates)
    sample = _sample_unique_records(candidates, count, rng) if count else []
    rows = []
    total_matches = total_gold = exact = 0
    model.eval()
    for rec in sample:
        sequence = tuple(rec.text) + tuple(rec.target)
        prompt_len = min(max(1, int(prompt_tokens)), len(sequence) - 1)
        prompt = sequence[:prompt_len]
        gold = sequence[prompt_len:prompt_len + max(0, int(max_new_tokens))]
        if not gold:
            continue
        generated = multimodal_generate_tokens(
            model, vocab, prompt, view_dims=view_dims, views=rec.views,
            decode_text_tokens=EMPTY_TEXT_TOKENS, mode=mode,
            max_new_tokens=len(gold), temperature=temperature, top_k=top_k,
            device=device)
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
        "mode": str(mode),
        "decode_objective": str(decode_objective),
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


def multimodal_prompt_generation_report(
        model, prompts, vocab, view_dims, mode="full", max_new_tokens=32,
        temperature=0.0, top_k=0, device=DEV):
    from .text import split_words

    rows = []
    model.eval()
    for prompt in prompts or ():
        prompt_tokens = tuple(split_words(str(prompt)))
        if not prompt_tokens:
            continue
        generated = multimodal_generate_tokens(
            model, vocab, prompt_tokens, view_dims=view_dims, views={},
            decode_text_tokens=EMPTY_TEXT_TOKENS, mode=mode,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, device=device)
        rows.append({
            "prompt": str(prompt),
            "prompt_tokens": list(prompt_tokens),
            "generated": list(generated),
        })
    generated_signatures = {tuple(row["generated"]) for row in rows}
    return {
        "enabled": bool(rows),
        "skipped": not bool(rows),
        "mode": str(mode),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "unique_generation_count": int(len(generated_signatures)),
        "all_generations_identical": (
            bool(len(rows) > 1 and len(generated_signatures) == 1)),
        "samples": rows,
    }


def _batch_from_records(records, vocab, device, view_dims,
                        decode_objective="target", return_decode_text=False):
    feature_rows = {name: [] for name in view_dims}
    texts, decode_texts, targets = [], [], []
    for rec in records:
        for name, dim in view_dims.items():
            feature_rows[name].append(
                rec.views.get(name, np.zeros(int(dim), dtype=np.float32)))
        latent_text, decode_text, target = _decode_tokens_for_record(
            rec, decode_objective=decode_objective)
        texts.append(latent_text)
        decode_texts.append(decode_text)
        targets.append(target)
    features = [
        torch.tensor(np.stack(feature_rows[name]), dtype=torch.float32, device=device)
        for name in view_dims
    ]
    txt = _pack_token_rows(texts, vocab, device)
    target = _pack_token_rows(targets, vocab, device)
    if not return_decode_text:
        return features, txt, target
    decode_txt = txt if decode_objective == "target" else _pack_token_rows(
        decode_texts, vocab, device)
    return features, txt, target, decode_txt


def _sample_records(records, n, rng, weights=None):
    if weights is None:
        idx = rng.integers(len(records), size=int(n))
    else:
        idx = rng.choice(len(records), size=int(n), replace=True, p=weights)
    return [records[int(i)] for i in idx]


def _sample_unique_records(records, n, rng):
    if not records:
        return []
    n = int(n)
    if n <= 0 or n >= len(records):
        return list(records)
    idx = rng.choice(len(records), size=n, replace=False)
    return [records[int(i)] for i in idx]


def _minmax_scale(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _sequence_order(rec, fallback):
    meta = rec.meta if isinstance(rec.meta, dict) else {}
    raw_order = meta.get(
        "chunk_index", meta.get("order", meta.get("position", fallback)))
    try:
        return float(raw_order)
    except (TypeError, ValueError):
        return float(fallback)


def _continuation_source(rec):
    meta = rec.meta if isinstance(rec.meta, dict) else {}
    return str(meta.get("source", meta.get("document", meta.get("dataset", ""))))


def infer_multimodal_decode_objective(records, requested="auto"):
    requested = str(requested)
    if requested not in DECODE_OBJECTIVES:
        raise ValueError(f"unknown multimodal decode objective {requested!r}")
    if requested != "auto":
        return requested
    groups = {}
    for pos, rec in enumerate(records or ()):
        if rec.views or not rec.text or not rec.target:
            continue
        groups.setdefault(_continuation_source(rec), []).append(
            (_sequence_order(rec, pos), pos, rec))
    considered = matches = 0
    for rows in groups.values():
        rows = sorted(rows, key=lambda row: (row[0], row[1]))
        for (_a_order, _a_pos, a), (_b_order, _b_pos, b) in zip(rows, rows[1:]):
            considered += 1
            if tuple(a.target) == tuple(b.text):
                matches += 1
    if considered and matches / float(considered) >= 0.5:
        return "causal"
    targetless_text_records = sum(
        1 for rec in records or () if rec.text and not rec.target)
    target_records = sum(1 for rec in records or () if rec.target)
    if targetless_text_records and target_records == 0:
        return "causal"
    return "target"


def multimodal_decode_objective_report(records, requested, resolved):
    groups = {}
    considered = matches = 0
    for pos, rec in enumerate(records or ()):
        if rec.views or not rec.text or not rec.target:
            continue
        groups.setdefault(_continuation_source(rec), []).append(
            (_sequence_order(rec, pos), pos, rec))
    for rows in groups.values():
        rows = sorted(rows, key=lambda row: (row[0], row[1]))
        for (_a_order, _a_pos, a), (_b_order, _b_pos, b) in zip(rows, rows[1:]):
            considered += 1
            if tuple(a.target) == tuple(b.text):
                matches += 1
    target_records = sum(1 for rec in records or () if rec.target)
    text_records = sum(1 for rec in records or () if rec.text)
    targetless_text_records = sum(
        1 for rec in records or () if rec.text and not rec.target)
    view_records = sum(1 for rec in records or () if rec.views)
    causal_reason = ""
    if resolved == "causal":
        if considered and matches / float(considered) >= 0.5:
            causal_reason = "continuation"
        elif targetless_text_records and target_records == 0:
            causal_reason = "targetless_text"
    return {
        "requested": str(requested),
        "resolved": str(resolved),
        "causal_reason": causal_reason,
        "text_records": int(text_records),
        "target_records": int(target_records),
        "targetless_text_records": int(targetless_text_records),
        "view_records": int(view_records),
        "continuation_candidates": int(considered),
        "continuation_matches": int(matches),
        "continuation_match_rate": (
            float(matches) / float(considered) if considered else 0.0),
    }


def mine_multimodal_sequence_pairs(records, split="train", n=0, seed=0):
    """Pair adjacent multimodal records from source/order metadata."""
    groups = {}
    total = 0
    for pos, rec in enumerate(records):
        if rec.split != split:
            continue
        total += 1
        meta = rec.meta if isinstance(rec.meta, dict) else {}
        source = str(meta.get("source", meta.get("document", "__records__")))
        groups.setdefault(source, []).append((_sequence_order(rec, pos), pos, rec))
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


def _sample_pairs(pairs, n, rng):
    if not pairs:
        return []
    return [pairs[int(rng.integers(len(pairs)))] for _ in range(int(n))]


def token_loss(logits, ids, pad):
    if ids.shape[1] < 2:
        return logits.sum() * 0.0
    targets = ids[:, 1:]
    mask = targets.ne(pad)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    raw = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                          targets.reshape(-1), reduction="none").view_as(targets)
    return raw.masked_select(mask).mean()


def repetition_unlikelihood_loss(logits_by_mode, ids, pad, window=32):
    """Penalize recent non-gold repeats that create free-generation attractors."""
    if not logits_by_mode:
        zero = ids.float().sum() * 0.0
        return zero, {
            "enabled": False, "skipped": True, "skip_reason": "empty_logits",
            "tokens": 0, "candidates": 0, "window": int(window),
        }
    first_logits = next(iter(logits_by_mode.values()))
    zero = first_logits.sum() * 0.0
    if ids.shape[1] < 2 or int(window) <= 0:
        return zero, {
            "enabled": False, "skipped": True, "skip_reason": "too_short",
            "tokens": 0, "candidates": 0, "window": int(window),
        }
    targets = ids[:, 1:]
    valid = targets.ne(int(pad))
    if not bool(valid.any()):
        return zero, {
            "enabled": False, "skipped": True, "skip_reason": "empty_targets",
            "tokens": 0, "candidates": 0, "window": int(window),
        }
    losses = []
    window = max(1, int(window))
    max_steps = logits_by_mode[next(iter(logits_by_mode))].shape[1] - 1
    max_steps = min(max_steps, ids.shape[1] - 1)
    if max_steps <= 0:
        return zero, {
            "enabled": False, "skipped": True, "skip_reason": "too_short",
            "tokens": 0, "candidates": 0, "window": int(window),
        }
    targets = ids[:, 1:1 + max_steps]
    valid = targets.ne(int(pad))
    candidate_any = torch.zeros_like(valid, dtype=torch.bool)
    candidate_total = valid.new_zeros((), dtype=torch.long)
    for _mode, logits in logits_by_mode.items():
        probs = torch.softmax(logits[:, :max_steps].float(), dim=-1)
        mode_losses = []
        for back in range(window):
            candidates = ids.new_full(targets.shape, int(pad))
            width = max_steps - back
            if width <= 0:
                break
            candidates[:, back:] = ids[:, :width]
            mask = valid & candidates.ne(int(pad)) & candidates.ne(targets)
            if not bool(mask.any()):
                continue
            neg_probs = probs.gather(
                -1, candidates.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            neg_loss = -torch.log1p(
                -neg_probs.clamp(min=1e-6, max=1.0 - 1e-6))
            mode_losses.append(neg_loss.masked_select(mask).mean())
            candidate_any |= mask
            candidate_total = candidate_total + mask.sum()
        if mode_losses:
            losses.append(sum(mode_losses) / len(mode_losses))
    if not losses:
        return zero, {
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


def continuation_repair_loss(
        model, features, txt, ids, modes, pad, repair_steps=4,
        prompt_frac=0.5, temperature=0.0, top_k=0):
    """Train recovery from the model's own free-running continuation errors."""
    zero = ids.new_zeros((), dtype=torch.float32)
    if ids.shape[0] == 0 or ids.shape[1] < 2 or int(repair_steps) <= 0:
        return zero, {
            "enabled": False, "skipped": True, "skip_reason": "too_short",
            "tokens": 0, "generated_tokens": 0, "changed_tokens": 0,
        }
    lengths = ids.ne(int(pad)).sum(dim=1).detach().cpu().tolist()
    prompt_lens = []
    repair_lens = []
    for length in lengths:
        length = int(length)
        if length < 2:
            prompt_lens.append(0)
            repair_lens.append(0)
            continue
        prompt_len = int(round(float(length) * float(prompt_frac)))
        prompt_len = max(1, min(prompt_len, length - 1))
        repair_len = min(int(repair_steps), length - prompt_len)
        prompt_lens.append(prompt_len)
        repair_lens.append(repair_len)
    if not any(repair_lens):
        return zero, {
            "enabled": False, "skipped": True,
            "skip_reason": "no_repair_targets", "tokens": 0,
            "generated_tokens": 0, "changed_tokens": 0,
        }

    mixed_ids = ids.detach().clone()
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
            row_index = torch.tensor(active_rows, dtype=torch.long, device=ids.device)
            sub_features = [x.index_select(0, row_index) for x in features]
            sub_txt = txt.index_select(0, row_index)
            context_lens = [prompt_lens[i] + step for i in active_rows]
            max_context = max(context_lens)
            context = ids.new_full((len(active_rows), max_context), int(pad))
            for out_i, row_i in enumerate(active_rows):
                context[out_i, :context_lens[out_i]] = mixed_ids[
                    row_i, :context_lens[out_i]]
            logits = model(sub_features, sub_txt, context, mode="full")
            for out_i, row_i in enumerate(active_rows):
                next_id = _generation_next_id(
                    logits[out_i, context_lens[out_i] - 1], pad,
                    temperature=temperature, top_k=top_k)
                mixed_ids[row_i, prompt_lens[row_i] + step] = next_id
    if was_training:
        model.train()

    mode_losses = []
    token_counts = []
    for mode in modes:
        logits = model(features, txt, mixed_ids, mode=mode)
        if logits.shape[1] < 2:
            continue
        targets = ids[:, 1:]
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
        if bool(mask.any()):
            mode_losses.append(raw.masked_select(mask).mean())
            token_counts.append(int(mask.sum().detach().cpu()))
    if not mode_losses:
        return zero, {
            "enabled": False, "skipped": True,
            "skip_reason": "empty_loss_mask", "tokens": 0,
            "generated_tokens": int(sum(repair_lens)), "changed_tokens": 0,
        }
    changed = 0
    generated = 0
    for row_i, repair_len in enumerate(repair_lens):
        if repair_len <= 0:
            continue
        start = prompt_lens[row_i]
        end = start + repair_len
        generated += repair_len
        changed += int((mixed_ids[row_i, start:end] != ids[row_i, start:end]).sum().cpu())
    return sum(mode_losses) / len(mode_losses), {
        "enabled": True,
        "skipped": False,
        "tokens": int(sum(token_counts)),
        "generated_tokens": int(generated),
        "changed_tokens": int(changed),
        "steps": int(repair_steps),
        "prompt_frac": float(prompt_frac),
        "temperature": float(temperature),
        "top_k": int(top_k),
    }


def logit_agreement_loss(logits_by_mode, ids, pad):
    items = list(logits_by_mode.items())
    if len(items) < 2 or ids.shape[1] < 2:
        return next(iter(logits_by_mode.values())).sum() * 0.0
    mask = ids[:, 1:].ne(pad)
    if not bool(mask.any()):
        return next(iter(logits_by_mode.values())).sum() * 0.0
    losses = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            _mode_a, logits_a = items[i]
            _mode_b, logits_b = items[j]
            a = logits_a[:, :-1][mask]
            b = logits_b[:, :-1][mask]
            losses.append(0.5 * (
                F.kl_div(F.log_softmax(a, dim=-1), F.softmax(b.detach(), dim=-1),
                         reduction="batchmean")
                + F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a.detach(), dim=-1),
                           reduction="batchmean")
            ))
    return torch.stack(losses).mean()


def latent_multimodal_concept_loss_from_views(views, invariance_w=25.0,
                                              variance_w=25.0,
                                              covariance_w=1.0,
                                              variance_target=1.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    losses = []
    modes = list(views)
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            losses.append(latent_concept_vicreg_loss(
                views[modes[i]], views[modes[j]],
                invariance_weight=invariance_w,
                variance_weight=variance_w,
                covariance_weight=covariance_w,
                variance_target=variance_target))
    if not losses:
        return next(iter(views.values())).sum() * 0.0
    return torch.stack(losses).mean()


def latent_multimodal_factorization_loss_from_views(
        views, variance_target=0.05, separation_margin=0.2,
        covariance_w=0.05):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    losses = [
        latent_concept_slot_factorization_loss(
            slots, variance_target=variance_target,
            separation_margin=separation_margin, covariance_weight=covariance_w)
        for slots in views.values()
    ]
    return torch.stack(losses).mean() if losses else next(iter(views.values())).sum() * 0.0


def latent_multimodal_fer_loss_from_views(
        views, fragmentation_w=1.0, correlation_w=1.0, balance_w=0.1):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    losses = [
        latent_concept_fer_loss(
            slots, fragmentation_w=fragmentation_w,
            correlation_w=correlation_w, balance_w=balance_w)
        for slots in views.values()
    ]
    return torch.stack(losses).mean() if losses else next(iter(views.values())).sum() * 0.0


def latent_multimodal_fer_metrics_from_views(views):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return {"fer_score": 0.0, "fragmentation": 0.0,
                "slot_correlation": 0.0, "slot_imbalance": 0.0,
                "mode_count": 0, "modes": {}}
    keys = ("fer_score", "fragmentation", "slot_correlation", "slot_imbalance")
    totals = {key: 0.0 for key in keys}
    modes = {}
    for mode, slots in views.items():
        metrics = latent_concept_fer_metrics(slots)
        report = {key: float(metrics[key].detach()) for key in keys}
        modes[mode] = report
        for key in keys:
            totals[key] += report[key]
    count = float(max(1, len(modes)))
    return {**{key: totals[key] / count for key in keys},
            "mode_count": len(modes),
            "modes": modes}


def latent_multimodal_fer_scores_from_views(views):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        zero = torch.zeros(0)
        return zero, {"fragmentation": zero, "slot_correlation": zero,
                      "slot_imbalance": zero}
    keys = ("fragmentation", "slot_correlation", "slot_imbalance")
    score_rows = []
    part_rows = {key: [] for key in keys}
    batch_size = None
    for slots in views.values():
        scores, parts = latent_concept_fer_scores(slots)
        if batch_size is None:
            batch_size = int(scores.shape[0])
        elif int(scores.shape[0]) != batch_size:
            raise ValueError("multimodal FER score views must share batch size")
        score_rows.append(scores)
        for key in keys:
            part_rows[key].append(parts[key])
    scores = torch.stack(score_rows).mean(0)
    parts = {key: torch.stack(values).mean(0)
             for key, values in part_rows.items()}
    return scores, parts


def latent_multimodal_bridge_loss_from_views(model, views):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return next(iter(views.values())).sum() * 0.0
    losses = [
        latent_concept_bridge_loss(
            slots, memory.active(), memory.active_relations(),
            memory.active_transitions())
        for slots in views.values()
    ]
    return torch.stack(losses).mean() if losses else next(iter(views.values())).sum() * 0.0


def latent_multimodal_bridge_graph_state(model):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return None
    return latent_concept_graph_snapshot(memory)


def latent_multimodal_bridge_metrics_from_views(model, views, bridge_graph=None):
    graph_source = "snapshot" if bridge_graph is not None else "current"
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 1.0, "mode_count": 0, "modes": {},
                "graph_ready": False, "graph_source": graph_source,
                "skipped": True, "skip_reason": "no_views"}
    if bridge_graph is None:
        bridge_graph = latent_multimodal_bridge_graph_state(model)
    if getattr(model, "latent_concepts", None) is None or bridge_graph is None:
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 1.0, "mode_count": 0, "modes": {},
                "memory_filled": 0, "relation_updates": 0,
                "transition_updates": 0, "graph_ready": False,
                "graph_source": graph_source, "skipped": True,
                "skip_reason": "latent_concept_memory_unavailable"}
    active_memory = bridge_graph["memory"]
    relations = bridge_graph["relations"]
    transitions = bridge_graph["transitions"]
    filled = int(active_memory.shape[0])
    relation_updates = int(bridge_graph.get("relation_updates", 0))
    transition_updates = int(bridge_graph.get("transition_updates", 0))
    if not latent_concept_graph_ready(graph_state=bridge_graph):
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 1.0, "mode_count": len(views),
                "modes": {}, "memory_filled": filled,
                "relation_updates": relation_updates,
                "transition_updates": transition_updates,
                "graph_ready": False, "graph_source": graph_source,
                "skipped": True,
                "skip_reason": "latent_concept_graph_unavailable"}
    keys = ("bridge_score", "bridge_entropy", "bridge_connectivity")
    totals = {key: 0.0 for key in keys}
    modes = {}
    for mode, slots in views.items():
        score, entropy, connectivity = latent_concept_bridge_scores(
            slots, active_memory, relations, transitions, require_graph=True)
        report = {
            "bridge_score": float(score.detach().mean()) if score.numel() else 0.0,
            "bridge_entropy": (
                float(entropy.detach().mean()) if entropy.numel() else 0.0),
            "bridge_connectivity": (
                float(connectivity.detach().mean()) if connectivity.numel() else 1.0),
        }
        modes[mode] = report
        for key in keys:
            totals[key] += report[key]
    count = float(max(1, len(modes)))
    return {**{key: totals[key] / count for key in keys},
            "mode_count": len(modes),
            "modes": modes,
            "memory_filled": filled,
            "relation_updates": relation_updates,
            "transition_updates": transition_updates,
            "graph_ready": True,
            "graph_source": graph_source,
            "skipped": False}


def latent_multimodal_memory_loss_from_views(model, views, temperature=0.1,
                                             balance_w=0.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return next(iter(views.values())).sum() * 0.0
    losses = [
        latent_concept_memory_loss(
            slots, memory.active(), temperature=temperature, balance_w=balance_w)
        for slots in views.values()
    ]
    return torch.stack(losses).mean() if losses else next(iter(views.values())).sum() * 0.0


def latent_multimodal_memory_consolidation_loss_from_views(
        model, views, temperature=0.1, balance_w=0.0, anchor_w=1.0,
        fer_w=0.0, fer_fragmentation_w=1.0, fer_correlation_w=1.0,
        fer_balance_w=0.1):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    zero = (next(iter(views.values())).sum() * 0.0
            if views else torch.tensor(0.0))
    metrics = {"memory_loss": zero, "anchor_loss": zero, "fer_loss": zero,
               "nearest_cosine": zero, "memory_active": 0, "skipped": True}
    if not views:
        return zero, metrics
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return zero, metrics
    active = memory.active()
    metrics["memory_active"] = int(active.shape[0])
    if active.numel() == 0:
        return zero, metrics
    losses = []
    by_key = {"memory_loss": [], "anchor_loss": [], "fer_loss": [],
              "nearest_cosine": []}
    for slots in views.values():
        loss, view_metrics = latent_concept_memory_consolidation_loss(
            slots, active, temperature=temperature, balance_w=balance_w,
            anchor_w=anchor_w, fer_w=fer_w,
            fer_fragmentation_w=fer_fragmentation_w,
            fer_correlation_w=fer_correlation_w, fer_balance_w=fer_balance_w)
        if bool(view_metrics.get("skipped", True)):
            continue
        losses.append(loss)
        for key in by_key:
            by_key[key].append(view_metrics[key])
    if not losses:
        return zero, metrics
    metrics = {key: torch.stack(values).mean() for key, values in by_key.items()}
    metrics["memory_active"] = int(active.shape[0])
    metrics["skipped"] = False
    return torch.stack(losses).mean(), metrics


def latent_multimodal_discovery_loss_from_views(
        model, views, curiosity_w=1.0, graph_w=1.0, cycle_w=1.0,
        bridge_w=1.0, fer_w=0.0, curiosity_temperature=0.1,
        curiosity_self_loop_w=0.05, curiosity_transitive_steps=2,
        curiosity_transitive_w=0.1, graph_temperature=0.1,
        graph_self_loop_w=0.05, graph_transitive_steps=2,
        graph_transitive_w=0.1, graph_target_power=1.0,
        cycle_temperature=0.1, cycle_self_loop_w=0.05,
        cycle_transitive_steps=2, cycle_transitive_w=0.1,
        cycle_target_power=1.0, cycle_consistency_w=0.5,
        fer_fragmentation_w=1.0, fer_correlation_w=1.0,
        fer_balance_w=0.1):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    zero = (next(iter(views.values())).sum() * 0.0
            if views else torch.tensor(0.0))
    metric_keys = (
        "curiosity_loss", "curiosity_novelty", "curiosity_association",
        "graph_loss", "graph_kl", "graph_cosine",
        "cycle_loss", "cycle_forward_kl", "cycle_reverse_kl",
        "cycle_source_cycle_kl", "cycle_target_cycle_kl",
        "insight_loss", "insight_score", "insight_kl", "insight_cosine",
        "insight_missing_mass", "insight_reachable_mass", "insight_gain",
        "bridge_loss", "bridge_score", "bridge_entropy",
        "bridge_connectivity", "fer_loss")
    metrics = {key: zero for key in metric_keys}
    metrics.update({"memory_active": 0, "graph_ready": False, "skipped": True})
    if not views:
        return zero, metrics
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return zero, metrics
    active = memory.active()
    metrics["memory_active"] = int(active.shape[0])
    if active.numel() == 0:
        return zero, metrics
    full = views.get("full")
    if full is None:
        full = next(iter(views.values()))
    base_loss, base_metrics = latent_concept_discovery_loss(
        full, active, relations=memory.active_relations(),
        transitions=memory.active_transitions(),
        prediction_relations=memory.active_prediction_relations(),
        curiosity_w=curiosity_w, graph_w=0.0, cycle_w=0.0,
        bridge_w=bridge_w, fer_w=fer_w,
        curiosity_temperature=curiosity_temperature,
        curiosity_self_loop_w=curiosity_self_loop_w,
        curiosity_transitive_steps=curiosity_transitive_steps,
        curiosity_transitive_w=curiosity_transitive_w,
        fer_fragmentation_w=fer_fragmentation_w,
        fer_correlation_w=fer_correlation_w,
        fer_balance_w=fer_balance_w)
    losses = [base_loss]
    by_key = {key: [base_metrics[key]] for key in metric_keys}
    graph_modes = [mode for mode in ("sensor_only", "text_only")
                   if mode in views and views[mode] is not full]
    for mode in graph_modes:
        loss, view_metrics = latent_concept_discovery_loss(
            full, active, relations=memory.active_relations(),
            transitions=memory.active_transitions(),
            prediction_relations=memory.active_prediction_relations(),
            source_slots=views[mode], target_slots=full,
            curiosity_w=0.0, graph_w=graph_w, cycle_w=cycle_w,
            bridge_w=0.0, fer_w=0.0,
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
            cycle_consistency_w=cycle_consistency_w)
        losses.append(loss)
        for key in metric_keys:
            by_key[key].append(view_metrics[key])
        metrics["graph_ready"] = bool(
            metrics["graph_ready"] or view_metrics["graph_ready"])
    metrics = {
        key: torch.stack(values).mean() for key, values in by_key.items()
    } | {
        "memory_active": int(active.shape[0]),
        "graph_ready": bool(metrics["graph_ready"] or base_metrics["graph_ready"]),
        "skipped": bool(base_metrics.get("skipped", True) and not graph_modes),
    }
    return torch.stack(losses).mean(), metrics


def latent_multimodal_reanalysis_loss_from_views(
        model, views, graph_w=1.0, cycle_w=0.5, bridge_w=0.5,
        fer_w=0.0, temperature=0.1, self_loop_w=0.05,
        transitive_steps=2, transitive_w=0.1, target_power=1.0,
        cycle_consistency_w=0.5, fer_fragmentation_w=1.0,
        fer_correlation_w=1.0, fer_balance_w=0.1):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    zero = (next(iter(views.values())).sum() * 0.0
            if views else torch.tensor(0.0))
    metric_keys = (
        "closure_loss", "closure_kl", "closure_cosine",
        "cycle_loss", "bridge_loss", "fer_loss")
    metrics = {key: zero for key in metric_keys}
    metrics.update({"memory_active": 0, "graph_ready": False, "skipped": True})
    if not views:
        return zero, metrics
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return zero, metrics
    active = memory.active()
    metrics["memory_active"] = int(active.shape[0])
    if active.numel() == 0:
        return zero, metrics
    anchor = views.get("full")
    if anchor is None:
        anchor = next(iter(views.values()))
    anchor = anchor.detach()
    probe_modes = [
        mode for mode in ("sensor_only", "text_only")
        if mode in views and views[mode] is not anchor
    ]
    if not probe_modes:
        probe_modes = [mode for mode, slots in views.items() if slots is not anchor]
    if not probe_modes:
        probe_modes = ["full"]
    losses = []
    by_key = {key: [] for key in metric_keys}
    graph_ready = False
    for mode in probe_modes:
        probe = views[mode] if mode in views else anchor
        loss, view_metrics = latent_concept_reanalysis_loss(
            probe, anchor, active, relations=memory.active_relations(),
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
        if bool(view_metrics.get("skipped", True)):
            continue
        losses.append(loss)
        for key in metric_keys:
            by_key[key].append(view_metrics[key])
        graph_ready = bool(graph_ready or view_metrics["graph_ready"])
    if not losses:
        return zero, metrics
    metrics = {
        key: torch.stack(values).mean() if values else zero
        for key, values in by_key.items()
    } | {"memory_active": int(active.shape[0]),
         "graph_ready": bool(graph_ready),
         "skipped": False}
    return torch.stack(losses).mean(), metrics


def latent_multimodal_gap_loss_from_views(
        model, views, temperature=0.1, self_loop_w=0.0,
        transitive_steps=2, transitive_w=0.1, target_power=1.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    zero = (next(iter(views.values())).sum() * 0.0
            if views else torch.tensor(0.0))
    metric_keys = (
        "gap_loss", "gap_kl", "gap_cosine", "gap_entropy",
        "gap_target_mass", "gap_present_overlap")
    metrics = {key: zero for key in metric_keys}
    metrics.update({"memory_active": 0, "graph_ready": False, "skipped": True})
    if not views:
        return zero, metrics
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return zero, metrics
    active = memory.active()
    metrics["memory_active"] = int(active.shape[0])
    if active.numel() == 0:
        return zero, metrics
    losses = []
    by_key = {key: [] for key in metric_keys}
    graph_ready = False
    for slots in views.values():
        loss, view_metrics = latent_concept_memory_gap_loss(
            slots, active, relations=memory.active_relations(),
            transitions=memory.active_transitions(), temperature=temperature,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w, target_power=target_power)
        if bool(view_metrics.get("skipped", True)):
            graph_ready = bool(graph_ready or view_metrics.get("graph_ready", False))
            continue
        losses.append(loss)
        for key in metric_keys:
            by_key[key].append(view_metrics[key])
        graph_ready = bool(graph_ready or view_metrics["graph_ready"])
    if not losses:
        return zero, metrics | {"graph_ready": bool(graph_ready)}
    metrics = {
        key: torch.stack(values).mean() if values else zero
        for key, values in by_key.items()
    } | {"memory_active": int(active.shape[0]),
         "graph_ready": bool(graph_ready),
         "skipped": False}
    return torch.stack(losses).mean(), metrics


def latent_multimodal_association_loss_from_views(
        model, views, temperature=0.1, target_power=1.0, self_loop_w=0.05,
        transitive_steps=1, transitive_w=0.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return next(iter(views.values())).sum() * 0.0
    losses = [
        latent_concept_association_loss(
            slots, memory.active(), memory.active_relations(),
            temperature=temperature, target_power=target_power,
            self_loop_w=self_loop_w, transitive_steps=transitive_steps,
            transitive_w=transitive_w)
        for slots in views.values()
    ]
    return torch.stack(losses).mean() if losses else next(iter(views.values())).sum() * 0.0


def latent_multimodal_composition_loss_from_views(
        model, views, temperature=0.1, self_loop_w=0.0, transitive_steps=2,
        transitive_w=0.1, margin=0.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return next(iter(views.values())).sum() * 0.0
    losses = [
        latent_concept_composition_loss(
            slots, memory.active(), memory.active_relations(),
            temperature=temperature, self_loop_w=self_loop_w,
            transitive_steps=transitive_steps, transitive_w=transitive_w,
            margin=margin)
        for slots in views.values()
    ]
    return torch.stack(losses).mean() if losses else next(iter(views.values())).sum() * 0.0


def latent_multimodal_graph_prediction_loss_from_views(
        model, views, temperature=0.1, self_loop_w=0.05, transitive_steps=2,
        transitive_w=0.1, target_power=1.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return next(iter(views.values())).sum() * 0.0
    target = views.get("full")
    if target is None:
        return next(iter(views.values())).sum() * 0.0
    losses = []
    for mode, source in views.items():
        if mode == "full":
            continue
        losses.append(latent_concept_graph_prediction_loss(
            source, target.detach(), memory.active(), memory.active_prediction_relations(),
            temperature=temperature, self_loop_w=self_loop_w,
            transitive_steps=transitive_steps, transitive_w=transitive_w,
            target_power=target_power))
    return torch.stack(losses).mean() if losses else target.sum() * 0.0


def latent_multimodal_completion_loss_from_views(model, views, temperature=0.1):
    """Predict the full fused concept state from partial modality views.

    This is a self-supervised completion objective: the target is the model's own
    full-view latent concept state, and each partial view must learn to recover
    that state without task labels or modality-specific rules.
    """
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    return latent_concept_completion_loss(
        _multimodal_completion_predictor(model), views,
        temperature=temperature, full_key="full")


def _multimodal_completion_predictor(model):
    return getattr(
        model, "concept_completion_predictor",
        getattr(model, "concept_sequence_predictor", None))


def latent_multimodal_sequence_prediction_loss(
        model, pair_batch, vocab, view_dims, device=DEV, temperature=0.1,
        view_dropout=0.0, decode_objective="target"):
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "concept_sequence_predictor", None) is None
            or not pair_batch or len(pair_batch) <= 1):
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        return torch.tensor(0.0, device=dev)
    anchors = [a for a, _b in pair_batch]
    positives = [b for _a, b in pair_batch]
    anchor_features, anchor_txt, _anchor_ids = _batch_from_records(
        anchors, vocab, device, view_dims, decode_objective=decode_objective)
    positive_features, positive_txt, _positive_ids = _batch_from_records(
        positives, vocab, device, view_dims, decode_objective=decode_objective)
    source_slots = model.latent_concept_states(
        anchor_features, anchor_txt, mode="full",
        view_dropout=view_dropout, project=False)
    target_slots = model.latent_concept_states(
        positive_features, positive_txt, mode="full",
        view_dropout=0.0, project=False)
    return latent_concept_sequence_prediction_loss(
        model.concept_sequence_predictor, source_slots, target_slots,
        temperature=temperature)


@torch.no_grad()
def update_multimodal_sequence_transitions(model, pair_batch, vocab, view_dims,
                                           device=DEV, decay=0.99,
                                           decode_objective="target"):
    memory = getattr(model, "latent_concept_memory", None)
    if (getattr(model, "latent_concepts", None) is None
            or memory is None or not pair_batch):
        return 0
    anchors = [a for a, _b in pair_batch]
    positives = [b for _a, b in pair_batch]
    anchor_features, anchor_txt, _anchor_ids = _batch_from_records(
        anchors, vocab, device, view_dims, decode_objective=decode_objective)
    positive_features, positive_txt, _positive_ids = _batch_from_records(
        positives, vocab, device, view_dims, decode_objective=decode_objective)
    source_slots = model.latent_concept_states(
        anchor_features, anchor_txt, mode="full", project=True)
    target_slots = model.latent_concept_states(
        positive_features, positive_txt, mode="full", project=True)
    return int(memory.update_transitions(source_slots, target_slots, decay=decay))


@torch.no_grad()
def update_multimodal_latent_memory(model, slots, momentum=0.95, relation_decay=None):
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return 0
    return memory.update(slots, momentum=momentum, relation_decay=relation_decay)


@torch.no_grad()
def update_multimodal_latent_transitions(model, views, decay=0.99):
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return 0
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    target = views.get("full")
    if target is None:
        return 0
    updates = 0
    for mode, source in views.items():
        if mode == "full":
            continue
        updates += int(memory.update_transitions(source, target, decay=decay))
    return int(updates)


def latent_multimodal_neighborhood_loss_from_views(views, temperature=0.1,
                                                   margin=0.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    anchor = views.get("full", next(iter(views.values())))
    if anchor.shape[0] <= 1:
        return anchor.sum() * 0.0
    with torch.no_grad():
        z = F.normalize(anchor.detach().reshape(anchor.shape[0], -1), dim=-1)
        sim = z.matmul(z.t())
        eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        nearest = sim.masked_fill(eye, -float("inf")).argmax(-1)
    losses = []
    for slots in views.values():
        positive = slots.index_select(0, nearest)
        losses.append(latent_concept_neighborhood_loss(
            anchor, positive, temperature=temperature, margin=margin))
    return torch.stack(losses).mean() if losses else anchor.sum() * 0.0


def latent_multimodal_transition_loss_from_views(views, temperature=0.1,
                                                 margin=0.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    anchor = views.get("full", next(iter(views.values())))
    if anchor.shape[0] <= 1:
        return anchor.sum() * 0.0
    with torch.no_grad():
        z = F.normalize(anchor.detach().reshape(anchor.shape[0], -1), dim=-1)
        sim = z.matmul(z.t())
        eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        nearest = sim.masked_fill(eye, -float("inf")).argmax(-1)
    anchor_positive = anchor.index_select(0, nearest)
    losses = []
    for slots in views.values():
        if slots is anchor:
            continue
        losses.append(latent_concept_transition_consistency_loss(
            anchor, anchor_positive, slots, slots.index_select(0, nearest),
            temperature=temperature, margin=margin))
    return torch.stack(losses).mean() if losses else anchor.sum() * 0.0


def _latent_batch_neighbor_clusters(anchor, min_cluster_size=2):
    if anchor is None or anchor.shape[0] < int(min_cluster_size):
        if anchor is None:
            return None
        return torch.full((anchor.shape[0],), -1, dtype=torch.long, device=anchor.device)
    with torch.no_grad():
        z = F.normalize(anchor.detach().reshape(anchor.shape[0], -1), dim=-1)
        sim = z.matmul(z.t())
        eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        nearest = sim.masked_fill(eye, -float("inf")).argmax(-1).detach().cpu().tolist()
    parent = list(range(anchor.shape[0]))

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
    counts = {}
    roots = []
    for i in range(anchor.shape[0]):
        root = find(i)
        roots.append(root)
        counts[root] = counts.get(root, 0) + 1
    root_to_cluster = {
        root: cluster_id for cluster_id, root in enumerate(
            sorted(root for root, count in counts.items()
                   if count >= int(min_cluster_size)))
    }
    ids = [root_to_cluster.get(root, -1) for root in roots]
    return torch.tensor(ids, dtype=torch.long, device=anchor.device)


def latent_multimodal_cluster_loss_from_views(views, temperature=0.1,
                                              margin=0.0, min_cluster_size=2):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    anchor = views.get("full", next(iter(views.values())))
    cluster_ids = _latent_batch_neighbor_clusters(
        anchor, min_cluster_size=min_cluster_size)
    if cluster_ids is None or not bool(cluster_ids.ge(0).any()):
        return anchor.sum() * 0.0
    losses = [
        latent_concept_cluster_prototype_loss(
            slots, cluster_ids, temperature=temperature, margin=margin,
            min_cluster_size=min_cluster_size)
        for slots in views.values()
    ]
    return torch.stack(losses).mean() if losses else anchor.sum() * 0.0


def latent_multimodal_graph_prediction_examples(
        model, records, vocab, view_dims, n=0, seed=0, device=DEV,
        temperature=0.1, self_loop_w=0.05, transitive_steps=2,
        transitive_w=0.1, target_power=1.0, decode_objective="target"):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return [], {"n_records": 0, "n_selected": 0, "skipped": True}
    rng = np.random.default_rng(seed)
    sample = _sample_unique_records(
        records, int(n) if int(n) > 0 else len(records), rng)
    scored = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(sample), 64):
            batch_records = sample[off:off + 64]
            features, txt, ids = _batch_from_records(
                batch_records, vocab, device, view_dims,
                decode_objective=decode_objective)
            target = model.latent_concept_states(
                features, txt, mode="full", project=True)
            scores_by_mode = []
            for mode in ("sensor_only", "text_only"):
                source = model.latent_concept_states(
                    features, txt, mode=mode, project=True)
                scores, _parts = latent_concept_graph_prediction_scores(
                    source, target, memory.active(), memory.active_prediction_relations(),
                    temperature=temperature, self_loop_w=self_loop_w,
                    transitive_steps=transitive_steps, transitive_w=transitive_w,
                    target_power=target_power)
                scores_by_mode.append(scores)
            batch_scores = torch.stack(scores_by_mode).mean(0)
            for i, rec in enumerate(batch_records):
                scored.append((float(batch_scores[i].detach().cpu()), rec))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [rec for _score, rec in scored], {
        "n_records": len(sample),
        "n_selected": len(scored),
        "mean_score": float(np.mean([s for s, _rec in scored])) if scored else 0.0,
        "max_score": float(max([s for s, _rec in scored])) if scored else 0.0,
        "skipped": False,
    }


def latent_multimodal_fer_examples(
        model, records, vocab, view_dims, n=0, seed=0, device=DEV,
        decode_objective="target"):
    if getattr(model, "latent_concepts", None) is None:
        return [], {"n_records": 0, "n_selected": 0,
                    "mean_fer_score": 0.0, "max_fer_score": 0.0,
                    "mean_fer_fragmentation": 0.0,
                    "mean_fer_slot_correlation": 0.0,
                    "mean_fer_slot_imbalance": 0.0,
                    "skipped": True}
    rng = np.random.default_rng(seed)
    sample = _sample_unique_records(
        records, int(n) if int(n) > 0 else len(records), rng)
    scored = []
    score_values = []
    fragmentation_values = []
    correlation_values = []
    imbalance_values = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(sample), 64):
            batch_records = sample[off:off + 64]
            features, txt, _ids = _batch_from_records(
                batch_records, vocab, device, view_dims,
                decode_objective=decode_objective)
            views = {
                mode: model.latent_concept_states(
                    features, txt, mode=mode, project=True)
                for mode in MODES
            }
            scores, parts = latent_multimodal_fer_scores_from_views(views)
            fragmentation = parts["fragmentation"]
            correlation = parts["slot_correlation"]
            imbalance = parts["slot_imbalance"]
            for i, rec in enumerate(batch_records):
                score = float(scores[i].detach().cpu())
                scored.append((score, rec))
                score_values.append(score)
                fragmentation_values.append(float(fragmentation[i].detach().cpu()))
                correlation_values.append(float(correlation[i].detach().cpu()))
                imbalance_values.append(float(imbalance[i].detach().cpu()))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [rec for _score, rec in scored]
    return selected, {"n_records": len(sample),
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


def evaluate(model, records, vocab, view_dims, n=200, seed=1,
             device=DEV, mode="full", decode_objective="target"):
    rng = np.random.default_rng(seed)
    count = min(int(n), len(records)) if int(n) > 0 else len(records)
    sample = _sample_records(records, count, rng) if count else []
    model.eval()
    losses, correct, total, exact, rows = [], 0, 0, 0, 0
    with torch.no_grad():
        for off in range(0, len(sample), 64):
            batch_records = sample[off:off + 64]
            features, _txt, ids, decode_txt = _batch_from_records(
                batch_records, vocab, device, view_dims,
                decode_objective=decode_objective, return_decode_text=True)
            logits = model(features, decode_txt, ids, mode=mode)
            loss = token_loss(logits, ids, vocab.pad)
            losses.append(float(loss.detach().cpu()))
            if ids.shape[1] >= 2:
                targets = ids[:, 1:]
                mask = targets.ne(vocab.pad)
                pred = logits[:, :-1].argmax(-1)
                hit = pred.eq(targets) & mask
                correct += int(hit.sum())
                total += int(mask.sum())
                row_ok = (hit | ~mask).all(-1)
                exact += int(row_ok.sum())
                rows += int(row_ok.numel())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "token_acc": correct / max(1, total),
        "exact": exact / max(1, rows),
        "target_tokens": int(total),
        "target_rows": int(rows),
        "token_skipped": total == 0,
        "exact_skipped": rows == 0,
        "n_records": int(len(sample)),
        "mode": mode,
        "decode_objective": str(decode_objective),
    }


def latent_multimodal_fer_eval(model, records, vocab, view_dims, n=200,
                               seed=1, device=DEV,
                               decode_objective="target"):
    if getattr(model, "latent_concepts", None) is None:
        return {"fer_score": 0.0, "fragmentation": 0.0,
                "slot_correlation": 0.0, "slot_imbalance": 0.0,
                "n_records": 0, "sampled": False, "skipped": True}
    rng = np.random.default_rng(seed)
    count = min(int(n), len(records)) if int(n) > 0 else len(records)
    sample = _sample_records(records, count, rng) if count else []
    if not sample:
        return {"fer_score": 0.0, "fragmentation": 0.0,
                "slot_correlation": 0.0, "slot_imbalance": 0.0,
                "n_records": 0, "sampled": False, "skipped": True}
    keys = ("fer_score", "fragmentation", "slot_correlation", "slot_imbalance")
    totals = {key: 0.0 for key in keys}
    mode_totals = {mode: {key: 0.0 for key in keys} for mode in MODES}
    total = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(sample), 64):
            batch_records = sample[off:off + 64]
            features, txt, _ids = _batch_from_records(
                batch_records, vocab, device, view_dims,
                decode_objective=decode_objective)
            views = {
                mode: model.latent_concept_states(
                    features, txt, mode=mode, project=True)
                for mode in MODES
            }
            metrics = latent_multimodal_fer_metrics_from_views(views)
            weight = len(batch_records)
            total += weight
            for key in keys:
                totals[key] += float(metrics[key]) * weight
            for mode, mode_metrics in metrics.get("modes", {}).items():
                for key in keys:
                    mode_totals[mode][key] += float(mode_metrics[key]) * weight
    if total == 0:
        return {"fer_score": 0.0, "fragmentation": 0.0,
                "slot_correlation": 0.0, "slot_imbalance": 0.0,
                "n_records": 0, "sampled": False, "skipped": True}
    report = {key: totals[key] / float(total) for key in keys}
    report.update({
        "n_records": total,
        "sampled": bool(int(n) > 0 and int(n) < len(records)),
        "skipped": False,
        "modes": {
            mode: {key: vals[key] / float(total) for key in keys}
            for mode, vals in mode_totals.items()
        },
    })
    return report


def latent_multimodal_bridge_eval(model, records, vocab, view_dims, n=200,
                                  seed=1, device=DEV,
                                  decode_objective="target"):
    bridge_graph = latent_multimodal_bridge_graph_state(model)
    if getattr(model, "latent_concepts", None) is None or bridge_graph is None:
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 1.0, "n_records": 0,
                "sampled": False, "memory_filled": 0,
                "relation_updates": 0, "transition_updates": 0,
                "graph_ready": False, "graph_source": "current",
                "skipped": True, "modes": {},
                "skip_reason": "latent_concept_memory_unavailable"}
    rng = np.random.default_rng(seed)
    count = min(int(n), len(records)) if int(n) > 0 else len(records)
    sample = _sample_records(records, count, rng) if count else []
    if not sample:
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 1.0, "n_records": 0,
                "sampled": False, "memory_filled": 0,
                "relation_updates": 0, "transition_updates": 0,
                "graph_ready": False, "graph_source": "current",
                "skipped": True, "modes": {}, "skip_reason": "no_records"}
    filled = int(bridge_graph["memory"].shape[0])
    relation_updates = int(bridge_graph.get("relation_updates", 0))
    transition_updates = int(bridge_graph.get("transition_updates", 0))
    if not latent_concept_graph_ready(graph_state=bridge_graph):
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 1.0, "n_records": len(sample),
                "sampled": bool(int(n) > 0 and int(n) < len(records)),
                "memory_filled": filled, "relation_updates": relation_updates,
                "transition_updates": transition_updates,
                "graph_ready": False, "graph_source": "current",
                "skipped": True, "modes": {},
                "skip_reason": "latent_concept_graph_unavailable"}
    keys = ("bridge_score", "bridge_entropy", "bridge_connectivity")
    totals = {key: 0.0 for key in keys}
    mode_totals = {mode: {key: 0.0 for key in keys} for mode in MODES}
    total = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(sample), 64):
            batch_records = sample[off:off + 64]
            features, txt, _ids = _batch_from_records(
                batch_records, vocab, device, view_dims,
                decode_objective=decode_objective)
            views = {
                mode: model.latent_concept_states(
                    features, txt, mode=mode, project=True)
                for mode in MODES
            }
            metrics = latent_multimodal_bridge_metrics_from_views(
                model, views, bridge_graph=bridge_graph)
            weight = len(batch_records)
            total += weight
            for key in keys:
                totals[key] += float(metrics[key]) * weight
            for mode, mode_metrics in metrics.get("modes", {}).items():
                for key in keys:
                    mode_totals[mode][key] += float(mode_metrics[key]) * weight
    if total == 0:
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 1.0, "n_records": 0,
                "sampled": False, "skipped": True, "modes": {}}
    report = {key: totals[key] / float(total) for key in keys}
    report.update({
        "n_records": total,
        "sampled": bool(int(n) > 0 and int(n) < len(records)),
        "memory_filled": filled,
        "relation_updates": relation_updates,
        "transition_updates": transition_updates,
        "graph_ready": True,
        "graph_source": "current",
        "skipped": False,
        "modes": {
            mode: {key: vals[key] / float(total) for key in keys}
            for mode, vals in mode_totals.items()
        },
    })
    return report


def latent_multimodal_gap_eval(model, records, vocab, view_dims, n=200,
                               seed=1, device=DEV,
                               decode_objective="target"):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return {"gap_score": 0.0, "gap_kl": 0.0, "gap_cosine": 0.0,
                "gap_entropy": 0.0, "gap_target_mass": 0.0,
                "gap_present_overlap": 0.0, "usable_gap_records": 0,
                "n_records": 0, "sampled": False, "memory_filled": 0,
                "graph_ready": False, "skipped": True,
                "skip_reason": "latent_concept_memory_unavailable"}
    rng = np.random.default_rng(seed)
    count = min(int(n), len(records)) if int(n) > 0 else len(records)
    sample = _sample_records(records, count, rng) if count else []
    filled = int(getattr(memory, "filled", torch.zeros((), dtype=torch.long)).item())
    if not sample:
        return {"gap_score": 0.0, "gap_kl": 0.0, "gap_cosine": 0.0,
                "gap_entropy": 0.0, "gap_target_mass": 0.0,
                "gap_present_overlap": 0.0, "usable_gap_records": 0,
                "n_records": 0, "sampled": False, "memory_filled": filled,
                "graph_ready": False, "skipped": True,
                "skip_reason": "no_records"}
    active = memory.active()
    if active.numel() == 0:
        return {"gap_score": 0.0, "gap_kl": 0.0, "gap_cosine": 0.0,
                "gap_entropy": 0.0, "gap_target_mass": 0.0,
                "gap_present_overlap": 0.0, "usable_gap_records": 0,
                "n_records": len(sample),
                "sampled": bool(int(n) > 0 and int(n) < len(records)),
                "memory_filled": filled, "graph_ready": False, "skipped": True,
                "skip_reason": "latent_concept_memory_empty"}
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
        for off in range(0, len(sample), 64):
            batch_records = sample[off:off + 64]
            features, txt, _ids = _batch_from_records(
                batch_records, vocab, device, view_dims,
                decode_objective=decode_objective)
            slots = model.latent_concept_states(
                features, txt, mode="full", project=True)
            scores, parts = latent_concept_memory_gap_scores(
                slots, active, relations=memory.active_relations(),
                transitions=memory.active_transitions())
            graph_ready = bool(graph_ready or parts.get("graph_ready", False))
            usable = parts.get("usable")
            if usable is not None:
                usable_count += int(usable.detach().sum().item())
            for key, part_key in (
                    ("gap_score", None),
                    ("gap_kl", "kl"),
                    ("gap_cosine", "cosine"),
                    ("gap_entropy", "entropy"),
                    ("gap_target_mass", "target_mass"),
                    ("gap_present_overlap", "present_overlap")):
                row = (scores if part_key is None else parts.get(
                    part_key, scores.new_zeros(scores.shape)))
                values[key].extend(float(x) for x in row.detach().cpu().tolist())

    def mean_value(name):
        rows = values[name]
        return float(np.mean(rows)) if rows else 0.0

    return {"gap_score": mean_value("gap_score"),
            "gap_kl": mean_value("gap_kl"),
            "gap_cosine": mean_value("gap_cosine"),
            "gap_entropy": mean_value("gap_entropy"),
            "gap_target_mass": mean_value("gap_target_mass"),
            "gap_present_overlap": mean_value("gap_present_overlap"),
            "usable_gap_records": int(usable_count),
            "n_records": len(sample),
            "sampled": bool(int(n) > 0 and int(n) < len(records)),
            "memory_filled": filled,
            "graph_ready": bool(graph_ready),
            "skipped": not (graph_ready and usable_count > 0)}


def latent_multimodal_sequence_eval(model, records, vocab, view_dims, n=200,
                                    seed=1, device=DEV,
                                    decode_objective="target"):
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "concept_sequence_predictor", None) is None):
        return {"sequence_acc": 0.0, "n_records": 0, "n_pairs": 0,
                "sampled": False, "skipped": True}
    pairs, mine_report = mine_multimodal_sequence_pairs(
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
            anchor_features, anchor_txt, _anchor_ids = _batch_from_records(
                anchors, vocab, device, view_dims,
                decode_objective=decode_objective)
            positive_features, positive_txt, _positive_ids = _batch_from_records(
                positives, vocab, device, view_dims,
                decode_objective=decode_objective)
            source_slots = model.latent_concept_states(
                anchor_features, anchor_txt, mode="full", project=False)
            target_slots = model.latent_concept_states(
                positive_features, positive_txt, mode="full", project=False)
            predicted = model.concept_sequence_predictor(source_slots)
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


def multimodal_score_components(mode_metrics, fer_eval=None, bridge_eval=None,
                                gap_eval=None, sequence_eval=None,
                                generation_eval=None, metric="mastery",
                                margin_w=0.1):
    metric = str(metric)
    if metric not in MULTIMODAL_SCORE_METRICS:
        raise ValueError(f"unknown multimodal score metric {metric!r}")
    margin_w = float(margin_w)
    mode_metrics = dict(mode_metrics or {})
    active_modes = [
        row for row in mode_metrics.values()
        if int(row.get("n_records", 0)) > 0
    ]
    token_values = [
        float(row.get("token_acc", 0.0)) for row in active_modes
        if not bool(row.get("token_skipped", False))
    ]
    exact_values = [
        float(row.get("exact", 0.0)) for row in active_modes
        if not bool(row.get("exact_skipped", False))
    ]
    token_score = float(np.mean(token_values)) if token_values else 0.0
    exact_score = float(np.mean(exact_values)) if exact_values else 0.0
    mode_floor = float(min(token_values)) if token_values else 0.0
    mode_gap = float(token_score - mode_floor) if token_values else 0.0
    fer_eval = fer_eval or {"skipped": True}
    bridge_eval = bridge_eval or {"skipped": True}
    gap_eval = gap_eval or {"skipped": True}
    sequence_eval = sequence_eval or {"skipped": True}
    include_generation = generation_eval is not None or metric == "generation"
    generation_eval = generation_eval or {"skipped": True}
    fer_raw_score = max(0.0, float(fer_eval.get("fer_score", 0.0)))
    fer_score = (0.0 if bool(fer_eval.get("skipped", False))
                 else 1.0 / (1.0 + fer_raw_score))
    bridge_raw_score = max(0.0, float(bridge_eval.get("bridge_score", 0.0)))
    bridge_resolution = 1.0 / (1.0 + bridge_raw_score)
    bridge_connectivity = min(1.0, max(0.0, float(
        bridge_eval.get("bridge_connectivity", 1.0))))
    bridge_score = (0.0 if bool(bridge_eval.get("skipped", False))
                    else 0.5 * (bridge_resolution + bridge_connectivity))
    gap_raw_score = max(0.0, float(gap_eval.get("gap_score", 0.0)))
    gap_resolution = 1.0 / (1.0 + gap_raw_score)
    connection_parts = []
    if not bool(bridge_eval.get("skipped", False)):
        connection_parts.append(bridge_score)
    if not bool(gap_eval.get("skipped", False)):
        connection_parts.append(gap_resolution)
    connection_score = (
        float(np.mean(connection_parts)) if connection_parts else 0.0)
    sequence_acc = float(sequence_eval.get("sequence_acc", 0.0))
    sequence_margin = float(sequence_eval.get("margin", 0.0))
    sequence_score = (0.0 if bool(sequence_eval.get("skipped", False))
                      else sequence_acc + margin_w * sequence_margin)
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
    scores = {"token": token_score, "exact": exact_score,
              "fer": fer_score, "bridge": bridge_score,
              "connection": connection_score,
              "sequence": sequence_score}
    skipped = {"token": not token_values, "exact": not exact_values,
               "fer": bool(fer_eval.get("skipped", False)),
               "bridge": bool(bridge_eval.get("skipped", False)),
               "connection": not bool(connection_parts),
               "sequence": bool(sequence_eval.get("skipped", False))}
    if include_generation:
        scores["generation"] = generation_score
        skipped["generation"] = bool(generation_eval.get("skipped", False))
    active_scores = [scores[name] for name in scores if not skipped[name]]
    if not active_scores:
        active_scores = [0.0]
    active_mean_score = float(np.mean(active_scores))
    floor_score = float(min(active_scores))
    balanced_score = 0.5 * (active_mean_score + floor_score)
    all_score = float(np.mean(list(scores.values())))
    signal_coverage = sum(0 if skipped[name] else 1 for name in scores) / float(
        len(scores))
    mode_floor_score = mode_floor if token_values else active_mean_score
    mastery_score = (
        0.4 * active_mean_score
        + 0.2 * balanced_score
        + 0.2 * mode_floor_score
        + 0.2 * signal_coverage)
    if metric == "token":
        score = token_score
    elif metric == "exact":
        score = exact_score
    elif metric == "generation":
        score = generation_score
    elif metric == "fer":
        score = fer_score
    elif metric == "bridge":
        score = bridge_score
    elif metric == "connection":
        score = connection_score
    elif metric == "sequence":
        score = sequence_score
    elif metric == "all":
        score = all_score
    elif metric == "balanced":
        score = balanced_score
    else:
        score = mastery_score
    return {"metric": metric,
            "margin_w": margin_w,
            "score": float(score),
            "all_score": float(all_score),
            "active_mean_score": float(active_mean_score),
            "floor_score": float(floor_score),
            "balanced_score": float(balanced_score),
            "mastery_score": float(mastery_score),
            "signal_coverage": float(signal_coverage),
            "token_score": float(token_score),
            "exact_score": float(exact_score),
            "generation_score": float(generation_score),
            "generation_token_acc": float(generation_token_acc),
            "generation_exact": float(generation_exact),
            "generation_diversity": float(generation_diversity),
            "generation_collapse_penalty": float(generation_collapse_penalty),
            "mode_floor": float(mode_floor),
            "mode_floor_score": float(mode_floor_score),
            "mode_gap": float(mode_gap),
            "fer_score": float(fer_score),
            "fer_raw_score": float(fer_raw_score),
            "fer_fragmentation": float(fer_eval.get("fragmentation", 0.0)),
            "fer_slot_correlation": float(fer_eval.get("slot_correlation", 0.0)),
            "fer_slot_imbalance": float(fer_eval.get("slot_imbalance", 0.0)),
            "bridge_score": float(bridge_score),
            "bridge_raw_score": float(bridge_raw_score),
            "bridge_resolution": float(bridge_resolution),
            "bridge_connectivity": float(bridge_connectivity),
            "bridge_entropy": float(bridge_eval.get("bridge_entropy", 0.0)),
            "gap_raw_score": float(gap_raw_score),
            "gap_resolution": float(gap_resolution),
            "gap_target_mass": float(gap_eval.get("gap_target_mass", 0.0)),
            "connection_score": float(connection_score),
            "sequence_score": float(sequence_score),
            "sequence_acc": sequence_acc,
            "sequence_margin": sequence_margin,
            "token_skipped": skipped["token"],
            "exact_skipped": skipped["exact"],
            "generation_skipped": bool(skipped.get("generation", True)),
            "fer_skipped": skipped["fer"],
            "bridge_skipped": skipped["bridge"],
            "connection_skipped": skipped["connection"],
            "sequence_skipped": skipped["sequence"],
            "mode_scores": {
                mode: {
                    "token_acc": float(row.get("token_acc", 0.0)),
                    "exact": float(row.get("exact", 0.0)),
                    "loss": float(row.get("loss", 0.0)),
                    "target_tokens": int(row.get("target_tokens", 0)),
                    "target_rows": int(row.get("target_rows", 0)),
                    "token_skipped": bool(row.get("token_skipped", False)),
                    "exact_skipped": bool(row.get("exact_skipped", False)),
                }
                for mode, row in sorted(mode_metrics.items())
            }}


def multimodal_self_teach_weight_plan(score_components, budget=0.0,
                                      history_prior=None,
                                      history_prior_w=0.5,
                                      available_objectives=None):
    budget = float(budget)
    if budget < 0.0:
        raise ValueError("multimodal self-teach budget must be non-negative")
    history_prior_w = float(history_prior_w)
    if history_prior_w < 0.0:
        raise ValueError(
            "multimodal self-teach history prior weight must be non-negative")
    history_prior = history_prior if isinstance(history_prior, dict) else {}
    history_deficits = (
        history_prior.get("signal_deficits")
        if isinstance(history_prior.get("signal_deficits"), dict) else {})
    concept_connection_signal = max(
        0.0, _mm_float(history_prior.get("concept_connection_signal", 0.0)))
    extras = {key: 0.0 for key in MULTIMODAL_SELF_TEACH_WEIGHT_KEYS}
    if available_objectives is None:
        available = set(MULTIMODAL_SELF_TEACH_WEIGHT_KEYS)
    else:
        available = {
            str(key) for key in available_objectives
            if str(key) in MULTIMODAL_SELF_TEACH_WEIGHT_KEYS}
    deficits = {}
    current_deficits = {}
    unavailable = {}
    active = []
    score_components = dict(score_components or {})
    for signal in MULTIMODAL_SELF_TEACH_SIGNALS:
        if bool(score_components.get(
                MULTIMODAL_SELF_TEACH_SKIP_KEYS[signal], False)):
            continue
        score_key = MULTIMODAL_SELF_TEACH_SCORE_KEYS[signal]
        if score_key not in score_components:
            continue
        score = float(score_components.get(score_key, 0.0))
        quality = min(1.0, max(0.0, score))
        current_deficit = max(0.0, 1.0 - quality)
        history_deficit = max(
            0.0, _mm_float(history_deficits.get(signal, 0.0)))
        if signal == "connection":
            history_deficit = max(history_deficit, concept_connection_signal)
        deficit = max(current_deficit, history_prior_w * history_deficit)
        current_deficits[signal] = float(current_deficit)
        deficits[signal] = float(deficit)
        objectives = tuple(
            key for key in MULTIMODAL_SELF_TEACH_SIGNAL_OBJECTIVES[signal]
            if key in available)
        if not objectives:
            if deficit > 0.0:
                unavailable[signal] = {
                    "deficit": float(deficit),
                    "objectives": list(
                        MULTIMODAL_SELF_TEACH_SIGNAL_OBJECTIVES[signal]),
                }
            continue
        if deficit > 0.0:
            active.append((signal, deficit, objectives))
    total_deficit = sum(deficit for _signal, deficit, _objectives in active)
    if budget > 0.0 and total_deficit > 0.0:
        for signal, deficit, objectives in active:
            signal_budget = budget * deficit / total_deficit
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
            signal: _mm_float(value)
            for signal, value in history_deficits.items()
            if signal in MULTIMODAL_SELF_TEACH_SIGNALS},
        "history_prior_enabled": bool(history_prior.get("enabled", False)),
        "history_prior_entry_count": int(history_prior.get("entry_count", 0) or 0),
        "history_prior_top_signal": history_prior.get("top_signal"),
        "history_prior_w": float(history_prior_w),
        "history_concept_connection_signal": float(concept_connection_signal),
        "available_objectives": [
            key for key in MULTIMODAL_SELF_TEACH_WEIGHT_KEYS if key in available],
        "unavailable_signals": unavailable,
        "active_signals": [signal for signal, _deficit, _objectives in active],
        "weight_extras": {key: float(value) for key, value in extras.items()},
    }


def multimodal_self_teach_weight_maps(score_components=None, budget=0.0,
                                      history_prior=None, history_prior_w=0.5,
                                      available_objectives=None,
                                      **base_weights):
    budget = float(budget)
    if budget < 0.0:
        raise ValueError("multimodal self-teach budget must be non-negative")
    missing = [
        key for key in MULTIMODAL_SELF_TEACH_WEIGHT_KEYS
        if key not in base_weights]
    if missing:
        raise ValueError(
            "missing multimodal self-teach base weights: "
            + ", ".join(sorted(missing)))
    base = {
        key: float(base_weights[key])
        for key in MULTIMODAL_SELF_TEACH_WEIGHT_KEYS}
    effective = dict(base)
    if budget == 0.0:
        return None, base, effective
    if score_components is None:
        raise ValueError(
            "multimodal self-teach needs score components when budget is positive")
    plan = multimodal_self_teach_weight_plan(
        score_components, budget=budget,
        history_prior=history_prior, history_prior_w=history_prior_w,
        available_objectives=available_objectives)
    extras = plan["weight_extras"]
    effective = {
        key: float(base[key]) + float(extras.get(key, 0.0))
        for key in MULTIMODAL_SELF_TEACH_WEIGHT_KEYS}
    return plan, base, effective


def multimodal_eval_bundle(model, records, vocab, view_dims, n=200, seed=1,
                           device=DEV, score_metric="mastery",
                           score_margin_w=0.1, decode_objective="target",
                           modes=None, generation_eval_n=0,
                           generation_prompt_tokens=16,
                           generation_max_new_tokens=32,
                           generation_temperature=0.0,
                           generation_top_k=0):
    modes = tuple(modes or multimodal_active_modes(view_dims, records))
    metrics = {
        mode: evaluate(model, records, vocab, view_dims, n=n,
                       seed=seed + 17 + i * 997, device=device, mode=mode,
                       decode_objective=decode_objective)
        for i, mode in enumerate(modes)
    }
    fer = latent_multimodal_fer_eval(
        model, records, vocab, view_dims, n=n, seed=seed + 29, device=device,
        decode_objective=decode_objective)
    bridge = latent_multimodal_bridge_eval(
        model, records, vocab, view_dims, n=n, seed=seed + 31, device=device,
        decode_objective=decode_objective)
    gap = latent_multimodal_gap_eval(
        model, records, vocab, view_dims, n=n, seed=seed + 33, device=device,
        decode_objective=decode_objective)
    sequence = latent_multimodal_sequence_eval(
        model, records, vocab, view_dims, n=n, seed=seed + 37, device=device,
        decode_objective=decode_objective)
    if int(generation_eval_n) > 0 and modes:
        generation_mode = "full" if "full" in modes else modes[0]
        generation = multimodal_generation_eval(
            model, records, vocab, view_dims, n=generation_eval_n,
            seed=seed + 41, device=device, mode=generation_mode,
            decode_objective=decode_objective,
            prompt_tokens=generation_prompt_tokens,
            max_new_tokens=generation_max_new_tokens,
            temperature=generation_temperature, top_k=generation_top_k)
    elif int(generation_eval_n) > 0:
        generation = {
            "enabled": False, "skipped": True,
            "skip_reason": "no_active_modes",
        }
    else:
        generation = {
            "enabled": False, "skipped": True,
            "skip_reason": "generation_eval_n_zero",
        }
    return {"teacher_forced": metrics,
            "generation": generation,
            "latent_fer": fer,
            "latent_bridge": bridge,
            "latent_gap": gap,
            "latent_sequence": sequence,
            "score_components": multimodal_score_components(
                metrics, fer_eval=fer, bridge_eval=bridge,
                gap_eval=gap, sequence_eval=sequence,
                generation_eval=(
                    generation if int(generation_eval_n) > 0 else None),
                metric=score_metric, margin_w=score_margin_w)}


def multimodal_representation_progress_report(before_bundle, after_bundle):
    if not isinstance(before_bundle, dict) or not isinstance(after_bundle, dict):
        return {"enabled": False}
    before_scores = before_bundle.get("score_components")
    after_scores = after_bundle.get("score_components")
    if not isinstance(before_scores, dict) or not isinstance(after_scores, dict):
        return {"enabled": False}
    deltas = {}
    for key in MULTIMODAL_REPRESENTATION_PROGRESS_KEYS:
        if key in before_scores and key in after_scores:
            deltas[key] = float(
                _mm_float(after_scores.get(key), 0.0)
                - _mm_float(before_scores.get(key), 0.0))
    active_signals = []
    signal_before = {}
    signal_after = {}
    signal_deltas = {}
    for key in MULTIMODAL_REPRESENTATION_SIGNAL_KEYS:
        signal = key[:-len("_score")]
        if bool(before_scores.get(f"{signal}_skipped", False)) and bool(
                after_scores.get(f"{signal}_skipped", False)):
            continue
        before_value = _mm_float(before_scores.get(key), 0.0)
        after_value = _mm_float(after_scores.get(key), 0.0)
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
            key: _mm_float(before_scores.get(key), 0.0)
            for key in MULTIMODAL_REPRESENTATION_PROGRESS_KEYS
            if key in before_scores
        },
        "after": {
            key: _mm_float(after_scores.get(key), 0.0)
            for key in MULTIMODAL_REPRESENTATION_PROGRESS_KEYS
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


def multimodal_transfer_calibration_report(
        before_bundle, after_bundle, score_min_delta=0.0,
        insight_accept_w=0.25, insight_min_delta=0.0, gate=True):
    """Decide whether an imported checkpoint helps the current manifest."""
    if not isinstance(before_bundle, dict) or not isinstance(after_bundle, dict):
        return {"enabled": False, "accepted": True, "gate_enabled": bool(gate)}
    before_scores = before_bundle.get("score_components")
    after_scores = after_bundle.get("score_components")
    if not isinstance(before_scores, dict) or not isinstance(after_scores, dict):
        return {"enabled": False, "accepted": True, "gate_enabled": bool(gate)}
    progress = multimodal_representation_progress_report(
        before_bundle, after_bundle)
    score_before = _mm_float(before_scores.get("score", 0.0), 0.0)
    score_after = _mm_float(after_scores.get("score", 0.0), 0.0)
    score_delta = float(score_after - score_before)
    active_signals = progress.get("active_signals")
    insight_gate = bool(active_signals)
    organization_delta = _mm_float(
        progress.get("organization_score_delta", 0.0), 0.0)
    insight_allowed = bool(not insight_gate or organization_delta >= -1.0e-9)
    decision = concept_round_selection_decision(
        score_delta, score_min_delta,
        insight_delta=organization_delta,
        insight_allowed=insight_allowed,
        insight_gate=insight_gate,
        insight_accept_w=insight_accept_w,
        insight_min_delta=insight_min_delta)
    calibration_accept = bool(decision["selected"])
    accepted = bool(calibration_accept or not gate)
    return {
        "enabled": True,
        "gate_enabled": bool(gate),
        "accepted": accepted,
        "calibration_accept": calibration_accept,
        "reverted": bool(gate and not calibration_accept),
        "score_before": float(score_before),
        "score_after": float(score_after),
        "score_delta": float(score_delta),
        "score_min_delta": float(score_min_delta),
        "selected_by_score": bool(decision["selected_by_score"]),
        "selected_by_insight": bool(decision["selected_by_insight"]),
        "insight_score_boost": float(decision["insight_score_boost"]),
        "insight_effective_delta": float(decision["insight_effective_delta"]),
        "insight_gate": bool(insight_gate),
        "insight_allowed": bool(insight_allowed),
        "insight_accept_w": float(insight_accept_w),
        "insight_min_delta": float(insight_min_delta),
        "representation_progress": progress,
        "before_score_components": before_scores,
        "after_score_components": after_scores,
    }


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


def _bridge_quality(report):
    report = dict(report or {})
    if bool(report.get("skipped", True)):
        return 0.0
    raw = max(0.0, float(report.get("bridge_score", 0.0)))
    resolution = 1.0 / (1.0 + raw)
    connectivity = min(1.0, max(0.0, float(
        report.get("bridge_connectivity", 1.0))))
    return float(0.5 * (resolution + connectivity))


def multimodal_bridge_selection_insight(before_bridge, after_bridge, enabled=True):
    if not enabled or before_bridge is None or after_bridge is None:
        return {
            "skipped": True,
            "bridge_insight_delta": 0.0,
            "bridge_insight_allowed": True,
        }
    before_bridge = dict(before_bridge or {})
    after_bridge = dict(after_bridge or {})
    before_skipped = bool(before_bridge.get("skipped", True))
    after_skipped = bool(after_bridge.get("skipped", True))
    before_quality = _bridge_quality(before_bridge)
    after_quality = _bridge_quality(after_bridge)
    before_score = (
        max(0.0, float(before_bridge.get("bridge_score", 0.0)))
        if not before_skipped else 0.0)
    after_score = (
        max(0.0, float(after_bridge.get("bridge_score", 0.0)))
        if not after_skipped else 0.0)
    before_connectivity = (
        min(1.0, max(0.0, float(before_bridge.get("bridge_connectivity", 1.0))))
        if not before_skipped else 0.0)
    after_connectivity = (
        min(1.0, max(0.0, float(after_bridge.get("bridge_connectivity", 1.0))))
        if not after_skipped else 0.0)
    quality_gain = float(after_quality - before_quality)
    return {
        "skipped": bool(before_skipped and after_skipped),
        "bridge_insight_delta": quality_gain,
        "bridge_insight_allowed": bool(quality_gain >= -1e-9),
        "bridge_quality_before": float(before_quality),
        "bridge_quality_after": float(after_quality),
        "bridge_quality_gain": quality_gain,
        "bridge_score_before": float(before_score),
        "bridge_score_after": float(after_score),
        "bridge_score_reduction": float(before_score - after_score),
        "bridge_connectivity_before": float(before_connectivity),
        "bridge_connectivity_after": float(after_connectivity),
        "bridge_connectivity_gain": float(after_connectivity - before_connectivity),
        "bridge_before_skipped": bool(before_skipped),
        "bridge_after_skipped": bool(after_skipped),
    }


def _compact_multimodal_train_metrics(metrics):
    omitted = {
        "latent_fer_study_reports",
        "latent_discovery_study_reports",
        "latent_completion_study_reports",
        "latent_study_reports",
        "selection",
    }
    return {key: value for key, value in dict(metrics).items()
            if key not in omitted}


def multimodal_weight_update_snapshot(model, max_tensors=24, max_values=8):
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


def multimodal_weight_update_report(before_snapshot, after_snapshot, atol=1e-12):
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


def multimodal_learning_event_report(report):
    """Summarize whether multimodal training produced reusable learning evidence."""
    report = report if isinstance(report, dict) else {}
    train_metrics = report.get("train_metrics")
    if not isinstance(train_metrics, dict):
        train_metrics = report
    selection = train_metrics.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    rounds = [
        row for row in selection.get("rounds", ())
        if isinstance(row, dict)
    ]
    weight_update = train_metrics.get("weight_update")
    weight_update = weight_update if isinstance(weight_update, dict) else {}
    weight_moved = bool(
        train_metrics.get("weight_update_changed",
                          weight_update.get("changed", False)))
    selection_enabled = bool(selection.get("enabled", False))
    accepted_update = bool(selection.get("accepted_update", False))
    update_applied = bool(accepted_update or not selection_enabled)
    score_gain = max(
        0.0,
        _mm_float(selection.get("selected_score_delta", 0.0), 0.0),
        _mm_float(selection.get("selected_effective_delta", 0.0), 0.0),
    )
    bridge_insight = selection.get("selected_bridge_insight")
    if not isinstance(bridge_insight, dict):
        bridge_insight = {}
    bridge_delta = max(
        0.0,
        _mm_float(bridge_insight.get("bridge_insight_delta", 0.0), 0.0),
        _mm_float(bridge_insight.get("bridge_quality_gain", 0.0), 0.0),
        _mm_float(bridge_insight.get("bridge_connectivity_gain", 0.0), 0.0),
    )
    if not bridge_delta:
        bridge_delta = max((
            max(
                0.0,
                _mm_float(
                    (row.get("bridge_insight") or {}).get(
                        "bridge_insight_delta", 0.0), 0.0),
                _mm_float(
                    (row.get("bridge_insight") or {}).get(
                        "bridge_quality_gain", 0.0), 0.0),
                _mm_float(
                    (row.get("bridge_insight") or {}).get(
                        "bridge_connectivity_gain", 0.0), 0.0),
            )
            for row in rounds
            if bool(row.get("selected", False))
        ), default=0.0)
    concept_connection = bool(
        selection.get("selected_by_insight", False) or bridge_delta > 0.0)
    progress = train_metrics.get("representation_progress")
    if not isinstance(progress, dict):
        progress = report.get("representation_progress")
    progress = progress if isinstance(progress, dict) else {}
    representation_delta = _mm_float(
        progress.get("organization_score_delta", 0.0), 0.0)
    representation_gain = max(
        0.0,
        representation_delta,
        _mm_float(progress.get("positive_signal_gain", 0.0), 0.0),
    )
    representation_event = bool(
        progress.get("representation_insight_event", False)
        or representation_gain > 0.0)
    triggered = bool(
        update_applied and weight_moved
        and (score_gain > 0.0 or concept_connection or representation_event))
    if concept_connection:
        kind = "concept_connection"
        top_signal = "connection"
    elif representation_event:
        kind = "representation_reorganization"
        top_signal = str(progress.get("top_gain_signal", ""))
    elif score_gain > 0.0:
        kind = "score_gain"
        top_signal = str((train_metrics.get("self_teach_plan") or {}).get(
            "top_signal", ""))
    else:
        kind = "parameter_update"
        top_signal = ""
    if top_signal not in MULTIMODAL_SELF_TEACH_SIGNALS:
        top_signal = ""
    event_score = min(
        1.0,
        score_gain + bridge_delta + max(0.0, representation_gain))
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
        "weight_update_changed_tensor_count": int(_mm_int(
            train_metrics.get("weight_update_changed_tensor_count",
                              weight_update.get("changed_tensor_count", 0)), 0)),
        "weight_update_changed_value_count": int(_mm_int(
            train_metrics.get("weight_update_changed_value_count",
                              weight_update.get("changed_value_count", 0)), 0)),
        "weight_update_max_abs_delta": float(_mm_float(
            train_metrics.get("weight_update_max_abs_delta",
                              weight_update.get("max_abs_delta", 0.0)), 0.0)),
        "attempted_weight_update_count": int(_mm_int(
            train_metrics.get("attempted_weight_update_count",
                              selection.get("attempted_weight_update_count", 0)), 0)),
        "score_gain": float(score_gain),
        "concept_connection": bool(concept_connection),
        "concept_connection_delta": float(bridge_delta),
        "representation_event": bool(representation_event),
        "representation_organization_score_delta": float(representation_delta),
        "representation_positive_signal_gain": float(
            _mm_float(progress.get("positive_signal_gain", 0.0), 0.0)),
    }


def _multimodal_sequence_surprise_rows(
        model, pairs, vocab, view_dims, device=DEV, temperature=0.1,
        decode_objective="target"):
    rows = []
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "concept_sequence_predictor", None) is None
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
            anchor_features, anchor_txt, _anchor_ids = _batch_from_records(
                anchors, vocab, device, view_dims,
                decode_objective=decode_objective)
            positive_features, positive_txt, _positive_ids = _batch_from_records(
                positives, vocab, device, view_dims,
                decode_objective=decode_objective)
            source_slots = model.latent_concept_states(
                anchor_features, anchor_txt, mode="full", project=False)
            target_slots = model.latent_concept_states(
                positive_features, positive_txt, mode="full", project=False)
            scores, parts = latent_concept_sequence_prediction_scores(
                model.concept_sequence_predictor, source_slots, target_slots,
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


def latent_multimodal_completion_examples(
        model, records, vocab, view_dims, n=0, seed=0, device=DEV,
        temperature=0.1, decode_objective="target"):
    if (getattr(model, "latent_concepts", None) is None
            or _multimodal_completion_predictor(model) is None):
        return [], {"n_records": 0, "n_selected": 0,
                    "mean_score": 0.0, "max_score": 0.0,
                    "mean_completion_surprise": 0.0,
                    "max_completion_surprise": 0.0, "skipped": True}
    rng = np.random.default_rng(seed)
    candidates = _sample_unique_records(
        records, int(n) if int(n) > 0 else len(records), rng)
    rows = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(candidates), 64):
            batch_records = candidates[off:off + 64]
            features, txt, _ids = _batch_from_records(
                batch_records, vocab, device, view_dims,
                decode_objective=decode_objective)
            views = {
                mode: model.latent_concept_states(
                    features, txt, mode=mode, project=False)
                for mode in MODES
            }
            scores, parts = latent_concept_completion_scores(
                _multimodal_completion_predictor(model), views,
                temperature=temperature, full_key="full")
            mode_parts = parts.get("modes", {})
            sensor_parts = mode_parts.get("sensor_only", {})
            text_parts = mode_parts.get("text_only", {})
            zero = scores.new_zeros(scores.shape)
            sensor_surprise = sensor_parts.get("surprise", zero)
            text_surprise = text_parts.get("surprise", zero)
            sensor_cosine = sensor_parts.get("positive_cosine", zero)
            text_cosine = text_parts.get("positive_cosine", zero)
            sensor_gap = sensor_parts.get("closure_gap", zero)
            text_gap = text_parts.get("closure_gap", zero)
            sensor_rank = sensor_parts.get("rank", zero)
            text_rank = text_parts.get("rank", zero)
            for i, rec in enumerate(batch_records):
                rows.append({
                    "record": rec,
                    "completion_surprise": float(scores[i].detach().cpu()),
                    "completion_cross_entropy": float(
                        parts["cross_entropy"][i].detach().cpu()),
                    "completion_cosine": float(
                        parts["positive_cosine"][i].detach().cpu()),
                    "completion_gap": float(parts["closure_gap"][i].detach().cpu()),
                    "completion_rank": float(parts["rank"][i].detach().cpu()),
                    "sensor_completion_surprise": float(
                        sensor_surprise[i].detach().cpu()),
                    "sensor_completion_cosine": float(
                        sensor_cosine[i].detach().cpu()),
                    "sensor_completion_gap": float(sensor_gap[i].detach().cpu()),
                    "sensor_completion_rank": float(sensor_rank[i].detach().cpu()),
                    "text_completion_surprise": float(
                        text_surprise[i].detach().cpu()),
                    "text_completion_cosine": float(text_cosine[i].detach().cpu()),
                    "text_completion_gap": float(text_gap[i].detach().cpu()),
                    "text_completion_rank": float(text_rank[i].detach().cpu()),
                })
    rows.sort(key=lambda row: row["completion_surprise"], reverse=True)
    selected = [row["record"] for row in rows]

    def mean_field(name):
        return float(np.mean([row[name] for row in rows])) if rows else 0.0

    def max_field(name):
        return float(max([row[name] for row in rows])) if rows else 0.0

    return selected, {"n_records": len(candidates),
                      "n_selected": len(selected),
                      "mean_score": mean_field("completion_surprise"),
                      "max_score": max_field("completion_surprise"),
                      "mean_completion_surprise": mean_field(
                          "completion_surprise"),
                      "max_completion_surprise": max_field(
                          "completion_surprise"),
                      "mean_completion_cross_entropy": mean_field(
                          "completion_cross_entropy"),
                      "mean_completion_cosine": mean_field(
                          "completion_cosine"),
                      "mean_completion_gap": mean_field("completion_gap"),
                      "mean_completion_rank": mean_field("completion_rank"),
                      "mean_sensor_completion_surprise": mean_field(
                          "sensor_completion_surprise"),
                      "mean_sensor_completion_cosine": mean_field(
                          "sensor_completion_cosine"),
                      "mean_sensor_completion_gap": mean_field(
                          "sensor_completion_gap"),
                      "mean_sensor_completion_rank": mean_field(
                          "sensor_completion_rank"),
                      "mean_text_completion_surprise": mean_field(
                          "text_completion_surprise"),
                      "mean_text_completion_cosine": mean_field(
                          "text_completion_cosine"),
                      "mean_text_completion_gap": mean_field(
                          "text_completion_gap"),
                      "mean_text_completion_rank": mean_field(
                          "text_completion_rank"),
                      "temperature": float(temperature),
                      "skipped": not bool(rows)}


def latent_multimodal_discovery_examples(
        model, records, vocab, view_dims, n=0, seed=0, device=DEV,
        curiosity_temperature=0.1, curiosity_self_loop_w=0.05,
        curiosity_transitive_steps=2, curiosity_transitive_w=0.1,
        graph_temperature=0.1, graph_self_loop_w=0.05,
        graph_transitive_steps=2, graph_transitive_w=0.1,
        graph_target_power=1.0, cycle_temperature=0.1,
        cycle_self_loop_w=0.05, cycle_transitive_steps=2,
        cycle_transitive_w=0.1, cycle_target_power=1.0, cycle_w=0.5,
        sequence_temperature=0.1, completion_temperature=0.1,
        decode_objective="target"):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return [], {"n_records": 0, "n_selected": 0,
                    "mean_score": 0.0, "max_score": 0.0, "skipped": True}
    rng = np.random.default_rng(seed)
    candidates = _sample_unique_records(
        records, int(n) if int(n) > 0 else len(records), rng)
    candidate_ids = {rec.rec_id for rec in candidates}
    sequence_pairs, sequence_mining = mine_multimodal_sequence_pairs(
        records, split="train")
    if candidate_ids:
        sequence_pairs = [
            (a, b) for a, b in sequence_pairs
            if a.rec_id in candidate_ids and b.rec_id in candidate_ids
        ]
    sequence_rows = _multimodal_sequence_surprise_rows(
        model, sequence_pairs, vocab, view_dims, device=device,
        temperature=sequence_temperature, decode_objective=decode_objective)
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
        active_memory = memory.active()
        active_relations = memory.active_relations()
        prediction_relations = memory.active_prediction_relations()
        active_transitions = memory.active_transitions()
        graph_ready = latent_concept_graph_ready(memory)
        for off in range(0, len(candidates), 64):
            batch_records = candidates[off:off + 64]
            features, txt, _ids = _batch_from_records(
                batch_records, vocab, device, view_dims,
                decode_objective=decode_objective)
            views = {
                mode: model.latent_concept_states(
                    features, txt, mode=mode, project=True)
                for mode in MODES
            }
            full_slots = views["full"]
            completion, completion_parts = latent_concept_completion_scores(
                _multimodal_completion_predictor(model), views,
                temperature=completion_temperature, full_key="full")
            curiosity, curiosity_parts = latent_concept_graph_curiosity_scores(
                full_slots, active_memory, active_relations,
                temperature=curiosity_temperature,
                self_loop_w=curiosity_self_loop_w,
                transitive_steps=curiosity_transitive_steps,
                transitive_w=curiosity_transitive_w)
            gap, gap_parts = latent_concept_memory_gap_scores(
                full_slots, active_memory, relations=active_relations,
                transitions=active_transitions, temperature=graph_temperature,
                self_loop_w=0.0, transitive_steps=graph_transitive_steps,
                transitive_w=graph_transitive_w,
                target_power=graph_target_power)
            graph_rows = []
            cycle_rows = []
            insight_rows = []
            graph_parts_rows = {
                "kl": [], "cosine": []}
            cycle_parts_rows = {
                "forward_kl": [], "reverse_kl": [],
                "source_cycle_kl": [], "target_cycle_kl": []}
            insight_parts_rows = {
                "loss": [], "kl": [], "cosine": [], "missing_mass": [],
                "reachable_mass": [], "gain": []}
            for mode in ("sensor_only", "text_only"):
                source = views.get(mode)
                if source is None:
                    continue
                graph, graph_parts = latent_concept_graph_prediction_scores(
                    source, full_slots, active_memory, prediction_relations,
                    temperature=graph_temperature, self_loop_w=graph_self_loop_w,
                    transitive_steps=graph_transitive_steps,
                    transitive_w=graph_transitive_w,
                    target_power=graph_target_power)
                cycle, cycle_parts = latent_concept_graph_cycle_scores(
                    source, full_slots, active_memory, prediction_relations,
                    temperature=cycle_temperature, self_loop_w=cycle_self_loop_w,
                    transitive_steps=cycle_transitive_steps,
                    transitive_w=cycle_transitive_w,
                    target_power=cycle_target_power, cycle_w=cycle_w)
                insight, insight_parts = latent_concept_insight_scores(
                    source, full_slots, active_memory, relations=active_relations,
                    transitions=active_transitions, temperature=graph_temperature,
                    self_loop_w=graph_self_loop_w,
                    transitive_steps=graph_transitive_steps,
                    transitive_w=graph_transitive_w,
                    target_power=graph_target_power)
                graph_rows.append(graph)
                cycle_rows.append(cycle)
                insight_rows.append(insight)
                for key in graph_parts_rows:
                    graph_parts_rows[key].append(
                        graph_parts.get(key, graph.new_zeros(graph.shape)))
                for key in cycle_parts_rows:
                    cycle_parts_rows[key].append(
                        cycle_parts.get(key, cycle.new_zeros(cycle.shape)))
                for key in insight_parts_rows:
                    insight_parts_rows[key].append(
                        insight_parts.get(
                            key, insight.new_zeros(insight.shape)))
            zero = full_slots.reshape(full_slots.shape[0], -1).sum(-1) * 0.0
            graph = torch.stack(graph_rows).mean(0) if graph_rows else zero
            cycle = torch.stack(cycle_rows).mean(0) if cycle_rows else zero
            insight = torch.stack(insight_rows).mean(0) if insight_rows else zero
            graph_parts = {
                key: (torch.stack(values).mean(0) if values else zero)
                for key, values in graph_parts_rows.items()}
            cycle_parts = {
                key: (torch.stack(values).mean(0) if values else zero)
                for key, values in cycle_parts_rows.items()}
            insight_parts = {
                key: (torch.stack(values).mean(0) if values else zero)
                for key, values in insight_parts_rows.items()}
            fer_scores, fer_parts = latent_multimodal_fer_scores_from_views(views)
            bridge, bridge_entropy, bridge_connectivity = latent_concept_bridge_scores(
                full_slots, active_memory, active_relations, active_transitions)
            gap_kl = gap_parts.get("kl", gap.new_zeros(gap.shape))
            gap_cosine = gap_parts.get("cosine", gap.new_zeros(gap.shape))
            gap_entropy = gap_parts.get("entropy", gap.new_zeros(gap.shape))
            gap_target_mass = gap_parts.get("target_mass", gap.new_zeros(gap.shape))
            gap_present_overlap = gap_parts.get(
                "present_overlap", gap.new_zeros(gap.shape))
            novelty = curiosity_parts.get(
                "novelty", curiosity.new_zeros(curiosity.shape))
            association = curiosity_parts.get(
                "association", curiosity.new_zeros(curiosity.shape))
            completion_ce = completion_parts.get(
                "cross_entropy", completion.new_zeros(completion.shape))
            completion_cosine = completion_parts.get(
                "positive_cosine", completion.new_zeros(completion.shape))
            completion_gap = completion_parts.get(
                "closure_gap", completion.new_zeros(completion.shape))
            completion_rank = completion_parts.get(
                "rank", completion.new_zeros(completion.shape))
            for i, rec in enumerate(batch_records):
                seq = sequence_by_id.get(rec.rec_id, {})
                rows.append({
                    "record": rec,
                    "completion_surprise": float(completion[i].detach().cpu()),
                    "completion_cross_entropy": float(
                        completion_ce[i].detach().cpu()),
                    "completion_cosine": float(
                        completion_cosine[i].detach().cpu()),
                    "completion_gap": float(completion_gap[i].detach().cpu()),
                    "completion_rank": float(completion_rank[i].detach().cpu()),
                    "curiosity": float(curiosity[i].detach().cpu()),
                    "novelty": float(novelty[i].detach().cpu()),
                    "association": float(association[i].detach().cpu()),
                    "graph": float(graph[i].detach().cpu()),
                    "graph_kl": float(graph_parts["kl"][i].detach().cpu()),
                    "graph_cosine": float(graph_parts["cosine"][i].detach().cpu()),
                    "cycle": float(cycle[i].detach().cpu()),
                    "forward_kl": float(
                        cycle_parts["forward_kl"][i].detach().cpu()),
                    "reverse_kl": float(
                        cycle_parts["reverse_kl"][i].detach().cpu()),
                    "source_cycle_kl": float(
                        cycle_parts["source_cycle_kl"][i].detach().cpu()),
                    "target_cycle_kl": float(
                        cycle_parts["target_cycle_kl"][i].detach().cpu()),
                    "insight": float(insight[i].detach().cpu()),
                    "insight_loss": float(
                        insight_parts["loss"][i].detach().cpu()),
                    "insight_kl": float(
                        insight_parts["kl"][i].detach().cpu()),
                    "insight_cosine": float(
                        insight_parts["cosine"][i].detach().cpu()),
                    "insight_missing_mass": float(
                        insight_parts["missing_mass"][i].detach().cpu()),
                    "insight_reachable_mass": float(
                        insight_parts["reachable_mass"][i].detach().cpu()),
                    "insight_gain": float(
                        insight_parts["gain"][i].detach().cpu()),
                    "fer_score": float(fer_scores[i].detach().cpu()),
                    "fer_fragmentation": float(
                        fer_parts["fragmentation"][i].detach().cpu()),
                    "fer_slot_correlation": float(
                        fer_parts["slot_correlation"][i].detach().cpu()),
                    "fer_slot_imbalance": float(
                        fer_parts["slot_imbalance"][i].detach().cpu()),
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
        "curiosity", "gap", "insight", "completion_surprise", "graph", "cycle",
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
                      "n_selected": len(selected),
                      "mean_score": mean_field("score"),
                      "max_score": max_field("score"),
                      "mean_curiosity": mean_field("curiosity"),
                      "mean_novelty": mean_field("novelty"),
                      "mean_association": mean_field("association"),
                      "mean_completion_surprise": mean_field(
                          "completion_surprise"),
                      "max_completion_surprise": max_field(
                          "completion_surprise"),
                      "mean_completion_cross_entropy": mean_field(
                          "completion_cross_entropy"),
                      "mean_completion_cosine": mean_field("completion_cosine"),
                      "mean_completion_gap": mean_field("completion_gap"),
                      "mean_completion_rank": mean_field("completion_rank"),
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
                      "memory_filled": int(active_memory.shape[0]),
                      "graph_ready": bool(graph_ready),
                      "skipped": False}


def train(manifest, root=None, steps=400, batch=32, d=96, lr=1e-3, seed=0,
          device=DEV, log_every=100, layers=3, heads=4, max_len=128,
          max_vocab=0,
          source_balance_w=MULTIMODAL_DEFAULT_SOURCE_BALANCE_W,
          objective_profile="language",
          decode_objective="auto",
          view_tokens=4, txt_tokens=8, trunk_arch="mlp",
          trunk_width=128, trunk_depth=1, text_layers=1,
          text_arch="transformer", modality_dropout=0.0, decode_w=1.0,
          agreement_w=0.0,
          continuation_repair_w=0.0, continuation_repair_steps=4,
          continuation_repair_prompt_frac=0.5,
          continuation_repair_temperature=0.0,
          continuation_repair_top_k=0,
          repetition_unlikelihood_w=0.0, repetition_unlikelihood_window=32,
          text_checkpoint=None, multimodal_checkpoint=None,
          concept_tokens=4, fusion_layers=1,
          latent_concept_slots=0, latent_concept_layers=1,
          latent_concept_prefix=False, latent_concept_w=0.0,
          latent_concept_view_dropout=0.1,
          latent_concept_invariance_w=25.0,
          latent_concept_variance_w=25.0,
          latent_concept_covariance_w=1.0,
          latent_concept_variance_target=1.0,
          latent_concept_factorization_w=0.0,
          latent_concept_factorization_variance=0.05,
          latent_concept_factorization_margin=0.2,
          latent_concept_factorization_covariance_w=0.05,
          latent_concept_fer_w=0.0,
          latent_concept_fer_fragmentation_w=1.0,
          latent_concept_fer_correlation_w=1.0,
          latent_concept_fer_balance_w=0.1,
          latent_concept_fer_probe_n=0,
          latent_concept_fer_hard_max=0,
          latent_concept_fer_refresh_steps=0,
          latent_concept_discovery_probe_n=0,
          latent_concept_discovery_hard_max=0,
          latent_concept_discovery_refresh_steps=0,
          latent_concept_completion_probe_n=0,
          latent_concept_completion_hard_max=0,
          latent_concept_completion_refresh_steps=0,
          latent_concept_memory_w=0.0,
          latent_concept_memory_size=0,
          latent_concept_topk=0,
          latent_concept_memory_temperature=0.1,
          latent_concept_memory_momentum=0.95,
          latent_concept_memory_balance_w=0.01,
          latent_concept_consolidation_w=0.0,
          latent_concept_consolidation_temperature=0.1,
          latent_concept_consolidation_balance_w=0.01,
          latent_concept_consolidation_anchor_w=1.0,
          latent_concept_consolidation_fer_w=0.0,
          latent_concept_discovery_w=0.0,
          latent_concept_discovery_curiosity_w=1.0,
          latent_concept_discovery_graph_w=1.0,
          latent_concept_discovery_cycle_w=1.0,
          latent_concept_discovery_bridge_w=1.0,
          latent_concept_discovery_fer_w=0.0,
          latent_concept_reanalysis_w=0.0,
          latent_concept_reanalysis_graph_w=1.0,
          latent_concept_reanalysis_cycle_w=0.5,
          latent_concept_reanalysis_bridge_w=0.5,
          latent_concept_reanalysis_fer_w=0.0,
          latent_concept_reanalysis_cycle_consistency_w=0.5,
          latent_concept_gap_w=0.0,
          latent_concept_gap_temperature=0.1,
          latent_concept_gap_self_loop_w=0.0,
          latent_concept_gap_transitive_steps=2,
          latent_concept_gap_transitive_w=0.1,
          latent_concept_gap_target_power=1.0,
          latent_concept_association_w=0.0,
          latent_concept_association_temperature=0.1,
          latent_concept_association_decay=0.99,
          latent_concept_association_target_power=1.0,
          latent_concept_association_self_loop_w=0.05,
          latent_concept_association_transitive_steps=2,
          latent_concept_association_transitive_w=0.1,
          latent_concept_composition_w=0.0,
          latent_concept_composition_temperature=0.1,
          latent_concept_composition_self_loop_w=0.0,
          latent_concept_composition_transitive_steps=2,
          latent_concept_composition_transitive_w=0.1,
          latent_concept_composition_margin=0.0,
          latent_concept_graph_predict_w=0.0,
          latent_concept_graph_predict_temperature=0.1,
          latent_concept_graph_predict_self_loop_w=0.05,
          latent_concept_graph_predict_transitive_steps=2,
          latent_concept_graph_predict_transitive_w=0.1,
          latent_concept_graph_predict_target_power=1.0,
          latent_concept_bridge_w=0.0,
          latent_concept_completion_w=0.0,
          latent_concept_completion_temperature=0.1,
          latent_concept_sequence_w=0.0,
          latent_concept_sequence_batch=0,
          latent_concept_sequence_temperature=0.1,
          latent_concept_neighborhood_w=0.0,
          latent_concept_neighborhood_temperature=0.1,
          latent_concept_neighborhood_margin=0.0,
          latent_concept_transition_w=0.0,
          latent_concept_transition_temperature=0.1,
          latent_concept_transition_margin=0.0,
          latent_concept_cluster_w=0.0,
          latent_concept_cluster_temperature=0.1,
          latent_concept_cluster_margin=0.0,
          latent_concept_cluster_min_size=2,
          representation_probe_n=0,
          text_transfer_probe_n=DEFAULT_TEXT_TRANSFER_PROBE_N,
          text_transfer_score_min_delta=DEFAULT_TEXT_TRANSFER_SCORE_MIN_DELTA,
          text_transfer_insight_accept_w=DEFAULT_TEXT_TRANSFER_INSIGHT_ACCEPT_W,
          text_transfer_insight_min_delta=0.0,
          text_transfer_gate=True,
          self_teach_w=0.0, self_teach_history_prior_w=0.5,
          select_best=False, selection_rounds=1,
          selection_score_metric="mastery",
          selection_score_margin_w=0.1,
          selection_score_min_delta=0.0,
          selection_score_patience=0,
          selection_signal_regression_tolerance=(
              MULTIMODAL_DEFAULT_SIGNAL_REGRESSION_TOLERANCE),
          selection_eval_n=200,
          selection_generation_n=0,
          selection_generation_prompt_tokens=16,
          selection_generation_max_new_tokens=32,
          selection_generation_temperature=0.0,
          selection_generation_top_k=0,
          selection_insight_accept_w=0.25,
          selection_insight_min_delta=0.0):
    ckpt_latents = text_checkpoint_latent_config(
        text_checkpoint, device="cpu") if text_checkpoint else {}
    if text_checkpoint and latent_concept_slots <= 0:
        if ckpt_latents.get("latent_concept_slots", 0) > 0:
            latent_concept_slots = ckpt_latents["latent_concept_slots"]
            latent_concept_layers = ckpt_latents["latent_concept_layers"]
    if (text_checkpoint and latent_concept_topk <= 0
            and ckpt_latents.get("latent_concept_topk", 0) > 0):
        latent_concept_topk = ckpt_latents["latent_concept_topk"]
    if (text_checkpoint and latent_concept_memory_size <= 0
            and ckpt_latents.get("latent_concept_memory_size", 0) > 0):
        latent_concept_memory_size = ckpt_latents["latent_concept_memory_size"]
    profile_kwargs = multimodal_objective_profile_kwargs(
        objective_profile,
        continuation_repair_w=continuation_repair_w,
        continuation_repair_steps=continuation_repair_steps,
        repetition_unlikelihood_w=repetition_unlikelihood_w,
        repetition_unlikelihood_window=repetition_unlikelihood_window,
        selection_generation_n=selection_generation_n,
        latent_concept_slots=latent_concept_slots,
        latent_concept_memory_size=latent_concept_memory_size,
        latent_concept_factorization_w=latent_concept_factorization_w,
        latent_concept_fer_w=latent_concept_fer_w,
        latent_concept_memory_w=latent_concept_memory_w,
        latent_concept_consolidation_w=latent_concept_consolidation_w,
        latent_concept_discovery_w=latent_concept_discovery_w,
        latent_concept_gap_w=latent_concept_gap_w,
        latent_concept_association_w=latent_concept_association_w,
        latent_concept_composition_w=latent_concept_composition_w,
        latent_concept_graph_predict_w=latent_concept_graph_predict_w,
        latent_concept_bridge_w=latent_concept_bridge_w,
        latent_concept_completion_w=latent_concept_completion_w,
        latent_concept_sequence_w=latent_concept_sequence_w,
        latent_concept_neighborhood_w=latent_concept_neighborhood_w,
        latent_concept_transition_w=latent_concept_transition_w,
        latent_concept_cluster_w=latent_concept_cluster_w,
        self_teach_w=self_teach_w,
        latent_concept_fer_probe_n=latent_concept_fer_probe_n,
        latent_concept_fer_hard_max=latent_concept_fer_hard_max,
        latent_concept_fer_refresh_steps=latent_concept_fer_refresh_steps,
        latent_concept_discovery_probe_n=latent_concept_discovery_probe_n,
        latent_concept_discovery_hard_max=latent_concept_discovery_hard_max,
        latent_concept_discovery_refresh_steps=(
            latent_concept_discovery_refresh_steps),
        latent_concept_completion_probe_n=latent_concept_completion_probe_n,
        latent_concept_completion_hard_max=latent_concept_completion_hard_max,
        latent_concept_completion_refresh_steps=(
            latent_concept_completion_refresh_steps),
        representation_probe_n=representation_probe_n,
        selection_rounds=selection_rounds,
        selection_score_patience=selection_score_patience)
    objective_profile = profile_kwargs["objective_profile"]
    objective_profile_report = profile_kwargs["objective_profile_report"]
    continuation_repair_w = profile_kwargs["continuation_repair_w"]
    continuation_repair_steps = profile_kwargs["continuation_repair_steps"]
    repetition_unlikelihood_w = profile_kwargs["repetition_unlikelihood_w"]
    repetition_unlikelihood_window = profile_kwargs["repetition_unlikelihood_window"]
    selection_generation_n = profile_kwargs["selection_generation_n"]
    latent_concept_slots = profile_kwargs["latent_concept_slots"]
    latent_concept_memory_size = profile_kwargs["latent_concept_memory_size"]
    latent_concept_factorization_w = profile_kwargs[
        "latent_concept_factorization_w"]
    latent_concept_fer_w = profile_kwargs["latent_concept_fer_w"]
    latent_concept_memory_w = profile_kwargs["latent_concept_memory_w"]
    latent_concept_consolidation_w = profile_kwargs[
        "latent_concept_consolidation_w"]
    latent_concept_discovery_w = profile_kwargs["latent_concept_discovery_w"]
    latent_concept_gap_w = profile_kwargs["latent_concept_gap_w"]
    latent_concept_association_w = profile_kwargs["latent_concept_association_w"]
    latent_concept_composition_w = profile_kwargs["latent_concept_composition_w"]
    latent_concept_graph_predict_w = profile_kwargs[
        "latent_concept_graph_predict_w"]
    latent_concept_bridge_w = profile_kwargs["latent_concept_bridge_w"]
    latent_concept_completion_w = profile_kwargs["latent_concept_completion_w"]
    latent_concept_sequence_w = profile_kwargs["latent_concept_sequence_w"]
    latent_concept_neighborhood_w = profile_kwargs[
        "latent_concept_neighborhood_w"]
    latent_concept_transition_w = profile_kwargs["latent_concept_transition_w"]
    latent_concept_cluster_w = profile_kwargs["latent_concept_cluster_w"]
    self_teach_w = profile_kwargs["self_teach_w"]
    latent_concept_fer_probe_n = profile_kwargs["latent_concept_fer_probe_n"]
    latent_concept_fer_hard_max = profile_kwargs["latent_concept_fer_hard_max"]
    latent_concept_fer_refresh_steps = profile_kwargs[
        "latent_concept_fer_refresh_steps"]
    latent_concept_discovery_probe_n = profile_kwargs[
        "latent_concept_discovery_probe_n"]
    latent_concept_discovery_hard_max = profile_kwargs[
        "latent_concept_discovery_hard_max"]
    latent_concept_discovery_refresh_steps = profile_kwargs[
        "latent_concept_discovery_refresh_steps"]
    latent_concept_completion_probe_n = profile_kwargs[
        "latent_concept_completion_probe_n"]
    latent_concept_completion_hard_max = profile_kwargs[
        "latent_concept_completion_hard_max"]
    latent_concept_completion_refresh_steps = profile_kwargs[
        "latent_concept_completion_refresh_steps"]
    representation_probe_n = profile_kwargs["representation_probe_n"]
    selection_rounds = profile_kwargs["selection_rounds"]
    selection_score_patience = profile_kwargs["selection_score_patience"]
    latent_weights = (
        latent_concept_w, latent_concept_factorization_w, latent_concept_memory_w,
        latent_concept_fer_w, latent_concept_consolidation_w,
        latent_concept_discovery_w,
        latent_concept_reanalysis_w,
        latent_concept_gap_w,
        latent_concept_association_w, latent_concept_composition_w,
        latent_concept_graph_predict_w, latent_concept_bridge_w,
        latent_concept_completion_w,
        latent_concept_sequence_w,
        latent_concept_neighborhood_w,
    latent_concept_transition_w, latent_concept_cluster_w)
    if float(decode_w) < 0.0:
        raise ValueError("decoder loss weight must be non-negative")
    continuation_repair_w = float(continuation_repair_w)
    continuation_repair_steps = int(continuation_repair_steps)
    continuation_repair_prompt_frac = float(continuation_repair_prompt_frac)
    continuation_repair_temperature = float(continuation_repair_temperature)
    continuation_repair_top_k = int(continuation_repair_top_k)
    if continuation_repair_w < 0.0:
        raise ValueError("continuation repair loss weight must be non-negative")
    if continuation_repair_steps < 0:
        raise ValueError("continuation repair steps must be non-negative")
    if (continuation_repair_prompt_frac <= 0.0
            or continuation_repair_prompt_frac >= 1.0):
        raise ValueError("continuation repair prompt fraction must be in (0, 1)")
    if continuation_repair_temperature < 0.0:
        raise ValueError("continuation repair temperature must be non-negative")
    if continuation_repair_top_k < 0:
        raise ValueError("continuation repair top-k must be non-negative")
    repetition_unlikelihood_w = float(repetition_unlikelihood_w)
    repetition_unlikelihood_window = int(repetition_unlikelihood_window)
    if repetition_unlikelihood_w < 0.0:
        raise ValueError("repetition unlikelihood loss weight must be non-negative")
    if repetition_unlikelihood_window < 0:
        raise ValueError("repetition unlikelihood window must be non-negative")
    source_balance_w = float(source_balance_w)
    if source_balance_w < 0.0:
        raise ValueError("multimodal source balance weight must be non-negative")
    representation_probe_n = int(representation_probe_n)
    if representation_probe_n < 0:
        raise ValueError(
            "multimodal representation probe count must be non-negative")
    text_transfer_probe_n = int(text_transfer_probe_n)
    if text_transfer_probe_n < 0:
        raise ValueError("text transfer probe count must be non-negative")
    text_transfer_score_min_delta = float(text_transfer_score_min_delta)
    if text_transfer_score_min_delta < 0.0:
        raise ValueError("text transfer score min delta must be non-negative")
    text_transfer_insight_accept_w = float(text_transfer_insight_accept_w)
    if text_transfer_insight_accept_w < 0.0:
        raise ValueError("text transfer insight accept weight must be non-negative")
    text_transfer_insight_min_delta = float(text_transfer_insight_min_delta)
    if text_transfer_insight_min_delta < 0.0:
        raise ValueError("text transfer insight min delta must be non-negative")
    self_teach_w = float(self_teach_w)
    if self_teach_w < 0.0:
        raise ValueError("multimodal self-teach weight must be non-negative")
    self_teach_history_prior_w = float(self_teach_history_prior_w)
    if self_teach_history_prior_w < 0.0:
        raise ValueError(
            "multimodal self-teach history prior weight must be non-negative")
    if any(float(w) < 0.0 for w in latent_weights):
        raise ValueError("latent concept loss weights must be non-negative")
    if int(latent_concept_fer_probe_n) < 0:
        raise ValueError("latent concept FER probe count must be non-negative")
    if int(latent_concept_fer_hard_max) < 0:
        raise ValueError("latent concept FER hard max must be non-negative")
    if int(latent_concept_fer_refresh_steps) < 0:
        raise ValueError("latent concept FER refresh steps must be non-negative")
    if int(latent_concept_discovery_probe_n) < 0:
        raise ValueError("latent concept discovery probe count must be non-negative")
    if int(latent_concept_discovery_hard_max) < 0:
        raise ValueError("latent concept discovery hard max must be non-negative")
    if int(latent_concept_discovery_refresh_steps) < 0:
        raise ValueError("latent concept discovery refresh steps must be non-negative")
    if int(latent_concept_completion_probe_n) < 0:
        raise ValueError("latent concept completion probe count must be non-negative")
    if int(latent_concept_completion_hard_max) < 0:
        raise ValueError("latent concept completion hard max must be non-negative")
    if int(latent_concept_completion_refresh_steps) < 0:
        raise ValueError("latent concept completion refresh steps must be non-negative")
    fer_hard_enabled = int(latent_concept_fer_hard_max) > 0
    discovery_hard_enabled = int(latent_concept_discovery_hard_max) > 0
    completion_hard_enabled = int(latent_concept_completion_hard_max) > 0
    hard_study_enabled = bool(
        fer_hard_enabled or discovery_hard_enabled or completion_hard_enabled)
    hard_study_strategy = (
        "discovery" if discovery_hard_enabled
        else "completion" if completion_hard_enabled
        else "fer" if fer_hard_enabled else "none")
    if discovery_hard_enabled and latent_concept_memory_size <= 0:
        raise ValueError("latent concept discovery hard study requires memory_size > 0")
    if float(latent_concept_consolidation_w) and latent_concept_memory_size <= 0:
        raise ValueError("latent concept consolidation requires memory_size > 0")
    if (any(float(w) > 0.0 for w in latent_weights)
            or latent_concept_memory_size > 0
            or hard_study_enabled) and latent_concept_slots <= 0:
        raise ValueError("latent concept losses require latent_concept_slots > 0")
    if float(latent_concept_consolidation_temperature) <= 0.0:
        raise ValueError("latent concept consolidation temperature must be positive")
    if float(latent_concept_consolidation_balance_w) < 0.0:
        raise ValueError("latent concept consolidation balance weight must be non-negative")
    if float(latent_concept_consolidation_anchor_w) < 0.0:
        raise ValueError("latent concept consolidation anchor weight must be non-negative")
    if float(latent_concept_consolidation_fer_w) < 0.0:
        raise ValueError("latent concept consolidation FER weight must be non-negative")
    discovery_component_weights = (
        latent_concept_discovery_curiosity_w,
        latent_concept_discovery_graph_w,
        latent_concept_discovery_cycle_w,
        latent_concept_discovery_bridge_w,
        latent_concept_discovery_fer_w)
    if any(float(w) < 0.0 for w in discovery_component_weights):
        raise ValueError("latent concept discovery component weights must be non-negative")
    if float(latent_concept_discovery_w) and latent_concept_memory_size <= 0:
        raise ValueError("latent concept discovery requires memory_size > 0")
    reanalysis_component_weights = (
        latent_concept_reanalysis_graph_w,
        latent_concept_reanalysis_cycle_w,
        latent_concept_reanalysis_bridge_w,
        latent_concept_reanalysis_fer_w,
        latent_concept_reanalysis_cycle_consistency_w)
    if any(float(w) < 0.0 for w in reanalysis_component_weights):
        raise ValueError("latent concept reanalysis component weights must be non-negative")
    if float(latent_concept_reanalysis_w) and latent_concept_memory_size <= 0:
        raise ValueError("latent concept reanalysis requires memory_size > 0")
    if float(latent_concept_gap_w) and latent_concept_memory_size <= 0:
        raise ValueError("latent concept gap loss requires memory_size > 0")
    if float(latent_concept_gap_temperature) <= 0.0:
        raise ValueError("latent concept gap temperature must be positive")
    if float(latent_concept_gap_self_loop_w) < 0.0:
        raise ValueError("latent concept gap self-loop weight must be non-negative")
    if int(latent_concept_gap_transitive_steps) < 1:
        raise ValueError("latent concept gap transitive steps must be positive")
    if float(latent_concept_gap_transitive_w) < 0.0:
        raise ValueError("latent concept gap transitive weight must be non-negative")
    if float(latent_concept_gap_target_power) <= 0.0:
        raise ValueError("latent concept gap target power must be positive")
    if int(latent_concept_sequence_batch) < 0 or int(latent_concept_sequence_batch) == 1:
        raise ValueError("latent concept sequence batch must be 0 or at least 2")
    if float(latent_concept_completion_temperature) <= 0.0:
        raise ValueError("latent concept completion temperature must be positive")
    if float(latent_concept_sequence_temperature) <= 0.0:
        raise ValueError("latent concept sequence temperature must be positive")
    if selection_score_metric not in MULTIMODAL_SCORE_METRICS:
        raise ValueError(
            f"unknown multimodal selection score metric {selection_score_metric!r}")
    if int(selection_rounds) <= 0:
        raise ValueError("multimodal selection rounds must be positive")
    if float(selection_score_min_delta) < 0.0:
        raise ValueError("multimodal selection score min delta must be non-negative")
    if int(selection_score_patience) < 0:
        raise ValueError("multimodal selection score patience must be non-negative")
    selection_signal_regression_tolerance = float(
        selection_signal_regression_tolerance)
    if selection_signal_regression_tolerance < 0.0:
        raise ValueError(
            "multimodal selection signal regression tolerance must be non-negative")
    if int(selection_eval_n) < 0:
        raise ValueError("multimodal selection eval count must be non-negative")
    selection_generation_n = int(selection_generation_n)
    selection_generation_prompt_tokens = int(selection_generation_prompt_tokens)
    selection_generation_max_new_tokens = int(selection_generation_max_new_tokens)
    selection_generation_temperature = float(selection_generation_temperature)
    selection_generation_top_k = int(selection_generation_top_k)
    if selection_generation_n < 0:
        raise ValueError(
            "multimodal selection generation count must be non-negative")
    if selection_generation_prompt_tokens <= 0:
        raise ValueError(
            "multimodal selection generation prompt tokens must be positive")
    if selection_generation_max_new_tokens < 0:
        raise ValueError(
            "multimodal selection generation max new tokens must be non-negative")
    if selection_generation_temperature < 0.0:
        raise ValueError(
            "multimodal selection generation temperature must be non-negative")
    if selection_generation_top_k < 0:
        raise ValueError(
            "multimodal selection generation top-k must be non-negative")
    selection_insight_accept_w = float(selection_insight_accept_w)
    if selection_insight_accept_w < 0.0:
        raise ValueError("multimodal selection insight accept weight must be non-negative")
    selection_insight_min_delta = float(selection_insight_min_delta)
    if selection_insight_min_delta < 0.0:
        raise ValueError("multimodal selection insight min delta must be non-negative")
    records = load_manifest(manifest, root=root)
    requested_decode_objective = str(decode_objective)
    decode_objective = infer_multimodal_decode_objective(
        records, requested=requested_decode_objective)
    decode_objective_info = multimodal_decode_objective_report(
        records, requested_decode_objective, decode_objective)
    train_records, eval_records = split_records(records)
    training_sampling_weights = multimodal_training_sampling_weights(
        train_records, source_balance_w=source_balance_w)
    source_balance_train_report = multimodal_source_balance_report(
        train_records, source_balance_w, weights=training_sampling_weights)
    sequence_batch = int(latent_concept_sequence_batch) or max(2, batch // 2)
    sequence_pairs, sequence_report = mine_multimodal_sequence_pairs(
        records, split="train")
    view_dims = feature_dims(records)
    active_modes = multimodal_active_modes(view_dims, records)
    self_teach_available_objectives = multimodal_self_teach_available_objectives(
        latent_concept_slots=latent_concept_slots,
        active_mode_count=len(active_modes))
    selection_generation_kwargs = {
        "generation_eval_n": int(selection_generation_n),
        "generation_prompt_tokens": int(selection_generation_prompt_tokens),
        "generation_max_new_tokens": int(selection_generation_max_new_tokens),
        "generation_temperature": float(selection_generation_temperature),
        "generation_top_k": int(selection_generation_top_k),
    }
    vocab = build_vocab(records, max_size=(int(max_vocab) or None))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MultimodalLM(
        len(vocab), view_dims=view_dims, d=d, layers=layers,
        heads=heads, pad=vocab.pad, max_len=max_len, view_tokens=view_tokens,
        txt_tokens=txt_tokens, trunk_arch=trunk_arch,
        trunk_width=trunk_width, trunk_depth=trunk_depth, text_layers=text_layers,
        text_arch=text_arch, modality_dropout=modality_dropout,
        concept_tokens=concept_tokens, fusion_layers=fusion_layers,
        latent_concept_slots=latent_concept_slots,
        latent_concept_layers=latent_concept_layers,
        latent_concept_prefix=latent_concept_prefix,
        latent_concept_memory_size=latent_concept_memory_size,
        latent_concept_topk=latent_concept_topk).to(device)
    text_transfer_pre_state = None
    text_transfer_before_bundle = None
    if text_checkpoint and text_transfer_probe_n > 0:
        text_transfer_pre_state = _model_state_copy(model)
        text_transfer_before_bundle = multimodal_eval_bundle(
            model, eval_records, vocab, view_dims, n=text_transfer_probe_n,
            seed=seed + 149, device=device,
            score_metric=selection_score_metric,
            score_margin_w=selection_score_margin_w,
            decode_objective=decode_objective,
            **selection_generation_kwargs)
    if text_checkpoint:
        text_import_report = import_text_checkpoint(
            model, vocab, text_checkpoint, device=device)
        if text_transfer_before_bundle is not None:
            text_transfer_after_bundle = multimodal_eval_bundle(
                model, eval_records, vocab, view_dims, n=text_transfer_probe_n,
                seed=seed + 149, device=device,
                score_metric=selection_score_metric,
                score_margin_w=selection_score_margin_w,
                decode_objective=decode_objective,
                **selection_generation_kwargs)
            calibration = multimodal_transfer_calibration_report(
                text_transfer_before_bundle, text_transfer_after_bundle,
                score_min_delta=text_transfer_score_min_delta,
                insight_accept_w=text_transfer_insight_accept_w,
                insight_min_delta=text_transfer_insight_min_delta,
                gate=text_transfer_gate)
            text_import_report["target_calibration"] = calibration
            text_import_report["accepted"] = bool(calibration["accepted"])
            text_import_report["reverted"] = bool(calibration["reverted"])
            text_import_report["target_probe_n"] = int(text_transfer_probe_n)
            if calibration["reverted"] and text_transfer_pre_state is not None:
                model.load_state_dict(text_transfer_pre_state, strict=False)
        else:
            text_import_report["accepted"] = True
            text_import_report["reverted"] = False
            text_import_report["target_probe_n"] = 0
            text_import_report["target_calibration"] = {
                "enabled": False,
                "accepted": True,
                "gate_enabled": bool(text_transfer_gate),
            }
        model.text_checkpoint_transfer = text_import_report
    else:
        model.text_checkpoint_transfer = {}
    if multimodal_checkpoint:
        import_multimodal_checkpoint(
            model, vocab, multimodal_checkpoint, device=device)
    else:
        model.multimodal_checkpoint_transfer = {}
    text_checkpoint_history_prior = {"enabled": False, "entry_count": 0,
                                     "signal_deficits": {}}
    text_checkpoint_history_prior_raw = {"enabled": False, "entry_count": 0,
                                         "signal_deficits": {}}
    if text_checkpoint:
        text_checkpoint_history_prior_raw = multimodal_text_history_self_teach_prior(
            load_text_checkpoint_payload(text_checkpoint, device="cpu"),
            enabled=self_teach_w > 0.0)
        text_transfer_accepted = bool(
            getattr(model, "text_checkpoint_transfer", {}).get("accepted", True))
        if text_transfer_accepted:
            text_checkpoint_history_prior = text_checkpoint_history_prior_raw
        else:
            text_checkpoint_history_prior = (
                {
                    "enabled": False,
                    "entry_count": int(
                        text_checkpoint_history_prior_raw.get("entry_count", 0) or 0),
                    "signal_deficits": {},
                    "top_signal": None,
                    "concept_connection_signal": 0.0,
                    "disabled_reason": "text_transfer_rejected_by_target_probe",
                    "transfer_accepted": False,
                    "raw_enabled": bool(
                        text_checkpoint_history_prior_raw.get("enabled", False)),
                    "raw_signal_deficits": (
                        text_checkpoint_history_prior_raw.get(
                            "signal_deficits", {})),
                    "raw_top_signal": text_checkpoint_history_prior_raw.get(
                        "top_signal"),
                })
    multimodal_checkpoint_history_prior = {
        "enabled": False, "entry_count": 0, "signal_deficits": {}}
    if multimodal_checkpoint:
        multimodal_checkpoint_history_prior = (
            multimodal_checkpoint_representation_self_teach_prior(
                load_multimodal_checkpoint_payload(
                    multimodal_checkpoint, device="cpu"),
                enabled=self_teach_w > 0.0))
    self_teach_history_prior = merge_multimodal_self_teach_priors(
        text_checkpoint_history_prior, multimodal_checkpoint_history_prior)
    weight_update_before = multimodal_weight_update_snapshot(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    study_pool = []
    study_reports = []
    last = {}
    selection = {"enabled": False}
    selection_boundaries = {}
    best_state = None
    best_score = 0.0
    best_round = 0
    best_bundle = None
    best_bridge_eval = None
    best_metrics = {}
    best_study_reports = []
    rounds_report = []
    no_improve_rounds = 0
    stopped_early = False
    stop_round = 0
    self_teach_base_weights = {
        "decode_w": float(decode_w),
        "continuation_repair_w": float(continuation_repair_w),
        "repetition_unlikelihood_w": float(repetition_unlikelihood_w),
        "agreement_w": float(agreement_w),
        "latent_concept_w": float(latent_concept_w),
        "latent_concept_factorization_w": float(latent_concept_factorization_w),
        "latent_concept_fer_w": float(latent_concept_fer_w),
        "latent_concept_discovery_w": float(latent_concept_discovery_w),
        "latent_concept_gap_w": float(latent_concept_gap_w),
        "latent_concept_graph_predict_w": float(latent_concept_graph_predict_w),
        "latent_concept_bridge_w": float(latent_concept_bridge_w),
        "latent_concept_completion_w": float(latent_concept_completion_w),
        "latent_concept_sequence_w": float(latent_concept_sequence_w),
        "latent_concept_transition_w": float(latent_concept_transition_w),
    }
    self_teach_effective_weights = dict(self_teach_base_weights)
    self_teach_plan = None
    self_teach_reports = []

    def set_objective_weights(weights):
        nonlocal decode_w, agreement_w, latent_concept_w
        nonlocal continuation_repair_w, repetition_unlikelihood_w
        nonlocal latent_concept_factorization_w, latent_concept_fer_w
        nonlocal latent_concept_discovery_w, latent_concept_gap_w
        nonlocal latent_concept_graph_predict_w, latent_concept_bridge_w
        nonlocal latent_concept_completion_w, latent_concept_sequence_w
        nonlocal latent_concept_transition_w
        decode_w = weights["decode_w"]
        continuation_repair_w = weights["continuation_repair_w"]
        repetition_unlikelihood_w = weights["repetition_unlikelihood_w"]
        agreement_w = weights["agreement_w"]
        latent_concept_w = weights["latent_concept_w"]
        latent_concept_factorization_w = (
            weights["latent_concept_factorization_w"])
        latent_concept_fer_w = weights["latent_concept_fer_w"]
        latent_concept_discovery_w = weights["latent_concept_discovery_w"]
        latent_concept_gap_w = weights["latent_concept_gap_w"]
        latent_concept_graph_predict_w = (
            weights["latent_concept_graph_predict_w"])
        latent_concept_bridge_w = weights["latent_concept_bridge_w"]
        latent_concept_completion_w = weights["latent_concept_completion_w"]
        latent_concept_sequence_w = weights["latent_concept_sequence_w"]
        latent_concept_transition_w = weights["latent_concept_transition_w"]

    def update_self_teach_weights(score_components=None, round_id=1):
        nonlocal self_teach_plan, self_teach_effective_weights
        plan, _base, effective = multimodal_self_teach_weight_maps(
            score_components, budget=self_teach_w,
            history_prior=self_teach_history_prior,
            history_prior_w=self_teach_history_prior_w,
            available_objectives=self_teach_available_objectives,
            **self_teach_base_weights)
        self_teach_plan = plan
        self_teach_effective_weights = effective
        set_objective_weights(effective)
        if plan is not None:
            self_teach_reports.append(plan | {"round": int(round_id)})
        return plan

    def current_latent_weights():
        return (
            latent_concept_w, latent_concept_factorization_w,
            latent_concept_memory_w, latent_concept_fer_w,
            latent_concept_consolidation_w, latent_concept_discovery_w,
            latent_concept_reanalysis_w, latent_concept_gap_w,
            latent_concept_association_w, latent_concept_composition_w,
            latent_concept_graph_predict_w, latent_concept_bridge_w,
            latent_concept_completion_w, latent_concept_sequence_w,
            latent_concept_neighborhood_w, latent_concept_transition_w,
            latent_concept_cluster_w)

    def selection_row(round_id, round_steps, bundle):
        score = float(bundle["score_components"]["score"])
        return {
            "round": int(round_id),
            "steps": int(round_steps),
            "selected": False,
            "score": score,
            "score_components": bundle["score_components"],
            "teacher_forced": bundle["teacher_forced"],
            "generation": bundle.get("generation", {}),
            "latent_fer": bundle["latent_fer"],
            "latent_bridge": bundle["latent_bridge"],
            "latent_sequence": bundle["latent_sequence"],
            "self_teach_plan": self_teach_plan,
            "self_teach_effective_weights": dict(self_teach_effective_weights),
        }

    before_bundle = None
    representation_before_bundle = None
    representation_progress_probe_n = 0
    representation_progress_seed = seed
    if select_best or self_teach_w > 0.0:
        before_bundle = multimodal_eval_bundle(
            model, eval_records, vocab, view_dims, n=selection_eval_n,
            seed=seed, device=device, score_metric=selection_score_metric,
            score_margin_w=selection_score_margin_w,
            decode_objective=decode_objective,
            **selection_generation_kwargs)
        representation_before_bundle = before_bundle
        representation_progress_probe_n = int(selection_eval_n)
        representation_progress_seed = seed
    if representation_probe_n > 0:
        representation_before_bundle = multimodal_eval_bundle(
            model, eval_records, vocab, view_dims, n=representation_probe_n,
            seed=seed + 211, device=device,
            score_metric=selection_score_metric,
            score_margin_w=selection_score_margin_w,
            decode_objective=decode_objective,
            **selection_generation_kwargs)
        representation_progress_probe_n = int(representation_probe_n)
        representation_progress_seed = seed + 211
    if self_teach_w > 0.0:
        update_self_teach_weights(
            before_bundle["score_components"], round_id=1)

    if select_best:
        schedule = _step_schedule(steps, selection_rounds)
        if not schedule:
            raise ValueError("multimodal selected training requires at least one step")
        cursor = 0
        for round_id, round_steps in enumerate(schedule, start=1):
            cursor += int(round_steps)
            selection_boundaries[cursor] = (round_id, int(round_steps))
        initial_row = selection_row(0, 0, before_bundle)
        best_state = _model_state_copy(model)
        best_score = float(initial_row["score"])
        best_bundle = before_bundle
        best_bridge_eval = initial_row["latent_bridge"]
        best_metrics = {"selection_initial": True}
        rounds_report = [initial_row]

    def selected_id_sample(selected, limit=16):
        return [rec.rec_id for rec in selected[:int(limit)]]

    def refresh_study_pool(step):
        nonlocal study_pool
        if hard_study_strategy == "discovery":
            probe_n = (int(latent_concept_discovery_probe_n)
                       if int(latent_concept_discovery_probe_n) > 0
                       else max(batch * 4, 1))
            hard_max = int(latent_concept_discovery_hard_max)
            selected, report = latent_multimodal_discovery_examples(
                model, train_records, vocab, view_dims, n=probe_n,
                seed=seed + 1301 + int(step), device=device,
                curiosity_temperature=latent_concept_association_temperature,
                curiosity_self_loop_w=latent_concept_association_self_loop_w,
                curiosity_transitive_steps=latent_concept_association_transitive_steps,
                curiosity_transitive_w=latent_concept_association_transitive_w,
                graph_temperature=latent_concept_graph_predict_temperature,
                graph_self_loop_w=latent_concept_graph_predict_self_loop_w,
                graph_transitive_steps=latent_concept_graph_predict_transitive_steps,
                graph_transitive_w=latent_concept_graph_predict_transitive_w,
                graph_target_power=latent_concept_graph_predict_target_power,
                cycle_temperature=latent_concept_graph_predict_temperature,
                cycle_self_loop_w=latent_concept_graph_predict_self_loop_w,
                cycle_transitive_steps=latent_concept_graph_predict_transitive_steps,
                cycle_transitive_w=latent_concept_graph_predict_transitive_w,
                cycle_target_power=latent_concept_graph_predict_target_power,
                sequence_temperature=latent_concept_sequence_temperature,
                completion_temperature=latent_concept_completion_temperature,
                decode_objective=decode_objective)
        elif hard_study_strategy == "completion":
            probe_n = (int(latent_concept_completion_probe_n)
                       if int(latent_concept_completion_probe_n) > 0
                       else max(batch * 4, 1))
            hard_max = int(latent_concept_completion_hard_max)
            selected, report = latent_multimodal_completion_examples(
                model, train_records, vocab, view_dims, n=probe_n,
                seed=seed + 1301 + int(step), device=device,
                temperature=latent_concept_completion_temperature,
                decode_objective=decode_objective)
        else:
            probe_n = (int(latent_concept_fer_probe_n)
                       if int(latent_concept_fer_probe_n) > 0
                       else max(batch * 4, 1))
            hard_max = int(latent_concept_fer_hard_max)
            selected, report = latent_multimodal_fer_examples(
                model, train_records, vocab, view_dims, n=probe_n,
                seed=seed + 1301 + int(step), device=device,
                decode_objective=decode_objective)
        selected = selected[:hard_max]
        if selected:
            study_pool = selected
        report = report | {
            "step": int(step),
            "strategy": hard_study_strategy,
            "capped": True,
            "n_error_records_used": len(selected),
            "pool_active": bool(study_pool),
            "pool_size": len(study_pool),
            "hard_record_ids": selected_id_sample(selected),
            "hard_record_count": len(selected),
        }
        study_reports.append(report)

    for st in range(1, int(steps) + 1):
        model.train()
        refresh_steps = (
            int(latent_concept_discovery_refresh_steps)
            if hard_study_strategy == "discovery"
            else int(latent_concept_completion_refresh_steps)
            if hard_study_strategy == "completion"
            else int(latent_concept_fer_refresh_steps))
        refresh_due = (
            hard_study_enabled
            and (not study_pool or st == 1 or (
                refresh_steps and (st - 1) % refresh_steps == 0)))
        if refresh_due:
            refresh_study_pool(st)
            model.train()
        batch_source = study_pool if study_pool else train_records
        batch_weights = (
            training_sampling_weights if batch_source is train_records
            else multimodal_training_sampling_weights(
                batch_source, source_balance_w=source_balance_w))
        features, txt, ids, decode_txt = _batch_from_records(
            _sample_records(batch_source, batch, rng, weights=batch_weights),
            vocab, device, view_dims, decode_objective=decode_objective,
            return_decode_text=True)
        needs_latent = bool(any(float(w) > 0.0 for w in current_latent_weights())
                            or latent_concept_memory_size > 0
                            or hard_study_enabled)
        needs_logits = bool(float(decode_w) > 0.0 or float(agreement_w) > 0.0
                            or float(continuation_repair_w) > 0.0
                            or float(repetition_unlikelihood_w) > 0.0)
        if decode_txt is txt or not needs_latent:
            bundles_by_mode = {
                mode: model.mode_bundle(
                    features, decode_txt, ids, mode=mode,
                    need_latent=needs_latent,
                    latent_view_dropout=latent_concept_view_dropout,
                    latent_project=True,
                    need_logits=needs_logits)
                for mode in active_modes
            }
            latent_bundles_by_mode = bundles_by_mode
            logit_bundles_by_mode = bundles_by_mode
        else:
            latent_bundles_by_mode = {
                mode: model.mode_bundle(
                    features, txt, ids, mode=mode, need_latent=True,
                    latent_view_dropout=latent_concept_view_dropout,
                    latent_project=True, need_logits=False)
                for mode in active_modes
            }
            logit_bundles_by_mode = {
                mode: model.mode_bundle(
                    features, decode_txt, ids, mode=mode, need_latent=False,
                    latent_view_dropout=0.0, latent_project=False,
                    need_logits=needs_logits)
                for mode in active_modes
            }
        logits_by_mode = {mode: bundle["logits"]
                          for mode, bundle in logit_bundles_by_mode.items()}
        base_loss = sum(token_loss(logits, ids, vocab.pad)
                        for logits in logits_by_mode.values()) / len(logits_by_mode)
        agreement = (logit_agreement_loss(logits_by_mode, ids, vocab.pad)
                     if agreement_w else base_loss * 0.0)
        if repetition_unlikelihood_w:
            repetition_unlikelihood, repetition_unlikelihood_metrics = (
                repetition_unlikelihood_loss(
                    logits_by_mode, ids, vocab.pad,
                    window=repetition_unlikelihood_window))
        else:
            repetition_unlikelihood = base_loss * 0.0
            repetition_unlikelihood_metrics = {
                "enabled": False,
                "skipped": True,
                "skip_reason": "weight_zero",
                "tokens": 0,
                "candidates": 0,
                "window": int(repetition_unlikelihood_window),
            }
        if continuation_repair_w and decode_objective == "causal":
            continuation_repair, continuation_repair_metrics = (
                continuation_repair_loss(
                    model, features, decode_txt, ids, active_modes, vocab.pad,
                    repair_steps=continuation_repair_steps,
                    prompt_frac=continuation_repair_prompt_frac,
                    temperature=continuation_repair_temperature,
                    top_k=continuation_repair_top_k))
        else:
            continuation_repair = base_loss * 0.0
            continuation_repair_metrics = {
                "enabled": False,
                "skipped": True,
                "skip_reason": (
                    "weight_zero" if not continuation_repair_w
                    else "requires_causal_decode"),
                "tokens": 0,
                "generated_tokens": 0,
                "changed_tokens": 0,
                "steps": int(continuation_repair_steps),
                "prompt_frac": float(continuation_repair_prompt_frac),
                "temperature": float(continuation_repair_temperature),
                "top_k": int(continuation_repair_top_k),
            }
        latent_views = {
            mode: bundle.get("latent_concepts")
            for mode, bundle in latent_bundles_by_mode.items()
        }
        latent_concept = (
            latent_multimodal_concept_loss_from_views(
                latent_views, invariance_w=latent_concept_invariance_w,
                variance_w=latent_concept_variance_w,
                covariance_w=latent_concept_covariance_w,
                variance_target=latent_concept_variance_target)
            if latent_concept_w else base_loss * 0.0)
        latent_factorization = (
            latent_multimodal_factorization_loss_from_views(
                latent_views,
                variance_target=latent_concept_factorization_variance,
                separation_margin=latent_concept_factorization_margin,
                covariance_w=latent_concept_factorization_covariance_w)
            if latent_concept_factorization_w else base_loss * 0.0)
        latent_fer = (
            latent_multimodal_fer_loss_from_views(
                latent_views,
                fragmentation_w=latent_concept_fer_fragmentation_w,
                correlation_w=latent_concept_fer_correlation_w,
                balance_w=latent_concept_fer_balance_w)
            if latent_concept_fer_w else base_loss * 0.0)
        latent_memory = (
            latent_multimodal_memory_loss_from_views(
                model, latent_views, temperature=latent_concept_memory_temperature,
                balance_w=latent_concept_memory_balance_w)
            if latent_concept_memory_w else base_loss * 0.0)
        if latent_concept_consolidation_w:
            latent_consolidation, consolidation_metrics = (
                latent_multimodal_memory_consolidation_loss_from_views(
                    model, latent_views,
                    temperature=latent_concept_consolidation_temperature,
                    balance_w=latent_concept_consolidation_balance_w,
                    anchor_w=latent_concept_consolidation_anchor_w,
                    fer_w=latent_concept_consolidation_fer_w,
                    fer_fragmentation_w=latent_concept_fer_fragmentation_w,
                    fer_correlation_w=latent_concept_fer_correlation_w,
                    fer_balance_w=latent_concept_fer_balance_w))
        else:
            latent_consolidation = base_loss * 0.0
            memory = getattr(model, "latent_concept_memory", None)
            consolidation_zero = base_loss.detach() * 0.0
            consolidation_metrics = {
                "memory_loss": consolidation_zero,
                "anchor_loss": consolidation_zero,
                "fer_loss": consolidation_zero,
                "nearest_cosine": consolidation_zero,
                "memory_active": int(
                    getattr(memory, "filled", torch.zeros((), dtype=torch.long)).item())
                if memory is not None else 0,
                "skipped": True,
            }
        latent_association = (
            latent_multimodal_association_loss_from_views(
                model, latent_views,
                temperature=latent_concept_association_temperature,
                target_power=latent_concept_association_target_power,
                self_loop_w=latent_concept_association_self_loop_w,
                transitive_steps=latent_concept_association_transitive_steps,
                transitive_w=latent_concept_association_transitive_w)
            if latent_concept_association_w else base_loss * 0.0)
        if latent_concept_discovery_w:
            latent_discovery, discovery_metrics = (
                latent_multimodal_discovery_loss_from_views(
                    model, latent_views,
                    curiosity_w=latent_concept_discovery_curiosity_w,
                    graph_w=latent_concept_discovery_graph_w,
                    cycle_w=latent_concept_discovery_cycle_w,
                    bridge_w=latent_concept_discovery_bridge_w,
                    fer_w=latent_concept_discovery_fer_w,
                    curiosity_temperature=latent_concept_association_temperature,
                    curiosity_self_loop_w=latent_concept_association_self_loop_w,
                    curiosity_transitive_steps=(
                        latent_concept_association_transitive_steps),
                    curiosity_transitive_w=(
                        latent_concept_association_transitive_w),
                    graph_temperature=latent_concept_graph_predict_temperature,
                    graph_self_loop_w=latent_concept_graph_predict_self_loop_w,
                    graph_transitive_steps=(
                        latent_concept_graph_predict_transitive_steps),
                    graph_transitive_w=(
                        latent_concept_graph_predict_transitive_w),
                    graph_target_power=latent_concept_graph_predict_target_power,
                    cycle_temperature=latent_concept_graph_predict_temperature,
                    cycle_self_loop_w=latent_concept_graph_predict_self_loop_w,
                    cycle_transitive_steps=(
                        latent_concept_graph_predict_transitive_steps),
                    cycle_transitive_w=(
                        latent_concept_graph_predict_transitive_w),
                    cycle_target_power=latent_concept_graph_predict_target_power,
                    fer_fragmentation_w=latent_concept_fer_fragmentation_w,
                    fer_correlation_w=latent_concept_fer_correlation_w,
                    fer_balance_w=latent_concept_fer_balance_w))
        else:
            latent_discovery = base_loss * 0.0
            discovery_zero = base_loss.detach() * 0.0
            memory = getattr(model, "latent_concept_memory", None)
            discovery_metrics = {
                "curiosity_loss": discovery_zero,
                "curiosity_novelty": discovery_zero,
                "curiosity_association": discovery_zero,
                "graph_loss": discovery_zero,
                "graph_kl": discovery_zero,
                "graph_cosine": discovery_zero,
                "cycle_loss": discovery_zero,
                "cycle_forward_kl": discovery_zero,
                "cycle_reverse_kl": discovery_zero,
                "cycle_source_cycle_kl": discovery_zero,
                "cycle_target_cycle_kl": discovery_zero,
                "insight_loss": discovery_zero,
                "insight_score": discovery_zero,
                "insight_kl": discovery_zero,
                "insight_cosine": discovery_zero,
                "insight_missing_mass": discovery_zero,
                "insight_reachable_mass": discovery_zero,
                "insight_gain": discovery_zero,
                "bridge_loss": discovery_zero,
                "bridge_score": discovery_zero,
                "bridge_entropy": discovery_zero,
                "bridge_connectivity": discovery_zero,
                "fer_loss": discovery_zero,
                "memory_active": int(
                    getattr(memory, "filled", torch.zeros((), dtype=torch.long)).item())
                if memory is not None else 0,
                "graph_ready": False,
                "skipped": True,
            }
        if latent_concept_reanalysis_w:
            latent_reanalysis, reanalysis_metrics = (
                latent_multimodal_reanalysis_loss_from_views(
                    model, latent_views,
                    graph_w=latent_concept_reanalysis_graph_w,
                    cycle_w=latent_concept_reanalysis_cycle_w,
                    bridge_w=latent_concept_reanalysis_bridge_w,
                    fer_w=latent_concept_reanalysis_fer_w,
                    temperature=latent_concept_graph_predict_temperature,
                    self_loop_w=latent_concept_graph_predict_self_loop_w,
                    transitive_steps=(
                        latent_concept_graph_predict_transitive_steps),
                    transitive_w=(
                        latent_concept_graph_predict_transitive_w),
                    target_power=latent_concept_graph_predict_target_power,
                    cycle_consistency_w=(
                        latent_concept_reanalysis_cycle_consistency_w),
                    fer_fragmentation_w=latent_concept_fer_fragmentation_w,
                    fer_correlation_w=latent_concept_fer_correlation_w,
                    fer_balance_w=latent_concept_fer_balance_w))
        else:
            latent_reanalysis = base_loss * 0.0
            reanalysis_zero = base_loss.detach() * 0.0
            memory = getattr(model, "latent_concept_memory", None)
            reanalysis_metrics = {
                "closure_loss": reanalysis_zero,
                "closure_kl": reanalysis_zero,
                "closure_cosine": reanalysis_zero,
                "cycle_loss": reanalysis_zero,
                "bridge_loss": reanalysis_zero,
                "fer_loss": reanalysis_zero,
                "memory_active": int(
                    getattr(memory, "filled", torch.zeros((), dtype=torch.long)).item())
                if memory is not None else 0,
                "graph_ready": False,
                "skipped": True,
            }
        if latent_concept_gap_w:
            latent_gap, gap_metrics = latent_multimodal_gap_loss_from_views(
                model, latent_views,
                temperature=latent_concept_gap_temperature,
                self_loop_w=latent_concept_gap_self_loop_w,
                transitive_steps=latent_concept_gap_transitive_steps,
                transitive_w=latent_concept_gap_transitive_w,
                target_power=latent_concept_gap_target_power)
        else:
            latent_gap = base_loss * 0.0
            gap_zero = base_loss.detach() * 0.0
            memory = getattr(model, "latent_concept_memory", None)
            gap_metrics = {
                "gap_loss": gap_zero,
                "gap_kl": gap_zero,
                "gap_cosine": gap_zero,
                "gap_entropy": gap_zero,
                "gap_target_mass": gap_zero,
                "gap_present_overlap": gap_zero,
                "memory_active": int(
                    getattr(memory, "filled", torch.zeros((), dtype=torch.long)).item())
                if memory is not None else 0,
                "graph_ready": False,
                "skipped": True,
            }
        latent_composition = (
            latent_multimodal_composition_loss_from_views(
                model, latent_views,
                temperature=latent_concept_composition_temperature,
                self_loop_w=latent_concept_composition_self_loop_w,
                transitive_steps=latent_concept_composition_transitive_steps,
                transitive_w=latent_concept_composition_transitive_w,
                margin=latent_concept_composition_margin)
            if latent_concept_composition_w else base_loss * 0.0)
        latent_graph_predict = (
            latent_multimodal_graph_prediction_loss_from_views(
                model, latent_views,
                temperature=latent_concept_graph_predict_temperature,
                self_loop_w=latent_concept_graph_predict_self_loop_w,
                transitive_steps=latent_concept_graph_predict_transitive_steps,
                transitive_w=latent_concept_graph_predict_transitive_w,
                target_power=latent_concept_graph_predict_target_power)
            if latent_concept_graph_predict_w else base_loss * 0.0)
        latent_bridge = (
            latent_multimodal_bridge_loss_from_views(model, latent_views)
            if latent_concept_bridge_w else base_loss * 0.0)
        if latent_concept_completion_w:
            latent_completion, completion_metrics = (
                latent_multimodal_completion_loss_from_views(
                    model, latent_views,
                    temperature=latent_concept_completion_temperature))
        else:
            latent_completion = base_loss * 0.0
            completion_zero = base_loss.detach() * 0.0
            completion_metrics = {
                "completion_loss": completion_zero,
                "view_count": 0,
                "skipped": True,
                "modes": {},
            }
        latent_sequence = base_loss * 0.0
        if latent_concept_sequence_w and sequence_pairs:
            sequence_pair_batch = _sample_pairs(sequence_pairs, sequence_batch, rng)
            latent_sequence = latent_multimodal_sequence_prediction_loss(
                model, sequence_pair_batch, vocab, view_dims, device=device,
                temperature=latent_concept_sequence_temperature,
                view_dropout=latent_concept_view_dropout,
                decode_objective=decode_objective)
        latent_neighborhood = (
            latent_multimodal_neighborhood_loss_from_views(
                latent_views, temperature=latent_concept_neighborhood_temperature,
                margin=latent_concept_neighborhood_margin)
            if latent_concept_neighborhood_w else base_loss * 0.0)
        latent_transition = (
            latent_multimodal_transition_loss_from_views(
                latent_views, temperature=latent_concept_transition_temperature,
                margin=latent_concept_transition_margin)
            if latent_concept_transition_w else base_loss * 0.0)
        latent_cluster = (
            latent_multimodal_cluster_loss_from_views(
                latent_views, temperature=latent_concept_cluster_temperature,
                margin=latent_concept_cluster_margin,
                min_cluster_size=latent_concept_cluster_min_size)
            if latent_concept_cluster_w else base_loss * 0.0)
        loss = (float(decode_w) * base_loss + float(agreement_w) * agreement
                + float(continuation_repair_w) * continuation_repair
                + float(repetition_unlikelihood_w) * repetition_unlikelihood
                + float(latent_concept_w) * latent_concept
                + float(latent_concept_factorization_w) * latent_factorization
                + float(latent_concept_fer_w) * latent_fer
                + float(latent_concept_memory_w) * latent_memory
                + float(latent_concept_consolidation_w) * latent_consolidation
                + float(latent_concept_discovery_w) * latent_discovery
                + float(latent_concept_reanalysis_w) * latent_reanalysis
                + float(latent_concept_gap_w) * latent_gap
                + float(latent_concept_association_w) * latent_association
                + float(latent_concept_composition_w) * latent_composition
                + float(latent_concept_graph_predict_w) * latent_graph_predict
                + float(latent_concept_bridge_w) * latent_bridge
                + float(latent_concept_completion_w) * latent_completion
                + float(latent_concept_sequence_w) * latent_sequence
                + float(latent_concept_neighborhood_w) * latent_neighborhood
                + float(latent_concept_transition_w) * latent_transition
                + float(latent_concept_cluster_w) * latent_cluster)
        opt.zero_grad()
        loss.backward()
        opt.step()
        memory_updates = int(update_multimodal_latent_memory(
            model, latent_views.get("full") if needs_latent else None,
            momentum=latent_concept_memory_momentum,
            relation_decay=(latent_concept_association_decay
                            if (latent_concept_association_w
                                or latent_concept_composition_w
                                or latent_concept_graph_predict_w
                                or latent_concept_completion_w
                                or latent_concept_discovery_w
                                or latent_concept_reanalysis_w
                                or latent_concept_gap_w
                                or latent_concept_bridge_w) else None)))
        transition_updates = 0
        if (latent_concept_graph_predict_w or latent_concept_bridge_w
                or latent_concept_discovery_w
                or latent_concept_reanalysis_w
                or latent_concept_gap_w
                or latent_concept_sequence_w):
            transition_updates = int(update_multimodal_latent_transitions(
                model, latent_views, decay=latent_concept_association_decay))
        sequence_transition_updates = 0
        if sequence_pairs and (latent_concept_sequence_w
                               or latent_concept_graph_predict_w
                               or latent_concept_gap_w
                               or latent_concept_bridge_w):
            sequence_pair_batch = _sample_pairs(sequence_pairs, sequence_batch, rng)
            sequence_transition_updates = int(update_multimodal_sequence_transitions(
                model, sequence_pair_batch, vocab, view_dims, device=device,
                decay=latent_concept_association_decay,
                decode_objective=decode_objective))
        fer_metrics = latent_multimodal_fer_metrics_from_views(latent_views)
        bridge_metrics = latent_multimodal_bridge_metrics_from_views(model, latent_views)
        last = {
            "loss": float(loss.detach()),
            "objective_profile": objective_profile,
            "objective_profile_report": objective_profile_report,
            "decode_objective": str(decode_objective),
            "decode_objective_report": decode_objective_info,
            "decode_w": float(decode_w),
            "token_loss": float(base_loss.detach()),
            "agreement_loss": float(agreement.detach()),
            "repetition_unlikelihood_loss": float(
                repetition_unlikelihood.detach()),
            "repetition_unlikelihood_w": float(repetition_unlikelihood_w),
            "repetition_unlikelihood_window": int(repetition_unlikelihood_window),
            "repetition_unlikelihood_enabled": bool(
                repetition_unlikelihood_metrics["enabled"]),
            "repetition_unlikelihood_skipped": bool(
                repetition_unlikelihood_metrics["skipped"]),
            "repetition_unlikelihood_skip_reason": str(
                repetition_unlikelihood_metrics.get("skip_reason", "")),
            "repetition_unlikelihood_tokens": int(
                repetition_unlikelihood_metrics["tokens"]),
            "repetition_unlikelihood_candidates": int(
                repetition_unlikelihood_metrics["candidates"]),
            "continuation_repair_loss": float(continuation_repair.detach()),
            "continuation_repair_w": float(continuation_repair_w),
            "continuation_repair_steps": int(continuation_repair_steps),
            "continuation_repair_prompt_frac": float(
                continuation_repair_prompt_frac),
            "continuation_repair_temperature": float(
                continuation_repair_temperature),
            "continuation_repair_top_k": int(continuation_repair_top_k),
            "continuation_repair_enabled": bool(
                continuation_repair_metrics["enabled"]),
            "continuation_repair_skipped": bool(
                continuation_repair_metrics["skipped"]),
            "continuation_repair_skip_reason": str(
                continuation_repair_metrics.get("skip_reason", "")),
            "continuation_repair_tokens": int(
                continuation_repair_metrics["tokens"]),
            "continuation_repair_generated_tokens": int(
                continuation_repair_metrics["generated_tokens"]),
            "continuation_repair_changed_tokens": int(
                continuation_repair_metrics["changed_tokens"]),
            "self_teach_w": float(self_teach_w),
            "self_teach_history_prior_w": float(self_teach_history_prior_w),
            **source_balance_train_report,
            "text_checkpoint_history_prior": text_checkpoint_history_prior,
            "multimodal_checkpoint_history_prior": (
                multimodal_checkpoint_history_prior),
            "self_teach_history_prior": self_teach_history_prior,
            "self_teach_plan": self_teach_plan,
            "self_teach_available_objectives": list(
                self_teach_available_objectives),
            "self_teach_base_weights": dict(self_teach_base_weights),
            "self_teach_effective_weights": dict(self_teach_effective_weights),
            "self_teach_reports": list(self_teach_reports),
            "latent_concept_loss": float(latent_concept.detach()),
            "latent_factorization_loss": float(latent_factorization.detach()),
            "latent_fer_loss": float(latent_fer.detach()),
            "latent_fer_w": float(latent_concept_fer_w),
            "latent_fer_fragmentation_w": float(latent_concept_fer_fragmentation_w),
            "latent_fer_correlation_w": float(latent_concept_fer_correlation_w),
            "latent_fer_balance_w": float(latent_concept_fer_balance_w),
            "latent_fer_score": float(fer_metrics["fer_score"]),
            "latent_fer_fragmentation": float(fer_metrics["fragmentation"]),
            "latent_fer_slot_correlation": float(fer_metrics["slot_correlation"]),
            "latent_fer_slot_imbalance": float(fer_metrics["slot_imbalance"]),
            "latent_fer_probe_n": int(latent_concept_fer_probe_n),
            "latent_fer_hard_max": int(latent_concept_fer_hard_max),
            "latent_fer_refresh_steps": int(latent_concept_fer_refresh_steps),
            "latent_fer_study_pool_size": (
                len(study_pool) if hard_study_strategy == "fer" else 0),
            "latent_fer_hard_record_ids": (
                selected_id_sample(study_pool) if hard_study_strategy == "fer"
                else []),
            "latent_fer_study_reports": (
                list(study_reports) if hard_study_strategy == "fer" else []),
            "latent_discovery_probe_n": int(latent_concept_discovery_probe_n),
            "latent_discovery_hard_max": int(latent_concept_discovery_hard_max),
            "latent_discovery_refresh_steps": int(
                latent_concept_discovery_refresh_steps),
            "latent_discovery_study_pool_size": (
                len(study_pool) if hard_study_strategy == "discovery" else 0),
            "latent_discovery_hard_record_ids": (
                selected_id_sample(study_pool)
                if hard_study_strategy == "discovery" else []),
            "latent_discovery_study_reports": (
                list(study_reports)
                if hard_study_strategy == "discovery" else []),
            "latent_completion_probe_n": int(latent_concept_completion_probe_n),
            "latent_completion_hard_max": int(latent_concept_completion_hard_max),
            "latent_completion_refresh_steps": int(
                latent_concept_completion_refresh_steps),
            "latent_completion_study_pool_size": (
                len(study_pool) if hard_study_strategy == "completion" else 0),
            "latent_completion_hard_record_ids": (
                selected_id_sample(study_pool)
                if hard_study_strategy == "completion" else []),
            "latent_completion_study_reports": (
                list(study_reports)
                if hard_study_strategy == "completion" else []),
            "latent_study_strategy": hard_study_strategy,
            "latent_study_pool_size": len(study_pool),
            "latent_study_hard_record_ids": selected_id_sample(study_pool),
            "latent_study_reports": list(study_reports),
            "latent_memory_loss": float(latent_memory.detach()),
            "latent_consolidation_loss": float(latent_consolidation.detach()),
            "latent_consolidation_w": float(latent_concept_consolidation_w),
            "latent_consolidation_temperature": float(
                latent_concept_consolidation_temperature),
            "latent_consolidation_balance_w": float(
                latent_concept_consolidation_balance_w),
            "latent_consolidation_anchor_w": float(
                latent_concept_consolidation_anchor_w),
            "latent_consolidation_fer_w": float(latent_concept_consolidation_fer_w),
            "latent_consolidation_memory_loss": float(
                consolidation_metrics["memory_loss"].detach()),
            "latent_consolidation_anchor_loss": float(
                consolidation_metrics["anchor_loss"].detach()),
            "latent_consolidation_fer_loss": float(
                consolidation_metrics["fer_loss"].detach()),
            "latent_consolidation_nearest_cosine": float(
                consolidation_metrics["nearest_cosine"].detach()),
            "latent_consolidation_memory_active": int(
                consolidation_metrics["memory_active"]),
            "latent_consolidation_skipped": bool(consolidation_metrics["skipped"]),
            "latent_discovery_loss": float(latent_discovery.detach()),
            "latent_discovery_w": float(latent_concept_discovery_w),
            "latent_discovery_curiosity_w": float(
                latent_concept_discovery_curiosity_w),
            "latent_discovery_graph_w": float(latent_concept_discovery_graph_w),
            "latent_discovery_cycle_w": float(latent_concept_discovery_cycle_w),
            "latent_discovery_bridge_w": float(latent_concept_discovery_bridge_w),
            "latent_discovery_fer_w": float(latent_concept_discovery_fer_w),
            "latent_discovery_curiosity_loss": float(
                discovery_metrics["curiosity_loss"].detach()),
            "latent_discovery_graph_loss": float(
                discovery_metrics["graph_loss"].detach()),
            "latent_discovery_cycle_loss": float(
                discovery_metrics["cycle_loss"].detach()),
            "latent_discovery_insight_loss": float(
                discovery_metrics["insight_loss"].detach()),
            "latent_discovery_insight_score": float(
                discovery_metrics["insight_score"].detach()),
            "latent_discovery_insight_missing_mass": float(
                discovery_metrics["insight_missing_mass"].detach()),
            "latent_discovery_insight_reachable_mass": float(
                discovery_metrics["insight_reachable_mass"].detach()),
            "latent_discovery_insight_gain": float(
                discovery_metrics["insight_gain"].detach()),
            "latent_discovery_bridge_loss": float(
                discovery_metrics["bridge_loss"].detach()),
            "latent_discovery_fer_loss": float(
                discovery_metrics["fer_loss"].detach()),
            "latent_discovery_memory_active": int(
                discovery_metrics["memory_active"]),
            "latent_discovery_graph_ready": bool(
                discovery_metrics["graph_ready"]),
            "latent_discovery_skipped": bool(discovery_metrics["skipped"]),
            "latent_reanalysis_loss": float(latent_reanalysis.detach()),
            "latent_reanalysis_w": float(latent_concept_reanalysis_w),
            "latent_reanalysis_graph_w": float(
                latent_concept_reanalysis_graph_w),
            "latent_reanalysis_cycle_w": float(
                latent_concept_reanalysis_cycle_w),
            "latent_reanalysis_bridge_w": float(
                latent_concept_reanalysis_bridge_w),
            "latent_reanalysis_fer_w": float(latent_concept_reanalysis_fer_w),
            "latent_reanalysis_cycle_consistency_w": float(
                latent_concept_reanalysis_cycle_consistency_w),
            "latent_reanalysis_closure_loss": float(
                reanalysis_metrics["closure_loss"].detach()),
            "latent_reanalysis_cycle_loss": float(
                reanalysis_metrics["cycle_loss"].detach()),
            "latent_reanalysis_bridge_loss": float(
                reanalysis_metrics["bridge_loss"].detach()),
            "latent_reanalysis_fer_loss": float(
                reanalysis_metrics["fer_loss"].detach()),
            "latent_reanalysis_memory_active": int(
                reanalysis_metrics["memory_active"]),
            "latent_reanalysis_graph_ready": bool(
                reanalysis_metrics["graph_ready"]),
            "latent_reanalysis_skipped": bool(reanalysis_metrics["skipped"]),
            "latent_gap_loss": float(latent_gap.detach()),
            "latent_gap_w": float(latent_concept_gap_w),
            "latent_gap_temperature": float(latent_concept_gap_temperature),
            "latent_gap_self_loop_w": float(latent_concept_gap_self_loop_w),
            "latent_gap_transitive_steps": int(latent_concept_gap_transitive_steps),
            "latent_gap_transitive_w": float(latent_concept_gap_transitive_w),
            "latent_gap_target_power": float(latent_concept_gap_target_power),
            "latent_gap_kl": float(gap_metrics["gap_kl"].detach()),
            "latent_gap_cosine": float(gap_metrics["gap_cosine"].detach()),
            "latent_gap_entropy": float(gap_metrics["gap_entropy"].detach()),
            "latent_gap_target_mass": float(
                gap_metrics["gap_target_mass"].detach()),
            "latent_gap_present_overlap": float(
                gap_metrics["gap_present_overlap"].detach()),
            "latent_gap_memory_active": int(gap_metrics["memory_active"]),
            "latent_gap_graph_ready": bool(gap_metrics["graph_ready"]),
            "latent_gap_skipped": bool(gap_metrics["skipped"]),
            "latent_association_loss": float(latent_association.detach()),
            "latent_composition_loss": float(latent_composition.detach()),
            "latent_graph_predict_loss": float(latent_graph_predict.detach()),
            "latent_bridge_loss": float(latent_bridge.detach()),
            "latent_bridge_w": float(latent_concept_bridge_w),
            "latent_completion_loss": float(latent_completion.detach()),
            "latent_completion_w": float(latent_concept_completion_w),
            "latent_completion_temperature": float(
                latent_concept_completion_temperature),
            "latent_completion_view_count": int(completion_metrics["view_count"]),
            "latent_completion_skipped": bool(completion_metrics["skipped"]),
            "latent_sequence_loss": float(latent_sequence.detach()),
            "latent_sequence_w": float(latent_concept_sequence_w),
            "latent_sequence_batch": int(sequence_batch),
            "latent_sequence_temperature": float(
                latent_concept_sequence_temperature),
            "latent_sequence_pairs": len(sequence_pairs),
            "latent_sequence_report": sequence_report,
            "latent_sequence_transition_last_batch_updates": int(
                sequence_transition_updates),
            "latent_bridge_score": float(bridge_metrics["bridge_score"]),
            "latent_bridge_entropy": float(bridge_metrics["bridge_entropy"]),
            "latent_bridge_connectivity": float(
                bridge_metrics["bridge_connectivity"]),
            "latent_bridge_skipped": bool(bridge_metrics.get("skipped", False)),
            "latent_bridge_graph_ready": bool(
                bridge_metrics.get("graph_ready", False)),
            "latent_neighborhood_loss": float(latent_neighborhood.detach()),
            "latent_transition_loss": float(latent_transition.detach()),
            "latent_cluster_loss": float(latent_cluster.detach()),
            "latent_memory_last_batch_updates": memory_updates,
            "latent_graph_transition_last_batch_updates": transition_updates,
            "latent_memory_active": int(
                getattr(getattr(model, "latent_concept_memory", None),
                        "filled", torch.zeros((), dtype=torch.long)).item()),
            "latent_association_relation_updates": int(
                getattr(getattr(model, "latent_concept_memory", None),
                        "relation_updates", torch.zeros((), dtype=torch.long)).item()),
            "latent_association_active_edges": int(
                getattr(getattr(model, "latent_concept_memory", None),
                        "relations", torch.zeros(0)).gt(0).sum().item()),
            "latent_graph_transition_updates": int(
                getattr(getattr(model, "latent_concept_memory", None),
                        "transition_updates", torch.zeros((), dtype=torch.long)).item()),
            "latent_graph_transition_active_edges": int(
                getattr(getattr(model, "latent_concept_memory", None),
                        "transitions", torch.zeros(0)).gt(0).sum().item()),
        }
        if st % int(log_every) == 0 or st == int(steps):
            print(f"  multimodal {st}/{steps} loss {last['loss']:.3f} "
                  f"token {last['token_loss']:.3f} agree {last['agreement_loss']:.3f} "
                  f"repeat {last['repetition_unlikelihood_loss']:.3f} "
                  f"latent {last['latent_concept_loss']:.3f} "
                  f"fer {last['latent_fer_loss']:.3f} "
                  f"memory {last['latent_memory_loss']:.3f} "
                  f"consolidate {last['latent_consolidation_loss']:.3f} "
                  f"discover {last['latent_discovery_loss']:.3f} "
                  f"reanalyze {last['latent_reanalysis_loss']:.3f} "
                  f"gap {last['latent_gap_loss']:.3f} "
                  f"complete {last['latent_completion_loss']:.3f} "
                  f"sequence {last['latent_sequence_loss']:.3f}",
                  flush=True)
        if select_best and st in selection_boundaries:
            round_id, round_steps = selection_boundaries[st]
            bundle = multimodal_eval_bundle(
                model, eval_records, vocab, view_dims, n=selection_eval_n,
                seed=seed, device=device, score_metric=selection_score_metric,
                score_margin_w=selection_score_margin_w,
                decode_objective=decode_objective,
                **selection_generation_kwargs)
            row = selection_row(round_id, round_steps, bundle)
            round_weight_update = multimodal_weight_update_report(
                weight_update_before, multimodal_weight_update_snapshot(model))
            score_delta_from_best = float(row["score"] - best_score)
            bridge_insight_gate = bool(
                latent_concept_bridge_w and latent_concept_memory_size > 0)
            insight = multimodal_bridge_selection_insight(
                best_bridge_eval, row["latent_bridge"],
                enabled=bridge_insight_gate)
            signal_regression = signal_regression_report(
                (best_bundle or before_bundle)["score_components"],
                row["score_components"],
                MULTIMODAL_SELF_TEACH_SCORE_KEYS,
                skip_keys=MULTIMODAL_SELF_TEACH_SKIP_KEYS,
                tolerance=selection_signal_regression_tolerance,
                signals=MULTIMODAL_SELF_TEACH_SIGNALS)
            decision = concept_round_selection_decision(
                score_delta_from_best, selection_score_min_delta,
                insight_delta=insight["bridge_insight_delta"],
                insight_allowed=insight["bridge_insight_allowed"],
                insight_gate=bridge_insight_gate,
                insight_accept_w=selection_insight_accept_w,
                insight_min_delta=selection_insight_min_delta)
            pre_signal_selected = bool(decision["selected"])
            selected = bool(
                pre_signal_selected and signal_regression["allowed"])
            blocked_by_signal_regression = bool(
                pre_signal_selected and not signal_regression["allowed"])
            row = row | {
                "selected": bool(selected),
                "score_delta_from_best": score_delta_from_best,
                "signal_regression_gate": True,
                "signal_regression_allowed": bool(
                    signal_regression["allowed"]),
                "signal_regression": signal_regression,
                "signal_regression_tolerance": float(
                    selection_signal_regression_tolerance),
                "signal_regression_max": float(
                    signal_regression["max_regression"]),
                "signal_regression_signals": list(
                    signal_regression["regressed_signals"]),
                "bridge_insight_gate": bool(bridge_insight_gate),
                "bridge_insight_delta": float(insight["bridge_insight_delta"]),
                "bridge_insight_allowed": bool(
                    insight["bridge_insight_allowed"]),
                "bridge_insight": insight,
                "selected_by_score": bool(
                    selected and decision["selected_by_score"]),
                "selected_by_insight": bool(
                    selected and decision["selected_by_insight"]),
                "pre_signal_selected": bool(pre_signal_selected),
                "pre_signal_selected_by_score": bool(
                    decision["selected_by_score"]),
                "pre_signal_selected_by_insight": bool(
                    decision["selected_by_insight"]),
                "blocked_by_signal_regression": bool(
                    blocked_by_signal_regression),
                "blocked_reasons": (
                    ["signal_regression"] if blocked_by_signal_regression
                    else []),
                "insight_score_boost": float(decision["insight_score_boost"]),
                "insight_effective_delta": float(
                    decision["insight_effective_delta"]),
                "self_teach_plan": self_teach_plan,
                "self_teach_effective_weights": dict(
                    self_teach_effective_weights),
                "weight_update": round_weight_update,
                "weight_update_changed": bool(round_weight_update["changed"]),
                "weight_update_changed_tensor_count": int(
                    round_weight_update["changed_tensor_count"]),
                "weight_update_changed_value_count": int(
                    round_weight_update["changed_value_count"]),
                "weight_update_max_abs_delta": float(
                    round_weight_update["max_abs_delta"]),
                "train_metrics": _compact_multimodal_train_metrics(last),
            }
            rounds_report.append(row)
            if selected:
                best_score = float(row["score"])
                best_round = int(round_id)
                best_state = _model_state_copy(model)
                best_bundle = bundle
                best_bridge_eval = row["latent_bridge"]
                best_metrics = dict(last)
                best_study_reports = list(study_reports)
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                if (selection_score_patience
                        and no_improve_rounds >= int(selection_score_patience)):
                    stopped_early = True
                    stop_round = int(round_id)
                    break
            if self_teach_w > 0.0 and st < int(steps):
                update_self_teach_weights(
                    row["score_components"], round_id=round_id + 1)
    if select_best and best_state is not None:
        model.load_state_dict(best_state, strict=False)
        for row in rounds_report:
            row["selected"] = row["round"] == best_round
        selection = {
            "enabled": True,
            "rounds_requested": int(selection_rounds),
            "rounds_planned": len(selection_boundaries),
            "rounds_run": len(rounds_report) - 1,
            "score_metric": selection_score_metric,
            "score_margin_w": float(selection_score_margin_w),
            "score_min_delta": float(selection_score_min_delta),
            "score_patience": int(selection_score_patience),
            "signal_regression_gate": True,
            "signal_regression_tolerance": float(
                selection_signal_regression_tolerance),
            "selection_eval_n": int(selection_eval_n),
            "selection_generation_n": int(selection_generation_n),
            "selection_generation_prompt_tokens": int(
                selection_generation_prompt_tokens),
            "selection_generation_max_new_tokens": int(
                selection_generation_max_new_tokens),
            "selection_generation_temperature": float(
                selection_generation_temperature),
            "selection_generation_top_k": int(selection_generation_top_k),
            "self_teach_w": float(self_teach_w),
            "self_teach_available_objectives": list(
                self_teach_available_objectives),
            "self_teach_reports": list(self_teach_reports),
            "bridge_insight_gate": bool(
                latent_concept_bridge_w and latent_concept_memory_size > 0),
            "insight_accept_w": float(selection_insight_accept_w),
            "insight_min_delta": float(selection_insight_min_delta),
            "stopped_early": bool(stopped_early),
            "stop_round": int(stop_round),
            "no_improve_rounds": int(no_improve_rounds),
            "selected_round": int(best_round),
            "accepted_update": bool(best_round > 0),
            "selected_score": float(best_score),
            "before_score": float(rounds_report[0]["score"]),
            "selected_score_delta": float(best_score - rounds_report[0]["score"]),
            "attempted_weight_update_count": int(
                sum(1 for row in rounds_report
                    if int(row.get("round", 0)) > 0
                    and bool(row.get("weight_update_changed", False)))),
            "signal_regression_blocked_round_count": int(
                sum(1 for row in rounds_report
                    if bool(row.get("blocked_by_signal_regression", False)))),
            "rounds": rounds_report,
        }
        selected_rows = [row for row in rounds_report if row["round"] == best_round]
        selected_self_teach_plan = self_teach_plan
        selected_self_teach_effective_weights = dict(self_teach_effective_weights)
        if selected_rows:
            selected_row = selected_rows[0]
            selected_self_teach_plan = selected_row.get("self_teach_plan")
            selected_self_teach_effective_weights = selected_row.get(
                "self_teach_effective_weights", selected_self_teach_effective_weights)
            selection["selected_by_score"] = bool(
                selected_row.get("selected_by_score", False))
            selection["selected_by_insight"] = bool(
                selected_row.get("selected_by_insight", False))
            selection["selected_insight_score_boost"] = float(
                selected_row.get("insight_score_boost", 0.0))
            selection["selected_insight_effective_delta"] = float(
                selected_row.get("insight_effective_delta", 0.0))
            selection["selected_bridge_insight"] = selected_row.get(
                "bridge_insight")
            selection["selected_signal_regression"] = selected_row.get(
                "signal_regression")
            selection["selected_signal_regression_allowed"] = bool(
                selected_row.get("signal_regression_allowed", True))
            selection["selected_self_teach_plan"] = selected_self_teach_plan
            selection["selected_self_teach_effective_weights"] = (
                selected_self_teach_effective_weights)
        active_study_reports = list(best_study_reports)
        best_metrics = best_metrics | {
            "selection": selection,
            "self_teach_w": float(self_teach_w),
            "self_teach_history_prior_w": float(self_teach_history_prior_w),
            "text_checkpoint_history_prior": text_checkpoint_history_prior,
            "multimodal_checkpoint_history_prior": (
                multimodal_checkpoint_history_prior),
            "self_teach_history_prior": self_teach_history_prior,
            "self_teach_plan": selected_self_teach_plan,
            "self_teach_base_weights": dict(self_teach_base_weights),
            "self_teach_effective_weights": dict(
                selected_self_teach_effective_weights),
            "self_teach_reports": list(self_teach_reports),
            "latent_fer_study_reports": (
                active_study_reports if hard_study_strategy == "fer" else []),
            "latent_discovery_study_reports": (
                active_study_reports
                if hard_study_strategy == "discovery" else []),
            "latent_completion_study_reports": (
                active_study_reports
                if hard_study_strategy == "completion" else []),
            "latent_study_reports": active_study_reports,
        }
        if best_round == 0:
            best_metrics = best_metrics | {
                "latent_study_strategy": hard_study_strategy,
                "latent_study_pool_size": 0,
                "latent_study_hard_record_ids": [],
                "latent_fer_study_pool_size": 0,
                "latent_fer_hard_record_ids": [],
                "latent_discovery_study_pool_size": 0,
                "latent_discovery_hard_record_ids": [],
                "latent_completion_study_pool_size": 0,
                "latent_completion_hard_record_ids": [],
            }
        last = best_metrics
    else:
        selection = {
            "enabled": False,
            "self_teach_w": float(self_teach_w),
            "selection_generation_n": int(selection_generation_n),
            "selection_generation_prompt_tokens": int(
                selection_generation_prompt_tokens),
            "selection_generation_max_new_tokens": int(
                selection_generation_max_new_tokens),
            "selection_generation_temperature": float(
                selection_generation_temperature),
            "selection_generation_top_k": int(selection_generation_top_k),
            "text_checkpoint_history_prior": text_checkpoint_history_prior,
            "multimodal_checkpoint_history_prior": (
                multimodal_checkpoint_history_prior),
            "self_teach_history_prior": self_teach_history_prior,
            "self_teach_reports": list(self_teach_reports),
            "self_teach_plan": self_teach_plan,
        }
        last = dict(last) | {"selection": selection}
        if self_teach_plan is not None:
            last = last | {
                "self_teach_w": float(self_teach_w),
                "self_teach_history_prior_w": float(self_teach_history_prior_w),
                "text_checkpoint_history_prior": text_checkpoint_history_prior,
                "multimodal_checkpoint_history_prior": (
                    multimodal_checkpoint_history_prior),
                "self_teach_history_prior": self_teach_history_prior,
                "self_teach_plan": self_teach_plan,
                "self_teach_base_weights": dict(self_teach_base_weights),
                "self_teach_effective_weights": dict(self_teach_effective_weights),
                "self_teach_reports": list(self_teach_reports),
            }
    last = dict(last) | {
        **source_balance_train_report,
        "decode_objective": str(decode_objective),
        "selection_generation_n": int(selection_generation_n),
        "selection_generation_prompt_tokens": int(
            selection_generation_prompt_tokens),
        "selection_generation_max_new_tokens": int(
            selection_generation_max_new_tokens),
        "selection_generation_temperature": float(selection_generation_temperature),
        "selection_generation_top_k": int(selection_generation_top_k),
        "selection_signal_regression_tolerance": float(
            selection_signal_regression_tolerance),
        "decode_objective_report": decode_objective_info,
        "active_modes": list(active_modes),
        "self_teach_history_prior_w": float(self_teach_history_prior_w),
        "text_checkpoint_history_prior": text_checkpoint_history_prior,
        "text_checkpoint_history_prior_raw": text_checkpoint_history_prior_raw,
        "multimodal_checkpoint_history_prior": (
            multimodal_checkpoint_history_prior),
        "self_teach_history_prior": self_teach_history_prior,
        "text_checkpoint_transfer": getattr(model, "text_checkpoint_transfer", {}),
        "multimodal_checkpoint_transfer": getattr(
            model, "multimodal_checkpoint_transfer", {}),
    }
    weight_update = multimodal_weight_update_report(
        weight_update_before, multimodal_weight_update_snapshot(model))
    representation_progress = {"enabled": False}
    if representation_before_bundle is not None:
        representation_after_bundle = multimodal_eval_bundle(
            model, eval_records, vocab, view_dims,
            n=representation_progress_probe_n, seed=representation_progress_seed,
            device=device, score_metric=selection_score_metric,
            score_margin_w=selection_score_margin_w,
            decode_objective=decode_objective,
            **selection_generation_kwargs)
        representation_progress = multimodal_representation_progress_report(
            representation_before_bundle, representation_after_bundle)
    attempted_weight_update_count = 0
    if select_best:
        attempted_weight_update_count = sum(
            1 for row in rounds_report
            if int(row.get("round", 0)) > 0
            and bool(row.get("weight_update_changed", False)))
    elif int(steps) > 0 and bool(weight_update["changed"]):
        attempted_weight_update_count = 1
    last = dict(last) | {
        "weight_update": weight_update,
        "weight_update_changed": bool(weight_update["changed"]),
        "weight_update_changed_tensor_count": int(
            weight_update["changed_tensor_count"]),
        "weight_update_changed_value_count": int(
            weight_update["changed_value_count"]),
        "weight_update_max_abs_delta": float(weight_update["max_abs_delta"]),
        "attempted_weight_update_count": int(attempted_weight_update_count),
        "representation_probe_n": int(representation_progress_probe_n),
        "representation_progress": representation_progress,
    }
    last["learning_event"] = multimodal_learning_event_report(last)
    model.train_metrics = last
    model.latent_fer_study_reports = (
        last.get("latent_fer_study_reports", []))
    model.latent_discovery_study_reports = (
        last.get("latent_discovery_study_reports", []))
    model.latent_completion_study_reports = (
        last.get("latent_completion_study_reports", []))
    model.latent_study_reports = last.get("latent_study_reports", [])
    model.manifest_info = {
        "path": manifest,
        "root": root,
        "records": len(records),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "view_dims": dict(view_dims),
        "view_names": list(view_dims.keys()),
        "view_count": len(view_dims),
        "active_modes": list(active_modes),
        "decode_objective": str(decode_objective),
        "decode_objective_report": decode_objective_info,
        "source_balance_w": float(source_balance_w),
        "source_count": int(source_balance_train_report["training_source_count"]),
        "source_counts": dict(
            source_balance_train_report["training_source_record_counts"]),
        "max_source_record_frac": float(
            source_balance_train_report["training_max_source_record_frac"]),
        "min_source_record_frac": float(
            source_balance_train_report["training_min_source_record_frac"]),
    }
    return model, vocab, records, train_records, eval_records, view_dims


def run(manifest, root=None, steps=400, seed=0, device=DEV, eval_n=200,
        checkpoint=None, out=None, generation_n=0,
        generation_prompt_tokens=16, generation_max_new_tokens=32,
        generation_temperature=0.0, generation_top_k=0,
        generate_prompts=None, **kwargs):
    model, vocab, records, train_records, eval_records, view_dims = train(
        manifest, root=root, steps=steps, seed=seed, device=device, **kwargs)
    decode_objective = str(
        getattr(model, "manifest_info", {}).get("decode_objective", "target"))
    active_modes = tuple(
        getattr(model, "manifest_info", {}).get("active_modes")
        or multimodal_active_modes(view_dims, records))
    metrics = {
        mode: evaluate(model, eval_records, vocab, view_dims, n=eval_n,
                       seed=seed + 17, device=device, mode=mode,
                       decode_objective=decode_objective)
        for mode in active_modes
    }
    generation_mode = "full" if "full" in active_modes else active_modes[0]
    generation = (
        multimodal_generation_eval(
            model, eval_records, vocab, view_dims, n=generation_n,
            seed=seed + 41, device=device, mode=generation_mode,
            decode_objective=decode_objective,
            prompt_tokens=generation_prompt_tokens,
            max_new_tokens=generation_max_new_tokens,
            temperature=generation_temperature, top_k=generation_top_k)
        if int(generation_n) > 0 else {
            "enabled": False, "skipped": True,
            "skip_reason": "generation_n_is_zero",
        })
    prompt_generations = (
        multimodal_prompt_generation_report(
            model, generate_prompts, vocab, view_dims, mode=generation_mode,
            max_new_tokens=generation_max_new_tokens,
            temperature=generation_temperature, top_k=generation_top_k,
            device=device)
        if generate_prompts else {
            "enabled": False, "skipped": True,
            "skip_reason": "no_generate_prompts",
        })
    latent_probe = {}
    if getattr(model, "latent_concept_memory", None) is not None:
        selected, latent_probe = latent_multimodal_graph_prediction_examples(
            model, train_records, vocab, view_dims,
            n=min(64, len(train_records)), seed=seed + 23, device=device,
            decode_objective=decode_objective)
        latent_probe["top_ids"] = [r.rec_id for r in selected[:8]]
    latent_fer_probe = latent_multimodal_fer_eval(
        model, eval_records, vocab, view_dims, n=eval_n, seed=seed + 29,
        device=device, decode_objective=decode_objective)
    latent_fer_hard_selected, latent_fer_hard_probe = latent_multimodal_fer_examples(
        model, train_records, vocab, view_dims,
        n=min(64, len(train_records)), seed=seed + 30, device=device,
        decode_objective=decode_objective)
    latent_fer_hard_probe["top_ids"] = [
        r.rec_id for r in latent_fer_hard_selected[:8]]
    latent_completion_selected, latent_completion_probe = (
        latent_multimodal_completion_examples(
            model, train_records, vocab, view_dims,
            n=min(64, len(train_records)), seed=seed + 33, device=device,
            decode_objective=decode_objective))
    latent_completion_probe["top_ids"] = [
        r.rec_id for r in latent_completion_selected[:8]]
    latent_discovery_selected, latent_discovery_probe = (
        latent_multimodal_discovery_examples(
            model, train_records, vocab, view_dims,
            n=min(64, len(train_records)), seed=seed + 32, device=device,
            decode_objective=decode_objective))
    latent_discovery_probe["top_ids"] = [
        r.rec_id for r in latent_discovery_selected[:8]]
    latent_bridge_probe = latent_multimodal_bridge_eval(
        model, eval_records, vocab, view_dims, n=eval_n, seed=seed + 31,
        device=device, decode_objective=decode_objective)
    latent_sequence_probe = latent_multimodal_sequence_eval(
        model, eval_records, vocab, view_dims, n=eval_n, seed=seed + 37,
        device=device, decode_objective=decode_objective)
    architecture = dict(model.config)
    architecture["reader_prefix_tokens"] = (
        int(model.config["view_tokens"]) * len(model.config["view_names"])
        + int(model.config["txt_tokens"]))
    architecture["latent_concept_prefix_tokens"] = (
        int(model.latent_concept_slots) if bool(model.latent_concept_prefix) else 0)
    architecture["prefix_tokens"] = (
        architecture["reader_prefix_tokens"] + int(model.config["concept_tokens"])
        + architecture["latent_concept_prefix_tokens"])
    report = {
        "experiment": "generic_multimodal_bridge",
        "steps": int(steps),
        "seed": int(seed),
        "manifest": model.manifest_info,
        "architecture": architecture,
        "train_metrics": getattr(model, "train_metrics", {}),
        "learning_event": getattr(model, "train_metrics", {}).get(
            "learning_event", {}),
        "representation_progress": getattr(model, "train_metrics", {}).get(
            "representation_progress", {"enabled": False}),
        "selection": getattr(model, "train_metrics", {}).get(
            "selection", {"enabled": False}),
        "teacher_forced": metrics,
        "generation": generation,
        "prompt_generations": prompt_generations,
        "latent_graph_probe": latent_probe,
        "latent_fer_probe": latent_fer_probe,
        "latent_fer_hard_probe": latent_fer_hard_probe,
        "latent_completion_hard_probe": latent_completion_probe,
        "latent_discovery_probe": latent_discovery_probe,
        "latent_bridge_probe": latent_bridge_probe,
        "latent_sequence_probe": latent_sequence_probe,
        "text_checkpoint_transfer": getattr(model, "text_checkpoint_transfer", {}),
        "multimodal_checkpoint_transfer": getattr(
            model, "multimodal_checkpoint_transfer", {}),
        "gate": metrics["full"]["token_acc"] >= 0.50,
    }
    previous_learning_history = {}
    previous_multimodal_checkpoint = kwargs.get("multimodal_checkpoint")
    if previous_multimodal_checkpoint:
        previous_learning_history = multimodal_learning_history_from_payload(
            load_multimodal_checkpoint_payload(
                previous_multimodal_checkpoint, device="cpu"))
    report["multimodal_learning_history"] = (
        multimodal_learning_history_with_entry(previous_learning_history, report))
    report["multimodal_learning_history_count"] = int(
        report["multimodal_learning_history"]["entry_count"])
    report["multimodal_learning_history_summary"] = (
        multimodal_checkpoint_learning_history_summary({
            "multimodal_learning_history": report["multimodal_learning_history"]}))
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos,
                    "d": model.lm.tok.embedding_dim, "model_config": model.config,
                    "manifest": model.manifest_info, "report": report,
                    "multimodal_learning_history": report[
                        "multimodal_learning_history"]}, checkpoint)
        report["checkpoint"] = checkpoint
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    return report


def _write_selftest_manifest(path):
    rng = np.random.default_rng(0)
    rows = []
    labels = ("alpha", "beta", "gamma", "delta")
    for i in range(12):
        label = labels[i % len(labels)]
        base = np.zeros(4, dtype=np.float32)
        base[i % 4] = 1.0
        sensor_a = (base + 0.01 * rng.normal(size=4)).astype(float).tolist()
        sensor_b = (base[::-1] + 0.01 * rng.normal(size=4)).astype(float).tolist()
        rows.append({
            "id": f"mm-{i}",
            "split": "eval" if i >= 8 else "train",
            "text": ["sample", str(i), "label", label],
            "views": {"sensor_a": sensor_a, "sensor_b": sensor_b},
            "target": ["extract", "sample", str(i), "label", label, "done", "."],
            "meta": {
                "source": "selftest-eval" if i >= 8 else "selftest-train",
                "chunk_index": i - 8 if i >= 8 else i,
            },
        })
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def selftest():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "mm.jsonl")
        _write_selftest_manifest(manifest)
        records = load_manifest(manifest)
        train_records, eval_records = split_records(records)
        vocab = build_vocab(records)
        view_dims = feature_dims(records)
        implicit_manifest = os.path.join(tmpdir, "mm_implicit_source.jsonl")
        with open(implicit_manifest, "w") as f:
            f.write(json.dumps({
                "id": "implicit-source",
                "split": "train",
                "text": ["implicit", "source"],
                "views": {"sensor": [1.0, 0.0]},
                "target": ["ok"],
            }) + "\n")
            f.write(json.dumps({
                "id": "top-source",
                "split": "train",
                "text": ["top", "source"],
                "views": {"sensor": [0.0, 1.0]},
                "target": ["ok"],
                "dataset": "top-level-dataset",
            }) + "\n")
        implicit_records = load_manifest(implicit_manifest)
        assert implicit_records[0].meta["source"] == os.path.abspath(
            implicit_manifest)
        assert implicit_records[1].meta["dataset"] == "top-level-dataset"
        continuation_manifest = os.path.join(tmpdir, "mm_continuation.jsonl")
        continuation_rows = [
            {"id": "c0", "split": "train", "text": ["a", "b"],
             "target": ["c", "d"], "meta": {"source": "doc", "chunk_index": 0}},
            {"id": "c1", "split": "train", "text": ["c", "d"],
             "target": ["e", "f"], "meta": {"source": "doc", "chunk_index": 1}},
            {"id": "c2", "split": "eval", "text": ["e", "f"],
             "target": ["g", "h"], "meta": {"source": "doc", "chunk_index": 2}},
        ]
        with open(continuation_manifest, "w") as f:
            for row in continuation_rows:
                f.write(json.dumps(row) + "\n")
        continuation_records = load_manifest(continuation_manifest)
        assert infer_multimodal_decode_objective(
            continuation_records, requested="auto") == "causal"
        continuation_vocab = build_vocab(continuation_records)
        _features, latent_txt, decoder_ids, decoder_txt = _batch_from_records(
            continuation_records[:1], continuation_vocab, "cpu", {},
            decode_objective="causal", return_decode_text=True)
        assert latent_txt.shape[1] == 4
        assert decoder_ids.shape[1] == 4
        assert decoder_txt.shape[1] == 1
        imbalanced_records = [
            MultimodalRecord(
                f"bulk-{i}", "train", ("bulk", str(i)), ("target",),
                {}, {"source": "bulk"})
            for i in range(4)
        ] + [
            MultimodalRecord(
                "rare-0", "train", ("rare",), ("target",), {},
                {"source": "rare"})
        ]
        assert multimodal_training_sampling_weights(
            imbalanced_records, source_balance_w=0.0) is None
        imbalanced_weights = multimodal_training_sampling_weights(
            imbalanced_records, source_balance_w=1.0)
        assert imbalanced_weights is not None
        assert math.isclose(float(imbalanced_weights.sum()), 1.0, rel_tol=1e-6)
        bulk_mass = float(sum(
            weight for rec, weight in zip(imbalanced_records, imbalanced_weights)
            if multimodal_record_source(rec) == "bulk"))
        rare_mass = float(sum(
            weight for rec, weight in zip(imbalanced_records, imbalanced_weights)
            if multimodal_record_source(rec) == "rare"))
        assert math.isclose(bulk_mass, rare_mass, rel_tol=1e-6, abs_tol=1e-6)
        balance_report = multimodal_source_balance_report(
            imbalanced_records, 1.0, weights=imbalanced_weights)
        assert balance_report["training_source_balance_sampling"] is True
        assert balance_report["training_source_count"] == 2
        assert balance_report["training_max_source_record_frac"] == 0.8
        model = MultimodalLM(
            len(vocab), view_dims=view_dims, d=32, layers=1,
            heads=4, pad=vocab.pad, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8).to("cpu")
        from .text import (
            TextReadingLM,
            checkpoint_payload as text_checkpoint_payload,
            reading_mastery_history_with_entry,
        )
        text_ckpt = os.path.join(tmpdir, "text_reading.pt")
        text_model = TextReadingLM(
            len(vocab), d=32, layers=1, heads=4, pad=vocab.pad,
            latent_concept_slots=3, latent_concept_topk=2,
            latent_concept_memory_size=8).to("cpu")
        with torch.no_grad():
            sample_idx = vocab.stoi["sample"]
            text_model.txt.emb.weight[sample_idx].fill_(0.125)
            memory_values = torch.linspace(
                0.0, 1.0,
                steps=text_model.latent_concept_memory.memory.numel())
            text_model.latent_concept_memory.memory.copy_(
                memory_values.view_as(text_model.latent_concept_memory.memory))
            text_model.latent_concept_memory.filled.fill_(1)
            for param in text_model.reading_predictor.parameters():
                param.fill_(0.03125)
            for param in text_model.reading_completion_predictor.parameters():
                param.fill_(0.0625)
        text_report = {
            "experiment": "text-raw-reading-selftest",
            "latent_concept_topk": 2,
            "reading_replay_bank": {
                "record_count": 3,
                "priority_record_count": 2,
            },
            "replay_study_used": True,
            "replay_study_records": 3,
            "study_record_count": 5,
            "study_train_records": 4,
            "before_score_components": {
                "mastery_score": 0.2,
                "signal_coverage": 0.3,
            },
            "after_score_components": {
                "mastery_score": 0.4,
                "signal_coverage": 0.5,
                "balanced_score": 0.45,
                "floor_score": 0.35,
                "language_score": 0.4,
                "lm_token_acc": 0.4,
                "fer_score": 0.9,
                "bridge_score": 0.8,
                "bridge_connectivity": 0.75,
                "sequence_score": 0.0,
                "language_skipped": False,
                "fer_skipped": False,
                "bridge_skipped": False,
                "sequence_skipped": False,
            },
            "selection": {
                "enabled": True,
                "accepted_update": True,
                "selected_round": 1,
                "selected_by_insight": True,
                "selected_score_delta": 0.02,
                "attempted_weight_update_count": 1,
                "rounds": [{
                    "round": 1,
                    "selected": True,
                    "concept_insight_delta": 0.8,
                    "selected_by_insight": True,
                    "weight_update_changed": True,
                    "weight_update_changed_tensor_count": 5,
                    "weight_update_changed_value_count": 7,
                    "weight_update_max_abs_delta": 0.02,
                    "training_priority_sampling": True,
                    "training_priority_record_count": 2,
                    "training_priority_mean": 0.5,
                    "training_priority_max": 1.5,
                }],
                "self_teach_reports": [{
                    "enabled": True,
                    "top_signal": "sequence",
                    "active_signals": ["sequence"],
                    "history_prior_enabled": False,
                    "history_prior_entry_count": 0,
                    "history_prior_top_signal": "",
                }],
            },
            "representation_progress": {
                "enabled": True,
                "active_signals": ["language", "fer", "bridge", "sequence"],
                "signal_after": {
                    "language": 0.4, "fer": 0.9,
                    "bridge": 0.8, "sequence": 0.0},
                "signal_deltas": {
                    "language": -0.1, "fer": 0.1,
                    "bridge": 0.2, "sequence": -0.3},
                "organization_score_before": 0.25,
                "organization_score_after": 0.55,
                "organization_score_delta": 0.3,
                "positive_signal_gain": 0.3,
                "negative_signal_drift": 0.3,
                "top_gain_signal": "bridge",
                "top_regression_signal": "sequence",
                "representation_insight_event": True,
            },
        }
        text_report["reading_mastery_history"] = (
            reading_mastery_history_with_entry([], text_report))
        text_report["reading_mastery_history_count"] = (
            text_report["reading_mastery_history"]["entry_count"])
        torch.save(text_checkpoint_payload(
            text_model, vocab, 32, 1, 4, text_report), text_ckpt)
        text_ckpt_payload = load_text_checkpoint_payload(text_ckpt, device="cpu")
        text_latent_config = text_checkpoint_latent_config(text_ckpt, device="cpu")
        assert text_latent_config["latent_concept_topk"] == 2
        text_history = text_checkpoint_reading_history(text_ckpt_payload)
        assert text_history["entry_count"] == 1
        assert text_history["entries"][0]["learning_event_triggered"] is True
        assert text_history["entries"][0]["learning_event_top_signal"] == "connection"
        assert text_history["entries"][0]["replay_study_used"] is True
        assert text_history["entries"][0]["training_priority_sampling"] is True
        text_history_prior = multimodal_text_history_self_teach_prior(
            text_ckpt_payload, enabled=True)
        assert text_history_prior["enabled"] is True
        assert text_history_prior["entry_count"] == 1
        assert text_history_prior["top_signal"] == "connection"
        assert text_history_prior["signal_deficits"]["token"] > 0.0
        assert text_history_prior["signal_deficits"]["sequence"] > 0.0
        assert text_history_prior["concept_connection_signal"] > 0.0
        assert text_history_prior["signal_deficits"]["connection"] > 0.0
        assert text_history_prior["signal_deficits"]["bridge"] > 0.0
        assert text_history_prior[
            "reading_representation_summary"]["enabled"] is True
        assert text_history_prior[
            "reading_representation_summary"]["top_gain_signal"] == "bridge"
        assert text_history_prior[
            "reading_priority_study_summary"]["enabled"] is True
        priority_study = text_checkpoint_priority_study_summary(text_ckpt_payload)
        assert priority_study["enabled"] is True
        assert priority_study["replay_study_entry_count"] == 1
        assert priority_study["training_priority_entry_count"] == 1
        assert priority_study["max_training_priority_record_count"] == 2
        event_only_prior = multimodal_text_history_self_teach_prior({
            "reading_mastery_history": {
                "entries": [{
                    "learning_event": {
                        "triggered": True,
                        "kind": "representation_reorganization",
                        "top_signal": "cluster",
                        "event_score": 0.6,
                    },
                }],
            },
        }, enabled=True)
        assert event_only_prior["signal_deficits"]["sequence"] > 0.0
        language_event_prior = multimodal_text_history_self_teach_prior({
            "reading_mastery_history": {
                "entries": [{
                    "learning_event": {
                        "triggered": True,
                        "kind": "representation_reorganization",
                        "top_signal": "language",
                        "event_score": 0.6,
                    },
                }],
            },
        }, enabled=True)
        assert language_event_prior["top_signal"] == "token"
        assert language_event_prior["signal_deficits"]["token"] > 0.0
        concept_prior = text_checkpoint_concept_insight_prior(
            text_ckpt_payload, enabled=True)
        assert concept_prior["concept_connection_signal"] > 0.0
        assert concept_prior["selected_by_insight_count"] == 1
        assert concept_prior["accepted_update_count"] == 1
        learning_event = text_checkpoint_learning_event_summary(text_ckpt_payload)
        assert learning_event["triggered_count"] == 1
        assert learning_event["latest_triggered"] is True
        assert learning_event["top_signal"] == "connection"
        assert learning_event["top_kind"] == "concept_connection"
        assert learning_event["max_event_score"] > 0.0
        weight_update = text_checkpoint_weight_update_summary(text_ckpt_payload)
        assert weight_update["changed"] is True
        assert weight_update["changed_tensor_count"] == 5
        assert weight_update["changed_value_count"] == 7
        assert weight_update["max_abs_delta"] == 0.02
        assert weight_update["attempted_weight_update_count"] == 1
        transfer_model = MultimodalLM(
            len(vocab), view_dims=view_dims, d=32, layers=1,
            heads=4, pad=vocab.pad, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8).to("cpu")
        transfer_report = import_text_checkpoint(
            transfer_model, vocab, text_ckpt, device="cpu")
        assert transfer_report["copied"] is True
        assert transfer_report["checkpoint_experiment"] == "text-raw-reading-selftest"
        assert transfer_report["source_latent_concept_topk"] == 2
        assert transfer_report["source_reading_mastery_history_count"] == 1
        assert (transfer_report["source_reading_mastery_latest_self_teach_signal"]
                == "sequence")
        assert transfer_report["source_reading_concept_connection_signal"] > 0.0
        assert transfer_report["source_reading_concept_insight_max_delta"] >= 0.8
        assert transfer_report["source_reading_concept_insight_selected_count"] == 1
        assert transfer_report["source_reading_concept_insight_accepted_count"] == 1
        assert transfer_report["source_reading_weight_update_changed"] is True
        assert transfer_report[
            "source_reading_weight_update_changed_tensor_count"] == 5
        assert transfer_report[
            "source_reading_weight_update_changed_value_count"] == 7
        assert transfer_report[
            "source_reading_weight_update_max_abs_delta"] == 0.02
        assert transfer_report[
            "source_reading_attempted_weight_update_count"] == 1
        assert transfer_report[
            "source_reading_learning_event_triggered"] is True
        assert transfer_report[
            "source_reading_learning_event_triggered_count"] == 1
        assert transfer_report[
            "source_reading_learning_event_top_signal"] == "connection"
        assert transfer_report[
            "source_reading_learning_event_top_kind"] == "concept_connection"
        assert transfer_report[
            "source_reading_learning_event_max_score"] > 0.0
        assert transfer_report[
            "source_reading_representation_progress_enabled"] is True
        assert transfer_report[
            "source_reading_representation_top_gain_signal"] == "bridge"
        assert transfer_report[
            "source_reading_representation_insight_event_count"] == 1
        assert transfer_report["source_reading_priority_study_enabled"] is True
        assert transfer_report["source_reading_replay_study_entry_count"] == 1
        assert transfer_report[
            "source_reading_training_priority_entry_count"] == 1
        assert transfer_report[
            "source_reading_latest_training_priority_sampling"] is True
        assert transfer_report[
            "source_reading_training_priority_record_count"] == 2
        assert transfer_report["source_reading_replay_bank_records"] == 3
        assert transfer_report["source_reading_replay_priority_records"] == 2
        assert (transfer_report[
            "source_reading_learning_event_top_signal"] == "connection")
        assert transfer_report["copied_token_embeddings"] > 0
        assert transfer_report["copied_latent_tensor_count"] > 0
        assert transfer_report["copied_sequence_tensor_count"] > 0
        assert transfer_report["copied_completion_tensor_count"] > 0
        assert torch.allclose(
            transfer_model.txt.emb.weight[sample_idx],
            text_model.txt.emb.weight[sample_idx])
        assert torch.allclose(
            transfer_model.latent_concept_memory.memory,
            text_model.latent_concept_memory.memory)
        assert torch.allclose(
            transfer_model.concept_sequence_predictor[0].weight,
            text_model.reading_predictor[0].weight)
        assert torch.allclose(
            transfer_model.concept_completion_predictor[0].weight,
            text_model.reading_completion_predictor[0].weight)
        batch = records[:2]
        features, txt, ids = _batch_from_records(
            batch, vocab, "cpu", view_dims)
        logits = model(features, txt, ids)
        assert logits.shape == (2, ids.shape[1], len(vocab)), logits.shape
        prefix, concepts = model.encode_prefix(features, txt, mode="full")
        assert concepts.shape == (2, 2, 32)
        latent = model.latent_concept_states(features, txt, mode="full", project=True)
        assert latent.shape == (2, 3, 32)
        views = {
            mode: model.latent_concept_states(features, txt, mode=mode, project=True)
            for mode in MODES
        }
        assert torch.isfinite(latent_multimodal_concept_loss_from_views(views))
        assert torch.isfinite(latent_multimodal_factorization_loss_from_views(views))
        assert torch.isfinite(latent_multimodal_fer_loss_from_views(views))
        fer_metrics = latent_multimodal_fer_metrics_from_views(views)
        assert math.isfinite(fer_metrics["fer_score"])
        fer_scores, fer_parts = latent_multimodal_fer_scores_from_views(views)
        assert fer_scores.shape == (2,)
        assert fer_parts["fragmentation"].shape == (2,)
        fer_eval = latent_multimodal_fer_eval(
            model, records, vocab, view_dims, n=4, device="cpu")
        assert fer_eval["skipped"] is False
        assert math.isfinite(fer_eval["fer_score"])
        fer_selected, fer_report = latent_multimodal_fer_examples(
            model, records, vocab, view_dims, n=4, device="cpu")
        assert fer_selected and fer_report["skipped"] is False
        assert math.isfinite(fer_report["mean_fer_score"])
        cold_bridge_eval = latent_multimodal_bridge_eval(
            model, records, vocab, view_dims, n=4, device="cpu")
        assert cold_bridge_eval["skipped"] is True
        assert cold_bridge_eval["graph_ready"] is False
        assert update_multimodal_latent_memory(
            model, views["full"], relation_decay=0.5) > 0
        assert update_multimodal_latent_transitions(model, views, decay=0.5) > 0
        assert int(model.latent_concept_memory.transition_updates.item()) > 0
        seq_pairs, seq_report = mine_multimodal_sequence_pairs(records, split="train")
        assert seq_report["n_pairs"] == 7 and seq_report["skipped"] is False
        assert torch.isfinite(latent_multimodal_sequence_prediction_loss(
            model, seq_pairs[:2], vocab, view_dims, device="cpu", temperature=0.1))
        assert update_multimodal_sequence_transitions(
            model, seq_pairs[:2], vocab, view_dims, device="cpu", decay=0.5) > 0
        seq_eval = latent_multimodal_sequence_eval(
            model, records, vocab, view_dims, n=0, device="cpu")
        assert seq_eval["skipped"] is False
        assert math.isfinite(seq_eval["margin"])
        assert torch.isfinite(latent_multimodal_memory_loss_from_views(model, views))
        consolidation_loss, consolidation_metrics = (
            latent_multimodal_memory_consolidation_loss_from_views(
                model, views, anchor_w=1.0, fer_w=0.1))
        assert torch.isfinite(consolidation_loss)
        assert consolidation_metrics["skipped"] is False
        assert consolidation_metrics["memory_active"] > 0
        assert torch.isfinite(consolidation_metrics["anchor_loss"])
        assert torch.isfinite(latent_multimodal_association_loss_from_views(model, views))
        assert torch.isfinite(latent_multimodal_composition_loss_from_views(model, views))
        assert torch.isfinite(latent_multimodal_graph_prediction_loss_from_views(model, views))
        assert torch.isfinite(latent_multimodal_bridge_loss_from_views(model, views))
        completion_loss, completion_metrics = (
            latent_multimodal_completion_loss_from_views(model, views))
        assert torch.isfinite(completion_loss)
        assert completion_metrics["skipped"] is False
        assert completion_metrics["view_count"] >= 1
        completion_selected, completion_report = (
            latent_multimodal_completion_examples(
                model, records, vocab, view_dims, n=4, device="cpu"))
        assert completion_selected and completion_report["skipped"] is False
        assert math.isfinite(completion_report["mean_completion_surprise"])
        gap_loss, gap_metrics = latent_multimodal_gap_loss_from_views(model, views)
        assert torch.isfinite(gap_loss)
        assert gap_metrics["skipped"] is False
        assert gap_metrics["memory_active"] > 0
        assert torch.isfinite(gap_metrics["gap_kl"])
        bridge_metrics = latent_multimodal_bridge_metrics_from_views(model, views)
        assert math.isfinite(bridge_metrics["bridge_score"])
        bridge_eval = latent_multimodal_bridge_eval(
            model, records, vocab, view_dims, n=4, device="cpu")
        assert bridge_eval["skipped"] is False
        assert bridge_eval["graph_ready"] is True
        assert math.isfinite(bridge_eval["bridge_score"])
        synthetic_scores = {
            "token_skipped": False,
            "fer_skipped": False,
            "bridge_skipped": False,
            "connection_skipped": False,
            "sequence_skipped": False,
            "token_score": 1.0,
            "mode_floor_score": 0.0,
            "fer_score": 1.0,
            "bridge_score": 0.25,
            "connection_score": 1.0,
            "sequence_score": 0.5,
        }
        self_teach_plan = multimodal_self_teach_weight_plan(
            synthetic_scores, budget=0.12)
        assert self_teach_plan["enabled"] is True
        assert self_teach_plan["top_signal"] == "mode_floor"
        assert self_teach_plan["weight_extras"]["latent_concept_completion_w"] > 0.0
        assert self_teach_plan["weight_extras"]["latent_concept_bridge_w"] > 0.0
        assert self_teach_plan["weight_extras"]["latent_concept_sequence_w"] > 0.0
        assert math.isclose(sum(self_teach_plan["weight_extras"].values()),
                            0.12, rel_tol=1e-6, abs_tol=1e-6)
        history_scores = dict(synthetic_scores)
        history_scores["mode_floor_score"] = 1.0
        history_scores["bridge_score"] = 1.0
        history_scores["sequence_score"] = 1.0
        history_self_teach_plan = multimodal_self_teach_weight_plan(
            history_scores, budget=0.08,
            history_prior={
                "enabled": True,
                "entry_count": 1,
                "top_signal": "sequence",
                "signal_deficits": {"sequence": 1.0},
            },
            history_prior_w=1.0)
        assert history_self_teach_plan["top_signal"] == "sequence"
        assert history_self_teach_plan["history_prior_enabled"] is True
        assert history_self_teach_plan["history_prior_entry_count"] == 1
        assert (history_self_teach_plan["weight_extras"][
            "latent_concept_sequence_w"] > 0.0)
        assert math.isclose(
            sum(history_self_teach_plan["weight_extras"].values()),
            0.08, rel_tol=1e-6, abs_tol=1e-6)
        concept_history_plan = multimodal_self_teach_weight_plan(
            history_scores, budget=0.07,
            history_prior={
                "enabled": True,
                "entry_count": 1,
                "top_signal": "connection",
                "signal_deficits": {},
                "concept_connection_signal": 0.9,
            },
            history_prior_w=1.0)
        assert concept_history_plan["top_signal"] == "connection"
        assert (concept_history_plan["history_concept_connection_signal"]
                == 0.9)
        assert concept_history_plan["weight_extras"]["latent_concept_bridge_w"] > 0.0
        assert concept_history_plan["weight_extras"]["latent_concept_discovery_w"] > 0.0
        assert concept_history_plan["weight_extras"]["latent_concept_gap_w"] > 0.0
        assert (concept_history_plan["weight_extras"][
            "latent_concept_graph_predict_w"] > 0.0)
        assert math.isclose(
            sum(concept_history_plan["weight_extras"].values()),
            0.07, rel_tol=1e-6, abs_tol=1e-6)
        positive_insight = multimodal_bridge_selection_insight(
            {"skipped": False, "bridge_score": 0.8, "bridge_connectivity": 0.2},
            {"skipped": False, "bridge_score": 0.4, "bridge_connectivity": 0.5})
        assert positive_insight["bridge_insight_allowed"] is True
        assert positive_insight["bridge_insight_delta"] > 0.0
        positive_decision = concept_round_selection_decision(
            -0.01, 0.0, insight_delta=positive_insight["bridge_insight_delta"],
            insight_allowed=positive_insight["bridge_insight_allowed"],
            insight_gate=True, insight_accept_w=1.0)
        assert positive_decision["selected_by_insight"] is True
        mm_regression = signal_regression_report(
            {"generation_score": 0.30, "generation_skipped": False},
            {"generation_score": 0.25, "generation_skipped": False},
            MULTIMODAL_SELF_TEACH_SCORE_KEYS,
            skip_keys=MULTIMODAL_SELF_TEACH_SKIP_KEYS,
            tolerance=0.02,
            signals=("generation",))
        assert mm_regression["allowed"] is False
        assert mm_regression["regressions"]["generation"] > 0.02
        mm_tolerated_regression = signal_regression_report(
            {"generation_score": 0.30, "generation_skipped": False},
            {"generation_score": 0.29, "generation_skipped": False},
            MULTIMODAL_SELF_TEACH_SCORE_KEYS,
            skip_keys=MULTIMODAL_SELF_TEACH_SKIP_KEYS,
            tolerance=0.02,
            signals=("generation",))
        assert mm_tolerated_regression["allowed"] is True
        mm_nonfinite_regression = signal_regression_report(
            {"sequence_score": 0.30, "sequence_skipped": False},
            {"sequence_score": float("nan"), "sequence_skipped": False},
            MULTIMODAL_SELF_TEACH_SCORE_KEYS,
            skip_keys=MULTIMODAL_SELF_TEACH_SKIP_KEYS,
            tolerance=0.02,
            signals=("sequence",))
        assert mm_nonfinite_regression["allowed"] is False
        assert "sequence" in mm_nonfinite_regression["nonfinite_signals"]
        synthetic_learning_event = multimodal_learning_event_report({
            "weight_update_changed": True,
            "weight_update_changed_tensor_count": 2,
            "weight_update_changed_value_count": 3,
            "weight_update_max_abs_delta": 0.01,
            "selection": {
                "enabled": True,
                "accepted_update": True,
                "selected_score_delta": 0.01,
                "selected_by_insight": True,
                "selected_bridge_insight": positive_insight,
            },
            "representation_progress": {
                "enabled": True,
                "organization_score_delta": 0.1,
                "positive_signal_gain": 0.2,
                "top_gain_signal": "bridge",
                "representation_insight_event": True,
            },
        })
        assert synthetic_learning_event["triggered"] is True
        assert synthetic_learning_event["kind"] == "concept_connection"
        assert synthetic_learning_event["top_signal"] == "connection"
        event_only_prior = multimodal_checkpoint_representation_self_teach_prior({
            "report": {"learning_event": synthetic_learning_event},
        }, enabled=True)
        assert event_only_prior["enabled"] is True
        assert event_only_prior["signal_deficits"]["connection"] > 0.0
        synthetic_history = multimodal_learning_history_with_entry([], {
            "experiment": "synthetic-mm-history",
            "steps": 1,
            "seed": 7,
            "manifest": {
                "path": manifest,
                "records": len(records),
                "train_records": len(train_records),
                "eval_records": len(eval_records),
                "view_names": list(view_dims.keys()),
                "view_count": len(view_dims),
            },
            "architecture": model.config,
            "learning_event": synthetic_learning_event,
            "representation_progress": {
                "enabled": True,
                "signal_after": {"bridge": 0.2},
                "signal_deltas": {"bridge": 0.1},
                "organization_score_delta": 0.1,
                "positive_signal_gain": 0.2,
                "top_gain_signal": "bridge",
                "representation_insight_event": True,
            },
            "selection": {
                "enabled": True,
                "accepted_update": True,
                "selected_round": 1,
                "selected_by_insight": True,
                "selected_bridge_insight": positive_insight,
            },
            "train_metrics": {
                "weight_update_changed": True,
                "weight_update_changed_tensor_count": 2,
                "weight_update_changed_value_count": 3,
                "weight_update_max_abs_delta": 0.01,
                "attempted_weight_update_count": 1,
            },
        })
        assert synthetic_history["entry_count"] == 1
        assert (synthetic_history["entries"][0]["learning_event_triggered"]
                is True)
        assert (synthetic_history["entries"][0][
            "learning_event_top_signal"] == "connection")
        history_prior = multimodal_checkpoint_representation_self_teach_prior({
            "multimodal_learning_history": synthetic_history,
        }, enabled=True)
        assert history_prior["enabled"] is True
        assert history_prior["entry_count"] == 1
        assert history_prior["signal_deficits"]["connection"] > 0.0
        history_summary = multimodal_checkpoint_learning_history_summary({
            "multimodal_learning_history": synthetic_history,
        })
        assert history_summary["triggered_count"] == 1
        assert history_summary["top_signal"] == "connection"
        negative_insight = multimodal_bridge_selection_insight(
            {"skipped": False, "bridge_score": 0.4, "bridge_connectivity": 0.5},
            {"skipped": False, "bridge_score": 0.8, "bridge_connectivity": 0.2})
        assert negative_insight["bridge_insight_allowed"] is False
        negative_decision = concept_round_selection_decision(
            0.2, 0.0, insight_delta=negative_insight["bridge_insight_delta"],
            insight_allowed=negative_insight["bridge_insight_allowed"],
            insight_gate=True, insight_accept_w=1.0)
        assert negative_decision["selected"] is False
        positive_transfer = multimodal_transfer_calibration_report(
            {"score_components": {
                "score": 0.50,
                "fer_score": 0.2,
                "bridge_score": 0.2,
                "sequence_score": 0.2,
                "fer_skipped": False,
                "bridge_skipped": False,
                "sequence_skipped": False,
            }},
            {"score_components": {
                "score": 0.49,
                "fer_score": 0.5,
                "bridge_score": 0.5,
                "sequence_score": 0.5,
                "fer_skipped": False,
                "bridge_skipped": False,
                "sequence_skipped": False,
            }},
            insight_accept_w=1.0)
        assert positive_transfer["accepted"] is True
        assert positive_transfer["selected_by_insight"] is True
        negative_transfer = multimodal_transfer_calibration_report(
            {"score_components": {
                "score": 0.50,
                "fer_score": 0.7,
                "bridge_score": 0.7,
                "sequence_score": 0.7,
                "fer_skipped": False,
                "bridge_skipped": False,
                "sequence_skipped": False,
            }},
            {"score_components": {
                "score": 0.70,
                "fer_score": 0.1,
                "bridge_score": 0.1,
                "sequence_score": 0.1,
                "fer_skipped": False,
                "bridge_skipped": False,
                "sequence_skipped": False,
            }},
            insight_accept_w=1.0)
        assert negative_transfer["accepted"] is False
        assert negative_transfer["reverted"] is True
        assert negative_transfer["insight_allowed"] is False
        graph_selected, graph_report = latent_multimodal_graph_prediction_examples(
            model, records, vocab, view_dims, n=4, device="cpu")
        assert graph_selected and graph_report["skipped"] is False
        assert len({r.rec_id for r in graph_selected}) == len(graph_selected)
        discovery_selected, discovery_report = latent_multimodal_discovery_examples(
            model, records, vocab, view_dims, n=4, device="cpu")
        assert discovery_selected and discovery_report["skipped"] is False
        assert math.isfinite(discovery_report["mean_score"])
        assert "mean_gap_score" in discovery_report
        assert "mean_insight_score" in discovery_report
        assert "mean_cycle_score" in discovery_report
        assert "mean_completion_surprise" in discovery_report
        assert "mean_sequence_surprise" in discovery_report
        discovery_loss, discovery_metrics = (
            latent_multimodal_discovery_loss_from_views(
                model, views, graph_w=1.0, cycle_w=1.0,
                bridge_w=1.0, fer_w=0.1))
        assert torch.isfinite(discovery_loss)
        assert discovery_metrics["skipped"] is False
        assert discovery_metrics["memory_active"] > 0
        assert torch.isfinite(discovery_metrics["graph_loss"])
        assert torch.isfinite(discovery_metrics["insight_loss"])
        assert torch.isfinite(discovery_metrics["bridge_loss"])
        reanalysis_loss, reanalysis_metrics = (
            latent_multimodal_reanalysis_loss_from_views(
                model, views, graph_w=1.0, cycle_w=1.0,
                bridge_w=1.0, fer_w=0.1))
        assert torch.isfinite(reanalysis_loss)
        assert reanalysis_metrics["skipped"] is False
        assert reanalysis_metrics["memory_active"] > 0
        assert torch.isfinite(reanalysis_metrics["closure_loss"])
        assert torch.isfinite(latent_multimodal_neighborhood_loss_from_views(views))
        assert torch.isfinite(latent_multimodal_transition_loss_from_views(views))
        assert torch.isfinite(latent_multimodal_cluster_loss_from_views(views))
        trained_model, *_ = train(
            manifest, steps=2, batch=2, d=32, layers=1, heads=4, device="cpu",
            objective_profile="manual",
            log_every=1, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8, latent_concept_memory_w=0.01,
            latent_concept_fer_w=0.01,
            latent_concept_discovery_w=0.01,
            latent_concept_discovery_fer_w=0.01,
            latent_concept_reanalysis_w=0.01,
            latent_concept_reanalysis_fer_w=0.01,
            latent_concept_gap_w=0.01,
            latent_concept_fer_probe_n=4,
            latent_concept_fer_hard_max=2,
            latent_concept_fer_refresh_steps=1,
            latent_concept_association_w=0.01,
            latent_concept_composition_w=0.01,
            latent_concept_graph_predict_w=0.01,
            latent_concept_bridge_w=0.01,
            latent_concept_completion_w=0.01,
            latent_concept_sequence_w=0.01,
            latent_concept_sequence_batch=2,
            latent_concept_neighborhood_w=0.01,
            latent_concept_transition_w=0.01,
            latent_concept_cluster_w=0.01,
            representation_probe_n=4)
        assert trained_model.train_metrics["latent_graph_transition_updates"] > 0
        assert trained_model.train_metrics["latent_fer_w"] == 0.01
        assert math.isfinite(trained_model.train_metrics["latent_fer_score"])
        assert trained_model.train_metrics["latent_discovery_w"] == 0.01
        assert trained_model.train_metrics["latent_discovery_fer_w"] == 0.01
        assert trained_model.train_metrics["latent_discovery_skipped"] is False
        assert math.isfinite(trained_model.train_metrics["latent_discovery_loss"])
        assert math.isfinite(
            trained_model.train_metrics["latent_discovery_graph_loss"])
        assert math.isfinite(
            trained_model.train_metrics["latent_discovery_insight_loss"])
        assert trained_model.train_metrics["latent_reanalysis_w"] == 0.01
        assert trained_model.train_metrics["latent_reanalysis_fer_w"] == 0.01
        assert trained_model.train_metrics["latent_reanalysis_skipped"] is False
        assert math.isfinite(trained_model.train_metrics["latent_reanalysis_loss"])
        assert math.isfinite(
            trained_model.train_metrics["latent_reanalysis_closure_loss"])
        assert trained_model.train_metrics["latent_gap_w"] == 0.01
        assert trained_model.train_metrics["latent_gap_skipped"] is False
        assert math.isfinite(trained_model.train_metrics["latent_gap_loss"])
        assert math.isfinite(trained_model.train_metrics["latent_gap_kl"])
        assert trained_model.train_metrics["latent_fer_study_pool_size"] == 2
        assert trained_model.train_metrics["latent_fer_study_reports"]
        assert len(set(trained_model.train_metrics["latent_fer_hard_record_ids"])) == 2
        assert trained_model.train_metrics["latent_bridge_w"] == 0.01
        assert trained_model.train_metrics["latent_bridge_graph_ready"] is True
        assert trained_model.train_metrics["latent_bridge_skipped"] is False
        assert math.isfinite(trained_model.train_metrics["latent_bridge_score"])
        assert trained_model.train_metrics["latent_completion_w"] == 0.01
        assert trained_model.train_metrics["latent_completion_skipped"] is False
        assert trained_model.train_metrics["latent_completion_view_count"] >= 1
        assert math.isfinite(trained_model.train_metrics["latent_completion_loss"])
        assert trained_model.train_metrics["latent_sequence_w"] == 0.01
        assert trained_model.train_metrics["latent_sequence_pairs"] == 7
        assert math.isfinite(trained_model.train_metrics["latent_sequence_loss"])
        assert (trained_model.train_metrics[
            "latent_sequence_transition_last_batch_updates"] > 0)
        assert trained_model.train_metrics["weight_update_changed"] is True
        assert (trained_model.train_metrics[
            "weight_update_changed_tensor_count"] > 0)
        assert (trained_model.train_metrics[
            "weight_update_changed_value_count"] > 0)
        assert trained_model.train_metrics["weight_update_max_abs_delta"] > 0.0
        assert trained_model.train_metrics["learning_event"]["enabled"] is True
        assert (trained_model.train_metrics["learning_event"][
            "weight_update_changed"] is True)
        rep_progress = trained_model.train_metrics["representation_progress"]
        assert rep_progress["enabled"] is True
        assert trained_model.train_metrics["representation_probe_n"] == 4
        assert "fer" in rep_progress["active_signals"]
        assert "bridge" in rep_progress["active_signals"]
        assert "sequence" in rep_progress["active_signals"]
        assert math.isfinite(rep_progress["organization_score_before"])
        assert math.isfinite(rep_progress["organization_score_after"])
        assert math.isfinite(rep_progress["organization_score_delta"])
        assert isinstance(rep_progress["representation_insight_event"], bool)
        mm_ckpt = os.path.join(tmpdir, "multimodal_continue.pt")
        mm_report = {
            "experiment": "multimodal-selftest",
            "steps": 2,
            "seed": 0,
            "manifest": trained_model.manifest_info,
            "architecture": trained_model.config,
            "train_metrics": trained_model.train_metrics,
            "learning_event": trained_model.train_metrics["learning_event"],
            "representation_progress": rep_progress,
        }
        mm_history = multimodal_learning_history_with_entry([], mm_report)
        mm_report["multimodal_learning_history"] = mm_history
        mm_report["multimodal_learning_history_count"] = mm_history["entry_count"]
        torch.save({
            "state_dict": trained_model.state_dict(),
            "vocab": vocab.itos,
            "d": trained_model.lm.tok.embedding_dim,
            "model_config": trained_model.config,
            "manifest": trained_model.manifest_info,
            "report": mm_report,
            "multimodal_learning_history": mm_history,
        }, mm_ckpt)
        mm_payload = load_multimodal_checkpoint_payload(mm_ckpt, device="cpu")
        assert (multimodal_learning_history_from_payload(mm_payload)[
            "entry_count"] == 1)
        mm_prior = multimodal_checkpoint_representation_self_teach_prior(
            mm_payload, enabled=True)
        assert mm_prior["enabled"] is True
        assert mm_prior["entry_count"] == 1
        assert mm_prior["signal_deficits"]
        fresh_model = MultimodalLM(
            len(vocab), view_dims=view_dims, d=32, layers=1,
            heads=4, pad=vocab.pad, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8).to("cpu")
        mm_transfer = import_multimodal_checkpoint(
            fresh_model, vocab, mm_ckpt, device="cpu")
        assert mm_transfer["copied"] is True
        assert mm_transfer["checkpoint_experiment"] == "multimodal-selftest"
        assert mm_transfer["source_representation_prior"]["enabled"] is True
        assert "source_learning_event_triggered" in mm_transfer
        assert (mm_transfer["source_learning_event_summary"].get("enabled")
                is True)
        assert mm_transfer["source_learning_history_count"] == 1
        assert mm_transfer["source_learning_event_triggered_count"] >= 0
        assert mm_transfer["source_learning_history_summary"]["event_count"] == 1
        assert mm_transfer["copied_token_embeddings"] > 0
        assert mm_transfer["copied_exact_tensor_count"] > 0
        assert torch.allclose(
            fresh_model.lm.tok.weight[sample_idx],
            trained_model.lm.tok.weight[sample_idx])
        self_teach_model, *_ = train(
            manifest, steps=1, batch=2, d=32, layers=1, heads=4, device="cpu",
            objective_profile="manual",
            log_every=10, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8, text_checkpoint=text_ckpt,
            self_teach_w=0.05, self_teach_history_prior_w=1.0,
            text_transfer_probe_n=0)
        assert self_teach_model.latent_concept_topk == 2
        assert self_teach_model.train_metrics["self_teach_w"] == 0.05
        assert self_teach_model.train_metrics[
            "self_teach_history_prior_w"] == 1.0
        assert self_teach_model.train_metrics[
            "text_checkpoint_history_prior"]["enabled"] is True
        assert self_teach_model.train_metrics[
            "text_checkpoint_history_prior"]["entry_count"] == 1
        assert self_teach_model.train_metrics["self_teach_plan"]["enabled"] is True
        assert self_teach_model.train_metrics["self_teach_plan"][
            "history_prior_enabled"] is True
        assert self_teach_model.train_metrics["self_teach_plan"][
            "history_prior_entry_count"] == 1
        assert self_teach_model.train_metrics["self_teach_plan"][
            "history_signal_deficits"]["sequence"] > 0.0
        assert self_teach_model.train_metrics["self_teach_plan"][
            "history_concept_connection_signal"] > 0.0
        assert (self_teach_model.train_metrics["self_teach_plan"][
            "weight_extras"]["latent_concept_sequence_w"] > 0.0)
        assert self_teach_model.text_checkpoint_transfer[
            "source_reading_mastery_history_count"] == 1
        assert self_teach_model.text_checkpoint_transfer[
            "source_reading_concept_connection_signal"] > 0.0
        assert self_teach_model.text_checkpoint_transfer[
            "source_reading_weight_update_changed"] is True
        assert self_teach_model.text_checkpoint_transfer[
            "source_reading_learning_event_triggered"] is True
        assert self_teach_model.train_metrics["weight_update_changed"] is True
        assert self_teach_model.train_metrics["selection"]["enabled"] is False
        assert self_teach_model.train_metrics["selection"]["self_teach_reports"]
        self_teach_extra_sum = sum(
            self_teach_model.train_metrics["self_teach_plan"][
                "weight_extras"].values())
        assert self_teach_extra_sum > 0.0
        self_teach_delta = sum(
            self_teach_model.train_metrics["self_teach_effective_weights"][key]
            - self_teach_model.train_metrics["self_teach_base_weights"][key]
            for key in MULTIMODAL_SELF_TEACH_WEIGHT_KEYS)
        assert math.isclose(
            self_teach_delta, self_teach_extra_sum,
            rel_tol=1e-6, abs_tol=1e-6)
        generation_plan = multimodal_self_teach_weight_plan({
            "token_score": 1.0,
            "generation_score": 0.0,
            "mode_floor_score": 1.0,
            "fer_score": 1.0,
            "bridge_score": 1.0,
            "connection_score": 1.0,
            "sequence_score": 1.0,
            "token_skipped": False,
            "generation_skipped": False,
            "fer_skipped": False,
            "bridge_skipped": False,
            "connection_skipped": False,
            "sequence_skipped": False,
        }, budget=0.09)
        assert generation_plan["top_signal"] == "generation"
        assert generation_plan["signal_deficits"]["generation"] > 0.0
        assert math.isclose(
            generation_plan["weight_extras"]["decode_w"], 0.03,
            rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(
            generation_plan["weight_extras"]["continuation_repair_w"], 0.03,
            rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(
            generation_plan["weight_extras"]["repetition_unlikelihood_w"], 0.03,
            rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(
            sum(generation_plan["weight_extras"].values()), 0.09,
            rel_tol=1e-6, abs_tol=1e-6)
        language_only_objectives = multimodal_self_teach_available_objectives(
            latent_concept_slots=0, active_mode_count=1)
        assert "decode_w" in language_only_objectives
        assert "continuation_repair_w" in language_only_objectives
        assert "repetition_unlikelihood_w" in language_only_objectives
        assert "latent_concept_sequence_w" not in language_only_objectives
        language_only_plan = multimodal_self_teach_weight_plan({
            "token_score": 1.0,
            "generation_score": 0.0,
            "mode_floor_score": 0.0,
            "fer_score": 1.0,
            "bridge_score": 1.0,
            "connection_score": 1.0,
            "sequence_score": 0.0,
            "token_skipped": False,
            "generation_skipped": False,
            "fer_skipped": False,
            "bridge_skipped": False,
            "connection_skipped": False,
            "sequence_skipped": False,
        }, budget=0.09, available_objectives=language_only_objectives)
        assert language_only_plan["top_signal"] == "generation"
        assert "sequence" in language_only_plan["unavailable_signals"]
        assert all(
            language_only_plan["weight_extras"][key] == 0.0
            for key in MULTIMODAL_LATENT_SELF_TEACH_WEIGHT_KEYS)
        assert math.isclose(
            language_only_plan["weight_extras"]["decode_w"], 0.03,
            rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(
            language_only_plan["weight_extras"]["continuation_repair_w"], 0.03,
            rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(
            language_only_plan["weight_extras"]["repetition_unlikelihood_w"], 0.03,
            rel_tol=1e-6, abs_tol=1e-6)
        calibrated_transfer_model, *_ = train(
            manifest, steps=1, batch=2, d=32, layers=1, heads=4, device="cpu",
            objective_profile="manual",
            log_every=10, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8, text_checkpoint=text_ckpt,
            self_teach_w=0.05, self_teach_history_prior_w=1.0,
            text_transfer_probe_n=4, text_transfer_insight_accept_w=1.0)
        calibrated_report = calibrated_transfer_model.text_checkpoint_transfer
        assert calibrated_report["target_calibration"]["enabled"] is True
        assert "score_delta" in calibrated_report["target_calibration"]
        if calibrated_report["target_calibration"]["accepted"]:
            assert calibrated_transfer_model.train_metrics[
                "text_checkpoint_history_prior"]["enabled"] is True
        else:
            assert calibrated_transfer_model.train_metrics[
                "text_checkpoint_history_prior"]["enabled"] is False
            assert calibrated_transfer_model.train_metrics[
                "text_checkpoint_history_prior"]["disabled_reason"] == (
                    "text_transfer_rejected_by_target_probe")
        continued_model, *_ = train(
            manifest, steps=1, batch=2, d=32, layers=1, heads=4, device="cpu",
            objective_profile="manual",
            log_every=10, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8, multimodal_checkpoint=mm_ckpt,
            self_teach_w=0.05, self_teach_history_prior_w=1.0,
            representation_probe_n=4)
        assert continued_model.multimodal_checkpoint_transfer["copied"] is True
        assert (continued_model.train_metrics[
            "multimodal_checkpoint_history_prior"]["enabled"] is True)
        assert continued_model.train_metrics[
            "self_teach_history_prior"]["enabled"] is True
        assert continued_model.train_metrics["self_teach_plan"][
            "history_prior_enabled"] is True
        assert continued_model.train_metrics["self_teach_plan"][
            "history_prior_entry_count"] >= 1
        assert continued_model.train_metrics["weight_update_changed"] is True
        imbalanced_manifest = os.path.join(tmpdir, "mm_imbalanced_sources.jsonl")
        imbalanced_rows = []
        for i in range(8):
            source = "rare" if i in (5, 7) else "bulk"
            split = "eval" if i >= 6 else "train"
            vec = np.zeros(3, dtype=np.float32)
            vec[i % 3] = 1.0
            imbalanced_rows.append({
                "id": f"{source}-{i}",
                "split": split,
                "text": ["caption", source, str(i)],
                "views": {"sensor": vec.astype(float).tolist()},
                "target": ["source", source, "done"],
                "meta": {"source": source, "chunk_index": i},
            })
        with open(imbalanced_manifest, "w") as f:
            for row in imbalanced_rows:
                f.write(json.dumps(row) + "\n")
        balanced_model, *_ = train(
            imbalanced_manifest, steps=1, batch=2, d=32, layers=1, heads=4,
            device="cpu", objective_profile="manual",
            log_every=10, view_tokens=2, txt_tokens=4,
            source_balance_w=1.0)
        assert balanced_model.train_metrics[
            "training_source_balance_sampling"] is True
        assert balanced_model.train_metrics["training_source_balance_w"] == 1.0
        assert balanced_model.train_metrics["training_source_count"] == 2
        assert balanced_model.manifest_info["source_count"] == 2
        language_profile_model, *_ = train(
            manifest, steps=2, batch=2, d=32, layers=1, heads=4, device="cpu",
            log_every=10, view_tokens=2, txt_tokens=4, concept_tokens=2)
        language_profile = language_profile_model.train_metrics[
            "objective_profile_report"]
        assert language_profile_model.train_metrics["objective_profile"] == "language"
        assert language_profile["language_first"] is True
        assert language_profile["applied"] is True
        assert language_profile["updates"]["continuation_repair_w"]["to"] == (
            MULTIMODAL_LANGUAGE_OBJECTIVE_FLOORS["continuation_repair_w"])
        assert language_profile["updates"]["repetition_unlikelihood_w"]["to"] == (
            MULTIMODAL_LANGUAGE_OBJECTIVE_FLOORS["repetition_unlikelihood_w"])
        assert language_profile["updates"]["selection_generation_n"]["to"] == (
            MULTIMODAL_LANGUAGE_OBJECTIVE_FLOORS["selection_generation_n"])
        assert (language_profile_model.train_metrics["continuation_repair_w"]
                == MULTIMODAL_LANGUAGE_OBJECTIVE_FLOORS["continuation_repair_w"])
        assert (language_profile_model.train_metrics["repetition_unlikelihood_w"]
                == MULTIMODAL_LANGUAGE_OBJECTIVE_FLOORS[
                    "repetition_unlikelihood_w"])
        assert (language_profile_model.train_metrics["selection_generation_n"]
                == MULTIMODAL_LANGUAGE_OBJECTIVE_FLOORS[
                    "selection_generation_n"])
        assert language_profile_model.train_metrics[
            "continuation_repair_skipped"] is True
        assert language_profile_model.latent_concept_slots == 0
        assert language_profile_model.train_metrics["active_modes"] == list(MODES)
        mastery_profile_model, *_ = train(
            manifest, steps=2, batch=2, d=32, layers=1, heads=4, device="cpu",
            objective_profile="mastery",
            log_every=10, view_tokens=2, txt_tokens=4, concept_tokens=2)
        mastery_profile = mastery_profile_model.train_metrics[
            "objective_profile_report"]
        assert mastery_profile_model.train_metrics["objective_profile"] == "mastery"
        assert mastery_profile["applied"] is True
        assert mastery_profile["updates"]["latent_concept_slots"]["to"] == (
            MULTIMODAL_DEFAULT_LATENT_CONCEPT_SLOTS)
        assert mastery_profile["updates"]["latent_concept_memory_size"]["to"] == (
            MULTIMODAL_DEFAULT_LATENT_CONCEPT_MEMORY_SIZE)
        assert (mastery_profile_model.latent_concept_slots
                == MULTIMODAL_DEFAULT_LATENT_CONCEPT_SLOTS)
        assert mastery_profile_model.train_metrics["latent_memory_active"] > 0
        assert mastery_profile_model.train_metrics["latent_completion_w"] >= 0.10
        assert mastery_profile_model.train_metrics["latent_completion_skipped"] is False
        assert mastery_profile_model.train_metrics["latent_sequence_w"] >= 0.10
        assert mastery_profile_model.train_metrics["latent_sequence_pairs"] > 0
        assert mastery_profile_model.train_metrics["latent_gap_w"] >= 0.05
        assert mastery_profile_model.train_metrics["latent_gap_skipped"] is False
        assert mastery_profile_model.train_metrics["self_teach_plan"]["enabled"] is True
        no_target_manifest = os.path.join(tmpdir, "mm_no_target.jsonl")
        no_target_rows = []
        for i in range(6):
            base = np.zeros(3, dtype=np.float32)
            base[i % 3] = 1.0
            no_target_rows.append({
                "id": f"untargeted-{i}",
                "split": "eval" if i >= 4 else "train",
                "text": ["caption", "cluster", str(i % 3)],
                "views": {"sensor": base.astype(float).tolist()},
                "meta": {"source": "untargeted", "chunk_index": i},
            })
        with open(no_target_manifest, "w") as f:
            for row in no_target_rows:
                f.write(json.dumps(row) + "\n")
        no_target_records = load_manifest(no_target_manifest)
        assert no_target_records[0].target == ()
        assert infer_multimodal_decode_objective(
            no_target_records, requested="auto") == "causal"
        no_target_model, no_target_vocab, _all, _trn, no_target_eval, no_target_dims = train(
            no_target_manifest, steps=1, batch=2, d=32, layers=1, heads=4,
            device="cpu", objective_profile="manual",
            log_every=10, view_tokens=2, txt_tokens=4,
            decode_w=0.0, concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8, latent_concept_memory_w=0.01,
            latent_concept_bridge_w=0.01)
        assert no_target_model.train_metrics["decode_w"] == 0.0
        assert no_target_model.train_metrics["decode_objective"] == "causal"
        assert (no_target_model.train_metrics["decode_objective_report"][
            "causal_reason"] == "targetless_text")
        no_target_bundle = multimodal_eval_bundle(
            no_target_model, no_target_eval, no_target_vocab, no_target_dims,
            n=0, device="cpu", score_metric="mastery",
            decode_objective=no_target_model.manifest_info["decode_objective"])
        assert no_target_bundle["score_components"]["token_skipped"] is False
        assert no_target_bundle["score_components"]["exact_skipped"] is False
        assert no_target_bundle["score_components"]["bridge_skipped"] is False
        code_path = os.path.join(tmpdir, "toy_code.py")
        with open(code_path, "w") as f:
            f.write(
                "def add(a, b):\n"
                "    return a + b\n\n"
                "def factorial(n):\n"
                "    out = 1\n"
                "    for i in range(2, n + 1):\n"
                "        out *= i\n"
                "    return out\n")
        causal_manifest = os.path.join(tmpdir, "code_causal.jsonl")
        causal_report = build_text_causal_lm_manifest(
            [code_path], causal_manifest, max_tokens=10, min_tokens=4,
            stride=5, eval_frac=0.25, seed=0)
        assert causal_report["target_records"] == 0
        assert causal_report["targetless"] is True
        assert causal_report["split_report"]["eval_split_policy"] == (
            "contiguous_window")
        assert causal_report["window_stride"] == 5
        causal_records = load_manifest(causal_manifest)
        assert len(causal_records) >= 2
        assert all(not rec.target for rec in causal_records)
        starts = [int(rec.meta["token_start"]) for rec in causal_records]
        assert any((b - a) == 5 for a, b in zip(starts, starts[1:]))
        assert infer_multimodal_decode_objective(
            causal_records, requested="auto") == "causal"
        assert multimodal_active_modes(feature_dims(causal_records),
                                       causal_records) == ("full",)
        other_code_path = os.path.join(tmpdir, "toy_code_other.py")
        with open(other_code_path, "w") as f:
            f.write(
                "class Counter:\n"
                "    def __init__(self):\n"
                "        self.value = 0\n"
                "    def inc(self):\n"
                "        self.value += 1\n")
        source_split_manifest = os.path.join(tmpdir, "source_split_causal.jsonl")
        source_split_report = build_text_causal_lm_manifest(
            [code_path, other_code_path], source_split_manifest,
            max_tokens=8, min_tokens=4, stride=4, eval_frac=0.5, seed=1)
        assert source_split_report["target_records"] == 0
        assert source_split_report["split_report"]["eval_split_policy"] == "source"
        source_split_records = load_manifest(source_split_manifest)
        assert all(not rec.target for rec in source_split_records)
        train_sources = {
            rec.meta["source"] for rec in source_split_records
            if rec.split == "train"
        }
        eval_sources = {
            rec.meta["source"] for rec in source_split_records
            if rec.split == "eval"
        }
        assert train_sources
        assert eval_sources
        assert train_sources.isdisjoint(eval_sources)
        causal_model, causal_vocab, _all, _trn, causal_eval, causal_dims = train(
            causal_manifest, steps=2, batch=2, d=32, layers=1, heads=4,
            device="cpu", objective_profile="language",
            log_every=10, view_tokens=2, txt_tokens=4, concept_tokens=2)
        assert causal_model.train_metrics["objective_profile"] == "language"
        assert causal_model.train_metrics["active_modes"] == ["full"]
        assert causal_model.train_metrics["decode_objective"] == "causal"
        assert causal_model.train_metrics["token_loss"] > 0.1
        assert causal_model.train_metrics["continuation_repair_w"] > 0.0
        assert causal_model.train_metrics["continuation_repair_enabled"] is True
        assert causal_model.train_metrics["continuation_repair_tokens"] > 0
        assert causal_model.train_metrics["repetition_unlikelihood_w"] > 0.0
        assert causal_model.train_metrics["repetition_unlikelihood_enabled"] is True
        assert causal_model.train_metrics["repetition_unlikelihood_candidates"] > 0
        causal_self_teach_model, *_ = train(
            causal_manifest, steps=1, batch=2, d=32, layers=1, heads=4,
            device="cpu", objective_profile="language",
            log_every=10, view_tokens=2, txt_tokens=4, concept_tokens=2,
            latent_concept_slots=0, latent_concept_memory_size=0,
            self_teach_w=0.05, text_transfer_probe_n=0)
        assert causal_self_teach_model.latent_concept_slots == 0
        assert causal_self_teach_model.train_metrics["self_teach_w"] == 0.05
        assert causal_self_teach_model.train_metrics[
            "self_teach_plan"]["enabled"] is True
        assert all(
            key in causal_self_teach_model.train_metrics[
                "self_teach_available_objectives"]
            for key in (
                "decode_w", "continuation_repair_w",
                "repetition_unlikelihood_w"))
        assert all(
            causal_self_teach_model.train_metrics[
                "self_teach_effective_weights"][key] == 0.0
            for key in MULTIMODAL_LATENT_SELF_TEACH_WEIGHT_KEYS)
        assert causal_self_teach_model.train_metrics[
            "self_teach_effective_weights"]["decode_w"] >= (
                causal_self_teach_model.train_metrics[
                    "self_teach_base_weights"]["decode_w"])
        causal_bundle = multimodal_eval_bundle(
            causal_model, causal_eval, causal_vocab, causal_dims, n=0,
            device="cpu", decode_objective="causal", generation_eval_n=2,
            generation_prompt_tokens=3, generation_max_new_tokens=4)
        assert list(causal_bundle["teacher_forced"]) == ["full"]
        assert causal_bundle["generation"]["enabled"] is True
        assert causal_bundle["score_components"]["token_skipped"] is False
        assert causal_bundle["score_components"]["generation_skipped"] is False
        assert math.isfinite(
            causal_bundle["score_components"]["generation_score"])
        causal_generation = multimodal_generation_eval(
            causal_model, causal_eval, causal_vocab, causal_dims, n=1,
            device="cpu", decode_objective="causal", prompt_tokens=3,
            max_new_tokens=4)
        assert causal_generation["enabled"] is True
        assert causal_generation["mode"] == "full"
        assert causal_generation["unique_generation_count"] >= 1
        assert isinstance(causal_generation["all_generations_identical"], bool)
        assert causal_generation["samples"]
        assert "generated" in causal_generation["samples"][0]
        prompt_generation = multimodal_prompt_generation_report(
            causal_model, ["def add"], causal_vocab, causal_dims,
            device="cpu", max_new_tokens=4)
        assert prompt_generation["enabled"] is True
        assert prompt_generation["unique_generation_count"] >= 1
        assert prompt_generation["samples"][0]["prompt_tokens"]
        assert "generated" in prompt_generation["samples"][0]
        completion_model, *_ = train(
            manifest, steps=1, batch=2, d=32, layers=1, heads=4, device="cpu",
            objective_profile="manual",
            log_every=10, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_completion_probe_n=4,
            latent_concept_completion_hard_max=2,
            latent_concept_completion_refresh_steps=1,
            latent_concept_completion_w=0.01)
        assert completion_model.train_metrics["latent_study_strategy"] == "completion"
        assert completion_model.train_metrics["latent_completion_study_pool_size"] == 2
        assert completion_model.train_metrics["latent_completion_study_reports"]
        assert completion_model.train_metrics["latent_completion_skipped"] is False
        assert any(r.get("strategy") == "completion"
                   and "mean_completion_surprise" in r
                   for r in completion_model.train_metrics["latent_study_reports"])
        discovery_model, *_ = train(
            manifest, steps=2, batch=2, d=32, layers=1, heads=4, device="cpu",
            objective_profile="manual",
            log_every=2, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8,
            latent_concept_discovery_probe_n=4,
            latent_concept_discovery_hard_max=2,
            latent_concept_discovery_refresh_steps=1,
            latent_concept_consolidation_w=0.01,
            latent_concept_consolidation_fer_w=0.01,
            latent_concept_graph_predict_w=0.01,
            latent_concept_bridge_w=0.01,
            latent_concept_sequence_w=0.01,
            latent_concept_sequence_batch=2)
        assert discovery_model.train_metrics["latent_study_strategy"] == "discovery"
        assert discovery_model.train_metrics["latent_consolidation_w"] == 0.01
        assert discovery_model.train_metrics["latent_consolidation_fer_w"] == 0.01
        assert discovery_model.train_metrics["latent_consolidation_skipped"] is False
        assert math.isfinite(discovery_model.train_metrics[
            "latent_consolidation_anchor_loss"])
        assert discovery_model.train_metrics["latent_discovery_study_pool_size"] == 2
        assert discovery_model.train_metrics["latent_discovery_study_reports"]
        assert len(set(discovery_model.train_metrics[
            "latent_discovery_hard_record_ids"])) == 2
        assert any(r.get("strategy") == "discovery"
                   for r in discovery_model.train_metrics["latent_study_reports"])
        assert any("mean_gap_score" in r
                   for r in discovery_model.train_metrics["latent_study_reports"]
                   if r.get("strategy") == "discovery")
        assert any("mean_insight_score" in r
                   for r in discovery_model.train_metrics["latent_study_reports"]
                   if r.get("strategy") == "discovery")
        assert any("mean_completion_surprise" in r
                   for r in discovery_model.train_metrics["latent_study_reports"]
                   if r.get("strategy") == "discovery")
        selected_model, *_ = train(
            manifest, steps=2, batch=2, d=32, layers=1, heads=4, device="cpu",
            objective_profile="manual",
            log_every=10, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8,
            latent_concept_discovery_probe_n=4,
            latent_concept_discovery_hard_max=2,
            latent_concept_discovery_refresh_steps=1,
            latent_concept_graph_predict_w=0.01,
            latent_concept_bridge_w=0.01,
            latent_concept_sequence_w=0.01,
            latent_concept_sequence_batch=2,
            self_teach_w=0.05,
            select_best=True, selection_rounds=2, selection_eval_n=4,
            selection_score_min_delta=999.0, selection_score_patience=1)
        selection = selected_model.train_metrics["selection"]
        assert selection["enabled"] is True
        assert selection["stopped_early"] is True
        assert selection["rounds_run"] == 1
        assert selection["stop_round"] == 1
        assert selection["accepted_update"] is False
        assert selection["selected_round"] == 0
        assert "score_delta_from_best" in selection["rounds"][1]
        assert "bridge_insight_delta" in selection["rounds"][1]
        assert selection["bridge_insight_gate"] is True
        assert selection["selected_by_score"] is False
        assert selection["selected_by_insight"] is False
        assert selection["signal_regression_gate"] is True
        assert selection["signal_regression_tolerance"] == 0.02
        assert selection["self_teach_w"] == 0.05
        assert selection["self_teach_reports"]
        assert selection["selected_self_teach_plan"]["enabled"] is True
        assert selection["rounds"][1]["self_teach_plan"]["enabled"] is True
        assert selection["attempted_weight_update_count"] > 0
        assert selection["rounds"][1]["weight_update_changed"] is True
        assert (selection["rounds"][1][
            "weight_update_changed_tensor_count"] > 0)
        assert (selection["rounds"][1][
            "weight_update_changed_value_count"] > 0)
        assert selected_model.train_metrics["weight_update_changed"] is False
        assert selected_model.train_metrics["attempted_weight_update_count"] > 0
        assert selected_model.train_metrics["latent_study_reports"] == []
        assert selected_model.train_metrics["latent_discovery_study_reports"] == []
        assert selected_model.train_metrics["latent_study_pool_size"] == 0
        assert selected_model.latent_study_reports == []
    print("multimodal selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", default=None,
                    help="JSONL manifest with text/features and optional target tokens")
    ap.add_argument("--root", default=None,
                    help="root for relative .npy feature paths inside the manifest")
    ap.add_argument("--text-data", action="append", default=None,
                    help=("raw text/code source to convert into targetless "
                          "causal LM windows before multimodal training"))
    ap.add_argument("--text-field", default="text",
                    help="JSON/JSONL field to read when --text-data points at structured data")
    ap.add_argument("--text-window", type=int, default=128,
                    help="tokens per causal LM window built from --text-data")
    ap.add_argument("--text-stride", type=int, default=0,
                    help=("stride for causal LM windows; 0 uses half of "
                          "--text-window for overlap"))
    ap.add_argument("--text-min-tokens", type=int, default=8,
                    help="minimum tokens required for a causal LM window")
    ap.add_argument("--text-eval-frac", type=float, default=0.10,
                    help="fraction of causal LM windows held out for eval")
    ap.add_argument("--text-manifest-out", default=None,
                    help="optional path to save the generated causal LM manifest")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=DEV)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=96, dest="d")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=128, dest="max_len")
    ap.add_argument("--max-vocab", type=int, default=0, dest="max_vocab",
                    help="cap vocabulary to the N most frequent tokens "
                         "(0 = uncapped); rest fall back to <unk>")
    ap.add_argument("--source-balance-w", type=float,
                    default=MULTIMODAL_DEFAULT_SOURCE_BALANCE_W,
                    dest="source_balance_w",
                    help=("smooth source-balanced minibatch sampling from "
                          "manifest metadata; 0 disables"))
    ap.add_argument("--objective-profile", choices=MULTIMODAL_OBJECTIVE_PROFILES,
                    default="language",
                    help=("generic multimodal objective posture; mastery enables "
                          "schema-free concept/self-teach floors; language keeps "
                          "token modeling first"))
    ap.add_argument("--decode-objective", choices=DECODE_OBJECTIVES,
                    default="auto", dest="decode_objective",
                    help=("decoder target construction: target keeps manifest "
                          "targets, causal trains contiguous next-token windows, "
                          "auto detects continuation chunks"))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=100, dest="log_every")
    ap.add_argument("--agreement-w", type=float, default=0.0, dest="agreement_w",
                    help="cross-mode token-distribution agreement loss weight")
    ap.add_argument("--decode-w", type=float, default=1.0, dest="decode_w",
                    help="target-token decoder loss weight; set 0 for latent-only bridge training")
    ap.add_argument("--continuation-repair-w", type=float, default=0.0,
                    dest="continuation_repair_w",
                    help=("loss weight for recovery from the model's own "
                          "free-running continuation tokens"))
    ap.add_argument("--continuation-repair-steps", type=int, default=4,
                    dest="continuation_repair_steps",
                    help="number of self-generated tokens used in repair contexts")
    ap.add_argument("--continuation-repair-prompt-frac", type=float, default=0.5,
                    dest="continuation_repair_prompt_frac",
                    help="fraction of each causal window kept as the gold prompt")
    ap.add_argument("--continuation-repair-temperature", type=float, default=0.0,
                    dest="continuation_repair_temperature",
                    help="0 uses greedy self-rollout for repair")
    ap.add_argument("--continuation-repair-top-k", type=int, default=0,
                    dest="continuation_repair_top_k",
                    help="optional top-k sampling cap for repair self-rollout")
    ap.add_argument("--repetition-unlikelihood-w", type=float, default=0.0,
                    dest="repetition_unlikelihood_w",
                    help=("loss weight that discourages assigning probability "
                          "to recent non-gold repeats"))
    ap.add_argument("--repetition-unlikelihood-window", type=int, default=32,
                    dest="repetition_unlikelihood_window",
                    help="recent context window used for repeat negatives")
    ap.add_argument("--text-checkpoint", default=None, dest="text_checkpoint",
                    help="optional thinking.text checkpoint for text/latent warm start")
    ap.add_argument("--multimodal-checkpoint", default=None,
                    dest="multimodal_checkpoint",
                    help=("optional thinking.multimodal checkpoint for compatible "
                          "warm start / continuation"))
    ap.add_argument("--concept-tokens", type=int, default=4, dest="concept_tokens",
                    help="shared latent fusion tokens")
    ap.add_argument("--fusion-layers", type=int, default=1, dest="fusion_layers",
                    help="transformer layers used by latent prefix fusion")
    ap.add_argument("--latent-concept-slots", type=int, default=0,
                    dest="latent_concept_slots")
    ap.add_argument("--latent-concept-topk", type=int, default=0,
                    dest="latent_concept_topk",
                    help="keep only top-k concept slots per example (structural sparsity)")
    ap.add_argument("--latent-concept-layers", type=int, default=1,
                    dest="latent_concept_layers")
    ap.add_argument("--latent-concept-prefix", action="store_true",
                    dest="latent_concept_prefix")
    ap.add_argument("--latent-concept-w", type=float, default=0.0,
                    dest="latent_concept_w")
    ap.add_argument("--latent-concept-view-dropout", type=float, default=0.1,
                    dest="latent_concept_view_dropout")
    ap.add_argument("--latent-concept-invariance-w", type=float, default=25.0,
                    dest="latent_concept_invariance_w")
    ap.add_argument("--latent-concept-variance-w", type=float, default=25.0,
                    dest="latent_concept_variance_w")
    ap.add_argument("--latent-concept-covariance-w", type=float, default=1.0,
                    dest="latent_concept_covariance_w")
    ap.add_argument("--latent-concept-variance-target", type=float, default=1.0,
                    dest="latent_concept_variance_target")
    ap.add_argument("--latent-concept-factorization-w", type=float, default=0.0,
                    dest="latent_concept_factorization_w")
    ap.add_argument("--latent-concept-factorization-variance", type=float,
                    default=0.05, dest="latent_concept_factorization_variance")
    ap.add_argument("--latent-concept-factorization-margin", type=float,
                    default=0.2, dest="latent_concept_factorization_margin")
    ap.add_argument("--latent-concept-factorization-covariance-w", type=float,
                    default=0.05, dest="latent_concept_factorization_covariance_w")
    ap.add_argument("--latent-concept-fer-w", type=float, default=0.0,
                    dest="latent_concept_fer_w")
    ap.add_argument("--latent-concept-fer-fragmentation-w", type=float, default=1.0,
                    dest="latent_concept_fer_fragmentation_w")
    ap.add_argument("--latent-concept-fer-correlation-w", type=float, default=1.0,
                    dest="latent_concept_fer_correlation_w")
    ap.add_argument("--latent-concept-fer-balance-w", type=float, default=0.1,
                    dest="latent_concept_fer_balance_w")
    ap.add_argument("--latent-concept-fer-probe-n", type=int, default=0,
                    dest="latent_concept_fer_probe_n")
    ap.add_argument("--latent-concept-fer-hard-max", type=int, default=0,
                    dest="latent_concept_fer_hard_max")
    ap.add_argument("--latent-concept-fer-refresh-steps", type=int, default=0,
                    dest="latent_concept_fer_refresh_steps")
    ap.add_argument("--latent-concept-discovery-probe-n", type=int, default=0,
                    dest="latent_concept_discovery_probe_n")
    ap.add_argument("--latent-concept-discovery-hard-max", type=int, default=0,
                    dest="latent_concept_discovery_hard_max")
    ap.add_argument("--latent-concept-discovery-refresh-steps", type=int, default=0,
                    dest="latent_concept_discovery_refresh_steps")
    ap.add_argument("--latent-concept-completion-probe-n", type=int, default=0,
                    dest="latent_concept_completion_probe_n")
    ap.add_argument("--latent-concept-completion-hard-max", type=int, default=0,
                    dest="latent_concept_completion_hard_max")
    ap.add_argument("--latent-concept-completion-refresh-steps", type=int,
                    default=0, dest="latent_concept_completion_refresh_steps")
    ap.add_argument("--latent-concept-memory-w", type=float, default=0.0,
                    dest="latent_concept_memory_w")
    ap.add_argument("--latent-concept-memory-size", type=int, default=0,
                    dest="latent_concept_memory_size")
    ap.add_argument("--latent-concept-memory-temperature", type=float, default=0.1,
                    dest="latent_concept_memory_temperature")
    ap.add_argument("--latent-concept-memory-momentum", type=float, default=0.95,
                    dest="latent_concept_memory_momentum")
    ap.add_argument("--latent-concept-memory-balance-w", type=float, default=0.01,
                    dest="latent_concept_memory_balance_w")
    ap.add_argument("--latent-concept-consolidation-w", type=float, default=0.0,
                    dest="latent_concept_consolidation_w")
    ap.add_argument("--latent-concept-consolidation-temperature", type=float,
                    default=0.1, dest="latent_concept_consolidation_temperature")
    ap.add_argument("--latent-concept-consolidation-balance-w", type=float,
                    default=0.01, dest="latent_concept_consolidation_balance_w")
    ap.add_argument("--latent-concept-consolidation-anchor-w", type=float,
                    default=1.0, dest="latent_concept_consolidation_anchor_w")
    ap.add_argument("--latent-concept-consolidation-fer-w", type=float,
                    default=0.0, dest="latent_concept_consolidation_fer_w")
    ap.add_argument("--latent-concept-discovery-w", type=float, default=0.0,
                    dest="latent_concept_discovery_w")
    ap.add_argument("--latent-concept-discovery-curiosity-w", type=float,
                    default=1.0, dest="latent_concept_discovery_curiosity_w")
    ap.add_argument("--latent-concept-discovery-graph-w", type=float,
                    default=1.0, dest="latent_concept_discovery_graph_w")
    ap.add_argument("--latent-concept-discovery-cycle-w", type=float,
                    default=1.0, dest="latent_concept_discovery_cycle_w")
    ap.add_argument("--latent-concept-discovery-bridge-w", type=float,
                    default=1.0, dest="latent_concept_discovery_bridge_w")
    ap.add_argument("--latent-concept-discovery-fer-w", type=float,
                    default=0.0, dest="latent_concept_discovery_fer_w")
    ap.add_argument("--latent-concept-reanalysis-w", type=float, default=0.0,
                    dest="latent_concept_reanalysis_w")
    ap.add_argument("--latent-concept-reanalysis-graph-w", type=float,
                    default=1.0, dest="latent_concept_reanalysis_graph_w")
    ap.add_argument("--latent-concept-reanalysis-cycle-w", type=float,
                    default=0.5, dest="latent_concept_reanalysis_cycle_w")
    ap.add_argument("--latent-concept-reanalysis-bridge-w", type=float,
                    default=0.5, dest="latent_concept_reanalysis_bridge_w")
    ap.add_argument("--latent-concept-reanalysis-fer-w", type=float,
                    default=0.0, dest="latent_concept_reanalysis_fer_w")
    ap.add_argument("--latent-concept-reanalysis-cycle-consistency-w",
                    type=float, default=0.5,
                    dest="latent_concept_reanalysis_cycle_consistency_w")
    ap.add_argument("--latent-concept-gap-w", type=float, default=0.0,
                    dest="latent_concept_gap_w")
    ap.add_argument("--latent-concept-gap-temperature", type=float, default=0.1,
                    dest="latent_concept_gap_temperature")
    ap.add_argument("--latent-concept-gap-self-loop-w", type=float, default=0.0,
                    dest="latent_concept_gap_self_loop_w")
    ap.add_argument("--latent-concept-gap-transitive-steps", type=int, default=2,
                    dest="latent_concept_gap_transitive_steps")
    ap.add_argument("--latent-concept-gap-transitive-w", type=float, default=0.1,
                    dest="latent_concept_gap_transitive_w")
    ap.add_argument("--latent-concept-gap-target-power", type=float, default=1.0,
                    dest="latent_concept_gap_target_power")
    ap.add_argument("--latent-concept-association-w", type=float, default=0.0,
                    dest="latent_concept_association_w")
    ap.add_argument("--latent-concept-association-temperature", type=float,
                    default=0.1, dest="latent_concept_association_temperature")
    ap.add_argument("--latent-concept-association-decay", type=float, default=0.99,
                    dest="latent_concept_association_decay")
    ap.add_argument("--latent-concept-association-target-power", type=float,
                    default=1.0, dest="latent_concept_association_target_power")
    ap.add_argument("--latent-concept-association-self-loop-w", type=float,
                    default=0.05, dest="latent_concept_association_self_loop_w")
    ap.add_argument("--latent-concept-association-transitive-steps", type=int,
                    default=2, dest="latent_concept_association_transitive_steps")
    ap.add_argument("--latent-concept-association-transitive-w", type=float,
                    default=0.1, dest="latent_concept_association_transitive_w")
    ap.add_argument("--latent-concept-composition-w", type=float, default=0.0,
                    dest="latent_concept_composition_w")
    ap.add_argument("--latent-concept-composition-temperature", type=float,
                    default=0.1, dest="latent_concept_composition_temperature")
    ap.add_argument("--latent-concept-composition-self-loop-w", type=float,
                    default=0.0, dest="latent_concept_composition_self_loop_w")
    ap.add_argument("--latent-concept-composition-transitive-steps", type=int,
                    default=2, dest="latent_concept_composition_transitive_steps")
    ap.add_argument("--latent-concept-composition-transitive-w", type=float,
                    default=0.1, dest="latent_concept_composition_transitive_w")
    ap.add_argument("--latent-concept-composition-margin", type=float, default=0.0,
                    dest="latent_concept_composition_margin")
    ap.add_argument("--latent-concept-graph-predict-w", type=float, default=0.0,
                    dest="latent_concept_graph_predict_w")
    ap.add_argument("--latent-concept-graph-predict-temperature", type=float,
                    default=0.1, dest="latent_concept_graph_predict_temperature")
    ap.add_argument("--latent-concept-graph-predict-self-loop-w", type=float,
                    default=0.05, dest="latent_concept_graph_predict_self_loop_w")
    ap.add_argument("--latent-concept-graph-predict-transitive-steps", type=int,
                    default=2, dest="latent_concept_graph_predict_transitive_steps")
    ap.add_argument("--latent-concept-graph-predict-transitive-w", type=float,
                    default=0.1, dest="latent_concept_graph_predict_transitive_w")
    ap.add_argument("--latent-concept-graph-predict-target-power", type=float,
                    default=1.0, dest="latent_concept_graph_predict_target_power")
    ap.add_argument("--latent-concept-bridge-w", type=float, default=0.0,
                    dest="latent_concept_bridge_w",
                    help="weight for label-free weak-connection bridge closure")
    ap.add_argument("--latent-concept-completion-w", type=float, default=0.0,
                    dest="latent_concept_completion_w",
                    help=("weight for partial modality views predicting the full "
                          "shared latent concept state"))
    ap.add_argument("--latent-concept-completion-temperature", type=float,
                    default=0.1, dest="latent_concept_completion_temperature")
    ap.add_argument("--latent-concept-sequence-w", type=float, default=0.0,
                    dest="latent_concept_sequence_w",
                    help="weight for adjacent-record latent concept prediction")
    ap.add_argument("--latent-concept-sequence-batch", type=int, default=0,
                    dest="latent_concept_sequence_batch")
    ap.add_argument("--latent-concept-sequence-temperature", type=float,
                    default=0.1, dest="latent_concept_sequence_temperature")
    ap.add_argument("--latent-concept-neighborhood-w", type=float, default=0.0,
                    dest="latent_concept_neighborhood_w")
    ap.add_argument("--latent-concept-neighborhood-temperature", type=float,
                    default=0.1, dest="latent_concept_neighborhood_temperature")
    ap.add_argument("--latent-concept-neighborhood-margin", type=float, default=0.0,
                    dest="latent_concept_neighborhood_margin")
    ap.add_argument("--latent-concept-transition-w", type=float, default=0.0,
                    dest="latent_concept_transition_w")
    ap.add_argument("--latent-concept-transition-temperature", type=float,
                    default=0.1, dest="latent_concept_transition_temperature")
    ap.add_argument("--latent-concept-transition-margin", type=float, default=0.0,
                    dest="latent_concept_transition_margin")
    ap.add_argument("--latent-concept-cluster-w", type=float, default=0.0,
                    dest="latent_concept_cluster_w")
    ap.add_argument("--latent-concept-cluster-temperature", type=float, default=0.1,
                    dest="latent_concept_cluster_temperature")
    ap.add_argument("--latent-concept-cluster-margin", type=float, default=0.0,
                    dest="latent_concept_cluster_margin")
    ap.add_argument("--latent-concept-cluster-min-size", type=int, default=2,
                    dest="latent_concept_cluster_min_size")
    ap.add_argument("--self-teach-w", type=float, default=0.0,
                    dest="self_teach_w",
                    help=("budget for reallocating multimodal objective weight "
                          "from the model's own eval deficits"))
    ap.add_argument("--self-teach-history-prior-w", type=float, default=0.5,
                    dest="self_teach_history_prior_w",
                    help=("blend weight for text-checkpoint reading history "
                          "inside multimodal self-teach allocation"))
    ap.add_argument("--representation-probe-n", type=int, default=0,
                    dest="representation_probe_n",
                    help=("optional before/after probe count for label-free "
                          "representation organization progress"))
    ap.add_argument("--text-transfer-probe-n", type=int,
                    default=DEFAULT_TEXT_TRANSFER_PROBE_N,
                    dest="text_transfer_probe_n",
                    help=("probe current manifest before/after text-checkpoint "
                          "import; positive values enable target-aware transfer "
                          "calibration"))
    ap.add_argument("--text-transfer-score-min-delta", type=float,
                    default=DEFAULT_TEXT_TRANSFER_SCORE_MIN_DELTA,
                    dest="text_transfer_score_min_delta",
                    help="minimum target score gain required to accept text transfer")
    ap.add_argument("--text-transfer-insight-accept-w",
                    "--text-transfer-insight-w", type=float,
                    default=DEFAULT_TEXT_TRANSFER_INSIGHT_ACCEPT_W,
                    dest="text_transfer_insight_accept_w",
                    help=("accept useful representation-organization gains even "
                          "when surface score is close"))
    ap.add_argument("--text-transfer-insight-min-delta", type=float, default=0.0,
                    dest="text_transfer_insight_min_delta",
                    help="minimum organization gain for insight-based transfer")
    ap.add_argument("--text-transfer-gate", action=argparse.BooleanOptionalAction,
                    default=True, dest="text_transfer_gate",
                    help="roll back text checkpoint import when target probe rejects it")
    ap.add_argument("--view-tokens", type=int, default=4, dest="view_tokens",
                    help="prefix tokens allocated to each named manifest feature view")
    ap.add_argument("--txt-tokens", type=int, default=8, dest="txt_tokens")
    ap.add_argument("--trunk-arch", default="mlp", choices=TRUNK_ARCHES, dest="trunk_arch")
    ap.add_argument("--trunk-width", type=int, default=128, dest="trunk_width")
    ap.add_argument("--trunk-depth", type=int, default=1, dest="trunk_depth")
    ap.add_argument("--text-layers", type=int, default=1, dest="text_layers")
    ap.add_argument("--text-arch", choices=TEXT_TRUNK_ARCHES, default="transformer",
                    dest="text_arch")
    ap.add_argument("--modality-dropout", type=float, default=0.0, dest="modality_dropout")
    ap.add_argument("--eval-n", type=int, default=200, dest="eval_n")
    ap.add_argument("--generation-n", type=int, default=0, dest="generation_n",
                    help=("number of eval records to probe with free "
                          "autoregressive continuation"))
    ap.add_argument("--generation-prompt-tokens", type=int, default=16,
                    dest="generation_prompt_tokens",
                    help="number of gold tokens used as the generation prompt")
    ap.add_argument("--generation-max-new-tokens", type=int, default=32,
                    dest="generation_max_new_tokens",
                    help="maximum continuation tokens to sample per prompt")
    ap.add_argument("--generation-temperature", type=float, default=0.0,
                    dest="generation_temperature",
                    help="0 uses greedy continuation; positive values sample")
    ap.add_argument("--generation-top-k", type=int, default=0,
                    dest="generation_top_k",
                    help="optional top-k sampling cap for generation")
    ap.add_argument("--generate-prompt", action="append", default=None,
                    dest="generate_prompts",
                    help="free-form prompt to continue after training")
    ap.add_argument("--select-best", action=argparse.BooleanOptionalAction,
                    default=False, dest="select_best",
                    help="reload the best self-scored multimodal checkpoint")
    ap.add_argument("--selection-rounds", type=int, default=1,
                    dest="selection_rounds")
    ap.add_argument("--selection-score-metric", choices=MULTIMODAL_SCORE_METRICS,
                    default="mastery", dest="selection_score_metric")
    ap.add_argument("--selection-score-margin-w", type=float, default=0.1,
                    dest="selection_score_margin_w")
    ap.add_argument("--selection-score-min-delta", type=float, default=0.0,
                    dest="selection_score_min_delta")
    ap.add_argument("--selection-score-patience", type=int, default=0,
                    dest="selection_score_patience")
    ap.add_argument("--selection-signal-regression-tolerance", type=float,
                    default=MULTIMODAL_DEFAULT_SIGNAL_REGRESSION_TOLERANCE,
                    dest="selection_signal_regression_tolerance",
                    help=("maximum tolerated active-signal regression when "
                          "accepting a multimodal selection round"))
    ap.add_argument("--selection-eval-n", type=int, default=200,
                    dest="selection_eval_n")
    ap.add_argument("--selection-generation-n", type=int, default=0,
                    dest="selection_generation_n",
                    help=("free-generation eval samples used inside selection/"
                          "self-teach scoring; language profile raises this"))
    ap.add_argument("--selection-generation-prompt-tokens", type=int, default=16,
                    dest="selection_generation_prompt_tokens")
    ap.add_argument("--selection-generation-max-new-tokens", type=int, default=32,
                    dest="selection_generation_max_new_tokens")
    ap.add_argument("--selection-generation-temperature", type=float, default=0.0,
                    dest="selection_generation_temperature")
    ap.add_argument("--selection-generation-top-k", type=int, default=0,
                    dest="selection_generation_top_k")
    ap.add_argument("--selection-insight-accept-w", "--selection-insight-w",
                    type=float, default=0.25,
                    dest="selection_insight_accept_w")
    ap.add_argument("--selection-insight-min-delta", type=float, default=0.0,
                    dest="selection_insight_min_delta")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="runs/multimodal.json")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if args.manifest and args.text_data:
        ap.error("--manifest and --text-data are mutually exclusive")
    if not args.manifest and not args.text_data:
        ap.error("--manifest or --text-data is required")
    if args.text_window <= 1:
        ap.error("--text-window must be greater than one")
    if args.text_stride < 0:
        ap.error("--text-stride must be non-negative")
    if args.text_min_tokens <= 1:
        ap.error("--text-min-tokens must be greater than one")
    if args.text_eval_frac < 0.0 or args.text_eval_frac >= 1.0:
        ap.error("--text-eval-frac must be in [0, 1)")
    if args.eval_n < 0:
        ap.error("--eval-n must be non-negative")
    if args.generation_n < 0:
        ap.error("--generation-n must be non-negative")
    if args.generation_prompt_tokens <= 0:
        ap.error("--generation-prompt-tokens must be positive")
    if args.generation_max_new_tokens < 0:
        ap.error("--generation-max-new-tokens must be non-negative")
    if args.generation_temperature < 0.0:
        ap.error("--generation-temperature must be non-negative")
    if args.generation_top_k < 0:
        ap.error("--generation-top-k must be non-negative")
    if args.selection_generation_n < 0:
        ap.error("--selection-generation-n must be non-negative")
    if args.selection_generation_prompt_tokens <= 0:
        ap.error("--selection-generation-prompt-tokens must be positive")
    if args.selection_generation_max_new_tokens < 0:
        ap.error("--selection-generation-max-new-tokens must be non-negative")
    if args.selection_generation_temperature < 0.0:
        ap.error("--selection-generation-temperature must be non-negative")
    if args.selection_generation_top_k < 0:
        ap.error("--selection-generation-top-k must be non-negative")
    positive = {
        "--steps": args.steps, "--batch": args.batch, "--dim": args.d,
        "--layers": args.layers, "--heads": args.heads, "--max-len": args.max_len,
        "--log-every": args.log_every, "--view-tokens": args.view_tokens,
        "--txt-tokens": args.txt_tokens,
        "--trunk-width": args.trunk_width, "--trunk-depth": args.trunk_depth,
        "--text-layers": args.text_layers, "--concept-tokens": args.concept_tokens,
        "--fusion-layers": args.fusion_layers,
    }
    for name, value in positive.items():
        if value <= 0:
            ap.error(f"{name} must be positive")
    if args.lr <= 0.0:
        ap.error("--lr must be positive")
    if args.source_balance_w < 0.0:
        ap.error("--source-balance-w must be non-negative")
    if args.d % args.heads != 0:
        ap.error("--dim must be divisible by --heads")
    if (args.d // args.heads) % 2 != 0:
        ap.error("--dim / --heads must be even for rope attention")
    if args.text_arch in ("relational", "abstractor") and args.heads % 2 != 0:
        ap.error("--text-arch relational/abstractor require an even --heads value")
    if args.modality_dropout < 0.0 or args.modality_dropout > 1.0:
        ap.error("--modality-dropout must be in [0, 1]")
    if args.latent_concept_slots < 0:
        ap.error("--latent-concept-slots must be non-negative")
    if args.latent_concept_layers <= 0:
        ap.error("--latent-concept-layers must be positive")
    latent_weights = [
        args.latent_concept_w, args.latent_concept_factorization_w,
        args.latent_concept_fer_w, args.latent_concept_memory_w,
        args.latent_concept_consolidation_w,
        args.latent_concept_discovery_w,
        args.latent_concept_reanalysis_w,
        args.latent_concept_gap_w,
        args.latent_concept_association_w,
        args.latent_concept_composition_w, args.latent_concept_graph_predict_w,
        args.latent_concept_bridge_w,
        args.latent_concept_completion_w,
        args.latent_concept_sequence_w,
        args.latent_concept_neighborhood_w, args.latent_concept_transition_w,
        args.latent_concept_cluster_w,
    ]
    if any(w < 0.0 for w in latent_weights) or args.agreement_w < 0.0 or args.decode_w < 0.0:
        ap.error("decoder/agreement/latent loss weights must be non-negative")
    if args.continuation_repair_w < 0.0:
        ap.error("--continuation-repair-w must be non-negative")
    if args.continuation_repair_steps < 0:
        ap.error("--continuation-repair-steps must be non-negative")
    if (args.continuation_repair_prompt_frac <= 0.0
            or args.continuation_repair_prompt_frac >= 1.0):
        ap.error("--continuation-repair-prompt-frac must be in (0, 1)")
    if args.continuation_repair_temperature < 0.0:
        ap.error("--continuation-repair-temperature must be non-negative")
    if args.continuation_repair_top_k < 0:
        ap.error("--continuation-repair-top-k must be non-negative")
    if args.repetition_unlikelihood_w < 0.0:
        ap.error("--repetition-unlikelihood-w must be non-negative")
    if args.repetition_unlikelihood_window < 0:
        ap.error("--repetition-unlikelihood-window must be non-negative")
    if args.self_teach_w < 0.0:
        ap.error("--self-teach-w must be non-negative")
    if args.self_teach_history_prior_w < 0.0:
        ap.error("--self-teach-history-prior-w must be non-negative")
    if args.representation_probe_n < 0:
        ap.error("--representation-probe-n must be non-negative")
    if args.text_transfer_probe_n < 0:
        ap.error("--text-transfer-probe-n must be non-negative")
    if args.text_transfer_score_min_delta < 0.0:
        ap.error("--text-transfer-score-min-delta must be non-negative")
    if args.text_transfer_insight_accept_w < 0.0:
        ap.error("--text-transfer-insight-accept-w must be non-negative")
    if args.text_transfer_insight_min_delta < 0.0:
        ap.error("--text-transfer-insight-min-delta must be non-negative")
    if args.latent_concept_view_dropout < 0.0 or args.latent_concept_view_dropout >= 1.0:
        ap.error("--latent-concept-view-dropout must be in [0, 1)")
    if (args.latent_concept_fer_fragmentation_w < 0.0
            or args.latent_concept_fer_correlation_w < 0.0
            or args.latent_concept_fer_balance_w < 0.0):
        ap.error("latent FER component weights must be non-negative")
    if args.latent_concept_fer_probe_n < 0:
        ap.error("--latent-concept-fer-probe-n must be non-negative")
    if args.latent_concept_fer_hard_max < 0:
        ap.error("--latent-concept-fer-hard-max must be non-negative")
    if args.latent_concept_fer_refresh_steps < 0:
        ap.error("--latent-concept-fer-refresh-steps must be non-negative")
    if args.latent_concept_discovery_probe_n < 0:
        ap.error("--latent-concept-discovery-probe-n must be non-negative")
    if args.latent_concept_discovery_hard_max < 0:
        ap.error("--latent-concept-discovery-hard-max must be non-negative")
    if args.latent_concept_discovery_refresh_steps < 0:
        ap.error("--latent-concept-discovery-refresh-steps must be non-negative")
    if args.latent_concept_completion_probe_n < 0:
        ap.error("--latent-concept-completion-probe-n must be non-negative")
    if args.latent_concept_completion_hard_max < 0:
        ap.error("--latent-concept-completion-hard-max must be non-negative")
    if args.latent_concept_completion_refresh_steps < 0:
        ap.error("--latent-concept-completion-refresh-steps must be non-negative")
    latent_options_need_slots = (
        any(w > 0.0 for w in latent_weights)
        or args.latent_concept_memory_size > 0
        or args.latent_concept_prefix
        or args.latent_concept_fer_hard_max > 0
        or args.latent_concept_discovery_hard_max > 0
        or args.latent_concept_completion_hard_max > 0)
    if latent_options_need_slots and args.latent_concept_slots <= 0:
        ap.error("latent concept options require --latent-concept-slots > 0")
    if args.latent_concept_memory_temperature <= 0.0:
        ap.error("--latent-concept-memory-temperature must be positive")
    if args.latent_concept_memory_momentum < 0.0 or args.latent_concept_memory_momentum >= 1.0:
        ap.error("--latent-concept-memory-momentum must be in [0, 1)")
    if args.latent_concept_consolidation_temperature <= 0.0:
        ap.error("--latent-concept-consolidation-temperature must be positive")
    if (args.latent_concept_consolidation_balance_w < 0.0
            or args.latent_concept_consolidation_anchor_w < 0.0
            or args.latent_concept_consolidation_fer_w < 0.0):
        ap.error("latent concept consolidation component weights must be non-negative")
    if (args.latent_concept_consolidation_w > 0.0
            and args.latent_concept_memory_size <= 0):
        ap.error("--latent-concept-consolidation-w requires --latent-concept-memory-size > 0")
    if (args.latent_concept_discovery_curiosity_w < 0.0
            or args.latent_concept_discovery_graph_w < 0.0
            or args.latent_concept_discovery_cycle_w < 0.0
            or args.latent_concept_discovery_bridge_w < 0.0
            or args.latent_concept_discovery_fer_w < 0.0):
        ap.error("latent concept discovery component weights must be non-negative")
    if (args.latent_concept_discovery_w > 0.0
            and args.latent_concept_memory_size <= 0):
        ap.error("--latent-concept-discovery-w requires --latent-concept-memory-size > 0")
    if (args.latent_concept_reanalysis_graph_w < 0.0
            or args.latent_concept_reanalysis_cycle_w < 0.0
            or args.latent_concept_reanalysis_bridge_w < 0.0
            or args.latent_concept_reanalysis_fer_w < 0.0
            or args.latent_concept_reanalysis_cycle_consistency_w < 0.0):
        ap.error("latent concept reanalysis component weights must be non-negative")
    if (args.latent_concept_reanalysis_w > 0.0
            and args.latent_concept_memory_size <= 0):
        ap.error("--latent-concept-reanalysis-w requires --latent-concept-memory-size > 0")
    if (args.latent_concept_gap_w > 0.0
            and args.latent_concept_memory_size <= 0):
        ap.error("--latent-concept-gap-w requires --latent-concept-memory-size > 0")
    if args.latent_concept_gap_temperature <= 0.0:
        ap.error("--latent-concept-gap-temperature must be positive")
    if (args.latent_concept_gap_self_loop_w < 0.0
            or args.latent_concept_gap_transitive_w < 0.0):
        ap.error("latent concept gap graph weights must be non-negative")
    if args.latent_concept_gap_transitive_steps <= 0:
        ap.error("--latent-concept-gap-transitive-steps must be positive")
    if args.latent_concept_gap_target_power <= 0.0:
        ap.error("--latent-concept-gap-target-power must be positive")
    if args.latent_concept_association_temperature <= 0.0:
        ap.error("--latent-concept-association-temperature must be positive")
    if args.latent_concept_composition_temperature <= 0.0:
        ap.error("--latent-concept-composition-temperature must be positive")
    if args.latent_concept_graph_predict_temperature <= 0.0:
        ap.error("--latent-concept-graph-predict-temperature must be positive")
    if args.latent_concept_completion_temperature <= 0.0:
        ap.error("--latent-concept-completion-temperature must be positive")
    if args.latent_concept_sequence_batch < 0 or args.latent_concept_sequence_batch == 1:
        ap.error("--latent-concept-sequence-batch must be 0 or at least 2")
    if args.latent_concept_sequence_temperature <= 0.0:
        ap.error("--latent-concept-sequence-temperature must be positive")
    if args.latent_concept_neighborhood_temperature <= 0.0:
        ap.error("--latent-concept-neighborhood-temperature must be positive")
    if args.latent_concept_transition_temperature <= 0.0:
        ap.error("--latent-concept-transition-temperature must be positive")
    if args.latent_concept_cluster_temperature <= 0.0:
        ap.error("--latent-concept-cluster-temperature must be positive")
    if args.selection_rounds <= 0:
        ap.error("--selection-rounds must be positive")
    if args.selection_score_min_delta < 0.0:
        ap.error("--selection-score-min-delta must be non-negative")
    if args.selection_score_patience < 0:
        ap.error("--selection-score-patience must be non-negative")
    if args.selection_signal_regression_tolerance < 0.0:
        ap.error("--selection-signal-regression-tolerance must be non-negative")
    if args.selection_eval_n < 0:
        ap.error("--selection-eval-n must be non-negative")
    if args.selection_insight_accept_w < 0.0:
        ap.error("--selection-insight-accept-w must be non-negative")
    if args.selection_insight_min_delta < 0.0:
        ap.error("--selection-insight-min-delta must be non-negative")
    generated_manifest_tmp = None
    generated_manifest_report = None
    manifest = args.manifest
    root = args.root
    if args.text_data:
        if args.text_manifest_out:
            manifest = args.text_manifest_out
        elif args.out:
            stem, _ext = os.path.splitext(args.out)
            manifest = stem + ".manifest.jsonl"
        else:
            generated_manifest_tmp = tempfile.TemporaryDirectory()
            manifest = os.path.join(
                generated_manifest_tmp.name, "causal_lm_manifest.jsonl")
        generated_manifest_report = build_text_causal_lm_manifest(
            args.text_data, manifest, text_field=args.text_field,
            max_tokens=args.text_window, min_tokens=args.text_min_tokens,
            stride=args.text_stride, eval_frac=args.text_eval_frac,
            seed=args.seed)
        root = None
    report = run(
        manifest, root=root, steps=args.steps, seed=args.seed,
        device=args.device, eval_n=args.eval_n, checkpoint=args.checkpoint,
        out=args.out, generation_n=args.generation_n,
        generation_prompt_tokens=args.generation_prompt_tokens,
        generation_max_new_tokens=args.generation_max_new_tokens,
        generation_temperature=args.generation_temperature,
        generation_top_k=args.generation_top_k,
        generate_prompts=args.generate_prompts,
        batch=args.batch, d=args.d, lr=args.lr, layers=args.layers,
        heads=args.heads, max_len=args.max_len, max_vocab=args.max_vocab,
        source_balance_w=args.source_balance_w,
        objective_profile=args.objective_profile,
        decode_objective=args.decode_objective,
        view_tokens=args.view_tokens,
        txt_tokens=args.txt_tokens,
        trunk_arch=args.trunk_arch, trunk_width=args.trunk_width,
        trunk_depth=args.trunk_depth, text_layers=args.text_layers,
        text_arch=args.text_arch, modality_dropout=args.modality_dropout,
        decode_w=args.decode_w, agreement_w=args.agreement_w,
        continuation_repair_w=args.continuation_repair_w,
        continuation_repair_steps=args.continuation_repair_steps,
        continuation_repair_prompt_frac=args.continuation_repair_prompt_frac,
        continuation_repair_temperature=args.continuation_repair_temperature,
        continuation_repair_top_k=args.continuation_repair_top_k,
        repetition_unlikelihood_w=args.repetition_unlikelihood_w,
        repetition_unlikelihood_window=args.repetition_unlikelihood_window,
        text_checkpoint=args.text_checkpoint,
        multimodal_checkpoint=args.multimodal_checkpoint,
        concept_tokens=args.concept_tokens, fusion_layers=args.fusion_layers,
        latent_concept_slots=args.latent_concept_slots,
        latent_concept_layers=args.latent_concept_layers,
        latent_concept_prefix=args.latent_concept_prefix,
        latent_concept_w=args.latent_concept_w,
        latent_concept_view_dropout=args.latent_concept_view_dropout,
        latent_concept_invariance_w=args.latent_concept_invariance_w,
        latent_concept_variance_w=args.latent_concept_variance_w,
        latent_concept_covariance_w=args.latent_concept_covariance_w,
        latent_concept_variance_target=args.latent_concept_variance_target,
        latent_concept_factorization_w=args.latent_concept_factorization_w,
        latent_concept_factorization_variance=(
            args.latent_concept_factorization_variance),
        latent_concept_factorization_margin=(
            args.latent_concept_factorization_margin),
        latent_concept_factorization_covariance_w=(
            args.latent_concept_factorization_covariance_w),
        latent_concept_fer_w=args.latent_concept_fer_w,
        latent_concept_fer_fragmentation_w=(
            args.latent_concept_fer_fragmentation_w),
        latent_concept_fer_correlation_w=args.latent_concept_fer_correlation_w,
        latent_concept_fer_balance_w=args.latent_concept_fer_balance_w,
        latent_concept_fer_probe_n=args.latent_concept_fer_probe_n,
        latent_concept_fer_hard_max=args.latent_concept_fer_hard_max,
        latent_concept_fer_refresh_steps=args.latent_concept_fer_refresh_steps,
        latent_concept_discovery_probe_n=args.latent_concept_discovery_probe_n,
        latent_concept_discovery_hard_max=args.latent_concept_discovery_hard_max,
        latent_concept_discovery_refresh_steps=(
            args.latent_concept_discovery_refresh_steps),
        latent_concept_completion_probe_n=args.latent_concept_completion_probe_n,
        latent_concept_completion_hard_max=args.latent_concept_completion_hard_max,
        latent_concept_completion_refresh_steps=(
            args.latent_concept_completion_refresh_steps),
        latent_concept_memory_w=args.latent_concept_memory_w,
        latent_concept_memory_size=args.latent_concept_memory_size,
        latent_concept_topk=args.latent_concept_topk,
        latent_concept_memory_temperature=args.latent_concept_memory_temperature,
        latent_concept_memory_momentum=args.latent_concept_memory_momentum,
        latent_concept_memory_balance_w=args.latent_concept_memory_balance_w,
        latent_concept_consolidation_w=args.latent_concept_consolidation_w,
        latent_concept_consolidation_temperature=(
            args.latent_concept_consolidation_temperature),
        latent_concept_consolidation_balance_w=(
            args.latent_concept_consolidation_balance_w),
        latent_concept_consolidation_anchor_w=(
            args.latent_concept_consolidation_anchor_w),
        latent_concept_consolidation_fer_w=args.latent_concept_consolidation_fer_w,
        latent_concept_discovery_w=args.latent_concept_discovery_w,
        latent_concept_discovery_curiosity_w=(
            args.latent_concept_discovery_curiosity_w),
        latent_concept_discovery_graph_w=args.latent_concept_discovery_graph_w,
        latent_concept_discovery_cycle_w=args.latent_concept_discovery_cycle_w,
        latent_concept_discovery_bridge_w=args.latent_concept_discovery_bridge_w,
        latent_concept_discovery_fer_w=args.latent_concept_discovery_fer_w,
        latent_concept_reanalysis_w=args.latent_concept_reanalysis_w,
        latent_concept_reanalysis_graph_w=args.latent_concept_reanalysis_graph_w,
        latent_concept_reanalysis_cycle_w=args.latent_concept_reanalysis_cycle_w,
        latent_concept_reanalysis_bridge_w=(
            args.latent_concept_reanalysis_bridge_w),
        latent_concept_reanalysis_fer_w=args.latent_concept_reanalysis_fer_w,
        latent_concept_reanalysis_cycle_consistency_w=(
            args.latent_concept_reanalysis_cycle_consistency_w),
        latent_concept_gap_w=args.latent_concept_gap_w,
        latent_concept_gap_temperature=args.latent_concept_gap_temperature,
        latent_concept_gap_self_loop_w=args.latent_concept_gap_self_loop_w,
        latent_concept_gap_transitive_steps=(
            args.latent_concept_gap_transitive_steps),
        latent_concept_gap_transitive_w=args.latent_concept_gap_transitive_w,
        latent_concept_gap_target_power=args.latent_concept_gap_target_power,
        latent_concept_association_w=args.latent_concept_association_w,
        latent_concept_association_temperature=(
            args.latent_concept_association_temperature),
        latent_concept_association_decay=args.latent_concept_association_decay,
        latent_concept_association_target_power=(
            args.latent_concept_association_target_power),
        latent_concept_association_self_loop_w=(
            args.latent_concept_association_self_loop_w),
        latent_concept_association_transitive_steps=(
            args.latent_concept_association_transitive_steps),
        latent_concept_association_transitive_w=(
            args.latent_concept_association_transitive_w),
        latent_concept_composition_w=args.latent_concept_composition_w,
        latent_concept_composition_temperature=(
            args.latent_concept_composition_temperature),
        latent_concept_composition_self_loop_w=(
            args.latent_concept_composition_self_loop_w),
        latent_concept_composition_transitive_steps=(
            args.latent_concept_composition_transitive_steps),
        latent_concept_composition_transitive_w=(
            args.latent_concept_composition_transitive_w),
        latent_concept_composition_margin=args.latent_concept_composition_margin,
        latent_concept_graph_predict_w=args.latent_concept_graph_predict_w,
        latent_concept_graph_predict_temperature=(
            args.latent_concept_graph_predict_temperature),
        latent_concept_graph_predict_self_loop_w=(
            args.latent_concept_graph_predict_self_loop_w),
        latent_concept_graph_predict_transitive_steps=(
            args.latent_concept_graph_predict_transitive_steps),
        latent_concept_graph_predict_transitive_w=(
            args.latent_concept_graph_predict_transitive_w),
        latent_concept_graph_predict_target_power=(
            args.latent_concept_graph_predict_target_power),
        latent_concept_bridge_w=args.latent_concept_bridge_w,
        latent_concept_completion_w=args.latent_concept_completion_w,
        latent_concept_completion_temperature=(
            args.latent_concept_completion_temperature),
        latent_concept_sequence_w=args.latent_concept_sequence_w,
        latent_concept_sequence_batch=args.latent_concept_sequence_batch,
        latent_concept_sequence_temperature=(
            args.latent_concept_sequence_temperature),
        latent_concept_neighborhood_w=args.latent_concept_neighborhood_w,
        latent_concept_neighborhood_temperature=(
            args.latent_concept_neighborhood_temperature),
        latent_concept_neighborhood_margin=args.latent_concept_neighborhood_margin,
        latent_concept_transition_w=args.latent_concept_transition_w,
        latent_concept_transition_temperature=(
            args.latent_concept_transition_temperature),
        latent_concept_transition_margin=args.latent_concept_transition_margin,
        latent_concept_cluster_w=args.latent_concept_cluster_w,
        latent_concept_cluster_temperature=args.latent_concept_cluster_temperature,
        latent_concept_cluster_margin=args.latent_concept_cluster_margin,
        latent_concept_cluster_min_size=args.latent_concept_cluster_min_size,
        representation_probe_n=args.representation_probe_n,
        text_transfer_probe_n=args.text_transfer_probe_n,
        text_transfer_score_min_delta=args.text_transfer_score_min_delta,
        text_transfer_insight_accept_w=args.text_transfer_insight_accept_w,
        text_transfer_insight_min_delta=args.text_transfer_insight_min_delta,
        text_transfer_gate=args.text_transfer_gate,
        self_teach_w=args.self_teach_w,
        self_teach_history_prior_w=args.self_teach_history_prior_w,
        select_best=args.select_best,
        selection_rounds=args.selection_rounds,
        selection_score_metric=args.selection_score_metric,
        selection_score_margin_w=args.selection_score_margin_w,
        selection_score_min_delta=args.selection_score_min_delta,
        selection_score_patience=args.selection_score_patience,
        selection_signal_regression_tolerance=(
            args.selection_signal_regression_tolerance),
        selection_eval_n=args.selection_eval_n,
        selection_generation_n=args.selection_generation_n,
        selection_generation_prompt_tokens=args.selection_generation_prompt_tokens,
        selection_generation_max_new_tokens=args.selection_generation_max_new_tokens,
        selection_generation_temperature=args.selection_generation_temperature,
        selection_generation_top_k=args.selection_generation_top_k,
        selection_insight_accept_w=args.selection_insight_accept_w,
        selection_insight_min_delta=args.selection_insight_min_delta,
        log_every=args.log_every)
    if generated_manifest_report is not None:
        report["generated_text_manifest"] = generated_manifest_report
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1)
    if generated_manifest_tmp is not None:
        generated_manifest_tmp.cleanup()
    return report


if __name__ == "__main__":
    main()
