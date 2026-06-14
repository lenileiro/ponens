"""Generic multimodal prefix bridge.

This module intentionally has no built-in sensory oracle or fixed modality schema.
It trains from a JSONL manifest where each record supplies optional named feature
views, text tokens, and an optional target token sequence.  The bridge learns to fuse those
views into continuous prefixes for one ScratchpadLM decoder.

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
from .selection import concept_round_selection_decision
from .trace import Vocab

DEV = get_device()
MODES = ("full", "sensor_only", "text_only")
TRUNK_ARCHES = ("mlp", "residual")
TEXT_TRUNK_ARCHES = ("transformer", "standard", "relational", "abstractor")
FEATURE_VIEW_KEYS = ("views", "features", "feature_views")
TEXT_KEYS = ("text_tokens", "tokens", "text", "caption")
TARGET_KEYS = ("target_tokens", "target", "trace_tokens", "trace")
MULTIMODAL_SCORE_METRICS = (
    "token", "exact", "fer", "bridge", "sequence", "all", "balanced", "mastery")


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


def _record_from_json(obj, idx, root=None):
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
    return MultimodalRecord(rec_id, split, text, target, views, meta)


def load_manifest(path, root=None):
    records = []
    with open(path) as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            records.append(_record_from_json(json.loads(line), idx, root=root))
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


def split_records(records):
    train = [r for r in records if r.split == "train"]
    evals = [r for r in records if r.split == "eval"]
    return train, (evals or train)


def build_vocab(records):
    toks = []
    for rec in records:
        toks.extend(rec.text)
        toks.extend(rec.target)
    return Vocab(toks)


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


def text_checkpoint_latent_config(path, device="cpu"):
    if not path:
        return {}
    ckpt = load_text_checkpoint_payload(path, device=device)
    return {
        "latent_concept_slots": int(ckpt.get("latent_concept_slots", 0)),
        "latent_concept_layers": int(ckpt.get("latent_concept_layers", 1)),
        "latent_concept_prefix": bool(ckpt.get("latent_concept_prefix", False)),
        "latent_concept_refine": bool(ckpt.get("latent_concept_refine", False)),
        "latent_concept_refine_gate_init": float(
            ckpt.get("latent_concept_refine_gate_init", -2.0)),
        "latent_concept_memory_size": int(
            ckpt.get("latent_concept_memory_size", 0)),
    }


def import_text_checkpoint(model, vocab, checkpoint, device=DEV):
    """Warm-start the generic text trunk and latent concept modules by token identity."""
    ckpt = load_text_checkpoint_payload(checkpoint, device=device)
    state = ckpt["state_dict"]
    src_vocab = {tok: i for i, tok in enumerate(ckpt["vocab"])}
    dst_state = model.state_dict()
    report = {
        "checkpoint": checkpoint,
        "checkpoint_experiment": ckpt.get("report", {}).get("experiment"),
        "source_vocab_size": len(src_vocab),
        "target_vocab_size": len(vocab),
        "source_d": int(ckpt.get("d", 0) or 0),
        "target_d": int(model.config["d"]),
        "source_latent_concept_slots": int(ckpt.get("latent_concept_slots", 0)),
        "target_latent_concept_slots": int(model.latent_concept_slots),
        "source_latent_concept_memory_size": int(
            ckpt.get("latent_concept_memory_size", 0)),
        "target_latent_concept_memory_size": int(model.latent_concept_memory_size),
        "copied_token_embeddings": 0,
        "overlap_tokens": 0,
        "copied_position_rows": 0,
        "copied_text_tensors": [],
        "copied_latent_tensors": [],
        "copied_sequence_tensors": [],
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
    report["copied_text_tensor_count"] = len(report["copied_text_tensors"])
    report["copied_latent_tensor_count"] = len(report["copied_latent_tensors"])
    report["copied_sequence_tensor_count"] = len(report["copied_sequence_tensors"])
    report["copied"] = bool(
        report["copied_token_embeddings"]
        or report["copied_position_rows"]
        or report["copied_text_tensor_count"]
        or report["copied_latent_tensor_count"]
        or report["copied_sequence_tensor_count"])
    model.text_checkpoint_transfer = report
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
                 latent_concept_prefix=False, latent_concept_memory_size=0):
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
        }
        self.modality_dropout = float(modality_dropout)
        self.latent_concept_slots = int(latent_concept_slots)
        self.latent_concept_layers = int(latent_concept_layers)
        self.latent_concept_prefix = bool(latent_concept_prefix)
        self.latent_concept_memory_size = int(latent_concept_memory_size)
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
            mixer_layers=self.latent_concept_layers)
            if self.latent_concept_slots > 0 else None)
        self.concept_sequence_predictor = (
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


def _batch_from_records(records, vocab, device, view_dims):
    feature_rows = {name: [] for name in view_dims}
    texts, targets = [], []
    for rec in records:
        for name, dim in view_dims.items():
            feature_rows[name].append(
                rec.views.get(name, np.zeros(int(dim), dtype=np.float32)))
        texts.append(vocab.enc(rec.text))
        targets.append(vocab.enc(rec.target))
    batch = len(records)
    features = [
        torch.tensor(np.stack(feature_rows[name]), dtype=torch.float32, device=device)
        for name in view_dims
    ]
    max_txt = max(len(t) for t in texts)
    txt = torch.full((batch, max_txt), vocab.pad, dtype=torch.long, device=device)
    for i, ids in enumerate(texts):
        txt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    max_target = max(len(t) for t in targets)
    target = torch.full((batch, max_target), vocab.pad, dtype=torch.long, device=device)
    for i, ids in enumerate(targets):
        target[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return features, txt, target


def _sample_records(records, n, rng):
    idx = rng.integers(len(records), size=int(n))
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


def latent_multimodal_sequence_prediction_loss(
        model, pair_batch, vocab, view_dims, device=DEV, temperature=0.1,
        view_dropout=0.0):
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
        anchors, vocab, device, view_dims)
    positive_features, positive_txt, _positive_ids = _batch_from_records(
        positives, vocab, device, view_dims)
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
                                           device=DEV, decay=0.99):
    memory = getattr(model, "latent_concept_memory", None)
    if (getattr(model, "latent_concepts", None) is None
            or memory is None or not pair_batch):
        return 0
    anchors = [a for a, _b in pair_batch]
    positives = [b for _a, b in pair_batch]
    anchor_features, anchor_txt, _anchor_ids = _batch_from_records(
        anchors, vocab, device, view_dims)
    positive_features, positive_txt, _positive_ids = _batch_from_records(
        positives, vocab, device, view_dims)
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
        transitive_w=0.1, target_power=1.0):
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
                batch_records, vocab, device, view_dims)
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
        model, records, vocab, view_dims, n=0, seed=0, device=DEV):
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
                batch_records, vocab, device, view_dims)
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
             device=DEV, mode="full"):
    rng = np.random.default_rng(seed)
    count = min(int(n), len(records)) if int(n) > 0 else len(records)
    sample = _sample_records(records, count, rng) if count else []
    model.eval()
    losses, correct, total, exact, rows = [], 0, 0, 0, 0
    with torch.no_grad():
        for off in range(0, len(sample), 64):
            batch_records = sample[off:off + 64]
            features, txt, ids = _batch_from_records(
                batch_records, vocab, device, view_dims)
            logits = model(features, txt, ids, mode=mode)
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
    }


def latent_multimodal_fer_eval(model, records, vocab, view_dims, n=200,
                               seed=1, device=DEV):
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
                batch_records, vocab, device, view_dims)
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
                                  seed=1, device=DEV):
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
                batch_records, vocab, device, view_dims)
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


def latent_multimodal_sequence_eval(model, records, vocab, view_dims, n=200,
                                    seed=1, device=DEV):
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
                anchors, vocab, device, view_dims)
            positive_features, positive_txt, _positive_ids = _batch_from_records(
                positives, vocab, device, view_dims)
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
                                sequence_eval=None, metric="mastery",
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
    sequence_eval = sequence_eval or {"skipped": True}
    fer_raw_score = max(0.0, float(fer_eval.get("fer_score", 0.0)))
    fer_score = (0.0 if bool(fer_eval.get("skipped", False))
                 else 1.0 / (1.0 + fer_raw_score))
    bridge_raw_score = max(0.0, float(bridge_eval.get("bridge_score", 0.0)))
    bridge_resolution = 1.0 / (1.0 + bridge_raw_score)
    bridge_connectivity = min(1.0, max(0.0, float(
        bridge_eval.get("bridge_connectivity", 1.0))))
    bridge_score = (0.0 if bool(bridge_eval.get("skipped", False))
                    else 0.5 * (bridge_resolution + bridge_connectivity))
    sequence_acc = float(sequence_eval.get("sequence_acc", 0.0))
    sequence_margin = float(sequence_eval.get("margin", 0.0))
    sequence_score = (0.0 if bool(sequence_eval.get("skipped", False))
                      else sequence_acc + margin_w * sequence_margin)
    scores = {"token": token_score, "exact": exact_score,
              "fer": fer_score, "bridge": bridge_score,
              "sequence": sequence_score}
    skipped = {"token": not token_values, "exact": not exact_values,
               "fer": bool(fer_eval.get("skipped", False)),
               "bridge": bool(bridge_eval.get("skipped", False)),
               "sequence": bool(sequence_eval.get("skipped", False))}
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
    elif metric == "fer":
        score = fer_score
    elif metric == "bridge":
        score = bridge_score
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
            "sequence_score": float(sequence_score),
            "sequence_acc": sequence_acc,
            "sequence_margin": sequence_margin,
            "token_skipped": skipped["token"],
            "exact_skipped": skipped["exact"],
            "fer_skipped": skipped["fer"],
            "bridge_skipped": skipped["bridge"],
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


def multimodal_eval_bundle(model, records, vocab, view_dims, n=200, seed=1,
                           device=DEV, score_metric="mastery",
                           score_margin_w=0.1):
    metrics = {
        mode: evaluate(model, records, vocab, view_dims, n=n,
                       seed=seed + 17 + i * 997, device=device, mode=mode)
        for i, mode in enumerate(MODES)
    }
    fer = latent_multimodal_fer_eval(
        model, records, vocab, view_dims, n=n, seed=seed + 29, device=device)
    bridge = latent_multimodal_bridge_eval(
        model, records, vocab, view_dims, n=n, seed=seed + 31, device=device)
    sequence = latent_multimodal_sequence_eval(
        model, records, vocab, view_dims, n=n, seed=seed + 37, device=device)
    return {"teacher_forced": metrics,
            "latent_fer": fer,
            "latent_bridge": bridge,
            "latent_sequence": sequence,
            "score_components": multimodal_score_components(
                metrics, fer_eval=fer, bridge_eval=bridge,
                sequence_eval=sequence, metric=score_metric,
                margin_w=score_margin_w)}


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
        "latent_study_reports",
        "selection",
    }
    return {key: value for key, value in dict(metrics).items()
            if key not in omitted}


def _multimodal_sequence_surprise_rows(
        model, pairs, vocab, view_dims, device=DEV, temperature=0.1):
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
                anchors, vocab, device, view_dims)
            positive_features, positive_txt, _positive_ids = _batch_from_records(
                positives, vocab, device, view_dims)
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


def latent_multimodal_discovery_examples(
        model, records, vocab, view_dims, n=0, seed=0, device=DEV,
        curiosity_temperature=0.1, curiosity_self_loop_w=0.05,
        curiosity_transitive_steps=2, curiosity_transitive_w=0.1,
        graph_temperature=0.1, graph_self_loop_w=0.05,
        graph_transitive_steps=2, graph_transitive_w=0.1,
        graph_target_power=1.0, cycle_temperature=0.1,
        cycle_self_loop_w=0.05, cycle_transitive_steps=2,
        cycle_transitive_w=0.1, cycle_target_power=1.0, cycle_w=0.5,
        sequence_temperature=0.1):
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
        temperature=sequence_temperature)
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
                batch_records, vocab, device, view_dims)
            views = {
                mode: model.latent_concept_states(
                    features, txt, mode=mode, project=True)
                for mode in MODES
            }
            full_slots = views["full"]
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
            for i, rec in enumerate(batch_records):
                seq = sequence_by_id.get(rec.rec_id, {})
                rows.append({
                    "record": rec,
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
        "curiosity", "gap", "insight", "graph", "cycle", "fer_score", "bridge",
        "sequence_surprise")
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
          view_tokens=4, txt_tokens=8, trunk_arch="mlp",
          trunk_width=128, trunk_depth=1, text_layers=1,
          text_arch="transformer", modality_dropout=0.0, decode_w=1.0,
          agreement_w=0.0,
          text_checkpoint=None, concept_tokens=4, fusion_layers=1,
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
          latent_concept_memory_w=0.0,
          latent_concept_memory_size=0,
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
          select_best=False, selection_rounds=1,
          selection_score_metric="mastery",
          selection_score_margin_w=0.1,
          selection_score_min_delta=0.0,
          selection_score_patience=0,
          selection_eval_n=200,
          selection_insight_accept_w=0.25,
          selection_insight_min_delta=0.0):
    ckpt_latents = text_checkpoint_latent_config(
        text_checkpoint, device="cpu") if text_checkpoint else {}
    if text_checkpoint and latent_concept_slots <= 0:
        if ckpt_latents.get("latent_concept_slots", 0) > 0:
            latent_concept_slots = ckpt_latents["latent_concept_slots"]
            latent_concept_layers = ckpt_latents["latent_concept_layers"]
    if (text_checkpoint and latent_concept_memory_size <= 0
            and ckpt_latents.get("latent_concept_memory_size", 0) > 0):
        latent_concept_memory_size = ckpt_latents["latent_concept_memory_size"]
    latent_weights = (
        latent_concept_w, latent_concept_factorization_w, latent_concept_memory_w,
        latent_concept_fer_w, latent_concept_consolidation_w,
        latent_concept_discovery_w,
        latent_concept_reanalysis_w,
        latent_concept_gap_w,
        latent_concept_association_w, latent_concept_composition_w,
        latent_concept_graph_predict_w, latent_concept_bridge_w,
        latent_concept_sequence_w,
        latent_concept_neighborhood_w,
        latent_concept_transition_w, latent_concept_cluster_w)
    if float(decode_w) < 0.0:
        raise ValueError("decoder loss weight must be non-negative")
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
    fer_hard_enabled = int(latent_concept_fer_hard_max) > 0
    discovery_hard_enabled = int(latent_concept_discovery_hard_max) > 0
    hard_study_enabled = bool(fer_hard_enabled or discovery_hard_enabled)
    hard_study_strategy = (
        "discovery" if discovery_hard_enabled
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
    if int(selection_eval_n) < 0:
        raise ValueError("multimodal selection eval count must be non-negative")
    selection_insight_accept_w = float(selection_insight_accept_w)
    if selection_insight_accept_w < 0.0:
        raise ValueError("multimodal selection insight accept weight must be non-negative")
    selection_insight_min_delta = float(selection_insight_min_delta)
    if selection_insight_min_delta < 0.0:
        raise ValueError("multimodal selection insight min delta must be non-negative")
    records = load_manifest(manifest, root=root)
    train_records, eval_records = split_records(records)
    sequence_batch = int(latent_concept_sequence_batch) or max(2, batch // 2)
    sequence_pairs, sequence_report = mine_multimodal_sequence_pairs(
        records, split="train")
    view_dims = feature_dims(records)
    vocab = build_vocab(records)
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
        latent_concept_memory_size=latent_concept_memory_size).to(device)
    if text_checkpoint:
        import_text_checkpoint(model, vocab, text_checkpoint, device=device)
    else:
        model.text_checkpoint_transfer = {}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    study_pool = []
    study_reports = []
    last = {}
    selection = {"enabled": False}
    selection_boundaries = {}
    best_state = None
    best_score = 0.0
    best_round = 0
    best_bridge_eval = None
    best_metrics = {}
    best_study_reports = []
    rounds_report = []
    no_improve_rounds = 0
    stopped_early = False
    stop_round = 0

    def selection_row(round_id, round_steps, bundle):
        score = float(bundle["score_components"]["score"])
        return {
            "round": int(round_id),
            "steps": int(round_steps),
            "selected": False,
            "score": score,
            "score_components": bundle["score_components"],
            "teacher_forced": bundle["teacher_forced"],
            "latent_fer": bundle["latent_fer"],
            "latent_bridge": bundle["latent_bridge"],
            "latent_sequence": bundle["latent_sequence"],
        }

    if select_best:
        schedule = _step_schedule(steps, selection_rounds)
        if not schedule:
            raise ValueError("multimodal selected training requires at least one step")
        cursor = 0
        for round_id, round_steps in enumerate(schedule, start=1):
            cursor += int(round_steps)
            selection_boundaries[cursor] = (round_id, int(round_steps))
        before_bundle = multimodal_eval_bundle(
            model, eval_records, vocab, view_dims, n=selection_eval_n,
            seed=seed, device=device, score_metric=selection_score_metric,
            score_margin_w=selection_score_margin_w)
        initial_row = selection_row(0, 0, before_bundle)
        best_state = _model_state_copy(model)
        best_score = float(initial_row["score"])
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
                sequence_temperature=latent_concept_sequence_temperature)
        else:
            probe_n = (int(latent_concept_fer_probe_n)
                       if int(latent_concept_fer_probe_n) > 0
                       else max(batch * 4, 1))
            hard_max = int(latent_concept_fer_hard_max)
            selected, report = latent_multimodal_fer_examples(
                model, train_records, vocab, view_dims, n=probe_n,
                seed=seed + 1301 + int(step), device=device)
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
            else int(latent_concept_fer_refresh_steps))
        refresh_due = (
            hard_study_enabled
            and (not study_pool or st == 1 or (
                refresh_steps and (st - 1) % refresh_steps == 0)))
        if refresh_due:
            refresh_study_pool(st)
            model.train()
        batch_source = study_pool if study_pool else train_records
        features, txt, ids = _batch_from_records(
            _sample_records(batch_source, batch, rng), vocab, device, view_dims)
        needs_latent = bool(any(float(w) > 0.0 for w in latent_weights)
                            or latent_concept_memory_size > 0
                            or hard_study_enabled)
        bundles_by_mode = {
            mode: model.mode_bundle(
                features, txt, ids, mode=mode, need_latent=needs_latent,
                latent_view_dropout=latent_concept_view_dropout,
                latent_project=True,
                need_logits=bool(float(decode_w) > 0.0 or float(agreement_w) > 0.0))
            for mode in MODES
        }
        logits_by_mode = {mode: bundle["logits"]
                          for mode, bundle in bundles_by_mode.items()}
        base_loss = sum(token_loss(logits, ids, vocab.pad)
                        for logits in logits_by_mode.values()) / len(logits_by_mode)
        agreement = (logit_agreement_loss(logits_by_mode, ids, vocab.pad)
                     if agreement_w else base_loss * 0.0)
        latent_views = {
            mode: bundle.get("latent_concepts") for mode, bundle in bundles_by_mode.items()
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
        latent_sequence = base_loss * 0.0
        if latent_concept_sequence_w and sequence_pairs:
            sequence_pair_batch = _sample_pairs(sequence_pairs, sequence_batch, rng)
            latent_sequence = latent_multimodal_sequence_prediction_loss(
                model, sequence_pair_batch, vocab, view_dims, device=device,
                temperature=latent_concept_sequence_temperature,
                view_dropout=latent_concept_view_dropout)
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
                decay=latent_concept_association_decay))
        fer_metrics = latent_multimodal_fer_metrics_from_views(latent_views)
        bridge_metrics = latent_multimodal_bridge_metrics_from_views(model, latent_views)
        last = {
            "loss": float(loss.detach()),
            "decode_w": float(decode_w),
            "token_loss": float(base_loss.detach()),
            "agreement_loss": float(agreement.detach()),
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
                  f"latent {last['latent_concept_loss']:.3f} "
                  f"fer {last['latent_fer_loss']:.3f} "
                  f"memory {last['latent_memory_loss']:.3f} "
                  f"consolidate {last['latent_consolidation_loss']:.3f} "
                  f"discover {last['latent_discovery_loss']:.3f} "
                  f"reanalyze {last['latent_reanalysis_loss']:.3f} "
                  f"gap {last['latent_gap_loss']:.3f} "
                  f"sequence {last['latent_sequence_loss']:.3f}",
                  flush=True)
        if select_best and st in selection_boundaries:
            round_id, round_steps = selection_boundaries[st]
            bundle = multimodal_eval_bundle(
                model, eval_records, vocab, view_dims, n=selection_eval_n,
                seed=seed, device=device, score_metric=selection_score_metric,
                score_margin_w=selection_score_margin_w)
            row = selection_row(round_id, round_steps, bundle)
            score_delta_from_best = float(row["score"] - best_score)
            bridge_insight_gate = bool(
                latent_concept_bridge_w and latent_concept_memory_size > 0)
            insight = multimodal_bridge_selection_insight(
                best_bridge_eval, row["latent_bridge"],
                enabled=bridge_insight_gate)
            decision = concept_round_selection_decision(
                score_delta_from_best, selection_score_min_delta,
                insight_delta=insight["bridge_insight_delta"],
                insight_allowed=insight["bridge_insight_allowed"],
                insight_gate=bridge_insight_gate,
                insight_accept_w=selection_insight_accept_w,
                insight_min_delta=selection_insight_min_delta)
            selected = decision["selected"]
            row = row | {
                "selected": bool(selected),
                "score_delta_from_best": score_delta_from_best,
                "bridge_insight_gate": bool(bridge_insight_gate),
                "bridge_insight_delta": float(insight["bridge_insight_delta"]),
                "bridge_insight_allowed": bool(
                    insight["bridge_insight_allowed"]),
                "bridge_insight": insight,
                "selected_by_score": bool(decision["selected_by_score"]),
                "selected_by_insight": bool(decision["selected_by_insight"]),
                "insight_score_boost": float(decision["insight_score_boost"]),
                "insight_effective_delta": float(
                    decision["insight_effective_delta"]),
                "train_metrics": _compact_multimodal_train_metrics(last),
            }
            rounds_report.append(row)
            if selected:
                best_score = float(row["score"])
                best_round = int(round_id)
                best_state = _model_state_copy(model)
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
            "selection_eval_n": int(selection_eval_n),
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
            "rounds": rounds_report,
        }
        selected_rows = [row for row in rounds_report if row["round"] == best_round]
        if selected_rows:
            selected_row = selected_rows[0]
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
        active_study_reports = list(best_study_reports)
        best_metrics = best_metrics | {
            "selection": selection,
            "latent_fer_study_reports": (
                active_study_reports if hard_study_strategy == "fer" else []),
            "latent_discovery_study_reports": (
                active_study_reports
                if hard_study_strategy == "discovery" else []),
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
            }
        last = best_metrics
    else:
        selection = {"enabled": False}
        last = dict(last) | {"selection": selection}
    model.train_metrics = last
    model.latent_fer_study_reports = (
        last.get("latent_fer_study_reports", []))
    model.latent_discovery_study_reports = (
        last.get("latent_discovery_study_reports", []))
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
    }
    return model, vocab, records, train_records, eval_records, view_dims


def run(manifest, root=None, steps=400, seed=0, device=DEV, eval_n=200,
        checkpoint=None, out=None, **kwargs):
    model, vocab, records, train_records, eval_records, view_dims = train(
        manifest, root=root, steps=steps, seed=seed, device=device, **kwargs)
    metrics = {
        mode: evaluate(model, eval_records, vocab, view_dims, n=eval_n,
                       seed=seed + 17, device=device, mode=mode)
        for mode in MODES
    }
    latent_probe = {}
    if getattr(model, "latent_concept_memory", None) is not None:
        selected, latent_probe = latent_multimodal_graph_prediction_examples(
            model, train_records, vocab, view_dims,
            n=min(64, len(train_records)), seed=seed + 23, device=device)
        latent_probe["top_ids"] = [r.rec_id for r in selected[:8]]
    latent_fer_probe = latent_multimodal_fer_eval(
        model, eval_records, vocab, view_dims, n=eval_n, seed=seed + 29,
        device=device)
    latent_fer_hard_selected, latent_fer_hard_probe = latent_multimodal_fer_examples(
        model, train_records, vocab, view_dims,
        n=min(64, len(train_records)), seed=seed + 30, device=device)
    latent_fer_hard_probe["top_ids"] = [
        r.rec_id for r in latent_fer_hard_selected[:8]]
    latent_discovery_selected, latent_discovery_probe = (
        latent_multimodal_discovery_examples(
            model, train_records, vocab, view_dims,
            n=min(64, len(train_records)), seed=seed + 32, device=device))
    latent_discovery_probe["top_ids"] = [
        r.rec_id for r in latent_discovery_selected[:8]]
    latent_bridge_probe = latent_multimodal_bridge_eval(
        model, eval_records, vocab, view_dims, n=eval_n, seed=seed + 31,
        device=device)
    latent_sequence_probe = latent_multimodal_sequence_eval(
        model, eval_records, vocab, view_dims, n=eval_n, seed=seed + 37,
        device=device)
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
        "selection": getattr(model, "train_metrics", {}).get(
            "selection", {"enabled": False}),
        "teacher_forced": metrics,
        "latent_graph_probe": latent_probe,
        "latent_fer_probe": latent_fer_probe,
        "latent_fer_hard_probe": latent_fer_hard_probe,
        "latent_discovery_probe": latent_discovery_probe,
        "latent_bridge_probe": latent_bridge_probe,
        "latent_sequence_probe": latent_sequence_probe,
        "text_checkpoint_transfer": getattr(model, "text_checkpoint_transfer", {}),
        "gate": metrics["full"]["token_acc"] >= 0.50,
    }
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos,
                    "d": model.lm.tok.embedding_dim, "model_config": model.config,
                    "manifest": model.manifest_info, "report": report}, checkpoint)
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
        vocab = build_vocab(records)
        view_dims = feature_dims(records)
        model = MultimodalLM(
            len(vocab), view_dims=view_dims, d=32, layers=1,
            heads=4, pad=vocab.pad, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8).to("cpu")
        from .text import TextFactLM, checkpoint_payload as text_checkpoint_payload
        text_ckpt = os.path.join(tmpdir, "text_reading.pt")
        text_model = TextFactLM(
            len(vocab), d=32, layers=1, heads=4, pad=vocab.pad,
            fact_schema=None, latent_concept_slots=3,
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
        torch.save(text_checkpoint_payload(
            text_model, vocab, 32, 1, 4,
            {"experiment": "text-raw-reading-selftest"}), text_ckpt)
        transfer_model = MultimodalLM(
            len(vocab), view_dims=view_dims, d=32, layers=1,
            heads=4, pad=vocab.pad, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8).to("cpu")
        transfer_report = import_text_checkpoint(
            transfer_model, vocab, text_ckpt, device="cpu")
        assert transfer_report["copied"] is True
        assert transfer_report["checkpoint_experiment"] == "text-raw-reading-selftest"
        assert transfer_report["copied_token_embeddings"] > 0
        assert transfer_report["copied_latent_tensor_count"] > 0
        assert transfer_report["copied_sequence_tensor_count"] > 0
        assert torch.allclose(
            transfer_model.txt.emb.weight[sample_idx],
            text_model.txt.emb.weight[sample_idx])
        assert torch.allclose(
            transfer_model.latent_concept_memory.memory,
            text_model.latent_concept_memory.memory)
        assert torch.allclose(
            transfer_model.concept_sequence_predictor[0].weight,
            text_model.reading_predictor[0].weight)
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
        negative_insight = multimodal_bridge_selection_insight(
            {"skipped": False, "bridge_score": 0.4, "bridge_connectivity": 0.5},
            {"skipped": False, "bridge_score": 0.8, "bridge_connectivity": 0.2})
        assert negative_insight["bridge_insight_allowed"] is False
        negative_decision = concept_round_selection_decision(
            0.2, 0.0, insight_delta=negative_insight["bridge_insight_delta"],
            insight_allowed=negative_insight["bridge_insight_allowed"],
            insight_gate=True, insight_accept_w=1.0)
        assert negative_decision["selected"] is False
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
            latent_concept_sequence_w=0.01,
            latent_concept_sequence_batch=2,
            latent_concept_neighborhood_w=0.01,
            latent_concept_transition_w=0.01,
            latent_concept_cluster_w=0.01)
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
        assert trained_model.train_metrics["latent_sequence_w"] == 0.01
        assert trained_model.train_metrics["latent_sequence_pairs"] == 7
        assert math.isfinite(trained_model.train_metrics["latent_sequence_loss"])
        assert (trained_model.train_metrics[
            "latent_sequence_transition_last_batch_updates"] > 0)
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
        no_target_model, no_target_vocab, _all, _trn, no_target_eval, no_target_dims = train(
            no_target_manifest, steps=1, batch=2, d=32, layers=1, heads=4,
            device="cpu", log_every=10, view_tokens=2, txt_tokens=4,
            decode_w=0.0, concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8, latent_concept_memory_w=0.01,
            latent_concept_bridge_w=0.01)
        assert no_target_model.train_metrics["decode_w"] == 0.0
        no_target_bundle = multimodal_eval_bundle(
            no_target_model, no_target_eval, no_target_vocab, no_target_dims,
            n=0, device="cpu", score_metric="mastery")
        assert no_target_bundle["score_components"]["token_skipped"] is True
        assert no_target_bundle["score_components"]["exact_skipped"] is True
        assert no_target_bundle["score_components"]["bridge_skipped"] is False
        discovery_model, *_ = train(
            manifest, steps=2, batch=2, d=32, layers=1, heads=4, device="cpu",
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
        selected_model, *_ = train(
            manifest, steps=2, batch=2, d=32, layers=1, heads=4, device="cpu",
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
    ap.add_argument("--agreement-w", type=float, default=0.0, dest="agreement_w",
                    help="cross-mode token-distribution agreement loss weight")
    ap.add_argument("--decode-w", type=float, default=1.0, dest="decode_w",
                    help="target-token decoder loss weight; set 0 for latent-only bridge training")
    ap.add_argument("--text-checkpoint", default=None, dest="text_checkpoint",
                    help="optional thinking.text checkpoint for text/latent warm start")
    ap.add_argument("--concept-tokens", type=int, default=4, dest="concept_tokens",
                    help="shared latent fusion tokens")
    ap.add_argument("--fusion-layers", type=int, default=1, dest="fusion_layers",
                    help="transformer layers used by latent prefix fusion")
    ap.add_argument("--latent-concept-slots", type=int, default=0,
                    dest="latent_concept_slots")
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
    ap.add_argument("--selection-eval-n", type=int, default=200,
                    dest="selection_eval_n")
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
    if not args.manifest:
        ap.error("--manifest is required")
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
        args.latent_concept_sequence_w,
        args.latent_concept_neighborhood_w, args.latent_concept_transition_w,
        args.latent_concept_cluster_w,
    ]
    if any(w < 0.0 for w in latent_weights) or args.agreement_w < 0.0 or args.decode_w < 0.0:
        ap.error("decoder/agreement/latent loss weights must be non-negative")
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
    latent_options_need_slots = (
        any(w > 0.0 for w in latent_weights)
        or args.latent_concept_memory_size > 0
        or args.latent_concept_prefix
        or args.latent_concept_fer_hard_max > 0
        or args.latent_concept_discovery_hard_max > 0)
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
    if args.selection_eval_n < 0:
        ap.error("--selection-eval-n must be non-negative")
    if args.selection_insight_accept_w < 0.0:
        ap.error("--selection-insight-accept-w must be non-negative")
    if args.selection_insight_min_delta < 0.0:
        ap.error("--selection-insight-min-delta must be non-negative")
    report = run(
        args.manifest, root=args.root, steps=args.steps, seed=args.seed,
        device=args.device, eval_n=args.eval_n, checkpoint=args.checkpoint,
        out=args.out, batch=args.batch, d=args.d, lr=args.lr, layers=args.layers,
        heads=args.heads, max_len=args.max_len, view_tokens=args.view_tokens,
        txt_tokens=args.txt_tokens,
        trunk_arch=args.trunk_arch, trunk_width=args.trunk_width,
        trunk_depth=args.trunk_depth, text_layers=args.text_layers,
        text_arch=args.text_arch, modality_dropout=args.modality_dropout,
        decode_w=args.decode_w, agreement_w=args.agreement_w,
        text_checkpoint=args.text_checkpoint,
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
        latent_concept_memory_w=args.latent_concept_memory_w,
        latent_concept_memory_size=args.latent_concept_memory_size,
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
        select_best=args.select_best,
        selection_rounds=args.selection_rounds,
        selection_score_metric=args.selection_score_metric,
        selection_score_margin_w=args.selection_score_margin_w,
        selection_score_min_delta=args.selection_score_min_delta,
        selection_score_patience=args.selection_score_patience,
        selection_eval_n=args.selection_eval_n,
        selection_insight_accept_w=args.selection_insight_accept_w,
        selection_insight_min_delta=args.selection_insight_min_delta,
        log_every=args.log_every)
    return report


if __name__ == "__main__":
    main()
