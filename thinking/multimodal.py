"""Generic multimodal prefix bridge.

This module intentionally has no built-in sensory oracle or fixed modality schema.
It trains from a JSONL manifest where each record supplies optional named feature
views, text tokens, and a target token sequence.  The bridge learns to fuse those
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
    latent_concept_association_loss,
    latent_concept_bridge_loss,
    latent_concept_bridge_scores,
    latent_concept_cluster_prototype_loss,
    latent_concept_composition_loss,
    latent_concept_fer_loss,
    latent_concept_fer_metrics,
    latent_concept_graph_prediction_loss,
    latent_concept_graph_prediction_scores,
    latent_concept_memory_loss,
    latent_concept_neighborhood_loss,
    latent_concept_slot_factorization_loss,
    latent_concept_transition_consistency_loss,
    latent_concept_vicreg_loss,
)
from .trace import Vocab

DEV = get_device()
MODES = ("full", "sensor_only", "text_only")
TRUNK_ARCHES = ("mlp", "residual")
TEXT_TRUNK_ARCHES = ("transformer", "standard", "relational", "abstractor")
FEATURE_VIEW_KEYS = ("views", "features", "feature_views")
TEXT_KEYS = ("text_tokens", "tokens", "text", "caption")
TARGET_KEYS = ("target_tokens", "target", "trace_tokens", "trace")


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
        raise ValueError(f"{rec_id}: manifest record must contain target tokens")
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
                copied_key = "copied_text_tensors"
            elif name.startswith(latent_prefixes):
                copied_key = "copied_latent_tensors"
            else:
                continue
            dst_val = dst_state.get(name)
            if dst_val is None:
                report["skipped_missing"].append(name)
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
                    "name": name,
                    "source": list(src_val.shape),
                    "target": list(dst_val.shape),
                })
                continue
            dst_val.copy_(src_val.to(device=dst_val.device, dtype=dst_val.dtype))
            report[copied_key].append(name)
    report["copied_text_tensor_count"] = len(report["copied_text_tensors"])
    report["copied_latent_tensor_count"] = len(report["copied_latent_tensors"])
    report["copied"] = bool(
        report["copied_token_embeddings"]
        or report["copied_position_rows"]
        or report["copied_text_tensor_count"]
        or report["copied_latent_tensor_count"])
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

    def mode_bundle(self, features, txt, ids, mode="full", need_latent=False,
                    latent_view_dropout=0.0, latent_project=False):
        prefix, concepts = self.encode_prefix(features, txt, mode=mode)
        latent_state_tensor = None
        if (self.latent_concept_prefix or need_latent) and self.latent_concepts is not None:
            latent_state_tensor = self.latent_concept_states_from_prefix(
                prefix, view_dropout=latent_view_dropout, project=latent_project)
        decoder_prefix = self.decoder_prefix_from_encoded(
            prefix, latent_state_tensor=latent_state_tensor)
        logits = self.lm(ids, prefix=decoder_prefix)[:, decoder_prefix.shape[1]:]
        out = {"logits": logits, "prefix": prefix, "concepts": concepts}
        if need_latent:
            out["latent_concepts"] = latent_state_tensor
        return out

    def forward(self, features, txt, ids, mode="full"):
        prefix = self.decoder_prefix(features, txt, mode=mode)
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


def latent_multimodal_bridge_metrics_from_views(model, views):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 0.0, "mode_count": 0, "modes": {}}
    memory = getattr(model, "latent_concept_memory", None)
    if memory is None:
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 1.0, "mode_count": 0, "modes": {}}
    keys = ("bridge_score", "bridge_entropy", "bridge_connectivity")
    totals = {key: 0.0 for key in keys}
    modes = {}
    for mode, slots in views.items():
        score, entropy, connectivity = latent_concept_bridge_scores(
            slots, memory.active(), memory.active_relations(),
            memory.active_transitions())
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
            "modes": modes}


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
    sample = _sample_records(records, int(n) if int(n) > 0 else len(records), rng)
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
    if (getattr(model, "latent_concepts", None) is None
            or getattr(model, "latent_concept_memory", None) is None):
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 1.0, "n_records": 0,
                "sampled": False, "skipped": True, "modes": {}}
    rng = np.random.default_rng(seed)
    count = min(int(n), len(records)) if int(n) > 0 else len(records)
    sample = _sample_records(records, count, rng) if count else []
    if not sample:
        return {"bridge_score": 0.0, "bridge_entropy": 0.0,
                "bridge_connectivity": 1.0, "n_records": 0,
                "sampled": False, "skipped": True, "modes": {}}
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
            metrics = latent_multimodal_bridge_metrics_from_views(model, views)
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
        "skipped": False,
        "modes": {
            mode: {key: vals[key] / float(total) for key in keys}
            for mode, vals in mode_totals.items()
        },
    })
    return report


def train(manifest, root=None, steps=400, batch=32, d=96, lr=1e-3, seed=0,
          device=DEV, log_every=100, layers=3, heads=4, max_len=128,
          view_tokens=4, txt_tokens=8, trunk_arch="mlp",
          trunk_width=128, trunk_depth=1, text_layers=1,
          text_arch="transformer", modality_dropout=0.0, agreement_w=0.0,
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
          latent_concept_memory_w=0.0,
          latent_concept_memory_size=0,
          latent_concept_memory_temperature=0.1,
          latent_concept_memory_momentum=0.95,
          latent_concept_memory_balance_w=0.01,
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
          latent_concept_neighborhood_w=0.0,
          latent_concept_neighborhood_temperature=0.1,
          latent_concept_neighborhood_margin=0.0,
          latent_concept_transition_w=0.0,
          latent_concept_transition_temperature=0.1,
          latent_concept_transition_margin=0.0,
          latent_concept_cluster_w=0.0,
          latent_concept_cluster_temperature=0.1,
          latent_concept_cluster_margin=0.0,
          latent_concept_cluster_min_size=2):
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
        latent_concept_fer_w, latent_concept_association_w, latent_concept_composition_w,
        latent_concept_graph_predict_w, latent_concept_bridge_w,
        latent_concept_neighborhood_w,
        latent_concept_transition_w, latent_concept_cluster_w)
    if any(float(w) < 0.0 for w in latent_weights):
        raise ValueError("latent concept loss weights must be non-negative")
    if (any(float(w) > 0.0 for w in latent_weights)
            or latent_concept_memory_size > 0) and latent_concept_slots <= 0:
        raise ValueError("latent concept losses require latent_concept_slots > 0")
    records = load_manifest(manifest, root=root)
    train_records, eval_records = split_records(records)
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
    last = {}
    for st in range(1, int(steps) + 1):
        model.train()
        features, txt, ids = _batch_from_records(
            _sample_records(train_records, batch, rng), vocab, device, view_dims)
        needs_latent = bool(any(float(w) > 0.0 for w in latent_weights)
                            or latent_concept_memory_size > 0)
        bundles_by_mode = {
            mode: model.mode_bundle(
                features, txt, ids, mode=mode, need_latent=needs_latent,
                latent_view_dropout=latent_concept_view_dropout,
                latent_project=True)
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
        latent_association = (
            latent_multimodal_association_loss_from_views(
                model, latent_views,
                temperature=latent_concept_association_temperature,
                target_power=latent_concept_association_target_power,
                self_loop_w=latent_concept_association_self_loop_w,
                transitive_steps=latent_concept_association_transitive_steps,
                transitive_w=latent_concept_association_transitive_w)
            if latent_concept_association_w else base_loss * 0.0)
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
        loss = (base_loss + float(agreement_w) * agreement
                + float(latent_concept_w) * latent_concept
                + float(latent_concept_factorization_w) * latent_factorization
                + float(latent_concept_fer_w) * latent_fer
                + float(latent_concept_memory_w) * latent_memory
                + float(latent_concept_association_w) * latent_association
                + float(latent_concept_composition_w) * latent_composition
                + float(latent_concept_graph_predict_w) * latent_graph_predict
                + float(latent_concept_bridge_w) * latent_bridge
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
                                or latent_concept_bridge_w) else None)))
        transition_updates = 0
        if latent_concept_graph_predict_w or latent_concept_bridge_w:
            transition_updates = int(update_multimodal_latent_transitions(
                model, latent_views, decay=latent_concept_association_decay))
        fer_metrics = latent_multimodal_fer_metrics_from_views(latent_views)
        bridge_metrics = latent_multimodal_bridge_metrics_from_views(model, latent_views)
        last = {
            "loss": float(loss.detach()),
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
            "latent_memory_loss": float(latent_memory.detach()),
            "latent_association_loss": float(latent_association.detach()),
            "latent_composition_loss": float(latent_composition.detach()),
            "latent_graph_predict_loss": float(latent_graph_predict.detach()),
            "latent_bridge_loss": float(latent_bridge.detach()),
            "latent_bridge_w": float(latent_concept_bridge_w),
            "latent_bridge_score": float(bridge_metrics["bridge_score"]),
            "latent_bridge_entropy": float(bridge_metrics["bridge_entropy"]),
            "latent_bridge_connectivity": float(
                bridge_metrics["bridge_connectivity"]),
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
                  f"memory {last['latent_memory_loss']:.3f}",
                  flush=True)
    model.train_metrics = last
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
    latent_bridge_probe = latent_multimodal_bridge_eval(
        model, eval_records, vocab, view_dims, n=eval_n, seed=seed + 31,
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
        "teacher_forced": metrics,
        "latent_graph_probe": latent_probe,
        "latent_fer_probe": latent_fer_probe,
        "latent_bridge_probe": latent_bridge_probe,
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
        fer_eval = latent_multimodal_fer_eval(
            model, records, vocab, view_dims, n=4, device="cpu")
        assert fer_eval["skipped"] is False
        assert math.isfinite(fer_eval["fer_score"])
        assert update_multimodal_latent_memory(
            model, views["full"], relation_decay=0.5) > 0
        assert update_multimodal_latent_transitions(model, views, decay=0.5) > 0
        assert int(model.latent_concept_memory.transition_updates.item()) > 0
        assert torch.isfinite(latent_multimodal_memory_loss_from_views(model, views))
        assert torch.isfinite(latent_multimodal_association_loss_from_views(model, views))
        assert torch.isfinite(latent_multimodal_composition_loss_from_views(model, views))
        assert torch.isfinite(latent_multimodal_graph_prediction_loss_from_views(model, views))
        assert torch.isfinite(latent_multimodal_bridge_loss_from_views(model, views))
        bridge_metrics = latent_multimodal_bridge_metrics_from_views(model, views)
        assert math.isfinite(bridge_metrics["bridge_score"])
        bridge_eval = latent_multimodal_bridge_eval(
            model, records, vocab, view_dims, n=4, device="cpu")
        assert bridge_eval["skipped"] is False
        assert math.isfinite(bridge_eval["bridge_score"])
        assert torch.isfinite(latent_multimodal_neighborhood_loss_from_views(views))
        assert torch.isfinite(latent_multimodal_transition_loss_from_views(views))
        assert torch.isfinite(latent_multimodal_cluster_loss_from_views(views))
        trained_model, *_ = train(
            manifest, steps=1, batch=2, d=32, layers=1, heads=4, device="cpu",
            log_every=1, view_tokens=2, txt_tokens=4,
            concept_tokens=2, latent_concept_slots=3,
            latent_concept_memory_size=8, latent_concept_memory_w=0.01,
            latent_concept_fer_w=0.01,
            latent_concept_association_w=0.01,
            latent_concept_composition_w=0.01,
            latent_concept_graph_predict_w=0.01,
            latent_concept_bridge_w=0.01,
            latent_concept_neighborhood_w=0.01,
            latent_concept_transition_w=0.01,
            latent_concept_cluster_w=0.01)
        assert trained_model.train_metrics["latent_graph_transition_updates"] > 0
        assert trained_model.train_metrics["latent_fer_w"] == 0.01
        assert math.isfinite(trained_model.train_metrics["latent_fer_score"])
        assert trained_model.train_metrics["latent_bridge_w"] == 0.01
        assert math.isfinite(trained_model.train_metrics["latent_bridge_score"])
    print("multimodal selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", default=None,
                    help="JSONL manifest with text/features/target tokens")
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
        args.latent_concept_association_w,
        args.latent_concept_composition_w, args.latent_concept_graph_predict_w,
        args.latent_concept_bridge_w,
        args.latent_concept_neighborhood_w, args.latent_concept_transition_w,
        args.latent_concept_cluster_w,
    ]
    if any(w < 0.0 for w in latent_weights) or args.agreement_w < 0.0:
        ap.error("agreement/latent loss weights must be non-negative")
    if args.latent_concept_view_dropout < 0.0 or args.latent_concept_view_dropout >= 1.0:
        ap.error("--latent-concept-view-dropout must be in [0, 1)")
    if (args.latent_concept_fer_fragmentation_w < 0.0
            or args.latent_concept_fer_correlation_w < 0.0
            or args.latent_concept_fer_balance_w < 0.0):
        ap.error("latent FER component weights must be non-negative")
    if (any(w > 0.0 for w in latent_weights)
            or args.latent_concept_memory_size > 0
            or args.latent_concept_prefix) and args.latent_concept_slots <= 0:
        ap.error("latent concept options require --latent-concept-slots > 0")
    if args.latent_concept_memory_temperature <= 0.0:
        ap.error("--latent-concept-memory-temperature must be positive")
    if args.latent_concept_memory_momentum < 0.0 or args.latent_concept_memory_momentum >= 1.0:
        ap.error("--latent-concept-memory-momentum must be in [0, 1)")
    if args.latent_concept_association_temperature <= 0.0:
        ap.error("--latent-concept-association-temperature must be positive")
    if args.latent_concept_composition_temperature <= 0.0:
        ap.error("--latent-concept-composition-temperature must be positive")
    if args.latent_concept_graph_predict_temperature <= 0.0:
        ap.error("--latent-concept-graph-predict-temperature must be positive")
    if args.latent_concept_neighborhood_temperature <= 0.0:
        ap.error("--latent-concept-neighborhood-temperature must be positive")
    if args.latent_concept_transition_temperature <= 0.0:
        ap.error("--latent-concept-transition-temperature must be positive")
    if args.latent_concept_cluster_temperature <= 0.0:
        ap.error("--latent-concept-cluster-temperature must be positive")
    report = run(
        args.manifest, root=args.root, steps=args.steps, seed=args.seed,
        device=args.device, eval_n=args.eval_n, checkpoint=args.checkpoint,
        out=args.out, batch=args.batch, d=args.d, lr=args.lr, layers=args.layers,
        heads=args.heads, max_len=args.max_len, view_tokens=args.view_tokens,
        txt_tokens=args.txt_tokens,
        trunk_arch=args.trunk_arch, trunk_width=args.trunk_width,
        trunk_depth=args.trunk_depth, text_layers=args.text_layers,
        text_arch=args.text_arch, modality_dropout=args.modality_dropout,
        agreement_w=args.agreement_w, text_checkpoint=args.text_checkpoint,
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
        latent_concept_memory_w=args.latent_concept_memory_w,
        latent_concept_memory_size=args.latent_concept_memory_size,
        latent_concept_memory_temperature=args.latent_concept_memory_temperature,
        latent_concept_memory_momentum=args.latent_concept_memory_momentum,
        latent_concept_memory_balance_w=args.latent_concept_memory_balance_w,
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
        log_every=args.log_every)
    return report


if __name__ == "__main__":
    main()
