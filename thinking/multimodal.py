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
import math
import os
import string
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from scratchpad_model import CausalBlock, ScratchpadLM

from .audio import (ENVELOPES, PITCH_NAMES, TIMBRES, render_tone, sample_clip, spectrogram)
from .concepts import (
    LatentConceptHead,
    LatentConceptMemory,
    SchemaConceptHead,
    SchemaConceptRefiner,
    latent_concept_composition_loss,
    latent_concept_graph_prediction_loss,
    latent_concept_graph_prediction_scores,
    latent_concept_graph_curiosity_scores,
    latent_concept_cluster_prototype_loss,
    latent_concept_neighborhood_loss,
    latent_concept_slot_factorization_loss,
    latent_concept_transition_consistency_loss,
    latent_concept_vicreg_loss,
    schema_concept_batch_centroid_loss,
    schema_concept_contrastive_loss,
    schema_concept_prototype_alignment_loss,
    schema_concept_prototype_spread_loss,
    schema_concept_state_spread_loss,
)
from .vision import COLORS, SHAPES, ObjectSpec, render_object, sample_object
from .trace import Vocab

DEV = get_device()
COLOR_NAMES = tuple(COLORS)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SURFACES = os.path.join(ROOT, "data", "multimodal_transcripts.json")
MODES = ("full", "sensor_only", "text_only")
MULTIMODAL_STUDY_METRICS = ("latent", "concept", "decoder", "balanced")
TRUNK_ARCHES = ("conv", "residual")
TEXT_TRUNK_ARCHES = ("transformer", "standard", "relational", "abstractor")
FACTOR_VALUES = {
    "color": COLOR_NAMES,
    "shape": SHAPES,
    "pitch": PITCH_NAMES,
    "timbre": TIMBRES,
    "env": ENVELOPES,
}
FACTOR_KEYS = {
    "color": ("p0", "color"),
    "shape": ("p0", "shape"),
    "pitch": ("a0", "pitch"),
    "timbre": ("a0", "timbre"),
    "env": ("a0", "env"),
}
VALUE_POS = {"color": 4, "shape": 9, "pitch": 14, "timbre": 19, "env": 24}  # value tokens
FACTOR_INDEX = {k: {v: i for i, v in enumerate(vals)}
                for k, vals in FACTOR_VALUES.items()}
FORMATTER = string.Formatter()


def _split_words(s):
    return s.split()


def _template_fields(tpl):
    fields = []
    for _literal, field, _format_spec, _conversion in FORMATTER.parse(tpl):
        if field is not None:
            fields.append(field)
    return fields


def _check_template_fields(path, split, tpl):
    required = set(FACTOR_VALUES)
    fields = set(_template_fields(tpl))
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{path}:{split} template missing placeholders {missing}: {tpl}")
    unsupported = sorted(fields - required)
    if unsupported:
        raise ValueError(f"{path}:{split} template has unsupported placeholders "
                         f"{sorted(unsupported)}: {tpl}")


def _check_gold(path, split, facts):
    missing = sorted(set(FACTOR_VALUES) - set(facts))
    if missing:
        raise ValueError(f"{path}:{split} example missing facts {missing}")
    gold = {k: facts[k] for k in FACTOR_VALUES}
    bad = {k: v for k, v in gold.items() if v not in FACTOR_VALUES[k]}
    if bad:
        raise ValueError(f"{path}:{split} example has invalid fact values {bad}")
    return gold


def _normalize_text_example(path, split, idx, rec):
    if not isinstance(rec, dict):
        raise ValueError(f"{path}:{split}_examples[{idx}] must be an object")
    if "tokens" in rec:
        tokens = list(rec["tokens"])
    elif "text" in rec:
        tokens = _split_words(rec["text"])
    else:
        raise ValueError(f"{path}:{split}_examples[{idx}] must contain text or tokens")
    if not tokens or not all(isinstance(t, str) for t in tokens):
        raise ValueError(f"{path}:{split}_examples[{idx}] has empty or non-string tokens")
    facts = rec.get("facts")
    if not isinstance(facts, dict):
        raise ValueError(f"{path}:{split}_examples[{idx}] must contain a facts object")
    return {"tokens": tokens, "facts": _check_gold(path, split, facts)}


def _load_examples(data, path, split):
    raw = data.get(f"{split}_examples", data.get("examples", {}).get(split, []))
    return [_normalize_text_example(path, split, i, rec) for i, rec in enumerate(raw)]


def load_text_surfaces(path=None):
    path = path or DEFAULT_SURFACES
    with open(path) as f:
        data = json.load(f)
    out = {"train": list(data.get("train", [])), "eval": list(data.get("eval", [])),
           "train_examples": _load_examples(data, path, "train"),
           "eval_examples": _load_examples(data, path, "eval")}
    for split in ("train", "eval"):
        if not out[split] and not out[f"{split}_examples"]:
            raise ValueError(f"{path} must contain {split} templates or {split}_examples")
    for split, bank in out.items():
        if split.endswith("_examples"):
            continue
        for tpl in bank:
            _check_template_fields(path, split, tpl)
    return out


def render_transcript(gold, rng, surfaces, split="train"):
    bank = surfaces[split]
    tpl = bank[int(rng.integers(len(bank)))]
    return _split_words(tpl.format(**gold))


def sample_gold(rng):
    obj = sample_object(rng, slot="p0")
    clip = sample_clip(rng)
    return {"color": obj.color, "shape": obj.shape, "pitch": clip["pitch"],
            "timbre": clip["timbre"], "env": clip["envelope"]}


def sample_text_and_gold(rng, surfaces, split="train"):
    templates = surfaces[split]
    examples = surfaces.get(f"{split}_examples", [])
    pick = int(rng.integers(len(templates) + len(examples)))
    if pick < len(examples):
        ex = examples[pick]
        return list(ex["tokens"]), dict(ex["facts"])
    gold = sample_gold(rng)
    tpl = templates[pick - len(examples)]
    return _split_words(tpl.format(**gold)), gold


def render_modalities(gold, rng):
    base = sample_object(rng, slot="p0")
    obj = ObjectSpec("p0", gold["color"], gold["shape"], x=base.x, y=base.y, scale=base.scale)
    clip = sample_clip(rng)
    img = render_object(obj, size=32)
    aud = spectrogram(render_tone(gold["pitch"], gold["timbre"], gold["env"],
                                  clip["detune"], clip["amp"], clip["phase"], rng=rng))
    return img, aud


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


def _grid_pool(n_tokens):
    n = int(n_tokens)
    if n <= 0:
        raise ValueError("prefix token count must be positive")
    h = int(math.sqrt(n))
    while h > 1 and n % h:
        h -= 1
    return h, n // h


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
    """Warm-start multimodal text/concept modules from a text learner checkpoint.

    The transfer is identity-based and shape-checked: shared vocabulary rows are copied by token
    string, position rows are copied up to the common length, and encoder/latent concept tensors
    move only when the current multimodal architecture actually matches the source tensor.
    """
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
        "source_text_arch": ckpt.get("text_encoder_arch", "transformer"),
        "target_text_arch": model.txt.arch,
        "source_text_layers": int(ckpt.get("text_encoder_layers", 0) or 0),
        "target_text_layers": int(model.txt.layers),
        "source_latent_concept_slots": int(ckpt.get("latent_concept_slots", 0)),
        "source_latent_concept_memory_size": int(
            ckpt.get("latent_concept_memory_size", 0)),
        "target_latent_concept_slots": int(model.latent_concept_slots),
        "target_latent_concept_memory_size": int(
            getattr(model, "latent_concept_memory_size", 0)),
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
    latent_prefixes = ("latent_concepts.", "latent_concept_refiner.",
                       "latent_concept_memory.")
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


class ResidualConvBlock(nn.Module):
    """Small residual conv block for multimodal sensory prefix encoders."""

    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class PatchTrunk(nn.Module):
    """Conv trunk -> a short sequence of d-dim prefix embeddings + modality embedding."""

    def __init__(self, in_ch, d, n_tokens=4, modality=0, n_modalities=3, pool=(2, 2),
                 arch="conv", width=64, depth=1):
        """pool: trunk output cells -> prefix tokens. Audio needs FREQUENCY-preserving pooling
        ((8,1): 8 frequency bands x collapsed time) -- 2x2 pooling crushed the frequency axis
        and pitch accuracy with it (0.32; every other factor >=0.93)."""
        super().__init__()
        arch = str(arch)
        if arch not in TRUNK_ARCHES:
            raise ValueError(f"unknown multimodal trunk arch {arch!r}")
        self.n_tokens = int(n_tokens)
        self.arch = arch
        self.width = int(width)
        self.depth = int(depth)
        self.pool = (int(pool[0]), int(pool[1]))
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("multimodal trunk width/depth must be positive")
        if self.n_tokens != self.pool[0] * self.pool[1]:
            raise ValueError("n_tokens must match the configured trunk pool cells")
        if arch == "conv":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(48, self.width, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(self.pool),
            )
        else:
            blocks = [
                nn.Conv2d(in_ch, self.width, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(self.width, self.width, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            ]
            for _ in range(self.depth):
                blocks.append(ResidualConvBlock(self.width))
            blocks.append(nn.AdaptiveAvgPool2d(self.pool))
            self.conv = nn.Sequential(*blocks)
        self.proj = nn.Linear(self.width, d)
        self.mod = nn.Embedding(n_modalities, d)
        self.modality = modality
        self.posn = nn.Parameter(torch.zeros(self.n_tokens, d))

    def forward(self, x):
        h = self.conv(x)                                    # B,64,2,2
        h = h.flatten(2).transpose(1, 2)                    # B,4,64
        return self.proj(h) + self.mod.weight[self.modality] + self.posn[None]


class TextTrunk(nn.Module):
    """Transcript encoder -> per-token prefix embeddings.  Pads are zeroed after encoding."""

    def __init__(self, vocab_size, d, pad=0, n_tokens=8, heads=4, layers=1, modality=2,
                 n_modalities=3, max_len=64, arch="transformer"):
        super().__init__()
        self.pad = pad
        self.modality = modality
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
        self.mod = nn.Embedding(n_modalities, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, ids):
        L = ids.shape[1]
        pos = torch.arange(L, device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        pad_mask = ids.eq(self.pad)
        h = self.emb(ids) + self.pos(pos)[None]
        if self.enc is not None:
            h = self.enc(h, src_key_padding_mask=pad_mask)
        else:
            for block in self.blocks:
                h = block(h, ids, pad_mask)
        h = self.ln(h + self.mod.weight[self.modality])
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        if h.shape[1] > self.n_tokens:
            return h[:, :self.n_tokens]
        if h.shape[1] < self.n_tokens:
            pad = torch.zeros(h.shape[0], self.n_tokens - h.shape[1], h.shape[2],
                              dtype=h.dtype, device=h.device)
            h = torch.cat([h, pad], dim=1)
        return h


class ConceptFusion(nn.Module):
    """Shared concept-token mixer for image/audio/text prefixes before decoding."""

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

    def forward(self, ip, ap, tp):
        q = self.queries.unsqueeze(0).expand(ip.shape[0], -1, -1)
        h = torch.cat([q, ip, ap, tp], dim=1)
        h = self.ln(self.enc(h))
        return h, h[:, :self.concept_tokens]


class MultimodalLM(nn.Module):
    """Image, audio, and transcript readers feeding one trace decoder."""

    def __init__(self, vocab_size, d=96, layers=3, heads=4, pad=0, max_len=128,
                 img_tokens=4, aud_tokens=8, txt_tokens=8, trunk_arch="conv",
                 trunk_width=64, trunk_depth=1, text_layers=1, modality_dropout=0.0,
                 text_arch="transformer", concept_tokens=4, fusion_layers=1,
                 concept_prefix=False, concept_refine=False,
                 concept_refine_gate_init=-2.0,
                 concept_mixer_layers=0, concept_mixer_gate_init=-2.0,
                 latent_concept_slots=0, latent_concept_layers=1,
                 latent_concept_prefix=False,
                 latent_concept_refine=False,
                 latent_concept_refine_gate_init=-2.0,
                 latent_concept_memory_size=0):
        super().__init__()
        if img_tokens <= 0 or aud_tokens <= 0 or txt_tokens <= 0:
            raise ValueError("multimodal prefix token counts must be positive")
        if modality_dropout < 0.0 or modality_dropout > 1.0:
            raise ValueError("modality_dropout must be in [0, 1]")
        if int(latent_concept_slots) < 0:
            raise ValueError("latent concept slots must be non-negative")
        if int(latent_concept_slots) > 0 and int(latent_concept_layers) <= 0:
            raise ValueError("latent concept layers must be positive")
        if (latent_concept_prefix or latent_concept_refine) and int(latent_concept_slots) <= 0:
            raise ValueError("latent concept prefix/refine require latent slots")
        if int(latent_concept_memory_size) < 0:
            raise ValueError("latent concept memory size must be non-negative")
        if int(latent_concept_memory_size) and int(latent_concept_slots) <= 0:
            raise ValueError("latent concept memory requires latent slots")
        trunk_arch = str(trunk_arch)
        text_arch = str(text_arch)
        if text_arch not in TEXT_TRUNK_ARCHES:
            raise ValueError(f"unknown text trunk architecture {text_arch!r}")
        self.config = {
            "vocab_size": int(vocab_size), "d": int(d), "layers": int(layers),
            "heads": int(heads), "pad": int(pad), "max_len": int(max_len),
            "img_tokens": int(img_tokens), "aud_tokens": int(aud_tokens),
            "txt_tokens": int(txt_tokens), "trunk_arch": trunk_arch,
            "trunk_width": int(trunk_width), "trunk_depth": int(trunk_depth),
            "text_layers": int(text_layers), "text_arch": text_arch,
            "modality_dropout": float(modality_dropout),
            "fusion": "concept", "concept_tokens": int(concept_tokens),
            "fusion_layers": int(fusion_layers),
            "concept_prefix": bool(concept_prefix),
            "concept_refine": bool(concept_refine),
            "concept_refine_gate_init": float(concept_refine_gate_init),
            "concept_mixer_layers": int(concept_mixer_layers),
            "concept_mixer_gate_init": float(concept_mixer_gate_init),
            "latent_concept_slots": int(latent_concept_slots),
            "latent_concept_layers": int(latent_concept_layers),
            "latent_concept_prefix": bool(latent_concept_prefix),
            "latent_concept_refine": bool(latent_concept_refine),
            "latent_concept_refine_gate_init": float(latent_concept_refine_gate_init),
            "latent_concept_memory_size": int(latent_concept_memory_size),
        }
        self.modality_dropout = float(modality_dropout)
        self.concept_prefix = bool(concept_prefix)
        self.concept_refine = bool(concept_refine)
        self.latent_concept_slots = int(latent_concept_slots)
        self.latent_concept_layers = int(latent_concept_layers)
        self.latent_concept_prefix = bool(latent_concept_prefix)
        self.latent_concept_refine = bool(latent_concept_refine)
        self.latent_concept_memory_size = int(latent_concept_memory_size)
        self.lm = ScratchpadLM(vocab_size, d=d, layers=layers, heads=heads, max_len=max_len,
                               pad=pad, pointer=False, loop=False)
        img_pool = _grid_pool(img_tokens)
        self.img = PatchTrunk(3, d, n_tokens=img_tokens, modality=0,
                              pool=img_pool, arch=trunk_arch,
                              width=trunk_width, depth=trunk_depth)
        self.aud = PatchTrunk(1, d, n_tokens=aud_tokens, modality=1,
                              pool=(aud_tokens, 1), arch=trunk_arch,
                              width=trunk_width, depth=trunk_depth)
        self.txt = TextTrunk(vocab_size, d, pad=pad, n_tokens=txt_tokens, heads=heads,
                             layers=text_layers, modality=2, arch=text_arch)
        self.fusion = ConceptFusion(d, heads=heads, concept_tokens=concept_tokens,
                                    layers=fusion_layers)
        self.factor_concepts = SchemaConceptHead(
            [FACTOR_KEYS[factor] for factor in VALUE_POS],
            [FACTOR_VALUES[factor] for factor in VALUE_POS],
            d, mixer_layers=int(concept_mixer_layers), mixer_heads=heads,
            mixer_gate_init=float(concept_mixer_gate_init))
        self.concept_refiner = (SchemaConceptRefiner(
            d, heads=heads, gate_init=concept_refine_gate_init)
            if self.concept_refine else None)
        self.latent_concepts = (LatentConceptHead(
            self.latent_concept_slots, d, heads=heads,
            mixer_layers=self.latent_concept_layers)
            if self.latent_concept_slots > 0 else None)
        self.latent_concept_memory = (LatentConceptMemory(
            self.latent_concept_memory_size, d)
            if self.latent_concept_memory_size > 0 else None)
        self.latent_concept_refiner = (SchemaConceptRefiner(
            d, heads=heads, gate_init=latent_concept_refine_gate_init)
            if self.latent_concept_refine and self.latent_concepts is not None else None)

    def enable_latent_concepts(self, slots, heads=None, layers=1):
        slots = int(slots)
        latent_heads = int(heads or self.config["heads"])
        latent_layers = int(layers)
        if slots <= 0:
            self.latent_concepts = None
            self.latent_concept_slots = 0
            self.latent_concept_prefix = False
            self.latent_concept_refine = False
            self.latent_concept_refiner = None
            self.latent_concept_memory = None
            self.latent_concept_memory_size = 0
            self.config["latent_concept_slots"] = 0
            self.config["latent_concept_prefix"] = False
            self.config["latent_concept_refine"] = False
            self.config["latent_concept_memory_size"] = 0
            return self
        if (self.latent_concepts is None
                or self.latent_concept_slots != slots
                or getattr(self.latent_concepts, "heads", latent_heads) != latent_heads
                or getattr(self.latent_concepts, "mixer_layers", latent_layers)
                != latent_layers):
            self.latent_concepts = LatentConceptHead(
                slots, self.config["d"], heads=latent_heads,
                mixer_layers=latent_layers)
            self.latent_concepts.to(next(self.parameters()).device)
        self.latent_concept_slots = slots
        self.latent_concept_layers = latent_layers
        self.config["latent_concept_slots"] = slots
        self.config["latent_concept_layers"] = latent_layers
        return self

    def enable_latent_concept_memory(self, size):
        size = int(size)
        if size <= 0 or self.latent_concepts is None:
            self.latent_concept_memory = None
            self.latent_concept_memory_size = 0
            self.config["latent_concept_memory_size"] = 0
            return self
        if (self.latent_concept_memory is None
                or self.latent_concept_memory_size != size):
            self.latent_concept_memory = LatentConceptMemory(size, self.config["d"])
            self.latent_concept_memory.to(next(self.parameters()).device)
        self.latent_concept_memory_size = size
        self.config["latent_concept_memory_size"] = size
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

    def enable_latent_concept_refiner(self, heads=None, gate_init=-2.0):
        if self.latent_concepts is None:
            self.latent_concept_refine = False
            self.config["latent_concept_refine"] = False
            return self
        if self.latent_concept_refiner is None:
            self.latent_concept_refiner = SchemaConceptRefiner(
                self.config["d"], heads=int(heads or self.config["heads"]),
                gate_init=gate_init)
            self.latent_concept_refiner.to(next(self.parameters()).device)
            self.config["latent_concept_refine_gate_init"] = float(gate_init)
        self.latent_concept_refine = True
        self.config["latent_concept_refine"] = True
        return self

    def _apply_modality_dropout(self, ip, ap, tp):
        if not self.training or self.modality_dropout <= 0.0:
            return ip, ap, tp
        keep = 1.0 - self.modality_dropout
        if keep <= 0.0:
            return ip * 0.0, ap * 0.0, tp * 0.0
        masks = [
            torch.rand(ip.shape[0], 1, 1, device=ip.device).lt(keep).to(ip.dtype) / keep,
            torch.rand(ap.shape[0], 1, 1, device=ap.device).lt(keep).to(ap.dtype) / keep,
            torch.rand(tp.shape[0], 1, 1, device=tp.device).lt(keep).to(tp.dtype) / keep,
        ]
        return ip * masks[0], ap * masks[1], tp * masks[2]

    def encode_prefix(self, img, aud, txt, mode="full"):
        ip, ap, tp = self.img(img), self.aud(aud), self.txt(txt)
        if mode == "text_only":
            ip, ap = ip * 0.0, ap * 0.0
        elif mode == "sensor_only":
            tp = tp * 0.0
        elif mode != "full":
            raise ValueError(f"unknown multimodal mode {mode!r}")
        elif self.modality_dropout > 0.0:
            ip, ap, tp = self._apply_modality_dropout(ip, ap, tp)
        prefix, concepts = self.fusion(ip, ap, tp)
        if self.concept_refiner is not None:
            source = self._source_from_prefix(prefix, concepts)
            factor_state_tensor = self.factor_concepts.state_tensor(source)
            prefix = self.concept_refiner(prefix, factor_state_tensor)
            concepts = prefix[:, :self.fusion.concept_tokens]
        if self.latent_concept_refiner is not None and self.latent_concepts is not None:
            latent = self.latent_concepts(prefix)
            prefix = self.latent_concept_refiner(prefix, latent)
            concepts = prefix[:, :self.fusion.concept_tokens]
        return prefix, concepts

    def _factor_states_from_tensor(self, state_tensor):
        return {factor: state_tensor[:, i] for i, factor in enumerate(VALUE_POS)}

    def _source_from_prefix(self, prefix, concepts):
        return concepts if concepts is not None else prefix

    def factor_concept_states(self, img, aud, txt, mode="full"):
        prefix, concepts = self.encode_prefix(img, aud, txt, mode=mode)
        source = self._source_from_prefix(prefix, concepts)
        return self._factor_states_from_tensor(self.factor_concepts.state_tensor(source))

    def factor_concept_geometry_states(self, img, aud, txt, mode="full"):
        prefix, concepts = self.encode_prefix(img, aud, txt, mode=mode)
        source = self._source_from_prefix(prefix, concepts)
        state_tensor = self.factor_concepts.geometry_state_tensor(source)
        return self._factor_states_from_tensor(state_tensor)

    def factor_geometry_from_states(self, states):
        state_tensor = torch.stack([states[factor] for factor in VALUE_POS], dim=1)
        projected = self.factor_concepts.geometry_state_tensor_from_states(state_tensor)
        return {factor: projected[:, i] for i, factor in enumerate(VALUE_POS)}

    def factor_logits_from_state_tensor(self, state_tensor):
        logits_by_key = self.factor_concepts.logits_from_state_tensor(state_tensor)
        return {factor: logits_by_key[FACTOR_KEYS[factor]] for factor in VALUE_POS}

    def factor_logits_from_states(self, states):
        states_by_key = {FACTOR_KEYS[factor]: states[factor] for factor in VALUE_POS}
        logits_by_key = self.factor_concepts.logits_from_states(states_by_key)
        return {factor: logits_by_key[FACTOR_KEYS[factor]] for factor in VALUE_POS}

    def latent_factor_state_tensor_from_slots(self, latent_slots):
        if latent_slots is None:
            return None
        return self.factor_concepts.state_tensor(latent_slots)

    def latent_factor_logits_from_slots(self, latent_slots):
        state_tensor = self.latent_factor_state_tensor_from_slots(latent_slots)
        if state_tensor is None:
            return {}
        return self.factor_logits_from_state_tensor(state_tensor)

    def latent_factor_logits(self, img, aud, txt, mode="full"):
        latent = self.latent_concept_states(img, aud, txt, mode=mode)
        return self.latent_factor_logits_from_slots(latent)

    def factor_logits(self, img, aud, txt, mode="full"):
        return self.factor_logits_from_states(
            self.factor_concept_states(img, aud, txt, mode=mode))

    def latent_concept_states_from_prefix(self, prefix, view_dropout=0.0, project=False):
        if self.latent_concepts is None:
            return None
        if self.training and view_dropout > 0.0:
            prefix = F.dropout(prefix, p=float(view_dropout), training=True)
        return self.latent_concepts(prefix, project=project)

    def latent_concept_states(self, img, aud, txt, mode="full", view_dropout=0.0,
                              project=False):
        prefix, _concepts = self.encode_prefix(img, aud, txt, mode=mode)
        return self.latent_concept_states_from_prefix(
            prefix, view_dropout=view_dropout, project=project)

    def decoder_prefix_from_encoded(self, prefix, concepts, factor_state_tensor=None,
                                    latent_state_tensor=None):
        extra = []
        if self.latent_concept_prefix and self.latent_concepts is not None:
            if latent_state_tensor is None:
                latent_state_tensor = self.latent_concepts(prefix)
            extra.append(latent_state_tensor)
        if self.concept_prefix:
            if factor_state_tensor is None:
                source = self._source_from_prefix(prefix, concepts)
                factor_state_tensor = self.factor_concepts.state_tensor(source)
            extra.append(factor_state_tensor)
        if extra:
            prefix = torch.cat(extra + [prefix], dim=1)
        return prefix

    def decoder_prefix(self, img, aud, txt, mode="full"):
        prefix, concepts = self.encode_prefix(img, aud, txt, mode=mode)
        return self.decoder_prefix_from_encoded(prefix, concepts)

    def mode_bundle(self, img, aud, txt, ids, mode="full", need_factor=False,
                    need_geometry=False, need_latent=False,
                    need_latent_factor=False, latent_view_dropout=0.0,
                    latent_project=False):
        prefix, concepts = self.encode_prefix(img, aud, txt, mode=mode)
        source = self._source_from_prefix(prefix, concepts)
        factor_state_tensor = None
        latent_state_tensor = None
        if self.concept_prefix or need_factor or need_geometry:
            factor_state_tensor = self.factor_concepts.state_tensor(source)
        if self.latent_concept_prefix and self.latent_concepts is not None:
            latent_state_tensor = self.latent_concepts(prefix)
        decoder_prefix = self.decoder_prefix_from_encoded(
            prefix, concepts, factor_state_tensor=factor_state_tensor,
            latent_state_tensor=latent_state_tensor)
        logits = self.lm(ids, prefix=decoder_prefix)[:, decoder_prefix.shape[1]:]
        out = {"logits": logits, "prefix": prefix, "concepts": concepts}
        if need_latent:
            out["latent_concepts"] = self.latent_concept_states_from_prefix(
                prefix, view_dropout=latent_view_dropout, project=latent_project)
        if need_latent_factor:
            if latent_state_tensor is None and self.latent_concepts is not None:
                latent_state_tensor = self.latent_concepts(prefix)
            out["latent_factor_logits"] = self.latent_factor_logits_from_slots(
                latent_state_tensor)
        if need_factor or need_geometry:
            if factor_state_tensor is None:
                factor_state_tensor = self.factor_concepts.state_tensor(source)
            out["factor_states"] = self._factor_states_from_tensor(factor_state_tensor)
            out["factor_logits"] = self.factor_logits_from_state_tensor(factor_state_tensor)
        if need_geometry:
            geometry_tensor = self.factor_concepts.geometry_state_tensor_from_states(
                factor_state_tensor)
            out["factor_geometry_states"] = self._factor_states_from_tensor(
                geometry_tensor)
        return out

    def forward(self, img, aud, txt, ids, mode="full"):
        prefix = self.decoder_prefix(img, aud, txt, mode=mode)
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
    for split in ("train_examples", "eval_examples"):
        for ex in surfaces.get(split, []):
            toks += list(ex["tokens"])
    return Vocab(toks)


def sample_example(rng, surfaces, text_split="train"):
    txt, gold = sample_text_and_gold(rng, surfaces, split=text_split)
    img, aud = render_modalities(gold, rng)
    toks = trace_tokens(gold)
    return img, aud, txt, toks, gold


def _sample_examples(n, rng, surfaces, text_split="train"):
    return [sample_example(rng, surfaces, text_split=text_split) for _ in range(n)]


def _batch_from_examples(examples, vocab, device):
    imgs, auds, txts, seqs, golds = [], [], [], [], []
    for img, aud, txt, toks, gold in examples:
        imgs.append(img)
        auds.append(aud)
        txts.append(vocab.enc(txt))
        seqs.append(vocab.enc(toks))
        golds.append(gold)
    n = len(examples)
    x = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
    a = torch.tensor(np.stack(auds), dtype=torch.float32, device=device)
    max_txt = max(len(t) for t in txts)
    t = torch.full((n, max_txt), vocab.pad, dtype=torch.long, device=device)
    for i, ids in enumerate(txts):
        t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    ids = torch.tensor(seqs, dtype=torch.long, device=device)   # fixed grammar = fixed length
    return x, a, t, ids, golds


def _batch(n, rng, vocab, device, surfaces, text_split="train"):
    return _batch_from_examples(_sample_examples(n, rng, surfaces, text_split=text_split),
                                vocab, device)


def _sample_from_pool(pool, n, rng):
    if not pool:
        return []
    return [pool[int(rng.integers(len(pool)))] for _ in range(n)]


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


def concept_evaluate(model, vocab, surfaces, n=200, seed=11, device=DEV, text_split="eval",
                     mode="full"):
    """Auxiliary concept-token factor accuracy before the trace decoder."""
    rng = np.random.default_rng(seed)
    model.eval()
    hits = {k: 0 for k in VALUE_POS}
    with torch.no_grad():
        for off in range(0, n, 50):
            b = min(50, n - off)
            img, aud, txt, _ids, golds = _batch(b, rng, vocab, device, surfaces,
                                                text_split=text_split)
            logits = model.factor_logits(img, aud, txt, mode=mode)
            for factor in VALUE_POS:
                pred = logits[factor].argmax(-1)
                values = FACTOR_VALUES[factor]
                for r in range(b):
                    hits[factor] += int(values[int(pred[r])] == golds[r][factor])
    return {k: v / n for k, v in hits.items()}


def latent_factor_evaluate(model, vocab, surfaces, n=200, seed=12, device=DEV,
                           text_split="eval", mode="full"):
    """Factor accuracy when data-defined factor heads read only latent slots."""
    if getattr(model, "latent_concepts", None) is None:
        return {k: 0.0 for k in VALUE_POS} | {"n_records": 0, "skipped": True}
    rng = np.random.default_rng(seed)
    model.eval()
    hits = {k: 0 for k in VALUE_POS}
    with torch.no_grad():
        for off in range(0, n, 50):
            b = min(50, n - off)
            img, aud, txt, _ids, golds = _batch(b, rng, vocab, device, surfaces,
                                                text_split=text_split)
            logits = model.latent_factor_logits(img, aud, txt, mode=mode)
            for factor in VALUE_POS:
                pred = logits[factor].argmax(-1)
                values = FACTOR_VALUES[factor]
                for r in range(b):
                    hits[factor] += int(values[int(pred[r])] == golds[r][factor])
    return {k: v / n for k, v in hits.items()} | {"n_records": int(n), "skipped": False}


def multimodal_factor_record_outcomes(model, vocab, surfaces, n=0, seed=0, device=DEV,
                                      text_split="train", metric="latent", modes=MODES,
                                      examples=None):
    """Return examples currently wrong/right under an upstream factor metric."""
    metric = str(metric)
    if metric not in MULTIMODAL_STUDY_METRICS:
        raise ValueError(f"unknown multimodal study score metric {metric!r}")
    active_metrics = (
        ("latent", "concept", "decoder") if metric == "balanced" else (metric,))
    if "latent" in active_metrics and getattr(model, "latent_concepts", None) is None:
        if metric == "latent":
            return [], [], {"n_records": 0, "sampled": False, "n_error_records": 0,
                            "n_correct_records": 0, "n_factor_checks": 0,
                            "n_errors": 0, "factor_error_rate": 0.0,
                            "metric": metric, "active_metrics": [],
                            "modes": list(modes), "by_metric": {},
                            "by_mode": {}, "by_factor": {}, "skipped": True}
        active_metrics = tuple(name for name in active_metrics if name != "latent")
    if not active_metrics:
        return [], [], {"n_records": 0, "sampled": False, "n_error_records": 0,
                        "n_correct_records": 0, "n_factor_checks": 0,
                        "n_errors": 0, "factor_error_rate": 0.0,
                        "metric": metric, "active_metrics": [],
                        "modes": list(modes), "by_metric": {},
                        "by_mode": {}, "by_factor": {}, "skipped": True}
    if examples is None:
        rng = np.random.default_rng(seed)
        count = int(n)
        if count <= 0:
            count = 1
        examples = _sample_examples(count, rng, surfaces, text_split=text_split)
        sampled = True
    else:
        examples = list(examples)
        sampled = False
        if n and n < len(examples):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(examples), size=int(n), replace=False)
            examples = [examples[int(i)] for i in idx]
            sampled = True
    errors = []
    correct = []
    by_metric = {name: {"checks": 0, "errors": 0} for name in active_metrics}
    by_mode = {mode: {"checks": 0, "errors": 0} for mode in modes}
    by_factor = {factor: {"checks": 0, "errors": 0} for factor in VALUE_POS}
    n_checks = 0
    n_errors = 0
    model.eval()
    with torch.no_grad():
        for off in range(0, len(examples), 64):
            batch_examples = examples[off:off + 64]
            img, aud, txt, ids, golds = _batch_from_examples(batch_examples, vocab, device)
            wrong_rows = torch.zeros(len(batch_examples), dtype=torch.bool, device=device)
            for mode in modes:
                for active_metric in active_metrics:
                    if active_metric == "latent":
                        logits_by_factor = model.latent_factor_logits(
                            img, aud, txt, mode=mode)
                    elif active_metric == "concept":
                        logits_by_factor = model.factor_logits(img, aud, txt, mode=mode)
                    else:
                        logits = model(img, aud, txt, ids, mode=mode)
                        logits_by_factor = {
                            factor: logits[:, pos - 1].index_select(
                                -1, _candidate_ids(vocab, factor, device))
                            for factor, pos in VALUE_POS.items()
                        }
                    for factor in VALUE_POS:
                        targets = concept_factor_targets(golds, factor, device)
                        pred = logits_by_factor[factor].argmax(-1)
                        misses = pred.ne(targets)
                        wrong_rows |= misses
                        err_count = int(misses.sum())
                        check_count = int(misses.numel())
                        by_metric[active_metric]["checks"] += check_count
                        by_metric[active_metric]["errors"] += err_count
                        by_mode[mode]["checks"] += check_count
                        by_mode[mode]["errors"] += err_count
                        by_factor[factor]["checks"] += check_count
                        by_factor[factor]["errors"] += err_count
                        n_checks += check_count
                        n_errors += err_count
            for i, ex in enumerate(batch_examples):
                if bool(wrong_rows[i]):
                    errors.append(ex)
                else:
                    correct.append(ex)
    return errors, correct, {"n_records": len(examples),
                             "sampled": sampled,
                             "n_error_records": len(errors),
                             "n_correct_records": len(correct),
                             "n_factor_checks": n_checks,
                             "n_errors": n_errors,
                             "factor_error_rate": n_errors / max(1, n_checks),
                             "metric": metric,
                             "active_metrics": list(active_metrics),
                             "modes": list(modes),
                             "by_metric": by_metric,
                             "by_mode": by_mode,
                             "by_factor": by_factor,
                             "skipped": False}


def multimodal_latent_curiosity_examples(
        model, vocab, surfaces, n=0, seed=0, device=DEV, text_split="train",
        mode="full", temperature=0.1, transitive_steps=2, transitive_w=0.1,
        examples=None):
    memory = getattr(model, "latent_concept_memory", None)
    if getattr(model, "latent_concepts", None) is None or memory is None:
        return [], {"n_records": 0, "sampled": False, "n_selected": 0,
                    "mean_score": 0.0, "max_score": 0.0,
                    "mean_novelty": 0.0, "mean_association": 0.0,
                    "mode": mode, "skipped": True}
    if examples is None:
        rng = np.random.default_rng(seed)
        count = int(n) if int(n) > 0 else 1
        examples = _sample_examples(count, rng, surfaces, text_split=text_split)
        sampled = True
    else:
        examples = list(examples)
        sampled = False
        if n and n < len(examples):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(examples), size=int(n), replace=False)
            examples = [examples[int(i)] for i in idx]
            sampled = True
    scored = []
    score_values = []
    novelty_values = []
    association_values = []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(examples), 64):
            batch_examples = examples[off:off + 64]
            img, aud, txt, _ids, _golds = _batch_from_examples(
                batch_examples, vocab, device)
            slots = model.latent_concept_states(
                img, aud, txt, mode=mode, project=True)
            scores, parts = latent_concept_graph_curiosity_scores(
                slots, memory.active(), memory.active_relations(),
                temperature=temperature, transitive_steps=transitive_steps,
                transitive_w=transitive_w)
            novelty = parts.get("novelty", scores.new_zeros(scores.shape))
            association = parts.get("association", scores.new_zeros(scores.shape))
            for i, ex in enumerate(batch_examples):
                score = float(scores[i].detach().cpu())
                scored.append((score, ex))
                score_values.append(score)
                novelty_values.append(float(novelty[i].detach().cpu()))
                association_values.append(float(association[i].detach().cpu()))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [ex for _score, ex in scored]
    return selected, {"n_records": len(examples),
                      "sampled": sampled,
                      "n_selected": len(selected),
                      "mean_score": float(np.mean(score_values)) if score_values else 0.0,
                      "max_score": float(max(score_values)) if score_values else 0.0,
                      "mean_novelty": (
                          float(np.mean(novelty_values)) if novelty_values else 0.0),
                      "mean_association": (
                          float(np.mean(association_values))
                          if association_values else 0.0),
                      "mode": mode,
                      "skipped": False}


def concept_geometry_evaluate(model, vocab, surfaces, n=200, seed=13, device=DEV,
                              text_split="eval", mode="full"):
    """Same-value geometry diagnostic for schema concept states.

    This mirrors the text fact-concept geometry eval: states for the same factor value should be
    nearest neighbors more often than chance and have higher cosine similarity than different
    values for the same factor.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    nearest_correct = {k: 0 for k in VALUE_POS}
    nearest_total = {k: 0 for k in VALUE_POS}
    same_sum = {k: 0.0 for k in VALUE_POS}
    diff_sum = {k: 0.0 for k in VALUE_POS}
    same_pairs = {k: 0 for k in VALUE_POS}
    diff_pairs = {k: 0 for k in VALUE_POS}
    with torch.no_grad():
        for off in range(0, n, 50):
            b = min(50, n - off)
            img, aud, txt, _ids, golds = _batch(b, rng, vocab, device, surfaces,
                                                text_split=text_split)
            states = model.factor_concept_geometry_states(img, aud, txt, mode=mode)
            for factor, state in states.items():
                labels = concept_factor_targets(golds, factor, state.device)
                if int(labels.shape[0]) < 2:
                    continue
                z = F.normalize(state, dim=-1)
                sim = z.matmul(z.t())
                count = int(labels.shape[0])
                eye = torch.eye(count, dtype=torch.bool, device=sim.device)
                same = labels[:, None].eq(labels[None, :]) & ~eye
                diff = labels[:, None].ne(labels[None, :])
                if bool(same.any()):
                    same_sum[factor] += float(sim[same].sum())
                    same_pairs[factor] += int(same.sum())
                if bool(diff.any()):
                    diff_sum[factor] += float(sim[diff].sum())
                    diff_pairs[factor] += int(diff.sum())
                rows = same.any(-1) & diff.any(-1)
                if bool(rows.any()):
                    nearest = sim.masked_fill(eye, -float("inf")).argmax(-1)
                    nearest_correct[factor] += int(labels[nearest][rows].eq(labels[rows]).sum())
                    nearest_total[factor] += int(rows.sum())
    by_factor = {}
    for factor in VALUE_POS:
        same_mean = same_sum[factor] / max(1, same_pairs[factor])
        diff_mean = diff_sum[factor] / max(1, diff_pairs[factor])
        by_factor[factor] = {
            "nearest_same_acc": nearest_correct[factor] / max(1, nearest_total[factor]),
            "same_mean": same_mean,
            "diff_mean": diff_mean,
            "margin": same_mean - diff_mean,
            "n_nearest": int(nearest_total[factor]),
            "same_pairs": int(same_pairs[factor]),
            "diff_pairs": int(diff_pairs[factor]),
        }
    return {
        "by_factor": by_factor,
        "mean": {
            "nearest_same_acc": float(np.mean([
                row["nearest_same_acc"] for row in by_factor.values()
            ])),
            "margin": float(np.mean([row["margin"] for row in by_factor.values()])),
            "same_mean": float(np.mean([row["same_mean"] for row in by_factor.values()])),
            "diff_mean": float(np.mean([row["diff_mean"] for row in by_factor.values()])),
        },
        "n_records": int(n),
    }


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
    if not surfaces.get("eval"):
        return {}
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
    if not surfaces.get("eval"):
        return {}
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


def _candidate_ids(vocab, factor, device):
    return torch.tensor([vocab.stoi[v] for v in FACTOR_VALUES[factor]],
                        dtype=torch.long, device=device)


def value_agreement_loss(logits_by_mode, vocab):
    """Align factor-value distributions across full, sensor-only, and text-only paths."""
    items = list(logits_by_mode.items())
    if len(items) < 2:
        return next(iter(logits_by_mode.values())).sum() * 0.0
    losses = []
    device = items[0][1].device
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            _mode_a, logits_a = items[i]
            _mode_b, logits_b = items[j]
            for factor, pos in VALUE_POS.items():
                cand = _candidate_ids(vocab, factor, device)
                a = logits_a[:, pos - 1].index_select(-1, cand)
                b = logits_b[:, pos - 1].index_select(-1, cand)
                losses.append(0.5 * (
                    F.kl_div(F.log_softmax(a, dim=-1), F.softmax(b.detach(), dim=-1),
                             reduction="batchmean")
                    + F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a.detach(), dim=-1),
                               reduction="batchmean")
                ))
    return torch.stack(losses).mean()


def concept_factor_targets(golds, factor, device):
    return torch.tensor([FACTOR_INDEX[factor][gold[factor]] for gold in golds],
                        dtype=torch.long, device=device)


def concept_factor_target_ids(golds, device):
    return {factor: concept_factor_targets(golds, factor, device) for factor in VALUE_POS}


def concept_factor_loss(factor_logits_by_mode, golds):
    if not factor_logits_by_mode:
        return torch.tensor(0.0)
    first_mode = next(iter(factor_logits_by_mode.values()))
    first_factor = next(iter(first_mode.values()))
    device = first_factor.device
    losses = []
    for logits_by_factor in factor_logits_by_mode.values():
        for factor, logits in logits_by_factor.items():
            targets = concept_factor_targets(golds, factor, device)
            losses.append(F.cross_entropy(logits, targets))
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def concept_factor_contrastive_loss(factor_states_by_mode, golds, temperature=0.1):
    if not factor_states_by_mode:
        return torch.tensor(0.0)
    first_mode = next(iter(factor_states_by_mode.values()))
    first_factor = next(iter(first_mode.values()))
    device = first_factor.device
    targets = concept_factor_target_ids(golds, device)
    losses = [
        schema_concept_contrastive_loss(states, targets, temperature=temperature)
        for states in factor_states_by_mode.values()
    ]
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def concept_factor_centroid_loss(factor_geometry_states_by_mode, golds, temperature=0.1,
                                 margin=0.0):
    if not factor_geometry_states_by_mode:
        return torch.tensor(0.0)
    first_mode = next(iter(factor_geometry_states_by_mode.values()))
    first_factor = next(iter(first_mode.values()))
    device = first_factor.device
    targets = concept_factor_target_ids(golds, device)
    losses = [
        schema_concept_batch_centroid_loss(
            states, targets, temperature=temperature, margin=margin)
        for states in factor_geometry_states_by_mode.values()
    ]
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def concept_factor_prototype_loss(model, factor_geometry_states_by_mode, golds,
                                  temperature=0.1):
    if not factor_geometry_states_by_mode:
        return torch.tensor(0.0)
    first_mode = next(iter(factor_geometry_states_by_mode.values()))
    first_factor = next(iter(first_mode.values()))
    device = first_factor.device
    targets = concept_factor_target_ids(golds, device)
    prototypes = {
        factor: model.factor_concepts.geometry_prototypes[i]
        for i, factor in enumerate(VALUE_POS)
    }
    losses = [
        schema_concept_prototype_alignment_loss(
            states, targets, prototypes, temperature=temperature)
        for states in factor_geometry_states_by_mode.values()
    ]
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def concept_factor_prototype_spread_loss(model, margin=0.2):
    prototypes = {
        factor: model.factor_concepts.geometry_prototypes[i]
        for i, factor in enumerate(VALUE_POS)
    }
    return schema_concept_prototype_spread_loss(prototypes, margin=margin)


def concept_factor_state_spread_loss(factor_geometry_states_by_mode, golds,
                                     variance_target=0.05,
                                     centroid_margin=0.2,
                                     covariance_weight=0.05):
    if not factor_geometry_states_by_mode:
        return torch.tensor(0.0)
    first_mode = next(iter(factor_geometry_states_by_mode.values()))
    first_factor = next(iter(first_mode.values()))
    device = first_factor.device
    targets = concept_factor_target_ids(golds, device)
    losses = [
        schema_concept_state_spread_loss(
            states, targets, variance_target=variance_target,
            centroid_margin=centroid_margin, covariance_weight=covariance_weight)
        for states in factor_geometry_states_by_mode.values()
    ]
    if not losses:
        return torch.tensor(0.0, device=device)
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


def latent_multimodal_memory_loss_from_views(model, views, temperature=0.1,
                                             balance_w=0.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    if getattr(model, "latent_concept_memory", None) is None:
        return next(iter(views.values())).sum() * 0.0
    losses = [
        model.latent_concept_memory_loss(
            slots, temperature=temperature, balance_w=balance_w)
        for slots in views.values()
    ]
    return torch.stack(losses).mean() if losses else next(iter(views.values())).sum() * 0.0


def latent_multimodal_association_loss_from_views(
        model, views, temperature=0.1, target_power=1.0, self_loop_w=0.05,
        transitive_steps=1, transitive_w=0.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    if getattr(model, "latent_concept_memory", None) is None:
        return next(iter(views.values())).sum() * 0.0
    losses = [
        model.latent_concept_association_loss(
            slots, temperature=temperature, target_power=target_power,
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
    if getattr(model, "latent_concept_memory", None) is None:
        return next(iter(views.values())).sum() * 0.0
    losses = []
    for slots in views.values():
        if hasattr(model, "latent_concept_composition_loss"):
            losses.append(model.latent_concept_composition_loss(
                slots, temperature=temperature, self_loop_w=self_loop_w,
                transitive_steps=transitive_steps, transitive_w=transitive_w,
                margin=margin))
        else:
            losses.append(latent_concept_composition_loss(
                slots, model.latent_concept_memory.active(),
                model.latent_concept_memory.active_relations(),
                temperature=temperature, self_loop_w=self_loop_w,
                transitive_steps=transitive_steps, transitive_w=transitive_w,
                margin=margin))
    return torch.stack(losses).mean() if losses else next(iter(views.values())).sum() * 0.0


def latent_multimodal_graph_prediction_loss_from_views(
        model, views, temperature=0.1, self_loop_w=0.05, transitive_steps=2,
        transitive_w=0.1, target_power=1.0):
    views = {mode: slots for mode, slots in views.items() if slots is not None}
    if not views:
        return torch.tensor(0.0)
    if getattr(model, "latent_concept_memory", None) is None:
        return next(iter(views.values())).sum() * 0.0
    target = views.get("full")
    if target is None:
        return next(iter(views.values())).sum() * 0.0
    losses = []
    for mode, source in views.items():
        if mode == "full":
            continue
        if hasattr(model, "latent_concept_graph_prediction_loss"):
            losses.append(model.latent_concept_graph_prediction_loss(
                source, target.detach(), temperature=temperature,
                self_loop_w=self_loop_w, transitive_steps=transitive_steps,
                transitive_w=transitive_w, target_power=target_power))
        else:
            losses.append(latent_concept_graph_prediction_loss(
                source, target.detach(), model.latent_concept_memory.active(),
                model.latent_concept_memory.active_relations(),
                temperature=temperature, self_loop_w=self_loop_w,
                transitive_steps=transitive_steps, transitive_w=transitive_w,
                target_power=target_power))
    return torch.stack(losses).mean() if losses else target.sum() * 0.0


@torch.no_grad()
def update_multimodal_latent_memory(model, slots, momentum=0.95,
                                    relation_decay=None):
    if getattr(model, "latent_concept_memory", None) is None:
        return 0
    return model.update_latent_concept_memory(
        slots, momentum=momentum, relation_decay=relation_decay)


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


def latent_multimodal_concept_loss(model, img, aud, txt, view_dropout=0.1,
                                   invariance_w=25.0, variance_w=25.0,
                                   covariance_w=1.0, variance_target=1.0):
    if getattr(model, "latent_concepts", None) is None:
        return img.sum() * 0.0
    views = {
        mode: model.latent_concept_states(
            img, aud, txt, mode=mode, view_dropout=view_dropout, project=True)
        for mode in MODES
    }
    return latent_multimodal_concept_loss_from_views(
        views, invariance_w=invariance_w, variance_w=variance_w,
        covariance_w=covariance_w, variance_target=variance_target)


def latent_factor_loss(latent_factor_logits_by_mode, golds):
    if not latent_factor_logits_by_mode:
        return torch.tensor(0.0)
    return concept_factor_loss(latent_factor_logits_by_mode, golds)


def concept_factor_agreement_loss(factor_logits_by_mode):
    items = list(factor_logits_by_mode.items())
    if len(items) < 2:
        return next(iter(next(iter(factor_logits_by_mode.values())).values())).sum() * 0.0
    losses = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            _mode_a, factors_a = items[i]
            _mode_b, factors_b = items[j]
            for factor in VALUE_POS:
                a = factors_a[factor]
                b = factors_b[factor]
                losses.append(0.5 * (
                    F.kl_div(F.log_softmax(a, dim=-1), F.softmax(b.detach(), dim=-1),
                             reduction="batchmean")
                    + F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a.detach(), dim=-1),
                               reduction="batchmean")
                ))
    return torch.stack(losses).mean()


def concept_full_distill_loss(factor_logits_by_mode, temperature=1.0):
    if "full" not in factor_logits_by_mode:
        return next(iter(next(iter(factor_logits_by_mode.values())).values())).sum() * 0.0
    temp = max(float(temperature), 1e-6)
    full = factor_logits_by_mode["full"]
    losses = []
    for mode, logits_by_factor in factor_logits_by_mode.items():
        if mode == "full":
            continue
        for factor in VALUE_POS:
            teacher = F.softmax(full[factor].detach() / temp, dim=-1)
            student = F.log_softmax(logits_by_factor[factor] / temp, dim=-1)
            losses.append(F.kl_div(student, teacher, reduction="batchmean") * temp * temp)
    if not losses:
        return next(iter(full.values())).sum() * 0.0
    return torch.stack(losses).mean()


def concept_full_rank_distill_loss(factor_logits_by_mode, golds, margin=0.0):
    """Preserve full-path correct concept winners and margins in partial modes."""
    if "full" not in factor_logits_by_mode:
        return next(iter(next(iter(factor_logits_by_mode.values())).values())).sum() * 0.0
    full = factor_logits_by_mode["full"]
    device = next(iter(full.values())).device
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=device)
    losses = []
    for factor in VALUE_POS:
        teacher_logits = full[factor].detach()
        if teacher_logits.shape[-1] < 2:
            continue
        targets = concept_factor_targets(golds, factor, device)
        correct = teacher_logits.argmax(-1).eq(targets)
        if not bool(correct.any()):
            continue
        teacher_rows = teacher_logits[correct]
        target_rows = targets[correct]
        row_idx = torch.arange(target_rows.shape[0], device=device)
        other_mask = torch.ones_like(teacher_rows, dtype=torch.bool)
        other_mask[row_idx, target_rows] = False
        teacher_target = teacher_rows[row_idx, target_rows]
        teacher_other = teacher_rows.masked_fill(~other_mask, -float("inf")).max(-1).values
        teacher_margin = (teacher_target - teacher_other).clamp_min(0.0)
        desired_margin = torch.maximum(teacher_margin, margin_t.expand_as(teacher_margin))
        for mode, logits_by_factor in factor_logits_by_mode.items():
            if mode == "full":
                continue
            student_rows = logits_by_factor[factor][correct]
            student_target = student_rows[row_idx, target_rows]
            student_other = student_rows.masked_fill(~other_mask, -float("inf")).max(-1).values
            losses.append(F.cross_entropy(student_rows, target_rows))
            losses.append(F.softplus(student_other + desired_margin - student_target).mean())
    if not losses:
        return next(iter(full.values())).sum() * 0.0
    return torch.stack(losses).mean()


def concept_state_transfer_loss(factor_states_by_mode, factor_logits_by_mode, golds,
                                margin=0.0, teacher_mode="full"):
    """Move partial-mode concept states toward detached full-mode states on learned errors.

    This is the multimodal analog of the text study transfer bridge: the stable side is detached,
    and the rows are chosen from current model behavior plus batch labels, not hand-authored rules.
    """
    if teacher_mode not in factor_states_by_mode or teacher_mode not in factor_logits_by_mode:
        return next(iter(next(iter(factor_logits_by_mode.values())).values())).sum() * 0.0
    teacher_states = factor_states_by_mode[teacher_mode]
    teacher_logits = factor_logits_by_mode[teacher_mode]
    device = next(iter(teacher_logits.values())).device
    margin_t = torch.tensor(float(margin), dtype=torch.float32, device=device)
    losses = []
    for factor in VALUE_POS:
        targets = concept_factor_targets(golds, factor, device)
        teacher_correct = teacher_logits[factor].argmax(-1).eq(targets)
        if not bool(teacher_correct.any()):
            continue
        for mode, states in factor_states_by_mode.items():
            if mode == teacher_mode:
                continue
            student_logits = factor_logits_by_mode[mode][factor]
            student_hard = ~student_logits.argmax(-1).eq(targets)
            selected = teacher_correct & student_hard
            if not bool(selected.any()):
                selected = teacher_correct
            teacher_vec = F.normalize(teacher_states[factor][selected], dim=-1).detach()
            student_vec = F.normalize(states[factor][selected], dim=-1)
            positive = (student_vec * teacher_vec).sum(-1)
            negatives = []
            for other_factor, other_state in states.items():
                if other_factor != factor:
                    negatives.append(
                        (F.normalize(other_state[selected], dim=-1) * teacher_vec).sum(-1))
            for other_factor, other_state in teacher_states.items():
                if other_factor != factor:
                    negatives.append(
                        (student_vec * F.normalize(other_state[selected], dim=-1).detach()
                         ).sum(-1))
            losses.append(F.softplus(margin_t - positive).mean())
            if negatives:
                negative = torch.stack(negatives, dim=0).max(dim=0).values
                losses.append(F.softplus(negative + margin_t - positive).mean())
    if not losses:
        return next(iter(teacher_logits.values())).sum() * 0.0
    return torch.stack(losses).mean()


def train(steps=400, batch=32, d=96, lr=1e-3, seed=0, device=DEV, log_every=100, value_w=6.0,
          surfaces_path=None, layers=3, heads=4, max_len=128, img_tokens=4, aud_tokens=8,
          txt_tokens=8, trunk_arch="conv", trunk_width=64, trunk_depth=1, text_layers=1,
          text_arch="transformer", modality_dropout=0.0, agreement_w=0.0,
          text_checkpoint=None,
          concept_tokens=4, fusion_layers=1,
          concept_prefix=False, concept_refine=False,
          concept_refine_gate_init=-2.0,
          concept_mixer_layers=0, concept_mixer_gate_init=-2.0,
          latent_concept_slots=0, latent_concept_layers=1,
          latent_concept_prefix=False, latent_concept_refine=False,
          latent_concept_refine_gate_init=-2.0,
          latent_concept_w=0.0, latent_concept_view_dropout=0.1,
          latent_concept_invariance_w=25.0, latent_concept_variance_w=25.0,
          latent_concept_covariance_w=1.0, latent_concept_variance_target=1.0,
          latent_concept_factorization_w=0.0,
          latent_concept_factorization_variance=0.05,
          latent_concept_factorization_margin=0.2,
          latent_concept_factorization_covariance_w=0.05,
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
          latent_concept_factor_w=0.0,
          study_strategy="random", study_score_metric="balanced",
          study_probe_n=0, study_hard_max=0, study_refresh_steps=0,
          concept_w=0.0, concept_agreement_w=0.0,
          concept_distill_w=0.0, concept_distill_temperature=1.0,
          concept_rank_distill_w=0.0, concept_rank_distill_margin=0.0,
          concept_transfer_w=0.0, concept_transfer_margin=0.0,
          concept_contrast_w=0.0, concept_contrast_temperature=0.1,
          concept_centroid_w=0.0, concept_centroid_temperature=0.1,
          concept_centroid_margin=0.0,
          concept_prototype_w=0.0, concept_prototype_spread_w=0.0,
          concept_prototype_spread_margin=0.2,
          concept_state_spread_w=0.0, concept_state_spread_variance=0.05,
          concept_state_spread_margin=0.2,
          concept_state_spread_covariance_w=0.05):
    ckpt_latents = text_checkpoint_latent_config(
        text_checkpoint, device="cpu") if text_checkpoint else {}
    if text_checkpoint and latent_concept_slots <= 0:
        if ckpt_latents.get("latent_concept_slots", 0) > 0:
            latent_concept_slots = ckpt_latents["latent_concept_slots"]
            latent_concept_layers = ckpt_latents["latent_concept_layers"]
    if (text_checkpoint and latent_concept_memory_size <= 0
            and ckpt_latents.get("latent_concept_memory_size", 0) > 0):
        latent_concept_memory_size = ckpt_latents["latent_concept_memory_size"]
    if latent_concept_association_w > 0.0 and latent_concept_memory_size <= 0:
        raise ValueError("latent concept association requires latent concept memory")
    if (latent_concept_w < 0.0 or latent_concept_factor_w < 0.0
            or latent_concept_factorization_w < 0.0
            or latent_concept_memory_w < 0.0
            or latent_concept_association_w < 0.0
            or latent_concept_composition_w < 0.0
            or latent_concept_graph_predict_w < 0.0
            or latent_concept_neighborhood_w < 0.0
            or latent_concept_transition_w < 0.0
            or latent_concept_cluster_w < 0.0):
        raise ValueError("latent concept loss weights must be non-negative")
    if ((latent_concept_w > 0.0 or latent_concept_factor_w > 0.0
         or latent_concept_factorization_w > 0.0
         or latent_concept_memory_w > 0.0
         or latent_concept_association_w > 0.0
         or latent_concept_composition_w > 0.0
         or latent_concept_graph_predict_w > 0.0
         or latent_concept_memory_size > 0
         or latent_concept_neighborhood_w > 0.0
         or latent_concept_transition_w > 0.0
         or latent_concept_cluster_w > 0.0)
            and latent_concept_slots <= 0):
        raise ValueError("latent concept losses require latent_concept_slots > 0")
    if latent_concept_factorization_variance < 0.0:
        raise ValueError("latent concept factorization variance must be non-negative")
    if latent_concept_factorization_margin < 0.0:
        raise ValueError("latent concept factorization margin must be non-negative")
    if latent_concept_factorization_covariance_w < 0.0:
        raise ValueError(
            "latent concept factorization covariance weight must be non-negative")
    if int(latent_concept_memory_size) < 0:
        raise ValueError("latent concept memory size must be non-negative")
    if latent_concept_memory_temperature <= 0.0:
        raise ValueError("latent concept memory temperature must be positive")
    if (latent_concept_memory_momentum < 0.0
            or latent_concept_memory_momentum >= 1.0):
        raise ValueError("latent concept memory momentum must be in [0, 1)")
    if latent_concept_memory_balance_w < 0.0:
        raise ValueError("latent concept memory balance weight must be non-negative")
    if latent_concept_association_temperature <= 0.0:
        raise ValueError("latent concept association temperature must be positive")
    if (latent_concept_association_decay < 0.0
            or latent_concept_association_decay >= 1.0):
        raise ValueError("latent concept association decay must be in [0, 1)")
    if latent_concept_association_target_power <= 0.0:
        raise ValueError("latent concept association target power must be positive")
    if latent_concept_association_self_loop_w < 0.0:
        raise ValueError(
            "latent concept association self-loop weight must be non-negative")
    if int(latent_concept_association_transitive_steps) < 1:
        raise ValueError(
            "latent concept association transitive steps must be positive")
    if latent_concept_association_transitive_w < 0.0:
        raise ValueError(
            "latent concept association transitive weight must be non-negative")
    if latent_concept_composition_w > 0.0 and latent_concept_memory_size <= 0:
        raise ValueError("latent concept composition requires latent concept memory")
    if latent_concept_composition_temperature <= 0.0:
        raise ValueError("latent concept composition temperature must be positive")
    if latent_concept_composition_self_loop_w < 0.0:
        raise ValueError(
            "latent concept composition self-loop weight must be non-negative")
    if int(latent_concept_composition_transitive_steps) < 1:
        raise ValueError(
            "latent concept composition transitive steps must be positive")
    if latent_concept_composition_transitive_w < 0.0:
        raise ValueError(
            "latent concept composition transitive weight must be non-negative")
    if latent_concept_composition_margin < 0.0:
        raise ValueError("latent concept composition margin must be non-negative")
    if latent_concept_graph_predict_w > 0.0 and latent_concept_memory_size <= 0:
        raise ValueError("latent concept graph prediction requires latent concept memory")
    if latent_concept_graph_predict_temperature <= 0.0:
        raise ValueError("latent concept graph prediction temperature must be positive")
    if latent_concept_graph_predict_self_loop_w < 0.0:
        raise ValueError(
            "latent concept graph prediction self-loop weight must be non-negative")
    if int(latent_concept_graph_predict_transitive_steps) < 1:
        raise ValueError(
            "latent concept graph prediction transitive steps must be positive")
    if latent_concept_graph_predict_transitive_w < 0.0:
        raise ValueError(
            "latent concept graph prediction transitive weight must be non-negative")
    if latent_concept_graph_predict_target_power <= 0.0:
        raise ValueError("latent concept graph prediction target power must be positive")
    if latent_concept_neighborhood_temperature <= 0.0:
        raise ValueError("latent concept neighborhood temperature must be positive")
    if latent_concept_neighborhood_margin < 0.0:
        raise ValueError("latent concept neighborhood margin must be non-negative")
    if latent_concept_transition_temperature <= 0.0:
        raise ValueError("latent concept transition temperature must be positive")
    if latent_concept_transition_margin < 0.0:
        raise ValueError("latent concept transition margin must be non-negative")
    if latent_concept_cluster_temperature <= 0.0:
        raise ValueError("latent concept cluster temperature must be positive")
    if latent_concept_cluster_margin < 0.0:
        raise ValueError("latent concept cluster margin must be non-negative")
    if int(latent_concept_cluster_min_size) < 2:
        raise ValueError("latent concept cluster min size must be at least two")
    study_strategy = str(study_strategy)
    study_score_metric = str(study_score_metric)
    if study_strategy not in ("random", "errors", "curiosity"):
        raise ValueError(f"unknown multimodal study strategy {study_strategy!r}")
    if study_score_metric not in MULTIMODAL_STUDY_METRICS:
        raise ValueError(f"unknown multimodal study score metric {study_score_metric!r}")
    if study_strategy == "errors" and study_score_metric == "latent" and latent_concept_slots <= 0:
        raise ValueError("latent multimodal study requires latent_concept_slots > 0")
    if study_strategy == "curiosity" and latent_concept_memory_size <= 0:
        raise ValueError("multimodal curiosity study requires latent concept memory")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    surfaces = load_text_surfaces(surfaces_path)
    vocab = build_vocab(surfaces)
    model = MultimodalLM(len(vocab), d=d, layers=layers, heads=heads, pad=vocab.pad,
                         max_len=max_len, img_tokens=img_tokens, aud_tokens=aud_tokens,
                         txt_tokens=txt_tokens, trunk_arch=trunk_arch, trunk_width=trunk_width,
                         trunk_depth=trunk_depth, text_layers=text_layers,
                         text_arch=text_arch, modality_dropout=modality_dropout,
                         concept_tokens=concept_tokens,
                         fusion_layers=fusion_layers,
                         concept_prefix=concept_prefix,
                         concept_refine=concept_refine,
                         concept_refine_gate_init=(
                             concept_refine_gate_init),
                         concept_mixer_layers=concept_mixer_layers,
                         concept_mixer_gate_init=concept_mixer_gate_init,
                         latent_concept_slots=latent_concept_slots,
                         latent_concept_layers=latent_concept_layers,
                         latent_concept_prefix=latent_concept_prefix,
                         latent_concept_refine=latent_concept_refine,
                         latent_concept_refine_gate_init=(
                             latent_concept_refine_gate_init),
                         latent_concept_memory_size=(
                             latent_concept_memory_size)).to(device)
    if text_checkpoint:
        import_text_checkpoint(model, vocab, text_checkpoint, device=device)
    else:
        model.text_checkpoint_transfer = {}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    study_reports = []
    study_pool = []

    def refresh_study_pool(step):
        nonlocal study_pool
        probe_n = int(study_probe_n) if int(study_probe_n) > 0 else max(batch * 4, 1)
        if study_strategy == "curiosity":
            selected, hard_report = multimodal_latent_curiosity_examples(
                model, vocab, surfaces, n=probe_n, seed=seed + 1009 + int(step),
                device=device, text_split="train", mode="full",
                temperature=latent_concept_association_temperature,
                transitive_steps=latent_concept_association_transitive_steps,
                transitive_w=latent_concept_association_transitive_w)
            hard_report = hard_report | {"strategy": "curiosity"}
        else:
            errors, _correct, hard_report = multimodal_factor_record_outcomes(
                model, vocab, surfaces, n=probe_n, seed=seed + 1009 + int(step),
                device=device, text_split="train", metric=study_score_metric)
            selected = list(errors)
            hard_report = hard_report | {"strategy": "errors"}
        if study_hard_max and len(selected) > int(study_hard_max):
            if study_strategy == "curiosity":
                selected = selected[:int(study_hard_max)]
            else:
                cap_rng = np.random.default_rng(seed + 2003 + int(step))
                idx = cap_rng.choice(len(selected), size=int(study_hard_max), replace=False)
                selected = [selected[int(i)] for i in idx]
            hard_report = hard_report | {"capped": True,
                                         "n_error_records_used": len(selected)}
        else:
            hard_report = hard_report | {"capped": False,
                                         "n_error_records_used": len(selected)}
        if selected:
            study_pool = selected
        hard_report = hard_report | {"step": int(step),
                                     "pool_active": bool(study_pool),
                                     "pool_size": len(study_pool)}
        study_reports.append(hard_report)

    last_base = last_agreement = last_concept = 0.0
    last_concept_agreement = last_concept_distill = last_concept_rank_distill = 0.0
    last_concept_transfer = last_concept_contrast = 0.0
    last_concept_centroid = 0.0
    last_concept_prototype = last_concept_prototype_spread = 0.0
    last_concept_state_spread = 0.0
    last_latent_concept = 0.0
    last_latent_factorization = 0.0
    last_latent_memory = 0.0
    last_latent_memory_updates = 0
    last_latent_association = 0.0
    last_latent_composition = 0.0
    last_latent_graph_predict = 0.0
    last_latent_neighborhood = 0.0
    last_latent_transition = 0.0
    last_latent_cluster = 0.0
    last_latent_factor = 0.0
    for st in range(1, steps + 1):
        model.train()
        if study_strategy in ("errors", "curiosity") and (st == 1 or (
                study_refresh_steps and (st - 1) % int(study_refresh_steps) == 0)):
            refresh_study_pool(st)
            model.train()
        if study_strategy in ("errors", "curiosity") and study_pool:
            img, aud, txt, ids, golds = _batch_from_examples(
                _sample_from_pool(study_pool, batch, rng), vocab, device)
        else:
            img, aud, txt, ids, golds = _batch(batch, rng, vocab, device, surfaces,
                                               text_split="train")
        needs_factor_batch = (
            concept_w or concept_agreement_w or concept_distill_w
            or concept_rank_distill_w or concept_transfer_w
            or concept_contrast_w or concept_centroid_w or concept_prototype_w
            or concept_state_spread_w)
        needs_geometry_batch = (
            concept_contrast_w or concept_centroid_w or concept_prototype_w
            or concept_state_spread_w)
        needs_latent_batch = bool(
            latent_concept_w or latent_concept_factorization_w
            or latent_concept_memory_w
            or latent_concept_association_w
            or latent_concept_composition_w
            or latent_concept_graph_predict_w
            or latent_concept_neighborhood_w
            or latent_concept_transition_w
            or latent_concept_cluster_w)
        needs_latent_factor_batch = bool(latent_concept_factor_w)
        bundles_by_mode = {
            mode: model.mode_bundle(
                img, aud, txt, ids, mode=mode, need_factor=needs_factor_batch,
                need_geometry=needs_geometry_batch, need_latent=needs_latent_batch,
                need_latent_factor=needs_latent_factor_batch,
                latent_view_dropout=latent_concept_view_dropout,
                latent_project=True)
            for mode in MODES
        }
        logits_by_mode = {mode: bundle["logits"]
                          for mode, bundle in bundles_by_mode.items()}
        losses = [token_loss(logits, ids, value_w=value_w)
                  for logits in logits_by_mode.values()]
        base_loss = sum(losses) / len(losses)
        agreement = (value_agreement_loss(logits_by_mode, vocab)
                     if agreement_w else base_loss * 0.0)
        if needs_factor_batch:
            factor_states_by_mode = {
                mode: bundle["factor_states"] for mode, bundle in bundles_by_mode.items()
            }
            factor_logits_by_mode = {
                mode: bundle["factor_logits"] for mode, bundle in bundles_by_mode.items()
            }
            concept_loss = (
                concept_factor_loss(factor_logits_by_mode, golds)
                if concept_w else base_loss * 0.0)
            concept_agreement = (
                concept_factor_agreement_loss(factor_logits_by_mode)
                if concept_agreement_w else base_loss * 0.0)
            concept_distill = (
                concept_full_distill_loss(
                    factor_logits_by_mode, temperature=concept_distill_temperature)
                if concept_distill_w else base_loss * 0.0)
            concept_rank_distill = (
                concept_full_rank_distill_loss(
                    factor_logits_by_mode, golds, margin=concept_rank_distill_margin)
                if concept_rank_distill_w else base_loss * 0.0)
            concept_transfer = (
                concept_state_transfer_loss(
                    factor_states_by_mode, factor_logits_by_mode, golds,
                    margin=concept_transfer_margin)
                if concept_transfer_w else base_loss * 0.0)
            factor_geometry_states_by_mode = (
                {mode: bundle["factor_geometry_states"]
                 for mode, bundle in bundles_by_mode.items()}
                if needs_geometry_batch else {})
            concept_contrast = (
                concept_factor_contrastive_loss(
                    factor_geometry_states_by_mode, golds,
                    temperature=concept_contrast_temperature)
                if concept_contrast_w else base_loss * 0.0)
            concept_centroid = (
                concept_factor_centroid_loss(
                    factor_geometry_states_by_mode, golds,
                    temperature=concept_centroid_temperature,
                    margin=concept_centroid_margin)
                if concept_centroid_w else base_loss * 0.0)
            concept_prototype = (
                concept_factor_prototype_loss(
                    model, factor_geometry_states_by_mode, golds,
                    temperature=concept_contrast_temperature)
                if concept_prototype_w else base_loss * 0.0)
            concept_state_spread = (
                concept_factor_state_spread_loss(
                    factor_geometry_states_by_mode, golds,
                    variance_target=concept_state_spread_variance,
                    centroid_margin=concept_state_spread_margin,
                    covariance_weight=concept_state_spread_covariance_w)
                if concept_state_spread_w else base_loss * 0.0)
        else:
            concept_loss = base_loss * 0.0
            concept_agreement = base_loss * 0.0
            concept_distill = base_loss * 0.0
            concept_rank_distill = base_loss * 0.0
            concept_transfer = base_loss * 0.0
            concept_contrast = base_loss * 0.0
            concept_centroid = base_loss * 0.0
            concept_prototype = base_loss * 0.0
            concept_state_spread = base_loss * 0.0
        latent_concept = (
            latent_multimodal_concept_loss_from_views(
                {mode: bundle["latent_concepts"]
                 for mode, bundle in bundles_by_mode.items()},
                invariance_w=latent_concept_invariance_w,
                variance_w=latent_concept_variance_w,
                covariance_w=latent_concept_covariance_w,
                variance_target=latent_concept_variance_target)
            if latent_concept_w else base_loss * 0.0)
        latent_factorization = (
            latent_multimodal_factorization_loss_from_views(
                {mode: bundle["latent_concepts"]
                 for mode, bundle in bundles_by_mode.items()},
                variance_target=latent_concept_factorization_variance,
                separation_margin=latent_concept_factorization_margin,
                covariance_w=latent_concept_factorization_covariance_w)
            if latent_concept_factorization_w else base_loss * 0.0)
        latent_memory = (
            latent_multimodal_memory_loss_from_views(
                model,
                {mode: bundle["latent_concepts"]
                 for mode, bundle in bundles_by_mode.items()},
                temperature=latent_concept_memory_temperature,
                balance_w=latent_concept_memory_balance_w)
            if latent_concept_memory_w else base_loss * 0.0)
        latent_association = (
            latent_multimodal_association_loss_from_views(
                model,
                {mode: bundle["latent_concepts"]
                 for mode, bundle in bundles_by_mode.items()},
                temperature=latent_concept_association_temperature,
                target_power=latent_concept_association_target_power,
                self_loop_w=latent_concept_association_self_loop_w,
                transitive_steps=latent_concept_association_transitive_steps,
                transitive_w=latent_concept_association_transitive_w)
            if latent_concept_association_w else base_loss * 0.0)
        latent_composition = (
            latent_multimodal_composition_loss_from_views(
                model,
                {mode: bundle["latent_concepts"]
                 for mode, bundle in bundles_by_mode.items()},
                temperature=latent_concept_composition_temperature,
                self_loop_w=latent_concept_composition_self_loop_w,
                transitive_steps=latent_concept_composition_transitive_steps,
                transitive_w=latent_concept_composition_transitive_w,
                margin=latent_concept_composition_margin)
            if latent_concept_composition_w else base_loss * 0.0)
        latent_graph_predict = (
            latent_multimodal_graph_prediction_loss_from_views(
                model,
                {mode: bundle["latent_concepts"]
                 for mode, bundle in bundles_by_mode.items()},
                temperature=latent_concept_graph_predict_temperature,
                self_loop_w=latent_concept_graph_predict_self_loop_w,
                transitive_steps=latent_concept_graph_predict_transitive_steps,
                transitive_w=latent_concept_graph_predict_transitive_w,
                target_power=latent_concept_graph_predict_target_power)
            if latent_concept_graph_predict_w else base_loss * 0.0)
        latent_neighborhood = (
            latent_multimodal_neighborhood_loss_from_views(
                {mode: bundle["latent_concepts"]
                 for mode, bundle in bundles_by_mode.items()},
                temperature=latent_concept_neighborhood_temperature,
                margin=latent_concept_neighborhood_margin)
            if latent_concept_neighborhood_w else base_loss * 0.0)
        latent_transition = (
            latent_multimodal_transition_loss_from_views(
                {mode: bundle["latent_concepts"]
                 for mode, bundle in bundles_by_mode.items()},
                temperature=latent_concept_transition_temperature,
                margin=latent_concept_transition_margin)
            if latent_concept_transition_w else base_loss * 0.0)
        latent_cluster = (
            latent_multimodal_cluster_loss_from_views(
                {mode: bundle["latent_concepts"]
                 for mode, bundle in bundles_by_mode.items()},
                temperature=latent_concept_cluster_temperature,
                margin=latent_concept_cluster_margin,
                min_cluster_size=latent_concept_cluster_min_size)
            if latent_concept_cluster_w else base_loss * 0.0)
        latent_factor = (
            latent_factor_loss(
                {mode: bundle["latent_factor_logits"]
                 for mode, bundle in bundles_by_mode.items()},
                golds)
            if latent_concept_factor_w else base_loss * 0.0)
        concept_prototype_spread = (
            concept_factor_prototype_spread_loss(
                model, margin=concept_prototype_spread_margin)
            if concept_prototype_spread_w else base_loss * 0.0)
        loss = (base_loss + float(agreement_w) * agreement
                + float(concept_w) * concept_loss
                + float(concept_agreement_w) * concept_agreement
                + float(concept_distill_w) * concept_distill
                + float(concept_rank_distill_w) * concept_rank_distill
                + float(concept_transfer_w) * concept_transfer
                + float(concept_contrast_w) * concept_contrast
                + float(concept_centroid_w) * concept_centroid
                + float(concept_prototype_w) * concept_prototype
                + float(concept_prototype_spread_w) * concept_prototype_spread
                + float(concept_state_spread_w) * concept_state_spread
                + float(latent_concept_w) * latent_concept
                + float(latent_concept_factorization_w) * latent_factorization
                + float(latent_concept_memory_w) * latent_memory
                + float(latent_concept_association_w) * latent_association
                + float(latent_concept_composition_w) * latent_composition
                + float(latent_concept_graph_predict_w) * latent_graph_predict
                + float(latent_concept_neighborhood_w) * latent_neighborhood
                + float(latent_concept_transition_w) * latent_transition
                + float(latent_concept_cluster_w) * latent_cluster
                + float(latent_concept_factor_w) * latent_factor)
        opt.zero_grad()
        loss.backward()
        opt.step()
        full_latent_for_memory = (
            bundles_by_mode.get("full", {}).get("latent_concepts")
            if needs_latent_batch else None)
        last_latent_memory_updates = int(update_multimodal_latent_memory(
            model, full_latent_for_memory, momentum=latent_concept_memory_momentum,
            relation_decay=(latent_concept_association_decay
                            if (latent_concept_association_w
                                or latent_concept_composition_w
                                or latent_concept_graph_predict_w) else None)))
        last_base = float(base_loss.detach())
        last_agreement = float(agreement.detach())
        last_concept = float(concept_loss.detach())
        last_concept_agreement = float(concept_agreement.detach())
        last_concept_distill = float(concept_distill.detach())
        last_concept_rank_distill = float(concept_rank_distill.detach())
        last_concept_transfer = float(concept_transfer.detach())
        last_concept_contrast = float(concept_contrast.detach())
        last_concept_centroid = float(concept_centroid.detach())
        last_concept_prototype = float(concept_prototype.detach())
        last_concept_prototype_spread = float(concept_prototype_spread.detach())
        last_concept_state_spread = float(concept_state_spread.detach())
        last_latent_concept = float(latent_concept.detach())
        last_latent_factorization = float(latent_factorization.detach())
        last_latent_memory = float(latent_memory.detach())
        last_latent_association = float(latent_association.detach())
        last_latent_composition = float(latent_composition.detach())
        last_latent_graph_predict = float(latent_graph_predict.detach())
        last_latent_neighborhood = float(latent_neighborhood.detach())
        last_latent_transition = float(latent_transition.detach())
        last_latent_cluster = float(latent_cluster.detach())
        last_latent_factor = float(latent_factor.detach())
        if st % log_every == 0 or st == steps:
            print(f"  m0 {st}/{steps} loss {loss.item():.3f} "
                  f"base {last_base:.3f} agree {last_agreement:.3f} "
                  f"concept {last_concept:.3f} "
                  f"concept-agree {last_concept_agreement:.3f} "
                  f"concept-distill {last_concept_distill:.3f} "
                  f"concept-rank {last_concept_rank_distill:.3f} "
                  f"concept-transfer {last_concept_transfer:.3f} "
                  f"concept-contrast {last_concept_contrast:.3f} "
                  f"concept-centroid {last_concept_centroid:.3f} "
                  f"concept-proto {last_concept_prototype:.3f} "
                  f"concept-proto-spread {last_concept_prototype_spread:.3f} "
                  f"concept-state-spread {last_concept_state_spread:.3f} "
                  f"latent {last_latent_concept:.3f} "
                  f"latent-factorize {last_latent_factorization:.3f} "
                  f"latent-memory {last_latent_memory:.3f} "
                  f"latent-assoc {last_latent_association:.3f} "
                  f"latent-compose {last_latent_composition:.3f} "
                  f"latent-graph-predict {last_latent_graph_predict:.3f} "
                  f"latent-neighborhood {last_latent_neighborhood:.3f} "
                  f"latent-transition {last_latent_transition:.3f} "
                  f"latent-cluster {last_latent_cluster:.3f} "
                  f"latent-factor {last_latent_factor:.3f}",
                  flush=True)
    model.train_metrics = {"token_loss": last_base, "agreement_loss": last_agreement,
                           "concept_loss": last_concept,
                           "concept_agreement_loss": last_concept_agreement,
                           "concept_distill_loss": last_concept_distill,
                           "concept_rank_distill_loss": last_concept_rank_distill,
                           "concept_transfer_loss": last_concept_transfer,
                           "concept_contrast_loss": last_concept_contrast,
                           "concept_centroid_loss": last_concept_centroid,
                           "concept_prototype_loss": last_concept_prototype,
                           "concept_prototype_spread_loss": (
                               last_concept_prototype_spread),
                           "concept_state_spread_loss": last_concept_state_spread,
                           "latent_concept_loss": last_latent_concept,
                           "latent_factorization_loss": last_latent_factorization,
                           "latent_memory_loss": last_latent_memory,
                           "latent_association_loss": last_latent_association,
                           "latent_association_w": float(
                               latent_concept_association_w),
                           "latent_association_temperature": float(
                               latent_concept_association_temperature),
                           "latent_association_decay": float(
                               latent_concept_association_decay),
                           "latent_association_target_power": float(
                               latent_concept_association_target_power),
                           "latent_association_self_loop_w": float(
                               latent_concept_association_self_loop_w),
                           "latent_association_transitive_steps": int(
                               latent_concept_association_transitive_steps),
                           "latent_association_transitive_w": float(
                               latent_concept_association_transitive_w),
                           "latent_association_relation_updates": int(
                               getattr(getattr(model, "latent_concept_memory", None),
                                       "relation_updates",
                                       torch.zeros((), dtype=torch.long)).item()),
                           "latent_association_active_edges": int(
                               getattr(getattr(model, "latent_concept_memory", None),
                                       "relations", torch.zeros(0)).gt(0).sum().item()),
                           "latent_composition_loss": last_latent_composition,
                           "latent_composition_w": float(
                               latent_concept_composition_w),
                           "latent_composition_temperature": float(
                               latent_concept_composition_temperature),
                           "latent_composition_self_loop_w": float(
                               latent_concept_composition_self_loop_w),
                           "latent_composition_transitive_steps": int(
                               latent_concept_composition_transitive_steps),
                           "latent_composition_transitive_w": float(
                               latent_concept_composition_transitive_w),
                           "latent_composition_margin": float(
                               latent_concept_composition_margin),
                           "latent_graph_predict_loss": last_latent_graph_predict,
                           "latent_graph_predict_w": float(
                               latent_concept_graph_predict_w),
                           "latent_graph_predict_temperature": float(
                               latent_concept_graph_predict_temperature),
                           "latent_graph_predict_self_loop_w": float(
                               latent_concept_graph_predict_self_loop_w),
                           "latent_graph_predict_transitive_steps": int(
                               latent_concept_graph_predict_transitive_steps),
                           "latent_graph_predict_transitive_w": float(
                               latent_concept_graph_predict_transitive_w),
                           "latent_graph_predict_target_power": float(
                               latent_concept_graph_predict_target_power),
                           "latent_memory_size": int(
                               getattr(model, "latent_concept_memory_size", 0)),
                           "latent_memory_active": int(
                               getattr(getattr(model, "latent_concept_memory", None),
                                       "filled", torch.zeros((), dtype=torch.long)).item()),
                           "latent_memory_updates": int(
                               getattr(getattr(model, "latent_concept_memory", None),
                                       "updates", torch.zeros((), dtype=torch.long)).item()),
                           "latent_memory_last_batch_updates": int(
                               last_latent_memory_updates),
                           "latent_neighborhood_loss": last_latent_neighborhood,
                           "latent_transition_loss": last_latent_transition,
                           "latent_cluster_loss": last_latent_cluster,
                           "latent_factor_loss": last_latent_factor}
    model.study_reports = study_reports
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
        counterfactual_n=40, free_counterfactual_n=20, surfaces_path=None, checkpoint=None,
        batch=32, d=96, lr=1e-3, layers=3, heads=4, max_len=128, img_tokens=4, aud_tokens=8,
        txt_tokens=8, trunk_arch="conv", trunk_width=64, trunk_depth=1, text_layers=1,
        text_arch="transformer", modality_dropout=0.0, agreement_w=0.0,
        text_checkpoint=None,
        concept_tokens=4, fusion_layers=1,
        concept_prefix=False, concept_refine=False,
        concept_refine_gate_init=-2.0,
        concept_mixer_layers=0, concept_mixer_gate_init=-2.0,
        latent_concept_slots=0, latent_concept_layers=1,
        latent_concept_prefix=False, latent_concept_refine=False,
        latent_concept_refine_gate_init=-2.0,
        latent_concept_w=0.0, latent_concept_view_dropout=0.1,
        latent_concept_invariance_w=25.0, latent_concept_variance_w=25.0,
        latent_concept_covariance_w=1.0, latent_concept_variance_target=1.0,
        latent_concept_factorization_w=0.0,
        latent_concept_factorization_variance=0.05,
        latent_concept_factorization_margin=0.2,
        latent_concept_factorization_covariance_w=0.05,
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
        latent_concept_factor_w=0.0,
        study_strategy="random", study_score_metric="balanced",
        study_probe_n=0, study_hard_max=0, study_refresh_steps=0,
        concept_w=0.0, concept_agreement_w=0.0,
        concept_distill_w=0.0, concept_distill_temperature=1.0,
        concept_rank_distill_w=0.0, concept_rank_distill_margin=0.0,
        concept_transfer_w=0.0, concept_transfer_margin=0.0,
        concept_contrast_w=0.0, concept_contrast_temperature=0.1,
        concept_centroid_w=0.0, concept_centroid_temperature=0.1,
        concept_centroid_margin=0.0,
        concept_prototype_w=0.0, concept_prototype_spread_w=0.0,
        concept_prototype_spread_margin=0.2,
        concept_state_spread_w=0.0, concept_state_spread_variance=0.05,
        concept_state_spread_margin=0.2, concept_state_spread_covariance_w=0.05,
        log_every=100):
    model, vocab, surfaces = train(steps=steps, seed=seed, device=device, value_w=value_w,
                                   surfaces_path=surfaces_path, batch=batch, d=d, lr=lr,
                                   layers=layers, heads=heads, max_len=max_len,
                                   img_tokens=img_tokens, aud_tokens=aud_tokens,
                                   txt_tokens=txt_tokens, trunk_arch=trunk_arch,
                                   trunk_width=trunk_width, trunk_depth=trunk_depth,
                                   text_layers=text_layers,
                                   text_arch=text_arch, modality_dropout=modality_dropout,
                                   agreement_w=agreement_w,
                                   text_checkpoint=text_checkpoint,
                                   concept_tokens=concept_tokens,
                                   fusion_layers=fusion_layers,
                                   concept_prefix=concept_prefix,
                                   concept_refine=concept_refine,
                                   concept_refine_gate_init=concept_refine_gate_init,
                                   concept_mixer_layers=concept_mixer_layers,
                                   concept_mixer_gate_init=concept_mixer_gate_init,
                                   latent_concept_slots=latent_concept_slots,
                                   latent_concept_layers=latent_concept_layers,
                                   latent_concept_prefix=latent_concept_prefix,
                                   latent_concept_refine=latent_concept_refine,
                                   latent_concept_refine_gate_init=(
                                       latent_concept_refine_gate_init),
                                   latent_concept_w=latent_concept_w,
                                   latent_concept_view_dropout=(
                                       latent_concept_view_dropout),
                                   latent_concept_invariance_w=(
                                       latent_concept_invariance_w),
                                   latent_concept_variance_w=latent_concept_variance_w,
                                   latent_concept_covariance_w=(
                                       latent_concept_covariance_w),
                                   latent_concept_variance_target=(
                                       latent_concept_variance_target),
                                   latent_concept_factorization_w=(
                                       latent_concept_factorization_w),
                                   latent_concept_factorization_variance=(
                                       latent_concept_factorization_variance),
                                   latent_concept_factorization_margin=(
                                       latent_concept_factorization_margin),
                                   latent_concept_factorization_covariance_w=(
                                       latent_concept_factorization_covariance_w),
                                   latent_concept_memory_w=latent_concept_memory_w,
                                   latent_concept_memory_size=(
                                       latent_concept_memory_size),
                                   latent_concept_memory_temperature=(
                                       latent_concept_memory_temperature),
                                   latent_concept_memory_momentum=(
                                       latent_concept_memory_momentum),
                                   latent_concept_memory_balance_w=(
                                       latent_concept_memory_balance_w),
                                   latent_concept_association_w=(
                                       latent_concept_association_w),
                                   latent_concept_association_temperature=(
                                       latent_concept_association_temperature),
                                   latent_concept_association_decay=(
                                       latent_concept_association_decay),
                                   latent_concept_association_target_power=(
                                       latent_concept_association_target_power),
                                   latent_concept_association_self_loop_w=(
                                       latent_concept_association_self_loop_w),
                                   latent_concept_association_transitive_steps=(
                                       latent_concept_association_transitive_steps),
                                   latent_concept_association_transitive_w=(
                                       latent_concept_association_transitive_w),
                                   latent_concept_composition_w=(
                                       latent_concept_composition_w),
                                   latent_concept_composition_temperature=(
                                       latent_concept_composition_temperature),
                                   latent_concept_composition_self_loop_w=(
                                       latent_concept_composition_self_loop_w),
                                   latent_concept_composition_transitive_steps=(
                                       latent_concept_composition_transitive_steps),
                                   latent_concept_composition_transitive_w=(
                                       latent_concept_composition_transitive_w),
                                   latent_concept_composition_margin=(
                                       latent_concept_composition_margin),
                                   latent_concept_graph_predict_w=(
                                       latent_concept_graph_predict_w),
                                   latent_concept_graph_predict_temperature=(
                                       latent_concept_graph_predict_temperature),
                                   latent_concept_graph_predict_self_loop_w=(
                                       latent_concept_graph_predict_self_loop_w),
                                   latent_concept_graph_predict_transitive_steps=(
                                       latent_concept_graph_predict_transitive_steps),
                                   latent_concept_graph_predict_transitive_w=(
                                       latent_concept_graph_predict_transitive_w),
                                   latent_concept_graph_predict_target_power=(
                                       latent_concept_graph_predict_target_power),
                                   latent_concept_neighborhood_w=(
                                       latent_concept_neighborhood_w),
                                   latent_concept_neighborhood_temperature=(
                                       latent_concept_neighborhood_temperature),
                                   latent_concept_neighborhood_margin=(
                                       latent_concept_neighborhood_margin),
                                   latent_concept_transition_w=(
                                       latent_concept_transition_w),
                                   latent_concept_transition_temperature=(
                                       latent_concept_transition_temperature),
                                   latent_concept_transition_margin=(
                                       latent_concept_transition_margin),
                                   latent_concept_cluster_w=latent_concept_cluster_w,
                                   latent_concept_cluster_temperature=(
                                       latent_concept_cluster_temperature),
                                   latent_concept_cluster_margin=(
                                       latent_concept_cluster_margin),
                                   latent_concept_cluster_min_size=(
                                       latent_concept_cluster_min_size),
                                   latent_concept_factor_w=latent_concept_factor_w,
                                   study_strategy=study_strategy,
                                   study_score_metric=study_score_metric,
                                   study_probe_n=study_probe_n,
                                   study_hard_max=study_hard_max,
                                   study_refresh_steps=study_refresh_steps,
                                   concept_w=concept_w,
                                   concept_agreement_w=concept_agreement_w,
                                   concept_distill_w=concept_distill_w,
                                   concept_distill_temperature=(
                                       concept_distill_temperature),
                                   concept_rank_distill_w=concept_rank_distill_w,
                                   concept_rank_distill_margin=concept_rank_distill_margin,
                                   concept_transfer_w=concept_transfer_w,
                                   concept_transfer_margin=concept_transfer_margin,
                                   concept_contrast_w=concept_contrast_w,
                                   concept_contrast_temperature=concept_contrast_temperature,
                                   concept_centroid_w=concept_centroid_w,
                                   concept_centroid_temperature=concept_centroid_temperature,
                                   concept_centroid_margin=concept_centroid_margin,
                                   concept_prototype_w=concept_prototype_w,
                                   concept_prototype_spread_w=concept_prototype_spread_w,
                                   concept_prototype_spread_margin=(
                                       concept_prototype_spread_margin),
                                   concept_state_spread_w=concept_state_spread_w,
                                   concept_state_spread_variance=(
                                       concept_state_spread_variance),
                                   concept_state_spread_margin=concept_state_spread_margin,
                                   concept_state_spread_covariance_w=(
                                       concept_state_spread_covariance_w),
                                   log_every=log_every)
    full = evaluate(model, vocab, surfaces, n=eval_n, device=device, text_split="eval",
                    mode="full")
    text = evaluate(model, vocab, surfaces, n=eval_n, device=device, text_split="eval",
                    mode="text_only")
    sensor = evaluate(model, vocab, surfaces, n=eval_n, device=device, text_split="eval",
                      mode="sensor_only")
    concept_full = concept_evaluate(model, vocab, surfaces, n=eval_n, device=device,
                                    text_split="eval", mode="full")
    concept_text = concept_evaluate(model, vocab, surfaces, n=eval_n, device=device,
                                    text_split="eval", mode="text_only")
    concept_sensor = concept_evaluate(model, vocab, surfaces, n=eval_n, device=device,
                                      text_split="eval", mode="sensor_only")
    latent_factor_full = latent_factor_evaluate(
        model, vocab, surfaces, n=eval_n, device=device, text_split="eval", mode="full")
    latent_factor_text = latent_factor_evaluate(
        model, vocab, surfaces, n=eval_n, device=device, text_split="eval",
        mode="text_only")
    latent_factor_sensor = latent_factor_evaluate(
        model, vocab, surfaces, n=eval_n, device=device, text_split="eval",
        mode="sensor_only")
    concept_geometry_full = concept_geometry_evaluate(
        model, vocab, surfaces, n=eval_n, device=device, text_split="eval", mode="full")
    concept_geometry_text = concept_geometry_evaluate(
        model, vocab, surfaces, n=eval_n, device=device, text_split="eval",
        mode="text_only")
    concept_geometry_sensor = concept_geometry_evaluate(
        model, vocab, surfaces, n=eval_n, device=device, text_split="eval",
        mode="sensor_only")
    free_text = free_evaluate(model, vocab, surfaces, n=free_n, device=device, text_split="eval",
                              mode="text_only") if free_n else {}
    counterfactual = counterfactual_text_evaluate(
        model, vocab, surfaces, n=counterfactual_n, device=device) if counterfactual_n else {}
    free_counterfactual = free_counterfactual_text_evaluate(
        model, vocab, surfaces, n=free_counterfactual_n, device=device
    ) if free_counterfactual_n else {}
    thresholds = gate_thresholds()
    architecture = dict(model.config)
    architecture["img_pool"] = list(model.img.pool)
    architecture["aud_pool"] = list(model.aud.pool)
    architecture["reader_prefix_tokens"] = int(img_tokens) + int(aud_tokens) + int(txt_tokens)
    architecture["schema_concept_prefix_tokens"] = (
        len(VALUE_POS) if bool(concept_prefix) else 0)
    architecture["latent_concept_prefix_tokens"] = (
        int(model.latent_concept_slots) if bool(latent_concept_prefix) else 0)
    architecture["prefix_tokens"] = (
        architecture["reader_prefix_tokens"] + int(concept_tokens)
        + architecture["schema_concept_prefix_tokens"]
        + architecture["latent_concept_prefix_tokens"])
    text_transfer = getattr(model, "text_checkpoint_transfer", {})
    report = {"experiment": "m0_multimodal_bridge", "steps": steps, "batch": int(batch),
              "lr": float(lr), "value_w": float(value_w),
              "agreement_w": float(agreement_w),
              "text_checkpoint": text_checkpoint,
              "text_checkpoint_transfer": text_transfer,
              "text_arch": text_arch,
              "concept_prefix": bool(concept_prefix),
              "concept_refine": bool(concept_refine),
              "concept_refine_gate_init": float(concept_refine_gate_init),
              "concept_mixer_layers": int(concept_mixer_layers),
              "concept_mixer_gate_init": float(concept_mixer_gate_init),
              "latent_concept_slots": int(model.latent_concept_slots),
              "latent_concept_layers": int(model.latent_concept_layers),
              "latent_concept_prefix": bool(latent_concept_prefix),
              "latent_concept_refine": bool(latent_concept_refine),
              "latent_concept_refine_gate_init": float(
                  latent_concept_refine_gate_init),
              "latent_concept_w": float(latent_concept_w),
              "latent_concept_view_dropout": float(latent_concept_view_dropout),
              "latent_concept_invariance_w": float(latent_concept_invariance_w),
              "latent_concept_variance_w": float(latent_concept_variance_w),
              "latent_concept_covariance_w": float(latent_concept_covariance_w),
              "latent_concept_variance_target": float(latent_concept_variance_target),
              "latent_concept_factorization_w": float(latent_concept_factorization_w),
              "latent_concept_factorization_variance": float(
                  latent_concept_factorization_variance),
              "latent_concept_factorization_margin": float(
                  latent_concept_factorization_margin),
              "latent_concept_factorization_covariance_w": float(
                  latent_concept_factorization_covariance_w),
              "latent_concept_memory_w": float(latent_concept_memory_w),
              "latent_concept_memory_size": int(
                  getattr(model, "latent_concept_memory_size", 0)),
              "latent_concept_memory_temperature": float(
                  latent_concept_memory_temperature),
              "latent_concept_memory_momentum": float(latent_concept_memory_momentum),
              "latent_concept_memory_balance_w": float(
                  latent_concept_memory_balance_w),
              "latent_concept_association_w": float(latent_concept_association_w),
              "latent_concept_association_temperature": float(
                  latent_concept_association_temperature),
              "latent_concept_association_decay": float(
                  latent_concept_association_decay),
              "latent_concept_association_target_power": float(
                  latent_concept_association_target_power),
              "latent_concept_association_self_loop_w": float(
                  latent_concept_association_self_loop_w),
              "latent_concept_association_transitive_steps": int(
                  latent_concept_association_transitive_steps),
              "latent_concept_association_transitive_w": float(
                  latent_concept_association_transitive_w),
              "latent_concept_composition_w": float(latent_concept_composition_w),
              "latent_concept_composition_temperature": float(
                  latent_concept_composition_temperature),
              "latent_concept_composition_self_loop_w": float(
                  latent_concept_composition_self_loop_w),
              "latent_concept_composition_transitive_steps": int(
                  latent_concept_composition_transitive_steps),
              "latent_concept_composition_transitive_w": float(
                  latent_concept_composition_transitive_w),
              "latent_concept_composition_margin": float(
                  latent_concept_composition_margin),
              "latent_concept_graph_predict_w": float(
                  latent_concept_graph_predict_w),
              "latent_concept_graph_predict_temperature": float(
                  latent_concept_graph_predict_temperature),
              "latent_concept_graph_predict_self_loop_w": float(
                  latent_concept_graph_predict_self_loop_w),
              "latent_concept_graph_predict_transitive_steps": int(
                  latent_concept_graph_predict_transitive_steps),
              "latent_concept_graph_predict_transitive_w": float(
                  latent_concept_graph_predict_transitive_w),
              "latent_concept_graph_predict_target_power": float(
                  latent_concept_graph_predict_target_power),
              "latent_concept_neighborhood_w": float(latent_concept_neighborhood_w),
              "latent_concept_neighborhood_temperature": float(
                  latent_concept_neighborhood_temperature),
              "latent_concept_neighborhood_margin": float(
                  latent_concept_neighborhood_margin),
              "latent_concept_transition_w": float(latent_concept_transition_w),
              "latent_concept_transition_temperature": float(
                  latent_concept_transition_temperature),
              "latent_concept_transition_margin": float(
                  latent_concept_transition_margin),
              "latent_concept_cluster_w": float(latent_concept_cluster_w),
              "latent_concept_cluster_temperature": float(
                  latent_concept_cluster_temperature),
              "latent_concept_cluster_margin": float(latent_concept_cluster_margin),
              "latent_concept_cluster_min_size": int(latent_concept_cluster_min_size),
              "latent_concept_factor_w": float(latent_concept_factor_w),
              "study_strategy": study_strategy,
              "study_score_metric": study_score_metric,
              "study_probe_n": int(study_probe_n),
              "study_hard_max": int(study_hard_max),
              "study_refresh_steps": int(study_refresh_steps),
              "concept_w": float(concept_w),
              "concept_agreement_w": float(concept_agreement_w),
              "concept_distill_w": float(concept_distill_w),
              "concept_distill_temperature": float(concept_distill_temperature),
              "concept_rank_distill_w": float(concept_rank_distill_w),
              "concept_rank_distill_margin": float(concept_rank_distill_margin),
              "concept_transfer_w": float(concept_transfer_w),
              "concept_transfer_margin": float(concept_transfer_margin),
              "concept_contrast_w": float(concept_contrast_w),
              "concept_contrast_temperature": float(concept_contrast_temperature),
              "concept_centroid_w": float(concept_centroid_w),
              "concept_centroid_temperature": float(concept_centroid_temperature),
              "concept_centroid_margin": float(concept_centroid_margin),
              "concept_prototype_w": float(concept_prototype_w),
              "concept_prototype_spread_w": float(concept_prototype_spread_w),
              "concept_prototype_spread_margin": float(
                  concept_prototype_spread_margin),
              "concept_state_spread_w": float(concept_state_spread_w),
              "concept_state_spread_variance": float(concept_state_spread_variance),
              "concept_state_spread_margin": float(concept_state_spread_margin),
              "concept_state_spread_covariance_w": float(
                  concept_state_spread_covariance_w),
              "concept_transfer_variant": "full_correct_detached_vector",
              "train_metrics": getattr(model, "train_metrics", {}),
              "study_hard_examples": getattr(model, "study_reports", []),
              "architecture": architecture,
              "eval_n": int(eval_n), "free_n": int(free_n),
              "counterfactual_n": int(counterfactual_n),
              "free_counterfactual_n": int(free_counterfactual_n),
              "text_surfaces": {"path": surfaces_path or DEFAULT_SURFACES,
                                "train_templates": len(surfaces["train"]),
                                "eval_templates": len(surfaces["eval"]),
                                "train_examples": len(surfaces.get("train_examples", [])),
                                "eval_examples": len(surfaces.get("eval_examples", []))},
              "teacher_forced": {"full": full, "text_only_eval_phrasings": text,
                                 "sensor_only": sensor},
              "concept_head": {"full": concept_full,
                               "text_only_eval_phrasings": concept_text,
                               "sensor_only": concept_sensor},
              "latent_factor_head": {"full": latent_factor_full,
                                     "text_only_eval_phrasings": latent_factor_text,
                                     "sensor_only": latent_factor_sensor},
              "concept_geometry": {"full": concept_geometry_full,
                                   "text_only_eval_phrasings": concept_geometry_text,
                                   "sensor_only": concept_geometry_sensor},
              "free_text_only_eval_phrasings": free_text,
              "counterfactual_text_only_eval_phrasings": counterfactual,
              "free_counterfactual_text_only_eval_phrasings": free_counterfactual,
              "chance": CHANCE, "gate_thresholds": thresholds,
              "concept_gate": all(concept_full[k] >= thresholds[k]
                                  and concept_text[k] >= thresholds[k]
                                  and concept_sensor[k] >= thresholds[k]
                                  for k in VALUE_POS),
              "latent_factor_gate": (
                  not latent_factor_full.get("skipped", False)
                  and all(latent_factor_full[k] >= thresholds[k]
                          and latent_factor_text[k] >= thresholds[k]
                          and latent_factor_sensor[k] >= thresholds[k]
                          for k in VALUE_POS)),
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
                    "d": model.lm.tok.embedding_dim, "model_config": model.config,
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
    assert all(t in vocab.stoi for t in toks + txt + txt_eval), \
        "grammar/text token missing from vocab"
    example_surfaces = {
        "train": [],
        "eval": [],
        "train_examples": [{"tokens": ["external", "sentence", "says", "red", "circle"],
                            "facts": {"color": "red", "shape": "circle", "pitch": "n440",
                                      "timbre": "saw", "env": "decay"}}],
        "eval_examples": [{"tokens": ["external", "sentence", "says", "blue", "square"],
                           "facts": {"color": "blue", "shape": "square", "pitch": "n262",
                                     "timbre": "sine", "env": "flat"}}],
    }
    ex_vocab = build_vocab(example_surfaces)
    _img2, _aud2, ex_txt, ex_toks, ex_gold = sample_example(
        np.random.default_rng(1), example_surfaces)
    assert ex_gold["color"] == "red" and ex_txt[0] == "external"
    assert all(t in ex_vocab.stoi for t in ex_txt + ex_toks)
    model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad).to("cpu")
    x, a, tt, ids, golds = _batch(2, rng, vocab, "cpu", surfaces)
    assert _grid_pool(4) == (2, 2) and _grid_pool(8) == (2, 4)
    assert model.config["fusion"] == "concept"
    logits = model(x, a, tt, ids)
    assert logits.shape == (2, ids.shape[1], len(vocab)), logits.shape
    masked_head = SchemaConceptHead([("demo", "key")], [["a", "b"]], 8)
    masked_src = torch.randn(2, 3, 8)
    masked = torch.tensor([[True, True, True], [False, True, True]])
    masked_states = masked_head.state_tensor(masked_src, mask=masked)
    assert torch.isfinite(masked_states).all()
    assert torch.allclose(masked_states[0], torch.zeros_like(masked_states[0]))
    latent_head = LatentConceptHead(2, 8, heads=2, mixer_layers=1)
    latent_masked = latent_head(masked_src, mask=masked, project=True)
    assert torch.isfinite(latent_masked).all()
    assert torch.allclose(latent_masked[0], torch.zeros_like(latent_masked[0]))
    bundle = model.mode_bundle(x, a, tt, ids, mode="full", need_factor=True,
                               need_geometry=True)
    assert bundle["logits"].shape == logits.shape
    assert set(bundle["factor_states"]) == set(VALUE_POS)
    assert set(bundle["factor_logits"]) == set(VALUE_POS)
    assert set(bundle["factor_geometry_states"]) == set(VALUE_POS)
    logits_text = model(x, a, tt, ids, mode="text_only")
    assert logits_text.shape == logits.shape
    prefix, concepts = model.encode_prefix(x, a, tt, mode="full")
    assert concepts is not None and concepts.shape == (2, model.config["concept_tokens"], 32)
    assert prefix.shape[1] == (model.config["img_tokens"] + model.config["aud_tokens"]
                               + model.config["txt_tokens"] + model.config["concept_tokens"])
    assert model.decoder_prefix(x, a, tt, mode="full").shape[1] == prefix.shape[1]
    prefix_model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad,
                                concept_prefix=True).to("cpu")
    prefix_base, _prefix_concepts = prefix_model.encode_prefix(x, a, tt, mode="full")
    prefix_decoder = prefix_model.decoder_prefix(x, a, tt, mode="full")
    assert prefix_model.config["concept_prefix"] is True
    assert prefix_decoder.shape[1] == prefix_base.shape[1] + len(VALUE_POS)
    assert prefix_model(x, a, tt, ids).shape == logits.shape
    refine_model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad,
                                concept_refine=True).to("cpu")
    refine_prefix, refine_concepts = refine_model.encode_prefix(x, a, tt, mode="full")
    assert refine_model.config["concept_refine"] is True
    assert refine_concepts.shape == (2, refine_model.config["concept_tokens"], 32)
    assert refine_prefix.shape == prefix.shape
    assert any(name.startswith("concept_refiner.")
               for name, _param in refine_model.named_parameters())
    assert refine_model(x, a, tt, ids).shape == logits.shape
    mixer_model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad,
                               concept_mixer_layers=1).to("cpu")
    mixer_states = mixer_model.factor_concept_states(x, a, tt, mode="full")
    assert set(mixer_states) == set(VALUE_POS)
    assert any(name.startswith("factor_concepts.mixer.")
               for name, _param in mixer_model.named_parameters())
    latent_model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad,
                                latent_concept_slots=3,
                                latent_concept_layers=1).to("cpu")
    latent_states = latent_model.latent_concept_states(
        x, a, tt, mode="full", project=True)
    assert latent_states.shape == (2, 3, 32)
    latent_bundle = latent_model.mode_bundle(
        x, a, tt, ids, mode="full", need_latent=True,
        latent_view_dropout=0.1, latent_project=True)
    assert latent_bundle["latent_concepts"].shape == (2, 3, 32)
    assert torch.isfinite(latent_multimodal_concept_loss(
        latent_model, x, a, tt, view_dropout=0.1))
    latent_views = {
        mode: latent_model.latent_concept_states(
            x, a, tt, mode=mode, project=True)
        for mode in MODES
    }
    assert torch.isfinite(latent_multimodal_factorization_loss_from_views(
        latent_views, variance_target=0.01, separation_margin=0.2))
    latent_model.enable_latent_concept_memory(4)
    assert update_multimodal_latent_memory(
        latent_model, latent_views["full"], relation_decay=0.5) > 0
    assert int(latent_model.latent_concept_memory.relation_updates.item()) > 0
    assert torch.isfinite(latent_multimodal_memory_loss_from_views(
        latent_model, latent_views, temperature=0.2))
    assert torch.isfinite(latent_multimodal_association_loss_from_views(
        latent_model, latent_views, temperature=0.2))
    assert torch.isfinite(latent_multimodal_association_loss_from_views(
        latent_model, latent_views, temperature=0.2,
        transitive_steps=2, transitive_w=0.1))
    assert torch.isfinite(latent_multimodal_composition_loss_from_views(
        latent_model, latent_views, temperature=0.2,
        transitive_steps=2, transitive_w=0.1))
    assert torch.isfinite(latent_multimodal_graph_prediction_loss_from_views(
        latent_model, latent_views, temperature=0.2,
        transitive_steps=2, transitive_w=0.1))
    curious_examples, curious_report = multimodal_latent_curiosity_examples(
        latent_model, vocab, surfaces, n=4, seed=5, device="cpu",
        transitive_steps=2, transitive_w=0.1)
    assert curious_examples and curious_report["skipped"] is False
    assert torch.isfinite(latent_multimodal_neighborhood_loss_from_views(
        latent_views, temperature=0.2))
    assert torch.isfinite(latent_multimodal_transition_loss_from_views(
        latent_views, temperature=0.2))
    assert torch.isfinite(latent_multimodal_cluster_loss_from_views(
        latent_views, temperature=0.2, min_cluster_size=2))
    latent_prefix_model = MultimodalLM(
        len(vocab), d=32, layers=2, heads=4, pad=vocab.pad,
        latent_concept_slots=3, latent_concept_prefix=True,
        latent_concept_refine=True).to("cpu")
    latent_prefix_base, _latent_prefix_concepts = (
        latent_prefix_model.encode_prefix(x, a, tt, mode="full"))
    latent_prefix_dec = latent_prefix_model.decoder_prefix(x, a, tt, mode="full")
    assert latent_prefix_dec.shape[1] == latent_prefix_base.shape[1] + 3
    assert latent_prefix_model(x, a, tt, ids).shape == logits.shape
    latent_factor_logits = latent_prefix_model.latent_factor_logits(
        x, a, tt, mode="full")
    assert set(latent_factor_logits) == set(VALUE_POS)
    assert torch.isfinite(latent_factor_loss(
        {"full": latent_factor_logits}, golds))
    latent_factor_eval = latent_factor_evaluate(
        latent_prefix_model, vocab, surfaces, n=4, seed=6, device="cpu", mode="full")
    assert set(VALUE_POS).issubset(latent_factor_eval)
    assert latent_factor_eval["n_records"] == 4
    assert latent_factor_eval["skipped"] is False
    with tempfile.TemporaryDirectory() as tmpdir:
        from .text import ReadingRecord, TextFactLM, build_reading_vocab, checkpoint_payload

        reading_records = [
            ReadingRecord("transfer-train-0", "train",
                          tuple(_split_words("red circle shared language"))),
            ReadingRecord("transfer-eval-0", "eval",
                          tuple(_split_words("blue square shared language"))),
        ]
        text_vocab = build_reading_vocab(reading_records)
        text_model = TextFactLM(
            len(text_vocab), d=32, layers=1, heads=4, pad=text_vocab.pad,
            fact_schema=None, latent_concept_slots=3,
            latent_concept_layers=1, latent_concept_memory_size=5).to("cpu")
        with torch.no_grad():
            text_model.txt.emb.weight[text_vocab.stoi["red"]].fill_(0.314)
            text_model.latent_concepts.queries.fill_(0.271)
            text_model.latent_concept_memory.memory.fill_(0.123)
            text_model.latent_concept_memory.relations.fill_(0.456)
            text_model.latent_concept_memory.filled.fill_(5)
            text_model.latent_concept_memory.updates.fill_(7)
            text_model.latent_concept_memory.relation_updates.fill_(9)
        text_ckpt = os.path.join(tmpdir, "text-reading.pt")
        torch.save(
            checkpoint_payload(
                text_model, text_vocab, 32, 1, 4,
                {"experiment": "multimodal-transfer-selftest"}),
            text_ckpt)
        transfer_model = MultimodalLM(
            len(vocab), d=32, layers=2, heads=4, pad=vocab.pad,
            latent_concept_slots=3, latent_concept_layers=1,
            latent_concept_memory_size=5).to("cpu")
        red_before = transfer_model.txt.emb.weight[vocab.stoi["red"]].clone()
        latent_before = transfer_model.latent_concepts.queries.clone()
        memory_before = transfer_model.latent_concept_memory.memory.clone()
        transfer = import_text_checkpoint(transfer_model, vocab, text_ckpt, device="cpu")
        assert transfer["copied"] is True
        assert transfer["copied_token_embeddings"] > 0
        assert transfer["copied_position_rows"] > 0
        assert transfer["copied_text_tensor_count"] > 0
        assert transfer["copied_latent_tensor_count"] > 0
        red_after = transfer_model.txt.emb.weight[vocab.stoi["red"]]
        assert not torch.allclose(red_before, red_after)
        assert torch.allclose(red_after, torch.full_like(red_after, 0.314))
        assert not torch.allclose(latent_before, transfer_model.latent_concepts.queries)
        assert torch.allclose(
            transfer_model.latent_concepts.queries,
            text_model.latent_concepts.queries)
        assert not torch.allclose(memory_before,
                                  transfer_model.latent_concept_memory.memory)
        assert torch.allclose(transfer_model.latent_concept_memory.memory,
                              text_model.latent_concept_memory.memory)
        assert torch.allclose(transfer_model.latent_concept_memory.relations,
                              text_model.latent_concept_memory.relations)
        assert int(transfer_model.latent_concept_memory.filled.item()) == 5
        assert int(transfer_model.latent_concept_memory.relation_updates.item()) == 9
        ckpt_latents = text_checkpoint_latent_config(text_ckpt, device="cpu")
        assert ckpt_latents["latent_concept_slots"] == 3
        assert ckpt_latents["latent_concept_memory_size"] == 5
        auto_model, _auto_vocab, _auto_surfaces = train(
            steps=1, batch=2, d=32, layers=1, heads=4, seed=9, device="cpu",
            log_every=1, text_checkpoint=text_ckpt,
            latent_concept_factorization_w=0.1,
            latent_concept_memory_w=0.1,
            latent_concept_association_w=0.1,
            latent_concept_composition_w=0.1,
            latent_concept_graph_predict_w=0.1,
            latent_concept_neighborhood_w=0.1,
            latent_concept_transition_w=0.1,
            latent_concept_cluster_w=0.1,
            study_strategy="curiosity", study_probe_n=4, study_hard_max=2)
        assert auto_model.latent_concept_slots == 3
        assert auto_model.latent_concept_memory_size == 5
        assert auto_model.text_checkpoint_transfer["copied"] is True
        assert auto_model.train_metrics["latent_factorization_loss"] >= 0.0
        assert auto_model.train_metrics["latent_memory_loss"] >= 0.0
        assert auto_model.train_metrics["latent_memory_active"] > 0
        assert auto_model.train_metrics["latent_association_loss"] >= 0.0
        assert auto_model.train_metrics["latent_association_transitive_steps"] == 2
        assert auto_model.train_metrics["latent_association_transitive_w"] == 0.1
        assert auto_model.train_metrics["latent_association_relation_updates"] > 0
        assert auto_model.train_metrics["latent_composition_w"] == 0.1
        assert auto_model.train_metrics["latent_composition_loss"] >= 0.0
        assert auto_model.train_metrics["latent_graph_predict_w"] == 0.1
        assert math.isfinite(auto_model.train_metrics["latent_graph_predict_loss"])
        assert auto_model.train_metrics["latent_graph_predict_loss"] >= -1e-6
        assert auto_model.study_reports[-1]["strategy"] == "curiosity"
        assert auto_model.train_metrics["latent_neighborhood_loss"] >= 0.0
        assert auto_model.train_metrics["latent_transition_loss"] >= 0.0
        assert auto_model.train_metrics["latent_cluster_loss"] >= 0.0
    latent_errors, latent_correct, latent_report = multimodal_factor_record_outcomes(
        latent_prefix_model, vocab, surfaces, n=4, seed=7, device="cpu",
        metric="latent")
    assert latent_report["n_records"] == 4
    assert latent_report["n_error_records"] + latent_report["n_correct_records"] == 4
    assert len(latent_errors) + len(latent_correct) == 4
    decoder_errors, decoder_correct, decoder_report = multimodal_factor_record_outcomes(
        latent_prefix_model, vocab, surfaces, n=4, seed=8, device="cpu",
        metric="decoder", modes=("text_only",))
    assert decoder_report["metric"] == "decoder"
    assert len(decoder_errors) + len(decoder_correct) == 4
    balanced_errors, balanced_correct, balanced_report = (
        multimodal_factor_record_outcomes(
            latent_prefix_model, vocab, surfaces, n=4, seed=9, device="cpu",
            metric="balanced", modes=("text_only",)))
    assert balanced_report["metric"] == "balanced"
    assert set(balanced_report["active_metrics"]) == {
        "latent", "concept", "decoder"}
    assert set(balanced_report["by_metric"]) == set(balanced_report["active_metrics"])
    assert len(balanced_errors) + len(balanced_correct) == 4
    rel_txt_model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad,
                                 text_arch="relational", text_layers=1).to("cpu")
    assert rel_txt_model.txt.arch == "relational"
    assert rel_txt_model(x, a, tt, ids, mode="text_only").shape == logits.shape
    factor_logits = {mode: model.factor_logits(x, a, tt, mode=mode) for mode in MODES}
    factor_states = {mode: model.factor_concept_states(x, a, tt, mode=mode) for mode in MODES}
    factor_geometry_states = {
        mode: model.factor_geometry_from_states(states)
        for mode, states in factor_states.items()
    }
    assert set(factor_logits["full"]) == set(VALUE_POS)
    assert set(factor_states["full"]) == set(VALUE_POS)
    assert factor_states["full"]["color"].shape == (2, 32)
    concept_loss = concept_factor_loss(factor_logits, golds)
    assert torch.isfinite(concept_loss), concept_loss
    assert torch.isfinite(concept_factor_contrastive_loss(
        factor_geometry_states, golds, temperature=0.2))
    assert torch.isfinite(concept_factor_centroid_loss(
        factor_geometry_states, golds, temperature=0.2))
    assert torch.isfinite(concept_factor_prototype_loss(
        model, factor_geometry_states, golds, temperature=0.2))
    assert torch.isfinite(concept_factor_prototype_spread_loss(model, margin=0.2))
    assert torch.isfinite(concept_factor_state_spread_loss(factor_geometry_states, golds))
    assert torch.isfinite(concept_factor_agreement_loss(factor_logits))
    assert torch.isfinite(concept_full_distill_loss(factor_logits, temperature=1.25))
    assert torch.isfinite(concept_full_rank_distill_loss(factor_logits, golds, margin=0.05))
    assert torch.isfinite(concept_state_transfer_loss(factor_states, factor_logits, golds,
                                                       margin=0.05))
    geom_eval = concept_geometry_evaluate(
        model, vocab, surfaces, n=4, seed=5, device="cpu", mode="full")
    assert set(geom_eval["by_factor"]) == set(VALUE_POS)
    assert set(geom_eval["mean"]) >= {"nearest_same_acc", "margin"}
    res_model = MultimodalLM(len(vocab), d=32, layers=2, heads=4, pad=vocab.pad,
                             img_tokens=4, aud_tokens=8, txt_tokens=6,
                             trunk_arch="residual", trunk_width=32, trunk_depth=1,
                             text_layers=2, text_arch="relational",
                             modality_dropout=0.1, concept_tokens=3,
                             fusion_layers=2).to("cpu")
    assert res_model.img.arch == "residual" and res_model.img.pool == (2, 2)
    assert res_model.txt.layers == 2 and res_model.txt.n_tokens == 6
    assert res_model.txt.arch == "relational"
    assert res_model.config["fusion_layers"] == 2
    res_logits = {mode: res_model(x, a, tt, ids, mode=mode) for mode in MODES}
    assert all(v.shape == logits.shape for v in res_logits.values())
    agree = value_agreement_loss(res_logits, vocab)
    assert torch.isfinite(agree), agree
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
    ap.add_argument("--device", default=DEV)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=96, dest="d")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=128, dest="max_len")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=100, dest="log_every")
    ap.add_argument("--value-w", type=float, default=6.0, dest="value_w")
    ap.add_argument("--agreement-w", type=float, default=0.0, dest="agreement_w",
                    help="cross-mode factor-value distribution agreement loss weight")
    ap.add_argument("--text-checkpoint", default=None, dest="text_checkpoint",
                    help=("optional thinking.text checkpoint used to warm-start the "
                          "multimodal transcript encoder and matching latent concept slots"))
    ap.add_argument("--concept-tokens", type=int, default=4, dest="concept_tokens",
                    help="shared concept tokens prepended before the decoder in concept fusion")
    ap.add_argument("--fusion-layers", type=int, default=1, dest="fusion_layers",
                    help="transformer layers used by concept prefix fusion")
    ap.add_argument("--concept-prefix", action="store_true", dest="concept_prefix",
                    help="prepend schema concept states to the decoder prefix")
    ap.add_argument("--concept-refine", action="store_true", dest="concept_refine",
                    help=("refine upstream multimodal prefix states with learned schema "
                          "concept feedback"))
    ap.add_argument("--concept-refine-gate-init", type=float, default=-2.0,
                    dest="concept_refine_gate_init",
                    help="initial logit for the learned concept-refinement residual gate")
    ap.add_argument("--concept-mixer-layers", type=int, default=0,
                    dest="concept_mixer_layers",
                    help="self-attention layers for the schema factor concept workspace")
    ap.add_argument("--concept-mixer-gate-init", type=float, default=-2.0,
                    dest="concept_mixer_gate_init",
                    help="initial logit for the learned schema factor concept workspace gate")
    ap.add_argument("--latent-concept-slots", type=int, default=0,
                    dest="latent_concept_slots",
                    help="schema-free latent concept slots aligned across modality views")
    ap.add_argument("--latent-concept-layers", type=int, default=1,
                    dest="latent_concept_layers",
                    help="self-attention layers inside latent concept slots")
    ap.add_argument("--latent-concept-prefix", action="store_true",
                    dest="latent_concept_prefix",
                    help="prepend schema-free latent concept slots to the decoder prefix")
    ap.add_argument("--latent-concept-refine", action="store_true",
                    dest="latent_concept_refine",
                    help="refine fused multimodal prefix states with latent concept slots")
    ap.add_argument("--latent-concept-refine-gate-init", type=float, default=-2.0,
                    dest="latent_concept_refine_gate_init",
                    help="initial logit for latent concept refinement residual gate")
    ap.add_argument("--latent-concept-w", type=float, default=0.0,
                    dest="latent_concept_w",
                    help="weight for schema-free latent multimodal concept loss")
    ap.add_argument("--latent-concept-view-dropout", type=float, default=0.1,
                    dest="latent_concept_view_dropout",
                    help="feature dropout used for latent multimodal concept views")
    ap.add_argument("--latent-concept-invariance-w", type=float, default=25.0,
                    dest="latent_concept_invariance_w",
                    help="invariance term weight inside latent concept loss")
    ap.add_argument("--latent-concept-variance-w", type=float, default=25.0,
                    dest="latent_concept_variance_w",
                    help="anti-collapse variance term weight inside latent concept loss")
    ap.add_argument("--latent-concept-covariance-w", type=float, default=1.0,
                    dest="latent_concept_covariance_w",
                    help="decorrelation term weight inside latent concept loss")
    ap.add_argument("--latent-concept-variance-target", type=float, default=1.0,
                    dest="latent_concept_variance_target",
                    help="minimum per-dimension std target for latent concept slots")
    ap.add_argument("--latent-concept-factorization-w", type=float, default=0.0,
                    dest="latent_concept_factorization_w",
                    help="weight for schema-free latent slot factorization")
    ap.add_argument("--latent-concept-factorization-variance", type=float,
                    default=0.05, dest="latent_concept_factorization_variance",
                    help="minimum per-slot batch variation for latent concept slots")
    ap.add_argument("--latent-concept-factorization-margin", type=float,
                    default=0.2, dest="latent_concept_factorization_margin",
                    help="allowed same-example cosine before slot separation penalty")
    ap.add_argument("--latent-concept-factorization-covariance-w", type=float,
                    default=0.05,
                    dest="latent_concept_factorization_covariance_w",
                    help="decorrelation weight inside latent slot factorization")
    ap.add_argument("--latent-concept-memory-w", type=float, default=0.0,
                    dest="latent_concept_memory_w",
                    help="weight for checkpointed schema-free latent concept memory")
    ap.add_argument("--latent-concept-memory-size", type=int, default=0,
                    dest="latent_concept_memory_size",
                    help="persistent latent concept prototypes; 0 disables memory")
    ap.add_argument("--latent-concept-memory-temperature", type=float, default=0.1,
                    dest="latent_concept_memory_temperature",
                    help="contrastive temperature for latent concept memory")
    ap.add_argument("--latent-concept-memory-momentum", type=float, default=0.95,
                    dest="latent_concept_memory_momentum",
                    help="EMA momentum used when updating latent concept memory")
    ap.add_argument("--latent-concept-memory-balance-w", type=float, default=0.01,
                    dest="latent_concept_memory_balance_w",
                    help="usage-balance weight for latent concept memory")
    ap.add_argument("--latent-concept-association-w", type=float, default=0.0,
                    dest="latent_concept_association_w",
                    help="weight for persistent self-mined latent concept association graph")
    ap.add_argument("--latent-concept-association-temperature", type=float,
                    default=0.1, dest="latent_concept_association_temperature",
                    help="contrastive temperature for latent concept associations")
    ap.add_argument("--latent-concept-association-decay", type=float, default=0.99,
                    dest="latent_concept_association_decay",
                    help="EMA decay for persistent latent association graph updates")
    ap.add_argument("--latent-concept-association-target-power", type=float,
                    default=1.0, dest="latent_concept_association_target_power",
                    help="sharpening power for association graph targets")
    ap.add_argument("--latent-concept-association-self-loop-w", type=float,
                    default=0.05, dest="latent_concept_association_self_loop_w",
                    help="self-loop weight in latent association graph targets")
    ap.add_argument("--latent-concept-association-transitive-steps", type=int,
                    default=2, dest="latent_concept_association_transitive_steps",
                    help="graph walk depth for inferred latent concept associations")
    ap.add_argument("--latent-concept-association-transitive-w", type=float,
                    default=0.1, dest="latent_concept_association_transitive_w",
                    help="weight for multi-hop inferred latent concept associations")
    ap.add_argument("--latent-concept-composition-w", type=float, default=0.0,
                    dest="latent_concept_composition_w",
                    help=("weight for graph-composition loss over self-mined "
                          "latent concept relations"))
    ap.add_argument("--latent-concept-composition-temperature", type=float,
                    default=0.1, dest="latent_concept_composition_temperature",
                    help="contrastive temperature for latent graph composition")
    ap.add_argument("--latent-concept-composition-self-loop-w", type=float,
                    default=0.0, dest="latent_concept_composition_self_loop_w",
                    help="self-loop weight in latent graph composition targets")
    ap.add_argument("--latent-concept-composition-transitive-steps", type=int,
                    default=2, dest="latent_concept_composition_transitive_steps",
                    help="graph walk depth for latent concept composition")
    ap.add_argument("--latent-concept-composition-transitive-w", type=float,
                    default=0.1, dest="latent_concept_composition_transitive_w",
                    help="weight for multi-hop inferred concept composition")
    ap.add_argument("--latent-concept-composition-margin", type=float, default=0.0,
                    dest="latent_concept_composition_margin",
                    help="minimum margin over non-target concept compositions")
    ap.add_argument("--latent-concept-graph-predict-w", type=float, default=0.0,
                    dest="latent_concept_graph_predict_w",
                    help=("weight for partial-view latent concept prediction through "
                          "the self-mined graph"))
    ap.add_argument("--latent-concept-graph-predict-temperature", type=float,
                    default=0.1, dest="latent_concept_graph_predict_temperature",
                    help="contrastive temperature for latent graph prediction")
    ap.add_argument("--latent-concept-graph-predict-self-loop-w", type=float,
                    default=0.05, dest="latent_concept_graph_predict_self_loop_w",
                    help="self-loop weight in latent graph prediction targets")
    ap.add_argument("--latent-concept-graph-predict-transitive-steps", type=int,
                    default=2, dest="latent_concept_graph_predict_transitive_steps",
                    help="graph walk depth for latent graph prediction")
    ap.add_argument("--latent-concept-graph-predict-transitive-w", type=float,
                    default=0.1, dest="latent_concept_graph_predict_transitive_w",
                    help="weight for multi-hop graph prediction targets")
    ap.add_argument("--latent-concept-graph-predict-target-power", type=float,
                    default=1.0, dest="latent_concept_graph_predict_target_power",
                    help="sharpening power for full-view latent concept targets")
    ap.add_argument("--latent-concept-neighborhood-w", type=float, default=0.0,
                    dest="latent_concept_neighborhood_w",
                    help=("weight for self-mined latent neighborhood alignment "
                          "across multimodal views"))
    ap.add_argument("--latent-concept-neighborhood-temperature", type=float,
                    default=0.1, dest="latent_concept_neighborhood_temperature",
                    help="contrastive temperature for multimodal latent neighborhoods")
    ap.add_argument("--latent-concept-neighborhood-margin", type=float, default=0.0,
                    dest="latent_concept_neighborhood_margin",
                    help="minimum margin over other multimodal latent neighbors")
    ap.add_argument("--latent-concept-transition-w", type=float, default=0.0,
                    dest="latent_concept_transition_w",
                    help=("weight for label-free latent transition alignment "
                          "across multimodal views"))
    ap.add_argument("--latent-concept-transition-temperature", type=float,
                    default=0.1, dest="latent_concept_transition_temperature",
                    help="contrastive temperature for multimodal latent transitions")
    ap.add_argument("--latent-concept-transition-margin", type=float, default=0.0,
                    dest="latent_concept_transition_margin",
                    help="minimum margin over other multimodal latent transitions")
    ap.add_argument("--latent-concept-cluster-w", type=float, default=0.0,
                    dest="latent_concept_cluster_w",
                    help=("weight for self-mined multimodal latent cluster "
                          "prototype consolidation"))
    ap.add_argument("--latent-concept-cluster-temperature", type=float, default=0.1,
                    dest="latent_concept_cluster_temperature",
                    help="contrastive temperature for multimodal latent clusters")
    ap.add_argument("--latent-concept-cluster-margin", type=float, default=0.0,
                    dest="latent_concept_cluster_margin",
                    help="minimum margin over other multimodal latent cluster prototypes")
    ap.add_argument("--latent-concept-cluster-min-size", type=int, default=2,
                    dest="latent_concept_cluster_min_size",
                    help="minimum records per mined multimodal latent cluster")
    ap.add_argument("--latent-concept-factor-w", type=float, default=0.0,
                    dest="latent_concept_factor_w",
                    help=("weight for predicting data-defined multimodal factors "
                          "from schema-free latent slots"))
    ap.add_argument("--study-strategy", choices=("random", "errors", "curiosity"),
                    default="random",
                    dest="study_strategy",
                    help=("sample normal batches, train on current errors, "
                          "or use concept-graph curiosity"))
    ap.add_argument("--study-score-metric", choices=MULTIMODAL_STUDY_METRICS,
                    default="balanced", dest="study_score_metric",
                    help="metric used to mine multimodal hard examples")
    ap.add_argument("--study-probe-n", type=int, default=0, dest="study_probe_n",
                    help="number of generated train examples to probe for current errors")
    ap.add_argument("--study-hard-max", type=int, default=0, dest="study_hard_max",
                    help="cap mined hard examples kept in the active study pool")
    ap.add_argument("--study-refresh-steps", type=int, default=0,
                    dest="study_refresh_steps",
                    help="refresh mined hard examples every N steps; 0 means once")
    ap.add_argument("--concept-w", type=float, default=0.0, dest="concept_w",
                    help="supervised upstream concept-token factor loss weight")
    ap.add_argument("--concept-agreement-w", type=float, default=0.0,
                    dest="concept_agreement_w",
                    help="cross-mode agreement loss on upstream concept heads")
    ap.add_argument("--concept-distill-w", type=float, default=0.0,
                    dest="concept_distill_w",
                    help="distill full multimodal concept distributions into partial modes")
    ap.add_argument("--concept-distill-temperature", type=float, default=1.0,
                    dest="concept_distill_temperature",
                    help="temperature for full-to-partial upstream concept distillation")
    ap.add_argument("--concept-rank-distill-w", type=float, default=0.0,
                    dest="concept_rank_distill_w",
                    help="preserve full-path correct concept winners and margins in partial modes")
    ap.add_argument("--concept-rank-distill-margin", type=float, default=0.0,
                    dest="concept_rank_distill_margin",
                    help="minimum margin for full-to-partial concept rank distillation")
    ap.add_argument("--concept-transfer-w", type=float, default=0.0,
                    dest="concept_transfer_w",
                    help=("correct-detached vector transfer from full multimodal concept "
                          "states into partial modes"))
    ap.add_argument("--concept-transfer-margin", type=float, default=0.0,
                    dest="concept_transfer_margin",
                    help="minimum margin for upstream concept vector transfer")
    ap.add_argument("--concept-contrast-w", type=float, default=0.0,
                    dest="concept_contrast_w",
                    help="same-value concept-state contrastive geometry loss weight")
    ap.add_argument("--concept-contrast-temperature", type=float, default=0.1,
                    dest="concept_contrast_temperature",
                    help="temperature for same-value concept-state contrastive loss")
    ap.add_argument("--concept-centroid-w", type=float, default=0.0,
                    dest="concept_centroid_w",
                    help="batch-discovered same-value concept centroid loss weight")
    ap.add_argument("--concept-centroid-temperature", type=float, default=0.1,
                    dest="concept_centroid_temperature",
                    help="temperature for batch-discovered concept centroid loss")
    ap.add_argument("--concept-centroid-margin", type=float, default=0.0,
                    dest="concept_centroid_margin",
                    help="minimum margin between own centroid and other value centroids")
    ap.add_argument("--concept-prototype-w", type=float, default=0.0,
                    dest="concept_prototype_w",
                    help="prototype classification loss weight for concept geometry states")
    ap.add_argument("--concept-prototype-spread-w", type=float, default=0.0,
                    dest="concept_prototype_spread_w",
                    help="same-key value prototype anti-collapse loss weight")
    ap.add_argument("--concept-prototype-spread-margin", type=float, default=0.2,
                    dest="concept_prototype_spread_margin",
                    help="maximum allowed same-key prototype cosine before spread penalty")
    ap.add_argument("--concept-state-spread-w", type=float, default=0.0,
                    dest="concept_state_spread_w",
                    help="batch state anti-collapse loss weight for concept geometry states")
    ap.add_argument("--concept-state-spread-variance", type=float, default=0.05,
                    dest="concept_state_spread_variance",
                    help="minimum normalized per-dimension concept-state std target")
    ap.add_argument("--concept-state-spread-margin", type=float, default=0.2,
                    dest="concept_state_spread_margin",
                    help="maximum observed-value centroid cosine before state spread penalty")
    ap.add_argument("--concept-state-spread-covariance-w", type=float, default=0.05,
                    dest="concept_state_spread_covariance_w",
                    help="relative decorrelation weight inside concept-state spread loss")
    ap.add_argument("--img-tokens", type=int, default=4, dest="img_tokens")
    ap.add_argument("--aud-tokens", type=int, default=8, dest="aud_tokens")
    ap.add_argument("--txt-tokens", type=int, default=8, dest="txt_tokens")
    ap.add_argument("--trunk-arch", default="conv", choices=TRUNK_ARCHES, dest="trunk_arch")
    ap.add_argument("--trunk-width", type=int, default=64, dest="trunk_width")
    ap.add_argument("--trunk-depth", type=int, default=1, dest="trunk_depth")
    ap.add_argument("--text-layers", type=int, default=1, dest="text_layers")
    ap.add_argument("--text-arch", choices=TEXT_TRUNK_ARCHES, default="transformer",
                    dest="text_arch",
                    help=("transcript encoder architecture: transformer keeps the legacy "
                          "encoder; relational/abstractor use symbolic-value attention"))
    ap.add_argument("--modality-dropout", type=float, default=0.0, dest="modality_dropout")
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
    positive = {
        "--steps": args.steps, "--batch": args.batch, "--dim": args.d,
        "--layers": args.layers, "--heads": args.heads, "--max-len": args.max_len,
        "--log-every": args.log_every, "--img-tokens": args.img_tokens,
        "--aud-tokens": args.aud_tokens, "--txt-tokens": args.txt_tokens,
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
    if (args.agreement_w < 0.0 or args.concept_w < 0.0
            or args.concept_agreement_w < 0.0 or args.concept_distill_w < 0.0
            or args.concept_rank_distill_w < 0.0 or args.concept_transfer_w < 0.0
            or args.concept_contrast_w < 0.0 or args.concept_centroid_w < 0.0
            or args.concept_prototype_w < 0.0
            or args.concept_prototype_spread_w < 0.0
            or args.concept_state_spread_w < 0.0):
        ap.error("agreement/concept loss weights must be non-negative")
    if args.concept_rank_distill_margin < 0.0:
        ap.error("--concept-rank-distill-margin must be non-negative")
    if args.concept_transfer_margin < 0.0:
        ap.error("--concept-transfer-margin must be non-negative")
    if args.concept_distill_temperature <= 0.0:
        ap.error("--concept-distill-temperature must be positive")
    if args.concept_contrast_temperature <= 0.0:
        ap.error("--concept-contrast-temperature must be positive")
    if args.concept_centroid_temperature <= 0.0:
        ap.error("--concept-centroid-temperature must be positive")
    if args.concept_centroid_margin < 0.0:
        ap.error("--concept-centroid-margin must be non-negative")
    if args.concept_prototype_spread_margin < -1.0:
        ap.error("--concept-prototype-spread-margin must be >= -1")
    if args.concept_state_spread_variance < 0.0:
        ap.error("--concept-state-spread-variance must be non-negative")
    if args.concept_state_spread_margin < -1.0:
        ap.error("--concept-state-spread-margin must be >= -1")
    if args.concept_state_spread_covariance_w < 0.0:
        ap.error("--concept-state-spread-covariance-w must be non-negative")
    if not math.isfinite(args.concept_refine_gate_init):
        ap.error("--concept-refine-gate-init must be finite")
    if args.concept_mixer_layers < 0:
        ap.error("--concept-mixer-layers must be non-negative")
    if not math.isfinite(args.concept_mixer_gate_init):
        ap.error("--concept-mixer-gate-init must be finite")
    checkpoint_latent_slots = 0
    if args.text_checkpoint:
        try:
            checkpoint_latent_slots = text_checkpoint_latent_config(
                args.text_checkpoint, device="cpu").get("latent_concept_slots", 0)
        except Exception as exc:
            ap.error(f"--text-checkpoint could not be loaded: {exc}")
    effective_latent_slots = args.latent_concept_slots or checkpoint_latent_slots
    if args.latent_concept_slots < 0:
        ap.error("--latent-concept-slots must be non-negative")
    if args.latent_concept_layers <= 0:
        ap.error("--latent-concept-layers must be positive")
    if args.latent_concept_w < 0.0:
        ap.error("--latent-concept-w must be non-negative")
    if args.latent_concept_w > 0.0 and effective_latent_slots <= 0:
        ap.error("--latent-concept-w requires --latent-concept-slots > 0")
    if args.latent_concept_factorization_w < 0.0:
        ap.error("--latent-concept-factorization-w must be non-negative")
    if args.latent_concept_factorization_w > 0.0 and effective_latent_slots <= 0:
        ap.error("--latent-concept-factorization-w requires --latent-concept-slots > 0")
    if args.latent_concept_factorization_variance < 0.0:
        ap.error("--latent-concept-factorization-variance must be non-negative")
    if args.latent_concept_factorization_margin < 0.0:
        ap.error("--latent-concept-factorization-margin must be non-negative")
    if args.latent_concept_factorization_covariance_w < 0.0:
        ap.error("--latent-concept-factorization-covariance-w must be non-negative")
    if args.latent_concept_memory_w < 0.0:
        ap.error("--latent-concept-memory-w must be non-negative")
    if args.latent_concept_memory_size < 0:
        ap.error("--latent-concept-memory-size must be non-negative")
    if ((args.latent_concept_memory_w > 0.0
         or args.latent_concept_association_w > 0.0
         or args.latent_concept_composition_w > 0.0
         or args.latent_concept_graph_predict_w > 0.0
         or args.latent_concept_memory_size > 0)
            and effective_latent_slots <= 0):
        ap.error("--latent-concept-memory/association/composition/graph-predict requires "
                 "--latent-concept-slots > 0")
    if args.latent_concept_memory_temperature <= 0.0:
        ap.error("--latent-concept-memory-temperature must be positive")
    if (args.latent_concept_memory_momentum < 0.0
            or args.latent_concept_memory_momentum >= 1.0):
        ap.error("--latent-concept-memory-momentum must be in [0, 1)")
    if args.latent_concept_memory_balance_w < 0.0:
        ap.error("--latent-concept-memory-balance-w must be non-negative")
    if args.latent_concept_association_w < 0.0:
        ap.error("--latent-concept-association-w must be non-negative")
    if args.latent_concept_association_temperature <= 0.0:
        ap.error("--latent-concept-association-temperature must be positive")
    if (args.latent_concept_association_decay < 0.0
            or args.latent_concept_association_decay >= 1.0):
        ap.error("--latent-concept-association-decay must be in [0, 1)")
    if args.latent_concept_association_target_power <= 0.0:
        ap.error("--latent-concept-association-target-power must be positive")
    if args.latent_concept_association_self_loop_w < 0.0:
        ap.error("--latent-concept-association-self-loop-w must be non-negative")
    if args.latent_concept_association_transitive_steps < 1:
        ap.error("--latent-concept-association-transitive-steps must be positive")
    if args.latent_concept_association_transitive_w < 0.0:
        ap.error("--latent-concept-association-transitive-w must be non-negative")
    if args.latent_concept_composition_w < 0.0:
        ap.error("--latent-concept-composition-w must be non-negative")
    if args.latent_concept_composition_temperature <= 0.0:
        ap.error("--latent-concept-composition-temperature must be positive")
    if args.latent_concept_composition_self_loop_w < 0.0:
        ap.error("--latent-concept-composition-self-loop-w must be non-negative")
    if args.latent_concept_composition_transitive_steps < 1:
        ap.error("--latent-concept-composition-transitive-steps must be positive")
    if args.latent_concept_composition_transitive_w < 0.0:
        ap.error("--latent-concept-composition-transitive-w must be non-negative")
    if args.latent_concept_composition_margin < 0.0:
        ap.error("--latent-concept-composition-margin must be non-negative")
    if args.latent_concept_graph_predict_w < 0.0:
        ap.error("--latent-concept-graph-predict-w must be non-negative")
    if args.latent_concept_graph_predict_temperature <= 0.0:
        ap.error("--latent-concept-graph-predict-temperature must be positive")
    if args.latent_concept_graph_predict_self_loop_w < 0.0:
        ap.error("--latent-concept-graph-predict-self-loop-w must be non-negative")
    if args.latent_concept_graph_predict_transitive_steps < 1:
        ap.error("--latent-concept-graph-predict-transitive-steps must be positive")
    if args.latent_concept_graph_predict_transitive_w < 0.0:
        ap.error("--latent-concept-graph-predict-transitive-w must be non-negative")
    if args.latent_concept_graph_predict_target_power <= 0.0:
        ap.error("--latent-concept-graph-predict-target-power must be positive")
    if args.latent_concept_neighborhood_w < 0.0:
        ap.error("--latent-concept-neighborhood-w must be non-negative")
    if args.latent_concept_neighborhood_w > 0.0 and effective_latent_slots <= 0:
        ap.error("--latent-concept-neighborhood-w requires --latent-concept-slots > 0")
    if args.latent_concept_neighborhood_temperature <= 0.0:
        ap.error("--latent-concept-neighborhood-temperature must be positive")
    if args.latent_concept_neighborhood_margin < 0.0:
        ap.error("--latent-concept-neighborhood-margin must be non-negative")
    if args.latent_concept_transition_w < 0.0:
        ap.error("--latent-concept-transition-w must be non-negative")
    if args.latent_concept_transition_w > 0.0 and effective_latent_slots <= 0:
        ap.error("--latent-concept-transition-w requires --latent-concept-slots > 0")
    if args.latent_concept_transition_temperature <= 0.0:
        ap.error("--latent-concept-transition-temperature must be positive")
    if args.latent_concept_transition_margin < 0.0:
        ap.error("--latent-concept-transition-margin must be non-negative")
    if args.latent_concept_cluster_w < 0.0:
        ap.error("--latent-concept-cluster-w must be non-negative")
    if args.latent_concept_cluster_w > 0.0 and effective_latent_slots <= 0:
        ap.error("--latent-concept-cluster-w requires --latent-concept-slots > 0")
    if args.latent_concept_cluster_temperature <= 0.0:
        ap.error("--latent-concept-cluster-temperature must be positive")
    if args.latent_concept_cluster_margin < 0.0:
        ap.error("--latent-concept-cluster-margin must be non-negative")
    if args.latent_concept_cluster_min_size < 2:
        ap.error("--latent-concept-cluster-min-size must be at least two")
    if args.latent_concept_factor_w < 0.0:
        ap.error("--latent-concept-factor-w must be non-negative")
    if args.latent_concept_factor_w > 0.0 and effective_latent_slots <= 0:
        ap.error("--latent-concept-factor-w requires --latent-concept-slots > 0")
    if ((args.latent_concept_prefix or args.latent_concept_refine)
            and effective_latent_slots <= 0):
        ap.error("latent concept prefix/refine require --latent-concept-slots > 0")
    if not math.isfinite(args.latent_concept_refine_gate_init):
        ap.error("--latent-concept-refine-gate-init must be finite")
    if args.latent_concept_view_dropout < 0.0 or args.latent_concept_view_dropout >= 1.0:
        ap.error("--latent-concept-view-dropout must be in [0, 1)")
    if (args.latent_concept_invariance_w < 0.0
            or args.latent_concept_variance_w < 0.0
            or args.latent_concept_covariance_w < 0.0):
        ap.error("latent concept loss weights must be non-negative")
    if args.latent_concept_variance_target < 0.0:
        ap.error("--latent-concept-variance-target must be non-negative")
    if args.study_probe_n < 0:
        ap.error("--study-probe-n must be non-negative")
    if args.study_hard_max < 0:
        ap.error("--study-hard-max must be non-negative")
    if args.study_refresh_steps < 0:
        ap.error("--study-refresh-steps must be non-negative")
    if (args.study_strategy == "errors" and args.study_score_metric == "latent"
            and effective_latent_slots <= 0):
        ap.error("--study-score-metric latent requires --latent-concept-slots > 0")
    if args.modality_dropout < 0.0 or args.modality_dropout > 1.0:
        ap.error("--modality-dropout must be in [0, 1]")
    if args.eval_n <= 0:
        ap.error("--eval-n must be positive")
    if args.free_n < 0 or args.counterfactual_n < 0 or args.free_counterfactual_n < 0:
        ap.error("free/counterfactual eval counts must be non-negative")
    report = run(steps=args.steps, seed=args.seed, value_w=args.value_w,
                 eval_n=args.eval_n, free_n=args.free_n,
                 counterfactual_n=args.counterfactual_n,
                 free_counterfactual_n=args.free_counterfactual_n,
                 surfaces_path=args.surfaces,
                 checkpoint=args.checkpoint, batch=args.batch, d=args.d, lr=args.lr,
                 layers=args.layers, heads=args.heads, max_len=args.max_len,
                 img_tokens=args.img_tokens, aud_tokens=args.aud_tokens,
                 txt_tokens=args.txt_tokens, trunk_arch=args.trunk_arch,
                 trunk_width=args.trunk_width, trunk_depth=args.trunk_depth,
                 text_layers=args.text_layers, text_arch=args.text_arch,
                 modality_dropout=args.modality_dropout,
                 agreement_w=args.agreement_w,
                 text_checkpoint=args.text_checkpoint,
                 concept_tokens=args.concept_tokens,
                 fusion_layers=args.fusion_layers, concept_prefix=args.concept_prefix,
                 concept_refine=args.concept_refine,
                 concept_refine_gate_init=args.concept_refine_gate_init,
                 concept_mixer_layers=args.concept_mixer_layers,
                 concept_mixer_gate_init=args.concept_mixer_gate_init,
                 latent_concept_slots=args.latent_concept_slots,
                 latent_concept_layers=args.latent_concept_layers,
                 latent_concept_prefix=args.latent_concept_prefix,
                 latent_concept_refine=args.latent_concept_refine,
                 latent_concept_refine_gate_init=args.latent_concept_refine_gate_init,
                 latent_concept_w=args.latent_concept_w,
                 latent_concept_view_dropout=args.latent_concept_view_dropout,
                 latent_concept_invariance_w=args.latent_concept_invariance_w,
                 latent_concept_variance_w=args.latent_concept_variance_w,
                 latent_concept_covariance_w=args.latent_concept_covariance_w,
                 latent_concept_variance_target=args.latent_concept_variance_target,
                 latent_concept_factorization_w=(
                     args.latent_concept_factorization_w),
                 latent_concept_factorization_variance=(
                     args.latent_concept_factorization_variance),
                 latent_concept_factorization_margin=(
                     args.latent_concept_factorization_margin),
                 latent_concept_factorization_covariance_w=(
                     args.latent_concept_factorization_covariance_w),
                 latent_concept_memory_w=args.latent_concept_memory_w,
                 latent_concept_memory_size=args.latent_concept_memory_size,
                 latent_concept_memory_temperature=(
                     args.latent_concept_memory_temperature),
                 latent_concept_memory_momentum=(
                     args.latent_concept_memory_momentum),
                 latent_concept_memory_balance_w=(
                     args.latent_concept_memory_balance_w),
                 latent_concept_association_w=(
                     args.latent_concept_association_w),
                 latent_concept_association_temperature=(
                     args.latent_concept_association_temperature),
                 latent_concept_association_decay=(
                     args.latent_concept_association_decay),
                 latent_concept_association_target_power=(
                     args.latent_concept_association_target_power),
                 latent_concept_association_self_loop_w=(
                     args.latent_concept_association_self_loop_w),
                 latent_concept_association_transitive_steps=(
                     args.latent_concept_association_transitive_steps),
                 latent_concept_association_transitive_w=(
                     args.latent_concept_association_transitive_w),
                 latent_concept_composition_w=(
                     args.latent_concept_composition_w),
                 latent_concept_composition_temperature=(
                     args.latent_concept_composition_temperature),
                 latent_concept_composition_self_loop_w=(
                     args.latent_concept_composition_self_loop_w),
                 latent_concept_composition_transitive_steps=(
                     args.latent_concept_composition_transitive_steps),
                 latent_concept_composition_transitive_w=(
                     args.latent_concept_composition_transitive_w),
                 latent_concept_composition_margin=(
                     args.latent_concept_composition_margin),
                 latent_concept_graph_predict_w=(
                     args.latent_concept_graph_predict_w),
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
                 latent_concept_neighborhood_w=args.latent_concept_neighborhood_w,
                 latent_concept_neighborhood_temperature=(
                     args.latent_concept_neighborhood_temperature),
                 latent_concept_neighborhood_margin=(
                     args.latent_concept_neighborhood_margin),
                 latent_concept_transition_w=args.latent_concept_transition_w,
                 latent_concept_transition_temperature=(
                     args.latent_concept_transition_temperature),
                 latent_concept_transition_margin=(
                     args.latent_concept_transition_margin),
                 latent_concept_cluster_w=args.latent_concept_cluster_w,
                 latent_concept_cluster_temperature=(
                     args.latent_concept_cluster_temperature),
                 latent_concept_cluster_margin=args.latent_concept_cluster_margin,
                 latent_concept_cluster_min_size=(
                     args.latent_concept_cluster_min_size),
                 latent_concept_factor_w=args.latent_concept_factor_w,
                 study_strategy=args.study_strategy,
                 study_score_metric=args.study_score_metric,
                 study_probe_n=args.study_probe_n,
                 study_hard_max=args.study_hard_max,
                 study_refresh_steps=args.study_refresh_steps,
                 concept_w=args.concept_w,
                 concept_agreement_w=args.concept_agreement_w,
                 concept_distill_w=args.concept_distill_w,
                 concept_distill_temperature=args.concept_distill_temperature,
                 concept_rank_distill_w=args.concept_rank_distill_w,
                 concept_rank_distill_margin=args.concept_rank_distill_margin,
                 concept_transfer_w=args.concept_transfer_w,
                 concept_transfer_margin=args.concept_transfer_margin,
                 concept_contrast_w=args.concept_contrast_w,
                 concept_contrast_temperature=args.concept_contrast_temperature,
                 concept_centroid_w=args.concept_centroid_w,
                 concept_centroid_temperature=args.concept_centroid_temperature,
                 concept_centroid_margin=args.concept_centroid_margin,
                 concept_prototype_w=args.concept_prototype_w,
                 concept_prototype_spread_w=args.concept_prototype_spread_w,
                 concept_prototype_spread_margin=(
                     args.concept_prototype_spread_margin),
                 concept_state_spread_w=args.concept_state_spread_w,
                 concept_state_spread_variance=args.concept_state_spread_variance,
                 concept_state_spread_margin=args.concept_state_spread_margin,
                 concept_state_spread_covariance_w=(
                     args.concept_state_spread_covariance_w),
                 log_every=args.log_every,
                 device=args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
