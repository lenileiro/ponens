"""Manifest-conditioned latent rectified flow for real image generation.

This module trains a compact image autoencoder plus a latent-space rectified flow.
Production training is manifest-driven: captions, image/text embedding sidecars,
quality scores, and preference pairs come from data rather than hard-coded visual
factors.

The design follows the current scalable image-generation pattern at toy scale:

  image -> semantic autoencoder latent -> rectified flow in latent space -> decoder -> image

  python -m thinking.image_latent --selftest
  python -m thinking.image_latent --train --cond-mode text --image-manifest data/images/train.jsonl \
      --image-root data/images --flow-arch dit --ae-steps 400 --flow-steps 400 \
      --cond-drop 0.1 --cfg-scale 1.5 --cfg-rescale 0.7 --sample-steps 8 \
      --out runs/image_latent_dit.pt
  python -m thinking.image_latent --eval-checkpoint runs/image_latent_dit.pt \
      --cfg-scales 1.0,1.25,1.5,2.0 --cfg-rescales 0.0,0.7 \
      --sample-steps-list 4,8,16 \
      --eval-seeds 1,2,3 --eval-out runs/image_latent_dit_sweep.json
  python -m thinking.image_latent --train --cond-mode text --flow-arch dit \
      --image-manifest data/images/train.jsonl --image-root data/images \
      --ae-steps 400 --flow-steps 400
"""
import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict, defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_activation_checkpoint

from .image_data import (ImageTextRecord, build_caption_vocab, caption_ids,
                         load_image_tensor, normalized_sampling_weights,
                         read_image_manifest, sample_image_text_batch,
                         summarize_records)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_METHODS = ("euler", "heun", "midpoint", "rk4", "adaptive-heun")
SAMPLE_SCHEDULES = ("linear", "quadratic", "sqrt", "cosine", "karras")
KARRAS_SCHEDULE_RHO = 7.0
KARRAS_SCHEDULE_SIGMA_MIN = 1.0e-3
KARRAS_SCHEDULE_SIGMA_MAX = 1.0
CFG_MODES = ("standard", "cfgpp")
CFG_SCHEDULES = ("constant", "linear", "cosine", "triangular")
MMDIT_ATTN_IMPLS = ("manual", "sdpa", "linear", "auto")
DIT_POS_EMBEDS = ("learned", "sincos2d", "rope2d")
DIT_MLPS = ("gelu", "swiglu")
FLOW_TIME_EMBEDS = ("scalar", "fourier")
FLOW_LOSS_WEIGHTS = ("none", "min-snr-v", "soft-min-snr-v")
FLOW_NOISE_COUPLINGS = ("random", "sliced_ot")
FLOW_REPA_MODES = ("pooled", "token", "both", "auto")
FLOW_BOUNDARY_MODES = ("none", "right-linear", "double-linear", "double-cosine")
TIME_SAMPLINGS = ("uniform", "logit-normal", "mode", "adaptive")
DEFAULT_TIME_MODE_SCALE = 1.29
MAX_TIME_MODE_SCALE = 1.75
TIME_SHIFT_MODES = ("manual", "dim", "auto")
AE_ARCHES = ("semantic", "residual", "hf-vae")
FLOW_DISTILL_TEACHERS = ("raw", "ema", "auto")
DEFAULT_GUIDANCE_INTERVAL = (0.0, 1.0)
DEFAULT_CHURN_INTERVAL = (0.0, 0.8)
LATENT_NORMALIZE_MODES = ("none", "global", "channel", "auto")


def activation_checkpoint(fn, *args):
    """Checkpoint a module call with the non-reentrant implementation when available."""
    try:
        return torch_activation_checkpoint(fn, *args, use_reentrant=False)
    except TypeError:
        return torch_activation_checkpoint(fn, *args)


def should_checkpoint_blocks(module):
    return bool(getattr(module, "checkpoint_blocks", False)
                and module.training and torch.is_grad_enabled())

class SemanticAutoencoder(nn.Module):
    """Small convolutional autoencoder for manifest image latents."""

    def __init__(self, latent_ch=16, hidden=64):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.downsample = 4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, latent_ch, 3, padding=1),
            nn.GroupNorm(1, latent_ch),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden // 2, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return {"latent": z, "recon": self.decode(z)}


def _norm(ch):
    groups = min(8, int(ch))
    while groups > 1 and int(ch) % groups:
        groups -= 1
    return nn.GroupNorm(groups, int(ch))


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            _norm(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            _norm(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualAutoencoder(nn.Module):
    """Deeper representation AE with configurable spatial compression.

    This is still lightweight enough for local tests, but it has the shape we need for real
    image work: residual stages, high-dimensional latents, and explicit downsample control.
    """

    def __init__(self, latent_ch=16, hidden=64, downsample=8, res_blocks=1):
        super().__init__()
        downsample = int(downsample)
        if downsample < 2 or downsample & (downsample - 1):
            raise ValueError("latent_downsample must be a power of two >= 2")
        self.latent_ch = int(latent_ch)
        self.downsample = downsample
        self.res_blocks = int(res_blocks)
        levels = int(math.log2(downsample))
        stem_ch = max(8, hidden // 2)
        enc = [nn.Conv2d(3, stem_ch, 3, padding=1)]
        channels = [stem_ch]
        ch = stem_ch
        for i in range(levels):
            out_ch = min(int(hidden) * (2 ** i), int(hidden) * 4)
            enc.extend([nn.GELU(), nn.Conv2d(ch, out_ch, 4, stride=2, padding=1)])
            enc.extend(ResidualBlock(out_ch) for _ in range(max(0, self.res_blocks)))
            ch = out_ch
            channels.append(ch)
        enc.extend([nn.GELU(), nn.Conv2d(ch, latent_ch, 3, padding=1), _norm(latent_ch)])
        self.encoder = nn.Sequential(*enc)

        dec_ch = channels[-1]
        dec = [nn.Conv2d(latent_ch, dec_ch, 3, padding=1)]
        for level in reversed(range(levels)):
            out_ch = channels[level]
            dec.extend(ResidualBlock(dec_ch) for _ in range(max(0, self.res_blocks)))
            dec.extend([nn.GELU(), nn.ConvTranspose2d(dec_ch, out_ch, 4,
                                                      stride=2, padding=1)])
            dec_ch = out_ch
        dec.extend([nn.GELU(), nn.Conv2d(dec_ch, 3, 3, padding=1), nn.Tanh()])
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return {"latent": z, "recon": self.decode(z)}


def _config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class HFAutoencoderKL(nn.Module):
    """Frozen Diffusers AutoencoderKL adapter for pretrained latent image spaces."""

    def __init__(self, model_id, subfolder="", scaling_factor=0.0, latent_downsample=8):
        super().__init__()
        if not model_id:
            raise ValueError("ae_arch='hf-vae' requires --ae-hf-model")
        try:
            from diffusers import AutoencoderKL
        except Exception as e:  # pragma: no cover - optional GPU dependency.
            raise ImportError(
                "ae_arch='hf-vae' requires diffusers. Install it with "
                "`pip install diffusers transformers accelerate safetensors`."
            ) from e
        kwargs = {}
        if subfolder:
            kwargs["subfolder"] = str(subfolder)
        self.model_id = str(model_id)
        self.subfolder = str(subfolder or "")
        self.vae = AutoencoderKL.from_pretrained(self.model_id, **kwargs)
        config = getattr(self.vae, "config", None)
        self.latent_ch = int(_config_value(config, "latent_channels", 4) or 4)
        scale = float(scaling_factor or 0.0)
        if scale <= 0.0:
            scale = float(_config_value(config, "scaling_factor", 0.18215) or 0.18215)
        self.scaling_factor = float(scale)
        self.downsample = int(latent_downsample or 8)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    def _param_dtype(self):
        try:
            return next(self.vae.parameters()).dtype
        except StopIteration:  # pragma: no cover - AutoencoderKL always has params.
            return torch.float32

    def train(self, mode=True):
        super().train(False)
        self.vae.eval()
        return self

    def encode(self, x):
        x_in = x.to(dtype=self._param_dtype())
        out = self.vae.encode(x_in)
        dist = getattr(out, "latent_dist", None)
        if dist is not None:
            z = dist.mode() if hasattr(dist, "mode") else dist.sample()
        elif hasattr(out, "latents"):
            z = out.latents
        else:
            z = out[0]
        return z * float(self.scaling_factor)

    def decode(self, z):
        z_in = (z / float(self.scaling_factor)).to(dtype=self._param_dtype())
        out = self.vae.decode(z_in)
        sample = getattr(out, "sample", out[0] if isinstance(out, tuple) else out)
        return sample.clamp(-1.0, 1.0)

    def forward(self, x):
        z = self.encode(x)
        return {"latent": z, "recon": self.decode(z)}


def make_autoencoder(ae_arch="semantic", latent_ch=16, hidden=64, latent_downsample=4,
                     ae_res_blocks=1, ae_hf_model="", ae_hf_subfolder="",
                     ae_hf_scaling_factor=0.0):
    if ae_arch == "semantic":
        if int(latent_downsample) != 4:
            raise ValueError("semantic AE uses latent_downsample=4; use --ae-arch residual")
        return SemanticAutoencoder(latent_ch=latent_ch, hidden=hidden)
    if ae_arch == "residual":
        return ResidualAutoencoder(latent_ch=latent_ch, hidden=hidden,
                                   downsample=latent_downsample,
                                   res_blocks=ae_res_blocks)
    if ae_arch == "hf-vae":
        return HFAutoencoderKL(
            ae_hf_model, subfolder=ae_hf_subfolder,
            scaling_factor=ae_hf_scaling_factor,
            latent_downsample=latent_downsample)
    raise ValueError(f"unknown autoencoder architecture {ae_arch!r}")


def normalize_image_size(size, default=None):
    if size is None or (isinstance(size, str) and size == ""):
        if default is None:
            return None
        return normalize_image_size(default)
    if isinstance(size, str):
        raw = size.strip().lower()
        if raw in ("", "0", "none"):
            if default is None:
                return None
            return normalize_image_size(default)
        for sep in ("x", ",", ":"):
            if sep in raw:
                parts = [p.strip() for p in raw.split(sep)]
                if len(parts) != 2:
                    raise ValueError(f"image size {size!r} must be SIZE or HxW")
                h, w = int(parts[0]), int(parts[1])
                break
        else:
            h = w = int(raw)
    elif isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError("image size tuple must be (height, width)")
        h, w = int(size[0]), int(size[1])
    else:
        h = w = int(size)
    if h <= 0 or w <= 0:
        if (h == 0 or w == 0) and default is not None:
            return normalize_image_size(default)
        raise ValueError("image size must be positive")
    return h, w


def image_size_value(size):
    h, w = normalize_image_size(size, default=32)
    return int(h) if h == w else [int(h), int(w)]


def image_size_pair_value(size):
    h, w = image_hw(size)
    return [int(h), int(w)]


def image_size_key(size):
    h, w = image_hw(size)
    return f"{int(h)}x{int(w)}"


def normalize_image_size_buckets(sizes):
    if sizes is None or sizes == "":
        return ()
    if isinstance(sizes, str):
        raw = sizes.strip()
        if raw.lower() in ("", "0", "none"):
            return ()
        if ";" in raw:
            parts = raw.split(";")
        elif "|" in raw:
            parts = raw.split("|")
        elif any(ch.isspace() for ch in raw):
            parts = raw.split()
        elif "," in raw and ("x" in raw.lower() or ":" in raw):
            parts = raw.split(",")
        else:
            parts = [raw]
    else:
        if (isinstance(sizes, (tuple, list)) and len(sizes) == 2
                and all(isinstance(x, (int, float, np.integer)) for x in sizes)):
            parts = [sizes]
        else:
            parts = list(sizes)
    buckets = []
    seen = set()
    for part in parts:
        if isinstance(part, str) and not part.strip():
            continue
        bucket = image_hw(part)
        key = image_size_key(bucket)
        if key in seen:
            continue
        seen.add(key)
        buckets.append(bucket)
    return tuple(buckets)


def image_size_buckets_value(buckets):
    return [image_size_pair_value(size) for size in buckets]


def image_hw(size, default=32):
    return normalize_image_size(size, default=default)


def image_side(size, name="image size"):
    h, w = image_hw(size)
    if h != w:
        raise ValueError(f"{name} must be square for synthetic factor renders; got {h}x{w}")
    return int(h)


def image_size_aspect(size):
    h, w = image_hw(size)
    return float(w) / max(1.0, float(h))


def image_size_area(size):
    h, w = image_hw(size)
    return int(h) * int(w)


def bucket_records_by_aspect(records, size_buckets):
    buckets = tuple(size_buckets)
    groups = {bucket: [] for bucket in buckets}
    missing_dims = 0
    if len(buckets) <= 1:
        if buckets:
            groups[buckets[0]] = list(records)
        return groups, missing_dims
    bucket_aspects = {bucket: image_size_aspect(bucket) for bucket in buckets}
    for rec in records:
        width = int(getattr(rec, "width", 0) or 0)
        height = int(getattr(rec, "height", 0) or 0)
        if width <= 0 or height <= 0:
            missing_dims += 1
            continue
        aspect = float(width) / max(1.0, float(height))
        bucket = min(
            buckets,
            key=lambda b: abs(math.log(max(aspect, 1.0e-8) / max(bucket_aspects[b], 1.0e-8))))
        groups[bucket].append(rec)
    return groups, missing_dims


def choose_image_size_bucket(rng, size_buckets):
    buckets = tuple(size_buckets)
    if not buckets:
        raise ValueError("no image size buckets configured")
    if len(buckets) == 1:
        return buckets[0]
    return buckets[int(rng.integers(len(buckets)))]


def size_bucket_weight_sums(records, size_buckets, bucket_records=None, record_weights=None):
    buckets = tuple(size_buckets)
    rows_all = list(records)
    out = {}
    for bucket in buckets:
        rows = list(bucket_records.get(bucket) or rows_all) if bucket_records is not None else rows_all
        weights = weights_for_records(rows, record_weights)
        out[bucket] = float(np.sum(weights)) if weights is not None else float(len(rows))
    return out


def size_bucket_sampling_probs(records, size_buckets, bucket_records=None, record_weights=None):
    buckets = tuple(size_buckets)
    if len(buckets) <= 1 or record_weights is None:
        return None
    sums = size_bucket_weight_sums(
        records, buckets, bucket_records=bucket_records, record_weights=record_weights)
    weights = np.asarray([float(sums[bucket]) for bucket in buckets], dtype=np.float64)
    return normalized_sampling_weights(weights, len(buckets))


def size_bucket_sampling_report(records, size_buckets, bucket_records=None, record_weights=None,
                                bucket_probs=None):
    buckets = tuple(size_buckets)
    if not buckets:
        return {
            "size_bucket_sampling_mode": "",
            "size_bucket_weight_sums": {},
            "size_bucket_sampling_probs": {},
        }
    sums = size_bucket_weight_sums(
        records, buckets, bucket_records=bucket_records, record_weights=record_weights)
    if bucket_probs is None:
        probs = np.ones(len(buckets), dtype=np.float64) / float(len(buckets))
        mode = "uniform"
    else:
        probs = np.asarray(bucket_probs, dtype=np.float64)
        mode = "weighted"
    return {
        "size_bucket_sampling_mode": mode,
        "size_bucket_weight_sums": {
            image_size_key(bucket): float(sums[bucket]) for bucket in buckets
        },
        "size_bucket_sampling_probs": {
            image_size_key(bucket): float(probs[i]) for i, bucket in enumerate(buckets)
        },
    }


def choose_image_size_bucket_weighted(rng, size_buckets, bucket_probs=None):
    buckets = tuple(size_buckets)
    if not buckets:
        raise ValueError("no image size buckets configured")
    if len(buckets) == 1:
        return buckets[0]
    if bucket_probs is None:
        return choose_image_size_bucket(rng, buckets)
    probs = normalized_sampling_weights(bucket_probs, len(buckets))
    return buckets[int(rng.choice(len(buckets), p=probs))]


def size_curriculum_switch_step(flow_steps, frac=0.0):
    frac = float(frac)
    if frac < 0.0 or frac > 1.0:
        raise ValueError("size_curriculum_frac must be in [0, 1]")
    steps = int(flow_steps)
    if frac <= 0.0 or steps <= 0:
        return 0
    return min(steps, max(1, int(math.ceil(float(steps) * frac))))


def size_curriculum_bucket_order(size_buckets):
    buckets = tuple(size_buckets)
    return tuple(sorted(
        buckets,
        key=lambda b: (image_size_area(b), int(image_hw(b)[0]), int(image_hw(b)[1]))))


def active_size_curriculum_buckets(size_buckets, flow_step=0, flow_steps=0,
                                   size_curriculum_frac=0.0):
    buckets = tuple(size_buckets)
    if len(buckets) <= 1:
        return buckets
    switch_step = size_curriculum_switch_step(flow_steps, size_curriculum_frac)
    if switch_step <= 0 or int(flow_step) >= switch_step:
        return buckets
    progress = float(int(flow_step) + 1) / float(max(1, switch_step))
    active_n = min(len(buckets), max(1, int(math.floor(progress * len(buckets)))))
    active = set(size_curriculum_bucket_order(buckets)[:active_n])
    return tuple(bucket for bucket in buckets if bucket in active)


def size_bucket_probs_subset(size_buckets, bucket_probs, active_buckets):
    active_buckets = tuple(active_buckets)
    if bucket_probs is None or len(active_buckets) <= 1:
        return None
    buckets = tuple(size_buckets)
    prob_map = {
        bucket: float(prob)
        for bucket, prob in zip(buckets, np.asarray(bucket_probs, dtype=np.float64))
    }
    probs = np.asarray([prob_map.get(bucket, 0.0) for bucket in active_buckets],
                       dtype=np.float64)
    return normalized_sampling_weights(probs, len(active_buckets))


def image_size_bucket_keys(size_buckets):
    return {image_size_key(bucket) for bucket in tuple(size_buckets)}


GEOMETRY_FEATURE_DIM = 20


def _transform_meta_int(meta, key, default=0):
    if not isinstance(meta, dict):
        return int(default)
    try:
        return int(meta.get(key, default) or default)
    except (TypeError, ValueError):
        return int(default)


def image_geometry_feature_tensor(records=None, target_size=32, batch=0, device=DEV,
                                  transform_metadata=None):
    """Return generic image transform features for latent image conditioning."""
    target_h, target_w = image_hw(target_size, default=32)
    if records is None:
        rows = [None for _ in range(int(batch or 1))]
    else:
        rows = list(records)
    if batch and int(batch) != len(rows):
        if records is not None:
            raise ValueError(
                f"geometry feature batch {int(batch)} does not match records {len(rows)}")
        rows = [None for _ in range(int(batch))]
    if transform_metadata is None:
        metadata_rows = [None for _ in rows]
    else:
        metadata_rows = list(transform_metadata)
        if len(metadata_rows) != len(rows):
            raise ValueError(
                f"geometry metadata batch {len(metadata_rows)} "
                f"does not match records {len(rows)}")
    vals = []
    base = 1024.0
    for rec, meta in zip(rows, metadata_rows):
        orig_w = _transform_meta_int(
            meta, "original_width",
            int(getattr(rec, "width", 0) or 0) if rec is not None else 0)
        orig_h = _transform_meta_int(
            meta, "original_height",
            int(getattr(rec, "height", 0) or 0) if rec is not None else 0)
        cur_target_h = _transform_meta_int(meta, "target_height", target_h)
        cur_target_w = _transform_meta_int(meta, "target_width", target_w)
        if orig_w <= 0:
            orig_w = int(cur_target_w)
        if orig_h <= 0:
            orig_h = int(cur_target_h)
        orig_h_f = max(1.0, float(orig_h))
        orig_w_f = max(1.0, float(orig_w))
        target_h_f = max(1.0, float(cur_target_h))
        target_w_f = max(1.0, float(cur_target_w))
        crop_top = max(0.0, float(_transform_meta_int(meta, "crop_top", 0)))
        crop_left = max(0.0, float(_transform_meta_int(meta, "crop_left", 0)))
        crop_h = max(1.0, float(_transform_meta_int(meta, "crop_height", orig_h)))
        crop_w = max(1.0, float(_transform_meta_int(meta, "crop_width", orig_w)))
        pad_top = max(0.0, float(_transform_meta_int(meta, "pad_top", 0)))
        pad_left = max(0.0, float(_transform_meta_int(meta, "pad_left", 0)))
        pad_bottom = max(0.0, float(_transform_meta_int(meta, "pad_bottom", 0)))
        pad_right = max(0.0, float(_transform_meta_int(meta, "pad_right", 0)))
        hflip = 1.0 if (isinstance(meta, dict) and bool(meta.get("hflip", False))) else 0.0
        vals.append([
            math.log(orig_h_f / base),
            math.log(orig_w_f / base),
            math.log(target_h_f / base),
            math.log(target_w_f / base),
            math.log(orig_w_f / orig_h_f),
            math.log(target_w_f / target_h_f),
            math.log(target_h_f / orig_h_f),
            math.log(target_w_f / orig_w_f),
            crop_top / orig_h_f,
            crop_left / orig_w_f,
            crop_h / orig_h_f,
            crop_w / orig_w_f,
            math.log(crop_w / crop_h),
            math.log(target_h_f / crop_h),
            math.log(target_w_f / crop_w),
            pad_top / target_h_f,
            pad_left / target_w_f,
            pad_bottom / target_h_f,
            pad_right / target_w_f,
            hflip,
        ])
    return torch.tensor(vals, dtype=torch.float32, device=device)


def geometry_features_from_size_key(size_key, batch, device=DEV):
    text = str(size_key or "").strip()
    target = image_hw(text, default=32) if text else 32
    return image_geometry_feature_tensor(
        records=None, target_size=target, batch=int(batch), device=device)


def image_geometry_embedding(features, cond_dim):
    """Map numeric geometry features to a fixed Fourier embedding of cond_dim."""
    cond_dim = int(cond_dim)
    if cond_dim <= 0:
        raise ValueError("geometry condition requires a positive cond_dim")
    features = features.float()
    _b, k = int(features.shape[0]), int(features.shape[1])
    per_feature = max(1, int(math.ceil(cond_dim / float(max(1, k * 2)))))
    freqs = torch.pow(
        torch.tensor(2.0, device=features.device, dtype=features.dtype),
        torch.arange(per_feature, device=features.device, dtype=features.dtype))
    angles = math.pi * features[:, :, None] * freqs[None, None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(1)
    if int(emb.shape[1]) < cond_dim:
        emb = F.pad(emb, (0, cond_dim - int(emb.shape[1])), value=0.0)
    emb = emb[:, :cond_dim]
    return emb / math.sqrt(float(max(1, cond_dim)))


def attach_image_geometry_condition(cond, records=None, target_size=32, enabled=False,
                                    geometry_features=None, batch=0,
                                    transform_metadata=None):
    """Append crop/size geometry conditioning while preserving it under CFG nulls."""
    if not enabled:
        return cond
    vec = condition_vector(cond)
    b = int(vec.shape[0])
    if geometry_features is None:
        geometry_features = image_geometry_feature_tensor(
            records=records, target_size=target_size, batch=batch or b, device=vec.device,
            transform_metadata=transform_metadata)
    else:
        geometry_features = geometry_features.to(device=vec.device, dtype=torch.float32)
    if int(geometry_features.shape[0]) != b:
        raise ValueError(
            f"geometry feature batch {int(geometry_features.shape[0])} "
            f"does not match condition batch {b}")
    geom = image_geometry_embedding(geometry_features, int(vec.shape[-1])).to(
        device=vec.device, dtype=vec.dtype)
    if not isinstance(cond, dict):
        return cond + geom
    out = dict(cond)
    out["geometry_features"] = geometry_features
    out["geometry_vec"] = geom
    out["vec"] = cond["vec"] + geom
    if "tokens" in cond:
        geom_token = geom[:, None, :].to(dtype=cond["tokens"].dtype)
        out["tokens"] = torch.cat([cond["tokens"], geom_token], dim=1)
        if "mask" in cond:
            geom_mask = torch.zeros((b, 1), dtype=torch.bool, device=cond["mask"].device)
            out["mask"] = torch.cat([cond["mask"], geom_mask], dim=1)
    if "null_vec" in cond:
        out["null_vec"] = cond["null_vec"] + geom
    if "null_tokens" in cond and "tokens" in out:
        geom_null = geom[:, None, :].to(dtype=cond["null_tokens"].dtype)
        out["null_tokens"] = torch.cat([cond["null_tokens"], geom_null], dim=1)
        if "null_mask" in cond:
            geom_null_mask = torch.zeros(
                (b, 1), dtype=torch.bool, device=cond["null_mask"].device)
            out["null_mask"] = torch.cat([cond["null_mask"], geom_null_mask], dim=1)
    return out


def attach_cached_image_geometry_condition(cond, payload, enabled=False, device=DEV):
    if not enabled:
        return cond
    geometry_features = payload.get("image_geometry_features")
    if geometry_features is not None:
        return attach_image_geometry_condition(
            cond, enabled=True, geometry_features=geometry_features.to(device=device))
    size_key = str(payload.get("size_bucket", "") or "")
    if size_key:
        features = geometry_features_from_size_key(
            size_key, condition_batch(cond), device=device)
        return attach_image_geometry_condition(
            cond, enabled=True, geometry_features=features)
    return attach_image_geometry_condition(
        cond, enabled=True, target_size=32, batch=condition_batch(cond))


def attach_image_geometry_mode(flow, enabled=False):
    setattr(flow, "uses_image_geometry_condition", bool(enabled))
    return flow


def flow_uses_image_geometry(flow):
    return bool(getattr(flow, "uses_image_geometry_condition", False))


def quality_sampling_weights(records, strength=0.0):
    rows = list(records)
    strength = float(strength)
    if strength < 0.0:
        raise ValueError("image_quality_weight must be non-negative")
    report = {
        "image_quality_weight": strength,
        "image_quality_weighted": False,
        "image_quality_weight_source": "",
        "image_quality_weight_records": 0,
        "image_quality_weight_missing": 0,
    }
    if strength <= 0.0 or not rows:
        return None, report
    vals = np.asarray([
        float(rec.aesthetic) if rec.aesthetic is not None else np.nan
        for rec in rows
    ], dtype=np.float64)
    finite = np.isfinite(vals)
    missing = int(len(vals) - int(finite.sum()))
    report["image_quality_weight_missing"] = missing
    if not finite.any():
        return None, report
    valid = vals[finite]
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    mean = float(np.mean(valid))
    report.update({
        "image_quality_score_min": lo,
        "image_quality_score_mean": mean,
        "image_quality_score_max": hi,
    })
    if hi <= lo:
        return None, report
    fill = float(np.median(valid))
    vals = np.where(finite, vals, fill)
    score = np.clip((vals - lo) / max(hi - lo, 1.0e-12), 0.0, 1.0)
    logw = strength * score
    logw -= float(np.max(logw))
    weights = np.exp(logw)
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
        return None, report
    report.update({
        "image_quality_weighted": True,
        "image_quality_weight_source": "aesthetic_score_quality",
        "image_quality_weight_records": int(len(rows)),
        "image_quality_weight_min": float(np.min(weights)),
        "image_quality_weight_mean": float(np.mean(weights)),
        "image_quality_weight_max": float(np.max(weights)),
        "image_quality_weight_ratio": float(np.max(weights) / max(np.min(weights), 1.0e-12)),
    })
    return weights.astype(np.float64), report


def record_source_key(rec):
    source = str(getattr(rec, "source", "") or "").strip()
    return source or "__unknown__"


def parse_source_weight_spec(spec):
    spec = str(spec or "").strip()
    if not spec:
        return {}
    if spec.startswith("{"):
        raw = json.loads(spec)
        if not isinstance(raw, dict):
            raise ValueError("image_source_weights JSON must be an object")
        items = raw.items()
    else:
        items = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                key, value = part.split("=", 1)
            elif ":" in part:
                key, value = part.split(":", 1)
            else:
                raise ValueError(
                    "image_source_weights entries must be source=weight or source:weight"
                )
            items.append((key, value))
    weights = {}
    for key, value in items:
        key = str(key).strip()
        if not key:
            raise ValueError("image_source_weights contains an empty source key")
        val = float(value)
        if val < 0.0:
            raise ValueError("image_source_weights must be non-negative")
        weights[key] = val
    return weights


def source_sampling_weights(records, spec=""):
    rows = list(records)
    counts = {}
    for rec in rows:
        key = record_source_key(rec)
        counts[key] = counts.get(key, 0) + 1
    weights_by_source = parse_source_weight_spec(spec)
    report = {
        "image_source_weights": dict(sorted(weights_by_source.items())),
        "image_source_weighted": False,
        "image_source_weight_records": int(len(rows)),
        "image_source_weight_default": float(weights_by_source.get("*", 1.0)),
        "image_source_counts": dict(sorted(counts.items())),
    }
    if not weights_by_source or not rows:
        return None, report
    default = float(weights_by_source.get("*", 1.0))
    weights = np.asarray([
        float(weights_by_source.get(record_source_key(rec), default))
        for rec in rows
    ], dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError("image_source_weights must be finite")
    if float(weights.sum()) <= 0.0:
        raise ValueError("image_source_weights give every manifest record zero probability")
    report.update({
        "image_source_weighted": True,
        "image_source_weight_min": float(np.min(weights)),
        "image_source_weight_mean": float(np.mean(weights)),
        "image_source_weight_max": float(np.max(weights)),
        "image_source_weight_ratio": float(
            np.max(weights) / max(np.min(weights), 1.0e-12)
        ),
    })
    return weights, report


def combine_sampling_weights(*weight_sets):
    arrays = [np.asarray(w, dtype=np.float64) for w in weight_sets if w is not None]
    if not arrays:
        return None
    shape = arrays[0].shape
    out = np.ones(shape, dtype=np.float64)
    for arr in arrays:
        if arr.shape != shape:
            raise ValueError(f"sampling weight shape mismatch: {arr.shape} vs {shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("sampling weights must be finite")
        if np.any(arr < 0.0):
            raise ValueError("sampling weights must be non-negative")
        out *= arr
    if float(out.sum()) <= 0.0:
        raise ValueError("combined sampling weights have zero probability mass")
    return out


def quality_score_stats(records):
    vals = np.asarray([
        float(rec.aesthetic) if rec.aesthetic is not None else np.nan
        for rec in records
    ], dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        return {
            "n": 0,
            "missing": int(len(vals)),
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "has_range": False,
        }
    valid = vals[finite]
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    return {
        "n": int(finite.sum()),
        "missing": int(len(vals) - int(finite.sum())),
        "min": lo,
        "mean": float(np.mean(valid)),
        "max": hi,
        "has_range": bool(hi > lo),
    }


def quality_target_tensor(records, stats, device=DEV):
    vals = torch.tensor([
        float(rec.aesthetic) if rec.aesthetic is not None else float("nan")
        for rec in records
    ], dtype=torch.float32, device=device)
    mask = torch.isfinite(vals)
    target = torch.zeros_like(vals)
    if bool(mask.any()):
        lo = float(stats.get("min", 0.0))
        hi = float(stats.get("max", lo))
        if hi > lo:
            target[mask] = ((vals[mask] - lo) / max(hi - lo, 1.0e-12)).clamp(0.0, 1.0)
        else:
            target[mask] = 0.5
    return target, mask


def image_quality_score_loss(scorer, z, records, stats, prefix="quality_score"):
    if scorer is None:
        zero = z.sum() * 0.0
        return zero, {}
    target, mask = quality_target_tensor(records, stats, device=z.device)
    return image_quality_score_target_loss(
        scorer, z, target, mask, prefix=prefix)


def image_quality_score_target_loss(scorer, z, target, mask, prefix="quality_score"):
    if scorer is None:
        zero = z.sum() * 0.0
        return zero, {}
    target = target.to(device=z.device, dtype=torch.float32).flatten()
    mask = mask.to(device=z.device, dtype=torch.bool).flatten()
    n = int(mask.sum().detach().cpu())
    if n <= 0:
        zero = z.sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_n": torch.tensor(0.0, device=z.device),
        }
    pred = scorer.score(z)
    loss = F.mse_loss(pred[mask], target[mask])
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_pred_mean": pred[mask].detach().mean(),
        f"{prefix}_target_mean": target[mask].detach().mean(),
        f"{prefix}_n": torch.tensor(float(n), device=z.device),
    }


def image_quality_rank_target_loss(scorer, z, target, mask, prefix="quality_rank",
                                   margin=0.0):
    """Pairwise quality preference loss from normalized manifest score targets."""
    if scorer is None:
        zero = z.sum() * 0.0
        return zero, {}
    target = target.to(device=z.device, dtype=torch.float32).flatten()
    mask = mask.to(device=z.device, dtype=torch.bool).flatten()
    valid_pairs = mask[:, None] & mask[None, :]
    target_gap = target[:, None] - target[None, :]
    prefer = valid_pairs & target_gap.gt(1.0e-6)
    n_pairs = int(prefer.sum().detach().cpu())
    if n_pairs <= 0:
        zero = z.sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_pairs": torch.tensor(0.0, device=z.device),
        }
    logits = scorer.logits(z)
    pred_gap = logits[:, None] - logits[None, :]
    pair_weights = target_gap[prefer].detach().clamp_min(1.0e-6)
    pair_loss = F.softplus(float(margin) - pred_gap[prefer])
    loss = (pair_loss * pair_weights).sum() / pair_weights.sum().clamp_min(1.0e-6)
    pair_acc = pred_gap[prefer].gt(0.0).to(torch.float32).mean()
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_pairs": torch.tensor(float(n_pairs), device=z.device),
        f"{prefix}_acc": pair_acc.detach(),
        f"{prefix}_target_gap_mean": target_gap[prefer].detach().mean(),
        f"{prefix}_pred_gap_mean": pred_gap[prefer].detach().mean(),
    }


def image_quality_rank_loss(scorer, z, records, stats, prefix="quality_rank", margin=0.0):
    if scorer is None:
        zero = z.sum() * 0.0
        return zero, {}
    target, mask = quality_target_tensor(records, stats, device=z.device)
    return image_quality_rank_target_loss(
        scorer, z, target, mask, prefix=prefix, margin=margin)


def _read_jsonl_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_relative_path(path, base=""):
    path = str(path)
    if not base or os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base, path))


def _optional_embedding_tuple(raw):
    if raw in ("", None):
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        raw = json.loads(raw)
    vals = tuple(float(x) for x in raw)
    return vals or None


def _optional_embedding_sequence(raw):
    if raw in ("", None):
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        raw = json.loads(raw)
    rows = tuple(tuple(float(x) for x in row) for row in raw)
    rows = tuple(row for row in rows if row)
    return rows or None


def read_image_preference_pairs(path, root="", max_pairs=0):
    """Read generic chosen/rejected image pairs for quality-scorer pretraining."""
    if not path:
        return []
    manifest_dir = os.path.dirname(os.path.abspath(path))
    base = root or manifest_dir
    pairs = []
    skipped = 0
    for row in _read_jsonl_rows(path):
        chosen = (
            row.get("chosen_image") or row.get("chosen") or row.get("winner_image")
            or row.get("preferred_image")
        )
        rejected = (
            row.get("rejected_image") or row.get("rejected") or row.get("loser_image")
            or row.get("nonpreferred_image")
        )
        if not chosen or not rejected:
            skipped += 1
            continue
        try:
            gap = float(row.get("score_gap", row.get("preference_gap", 1.0)) or 1.0)
        except (TypeError, ValueError):
            gap = 1.0
        if not math.isfinite(gap) or gap <= 0.0:
            gap = 1.0
        text_embedding = _optional_embedding_tuple(
            row.get("text_embedding") or row.get("prompt_embedding")
            or row.get("caption_embedding")
        )
        text_embedding_sequence = _optional_embedding_sequence(
            row.get("text_embedding_sequence") or row.get("prompt_embedding_sequence")
            or row.get("caption_embedding_sequence")
        )
        metadata = {
            str(k): v for k, v in row.items()
            if str(k).startswith("chosen_") or str(k).startswith("rejected_")
        }
        pairs.append({
            "prompt": str(row.get("prompt") or row.get("caption") or ""),
            "chosen_path": _resolve_relative_path(chosen, base=base),
            "rejected_path": _resolve_relative_path(rejected, base=base),
            "score_gap": float(gap),
            "text_embedding": text_embedding,
            "text_embedding_sequence": text_embedding_sequence,
            "source": str(row.get("preference_source") or row.get("source") or ""),
            "metadata": metadata,
        })
        if max_pairs and len(pairs) >= int(max_pairs):
            break
    if not pairs:
        raise ValueError(
            f"image preference manifest {path!r} yielded no chosen/rejected pairs "
            f"(skipped {skipped})")
    return pairs


def image_preference_pair_report(pairs, caption_cond_source="tokens"):
    rows = list(pairs or [])
    n = len(rows)
    with_prompt = sum(1 for row in rows if str(row.get("prompt") or "").strip())
    with_pooled = sum(1 for row in rows if row.get("text_embedding") is not None)
    with_sequence = sum(1 for row in rows if row.get("text_embedding_sequence") is not None)
    with_sample_metadata = sum(1 for row in rows if row.get("metadata"))

    def metadata_counts(key):
        out = defaultdict(int)
        for row in rows:
            val = (row.get("metadata") or {}).get(key)
            if val not in ("", None):
                out[str(val)] += 1
        return dict(sorted(out.items()))

    source = str(caption_cond_source or "tokens")
    if source == "embedding":
        usable = sum(
            1 for row in rows
            if row.get("text_embedding_sequence") is not None
            or row.get("text_embedding") is not None)
    else:
        usable = with_prompt
    return {
        "image_preference_pairs_with_prompt": int(with_prompt),
        "image_preference_pairs_with_text_embedding": int(with_pooled),
        "image_preference_pairs_with_text_embedding_sequence": int(with_sequence),
        "image_preference_pairs_direct_flow_usable": int(usable),
        "image_preference_pairs_direct_flow_missing_condition": int(max(0, n - usable)),
        "image_preference_pairs_with_sample_metadata": int(with_sample_metadata),
        "image_preference_chosen_sample_methods": metadata_counts(
            "chosen_sample_method"),
        "image_preference_rejected_sample_methods": metadata_counts(
            "rejected_sample_method"),
        "image_preference_chosen_cfg_schedules": metadata_counts(
            "chosen_sample_cfg_schedule"),
        "image_preference_rejected_cfg_schedules": metadata_counts(
            "rejected_sample_cfg_schedule"),
    }


def sample_image_preference_pair_batch(pairs, rng, batch=32, size=32, device=DEV,
                                       crop_mode="center", hflip_prob=0.0,
                                       return_pairs=False):
    rows = list(pairs)
    if not rows:
        raise ValueError("no image preference pairs available")
    idx = rng.integers(0, len(rows), size=max(1, int(batch)))
    chosen = []
    rejected = []
    gaps = []
    selected_pairs = []
    for i in idx:
        pair = rows[int(i)]
        flip = bool(float(hflip_prob) > 0.0 and rng.random() < float(hflip_prob))
        chosen.append(load_image_tensor(
            pair["chosen_path"], size=size, device=device, center_crop=(crop_mode == "center"),
            crop_mode=crop_mode, hflip=flip, rng=rng))
        rejected.append(load_image_tensor(
            pair["rejected_path"], size=size, device=device, center_crop=(crop_mode == "center"),
            crop_mode=crop_mode, hflip=flip, rng=rng))
        gaps.append(float(pair.get("score_gap", 1.0)))
        selected_pairs.append(pair)
    out = (
        torch.stack(chosen, dim=0),
        torch.stack(rejected, dim=0),
        torch.tensor(gaps, dtype=torch.float32, device=device),
    )
    if return_pairs:
        return out + (selected_pairs,)
    return out


def image_quality_preference_pair_loss(scorer, z_chosen, z_rejected, score_gaps=None,
                                       prefix="quality_preference", margin=0.0):
    """Pairwise preference loss from chosen/rejected image latents."""
    if scorer is None:
        zero = z_chosen.sum() * 0.0
        return zero, {}
    chosen_logits = scorer.logits(z_chosen)
    rejected_logits = scorer.logits(z_rejected)
    pred_gap = chosen_logits - rejected_logits
    if score_gaps is None:
        weights = torch.ones_like(pred_gap)
    else:
        weights = score_gaps.to(device=pred_gap.device, dtype=torch.float32).flatten()
        if weights.numel() != pred_gap.numel():
            weights = torch.ones_like(pred_gap)
        weights = weights.clamp_min(1.0e-6)
        weights = weights / weights.mean().clamp_min(1.0e-6)
    pair_loss = F.softplus(float(margin) - pred_gap)
    loss = (pair_loss * weights).mean()
    acc = pred_gap.gt(0.0).to(torch.float32).mean()
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_pairs": torch.tensor(float(pred_gap.numel()), device=pred_gap.device),
        f"{prefix}_acc": acc.detach(),
        f"{prefix}_pred_gap_mean": pred_gap.detach().mean(),
        f"{prefix}_score_gap_mean": weights.detach().mean(),
    }


def image_preference_pair_condition(pairs, conditioner, prompt_vocab, source="tokens",
                                    caption_max_len=64, device=DEV, return_tokens=False):
    source = str(source or "tokens")
    if source == "embedding":
        seqs = [pair.get("text_embedding_sequence") for pair in pairs]
        if all(seq is not None for seq in seqs):
            max_len = max((len(seq) for seq in seqs), default=1)
            dim = len(seqs[0][0]) if seqs and seqs[0] else 0
            if dim <= 0:
                return None
            embs = torch.zeros((len(seqs), max(1, max_len), dim),
                               dtype=torch.float32, device=device)
            for i, seq in enumerate(seqs):
                for j, row in enumerate(seq):
                    embs[i, j] = torch.tensor(row, dtype=torch.float32, device=device)
            return conditioner(embs, return_tokens=text_condition_return_tokens(
                conditioner, return_tokens=return_tokens))
        pooled = [pair.get("text_embedding") for pair in pairs]
        if not all(vec is not None for vec in pooled):
            return None
        embs = torch.tensor(pooled, dtype=torch.float32, device=device)
        return conditioner(embs, return_tokens=text_condition_return_tokens(
            conditioner, return_tokens=return_tokens))
    prompts = [str(pair.get("prompt") or "") for pair in pairs]
    return caption_condition(prompts, conditioner, prompt_vocab, max_len=caption_max_len,
                             device=device, return_tokens=return_tokens)


def latent_flow_preference_pair_loss(flow, z_chosen, z_rejected, cond, score_gaps=None,
                                     cond_drop=0.0, time_sampling="uniform",
                                     time_logit_mean=0.0, time_logit_std=1.0,
                                     time_mode_scale=DEFAULT_TIME_MODE_SCALE,
                                     time_shift=1.0, latent_stats=None,
                                     flow_loss_weight="none",
                                     flow_loss_weight_gamma=5.0,
                                     flow_loss_weight_normalize=True,
                                     margin=0.0, prefix="flow_preference",
                                     adaptive_sampler=None):
    """Direct pairwise preference loss on the generator's rectified-flow objective."""
    if z_chosen.shape != z_rejected.shape:
        raise ValueError(
            f"preference latent shapes differ: {tuple(z_chosen.shape)} vs "
            f"{tuple(z_rejected.shape)}")
    latent_stats = flow_latent_stats(flow) if latent_stats is None else latent_stats
    zc = normalize_latent(z_chosen, latent_stats)
    zr = normalize_latent(z_rejected, latent_stats)
    x0 = torch.randn_like(zc)
    t = sample_flow_times(zc.shape[0], device=zc.device, mode=time_sampling,
                          logit_mean=time_logit_mean, logit_std=time_logit_std,
                          mode_scale=time_mode_scale, time_shift=time_shift,
                          adaptive_sampler=adaptive_sampler)
    cond_model = condition_dropout(cond, cond_drop)
    ztc = (1.0 - t) * x0 + t * zc
    ztr = (1.0 - t) * x0 + t * zr
    target_c = zc - x0
    target_r = zr - x0
    pred_c = flow_velocity(flow, ztc, t, cond_model)
    pred_r = flow_velocity(flow, ztr, t, cond_model)
    loss_c = (pred_c - target_c).float().pow(2).flatten(1).mean(dim=1)
    loss_r = (pred_r - target_r).float().pow(2).flatten(1).mean(dim=1)
    if flow_loss_weight != "none":
        weights = flow_loss_time_weights(
            t, mode=flow_loss_weight, gamma=flow_loss_weight_gamma,
            normalize=flow_loss_weight_normalize).to(loss_c.dtype)
        loss_c = loss_c * weights
        loss_r = loss_r * weights
    if score_gaps is None:
        gap_weight = torch.ones_like(loss_c)
    else:
        gap_weight = torch.log1p(score_gaps.to(device=loss_c.device).float()).clamp(0.1, 5.0)
    logits = (loss_r - loss_c - float(margin)) * gap_weight
    loss = F.softplus(-logits).mean()
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_chosen_velocity_mse": loss_c.mean().detach(),
        f"{prefix}_rejected_velocity_mse": loss_r.mean().detach(),
        f"{prefix}_loss_gap": (loss_r - loss_c).mean().detach(),
        f"{prefix}_acc": loss_c.lt(loss_r).float().mean().detach(),
        f"{prefix}_margin": torch.tensor(float(margin), device=loss_c.device),
        f"{prefix}_gap_weight_mean": gap_weight.mean().detach(),
        f"{prefix}_pairs": torch.tensor(float(loss_c.numel()), device=loss_c.device),
    }


def record_weight_lookup(records, weights):
    if weights is None:
        return None
    return {id(rec): float(w) for rec, w in zip(records, weights)}


def weights_for_records(records, record_weights):
    if record_weights is None:
        return None
    weights = [float(record_weights.get(id(rec), 1.0)) for rec in records]
    return np.asarray(weights, dtype=np.float64)


def cat_text_embedding_tensors(chunks):
    chunks = [x for x in chunks if x is not None]
    if not chunks:
        return None
    if all(x.ndim == 2 for x in chunks):
        return torch.cat(chunks, dim=0).contiguous()
    dims = {int(x.shape[-1]) for x in chunks}
    if len(dims) != 1:
        raise ValueError(f"text embedding chunks have mixed dimensions: {sorted(dims)}")
    max_len = max(int(x.shape[1]) if x.ndim == 3 else 1 for x in chunks)
    padded = []
    for x in chunks:
        if x.ndim == 2:
            x = x[:, None, :]
        elif x.ndim != 3:
            raise ValueError(f"expected text embeddings as B,D or B,T,D, got {tuple(x.shape)}")
        if int(x.shape[1]) < max_len:
            pad = torch.zeros(
                (int(x.shape[0]), max_len - int(x.shape[1]), int(x.shape[2])),
                dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        padded.append(x)
    return torch.cat(padded, dim=0).contiguous()


def sample_bucketed_image_text_batch(records, rng, batch=32, size_buckets=(), bucket_records=None,
                                     device=DEV, return_records=False, crop_mode="center",
                                     hflip_prob=0.0, record_weights=None,
                                     bucket_probs=None, return_metadata=False):
    bucket = choose_image_size_bucket_weighted(rng, size_buckets, bucket_probs=bucket_probs)
    rows = list(records)
    if bucket_records is not None:
        rows = list(bucket_records.get(bucket) or rows)
    if not rows:
        rows = list(records)
    weights = weights_for_records(rows, record_weights)
    payload = sample_image_text_batch(
        rows, rng, batch=batch, size=bucket, device=device, return_records=return_records,
        crop_mode=crop_mode, hflip_prob=hflip_prob, weights=weights,
        return_metadata=return_metadata)
    return (bucket,) + payload


def ae_latent_shape(ae, size):
    downsample = int(getattr(ae, "downsample", 4))
    h, w = image_hw(size)
    if h % downsample or w % downsample:
        raise ValueError(
            f"image size {h}x{w} must be divisible by AE downsample {downsample}"
        )
    return int(ae.latent_ch), int(h) // downsample, int(w) // downsample


class PromptConditioner(nn.Module):
    """Small learned text encoder that maps prompt tokens to continuous conditions."""

    def __init__(self, vocab_size, cond_dim=64, hidden=64, heads=4, max_len=32, pad=0):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.pad = int(pad)
        self.max_len = int(max_len)
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=pad)
        self.pos = nn.Embedding(max_len, hidden)
        self.null_vec = nn.Parameter(torch.zeros(cond_dim))
        self.null_tokens = nn.Parameter(torch.zeros(max_len, cond_dim))
        heads = max(1, min(heads, hidden))
        while hidden % heads:
            heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, cond_dim))

    def null_token_rows(self, length, device=None, dtype=None):
        length = int(length)
        rows = self.null_tokens
        if length <= rows.shape[0]:
            out = rows[:length]
        else:
            tail = rows[-1:].expand(length - rows.shape[0], -1)
            out = torch.cat([rows, tail], dim=0)
        return out.to(device=device, dtype=dtype)

    def forward(self, ids, return_tokens=False):
        b, l = ids.shape
        pos = torch.arange(l, device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        mask = ids.eq(self.pad)
        h = self.emb(ids) + self.pos(pos)[None]
        h = self.enc(h, src_key_padding_mask=mask)
        keep = (~mask).to(h.dtype).unsqueeze(-1)
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        vec = self.out(pooled)
        if return_tokens:
            tokens = self.out(h)
            null_vec = self.null_vec.to(device=ids.device, dtype=vec.dtype)[None].expand(b, -1)
            null_tokens = self.null_token_rows(
                l, device=ids.device, dtype=tokens.dtype)[None].expand(b, -1, -1)
            null_mask = torch.zeros((b, l), dtype=torch.bool, device=ids.device)
            return {
                "vec": vec,
                "tokens": tokens,
                "mask": mask,
                "null_vec": null_vec,
                "null_tokens": null_tokens,
                "null_mask": null_mask,
            }
        return vec


class PrecomputedTextConditioner(nn.Module):
    """Project external caption embeddings into the model's conditioning space."""

    def __init__(self, input_dim, cond_dim=64, hidden=64, pooled_input_dim=0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.pooled_input_dim = int(pooled_input_dim or 0)
        self.cond_dim = int(cond_dim)
        self.null_vec = nn.Parameter(torch.zeros(cond_dim))
        self.null_token = nn.Parameter(torch.zeros(cond_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.cond_dim),
        )
        self.pooled_net = (
            nn.Sequential(
                nn.LayerNorm(self.pooled_input_dim),
                nn.Linear(self.pooled_input_dim, int(hidden)),
                nn.GELU(),
                nn.Linear(int(hidden), self.cond_dim),
            )
            if self.pooled_input_dim > 0 else None
        )

    def _forward_tensor(self, embeddings, return_tokens=False):
        embeddings = embeddings.float()
        if embeddings.ndim == 3:
            b, l = int(embeddings.shape[0]), int(embeddings.shape[1])
            tokens = self.net(embeddings)
            mask = embeddings.abs().sum(dim=-1).eq(0)
            keep = (~mask).to(tokens.dtype).unsqueeze(-1)
            pooled = (tokens * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
            if return_tokens:
                null_vec = self.null_vec.to(
                    device=embeddings.device, dtype=pooled.dtype)[None].expand(b, -1)
                null_tokens = self.null_token.to(
                    device=embeddings.device, dtype=tokens.dtype)[None, None, :].expand(
                        b, l, -1)
                null_mask = torch.zeros((b, l), dtype=torch.bool, device=embeddings.device)
                return {"vec": pooled, "tokens": tokens.masked_fill(mask[:, :, None], 0.0),
                        "mask": mask, "null_vec": null_vec,
                        "null_tokens": null_tokens, "null_mask": null_mask}
            return pooled
        vec = self.net(embeddings)
        if return_tokens:
            b = int(vec.shape[0])
            mask = torch.zeros((vec.shape[0], 1), dtype=torch.bool, device=vec.device)
            null_vec = self.null_vec.to(device=vec.device, dtype=vec.dtype)[None].expand(b, -1)
            null_tokens = self.null_token.to(
                device=vec.device, dtype=vec.dtype)[None, None, :].expand(b, 1, -1)
            null_mask = torch.zeros((b, 1), dtype=torch.bool, device=vec.device)
            return {"vec": vec, "tokens": vec[:, None, :], "mask": mask,
                    "null_vec": null_vec, "null_tokens": null_tokens,
                    "null_mask": null_mask}
        return vec

    def forward(self, embeddings, return_tokens=False):
        if isinstance(embeddings, dict):
            sequence = embeddings.get("sequence", embeddings.get("tokens"))
            pooled_raw = embeddings.get("pooled")
            pooled_mask = embeddings.get("pooled_mask")
            seq_cond = (
                self._forward_tensor(sequence, return_tokens=True)
                if sequence is not None else None
            )
            pooled_vec = None
            if pooled_raw is not None:
                pooled_raw = pooled_raw.float()
                if self.pooled_net is not None:
                    pooled_vec = self.pooled_net(pooled_raw)
                else:
                    pooled_vec = self.net(pooled_raw)
                raw_missing = pooled_raw.abs().sum(dim=-1).eq(0)
                pooled_mask = (
                    raw_missing if pooled_mask is None
                    else pooled_mask.to(device=pooled_vec.device, dtype=torch.bool) | raw_missing
                )
            if seq_cond is None:
                if pooled_vec is None:
                    raise ValueError("precomputed text condition payload is empty")
                if return_tokens:
                    b = int(pooled_vec.shape[0])
                    mask = (
                        pooled_mask if pooled_mask is not None
                        else torch.zeros((b,), dtype=torch.bool, device=pooled_vec.device)
                    )
                    return {
                        "vec": pooled_vec,
                        "tokens": pooled_vec[:, None, :],
                        "mask": mask[:, None],
                        "null_vec": self.null_vec.to(
                            device=pooled_vec.device, dtype=pooled_vec.dtype
                        )[None].expand(b, -1),
                        "null_tokens": self.null_token.to(
                            device=pooled_vec.device, dtype=pooled_vec.dtype
                        )[None, None, :].expand(b, 1, -1),
                        "null_mask": torch.zeros((b, 1), dtype=torch.bool,
                                                 device=pooled_vec.device),
                    }
                return pooled_vec
            if pooled_vec is None:
                return seq_cond if return_tokens else seq_cond["vec"]
            seq_vec = seq_cond["vec"]
            pooled_keep = (
                torch.ones(seq_vec.shape[0], device=seq_vec.device, dtype=seq_vec.dtype)
                if pooled_mask is None else (~pooled_mask).to(seq_vec.dtype)
            )
            vec = (
                seq_vec + pooled_vec.to(seq_vec.dtype) * pooled_keep[:, None]
            ) / (1.0 + pooled_keep[:, None]).clamp_min(1.0)
            if not return_tokens:
                return vec
            pooled_token = pooled_vec.to(seq_cond["tokens"].dtype)[:, None, :]
            pooled_token_mask = (
                torch.zeros((seq_vec.shape[0], 1), dtype=torch.bool, device=seq_vec.device)
                if pooled_mask is None else pooled_mask[:, None].to(device=seq_vec.device)
            )
            tokens = torch.cat([pooled_token, seq_cond["tokens"]], dim=1)
            mask = torch.cat([pooled_token_mask, seq_cond["mask"]], dim=1)
            null_tokens = self.null_token.to(
                device=tokens.device, dtype=tokens.dtype)[None, None, :].expand(
                    tokens.shape[0], tokens.shape[1], -1)
            return {
                "vec": vec,
                "tokens": tokens.masked_fill(mask[:, :, None], 0.0),
                "mask": mask,
                "null_vec": seq_cond["null_vec"],
                "null_tokens": null_tokens,
                "null_mask": torch.zeros_like(mask),
            }
        return self._forward_tensor(embeddings, return_tokens=return_tokens)


class ImageTextAligner(nn.Module):
    """Contrastive bridge between image latents and caption conditions.

    This is the manifest-data equivalent of the synthetic fact heads: it gives real-image runs a
    generic paired image/text semantic signal without renderer-specific labels.
    """

    def __init__(self, latent_ch=16, cond_dim=64, hidden=64, embed_dim=128,
                 temperature=0.07):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.image = nn.Sequential(
            nn.LayerNorm(int(latent_ch)),
            nn.Linear(int(latent_ch), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        self.text = nn.Sequential(
            nn.LayerNorm(int(cond_dim)),
            nn.Linear(int(cond_dim), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        temp = max(float(temperature), 1.0e-4)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))

    def encode_image(self, z):
        pooled = z.float().mean(dim=(2, 3))
        return F.normalize(self.image(pooled), dim=-1)

    def encode_text(self, cond):
        vec = condition_vector(cond).float()
        return F.normalize(self.text(vec), dim=-1)

    def forward(self, z, cond):
        return self.encode_image(z), self.encode_text(cond)

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)


class ImageFeatureAligner(nn.Module):
    """Contrastive bridge between AE latents and external image features."""

    def __init__(self, latent_ch=16, feature_dim=768, hidden=64, embed_dim=128,
                 temperature=0.07):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim)
        self.image = nn.Sequential(
            nn.LayerNorm(int(latent_ch)),
            nn.Linear(int(latent_ch), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        self.feature = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        temp = max(float(temperature), 1.0e-4)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))

    def encode_image(self, z):
        pooled = z.float().mean(dim=(2, 3))
        return F.normalize(self.image(pooled), dim=-1)

    def encode_feature(self, features):
        return F.normalize(self.feature(features.float()), dim=-1)

    def forward(self, z, features):
        return self.encode_image(z), self.encode_feature(features)

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)


class ImageQualityScorer(nn.Module):
    """Predict normalized aesthetic/quality metadata from image latents."""

    def __init__(self, latent_ch=16, hidden=64):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.net = nn.Sequential(
            nn.LayerNorm(self.latent_ch),
            nn.Linear(self.latent_ch, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    def logits(self, z):
        pooled = z.float().mean(dim=(2, 3))
        return self.net(pooled).squeeze(-1)

    def score(self, z):
        return torch.sigmoid(self.logits(z))

    def forward(self, z):
        return self.score(z)


class FlowFeatureAligner(nn.Module):
    """Contrastive bridge between denoiser hidden states and external image features.

    This is the REPA-style training signal: it aligns the flow transformer's noisy-step
    representation to a generic visual embedding without changing the generated latent target.
    """

    def __init__(self, hidden_dim=96, feature_dim=768, hidden=96, embed_dim=128,
                 temperature=0.07):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim)
        self.hidden = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        self.feature = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        temp = max(float(temperature), 1.0e-4)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))

    def encode_hidden(self, tokens):
        if tokens.ndim == 3:
            pooled = tokens.float().mean(dim=1)
        elif tokens.ndim == 2:
            pooled = tokens.float()
        else:
            raise ValueError(f"expected hidden tokens as B,N,H or B,H, got {tuple(tokens.shape)}")
        return F.normalize(self.hidden(pooled), dim=-1)

    def encode_hidden_tokens(self, tokens):
        if tokens.ndim == 2:
            tokens = tokens[:, None, :]
        if tokens.ndim != 3:
            raise ValueError(f"expected hidden tokens as B,N,H or B,H, got {tuple(tokens.shape)}")
        return F.normalize(self.hidden(tokens.float()), dim=-1)

    def encode_feature(self, features):
        if features.ndim == 3:
            features = pool_embedding_sequence(features)
        return F.normalize(self.feature(features.float()), dim=-1)

    def encode_feature_tokens(self, features):
        if features.ndim == 2:
            features = features[:, None, :]
        if features.ndim != 3:
            raise ValueError(f"expected feature tokens as B,M,D or B,D, got {tuple(features.shape)}")
        return F.normalize(self.feature(features.float()), dim=-1)

    def forward(self, tokens, features):
        return self.encode_hidden(tokens), self.encode_feature(features)

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)


class FlowLatentAligner(nn.Module):
    """Align denoiser hidden states to clean latent patches from the same image.

    This is a no-extra-data REPA-style signal. External REPA needs manifest image
    embeddings; this path only needs the frozen AE latent endpoint, so it works for
    synthetic factors, caption manifests, and cached real-image latents.
    """

    def __init__(self, hidden_dim=96, latent_ch=16, patch_size=1, hidden=96,
                 embed_dim=128, temperature=0.07):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_ch = int(latent_ch)
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError("latent alignment patch_size must be positive")
        self.latent_token_dim = self.latent_ch * self.patch_size * self.patch_size
        self.embed_dim = int(embed_dim)
        self.hidden = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        self.latent = nn.Sequential(
            nn.LayerNorm(self.latent_token_dim),
            nn.Linear(self.latent_token_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        temp = max(float(temperature), 1.0e-4)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))

    def encode_hidden(self, tokens):
        if tokens.ndim == 3:
            pooled = tokens.float().mean(dim=1)
        elif tokens.ndim == 2:
            pooled = tokens.float()
        else:
            raise ValueError(f"expected hidden tokens as B,N,H or B,H, got {tuple(tokens.shape)}")
        return F.normalize(self.hidden(pooled), dim=-1)

    def encode_hidden_tokens(self, tokens):
        if tokens.ndim == 2:
            tokens = tokens[:, None, :]
        if tokens.ndim != 3:
            raise ValueError(f"expected hidden tokens as B,N,H or B,H, got {tuple(tokens.shape)}")
        return F.normalize(self.hidden(tokens.float()), dim=-1)

    def latent_tokens(self, z):
        tokens, _th, _tw = patchify_latents(z.float(), patch_size=self.patch_size)
        return tokens

    def encode_latent(self, z):
        pooled = self.latent_tokens(z).mean(dim=1)
        return F.normalize(self.latent(pooled), dim=-1)

    def encode_latent_tokens(self, z):
        return F.normalize(self.latent(self.latent_tokens(z)), dim=-1)

    def forward(self, tokens, z):
        return self.encode_hidden(tokens), self.encode_latent(z)

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)


class LatentFlowNet(nn.Module):
    """Velocity field v_theta(z_t, t, canonical_facts)."""

    def __init__(self, latent_ch=16, hidden=64, cond_dim=None):
        super().__init__()
        cond_dim = int(cond_dim or hidden)
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch + 1 + hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, latent_ch, 3, padding=1),
        )

    def forward(self, z, t, cond):
        cond = condition_vector(cond)
        if t.ndim == 1:
            t = t[:, None, None, None]
        if t.ndim == 2:
            t = t[:, :, None, None]
        b, _, h, w = z.shape
        tchan = t.expand(b, 1, h, w)
        c = self.cond(cond).view(b, -1, 1, 1).expand(b, -1, h, w)
        return self.net(torch.cat([z, tchan, c], dim=1))


def make_velocity_head(hidden, latent_ch, width_mult=1):
    width_mult = int(width_mult)
    if width_mult <= 1:
        return nn.Linear(hidden, latent_ch)
    width = int(hidden) * width_mult
    return nn.Sequential(
        nn.Linear(hidden, width),
        nn.GELU(),
        nn.Linear(width, width),
        nn.GELU(),
        nn.Linear(width, latent_ch),
    )


class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden, expansion=4):
        super().__init__()
        inner = max(1, int(int(hidden) * int(expansion) * 2 / 3))
        self.proj = nn.Linear(hidden, inner * 2)
        self.out = nn.Linear(inner, hidden)

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(value * F.silu(gate))


def make_dit_feed_forward(hidden, mlp="gelu"):
    mlp = str(mlp)
    if mlp == "gelu":
        return nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
    if mlp == "swiglu":
        return SwiGLUFeedForward(hidden)
    raise ValueError(f"unknown DiT MLP {mlp!r}")


def latent_patch_grid(height, width, patch_size=1):
    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError("latent_patch_size must be positive")
    height = int(height)
    width = int(width)
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"latent grid {height}x{width} is not divisible by latent_patch_size={patch_size}")
    return height // patch_size, width // patch_size


def latent_token_count(latent_shape, patch_size=1):
    _c, h, w = latent_shape
    th, tw = latent_patch_grid(h, w, patch_size=patch_size)
    return int(th) * int(tw)


def patchify_latents(z, patch_size=1):
    patch_size = int(patch_size)
    b, c, h, w = z.shape
    th, tw = latent_patch_grid(h, w, patch_size=patch_size)
    if patch_size == 1:
        return z.flatten(2).transpose(1, 2), th, tw
    return (
        z.reshape(b, c, th, patch_size, tw, patch_size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b, th * tw, c * patch_size * patch_size),
        th,
        tw,
    )


def unpatchify_latents(tokens, latent_ch, height, width, patch_size=1):
    patch_size = int(patch_size)
    b = tokens.shape[0]
    th, tw = latent_patch_grid(height, width, patch_size=patch_size)
    if patch_size == 1:
        return tokens.transpose(1, 2).reshape(b, latent_ch, height, width)
    return (
        tokens.reshape(b, th, tw, latent_ch, patch_size, patch_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, latent_ch, height, width)
    )


def sinusoidal_1d_positions(length, dim, device=None, dtype=None):
    dim = int(dim)
    if dim <= 0:
        return torch.empty((int(length), 0), device=device, dtype=dtype or torch.float32)
    pos = torch.arange(int(length), device=device, dtype=torch.float32)[:, None]
    freqs = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    freqs = torch.exp(-math.log(10000.0) * freqs / float(max(1, dim)))
    angles = pos * freqs[None, :]
    out = torch.zeros((int(length), dim), device=device, dtype=torch.float32)
    out[:, 0::2] = torch.sin(angles)
    out[:, 1::2] = torch.cos(angles[:, :out[:, 1::2].shape[1]])
    return out.to(dtype=dtype) if dtype is not None else out


def sinusoidal_2d_positions(height, width, dim, device=None, dtype=None):
    dim = int(dim)
    h_dim = dim // 2
    w_dim = dim - h_dim
    y = sinusoidal_1d_positions(height, h_dim, device=device, dtype=dtype)
    x = sinusoidal_1d_positions(width, w_dim, device=device, dtype=dtype)
    y = y[:, None, :].expand(int(height), int(width), h_dim)
    x = x[None, :, :].expand(int(height), int(width), w_dim)
    return torch.cat([y, x], dim=-1).reshape(1, int(height) * int(width), dim)


def flow_time_features(t, mode="scalar", dim=1, dtype=None):
    mode = str(mode or "scalar")
    if mode not in FLOW_TIME_EMBEDS:
        raise ValueError(f"unknown flow time embedding {mode!r}")
    if t.ndim > 2:
        t = t.flatten(1)[:, :1]
    elif t.ndim == 1:
        t = t[:, None]
    t = t.float()
    if mode == "scalar":
        return t.to(dtype=dtype) if dtype is not None else t
    dim = int(dim)
    if dim <= 1:
        return t.to(dtype=dtype) if dtype is not None else t
    base_dim = dim - 1
    half = max(1, base_dim // 2)
    freqs = torch.exp(
        torch.linspace(0.0, math.log(10000.0), half, device=t.device, dtype=torch.float32))
    angles = t * freqs[None, :] * (2.0 * math.pi)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if int(emb.shape[1]) < base_dim:
        emb = F.pad(emb, (0, base_dim - int(emb.shape[1])), value=0.0)
    emb = emb[:, :base_dim]
    out = torch.cat([t, emb], dim=1)
    return out.to(dtype=dtype) if dtype is not None else out


def rotary_2d_factors(height, width, head_dim, device=None, dtype=None):
    head_dim = int(head_dim)
    if head_dim % 2:
        raise ValueError("rope2d requires an even attention head dimension")
    total_pairs = head_dim // 2
    y_pairs = total_pairs // 2
    x_pairs = total_pairs - y_pairs
    y_pos = torch.arange(int(height), device=device, dtype=torch.float32)
    x_pos = torch.arange(int(width), device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_pos, x_pos, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    cos = torch.ones((int(height) * int(width), head_dim), device=device, dtype=torch.float32)
    sin = torch.zeros_like(cos)

    def fill_axis(start, pairs, pos):
        if pairs <= 0:
            return
        freqs = torch.arange(pairs, device=device, dtype=torch.float32)
        freqs = torch.exp(-math.log(10000.0) * freqs / float(max(1, pairs)))
        angles = pos[:, None] * freqs[None, :]
        axis_cos = torch.cos(angles)
        axis_sin = torch.sin(angles)
        end = start + 2 * pairs
        cos[:, start:end:2] = axis_cos
        cos[:, start + 1:end:2] = axis_cos
        sin[:, start:end:2] = axis_sin
        sin[:, start + 1:end:2] = axis_sin

    fill_axis(0, y_pairs, yy)
    fill_axis(2 * y_pairs, x_pairs, xx)
    cos = cos.reshape(1, 1, int(height) * int(width), head_dim)
    sin = sin.reshape(1, 1, int(height) * int(width), head_dim)
    if dtype is not None:
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
    return cos, sin


def apply_rotary_factors(x, cos, sin):
    if x.shape[-1] % 2:
        raise ValueError("rotary embedding requires an even feature dimension")
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_even = cos[..., 0::2].to(dtype=x.dtype, device=x.device)
    sin_even = sin[..., 0::2].to(dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos_even - x_odd * sin_even
    out[..., 1::2] = x_even * sin_even + x_odd * cos_even
    return out


class AdaDiTBlock(nn.Module):
    """Self-attention DiT block with adaLN-Zero residual gating."""

    def __init__(self, hidden=96, heads=4, mlp="gelu"):
        super().__init__()
        mlp = str(mlp)
        if mlp not in DIT_MLPS:
            raise ValueError(f"unknown DiT MLP {mlp!r}")
        self.dit_mlp = mlp
        self.uses_zero_residual_gating = True
        self.attn_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = make_dit_feed_forward(hidden, mlp=mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 4))
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 2))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x, cond_ctx):
        attn_shift, attn_scale, ff_shift, ff_scale = self.ada(cond_ctx).chunk(4, dim=-1)
        attn_gate, ff_gate = self.gate(cond_ctx).chunk(2, dim=-1)
        q = self._modulate(self.attn_norm(x), attn_shift, attn_scale)
        x = x + attn_gate[:, None, :] * self.attn(q, q, q, need_weights=False)[0]
        ff = self._modulate(self.ff_norm(x), ff_shift, ff_scale)
        return x + ff_gate[:, None, :] * self.ff(ff)


class LatentDiTFlowNet(nn.Module):
    """Patch-token transformer velocity field over semantic image latents.

    This is intentionally tiny, but it matches the scalable architecture shape: flatten the
    autoencoder latent grid into tokens, condition every token on time + canonical facts, run
    transformer blocks, then project token velocities back to the latent grid.
    """

    def __init__(self, latent_ch=16, hidden=96, depth=3, heads=4, cond_dim=None,
                 max_tokens=256, head_width_mult=1, pos_embed="learned",
                 checkpoint_blocks=False, latent_patch_size=1,
                 time_embed="scalar", time_embed_dim=1, mlp="gelu"):
        super().__init__()
        latent_patch_size = int(latent_patch_size)
        if latent_patch_size <= 0:
            raise ValueError("latent_patch_size must be positive")
        mlp = str(mlp)
        if mlp not in DIT_MLPS:
            raise ValueError(f"unknown DiT MLP {mlp!r}")
        pos_embed = str(pos_embed)
        if pos_embed not in DIT_POS_EMBEDS:
            raise ValueError(f"unknown DiT positional embedding {pos_embed!r}")
        if pos_embed == "rope2d":
            raise ValueError("rope2d positional embedding is only supported by MM-DiT")
        cond_dim = int(cond_dim or hidden)
        time_embed = str(time_embed or "scalar")
        if time_embed not in FLOW_TIME_EMBEDS:
            raise ValueError(f"unknown flow time embedding {time_embed!r}")
        self.latent_ch = int(latent_ch)
        self.latent_patch_size = latent_patch_size
        self.token_dim = self.latent_ch * self.latent_patch_size * self.latent_patch_size
        self.hidden = int(hidden)
        self.hidden_feature_dim = int(hidden)
        self.max_tokens = int(max_tokens)
        self.head_width_mult = int(head_width_mult)
        self.dit_pos_embed = pos_embed
        self.dit_mlp = mlp
        self.flow_time_embed = time_embed
        self.flow_time_embed_dim = int(time_embed_dim if time_embed != "scalar" else 1)
        self.uses_adaptive_modulation = True
        self.uses_residual_gating = True
        self.uses_zero_residual_gating = True
        self.uses_2d_pos_embed = pos_embed == "sincos2d"
        self.uses_rope2d_pos_embed = False
        self.uses_swiglu_mlp = mlp == "swiglu"
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.uses_activation_checkpointing = self.checkpoint_blocks
        self.in_proj = nn.Linear(self.token_dim, hidden)
        self.pos = (
            nn.Parameter(torch.zeros(1, max_tokens, hidden))
            if pos_embed == "learned" else None
        )
        self.cond = nn.Sequential(
            nn.Linear(cond_dim + self.flow_time_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([
            AdaDiTBlock(hidden, heads, mlp=mlp) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = make_velocity_head(hidden, self.token_dim, self.head_width_mult)

    def image_pos(self, h, w, device, dtype):
        if self.dit_pos_embed == "learned":
            return self.pos[:, :int(h) * int(w)].to(device=device, dtype=dtype)
        return sinusoidal_2d_positions(h, w, self.hidden, device=device, dtype=dtype)

    def forward(self, z, t, cond, return_features=False):
        cond = condition_vector(cond)
        if t.ndim > 2:
            t = t.flatten(1)[:, :1]
        elif t.ndim == 1:
            t = t[:, None]
        b, c, h, w = z.shape
        th, tw = latent_patch_grid(h, w, patch_size=self.latent_patch_size)
        n = th * tw
        if n > self.max_tokens:
            raise ValueError(f"latent token count {n} exceeds max_tokens={self.max_tokens}")
        toks, th, tw = patchify_latents(z, patch_size=self.latent_patch_size)
        t_feat = flow_time_features(
            t, mode=self.flow_time_embed, dim=self.flow_time_embed_dim,
            dtype=cond.dtype)
        ctx = self.cond(torch.cat([cond, t_feat], dim=1))[:, None, :]
        x = self.in_proj(toks)
        x = x + self.image_pos(th, tw, x.device, x.dtype) + ctx
        cond_ctx = ctx[:, 0, :]
        for block in self.blocks:
            if should_checkpoint_blocks(self):
                x = activation_checkpoint(block, x, cond_ctx)
            else:
                x = block(x, cond_ctx)
        features = self.norm(x)
        v = self.out_proj(features)
        velocity = unpatchify_latents(
            v, c, h, w, patch_size=self.latent_patch_size)
        if return_features:
            return velocity, {"image_tokens": features}
        return velocity


class CrossDiTBlock(nn.Module):
    """Image-token block with prompt cross-attention and adaLN-Zero gating."""

    def __init__(self, hidden=96, heads=4, mlp="gelu"):
        super().__init__()
        mlp = str(mlp)
        if mlp not in DIT_MLPS:
            raise ValueError(f"unknown DiT MLP {mlp!r}")
        self.dit_mlp = mlp
        self.self_norm = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.cross_norm = nn.LayerNorm(hidden)
        self.ctx_norm = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = make_dit_feed_forward(hidden, mlp=mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 6))
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 3))
        self.uses_zero_residual_gating = True
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x, ctx, cond_ctx, ctx_mask=None):
        (self_shift, self_scale, cross_shift, cross_scale,
         ff_shift, ff_scale) = self.ada(cond_ctx).chunk(6, dim=-1)
        self_gate, cross_gate, ff_gate = self.gate(cond_ctx).chunk(3, dim=-1)
        q = self._modulate(self.self_norm(x), self_shift, self_scale)
        x = x + self_gate[:, None, :] * self.self_attn(q, q, q, need_weights=False)[0]
        cross_q = self._modulate(self.cross_norm(x), cross_shift, cross_scale)
        ctx_norm = self.ctx_norm(ctx)
        x = x + cross_gate[:, None, :] * self.cross_attn(
            cross_q, ctx_norm, ctx_norm, key_padding_mask=ctx_mask, need_weights=False)[0]
        ff = self._modulate(self.ff_norm(x), ff_shift, ff_scale)
        return x + ff_gate[:, None, :] * self.ff(ff)


class LatentCrossDiTFlowNet(nn.Module):
    """Latent DiT that lets image tokens attend to prompt/fact condition tokens."""

    uses_cond_tokens = True
    uses_adaptive_modulation = True
    uses_residual_gating = True

    def __init__(self, latent_ch=16, hidden=96, depth=3, heads=4, cond_dim=None,
                 max_tokens=256, head_width_mult=1, pos_embed="learned",
                 checkpoint_blocks=False, mlp="gelu", latent_patch_size=1,
                 time_embed="scalar", time_embed_dim=1):
        super().__init__()
        latent_patch_size = int(latent_patch_size)
        if latent_patch_size <= 0:
            raise ValueError("latent_patch_size must be positive")
        mlp = str(mlp)
        if mlp not in DIT_MLPS:
            raise ValueError(f"unknown DiT MLP {mlp!r}")
        pos_embed = str(pos_embed)
        if pos_embed not in DIT_POS_EMBEDS:
            raise ValueError(f"unknown DiT positional embedding {pos_embed!r}")
        if pos_embed == "rope2d":
            raise ValueError("rope2d positional embedding is only supported by MM-DiT")
        cond_dim = int(cond_dim or hidden)
        time_embed = str(time_embed or "scalar")
        if time_embed not in FLOW_TIME_EMBEDS:
            raise ValueError(f"unknown flow time embedding {time_embed!r}")
        self.latent_ch = int(latent_ch)
        self.latent_patch_size = latent_patch_size
        self.token_dim = self.latent_ch * self.latent_patch_size * self.latent_patch_size
        self.hidden = int(hidden)
        self.hidden_feature_dim = int(hidden)
        self.max_tokens = int(max_tokens)
        self.head_width_mult = int(head_width_mult)
        self.dit_pos_embed = pos_embed
        self.dit_mlp = mlp
        self.flow_time_embed = time_embed
        self.flow_time_embed_dim = int(time_embed_dim if time_embed != "scalar" else 1)
        self.uses_zero_residual_gating = True
        self.uses_2d_pos_embed = pos_embed == "sincos2d"
        self.uses_rope2d_pos_embed = False
        self.uses_swiglu_mlp = mlp == "swiglu"
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.uses_activation_checkpointing = self.checkpoint_blocks
        self.in_proj = nn.Linear(self.token_dim, hidden)
        self.pos = (
            nn.Parameter(torch.zeros(1, max_tokens, hidden))
            if pos_embed == "learned" else None
        )
        self.cond = nn.Sequential(
            nn.Linear(cond_dim + self.flow_time_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.ctx_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([
            CrossDiTBlock(hidden, heads, mlp=mlp) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = make_velocity_head(hidden, self.token_dim, self.head_width_mult)

    def image_pos(self, h, w, device, dtype):
        if self.dit_pos_embed == "learned":
            return self.pos[:, :int(h) * int(w)].to(device=device, dtype=dtype)
        return sinusoidal_2d_positions(h, w, self.hidden, device=device, dtype=dtype)

    def _context(self, cond):
        if isinstance(cond, dict):
            tokens = cond["tokens"]
            mask = cond.get("mask")
        else:
            tokens = cond[:, None, :]
            mask = None
        return self.ctx_proj(tokens), mask

    def forward(self, z, t, cond, return_features=False):
        cond_vec = condition_vector(cond)
        if t.ndim > 2:
            t = t.flatten(1)[:, :1]
        elif t.ndim == 1:
            t = t[:, None]
        b, c, h, w = z.shape
        th, tw = latent_patch_grid(h, w, patch_size=self.latent_patch_size)
        n = th * tw
        if n > self.max_tokens:
            raise ValueError(f"latent token count {n} exceeds max_tokens={self.max_tokens}")
        toks, th, tw = patchify_latents(z, patch_size=self.latent_patch_size)
        t_feat = flow_time_features(
            t, mode=self.flow_time_embed, dim=self.flow_time_embed_dim,
            dtype=cond_vec.dtype)
        global_ctx = self.cond(torch.cat([cond_vec, t_feat], dim=1))[:, None, :]
        ctx, ctx_mask = self._context(cond)
        x = self.in_proj(toks)
        x = x + self.image_pos(th, tw, x.device, x.dtype) + global_ctx
        cond_ctx = global_ctx[:, 0, :]
        for block in self.blocks:
            if should_checkpoint_blocks(self):
                def block_forward(x_arg, ctx_arg, cond_arg, block=block):
                    return block(x_arg, ctx_arg, cond_arg, ctx_mask=ctx_mask)
                x = activation_checkpoint(block_forward, x, ctx, cond_ctx)
            else:
                x = block(x, ctx, cond_ctx, ctx_mask=ctx_mask)
        features = self.norm(x)
        v = self.out_proj(features)
        velocity = unpatchify_latents(
            v, c, h, w, patch_size=self.latent_patch_size)
        if return_features:
            return velocity, {"image_tokens": features}
        return velocity


class HeadRMSNorm(nn.Module):
    """RMS-normalize query/key vectors per attention head."""

    def __init__(self, head_dim, eps=1.0e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(head_dim)))

    def forward(self, x):
        dtype = x.dtype
        x_float = x.float()
        denom = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_float * denom).to(dtype) * self.weight.to(dtype=dtype)


class MMDiTBlock(nn.Module):
    """Tiny dual-stream block with separate image/text projections and joint attention."""

    def __init__(self, hidden=96, heads=4, qk_norm=False, attn_impl="manual",
                 image_rope=False, mlp="gelu"):
        super().__init__()
        mlp = str(mlp)
        if mlp not in DIT_MLPS:
            raise ValueError(f"unknown DiT MLP {mlp!r}")
        if hidden % heads:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        attn_impl = str(attn_impl)
        if attn_impl not in MMDIT_ATTN_IMPLS:
            raise ValueError(f"unknown MM-DiT attention implementation {attn_impl!r}")
        self.heads = int(heads)
        self.head_dim = hidden // heads
        if image_rope and self.head_dim % 2:
            raise ValueError("rope2d requires an even MM-DiT head dimension")
        self.uses_qk_norm = bool(qk_norm)
        self.attn_impl = attn_impl
        self.dit_mlp = mlp
        self.uses_image_rope2d = bool(image_rope)
        self.uses_swiglu_mlp = mlp == "swiglu"
        self.uses_zero_residual_gating = True
        self.scale = self.head_dim ** -0.5
        self.img_norm = nn.LayerNorm(hidden)
        self.ctx_norm = nn.LayerNorm(hidden)
        self.img_qkv = nn.Linear(hidden, hidden * 3)
        self.ctx_qkv = nn.Linear(hidden, hidden * 3)
        self.img_q_norm = HeadRMSNorm(self.head_dim) if self.uses_qk_norm else nn.Identity()
        self.img_k_norm = HeadRMSNorm(self.head_dim) if self.uses_qk_norm else nn.Identity()
        self.ctx_q_norm = HeadRMSNorm(self.head_dim) if self.uses_qk_norm else nn.Identity()
        self.ctx_k_norm = HeadRMSNorm(self.head_dim) if self.uses_qk_norm else nn.Identity()
        self.img_out = nn.Linear(hidden, hidden)
        self.ctx_out = nn.Linear(hidden, hidden)
        self.img_ff_norm = nn.LayerNorm(hidden)
        self.ctx_ff_norm = nn.LayerNorm(hidden)
        self.img_ff = make_dit_feed_forward(hidden, mlp=mlp)
        self.ctx_ff = make_dit_feed_forward(hidden, mlp=mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 8))
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 4))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _qkv(self, proj, x):
        b, n, h = x.shape
        q, k, v = proj(x).view(b, n, 3, self.heads, self.head_dim).unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    @staticmethod
    def _sdpa_available():
        return hasattr(F, "scaled_dot_product_attention")

    @staticmethod
    def _key_mask(ctx_mask, b, n_img, device):
        if ctx_mask is None:
            return None
        img_mask = torch.zeros((b, n_img), dtype=torch.bool, device=device)
        return torch.cat([img_mask, ctx_mask], dim=1)

    def _manual_attention(self, q, k, v, key_mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_mask is not None:
            attn = attn.masked_fill(key_mask[:, None, None, :], torch.finfo(attn.dtype).min)
        return torch.softmax(attn, dim=-1).matmul(v)

    def _sdpa_attention(self, q, k, v, key_mask=None):
        attn_mask = None
        if key_mask is not None:
            attn_mask = torch.zeros((q.shape[0], 1, 1, key_mask.shape[1]),
                                    dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(key_mask[:, None, None, :],
                                              torch.finfo(q.dtype).min)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    def _linear_attention(self, q, k, v, key_mask=None, eps=1.0e-6):
        q_feat = F.elu(q.float()) + 1.0
        k_feat = F.elu(k.float()) + 1.0
        v_float = v.float()
        if key_mask is not None:
            keep = (~key_mask).to(dtype=k_feat.dtype, device=k_feat.device)[:, None, :, None]
            k_feat = k_feat * keep
            v_float = v_float * keep
        k_sum = k_feat.sum(dim=2)
        kv = torch.einsum("bhnd,bhne->bhde", k_feat, v_float)
        denom = torch.einsum("bhnd,bhd->bhn", q_feat, k_sum).clamp_min(float(eps))
        out = torch.einsum("bhnd,bhde,bhn->bhne", q_feat, kv, denom.reciprocal())
        return out.to(dtype=v.dtype)

    def _joint_attention(self, q, k, v, key_mask=None):
        if self.attn_impl == "manual":
            return self._manual_attention(q, k, v, key_mask=key_mask)
        if self.attn_impl == "linear":
            return self._linear_attention(q, k, v, key_mask=key_mask)
        if self._sdpa_available():
            return self._sdpa_attention(q, k, v, key_mask=key_mask)
        if self.attn_impl == "sdpa":
            raise RuntimeError("scaled_dot_product_attention is unavailable in this torch build")
        return self._manual_attention(q, k, v, key_mask=key_mask)

    def forward(self, img, ctx, cond_ctx, ctx_mask=None, image_rope=None):
        b, n_img, h = img.shape
        n_ctx = ctx.shape[1]
        (img_attn_shift, img_attn_scale, ctx_attn_shift, ctx_attn_scale,
         img_ff_shift, img_ff_scale, ctx_ff_shift, ctx_ff_scale) = self.ada(cond_ctx).chunk(
             8, dim=-1)
        img_attn_gate, ctx_attn_gate, img_ff_gate, ctx_ff_gate = self.gate(cond_ctx).chunk(
            4, dim=-1)
        img_attn = self._modulate(self.img_norm(img), img_attn_shift, img_attn_scale)
        ctx_attn = self._modulate(self.ctx_norm(ctx), ctx_attn_shift, ctx_attn_scale)
        qi, ki, vi = self._qkv(self.img_qkv, img_attn)
        qc, kc, vc = self._qkv(self.ctx_qkv, ctx_attn)
        qi, ki = self.img_q_norm(qi), self.img_k_norm(ki)
        qc, kc = self.ctx_q_norm(qc), self.ctx_k_norm(kc)
        if image_rope is not None:
            rope_cos, rope_sin = image_rope
            qi = apply_rotary_factors(qi, rope_cos, rope_sin)
            ki = apply_rotary_factors(ki, rope_cos, rope_sin)
        q = torch.cat([qi, qc], dim=2)
        k = torch.cat([ki, kc], dim=2)
        v = torch.cat([vi, vc], dim=2)
        key_mask = self._key_mask(ctx_mask, b, n_img, q.device)
        mixed = self._joint_attention(q, k, v, key_mask=key_mask)
        mixed = mixed.transpose(1, 2).reshape(b, n_img + n_ctx, h)
        img_delta, ctx_delta = mixed[:, :n_img], mixed[:, n_img:]
        img = img + img_attn_gate[:, None, :] * self.img_out(img_delta)
        ctx = ctx + ctx_attn_gate[:, None, :] * self.ctx_out(ctx_delta)
        img_ff = self._modulate(self.img_ff_norm(img), img_ff_shift, img_ff_scale)
        ctx_ff = self._modulate(self.ctx_ff_norm(ctx), ctx_ff_shift, ctx_ff_scale)
        img = img + img_ff_gate[:, None, :] * self.img_ff(img_ff)
        ctx = ctx + ctx_ff_gate[:, None, :] * self.ctx_ff(ctx_ff)
        if ctx_mask is not None:
            ctx = ctx.masked_fill(ctx_mask[:, :, None], 0.0)
        return img, ctx


class LatentMMDiTFlowNet(nn.Module):
    """Toy MM-DiT latent flow with bidirectional image/condition token mixing."""

    uses_cond_tokens = True
    uses_adaptive_modulation = True
    uses_residual_gating = True

    def __init__(self, latent_ch=16, hidden=96, depth=3, heads=4, cond_dim=None,
                 max_tokens=256, head_width_mult=1, qk_norm=False,
                 attn_impl="manual", pos_embed="learned", checkpoint_blocks=False,
                 mlp="gelu", latent_patch_size=1,
                 time_embed="scalar", time_embed_dim=1):
        super().__init__()
        latent_patch_size = int(latent_patch_size)
        if latent_patch_size <= 0:
            raise ValueError("latent_patch_size must be positive")
        mlp = str(mlp)
        if mlp not in DIT_MLPS:
            raise ValueError(f"unknown DiT MLP {mlp!r}")
        attn_impl = str(attn_impl)
        if attn_impl not in MMDIT_ATTN_IMPLS:
            raise ValueError(f"unknown MM-DiT attention implementation {attn_impl!r}")
        pos_embed = str(pos_embed)
        if pos_embed not in DIT_POS_EMBEDS:
            raise ValueError(f"unknown DiT positional embedding {pos_embed!r}")
        cond_dim = int(cond_dim or hidden)
        time_embed = str(time_embed or "scalar")
        if time_embed not in FLOW_TIME_EMBEDS:
            raise ValueError(f"unknown flow time embedding {time_embed!r}")
        self.latent_ch = int(latent_ch)
        self.latent_patch_size = latent_patch_size
        self.token_dim = self.latent_ch * self.latent_patch_size * self.latent_patch_size
        self.hidden = int(hidden)
        self.hidden_feature_dim = int(hidden)
        self.max_tokens = int(max_tokens)
        self.head_width_mult = int(head_width_mult)
        self.uses_qk_norm = bool(qk_norm)
        self.attn_impl = attn_impl
        self.dit_mlp = mlp
        self.heads = int(heads)
        self.head_dim = self.hidden // self.heads
        self.dit_pos_embed = pos_embed
        self.flow_time_embed = time_embed
        self.flow_time_embed_dim = int(time_embed_dim if time_embed != "scalar" else 1)
        self.uses_zero_residual_gating = True
        self.uses_2d_pos_embed = pos_embed in ("sincos2d", "rope2d")
        self.uses_rope2d_pos_embed = pos_embed == "rope2d"
        self.uses_swiglu_mlp = mlp == "swiglu"
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.uses_activation_checkpointing = self.checkpoint_blocks
        self.in_proj = nn.Linear(self.token_dim, hidden)
        self.pos = (
            nn.Parameter(torch.zeros(1, max_tokens, hidden))
            if pos_embed == "learned" else None
        )
        self.time = nn.Sequential(nn.Linear(cond_dim + self.flow_time_embed_dim, hidden),
                                  nn.GELU(),
                                  nn.Linear(hidden, hidden))
        self.ctx_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                hidden, heads, qk_norm=self.uses_qk_norm, attn_impl=self.attn_impl,
                image_rope=self.uses_rope2d_pos_embed, mlp=mlp
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = make_velocity_head(hidden, self.token_dim, self.head_width_mult)

    def image_pos(self, h, w, device, dtype):
        if self.dit_pos_embed == "learned":
            return self.pos[:, :int(h) * int(w)].to(device=device, dtype=dtype)
        if self.dit_pos_embed == "sincos2d":
            return sinusoidal_2d_positions(h, w, self.hidden, device=device, dtype=dtype)
        return None

    def image_rope(self, h, w, device, dtype):
        if not self.uses_rope2d_pos_embed:
            return None
        return rotary_2d_factors(h, w, self.head_dim, device=device, dtype=dtype)

    def _context(self, cond):
        if isinstance(cond, dict):
            tokens = cond["tokens"]
            mask = cond.get("mask")
        else:
            tokens = cond[:, None, :]
            mask = None
        return self.ctx_proj(tokens), mask

    def forward(self, z, t, cond, return_features=False):
        cond_vec = condition_vector(cond)
        if t.ndim > 2:
            t = t.flatten(1)[:, :1]
        elif t.ndim == 1:
            t = t[:, None]
        b, c, h, w = z.shape
        th, tw = latent_patch_grid(h, w, patch_size=self.latent_patch_size)
        n = th * tw
        if n > self.max_tokens:
            raise ValueError(f"latent token count {n} exceeds max_tokens={self.max_tokens}")
        t_feat = flow_time_features(
            t, mode=self.flow_time_embed, dim=self.flow_time_embed_dim,
            dtype=cond_vec.dtype)
        cond_ctx = self.time(torch.cat([cond_vec, t_feat], dim=1))
        toks, th, tw = patchify_latents(z, patch_size=self.latent_patch_size)
        img = self.in_proj(toks)
        image_pos = self.image_pos(th, tw, img.device, img.dtype)
        if image_pos is not None:
            img = img + image_pos
        img = img + cond_ctx[:, None, :]
        ctx, ctx_mask = self._context(cond)
        image_rope = self.image_rope(th, tw, img.device, img.dtype)
        for block in self.blocks:
            if should_checkpoint_blocks(self):
                def block_forward(img_arg, ctx_arg, cond_arg, block=block):
                    return block(
                        img_arg, ctx_arg, cond_arg, ctx_mask=ctx_mask,
                        image_rope=image_rope)
                img, ctx = activation_checkpoint(block_forward, img, ctx, cond_ctx)
            else:
                img, ctx = block(img, ctx, cond_ctx, ctx_mask=ctx_mask,
                                 image_rope=image_rope)
        features = self.norm(img)
        v = self.out_proj(features)
        velocity = unpatchify_latents(
            v, c, h, w, patch_size=self.latent_patch_size)
        if return_features:
            return velocity, {"image_tokens": features}
        return velocity


def make_flow(flow_arch="conv", latent_ch=16, hidden=64, dit_depth=3, dit_heads=4,
              cond_dim=None, dit_head_width_mult=1, latent_max_tokens=256,
              dit_qk_norm=False, dit_attn_impl="auto", dit_pos_embed="learned",
              dit_mlp="gelu", latent_patch_size=1, flow_checkpoint_blocks=False,
              flow_time_embed="scalar", flow_time_embed_dim=0):
    latent_patch_size = int(latent_patch_size)
    if latent_patch_size <= 0:
        raise ValueError("latent_patch_size must be positive")
    if flow_arch == "conv":
        if latent_patch_size != 1:
            raise ValueError("latent_patch_size is only supported by DiT/CrossDiT/MM-DiT flows")
        return LatentFlowNet(latent_ch=latent_ch, hidden=hidden, cond_dim=cond_dim)
    flow_time_embed = str(flow_time_embed or "scalar")
    if flow_time_embed not in FLOW_TIME_EMBEDS:
        raise ValueError(f"unknown flow time embedding {flow_time_embed!r}")
    flow_time_embed_dim = int(flow_time_embed_dim or 0)
    if flow_time_embed_dim < 0:
        raise ValueError("flow_time_embed_dim must be non-negative")
    if flow_time_embed == "scalar":
        flow_time_embed_dim = 1
    elif flow_time_embed_dim <= 1:
        flow_time_embed_dim = int(hidden)
    dit_mlp = str(dit_mlp)
    if dit_mlp not in DIT_MLPS:
        raise ValueError(f"unknown DiT MLP {dit_mlp!r}")
    dit_pos_embed = str(dit_pos_embed)
    if dit_pos_embed not in DIT_POS_EMBEDS:
        raise ValueError(f"unknown DiT positional embedding {dit_pos_embed!r}")
    if dit_pos_embed == "rope2d" and flow_arch != "mmdit":
        raise ValueError("rope2d positional embedding is only supported by MM-DiT")
    dit_head_width_mult = int(dit_head_width_mult)
    if dit_head_width_mult <= 0:
        raise ValueError("dit_head_width_mult must be positive")
    if flow_arch == "dit":
        heads = max(1, min(dit_heads, hidden // 16))
        while hidden % heads:
            heads -= 1
        return LatentDiTFlowNet(latent_ch=latent_ch, hidden=hidden, depth=dit_depth, heads=heads,
                                cond_dim=cond_dim,
                                head_width_mult=dit_head_width_mult,
                                max_tokens=latent_max_tokens,
                                pos_embed=dit_pos_embed,
                                checkpoint_blocks=flow_checkpoint_blocks,
                                latent_patch_size=latent_patch_size,
                                time_embed=flow_time_embed,
                                time_embed_dim=flow_time_embed_dim,
                                mlp=dit_mlp)
    if flow_arch == "crossdit":
        heads = max(1, min(dit_heads, hidden // 16))
        while hidden % heads:
            heads -= 1
        return LatentCrossDiTFlowNet(latent_ch=latent_ch, hidden=hidden, depth=dit_depth,
                                     heads=heads, cond_dim=cond_dim,
                                     head_width_mult=dit_head_width_mult,
                                     max_tokens=latent_max_tokens,
                                     pos_embed=dit_pos_embed,
                                     checkpoint_blocks=flow_checkpoint_blocks,
                                     mlp=dit_mlp,
                                     latent_patch_size=latent_patch_size,
                                     time_embed=flow_time_embed,
                                     time_embed_dim=flow_time_embed_dim)
    if flow_arch == "mmdit":
        heads = max(1, min(dit_heads, hidden // 16))
        while hidden % heads:
            heads -= 1
        return LatentMMDiTFlowNet(latent_ch=latent_ch, hidden=hidden, depth=dit_depth,
                                  heads=heads, cond_dim=cond_dim,
                                  head_width_mult=dit_head_width_mult,
                                  max_tokens=latent_max_tokens,
                                  qk_norm=dit_qk_norm,
                                  attn_impl=dit_attn_impl,
                                  pos_embed=dit_pos_embed,
                                  mlp=dit_mlp,
                                  latent_patch_size=latent_patch_size,
                                  checkpoint_blocks=flow_checkpoint_blocks,
                                  time_embed=flow_time_embed,
                                  time_embed_dim=flow_time_embed_dim)
    raise ValueError(f"unknown latent flow architecture {flow_arch!r}")


def attach_text_aligner(flow, text_aligner):
    # Keep the aligner available for eval without registering it as part of flow.state_dict().
    if hasattr(flow, "_modules"):
        flow._modules.pop("text_aligner", None)
    flow.__dict__["text_aligner"] = text_aligner
    return flow


def attach_image_feature_aligner(flow, image_feature_aligner):
    # Keep the aligner available for eval without registering it as part of flow.state_dict().
    if hasattr(flow, "_modules"):
        flow._modules.pop("image_feature_aligner", None)
    flow.__dict__["image_feature_aligner"] = image_feature_aligner
    return flow


def attach_image_quality_scorer(flow, image_quality_scorer):
    # Keep the scorer available for eval/guidance without registering it in flow.state_dict().
    if hasattr(flow, "_modules"):
        flow._modules.pop("image_quality_scorer", None)
    flow.__dict__["image_quality_scorer"] = image_quality_scorer
    return flow


def attach_flow_repa_aligner(flow, flow_repa_aligner):
    # Keep the auxiliary REPA head out of flow.state_dict(); checkpoint it explicitly.
    if hasattr(flow, "_modules"):
        flow._modules.pop("flow_repa_aligner", None)
    flow.__dict__["flow_repa_aligner"] = flow_repa_aligner
    return flow


def attach_flow_self_repa_aligner(flow, flow_self_repa_aligner):
    # Keep the auxiliary self-REPA head out of flow.state_dict(); checkpoint it explicitly.
    if hasattr(flow, "_modules"):
        flow._modules.pop("flow_self_repa_aligner", None)
    flow.__dict__["flow_self_repa_aligner"] = flow_self_repa_aligner
    return flow


def attach_flow_boundary_mode(flow, mode="none"):
    mode = str(mode or "none")
    if mode not in FLOW_BOUNDARY_MODES:
        raise ValueError(f"unknown flow boundary mode {mode!r}")
    flow.flow_boundary_mode = mode
    return flow


def get_flow_boundary_mode(flow):
    return str(getattr(flow, "flow_boundary_mode", "none") or "none")


def _time_like_latent(t, z):
    tt = t
    if not torch.is_tensor(tt):
        tt = torch.tensor(float(tt), device=z.device, dtype=z.dtype)
    tt = tt.to(device=z.device, dtype=z.dtype)
    if tt.ndim == 0:
        tt = tt.reshape(1)
    if tt.ndim == 1:
        tt = tt[:, None]
    while tt.ndim < z.ndim:
        tt = tt.unsqueeze(-1)
    return tt.clamp(0.0, 1.0)


def apply_flow_boundary_velocity(raw_velocity, z, t, mode="none"):
    """Parameterize RF velocity so known endpoint behavior is exact.

    Latents are normalized before flow training in production runs.  In that space,
    the RF endpoint field has useful approximate boundary values: v(x, 0) = -x for
    pure Gaussian noise and v(x, 1) = x for clean zero-mean data.
    """
    mode = str(mode or "none")
    if mode == "none":
        return raw_velocity
    if mode not in FLOW_BOUNDARY_MODES:
        raise ValueError(f"unknown flow boundary mode {mode!r}")
    tt = _time_like_latent(t, z)
    if mode == "right-linear":
        return tt * z + (1.0 - tt) * raw_velocity
    if mode == "double-linear":
        return (2.0 * tt - 1.0) * z + (4.0 * tt * (1.0 - tt)) * raw_velocity
    if mode == "double-cosine":
        return -torch.cos(math.pi * tt) * z + torch.sin(math.pi * tt) * raw_velocity
    raise ValueError(f"unknown flow boundary mode {mode!r}")


def flow_velocity(flow, z, t, cond, return_features=False):
    out = flow(z, t, cond, return_features=return_features)
    if return_features:
        raw_velocity, features = out
        return (
            apply_flow_boundary_velocity(raw_velocity, z, t, mode=get_flow_boundary_mode(flow)),
            features,
        )
    return apply_flow_boundary_velocity(out, z, t, mode=get_flow_boundary_mode(flow))


def _parse_number_list(s, cast=float):
    if isinstance(s, (tuple, list)):
        return tuple(cast(x) for x in s)
    return tuple(cast(x.strip()) for x in str(s).split(",") if x.strip())


def _parse_string_list(s):
    if isinstance(s, (tuple, list)):
        return tuple(str(x).strip() for x in s if str(x).strip())
    return tuple(x.strip() for x in str(s).split(",") if x.strip())


def _parse_interval(s):
    vals = _parse_number_list(s, float)
    if len(vals) != 2:
        raise ValueError(f"expected interval start,end, got {s!r}")
    return validate_guidance_interval(vals)


def validate_guidance_interval(interval, name="guidance interval"):
    start, end = (float(interval[0]), float(interval[1]))
    if start < 0.0 or end > 1.0 or start > end:
        raise ValueError(f"{name} must satisfy 0 <= start <= end <= 1")
    return (start, end)


def interval_active(t, interval):
    start, end = validate_guidance_interval(interval)
    t = float(t)
    return start <= t <= end


def validate_cfg_schedule(schedule):
    schedule = str(schedule or "constant")
    if schedule not in CFG_SCHEDULES:
        raise ValueError(f"unknown cfg_schedule {schedule!r}")
    return schedule


def cfg_schedule_strength(t, interval=DEFAULT_GUIDANCE_INTERVAL, schedule="constant"):
    """Return the extra-CFG multiplier over flow time; 0 means condition-only."""
    start, end = validate_guidance_interval(interval, name="cfg_interval")
    schedule = validate_cfg_schedule(schedule)
    t = float(t)
    if t < start or t > end:
        return 0.0
    if end <= start:
        progress = 1.0
    else:
        progress = (t - start) / max(1.0e-8, end - start)
    progress = max(0.0, min(1.0, float(progress)))
    if schedule == "constant":
        return 1.0
    if schedule == "linear":
        return progress
    if schedule == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * progress)
    if schedule == "triangular":
        return math.sin(math.pi * progress)
    raise ValueError(f"unknown cfg_schedule {schedule!r}")


def effective_cfg_scale_at(t, cfg_scale=1.0, cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                           cfg_schedule="constant"):
    strength = cfg_schedule_strength(
        t, interval=cfg_interval, schedule=cfg_schedule)
    return 1.0 + (float(cfg_scale) - 1.0) * float(strength), float(strength)


def format_interval(interval):
    start, end = validate_guidance_interval(interval)
    return f"{start:g},{end:g}"


def _seeded_randn(shape, device, seed):
    dev = torch.device(device)
    gen_device = "cpu" if dev.type == "mps" else dev.type
    try:
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(int(seed))
        return torch.randn(shape, generator=gen, device=device)
    except (RuntimeError, TypeError):
        torch.manual_seed(int(seed))
        return torch.randn(shape, device=device)


def apply_flow_time_shift(t, shift=1.0):
    """Shift data-time t through the equivalent rectified-flow noise schedule.

    The repo uses t=0 as pure noise and t=1 as clean data.  Image RF schedulers commonly
    shift the noise-time schedule toward higher-noise regions for harder/high-resolution
    generation.  Applying the shift to sigma=1-t and converting back keeps endpoints fixed.
    """
    shift = float(shift)
    if shift <= 0.0:
        raise ValueError("time_shift must be positive")
    if shift == 1.0:
        return t
    return t / (shift + (1.0 - shift) * t).clamp_min(1.0e-12)


def invert_flow_time_shift(t, shift=1.0, eps=1.0e-12):
    """Invert apply_flow_time_shift for binning adaptive sampler statistics."""
    shift = float(shift)
    if shift <= 0.0:
        raise ValueError("time_shift must be positive")
    if shift == 1.0:
        return t.clamp(0.0, 1.0)
    denom = (1.0 - t * (1.0 - shift)).clamp_min(float(eps))
    return (t * shift / denom).clamp(0.0, 1.0)


def mode_flow_time_base(u, mode_scale=DEFAULT_TIME_MODE_SCALE):
    """SD3-style mode-biased data-time density for rectified-flow training.

    The transform keeps endpoints fixed and bends uniform samples toward perceptually
    useful middle noise scales.  The monotonicity bound prevents folded densities.
    """
    scale = float(mode_scale)
    if scale < 0.0 or scale >= MAX_TIME_MODE_SCALE:
        raise ValueError(f"time_mode_scale must be in [0, {MAX_TIME_MODE_SCALE})")
    if scale == 0.0:
        return u
    base = u + scale * (torch.cos(0.5 * math.pi * u).square() - 1.0 + u)
    return base.clamp(0.0, 1.0)


class AdaptiveTimestepSampler:
    """Loss-adaptive rectified-flow timestep sampler.

    The sampler keeps an EMA of per-bin velocity loss and samples higher-loss bins
    more often, with uniform mixing and probability floors so coverage stays broad.
    Bins live in pre-shift time; the returned times still pass through the configured
    flow time shift.
    """

    def __init__(self, bins=32, momentum=0.95, uniform_mix=0.05,
                 min_prob=0.001, loss_power=1.0):
        bins = int(bins)
        if bins <= 1:
            raise ValueError("adaptive timestep bins must be > 1")
        momentum = float(momentum)
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError("adaptive timestep momentum must be in [0, 1)")
        uniform_mix = float(uniform_mix)
        min_prob = float(min_prob)
        loss_power = float(loss_power)
        if uniform_mix < 0.0 or uniform_mix > 1.0:
            raise ValueError("adaptive timestep uniform mix must be in [0, 1]")
        if min_prob < 0.0 or min_prob >= 1.0:
            raise ValueError("adaptive timestep min probability must be in [0, 1)")
        if loss_power <= 0.0:
            raise ValueError("adaptive timestep loss power must be positive")
        self.bins = bins
        self.momentum = momentum
        self.uniform_mix = uniform_mix
        self.min_prob = min_prob
        self.loss_power = loss_power
        self.loss_ema = torch.ones(bins, dtype=torch.float64)
        self.counts = torch.zeros(bins, dtype=torch.long)
        self.samples = torch.zeros(bins, dtype=torch.long)
        self.updates = 0

    def probabilities(self):
        scores = self.loss_ema.clamp_min(1.0e-12).pow(float(self.loss_power))
        probs = scores / scores.sum().clamp_min(1.0e-12)
        if self.uniform_mix > 0.0:
            probs = (
                (1.0 - self.uniform_mix) * probs
                + self.uniform_mix / float(self.bins)
            )
        if self.min_prob > 0.0:
            probs = probs.clamp_min(float(self.min_prob))
            probs = probs / probs.sum().clamp_min(1.0e-12)
        return probs

    def sample(self, batch, device=DEV, time_shift=1.0):
        probs = self.probabilities()
        idx = torch.multinomial(probs, int(batch), replacement=True)
        self.samples += torch.bincount(idx, minlength=self.bins).to(self.samples.dtype)
        u = (
            idx.to(device=device, dtype=torch.float32) + torch.rand(
                int(batch), device=device)
        ) / float(self.bins)
        while u.ndim < 4:
            u = u.unsqueeze(-1)
        return apply_flow_time_shift(u, shift=time_shift)

    @torch.no_grad()
    def update(self, t, losses, time_shift=1.0):
        if t is None or losses is None:
            return
        flat_t = invert_flow_time_shift(
            t.detach().float().flatten(), shift=time_shift).cpu()
        flat_loss = losses.detach().float().flatten().cpu()
        keep = torch.isfinite(flat_t) & torch.isfinite(flat_loss)
        if not bool(keep.any()):
            return
        flat_t = flat_t[keep].clamp(0.0, 1.0 - 1.0e-7)
        flat_loss = flat_loss[keep].clamp_min(0.0).double()
        idx = torch.clamp((flat_t * float(self.bins)).long(), 0, self.bins - 1)
        for bin_id in torch.unique(idx):
            mask = idx == bin_id
            mean_loss = flat_loss[mask].mean()
            j = int(bin_id.item())
            if int(self.counts[j]) == 0:
                self.loss_ema[j] = mean_loss
            else:
                self.loss_ema[j] = (
                    float(self.momentum) * self.loss_ema[j]
                    + (1.0 - float(self.momentum)) * mean_loss
                )
            self.counts[j] += int(mask.sum().item())
        self.updates += 1

    def report(self, prefix="time_adaptive"):
        probs = self.probabilities()
        return {
            f"{prefix}_enabled": True,
            f"{prefix}_bins": int(self.bins),
            f"{prefix}_momentum": float(self.momentum),
            f"{prefix}_uniform_mix": float(self.uniform_mix),
            f"{prefix}_min_prob": float(self.min_prob),
            f"{prefix}_loss_power": float(self.loss_power),
            f"{prefix}_updates": int(self.updates),
            f"{prefix}_observed_bins": int((self.counts > 0).sum().item()),
            f"{prefix}_sampled_bins": int((self.samples > 0).sum().item()),
            f"{prefix}_prob_min": float(probs.min().item()),
            f"{prefix}_prob_max": float(probs.max().item()),
            f"{prefix}_prob_entropy": float(
                -(probs * probs.clamp_min(1.0e-12).log()).sum().item()
            ),
            f"{prefix}_loss_ema_min": float(self.loss_ema.min().item()),
            f"{prefix}_loss_ema_max": float(self.loss_ema.max().item()),
        }


def latent_effective_dim(latent_shape):
    dim = 1
    for value in tuple(latent_shape):
        dim *= int(value)
    return int(dim)


def resolve_time_shift(time_shift=1.0, mode="manual", latent_shape=None,
                       ref_dim=1024.0, dim_power=0.5):
    mode = str(mode)
    if mode not in TIME_SHIFT_MODES:
        raise ValueError(f"unknown time shift mode {mode!r}")
    base = float(time_shift)
    if base <= 0.0:
        raise ValueError("time_shift must be positive")
    ref_dim = float(ref_dim)
    if ref_dim <= 0.0:
        raise ValueError("time_shift_ref_dim must be positive")
    dim_power = float(dim_power)
    effective_dim = latent_effective_dim(latent_shape) if latent_shape is not None else 0
    effective_mode = mode
    if mode == "auto":
        effective_mode = "dim" if effective_dim > ref_dim else "manual"
    if effective_mode == "manual":
        return {
            "time_shift": base,
            "time_shift_requested": base,
            "time_shift_mode": mode,
            "time_shift_effective_mode": effective_mode,
            "time_shift_auto_policy": "dim_when_latent_dim_exceeds_ref",
            "time_shift_ref_dim": float(ref_dim),
            "time_shift_dim_power": float(dim_power),
            "latent_effective_dim": int(effective_dim),
            "time_shift_dim_scale": 1.0,
        }
    if latent_shape is None:
        raise ValueError("dimension-aware time shift requires a latent shape")
    scale = (float(effective_dim) / ref_dim) ** dim_power
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("dimension-aware time shift produced a non-positive scale")
    return {
        "time_shift": float(base * scale),
        "time_shift_requested": base,
        "time_shift_mode": mode,
        "time_shift_effective_mode": effective_mode,
        "time_shift_auto_policy": "dim_when_latent_dim_exceeds_ref",
        "time_shift_ref_dim": float(ref_dim),
        "time_shift_dim_power": float(dim_power),
        "latent_effective_dim": int(effective_dim),
        "time_shift_dim_scale": float(scale),
    }


def karras_flow_time_base(u, rho=KARRAS_SCHEDULE_RHO,
                          sigma_min=KARRAS_SCHEDULE_SIGMA_MIN,
                          sigma_max=KARRAS_SCHEDULE_SIGMA_MAX):
    """Karras-style data-time placement for rectified-flow sampling.

    EDM schedules are expressed in decreasing noise sigma. This repo samples in
    data-time t, where t=0 is pure noise and t=1 is clean data, so we map the
    monotonically decreasing sigma schedule back to monotonically increasing t.
    """
    rho = float(rho)
    sigma_min = float(sigma_min)
    sigma_max = float(sigma_max)
    if rho <= 0.0:
        raise ValueError("karras rho must be positive")
    if sigma_min < 0.0 or sigma_max <= sigma_min:
        raise ValueError("karras sigma range must satisfy 0 <= min < max")
    inv_rho = 1.0 / rho
    sigma_min_r = sigma_min ** inv_rho
    sigma_max_r = sigma_max ** inv_rho
    sigmas = (
        sigma_max_r + u * (sigma_min_r - sigma_max_r)
    ).clamp_min(0.0).pow(rho)
    base = (sigma_max - sigmas) / (sigma_max - sigma_min)
    return base.clamp(0.0, 1.0)


def flow_time_schedule(steps, device=DEV, shift=1.0, schedule="linear"):
    steps = max(1, int(steps))
    schedule = str(schedule)
    u = torch.linspace(0.0, 1.0, steps + 1, device=device)
    if schedule == "linear":
        base = u
    elif schedule == "quadratic":
        base = u.square()
    elif schedule == "sqrt":
        base = torch.sqrt(u)
    elif schedule == "cosine":
        base = 0.5 - 0.5 * torch.cos(math.pi * u)
    elif schedule == "karras":
        base = karras_flow_time_base(u)
    else:
        raise ValueError(f"unknown sample schedule {schedule!r}")
    return apply_flow_time_shift(base, shift=shift)


def sample_flow_times(batch, device=DEV, mode="uniform", logit_mean=0.0, logit_std=1.0,
                      mode_scale=DEFAULT_TIME_MODE_SCALE, time_shift=1.0,
                      adaptive_sampler=None):
    """Sample rectified-flow interpolation times.

    Uniform is the original rectified-flow baseline.  Logit-normal follows the SD3-style
    direction of spending more training mass on perceptually useful intermediate noise scales.
    """
    if mode == "uniform":
        t = torch.rand(batch, 1, 1, 1, device=device)
    elif mode == "logit-normal":
        eps = torch.randn(batch, 1, 1, 1, device=device) * float(logit_std) + float(logit_mean)
        t = torch.sigmoid(eps)
    elif mode == "mode":
        u = torch.rand(batch, 1, 1, 1, device=device)
        t = mode_flow_time_base(u, mode_scale=mode_scale)
    elif mode == "adaptive":
        if adaptive_sampler is not None:
            return adaptive_sampler.sample(batch, device=device, time_shift=time_shift)
        eps = torch.randn(batch, 1, 1, 1, device=device) * float(logit_std) + float(logit_mean)
        t = torch.sigmoid(eps)
    else:
        raise ValueError(f"unknown time sampling mode {mode!r}")
    return apply_flow_time_shift(t, shift=time_shift)


def time_curriculum_switch_step(flow_steps, frac=0.0):
    frac = float(frac)
    if frac < 0.0 or frac > 1.0:
        raise ValueError("time_curriculum_frac must be in [0, 1]")
    if frac <= 0.0:
        return 0
    steps = int(flow_steps)
    if steps <= 0:
        return 0
    return min(steps, max(1, int(math.ceil(float(steps) * frac))))


def active_time_sampling_mode(time_sampling, flow_step=0, flow_steps=0,
                              time_curriculum_frac=0.0):
    mode = str(time_sampling)
    if mode not in TIME_SAMPLINGS:
        raise ValueError(f"unknown time sampling mode {mode!r}")
    switch_step = time_curriculum_switch_step(flow_steps, time_curriculum_frac)
    if switch_step <= 0:
        return mode
    return mode if int(flow_step) < switch_step else "uniform"


def flow_loss_time_weights(t, mode="none", gamma=5.0, normalize=True, eps=1.0e-5):
    """Per-example weighting for rectified-flow velocity loss.

    The repo's data-time convention is t=0 noise and t=1 data, so the signal-to-noise
    ratio is (t / (1 - t))^2.  The *-v modes use the velocity-target form of Min-SNR
    weighting, dividing by SNR+1, and optionally normalize the batch mean back to one.
    """
    mode = str(mode)
    if mode not in FLOW_LOSS_WEIGHTS:
        raise ValueError(f"unknown flow loss weighting mode {mode!r}")
    flat_t = t.flatten(1)[:, 0].float().clamp(float(eps), 1.0 - float(eps))
    if mode == "none":
        return torch.ones_like(flat_t)
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("flow_loss_weight_gamma must be positive")
    snr = flat_t.square() / (1.0 - flat_t).square().clamp_min(float(eps))
    if mode == "min-snr-v":
        base = torch.minimum(snr, torch.full_like(snr, gamma))
    elif mode == "soft-min-snr-v":
        base = (snr * gamma) / (snr + gamma).clamp_min(float(eps))
    weight = base / (snr + 1.0).clamp_min(float(eps))
    if normalize:
        weight = weight / weight.mean().detach().clamp_min(float(eps))
    return weight


def latent_stats_enabled(stats):
    return bool(stats and stats.get("mode", "none") != "none"
                and stats.get("mean") is not None and stats.get("std") is not None)


def latent_stats_to_device(stats, device):
    if not latent_stats_enabled(stats):
        return {"mode": "none", "n": int((stats or {}).get("n", 0))}
    return {
        "mode": stats.get("mode", "channel"),
        "n": int(stats.get("n", 0)),
        "mean": stats["mean"].to(device=device),
        "std": stats["std"].to(device=device),
    }


def attach_latent_stats(flow, stats):
    flow.latent_stats = latent_stats_to_device(stats, next(flow.parameters()).device)
    return flow


def flow_latent_stats(flow):
    return getattr(flow, "latent_stats", {"mode": "none", "n": 0})


def normalize_latent(z, stats):
    if not latent_stats_enabled(stats):
        return z
    mean = stats["mean"].to(device=z.device, dtype=z.dtype)
    std = stats["std"].to(device=z.device, dtype=z.dtype)
    return (z - mean) / std


def denormalize_latent(z, stats):
    if not latent_stats_enabled(stats):
        return z
    mean = stats["mean"].to(device=z.device, dtype=z.dtype)
    std = stats["std"].to(device=z.device, dtype=z.dtype)
    return z * std + mean


@torch.no_grad()
def estimate_latent_stats_records(ae, records, n=512, batch=64, seed=123, size=32, device=DEV,
                                  mode="none", eps=1.0e-6, crop_mode="center",
                                  hflip_prob=0.0, weights=None):
    mode = str(mode)
    if mode == "none":
        return {"mode": "none", "n": 0}
    if mode not in ("global", "channel"):
        raise ValueError(f"unknown latent normalization mode {mode!r}")
    n = max(1, int(n))
    rng = np.random.default_rng(seed)
    ae.eval()
    count = 0
    seen = 0
    sums = sums2 = None
    while seen < n:
        b = min(batch, n - seen)
        x, _captions = sample_image_text_batch(
            records, rng, batch=b, size=size, device=device,
            crop_mode=crop_mode, hflip_prob=hflip_prob, weights=weights)
        z = ae.encode(x).detach().float().cpu().double()
        if mode == "global":
            cur_sum = z.sum()
            cur_sum2 = z.pow(2).sum()
            cur_count = z.numel()
        else:
            cur_sum = z.sum(dim=(0, 2, 3))
            cur_sum2 = z.pow(2).sum(dim=(0, 2, 3))
            cur_count = z.shape[0] * z.shape[2] * z.shape[3]
        if sums is None:
            sums = torch.zeros_like(cur_sum)
            sums2 = torch.zeros_like(cur_sum2)
        sums += cur_sum
        sums2 += cur_sum2
        count += int(cur_count)
        seen += b
    mean = sums / max(1, count)
    var = (sums2 / max(1, count) - mean.pow(2)).clamp_min(float(eps) ** 2)
    std = var.sqrt()
    if mode == "global":
        mean = mean.view(1, 1, 1, 1)
        std = std.view(1, 1, 1, 1)
    else:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    return {
        "mode": mode,
        "n": int(seen),
        "mean": mean.float(),
        "std": std.float(),
    }


@torch.no_grad()
def estimate_latent_stats_record_buckets(ae, records, size_buckets, bucket_records=None,
                                         n=512, batch=64, seed=123, device=DEV,
                                         mode="none", eps=1.0e-6, crop_mode="center",
                                         hflip_prob=0.0, record_weights=None,
                                         bucket_probs=None):
    mode = str(mode)
    if mode == "none":
        return {"mode": "none", "n": 0}
    if mode not in ("global", "channel"):
        raise ValueError(f"unknown latent normalization mode {mode!r}")
    buckets = tuple(size_buckets)
    if not buckets:
        raise ValueError("latent stats buckets require at least one image size")
    n = max(1, int(n))
    rng = np.random.default_rng(seed)
    ae.eval()
    count = 0
    seen = 0
    sums = sums2 = None
    while seen < n:
        b = min(batch, n - seen)
        _bucket, x, _captions = sample_bucketed_image_text_batch(
            records, rng, batch=b, size_buckets=buckets, bucket_records=bucket_records,
            device=device, crop_mode=crop_mode, hflip_prob=hflip_prob,
            record_weights=record_weights, bucket_probs=bucket_probs)
        z = ae.encode(x).detach().float().cpu().double()
        if mode == "global":
            cur_sum = z.sum()
            cur_sum2 = z.pow(2).sum()
            cur_count = z.numel()
        else:
            cur_sum = z.sum(dim=(0, 2, 3))
            cur_sum2 = z.pow(2).sum(dim=(0, 2, 3))
            cur_count = z.shape[0] * z.shape[2] * z.shape[3]
        if sums is None:
            sums = torch.zeros_like(cur_sum)
            sums2 = torch.zeros_like(cur_sum2)
        sums += cur_sum
        sums2 += cur_sum2
        count += int(cur_count)
        seen += b
    mean = sums / max(1, count)
    var = (sums2 / max(1, count) - mean.pow(2)).clamp_min(float(eps) ** 2)
    std = var.sqrt()
    if mode == "global":
        mean = mean.view(1, 1, 1, 1)
        std = std.view(1, 1, 1, 1)
    else:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    return {
        "mode": mode,
        "n": int(seen),
        "mean": mean.float(),
        "std": std.float(),
    }


@torch.no_grad()
def estimate_latent_stats_tensor(latents, n=512, seed=123, mode="none", eps=1.0e-6):
    mode = str(mode)
    if mode == "none":
        return {"mode": "none", "n": 0}
    if mode not in ("global", "channel"):
        raise ValueError(f"unknown latent normalization mode {mode!r}")
    latents = latents.detach().float().cpu()
    if latents.ndim != 4:
        raise ValueError(f"expected cached BCHW latents, got shape {tuple(latents.shape)}")
    total = int(latents.shape[0])
    if total <= 0:
        raise ValueError("latent cache is empty")
    n = min(max(1, int(n)), total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n, replace=total < n)
    z = latents[torch.tensor(idx, dtype=torch.long)].double()
    if mode == "global":
        mean = z.mean().view(1, 1, 1, 1)
        std = z.std(unbiased=False).clamp_min(float(eps)).view(1, 1, 1, 1)
    else:
        mean = z.mean(dim=(0, 2, 3)).view(1, -1, 1, 1)
        std = z.std(dim=(0, 2, 3), unbiased=False).clamp_min(float(eps)).view(1, -1, 1, 1)
    return {
        "mode": mode,
        "n": int(n),
        "mean": mean.float(),
        "std": std.float(),
    }


def maybe_embedding_sha256(values):
    if values is None:
        return ""
    arr = np.asarray(values, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def autoencoder_cache_identity(ae):
    if isinstance(ae, HFAutoencoderKL):
        return {
            "arch": "hf-vae",
            "model_id": ae.model_id,
            "subfolder": ae.subfolder,
            "scaling_factor": float(ae.scaling_factor),
            "downsample": int(ae.downsample),
            "latent_ch": int(ae.latent_ch),
        }
    h = hashlib.sha256()
    for name, tensor in sorted(ae.state_dict().items()):
        h.update(str(name).encode("utf-8"))
        h.update(str(tuple(int(x) for x in tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {
        "arch": ae.__class__.__name__,
        "state_sha256": h.hexdigest(),
        "downsample": int(getattr(ae, "downsample", 0) or 0),
        "latent_ch": int(getattr(ae, "latent_ch", 0) or 0),
    }


def image_latent_cache_fingerprint(ae, rows, size=32, caption_max_len=64,
                                   cond_source="tokens", include_image_embeddings=False,
                                   include_image_embedding_sequences=False,
                                   include_image_geometry=False,
                                   include_quality_targets=False, quality_stats=None,
                                   crop_mode="center", hflip_prob=0.0, seed=0,
                                   cache_dtype="fp32", shard_size=1024,
                                   size_buckets=(), weighted=False, precision="fp32"):
    record_rows = []
    for rec in rows:
        try:
            stat = os.stat(rec.path)
            file_size = int(stat.st_size)
            file_mtime_ns = int(stat.st_mtime_ns)
        except OSError:
            file_size = -1
            file_mtime_ns = -1
        record_rows.append({
            "path": os.path.abspath(rec.path),
            "caption": str(rec.caption),
            "split": str(rec.split),
            "source": str(rec.source),
            "aesthetic": None if rec.aesthetic is None else float(rec.aesthetic),
            "width": int(rec.width),
            "height": int(rec.height),
            "file_size": file_size,
            "file_mtime_ns": file_mtime_ns,
            "text_embedding_sha256": maybe_embedding_sha256(rec.text_embedding),
            "text_embedding_sequence_sha256": maybe_embedding_sha256(
                rec.text_embedding_sequence),
            "image_embedding_sha256": maybe_embedding_sha256(rec.image_embedding),
            "image_embedding_sequence_sha256": maybe_embedding_sha256(
                rec.image_embedding_sequence),
        })
    payload = {
        "version": 2,
        "ae": autoencoder_cache_identity(ae),
        "size": image_size_pair_value(size),
        "size_buckets": [image_size_pair_value(bucket) for bucket in size_buckets],
        "caption_max_len": int(caption_max_len),
        "cond_source": str(cond_source),
        "include_image_embeddings": bool(include_image_embeddings),
        "include_image_embedding_sequences": bool(include_image_embedding_sequences),
        "include_image_geometry": bool(include_image_geometry),
        "image_geometry_feature_dim": int(GEOMETRY_FEATURE_DIM),
        "include_quality_targets": bool(include_quality_targets),
        "quality_stats": (
            {
                "n": int(quality_stats.get("n", 0)),
                "missing": int(quality_stats.get("missing", 0)),
                "min": float(quality_stats.get("min", 0.0)),
                "mean": float(quality_stats.get("mean", 0.0)),
                "max": float(quality_stats.get("max", 0.0)),
                "has_range": bool(quality_stats.get("has_range", False)),
            }
            if quality_stats is not None else None
        ),
        "crop_mode": str(crop_mode),
        "hflip_prob": float(hflip_prob),
        "seed": int(seed),
        "cache_dtype": str(cache_dtype),
        "shard_size": int(shard_size),
        "weighted": bool(weighted),
        "precision": str(precision),
        "records": record_rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _latent_cache_meta_path(cache_dir):
    return os.path.join(os.path.abspath(cache_dir), "meta.json")


def load_disk_image_latent_cache(cache_dir, expected_fingerprint=None):
    cache_dir = os.path.abspath(cache_dir)
    meta_path = _latent_cache_meta_path(cache_dir)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if expected_fingerprint and meta.get("fingerprint") != expected_fingerprint:
        return None
    fmt = str(meta.get("format", ""))
    if fmt == "image_latent_cache_bucketed_v1":
        buckets = []
        total_bytes = int(os.path.getsize(meta_path))
        total_shards = 0
        for bucket_meta in meta.get("buckets", []):
            key = str(bucket_meta["key"])
            subcache = load_disk_image_latent_cache(os.path.join(cache_dir, key))
            if subcache is None:
                return None
            total_bytes += int(subcache.get("bytes", 0))
            total_shards += int(subcache.get("shard_count", len(subcache.get("shards", []))))
            buckets.append({
                "key": key,
                "size": list(bucket_meta.get("size", [])),
                "cache": subcache,
                "records": int(subcache.get("records", bucket_meta.get("records", 0))),
                "weight_sum": float(subcache.get(
                    "weight_sum", bucket_meta.get("weight_sum", 0.0))),
            })
        cache = {
            "backend": str(meta.get("backend", "bucketed_disk")),
            "cache_dir": cache_dir,
            "meta_path": meta_path,
            "buckets": buckets,
            "bucket_sampling_mode": str(meta.get("bucket_sampling_mode", "uniform")),
            "bucket_sampling_probs": dict(meta.get("bucket_sampling_probs", {})),
            "records": int(meta.get("records", sum(b["records"] for b in buckets))),
            "bucket_count": int(meta.get("bucket_count", len(buckets))),
            "shard_count": int(meta.get("shard_count", total_shards)),
            "bytes": int(meta.get("bytes", total_bytes)) + int(os.path.getsize(meta_path)),
            "cond_source": str(meta.get("cond_source", "")),
            "has_image_embeddings": bool(meta.get("has_image_embeddings", False)),
            "has_image_embedding_sequences": bool(meta.get(
                "has_image_embedding_sequences", False)),
            "has_image_geometry": bool(meta.get("has_image_geometry", False)),
            "has_quality_targets": bool(meta.get("has_quality_targets", False)),
            "latent_dtype": str(meta.get("latent_dtype", "")),
            "weighted": bool(meta.get("weighted", False)),
            "weight_sum": float(meta.get("weight_sum", 0.0)),
            "fingerprint": meta.get("fingerprint", ""),
            "cache_reused": True,
        }
        return cache
    if fmt != "image_latent_cache_v1":
        return None
    shards = []
    for shard_meta in meta.get("shards", []):
        filename = str(shard_meta["file"])
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            return None
        shards.append({
            "path": path,
            "file": filename,
            "count": int(shard_meta.get("count", 0)),
            "bytes": int(shard_meta.get("bytes", os.path.getsize(path))),
            "weight_sum": float(shard_meta.get("weight_sum", shard_meta.get("count", 0))),
            "latent_dtype": str(shard_meta.get("latent_dtype", meta.get("latent_dtype", ""))),
            "text_embedding_shape": list(shard_meta.get("text_embedding_shape", [])),
            "text_pooled_embedding_shape": list(shard_meta.get(
                "text_pooled_embedding_shape", [])),
            "image_embedding_sequence_shape": list(shard_meta.get(
                "image_embedding_sequence_shape", [])),
        })
    return {
        "backend": "disk",
        "cache_dir": cache_dir,
        "meta_path": meta_path,
        "records": int(meta.get("records", sum(s["count"] for s in shards))),
        "latent_shape": tuple(int(x) for x in meta.get("latent_shape", [])),
        "cond_source": str(meta.get("cond_source", "")),
        "has_image_embeddings": bool(meta.get("has_image_embeddings", False)),
        "has_image_embedding_sequences": bool(meta.get(
            "has_image_embedding_sequences", False)),
        "has_image_geometry": bool(meta.get("has_image_geometry", False)),
        "has_quality_targets": bool(meta.get("has_quality_targets", False)),
        "latent_dtype": str(meta.get("latent_dtype", "")),
        "shard_size": int(meta.get("shard_size", 0)),
        "shards": shards,
        "shard_count": int(len(shards)),
        "bytes": int(meta.get("bytes", sum(s["bytes"] for s in shards))) + int(
            os.path.getsize(meta_path)),
        "weighted": bool(meta.get("weighted", False)),
        "weight_sum": float(meta.get("weight_sum", 0.0)),
        "fingerprint": meta.get("fingerprint", ""),
        "cache_reused": True,
    }


@torch.no_grad()
def build_image_latent_cache(ae, records, prompt_vocab, caption_max_len=64, max_records=0,
                             batch=64, seed=0, size=32, device=DEV, precision="fp32",
                             cond_source="tokens", cache_dir="", shard_size=1024,
                             include_image_embeddings=False, crop_mode="center",
                             include_image_embedding_sequences=False,
                             include_image_geometry=False,
                             include_quality_targets=False, quality_stats=None,
                             hflip_prob=0.0, size_buckets=(), bucket_records=None,
                             record_weights=None, cache_dtype="fp32"):
    rows = list(records)
    if not rows:
        raise ValueError("cannot build latent cache from empty records")
    cache_dtype = str(cache_dtype)
    latent_cache_torch_dtype(cache_dtype)
    rng = np.random.default_rng(seed)
    if max_records and int(max_records) < len(rows):
        probs = normalized_sampling_weights(weights_for_records(rows, record_weights), len(rows))
        chosen_idx = rng.choice(len(rows), size=int(max_records), replace=False, p=probs)
        rows = [rows[int(i)] for i in chosen_idx]
    buckets = normalize_image_size_buckets(size_buckets)
    cache_fingerprint = image_latent_cache_fingerprint(
        ae, rows, size=size, caption_max_len=caption_max_len,
        cond_source=cond_source, include_image_embeddings=include_image_embeddings,
        include_image_embedding_sequences=include_image_embedding_sequences,
        include_image_geometry=include_image_geometry,
        include_quality_targets=include_quality_targets,
        quality_stats=quality_stats,
        crop_mode=crop_mode, hflip_prob=hflip_prob, seed=seed,
        cache_dtype=cache_dtype, shard_size=shard_size, size_buckets=buckets,
        weighted=record_weights is not None, precision=precision)
    if cache_dir:
        existing = load_disk_image_latent_cache(
            cache_dir, expected_fingerprint=cache_fingerprint)
        if existing is not None:
            return existing
    if len(buckets) > 1:
        limited_bucket_records, _missing_dims = bucket_records_by_aspect(rows, buckets)
        subcaches = []
        total_bytes = 0
        total_records = 0
        total_shards = 0
        total_weight = 0.0
        weighted = bool(record_weights is not None)
        for bucket in buckets:
            bucket_key = image_size_key(bucket)
            bucket_rows = list(limited_bucket_records.get(bucket) or rows)
            bucket_dir = os.path.join(cache_dir, bucket_key) if cache_dir else ""
            subcache = build_image_latent_cache(
                ae, bucket_rows, prompt_vocab, caption_max_len=caption_max_len,
                max_records=0, batch=batch, seed=seed + len(subcaches) * 997,
                size=bucket, device=device, precision=precision, cond_source=cond_source,
                cache_dir=bucket_dir, shard_size=shard_size,
                include_image_embeddings=include_image_embeddings,
                include_image_embedding_sequences=include_image_embedding_sequences,
                include_image_geometry=include_image_geometry,
                include_quality_targets=include_quality_targets,
                quality_stats=quality_stats,
                crop_mode=crop_mode, hflip_prob=hflip_prob,
                record_weights=record_weights, cache_dtype=cache_dtype)
            total_bytes += int(subcache.get("bytes", 0))
            total_records += int(subcache.get("records", 0))
            total_shards += int(subcache.get("shard_count", len(subcache.get("shards", []))))
            total_weight += float(subcache.get("weight_sum", subcache.get("records", 0)))
            subcaches.append({
                "key": bucket_key,
                "size": image_size_pair_value(bucket),
                "cache": subcache,
                "records": int(subcache.get("records", 0)),
                "weight_sum": float(subcache.get("weight_sum", subcache.get("records", 0))),
            })
        bucket_weights = np.asarray([
            float(b.get("weight_sum", b.get("records", 0))) for b in subcaches
        ], dtype=np.float64)
        bucket_probs = (
            normalized_sampling_weights(bucket_weights, len(subcaches))
            if weighted else np.ones(len(subcaches), dtype=np.float64) / float(len(subcaches))
        )
        backend = "bucketed_disk" if cache_dir else "bucketed_memory"
        cache = {
            "backend": backend,
            "cache_dir": os.path.abspath(cache_dir) if cache_dir else "",
            "buckets": subcaches,
            "bucket_sampling_mode": "weighted" if weighted else "uniform",
            "bucket_sampling_probs": {
                subcaches[i]["key"]: float(bucket_probs[i]) for i in range(len(subcaches))
            },
            "records": int(total_records),
            "bucket_count": int(len(subcaches)),
            "shard_count": int(total_shards),
            "bytes": int(total_bytes),
            "cond_source": cond_source,
            "has_image_embeddings": bool(include_image_embeddings),
            "has_image_embedding_sequences": bool(include_image_embedding_sequences),
            "has_image_geometry": bool(include_image_geometry),
            "has_quality_targets": bool(include_quality_targets),
            "latent_dtype": cache_dtype,
            "weighted": bool(weighted),
            "weight_sum": float(total_weight),
            "fingerprint": cache_fingerprint,
            "cache_reused": False,
        }
        if cache_dir:
            os.makedirs(cache["cache_dir"], exist_ok=True)
            meta_path = os.path.join(cache["cache_dir"], "meta.json")
            with open(meta_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({
                    "format": "image_latent_cache_bucketed_v1",
                    "backend": backend,
                    "buckets": [
                        {
                            "key": b["key"],
                            "size": b["size"],
                            "records": b["records"],
                            "weight_sum": b["weight_sum"],
                        }
                        for b in subcaches
                    ],
                    "bucket_sampling_mode": cache["bucket_sampling_mode"],
                    "bucket_sampling_probs": cache["bucket_sampling_probs"],
                    "records": int(total_records),
                    "shard_count": int(total_shards),
                    "bytes": int(total_bytes),
                    "cond_source": cond_source,
                    "has_image_embeddings": bool(include_image_embeddings),
                    "has_image_embedding_sequences": bool(
                        include_image_embedding_sequences),
                    "has_image_geometry": bool(include_image_geometry),
                    "has_quality_targets": bool(include_quality_targets),
                    "latent_dtype": cache_dtype,
                    "weighted": bool(weighted),
                    "weight_sum": float(total_weight),
                    "fingerprint": cache_fingerprint,
                }, f, indent=1, sort_keys=True)
            os.replace(meta_path + ".tmp", meta_path)
            cache["meta_path"] = meta_path
            cache["bytes"] += int(os.path.getsize(meta_path))
        return cache
    if cond_source == "embedding":
        infer_text_embedding_dim(rows)
    if include_image_embeddings:
        infer_image_embedding_dim(rows)
    if include_image_embedding_sequences and not records_have_image_embedding_sequences(rows):
        raise ValueError("image embedding sequence cache requested but no records have sequences")
    cache_quality_stats = quality_stats or quality_score_stats(rows)
    if cache_dir:
        return build_disk_image_latent_cache(
            ae, rows, prompt_vocab, caption_max_len=caption_max_len,
            batch=batch, size=size, device=device, precision=precision,
            cond_source=cond_source, cache_dir=cache_dir, shard_size=shard_size,
            include_image_embeddings=include_image_embeddings,
            include_image_embedding_sequences=include_image_embedding_sequences,
            include_image_geometry=include_image_geometry,
            include_quality_targets=include_quality_targets,
            quality_stats=cache_quality_stats,
            seed=seed,
            crop_mode=crop_mode, hflip_prob=hflip_prob, record_weights=record_weights,
            cache_dtype=cache_dtype, cache_fingerprint=cache_fingerprint)
    latents, captions, embeddings, pooled_embeddings, pooled_masks = [], [], [], [], []
    image_embeddings, image_embedding_sequences, image_geometry_features = [], [], []
    quality_targets, quality_masks = [], []
    sequence_max_len = image_embedding_sequence_max_len(rows)
    weights = weights_for_records(rows, record_weights)
    ae.eval()
    crop_rng = np.random.default_rng(seed + 17)
    for start in range(0, len(rows), max(1, int(batch))):
        chunk = rows[start:start + max(1, int(batch))]
        imgs, transform_metadata = [], []
        for rec in chunk:
            loaded = load_image_tensor(
                rec.path, size=size, device=device, crop_mode=crop_mode,
                hflip=(hflip_prob > 0.0 and float(crop_rng.random()) < float(hflip_prob)),
                rng=crop_rng, return_metadata=include_image_geometry)
            if include_image_geometry:
                img, meta = loaded
                transform_metadata.append(meta)
            else:
                img = loaded
            imgs.append(img)
        x = torch.stack(imgs, dim=0)
        with amp_autocast(device, precision):
            z = ae.encode(x)
        latents.append(cache_latent_tensor(z, cache_dtype))
        captions.extend(rec.caption for rec in chunk)
        if cond_source == "embedding":
            embeddings.append(record_text_embedding_tensor(chunk, device="cpu"))
            pooled, pooled_mask = record_text_pooled_embedding_tensor(chunk, device="cpu")
            if pooled is not None and embeddings[-1].ndim == 3:
                pooled_embeddings.append(pooled)
                pooled_masks.append(pooled_mask)
        if include_image_embeddings:
            image_embeddings.append(record_image_embedding_tensor(chunk, device="cpu"))
        if include_image_embedding_sequences:
            image_embedding_sequences.append(record_image_embedding_sequence_tensor(
                chunk, device="cpu", max_len=sequence_max_len))
        if include_image_geometry:
            image_geometry_features.append(image_geometry_feature_tensor(
                chunk, target_size=size, device="cpu",
                transform_metadata=transform_metadata))
        if include_quality_targets:
            q_target, q_mask = quality_target_tensor(
                chunk, cache_quality_stats, device="cpu")
            quality_targets.append(q_target.cpu())
            quality_masks.append(q_mask.cpu())
    cache = {
        "backend": "memory",
        "latents": torch.cat(latents, dim=0).contiguous(),
        "captions": captions,
        "caption_ids": (
            caption_ids(captions, prompt_vocab, max_len=caption_max_len, device="cpu")
            if cond_source == "tokens" else None
        ),
        "text_embeddings": cat_text_embedding_tensors(embeddings),
        "text_pooled_embeddings": (
            torch.cat(pooled_embeddings, dim=0).contiguous()
            if pooled_embeddings else None
        ),
        "text_pooled_masks": (
            torch.cat(pooled_masks, dim=0).contiguous() if pooled_masks else None
        ),
        "image_embeddings": (
            torch.cat(image_embeddings, dim=0).contiguous() if image_embeddings else None
        ),
        "image_embedding_sequences": (
            torch.cat(image_embedding_sequences, dim=0).contiguous()
            if image_embedding_sequences else None
        ),
        "image_geometry_features": (
            torch.cat(image_geometry_features, dim=0).contiguous()
            if image_geometry_features else None
        ),
        "quality_targets": (
            torch.cat(quality_targets, dim=0).contiguous() if quality_targets else None
        ),
        "quality_masks": (
            torch.cat(quality_masks, dim=0).contiguous() if quality_masks else None
        ),
        "cond_source": cond_source,
        "has_image_embeddings": bool(image_embeddings),
        "has_image_embedding_sequences": bool(image_embedding_sequences),
        "has_image_geometry": bool(image_geometry_features),
        "has_quality_targets": bool(quality_targets),
        "latent_dtype": cache_dtype,
        "records": len(rows),
        "latent_shape": tuple(int(x) for x in latents[0].shape[1:]),
        "bytes": int(sum(z.numel() * z.element_size() for z in latents)),
        "weights": (
            torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        ),
        "weighted": bool(weights is not None),
        "weight_sum": float(np.sum(weights)) if weights is not None else float(len(rows)),
        "fingerprint": cache_fingerprint,
        "cache_reused": False,
    }
    if cache["text_embeddings"] is not None:
        cache["text_embedding_shape"] = [
            int(x) for x in cache["text_embeddings"].shape
        ]
    if cache["text_pooled_embeddings"] is not None:
        cache["text_pooled_embedding_shape"] = [
            int(x) for x in cache["text_pooled_embeddings"].shape
        ]
    if cache["weights"] is not None:
        cache["bytes"] += int(cache["weights"].numel() * cache["weights"].element_size())
    if cache["text_embeddings"] is not None:
        cache["bytes"] += int(
            cache["text_embeddings"].numel() * cache["text_embeddings"].element_size()
        )
    if cache["text_pooled_embeddings"] is not None:
        cache["bytes"] += int(
            cache["text_pooled_embeddings"].numel()
            * cache["text_pooled_embeddings"].element_size()
        )
    if cache["text_pooled_masks"] is not None:
        cache["bytes"] += int(
            cache["text_pooled_masks"].numel() * cache["text_pooled_masks"].element_size()
        )
    if cache["image_embeddings"] is not None:
        cache["bytes"] += int(
            cache["image_embeddings"].numel() * cache["image_embeddings"].element_size()
        )
    if cache["image_embedding_sequences"] is not None:
        cache["bytes"] += int(
            cache["image_embedding_sequences"].numel()
            * cache["image_embedding_sequences"].element_size()
        )
    if cache["image_geometry_features"] is not None:
        cache["bytes"] += int(
            cache["image_geometry_features"].numel()
            * cache["image_geometry_features"].element_size()
        )
    if cache["quality_targets"] is not None:
        cache["bytes"] += int(
            cache["quality_targets"].numel() * cache["quality_targets"].element_size()
        )
    if cache["quality_masks"] is not None:
        cache["bytes"] += int(
            cache["quality_masks"].numel() * cache["quality_masks"].element_size()
        )
    return cache


@torch.no_grad()
def build_disk_image_latent_cache(ae, rows, prompt_vocab, caption_max_len=64, batch=64,
                                  size=32, device=DEV, precision="fp32",
                                  cond_source="tokens", cache_dir="", shard_size=1024,
                                  include_image_embeddings=False, seed=0,
                                  include_image_embedding_sequences=False,
                                  include_image_geometry=False,
                                  include_quality_targets=False, quality_stats=None,
                                  crop_mode="center", hflip_prob=0.0,
                                  record_weights=None, cache_dtype="fp32",
                                  cache_fingerprint=""):
    if not cache_dir:
        raise ValueError("cache_dir is required for disk latent cache")
    cache_dtype = str(cache_dtype)
    latent_cache_torch_dtype(cache_dtype)
    shard_size = int(shard_size)
    if shard_size <= 0:
        raise ValueError("flow cache shard size must be positive")
    rows = list(rows)
    if not rows:
        raise ValueError("cannot build latent cache from empty records")
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    shards = []
    total_bytes = 0
    total_weight = 0.0
    latent_shape = None
    sequence_max_len = image_embedding_sequence_max_len(rows)
    cache_quality_stats = quality_stats or quality_score_stats(rows)
    ae.eval()
    crop_rng = np.random.default_rng(seed + 17)
    for shard_i, start in enumerate(range(0, len(rows), shard_size)):
        chunk_rows = rows[start:start + shard_size]
        latents = []
        transform_metadata = []
        for enc_start in range(0, len(chunk_rows), max(1, int(batch))):
            enc_rows = chunk_rows[enc_start:enc_start + max(1, int(batch))]
            imgs = []
            for rec in enc_rows:
                loaded = load_image_tensor(
                    rec.path, size=size, device=device, crop_mode=crop_mode,
                    hflip=(
                        hflip_prob > 0.0
                        and float(crop_rng.random()) < float(hflip_prob)
                    ),
                    rng=crop_rng, return_metadata=include_image_geometry)
                if include_image_geometry:
                    img, meta = loaded
                    transform_metadata.append(meta)
                else:
                    img = loaded
                imgs.append(img)
            x = torch.stack(imgs, dim=0)
            with amp_autocast(device, precision):
                z = ae.encode(x)
            latents.append(cache_latent_tensor(z, cache_dtype))
        captions = [rec.caption for rec in chunk_rows]
        chunk_weights = weights_for_records(chunk_rows, record_weights)
        chunk_weight_tensor = (
            torch.tensor(chunk_weights, dtype=torch.float32)
            if chunk_weights is not None else None
        )
        chunk_weight_sum = (
            float(np.sum(chunk_weights)) if chunk_weights is not None
            else float(len(chunk_rows))
        )
        q_target, q_mask = (
            quality_target_tensor(
                chunk_rows, cache_quality_stats, device="cpu")
            if include_quality_targets else (None, None)
        )
        total_weight += chunk_weight_sum
        shard = {
            "latents": torch.cat(latents, dim=0).contiguous(),
            "captions": captions,
            "caption_ids": (
                caption_ids(captions, prompt_vocab, max_len=caption_max_len, device="cpu")
                if cond_source == "tokens" else None
            ),
            "text_embeddings": (
                record_text_embedding_tensor(chunk_rows, device="cpu")
                if cond_source == "embedding" else None
            ),
            "text_pooled_embeddings": None,
            "text_pooled_masks": None,
            "image_embeddings": (
                record_image_embedding_tensor(chunk_rows, device="cpu")
                if include_image_embeddings else None
            ),
            "image_embedding_sequences": (
                record_image_embedding_sequence_tensor(
                    chunk_rows, device="cpu", max_len=sequence_max_len)
                if include_image_embedding_sequences else None
            ),
            "image_geometry_features": (
                image_geometry_feature_tensor(
                    chunk_rows, target_size=size, device="cpu",
                    transform_metadata=transform_metadata)
                if include_image_geometry else None
            ),
            "quality_targets": q_target.cpu().contiguous() if q_target is not None else None,
            "quality_masks": q_mask.cpu().contiguous() if q_mask is not None else None,
            "cond_source": cond_source,
            "has_image_embeddings": bool(include_image_embeddings),
            "has_image_embedding_sequences": bool(include_image_embedding_sequences),
            "has_image_geometry": bool(include_image_geometry),
            "has_quality_targets": bool(include_quality_targets),
            "latent_dtype": cache_dtype,
            "weights": chunk_weight_tensor,
            "weighted": bool(chunk_weight_tensor is not None),
            "weight_sum": float(chunk_weight_sum),
            "start": int(start),
            "count": int(len(chunk_rows)),
        }
        if cond_source == "embedding" and shard["text_embeddings"] is not None:
            pooled, pooled_mask = record_text_pooled_embedding_tensor(
                chunk_rows, device="cpu")
            if pooled is not None and shard["text_embeddings"].ndim == 3:
                shard["text_pooled_embeddings"] = pooled
                shard["text_pooled_masks"] = pooled_mask
        if shard["text_embeddings"] is not None:
            shard["text_embedding_shape"] = [
                int(x) for x in shard["text_embeddings"].shape
            ]
        if shard["text_pooled_embeddings"] is not None:
            shard["text_pooled_embedding_shape"] = [
                int(x) for x in shard["text_pooled_embeddings"].shape
            ]
        if shard["image_embedding_sequences"] is not None:
            shard["image_embedding_sequence_shape"] = [
                int(x) for x in shard["image_embedding_sequences"].shape
            ]
        if latent_shape is None:
            latent_shape = tuple(int(x) for x in shard["latents"].shape[1:])
        name = f"shard_{shard_i:06d}.pt"
        path = os.path.join(cache_dir, name)
        tmp_path = path + ".tmp"
        torch.save(shard, tmp_path)
        os.replace(tmp_path, path)
        file_bytes = int(os.path.getsize(path))
        total_bytes += file_bytes
        shards.append({"path": path, "file": name, "count": int(len(chunk_rows)),
                       "bytes": file_bytes, "weight_sum": float(chunk_weight_sum),
                       "latent_dtype": cache_dtype,
                       "text_embedding_shape": shard.get("text_embedding_shape", []),
                       "text_pooled_embedding_shape": shard.get(
                           "text_pooled_embedding_shape", []),
                       "image_embedding_sequence_shape": shard.get(
                           "image_embedding_sequence_shape", [])})
    meta = {
        "format": "image_latent_cache_v1",
        "backend": "disk",
        "cache_dir": cache_dir,
        "records": int(len(rows)),
        "latent_shape": list(latent_shape or ()),
        "cond_source": cond_source,
        "has_image_embeddings": bool(include_image_embeddings),
        "has_image_embedding_sequences": bool(include_image_embedding_sequences),
        "has_image_geometry": bool(include_image_geometry),
        "has_quality_targets": bool(include_quality_targets),
        "latent_dtype": cache_dtype,
        "shard_size": int(shard_size),
        "shards": [
            {
                "file": s["file"],
                "count": int(s["count"]),
                "bytes": int(s["bytes"]),
                "weight_sum": float(s["weight_sum"]),
                "latent_dtype": str(s.get("latent_dtype", cache_dtype)),
                "text_embedding_shape": list(s.get("text_embedding_shape", [])),
                "text_pooled_embedding_shape": list(s.get(
                    "text_pooled_embedding_shape", [])),
                "image_embedding_sequence_shape": list(s.get(
                    "image_embedding_sequence_shape", [])),
            }
            for s in shards
        ],
        "bytes": int(total_bytes),
        "weighted": bool(record_weights is not None),
        "weight_sum": float(total_weight),
        "fingerprint": str(cache_fingerprint or ""),
    }
    meta_path = os.path.join(cache_dir, "meta.json")
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    os.replace(tmp_meta, meta_path)
    total_bytes += int(os.path.getsize(meta_path))
    return {
        "backend": "disk",
        "cache_dir": cache_dir,
        "meta_path": meta_path,
        "records": int(len(rows)),
        "latent_shape": tuple(int(x) for x in latent_shape),
        "cond_source": cond_source,
        "has_image_embeddings": bool(include_image_embeddings),
        "has_image_embedding_sequences": bool(include_image_embedding_sequences),
        "has_image_geometry": bool(include_image_geometry),
        "has_quality_targets": bool(include_quality_targets),
        "latent_dtype": cache_dtype,
        "shard_size": int(shard_size),
        "shards": shards,
        "bytes": int(total_bytes),
        "weighted": bool(record_weights is not None),
        "weight_sum": float(total_weight),
        "fingerprint": str(cache_fingerprint or ""),
        "cache_reused": False,
    }


def latent_cache_backend(cache):
    return str((cache or {}).get("backend", "memory" if cache is not None else ""))


def latent_cache_dtype_name(cache):
    if cache is None:
        return ""
    dtype = str(cache.get("latent_dtype", "") or "")
    if dtype:
        return dtype
    backend = latent_cache_backend(cache)
    if backend in ("bucketed_memory", "bucketed_disk"):
        buckets = list(cache.get("buckets", []))
        dtypes = {
            latent_cache_dtype_name(bucket.get("cache"))
            for bucket in buckets
            if latent_cache_dtype_name(bucket.get("cache"))
        }
        if len(dtypes) == 1:
            return next(iter(dtypes))
        if dtypes:
            return "mixed"
    if backend == "disk":
        dtypes = {str(s.get("latent_dtype", "") or "") for s in cache.get("shards", [])}
        dtypes.discard("")
        if len(dtypes) == 1:
            return next(iter(dtypes))
        if dtypes:
            return "mixed"
    tensor = cache.get("latents") if isinstance(cache, dict) else None
    if torch.is_tensor(tensor):
        return str(tensor.dtype).replace("torch.", "")
    return ""


def latent_cache_text_embedding_shapes(cache):
    if cache is None:
        return []
    backend = latent_cache_backend(cache)
    if backend in ("bucketed_memory", "bucketed_disk"):
        shapes = []
        for bucket in cache.get("buckets", []):
            shapes.extend(latent_cache_text_embedding_shapes(bucket.get("cache")))
        return shapes
    if backend == "disk":
        return [
            [int(x) for x in shard.get("text_embedding_shape", [])]
            for shard in cache.get("shards", [])
            if shard.get("text_embedding_shape")
        ]
    shape = cache.get("text_embedding_shape")
    if shape:
        return [[int(x) for x in shape]]
    tensor = cache.get("text_embeddings")
    if torch.is_tensor(tensor):
        return [[int(x) for x in tensor.shape]]
    return []


def latent_cache_text_embedding_report(cache):
    shapes = latent_cache_text_embedding_shapes(cache)
    if not shapes:
        return {
            "flow_cache_text_embedding_ndim": 0,
            "flow_cache_text_embedding_dim": 0,
            "flow_cache_text_embedding_seq_len": 0,
            "flow_cache_text_embedding_shapes": [],
        }
    ndim = max(len(shape) for shape in shapes)
    dim = max(int(shape[-1]) for shape in shapes if shape)
    seq_len = max(int(shape[1]) if len(shape) == 3 else 1 for shape in shapes)
    return {
        "flow_cache_text_embedding_ndim": int(ndim),
        "flow_cache_text_embedding_dim": int(dim),
        "flow_cache_text_embedding_seq_len": int(seq_len),
        "flow_cache_text_embedding_shapes": shapes,
    }


def latent_cache_text_pooled_embedding_shapes(cache):
    if cache is None:
        return []
    backend = latent_cache_backend(cache)
    if backend in ("bucketed_memory", "bucketed_disk"):
        shapes = []
        for bucket in cache.get("buckets", []):
            shapes.extend(latent_cache_text_pooled_embedding_shapes(bucket.get("cache")))
        return shapes
    if backend == "disk":
        return [
            [int(x) for x in shard.get("text_pooled_embedding_shape", [])]
            for shard in cache.get("shards", [])
            if shard.get("text_pooled_embedding_shape")
        ]
    shape = cache.get("text_pooled_embedding_shape")
    if shape:
        return [[int(x) for x in shape]]
    tensor = cache.get("text_pooled_embeddings")
    if torch.is_tensor(tensor):
        return [[int(x) for x in tensor.shape]]
    return []


def latent_cache_text_pooled_embedding_report(cache):
    shapes = latent_cache_text_pooled_embedding_shapes(cache)
    if not shapes:
        return {
            "flow_cache_text_pooled_embedding_dim": 0,
            "flow_cache_text_pooled_embedding_shapes": [],
        }
    dim = max(int(shape[-1]) for shape in shapes if shape)
    return {
        "flow_cache_text_pooled_embedding_dim": int(dim),
        "flow_cache_text_pooled_embedding_shapes": shapes,
    }


def latent_cache_image_embedding_sequence_shapes(cache):
    if cache is None:
        return []
    backend = latent_cache_backend(cache)
    if backend in ("bucketed_memory", "bucketed_disk"):
        shapes = []
        for bucket in cache.get("buckets", []):
            shapes.extend(latent_cache_image_embedding_sequence_shapes(bucket.get("cache")))
        return shapes
    if backend == "disk":
        return [
            [int(x) for x in shard.get("image_embedding_sequence_shape", [])]
            for shard in cache.get("shards", [])
            if shard.get("image_embedding_sequence_shape")
        ]
    tensor = cache.get("image_embedding_sequences")
    if torch.is_tensor(tensor):
        return [[int(x) for x in tensor.shape]]
    return []


def latent_cache_image_embedding_sequence_report(cache):
    shapes = latent_cache_image_embedding_sequence_shapes(cache)
    if not shapes:
        return {
            "flow_cache_image_embedding_sequence_ndim": 0,
            "flow_cache_image_embedding_sequence_dim": 0,
            "flow_cache_image_embedding_sequence_len": 0,
            "flow_cache_image_embedding_sequence_shapes": [],
        }
    ndim = max(len(shape) for shape in shapes)
    dim = max(int(shape[-1]) for shape in shapes if shape)
    seq_len = max(int(shape[1]) if len(shape) == 3 else 1 for shape in shapes)
    return {
        "flow_cache_image_embedding_sequence_ndim": int(ndim),
        "flow_cache_image_embedding_sequence_dim": int(dim),
        "flow_cache_image_embedding_sequence_len": int(seq_len),
        "flow_cache_image_embedding_sequence_shapes": shapes,
    }


def _latent_cache_shard_offsets(cache):
    offsets, pos = [], 0
    for shard in cache.get("shards", []):
        offsets.append(pos)
        pos += int(shard["count"])
    return offsets


def configure_latent_cache_runtime(cache, max_loaded_shards=0):
    if cache is None:
        return cache
    max_loaded_shards = int(max_loaded_shards)
    if max_loaded_shards < 0:
        raise ValueError("max_loaded_shards must be non-negative")
    backend = latent_cache_backend(cache)
    cache["max_loaded_shards"] = max_loaded_shards
    cache["_shard_cache_hits"] = 0
    cache["_shard_cache_misses"] = 0
    cache["_shard_cache_loads"] = 0
    if backend == "disk" and max_loaded_shards > 0:
        cache["_loaded_shards"] = OrderedDict()
    elif "_loaded_shards" in cache:
        cache.pop("_loaded_shards", None)
    if backend in ("bucketed_memory", "bucketed_disk"):
        for bucket in cache.get("buckets", []):
            configure_latent_cache_runtime(bucket.get("cache"), max_loaded_shards)
    return cache


def latent_cache_runtime_report(cache):
    if cache is None:
        return {
            "flow_cache_max_loaded_shards": 0,
            "flow_cache_loaded_shards": 0,
            "flow_cache_shard_loads": 0,
            "flow_cache_shard_cache_hits": 0,
            "flow_cache_shard_cache_misses": 0,
        }
    backend = latent_cache_backend(cache)
    if backend in ("bucketed_memory", "bucketed_disk"):
        rows = [latent_cache_runtime_report(bucket.get("cache"))
                for bucket in cache.get("buckets", [])]
        return {
            "flow_cache_max_loaded_shards": int(cache.get("max_loaded_shards", 0)),
            "flow_cache_loaded_shards": int(sum(r["flow_cache_loaded_shards"] for r in rows)),
            "flow_cache_shard_loads": int(sum(r["flow_cache_shard_loads"] for r in rows)),
            "flow_cache_shard_cache_hits": int(sum(
                r["flow_cache_shard_cache_hits"] for r in rows)),
            "flow_cache_shard_cache_misses": int(sum(
                r["flow_cache_shard_cache_misses"] for r in rows)),
        }
    return {
        "flow_cache_max_loaded_shards": int(cache.get("max_loaded_shards", 0)),
        "flow_cache_loaded_shards": int(len(cache.get("_loaded_shards", {}))),
        "flow_cache_shard_loads": int(cache.get("_shard_cache_loads", 0)),
        "flow_cache_shard_cache_hits": int(cache.get("_shard_cache_hits", 0)),
        "flow_cache_shard_cache_misses": int(cache.get("_shard_cache_misses", 0)),
    }


def _load_latent_cache_shard(shard, cache=None):
    if cache is None:
        return torch.load(shard["path"], map_location="cpu")
    max_loaded = int(cache.get("max_loaded_shards", 0) or 0)
    path = shard["path"]
    if max_loaded <= 0:
        cache["_shard_cache_misses"] = int(cache.get("_shard_cache_misses", 0)) + 1
        cache["_shard_cache_loads"] = int(cache.get("_shard_cache_loads", 0)) + 1
        return torch.load(path, map_location="cpu")
    loaded = cache.setdefault("_loaded_shards", OrderedDict())
    if path in loaded:
        shard_payload = loaded.pop(path)
        loaded[path] = shard_payload
        cache["_shard_cache_hits"] = int(cache.get("_shard_cache_hits", 0)) + 1
        return shard_payload
    cache["_shard_cache_misses"] = int(cache.get("_shard_cache_misses", 0)) + 1
    cache["_shard_cache_loads"] = int(cache.get("_shard_cache_loads", 0)) + 1
    shard_payload = torch.load(path, map_location="cpu")
    loaded[path] = shard_payload
    while len(loaded) > max_loaded:
        loaded.popitem(last=False)
    return shard_payload

def _latent_cache_payload(source, ids=None, embeddings=None, pooled_embeddings=None,
                          pooled_masks=None, image_embeddings=None,
                          image_embedding_sequences=None, image_geometry_features=None,
                          quality_targets=None, quality_masks=None, size_bucket=""):
    return {
        "cond_source": source,
        "caption_ids": ids,
        "text_embeddings": embeddings,
        "text_pooled_embeddings": pooled_embeddings,
        "text_pooled_masks": pooled_masks,
        "image_embeddings": image_embeddings,
        "image_embedding_sequences": image_embedding_sequences,
        "image_geometry_features": image_geometry_features,
        "quality_targets": quality_targets,
        "quality_masks": quality_masks,
        "size_bucket": size_bucket,
    }


def sample_latent_cache(cache, rng, batch, device=DEV, bucket_keys=None):
    backend = latent_cache_backend(cache)
    if backend in ("bucketed_memory", "bucketed_disk"):
        buckets = list(cache.get("buckets", []))
        if bucket_keys is not None:
            allowed = {str(key) for key in bucket_keys if str(key)}
            if allowed:
                buckets = [b for b in buckets if str(b.get("key", "")) in allowed]
        if not buckets:
            raise ValueError("bucketed latent cache has no buckets")
        bucket_probs = None
        if cache.get("weighted"):
            bucket_weights = np.asarray([
                float(b.get("weight_sum", b.get("records", 0))) for b in buckets
            ], dtype=np.float64)
            bucket_probs = normalized_sampling_weights(bucket_weights, len(buckets))
        bucket_i = (
            int(rng.choice(len(buckets), p=bucket_probs))
            if bucket_probs is not None else int(rng.integers(len(buckets)))
        )
        selected = buckets[bucket_i]
        z1, payload = sample_latent_cache(
            selected["cache"], rng, batch, device=device)
        payload = dict(payload)
        payload["size_bucket"] = str(selected.get("key", ""))
        return z1, payload
    if backend == "disk":
        shards = list(cache.get("shards", []))
        if not shards:
            raise ValueError("disk latent cache has no shards")
        shard_weights = np.asarray([
            float(s.get("weight_sum", s["count"])) if cache.get("weighted")
            else float(s["count"])
            for s in shards
        ], dtype=np.float64)
        probs = normalized_sampling_weights(shard_weights, len(shards))
        shard_i = int(rng.choice(len(shards), p=probs))
        shard = _load_latent_cache_shard(shards[shard_i], cache=cache)
        n = int(shard["latents"].shape[0])
        shard_probs = normalized_sampling_weights(
            shard.get("weights"), n) if shard.get("weights") is not None else None
        if shard_probs is None:
            idx_np = rng.integers(0, n, size=int(batch))
        else:
            idx_np = rng.choice(n, size=int(batch), replace=True, p=shard_probs)
        idx = torch.tensor(idx_np, dtype=torch.long)
        z1 = shard["latents"][idx].to(device=device, dtype=torch.float32)
        image_embs = (
            shard["image_embeddings"][idx] if shard.get("image_embeddings") is not None else None
        )
        image_seq = (
            shard["image_embedding_sequences"][idx]
            if shard.get("image_embedding_sequences") is not None else None
        )
        image_geom = (
            shard["image_geometry_features"][idx]
            if shard.get("image_geometry_features") is not None else None
        )
        quality_targets = (
            shard["quality_targets"][idx] if shard.get("quality_targets") is not None else None
        )
        quality_masks = (
            shard["quality_masks"][idx] if shard.get("quality_masks") is not None else None
        )
        pooled_text = (
            shard["text_pooled_embeddings"][idx]
            if shard.get("text_pooled_embeddings") is not None else None
        )
        pooled_text_masks = (
            shard["text_pooled_masks"][idx]
            if shard.get("text_pooled_masks") is not None else None
        )
        if cache["cond_source"] == "embedding":
            payload = _latent_cache_payload(
                "embedding", embeddings=shard["text_embeddings"][idx],
                pooled_embeddings=pooled_text, pooled_masks=pooled_text_masks,
                image_embeddings=image_embs, image_embedding_sequences=image_seq,
                image_geometry_features=image_geom,
                quality_targets=quality_targets, quality_masks=quality_masks)
        else:
            payload = _latent_cache_payload(
                "tokens", ids=shard["caption_ids"][idx], image_embeddings=image_embs,
                image_embedding_sequences=image_seq,
                image_geometry_features=image_geom,
                quality_targets=quality_targets, quality_masks=quality_masks)
        return z1, payload
    cache_probs = normalized_sampling_weights(
        cache.get("weights"), int(cache["records"])) if cache.get("weights") is not None else None
    if cache_probs is None:
        idx_np = rng.integers(0, int(cache["records"]), size=int(batch))
    else:
        idx_np = rng.choice(int(cache["records"]), size=int(batch), replace=True, p=cache_probs)
    idx = torch.tensor(idx_np, dtype=torch.long)
    z1 = cache["latents"][idx].to(device=device, dtype=torch.float32)
    image_embs = (
        cache["image_embeddings"][idx] if cache.get("image_embeddings") is not None else None
    )
    image_seq = (
        cache["image_embedding_sequences"][idx]
        if cache.get("image_embedding_sequences") is not None else None
    )
    image_geom = (
        cache["image_geometry_features"][idx]
        if cache.get("image_geometry_features") is not None else None
    )
    quality_targets = (
        cache["quality_targets"][idx] if cache.get("quality_targets") is not None else None
    )
    quality_masks = (
        cache["quality_masks"][idx] if cache.get("quality_masks") is not None else None
    )
    pooled_text = (
        cache["text_pooled_embeddings"][idx]
        if cache.get("text_pooled_embeddings") is not None else None
    )
    pooled_text_masks = (
        cache["text_pooled_masks"][idx]
        if cache.get("text_pooled_masks") is not None else None
    )
    if cache["cond_source"] == "embedding":
        payload = _latent_cache_payload(
            "embedding", embeddings=cache["text_embeddings"][idx],
            pooled_embeddings=pooled_text, pooled_masks=pooled_text_masks,
            image_embeddings=image_embs, image_embedding_sequences=image_seq,
            image_geometry_features=image_geom,
            quality_targets=quality_targets, quality_masks=quality_masks)
    else:
        payload = _latent_cache_payload(
            "tokens", ids=cache["caption_ids"][idx], image_embeddings=image_embs,
            image_embedding_sequences=image_seq,
            image_geometry_features=image_geom,
            quality_targets=quality_targets, quality_masks=quality_masks)
    return z1, payload


@torch.no_grad()
def estimate_latent_stats_cache(cache, n=512, seed=123, mode="none"):
    backend = latent_cache_backend(cache)
    if backend in ("bucketed_memory", "bucketed_disk") or bool(cache.get("weighted")):
        if mode == "none":
            return {"mode": "none", "n": 0}
        if mode not in ("global", "channel"):
            raise ValueError(f"unknown latent normalization mode {mode!r}")
        total = int(cache["records"])
        if total <= 0:
            raise ValueError("latent cache is empty")
        n = min(max(1, int(n)), total)
        rng = np.random.default_rng(seed)
        count = 0
        seen = 0
        sums = sums2 = None
        while seen < n:
            b = min(64, n - seen)
            z, _payload = sample_latent_cache(cache, rng, b, device="cpu")
            z = z.detach().float().cpu().double()
            if mode == "global":
                cur_sum = z.sum()
                cur_sum2 = z.pow(2).sum()
                cur_count = z.numel()
            else:
                cur_sum = z.sum(dim=(0, 2, 3))
                cur_sum2 = z.pow(2).sum(dim=(0, 2, 3))
                cur_count = z.shape[0] * z.shape[2] * z.shape[3]
            if sums is None:
                sums = torch.zeros_like(cur_sum)
                sums2 = torch.zeros_like(cur_sum2)
            sums += cur_sum
            sums2 += cur_sum2
            count += int(cur_count)
            seen += b
        mean = sums / max(1, count)
        var = (sums2 / max(1, count) - mean.pow(2)).clamp_min(1.0e-6 ** 2)
        std = var.sqrt()
        if mode == "global":
            mean = mean.view(1, 1, 1, 1)
            std = std.view(1, 1, 1, 1)
        else:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        return {
            "mode": mode,
            "n": int(seen),
            "mean": mean.float(),
            "std": std.float(),
        }
    if backend != "disk":
        return estimate_latent_stats_tensor(cache["latents"], n=n, seed=seed, mode=mode)
    if mode == "none":
        return {"mode": "none", "n": 0}
    total = int(cache["records"])
    if total <= 0:
        raise ValueError("latent cache is empty")
    n = min(max(1, int(n)), total)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(total, size=n, replace=total < n)
    offsets = _latent_cache_shard_offsets(cache)
    grouped = {}
    for global_i in idxs:
        shard_i = int(np.searchsorted(offsets, int(global_i), side="right") - 1)
        grouped.setdefault(shard_i, []).append(int(global_i) - offsets[shard_i])
    samples = []
    for shard_i in sorted(grouped):
        shard = _load_latent_cache_shard(cache["shards"][shard_i])
        local = torch.tensor(grouped[shard_i], dtype=torch.long)
        samples.append(shard["latents"][local])
    latents = torch.cat(samples, dim=0)
    return estimate_latent_stats_tensor(latents, n=n, seed=seed + 1, mode=mode)


def latent_stats_state(stats):
    if not latent_stats_enabled(stats):
        return {"mode": "none", "n": int((stats or {}).get("n", 0))}
    return {
        "mode": stats.get("mode", "channel"),
        "n": int(stats.get("n", 0)),
        "mean": stats["mean"].detach().cpu(),
        "std": stats["std"].detach().cpu(),
    }


def latent_stats_report(stats):
    if not latent_stats_enabled(stats):
        return {"latent_normalize": "none", "latent_norm_n": 0}
    mean = stats["mean"].detach().float().cpu()
    std = stats["std"].detach().float().cpu()
    return {
        "latent_normalize": stats.get("mode", "channel"),
        "latent_norm_n": int(stats.get("n", 0)),
        "latent_norm_mean_abs": float(mean.abs().mean()),
        "latent_norm_std_mean": float(std.mean()),
        "latent_norm_std_min": float(std.min()),
        "latent_norm_std_max": float(std.max()),
    }


def resolve_latent_normalize(mode, image_records=None, ae_arch="semantic"):
    """Choose the effective latent normalization mode for a training run."""
    mode = str(mode or "none")
    if mode not in LATENT_NORMALIZE_MODES:
        raise ValueError(f"unknown latent normalization mode {mode!r}")
    if mode != "auto":
        return mode
    if image_records is not None or ae_arch == "hf-vae":
        return "channel"
    return "none"


AE_RECON_LOSSES = ("mse", "l1", "hybrid")
TRAIN_PRECISIONS = ("fp32", "bf16", "fp16")
LATENT_CACHE_DTYPES = ("fp32", "bf16", "fp16")


def _device_type(device):
    return torch.device(device).type


def amp_config(device, precision):
    precision = str(precision)
    if precision not in TRAIN_PRECISIONS:
        raise ValueError(f"unknown training precision {precision!r}")
    dev_type = _device_type(device)
    enabled = precision != "fp32" and dev_type == "cuda"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return {
        "requested": precision,
        "device_type": dev_type,
        "enabled": bool(enabled),
        "dtype": dtype,
        "dtype_name": str(dtype).replace("torch.", ""),
    }


def amp_autocast(device, precision):
    cfg = amp_config(device, precision)
    if not cfg["enabled"]:
        return nullcontext()
    return torch.autocast(device_type=cfg["device_type"], dtype=cfg["dtype"], enabled=True)


def amp_grad_scaler(device, precision):
    cfg = amp_config(device, precision)
    return torch.amp.GradScaler("cuda", enabled=bool(cfg["enabled"] and precision == "fp16"))


def latent_cache_torch_dtype(dtype):
    dtype = str(dtype)
    if dtype not in LATENT_CACHE_DTYPES:
        raise ValueError(f"unknown latent cache dtype {dtype!r}")
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype]


def cache_latent_tensor(z, dtype="fp32"):
    return z.detach().to(dtype=latent_cache_torch_dtype(dtype), device="cpu")


def image_gradient_loss(pred, target):
    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_target = target[..., :, 1:] - target[..., :, :-1]
    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_target = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)


def multiscale_recon_loss(pred, target, levels=3):
    losses = []
    cur_pred, cur_target = pred, target
    for _ in range(max(1, int(levels))):
        losses.append(F.l1_loss(cur_pred, cur_target))
        if min(cur_pred.shape[-2:]) < 4:
            break
        cur_pred = F.avg_pool2d(cur_pred, kernel_size=2, stride=2)
        cur_target = F.avg_pool2d(cur_target, kernel_size=2, stride=2)
    return torch.stack(losses).mean()


def frequency_recon_loss(pred, target, eps=1.0e-8):
    pred_f = pred.float()
    target_f = target.float()
    pred_fft = torch.fft.rfft2(pred_f, dim=(-2, -1), norm="ortho")
    target_fft = torch.fft.rfft2(target_f, dim=(-2, -1), norm="ortho")
    pred_mag = torch.log1p(pred_fft.abs().clamp_min(float(eps)))
    target_mag = torch.log1p(target_fft.abs().clamp_min(float(eps)))
    h, w = pred_f.shape[-2:]
    fy = torch.fft.fftfreq(int(h), device=pred_f.device, dtype=pred_f.dtype).abs()
    fx = torch.fft.rfftfreq(int(w), device=pred_f.device, dtype=pred_f.dtype).abs()
    radius = torch.sqrt(fy[:, None].pow(2) + fx[None, :].pow(2))
    weight = 1.0 + radius / radius.max().clamp_min(float(eps))
    return (pred_mag - target_mag).abs().mul(weight[None, None]).mean()


def reconstruction_loss_parts(pred, target, mode="mse", grad_w=0.0, ms_w=0.0,
                              fft_w=0.0, latent=None, latent_reg_w=0.0):
    if mode not in AE_RECON_LOSSES:
        raise ValueError(f"unknown AE reconstruction loss {mode!r}")
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    if mode == "mse":
        base = mse
    elif mode == "l1":
        base = l1
    else:
        base = 0.5 * (mse + l1)
    loss = base
    parts = {
        "recon_mse": mse.detach(),
        "recon_l1": l1.detach(),
        "recon_base": base.detach(),
    }
    if grad_w > 0.0:
        grad = image_gradient_loss(pred, target)
        loss = loss + float(grad_w) * grad
        parts["recon_grad_l1"] = grad.detach()
    if ms_w > 0.0:
        ms = multiscale_recon_loss(pred, target)
        loss = loss + float(ms_w) * ms
        parts["recon_multiscale_l1"] = ms.detach()
    if fft_w > 0.0:
        fft = frequency_recon_loss(pred, target)
        loss = loss + float(fft_w) * fft
        parts["recon_fft_l1"] = fft.detach()
    if latent is not None and latent_reg_w > 0.0:
        lat = latent.float().pow(2).mean()
        loss = loss + float(latent_reg_w) * lat
        parts["latent_l2"] = lat.detach()
    return loss, parts


def flow_uses_cond_tokens(flow):
    return bool(getattr(flow, "uses_cond_tokens", False))


def flow_hidden_feature_dim(flow):
    return int(getattr(flow, "hidden_feature_dim", 0) or 0)


def condition_vector(cond):
    return cond["vec"] if isinstance(cond, dict) else cond


def conditioner_has_learned_null(conditioner):
    return bool(conditioner is not None and hasattr(conditioner, "null_vec"))


def text_condition_return_tokens(conditioner, return_tokens=False):
    return bool(return_tokens or conditioner_has_learned_null(conditioner))


def image_text_alignment_loss(aligner, z, cond, prefix="caption_align", mask=None):
    if aligner is None:
        zero = z.sum() * 0.0
        return zero, {}
    img_emb, txt_emb = aligner(z, cond)
    if mask is not None:
        mask = mask.to(device=img_emb.device, dtype=torch.bool).flatten()
        img_emb = img_emb[mask]
        txt_emb = txt_emb[mask]
    n = int(img_emb.shape[0])
    if n <= 0:
        zero = z.sum() * 0.0 + condition_vector(cond).sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_i2t_acc": zero.detach(),
            f"{prefix}_t2i_acc": zero.detach(),
            f"{prefix}_n": torch.tensor(0.0, device=z.device),
        }
    logits = aligner.scale().to(img_emb.dtype) * img_emb.matmul(txt_emb.t())
    targets = torch.arange(n, device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    )
    i2t = logits.argmax(dim=1).eq(targets).float().mean()
    t2i = logits.argmax(dim=0).eq(targets).float().mean()
    diag = logits.diag().mean()
    offdiag = ((logits.sum() - logits.diag().sum()) / max(1, n * n - n)
               if n > 1 else logits.diag().mean())
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_i2t_acc": i2t.detach(),
        f"{prefix}_t2i_acc": t2i.detach(),
        f"{prefix}_diag_logit": diag.detach(),
        f"{prefix}_offdiag_logit": offdiag.detach(),
        f"{prefix}_n": torch.tensor(float(n), device=z.device),
    }


def image_feature_alignment_loss(aligner, z, features, prefix="image_feature_align", mask=None):
    if aligner is None:
        zero = z.sum() * 0.0
        return zero, {}
    features = features.to(device=z.device)
    img_emb, feat_emb = aligner(z, features)
    if mask is not None:
        mask = mask.to(device=img_emb.device, dtype=torch.bool).flatten()
        img_emb = img_emb[mask]
        feat_emb = feat_emb[mask]
    n = int(img_emb.shape[0])
    if n <= 0:
        zero = z.sum() * 0.0 + features.sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_i2f_acc": zero.detach(),
            f"{prefix}_f2i_acc": zero.detach(),
            f"{prefix}_n": torch.tensor(0.0, device=z.device),
        }
    logits = aligner.scale().to(img_emb.dtype) * img_emb.matmul(feat_emb.t())
    targets = torch.arange(n, device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    )
    i2f = logits.argmax(dim=1).eq(targets).float().mean()
    f2i = logits.argmax(dim=0).eq(targets).float().mean()
    diag = logits.diag().mean()
    offdiag = ((logits.sum() - logits.diag().sum()) / max(1, n * n - n)
               if n > 1 else logits.diag().mean())
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_i2f_acc": i2f.detach(),
        f"{prefix}_f2i_acc": f2i.detach(),
        f"{prefix}_diag_logit": diag.detach(),
        f"{prefix}_offdiag_logit": offdiag.detach(),
        f"{prefix}_n": torch.tensor(float(n), device=z.device),
    }


def pool_embedding_sequence(x):
    if x.ndim != 3:
        return x
    mask = x.float().abs().sum(dim=-1).gt(0)
    keep = mask.to(x.dtype).unsqueeze(-1)
    return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)


def embedding_pair_similarity(image_vec, text_vec, prefix="embedding_pair"):
    image_vec = pool_embedding_sequence(image_vec)
    text_vec = pool_embedding_sequence(text_vec)
    image_vec = F.normalize(image_vec.float(), dim=-1, eps=1.0e-8)
    text_vec = F.normalize(text_vec.float(), dim=-1, eps=1.0e-8)
    logits = image_vec.matmul(text_vec.t())
    targets = torch.arange(logits.shape[0], device=logits.device)
    return {
        f"{prefix}_cos": logits.diag().mean().detach(),
        f"{prefix}_i2t_acc": logits.argmax(dim=1).eq(targets).float().mean().detach(),
        f"{prefix}_t2i_acc": logits.argmax(dim=0).eq(targets).float().mean().detach(),
        f"{prefix}_n": torch.tensor(float(logits.shape[0]), device=logits.device),
    }


def _embedding_covariance(x):
    x = x.float()
    d = int(x.shape[-1])
    if int(x.shape[0]) <= 1:
        return torch.zeros((d, d), dtype=x.dtype, device=x.device)
    centered = x - x.mean(dim=0, keepdim=True)
    return centered.t().matmul(centered) / float(int(x.shape[0]) - 1)


def _trace_sqrt_product(a, b):
    prod = a.matmul(b)
    vals = torch.linalg.eigvals(prod).real.clamp_min(0.0).sqrt()
    return vals.sum()


def _rbf_mmd(x, y, eps=1.0e-8):
    z = torch.cat([x, y], dim=0)
    if int(z.shape[0]) <= 1:
        sigma2 = torch.tensor(1.0, dtype=x.dtype, device=x.device)
    else:
        dists = torch.pdist(z.float(), p=2).pow(2)
        positive = dists[dists > eps]
        sigma2 = (
            positive.median() if int(positive.numel()) else
            torch.tensor(1.0, dtype=x.dtype, device=x.device)
        ).clamp_min(eps)
    kxx = torch.exp(-torch.cdist(x.float(), x.float()).pow(2) / (2.0 * sigma2)).mean()
    kyy = torch.exp(-torch.cdist(y.float(), y.float()).pow(2) / (2.0 * sigma2)).mean()
    kxy = torch.exp(-torch.cdist(x.float(), y.float()).pow(2) / (2.0 * sigma2)).mean()
    return (kxx + kyy - 2.0 * kxy).clamp_min(0.0), sigma2


def _pairwise_summary(x, eps=1.0e-8):
    n = int(x.shape[0])
    if n <= 1:
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return zero, zero
    cos = x.float().matmul(x.float().t())
    dist = torch.cdist(x.float(), x.float())
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return cos[mask].mean(), dist[mask].mean()


def _support_radius(x, eps=1.0e-8):
    n = int(x.shape[0])
    if n <= 1:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)
    dist = torch.cdist(x.float(), x.float())
    dist = dist.masked_fill(torch.eye(n, dtype=torch.bool, device=x.device), float("inf"))
    nearest = dist.min(dim=1).values
    finite = nearest[torch.isfinite(nearest)]
    if int(finite.numel()) <= 0:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return finite.median().clamp_min(0.0)


def embedding_distribution_metrics(generated_vec, real_vec, prefix="embedding_distribution"):
    generated_vec = F.normalize(pool_embedding_sequence(generated_vec).float(), dim=-1, eps=1.0e-8)
    real_vec = F.normalize(pool_embedding_sequence(real_vec).float(), dim=-1, eps=1.0e-8)
    if int(generated_vec.shape[-1]) != int(real_vec.shape[-1]):
        raise ValueError(
            f"generated/real embedding dims differ: "
            f"{int(generated_vec.shape[-1])} vs {int(real_vec.shape[-1])}"
        )
    n = min(int(generated_vec.shape[0]), int(real_vec.shape[0]))
    if n <= 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_matched_cos": 0.0,
            f"{prefix}_mean_l2": 0.0,
            f"{prefix}_mean_sq": 0.0,
            f"{prefix}_mean_gap_l2": 0.0,
            f"{prefix}_cov_fro": 0.0,
            f"{prefix}_frechet": 0.0,
            f"{prefix}_mmd_rbf": 0.0,
            f"{prefix}_mmd_rbf_sigma2": 1.0,
            f"{prefix}_generated_pairwise_cos": 0.0,
            f"{prefix}_real_pairwise_cos": 0.0,
            f"{prefix}_generated_pairwise_l2": 0.0,
            f"{prefix}_real_pairwise_l2": 0.0,
            f"{prefix}_diversity_l2_ratio": 0.0,
            f"{prefix}_nearest_real_l2": 0.0,
            f"{prefix}_nearest_generated_l2": 0.0,
            f"{prefix}_support_precision": 0.0,
            f"{prefix}_support_recall": 0.0,
            f"{prefix}_real_support_radius": 0.0,
            f"{prefix}_generated_support_radius": 0.0,
        }
    generated_vec = generated_vec[:n]
    real_vec = real_vec[:n]
    diff = generated_vec - real_vec
    mean_diff = generated_vec.mean(dim=0) - real_vec.mean(dim=0)
    gen_cov = _embedding_covariance(generated_vec)
    real_cov = _embedding_covariance(real_vec)
    cov_diff = gen_cov - real_cov
    gen_pair_cos, gen_pair_l2 = _pairwise_summary(generated_vec)
    real_pair_cos, real_pair_l2 = _pairwise_summary(real_vec)
    cross_l2 = torch.cdist(generated_vec.float(), real_vec.float())
    nearest_real = cross_l2.min(dim=1).values
    nearest_generated = cross_l2.min(dim=0).values
    real_radius = _support_radius(real_vec)
    generated_radius = _support_radius(generated_vec)
    support_precision = nearest_real.le(real_radius).float().mean()
    support_recall = nearest_generated.le(generated_radius).float().mean()
    frechet = (
        mean_diff.pow(2).sum()
        + torch.trace(gen_cov)
        + torch.trace(real_cov)
        - 2.0 * _trace_sqrt_product(gen_cov, real_cov)
    ).clamp_min(0.0)
    mmd, sigma2 = _rbf_mmd(generated_vec, real_vec)
    return {
        f"{prefix}_n": int(n),
        f"{prefix}_matched_cos": float((generated_vec * real_vec).sum(dim=-1).mean().detach().cpu()),
        f"{prefix}_mean_l2": float(diff.norm(dim=-1).mean().detach().cpu()),
        f"{prefix}_mean_sq": float(diff.pow(2).sum(dim=-1).mean().detach().cpu()),
        f"{prefix}_mean_gap_l2": float(mean_diff.norm().detach().cpu()),
        f"{prefix}_cov_fro": float(cov_diff.norm().detach().cpu()),
        f"{prefix}_frechet": float(frechet.detach().cpu()),
        f"{prefix}_mmd_rbf": float(mmd.detach().cpu()),
        f"{prefix}_mmd_rbf_sigma2": float(sigma2.detach().cpu()),
        f"{prefix}_generated_pairwise_cos": float(gen_pair_cos.detach().cpu()),
        f"{prefix}_real_pairwise_cos": float(real_pair_cos.detach().cpu()),
        f"{prefix}_generated_pairwise_l2": float(gen_pair_l2.detach().cpu()),
        f"{prefix}_real_pairwise_l2": float(real_pair_l2.detach().cpu()),
        f"{prefix}_diversity_l2_ratio": float(
            (gen_pair_l2 / real_pair_l2.clamp_min(1.0e-8)).detach().cpu()
        ),
        f"{prefix}_nearest_real_l2": float(nearest_real.mean().detach().cpu()),
        f"{prefix}_nearest_generated_l2": float(nearest_generated.mean().detach().cpu()),
        f"{prefix}_support_precision": float(support_precision.detach().cpu()),
        f"{prefix}_support_recall": float(support_recall.detach().cpu()),
        f"{prefix}_real_support_radius": float(real_radius.detach().cpu()),
        f"{prefix}_generated_support_radius": float(generated_radius.detach().cpu()),
    }


def can_score_external_text_image(records, image_feature_aligner):
    if image_feature_aligner is None:
        return False
    try:
        text_dim = infer_text_embedding_dim(records)
    except ValueError:
        return False
    return int(text_dim) == int(getattr(image_feature_aligner, "feature_dim", 0) or 0)


def flow_hidden_feature_alignment_loss(aligner, hidden_tokens, features, prefix="flow_repa",
                                       mask=None):
    if aligner is None:
        zero = hidden_tokens.sum() * 0.0
        return zero, {}
    features = features.to(device=hidden_tokens.device)
    hidden_emb, feat_emb = aligner(hidden_tokens, features)
    if mask is not None:
        mask = mask.to(device=hidden_emb.device, dtype=torch.bool).flatten()
        hidden_emb = hidden_emb[mask]
        feat_emb = feat_emb[mask]
    n = int(hidden_emb.shape[0])
    if n <= 0:
        zero = hidden_tokens.sum() * 0.0 + features.sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_h2f_acc": zero.detach(),
            f"{prefix}_f2h_acc": zero.detach(),
            f"{prefix}_n": torch.tensor(0.0, device=hidden_tokens.device),
        }
    logits = aligner.scale().to(hidden_emb.dtype) * hidden_emb.matmul(feat_emb.t())
    targets = torch.arange(n, device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    )
    h2f = logits.argmax(dim=1).eq(targets).float().mean()
    f2h = logits.argmax(dim=0).eq(targets).float().mean()
    diag = logits.diag().mean()
    offdiag = ((logits.sum() - logits.diag().sum()) / max(1, n * n - n)
               if n > 1 else logits.diag().mean())
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_h2f_acc": h2f.detach(),
        f"{prefix}_f2h_acc": f2h.detach(),
        f"{prefix}_diag_logit": diag.detach(),
        f"{prefix}_offdiag_logit": offdiag.detach(),
        f"{prefix}_n": torch.tensor(float(n), device=hidden_tokens.device),
    }


def flow_hidden_feature_token_alignment_loss(aligner, hidden_tokens, feature_tokens,
                                             prefix="flow_repa_token"):
    if aligner is None:
        zero = hidden_tokens.sum() * 0.0
        return zero, {}
    feature_tokens = feature_tokens.to(device=hidden_tokens.device)
    hidden_emb = aligner.encode_hidden_tokens(hidden_tokens)
    feat_emb = aligner.encode_feature_tokens(feature_tokens)
    valid = feature_tokens.float().abs().sum(dim=-1).gt(0)
    bsz, hidden_n, _dim = hidden_emb.shape
    feat_n = int(feat_emb.shape[1])
    if int(valid.sum()) <= 0 or hidden_n <= 0 or feat_n <= 0:
        zero = hidden_tokens.sum() * 0.0 + feature_tokens.sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_h2f_cos": zero.detach(),
            f"{prefix}_f2h_cos": zero.detach(),
            f"{prefix}_n": torch.tensor(0.0, device=hidden_tokens.device),
        }
    sim = torch.bmm(hidden_emb, feat_emb.transpose(1, 2))
    masked_sim = sim.masked_fill(~valid[:, None, :], -float("inf"))
    h2f = masked_sim.max(dim=2).values
    f2h = sim.max(dim=1).values
    f2h = f2h.masked_fill(~valid, 0.0)
    valid_count = valid.sum().clamp_min(1).to(f2h.dtype)
    h2f_mean = h2f[torch.isfinite(h2f)].mean()
    f2h_mean = f2h.sum() / valid_count
    match_loss = 0.5 * ((1.0 - h2f_mean) + (1.0 - f2h_mean))
    return match_loss, {
        f"{prefix}_loss": match_loss.detach(),
        f"{prefix}_h2f_cos": h2f_mean.detach(),
        f"{prefix}_f2h_cos": f2h_mean.detach(),
        f"{prefix}_n": torch.tensor(float(bsz), device=hidden_tokens.device),
        f"{prefix}_hidden_tokens": torch.tensor(float(hidden_n), device=hidden_tokens.device),
        f"{prefix}_feature_tokens": valid.float().sum(dim=1).mean().detach(),
    }


def flow_hidden_latent_alignment_loss(aligner, hidden_tokens, clean_latent,
                                      prefix="flow_self_repa", mask=None):
    if aligner is None:
        zero = hidden_tokens.sum() * 0.0
        return zero, {}
    clean_latent = clean_latent.to(device=hidden_tokens.device)
    hidden_emb, latent_emb = aligner(hidden_tokens, clean_latent)
    if mask is not None:
        mask = mask.to(device=hidden_emb.device, dtype=torch.bool).flatten()
        hidden_emb = hidden_emb[mask]
        latent_emb = latent_emb[mask]
    n = int(hidden_emb.shape[0])
    if n <= 0:
        zero = hidden_tokens.sum() * 0.0 + clean_latent.sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_h2z_cos": zero.detach(),
            f"{prefix}_n": torch.tensor(0.0, device=hidden_tokens.device),
        }
    cos = (hidden_emb * latent_emb.detach()).sum(dim=-1)
    loss = (1.0 - cos).mean()
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_h2z_cos": cos.mean().detach(),
        f"{prefix}_n": torch.tensor(float(n), device=hidden_tokens.device),
    }


def flow_hidden_latent_token_alignment_loss(aligner, hidden_tokens, clean_latent,
                                            prefix="flow_self_repa_token"):
    if aligner is None:
        zero = hidden_tokens.sum() * 0.0
        return zero, {}
    clean_latent = clean_latent.to(device=hidden_tokens.device)
    hidden_emb = aligner.encode_hidden_tokens(hidden_tokens)
    latent_emb = aligner.encode_latent_tokens(clean_latent).detach()
    if hidden_emb.shape[:2] != latent_emb.shape[:2]:
        raise ValueError(
            f"self-REPA token counts differ: hidden={tuple(hidden_emb.shape)} "
            f"latent={tuple(latent_emb.shape)}"
        )
    if int(hidden_emb.shape[1]) <= 0:
        zero = hidden_tokens.sum() * 0.0 + clean_latent.sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_h2z_cos": zero.detach(),
            f"{prefix}_n": torch.tensor(0.0, device=hidden_tokens.device),
        }
    cos = (hidden_emb * latent_emb).sum(dim=-1)
    loss = (1.0 - cos).mean()
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_h2z_cos": cos.mean().detach(),
        f"{prefix}_n": torch.tensor(float(cos.numel()), device=hidden_tokens.device),
        f"{prefix}_tokens": torch.tensor(float(hidden_emb.shape[1]),
                                         device=hidden_tokens.device),
    }


def flow_hidden_self_representation_alignment_loss(hidden_tokens, target_tokens,
                                                   prefix="flow_sra",
                                                   pooled=False):
    if hidden_tokens.ndim != 3 or target_tokens.ndim != 3:
        raise ValueError(
            f"expected self-representation tokens as B,N,H, got "
            f"{tuple(hidden_tokens.shape)} and {tuple(target_tokens.shape)}")
    if hidden_tokens.shape != target_tokens.shape:
        raise ValueError(
            f"self-representation token shapes differ: hidden={tuple(hidden_tokens.shape)} "
            f"target={tuple(target_tokens.shape)}")
    if pooled:
        hidden_emb = hidden_tokens.float().mean(dim=1)
        target_emb = target_tokens.float().mean(dim=1)
    else:
        hidden_emb = hidden_tokens.float()
        target_emb = target_tokens.float()
    hidden_emb = F.normalize(hidden_emb, dim=-1)
    target_emb = F.normalize(target_emb.detach(), dim=-1)
    cos = (hidden_emb * target_emb).sum(dim=-1)
    loss = (1.0 - cos).mean()
    parts = {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_h2t_cos": cos.mean().detach(),
        f"{prefix}_n": torch.tensor(float(cos.numel()), device=hidden_tokens.device),
    }
    if not pooled:
        parts[f"{prefix}_tokens"] = torch.tensor(
            float(hidden_tokens.shape[1]), device=hidden_tokens.device)
    return loss, parts


def condition_batch(cond):
    return int(condition_vector(cond).shape[0])


def zero_condition(cond):
    if not isinstance(cond, dict):
        return torch.zeros_like(cond)
    out = dict(cond)
    out["vec"] = torch.zeros_like(cond["vec"])
    out["tokens"] = torch.zeros_like(cond["tokens"])
    return out


def null_condition(cond):
    """Return a trained unconditional condition when present, otherwise zeros."""
    if not isinstance(cond, dict) or "null_vec" not in cond:
        return zero_condition(cond)
    out = dict(cond)
    out["vec"] = cond["null_vec"].to(device=cond["vec"].device, dtype=cond["vec"].dtype)
    if "tokens" in cond:
        if "null_tokens" in cond:
            out["tokens"] = cond["null_tokens"].to(
                device=cond["tokens"].device, dtype=cond["tokens"].dtype)
        else:
            out["tokens"] = torch.zeros_like(cond["tokens"])
    if "mask" in cond:
        if "null_mask" in cond:
            out["mask"] = cond["null_mask"].to(device=cond["mask"].device, dtype=torch.bool)
        else:
            out["mask"] = torch.zeros_like(cond["mask"], dtype=torch.bool)
    return out


def condition_has_null(cond):
    return isinstance(cond, dict) and "null_vec" in cond


def _pad_condition_values_for_concat(key, values):
    if not values:
        return values
    if key in ("tokens", "null_tokens") and values[0].ndim == 3:
        max_len = max(int(val.shape[1]) for val in values)
        out = []
        for val in values:
            pad = max_len - int(val.shape[1])
            if pad > 0:
                val = F.pad(val, (0, 0, 0, pad), value=0.0)
            out.append(val)
        return out
    if key in ("mask", "null_mask") and values[0].ndim == 2:
        max_len = max(int(val.shape[1]) for val in values)
        out = []
        for val in values:
            pad = max_len - int(val.shape[1])
            if pad > 0:
                val = F.pad(val, (0, pad), value=True)
            out.append(val)
        return out
    return values


def concat_conditions(*conds):
    if not conds:
        raise ValueError("concat_conditions requires at least one condition")
    if not isinstance(conds[0], dict):
        if any(isinstance(cond, dict) for cond in conds):
            raise ValueError("cannot concatenate mixed tensor/dict conditions")
        return torch.cat(conds, dim=0)
    if any(not isinstance(cond, dict) for cond in conds):
        raise ValueError("cannot concatenate mixed tensor/dict conditions")
    first_batch = condition_batch(conds[0])
    out = dict(conds[0])
    for key, first in conds[0].items():
        if not torch.is_tensor(first) or int(first.shape[0]) != first_batch:
            continue
        values = []
        for cond in conds:
            if key not in cond:
                raise ValueError(f"condition is missing batched field {key!r}")
            val = cond[key]
            if not torch.is_tensor(val):
                raise ValueError(f"condition field {key!r} is not a tensor")
            values.append(val)
        values = _pad_condition_values_for_concat(key, values)
        out[key] = torch.cat(values, dim=0)
    return out


def repeat_condition_rows(cond, repeats):
    repeats = int(repeats)
    if cond is None:
        return None
    if repeats <= 1:
        return cond
    if not isinstance(cond, dict):
        return cond.repeat_interleave(repeats, dim=0)
    batch = condition_batch(cond)
    out = dict(cond)
    for key, val in cond.items():
        if torch.is_tensor(val) and int(val.shape[0]) == batch:
            out[key] = val.repeat_interleave(repeats, dim=0)
    return out


def condition_dropout(cond, p=0.0):
    """Classifier-free condition dropout over whole condition rows during training."""
    if p <= 0.0:
        return cond
    vec = condition_vector(cond)
    if p >= 1.0:
        out = null_condition(cond)
        if isinstance(out, dict):
            out["drop_mask"] = torch.ones(
                vec.shape[0], dtype=torch.bool, device=vec.device)
        return out
    drop = torch.rand(vec.shape[0], device=vec.device) < float(p)
    keep = (~drop).to(vec.dtype)[:, None]
    if not isinstance(cond, dict):
        return cond * keep
    null = null_condition(cond)
    out = dict(cond)
    out["vec"] = torch.where(drop[:, None], null["vec"], cond["vec"])
    out["tokens"] = torch.where(drop[:, None, None], null["tokens"], cond["tokens"])
    if "mask" in cond:
        out["mask"] = torch.where(drop[:, None], null.get("mask", cond["mask"]), cond["mask"])
    out["drop_mask"] = drop
    return out


def condition_active_mask(cond, keepdim=False):
    vec = condition_vector(cond)
    if isinstance(cond, dict) and "drop_mask" in cond:
        active = ~cond["drop_mask"].to(device=vec.device, dtype=torch.bool).flatten()
    else:
        active = vec.detach().abs().sum(dim=1).gt(0)
    return active[:, None] if keepdim else active


def text_alignment_guidance_step(text_aligner, z, cond, weight=0.0, step_size=1.0,
                                 eps=1.0e-6):
    """Sampling-time latent guidance toward a learned image/text alignment score."""
    if weight <= 0.0:
        return z
    if text_aligner is None:
        raise ValueError("text alignment guidance requires a checkpoint text_aligner")
    z_var = z.detach().requires_grad_(True)
    with torch.enable_grad():
        img_emb, txt_emb = text_aligner(z_var, cond)
        score = text_aligner.scale().to(img_emb.dtype) * (img_emb * txt_emb).sum(dim=-1)
        objective = score.sum()
        grad = torch.autograd.grad(objective, z_var, allow_unused=False)[0]
    flat = grad.flatten(1)
    denom = flat.norm(dim=1).view((-1,) + (1,) * (grad.ndim - 1)).clamp_min(float(eps))
    guided = z_var + float(weight) * abs(float(step_size)) * grad / denom
    return guided.detach()


def image_feature_guidance_step(image_feature_aligner, z, features, weight=0.0,
                                step_size=1.0, eps=1.0e-6):
    """Sampling-time latent guidance toward an external image/text feature target."""
    if weight <= 0.0:
        return z
    if image_feature_aligner is None:
        raise ValueError("image feature guidance requires a checkpoint image_feature_aligner")
    features = pool_embedding_sequence(features).to(device=z.device)
    z_var = z.detach().requires_grad_(True)
    with torch.enable_grad():
        img_emb, feat_emb = image_feature_aligner(z_var, features)
        score = image_feature_aligner.scale().to(img_emb.dtype) * (
            img_emb * feat_emb
        ).sum(dim=-1)
        objective = score.sum()
        grad = torch.autograd.grad(objective, z_var, allow_unused=False)[0]
    flat = grad.flatten(1)
    denom = flat.norm(dim=1).view((-1,) + (1,) * (grad.ndim - 1)).clamp_min(float(eps))
    guided = z_var + float(weight) * abs(float(step_size)) * grad / denom
    return guided.detach()


def image_quality_guidance_step(scorer, z, weight=0.0, step_size=1.0, eps=1.0e-6):
    """Sampling-time latent guidance toward a learned manifest quality score."""
    if weight <= 0.0:
        return z
    if scorer is None:
        raise ValueError("quality guidance requires a checkpoint image_quality_scorer")
    z_var = z.detach().requires_grad_(True)
    with torch.enable_grad():
        objective = scorer.score(z_var).sum()
        grad = torch.autograd.grad(objective, z_var, allow_unused=False)[0]
    flat = grad.flatten(1)
    denom = flat.norm(dim=1).view((-1,) + (1,) * (grad.ndim - 1)).clamp_min(float(eps))
    guided = z_var + float(weight) * abs(float(step_size)) * grad / denom
    return guided.detach()


def couple_flow_noise_to_data(x0, z1, mode="random", projections=1, eps=1.0e-8):
    mode = str(mode)
    if mode not in FLOW_NOISE_COUPLINGS:
        raise ValueError(f"unknown flow noise coupling {mode!r}")
    projections = int(projections)
    if projections <= 0:
        raise ValueError("flow noise coupling projections must be positive")
    before = (x0.float() - z1.float()).pow(2).flatten(1).mean(dim=1).mean()
    if mode == "random" or int(x0.shape[0]) <= 1:
        return x0, {
            "flow_noise_pair_mse_before": before.detach(),
            "flow_noise_pair_mse_after": before.detach(),
            "flow_noise_pair_mse_delta": torch.zeros((), device=x0.device),
            "flow_noise_coupling_active": torch.tensor(0.0, device=x0.device),
            "flow_noise_coupling_accepted": torch.tensor(0.0, device=x0.device),
            "flow_noise_coupling_projections": torch.tensor(0.0, device=x0.device),
            "flow_noise_coupling_selected_projection": torch.tensor(-1.0, device=x0.device),
        }
    flat_noise = x0.float().flatten(1)
    flat_data = z1.float().flatten(1)
    best_matched = None
    best_after = None
    best_i = 0
    for proj_i in range(projections):
        direction = torch.randn(flat_noise.shape[1], device=x0.device, dtype=torch.float32)
        direction = direction / direction.norm().clamp_min(float(eps))
        noise_order = torch.argsort(flat_noise.matmul(direction), dim=0)
        data_order = torch.argsort(flat_data.matmul(direction), dim=0)
        matched = torch.empty_like(x0)
        matched[data_order] = x0[noise_order]
        after_i = (matched.float() - z1.float()).pow(2).flatten(1).mean(dim=1).mean()
        if best_after is None or bool(after_i < best_after):
            best_matched = matched
            best_after = after_i
            best_i = int(proj_i)
    if bool(best_after >= before):
        return x0, {
            "flow_noise_pair_mse_before": before.detach(),
            "flow_noise_pair_mse_after": before.detach(),
            "flow_noise_pair_mse_delta": torch.zeros((), device=x0.device),
            "flow_noise_coupling_active": torch.tensor(1.0, device=x0.device),
            "flow_noise_coupling_accepted": torch.tensor(0.0, device=x0.device),
            "flow_noise_coupling_projections": torch.tensor(float(projections), device=x0.device),
            "flow_noise_coupling_selected_projection": torch.tensor(float(best_i), device=x0.device),
        }
    after = best_after
    return best_matched, {
        "flow_noise_pair_mse_before": before.detach(),
        "flow_noise_pair_mse_after": after.detach(),
        "flow_noise_pair_mse_delta": (before - after).detach(),
        "flow_noise_coupling_active": torch.tensor(1.0, device=x0.device),
        "flow_noise_coupling_accepted": torch.tensor(1.0, device=x0.device),
        "flow_noise_coupling_projections": torch.tensor(float(projections), device=x0.device),
        "flow_noise_coupling_selected_projection": torch.tensor(float(best_i), device=x0.device),
    }


def effective_flow_repa_mode(mode, image_feature_tokens=None):
    mode = str(mode or "pooled")
    if mode not in FLOW_REPA_MODES:
        raise ValueError(f"unknown flow_repa_mode {mode!r}")
    if mode == "auto":
        return "both" if image_feature_tokens is not None else "pooled"
    return mode


def effective_flow_self_repa_mode(mode):
    mode = str(mode or "pooled")
    if mode not in FLOW_REPA_MODES:
        raise ValueError(f"unknown flow_self_repa_mode {mode!r}")
    return "both" if mode == "auto" else mode


def effective_flow_sra_mode(mode):
    mode = str(mode or "token")
    if mode not in FLOW_REPA_MODES:
        raise ValueError(f"unknown flow_sra_mode {mode!r}")
    return "both" if mode == "auto" else mode


def flow_repa_mode_id(mode):
    return float(FLOW_REPA_MODES.index(str(mode)) if str(mode) in FLOW_REPA_MODES else -1)


def latent_flow_losses(flow, z1, cond, cond_drop=0.0, ae=None,
                       time_sampling="uniform", time_logit_mean=0.0,
                       time_logit_std=1.0,
                       time_mode_scale=DEFAULT_TIME_MODE_SCALE, time_shift=1.0,
                       latent_stats=None, consistency_w=0.0,
                       endpoint_w=0.0,
                       flow_noise_coupling="random", flow_noise_coupling_projections=1,
                       flow_loss_weight="none", flow_loss_weight_gamma=5.0,
                       flow_loss_weight_normalize=True,
                       text_aligner=None, text_align_w=0.0,
                       feature_aligner=None, image_features=None, feature_align_w=0.0,
                       repa_aligner=None, repa_w=0.0, image_feature_tokens=None,
                       repa_mode="pooled",
                       self_repa_aligner=None, self_repa_w=0.0,
                       self_repa_mode="pooled",
                       sra_w=0.0, sra_time_gap=0.25, sra_mode="token",
                       quality_scorer=None, quality_records=None, quality_stats=None,
                       quality_targets=None, quality_masks=None,
                       quality_w=0.0, quality_rank_w=0.0, quality_rank_margin=0.0,
                       adaptive_sampler=None):
    latent_stats = flow_latent_stats(flow) if latent_stats is None else latent_stats
    endpoint_w = float(endpoint_w)
    if endpoint_w < 0.0:
        raise ValueError("endpoint_w must be non-negative")
    z1_model = normalize_latent(z1, latent_stats)
    x0 = torch.randn_like(z1_model)
    x0, coupling_parts = couple_flow_noise_to_data(
        x0, z1_model, mode=flow_noise_coupling,
        projections=flow_noise_coupling_projections)
    t = sample_flow_times(z1.shape[0], device=z1.device, mode=time_sampling,
                          logit_mean=time_logit_mean, logit_std=time_logit_std,
                          mode_scale=time_mode_scale, time_shift=time_shift,
                          adaptive_sampler=adaptive_sampler)
    zt = (1.0 - t) * x0 + t * z1_model
    target = z1_model - x0
    cond_model = condition_dropout(cond, cond_drop)
    flow_features = None
    need_flow_features = (
        (repa_aligner is not None and repa_w > 0.0)
        or (self_repa_aligner is not None and self_repa_w > 0.0)
        or sra_w > 0.0
    )
    if need_flow_features:
        pred, flow_features = flow_velocity(flow, zt, t, cond_model, return_features=True)
    else:
        pred = flow_velocity(flow, zt, t, cond_model)
    per_sample_velocity = (pred - target).float().pow(2).flatten(1).mean(dim=1)
    if time_sampling == "adaptive" and adaptive_sampler is not None:
        adaptive_sampler.update(t, per_sample_velocity, time_shift=time_shift)
    if flow_loss_weight == "none":
        velocity_weights = torch.ones(z1.shape[0], device=z1.device)
    else:
        velocity_weights = flow_loss_time_weights(
            t, mode=flow_loss_weight, gamma=flow_loss_weight_gamma,
            normalize=flow_loss_weight_normalize)
    velocity_unweighted = per_sample_velocity.mean()
    velocity = (per_sample_velocity * velocity_weights.to(per_sample_velocity.dtype)).mean()
    endpoint_pred = zt + (1.0 - t) * pred
    per_sample_endpoint = (endpoint_pred - z1_model).float().pow(2).flatten(1).mean(dim=1)
    endpoint_unweighted = per_sample_endpoint.mean()
    endpoint_weighted = (
        per_sample_endpoint * velocity_weights.to(per_sample_endpoint.dtype)).mean()
    total = velocity
    parts = {
        "velocity_mse": velocity.detach(),
        "velocity_mse_unweighted": velocity_unweighted.detach(),
        "flow_endpoint_target_mse": endpoint_weighted.detach(),
        "flow_endpoint_target_mse_unweighted": endpoint_unweighted.detach(),
        "flow_endpoint_w": torch.tensor(float(endpoint_w), device=z1.device),
        "velocity_weight_mean": velocity_weights.detach().mean(),
        "velocity_weight_min": velocity_weights.detach().min(),
        "velocity_weight_max": velocity_weights.detach().max(),
        "velocity_weight_gamma": torch.tensor(float(flow_loss_weight_gamma), device=z1.device),
        "time_mean": t.detach().mean(),
        "time_std": t.detach().std(unbiased=False),
        "time_shift": torch.tensor(float(time_shift), device=z1.device),
    }
    parts.update(coupling_parts)
    if endpoint_w > 0.0:
        total = total + float(endpoint_w) * endpoint_weighted
    if consistency_w > 0.0:
        t2 = sample_flow_times(z1.shape[0], device=z1.device, mode=time_sampling,
                               logit_mean=time_logit_mean, logit_std=time_logit_std,
                               mode_scale=time_mode_scale, time_shift=time_shift,
                               adaptive_sampler=adaptive_sampler)
        zt2 = (1.0 - t2) * x0 + t2 * z1_model
        pred2 = flow_velocity(flow, zt2, t2, cond_model)
        endpoint1 = endpoint_pred
        endpoint2 = zt2 + (1.0 - t2) * pred2
        consistency = 0.5 * (
            F.mse_loss(endpoint1, endpoint2.detach())
            + F.mse_loss(endpoint2, endpoint1.detach())
        )
        total = total + float(consistency_w) * consistency
        parts["endpoint_consistency_mse"] = consistency.detach()
    z_clean = None
    if text_aligner is not None and text_align_w > 0.0:
        if z_clean is None:
            z_clean = denormalize_latent(endpoint_pred, latent_stats)
        keep = condition_active_mask(cond_model)
        text_align, text_parts = image_text_alignment_loss(
            text_aligner, z_clean, cond_model, prefix="flow_caption_align", mask=keep)
        total = total + float(text_align_w) * text_align
        parts.update(text_parts)
    if quality_scorer is not None and (quality_w > 0.0 or quality_rank_w > 0.0):
        if z_clean is None:
            z_clean = denormalize_latent(endpoint_pred, latent_stats)
        if quality_targets is None or quality_masks is None:
            if quality_records is None or quality_stats is None:
                raise ValueError("flow quality loss requires quality records/stats or cached targets")
            quality_targets, quality_masks = quality_target_tensor(
                quality_records, quality_stats, device=z_clean.device)
        if quality_w > 0.0:
            quality_loss, quality_parts = image_quality_score_target_loss(
                quality_scorer, z_clean, quality_targets, quality_masks,
                prefix="flow_quality_score")
            total = total + float(quality_w) * quality_loss
            parts.update(quality_parts)
        if quality_rank_w > 0.0:
            quality_rank, quality_rank_parts = image_quality_rank_target_loss(
                quality_scorer, z_clean, quality_targets, quality_masks,
                prefix="flow_quality_rank", margin=quality_rank_margin)
            total = total + float(quality_rank_w) * quality_rank
            parts.update(quality_rank_parts)
    if feature_aligner is not None and feature_align_w > 0.0:
        if image_features is None:
            raise ValueError("flow feature alignment requires image_features")
        if z_clean is None:
            z_clean = denormalize_latent(endpoint_pred, latent_stats)
        feature_align, feature_parts = image_feature_alignment_loss(
            feature_aligner, z_clean, image_features, prefix="flow_image_feature_align")
        total = total + float(feature_align_w) * feature_align
        parts.update(feature_parts)
    if repa_aligner is not None and repa_w > 0.0:
        if flow_features is None or "image_tokens" not in flow_features:
            raise ValueError("flow REPA alignment requires transformer image-token features")
        active_repa_mode = effective_flow_repa_mode(
            repa_mode, image_feature_tokens=image_feature_tokens)
        repa_losses = []
        repa_parts = {}
        if active_repa_mode in ("pooled", "both"):
            pooled_features = image_features
            if pooled_features is None and image_feature_tokens is not None:
                pooled_features = pool_embedding_sequence(image_feature_tokens)
            if pooled_features is None:
                raise ValueError("pooled flow REPA alignment requires image_features")
            pooled_repa, pooled_parts = flow_hidden_feature_alignment_loss(
                repa_aligner, flow_features["image_tokens"], pooled_features,
                prefix="flow_repa")
            repa_losses.append(pooled_repa)
            repa_parts.update(pooled_parts)
        if active_repa_mode in ("token", "both"):
            if image_feature_tokens is None:
                raise ValueError("token flow REPA alignment requires image_embedding_sequence rows")
            token_repa, token_parts = flow_hidden_feature_token_alignment_loss(
                repa_aligner, flow_features["image_tokens"], image_feature_tokens,
                prefix="flow_repa_token")
            repa_losses.append(token_repa)
            repa_parts.update(token_parts)
        if not repa_losses:
            raise ValueError(f"flow REPA mode {active_repa_mode!r} did not select any losses")
        repa_align = torch.stack(repa_losses).mean()
        total = total + float(repa_w) * repa_align
        parts.update(repa_parts)
        parts["flow_repa_loss"] = repa_align.detach()
        parts["flow_repa_w"] = torch.tensor(float(repa_w), device=z1.device)
        parts["flow_repa_mode_id"] = torch.tensor(
            flow_repa_mode_id(active_repa_mode), device=z1.device)
        parts["flow_repa_components"] = torch.tensor(float(len(repa_losses)), device=z1.device)
    if self_repa_aligner is not None and self_repa_w > 0.0:
        if flow_features is None or "image_tokens" not in flow_features:
            raise ValueError("flow self-REPA alignment requires transformer image-token features")
        active_self_repa_mode = effective_flow_self_repa_mode(self_repa_mode)
        self_repa_losses = []
        self_repa_parts = {}
        clean_target = z1_model.detach()
        if active_self_repa_mode in ("pooled", "both"):
            pooled_self_repa, pooled_self_parts = flow_hidden_latent_alignment_loss(
                self_repa_aligner, flow_features["image_tokens"], clean_target,
                prefix="flow_self_repa")
            self_repa_losses.append(pooled_self_repa)
            self_repa_parts.update(pooled_self_parts)
        if active_self_repa_mode in ("token", "both"):
            token_self_repa, token_self_parts = flow_hidden_latent_token_alignment_loss(
                self_repa_aligner, flow_features["image_tokens"], clean_target,
                prefix="flow_self_repa_token")
            self_repa_losses.append(token_self_repa)
            self_repa_parts.update(token_self_parts)
        if not self_repa_losses:
            raise ValueError(
                f"flow self-REPA mode {active_self_repa_mode!r} did not select any losses")
        self_repa_align = torch.stack(self_repa_losses).mean()
        total = total + float(self_repa_w) * self_repa_align
        parts.update(self_repa_parts)
        parts["flow_self_repa_loss"] = self_repa_align.detach()
        parts["flow_self_repa_w"] = torch.tensor(float(self_repa_w), device=z1.device)
        parts["flow_self_repa_mode_id"] = torch.tensor(
            flow_repa_mode_id(active_self_repa_mode), device=z1.device)
        parts["flow_self_repa_components"] = torch.tensor(
            float(len(self_repa_losses)), device=z1.device)
    if sra_w > 0.0:
        if flow_features is None or "image_tokens" not in flow_features:
            raise ValueError("flow SRA alignment requires transformer image-token features")
        gap = float(sra_time_gap)
        if gap <= 0.0 or gap > 1.0:
            raise ValueError("flow_sra_time_gap must be in (0, 1]")
        active_sra_mode = effective_flow_sra_mode(sra_mode)
        t_teacher = t + gap * (1.0 - t)
        zt_teacher = (1.0 - t_teacher) * x0 + t_teacher * z1_model
        with torch.no_grad():
            _teacher_pred, teacher_features = flow_velocity(
                flow,
                zt_teacher, t_teacher, cond_model, return_features=True)
        if teacher_features is None or "image_tokens" not in teacher_features:
            raise ValueError("flow SRA teacher pass requires transformer image-token features")
        teacher_tokens = teacher_features["image_tokens"].detach()
        sra_losses = []
        sra_parts = {}
        if active_sra_mode in ("pooled", "both"):
            pooled_sra, pooled_parts = flow_hidden_self_representation_alignment_loss(
                flow_features["image_tokens"], teacher_tokens,
                prefix="flow_sra", pooled=True)
            sra_losses.append(pooled_sra)
            sra_parts.update(pooled_parts)
        if active_sra_mode in ("token", "both"):
            token_sra, token_parts = flow_hidden_self_representation_alignment_loss(
                flow_features["image_tokens"], teacher_tokens,
                prefix="flow_sra_token", pooled=False)
            sra_losses.append(token_sra)
            sra_parts.update(token_parts)
        if not sra_losses:
            raise ValueError(f"flow SRA mode {active_sra_mode!r} did not select any losses")
        sra_align = torch.stack(sra_losses).mean()
        total = total + float(sra_w) * sra_align
        parts.update(sra_parts)
        parts["flow_sra_loss"] = sra_align.detach()
        parts["flow_sra_w"] = torch.tensor(float(sra_w), device=z1.device)
        parts["flow_sra_mode_id"] = torch.tensor(
            flow_repa_mode_id(active_sra_mode), device=z1.device)
        parts["flow_sra_components"] = torch.tensor(float(len(sra_losses)), device=z1.device)
        parts["flow_sra_time_gap"] = torch.tensor(gap, device=z1.device)
        parts["flow_sra_teacher_time_mean"] = t_teacher.detach().mean()
    return total, parts


def latent_flow_self_distill_losses(flow, teacher_flow, z1, cond, teacher_cond=None,
                                    time_sampling="uniform", time_logit_mean=0.0,
                                    time_logit_std=1.0,
                                    time_mode_scale=DEFAULT_TIME_MODE_SCALE,
                                    time_shift=1.0,
                                    latent_stats=None, time_gap=0.25,
                                    guidance_w=0.0, guidance_cfg_scale=1.0,
                                    guidance_cfg_rescale=0.0):
    """Distill a frozen same-model teacher into cleaner endpoint predictions.

    The student starts from a noisier point on the same rectified-flow path and matches the
    teacher's clean endpoint prediction at a later data-time. This is generic over the condition
    payload and latent source; it only assumes the flow can predict velocity from (z_t, t, cond).
    """
    gap = float(time_gap)
    if gap <= 0.0 or gap > 1.0:
        raise ValueError("flow_distill_time_gap must be in (0, 1]")
    latent_stats = flow_latent_stats(flow) if latent_stats is None else latent_stats
    z1_model = normalize_latent(z1, latent_stats)
    x0 = torch.randn_like(z1_model)
    t = sample_flow_times(z1.shape[0], device=z1.device, mode=time_sampling,
                          logit_mean=time_logit_mean, logit_std=time_logit_std,
                          mode_scale=time_mode_scale, time_shift=time_shift)
    t_teacher = (t + gap * (1.0 - t)).clamp(max=1.0 - 1.0e-4)
    zt = (1.0 - t) * x0 + t * z1_model
    zt_teacher = (1.0 - t_teacher) * x0 + t_teacher * z1_model
    pred = flow_velocity(flow, zt, t, cond)
    endpoint = zt + (1.0 - t) * pred
    with torch.no_grad():
        teacher_input = cond if teacher_cond is None else teacher_cond
        teacher_pred = flow_velocity(teacher_flow, zt_teacher, t_teacher, teacher_input)
        teacher_endpoint = zt_teacher + (1.0 - t_teacher) * teacher_pred
    endpoint_loss = F.mse_loss(endpoint, teacher_endpoint.detach())
    guidance_w = float(guidance_w)
    if guidance_w < 0.0:
        raise ValueError("guidance_w must be non-negative")
    guidance_loss = endpoint_loss * 0.0
    guidance_teacher_endpoint_mse = endpoint_loss * 0.0
    if guidance_w > 0.0:
        with torch.no_grad():
            teacher_uncond = null_condition(teacher_input)
            teacher_batch = int(z1.shape[0])
            t_pair = expand_time_batch(t_teacher, teacher_batch)
            guided_pair = flow_velocity(
                teacher_flow,
                torch.cat([zt_teacher, zt_teacher], dim=0),
                torch.cat([t_pair, t_pair], dim=0),
                concat_conditions(teacher_uncond, teacher_input),
            )
            teacher_uncond_pred, teacher_cond_pred = guided_pair.chunk(2, dim=0)
            teacher_guided_pred = teacher_uncond_pred + float(guidance_cfg_scale) * (
                teacher_cond_pred - teacher_uncond_pred)
            teacher_guided_pred = rescale_guided_velocity(
                teacher_guided_pred, teacher_cond_pred, cfg_rescale=guidance_cfg_rescale)
            teacher_guided_endpoint = zt_teacher + (1.0 - t_teacher) * teacher_guided_pred
        guidance_loss = F.mse_loss(endpoint, teacher_guided_endpoint.detach())
        guidance_teacher_endpoint_mse = F.mse_loss(
            teacher_guided_endpoint.detach(), z1_model).detach()
    total = endpoint_loss + guidance_w * guidance_loss
    parts = {
        "distill_endpoint_mse": total.detach(),
        "distill_base_endpoint_mse": endpoint_loss.detach(),
        "distill_guidance_endpoint_mse": guidance_loss.detach(),
        "distill_guidance_w": torch.tensor(guidance_w, device=z1.device),
        "distill_guidance_cfg_scale": torch.tensor(float(guidance_cfg_scale), device=z1.device),
        "distill_guidance_cfg_rescale": torch.tensor(
            float(guidance_cfg_rescale), device=z1.device),
        "distill_student_endpoint_mse": F.mse_loss(endpoint.detach(), z1_model).detach(),
        "distill_teacher_endpoint_mse": F.mse_loss(
            teacher_endpoint.detach(), z1_model).detach(),
        "distill_guidance_teacher_endpoint_mse": guidance_teacher_endpoint_mse,
        "distill_time_mean": t.detach().mean(),
        "distill_teacher_time_mean": t_teacher.detach().mean(),
        "distill_time_gap": (t_teacher - t).detach().mean(),
        "distill_requested_time_gap": torch.tensor(gap, device=z1.device),
        "time_shift": torch.tensor(float(time_shift), device=z1.device),
    }
    return total, parts


@torch.no_grad()
def flow_endpoint_metrics(flow, z1, cond, time_sampling="uniform", time_logit_mean=0.0,
                          time_logit_std=1.0,
                          time_mode_scale=DEFAULT_TIME_MODE_SCALE,
                          time_shift=1.0, latent_stats=None):
    """Measure whether different times on the same path predict the same clean endpoint."""
    latent_stats = flow_latent_stats(flow) if latent_stats is None else latent_stats
    z1_model = normalize_latent(z1, latent_stats)
    x0 = torch.randn_like(z1_model)
    t1 = sample_flow_times(z1.shape[0], device=z1.device, mode=time_sampling,
                           logit_mean=time_logit_mean, logit_std=time_logit_std,
                           mode_scale=time_mode_scale, time_shift=time_shift)
    t2 = sample_flow_times(z1.shape[0], device=z1.device, mode=time_sampling,
                           logit_mean=time_logit_mean, logit_std=time_logit_std,
                           mode_scale=time_mode_scale, time_shift=time_shift)

    def endpoint_at(t):
        zt = (1.0 - t) * x0 + t * z1_model
        pred = flow_velocity(flow, zt, t, cond)
        return zt + (1.0 - t) * pred

    endpoint1 = endpoint_at(t1)
    endpoint2 = endpoint_at(t2)
    return {
        "latent_endpoint_mse": float(F.mse_loss(endpoint1, z1_model).detach().cpu()),
        "latent_endpoint_consistency_mse": float(
            F.mse_loss(endpoint1, endpoint2).detach().cpu()
        ),
        "latent_endpoint_time_gap": float((t1 - t2).abs().mean().detach().cpu()),
    }


def latent_flow_loss(flow, z1, cond, cond_drop=0.0, time_sampling="uniform",
                     time_logit_mean=0.0, time_logit_std=1.0,
                     time_mode_scale=DEFAULT_TIME_MODE_SCALE, time_shift=1.0,
                     latent_stats=None, flow_loss_weight="none",
                     flow_loss_weight_gamma=5.0, flow_loss_weight_normalize=True):
    loss, _parts = latent_flow_losses(flow, z1, cond, cond_drop=cond_drop,
                                      time_sampling=time_sampling,
                                      time_logit_mean=time_logit_mean,
                                      time_logit_std=time_logit_std,
                                      time_mode_scale=time_mode_scale,
                                      time_shift=time_shift,
                                      latent_stats=latent_stats,
                                      flow_loss_weight=flow_loss_weight,
                                      flow_loss_weight_gamma=flow_loss_weight_gamma,
                                      flow_loss_weight_normalize=flow_loss_weight_normalize)
    return loss


def caption_condition(captions, conditioner, vocab, max_len=64, device=DEV, return_tokens=False):
    if conditioner is None or vocab is None:
        raise ValueError("caption conditioning requires conditioner and caption vocab")
    ids = caption_ids(captions, vocab, max_len=max_len, device=device)
    return conditioner(ids, return_tokens=text_condition_return_tokens(
        conditioner, return_tokens=return_tokens))


def caption_condition_ids(ids, conditioner, device=DEV, return_tokens=False):
    if conditioner is None:
        raise ValueError("caption id conditioning requires conditioner")
    return conditioner(ids.to(device=device), return_tokens=text_condition_return_tokens(
        conditioner, return_tokens=return_tokens))


def parse_sample_prompts(raw):
    if not raw:
        return ()
    if isinstance(raw, (tuple, list)):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    text = str(raw).strip()
    if not text:
        return ()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, str):
            return (parsed.strip(),) if parsed.strip() else ()
        if isinstance(parsed, (tuple, list)):
            return tuple(str(x).strip() for x in parsed if str(x).strip())
    except json.JSONDecodeError:
        pass
    return tuple(x.strip() for x in text.replace("\n", ";").split(";") if x.strip())


def expand_negative_prompts(raw, n):
    prompts = parse_sample_prompts(raw)
    if not prompts:
        return ()
    n = int(n)
    if len(prompts) == 1:
        return tuple(prompts[0] for _ in range(n))
    if len(prompts) == n:
        return prompts
    raise ValueError("negative prompts must be empty, one prompt, or match sample prompts")


@torch.no_grad()
def prompt_embedding_condition(prompts, conditioner, embed_backend="stats", embed_model="",
                               embed_device=None, embed_dtype="auto", embed_normalize=True,
                               embed_stats_dim=0, trust_remote_code=False, device=DEV,
                               return_tokens=False, return_embedding=False,
                               text_sequence_model="", text_sequence_max_length=0):
    if conditioner is None:
        raise ValueError("prompt embedding conditioning requires a loaded conditioner")
    input_dim = int(getattr(conditioner, "input_dim", 0) or 0)
    if input_dim <= 0:
        raise ValueError("prompt embedding conditioning requires a PrecomputedTextConditioner")
    stats_dim = int(embed_stats_dim or input_dim)
    from .image_embed import make_embedder
    text_mode = "both" if return_tokens else "pooled"
    embedder = make_embedder(
        backend=embed_backend, model=embed_model, device=embed_device or device,
        dtype=embed_dtype, normalize=embed_normalize, stats_dim=stats_dim,
        text_mode=text_mode, trust_remote_code=trust_remote_code,
        text_sequence_model=text_sequence_model,
        text_sequence_max_length=text_sequence_max_length)
    records = [ImageTextRecord(path="", caption=str(prompt), split="sample")
               for prompt in prompts]
    encoded = embedder.encode_texts(records)
    if isinstance(encoded, dict):
        feature_embeddings = encoded.get("pooled")
        cond_embeddings = (
            encoded.get("tokens") if return_tokens and encoded.get("tokens") is not None
            else feature_embeddings
        )
    else:
        feature_embeddings = encoded if torch.is_tensor(encoded) else None
        cond_embeddings = encoded
    if isinstance(cond_embeddings, list):
        max_len = max((int(row.shape[0]) for row in cond_embeddings), default=1)
        cond_tensor = torch.zeros(
            (len(cond_embeddings), max(1, max_len), input_dim), dtype=torch.float32)
        for i, row in enumerate(cond_embeddings):
            row = row.detach().float()
            if int(row.shape[-1]) != input_dim:
                raise ValueError(
                    f"prompt embedding dim {int(row.shape[-1])} does not match "
                    f"checkpoint text_embedding_in_dim {input_dim}"
                )
            cond_tensor[i, :int(row.shape[0])] = row.cpu()
        cond_embeddings = cond_tensor
    cond_embeddings = cond_embeddings.to(device=device)
    if int(cond_embeddings.shape[-1]) != input_dim:
        raise ValueError(
            f"prompt embedding dim {int(cond_embeddings.shape[-1])} does not match "
            f"checkpoint text_embedding_in_dim {input_dim}"
        )
    pooled_dim = int(getattr(conditioner, "pooled_input_dim", 0) or 0)
    if (pooled_dim > 0 and return_tokens and torch.is_tensor(feature_embeddings)
            and cond_embeddings.ndim == 3):
        pooled = feature_embeddings.detach().float()
        if int(pooled.shape[-1]) != pooled_dim:
            raise ValueError(
                f"prompt pooled embedding dim {int(pooled.shape[-1])} does not match "
                f"checkpoint text_pooled_embedding_in_dim {pooled_dim}"
            )
        cond_payload = {
            "sequence": cond_embeddings,
            "pooled": pooled.to(device=device),
            "pooled_mask": torch.zeros(
                (int(cond_embeddings.shape[0]),), dtype=torch.bool, device=device),
        }
    else:
        cond_payload = cond_embeddings
    cond = conditioner(cond_payload, return_tokens=text_condition_return_tokens(
        conditioner, return_tokens=return_tokens))
    if return_embedding:
        if torch.is_tensor(feature_embeddings):
            feature_embeddings = feature_embeddings.to(device=device)
        elif isinstance(feature_embeddings, list):
            pooled = []
            for row in feature_embeddings:
                row = row.detach().float()
                pooled.append(row.mean(dim=0))
            feature_embeddings = torch.stack(pooled, dim=0).to(device=device)
        else:
            feature_embeddings = condition_vector(cond)
        return cond, feature_embeddings
    return cond


@torch.no_grad()
def text_prompt_condition(prompts, conditioner, prompt_vocab=None, caption_max_len=64,
                          source="tokens", device=DEV, return_tokens=False,
                          embed_backend="stats", embed_model="", embed_device=None,
                          embed_dtype="auto", embed_normalize=True, embed_stats_dim=0,
                          trust_remote_code=False, return_embedding=False,
                          text_sequence_model="", text_sequence_max_length=0):
    source = str(source or "tokens")
    if source == "embedding":
        return prompt_embedding_condition(
            prompts, conditioner, embed_backend=embed_backend, embed_model=embed_model,
            embed_device=embed_device, embed_dtype=embed_dtype,
            embed_normalize=embed_normalize, embed_stats_dim=embed_stats_dim,
            trust_remote_code=trust_remote_code, device=device, return_tokens=return_tokens,
            return_embedding=return_embedding, text_sequence_model=text_sequence_model,
            text_sequence_max_length=text_sequence_max_length)
    cond = caption_condition(
        prompts, conditioner, prompt_vocab, max_len=caption_max_len, device=device,
        return_tokens=return_tokens)
    if return_embedding:
        return cond, None
    return cond


def infer_text_embedding_dim(records):
    seq_dims = {
        len(rec.text_embedding_sequence[0])
        for rec in records
        if rec.text_embedding_sequence is not None and len(rec.text_embedding_sequence) > 0
    }
    if seq_dims:
        if len(seq_dims) != 1:
            raise ValueError(
                f"manifest text embedding sequences have mixed dimensions: {sorted(seq_dims)}")
        dim = next(iter(seq_dims))
        missing = sum(
            1 for rec in records
            if rec.text_embedding_sequence is None and rec.text_embedding is None)
        fallback_bad = sum(
            1 for rec in records
            if rec.text_embedding_sequence is None
            and rec.text_embedding is not None
            and len(rec.text_embedding) != dim)
        if missing:
            raise ValueError(f"{missing} manifest records are missing text embeddings")
        if fallback_bad:
            raise ValueError(
                f"{fallback_bad} manifest records have pooled text fallback dims "
                f"that do not match sequence dim {dim}")
        return int(dim)
    dims = {len(rec.text_embedding) for rec in records if rec.text_embedding is not None}
    if not dims:
        return 0
    if len(dims) != 1:
        raise ValueError(f"manifest text embeddings have mixed dimensions: {sorted(dims)}")
    dim = next(iter(dims))
    missing = sum(
        1 for rec in records
        if rec.text_embedding is None and rec.text_embedding_sequence is None)
    if missing:
        raise ValueError(f"{missing} manifest records are missing text embeddings")
    return int(dim)


def infer_text_pooled_embedding_dim(records):
    dims = {len(rec.text_embedding) for rec in records if rec.text_embedding is not None}
    if not dims:
        return 0
    if len(dims) != 1:
        raise ValueError(f"manifest pooled text embeddings have mixed dimensions: {sorted(dims)}")
    return int(next(iter(dims)))


def infer_image_embedding_dim(records):
    dims = {len(rec.image_embedding) for rec in records if rec.image_embedding is not None}
    dims.update({
        len(rec.image_embedding_sequence[0])
        for rec in records
        if rec.image_embedding_sequence is not None and len(rec.image_embedding_sequence) > 0
    })
    if not dims:
        return 0
    if len(dims) != 1:
        raise ValueError(f"manifest image embeddings have mixed dimensions: {sorted(dims)}")
    dim = next(iter(dims))
    missing = sum(
        1 for rec in records
        if rec.image_embedding is None and rec.image_embedding_sequence is None)
    if missing:
        raise ValueError(f"{missing} manifest records are missing image embeddings")
    return int(dim)


def records_have_image_embedding_sequences(records):
    return any(rec.image_embedding_sequence is not None for rec in records)


def image_embedding_sequence_max_len(records):
    return max(
        (len(rec.image_embedding_sequence)
         for rec in records if rec.image_embedding_sequence is not None),
        default=0)


def resolve_caption_cond_source(source, records):
    source = str(source or "tokens")
    if source not in ("tokens", "embedding", "auto"):
        raise ValueError(f"unknown caption condition source {source!r}")
    if source == "tokens":
        return "tokens", 0
    try:
        dim = infer_text_embedding_dim(records)
    except ValueError:
        if source == "auto":
            return "tokens", 0
        raise
    if dim <= 0:
        if source == "embedding":
            raise ValueError("caption condition source 'embedding' requires text_embedding rows")
        return "tokens", 0
    return "embedding", dim


def record_text_embedding_tensor(records, device=DEV):
    dim = infer_text_embedding_dim(records)
    if dim <= 0:
        raise ValueError("records do not have text embeddings")
    has_sequence = any(rec.text_embedding_sequence is not None for rec in records)
    if not has_sequence:
        return torch.tensor([rec.text_embedding for rec in records], dtype=torch.float32,
                            device=device)
    rows = []
    max_len = 1
    for rec in records:
        if rec.text_embedding_sequence is not None:
            row = [list(x) for x in rec.text_embedding_sequence]
        else:
            row = [list(rec.text_embedding)]
        rows.append(row)
        max_len = max(max_len, len(row))
    out = torch.zeros((len(rows), max_len, dim), dtype=torch.float32, device=device)
    for i, row in enumerate(rows):
        out[i, :len(row)] = torch.tensor(row, dtype=torch.float32, device=device)
    return out


def record_text_pooled_embedding_tensor(records, device=DEV):
    dim = infer_text_pooled_embedding_dim(records)
    if dim <= 0:
        return None, None
    out = torch.zeros((len(records), dim), dtype=torch.float32, device=device)
    mask = torch.ones((len(records),), dtype=torch.bool, device=device)
    for i, rec in enumerate(records):
        if rec.text_embedding is None:
            continue
        out[i] = torch.tensor(rec.text_embedding, dtype=torch.float32, device=device)
        mask[i] = False
    return out, mask


def record_text_condition_payload(records, device=DEV):
    primary = record_text_embedding_tensor(records, device=device)
    if primary.ndim != 3:
        return primary
    pooled, pooled_mask = record_text_pooled_embedding_tensor(records, device=device)
    if pooled is None:
        return primary
    return {"sequence": primary, "pooled": pooled, "pooled_mask": pooled_mask}


def record_image_embedding_tensor(records, device=DEV):
    dim = infer_image_embedding_dim(records)
    if dim <= 0:
        raise ValueError("records do not have image embeddings")
    rows = []
    for rec in records:
        if rec.image_embedding is not None:
            rows.append(list(rec.image_embedding))
        elif rec.image_embedding_sequence is not None:
            seq = torch.tensor(rec.image_embedding_sequence, dtype=torch.float32)
            rows.append(seq.mean(dim=0).tolist())
        else:
            raise ValueError("record is missing image embedding")
    return torch.tensor(rows, dtype=torch.float32, device=device)


def record_image_embedding_sequence_tensor(records, device=DEV, max_len=0):
    dim = infer_image_embedding_dim(records)
    if dim <= 0:
        raise ValueError("records do not have image embeddings")
    rows = []
    max_len = int(max_len or 0)
    for rec in records:
        if rec.image_embedding_sequence is not None:
            row = [list(x) for x in rec.image_embedding_sequence]
        elif rec.image_embedding is not None:
            row = [list(rec.image_embedding)]
        else:
            raise ValueError("record is missing image embedding")
        rows.append(row)
        max_len = max(max_len, len(row))
    out = torch.zeros((len(rows), max(1, max_len), dim), dtype=torch.float32, device=device)
    for i, row in enumerate(rows):
        out[i, :len(row)] = torch.tensor(row, dtype=torch.float32, device=device)
    return out


def caption_record_condition(captions, records, conditioner, vocab, source="tokens",
                             max_len=64, device=DEV, return_tokens=False):
    if source == "embedding":
        embs = record_text_condition_payload(records, device=device)
        return conditioner(embs, return_tokens=text_condition_return_tokens(
            conditioner, return_tokens=return_tokens))
    return caption_condition(captions, conditioner, vocab, max_len=max_len, device=device,
                             return_tokens=return_tokens)


def cached_caption_condition(cache, idx, conditioner, source="tokens", device=DEV,
                             return_tokens=False):
    if "latents" not in cache and "shards" not in cache:
        return cached_caption_payload_condition(
            cache, conditioner, source=source, device=device, return_tokens=return_tokens)
    if source == "embedding":
        embs = cache["text_embeddings"][idx].to(device=device)
        pooled = cache.get("text_pooled_embeddings")
        pooled_masks = cache.get("text_pooled_masks")
        if pooled is not None:
            embs = {
                "sequence": embs,
                "pooled": pooled[idx].to(device=device),
                "pooled_mask": (
                    pooled_masks[idx].to(device=device) if pooled_masks is not None else None
                ),
            }
        return conditioner(embs, return_tokens=text_condition_return_tokens(
            conditioner, return_tokens=return_tokens))
    ids = cache["caption_ids"][idx]
    return caption_condition_ids(ids, conditioner, device=device, return_tokens=return_tokens)


def cached_caption_payload_condition(payload, conditioner, source="tokens", device=DEV,
                                     return_tokens=False):
    if source == "embedding":
        embs = payload["text_embeddings"].to(device=device)
        pooled = payload.get("text_pooled_embeddings")
        if pooled is not None:
            pooled_masks = payload.get("text_pooled_masks")
            embs = {
                "sequence": embs,
                "pooled": pooled.to(device=device),
                "pooled_mask": (
                    pooled_masks.to(device=device) if pooled_masks is not None else None
                ),
            }
        return conditioner(embs, return_tokens=text_condition_return_tokens(
            conditioner, return_tokens=return_tokens))
    ids = payload["caption_ids"]
    return caption_condition_ids(ids, conditioner, device=device, return_tokens=return_tokens)


@torch.no_grad()
def rescale_guided_velocity(v_guided, v_cond, cfg_rescale=0.0, eps=1.0e-6):
    cfg_rescale = float(cfg_rescale)
    if cfg_rescale <= 0.0:
        return v_guided
    if cfg_rescale > 1.0:
        raise ValueError("cfg_rescale must be in [0, 1]")
    reduce_dims = tuple(range(1, v_guided.ndim))
    guided_std = v_guided.float().std(dim=reduce_dims, keepdim=True, unbiased=False)
    cond_std = v_cond.float().std(dim=reduce_dims, keepdim=True, unbiased=False)
    v_rescaled = v_guided * (cond_std / guided_std.clamp_min(float(eps))).to(v_guided.dtype)
    return cfg_rescale * v_rescaled + (1.0 - cfg_rescale) * v_guided


def sampler_tensor_stats(x):
    raw = x.detach().float()
    finite = torch.isfinite(raw)
    finite_frac = float(finite.float().mean().detach().cpu())
    clean = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    flat = clean.flatten(1) if clean.ndim > 1 else clean.reshape(1, -1)
    rms = flat.square().mean(dim=1).sqrt()
    return {
        "finite_frac": finite_frac,
        "rms_mean": float(rms.mean().detach().cpu()),
        "rms_max": float(rms.max().detach().cpu()),
        "abs_max": float(clean.abs().max().detach().cpu()),
    }


def init_sample_trace(steps, sample_finite_guard=False, sample_velocity_clip=0.0,
                      sample_latent_clip=0.0):
    return {
        "sample_trace_steps": int(steps),
        "sample_trace_solver_steps": 0,
        "sample_trace_solver_rejects": 0,
        "sample_trace_solver_velocity_evals": 0,
        "sample_trace_solver_error_max": 0.0,
        "sample_trace_solver_error_mean": 0.0,
        "sample_trace_solver_error_count": 0,
        "sample_trace_cfg_scale_min": 1.0,
        "sample_trace_cfg_scale_max": 1.0,
        "sample_trace_cfg_scale_mean": 1.0,
        "sample_trace_cfg_scale_count": 0,
        "sample_trace_cfg_guided_evals": 0,
        "sample_finite_guard": bool(sample_finite_guard),
        "sample_velocity_clip": float(sample_velocity_clip),
        "sample_latent_clip": float(sample_latent_clip),
        "sample_trace_min_velocity_finite_frac": 1.0,
        "sample_trace_max_velocity_rms": 0.0,
        "sample_trace_max_velocity_abs": 0.0,
        "sample_trace_velocity_nonfinite_steps": 0,
        "sample_trace_min_latent_finite_frac": 1.0,
        "sample_trace_max_latent_rms": 0.0,
        "sample_trace_max_latent_abs": 0.0,
        "sample_trace_latent_nonfinite_steps": 0,
        "sample_trace_finite_guard_events": 0,
        "sample_trace_velocity_clip_events": 0,
        "sample_trace_latent_clip_events": 0,
        "sample_trace_stabilized": bool(
            sample_finite_guard or sample_velocity_clip > 0.0 or sample_latent_clip > 0.0
        ),
    }


def prefix_sample_trace(trace, prefix):
    out = {}
    for key, value in dict(trace).items():
        if key.startswith("sample_trace_"):
            out[f"{prefix}_trace_{key[len('sample_trace_'):]}"] = value
        elif key.startswith("sample_"):
            out[f"{prefix}_{key[len('sample_'):]}"] = value
        else:
            out[f"{prefix}_{key}"] = value
    return out


def merge_sample_traces(left, right):
    if not left:
        return dict(right)
    out = dict(left)
    for key, value in dict(right).items():
        if key.endswith("_min_finite_frac") or key.endswith("_final_latent_finite_frac"):
            out[key] = min(float(out.get(key, 1.0)), float(value))
        elif (key.endswith("_max_rms") or key.endswith("_max_abs")
              or key.endswith("_final_latent_rms") or key.endswith("_final_latent_abs")):
            out[key] = max(float(out.get(key, 0.0)), float(value))
        elif key.endswith("_nonfinite_steps") or key.endswith("_events"):
            out[key] = int(out.get(key, 0)) + int(value)
        elif key.endswith("_solver_steps") or key.endswith("_solver_rejects"):
            out[key] = int(out.get(key, 0)) + int(value)
        elif key.endswith("_solver_velocity_evals") or key.endswith("_solver_error_count"):
            out[key] = int(out.get(key, 0)) + int(value)
        elif key.endswith("_cfg_scale_count") or key.endswith("_cfg_guided_evals"):
            out[key] = int(out.get(key, 0)) + int(value)
        elif key.endswith("_cfg_scale_min"):
            out[key] = min(float(out.get(key, 1.0)), float(value))
        elif key.endswith("_cfg_scale_max"):
            out[key] = max(float(out.get(key, 1.0)), float(value))
        elif key.endswith("_cfg_scale_mean"):
            left_count = int(out.get("sample_trace_cfg_scale_count", 0))
            right_count = int(dict(right).get("sample_trace_cfg_scale_count", 0))
            total = left_count + right_count
            if total > 0:
                out[key] = (
                    float(out.get(key, 1.0)) * left_count
                    + float(value) * right_count
                ) / float(total)
            else:
                out[key] = float(value)
        elif key.endswith("_solver_error_max"):
            out[key] = max(float(out.get(key, 0.0)), float(value))
        elif key.endswith("_solver_error_mean"):
            left_count = int(out.get("sample_trace_solver_error_count", 0))
            right_count = int(dict(right).get("sample_trace_solver_error_count", 0))
            total = left_count + right_count
            if total > 0:
                out[key] = (
                    float(out.get(key, 0.0)) * left_count
                    + float(value) * right_count
                ) / float(total)
            else:
                out[key] = float(value)
        else:
            out[key] = value
    return out


def update_sample_trace(trace, role, x):
    stats = sampler_tensor_stats(x)
    finite_key = f"sample_trace_min_{role}_finite_frac"
    rms_key = f"sample_trace_max_{role}_rms"
    abs_key = f"sample_trace_max_{role}_abs"
    nonfinite_key = f"sample_trace_{role}_nonfinite_steps"
    trace[finite_key] = min(float(trace.get(finite_key, 1.0)), stats["finite_frac"])
    trace[rms_key] = max(float(trace.get(rms_key, 0.0)), stats["rms_max"])
    trace[abs_key] = max(float(trace.get(abs_key, 0.0)), stats["abs_max"])
    if stats["finite_frac"] < 1.0:
        trace[nonfinite_key] = int(trace.get(nonfinite_key, 0)) + 1
    return stats


def finite_guard_tensor(x):
    nonfinite = ~torch.isfinite(x)
    events = int(nonfinite.flatten(1).any(dim=1).sum().detach().cpu()) if x.ndim > 1 else (
        1 if bool(nonfinite.any().detach().cpu()) else 0
    )
    if events:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, events


def clip_tensor_rms(x, max_rms, eps=1.0e-8):
    max_rms = float(max_rms)
    if max_rms <= 0.0:
        return x, 0
    flat = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).flatten(1)
    rms = flat.square().mean(dim=1).sqrt()
    scale = (max_rms / rms.clamp_min(float(eps))).clamp(max=1.0)
    events = int(scale.lt(1.0).sum().detach().cpu())
    if events:
        view_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        x = x * scale.to(dtype=x.dtype, device=x.device).reshape(view_shape)
    return x, events


def clip_tensor_abs(x, max_abs):
    max_abs = float(max_abs)
    if max_abs <= 0.0:
        return x, 0
    clean = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    events = int(clean.abs().flatten(1).amax(dim=1).gt(max_abs).sum().detach().cpu())
    if events:
        x = x.clamp(-max_abs, max_abs)
    return x, events


def dynamic_threshold_tensor(x, percentile=0.0, max_abs=1.0, eps=1.0e-8):
    """Imagen-style per-sample dynamic thresholding for decoded image tensors."""
    percentile = float(percentile)
    max_abs = float(max_abs)
    if percentile <= 0.0:
        return x, 0
    if percentile > 1.0:
        raise ValueError("dynamic threshold percentile must be in (0, 1]")
    if max_abs <= 0.0:
        raise ValueError("dynamic threshold max_abs must be positive")
    if x.ndim < 2:
        flat = x.detach().float().abs().reshape(1, -1)
        view_shape = (1,) * x.ndim
    else:
        flat = x.detach().float().abs().flatten(1)
        view_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    thresh = torch.quantile(flat, percentile, dim=1).clamp_min(float(max_abs))
    events = int(thresh.gt(float(max_abs) + float(eps)).sum().detach().cpu())
    if events:
        thresh_view = thresh.to(device=x.device, dtype=x.dtype).reshape(view_shape)
        x = x.clamp(-thresh_view, thresh_view) * (float(max_abs) / thresh_view)
    return x, events


def expand_time_batch(t, batch):
    batch = int(batch)
    if t.ndim == 0:
        return t.reshape(1).expand(batch)
    if int(t.shape[0]) == batch:
        return t
    if int(t.shape[0]) == 1:
        return t.expand((batch,) + tuple(t.shape[1:]))
    raise ValueError("time batch must match latent batch")


@torch.no_grad()
def guided_velocity(flow, z, t, cond, cfg_scale=1.0, cfg_rescale=0.0, cfg_uncond=None,
                    cfg_mode="standard"):
    """Classifier-free guidance in latent velocity space."""
    cfg_mode = str(cfg_mode)
    if cfg_mode not in CFG_MODES:
        raise ValueError(f"unknown cfg_mode {cfg_mode!r}")
    if cfg_scale == 1.0 and cfg_mode == "standard":
        return flow_velocity(flow, z, t, cond)
    uncond = null_condition(cond) if cfg_uncond is None else cfg_uncond
    batch = int(z.shape[0])
    if condition_batch(cond) != batch or condition_batch(uncond) != batch:
        raise ValueError("CFG condition batch must match latent batch")
    t = expand_time_batch(t, batch)
    v = flow_velocity(
        flow,
        torch.cat([z, z], dim=0),
        torch.cat([t, t], dim=0),
        concat_conditions(uncond, cond),
    )
    v_uncond, v_cond = v.chunk(2, dim=0)
    guided = v_uncond + cfg_scale * (v_cond - v_uncond)
    guided = rescale_guided_velocity(guided, v_cond, cfg_rescale=cfg_rescale)
    if cfg_mode == "standard":
        return guided
    t_mix = t
    if t_mix.ndim == 1:
        t_mix = t_mix[:, None]
    while t_mix.ndim < guided.ndim:
        t_mix = t_mix.unsqueeze(-1)
    t_mix = t_mix.to(dtype=guided.dtype, device=guided.device).clamp(0.0, 1.0)
    return (1.0 - t_mix) * guided + t_mix * v_uncond


@torch.no_grad()
def sample_latents(flow, cond, latent_shape=(16, 8, 8), steps=16, device=DEV, seed=0,
                   cfg_scale=1.0, cfg_rescale=0.0, ae=None,
                   sample_method="euler",
                   cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                   cfg_schedule="constant",
                   sample_time_shift=1.0, sample_schedule="linear", cfg_uncond=None,
                   sample_churn=0.0, sample_churn_interval=DEFAULT_CHURN_INTERVAL,
                   text_guidance_w=0.0, text_guidance_aligner=None,
                   text_guidance_cond=None,
                   text_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                   feature_guidance_w=0.0, feature_guidance_aligner=None,
                   feature_guidance_features=None,
                   feature_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                   quality_guidance_w=0.0, quality_guidance_scorer=None,
                   quality_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                   sample_finite_guard=False, sample_velocity_clip=0.0,
                   sample_latent_clip=0.0, cfg_mode="standard", return_trace=False):
    batch = condition_batch(cond)
    if cfg_uncond is not None and condition_batch(cfg_uncond) != batch:
        raise ValueError("cfg_uncond batch must match cond batch")
    if text_guidance_cond is not None and condition_batch(text_guidance_cond) != batch:
        raise ValueError("text_guidance_cond batch must match cond batch")
    latent_stats = flow_latent_stats(flow)
    z = _seeded_randn((batch,) + tuple(latent_shape), device=device, seed=seed)
    flow.eval()
    if ae is not None:
        ae.eval()
    if text_guidance_w < 0.0:
        raise ValueError("text_guidance_w must be non-negative")
    if text_guidance_w > 0.0 and text_guidance_aligner is None:
        raise ValueError("text alignment guidance requires a checkpoint text_aligner")
    if text_guidance_cond is None:
        text_guidance_cond = cond
    if feature_guidance_w < 0.0:
        raise ValueError("feature_guidance_w must be non-negative")
    if feature_guidance_w > 0.0:
        if feature_guidance_aligner is None:
            raise ValueError("image feature guidance requires a checkpoint image_feature_aligner")
        if feature_guidance_features is None:
            raise ValueError("image feature guidance requires prompt/image feature targets")
        feature_guidance_features = pool_embedding_sequence(
            feature_guidance_features).to(device=device)
        if int(feature_guidance_features.shape[0]) != batch:
            raise ValueError("feature guidance feature batch must match cond batch")
        feature_dim = int(getattr(feature_guidance_aligner, "feature_dim", 0) or 0)
        if int(feature_guidance_features.shape[-1]) != feature_dim:
            raise ValueError(
                f"feature guidance dim {int(feature_guidance_features.shape[-1])} "
                f"does not match checkpoint image_embedding_in_dim {feature_dim}"
            )
    if quality_guidance_w < 0.0:
        raise ValueError("quality_guidance_w must be non-negative")
    if quality_guidance_w > 0.0 and quality_guidance_scorer is None:
        raise ValueError("quality guidance requires a checkpoint image_quality_scorer")
    if sample_method not in SAMPLE_METHODS:
        raise ValueError(f"unknown sample method {sample_method!r}")
    if sample_schedule not in SAMPLE_SCHEDULES:
        raise ValueError(f"unknown sample schedule {sample_schedule!r}")
    sample_churn = float(sample_churn)
    if sample_churn < 0.0:
        raise ValueError("sample_churn must be non-negative")
    sample_velocity_clip = float(sample_velocity_clip)
    sample_latent_clip = float(sample_latent_clip)
    if sample_velocity_clip < 0.0:
        raise ValueError("sample_velocity_clip must be non-negative")
    if sample_latent_clip < 0.0:
        raise ValueError("sample_latent_clip must be non-negative")
    if cfg_rescale < 0.0 or cfg_rescale > 1.0:
        raise ValueError("cfg_rescale must be in [0, 1]")
    cfg_mode = str(cfg_mode)
    if cfg_mode not in CFG_MODES:
        raise ValueError(f"unknown cfg_mode {cfg_mode!r}")
    cfg_interval = validate_guidance_interval(cfg_interval, name="cfg_interval")
    cfg_schedule = validate_cfg_schedule(cfg_schedule)
    sample_churn_interval = validate_guidance_interval(
        sample_churn_interval, name="sample_churn_interval")
    text_guidance_interval = validate_guidance_interval(
        text_guidance_interval, name="text_guidance_interval")
    feature_guidance_interval = validate_guidance_interval(
        feature_guidance_interval, name="feature_guidance_interval")
    quality_guidance_interval = validate_guidance_interval(
        quality_guidance_interval, name="quality_guidance_interval")
    schedule = flow_time_schedule(
        steps, device=device, shift=sample_time_shift, schedule=sample_schedule)
    trace = init_sample_trace(
        steps, sample_finite_guard=sample_finite_guard,
        sample_velocity_clip=sample_velocity_clip,
        sample_latent_clip=sample_latent_clip)
    trace["sample_trace_cfg_schedule"] = cfg_schedule
    trace["sample_trace_requested_cfg_scale"] = float(cfg_scale)
    update_sample_trace(trace, "latent", z)

    def stabilize_latent(z_in):
        update_sample_trace(trace, "latent", z_in)
        if sample_finite_guard:
            z_in, events = finite_guard_tensor(z_in)
            trace["sample_trace_finite_guard_events"] += int(events)
        if sample_latent_clip > 0.0:
            z_in, events = clip_tensor_abs(z_in, sample_latent_clip)
            trace["sample_trace_latent_clip_events"] += int(events)
        return z_in

    def stabilize_velocity(v_in):
        update_sample_trace(trace, "velocity", v_in)
        if sample_finite_guard:
            v_in, events = finite_guard_tensor(v_in)
            trace["sample_trace_finite_guard_events"] += int(events)
        if sample_velocity_clip > 0.0:
            v_in, events = clip_tensor_rms(v_in, sample_velocity_clip)
            trace["sample_trace_velocity_clip_events"] += int(events)
        return v_in

    def record_solver_error(err_ratio):
        err_ratio = float(err_ratio)
        count = int(trace.get("sample_trace_solver_error_count", 0))
        mean = float(trace.get("sample_trace_solver_error_mean", 0.0))
        trace["sample_trace_solver_error_mean"] = (
            (mean * count + err_ratio) / float(count + 1)
        )
        trace["sample_trace_solver_error_count"] = count + 1
        trace["sample_trace_solver_error_max"] = max(
            float(trace.get("sample_trace_solver_error_max", 0.0)), err_ratio)

    def record_cfg_scale(effective_scale, strength):
        effective_scale = float(effective_scale)
        count = int(trace.get("sample_trace_cfg_scale_count", 0))
        if count == 0:
            trace["sample_trace_cfg_scale_mean"] = effective_scale
            trace["sample_trace_cfg_scale_min"] = effective_scale
            trace["sample_trace_cfg_scale_max"] = effective_scale
        else:
            mean = float(trace.get("sample_trace_cfg_scale_mean", 1.0))
            trace["sample_trace_cfg_scale_mean"] = (
                (mean * count + effective_scale) / float(count + 1)
            )
            trace["sample_trace_cfg_scale_min"] = min(
                float(trace.get("sample_trace_cfg_scale_min", effective_scale)),
                effective_scale)
            trace["sample_trace_cfg_scale_max"] = max(
                float(trace.get("sample_trace_cfg_scale_max", effective_scale)),
                effective_scale)
        trace["sample_trace_cfg_scale_count"] = count + 1
        if float(strength) > 1.0e-8:
            trace["sample_trace_cfg_guided_evals"] = (
                int(trace.get("sample_trace_cfg_guided_evals", 0)) + 1
            )

    for i in range(steps):
        t_scalar = float(schedule[i].detach().cpu())
        t_next_scalar = float(schedule[i + 1].detach().cpu())
        t_mid_scalar = 0.5 * (t_scalar + t_next_scalar)
        dt = t_next_scalar - t_scalar
        if sample_churn > 0.0 and interval_active(t_scalar, sample_churn_interval):
            noise_std = sample_churn * math.sqrt(abs(float(dt))) * max(
                0.0, 1.0 - float(t_next_scalar))
            if noise_std > 0.0:
                z = z + noise_std * _seeded_randn(
                    tuple(z.shape), device=device, seed=int(seed) + 104729 * (i + 1))
                z = stabilize_latent(z)

        def velocity_at(z_in, raw_t):
            t = torch.full((batch, 1, 1, 1), float(raw_t), device=device)
            step_cfg, cfg_strength = effective_cfg_scale_at(
                float(raw_t), cfg_scale=cfg_scale, cfg_interval=cfg_interval,
                cfg_schedule=cfg_schedule)
            step_cfg_mode = cfg_mode if cfg_strength > 1.0e-8 else "standard"
            record_cfg_scale(step_cfg, cfg_strength)
            v = guided_velocity(flow, z_in, t, cond, cfg_scale=step_cfg,
                                cfg_rescale=cfg_rescale, cfg_uncond=cfg_uncond,
                                cfg_mode=step_cfg_mode)
            trace["sample_trace_solver_velocity_evals"] = (
                int(trace.get("sample_trace_solver_velocity_evals", 0)) + 1
            )
            return stabilize_velocity(v)

        if sample_method == "euler":
            v0 = velocity_at(z, t_scalar)
            z = z + dt * v0
            trace["sample_trace_solver_steps"] += 1
        elif sample_method == "heun":
            v0 = velocity_at(z, t_scalar)
            z_pred = z + dt * v0
            v1 = velocity_at(z_pred, t_next_scalar)
            z = z + 0.5 * dt * (v0 + v1)
            trace["sample_trace_solver_steps"] += 1
        elif sample_method == "midpoint":
            v0 = velocity_at(z, t_scalar)
            z_mid = z + 0.5 * dt * v0
            v_mid = velocity_at(z_mid, t_mid_scalar)
            z = z + dt * v_mid
            trace["sample_trace_solver_steps"] += 1
        elif sample_method == "rk4":
            v0 = velocity_at(z, t_scalar)
            z_k2 = z + 0.5 * dt * v0
            v2 = velocity_at(z_k2, t_mid_scalar)
            z_k3 = z + 0.5 * dt * v2
            v3 = velocity_at(z_k3, t_mid_scalar)
            z_k4 = z + dt * v3
            v4 = velocity_at(z_k4, t_next_scalar)
            z = z + (dt / 6.0) * (v0 + 2.0 * v2 + 2.0 * v3 + v4)
            trace["sample_trace_solver_steps"] += 1
        elif sample_method == "adaptive-heun":
            # Embedded Heun/Euler control: accept Heun, use Euler gap as local error.
            direction = 1.0 if dt >= 0.0 else -1.0
            t_cur = float(t_scalar)
            target = float(t_next_scalar)
            h = float(dt)
            # Conservative fixed tolerances; high-res runs can include this method in sweeps.
            rtol, atol = 0.03, 0.002
            min_abs_h = max(1.0e-5, abs(float(dt)) / 256.0)
            rejects_for_interval = 0
            while direction * (target - t_cur) > 1.0e-8:
                remaining = target - t_cur
                if direction * h > direction * remaining:
                    h = remaining
                v_start = velocity_at(z, t_cur)
                z_euler = z + h * v_start
                v_end = velocity_at(z_euler, t_cur + h)
                z_heun = z + 0.5 * h * (v_start + v_end)
                err = (z_heun.float() - z_euler.float()).flatten(1).pow(2).mean(1).sqrt()
                z_rms = z.float().flatten(1).pow(2).mean(1).sqrt()
                next_rms = z_heun.float().flatten(1).pow(2).mean(1).sqrt()
                tol = float(atol) + float(rtol) * torch.maximum(z_rms, next_rms)
                err_ratio = float((err / tol.clamp_min(1.0e-8)).max().detach().cpu())
                accept = (
                    err_ratio <= 1.0 or abs(float(h)) <= min_abs_h
                    or rejects_for_interval >= 16
                )
                if accept:
                    z = stabilize_latent(z_heun)
                    t_cur += h
                    trace["sample_trace_solver_steps"] += 1
                    record_solver_error(err_ratio)
                    if err_ratio <= 1.0e-8:
                        grow = 2.0
                    else:
                        grow = min(2.0, max(0.5, 0.9 * err_ratio ** -0.5))
                    h = direction * min(abs(h) * grow, abs(target - t_cur))
                    if abs(h) <= 0.0:
                        break
                else:
                    rejects_for_interval += 1
                    trace["sample_trace_solver_rejects"] += 1
                    shrink = max(0.25, min(0.5, 0.9 * err_ratio ** -0.5))
                    h = direction * max(min_abs_h, abs(h) * shrink)
        else:
            raise ValueError(f"unknown sample method {sample_method!r}")
        z = stabilize_latent(z)
        if (text_guidance_w > 0.0
                and interval_active(t_scalar, text_guidance_interval)):
            z_raw = denormalize_latent(z, latent_stats)
            z_raw = text_alignment_guidance_step(
                text_guidance_aligner, z_raw, text_guidance_cond,
                weight=text_guidance_w, step_size=dt)
            z = normalize_latent(z_raw, latent_stats)
            z = stabilize_latent(z)
        if (feature_guidance_w > 0.0
                and interval_active(t_scalar, feature_guidance_interval)):
            z_raw = denormalize_latent(z, latent_stats)
            z_raw = image_feature_guidance_step(
                feature_guidance_aligner, z_raw, feature_guidance_features,
                weight=feature_guidance_w, step_size=dt)
            z = normalize_latent(z_raw, latent_stats)
            z = stabilize_latent(z)
        if (quality_guidance_w > 0.0
                and interval_active(t_scalar, quality_guidance_interval)):
            z_raw = denormalize_latent(z, latent_stats)
            z_raw = image_quality_guidance_step(
                quality_guidance_scorer, z_raw,
                weight=quality_guidance_w, step_size=dt)
            z = normalize_latent(z_raw, latent_stats)
            z = stabilize_latent(z)
    final_stats = sampler_tensor_stats(z)
    trace.update({
        "sample_trace_final_latent_finite_frac": final_stats["finite_frac"],
        "sample_trace_final_latent_rms": final_stats["rms_mean"],
        "sample_trace_final_latent_abs": final_stats["abs_max"],
    })
    out = denormalize_latent(z, latent_stats)
    if return_trace:
        return out, trace
    return out


@torch.no_grad()
def sample_images(ae, flow, cond, latent_shape=(16, 8, 8), steps=16, device=DEV, seed=0,
                  cfg_scale=1.0, cfg_rescale=0.0,
                  sample_method="euler",
                  cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                  cfg_schedule="constant",
                  sample_time_shift=1.0, sample_schedule="linear", cfg_uncond=None,
                  sample_churn=0.0, sample_churn_interval=DEFAULT_CHURN_INTERVAL,
                  text_guidance_w=0.0, text_guidance_aligner=None,
                  text_guidance_cond=None,
                  text_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                  feature_guidance_w=0.0, feature_guidance_aligner=None,
                  feature_guidance_features=None,
                  feature_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                  quality_guidance_w=0.0, quality_guidance_scorer=None,
                  quality_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                  sample_finite_guard=False, sample_velocity_clip=0.0,
                  sample_latent_clip=0.0, cfg_mode="standard",
                  sample_pixel_dynamic_threshold_percentile=0.0,
                  sample_pixel_dynamic_threshold_max=1.0,
                  return_trace=False):
    latent_out = sample_latents(
        flow, cond, latent_shape=latent_shape, steps=steps, device=device,
        seed=seed, cfg_scale=cfg_scale, cfg_rescale=cfg_rescale,
        ae=ae, sample_method=sample_method, cfg_interval=cfg_interval,
        cfg_schedule=cfg_schedule,
        sample_time_shift=sample_time_shift,
        sample_schedule=sample_schedule, cfg_uncond=cfg_uncond,
        sample_churn=sample_churn,
        sample_churn_interval=sample_churn_interval,
        text_guidance_w=text_guidance_w,
        text_guidance_aligner=text_guidance_aligner,
        text_guidance_cond=text_guidance_cond,
        text_guidance_interval=text_guidance_interval,
        feature_guidance_w=feature_guidance_w,
        feature_guidance_aligner=feature_guidance_aligner,
        feature_guidance_features=feature_guidance_features,
        feature_guidance_interval=feature_guidance_interval,
        quality_guidance_w=quality_guidance_w,
        quality_guidance_scorer=quality_guidance_scorer,
        quality_guidance_interval=quality_guidance_interval,
        sample_finite_guard=sample_finite_guard,
        sample_velocity_clip=sample_velocity_clip,
        sample_latent_clip=sample_latent_clip,
        cfg_mode=cfg_mode,
        return_trace=return_trace)
    if return_trace:
        z, trace = latent_out
    else:
        z, trace = latent_out, None
    ae.eval()
    sample = ae.decode(z)
    sample, threshold_events = dynamic_threshold_tensor(
        sample, percentile=sample_pixel_dynamic_threshold_percentile,
        max_abs=sample_pixel_dynamic_threshold_max)
    sample = sample.clamp(-1.0, 1.0)
    if return_trace:
        trace["sample_pixel_dynamic_threshold_percentile"] = float(
            sample_pixel_dynamic_threshold_percentile)
        trace["sample_pixel_dynamic_threshold_max"] = float(
            sample_pixel_dynamic_threshold_max)
        trace["sample_trace_pixel_dynamic_threshold_events"] = int(threshold_events)
        return sample, trace
    return sample


def _rgb8_from_samples(samples):
    if samples.ndim != 4 or samples.shape[1] != 3:
        raise ValueError(f"expected BCHW RGB samples, got shape {tuple(samples.shape)}")
    arr = torch.nan_to_num(
        samples.detach().cpu().float(), nan=-1.0, posinf=1.0, neginf=-1.0
    ).clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).round().to(torch.uint8)
    return arr.permute(0, 2, 3, 1).contiguous().numpy()


def sample_health_metrics(samples, prefix="sample", eps=1.0e-8):
    if samples.ndim != 4 or samples.shape[1] != 3:
        raise ValueError(f"expected BCHW RGB samples, got shape {tuple(samples.shape)}")
    raw = samples.detach().float()
    finite = torch.isfinite(raw)
    image_nonfinite = (~finite).flatten(1).any(dim=1)
    clean = torch.nan_to_num(raw, nan=-1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    rgb = (clean + 1.0) * 0.5
    lum = (
        0.2126 * rgb[:, 0]
        + 0.7152 * rgb[:, 1]
        + 0.0722 * rgb[:, 2]
    )
    rgb_flat = rgb.flatten(1)
    lum_flat = lum.flatten(1)
    rgb_std = rgb_flat.std(dim=1, unbiased=False)
    lum_std = lum_flat.std(dim=1, unbiased=False)
    dynamic_range = lum_flat.max(dim=1).values - lum_flat.min(dim=1).values
    low_frac = lum_flat.le(0.01).float().mean(dim=1)
    high_frac = lum_flat.ge(0.99).float().mean(dim=1)
    collapsed = image_nonfinite | dynamic_range.lt(0.05) | lum_std.lt(0.01)
    health_score = (
        lum_std.mean()
        * dynamic_range.mean().clamp_min(float(eps))
        * (1.0 - collapsed.float().mean())
    )
    return {
        f"{prefix}_finite_frac": float(finite.float().mean().detach().cpu()),
        f"{prefix}_nonfinite_frac": float((~finite).float().mean().detach().cpu()),
        f"{prefix}_luminance_mean": float(lum_flat.mean().detach().cpu()),
        f"{prefix}_luminance_std": float(lum_std.mean().detach().cpu()),
        f"{prefix}_rgb_std": float(rgb_std.mean().detach().cpu()),
        f"{prefix}_dynamic_range": float(dynamic_range.mean().detach().cpu()),
        f"{prefix}_low_luminance_frac": float(low_frac.mean().detach().cpu()),
        f"{prefix}_high_luminance_frac": float(high_frac.mean().detach().cpu()),
        f"{prefix}_collapsed_frac": float(collapsed.float().mean().detach().cpu()),
        f"{prefix}_health_score": float(health_score.detach().cpu()),
    }


def sample_candidate_health_scores(samples, eps=1.0e-8):
    if samples.ndim != 4 or samples.shape[1] != 3:
        raise ValueError(f"expected BCHW RGB samples, got shape {tuple(samples.shape)}")
    raw = samples.detach().float()
    finite = torch.isfinite(raw)
    image_nonfinite = (~finite).flatten(1).any(dim=1)
    finite_frac = finite.float().flatten(1).mean(dim=1)
    clean = torch.nan_to_num(raw, nan=-1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    rgb = (clean + 1.0) * 0.5
    lum = (
        0.2126 * rgb[:, 0]
        + 0.7152 * rgb[:, 1]
        + 0.0722 * rgb[:, 2]
    )
    rgb_std = rgb.flatten(1).std(dim=1, unbiased=False)
    lum_flat = lum.flatten(1)
    lum_std = lum_flat.std(dim=1, unbiased=False)
    dynamic_range = lum_flat.max(dim=1).values - lum_flat.min(dim=1).values
    collapsed = image_nonfinite | dynamic_range.lt(0.05) | lum_std.lt(0.01)
    return (
        finite_frac
        + lum_std * dynamic_range.clamp_min(float(eps))
        + 0.05 * rgb_std
        - collapsed.float()
        - image_nonfinite.float()
    )


def write_ppm_grid(samples, path, rows, cols, pad=2, bg=32):
    """Write a binary PPM grid without image-library dependencies."""
    arr = _rgb8_from_samples(samples)
    n, h, w, c = arr.shape
    if c != 3:
        raise ValueError(f"expected RGB samples, got {c} channels")
    rows, cols = int(rows), int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("grid rows/cols must be positive")
    if rows * cols < n:
        raise ValueError(f"grid {rows}x{cols} cannot fit {n} samples")
    pad = max(0, int(pad))
    out_h = rows * h + max(0, rows - 1) * pad
    out_w = cols * w + max(0, cols - 1) * pad
    grid = np.full((out_h, out_w, 3), int(bg), dtype=np.uint8)
    for idx in range(n):
        r = idx // cols
        col = idx % cols
        y0 = r * (h + pad)
        x0 = col * (w + pad)
        grid[y0:y0 + h, x0:x0 + w] = arr[idx]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"P6\n{out_w} {out_h}\n255\n".encode("ascii"))
        f.write(grid.tobytes())
    return {
        "sample_grid": path,
        "sample_grid_rows": rows,
        "sample_grid_cols": cols,
        "sample_grid_n": int(n),
        "sample_grid_tile_h": int(h),
        "sample_grid_tile_w": int(w),
        "sample_grid_pad": pad,
        "sample_grid_format": "ppm",
    }


def _sample_manifest_image_dir(manifest_path, image_dir=""):
    if image_dir:
        return image_dir
    base = os.path.splitext(os.path.basename(manifest_path))[0] or "samples"
    return os.path.join(os.path.dirname(manifest_path) or ".", f"{base}_images")


def write_sample_manifest(samples, manifest_path, captions, image_dir="", metadata=None,
                          row_metadata=None, prefix="sample", split="generated"):
    """Write generated samples as individual PPM files plus a JSONL image manifest."""
    captions = [str(c) for c in captions]
    arr = _rgb8_from_samples(samples)
    n, h, w, c = arr.shape
    if c != 3:
        raise ValueError(f"expected RGB samples, got {c} channels")
    if len(captions) != n:
        raise ValueError(f"expected {n} captions/prompts, got {len(captions)}")
    manifest_path = str(manifest_path)
    out_dir = _sample_manifest_image_dir(manifest_path, image_dir=image_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    rows = []
    metadata = dict(metadata or {})
    if row_metadata is None:
        row_metadata = [{} for _ in range(n)]
    if len(row_metadata) != n:
        raise ValueError(f"expected {n} per-row metadata records, got {len(row_metadata)}")
    for idx, (img, caption) in enumerate(zip(arr, captions)):
        name = f"{prefix}_{idx:05d}.ppm"
        img_path = os.path.join(out_dir, name)
        with open(img_path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(img.tobytes())
        rel = os.path.relpath(os.path.abspath(img_path), manifest_dir)
        row = {
            "image": rel,
            "caption": caption,
            "prompt": caption,
            "split": split,
            "source": "image_latent_generated",
            "width": int(w),
            "height": int(h),
            "sample_index": int(idx),
        }
        row.update(metadata)
        row.update(dict(row_metadata[idx] or {}))
        rows.append(row)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "sample_manifest": manifest_path,
        "sample_manifest_image_dir": out_dir,
        "sample_manifest_records": int(n),
        "sample_manifest_format": "jsonl",
        "sample_manifest_image_format": "ppm",
        "sample_manifest_split": split,
    }


@torch.no_grad()
def select_prompt_candidates(ae, samples, cond, prompts, candidates_per_prompt=1,
                             text_aligner=None, image_feature_aligner=None,
                             prompt_features=None, quality_scorer=None):
    candidates_per_prompt = int(candidates_per_prompt)
    if candidates_per_prompt <= 0:
        raise ValueError("candidates_per_prompt must be positive")
    n = len(prompts)
    expected = n * candidates_per_prompt
    if int(samples.shape[0]) != expected:
        raise ValueError(
            f"expected {expected} prompt candidates, got {int(samples.shape[0])}"
        )
    meta = {
        "sample_grid_candidates_per_prompt": candidates_per_prompt,
        "sample_grid_selection_scorer": "none",
        "sample_grid_selected_candidate_indices": [0 for _ in range(n)],
    }
    if candidates_per_prompt == 1:
        return samples, meta

    def normalize_candidate_scores(raw_scores):
        raw_scores = raw_scores.reshape(n, candidates_per_prompt).float()
        centered = raw_scores - raw_scores.mean(dim=1, keepdim=True)
        scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        return centered / scale

    scorers = []
    z = None
    if text_aligner is not None:
        text_aligner.eval()
        z = ae.encode(samples)
        img_emb, txt_emb = text_aligner(z, cond)
        scorers.append(("text_aligner", (img_emb.float() * txt_emb.float()).sum(dim=-1)))
    if image_feature_aligner is not None and prompt_features is not None:
        feature_dim = int(getattr(image_feature_aligner, "feature_dim", 0) or 0)
        feature_rows = pool_embedding_sequence(prompt_features).to(device=samples.device)
        if int(feature_rows.shape[-1]) == feature_dim:
            image_feature_aligner.eval()
            if z is None:
                z = ae.encode(samples)
            target_features = feature_rows.repeat_interleave(candidates_per_prompt, dim=0)
            img_feat, prompt_feat = image_feature_aligner(z, target_features)
            scorers.append((
                "image_feature_aligner",
                (img_feat.float() * prompt_feat.float()).sum(dim=-1),
            ))
        else:
            meta["sample_grid_feature_selection_skipped"] = (
                f"prompt feature dim {int(feature_rows.shape[-1])} != "
                f"image feature dim {feature_dim}"
            )
    if quality_scorer is not None:
        quality_scorer.eval()
        if z is None:
            z = ae.encode(samples)
        scorers.append(("quality_scorer", quality_scorer.score(z).float()))
    if not scorers:
        scores = sample_candidate_health_scores(samples).reshape(n, candidates_per_prompt)
        best = scores.argmax(dim=1)
        base = torch.arange(n, device=samples.device) * candidates_per_prompt
        selected = base + best.to(device=samples.device)
        chosen = samples.index_select(0, selected)
        best_scores = scores.gather(1, best[:, None]).squeeze(1)
        meta.update({
            "sample_grid_selection_scorer": "sample_health",
            "sample_grid_selected_candidate_indices": [
                int(x) for x in best.detach().cpu().tolist()
            ],
            "sample_grid_selection_score_mean": float(best_scores.mean().detach().cpu()),
            "sample_grid_selection_score_min": float(best_scores.min().detach().cpu()),
            "sample_grid_selection_score_max": float(best_scores.max().detach().cpu()),
        })
        return chosen.contiguous(), meta
    component_scores = [
        (name, normalize_candidate_scores(score))
        for name, score in scorers
    ]
    scores = torch.stack([score for _name, score in component_scores], dim=0).mean(dim=0)
    best = scores.argmax(dim=1)
    base = torch.arange(n, device=samples.device) * candidates_per_prompt
    selected = base + best.to(device=samples.device)
    chosen = samples.index_select(0, selected)
    best_scores = scores.gather(1, best[:, None]).squeeze(1)
    scorer_labels = {
        "text_aligner": "text_aligner_cosine",
        "image_feature_aligner": "image_feature_aligner_cosine",
        "quality_scorer": "quality_scorer_score",
    }
    selection_meta = {
        "sample_grid_selection_scorer": "+".join(
            scorer_labels.get(name, name) for name, _score in scorers
        ),
        "sample_grid_selection_component_count": int(len(component_scores)),
        "sample_grid_selected_candidate_indices": [
            int(x) for x in best.detach().cpu().tolist()
        ],
        "sample_grid_selection_score_mean": float(best_scores.mean().detach().cpu()),
        "sample_grid_selection_score_min": float(best_scores.min().detach().cpu()),
        "sample_grid_selection_score_max": float(best_scores.max().detach().cpu()),
    }
    for name, score in component_scores:
        selected_scores = score.gather(1, best[:, None]).squeeze(1)
        selection_meta[f"sample_grid_selection_{name}_mean"] = float(
            selected_scores.mean().detach().cpu())
        selection_meta[f"sample_grid_selection_{name}_min"] = float(
            selected_scores.min().detach().cpu())
        selection_meta[f"sample_grid_selection_{name}_max"] = float(
            selected_scores.max().detach().cpu())
    meta.update(selection_meta)
    return chosen.contiguous(), meta


def _cycle_values(values, default, cast):
    if values is None:
        return (cast(default),)
    vals = tuple(cast(x) for x in values)
    if not vals:
        return (cast(default),)
    return vals


def prompt_candidate_sampler_settings(candidates_per_prompt, seed=0, cfg_scale=1.0,
                                      cfg_rescale=0.0, cfg_mode="standard",
                                      cfg_schedule="constant", sample_steps=4,
                                      sample_method="euler", sample_schedule="linear",
                                      sample_churn=0.0, cfg_scales=None,
                                      cfg_rescales=None, cfg_modes=None,
                                      cfg_schedules=None, sample_steps_list=None,
                                      sample_methods=None, sample_schedules=None,
                                      sample_churns=None, seeds=None):
    """Return cyclic per-candidate sampler settings for prompt preference mining."""
    n = int(candidates_per_prompt)
    if n <= 0:
        raise ValueError("candidates_per_prompt must be positive")
    if n == 1:
        cfg_scales = cfg_rescales = cfg_modes = cfg_schedules = None
        sample_steps_list = sample_methods = sample_schedules = sample_churns = None
        seeds = None
    cfg_scale_vals = _cycle_values(cfg_scales, cfg_scale, float)
    cfg_rescale_vals = _cycle_values(cfg_rescales, cfg_rescale, float)
    cfg_mode_vals = _cycle_values(cfg_modes, cfg_mode, str)
    cfg_schedule_vals = _cycle_values(cfg_schedules, cfg_schedule, str)
    step_vals = _cycle_values(sample_steps_list, sample_steps, int)
    method_vals = _cycle_values(sample_methods, sample_method, str)
    schedule_vals = _cycle_values(sample_schedules, sample_schedule, str)
    churn_vals = _cycle_values(sample_churns, sample_churn, float)
    seed_vals = tuple(int(x) for x in seeds) if seeds else ()
    for value in cfg_rescale_vals:
        if value < 0.0 or value > 1.0:
            raise ValueError("candidate cfg_rescale values must be in [0, 1]")
    for value in cfg_mode_vals:
        if value not in CFG_MODES:
            raise ValueError(f"unknown candidate cfg_mode {value!r}")
    for value in cfg_schedule_vals:
        validate_cfg_schedule(value)
    for value in step_vals:
        if int(value) <= 0:
            raise ValueError("candidate sample steps must be positive")
    for value in method_vals:
        if value not in SAMPLE_METHODS:
            raise ValueError(f"unknown candidate sample method {value!r}")
    for value in schedule_vals:
        if value not in SAMPLE_SCHEDULES:
            raise ValueError(f"unknown candidate sample schedule {value!r}")
    for value in churn_vals:
        if value < 0.0:
            raise ValueError("candidate sample churn values must be non-negative")
    settings = []
    base_seed = int(seed)
    for idx in range(n):
        settings.append({
            "candidate_index": int(idx),
            "cfg_scale": float(cfg_scale_vals[idx % len(cfg_scale_vals)]),
            "cfg_rescale": float(cfg_rescale_vals[idx % len(cfg_rescale_vals)]),
            "cfg_mode": str(cfg_mode_vals[idx % len(cfg_mode_vals)]),
            "cfg_schedule": str(cfg_schedule_vals[idx % len(cfg_schedule_vals)]),
            "sample_steps": int(step_vals[idx % len(step_vals)]),
            "sample_method": str(method_vals[idx % len(method_vals)]),
            "sample_schedule": str(schedule_vals[idx % len(schedule_vals)]),
            "sample_churn": float(churn_vals[idx % len(churn_vals)]),
            "seed": (
                int(seed_vals[idx % len(seed_vals)])
                if seed_vals else base_seed + 104729 * int(idx)
            ),
        })
    return settings


@torch.no_grad()
def save_caption_sample_grid(ae, flow, records, path, size=32, device=DEV, conditioner=None,
                             prompt_vocab=None, caption_max_len=64, cfg_scale=1.0,
                             cfg_rescale=0.0, sample_steps=4, sample_method="euler",
                             cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                             cfg_schedule="constant",
                             samples=16, seed=0, caption_cond_source="tokens",
                             sample_time_shift=1.0, sample_schedule="linear",
                             sample_churn=0.0,
                             sample_churn_interval=DEFAULT_CHURN_INTERVAL,
                             sample_finite_guard=False, sample_velocity_clip=0.0,
                             sample_latent_clip=0.0, cfg_mode="standard",
                             sample_pixel_dynamic_threshold_percentile=0.0,
                             sample_pixel_dynamic_threshold_max=1.0,
                             sample_manifest_out="", sample_image_dir=""):
    ae.eval()
    flow.eval()
    rng = np.random.default_rng(seed)
    n = min(max(1, int(samples)), len(records))
    idx = rng.choice(len(records), size=n, replace=len(records) < n)
    chosen = [records[int(i)] for i in idx]
    captions = [rec.caption for rec in chosen]
    cond = caption_record_condition(
        captions, chosen, conditioner, prompt_vocab, source=caption_cond_source,
        max_len=caption_max_len, device=device, return_tokens=flow_uses_cond_tokens(flow))
    cond = attach_image_geometry_condition(
        cond, records=chosen, target_size=size, enabled=flow_uses_image_geometry(flow))
    sample, trace = sample_images(
        ae, flow, cond, latent_shape=ae_latent_shape(ae, size),
        steps=sample_steps, device=device, seed=seed, cfg_scale=cfg_scale,
        cfg_rescale=cfg_rescale,
        sample_method=sample_method, cfg_interval=cfg_interval,
        cfg_schedule=cfg_schedule,
        sample_time_shift=sample_time_shift,
        sample_schedule=sample_schedule,
        sample_churn=sample_churn,
        sample_churn_interval=sample_churn_interval,
        sample_finite_guard=sample_finite_guard,
        sample_velocity_clip=sample_velocity_clip,
        sample_latent_clip=sample_latent_clip,
        sample_pixel_dynamic_threshold_percentile=(
            sample_pixel_dynamic_threshold_percentile),
        sample_pixel_dynamic_threshold_max=sample_pixel_dynamic_threshold_max,
        cfg_mode=cfg_mode,
        return_trace=True)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    meta = write_ppm_grid(sample, path, rows=rows, cols=cols)
    meta.update({
        "sample_grid_cfg_scale": float(cfg_scale),
        "sample_grid_cfg_rescale": float(cfg_rescale),
        "sample_grid_cfg_mode": cfg_mode,
        "sample_grid_cfg_schedule": cfg_schedule,
        "sample_grid_sample_steps": int(sample_steps),
        "sample_grid_sample_time_shift": float(sample_time_shift),
        "sample_grid_sample_method": sample_method,
        "sample_grid_sample_schedule": sample_schedule,
        "sample_grid_sample_churn": float(sample_churn),
        "sample_grid_pixel_dynamic_threshold_percentile": float(
            sample_pixel_dynamic_threshold_percentile),
        "sample_grid_pixel_dynamic_threshold_max": float(
            sample_pixel_dynamic_threshold_max),
        "sample_grid_sample_churn_interval": list(validate_guidance_interval(
            sample_churn_interval, name="sample_churn_interval")),
        "sample_grid_cfg_interval": list(validate_guidance_interval(cfg_interval)),
        "sample_grid_cond_mode": "caption",
        "sample_grid_caption_cond_source": caption_cond_source,
        "sample_grid_seed": int(seed),
        "sample_grid_caption_count": int(n),
        "sample_grid_captions": captions[:min(5, len(captions))],
    })
    meta.update(prefix_sample_trace(trace, "sample_grid"))
    meta.update(sample_health_metrics(sample, prefix="sample_grid"))
    if sample_manifest_out:
        row_meta = [
            {
                "conditioning_mode": "caption",
                "conditioning_caption": rec.caption,
                "conditioning_image": rec.path,
                "conditioning_source": rec.source,
            }
            for rec in chosen
        ]
        meta.update(write_sample_manifest(
            sample, sample_manifest_out, captions, image_dir=sample_image_dir,
            metadata={
                "sample_grid": path,
                "sample_cfg_scale": float(cfg_scale),
                "sample_cfg_rescale": float(cfg_rescale),
                "sample_cfg_mode": cfg_mode,
                "sample_cfg_schedule": cfg_schedule,
                "sample_steps": int(sample_steps),
                "sample_method": sample_method,
                "sample_schedule": sample_schedule,
                "sample_seed": int(seed),
                "caption_cond_source": caption_cond_source,
            },
            row_metadata=row_meta, prefix="caption_sample"))
    return meta


@torch.no_grad()
def save_text_prompt_sample_grid(ae, flow, prompts, path, size=32, device=DEV, conditioner=None,
                                 prompt_vocab=None, caption_max_len=64, cfg_scale=1.0,
                                 cfg_rescale=0.0, sample_steps=4, sample_method="euler",
                                 cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                                 cfg_schedule="constant", seed=0,
                                 caption_cond_source="tokens", sample_time_shift=1.0,
                                 sample_schedule="linear",
                                 sample_churn=0.0,
                                 sample_churn_interval=DEFAULT_CHURN_INTERVAL,
                                 prompt_embed_backend="stats", prompt_embed_model="",
                                 prompt_embed_device=None, prompt_embed_dtype="auto",
                                 prompt_embed_normalize=True, prompt_embed_stats_dim=0,
                                 prompt_embed_trust_remote_code=False,
                                 prompt_embed_text_sequence_model="",
                                 prompt_embed_text_sequence_max_length=0,
                                 negative_prompts=(), candidates_per_prompt=1,
                                 text_guidance_w=0.0,
                                 text_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                                 feature_guidance_w=0.0,
                                 feature_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                                 quality_guidance_w=0.0,
                                 quality_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                                 sample_finite_guard=False, sample_velocity_clip=0.0,
                                 sample_latent_clip=0.0, cfg_mode="standard",
                                 sample_pixel_dynamic_threshold_percentile=0.0,
                                 sample_pixel_dynamic_threshold_max=1.0,
                                 sample_manifest_out="", sample_image_dir="",
                                 sample_candidates_manifest_out="",
                                 sample_candidates_image_dir="",
                                 candidate_cfg_scales=None,
                                 candidate_cfg_rescales=None,
                                 candidate_cfg_modes=None,
                                 candidate_cfg_schedules=None,
                                 candidate_sample_steps_list=None,
                                 candidate_sample_methods=None,
                                 candidate_sample_schedules=None,
                                 candidate_sample_churns=None,
                                 candidate_seeds=None):
    ae.eval()
    flow.eval()
    prompts = tuple(str(p).strip() for p in prompts if str(p).strip())
    if not prompts:
        raise ValueError("at least one sample prompt is required")
    candidates_per_prompt = int(candidates_per_prompt)
    if candidates_per_prompt <= 0:
        raise ValueError("candidates_per_prompt must be positive")
    text_guidance_w = float(text_guidance_w)
    if text_guidance_w < 0.0:
        raise ValueError("text_guidance_w must be non-negative")
    text_guidance_interval = validate_guidance_interval(
        text_guidance_interval, name="text_guidance_interval")
    feature_guidance_w = float(feature_guidance_w)
    if feature_guidance_w < 0.0:
        raise ValueError("feature_guidance_w must be non-negative")
    feature_guidance_interval = validate_guidance_interval(
        feature_guidance_interval, name="feature_guidance_interval")
    quality_guidance_w = float(quality_guidance_w)
    if quality_guidance_w < 0.0:
        raise ValueError("quality_guidance_w must be non-negative")
    quality_guidance_interval = validate_guidance_interval(
        quality_guidance_interval, name="quality_guidance_interval")
    prompt_return_tokens = flow_uses_cond_tokens(flow)
    cond, prompt_features = text_prompt_condition(
        prompts, conditioner, prompt_vocab=prompt_vocab, caption_max_len=caption_max_len,
        source=caption_cond_source, device=device, return_tokens=prompt_return_tokens,
        embed_backend=prompt_embed_backend, embed_model=prompt_embed_model,
        embed_device=prompt_embed_device, embed_dtype=prompt_embed_dtype,
        embed_normalize=prompt_embed_normalize, embed_stats_dim=prompt_embed_stats_dim,
        trust_remote_code=prompt_embed_trust_remote_code,
        text_sequence_model=prompt_embed_text_sequence_model,
        text_sequence_max_length=prompt_embed_text_sequence_max_length,
        return_embedding=True)
    text_only_cond = cond
    cond = attach_image_geometry_condition(
        cond, target_size=size, batch=len(prompts),
        enabled=flow_uses_image_geometry(flow))
    negative_prompts = expand_negative_prompts(negative_prompts, len(prompts))
    cfg_uncond = None
    if negative_prompts:
        cfg_uncond = text_prompt_condition(
            negative_prompts, conditioner, prompt_vocab=prompt_vocab,
            caption_max_len=caption_max_len, source=caption_cond_source, device=device,
            return_tokens=prompt_return_tokens,
            embed_backend=prompt_embed_backend, embed_model=prompt_embed_model,
            embed_device=prompt_embed_device, embed_dtype=prompt_embed_dtype,
            embed_normalize=prompt_embed_normalize, embed_stats_dim=prompt_embed_stats_dim,
            trust_remote_code=prompt_embed_trust_remote_code,
            text_sequence_model=prompt_embed_text_sequence_model,
            text_sequence_max_length=prompt_embed_text_sequence_max_length)
        cfg_uncond = attach_image_geometry_condition(
            cfg_uncond, target_size=size, batch=len(prompts),
            enabled=flow_uses_image_geometry(flow))
    text_aligner = getattr(flow, "text_aligner", None)
    image_feature_aligner = getattr(flow, "image_feature_aligner", None)
    quality_scorer = getattr(flow, "image_quality_scorer", None)
    candidate_settings = prompt_candidate_sampler_settings(
        candidates_per_prompt, seed=seed, cfg_scale=cfg_scale,
        cfg_rescale=cfg_rescale, cfg_mode=cfg_mode, cfg_schedule=cfg_schedule,
        sample_steps=sample_steps, sample_method=sample_method,
        sample_schedule=sample_schedule, sample_churn=sample_churn,
        cfg_scales=candidate_cfg_scales, cfg_rescales=candidate_cfg_rescales,
        cfg_modes=candidate_cfg_modes, cfg_schedules=candidate_cfg_schedules,
        sample_steps_list=candidate_sample_steps_list,
        sample_methods=candidate_sample_methods,
        sample_schedules=candidate_sample_schedules,
        sample_churns=candidate_sample_churns, seeds=candidate_seeds)
    candidate_samples, trace = [], {}
    for setting in candidate_settings:
        sample_i, trace_i = sample_images(
            ae, flow, cond, latent_shape=ae_latent_shape(ae, size),
            steps=setting["sample_steps"], device=device, seed=setting["seed"],
            cfg_scale=setting["cfg_scale"],
            cfg_rescale=setting["cfg_rescale"],
            sample_method=setting["sample_method"], cfg_interval=cfg_interval,
            cfg_schedule=setting["cfg_schedule"],
            sample_time_shift=sample_time_shift,
            sample_schedule=setting["sample_schedule"], cfg_uncond=cfg_uncond,
            sample_churn=setting["sample_churn"],
            sample_churn_interval=sample_churn_interval,
            text_guidance_w=text_guidance_w,
            text_guidance_aligner=text_aligner,
            text_guidance_cond=text_only_cond,
            text_guidance_interval=text_guidance_interval,
            feature_guidance_w=feature_guidance_w,
            feature_guidance_aligner=image_feature_aligner,
            feature_guidance_features=prompt_features,
            feature_guidance_interval=feature_guidance_interval,
            quality_guidance_w=quality_guidance_w,
            quality_guidance_scorer=quality_scorer,
            quality_guidance_interval=quality_guidance_interval,
            sample_finite_guard=sample_finite_guard,
            sample_velocity_clip=sample_velocity_clip,
            sample_latent_clip=sample_latent_clip,
            sample_pixel_dynamic_threshold_percentile=(
                sample_pixel_dynamic_threshold_percentile),
            sample_pixel_dynamic_threshold_max=sample_pixel_dynamic_threshold_max,
            cfg_mode=setting["cfg_mode"],
            return_trace=True)
        candidate_samples.append(sample_i)
        trace = merge_sample_traces(trace, trace_i)
    candidate_sample = torch.stack(candidate_samples, dim=1).reshape(
        len(prompts) * candidates_per_prompt, *candidate_samples[0].shape[1:]
    ).contiguous()
    sample_text_only_cond = repeat_condition_rows(text_only_cond, candidates_per_prompt)
    sample, selection_meta = select_prompt_candidates(
        ae, candidate_sample, sample_text_only_cond, prompts,
        candidates_per_prompt=candidates_per_prompt,
        text_aligner=text_aligner, image_feature_aligner=image_feature_aligner,
        prompt_features=prompt_features, quality_scorer=quality_scorer)
    n = len(prompts)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    meta = write_ppm_grid(sample, path, rows=rows, cols=cols)
    meta.update({
        "sample_grid_cfg_scale": float(cfg_scale),
        "sample_grid_cfg_rescale": float(cfg_rescale),
        "sample_grid_cfg_mode": cfg_mode,
        "sample_grid_cfg_schedule": cfg_schedule,
        "sample_grid_sample_steps": int(sample_steps),
        "sample_grid_sample_time_shift": float(sample_time_shift),
        "sample_grid_sample_method": sample_method,
        "sample_grid_sample_schedule": sample_schedule,
        "sample_grid_sample_churn": float(sample_churn),
        "sample_grid_pixel_dynamic_threshold_percentile": float(
            sample_pixel_dynamic_threshold_percentile),
        "sample_grid_pixel_dynamic_threshold_max": float(
            sample_pixel_dynamic_threshold_max),
        "sample_grid_candidate_sweep_mode": "cyclic",
        "sample_grid_candidate_cfg_scales": [
            setting["cfg_scale"] for setting in candidate_settings],
        "sample_grid_candidate_cfg_rescales": [
            setting["cfg_rescale"] for setting in candidate_settings],
        "sample_grid_candidate_cfg_modes": [
            setting["cfg_mode"] for setting in candidate_settings],
        "sample_grid_candidate_cfg_schedules": [
            setting["cfg_schedule"] for setting in candidate_settings],
        "sample_grid_candidate_sample_steps": [
            setting["sample_steps"] for setting in candidate_settings],
        "sample_grid_candidate_sample_methods": [
            setting["sample_method"] for setting in candidate_settings],
        "sample_grid_candidate_sample_schedules": [
            setting["sample_schedule"] for setting in candidate_settings],
        "sample_grid_candidate_sample_churns": [
            setting["sample_churn"] for setting in candidate_settings],
        "sample_grid_candidate_seeds": [
            setting["seed"] for setting in candidate_settings],
        "sample_grid_sample_churn_interval": list(validate_guidance_interval(
            sample_churn_interval, name="sample_churn_interval")),
        "sample_grid_cfg_interval": list(validate_guidance_interval(cfg_interval)),
        "sample_grid_text_guidance_w": float(text_guidance_w),
        "sample_grid_text_guidance_interval": list(text_guidance_interval),
        "sample_grid_text_guidance_scorer": (
            "text_aligner" if text_guidance_w > 0.0 else "none"
        ),
        "sample_grid_feature_guidance_w": float(feature_guidance_w),
        "sample_grid_feature_guidance_interval": list(feature_guidance_interval),
        "sample_grid_feature_guidance_scorer": (
            "image_feature_aligner" if feature_guidance_w > 0.0 else "none"
        ),
        "sample_grid_quality_guidance_w": float(quality_guidance_w),
        "sample_grid_quality_guidance_interval": list(quality_guidance_interval),
        "sample_grid_quality_guidance_scorer": (
            "image_quality_scorer" if quality_guidance_w > 0.0 else "none"
        ),
        "sample_grid_cond_mode": "prompt",
        "sample_grid_caption_cond_source": caption_cond_source,
        "sample_grid_cfg_uncond_mode": (
            "negative_prompt" if negative_prompts
            else "learned_null" if condition_has_null(cond)
            else "zero"
        ),
        "sample_grid_negative_prompt_count": int(len(negative_prompts)),
        "sample_grid_negative_prompts": list(
            negative_prompts[:min(8, len(negative_prompts))]),
        "sample_grid_seed": int(seed),
        "sample_grid_prompt_count": int(n),
        "sample_grid_prompts": list(prompts[:min(8, len(prompts))]),
        "sample_grid_prompt_embed_backend": (
            prompt_embed_backend if caption_cond_source == "embedding" else ""
        ),
        "sample_grid_prompt_embed_model": (
            prompt_embed_model if caption_cond_source == "embedding" else ""
        ),
        "sample_grid_prompt_embed_text_sequence_model": (
            prompt_embed_text_sequence_model if caption_cond_source == "embedding" else ""
        ),
        "sample_grid_prompt_embed_text_sequence_max_length": (
            int(prompt_embed_text_sequence_max_length)
            if caption_cond_source == "embedding" else 0
        ),
        "sample_grid_prompt_embed_normalize": (
            bool(prompt_embed_normalize) if caption_cond_source == "embedding" else False
        ),
        "sample_grid_prompt_embed_text_mode": (
            "both" if caption_cond_source == "embedding" and prompt_return_tokens
            else "pooled" if caption_cond_source == "embedding" else ""
        ),
    })
    meta.update(selection_meta)
    meta.update(prefix_sample_trace(trace, "sample_grid"))
    meta.update(sample_health_metrics(sample, prefix="sample_grid"))
    if sample_candidates_manifest_out:
        selected = selection_meta.get("sample_grid_selected_candidate_indices", [])
        prompt_text_embeddings = None
        if caption_cond_source == "embedding" and torch.is_tensor(prompt_features):
            prompt_text_embeddings = (
                pool_embedding_sequence(prompt_features).detach().float().cpu().tolist()
            )
        candidate_captions = []
        candidate_rows = []
        for prompt_idx, prompt in enumerate(prompts):
            selected_idx = int(selected[prompt_idx]) if prompt_idx < len(selected) else 0
            for candidate_idx in range(candidates_per_prompt):
                setting = candidate_settings[candidate_idx]
                candidate_captions.append(prompt)
                row = {
                    "conditioning_mode": "prompt",
                    "conditioning_prompt": prompt,
                    "negative_prompt": (
                        negative_prompts[prompt_idx] if negative_prompts else ""),
                    "prompt_id": f"prompt_{prompt_idx:05d}",
                    "preference_group": f"prompt_{prompt_idx:05d}",
                    "candidate_index": int(candidate_idx),
                    "selected_candidate_index": int(selected_idx),
                    "selected_by_internal_sampler": bool(candidate_idx == selected_idx),
                    "candidates_per_prompt": int(candidates_per_prompt),
                    "sample_cfg_scale": float(setting["cfg_scale"]),
                    "sample_cfg_rescale": float(setting["cfg_rescale"]),
                    "sample_cfg_mode": setting["cfg_mode"],
                    "sample_cfg_schedule": setting["cfg_schedule"],
                    "sample_steps": int(setting["sample_steps"]),
                    "sample_method": setting["sample_method"],
                    "sample_schedule": setting["sample_schedule"],
                    "sample_churn": float(setting["sample_churn"]),
                    "sample_seed": int(setting["seed"]),
                    "sample_candidate_sweep_mode": "cyclic",
                    "sample_pixel_dynamic_threshold_percentile": float(
                        sample_pixel_dynamic_threshold_percentile),
                    "sample_pixel_dynamic_threshold_max": float(
                        sample_pixel_dynamic_threshold_max),
                }
                if prompt_text_embeddings is not None:
                    row["text_embedding"] = prompt_text_embeddings[prompt_idx]
                candidate_rows.append(row)
        candidate_meta = write_sample_manifest(
            candidate_sample, sample_candidates_manifest_out, candidate_captions,
            image_dir=sample_candidates_image_dir,
            metadata={
                "sample_grid": path,
                "sample_cfg_scale": float(cfg_scale),
                "sample_cfg_rescale": float(cfg_rescale),
                "sample_cfg_mode": cfg_mode,
                "sample_cfg_schedule": cfg_schedule,
                "sample_steps": int(sample_steps),
                "sample_method": sample_method,
                "sample_schedule": sample_schedule,
                "sample_seed": int(seed),
                "caption_cond_source": caption_cond_source,
                "sample_candidate_set": "all_prompt_candidates",
                "sample_candidate_sweep_mode": "cyclic",
                "sample_pixel_dynamic_threshold_percentile": float(
                    sample_pixel_dynamic_threshold_percentile),
                "sample_pixel_dynamic_threshold_max": float(
                    sample_pixel_dynamic_threshold_max),
                "prompt_embed_backend": (
                    prompt_embed_backend if caption_cond_source == "embedding" else ""),
                "prompt_embed_model": (
                    prompt_embed_model if caption_cond_source == "embedding" else ""),
            },
            row_metadata=candidate_rows, prefix="prompt_candidate")
        meta.update({
            "sample_candidates_manifest": candidate_meta["sample_manifest"],
            "sample_candidates_manifest_image_dir": (
                candidate_meta["sample_manifest_image_dir"]),
            "sample_candidates_manifest_records": (
                candidate_meta["sample_manifest_records"]),
            "sample_candidates_manifest_format": (
                candidate_meta["sample_manifest_format"]),
            "sample_candidates_manifest_image_format": (
                candidate_meta["sample_manifest_image_format"]),
        })
    if sample_manifest_out:
        selected = selection_meta.get("sample_grid_selected_candidate_indices", [])
        row_meta = []
        for idx, prompt in enumerate(prompts):
            selected_idx = int(selected[idx]) if idx < len(selected) else 0
            setting = candidate_settings[max(0, min(selected_idx, candidates_per_prompt - 1))]
            row_meta.append({
                "conditioning_mode": "prompt",
                "conditioning_prompt": prompt,
                "negative_prompt": negative_prompts[idx] if negative_prompts else "",
                "selected_candidate_index": selected_idx,
                "candidates_per_prompt": int(candidates_per_prompt),
                "sample_cfg_scale": float(setting["cfg_scale"]),
                "sample_cfg_rescale": float(setting["cfg_rescale"]),
                "sample_cfg_mode": setting["cfg_mode"],
                "sample_cfg_schedule": setting["cfg_schedule"],
                "sample_steps": int(setting["sample_steps"]),
                "sample_method": setting["sample_method"],
                "sample_schedule": setting["sample_schedule"],
                "sample_churn": float(setting["sample_churn"]),
                "sample_seed": int(setting["seed"]),
                "sample_candidate_sweep_mode": "cyclic",
                "sample_pixel_dynamic_threshold_percentile": float(
                    sample_pixel_dynamic_threshold_percentile),
                "sample_pixel_dynamic_threshold_max": float(
                    sample_pixel_dynamic_threshold_max),
            })
        meta.update(write_sample_manifest(
            sample, sample_manifest_out, list(prompts), image_dir=sample_image_dir,
            metadata={
                "sample_grid": path,
                "sample_cfg_scale": float(cfg_scale),
                "sample_cfg_rescale": float(cfg_rescale),
                "sample_cfg_mode": cfg_mode,
                "sample_cfg_schedule": cfg_schedule,
                "sample_steps": int(sample_steps),
                "sample_method": sample_method,
                "sample_schedule": sample_schedule,
                "sample_seed": int(seed),
                "caption_cond_source": caption_cond_source,
                "sample_pixel_dynamic_threshold_percentile": float(
                    sample_pixel_dynamic_threshold_percentile),
                "sample_pixel_dynamic_threshold_max": float(
                    sample_pixel_dynamic_threshold_max),
                "prompt_embed_backend": (
                    prompt_embed_backend if caption_cond_source == "embedding" else ""),
                "prompt_embed_model": (
                    prompt_embed_model if caption_cond_source == "embedding" else ""),
            },
            row_metadata=row_meta, prefix="prompt_sample"))
    return meta


@torch.no_grad()
def evaluate_image_records(ae, flow, records, n=128, batch=64, seed=10, size=32, device=DEV,
                           conditioner=None, prompt_vocab=None, caption_max_len=64,
                           cfg_scale=1.0, cfg_rescale=0.0,
                           sample_steps=4, sample_method="euler",
                           cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                           cfg_schedule="constant", text_aligner=None,
                           image_feature_aligner=None,
                           image_quality_scorer=None,
                           caption_cond_source="tokens", sample_time_shift=1.0,
                           sample_schedule="linear",
                           time_shift=1.0,
                           sample_churn=0.0,
                           sample_churn_interval=DEFAULT_CHURN_INTERVAL,
                           text_guidance_w=0.0,
                           text_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                           feature_guidance_w=0.0,
                           feature_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                           quality_guidance_w=0.0,
                           quality_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                           generated_eval_n=0, generated_eval_candidates_per_prompt=1,
                           sample_finite_guard=False, sample_velocity_clip=0.0,
                           sample_latent_clip=0.0, cfg_mode="standard"):
    rng = np.random.default_rng(seed)
    ae.eval()
    flow.eval()
    recon_losses, flow_losses = [], []
    endpoint_mses, endpoint_consistency_mses, endpoint_time_gaps = [], [], []
    latent_means, latent_stds = [], []
    align_losses, align_i2t, align_t2i = [], [], []
    feature_losses, feature_i2f, feature_f2i = [], [], []
    ext_pair_cos, ext_pair_i2t, ext_pair_t2i = [], [], []
    quality_scores = []
    score_external_text_image = can_score_external_text_image(records, image_feature_aligner)
    text_guidance_interval = validate_guidance_interval(
        text_guidance_interval, name="text_guidance_interval")
    feature_guidance_interval = validate_guidance_interval(
        feature_guidance_interval, name="feature_guidance_interval")
    quality_guidance_interval = validate_guidance_interval(
        quality_guidance_interval, name="quality_guidance_interval")
    generated_eval_n = int(generated_eval_n)
    if generated_eval_n < 0:
        raise ValueError("generated_eval_n must be non-negative")
    generated_eval_candidates_per_prompt = int(generated_eval_candidates_per_prompt)
    if generated_eval_candidates_per_prompt <= 0:
        raise ValueError("generated_eval_candidates_per_prompt must be positive")
    if text_guidance_w < 0.0:
        raise ValueError("text_guidance_w must be non-negative")
    if text_guidance_w > 0.0 and text_aligner is None:
        raise ValueError("manifest text guidance requires a checkpoint text_aligner")
    if feature_guidance_w < 0.0:
        raise ValueError("feature_guidance_w must be non-negative")
    if feature_guidance_w > 0.0:
        if image_feature_aligner is None:
            raise ValueError("manifest feature guidance requires a checkpoint image_feature_aligner")
        if not score_external_text_image:
            raise ValueError(
                "manifest feature guidance requires text embeddings in the same dimension as "
                "image embeddings"
            )
    if quality_guidance_w < 0.0:
        raise ValueError("quality_guidance_w must be non-negative")
    if quality_guidance_w > 0.0 and image_quality_scorer is None:
        raise ValueError("manifest quality guidance requires a checkpoint image_quality_scorer")
    cfg_schedule = validate_cfg_schedule(cfg_schedule)
    total = 0
    while total < n:
        b = min(batch, n - total)
        x, captions, chosen_records = sample_image_text_batch(
            records, rng, batch=b, size=size, device=device, return_records=True)
        out = ae(x)
        z = out["latent"]
        recon_losses.append(float(F.mse_loss(out["recon"], x).detach().cpu()))
        cond = caption_record_condition(
            captions, chosen_records, conditioner, prompt_vocab, source=caption_cond_source,
            max_len=caption_max_len, device=device, return_tokens=flow_uses_cond_tokens(flow))
        flow_cond = attach_image_geometry_condition(
            cond, records=chosen_records, target_size=size,
            enabled=flow_uses_image_geometry(flow))
        if text_aligner is not None:
            _align_loss, align_parts = image_text_alignment_loss(
                text_aligner, z, cond, prefix="caption_retrieval")
            align_losses.append(float(align_parts["caption_retrieval_loss"].detach().cpu()))
            align_i2t.append(float(align_parts["caption_retrieval_i2t_acc"].detach().cpu()))
            align_t2i.append(float(align_parts["caption_retrieval_t2i_acc"].detach().cpu()))
        if image_feature_aligner is not None:
            image_features = record_image_embedding_tensor(chosen_records, device=device)
            _feature_loss, feature_parts = image_feature_alignment_loss(
                image_feature_aligner, z, image_features, prefix="image_feature_retrieval")
            feature_losses.append(
                float(feature_parts["image_feature_retrieval_loss"].detach().cpu()))
            feature_i2f.append(
                float(feature_parts["image_feature_retrieval_i2f_acc"].detach().cpu()))
            feature_f2i.append(
                float(feature_parts["image_feature_retrieval_f2i_acc"].detach().cpu()))
            if score_external_text_image:
                text_features = record_text_embedding_tensor(chosen_records, device=device)
                pair_parts = embedding_pair_similarity(
                    image_feature_aligner.encode_image(z),
                    image_feature_aligner.encode_feature(text_features),
                    prefix="external_text_image_score")
                ext_pair_cos.append(float(
                    pair_parts["external_text_image_score_cos"].detach().cpu()))
                ext_pair_i2t.append(float(
                    pair_parts["external_text_image_score_i2t_acc"].detach().cpu()))
                ext_pair_t2i.append(float(
                    pair_parts["external_text_image_score_t2i_acc"].detach().cpu()))
        if image_quality_scorer is not None:
            quality_scores.append(float(image_quality_scorer.score(z).detach().mean().cpu()))
        flow_losses.append(float(latent_flow_loss(
            flow, z, flow_cond, time_shift=time_shift).detach().cpu()))
        endpoint_metrics = flow_endpoint_metrics(flow, z, flow_cond, time_shift=time_shift)
        endpoint_mses.append(endpoint_metrics["latent_endpoint_mse"])
        endpoint_consistency_mses.append(endpoint_metrics["latent_endpoint_consistency_mse"])
        endpoint_time_gaps.append(endpoint_metrics["latent_endpoint_time_gap"])
        latent_means.append(float(z.mean().detach().cpu()))
        latent_stds.append(float(z.std().detach().cpu()))
        total += b

    sample_target_n = (
        int(generated_eval_n) if generated_eval_n > 0
        else min(int(batch), max(1, int(sample_steps)))
    )
    sample_target_n = max(1, sample_target_n)
    sample_batch = max(1, int(batch))
    sample_rng = np.random.default_rng(seed + 17)
    sample_parts, sample_x_parts, sample_captions, sample_records = [], [], [], []
    selection_scorer = "none"
    selection_component_count = 0
    selected_candidate_indices = []
    selection_scores = []
    selection_component_scores = defaultdict(list)
    sample_done = 0
    sample_trace = {}
    while sample_done < sample_target_n:
        sample_b = min(sample_batch, sample_target_n - sample_done)
        chunk_x, chunk_captions, chunk_records = sample_image_text_batch(
            records, sample_rng, batch=sample_b, size=size, device=device,
            return_records=True)
        chunk_cond = caption_record_condition(
            chunk_captions, chunk_records, conditioner, prompt_vocab,
            source=caption_cond_source, max_len=caption_max_len, device=device,
            return_tokens=flow_uses_cond_tokens(flow))
        chunk_flow_cond = attach_image_geometry_condition(
            chunk_cond, records=chunk_records, target_size=size,
            enabled=flow_uses_image_geometry(flow))
        chunk_sample_cond = repeat_condition_rows(
            chunk_flow_cond, generated_eval_candidates_per_prompt)
        chunk_text_sample_cond = repeat_condition_rows(
            chunk_cond, generated_eval_candidates_per_prompt)
        chunk_text_features = (
            record_text_embedding_tensor(chunk_records, device=device)
            if (feature_guidance_w > 0.0 or image_feature_aligner is not None) else None
        )
        chunk_guidance_features = (
            chunk_text_features.repeat_interleave(
                generated_eval_candidates_per_prompt, dim=0)
            if feature_guidance_w > 0.0 and torch.is_tensor(chunk_text_features) else None
        )
        chunk_sample, chunk_trace = sample_images(
            ae, flow, chunk_sample_cond,
            latent_shape=ae_latent_shape(ae, size),
            steps=sample_steps, device=device,
            seed=(seed if sample_done == 0 else seed + sample_done),
            cfg_scale=cfg_scale,
            cfg_rescale=cfg_rescale,
            sample_method=sample_method, cfg_interval=cfg_interval,
            cfg_schedule=cfg_schedule,
            sample_time_shift=sample_time_shift,
            sample_schedule=sample_schedule,
            sample_churn=sample_churn,
            sample_churn_interval=sample_churn_interval,
            text_guidance_w=text_guidance_w,
            text_guidance_aligner=text_aligner,
            text_guidance_cond=chunk_text_sample_cond,
            text_guidance_interval=text_guidance_interval,
            feature_guidance_w=feature_guidance_w,
            feature_guidance_aligner=image_feature_aligner,
            feature_guidance_features=chunk_guidance_features,
            feature_guidance_interval=feature_guidance_interval,
            quality_guidance_w=quality_guidance_w,
            quality_guidance_scorer=image_quality_scorer,
            quality_guidance_interval=quality_guidance_interval,
            sample_finite_guard=sample_finite_guard,
            sample_velocity_clip=sample_velocity_clip,
            sample_latent_clip=sample_latent_clip,
            cfg_mode=cfg_mode,
            return_trace=True)
        chunk_sample, chunk_selection = select_prompt_candidates(
            ae, chunk_sample, chunk_text_sample_cond, chunk_captions,
            candidates_per_prompt=generated_eval_candidates_per_prompt,
            text_aligner=text_aligner, image_feature_aligner=image_feature_aligner,
            prompt_features=chunk_text_features, quality_scorer=image_quality_scorer)
        sample_parts.append(chunk_sample)
        sample_trace = merge_sample_traces(sample_trace, chunk_trace)
        sample_x_parts.append(chunk_x)
        sample_captions.extend(chunk_captions)
        sample_records.extend(chunk_records)
        selection_scorer = str(chunk_selection.get(
            "sample_grid_selection_scorer", selection_scorer))
        selection_component_count = max(
            int(selection_component_count),
            int(chunk_selection.get("sample_grid_selection_component_count", 0) or 0))
        selected_candidate_indices.extend(
            int(x) for x in chunk_selection.get(
                "sample_grid_selected_candidate_indices", []))
        if "sample_grid_selection_score_mean" in chunk_selection:
            selection_scores.append(float(chunk_selection["sample_grid_selection_score_mean"]))
        for key, value in chunk_selection.items():
            if (key.startswith("sample_grid_selection_")
                    and key.endswith("_mean")
                    and key != "sample_grid_selection_score_mean"):
                selection_component_scores[key].append(float(value))
        sample_done += sample_b
    sample = torch.cat(sample_parts, dim=0)
    sample_x = torch.cat(sample_x_parts, dim=0)
    sample_cond = caption_record_condition(
        sample_captions, sample_records, conditioner, prompt_vocab, source=caption_cond_source,
        max_len=caption_max_len, device=device, return_tokens=flow_uses_cond_tokens(flow))
    report = {
        "n": int(total),
        "recon_mse": float(np.mean(recon_losses)),
        "latent_velocity_mse": float(np.mean(flow_losses)),
        "latent_endpoint_mse": float(np.mean(endpoint_mses)),
        "latent_endpoint_consistency_mse": float(np.mean(endpoint_consistency_mses)),
        "latent_endpoint_time_gap": float(np.mean(endpoint_time_gaps)),
        "latent_mean": float(np.mean(latent_means)),
        "latent_std": float(np.mean(latent_stds)),
        "sample_min": float(sample.min().detach().cpu()),
        "sample_max": float(sample.max().detach().cpu()),
        "caption_sample_mse": float(F.mse_loss(sample, sample_x[:sample.shape[0]]).detach().cpu()),
        "cfg_scale": float(cfg_scale),
        "cfg_rescale": float(cfg_rescale),
        "cfg_mode": cfg_mode,
        "cfg_schedule": cfg_schedule,
        "cfg_interval": list(validate_guidance_interval(cfg_interval)),
        "sample_steps": int(sample_steps),
        "generated_eval_n": int(sample.shape[0]),
        "generated_eval_n_requested": int(generated_eval_n),
        "generated_eval_batch": int(sample_batch),
        "generated_eval_candidates_per_prompt": int(generated_eval_candidates_per_prompt),
        "generated_eval_raw_candidates": int(
            sample.shape[0] * generated_eval_candidates_per_prompt),
        "generated_eval_selection_scorer": selection_scorer,
        "generated_eval_selection_component_count": int(selection_component_count),
        "generated_eval_selected_candidate_indices": selected_candidate_indices,
        "generated_eval_selection_score_mean": (
            float(np.mean(selection_scores)) if selection_scores else 0.0),
        "sample_time_shift": float(sample_time_shift),
        "time_shift": float(time_shift),
        "sample_method": sample_method,
        "sample_schedule": sample_schedule,
        "sample_churn": float(sample_churn),
        "sample_churn_interval": list(validate_guidance_interval(
            sample_churn_interval, name="sample_churn_interval")),
        "text_guidance_w": float(text_guidance_w),
        "text_guidance_interval": list(text_guidance_interval),
        "feature_guidance_w": float(feature_guidance_w),
        "feature_guidance_interval": list(feature_guidance_interval),
        "quality_guidance_w": float(quality_guidance_w),
        "quality_guidance_interval": list(quality_guidance_interval),
        "cond_mode": "text",
        "caption_cond_source": caption_cond_source,
        "data_mode": "image_manifest",
        "eval_captions": sample_captions[:min(3, len(sample_captions))],
    }
    report.update(sample_trace)
    report.update(sample_health_metrics(sample))
    for key, values in selection_component_scores.items():
        report[key.replace("sample_grid_", "generated_eval_")] = float(np.mean(values))
    if text_aligner is not None:
        sample_z = ae.encode(sample)
        _sample_align_loss, sample_align_parts = image_text_alignment_loss(
            text_aligner, sample_z, sample_cond, prefix="generated_caption_retrieval")
        if align_losses:
            report.update({
                "caption_retrieval_loss": float(np.mean(align_losses)),
                "caption_retrieval_i2t_acc": float(np.mean(align_i2t)),
                "caption_retrieval_t2i_acc": float(np.mean(align_t2i)),
            })
        report.update({
            "generated_caption_retrieval_loss": float(
                sample_align_parts["generated_caption_retrieval_loss"].detach().cpu()),
            "generated_caption_retrieval_i2t_acc": float(
                sample_align_parts["generated_caption_retrieval_i2t_acc"].detach().cpu()),
            "generated_caption_retrieval_t2i_acc": float(
                sample_align_parts["generated_caption_retrieval_t2i_acc"].detach().cpu()),
            "generated_caption_retrieval_n": int(
                sample_align_parts["generated_caption_retrieval_n"].detach().cpu()),
        })
    if image_feature_aligner is not None:
        sample_z = ae.encode(sample)
        sample_features = record_image_embedding_tensor(
            sample_records[:sample.shape[0]], device=device)
        generated_feature_emb = image_feature_aligner.encode_image(sample_z)
        real_feature_emb = image_feature_aligner.encode_feature(sample_features)
        _sample_feature_loss, sample_feature_parts = image_feature_alignment_loss(
            image_feature_aligner, sample_z, sample_features,
            prefix="generated_image_feature_retrieval")
        if feature_losses:
            report.update({
                "image_feature_retrieval_loss": float(np.mean(feature_losses)),
                "image_feature_retrieval_i2f_acc": float(np.mean(feature_i2f)),
                "image_feature_retrieval_f2i_acc": float(np.mean(feature_f2i)),
            })
        if ext_pair_cos:
            report.update({
                "external_text_image_score_cos": float(np.mean(ext_pair_cos)),
                "external_text_image_score_i2t_acc": float(np.mean(ext_pair_i2t)),
                "external_text_image_score_t2i_acc": float(np.mean(ext_pair_t2i)),
            })
        report.update({
            "generated_image_feature_retrieval_loss": float(
                sample_feature_parts["generated_image_feature_retrieval_loss"].detach().cpu()),
            "generated_image_feature_retrieval_i2f_acc": float(
                sample_feature_parts["generated_image_feature_retrieval_i2f_acc"].detach().cpu()),
            "generated_image_feature_retrieval_f2i_acc": float(
                sample_feature_parts["generated_image_feature_retrieval_f2i_acc"].detach().cpu()),
            "generated_image_feature_retrieval_n": int(
                sample_feature_parts["generated_image_feature_retrieval_n"].detach().cpu()),
        })
        report.update(embedding_distribution_metrics(
            generated_feature_emb, real_feature_emb,
            prefix="generated_image_feature_distribution"))
        if score_external_text_image:
            sample_text_features = record_text_embedding_tensor(
                sample_records[:sample.shape[0]], device=device)
            generated_pair_parts = embedding_pair_similarity(
                generated_feature_emb,
                image_feature_aligner.encode_feature(sample_text_features),
                prefix="generated_external_text_image_score")
            report.update({
                "generated_external_text_image_score_cos": float(
                    generated_pair_parts[
                        "generated_external_text_image_score_cos"].detach().cpu()),
                "generated_external_text_image_score_i2t_acc": float(
                    generated_pair_parts[
                        "generated_external_text_image_score_i2t_acc"].detach().cpu()),
                "generated_external_text_image_score_t2i_acc": float(
                    generated_pair_parts[
                        "generated_external_text_image_score_t2i_acc"].detach().cpu()),
                "generated_external_text_image_score_n": int(
                    generated_pair_parts[
                        "generated_external_text_image_score_n"].detach().cpu()),
            })
    if image_quality_scorer is not None:
        sample_z = ae.encode(sample)
        report.update({
            "image_quality_score_pred_mean": float(np.mean(quality_scores))
            if quality_scores else 0.0,
            "generated_image_quality_score_pred_mean": float(
                image_quality_scorer.score(sample_z).detach().mean().cpu()),
        })
    report.update(summarize_records(records))
    return report


def load_flow_state(flow, state_dict):
    missing, unexpected = flow.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"flow checkpoint mismatch: missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return {"missing": list(missing), "unexpected": list(unexpected)}


def load_conditioner_state(conditioner, state_dict):
    missing, unexpected = conditioner.load_state_dict(state_dict, strict=False)
    allowed_missing = [
        key for key in missing
        if key.startswith("null_") or key in ("null_vec", "null_token", "null_tokens")
    ]
    bad_missing = sorted(set(missing) - set(allowed_missing))
    if bad_missing or unexpected:
        raise RuntimeError(
            f"conditioner checkpoint mismatch: missing={bad_missing}, "
            f"unexpected={list(unexpected)}"
        )
    return {"missing": list(missing), "unexpected": list(unexpected)}


def clone_state_dict(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def ema_effective_decay(target_decay, step, warmup=True):
    if target_decay <= 0.0:
        return 0.0
    if not warmup:
        return float(target_decay)
    # Use a generic warmup so high target decays do not average mostly initialization
    # during short runs. This approaches the requested decay as training gets longer.
    warm = (1.0 + float(step)) / (10.0 + float(step))
    return float(min(target_decay, warm))


@torch.no_grad()
def update_ema_state(ema_state, module, decay):
    state = module.state_dict()
    for key, val in state.items():
        cur = val.detach()
        if key not in ema_state:
            ema_state[key] = cur.clone()
        elif torch.is_floating_point(ema_state[key]):
            ema_state[key].mul_(decay).add_(cur, alpha=1.0 - decay)
        else:
            ema_state[key].copy_(cur)


def load_checkpoint(path, device=DEV, prefer_ema=True):
    ckpt = torch.load(path, map_location=device)
    report = ckpt.get("report", {})
    latent_ch = int(ckpt.get("latent_ch", report.get("latent_ch", 16)))
    raw_image_size = ckpt.get("image_size", report.get("image_size", None))
    missing_size = raw_image_size is None or (
        isinstance(raw_image_size, str) and raw_image_size == "")
    if missing_size and ckpt.get("image_h") and ckpt.get("image_w"):
        raw_image_size = (ckpt["image_h"], ckpt["image_w"])
    image_size = image_hw(raw_image_size, default=32)
    hidden = int(ckpt.get("hidden", report.get("hidden", 64)))
    flow_arch = ckpt.get("flow_arch", report.get("flow_arch", "conv"))
    ae_arch = ckpt.get("ae_arch", report.get("ae_arch", "semantic"))
    latent_downsample = int(ckpt.get("latent_downsample",
                                     report.get("latent_downsample", 4)))
    ae_res_blocks = int(ckpt.get("ae_res_blocks", report.get("ae_res_blocks", 0)))
    latent_max_tokens = int(ckpt.get("latent_max_tokens",
                                     report.get("latent_max_tokens", 256)))
    latent_patch_size = int(ckpt.get("latent_patch_size",
                                     report.get("latent_patch_size", 1)) or 1)
    if latent_patch_size <= 0:
        raise ValueError("latent_patch_size must be positive")
    dit_depth = int(ckpt.get("dit_depth", report.get("dit_depth", 3)))
    dit_heads = int(ckpt.get("dit_heads", report.get("dit_heads", 4)))
    dit_head_width_mult = int(ckpt.get("dit_head_width_mult",
                                       report.get("dit_head_width_mult", 1)))
    dit_qk_norm = bool(ckpt.get("dit_qk_norm", report.get("dit_qk_norm", False)))
    dit_attn_impl = str(ckpt.get("dit_attn_impl", report.get("dit_attn_impl", "manual")))
    if dit_attn_impl not in MMDIT_ATTN_IMPLS:
        raise ValueError(f"unknown MM-DiT attention implementation {dit_attn_impl!r}")
    dit_pos_embed = str(ckpt.get("dit_pos_embed", report.get("dit_pos_embed", "learned"))
                        or "learned")
    if dit_pos_embed not in DIT_POS_EMBEDS:
        raise ValueError(f"unknown DiT positional embedding {dit_pos_embed!r}")
    dit_mlp = str(ckpt.get("dit_mlp", report.get("dit_mlp", "gelu")) or "gelu")
    if dit_mlp not in DIT_MLPS:
        raise ValueError(f"unknown DiT MLP {dit_mlp!r}")
    flow_time_embed = str(ckpt.get(
        "flow_time_embed", report.get("flow_time_embed", "scalar")) or "scalar")
    if flow_time_embed not in FLOW_TIME_EMBEDS:
        raise ValueError(f"unknown flow time embedding {flow_time_embed!r}")
    flow_time_embed_dim = int(ckpt.get(
        "flow_time_embed_dim", report.get("flow_time_embed_dim", 1)) or 1)
    flow_checkpoint_blocks = bool(ckpt.get(
        "flow_checkpoint_blocks", report.get("flow_checkpoint_blocks", False)))
    checkpoint_flow_boundary_mode = str(ckpt.get(
        "flow_boundary_mode", report.get("flow_boundary_mode", "none")) or "none")
    if checkpoint_flow_boundary_mode not in FLOW_BOUNDARY_MODES:
        raise ValueError(f"unknown flow boundary mode {checkpoint_flow_boundary_mode!r}")
    cond_mode = ckpt.get("cond_mode", report.get("cond_mode", "text"))
    data_mode = ckpt.get("data_mode", report.get("data_mode", "image_manifest"))
    if cond_mode != "text" or data_mode != "image_manifest":
        raise ValueError("checkpoint must be manifest text-conditioned")
    cond_dim = int(ckpt.get("cond_dim", report.get("cond_dim", 0)) or 0)
    if cond_dim <= 0:
        raise ValueError("checkpoint is missing cond_dim")
    caption_cond_source = ckpt.get("caption_cond_source",
                                   report.get("caption_cond_source", "tokens"))
    text_embedding_in_dim = int(ckpt.get(
        "text_embedding_in_dim", report.get("text_embedding_in_dim", 0)) or 0)
    text_pooled_embedding_in_dim = int(ckpt.get(
        "text_pooled_embedding_in_dim",
        report.get("text_pooled_embedding_in_dim", 0)) or 0)
    text_embed_dim = int(ckpt.get("text_embed_dim", report.get("text_embed_dim", 128)) or 128)
    image_embedding_in_dim = int(ckpt.get(
        "image_embedding_in_dim", report.get("image_embedding_in_dim", 0)) or 0)
    image_feature_embed_dim = int(ckpt.get(
        "image_feature_embed_dim", report.get("image_feature_embed_dim", 128)) or 128)
    image_quality_score_w = float(ckpt.get(
        "image_quality_score_w", report.get("image_quality_score_w", 0.0)) or 0.0)
    flow_quality_score_w = float(ckpt.get(
        "flow_quality_score_w", report.get("flow_quality_score_w", 0.0)) or 0.0)
    flow_repa_embed_dim = int(ckpt.get(
        "flow_repa_embed_dim", report.get("flow_repa_embed_dim", 128)) or 128)
    flow_repa_mode = str(ckpt.get("flow_repa_mode", report.get("flow_repa_mode", "pooled"))
                         or "pooled")
    flow_self_repa_embed_dim = int(ckpt.get(
        "flow_self_repa_embed_dim",
        report.get("flow_self_repa_embed_dim", 128)) or 128)
    flow_self_repa_mode = str(ckpt.get(
        "flow_self_repa_mode", report.get("flow_self_repa_mode", "pooled")) or "pooled")
    flow_sra_mode = str(ckpt.get(
        "flow_sra_mode", report.get("flow_sra_mode", "token")) or "token")
    ae_hf_model = str(ckpt.get("ae_hf_model", report.get("ae_hf_model", "")) or "")
    ae_hf_subfolder = str(ckpt.get(
        "ae_hf_subfolder", report.get("ae_hf_subfolder", "")) or "")
    ae_hf_scaling_factor = float(ckpt.get(
        "ae_hf_scaling_factor", report.get("ae_hf_scaling_factor", 0.0)) or 0.0)
    caption_max_len = int(ckpt.get("caption_max_len", report.get("caption_max_len", 32)) or 32)
    prompt_templates = ()
    prompt_vocab = ckpt.get("prompt_vocab") or None
    conditioner = None
    if cond_mode == "text":
        if caption_cond_source == "embedding":
            if text_embedding_in_dim <= 0:
                raise ValueError("embedding-conditioned checkpoint is missing text_embedding_in_dim")
            conditioner = PrecomputedTextConditioner(
                text_embedding_in_dim, cond_dim=cond_dim, hidden=hidden,
                pooled_input_dim=(
                    text_pooled_embedding_in_dim
                    if text_pooled_embedding_in_dim > 0
                    and text_pooled_embedding_in_dim != text_embedding_in_dim
                    else 0
                )).to(device)
        else:
            if prompt_vocab is None:
                raise ValueError("token-conditioned checkpoint is missing prompt_vocab")
            conditioner = PromptConditioner(len(prompt_vocab), cond_dim=cond_dim,
                                            hidden=hidden, max_len=caption_max_len).to(device)
        cond_state = ckpt["conditioner_state_dict"]
        if prefer_ema and ckpt.get("conditioner_ema_state_dict"):
            cond_state = ckpt["conditioner_ema_state_dict"]
        load_conditioner_state(conditioner, cond_state)
        conditioner.eval()
    text_aligner = None
    if ckpt.get("text_aligner_state_dict"):
        text_aligner = ImageTextAligner(
            latent_ch=latent_ch, cond_dim=cond_dim, hidden=hidden,
            embed_dim=text_embed_dim).to(device)
        text_aligner.load_state_dict(ckpt["text_aligner_state_dict"])
        text_aligner.eval()
    image_feature_aligner = None
    if ckpt.get("image_feature_aligner_state_dict"):
        if image_embedding_in_dim <= 0:
            raise ValueError(
                "image-feature-aligned checkpoint is missing image_embedding_in_dim")
        image_feature_aligner = ImageFeatureAligner(
            latent_ch=latent_ch, feature_dim=image_embedding_in_dim, hidden=hidden,
            embed_dim=image_feature_embed_dim).to(device)
        image_feature_aligner.load_state_dict(ckpt["image_feature_aligner_state_dict"])
        image_feature_aligner.eval()
    image_quality_scorer = None
    if ckpt.get("image_quality_scorer_state_dict"):
        image_quality_scorer = ImageQualityScorer(
            latent_ch=latent_ch, hidden=hidden).to(device)
        image_quality_scorer.load_state_dict(ckpt["image_quality_scorer_state_dict"])
        image_quality_scorer.eval()
    ae = make_autoencoder(ae_arch=ae_arch, latent_ch=latent_ch, hidden=hidden,
                          latent_downsample=latent_downsample,
                          ae_res_blocks=ae_res_blocks,
                          ae_hf_model=ae_hf_model,
                          ae_hf_subfolder=ae_hf_subfolder,
                          ae_hf_scaling_factor=ae_hf_scaling_factor).to(device)
    latent_ch = int(getattr(ae, "latent_ch", latent_ch))
    flow = make_flow(flow_arch=flow_arch, latent_ch=latent_ch, hidden=hidden,
                     dit_depth=dit_depth, dit_heads=dit_heads, cond_dim=cond_dim,
                     dit_head_width_mult=dit_head_width_mult,
                     latent_max_tokens=latent_max_tokens,
                     dit_qk_norm=dit_qk_norm,
                     dit_attn_impl=dit_attn_impl,
                     dit_pos_embed=dit_pos_embed,
                     dit_mlp=dit_mlp,
                     latent_patch_size=latent_patch_size,
                     flow_checkpoint_blocks=flow_checkpoint_blocks,
                     flow_time_embed=flow_time_embed,
                     flow_time_embed_dim=flow_time_embed_dim).to(device)
    attach_image_geometry_mode(flow, bool(ckpt.get(
        "image_geometry_cond", report.get("image_geometry_cond", False))))
    flow_repa_aligner = None
    if ckpt.get("flow_repa_aligner_state_dict"):
        if image_embedding_in_dim <= 0:
            raise ValueError("REPA-aligned checkpoint is missing image_embedding_in_dim")
        hidden_feature_dim = flow_hidden_feature_dim(flow)
        if hidden_feature_dim <= 0:
            raise ValueError("REPA-aligned checkpoint requires a transformer flow")
        flow_repa_aligner = FlowFeatureAligner(
            hidden_dim=hidden_feature_dim, feature_dim=image_embedding_in_dim,
            hidden=hidden, embed_dim=flow_repa_embed_dim).to(device)
        flow_repa_aligner.load_state_dict(ckpt["flow_repa_aligner_state_dict"])
        flow_repa_aligner.eval()
    flow_self_repa_aligner = None
    if ckpt.get("flow_self_repa_aligner_state_dict"):
        hidden_feature_dim = flow_hidden_feature_dim(flow)
        if hidden_feature_dim <= 0:
            raise ValueError("self-REPA-aligned checkpoint requires a transformer flow")
        flow_self_repa_aligner = FlowLatentAligner(
            hidden_dim=hidden_feature_dim, latent_ch=latent_ch,
            patch_size=latent_patch_size, hidden=hidden,
            embed_dim=flow_self_repa_embed_dim).to(device)
        flow_self_repa_aligner.load_state_dict(ckpt["flow_self_repa_aligner_state_dict"])
        flow_self_repa_aligner.eval()
    latent_stats = latent_stats_to_device(
        ckpt.get("latent_stats", {"mode": ckpt.get("latent_normalize", "none")}),
        device)
    attach_latent_stats(flow, latent_stats)
    ae_state = ckpt.get("autoencoder_state_dict", {})
    if ae_state:
        ae.load_state_dict(ae_state)
    flow_state = ckpt["flow_state_dict"]
    ema_available = bool(ckpt.get("flow_ema_state_dict"))
    ema_loaded = bool(prefer_ema and ema_available)
    if ema_loaded:
        flow_state = ckpt["flow_ema_state_dict"]
    flow_load = load_flow_state(flow, flow_state)
    attach_text_aligner(flow, text_aligner)
    attach_image_feature_aligner(flow, image_feature_aligner)
    attach_image_quality_scorer(flow, image_quality_scorer)
    attach_flow_repa_aligner(flow, flow_repa_aligner)
    attach_flow_self_repa_aligner(flow, flow_self_repa_aligner)
    attach_flow_boundary_mode(flow, checkpoint_flow_boundary_mode)
    ae.eval()
    flow.eval()
    return ae, flow, conditioner, prompt_vocab, prompt_templates, {
        "checkpoint": path,
        "checkpoint_report": report,
        "image_size": image_size_value(image_size),
        "image_h": int(image_size[0]),
        "image_w": int(image_size[1]),
        "size_buckets": report.get("size_buckets", []),
        "size_bucket_count": int(report.get(
            "size_bucket_count", len(report.get("size_buckets", [])))),
        "latent_ch": latent_ch,
        "ae_arch": ae_arch,
        "ae_external": ae_arch == "hf-vae",
        "ae_hf_model": ae_hf_model,
        "ae_hf_subfolder": ae_hf_subfolder,
        "ae_hf_scaling_factor": float(getattr(ae, "scaling_factor", ae_hf_scaling_factor)),
        "latent_downsample": int(getattr(ae, "downsample", latent_downsample)),
        "ae_res_blocks": int(ae_res_blocks) if ae_arch == "residual" else 0,
        "latent_max_tokens": int(latent_max_tokens),
        "latent_patch_size": int(latent_patch_size),
        "hidden": hidden,
        "flow_arch": flow_arch,
        "dit_depth": dit_depth if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "dit_heads": dit_heads if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "dit_head_width_mult": (
            dit_head_width_mult if flow_arch in ("dit", "crossdit", "mmdit") else 1
        ),
        "dit_qk_norm": bool(dit_qk_norm) if flow_arch == "mmdit" else False,
        "dit_attn_impl": dit_attn_impl if flow_arch == "mmdit" else "manual",
        "dit_pos_embed": dit_pos_embed if flow_arch in ("dit", "crossdit", "mmdit") else "",
        "dit_mlp": dit_mlp if flow_arch in ("dit", "crossdit", "mmdit") else "",
        "flow_time_embed": flow_time_embed if flow_arch in ("dit", "crossdit", "mmdit") else "",
        "flow_time_embed_dim": (
            int(flow_time_embed_dim) if flow_arch in ("dit", "crossdit", "mmdit") else 1
        ),
        "uses_swiglu_mlp": bool(getattr(flow, "uses_swiglu_mlp", False)),
        "flow_checkpoint_blocks": bool(getattr(flow, "checkpoint_blocks", False)),
        "flow_boundary_mode": get_flow_boundary_mode(flow),
        "activation_checkpointing": bool(getattr(flow, "uses_activation_checkpointing", False)),
        "uses_2d_pos_embed": bool(getattr(flow, "uses_2d_pos_embed", False)),
        "uses_rope2d_pos_embed": bool(getattr(flow, "uses_rope2d_pos_embed", False)),
        "adaptive_modulation": bool(getattr(flow, "uses_adaptive_modulation", False)),
        "residual_gating": bool(getattr(flow, "uses_residual_gating", False)),
        "zero_residual_gating": bool(getattr(flow, "uses_zero_residual_gating", False)),
        "flow_load": flow_load,
        "ema_available": ema_available,
        "ema_loaded": ema_loaded,
        "flow_ema_decay": float(ckpt.get("flow_ema_decay", report.get("flow_ema_decay", 0.0))),
        "flow_consistency_w": float(ckpt.get(
            "flow_consistency_w", report.get("flow_consistency_w", 0.0))),
        "time_sampling": str(ckpt.get("time_sampling", report.get("time_sampling", "uniform"))),
        "time_curriculum_frac": float(ckpt.get(
            "time_curriculum_frac", report.get("time_curriculum_frac", 0.0))),
        "time_curriculum_switch_step": int(ckpt.get(
            "time_curriculum_switch_step", report.get("time_curriculum_switch_step", 0))),
        "time_curriculum_final_sampling": str(ckpt.get(
            "time_curriculum_final_sampling",
            report.get("time_curriculum_final_sampling", "uniform"))),
        "size_curriculum_frac": float(ckpt.get(
            "size_curriculum_frac", report.get("size_curriculum_frac", 0.0))),
        "size_curriculum_switch_step": int(ckpt.get(
            "size_curriculum_switch_step", report.get("size_curriculum_switch_step", 0))),
        "size_curriculum_bucket_order": list(ckpt.get(
            "size_curriculum_bucket_order", report.get("size_curriculum_bucket_order", []))),
        "time_shift": float(ckpt.get("time_shift", report.get("time_shift", 1.0))),
        "time_shift_requested": float(ckpt.get(
            "time_shift_requested", report.get(
                "time_shift_requested", ckpt.get("time_shift", report.get("time_shift", 1.0))))),
        "time_shift_mode": str(ckpt.get(
            "time_shift_mode", report.get("time_shift_mode", "manual"))),
        "time_shift_effective_mode": str(ckpt.get(
            "time_shift_effective_mode", report.get(
                "time_shift_effective_mode",
                ckpt.get("time_shift_mode", report.get("time_shift_mode", "manual"))))),
        "time_shift_auto_policy": str(ckpt.get(
            "time_shift_auto_policy", report.get("time_shift_auto_policy", ""))),
        "time_shift_ref_dim": float(ckpt.get(
            "time_shift_ref_dim", report.get("time_shift_ref_dim", 1024.0))),
        "time_shift_dim_power": float(ckpt.get(
            "time_shift_dim_power", report.get("time_shift_dim_power", 0.5))),
        "latent_effective_dim": int(ckpt.get(
            "latent_effective_dim", report.get("latent_effective_dim", 0)) or 0),
        "time_shift_dim_scale": float(ckpt.get(
            "time_shift_dim_scale", report.get("time_shift_dim_scale", 1.0))),
        "flow_loss_weight": str(ckpt.get(
            "flow_loss_weight", report.get("flow_loss_weight", "none"))),
        "flow_noise_coupling": str(ckpt.get(
            "flow_noise_coupling", report.get("flow_noise_coupling", "random"))),
        "flow_noise_coupling_projections": int(ckpt.get(
            "flow_noise_coupling_projections",
            report.get("flow_noise_coupling_projections", 1)) or 1),
        "flow_endpoint_w": float(ckpt.get(
            "flow_endpoint_w", report.get("flow_endpoint_w", 0.0)) or 0.0),
        "flow_loss_weight_gamma": float(ckpt.get(
            "flow_loss_weight_gamma", report.get("flow_loss_weight_gamma", 5.0))),
        "flow_loss_weight_normalize": bool(ckpt.get(
            "flow_loss_weight_normalize", report.get("flow_loss_weight_normalize", True))),
        "sample_time_shift": float(ckpt.get(
            "sample_time_shift", report.get("sample_time_shift",
                                             report.get("time_shift", 1.0)))),
        "cfg_mode": str(ckpt.get("cfg_mode", report.get("cfg_mode", "standard"))
                        or "standard"),
        "cfg_schedule": str(ckpt.get(
            "cfg_schedule", report.get("cfg_schedule", "constant")) or "constant"),
        "sample_churn": float(ckpt.get(
            "sample_churn", report.get("sample_churn", 0.0)) or 0.0),
        "sample_churn_interval": tuple(ckpt.get(
            "sample_churn_interval",
            report.get("sample_churn_interval", DEFAULT_CHURN_INTERVAL))
            or DEFAULT_CHURN_INTERVAL),
        "sample_finite_guard": bool(ckpt.get(
            "sample_finite_guard", report.get("sample_finite_guard", False))),
        "sample_velocity_clip": float(ckpt.get(
            "sample_velocity_clip", report.get("sample_velocity_clip", 0.0)) or 0.0),
        "sample_latent_clip": float(ckpt.get(
            "sample_latent_clip", report.get("sample_latent_clip", 0.0)) or 0.0),
        "flow_ema_warmup": bool(ckpt.get("flow_ema_warmup",
                                         report.get("flow_ema_warmup", False))),
        "flow_ema_effective_decay": float(ckpt.get(
            "flow_ema_effective_decay", report.get("flow_ema_effective_decay", 0.0))),
        "latent_normalize_requested": str(ckpt.get(
            "latent_normalize_requested",
            report.get("latent_normalize_requested",
                       ckpt.get("latent_normalize",
                                report.get("latent_normalize", "none"))))),
        **latent_stats_report(latent_stats),
        "cond_mode": cond_mode,
        "data_mode": data_mode,
        "image_geometry_cond": flow_uses_image_geometry(flow),
        "image_manifest": ckpt.get("image_manifest", report.get("image_manifest", "")),
        "image_root": ckpt.get("image_root", report.get("image_root", "")),
        "image_split": ckpt.get("image_split", report.get("image_split", "")),
        "caption_max_len": caption_max_len,
        "caption_cond_source": caption_cond_source,
        "text_embedding_in_dim": int(text_embedding_in_dim),
        "text_pooled_embedding_in_dim": int(text_pooled_embedding_in_dim),
        "text_condition_dual_pooled_sequence": bool(
            caption_cond_source == "embedding" and text_pooled_embedding_in_dim > 0
            and text_pooled_embedding_in_dim != text_embedding_in_dim
        ),
        "cond_dim": cond_dim,
        "text_aligner": text_aligner is not None,
        "text_embed_dim": text_embed_dim,
        "image_feature_aligner": image_feature_aligner is not None,
        "image_embedding_in_dim": int(image_embedding_in_dim),
        "image_feature_embed_dim": int(image_feature_embed_dim),
        "image_quality_scorer": image_quality_scorer is not None,
        "image_quality_score_w": float(image_quality_score_w),
        "flow_quality_score_w": float(flow_quality_score_w),
        "flow_repa_aligner": flow_repa_aligner is not None,
        "flow_repa_embed_dim": int(flow_repa_embed_dim),
        "flow_repa_mode": str(flow_repa_mode),
        "flow_self_repa_aligner": flow_self_repa_aligner is not None,
        "flow_self_repa_embed_dim": int(flow_self_repa_embed_dim),
        "flow_self_repa_mode": str(flow_self_repa_mode),
        "flow_sra_mode": str(flow_sra_mode),
        "image_text_align_w": float(ckpt.get(
            "image_text_align_w", report.get("image_text_align_w", 0.0))),
        "flow_text_align_w": float(ckpt.get(
            "flow_text_align_w", report.get("flow_text_align_w", 0.0))),
        "image_feature_align_w": float(ckpt.get(
            "image_feature_align_w", report.get("image_feature_align_w", 0.0))),
        "flow_feature_align_w": float(ckpt.get(
            "flow_feature_align_w", report.get("flow_feature_align_w", 0.0))),
        "flow_repa_w": float(ckpt.get("flow_repa_w", report.get("flow_repa_w", 0.0))),
        "flow_repa_steps": int(ckpt.get(
            "flow_repa_steps", report.get("flow_repa_steps", 0)) or 0),
        "flow_self_repa_w": float(ckpt.get(
            "flow_self_repa_w", report.get("flow_self_repa_w", 0.0))),
        "flow_self_repa_steps": int(ckpt.get(
            "flow_self_repa_steps", report.get("flow_self_repa_steps", 0)) or 0),
        "flow_sra_w": float(ckpt.get("flow_sra_w", report.get("flow_sra_w", 0.0))),
        "flow_sra_steps": int(ckpt.get(
            "flow_sra_steps", report.get("flow_sra_steps", 0)) or 0),
        "flow_sra_time_gap": float(ckpt.get(
            "flow_sra_time_gap", report.get("flow_sra_time_gap", 0.25)) or 0.25),
    }


@torch.no_grad()
def image_record_sweep(ae, flow, records, cfg_scales=(1.0, 1.5), sample_steps_list=(4, 8),
                       n=128, batch=64, seed=10, size=32, device=DEV, conditioner=None,
                       prompt_vocab=None, caption_max_len=64, sample_method="euler",
                       sample_methods=None, cfg_rescales=(0.0,),
                       cfg_modes=None,
                       sample_schedule="linear", sample_schedules=None,
                       cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                       cfg_schedule="constant", cfg_schedules=None,
                       text_aligner=None, image_feature_aligner=None,
                       image_quality_scorer=None,
                       caption_cond_source="tokens", sample_time_shift=1.0,
                       time_shift=1.0,
                       sample_churn=0.0, sample_churns=None,
                       sample_churn_interval=DEFAULT_CHURN_INTERVAL,
                       text_guidance_weights=(0.0,),
                       text_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                       feature_guidance_weights=(0.0,),
                       feature_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                       quality_guidance_weights=(0.0,),
                       quality_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                       generated_eval_n=0, generated_eval_candidates_per_prompt=1,
                       sample_finite_guard=False, sample_velocity_clip=0.0,
                       sample_latent_clip=0.0):
    if sample_methods is None:
        sample_methods = (sample_method,)
    if sample_schedules is None:
        sample_schedules = (sample_schedule,)
    if cfg_modes is None:
        cfg_modes = ("standard",)
    if cfg_schedules is None:
        cfg_schedules = (cfg_schedule,)
    if sample_churns is None:
        sample_churns = (sample_churn,)
    cfg_rescales = tuple(float(x) for x in cfg_rescales)
    cfg_modes = tuple(str(x) for x in cfg_modes)
    bad_cfg_modes = sorted(set(cfg_modes) - set(CFG_MODES))
    if bad_cfg_modes:
        raise ValueError(f"unknown cfg mode(s): {','.join(bad_cfg_modes)}")
    cfg_schedules = tuple(validate_cfg_schedule(x) for x in cfg_schedules)
    sample_churns = tuple(float(x) for x in sample_churns)
    if any(x < 0.0 for x in sample_churns):
        raise ValueError("sample_churns must be non-negative")
    sample_churn_interval = validate_guidance_interval(
        sample_churn_interval, name="sample_churn_interval")
    text_guidance_weights = tuple(float(x) for x in (text_guidance_weights or (0.0,)))
    feature_guidance_weights = tuple(float(x) for x in (feature_guidance_weights or (0.0,)))
    quality_guidance_weights = tuple(float(x) for x in (quality_guidance_weights or (0.0,)))
    generated_eval_candidates_per_prompt = int(generated_eval_candidates_per_prompt)
    if generated_eval_candidates_per_prompt <= 0:
        raise ValueError("generated_eval_candidates_per_prompt must be positive")
    rows = []
    for cfg_scale in cfg_scales:
        for cfg_rescale in cfg_rescales:
            for cfg_mode in cfg_modes:
                for cfg_sched in cfg_schedules:
                    for sample_steps in sample_steps_list:
                        for method in sample_methods:
                            for schedule in sample_schedules:
                                for churn in sample_churns:
                                    for text_w in text_guidance_weights:
                                        for feature_w in feature_guidance_weights:
                                            for quality_w in quality_guidance_weights:
                                                row = evaluate_image_records(
                                                    ae, flow, records, n=n, batch=batch,
                                                    seed=seed, size=size, device=device,
                                                    conditioner=conditioner,
                                                    prompt_vocab=prompt_vocab,
                                                    caption_max_len=caption_max_len,
                                                    cfg_scale=float(cfg_scale),
                                                    cfg_rescale=float(cfg_rescale),
                                                    cfg_mode=cfg_mode,
                                                    cfg_schedule=cfg_sched,
                                                    sample_steps=int(sample_steps),
                                                    sample_method=method,
                                                    sample_schedule=schedule,
                                                    cfg_interval=cfg_interval,
                                                    text_aligner=text_aligner,
                                                    image_feature_aligner=image_feature_aligner,
                                                    image_quality_scorer=image_quality_scorer,
                                                    caption_cond_source=caption_cond_source,
                                                    sample_time_shift=sample_time_shift,
                                                    time_shift=time_shift,
                                                    sample_churn=float(churn),
                                                    sample_churn_interval=sample_churn_interval,
                                                    text_guidance_w=float(text_w),
                                                    text_guidance_interval=text_guidance_interval,
                                                    feature_guidance_w=float(feature_w),
                                                    feature_guidance_interval=(
                                                        feature_guidance_interval),
                                                    quality_guidance_w=float(quality_w),
                                                    quality_guidance_interval=(
                                                        quality_guidance_interval),
                                                    generated_eval_n=generated_eval_n,
                                                    generated_eval_candidates_per_prompt=(
                                                        generated_eval_candidates_per_prompt),
                                                    sample_finite_guard=sample_finite_guard,
                                                    sample_velocity_clip=sample_velocity_clip,
                                                    sample_latent_clip=sample_latent_clip)
                                                row["sweep_key"] = (
                                                    f"cfg={float(cfg_scale):g};"
                                                    f"rescale={float(cfg_rescale):g};"
                                                    f"cfgmode={cfg_mode};"
                                                    f"cfgschedule={cfg_sched};"
                                                    f"steps={int(sample_steps)};"
                                                    f"method={method};"
                                                    f"schedule={schedule};"
                                                    f"churn={float(churn):g};"
                                                    f"churnint="
                                                    f"{format_interval(sample_churn_interval)};"
                                                    f"shift={float(sample_time_shift):g};"
                                                    f"finiteguard="
                                                    f"{int(bool(sample_finite_guard))};"
                                                    f"vclip={float(sample_velocity_clip):g};"
                                                    f"zclip={float(sample_latent_clip):g};"
                                                    f"candidates="
                                                    f"{int(generated_eval_candidates_per_prompt)};"
                                                    f"text={float(text_w):g};"
                                                    f"feature={float(feature_w):g};"
                                                    f"quality={float(quality_w):g};"
                                                    f"cfgint={format_interval(cfg_interval)};"
                                                    f"textint="
                                                    f"{format_interval(text_guidance_interval)};"
                                                    f"featureint="
                                                    f"{format_interval(feature_guidance_interval)};"
                                                    f"qualityint="
                                                    f"{format_interval(quality_guidance_interval)}"
                                                )
                                                rows.append(row)
    return rows


SWEEP_METRICS = (
    "caption_sample_mse",
    "caption_retrieval_loss",
    "caption_retrieval_i2t_acc",
    "caption_retrieval_t2i_acc",
    "generated_caption_retrieval_loss",
    "generated_caption_retrieval_i2t_acc",
    "generated_caption_retrieval_t2i_acc",
    "image_feature_retrieval_loss",
    "image_feature_retrieval_i2f_acc",
    "image_feature_retrieval_f2i_acc",
    "generated_image_feature_retrieval_loss",
    "generated_image_feature_retrieval_i2f_acc",
    "generated_image_feature_retrieval_f2i_acc",
    "generated_image_feature_distribution_matched_cos",
    "generated_image_feature_distribution_mean_l2",
    "generated_image_feature_distribution_mean_sq",
    "generated_image_feature_distribution_mean_gap_l2",
    "generated_image_feature_distribution_cov_fro",
    "generated_image_feature_distribution_frechet",
    "generated_image_feature_distribution_mmd_rbf",
    "generated_image_feature_distribution_generated_pairwise_l2",
    "generated_image_feature_distribution_real_pairwise_l2",
    "generated_image_feature_distribution_diversity_l2_ratio",
    "generated_image_feature_distribution_nearest_real_l2",
    "generated_image_feature_distribution_nearest_generated_l2",
    "generated_image_feature_distribution_support_precision",
    "generated_image_feature_distribution_support_recall",
    "external_text_image_score_cos",
    "external_text_image_score_i2t_acc",
    "external_text_image_score_t2i_acc",
    "generated_external_text_image_score_cos",
    "generated_external_text_image_score_i2t_acc",
    "generated_external_text_image_score_t2i_acc",
    "image_quality_score_pred_mean",
    "generated_image_quality_score_pred_mean",
    "generated_eval_raw_candidates",
    "generated_eval_selection_component_count",
    "generated_eval_selection_score_mean",
    "generated_eval_selection_text_aligner_mean",
    "generated_eval_selection_image_feature_aligner_mean",
    "generated_eval_selection_quality_scorer_mean",
    "sample_center_target_mse",
    "sample_finite_frac",
    "sample_nonfinite_frac",
    "sample_trace_solver_steps",
    "sample_trace_solver_rejects",
    "sample_trace_solver_velocity_evals",
    "sample_trace_solver_error_max",
    "sample_trace_solver_error_mean",
    "sample_trace_solver_error_count",
    "sample_trace_cfg_scale_min",
    "sample_trace_cfg_scale_max",
    "sample_trace_cfg_scale_mean",
    "sample_trace_cfg_scale_count",
    "sample_trace_cfg_guided_evals",
    "sample_trace_min_velocity_finite_frac",
    "sample_trace_min_latent_finite_frac",
    "sample_trace_final_latent_finite_frac",
    "sample_trace_velocity_nonfinite_steps",
    "sample_trace_latent_nonfinite_steps",
    "sample_trace_finite_guard_events",
    "sample_trace_velocity_clip_events",
    "sample_trace_latent_clip_events",
    "sample_trace_max_velocity_rms",
    "sample_trace_max_velocity_abs",
    "sample_trace_max_latent_rms",
    "sample_trace_max_latent_abs",
    "sample_trace_final_latent_rms",
    "sample_trace_final_latent_abs",
    "sample_luminance_mean",
    "sample_luminance_std",
    "sample_rgb_std",
    "sample_dynamic_range",
    "sample_low_luminance_frac",
    "sample_high_luminance_frac",
    "sample_collapsed_frac",
    "sample_health_score",
    "recon_mse",
    "latent_velocity_mse",
    "latent_endpoint_mse",
    "latent_endpoint_consistency_mse",
    "latent_endpoint_time_gap",
)


EVAL_WEIGHT_MODES = ("raw", "ema", "auto")


def report_selection_key(report):
    inf = 1.0e30
    caption_mse = report.get("caption_sample_mse", inf)
    center_mse = report.get("sample_center_target_mse", report.get("recon_mse", inf))
    return (
        float(report.get("generated_external_text_image_score_cos", 0.0)),
        float(report.get("generated_image_quality_score_pred_mean", 0.0)),
        float(report.get("generated_eval_selection_score_mean", 0.0)),
        float(report.get("generated_image_feature_retrieval_i2f_acc", 0.0)),
        -float(report.get("generated_image_feature_distribution_frechet", inf)),
        -float(report.get("generated_image_feature_distribution_mmd_rbf", inf)),
        float(report.get("generated_image_feature_distribution_support_precision", 0.0)),
        float(report.get("generated_image_feature_distribution_support_recall", 0.0)),
        -abs(math.log(max(float(
            report.get("generated_image_feature_distribution_diversity_l2_ratio", 1.0)
        ), 1.0e-8))),
        float(report.get("sample_health_score", 0.0)),
        float(report.get("sample_finite_frac", 0.0)),
        -float(report.get("sample_nonfinite_frac", 1.0)),
        float(report.get("sample_trace_min_velocity_finite_frac", 0.0)),
        float(report.get("sample_trace_min_latent_finite_frac", 0.0)),
        float(report.get("sample_trace_final_latent_finite_frac", 0.0)),
        -float(report.get("sample_trace_velocity_nonfinite_steps", 1.0e9)),
        -float(report.get("sample_trace_latent_nonfinite_steps", 1.0e9)),
        float(report.get("sample_dynamic_range", 0.0)),
        float(report.get("sample_luminance_std", 0.0)),
        -float(report.get("sample_collapsed_frac", 1.0)),
        float(report.get("external_text_image_score_cos", 0.0)),
        float(report.get("image_feature_retrieval_i2f_acc", 0.0)),
        float(report.get("generated_caption_retrieval_i2t_acc", 0.0)),
        float(report.get("caption_retrieval_i2t_acc", 0.0)),
        -float(caption_mse),
        -float(center_mse),
        -float(report.get("latent_velocity_mse", inf)),
        -float(report.get("latent_endpoint_consistency_mse", inf)),
    )


def aggregate_selection_key(report):
    inf = 1.0e30
    caption_mse = report.get("caption_sample_mse_mean", inf)
    center_mse = report.get("sample_center_target_mse_mean", report.get("recon_mse_mean", inf))
    return (
        float(report.get("generated_external_text_image_score_cos_mean", 0.0)),
        float(report.get("generated_image_quality_score_pred_mean_mean", 0.0)),
        float(report.get("generated_eval_selection_score_mean_mean", 0.0)),
        float(report.get("generated_image_feature_retrieval_i2f_acc_mean", 0.0)),
        -float(report.get("generated_image_feature_distribution_frechet_mean", inf)),
        -float(report.get("generated_image_feature_distribution_mmd_rbf_mean", inf)),
        float(report.get("generated_image_feature_distribution_support_precision_mean", 0.0)),
        float(report.get("generated_image_feature_distribution_support_recall_mean", 0.0)),
        -abs(math.log(max(float(
            report.get("generated_image_feature_distribution_diversity_l2_ratio_mean", 1.0)
        ), 1.0e-8))),
        float(report.get("sample_health_score_mean", 0.0)),
        float(report.get("sample_finite_frac_mean", 0.0)),
        -float(report.get("sample_nonfinite_frac_mean", 1.0)),
        float(report.get("sample_trace_min_velocity_finite_frac_mean", 0.0)),
        float(report.get("sample_trace_min_latent_finite_frac_mean", 0.0)),
        float(report.get("sample_trace_final_latent_finite_frac_mean", 0.0)),
        -float(report.get("sample_trace_velocity_nonfinite_steps_mean", 1.0e9)),
        -float(report.get("sample_trace_latent_nonfinite_steps_mean", 1.0e9)),
        float(report.get("sample_dynamic_range_mean", 0.0)),
        float(report.get("sample_luminance_std_mean", 0.0)),
        -float(report.get("sample_collapsed_frac_mean", 1.0)),
        float(report.get("external_text_image_score_cos_mean", 0.0)),
        float(report.get("image_feature_retrieval_i2f_acc_mean", 0.0)),
        float(report.get("generated_caption_retrieval_i2t_acc_mean", 0.0)),
        float(report.get("caption_retrieval_i2t_acc_mean", 0.0)),
        -float(caption_mse),
        -float(center_mse),
        -float(report.get("latent_velocity_mse_mean", inf)),
        -float(report.get("latent_endpoint_consistency_mse_mean", inf)),
    )


def eval_report_summary(report):
    keys = (
        "eval_weight_mode",
        "cfg_scale",
        "cfg_rescale",
        "cfg_mode",
        "cfg_schedule",
        "cfg_interval",
        "sample_steps",
        "generated_eval_n",
        "generated_eval_n_requested",
        "generated_eval_candidates_per_prompt",
        "generated_eval_raw_candidates",
        "generated_eval_selection_component_count",
        "sample_time_shift",
        "time_shift",
        "sample_method",
        "sample_schedule",
        "sample_churn",
        "sample_churn_interval",
        "text_guidance_w",
        "text_guidance_interval",
        "feature_guidance_w",
        "feature_guidance_interval",
        "quality_guidance_w",
        "quality_guidance_interval",
        "caption_sample_mse",
        "caption_retrieval_i2t_acc",
        "caption_retrieval_t2i_acc",
        "generated_caption_retrieval_i2t_acc",
        "generated_caption_retrieval_t2i_acc",
        "image_feature_retrieval_i2f_acc",
        "image_feature_retrieval_f2i_acc",
        "generated_image_feature_retrieval_i2f_acc",
        "generated_image_feature_retrieval_f2i_acc",
        "generated_image_feature_distribution_matched_cos",
        "generated_image_feature_distribution_mean_l2",
        "generated_image_feature_distribution_mean_sq",
        "generated_image_feature_distribution_mean_gap_l2",
        "generated_image_feature_distribution_cov_fro",
        "generated_image_feature_distribution_frechet",
        "generated_image_feature_distribution_mmd_rbf",
        "generated_image_feature_distribution_generated_pairwise_l2",
        "generated_image_feature_distribution_real_pairwise_l2",
        "generated_image_feature_distribution_diversity_l2_ratio",
        "generated_image_feature_distribution_nearest_real_l2",
        "generated_image_feature_distribution_nearest_generated_l2",
        "generated_image_feature_distribution_support_precision",
        "generated_image_feature_distribution_support_recall",
        "external_text_image_score_cos",
        "generated_external_text_image_score_cos",
        "image_quality_score_pred_mean",
        "generated_image_quality_score_pred_mean",
        "sample_center_target_mse",
        "sample_finite_frac",
        "sample_nonfinite_frac",
        "sample_finite_guard",
        "sample_velocity_clip",
        "sample_latent_clip",
        "sample_trace_solver_steps",
        "sample_trace_solver_rejects",
        "sample_trace_solver_velocity_evals",
        "sample_trace_solver_error_max",
        "sample_trace_solver_error_mean",
        "sample_trace_solver_error_count",
        "sample_trace_cfg_schedule",
        "sample_trace_requested_cfg_scale",
        "sample_trace_cfg_scale_min",
        "sample_trace_cfg_scale_max",
        "sample_trace_cfg_scale_mean",
        "sample_trace_cfg_scale_count",
        "sample_trace_cfg_guided_evals",
        "sample_trace_min_velocity_finite_frac",
        "sample_trace_min_latent_finite_frac",
        "sample_trace_final_latent_finite_frac",
        "sample_trace_velocity_nonfinite_steps",
        "sample_trace_latent_nonfinite_steps",
        "sample_trace_finite_guard_events",
        "sample_trace_velocity_clip_events",
        "sample_trace_latent_clip_events",
        "sample_trace_max_velocity_rms",
        "sample_trace_max_velocity_abs",
        "sample_trace_max_latent_rms",
        "sample_trace_max_latent_abs",
        "sample_trace_final_latent_rms",
        "sample_trace_final_latent_abs",
        "sample_luminance_mean",
        "sample_luminance_std",
        "sample_rgb_std",
        "sample_dynamic_range",
        "sample_low_luminance_frac",
        "sample_high_luminance_frac",
        "sample_collapsed_frac",
        "sample_health_score",
        "recon_mse",
        "latent_velocity_mse",
        "latent_endpoint_mse",
        "latent_endpoint_consistency_mse",
        "latent_endpoint_time_gap",
    )
    return {k: report[k] for k in keys if k in report}


def aggregate_sweep_rows(rows):
    grouped = {}
    for row in rows:
        key = (float(row["cfg_scale"]), int(row["sample_steps"]),
               str(row.get("sample_method", "euler")),
               str(row.get("sample_schedule", "linear")),
               float(row.get("sample_time_shift", 1.0)),
               float(row.get("sample_churn", 0.0)),
               tuple(float(x) for x in row.get("sample_churn_interval",
                                                DEFAULT_CHURN_INTERVAL)),
               float(row.get("cfg_rescale", 0.0)),
               str(row.get("cfg_mode", "standard")),
               str(row.get("cfg_schedule", "constant")),
               tuple(float(x) for x in row.get("cfg_interval", DEFAULT_GUIDANCE_INTERVAL)),
               float(row.get("text_guidance_w", 0.0)),
               tuple(float(x) for x in row.get("text_guidance_interval",
                                                DEFAULT_GUIDANCE_INTERVAL)),
               float(row.get("feature_guidance_w", 0.0)),
               tuple(float(x) for x in row.get("feature_guidance_interval",
                                                DEFAULT_GUIDANCE_INTERVAL)),
               float(row.get("quality_guidance_w", 0.0)),
               tuple(float(x) for x in row.get("quality_guidance_interval",
                                                DEFAULT_GUIDANCE_INTERVAL)),
               int(row.get("generated_eval_candidates_per_prompt", 1)),
               bool(row.get("sample_finite_guard", False)),
               float(row.get("sample_velocity_clip", 0.0)),
               float(row.get("sample_latent_clip", 0.0)))
        grouped.setdefault(key, []).append(row)
    out = []
    for (cfg_scale, sample_steps, sample_method, sample_schedule, sample_time_shift,
         sample_churn, sample_churn_interval,
         cfg_rescale, cfg_mode, cfg_schedule,
         cfg_interval,
         text_guidance_w, text_guidance_interval,
         feature_guidance_w, feature_guidance_interval,
         quality_guidance_w, quality_guidance_interval,
         generated_eval_candidates_per_prompt,
         sample_finite_guard, sample_velocity_clip, sample_latent_clip), group in sorted(
             grouped.items()):
        agg = {
            "sweep_key": (
                f"cfg={cfg_scale:g};steps={sample_steps};method={sample_method};"
                f"schedule={sample_schedule};shift={sample_time_shift:g};"
                f"churn={sample_churn:g};churnint={format_interval(sample_churn_interval)};"
                f"rescale={cfg_rescale:g};cfgmode={cfg_mode};"
                f"cfgschedule={cfg_schedule};"
                f"text={text_guidance_w:g};"
                f"feature={feature_guidance_w:g};"
                f"quality={quality_guidance_w:g};"
                f"candidates={int(generated_eval_candidates_per_prompt)};"
                f"finiteguard={int(bool(sample_finite_guard))};"
                f"vclip={sample_velocity_clip:g};"
                f"zclip={sample_latent_clip:g};"
                f"cfgint={format_interval(cfg_interval)};"
                f"textint={format_interval(text_guidance_interval)};"
                f"featureint={format_interval(feature_guidance_interval)};"
                f"qualityint={format_interval(quality_guidance_interval)}"
            ),
            "cfg_scale": float(cfg_scale),
            "cfg_rescale": float(cfg_rescale),
            "cfg_mode": cfg_mode,
            "cfg_schedule": cfg_schedule,
            "cfg_interval": list(cfg_interval),
            "sample_steps": int(sample_steps),
            "sample_method": sample_method,
            "sample_schedule": sample_schedule,
            "sample_time_shift": float(sample_time_shift),
            "sample_churn": float(sample_churn),
            "sample_churn_interval": list(sample_churn_interval),
            "text_guidance_w": float(text_guidance_w),
            "text_guidance_interval": list(text_guidance_interval),
            "feature_guidance_w": float(feature_guidance_w),
            "feature_guidance_interval": list(feature_guidance_interval),
            "quality_guidance_w": float(quality_guidance_w),
            "quality_guidance_interval": list(quality_guidance_interval),
            "generated_eval_candidates_per_prompt": int(
                generated_eval_candidates_per_prompt),
            "sample_finite_guard": bool(sample_finite_guard),
            "sample_velocity_clip": float(sample_velocity_clip),
            "sample_latent_clip": float(sample_latent_clip),
            "runs": len(group),
            "eval_seeds": [int(r["eval_seed"]) for r in group if "eval_seed" in r],
        }
        for metric in SWEEP_METRICS:
            vals = np.asarray([float(r[metric]) for r in group if metric in r],
                              dtype=np.float64)
            if vals.size == 0:
                continue
            agg[f"{metric}_mean"] = float(vals.mean())
            agg[f"{metric}_std"] = float(vals.std())
            agg[f"{metric}_min"] = float(vals.min())
            agg[f"{metric}_max"] = float(vals.max())
        out.append(agg)
    return out


def evaluate_checkpoint(path, cfg_scales=(1.0, 1.5), sample_steps_list=(4, 8),
                        n=128, batch=64, seed=10, eval_seeds=None, size=32, device=DEV,
                        prefer_ema=True, weight_mode=None,
                        sample_method="euler", sample_methods=None,
                        sample_schedule="linear", sample_schedules=None,
                        cfg_rescales=(0.0,),
                        cfg_modes=None,
                        cfg_schedule="constant", cfg_schedules=None,
                        sample_churn=0.0, sample_churns=None,
                        sample_churn_interval=DEFAULT_CHURN_INTERVAL,
                        cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                        text_guidance_weights=(0.0,),
                        text_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                        feature_guidance_weights=(0.0,),
                        feature_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                        quality_guidance_weights=(0.0,),
                        quality_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                        eval_image_manifest="", eval_image_root="", eval_image_split="eval",
                        eval_image_min_aesthetic=None, eval_image_max_records=0,
                        sample_time_shift=None, generated_eval_n=0,
                        generated_eval_candidates_per_prompt=1,
                        sample_finite_guard=False, sample_velocity_clip=0.0,
                        sample_latent_clip=0.0):
    if weight_mode is None:
        weight_mode = "ema" if prefer_ema else "raw"
    if weight_mode not in EVAL_WEIGHT_MODES:
        raise ValueError(f"unknown checkpoint weight mode {weight_mode!r}")
    eval_seeds = tuple(eval_seeds) if eval_seeds is not None else (seed,)
    if cfg_modes is None:
        cfg_modes = ("standard",)
    cfg_modes = tuple(str(x) for x in cfg_modes)
    bad_cfg_modes = sorted(set(cfg_modes) - set(CFG_MODES))
    if bad_cfg_modes:
        raise ValueError(f"unknown cfg mode(s): {','.join(bad_cfg_modes)}")
    if cfg_schedules is None:
        cfg_schedules = (cfg_schedule,)
    cfg_schedules = tuple(validate_cfg_schedule(x) for x in cfg_schedules)
    text_guidance_weights = tuple(float(x) for x in (text_guidance_weights or (0.0,)))
    feature_guidance_weights = tuple(float(x) for x in (feature_guidance_weights or (0.0,)))
    quality_guidance_weights = tuple(float(x) for x in (quality_guidance_weights or (0.0,)))
    if sample_churns is None:
        sample_churns = (sample_churn,)
    sample_churns = tuple(float(x) for x in sample_churns)
    if any(x < 0.0 for x in sample_churns):
        raise ValueError("sample_churns must be non-negative")
    sample_churn_interval = validate_guidance_interval(
        sample_churn_interval, name="sample_churn_interval")
    generated_eval_n = int(generated_eval_n)
    generated_eval_candidates_per_prompt = int(generated_eval_candidates_per_prompt)
    if any(w < 0.0 for w in text_guidance_weights):
        raise ValueError("text guidance weights must be non-negative")
    if any(w < 0.0 for w in feature_guidance_weights):
        raise ValueError("feature guidance weights must be non-negative")
    if any(w < 0.0 for w in quality_guidance_weights):
        raise ValueError("quality guidance weights must be non-negative")
    if generated_eval_n < 0:
        raise ValueError("generated_eval_n must be non-negative")
    if generated_eval_candidates_per_prompt <= 0:
        raise ValueError("generated_eval_candidates_per_prompt must be positive")
    sample_velocity_clip = float(sample_velocity_clip)
    sample_latent_clip = float(sample_latent_clip)
    if sample_velocity_clip < 0.0:
        raise ValueError("sample_velocity_clip must be non-negative")
    if sample_latent_clip < 0.0:
        raise ValueError("sample_latent_clip must be non-negative")

    def run_mode(mode):
        ae, flow, conditioner, prompt_vocab, _prompt_templates, meta = load_checkpoint(
            path, device=device, prefer_ema=(mode == "ema"))
        actual_mode = "ema" if meta["ema_loaded"] else "raw"
        eval_size = image_hw(size, default=meta.get("image_size", 32) or 32)
        actual_sample_time_shift = (
            float(meta.get("sample_time_shift", meta.get("time_shift", 1.0)))
            if sample_time_shift is None else float(sample_time_shift)
        )
        if actual_sample_time_shift <= 0.0:
            raise ValueError("sample_time_shift must be positive")
        train_time_shift = float(meta.get("time_shift", actual_sample_time_shift))
        manifest_path = eval_image_manifest or meta.get("image_manifest", "")
        if not manifest_path:
            raise ValueError("checkpoint eval requires an image manifest")
        split = eval_image_split or "eval"
        image_root = eval_image_root or meta.get("image_root", "")
        records = read_image_manifest(
            manifest_path, root=image_root, split=split,
            min_aesthetic=eval_image_min_aesthetic,
            max_records=eval_image_max_records)
        rows = []
        for eval_seed in eval_seeds:
            for row in image_record_sweep(
                    ae, flow, records, cfg_scales=cfg_scales,
                    sample_steps_list=sample_steps_list, n=n, batch=batch,
                    seed=int(eval_seed), size=eval_size, device=device,
                    conditioner=conditioner, prompt_vocab=prompt_vocab,
                    caption_max_len=meta["caption_max_len"],
                    sample_method=sample_method, sample_methods=sample_methods,
                    sample_schedule=sample_schedule, sample_schedules=sample_schedules,
                    sample_churn=sample_churn,
                    sample_churns=sample_churns,
                    sample_churn_interval=sample_churn_interval,
                    cfg_rescales=cfg_rescales,
                    cfg_modes=cfg_modes,
                    cfg_schedule=cfg_schedule,
                    cfg_schedules=cfg_schedules,
                    cfg_interval=cfg_interval,
                    text_aligner=getattr(flow, "text_aligner", None),
                    image_feature_aligner=getattr(flow, "image_feature_aligner", None),
                    image_quality_scorer=getattr(flow, "image_quality_scorer", None),
                    caption_cond_source=meta["caption_cond_source"],
                    sample_time_shift=actual_sample_time_shift,
                    time_shift=train_time_shift,
                    text_guidance_weights=text_guidance_weights,
                    text_guidance_interval=text_guidance_interval,
                    feature_guidance_weights=feature_guidance_weights,
                    feature_guidance_interval=feature_guidance_interval,
                    quality_guidance_weights=quality_guidance_weights,
                    quality_guidance_interval=quality_guidance_interval,
                    generated_eval_n=generated_eval_n,
                    generated_eval_candidates_per_prompt=(
                        generated_eval_candidates_per_prompt),
                    sample_finite_guard=sample_finite_guard,
                    sample_velocity_clip=sample_velocity_clip,
                    sample_latent_clip=sample_latent_clip):
                row["eval_seed"] = int(eval_seed)
                row["checkpoint_weight_mode"] = actual_mode
                rows.append(row)
        aggregate = aggregate_sweep_rows(rows)
        best = max(aggregate, key=aggregate_selection_key)
        report = {
            "experiment": "image_latent_manifest_sampler_sweep",
            **meta,
            "checkpoint_weight_mode": actual_mode,
            "requested_checkpoint_weight_mode": weight_mode,
            "selected_checkpoint_weights": actual_mode,
            "n": int(n),
            "image_size": image_size_value(eval_size),
            "image_h": int(eval_size[0]),
            "image_w": int(eval_size[1]),
            "sample_method": sample_method,
            "sample_methods": list(sample_methods or (sample_method,)),
            "sample_schedule": sample_schedule,
            "sample_schedules": list(sample_schedules or (sample_schedule,)),
            "cfg_modes": list(cfg_modes),
            "cfg_schedule": cfg_schedule,
            "cfg_schedules": list(cfg_schedules),
            "sample_churn": float(sample_churn),
            "sample_churns": [float(x) for x in sample_churns],
            "sample_churn_interval": list(sample_churn_interval),
            "sample_finite_guard": bool(sample_finite_guard),
            "sample_velocity_clip": float(sample_velocity_clip),
            "sample_latent_clip": float(sample_latent_clip),
            "cfg_rescales": [float(x) for x in cfg_rescales],
            "sample_time_shift": float(actual_sample_time_shift),
            "time_shift": float(train_time_shift),
            "cfg_interval": list(validate_guidance_interval(cfg_interval)),
            "text_guidance_weights": [float(x) for x in text_guidance_weights],
            "text_guidance_interval": list(validate_guidance_interval(
                text_guidance_interval, name="text_guidance_interval")),
            "feature_guidance_weights": [float(x) for x in feature_guidance_weights],
            "feature_guidance_interval": list(validate_guidance_interval(
                feature_guidance_interval, name="feature_guidance_interval")),
            "quality_guidance_weights": [float(x) for x in quality_guidance_weights],
            "quality_guidance_interval": list(validate_guidance_interval(
                quality_guidance_interval, name="quality_guidance_interval")),
            "eval_seeds": [int(s) for s in eval_seeds],
            "eval_image_manifest": manifest_path,
            "eval_image_root": image_root,
            "eval_image_split": split,
            "eval_image_min_aesthetic": (
                float(eval_image_min_aesthetic)
                if eval_image_min_aesthetic is not None else None
            ),
            "eval_image_max_records": int(eval_image_max_records),
            "rows": rows,
            "aggregate": aggregate,
            "best": best,
        }
        report.update(summarize_records(records))
        return report

    if weight_mode != "auto":
        return run_mode(weight_mode)

    candidates = {"raw": run_mode("raw")}
    if candidates["raw"]["ema_available"]:
        candidates["ema"] = run_mode("ema")
    selected = max(candidates, key=lambda mode: aggregate_selection_key(candidates[mode]["best"]))
    report = candidates[selected]
    report["requested_checkpoint_weight_mode"] = "auto"
    report["selected_checkpoint_weights"] = selected
    report["weight_eval_candidates"] = {
        mode: {
            "checkpoint_weight_mode": candidate["checkpoint_weight_mode"],
            "ema_loaded": candidate["ema_loaded"],
            "best": candidate["best"],
        }
        for mode, candidate in candidates.items()
    }
    return report


def selected_grid_settings(report, fallback_cfg=1.0, fallback_cfg_rescale=0.0, fallback_steps=4,
                           fallback_method="euler",
                           fallback_cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                           fallback_cfg_schedule="constant",
                           fallback_cfg_mode="standard",
                           fallback_sample_time_shift=1.0,
                           fallback_sample_schedule="linear",
                           fallback_sample_churn=0.0,
                           fallback_sample_churn_interval=DEFAULT_CHURN_INTERVAL,
                           fallback_sample_finite_guard=False,
                           fallback_sample_velocity_clip=0.0,
                           fallback_sample_latent_clip=0.0):
    best = report.get("best") or report
    return {
        "cfg_scale": float(best.get("cfg_scale", fallback_cfg)),
        "cfg_rescale": float(best.get("cfg_rescale", fallback_cfg_rescale)),
        "cfg_mode": str(best.get("cfg_mode", fallback_cfg_mode)),
        "cfg_schedule": str(best.get("cfg_schedule", fallback_cfg_schedule)),
        "cfg_interval": tuple(best.get("cfg_interval", fallback_cfg_interval)),
        "sample_steps": int(best.get("sample_steps", fallback_steps)),
        "sample_time_shift": float(best.get("sample_time_shift",
                                            fallback_sample_time_shift)),
        "sample_method": str(best.get("sample_method", fallback_method)),
        "sample_schedule": str(best.get("sample_schedule", fallback_sample_schedule)),
        "sample_churn": float(best.get("sample_churn", fallback_sample_churn)),
        "sample_churn_interval": tuple(best.get(
            "sample_churn_interval", fallback_sample_churn_interval)),
        "sample_finite_guard": bool(best.get(
            "sample_finite_guard", fallback_sample_finite_guard)),
        "sample_velocity_clip": float(best.get(
            "sample_velocity_clip", fallback_sample_velocity_clip)),
        "sample_latent_clip": float(best.get(
            "sample_latent_clip", fallback_sample_latent_clip)),
    }


def train_latent_flow(ae_steps=200, flow_steps=200, batch=64, latent_ch=16, hidden=64,
                      lr=2e-4, seed=0, size=32, device=DEV, flow_arch="conv",
                      dit_depth=3, dit_heads=4, cond_drop=0.0, cfg_scale=1.0,
                      cfg_rescale=0.0, cfg_mode="standard",
                      flow_boundary_mode="none",
                      dit_head_width_mult=1, latent_max_tokens=256,
                      dit_pos_embed="learned",
                      dit_mlp="gelu",
                      flow_time_embed="scalar", flow_time_embed_dim=0,
                      latent_patch_size=1,
                      flow_checkpoint_blocks=False,
                      ae_arch="semantic", latent_downsample=4, ae_res_blocks=1,
                      ae_hf_model="", ae_hf_subfolder="", ae_hf_scaling_factor=0.0,
                      ae_recon_loss="mse", ae_grad_w=0.0, ae_ms_w=0.0, ae_fft_w=0.0,
                      ae_latent_reg_w=0.0,
                      image_text_align_w=0.0, flow_text_align_w=0.0, text_embed_dim=128,
                      image_feature_align_w=0.0, flow_feature_align_w=0.0,
                      image_feature_embed_dim=128,
                      flow_repa_w=0.0, flow_repa_steps=0, flow_repa_embed_dim=128,
                      flow_repa_mode="pooled",
                      flow_self_repa_w=0.0, flow_self_repa_steps=0,
                      flow_self_repa_embed_dim=128, flow_self_repa_mode="pooled",
                      flow_sra_w=0.0, flow_sra_steps=0, flow_sra_time_gap=0.25,
                      flow_sra_mode="token",
                      sample_steps=4,
                      flow_consistency_w=0.0, flow_endpoint_w=0.0,
                      flow_distill_steps=0, flow_distill_w=1.0,
                      flow_distill_time_gap=0.25, flow_distill_teacher="auto",
                      flow_guidance_distill_w=0.0,
                      flow_guidance_distill_cfg_scale=1.5,
                      flow_guidance_distill_cfg_rescale=0.0,
                      cond_mode="text", text_cond_dim=0,
                      size_buckets=(),
                      size_curriculum_frac=0.0,
                      image_manifest="", image_root="", image_split="train",
                      image_min_aesthetic=None, image_max_records=0,
                      image_quality_weight=0.0,
                      image_source_weights="",
                      image_quality_score_w=0.0, flow_quality_score_w=0.0,
                      quality_score_steps=0,
                      image_quality_rank_w=0.0, quality_score_rank_w=0.0,
                      flow_quality_rank_w=0.0, quality_rank_margin=0.0,
                      image_preference_manifest="", image_preference_root="",
                      image_preference_max_pairs=0, image_preference_w=0.0,
                      flow_preference_w=0.0, flow_preference_margin=0.0,
                      flow_preference_batch=0,
                      image_geometry_cond=False,
                      caption_vocab_max=8192, caption_max_len=64,
                      caption_cond_source="tokens",
                      image_crop_mode="center", image_hflip_prob=0.0,
                      dit_qk_norm=False, dit_attn_impl="auto",
                      time_sampling="uniform",
                      time_logit_mean=0.0, time_logit_std=1.0,
                      time_mode_scale=DEFAULT_TIME_MODE_SCALE, time_shift=1.0,
                      time_curriculum_frac=0.0,
                      time_adaptive_bins=32, time_adaptive_momentum=0.95,
                      time_adaptive_uniform_mix=0.05,
                      time_adaptive_min_prob=0.001,
                      time_adaptive_loss_power=1.0,
                      time_shift_mode="auto", time_shift_ref_dim=1024.0,
                      time_shift_dim_power=0.5,
                      flow_noise_coupling="random", flow_noise_coupling_projections=1,
                      flow_loss_weight="none", flow_loss_weight_gamma=5.0,
                      flow_loss_weight_normalize=True,
                      latent_normalize="auto", latent_stat_samples=512,
                      cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                      cfg_schedule="constant",
                      sample_method="euler",
                      sample_schedule="linear",
                      sample_churn=0.0,
                      sample_churn_interval=DEFAULT_CHURN_INTERVAL,
                      sample_finite_guard=False, sample_velocity_clip=0.0,
                      sample_latent_clip=0.0,
                      flow_ema_decay=0.0, flow_ema_warmup=True, eval_with_ema=True,
                      eval_weight_mode="auto",
                      eval_generated_candidates_per_prompt=1,
                      train_precision="fp32", ae_accum_steps=1, flow_accum_steps=1,
                      grad_clip=0.0,
                      flow_cache_latents=False, flow_cache_records=0, flow_cache_batch=64,
                      flow_cache_dir="", flow_cache_shard_size=1024, flow_cache_dtype="fp32",
                      flow_cache_max_loaded_shards=0,
                      return_conditioner=False, return_ema=False, return_aligner=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    size = image_hw(size, default=32)
    requested_size_buckets = normalize_image_size_buckets(size_buckets)
    if cond_mode != "text":
        raise ValueError(f"unknown condition mode {cond_mode!r}")
    if ae_arch not in AE_ARCHES:
        raise ValueError(f"unknown autoencoder architecture {ae_arch!r}")
    if ae_arch == "hf-vae" and not image_manifest:
        raise ValueError("ae_arch='hf-vae' requires image_manifest training")
    if ae_arch == "hf-vae" and not ae_hf_model:
        raise ValueError("ae_arch='hf-vae' requires ae_hf_model")
    if cfg_rescale < 0.0 or cfg_rescale > 1.0:
        raise ValueError("cfg_rescale must be in [0, 1]")
    cfg_mode = str(cfg_mode)
    if cfg_mode not in CFG_MODES:
        raise ValueError(f"unknown cfg_mode {cfg_mode!r}")
    cfg_schedule = validate_cfg_schedule(cfg_schedule)
    flow_boundary_mode = str(flow_boundary_mode or "none")
    if flow_boundary_mode not in FLOW_BOUNDARY_MODES:
        raise ValueError(f"unknown flow boundary mode {flow_boundary_mode!r}")
    image_records = None
    if image_manifest:
        if cond_mode != "text":
            raise ValueError("image manifests require cond_mode='text'")
        image_records = read_image_manifest(
            image_manifest, root=image_root, split=image_split,
            min_aesthetic=image_min_aesthetic, max_records=image_max_records)
        flow_repa_mode = str(flow_repa_mode)
        if flow_repa_mode not in FLOW_REPA_MODES:
            raise ValueError(f"unknown flow_repa_mode {flow_repa_mode!r}")
        caption_cond_source, text_embedding_in_dim = resolve_caption_cond_source(
            caption_cond_source, image_records)
        text_pooled_embedding_in_dim = (
            infer_text_pooled_embedding_dim(image_records)
            if caption_cond_source == "embedding" else 0
        )
        image_embedding_in_dim = (
            infer_image_embedding_dim(image_records)
            if (image_feature_align_w > 0.0 or flow_feature_align_w > 0.0
                or flow_repa_w > 0.0) else 0
        )
        if (image_feature_align_w > 0.0 or flow_feature_align_w > 0.0 or flow_repa_w > 0.0
                ) and image_embedding_in_dim <= 0:
            raise ValueError("image feature/REPA alignment requires image_embedding rows")
        if flow_repa_w > 0.0 and flow_repa_mode in ("token", "both"):
            if not records_have_image_embedding_sequences(image_records):
                raise ValueError(
                    f"flow_repa_mode={flow_repa_mode!r} requires image_embedding_sequence rows")
    else:
        raise ValueError("image_latent training requires image_manifest data")
    flow_self_repa_mode = str(flow_self_repa_mode or "pooled")
    if flow_self_repa_mode not in FLOW_REPA_MODES:
        raise ValueError(f"unknown flow_self_repa_mode {flow_self_repa_mode!r}")
    flow_sra_mode = str(flow_sra_mode or "token")
    if flow_sra_mode not in FLOW_REPA_MODES:
        raise ValueError(f"unknown flow_sra_mode {flow_sra_mode!r}")
    train_size_buckets = requested_size_buckets or (size,)
    bucket_records, bucket_missing_dims = bucket_records_by_aspect(
        image_records, train_size_buckets)
    if time_sampling not in TIME_SAMPLINGS:
        raise ValueError(f"unknown time sampling mode {time_sampling!r}")
    time_mode_scale = float(time_mode_scale)
    if time_mode_scale < 0.0 or time_mode_scale >= MAX_TIME_MODE_SCALE:
        raise ValueError(f"time_mode_scale must be in [0, {MAX_TIME_MODE_SCALE})")
    time_curriculum_frac = float(time_curriculum_frac)
    if time_curriculum_frac < 0.0 or time_curriculum_frac > 1.0:
        raise ValueError("time_curriculum_frac must be in [0, 1]")
    size_curriculum_frac = float(size_curriculum_frac)
    if size_curriculum_frac < 0.0 or size_curriculum_frac > 1.0:
        raise ValueError("size_curriculum_frac must be in [0, 1]")
    if time_shift <= 0.0:
        raise ValueError("time_shift must be positive")
    if time_shift_mode not in TIME_SHIFT_MODES:
        raise ValueError(f"unknown time shift mode {time_shift_mode!r}")
    if time_shift_ref_dim <= 0.0:
        raise ValueError("time_shift_ref_dim must be positive")
    if flow_loss_weight not in FLOW_LOSS_WEIGHTS:
        raise ValueError(f"unknown flow loss weighting mode {flow_loss_weight!r}")
    flow_noise_coupling = str(flow_noise_coupling)
    if flow_noise_coupling not in FLOW_NOISE_COUPLINGS:
        raise ValueError(f"unknown flow noise coupling {flow_noise_coupling!r}")
    flow_noise_coupling_projections = int(flow_noise_coupling_projections)
    if flow_noise_coupling_projections <= 0:
        raise ValueError("flow_noise_coupling_projections must be positive")
    if flow_loss_weight_gamma <= 0.0:
        raise ValueError("flow_loss_weight_gamma must be positive")
    time_adaptive_bins = int(time_adaptive_bins)
    time_adaptive_momentum = float(time_adaptive_momentum)
    time_adaptive_uniform_mix = float(time_adaptive_uniform_mix)
    time_adaptive_min_prob = float(time_adaptive_min_prob)
    time_adaptive_loss_power = float(time_adaptive_loss_power)
    if time_adaptive_bins <= 1:
        raise ValueError("time_adaptive_bins must be > 1")
    if time_adaptive_momentum < 0.0 or time_adaptive_momentum >= 1.0:
        raise ValueError("time_adaptive_momentum must be in [0, 1)")
    if time_adaptive_uniform_mix < 0.0 or time_adaptive_uniform_mix > 1.0:
        raise ValueError("time_adaptive_uniform_mix must be in [0, 1]")
    if time_adaptive_min_prob < 0.0 or time_adaptive_min_prob >= 1.0:
        raise ValueError("time_adaptive_min_prob must be in [0, 1)")
    if time_adaptive_loss_power <= 0.0:
        raise ValueError("time_adaptive_loss_power must be positive")
    requested_latent_normalize = str(latent_normalize or "none")
    effective_latent_normalize = resolve_latent_normalize(
        requested_latent_normalize, image_records=image_records, ae_arch=ae_arch)
    if image_quality_weight < 0.0:
        raise ValueError("image_quality_weight must be non-negative")
    source_weight_map = parse_source_weight_spec(image_source_weights)
    if (image_quality_score_w < 0.0 or flow_quality_score_w < 0.0
            or image_quality_rank_w < 0.0 or quality_score_rank_w < 0.0
            or flow_quality_rank_w < 0.0):
        raise ValueError("quality score weights must be non-negative")
    if quality_rank_margin < 0.0:
        raise ValueError("quality_rank_margin must be non-negative")
    quality_score_steps = int(quality_score_steps)
    if quality_score_steps < 0:
        raise ValueError("quality_score_steps must be non-negative")
    image_preference_max_pairs = int(image_preference_max_pairs)
    if image_preference_max_pairs < 0:
        raise ValueError("image_preference_max_pairs must be non-negative")
    image_preference_w = float(image_preference_w)
    if image_preference_w < 0.0:
        raise ValueError("image_preference_w must be non-negative")
    if image_preference_w > 0.0 and not image_preference_manifest:
        raise ValueError("image_preference_w requires image_preference_manifest")
    if image_preference_w > 0.0 and quality_score_steps <= 0:
        raise ValueError("image_preference_w requires quality_score_steps > 0")
    flow_preference_w = float(flow_preference_w)
    if flow_preference_w < 0.0:
        raise ValueError("flow_preference_w must be non-negative")
    flow_preference_margin = float(flow_preference_margin)
    if flow_preference_margin < 0.0:
        raise ValueError("flow_preference_margin must be non-negative")
    flow_preference_batch = int(flow_preference_batch)
    if flow_preference_batch < 0:
        raise ValueError("flow_preference_batch must be non-negative")
    if flow_preference_w > 0.0 and not image_preference_manifest:
        raise ValueError("flow_preference_w requires image_preference_manifest")
    if quality_score_rank_w > 0.0 and quality_score_steps <= 0:
        raise ValueError("quality_score_rank_w requires quality_score_steps > 0")
    if ((flow_quality_score_w > 0.0 or flow_quality_rank_w > 0.0)
            and image_quality_score_w <= 0.0 and image_quality_rank_w <= 0.0
            and quality_score_steps <= 0):
        raise ValueError(
            "flow quality losses require image_quality_score_w/image_quality_rank_w > 0 "
            "or quality_score_steps > 0")
    if image_crop_mode not in ("center", "random", "none", "pad"):
        raise ValueError(f"unknown image crop mode {image_crop_mode!r}")
    if image_hflip_prob < 0.0 or image_hflip_prob > 1.0:
        raise ValueError("image_hflip_prob must be in [0, 1]")
    if ae_recon_loss not in AE_RECON_LOSSES:
        raise ValueError(f"unknown AE reconstruction loss {ae_recon_loss!r}")
    if ae_grad_w < 0.0 or ae_ms_w < 0.0 or ae_fft_w < 0.0 or ae_latent_reg_w < 0.0:
        raise ValueError("AE reconstruction weights must be non-negative")
    if image_text_align_w < 0.0 or flow_text_align_w < 0.0:
        raise ValueError("image/text alignment weights must be non-negative")
    if image_feature_align_w < 0.0 or flow_feature_align_w < 0.0:
        raise ValueError("image feature alignment weights must be non-negative")
    if flow_repa_w < 0.0:
        raise ValueError("flow_repa_w must be non-negative")
    if flow_repa_steps < 0:
        raise ValueError("flow_repa_steps must be non-negative")
    if flow_self_repa_w < 0.0:
        raise ValueError("flow_self_repa_w must be non-negative")
    if flow_self_repa_steps < 0:
        raise ValueError("flow_self_repa_steps must be non-negative")
    if flow_sra_w < 0.0:
        raise ValueError("flow_sra_w must be non-negative")
    if flow_sra_steps < 0:
        raise ValueError("flow_sra_steps must be non-negative")
    if flow_sra_time_gap <= 0.0 or flow_sra_time_gap > 1.0:
        raise ValueError("flow_sra_time_gap must be in (0, 1]")
    if text_embed_dim <= 0:
        raise ValueError("text_embed_dim must be positive")
    if image_feature_embed_dim <= 0:
        raise ValueError("image_feature_embed_dim must be positive")
    if flow_repa_embed_dim <= 0:
        raise ValueError("flow_repa_embed_dim must be positive")
    if flow_self_repa_embed_dim <= 0:
        raise ValueError("flow_self_repa_embed_dim must be positive")
    amp_cfg = amp_config(device, train_precision)
    ae_accum_steps = max(1, int(ae_accum_steps))
    flow_accum_steps = max(1, int(flow_accum_steps))
    if grad_clip < 0.0:
        raise ValueError("grad_clip must be non-negative")
    if flow_consistency_w < 0.0:
        raise ValueError("flow_consistency_w must be non-negative")
    if flow_endpoint_w < 0.0:
        raise ValueError("flow_endpoint_w must be non-negative")
    flow_distill_steps = int(flow_distill_steps)
    if flow_distill_steps < 0:
        raise ValueError("flow_distill_steps must be non-negative")
    if flow_distill_w < 0.0:
        raise ValueError("flow_distill_w must be non-negative")
    if flow_distill_time_gap <= 0.0 or flow_distill_time_gap > 1.0:
        raise ValueError("flow_distill_time_gap must be in (0, 1]")
    if flow_guidance_distill_w < 0.0:
        raise ValueError("flow_guidance_distill_w must be non-negative")
    if flow_guidance_distill_cfg_scale < 1.0:
        raise ValueError("flow_guidance_distill_cfg_scale must be >= 1")
    if flow_guidance_distill_cfg_rescale < 0.0 or flow_guidance_distill_cfg_rescale > 1.0:
        raise ValueError("flow_guidance_distill_cfg_rescale must be in [0, 1]")
    flow_distill_teacher = str(flow_distill_teacher)
    if flow_distill_teacher not in FLOW_DISTILL_TEACHERS:
        raise ValueError(f"unknown flow distillation teacher {flow_distill_teacher!r}")
    if (flow_distill_steps > 0 and flow_distill_w > 0.0
            and flow_distill_teacher == "ema" and flow_ema_decay <= 0.0):
        raise ValueError("flow_distill_teacher='ema' requires flow_ema_decay > 0")
    flow_cache_dir = str(flow_cache_dir or "")
    flow_cache_latents = bool(flow_cache_latents or flow_cache_dir)
    if flow_cache_records < 0 or flow_cache_batch <= 0:
        raise ValueError("flow cache record/batch settings must be non-negative/positive")
    if flow_cache_shard_size <= 0:
        raise ValueError("flow cache shard size must be positive")
    flow_cache_dtype = str(flow_cache_dtype)
    latent_cache_torch_dtype(flow_cache_dtype)
    flow_cache_max_loaded_shards = int(flow_cache_max_loaded_shards)
    if flow_cache_max_loaded_shards < 0:
        raise ValueError("flow_cache_max_loaded_shards must be non-negative")
    if sample_method not in SAMPLE_METHODS:
        raise ValueError(f"unknown sample method {sample_method!r}")
    if sample_schedule not in SAMPLE_SCHEDULES:
        raise ValueError(f"unknown sample schedule {sample_schedule!r}")
    sample_churn = float(sample_churn)
    if sample_churn < 0.0:
        raise ValueError("sample_churn must be non-negative")
    sample_velocity_clip = float(sample_velocity_clip)
    sample_latent_clip = float(sample_latent_clip)
    if sample_velocity_clip < 0.0:
        raise ValueError("sample_velocity_clip must be non-negative")
    if sample_latent_clip < 0.0:
        raise ValueError("sample_latent_clip must be non-negative")
    sample_churn_interval = validate_guidance_interval(
        sample_churn_interval, name="sample_churn_interval")
    eval_generated_candidates_per_prompt = int(eval_generated_candidates_per_prompt)
    if eval_generated_candidates_per_prompt <= 0:
        raise ValueError("eval_generated_candidates_per_prompt must be positive")
    cfg_interval = validate_guidance_interval(cfg_interval, name="cfg_interval")
    if dit_head_width_mult <= 0:
        raise ValueError("dit_head_width_mult must be positive")
    if dit_attn_impl not in MMDIT_ATTN_IMPLS:
        raise ValueError(f"unknown MM-DiT attention implementation {dit_attn_impl!r}")
    dit_pos_embed = str(dit_pos_embed)
    if dit_pos_embed not in DIT_POS_EMBEDS:
        raise ValueError(f"unknown DiT positional embedding {dit_pos_embed!r}")
    dit_mlp = str(dit_mlp)
    if dit_mlp not in DIT_MLPS:
        raise ValueError(f"unknown DiT MLP {dit_mlp!r}")
    flow_time_embed = str(flow_time_embed or "scalar")
    if flow_time_embed not in FLOW_TIME_EMBEDS:
        raise ValueError(f"unknown flow time embedding {flow_time_embed!r}")
    flow_time_embed_dim = int(flow_time_embed_dim or 0)
    if flow_time_embed_dim < 0:
        raise ValueError("flow_time_embed_dim must be non-negative")
    flow_checkpoint_blocks = bool(flow_checkpoint_blocks)
    if latent_max_tokens <= 0:
        raise ValueError("latent_max_tokens must be positive")
    latent_patch_size = int(latent_patch_size)
    if latent_patch_size <= 0:
        raise ValueError("latent_patch_size must be positive")
    if latent_patch_size != 1 and flow_arch == "conv":
        raise ValueError("latent_patch_size is only supported by DiT/CrossDiT/MM-DiT flows")
    if flow_ema_decay < 0.0 or flow_ema_decay >= 1.0:
        raise ValueError("flow_ema_decay must be in [0, 1)")
    if eval_weight_mode not in EVAL_WEIGHT_MODES:
        raise ValueError(f"unknown eval weight mode {eval_weight_mode!r}")
    prompt_vocab = None
    image_sample_weights = None
    image_record_weights = None
    image_source_report = {
        "image_source_weights": dict(sorted(source_weight_map.items())),
        "image_source_weighted": False,
        "image_source_weight_records": 0,
        "image_source_weight_default": float(source_weight_map.get("*", 1.0)),
        "image_source_counts": {},
    }
    image_quality_report = {
        "image_quality_weight": float(image_quality_weight),
        "image_quality_weighted": False,
        "image_quality_weight_source": "",
        "image_quality_weight_records": 0,
        "image_quality_weight_missing": 0,
    }
    quality_weights, image_quality_report = quality_sampling_weights(
        image_records, image_quality_weight)
    source_weights, image_source_report = source_sampling_weights(
        image_records, image_source_weights)
    image_sample_weights = combine_sampling_weights(quality_weights, source_weights)
    image_record_weights = record_weight_lookup(image_records, image_sample_weights)
    image_bucket_probs = (
        size_bucket_sampling_probs(
            image_records, train_size_buckets, bucket_records=bucket_records,
            record_weights=image_record_weights)
    )
    image_bucket_report = (
        size_bucket_sampling_report(
            image_records, train_size_buckets, bucket_records=bucket_records,
            record_weights=image_record_weights, bucket_probs=image_bucket_probs)
    )
    image_preference_pairs = (
        read_image_preference_pairs(
            image_preference_manifest,
            root=image_preference_root or image_root,
            max_pairs=image_preference_max_pairs)
        if image_preference_manifest else []
    )
    image_preference_report = image_preference_pair_report(
        image_preference_pairs, caption_cond_source=caption_cond_source)
    if (flow_preference_w > 0.0 and image_preference_pairs
            and image_preference_report["image_preference_pairs_direct_flow_usable"] <= 0):
        raise ValueError(
            "flow_preference_w has no usable preference-pair conditions; embedding "
            "conditioning requires text_embedding or text_embedding_sequence in the "
            "preference manifest")
    image_quality_score_stats = quality_score_stats(image_records)
    scalar_quality_requested = (
        image_quality_score_w > 0.0 or flow_quality_score_w > 0.0
        or image_quality_rank_w > 0.0 or quality_score_rank_w > 0.0
        or flow_quality_rank_w > 0.0
        or (quality_score_steps > 0 and image_preference_w <= 0.0)
    )
    if scalar_quality_requested and image_quality_score_stats["n"] <= 0:
        raise ValueError("quality score losses require manifest aesthetic/score/quality values")
    if (image_quality_rank_w > 0.0 or quality_score_rank_w > 0.0
            or flow_quality_rank_w > 0.0) and not image_quality_score_stats["has_range"]:
        raise ValueError("quality rank losses require at least two distinct quality values")
    if cond_mode == "text":
        prompt_vocab = (
            None if caption_cond_source == "embedding"
            else build_caption_vocab(image_records, max_vocab=caption_vocab_max)
        )
    cond_dim = 0
    conditioner = None
    if cond_mode == "text":
        cond_dim = int(text_cond_dim or hidden)
        if caption_cond_source == "embedding":
            text_pooled_embedding_in_dim = infer_text_pooled_embedding_dim(image_records)
            conditioner = PrecomputedTextConditioner(
                text_embedding_in_dim, cond_dim=cond_dim, hidden=hidden,
                pooled_input_dim=(
                    text_pooled_embedding_in_dim
                    if text_pooled_embedding_in_dim > 0
                    and text_pooled_embedding_in_dim != text_embedding_in_dim
                    else 0
                )).to(device)
        else:
            text_pooled_embedding_in_dim = 0
            conditioner = PromptConditioner(len(prompt_vocab), cond_dim=cond_dim,
                                            hidden=hidden,
                                            max_len=caption_max_len).to(device)
    ae = make_autoencoder(ae_arch=ae_arch, latent_ch=latent_ch, hidden=hidden,
                          latent_downsample=latent_downsample,
                          ae_res_blocks=ae_res_blocks,
                          ae_hf_model=ae_hf_model,
                          ae_hf_subfolder=ae_hf_subfolder,
                          ae_hf_scaling_factor=ae_hf_scaling_factor).to(device)
    latent_ch = int(getattr(ae, "latent_ch", latent_ch))
    ae_has_trainable_params = any(p.requires_grad for p in ae.parameters())
    ae_trainable_param_count = int(sum(p.numel() for p in ae.parameters() if p.requires_grad))
    text_aligner = None
    if image_text_align_w > 0.0 or flow_text_align_w > 0.0:
        text_aligner = ImageTextAligner(
            latent_ch=latent_ch, cond_dim=cond_dim, hidden=hidden,
            embed_dim=text_embed_dim).to(device)
    image_feature_aligner = None
    if image_feature_align_w > 0.0 or flow_feature_align_w > 0.0:
        image_feature_aligner = ImageFeatureAligner(
            latent_ch=latent_ch, feature_dim=image_embedding_in_dim, hidden=hidden,
            embed_dim=image_feature_embed_dim).to(device)
    image_quality_scorer = None
    if (image_quality_score_w > 0.0 or flow_quality_score_w > 0.0
            or image_quality_rank_w > 0.0 or quality_score_rank_w > 0.0
            or flow_quality_rank_w > 0.0 or quality_score_steps > 0):
        image_quality_scorer = ImageQualityScorer(latent_ch=latent_ch, hidden=hidden).to(device)
    latent_shape = ae_latent_shape(ae, size)
    latent_cells = latent_shape[1] * latent_shape[2]
    latent_tokens = latent_token_count(latent_shape, patch_size=latent_patch_size)
    train_latent_shapes = [ae_latent_shape(ae, bucket) for bucket in train_size_buckets]
    max_train_latent_cells = max(shape[1] * shape[2] for shape in train_latent_shapes)
    max_train_latent_tokens = max(
        latent_token_count(shape, patch_size=latent_patch_size)
        for shape in train_latent_shapes
    )
    max_train_latent_shape = max(
        train_latent_shapes, key=lambda shape: shape[1] * shape[2])
    time_shift_report = resolve_time_shift(
        time_shift, mode=time_shift_mode, latent_shape=max_train_latent_shape,
        ref_dim=time_shift_ref_dim, dim_power=time_shift_dim_power)
    effective_time_shift = float(time_shift_report["time_shift"])
    max_required_latent_tokens = max(int(latent_tokens), int(max_train_latent_tokens))
    if flow_arch in ("dit", "crossdit", "mmdit") and max_required_latent_tokens > int(
            latent_max_tokens):
        raise ValueError(
            f"latent token count {max_required_latent_tokens} exceeds "
            f"latent_max_tokens={latent_max_tokens}; "
            "increase --latent-max-tokens, use a larger --latent-patch-size, "
            "or use a larger --latent-downsample"
        )
    flow = make_flow(flow_arch=flow_arch, latent_ch=latent_ch, hidden=hidden,
                     dit_depth=dit_depth, dit_heads=dit_heads, cond_dim=cond_dim,
                     dit_head_width_mult=dit_head_width_mult,
                     latent_max_tokens=latent_max_tokens,
                     dit_qk_norm=bool(dit_qk_norm),
                     dit_attn_impl=dit_attn_impl,
                     dit_pos_embed=dit_pos_embed,
                     dit_mlp=dit_mlp,
                     latent_patch_size=latent_patch_size,
                     flow_checkpoint_blocks=flow_checkpoint_blocks,
                     flow_time_embed=flow_time_embed,
                     flow_time_embed_dim=flow_time_embed_dim).to(device)
    attach_image_geometry_mode(flow, image_geometry_cond)
    flow_repa_aligner = None
    if flow_repa_w > 0.0:
        hidden_feature_dim = flow_hidden_feature_dim(flow)
        if hidden_feature_dim <= 0:
            raise ValueError("flow REPA alignment requires DiT/CrossDiT/MM-DiT flow architecture")
        flow_repa_aligner = FlowFeatureAligner(
            hidden_dim=hidden_feature_dim, feature_dim=image_embedding_in_dim,
            hidden=hidden, embed_dim=flow_repa_embed_dim).to(device)
    flow_self_repa_aligner = None
    if flow_self_repa_w > 0.0:
        hidden_feature_dim = flow_hidden_feature_dim(flow)
        if hidden_feature_dim <= 0:
            raise ValueError(
                "flow self-REPA alignment requires DiT/CrossDiT/MM-DiT flow architecture")
        flow_self_repa_aligner = FlowLatentAligner(
            hidden_dim=hidden_feature_dim, latent_ch=latent_ch,
            patch_size=latent_patch_size, hidden=hidden,
            embed_dim=flow_self_repa_embed_dim).to(device)
    if flow_sra_w > 0.0 and flow_hidden_feature_dim(flow) <= 0:
        raise ValueError("flow SRA alignment requires DiT/CrossDiT/MM-DiT flow architecture")
    attach_text_aligner(flow, text_aligner)
    attach_image_feature_aligner(flow, image_feature_aligner)
    attach_image_quality_scorer(flow, image_quality_scorer)
    attach_flow_repa_aligner(flow, flow_repa_aligner)
    attach_flow_self_repa_aligner(flow, flow_self_repa_aligner)
    attach_flow_boundary_mode(flow, flow_boundary_mode)
    ae_params = [p for p in ae.parameters() if p.requires_grad]
    if text_aligner is not None and image_text_align_w > 0.0:
        ae_params += list(conditioner.parameters()) + list(text_aligner.parameters())
    if image_feature_aligner is not None and image_feature_align_w > 0.0:
        ae_params += list(image_feature_aligner.parameters())
    if image_quality_scorer is not None and (
            image_quality_score_w > 0.0 or image_quality_rank_w > 0.0):
        ae_params += list(image_quality_scorer.parameters())
    opt_ae = (
        torch.optim.AdamW(ae_params, lr=lr, weight_decay=0.01)
        if ae_params else None
    )
    scaler = amp_grad_scaler(device, train_precision)
    if ae_has_trainable_params:
        ae.train()
    else:
        ae.eval()
    if text_aligner is not None:
        text_aligner.train()
    if image_feature_aligner is not None:
        image_feature_aligner.train()
    if image_quality_scorer is not None:
        image_quality_scorer.train()
    size_bucket_sample_counts = {image_size_key(bucket): 0 for bucket in train_size_buckets}
    last_ae = {}
    ae_train_steps_run = 0
    for _ in range(ae_steps if opt_ae is not None else 0):
        ae_train_steps_run += 1
        opt_ae.zero_grad(set_to_none=True)
        for _micro in range(ae_accum_steps):
            batch_size, x, captions, chosen_records = sample_bucketed_image_text_batch(
                image_records, rng, batch=batch, size_buckets=train_size_buckets,
                bucket_records=bucket_records, device=device, return_records=True,
                crop_mode=image_crop_mode, hflip_prob=image_hflip_prob,
                record_weights=image_record_weights, bucket_probs=image_bucket_probs)
            size_bucket_sample_counts[image_size_key(batch_size)] += int(batch)
            with amp_autocast(device, train_precision):
                out = ae(x)
                loss, parts = reconstruction_loss_parts(
                    out["recon"], x, mode=ae_recon_loss, grad_w=ae_grad_w,
                    ms_w=ae_ms_w, fft_w=ae_fft_w, latent=out.get("latent"),
                    latent_reg_w=ae_latent_reg_w)
                if text_aligner is not None and image_text_align_w > 0.0:
                    cond_vec = caption_record_condition(
                        captions, chosen_records, conditioner, prompt_vocab,
                        source=caption_cond_source, max_len=caption_max_len,
                        device=device, return_tokens=False)
                    align_loss, align_parts = image_text_alignment_loss(
                        text_aligner, out["latent"], cond_vec, prefix="caption_align")
                    loss = loss + float(image_text_align_w) * align_loss
                    parts.update(align_parts)
                if image_feature_aligner is not None and image_feature_align_w > 0.0:
                    image_features = record_image_embedding_tensor(
                        chosen_records, device=device)
                    feature_loss, feature_parts = image_feature_alignment_loss(
                        image_feature_aligner, out["latent"], image_features,
                        prefix="image_feature_align")
                    loss = loss + float(image_feature_align_w) * feature_loss
                    parts.update(feature_parts)
                if image_quality_scorer is not None and image_quality_score_w > 0.0:
                    quality_loss, quality_parts = image_quality_score_loss(
                        image_quality_scorer, out["latent"], chosen_records,
                        image_quality_score_stats, prefix="quality_score")
                    loss = loss + float(image_quality_score_w) * quality_loss
                    parts.update(quality_parts)
                if image_quality_scorer is not None and image_quality_rank_w > 0.0:
                    quality_rank, quality_rank_parts = image_quality_rank_loss(
                        image_quality_scorer, out["latent"], chosen_records,
                        image_quality_score_stats, prefix="quality_rank",
                        margin=quality_rank_margin)
                    loss = loss + float(image_quality_rank_w) * quality_rank
                    parts.update(quality_rank_parts)
                scaled_loss = loss / float(ae_accum_steps)
            scaler.scale(scaled_loss).backward()
            last_ae = {k: float(v.detach().cpu()) for k, v in parts.items()}
            last_ae["total_loss"] = float(loss.detach().cpu())
        if grad_clip > 0.0:
            scaler.unscale_(opt_ae)
            grad_norm = torch.nn.utils.clip_grad_norm_(ae_params, float(grad_clip))
            last_ae["grad_norm"] = float(grad_norm.detach().cpu())
        scaler.step(opt_ae)
        scaler.update()

    flow_params = list(flow.parameters()) + ([] if conditioner is None
                                             else list(conditioner.parameters()))
    if text_aligner is not None and flow_text_align_w > 0.0:
        flow_params += list(text_aligner.parameters())
    if image_feature_aligner is not None and flow_feature_align_w > 0.0:
        flow_params += list(image_feature_aligner.parameters())
    if flow_repa_aligner is not None and flow_repa_w > 0.0:
        flow_params += list(flow_repa_aligner.parameters())
    if flow_self_repa_aligner is not None and flow_self_repa_w > 0.0:
        flow_params += list(flow_self_repa_aligner.parameters())
    opt_flow = torch.optim.AdamW(flow_params, lr=lr, weight_decay=0.01)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    flow_cache = None
    quality_score_steps_run = 0
    last_quality_score = {}
    has_image_embedding_sequences = (
        records_have_image_embedding_sequences(image_records)
        if image_records is not None else False)
    flow_repa_cache_sequences = (
        flow_repa_w > 0.0
        and (flow_repa_mode in ("token", "both")
             or (flow_repa_mode == "auto" and has_image_embedding_sequences))
    )
    if flow_cache_latents:
        flow_cache = build_image_latent_cache(
            ae, image_records, prompt_vocab, caption_max_len=caption_max_len,
            max_records=flow_cache_records, batch=flow_cache_batch, seed=seed + 211,
            size=size, device=device, precision=train_precision,
            cond_source=caption_cond_source, cache_dir=flow_cache_dir,
            shard_size=flow_cache_shard_size,
            include_image_embeddings=flow_feature_align_w > 0.0 or flow_repa_w > 0.0,
            include_image_embedding_sequences=flow_repa_cache_sequences,
            include_image_geometry=image_geometry_cond,
            include_quality_targets=(
                flow_quality_score_w > 0.0 or flow_quality_rank_w > 0.0
                or (quality_score_steps > 0 and image_quality_score_stats["n"] > 0)),
            quality_stats=image_quality_score_stats,
            crop_mode=image_crop_mode, hflip_prob=image_hflip_prob,
            size_buckets=train_size_buckets, bucket_records=bucket_records,
            record_weights=image_record_weights, cache_dtype=flow_cache_dtype)
        configure_latent_cache_runtime(
            flow_cache, max_loaded_shards=flow_cache_max_loaded_shards)
    if flow_cache is not None:
        latent_stats = estimate_latent_stats_cache(
            flow_cache, n=latent_stat_samples, seed=seed + 97,
            mode=effective_latent_normalize)
    elif len(train_size_buckets) > 1:
        latent_stats = estimate_latent_stats_record_buckets(
            ae, image_records, train_size_buckets, bucket_records=bucket_records,
            n=latent_stat_samples, batch=batch, seed=seed + 97, device=device,
            mode=effective_latent_normalize, crop_mode=image_crop_mode,
            hflip_prob=image_hflip_prob, record_weights=image_record_weights,
            bucket_probs=image_bucket_probs)
    else:
        latent_stats = estimate_latent_stats_records(
            ae, image_records, n=latent_stat_samples, batch=batch, seed=seed + 97,
            size=size, device=device, mode=effective_latent_normalize,
            crop_mode=image_crop_mode, hflip_prob=image_hflip_prob,
            weights=image_sample_weights)
    attach_latent_stats(flow, latent_stats)

    if image_quality_scorer is not None and quality_score_steps > 0:
        for p in image_quality_scorer.parameters():
            p.requires_grad_(True)
        image_quality_scorer.train()
        quality_params = list(image_quality_scorer.parameters())
        opt_quality = torch.optim.AdamW(quality_params, lr=lr, weight_decay=0.01)
        quality_live_records = [
            rec for rec in image_records
            if rec.aesthetic is not None and np.isfinite(float(rec.aesthetic))
        ]
        quality_bucket_records, _quality_missing_dims = (
            bucket_records_by_aspect(quality_live_records, train_size_buckets)
            if image_records is not None and quality_live_records else ({}, 0)
        )
        quality_record_weights = (
            record_weight_lookup(
                quality_live_records,
                weights_for_records(quality_live_records, image_record_weights))
            if image_record_weights is not None and quality_live_records else None
        )
        quality_bucket_probs = (
            size_bucket_sampling_probs(
                quality_live_records, train_size_buckets,
                bucket_records=quality_bucket_records,
                record_weights=quality_record_weights)
            if quality_live_records else None
        )
        preference_live_pairs = image_preference_pairs if image_preference_w > 0.0 else []
        for _quality_step in range(int(quality_score_steps)):
            opt_quality.zero_grad(set_to_none=True)
            quality_targets = None
            quality_masks = None
            chosen_quality_records = None
            quality_loss = None
            quality_parts = {}
            if flow_cache is not None and flow_cache.get("has_quality_targets", False):
                zq, quality_payload = sample_latent_cache(
                    flow_cache, rng, batch, device=device)
                quality_targets = quality_payload.get("quality_targets")
                quality_masks = quality_payload.get("quality_masks")
                if quality_targets is None or quality_masks is None:
                    raise ValueError("quality scorer pretrain cache is missing targets")
                with amp_autocast(device, train_precision):
                    quality_loss, quality_parts = image_quality_score_target_loss(
                        image_quality_scorer, zq, quality_targets, quality_masks,
                        prefix="quality_score_pretrain")
            else:
                if not quality_live_records and not preference_live_pairs:
                    raise ValueError(
                        "quality_score_steps require manifest quality values or preference pairs")
                if quality_live_records:
                    batch_size, xq, _captions_q, chosen_quality_records = (
                        sample_bucketed_image_text_batch(
                            quality_live_records, rng, batch=batch,
                            size_buckets=train_size_buckets,
                            bucket_records=quality_bucket_records,
                            device=device, return_records=True,
                            crop_mode=image_crop_mode, hflip_prob=image_hflip_prob,
                            record_weights=quality_record_weights,
                            bucket_probs=quality_bucket_probs))
                    size_bucket_sample_counts[image_size_key(batch_size)] += int(batch)
                    with torch.no_grad(), amp_autocast(device, train_precision):
                        zq = ae.encode(xq)
                    with amp_autocast(device, train_precision):
                        quality_loss, quality_parts = image_quality_score_loss(
                            image_quality_scorer, zq, chosen_quality_records,
                            image_quality_score_stats, prefix="quality_score_pretrain")
            if (quality_loss is not None and not quality_loss.requires_grad
                    and quality_live_records):
                batch_size, xq, _captions_q, chosen_quality_records = (
                    sample_bucketed_image_text_batch(
                        quality_live_records, rng, batch=batch,
                        size_buckets=train_size_buckets,
                        bucket_records=quality_bucket_records,
                        device=device, return_records=True,
                        crop_mode=image_crop_mode, hflip_prob=image_hflip_prob,
                        record_weights=quality_record_weights,
                        bucket_probs=quality_bucket_probs))
                size_bucket_sample_counts[image_size_key(batch_size)] += int(batch)
                with torch.no_grad(), amp_autocast(device, train_precision):
                    zq = ae.encode(xq)
                with amp_autocast(device, train_precision):
                    quality_loss, quality_parts = image_quality_score_loss(
                        image_quality_scorer, zq, chosen_quality_records,
                        image_quality_score_stats, prefix="quality_score_pretrain")
                quality_targets = None
                quality_masks = None
            if preference_live_pairs:
                pref_size = choose_image_size_bucket_weighted(
                    rng, train_size_buckets, bucket_probs=image_bucket_probs)
                x_pref_chosen, x_pref_rejected, pref_gaps = (
                    sample_image_preference_pair_batch(
                        preference_live_pairs, rng, batch=batch, size=pref_size,
                        device=device, crop_mode=image_crop_mode,
                        hflip_prob=image_hflip_prob))
                size_bucket_sample_counts[image_size_key(pref_size)] += int(batch)
                with torch.no_grad(), amp_autocast(device, train_precision):
                    z_pref_chosen = ae.encode(x_pref_chosen)
                    z_pref_rejected = ae.encode(x_pref_rejected)
                with amp_autocast(device, train_precision):
                    pref_loss, pref_parts = image_quality_preference_pair_loss(
                        image_quality_scorer, z_pref_chosen, z_pref_rejected,
                        pref_gaps, prefix="quality_preference_pretrain",
                        margin=quality_rank_margin)
                quality_loss = (
                    float(image_preference_w) * pref_loss
                    if quality_loss is None else
                    quality_loss + float(image_preference_w) * pref_loss
                )
                quality_parts.update(pref_parts)
            if quality_score_rank_w > 0.0 and quality_loss is not None:
                with amp_autocast(device, train_precision):
                    if quality_targets is not None and quality_masks is not None:
                        rank_loss, rank_parts = image_quality_rank_target_loss(
                            image_quality_scorer, zq, quality_targets, quality_masks,
                            prefix="quality_rank_pretrain", margin=quality_rank_margin)
                    else:
                        rank_loss, rank_parts = image_quality_rank_loss(
                            image_quality_scorer, zq, chosen_quality_records,
                            image_quality_score_stats, prefix="quality_rank_pretrain",
                            margin=quality_rank_margin)
                quality_loss = quality_loss + float(quality_score_rank_w) * rank_loss
                quality_parts.update(rank_parts)
            if quality_loss is None:
                continue
            last_quality_score = {
                k: float(v.detach().cpu()) for k, v in quality_parts.items()
            }
            last_quality_score["total_loss"] = float(quality_loss.detach().cpu())
            if not quality_loss.requires_grad:
                continue
            scaler.scale(quality_loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(opt_quality)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    quality_params, float(grad_clip))
                last_quality_score["grad_norm"] = float(grad_norm.detach().cpu())
            scaler.step(opt_quality)
            scaler.update()
            quality_score_steps_run += 1

    if image_quality_scorer is not None:
        image_quality_scorer.eval()
        for p in image_quality_scorer.parameters():
            p.requires_grad_(False)

    def sample_flow_distill_batch(teacher_conditioner=None):
        return_tokens = flow_uses_cond_tokens(flow)
        if flow_cache is not None:
            z1, cache_payload = sample_latent_cache(flow_cache, rng, batch, device=device)
            cond = cached_caption_payload_condition(
                cache_payload, conditioner, source=caption_cond_source,
                device=device, return_tokens=return_tokens)
            cond = attach_cached_image_geometry_condition(
                cond, cache_payload, enabled=image_geometry_cond, device=device)
            teacher_cond = (
                cached_caption_payload_condition(
                    cache_payload, teacher_conditioner, source=caption_cond_source,
                    device=device, return_tokens=return_tokens)
                if teacher_conditioner is not None else cond
            )
            if teacher_conditioner is not None:
                teacher_cond = attach_cached_image_geometry_condition(
                    teacher_cond, cache_payload, enabled=image_geometry_cond, device=device)
            return z1, cond, teacher_cond
        flow_batch = sample_bucketed_image_text_batch(
            image_records, rng, batch=batch, size_buckets=train_size_buckets,
            bucket_records=bucket_records, device=device, return_records=True,
            crop_mode=image_crop_mode, hflip_prob=image_hflip_prob,
            record_weights=image_record_weights, bucket_probs=image_bucket_probs,
            return_metadata=image_geometry_cond)
        if image_geometry_cond:
            _batch_size, x, captions, chosen_records, transform_metadata = flow_batch
        else:
            _batch_size, x, captions, chosen_records = flow_batch
            transform_metadata = None
        with torch.no_grad(), amp_autocast(device, train_precision):
            z1 = ae.encode(x)
        if caption_cond_source == "embedding":
            embs = record_text_embedding_tensor(chosen_records, device=device)
            cond = conditioner(embs, return_tokens=text_condition_return_tokens(
                conditioner, return_tokens=return_tokens))
            cond = attach_image_geometry_condition(
                cond, records=chosen_records, target_size=_batch_size,
                enabled=image_geometry_cond, transform_metadata=transform_metadata)
            teacher_cond = (
                teacher_conditioner(embs, return_tokens=text_condition_return_tokens(
                    teacher_conditioner, return_tokens=return_tokens))
                if teacher_conditioner is not None else cond
            )
            if teacher_conditioner is not None:
                teacher_cond = attach_image_geometry_condition(
                    teacher_cond, records=chosen_records, target_size=_batch_size,
                    enabled=image_geometry_cond, transform_metadata=transform_metadata)
        else:
            ids = caption_ids(captions, prompt_vocab, max_len=caption_max_len, device=device)
            cond = conditioner(ids, return_tokens=text_condition_return_tokens(
                conditioner, return_tokens=return_tokens))
            cond = attach_image_geometry_condition(
                cond, records=chosen_records, target_size=_batch_size,
                enabled=image_geometry_cond, transform_metadata=transform_metadata)
            teacher_cond = (
                teacher_conditioner(ids, return_tokens=text_condition_return_tokens(
                    teacher_conditioner, return_tokens=return_tokens))
                if teacher_conditioner is not None else cond
            )
            if teacher_conditioner is not None:
                teacher_cond = attach_image_geometry_condition(
                    teacher_cond, records=chosen_records, target_size=_batch_size,
                    enabled=image_geometry_cond, transform_metadata=transform_metadata)
        return z1, cond, teacher_cond

    flow.train()
    if conditioner is not None:
        conditioner.train()
    if text_aligner is not None:
        text_aligner.train()
    if image_feature_aligner is not None:
        image_feature_aligner.train()
    if image_quality_scorer is not None:
        image_quality_scorer.eval()
    if flow_repa_aligner is not None:
        flow_repa_aligner.train()
    if flow_self_repa_aligner is not None:
        flow_self_repa_aligner.train()
    last_flow = {}
    flow_preference_steps_run = 0
    flow_preference_skipped = 0
    flow_preference_skip_reason = ""
    flow_ema = clone_state_dict(flow) if flow_ema_decay > 0.0 else None
    conditioner_ema = (clone_state_dict(conditioner)
                       if flow_ema_decay > 0.0 and conditioner is not None else None)
    ema_updates = 0
    last_ema_decay = 0.0
    curriculum_switch_step = time_curriculum_switch_step(flow_steps, time_curriculum_frac)
    adaptive_time_sampler = (
        AdaptiveTimestepSampler(
            bins=time_adaptive_bins,
            momentum=time_adaptive_momentum,
            uniform_mix=time_adaptive_uniform_mix,
            min_prob=time_adaptive_min_prob,
            loss_power=time_adaptive_loss_power)
        if time_sampling == "adaptive" else None
    )
    size_curriculum_switch = size_curriculum_switch_step(
        flow_steps, size_curriculum_frac)
    for flow_step in range(flow_steps):
        opt_flow.zero_grad(set_to_none=True)
        active_flow_repa_w = float(flow_repa_w)
        if flow_repa_steps > 0 and flow_step >= int(flow_repa_steps):
            active_flow_repa_w = 0.0
        active_flow_self_repa_w = float(flow_self_repa_w)
        if flow_self_repa_steps > 0 and flow_step >= int(flow_self_repa_steps):
            active_flow_self_repa_w = 0.0
        active_flow_sra_w = float(flow_sra_w)
        if flow_sra_steps > 0 and flow_step >= int(flow_sra_steps):
            active_flow_sra_w = 0.0
        active_time_sampling = active_time_sampling_mode(
            time_sampling, flow_step=flow_step, flow_steps=flow_steps,
            time_curriculum_frac=time_curriculum_frac)
        active_size_buckets = active_size_curriculum_buckets(
            train_size_buckets, flow_step=flow_step, flow_steps=flow_steps,
            size_curriculum_frac=size_curriculum_frac)
        active_bucket_records = {
            bucket: bucket_records.get(bucket, []) for bucket in active_size_buckets
        }
        active_bucket_probs = size_bucket_probs_subset(
            train_size_buckets, image_bucket_probs, active_size_buckets)
        active_bucket_keys = image_size_bucket_keys(active_size_buckets)
        for _micro in range(flow_accum_steps):
            if flow_cache is not None:
                z1, cache_payload = sample_latent_cache(
                    flow_cache, rng, batch, device=device,
                    bucket_keys=active_bucket_keys)
                cache_bucket = str(cache_payload.get("size_bucket", ""))
                if cache_bucket:
                    size_bucket_sample_counts[cache_bucket] = (
                        int(size_bucket_sample_counts.get(cache_bucket, 0)) + int(batch)
                    )
                chosen_records = None
                cond = cached_caption_payload_condition(
                    cache_payload, conditioner, source=caption_cond_source,
                    device=device, return_tokens=flow_uses_cond_tokens(flow))
                cond = attach_cached_image_geometry_condition(
                    cond, cache_payload, enabled=image_geometry_cond, device=device)
                image_features = cache_payload.get("image_embeddings")
                image_feature_tokens = cache_payload.get("image_embedding_sequences")
                quality_targets = cache_payload.get("quality_targets")
                quality_masks = cache_payload.get("quality_masks")
            else:
                flow_batch = sample_bucketed_image_text_batch(
                    image_records, rng, batch=batch, size_buckets=active_size_buckets,
                    bucket_records=active_bucket_records, device=device, return_records=True,
                    crop_mode=image_crop_mode, hflip_prob=image_hflip_prob,
                    record_weights=image_record_weights, bucket_probs=active_bucket_probs,
                    return_metadata=image_geometry_cond)
                if image_geometry_cond:
                    batch_size, x, captions, chosen_records, transform_metadata = flow_batch
                else:
                    batch_size, x, captions, chosen_records = flow_batch
                    transform_metadata = None
                size_bucket_sample_counts[image_size_key(batch_size)] += int(batch)
                with torch.no_grad(), amp_autocast(device, train_precision):
                    z1 = ae.encode(x)
                cond = caption_record_condition(
                    captions, chosen_records, conditioner, prompt_vocab,
                    source=caption_cond_source, max_len=caption_max_len, device=device,
                    return_tokens=flow_uses_cond_tokens(flow))
                cond = attach_image_geometry_condition(
                    cond, records=chosen_records, target_size=batch_size,
                    enabled=image_geometry_cond, transform_metadata=transform_metadata)
                image_features = (
                    record_image_embedding_tensor(chosen_records, device=device)
                    if flow_feature_align_w > 0.0 or active_flow_repa_w > 0.0 else None
                )
                image_feature_tokens = (
                    record_image_embedding_sequence_tensor(chosen_records, device=device)
                    if active_flow_repa_w > 0.0
                    and (flow_repa_mode in ("token", "both")
                         or (flow_repa_mode == "auto" and has_image_embedding_sequences))
                    else None
                )
                quality_targets = None
                quality_masks = None
            flow_pref_payload = None
            if flow_preference_w > 0.0 and image_preference_pairs:
                pref_batch = max(1, int(flow_preference_batch or batch))
                pref_size = choose_image_size_bucket_weighted(
                    rng, active_size_buckets, bucket_probs=active_bucket_probs)
                x_pref_chosen, x_pref_rejected, pref_gaps, pref_pairs = (
                    sample_image_preference_pair_batch(
                        image_preference_pairs, rng, batch=pref_batch, size=pref_size,
                        device=device, crop_mode=image_crop_mode,
                        hflip_prob=image_hflip_prob, return_pairs=True))
                size_bucket_sample_counts[image_size_key(pref_size)] = (
                    int(size_bucket_sample_counts.get(image_size_key(pref_size), 0))
                    + int(pref_batch)
                )
                with torch.no_grad(), amp_autocast(device, train_precision):
                    z_pref_chosen = ae.encode(x_pref_chosen)
                    z_pref_rejected = ae.encode(x_pref_rejected)
                pref_cond = image_preference_pair_condition(
                    pref_pairs, conditioner, prompt_vocab, source=caption_cond_source,
                    caption_max_len=caption_max_len, device=device,
                    return_tokens=flow_uses_cond_tokens(flow))
                if pref_cond is not None:
                    pref_cond = attach_image_geometry_condition(
                        pref_cond, target_size=pref_size, batch=pref_batch,
                        enabled=image_geometry_cond)
                if pref_cond is None:
                    flow_preference_skipped += 1
                    flow_preference_skip_reason = (
                        "missing_preference_text_embedding"
                        if caption_cond_source == "embedding" else "missing_preference_condition"
                    )
                else:
                    flow_pref_payload = (
                        z_pref_chosen, z_pref_rejected, pref_cond, pref_gaps)
            with amp_autocast(device, train_precision):
                loss, parts = latent_flow_losses(
                    flow, z1, cond, cond_drop=cond_drop, ae=ae,
                    time_sampling=active_time_sampling, time_logit_mean=time_logit_mean,
                    time_logit_std=time_logit_std,
                    time_mode_scale=time_mode_scale, time_shift=effective_time_shift,
                    flow_noise_coupling=flow_noise_coupling,
                    flow_noise_coupling_projections=flow_noise_coupling_projections,
                    flow_loss_weight=flow_loss_weight,
                    flow_loss_weight_gamma=flow_loss_weight_gamma,
                    flow_loss_weight_normalize=flow_loss_weight_normalize,
                    consistency_w=flow_consistency_w,
                    endpoint_w=flow_endpoint_w,
                    text_aligner=text_aligner, text_align_w=flow_text_align_w,
                    feature_aligner=image_feature_aligner, image_features=image_features,
                    feature_align_w=flow_feature_align_w,
                    repa_aligner=flow_repa_aligner, repa_w=active_flow_repa_w,
                    image_feature_tokens=image_feature_tokens,
                    repa_mode=flow_repa_mode,
                    self_repa_aligner=flow_self_repa_aligner,
                    self_repa_w=active_flow_self_repa_w,
                    self_repa_mode=flow_self_repa_mode,
                    sra_w=active_flow_sra_w,
                    sra_time_gap=flow_sra_time_gap,
                    sra_mode=flow_sra_mode,
                    quality_scorer=image_quality_scorer,
                    quality_records=chosen_records,
                    quality_stats=image_quality_score_stats,
                    quality_targets=quality_targets,
                    quality_masks=quality_masks,
                    quality_w=flow_quality_score_w,
                    quality_rank_w=flow_quality_rank_w,
                    quality_rank_margin=quality_rank_margin,
                    adaptive_sampler=adaptive_time_sampler)
                if flow_pref_payload is not None:
                    z_pref_chosen, z_pref_rejected, pref_cond, pref_gaps = flow_pref_payload
                    pref_loss, pref_parts = latent_flow_preference_pair_loss(
                        flow, z_pref_chosen, z_pref_rejected, pref_cond,
                        score_gaps=pref_gaps, cond_drop=cond_drop,
                        time_sampling=active_time_sampling,
                        time_logit_mean=time_logit_mean,
                        time_logit_std=time_logit_std,
                        time_mode_scale=time_mode_scale,
                        time_shift=effective_time_shift,
                        flow_loss_weight=flow_loss_weight,
                        flow_loss_weight_gamma=flow_loss_weight_gamma,
                        flow_loss_weight_normalize=flow_loss_weight_normalize,
                        margin=flow_preference_margin,
                        adaptive_sampler=adaptive_time_sampler)
                    loss = loss + float(flow_preference_w) * pref_loss
                    parts.update(pref_parts)
                    parts["flow_preference_w"] = torch.tensor(
                        float(flow_preference_w), device=z1.device)
                    parts["flow_preference_active"] = torch.tensor(1.0, device=z1.device)
                    flow_preference_steps_run += 1
                elif flow_preference_w > 0.0:
                    parts["flow_preference_w"] = torch.tensor(
                        float(flow_preference_w), device=z1.device)
                    parts["flow_preference_active"] = torch.tensor(0.0, device=z1.device)
                scaled_loss = loss / float(flow_accum_steps)
            scaler.scale(scaled_loss).backward()
            last_flow = {"total_loss": float(loss.detach().cpu())}
            last_flow.update({k: float(v.detach().cpu()) for k, v in parts.items()})
            last_flow["time_sampling"] = active_time_sampling
            last_flow["time_curriculum_step"] = int(flow_step)
            last_flow["size_curriculum_step"] = int(flow_step)
            last_flow["active_size_buckets"] = [
                image_size_key(bucket) for bucket in active_size_buckets
            ]
        if grad_clip > 0.0:
            scaler.unscale_(opt_flow)
            grad_norm = torch.nn.utils.clip_grad_norm_(flow_params, float(grad_clip))
            last_flow["grad_norm"] = float(grad_norm.detach().cpu())
        scaler.step(opt_flow)
        scaler.update()
        if flow_ema is not None:
            ema_updates += 1
            last_ema_decay = ema_effective_decay(flow_ema_decay, ema_updates,
                                                 warmup=flow_ema_warmup)
            update_ema_state(flow_ema, flow, last_ema_decay)
            if conditioner is not None:
                update_ema_state(conditioner_ema, conditioner, last_ema_decay)

    last_distill = {}
    flow_distill_steps_run = 0
    flow_distill_teacher_used = ""
    if flow_distill_steps > 0 and flow_distill_w > 0.0:
        distill_time_sampling = active_time_sampling_mode(
            time_sampling, flow_step=flow_steps, flow_steps=flow_steps,
            time_curriculum_frac=time_curriculum_frac)
        if flow_distill_teacher == "ema":
            flow_distill_teacher_used = "ema"
        elif flow_distill_teacher == "auto" and flow_ema is not None:
            flow_distill_teacher_used = "ema"
        else:
            flow_distill_teacher_used = "raw"
        teacher_state = (
            flow_ema if flow_distill_teacher_used == "ema" else clone_state_dict(flow)
        )
        teacher_flow = make_flow(
            flow_arch=flow_arch, latent_ch=latent_ch, hidden=hidden,
            dit_depth=dit_depth, dit_heads=dit_heads, cond_dim=cond_dim,
            dit_head_width_mult=dit_head_width_mult,
            latent_max_tokens=latent_max_tokens,
            dit_qk_norm=dit_qk_norm, dit_attn_impl=dit_attn_impl,
            dit_pos_embed=dit_pos_embed,
            dit_mlp=dit_mlp,
            latent_patch_size=latent_patch_size,
            flow_checkpoint_blocks=flow_checkpoint_blocks,
            flow_time_embed=flow_time_embed,
            flow_time_embed_dim=flow_time_embed_dim).to(device)
        load_flow_state(teacher_flow, teacher_state)
        attach_latent_stats(teacher_flow, latent_stats)
        teacher_flow.eval()
        for p in teacher_flow.parameters():
            p.requires_grad_(False)
        teacher_conditioner = None
        if conditioner is not None:
            teacher_conditioner = copy.deepcopy(conditioner).to(device)
            if flow_distill_teacher_used == "ema" and conditioner_ema is not None:
                teacher_conditioner.load_state_dict(conditioner_ema)
            teacher_conditioner.eval()
            for p in teacher_conditioner.parameters():
                p.requires_grad_(False)
        flow.train()
        if conditioner is not None:
            conditioner.train()
        if image_quality_scorer is not None:
            image_quality_scorer.eval()
        for _distill_step in range(flow_distill_steps):
            opt_flow.zero_grad(set_to_none=True)
            for _micro in range(flow_accum_steps):
                z1, cond, teacher_cond = sample_flow_distill_batch(
                    teacher_conditioner=teacher_conditioner)
                with amp_autocast(device, train_precision):
                    loss, parts = latent_flow_self_distill_losses(
                        flow, teacher_flow, z1, cond, teacher_cond=teacher_cond,
                        time_sampling=distill_time_sampling,
                        time_logit_mean=time_logit_mean, time_logit_std=time_logit_std,
                        time_shift=effective_time_shift, latent_stats=latent_stats,
                        time_gap=flow_distill_time_gap,
                        guidance_w=flow_guidance_distill_w,
                        guidance_cfg_scale=flow_guidance_distill_cfg_scale,
                        guidance_cfg_rescale=flow_guidance_distill_cfg_rescale)
                    scaled_loss = (float(flow_distill_w) * loss) / float(flow_accum_steps)
                scaler.scale(scaled_loss).backward()
                last_distill = {
                    "total_loss": float((float(flow_distill_w) * loss).detach().cpu()),
                    "flow_distill_w": float(flow_distill_w),
                    "teacher": flow_distill_teacher_used,
                    "time_sampling": distill_time_sampling,
                }
                last_distill.update({
                    k: float(v.detach().cpu()) for k, v in parts.items()
                })
            if grad_clip > 0.0:
                scaler.unscale_(opt_flow)
                grad_norm = torch.nn.utils.clip_grad_norm_(flow_params, float(grad_clip))
                last_distill["grad_norm"] = float(grad_norm.detach().cpu())
            scaler.step(opt_flow)
            scaler.update()
            flow_distill_steps_run += 1
            if flow_ema is not None:
                ema_updates += 1
                last_ema_decay = ema_effective_decay(flow_ema_decay, ema_updates,
                                                     warmup=flow_ema_warmup)
                update_ema_state(flow_ema, flow, last_ema_decay)
                if conditioner is not None:
                    update_ema_state(conditioner_ema, conditioner, last_ema_decay)

    if conditioner is not None:
        conditioner.eval()
    if text_aligner is not None:
        text_aligner.eval()
    if image_feature_aligner is not None:
        image_feature_aligner.eval()
    if image_quality_scorer is not None:
        image_quality_scorer.eval()
    if flow_repa_aligner is not None:
        flow_repa_aligner.eval()
    if flow_self_repa_aligner is not None:
        flow_self_repa_aligner.eval()
    raw_flow = clone_state_dict(flow)
    raw_conditioner = clone_state_dict(conditioner) if conditioner is not None else None
    requested_eval_weight_mode = "raw" if not eval_with_ema else eval_weight_mode
    if requested_eval_weight_mode == "auto":
        candidate_modes = ["raw"] + (["ema"] if flow_ema is not None else [])
    elif requested_eval_weight_mode == "ema" and flow_ema is None:
        candidate_modes = ["raw"]
    else:
        candidate_modes = [requested_eval_weight_mode]
    candidate_reports = {}
    for mode in candidate_modes:
        if mode == "ema":
            load_flow_state(flow, flow_ema)
            if conditioner is not None:
                conditioner.load_state_dict(conditioner_ema)
        else:
            load_flow_state(flow, raw_flow)
            if conditioner is not None:
                conditioner.load_state_dict(raw_conditioner)
        candidate = evaluate_image_records(
            ae, flow, image_records, seed=seed + 1, size=size, device=device,
            conditioner=conditioner, prompt_vocab=prompt_vocab,
            caption_max_len=caption_max_len, cfg_scale=cfg_scale,
            cfg_rescale=cfg_rescale,
            cfg_mode=cfg_mode,
            cfg_schedule=cfg_schedule,
            sample_steps=sample_steps, sample_method=sample_method,
            sample_schedule=sample_schedule,
            cfg_interval=cfg_interval, text_aligner=text_aligner,
            image_feature_aligner=image_feature_aligner,
            image_quality_scorer=image_quality_scorer,
            caption_cond_source=caption_cond_source,
            sample_time_shift=effective_time_shift, time_shift=effective_time_shift,
            sample_churn=sample_churn,
            sample_churn_interval=sample_churn_interval,
            generated_eval_candidates_per_prompt=(
                eval_generated_candidates_per_prompt),
            sample_finite_guard=sample_finite_guard,
            sample_velocity_clip=sample_velocity_clip,
            sample_latent_clip=sample_latent_clip)
        candidate["eval_weight_mode"] = mode
        candidate_reports[mode] = candidate
    selected_eval_weights = max(candidate_reports, key=lambda mode: report_selection_key(
        candidate_reports[mode]))
    report = dict(candidate_reports[selected_eval_weights])
    load_flow_state(flow, raw_flow)
    if conditioner is not None:
        conditioner.load_state_dict(raw_conditioner)
    report.update({
        "experiment": "image_latent_manifest_rectified_flow",
        "ae_steps": int(ae_steps),
        "ae_train_steps_run": int(ae_train_steps_run),
        "ae_trainable": bool(ae_has_trainable_params),
        "ae_trainable_param_count": int(ae_trainable_param_count),
        "quality_score_steps": int(quality_score_steps),
        "quality_score_steps_run": int(quality_score_steps_run),
        "flow_steps": int(flow_steps),
        "batch": int(batch),
        "flow_cache_latents": bool(flow_cache is not None),
        "flow_cache_requested": bool(flow_cache_latents),
        "flow_cache_backend": latent_cache_backend(flow_cache) if flow_cache is not None else "",
        "flow_cache_dir": str(flow_cache.get("cache_dir", "")) if flow_cache is not None else "",
        "flow_cache_records": int(flow_cache["records"]) if flow_cache is not None else 0,
        "flow_cache_max_records": int(flow_cache_records),
        "flow_cache_batch": int(flow_cache_batch),
        "flow_cache_shard_size": int(flow_cache_shard_size),
        "flow_cache_reused": bool(
            flow_cache.get("cache_reused", False) if flow_cache is not None else False
        ),
        "flow_cache_latent_dtype": (
            latent_cache_dtype_name(flow_cache) if flow_cache is not None else flow_cache_dtype
        ),
        "flow_cache_shards": (
            int(flow_cache.get("shard_count", len(flow_cache.get("shards", []))))
            if flow_cache is not None else 0
        ),
        "flow_cache_bucket_count": int(flow_cache.get("bucket_count", 0))
        if flow_cache is not None else 0,
        "flow_cache_bucket_sampling_mode": (
            str(flow_cache.get("bucket_sampling_mode", ""))
            if flow_cache is not None else ""
        ),
        "flow_cache_bucket_sampling_probs": (
            dict(flow_cache.get("bucket_sampling_probs", {}))
            if flow_cache is not None else {}
        ),
        "flow_cache_bytes": int(flow_cache["bytes"]) if flow_cache is not None else 0,
        "flow_cache_cond_source": (
            str(flow_cache["cond_source"]) if flow_cache is not None else caption_cond_source
        ),
        "flow_cache_weighted": bool(
            flow_cache.get("weighted", False) if flow_cache is not None else False
        ),
        "flow_cache_has_quality_targets": bool(
            flow_cache.get("has_quality_targets", False) if flow_cache is not None else False
        ),
        "flow_cache_has_image_geometry": bool(
            flow_cache.get("has_image_geometry", False) if flow_cache is not None else False
        ),
        **latent_cache_runtime_report(flow_cache),
        **latent_cache_text_embedding_report(flow_cache),
        **latent_cache_text_pooled_embedding_report(flow_cache),
        **latent_cache_image_embedding_sequence_report(flow_cache),
        "ae_accum_steps": int(ae_accum_steps),
        "flow_accum_steps": int(flow_accum_steps),
        "ae_effective_batch": int(batch) * int(ae_accum_steps),
        "flow_effective_batch": int(batch) * int(flow_accum_steps),
        "train_precision": train_precision,
        "train_amp_enabled": bool(amp_cfg["enabled"]),
        "train_amp_dtype": amp_cfg["dtype_name"],
        "grad_clip": float(grad_clip),
        "image_size": image_size_value(size),
        "image_h": int(size[0]),
        "image_w": int(size[1]),
        "size_buckets": image_size_buckets_value(train_size_buckets),
        "size_bucket_count": int(len(train_size_buckets)),
        "size_bucket_sample_counts": dict(sorted(size_bucket_sample_counts.items())),
        "size_bucket_missing_dims": int(bucket_missing_dims),
        **image_bucket_report,
        "size_curriculum_frac": float(size_curriculum_frac),
        "size_curriculum_switch_step": int(size_curriculum_switch),
        "size_curriculum_enabled": bool(size_curriculum_switch > 0),
        "size_curriculum_final_buckets": [
            image_size_key(bucket) for bucket in train_size_buckets
        ],
        "size_curriculum_bucket_order": [
            image_size_key(bucket) for bucket in size_curriculum_bucket_order(
                train_size_buckets)
        ],
        "latent_ch": int(latent_ch),
        "ae_arch": ae_arch,
        "ae_external": ae_arch == "hf-vae",
        "ae_hf_model": ae_hf_model if ae_arch == "hf-vae" else "",
        "ae_hf_subfolder": ae_hf_subfolder if ae_arch == "hf-vae" else "",
        "ae_hf_scaling_factor": (
            float(getattr(ae, "scaling_factor", ae_hf_scaling_factor))
            if ae_arch == "hf-vae" else 0.0
        ),
        "latent_downsample": int(getattr(ae, "downsample", latent_downsample)),
        "ae_res_blocks": int(ae_res_blocks) if ae_arch == "residual" else 0,
        "latent_h": int(latent_shape[1]),
        "latent_w": int(latent_shape[2]),
        "latent_cells": int(latent_cells),
        "latent_tokens": int(latent_tokens),
        "max_train_latent_h": int(max_train_latent_shape[1]),
        "max_train_latent_w": int(max_train_latent_shape[2]),
        "max_train_latent_cells": int(max_train_latent_cells),
        "max_train_latent_tokens": int(max_train_latent_tokens),
        "latent_max_tokens": int(latent_max_tokens),
        "latent_patch_size": int(latent_patch_size),
        "hidden": int(hidden),
        "flow_arch": flow_arch,
        "dit_depth": int(dit_depth) if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "dit_heads": int(dit_heads) if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "dit_head_width_mult": (
            int(dit_head_width_mult) if flow_arch in ("dit", "crossdit", "mmdit") else 1
        ),
        "dit_qk_norm": bool(dit_qk_norm) if flow_arch == "mmdit" else False,
        "dit_attn_impl": dit_attn_impl if flow_arch == "mmdit" else "manual",
        "dit_pos_embed": dit_pos_embed if flow_arch in ("dit", "crossdit", "mmdit") else "",
        "dit_mlp": dit_mlp if flow_arch in ("dit", "crossdit", "mmdit") else "",
        "flow_time_embed": (
            flow_time_embed if flow_arch in ("dit", "crossdit", "mmdit") else ""
        ),
        "flow_time_embed_dim": (
            int(getattr(flow, "flow_time_embed_dim", 1))
            if flow_arch in ("dit", "crossdit", "mmdit") else 1
        ),
        "flow_checkpoint_blocks": bool(getattr(flow, "checkpoint_blocks", False)),
        "flow_boundary_mode": get_flow_boundary_mode(flow),
        "activation_checkpointing": bool(getattr(flow, "uses_activation_checkpointing", False)),
        "uses_2d_pos_embed": bool(getattr(flow, "uses_2d_pos_embed", False)),
        "uses_rope2d_pos_embed": bool(getattr(flow, "uses_rope2d_pos_embed", False)),
        "uses_swiglu_mlp": bool(getattr(flow, "uses_swiglu_mlp", False)),
        "adaptive_modulation": bool(getattr(flow, "uses_adaptive_modulation", False)),
        "residual_gating": bool(getattr(flow, "uses_residual_gating", False)),
        "zero_residual_gating": bool(getattr(flow, "uses_zero_residual_gating", False)),
        "ae_recon_loss": ae_recon_loss,
        "ae_grad_w": float(ae_grad_w),
        "ae_ms_w": float(ae_ms_w),
        "ae_fft_w": float(ae_fft_w),
        "ae_latent_reg_w": float(ae_latent_reg_w),
        "image_text_align_w": float(image_text_align_w),
        "flow_text_align_w": float(flow_text_align_w),
        "text_embed_dim": int(text_embed_dim),
        "text_aligner": text_aligner is not None,
        "image_feature_align_w": float(image_feature_align_w),
        "flow_feature_align_w": float(flow_feature_align_w),
        "image_feature_embed_dim": int(image_feature_embed_dim),
        "image_feature_aligner": image_feature_aligner is not None,
        "flow_repa_w": float(flow_repa_w),
        "flow_repa_steps": int(flow_repa_steps),
        "flow_repa_active_steps": (
            int(flow_steps) if flow_repa_w > 0.0 and int(flow_repa_steps) == 0
            else min(int(flow_steps), int(flow_repa_steps)) if flow_repa_w > 0.0 else 0
        ),
        "flow_repa_mode": str(flow_repa_mode),
        "flow_repa_token_sequences": bool(flow_repa_cache_sequences),
        "flow_repa_embed_dim": int(flow_repa_embed_dim),
        "flow_repa_aligner": flow_repa_aligner is not None,
        "flow_self_repa_w": float(flow_self_repa_w),
        "flow_self_repa_steps": int(flow_self_repa_steps),
        "flow_self_repa_active_steps": (
            int(flow_steps) if flow_self_repa_w > 0.0 and int(flow_self_repa_steps) == 0
            else min(int(flow_steps), int(flow_self_repa_steps))
            if flow_self_repa_w > 0.0 else 0
        ),
        "flow_self_repa_mode": str(flow_self_repa_mode),
        "flow_self_repa_embed_dim": int(flow_self_repa_embed_dim),
        "flow_self_repa_aligner": flow_self_repa_aligner is not None,
        "flow_sra_w": float(flow_sra_w),
        "flow_sra_steps": int(flow_sra_steps),
        "flow_sra_active_steps": (
            int(flow_steps) if flow_sra_w > 0.0 and int(flow_sra_steps) == 0
            else min(int(flow_steps), int(flow_sra_steps)) if flow_sra_w > 0.0 else 0
        ),
        "flow_sra_time_gap": float(flow_sra_time_gap),
        "flow_sra_mode": str(flow_sra_mode),
        "flow_preference_w": float(flow_preference_w),
        "flow_preference_margin": float(flow_preference_margin),
        "flow_preference_batch": int(flow_preference_batch),
        "flow_preference_steps_run": int(flow_preference_steps_run),
        "flow_preference_skipped": int(flow_preference_skipped),
        "flow_preference_skip_reason": flow_preference_skip_reason,
        "cond_drop": float(cond_drop),
        "image_geometry_cond": bool(image_geometry_cond),
        "cfg_rescale": float(cfg_rescale),
        "cfg_mode": cfg_mode,
        "cfg_schedule": cfg_schedule,
        "cfg_interval": list(cfg_interval),
        "sample_method": sample_method,
        "sample_schedule": sample_schedule,
        "flow_consistency_w": float(flow_consistency_w),
        "flow_endpoint_w": float(flow_endpoint_w),
        "flow_distill_steps": int(flow_distill_steps),
        "flow_distill_steps_run": int(flow_distill_steps_run),
        "flow_distill_w": float(flow_distill_w),
        "flow_distill_time_gap": float(flow_distill_time_gap),
        "flow_distill_teacher": flow_distill_teacher,
        "flow_distill_teacher_used": flow_distill_teacher_used,
        "flow_guidance_distill_w": float(flow_guidance_distill_w),
        "flow_guidance_distill_cfg_scale": float(flow_guidance_distill_cfg_scale),
        "flow_guidance_distill_cfg_rescale": float(flow_guidance_distill_cfg_rescale),
        "flow_ema_decay": float(flow_ema_decay),
        "flow_ema_warmup": bool(flow_ema_warmup),
        "flow_ema_updates": int(ema_updates),
        "flow_ema_effective_decay": float(last_ema_decay),
        "ema_available": flow_ema is not None,
        "eval_weight_mode": requested_eval_weight_mode,
        "selected_eval_weights": selected_eval_weights,
        "eval_with_ema": selected_eval_weights == "ema",
        "eval_generated_candidates_per_prompt": int(
            eval_generated_candidates_per_prompt),
        "weight_eval_candidates": {
            mode: eval_report_summary(candidate)
            for mode, candidate in candidate_reports.items()
        },
        "time_sampling": time_sampling,
        "time_curriculum_frac": float(time_curriculum_frac),
        "time_curriculum_switch_step": int(curriculum_switch_step),
        "time_curriculum_final_sampling": (
            "uniform" if curriculum_switch_step > 0 and curriculum_switch_step < int(flow_steps)
            else time_sampling
        ),
        "time_logit_mean": float(time_logit_mean),
        "time_logit_std": float(time_logit_std),
        "time_adaptive_requested": time_sampling == "adaptive",
        "time_adaptive_bins": int(time_adaptive_bins),
        "time_adaptive_momentum": float(time_adaptive_momentum),
        "time_adaptive_uniform_mix": float(time_adaptive_uniform_mix),
        "time_adaptive_min_prob": float(time_adaptive_min_prob),
        "time_adaptive_loss_power": float(time_adaptive_loss_power),
        **(
            adaptive_time_sampler.report()
            if adaptive_time_sampler is not None else {
                "time_adaptive_enabled": False,
                "time_adaptive_updates": 0,
            }
        ),
        **time_shift_report,
        "flow_loss_weight": flow_loss_weight,
        "flow_noise_coupling": flow_noise_coupling,
        "flow_noise_coupling_projections": int(flow_noise_coupling_projections),
        "flow_loss_weight_gamma": float(flow_loss_weight_gamma),
        "flow_loss_weight_normalize": bool(flow_loss_weight_normalize),
        "sample_time_shift": float(effective_time_shift),
        "sample_finite_guard": bool(sample_finite_guard),
        "sample_velocity_clip": float(sample_velocity_clip),
        "sample_latent_clip": float(sample_latent_clip),
        "latent_stat_samples": int(latent_stat_samples),
        "latent_normalize_requested": requested_latent_normalize,
        **latent_stats_report(latent_stats),
        "cond_mode": cond_mode,
        "conditioner_learned_null": conditioner_has_learned_null(conditioner),
        "cfg_uncond_default": (
            "learned_null" if conditioner_has_learned_null(conditioner) else "zero"
        ),
        "data_mode": "image_manifest",
        "image_manifest": image_manifest,
        "image_root": image_root,
        "image_split": image_split,
        "image_min_aesthetic": (
            float(image_min_aesthetic) if image_min_aesthetic is not None else None
        ),
        **image_quality_report,
        **image_source_report,
        "image_sampling_weighted": image_sample_weights is not None,
        "image_sampling_weight_min": (
            float(np.min(image_sample_weights)) if image_sample_weights is not None else 0.0
        ),
        "image_sampling_weight_mean": (
            float(np.mean(image_sample_weights)) if image_sample_weights is not None else 0.0
        ),
        "image_sampling_weight_max": (
            float(np.max(image_sample_weights)) if image_sample_weights is not None else 0.0
        ),
        "image_quality_score_w": float(image_quality_score_w),
        "flow_quality_score_w": float(flow_quality_score_w),
        "image_quality_rank_w": float(image_quality_rank_w),
        "quality_score_rank_w": float(quality_score_rank_w),
        "flow_quality_rank_w": float(flow_quality_rank_w),
        "quality_rank_margin": float(quality_rank_margin),
        "image_preference_manifest": image_preference_manifest,
        "image_preference_root": image_preference_root or image_root,
        "image_preference_max_pairs": int(image_preference_max_pairs),
        "image_preference_w": float(image_preference_w),
        "image_preference_pairs": int(len(image_preference_pairs)),
        **image_preference_report,
        "image_quality_scorer": image_quality_scorer is not None,
        "image_quality_score_records": int(image_quality_score_stats["n"]),
        "image_quality_score_missing": int(image_quality_score_stats["missing"]),
        "image_quality_score_min": float(image_quality_score_stats["min"]),
        "image_quality_score_mean": float(image_quality_score_stats["mean"]),
        "image_quality_score_max": float(image_quality_score_stats["max"]),
        "image_quality_score_has_range": bool(image_quality_score_stats["has_range"]),
        "image_crop_mode": image_crop_mode,
        "image_hflip_prob": float(image_hflip_prob),
        "caption_vocab_max": int(caption_vocab_max),
        "caption_max_len": int(caption_max_len),
        "caption_cond_source": caption_cond_source,
        "text_embedding_in_dim": int(text_embedding_in_dim),
        "text_pooled_embedding_in_dim": int(text_pooled_embedding_in_dim),
        "text_condition_dual_pooled_sequence": bool(
            caption_cond_source == "embedding" and text_pooled_embedding_in_dim > 0
            and text_pooled_embedding_in_dim != text_embedding_in_dim
        ),
        "image_embedding_in_dim": int(image_embedding_in_dim),
        "cond_dim": int(cond_dim),
        "prompt_templates": [],
        "prompt_vocab_size": len(prompt_vocab) if prompt_vocab is not None else 0,
        "last_ae": last_ae,
        "last_quality_score": last_quality_score,
        "last_flow": last_flow,
        "last_distill": last_distill,
        "last_flow_loss": last_flow.get("velocity_mse"),
    })
    report.update(summarize_records(image_records))
    if return_ema:
        if return_conditioner:
            if return_aligner:
                return (
                    ae, flow, conditioner, prompt_vocab, text_aligner, report,
                    flow_ema, conditioner_ema
                )
            return ae, flow, conditioner, prompt_vocab, report, flow_ema, conditioner_ema
        return ae, flow, report, flow_ema
    if return_conditioner:
        if return_aligner:
            return ae, flow, conditioner, prompt_vocab, text_aligner, report
        return ae, flow, conditioner, prompt_vocab, report
    return ae, flow, report


def selftest():
    small_shift = resolve_time_shift(
        2.0, mode="auto", latent_shape=(4, 8, 8), ref_dim=1024.0, dim_power=0.5)
    large_shift = resolve_time_shift(
        1.0, mode="auto", latent_shape=(16, 16, 16), ref_dim=1024.0, dim_power=0.5)
    explicit_dim_shift = resolve_time_shift(
        1.0, mode="dim", latent_shape=(4, 8, 8), ref_dim=1024.0, dim_power=0.5)
    assert small_shift["time_shift"] == 2.0
    assert small_shift["time_shift_effective_mode"] == "manual"
    assert large_shift["time_shift_effective_mode"] == "dim"
    assert math.isclose(large_shift["time_shift_dim_scale"], 2.0)
    assert explicit_dim_shift["time_shift_effective_mode"] == "dim"
    assert explicit_dim_shift["time_shift_dim_scale"] < 1.0
    linear_schedule = flow_time_schedule(8, device="cpu", shift=1.0, schedule="linear")
    karras_schedule = flow_time_schedule(8, device="cpu", shift=1.0, schedule="karras")
    assert torch.allclose(karras_schedule[:1], torch.zeros(1))
    assert torch.allclose(karras_schedule[-1:], torch.ones(1))
    assert bool(torch.all(karras_schedule[1:] > karras_schedule[:-1]))
    assert not torch.allclose(karras_schedule, linear_schedule)
    curriculum_buckets = normalize_image_size_buckets("16x16,32x16,32x32")
    assert [image_size_key(b) for b in size_curriculum_bucket_order(
        curriculum_buckets)] == ["16x16", "32x16", "32x32"]
    assert [image_size_key(b) for b in active_size_curriculum_buckets(
        curriculum_buckets, flow_step=0, flow_steps=4,
        size_curriculum_frac=0.5)] == ["16x16"]
    assert [image_size_key(b) for b in active_size_curriculum_buckets(
        curriculum_buckets, flow_step=1, flow_steps=4,
        size_curriculum_frac=0.5)] == ["16x16", "32x16", "32x32"]
    subset_probs = size_bucket_probs_subset(
        curriculum_buckets, np.asarray([0.2, 0.3, 0.5]),
        curriculum_buckets[:2])
    assert np.allclose(subset_probs, np.asarray([0.4, 0.6]))
    z = torch.randn(2, 3, 4, 4)
    raw = torch.randn_like(z)
    assert torch.allclose(
        apply_flow_boundary_velocity(raw, z, torch.zeros(2, 1, 1, 1),
                                     mode="double-cosine"),
        -z,
        atol=1.0e-6,
    )
    assert torch.allclose(
        apply_flow_boundary_velocity(raw, z, torch.ones(2, 1, 1, 1),
                                     mode="double-cosine"),
        z,
        atol=1.0e-6,
    )
    assert torch.allclose(
        apply_flow_boundary_velocity(raw, z, torch.full((2, 1, 1, 1), 0.5),
                                     mode="double-cosine"),
        raw,
        atol=1.0e-6,
    )
    with tempfile.TemporaryDirectory() as td:
        rows = []
        rng = np.random.default_rng(11)
        for i in range(3):
            arr = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
            arr[:, :, i % 3] = np.linspace(0, 255, 16, dtype=np.uint8)[None, :]
            name = f"image_{i}.ppm"
            with open(os.path.join(td, name), "wb") as f:
                f.write(b"P6\n16 16\n255\n")
                f.write(arr.tobytes())
            rows.append({
                "image": name,
                "caption": f"manifest texture sample {i}",
                "split": "train",
                "source": "fixture",
                "aesthetic": float(i + 1),
                "text_embedding": [1.0 if j == i else 0.0 for j in range(3)],
                "text_embedding_sequence": [
                    [1.0 if j == (i % 2) else 0.0 for j in range(2)],
                    [0.5 if j == ((i + 1) % 2) else 0.0 for j in range(2)],
                ],
                "image_embedding": [1.0 if j == (2 - i) else 0.0 for j in range(3)],
            })
        manifest = os.path.join(td, "manifest.jsonl")
        with open(manifest, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        manifest_records = read_image_manifest(manifest, root=td)
        dual_payload = record_text_condition_payload(manifest_records[:2], device="cpu")
        assert isinstance(dual_payload, dict)
        assert tuple(dual_payload["sequence"].shape) == (2, 2, 2)
        assert tuple(dual_payload["pooled"].shape) == (2, 3)
        dual_conditioner = PrecomputedTextConditioner(
            input_dim=2, cond_dim=8, hidden=16, pooled_input_dim=3)
        dual_cond = dual_conditioner(dual_payload, return_tokens=True)
        assert tuple(dual_cond["vec"].shape) == (2, 8)
        assert tuple(dual_cond["tokens"].shape) == (2, 3, 8)
        assert tuple(dual_cond["mask"].shape) == (2, 3)
        assert not bool(dual_cond["mask"][0, 0])
        geom_cond = attach_image_geometry_condition(
            dual_cond, records=manifest_records[:2], target_size=(16, 16), enabled=True)
        assert tuple(geom_cond["tokens"].shape) == (2, 4, 8)
        assert tuple(geom_cond["null_tokens"].shape) == (2, 4, 8)
        assert tuple(geom_cond["geometry_features"].shape) == (2, GEOMETRY_FEATURE_DIM)
        assert geom_cond["tokens"][:, -1].detach().abs().sum().item() > 0.0
        assert null_condition(geom_cond)["tokens"][:, -1].detach().abs().sum().item() > 0.0
        crop_meta = {
            "original_height": 10,
            "original_width": 20,
            "target_height": 16,
            "target_width": 16,
            "crop_top": 2,
            "crop_left": 4,
            "crop_height": 10,
            "crop_width": 10,
            "pad_top": 1,
            "pad_bottom": 1,
            "pad_left": 0,
            "pad_right": 0,
            "hflip": True,
        }
        geom_features = image_geometry_feature_tensor(
            manifest_records[:1], target_size=(16, 16),
            transform_metadata=[crop_meta], device="cpu")
        assert tuple(geom_features.shape) == (1, GEOMETRY_FEATURE_DIM)
        assert abs(float(geom_features[0, 8]) - 0.2) < 1.0e-6
        assert abs(float(geom_features[0, 9]) - 0.2) < 1.0e-6
        assert float(geom_features[0, -1]) == 1.0
        candidate_settings = prompt_candidate_sampler_settings(
            3, seed=11, cfg_scale=1.0, cfg_rescale=0.0,
            cfg_scales=(1.0, 2.0), cfg_rescales=(0.0, 0.5),
            cfg_modes=("standard", "cfgpp"), cfg_schedules=("constant", "triangular"),
            sample_steps_list=(4, 8), sample_methods=("euler", "heun"),
            sample_schedules=("linear", "cosine"), sample_churns=(0.0, 0.1),
            seeds=(101, 202))
        assert [row["cfg_scale"] for row in candidate_settings] == [1.0, 2.0, 1.0]
        assert [row["sample_method"] for row in candidate_settings] == [
            "euler", "heun", "euler"]
        assert [row["seed"] for row in candidate_settings] == [101, 202, 101]
        single_setting = prompt_candidate_sampler_settings(
            1, seed=7, cfg_scale=3.0, cfg_scales=(1.0, 2.0),
            sample_steps=9, sample_steps_list=(4, 8))[0]
        assert single_setting["cfg_scale"] == 3.0
        assert single_setting["sample_steps"] == 9 and single_setting["seed"] == 7
        dyn = torch.tensor([[[[0.0, 0.5], [2.0, -4.0]]]], dtype=torch.float32)
        dyn_out, dyn_events = dynamic_threshold_tensor(
            dyn, percentile=0.75, max_abs=1.0)
        assert dyn_events == 1
        assert float(dyn_out.abs().max()) <= 1.0 + 1.0e-6
        dyn_same, dyn_same_events = dynamic_threshold_tensor(dyn, percentile=0.0)
        assert dyn_same_events == 0 and torch.allclose(dyn_same, dyn)
        pref_manifest = os.path.join(td, "preferences.jsonl")
        with open(pref_manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "prompt": "manifest texture sample 1",
                "chosen_image": "image_1.ppm",
                "rejected_image": "image_0.ppm",
                "score_gap": 0.7,
                "chosen_sample_method": "adaptive-heun",
                "rejected_sample_method": "heun",
                "chosen_sample_cfg_schedule": "triangular",
                "rejected_sample_cfg_schedule": "constant",
            }) + "\n")
        pref_pairs = read_image_preference_pairs(pref_manifest, root=td)
        pref_report = image_preference_pair_report(pref_pairs)
        assert pref_pairs[0]["metadata"]["chosen_sample_method"] == "adaptive-heun"
        assert pref_report["image_preference_pairs_with_sample_metadata"] == 1
        assert pref_report["image_preference_chosen_cfg_schedules"] == {
            "triangular": 1}
        ae, flow, conditioner, vocab, report = train_latent_flow(
            ae_steps=1, flow_steps=1, batch=2, latent_ch=4, hidden=16,
            flow_arch="dit", dit_depth=1, dit_heads=2, seed=5, device="cpu",
            cond_mode="text", text_cond_dim=8, image_manifest=manifest,
            image_root=td, image_split="train", caption_max_len=8,
            image_max_records=3, sample_steps=1, flow_distill_steps=1,
            flow_guidance_distill_w=0.01, flow_ema_decay=0.9,
            flow_boundary_mode="double-cosine",
            flow_time_embed="fourier", flow_time_embed_dim=8,
            dit_mlp="swiglu",
            time_sampling="adaptive", time_adaptive_bins=4,
            time_adaptive_uniform_mix=0.1,
            image_geometry_cond=True,
            return_conditioner=True)
        assert report["experiment"] == "image_latent_manifest_rectified_flow"
        assert report["data_mode"] == "image_manifest"
        assert report["cond_mode"] == "text"
        assert report["image_records"] == 3
        assert report["time_shift_mode"] == "auto"
        assert report["time_shift_effective_mode"] == "manual"
        assert report["time_sampling"] == "adaptive"
        assert report["time_adaptive_enabled"] is True
        assert report["time_adaptive_updates"] >= 1
        assert report["time_adaptive_observed_bins"] >= 1
        assert report["image_geometry_cond"] is True
        assert report["flow_time_embed"] == "fourier"
        assert report["flow_time_embed_dim"] == 8
        assert report["dit_mlp"] == "swiglu"
        assert report["uses_swiglu_mlp"] is True
        assert report["adaptive_modulation"] is True
        assert report["residual_gating"] is True
        assert report["zero_residual_gating"] is True
        assert report["flow_boundary_mode"] == "double-cosine"
        assert report["flow_distill_steps_run"] == 1
        assert report["flow_distill_teacher_used"] == "ema"
        assert "recon_mse" in report
        assert conditioner is not None and vocab
        sample_path = os.path.join(td, "samples.ppm")
        meta = save_caption_sample_grid(
            ae, flow, read_image_manifest(manifest, root=td), sample_path,
            size=16, device="cpu", conditioner=conditioner, prompt_vocab=vocab,
            caption_max_len=8, cfg_scale=2.0, cfg_schedule="triangular",
            sample_steps=2, sample_method="adaptive-heun", samples=1)
        assert meta["sample_grid_cond_mode"] == "caption"
        assert meta["sample_grid_sample_method"] == "adaptive-heun"
        assert meta["sample_grid_cfg_schedule"] == "triangular"
        assert meta["sample_grid_trace_solver_steps"] >= 1
        assert meta["sample_grid_trace_solver_velocity_evals"] >= 2
        assert meta["sample_grid_trace_cfg_scale_max"] > 1.0
        assert meta["sample_grid_trace_cfg_guided_evals"] >= 1
        assert os.path.exists(sample_path)
    crossdit = make_flow(
        flow_arch="crossdit", latent_ch=4, hidden=16, dit_depth=1, dit_heads=2,
        cond_dim=8, dit_mlp="swiglu", latent_max_tokens=16,
        flow_time_embed="fourier", flow_time_embed_dim=8)
    crossdit_cond = {
        "vec": torch.randn(2, 8),
        "tokens": torch.randn(2, 3, 8),
        "mask": torch.zeros(2, 3, dtype=torch.bool),
    }
    crossdit_out = crossdit(
        torch.randn(2, 4, 4, 4), torch.rand(2, 1, 1, 1), crossdit_cond)
    assert tuple(crossdit_out.shape) == (2, 4, 4, 4)
    assert getattr(crossdit, "uses_adaptive_modulation", False) is True
    assert getattr(crossdit, "uses_zero_residual_gating", False) is True
    assert getattr(crossdit, "uses_swiglu_mlp", False) is True
    mmdit = make_flow(
        flow_arch="mmdit", latent_ch=4, hidden=16, dit_depth=1, dit_heads=2,
        cond_dim=8, dit_qk_norm=True, dit_attn_impl="auto", dit_pos_embed="rope2d",
        dit_mlp="swiglu", latent_max_tokens=16,
        flow_time_embed="fourier", flow_time_embed_dim=8)
    mmdit_cond = {
        "vec": torch.randn(2, 8),
        "tokens": torch.randn(2, 3, 8),
        "mask": torch.zeros(2, 3, dtype=torch.bool),
    }
    mmdit_out = mmdit(torch.randn(2, 4, 4, 4), torch.rand(2, 1, 1, 1), mmdit_cond)
    assert tuple(mmdit_out.shape) == (2, 4, 4, 4)
    assert getattr(mmdit, "attn_impl", "") == "auto"
    assert getattr(mmdit, "flow_time_embed", "") == "fourier"
    assert getattr(mmdit, "flow_time_embed_dim", 0) == 8
    print("image_latent selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval-checkpoint", default="", dest="eval_checkpoint",
                    help="load a saved image_latent checkpoint and run a sampler sweep")
    ap.add_argument("--eval-image-manifest", default="", dest="eval_image_manifest",
                    help="JSONL/CSV/TSV captioned image manifest for real-image checkpoint eval")
    ap.add_argument("--eval-image-root", default="", dest="eval_image_root",
                    help="base directory for relative paths in --eval-image-manifest")
    ap.add_argument("--eval-image-split", default="eval", dest="eval_image_split",
                    help="manifest split used for real-image checkpoint eval")
    ap.add_argument("--eval-image-min-aesthetic", type=float, default=None,
                    dest="eval_image_min_aesthetic",
                    help="optional minimum aesthetic/quality score for eval manifest rows")
    ap.add_argument("--eval-image-max-records", type=int, default=0,
                    dest="eval_image_max_records",
                    help="cap eval manifest records for smoke tests; 0 means all")
    ap.add_argument("--eval-generated-samples", type=int, default=0,
                    dest="eval_generated_samples",
                    help=("generated manifest samples per eval row; 0 uses "
                          "min(batch,sample_steps)"))
    ap.add_argument("--eval-generated-candidates-per-prompt", type=int, default=1,
                    dest="eval_generated_candidates_per_prompt",
                    help=("manifest eval candidates drawn per caption before learned "
                          "aligner/quality reranking"))
    ap.add_argument("--ae-steps", type=int, default=200, dest="ae_steps")
    ap.add_argument("--flow-steps", type=int, default=200, dest="flow_steps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ae-accum-steps", type=int, default=1, dest="ae_accum_steps",
                    help="AE gradient accumulation microsteps; effective batch=batch*steps")
    ap.add_argument("--flow-accum-steps", type=int, default=1, dest="flow_accum_steps",
                    help="flow gradient accumulation microsteps; effective batch=batch*steps")
    ap.add_argument("--flow-cache-latents", action="store_true", dest="flow_cache_latents",
                    help="precompute AE latents for manifest flow training")
    ap.add_argument("--flow-cache-records", type=int, default=0, dest="flow_cache_records",
                    help="maximum manifest records to cache for flow training; 0 means all")
    ap.add_argument("--flow-cache-batch", type=int, default=64, dest="flow_cache_batch",
                    help="batch size used while building the AE latent cache")
    ap.add_argument("--flow-cache-dir", default="", dest="flow_cache_dir",
                    help="directory for disk-backed manifest latent cache; implies cache")
    ap.add_argument("--flow-cache-shard-size", type=int, default=1024,
                    dest="flow_cache_shard_size",
                    help="records per shard for --flow-cache-dir")
    ap.add_argument("--flow-cache-dtype", default="fp32", choices=LATENT_CACHE_DTYPES,
                    dest="flow_cache_dtype",
                    help="dtype used to store cached AE latents")
    ap.add_argument("--flow-cache-max-loaded-shards", type=int, default=0,
                    dest="flow_cache_max_loaded_shards",
                    help="LRU disk-cache shards kept loaded in CPU RAM; 0 disables")
    ap.add_argument("--train-precision", default="fp32", choices=TRAIN_PRECISIONS,
                    dest="train_precision",
                    help="training precision; bf16/fp16 AMP is enabled on CUDA")
    ap.add_argument("--grad-clip", type=float, default=0.0, dest="grad_clip",
                    help="clip AE/flow gradient norm after accumulation; 0 disables")
    ap.add_argument("--size", default="0",
                    help="image size as SIZE or HxW; 0 means 32 for train or checkpoint size for eval")
    ap.add_argument("--size-buckets", default="", dest="size_buckets",
                    help=("manifest training buckets as comma/space/semicolon separated HxW values; "
                          "omitted uses --size"))
    ap.add_argument("--size-curriculum-frac", type=float, default=0.0,
                    dest="size_curriculum_frac",
                    help=("fraction of flow training that progressively unlocks "
                          "larger image buckets by latent area; 0 disables"))
    ap.add_argument("--latent-ch", type=int, default=16, dest="latent_ch")
    ap.add_argument("--ae-arch", default="semantic", choices=AE_ARCHES,
                    dest="ae_arch",
                    help="autoencoder architecture for image latents")
    ap.add_argument("--ae-hf-model", default="", dest="ae_hf_model",
                    help="Diffusers AutoencoderKL model id when --ae-arch hf-vae")
    ap.add_argument("--ae-hf-subfolder", default="", dest="ae_hf_subfolder",
                    help="optional Hugging Face subfolder for --ae-hf-model")
    ap.add_argument("--ae-hf-scaling-factor", type=float, default=0.0,
                    dest="ae_hf_scaling_factor",
                    help="override HF VAE latent scaling factor; 0 uses model config")
    ap.add_argument("--latent-downsample", type=int, default=4, dest="latent_downsample",
                    help="AE spatial compression factor; residual AE supports powers of two")
    ap.add_argument("--ae-res-blocks", type=int, default=1, dest="ae_res_blocks",
                    help="residual blocks per AE stage when --ae-arch residual")
    ap.add_argument("--latent-max-tokens", type=int, default=256, dest="latent_max_tokens",
                    help="maximum latent grid tokens for DiT/CrossDiT/MM-DiT flows")
    ap.add_argument("--latent-patch-size", type=int, default=1, dest="latent_patch_size",
                    help="spatial latent patch size for DiT/CrossDiT/MM-DiT image tokens")
    ap.add_argument("--ae-recon-loss", default="mse", choices=AE_RECON_LOSSES,
                    dest="ae_recon_loss",
                    help="base reconstruction loss for the image autoencoder")
    ap.add_argument("--ae-grad-w", type=float, default=0.0, dest="ae_grad_w",
                    help="edge/gradient reconstruction loss weight")
    ap.add_argument("--ae-ms-w", type=float, default=0.0, dest="ae_ms_w",
                    help="multi-scale reconstruction loss weight")
    ap.add_argument("--ae-fft-w", type=float, default=0.0, dest="ae_fft_w",
                    help="frequency-spectrum reconstruction loss weight")
    ap.add_argument("--ae-latent-reg-w", type=float, default=0.0, dest="ae_latent_reg_w",
                    help="latent L2 regularization weight during AE training")
    ap.add_argument("--image-text-align-w", type=float, default=0.0,
                    dest="image_text_align_w",
                    help="contrastive image-latent/caption alignment weight during AE training")
    ap.add_argument("--flow-text-align-w", type=float, default=0.0,
                    dest="flow_text_align_w",
                    help="contrastive caption alignment weight on predicted flow endpoints")
    ap.add_argument("--text-embed-dim", type=int, default=128, dest="text_embed_dim",
                    help="shared image/text embedding width for manifest alignment")
    ap.add_argument("--image-feature-align-w", type=float, default=0.0,
                    dest="image_feature_align_w",
                    help="contrastive AE-latent/external-image-feature alignment weight")
    ap.add_argument("--flow-feature-align-w", type=float, default=0.0,
                    dest="flow_feature_align_w",
                    help="contrastive external image-feature weight on predicted flow endpoints")
    ap.add_argument("--image-feature-embed-dim", type=int, default=128,
                    dest="image_feature_embed_dim",
                    help="shared embedding width for latent/image-feature alignment")
    ap.add_argument("--flow-repa-w", type=float, default=0.0, dest="flow_repa_w",
                    help="REPA-style hidden-state/image-feature alignment weight")
    ap.add_argument("--flow-repa-steps", type=int, default=0, dest="flow_repa_steps",
                    help="limit REPA alignment to the first N flow steps; 0 means all steps")
    ap.add_argument("--flow-repa-embed-dim", type=int, default=128,
                    dest="flow_repa_embed_dim",
                    help="shared embedding width for REPA hidden/image-feature alignment")
    ap.add_argument("--flow-repa-mode", default="pooled", choices=FLOW_REPA_MODES,
                    dest="flow_repa_mode",
                    help="REPA target: pooled image feature, token sequence, both, or auto")
    ap.add_argument("--flow-self-repa-w", type=float, default=0.0,
                    dest="flow_self_repa_w",
                    help=("no-extra-data REPA-style hidden/clean-latent alignment "
                          "weight for transformer flows"))
    ap.add_argument("--flow-self-repa-steps", type=int, default=0,
                    dest="flow_self_repa_steps",
                    help="limit self-REPA alignment to the first N flow steps; 0 means all steps")
    ap.add_argument("--flow-self-repa-embed-dim", type=int, default=128,
                    dest="flow_self_repa_embed_dim",
                    help="shared embedding width for self-REPA hidden/latent alignment")
    ap.add_argument("--flow-self-repa-mode", default="pooled", choices=FLOW_REPA_MODES,
                    dest="flow_self_repa_mode",
                    help="self-REPA target: pooled clean latent, latent patch tokens, both, or auto")
    ap.add_argument("--flow-sra-w", type=float, default=0.0, dest="flow_sra_w",
                    help=("self-representation alignment weight: match noisy-step hidden "
                          "tokens to detached cleaner-step hidden tokens"))
    ap.add_argument("--flow-sra-steps", type=int, default=0, dest="flow_sra_steps",
                    help="limit flow SRA alignment to the first N flow steps; 0 means all steps")
    ap.add_argument("--flow-sra-time-gap", type=float, default=0.25,
                    dest="flow_sra_time_gap",
                    help="fractional move toward clean data for the detached SRA teacher pass")
    ap.add_argument("--flow-sra-mode", default="token", choices=FLOW_REPA_MODES,
                    dest="flow_sra_mode",
                    help="flow SRA target: pooled hidden state, patch tokens, both, or auto")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--flow-arch", default="conv", choices=("conv", "dit", "crossdit", "mmdit"),
                    dest="flow_arch")
    ap.add_argument("--dit-depth", type=int, default=3, dest="dit_depth")
    ap.add_argument("--dit-heads", type=int, default=4, dest="dit_heads")
    ap.add_argument("--dit-head-width-mult", type=int, default=1,
                    dest="dit_head_width_mult",
                    help="width multiplier for the DiT/MM-DiT latent velocity head")
    ap.add_argument("--dit-qk-norm", action="store_true", dest="dit_qk_norm",
                    help="enable per-head QK RMSNorm in MM-DiT attention")
    ap.add_argument("--dit-attn-impl", default="auto", choices=MMDIT_ATTN_IMPLS,
                    dest="dit_attn_impl",
                    help=("MM-DiT attention implementation; auto uses exact PyTorch SDPA "
                          "when available, linear is an explicit approximation"))
    ap.add_argument("--dit-pos-embed", default="learned", choices=DIT_POS_EMBEDS,
                    dest="dit_pos_embed",
                    help=("image-token positional embedding for DiT/CrossDiT/MM-DiT flows; "
                          "rope2d is MM-DiT-only"))
    ap.add_argument("--dit-mlp", default="gelu", choices=DIT_MLPS, dest="dit_mlp",
                    help="feed-forward block for CrossDiT/MM-DiT image-token transformers")
    ap.add_argument("--flow-time-embed", default="scalar", choices=FLOW_TIME_EMBEDS,
                    dest="flow_time_embed",
                    help=("timestep embedding for DiT/CrossDiT/MM-DiT flows; "
                          "scalar keeps legacy behavior"))
    ap.add_argument("--flow-time-embed-dim", type=int, default=0,
                    dest="flow_time_embed_dim",
                    help=("Fourier timestep embedding width; 0 uses hidden width "
                          "when --flow-time-embed=fourier"))
    ap.add_argument("--flow-checkpoint-blocks", action="store_true",
                    dest="flow_checkpoint_blocks",
                    help=("checkpoint DiT/CrossDiT/MM-DiT transformer blocks during "
                          "flow training to reduce activation memory"))
    ap.add_argument("--cond-drop", type=float, default=0.0, dest="cond_drop")
    ap.add_argument("--cfg-scale", type=float, default=1.0, dest="cfg_scale")
    ap.add_argument("--cfg-rescale", type=float, default=0.0, dest="cfg_rescale",
                    help="rescale guided velocity toward conditional velocity std; 0 disables")
    ap.add_argument("--cfg-mode", default="standard", choices=CFG_MODES, dest="cfg_mode",
                    help="classifier-free guidance variant for sampling")
    ap.add_argument("--cfg-schedule", default="constant", choices=CFG_SCHEDULES,
                    dest="cfg_schedule",
                    help="time schedule for extra CFG over --cfg-interval")
    ap.add_argument("--flow-boundary-mode", default="none", choices=FLOW_BOUNDARY_MODES,
                    dest="flow_boundary_mode",
                    help=("boundary-conditioned RF velocity parameterization; double-cosine "
                          "enforces v(x,0)=-x and v(x,1)=x in normalized latent space"))
    ap.add_argument("--cfg-interval", default="0.0,1.0", dest="cfg_interval",
                    help="CFG active interval over rectified-flow time, formatted start,end")
    ap.add_argument("--sample-steps", type=int, default=4, dest="sample_steps")
    ap.add_argument("--sample-method", default="euler", choices=SAMPLE_METHODS,
                    dest="sample_method",
                    help="ODE sampler method for latent image generation")
    ap.add_argument("--sample-methods", default="",
                    dest="sample_methods",
                    help=("comma-separated sampler methods for --eval-checkpoint sweeps "
                          "and multi-candidate prompt grids"))
    ap.add_argument("--sample-schedule", default="linear", choices=SAMPLE_SCHEDULES,
                    dest="sample_schedule",
                    help="timestep placement schedule for latent image generation")
    ap.add_argument("--sample-schedules", default="",
                    dest="sample_schedules",
                    help=("comma-separated timestep schedules for --eval-checkpoint sweeps "
                          "and multi-candidate prompt grids"))
    ap.add_argument("--sample-churn", type=float, default=0.0, dest="sample_churn",
                    help="stochastic latent sampler churn; 0 keeps deterministic ODE sampling")
    ap.add_argument("--sample-churns", default="", dest="sample_churns",
                    help=("comma-separated sampler churn values for --eval-checkpoint sweeps "
                          "and multi-candidate prompt grids"))
    ap.add_argument("--sample-churn-interval", default="0.0,0.8",
                    dest="sample_churn_interval",
                    help="flow-time interval where stochastic sampler churn is active")
    ap.add_argument("--sample-finite-guard", action="store_true",
                    dest="sample_finite_guard",
                    help="replace non-finite latent/velocity sampler values with zeros")
    ap.add_argument("--sample-velocity-clip", type=float, default=0.0,
                    dest="sample_velocity_clip",
                    help="per-sample velocity RMS clip during sampling; 0 disables")
    ap.add_argument("--sample-latent-clip", type=float, default=0.0,
                    dest="sample_latent_clip",
                    help="absolute latent clamp during sampling; 0 disables")
    ap.add_argument("--sample-pixel-dynamic-threshold-percentile", type=float, default=0.0,
                    dest="sample_pixel_dynamic_threshold_percentile",
                    help=("decoded-pixel dynamic threshold percentile in (0,1]; "
                          "0 disables"))
    ap.add_argument("--sample-pixel-dynamic-threshold-max", type=float, default=1.0,
                    dest="sample_pixel_dynamic_threshold_max",
                    help="decoded-pixel dynamic threshold target max absolute value")
    ap.add_argument("--eval-text-guidance-weights", default="",
                    dest="eval_text_guidance_weights",
                    help=("comma-separated text-aligner guidance weights for manifest "
                          "--eval-checkpoint sweeps"))
    ap.add_argument("--eval-feature-guidance-weights", default="",
                    dest="eval_feature_guidance_weights",
                    help=("comma-separated external-feature guidance weights for manifest "
                          "--eval-checkpoint sweeps"))
    ap.add_argument("--eval-quality-guidance-weights", default="",
                    dest="eval_quality_guidance_weights",
                    help=("comma-separated quality guidance weights for manifest "
                          "--eval-checkpoint sweeps"))
    ap.add_argument("--cfg-scales", default="1.0,1.25,1.5,2.0", dest="cfg_scales",
                    help=("comma-separated CFG scales for --eval-checkpoint and "
                          "multi-candidate prompt grids"))
    ap.add_argument("--cfg-rescales", default="0.0", dest="cfg_rescales",
                    help=("comma-separated CFG rescale values for --eval-checkpoint and "
                          "multi-candidate prompt grids"))
    ap.add_argument("--cfg-modes", default="", dest="cfg_modes",
                    help=("comma-separated CFG modes for --eval-checkpoint/prompt candidates; "
                          "default uses --cfg-mode"))
    ap.add_argument("--cfg-schedules", default="", dest="cfg_schedules",
                    help=("comma-separated CFG schedules for --eval-checkpoint/prompt candidates; "
                          "default uses --cfg-schedule"))
    ap.add_argument("--sample-steps-list", default="4,8,16", dest="sample_steps_list",
                    help=("comma-separated sampler step counts for --eval-checkpoint and "
                          "multi-candidate prompt grids"))
    ap.add_argument("--eval-seeds", default="", dest="eval_seeds",
                    help=("comma-separated eval/candidate seeds for --eval-checkpoint or "
                          "multi-candidate prompt grids; default uses --seed"))
    ap.add_argument("--flow-consistency-w", type=float, default=0.0,
                    dest="flow_consistency_w",
                    help="same-path clean-endpoint consistency loss weight for latent flow")
    ap.add_argument("--flow-endpoint-w", type=float, default=0.0,
                    dest="flow_endpoint_w",
                    help="direct clean-endpoint latent prediction loss weight for latent flow")
    ap.add_argument("--flow-noise-coupling", default="random", choices=FLOW_NOISE_COUPLINGS,
                    dest="flow_noise_coupling",
                    help="source-noise/data pairing for flow matching")
    ap.add_argument("--flow-noise-coupling-projections", type=int, default=1,
                    dest="flow_noise_coupling_projections",
                    help="random projections to try for sliced_ot flow noise coupling")
    ap.add_argument("--flow-distill-steps", type=int, default=0,
                    dest="flow_distill_steps",
                    help="post-flow own-model endpoint distillation steps")
    ap.add_argument("--flow-distill-w", type=float, default=1.0,
                    dest="flow_distill_w",
                    help="loss weight for own-model endpoint distillation")
    ap.add_argument("--flow-distill-time-gap", type=float, default=0.25,
                    dest="flow_distill_time_gap",
                    help="fractional move toward clean data time for the frozen teacher")
    ap.add_argument("--flow-distill-teacher", default="auto", choices=FLOW_DISTILL_TEACHERS,
                    dest="flow_distill_teacher",
                    help="teacher snapshot for own-model distillation")
    ap.add_argument("--flow-guidance-distill-w", type=float, default=0.0,
                    dest="flow_guidance_distill_w",
                    help="inside distillation, match a CFG-guided frozen-teacher endpoint")
    ap.add_argument("--flow-guidance-distill-cfg-scale", type=float, default=1.5,
                    dest="flow_guidance_distill_cfg_scale",
                    help="CFG scale used by the frozen teacher for guidance distillation")
    ap.add_argument("--flow-guidance-distill-cfg-rescale", type=float, default=0.0,
                    dest="flow_guidance_distill_cfg_rescale",
                    help="CFG rescale used by the frozen teacher for guidance distillation")
    ap.add_argument("--flow-ema-decay", type=float, default=0.0, dest="flow_ema_decay",
                    help="EMA decay for flow/conditioner weights; 0 disables EMA")
    ap.add_argument("--no-ema-warmup", action="store_true", dest="no_ema_warmup",
                    help="use the exact EMA decay from the first update instead of ramping it")
    ap.add_argument("--ema-eval-mode", default="auto", choices=EVAL_WEIGHT_MODES,
                    dest="ema_eval_mode",
                    help="which weights to use for the final train report")
    ap.add_argument("--time-sampling", default="uniform", choices=TIME_SAMPLINGS,
                    dest="time_sampling",
                    help="rectified-flow training timestep distribution")
    ap.add_argument("--time-logit-mean", type=float, default=0.0, dest="time_logit_mean",
                    help="mean for --time-sampling logit-normal")
    ap.add_argument("--time-logit-std", type=float, default=1.0, dest="time_logit_std",
                    help="stddev for --time-sampling logit-normal")
    ap.add_argument("--time-curriculum-frac", type=float, default=0.0,
                    dest="time_curriculum_frac",
                    help=("fraction of flow training that uses --time-sampling before "
                          "switching timestep sampling to uniform; 0 disables"))
    ap.add_argument("--time-adaptive-bins", type=int, default=32,
                    dest="time_adaptive_bins",
                    help="number of loss-tracked bins for --time-sampling adaptive")
    ap.add_argument("--time-adaptive-momentum", type=float, default=0.95,
                    dest="time_adaptive_momentum",
                    help="EMA momentum for adaptive timestep bin losses")
    ap.add_argument("--time-adaptive-uniform-mix", type=float, default=0.05,
                    dest="time_adaptive_uniform_mix",
                    help="uniform probability mixed into adaptive timestep sampling")
    ap.add_argument("--time-adaptive-min-prob", type=float, default=0.001,
                    dest="time_adaptive_min_prob",
                    help="minimum adaptive timestep bin probability before renormalization")
    ap.add_argument("--time-adaptive-loss-power", type=float, default=1.0,
                    dest="time_adaptive_loss_power",
                    help="power applied to adaptive timestep loss EMA before sampling")
    ap.add_argument("--time-shift", type=float, default=1.0, dest="time_shift",
                    help="rectified-flow data-time shift; >1 biases training toward noise")
    ap.add_argument("--time-shift-mode", default="auto", choices=TIME_SHIFT_MODES,
                    dest="time_shift_mode",
                    help=("manual uses --time-shift as-is; dim scales it by latent "
                          "dimension; auto uses dim only when latent dimension exceeds "
                          "--time-shift-ref-dim"))
    ap.add_argument("--time-shift-ref-dim", type=float, default=1024.0,
                    dest="time_shift_ref_dim",
                    help="reference latent element count for --time-shift-mode dim")
    ap.add_argument("--time-shift-dim-power", type=float, default=0.5,
                    dest="time_shift_dim_power",
                    help="power used for dimension-aware time-shift scaling")
    ap.add_argument("--flow-loss-weight", default="none", choices=FLOW_LOSS_WEIGHTS,
                    dest="flow_loss_weight",
                    help="per-timestep velocity loss weighting")
    ap.add_argument("--flow-loss-weight-gamma", type=float, default=5.0,
                    dest="flow_loss_weight_gamma",
                    help="gamma for Min-SNR-style flow loss weighting")
    ap.add_argument("--no-flow-loss-weight-normalize", action="store_true",
                    dest="no_flow_loss_weight_normalize",
                    help="do not normalize weighted velocity loss to batch mean 1")
    ap.add_argument("--sample-time-shift", type=float, default=None,
                    dest="sample_time_shift",
                    help="override checkpoint sample-time shift; omitted uses checkpoint metadata")
    ap.add_argument("--latent-normalize", default="auto",
                    choices=LATENT_NORMALIZE_MODES, dest="latent_normalize",
                    help=("normalize AE latents before flow training/sampling; auto uses "
                          "channel stats for real-image/external-VAE runs"))
    ap.add_argument("--latent-stat-samples", type=int, default=512,
                    dest="latent_stat_samples",
                    help="number of AE samples used to estimate latent normalization stats")
    ap.add_argument("--cond-mode", default="text", choices=("text",), dest="cond_mode",
                    help="caption/text conditioning source for manifest latent flow")
    ap.add_argument("--text-cond-dim", type=int, default=0, dest="text_cond_dim",
                    help="text condition vector width; default uses --hidden")
    ap.add_argument("--image-manifest", default="", dest="image_manifest",
                    help="JSONL/CSV/TSV captioned image manifest for real image-text training")
    ap.add_argument("--image-root", default="", dest="image_root",
                    help="base directory for relative paths in --image-manifest")
    ap.add_argument("--image-split", default="train", dest="image_split",
                    help="manifest split to train/evaluate")
    ap.add_argument("--image-min-aesthetic", type=float, default=None,
                    dest="image_min_aesthetic",
                    help="optional minimum aesthetic/quality score for manifest rows")
    ap.add_argument("--image-quality-weight", type=float, default=0.0,
                    dest="image_quality_weight",
                    help=("sample manifest rows by normalized aesthetic/score/quality metadata; "
                          "0 keeps uniform sampling"))
    ap.add_argument("--image-source-weights", default="",
                    dest="image_source_weights",
                    help=("comma-separated source=weight data-mixture controls for image "
                          "manifest rows; '*' sets the default source weight"))
    ap.add_argument("--image-quality-score-w", type=float, default=0.0,
                    dest="image_quality_score_w",
                    help="train a latent aesthetic/quality score head on manifest metadata")
    ap.add_argument("--flow-quality-score-w", type=float, default=0.0,
                    dest="flow_quality_score_w",
                    help=("preserve the learned quality score on predicted flow endpoints; "
                          "requires --image-quality-score-w or --quality-score-steps"))
    ap.add_argument("--quality-score-steps", type=int, default=0,
                    dest="quality_score_steps",
                    help=("standalone latent quality-scorer pretrain steps from cached "
                          "or encoded manifest latents"))
    ap.add_argument("--image-quality-rank-w", type=float, default=0.0,
                    dest="image_quality_rank_w",
                    help="AE-phase pairwise quality preference ranking loss weight")
    ap.add_argument("--quality-score-rank-w", type=float, default=0.0,
                    dest="quality_score_rank_w",
                    help="standalone scorer pretrain pairwise quality ranking loss weight")
    ap.add_argument("--flow-quality-rank-w", type=float, default=0.0,
                    dest="flow_quality_rank_w",
                    help="preserve batch quality preference ordering on flow endpoints")
    ap.add_argument("--quality-rank-margin", type=float, default=0.0,
                    dest="quality_rank_margin",
                    help="minimum scorer-logit margin for quality preference ranking")
    ap.add_argument("--image-preference-manifest", default="",
                    dest="image_preference_manifest",
                    help=("JSONL chosen/rejected image-pair manifest for latent quality "
                          "scorer pretraining"))
    ap.add_argument("--image-preference-root", default="",
                    dest="image_preference_root",
                    help="base directory for relative paths in --image-preference-manifest")
    ap.add_argument("--image-preference-max-pairs", type=int, default=0,
                    dest="image_preference_max_pairs",
                    help="cap preference pairs loaded for quality scorer pretraining; 0 means all")
    ap.add_argument("--image-preference-w", type=float, default=0.0,
                    dest="image_preference_w",
                    help=("pairwise chosen/rejected preference loss weight inside "
                          "--quality-score-steps"))
    ap.add_argument("--flow-preference-w", type=float, default=0.0,
                    dest="flow_preference_w",
                    help=("direct chosen/rejected preference loss weight on the "
                          "latent rectified-flow generator"))
    ap.add_argument("--flow-preference-margin", type=float, default=0.0,
                    dest="flow_preference_margin",
                    help="minimum per-pair velocity-loss gap for direct flow preference")
    ap.add_argument("--flow-preference-batch", type=int, default=0,
                    dest="flow_preference_batch",
                    help="preference pairs per flow update; 0 reuses --batch")
    ap.add_argument("--image-max-records", type=int, default=0, dest="image_max_records",
                    help="cap manifest rows for smoke tests; 0 means all")
    ap.add_argument("--caption-vocab-max", type=int, default=8192, dest="caption_vocab_max",
                    help="maximum caption vocabulary size for image manifests")
    ap.add_argument("--caption-max-len", type=int, default=64, dest="caption_max_len",
                    help="maximum caption tokens for image manifests")
    ap.add_argument("--caption-cond-source", default="tokens",
                    choices=("tokens", "embedding", "auto"), dest="caption_cond_source",
                    help="caption conditioning source for image manifests")
    ap.add_argument("--image-geometry-cond", action="store_true",
                    dest="image_geometry_cond",
                    help=("append fixed original/target size and aspect conditioning "
                          "to latent flow text conditions"))
    ap.add_argument("--image-crop-mode", default="center",
                    choices=("center", "random", "none", "pad"), dest="image_crop_mode",
                    help="crop mode for manifest training images")
    ap.add_argument("--image-hflip-prob", type=float, default=0.0,
                    dest="image_hflip_prob",
                    help="random horizontal flip probability for manifest training images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/image_latent_flow.pt")
    ap.add_argument("--eval-out", default="", dest="eval_out",
                    help="JSON path for --eval-checkpoint report")
    ap.add_argument("--sample-grid-out", default="", dest="sample_grid_out",
                    help="optional PPM path for a prompt/caption generated sample grid")
    ap.add_argument("--sample-grid-samples", type=int, default=1,
                    dest="sample_grid_samples",
                    help="generated samples per color/shape condition in --sample-grid-out")
    ap.add_argument("--sample-manifest-out", default="", dest="sample_manifest_out",
                    help=("optional JSONL manifest path for individual generated samples "
                          "from --sample-grid-out"))
    ap.add_argument("--sample-image-dir", default="", dest="sample_image_dir",
                    help=("optional directory for individual generated PPM samples; "
                          "default is derived from --sample-manifest-out"))
    ap.add_argument("--sample-candidates-manifest-out", default="",
                    dest="sample_candidates_manifest_out",
                    help=("optional JSONL manifest path for all prompt candidates before "
                          "internal selection; useful for reward/preference mining"))
    ap.add_argument("--sample-candidates-image-dir", default="",
                    dest="sample_candidates_image_dir",
                    help=("optional directory for all prompt candidate PPM samples; "
                          "default is derived from --sample-candidates-manifest-out"))
    ap.add_argument("--sample-prompts", default="", dest="sample_prompts",
                    help="semicolon/newline separated prompts for --sample-grid-out")
    ap.add_argument("--sample-negative-prompts", default="", dest="sample_negative_prompts",
                    help=("optional semicolon/newline separated negative prompts for "
                          "--sample-prompts CFG baselines"))
    ap.add_argument("--sample-candidates-per-prompt", type=int, default=1,
                    dest="sample_candidates_per_prompt",
                    help=("number of candidates to draw per sample prompt; compatible "
                          "aligners/quality scorers select the best candidate"))
    ap.add_argument("--sample-text-guidance-w", type=float, default=0.0,
                    dest="sample_text_guidance_w",
                    help=("sampling-time text-image alignment guidance weight for "
                          "--sample-prompts; requires a checkpoint text_aligner"))
    ap.add_argument("--sample-text-guidance-interval", default="0.0,1.0",
                    dest="sample_text_guidance_interval",
                    help=("text alignment guidance active interval over flow time, "
                          "formatted start,end"))
    ap.add_argument("--sample-feature-guidance-w", type=float, default=0.0,
                    dest="sample_feature_guidance_w",
                    help=("sampling-time external feature guidance weight for "
                          "--sample-prompts; requires prompt embeddings and a checkpoint "
                          "image_feature_aligner"))
    ap.add_argument("--sample-feature-guidance-interval", default="0.0,1.0",
                    dest="sample_feature_guidance_interval",
                    help=("external feature guidance active interval over flow time, "
                          "formatted start,end"))
    ap.add_argument("--sample-quality-guidance-w", type=float, default=0.0,
                    dest="sample_quality_guidance_w",
                    help=("sampling-time manifest quality guidance weight for "
                          "--sample-prompts; requires a checkpoint image_quality_scorer"))
    ap.add_argument("--sample-quality-guidance-interval", default="0.0,1.0",
                    dest="sample_quality_guidance_interval",
                    help=("quality guidance active interval over flow time, "
                          "formatted start,end"))
    ap.add_argument("--prompt-embed-backend", default="stats",
                    choices=("stats", "hf"), dest="prompt_embed_backend",
                    help="text embedding backend used by --sample-prompts with embedding conditioning")
    ap.add_argument("--prompt-embed-model", default="", dest="prompt_embed_model",
                    help="Hugging Face model id for --prompt-embed-backend hf")
    ap.add_argument("--prompt-embed-text-sequence-model", default="",
                    dest="prompt_embed_text_sequence_model",
                    help=("optional Hugging Face text encoder for token-level live prompt "
                          "conditioning; primary --prompt-embed-model still supplies pooled "
                          "prompt features"))
    ap.add_argument("--prompt-embed-text-sequence-max-length", type=int, default=0,
                    dest="prompt_embed_text_sequence_max_length",
                    help="optional tokenizer max_length for --prompt-embed-text-sequence-model")
    ap.add_argument("--prompt-embed-device", default="", dest="prompt_embed_device",
                    help="device for live prompt embedding; default uses image_latent device")
    ap.add_argument("--prompt-embed-dtype", default="auto",
                    choices=("auto", "fp32", "fp16", "bf16"), dest="prompt_embed_dtype")
    ap.add_argument("--prompt-embed-stats-dim", type=int, default=0,
                    dest="prompt_embed_stats_dim",
                    help="stats backend prompt embedding width; 0 uses checkpoint input dim")
    ap.add_argument("--prompt-embed-no-normalize", action="store_true",
                    dest="prompt_embed_no_normalize",
                    help="do not L2-normalize live prompt embeddings")
    ap.add_argument("--prompt-embed-trust-remote-code", action="store_true",
                    dest="prompt_embed_trust_remote_code",
                    help="pass trust_remote_code=True to Hugging Face prompt embedder")
    ap.add_argument("--checkpoint-weight-mode", default="auto", choices=EVAL_WEIGHT_MODES,
                    dest="checkpoint_weight_mode",
                    help="which checkpoint weights to sweep: raw, ema, or measured auto-select")
    args = ap.parse_args(argv)
    try:
        cfg_interval = _parse_interval(args.cfg_interval)
        sample_churn_interval = _parse_interval(args.sample_churn_interval)
        cli_size = normalize_image_size(args.size, default=None)
        cli_size_buckets = normalize_image_size_buckets(args.size_buckets)
        sample_prompts = parse_sample_prompts(args.sample_prompts)
        sample_negative_prompts = parse_sample_prompts(args.sample_negative_prompts)
        eval_text_guidance_weights = (
            _parse_number_list(args.eval_text_guidance_weights, float)
            if args.eval_text_guidance_weights else (0.0,)
        )
        eval_feature_guidance_weights = (
            _parse_number_list(args.eval_feature_guidance_weights, float)
            if args.eval_feature_guidance_weights else (0.0,)
        )
        eval_quality_guidance_weights = (
            _parse_number_list(args.eval_quality_guidance_weights, float)
            if args.eval_quality_guidance_weights else (0.0,)
        )
        sample_churns = (
            _parse_number_list(args.sample_churns, float)
            if args.sample_churns else None
        )
        cfg_scales = _parse_number_list(args.cfg_scales, float)
        cfg_rescales = _parse_number_list(args.cfg_rescales, float)
        sample_steps_list = _parse_number_list(args.sample_steps_list, int)
        sample_methods = (
            _parse_string_list(args.sample_methods) if args.sample_methods else None
        )
        eval_seeds = _parse_number_list(args.eval_seeds, int) if args.eval_seeds else None
        sample_text_guidance_interval = _parse_interval(args.sample_text_guidance_interval)
        sample_feature_guidance_interval = _parse_interval(
            args.sample_feature_guidance_interval)
        sample_quality_guidance_interval = _parse_interval(
            args.sample_quality_guidance_interval)
        parse_source_weight_spec(args.image_source_weights)
        sample_schedules = (
            _parse_string_list(args.sample_schedules) if args.sample_schedules else None
        )
        cfg_modes = (
            _parse_string_list(args.cfg_modes) if args.cfg_modes else (args.cfg_mode,)
        )
        cfg_schedules = (
            _parse_string_list(args.cfg_schedules)
            if args.cfg_schedules else (args.cfg_schedule,)
        )
    except ValueError as e:
        ap.error(str(e))
    if args.selftest:
        selftest()
        return
    if args.train and not args.image_manifest:
        ap.error("--train requires --image-manifest")
    if sample_prompts and not args.sample_grid_out:
        ap.error("--sample-prompts requires --sample-grid-out")
    if args.sample_manifest_out and not args.sample_grid_out:
        ap.error("--sample-manifest-out requires --sample-grid-out")
    if args.sample_image_dir and not args.sample_manifest_out:
        ap.error("--sample-image-dir requires --sample-manifest-out")
    if args.sample_candidates_manifest_out and not args.sample_grid_out:
        ap.error("--sample-candidates-manifest-out requires --sample-grid-out")
    if args.sample_candidates_manifest_out and not sample_prompts:
        ap.error("--sample-candidates-manifest-out requires --sample-prompts")
    if args.sample_candidates_manifest_out and args.sample_candidates_per_prompt <= 1:
        ap.error("--sample-candidates-manifest-out requires --sample-candidates-per-prompt > 1")
    if args.sample_candidates_image_dir and not args.sample_candidates_manifest_out:
        ap.error("--sample-candidates-image-dir requires --sample-candidates-manifest-out")
    if sample_negative_prompts and not sample_prompts:
        ap.error("--sample-negative-prompts requires --sample-prompts")
    if args.sample_candidates_per_prompt <= 0:
        ap.error("--sample-candidates-per-prompt must be positive")
    if args.sample_candidates_per_prompt > 1 and not sample_prompts:
        ap.error("--sample-candidates-per-prompt > 1 requires --sample-prompts")
    if args.eval_generated_samples < 0:
        ap.error("--eval-generated-samples must be non-negative")
    if args.eval_generated_candidates_per_prompt <= 0:
        ap.error("--eval-generated-candidates-per-prompt must be positive")
    if args.sample_velocity_clip < 0.0:
        ap.error("--sample-velocity-clip must be non-negative")
    if args.sample_latent_clip < 0.0:
        ap.error("--sample-latent-clip must be non-negative")
    if args.sample_pixel_dynamic_threshold_percentile < 0.0:
        ap.error("--sample-pixel-dynamic-threshold-percentile must be non-negative")
    if args.sample_pixel_dynamic_threshold_percentile > 1.0:
        ap.error("--sample-pixel-dynamic-threshold-percentile must be <= 1")
    if args.sample_pixel_dynamic_threshold_max <= 0.0:
        ap.error("--sample-pixel-dynamic-threshold-max must be positive")
    if args.flow_time_embed_dim < 0:
        ap.error("--flow-time-embed-dim must be non-negative")
    if args.flow_endpoint_w < 0.0:
        ap.error("--flow-endpoint-w must be non-negative")
    if args.flow_self_repa_w < 0.0:
        ap.error("--flow-self-repa-w must be non-negative")
    if args.flow_self_repa_steps < 0:
        ap.error("--flow-self-repa-steps must be non-negative")
    if args.flow_self_repa_embed_dim <= 0:
        ap.error("--flow-self-repa-embed-dim must be positive")
    if args.flow_sra_w < 0.0:
        ap.error("--flow-sra-w must be non-negative")
    if args.flow_sra_steps < 0:
        ap.error("--flow-sra-steps must be non-negative")
    if args.flow_sra_time_gap <= 0.0 or args.flow_sra_time_gap > 1.0:
        ap.error("--flow-sra-time-gap must be in (0, 1]")
    if args.flow_noise_coupling_projections <= 0:
        ap.error("--flow-noise-coupling-projections must be positive")
    if args.time_adaptive_bins <= 1:
        ap.error("--time-adaptive-bins must be > 1")
    if args.time_adaptive_momentum < 0.0 or args.time_adaptive_momentum >= 1.0:
        ap.error("--time-adaptive-momentum must be in [0, 1)")
    if args.time_adaptive_uniform_mix < 0.0 or args.time_adaptive_uniform_mix > 1.0:
        ap.error("--time-adaptive-uniform-mix must be in [0, 1]")
    if args.time_adaptive_min_prob < 0.0 or args.time_adaptive_min_prob >= 1.0:
        ap.error("--time-adaptive-min-prob must be in [0, 1)")
    if args.time_adaptive_loss_power <= 0.0:
        ap.error("--time-adaptive-loss-power must be positive")
    if args.flow_guidance_distill_w < 0.0:
        ap.error("--flow-guidance-distill-w must be non-negative")
    if args.flow_guidance_distill_cfg_scale < 1.0:
        ap.error("--flow-guidance-distill-cfg-scale must be >= 1")
    if args.flow_guidance_distill_cfg_rescale < 0.0 or args.flow_guidance_distill_cfg_rescale > 1.0:
        ap.error("--flow-guidance-distill-cfg-rescale must be in [0, 1]")
    if args.flow_cache_max_loaded_shards < 0:
        ap.error("--flow-cache-max-loaded-shards must be non-negative")
    if args.quality_score_steps < 0:
        ap.error("--quality-score-steps must be non-negative")
    if (args.image_quality_rank_w < 0.0 or args.quality_score_rank_w < 0.0
            or args.flow_quality_rank_w < 0.0):
        ap.error("quality rank weights must be non-negative")
    if args.quality_rank_margin < 0.0:
        ap.error("--quality-rank-margin must be non-negative")
    if args.image_preference_max_pairs < 0:
        ap.error("--image-preference-max-pairs must be non-negative")
    if args.image_preference_w < 0.0:
        ap.error("--image-preference-w must be non-negative")
    if args.image_preference_w > 0.0 and not args.image_preference_manifest:
        ap.error("--image-preference-w requires --image-preference-manifest")
    if args.image_preference_w > 0.0 and args.quality_score_steps <= 0:
        ap.error("--image-preference-w requires --quality-score-steps > 0")
    if args.flow_preference_w < 0.0:
        ap.error("--flow-preference-w must be non-negative")
    if args.flow_preference_margin < 0.0:
        ap.error("--flow-preference-margin must be non-negative")
    if args.flow_preference_batch < 0:
        ap.error("--flow-preference-batch must be non-negative")
    if args.flow_preference_w > 0.0 and not args.image_preference_manifest:
        ap.error("--flow-preference-w requires --image-preference-manifest")
    if args.image_preference_manifest and not args.image_manifest and args.train:
        ap.error("--image-preference-manifest requires --image-manifest training")
    if args.sample_text_guidance_w < 0.0:
        ap.error("--sample-text-guidance-w must be non-negative")
    if args.sample_text_guidance_w > 0.0 and not sample_prompts:
        ap.error("--sample-text-guidance-w requires --sample-prompts")
    if any(w < 0.0 for w in eval_text_guidance_weights):
        ap.error("--eval-text-guidance-weights must be non-negative")
    if any(w < 0.0 for w in eval_feature_guidance_weights):
        ap.error("--eval-feature-guidance-weights must be non-negative")
    if any(w < 0.0 for w in eval_quality_guidance_weights):
        ap.error("--eval-quality-guidance-weights must be non-negative")
    if args.sample_feature_guidance_w < 0.0:
        ap.error("--sample-feature-guidance-w must be non-negative")
    if args.sample_feature_guidance_w > 0.0 and not sample_prompts:
        ap.error("--sample-feature-guidance-w requires --sample-prompts")
    if args.sample_quality_guidance_w < 0.0:
        ap.error("--sample-quality-guidance-w must be non-negative")
    if args.sample_quality_guidance_w > 0.0 and not sample_prompts:
        ap.error("--sample-quality-guidance-w requires --sample-prompts")
    if args.prompt_embed_text_sequence_max_length < 0:
        ap.error("--prompt-embed-text-sequence-max-length must be non-negative")
    if args.prompt_embed_text_sequence_model and args.prompt_embed_backend != "hf":
        ap.error("--prompt-embed-text-sequence-model requires --prompt-embed-backend hf")
    if args.prompt_embed_text_sequence_model and not sample_prompts:
        ap.error("--prompt-embed-text-sequence-model requires --sample-prompts")
    if args.sample_churn < 0.0:
        ap.error("--sample-churn must be non-negative")
    if sample_churns is not None and any(x < 0.0 for x in sample_churns):
        ap.error("--sample-churns must be non-negative")
    if any(x < 0.0 for x in cfg_rescales) or any(x > 1.0 for x in cfg_rescales):
        ap.error("--cfg-rescales values must be in [0, 1]")
    if any(x <= 0 for x in sample_steps_list):
        ap.error("--sample-steps-list values must be positive")
    if sample_methods is not None:
        bad_methods = sorted(set(sample_methods) - set(SAMPLE_METHODS))
        if bad_methods:
            ap.error(f"unknown sample method(s): {','.join(bad_methods)}")
    if args.time_curriculum_frac < 0.0 or args.time_curriculum_frac > 1.0:
        ap.error("--time-curriculum-frac must be in [0, 1]")
    if args.size_curriculum_frac < 0.0 or args.size_curriculum_frac > 1.0:
        ap.error("--size-curriculum-frac must be in [0, 1]")
    if sample_prompts and args.prompt_embed_backend == "hf" and not args.prompt_embed_model:
        ap.error("--prompt-embed-backend hf requires --prompt-embed-model")
    if sample_schedules is not None:
        bad_schedules = sorted(set(sample_schedules) - set(SAMPLE_SCHEDULES))
        if bad_schedules:
            ap.error(f"unknown sample schedule(s): {','.join(bad_schedules)}")
    bad_cfg_modes = sorted(set(cfg_modes) - set(CFG_MODES))
    if bad_cfg_modes:
        ap.error(f"unknown cfg mode(s): {','.join(bad_cfg_modes)}")
    bad_cfg_schedules = sorted(set(cfg_schedules) - set(CFG_SCHEDULES))
    if bad_cfg_schedules:
        ap.error(f"unknown cfg schedule(s): {','.join(bad_cfg_schedules)}")
    if args.image_source_weights and not args.image_manifest:
        ap.error("--image-source-weights requires --image-manifest")
    if args.eval_checkpoint:
        report = evaluate_checkpoint(
            args.eval_checkpoint,
            cfg_scales=cfg_scales,
            cfg_rescales=cfg_rescales,
            sample_steps_list=sample_steps_list,
            seed=args.seed,
            size=cli_size,
            eval_seeds=eval_seeds,
            prefer_ema=args.checkpoint_weight_mode != "raw",
            weight_mode=args.checkpoint_weight_mode,
            sample_method=args.sample_method,
            sample_methods=sample_methods,
            sample_schedule=args.sample_schedule,
            sample_schedules=sample_schedules,
            sample_churn=args.sample_churn,
            sample_churns=sample_churns,
            sample_churn_interval=sample_churn_interval,
            cfg_interval=cfg_interval,
            cfg_schedule=args.cfg_schedule,
            cfg_schedules=cfg_schedules,
            text_guidance_weights=eval_text_guidance_weights,
            feature_guidance_weights=eval_feature_guidance_weights,
            quality_guidance_weights=eval_quality_guidance_weights,
            eval_image_manifest=args.eval_image_manifest,
            eval_image_root=args.eval_image_root,
            eval_image_split=args.eval_image_split,
            eval_image_min_aesthetic=args.eval_image_min_aesthetic,
            eval_image_max_records=args.eval_image_max_records,
            sample_time_shift=args.sample_time_shift,
            generated_eval_n=args.eval_generated_samples,
            generated_eval_candidates_per_prompt=(
                args.eval_generated_candidates_per_prompt),
            sample_finite_guard=args.sample_finite_guard,
            sample_velocity_clip=args.sample_velocity_clip,
            sample_latent_clip=args.sample_latent_clip,
            cfg_modes=cfg_modes,
        )
        if args.sample_grid_out:
            selected_weights = report.get("selected_checkpoint_weights",
                                          report.get("checkpoint_weight_mode", "ema"))
            ae, flow, conditioner, prompt_vocab, prompt_templates, meta = load_checkpoint(
                args.eval_checkpoint, prefer_ema=(selected_weights == "ema"))
            settings = selected_grid_settings(
                report,
                fallback_cfg=args.cfg_scale,
                fallback_cfg_rescale=args.cfg_rescale,
                fallback_steps=args.sample_steps,
                fallback_method=args.sample_method,
                fallback_sample_schedule=args.sample_schedule,
                fallback_sample_churn=args.sample_churn,
                fallback_sample_churn_interval=sample_churn_interval,
                fallback_sample_finite_guard=args.sample_finite_guard,
                fallback_sample_velocity_clip=args.sample_velocity_clip,
                fallback_sample_latent_clip=args.sample_latent_clip,
                fallback_cfg_mode=args.cfg_mode,
                fallback_cfg_schedule=args.cfg_schedule,
                fallback_cfg_interval=cfg_interval,
                fallback_sample_time_shift=(
                    args.sample_time_shift if args.sample_time_shift is not None
                    else report.get("sample_time_shift", 1.0)
                ))
            grid_size = image_hw(
                cli_size, default=report.get("image_size", meta.get("image_size", 32)) or 32)
            if sample_prompts:
                grid_meta = save_text_prompt_sample_grid(
                    ae, flow, sample_prompts, args.sample_grid_out, conditioner=conditioner,
                    prompt_vocab=prompt_vocab, caption_max_len=meta["caption_max_len"],
                    size=grid_size,
                    cfg_scale=settings["cfg_scale"],
                    cfg_rescale=settings["cfg_rescale"],
                    cfg_mode=settings["cfg_mode"],
                    cfg_schedule=settings["cfg_schedule"],
                    cfg_interval=settings["cfg_interval"],
                    sample_time_shift=settings["sample_time_shift"],
                    sample_steps=settings["sample_steps"],
                    sample_method=settings["sample_method"],
                    sample_schedule=settings["sample_schedule"],
                    sample_churn=settings["sample_churn"],
                    sample_churn_interval=settings["sample_churn_interval"],
                    seed=args.seed + 991,
                    caption_cond_source=meta["caption_cond_source"] or "tokens",
                    prompt_embed_backend=args.prompt_embed_backend,
                    prompt_embed_model=args.prompt_embed_model,
                    prompt_embed_device=args.prompt_embed_device or None,
                    prompt_embed_dtype=args.prompt_embed_dtype,
                    prompt_embed_normalize=not args.prompt_embed_no_normalize,
                    prompt_embed_stats_dim=args.prompt_embed_stats_dim,
                    prompt_embed_text_sequence_model=args.prompt_embed_text_sequence_model,
                    prompt_embed_text_sequence_max_length=(
                        args.prompt_embed_text_sequence_max_length),
                    prompt_embed_trust_remote_code=args.prompt_embed_trust_remote_code,
                    negative_prompts=sample_negative_prompts,
                    candidates_per_prompt=args.sample_candidates_per_prompt,
                    text_guidance_w=args.sample_text_guidance_w,
                    text_guidance_interval=sample_text_guidance_interval,
                    feature_guidance_w=args.sample_feature_guidance_w,
                    feature_guidance_interval=sample_feature_guidance_interval,
                    quality_guidance_w=args.sample_quality_guidance_w,
                    quality_guidance_interval=sample_quality_guidance_interval,
                    sample_finite_guard=settings["sample_finite_guard"],
                    sample_velocity_clip=settings["sample_velocity_clip"],
                    sample_latent_clip=settings["sample_latent_clip"],
                    sample_pixel_dynamic_threshold_percentile=(
                        args.sample_pixel_dynamic_threshold_percentile),
                    sample_pixel_dynamic_threshold_max=(
                        args.sample_pixel_dynamic_threshold_max),
                    sample_manifest_out=args.sample_manifest_out,
                    sample_image_dir=args.sample_image_dir,
                    sample_candidates_manifest_out=(
                        args.sample_candidates_manifest_out),
                    sample_candidates_image_dir=args.sample_candidates_image_dir,
                    candidate_cfg_scales=cfg_scales,
                    candidate_cfg_rescales=cfg_rescales,
                    candidate_cfg_modes=cfg_modes,
                    candidate_cfg_schedules=cfg_schedules,
                    candidate_sample_steps_list=sample_steps_list,
                    candidate_sample_methods=sample_methods,
                    candidate_sample_schedules=sample_schedules,
                    candidate_sample_churns=sample_churns,
                    candidate_seeds=eval_seeds)
            elif report.get("experiment") == "image_latent_manifest_sampler_sweep":
                grid_records = read_image_manifest(
                    report["eval_image_manifest"], root=report.get("eval_image_root", ""),
                    split=report.get("eval_image_split", "eval"),
                    min_aesthetic=report.get("eval_image_min_aesthetic"),
                    max_records=report.get("eval_image_max_records", 0))
                grid_meta = save_caption_sample_grid(
                    ae, flow, grid_records, args.sample_grid_out, conditioner=conditioner,
                    prompt_vocab=prompt_vocab, caption_max_len=meta["caption_max_len"],
                    size=grid_size,
                    cfg_scale=settings["cfg_scale"],
                    cfg_rescale=settings["cfg_rescale"],
                    cfg_mode=settings["cfg_mode"],
                    cfg_schedule=settings["cfg_schedule"],
                    cfg_interval=settings["cfg_interval"],
                    sample_time_shift=settings["sample_time_shift"],
                    sample_steps=settings["sample_steps"],
                    sample_method=settings["sample_method"],
                    sample_schedule=settings["sample_schedule"],
                    sample_churn=settings["sample_churn"],
                    sample_churn_interval=settings["sample_churn_interval"],
                    sample_finite_guard=settings["sample_finite_guard"],
                    sample_velocity_clip=settings["sample_velocity_clip"],
                    sample_latent_clip=settings["sample_latent_clip"],
                    sample_pixel_dynamic_threshold_percentile=(
                        args.sample_pixel_dynamic_threshold_percentile),
                    sample_pixel_dynamic_threshold_max=(
                        args.sample_pixel_dynamic_threshold_max),
                    samples=args.sample_grid_samples,
                    seed=args.seed + 991,
                    caption_cond_source=meta["caption_cond_source"],
                    sample_manifest_out=args.sample_manifest_out,
                    sample_image_dir=args.sample_image_dir)
            else:
                ap.error(
                    "--sample-grid-out with --eval-checkpoint requires --sample-prompts "
                    "or --eval-image-manifest")
            grid_meta["sample_grid_checkpoint_weight_mode"] = (
                "ema" if meta["ema_loaded"] else "raw")
            report.update(grid_meta)
        if args.eval_out:
            os.makedirs(os.path.dirname(args.eval_out) or ".", exist_ok=True)
            with open(args.eval_out, "w") as f:
                json.dump(report, f, indent=1)
            print(json.dumps(report, indent=1))
            print(f"saved -> {args.eval_out}")
        else:
            print(json.dumps(report, indent=1))
        return
    if not args.train:
        ap.error("use --selftest, --train, or --eval-checkpoint")
    run_size = cli_size or (32, 32)
    (ae, flow, conditioner, prompt_vocab, text_aligner, report,
     flow_ema, conditioner_ema) = train_latent_flow(
        ae_steps=args.ae_steps, flow_steps=args.flow_steps, batch=args.batch,
        latent_ch=args.latent_ch, hidden=args.hidden, lr=args.lr,
        seed=args.seed, size=run_size, flow_arch=args.flow_arch, dit_depth=args.dit_depth,
        dit_heads=args.dit_heads, cond_drop=args.cond_drop, cfg_scale=args.cfg_scale,
        cfg_rescale=args.cfg_rescale, cfg_mode=args.cfg_mode,
        cfg_schedule=args.cfg_schedule,
        flow_boundary_mode=args.flow_boundary_mode,
        dit_head_width_mult=args.dit_head_width_mult,
        dit_qk_norm=args.dit_qk_norm,
        dit_attn_impl=args.dit_attn_impl,
        dit_pos_embed=args.dit_pos_embed,
        dit_mlp=args.dit_mlp,
        flow_time_embed=args.flow_time_embed,
        flow_time_embed_dim=args.flow_time_embed_dim,
        flow_checkpoint_blocks=args.flow_checkpoint_blocks,
        latent_max_tokens=args.latent_max_tokens,
        latent_patch_size=args.latent_patch_size,
        ae_arch=args.ae_arch,
        latent_downsample=args.latent_downsample, ae_res_blocks=args.ae_res_blocks,
        ae_hf_model=args.ae_hf_model,
        ae_hf_subfolder=args.ae_hf_subfolder,
        ae_hf_scaling_factor=args.ae_hf_scaling_factor,
        ae_recon_loss=args.ae_recon_loss, ae_grad_w=args.ae_grad_w,
        ae_ms_w=args.ae_ms_w, ae_fft_w=args.ae_fft_w,
        ae_latent_reg_w=args.ae_latent_reg_w,
        image_text_align_w=args.image_text_align_w,
        flow_text_align_w=args.flow_text_align_w,
        text_embed_dim=args.text_embed_dim,
        image_feature_align_w=args.image_feature_align_w,
        flow_feature_align_w=args.flow_feature_align_w,
        image_feature_embed_dim=args.image_feature_embed_dim,
        flow_repa_w=args.flow_repa_w,
        flow_repa_steps=args.flow_repa_steps,
        flow_repa_embed_dim=args.flow_repa_embed_dim,
        flow_repa_mode=args.flow_repa_mode,
        flow_self_repa_w=args.flow_self_repa_w,
        flow_self_repa_steps=args.flow_self_repa_steps,
        flow_self_repa_embed_dim=args.flow_self_repa_embed_dim,
        flow_self_repa_mode=args.flow_self_repa_mode,
        flow_sra_w=args.flow_sra_w,
        flow_sra_steps=args.flow_sra_steps,
        flow_sra_time_gap=args.flow_sra_time_gap,
        flow_sra_mode=args.flow_sra_mode,
        sample_steps=args.sample_steps, cond_mode=args.cond_mode,
        flow_consistency_w=args.flow_consistency_w,
        flow_endpoint_w=args.flow_endpoint_w,
        flow_noise_coupling=args.flow_noise_coupling,
        flow_noise_coupling_projections=args.flow_noise_coupling_projections,
        size_buckets=cli_size_buckets,
        size_curriculum_frac=args.size_curriculum_frac,
        flow_distill_steps=args.flow_distill_steps,
        flow_distill_w=args.flow_distill_w,
        flow_distill_time_gap=args.flow_distill_time_gap,
        flow_distill_teacher=args.flow_distill_teacher,
        flow_guidance_distill_w=args.flow_guidance_distill_w,
        flow_guidance_distill_cfg_scale=args.flow_guidance_distill_cfg_scale,
        flow_guidance_distill_cfg_rescale=args.flow_guidance_distill_cfg_rescale,
        text_cond_dim=args.text_cond_dim,
        image_manifest=args.image_manifest, image_root=args.image_root,
        image_split=args.image_split, image_min_aesthetic=args.image_min_aesthetic,
        image_max_records=args.image_max_records,
        image_quality_weight=args.image_quality_weight,
        image_source_weights=args.image_source_weights,
        image_quality_score_w=args.image_quality_score_w,
        flow_quality_score_w=args.flow_quality_score_w,
        quality_score_steps=args.quality_score_steps,
        image_quality_rank_w=args.image_quality_rank_w,
        quality_score_rank_w=args.quality_score_rank_w,
        flow_quality_rank_w=args.flow_quality_rank_w,
        quality_rank_margin=args.quality_rank_margin,
        image_preference_manifest=args.image_preference_manifest,
        image_preference_root=args.image_preference_root,
        image_preference_max_pairs=args.image_preference_max_pairs,
        image_preference_w=args.image_preference_w,
        flow_preference_w=args.flow_preference_w,
        flow_preference_margin=args.flow_preference_margin,
        flow_preference_batch=args.flow_preference_batch,
        image_geometry_cond=args.image_geometry_cond,
        caption_vocab_max=args.caption_vocab_max,
        caption_max_len=args.caption_max_len, caption_cond_source=args.caption_cond_source,
        image_crop_mode=args.image_crop_mode, image_hflip_prob=args.image_hflip_prob,
        time_sampling=args.time_sampling, time_logit_mean=args.time_logit_mean,
        time_logit_std=args.time_logit_std,
        time_curriculum_frac=args.time_curriculum_frac,
        time_adaptive_bins=args.time_adaptive_bins,
        time_adaptive_momentum=args.time_adaptive_momentum,
        time_adaptive_uniform_mix=args.time_adaptive_uniform_mix,
        time_adaptive_min_prob=args.time_adaptive_min_prob,
        time_adaptive_loss_power=args.time_adaptive_loss_power,
        time_shift=args.time_shift,
        time_shift_mode=args.time_shift_mode,
        time_shift_ref_dim=args.time_shift_ref_dim,
        time_shift_dim_power=args.time_shift_dim_power,
        flow_loss_weight=args.flow_loss_weight,
        flow_loss_weight_gamma=args.flow_loss_weight_gamma,
        flow_loss_weight_normalize=not args.no_flow_loss_weight_normalize,
        latent_normalize=args.latent_normalize,
        latent_stat_samples=args.latent_stat_samples,
        cfg_interval=cfg_interval,
        sample_method=args.sample_method,
        sample_schedule=args.sample_schedule,
        sample_churn=args.sample_churn,
        sample_churn_interval=sample_churn_interval,
        sample_finite_guard=args.sample_finite_guard,
        sample_velocity_clip=args.sample_velocity_clip,
        sample_latent_clip=args.sample_latent_clip,
        flow_ema_decay=args.flow_ema_decay,
        flow_ema_warmup=not args.no_ema_warmup,
        eval_with_ema=args.ema_eval_mode != "raw",
        eval_weight_mode=args.ema_eval_mode,
        eval_generated_candidates_per_prompt=args.eval_generated_candidates_per_prompt,
        train_precision=args.train_precision,
        ae_accum_steps=args.ae_accum_steps,
        flow_accum_steps=args.flow_accum_steps,
        grad_clip=args.grad_clip,
        flow_cache_latents=args.flow_cache_latents,
        flow_cache_records=args.flow_cache_records,
        flow_cache_batch=args.flow_cache_batch,
        flow_cache_dir=args.flow_cache_dir,
        flow_cache_shard_size=args.flow_cache_shard_size,
        flow_cache_dtype=args.flow_cache_dtype,
        flow_cache_max_loaded_shards=args.flow_cache_max_loaded_shards,
        return_conditioner=True, return_ema=True, return_aligner=True)
    if args.sample_grid_out:
        raw_flow = clone_state_dict(flow)
        raw_conditioner = clone_state_dict(conditioner) if conditioner is not None else None
        grid_weight_mode = "raw"
        if report.get("selected_eval_weights") == "ema" and flow_ema is not None:
            load_flow_state(flow, flow_ema)
            if conditioner is not None and conditioner_ema:
                conditioner.load_state_dict(conditioner_ema)
            grid_weight_mode = "ema"
        settings = selected_grid_settings(
            report,
            fallback_cfg=args.cfg_scale,
            fallback_cfg_rescale=args.cfg_rescale,
            fallback_steps=args.sample_steps,
            fallback_method=args.sample_method,
            fallback_sample_schedule=args.sample_schedule,
            fallback_sample_churn=args.sample_churn,
            fallback_sample_churn_interval=sample_churn_interval,
            fallback_sample_finite_guard=args.sample_finite_guard,
            fallback_sample_velocity_clip=args.sample_velocity_clip,
            fallback_sample_latent_clip=args.sample_latent_clip,
            fallback_cfg_mode=args.cfg_mode,
            fallback_cfg_schedule=args.cfg_schedule,
            fallback_cfg_interval=cfg_interval,
            fallback_sample_time_shift=report.get(
                "sample_time_shift", report.get("time_shift", args.time_shift)))
        if sample_prompts:
            grid_meta = save_text_prompt_sample_grid(
                ae, flow, sample_prompts, args.sample_grid_out, conditioner=conditioner,
                prompt_vocab=prompt_vocab, caption_max_len=args.caption_max_len,
                size=run_size,
                cfg_scale=settings["cfg_scale"],
                cfg_rescale=settings["cfg_rescale"],
                cfg_mode=settings["cfg_mode"],
                cfg_schedule=settings["cfg_schedule"],
                cfg_interval=settings["cfg_interval"],
                sample_steps=settings["sample_steps"],
                sample_time_shift=settings["sample_time_shift"],
                sample_method=settings["sample_method"],
                sample_schedule=settings["sample_schedule"],
                sample_churn=settings["sample_churn"],
                sample_churn_interval=settings["sample_churn_interval"],
                seed=args.seed + 991,
                caption_cond_source=report.get("caption_cond_source", "tokens") or "tokens",
                prompt_embed_backend=args.prompt_embed_backend,
                prompt_embed_model=args.prompt_embed_model,
                prompt_embed_device=args.prompt_embed_device or None,
                prompt_embed_dtype=args.prompt_embed_dtype,
                prompt_embed_normalize=not args.prompt_embed_no_normalize,
                prompt_embed_stats_dim=args.prompt_embed_stats_dim,
                prompt_embed_text_sequence_model=args.prompt_embed_text_sequence_model,
                prompt_embed_text_sequence_max_length=(
                    args.prompt_embed_text_sequence_max_length),
                prompt_embed_trust_remote_code=args.prompt_embed_trust_remote_code,
                negative_prompts=sample_negative_prompts,
                candidates_per_prompt=args.sample_candidates_per_prompt,
                text_guidance_w=args.sample_text_guidance_w,
                text_guidance_interval=sample_text_guidance_interval,
                feature_guidance_w=args.sample_feature_guidance_w,
                feature_guidance_interval=sample_feature_guidance_interval,
                quality_guidance_w=args.sample_quality_guidance_w,
                quality_guidance_interval=sample_quality_guidance_interval,
                sample_finite_guard=settings["sample_finite_guard"],
                sample_velocity_clip=settings["sample_velocity_clip"],
                sample_latent_clip=settings["sample_latent_clip"],
                sample_pixel_dynamic_threshold_percentile=(
                    args.sample_pixel_dynamic_threshold_percentile),
                sample_pixel_dynamic_threshold_max=args.sample_pixel_dynamic_threshold_max,
                sample_manifest_out=args.sample_manifest_out,
                sample_image_dir=args.sample_image_dir,
                sample_candidates_manifest_out=args.sample_candidates_manifest_out,
                sample_candidates_image_dir=args.sample_candidates_image_dir,
                candidate_cfg_scales=cfg_scales,
                candidate_cfg_rescales=cfg_rescales,
                candidate_cfg_modes=cfg_modes,
                candidate_cfg_schedules=cfg_schedules,
                candidate_sample_steps_list=sample_steps_list,
                candidate_sample_methods=sample_methods,
                candidate_sample_schedules=sample_schedules,
                candidate_sample_churns=sample_churns,
                candidate_seeds=eval_seeds)
        elif args.image_manifest:
            grid_records = read_image_manifest(
                args.image_manifest, root=args.image_root, split=args.image_split,
                min_aesthetic=args.image_min_aesthetic, max_records=args.image_max_records)
            grid_meta = save_caption_sample_grid(
                ae, flow, grid_records, args.sample_grid_out, conditioner=conditioner,
                prompt_vocab=prompt_vocab, caption_max_len=args.caption_max_len,
                size=run_size,
                cfg_scale=settings["cfg_scale"],
                cfg_rescale=settings["cfg_rescale"],
                cfg_mode=settings["cfg_mode"],
                cfg_schedule=settings["cfg_schedule"],
                cfg_interval=settings["cfg_interval"],
                sample_steps=settings["sample_steps"],
                sample_time_shift=settings["sample_time_shift"],
                sample_method=settings["sample_method"],
                sample_schedule=settings["sample_schedule"],
                sample_churn=settings["sample_churn"],
                sample_churn_interval=settings["sample_churn_interval"],
                sample_finite_guard=settings["sample_finite_guard"],
                sample_velocity_clip=settings["sample_velocity_clip"],
                sample_latent_clip=settings["sample_latent_clip"],
                sample_pixel_dynamic_threshold_percentile=(
                    args.sample_pixel_dynamic_threshold_percentile),
                sample_pixel_dynamic_threshold_max=args.sample_pixel_dynamic_threshold_max,
                samples=args.sample_grid_samples,
                seed=args.seed + 991,
                caption_cond_source=report.get("caption_cond_source", "tokens"),
                sample_manifest_out=args.sample_manifest_out,
                sample_image_dir=args.sample_image_dir)
        else:
            ap.error("--sample-grid-out requires --sample-prompts or --image-manifest")
        grid_meta["sample_grid_checkpoint_weight_mode"] = grid_weight_mode
        report.update(grid_meta)
        load_flow_state(flow, raw_flow)
        if conditioner is not None and raw_conditioner is not None:
            conditioner.load_state_dict(raw_conditioner)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "autoencoder_state_dict": (
            {} if report.get("ae_external", False) else ae.state_dict()
        ),
        "flow_state_dict": flow.state_dict(),
        "report": report,
        "latent_ch": report.get("latent_ch", args.latent_ch),
        "image_size": image_size_value(run_size),
        "image_h": int(run_size[0]),
        "image_w": int(run_size[1]),
        "size_buckets": report.get("size_buckets", []),
        "size_bucket_count": report.get("size_bucket_count", 0),
        "size_bucket_sampling_mode": report.get("size_bucket_sampling_mode", ""),
        "size_bucket_sampling_probs": report.get("size_bucket_sampling_probs", {}),
        "size_curriculum_frac": report.get(
            "size_curriculum_frac", args.size_curriculum_frac),
        "size_curriculum_switch_step": report.get("size_curriculum_switch_step", 0),
        "size_curriculum_bucket_order": report.get("size_curriculum_bucket_order", []),
        "ae_arch": args.ae_arch,
        "ae_external": report.get("ae_external", False),
        "ae_hf_model": args.ae_hf_model,
        "ae_hf_subfolder": args.ae_hf_subfolder,
        "ae_hf_scaling_factor": report.get("ae_hf_scaling_factor", args.ae_hf_scaling_factor),
        "latent_downsample": report.get("latent_downsample", args.latent_downsample),
        "ae_res_blocks": report.get("ae_res_blocks", args.ae_res_blocks),
        "latent_max_tokens": args.latent_max_tokens,
        "latent_patch_size": report.get("latent_patch_size", args.latent_patch_size),
        "ae_recon_loss": args.ae_recon_loss,
        "ae_grad_w": args.ae_grad_w,
        "ae_ms_w": args.ae_ms_w,
        "ae_fft_w": args.ae_fft_w,
        "ae_latent_reg_w": args.ae_latent_reg_w,
        "image_text_align_w": args.image_text_align_w,
        "flow_text_align_w": args.flow_text_align_w,
        "text_embed_dim": args.text_embed_dim,
        "image_feature_align_w": args.image_feature_align_w,
        "flow_feature_align_w": args.flow_feature_align_w,
        "image_feature_embed_dim": args.image_feature_embed_dim,
        "flow_repa_w": args.flow_repa_w,
        "flow_repa_steps": args.flow_repa_steps,
        "flow_repa_embed_dim": args.flow_repa_embed_dim,
        "flow_repa_mode": args.flow_repa_mode,
        "flow_self_repa_w": args.flow_self_repa_w,
        "flow_self_repa_steps": args.flow_self_repa_steps,
        "flow_self_repa_embed_dim": args.flow_self_repa_embed_dim,
        "flow_self_repa_mode": args.flow_self_repa_mode,
        "flow_sra_w": args.flow_sra_w,
        "flow_sra_steps": args.flow_sra_steps,
        "flow_sra_time_gap": args.flow_sra_time_gap,
        "flow_sra_mode": args.flow_sra_mode,
        "hidden": args.hidden,
        "ae_accum_steps": args.ae_accum_steps,
        "flow_accum_steps": args.flow_accum_steps,
        "flow_cache_latents": report.get("flow_cache_latents", False),
        "flow_cache_backend": report.get("flow_cache_backend", ""),
        "flow_cache_dir": report.get("flow_cache_dir", ""),
        "flow_cache_records": report.get("flow_cache_records", 0),
        "flow_cache_max_records": args.flow_cache_records,
        "flow_cache_batch": args.flow_cache_batch,
        "flow_cache_shard_size": args.flow_cache_shard_size,
        "flow_cache_reused": report.get("flow_cache_reused", False),
        "flow_cache_latent_dtype": report.get(
            "flow_cache_latent_dtype", args.flow_cache_dtype),
        "flow_cache_max_loaded_shards": report.get(
            "flow_cache_max_loaded_shards", args.flow_cache_max_loaded_shards),
        "flow_cache_loaded_shards": report.get("flow_cache_loaded_shards", 0),
        "flow_cache_shard_loads": report.get("flow_cache_shard_loads", 0),
        "flow_cache_shard_cache_hits": report.get("flow_cache_shard_cache_hits", 0),
        "flow_cache_shard_cache_misses": report.get("flow_cache_shard_cache_misses", 0),
        "flow_cache_shards": report.get("flow_cache_shards", 0),
        "flow_cache_bytes": report.get("flow_cache_bytes", 0),
        "flow_cache_weighted": report.get("flow_cache_weighted", False),
        "flow_cache_has_quality_targets": report.get("flow_cache_has_quality_targets", False),
        "train_precision": args.train_precision,
        "train_amp_enabled": report.get("train_amp_enabled", False),
        "grad_clip": args.grad_clip,
        "flow_arch": args.flow_arch,
        "dit_depth": args.dit_depth,
        "dit_heads": args.dit_heads,
        "dit_head_width_mult": args.dit_head_width_mult,
        "dit_qk_norm": report.get("dit_qk_norm", False),
        "dit_attn_impl": report.get("dit_attn_impl", "manual"),
        "dit_pos_embed": report.get("dit_pos_embed", "") or args.dit_pos_embed,
        "dit_mlp": report.get("dit_mlp", "") or args.dit_mlp,
        "flow_time_embed": report.get("flow_time_embed", args.flow_time_embed),
        "flow_time_embed_dim": report.get("flow_time_embed_dim", args.flow_time_embed_dim),
        "flow_checkpoint_blocks": report.get("flow_checkpoint_blocks", False),
        "flow_boundary_mode": report.get("flow_boundary_mode", args.flow_boundary_mode),
        "activation_checkpointing": report.get("activation_checkpointing", False),
        "zero_residual_gating": report.get("zero_residual_gating", False),
        "uses_2d_pos_embed": report.get("uses_2d_pos_embed", False),
        "uses_rope2d_pos_embed": report.get("uses_rope2d_pos_embed", False),
        "uses_swiglu_mlp": report.get("uses_swiglu_mlp", False),
        "cond_mode": args.cond_mode,
        "cond_dim": report["cond_dim"],
        "data_mode": report.get("data_mode", "image_manifest"),
        "image_manifest": args.image_manifest,
        "image_root": args.image_root,
        "image_split": args.image_split,
        "image_min_aesthetic": args.image_min_aesthetic,
        "image_quality_weight": args.image_quality_weight,
        "image_quality_weighted": report.get("image_quality_weighted", False),
        "image_source_weights": args.image_source_weights,
        "image_source_weighted": report.get("image_source_weighted", False),
        "image_source_counts": report.get("image_source_counts", {}),
        "image_sampling_weighted": report.get("image_sampling_weighted", False),
        "image_quality_score_w": args.image_quality_score_w,
        "flow_quality_score_w": args.flow_quality_score_w,
        "quality_score_steps": args.quality_score_steps,
        "quality_score_steps_run": report.get("quality_score_steps_run", 0),
        "image_quality_rank_w": args.image_quality_rank_w,
        "quality_score_rank_w": args.quality_score_rank_w,
        "flow_quality_rank_w": args.flow_quality_rank_w,
        "quality_rank_margin": args.quality_rank_margin,
        "image_preference_manifest": args.image_preference_manifest,
        "image_preference_root": args.image_preference_root,
        "image_preference_max_pairs": args.image_preference_max_pairs,
        "image_preference_w": args.image_preference_w,
        "image_preference_pairs": report.get("image_preference_pairs", 0),
        "image_preference_pairs_with_prompt": report.get(
            "image_preference_pairs_with_prompt", 0),
        "image_preference_pairs_with_text_embedding": report.get(
            "image_preference_pairs_with_text_embedding", 0),
        "image_preference_pairs_with_text_embedding_sequence": report.get(
            "image_preference_pairs_with_text_embedding_sequence", 0),
        "image_preference_pairs_direct_flow_usable": report.get(
            "image_preference_pairs_direct_flow_usable", 0),
        "image_preference_pairs_direct_flow_missing_condition": report.get(
            "image_preference_pairs_direct_flow_missing_condition", 0),
        "image_preference_pairs_with_sample_metadata": report.get(
            "image_preference_pairs_with_sample_metadata", 0),
        "image_preference_chosen_sample_methods": report.get(
            "image_preference_chosen_sample_methods", {}),
        "image_preference_rejected_sample_methods": report.get(
            "image_preference_rejected_sample_methods", {}),
        "image_preference_chosen_cfg_schedules": report.get(
            "image_preference_chosen_cfg_schedules", {}),
        "image_preference_rejected_cfg_schedules": report.get(
            "image_preference_rejected_cfg_schedules", {}),
        "flow_preference_w": args.flow_preference_w,
        "flow_preference_margin": args.flow_preference_margin,
        "flow_preference_batch": args.flow_preference_batch,
        "flow_preference_steps_run": report.get("flow_preference_steps_run", 0),
        "flow_preference_skipped": report.get("flow_preference_skipped", 0),
        "flow_preference_skip_reason": report.get("flow_preference_skip_reason", ""),
        "image_quality_scorer": report.get("image_quality_scorer", False),
        "image_max_records": args.image_max_records,
        "image_crop_mode": report.get("image_crop_mode", ""),
        "image_hflip_prob": report.get("image_hflip_prob", 0.0),
        "caption_vocab_max": args.caption_vocab_max,
        "caption_max_len": report.get("caption_max_len", 0),
        "caption_cond_source": report.get("caption_cond_source", ""),
        "text_embedding_in_dim": report.get("text_embedding_in_dim", 0),
        "text_pooled_embedding_in_dim": report.get("text_pooled_embedding_in_dim", 0),
        "text_condition_dual_pooled_sequence": report.get(
            "text_condition_dual_pooled_sequence", False),
        "image_geometry_cond": report.get("image_geometry_cond", False),
        "image_embedding_in_dim": report.get("image_embedding_in_dim", 0),
        "cond_drop": args.cond_drop,
        "cfg_scale": args.cfg_scale,
        "cfg_rescale": args.cfg_rescale,
        "cfg_mode": args.cfg_mode,
        "cfg_schedule": args.cfg_schedule,
        "flow_boundary_mode": report.get("flow_boundary_mode", args.flow_boundary_mode),
        "cfg_interval": list(cfg_interval),
        "sample_steps": args.sample_steps,
        "sample_method": args.sample_method,
        "sample_schedule": args.sample_schedule,
        "sample_churn": args.sample_churn,
        "sample_churn_interval": list(sample_churn_interval),
        "sample_finite_guard": args.sample_finite_guard,
        "sample_velocity_clip": args.sample_velocity_clip,
        "sample_latent_clip": args.sample_latent_clip,
        "sample_pixel_dynamic_threshold_percentile": (
            args.sample_pixel_dynamic_threshold_percentile),
        "sample_pixel_dynamic_threshold_max": args.sample_pixel_dynamic_threshold_max,
        "flow_consistency_w": args.flow_consistency_w,
        "flow_distill_steps": report.get("flow_distill_steps", args.flow_distill_steps),
        "flow_distill_steps_run": report.get("flow_distill_steps_run", 0),
        "flow_distill_w": report.get("flow_distill_w", args.flow_distill_w),
        "flow_distill_time_gap": report.get(
            "flow_distill_time_gap", args.flow_distill_time_gap),
        "flow_distill_teacher": report.get("flow_distill_teacher", args.flow_distill_teacher),
        "flow_distill_teacher_used": report.get("flow_distill_teacher_used", ""),
        "flow_guidance_distill_w": report.get(
            "flow_guidance_distill_w", args.flow_guidance_distill_w),
        "flow_guidance_distill_cfg_scale": report.get(
            "flow_guidance_distill_cfg_scale", args.flow_guidance_distill_cfg_scale),
        "flow_guidance_distill_cfg_rescale": report.get(
            "flow_guidance_distill_cfg_rescale", args.flow_guidance_distill_cfg_rescale),
        "flow_ema_decay": args.flow_ema_decay,
        "flow_ema_warmup": not args.no_ema_warmup,
        "flow_ema_effective_decay": report.get("flow_ema_effective_decay", 0.0),
        "eval_weight_mode": report.get("eval_weight_mode", "raw"),
        "selected_eval_weights": report.get("selected_eval_weights", "raw"),
        "time_sampling": args.time_sampling,
        "time_curriculum_frac": report.get("time_curriculum_frac", args.time_curriculum_frac),
        "time_curriculum_switch_step": report.get("time_curriculum_switch_step", 0),
        "time_curriculum_final_sampling": report.get(
            "time_curriculum_final_sampling", args.time_sampling),
        "time_logit_mean": args.time_logit_mean,
        "time_logit_std": args.time_logit_std,
        "time_adaptive_requested": report.get(
            "time_adaptive_requested", args.time_sampling == "adaptive"),
        "time_adaptive_enabled": report.get("time_adaptive_enabled", False),
        "time_adaptive_bins": report.get("time_adaptive_bins", args.time_adaptive_bins),
        "time_adaptive_momentum": report.get(
            "time_adaptive_momentum", args.time_adaptive_momentum),
        "time_adaptive_uniform_mix": report.get(
            "time_adaptive_uniform_mix", args.time_adaptive_uniform_mix),
        "time_adaptive_min_prob": report.get(
            "time_adaptive_min_prob", args.time_adaptive_min_prob),
        "time_adaptive_loss_power": report.get(
            "time_adaptive_loss_power", args.time_adaptive_loss_power),
        "time_adaptive_updates": report.get("time_adaptive_updates", 0),
        "time_adaptive_observed_bins": report.get("time_adaptive_observed_bins", 0),
        "time_adaptive_prob_min": report.get("time_adaptive_prob_min", 0.0),
        "time_adaptive_prob_max": report.get("time_adaptive_prob_max", 0.0),
        "time_adaptive_prob_entropy": report.get("time_adaptive_prob_entropy", 0.0),
        "time_shift": report.get("time_shift", args.time_shift),
        "time_shift_requested": args.time_shift,
        "time_shift_mode": args.time_shift_mode,
        "time_shift_effective_mode": report.get(
            "time_shift_effective_mode", args.time_shift_mode),
        "time_shift_auto_policy": report.get("time_shift_auto_policy", ""),
        "time_shift_ref_dim": args.time_shift_ref_dim,
        "time_shift_dim_power": args.time_shift_dim_power,
        "latent_effective_dim": report.get("latent_effective_dim", 0),
        "time_shift_dim_scale": report.get("time_shift_dim_scale", 1.0),
        "flow_loss_weight": args.flow_loss_weight,
        "flow_noise_coupling": report.get("flow_noise_coupling", args.flow_noise_coupling),
        "flow_noise_coupling_projections": report.get(
            "flow_noise_coupling_projections", args.flow_noise_coupling_projections),
        "flow_endpoint_w": report.get("flow_endpoint_w", args.flow_endpoint_w),
        "flow_loss_weight_gamma": args.flow_loss_weight_gamma,
        "flow_loss_weight_normalize": not args.no_flow_loss_weight_normalize,
        "sample_time_shift": report.get("sample_time_shift", report.get("time_shift", args.time_shift)),
        "latent_normalize": report.get("latent_normalize", args.latent_normalize),
        "latent_normalize_requested": args.latent_normalize,
        "latent_stat_samples": args.latent_stat_samples,
        "latent_stats": latent_stats_state(flow_latent_stats(flow)),
        "prompt_templates": [],
        "prompt_vocab": prompt_vocab if prompt_vocab is not None else {},
        "conditioner_state_dict": (conditioner.state_dict() if conditioner is not None else {}),
        "text_aligner_state_dict": (
            text_aligner.state_dict() if text_aligner is not None else {}
        ),
        "image_feature_aligner_state_dict": (
            getattr(flow, "image_feature_aligner", None).state_dict()
            if getattr(flow, "image_feature_aligner", None) is not None else {}
        ),
        "image_quality_scorer_state_dict": (
            getattr(flow, "image_quality_scorer", None).state_dict()
            if getattr(flow, "image_quality_scorer", None) is not None else {}
        ),
        "flow_repa_aligner_state_dict": (
            getattr(flow, "flow_repa_aligner", None).state_dict()
            if getattr(flow, "flow_repa_aligner", None) is not None else {}
        ),
        "flow_self_repa_aligner_state_dict": (
            getattr(flow, "flow_self_repa_aligner", None).state_dict()
            if getattr(flow, "flow_self_repa_aligner", None) is not None else {}
        ),
        "flow_ema_state_dict": flow_ema if flow_ema is not None else {},
        "conditioner_ema_state_dict": (conditioner_ema if conditioner_ema is not None else {}),
    }, args.out)
    print(json.dumps(report, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
