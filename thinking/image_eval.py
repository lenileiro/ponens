"""Standalone embedding-space evaluation for image generation data.

Training reports in `image_latent` already score checkpoint samples.  This module
keeps the same idea outside the checkpoint path: compare any real/generated
manifest pair that has image/text embedding fields, including web corpora,
recaptioned data, generated sample manifests, or future multimodal image outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .image_data import (
    ImageTextRecord,
    _coerce_record,
    caption_tokens,
    load_image_tensor,
    merge_embedding_sidecar,
    read_image_manifest,
    summarize_records,
)


EMBEDDING_KINDS = ("image", "text", "image_sequence", "text_sequence")
DIM_POLICIES = ("error", "largest")
EPS = 1.0e-12
VISUAL_FEATURE_NAMES = (
    "layout_mass",
    "layout_cx",
    "layout_cy",
    "layout_var_x",
    "layout_var_y",
    "layout_cov_xy",
    "edge_mass",
    "edge_cx",
    "edge_cy",
    "edge_var_x",
    "edge_var_y",
    "edge_cov_xy",
    "detail_energy",
    "luminance_std",
    "dynamic_range",
    "channel_std",
)


@dataclass(frozen=True)
class MatrixRows:
    matrix: np.ndarray
    records: tuple[ImageTextRecord, ...]
    report: dict


def _read_rows_stream(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    if ext in (".csv", ".tsv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            delimiter = "\t" if ext == ".tsv" else ","
            try:
                has_header = csv.Sniffer().has_header(sample) if sample else True
            except csv.Error:
                has_header = True
            if has_header:
                yield from csv.DictReader(f, delimiter=delimiter)
            else:
                for row in csv.reader(f, delimiter=delimiter):
                    if not row:
                        continue
                    yield {
                        "image": row[0],
                        "caption": row[1] if len(row) > 1 else "",
                        "split": row[2] if len(row) > 2 else "train",
                    }
        return
    raise ValueError(f"unsupported image manifest extension: {ext!r}")


def iter_image_records(path, root="", split="", min_aesthetic=None, max_records=0,
                       max_nsfw=None, max_watermark=None):
    manifest_dir = os.path.dirname(os.path.abspath(path))
    kept = 0
    for row in _read_rows_stream(path):
        rec = _coerce_record(row, manifest_dir=manifest_dir, root=root)
        if split and rec.split != split:
            continue
        if min_aesthetic is not None and rec.aesthetic is not None:
            if rec.aesthetic < float(min_aesthetic):
                continue
        if max_nsfw is not None and rec.nsfw is not None:
            if rec.nsfw > float(max_nsfw):
                continue
        if max_watermark is not None and rec.watermark is not None:
            if rec.watermark > float(max_watermark):
                continue
        yield rec
        kept += 1
        if max_records and kept >= int(max_records):
            break


def load_records(path, root="", split="", min_aesthetic=None, max_records=0,
                 max_nsfw=None, max_watermark=None, embedding_sidecar="",
                 embedding_key="image", embedding_overwrite=False):
    if embedding_sidecar:
        records = read_image_manifest(
            path, root=root, split=split, min_aesthetic=min_aesthetic,
            max_records=max_records, max_nsfw=max_nsfw, max_watermark=max_watermark)
        merged, merge_report = merge_embedding_sidecar(
            records, embedding_sidecar, root=root, key=embedding_key,
            overwrite=embedding_overwrite)
        return tuple(merged), merge_report
    records = tuple(iter_image_records(
        path, root=root, split=split, min_aesthetic=min_aesthetic,
        max_records=max_records, max_nsfw=max_nsfw, max_watermark=max_watermark))
    if not records:
        raise ValueError(f"image manifest {path!r} yielded no records for split={split!r}")
    return records, {}


def _pool_sequence(seq):
    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        return None
    mask = np.abs(arr).sum(axis=1) > 0.0
    if not np.any(mask):
        mask = np.ones(arr.shape[0], dtype=bool)
    return arr[mask].mean(axis=0)


def record_embedding(rec: ImageTextRecord, kind="image"):
    if kind == "image":
        raw = rec.image_embedding
    elif kind == "text":
        raw = rec.text_embedding
    elif kind == "image_sequence":
        raw = rec.image_embedding_sequence
    elif kind == "text_sequence":
        raw = rec.text_embedding_sequence
    else:
        raise ValueError(f"unknown embedding kind {kind!r}")
    if raw is None:
        return None
    if kind.endswith("_sequence"):
        return _pool_sequence(raw)
    return np.asarray(raw, dtype=np.float64)


def _normalize_rows(x, eps=1.0e-12):
    x = np.asarray(x, dtype=np.float64)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, float(eps))


def _weighted_moments_2d(weight, xx, yy, eps=1.0e-12):
    denom = max(float(np.sum(weight)), float(eps))
    cx = float(np.sum(weight * xx) / denom)
    cy = float(np.sum(weight * yy) / denom)
    dx = xx - cx
    dy = yy - cy
    return [
        float(denom / max(1, int(weight.shape[0]) * int(weight.shape[1]))),
        cx,
        cy,
        float(np.sum(weight * dx * dx) / denom),
        float(np.sum(weight * dy * dy) / denom),
        float(np.sum(weight * dx * dy) / denom),
    ]


def visual_physics_feature(image, eps=1.0e-12):
    """Return category-free visual structure stats for an image tensor/array.

    The feature vector mirrors the training-side visual physics loss: visual
    mass, edge mass, centroids, spread, orientation, and simple image-health
    statistics. It does not encode object labels, grammar, or color rules.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"visual physics feature expects CHW image, got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    _c, h, w = arr.shape
    if h <= 0 or w <= 0:
        raise ValueError("visual physics feature requires non-empty spatial dimensions")
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, int(h), dtype=np.float64),
        np.linspace(-1.0, 1.0, int(w), dtype=np.float64),
        indexing="ij",
    )
    mass = np.mean(arr * arr, axis=0)
    dx = arr[:, :, 1:] - arr[:, :, :-1]
    dy = arr[:, 1:, :] - arr[:, :-1, :]
    dx_energy = np.pad(np.mean(dx * dx, axis=0), ((0, 0), (0, 1)))
    dy_energy = np.pad(np.mean(dy * dy, axis=0), ((0, 1), (0, 0)))
    edge = np.sqrt(dx_energy + dy_energy + float(eps))
    layout = _weighted_moments_2d(mass + edge, xx, yy, eps=eps)
    edge_layout = _weighted_moments_2d(edge, xx, yy, eps=eps)
    lum = arr.mean(axis=0)
    detail = float(np.sqrt(np.mean(dx_energy + dy_energy)))
    dynamic = float(np.max(lum) - np.min(lum))
    channel_std = float(np.mean(np.std(arr.reshape(arr.shape[0], -1), axis=1)))
    return np.asarray(
        layout + edge_layout + [
            detail,
            float(np.std(lum)),
            dynamic,
            channel_std,
        ],
        dtype=np.float64,
    )


def _visual_feature_summary(mat, prefix):
    mat = np.asarray(mat, dtype=np.float64)
    report = {
        f"{prefix}_feature_names": list(VISUAL_FEATURE_NAMES),
        f"{prefix}_n": int(mat.shape[0]) if mat.ndim == 2 else 0,
    }
    if mat.ndim != 2 or mat.shape[0] <= 0:
        for name in ("detail_energy", "luminance_std", "dynamic_range", "channel_std"):
            report[f"{prefix}_{name}_mean"] = 0.0
            report[f"{prefix}_{name}_p05"] = 0.0
        return report
    for name in ("detail_energy", "luminance_std", "dynamic_range", "channel_std"):
        idx = VISUAL_FEATURE_NAMES.index(name)
        vals = mat[:, idx]
        report[f"{prefix}_{name}_mean"] = float(np.mean(vals))
        report[f"{prefix}_{name}_p05"] = _finite_percentile(vals, 5)
    return report


def visual_physics_matrix(records: Sequence[ImageTextRecord], size=64,
                          max_records=0, prefix="visual_physics"):
    size = int(size or 0)
    if size <= 0:
        return MatrixRows(
            matrix=np.zeros((0, len(VISUAL_FEATURE_NAMES)), dtype=np.float64),
            records=tuple(),
            report={
                f"{prefix}_enabled": False,
                f"{prefix}_size": int(size),
                f"{prefix}_records": int(len(records)),
                f"{prefix}_usable": 0,
                f"{prefix}_failed": 0,
                f"{prefix}_max_records": int(max_records or 0),
                f"{prefix}_feature_names": list(VISUAL_FEATURE_NAMES),
            },
        )
    rows = []
    kept = []
    failures = []
    limit = int(max_records or 0)
    for rec in records:
        if limit and len(kept) + len(failures) >= limit:
            break
        try:
            img = load_image_tensor(
                rec.path, size=size, device="cpu", crop_mode="center")
            rows.append(visual_physics_feature(img.detach().cpu().numpy()))
            kept.append(rec)
        except Exception as exc:
            failures.append({"image": rec.path, "error": str(exc)[:200]})
    matrix = (
        np.stack(rows, axis=0).astype(np.float64, copy=False)
        if rows else np.zeros((0, len(VISUAL_FEATURE_NAMES)), dtype=np.float64)
    )
    report = {
        f"{prefix}_enabled": True,
        f"{prefix}_size": int(size),
        f"{prefix}_records": int(len(records)),
        f"{prefix}_usable": int(matrix.shape[0]),
        f"{prefix}_failed": int(len(failures)),
        f"{prefix}_max_records": int(limit),
        f"{prefix}_failures": failures[:5],
        f"{prefix}_feature_names": list(VISUAL_FEATURE_NAMES),
    }
    report.update(_visual_feature_summary(matrix, prefix))
    return MatrixRows(matrix=matrix, records=tuple(kept), report=report)


def _clamp01(x):
    if x is None or not np.isfinite(float(x)):
        return 0.0
    return float(min(1.0, max(0.0, float(x))))


def _lower_is_better_score(x):
    if x is None or not np.isfinite(float(x)):
        return 0.0
    return float(1.0 / (1.0 + max(0.0, float(x))))


def _ratio_target_one_score(x):
    if x is None or not np.isfinite(float(x)) or float(x) <= 0.0:
        return 0.0
    return float(np.exp(-abs(np.log(max(float(x), EPS)))))


def _cosine_score(x):
    if x is None or not np.isfinite(float(x)):
        return 0.0
    return _clamp01((float(x) + 1.0) * 0.5)


def _weighted_geomean(parts):
    usable = [
        (name, _clamp01(value), float(weight))
        for name, value, weight in parts
        if weight > 0.0 and value is not None and np.isfinite(float(value))
    ]
    if not usable:
        return 0.0, {}
    total_w = sum(weight for _name, _value, weight in usable)
    score = float(np.exp(sum(
        weight * np.log(max(value, EPS)) for _name, value, weight in usable
    ) / max(total_w, EPS)))
    return score, {name: {"score": value, "weight": weight} for name, value, weight in usable}


def embedding_matrix(records: Sequence[ImageTextRecord], kind="image",
                     dim_policy="error", normalize=True, prefix="embedding"):
    if kind not in EMBEDDING_KINDS:
        raise ValueError(f"unknown embedding kind {kind!r}")
    if dim_policy not in DIM_POLICIES:
        raise ValueError(f"unknown dim policy {dim_policy!r}")
    rows = []
    kept_records = []
    missing = nonfinite = empty = 0
    dims = Counter()
    raw_rows = []
    raw_records = []
    for rec in records:
        vec = record_embedding(rec, kind=kind)
        if vec is None:
            missing += 1
            continue
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)
        if vec.size <= 0:
            empty += 1
            continue
        if not np.all(np.isfinite(vec)):
            nonfinite += 1
            continue
        dims[int(vec.size)] += 1
        raw_rows.append(vec)
        raw_records.append(rec)
    if raw_rows:
        if len(dims) > 1:
            if dim_policy == "error":
                raise ValueError(
                    f"{prefix} {kind} embeddings have mixed dims: {dict(sorted(dims.items()))}"
                )
            target_dim = dims.most_common(1)[0][0]
        else:
            target_dim = next(iter(dims))
        for rec, vec in zip(raw_records, raw_rows):
            if int(vec.size) != int(target_dim):
                continue
            rows.append(vec)
            kept_records.append(rec)
    skipped_dim = len(raw_rows) - len(rows)
    if not rows:
        mat = np.zeros((0, 0), dtype=np.float64)
    else:
        mat = np.stack(rows, axis=0).astype(np.float64, copy=False)
        if normalize:
            mat = _normalize_rows(mat)
    report = {
        f"{prefix}_{kind}_records": int(len(records)),
        f"{prefix}_{kind}_usable": int(mat.shape[0]),
        f"{prefix}_{kind}_missing": int(missing),
        f"{prefix}_{kind}_empty": int(empty),
        f"{prefix}_{kind}_nonfinite": int(nonfinite),
        f"{prefix}_{kind}_skipped_dim": int(skipped_dim),
        f"{prefix}_{kind}_dims": dict(sorted(dims.items())),
        f"{prefix}_{kind}_dim_policy": dim_policy,
        f"{prefix}_{kind}_normalized": bool(normalize),
    }
    if mat.size:
        report[f"{prefix}_{kind}_dim"] = int(mat.shape[1])
    return MatrixRows(matrix=mat, records=tuple(kept_records), report=report)


def _sqdist(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = (
        np.sum(a * a, axis=1, keepdims=True)
        + np.sum(b * b, axis=1, keepdims=True).T
        - 2.0 * a.dot(b.T)
    )
    return np.maximum(out, 0.0)


def _pairwise_l2_mean(x):
    n = int(x.shape[0])
    if n <= 1:
        return 0.0, 0.0
    dist = np.sqrt(_sqdist(x, x))
    mask = ~np.eye(n, dtype=bool)
    vals = dist[mask]
    cos = x.dot(x.T)[mask]
    return float(vals.mean()), float(cos.mean())


def _support_radius(x):
    n = int(x.shape[0])
    if n <= 1:
        return 0.0
    dist = np.sqrt(_sqdist(x, x))
    dist[np.eye(n, dtype=bool)] = np.inf
    nearest = dist.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    return float(np.median(finite)) if finite.size else 0.0


def _finite_percentile(vals, q, default=0.0):
    vals = np.asarray(vals, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return float(default)
    return float(np.percentile(vals, float(q)))


def _nearest_stats(vals, prefix):
    vals = np.asarray(vals, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_min": 0.0,
            f"{prefix}_p01": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_n": int(vals.size),
        f"{prefix}_min": float(vals.min()),
        f"{prefix}_p01": _finite_percentile(vals, 1),
        f"{prefix}_p05": _finite_percentile(vals, 5),
        f"{prefix}_p10": _finite_percentile(vals, 10),
        f"{prefix}_p50": _finite_percentile(vals, 50),
        f"{prefix}_mean": float(vals.mean()),
        f"{prefix}_p90": _finite_percentile(vals, 90),
        f"{prefix}_max": float(vals.max()),
    }


def _self_nearest_l2(x):
    n = int(x.shape[0])
    if n <= 1:
        return np.zeros((0,), dtype=np.float64)
    dist = np.sqrt(_sqdist(x, x))
    dist[np.eye(n, dtype=bool)] = np.inf
    vals = dist.min(axis=1)
    return vals[np.isfinite(vals)]


def embedding_neighborhood_metrics(generated, real, prefix="image_distribution"):
    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if generated.ndim != 2 or real.ndim != 2:
        raise ValueError("generated and real embeddings must be matrices")
    if generated.shape[0] <= 0 or real.shape[0] <= 0:
        out = {
            f"{prefix}_generated_nearest_generated_l2_n": 0,
            f"{prefix}_real_nearest_real_l2_n": 0,
            f"{prefix}_generated_nearest_real_l2_n": 0,
            f"{prefix}_real_nearest_generated_l2_n": 0,
            f"{prefix}_generated_duplicate_l2_ratio_to_real_p05": 0.0,
            f"{prefix}_generated_nearest_real_l2_ratio_to_real_p01": 0.0,
            f"{prefix}_generated_nearest_real_l2_ratio_to_real_p05": 0.0,
        }
        return out
    if generated.shape[1] != real.shape[1]:
        raise ValueError(
            f"generated/real embedding dims differ: {generated.shape[1]} vs {real.shape[1]}"
        )
    gen_self = _self_nearest_l2(generated)
    real_self = _self_nearest_l2(real)
    cross = np.sqrt(_sqdist(generated, real))
    gen_to_real = cross.min(axis=1)
    real_to_gen = cross.min(axis=0)
    real_p05 = max(_finite_percentile(real_self, 5), EPS)
    out = {}
    out.update(_nearest_stats(
        gen_self, f"{prefix}_generated_nearest_generated_l2"))
    out.update(_nearest_stats(
        real_self, f"{prefix}_real_nearest_real_l2"))
    out.update(_nearest_stats(
        gen_to_real, f"{prefix}_generated_nearest_real_l2"))
    out.update(_nearest_stats(
        real_to_gen, f"{prefix}_real_nearest_generated_l2"))
    out[f"{prefix}_generated_duplicate_l2_ratio_to_real_p05"] = float(
        _finite_percentile(gen_self, 5) / real_p05
        if gen_self.size else 0.0)
    out[f"{prefix}_generated_nearest_real_l2_ratio_to_real_p01"] = float(
        _finite_percentile(gen_to_real, 1) / real_p05
        if gen_to_real.size else 0.0)
    out[f"{prefix}_generated_nearest_real_l2_ratio_to_real_p05"] = float(
        _finite_percentile(gen_to_real, 5) / real_p05
        if gen_to_real.size else 0.0)
    return out


def _trace_sqrt_product(a, b):
    vals = np.linalg.eigvals(a.dot(b)).real
    vals = np.maximum(vals, 0.0)
    return float(np.sqrt(vals).sum())


def _rbf_mmd(x, y, eps=1.0e-12):
    z = np.concatenate([x, y], axis=0)
    if z.shape[0] <= 1:
        sigma2 = 1.0
    else:
        d = _sqdist(z, z)
        vals = d[np.triu_indices(d.shape[0], k=1)]
        vals = vals[vals > eps]
        sigma2 = float(np.median(vals)) if vals.size else 1.0
        sigma2 = max(sigma2, eps)
    kxx = np.exp(-_sqdist(x, x) / (2.0 * sigma2)).mean()
    kyy = np.exp(-_sqdist(y, y) / (2.0 * sigma2)).mean()
    kxy = np.exp(-_sqdist(x, y) / (2.0 * sigma2)).mean()
    return float(max(kxx + kyy - 2.0 * kxy, 0.0)), float(sigma2)


def embedding_distribution_metrics(generated, real, prefix="image_embedding_distribution"):
    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if generated.ndim != 2 or real.ndim != 2:
        raise ValueError("generated and real embeddings must be matrices")
    if generated.shape[0] <= 0 or real.shape[0] <= 0:
        return {
            f"{prefix}_generated_n": int(generated.shape[0]),
            f"{prefix}_real_n": int(real.shape[0]),
            f"{prefix}_dim": int(generated.shape[1] if generated.ndim == 2 else 0),
            f"{prefix}_frechet": 0.0,
            f"{prefix}_mmd_rbf": 0.0,
            f"{prefix}_support_precision": 0.0,
            f"{prefix}_support_recall": 0.0,
        }
    if generated.shape[1] != real.shape[1]:
        raise ValueError(
            f"generated/real embedding dims differ: {generated.shape[1]} vs {real.shape[1]}"
        )
    gen_mean = generated.mean(axis=0)
    real_mean = real.mean(axis=0)
    gen_cov = (
        np.cov(generated, rowvar=False)
        if generated.shape[0] > 1 else np.zeros((generated.shape[1], generated.shape[1]))
    )
    real_cov = (
        np.cov(real, rowvar=False)
        if real.shape[0] > 1 else np.zeros((real.shape[1], real.shape[1]))
    )
    gen_cov = np.atleast_2d(gen_cov)
    real_cov = np.atleast_2d(real_cov)
    mean_gap = gen_mean - real_mean
    cov_gap = gen_cov - real_cov
    frechet = (
        float(mean_gap.dot(mean_gap))
        + float(np.trace(gen_cov))
        + float(np.trace(real_cov))
        - 2.0 * _trace_sqrt_product(gen_cov, real_cov)
    )
    frechet = max(float(frechet), 0.0)
    mmd, sigma2 = _rbf_mmd(generated, real)
    gen_pair_l2, gen_pair_cos = _pairwise_l2_mean(generated)
    real_pair_l2, real_pair_cos = _pairwise_l2_mean(real)
    cross = np.sqrt(_sqdist(generated, real))
    nearest_real = cross.min(axis=1)
    nearest_generated = cross.min(axis=0)
    real_radius = _support_radius(real)
    gen_radius = _support_radius(generated)
    paired_n = min(int(generated.shape[0]), int(real.shape[0]))
    paired_cos = (
        float(np.sum(generated[:paired_n] * real[:paired_n], axis=1).mean())
        if paired_n > 0 else 0.0
    )
    return {
        f"{prefix}_generated_n": int(generated.shape[0]),
        f"{prefix}_real_n": int(real.shape[0]),
        f"{prefix}_paired_n": int(paired_n),
        f"{prefix}_dim": int(generated.shape[1]),
        f"{prefix}_paired_cos": paired_cos,
        f"{prefix}_mean_gap_l2": float(np.linalg.norm(mean_gap)),
        f"{prefix}_cov_fro": float(np.linalg.norm(cov_gap)),
        f"{prefix}_frechet": frechet,
        f"{prefix}_mmd_rbf": mmd,
        f"{prefix}_mmd_rbf_sigma2": sigma2,
        f"{prefix}_generated_pairwise_l2": gen_pair_l2,
        f"{prefix}_real_pairwise_l2": real_pair_l2,
        f"{prefix}_generated_pairwise_cos": gen_pair_cos,
        f"{prefix}_real_pairwise_cos": real_pair_cos,
        f"{prefix}_diversity_l2_ratio": float(gen_pair_l2 / max(real_pair_l2, 1.0e-12)),
        f"{prefix}_nearest_real_l2": float(nearest_real.mean()),
        f"{prefix}_nearest_generated_l2": float(nearest_generated.mean()),
        f"{prefix}_real_support_radius": real_radius,
        f"{prefix}_generated_support_radius": gen_radius,
        f"{prefix}_support_precision": float(np.mean(nearest_real <= real_radius)),
        f"{prefix}_support_recall": float(np.mean(nearest_generated <= gen_radius)),
    }


def visual_physics_distribution_metrics(generated, real, prefix="visual_physics_distribution"):
    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if generated.ndim != 2 or real.ndim != 2:
        raise ValueError("generated and real visual physics features must be matrices")
    if generated.shape[0] <= 0 or real.shape[0] <= 0:
        out = {
            f"{prefix}_generated_n": int(generated.shape[0]) if generated.ndim == 2 else 0,
            f"{prefix}_real_n": int(real.shape[0]) if real.ndim == 2 else 0,
            f"{prefix}_available": False,
            f"{prefix}_mean_feature_l1": 0.0,
            f"{prefix}_detail_energy_ratio": 0.0,
            f"{prefix}_dynamic_range_ratio": 0.0,
            f"{prefix}_luminance_std_ratio": 0.0,
        }
        return out
    if generated.shape[1] != real.shape[1]:
        raise ValueError(
            f"generated/real visual feature dims differ: "
            f"{generated.shape[1]} vs {real.shape[1]}"
        )
    out = embedding_distribution_metrics(generated, real, prefix=prefix)
    out.update(embedding_neighborhood_metrics(generated, real, prefix=prefix))
    gen_mean = generated.mean(axis=0)
    real_mean = real.mean(axis=0)
    detail_idx = VISUAL_FEATURE_NAMES.index("detail_energy")
    dynamic_idx = VISUAL_FEATURE_NAMES.index("dynamic_range")
    lum_idx = VISUAL_FEATURE_NAMES.index("luminance_std")
    out.update({
        f"{prefix}_available": True,
        f"{prefix}_mean_feature_l1": float(np.mean(np.abs(gen_mean - real_mean))),
        f"{prefix}_detail_energy_ratio": float(
            gen_mean[detail_idx] / max(real_mean[detail_idx], EPS)),
        f"{prefix}_dynamic_range_ratio": float(
            gen_mean[dynamic_idx] / max(real_mean[dynamic_idx], EPS)),
        f"{prefix}_luminance_std_ratio": float(
            gen_mean[lum_idx] / max(real_mean[lum_idx], EPS)),
    })
    return out


def image_text_alignment_metrics(records: Sequence[ImageTextRecord], prefix="image_text_alignment",
                                 dim_policy="error", normalize=True):
    image_rows = []
    text_rows = []
    paired_records = []
    missing = nonfinite = dim_mismatch = 0
    dims = Counter()
    for rec in records:
        image = record_embedding(rec, "image")
        text = record_embedding(rec, "text")
        if image is None or text is None:
            missing += 1
            continue
        image = np.asarray(image, dtype=np.float64).reshape(-1)
        text = np.asarray(text, dtype=np.float64).reshape(-1)
        if image.size != text.size:
            dim_mismatch += 1
            continue
        if not (np.all(np.isfinite(image)) and np.all(np.isfinite(text))):
            nonfinite += 1
            continue
        dims[int(image.size)] += 1
        image_rows.append(image)
        text_rows.append(text)
        paired_records.append(rec)
    if image_rows and len(dims) > 1:
        if dim_policy == "error":
            raise ValueError(
                f"{prefix} image/text embeddings have mixed dims: "
                f"{dict(sorted(dims.items()))}"
            )
        target = dims.most_common(1)[0][0]
        keep = [i for i, row in enumerate(image_rows) if int(row.size) == int(target)]
        image_rows = [image_rows[i] for i in keep]
        text_rows = [text_rows[i] for i in keep]
        paired_records = [paired_records[i] for i in keep]
    if not image_rows:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_missing": int(missing),
            f"{prefix}_nonfinite": int(nonfinite),
            f"{prefix}_dim_mismatch": int(dim_mismatch),
            f"{prefix}_dims": dict(sorted(dims.items())),
        }
    image = np.stack(image_rows, axis=0)
    text = np.stack(text_rows, axis=0)
    if normalize:
        image = _normalize_rows(image)
        text = _normalize_rows(text)
    logits = image.dot(text.T)
    n = int(logits.shape[0])
    targets = np.arange(n)
    diag = np.diag(logits)
    if n > 1:
        offdiag = logits.copy()
        offdiag[targets, targets] = -np.inf
        hardest = offdiag.max(axis=1)
        margin = diag - hardest
    else:
        margin = np.array([diag[0]], dtype=np.float64)
    return {
        f"{prefix}_n": n,
        f"{prefix}_missing": int(missing),
        f"{prefix}_nonfinite": int(nonfinite),
        f"{prefix}_dim_mismatch": int(dim_mismatch),
        f"{prefix}_dims": dict(sorted(dims.items())),
        f"{prefix}_cos": float(diag.mean()),
        f"{prefix}_cos_min": float(diag.min()),
        f"{prefix}_cos_max": float(diag.max()),
        f"{prefix}_i2t_acc": float(np.mean(logits.argmax(axis=1) == targets)),
        f"{prefix}_t2i_acc": float(np.mean(logits.argmax(axis=0) == targets)),
        f"{prefix}_hard_negative_margin": float(margin.mean()),
        f"{prefix}_caption_token_mean": float(np.mean([
            len(caption_tokens(rec.caption)) for rec in paired_records
        ])),
    }


def image_eval_composite_score(report, embedding_kind="image"):
    prefix = f"{embedding_kind}_distribution"
    frechet = report.get(f"{prefix}_frechet")
    mmd = report.get(f"{prefix}_mmd_rbf")
    nearest_real = report.get(f"{prefix}_nearest_real_l2")
    precision = report.get(f"{prefix}_support_precision")
    recall = report.get(f"{prefix}_support_recall")
    diversity_ratio = report.get(f"{prefix}_diversity_l2_ratio")
    align_n = int(report.get("generated_image_text_alignment_n", 0) or 0)

    distribution_score, distribution_parts = _weighted_geomean([
        ("frechet", _lower_is_better_score(frechet), 0.45),
        ("mmd_rbf", _lower_is_better_score(mmd), 0.35),
        ("nearest_real_l2", _lower_is_better_score(nearest_real), 0.20),
    ])
    support_score, support_parts = _weighted_geomean([
        ("support_precision", _clamp01(precision), 0.50),
        ("support_recall", _clamp01(recall), 0.50),
    ])
    diversity_score = _ratio_target_one_score(diversity_ratio)
    components = [
        ("distribution", distribution_score, 0.35),
        ("support", support_score, 0.25),
        ("diversity", diversity_score, 0.20),
    ]
    alignment_parts = {}
    alignment_score = None
    if align_n > 0:
        alignment_score, alignment_parts = _weighted_geomean([
            ("image_text_cos", _cosine_score(
                report.get("generated_image_text_alignment_cos")), 0.50),
            ("image_to_text_retrieval", _clamp01(
                report.get("generated_image_text_alignment_i2t_acc")), 0.25),
            ("text_to_image_retrieval", _clamp01(
                report.get("generated_image_text_alignment_t2i_acc")), 0.25),
        ])
        components.append(("alignment", alignment_score, 0.20))
    visual_parts = {}
    visual_score = None
    if report.get("visual_physics_distribution_available"):
        visual_score, visual_parts = _weighted_geomean([
            ("visual_frechet", _lower_is_better_score(
                report.get("visual_physics_distribution_frechet")), 0.25),
            ("visual_mmd_rbf", _lower_is_better_score(
                report.get("visual_physics_distribution_mmd_rbf")), 0.20),
            ("visual_mean_feature_l1", _lower_is_better_score(
                report.get("visual_physics_distribution_mean_feature_l1")), 0.20),
            ("visual_detail_energy_ratio", _ratio_target_one_score(
                report.get("visual_physics_distribution_detail_energy_ratio")), 0.15),
            ("visual_dynamic_range_ratio", _ratio_target_one_score(
                report.get("visual_physics_distribution_dynamic_range_ratio")), 0.10),
            ("visual_luminance_std_ratio", _ratio_target_one_score(
                report.get("visual_physics_distribution_luminance_std_ratio")), 0.10),
        ])
        components.append(("visual_physics", visual_score, 0.15))
    score, component_weights = _weighted_geomean(components)
    out = {
        "image_eval_score": float(score),
        "image_eval_score_components": component_weights,
        "image_eval_distribution_score": float(distribution_score),
        "image_eval_distribution_parts": distribution_parts,
        "image_eval_support_score": float(support_score),
        "image_eval_support_parts": support_parts,
        "image_eval_diversity_score": float(diversity_score),
        "image_eval_alignment_available": bool(align_n > 0),
        "image_eval_visual_physics_available": bool(visual_score is not None),
    }
    if alignment_score is not None:
        out["image_eval_alignment_score"] = float(alignment_score)
        out["image_eval_alignment_parts"] = alignment_parts
    if visual_score is not None:
        out["image_eval_visual_physics_score"] = float(visual_score)
        out["image_eval_visual_physics_parts"] = visual_parts
    return out


def image_eval_gate(report, min_score=0.0, min_support_precision=0.0,
                    min_support_recall=0.0, min_image_text_cos=None,
                    max_frechet=None, max_mmd_rbf=None,
                    min_generated_neighbor_l2_p05=None,
                    min_generated_real_l2_p01=None,
                    max_visual_physics_l1=None,
                    max_visual_physics_mmd_rbf=None,
                    min_visual_detail_ratio=None,
                    min_visual_dynamic_range_ratio=None,
                    min_visual_luminance_std_ratio=None,
                    embedding_kind="image"):
    prefix = f"{embedding_kind}_distribution"
    checks = []

    def add_check(name, value, threshold, mode):
        if threshold is None:
            return
        value = float(value or 0.0)
        threshold = float(threshold)
        if mode == "min":
            passed = value >= threshold
        elif mode == "max":
            passed = value <= threshold
        else:
            raise ValueError(f"unknown gate mode {mode!r}")
        checks.append({
            "name": name,
            "value": value,
            "threshold": threshold,
            "mode": mode,
            "pass": bool(passed),
        })

    add_check("image_eval_score", report.get("image_eval_score"), min_score, "min")
    add_check(
        f"{prefix}_support_precision",
        report.get(f"{prefix}_support_precision"),
        min_support_precision,
        "min",
    )
    add_check(
        f"{prefix}_support_recall",
        report.get(f"{prefix}_support_recall"),
        min_support_recall,
        "min",
    )
    add_check(
        "generated_image_text_alignment_cos",
        report.get("generated_image_text_alignment_cos"),
        min_image_text_cos,
        "min",
    )
    add_check(f"{prefix}_frechet", report.get(f"{prefix}_frechet"), max_frechet, "max")
    add_check(f"{prefix}_mmd_rbf", report.get(f"{prefix}_mmd_rbf"), max_mmd_rbf, "max")
    add_check(
        f"{prefix}_generated_nearest_generated_l2_p05",
        report.get(f"{prefix}_generated_nearest_generated_l2_p05"),
        min_generated_neighbor_l2_p05,
        "min",
    )
    add_check(
        f"{prefix}_generated_nearest_real_l2_p01",
        report.get(f"{prefix}_generated_nearest_real_l2_p01"),
        min_generated_real_l2_p01,
        "min",
    )
    visual_gate_requested = any(
        value is not None for value in (
            max_visual_physics_l1,
            max_visual_physics_mmd_rbf,
            min_visual_detail_ratio,
            min_visual_dynamic_range_ratio,
            min_visual_luminance_std_ratio,
        )
    )
    if visual_gate_requested and not report.get("visual_physics_distribution_available"):
        checks.append({
            "name": "visual_physics_distribution_available",
            "value": 0.0,
            "threshold": 1.0,
            "mode": "min",
            "pass": False,
        })
    add_check(
        "visual_physics_distribution_mean_feature_l1",
        report.get("visual_physics_distribution_mean_feature_l1"),
        max_visual_physics_l1,
        "max",
    )
    add_check(
        "visual_physics_distribution_mmd_rbf",
        report.get("visual_physics_distribution_mmd_rbf"),
        max_visual_physics_mmd_rbf,
        "max",
    )
    add_check(
        "visual_physics_distribution_detail_energy_ratio",
        report.get("visual_physics_distribution_detail_energy_ratio"),
        min_visual_detail_ratio,
        "min",
    )
    add_check(
        "visual_physics_distribution_dynamic_range_ratio",
        report.get("visual_physics_distribution_dynamic_range_ratio"),
        min_visual_dynamic_range_ratio,
        "min",
    )
    add_check(
        "visual_physics_distribution_luminance_std_ratio",
        report.get("visual_physics_distribution_luminance_std_ratio"),
        min_visual_luminance_std_ratio,
        "min",
    )
    passed = all(row["pass"] for row in checks)
    return {
        "image_eval_gate_pass": bool(passed),
        "image_eval_gate_checks": checks,
        "image_eval_gate_failed": [row for row in checks if not row["pass"]],
    }


def evaluate_records(real_records: Sequence[ImageTextRecord],
                     generated_records: Sequence[ImageTextRecord],
                     embedding_kind="image", dim_policy="error", normalize=True,
                     visual_stats_size=0, visual_stats_max_records=0):
    real_matrix = embedding_matrix(
        real_records, kind=embedding_kind, dim_policy=dim_policy,
        normalize=normalize, prefix="real")
    gen_matrix = embedding_matrix(
        generated_records, kind=embedding_kind, dim_policy=dim_policy,
        normalize=normalize, prefix="generated")
    report = {
        "experiment": "image_embedding_eval",
        "embedding_kind": embedding_kind,
        "embedding_normalize": bool(normalize),
        "dim_policy": dim_policy,
        "real": summarize_records(real_records),
        "generated": summarize_records(generated_records),
    }
    report.update(real_matrix.report)
    report.update(gen_matrix.report)
    report.update(embedding_distribution_metrics(
        gen_matrix.matrix, real_matrix.matrix,
        prefix=f"{embedding_kind}_distribution"))
    report.update(embedding_neighborhood_metrics(
        gen_matrix.matrix, real_matrix.matrix,
        prefix=f"{embedding_kind}_distribution"))
    report.update(image_text_alignment_metrics(
        real_records, prefix="real_image_text_alignment",
        dim_policy=dim_policy, normalize=normalize))
    report.update(image_text_alignment_metrics(
        generated_records, prefix="generated_image_text_alignment",
        dim_policy=dim_policy, normalize=normalize))
    if int(visual_stats_size or 0) > 0:
        real_visual = visual_physics_matrix(
            real_records, size=visual_stats_size,
            max_records=visual_stats_max_records,
            prefix="real_visual_physics")
        generated_visual = visual_physics_matrix(
            generated_records, size=visual_stats_size,
            max_records=visual_stats_max_records,
            prefix="generated_visual_physics")
        report.update(real_visual.report)
        report.update(generated_visual.report)
        report.update(visual_physics_distribution_metrics(
            generated_visual.matrix, real_visual.matrix,
            prefix="visual_physics_distribution"))
    else:
        report.update({
            "real_visual_physics_enabled": False,
            "generated_visual_physics_enabled": False,
            "visual_physics_distribution_available": False,
        })
    report.update(image_eval_composite_score(report, embedding_kind=embedding_kind))
    return report


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_ppm(path, pixels):
    pixels = np.asarray(pixels, dtype=np.uint8)
    if pixels.ndim == 1:
        pixels = np.tile(pixels.reshape(1, 1, 3), (16, 16, 1))
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("PPM test fixture must be HWC RGB")
    with open(path, "wb") as f:
        f.write(f"P6\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode("ascii"))
        f.write(pixels.tobytes())


def _visual_fixture(seed=0, collapsed=False):
    rng = np.random.default_rng(int(seed))
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 32, dtype=np.float64),
        np.linspace(0.0, 1.0, 32, dtype=np.float64),
        indexing="ij",
    )
    if collapsed:
        base = np.full((32, 32, 3), 0.52, dtype=np.float64)
        base += rng.normal(0.0, 0.003, size=base.shape)
        return np.clip(base * 255.0, 0.0, 255.0).astype(np.uint8)
    waves = 0.5 + 0.25 * np.sin((xx * (3.0 + seed) + yy * 1.5) * np.pi)
    spot = np.exp(-(((xx - 0.35 - 0.1 * seed) ** 2) + ((yy - 0.58) ** 2)) / 0.025)
    stripe = ((np.floor((xx + yy * 0.35) * (5 + seed)) % 2) * 0.18)
    img = np.stack([
        waves + 0.35 * spot,
        0.35 + stripe + 0.18 * yy,
        0.30 + 0.40 * (1.0 - xx) + 0.20 * spot,
    ], axis=-1)
    img += rng.normal(0.0, 0.015, size=img.shape)
    return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)


def selftest():
    with tempfile.TemporaryDirectory() as td:
        real = os.path.join(td, "real.jsonl")
        good = os.path.join(td, "good.jsonl")
        bad = os.path.join(td, "bad.jsonl")
        memorized = os.path.join(td, "memorized.jsonl")
        _write_ppm(os.path.join(td, "r0.ppm"), _visual_fixture(seed=0))
        _write_ppm(os.path.join(td, "r1.ppm"), _visual_fixture(seed=1))
        _write_ppm(os.path.join(td, "g0.ppm"), _visual_fixture(seed=0))
        _write_ppm(os.path.join(td, "g1.ppm"), _visual_fixture(seed=1))
        _write_ppm(os.path.join(td, "b0.ppm"), _visual_fixture(seed=2, collapsed=True))
        _write_ppm(os.path.join(td, "b1.ppm"), _visual_fixture(seed=3, collapsed=True))
        _write_ppm(os.path.join(td, "m0.ppm"), _visual_fixture(seed=0))
        _write_ppm(os.path.join(td, "m1.ppm"), _visual_fixture(seed=1))
        _write_jsonl(real, [
            {"image": "r0.ppm", "caption": "red square", "split": "eval",
             "image_embedding": [1.0, 0.0], "text_embedding": [1.0, 0.0]},
            {"image": "r1.ppm", "caption": "blue circle", "split": "eval",
             "image_embedding": [0.0, 1.0], "text_embedding": [0.0, 1.0]},
        ])
        _write_jsonl(good, [
            {"image": "g0.ppm", "caption": "red square", "split": "eval",
             "image_embedding": [0.98, 0.02], "text_embedding": [1.0, 0.0]},
            {"image": "g1.ppm", "caption": "blue circle", "split": "eval",
             "image_embedding": [0.02, 0.98], "text_embedding": [0.0, 1.0]},
        ])
        _write_jsonl(bad, [
            {"image": "b0.ppm", "caption": "red square", "split": "eval",
             "image_embedding": [-1.0, 0.0], "text_embedding": [1.0, 0.0]},
            {"image": "b1.ppm", "caption": "blue circle", "split": "eval",
             "image_embedding": [-1.0, 0.0], "text_embedding": [0.0, 1.0]},
        ])
        _write_jsonl(memorized, [
            {"image": "m0.ppm", "caption": "red square", "split": "eval",
             "image_embedding": [1.0, 0.0], "text_embedding": [1.0, 0.0]},
            {"image": "m1.ppm", "caption": "blue circle", "split": "eval",
             "image_embedding": [0.0, 1.0], "text_embedding": [0.0, 1.0]},
        ])
        real_records, _ = load_records(real, split="eval")
        good_records, _ = load_records(good, split="eval")
        bad_records, _ = load_records(bad, split="eval")
        memorized_records, _ = load_records(memorized, split="eval")
        good_report = evaluate_records(real_records, good_records, visual_stats_size=16)
        bad_report = evaluate_records(real_records, bad_records, visual_stats_size=16)
        memorized_report = evaluate_records(
            real_records, memorized_records, visual_stats_size=16)
        assert good_report["image_distribution_generated_n"] == 2
        assert good_report["generated_image_text_alignment_n"] == 2
        assert good_report["generated_image_text_alignment_i2t_acc"] == 1.0
        assert good_report["visual_physics_distribution_available"] is True
        assert good_report["generated_visual_physics_usable"] == 2
        assert bad_report["visual_physics_distribution_detail_energy_ratio"] < (
            good_report["visual_physics_distribution_detail_energy_ratio"])
        assert bad_report["visual_physics_distribution_dynamic_range_ratio"] < (
            good_report["visual_physics_distribution_dynamic_range_ratio"])
        assert good_report["image_distribution_frechet"] < bad_report[
            "image_distribution_frechet"]
        assert good_report["image_distribution_mmd_rbf"] < bad_report[
            "image_distribution_mmd_rbf"]
        assert good_report["image_distribution_generated_nearest_generated_l2_p05"] > (
            bad_report["image_distribution_generated_nearest_generated_l2_p05"])
        assert memorized_report[
            "image_distribution_generated_nearest_real_l2_p01"] == 0.0
        assert good_report["image_eval_score"] > bad_report["image_eval_score"]
        good_gate = image_eval_gate(good_report, min_score=0.01)
        bad_gate = image_eval_gate(bad_report, min_score=good_report["image_eval_score"] + 0.01)
        duplicate_gate = image_eval_gate(
            bad_report, min_generated_neighbor_l2_p05=0.01)
        memorized_gate = image_eval_gate(
            memorized_report, min_generated_real_l2_p01=1.0e-6)
        visual_gate = image_eval_gate(
            bad_report, min_visual_detail_ratio=0.10,
            min_visual_dynamic_range_ratio=0.10)
        assert good_gate["image_eval_gate_pass"] is True
        assert bad_gate["image_eval_gate_pass"] is False
        assert duplicate_gate["image_eval_gate_pass"] is False
        assert memorized_gate["image_eval_gate_pass"] is False
        assert visual_gate["image_eval_gate_pass"] is False
        assert bad_gate["image_eval_gate_failed"][0]["name"] == "image_eval_score"
    print("image_eval selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real-manifest", default="", help="real/reference image manifest")
    ap.add_argument("--generated-manifest", default="", help="generated/candidate image manifest")
    ap.add_argument("--real-root", default="", help="root for relative real image paths")
    ap.add_argument("--generated-root", default="", help="root for relative generated image paths")
    ap.add_argument("--real-split", default="", help="optional split filter for real records")
    ap.add_argument("--generated-split", default="", help="optional split filter for generated records")
    ap.add_argument("--min-aesthetic", type=float, default=None,
                    help="optional minimum aesthetic/quality score for both manifests")
    ap.add_argument("--max-nsfw", type=float, default=None,
                    help="optional maximum nsfw/image_nsfw/prompt_nsfw for both manifests")
    ap.add_argument("--max-watermark", type=float, default=None,
                    help="optional maximum watermark/watermark_score for both manifests")
    ap.add_argument("--max-records", type=int, default=2048,
                    help="maximum records per manifest before O(n^2) metrics; 0 means all")
    ap.add_argument("--real-max-records", type=int, default=0,
                    help="override --max-records for real records")
    ap.add_argument("--generated-max-records", type=int, default=0,
                    help="override --max-records for generated records")
    ap.add_argument("--embedding-kind", choices=EMBEDDING_KINDS, default="image",
                    help="embedding field used for distribution metrics")
    ap.add_argument("--dim-policy", choices=DIM_POLICIES, default="error",
                    help="how to handle mixed embedding dimensions")
    ap.add_argument("--no-normalize", action="store_true",
                    help="do not L2-normalize embeddings before scoring")
    ap.add_argument("--real-embedding-sidecar", default="",
                    help="optional embedding JSONL sidecar merged into the real manifest")
    ap.add_argument("--generated-embedding-sidecar", default="",
                    help="optional embedding JSONL sidecar merged into the generated manifest")
    ap.add_argument("--embedding-key", default="image", choices=("image", "caption"),
                    help="key used when merging sidecar embeddings")
    ap.add_argument("--embedding-overwrite", action="store_true",
                    help="overwrite manifest embeddings with sidecar rows")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="minimum composite image_eval_score for the quality gate")
    ap.add_argument("--min-support-precision", type=float, default=0.0,
                    help="minimum generated support precision for the quality gate")
    ap.add_argument("--min-support-recall", type=float, default=0.0,
                    help="minimum generated support recall for the quality gate")
    ap.add_argument("--min-image-text-cos", type=float, default=None,
                    help="minimum generated image/text cosine for the quality gate")
    ap.add_argument("--max-frechet", type=float, default=None,
                    help="maximum embedding Fréchet distance for the quality gate")
    ap.add_argument("--max-mmd-rbf", type=float, default=None,
                    help="maximum RBF-MMD distance for the quality gate")
    ap.add_argument("--min-generated-neighbor-l2-p05", type=float, default=None,
                    help=("minimum generated/generated nearest-neighbor L2 p05; "
                          "catches collapsed duplicate outputs"))
    ap.add_argument("--min-generated-real-l2-p01", type=float, default=None,
                    help=("minimum generated/real nearest-neighbor L2 p01; "
                          "catches over-near training-set copies"))
    ap.add_argument("--visual-stats-size", type=int, default=0,
                    help=("compute generic visual-physics stats at this image size; "
                          "0 disables pixel-structure evaluation"))
    ap.add_argument("--visual-stats-max-records", type=int, default=0,
                    help=("maximum records per manifest for visual-physics stats; "
                          "0 means use all loaded records"))
    ap.add_argument("--max-visual-physics-l1", type=float, default=None,
                    help="maximum mean visual-physics feature L1 drift")
    ap.add_argument("--max-visual-physics-mmd-rbf", type=float, default=None,
                    help="maximum visual-physics RBF-MMD distance")
    ap.add_argument("--min-visual-detail-ratio", type=float, default=None,
                    help="minimum generated/reference detail-energy ratio")
    ap.add_argument("--min-visual-dynamic-range-ratio", type=float, default=None,
                    help="minimum generated/reference luminance dynamic-range ratio")
    ap.add_argument("--min-visual-luminance-std-ratio", type=float, default=None,
                    help="minimum generated/reference luminance standard-deviation ratio")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="exit non-zero when configured quality-gate checks fail")
    ap.add_argument("--report-out", default="", help="optional JSON report path")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return {"selftest": True}
    if not args.real_manifest or not args.generated_manifest:
        ap.error("--real-manifest and --generated-manifest are required")
    if (args.min_generated_neighbor_l2_p05 is not None
            and args.min_generated_neighbor_l2_p05 < 0.0):
        ap.error("--min-generated-neighbor-l2-p05 must be non-negative")
    if (args.min_generated_real_l2_p01 is not None
            and args.min_generated_real_l2_p01 < 0.0):
        ap.error("--min-generated-real-l2-p01 must be non-negative")
    if args.visual_stats_size < 0:
        ap.error("--visual-stats-size must be non-negative")
    if args.visual_stats_max_records < 0:
        ap.error("--visual-stats-max-records must be non-negative")
    for name in (
            "max_visual_physics_l1", "max_visual_physics_mmd_rbf",
            "min_visual_detail_ratio", "min_visual_dynamic_range_ratio",
            "min_visual_luminance_std_ratio"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            ap.error(f"--{name.replace('_', '-')} must be non-negative")

    shared_max = int(args.max_records)
    real_max = int(args.real_max_records or shared_max)
    generated_max = int(args.generated_max_records or shared_max)
    real_records, real_merge = load_records(
        args.real_manifest, root=args.real_root, split=args.real_split,
        min_aesthetic=args.min_aesthetic, max_records=real_max,
        max_nsfw=args.max_nsfw, max_watermark=args.max_watermark,
        embedding_sidecar=args.real_embedding_sidecar,
        embedding_key=args.embedding_key, embedding_overwrite=args.embedding_overwrite)
    generated_records, generated_merge = load_records(
        args.generated_manifest, root=args.generated_root, split=args.generated_split,
        min_aesthetic=args.min_aesthetic, max_records=generated_max,
        max_nsfw=args.max_nsfw, max_watermark=args.max_watermark,
        embedding_sidecar=args.generated_embedding_sidecar,
        embedding_key=args.embedding_key, embedding_overwrite=args.embedding_overwrite)
    report = evaluate_records(
        real_records, generated_records, embedding_kind=args.embedding_kind,
        dim_policy=args.dim_policy, normalize=not args.no_normalize,
        visual_stats_size=args.visual_stats_size,
        visual_stats_max_records=args.visual_stats_max_records)
    report.update(image_eval_gate(
        report,
        min_score=args.min_score,
        min_support_precision=args.min_support_precision,
        min_support_recall=args.min_support_recall,
        min_image_text_cos=args.min_image_text_cos,
        max_frechet=args.max_frechet,
        max_mmd_rbf=args.max_mmd_rbf,
        min_generated_neighbor_l2_p05=args.min_generated_neighbor_l2_p05,
        min_generated_real_l2_p01=args.min_generated_real_l2_p01,
        max_visual_physics_l1=args.max_visual_physics_l1,
        max_visual_physics_mmd_rbf=args.max_visual_physics_mmd_rbf,
        min_visual_detail_ratio=args.min_visual_detail_ratio,
        min_visual_dynamic_range_ratio=args.min_visual_dynamic_range_ratio,
        min_visual_luminance_std_ratio=args.min_visual_luminance_std_ratio,
        embedding_kind=args.embedding_kind,
    ))
    report.update({
        "real_manifest": args.real_manifest,
        "generated_manifest": args.generated_manifest,
        "real_root": args.real_root,
        "generated_root": args.generated_root,
        "real_split": args.real_split,
        "generated_split": args.generated_split,
        "max_records": shared_max,
        "real_max_records": real_max,
        "generated_max_records": generated_max,
        "visual_stats_size": int(args.visual_stats_size),
        "visual_stats_max_records": int(args.visual_stats_max_records),
        "min_aesthetic": (
            float(args.min_aesthetic) if args.min_aesthetic is not None else None),
        "max_nsfw": float(args.max_nsfw) if args.max_nsfw is not None else None,
        "max_watermark": (
            float(args.max_watermark) if args.max_watermark is not None else None),
        "min_generated_neighbor_l2_p05": (
            float(args.min_generated_neighbor_l2_p05)
            if args.min_generated_neighbor_l2_p05 is not None else None),
        "min_generated_real_l2_p01": (
            float(args.min_generated_real_l2_p01)
            if args.min_generated_real_l2_p01 is not None else None),
        "max_visual_physics_l1": (
            float(args.max_visual_physics_l1)
            if args.max_visual_physics_l1 is not None else None),
        "max_visual_physics_mmd_rbf": (
            float(args.max_visual_physics_mmd_rbf)
            if args.max_visual_physics_mmd_rbf is not None else None),
        "min_visual_detail_ratio": (
            float(args.min_visual_detail_ratio)
            if args.min_visual_detail_ratio is not None else None),
        "min_visual_dynamic_range_ratio": (
            float(args.min_visual_dynamic_range_ratio)
            if args.min_visual_dynamic_range_ratio is not None else None),
        "min_visual_luminance_std_ratio": (
            float(args.min_visual_luminance_std_ratio)
            if args.min_visual_luminance_std_ratio is not None else None),
    })
    if real_merge:
        report["real_embedding_merge"] = real_merge
    if generated_merge:
        report["generated_embedding_merge"] = generated_merge

    text = json.dumps(report, indent=1)
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"saved -> {args.report_out}")
    print(text)
    if args.fail_on_gate and not report.get("image_eval_gate_pass", False):
        sys.exit(2)
    return report


if __name__ == "__main__":
    main()
