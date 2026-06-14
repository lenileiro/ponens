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
    merge_embedding_sidecar,
    read_image_manifest,
    summarize_records,
)


EMBEDDING_KINDS = ("image", "text", "image_sequence", "text_sequence")
DIM_POLICIES = ("error", "largest")
EPS = 1.0e-12


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
    }
    if alignment_score is not None:
        out["image_eval_alignment_score"] = float(alignment_score)
        out["image_eval_alignment_parts"] = alignment_parts
    return out


def image_eval_gate(report, min_score=0.0, min_support_precision=0.0,
                    min_support_recall=0.0, min_image_text_cos=None,
                    max_frechet=None, max_mmd_rbf=None, embedding_kind="image"):
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
    passed = all(row["pass"] for row in checks)
    return {
        "image_eval_gate_pass": bool(passed),
        "image_eval_gate_checks": checks,
        "image_eval_gate_failed": [row for row in checks if not row["pass"]],
    }


def evaluate_records(real_records: Sequence[ImageTextRecord],
                     generated_records: Sequence[ImageTextRecord],
                     embedding_kind="image", dim_policy="error", normalize=True):
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
    report.update(image_text_alignment_metrics(
        real_records, prefix="real_image_text_alignment",
        dim_policy=dim_policy, normalize=normalize))
    report.update(image_text_alignment_metrics(
        generated_records, prefix="generated_image_text_alignment",
        dim_policy=dim_policy, normalize=normalize))
    report.update(image_eval_composite_score(report, embedding_kind=embedding_kind))
    return report


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def selftest():
    with tempfile.TemporaryDirectory() as td:
        real = os.path.join(td, "real.jsonl")
        good = os.path.join(td, "good.jsonl")
        bad = os.path.join(td, "bad.jsonl")
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
        real_records, _ = load_records(real, split="eval")
        good_records, _ = load_records(good, split="eval")
        bad_records, _ = load_records(bad, split="eval")
        good_report = evaluate_records(real_records, good_records)
        bad_report = evaluate_records(real_records, bad_records)
        assert good_report["image_distribution_generated_n"] == 2
        assert good_report["generated_image_text_alignment_n"] == 2
        assert good_report["generated_image_text_alignment_i2t_acc"] == 1.0
        assert good_report["image_distribution_frechet"] < bad_report[
            "image_distribution_frechet"]
        assert good_report["image_distribution_mmd_rbf"] < bad_report[
            "image_distribution_mmd_rbf"]
        assert good_report["image_eval_score"] > bad_report["image_eval_score"]
        good_gate = image_eval_gate(good_report, min_score=0.01)
        bad_gate = image_eval_gate(bad_report, min_score=good_report["image_eval_score"] + 0.01)
        assert good_gate["image_eval_gate_pass"] is True
        assert bad_gate["image_eval_gate_pass"] is False
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
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="exit non-zero when configured quality-gate checks fail")
    ap.add_argument("--report-out", default="", help="optional JSON report path")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return {"selftest": True}
    if not args.real_manifest or not args.generated_manifest:
        ap.error("--real-manifest and --generated-manifest are required")

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
        dim_policy=args.dim_policy, normalize=not args.no_normalize)
    report.update(image_eval_gate(
        report,
        min_score=args.min_score,
        min_support_precision=args.min_support_precision,
        min_support_recall=args.min_support_recall,
        min_image_text_cos=args.min_image_text_cos,
        max_frechet=args.max_frechet,
        max_mmd_rbf=args.max_mmd_rbf,
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
        "min_aesthetic": (
            float(args.min_aesthetic) if args.min_aesthetic is not None else None),
        "max_nsfw": float(args.max_nsfw) if args.max_nsfw is not None else None,
        "max_watermark": (
            float(args.max_watermark) if args.max_watermark is not None else None),
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
