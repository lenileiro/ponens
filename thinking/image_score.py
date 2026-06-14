"""Generic image-quality and preference scoring for captioned manifests.

This module deliberately keeps reward/preference signals out of the generator
architecture.  It reads the same image manifest format as `image_data`, writes a
scored manifest that existing training code can consume through the generic
`quality_score` / `aesthetic` fields, and optionally merges arbitrary external
score sidecars from preference models.

The dependency-free `stats` scorer is a local smoke-test proxy, not a human
preference model.  Production runs should merge an external sidecar from a
reward/preference model, or ensemble that sidecar with embedding alignment and
technical image-health signals.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from .image_data import (
    ImageTextRecord,
    _coerce_record,
    _manifest_image_path,
    caption_tokens,
    image_text_embedding_cosine,
    load_image_tensor,
)


BACKENDS = ("stats", "embedding", "external", "ensemble")
KEYS = ("image", "basename", "caption")
NORMALIZE_MODES = ("auto", "minmax", "none")
EPS = 1.0e-12
DEFAULT_EXTERNAL_FIELDS = (
    "quality_score",
    "preference_score",
    "reward_score",
    "image_reward",
    "pickscore",
    "hps",
    "hpsv2",
    "aesthetic",
    "aesthetic_score",
    "score",
    "quality",
)


@dataclass(frozen=True)
class ManifestRow:
    row: dict
    record: ImageTextRecord
    index: int


def _read_rows_stream(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    if ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                has_header = csv.Sniffer().has_header(sample) if sample.strip() else True
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
    raise ValueError(f"unsupported manifest extension {ext!r}; use .jsonl, .csv, or .tsv")


def iter_manifest_rows(path: str, root: str = "", split: str = "", max_records: int = 0):
    manifest_dir = os.path.dirname(os.path.abspath(path))
    kept = 0
    for idx, row in enumerate(_read_rows_stream(path)):
        rec = _coerce_record(row, manifest_dir=manifest_dir, root=root)
        if split and rec.split != split:
            continue
        yield ManifestRow(row=dict(row), record=rec, index=idx)
        kept += 1
        if max_records and kept >= int(max_records):
            break


def _caption_key(caption: str) -> str:
    return " ".join(str(caption).strip().lower().split())


def _path_key(path: str, base: str = "") -> str:
    p = str(path)
    if base and not os.path.isabs(p):
        p = os.path.join(base, p)
    return os.path.normcase(os.path.abspath(os.path.normpath(p)))


def _row_key(row: dict, directory: str, root: str = "", key: str = "image"):
    if key == "caption":
        caption = row.get("caption") or row.get("text") or row.get("prompt")
        return _caption_key(caption) if caption is not None else None
    image = row.get("image") or row.get("path") or row.get("file") or row.get("filepath")
    if image is None:
        return None
    if key == "basename":
        return os.path.basename(str(image))
    if key != "image":
        raise ValueError(f"unknown key {key!r}")
    return _path_key(str(image), base=root or directory)


def _record_key(rec: ImageTextRecord, key: str = "image"):
    if key == "caption":
        return _caption_key(rec.caption)
    if key == "basename":
        return os.path.basename(rec.path)
    if key != "image":
        raise ValueError(f"unknown key {key!r}")
    return _path_key(rec.path)


def _optional_float(raw):
    if raw in ("", None):
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _stats(vals: Iterable[float]):
    rows = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not rows:
        return {}
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _clamp01(x: float) -> float:
    if not math.isfinite(float(x)):
        return 0.0
    return float(min(1.0, max(0.0, float(x))))


def _positive_saturating(x: float, scale: float) -> float:
    if not math.isfinite(float(x)) or x <= 0.0:
        return 0.0
    return float(1.0 - math.exp(-float(x) / max(float(scale), EPS)))


def _weighted_geomean(parts: Sequence[tuple[str, float | None, float]]):
    usable = []
    for name, value, weight in parts:
        if weight <= 0.0 or value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        usable.append((name, _clamp01(value), float(weight)))
    if not usable:
        return None, {}
    total = sum(w for _name, _value, w in usable)
    score = math.exp(sum(w * math.log(max(v, EPS)) for _name, v, w in usable) / total)
    return float(score), {name: {"score": float(value), "weight": float(weight)}
                          for name, value, weight in usable}


def technical_quality(path: str, image_size: int = 256) -> tuple[float, dict]:
    """Return a generic image-health score in [0, 1] plus component metrics."""
    x = load_image_tensor(path, size=max(1, int(image_size)), device="cpu", center_crop=False)
    rgb = ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    r, g, b = rgb[0], rgb[1], rgb[2]
    lum = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)
    mean = float(lum.mean())
    std = float(lum.std(unbiased=False))
    dyn = float(lum.max() - lum.min())
    if lum.shape[0] > 1:
        gy = (lum[1:, :] - lum[:-1, :]).abs()
    else:
        gy = torch.zeros_like(lum)
    if lum.shape[1] > 1:
        gx = (lum[:, 1:] - lum[:, :-1]).abs()
    else:
        gx = torch.zeros_like(lum)
    grad_mean = float(0.5 * (gx.mean() + gy.mean()))
    clipped = float(((rgb <= 0.02) | (rgb >= 0.98)).float().mean())
    bins = torch.histc(lum.reshape(-1), bins=32, min=0.0, max=1.0).float()
    probs = bins / bins.sum().clamp_min(1.0)
    entropy = float(-(probs[probs > 0.0] * probs[probs > 0.0].log()).sum() / math.log(32.0))
    saturation = float((rgb.max(dim=0).values - rgb.min(dim=0).values).mean())

    exposure_score = math.exp(-((mean - 0.5) / 0.32) ** 2)
    contrast_score = _positive_saturating(std, 0.18)
    dynamic_range_score = _clamp01(dyn / 0.75)
    sharpness_score = _positive_saturating(grad_mean, 0.035)
    clipping_score = math.exp(-max(0.0, clipped - 0.02) / 0.18)
    entropy_score = _clamp01(entropy)
    saturation_score = _positive_saturating(saturation, 0.20)
    score, parts = _weighted_geomean([
        ("exposure", exposure_score, 0.15),
        ("contrast", contrast_score, 0.20),
        ("dynamic_range", dynamic_range_score, 0.15),
        ("sharpness", sharpness_score, 0.20),
        ("clipping", clipping_score, 0.10),
        ("entropy", entropy_score, 0.15),
        ("saturation", saturation_score, 0.05),
    ])
    components = {
        "technical_quality_score": float(score or 0.0),
        "technical_quality_parts": parts,
        "luminance_mean": mean,
        "luminance_std": std,
        "luminance_dynamic_range": dyn,
        "gradient_mean": grad_mean,
        "clipped_pixel_fraction": clipped,
        "luminance_entropy": entropy,
        "mean_saturation": saturation,
    }
    return float(score or 0.0), components


def embedding_quality(rec: ImageTextRecord) -> tuple[float | None, dict]:
    cos, reason = image_text_embedding_cosine(rec)
    if cos is None:
        return None, {"embedding_quality_missing": reason}
    score = _clamp01((float(cos) + 1.0) * 0.5)
    return score, {
        "embedding_alignment_score": float(score),
        "image_text_cosine": float(cos),
    }


def _select_external_score(row: dict, score_field: str = "") -> tuple[float | None, str]:
    fields = [score_field] if score_field else list(DEFAULT_EXTERNAL_FIELDS)
    for field in fields:
        if not field:
            continue
        val = _optional_float(row.get(field))
        if val is not None:
            return val, field
    return None, ""


def load_external_scores(path: str, root: str = "", key: str = "image", score_field: str = ""):
    if not path:
        return {}, {
            "external_score_sidecar": "",
            "external_score_indexed": 0,
        }
    directory = os.path.dirname(os.path.abspath(path))
    index = {}
    fields = Counter()
    duplicate = skipped_key = skipped_score = 0
    raw_vals = []
    for row in _read_rows_stream(path):
        row_key = _row_key(row, directory=directory, root=root, key=key)
        if row_key is None:
            skipped_key += 1
            continue
        score, field = _select_external_score(row, score_field=score_field)
        if score is None:
            skipped_score += 1
            continue
        if row_key in index:
            duplicate += 1
            continue
        fields[field] += 1
        raw_vals.append(score)
        index[row_key] = {
            "raw": float(score),
            "field": field,
            "row": row,
        }
    report = {
        "external_score_sidecar": path,
        "external_score_key": key,
        "external_score_requested_field": score_field,
        "external_score_indexed": int(len(index)),
        "external_score_duplicate_keys": int(duplicate),
        "external_score_skipped_no_key": int(skipped_key),
        "external_score_skipped_no_score": int(skipped_score),
        "external_score_fields": dict(sorted(fields.items())),
        "external_score_raw_stats": _stats(raw_vals),
    }
    return index, report


def normalize_external_scores(index: dict, mode: str = "auto") -> dict:
    if not index:
        return {
            "external_score_normalize": mode,
            "external_score_normalized": False,
        }
    vals = np.asarray([row["raw"] for row in index.values()], dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        for row in index.values():
            row["score"] = None
        return {
            "external_score_normalize": mode,
            "external_score_normalized": False,
        }
    lo = float(np.min(vals[finite]))
    hi = float(np.max(vals[finite]))
    use_minmax = mode == "minmax" or (mode == "auto" and (lo < 0.0 or hi > 1.0))
    for row in index.values():
        raw = float(row["raw"])
        if use_minmax:
            row["score"] = 0.5 if hi <= lo else _clamp01((raw - lo) / max(hi - lo, EPS))
        else:
            row["score"] = _clamp01(raw)
    return {
        "external_score_normalize": mode,
        "external_score_normalized": bool(use_minmax),
        "external_score_norm_min": lo,
        "external_score_norm_max": hi,
    }


def score_record(row: ManifestRow, backend: str = "stats", external_index: dict | None = None,
                 external_key: str = "image", image_size: int = 256,
                 technical_w: float = 1.0, alignment_w: float = 0.0,
                 external_w: float = 0.0):
    external_index = external_index or {}
    components = {}
    parts = []
    reasons = []
    if backend in ("stats", "ensemble") and technical_w > 0.0:
        score, metrics = technical_quality(row.record.path, image_size=image_size)
        components.update(metrics)
        parts.append(("technical", score, technical_w))
    if backend in ("embedding", "ensemble") and alignment_w > 0.0:
        score, metrics = embedding_quality(row.record)
        components.update(metrics)
        if score is None:
            reasons.append(metrics.get("embedding_quality_missing", "missing_embedding_alignment"))
        else:
            parts.append(("alignment", score, alignment_w))
    if backend in ("external", "ensemble") and external_w > 0.0:
        key = _record_key(row.record, key=external_key)
        ext = external_index.get(key)
        if ext is None or ext.get("score") is None:
            reasons.append("missing_external_score")
        else:
            score = float(ext["score"])
            components.update({
                "external_quality_score": score,
                "external_quality_score_raw": float(ext["raw"]),
                "external_quality_score_field": ext["field"],
            })
            parts.append(("external", score, external_w))
    final, final_parts = _weighted_geomean(parts)
    if final is None:
        return None, {
            "quality_score_missing": reasons or ["no_enabled_quality_component"],
            **components,
        }
    components.update({
        "quality_score": float(final),
        "quality_score_components": final_parts,
    })
    return float(final), components


def _prepare_output_row(input_row: dict, rec: ImageTextRecord, root: str = "") -> dict:
    row = dict(input_row)
    row["image"] = _manifest_image_path(rec.path, root=root)
    row["caption"] = rec.caption
    row["split"] = rec.split
    if rec.source and not row.get("source"):
        row["source"] = rec.source
    if rec.width and not row.get("width"):
        row["width"] = int(rec.width)
    if rec.height and not row.get("height"):
        row["height"] = int(rec.height)
    return row


def score_manifest(manifest: str, root: str = "", split: str = "", max_records: int = 0,
                   backend: str = "stats", out: str = "", sidecar_out: str = "",
                   report_out: str = "", external_sidecar: str = "",
                   external_root: str = "", external_key: str = "image",
                   external_score_field: str = "", external_normalize: str = "auto",
                   image_size: int = 256, technical_w: float = 1.0,
                   alignment_w: float = 0.0, external_w: float = 0.0,
                   drop_failed: bool = False):
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    if external_key not in KEYS:
        raise ValueError(f"unknown external key {external_key!r}")
    if external_normalize not in NORMALIZE_MODES:
        raise ValueError(f"unknown external normalize mode {external_normalize!r}")
    if backend == "embedding" and alignment_w <= 0.0:
        alignment_w = 1.0
    if backend == "external" and external_w <= 0.0:
        external_w = 1.0
    if backend == "stats" and technical_w <= 0.0:
        technical_w = 1.0

    external_index, external_report = load_external_scores(
        external_sidecar, root=external_root or root, key=external_key,
        score_field=external_score_field)
    external_report.update(normalize_external_scores(external_index, mode=external_normalize))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True) if out else None
    os.makedirs(os.path.dirname(sidecar_out) or ".", exist_ok=True) if sidecar_out else None
    out_f = open(out, "w", encoding="utf-8") if out else None
    side_f = open(sidecar_out, "w", encoding="utf-8") if sidecar_out else None
    score_vals = []
    component_vals: dict[str, list[float]] = {}
    failures = Counter()
    seen = written = side_written = scored = 0
    examples = []
    try:
        for row in iter_manifest_rows(manifest, root=root, split=split, max_records=max_records):
            seen += 1
            try:
                score, metrics = score_record(
                    row, backend=backend, external_index=external_index,
                    external_key=external_key, image_size=image_size,
                    technical_w=technical_w, alignment_w=alignment_w,
                    external_w=external_w)
            except Exception as e:
                score, metrics = None, {"quality_score_error": str(e)}
            output = _prepare_output_row(row.row, row.record, root=root)
            output["quality_backend"] = backend
            output["quality_score_version"] = "image_score_v1"
            output.update(metrics)
            if score is None:
                reasons = metrics.get("quality_score_missing") or [metrics.get(
                    "quality_score_error", "quality_score_failed")]
                for reason in reasons:
                    failures[str(reason)] += 1
                if drop_failed:
                    if len(examples) < 8:
                        examples.append({
                            "row": int(row.index),
                            "image": output.get("image", ""),
                            "caption": output.get("caption", ""),
                            "reasons": reasons,
                        })
                    continue
            else:
                scored += 1
                score_vals.append(float(score))
                output["quality_score"] = float(score)
                output["aesthetic"] = float(score)
                if "technical_quality_score" in metrics:
                    output["technical_quality_score"] = float(metrics["technical_quality_score"])
                if "embedding_alignment_score" in metrics:
                    output["embedding_alignment_score"] = float(metrics["embedding_alignment_score"])
                if "external_quality_score" in metrics:
                    output["external_quality_score"] = float(metrics["external_quality_score"])
                for name, data in metrics.get("quality_score_components", {}).items():
                    val = data.get("score")
                    if val is not None:
                        component_vals.setdefault(name, []).append(float(val))
            if out_f:
                out_f.write(json.dumps(output) + "\n")
                written += 1
            if side_f:
                side = {
                    "image": output.get("image", ""),
                    "caption": output.get("caption", ""),
                    "split": output.get("split", "train"),
                    "quality_score": output.get("quality_score"),
                    "aesthetic": output.get("aesthetic"),
                    "quality_backend": backend,
                    "quality_score_version": "image_score_v1",
                }
                for field in (
                    "technical_quality_score",
                    "embedding_alignment_score",
                    "external_quality_score",
                    "external_quality_score_raw",
                    "external_quality_score_field",
                    "image_text_cosine",
                ):
                    if field in output:
                        side[field] = output[field]
                side_f.write(json.dumps(side) + "\n")
                side_written += 1
    finally:
        if out_f:
            out_f.close()
        if side_f:
            side_f.close()

    report = {
        "experiment": "image_quality_score",
        "manifest": manifest,
        "root": root,
        "split": split,
        "backend": backend,
        "out": out,
        "sidecar_out": sidecar_out,
        "max_records": int(max_records),
        "image_size": int(image_size),
        "technical_w": float(technical_w),
        "alignment_w": float(alignment_w),
        "external_w": float(external_w),
        "records_seen": int(seen),
        "records_scored": int(scored),
        "records_written": int(written),
        "sidecar_records_written": int(side_written),
        "drop_failed": bool(drop_failed),
        "quality_score_stats": _stats(score_vals),
        "quality_component_stats": {
            name: _stats(vals) for name, vals in sorted(component_vals.items())
        },
        "quality_failures": dict(sorted(failures.items())),
        "quality_failure_examples": examples,
    }
    report.update(external_report)
    if report_out:
        os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return report


def _write_ppm(path: str, arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.uint8)
    h, w, _c = arr.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr.tobytes())


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def selftest():
    with tempfile.TemporaryDirectory() as td:
        flat = os.path.join(td, "flat.ppm")
        rich = os.path.join(td, "rich.ppm")
        x = np.linspace(0, 255, 32, dtype=np.uint8)
        grad = np.tile(x[None, :], (32, 1))
        checker = (((np.indices((32, 32)).sum(axis=0) % 2) * 85)).astype(np.uint8)
        rich_arr = np.stack([
            grad,
            np.flipud(grad),
            np.clip(80 + checker + grad // 4, 0, 255).astype(np.uint8),
        ], axis=-1)
        _write_ppm(flat, np.full((32, 32, 3), 128, dtype=np.uint8))
        _write_ppm(rich, rich_arr)
        manifest = os.path.join(td, "images.jsonl")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps({"image": "flat.ppm", "caption": "flat gray image"}) + "\n")
            f.write(json.dumps({"image": "rich.ppm", "caption": "detailed colorful image"}) + "\n")
        out = os.path.join(td, "scored.jsonl")
        report = score_manifest(
            manifest, root=td, backend="stats", out=out,
            sidecar_out=os.path.join(td, "scores.jsonl"),
            report_out=os.path.join(td, "report.json"), image_size=32)
        rows = _read_jsonl(out)
        assert report["records_scored"] == 2
        assert rows[1]["quality_score"] > rows[0]["quality_score"]
        assert rows[1]["technical_quality_score"] > rows[0]["technical_quality_score"]

        embed_manifest = os.path.join(td, "embed.jsonl")
        with open(embed_manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "image": "flat.ppm", "caption": "same",
                "image_embedding": [1.0, 0.0], "text_embedding": [1.0, 0.0],
            }) + "\n")
            f.write(json.dumps({
                "image": "rich.ppm", "caption": "different",
                "image_embedding": [1.0, 0.0], "text_embedding": [0.0, 1.0],
            }) + "\n")
        embed_out = os.path.join(td, "embed_scored.jsonl")
        score_manifest(embed_manifest, root=td, backend="embedding", out=embed_out)
        embed_rows = _read_jsonl(embed_out)
        assert embed_rows[0]["quality_score"] > embed_rows[1]["quality_score"]

        external = os.path.join(td, "external.jsonl")
        with open(external, "w", encoding="utf-8") as f:
            f.write(json.dumps({"image": "flat.ppm", "reward_score": 1.0}) + "\n")
            f.write(json.dumps({"image": "rich.ppm", "reward_score": 9.0}) + "\n")
        ext_out = os.path.join(td, "external_scored.jsonl")
        ext_report = score_manifest(
            manifest, root=td, backend="external", out=ext_out,
            external_sidecar=external, external_key="image", external_normalize="auto")
        ext_rows = _read_jsonl(ext_out)
        assert ext_report["external_score_normalized"] is True
        assert ext_rows[1]["quality_score"] > ext_rows[0]["quality_score"]

        ensemble_out = os.path.join(td, "ensemble_scored.jsonl")
        ens_report = score_manifest(
            manifest, root=td, backend="ensemble", out=ensemble_out,
            external_sidecar=external, external_key="image", image_size=32,
            technical_w=0.5, external_w=0.5, alignment_w=0.0)
        ens_rows = _read_jsonl(ensemble_out)
        assert ens_report["records_scored"] == 2
        assert "external" in ens_rows[1]["quality_score_components"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=False, default="",
                    help="input captioned image manifest")
    ap.add_argument("--root", default="", help="base directory for relative manifest images")
    ap.add_argument("--split", default="", help="optional split to score; default scores all rows")
    ap.add_argument("--max-records", type=int, default=0, help="cap scored rows; 0 means all")
    ap.add_argument("--backend", default="stats", choices=BACKENDS,
                    help="quality source: technical stats, image-text embedding, external, ensemble")
    ap.add_argument("--out", default="", help="scored manifest JSONL")
    ap.add_argument("--sidecar-out", default="", help="optional score-only JSONL sidecar")
    ap.add_argument("--report-out", default="", help="optional JSON report")
    ap.add_argument("--external-sidecar", default="",
                    help="JSONL/CSV/TSV with external model scores to merge")
    ap.add_argument("--external-root", default="",
                    help="base directory for relative external sidecar paths")
    ap.add_argument("--external-key", default="image", choices=KEYS,
                    help="join key for external sidecar")
    ap.add_argument("--external-score-field", default="",
                    help="explicit external score column; default searches common score fields")
    ap.add_argument("--external-normalize", default="auto", choices=NORMALIZE_MODES,
                    help="normalize external scores before writing quality_score")
    ap.add_argument("--image-size", type=int, default=256,
                    help="working image size for technical stats scorer")
    ap.add_argument("--technical-w", type=float, default=1.0,
                    help="ensemble weight for technical image-health score")
    ap.add_argument("--alignment-w", type=float, default=0.0,
                    help="ensemble weight for existing image/text embedding cosine")
    ap.add_argument("--external-w", type=float, default=0.0,
                    help="ensemble weight for external preference/reward score")
    ap.add_argument("--drop-failed", action="store_true",
                    help="drop rows that cannot be scored instead of preserving them")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        print("image_score selftest ok")
        return 0
    if not args.manifest:
        ap.error("--manifest is required unless --selftest is set")
    if not args.out and not args.sidecar_out and not args.report_out:
        ap.error("provide --out, --sidecar-out, or --report-out")
    if args.max_records < 0:
        ap.error("--max-records must be non-negative")
    if args.image_size <= 0:
        ap.error("--image-size must be positive")
    for name in ("technical_w", "alignment_w", "external_w"):
        if getattr(args, name) < 0.0:
            ap.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.backend in ("external", "ensemble") and args.external_w > 0.0:
        if not args.external_sidecar:
            ap.error("--external-sidecar is required when external scoring is enabled")
    report = score_manifest(
        args.manifest, root=args.root, split=args.split, max_records=args.max_records,
        backend=args.backend, out=args.out, sidecar_out=args.sidecar_out,
        report_out=args.report_out, external_sidecar=args.external_sidecar,
        external_root=args.external_root, external_key=args.external_key,
        external_score_field=args.external_score_field,
        external_normalize=args.external_normalize, image_size=args.image_size,
        technical_w=args.technical_w, alignment_w=args.alignment_w,
        external_w=args.external_w, drop_failed=args.drop_failed)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
