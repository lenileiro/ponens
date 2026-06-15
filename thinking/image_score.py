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
from concurrent.futures import ThreadPoolExecutor
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


BACKENDS = ("stats", "embedding", "external", "pickscore", "ensemble")
KEYS = ("image", "basename", "caption")
NORMALIZE_MODES = ("auto", "minmax", "none")
EPS = 1.0e-12
DEFAULT_PICKSCORE_MODEL = "yuvalkirstain/PickScore_v1"
DEFAULT_PICKSCORE_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
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


def _technical_rgb_tensor(path: str, image_size: int):
    """Fast RGB tensor loader for image-health stats.

    The shared training loader intentionally preserves transform metadata and torch-side
    interpolation semantics. The scorer only needs a fixed working image, so resizing
    through Pillow avoids materializing full-resolution tensors for large web images.
    """
    size = max(1, int(image_size))
    ext = os.path.splitext(path)[1].lower()
    if ext in (".ppm", ".pnm"):
        x = load_image_tensor(path, size=size, device="cpu", center_crop=False)
        return ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    try:
        from PIL import Image
    except Exception:
        x = load_image_tensor(path, size=size, device="cpu", center_crop=False)
        return ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), resample=resampling)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def technical_quality(path: str, image_size: int = 256) -> tuple[float, dict]:
    """Return a generic image-health score in [0, 1] plus component metrics."""
    rgb = _technical_rgb_tensor(path, image_size=image_size)
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


def _technical_quality_row(row: ManifestRow, image_size: int):
    try:
        score, metrics = technical_quality(row.record.path, image_size=image_size)
        return row.index, {"score": float(score), "metrics": metrics}
    except Exception as exc:
        return row.index, {"score": None, "metrics": {
            "technical_quality_error": str(exc),
        }}


def technical_quality_index(rows: Sequence[ManifestRow], image_size: int = 256,
                            workers: int = 0):
    workers = int(workers or 0)
    if workers <= 0:
        workers = min(8, max(1, (os.cpu_count() or 1)))
    workers = max(1, workers)
    index = {}
    if workers == 1 or len(rows) <= 1:
        for row in rows:
            row_id, item = _technical_quality_row(row, image_size)
            index[row_id] = item
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_technical_quality_row, row, image_size)
                for row in rows
            ]
            for future in futures:
                row_id, item = future.result()
                index[row_id] = item
    vals = [
        item["score"] for item in index.values()
        if item.get("score") is not None
    ]
    return index, {
        "technical_precomputed": True,
        "technical_workers": int(workers),
        "technical_rows": int(len(rows)),
        "technical_scored": int(len(vals)),
        "technical_failed": int(sum(
            1 for item in index.values() if item.get("score") is None)),
        "technical_quality_score_stats": _stats(vals),
    }


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


def _normalize_score_values(raw_values: Sequence[float | None], mode: str = "auto"):
    vals = np.asarray([
        float(v) if v is not None and math.isfinite(float(v)) else np.nan
        for v in raw_values
    ], dtype=np.float64)
    finite = np.isfinite(vals)
    out = [None for _ in raw_values]
    if not finite.any():
        return out, {
            "normalized": False,
            "min": None,
            "max": None,
        }
    lo = float(np.min(vals[finite]))
    hi = float(np.max(vals[finite]))
    use_minmax = mode == "minmax" or (mode == "auto" and (lo < 0.0 or hi > 1.0))
    for i, raw in enumerate(vals):
        if not math.isfinite(float(raw)):
            continue
        if use_minmax:
            out[i] = 0.5 if hi <= lo else _clamp01((float(raw) - lo) / max(hi - lo, EPS))
        else:
            out[i] = _clamp01(float(raw))
    return out, {
        "normalized": bool(use_minmax),
        "min": lo,
        "max": hi,
    }


def _pickscore_device(device: str):
    device = str(device or "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _pickscore_dtype(dtype: str, device: str):
    dtype = str(dtype or "auto")
    if dtype == "auto":
        return torch.float16 if str(device).startswith("cuda") else torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def _to_device(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def pickscore_quality_index(rows: Sequence[ManifestRow], model_name: str = "",
                            processor_name: str = "", device: str = "auto",
                            dtype: str = "auto", batch_size: int = 8,
                            max_length: int = 77, normalize: str = "auto"):
    """Score prompt/image alignment with a CLIP-style PickScore reward model."""
    if normalize not in NORMALIZE_MODES:
        raise ValueError(f"unknown PickScore normalize mode {normalize!r}")
    if int(batch_size) <= 0:
        raise ValueError("PickScore batch size must be positive")
    if int(max_length) <= 0:
        raise ValueError("PickScore max length must be positive")
    try:
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "PickScore backend requires pillow and transformers; install the image scoring "
            "dependencies or use --backend stats/external") from exc

    model_name = model_name or DEFAULT_PICKSCORE_MODEL
    processor_name = processor_name or DEFAULT_PICKSCORE_PROCESSOR
    device = _pickscore_device(device)
    torch_dtype = _pickscore_dtype(dtype, device)
    try:
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    except TypeError:
        model = AutoModel.from_pretrained(model_name)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(processor_name)

    raw_by_index: dict[int, float | None] = {}
    errors_by_index: dict[int, str] = {}
    raw_vals = []
    with torch.no_grad():
        for off in range(0, len(rows), int(batch_size)):
            batch = rows[off:off + int(batch_size)]
            images, captions, row_ids = [], [], []
            for row in batch:
                try:
                    with Image.open(row.record.path) as image:
                        images.append(image.convert("RGB").copy())
                    captions.append(row.record.caption)
                    row_ids.append(row.index)
                except Exception as exc:
                    raw_by_index[row.index] = None
                    errors_by_index[row.index] = str(exc)
            if not images:
                continue
            try:
                image_inputs = _to_device(
                    processor(images=images, return_tensors="pt"), device)
                text_inputs = _to_device(
                    processor(text=captions, padding=True, truncation=True,
                              max_length=int(max_length), return_tensors="pt"), device)
                image_features = model.get_image_features(**image_inputs).float()
                text_features = model.get_text_features(**text_inputs).float()
                image_features = torch.nn.functional.normalize(image_features, dim=-1)
                text_features = torch.nn.functional.normalize(text_features, dim=-1)
                logit_scale = getattr(model, "logit_scale", None)
                if logit_scale is not None:
                    scale = logit_scale.detach().float().exp().clamp(max=100.0)
                else:
                    scale = torch.ones((), device=image_features.device)
                raw = (image_features * text_features).sum(-1) * scale
                for row_id, score in zip(row_ids, raw.detach().cpu().tolist()):
                    raw_by_index[row_id] = float(score)
                    raw_vals.append(float(score))
            except Exception as exc:
                for row_id in row_ids:
                    raw_by_index[row_id] = None
                    errors_by_index[row_id] = str(exc)

    ordered_raw = [raw_by_index.get(row.index) for row in rows]
    normalized, norm_report = _normalize_score_values(ordered_raw, mode=normalize)
    index = {}
    for row, raw, score in zip(rows, ordered_raw, normalized):
        item = {
            "raw": raw,
            "score": score,
        }
        if row.index in errors_by_index:
            item["error"] = errors_by_index[row.index]
        index[row.index] = item
    report = {
        "pickscore_model": model_name,
        "pickscore_processor": processor_name,
        "pickscore_device": device,
        "pickscore_dtype": str(dtype),
        "pickscore_batch": int(batch_size),
        "pickscore_max_length": int(max_length),
        "pickscore_normalize": normalize,
        "pickscore_normalized": bool(norm_report["normalized"]),
        "pickscore_norm_min": norm_report["min"],
        "pickscore_norm_max": norm_report["max"],
        "pickscore_rows": int(len(rows)),
        "pickscore_scored": int(sum(1 for v in raw_by_index.values() if v is not None)),
        "pickscore_failed": int(sum(1 for v in raw_by_index.values() if v is None)),
        "pickscore_raw_stats": _stats(raw_vals),
        "pickscore_score_stats": _stats([
            item["score"] for item in index.values() if item.get("score") is not None
        ]),
    }
    return index, report


def score_record(row: ManifestRow, backend: str = "stats", external_index: dict | None = None,
                 pickscore_index: dict | None = None,
                 technical_index: dict | None = None,
                 external_key: str = "image", image_size: int = 256,
                 technical_w: float = 1.0, alignment_w: float = 0.0,
                 external_w: float = 0.0, pickscore_w: float = 0.0):
    external_index = external_index or {}
    pickscore_index = pickscore_index or {}
    technical_index = technical_index or {}
    components = {}
    parts = []
    reasons = []
    if backend in ("stats", "ensemble") and technical_w > 0.0:
        item = technical_index.get(row.index)
        if item is None:
            score, metrics = technical_quality(row.record.path, image_size=image_size)
        else:
            score = item.get("score")
            metrics = item.get("metrics", {})
        components.update(metrics)
        if score is None:
            reasons.append("technical_quality_error")
        else:
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
    if backend in ("pickscore", "ensemble") and pickscore_w > 0.0:
        ps = pickscore_index.get(row.index)
        if ps is None or ps.get("score") is None:
            reason = "missing_pickscore"
            if ps is not None and ps.get("error"):
                reason = "pickscore_error"
                components["pickscore_quality_error"] = str(ps["error"])
            reasons.append(reason)
        else:
            score = float(ps["score"])
            components.update({
                "pickscore_quality_score": score,
                "pickscore_quality_score_raw": float(ps["raw"]),
            })
            parts.append(("pickscore", score, pickscore_w))
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
                   technical_workers: int = 0,
                   alignment_w: float = 0.0, external_w: float = 0.0,
                   pickscore_w: float = 0.0, pickscore_model: str = "",
                   pickscore_processor: str = "", pickscore_device: str = "auto",
                   pickscore_dtype: str = "auto", pickscore_batch: int = 8,
                   pickscore_max_length: int = 77, pickscore_normalize: str = "auto",
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
    if backend == "pickscore" and pickscore_w <= 0.0:
        pickscore_w = 1.0
    if backend == "stats" and technical_w <= 0.0:
        technical_w = 1.0

    external_index, external_report = load_external_scores(
        external_sidecar, root=external_root or root, key=external_key,
        score_field=external_score_field)
    external_report.update(normalize_external_scores(external_index, mode=external_normalize))
    needs_pickscore = backend == "pickscore" or (
        backend == "ensemble" and float(pickscore_w) > 0.0)
    needs_technical = backend == "stats" or (
        backend == "ensemble" and float(technical_w) > 0.0)
    row_source = iter_manifest_rows(
        manifest, root=root, split=split, max_records=max_records)
    pickscore_index, pickscore_report = {}, {
        "pickscore_enabled": bool(needs_pickscore),
    }
    if needs_pickscore or (needs_technical and int(technical_workers or 0) != 1):
        row_source = list(row_source)
    technical_index, technical_report = {}, {
        "technical_precomputed": False,
        "technical_workers": 1,
    }
    if needs_technical and int(technical_workers or 0) != 1:
        technical_index, technical_report = technical_quality_index(
            row_source, image_size=image_size, workers=technical_workers)
    if needs_pickscore:
        pickscore_index, pickscore_report = pickscore_quality_index(
            row_source, model_name=pickscore_model, processor_name=pickscore_processor,
            device=pickscore_device, dtype=pickscore_dtype, batch_size=pickscore_batch,
            max_length=pickscore_max_length, normalize=pickscore_normalize)
        pickscore_report["pickscore_enabled"] = True

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
        for row in row_source:
            seen += 1
            try:
                score, metrics = score_record(
                    row, backend=backend, external_index=external_index,
                    pickscore_index=pickscore_index,
                    technical_index=technical_index,
                    external_key=external_key, image_size=image_size,
                    technical_w=technical_w, alignment_w=alignment_w,
                    external_w=external_w, pickscore_w=pickscore_w)
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
                if "pickscore_quality_score" in metrics:
                    output["pickscore_quality_score"] = float(metrics["pickscore_quality_score"])
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
                    "pickscore_quality_score",
                    "pickscore_quality_score_raw",
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
        "technical_workers_requested": int(technical_workers or 0),
        "alignment_w": float(alignment_w),
        "external_w": float(external_w),
        "pickscore_w": float(pickscore_w),
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
    report.update(technical_report)
    report.update(pickscore_report)
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
        assert report["technical_precomputed"] is True
        assert report["technical_workers"] >= 1
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
        assert ens_report["technical_precomputed"] is True
        assert "external" in ens_rows[1]["quality_score_components"]
        norm, norm_report = _normalize_score_values([1.0, 9.0], mode="auto")
        assert norm_report["normalized"] is True
        assert norm[0] == 0.0 and norm[1] == 1.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=False, default="",
                    help="input captioned image manifest")
    ap.add_argument("--root", default="", help="base directory for relative manifest images")
    ap.add_argument("--split", default="", help="optional split to score; default scores all rows")
    ap.add_argument("--max-records", type=int, default=0, help="cap scored rows; 0 means all")
    ap.add_argument("--backend", default="stats", choices=BACKENDS,
                    help=("quality source: technical stats, image-text embedding, external, "
                          "PickScore, or ensemble"))
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
    ap.add_argument("--technical-workers", type=int, default=0,
                    help=("parallel CPU workers for technical image-health scoring; "
                          "0 auto-selects up to 8, 1 preserves serial scoring"))
    ap.add_argument("--alignment-w", type=float, default=0.0,
                    help="ensemble weight for existing image/text embedding cosine")
    ap.add_argument("--external-w", type=float, default=0.0,
                    help="ensemble weight for external preference/reward score")
    ap.add_argument("--pickscore-w", type=float, default=0.0,
                    help="ensemble weight for PickScore prompt-image reward")
    ap.add_argument("--pickscore-model", default=DEFAULT_PICKSCORE_MODEL,
                    help="Hugging Face model id for PickScore reward scoring")
    ap.add_argument("--pickscore-processor", default=DEFAULT_PICKSCORE_PROCESSOR,
                    help="Hugging Face processor id for PickScore reward scoring")
    ap.add_argument("--pickscore-device", default="auto",
                    help="device for PickScore reward scoring; auto uses CUDA when available")
    ap.add_argument("--pickscore-dtype", default="auto",
                    choices=("auto", "fp32", "fp16", "bf16"),
                    help="dtype for PickScore reward scoring")
    ap.add_argument("--pickscore-batch", type=int, default=8,
                    help="batch size for PickScore reward scoring")
    ap.add_argument("--pickscore-max-length", type=int, default=77,
                    help="tokenizer max length for PickScore prompt text")
    ap.add_argument("--pickscore-normalize", default="auto", choices=NORMALIZE_MODES,
                    help="normalize raw PickScore rewards before writing quality_score")
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
    if args.technical_workers < 0:
        ap.error("--technical-workers must be non-negative")
    for name in ("technical_w", "alignment_w", "external_w", "pickscore_w"):
        if getattr(args, name) < 0.0:
            ap.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.pickscore_batch <= 0:
        ap.error("--pickscore-batch must be positive")
    if args.pickscore_max_length <= 0:
        ap.error("--pickscore-max-length must be positive")
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
        technical_w=args.technical_w, technical_workers=args.technical_workers,
        alignment_w=args.alignment_w,
        external_w=args.external_w, pickscore_w=args.pickscore_w,
        pickscore_model=args.pickscore_model,
        pickscore_processor=args.pickscore_processor,
        pickscore_device=args.pickscore_device,
        pickscore_dtype=args.pickscore_dtype,
        pickscore_batch=args.pickscore_batch,
        pickscore_max_length=args.pickscore_max_length,
        pickscore_normalize=args.pickscore_normalize,
        drop_failed=args.drop_failed)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
