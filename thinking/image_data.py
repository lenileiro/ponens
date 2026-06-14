"""Captioned image manifest utilities for real vision and image-generation rungs.

High-quality vision and generation work needs real image/text corpora.  This module is
deliberately light on dependencies:

* PPM is supported directly for tests and tiny local fixtures.
* JPEG/PNG/WebP use Pillow when it is installed on the GPU box.
* Manifests are JSONL, CSV, or TSV with at least an image path and caption/text field.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


IMAGE_EXTS = (".ppm", ".pnm", ".jpg", ".jpeg", ".png", ".webp", ".bmp")
TOKEN_RE = re.compile(r"[a-z0-9]+|[^\s\w]", re.IGNORECASE)


@dataclass(frozen=True)
class ImageTextRecord:
    path: str
    caption: str
    split: str = "train"
    source: str = ""
    aesthetic: float | None = None
    nsfw: float | None = None
    watermark: float | None = None
    width: int = 0
    height: int = 0
    text_embedding: tuple[float, ...] | None = None
    text_embedding_sequence: tuple[tuple[float, ...], ...] | None = None
    image_embedding: tuple[float, ...] | None = None
    image_embedding_sequence: tuple[tuple[float, ...], ...] | None = None


def caption_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]


def _longest_run(vals):
    best = run = 0
    prev = None
    for val in vals:
        if val == prev:
            run += 1
        else:
            prev = val
            run = 1
        if run > best:
            best = run
    return best


def caption_quality_metrics(text: str) -> dict[str, float | int]:
    toks = caption_tokens(text)
    token_count = len(toks)
    if token_count:
        counts = Counter(toks)
        unique_ratio = len(counts) / float(token_count)
        max_token_frequency = max(counts.values()) / float(token_count)
        max_token_run = _longest_run(toks) / float(token_count)
    else:
        unique_ratio = 0.0
        max_token_frequency = 0.0
        max_token_run = 0.0
    chars = [ch.lower() for ch in str(text) if not ch.isspace()]
    char_count = len(chars)
    max_char_run = (_longest_run(chars) / float(char_count)) if char_count else 0.0
    return {
        "caption_tokens": int(token_count),
        "caption_chars": int(char_count),
        "caption_unique_ratio": float(unique_ratio),
        "caption_max_token_frequency": float(max_token_frequency),
        "caption_max_token_run": float(max_token_run),
        "caption_max_char_run": float(max_char_run),
    }


def _coerce_optional_float(raw):
    if raw in ("", None):
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        low = raw.lower()
        if low in ("true", "yes"):
            return 1.0
        if low in ("false", "no"):
            return 0.0
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    return float(raw)


def _row_float(row, *keys, mode="first"):
    vals = []
    for key in keys:
        try:
            val = _coerce_optional_float(row.get(key))
        except (TypeError, ValueError):
            continue
        if val is not None:
            if mode == "first":
                return val
            vals.append(val)
    if not vals:
        return None
    if mode == "max":
        return max(vals)
    if mode == "min":
        return min(vals)
    raise ValueError(f"unknown float aggregation mode {mode!r}")


def _coerce_float_embedding(raw):
    if raw in ("", None):
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        raw = json.loads(raw)
    vals = tuple(float(x) for x in raw)
    if not vals:
        return None
    return vals


def _coerce_float_embedding_sequence(raw):
    if raw in ("", None):
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        raw = json.loads(raw)
    rows = tuple(tuple(float(x) for x in row) for row in raw)
    rows = tuple(row for row in rows if row)
    if not rows:
        return None
    dims = {len(row) for row in rows}
    if len(dims) != 1:
        raise ValueError(f"text embedding sequence has mixed dimensions: {sorted(dims)}")
    return rows


def _coerce_text_embedding(raw):
    return _coerce_float_embedding(raw)


def _coerce_text_embedding_sequence(raw):
    return _coerce_float_embedding_sequence(raw)


def _coerce_image_embedding(raw):
    return _coerce_float_embedding(raw)


def _coerce_image_embedding_sequence(raw):
    return _coerce_float_embedding_sequence(raw)


def _coerce_record(row, manifest_dir, root=""):
    image = row.get("image") or row.get("path") or row.get("file") or row.get("filepath")
    caption = row.get("caption") or row.get("text") or row.get("prompt")
    if not image or caption is None:
        raise ValueError("image manifest rows require an image/path/file and caption/text/prompt")
    base = root or manifest_dir
    path = str(image)
    if base and not os.path.isabs(path):
        path = os.path.join(base, path)
    aesthetic = _row_float(
        row, "aesthetic", "aesthetic_score", "score", "quality", "quality_score")
    nsfw = _row_float(
        row, "nsfw", "image_nsfw", "prompt_nsfw", "unsafe", "safety_score", mode="max")
    watermark = _row_float(
        row, "watermark", "watermark_score", "has_watermark", "text_watermark",
        mode="max")
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    text_embedding = _coerce_text_embedding(
        row.get("text_embedding",
                row.get("caption_embedding",
                        row.get("embedding", row.get("text_emb"))))
    )
    text_embedding_sequence = _coerce_text_embedding_sequence(
        row.get("text_embedding_sequence",
                row.get("text_token_embeddings",
                        row.get("caption_token_embeddings",
                                row.get("token_embeddings",
                                        row.get("text_tokens_embedding")))))
    )
    image_embedding = _coerce_image_embedding(
        row.get("image_embedding",
                row.get("visual_embedding",
                        row.get("vision_embedding",
                                row.get("clip_image_embedding",
                                        row.get("dino_embedding",
                                                row.get("image_emb",
                                                        row.get("visual_emb")))))))
    )
    image_embedding_sequence = _coerce_image_embedding_sequence(
        row.get("image_embedding_sequence",
                row.get("visual_embedding_sequence",
                        row.get("vision_embedding_sequence",
                                row.get("image_token_embeddings",
                                        row.get("visual_token_embeddings",
                                                row.get("dino_token_embeddings",
                                                        row.get("clip_image_token_embeddings")))))))
    )
    return ImageTextRecord(
        path=os.path.normpath(path),
        caption=str(caption),
        split=str(row.get("split") or "train"),
        source=str(
            row.get("source")
            or row.get("dataset")
            or row.get("domain")
            or row.get("collection")
            or row.get("bucket")
            or ""
        ),
        aesthetic=aesthetic,
        nsfw=nsfw,
        watermark=watermark,
        width=width,
        height=height,
        text_embedding=text_embedding,
        text_embedding_sequence=text_embedding_sequence,
        image_embedding=image_embedding,
        image_embedding_sequence=image_embedding_sequence,
    )


def _path_key(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _caption_key(caption):
    return " ".join(str(caption).strip().lower().split())


def _embedding_key_from_row(row, sidecar_dir, root="", key="image"):
    key = str(key)
    if key == "caption":
        caption = row.get("caption") or row.get("text") or row.get("prompt")
        return _caption_key(caption) if caption is not None else None
    image = row.get("image") or row.get("path") or row.get("file") or row.get("filepath")
    if not image:
        return None
    if key == "basename":
        return os.path.basename(str(image))
    if key != "image":
        raise ValueError(f"unknown embedding merge key {key!r}")
    base = root or sidecar_dir
    path = str(image)
    if base and not os.path.isabs(path):
        path = os.path.join(base, path)
    return _path_key(path)


def _embedding_key_from_record(rec, key="image"):
    key = str(key)
    if key == "caption":
        return _caption_key(rec.caption)
    if key == "basename":
        return os.path.basename(rec.path)
    if key != "image":
        raise ValueError(f"unknown embedding merge key {key!r}")
    return _path_key(rec.path)


def merge_embedding_sidecar(records, sidecar_path, root="", key="image", overwrite=False):
    rows = _read_manifest_rows(sidecar_path)
    sidecar_dir = os.path.dirname(os.path.abspath(sidecar_path))
    index = {}
    skipped_no_key = skipped_no_embedding = duplicate_keys = 0
    for row in rows:
        row_key = _embedding_key_from_row(row, sidecar_dir, root=root, key=key)
        if row_key is None:
            skipped_no_key += 1
            continue
        text_embedding = _coerce_text_embedding(
            row.get("text_embedding",
                    row.get("caption_embedding",
                            row.get("embedding", row.get("text_emb"))))
        )
        text_embedding_sequence = _coerce_text_embedding_sequence(
            row.get("text_embedding_sequence",
                    row.get("text_token_embeddings",
                            row.get("caption_token_embeddings",
                                    row.get("token_embeddings",
                                            row.get("text_tokens_embedding")))))
        )
        image_embedding = _coerce_image_embedding(
            row.get("image_embedding",
                    row.get("visual_embedding",
                            row.get("vision_embedding",
                                    row.get("clip_image_embedding",
                                            row.get("dino_embedding",
                                                    row.get("image_emb",
                                                            row.get("visual_emb")))))))
        )
        image_embedding_sequence = _coerce_image_embedding_sequence(
            row.get("image_embedding_sequence",
                    row.get("visual_embedding_sequence",
                            row.get("vision_embedding_sequence",
                                    row.get("image_token_embeddings",
                                            row.get("visual_token_embeddings",
                                                    row.get("dino_token_embeddings",
                                                            row.get("clip_image_token_embeddings")))))))
        )
        if (text_embedding is None and text_embedding_sequence is None
                and image_embedding is None and image_embedding_sequence is None):
            skipped_no_embedding += 1
            continue
        if row_key in index:
            duplicate_keys += 1
            continue
        index[row_key] = (
            text_embedding, text_embedding_sequence,
            image_embedding, image_embedding_sequence)

    merged = []
    matched = missing = 0
    text_added = image_added = 0
    text_sequence_added = image_sequence_added = 0
    text_preserved = image_preserved = 0
    text_sequence_preserved = image_sequence_preserved = 0
    for rec in records:
        rec_key = _embedding_key_from_record(rec, key=key)
        vals = index.get(rec_key)
        if vals is None:
            missing += 1
            merged.append(rec)
            continue
        matched += 1
        text_embedding, text_embedding_sequence, image_embedding, image_embedding_sequence = vals
        new_text = rec.text_embedding
        new_text_sequence = rec.text_embedding_sequence
        new_image = rec.image_embedding
        new_image_sequence = rec.image_embedding_sequence
        if text_embedding is not None:
            if overwrite or rec.text_embedding is None:
                if rec.text_embedding != text_embedding:
                    text_added += 1
                new_text = text_embedding
            else:
                text_preserved += 1
        if text_embedding_sequence is not None:
            if overwrite or rec.text_embedding_sequence is None:
                if rec.text_embedding_sequence != text_embedding_sequence:
                    text_sequence_added += 1
                new_text_sequence = text_embedding_sequence
            else:
                text_sequence_preserved += 1
        if image_embedding is not None:
            if overwrite or rec.image_embedding is None:
                if rec.image_embedding != image_embedding:
                    image_added += 1
                new_image = image_embedding
            else:
                image_preserved += 1
        if image_embedding_sequence is not None:
            if overwrite or rec.image_embedding_sequence is None:
                if rec.image_embedding_sequence != image_embedding_sequence:
                    image_sequence_added += 1
                new_image_sequence = image_embedding_sequence
            else:
                image_sequence_preserved += 1
        merged.append(replace(
            rec, text_embedding=new_text, text_embedding_sequence=new_text_sequence,
            image_embedding=new_image, image_embedding_sequence=new_image_sequence))

    text_dims = sorted({len(v[0]) for v in index.values() if v[0] is not None})
    text_sequence_dims = sorted({
        len(row)
        for _flat, seq, _img, _img_seq in index.values() if seq is not None
        for row in seq[:1]
    })
    text_sequence_lengths = [
        len(seq) for _flat, seq, _img, _img_seq in index.values() if seq is not None
    ]
    image_dims = sorted({len(v[2]) for v in index.values() if v[2] is not None})
    image_sequence_dims = sorted({
        len(row)
        for _flat, _seq, _img, img_seq in index.values() if img_seq is not None
        for row in img_seq[:1]
    })
    image_sequence_lengths = [
        len(img_seq) for _flat, _seq, _img, img_seq in index.values()
        if img_seq is not None
    ]
    report = {
        "embedding_sidecar": sidecar_path,
        "embedding_key": key,
        "embedding_rows": len(rows),
        "embedding_indexed": len(index),
        "embedding_skipped_no_key": int(skipped_no_key),
        "embedding_skipped_no_embedding": int(skipped_no_embedding),
        "embedding_duplicate_keys": int(duplicate_keys),
        "embedding_records_matched": int(matched),
        "embedding_records_missing": int(missing),
        "embedding_text_written": int(text_added),
        "embedding_text_sequence_written": int(text_sequence_added),
        "embedding_image_written": int(image_added),
        "embedding_image_sequence_written": int(image_sequence_added),
        "embedding_text_preserved": int(text_preserved),
        "embedding_text_sequence_preserved": int(text_sequence_preserved),
        "embedding_image_preserved": int(image_preserved),
        "embedding_image_sequence_preserved": int(image_sequence_preserved),
        "embedding_overwrite": bool(overwrite),
    }
    if text_dims:
        report["embedding_text_dims"] = text_dims
    if text_sequence_dims:
        report["embedding_text_sequence_dims"] = text_sequence_dims
        report["embedding_text_sequence_len_min"] = int(min(text_sequence_lengths))
        report["embedding_text_sequence_len_max"] = int(max(text_sequence_lengths))
    if image_dims:
        report["embedding_image_dims"] = image_dims
    if image_sequence_dims:
        report["embedding_image_sequence_dims"] = image_sequence_dims
        report["embedding_image_sequence_len_min"] = int(min(image_sequence_lengths))
        report["embedding_image_sequence_len_max"] = int(max(image_sequence_lengths))
    return merged, report


def _read_manifest_rows(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    elif ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            try:
                has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
            except csv.Error:
                has_header = False
            if has_header:
                return list(csv.DictReader(f, delimiter=delimiter))
            return [
                    {"path": r[0], "caption": r[1], "split": r[2] if len(r) > 2 else "train"}
                    for r in csv.reader(f, delimiter=delimiter) if len(r) >= 2
                ]
    raise ValueError(f"unsupported image manifest format {ext!r}; use .jsonl, .csv, or .tsv")


def read_image_manifest(path, root="", split="train", min_aesthetic=None, max_records=0,
                        max_nsfw=None, max_watermark=None):
    """Read captioned image records from JSONL/CSV/TSV.

    JSONL fields: image/path/file, caption/text/prompt, optional split/aesthetic/width/height.
    TSV rows are interpreted as: path<TAB>caption[<TAB>split].
    """
    manifest_dir = os.path.dirname(os.path.abspath(path))
    rows = _read_manifest_rows(path)

    records = []
    for row in rows:
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
        records.append(rec)
        if max_records and len(records) >= int(max_records):
            break
    if not records:
        raise ValueError(f"image manifest {path!r} yielded no records for split={split!r}")
    return records


def summarize_records(records: Iterable[ImageTextRecord]):
    rows = list(records)
    splits = {}
    sources = {}
    aesthetic = []
    nsfw = []
    watermark = []
    caption_lens = []
    for rec in rows:
        splits[rec.split] = splits.get(rec.split, 0) + 1
        source = str(getattr(rec, "source", "") or "").strip()
        if source:
            sources[source] = sources.get(source, 0) + 1
        if rec.aesthetic is not None:
            aesthetic.append(float(rec.aesthetic))
        if rec.nsfw is not None:
            nsfw.append(float(rec.nsfw))
        if rec.watermark is not None:
            watermark.append(float(rec.watermark))
        caption_lens.append(len(caption_tokens(rec.caption)))
    out = {
        "image_records": len(rows),
        "image_splits": splits,
        "image_sources": dict(sorted(sources.items())),
        "caption_token_mean": float(np.mean(caption_lens)) if caption_lens else 0.0,
        "text_embedding_records": sum(1 for r in rows if r.text_embedding is not None),
        "text_embedding_sequence_records": sum(
            1 for r in rows if r.text_embedding_sequence is not None),
        "image_embedding_records": sum(1 for r in rows if r.image_embedding is not None),
        "image_embedding_sequence_records": sum(
            1 for r in rows if r.image_embedding_sequence is not None),
    }
    dims = sorted({len(r.text_embedding) for r in rows if r.text_embedding is not None})
    if dims:
        out["text_embedding_dims"] = dims
    seq_dims = sorted({
        len(seq[0])
        for r in rows if r.text_embedding_sequence is not None
        for seq in (r.text_embedding_sequence,)
        if seq
    })
    seq_lens = [len(r.text_embedding_sequence)
                for r in rows if r.text_embedding_sequence is not None]
    if seq_dims:
        out["text_embedding_sequence_dims"] = seq_dims
        out["text_embedding_sequence_len_mean"] = float(np.mean(seq_lens))
        out["text_embedding_sequence_len_min"] = int(np.min(seq_lens))
        out["text_embedding_sequence_len_max"] = int(np.max(seq_lens))
    image_dims = sorted({len(r.image_embedding) for r in rows if r.image_embedding is not None})
    if image_dims:
        out["image_embedding_dims"] = image_dims
    image_seq_dims = sorted({
        len(seq[0])
        for r in rows if r.image_embedding_sequence is not None
        for seq in (r.image_embedding_sequence,)
        if seq
    })
    image_seq_lens = [len(r.image_embedding_sequence)
                      for r in rows if r.image_embedding_sequence is not None]
    if image_seq_dims:
        out["image_embedding_sequence_dims"] = image_seq_dims
        out["image_embedding_sequence_len_mean"] = float(np.mean(image_seq_lens))
        out["image_embedding_sequence_len_min"] = int(np.min(image_seq_lens))
        out["image_embedding_sequence_len_max"] = int(np.max(image_seq_lens))
    if aesthetic:
        out.update({
            "aesthetic_mean": float(np.mean(aesthetic)),
            "aesthetic_min": float(np.min(aesthetic)),
            "aesthetic_max": float(np.max(aesthetic)),
        })
    if nsfw:
        out.update({
            "nsfw_mean": float(np.mean(nsfw)),
            "nsfw_min": float(np.min(nsfw)),
            "nsfw_max": float(np.max(nsfw)),
        })
    if watermark:
        out.update({
            "watermark_mean": float(np.mean(watermark)),
            "watermark_min": float(np.min(watermark)),
            "watermark_max": float(np.max(watermark)),
        })
    return out


def _stats(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _record_embedding_vector(rec, side):
    if side == "text":
        vec = rec.text_embedding
        seq = rec.text_embedding_sequence
    elif side == "image":
        vec = rec.image_embedding
        seq = rec.image_embedding_sequence
    else:
        raise ValueError(f"unknown embedding side {side!r}")
    if vec is not None:
        arr = np.asarray(vec, dtype=np.float64)
    elif seq is not None:
        arr = np.asarray(seq, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > 0:
            arr = arr.mean(axis=0)
    else:
        return None, f"missing_{side}_embedding"
    if arr.ndim != 1 or arr.size <= 0:
        return None, f"invalid_{side}_embedding"
    if not np.all(np.isfinite(arr)):
        return None, f"nonfinite_{side}_embedding"
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return None, f"zero_{side}_embedding"
    return arr / norm, ""


def image_text_embedding_cosine(rec):
    text, text_reason = _record_embedding_vector(rec, "text")
    if text is None:
        return None, text_reason
    image, image_reason = _record_embedding_vector(rec, "image")
    if image is None:
        return None, image_reason
    if text.shape != image.shape:
        return None, "embedding_dim_mismatch"
    return float(np.dot(text, image)), ""


def filter_records_by_image_text_cosine(records, min_cosine=None, sample_errors=8):
    threshold = None if min_cosine is None else float(min_cosine)
    kept, rejected = [], []
    scores = []
    reject_causes = Counter()
    comparable = 0
    for idx, rec in enumerate(records):
        score, reason = image_text_embedding_cosine(rec)
        if score is None:
            reject_causes[reason] += 1
            if threshold is None:
                kept.append(rec)
            else:
                rejected.append({
                    "row": idx,
                    "path": rec.path,
                    "caption": rec.caption,
                    "reasons": [reason],
                })
            continue
        comparable += 1
        scores.append(score)
        if threshold is not None and score < threshold:
            reject_causes["image_text_cosine_below_threshold"] += 1
            rejected.append({
                "row": idx,
                "path": rec.path,
                "caption": rec.caption,
                "reasons": ["image_text_cosine_below_threshold"],
                "image_text_cosine": float(score),
            })
        else:
            kept.append(rec)
    report = {
        "image_text_cosine_filter_enabled": threshold is not None,
        "image_text_cosine_min": threshold,
        "image_text_cosine_records": int(comparable),
        "image_text_cosine_missing_or_invalid": int(len(records) - comparable),
        "image_text_cosine_stats": _stats(scores),
        "image_text_cosine_records_kept": len(kept),
        "image_text_cosine_records_rejected": len(rejected),
        "image_text_cosine_reject_causes": dict(sorted(reject_causes.items())),
        "image_text_cosine_error_examples": rejected[:max(0, int(sample_errors))],
    }
    return kept, report, rejected


def _lsh_projection_tables(dim, bits, tables, seed):
    projections = []
    for table in range(int(tables)):
        rng = np.random.default_rng(int(seed) + 1000003 * table + 9176 * int(dim))
        proj = rng.standard_normal((int(bits), int(dim))).astype(np.float64)
        norm = np.linalg.norm(proj, axis=1, keepdims=True)
        projections.append(proj / np.maximum(norm, 1.0e-12))
    return projections


def _lsh_key(vec, proj):
    signs = (proj @ vec) >= 0.0
    return np.packbits(signs.astype(np.uint8)).tobytes()


def _dedupe_quality_score(rec):
    if rec.aesthetic is not None:
        return float(rec.aesthetic)
    return 0.0


def filter_records_by_image_near_duplicates(records, max_cosine=None, lsh_bits=18,
                                            lsh_tables=4, seed=0,
                                            prefer_quality=True, sample_errors=8):
    threshold = None if max_cosine is None else float(max_cosine)
    if threshold is None:
        return list(records), {
            "image_near_duplicate_filter_enabled": False,
            "image_near_duplicate_max_cosine": None,
            "image_near_duplicate_records": 0,
            "image_near_duplicate_records_kept": len(records),
            "image_near_duplicate_records_rejected": 0,
        }, []
    bits = max(1, int(lsh_bits))
    tables = max(1, int(lsh_tables))
    indexed_vectors = {}
    projections_by_dim = {}
    buckets = {}
    kept_indices = []
    rejected = []
    reject_causes = Counter()
    invalid_causes = Counter()
    missing_or_invalid = 0
    comparable = 0
    exact_comparisons = 0
    candidate_pool_sizes = []
    order = list(range(len(records)))
    if prefer_quality:
        order.sort(key=lambda i: (-_dedupe_quality_score(records[i]), i))
    for idx in order:
        rec = records[idx]
        vec, reason = _record_embedding_vector(rec, "image")
        if vec is None:
            missing_or_invalid += 1
            invalid_causes[reason] += 1
            kept_indices.append(idx)
            continue
        comparable += 1
        dim = int(vec.shape[0])
        if dim not in projections_by_dim:
            projections_by_dim[dim] = _lsh_projection_tables(dim, bits, tables, seed)
        candidate_ids = set()
        keys = []
        for table, proj in enumerate(projections_by_dim[dim]):
            key = (dim, table, _lsh_key(vec, proj))
            keys.append(key)
            candidate_ids.update(buckets.get(key, ()))
        candidate_pool_sizes.append(len(candidate_ids))
        best_idx = None
        best_score = -2.0
        for cand_idx in candidate_ids:
            cand_vec = indexed_vectors.get(cand_idx)
            if cand_vec is None:
                continue
            score = float(np.dot(vec, cand_vec))
            exact_comparisons += 1
            if score > best_score:
                best_score = score
                best_idx = cand_idx
        if best_idx is not None and best_score >= threshold:
            reject_causes["image_near_duplicate"] += 1
            rejected.append({
                "row": idx,
                "path": rec.path,
                "caption": rec.caption,
                "reasons": ["image_near_duplicate"],
                "image_duplicate_cosine": float(best_score),
                "duplicate_of_row": int(best_idx),
                "duplicate_of_path": records[best_idx].path,
            })
            continue
        kept_indices.append(idx)
        indexed_vectors[idx] = vec
        for key in keys:
            buckets.setdefault(key, []).append(idx)
    kept_indices = sorted(kept_indices)
    kept = [records[i] for i in kept_indices]
    bucket_sizes = [len(v) for v in buckets.values()]
    report = {
        "image_near_duplicate_filter_enabled": True,
        "image_near_duplicate_max_cosine": float(threshold),
        "image_near_duplicate_lsh_bits": int(bits),
        "image_near_duplicate_lsh_tables": int(tables),
        "image_near_duplicate_lsh_seed": int(seed),
        "image_near_duplicate_prefer_quality": bool(prefer_quality),
        "image_near_duplicate_records": int(comparable),
        "image_near_duplicate_missing_or_invalid": int(missing_or_invalid),
        "image_near_duplicate_exact_comparisons": int(exact_comparisons),
        "image_near_duplicate_bucket_count": int(len(buckets)),
        "image_near_duplicate_bucket_size_stats": _stats(bucket_sizes),
        "image_near_duplicate_candidate_pool_stats": _stats(candidate_pool_sizes),
        "image_near_duplicate_records_kept": len(kept),
        "image_near_duplicate_records_rejected": len(rejected),
        "image_near_duplicate_reject_causes": dict(sorted(reject_causes.items())),
        "image_near_duplicate_invalid_causes": dict(sorted(invalid_causes.items())),
        "image_near_duplicate_error_examples": rejected[:max(0, int(sample_errors))],
    }
    return kept, report, rejected


def build_caption_vocab(records, max_vocab=8192, min_freq=1):
    counts = {}
    for rec in records:
        for tok in caption_tokens(rec.caption):
            counts[tok] = counts.get(tok, 0) + 1
    items = sorted(
        ((tok, n) for tok, n in counts.items() if n >= int(min_freq)),
        key=lambda kv: (-kv[1], kv[0]),
    )
    toks = ["<pad>", "<unk>"] + [tok for tok, _n in items[:max(0, int(max_vocab) - 2)]]
    return {tok: i for i, tok in enumerate(toks)}


def vocab_unknown_id(vocab):
    if "<unk>" in vocab:
        return int(vocab["<unk>"])
    nonpad = sorted(int(v) for v in vocab.values() if int(v) != 0)
    return int(nonpad[0]) if nonpad else 0


def caption_ids(captions, vocab, max_len=64, device="cpu"):
    max_len = max(1, int(max_len))
    ids = torch.zeros((len(captions), max_len), dtype=torch.long, device=device)
    unk = vocab_unknown_id(vocab)
    for i, caption in enumerate(captions):
        row = [vocab.get(tok, unk) for tok in caption_tokens(caption)[:max_len]]
        if row:
            ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return ids


def _read_ppm(path):
    with open(path, "rb") as f:
        raw = f.read()
    pos = 0

    def token():
        nonlocal pos
        while pos < len(raw) and raw[pos] in b" \t\r\n":
            pos += 1
        if pos < len(raw) and raw[pos] == ord("#"):
            while pos < len(raw) and raw[pos] not in b"\r\n":
                pos += 1
            return token()
        start = pos
        while pos < len(raw) and raw[pos] not in b" \t\r\n":
            pos += 1
        return raw[start:pos]

    magic = token()
    if magic not in (b"P6", b"P3"):
        raise ValueError(f"unsupported PPM magic {magic!r}")
    width = int(token())
    height = int(token())
    maxval = int(token())
    if maxval <= 0 or maxval > 255:
        raise ValueError("only 8-bit PPM files are supported")
    if magic == b"P6":
        if pos < len(raw) and raw[pos] in b" \t\r\n":
            pos += 1
        data = np.frombuffer(raw[pos:pos + width * height * 3], dtype=np.uint8)
    else:
        vals = [int(token()) for _ in range(width * height * 3)]
        data = np.asarray(vals, dtype=np.uint8)
    return data.reshape(height, width, 3)


def _ppm_dimensions(path):
    with open(path, "rb") as f:
        raw = f.read(4096)
    pos = 0

    def token():
        nonlocal pos
        while pos < len(raw) and raw[pos] in b" \t\r\n":
            pos += 1
        while pos < len(raw) and raw[pos] == ord("#"):
            while pos < len(raw) and raw[pos] not in b"\r\n":
                pos += 1
            while pos < len(raw) and raw[pos] in b" \t\r\n":
                pos += 1
        start = pos
        while pos < len(raw) and raw[pos] not in b" \t\r\n":
            pos += 1
        return raw[start:pos]

    magic = token()
    if magic not in (b"P6", b"P3"):
        raise ValueError(f"unsupported PPM magic {magic!r}")
    width = int(token())
    height = int(token())
    maxval = int(token())
    if maxval <= 0 or maxval > 255:
        raise ValueError("only 8-bit PPM files are supported")
    return width, height


def _png_dimensions_from_bytes(raw):
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise ValueError("invalid PNG header")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width <= 0 or height <= 0:
        raise ValueError("invalid PNG dimensions")
    return width, height


def _jpeg_dimensions_from_bytes(raw):
    if len(raw) < 4 or raw[:2] != b"\xff\xd8":
        raise ValueError("invalid JPEG header")
    i = 2
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while i + 3 < len(raw):
        while i < len(raw) and raw[i] != 0xFF:
            i += 1
        while i < len(raw) and raw[i] == 0xFF:
            i += 1
        if i >= len(raw):
            break
        marker = raw[i]
        i += 1
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > len(raw):
            break
        seg_len = int.from_bytes(raw[i:i + 2], "big")
        if seg_len < 2 or i + seg_len > len(raw):
            break
        if marker in sof_markers:
            if seg_len < 7:
                raise ValueError("invalid JPEG SOF segment")
            height = int.from_bytes(raw[i + 3:i + 5], "big")
            width = int.from_bytes(raw[i + 5:i + 7], "big")
            if width <= 0 or height <= 0:
                raise ValueError("invalid JPEG dimensions")
            return width, height
        i += seg_len
    raise ValueError("JPEG dimensions not found")


def _webp_dimensions_from_bytes(raw):
    if len(raw) < 20 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
        raise ValueError("invalid WebP header")
    chunk = raw[12:16]
    data = raw[20:]
    if chunk == b"VP8X":
        if len(data) < 10:
            raise ValueError("truncated WebP VP8X header")
        width = 1 + int.from_bytes(data[4:7], "little")
        height = 1 + int.from_bytes(data[7:10], "little")
    elif chunk == b"VP8 ":
        if len(data) < 10 or data[3:6] != b"\x9d\x01\x2a":
            raise ValueError("invalid WebP VP8 header")
        width = int.from_bytes(data[6:8], "little") & 0x3FFF
        height = int.from_bytes(data[8:10], "little") & 0x3FFF
    elif chunk == b"VP8L":
        if len(data) < 5 or data[0] != 0x2F:
            raise ValueError("invalid WebP VP8L header")
        bits = int.from_bytes(data[1:5], "little")
        width = 1 + (bits & 0x3FFF)
        height = 1 + ((bits >> 14) & 0x3FFF)
    else:
        raise ValueError(f"unsupported WebP chunk {chunk!r}")
    if width <= 0 or height <= 0:
        raise ValueError("invalid WebP dimensions")
    return width, height


def _bmp_dimensions_from_bytes(raw):
    if len(raw) < 26 or raw[:2] != b"BM":
        raise ValueError("invalid BMP header")
    dib_size = int.from_bytes(raw[14:18], "little")
    if dib_size == 12:
        width = int.from_bytes(raw[18:20], "little")
        height = int.from_bytes(raw[20:22], "little")
    else:
        width = int.from_bytes(raw[18:22], "little", signed=True)
        height = int.from_bytes(raw[22:26], "little", signed=True)
    width, height = abs(int(width)), abs(int(height))
    if width <= 0 or height <= 0:
        raise ValueError("invalid BMP dimensions")
    return width, height


def image_dimensions_from_bytes(raw, ext=""):
    ext = str(ext).lower()
    if ext in (".png", "") and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return _png_dimensions_from_bytes(raw)
    if ext in (".jpg", ".jpeg", "") and raw[:2] == b"\xff\xd8":
        return _jpeg_dimensions_from_bytes(raw)
    if ext in (".webp", "") and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return _webp_dimensions_from_bytes(raw)
    if ext in (".bmp", "") and raw[:2] == b"BM":
        return _bmp_dimensions_from_bytes(raw)
    raise ValueError(f"unsupported image header for extension {ext!r}")


def _pil_image(path):
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover - depends on optional runtime package.
        raise ImportError(
            "JPEG/PNG/WebP image loading requires Pillow. Install it with `pip install pillow` "
            "or use PPM fixtures for dependency-free tests."
        ) from e
    return Image.open(path).convert("RGB")


def image_dimensions(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".ppm", ".pnm"):
        return _ppm_dimensions(path)
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        with open(path, "rb") as f:
            raw = f.read()
        try:
            return image_dimensions_from_bytes(raw, ext=ext)
        except Exception:
            pass
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover - depends on optional runtime package.
        raise ImportError(
            "JPEG/PNG/WebP dimension checks require Pillow. Install it with `pip install pillow` "
            "or pass --no-check-images for manifest-only validation."
        ) from e
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def _target_hw(size):
    if not size:
        return None
    if isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError("image size tuple must be (height, width)")
        h, w = int(size[0]), int(size[1])
    else:
        h = w = int(size)
    if h <= 0 or w <= 0:
        raise ValueError("image size must be positive")
    return h, w


def load_image_tensor(path, size=256, device="cpu", center_crop=True,
                      crop_mode=None, hflip=False, rng=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".ppm", ".pnm"):
        arr = _read_ppm(path)
        x = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
    else:
        im = _pil_image(path)
        arr = np.asarray(im, dtype=np.uint8)
        x = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
    if crop_mode is None:
        crop_mode = "center" if center_crop else "none"
    crop_mode = str(crop_mode)
    if crop_mode not in ("center", "random", "none", "pad"):
        raise ValueError(f"unknown crop mode {crop_mode!r}")
    target_hw = _target_hw(size)
    if crop_mode in ("center", "random"):
        _c, h, w = x.shape
        side = min(h, w)
        if crop_mode == "random" and (h > side or w > side):
            if rng is None:
                rng = np.random.default_rng()
            y0 = int(rng.integers(0, h - side + 1)) if h > side else 0
            x0 = int(rng.integers(0, w - side + 1)) if w > side else 0
        else:
            y0 = (h - side) // 2
            x0 = (w - side) // 2
        x = x[:, y0:y0 + side, x0:x0 + side]
    elif crop_mode == "pad" and target_hw is not None:
        _c, h, w = x.shape
        target_h, target_w = target_hw
        scale = min(target_h / float(h), target_w / float(w))
        resize_h = max(1, int(round(h * scale)))
        resize_w = max(1, int(round(w * scale)))
        x = F.interpolate(x[None], size=(resize_h, resize_w), mode="bilinear",
                          align_corners=False)[0]
        pad_h = max(0, target_h - resize_h)
        pad_w = max(0, target_w - resize_w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        target_hw = None
    if hflip:
        x = torch.flip(x, dims=(2,))
    if target_hw is not None:
        x = F.interpolate(x[None], size=target_hw, mode="bilinear",
                          align_corners=False)[0]
    return x.to(device=device)


def _manifest_image_path(path, root=""):
    if not root:
        return path
    abs_root = os.path.abspath(root)
    abs_path = os.path.abspath(path)
    try:
        rel = os.path.relpath(abs_path, abs_root)
    except ValueError:
        return path
    if rel.startswith("..") or os.path.isabs(rel):
        return path
    return rel


def _record_to_manifest_row(rec: ImageTextRecord, root=""):
    row = {
        "image": _manifest_image_path(rec.path, root=root),
        "caption": rec.caption,
        "split": rec.split,
    }
    if rec.source:
        row["source"] = rec.source
    if rec.aesthetic is not None:
        row["aesthetic"] = float(rec.aesthetic)
    if rec.nsfw is not None:
        row["nsfw"] = float(rec.nsfw)
    if rec.watermark is not None:
        row["watermark"] = float(rec.watermark)
    if rec.width:
        row["width"] = int(rec.width)
    if rec.height:
        row["height"] = int(rec.height)
    if rec.text_embedding is not None:
        row["text_embedding"] = [float(x) for x in rec.text_embedding]
    if rec.text_embedding_sequence is not None:
        row["text_embedding_sequence"] = [
            [float(x) for x in seq_row] for seq_row in rec.text_embedding_sequence
        ]
    if rec.image_embedding is not None:
        row["image_embedding"] = [float(x) for x in rec.image_embedding]
    if rec.image_embedding_sequence is not None:
        row["image_embedding_sequence"] = [
            [float(x) for x in seq_row] for seq_row in rec.image_embedding_sequence
        ]
    return row


def inspect_image_manifest(path, root="", split="", min_aesthetic=None, max_records=0,
                           check_images=True, min_side=0, max_aspect=0.0,
                           min_caption_tokens=1, max_caption_tokens=0,
                           dedupe_paths=True, sample_errors=8,
                           max_nsfw=None, max_watermark=None,
                           min_caption_unique_ratio=0.0,
                           max_caption_token_frequency=0.0,
                           max_caption_token_run=0.0,
                           max_caption_char_run=0.0):
    """Validate and summarize a captioned-image manifest.

    This is data-plane tooling for real image generation: it catches missing files, corrupt
    images, duplicate paths, very short/long captions, low-resolution images, and extreme aspect
    ratios before a costly GPU run.
    """
    manifest_dir = os.path.dirname(os.path.abspath(path))
    rows = _read_manifest_rows(path)
    kept, rejected = [], []
    reject_causes = Counter()
    extension_counts = Counter()
    seen_kept_paths = set()
    seen_any_paths = Counter()
    skipped_split = 0
    skipped_aesthetic = 0
    skipped_nsfw = 0
    skipped_watermark = 0
    inspected = 0

    for row_index, row in enumerate(rows):
        try:
            rec = _coerce_record(row, manifest_dir=manifest_dir, root=root)
        except Exception as e:
            reason = "malformed_row"
            reject_causes[reason] += 1
            rejected.append({"row": row_index, "path": "", "caption": "",
                             "reasons": [reason], "error": str(e)})
            continue
        if split and rec.split != split:
            skipped_split += 1
            continue
        if min_aesthetic is not None and rec.aesthetic is not None:
            if rec.aesthetic < float(min_aesthetic):
                skipped_aesthetic += 1
                continue
        if max_nsfw is not None and rec.nsfw is not None:
            if rec.nsfw > float(max_nsfw):
                skipped_nsfw += 1
                reject_causes["nsfw_above_threshold"] += 1
                continue
        if max_watermark is not None and rec.watermark is not None:
            if rec.watermark > float(max_watermark):
                skipped_watermark += 1
                reject_causes["watermark_above_threshold"] += 1
                continue
        inspected += 1
        reasons = []
        caption_quality = caption_quality_metrics(rec.caption)
        toks = caption_tokens(rec.caption)
        if min_caption_tokens and len(toks) < int(min_caption_tokens):
            reasons.append("caption_too_short")
        if max_caption_tokens and len(toks) > int(max_caption_tokens):
            reasons.append("caption_too_long")
        if (min_caption_unique_ratio
                and caption_quality["caption_unique_ratio"]
                < float(min_caption_unique_ratio)):
            reasons.append("caption_unique_ratio_below_threshold")
        if (max_caption_token_frequency
                and caption_quality["caption_tokens"] > 1
                and caption_quality["caption_max_token_frequency"]
                > float(max_caption_token_frequency)):
            reasons.append("caption_token_frequency_above_threshold")
        if (max_caption_token_run
                and caption_quality["caption_tokens"] > 1
                and caption_quality["caption_max_token_run"]
                > float(max_caption_token_run)):
            reasons.append("caption_token_run_above_threshold")
        if (max_caption_char_run
                and caption_quality["caption_chars"] > 1
                and caption_quality["caption_max_char_run"]
                > float(max_caption_char_run)):
            reasons.append("caption_char_run_above_threshold")

        ext = os.path.splitext(rec.path)[1].lower() or "<none>"
        extension_counts[ext] += 1
        seen_any_paths[rec.path] += 1
        if ext not in IMAGE_EXTS:
            reasons.append("unsupported_extension")

        width, height = int(rec.width or 0), int(rec.height or 0)
        if not os.path.exists(rec.path):
            reasons.append("missing_file")
        elif check_images:
            try:
                width, height = image_dimensions(rec.path)
                rec = replace(rec, width=width, height=height)
            except Exception as e:
                reasons.append("invalid_image")
                rejected.append({"row": row_index, "path": rec.path, "caption": rec.caption,
                                 "reasons": list(reasons), "error": str(e)})
                for reason in reasons:
                    reject_causes[reason] += 1
                if max_records and inspected >= int(max_records):
                    break
                continue

        if width > 0 and height > 0:
            side = min(width, height)
            aspect = max(width, height) / max(1.0, float(side))
            if min_side and side < int(min_side):
                reasons.append("low_resolution")
            if max_aspect and aspect > float(max_aspect):
                reasons.append("extreme_aspect")

        if dedupe_paths and rec.path in seen_kept_paths:
            reasons.append("duplicate_path")

        if reasons:
            rejected.append({"row": row_index, "path": rec.path, "caption": rec.caption,
                             "reasons": reasons})
            for reason in reasons:
                reject_causes[reason] += 1
        else:
            kept.append(rec)
            seen_kept_paths.add(rec.path)
        if max_records and inspected >= int(max_records):
            break

    caption_quality = [caption_quality_metrics(rec.caption) for rec in kept]
    caption_lengths = [q["caption_tokens"] for q in caption_quality]
    caption_unique_ratios = [q["caption_unique_ratio"] for q in caption_quality]
    caption_max_token_frequencies = [
        q["caption_max_token_frequency"] for q in caption_quality
    ]
    caption_max_token_runs = [q["caption_max_token_run"] for q in caption_quality]
    caption_max_char_runs = [q["caption_max_char_run"] for q in caption_quality]
    widths = [rec.width for rec in kept if rec.width]
    heights = [rec.height for rec in kept if rec.height]
    min_sides = [min(rec.width, rec.height) for rec in kept if rec.width and rec.height]
    aspects = [
        max(rec.width, rec.height) / max(1.0, float(min(rec.width, rec.height)))
        for rec in kept if rec.width and rec.height
    ]
    aesthetic = [rec.aesthetic for rec in kept if rec.aesthetic is not None]
    nsfw = [rec.nsfw for rec in kept if rec.nsfw is not None]
    watermark = [rec.watermark for rec in kept if rec.watermark is not None]
    duplicate_total = sum(max(0, n - 1) for n in seen_any_paths.values())
    report = {
        "manifest": path,
        "root": root,
        "split_filter": split,
        "min_aesthetic": float(min_aesthetic) if min_aesthetic is not None else None,
        "max_nsfw": float(max_nsfw) if max_nsfw is not None else None,
        "max_watermark": float(max_watermark) if max_watermark is not None else None,
        "max_records": int(max_records),
        "check_images": bool(check_images),
        "min_side": int(min_side),
        "max_aspect": float(max_aspect),
        "min_caption_tokens": int(min_caption_tokens),
        "max_caption_tokens": int(max_caption_tokens),
        "min_caption_unique_ratio": float(min_caption_unique_ratio),
        "max_caption_token_frequency": float(max_caption_token_frequency),
        "max_caption_token_run": float(max_caption_token_run),
        "max_caption_char_run": float(max_caption_char_run),
        "dedupe_paths": bool(dedupe_paths),
        "rows_total": len(rows),
        "records_inspected": int(inspected),
        "records_kept": len(kept),
        "records_rejected": len(rejected),
        "records_skipped_split": int(skipped_split),
        "records_skipped_aesthetic": int(skipped_aesthetic),
        "records_skipped_nsfw": int(skipped_nsfw),
        "records_skipped_watermark": int(skipped_watermark),
        "duplicate_path_rows": int(duplicate_total),
        "reject_causes": dict(sorted(reject_causes.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "quality_pass_rate": float(len(kept) / inspected) if inspected else 0.0,
        "caption_token_stats": _stats(caption_lengths),
        "caption_unique_ratio_stats": _stats(caption_unique_ratios),
        "caption_max_token_frequency_stats": _stats(caption_max_token_frequencies),
        "caption_max_token_run_stats": _stats(caption_max_token_runs),
        "caption_max_char_run_stats": _stats(caption_max_char_runs),
        "width_stats": _stats(widths),
        "height_stats": _stats(heights),
        "min_side_stats": _stats(min_sides),
        "aspect_stats": _stats(aspects),
        "aesthetic_stats": _stats(aesthetic),
        "nsfw_stats": _stats(nsfw),
        "watermark_stats": _stats(watermark),
        "kept_summary": summarize_records(kept),
        "error_examples": rejected[:max(0, int(sample_errors))],
    }
    return report, kept, rejected


def write_image_manifest(records, path, root=""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_record_to_manifest_row(rec, root=root), sort_keys=True) + "\n")


def normalized_sampling_weights(weights, n):
    if weights is None:
        return None
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (int(n),):
        raise ValueError(f"sampling weights must have shape ({int(n)},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("sampling weights must be finite")
    if np.any(arr < 0.0):
        raise ValueError("sampling weights must be non-negative")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("sampling weights must have positive sum")
    return arr / total


def sample_image_text_batch(records, rng, batch=32, size=256, device="cpu",
                            return_records=False, crop_mode="center", hflip_prob=0.0,
                            weights=None):
    records = list(records)
    probs = normalized_sampling_weights(weights, len(records))
    if probs is None:
        idx = rng.integers(0, len(records), size=int(batch))
    else:
        idx = rng.choice(len(records), size=int(batch), replace=True, p=probs)
    chosen = [records[int(i)] for i in idx]
    hflip_prob = float(hflip_prob)
    imgs = [
        load_image_tensor(
            rec.path, size=size, device=device, crop_mode=crop_mode,
            hflip=(hflip_prob > 0.0 and float(rng.random()) < hflip_prob), rng=rng)
        for rec in chosen
    ]
    captions = [rec.caption for rec in chosen]
    if return_records:
        return torch.stack(imgs, dim=0), captions, chosen
    return torch.stack(imgs, dim=0), captions


def selftest():
    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "sample.ppm")
        arr = np.zeros((6, 8, 3), dtype=np.uint8)
        arr[:, :4, 0] = 255
        arr[:, 4:, 1] = 255
        with open(img, "wb") as f:
            f.write(b"P6\n8 6\n255\n")
            f.write(arr.tobytes())
        manifest = os.path.join(td, "manifest.jsonl")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "image": "sample.ppm",
                "caption": "red and green blocks",
                "split": "train",
                "source": "fixture",
                "text_embedding": [0.1, 0.2, 0.3],
                "text_embedding_sequence": [[0.1, 0.0], [0.0, 0.2]],
                "image_embedding": [0.4, 0.5, 0.6, 0.7],
                "image_embedding_sequence": [[0.4, 0.5], [0.6, 0.7], [0.8, 0.9]],
                "aesthetic": 7.5,
                "nsfw": 0.1,
                "watermark": 0.0,
            }) + "\n")
        records = read_image_manifest(manifest)
        assert len(records) == 1 and records[0].caption.startswith("red")
        assert records[0].source == "fixture"
        assert records[0].nsfw == 0.1 and records[0].watermark == 0.0
        assert records[0].text_embedding == (0.1, 0.2, 0.3)
        assert records[0].text_embedding_sequence == ((0.1, 0.0), (0.0, 0.2))
        assert records[0].image_embedding == (0.4, 0.5, 0.6, 0.7)
        assert records[0].image_embedding_sequence == (
            (0.4, 0.5), (0.6, 0.7), (0.8, 0.9))
        x = load_image_tensor(records[0].path, size=4)
        assert x.shape == (3, 4, 4) and float(x.max()) <= 1.0 and float(x.min()) >= -1.0
        xr = load_image_tensor(records[0].path, size=4, crop_mode="random",
                               rng=np.random.default_rng(2))
        xf = load_image_tensor(records[0].path, size=4, crop_mode="center", hflip=True)
        assert xr.shape == (3, 4, 4) and xf.shape == (3, 4, 4)
        assert torch.allclose(x[:, :, 0], xf[:, :, -1])
        xp = load_image_tensor(records[0].path, size=8, crop_mode="pad")
        assert xp.shape == (3, 8, 8)
        assert torch.allclose(xp[:, 0, :], torch.zeros_like(xp[:, 0, :]))
        assert torch.allclose(xp[:, -1, :], torch.zeros_like(xp[:, -1, :]))
        xt = load_image_tensor(records[0].path, size=(6, 10), crop_mode="pad")
        assert xt.shape == (3, 6, 10)
        vocab = build_caption_vocab(records)
        ids = caption_ids([records[0].caption], vocab, max_len=5)
        assert ids.shape == (1, 5) and int(ids[0, 0]) > 0
        repeated_metrics = caption_quality_metrics("echo echo echo echo")
        assert repeated_metrics["caption_unique_ratio"] == 0.25
        assert repeated_metrics["caption_max_token_frequency"] == 1.0
        assert repeated_metrics["caption_max_token_run"] == 1.0
        assert caption_quality_metrics("aaaaaaaaaa")["caption_max_char_run"] == 1.0
        xb, captions = sample_image_text_batch(
            records, np.random.default_rng(0), batch=2, size=4,
            crop_mode="random", hflip_prob=0.5)
        assert xb.shape == (2, 3, 4, 4) and captions == [records[0].caption] * 2
        xb_pad, captions_pad = sample_image_text_batch(
            records, np.random.default_rng(1), batch=2, size=8,
            crop_mode="pad")
        assert xb_pad.shape == (2, 3, 8, 8) and captions_pad == captions
        weighted_records = [records[0], replace(records[0], caption="weighted target")]
        _xw, weighted_captions = sample_image_text_batch(
            weighted_records, np.random.default_rng(2), batch=8, size=4,
            weights=[0.0, 1.0])
        assert weighted_captions == ["weighted target"] * 8
        summary = summarize_records(records)
        assert summary["image_records"] == 1 and summary["image_splits"]["train"] == 1
        assert summary["image_sources"] == {"fixture": 1}
        assert summary["text_embedding_records"] == 1 and summary["text_embedding_dims"] == [3]
        assert summary["text_embedding_sequence_records"] == 1
        assert summary["text_embedding_sequence_dims"] == [2]
        assert summary["image_embedding_records"] == 1 and summary["image_embedding_dims"] == [4]
        assert summary["image_embedding_sequence_records"] == 1
        assert summary["image_embedding_sequence_dims"] == [2]
        clean_manifest = os.path.join(td, "clean_manifest.jsonl")
        with open(clean_manifest, "w", encoding="utf-8") as f:
            for row in (
                    {"image": "sample.ppm", "caption": "red green blocks",
                     "split": "train", "source": "fixture", "text_embedding": [1.0, 0.0],
                     "text_embedding_sequence": [[1.0, 0.0], [0.0, 1.0]],
                     "image_embedding": [0.0, 1.0],
                     "image_embedding_sequence": [[0.0, 1.0], [1.0, 0.0]],
                     "nsfw": 0.0, "watermark": 0.0},
                    {"image": "sample.ppm", "caption": "duplicate patch", "split": "train"},
                    {"image": "missing.ppm", "caption": "x", "split": "train"},
                    {"image": "sample.ppm", "caption": "echo echo echo echo",
                     "split": "train"},
                    {"image": "sample.ppm", "caption": "held out patch", "split": "eval"}):
                f.write(json.dumps(row) + "\n")
        report, kept, rejected = inspect_image_manifest(
            clean_manifest, split="train", min_caption_tokens=2, check_images=True)
        assert report["records_kept"] == 1 and len(kept) == 1
        assert report["reject_causes"]["duplicate_path"] == 2
        assert report["reject_causes"]["missing_file"] == 1
        assert report["reject_causes"]["caption_too_short"] == 1
        assert report["records_skipped_split"] == 1 and rejected
        assert report["width_stats"]["min"] == 8.0 and report["height_stats"]["min"] == 6.0
        quality_report, quality_kept, quality_rejected = inspect_image_manifest(
            clean_manifest, split="train", min_caption_tokens=1, check_images=True,
            dedupe_paths=False, min_caption_unique_ratio=0.5,
            max_caption_token_frequency=0.6, max_caption_token_run=0.6)
        assert len(quality_kept) == 2 and quality_rejected
        assert (quality_report["reject_causes"][
            "caption_unique_ratio_below_threshold"] == 1)
        assert (quality_report["reject_causes"][
            "caption_token_frequency_above_threshold"] == 1)
        assert quality_report["reject_causes"]["caption_token_run_above_threshold"] == 1
        filtered = os.path.join(td, "filtered.jsonl")
        write_image_manifest(kept, filtered, root=td)
        with open(filtered, "r", encoding="utf-8") as f:
            filtered_row = json.loads(f.readline())
        assert filtered_row["image"] == "sample.ppm"
        assert filtered_row["source"] == "fixture"
        assert filtered_row["nsfw"] == 0.0 and filtered_row["watermark"] == 0.0
        assert filtered_row["text_embedding"] == [1.0, 0.0]
        assert filtered_row["text_embedding_sequence"] == [[1.0, 0.0], [0.0, 1.0]]
        assert filtered_row["image_embedding"] == [0.0, 1.0]
        sidecar = os.path.join(td, "embeddings.jsonl")
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "image": "sample.ppm",
                "text_embedding": [9.0, 8.0],
                "text_embedding_sequence": [[9.0, 0.0], [0.0, 8.0], [1.0, 1.0]],
                "image_embedding": [7.0, 6.0, 5.0],
                "image_embedding_sequence": [[7.0, 6.0], [5.0, 4.0]],
            }) + "\n")
        preserved, preserved_report = merge_embedding_sidecar(
            kept, sidecar, root=td, key="image", overwrite=False)
        assert preserved_report["embedding_records_matched"] == 1
        assert preserved_report["embedding_text_preserved"] == 1
        assert preserved_report["embedding_text_sequence_preserved"] == 1
        assert preserved_report["embedding_image_preserved"] == 1
        assert preserved_report["embedding_image_sequence_preserved"] == 1
        assert preserved[0].text_embedding == (1.0, 0.0)
        assert preserved[0].text_embedding_sequence == ((1.0, 0.0), (0.0, 1.0))
        overwritten, overwrite_report = merge_embedding_sidecar(
            kept, sidecar, root=td, key="image", overwrite=True)
        assert overwrite_report["embedding_text_written"] == 1
        assert overwrite_report["embedding_text_sequence_written"] == 1
        assert overwrite_report["embedding_text_sequence_dims"] == [2]
        assert overwrite_report["embedding_image_written"] == 1
        assert overwrite_report["embedding_image_sequence_written"] == 1
        assert overwrite_report["embedding_image_sequence_dims"] == [2]
        assert overwritten[0].text_embedding == (9.0, 8.0)
        assert overwritten[0].text_embedding_sequence == (
            (9.0, 0.0), (0.0, 8.0), (1.0, 1.0))
        assert overwritten[0].image_embedding == (7.0, 6.0, 5.0)
        assert overwritten[0].image_embedding_sequence == ((7.0, 6.0), (5.0, 4.0))
        reread = read_image_manifest(filtered, root=td, split="train")
        assert len(reread) == 1 and reread[0].width == 8 and reread[0].height == 6
        assert reread[0].source == "fixture"
        scored_manifest = os.path.join(td, "scored.jsonl")
        with open(scored_manifest, "w", encoding="utf-8") as f:
            for row in (
                    {"image": "sample.ppm", "caption": "safe image", "split": "train",
                     "image_nsfw": 0.1, "watermark_score": 0.0},
                    {"image": "sample.ppm", "caption": "unsafe image", "split": "train",
                     "image_nsfw": 0.9, "watermark_score": 0.0},
                    {"image": "sample.ppm", "caption": "watermarked image", "split": "train",
                     "image_nsfw": 0.1, "watermark_score": 0.8}):
                f.write(json.dumps(row) + "\n")
        score_report, score_kept, _score_rejected = inspect_image_manifest(
            scored_manifest, split="train", max_nsfw=0.5, max_watermark=0.5,
            dedupe_paths=False)
        assert score_report["records_kept"] == 1 and len(score_kept) == 1
        assert score_report["records_skipped_nsfw"] == 1
        assert score_report["records_skipped_watermark"] == 1
        assert score_report["nsfw_stats"]["max"] == 0.1
        aligned_records = [
            replace(records[0], caption="aligned", text_embedding=(1.0, 0.0),
                    image_embedding=(0.9, 0.1)),
            replace(records[0], caption="misaligned", text_embedding=(1.0, 0.0),
                    image_embedding=(0.0, 1.0)),
            replace(records[0], caption="missing", text_embedding=None,
                    text_embedding_sequence=None, image_embedding=(1.0, 0.0)),
        ]
        align_kept, align_report, align_rejected = filter_records_by_image_text_cosine(
            aligned_records, min_cosine=0.8)
        assert [r.caption for r in align_kept] == ["aligned"]
        assert len(align_rejected) == 2
        assert align_report["image_text_cosine_records"] == 2
        assert align_report["image_text_cosine_reject_causes"][
            "image_text_cosine_below_threshold"] == 1
        assert align_report["image_text_cosine_reject_causes"]["missing_text_embedding"] == 1
        dup_records = [
            replace(records[0], caption="low quality duplicate",
                    image_embedding=(1.0, 0.0), aesthetic=1.0),
            replace(records[0], caption="different image",
                    image_embedding=(0.0, 1.0), aesthetic=1.0),
            replace(records[0], caption="high quality duplicate",
                    image_embedding=(1.0, 0.0), aesthetic=9.0),
            replace(records[0], caption="missing embedding",
                    image_embedding=None, image_embedding_sequence=None),
        ]
        dedupe_kept, dedupe_report, dedupe_rejected = (
            filter_records_by_image_near_duplicates(
                dup_records, max_cosine=0.99, lsh_bits=8, lsh_tables=2, seed=0)
        )
        assert [r.caption for r in dedupe_kept] == [
            "different image", "high quality duplicate", "missing embedding"]
        assert len(dedupe_rejected) == 1
        assert dedupe_rejected[0]["caption"] == "low quality duplicate"
        assert dedupe_report["image_near_duplicate_records_rejected"] == 1
        assert dedupe_report["image_near_duplicate_invalid_causes"][
            "missing_image_embedding"] == 1
    print("image_data selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", default="", help="JSONL/CSV/TSV captioned image manifest")
    ap.add_argument("--root", default="", help="base directory for relative manifest paths")
    ap.add_argument("--split", default="",
                    help="optional split filter; default inspects all splits")
    ap.add_argument("--min-aesthetic", type=float, default=None,
                    help="skip rows with aesthetic/score/quality below this threshold")
    ap.add_argument("--max-nsfw", type=float, default=None,
                    help="skip rows with nsfw/image_nsfw/prompt_nsfw above this threshold")
    ap.add_argument("--max-records", type=int, default=0,
                    help="cap inspected records for smoke tests; 0 means all")
    ap.add_argument("--min-side", type=int, default=0,
                    help="reject images whose smaller side is below this size")
    ap.add_argument("--max-aspect", type=float, default=0.0,
                    help="reject images wider/taller than this aspect ratio; 0 disables")
    ap.add_argument("--max-watermark", type=float, default=None,
                    help="skip rows with watermark/watermark_score above this threshold")
    ap.add_argument("--min-caption-tokens", type=int, default=1,
                    help="reject captions shorter than this token count")
    ap.add_argument("--max-caption-tokens", type=int, default=0,
                    help="reject captions longer than this token count; 0 disables")
    ap.add_argument("--min-caption-unique-ratio", type=float, default=0.0,
                    help=("reject captions whose unique-token ratio is below this; "
                          "0 disables"))
    ap.add_argument("--max-caption-token-frequency", type=float, default=0.0,
                    help=("reject captions dominated by one token above this fraction; "
                          "0 disables"))
    ap.add_argument("--max-caption-token-run", type=float, default=0.0,
                    help=("reject captions with a consecutive repeated-token run above "
                          "this fraction; 0 disables"))
    ap.add_argument("--max-caption-char-run", type=float, default=0.0,
                    help=("reject captions with a repeated-character run above this "
                          "fraction; 0 disables"))
    ap.add_argument("--no-check-images", action="store_true",
                    help="skip image decode/header checks but still check paths and captions")
    ap.add_argument("--keep-duplicate-paths", action="store_true",
                    help="do not reject duplicate image paths")
    ap.add_argument("--sample-errors", type=int, default=8,
                    help="number of rejected examples to include in the report")
    ap.add_argument("--write-filtered", default="",
                    help="optional JSONL path for records that pass validation")
    ap.add_argument("--report-out", default="",
                    help="optional JSON path for the validation report")
    ap.add_argument("--embedding-manifest", default="",
                    help="optional sidecar manifest with text/image embeddings to merge")
    ap.add_argument("--embedding-root", default="",
                    help="base directory for relative paths in --embedding-manifest")
    ap.add_argument("--embedding-key", default="image",
                    choices=("image", "caption", "basename"),
                    help="join key for --embedding-manifest")
    ap.add_argument("--embedding-overwrite", action="store_true",
                    help="replace existing manifest embeddings with sidecar values")
    ap.add_argument("--min-image-text-cosine", type=float, default=None,
                    help=("after embedding merge, reject rows whose pooled text/image "
                          "embedding cosine is below this threshold"))
    ap.add_argument("--max-image-duplicate-cosine", type=float, default=None,
                    help=("after embedding merge, reject near-duplicate image embeddings "
                          "with cosine at or above this threshold"))
    ap.add_argument("--image-dedupe-lsh-bits", type=int, default=18,
                    help="random-hyperplane LSH bits per table for image duplicate filtering")
    ap.add_argument("--image-dedupe-lsh-tables", type=int, default=4,
                    help="number of LSH tables for image duplicate filtering")
    ap.add_argument("--image-dedupe-seed", type=int, default=0,
                    help="deterministic LSH seed for image duplicate filtering")
    ap.add_argument("--image-dedupe-keep-first", action="store_true",
                    help="keep first row among near duplicates instead of preferring quality score")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.manifest:
        ap.error("use --selftest or --manifest")
    if (args.min_image_text_cosine is not None
            and (args.min_image_text_cosine < -1.0 or args.min_image_text_cosine > 1.0)):
        ap.error("--min-image-text-cosine must be in [-1, 1]")
    if (args.max_image_duplicate_cosine is not None
            and (args.max_image_duplicate_cosine < -1.0
                 or args.max_image_duplicate_cosine > 1.0)):
        ap.error("--max-image-duplicate-cosine must be in [-1, 1]")
    for name in (
            "min_caption_unique_ratio",
            "max_caption_token_frequency",
            "max_caption_token_run",
            "max_caption_char_run"):
        val = float(getattr(args, name))
        if val < 0.0 or val > 1.0:
            ap.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.image_dedupe_lsh_bits <= 0:
        ap.error("--image-dedupe-lsh-bits must be positive")
    if args.image_dedupe_lsh_tables <= 0:
        ap.error("--image-dedupe-lsh-tables must be positive")
    report, kept, _rejected = inspect_image_manifest(
        args.manifest, root=args.root, split=args.split,
        min_aesthetic=args.min_aesthetic, max_records=args.max_records,
        max_nsfw=args.max_nsfw, max_watermark=args.max_watermark,
        check_images=not args.no_check_images, min_side=args.min_side,
        max_aspect=args.max_aspect, min_caption_tokens=args.min_caption_tokens,
        max_caption_tokens=args.max_caption_tokens,
        dedupe_paths=not args.keep_duplicate_paths,
        sample_errors=args.sample_errors,
        min_caption_unique_ratio=args.min_caption_unique_ratio,
        max_caption_token_frequency=args.max_caption_token_frequency,
        max_caption_token_run=args.max_caption_token_run,
        max_caption_char_run=args.max_caption_char_run)
    if args.embedding_manifest:
        kept, embedding_report = merge_embedding_sidecar(
            kept, args.embedding_manifest, root=args.embedding_root or args.root,
            key=args.embedding_key, overwrite=args.embedding_overwrite)
        report["embedding_merge"] = embedding_report
        report.update(summarize_records(kept))
    pre_embedding_filter_kept = len(kept)
    pre_embedding_filter_rejected = int(report["records_rejected"])
    kept_after_alignment, alignment_report, alignment_rejected = (
        filter_records_by_image_text_cosine(
            kept, min_cosine=args.min_image_text_cosine, sample_errors=args.sample_errors)
    )
    embedding_filter_rejected = []
    if args.min_image_text_cosine is not None:
        kept = kept_after_alignment
        embedding_filter_rejected.extend(alignment_rejected)
    kept_after_dedupe, dedupe_report, dedupe_rejected = (
        filter_records_by_image_near_duplicates(
            kept, max_cosine=args.max_image_duplicate_cosine,
            lsh_bits=args.image_dedupe_lsh_bits,
            lsh_tables=args.image_dedupe_lsh_tables,
            seed=args.image_dedupe_seed,
            prefer_quality=not args.image_dedupe_keep_first,
            sample_errors=args.sample_errors)
    )
    if args.max_image_duplicate_cosine is not None:
        kept = kept_after_dedupe
        embedding_filter_rejected.extend(dedupe_rejected)
    if args.min_image_text_cosine is not None or args.max_image_duplicate_cosine is not None:
        report["records_kept_before_embedding_filters"] = int(pre_embedding_filter_kept)
        report["records_rejected_before_embedding_filters"] = int(pre_embedding_filter_rejected)
        report["records_kept"] = len(kept)
        report["records_rejected"] = pre_embedding_filter_rejected + len(
            embedding_filter_rejected)
        inspected = max(1, int(report["records_inspected"]))
        report["quality_pass_rate"] = float(len(kept) / inspected)
        causes = Counter(report.get("reject_causes", {}))
        if args.min_image_text_cosine is not None:
            causes.update(alignment_report.get("image_text_cosine_reject_causes", {}))
        if args.max_image_duplicate_cosine is not None:
            causes.update(dedupe_report.get("image_near_duplicate_reject_causes", {}))
        report["reject_causes"] = dict(sorted(causes.items()))
        report["kept_summary"] = summarize_records(kept)
        report.update(summarize_records(kept))
    report["image_text_cosine_filter"] = alignment_report
    report["image_near_duplicate_filter"] = dedupe_report
    if args.write_filtered:
        write_image_manifest(kept, args.write_filtered, root=args.root)
        report["filtered_manifest"] = args.write_filtered
    text = json.dumps(report, indent=1)
    print(text)
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"saved -> {args.report_out}")


if __name__ == "__main__":
    main()
