"""Captioned image manifest utilities for real image-generation rungs.

The synthetic image rungs are useful for factor probes, but high-quality image generation needs
large image/text corpora.  This module is deliberately light on dependencies:

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
    aesthetic: float | None = None
    width: int = 0
    height: int = 0
    text_embedding: tuple[float, ...] | None = None
    image_embedding: tuple[float, ...] | None = None


def caption_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]


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


def _coerce_text_embedding(raw):
    return _coerce_float_embedding(raw)


def _coerce_image_embedding(raw):
    return _coerce_float_embedding(raw)


def _coerce_record(row, manifest_dir, root=""):
    image = row.get("image") or row.get("path") or row.get("file") or row.get("filepath")
    caption = row.get("caption") or row.get("text") or row.get("prompt")
    if not image or caption is None:
        raise ValueError("image manifest rows require an image/path/file and caption/text/prompt")
    base = root or manifest_dir
    path = str(image)
    if base and not os.path.isabs(path):
        path = os.path.join(base, path)
    aesthetic = row.get("aesthetic", row.get("score", row.get("quality")))
    if aesthetic in ("", None):
        aesthetic = None
    else:
        aesthetic = float(aesthetic)
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    text_embedding = _coerce_text_embedding(
        row.get("text_embedding",
                row.get("caption_embedding",
                        row.get("embedding", row.get("text_emb"))))
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
    return ImageTextRecord(
        path=os.path.normpath(path),
        caption=str(caption),
        split=str(row.get("split") or "train"),
        aesthetic=aesthetic,
        width=width,
        height=height,
        text_embedding=text_embedding,
        image_embedding=image_embedding,
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
        image_embedding = _coerce_image_embedding(
            row.get("image_embedding",
                    row.get("visual_embedding",
                            row.get("vision_embedding",
                                    row.get("clip_image_embedding",
                                            row.get("dino_embedding",
                                                    row.get("image_emb",
                                                            row.get("visual_emb")))))))
        )
        if text_embedding is None and image_embedding is None:
            skipped_no_embedding += 1
            continue
        if row_key in index:
            duplicate_keys += 1
            continue
        index[row_key] = (text_embedding, image_embedding)

    merged = []
    matched = missing = 0
    text_added = image_added = 0
    text_preserved = image_preserved = 0
    for rec in records:
        rec_key = _embedding_key_from_record(rec, key=key)
        vals = index.get(rec_key)
        if vals is None:
            missing += 1
            merged.append(rec)
            continue
        matched += 1
        text_embedding, image_embedding = vals
        new_text = rec.text_embedding
        new_image = rec.image_embedding
        if text_embedding is not None:
            if overwrite or rec.text_embedding is None:
                if rec.text_embedding != text_embedding:
                    text_added += 1
                new_text = text_embedding
            else:
                text_preserved += 1
        if image_embedding is not None:
            if overwrite or rec.image_embedding is None:
                if rec.image_embedding != image_embedding:
                    image_added += 1
                new_image = image_embedding
            else:
                image_preserved += 1
        merged.append(replace(rec, text_embedding=new_text, image_embedding=new_image))

    text_dims = sorted({len(v[0]) for v in index.values() if v[0] is not None})
    image_dims = sorted({len(v[1]) for v in index.values() if v[1] is not None})
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
        "embedding_image_written": int(image_added),
        "embedding_text_preserved": int(text_preserved),
        "embedding_image_preserved": int(image_preserved),
        "embedding_overwrite": bool(overwrite),
    }
    if text_dims:
        report["embedding_text_dims"] = text_dims
    if image_dims:
        report["embedding_image_dims"] = image_dims
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


def read_image_manifest(path, root="", split="train", min_aesthetic=None, max_records=0):
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
        records.append(rec)
        if max_records and len(records) >= int(max_records):
            break
    if not records:
        raise ValueError(f"image manifest {path!r} yielded no records for split={split!r}")
    return records


def summarize_records(records: Iterable[ImageTextRecord]):
    rows = list(records)
    splits = {}
    aesthetic = []
    caption_lens = []
    for rec in rows:
        splits[rec.split] = splits.get(rec.split, 0) + 1
        if rec.aesthetic is not None:
            aesthetic.append(float(rec.aesthetic))
        caption_lens.append(len(caption_tokens(rec.caption)))
    out = {
        "image_records": len(rows),
        "image_splits": splits,
        "caption_token_mean": float(np.mean(caption_lens)) if caption_lens else 0.0,
        "text_embedding_records": sum(1 for r in rows if r.text_embedding is not None),
        "image_embedding_records": sum(1 for r in rows if r.image_embedding is not None),
    }
    dims = sorted({len(r.text_embedding) for r in rows if r.text_embedding is not None})
    if dims:
        out["text_embedding_dims"] = dims
    image_dims = sorted({len(r.image_embedding) for r in rows if r.image_embedding is not None})
    if image_dims:
        out["image_embedding_dims"] = image_dims
    if aesthetic:
        out.update({
            "aesthetic_mean": float(np.mean(aesthetic)),
            "aesthetic_min": float(np.min(aesthetic)),
            "aesthetic_max": float(np.max(aesthetic)),
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


def caption_ids(captions, vocab, max_len=64, device="cpu"):
    max_len = max(1, int(max_len))
    ids = torch.zeros((len(captions), max_len), dtype=torch.long, device=device)
    unk = int(vocab.get("<unk>", 0))
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
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover - depends on optional runtime package.
        raise ImportError(
            "JPEG/PNG/WebP dimension checks require Pillow. Install it with `pip install pillow` "
            "or pass --no-check-images for manifest-only validation."
        ) from e
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def load_image_tensor(path, size=256, device="cpu", center_crop=True):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".ppm", ".pnm"):
        arr = _read_ppm(path)
        x = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
    else:
        im = _pil_image(path)
        arr = np.asarray(im, dtype=np.uint8)
        x = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
    if center_crop:
        _c, h, w = x.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        x = x[:, y0:y0 + side, x0:x0 + side]
    if size:
        x = F.interpolate(x[None], size=(int(size), int(size)), mode="bilinear",
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
    if rec.aesthetic is not None:
        row["aesthetic"] = float(rec.aesthetic)
    if rec.width:
        row["width"] = int(rec.width)
    if rec.height:
        row["height"] = int(rec.height)
    if rec.text_embedding is not None:
        row["text_embedding"] = [float(x) for x in rec.text_embedding]
    if rec.image_embedding is not None:
        row["image_embedding"] = [float(x) for x in rec.image_embedding]
    return row


def inspect_image_manifest(path, root="", split="", min_aesthetic=None, max_records=0,
                           check_images=True, min_side=0, max_aspect=0.0,
                           min_caption_tokens=1, max_caption_tokens=0,
                           dedupe_paths=True, sample_errors=8):
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
        inspected += 1
        reasons = []
        toks = caption_tokens(rec.caption)
        if min_caption_tokens and len(toks) < int(min_caption_tokens):
            reasons.append("caption_too_short")
        if max_caption_tokens and len(toks) > int(max_caption_tokens):
            reasons.append("caption_too_long")

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

    caption_lengths = [len(caption_tokens(rec.caption)) for rec in kept]
    widths = [rec.width for rec in kept if rec.width]
    heights = [rec.height for rec in kept if rec.height]
    min_sides = [min(rec.width, rec.height) for rec in kept if rec.width and rec.height]
    aspects = [
        max(rec.width, rec.height) / max(1.0, float(min(rec.width, rec.height)))
        for rec in kept if rec.width and rec.height
    ]
    aesthetic = [rec.aesthetic for rec in kept if rec.aesthetic is not None]
    duplicate_total = sum(max(0, n - 1) for n in seen_any_paths.values())
    report = {
        "manifest": path,
        "root": root,
        "split_filter": split,
        "min_aesthetic": float(min_aesthetic) if min_aesthetic is not None else None,
        "max_records": int(max_records),
        "check_images": bool(check_images),
        "min_side": int(min_side),
        "max_aspect": float(max_aspect),
        "min_caption_tokens": int(min_caption_tokens),
        "max_caption_tokens": int(max_caption_tokens),
        "dedupe_paths": bool(dedupe_paths),
        "rows_total": len(rows),
        "records_inspected": int(inspected),
        "records_kept": len(kept),
        "records_rejected": len(rejected),
        "records_skipped_split": int(skipped_split),
        "records_skipped_aesthetic": int(skipped_aesthetic),
        "duplicate_path_rows": int(duplicate_total),
        "reject_causes": dict(sorted(reject_causes.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "quality_pass_rate": float(len(kept) / inspected) if inspected else 0.0,
        "caption_token_stats": _stats(caption_lengths),
        "width_stats": _stats(widths),
        "height_stats": _stats(heights),
        "min_side_stats": _stats(min_sides),
        "aspect_stats": _stats(aspects),
        "aesthetic_stats": _stats(aesthetic),
        "kept_summary": summarize_records(kept),
        "error_examples": rejected[:max(0, int(sample_errors))],
    }
    return report, kept, rejected


def write_image_manifest(records, path, root=""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_record_to_manifest_row(rec, root=root), sort_keys=True) + "\n")


def sample_image_text_batch(records, rng, batch=32, size=256, device="cpu",
                            return_records=False):
    records = list(records)
    idx = rng.integers(0, len(records), size=int(batch))
    chosen = [records[int(i)] for i in idx]
    imgs = [load_image_tensor(rec.path, size=size, device=device) for rec in chosen]
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
                "text_embedding": [0.1, 0.2, 0.3],
                "image_embedding": [0.4, 0.5, 0.6, 0.7],
                "aesthetic": 7.5,
            }) + "\n")
        records = read_image_manifest(manifest)
        assert len(records) == 1 and records[0].caption.startswith("red")
        assert records[0].text_embedding == (0.1, 0.2, 0.3)
        assert records[0].image_embedding == (0.4, 0.5, 0.6, 0.7)
        x = load_image_tensor(records[0].path, size=4)
        assert x.shape == (3, 4, 4) and float(x.max()) <= 1.0 and float(x.min()) >= -1.0
        vocab = build_caption_vocab(records)
        ids = caption_ids([records[0].caption], vocab, max_len=5)
        assert ids.shape == (1, 5) and int(ids[0, 0]) > 0
        xb, captions = sample_image_text_batch(records, np.random.default_rng(0), batch=2, size=4)
        assert xb.shape == (2, 3, 4, 4) and captions == [records[0].caption] * 2
        summary = summarize_records(records)
        assert summary["image_records"] == 1 and summary["image_splits"]["train"] == 1
        assert summary["text_embedding_records"] == 1 and summary["text_embedding_dims"] == [3]
        assert summary["image_embedding_records"] == 1 and summary["image_embedding_dims"] == [4]
        qa_manifest = os.path.join(td, "qa.jsonl")
        with open(qa_manifest, "w", encoding="utf-8") as f:
            for row in (
                    {"image": "sample.ppm", "caption": "red green blocks",
                     "split": "train", "text_embedding": [1.0, 0.0],
                     "image_embedding": [0.0, 1.0]},
                    {"image": "sample.ppm", "caption": "duplicate patch", "split": "train"},
                    {"image": "missing.ppm", "caption": "x", "split": "train"},
                    {"image": "sample.ppm", "caption": "held out patch", "split": "eval"}):
                f.write(json.dumps(row) + "\n")
        report, kept, rejected = inspect_image_manifest(
            qa_manifest, split="train", min_caption_tokens=2, check_images=True)
        assert report["records_kept"] == 1 and len(kept) == 1
        assert report["reject_causes"]["duplicate_path"] == 1
        assert report["reject_causes"]["missing_file"] == 1
        assert report["reject_causes"]["caption_too_short"] == 1
        assert report["records_skipped_split"] == 1 and rejected
        assert report["width_stats"]["min"] == 8.0 and report["height_stats"]["min"] == 6.0
        filtered = os.path.join(td, "filtered.jsonl")
        write_image_manifest(kept, filtered, root=td)
        with open(filtered, "r", encoding="utf-8") as f:
            filtered_row = json.loads(f.readline())
        assert filtered_row["image"] == "sample.ppm"
        assert filtered_row["text_embedding"] == [1.0, 0.0]
        assert filtered_row["image_embedding"] == [0.0, 1.0]
        sidecar = os.path.join(td, "embeddings.jsonl")
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "image": "sample.ppm",
                "text_embedding": [9.0, 8.0],
                "image_embedding": [7.0, 6.0, 5.0],
            }) + "\n")
        preserved, preserved_report = merge_embedding_sidecar(
            kept, sidecar, root=td, key="image", overwrite=False)
        assert preserved_report["embedding_records_matched"] == 1
        assert preserved_report["embedding_text_preserved"] == 1
        assert preserved_report["embedding_image_preserved"] == 1
        assert preserved[0].text_embedding == (1.0, 0.0)
        overwritten, overwrite_report = merge_embedding_sidecar(
            kept, sidecar, root=td, key="image", overwrite=True)
        assert overwrite_report["embedding_text_written"] == 1
        assert overwrite_report["embedding_image_written"] == 1
        assert overwritten[0].text_embedding == (9.0, 8.0)
        assert overwritten[0].image_embedding == (7.0, 6.0, 5.0)
        reread = read_image_manifest(filtered, root=td, split="train")
        assert len(reread) == 1 and reread[0].width == 8 and reread[0].height == 6
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
    ap.add_argument("--max-records", type=int, default=0,
                    help="cap inspected records for smoke tests; 0 means all")
    ap.add_argument("--min-side", type=int, default=0,
                    help="reject images whose smaller side is below this size")
    ap.add_argument("--max-aspect", type=float, default=0.0,
                    help="reject images wider/taller than this aspect ratio; 0 disables")
    ap.add_argument("--min-caption-tokens", type=int, default=1,
                    help="reject captions shorter than this token count")
    ap.add_argument("--max-caption-tokens", type=int, default=0,
                    help="reject captions longer than this token count; 0 disables")
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
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.manifest:
        ap.error("use --selftest or --manifest")
    report, kept, _rejected = inspect_image_manifest(
        args.manifest, root=args.root, split=args.split,
        min_aesthetic=args.min_aesthetic, max_records=args.max_records,
        check_images=not args.no_check_images, min_side=args.min_side,
        max_aspect=args.max_aspect, min_caption_tokens=args.min_caption_tokens,
        max_caption_tokens=args.max_caption_tokens,
        dedupe_paths=not args.keep_duplicate_paths,
        sample_errors=args.sample_errors)
    if args.embedding_manifest:
        kept, embedding_report = merge_embedding_sidecar(
            kept, args.embedding_manifest, root=args.embedding_root or args.root,
            key=args.embedding_key, overwrite=args.embedding_overwrite)
        report["embedding_merge"] = embedding_report
        report.update(summarize_records(kept))
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
