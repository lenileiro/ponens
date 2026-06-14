"""Curate captioned image manifests before high-quality image generation training.

The generator should train on real image/text data that has already passed basic
quality, safety, caption, embedding-alignment, and duplicate filters. This module
keeps that as an explicit manifest-to-manifest step so fetch/caption/embed/score
pipelines can be audited before expensive GPU training.
"""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from dataclasses import replace
import hashlib
import io
import json
import os
import tempfile

import numpy as np

from .image_data import (
    ImageTextRecord,
    caption_quality_metrics,
    caption_tokens,
    filter_records_by_image_near_duplicates,
    filter_records_by_image_text_cosine,
    read_image_manifest,
    summarize_records,
)


def _rel_image_path(path, root=""):
    if not root:
        return path
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return path
    if rel.startswith("..") or os.path.isabs(rel):
        return path
    return rel


def _stable_unit_interval(text, seed=0):
    payload = f"{int(seed)}\0{text}".encode("utf-8")
    raw = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(raw, "little") / float(1 << 64)


def _split_key(rec, mode):
    if mode == "image":
        return rec.path
    if mode == "caption":
        return rec.caption
    if mode == "image_caption":
        return f"{rec.path}\0{rec.caption}"
    raise ValueError(f"unknown split key {mode!r}")


def _embedding_row(x):
    return [float(v) for v in x] if x is not None else None


def _embedding_matrix(x):
    if x is None:
        return None
    return [[float(v) for v in row] for row in x]


def record_to_manifest_row(rec: ImageTextRecord, root=""):
    row = {
        "image": _rel_image_path(rec.path, root=root),
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
    text_embedding = _embedding_row(rec.text_embedding)
    if text_embedding is not None:
        row["text_embedding"] = text_embedding
    text_embedding_sequence = _embedding_matrix(rec.text_embedding_sequence)
    if text_embedding_sequence is not None:
        row["text_embedding_sequence"] = text_embedding_sequence
    image_embedding = _embedding_row(rec.image_embedding)
    if image_embedding is not None:
        row["image_embedding"] = image_embedding
    image_embedding_sequence = _embedding_matrix(rec.image_embedding_sequence)
    if image_embedding_sequence is not None:
        row["image_embedding_sequence"] = image_embedding_sequence
    return row


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


def _record_quality(rec):
    if rec.aesthetic is not None:
        return float(rec.aesthetic)
    return 0.0


def _basic_filter(records, min_caption_tokens=1, max_caption_tokens=0,
                  min_width=0, min_height=0, min_pixels=0,
                  min_aspect=0.0, max_aspect=0.0,
                  min_caption_unique_ratio=0.0,
                  max_caption_token_frequency=0.0,
                  max_caption_token_run=0.0,
                  max_caption_char_run=0.0,
                  min_aesthetic=None, max_nsfw=None, max_watermark=None,
                  require_text_embedding=False, require_text_embedding_sequence=False,
                  require_image_embedding=False, require_image_embedding_sequence=False,
                  sample_errors=8):
    kept = []
    rejected = []
    causes = Counter()
    caption_lens = []
    caption_unique_ratios = []
    caption_max_token_frequencies = []
    caption_max_token_runs = []
    caption_max_char_runs = []
    aspect_ratios = []
    pixel_counts = []
    for idx, rec in enumerate(records):
        reasons = []
        quality = caption_quality_metrics(rec.caption)
        toks = caption_tokens(rec.caption)
        caption_lens.append(len(toks))
        caption_unique_ratios.append(quality["caption_unique_ratio"])
        caption_max_token_frequencies.append(quality["caption_max_token_frequency"])
        caption_max_token_runs.append(quality["caption_max_token_run"])
        caption_max_char_runs.append(quality["caption_max_char_run"])
        if len(toks) < int(min_caption_tokens):
            reasons.append("caption_too_short")
        if int(max_caption_tokens or 0) > 0 and len(toks) > int(max_caption_tokens):
            reasons.append("caption_too_long")
        if (min_caption_unique_ratio
                and quality["caption_unique_ratio"] < float(min_caption_unique_ratio)):
            reasons.append("caption_unique_ratio_below_threshold")
        if (max_caption_token_frequency
                and quality["caption_tokens"] > 1
                and quality["caption_max_token_frequency"]
                > float(max_caption_token_frequency)):
            reasons.append("caption_token_frequency_above_threshold")
        if (max_caption_token_run
                and quality["caption_tokens"] > 1
                and quality["caption_max_token_run"] > float(max_caption_token_run)):
            reasons.append("caption_token_run_above_threshold")
        if (max_caption_char_run
                and quality["caption_chars"] > 1
                and quality["caption_max_char_run"] > float(max_caption_char_run)):
            reasons.append("caption_char_run_above_threshold")
        if int(min_width or 0) > 0 and (not rec.width or rec.width < int(min_width)):
            reasons.append("width_too_small")
        if int(min_height or 0) > 0 and (not rec.height or rec.height < int(min_height)):
            reasons.append("height_too_small")
        if int(min_pixels or 0) > 0:
            pixels = int(rec.width or 0) * int(rec.height or 0)
            if pixels:
                pixel_counts.append(pixels)
            if pixels < int(min_pixels):
                reasons.append("pixels_too_small")
        if rec.width and rec.height:
            aspect = float(rec.width) / max(1.0, float(rec.height))
            aspect_ratios.append(aspect)
            if float(min_aspect or 0.0) > 0.0 and aspect < float(min_aspect):
                reasons.append("aspect_too_narrow")
            if float(max_aspect or 0.0) > 0.0 and aspect > float(max_aspect):
                reasons.append("aspect_too_wide")
        elif float(min_aspect or 0.0) > 0.0 or float(max_aspect or 0.0) > 0.0:
            reasons.append("missing_dimensions_for_aspect_filter")
        if min_aesthetic is not None:
            if rec.aesthetic is None or rec.aesthetic < float(min_aesthetic):
                reasons.append("aesthetic_below_threshold")
        if max_nsfw is not None:
            if rec.nsfw is not None and rec.nsfw > float(max_nsfw):
                reasons.append("nsfw_above_threshold")
        if max_watermark is not None:
            if rec.watermark is not None and rec.watermark > float(max_watermark):
                reasons.append("watermark_above_threshold")
        if require_text_embedding and rec.text_embedding is None:
            reasons.append("missing_text_embedding")
        if require_text_embedding_sequence and rec.text_embedding_sequence is None:
            reasons.append("missing_text_embedding_sequence")
        if require_image_embedding and rec.image_embedding is None:
            reasons.append("missing_image_embedding")
        if require_image_embedding_sequence and rec.image_embedding_sequence is None:
            reasons.append("missing_image_embedding_sequence")
        if reasons:
            causes.update(reasons)
            rejected.append({
                "row": int(idx),
                "path": rec.path,
                "caption": rec.caption,
                "reasons": reasons,
            })
            continue
        kept.append(rec)
    report = {
        "basic_filter_records_in": len(records),
        "basic_filter_records_kept": len(kept),
        "basic_filter_records_rejected": len(rejected),
        "basic_filter_reject_causes": dict(sorted(causes.items())),
        "basic_filter_error_examples": rejected[:max(0, int(sample_errors))],
        "caption_token_stats": _stats(caption_lens),
        "caption_unique_ratio_stats": _stats(caption_unique_ratios),
        "caption_max_token_frequency_stats": _stats(caption_max_token_frequencies),
        "caption_max_token_run_stats": _stats(caption_max_token_runs),
        "caption_max_char_run_stats": _stats(caption_max_char_runs),
        "aspect_ratio_stats": _stats(aspect_ratios),
        "pixel_count_stats": _stats(pixel_counts),
    }
    return kept, report, rejected


def _exact_dedupe(records, by=("image",), prefer_quality=True, sample_errors=8):
    by = tuple(x for x in by if x)
    if not by:
        return list(records), {
            "exact_dedupe_enabled": False,
            "exact_dedupe_records_kept": len(records),
            "exact_dedupe_records_rejected": 0,
        }, []
    order = list(range(len(records)))
    if prefer_quality:
        order.sort(key=lambda i: (-_record_quality(records[i]), i))
    seen = {}
    rejected = []
    causes = Counter()
    keep = set()
    for idx in order:
        rec = records[idx]
        parts = []
        if "image" in by:
            parts.append(os.path.normpath(rec.path))
        if "caption" in by:
            parts.append(" ".join(caption_tokens(rec.caption)))
        if not parts:
            keep.add(idx)
            continue
        key = "\0".join(parts)
        if key in seen:
            causes["exact_duplicate"] += 1
            rejected.append({
                "row": int(idx),
                "path": rec.path,
                "caption": rec.caption,
                "reasons": ["exact_duplicate"],
                "duplicate_of_row": int(seen[key]),
                "duplicate_of_path": records[seen[key]].path,
            })
            continue
        seen[key] = idx
        keep.add(idx)
    kept = [rec for i, rec in enumerate(records) if i in keep]
    report = {
        "exact_dedupe_enabled": True,
        "exact_dedupe_by": list(by),
        "exact_dedupe_prefer_quality": bool(prefer_quality),
        "exact_dedupe_records_kept": len(kept),
        "exact_dedupe_records_rejected": len(rejected),
        "exact_dedupe_reject_causes": dict(sorted(causes.items())),
        "exact_dedupe_error_examples": rejected[:max(0, int(sample_errors))],
    }
    return kept, report, rejected


def _assign_splits(records, eval_frac=None, seed=0, split_key="image_caption",
                   train_split="train", eval_split="eval"):
    if eval_frac is None:
        return list(records), {
            "split_rewrite_enabled": False,
            "split_counts": summarize_records(records).get("image_splits", {}),
        }
    frac = float(eval_frac)
    if frac < 0.0 or frac >= 1.0:
        raise ValueError("eval_frac must be in [0, 1)")
    out = []
    counts = Counter()
    for rec in records:
        score = _stable_unit_interval(_split_key(rec, split_key), seed=seed)
        split = eval_split if score < frac else train_split
        counts[split] += 1
        out.append(replace(rec, split=split))
    return out, {
        "split_rewrite_enabled": True,
        "eval_frac": float(frac),
        "split_seed": int(seed),
        "split_key": split_key,
        "train_split": train_split,
        "eval_split": eval_split,
        "split_counts": dict(sorted(counts.items())),
    }


def curate_manifest(
        manifest,
        root="",
        out="",
        split="",
        max_records=0,
        min_caption_tokens=1,
        max_caption_tokens=0,
        min_width=0,
        min_height=0,
        min_pixels=0,
        min_aspect=0.0,
        max_aspect=0.0,
        min_caption_unique_ratio=0.0,
        max_caption_token_frequency=0.0,
        max_caption_token_run=0.0,
        max_caption_char_run=0.0,
        min_aesthetic=None,
        max_nsfw=None,
        max_watermark=None,
        require_text_embedding=False,
        require_text_embedding_sequence=False,
        require_image_embedding=False,
        require_image_embedding_sequence=False,
        min_image_text_cosine=None,
        max_image_duplicate_cosine=None,
        exact_dedupe_by=("image",),
        dedupe_prefer_quality=True,
        near_dedupe_lsh_bits=18,
        near_dedupe_lsh_tables=4,
        seed=0,
        eval_frac=None,
        split_key="image_caption",
        sample_errors=8,
        report_out=""):
    records = read_image_manifest(
        manifest, root=root, split=split, max_records=max_records)
    original_count = len(records)
    report = {
        "experiment": "image_manifest_curation",
        "manifest": manifest,
        "root": root,
        "split": split,
        "max_records": int(max_records),
        "records_in": int(original_count),
        "min_caption_unique_ratio": float(min_caption_unique_ratio),
        "max_caption_token_frequency": float(max_caption_token_frequency),
        "max_caption_token_run": float(max_caption_token_run),
        "max_caption_char_run": float(max_caption_char_run),
        "input_summary": summarize_records(records),
    }
    records, basic_report, basic_rejected = _basic_filter(
        records,
        min_caption_tokens=min_caption_tokens,
        max_caption_tokens=max_caption_tokens,
        min_width=min_width,
        min_height=min_height,
        min_pixels=min_pixels,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        min_caption_unique_ratio=min_caption_unique_ratio,
        max_caption_token_frequency=max_caption_token_frequency,
        max_caption_token_run=max_caption_token_run,
        max_caption_char_run=max_caption_char_run,
        min_aesthetic=min_aesthetic,
        max_nsfw=max_nsfw,
        max_watermark=max_watermark,
        require_text_embedding=require_text_embedding,
        require_text_embedding_sequence=require_text_embedding_sequence,
        require_image_embedding=require_image_embedding,
        require_image_embedding_sequence=require_image_embedding_sequence,
        sample_errors=sample_errors)
    report.update(basic_report)
    all_rejected = list(basic_rejected)
    records, cosine_report, cosine_rejected = filter_records_by_image_text_cosine(
        records, min_cosine=min_image_text_cosine, sample_errors=sample_errors)
    report.update(cosine_report)
    all_rejected.extend(cosine_rejected)
    records, exact_report, exact_rejected = _exact_dedupe(
        records, by=exact_dedupe_by, prefer_quality=dedupe_prefer_quality,
        sample_errors=sample_errors)
    report.update(exact_report)
    all_rejected.extend(exact_rejected)
    records, near_report, near_rejected = filter_records_by_image_near_duplicates(
        records, max_cosine=max_image_duplicate_cosine,
        lsh_bits=near_dedupe_lsh_bits, lsh_tables=near_dedupe_lsh_tables,
        seed=seed, prefer_quality=dedupe_prefer_quality,
        sample_errors=sample_errors)
    report.update(near_report)
    all_rejected.extend(near_rejected)
    records, split_report = _assign_splits(
        records, eval_frac=eval_frac, seed=seed, split_key=split_key)
    report.update(split_report)
    report.update({
        "records_out": len(records),
        "records_rejected": len(all_rejected),
        "reject_examples": all_rejected[:max(0, int(sample_errors))],
        "output_summary": summarize_records(records) if records else {},
    })
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(record_to_manifest_row(rec, root=root),
                                   sort_keys=True) + "\n")
        report["out"] = out
    if report_out:
        os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        report["report_out"] = report_out
    return report


def _write_ppm(path, arr):
    with open(path, "wb") as f:
        f.write(f"P6\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii"))
        f.write(arr.astype(np.uint8).tobytes())


def selftest():
    with tempfile.TemporaryDirectory() as td:
        red = np.zeros((12, 12, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        blue = np.zeros((12, 12, 3), dtype=np.uint8)
        blue[:, :, 2] = 255
        _write_ppm(os.path.join(td, "red.ppm"), red)
        _write_ppm(os.path.join(td, "blue.ppm"), blue)
        manifest = os.path.join(td, "manifest.jsonl")
        rows = [
            {
                "image": "red.ppm",
                "caption": "red high quality block",
                "split": "train",
                "aesthetic": 0.9,
                "nsfw": 0.0,
                "watermark": 0.0,
                "width": 12,
                "height": 12,
                "text_embedding": [1.0, 0.0, 0.0],
                "image_embedding": [1.0, 0.0, 0.0],
            },
            {
                "image": "blue.ppm",
                "caption": "blue aligned sample",
                "split": "train",
                "aesthetic": 0.8,
                "nsfw": 0.0,
                "watermark": 0.0,
                "width": 12,
                "height": 12,
                "text_embedding": [0.0, 1.0, 0.0],
                "image_embedding": [0.0, 1.0, 0.0],
            },
            {
                "image": "red.ppm",
                "caption": "duplicate lower quality",
                "split": "train",
                "aesthetic": 0.1,
                "width": 12,
                "height": 12,
                "text_embedding": [1.0, 0.0, 0.0],
                "image_embedding": [1.0, 0.0, 0.0],
            },
            {
                "image": "blue.ppm",
                "caption": "misaligned unsafe sample",
                "split": "train",
                "aesthetic": 0.7,
                "nsfw": 0.9,
                "width": 12,
                "height": 12,
                "text_embedding": [1.0, 0.0, 0.0],
                "image_embedding": [0.0, 1.0, 0.0],
            },
            {
                "image": "blue.ppm",
                "caption": "echo echo echo echo",
                "split": "train",
                "aesthetic": 0.9,
                "nsfw": 0.0,
                "watermark": 0.0,
                "width": 12,
                "height": 12,
                "text_embedding": [0.0, 1.0, 0.0],
                "image_embedding": [0.0, 1.0, 0.0],
            },
        ]
        with open(manifest, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        out = os.path.join(td, "curated.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            report = main([
                "--manifest", manifest,
                "--root", td,
                "--out", out,
                "--min-caption-tokens", "2",
                "--min-aesthetic", "0.2",
                "--max-nsfw", "0.2",
                "--min-image-text-cosine", "0.5",
                "--exact-dedupe-by", "image",
                "--min-caption-unique-ratio", "0.5",
                "--max-caption-token-frequency", "0.6",
                "--max-caption-token-run", "0.6",
                "--eval-frac", "0.5",
                "--seed", "3",
            ])
        assert report["records_in"] == 5
        assert report["records_out"] == 2
        assert report["basic_filter_records_rejected"] == 3
        assert (report["basic_filter_reject_causes"][
            "caption_unique_ratio_below_threshold"] == 1)
        assert (report["basic_filter_reject_causes"][
            "caption_token_frequency_above_threshold"] == 1)
        assert report["basic_filter_reject_causes"]["caption_token_run_above_threshold"] == 1
        assert report["image_text_cosine_records_rejected"] == 0
        assert report["exact_dedupe_records_rejected"] == 0
        with open(out, "r", encoding="utf-8") as f:
            curated = [json.loads(line) for line in f if line.strip()]
        assert len(curated) == 2
        assert {row["image"] for row in curated} == {"red.ppm", "blue.ppm"}
        assert {row["split"] for row in curated}.issubset({"train", "eval"})
    print("image_curate selftest OK")


def _parse_dedupe_by(text):
    if not text:
        return ()
    vals = tuple(x.strip() for x in str(text).split(",") if x.strip())
    bad = sorted(set(vals) - {"image", "caption"})
    if bad:
        raise ValueError(f"unknown exact dedupe key(s): {','.join(bad)}")
    return vals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", default="", help="input image manifest")
    ap.add_argument("--root", default="", help="base directory for relative image paths")
    ap.add_argument("--split", default="", help="optional input split; default reads all")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--out", default="", help="curated output manifest JSONL")
    ap.add_argument("--report-out", default="", help="optional curation report JSON")
    ap.add_argument("--min-caption-tokens", type=int, default=1)
    ap.add_argument("--max-caption-tokens", type=int, default=0)
    ap.add_argument("--min-width", type=int, default=0)
    ap.add_argument("--min-height", type=int, default=0)
    ap.add_argument("--min-pixels", type=int, default=0)
    ap.add_argument("--min-aspect", type=float, default=0.0)
    ap.add_argument("--max-aspect", type=float, default=0.0)
    ap.add_argument("--min-caption-unique-ratio", type=float, default=0.0,
                    help="reject captions below this unique-token ratio; 0 disables")
    ap.add_argument("--max-caption-token-frequency", type=float, default=0.0,
                    help="reject captions dominated by one token above this fraction; 0 disables")
    ap.add_argument("--max-caption-token-run", type=float, default=0.0,
                    help="reject captions with consecutive repeated-token runs above this fraction")
    ap.add_argument("--max-caption-char-run", type=float, default=0.0,
                    help="reject captions with repeated-character runs above this fraction")
    ap.add_argument("--min-aesthetic", type=float, default=None)
    ap.add_argument("--max-nsfw", type=float, default=None)
    ap.add_argument("--max-watermark", type=float, default=None)
    ap.add_argument("--require-text-embedding", action="store_true")
    ap.add_argument("--require-text-embedding-sequence", action="store_true")
    ap.add_argument("--require-image-embedding", action="store_true")
    ap.add_argument("--require-image-embedding-sequence", action="store_true")
    ap.add_argument("--min-image-text-cosine", type=float, default=None,
                    help="drop rows whose normalized text/image embeddings are below this")
    ap.add_argument("--max-image-duplicate-cosine", type=float, default=None,
                    help="drop likely visual near-duplicates above this image cosine")
    ap.add_argument("--near-dedupe-lsh-bits", type=int, default=18)
    ap.add_argument("--near-dedupe-lsh-tables", type=int, default=4)
    ap.add_argument("--exact-dedupe-by", default="image",
                    help="comma-separated exact duplicate keys: image,caption, both, or empty")
    ap.add_argument("--no-dedupe-prefer-quality", action="store_true")
    ap.add_argument("--eval-frac", type=float, default=None,
                    help="rewrite splits with this deterministic eval fraction; omit to preserve")
    ap.add_argument("--split-key", default="image_caption",
                    choices=("image", "caption", "image_caption"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-errors", type=int, default=8)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return None
    if not args.manifest:
        ap.error("use --selftest or --manifest")
    if not args.out and not args.report_out:
        ap.error("provide --out and/or --report-out")
    if args.max_records < 0:
        ap.error("--max-records must be non-negative")
    if args.min_caption_tokens < 0 or args.max_caption_tokens < 0:
        ap.error("caption token limits must be non-negative")
    if args.max_caption_tokens and args.max_caption_tokens < args.min_caption_tokens:
        ap.error("--max-caption-tokens must be >= --min-caption-tokens")
    if args.min_width < 0 or args.min_height < 0 or args.min_pixels < 0:
        ap.error("dimension filters must be non-negative")
    if args.min_aspect < 0.0 or args.max_aspect < 0.0:
        ap.error("aspect filters must be non-negative")
    if args.max_aspect and args.min_aspect and args.max_aspect < args.min_aspect:
        ap.error("--max-aspect must be >= --min-aspect")
    for name in (
            "min_caption_unique_ratio",
            "max_caption_token_frequency",
            "max_caption_token_run",
            "max_caption_char_run"):
        val = float(getattr(args, name))
        if val < 0.0 or val > 1.0:
            ap.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.min_image_text_cosine is not None and not -1.0 <= args.min_image_text_cosine <= 1.0:
        ap.error("--min-image-text-cosine must be in [-1, 1]")
    if (args.max_image_duplicate_cosine is not None
            and not -1.0 <= args.max_image_duplicate_cosine <= 1.0):
        ap.error("--max-image-duplicate-cosine must be in [-1, 1]")
    if args.near_dedupe_lsh_bits <= 0 or args.near_dedupe_lsh_tables <= 0:
        ap.error("near-dedupe LSH settings must be positive")
    if args.eval_frac is not None and not 0.0 <= args.eval_frac < 1.0:
        ap.error("--eval-frac must be in [0, 1)")
    try:
        exact_dedupe_by = _parse_dedupe_by(args.exact_dedupe_by)
    except ValueError as exc:
        ap.error(str(exc))
    report = curate_manifest(
        args.manifest,
        root=args.root,
        out=args.out,
        split=args.split,
        max_records=args.max_records,
        min_caption_tokens=args.min_caption_tokens,
        max_caption_tokens=args.max_caption_tokens,
        min_width=args.min_width,
        min_height=args.min_height,
        min_pixels=args.min_pixels,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        min_caption_unique_ratio=args.min_caption_unique_ratio,
        max_caption_token_frequency=args.max_caption_token_frequency,
        max_caption_token_run=args.max_caption_token_run,
        max_caption_char_run=args.max_caption_char_run,
        min_aesthetic=args.min_aesthetic,
        max_nsfw=args.max_nsfw,
        max_watermark=args.max_watermark,
        require_text_embedding=args.require_text_embedding,
        require_text_embedding_sequence=args.require_text_embedding_sequence,
        require_image_embedding=args.require_image_embedding,
        require_image_embedding_sequence=args.require_image_embedding_sequence,
        min_image_text_cosine=args.min_image_text_cosine,
        max_image_duplicate_cosine=args.max_image_duplicate_cosine,
        exact_dedupe_by=exact_dedupe_by,
        dedupe_prefer_quality=not args.no_dedupe_prefer_quality,
        near_dedupe_lsh_bits=args.near_dedupe_lsh_bits,
        near_dedupe_lsh_tables=args.near_dedupe_lsh_tables,
        seed=args.seed,
        eval_frac=args.eval_frac,
        split_key=args.split_key,
        sample_errors=args.sample_errors,
        report_out=args.report_out)
    print(json.dumps(report, indent=1, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
