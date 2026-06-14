"""Build generic multimodal vision-read manifests from captioned image manifests.

The image generator writes captioned sample manifests.  `thinking.image_embed`
can attach pooled text/image embeddings to those rows.  This module converts the
embedded rows into the schema consumed by `thinking.multimodal`, using generic
named feature views instead of image-specific labels or hard-coded concepts.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
import os
import tempfile

import numpy as np

from .image_data import caption_tokens, read_image_manifest, summarize_records


VECTOR_SOURCES = ("embedding", "sequence_mean", "sequence_first", "sequence_flatten")
DEFAULT_SPLIT_MAP = {"train": "train", "eval": "eval", "generated": "eval"}


def _finite_vector(values, *, name, rec_id):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{rec_id}: {name} vector is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{rec_id}: {name} vector has non-finite values")
    return arr.astype(np.float32, copy=False)


def _sequence_vector(rows, *, mode, name, rec_id):
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"{rec_id}: {name} sequence must be [tokens, dim]")
    if not np.isfinite(arr).all():
        raise ValueError(f"{rec_id}: {name} sequence has non-finite values")
    if mode == "sequence_mean":
        return arr.mean(axis=0)
    if mode == "sequence_first":
        return arr[0]
    if mode == "sequence_flatten":
        return arr.reshape(-1)
    raise ValueError(f"unknown sequence vector mode {mode!r}")


def record_embedding_vector(rec, side="image", mode="embedding"):
    """Return a finite vector and a source label for one record embedding side."""
    side = str(side)
    mode = str(mode)
    if mode not in VECTOR_SOURCES:
        raise ValueError(f"unknown vector mode {mode!r}")
    rec_id = os.path.basename(str(rec.path)) or "<record>"
    if side == "image":
        pooled = rec.image_embedding
        sequence = rec.image_embedding_sequence
        name = "image_embedding"
    elif side == "text":
        pooled = rec.text_embedding
        sequence = rec.text_embedding_sequence
        name = "text_embedding"
    else:
        raise ValueError(f"unknown embedding side {side!r}")
    if mode == "embedding" and pooled is not None:
        return _finite_vector(pooled, name=name, rec_id=rec_id), "embedding"
    if mode != "embedding" and sequence is not None:
        return (
            _sequence_vector(sequence, mode=mode, name=f"{name}_sequence", rec_id=rec_id),
            mode,
        )
    if pooled is not None:
        return _finite_vector(pooled, name=name, rec_id=rec_id), "embedding"
    if sequence is not None:
        return (
            _sequence_vector(
                sequence, mode="sequence_mean", name=f"{name}_sequence", rec_id=rec_id),
            "sequence_mean",
        )
    return None, ""


def parse_split_map(raw):
    mapping = dict(DEFAULT_SPLIT_MAP)
    raw = str(raw or "").strip()
    if not raw:
        return mapping
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"split map entry {part!r} must be FROM=TO")
        src, dst = (x.strip() for x in part.split("=", 1))
        if not src or not dst:
            raise ValueError(f"split map entry {part!r} must be FROM=TO")
        mapping[src] = dst
    return mapping


def _token_list(text):
    tokens = caption_tokens(text)
    return tuple(tokens) if tokens else ("<empty>",)


def image_records_to_multimodal_rows(
        records, *, split_map=None, image_view="vision", text_view="text_feature",
        include_text_view=True, image_vector_mode="embedding",
        text_vector_mode="embedding", require_vision=True, id_prefix="vision"):
    split_map = split_map or DEFAULT_SPLIT_MAP
    rows = []
    report = {
        "input_records": 0,
        "records_written": 0,
        "records_skipped_missing_vision": 0,
        "records_skipped_missing_text_view": 0,
        "view_dims": OrderedDict(),
        "view_sources": Counter(),
        "splits": Counter(),
        "sources": Counter(),
    }
    for idx, rec in enumerate(records):
        report["input_records"] += 1
        row_id = f"{id_prefix}-{idx:06d}"
        views = OrderedDict()
        vision, vision_source = record_embedding_vector(
            rec, side="image", mode=image_vector_mode)
        if vision is None:
            if require_vision:
                report["records_skipped_missing_vision"] += 1
                continue
        else:
            views[str(image_view)] = vision.tolist()
            report["view_sources"][f"{image_view}:{vision_source}"] += 1
        if include_text_view:
            text_vec, text_source = record_embedding_vector(
                rec, side="text", mode=text_vector_mode)
            if text_vec is None:
                report["records_skipped_missing_text_view"] += 1
            else:
                views[str(text_view)] = text_vec.tolist()
                report["view_sources"][f"{text_view}:{text_source}"] += 1
        if not views:
            report["records_skipped_missing_vision"] += 1
            continue
        for name, vec in views.items():
            dim = len(vec)
            prior = report["view_dims"].get(name)
            if prior is None:
                report["view_dims"][name] = int(dim)
            elif int(prior) != int(dim):
                raise ValueError(
                    f"{row_id}: view {name!r} dimension {dim} does not match {prior}")
        split = split_map.get(rec.split, rec.split)
        tokens = _token_list(rec.caption)
        row = {
            "id": row_id,
            "split": split,
            "text": list(tokens),
            "target": list(tokens),
            "views": views,
            "meta": {
                "image": rec.path,
                "caption": rec.caption,
                "source": rec.source,
                "image_split": rec.split,
                "vision_read_target": "caption",
            },
        }
        rows.append(row)
        report["records_written"] += 1
        report["splits"][split] += 1
        if rec.source:
            report["sources"][rec.source] += 1
    report["view_dims"] = dict(report["view_dims"])
    report["view_sources"] = dict(sorted(report["view_sources"].items()))
    report["splits"] = dict(sorted(report["splits"].items()))
    report["sources"] = dict(sorted(report["sources"].items()))
    return rows, report


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build_manifest(
        manifests, *, root="", split="", max_records=0, split_map=None,
        image_view="vision", text_view="text_feature", include_text_view=True,
        image_vector_mode="embedding", text_vector_mode="embedding",
        require_vision=True, id_prefix="vision"):
    split_map = split_map or DEFAULT_SPLIT_MAP
    all_rows = []
    manifest_reports = []
    for manifest_idx, manifest in enumerate(manifests):
        records = read_image_manifest(
            manifest, root=root, split=split, max_records=max_records)
        rows, report = image_records_to_multimodal_rows(
            records, split_map=split_map, image_view=image_view,
            text_view=text_view, include_text_view=include_text_view,
            image_vector_mode=image_vector_mode, text_vector_mode=text_vector_mode,
            require_vision=require_vision, id_prefix=f"{id_prefix}{manifest_idx}")
        all_rows.extend(rows)
        report.update({
            "manifest": manifest,
            "summary": summarize_records(records),
        })
        manifest_reports.append(report)
    splits = Counter(row["split"] for row in all_rows)
    view_dims = OrderedDict()
    for row in all_rows:
        for name, vec in row["views"].items():
            view_dims.setdefault(name, len(vec))
    return all_rows, {
        "experiment": "vision_read_manifest",
        "manifests": list(manifests),
        "records_written": int(len(all_rows)),
        "splits": dict(sorted(splits.items())),
        "view_dims": dict(view_dims),
        "split_map": dict(sorted(split_map.items())),
        "image_view": str(image_view),
        "text_view": str(text_view) if include_text_view else "",
        "include_text_view": bool(include_text_view),
        "image_vector_mode": str(image_vector_mode),
        "text_vector_mode": str(text_vector_mode),
        "require_vision": bool(require_vision),
        "manifest_reports": manifest_reports,
    }


def selftest():
    with tempfile.TemporaryDirectory() as td:
        manifest = os.path.join(td, "images.jsonl")
        rows = [
            {
                "image": "a.ppm",
                "caption": "red square sample",
                "split": "train",
                "source": "fixture",
                "image_embedding": [1.0, 0.0, 0.0],
                "text_embedding": [0.9, 0.1, 0.0],
            },
            {
                "image": "b.ppm",
                "caption": "blue round sample",
                "split": "generated",
                "source": "fixture",
                "image_embedding_sequence": [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
                "text_embedding_sequence": [[0.0, 1.0, 0.0], [0.1, 0.8, 0.1]],
            },
        ]
        with open(manifest, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        out = os.path.join(td, "mm.jsonl")
        records, report = build_manifest(
            [manifest], root=td, split="", split_map=parse_split_map("generated=eval"),
            image_vector_mode="sequence_mean", text_vector_mode="sequence_mean")
        write_jsonl(records, out)
        assert report["records_written"] == 2
        assert report["splits"] == {"eval": 1, "train": 1}
        assert report["view_dims"]["vision"] == 3
        with open(out, "r", encoding="utf-8") as f:
            written = [json.loads(line) for line in f if line.strip()]
        assert written[0]["target"] == ["red", "square", "sample"]
        assert "vision" in written[1]["views"]
        assert written[1]["split"] == "eval"
        from .multimodal import feature_dims, load_manifest
        mm_records = load_manifest(out)
        assert feature_dims(mm_records)["vision"] == 3
    print("vision_read selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", action="append", default=[],
                    help="captioned image manifest with image/text embedding sidecars")
    ap.add_argument("--root", default="", help="base directory for relative image paths")
    ap.add_argument("--split", default="",
                    help="input split filter; empty keeps all input splits")
    ap.add_argument("--max-records", type=int, default=0,
                    help="per-manifest input record cap; 0 keeps all")
    ap.add_argument("--split-map", default="",
                    help="comma/semicolon split remap entries such as generated=eval")
    ap.add_argument("--out", default="", help="output multimodal JSONL manifest")
    ap.add_argument("--report-out", default="", help="optional JSON report path")
    ap.add_argument("--image-view", default="vision")
    ap.add_argument("--text-view", default="text_feature")
    ap.add_argument("--no-text-view", action="store_true",
                    help="omit pooled text embedding view; text tokens are still written")
    ap.add_argument("--image-vector-mode", default="embedding", choices=VECTOR_SOURCES)
    ap.add_argument("--text-vector-mode", default="embedding", choices=VECTOR_SOURCES)
    ap.add_argument("--allow-missing-vision", action="store_true",
                    help="write records even when only non-vision feature views exist")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.manifest:
        ap.error("--manifest is required unless --selftest is set")
    if not args.out:
        ap.error("--out is required")
    if args.max_records < 0:
        ap.error("--max-records must be non-negative")
    try:
        split_map = parse_split_map(args.split_map)
        rows, report = build_manifest(
            args.manifest, root=args.root, split=args.split,
            max_records=args.max_records, split_map=split_map,
            image_view=args.image_view, text_view=args.text_view,
            include_text_view=not args.no_text_view,
            image_vector_mode=args.image_vector_mode,
            text_vector_mode=args.text_vector_mode,
            require_vision=not args.allow_missing_vision)
    except ValueError as exc:
        ap.error(str(exc))
    write_jsonl(rows, args.out)
    report["out"] = args.out
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
