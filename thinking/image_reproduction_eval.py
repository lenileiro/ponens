"""Paired pixel/structure evaluation for shown-image reproduction manifests."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile

import numpy as np

from .image_data import load_image_tensor
from .image_latent import (
    REPRODUCTION_REPORT_METRICS,
    image_reproduction_metrics,
    image_reproduction_score,
    reference_reproduction_gate_failure_message,
    reference_reproduction_gate_report,
    summarize_reproduction_per_sample_metrics,
)


REFERENCE_KEYS = (
    "reference_image",
    "reference_path",
    "source_image",
    "target_image",
    "conditioning_image",
)


def _read_rows(path):
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
            yield from csv.DictReader(f, delimiter=delimiter)
        return
    raise ValueError(f"unsupported reproduction manifest extension: {ext!r}")


def _resolve_path(path, *, root="", manifest_dir=""):
    path = str(path or "")
    if not path:
        return ""
    if os.path.isabs(path):
        return os.path.normpath(path)
    base = root or manifest_dir
    return os.path.normpath(os.path.join(base, path)) if base else os.path.normpath(path)


def _row_image_path(row):
    return row.get("image") or row.get("path") or row.get("file") or row.get("filepath")


def _row_reference_path(row):
    for key in REFERENCE_KEYS:
        value = row.get(key)
        if value:
            return value
    return ""


def iter_reproduction_rows(path, *, root="", reference_root="", split="", max_records=0):
    manifest_dir = os.path.dirname(os.path.abspath(path))
    kept = 0
    for row_idx, row in enumerate(_read_rows(path)):
        row_split = str(row.get("split") or "")
        if split and row_split != split:
            continue
        image = _row_image_path(row)
        reference = _row_reference_path(row)
        if not image or not reference:
            yield {
                "index": int(row_idx),
                "usable": False,
                "error": "row requires generated image and reference_image fields",
                "row": row,
            }
            continue
        yield {
            "index": int(row_idx),
            "usable": True,
            "caption": str(row.get("caption") or row.get("prompt") or row.get("text") or ""),
            "generated": _resolve_path(image, root=root, manifest_dir=manifest_dir),
            "reference": _resolve_path(
                reference, root=reference_root, manifest_dir=manifest_dir),
            "split": row_split,
            "row": row,
        }
        kept += 1
        if max_records and kept >= int(max_records):
            break


def evaluate_reproduction_manifest(path, *, root="", reference_root="", split="",
                                   size=64, crop_mode="center", max_records=0,
                                   max_pair_rows=64):
    size = int(size)
    if size <= 0:
        raise ValueError("reproduction eval size must be positive")
    rows = list(iter_reproduction_rows(
        path, root=root, reference_root=reference_root, split=split,
        max_records=max_records))
    pair_metrics = []
    failures = []
    for item in rows:
        if not item.get("usable", False):
            failures.append({
                "index": int(item.get("index", -1)),
                "error": item.get("error", "unusable row"),
            })
            continue
        try:
            reference = load_image_tensor(
                item["reference"], size=size, device="cpu", crop_mode=crop_mode)
            generated = load_image_tensor(
                item["generated"], size=size, device="cpu", crop_mode=crop_mode)
            metrics = image_reproduction_metrics(
                reference.unsqueeze(0), generated.unsqueeze(0), prefix="reference_flow")
            metrics["reference_flow_score"] = image_reproduction_score(
                metrics, prefix="reference_flow")
            metrics.update({
                "index": int(item["index"]),
                "generated": item["generated"],
                "reference": item["reference"],
                "caption": item.get("caption", ""),
                "split": item.get("split", ""),
            })
            pair_metrics.append(metrics)
        except Exception as exc:
            failures.append({
                "index": int(item.get("index", -1)),
                "generated": item.get("generated", ""),
                "reference": item.get("reference", ""),
                "error": str(exc)[:240],
            })
    summary = summarize_reproduction_per_sample_metrics(
        pair_metrics, prefix="reference_flow")
    report = {
        "experiment": "image_reproduction_eval",
        "manifest": path,
        "root": root,
        "reference_root": reference_root,
        "split": split,
        "size": int(size),
        "crop_mode": crop_mode,
        "records": int(len(rows)),
        "usable": int(len(pair_metrics)),
        "failed": int(len(failures)),
        "max_records": int(max_records or 0),
        "failures": failures[:16],
        "pair_metrics": pair_metrics[:int(max_pair_rows)],
        "pair_metrics_truncated": bool(len(pair_metrics) > int(max_pair_rows)),
    }
    report.update(summary)
    report["sample_grid_reference_selected_denoise_score"] = float(
        summary.get("reference_flow_score_max", 0.0))
    report["reference_flow_metric_names"] = ["score", *REPRODUCTION_REPORT_METRICS]
    return report


def _write_ppm(path, pixels):
    pixels = np.asarray(pixels, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("PPM fixture must be HWC RGB")
    with open(path, "wb") as f:
        f.write(f"P6\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode("ascii"))
        f.write(pixels.tobytes())


def _fixture(seed=0, flat=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 24, dtype=np.float64),
        np.linspace(0.0, 1.0, 24, dtype=np.float64),
        indexing="ij",
    )
    if flat:
        img = np.full((24, 24, 3), 0.5, dtype=np.float64)
    else:
        img = np.stack([
            0.2 + 0.7 * xx,
            0.2 + 0.5 * yy,
            0.3 + 0.3 * np.sin((xx + yy) * np.pi * (3 + seed)),
        ], axis=-1)
        img += rng.normal(0.0, 0.01, size=img.shape)
    return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def selftest():
    with tempfile.TemporaryDirectory() as td:
        ref0 = os.path.join(td, "ref0.ppm")
        ref1 = os.path.join(td, "ref1.ppm")
        gen0 = os.path.join(td, "gen0.ppm")
        gen1 = os.path.join(td, "gen1.ppm")
        _write_ppm(ref0, _fixture(seed=0))
        _write_ppm(ref1, _fixture(seed=1))
        _write_ppm(gen0, _fixture(seed=0))
        _write_ppm(gen1, _fixture(seed=2, flat=True))
        manifest = os.path.join(td, "repro.jsonl")
        _write_jsonl(manifest, [
            {
                "image": "gen0.ppm",
                "caption": "same",
                "split": "reference_reproduction",
                "reference_image": "ref0.ppm",
            },
            {
                "image": "gen1.ppm",
                "caption": "flat",
                "split": "reference_reproduction",
                "reference_image": "ref1.ppm",
            },
        ])
        report = evaluate_reproduction_manifest(
            manifest, size=16, split="reference_reproduction")
        assert report["records"] == 2
        assert report["usable"] == 2
        assert report["reference_flow_pixel_mse_max"] > report[
            "reference_flow_pixel_mse_min"]
        assert report["reference_flow_physics_l1_max"] > report[
            "reference_flow_physics_l1_min"]
        gate = reference_reproduction_gate_report(
            report, max_pixel_mse=report["reference_flow_pixel_mse_max"] + 0.01)
        assert gate["sample_reference_quality_gate_passed"] is True
        fail_gate = reference_reproduction_gate_report(
            report, max_pixel_mse=report["reference_flow_pixel_mse_min"] + 0.001)
        assert fail_gate["sample_reference_quality_gate_passed"] is False
    print("image_reproduction_eval selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", default="", help="reference reproduction JSONL/CSV manifest")
    ap.add_argument("--root", default="", help="root for generated image paths")
    ap.add_argument("--reference-root", default="", help="root for reference image paths")
    ap.add_argument("--split", default="", help="optional split filter")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--crop-mode", default="center",
                    choices=("center", "random", "none", "pad"))
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--max-pair-rows", type=int, default=64)
    ap.add_argument("--max-pixel-mse", type=float, default=None)
    ap.add_argument("--max-pixel-mae", type=float, default=None)
    ap.add_argument("--max-structure-edge-l1", type=float, default=None)
    ap.add_argument("--max-structure-multiscale-l1", type=float, default=None)
    ap.add_argument("--max-structure-frequency-l1", type=float, default=None)
    ap.add_argument("--max-structure-ssim-loss", type=float, default=None)
    ap.add_argument("--max-texture-stats-l1", type=float, default=None)
    ap.add_argument("--max-physics-l1", type=float, default=None)
    ap.add_argument("--max-selected-score", type=float, default=None)
    ap.add_argument("--fail-on-gate", action="store_true")
    ap.add_argument("--report-out", default="")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return {"selftest": True}
    if not args.manifest:
        ap.error("--manifest is required")
    if args.size <= 0:
        ap.error("--size must be positive")
    if args.max_records < 0:
        ap.error("--max-records must be non-negative")
    if args.max_pair_rows < 0:
        ap.error("--max-pair-rows must be non-negative")
    for name in (
            "max_pixel_mse", "max_pixel_mae", "max_structure_edge_l1",
            "max_structure_multiscale_l1", "max_structure_frequency_l1",
            "max_structure_ssim_loss", "max_texture_stats_l1",
            "max_physics_l1", "max_selected_score"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            ap.error(f"--{name.replace('_', '-')} must be non-negative")
    report = evaluate_reproduction_manifest(
        args.manifest, root=args.root, reference_root=args.reference_root,
        split=args.split, size=args.size, crop_mode=args.crop_mode,
        max_records=args.max_records, max_pair_rows=args.max_pair_rows)
    report.update(reference_reproduction_gate_report(
        report,
        max_pixel_mse=args.max_pixel_mse,
        max_pixel_mae=args.max_pixel_mae,
        max_structure_edge_l1=args.max_structure_edge_l1,
        max_structure_multiscale_l1=args.max_structure_multiscale_l1,
        max_structure_frequency_l1=args.max_structure_frequency_l1,
        max_structure_ssim_loss=args.max_structure_ssim_loss,
        max_texture_stats_l1=args.max_texture_stats_l1,
        max_physics_l1=args.max_physics_l1,
        max_selected_score=args.max_selected_score,
    ))
    text = json.dumps(report, indent=1)
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"saved -> {args.report_out}")
    print(text)
    if args.fail_on_gate and not report.get("sample_reference_quality_gate_passed", False):
        msg = reference_reproduction_gate_failure_message(report)
        if msg:
            print(msg, file=sys.stderr)
        sys.exit(2)
    return report


if __name__ == "__main__":
    main()
