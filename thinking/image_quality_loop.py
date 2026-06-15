"""Closed-loop quality evaluation for generated image manifests.

This composes the existing image stack instead of adding task-specific visual
rules: embed real/generated images, score distribution and image/text alignment,
then optionally ask the generic multimodal reader to recover captions from the
vision feature view.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

from .image_data import read_image_manifest, summarize_records, write_image_manifest
from .image_embed import make_embedder, write_embedding_sidecar
from .image_eval import evaluate_records, image_eval_gate, load_records
from .vision_read import build_manifest, parse_split_map, write_jsonl


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _read_records(path, *, root="", split="", min_aesthetic=None, max_records=0,
                  max_nsfw=None, max_watermark=None):
    return read_image_manifest(
        path, root=root, split=split, min_aesthetic=min_aesthetic,
        max_records=max_records, max_nsfw=max_nsfw, max_watermark=max_watermark)


def _normalize_embedded_manifest(src_path, out_path, *, root="", split="",
                                 embedded_manifest="", embedded_root="",
                                 min_aesthetic=None, max_records=0,
                                 max_nsfw=None, max_watermark=None):
    source_records = _read_records(
        src_path, root=root, split=split, min_aesthetic=min_aesthetic,
        max_records=max_records, max_nsfw=max_nsfw, max_watermark=max_watermark)
    source_paths = {os.path.abspath(rec.path) for rec in source_records}
    embedded_records = _read_records(
        embedded_manifest, root=embedded_root or root, split=split,
        min_aesthetic=min_aesthetic,
        max_records=0, max_nsfw=max_nsfw, max_watermark=max_watermark)
    records = [
        rec for rec in embedded_records
        if os.path.abspath(rec.path) in source_paths
    ]
    write_image_manifest(records, out_path, root="")
    return {
        "embedded_manifest": out_path,
        "embedded_source_manifest": embedded_manifest,
        "embedded_source_root": embedded_root or root,
        "source_manifest": src_path,
        "source_root": root,
        "source_split": split,
        "records": len(records),
        "source_records": len(source_records),
        "embedded_records": len(embedded_records),
        "embedded_records_matched": len(records),
        "embedded_from_existing": True,
        "summary": summarize_records(records),
    }


def _embed_manifest(src_path, out_path, *, root="", split="", min_aesthetic=None,
                    max_records=0, max_nsfw=None, max_watermark=None, embedder=None,
                    features="both", batch=32):
    records = _read_records(
        src_path, root=root, split=split, min_aesthetic=min_aesthetic,
        max_records=max_records, max_nsfw=max_nsfw, max_watermark=max_watermark)
    report = write_embedding_sidecar(
        records, out_path, root="", embedder=embedder, features=features,
        batch=batch)
    report.update({
        "embedded_manifest": out_path,
        "source_manifest": src_path,
        "source_root": root,
        "source_split": split,
        "records": len(records),
        "embedded_from_existing": False,
        "summary": summarize_records(records),
    })
    return report


def materialize_embedded_manifest(src_path, out_path, *, root="", split="",
                                  embedded_manifest="", embedded_root="",
                                  min_aesthetic=None, max_records=0,
                                  max_nsfw=None, max_watermark=None,
                                  embedder=None, features="both", batch=32):
    """Return a manifest with absolute image paths and embedding fields."""
    if embedded_manifest:
        return _normalize_embedded_manifest(
            src_path, out_path, root=root, split=split,
            embedded_manifest=embedded_manifest,
            embedded_root=embedded_root or root,
            min_aesthetic=min_aesthetic, max_records=max_records,
            max_nsfw=max_nsfw, max_watermark=max_watermark)
    return _embed_manifest(
        src_path, out_path, root=root, split=split,
        min_aesthetic=min_aesthetic, max_records=max_records,
        max_nsfw=max_nsfw, max_watermark=max_watermark, embedder=embedder,
        features=features, batch=batch)


def _metric(report, path, default=0.0):
    cur = report
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    try:
        value = float(cur)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def vision_read_gate(report, min_sensor_token_acc=0.0, min_full_token_acc=0.0,
                     max_sensor_loss=None):
    checks = []

    def add(name, value, threshold, mode):
        if threshold is None:
            return
        value = float(value)
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

    add(
        "vision_read_sensor_only_token_acc",
        _metric(report, ("teacher_forced", "sensor_only", "token_acc")),
        min_sensor_token_acc,
        "min",
    )
    add(
        "vision_read_full_token_acc",
        _metric(report, ("teacher_forced", "full", "token_acc")),
        min_full_token_acc,
        "min",
    )
    add(
        "vision_read_sensor_only_loss",
        _metric(report, ("teacher_forced", "sensor_only", "loss")),
        max_sensor_loss,
        "max",
    )
    passed = all(row["pass"] for row in checks)
    return {
        "vision_read_gate_pass": bool(passed),
        "vision_read_gate_checks": checks,
        "vision_read_gate_failed": [row for row in checks if not row["pass"]],
    }


def run_vision_reader(vision_manifest, *, out_path="", checkpoint_path="",
                      steps=0, seed=0, device="cpu", batch=4, d=32,
                      layers=1, heads=2, max_len=64, eval_n=16,
                      concept_tokens=2, latent_concept_slots=2,
                      latent_concept_w=0.01,
                      latent_concept_factorization_w=0.01,
                      latent_concept_memory_size=8,
                      latent_concept_memory_w=0.01,
                      latent_concept_bridge_w=0.01):
    if int(steps) <= 0:
        return {}
    from .multimodal import run
    return run(
        vision_manifest, steps=int(steps), seed=int(seed), device=device,
        batch=int(batch), d=int(d), layers=int(layers), heads=int(heads),
        max_len=int(max_len), eval_n=int(eval_n), checkpoint=checkpoint_path or None,
        out=out_path or None, concept_tokens=int(concept_tokens),
        latent_concept_slots=int(latent_concept_slots),
        latent_concept_w=float(latent_concept_w),
        latent_concept_factorization_w=float(latent_concept_factorization_w),
        latent_concept_memory_size=int(latent_concept_memory_size),
        latent_concept_memory_w=float(latent_concept_memory_w),
        latent_concept_bridge_w=float(latent_concept_bridge_w))


def run_quality_loop(
        *, real_manifest, generated_manifest, work_dir,
        real_root="", generated_root="", real_split="", generated_split="",
        real_embedded_manifest="", generated_embedded_manifest="",
        real_embedded_root="", generated_embedded_root="",
        min_aesthetic=None, max_nsfw=None, max_watermark=None,
        embed_max_records=0, real_embed_max_records=0,
        generated_embed_max_records=0, eval_max_records=2048,
        real_eval_max_records=0, generated_eval_max_records=0,
        embedding_backend="stats", embedding_model="", embedding_device="cpu",
        embedding_dtype="auto", embedding_features="both",
        text_embed_mode="pooled", image_embed_mode="pooled",
        text_sequence_model="", text_sequence_max_length=0,
        embedding_batch=32, stats_dim=64, stats_image_size=32,
        trust_remote_code=False, normalize_embeddings=True,
        embedding_kind="image", dim_policy="error",
        min_score=0.0, min_support_precision=0.0, min_support_recall=0.0,
        min_image_text_cos=None, max_frechet=None, max_mmd_rbf=None,
        min_generated_neighbor_l2_p05=None,
        min_generated_real_l2_p01=None,
        vision_read_steps=0, vision_read_seed=0, vision_read_device="cpu",
        vision_read_batch=4, vision_read_dim=32, vision_read_layers=1,
        vision_read_heads=2, vision_read_max_len=64, vision_read_eval_n=16,
        vision_read_min_sensor_token_acc=0.0,
        vision_read_min_full_token_acc=0.0,
        vision_read_max_sensor_loss=None):
    work_dir = _ensure_dir(work_dir)
    embedder = None
    if not (real_embedded_manifest and generated_embedded_manifest):
        embedder = make_embedder(
            backend=embedding_backend, model=embedding_model,
            device=embedding_device, dtype=embedding_dtype,
            normalize=normalize_embeddings, stats_dim=stats_dim,
            stats_image_size=stats_image_size, text_mode=text_embed_mode,
            image_mode=image_embed_mode, trust_remote_code=trust_remote_code,
            text_sequence_model=text_sequence_model,
            text_sequence_max_length=text_sequence_max_length)
    shared_embed_max = int(embed_max_records or 0)
    real_embed_max = int(real_embed_max_records or shared_embed_max)
    gen_embed_max = int(generated_embed_max_records or shared_embed_max)
    real_embed_path = os.path.join(work_dir, "real_embedded.jsonl")
    generated_embed_path = os.path.join(work_dir, "generated_embedded.jsonl")
    real_embed_report = materialize_embedded_manifest(
        real_manifest, real_embed_path, root=real_root, split=real_split,
        embedded_manifest=real_embedded_manifest, embedded_root=real_embedded_root,
        min_aesthetic=min_aesthetic, max_records=real_embed_max,
        max_nsfw=max_nsfw, max_watermark=max_watermark, embedder=embedder,
        features=embedding_features, batch=embedding_batch)
    generated_embed_report = materialize_embedded_manifest(
        generated_manifest, generated_embed_path, root=generated_root,
        split=generated_split, embedded_manifest=generated_embedded_manifest,
        embedded_root=generated_embedded_root, min_aesthetic=min_aesthetic,
        max_records=gen_embed_max, max_nsfw=max_nsfw,
        max_watermark=max_watermark, embedder=embedder,
        features=embedding_features, batch=embedding_batch)

    shared_eval_max = int(eval_max_records or 0)
    real_eval_max = int(real_eval_max_records or shared_eval_max)
    gen_eval_max = int(generated_eval_max_records or shared_eval_max)
    real_records, real_merge = load_records(
        real_embed_path, split="", max_records=real_eval_max,
        min_aesthetic=min_aesthetic, max_nsfw=max_nsfw,
        max_watermark=max_watermark)
    generated_records, generated_merge = load_records(
        generated_embed_path, split="", max_records=gen_eval_max,
        min_aesthetic=min_aesthetic, max_nsfw=max_nsfw,
        max_watermark=max_watermark)
    embedding_report = evaluate_records(
        real_records, generated_records, embedding_kind=embedding_kind,
        dim_policy=dim_policy, normalize=normalize_embeddings)
    embedding_report.update(image_eval_gate(
        embedding_report, min_score=min_score,
        min_support_precision=min_support_precision,
        min_support_recall=min_support_recall,
        min_image_text_cos=min_image_text_cos, max_frechet=max_frechet,
        max_mmd_rbf=max_mmd_rbf,
        min_generated_neighbor_l2_p05=min_generated_neighbor_l2_p05,
        min_generated_real_l2_p01=min_generated_real_l2_p01,
        embedding_kind=embedding_kind))
    if real_merge:
        embedding_report["real_embedding_merge"] = real_merge
    if generated_merge:
        embedding_report["generated_embedding_merge"] = generated_merge

    vision_manifest = os.path.join(work_dir, "vision_read.jsonl")
    vision_rows, vision_manifest_report = build_manifest(
        [real_embed_path, generated_embed_path], root="", split="",
        split_map=parse_split_map("generated=eval"))
    write_jsonl(vision_rows, vision_manifest)
    vision_manifest_report["out"] = vision_manifest

    vision_report_path = os.path.join(work_dir, "vision_read_report.json")
    vision_checkpoint_path = os.path.join(work_dir, "vision_read.pt")
    vision_report = run_vision_reader(
        vision_manifest, out_path=vision_report_path,
        checkpoint_path=vision_checkpoint_path, steps=vision_read_steps,
        seed=vision_read_seed, device=vision_read_device,
        batch=vision_read_batch, d=vision_read_dim,
        layers=vision_read_layers, heads=vision_read_heads,
        max_len=vision_read_max_len, eval_n=vision_read_eval_n)
    vision_gate = vision_read_gate(
        vision_report, min_sensor_token_acc=vision_read_min_sensor_token_acc,
        min_full_token_acc=vision_read_min_full_token_acc,
        max_sensor_loss=vision_read_max_sensor_loss) if vision_report else {
            "vision_read_gate_pass": True,
            "vision_read_gate_checks": [],
            "vision_read_gate_failed": [],
            "vision_read_skipped": True,
        }
    embedding_gate_configured = bool(
        float(min_score or 0.0) > 0.0
        or float(min_support_precision or 0.0) > 0.0
        or float(min_support_recall or 0.0) > 0.0
        or min_image_text_cos is not None
        or max_frechet is not None
        or max_mmd_rbf is not None
        or min_generated_neighbor_l2_p05 is not None
        or min_generated_real_l2_p01 is not None)
    vision_gate_configured = bool(
        vision_report and (
            float(vision_read_min_sensor_token_acc or 0.0) > 0.0
            or float(vision_read_min_full_token_acc or 0.0) > 0.0
            or vision_read_max_sensor_loss is not None))
    combined_gate = bool(embedding_report.get("image_eval_gate_pass", False))
    if vision_report:
        combined_gate = combined_gate and bool(vision_gate.get("vision_read_gate_pass", False))

    report = {
        "experiment": "image_quality_loop",
        "real": real_embed_report,
        "generated": generated_embed_report,
        "embedding_eval": embedding_report,
        "vision_read_manifest": vision_manifest_report,
        "vision_read_eval": vision_report,
        "vision_read_gate": vision_gate,
        "embedding_gate_configured": embedding_gate_configured,
        "vision_read_gate_configured": vision_gate_configured,
        "quality_loop_gate_configured": bool(
            embedding_gate_configured or vision_gate_configured),
        "quality_loop_gate_pass": bool(combined_gate),
        "quality_loop_gate_failed": (
            list(embedding_report.get("image_eval_gate_failed", []))
            + list(vision_gate.get("vision_read_gate_failed", []))
        ),
        "work_dir": work_dir,
    }
    return report


def _write_ppm(path, color):
    color = np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
    tile = np.tile(color, (16, 16, 1))
    with open(path, "wb") as f:
        f.write(b"P6\n16 16\n255\n")
        f.write(tile.tobytes())


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def selftest():
    with tempfile.TemporaryDirectory() as td:
        real_dir = os.path.join(td, "real")
        gen_dir = os.path.join(td, "gen")
        os.makedirs(real_dir)
        os.makedirs(gen_dir)
        _write_ppm(os.path.join(real_dir, "red.ppm"), [220, 40, 30])
        _write_ppm(os.path.join(real_dir, "blue.ppm"), [30, 70, 220])
        _write_ppm(os.path.join(gen_dir, "red.ppm"), [210, 45, 35])
        _write_ppm(os.path.join(gen_dir, "blue.ppm"), [35, 80, 210])
        real_manifest = os.path.join(td, "real.jsonl")
        generated_manifest = os.path.join(td, "generated.jsonl")
        _write_jsonl(real_manifest, [
            {"image": "red.ppm", "caption": "red square", "split": "train"},
            {"image": "blue.ppm", "caption": "blue square", "split": "train"},
        ])
        _write_jsonl(generated_manifest, [
            {"image": "red.ppm", "caption": "red square", "split": "generated"},
            {"image": "blue.ppm", "caption": "blue square", "split": "generated"},
        ])
        report = run_quality_loop(
            real_manifest=real_manifest, generated_manifest=generated_manifest,
            real_root=real_dir, generated_root=gen_dir,
            work_dir=os.path.join(td, "loop"), stats_dim=8,
            vision_read_steps=1, vision_read_eval_n=2,
            vision_read_max_len=16)
        assert report["embedding_eval"]["image_distribution_generated_n"] == 2
        assert "image_distribution_generated_nearest_generated_l2_p05" in (
            report["embedding_eval"])
        assert "image_distribution_generated_nearest_real_l2_p01" in (
            report["embedding_eval"])
        assert report["vision_read_manifest"]["splits"] == {"eval": 2, "train": 2}
        assert report["vision_read_eval"]["manifest"]["view_dims"]["vision"] == 8
        assert "sensor_only" in report["vision_read_eval"]["teacher_forced"]
        extra_manifest = os.path.join(td, "extra_embedded.jsonl")
        _write_jsonl(extra_manifest, [
            {
                "image": os.path.join(real_dir, "red.ppm"),
                "caption": "red square",
                "split": "train",
                "image_embedding": [1.0, 0.0],
                "text_embedding": [1.0, 0.0],
            },
            {
                "image": os.path.join(real_dir, "blue.ppm"),
                "caption": "blue square",
                "split": "train",
                "image_embedding": [0.0, 1.0],
                "text_embedding": [0.0, 1.0],
            },
            {
                "image": os.path.join(real_dir, "extra.ppm"),
                "caption": "extra square",
                "split": "train",
                "image_embedding": [0.5, 0.5],
                "text_embedding": [0.5, 0.5],
            },
        ])
        filtered = materialize_embedded_manifest(
            real_manifest, os.path.join(td, "filtered_embedded.jsonl"),
            root=real_dir, split="train", embedded_manifest=extra_manifest)
        assert filtered["source_records"] == 2
        assert filtered["embedded_records"] == 3
        assert filtered["embedded_records_matched"] == 2
    print("image_quality_loop selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real-manifest", default="")
    ap.add_argument("--generated-manifest", default="")
    ap.add_argument("--real-root", default="")
    ap.add_argument("--generated-root", default="")
    ap.add_argument("--real-split", default="")
    ap.add_argument("--generated-split", default="")
    ap.add_argument("--real-embedded-manifest", default="")
    ap.add_argument("--generated-embedded-manifest", default="")
    ap.add_argument("--real-embedded-root", default="")
    ap.add_argument("--generated-embedded-root", default="")
    ap.add_argument("--work-dir", default="runs/image_quality_loop")
    ap.add_argument("--report-out", default="")
    ap.add_argument("--min-aesthetic", type=float, default=None)
    ap.add_argument("--max-nsfw", type=float, default=None)
    ap.add_argument("--max-watermark", type=float, default=None)
    ap.add_argument("--embed-max-records", type=int, default=0)
    ap.add_argument("--real-embed-max-records", type=int, default=0)
    ap.add_argument("--generated-embed-max-records", type=int, default=0)
    ap.add_argument("--eval-max-records", type=int, default=2048)
    ap.add_argument("--real-eval-max-records", type=int, default=0)
    ap.add_argument("--generated-eval-max-records", type=int, default=0)
    ap.add_argument("--embedding-backend", default="stats", choices=("stats", "hf"))
    ap.add_argument("--embedding-model", default="")
    ap.add_argument("--embedding-device", default="cpu")
    ap.add_argument("--embedding-dtype", default="auto",
                    choices=("auto", "fp32", "fp16", "bf16"))
    ap.add_argument("--embedding-features", default="both",
                    choices=("both", "image", "text"))
    ap.add_argument("--text-embed-mode", default="pooled",
                    choices=("pooled", "tokens", "both"))
    ap.add_argument("--image-embed-mode", default="pooled",
                    choices=("pooled", "tokens", "both"))
    ap.add_argument("--text-sequence-model", default="")
    ap.add_argument("--text-sequence-max-length", type=int, default=0)
    ap.add_argument("--embedding-batch", type=int, default=32)
    ap.add_argument("--stats-dim", type=int, default=64)
    ap.add_argument("--stats-image-size", type=int, default=32)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--embedding-kind", default="image",
                    choices=("image", "text", "image_sequence", "text_sequence"))
    ap.add_argument("--dim-policy", default="error", choices=("error", "largest"))
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--min-support-precision", type=float, default=0.0)
    ap.add_argument("--min-support-recall", type=float, default=0.0)
    ap.add_argument("--min-image-text-cos", type=float, default=None)
    ap.add_argument("--max-frechet", type=float, default=None)
    ap.add_argument("--max-mmd-rbf", type=float, default=None)
    ap.add_argument("--min-generated-neighbor-l2-p05", type=float, default=None,
                    help=("minimum generated/generated nearest-neighbor L2 p05; "
                          "catches collapsed duplicate outputs"))
    ap.add_argument("--min-generated-real-l2-p01", type=float, default=None,
                    help=("minimum generated/real nearest-neighbor L2 p01; "
                          "catches over-near training-set copies"))
    ap.add_argument("--vision-read-steps", type=int, default=0)
    ap.add_argument("--vision-read-seed", type=int, default=0)
    ap.add_argument("--vision-read-device", default="")
    ap.add_argument("--vision-read-batch", type=int, default=4)
    ap.add_argument("--vision-read-dim", type=int, default=32)
    ap.add_argument("--vision-read-layers", type=int, default=1)
    ap.add_argument("--vision-read-heads", type=int, default=2)
    ap.add_argument("--vision-read-max-len", type=int, default=64)
    ap.add_argument("--vision-read-eval-n", type=int, default=16)
    ap.add_argument("--vision-read-min-sensor-token-acc", type=float, default=0.0)
    ap.add_argument("--vision-read-min-full-token-acc", type=float, default=0.0)
    ap.add_argument("--vision-read-max-sensor-loss", type=float, default=None)
    ap.add_argument("--fail-on-gate", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return {"selftest": True}
    if not args.real_manifest or not args.generated_manifest:
        ap.error("--real-manifest and --generated-manifest are required")
    if args.embedding_batch <= 0:
        ap.error("--embedding-batch must be positive")
    for name in (
            "embed_max_records", "real_embed_max_records",
            "generated_embed_max_records", "eval_max_records",
            "real_eval_max_records", "generated_eval_max_records",
            "vision_read_steps"):
        if getattr(args, name) < 0:
            ap.error(f"--{name.replace('_', '-')} must be non-negative")
    if (args.min_generated_neighbor_l2_p05 is not None
            and args.min_generated_neighbor_l2_p05 < 0.0):
        ap.error("--min-generated-neighbor-l2-p05 must be non-negative")
    if (args.min_generated_real_l2_p01 is not None
            and args.min_generated_real_l2_p01 < 0.0):
        ap.error("--min-generated-real-l2-p01 must be non-negative")
    report = run_quality_loop(
        real_manifest=args.real_manifest,
        generated_manifest=args.generated_manifest,
        work_dir=args.work_dir,
        real_root=args.real_root,
        generated_root=args.generated_root,
        real_split=args.real_split,
        generated_split=args.generated_split,
        real_embedded_manifest=args.real_embedded_manifest,
        generated_embedded_manifest=args.generated_embedded_manifest,
        real_embedded_root=args.real_embedded_root,
        generated_embedded_root=args.generated_embedded_root,
        min_aesthetic=args.min_aesthetic,
        max_nsfw=args.max_nsfw,
        max_watermark=args.max_watermark,
        embed_max_records=args.embed_max_records,
        real_embed_max_records=args.real_embed_max_records,
        generated_embed_max_records=args.generated_embed_max_records,
        eval_max_records=args.eval_max_records,
        real_eval_max_records=args.real_eval_max_records,
        generated_eval_max_records=args.generated_eval_max_records,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_dtype=args.embedding_dtype,
        embedding_features=args.embedding_features,
        text_embed_mode=args.text_embed_mode,
        image_embed_mode=args.image_embed_mode,
        text_sequence_model=args.text_sequence_model,
        text_sequence_max_length=args.text_sequence_max_length,
        embedding_batch=args.embedding_batch,
        stats_dim=args.stats_dim,
        stats_image_size=args.stats_image_size,
        trust_remote_code=args.trust_remote_code,
        normalize_embeddings=not args.no_normalize,
        embedding_kind=args.embedding_kind,
        dim_policy=args.dim_policy,
        min_score=args.min_score,
        min_support_precision=args.min_support_precision,
        min_support_recall=args.min_support_recall,
        min_image_text_cos=args.min_image_text_cos,
        max_frechet=args.max_frechet,
        max_mmd_rbf=args.max_mmd_rbf,
        min_generated_neighbor_l2_p05=args.min_generated_neighbor_l2_p05,
        min_generated_real_l2_p01=args.min_generated_real_l2_p01,
        vision_read_steps=args.vision_read_steps,
        vision_read_seed=args.vision_read_seed,
        vision_read_device=args.vision_read_device or args.embedding_device,
        vision_read_batch=args.vision_read_batch,
        vision_read_dim=args.vision_read_dim,
        vision_read_layers=args.vision_read_layers,
        vision_read_heads=args.vision_read_heads,
        vision_read_max_len=args.vision_read_max_len,
        vision_read_eval_n=args.vision_read_eval_n,
        vision_read_min_sensor_token_acc=args.vision_read_min_sensor_token_acc,
        vision_read_min_full_token_acc=args.vision_read_min_full_token_acc,
        vision_read_max_sensor_loss=args.vision_read_max_sensor_loss)
    text = json.dumps(report, indent=1)
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"saved -> {args.report_out}")
    print(text)
    if args.fail_on_gate and not report.get("quality_loop_gate_pass", False):
        sys.exit(2)
    return report


if __name__ == "__main__":
    main()
