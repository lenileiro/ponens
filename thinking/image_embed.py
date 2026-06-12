"""Build generic text/image embedding sidecars for image manifests.

High-quality image generation needs semantically rich conditioning and visual-latent
alignment.  This module keeps that preprocessing outside the generator: it reads a
captioned image manifest and writes a JSONL sidecar with `text_embedding` and/or
`image_embedding` arrays that `thinking.image_data` can merge into the cleaned
training manifest.

The `stats` backend is dependency-free and exists for local smoke tests.  Real
GPU preprocessing should use `--backend hf` with a vision-language encoder such
as CLIP/SigLIP for text+image features, or an image encoder such as DINO/MAE with
`--features image`.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .image_data import (ImageTextRecord, caption_tokens, load_image_tensor,
                         read_image_manifest)


FEATURE_CHOICES = ("both", "image", "text")


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


def _normalize_rows(x, enabled=True):
    if not enabled:
        return x
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def _float_list(row):
    return [float(x) for x in row.detach().cpu().float().tolist()]


def _needs_image(features):
    return features in ("both", "image")


def _needs_text(features):
    return features in ("both", "text")


def _batched(seq, batch):
    batch = max(1, int(batch))
    for i in range(0, len(seq), batch):
        yield seq[i:i + batch]


def _hash_caption_embedding(caption, dim=64):
    dim = max(1, int(dim))
    vec = torch.zeros(dim, dtype=torch.float32)
    toks = caption_tokens(caption)
    if not toks:
        return vec
    for tok in toks:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little", signed=False)
        idx = raw % dim
        sign = 1.0 if ((raw >> 11) & 1) else -1.0
        vec[idx] += sign / math.sqrt(len(toks))
    vec[0] += min(len(toks), 255) / 255.0
    return vec


def _stats_image_embedding(path, dim=64, size=32):
    dim = max(1, int(dim))
    x = load_image_tensor(path, size=max(1, int(size)), device="cpu", center_crop=True)
    stats = torch.cat([
        x.mean(dim=(1, 2)),
        x.std(dim=(1, 2), unbiased=False),
        x.amin(dim=(1, 2)),
        x.amax(dim=(1, 2)),
    ])
    remaining = max(0, dim - int(stats.numel()))
    if remaining:
        grid_side = max(1, int(math.ceil(math.sqrt(remaining / 3.0))))
        pooled = F.adaptive_avg_pool2d(x[None], (grid_side, grid_side))[0].flatten()
        vec = torch.cat([stats, pooled])
    else:
        vec = stats
    if vec.numel() < dim:
        vec = F.pad(vec, (0, dim - int(vec.numel())))
    return vec[:dim]


def _image_array_for_processor(path):
    x = load_image_tensor(path, size=0, device="cpu", center_crop=False)
    arr = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    return arr.to(torch.uint8).permute(1, 2, 0).numpy()


def _to_device(inputs, device):
    return {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }


def _pooled_output(outputs):
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        val = getattr(outputs, name, None)
        if val is not None:
            return val
    last = getattr(outputs, "last_hidden_state", None)
    if last is not None:
        if last.ndim == 3:
            return last[:, 0]
        return last
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if getattr(first, "ndim", 0) == 3:
            return first[:, 0]
        return first
    raise RuntimeError("could not find pooled embeddings in model output")


@dataclass
class StatsEmbedder:
    dim: int = 64
    image_size: int = 32
    normalize: bool = True

    backend_name: str = "stats"
    model_name: str = "deterministic-stats"

    def encode_images(self, records: Sequence[ImageTextRecord]):
        rows = [_stats_image_embedding(rec.path, dim=self.dim, size=self.image_size)
                for rec in records]
        return _normalize_rows(torch.stack(rows, dim=0), self.normalize)

    def encode_texts(self, records: Sequence[ImageTextRecord]):
        rows = [_hash_caption_embedding(rec.caption, dim=self.dim) for rec in records]
        return _normalize_rows(torch.stack(rows, dim=0), self.normalize)


class HFEmbedder:
    def __init__(self, model_name, device="cpu", dtype="auto", normalize=True,
                 trust_remote_code=False):
        if not model_name:
            raise ValueError("--model is required for --backend hf")
        try:
            from transformers import AutoModel, AutoProcessor
        except Exception as e:  # pragma: no cover - optional GPU dependency.
            raise ImportError(
                "Hugging Face embedding requires transformers. Install it on the GPU box "
                "with `pip install transformers accelerate pillow`."
            ) from e
        self.backend_name = "hf"
        self.model_name = str(model_name)
        self.device = torch.device(device)
        self.normalize = bool(normalize)
        torch_dtype = None
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp32":
            torch_dtype = torch.float32
        elif dtype != "auto":
            raise ValueError("dtype must be auto, fp32, fp16, or bf16")
        kwargs = {"trust_remote_code": bool(trust_remote_code)}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        self.model = AutoModel.from_pretrained(self.model_name, **kwargs).to(self.device).eval()

    @torch.no_grad()
    def encode_images(self, records: Sequence[ImageTextRecord]):
        images = [_image_array_for_processor(rec.path) for rec in records]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = _to_device(inputs, self.device)
        with torch.inference_mode():
            if hasattr(self.model, "get_image_features"):
                feats = self.model.get_image_features(**inputs)
            else:
                feats = _pooled_output(self.model(**inputs))
        return _normalize_rows(feats, self.normalize)

    @torch.no_grad()
    def encode_texts(self, records: Sequence[ImageTextRecord]):
        captions = [rec.caption for rec in records]
        inputs = self.processor(text=captions, padding=True, truncation=True,
                                return_tensors="pt")
        inputs = _to_device(inputs, self.device)
        with torch.inference_mode():
            if hasattr(self.model, "get_text_features"):
                feats = self.model.get_text_features(**inputs)
            else:
                try:
                    feats = _pooled_output(self.model(**inputs))
                except Exception as e:
                    raise RuntimeError(
                        f"model {self.model_name!r} did not expose text features; "
                        "use --features image for image-only encoders"
                    ) from e
        return _normalize_rows(feats, self.normalize)


def make_embedder(backend="stats", model="", device="cpu", dtype="auto", normalize=True,
                  stats_dim=64, stats_image_size=32, trust_remote_code=False):
    backend = str(backend)
    if backend == "stats":
        return StatsEmbedder(dim=stats_dim, image_size=stats_image_size,
                             normalize=normalize)
    if backend == "hf":
        return HFEmbedder(model, device=device, dtype=dtype, normalize=normalize,
                          trust_remote_code=trust_remote_code)
    raise ValueError(f"unknown embedding backend {backend!r}")


def write_embedding_sidecar(records, out_path, root="", embedder=None, features="both",
                            batch=32):
    features = str(features)
    if features not in FEATURE_CHOICES:
        raise ValueError(f"features must be one of {FEATURE_CHOICES}")
    if embedder is None:
        embedder = StatsEmbedder()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows_written = 0
    text_dims, image_dims = set(), set()
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in _batched(list(records), batch):
            image_feats = None
            text_feats = None
            if _needs_image(features):
                image_feats = embedder.encode_images(chunk)
            if _needs_text(features):
                text_feats = embedder.encode_texts(chunk)
            for i, rec in enumerate(chunk):
                row = {
                    "image": _manifest_image_path(rec.path, root=root),
                    "caption": rec.caption,
                    "split": rec.split,
                    "embedding_backend": getattr(embedder, "backend_name", "unknown"),
                    "embedding_model": getattr(embedder, "model_name", ""),
                }
                if rec.width:
                    row["width"] = int(rec.width)
                if rec.height:
                    row["height"] = int(rec.height)
                if text_feats is not None:
                    vals = _float_list(text_feats[i])
                    row["text_embedding"] = vals
                    text_dims.add(len(vals))
                if image_feats is not None:
                    vals = _float_list(image_feats[i])
                    row["image_embedding"] = vals
                    image_dims.add(len(vals))
                f.write(json.dumps(row, sort_keys=True) + "\n")
                rows_written += 1
    report = {
        "embedding_sidecar": out_path,
        "records": int(rows_written),
        "features": features,
        "backend": getattr(embedder, "backend_name", "unknown"),
        "model": getattr(embedder, "model_name", ""),
        "batch": int(batch),
        "text_embedding_records": int(rows_written if _needs_text(features) else 0),
        "image_embedding_records": int(rows_written if _needs_image(features) else 0),
    }
    if text_dims:
        report["text_embedding_dims"] = sorted(text_dims)
    if image_dims:
        report["image_embedding_dims"] = sorted(image_dims)
    return report


def selftest():
    with tempfile.TemporaryDirectory() as td:
        img_a = os.path.join(td, "a.ppm")
        img_b = os.path.join(td, "b.ppm")
        arr_a = np.zeros((8, 10, 3), dtype=np.uint8)
        arr_a[:, :, 0] = 255
        arr_b = np.zeros((8, 10, 3), dtype=np.uint8)
        arr_b[:, :, 1] = 255
        for path, arr in ((img_a, arr_a), (img_b, arr_b)):
            with open(path, "wb") as f:
                f.write(b"P6\n10 8\n255\n")
                f.write(arr.tobytes())
        manifest = os.path.join(td, "manifest.jsonl")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps({"image": "a.ppm", "caption": "red block",
                                "split": "train"}) + "\n")
            f.write(json.dumps({"image": "b.ppm", "caption": "green block",
                                "split": "eval"}) + "\n")
        out = os.path.join(td, "embeddings.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            report = main([
                "--manifest", manifest,
                "--root", td,
                "--backend", "stats",
                "--features", "both",
                "--stats-dim", "16",
                "--out", out,
            ])
        assert report["records"] == 2
        assert report["text_embedding_dims"] == [16]
        assert report["image_embedding_dims"] == [16]
        with open(out, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert rows[0]["image"] == "a.ppm"
        assert len(rows[0]["text_embedding"]) == 16
        assert len(rows[0]["image_embedding"]) == 16
        assert abs(sum(x * x for x in rows[0]["text_embedding"]) - 1.0) < 1e-4
        assert abs(sum(x * x for x in rows[0]["image_embedding"]) - 1.0) < 1e-4
        image_only = os.path.join(td, "image_only.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            image_report = main([
                "--manifest", manifest,
                "--root", td,
                "--split", "train",
                "--features", "image",
                "--stats-dim", "8",
                "--out", image_only,
            ])
        assert image_report["records"] == 1
        with open(image_only, "r", encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert "image_embedding" in row and "text_embedding" not in row
    print("image_embed selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", default="", help="JSONL/CSV/TSV captioned image manifest")
    ap.add_argument("--root", default="", help="base directory for relative manifest paths")
    ap.add_argument("--split", default="",
                    help="optional split filter; default embeds all splits")
    ap.add_argument("--min-aesthetic", type=float, default=None,
                    help="skip rows with aesthetic/score/quality below this threshold")
    ap.add_argument("--max-records", type=int, default=0,
                    help="cap records for smoke tests; 0 means all")
    ap.add_argument("--out", default="", help="output sidecar JSONL path")
    ap.add_argument("--report-out", default="", help="optional JSON report path")
    ap.add_argument("--backend", default="stats", choices=("stats", "hf"),
                    help="embedding backend; use hf for real CLIP/SigLIP/DINO/MAE jobs")
    ap.add_argument("--model", default="",
                    help="Hugging Face model id for --backend hf")
    ap.add_argument("--features", default="both", choices=FEATURE_CHOICES,
                    help="which embedding columns to write")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    ap.add_argument("--no-normalize", action="store_true",
                    help="do not L2-normalize embedding rows")
    ap.add_argument("--stats-dim", type=int, default=64,
                    help="embedding width for the dependency-free stats backend")
    ap.add_argument("--stats-image-size", type=int, default=32,
                    help="image resize used by the stats backend")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="pass trust_remote_code=True to Hugging Face loaders")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return None
    if not args.manifest:
        ap.error("use --selftest or --manifest")
    if not args.out:
        ap.error("--out is required")
    records = read_image_manifest(
        args.manifest, root=args.root, split=args.split,
        min_aesthetic=args.min_aesthetic, max_records=args.max_records)
    embedder = make_embedder(
        backend=args.backend, model=args.model, device=args.device,
        dtype=args.dtype, normalize=not args.no_normalize,
        stats_dim=args.stats_dim, stats_image_size=args.stats_image_size,
        trust_remote_code=args.trust_remote_code)
    report = write_embedding_sidecar(
        records, args.out, root=args.root, embedder=embedder,
        features=args.features, batch=args.batch)
    report.update({
        "manifest": args.manifest,
        "root": args.root,
        "split": args.split,
        "min_aesthetic": float(args.min_aesthetic) if args.min_aesthetic is not None else None,
        "max_records": int(args.max_records),
        "normalize": not args.no_normalize,
    })
    text = json.dumps(report, indent=1)
    print(text)
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"saved -> {args.report_out}")
    return report


if __name__ == "__main__":
    main()
