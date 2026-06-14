"""Build generic text/image embedding sidecars for image manifests.

High-quality image generation needs semantically rich conditioning and visual-latent
alignment.  This module keeps that preprocessing outside the generator: it reads a
captioned image manifest and writes a JSONL sidecar with pooled `text_embedding`,
token-level `text_embedding_sequence`, and/or `image_embedding` arrays that
`thinking.image_data` can merge into the cleaned training manifest.

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
TEXT_MODE_CHOICES = ("pooled", "tokens", "both")


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


def _float_matrix(rows):
    return [[float(x) for x in row] for row in rows.detach().cpu().float().tolist()]


def _normalize_sequence_rows(rows, enabled=True):
    out = [row.float() for row in rows]
    if not enabled:
        return out
    return [F.normalize(row, dim=-1, eps=1e-8) for row in out]


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


def _dtype_from_name(dtype):
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    if dtype == "auto":
        return None
    raise ValueError("dtype must be auto, fp32, fp16, or bf16")


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


def _token_output(outputs):
    text_out = getattr(outputs, "text_model_output", None)
    if text_out is not None and getattr(text_out, "last_hidden_state", None) is not None:
        return text_out.last_hidden_state
    last = getattr(outputs, "last_hidden_state", None)
    if last is not None and last.ndim == 3:
        return last
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if getattr(first, "ndim", 0) == 3:
            return first
    raise RuntimeError("could not find token embeddings in model output")


def _masked_mean(tokens, mask=None):
    if mask is None:
        return tokens.mean(dim=1)
    keep = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)


@dataclass
class StatsEmbedder:
    dim: int = 64
    image_size: int = 32
    normalize: bool = True
    text_mode: str = "pooled"

    backend_name: str = "stats"
    model_name: str = "deterministic-stats"

    def encode_images(self, records: Sequence[ImageTextRecord]):
        rows = [_stats_image_embedding(rec.path, dim=self.dim, size=self.image_size)
                for rec in records]
        return _normalize_rows(torch.stack(rows, dim=0), self.normalize)

    def encode_texts(self, records: Sequence[ImageTextRecord]):
        if self.text_mode in ("tokens", "both"):
            rows = []
            for rec in records:
                toks = caption_tokens(rec.caption) or ["<empty>"]
                rows.append(torch.stack([
                    _hash_caption_embedding(tok, dim=self.dim) for tok in toks
                ], dim=0))
            token_rows = _normalize_sequence_rows(rows, self.normalize)
            if self.text_mode == "tokens":
                return token_rows
            pooled = [_hash_caption_embedding(rec.caption, dim=self.dim) for rec in records]
            return {
                "pooled": _normalize_rows(torch.stack(pooled, dim=0), self.normalize),
                "tokens": token_rows,
            }
        rows = [_hash_caption_embedding(rec.caption, dim=self.dim) for rec in records]
        return _normalize_rows(torch.stack(rows, dim=0), self.normalize)


class HFTextEncoder:
    """Token/pooled text features from a Hugging Face text encoder."""

    def __init__(self, model_name, device="cpu", dtype="auto", normalize=True,
                 trust_remote_code=False, max_length=0):
        if not model_name:
            raise ValueError("text sequence model is empty")
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except Exception as e:  # pragma: no cover - optional GPU dependency.
            raise ImportError(
                "Hugging Face text embedding requires transformers. Install it on the GPU box "
                "with `pip install transformers accelerate sentencepiece`."
            ) from e
        self.backend_name = "hf_text"
        self.model_name = str(model_name)
        self.device = torch.device(device)
        self.normalize = bool(normalize)
        self.max_length = int(max_length or 0)
        torch_dtype = _dtype_from_name(dtype)
        common_kwargs = {"trust_remote_code": bool(trust_remote_code)}
        model_kwargs = dict(common_kwargs)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        config = AutoConfig.from_pretrained(self.model_name, **common_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **common_kwargs)
        if getattr(config, "model_type", "") in ("t5", "mt5", "umt5"):
            try:
                from transformers import T5EncoderModel
                self.model = T5EncoderModel.from_pretrained(
                    self.model_name, **model_kwargs).to(self.device).eval()
            except Exception:
                self.model = AutoModel.from_pretrained(
                    self.model_name, **model_kwargs).to(self.device).eval()
        else:
            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs).to(
                self.device).eval()

    def _tokenize(self, captions):
        kwargs = {"padding": True, "truncation": True, "return_tensors": "pt"}
        if self.max_length > 0:
            kwargs["max_length"] = self.max_length
        return self.tokenizer(captions, **kwargs)

    @torch.no_grad()
    def encode_texts(self, records: Sequence[ImageTextRecord], text_mode="tokens"):
        captions = [rec.caption for rec in records]
        inputs = self._tokenize(captions)
        inputs = _to_device(inputs, self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            token_feats = _token_output(outputs)
        mask = inputs.get("attention_mask")
        rows = []
        for i in range(token_feats.shape[0]):
            length = (
                int(mask[i].detach().sum().cpu()) if mask is not None
                else int(token_feats.shape[1])
            )
            rows.append(token_feats[i, :max(1, length)].detach())
        token_rows = _normalize_sequence_rows(rows, self.normalize)
        if text_mode == "tokens":
            return token_rows
        pooled = _masked_mean(token_feats, mask=mask)
        if text_mode == "both":
            return {
                "pooled": _normalize_rows(pooled, self.normalize),
                "tokens": token_rows,
            }
        if text_mode == "pooled":
            return _normalize_rows(pooled, self.normalize)
        raise ValueError(f"text_mode must be one of {TEXT_MODE_CHOICES}")


class HFEmbedder:
    def __init__(self, model_name, device="cpu", dtype="auto", normalize=True,
                 text_mode="pooled", trust_remote_code=False,
                 text_sequence_model="", text_sequence_max_length=0):
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
        self.text_mode = str(text_mode)
        self.text_sequence_model_name = str(text_sequence_model or "")
        if self.text_mode not in TEXT_MODE_CHOICES:
            raise ValueError(f"text_mode must be one of {TEXT_MODE_CHOICES}")
        torch_dtype = _dtype_from_name(dtype)
        kwargs = {"trust_remote_code": bool(trust_remote_code)}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        self.model = AutoModel.from_pretrained(self.model_name, **kwargs).to(self.device).eval()
        self.text_sequence_encoder = None
        if self.text_sequence_model_name:
            self.text_sequence_encoder = HFTextEncoder(
                self.text_sequence_model_name, device=device, dtype=dtype,
                normalize=normalize, trust_remote_code=trust_remote_code,
                max_length=text_sequence_max_length)

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
            external_token_rows = None
            if self.text_sequence_encoder is not None and self.text_mode in ("tokens", "both"):
                external_token_rows = self.text_sequence_encoder.encode_texts(
                    records, text_mode="tokens")
                if self.text_mode == "tokens":
                    return external_token_rows
            if self.text_mode in ("tokens", "both"):
                outputs = self.model(**inputs)
                if external_token_rows is None:
                    token_feats = _token_output(outputs)
                    mask = inputs.get("attention_mask")
                    rows = []
                    for i in range(token_feats.shape[0]):
                        length = (
                            int(mask[i].detach().sum().cpu()) if mask is not None
                            else int(token_feats.shape[1])
                        )
                        rows.append(token_feats[i, :max(1, length)].detach())
                    token_rows = _normalize_sequence_rows(rows, self.normalize)
                else:
                    token_rows = external_token_rows
                if self.text_mode == "tokens":
                    return token_rows
                if hasattr(self.model, "get_text_features"):
                    pooled = self.model.get_text_features(**inputs)
                else:
                    pooled = _pooled_output(outputs)
                return {
                    "pooled": _normalize_rows(pooled, self.normalize),
                    "tokens": token_rows,
                }
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
                  stats_dim=64, stats_image_size=32, text_mode="pooled",
                  trust_remote_code=False, text_sequence_model="",
                  text_sequence_max_length=0):
    if text_mode not in TEXT_MODE_CHOICES:
        raise ValueError(f"text_mode must be one of {TEXT_MODE_CHOICES}")
    backend = str(backend)
    if backend == "stats":
        return StatsEmbedder(dim=stats_dim, image_size=stats_image_size,
                             normalize=normalize, text_mode=text_mode)
    if backend == "hf":
        return HFEmbedder(model, device=device, dtype=dtype, normalize=normalize,
                          text_mode=text_mode, trust_remote_code=trust_remote_code,
                          text_sequence_model=text_sequence_model,
                          text_sequence_max_length=text_sequence_max_length)
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
    text_dims, text_sequence_dims, text_sequence_lens, image_dims = set(), set(), [], set()
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
                    "text_embed_mode": getattr(embedder, "text_mode", "pooled"),
                }
                text_sequence_model = getattr(embedder, "text_sequence_model_name", "")
                if text_sequence_model:
                    row["text_sequence_model"] = text_sequence_model
                if rec.width:
                    row["width"] = int(rec.width)
                if rec.height:
                    row["height"] = int(rec.height)
                if text_feats is not None:
                    text_pooled = None
                    text_tokens = None
                    if isinstance(text_feats, dict):
                        text_pooled = text_feats.get("pooled")
                        text_tokens = text_feats.get("tokens")
                    elif isinstance(text_feats, list):
                        text_tokens = text_feats
                    else:
                        text_pooled = text_feats
                    if text_tokens is not None:
                        vals = _float_matrix(text_tokens[i])
                        row["text_embedding_sequence"] = vals
                        text_sequence_lens.append(len(vals))
                        if vals:
                            text_sequence_dims.add(len(vals[0]))
                    if text_pooled is not None:
                        vals = _float_list(text_pooled[i])
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
        "text_sequence_model": getattr(embedder, "text_sequence_model_name", ""),
        "batch": int(batch),
        "text_embed_mode": getattr(embedder, "text_mode", "pooled"),
        "text_embedding_records": (
            int(rows_written if _needs_text(features) and text_dims else 0)
        ),
        "text_embedding_sequence_records": (
            int(rows_written if _needs_text(features) and text_sequence_dims else 0)
        ),
        "image_embedding_records": int(rows_written if _needs_image(features) else 0),
    }
    if text_dims:
        report["text_embedding_dims"] = sorted(text_dims)
    if text_sequence_dims:
        report["text_embedding_sequence_dims"] = sorted(text_sequence_dims)
        report["text_embedding_sequence_len_min"] = int(min(text_sequence_lens))
        report["text_embedding_sequence_len_max"] = int(max(text_sequence_lens))
        report["text_embedding_sequence_len_mean"] = float(np.mean(text_sequence_lens))
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
        seq_out = os.path.join(td, "token_embeddings.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            seq_report = main([
                "--manifest", manifest,
                "--root", td,
                "--backend", "stats",
                "--features", "text",
                "--text-embed-mode", "tokens",
                "--stats-dim", "12",
                "--out", seq_out,
            ])
        assert seq_report["text_embedding_records"] == 0
        assert seq_report["text_embedding_sequence_records"] == 2
        assert seq_report["text_embedding_sequence_dims"] == [12]
        with open(seq_out, "r", encoding="utf-8") as f:
            seq_row = json.loads(f.readline())
        assert "text_embedding_sequence" in seq_row and "text_embedding" not in seq_row
        assert len(seq_row["text_embedding_sequence"]) == 2
        assert len(seq_row["text_embedding_sequence"][0]) == 12
        both_out = os.path.join(td, "both_text_embeddings.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            both_report = main([
                "--manifest", manifest,
                "--root", td,
                "--backend", "stats",
                "--features", "both",
                "--text-embed-mode", "both",
                "--stats-dim", "10",
                "--out", both_out,
            ])
        assert both_report["text_embedding_records"] == 2
        assert both_report["text_embedding_sequence_records"] == 2
        assert both_report["image_embedding_records"] == 2
        with open(both_out, "r", encoding="utf-8") as f:
            both_row = json.loads(f.readline())
        assert len(both_row["text_embedding"]) == 10
        assert len(both_row["text_embedding_sequence"][0]) == 10
        assert len(both_row["image_embedding"]) == 10
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
    ap.add_argument("--text-sequence-model", default="",
                    help=("optional Hugging Face text encoder for token-level "
                          "text_embedding_sequence rows; primary --model still supplies "
                          "image embeddings and pooled text embeddings"))
    ap.add_argument("--text-sequence-max-length", type=int, default=0,
                    help="optional tokenizer max_length for --text-sequence-model")
    ap.add_argument("--features", default="both", choices=FEATURE_CHOICES,
                    help="which embedding columns to write")
    ap.add_argument("--text-embed-mode", default="pooled", choices=TEXT_MODE_CHOICES,
                    help=("write pooled text_embedding vectors, token-level "
                          "text_embedding_sequence rows, or both"))
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
    if args.text_sequence_max_length < 0:
        ap.error("--text-sequence-max-length must be non-negative")
    if args.text_sequence_model and args.backend != "hf":
        ap.error("--text-sequence-model requires --backend hf")
    if args.text_sequence_model and args.text_embed_mode == "pooled":
        ap.error("--text-sequence-model requires --text-embed-mode tokens or both")
    records = read_image_manifest(
        args.manifest, root=args.root, split=args.split,
        min_aesthetic=args.min_aesthetic, max_records=args.max_records)
    embedder = make_embedder(
        backend=args.backend, model=args.model, device=args.device,
        dtype=args.dtype, normalize=not args.no_normalize,
        stats_dim=args.stats_dim, stats_image_size=args.stats_image_size,
        text_mode=args.text_embed_mode,
        trust_remote_code=args.trust_remote_code,
        text_sequence_model=args.text_sequence_model,
        text_sequence_max_length=args.text_sequence_max_length)
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
        "text_embed_mode": args.text_embed_mode,
        "text_sequence_model": args.text_sequence_model,
        "text_sequence_max_length": int(args.text_sequence_max_length),
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
