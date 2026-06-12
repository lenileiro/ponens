"""Captioned image manifest utilities for real image-generation rungs.

The synthetic image rungs are useful for factor probes, but high-quality image generation needs
large image/text corpora.  This module is deliberately light on dependencies:

* PPM is supported directly for tests and tiny local fixtures.
* JPEG/PNG/WebP use Pillow when it is installed on the GPU box.
* Manifests are JSONL, CSV, or TSV with at least an image path and caption/text field.
"""
from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
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


def caption_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]


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
    return ImageTextRecord(
        path=os.path.normpath(path),
        caption=str(caption),
        split=str(row.get("split") or "train"),
        aesthetic=aesthetic,
        width=width,
        height=height,
    )


def read_image_manifest(path, root="", split="train", min_aesthetic=None, max_records=0):
    """Read captioned image records from JSONL/CSV/TSV.

    JSONL fields: image/path/file, caption/text/prompt, optional split/aesthetic/width/height.
    TSV rows are interpreted as: path<TAB>caption[<TAB>split].
    """
    manifest_dir = os.path.dirname(os.path.abspath(path))
    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
            if has_header:
                rows = list(csv.DictReader(f, delimiter=delimiter))
            else:
                rows = [
                    {"path": r[0], "caption": r[1], "split": r[2] if len(r) > 2 else "train"}
                    for r in csv.reader(f, delimiter=delimiter) if len(r) >= 2
                ]
    else:
        raise ValueError(f"unsupported image manifest format {ext!r}; use .jsonl, .csv, or .tsv")

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
    for rec in rows:
        splits[rec.split] = splits.get(rec.split, 0) + 1
        if rec.aesthetic is not None:
            aesthetic.append(float(rec.aesthetic))
    out = {
        "image_records": len(rows),
        "image_splits": splits,
        "caption_token_mean": float(np.mean([len(caption_tokens(r.caption)) for r in rows])),
    }
    if aesthetic:
        out.update({
            "aesthetic_mean": float(np.mean(aesthetic)),
            "aesthetic_min": float(np.min(aesthetic)),
            "aesthetic_max": float(np.max(aesthetic)),
        })
    return out


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


def _pil_image(path):
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover - depends on optional runtime package.
        raise ImportError(
            "JPEG/PNG/WebP image loading requires Pillow. Install it with `pip install pillow` "
            "or use PPM fixtures for dependency-free tests."
        ) from e
    return Image.open(path).convert("RGB")


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


def sample_image_text_batch(records, rng, batch=32, size=256, device="cpu"):
    records = list(records)
    idx = rng.integers(0, len(records), size=int(batch))
    chosen = [records[int(i)] for i in idx]
    imgs = [load_image_tensor(rec.path, size=size, device=device) for rec in chosen]
    captions = [rec.caption for rec in chosen]
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
                "aesthetic": 7.5,
            }) + "\n")
        records = read_image_manifest(manifest)
        assert len(records) == 1 and records[0].caption.startswith("red")
        x = load_image_tensor(records[0].path, size=4)
        assert x.shape == (3, 4, 4) and float(x.max()) <= 1.0 and float(x.min()) >= -1.0
        vocab = build_caption_vocab(records)
        ids = caption_ids([records[0].caption], vocab, max_len=5)
        assert ids.shape == (1, 5) and int(ids[0, 0]) > 0
        xb, captions = sample_image_text_batch(records, np.random.default_rng(0), batch=2, size=4)
        assert xb.shape == (2, 3, 4, 4) and captions == [records[0].caption] * 2
        summary = summarize_records(records)
        assert summary["image_records"] == 1 and summary["image_splits"]["train"] == 1
    print("image_data selftest OK")
