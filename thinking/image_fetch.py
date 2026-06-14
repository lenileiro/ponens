"""Fetch captioned web image shards into the local manifest format.

This is intentionally dependency-light.  It streams WebDataset tar shards from HTTPS or local
paths, extracts image/json pairs, and writes a JSONL manifest consumable by thinking.image_data and
thinking.image_latent.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from urllib.parse import quote
from urllib.request import Request, urlopen

from .image_data import caption_tokens, image_dimensions_from_bytes


TEXT_TO_IMAGE_2M_1024_10K = (
    "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/"
    "data_1024_10K/data_000000.tar"
)
DIFFUSIONDB_PART_URL = (
    "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/"
    "images/part-{part:06d}.zip"
)
DIFFUSIONDB_METADATA_PARQUET = (
    "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
)
DIFFUSIONDB_2M_PARTS = 2000

HF_DATASET_TREE_API = "https://huggingface.co/api/datasets/{repo}/tree/main?recursive=1"
HF_DATASET_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

STATIC_SOURCES = {
    "text-to-image-2m-1024-10k": (TEXT_TO_IMAGE_2M_1024_10K,),
    "flux-1024-10k": (TEXT_TO_IMAGE_2M_1024_10K,),
}
HF_DATASET_SOURCES = {
    "text-to-image-2m-512-2m": {
        "repo": "jackyhate/text-to-image-2M",
        "prefix": "data_512_2M/",
    },
}
ZIP_SOURCES = {
    "diffusiondb-2m": {
        "parts": DIFFUSIONDB_2M_PARTS,
        "url": DIFFUSIONDB_PART_URL,
    },
}
SOURCES = tuple(sorted(set(STATIC_SOURCES) | set(HF_DATASET_SOURCES) | set(ZIP_SOURCES)))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".ppm", ".pnm", ".bmp"}
META_EXTS = {".json"}
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_name(name):
    base = os.path.basename(str(name))
    base = SAFE_NAME_RE.sub("_", base).strip("._")
    return base or "image"


def open_stream(url, timeout=60):
    if os.path.exists(url):
        return open(url, "rb")
    req = Request(str(url), headers={"User-Agent": "thinking-image-fetch/0.1"})
    return urlopen(req, timeout=float(timeout))


def materialize_path(url, tmpdir, timeout=60):
    if os.path.exists(url):
        return os.path.abspath(url)
    os.makedirs(tmpdir, exist_ok=True)
    suffix = os.path.splitext(str(url).split("?", 1)[0])[1] or ".bin"
    fd, path = tempfile.mkstemp(prefix="image_fetch_", suffix=suffix, dir=tmpdir)
    os.close(fd)
    try:
        with open_stream(url, timeout=timeout) as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def hf_dataset_tar_urls(repo, prefix, timeout=60):
    api = HF_DATASET_TREE_API.format(repo=quote(str(repo), safe="/"))
    with open_stream(api, timeout=timeout) as stream:
        files = json.load(stream)
    paths = []
    for item in files:
        path = str(item.get("path", ""))
        if item.get("type") != "file":
            continue
        if not path.startswith(prefix) or not path.endswith(".tar"):
            continue
        paths.append(path)
    if not paths:
        raise ValueError(f"no tar shards found for Hugging Face dataset {repo!r} prefix {prefix!r}")
    return tuple(
        HF_DATASET_RESOLVE.format(repo=quote(str(repo), safe="/"), path=quote(path, safe="/"))
        for path in sorted(paths)
    )


def diffusiondb_part_urls(start_part=1, end_part=0, source="diffusiondb-2m"):
    spec = ZIP_SOURCES[source]
    max_part = int(spec["parts"])
    start_part = max(1, int(start_part or 1))
    end_part = int(end_part or max_part)
    if end_part <= 0:
        end_part = max_part
    end_part = min(max_part, end_part)
    if start_part > end_part:
        return ()
    return tuple(spec["url"].format(part=part) for part in range(start_part, end_part + 1))


def shard_urls(source="", urls=(), timeout=60):
    out = []
    if source:
        if source in STATIC_SOURCES:
            out.extend(STATIC_SOURCES[source])
        elif source in HF_DATASET_SOURCES:
            spec = HF_DATASET_SOURCES[source]
            out.extend(hf_dataset_tar_urls(
                spec["repo"], spec["prefix"], timeout=timeout))
        else:
            raise ValueError(f"unknown image fetch source {source!r}; choices={list(SOURCES)}")
    out.extend(urls or ())
    if not out:
        raise ValueError("provide --source or one or more --url shards")
    return tuple(out)


def load_diffusiondb_metadata(path_or_url, parts=(), timeout=60):
    if not path_or_url:
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "--diffusiondb-metadata requires pyarrow; omit the flag or install pyarrow"
        ) from exc
    parts = {int(p) for p in parts if int(p) > 0}
    with tempfile.TemporaryDirectory() as td:
        path = materialize_path(path_or_url, td, timeout=timeout)
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        wanted = [
            "image_name", "prompt", "part_id", "width", "height",
            "image_nsfw", "prompt_nsfw", "seed", "step", "cfg", "sampler",
        ]
        columns = [col for col in wanted if col in available]
        filters = [("part_id", "in", sorted(parts))] if parts and "part_id" in available else None
        table = pq.read_table(path, columns=columns or None, filters=filters)
        out = {}
        for row in table.to_pylist():
            image_name = row.get("image_name") or row.get("image") or row.get("file_name")
            if not image_name:
                continue
            base = os.path.basename(str(image_name))
            meta = dict(row)
            if "prompt" in meta and "p" not in meta:
                meta["p"] = meta["prompt"]
            out[base] = meta
        return out


def caption_from_meta(meta):
    for key in ("caption", "text", "prompt", "description", "p"):
        val = meta.get(key)
        if val not in ("", None):
            return str(val)
    return ""


def meta_float(meta, *keys, mode="first"):
    vals = []
    for key in keys:
        val = meta.get(key)
        if val in ("", None):
            continue
        if isinstance(val, str):
            low = val.strip().lower()
            if not low:
                continue
            if low in ("true", "yes"):
                val = 1.0
            elif low in ("false", "no"):
                val = 0.0
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
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


def meta_int(meta, *keys):
    val = meta_float(meta, *keys)
    if val is None:
        return 0
    return int(val)


def image_base(member_name):
    stem, ext = os.path.splitext(os.path.basename(member_name))
    return stem, ext.lower()


def rel_image_path(path, root=""):
    if not root:
        return path
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return path
    if rel.startswith("..") or os.path.isabs(rel):
        return path
    return rel


def valid_caption(caption, min_tokens=1, max_tokens=0):
    toks = caption_tokens(caption)
    if len(toks) < int(min_tokens):
        return False
    if int(max_tokens or 0) > 0 and len(toks) > int(max_tokens):
        return False
    return True


def maybe_emit_pair(base, pending, image_dir, root, source_name, rows, args):
    item = pending.get(base)
    if not item or "image" not in item or "meta" not in item:
        return False
    meta = item["meta"]
    caption = caption_from_meta(meta)
    if not valid_caption(caption, args.min_caption_tokens, args.max_caption_tokens):
        pending.pop(base, None)
        return False
    image_name = safe_name(item["image_name"])
    _stem, ext = image_base(image_name)
    width = meta_int(meta, "width", "w")
    height = meta_int(meta, "height", "h")
    if width <= 0 or height <= 0:
        try:
            width, height = image_dimensions_from_bytes(item["image"], ext=ext)
        except Exception:
            width, height = 0, 0
    out_path = os.path.join(image_dir, image_name)
    if args.overwrite or not os.path.exists(out_path):
        with open(out_path, "wb") as f:
            f.write(item["image"])
    row = {
        "image": rel_image_path(out_path, root=root),
        "caption": caption,
        "split": "train",
        "source": source_name,
    }
    if width > 0 and height > 0:
        row["width"] = int(width)
        row["height"] = int(height)
    aesthetic = meta_float(
        meta, "aesthetic", "aesthetic_score", "score", "quality", "quality_score")
    nsfw = meta_float(
        meta, "nsfw", "image_nsfw", "prompt_nsfw", "unsafe", "safety_score",
        mode="max")
    watermark = meta_float(
        meta, "watermark", "watermark_score", "has_watermark", "text_watermark",
        mode="max")
    if aesthetic is not None:
        row["aesthetic"] = float(aesthetic)
    if nsfw is not None:
        row["nsfw"] = float(nsfw)
    if watermark is not None:
        row["watermark"] = float(watermark)
    rows.append(row)
    pending.pop(base, None)
    return True


def finalize_rows(rows, manifest, stats, args):
    rng = random.Random(int(args.seed))
    order = list(range(len(rows)))
    rng.shuffle(order)
    eval_n = int(round(len(rows) * float(args.eval_frac)))
    eval_ids = set(order[:eval_n])
    for i, row in enumerate(rows):
        row["split"] = "eval" if i in eval_ids else "train"
    with open(manifest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    stats["records_written"] = len(rows)
    stats["train_records"] = sum(1 for r in rows if r["split"] == "train")
    stats["eval_records"] = sum(1 for r in rows if r["split"] == "eval")
    stats["dimension_records"] = sum(
        1 for r in rows if int(r.get("width") or 0) > 0 and int(r.get("height") or 0) > 0)
    stats["aesthetic_records"] = sum(1 for r in rows if r.get("aesthetic") is not None)
    stats["nsfw_records"] = sum(1 for r in rows if r.get("nsfw") is not None)
    stats["watermark_records"] = sum(1 for r in rows if r.get("watermark") is not None)
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
    return stats


def fetch_webdataset_manifest(args):
    urls = shard_urls(args.source, args.url, timeout=args.timeout)
    image_dir = os.path.abspath(args.image_dir)
    manifest = os.path.abspath(args.manifest)
    root = os.path.abspath(args.root) if args.root else os.path.dirname(manifest)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)
    max_records = int(args.max_records or 0)
    rows = []
    stats = {
        "source": args.source,
        "urls": list(urls),
        "image_dir": image_dir,
        "manifest": manifest,
        "root": root,
        "max_records": max_records,
        "images_seen": 0,
        "meta_seen": 0,
        "records_written": 0,
    }
    for url in urls:
        pending = defaultdict(dict)
        with open_stream(url, timeout=args.timeout) as stream:
            with tarfile.open(fileobj=stream, mode="r|*") as tf:
                for member in tf:
                    if max_records and len(rows) >= max_records:
                        break
                    if not member.isfile():
                        continue
                    base, ext = image_base(member.name)
                    if ext not in IMAGE_EXTS and ext not in META_EXTS:
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                    if ext in IMAGE_EXTS:
                        pending[base]["image"] = data
                        pending[base]["image_name"] = member.name
                        stats["images_seen"] += 1
                    else:
                        pending[base]["meta"] = json.loads(data.decode("utf-8"))
                        stats["meta_seen"] += 1
                    maybe_emit_pair(base, pending, image_dir, root, args.source or "webdataset",
                                    rows, args)
        if max_records and len(rows) >= max_records:
            break
    return finalize_rows(rows, manifest, stats, args)


def load_zip_json_metadata(zf):
    meta = {}
    for info in zf.infolist():
        if info.is_dir() or not info.filename.endswith(".json"):
            continue
        with zf.open(info) as f:
            data = json.loads(f.read().decode("utf-8"))
        if not isinstance(data, dict):
            continue
        for key, val in data.items():
            if isinstance(val, dict):
                meta[os.path.basename(str(key))] = dict(val)
    return meta


def diffusiondb_part_range(args):
    start_part = max(1, int(args.diffusiondb_start_part or 1))
    end_part = int(args.diffusiondb_end_part or DIFFUSIONDB_2M_PARTS)
    end_part = min(DIFFUSIONDB_2M_PARTS, end_part)
    if start_part > end_part:
        return range(0)
    return range(start_part, end_part + 1)


def fetch_diffusiondb_manifest(args):
    if args.url:
        urls = tuple(args.url)
        parts = ()
    else:
        parts = tuple(diffusiondb_part_range(args))
        urls = diffusiondb_part_urls(
            start_part=args.diffusiondb_start_part,
            end_part=args.diffusiondb_end_part,
            source=args.source or "diffusiondb-2m")
    image_dir = os.path.abspath(args.image_dir)
    manifest = os.path.abspath(args.manifest)
    root = os.path.abspath(args.root) if args.root else os.path.dirname(manifest)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)
    max_records = int(args.max_records or 0)
    rows = []
    part_meta = {}
    if args.diffusiondb_metadata:
        part_meta = load_diffusiondb_metadata(args.diffusiondb_metadata, parts=parts,
                                              timeout=args.timeout)
    stats = {
        "source": args.source or "diffusiondb-zip",
        "format": "diffusiondb_zip",
        "urls": list(urls[:8]) + (["..."] if len(urls) > 8 else []),
        "image_dir": image_dir,
        "manifest": manifest,
        "root": root,
        "max_records": max_records,
        "images_seen": 0,
        "meta_seen": 0,
        "records_written": 0,
        "diffusiondb_metadata_records": len(part_meta),
    }
    with tempfile.TemporaryDirectory() as td:
        for url in urls:
            if max_records and len(rows) >= max_records:
                break
            zip_path = materialize_path(url, td, timeout=args.timeout)
            with zipfile.ZipFile(zip_path) as zf:
                zip_meta = load_zip_json_metadata(zf)
                stats["meta_seen"] += len(zip_meta)
                for info in sorted(zf.infolist(), key=lambda x: x.filename):
                    if max_records and len(rows) >= max_records:
                        break
                    if info.is_dir():
                        continue
                    base, ext = image_base(info.filename)
                    if ext not in IMAGE_EXTS:
                        continue
                    stats["images_seen"] += 1
                    image_name = os.path.basename(info.filename)
                    meta = {}
                    meta.update(zip_meta.get(image_name, {}))
                    meta.update(zip_meta.get(base, {}))
                    meta.update(part_meta.get(image_name, {}))
                    caption = caption_from_meta(meta)
                    if not valid_caption(caption, args.min_caption_tokens,
                                         args.max_caption_tokens):
                        continue
                    data = zf.read(info)
                    pending = defaultdict(dict)
                    pending[base]["image"] = data
                    pending[base]["image_name"] = image_name
                    pending[base]["meta"] = meta
                    maybe_emit_pair(base, pending, image_dir, root,
                                    args.source or "diffusiondb-zip", rows, args)
    return finalize_rows(rows, manifest, stats, args)


def fetch_manifest(args):
    fetch_format = str(getattr(args, "format", "auto") or "auto")
    urls = tuple(getattr(args, "url", ()) or ())
    auto_zip = (
        fetch_format == "auto"
        and not getattr(args, "source", "")
        and urls
        and all(str(url).split("?", 1)[0].lower().endswith(".zip") for url in urls)
    )
    if fetch_format == "diffusiondb-zip" or args.source in ZIP_SOURCES or auto_zip:
        return fetch_diffusiondb_manifest(args)
    return fetch_webdataset_manifest(args)


def _write_ppm(path):
    with open(path, "wb") as f:
        f.write(b"P6\n2 2\n255\n")
        f.write(bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]))


def selftest():
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "mini.tar")
        img = os.path.join(td, "sample.ppm")
        _write_ppm(img)
        meta = os.path.join(td, "sample.json")
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({
                "prompt": "red green blue white test image",
                "width": 2,
                "height": 2,
                "aesthetic": 6.5,
                "image_nsfw": 0.1,
                "watermark_score": 0.0,
            }, f)
        with tarfile.open(tar_path, "w") as tf:
            tf.add(img, arcname="sample.ppm")
            tf.add(meta, arcname="sample.json")
        args = argparse.Namespace(
            source="", url=[tar_path], image_dir=os.path.join(td, "images"),
            manifest=os.path.join(td, "manifest.jsonl"), root=os.path.join(td, "images"),
            format="auto", diffusiondb_start_part=1, diffusiondb_end_part=1,
            diffusiondb_metadata="",
            max_records=1, min_caption_tokens=2, max_caption_tokens=0,
            eval_frac=0.0, seed=0, timeout=10, overwrite=False, report_out="")
        stats = fetch_manifest(args)
        assert stats["records_written"] == 1
        row = json.loads(open(args.manifest, encoding="utf-8").readline())
        assert row["caption"].startswith("red green")
        assert row["width"] == 2 and row["height"] == 2
        assert row["aesthetic"] == 6.5 and row["nsfw"] == 0.1
        assert os.path.exists(os.path.join(args.root, row["image"]))
        zip_path = os.path.join(td, "diffusiondb.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(img, arcname="part-000001/sample.ppm")
            zf.writestr("part-000001/part-000001.json", json.dumps({
                "sample.ppm": {
                    "p": "small zip caption for diffusiondb",
                    "width": 2,
                    "height": 2,
                    "prompt_nsfw": 0.0,
                }
            }))
        args.source = ""
        args.url = [zip_path]
        args.format = "diffusiondb-zip"
        args.manifest = os.path.join(td, "diffusiondb.jsonl")
        args.image_dir = os.path.join(td, "zip_images")
        stats = fetch_manifest(args)
        assert stats["records_written"] == 1
        row = json.loads(open(args.manifest, encoding="utf-8").readline())
        assert row["caption"].startswith("small zip")
        assert row["nsfw"] == 0.0
    print("image_fetch selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="", choices=("",) + SOURCES,
                    help="known captioned-image source to stream")
    ap.add_argument("--format", default="auto", choices=("auto", "webdataset", "diffusiondb-zip"),
                    help="source format for custom --url inputs")
    ap.add_argument("--url", action="append", default=[],
                    help="additional WebDataset tar, DiffusionDB zip URL, or local path")
    ap.add_argument("--image-dir", default="data/images/web_fetch",
                    help="directory for extracted images")
    ap.add_argument("--manifest", default="data/images/train_web.jsonl",
                    help="output JSONL manifest path")
    ap.add_argument("--root", default="data/images",
                    help="root used to make manifest image paths relative")
    ap.add_argument("--max-records", type=int, default=128,
                    help="stop after this many image/caption pairs; 0 means all streamed shards")
    ap.add_argument("--eval-frac", type=float, default=0.05,
                    help="fraction of downloaded rows marked split=eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--min-caption-tokens", type=int, default=3)
    ap.add_argument("--max-caption-tokens", type=int, default=0)
    ap.add_argument("--diffusiondb-start-part", type=int, default=1,
                    help="first DiffusionDB part to fetch for --source diffusiondb-2m")
    ap.add_argument("--diffusiondb-end-part", type=int, default=0,
                    help="last DiffusionDB part to fetch; 0 means all available parts")
    ap.add_argument("--diffusiondb-metadata", default="",
                    help=("optional local path or URL to DiffusionDB metadata.parquet; "
                          f"official URL: {DIFFUSIONDB_METADATA_PARQUET}"))
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing image files")
    ap.add_argument("--report-out", default="", help="optional JSON report path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.max_records < 0:
        ap.error("--max-records must be non-negative")
    if args.eval_frac < 0.0 or args.eval_frac >= 1.0:
        ap.error("--eval-frac must be in [0, 1)")
    if args.min_caption_tokens < 0:
        ap.error("--min-caption-tokens must be non-negative")
    if args.max_caption_tokens < 0:
        ap.error("--max-caption-tokens must be non-negative")
    if args.diffusiondb_start_part <= 0:
        ap.error("--diffusiondb-start-part must be positive")
    if args.diffusiondb_end_part < 0:
        ap.error("--diffusiondb-end-part must be non-negative")
    if args.format == "webdataset" and args.source in ZIP_SOURCES:
        ap.error("--format webdataset is incompatible with DiffusionDB sources")
    stats = fetch_manifest(args)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
