"""Generate or merge improved captions for image manifests.

Modern high-quality text-to-image training depends heavily on descriptive, image-grounded captions.
This module keeps caption improvement as a data-plane preprocessing step: it reads the same generic
image manifest format as thinking.image_data, optionally runs a Hugging Face image-captioning model,
and writes a manifest that preserves the original caption for auditability.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
from dataclasses import replace
from typing import Sequence

import numpy as np
import torch

from .image_data import (ImageTextRecord, _record_to_manifest_row, caption_tokens,
                         load_image_tensor, read_image_manifest)


MODE_CHOICES = ("replace", "append", "fill-empty", "sidecar")
BACKEND_CHOICES = ("stats", "hf")


def _float_rgb_text(vals):
    return ", ".join(f"{float(v):.3f}" for v in vals)


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


def _image_array_for_processor(path):
    x = load_image_tensor(path, size=0, device="cpu", center_crop=False)
    arr = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    return arr.to(torch.uint8).permute(1, 2, 0).numpy()


def _to_device(inputs, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


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


class StatsCaptioner:
    """Dependency-free captioner for local smoke tests."""

    backend_name = "stats"
    model_name = "deterministic-image-stats"

    def __init__(self, image_size=32):
        self.image_size = max(1, int(image_size))

    def caption_batch(self, records: Sequence[ImageTextRecord], prompt=""):
        captions = []
        for rec in records:
            x = load_image_tensor(rec.path, size=self.image_size, device="cpu",
                                  center_crop=True)
            rgb = ((x.mean(dim=(1, 2)) + 1.0) * 0.5).clamp(0.0, 1.0)
            contrast = float(x.std(unbiased=False).detach().cpu())
            tokens = len(caption_tokens(rec.caption))
            parts = [
                f"image with mean rgb {_float_rgb_text(rgb.tolist())}",
                f"contrast {contrast:.3f}",
                f"original caption tokens {tokens}",
            ]
            if prompt:
                parts.append(f"instruction {str(prompt).strip()}")
            captions.append("; ".join(parts))
        return captions


class HFCaptioner:
    """Generic Hugging Face vision-language captioner."""

    def __init__(self, model_name, device="cpu", dtype="auto", trust_remote_code=False,
                 max_new_tokens=96, num_beams=3):
        if not model_name:
            raise ValueError("--model is required for --backend hf")
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
            from transformers import AutoModelForSeq2SeqLM, AutoModelForVision2Seq
            from transformers import AutoProcessor
        except Exception as e:  # pragma: no cover - optional GPU dependency.
            raise ImportError(
                "Hugging Face captioning requires transformers. Install it on the GPU box "
                "with `pip install transformers accelerate pillow sentencepiece`."
            ) from e
        self.backend_name = "hf"
        self.model_name = str(model_name)
        self.device = torch.device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        torch_dtype = _dtype_from_name(dtype)
        common_kwargs = {"trust_remote_code": bool(trust_remote_code)}
        model_kwargs = dict(common_kwargs)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self.processor = AutoProcessor.from_pretrained(self.model_name, **common_kwargs)
        config = AutoConfig.from_pretrained(self.model_name, **common_kwargs)
        model_loaders = []
        if getattr(config, "model_type", "") == "blip":
            try:
                from transformers import BlipForConditionalGeneration
                model_loaders.append(BlipForConditionalGeneration)
            except Exception:
                pass
        model_loaders.extend([
            AutoModelForVision2Seq,
            AutoModelForSeq2SeqLM,
            AutoModelForCausalLM,
        ])
        last_error = None
        for loader in model_loaders:
            try:
                self.model = loader.from_pretrained(self.model_name, **model_kwargs).to(
                    self.device).eval()
                break
            except Exception as e:
                last_error = e
        else:
            raise RuntimeError(f"could not load caption model {self.model_name!r}") from last_error

    @torch.no_grad()
    def caption_batch(self, records: Sequence[ImageTextRecord], prompt=""):
        images = [_image_array_for_processor(rec.path) for rec in records]
        prompt = str(prompt or "").strip()
        kwargs = {"images": images, "return_tensors": "pt", "padding": True}
        if prompt:
            kwargs["text"] = [prompt] * len(images)
        inputs = self.processor(**kwargs)
        inputs = _to_device(inputs, self.device)
        generate_kwargs = {
            "max_new_tokens": int(self.max_new_tokens),
            "num_beams": int(self.num_beams),
        }
        with torch.inference_mode():
            ids = self.model.generate(**inputs, **generate_kwargs)
        if hasattr(self.processor, "batch_decode"):
            captions = self.processor.batch_decode(ids, skip_special_tokens=True)
        else:
            captions = [str(x) for x in ids]
        return [" ".join(str(c).split()) for c in captions]


def make_captioner(backend="stats", model="", device="cpu", dtype="auto",
                   trust_remote_code=False, max_new_tokens=96, num_beams=3,
                   stats_image_size=32):
    backend = str(backend)
    if backend == "stats":
        return StatsCaptioner(image_size=stats_image_size)
    if backend == "hf":
        return HFCaptioner(model, device=device, dtype=dtype,
                           trust_remote_code=trust_remote_code,
                           max_new_tokens=max_new_tokens, num_beams=num_beams)
    raise ValueError(f"unknown caption backend {backend!r}")


def _batched(seq, batch):
    batch = max(1, int(batch))
    for i in range(0, len(seq), batch):
        yield seq[i:i + batch]


def choose_caption(original, generated, mode="replace", separator="; "):
    original = str(original or "").strip()
    generated = str(generated or "").strip()
    mode = str(mode)
    if mode == "replace":
        return generated or original
    if mode == "fill-empty":
        return original if original else generated
    if mode == "append":
        if original and generated:
            return original + str(separator) + generated
        return generated or original
    if mode == "sidecar":
        return original
    raise ValueError(f"unknown caption mode {mode!r}")


def write_captioned_manifest(records, out_path, root="", captioner=None, mode="replace",
                             batch=8, prompt="", separator="; "):
    if mode not in MODE_CHOICES:
        raise ValueError(f"mode must be one of {MODE_CHOICES}")
    if captioner is None:
        captioner = StatsCaptioner()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows_written = 0
    replaced = appended = filled = sidecar = empty_generated = 0
    generated_lengths = []
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in _batched(list(records), batch):
            generated = captioner.caption_batch(chunk, prompt=prompt)
            for rec, gen in zip(chunk, generated):
                gen = str(gen or "").strip()
                if not gen:
                    empty_generated += 1
                new_caption = choose_caption(
                    rec.caption, gen, mode=mode, separator=separator)
                out_rec = replace(rec, caption=new_caption)
                row = _record_to_manifest_row(out_rec, root=root)
                row["original_caption"] = rec.caption
                row["generated_caption"] = gen
                row["caption_backend"] = getattr(captioner, "backend_name", "unknown")
                row["caption_model"] = getattr(captioner, "model_name", "")
                row["caption_mode"] = mode
                if mode == "replace" and new_caption != rec.caption:
                    replaced += 1
                elif mode == "append" and new_caption != rec.caption:
                    appended += 1
                elif mode == "fill-empty" and new_caption != rec.caption:
                    filled += 1
                elif mode == "sidecar":
                    sidecar += 1
                generated_lengths.append(len(caption_tokens(gen)))
                f.write(json.dumps(row, sort_keys=True) + "\n")
                rows_written += 1
    report = {
        "caption_manifest": out_path,
        "records": int(rows_written),
        "backend": getattr(captioner, "backend_name", "unknown"),
        "model": getattr(captioner, "model_name", ""),
        "mode": mode,
        "batch": int(batch),
        "prompt": prompt,
        "caption_replaced": int(replaced),
        "caption_appended": int(appended),
        "caption_filled_empty": int(filled),
        "caption_sidecar_rows": int(sidecar),
        "caption_empty_generated": int(empty_generated),
        "generated_caption_token_mean": (
            float(np.mean(generated_lengths)) if generated_lengths else 0.0
        ),
        "generated_caption_token_min": (
            int(np.min(generated_lengths)) if generated_lengths else 0
        ),
        "generated_caption_token_max": (
            int(np.max(generated_lengths)) if generated_lengths else 0
        ),
    }
    return report


def _write_ppm(path, rgb):
    arr = np.zeros((6, 8, 3), dtype=np.uint8)
    arr[:, :] = np.asarray(rgb, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(b"P6\n8 6\n255\n")
        f.write(arr.tobytes())


def selftest():
    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "red.ppm")
        _write_ppm(img, (255, 0, 0))
        manifest = os.path.join(td, "manifest.jsonl")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "image": "red.ppm",
                "caption": "short red square",
                "split": "train",
                "source": "fixture",
                "width": 8,
                "height": 6,
            }) + "\n")
        out = os.path.join(td, "captioned.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            report = main([
                "--manifest", manifest,
                "--root", td,
                "--backend", "stats",
                "--mode", "replace",
                "--out", out,
            ])
        assert report["records"] == 1
        assert report["caption_replaced"] == 1
        row = json.loads(open(out, encoding="utf-8").readline())
        assert row["original_caption"] == "short red square"
        assert row["generated_caption"].startswith("image with mean rgb")
        assert row["caption"] == row["generated_caption"]
        assert row["image"] == "red.ppm"
        append_out = os.path.join(td, "append.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            append_report = main([
                "--manifest", manifest,
                "--root", td,
                "--backend", "stats",
                "--mode", "append",
                "--separator", " | ",
                "--out", append_out,
            ])
        assert append_report["caption_appended"] == 1
        append_row = json.loads(open(append_out, encoding="utf-8").readline())
        assert append_row["caption"].startswith("short red square | image with")
    print("image_caption selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", default="", help="JSONL/CSV/TSV captioned image manifest")
    ap.add_argument("--root", default="", help="base directory for relative manifest paths")
    ap.add_argument("--split", default="", help="optional split filter; default captions all")
    ap.add_argument("--max-records", type=int, default=0,
                    help="cap records for smoke tests; 0 means all")
    ap.add_argument("--out", default="", help="output captioned manifest JSONL")
    ap.add_argument("--report-out", default="", help="optional JSON report path")
    ap.add_argument("--backend", default="stats", choices=BACKEND_CHOICES)
    ap.add_argument("--model", default="", help="Hugging Face caption model id")
    ap.add_argument("--mode", default="replace", choices=MODE_CHOICES,
                    help="how generated captions are written into the output manifest")
    ap.add_argument("--prompt", default="",
                    help="optional captioning instruction for models that accept text prompts")
    ap.add_argument("--separator", default="; ",
                    help="separator used by --mode append")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--num-beams", type=int, default=3)
    ap.add_argument("--stats-image-size", type=int, default=32)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return None
    if not args.manifest:
        ap.error("use --selftest or --manifest")
    if not args.out:
        ap.error("--out is required")
    if args.backend == "hf" and not args.model:
        ap.error("--backend hf requires --model")
    if args.max_records < 0:
        ap.error("--max-records must be non-negative")
    if args.batch <= 0:
        ap.error("--batch must be positive")
    if args.max_new_tokens <= 0:
        ap.error("--max-new-tokens must be positive")
    if args.num_beams <= 0:
        ap.error("--num-beams must be positive")
    records = read_image_manifest(
        args.manifest, root=args.root, split=args.split,
        max_records=args.max_records)
    captioner = make_captioner(
        backend=args.backend, model=args.model, device=args.device,
        dtype=args.dtype, trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
        stats_image_size=args.stats_image_size)
    report = write_captioned_manifest(
        records, args.out, root=args.root, captioner=captioner, mode=args.mode,
        batch=args.batch, prompt=args.prompt, separator=args.separator)
    report.update({
        "manifest": args.manifest,
        "root": args.root,
        "split": args.split,
        "max_records": int(args.max_records),
        "dtype": args.dtype,
        "device": args.device,
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
