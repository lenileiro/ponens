"""Manifest-driven vision understanding with schema-free concept memory.

This is the production vision-understanding path: images come from a real
captioned manifest, concepts are latent slots learned from visual patches, and
self-learning uses the model's own persistent prototype/association memory.
No color/shape grammar, canonical visual facts, or task-specific rules are
required by the training loop.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .concepts import (LatentConceptHead, LatentConceptMemory,
                       latent_concept_graph_curiosity_scores,
                       latent_concept_vicreg_loss)
from .image_data import (load_image_tensor, read_image_manifest,
                         sample_image_text_batch, summarize_records)


DEV = "cuda" if torch.cuda.is_available() else "cpu"


class PatchVisionEncoder(nn.Module):
    """Turn an image into patch tokens without assuming any visual taxonomy."""

    def __init__(self, patch=8, dim=128, max_tokens=4096):
        super().__init__()
        self.patch = int(patch)
        self.dim = int(dim)
        self.max_tokens = int(max_tokens)
        if self.patch <= 0:
            raise ValueError("patch size must be positive")
        if self.dim <= 0:
            raise ValueError("vision dimension must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.proj = nn.Linear(3 * self.patch * self.patch, self.dim)
        self.pos = nn.Parameter(torch.randn(self.max_tokens, self.dim) * 0.02)
        self.ln = nn.LayerNorm(self.dim)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("PatchVisionEncoder expects images shaped [batch,3,h,w]")
        p = self.patch
        h, w = int(x.shape[-2]), int(x.shape[-1])
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        patches = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)
        n_tokens = int(patches.shape[1])
        if n_tokens > self.max_tokens:
            raise ValueError(
                f"image produced {n_tokens} patch tokens, above max_tokens={self.max_tokens}")
        return self.ln(self.proj(patches) + self.pos[:n_tokens].unsqueeze(0))


class VisionUnderstandingModel(nn.Module):
    """Patch encoder plus latent concept slots."""

    def __init__(self, dim=128, patch=8, slots=8, heads=4, layers=1, max_tokens=4096):
        super().__init__()
        self.encoder = PatchVisionEncoder(patch=patch, dim=dim, max_tokens=max_tokens)
        self.concepts = LatentConceptHead(slots=slots, d=dim, heads=heads,
                                          mixer_layers=layers)

    def forward(self, x, project=False, return_tokens=False):
        tokens = self.encoder(x)
        slots = self.concepts(tokens, project=project)
        if return_tokens:
            return slots, tokens
        return slots


class EmbeddingAligner(nn.Module):
    """Contrast latent visual concepts against generic manifest embeddings."""

    def __init__(self, dim, target_dim, embed_dim=128):
        super().__init__()
        dim = int(dim)
        target_dim = int(target_dim)
        embed_dim = int(embed_dim)
        if dim <= 0 or target_dim <= 0 or embed_dim <= 0:
            raise ValueError("embedding aligner dimensions must be positive")
        self.visual = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.target = nn.Sequential(
            nn.LayerNorm(target_dim),
            nn.Linear(target_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

    def encode_visual(self, slots):
        pooled = slots.mean(dim=1)
        return F.normalize(self.visual(pooled), dim=-1, eps=1e-8)

    def encode_target(self, target):
        return F.normalize(self.target(target.float()), dim=-1, eps=1e-8)


def _embedding_dim(records, field):
    dims = sorted({len(getattr(rec, field)) for rec in records
                   if getattr(rec, field) is not None})
    if not dims:
        return 0
    if len(dims) != 1:
        raise ValueError(f"{field} rows have mixed dimensions: {dims}")
    return int(dims[0])


def _require_embedding(records, field):
    missing = sum(1 for rec in records if getattr(rec, field) is None)
    if missing:
        raise ValueError(f"{field} alignment requires every manifest row to have {field}")


def _embedding_tensor(records, field, device):
    rows = [getattr(rec, field) for rec in records]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _alignment_loss(aligner, slots, target, temperature=0.07, prefix="align"):
    if slots.shape[0] <= 1:
        zero = slots.sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_i2t_acc": zero.detach(),
            f"{prefix}_cos": zero.detach(),
        }
    temp = max(float(temperature), 1.0e-6)
    visual = aligner.encode_visual(slots)
    target = aligner.encode_target(target)
    logits = visual.matmul(target.t()) / temp
    labels = torch.arange(slots.shape[0], device=slots.device)
    loss = 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))
    acc = logits.argmax(-1).eq(labels).float().mean()
    cos = (visual * target).sum(-1).mean()
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_i2t_acc": acc.detach(),
        f"{prefix}_cos": cos.detach(),
    }


def _float_parts(parts):
    return {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
            for k, v in parts.items()}


def train_vision_understanding(
        manifest,
        root="",
        split="train",
        max_records=0,
        steps=200,
        batch=16,
        size=128,
        patch=8,
        dim=128,
        slots=8,
        heads=4,
        layers=1,
        max_tokens=4096,
        lr=2e-4,
        seed=0,
        device=DEV,
        crop_mode="random",
        view_crop_mode="random",
        hflip_prob=0.5,
        view_w=1.0,
        memory_size=512,
        memory_w=0.1,
        memory_temperature=0.1,
        memory_balance_w=0.0,
        memory_momentum=0.95,
        relation_decay=0.99,
        association_w=0.05,
        composition_w=0.02,
        text_align_w=0.0,
        image_align_w=0.0,
        align_embed_dim=128,
        align_temperature=0.07,
        report_out="",
        out=""):
    if not manifest:
        raise ValueError("vision understanding training requires --manifest")
    steps = int(steps)
    batch = int(batch)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch <= 0:
        raise ValueError("batch must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    for name, value in {
            "view_w": view_w,
            "memory_w": memory_w,
            "memory_balance_w": memory_balance_w,
            "association_w": association_w,
            "composition_w": composition_w,
            "text_align_w": text_align_w,
            "image_align_w": image_align_w,
    }.items():
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    records = read_image_manifest(
        manifest, root=root, split=split, max_records=max_records)
    text_dim = _embedding_dim(records, "text_embedding")
    image_dim = _embedding_dim(records, "image_embedding")
    if text_align_w > 0.0:
        _require_embedding(records, "text_embedding")
    if image_align_w > 0.0:
        _require_embedding(records, "image_embedding")

    model = VisionUnderstandingModel(
        dim=dim, patch=patch, slots=slots, heads=heads, layers=layers,
        max_tokens=max_tokens).to(device)
    memory = LatentConceptMemory(memory_size, dim).to(device)
    text_aligner = (EmbeddingAligner(dim, text_dim, align_embed_dim).to(device)
                    if text_align_w > 0.0 else None)
    image_aligner = (EmbeddingAligner(dim, image_dim, align_embed_dim).to(device)
                     if image_align_w > 0.0 else None)
    params = list(model.parameters())
    if text_aligner is not None:
        params += list(text_aligner.parameters())
    if image_aligner is not None:
        params += list(image_aligner.parameters())
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.01)

    last = {}
    for step in range(steps):
        model.train()
        if text_aligner is not None:
            text_aligner.train()
        if image_aligner is not None:
            image_aligner.train()
        x1, _captions, chosen = sample_image_text_batch(
            records, rng, batch=batch, size=size, device=device, return_records=True,
            crop_mode=crop_mode, hflip_prob=hflip_prob)
        x2 = torch.stack([
            load_image_tensor(
                rec.path, size=size, device=device, crop_mode=view_crop_mode,
                hflip=(float(rng.random()) < float(hflip_prob)), rng=rng)
            for rec in chosen
        ], dim=0)

        opt.zero_grad(set_to_none=True)
        slots1 = model(x1)
        slots2 = model(x2)
        view_loss = latent_concept_vicreg_loss(
            slots1, slots2, invariance_weight=1.0,
            variance_weight=1.0, covariance_weight=0.05, variance_target=0.25)
        mem_loss = memory(
            slots1, temperature=memory_temperature, balance_w=memory_balance_w)
        assoc_loss = memory.association_loss(
            slots1, temperature=memory_temperature, transitive_steps=2,
            transitive_w=0.05)
        comp_loss = memory.composition_loss(
            slots1, temperature=memory_temperature, transitive_steps=2,
            transitive_w=0.05)
        loss = (float(view_w) * view_loss
                + float(memory_w) * mem_loss
                + float(association_w) * assoc_loss
                + float(composition_w) * comp_loss)
        parts = {
            "step": torch.tensor(float(step + 1), device=device),
            "view_loss": view_loss.detach(),
            "memory_loss": mem_loss.detach(),
            "association_loss": assoc_loss.detach(),
            "composition_loss": comp_loss.detach(),
        }
        if text_aligner is not None:
            target = _embedding_tensor(chosen, "text_embedding", device)
            text_loss, text_parts = _alignment_loss(
                text_aligner, slots1, target, temperature=align_temperature,
                prefix="text_align")
            loss = loss + float(text_align_w) * text_loss
            parts.update(text_parts)
        if image_aligner is not None:
            target = _embedding_tensor(chosen, "image_embedding", device)
            image_loss, image_parts = _alignment_loss(
                image_aligner, slots1, target, temperature=align_temperature,
                prefix="image_align")
            loss = loss + float(image_align_w) * image_loss
            parts.update(image_parts)
        parts["total_loss"] = loss.detach()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            update_slots = torch.cat([slots1.detach(), slots2.detach()], dim=0)
            memory.update(
                update_slots, momentum=memory_momentum, relation_decay=relation_decay)
            curiosity, curiosity_parts = latent_concept_graph_curiosity_scores(
                slots1.detach(), memory.active(), memory.active_relations(),
                temperature=memory_temperature, transitive_steps=2,
                transitive_w=0.05)
            parts["self_learning_curiosity_mean"] = curiosity.mean().detach()
            parts["self_learning_novelty_mean"] = (
                curiosity_parts["novelty"].mean().detach())
            parts["self_learning_association_mean"] = (
                curiosity_parts["association"].mean().detach())
            parts["concept_memory_filled"] = memory.filled.detach().float()
            parts["concept_memory_relation_updates"] = (
                memory.relation_updates.detach().float())
        last = _float_parts(parts)

    report = {
        "experiment": "vision_understanding_manifest_concepts",
        "data_mode": "image_manifest",
        "manifest": manifest,
        "root": root,
        "split": split,
        "max_records": int(max_records),
        "steps": int(steps),
        "batch": int(batch),
        "size": size,
        "patch": int(patch),
        "dim": int(dim),
        "heads": int(heads),
        "layers": int(layers),
        "concept_slots": int(slots),
        "concept_memory_size": int(memory_size),
        "concept_memory_filled": int(memory.filled.detach().cpu()),
        "concept_memory_updates": int(memory.updates.detach().cpu()),
        "concept_relation_updates": int(memory.relation_updates.detach().cpu()),
        "view_w": float(view_w),
        "memory_w": float(memory_w),
        "association_w": float(association_w),
        "composition_w": float(composition_w),
        "text_align_w": float(text_align_w),
        "image_align_w": float(image_align_w),
        "text_embedding_in_dim": int(text_dim),
        "image_embedding_in_dim": int(image_dim),
        "last": last,
    }
    report.update(summarize_records(records))
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "memory_state_dict": memory.state_dict(),
            "text_aligner_state_dict": (
                text_aligner.state_dict() if text_aligner is not None else {}),
            "image_aligner_state_dict": (
                image_aligner.state_dict() if image_aligner is not None else {}),
            "report": report,
        }, out)
    if report_out:
        os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
    return model, memory, report


def _write_ppm(path, arr):
    arr = np.asarray(arr, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(f"P6\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii"))
        f.write(arr.tobytes())


def selftest():
    with tempfile.TemporaryDirectory() as td:
        rows = []
        rng = np.random.default_rng(7)
        yy, xx = np.mgrid[0:24, 0:24]
        for i in range(4):
            base = np.zeros((24, 24, 3), dtype=np.uint8)
            wave = (127.5 + 90.0 * np.sin((xx + i * 3) / (2.2 + i))).astype(np.uint8)
            grain = rng.integers(0, 48, size=(24, 24), dtype=np.uint8)
            base[..., 0] = (wave + grain) % 255
            base[..., 1] = (yy * (5 + i) + grain) % 255
            base[..., 2] = ((xx + yy) * (3 + i) + 2 * grain) % 255
            image = f"texture_{i}.ppm"
            _write_ppm(os.path.join(td, image), base)
            emb = np.eye(4, dtype=np.float32)[i].tolist()
            rows.append({
                "image": image,
                "caption": f"natural texture study {i}",
                "split": "train",
                "source": "fixture",
                "text_embedding": emb,
                "image_embedding": emb[::-1],
            })
        manifest = os.path.join(td, "manifest.jsonl")
        with open(manifest, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        _model, memory, report = train_vision_understanding(
            manifest, root=td, steps=2, batch=2, size=16, patch=4, dim=16,
            slots=4, heads=4, layers=1, memory_size=16, text_align_w=0.1,
            image_align_w=0.1, align_embed_dim=8, device="cpu", seed=3)
        assert report["experiment"] == "vision_understanding_manifest_concepts"
        assert report["data_mode"] == "image_manifest"
        assert report["concept_memory_filled"] > 0
        assert report["concept_relation_updates"] > 0
        assert "view_loss" in report["last"]
        assert "self_learning_novelty_mean" in report["last"]
        assert int(memory.filled.detach().cpu()) == report["concept_memory_filled"]
    print("vision_understanding selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--root", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-records", type=int, default=0, dest="max_records")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", default="128")
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=DEV)
    ap.add_argument("--crop-mode", default="random",
                    choices=("center", "random", "none", "pad"), dest="crop_mode")
    ap.add_argument("--view-crop-mode", default="random",
                    choices=("center", "random", "none", "pad"), dest="view_crop_mode")
    ap.add_argument("--hflip-prob", type=float, default=0.5, dest="hflip_prob")
    ap.add_argument("--view-w", type=float, default=1.0, dest="view_w")
    ap.add_argument("--memory-size", type=int, default=512, dest="memory_size")
    ap.add_argument("--memory-w", type=float, default=0.1, dest="memory_w")
    ap.add_argument("--memory-temperature", type=float, default=0.1,
                    dest="memory_temperature")
    ap.add_argument("--memory-balance-w", type=float, default=0.0,
                    dest="memory_balance_w")
    ap.add_argument("--memory-momentum", type=float, default=0.95,
                    dest="memory_momentum")
    ap.add_argument("--relation-decay", type=float, default=0.99,
                    dest="relation_decay")
    ap.add_argument("--association-w", type=float, default=0.05,
                    dest="association_w")
    ap.add_argument("--composition-w", type=float, default=0.02,
                    dest="composition_w")
    ap.add_argument("--text-align-w", type=float, default=0.0, dest="text_align_w")
    ap.add_argument("--image-align-w", type=float, default=0.0, dest="image_align_w")
    ap.add_argument("--align-embed-dim", type=int, default=128,
                    dest="align_embed_dim")
    ap.add_argument("--align-temperature", type=float, default=0.07,
                    dest="align_temperature")
    ap.add_argument("--out", default="runs/vision_understanding.pt")
    ap.add_argument("--report-out", default="", dest="report_out")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.train:
        ap.error("pass --train or --selftest")
    if not args.manifest:
        ap.error("--train requires --manifest")
    if args.max_records < 0:
        ap.error("--max-records must be non-negative")
    if args.hflip_prob < 0.0 or args.hflip_prob > 1.0:
        ap.error("--hflip-prob must be in [0, 1]")
    try:
        size = int(args.size) if "x" not in str(args.size).lower() else tuple(
            int(x) for x in str(args.size).lower().split("x", 1))
    except ValueError as exc:
        ap.error(str(exc))
    _model, _memory, report = train_vision_understanding(
        manifest=args.manifest, root=args.root, split=args.split,
        max_records=args.max_records, steps=args.steps, batch=args.batch,
        size=size, patch=args.patch, dim=args.dim, slots=args.slots,
        heads=args.heads, layers=args.layers, max_tokens=args.max_tokens,
        lr=args.lr, seed=args.seed, device=args.device, crop_mode=args.crop_mode,
        view_crop_mode=args.view_crop_mode, hflip_prob=args.hflip_prob,
        view_w=args.view_w, memory_size=args.memory_size,
        memory_w=args.memory_w, memory_temperature=args.memory_temperature,
        memory_balance_w=args.memory_balance_w,
        memory_momentum=args.memory_momentum, relation_decay=args.relation_decay,
        association_w=args.association_w, composition_w=args.composition_w,
        text_align_w=args.text_align_w, image_align_w=args.image_align_w,
        align_embed_dim=args.align_embed_dim,
        align_temperature=args.align_temperature, report_out=args.report_out,
        out=args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
