"""Latent fact-conditioned rectified flow for the synthetic image rung.

This is Image-3: move generation from pixels into a compact visual latent.

The design follows the current scalable image-generation pattern at toy scale:

  image -> semantic autoencoder latent -> rectified flow in latent space -> decoder -> image

The autoencoder is not reconstruction-only.  It also predicts canonical color/shape facts from
the latent, which keeps the compression aligned with the FER/UFR requirement: factors should stay
linearly usable after compression instead of becoming another entangled bottleneck.

  python -m thinking.image_latent --selftest
  python -m thinking.image_latent --train --flow-arch dit --ae-steps 400 --flow-steps 400 \
      --cond-drop 0.1 --cfg-scale 1.5 --sample-steps 8 --flow-semantic-w 0.25 \
      --out runs/image_latent_dit.pt
  python -m thinking.image_latent --eval-checkpoint runs/image_latent_dit.pt \
      --cfg-scales 1.0,1.25,1.5,2.0 --sample-steps-list 4,8,16 \
      --eval-seeds 1,2,3 --roundtrip-samples 2 --eval-out runs/image_latent_dit_sweep.json
  python -m thinking.image_latent --train --cond-mode text --flow-arch dit \
      --ae-steps 400 --flow-steps 400 --flow-semantic-w 0.25
"""
import argparse
import json
import math
import os
import tempfile
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_data import (build_caption_vocab, caption_ids, load_image_tensor,
                         read_image_manifest, sample_image_text_batch, summarize_records)
from .image_flow import FACT_VOCAB, fact_condition
from .vision import COLORS, SHAPES, DEV, ObjectSpec, object_facts, render_object, sample_object

COLOR_NAMES = tuple(COLORS)
DEFAULT_PROMPT_TEMPLATES = (
    "a {color} {shape}",
    "the object is {color} and {shape}",
    "{color} object with {shape} shape",
    "render p0 as a {shape} colored {color}",
)
FACT_GROUPS = {
    pred: tuple(i for i, fact in enumerate(FACT_VOCAB) if fact[0] == pred)
    for pred in sorted({fact[0] for fact in FACT_VOCAB})
}
SAMPLE_METHODS = ("euler", "heun")
MMDIT_ATTN_IMPLS = ("manual", "sdpa", "auto")
DEFAULT_GUIDANCE_INTERVAL = (0.0, 1.0)


def _batch(n, rng, size=32, device=DEV, return_specs=False):
    imgs, conds, yc, ys, specs = [], [], [], [], []
    for _ in range(n):
        spec = sample_object(rng)
        specs.append(spec)
        imgs.append(render_object(spec, size=size) * 2.0 - 1.0)
        conds.append(fact_condition(object_facts(spec), device="cpu").numpy())
        yc.append(COLOR_NAMES.index(spec.color))
        ys.append(SHAPES.index(spec.shape))
    x = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
    cond = torch.tensor(np.stack(conds), dtype=torch.float32, device=device)
    yc = torch.tensor(yc, dtype=torch.long, device=device)
    ys = torch.tensor(ys, dtype=torch.long, device=device)
    if return_specs:
        return x, cond, yc, ys, specs
    return x, cond, yc, ys


def split_prompt(s):
    return str(s).lower().replace(".", " .").replace(",", " ,").split()


def render_prompt(spec, rng=None, templates=DEFAULT_PROMPT_TEMPLATES, index=None):
    if index is None:
        index = 0 if rng is None else int(rng.integers(len(templates)))
    tpl = templates[index % len(templates)]
    return split_prompt(tpl.format(color=spec.color, shape=spec.shape, slot=spec.slot))


def build_prompt_vocab(templates=DEFAULT_PROMPT_TEMPLATES):
    toks = ["<pad>"]
    for color in COLORS:
        for shape in SHAPES:
            spec = ObjectSpec("p0", color, shape)
            for i in range(len(templates)):
                toks.extend(render_prompt(spec, templates=templates, index=i))
    return {tok: i for i, tok in enumerate(dict.fromkeys(toks))}


def prompt_ids(specs, vocab, rng=None, templates=DEFAULT_PROMPT_TEMPLATES, device=DEV):
    rows = [render_prompt(spec, rng=rng, templates=templates) for spec in specs]
    max_len = max(len(r) for r in rows)
    ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    unk = 0
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor([vocab.get(tok, unk) for tok in row],
                                         dtype=torch.long, device=device)
    return ids


class SemanticAutoencoder(nn.Module):
    """Small convolutional AE with fact heads on the compressed latent."""

    def __init__(self, latent_ch=16, hidden=64):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.downsample = 4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, latent_ch, 3, padding=1),
            nn.GroupNorm(1, latent_ch),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden // 2, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.fact_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.color = nn.Sequential(
            nn.Linear(latent_ch * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(COLORS)),
        )
        self.shape = nn.Sequential(
            nn.Linear(latent_ch * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(SHAPES)),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def fact_logits(self, z):
        pooled = self.fact_pool(z).flatten(1)
        return {"color": self.color(pooled), "shape": self.shape(pooled)}

    def forward(self, x):
        z = self.encode(x)
        out = self.fact_logits(z)
        out["latent"] = z
        out["recon"] = self.decode(z)
        return out


def _norm(ch):
    groups = min(8, int(ch))
    while groups > 1 and int(ch) % groups:
        groups -= 1
    return nn.GroupNorm(groups, int(ch))


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            _norm(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            _norm(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualAutoencoder(nn.Module):
    """Deeper representation AE with configurable spatial compression.

    This is still lightweight enough for local tests, but it has the shape we need for real
    image work: residual stages, high-dimensional latents, and explicit downsample control.
    """

    def __init__(self, latent_ch=16, hidden=64, downsample=8, res_blocks=1):
        super().__init__()
        downsample = int(downsample)
        if downsample < 2 or downsample & (downsample - 1):
            raise ValueError("latent_downsample must be a power of two >= 2")
        self.latent_ch = int(latent_ch)
        self.downsample = downsample
        self.res_blocks = int(res_blocks)
        levels = int(math.log2(downsample))
        stem_ch = max(8, hidden // 2)
        enc = [nn.Conv2d(3, stem_ch, 3, padding=1)]
        channels = [stem_ch]
        ch = stem_ch
        for i in range(levels):
            out_ch = min(int(hidden) * (2 ** i), int(hidden) * 4)
            enc.extend([nn.GELU(), nn.Conv2d(ch, out_ch, 4, stride=2, padding=1)])
            enc.extend(ResidualBlock(out_ch) for _ in range(max(0, self.res_blocks)))
            ch = out_ch
            channels.append(ch)
        enc.extend([nn.GELU(), nn.Conv2d(ch, latent_ch, 3, padding=1), _norm(latent_ch)])
        self.encoder = nn.Sequential(*enc)

        dec_ch = channels[-1]
        dec = [nn.Conv2d(latent_ch, dec_ch, 3, padding=1)]
        for level in reversed(range(levels)):
            out_ch = channels[level]
            dec.extend(ResidualBlock(dec_ch) for _ in range(max(0, self.res_blocks)))
            dec.extend([nn.GELU(), nn.ConvTranspose2d(dec_ch, out_ch, 4,
                                                      stride=2, padding=1)])
            dec_ch = out_ch
        dec.extend([nn.GELU(), nn.Conv2d(dec_ch, 3, 3, padding=1), nn.Tanh()])
        self.decoder = nn.Sequential(*dec)
        self.fact_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.color = nn.Sequential(
            nn.Linear(latent_ch * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(COLORS)),
        )
        self.shape = nn.Sequential(
            nn.Linear(latent_ch * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(SHAPES)),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def fact_logits(self, z):
        pooled = self.fact_pool(z).flatten(1)
        return {"color": self.color(pooled), "shape": self.shape(pooled)}

    def forward(self, x):
        z = self.encode(x)
        out = self.fact_logits(z)
        out["latent"] = z
        out["recon"] = self.decode(z)
        return out


def make_autoencoder(ae_arch="semantic", latent_ch=16, hidden=64, latent_downsample=4,
                     ae_res_blocks=1):
    if ae_arch == "semantic":
        if int(latent_downsample) != 4:
            raise ValueError("semantic AE uses latent_downsample=4; use --ae-arch residual")
        return SemanticAutoencoder(latent_ch=latent_ch, hidden=hidden)
    if ae_arch == "residual":
        return ResidualAutoencoder(latent_ch=latent_ch, hidden=hidden,
                                   downsample=latent_downsample,
                                   res_blocks=ae_res_blocks)
    raise ValueError(f"unknown autoencoder architecture {ae_arch!r}")


def ae_latent_shape(ae, size):
    downsample = int(getattr(ae, "downsample", 4))
    if int(size) % downsample:
        raise ValueError(f"image size {size} must be divisible by AE downsample {downsample}")
    side = int(size) // downsample
    return int(ae.latent_ch), side, side


class PromptConditioner(nn.Module):
    """Small learned text encoder that maps prompt tokens to continuous conditions."""

    def __init__(self, vocab_size, cond_dim=64, hidden=64, heads=4, max_len=32, pad=0):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.pad = int(pad)
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=pad)
        self.pos = nn.Embedding(max_len, hidden)
        heads = max(1, min(heads, hidden))
        while hidden % heads:
            heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, cond_dim))

    def forward(self, ids, return_tokens=False):
        b, l = ids.shape
        pos = torch.arange(l, device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        mask = ids.eq(self.pad)
        h = self.emb(ids) + self.pos(pos)[None]
        h = self.enc(h, src_key_padding_mask=mask)
        keep = (~mask).to(h.dtype).unsqueeze(-1)
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        vec = self.out(pooled)
        if return_tokens:
            return {"vec": vec, "tokens": self.out(h), "mask": mask}
        return vec


class PrecomputedTextConditioner(nn.Module):
    """Project external caption embeddings into the model's conditioning space."""

    def __init__(self, input_dim, cond_dim=64, hidden=64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.cond_dim = int(cond_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.cond_dim),
        )

    def forward(self, embeddings, return_tokens=False):
        vec = self.net(embeddings.float())
        if return_tokens:
            mask = torch.zeros((vec.shape[0], 1), dtype=torch.bool, device=vec.device)
            return {"vec": vec, "tokens": vec[:, None, :], "mask": mask}
        return vec


class ImageTextAligner(nn.Module):
    """Contrastive bridge between image latents and caption conditions.

    This is the manifest-data equivalent of the synthetic fact heads: it gives real-image runs a
    generic paired image/text semantic signal without renderer-specific labels.
    """

    def __init__(self, latent_ch=16, cond_dim=64, hidden=64, embed_dim=128,
                 temperature=0.07):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.image = nn.Sequential(
            nn.LayerNorm(int(latent_ch)),
            nn.Linear(int(latent_ch), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        self.text = nn.Sequential(
            nn.LayerNorm(int(cond_dim)),
            nn.Linear(int(cond_dim), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        temp = max(float(temperature), 1.0e-4)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))

    def encode_image(self, z):
        pooled = z.float().mean(dim=(2, 3))
        return F.normalize(self.image(pooled), dim=-1)

    def encode_text(self, cond):
        vec = condition_vector(cond).float()
        return F.normalize(self.text(vec), dim=-1)

    def forward(self, z, cond):
        return self.encode_image(z), self.encode_text(cond)

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)


class ImageFeatureAligner(nn.Module):
    """Contrastive bridge between AE latents and external image features."""

    def __init__(self, latent_ch=16, feature_dim=768, hidden=64, embed_dim=128,
                 temperature=0.07):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim)
        self.image = nn.Sequential(
            nn.LayerNorm(int(latent_ch)),
            nn.Linear(int(latent_ch), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        self.feature = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.embed_dim),
        )
        temp = max(float(temperature), 1.0e-4)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))

    def encode_image(self, z):
        pooled = z.float().mean(dim=(2, 3))
        return F.normalize(self.image(pooled), dim=-1)

    def encode_feature(self, features):
        return F.normalize(self.feature(features.float()), dim=-1)

    def forward(self, z, features):
        return self.encode_image(z), self.encode_feature(features)

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)


class LatentFlowNet(nn.Module):
    """Velocity field v_theta(z_t, t, canonical_facts)."""

    def __init__(self, latent_ch=16, hidden=64, cond_dim=None):
        super().__init__()
        cond_dim = cond_dim or len(FACT_VOCAB)
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch + 1 + hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, latent_ch, 3, padding=1),
        )

    def forward(self, z, t, cond):
        cond = condition_vector(cond)
        if t.ndim == 1:
            t = t[:, None, None, None]
        if t.ndim == 2:
            t = t[:, :, None, None]
        b, _, h, w = z.shape
        tchan = t.expand(b, 1, h, w)
        c = self.cond(cond).view(b, -1, 1, 1).expand(b, -1, h, w)
        return self.net(torch.cat([z, tchan, c], dim=1))


def make_velocity_head(hidden, latent_ch, width_mult=1):
    width_mult = int(width_mult)
    if width_mult <= 1:
        return nn.Linear(hidden, latent_ch)
    width = int(hidden) * width_mult
    return nn.Sequential(
        nn.Linear(hidden, width),
        nn.GELU(),
        nn.Linear(width, width),
        nn.GELU(),
        nn.Linear(width, latent_ch),
    )


class LatentDiTFlowNet(nn.Module):
    """Patch-token transformer velocity field over semantic image latents.

    This is intentionally tiny, but it matches the scalable architecture shape: flatten the
    autoencoder latent grid into tokens, condition every token on time + canonical facts, run
    transformer blocks, then project token velocities back to the latent grid.
    """

    def __init__(self, latent_ch=16, hidden=96, depth=3, heads=4, cond_dim=None,
                 max_tokens=256, head_width_mult=1):
        super().__init__()
        cond_dim = cond_dim or len(FACT_VOCAB)
        self.latent_ch = int(latent_ch)
        self.hidden = int(hidden)
        self.max_tokens = int(max_tokens)
        self.head_width_mult = int(head_width_mult)
        self.in_proj = nn.Linear(latent_ch, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden))
        self.cond = nn.Sequential(
            nn.Linear(cond_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = make_velocity_head(hidden, latent_ch, self.head_width_mult)

    def forward(self, z, t, cond):
        cond = condition_vector(cond)
        if t.ndim > 2:
            t = t.flatten(1)[:, :1]
        elif t.ndim == 1:
            t = t[:, None]
        b, c, h, w = z.shape
        n = h * w
        if n > self.max_tokens:
            raise ValueError(f"latent token count {n} exceeds max_tokens={self.max_tokens}")
        toks = z.flatten(2).transpose(1, 2)                  # B,N,C
        ctx = self.cond(torch.cat([cond, t.to(cond.dtype)], dim=1))[:, None, :]
        x = self.in_proj(toks) + self.pos[:, :n] + ctx
        x = self.blocks(x)
        v = self.out_proj(self.norm(x))
        return v.transpose(1, 2).reshape(b, c, h, w)


class CrossDiTBlock(nn.Module):
    """Tiny image-token block with prompt-token cross-attention."""

    def __init__(self, hidden=96, heads=4):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.cross_norm = nn.LayerNorm(hidden)
        self.ctx_norm = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x, ctx, ctx_mask=None):
        q = self.self_norm(x)
        x = x + self.self_attn(q, q, q, need_weights=False)[0]
        x = x + self.cross_attn(self.cross_norm(x), self.ctx_norm(ctx), self.ctx_norm(ctx),
                                key_padding_mask=ctx_mask, need_weights=False)[0]
        return x + self.ff(self.ff_norm(x))


class LatentCrossDiTFlowNet(nn.Module):
    """Latent DiT that lets image tokens attend to prompt/fact condition tokens."""

    uses_cond_tokens = True

    def __init__(self, latent_ch=16, hidden=96, depth=3, heads=4, cond_dim=None,
                 max_tokens=256, head_width_mult=1):
        super().__init__()
        cond_dim = cond_dim or len(FACT_VOCAB)
        self.latent_ch = int(latent_ch)
        self.hidden = int(hidden)
        self.max_tokens = int(max_tokens)
        self.head_width_mult = int(head_width_mult)
        self.in_proj = nn.Linear(latent_ch, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden))
        self.cond = nn.Sequential(
            nn.Linear(cond_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.ctx_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([CrossDiTBlock(hidden, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = make_velocity_head(hidden, latent_ch, self.head_width_mult)

    def _context(self, cond):
        if isinstance(cond, dict):
            tokens = cond["tokens"]
            mask = cond.get("mask")
        else:
            tokens = cond[:, None, :]
            mask = None
        return self.ctx_proj(tokens), mask

    def forward(self, z, t, cond):
        cond_vec = condition_vector(cond)
        if t.ndim > 2:
            t = t.flatten(1)[:, :1]
        elif t.ndim == 1:
            t = t[:, None]
        b, c, h, w = z.shape
        n = h * w
        if n > self.max_tokens:
            raise ValueError(f"latent token count {n} exceeds max_tokens={self.max_tokens}")
        toks = z.flatten(2).transpose(1, 2)
        global_ctx = self.cond(torch.cat([cond_vec, t.to(cond_vec.dtype)], dim=1))[:, None, :]
        ctx, ctx_mask = self._context(cond)
        x = self.in_proj(toks) + self.pos[:, :n] + global_ctx
        for block in self.blocks:
            x = block(x, ctx, ctx_mask=ctx_mask)
        v = self.out_proj(self.norm(x))
        return v.transpose(1, 2).reshape(b, c, h, w)


class HeadRMSNorm(nn.Module):
    """RMS-normalize query/key vectors per attention head."""

    def __init__(self, head_dim, eps=1.0e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(head_dim)))

    def forward(self, x):
        dtype = x.dtype
        x_float = x.float()
        denom = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_float * denom).to(dtype) * self.weight.to(dtype=dtype)


class MMDiTBlock(nn.Module):
    """Tiny dual-stream block with separate image/text projections and joint attention."""

    def __init__(self, hidden=96, heads=4, qk_norm=False, attn_impl="manual"):
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        attn_impl = str(attn_impl)
        if attn_impl not in MMDIT_ATTN_IMPLS:
            raise ValueError(f"unknown MM-DiT attention implementation {attn_impl!r}")
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.uses_qk_norm = bool(qk_norm)
        self.attn_impl = attn_impl
        self.scale = self.head_dim ** -0.5
        self.img_norm = nn.LayerNorm(hidden)
        self.ctx_norm = nn.LayerNorm(hidden)
        self.img_qkv = nn.Linear(hidden, hidden * 3)
        self.ctx_qkv = nn.Linear(hidden, hidden * 3)
        self.img_q_norm = HeadRMSNorm(self.head_dim) if self.uses_qk_norm else nn.Identity()
        self.img_k_norm = HeadRMSNorm(self.head_dim) if self.uses_qk_norm else nn.Identity()
        self.ctx_q_norm = HeadRMSNorm(self.head_dim) if self.uses_qk_norm else nn.Identity()
        self.ctx_k_norm = HeadRMSNorm(self.head_dim) if self.uses_qk_norm else nn.Identity()
        self.img_out = nn.Linear(hidden, hidden)
        self.ctx_out = nn.Linear(hidden, hidden)
        self.img_ff_norm = nn.LayerNorm(hidden)
        self.ctx_ff_norm = nn.LayerNorm(hidden)
        self.img_ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(),
                                    nn.Linear(hidden * 4, hidden))
        self.ctx_ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(),
                                    nn.Linear(hidden * 4, hidden))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 8))
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 4))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _qkv(self, proj, x):
        b, n, h = x.shape
        q, k, v = proj(x).view(b, n, 3, self.heads, self.head_dim).unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    @staticmethod
    def _sdpa_available():
        return hasattr(F, "scaled_dot_product_attention")

    @staticmethod
    def _key_mask(ctx_mask, b, n_img, device):
        if ctx_mask is None:
            return None
        img_mask = torch.zeros((b, n_img), dtype=torch.bool, device=device)
        return torch.cat([img_mask, ctx_mask], dim=1)

    def _manual_attention(self, q, k, v, key_mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_mask is not None:
            attn = attn.masked_fill(key_mask[:, None, None, :], torch.finfo(attn.dtype).min)
        return torch.softmax(attn, dim=-1).matmul(v)

    def _sdpa_attention(self, q, k, v, key_mask=None):
        attn_mask = None
        if key_mask is not None:
            attn_mask = torch.zeros((q.shape[0], 1, 1, key_mask.shape[1]),
                                    dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(key_mask[:, None, None, :],
                                              torch.finfo(q.dtype).min)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    def _joint_attention(self, q, k, v, key_mask=None):
        if self.attn_impl == "manual":
            return self._manual_attention(q, k, v, key_mask=key_mask)
        if self._sdpa_available():
            return self._sdpa_attention(q, k, v, key_mask=key_mask)
        if self.attn_impl == "sdpa":
            raise RuntimeError("scaled_dot_product_attention is unavailable in this torch build")
        return self._manual_attention(q, k, v, key_mask=key_mask)

    def forward(self, img, ctx, cond_ctx, ctx_mask=None):
        b, n_img, h = img.shape
        n_ctx = ctx.shape[1]
        (img_attn_shift, img_attn_scale, ctx_attn_shift, ctx_attn_scale,
         img_ff_shift, img_ff_scale, ctx_ff_shift, ctx_ff_scale) = self.ada(cond_ctx).chunk(
             8, dim=-1)
        img_attn_gate, ctx_attn_gate, img_ff_gate, ctx_ff_gate = self.gate(cond_ctx).chunk(
            4, dim=-1)
        img_attn = self._modulate(self.img_norm(img), img_attn_shift, img_attn_scale)
        ctx_attn = self._modulate(self.ctx_norm(ctx), ctx_attn_shift, ctx_attn_scale)
        qi, ki, vi = self._qkv(self.img_qkv, img_attn)
        qc, kc, vc = self._qkv(self.ctx_qkv, ctx_attn)
        qi, ki = self.img_q_norm(qi), self.img_k_norm(ki)
        qc, kc = self.ctx_q_norm(qc), self.ctx_k_norm(kc)
        q = torch.cat([qi, qc], dim=2)
        k = torch.cat([ki, kc], dim=2)
        v = torch.cat([vi, vc], dim=2)
        key_mask = self._key_mask(ctx_mask, b, n_img, q.device)
        mixed = self._joint_attention(q, k, v, key_mask=key_mask)
        mixed = mixed.transpose(1, 2).reshape(b, n_img + n_ctx, h)
        img_delta, ctx_delta = mixed[:, :n_img], mixed[:, n_img:]
        img = img + (1.0 + img_attn_gate[:, None, :]) * self.img_out(img_delta)
        ctx = ctx + (1.0 + ctx_attn_gate[:, None, :]) * self.ctx_out(ctx_delta)
        img_ff = self._modulate(self.img_ff_norm(img), img_ff_shift, img_ff_scale)
        ctx_ff = self._modulate(self.ctx_ff_norm(ctx), ctx_ff_shift, ctx_ff_scale)
        img = img + (1.0 + img_ff_gate[:, None, :]) * self.img_ff(img_ff)
        ctx = ctx + (1.0 + ctx_ff_gate[:, None, :]) * self.ctx_ff(ctx_ff)
        if ctx_mask is not None:
            ctx = ctx.masked_fill(ctx_mask[:, :, None], 0.0)
        return img, ctx


class LatentMMDiTFlowNet(nn.Module):
    """Toy MM-DiT latent flow with bidirectional image/condition token mixing."""

    uses_cond_tokens = True
    uses_adaptive_modulation = True
    uses_residual_gating = True

    def __init__(self, latent_ch=16, hidden=96, depth=3, heads=4, cond_dim=None,
                 max_tokens=256, head_width_mult=1, qk_norm=False,
                 attn_impl="manual"):
        super().__init__()
        attn_impl = str(attn_impl)
        if attn_impl not in MMDIT_ATTN_IMPLS:
            raise ValueError(f"unknown MM-DiT attention implementation {attn_impl!r}")
        cond_dim = cond_dim or len(FACT_VOCAB)
        self.latent_ch = int(latent_ch)
        self.hidden = int(hidden)
        self.max_tokens = int(max_tokens)
        self.head_width_mult = int(head_width_mult)
        self.uses_qk_norm = bool(qk_norm)
        self.attn_impl = attn_impl
        self.in_proj = nn.Linear(latent_ch, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden))
        self.time = nn.Sequential(nn.Linear(cond_dim + 1, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden))
        self.ctx_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                hidden, heads, qk_norm=self.uses_qk_norm, attn_impl=self.attn_impl
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = make_velocity_head(hidden, latent_ch, self.head_width_mult)

    def _context(self, cond):
        if isinstance(cond, dict):
            tokens = cond["tokens"]
            mask = cond.get("mask")
        else:
            tokens = cond[:, None, :]
            mask = None
        return self.ctx_proj(tokens), mask

    def forward(self, z, t, cond):
        cond_vec = condition_vector(cond)
        if t.ndim > 2:
            t = t.flatten(1)[:, :1]
        elif t.ndim == 1:
            t = t[:, None]
        b, c, h, w = z.shape
        n = h * w
        if n > self.max_tokens:
            raise ValueError(f"latent token count {n} exceeds max_tokens={self.max_tokens}")
        cond_ctx = self.time(torch.cat([cond_vec, t.to(cond_vec.dtype)], dim=1))
        img = self.in_proj(z.flatten(2).transpose(1, 2)) + self.pos[:, :n]
        img = img + cond_ctx[:, None, :]
        ctx, ctx_mask = self._context(cond)
        for block in self.blocks:
            img, ctx = block(img, ctx, cond_ctx, ctx_mask=ctx_mask)
        v = self.out_proj(self.norm(img))
        return v.transpose(1, 2).reshape(b, c, h, w)


def make_flow(flow_arch="conv", latent_ch=16, hidden=64, dit_depth=3, dit_heads=4,
              cond_dim=None, dit_head_width_mult=1, latent_max_tokens=256,
              dit_qk_norm=False, dit_attn_impl="manual"):
    if flow_arch == "conv":
        return LatentFlowNet(latent_ch=latent_ch, hidden=hidden, cond_dim=cond_dim)
    dit_head_width_mult = int(dit_head_width_mult)
    if dit_head_width_mult <= 0:
        raise ValueError("dit_head_width_mult must be positive")
    if flow_arch == "dit":
        heads = max(1, min(dit_heads, hidden // 16))
        while hidden % heads:
            heads -= 1
        return LatentDiTFlowNet(latent_ch=latent_ch, hidden=hidden, depth=dit_depth, heads=heads,
                                cond_dim=cond_dim,
                                head_width_mult=dit_head_width_mult,
                                max_tokens=latent_max_tokens)
    if flow_arch == "crossdit":
        heads = max(1, min(dit_heads, hidden // 16))
        while hidden % heads:
            heads -= 1
        return LatentCrossDiTFlowNet(latent_ch=latent_ch, hidden=hidden, depth=dit_depth,
                                     heads=heads, cond_dim=cond_dim,
                                     head_width_mult=dit_head_width_mult,
                                     max_tokens=latent_max_tokens)
    if flow_arch == "mmdit":
        heads = max(1, min(dit_heads, hidden // 16))
        while hidden % heads:
            heads -= 1
        return LatentMMDiTFlowNet(latent_ch=latent_ch, hidden=hidden, depth=dit_depth,
                                  heads=heads, cond_dim=cond_dim,
                                  head_width_mult=dit_head_width_mult,
                                  max_tokens=latent_max_tokens,
                                  qk_norm=dit_qk_norm,
                                  attn_impl=dit_attn_impl)
    raise ValueError(f"unknown latent flow architecture {flow_arch!r}")


def attach_text_aligner(flow, text_aligner):
    # Keep the aligner available for eval without registering it as part of flow.state_dict().
    if hasattr(flow, "_modules"):
        flow._modules.pop("text_aligner", None)
    flow.__dict__["text_aligner"] = text_aligner
    return flow


def attach_image_feature_aligner(flow, image_feature_aligner):
    # Keep the aligner available for eval without registering it as part of flow.state_dict().
    if hasattr(flow, "_modules"):
        flow._modules.pop("image_feature_aligner", None)
    flow.__dict__["image_feature_aligner"] = image_feature_aligner
    return flow


def _parse_number_list(s, cast=float):
    if isinstance(s, (tuple, list)):
        return tuple(cast(x) for x in s)
    return tuple(cast(x.strip()) for x in str(s).split(",") if x.strip())


def _parse_templates(s):
    if not s:
        return DEFAULT_PROMPT_TEMPLATES
    if isinstance(s, (tuple, list)):
        return tuple(str(x) for x in s if str(x).strip())
    return tuple(x.strip() for x in str(s).split(";") if x.strip())


def _parse_string_list(s):
    if isinstance(s, (tuple, list)):
        return tuple(str(x).strip() for x in s if str(x).strip())
    return tuple(x.strip() for x in str(s).split(",") if x.strip())


def _parse_interval(s):
    vals = _parse_number_list(s, float)
    if len(vals) != 2:
        raise ValueError(f"expected interval start,end, got {s!r}")
    return validate_guidance_interval(vals)


def validate_guidance_interval(interval, name="guidance interval"):
    start, end = (float(interval[0]), float(interval[1]))
    if start < 0.0 or end > 1.0 or start > end:
        raise ValueError(f"{name} must satisfy 0 <= start <= end <= 1")
    return (start, end)


def interval_active(t, interval):
    start, end = validate_guidance_interval(interval)
    t = float(t)
    return start <= t <= end


def format_interval(interval):
    start, end = validate_guidance_interval(interval)
    return f"{start:g},{end:g}"


def _seeded_randn(shape, device, seed):
    dev = torch.device(device)
    gen_device = "cpu" if dev.type == "mps" else dev.type
    try:
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(int(seed))
        return torch.randn(shape, generator=gen, device=device)
    except (RuntimeError, TypeError):
        torch.manual_seed(int(seed))
        return torch.randn(shape, device=device)


def apply_flow_time_shift(t, shift=1.0):
    """Shift data-time t through the equivalent rectified-flow noise schedule.

    The repo uses t=0 as pure noise and t=1 as clean data.  Image RF schedulers commonly
    shift the noise-time schedule toward higher-noise regions for harder/high-resolution
    generation.  Applying the shift to sigma=1-t and converting back keeps endpoints fixed.
    """
    shift = float(shift)
    if shift <= 0.0:
        raise ValueError("time_shift must be positive")
    if shift == 1.0:
        return t
    return t / (shift + (1.0 - shift) * t).clamp_min(1.0e-12)


def flow_time_schedule(steps, device=DEV, shift=1.0):
    steps = max(1, int(steps))
    base = torch.linspace(0.0, 1.0, steps + 1, device=device)
    return apply_flow_time_shift(base, shift=shift)


def sample_flow_times(batch, device=DEV, mode="uniform", logit_mean=0.0, logit_std=1.0,
                      time_shift=1.0):
    """Sample rectified-flow interpolation times.

    Uniform is the original rectified-flow baseline.  Logit-normal follows the SD3-style
    direction of spending more training mass on perceptually useful intermediate noise scales.
    """
    if mode == "uniform":
        t = torch.rand(batch, 1, 1, 1, device=device)
    elif mode == "logit-normal":
        eps = torch.randn(batch, 1, 1, 1, device=device) * float(logit_std) + float(logit_mean)
        t = torch.sigmoid(eps)
    else:
        raise ValueError(f"unknown time sampling mode {mode!r}")
    return apply_flow_time_shift(t, shift=time_shift)


def latent_stats_enabled(stats):
    return bool(stats and stats.get("mode", "none") != "none"
                and stats.get("mean") is not None and stats.get("std") is not None)


def latent_stats_to_device(stats, device):
    if not latent_stats_enabled(stats):
        return {"mode": "none", "n": int((stats or {}).get("n", 0))}
    return {
        "mode": stats.get("mode", "channel"),
        "n": int(stats.get("n", 0)),
        "mean": stats["mean"].to(device=device),
        "std": stats["std"].to(device=device),
    }


def attach_latent_stats(flow, stats):
    flow.latent_stats = latent_stats_to_device(stats, next(flow.parameters()).device)
    return flow


def flow_latent_stats(flow):
    return getattr(flow, "latent_stats", {"mode": "none", "n": 0})


def normalize_latent(z, stats):
    if not latent_stats_enabled(stats):
        return z
    mean = stats["mean"].to(device=z.device, dtype=z.dtype)
    std = stats["std"].to(device=z.device, dtype=z.dtype)
    return (z - mean) / std


def denormalize_latent(z, stats):
    if not latent_stats_enabled(stats):
        return z
    mean = stats["mean"].to(device=z.device, dtype=z.dtype)
    std = stats["std"].to(device=z.device, dtype=z.dtype)
    return z * std + mean


@torch.no_grad()
def estimate_latent_stats(ae, n=512, batch=64, seed=123, size=32, device=DEV,
                          mode="none", eps=1.0e-6):
    mode = str(mode)
    if mode == "none":
        return {"mode": "none", "n": 0}
    if mode not in ("global", "channel"):
        raise ValueError(f"unknown latent normalization mode {mode!r}")
    n = max(1, int(n))
    rng = np.random.default_rng(seed)
    ae.eval()
    count = 0
    seen = 0
    sums = sums2 = None
    while seen < n:
        b = min(batch, n - seen)
        x, _cond, _yc, _ys = _batch(b, rng, size=size, device=device)
        z = ae.encode(x).detach().float().cpu().double()
        if mode == "global":
            cur_sum = z.sum()
            cur_sum2 = z.pow(2).sum()
            cur_count = z.numel()
        else:
            cur_sum = z.sum(dim=(0, 2, 3))
            cur_sum2 = z.pow(2).sum(dim=(0, 2, 3))
            cur_count = z.shape[0] * z.shape[2] * z.shape[3]
        if sums is None:
            sums = torch.zeros_like(cur_sum)
            sums2 = torch.zeros_like(cur_sum2)
        sums += cur_sum
        sums2 += cur_sum2
        count += int(cur_count)
        seen += b
    mean = sums / max(1, count)
    var = (sums2 / max(1, count) - mean.pow(2)).clamp_min(float(eps) ** 2)
    std = var.sqrt()
    if mode == "global":
        mean = mean.view(1, 1, 1, 1)
        std = std.view(1, 1, 1, 1)
    else:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    return {
        "mode": mode,
        "n": int(seen),
        "mean": mean.float(),
        "std": std.float(),
    }


@torch.no_grad()
def estimate_latent_stats_records(ae, records, n=512, batch=64, seed=123, size=32, device=DEV,
                                  mode="none", eps=1.0e-6, crop_mode="center",
                                  hflip_prob=0.0):
    mode = str(mode)
    if mode == "none":
        return {"mode": "none", "n": 0}
    if mode not in ("global", "channel"):
        raise ValueError(f"unknown latent normalization mode {mode!r}")
    n = max(1, int(n))
    rng = np.random.default_rng(seed)
    ae.eval()
    count = 0
    seen = 0
    sums = sums2 = None
    while seen < n:
        b = min(batch, n - seen)
        x, _captions = sample_image_text_batch(
            records, rng, batch=b, size=size, device=device,
            crop_mode=crop_mode, hflip_prob=hflip_prob)
        z = ae.encode(x).detach().float().cpu().double()
        if mode == "global":
            cur_sum = z.sum()
            cur_sum2 = z.pow(2).sum()
            cur_count = z.numel()
        else:
            cur_sum = z.sum(dim=(0, 2, 3))
            cur_sum2 = z.pow(2).sum(dim=(0, 2, 3))
            cur_count = z.shape[0] * z.shape[2] * z.shape[3]
        if sums is None:
            sums = torch.zeros_like(cur_sum)
            sums2 = torch.zeros_like(cur_sum2)
        sums += cur_sum
        sums2 += cur_sum2
        count += int(cur_count)
        seen += b
    mean = sums / max(1, count)
    var = (sums2 / max(1, count) - mean.pow(2)).clamp_min(float(eps) ** 2)
    std = var.sqrt()
    if mode == "global":
        mean = mean.view(1, 1, 1, 1)
        std = std.view(1, 1, 1, 1)
    else:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    return {
        "mode": mode,
        "n": int(seen),
        "mean": mean.float(),
        "std": std.float(),
    }


@torch.no_grad()
def estimate_latent_stats_tensor(latents, n=512, seed=123, mode="none", eps=1.0e-6):
    mode = str(mode)
    if mode == "none":
        return {"mode": "none", "n": 0}
    if mode not in ("global", "channel"):
        raise ValueError(f"unknown latent normalization mode {mode!r}")
    latents = latents.detach().float().cpu()
    if latents.ndim != 4:
        raise ValueError(f"expected cached BCHW latents, got shape {tuple(latents.shape)}")
    total = int(latents.shape[0])
    if total <= 0:
        raise ValueError("latent cache is empty")
    n = min(max(1, int(n)), total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n, replace=total < n)
    z = latents[torch.tensor(idx, dtype=torch.long)].double()
    if mode == "global":
        mean = z.mean().view(1, 1, 1, 1)
        std = z.std(unbiased=False).clamp_min(float(eps)).view(1, 1, 1, 1)
    else:
        mean = z.mean(dim=(0, 2, 3)).view(1, -1, 1, 1)
        std = z.std(dim=(0, 2, 3), unbiased=False).clamp_min(float(eps)).view(1, -1, 1, 1)
    return {
        "mode": mode,
        "n": int(n),
        "mean": mean.float(),
        "std": std.float(),
    }


@torch.no_grad()
def build_image_latent_cache(ae, records, prompt_vocab, caption_max_len=64, max_records=0,
                             batch=64, seed=0, size=32, device=DEV, precision="fp32",
                             cond_source="tokens", cache_dir="", shard_size=1024,
                             include_image_embeddings=False, crop_mode="center",
                             hflip_prob=0.0):
    rows = list(records)
    if not rows:
        raise ValueError("cannot build latent cache from empty records")
    rng = np.random.default_rng(seed)
    if max_records and int(max_records) < len(rows):
        chosen_idx = rng.choice(len(rows), size=int(max_records), replace=False)
        rows = [rows[int(i)] for i in chosen_idx]
    if cond_source == "embedding":
        infer_text_embedding_dim(rows)
    if include_image_embeddings:
        infer_image_embedding_dim(rows)
    if cache_dir:
        return build_disk_image_latent_cache(
            ae, rows, prompt_vocab, caption_max_len=caption_max_len,
            batch=batch, size=size, device=device, precision=precision,
            cond_source=cond_source, cache_dir=cache_dir, shard_size=shard_size,
            include_image_embeddings=include_image_embeddings, seed=seed,
            crop_mode=crop_mode, hflip_prob=hflip_prob)
    latents, captions, embeddings, image_embeddings = [], [], [], []
    ae.eval()
    crop_rng = np.random.default_rng(seed + 17)
    for start in range(0, len(rows), max(1, int(batch))):
        chunk = rows[start:start + max(1, int(batch))]
        x = torch.stack([
            load_image_tensor(
                rec.path, size=size, device=device, crop_mode=crop_mode,
                hflip=(hflip_prob > 0.0 and float(crop_rng.random()) < float(hflip_prob)),
                rng=crop_rng)
            for rec in chunk
        ], dim=0)
        with amp_autocast(device, precision):
            z = ae.encode(x)
        latents.append(z.detach().float().cpu())
        captions.extend(rec.caption for rec in chunk)
        if cond_source == "embedding":
            embeddings.append(record_text_embedding_tensor(chunk, device="cpu"))
        if include_image_embeddings:
            image_embeddings.append(record_image_embedding_tensor(chunk, device="cpu"))
    cache = {
        "backend": "memory",
        "latents": torch.cat(latents, dim=0).contiguous(),
        "captions": captions,
        "caption_ids": (
            caption_ids(captions, prompt_vocab, max_len=caption_max_len, device="cpu")
            if cond_source == "tokens" else None
        ),
        "text_embeddings": (
            torch.cat(embeddings, dim=0).contiguous() if embeddings else None
        ),
        "image_embeddings": (
            torch.cat(image_embeddings, dim=0).contiguous() if image_embeddings else None
        ),
        "cond_source": cond_source,
        "has_image_embeddings": bool(image_embeddings),
        "records": len(rows),
        "latent_shape": tuple(int(x) for x in latents[0].shape[1:]),
        "bytes": int(sum(z.numel() * z.element_size() for z in latents)),
    }
    if cache["text_embeddings"] is not None:
        cache["bytes"] += int(
            cache["text_embeddings"].numel() * cache["text_embeddings"].element_size()
        )
    if cache["image_embeddings"] is not None:
        cache["bytes"] += int(
            cache["image_embeddings"].numel() * cache["image_embeddings"].element_size()
        )
    return cache


@torch.no_grad()
def build_disk_image_latent_cache(ae, rows, prompt_vocab, caption_max_len=64, batch=64,
                                  size=32, device=DEV, precision="fp32",
                                  cond_source="tokens", cache_dir="", shard_size=1024,
                                  include_image_embeddings=False, seed=0,
                                  crop_mode="center", hflip_prob=0.0):
    if not cache_dir:
        raise ValueError("cache_dir is required for disk latent cache")
    shard_size = int(shard_size)
    if shard_size <= 0:
        raise ValueError("flow cache shard size must be positive")
    rows = list(rows)
    if not rows:
        raise ValueError("cannot build latent cache from empty records")
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    shards = []
    total_bytes = 0
    latent_shape = None
    ae.eval()
    crop_rng = np.random.default_rng(seed + 17)
    for shard_i, start in enumerate(range(0, len(rows), shard_size)):
        chunk_rows = rows[start:start + shard_size]
        latents = []
        for enc_start in range(0, len(chunk_rows), max(1, int(batch))):
            enc_rows = chunk_rows[enc_start:enc_start + max(1, int(batch))]
            x = torch.stack([
                load_image_tensor(
                    rec.path, size=size, device=device, crop_mode=crop_mode,
                    hflip=(
                        hflip_prob > 0.0
                        and float(crop_rng.random()) < float(hflip_prob)
                    ),
                    rng=crop_rng)
                for rec in enc_rows
            ], dim=0)
            with amp_autocast(device, precision):
                z = ae.encode(x)
            latents.append(z.detach().float().cpu())
        captions = [rec.caption for rec in chunk_rows]
        shard = {
            "latents": torch.cat(latents, dim=0).contiguous(),
            "captions": captions,
            "caption_ids": (
                caption_ids(captions, prompt_vocab, max_len=caption_max_len, device="cpu")
                if cond_source == "tokens" else None
            ),
            "text_embeddings": (
                record_text_embedding_tensor(chunk_rows, device="cpu")
                if cond_source == "embedding" else None
            ),
            "image_embeddings": (
                record_image_embedding_tensor(chunk_rows, device="cpu")
                if include_image_embeddings else None
            ),
            "cond_source": cond_source,
            "has_image_embeddings": bool(include_image_embeddings),
            "start": int(start),
            "count": int(len(chunk_rows)),
        }
        if latent_shape is None:
            latent_shape = tuple(int(x) for x in shard["latents"].shape[1:])
        name = f"shard_{shard_i:06d}.pt"
        path = os.path.join(cache_dir, name)
        tmp_path = path + ".tmp"
        torch.save(shard, tmp_path)
        os.replace(tmp_path, path)
        file_bytes = int(os.path.getsize(path))
        total_bytes += file_bytes
        shards.append({"path": path, "file": name, "count": int(len(chunk_rows)),
                       "bytes": file_bytes})
    meta = {
        "format": "image_latent_cache_v1",
        "backend": "disk",
        "cache_dir": cache_dir,
        "records": int(len(rows)),
        "latent_shape": list(latent_shape or ()),
        "cond_source": cond_source,
        "has_image_embeddings": bool(include_image_embeddings),
        "shard_size": int(shard_size),
        "shards": [
            {"file": s["file"], "count": int(s["count"]), "bytes": int(s["bytes"])}
            for s in shards
        ],
        "bytes": int(total_bytes),
    }
    meta_path = os.path.join(cache_dir, "meta.json")
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    os.replace(tmp_meta, meta_path)
    total_bytes += int(os.path.getsize(meta_path))
    return {
        "backend": "disk",
        "cache_dir": cache_dir,
        "meta_path": meta_path,
        "records": int(len(rows)),
        "latent_shape": tuple(int(x) for x in latent_shape),
        "cond_source": cond_source,
        "has_image_embeddings": bool(include_image_embeddings),
        "shard_size": int(shard_size),
        "shards": shards,
        "bytes": int(total_bytes),
    }


def latent_cache_backend(cache):
    return str((cache or {}).get("backend", "memory" if cache is not None else ""))


def _latent_cache_shard_offsets(cache):
    offsets, pos = [], 0
    for shard in cache.get("shards", []):
        offsets.append(pos)
        pos += int(shard["count"])
    return offsets


def _load_latent_cache_shard(shard):
    return torch.load(shard["path"], map_location="cpu")


def _latent_cache_payload(source, ids=None, embeddings=None, image_embeddings=None):
    return {
        "cond_source": source,
        "caption_ids": ids,
        "text_embeddings": embeddings,
        "image_embeddings": image_embeddings,
    }


def sample_latent_cache(cache, rng, batch, device=DEV):
    backend = latent_cache_backend(cache)
    if backend == "disk":
        shards = list(cache.get("shards", []))
        if not shards:
            raise ValueError("disk latent cache has no shards")
        counts = np.asarray([int(s["count"]) for s in shards], dtype=np.float64)
        probs = counts / counts.sum()
        shard_i = int(rng.choice(len(shards), p=probs))
        shard = _load_latent_cache_shard(shards[shard_i])
        n = int(shard["latents"].shape[0])
        idx = torch.tensor(rng.integers(0, n, size=int(batch)), dtype=torch.long)
        z1 = shard["latents"][idx].to(device=device)
        image_embs = (
            shard["image_embeddings"][idx] if shard.get("image_embeddings") is not None else None
        )
        if cache["cond_source"] == "embedding":
            payload = _latent_cache_payload(
                "embedding", embeddings=shard["text_embeddings"][idx],
                image_embeddings=image_embs)
        else:
            payload = _latent_cache_payload(
                "tokens", ids=shard["caption_ids"][idx], image_embeddings=image_embs)
        return z1, payload
    idx_np = rng.integers(0, int(cache["records"]), size=int(batch))
    idx = torch.tensor(idx_np, dtype=torch.long)
    z1 = cache["latents"][idx].to(device=device)
    image_embs = (
        cache["image_embeddings"][idx] if cache.get("image_embeddings") is not None else None
    )
    if cache["cond_source"] == "embedding":
        payload = _latent_cache_payload(
            "embedding", embeddings=cache["text_embeddings"][idx],
            image_embeddings=image_embs)
    else:
        payload = _latent_cache_payload(
            "tokens", ids=cache["caption_ids"][idx], image_embeddings=image_embs)
    return z1, payload


@torch.no_grad()
def estimate_latent_stats_cache(cache, n=512, seed=123, mode="none"):
    backend = latent_cache_backend(cache)
    if backend != "disk":
        return estimate_latent_stats_tensor(cache["latents"], n=n, seed=seed, mode=mode)
    if mode == "none":
        return {"mode": "none", "n": 0}
    total = int(cache["records"])
    if total <= 0:
        raise ValueError("latent cache is empty")
    n = min(max(1, int(n)), total)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(total, size=n, replace=total < n)
    offsets = _latent_cache_shard_offsets(cache)
    grouped = {}
    for global_i in idxs:
        shard_i = int(np.searchsorted(offsets, int(global_i), side="right") - 1)
        grouped.setdefault(shard_i, []).append(int(global_i) - offsets[shard_i])
    samples = []
    for shard_i in sorted(grouped):
        shard = _load_latent_cache_shard(cache["shards"][shard_i])
        local = torch.tensor(grouped[shard_i], dtype=torch.long)
        samples.append(shard["latents"][local])
    latents = torch.cat(samples, dim=0)
    return estimate_latent_stats_tensor(latents, n=n, seed=seed + 1, mode=mode)


def latent_stats_state(stats):
    if not latent_stats_enabled(stats):
        return {"mode": "none", "n": int((stats or {}).get("n", 0))}
    return {
        "mode": stats.get("mode", "channel"),
        "n": int(stats.get("n", 0)),
        "mean": stats["mean"].detach().cpu(),
        "std": stats["std"].detach().cpu(),
    }


def latent_stats_report(stats):
    if not latent_stats_enabled(stats):
        return {"latent_normalize": "none", "latent_norm_n": 0}
    mean = stats["mean"].detach().float().cpu()
    std = stats["std"].detach().float().cpu()
    return {
        "latent_normalize": stats.get("mode", "channel"),
        "latent_norm_n": int(stats.get("n", 0)),
        "latent_norm_mean_abs": float(mean.abs().mean()),
        "latent_norm_std_mean": float(std.mean()),
        "latent_norm_std_min": float(std.min()),
        "latent_norm_std_max": float(std.max()),
    }


AE_RECON_LOSSES = ("mse", "l1", "hybrid")
TRAIN_PRECISIONS = ("fp32", "bf16", "fp16")


def _device_type(device):
    return torch.device(device).type


def amp_config(device, precision):
    precision = str(precision)
    if precision not in TRAIN_PRECISIONS:
        raise ValueError(f"unknown training precision {precision!r}")
    dev_type = _device_type(device)
    enabled = precision != "fp32" and dev_type == "cuda"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return {
        "requested": precision,
        "device_type": dev_type,
        "enabled": bool(enabled),
        "dtype": dtype,
        "dtype_name": str(dtype).replace("torch.", ""),
    }


def amp_autocast(device, precision):
    cfg = amp_config(device, precision)
    if not cfg["enabled"]:
        return nullcontext()
    return torch.autocast(device_type=cfg["device_type"], dtype=cfg["dtype"], enabled=True)


def amp_grad_scaler(device, precision):
    cfg = amp_config(device, precision)
    return torch.amp.GradScaler("cuda", enabled=bool(cfg["enabled"] and precision == "fp16"))


def image_gradient_loss(pred, target):
    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_target = target[..., :, 1:] - target[..., :, :-1]
    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_target = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)


def multiscale_recon_loss(pred, target, levels=3):
    losses = []
    cur_pred, cur_target = pred, target
    for _ in range(max(1, int(levels))):
        losses.append(F.l1_loss(cur_pred, cur_target))
        if min(cur_pred.shape[-2:]) < 4:
            break
        cur_pred = F.avg_pool2d(cur_pred, kernel_size=2, stride=2)
        cur_target = F.avg_pool2d(cur_target, kernel_size=2, stride=2)
    return torch.stack(losses).mean()


def reconstruction_loss_parts(pred, target, mode="mse", grad_w=0.0, ms_w=0.0,
                              latent=None, latent_reg_w=0.0):
    if mode not in AE_RECON_LOSSES:
        raise ValueError(f"unknown AE reconstruction loss {mode!r}")
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    if mode == "mse":
        base = mse
    elif mode == "l1":
        base = l1
    else:
        base = 0.5 * (mse + l1)
    loss = base
    parts = {
        "recon_mse": mse.detach(),
        "recon_l1": l1.detach(),
        "recon_base": base.detach(),
    }
    if grad_w > 0.0:
        grad = image_gradient_loss(pred, target)
        loss = loss + float(grad_w) * grad
        parts["recon_grad_l1"] = grad.detach()
    if ms_w > 0.0:
        ms = multiscale_recon_loss(pred, target)
        loss = loss + float(ms_w) * ms
        parts["recon_multiscale_l1"] = ms.detach()
    if latent is not None and latent_reg_w > 0.0:
        lat = latent.float().pow(2).mean()
        loss = loss + float(latent_reg_w) * lat
        parts["latent_l2"] = lat.detach()
    return loss, parts


def autoencoder_loss(out, x, yc, ys, fact_w=0.25, recon_loss="mse", grad_w=0.0,
                     ms_w=0.0, latent_reg_w=0.0):
    recon, parts = reconstruction_loss_parts(
        out["recon"], x, mode=recon_loss, grad_w=grad_w, ms_w=ms_w,
        latent=out.get("latent"), latent_reg_w=latent_reg_w)
    facts = F.cross_entropy(out["color"], yc) + F.cross_entropy(out["shape"], ys)
    parts["fact_ce"] = facts.detach()
    return recon + fact_w * facts, parts


def flow_uses_cond_tokens(flow):
    return bool(getattr(flow, "uses_cond_tokens", False))


def condition_vector(cond):
    return cond["vec"] if isinstance(cond, dict) else cond


def image_text_alignment_loss(aligner, z, cond, prefix="caption_align", mask=None):
    if aligner is None:
        zero = z.sum() * 0.0
        return zero, {}
    img_emb, txt_emb = aligner(z, cond)
    if mask is not None:
        mask = mask.to(device=img_emb.device, dtype=torch.bool).flatten()
        img_emb = img_emb[mask]
        txt_emb = txt_emb[mask]
    n = int(img_emb.shape[0])
    if n <= 0:
        zero = z.sum() * 0.0 + condition_vector(cond).sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_i2t_acc": zero.detach(),
            f"{prefix}_t2i_acc": zero.detach(),
            f"{prefix}_n": torch.tensor(0.0, device=z.device),
        }
    logits = aligner.scale().to(img_emb.dtype) * img_emb.matmul(txt_emb.t())
    targets = torch.arange(n, device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    )
    i2t = logits.argmax(dim=1).eq(targets).float().mean()
    t2i = logits.argmax(dim=0).eq(targets).float().mean()
    diag = logits.diag().mean()
    offdiag = ((logits.sum() - logits.diag().sum()) / max(1, n * n - n)
               if n > 1 else logits.diag().mean())
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_i2t_acc": i2t.detach(),
        f"{prefix}_t2i_acc": t2i.detach(),
        f"{prefix}_diag_logit": diag.detach(),
        f"{prefix}_offdiag_logit": offdiag.detach(),
        f"{prefix}_n": torch.tensor(float(n), device=z.device),
    }


def image_feature_alignment_loss(aligner, z, features, prefix="image_feature_align", mask=None):
    if aligner is None:
        zero = z.sum() * 0.0
        return zero, {}
    features = features.to(device=z.device)
    img_emb, feat_emb = aligner(z, features)
    if mask is not None:
        mask = mask.to(device=img_emb.device, dtype=torch.bool).flatten()
        img_emb = img_emb[mask]
        feat_emb = feat_emb[mask]
    n = int(img_emb.shape[0])
    if n <= 0:
        zero = z.sum() * 0.0 + features.sum() * 0.0
        return zero, {
            f"{prefix}_loss": zero.detach(),
            f"{prefix}_i2f_acc": zero.detach(),
            f"{prefix}_f2i_acc": zero.detach(),
            f"{prefix}_n": torch.tensor(0.0, device=z.device),
        }
    logits = aligner.scale().to(img_emb.dtype) * img_emb.matmul(feat_emb.t())
    targets = torch.arange(n, device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    )
    i2f = logits.argmax(dim=1).eq(targets).float().mean()
    f2i = logits.argmax(dim=0).eq(targets).float().mean()
    diag = logits.diag().mean()
    offdiag = ((logits.sum() - logits.diag().sum()) / max(1, n * n - n)
               if n > 1 else logits.diag().mean())
    return loss, {
        f"{prefix}_loss": loss.detach(),
        f"{prefix}_i2f_acc": i2f.detach(),
        f"{prefix}_f2i_acc": f2i.detach(),
        f"{prefix}_diag_logit": diag.detach(),
        f"{prefix}_offdiag_logit": offdiag.detach(),
        f"{prefix}_n": torch.tensor(float(n), device=z.device),
    }


def condition_batch(cond):
    return int(condition_vector(cond).shape[0])


def fact_condition_or_none(cond):
    vec = condition_vector(cond)
    if vec.ndim == 2 and vec.shape[1] == len(FACT_VOCAB):
        return vec
    return None


def zero_condition(cond):
    if not isinstance(cond, dict):
        return torch.zeros_like(cond)
    out = dict(cond)
    out["vec"] = torch.zeros_like(cond["vec"])
    out["tokens"] = torch.zeros_like(cond["tokens"])
    return out


def condition_dropout(cond, p=0.0):
    """Classifier-free condition dropout: zero whole condition rows during training."""
    if p <= 0.0:
        return cond
    vec = condition_vector(cond)
    if p >= 1.0:
        return zero_condition(cond)
    keep = (torch.rand(vec.shape[0], 1, device=vec.device) >= p).to(vec.dtype)
    if not isinstance(cond, dict):
        return cond * keep
    out = dict(cond)
    out["vec"] = cond["vec"] * keep
    out["tokens"] = cond["tokens"] * keep[:, None, :]
    return out


def semantic_guidance_step(ae, z, fact_cond, weight=0.0, step_size=1.0, mode="decoded",
                           eps=1.0e-6):
    """Classifier-style latent guidance using the learned semantic AE heads."""
    if weight <= 0.0:
        return z
    if fact_cond is None:
        raise ValueError("semantic guidance requires canonical fact conditions")
    if mode not in ("latent", "decoded"):
        raise ValueError(f"unknown semantic guidance mode {mode!r}")
    z_var = z.detach().requires_grad_(True)
    with torch.enable_grad():
        if mode == "latent":
            logits = ae.fact_logits(z_var)
        else:
            logits = ae(ae.decode(z_var))                  # decode/re-read, matching eval
        loss, _parts = semantic_fact_loss_from_logits(logits, fact_cond,
                                                      suffix="_guidance_ce")
        grad = torch.autograd.grad(loss, z_var, allow_unused=False)[0]
    flat = grad.flatten(1)
    denom = flat.norm(dim=1).view((-1,) + (1,) * (grad.ndim - 1)).clamp_min(float(eps))
    guided = z_var - float(weight) * float(step_size) * grad / denom
    return guided.detach()


def semantic_fact_loss_from_logits(logits, cond, suffix="_endpoint_ce"):
    losses = {}
    for pred, idxs in FACT_GROUPS.items():
        if pred not in logits:
            continue
        if logits[pred].shape[-1] != len(idxs):
            raise ValueError(
                f"fact head {pred!r} has {logits[pred].shape[-1]} classes, "
                f"but FACT_VOCAB has {len(idxs)}"
            )
        group = cond[:, list(idxs)]
        active = group.sum(dim=1) > 0
        if not bool(active.any()):
            continue
        target = group[active].argmax(dim=1)
        losses[pred] = F.cross_entropy(logits[pred][active], target)
    if not losses:
        zero = sum(v.sum() for v in logits.values()) * 0.0
        return zero, {}
    loss = sum(losses.values()) / len(losses)
    parts = {f"{pred}{suffix}": val.detach() for pred, val in losses.items()}
    return loss, parts


def semantic_endpoint_loss(ae, z_clean, cond):
    """REPA-style endpoint alignment: clean latent should decode to conditioned facts."""
    return semantic_fact_loss_from_logits(ae.fact_logits(z_clean), cond)


def fact_targets_from_condition(cond):
    targets = {}
    active = {}
    for pred, idxs in FACT_GROUPS.items():
        group = cond[:, list(idxs)]
        targets[pred] = group.argmax(dim=1)
        active[pred] = group.sum(dim=1) > 0
    return targets, active


@torch.no_grad()
def latent_intervention_diagnostic(ae, n=64, batch=64, seed=123, size=32, device=DEV,
                                   strength=1.0):
    """Probe whether latent fact directions are reusable and minimally entangled.

    The probe is intentionally data-derived: it estimates one prototype latent per fact value,
    edits held-out latents with prototype differences, then asks whether only the requested fact
    changes after decoding/re-encoding. This measures the FER/UFR property without injecting
    symbolic rendering rules into the model.
    """
    n = int(n)
    if n <= 0:
        return {}
    ae.eval()
    rng = np.random.default_rng(seed)
    proto_n = max(n, sum(len(idxs) for idxs in FACT_GROUPS.values()) * 4)
    sums = counts = None
    total = 0
    while total < proto_n:
        b = min(batch, proto_n - total)
        x, cond, _yc, _ys = _batch(b, rng, size=size, device=device)
        z = ae.encode(x)
        targets, active = fact_targets_from_condition(cond)
        if sums is None:
            sums = {
                pred: torch.zeros((len(idxs),) + tuple(z.shape[1:]), device=device)
                for pred, idxs in FACT_GROUPS.items()
            }
            counts = {
                pred: torch.zeros((len(idxs),), dtype=torch.long, device=device)
                for pred, idxs in FACT_GROUPS.items()
            }
        for pred, idxs in FACT_GROUPS.items():
            for val in range(len(idxs)):
                mask = active[pred] & targets[pred].eq(val)
                if bool(mask.any()):
                    sums[pred][val] += z[mask].sum(dim=0)
                    counts[pred][val] += int(mask.sum())
        total += b

    prototypes = {}
    for pred, vals in sums.items():
        valid = counts[pred] > 0
        if int(valid.sum()) < 2:
            continue
        denom = counts[pred].clamp_min(1).to(vals.dtype).view((-1,) + (1,) * (vals.ndim - 1))
        prototypes[pred] = vals / denom

    latent_target_ok = image_target_ok = target_total = 0
    collateral_ok = collateral_total = 0
    pred_target = {pred: [0, 0] for pred in prototypes}
    pred_collateral = {pred: [0, 0] for pred in prototypes}
    total = 0
    while total < n:
        b = min(batch, n - total)
        x, cond, _yc, _ys = _batch(b, rng, size=size, device=device)
        z = ae.encode(x)
        targets, active = fact_targets_from_condition(cond)
        for pred, proto in prototypes.items():
            classes = proto.shape[0]
            mask = active[pred]
            if not bool(mask.any()) or classes < 2:
                continue
            z_src = z[mask]
            src = targets[pred][mask]
            target = (src + 1) % classes
            delta = proto[target] - proto[src]
            z_edit = z_src + float(strength) * delta
            latent_logits = ae.fact_logits(z_edit)
            image = ae.decode(z_edit).clamp(-1.0, 1.0)
            image_logits = ae(image)
            latent_pred = latent_logits[pred].argmax(dim=1)
            image_pred = image_logits[pred].argmax(dim=1)
            ok_latent = int(latent_pred.eq(target).sum())
            ok_image = int(image_pred.eq(target).sum())
            count = int(target.numel())
            latent_target_ok += ok_latent
            image_target_ok += ok_image
            target_total += count
            pred_target[pred][0] += ok_image
            pred_target[pred][1] += count
            for other in prototypes:
                if other == pred or other not in image_logits:
                    continue
                other_src = targets[other][mask]
                other_ok = int(image_logits[other].argmax(dim=1).eq(other_src).sum())
                collateral_ok += other_ok
                collateral_total += count
                pred_collateral[other][0] += other_ok
                pred_collateral[other][1] += count
        total += b

    def div(num, den):
        return float(num / den) if den else 0.0

    target_acc = div(image_target_ok, target_total)
    collateral_acc = div(collateral_ok, collateral_total)
    return {
        "latent_intervention_n": int(target_total),
        "latent_intervention_strength": float(strength),
        "latent_intervention_direct_target_acc": div(latent_target_ok, target_total),
        "latent_intervention_image_target_acc": target_acc,
        "latent_intervention_collateral_acc": collateral_acc,
        "latent_intervention_score": float(target_acc * collateral_acc),
        "latent_intervention_target_by_fact": {
            pred: div(ok, total) for pred, (ok, total) in pred_target.items() if total
        },
        "latent_intervention_collateral_by_fact": {
            pred: div(ok, total) for pred, (ok, total) in pred_collateral.items() if total
        },
    }


def latent_intervention_training_loss(ae, z, cond, strength=1.0, decoded_w=1.0,
                                      collateral_w=1.0):
    """Differentiable version of the intervention probe for semantic AE training.

    Prototypes are estimated from the current batch and detached. The optimizer therefore learns
    to make fact directions usable without being allowed to satisfy the loss by moving the
    prototype targets themselves inside the same step.
    """
    targets, active = fact_targets_from_condition(cond)
    available_logits = ae.fact_logits(z)
    losses = []
    direct_losses, decoded_losses, collateral_losses = [], [], []
    n_edits = 0
    for pred, idxs in FACT_GROUPS.items():
        if pred not in available_logits:
            continue
        present = [val for val in range(len(idxs))
                   if bool((active[pred] & targets[pred].eq(val)).any())]
        if len(present) < 2:
            continue
        prototypes = {}
        for val in present:
            mask = active[pred] & targets[pred].eq(val)
            prototypes[val] = z[mask].detach().mean(dim=0)
        for pos, src_val in enumerate(present):
            mask = active[pred] & targets[pred].eq(src_val)
            if not bool(mask.any()):
                continue
            tgt_val = present[(pos + 1) % len(present)]
            z_edit = z[mask] + float(strength) * (prototypes[tgt_val] - prototypes[src_val])
            tgt = torch.full((z_edit.shape[0],), tgt_val, dtype=torch.long, device=z.device)
            direct_logits = ae.fact_logits(z_edit)
            decoded = ae.decode(z_edit)
            decoded_logits = ae(decoded)
            direct = F.cross_entropy(direct_logits[pred], tgt)
            decoded_target = F.cross_entropy(decoded_logits[pred], tgt)
            step_losses = [direct, decoded_w * decoded_target]
            direct_losses.append(direct)
            decoded_losses.append(decoded_target)
            n_edits += int(z_edit.shape[0])
            for other, _other_idxs in FACT_GROUPS.items():
                if other == pred or other not in decoded_logits:
                    continue
                other_tgt = targets[other][mask]
                other_loss = F.cross_entropy(decoded_logits[other], other_tgt)
                step_losses.append(collateral_w * other_loss)
                collateral_losses.append(other_loss)
            losses.append(sum(step_losses) / len(step_losses))
    if not losses:
        zero = z.sum() * 0.0
        return zero, {
            "intervention_loss": zero.detach(),
            "intervention_edits": torch.tensor(0.0, device=z.device),
        }
    total = sum(losses) / len(losses)

    def mean_or_zero(vals):
        return (sum(vals) / len(vals)).detach() if vals else total.detach() * 0.0

    return total, {
        "intervention_loss": total.detach(),
        "intervention_direct_ce": mean_or_zero(direct_losses),
        "intervention_decoded_ce": mean_or_zero(decoded_losses),
        "intervention_collateral_ce": mean_or_zero(collateral_losses),
        "intervention_edits": torch.tensor(float(n_edits), device=z.device),
    }


def latent_factor_orthogonality_loss(z, cond, eps=1.0e-6):
    """Penalize overlap between data-derived latent fact subspaces.

    FER shows up when factors are accurate but internally tangled.  This loss estimates one latent
    prototype per fact value from the current batch, centers those prototypes within each predicate,
    and discourages different predicate subspaces from pointing in the same latent directions.  It
    is generic over FACT_VOCAB predicate groups; no color/shape-specific rule is injected.
    """
    targets, active = fact_targets_from_condition(cond)
    bases = {}
    basis_count = 0
    for pred, idxs in FACT_GROUPS.items():
        present = [val for val in range(len(idxs))
                   if bool((active[pred] & targets[pred].eq(val)).any())]
        if len(present) < 2:
            continue
        protos = []
        for val in present:
            mask = active[pred] & targets[pred].eq(val)
            protos.append(z[mask].mean(dim=0))
        flat = torch.stack(protos).flatten(1)
        flat = flat - flat.mean(dim=0, keepdim=True)
        norms = flat.norm(dim=1)
        keep = norms > float(eps)
        if not bool(keep.any()):
            continue
        basis = flat[keep] / norms[keep, None].clamp_min(float(eps))
        bases[pred] = basis
        basis_count += int(basis.shape[0])

    preds = sorted(bases)
    losses = []
    for i, pred_a in enumerate(preds):
        for pred_b in preds[i + 1:]:
            sim = bases[pred_a].matmul(bases[pred_b].transpose(0, 1))
            losses.append(sim.pow(2).mean())
    if not losses:
        zero = z.sum() * 0.0
        return zero, {
            "factor_orth_loss": zero.detach(),
            "factor_orth_pairs": torch.tensor(0.0, device=z.device),
            "factor_orth_bases": torch.tensor(float(basis_count), device=z.device),
        }
    total = sum(losses) / len(losses)
    return total, {
        "factor_orth_loss": total.detach(),
        "factor_orth_pairs": torch.tensor(float(len(losses)), device=z.device),
        "factor_orth_bases": torch.tensor(float(basis_count), device=z.device),
    }


def latent_flow_losses(flow, z1, cond, cond_drop=0.0, ae=None, semantic_w=0.0,
                       semantic_cond=None, time_sampling="uniform", time_logit_mean=0.0,
                       time_logit_std=1.0, time_shift=1.0,
                       latent_stats=None, consistency_w=0.0,
                       text_aligner=None, text_align_w=0.0,
                       feature_aligner=None, image_features=None, feature_align_w=0.0):
    latent_stats = flow_latent_stats(flow) if latent_stats is None else latent_stats
    z1_model = normalize_latent(z1, latent_stats)
    x0 = torch.randn_like(z1_model)
    t = sample_flow_times(z1.shape[0], device=z1.device, mode=time_sampling,
                          logit_mean=time_logit_mean, logit_std=time_logit_std,
                          time_shift=time_shift)
    zt = (1.0 - t) * x0 + t * z1_model
    target = z1_model - x0
    cond_model = condition_dropout(cond, cond_drop)
    pred = flow(zt, t, cond_model)
    velocity = F.mse_loss(pred, target)
    total = velocity
    parts = {
        "velocity_mse": velocity.detach(),
        "time_mean": t.detach().mean(),
        "time_std": t.detach().std(unbiased=False),
        "time_shift": torch.tensor(float(time_shift), device=z1.device),
    }
    if consistency_w > 0.0:
        t2 = sample_flow_times(z1.shape[0], device=z1.device, mode=time_sampling,
                               logit_mean=time_logit_mean, logit_std=time_logit_std,
                               time_shift=time_shift)
        zt2 = (1.0 - t2) * x0 + t2 * z1_model
        pred2 = flow(zt2, t2, cond_model)
        endpoint1 = zt + (1.0 - t) * pred
        endpoint2 = zt2 + (1.0 - t2) * pred2
        consistency = 0.5 * (
            F.mse_loss(endpoint1, endpoint2.detach())
            + F.mse_loss(endpoint2, endpoint1.detach())
        )
        total = total + float(consistency_w) * consistency
        parts["endpoint_consistency_mse"] = consistency.detach()
    z_clean = None
    if ae is not None and semantic_w > 0.0:
        z_clean = denormalize_latent(zt + (1.0 - t) * pred, latent_stats)
        sem_cond = cond if semantic_cond is None else semantic_cond
        cond_vec = condition_vector(cond_model)
        keep = cond_vec.detach().abs().sum(dim=1, keepdim=True).gt(0).to(sem_cond.dtype)
        semantic, sem_parts = semantic_endpoint_loss(ae, z_clean, sem_cond * keep)
        total = total + semantic_w * semantic
        parts["semantic_endpoint_ce"] = semantic.detach()
        parts.update(sem_parts)
    if text_aligner is not None and text_align_w > 0.0:
        if z_clean is None:
            z_clean = denormalize_latent(zt + (1.0 - t) * pred, latent_stats)
        cond_vec = condition_vector(cond_model)
        keep = cond_vec.detach().abs().sum(dim=1).gt(0)
        text_align, text_parts = image_text_alignment_loss(
            text_aligner, z_clean, cond_model, prefix="flow_caption_align", mask=keep)
        total = total + float(text_align_w) * text_align
        parts.update(text_parts)
    if feature_aligner is not None and feature_align_w > 0.0:
        if image_features is None:
            raise ValueError("flow feature alignment requires image_features")
        if z_clean is None:
            z_clean = denormalize_latent(zt + (1.0 - t) * pred, latent_stats)
        feature_align, feature_parts = image_feature_alignment_loss(
            feature_aligner, z_clean, image_features, prefix="flow_image_feature_align")
        total = total + float(feature_align_w) * feature_align
        parts.update(feature_parts)
    return total, parts


@torch.no_grad()
def flow_endpoint_metrics(flow, z1, cond, time_sampling="uniform", time_logit_mean=0.0,
                          time_logit_std=1.0, time_shift=1.0, latent_stats=None):
    """Measure whether different times on the same path predict the same clean endpoint."""
    latent_stats = flow_latent_stats(flow) if latent_stats is None else latent_stats
    z1_model = normalize_latent(z1, latent_stats)
    x0 = torch.randn_like(z1_model)
    t1 = sample_flow_times(z1.shape[0], device=z1.device, mode=time_sampling,
                           logit_mean=time_logit_mean, logit_std=time_logit_std,
                           time_shift=time_shift)
    t2 = sample_flow_times(z1.shape[0], device=z1.device, mode=time_sampling,
                           logit_mean=time_logit_mean, logit_std=time_logit_std,
                           time_shift=time_shift)

    def endpoint_at(t):
        zt = (1.0 - t) * x0 + t * z1_model
        pred = flow(zt, t, cond)
        return zt + (1.0 - t) * pred

    endpoint1 = endpoint_at(t1)
    endpoint2 = endpoint_at(t2)
    return {
        "latent_endpoint_mse": float(F.mse_loss(endpoint1, z1_model).detach().cpu()),
        "latent_endpoint_consistency_mse": float(
            F.mse_loss(endpoint1, endpoint2).detach().cpu()
        ),
        "latent_endpoint_time_gap": float((t1 - t2).abs().mean().detach().cpu()),
    }


def latent_flow_loss(flow, z1, cond, cond_drop=0.0, time_sampling="uniform",
                     time_logit_mean=0.0, time_logit_std=1.0, time_shift=1.0,
                     latent_stats=None):
    loss, _parts = latent_flow_losses(flow, z1, cond, cond_drop=cond_drop,
                                      time_sampling=time_sampling,
                                      time_logit_mean=time_logit_mean,
                                      time_logit_std=time_logit_std,
                                      time_shift=time_shift,
                                      latent_stats=latent_stats)
    return loss


def model_condition(specs, fact_cond, cond_mode="facts", conditioner=None, prompt_vocab=None,
                    prompt_templates=DEFAULT_PROMPT_TEMPLATES, rng=None, device=DEV,
                    return_tokens=False):
    if cond_mode == "facts":
        if return_tokens:
            mask = torch.zeros((fact_cond.shape[0], 1), dtype=torch.bool, device=fact_cond.device)
            return {"vec": fact_cond, "tokens": fact_cond[:, None, :], "mask": mask}
        return fact_cond
    if cond_mode == "text":
        if conditioner is None or prompt_vocab is None:
            raise ValueError("text conditioning requires conditioner and prompt_vocab")
        ids = prompt_ids(specs, prompt_vocab, rng=rng, templates=prompt_templates, device=device)
        return conditioner(ids, return_tokens=return_tokens)
    raise ValueError(f"unknown condition mode {cond_mode!r}")


def caption_condition(captions, conditioner, vocab, max_len=64, device=DEV, return_tokens=False):
    if conditioner is None or vocab is None:
        raise ValueError("caption conditioning requires conditioner and caption vocab")
    ids = caption_ids(captions, vocab, max_len=max_len, device=device)
    return conditioner(ids, return_tokens=return_tokens)


def caption_condition_ids(ids, conditioner, device=DEV, return_tokens=False):
    if conditioner is None:
        raise ValueError("caption id conditioning requires conditioner")
    return conditioner(ids.to(device=device), return_tokens=return_tokens)


def infer_text_embedding_dim(records):
    dims = {len(rec.text_embedding) for rec in records if rec.text_embedding is not None}
    if not dims:
        return 0
    if len(dims) != 1:
        raise ValueError(f"manifest text embeddings have mixed dimensions: {sorted(dims)}")
    dim = next(iter(dims))
    missing = sum(1 for rec in records if rec.text_embedding is None)
    if missing:
        raise ValueError(f"{missing} manifest records are missing text embeddings")
    return int(dim)


def infer_image_embedding_dim(records):
    dims = {len(rec.image_embedding) for rec in records if rec.image_embedding is not None}
    if not dims:
        return 0
    if len(dims) != 1:
        raise ValueError(f"manifest image embeddings have mixed dimensions: {sorted(dims)}")
    dim = next(iter(dims))
    missing = sum(1 for rec in records if rec.image_embedding is None)
    if missing:
        raise ValueError(f"{missing} manifest records are missing image embeddings")
    return int(dim)


def resolve_caption_cond_source(source, records):
    source = str(source or "tokens")
    if source not in ("tokens", "embedding", "auto"):
        raise ValueError(f"unknown caption condition source {source!r}")
    if source == "tokens":
        return "tokens", 0
    try:
        dim = infer_text_embedding_dim(records)
    except ValueError:
        if source == "auto":
            return "tokens", 0
        raise
    if dim <= 0:
        if source == "embedding":
            raise ValueError("caption condition source 'embedding' requires text_embedding rows")
        return "tokens", 0
    return "embedding", dim


def record_text_embedding_tensor(records, device=DEV):
    dim = infer_text_embedding_dim(records)
    if dim <= 0:
        raise ValueError("records do not have text embeddings")
    return torch.tensor([rec.text_embedding for rec in records], dtype=torch.float32,
                        device=device)


def record_image_embedding_tensor(records, device=DEV):
    dim = infer_image_embedding_dim(records)
    if dim <= 0:
        raise ValueError("records do not have image embeddings")
    return torch.tensor([rec.image_embedding for rec in records], dtype=torch.float32,
                        device=device)


def caption_record_condition(captions, records, conditioner, vocab, source="tokens",
                             max_len=64, device=DEV, return_tokens=False):
    if source == "embedding":
        embs = record_text_embedding_tensor(records, device=device)
        return conditioner(embs, return_tokens=return_tokens)
    return caption_condition(captions, conditioner, vocab, max_len=max_len, device=device,
                             return_tokens=return_tokens)


def cached_caption_condition(cache, idx, conditioner, source="tokens", device=DEV,
                             return_tokens=False):
    if "latents" not in cache and "shards" not in cache:
        return cached_caption_payload_condition(
            cache, conditioner, source=source, device=device, return_tokens=return_tokens)
    if source == "embedding":
        embs = cache["text_embeddings"][idx].to(device=device)
        return conditioner(embs, return_tokens=return_tokens)
    ids = cache["caption_ids"][idx]
    return caption_condition_ids(ids, conditioner, device=device, return_tokens=return_tokens)


def cached_caption_payload_condition(payload, conditioner, source="tokens", device=DEV,
                                     return_tokens=False):
    if source == "embedding":
        embs = payload["text_embeddings"].to(device=device)
        return conditioner(embs, return_tokens=return_tokens)
    ids = payload["caption_ids"]
    return caption_condition_ids(ids, conditioner, device=device, return_tokens=return_tokens)


@torch.no_grad()
def guided_velocity(flow, z, t, cond, cfg_scale=1.0):
    """Classifier-free guidance in latent velocity space."""
    if cfg_scale == 1.0:
        return flow(z, t, cond)
    uncond = zero_condition(cond)
    v_uncond = flow(z, t, uncond)
    v_cond = flow(z, t, cond)
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def sample_latents(flow, cond, latent_shape=(16, 8, 8), steps=16, device=DEV, seed=0,
                   cfg_scale=1.0, ae=None, semantic_cond=None, semantic_guidance_w=0.0,
                   semantic_guidance_mode="decoded", sample_method="euler",
                   cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                   semantic_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                   sample_time_shift=1.0):
    batch = condition_batch(cond)
    latent_stats = flow_latent_stats(flow)
    z = _seeded_randn((batch,) + tuple(latent_shape), device=device, seed=seed)
    flow.eval()
    if ae is not None:
        ae.eval()
    if semantic_cond is None:
        semantic_cond = fact_condition_or_none(cond)
    if semantic_guidance_w > 0.0 and ae is None:
        raise ValueError("semantic guidance requires ae")
    if sample_method not in SAMPLE_METHODS:
        raise ValueError(f"unknown sample method {sample_method!r}")
    cfg_interval = validate_guidance_interval(cfg_interval, name="cfg_interval")
    semantic_guidance_interval = validate_guidance_interval(
        semantic_guidance_interval, name="semantic_guidance_interval")
    schedule = flow_time_schedule(steps, device=device, shift=sample_time_shift)
    for i in range(steps):
        t_scalar = float(schedule[i].detach().cpu())
        t_next_scalar = float(schedule[i + 1].detach().cpu())
        dt = t_next_scalar - t_scalar
        t = torch.full((batch, 1, 1, 1), t_scalar, device=device)
        step_cfg = cfg_scale if interval_active(t_scalar, cfg_interval) else 1.0
        v0 = guided_velocity(flow, z, t, cond, cfg_scale=step_cfg)
        if sample_method == "euler":
            z = z + dt * v0
        else:
            z_pred = z + dt * v0
            t_next = torch.full((batch, 1, 1, 1), t_next_scalar, device=device)
            next_cfg = cfg_scale if interval_active(t_next_scalar, cfg_interval) else 1.0
            v1 = guided_velocity(flow, z_pred, t_next, cond, cfg_scale=next_cfg)
            z = z + 0.5 * dt * (v0 + v1)
        if (semantic_guidance_w > 0.0
                and interval_active(t_scalar, semantic_guidance_interval)):
            z_raw = denormalize_latent(z, latent_stats)
            z_raw = semantic_guidance_step(ae, z_raw, semantic_cond,
                                           weight=semantic_guidance_w,
                                           step_size=dt, mode=semantic_guidance_mode)
            z = normalize_latent(z_raw, latent_stats)
    return denormalize_latent(z, latent_stats)


@torch.no_grad()
def sample_images(ae, flow, cond, latent_shape=(16, 8, 8), steps=16, device=DEV, seed=0,
                  cfg_scale=1.0, semantic_cond=None, semantic_guidance_w=0.0,
                  semantic_guidance_mode="decoded", sample_method="euler",
                  cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                  semantic_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                  sample_time_shift=1.0):
    z = sample_latents(flow, cond, latent_shape=latent_shape, steps=steps, device=device,
                       seed=seed, cfg_scale=cfg_scale, ae=ae, semantic_cond=semantic_cond,
                       semantic_guidance_w=semantic_guidance_w,
                       semantic_guidance_mode=semantic_guidance_mode,
                       sample_method=sample_method, cfg_interval=cfg_interval,
                       semantic_guidance_interval=semantic_guidance_interval,
                       sample_time_shift=sample_time_shift)
    ae.eval()
    return ae.decode(z).clamp(-1.0, 1.0)


def _rgb8_from_samples(samples):
    if samples.ndim != 4 or samples.shape[1] != 3:
        raise ValueError(f"expected BCHW RGB samples, got shape {tuple(samples.shape)}")
    arr = samples.detach().cpu().float().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).round().to(torch.uint8)
    return arr.permute(0, 2, 3, 1).contiguous().numpy()


def write_ppm_grid(samples, path, rows, cols, pad=2, bg=32):
    """Write a binary PPM grid without image-library dependencies."""
    arr = _rgb8_from_samples(samples)
    n, h, w, c = arr.shape
    if c != 3:
        raise ValueError(f"expected RGB samples, got {c} channels")
    rows, cols = int(rows), int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("grid rows/cols must be positive")
    if rows * cols < n:
        raise ValueError(f"grid {rows}x{cols} cannot fit {n} samples")
    pad = max(0, int(pad))
    out_h = rows * h + max(0, rows - 1) * pad
    out_w = cols * w + max(0, cols - 1) * pad
    grid = np.full((out_h, out_w, 3), int(bg), dtype=np.uint8)
    for idx in range(n):
        r = idx // cols
        col = idx % cols
        y0 = r * (h + pad)
        x0 = col * (w + pad)
        grid[y0:y0 + h, x0:x0 + w] = arr[idx]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"P6\n{out_w} {out_h}\n255\n".encode("ascii"))
        f.write(grid.tobytes())
    return {
        "sample_grid": path,
        "sample_grid_rows": rows,
        "sample_grid_cols": cols,
        "sample_grid_n": int(n),
        "sample_grid_tile_h": int(h),
        "sample_grid_tile_w": int(w),
        "sample_grid_pad": pad,
        "sample_grid_format": "ppm",
    }


def make_condition_grid_specs(samples_per_combo=1):
    samples_per_combo = int(samples_per_combo)
    if samples_per_combo <= 0:
        raise ValueError("samples_per_combo must be positive")
    specs = []
    for color in COLORS:
        for shape in SHAPES:
            specs.extend(ObjectSpec("p0", color, shape) for _ in range(samples_per_combo))
    return specs, len(COLORS), len(SHAPES) * samples_per_combo


@torch.no_grad()
def save_sample_grid(ae, flow, path, size=32, device=DEV, cond_mode="facts", conditioner=None,
                     prompt_vocab=None, prompt_templates=DEFAULT_PROMPT_TEMPLATES,
                     cfg_scale=1.0, sample_steps=4, sample_method="euler",
                     semantic_guidance_w=0.0, semantic_guidance_mode="decoded",
                     cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                     semantic_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                     samples_per_combo=1, seed=0, sample_time_shift=1.0):
    ae.eval()
    flow.eval()
    specs, rows, cols = make_condition_grid_specs(samples_per_combo=samples_per_combo)
    fact_rows = [
        fact_condition(object_facts(spec), device=device).detach().cpu().numpy()
        for spec in specs
    ]
    fact_cond = torch.tensor(np.stack(fact_rows), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    cond = model_condition(specs, fact_cond, cond_mode=cond_mode, conditioner=conditioner,
                           prompt_vocab=prompt_vocab, prompt_templates=prompt_templates,
                           rng=rng, device=device, return_tokens=flow_uses_cond_tokens(flow))
    sample = sample_images(ae, flow, cond, latent_shape=ae_latent_shape(ae, size),
                           steps=sample_steps, device=device, seed=seed, cfg_scale=cfg_scale,
                           semantic_cond=fact_cond, semantic_guidance_w=semantic_guidance_w,
                           semantic_guidance_mode=semantic_guidance_mode,
                           sample_method=sample_method, cfg_interval=cfg_interval,
                           semantic_guidance_interval=semantic_guidance_interval,
                           sample_time_shift=sample_time_shift)
    meta = write_ppm_grid(sample, path, rows=rows, cols=cols)
    meta.update({
        "sample_grid_cfg_scale": float(cfg_scale),
        "sample_grid_sample_steps": int(sample_steps),
        "sample_grid_sample_time_shift": float(sample_time_shift),
        "sample_grid_sample_method": sample_method,
        "sample_grid_cfg_interval": list(validate_guidance_interval(cfg_interval)),
        "sample_grid_semantic_guidance_w": float(semantic_guidance_w),
        "sample_grid_semantic_guidance_mode": semantic_guidance_mode,
        "sample_grid_semantic_guidance_interval": list(validate_guidance_interval(
            semantic_guidance_interval)),
        "sample_grid_cond_mode": cond_mode,
        "sample_grid_seed": int(seed),
        "sample_grid_samples_per_combo": int(samples_per_combo),
    })
    return meta


@torch.no_grad()
def save_caption_sample_grid(ae, flow, records, path, size=32, device=DEV, conditioner=None,
                             prompt_vocab=None, caption_max_len=64, cfg_scale=1.0,
                             sample_steps=4, sample_method="euler",
                             cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                             samples=16, seed=0, caption_cond_source="tokens",
                             sample_time_shift=1.0):
    ae.eval()
    flow.eval()
    rng = np.random.default_rng(seed)
    n = min(max(1, int(samples)), len(records))
    idx = rng.choice(len(records), size=n, replace=len(records) < n)
    chosen = [records[int(i)] for i in idx]
    captions = [rec.caption for rec in chosen]
    cond = caption_record_condition(
        captions, chosen, conditioner, prompt_vocab, source=caption_cond_source,
        max_len=caption_max_len, device=device, return_tokens=flow_uses_cond_tokens(flow))
    sample = sample_images(ae, flow, cond, latent_shape=ae_latent_shape(ae, size),
                           steps=sample_steps, device=device, seed=seed, cfg_scale=cfg_scale,
                           sample_method=sample_method, cfg_interval=cfg_interval,
                           sample_time_shift=sample_time_shift)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    meta = write_ppm_grid(sample, path, rows=rows, cols=cols)
    meta.update({
        "sample_grid_cfg_scale": float(cfg_scale),
        "sample_grid_sample_steps": int(sample_steps),
        "sample_grid_sample_time_shift": float(sample_time_shift),
        "sample_grid_sample_method": sample_method,
        "sample_grid_cfg_interval": list(validate_guidance_interval(cfg_interval)),
        "sample_grid_cond_mode": "caption",
        "sample_grid_caption_cond_source": caption_cond_source,
        "sample_grid_seed": int(seed),
        "sample_grid_caption_count": int(n),
        "sample_grid_captions": captions[:min(5, len(captions))],
    })
    return meta


@torch.no_grad()
def conditional_roundtrip(ae, flow, size=32, device=DEV, cfg_scale=1.0, sample_steps=4,
                          samples_per_combo=1, seed=20, cond_mode="facts", conditioner=None,
                          prompt_vocab=None, prompt_templates=DEFAULT_PROMPT_TEMPLATES,
                          semantic_guidance_w=0.0, semantic_guidance_mode="decoded",
                          sample_method="euler", cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                          semantic_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                          sample_time_shift=1.0):
    """Generate from every canonical color/shape request and re-read facts from the image."""
    samples_per_combo = int(samples_per_combo)
    if samples_per_combo <= 0:
        return {
            "sample_roundtrip_n": 0,
            "sample_roundtrip_color_acc": 0.0,
            "sample_roundtrip_shape_acc": 0.0,
            "sample_roundtrip_both_acc": 0.0,
            "conditional_sample_mse": 0.0,
        }
    ae.eval()
    flow.eval()
    got_c = got_s = got_both = total = 0
    mses = []
    latent_shape = ae_latent_shape(ae, size)
    for ci, color in enumerate(COLORS):
        for si, shape in enumerate(SHAPES):
            spec = ObjectSpec("p0", color, shape)
            fact_cond = fact_condition(object_facts(spec), device=device)[None].repeat(
                samples_per_combo, 1)
            specs = [spec] * samples_per_combo
            rng = np.random.default_rng(seed + ci * 101 + si * 17)
            cond = model_condition(specs, fact_cond, cond_mode=cond_mode,
                                   conditioner=conditioner, prompt_vocab=prompt_vocab,
                                   prompt_templates=prompt_templates, rng=rng, device=device,
                                   return_tokens=flow_uses_cond_tokens(flow))
            sample = sample_images(ae, flow, cond, latent_shape=latent_shape, steps=sample_steps,
                                   device=device, seed=seed + ci * 101 + si * 17,
                                   cfg_scale=cfg_scale, semantic_cond=fact_cond,
                                   semantic_guidance_w=semantic_guidance_w,
                                   semantic_guidance_mode=semantic_guidance_mode,
                                   sample_method=sample_method, cfg_interval=cfg_interval,
                                   semantic_guidance_interval=semantic_guidance_interval,
                                   sample_time_shift=sample_time_shift)
            out = ae(sample)
            pc = out["color"].argmax(-1)
            ps = out["shape"].argmax(-1)
            target_c = torch.full((samples_per_combo,), ci, dtype=torch.long, device=device)
            target_s = torch.full((samples_per_combo,), si, dtype=torch.long, device=device)
            ok_c = pc.eq(target_c)
            ok_s = ps.eq(target_s)
            got_c += int(ok_c.sum())
            got_s += int(ok_s.sum())
            got_both += int((ok_c & ok_s).sum())
            total += samples_per_combo
            target = torch.tensor(render_object(spec, size=size) * 2.0 - 1.0,
                                  dtype=torch.float32, device=device)[None]
            target = target.expand_as(sample)
            mses.append(float(F.mse_loss(sample, target).detach().cpu()))
    return {
        "sample_roundtrip_n": int(total),
        "sample_roundtrip_color_acc": got_c / total,
        "sample_roundtrip_shape_acc": got_s / total,
        "sample_roundtrip_both_acc": got_both / total,
        "conditional_sample_mse": float(np.mean(mses)),
    }


@torch.no_grad()
def evaluate(ae, flow, n=128, batch=64, seed=10, size=32, device=DEV, cfg_scale=1.0,
             sample_steps=4, roundtrip_samples=1, cond_mode="facts", conditioner=None,
             prompt_vocab=None, prompt_templates=DEFAULT_PROMPT_TEMPLATES,
             intervention_samples=0, semantic_guidance_w=0.0,
             semantic_guidance_mode="decoded", sample_method="euler",
             cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
             semantic_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
             sample_time_shift=1.0, time_shift=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ae.eval()
    flow.eval()
    recon_losses, flow_losses = [], []
    endpoint_mses, endpoint_consistency_mses, endpoint_time_gaps = [], [], []
    factor_orth_losses, factor_orth_pairs, factor_orth_bases = [], [], []
    got_c = got_s = total = 0
    latent_means, latent_stds = [], []
    while total < n:
        b = min(batch, n - total)
        x, fact_cond, yc, ys, specs = _batch(b, rng, size=size, device=device, return_specs=True)
        out = ae(x)
        z = out["latent"]
        recon_losses.append(float(F.mse_loss(out["recon"], x).detach().cpu()))
        factor_orth, factor_parts = latent_factor_orthogonality_loss(z, fact_cond)
        if float(factor_parts["factor_orth_pairs"].detach().cpu()) > 0.0:
            factor_orth_losses.append(float(factor_orth.detach().cpu()))
            factor_orth_pairs.append(float(factor_parts["factor_orth_pairs"].detach().cpu()))
            factor_orth_bases.append(float(factor_parts["factor_orth_bases"].detach().cpu()))
        cond = model_condition(specs, fact_cond, cond_mode=cond_mode, conditioner=conditioner,
                               prompt_vocab=prompt_vocab, prompt_templates=prompt_templates,
                               rng=rng, device=device, return_tokens=flow_uses_cond_tokens(flow))
        flow_losses.append(float(latent_flow_loss(
            flow, z, cond, time_shift=time_shift).detach().cpu()))
        endpoint_metrics = flow_endpoint_metrics(flow, z, cond, time_shift=time_shift)
        endpoint_mses.append(endpoint_metrics["latent_endpoint_mse"])
        endpoint_consistency_mses.append(endpoint_metrics["latent_endpoint_consistency_mse"])
        endpoint_time_gaps.append(endpoint_metrics["latent_endpoint_time_gap"])
        got_c += int(out["color"].argmax(-1).eq(yc).sum())
        got_s += int(out["shape"].argmax(-1).eq(ys).sum())
        latent_means.append(float(z.mean().detach().cpu()))
        latent_stds.append(float(z.std().detach().cpu()))
        total += b

    spec = ObjectSpec("p0", "red", "circle")
    fact_cond = fact_condition(object_facts(spec), device=device)[None]
    cond = model_condition([spec], fact_cond, cond_mode=cond_mode, conditioner=conditioner,
                           prompt_vocab=prompt_vocab, prompt_templates=prompt_templates,
                           rng=rng, device=device, return_tokens=flow_uses_cond_tokens(flow))
    sample = sample_images(ae, flow, cond, latent_shape=ae_latent_shape(ae, size),
                           steps=sample_steps, device=device, seed=seed, cfg_scale=cfg_scale,
                           semantic_cond=fact_cond, semantic_guidance_w=semantic_guidance_w,
                           semantic_guidance_mode=semantic_guidance_mode,
                           sample_method=sample_method, cfg_interval=cfg_interval,
                           semantic_guidance_interval=semantic_guidance_interval,
                           sample_time_shift=sample_time_shift)
    target = torch.tensor(render_object(spec, size=size) * 2.0 - 1.0,
                          dtype=torch.float32, device=device)[None]
    report = {
        "n": int(total),
        "recon_mse": float(np.mean(recon_losses)),
        "latent_velocity_mse": float(np.mean(flow_losses)),
        "latent_color_acc": got_c / total,
        "latent_shape_acc": got_s / total,
        "latent_mean": float(np.mean(latent_means)),
        "latent_std": float(np.mean(latent_stds)),
        "sample_min": float(sample.min().detach().cpu()),
        "sample_max": float(sample.max().detach().cpu()),
        "sample_center_target_mse": float(F.mse_loss(sample, target).detach().cpu()),
        "latent_endpoint_mse": float(np.mean(endpoint_mses)),
        "latent_endpoint_consistency_mse": float(np.mean(endpoint_consistency_mses)),
        "latent_endpoint_time_gap": float(np.mean(endpoint_time_gaps)),
        "cfg_scale": float(cfg_scale),
        "cfg_interval": list(validate_guidance_interval(cfg_interval)),
        "sample_steps": int(sample_steps),
        "sample_time_shift": float(sample_time_shift),
        "time_shift": float(time_shift),
        "sample_method": sample_method,
        "semantic_guidance_w": float(semantic_guidance_w),
        "semantic_guidance_mode": semantic_guidance_mode,
        "semantic_guidance_interval": list(validate_guidance_interval(
            semantic_guidance_interval)),
    }
    if factor_orth_losses:
        report.update({
            "latent_factor_orth_loss": float(np.mean(factor_orth_losses)),
            "latent_factor_orth_pairs": float(np.mean(factor_orth_pairs)),
            "latent_factor_orth_bases": float(np.mean(factor_orth_bases)),
        })
    report.update(conditional_roundtrip(ae, flow, size=size, device=device, cfg_scale=cfg_scale,
                                        sample_steps=sample_steps,
                                        samples_per_combo=roundtrip_samples, seed=seed + 17,
                                        cond_mode=cond_mode, conditioner=conditioner,
                                        prompt_vocab=prompt_vocab,
                                        prompt_templates=prompt_templates,
                                        semantic_guidance_w=semantic_guidance_w,
                                        semantic_guidance_mode=semantic_guidance_mode,
                                        sample_method=sample_method,
                                        cfg_interval=cfg_interval,
                                        semantic_guidance_interval=semantic_guidance_interval,
                                        sample_time_shift=sample_time_shift))
    if intervention_samples:
        report.update(latent_intervention_diagnostic(
            ae, n=intervention_samples, batch=batch, seed=seed + 31, size=size, device=device))
    report["cond_mode"] = cond_mode
    return report


@torch.no_grad()
def evaluate_image_records(ae, flow, records, n=128, batch=64, seed=10, size=32, device=DEV,
                           conditioner=None, prompt_vocab=None, caption_max_len=64,
                           cfg_scale=1.0, sample_steps=4, sample_method="euler",
                           cfg_interval=DEFAULT_GUIDANCE_INTERVAL, text_aligner=None,
                           image_feature_aligner=None,
                           caption_cond_source="tokens", sample_time_shift=1.0,
                           time_shift=1.0):
    rng = np.random.default_rng(seed)
    ae.eval()
    flow.eval()
    recon_losses, flow_losses = [], []
    endpoint_mses, endpoint_consistency_mses, endpoint_time_gaps = [], [], []
    latent_means, latent_stds = [], []
    align_losses, align_i2t, align_t2i = [], [], []
    feature_losses, feature_i2f, feature_f2i = [], [], []
    total = 0
    while total < n:
        b = min(batch, n - total)
        x, captions, chosen_records = sample_image_text_batch(
            records, rng, batch=b, size=size, device=device, return_records=True)
        out = ae(x)
        z = out["latent"]
        recon_losses.append(float(F.mse_loss(out["recon"], x).detach().cpu()))
        cond = caption_record_condition(
            captions, chosen_records, conditioner, prompt_vocab, source=caption_cond_source,
            max_len=caption_max_len, device=device, return_tokens=flow_uses_cond_tokens(flow))
        if text_aligner is not None:
            _align_loss, align_parts = image_text_alignment_loss(
                text_aligner, z, cond, prefix="caption_retrieval")
            align_losses.append(float(align_parts["caption_retrieval_loss"].detach().cpu()))
            align_i2t.append(float(align_parts["caption_retrieval_i2t_acc"].detach().cpu()))
            align_t2i.append(float(align_parts["caption_retrieval_t2i_acc"].detach().cpu()))
        if image_feature_aligner is not None:
            image_features = record_image_embedding_tensor(chosen_records, device=device)
            _feature_loss, feature_parts = image_feature_alignment_loss(
                image_feature_aligner, z, image_features, prefix="image_feature_retrieval")
            feature_losses.append(
                float(feature_parts["image_feature_retrieval_loss"].detach().cpu()))
            feature_i2f.append(
                float(feature_parts["image_feature_retrieval_i2f_acc"].detach().cpu()))
            feature_f2i.append(
                float(feature_parts["image_feature_retrieval_f2i_acc"].detach().cpu()))
        flow_losses.append(float(latent_flow_loss(
            flow, z, cond, time_shift=time_shift).detach().cpu()))
        endpoint_metrics = flow_endpoint_metrics(flow, z, cond, time_shift=time_shift)
        endpoint_mses.append(endpoint_metrics["latent_endpoint_mse"])
        endpoint_consistency_mses.append(endpoint_metrics["latent_endpoint_consistency_mse"])
        endpoint_time_gaps.append(endpoint_metrics["latent_endpoint_time_gap"])
        latent_means.append(float(z.mean().detach().cpu()))
        latent_stds.append(float(z.std().detach().cpu()))
        total += b

    sample_x, sample_captions, sample_records = sample_image_text_batch(
        records, np.random.default_rng(seed + 17), batch=min(batch, max(1, sample_steps)),
        size=size, device=device, return_records=True)
    sample_cond = caption_record_condition(
        sample_captions, sample_records, conditioner, prompt_vocab, source=caption_cond_source,
        max_len=caption_max_len, device=device, return_tokens=flow_uses_cond_tokens(flow))
    sample = sample_images(ae, flow, sample_cond,
                           latent_shape=ae_latent_shape(ae, size),
                           steps=sample_steps, device=device, seed=seed, cfg_scale=cfg_scale,
                           sample_method=sample_method, cfg_interval=cfg_interval,
                           sample_time_shift=sample_time_shift)
    report = {
        "n": int(total),
        "recon_mse": float(np.mean(recon_losses)),
        "latent_velocity_mse": float(np.mean(flow_losses)),
        "latent_endpoint_mse": float(np.mean(endpoint_mses)),
        "latent_endpoint_consistency_mse": float(np.mean(endpoint_consistency_mses)),
        "latent_endpoint_time_gap": float(np.mean(endpoint_time_gaps)),
        "latent_mean": float(np.mean(latent_means)),
        "latent_std": float(np.mean(latent_stds)),
        "sample_min": float(sample.min().detach().cpu()),
        "sample_max": float(sample.max().detach().cpu()),
        "caption_sample_mse": float(F.mse_loss(sample, sample_x[:sample.shape[0]]).detach().cpu()),
        "cfg_scale": float(cfg_scale),
        "cfg_interval": list(validate_guidance_interval(cfg_interval)),
        "sample_steps": int(sample_steps),
        "sample_time_shift": float(sample_time_shift),
        "time_shift": float(time_shift),
        "sample_method": sample_method,
        "cond_mode": "text",
        "caption_cond_source": caption_cond_source,
        "data_mode": "image_manifest",
        "eval_captions": sample_captions[:min(3, len(sample_captions))],
    }
    if text_aligner is not None:
        sample_z = ae.encode(sample)
        _sample_align_loss, sample_align_parts = image_text_alignment_loss(
            text_aligner, sample_z, sample_cond, prefix="generated_caption_retrieval")
        if align_losses:
            report.update({
                "caption_retrieval_loss": float(np.mean(align_losses)),
                "caption_retrieval_i2t_acc": float(np.mean(align_i2t)),
                "caption_retrieval_t2i_acc": float(np.mean(align_t2i)),
            })
        report.update({
            "generated_caption_retrieval_loss": float(
                sample_align_parts["generated_caption_retrieval_loss"].detach().cpu()),
            "generated_caption_retrieval_i2t_acc": float(
                sample_align_parts["generated_caption_retrieval_i2t_acc"].detach().cpu()),
            "generated_caption_retrieval_t2i_acc": float(
                sample_align_parts["generated_caption_retrieval_t2i_acc"].detach().cpu()),
            "generated_caption_retrieval_n": int(
                sample_align_parts["generated_caption_retrieval_n"].detach().cpu()),
        })
    if image_feature_aligner is not None:
        sample_z = ae.encode(sample)
        sample_features = record_image_embedding_tensor(
            sample_records[:sample.shape[0]], device=device)
        _sample_feature_loss, sample_feature_parts = image_feature_alignment_loss(
            image_feature_aligner, sample_z, sample_features,
            prefix="generated_image_feature_retrieval")
        if feature_losses:
            report.update({
                "image_feature_retrieval_loss": float(np.mean(feature_losses)),
                "image_feature_retrieval_i2f_acc": float(np.mean(feature_i2f)),
                "image_feature_retrieval_f2i_acc": float(np.mean(feature_f2i)),
            })
        report.update({
            "generated_image_feature_retrieval_loss": float(
                sample_feature_parts["generated_image_feature_retrieval_loss"].detach().cpu()),
            "generated_image_feature_retrieval_i2f_acc": float(
                sample_feature_parts["generated_image_feature_retrieval_i2f_acc"].detach().cpu()),
            "generated_image_feature_retrieval_f2i_acc": float(
                sample_feature_parts["generated_image_feature_retrieval_f2i_acc"].detach().cpu()),
            "generated_image_feature_retrieval_n": int(
                sample_feature_parts["generated_image_feature_retrieval_n"].detach().cpu()),
        })
    report.update(summarize_records(records))
    return report


def load_flow_state(flow, state_dict):
    missing, unexpected = flow.load_state_dict(state_dict, strict=False)
    tolerated_missing = [k for k in missing if ".gate." in k]
    bad_missing = sorted(set(missing) - set(tolerated_missing))
    if bad_missing or unexpected:
        raise RuntimeError(
            f"flow checkpoint mismatch: missing={bad_missing}, unexpected={list(unexpected)}"
        )
    return {"missing": list(missing), "tolerated_missing": tolerated_missing}


def clone_state_dict(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def ema_effective_decay(target_decay, step, warmup=True):
    if target_decay <= 0.0:
        return 0.0
    if not warmup:
        return float(target_decay)
    # Use a generic warmup so high target decays do not average mostly initialization
    # during short runs. This approaches the requested decay as training gets longer.
    warm = (1.0 + float(step)) / (10.0 + float(step))
    return float(min(target_decay, warm))


@torch.no_grad()
def update_ema_state(ema_state, module, decay):
    state = module.state_dict()
    for key, val in state.items():
        cur = val.detach()
        if key not in ema_state:
            ema_state[key] = cur.clone()
        elif torch.is_floating_point(ema_state[key]):
            ema_state[key].mul_(decay).add_(cur, alpha=1.0 - decay)
        else:
            ema_state[key].copy_(cur)


def load_checkpoint(path, device=DEV, prefer_ema=True):
    ckpt = torch.load(path, map_location=device)
    report = ckpt.get("report", {})
    latent_ch = int(ckpt.get("latent_ch", report.get("latent_ch", 16)))
    image_size = int(ckpt.get("image_size", report.get("image_size", 32)))
    hidden = int(ckpt.get("hidden", report.get("hidden", 64)))
    flow_arch = ckpt.get("flow_arch", report.get("flow_arch", "conv"))
    ae_arch = ckpt.get("ae_arch", report.get("ae_arch", "semantic"))
    latent_downsample = int(ckpt.get("latent_downsample",
                                     report.get("latent_downsample", 4)))
    ae_res_blocks = int(ckpt.get("ae_res_blocks", report.get("ae_res_blocks", 0)))
    latent_max_tokens = int(ckpt.get("latent_max_tokens",
                                     report.get("latent_max_tokens", 256)))
    dit_depth = int(ckpt.get("dit_depth", report.get("dit_depth", 3)))
    dit_heads = int(ckpt.get("dit_heads", report.get("dit_heads", 4)))
    dit_head_width_mult = int(ckpt.get("dit_head_width_mult",
                                       report.get("dit_head_width_mult", 1)))
    dit_qk_norm = bool(ckpt.get("dit_qk_norm", report.get("dit_qk_norm", False)))
    dit_attn_impl = str(ckpt.get("dit_attn_impl", report.get("dit_attn_impl", "manual")))
    if dit_attn_impl not in MMDIT_ATTN_IMPLS:
        raise ValueError(f"unknown MM-DiT attention implementation {dit_attn_impl!r}")
    cond_mode = ckpt.get("cond_mode", report.get("cond_mode", "facts"))
    data_mode = ckpt.get("data_mode", report.get("data_mode", "synthetic_factors"))
    cond_dim = int(ckpt.get("cond_dim", report.get("cond_dim", len(FACT_VOCAB))))
    caption_cond_source = ckpt.get("caption_cond_source",
                                   report.get("caption_cond_source", "tokens"))
    text_embedding_in_dim = int(ckpt.get(
        "text_embedding_in_dim", report.get("text_embedding_in_dim", 0)) or 0)
    text_embed_dim = int(ckpt.get("text_embed_dim", report.get("text_embed_dim", 128)) or 128)
    image_embedding_in_dim = int(ckpt.get(
        "image_embedding_in_dim", report.get("image_embedding_in_dim", 0)) or 0)
    image_feature_embed_dim = int(ckpt.get(
        "image_feature_embed_dim", report.get("image_feature_embed_dim", 128)) or 128)
    caption_max_len = int(ckpt.get("caption_max_len", report.get("caption_max_len", 32)) or 32)
    if cond_mode == "text" and data_mode != "image_manifest":
        caption_max_len = 32
    prompt_templates = tuple(ckpt.get("prompt_templates", report.get("prompt_templates", []))
                             or DEFAULT_PROMPT_TEMPLATES)
    prompt_vocab = ckpt.get("prompt_vocab") or None
    conditioner = None
    if cond_mode == "text":
        if caption_cond_source == "embedding":
            if text_embedding_in_dim <= 0:
                raise ValueError("embedding-conditioned checkpoint is missing text_embedding_in_dim")
            conditioner = PrecomputedTextConditioner(
                text_embedding_in_dim, cond_dim=cond_dim, hidden=hidden).to(device)
        else:
            if prompt_vocab is None:
                prompt_vocab = build_prompt_vocab(prompt_templates)
            conditioner = PromptConditioner(len(prompt_vocab), cond_dim=cond_dim,
                                            hidden=hidden, max_len=caption_max_len).to(device)
        if caption_cond_source != "embedding" and prompt_vocab is None:
            prompt_vocab = build_prompt_vocab(prompt_templates)
        cond_state = ckpt["conditioner_state_dict"]
        if prefer_ema and ckpt.get("conditioner_ema_state_dict"):
            cond_state = ckpt["conditioner_ema_state_dict"]
        conditioner.load_state_dict(cond_state)
        conditioner.eval()
    text_aligner = None
    if ckpt.get("text_aligner_state_dict"):
        text_aligner = ImageTextAligner(
            latent_ch=latent_ch, cond_dim=cond_dim, hidden=hidden,
            embed_dim=text_embed_dim).to(device)
        text_aligner.load_state_dict(ckpt["text_aligner_state_dict"])
        text_aligner.eval()
    image_feature_aligner = None
    if ckpt.get("image_feature_aligner_state_dict"):
        if image_embedding_in_dim <= 0:
            raise ValueError(
                "image-feature-aligned checkpoint is missing image_embedding_in_dim")
        image_feature_aligner = ImageFeatureAligner(
            latent_ch=latent_ch, feature_dim=image_embedding_in_dim, hidden=hidden,
            embed_dim=image_feature_embed_dim).to(device)
        image_feature_aligner.load_state_dict(ckpt["image_feature_aligner_state_dict"])
        image_feature_aligner.eval()
    ae = make_autoencoder(ae_arch=ae_arch, latent_ch=latent_ch, hidden=hidden,
                          latent_downsample=latent_downsample,
                          ae_res_blocks=ae_res_blocks).to(device)
    flow = make_flow(flow_arch=flow_arch, latent_ch=latent_ch, hidden=hidden,
                     dit_depth=dit_depth, dit_heads=dit_heads, cond_dim=cond_dim,
                     dit_head_width_mult=dit_head_width_mult,
                     latent_max_tokens=latent_max_tokens,
                     dit_qk_norm=dit_qk_norm,
                     dit_attn_impl=dit_attn_impl).to(device)
    latent_stats = latent_stats_to_device(
        ckpt.get("latent_stats", {"mode": ckpt.get("latent_normalize", "none")}),
        device)
    attach_latent_stats(flow, latent_stats)
    ae.load_state_dict(ckpt["autoencoder_state_dict"])
    flow_state = ckpt["flow_state_dict"]
    ema_available = bool(ckpt.get("flow_ema_state_dict"))
    ema_loaded = bool(prefer_ema and ema_available)
    if ema_loaded:
        flow_state = ckpt["flow_ema_state_dict"]
    flow_load = load_flow_state(flow, flow_state)
    attach_text_aligner(flow, text_aligner)
    attach_image_feature_aligner(flow, image_feature_aligner)
    ae.eval()
    flow.eval()
    return ae, flow, conditioner, prompt_vocab, prompt_templates, {
        "checkpoint": path,
        "checkpoint_report": report,
        "image_size": image_size,
        "latent_ch": latent_ch,
        "ae_arch": ae_arch,
        "latent_downsample": int(getattr(ae, "downsample", latent_downsample)),
        "ae_res_blocks": int(ae_res_blocks) if ae_arch == "residual" else 0,
        "latent_max_tokens": int(latent_max_tokens),
        "hidden": hidden,
        "flow_arch": flow_arch,
        "dit_depth": dit_depth if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "dit_heads": dit_heads if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "dit_head_width_mult": (
            dit_head_width_mult if flow_arch in ("dit", "crossdit", "mmdit") else 1
        ),
        "dit_qk_norm": bool(dit_qk_norm) if flow_arch == "mmdit" else False,
        "dit_attn_impl": dit_attn_impl if flow_arch == "mmdit" else "manual",
        "adaptive_modulation": bool(getattr(flow, "uses_adaptive_modulation", False)),
        "residual_gating": bool(getattr(flow, "uses_residual_gating", False)),
        "flow_load": flow_load,
        "ema_available": ema_available,
        "ema_loaded": ema_loaded,
        "flow_ema_decay": float(ckpt.get("flow_ema_decay", report.get("flow_ema_decay", 0.0))),
        "flow_consistency_w": float(ckpt.get(
            "flow_consistency_w", report.get("flow_consistency_w", 0.0))),
        "time_shift": float(ckpt.get("time_shift", report.get("time_shift", 1.0))),
        "sample_time_shift": float(ckpt.get(
            "sample_time_shift", report.get("sample_time_shift",
                                             report.get("time_shift", 1.0)))),
        "flow_ema_warmup": bool(ckpt.get("flow_ema_warmup",
                                         report.get("flow_ema_warmup", False))),
        "flow_ema_effective_decay": float(ckpt.get(
            "flow_ema_effective_decay", report.get("flow_ema_effective_decay", 0.0))),
        **latent_stats_report(latent_stats),
        "cond_mode": cond_mode,
        "data_mode": data_mode,
        "image_manifest": ckpt.get("image_manifest", report.get("image_manifest", "")),
        "image_root": ckpt.get("image_root", report.get("image_root", "")),
        "image_split": ckpt.get("image_split", report.get("image_split", "")),
        "caption_max_len": caption_max_len,
        "caption_cond_source": caption_cond_source,
        "text_embedding_in_dim": int(text_embedding_in_dim),
        "cond_dim": cond_dim,
        "text_aligner": text_aligner is not None,
        "text_embed_dim": text_embed_dim,
        "image_feature_aligner": image_feature_aligner is not None,
        "image_embedding_in_dim": int(image_embedding_in_dim),
        "image_feature_embed_dim": int(image_feature_embed_dim),
        "image_text_align_w": float(ckpt.get(
            "image_text_align_w", report.get("image_text_align_w", 0.0))),
        "flow_text_align_w": float(ckpt.get(
            "flow_text_align_w", report.get("flow_text_align_w", 0.0))),
        "image_feature_align_w": float(ckpt.get(
            "image_feature_align_w", report.get("image_feature_align_w", 0.0))),
        "flow_feature_align_w": float(ckpt.get(
            "flow_feature_align_w", report.get("flow_feature_align_w", 0.0))),
    }


@torch.no_grad()
def sampler_sweep(ae, flow, cfg_scales=(1.0, 1.5), sample_steps_list=(4, 8),
                  n=128, batch=64, seed=10, size=32, device=DEV, roundtrip_samples=1,
                  cond_mode="facts", conditioner=None, prompt_vocab=None,
                  prompt_templates=DEFAULT_PROMPT_TEMPLATES, semantic_guidance_w=0.0,
                  semantic_guidance_weights=None, semantic_guidance_mode="decoded",
                  sample_method="euler", sample_methods=None,
                  cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                  semantic_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                  sample_time_shift=1.0, time_shift=1.0):
    if semantic_guidance_weights is None:
        semantic_guidance_weights = (semantic_guidance_w,)
    if sample_methods is None:
        sample_methods = (sample_method,)
    rows = []
    for cfg_scale in cfg_scales:
        for sample_steps in sample_steps_list:
            for method in sample_methods:
                for guidance_w in semantic_guidance_weights:
                    row = evaluate(ae, flow, n=n, batch=batch, seed=seed, size=size,
                                   device=device, cfg_scale=float(cfg_scale),
                                   sample_steps=int(sample_steps), roundtrip_samples=roundtrip_samples,
                                   cond_mode=cond_mode, conditioner=conditioner,
                                   prompt_vocab=prompt_vocab, prompt_templates=prompt_templates,
                                   semantic_guidance_w=float(guidance_w),
                                   semantic_guidance_mode=semantic_guidance_mode,
                                   sample_method=method,
                                   cfg_interval=cfg_interval,
                                   semantic_guidance_interval=semantic_guidance_interval,
                                   sample_time_shift=sample_time_shift,
                                   time_shift=time_shift)
                    row["sweep_key"] = (
                        f"cfg={float(cfg_scale):g};steps={int(sample_steps)};"
                        f"method={method};sem={float(guidance_w):g};"
                        f"shift={float(sample_time_shift):g};"
                        f"cfgint={format_interval(cfg_interval)};"
                        f"semint={format_interval(semantic_guidance_interval)}"
                    )
                    rows.append(row)
    return rows


@torch.no_grad()
def image_record_sweep(ae, flow, records, cfg_scales=(1.0, 1.5), sample_steps_list=(4, 8),
                       n=128, batch=64, seed=10, size=32, device=DEV, conditioner=None,
                       prompt_vocab=None, caption_max_len=64, sample_method="euler",
                       sample_methods=None, cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                       text_aligner=None, image_feature_aligner=None,
                       caption_cond_source="tokens", sample_time_shift=1.0,
                       time_shift=1.0):
    if sample_methods is None:
        sample_methods = (sample_method,)
    rows = []
    for cfg_scale in cfg_scales:
        for sample_steps in sample_steps_list:
            for method in sample_methods:
                row = evaluate_image_records(
                    ae, flow, records, n=n, batch=batch, seed=seed, size=size,
                    device=device, conditioner=conditioner, prompt_vocab=prompt_vocab,
                    caption_max_len=caption_max_len, cfg_scale=float(cfg_scale),
                    sample_steps=int(sample_steps), sample_method=method,
                    cfg_interval=cfg_interval, text_aligner=text_aligner,
                    image_feature_aligner=image_feature_aligner,
                    caption_cond_source=caption_cond_source,
                    sample_time_shift=sample_time_shift, time_shift=time_shift)
                row["semantic_guidance_w"] = 0.0
                row["semantic_guidance_mode"] = "none"
                row["semantic_guidance_interval"] = list(DEFAULT_GUIDANCE_INTERVAL)
                row["sweep_key"] = (
                    f"cfg={float(cfg_scale):g};steps={int(sample_steps)};"
                    f"method={method};shift={float(sample_time_shift):g};"
                    f"cfgint={format_interval(cfg_interval)}"
                )
                rows.append(row)
    return rows


SWEEP_METRICS = (
    "sample_roundtrip_color_acc",
    "sample_roundtrip_shape_acc",
    "sample_roundtrip_both_acc",
    "conditional_sample_mse",
    "caption_sample_mse",
    "caption_retrieval_loss",
    "caption_retrieval_i2t_acc",
    "caption_retrieval_t2i_acc",
    "generated_caption_retrieval_loss",
    "generated_caption_retrieval_i2t_acc",
    "generated_caption_retrieval_t2i_acc",
    "image_feature_retrieval_loss",
    "image_feature_retrieval_i2f_acc",
    "image_feature_retrieval_f2i_acc",
    "generated_image_feature_retrieval_loss",
    "generated_image_feature_retrieval_i2f_acc",
    "generated_image_feature_retrieval_f2i_acc",
    "sample_center_target_mse",
    "recon_mse",
    "latent_velocity_mse",
    "latent_endpoint_mse",
    "latent_endpoint_consistency_mse",
    "latent_endpoint_time_gap",
)


EVAL_WEIGHT_MODES = ("raw", "ema", "auto")


def report_selection_key(report):
    inf = 1.0e30
    conditional_mse = report.get("conditional_sample_mse", report.get("caption_sample_mse", inf))
    center_mse = report.get("sample_center_target_mse", report.get("recon_mse", inf))
    return (
        float(report.get("sample_roundtrip_both_acc", 0.0)),
        float(report.get("sample_roundtrip_shape_acc", 0.0)),
        float(report.get("sample_roundtrip_color_acc", 0.0)),
        float(report.get("generated_image_feature_retrieval_i2f_acc", 0.0)),
        float(report.get("image_feature_retrieval_i2f_acc", 0.0)),
        float(report.get("generated_caption_retrieval_i2t_acc", 0.0)),
        float(report.get("caption_retrieval_i2t_acc", 0.0)),
        -float(conditional_mse),
        -float(center_mse),
        -float(report.get("latent_velocity_mse", inf)),
        -float(report.get("latent_endpoint_consistency_mse", inf)),
    )


def aggregate_selection_key(report):
    inf = 1.0e30
    conditional_mse = report.get("conditional_sample_mse_mean",
                                 report.get("caption_sample_mse_mean", inf))
    center_mse = report.get("sample_center_target_mse_mean", report.get("recon_mse_mean", inf))
    return (
        float(report.get("sample_roundtrip_both_acc_mean", 0.0)),
        float(report.get("sample_roundtrip_shape_acc_mean", 0.0)),
        float(report.get("sample_roundtrip_color_acc_mean", 0.0)),
        float(report.get("generated_image_feature_retrieval_i2f_acc_mean", 0.0)),
        float(report.get("image_feature_retrieval_i2f_acc_mean", 0.0)),
        float(report.get("generated_caption_retrieval_i2t_acc_mean", 0.0)),
        float(report.get("caption_retrieval_i2t_acc_mean", 0.0)),
        -float(conditional_mse),
        -float(center_mse),
        -float(report.get("latent_velocity_mse_mean", inf)),
        -float(report.get("latent_endpoint_consistency_mse_mean", inf)),
    )


def eval_report_summary(report):
    keys = (
        "eval_weight_mode",
        "cfg_scale",
        "cfg_interval",
        "sample_steps",
        "sample_time_shift",
        "time_shift",
        "sample_method",
        "semantic_guidance_w",
        "semantic_guidance_mode",
        "semantic_guidance_interval",
        "sample_roundtrip_n",
        "sample_roundtrip_color_acc",
        "sample_roundtrip_shape_acc",
        "sample_roundtrip_both_acc",
        "conditional_sample_mse",
        "caption_sample_mse",
        "caption_retrieval_i2t_acc",
        "caption_retrieval_t2i_acc",
        "generated_caption_retrieval_i2t_acc",
        "generated_caption_retrieval_t2i_acc",
        "image_feature_retrieval_i2f_acc",
        "image_feature_retrieval_f2i_acc",
        "generated_image_feature_retrieval_i2f_acc",
        "generated_image_feature_retrieval_f2i_acc",
        "sample_center_target_mse",
        "recon_mse",
        "latent_velocity_mse",
        "latent_endpoint_mse",
        "latent_endpoint_consistency_mse",
        "latent_endpoint_time_gap",
        "latent_color_acc",
        "latent_shape_acc",
        "latent_intervention_n",
        "latent_intervention_direct_target_acc",
        "latent_intervention_image_target_acc",
        "latent_intervention_collateral_acc",
        "latent_intervention_score",
        "latent_factor_orth_loss",
        "latent_factor_orth_pairs",
        "latent_factor_orth_bases",
    )
    return {k: report[k] for k in keys if k in report}


def aggregate_sweep_rows(rows):
    grouped = {}
    for row in rows:
        key = (float(row["cfg_scale"]), int(row["sample_steps"]),
               str(row.get("sample_method", "euler")),
               float(row.get("sample_time_shift", 1.0)),
               float(row.get("semantic_guidance_w", 0.0)),
               str(row.get("semantic_guidance_mode", "decoded")),
               tuple(float(x) for x in row.get("cfg_interval", DEFAULT_GUIDANCE_INTERVAL)),
               tuple(float(x) for x in row.get("semantic_guidance_interval",
                                                DEFAULT_GUIDANCE_INTERVAL)))
        grouped.setdefault(key, []).append(row)
    out = []
    for (cfg_scale, sample_steps, sample_method, sample_time_shift, semantic_guidance_w,
         semantic_guidance_mode, cfg_interval, semantic_guidance_interval), group in sorted(
             grouped.items()):
        agg = {
            "sweep_key": (
                f"cfg={cfg_scale:g};steps={sample_steps};method={sample_method};"
                f"shift={sample_time_shift:g};sem={semantic_guidance_w:g};"
                f"cfgint={format_interval(cfg_interval)};"
                f"semint={format_interval(semantic_guidance_interval)}"
            ),
            "cfg_scale": float(cfg_scale),
            "cfg_interval": list(cfg_interval),
            "sample_steps": int(sample_steps),
            "sample_method": sample_method,
            "sample_time_shift": float(sample_time_shift),
            "semantic_guidance_w": float(semantic_guidance_w),
            "semantic_guidance_mode": semantic_guidance_mode,
            "semantic_guidance_interval": list(semantic_guidance_interval),
            "runs": len(group),
            "eval_seeds": [int(r["eval_seed"]) for r in group if "eval_seed" in r],
        }
        for metric in SWEEP_METRICS:
            vals = np.asarray([float(r[metric]) for r in group if metric in r],
                              dtype=np.float64)
            if vals.size == 0:
                continue
            agg[f"{metric}_mean"] = float(vals.mean())
            agg[f"{metric}_std"] = float(vals.std())
            agg[f"{metric}_min"] = float(vals.min())
            agg[f"{metric}_max"] = float(vals.max())
        out.append(agg)
    return out


def evaluate_checkpoint(path, cfg_scales=(1.0, 1.5), sample_steps_list=(4, 8),
                        n=128, batch=64, seed=10, eval_seeds=None, size=32, device=DEV,
                        roundtrip_samples=1, prefer_ema=True, weight_mode=None,
                        intervention_samples=0, semantic_guidance_w=0.0,
                        semantic_guidance_weights=None, semantic_guidance_mode="decoded",
                        sample_method="euler", sample_methods=None,
                        cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                        semantic_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                        eval_image_manifest="", eval_image_root="", eval_image_split="eval",
                        eval_image_min_aesthetic=None, eval_image_max_records=0,
                        sample_time_shift=None):
    if weight_mode is None:
        weight_mode = "ema" if prefer_ema else "raw"
    if weight_mode not in EVAL_WEIGHT_MODES:
        raise ValueError(f"unknown checkpoint weight mode {weight_mode!r}")
    eval_seeds = tuple(eval_seeds) if eval_seeds is not None else (seed,)

    def run_mode(mode):
        ae, flow, conditioner, prompt_vocab, prompt_templates, meta = load_checkpoint(
            path, device=device, prefer_ema=(mode == "ema"))
        actual_mode = "ema" if meta["ema_loaded"] else "raw"
        eval_size = int(size or meta.get("image_size", 32) or 32)
        actual_sample_time_shift = (
            float(meta.get("sample_time_shift", meta.get("time_shift", 1.0)))
            if sample_time_shift is None else float(sample_time_shift)
        )
        if actual_sample_time_shift <= 0.0:
            raise ValueError("sample_time_shift must be positive")
        train_time_shift = float(meta.get("time_shift", actual_sample_time_shift))
        manifest_path = eval_image_manifest
        if not manifest_path and meta.get("data_mode") == "image_manifest":
            manifest_path = meta.get("image_manifest", "")
        if meta.get("data_mode") == "image_manifest" and not manifest_path:
            raise ValueError(
                "image-manifest checkpoints require --eval-image-manifest for checkpoint eval"
            )
        if manifest_path and meta["cond_mode"] != "text":
            raise ValueError("manifest checkpoint eval requires a text-conditioned checkpoint")
        if manifest_path:
            guidance_values = [float(semantic_guidance_w)]
            if semantic_guidance_weights is not None:
                guidance_values.extend(float(w) for w in semantic_guidance_weights)
            if any(abs(w) > 0.0 for w in guidance_values):
                raise ValueError(
                    "manifest checkpoint eval has captions but no canonical fact labels; "
                    "keep semantic guidance weights at 0"
                )
            split = eval_image_split or "eval"
            image_root = eval_image_root or meta.get("image_root", "")
            records = read_image_manifest(
                manifest_path, root=image_root, split=split,
                min_aesthetic=eval_image_min_aesthetic,
                max_records=eval_image_max_records)
            rows = []
            for eval_seed in eval_seeds:
                for row in image_record_sweep(
                        ae, flow, records, cfg_scales=cfg_scales,
                        sample_steps_list=sample_steps_list, n=n, batch=batch,
                        seed=int(eval_seed), size=eval_size, device=device,
                        conditioner=conditioner, prompt_vocab=prompt_vocab,
                        caption_max_len=meta["caption_max_len"],
                        sample_method=sample_method, sample_methods=sample_methods,
                        cfg_interval=cfg_interval,
                        text_aligner=getattr(flow, "text_aligner", None),
                        image_feature_aligner=getattr(flow, "image_feature_aligner", None),
                        caption_cond_source=meta["caption_cond_source"],
                        sample_time_shift=actual_sample_time_shift,
                        time_shift=train_time_shift):
                    row["eval_seed"] = int(eval_seed)
                    row["checkpoint_weight_mode"] = actual_mode
                    rows.append(row)
            aggregate = aggregate_sweep_rows(rows)
            best = max(aggregate, key=aggregate_selection_key)
            report = {
                "experiment": "image_latent_manifest_sampler_sweep",
                **meta,
                "checkpoint_weight_mode": actual_mode,
                "requested_checkpoint_weight_mode": weight_mode,
                "selected_checkpoint_weights": actual_mode,
                "n": int(n),
                "image_size": int(eval_size),
                "sample_method": sample_method,
                "sample_methods": list(sample_methods or (sample_method,)),
                "sample_time_shift": float(actual_sample_time_shift),
                "time_shift": float(train_time_shift),
                "cfg_interval": list(validate_guidance_interval(cfg_interval)),
                "semantic_guidance_w": 0.0,
                "semantic_guidance_weights": [0.0],
                "semantic_guidance_mode": "none",
                "semantic_guidance_interval": list(DEFAULT_GUIDANCE_INTERVAL),
                "eval_seeds": [int(s) for s in eval_seeds],
                "eval_image_manifest": manifest_path,
                "eval_image_root": image_root,
                "eval_image_split": split,
                "eval_image_min_aesthetic": (
                    float(eval_image_min_aesthetic)
                    if eval_image_min_aesthetic is not None else None
                ),
                "eval_image_max_records": int(eval_image_max_records),
                "rows": rows,
                "aggregate": aggregate,
                "best": best,
            }
            report.update(summarize_records(records))
            return report

        rows = []
        for eval_seed in eval_seeds:
            for row in sampler_sweep(ae, flow, cfg_scales=cfg_scales,
                                     sample_steps_list=sample_steps_list, n=n, batch=batch,
                                     seed=int(eval_seed), size=eval_size, device=device,
                                     roundtrip_samples=roundtrip_samples,
                                     cond_mode=meta["cond_mode"], conditioner=conditioner,
                                     prompt_vocab=prompt_vocab,
                                     prompt_templates=prompt_templates,
                                     semantic_guidance_w=semantic_guidance_w,
                                     semantic_guidance_weights=semantic_guidance_weights,
                                     semantic_guidance_mode=semantic_guidance_mode,
                                     sample_method=sample_method,
                                     sample_methods=sample_methods,
                                     cfg_interval=cfg_interval,
                                     sample_time_shift=actual_sample_time_shift,
                                     time_shift=train_time_shift,
                                     semantic_guidance_interval=semantic_guidance_interval):
                row["eval_seed"] = int(eval_seed)
                row["checkpoint_weight_mode"] = actual_mode
                rows.append(row)
        aggregate = aggregate_sweep_rows(rows)
        best = max(aggregate, key=aggregate_selection_key)
        report = {
            "experiment": "image_latent_sampler_sweep",
            **meta,
            "checkpoint_weight_mode": actual_mode,
            "requested_checkpoint_weight_mode": weight_mode,
            "selected_checkpoint_weights": actual_mode,
            "n": int(n),
            "image_size": int(eval_size),
            "roundtrip_samples": int(roundtrip_samples),
            "sample_method": sample_method,
            "sample_methods": list(sample_methods or (sample_method,)),
            "sample_time_shift": float(actual_sample_time_shift),
            "time_shift": float(train_time_shift),
            "cfg_interval": list(validate_guidance_interval(cfg_interval)),
            "semantic_guidance_w": float(semantic_guidance_w),
            "semantic_guidance_weights": [
                float(w) for w in (semantic_guidance_weights or (semantic_guidance_w,))
            ],
            "semantic_guidance_mode": semantic_guidance_mode,
            "semantic_guidance_interval": list(validate_guidance_interval(
                semantic_guidance_interval)),
            "eval_seeds": [int(s) for s in eval_seeds],
            "rows": rows,
            "aggregate": aggregate,
            "best": best,
        }
        if intervention_samples:
            report.update(latent_intervention_diagnostic(
                ae, n=intervention_samples, batch=batch, seed=seed + 31, size=eval_size,
                device=device))
        return report

    if weight_mode != "auto":
        return run_mode(weight_mode)

    candidates = {"raw": run_mode("raw")}
    if candidates["raw"]["ema_available"]:
        candidates["ema"] = run_mode("ema")
    selected = max(candidates, key=lambda mode: aggregate_selection_key(candidates[mode]["best"]))
    report = candidates[selected]
    report["requested_checkpoint_weight_mode"] = "auto"
    report["selected_checkpoint_weights"] = selected
    report["weight_eval_candidates"] = {
        mode: {
            "checkpoint_weight_mode": candidate["checkpoint_weight_mode"],
            "ema_loaded": candidate["ema_loaded"],
            "best": candidate["best"],
        }
        for mode, candidate in candidates.items()
    }
    return report


def selected_grid_settings(report, fallback_cfg=1.0, fallback_steps=4,
                           fallback_method="euler", fallback_semantic_w=0.0,
                           fallback_semantic_mode="decoded",
                           fallback_cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                           fallback_semantic_interval=DEFAULT_GUIDANCE_INTERVAL,
                           fallback_sample_time_shift=1.0):
    best = report.get("best") or report
    return {
        "cfg_scale": float(best.get("cfg_scale", fallback_cfg)),
        "cfg_interval": tuple(best.get("cfg_interval", fallback_cfg_interval)),
        "sample_steps": int(best.get("sample_steps", fallback_steps)),
        "sample_time_shift": float(best.get("sample_time_shift",
                                            fallback_sample_time_shift)),
        "sample_method": str(best.get("sample_method", fallback_method)),
        "semantic_guidance_w": float(best.get("semantic_guidance_w", fallback_semantic_w)),
        "semantic_guidance_mode": str(best.get("semantic_guidance_mode",
                                               fallback_semantic_mode)),
        "semantic_guidance_interval": tuple(best.get(
            "semantic_guidance_interval", fallback_semantic_interval)),
    }


def train_latent_flow(ae_steps=200, flow_steps=200, batch=64, latent_ch=16, hidden=64,
                      lr=2e-4, fact_w=1.0, seed=0, size=32, device=DEV, flow_arch="conv",
                      dit_depth=3, dit_heads=4, cond_drop=0.0, cfg_scale=1.0,
                      dit_head_width_mult=1, latent_max_tokens=256,
                      ae_arch="semantic", latent_downsample=4, ae_res_blocks=1,
                      ae_recon_loss="mse", ae_grad_w=0.0, ae_ms_w=0.0,
                      ae_latent_reg_w=0.0,
                      image_text_align_w=0.0, flow_text_align_w=0.0, text_embed_dim=128,
                      image_feature_align_w=0.0, flow_feature_align_w=0.0,
                      image_feature_embed_dim=128,
                      sample_steps=4, roundtrip_samples=1, flow_semantic_w=0.0,
                      flow_consistency_w=0.0,
                      cond_mode="facts", text_cond_dim=0,
                      image_manifest="", image_root="", image_split="train",
                      image_min_aesthetic=None, image_max_records=0,
                      caption_vocab_max=8192, caption_max_len=64,
                      caption_cond_source="tokens",
                      image_crop_mode="center", image_hflip_prob=0.0,
                      dit_qk_norm=False, dit_attn_impl="manual",
                      prompt_templates=DEFAULT_PROMPT_TEMPLATES, time_sampling="uniform",
                      time_logit_mean=0.0, time_logit_std=1.0, time_shift=1.0,
                      latent_normalize="none", latent_stat_samples=512,
                      ae_intervention_w=0.0, ae_factor_orth_w=0.0,
                      semantic_guidance_w=0.0, semantic_guidance_mode="decoded",
                      cfg_interval=DEFAULT_GUIDANCE_INTERVAL,
                      semantic_guidance_interval=DEFAULT_GUIDANCE_INTERVAL,
                      sample_method="euler",
                      flow_ema_decay=0.0, flow_ema_warmup=True, eval_with_ema=True,
                      eval_weight_mode="auto", intervention_samples=32,
                      train_precision="fp32", ae_accum_steps=1, flow_accum_steps=1,
                      grad_clip=0.0,
                      flow_cache_latents=False, flow_cache_records=0, flow_cache_batch=64,
                      flow_cache_dir="", flow_cache_shard_size=1024,
                      return_conditioner=False, return_ema=False, return_aligner=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if cond_mode not in ("facts", "text"):
        raise ValueError(f"unknown condition mode {cond_mode!r}")
    image_records = None
    if image_manifest:
        if cond_mode != "text":
            raise ValueError("image manifests require cond_mode='text'")
        if flow_semantic_w > 0.0 or semantic_guidance_w > 0.0:
            raise ValueError("image manifest training has captions but no canonical fact labels")
        if ae_intervention_w > 0.0 or ae_factor_orth_w > 0.0 or intervention_samples:
            raise ValueError("fact intervention/orthogonality diagnostics require fact labels")
        image_records = read_image_manifest(
            image_manifest, root=image_root, split=image_split,
            min_aesthetic=image_min_aesthetic, max_records=image_max_records)
        caption_cond_source, text_embedding_in_dim = resolve_caption_cond_source(
            caption_cond_source, image_records)
        image_embedding_in_dim = (
            infer_image_embedding_dim(image_records)
            if image_feature_align_w > 0.0 or flow_feature_align_w > 0.0 else 0
        )
        if (image_feature_align_w > 0.0 or flow_feature_align_w > 0.0
                ) and image_embedding_in_dim <= 0:
            raise ValueError("image feature alignment requires image_embedding rows")
    elif (image_text_align_w > 0.0 or flow_text_align_w > 0.0
          or image_feature_align_w > 0.0 or flow_feature_align_w > 0.0):
        raise ValueError("image/text or image-feature alignment losses require image_manifest training")
    else:
        caption_cond_source, text_embedding_in_dim = "tokens", 0
        image_embedding_in_dim = 0
    if time_sampling not in ("uniform", "logit-normal"):
        raise ValueError(f"unknown time sampling mode {time_sampling!r}")
    if time_shift <= 0.0:
        raise ValueError("time_shift must be positive")
    if latent_normalize not in ("none", "global", "channel"):
        raise ValueError(f"unknown latent normalization mode {latent_normalize!r}")
    if image_crop_mode not in ("center", "random", "none"):
        raise ValueError(f"unknown image crop mode {image_crop_mode!r}")
    if image_hflip_prob < 0.0 or image_hflip_prob > 1.0:
        raise ValueError("image_hflip_prob must be in [0, 1]")
    if ae_recon_loss not in AE_RECON_LOSSES:
        raise ValueError(f"unknown AE reconstruction loss {ae_recon_loss!r}")
    if ae_grad_w < 0.0 or ae_ms_w < 0.0 or ae_latent_reg_w < 0.0:
        raise ValueError("AE reconstruction weights must be non-negative")
    if image_text_align_w < 0.0 or flow_text_align_w < 0.0:
        raise ValueError("image/text alignment weights must be non-negative")
    if image_feature_align_w < 0.0 or flow_feature_align_w < 0.0:
        raise ValueError("image feature alignment weights must be non-negative")
    if text_embed_dim <= 0:
        raise ValueError("text_embed_dim must be positive")
    if image_feature_embed_dim <= 0:
        raise ValueError("image_feature_embed_dim must be positive")
    amp_cfg = amp_config(device, train_precision)
    ae_accum_steps = max(1, int(ae_accum_steps))
    flow_accum_steps = max(1, int(flow_accum_steps))
    if grad_clip < 0.0:
        raise ValueError("grad_clip must be non-negative")
    if ae_intervention_w < 0.0:
        raise ValueError("ae_intervention_w must be non-negative")
    if ae_factor_orth_w < 0.0:
        raise ValueError("ae_factor_orth_w must be non-negative")
    if semantic_guidance_w < 0.0:
        raise ValueError("semantic_guidance_w must be non-negative")
    if flow_consistency_w < 0.0:
        raise ValueError("flow_consistency_w must be non-negative")
    flow_cache_dir = str(flow_cache_dir or "")
    flow_cache_latents = bool(flow_cache_latents or flow_cache_dir)
    if flow_cache_latents and image_records is None:
        raise ValueError("flow latent cache currently requires image_manifest training")
    if flow_cache_records < 0 or flow_cache_batch <= 0:
        raise ValueError("flow cache record/batch settings must be non-negative/positive")
    if flow_cache_shard_size <= 0:
        raise ValueError("flow cache shard size must be positive")
    if semantic_guidance_mode not in ("latent", "decoded"):
        raise ValueError(f"unknown semantic guidance mode {semantic_guidance_mode!r}")
    if sample_method not in SAMPLE_METHODS:
        raise ValueError(f"unknown sample method {sample_method!r}")
    cfg_interval = validate_guidance_interval(cfg_interval, name="cfg_interval")
    semantic_guidance_interval = validate_guidance_interval(
        semantic_guidance_interval, name="semantic_guidance_interval")
    if dit_head_width_mult <= 0:
        raise ValueError("dit_head_width_mult must be positive")
    if dit_attn_impl not in MMDIT_ATTN_IMPLS:
        raise ValueError(f"unknown MM-DiT attention implementation {dit_attn_impl!r}")
    if latent_max_tokens <= 0:
        raise ValueError("latent_max_tokens must be positive")
    if flow_ema_decay < 0.0 or flow_ema_decay >= 1.0:
        raise ValueError("flow_ema_decay must be in [0, 1)")
    if eval_weight_mode not in EVAL_WEIGHT_MODES:
        raise ValueError(f"unknown eval weight mode {eval_weight_mode!r}")
    prompt_vocab = None
    if cond_mode == "text":
        prompt_vocab = (
            None if image_records is not None and caption_cond_source == "embedding"
            else build_caption_vocab(image_records, max_vocab=caption_vocab_max)
            if image_records is not None
            else build_prompt_vocab(prompt_templates)
        )
    cond_dim = len(FACT_VOCAB)
    conditioner = None
    if cond_mode == "text":
        cond_dim = int(text_cond_dim or hidden)
        if image_records is not None and caption_cond_source == "embedding":
            conditioner = PrecomputedTextConditioner(
                text_embedding_in_dim, cond_dim=cond_dim, hidden=hidden).to(device)
        else:
            conditioner = PromptConditioner(len(prompt_vocab), cond_dim=cond_dim,
                                            hidden=hidden,
                                            max_len=caption_max_len if image_records is not None
                                            else 32).to(device)
    ae = make_autoencoder(ae_arch=ae_arch, latent_ch=latent_ch, hidden=hidden,
                          latent_downsample=latent_downsample,
                          ae_res_blocks=ae_res_blocks).to(device)
    text_aligner = None
    if image_records is not None and (image_text_align_w > 0.0 or flow_text_align_w > 0.0):
        text_aligner = ImageTextAligner(
            latent_ch=latent_ch, cond_dim=cond_dim, hidden=hidden,
            embed_dim=text_embed_dim).to(device)
    image_feature_aligner = None
    if image_records is not None and (
            image_feature_align_w > 0.0 or flow_feature_align_w > 0.0):
        image_feature_aligner = ImageFeatureAligner(
            latent_ch=latent_ch, feature_dim=image_embedding_in_dim, hidden=hidden,
            embed_dim=image_feature_embed_dim).to(device)
    latent_shape = ae_latent_shape(ae, size)
    latent_tokens = latent_shape[1] * latent_shape[2]
    if flow_arch in ("dit", "crossdit", "mmdit") and latent_tokens > int(latent_max_tokens):
        raise ValueError(
            f"latent token count {latent_tokens} exceeds latent_max_tokens={latent_max_tokens}; "
            "increase --latent-max-tokens or use a larger --latent-downsample"
        )
    flow = make_flow(flow_arch=flow_arch, latent_ch=latent_ch, hidden=hidden,
                     dit_depth=dit_depth, dit_heads=dit_heads, cond_dim=cond_dim,
                     dit_head_width_mult=dit_head_width_mult,
                     latent_max_tokens=latent_max_tokens,
                     dit_qk_norm=bool(dit_qk_norm),
                     dit_attn_impl=dit_attn_impl).to(device)
    attach_text_aligner(flow, text_aligner)
    attach_image_feature_aligner(flow, image_feature_aligner)
    ae_params = list(ae.parameters())
    if text_aligner is not None and image_text_align_w > 0.0:
        ae_params += list(conditioner.parameters()) + list(text_aligner.parameters())
    if image_feature_aligner is not None and image_feature_align_w > 0.0:
        ae_params += list(image_feature_aligner.parameters())
    opt_ae = torch.optim.AdamW(ae_params, lr=lr, weight_decay=0.01)
    scaler = amp_grad_scaler(device, train_precision)
    ae.train()
    if text_aligner is not None:
        text_aligner.train()
    if image_feature_aligner is not None:
        image_feature_aligner.train()
    last_ae = {}
    for _ in range(ae_steps):
        opt_ae.zero_grad(set_to_none=True)
        for _micro in range(ae_accum_steps):
            if image_records is None:
                x, fact_cond, yc, ys = _batch(batch, rng, size=size, device=device)
            else:
                x, captions, chosen_records = sample_image_text_batch(
                    image_records, rng, batch=batch, size=size, device=device,
                    return_records=True, crop_mode=image_crop_mode,
                    hflip_prob=image_hflip_prob)
            with amp_autocast(device, train_precision):
                out = ae(x)
                if image_records is None:
                    loss, parts = autoencoder_loss(
                        out, x, yc, ys, fact_w=fact_w, recon_loss=ae_recon_loss,
                        grad_w=ae_grad_w, ms_w=ae_ms_w, latent_reg_w=ae_latent_reg_w)
                else:
                    loss, parts = reconstruction_loss_parts(
                        out["recon"], x, mode=ae_recon_loss, grad_w=ae_grad_w,
                        ms_w=ae_ms_w, latent=out.get("latent"),
                        latent_reg_w=ae_latent_reg_w)
                    if text_aligner is not None and image_text_align_w > 0.0:
                        cond_vec = caption_record_condition(
                            captions, chosen_records, conditioner, prompt_vocab,
                            source=caption_cond_source, max_len=caption_max_len,
                            device=device, return_tokens=False)
                        align_loss, align_parts = image_text_alignment_loss(
                            text_aligner, out["latent"], cond_vec, prefix="caption_align")
                        loss = loss + float(image_text_align_w) * align_loss
                        parts.update(align_parts)
                    if image_feature_aligner is not None and image_feature_align_w > 0.0:
                        image_features = record_image_embedding_tensor(
                            chosen_records, device=device)
                        feature_loss, feature_parts = image_feature_alignment_loss(
                            image_feature_aligner, out["latent"], image_features,
                            prefix="image_feature_align")
                        loss = loss + float(image_feature_align_w) * feature_loss
                        parts.update(feature_parts)
                if image_records is None and ae_intervention_w > 0.0:
                    intervention, intervention_parts = latent_intervention_training_loss(
                        ae, out["latent"], fact_cond)
                    loss = loss + float(ae_intervention_w) * intervention
                    parts.update({f"latent_{k}": v for k, v in intervention_parts.items()})
                if image_records is None and ae_factor_orth_w > 0.0:
                    factor_orth, factor_orth_parts = latent_factor_orthogonality_loss(
                        out["latent"], fact_cond)
                    loss = loss + float(ae_factor_orth_w) * factor_orth
                    parts.update({f"latent_{k}": v for k, v in factor_orth_parts.items()})
                scaled_loss = loss / float(ae_accum_steps)
            scaler.scale(scaled_loss).backward()
            last_ae = {k: float(v.detach().cpu()) for k, v in parts.items()}
            last_ae["total_loss"] = float(loss.detach().cpu())
        if grad_clip > 0.0:
            scaler.unscale_(opt_ae)
            grad_norm = torch.nn.utils.clip_grad_norm_(ae_params, float(grad_clip))
            last_ae["grad_norm"] = float(grad_norm.detach().cpu())
        scaler.step(opt_ae)
        scaler.update()

    flow_params = list(flow.parameters()) + ([] if conditioner is None
                                             else list(conditioner.parameters()))
    if text_aligner is not None and flow_text_align_w > 0.0:
        flow_params += list(text_aligner.parameters())
    if image_feature_aligner is not None and flow_feature_align_w > 0.0:
        flow_params += list(image_feature_aligner.parameters())
    opt_flow = torch.optim.AdamW(flow_params, lr=lr, weight_decay=0.01)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    flow_cache = None
    if flow_cache_latents:
        flow_cache = build_image_latent_cache(
            ae, image_records, prompt_vocab, caption_max_len=caption_max_len,
            max_records=flow_cache_records, batch=flow_cache_batch, seed=seed + 211,
            size=size, device=device, precision=train_precision,
            cond_source=caption_cond_source, cache_dir=flow_cache_dir,
            shard_size=flow_cache_shard_size,
            include_image_embeddings=flow_feature_align_w > 0.0,
            crop_mode=image_crop_mode, hflip_prob=image_hflip_prob)
    if image_records is None:
        latent_stats = estimate_latent_stats(
            ae, n=latent_stat_samples, batch=batch, seed=seed + 97, size=size, device=device,
            mode=latent_normalize)
    elif flow_cache is not None:
        latent_stats = estimate_latent_stats_cache(
            flow_cache, n=latent_stat_samples, seed=seed + 97, mode=latent_normalize)
    else:
        latent_stats = estimate_latent_stats_records(
            ae, image_records, n=latent_stat_samples, batch=batch, seed=seed + 97,
            size=size, device=device, mode=latent_normalize,
            crop_mode=image_crop_mode, hflip_prob=image_hflip_prob)
    attach_latent_stats(flow, latent_stats)
    flow.train()
    if conditioner is not None:
        conditioner.train()
    if text_aligner is not None:
        text_aligner.train()
    if image_feature_aligner is not None:
        image_feature_aligner.train()
    last_flow = {}
    flow_ema = clone_state_dict(flow) if flow_ema_decay > 0.0 else None
    conditioner_ema = (clone_state_dict(conditioner)
                       if flow_ema_decay > 0.0 and conditioner is not None else None)
    ema_updates = 0
    last_ema_decay = 0.0
    for _ in range(flow_steps):
        opt_flow.zero_grad(set_to_none=True)
        for _micro in range(flow_accum_steps):
            if flow_cache is not None:
                z1, cache_payload = sample_latent_cache(flow_cache, rng, batch, device=device)
                fact_cond, specs = None, None
                cond = cached_caption_payload_condition(
                    cache_payload, conditioner, source=caption_cond_source,
                    device=device, return_tokens=flow_uses_cond_tokens(flow))
                image_features = cache_payload.get("image_embeddings")
            elif image_records is None:
                image_features = None
                x, fact_cond, _yc, _ys, specs = _batch(batch, rng, size=size, device=device,
                                                       return_specs=True)
                with torch.no_grad(), amp_autocast(device, train_precision):
                    z1 = ae.encode(x)
                cond = model_condition(
                    specs, fact_cond, cond_mode=cond_mode, conditioner=conditioner,
                    prompt_vocab=prompt_vocab, prompt_templates=prompt_templates,
                    rng=rng, device=device, return_tokens=flow_uses_cond_tokens(flow))
            else:
                x, captions, chosen_records = sample_image_text_batch(
                    image_records, rng, batch=batch, size=size, device=device,
                    return_records=True, crop_mode=image_crop_mode,
                    hflip_prob=image_hflip_prob)
                fact_cond, specs = None, None
                with torch.no_grad(), amp_autocast(device, train_precision):
                    z1 = ae.encode(x)
                cond = caption_record_condition(
                    captions, chosen_records, conditioner, prompt_vocab,
                    source=caption_cond_source, max_len=caption_max_len, device=device,
                    return_tokens=flow_uses_cond_tokens(flow))
                image_features = (
                    record_image_embedding_tensor(chosen_records, device=device)
                    if flow_feature_align_w > 0.0 else None
                )
            with amp_autocast(device, train_precision):
                loss, parts = latent_flow_losses(
                    flow, z1, cond, cond_drop=cond_drop, ae=ae,
                    semantic_w=flow_semantic_w, semantic_cond=fact_cond,
                    time_sampling=time_sampling, time_logit_mean=time_logit_mean,
                    time_logit_std=time_logit_std, time_shift=time_shift,
                    consistency_w=flow_consistency_w,
                    text_aligner=text_aligner, text_align_w=flow_text_align_w,
                    feature_aligner=image_feature_aligner, image_features=image_features,
                    feature_align_w=flow_feature_align_w)
                scaled_loss = loss / float(flow_accum_steps)
            scaler.scale(scaled_loss).backward()
            last_flow = {"total_loss": float(loss.detach().cpu())}
            last_flow.update({k: float(v.detach().cpu()) for k, v in parts.items()})
        if grad_clip > 0.0:
            scaler.unscale_(opt_flow)
            grad_norm = torch.nn.utils.clip_grad_norm_(flow_params, float(grad_clip))
            last_flow["grad_norm"] = float(grad_norm.detach().cpu())
        scaler.step(opt_flow)
        scaler.update()
        if flow_ema is not None:
            ema_updates += 1
            last_ema_decay = ema_effective_decay(flow_ema_decay, ema_updates,
                                                 warmup=flow_ema_warmup)
            update_ema_state(flow_ema, flow, last_ema_decay)
            if conditioner is not None:
                update_ema_state(conditioner_ema, conditioner, last_ema_decay)

    if conditioner is not None:
        conditioner.eval()
    if text_aligner is not None:
        text_aligner.eval()
    if image_feature_aligner is not None:
        image_feature_aligner.eval()
    raw_flow = clone_state_dict(flow)
    raw_conditioner = clone_state_dict(conditioner) if conditioner is not None else None
    requested_eval_weight_mode = "raw" if not eval_with_ema else eval_weight_mode
    if requested_eval_weight_mode == "auto":
        candidate_modes = ["raw"] + (["ema"] if flow_ema is not None else [])
    elif requested_eval_weight_mode == "ema" and flow_ema is None:
        candidate_modes = ["raw"]
    else:
        candidate_modes = [requested_eval_weight_mode]
    candidate_reports = {}
    for mode in candidate_modes:
        if mode == "ema":
            load_flow_state(flow, flow_ema)
            if conditioner is not None:
                conditioner.load_state_dict(conditioner_ema)
        else:
            load_flow_state(flow, raw_flow)
            if conditioner is not None:
                conditioner.load_state_dict(raw_conditioner)
        if image_records is None:
            candidate = evaluate(ae, flow, seed=seed + 1, size=size, device=device,
                                 cfg_scale=cfg_scale, sample_steps=sample_steps,
                                 roundtrip_samples=roundtrip_samples, cond_mode=cond_mode,
                                 conditioner=conditioner, prompt_vocab=prompt_vocab,
                                 prompt_templates=prompt_templates,
                                 intervention_samples=intervention_samples,
                                 semantic_guidance_w=semantic_guidance_w,
                                 semantic_guidance_mode=semantic_guidance_mode,
                                 sample_method=sample_method,
                                 cfg_interval=cfg_interval,
                                 semantic_guidance_interval=semantic_guidance_interval,
                                 sample_time_shift=time_shift,
                                 time_shift=time_shift)
        else:
            candidate = evaluate_image_records(
                ae, flow, image_records, seed=seed + 1, size=size, device=device,
                conditioner=conditioner, prompt_vocab=prompt_vocab,
                caption_max_len=caption_max_len, cfg_scale=cfg_scale,
                sample_steps=sample_steps, sample_method=sample_method,
                cfg_interval=cfg_interval, text_aligner=text_aligner,
                image_feature_aligner=image_feature_aligner,
                caption_cond_source=caption_cond_source,
                sample_time_shift=time_shift, time_shift=time_shift)
        candidate["eval_weight_mode"] = mode
        candidate_reports[mode] = candidate
    selected_eval_weights = max(candidate_reports, key=lambda mode: report_selection_key(
        candidate_reports[mode]))
    report = dict(candidate_reports[selected_eval_weights])
    load_flow_state(flow, raw_flow)
    if conditioner is not None:
        conditioner.load_state_dict(raw_conditioner)
    report.update({
        "experiment": "image3_latent_fact_conditioned_rectified_flow",
        "ae_steps": int(ae_steps),
        "flow_steps": int(flow_steps),
        "batch": int(batch),
        "flow_cache_latents": bool(flow_cache is not None),
        "flow_cache_requested": bool(flow_cache_latents),
        "flow_cache_backend": latent_cache_backend(flow_cache) if flow_cache is not None else "",
        "flow_cache_dir": str(flow_cache.get("cache_dir", "")) if flow_cache is not None else "",
        "flow_cache_records": int(flow_cache["records"]) if flow_cache is not None else 0,
        "flow_cache_max_records": int(flow_cache_records),
        "flow_cache_batch": int(flow_cache_batch),
        "flow_cache_shard_size": int(flow_cache_shard_size),
        "flow_cache_shards": (
            len(flow_cache.get("shards", [])) if flow_cache is not None else 0
        ),
        "flow_cache_bytes": int(flow_cache["bytes"]) if flow_cache is not None else 0,
        "flow_cache_cond_source": (
            str(flow_cache["cond_source"]) if flow_cache is not None else caption_cond_source
        ),
        "ae_accum_steps": int(ae_accum_steps),
        "flow_accum_steps": int(flow_accum_steps),
        "ae_effective_batch": int(batch) * int(ae_accum_steps),
        "flow_effective_batch": int(batch) * int(flow_accum_steps),
        "train_precision": train_precision,
        "train_amp_enabled": bool(amp_cfg["enabled"]),
        "train_amp_dtype": amp_cfg["dtype_name"],
        "grad_clip": float(grad_clip),
        "image_size": int(size),
        "latent_ch": int(latent_ch),
        "ae_arch": ae_arch,
        "latent_downsample": int(getattr(ae, "downsample", latent_downsample)),
        "ae_res_blocks": int(ae_res_blocks) if ae_arch == "residual" else 0,
        "latent_h": int(latent_shape[1]),
        "latent_w": int(latent_shape[2]),
        "latent_tokens": int(latent_tokens),
        "latent_max_tokens": int(latent_max_tokens),
        "hidden": int(hidden),
        "flow_arch": flow_arch,
        "dit_depth": int(dit_depth) if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "dit_heads": int(dit_heads) if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "dit_head_width_mult": (
            int(dit_head_width_mult) if flow_arch in ("dit", "crossdit", "mmdit") else 1
        ),
        "dit_qk_norm": bool(dit_qk_norm) if flow_arch == "mmdit" else False,
        "dit_attn_impl": dit_attn_impl if flow_arch == "mmdit" else "manual",
        "adaptive_modulation": bool(getattr(flow, "uses_adaptive_modulation", False)),
        "residual_gating": bool(getattr(flow, "uses_residual_gating", False)),
        "fact_w": float(fact_w),
        "ae_recon_loss": ae_recon_loss,
        "ae_grad_w": float(ae_grad_w),
        "ae_ms_w": float(ae_ms_w),
        "ae_latent_reg_w": float(ae_latent_reg_w),
        "image_text_align_w": float(image_text_align_w),
        "flow_text_align_w": float(flow_text_align_w),
        "text_embed_dim": int(text_embed_dim),
        "text_aligner": text_aligner is not None,
        "image_feature_align_w": float(image_feature_align_w),
        "flow_feature_align_w": float(flow_feature_align_w),
        "image_feature_embed_dim": int(image_feature_embed_dim),
        "image_feature_aligner": image_feature_aligner is not None,
        "ae_intervention_w": float(ae_intervention_w),
        "ae_factor_orth_w": float(ae_factor_orth_w),
        "cond_drop": float(cond_drop),
        "cfg_interval": list(cfg_interval),
        "sample_method": sample_method,
        "semantic_guidance_w": float(semantic_guidance_w),
        "semantic_guidance_mode": semantic_guidance_mode,
        "semantic_guidance_interval": list(semantic_guidance_interval),
        "flow_semantic_w": float(flow_semantic_w),
        "flow_consistency_w": float(flow_consistency_w),
        "flow_ema_decay": float(flow_ema_decay),
        "flow_ema_warmup": bool(flow_ema_warmup),
        "flow_ema_updates": int(ema_updates),
        "flow_ema_effective_decay": float(last_ema_decay),
        "ema_available": flow_ema is not None,
        "eval_weight_mode": requested_eval_weight_mode,
        "selected_eval_weights": selected_eval_weights,
        "eval_with_ema": selected_eval_weights == "ema",
        "intervention_samples": int(intervention_samples),
        "weight_eval_candidates": {
            mode: eval_report_summary(candidate)
            for mode, candidate in candidate_reports.items()
        },
        "time_sampling": time_sampling,
        "time_logit_mean": float(time_logit_mean),
        "time_logit_std": float(time_logit_std),
        "time_shift": float(time_shift),
        "sample_time_shift": float(time_shift),
        "latent_stat_samples": int(latent_stat_samples),
        **latent_stats_report(latent_stats),
        "cond_mode": cond_mode,
        "data_mode": "image_manifest" if image_records is not None else "synthetic_factors",
        "image_manifest": image_manifest,
        "image_root": image_root if image_records is not None else "",
        "image_split": image_split if image_records is not None else "",
        "image_min_aesthetic": (
            float(image_min_aesthetic) if image_min_aesthetic is not None else None
        ),
        "image_crop_mode": image_crop_mode if image_records is not None else "",
        "image_hflip_prob": float(image_hflip_prob) if image_records is not None else 0.0,
        "caption_vocab_max": int(caption_vocab_max) if image_records is not None else 0,
        "caption_max_len": int(caption_max_len) if image_records is not None else 0,
        "caption_cond_source": caption_cond_source if image_records is not None else "",
        "text_embedding_in_dim": int(text_embedding_in_dim),
        "image_embedding_in_dim": int(image_embedding_in_dim),
        "cond_dim": int(cond_dim),
        "prompt_templates": (
            [] if image_records is not None else list(prompt_templates) if cond_mode == "text"
            else []
        ),
        "prompt_vocab_size": len(prompt_vocab) if prompt_vocab is not None else 0,
        "last_ae": last_ae,
        "last_flow": last_flow,
        "last_flow_loss": last_flow.get("velocity_mse"),
        "fact_vocab": [list(f) for f in FACT_VOCAB],
    })
    if return_ema:
        if return_conditioner:
            if return_aligner:
                return (
                    ae, flow, conditioner, prompt_vocab, text_aligner, report,
                    flow_ema, conditioner_ema
                )
            return ae, flow, conditioner, prompt_vocab, report, flow_ema, conditioner_ema
        return ae, flow, report, flow_ema
    if return_conditioner:
        if return_aligner:
            return ae, flow, conditioner, prompt_vocab, text_aligner, report
        return ae, flow, conditioner, prompt_vocab, report
    return ae, flow, report


def selftest():
    ae, flow, report = train_latent_flow(ae_steps=2, flow_steps=2, batch=4, latent_ch=4,
                                         hidden=16, seed=0, device="cpu", sample_steps=2)
    assert report["experiment"] == "image3_latent_fact_conditioned_rectified_flow"
    assert report["latent_color_acc"] >= 0.0 and report["latent_shape_acc"] >= 0.0
    assert "sample_roundtrip_both_acc" in report and report["sample_roundtrip_n"] == 15
    assert "latent_endpoint_consistency_mse" in report
    assert "latent_intervention_score" in report and report["latent_intervention_n"] > 0
    assert "latent_factor_orth_loss" in report
    shifted = flow_time_schedule(4, device="cpu", shift=4.0)
    assert torch.allclose(shifted[[0, -1]], torch.tensor([0.0, 1.0]))
    assert 0.0 < float(shifted[1]) < 0.25
    spec = ObjectSpec("p0", "blue", "triangle")
    cond = fact_condition(object_facts(spec), device="cpu")[None]
    img = sample_images(ae, flow, cond, latent_shape=(4, 8, 8), steps=2, device="cpu", seed=0,
                        cfg_scale=1.5)
    assert img.shape == (1, 3, 32, 32)
    guided_img = sample_images(ae, flow, cond, latent_shape=(4, 8, 8), steps=1, device="cpu",
                               seed=0, cfg_scale=1.0, semantic_cond=cond,
                               semantic_guidance_w=0.05,
                               semantic_guidance_interval=(0.0, 0.5))
    assert guided_img.shape == (1, 3, 32, 32)
    grid_path = "/tmp/image_latent_selftest_grid.ppm"
    grid_meta = save_sample_grid(ae, flow, grid_path, size=32, device="cpu",
                                 sample_steps=1, samples_per_combo=1, seed=9)
    assert grid_meta["sample_grid_n"] == len(COLORS) * len(SHAPES)
    with open(grid_path, "rb") as f:
        assert f.read(2) == b"P6"
    ae2, flow2, report2 = train_latent_flow(ae_steps=1, flow_steps=1, batch=2, latent_ch=4,
                                            hidden=32, flow_arch="dit", dit_depth=1,
                                            dit_heads=2, seed=1, device="cpu", cond_drop=0.5,
                                            cfg_scale=1.5, sample_steps=1,
                                            flow_semantic_w=0.25, ae_intervention_w=0.1,
                                            ae_factor_orth_w=0.05,
                                            flow_consistency_w=0.1,
                                            latent_normalize="channel",
                                            latent_stat_samples=8,
                                            cfg_interval=(0.0, 0.5),
                                            semantic_guidance_interval=(0.0, 0.5))
    assert report2["flow_arch"] == "dit"
    assert report2["cond_drop"] == 0.5 and report2["cfg_scale"] == 1.5
    assert report2["flow_semantic_w"] == 0.25
    assert report2["flow_consistency_w"] == 0.1
    assert report2["ae_intervention_w"] == 0.1
    assert report2["ae_factor_orth_w"] == 0.05
    assert report2["latent_normalize"] == "channel" and report2["latent_norm_n"] == 8
    assert report2["cfg_interval"] == [0.0, 0.5]
    assert report2["semantic_guidance_interval"] == [0.0, 0.5]
    assert "latent_intervention_loss" in report2["last_ae"]
    assert "latent_factor_orth_loss" in report2["last_ae"]
    assert "semantic_endpoint_ce" in report2["last_flow"]
    assert "endpoint_consistency_mse" in report2["last_flow"]
    img2 = sample_images(ae2, flow2, cond, latent_shape=(4, 8, 8), steps=1, device="cpu",
                         seed=1, cfg_scale=1.5)
    assert img2.shape == (1, 3, 32, 32)
    ae_res, flow_res, report_res = train_latent_flow(
        ae_steps=1, flow_steps=1, batch=2, latent_ch=6, hidden=32, flow_arch="dit",
        dit_depth=1, dit_heads=2, seed=12, device="cpu", sample_steps=1,
        ae_arch="residual", latent_downsample=8, ae_res_blocks=1,
        latent_max_tokens=32, ae_recon_loss="hybrid", ae_grad_w=0.1, ae_ms_w=0.1,
        ae_latent_reg_w=0.01, train_precision="bf16", ae_accum_steps=2,
        flow_accum_steps=2, grad_clip=1.0, intervention_samples=0)
    assert report_res["ae_arch"] == "residual"
    assert report_res["ae_recon_loss"] == "hybrid"
    assert report_res["ae_accum_steps"] == 2 and report_res["flow_accum_steps"] == 2
    assert report_res["ae_effective_batch"] == 4 and report_res["flow_effective_batch"] == 4
    assert report_res["train_precision"] == "bf16" and report_res["train_amp_enabled"] is False
    assert report_res["grad_clip"] == 1.0
    assert report_res["latent_downsample"] == 8
    assert report_res["latent_h"] == 4 and report_res["latent_tokens"] == 16
    assert "recon_grad_l1" in report_res["last_ae"]
    assert "recon_multiscale_l1" in report_res["last_ae"]
    assert "latent_l2" in report_res["last_ae"]
    assert "grad_norm" in report_res["last_ae"] and "grad_norm" in report_res["last_flow"]
    img_res = sample_images(ae_res, flow_res, cond, latent_shape=ae_latent_shape(ae_res, 32),
                            steps=1, device="cpu", seed=12)
    assert img_res.shape == (1, 3, 32, 32)
    res_ckpt = "/tmp/image_latent_residual_selftest.pt"
    torch.save({
        "autoencoder_state_dict": ae_res.state_dict(),
        "flow_state_dict": flow_res.state_dict(),
        "report": report_res,
        "fact_vocab": FACT_VOCAB,
        "latent_ch": 6,
        "ae_arch": "residual",
        "latent_downsample": 8,
        "ae_res_blocks": 1,
        "latent_max_tokens": 32,
        "hidden": 32,
        "flow_arch": "dit",
        "dit_depth": 1,
        "dit_heads": 2,
        "dit_head_width_mult": 1,
        "cond_mode": "facts",
        "cond_dim": len(FACT_VOCAB),
        "latent_stats": latent_stats_state(flow_latent_stats(flow_res)),
    }, res_ckpt)
    ae_res2, _flow_res2, _cond_res2, _vocab_res2, _tpl_res2, meta_res2 = load_checkpoint(
        res_ckpt, device="cpu", prefer_ema=False)
    assert meta_res2["ae_arch"] == "residual"
    assert ae_latent_shape(ae_res2, 32) == (6, 4, 4)
    sweep = sampler_sweep(ae2, flow2, cfg_scales=(1.0, 1.5), sample_steps_list=(1,),
                          semantic_guidance_weights=(0.0, 0.05),
                          sample_methods=("euler", "heun"),
                          n=4, batch=2, seed=3, device="cpu")
    assert len(sweep) == 8 and "sample_roundtrip_both_acc" in sweep[0]
    agg = aggregate_sweep_rows([dict(r, eval_seed=i) for i, r in enumerate(sweep)])
    assert len(agg) == 8 and "sample_roundtrip_both_acc_mean" in agg[0]
    assert "semantic_guidance_w" in agg[0] and "sample_method" in agg[0]
    ae3, flow3, conditioner3, vocab3, report3 = train_latent_flow(
        ae_steps=1, flow_steps=1, batch=2, latent_ch=4, hidden=32, flow_arch="dit",
        dit_depth=1, dit_heads=2, seed=2, device="cpu", cond_mode="text",
        text_cond_dim=8, flow_semantic_w=0.1, sample_steps=1, return_conditioner=True)
    assert report3["cond_mode"] == "text" and report3["prompt_vocab_size"] > 0
    text_sweep = sampler_sweep(ae3, flow3, cfg_scales=(1.0,), sample_steps_list=(1,), n=4,
                               batch=2, seed=4, device="cpu", cond_mode="text",
                               conditioner=conditioner3, prompt_vocab=vocab3)
    assert len(text_sweep) == 1 and text_sweep[0]["cond_mode"] == "text"
    text_grid_path = "/tmp/image_latent_selftest_text_grid.ppm"
    text_grid_meta = save_sample_grid(
        ae3, flow3, text_grid_path, size=32, device="cpu", cond_mode="text",
        conditioner=conditioner3, prompt_vocab=vocab3, sample_steps=1, samples_per_combo=1,
        seed=10)
    assert text_grid_meta["sample_grid_cond_mode"] == "text"
    ae4, flow4, conditioner4, vocab4, report4 = train_latent_flow(
        ae_steps=1, flow_steps=1, batch=2, latent_ch=4, hidden=32, flow_arch="crossdit",
        dit_depth=1, dit_heads=2, seed=5, device="cpu", cond_mode="text",
        text_cond_dim=8, sample_steps=1, time_sampling="logit-normal",
        time_shift=2.0, return_conditioner=True)
    assert report4["flow_arch"] == "crossdit" and flow_uses_cond_tokens(flow4)
    assert report4["time_sampling"] == "logit-normal" and "time_mean" in report4["last_flow"]
    assert report4["time_shift"] == 2.0 and report4["sample_time_shift"] == 2.0
    assert report4["last_flow"]["time_shift"] == 2.0
    cross_sweep = sampler_sweep(ae4, flow4, cfg_scales=(1.0,), sample_steps_list=(1,), n=4,
                                batch=2, seed=6, device="cpu", cond_mode="text",
                                conditioner=conditioner4, prompt_vocab=vocab4,
                                sample_time_shift=2.0, time_shift=2.0)
    assert len(cross_sweep) == 1 and cross_sweep[0]["cond_mode"] == "text"
    assert cross_sweep[0]["sample_time_shift"] == 2.0 and cross_sweep[0]["time_shift"] == 2.0
    ae5, flow5, conditioner5, vocab5, report5 = train_latent_flow(
        ae_steps=1, flow_steps=1, batch=2, latent_ch=4, hidden=32, flow_arch="mmdit",
        dit_depth=1, dit_heads=2, seed=7, device="cpu", cond_mode="text",
        text_cond_dim=8, sample_steps=1, time_sampling="logit-normal",
        flow_ema_decay=0.5, dit_head_width_mult=2, dit_qk_norm=True,
        dit_attn_impl=("sdpa" if hasattr(F, "scaled_dot_product_attention") else "auto"),
        return_conditioner=True)
    assert report5["flow_arch"] == "mmdit" and flow_uses_cond_tokens(flow5)
    assert report5["dit_head_width_mult"] == 2
    assert report5["dit_qk_norm"] is True
    assert report5["dit_attn_impl"] in ("sdpa", "auto")
    assert any("q_norm" in k for k in flow5.state_dict())
    assert report5["adaptive_modulation"] is True
    assert report5["residual_gating"] is True
    assert report5["ema_available"] is True
    assert report5["eval_weight_mode"] == "auto"
    assert report5["selected_eval_weights"] in ("raw", "ema")
    assert report5["eval_with_ema"] == (report5["selected_eval_weights"] == "ema")
    assert set(report5["weight_eval_candidates"]) == {"raw", "ema"}
    assert "latent_intervention_score" in report5["weight_eval_candidates"]["raw"]
    assert report5["flow_ema_warmup"] is True and report5["flow_ema_effective_decay"] < 0.5
    legacy_state = {k: v for k, v in flow5.state_dict().items() if ".gate." not in k}
    compat_flow = make_flow(flow_arch="mmdit", latent_ch=4, hidden=32, dit_depth=1,
                            dit_heads=2, cond_dim=8, dit_head_width_mult=2,
                            dit_qk_norm=True,
                            dit_attn_impl=report5["dit_attn_impl"])
    compat_load = load_flow_state(compat_flow, legacy_state)
    assert compat_load["tolerated_missing"] and all(
        ".gate." in k for k in compat_load["tolerated_missing"])
    mm_sweep = sampler_sweep(ae5, flow5, cfg_scales=(1.0,), sample_steps_list=(1,), n=4,
                             batch=2, seed=8, device="cpu", cond_mode="text",
                             conditioner=conditioner5, prompt_vocab=vocab5)
    assert len(mm_sweep) == 1 and mm_sweep[0]["cond_mode"] == "text"
    with tempfile.TemporaryDirectory() as td:
        img_dir = os.path.join(td, "images")
        os.makedirs(img_dir, exist_ok=True)
        manifest = os.path.join(td, "manifest.jsonl")
        rows = []
        for i, (name, split, color) in enumerate((
                ("red.ppm", "train", (255, 0, 0)),
                ("green.ppm", "train", (0, 255, 0)),
                ("blue.ppm", "eval", (0, 0, 255)),
                ("white.ppm", "eval", (255, 255, 255)))):
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            arr[:, :] = np.asarray(color, dtype=np.uint8)
            with open(os.path.join(img_dir, name), "wb") as f:
                f.write(b"P6\n8 8\n255\n")
                f.write(arr.tobytes())
            rows.append({"image": name, "caption": f"{split} color patch {i}",
                         "split": split,
                         "text_embedding": [float(i), float(i + 1), float(i % 2)],
                         "image_embedding": [float(i), float(i + 2),
                                             float((i + 1) % 2), 1.0]})
        with open(manifest, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        ae6, flow6, conditioner6, vocab6, aligner6, report6 = train_latent_flow(
            ae_steps=1, flow_steps=1, batch=2, latent_ch=4, hidden=32,
            flow_arch="mmdit", dit_depth=1, dit_heads=2, seed=9,
            device="cpu", cond_mode="text", text_cond_dim=8,
            image_manifest=manifest, image_root=img_dir, image_split="train",
            image_max_records=2, caption_max_len=8, sample_steps=1,
            caption_cond_source="embedding",
            image_text_align_w=0.1, flow_text_align_w=0.1, text_embed_dim=12,
            image_feature_align_w=0.1, flow_feature_align_w=0.1,
            image_feature_embed_dim=10,
            flow_cache_latents=True, flow_cache_batch=1, flow_cache_records=2,
            intervention_samples=0, return_conditioner=True, return_aligner=True)
        assert report6["data_mode"] == "image_manifest"
        assert report6["text_aligner"] is True and aligner6 is not None
        assert report6["image_feature_aligner"] is True
        assert "caption_align_i2t_acc" in report6["last_ae"]
        assert "image_feature_align_i2f_acc" in report6["last_ae"]
        assert "flow_caption_align_i2t_acc" in report6["last_flow"]
        assert "flow_image_feature_align_i2f_acc" in report6["last_flow"]
        assert report6["image_embedding_in_dim"] == 4
        assert report6["flow_cache_latents"] is True
        assert report6["flow_cache_backend"] == "memory"
        assert report6["flow_cache_records"] == 2 and report6["flow_cache_bytes"] > 0
        assert report6["flow_cache_shards"] == 0
        disk_cache_dir = os.path.join(td, "latent_cache")
        _ae_disk, _flow_disk, _cond_disk, _vocab_disk, _aligner_disk, report_disk = (
            train_latent_flow(
                ae_steps=1, flow_steps=1, batch=2, latent_ch=4, hidden=32,
                flow_arch="mmdit", dit_depth=1, dit_heads=2, seed=13,
                device="cpu", cond_mode="text", text_cond_dim=8,
                image_manifest=manifest, image_root=img_dir, image_split="train",
                image_max_records=2, caption_max_len=8, sample_steps=1,
                caption_cond_source="embedding",
                image_text_align_w=0.1, flow_text_align_w=0.1, text_embed_dim=12,
                image_feature_align_w=0.1, flow_feature_align_w=0.1,
                image_feature_embed_dim=10,
                flow_cache_latents=True, flow_cache_batch=1, flow_cache_records=2,
                flow_cache_dir=disk_cache_dir, flow_cache_shard_size=1,
                latent_normalize="channel", latent_stat_samples=2,
                intervention_samples=0, return_conditioner=True, return_aligner=True))
        assert report_disk["flow_cache_backend"] == "disk"
        assert report_disk["flow_cache_records"] == 2
        assert report_disk["flow_cache_shards"] == 2
        assert report_disk["image_feature_aligner"] is True
        assert report_disk["flow_cache_bytes"] > 0
        assert os.path.exists(os.path.join(disk_cache_dir, "meta.json"))
        assert report_disk["latent_normalize"] == "channel"
        manifest_rows = read_image_manifest(manifest, root=img_dir, split="eval")
        img_sweep = image_record_sweep(
            ae6, flow6, manifest_rows, cfg_scales=(1.0,), sample_steps_list=(1,),
            n=2, batch=2, seed=10, device="cpu", conditioner=conditioner6,
            prompt_vocab=vocab6, caption_max_len=8, text_aligner=aligner6,
            image_feature_aligner=getattr(flow6, "image_feature_aligner", None),
            caption_cond_source="embedding")
        img_agg = aggregate_sweep_rows([dict(r, eval_seed=10) for r in img_sweep])
        assert "caption_sample_mse_mean" in img_agg[0]
        assert "generated_caption_retrieval_i2t_acc_mean" in img_agg[0]
        assert "generated_image_feature_retrieval_i2f_acc_mean" in img_agg[0]
        ckpt = os.path.join(td, "manifest.pt")
        torch.save({
            "autoencoder_state_dict": ae6.state_dict(),
            "flow_state_dict": flow6.state_dict(),
            "report": report6,
            "fact_vocab": FACT_VOCAB,
            "latent_ch": 4,
            "hidden": 32,
            "flow_arch": "mmdit",
            "dit_depth": 1,
            "dit_heads": 2,
            "dit_head_width_mult": 1,
            "cond_mode": "text",
            "cond_dim": report6["cond_dim"],
            "data_mode": "image_manifest",
            "image_manifest": manifest,
            "image_root": img_dir,
            "image_split": "train",
            "caption_max_len": 8,
            "caption_cond_source": "embedding",
            "text_embedding_in_dim": 3,
            "text_embed_dim": 12,
            "image_embedding_in_dim": 4,
            "image_feature_embed_dim": 10,
            "image_feature_align_w": 0.1,
            "flow_feature_align_w": 0.1,
            "latent_stats": latent_stats_state(flow_latent_stats(flow6)),
            "prompt_templates": [],
            "prompt_vocab": vocab6,
            "conditioner_state_dict": conditioner6.state_dict(),
            "text_aligner_state_dict": aligner6.state_dict(),
            "image_feature_aligner_state_dict": (
                getattr(flow6, "image_feature_aligner").state_dict()
            ),
            "flow_ema_state_dict": {},
            "conditioner_ema_state_dict": {},
        }, ckpt)
        eval6 = evaluate_checkpoint(
            ckpt, cfg_scales=(1.0,), sample_steps_list=(1,), n=2, batch=2,
            seed=11, eval_seeds=(11,), size=32, device="cpu",
            eval_image_manifest=manifest, eval_image_root=img_dir,
            eval_image_split="eval")
        assert eval6["experiment"] == "image_latent_manifest_sampler_sweep"
        assert eval6["best"]["caption_sample_mse_mean"] >= 0.0
        assert eval6["text_aligner"] is True
        assert eval6["image_feature_aligner"] is True
        assert "generated_caption_retrieval_i2t_acc_mean" in eval6["best"]
        assert "generated_image_feature_retrieval_i2f_acc_mean" in eval6["best"]
    print("image_latent selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval-checkpoint", default="", dest="eval_checkpoint",
                    help="load a saved image_latent checkpoint and run a sampler sweep")
    ap.add_argument("--eval-image-manifest", default="", dest="eval_image_manifest",
                    help="JSONL/CSV/TSV captioned image manifest for real-image checkpoint eval")
    ap.add_argument("--eval-image-root", default="", dest="eval_image_root",
                    help="base directory for relative paths in --eval-image-manifest")
    ap.add_argument("--eval-image-split", default="eval", dest="eval_image_split",
                    help="manifest split used for real-image checkpoint eval")
    ap.add_argument("--eval-image-min-aesthetic", type=float, default=None,
                    dest="eval_image_min_aesthetic",
                    help="optional minimum aesthetic/quality score for eval manifest rows")
    ap.add_argument("--eval-image-max-records", type=int, default=0,
                    dest="eval_image_max_records",
                    help="cap eval manifest records for smoke tests; 0 means all")
    ap.add_argument("--ae-steps", type=int, default=200, dest="ae_steps")
    ap.add_argument("--flow-steps", type=int, default=200, dest="flow_steps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ae-accum-steps", type=int, default=1, dest="ae_accum_steps",
                    help="AE gradient accumulation microsteps; effective batch=batch*steps")
    ap.add_argument("--flow-accum-steps", type=int, default=1, dest="flow_accum_steps",
                    help="flow gradient accumulation microsteps; effective batch=batch*steps")
    ap.add_argument("--flow-cache-latents", action="store_true", dest="flow_cache_latents",
                    help="precompute AE latents for manifest flow training")
    ap.add_argument("--flow-cache-records", type=int, default=0, dest="flow_cache_records",
                    help="maximum manifest records to cache for flow training; 0 means all")
    ap.add_argument("--flow-cache-batch", type=int, default=64, dest="flow_cache_batch",
                    help="batch size used while building the AE latent cache")
    ap.add_argument("--flow-cache-dir", default="", dest="flow_cache_dir",
                    help="directory for disk-backed manifest latent cache; implies cache")
    ap.add_argument("--flow-cache-shard-size", type=int, default=1024,
                    dest="flow_cache_shard_size",
                    help="records per shard for --flow-cache-dir")
    ap.add_argument("--train-precision", default="fp32", choices=TRAIN_PRECISIONS,
                    dest="train_precision",
                    help="training precision; bf16/fp16 AMP is enabled on CUDA")
    ap.add_argument("--grad-clip", type=float, default=0.0, dest="grad_clip",
                    help="clip AE/flow gradient norm after accumulation; 0 disables")
    ap.add_argument("--size", type=int, default=0,
                    help="square image size; 0 means 32 for train or checkpoint size for eval")
    ap.add_argument("--latent-ch", type=int, default=16, dest="latent_ch")
    ap.add_argument("--ae-arch", default="semantic", choices=("semantic", "residual"),
                    dest="ae_arch",
                    help="autoencoder architecture for image latents")
    ap.add_argument("--latent-downsample", type=int, default=4, dest="latent_downsample",
                    help="AE spatial compression factor; residual AE supports powers of two")
    ap.add_argument("--ae-res-blocks", type=int, default=1, dest="ae_res_blocks",
                    help="residual blocks per AE stage when --ae-arch residual")
    ap.add_argument("--latent-max-tokens", type=int, default=256, dest="latent_max_tokens",
                    help="maximum latent grid tokens for DiT/CrossDiT/MM-DiT flows")
    ap.add_argument("--ae-recon-loss", default="mse", choices=AE_RECON_LOSSES,
                    dest="ae_recon_loss",
                    help="base reconstruction loss for the image autoencoder")
    ap.add_argument("--ae-grad-w", type=float, default=0.0, dest="ae_grad_w",
                    help="edge/gradient reconstruction loss weight")
    ap.add_argument("--ae-ms-w", type=float, default=0.0, dest="ae_ms_w",
                    help="multi-scale reconstruction loss weight")
    ap.add_argument("--ae-latent-reg-w", type=float, default=0.0, dest="ae_latent_reg_w",
                    help="latent L2 regularization weight during AE training")
    ap.add_argument("--image-text-align-w", type=float, default=0.0,
                    dest="image_text_align_w",
                    help="contrastive image-latent/caption alignment weight during AE training")
    ap.add_argument("--flow-text-align-w", type=float, default=0.0,
                    dest="flow_text_align_w",
                    help="contrastive caption alignment weight on predicted flow endpoints")
    ap.add_argument("--text-embed-dim", type=int, default=128, dest="text_embed_dim",
                    help="shared image/text embedding width for manifest alignment")
    ap.add_argument("--image-feature-align-w", type=float, default=0.0,
                    dest="image_feature_align_w",
                    help="contrastive AE-latent/external-image-feature alignment weight")
    ap.add_argument("--flow-feature-align-w", type=float, default=0.0,
                    dest="flow_feature_align_w",
                    help="contrastive external image-feature weight on predicted flow endpoints")
    ap.add_argument("--image-feature-embed-dim", type=int, default=128,
                    dest="image_feature_embed_dim",
                    help="shared embedding width for latent/image-feature alignment")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fact-w", type=float, default=1.0, dest="fact_w")
    ap.add_argument("--ae-intervention-w", type=float, default=0.0,
                    dest="ae_intervention_w",
                    help="semantic AE latent fact-intervention loss weight")
    ap.add_argument("--ae-factor-orth-w", type=float, default=0.0,
                    dest="ae_factor_orth_w",
                    help="semantic AE cross-factor latent orthogonality loss weight")
    ap.add_argument("--flow-arch", default="conv", choices=("conv", "dit", "crossdit", "mmdit"),
                    dest="flow_arch")
    ap.add_argument("--dit-depth", type=int, default=3, dest="dit_depth")
    ap.add_argument("--dit-heads", type=int, default=4, dest="dit_heads")
    ap.add_argument("--dit-head-width-mult", type=int, default=1,
                    dest="dit_head_width_mult",
                    help="width multiplier for the DiT/MM-DiT latent velocity head")
    ap.add_argument("--dit-qk-norm", action="store_true", dest="dit_qk_norm",
                    help="enable per-head QK RMSNorm in MM-DiT attention")
    ap.add_argument("--dit-attn-impl", default="manual", choices=MMDIT_ATTN_IMPLS,
                    dest="dit_attn_impl",
                    help="MM-DiT attention implementation: manual, sdpa, or auto")
    ap.add_argument("--cond-drop", type=float, default=0.0, dest="cond_drop")
    ap.add_argument("--cfg-scale", type=float, default=1.0, dest="cfg_scale")
    ap.add_argument("--cfg-interval", default="0.0,1.0", dest="cfg_interval",
                    help="CFG active interval over rectified-flow time, formatted start,end")
    ap.add_argument("--sample-steps", type=int, default=4, dest="sample_steps")
    ap.add_argument("--sample-method", default="euler", choices=SAMPLE_METHODS,
                    dest="sample_method",
                    help="ODE sampler method for latent image generation")
    ap.add_argument("--sample-methods", default="",
                    dest="sample_methods",
                    help="comma-separated sampler methods for --eval-checkpoint sweeps")
    ap.add_argument("--semantic-guidance-w", type=float, default=0.0,
                    dest="semantic_guidance_w",
                    help="sampling-time semantic AE guidance weight")
    ap.add_argument("--semantic-guidance-weights", default="",
                    dest="semantic_guidance_weights",
                    help="comma-separated semantic guidance weights for --eval-checkpoint sweeps")
    ap.add_argument("--semantic-guidance-mode", default="decoded",
                    choices=("latent", "decoded"), dest="semantic_guidance_mode",
                    help="guide sampled latents using direct latent heads or decode/re-read heads")
    ap.add_argument("--semantic-guidance-interval", default="0.0,1.0",
                    dest="semantic_guidance_interval",
                    help="semantic guidance active interval over flow time, formatted start,end")
    ap.add_argument("--cfg-scales", default="1.0,1.25,1.5,2.0", dest="cfg_scales",
                    help="comma-separated CFG scales for --eval-checkpoint")
    ap.add_argument("--sample-steps-list", default="4,8,16", dest="sample_steps_list",
                    help="comma-separated sampler step counts for --eval-checkpoint")
    ap.add_argument("--eval-seeds", default="", dest="eval_seeds",
                    help="comma-separated eval seeds for --eval-checkpoint; default uses --seed")
    ap.add_argument("--roundtrip-samples", type=int, default=1, dest="roundtrip_samples")
    ap.add_argument("--intervention-samples", type=int, default=32,
                    dest="intervention_samples",
                    help="samples for latent fact intervention diagnostics; 0 disables")
    ap.add_argument("--flow-semantic-w", type=float, default=0.0, dest="flow_semantic_w",
                    help="semantic endpoint alignment weight for latent flow training")
    ap.add_argument("--flow-consistency-w", type=float, default=0.0,
                    dest="flow_consistency_w",
                    help="same-path clean-endpoint consistency loss weight for latent flow")
    ap.add_argument("--flow-ema-decay", type=float, default=0.0, dest="flow_ema_decay",
                    help="EMA decay for flow/conditioner weights; 0 disables EMA")
    ap.add_argument("--no-ema-warmup", action="store_true", dest="no_ema_warmup",
                    help="use the exact EMA decay from the first update instead of ramping it")
    ap.add_argument("--ema-eval-mode", default="auto", choices=EVAL_WEIGHT_MODES,
                    dest="ema_eval_mode",
                    help="which weights to use for the final train report")
    ap.add_argument("--no-ema-eval", action="store_true", dest="no_ema_eval",
                    help="compatibility alias for --ema-eval-mode raw")
    ap.add_argument("--time-sampling", default="uniform", choices=("uniform", "logit-normal"),
                    dest="time_sampling",
                    help="rectified-flow training timestep distribution")
    ap.add_argument("--time-logit-mean", type=float, default=0.0, dest="time_logit_mean",
                    help="mean for --time-sampling logit-normal")
    ap.add_argument("--time-logit-std", type=float, default=1.0, dest="time_logit_std",
                    help="stddev for --time-sampling logit-normal")
    ap.add_argument("--time-shift", type=float, default=1.0, dest="time_shift",
                    help="rectified-flow data-time shift; >1 biases training toward noise")
    ap.add_argument("--sample-time-shift", type=float, default=None,
                    dest="sample_time_shift",
                    help="override checkpoint sample-time shift; omitted uses checkpoint metadata")
    ap.add_argument("--latent-normalize", default="none",
                    choices=("none", "global", "channel"), dest="latent_normalize",
                    help="normalize AE latents before flow training/sampling")
    ap.add_argument("--latent-stat-samples", type=int, default=512,
                    dest="latent_stat_samples",
                    help="number of AE samples used to estimate latent normalization stats")
    ap.add_argument("--cond-mode", default="facts", choices=("facts", "text"), dest="cond_mode",
                    help="conditioning source for latent flow")
    ap.add_argument("--text-cond-dim", type=int, default=0, dest="text_cond_dim",
                    help="text condition vector width; default uses --hidden")
    ap.add_argument("--image-manifest", default="", dest="image_manifest",
                    help="JSONL/CSV/TSV captioned image manifest for real image-text training")
    ap.add_argument("--image-root", default="", dest="image_root",
                    help="base directory for relative paths in --image-manifest")
    ap.add_argument("--image-split", default="train", dest="image_split",
                    help="manifest split to train/evaluate")
    ap.add_argument("--image-min-aesthetic", type=float, default=None,
                    dest="image_min_aesthetic",
                    help="optional minimum aesthetic/quality score for manifest rows")
    ap.add_argument("--image-max-records", type=int, default=0, dest="image_max_records",
                    help="cap manifest rows for smoke tests; 0 means all")
    ap.add_argument("--caption-vocab-max", type=int, default=8192, dest="caption_vocab_max",
                    help="maximum caption vocabulary size for image manifests")
    ap.add_argument("--caption-max-len", type=int, default=64, dest="caption_max_len",
                    help="maximum caption tokens for image manifests")
    ap.add_argument("--caption-cond-source", default="tokens",
                    choices=("tokens", "embedding", "auto"), dest="caption_cond_source",
                    help="caption conditioning source for image manifests")
    ap.add_argument("--image-crop-mode", default="center",
                    choices=("center", "random", "none"), dest="image_crop_mode",
                    help="crop mode for manifest training images")
    ap.add_argument("--image-hflip-prob", type=float, default=0.0,
                    dest="image_hflip_prob",
                    help="random horizontal flip probability for manifest training images")
    ap.add_argument("--prompt-templates", default="", dest="prompt_templates",
                    help="semicolon-separated prompt templates using {color} and {shape}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/image_latent_flow.pt")
    ap.add_argument("--eval-out", default="", dest="eval_out",
                    help="JSON path for --eval-checkpoint report")
    ap.add_argument("--sample-grid-out", default="", dest="sample_grid_out",
                    help="optional PPM path for a color x shape generated sample grid")
    ap.add_argument("--sample-grid-samples", type=int, default=1,
                    dest="sample_grid_samples",
                    help="generated samples per color/shape condition in --sample-grid-out")
    ap.add_argument("--checkpoint-weight-mode", default="auto", choices=EVAL_WEIGHT_MODES,
                    dest="checkpoint_weight_mode",
                    help="which checkpoint weights to sweep: raw, ema, or measured auto-select")
    ap.add_argument("--no-ema-checkpoint", action="store_true", dest="no_ema_checkpoint",
                    help="compatibility alias for --checkpoint-weight-mode raw")
    args = ap.parse_args(argv)
    try:
        cfg_interval = _parse_interval(args.cfg_interval)
        semantic_guidance_interval = _parse_interval(args.semantic_guidance_interval)
    except ValueError as e:
        ap.error(str(e))
    if args.selftest:
        selftest()
        return
    if args.eval_checkpoint:
        report = evaluate_checkpoint(
            args.eval_checkpoint,
            cfg_scales=_parse_number_list(args.cfg_scales, float),
            sample_steps_list=_parse_number_list(args.sample_steps_list, int),
            seed=args.seed,
            size=args.size,
            eval_seeds=(_parse_number_list(args.eval_seeds, int) if args.eval_seeds else None),
            roundtrip_samples=args.roundtrip_samples,
            prefer_ema=not args.no_ema_checkpoint,
            weight_mode=("raw" if args.no_ema_checkpoint else args.checkpoint_weight_mode),
            intervention_samples=args.intervention_samples,
            semantic_guidance_w=args.semantic_guidance_w,
            semantic_guidance_weights=(
                _parse_number_list(args.semantic_guidance_weights, float)
                if args.semantic_guidance_weights else None
            ),
            semantic_guidance_mode=args.semantic_guidance_mode,
            sample_method=args.sample_method,
            sample_methods=(
                _parse_string_list(args.sample_methods) if args.sample_methods else None
            ),
            cfg_interval=cfg_interval,
            semantic_guidance_interval=semantic_guidance_interval,
            eval_image_manifest=args.eval_image_manifest,
            eval_image_root=args.eval_image_root,
            eval_image_split=args.eval_image_split,
            eval_image_min_aesthetic=args.eval_image_min_aesthetic,
            eval_image_max_records=args.eval_image_max_records,
            sample_time_shift=args.sample_time_shift,
        )
        if args.sample_grid_out:
            selected_weights = report.get("selected_checkpoint_weights",
                                          report.get("checkpoint_weight_mode", "ema"))
            ae, flow, conditioner, prompt_vocab, prompt_templates, meta = load_checkpoint(
                args.eval_checkpoint, prefer_ema=(selected_weights == "ema"))
            settings = selected_grid_settings(
                report,
                fallback_cfg=args.cfg_scale,
                fallback_steps=args.sample_steps,
                fallback_method=args.sample_method,
                fallback_semantic_w=args.semantic_guidance_w,
                fallback_semantic_mode=args.semantic_guidance_mode,
                fallback_cfg_interval=cfg_interval,
                fallback_semantic_interval=semantic_guidance_interval,
                fallback_sample_time_shift=(
                    args.sample_time_shift if args.sample_time_shift is not None
                    else report.get("sample_time_shift", 1.0)
                ))
            grid_size = int(args.size or report.get("image_size", meta.get("image_size", 32))
                            or 32)
            if report.get("experiment") == "image_latent_manifest_sampler_sweep":
                grid_records = read_image_manifest(
                    report["eval_image_manifest"], root=report.get("eval_image_root", ""),
                    split=report.get("eval_image_split", "eval"),
                    min_aesthetic=report.get("eval_image_min_aesthetic"),
                    max_records=report.get("eval_image_max_records", 0))
                grid_meta = save_caption_sample_grid(
                    ae, flow, grid_records, args.sample_grid_out, conditioner=conditioner,
                    prompt_vocab=prompt_vocab, caption_max_len=meta["caption_max_len"],
                    size=grid_size,
                    cfg_scale=settings["cfg_scale"],
                    cfg_interval=settings["cfg_interval"],
                    sample_time_shift=settings["sample_time_shift"],
                    sample_steps=settings["sample_steps"],
                    sample_method=settings["sample_method"],
                    samples=args.sample_grid_samples,
                    seed=args.seed + 991,
                    caption_cond_source=meta["caption_cond_source"])
            else:
                grid_meta = save_sample_grid(
                    ae, flow, args.sample_grid_out, size=grid_size, cond_mode=meta["cond_mode"],
                    conditioner=conditioner, prompt_vocab=prompt_vocab,
                    prompt_templates=prompt_templates,
                    cfg_scale=settings["cfg_scale"],
                    cfg_interval=settings["cfg_interval"],
                    sample_steps=settings["sample_steps"],
                    sample_time_shift=settings["sample_time_shift"],
                    sample_method=settings["sample_method"],
                    semantic_guidance_w=settings["semantic_guidance_w"],
                    semantic_guidance_mode=settings["semantic_guidance_mode"],
                    semantic_guidance_interval=settings["semantic_guidance_interval"],
                    samples_per_combo=args.sample_grid_samples,
                    seed=args.seed + 991)
            grid_meta["sample_grid_checkpoint_weight_mode"] = (
                "ema" if meta["ema_loaded"] else "raw")
            report.update(grid_meta)
        if args.eval_out:
            os.makedirs(os.path.dirname(args.eval_out) or ".", exist_ok=True)
            with open(args.eval_out, "w") as f:
                json.dump(report, f, indent=1)
            print(json.dumps(report, indent=1))
            print(f"saved -> {args.eval_out}")
        else:
            print(json.dumps(report, indent=1))
        return
    if not args.train:
        ap.error("use --selftest, --train, or --eval-checkpoint")
    templates = _parse_templates(args.prompt_templates)
    run_size = int(args.size or 32)
    (ae, flow, conditioner, prompt_vocab, text_aligner, report,
     flow_ema, conditioner_ema) = train_latent_flow(
        ae_steps=args.ae_steps, flow_steps=args.flow_steps, batch=args.batch,
        latent_ch=args.latent_ch, hidden=args.hidden, lr=args.lr, fact_w=args.fact_w,
        seed=args.seed, size=run_size, flow_arch=args.flow_arch, dit_depth=args.dit_depth,
        dit_heads=args.dit_heads, cond_drop=args.cond_drop, cfg_scale=args.cfg_scale,
        dit_head_width_mult=args.dit_head_width_mult,
        dit_qk_norm=args.dit_qk_norm,
        dit_attn_impl=args.dit_attn_impl,
        latent_max_tokens=args.latent_max_tokens, ae_arch=args.ae_arch,
        latent_downsample=args.latent_downsample, ae_res_blocks=args.ae_res_blocks,
        ae_recon_loss=args.ae_recon_loss, ae_grad_w=args.ae_grad_w,
        ae_ms_w=args.ae_ms_w, ae_latent_reg_w=args.ae_latent_reg_w,
        image_text_align_w=args.image_text_align_w,
        flow_text_align_w=args.flow_text_align_w,
        text_embed_dim=args.text_embed_dim,
        image_feature_align_w=args.image_feature_align_w,
        flow_feature_align_w=args.flow_feature_align_w,
        image_feature_embed_dim=args.image_feature_embed_dim,
        sample_steps=args.sample_steps, roundtrip_samples=args.roundtrip_samples,
        flow_semantic_w=args.flow_semantic_w, cond_mode=args.cond_mode,
        flow_consistency_w=args.flow_consistency_w,
        text_cond_dim=args.text_cond_dim, prompt_templates=templates,
        image_manifest=args.image_manifest, image_root=args.image_root,
        image_split=args.image_split, image_min_aesthetic=args.image_min_aesthetic,
        image_max_records=args.image_max_records, caption_vocab_max=args.caption_vocab_max,
        caption_max_len=args.caption_max_len, caption_cond_source=args.caption_cond_source,
        image_crop_mode=args.image_crop_mode, image_hflip_prob=args.image_hflip_prob,
        time_sampling=args.time_sampling, time_logit_mean=args.time_logit_mean,
        time_logit_std=args.time_logit_std,
        time_shift=args.time_shift,
        latent_normalize=args.latent_normalize,
        latent_stat_samples=args.latent_stat_samples,
        ae_intervention_w=args.ae_intervention_w,
        ae_factor_orth_w=args.ae_factor_orth_w,
        semantic_guidance_w=args.semantic_guidance_w,
        semantic_guidance_mode=args.semantic_guidance_mode,
        cfg_interval=cfg_interval,
        semantic_guidance_interval=semantic_guidance_interval,
        sample_method=args.sample_method,
        flow_ema_decay=args.flow_ema_decay,
        flow_ema_warmup=not args.no_ema_warmup,
        eval_with_ema=not args.no_ema_eval,
        eval_weight_mode=("raw" if args.no_ema_eval else args.ema_eval_mode),
        intervention_samples=args.intervention_samples,
        train_precision=args.train_precision,
        ae_accum_steps=args.ae_accum_steps,
        flow_accum_steps=args.flow_accum_steps,
        grad_clip=args.grad_clip,
        flow_cache_latents=args.flow_cache_latents,
        flow_cache_records=args.flow_cache_records,
        flow_cache_batch=args.flow_cache_batch,
        flow_cache_dir=args.flow_cache_dir,
        flow_cache_shard_size=args.flow_cache_shard_size,
        return_conditioner=True, return_ema=True, return_aligner=True)
    if args.sample_grid_out:
        raw_flow = clone_state_dict(flow)
        raw_conditioner = clone_state_dict(conditioner) if conditioner is not None else None
        grid_weight_mode = "raw"
        if report.get("selected_eval_weights") == "ema" and flow_ema is not None:
            load_flow_state(flow, flow_ema)
            if conditioner is not None and conditioner_ema:
                conditioner.load_state_dict(conditioner_ema)
            grid_weight_mode = "ema"
        settings = selected_grid_settings(
            report,
            fallback_cfg=args.cfg_scale,
            fallback_steps=args.sample_steps,
            fallback_method=args.sample_method,
            fallback_semantic_w=args.semantic_guidance_w,
            fallback_semantic_mode=args.semantic_guidance_mode,
            fallback_cfg_interval=cfg_interval,
            fallback_semantic_interval=semantic_guidance_interval,
            fallback_sample_time_shift=args.time_shift)
        if args.image_manifest:
            grid_records = read_image_manifest(
                args.image_manifest, root=args.image_root, split=args.image_split,
                min_aesthetic=args.image_min_aesthetic, max_records=args.image_max_records)
            grid_meta = save_caption_sample_grid(
                ae, flow, grid_records, args.sample_grid_out, conditioner=conditioner,
                prompt_vocab=prompt_vocab, caption_max_len=args.caption_max_len,
                size=run_size,
                cfg_scale=settings["cfg_scale"],
                cfg_interval=settings["cfg_interval"],
                sample_steps=settings["sample_steps"],
                sample_time_shift=settings["sample_time_shift"],
                sample_method=settings["sample_method"],
                samples=args.sample_grid_samples,
                seed=args.seed + 991,
                caption_cond_source=report.get("caption_cond_source", "tokens"))
        else:
            grid_meta = save_sample_grid(
                ae, flow, args.sample_grid_out, size=run_size, cond_mode=args.cond_mode,
                conditioner=conditioner,
                prompt_vocab=prompt_vocab, prompt_templates=templates,
                cfg_scale=settings["cfg_scale"],
                cfg_interval=settings["cfg_interval"],
                sample_steps=settings["sample_steps"],
                sample_time_shift=settings["sample_time_shift"],
                sample_method=settings["sample_method"],
                semantic_guidance_w=settings["semantic_guidance_w"],
                semantic_guidance_mode=settings["semantic_guidance_mode"],
                semantic_guidance_interval=settings["semantic_guidance_interval"],
                samples_per_combo=args.sample_grid_samples,
                seed=args.seed + 991)
        grid_meta["sample_grid_checkpoint_weight_mode"] = grid_weight_mode
        report.update(grid_meta)
        load_flow_state(flow, raw_flow)
        if conditioner is not None and raw_conditioner is not None:
            conditioner.load_state_dict(raw_conditioner)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "autoencoder_state_dict": ae.state_dict(),
        "flow_state_dict": flow.state_dict(),
        "report": report,
        "fact_vocab": FACT_VOCAB,
        "latent_ch": args.latent_ch,
        "image_size": run_size,
        "ae_arch": args.ae_arch,
        "latent_downsample": report.get("latent_downsample", args.latent_downsample),
        "ae_res_blocks": report.get("ae_res_blocks", args.ae_res_blocks),
        "latent_max_tokens": args.latent_max_tokens,
        "ae_recon_loss": args.ae_recon_loss,
        "ae_grad_w": args.ae_grad_w,
        "ae_ms_w": args.ae_ms_w,
        "ae_latent_reg_w": args.ae_latent_reg_w,
        "image_text_align_w": args.image_text_align_w,
        "flow_text_align_w": args.flow_text_align_w,
        "text_embed_dim": args.text_embed_dim,
        "image_feature_align_w": args.image_feature_align_w,
        "flow_feature_align_w": args.flow_feature_align_w,
        "image_feature_embed_dim": args.image_feature_embed_dim,
        "hidden": args.hidden,
        "ae_accum_steps": args.ae_accum_steps,
        "flow_accum_steps": args.flow_accum_steps,
        "flow_cache_latents": report.get("flow_cache_latents", False),
        "flow_cache_backend": report.get("flow_cache_backend", ""),
        "flow_cache_dir": report.get("flow_cache_dir", ""),
        "flow_cache_records": report.get("flow_cache_records", 0),
        "flow_cache_max_records": args.flow_cache_records,
        "flow_cache_batch": args.flow_cache_batch,
        "flow_cache_shard_size": args.flow_cache_shard_size,
        "flow_cache_shards": report.get("flow_cache_shards", 0),
        "flow_cache_bytes": report.get("flow_cache_bytes", 0),
        "train_precision": args.train_precision,
        "train_amp_enabled": report.get("train_amp_enabled", False),
        "grad_clip": args.grad_clip,
        "flow_arch": args.flow_arch,
        "dit_depth": args.dit_depth,
        "dit_heads": args.dit_heads,
        "dit_head_width_mult": args.dit_head_width_mult,
        "dit_qk_norm": report.get("dit_qk_norm", False),
        "dit_attn_impl": report.get("dit_attn_impl", "manual"),
        "cond_mode": args.cond_mode,
        "cond_dim": report["cond_dim"],
        "data_mode": report.get("data_mode", "synthetic_factors"),
        "image_manifest": args.image_manifest,
        "image_root": args.image_root,
        "image_split": args.image_split,
        "image_min_aesthetic": args.image_min_aesthetic,
        "image_max_records": args.image_max_records,
        "image_crop_mode": report.get("image_crop_mode", ""),
        "image_hflip_prob": report.get("image_hflip_prob", 0.0),
        "caption_vocab_max": args.caption_vocab_max,
        "caption_max_len": report.get("caption_max_len", 0),
        "caption_cond_source": report.get("caption_cond_source", ""),
        "text_embedding_in_dim": report.get("text_embedding_in_dim", 0),
        "image_embedding_in_dim": report.get("image_embedding_in_dim", 0),
        "cond_drop": args.cond_drop,
        "cfg_scale": args.cfg_scale,
        "cfg_interval": list(cfg_interval),
        "sample_steps": args.sample_steps,
        "sample_method": args.sample_method,
        "semantic_guidance_w": args.semantic_guidance_w,
        "semantic_guidance_mode": args.semantic_guidance_mode,
        "semantic_guidance_interval": list(semantic_guidance_interval),
        "intervention_samples": args.intervention_samples,
        "ae_intervention_w": args.ae_intervention_w,
        "ae_factor_orth_w": args.ae_factor_orth_w,
        "flow_semantic_w": args.flow_semantic_w,
        "flow_consistency_w": args.flow_consistency_w,
        "flow_ema_decay": args.flow_ema_decay,
        "flow_ema_warmup": not args.no_ema_warmup,
        "flow_ema_effective_decay": report.get("flow_ema_effective_decay", 0.0),
        "eval_weight_mode": report.get("eval_weight_mode", "raw"),
        "selected_eval_weights": report.get("selected_eval_weights", "raw"),
        "time_sampling": args.time_sampling,
        "time_logit_mean": args.time_logit_mean,
        "time_logit_std": args.time_logit_std,
        "time_shift": args.time_shift,
        "sample_time_shift": report.get("sample_time_shift", args.time_shift),
        "latent_normalize": args.latent_normalize,
        "latent_stat_samples": args.latent_stat_samples,
        "latent_stats": latent_stats_state(flow_latent_stats(flow)),
        "prompt_templates": list(templates) if args.cond_mode == "text" else [],
        "prompt_vocab": prompt_vocab if prompt_vocab is not None else {},
        "conditioner_state_dict": (conditioner.state_dict() if conditioner is not None else {}),
        "text_aligner_state_dict": (
            text_aligner.state_dict() if text_aligner is not None else {}
        ),
        "image_feature_aligner_state_dict": (
            getattr(flow, "image_feature_aligner", None).state_dict()
            if getattr(flow, "image_feature_aligner", None) is not None else {}
        ),
        "flow_ema_state_dict": flow_ema if flow_ema is not None else {},
        "conditioner_ema_state_dict": (conditioner_ema if conditioner_ema is not None else {}),
    }, args.out)
    print(json.dumps(report, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
