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
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LatentDiTFlowNet(nn.Module):
    """Patch-token transformer velocity field over semantic image latents.

    This is intentionally tiny, but it matches the scalable architecture shape: flatten the
    autoencoder latent grid into tokens, condition every token on time + canonical facts, run
    transformer blocks, then project token velocities back to the latent grid.
    """

    def __init__(self, latent_ch=16, hidden=96, depth=3, heads=4, cond_dim=None, max_tokens=256):
        super().__init__()
        cond_dim = cond_dim or len(FACT_VOCAB)
        self.latent_ch = int(latent_ch)
        self.hidden = int(hidden)
        self.max_tokens = int(max_tokens)
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
        self.out_proj = nn.Linear(hidden, latent_ch)

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
                 max_tokens=256):
        super().__init__()
        cond_dim = cond_dim or len(FACT_VOCAB)
        self.latent_ch = int(latent_ch)
        self.hidden = int(hidden)
        self.max_tokens = int(max_tokens)
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
        self.out_proj = nn.Linear(hidden, latent_ch)

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


class MMDiTBlock(nn.Module):
    """Tiny dual-stream block with separate image/text projections and joint attention."""

    def __init__(self, hidden=96, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** -0.5
        self.img_norm = nn.LayerNorm(hidden)
        self.ctx_norm = nn.LayerNorm(hidden)
        self.img_qkv = nn.Linear(hidden, hidden * 3)
        self.ctx_qkv = nn.Linear(hidden, hidden * 3)
        self.img_out = nn.Linear(hidden, hidden)
        self.ctx_out = nn.Linear(hidden, hidden)
        self.img_ff_norm = nn.LayerNorm(hidden)
        self.ctx_ff_norm = nn.LayerNorm(hidden)
        self.img_ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(),
                                    nn.Linear(hidden * 4, hidden))
        self.ctx_ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(),
                                    nn.Linear(hidden * 4, hidden))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 8))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def _qkv(self, proj, x):
        b, n, h = x.shape
        q, k, v = proj(x).view(b, n, 3, self.heads, self.head_dim).unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, img, ctx, cond_ctx, ctx_mask=None):
        b, n_img, h = img.shape
        n_ctx = ctx.shape[1]
        (img_attn_shift, img_attn_scale, ctx_attn_shift, ctx_attn_scale,
         img_ff_shift, img_ff_scale, ctx_ff_shift, ctx_ff_scale) = self.ada(cond_ctx).chunk(
             8, dim=-1)
        img_attn = self._modulate(self.img_norm(img), img_attn_shift, img_attn_scale)
        ctx_attn = self._modulate(self.ctx_norm(ctx), ctx_attn_shift, ctx_attn_scale)
        qi, ki, vi = self._qkv(self.img_qkv, img_attn)
        qc, kc, vc = self._qkv(self.ctx_qkv, ctx_attn)
        q = torch.cat([qi, qc], dim=2)
        k = torch.cat([ki, kc], dim=2)
        v = torch.cat([vi, vc], dim=2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if ctx_mask is not None:
            img_mask = torch.zeros((b, n_img), dtype=torch.bool, device=ctx_mask.device)
            key_mask = torch.cat([img_mask, ctx_mask], dim=1)
            attn = attn.masked_fill(key_mask[:, None, None, :], torch.finfo(attn.dtype).min)
        mixed = torch.softmax(attn, dim=-1).matmul(v).transpose(1, 2).reshape(b, n_img + n_ctx, h)
        img_delta, ctx_delta = mixed[:, :n_img], mixed[:, n_img:]
        img = img + self.img_out(img_delta)
        ctx = ctx + self.ctx_out(ctx_delta)
        img_ff = self._modulate(self.img_ff_norm(img), img_ff_shift, img_ff_scale)
        ctx_ff = self._modulate(self.ctx_ff_norm(ctx), ctx_ff_shift, ctx_ff_scale)
        img = img + self.img_ff(img_ff)
        ctx = ctx + self.ctx_ff(ctx_ff)
        if ctx_mask is not None:
            ctx = ctx.masked_fill(ctx_mask[:, :, None], 0.0)
        return img, ctx


class LatentMMDiTFlowNet(nn.Module):
    """Toy MM-DiT latent flow with bidirectional image/condition token mixing."""

    uses_cond_tokens = True
    uses_adaptive_modulation = True

    def __init__(self, latent_ch=16, hidden=96, depth=3, heads=4, cond_dim=None,
                 max_tokens=256):
        super().__init__()
        cond_dim = cond_dim or len(FACT_VOCAB)
        self.latent_ch = int(latent_ch)
        self.hidden = int(hidden)
        self.max_tokens = int(max_tokens)
        self.in_proj = nn.Linear(latent_ch, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden))
        self.time = nn.Sequential(nn.Linear(cond_dim + 1, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden))
        self.ctx_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([MMDiTBlock(hidden, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, latent_ch)

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


def make_flow(flow_arch="conv", latent_ch=16, hidden=64, dit_depth=3, dit_heads=4, cond_dim=None):
    if flow_arch == "conv":
        return LatentFlowNet(latent_ch=latent_ch, hidden=hidden, cond_dim=cond_dim)
    if flow_arch == "dit":
        heads = max(1, min(dit_heads, hidden // 16))
        while hidden % heads:
            heads -= 1
        return LatentDiTFlowNet(latent_ch=latent_ch, hidden=hidden, depth=dit_depth, heads=heads,
                                cond_dim=cond_dim)
    if flow_arch == "crossdit":
        heads = max(1, min(dit_heads, hidden // 16))
        while hidden % heads:
            heads -= 1
        return LatentCrossDiTFlowNet(latent_ch=latent_ch, hidden=hidden, depth=dit_depth,
                                     heads=heads, cond_dim=cond_dim)
    if flow_arch == "mmdit":
        heads = max(1, min(dit_heads, hidden // 16))
        while hidden % heads:
            heads -= 1
        return LatentMMDiTFlowNet(latent_ch=latent_ch, hidden=hidden, depth=dit_depth,
                                  heads=heads, cond_dim=cond_dim)
    raise ValueError(f"unknown latent flow architecture {flow_arch!r}")


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


def sample_flow_times(batch, device=DEV, mode="uniform", logit_mean=0.0, logit_std=1.0):
    """Sample rectified-flow interpolation times.

    Uniform is the original rectified-flow baseline.  Logit-normal follows the SD3-style
    direction of spending more training mass on perceptually useful intermediate noise scales.
    """
    if mode == "uniform":
        return torch.rand(batch, 1, 1, 1, device=device)
    if mode == "logit-normal":
        eps = torch.randn(batch, 1, 1, 1, device=device) * float(logit_std) + float(logit_mean)
        return torch.sigmoid(eps)
    raise ValueError(f"unknown time sampling mode {mode!r}")


def autoencoder_loss(out, x, yc, ys, fact_w=0.25):
    recon = F.mse_loss(out["recon"], x)
    facts = F.cross_entropy(out["color"], yc) + F.cross_entropy(out["shape"], ys)
    return recon + fact_w * facts, {"recon_mse": recon.detach(), "fact_ce": facts.detach()}


def flow_uses_cond_tokens(flow):
    return bool(getattr(flow, "uses_cond_tokens", False))


def condition_vector(cond):
    return cond["vec"] if isinstance(cond, dict) else cond


def condition_batch(cond):
    return int(condition_vector(cond).shape[0])


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


def semantic_endpoint_loss(ae, z_clean, cond):
    """REPA-style endpoint alignment: clean latent should decode to conditioned facts."""
    logits = ae.fact_logits(z_clean)
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
        return z_clean.sum() * 0.0, {}
    loss = sum(losses.values()) / len(losses)
    parts = {f"{pred}_endpoint_ce": val.detach() for pred, val in losses.items()}
    return loss, parts


def latent_flow_losses(flow, z1, cond, cond_drop=0.0, ae=None, semantic_w=0.0,
                       semantic_cond=None, time_sampling="uniform", time_logit_mean=0.0,
                       time_logit_std=1.0):
    x0 = torch.randn_like(z1)
    t = sample_flow_times(z1.shape[0], device=z1.device, mode=time_sampling,
                          logit_mean=time_logit_mean, logit_std=time_logit_std)
    zt = (1.0 - t) * x0 + t * z1
    target = z1 - x0
    cond_model = condition_dropout(cond, cond_drop)
    pred = flow(zt, t, cond_model)
    velocity = F.mse_loss(pred, target)
    total = velocity
    parts = {
        "velocity_mse": velocity.detach(),
        "time_mean": t.detach().mean(),
        "time_std": t.detach().std(unbiased=False),
    }
    if ae is not None and semantic_w > 0.0:
        z_clean = zt + (1.0 - t) * pred
        sem_cond = cond if semantic_cond is None else semantic_cond
        cond_vec = condition_vector(cond_model)
        keep = cond_vec.detach().abs().sum(dim=1, keepdim=True).gt(0).to(sem_cond.dtype)
        semantic, sem_parts = semantic_endpoint_loss(ae, z_clean, sem_cond * keep)
        total = total + semantic_w * semantic
        parts["semantic_endpoint_ce"] = semantic.detach()
        parts.update(sem_parts)
    return total, parts


def latent_flow_loss(flow, z1, cond, cond_drop=0.0, time_sampling="uniform",
                     time_logit_mean=0.0, time_logit_std=1.0):
    loss, _parts = latent_flow_losses(flow, z1, cond, cond_drop=cond_drop,
                                      time_sampling=time_sampling,
                                      time_logit_mean=time_logit_mean,
                                      time_logit_std=time_logit_std)
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
                   cfg_scale=1.0):
    batch = condition_batch(cond)
    z = _seeded_randn((batch,) + tuple(latent_shape), device=device, seed=seed)
    flow.eval()
    dt = 1.0 / max(1, steps)
    for i in range(steps):
        t = torch.full((batch, 1, 1, 1), i / max(1, steps), device=device)
        z = z + dt * guided_velocity(flow, z, t, cond, cfg_scale=cfg_scale)
    return z


@torch.no_grad()
def sample_images(ae, flow, cond, latent_shape=(16, 8, 8), steps=16, device=DEV, seed=0,
                  cfg_scale=1.0):
    z = sample_latents(flow, cond, latent_shape=latent_shape, steps=steps, device=device,
                       seed=seed, cfg_scale=cfg_scale)
    ae.eval()
    return ae.decode(z).clamp(-1.0, 1.0)


@torch.no_grad()
def conditional_roundtrip(ae, flow, size=32, device=DEV, cfg_scale=1.0, sample_steps=4,
                          samples_per_combo=1, seed=20, cond_mode="facts", conditioner=None,
                          prompt_vocab=None, prompt_templates=DEFAULT_PROMPT_TEMPLATES):
    """Generate from every canonical color/shape request and re-read facts from the image."""
    ae.eval()
    flow.eval()
    got_c = got_s = got_both = total = 0
    mses = []
    latent_shape = (ae.latent_ch, size // 4, size // 4)
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
                                   cfg_scale=cfg_scale)
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
             prompt_vocab=None, prompt_templates=DEFAULT_PROMPT_TEMPLATES):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ae.eval()
    flow.eval()
    recon_losses, flow_losses = [], []
    got_c = got_s = total = 0
    latent_means, latent_stds = [], []
    while total < n:
        b = min(batch, n - total)
        x, fact_cond, yc, ys, specs = _batch(b, rng, size=size, device=device, return_specs=True)
        out = ae(x)
        z = out["latent"]
        recon_losses.append(float(F.mse_loss(out["recon"], x).detach().cpu()))
        cond = model_condition(specs, fact_cond, cond_mode=cond_mode, conditioner=conditioner,
                               prompt_vocab=prompt_vocab, prompt_templates=prompt_templates,
                               rng=rng, device=device, return_tokens=flow_uses_cond_tokens(flow))
        flow_losses.append(float(latent_flow_loss(flow, z, cond).detach().cpu()))
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
    sample = sample_images(ae, flow, cond, latent_shape=(ae.latent_ch, size // 4, size // 4),
                           steps=sample_steps, device=device, seed=seed, cfg_scale=cfg_scale)
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
        "cfg_scale": float(cfg_scale),
        "sample_steps": int(sample_steps),
    }
    report.update(conditional_roundtrip(ae, flow, size=size, device=device, cfg_scale=cfg_scale,
                                        sample_steps=sample_steps,
                                        samples_per_combo=roundtrip_samples, seed=seed + 17,
                                        cond_mode=cond_mode, conditioner=conditioner,
                                        prompt_vocab=prompt_vocab,
                                        prompt_templates=prompt_templates))
    report["cond_mode"] = cond_mode
    return report


def load_checkpoint(path, device=DEV):
    ckpt = torch.load(path, map_location=device)
    report = ckpt.get("report", {})
    latent_ch = int(ckpt.get("latent_ch", report.get("latent_ch", 16)))
    hidden = int(ckpt.get("hidden", report.get("hidden", 64)))
    flow_arch = ckpt.get("flow_arch", report.get("flow_arch", "conv"))
    dit_depth = int(ckpt.get("dit_depth", report.get("dit_depth", 3)))
    dit_heads = int(ckpt.get("dit_heads", report.get("dit_heads", 4)))
    cond_mode = ckpt.get("cond_mode", report.get("cond_mode", "facts"))
    cond_dim = int(ckpt.get("cond_dim", report.get("cond_dim", len(FACT_VOCAB))))
    prompt_templates = tuple(ckpt.get("prompt_templates", report.get("prompt_templates", []))
                             or DEFAULT_PROMPT_TEMPLATES)
    prompt_vocab = ckpt.get("prompt_vocab") or None
    conditioner = None
    if cond_mode == "text":
        if prompt_vocab is None:
            prompt_vocab = build_prompt_vocab(prompt_templates)
        conditioner = PromptConditioner(len(prompt_vocab), cond_dim=cond_dim,
                                        hidden=hidden).to(device)
        conditioner.load_state_dict(ckpt["conditioner_state_dict"])
        conditioner.eval()
    ae = SemanticAutoencoder(latent_ch=latent_ch, hidden=hidden).to(device)
    flow = make_flow(flow_arch=flow_arch, latent_ch=latent_ch, hidden=hidden,
                     dit_depth=dit_depth, dit_heads=dit_heads, cond_dim=cond_dim).to(device)
    ae.load_state_dict(ckpt["autoencoder_state_dict"])
    flow.load_state_dict(ckpt["flow_state_dict"])
    ae.eval()
    flow.eval()
    return ae, flow, conditioner, prompt_vocab, prompt_templates, {
        "checkpoint": path,
        "checkpoint_report": report,
        "latent_ch": latent_ch,
        "hidden": hidden,
        "flow_arch": flow_arch,
        "dit_depth": dit_depth if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "dit_heads": dit_heads if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "adaptive_modulation": bool(getattr(flow, "uses_adaptive_modulation", False)),
        "cond_mode": cond_mode,
        "cond_dim": cond_dim,
    }


@torch.no_grad()
def sampler_sweep(ae, flow, cfg_scales=(1.0, 1.5), sample_steps_list=(4, 8),
                  n=128, batch=64, seed=10, size=32, device=DEV, roundtrip_samples=1,
                  cond_mode="facts", conditioner=None, prompt_vocab=None,
                  prompt_templates=DEFAULT_PROMPT_TEMPLATES):
    rows = []
    for cfg_scale in cfg_scales:
        for sample_steps in sample_steps_list:
            row = evaluate(ae, flow, n=n, batch=batch, seed=seed, size=size, device=device,
                           cfg_scale=float(cfg_scale), sample_steps=int(sample_steps),
                           roundtrip_samples=roundtrip_samples, cond_mode=cond_mode,
                           conditioner=conditioner, prompt_vocab=prompt_vocab,
                           prompt_templates=prompt_templates)
            row["sweep_key"] = f"cfg={float(cfg_scale):g};steps={int(sample_steps)}"
            rows.append(row)
    return rows


SWEEP_METRICS = (
    "sample_roundtrip_color_acc",
    "sample_roundtrip_shape_acc",
    "sample_roundtrip_both_acc",
    "conditional_sample_mse",
    "sample_center_target_mse",
    "latent_velocity_mse",
)


def aggregate_sweep_rows(rows):
    grouped = {}
    for row in rows:
        key = (float(row["cfg_scale"]), int(row["sample_steps"]))
        grouped.setdefault(key, []).append(row)
    out = []
    for (cfg_scale, sample_steps), group in sorted(grouped.items()):
        agg = {
            "sweep_key": f"cfg={cfg_scale:g};steps={sample_steps}",
            "cfg_scale": float(cfg_scale),
            "sample_steps": int(sample_steps),
            "runs": len(group),
            "eval_seeds": [int(r["eval_seed"]) for r in group if "eval_seed" in r],
        }
        for metric in SWEEP_METRICS:
            vals = np.asarray([float(r[metric]) for r in group], dtype=np.float64)
            agg[f"{metric}_mean"] = float(vals.mean())
            agg[f"{metric}_std"] = float(vals.std())
            agg[f"{metric}_min"] = float(vals.min())
            agg[f"{metric}_max"] = float(vals.max())
        out.append(agg)
    return out


def evaluate_checkpoint(path, cfg_scales=(1.0, 1.5), sample_steps_list=(4, 8),
                        n=128, batch=64, seed=10, eval_seeds=None, size=32, device=DEV,
                        roundtrip_samples=1):
    ae, flow, conditioner, prompt_vocab, prompt_templates, meta = load_checkpoint(path,
                                                                                  device=device)
    eval_seeds = tuple(eval_seeds) if eval_seeds is not None else (seed,)
    rows = []
    for eval_seed in eval_seeds:
        for row in sampler_sweep(ae, flow, cfg_scales=cfg_scales,
                                 sample_steps_list=sample_steps_list, n=n, batch=batch,
                                 seed=int(eval_seed), size=size, device=device,
                                 roundtrip_samples=roundtrip_samples,
                                 cond_mode=meta["cond_mode"], conditioner=conditioner,
                                 prompt_vocab=prompt_vocab,
                                 prompt_templates=prompt_templates):
            row["eval_seed"] = int(eval_seed)
            rows.append(row)
    aggregate = aggregate_sweep_rows(rows)
    best = max(aggregate, key=lambda r: (
        r["sample_roundtrip_both_acc_mean"],
        r["sample_roundtrip_shape_acc_mean"],
        r["sample_roundtrip_color_acc_mean"],
        -r["conditional_sample_mse_mean"],
    ))
    return {
        "experiment": "image_latent_sampler_sweep",
        **meta,
        "n": int(n),
        "roundtrip_samples": int(roundtrip_samples),
        "eval_seeds": [int(s) for s in eval_seeds],
        "rows": rows,
        "aggregate": aggregate,
        "best": best,
    }


def train_latent_flow(ae_steps=200, flow_steps=200, batch=64, latent_ch=16, hidden=64,
                      lr=2e-4, fact_w=1.0, seed=0, size=32, device=DEV, flow_arch="conv",
                      dit_depth=3, dit_heads=4, cond_drop=0.0, cfg_scale=1.0,
                      sample_steps=4, roundtrip_samples=1, flow_semantic_w=0.0,
                      cond_mode="facts", text_cond_dim=0,
                      prompt_templates=DEFAULT_PROMPT_TEMPLATES, time_sampling="uniform",
                      time_logit_mean=0.0, time_logit_std=1.0,
                      return_conditioner=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if cond_mode not in ("facts", "text"):
        raise ValueError(f"unknown condition mode {cond_mode!r}")
    if time_sampling not in ("uniform", "logit-normal"):
        raise ValueError(f"unknown time sampling mode {time_sampling!r}")
    prompt_vocab = build_prompt_vocab(prompt_templates) if cond_mode == "text" else None
    cond_dim = len(FACT_VOCAB)
    conditioner = None
    if cond_mode == "text":
        cond_dim = int(text_cond_dim or hidden)
        conditioner = PromptConditioner(len(prompt_vocab), cond_dim=cond_dim,
                                        hidden=hidden).to(device)
    ae = SemanticAutoencoder(latent_ch=latent_ch, hidden=hidden).to(device)
    flow = make_flow(flow_arch=flow_arch, latent_ch=latent_ch, hidden=hidden,
                     dit_depth=dit_depth, dit_heads=dit_heads, cond_dim=cond_dim).to(device)
    opt_ae = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=0.01)
    ae.train()
    last_ae = {}
    for _ in range(ae_steps):
        x, _cond, yc, ys = _batch(batch, rng, size=size, device=device)
        out = ae(x)
        loss, parts = autoencoder_loss(out, x, yc, ys, fact_w=fact_w)
        opt_ae.zero_grad()
        loss.backward()
        opt_ae.step()
        last_ae = {k: float(v.detach().cpu()) for k, v in parts.items()}

    flow_params = list(flow.parameters()) + ([] if conditioner is None
                                             else list(conditioner.parameters()))
    opt_flow = torch.optim.AdamW(flow_params, lr=lr, weight_decay=0.01)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    flow.train()
    if conditioner is not None:
        conditioner.train()
    last_flow = {}
    for _ in range(flow_steps):
        x, fact_cond, _yc, _ys, specs = _batch(batch, rng, size=size, device=device,
                                               return_specs=True)
        with torch.no_grad():
            z1 = ae.encode(x)
        cond = model_condition(specs, fact_cond, cond_mode=cond_mode, conditioner=conditioner,
                               prompt_vocab=prompt_vocab, prompt_templates=prompt_templates,
                               rng=rng, device=device, return_tokens=flow_uses_cond_tokens(flow))
        loss, parts = latent_flow_losses(flow, z1, cond, cond_drop=cond_drop, ae=ae,
                                         semantic_w=flow_semantic_w,
                                         semantic_cond=fact_cond,
                                         time_sampling=time_sampling,
                                         time_logit_mean=time_logit_mean,
                                         time_logit_std=time_logit_std)
        opt_flow.zero_grad()
        loss.backward()
        opt_flow.step()
        last_flow = {"total_loss": float(loss.detach().cpu())}
        last_flow.update({k: float(v.detach().cpu()) for k, v in parts.items()})

    if conditioner is not None:
        conditioner.eval()
    report = evaluate(ae, flow, seed=seed + 1, size=size, device=device, cfg_scale=cfg_scale,
                      sample_steps=sample_steps, roundtrip_samples=roundtrip_samples,
                      cond_mode=cond_mode, conditioner=conditioner, prompt_vocab=prompt_vocab,
                      prompt_templates=prompt_templates)
    report.update({
        "experiment": "image3_latent_fact_conditioned_rectified_flow",
        "ae_steps": int(ae_steps),
        "flow_steps": int(flow_steps),
        "batch": int(batch),
        "latent_ch": int(latent_ch),
        "hidden": int(hidden),
        "flow_arch": flow_arch,
        "dit_depth": int(dit_depth) if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "dit_heads": int(dit_heads) if flow_arch in ("dit", "crossdit", "mmdit") else 0,
        "adaptive_modulation": bool(getattr(flow, "uses_adaptive_modulation", False)),
        "fact_w": float(fact_w),
        "cond_drop": float(cond_drop),
        "flow_semantic_w": float(flow_semantic_w),
        "time_sampling": time_sampling,
        "time_logit_mean": float(time_logit_mean),
        "time_logit_std": float(time_logit_std),
        "cond_mode": cond_mode,
        "cond_dim": int(cond_dim),
        "prompt_templates": list(prompt_templates) if cond_mode == "text" else [],
        "prompt_vocab_size": len(prompt_vocab) if prompt_vocab is not None else 0,
        "last_ae": last_ae,
        "last_flow": last_flow,
        "last_flow_loss": last_flow.get("velocity_mse"),
        "fact_vocab": [list(f) for f in FACT_VOCAB],
    })
    if return_conditioner:
        return ae, flow, conditioner, prompt_vocab, report
    return ae, flow, report


def selftest():
    ae, flow, report = train_latent_flow(ae_steps=2, flow_steps=2, batch=4, latent_ch=4,
                                         hidden=16, seed=0, device="cpu", sample_steps=2)
    assert report["experiment"] == "image3_latent_fact_conditioned_rectified_flow"
    assert report["latent_color_acc"] >= 0.0 and report["latent_shape_acc"] >= 0.0
    assert "sample_roundtrip_both_acc" in report and report["sample_roundtrip_n"] == 15
    spec = ObjectSpec("p0", "blue", "triangle")
    cond = fact_condition(object_facts(spec), device="cpu")[None]
    img = sample_images(ae, flow, cond, latent_shape=(4, 8, 8), steps=2, device="cpu", seed=0,
                        cfg_scale=1.5)
    assert img.shape == (1, 3, 32, 32)
    ae2, flow2, report2 = train_latent_flow(ae_steps=1, flow_steps=1, batch=2, latent_ch=4,
                                            hidden=32, flow_arch="dit", dit_depth=1,
                                            dit_heads=2, seed=1, device="cpu", cond_drop=0.5,
                                            cfg_scale=1.5, sample_steps=1,
                                            flow_semantic_w=0.25)
    assert report2["flow_arch"] == "dit"
    assert report2["cond_drop"] == 0.5 and report2["cfg_scale"] == 1.5
    assert report2["flow_semantic_w"] == 0.25
    assert "semantic_endpoint_ce" in report2["last_flow"]
    img2 = sample_images(ae2, flow2, cond, latent_shape=(4, 8, 8), steps=1, device="cpu",
                         seed=1, cfg_scale=1.5)
    assert img2.shape == (1, 3, 32, 32)
    sweep = sampler_sweep(ae2, flow2, cfg_scales=(1.0, 1.5), sample_steps_list=(1,),
                          n=4, batch=2, seed=3, device="cpu")
    assert len(sweep) == 2 and "sample_roundtrip_both_acc" in sweep[0]
    agg = aggregate_sweep_rows([dict(r, eval_seed=i) for i, r in enumerate(sweep)])
    assert len(agg) == 2 and "sample_roundtrip_both_acc_mean" in agg[0]
    ae3, flow3, conditioner3, vocab3, report3 = train_latent_flow(
        ae_steps=1, flow_steps=1, batch=2, latent_ch=4, hidden=32, flow_arch="dit",
        dit_depth=1, dit_heads=2, seed=2, device="cpu", cond_mode="text",
        text_cond_dim=8, flow_semantic_w=0.1, sample_steps=1, return_conditioner=True)
    assert report3["cond_mode"] == "text" and report3["prompt_vocab_size"] > 0
    text_sweep = sampler_sweep(ae3, flow3, cfg_scales=(1.0,), sample_steps_list=(1,), n=4,
                               batch=2, seed=4, device="cpu", cond_mode="text",
                               conditioner=conditioner3, prompt_vocab=vocab3)
    assert len(text_sweep) == 1 and text_sweep[0]["cond_mode"] == "text"
    ae4, flow4, conditioner4, vocab4, report4 = train_latent_flow(
        ae_steps=1, flow_steps=1, batch=2, latent_ch=4, hidden=32, flow_arch="crossdit",
        dit_depth=1, dit_heads=2, seed=5, device="cpu", cond_mode="text",
        text_cond_dim=8, sample_steps=1, time_sampling="logit-normal",
        return_conditioner=True)
    assert report4["flow_arch"] == "crossdit" and flow_uses_cond_tokens(flow4)
    assert report4["time_sampling"] == "logit-normal" and "time_mean" in report4["last_flow"]
    cross_sweep = sampler_sweep(ae4, flow4, cfg_scales=(1.0,), sample_steps_list=(1,), n=4,
                                batch=2, seed=6, device="cpu", cond_mode="text",
                                conditioner=conditioner4, prompt_vocab=vocab4)
    assert len(cross_sweep) == 1 and cross_sweep[0]["cond_mode"] == "text"
    ae5, flow5, conditioner5, vocab5, report5 = train_latent_flow(
        ae_steps=1, flow_steps=1, batch=2, latent_ch=4, hidden=32, flow_arch="mmdit",
        dit_depth=1, dit_heads=2, seed=7, device="cpu", cond_mode="text",
        text_cond_dim=8, sample_steps=1, time_sampling="logit-normal",
        return_conditioner=True)
    assert report5["flow_arch"] == "mmdit" and flow_uses_cond_tokens(flow5)
    assert report5["adaptive_modulation"] is True
    mm_sweep = sampler_sweep(ae5, flow5, cfg_scales=(1.0,), sample_steps_list=(1,), n=4,
                             batch=2, seed=8, device="cpu", cond_mode="text",
                             conditioner=conditioner5, prompt_vocab=vocab5)
    assert len(mm_sweep) == 1 and mm_sweep[0]["cond_mode"] == "text"
    print("image_latent selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval-checkpoint", default="", dest="eval_checkpoint",
                    help="load a saved image_latent checkpoint and run a sampler sweep")
    ap.add_argument("--ae-steps", type=int, default=200, dest="ae_steps")
    ap.add_argument("--flow-steps", type=int, default=200, dest="flow_steps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--latent-ch", type=int, default=16, dest="latent_ch")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fact-w", type=float, default=1.0, dest="fact_w")
    ap.add_argument("--flow-arch", default="conv", choices=("conv", "dit", "crossdit", "mmdit"),
                    dest="flow_arch")
    ap.add_argument("--dit-depth", type=int, default=3, dest="dit_depth")
    ap.add_argument("--dit-heads", type=int, default=4, dest="dit_heads")
    ap.add_argument("--cond-drop", type=float, default=0.0, dest="cond_drop")
    ap.add_argument("--cfg-scale", type=float, default=1.0, dest="cfg_scale")
    ap.add_argument("--sample-steps", type=int, default=4, dest="sample_steps")
    ap.add_argument("--cfg-scales", default="1.0,1.25,1.5,2.0", dest="cfg_scales",
                    help="comma-separated CFG scales for --eval-checkpoint")
    ap.add_argument("--sample-steps-list", default="4,8,16", dest="sample_steps_list",
                    help="comma-separated sampler step counts for --eval-checkpoint")
    ap.add_argument("--eval-seeds", default="", dest="eval_seeds",
                    help="comma-separated eval seeds for --eval-checkpoint; default uses --seed")
    ap.add_argument("--roundtrip-samples", type=int, default=1, dest="roundtrip_samples")
    ap.add_argument("--flow-semantic-w", type=float, default=0.0, dest="flow_semantic_w",
                    help="semantic endpoint alignment weight for latent flow training")
    ap.add_argument("--time-sampling", default="uniform", choices=("uniform", "logit-normal"),
                    dest="time_sampling",
                    help="rectified-flow training timestep distribution")
    ap.add_argument("--time-logit-mean", type=float, default=0.0, dest="time_logit_mean",
                    help="mean for --time-sampling logit-normal")
    ap.add_argument("--time-logit-std", type=float, default=1.0, dest="time_logit_std",
                    help="stddev for --time-sampling logit-normal")
    ap.add_argument("--cond-mode", default="facts", choices=("facts", "text"), dest="cond_mode",
                    help="conditioning source for latent flow")
    ap.add_argument("--text-cond-dim", type=int, default=0, dest="text_cond_dim",
                    help="text condition vector width; default uses --hidden")
    ap.add_argument("--prompt-templates", default="", dest="prompt_templates",
                    help="semicolon-separated prompt templates using {color} and {shape}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/image_latent_flow.pt")
    ap.add_argument("--eval-out", default="", dest="eval_out",
                    help="JSON path for --eval-checkpoint report")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if args.eval_checkpoint:
        report = evaluate_checkpoint(
            args.eval_checkpoint,
            cfg_scales=_parse_number_list(args.cfg_scales, float),
            sample_steps_list=_parse_number_list(args.sample_steps_list, int),
            seed=args.seed,
            eval_seeds=(_parse_number_list(args.eval_seeds, int) if args.eval_seeds else None),
            roundtrip_samples=args.roundtrip_samples,
        )
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
    ae, flow, conditioner, prompt_vocab, report = train_latent_flow(
        ae_steps=args.ae_steps, flow_steps=args.flow_steps, batch=args.batch,
        latent_ch=args.latent_ch, hidden=args.hidden, lr=args.lr, fact_w=args.fact_w,
        seed=args.seed, flow_arch=args.flow_arch, dit_depth=args.dit_depth,
        dit_heads=args.dit_heads, cond_drop=args.cond_drop, cfg_scale=args.cfg_scale,
        sample_steps=args.sample_steps, roundtrip_samples=args.roundtrip_samples,
        flow_semantic_w=args.flow_semantic_w, cond_mode=args.cond_mode,
        text_cond_dim=args.text_cond_dim, prompt_templates=templates,
        time_sampling=args.time_sampling, time_logit_mean=args.time_logit_mean,
        time_logit_std=args.time_logit_std,
        return_conditioner=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "autoencoder_state_dict": ae.state_dict(),
        "flow_state_dict": flow.state_dict(),
        "report": report,
        "fact_vocab": FACT_VOCAB,
        "latent_ch": args.latent_ch,
        "hidden": args.hidden,
        "flow_arch": args.flow_arch,
        "dit_depth": args.dit_depth,
        "dit_heads": args.dit_heads,
        "cond_mode": args.cond_mode,
        "cond_dim": report["cond_dim"],
        "cond_drop": args.cond_drop,
        "cfg_scale": args.cfg_scale,
        "sample_steps": args.sample_steps,
        "flow_semantic_w": args.flow_semantic_w,
        "time_sampling": args.time_sampling,
        "time_logit_mean": args.time_logit_mean,
        "time_logit_std": args.time_logit_std,
        "prompt_templates": list(templates) if args.cond_mode == "text" else [],
        "prompt_vocab": prompt_vocab if prompt_vocab is not None else {},
        "conditioner_state_dict": (conditioner.state_dict() if conditioner is not None else {}),
    }, args.out)
    print(json.dumps(report, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
