#!/usr/bin/env python3
"""Capstone: ONE model that learns an invented grammar from the prompt, reasons, and is robust to BOTH walls at
once -- an UNSEEN word order (frontier B) AND UNBOUNDED depth (frontier A).

It folds the two cracks together on the multi-hop isa walk in a variable-grammar invented language:
  * looped weight-tied block + INPUT INJECTION, T=depth-1 loops  -> unbounded depth (langloop).
  * marker at VARIABLE position + train on many positions         -> the position shortcut is infeasible, so the
    model learns 'marker = the recurring token' and parses an unseen order (langrole).
  * LOCAL within-fact position ids (slot 0..K-1, repeated per fact) give subject->object order WITHOUT global
    positional encoding -> direction is readable yet depth never goes out of positional distribution.

Task: a chain a1->..->aN whose isa edges are windowed facts -- a K-slot window with the recurring marker at one of
`marker_slots`, child before parent at two other slots, rest FILLER; facts shuffled. Query a1 -> answer the top aN.
Train: depth 3-5, marker slots {0,1,2,4,5,6}. Test the four corners: trained, unseen-position, deep, and BOTH.

  python -m thinking.langcap --selftest
  python -m thinking.langcap --steps 16000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.reasoner import Block

NSYM = 60
PAD, SEP, QSEP, FILL = 0, 1, 2, 3
V0 = 4
VOCAB = V0 + NSYM
K = 5                                                       # window slots per fact (tight: less FILL dilution at depth)
TRAIN_SLOTS = (0, 1, 3, 4)
HELDOUT_SLOT = 2
NPOS = K + 4                                                # local position ids: 0..K-1 slots, SEP, QSEP, query, readout


def gen(rng, depth, marker_slots):
    """A chain a1->..->aN as SHUFFLED windowed isa facts (marker at a variable slot, child before parent), then
    query a1. Returns (token_ids, local_position_ids, answer=aN)."""
    pool = (rng.permutation(NSYM) + V0).tolist()
    M = pool.pop()
    chain = pool[:depth]
    edges = [(chain[i], chain[i + 1]) for i in range(depth - 1)]
    rng.shuffle(edges)
    seq, pos = [], []
    for c, p in edges:
        m_slot = int(marker_slots[int(rng.integers(len(marker_slots)))])
        rest = [j for j in range(K) if j != m_slot]
        a, b = sorted(rng.choice(rest, 2, replace=False).tolist())     # child slot < parent slot
        win = [FILL] * K; win[m_slot] = M; win[a] = c; win[b] = p
        seq += win + [SEP]; pos += list(range(K)) + [K]
    seq += [QSEP, chain[0]]; pos += [K + 1, K + 2]                      # query (local positions, depth-invariant)
    return seq, pos, chain[-1]


def _batch(rng, bs, depth, marker_slots, device):
    eps = [gen(rng, depth, marker_slots) for _ in range(bs)]
    ids = torch.tensor([s for s, _, _ in eps], device=device)
    pos = torch.tensor([p for _, p, _ in eps], device=device)
    ans = torch.tensor([a for _, _, a in eps], device=device)
    return ids, pos, ans


class LoopCap(nn.Module):
    def __init__(self, d=128, h=4):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.pos = nn.Embedding(NPOS, d)                   # LOCAL positions only -> depth-invariant, no global OOD
        self.readout = nn.Parameter(torch.zeros(1, 1, d))
        self.block = Block(d, h)                            # ONE tied block, reused every loop
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, VOCAB))

    def forward(self, ids, pos, T):
        B = ids.shape[0]
        e = self.tok(ids) + self.pos(pos)
        r = self.readout.expand(B, -1, -1) + self.pos.weight[K + 3]
        x = torch.cat([e, r], 1)                            # readout slot appended (holds the advancing node)
        h = torch.zeros_like(x)
        for _ in range(max(1, T)):
            h = self.block(h + x, None)                     # INPUT INJECTION + weight-tied recurrence; loop = hop
        return self.head(h[:, -1])                          # readout -> the conclusion aN


def train(steps, lo=3, hi=5, marker_slots=TRAIN_SLOTS, d=128, h=4, bs=64, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = LoopCap(d=d, h=h).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        depth = int(rng.integers(lo, hi + 1))
        ids, pos, ans = _batch(rng, bs, depth, marker_slots, device)
        loss = F.cross_entropy(m(ids, pos, T=depth - 1), ans)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def evaluate(m, depth, marker_slots, n=300, seed=999, device="cpu"):
    rng = np.random.default_rng(seed + depth + marker_slots[0]); m.eval(); ok = 0
    with torch.no_grad():
        for _ in range(0, n, 100):
            b = min(100, n - _)
            ids, pos, ans = _batch(rng, b, depth, marker_slots, device)
            ok += int((m(ids, pos, T=depth - 1).argmax(-1) == ans).sum())
    return ok / n


def corners(m, device="cpu"):
    held = (HELDOUT_SLOT,)
    return {
        "trained (seen order, depth5)": evaluate(m, 5, TRAIN_SLOTS, device=device),
        "unseen order (depth5)": evaluate(m, 5, held, device=device),
        "deep (seen order, depth12)": evaluate(m, 12, TRAIN_SLOTS, device=device),
        "BOTH (unseen order + depth12)": evaluate(m, 12, held, device=device),
    }


def selftest():
    m = train(20000, device="cpu")
    c = corners(m, device="cpu")
    for k, v in c.items():
        print(f"  {k:34s}: {v:.3f}")
    assert c["trained (seen order, depth5)"] > 0.9, "failed in-distribution"
    assert c["unseen order (depth5)"] > 0.8, "did not generalize to the unseen word order"
    assert c["deep (seen order, depth12)"] > 0.8, "did not length-generalize"
    assert c["BOTH (unseen order + depth12)"] > 0.65, "did not handle unseen order + deeper depth together"
    print("langcap selftest OK (learns the grammar, reasons, robust to BOTH unseen word order AND unbounded depth)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    m = train(a.steps, device=a.device)
    print("capstone -- four robustness corners (train depth 3-5, marker slots {0,1,2,4,5,6}):")
    for k, v in corners(m, device=a.device).items():
        print(f"  {k:34s}: {v:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
