#!/usr/bin/env python3
"""Cracking the length-generalization wall: a LOOPED, weight-tied reasoner for the multi-hop walk.

The autoregressive walk (thinking/langchain.py) is a TRAIN-DEPTH WALL -- perfect to the trained depth, then a
cliff (depth 5 -> 0.997, depth 6 -> 0.04). The literature converges on the fix: a single weight-tied block
applied T(n) times with INPUT INJECTION length-generalizes algorithmic/iterated tasks where fixed-depth
transformers fail (Looped Transformers for Length Generalization, arXiv:2409.15647; matches our own
computation-lengthgen finding that input injection is decisive). Here each LOOP = one hop, so a deeper chain just
needs MORE loops -- the per-hop step is reused, not re-learned per depth.

Two design choices attack both walls at once:
  * loops = hops + input injection  -> length-generalization (frontier A).
  * NoPE + ROLE embeddings (child/parent/query), permutation-invariant over the edge set, direction by role not
    position  -> binding by content/role, not position (the "form rules naturally, not by word position" ask).

Task: facts = the chain's (child, parent) edges, SHUFFLED; query = a1; answer = the top ancestor aN (the
multi-hop conclusion). Train on shallow chains (depth 3-5, T=2-4 loops); test DEEPER (depth up to 15, more loops).
A flat accuracy curve past the trained depth = the wall is cracked.

  python -m thinking.langloop --selftest
  python -m thinking.langloop --steps 8000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.reasoner import Block

NSYM = 60


class LoopWalk(nn.Module):
    """Encode the shuffled edge set + query once (NoPE, role-tagged); iterate ONE tied block T times with input
    injection; read the conclusion off a learned readout slot. T loops resolve a T-hop chain."""
    def __init__(self, nsym=NSYM, d=128, h=4):
        super().__init__()
        self.val = nn.Embedding(nsym, d)
        self.role = nn.Embedding(3, d)                          # 0=child, 1=parent, 2=query
        self.readout = nn.Parameter(torch.zeros(1, 1, d))
        self.block = Block(d, h)                                # ONE tied block, reused every loop
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, nsym))

    def encode(self, edges, query):
        B, E, _ = edges.shape
        c = self.val(edges[:, :, 0]) + self.role.weight[0]     # child slot
        p = self.val(edges[:, :, 1]) + self.role.weight[1]     # parent slot
        ep = torch.stack([c, p], 2).reshape(B, E * 2, -1)      # [c1 p1 c2 p2 ...]  (NoPE: order carries no info)
        q = (self.val(query) + self.role.weight[2]).unsqueeze(1)
        r = self.readout.expand(B, -1, -1)                     # learned readout slot (holds the advancing node)
        return torch.cat([ep, q, r], 1)

    def forward(self, edges, query, T, all_steps=False):
        x = self.encode(edges, query)
        h = torch.zeros_like(x)
        outs = []
        for _ in range(max(1, T)):
            h = self.block(h + x, None)                         # INPUT INJECTION + weight-tied recurrence
            if all_steps:
                outs.append(self.head(h[:, -1]))               # node reached after this hop (per-step supervision)
        return outs if all_steps else self.head(h[:, -1])      # readout slot -> the conclusion aN


def gen(rng, depth, n_dist=0):
    """A random chain a1->..->aN as SHUFFLED (child,parent) edges + n_dist DISTRACTOR edges (sources are non-chain
    entities, so every chain node keeps a unique parent). Distractors lengthen the CONTEXT independently of chain
    depth, so the per-hop lookup is trained over long edge sets. Query a1. Returns (edges, full chain)."""
    perm = rng.permutation(NSYM).tolist()
    chain = perm[:depth]
    rest = perm[depth:]                                          # non-chain entities (distractor sources)
    edges = [[chain[i], chain[i + 1]] for i in range(depth - 1)]
    for _ in range(n_dist):
        src = rest[int(rng.integers(len(rest)))]
        tgt = perm[int(rng.integers(NSYM))]
        edges.append([src, tgt])
    rng.shuffle(edges)
    return edges, chain


def _batch(rng, bs, depth, n_dist, device):
    eps = [gen(rng, depth, n_dist) for _ in range(bs)]
    edges = torch.tensor([e for e, _ in eps], device=device)
    chain = torch.tensor([c for _, c in eps], device=device)    # (B, depth): chain[:,0]=query a1 ... chain[:,-1]=aN
    return edges, chain[:, 0], chain


def train(steps, lo=3, hi=8, d=128, h=4, bs=64, lr=1e-3, seed=0, device="cpu", curric=True, dist_max=8):
    """Depth-recurrent training (arXiv:2603.21676) + broad context sampling (arXiv:2402.09371): PER-HOP
    intermediate supervision (each loop predicts the next node -> kills error compounding) + a CURRICULUM that
    grows max depth + DISTRACTOR edges sampled broadly so the lookup is trained on long contexts."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = LoopWalk(NSYM, d=d, h=h).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    warmup = steps // 2
    for s in range(steps):
        cur_hi = min(hi, lo + int((s / max(1, warmup)) * (hi - lo))) if curric else hi
        depth = int(rng.integers(lo, cur_hi + 1))              # curriculum: shallow first, grow to hi
        n_dist = int(rng.integers(0, dist_max + 1))            # broad context length, decoupled from depth
        edges, query, chain = _batch(rng, bs, depth, n_dist, device)
        outs = m(edges, query, T=depth - 1, all_steps=True)    # outs[t] = node after (t+1) hops
        loss = sum(F.cross_entropy(outs[t], chain[:, t + 1]) for t in range(depth - 1)) / (depth - 1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def eval_depth(m, depth, n=300, seed=999, device="cpu", n_dist=0):
    rng = np.random.default_rng(seed + depth); m.eval(); ok = 0
    with torch.no_grad():
        for _ in range(0, n, 100):
            b = min(100, n - _)
            edges, query, chain = _batch(rng, b, depth, n_dist, device)
            pred = m(edges, query, T=depth - 1).argmax(-1)
            ok += int((pred == chain[:, -1]).sum())            # reached the correct conclusion aN
    return ok / n


def self_train(d=128, h=4, device="cpu", lo=3, start_hi=6, target=32, stage_steps=1200,
               promote_thresh=0.85, dist_max=6, seed=0, verbose=True):
    """RECURSIVE SELF-TRAINING / expanding-horizon ratchet (arXiv:2511.07378): train at depths [lo, cur_hi]; once
    the model already EXTRAPOLATES to cur_hi+1 above threshold (the reusable per-hop step makes one-step
    extrapolation cheap), promote cur_hi -> cur_hi+1 and consolidate it. The horizon ratchets up one hop at a time,
    so it can climb far past a fixed curriculum. Returns (model, reached_depth)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = LoopWalk(NSYM, d=d, h=h).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    cur_hi = start_hi
    stage = 0
    while cur_hi < target and stage < 4 * (target - start_hi) + 20:
        for _ in range(stage_steps):
            depth = int(rng.integers(lo, cur_hi + 1))
            n_dist = int(rng.integers(0, dist_max + 1))
            edges, query, chain = _batch(rng, 64, depth, n_dist, device)
            outs = m(edges, query, T=depth - 1, all_steps=True)
            loss = sum(F.cross_entropy(outs[t], chain[:, t + 1]) for t in range(depth - 1)) / (depth - 1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        nxt = eval_depth(m, cur_hi + 1, n=200, device=device)            # can it already do one deeper?
        if verbose:
            print(f"  stage {stage:2d}: horizon {cur_hi:2d} -> probe d{cur_hi+1}: {nxt:.3f}"
                  f"{'  PROMOTE' if nxt >= promote_thresh else ''}", flush=True)
        if nxt >= promote_thresh:
            cur_hi += 1                                                   # ratchet the horizon up by one
        stage += 1
    return m, cur_hi


def lengthgen(steps=10000, lo=3, hi=8, test_max=30, d=128, h=4, device="cpu"):
    m = train(steps, lo, hi, d=d, h=h, device=device)
    print(f"depth-recurrent walk: trained depth {lo}-{hi} (curriculum + per-hop supervision), reach-conclusion "
          f"vs depth:")
    for depth in range(lo, test_max + 1):
        tag = "  (train)" if depth <= hi else "  (extrapolation)"
        print(f"  depth {depth:2d}: {eval_depth(m, depth, device=device):.3f}{tag}")
    return m


def selftest():
    m = train(14000, lo=3, hi=8, d=128, h=4, device="cpu", dist_max=8)
    d8 = eval_depth(m, 8, device="cpu")                          # trained max
    d20 = eval_depth(m, 20, device="cpu")                        # 2.5x
    d30 = eval_depth(m, 30, device="cpu")                        # ~4x -- the autoregressive walk is ~0.04 here
    print(f"langloop selftest: depth8(train) {d8:.3f} | depth20 {d20:.3f} | depth30 {d30:.3f} "
          f"(autoregressive walk cliffs to ~0.04 past train depth)")
    assert d8 > 0.95, f"failed in-train depth: {d8}"
    assert d20 > 0.75, f"did NOT length-generalize to 2.5x depth (depth20 {d20:.3f})"
    print("langloop selftest OK (per-hop supervision + curriculum length-generalize to ~2.5x depth; "
          "4x+ needs scale -- depth30 still degrades on tiny CPU)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftrain", action="store_true", help="recursive self-training (expanding-horizon ratchet)")
    ap.add_argument("--target", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--test-max", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.selftrain:
        m, reached = self_train(d=a.d, h=a.heads, device=a.device, target=a.target)
        print(f"self-training reached horizon depth {reached}; final length-gen curve:")
        for depth in range(5, a.test_max + 1, 3):
            print(f"  depth {depth:2d}: {eval_depth(m, depth, device=a.device):.3f}")
        return 0
    lengthgen(steps=a.steps, test_max=a.test_max, d=a.d, h=a.heads, device=a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
