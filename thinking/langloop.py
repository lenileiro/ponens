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

    def forward(self, edges, query, T):
        x = self.encode(edges, query)
        h = torch.zeros_like(x)
        for _ in range(max(1, T)):
            h = self.block(h + x, None)                         # INPUT INJECTION + weight-tied recurrence
        return self.head(h[:, -1])                              # readout slot -> the conclusion aN


def gen(rng, depth):
    """A random chain a1->..->aN as SHUFFLED (child,parent) edges; query a1; answer = top aN."""
    chain = (rng.permutation(NSYM)[:depth]).tolist()
    edges = [[chain[i], chain[i + 1]] for i in range(depth - 1)]
    rng.shuffle(edges)
    return edges, chain[0], chain[-1]


def _batch(rng, bs, depth, device):
    eps = [gen(rng, depth) for _ in range(bs)]
    edges = torch.tensor([e for e, _, _ in eps], device=device)
    query = torch.tensor([q for _, q, _ in eps], device=device)
    ans = torch.tensor([a for _, _, a in eps], device=device)
    return edges, query, ans


def train(steps, lo=3, hi=5, d=128, h=4, bs=64, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = LoopWalk(NSYM, d=d, h=h).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        depth = int(rng.integers(lo, hi + 1))                  # variable depth => variable T => reusable hop
        edges, query, ans = _batch(rng, bs, depth, device)
        logits = m(edges, query, T=depth - 1)                  # T = number of hops
        loss = F.cross_entropy(logits, ans)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def eval_depth(m, depth, n=300, seed=999, device="cpu", T=None):
    rng = np.random.default_rng(seed + depth); m.eval(); ok = 0
    with torch.no_grad():
        for _ in range(0, n, 100):
            b = min(100, n - _)
            edges, query, ans = _batch(rng, b, depth, device)
            pred = m(edges, query, T=(depth - 1 if T is None else T)).argmax(-1)
            ok += int((pred == ans).sum())
    return ok / n


def lengthgen(steps=8000, lo=3, hi=5, test_max=15, d=128, h=4, device="cpu"):
    m = train(steps, lo, hi, d=d, h=h, device=device)
    print(f"looped walk: trained depth {lo}-{hi} (T={lo-1}-{hi-1} loops), reach-conclusion vs depth "
          f"(T=depth-1 loops at test):")
    for depth in range(lo, test_max + 1):
        tag = "  (train)" if depth <= hi else "  (extrapolation)"
        print(f"  depth {depth:2d}: {eval_depth(m, depth, device=device):.3f}{tag}")
    return m


def selftest():
    m = train(5000, lo=3, hi=5, d=128, h=4, device="cpu")
    d5 = eval_depth(m, 5, device="cpu")                          # trained
    d10 = eval_depth(m, 10, device="cpu")                        # 2x deepest trained
    d14 = eval_depth(m, 14, device="cpu")                        # ~3x
    print(f"langloop selftest: depth5(train) {d5:.3f} | depth10 {d10:.3f} | depth14 {d14:.3f} "
          f"(autoregressive walk cliffs to ~0.04 past train depth)")
    assert d5 > 0.9, f"failed in-train depth: {d5}"
    assert d10 > 0.6, f"did NOT length-generalize (depth10 {d10:.3f}); the looped fix did not crack the wall"
    print("langloop selftest OK (looped block + input injection length-generalizes the multi-hop walk)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--test-max", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    lengthgen(steps=a.steps, test_max=a.test_max, d=a.d, h=a.heads, device=a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
