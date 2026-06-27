#!/usr/bin/env python3
"""Length-generalization over TEXT-LIKE input: variable-length, punctuation-delimited sentences (NOT one token per
reasoning step), with lengths drawn from a NORMAL distribution -- the way real language is structured.

Motivation (user insight): text does not map one-to-one; word/sentence length varies with punctuation on a roughly
normal distribution. So the reasoning UNIT should be "one sentence (found by punctuation)", invariant to how many
tokens that sentence happens to contain. Then the per-hop step is length-invariant and the walk length-generalizes
over naturally-varying text. This folds together everything learned:
  * one loop = one hop = one SENTENCE (depth-recurrent + input injection, arXiv:2603.21676) -> unbounded hops.
  * content vs FUNCTION words distinguished by RECURRENCE (function words repeat across sentences; entities are
    unique) -> no hardcoded word lists; the marker/role idea generalized to real text (langrole).
  * LOCAL position within a sentence (resets at each PERIOD) -> child-before-parent order is readable regardless
    of sentence length, and nothing goes out of positional distribution as text gets longer (langcap).
  * sentence length ~ NORMAL distribution in training -> the model sees the natural length spread incl. the tail,
    so it handles longer sentences at test (broad-sampling lever, arXiv:2402.09371).
  * PER-HOP supervision -> each loop emits the next node, killing error compounding.

Task: a chain a1->..->aN; each isa edge (child,parent) is rendered as a variable-length sentence -- child and
parent (unique content words) placed in order among a random number of recurring FUNCTION words, ended by a
PERIOD. Sentences shuffled. Query a1 -> walk to aN. Test deeper (more sentences) AND longer sentences (tail).

  python -m thinking.langtext --selftest
  python -m thinking.langtext --steps 14000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.reasoner import Block

PAD, PERIOD, QSEP = 0, 1, 2
NFUNC = 8                                                   # function-word vocabulary (recurring, content-free)
F0 = 3
NSYM = 60                                                   # content entities
C0 = F0 + NFUNC
VOCAB = C0 + NSYM
SMAX = 12                                                   # max sentence length (slots) -> local-position table size
NPOS = SMAX + 3                                             # local positions: 0..SMAX-1 within sentence, PERIOD, query, readout


def _sentence(rng, child, parent, mu, sigma):
    """Render one edge as a variable-length sentence: child before parent among random function words, length ~
    Normal(mu,sigma) clamped to [2,SMAX]. Returns (tokens, local_position_ids) WITHOUT the trailing period."""
    L = int(round(rng.normal(mu, sigma))); L = max(2, min(SMAX, L))
    slots = sorted(rng.choice(L, 2, replace=False).tolist())   # two content slots, child slot < parent slot
    toks = []
    for j in range(L):
        if j == slots[0]:
            toks.append(child)
        elif j == slots[1]:
            toks.append(parent)
        else:
            toks.append(F0 + int(rng.integers(NFUNC)))         # a function word (recurring, non-content)
    pos = list(range(L))
    return toks, pos


def gen(rng, depth, mu=5.0, sigma=2.0):
    """Chain a1->..->aN; each edge -> a punctuation-delimited variable-length sentence; sentences shuffled; then
    the query. Returns (token_ids, local_position_ids, full chain)."""
    perm = (rng.permutation(NSYM) + C0).tolist()
    chain = perm[:depth]
    edges = [(chain[i], chain[i + 1]) for i in range(depth - 1)]
    rng.shuffle(edges)
    seq, pos = [], []
    for c, p in edges:
        t, q = _sentence(rng, c, p, mu, sigma)
        seq += t + [PERIOD]; pos += q + [SMAX]                 # PERIOD = the sentence boundary anchor
    seq += [QSEP, chain[0]]; pos += [SMAX + 1, SMAX + 1]        # query
    return seq, pos, chain


def _batch(rng, bs, depth, mu, sigma, device):
    eps = [gen(rng, depth, mu, sigma) for _ in range(bs)]
    L = max(len(s) for s, _, _ in eps)
    ids = torch.full((bs, L), PAD, dtype=torch.long)
    pos = torch.full((bs, L), SMAX, dtype=torch.long)
    chain = torch.tensor([c for _, _, c in eps], dtype=torch.long)   # (B, depth)
    for i, (s, p, _) in enumerate(eps):
        ids[i, :len(s)] = torch.tensor(s); pos[i, :len(p)] = torch.tensor(p)
    return ids.to(device), pos.to(device), chain.to(device)


class TextWalk(nn.Module):
    """Looped reasoner over text: read the (padded) token+local-position sequence, append a readout slot, iterate a
    tied block T times with input injection; one loop = one hop = one sentence."""
    def __init__(self, d=160, h=4):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.pos = nn.Embedding(NPOS, d)
        self.readout = nn.Parameter(torch.zeros(1, 1, d))
        self.block = Block(d, h)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, VOCAB))

    def forward(self, ids, pos, T, all_steps=False):
        B = ids.shape[0]
        kpm = torch.cat([ids == PAD, torch.zeros(B, 1, dtype=torch.bool, device=ids.device)], 1)
        e = self.tok(ids) + self.pos(pos)
        r = self.readout.expand(B, -1, -1) + self.pos.weight[SMAX + 2]
        x = torch.cat([e, r], 1)
        h = torch.zeros_like(x)
        outs = []
        for _ in range(max(1, T)):
            h = self.block(h + x, kpm)                          # INPUT INJECTION + weight-tied recurrence
            if all_steps:
                outs.append(self.head(h[:, -1]))
        return outs if all_steps else self.head(h[:, -1])


def train(steps, lo=3, hi=8, mu=5.0, sigma=2.0, d=160, h=4, bs=48, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = TextWalk(d=d, h=h).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    warmup = steps // 2
    for s in range(steps):
        cur_hi = min(hi, lo + int((s / max(1, warmup)) * (hi - lo)))     # depth curriculum
        depth = int(rng.integers(lo, cur_hi + 1))
        ids, pos, chain = _batch(rng, bs, depth, mu, sigma, device)
        outs = m(ids, pos, T=depth - 1, all_steps=True)
        loss = sum(F.cross_entropy(outs[t], chain[:, t + 1]) for t in range(depth - 1)) / (depth - 1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def evaluate(m, depth, mu=5.0, sigma=2.0, n=300, seed=999, device="cpu"):
    rng = np.random.default_rng(seed + depth + int(mu)); m.eval(); ok = 0
    with torch.no_grad():
        for _ in range(0, n, 100):
            b = min(100, n - _)
            ids, pos, chain = _batch(rng, b, depth, mu, sigma, device)
            pred = m(ids, pos, T=depth - 1).argmax(-1)
            ok += int((pred == chain[:, -1]).sum())
    return ok / n


def self_train(d=160, h=4, device="cpu", lo=3, start_hi=6, target=22, stage_steps=1000,
               mu_lo=4.0, mu_hi=9.0, sigma=2.0, promote_thresh=0.83, seed=0, verbose=True):
    """Recursive self-training over TEXT: expanding-horizon ratchet (arXiv:2511.07378) on the variable-length,
    punctuation-delimited sentence task. Combines BOTH length axes: each hop is a variable-length sentence AND the
    depth horizon ratchets up. CRUCIAL for composition: sentence length is sampled BROADLY (mu ~ U[mu_lo, mu_hi])
    at EVERY stage, so each depth is consolidated across varied sentence lengths -> deep chains and long sentences
    generalize TOGETHER (a fixed sentence length makes long-sentence per-hop error compound over many hops).
    Returns (model, reached_depth)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = TextWalk(d=d, h=h).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    cur_hi = start_hi; stage = 0
    while cur_hi < target and stage < 4 * (target - start_hi) + 25:
        for _ in range(stage_steps):
            depth = int(rng.integers(lo, cur_hi + 1))
            bmu = float(rng.uniform(mu_lo, mu_hi))             # broad sentence-length sampling, decoupled from depth
            ids, pos, chain = _batch(rng, 48, depth, bmu, sigma, device)
            outs = m(ids, pos, T=depth - 1, all_steps=True)
            loss = sum(F.cross_entropy(outs[t], chain[:, t + 1]) for t in range(depth - 1)) / (depth - 1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        nxt = evaluate(m, cur_hi + 1, mu=mu_hi, sigma=sigma, n=200, device=device)   # probe at the LONG end
        if verbose:
            print(f"  stage {stage:2d}: horizon {cur_hi:2d} -> probe d{cur_hi+1}(mu{mu_hi:.0f}): {nxt:.3f}"
                  f"{'  PROMOTE' if nxt >= promote_thresh else ''}", flush=True)
        if nxt >= promote_thresh:
            cur_hi += 1
        stage += 1
    return m, cur_hi


def selftest():
    m = train(13000, lo=3, hi=8, mu=5.0, sigma=2.0, device="cpu")
    seen = evaluate(m, 8, device="cpu")                          # trained depth, trained sentence-length dist
    deep = evaluate(m, 18, device="cpu")                         # more sentences (2.25x depth)
    longs = evaluate(m, 8, mu=9.0, sigma=2.0, device="cpu")      # longer sentences (distribution tail), trained depth
    print(f"langtext selftest: trained(d8) {seen:.3f} | deeper(d18) {deep:.3f} | longer-sentences(mu9,d8) {longs:.3f}")
    assert seen > 0.9, f"failed in-distribution: {seen}"
    # The contribution: SENTENCE-LENGTH generalization (surface length of a unit) is solved -- punctuation makes the
    # 'one sentence = one hop' unit invariant to its token count, so longer sentences transfer. DEPTH (number of
    # reasoning steps) is a SEPARATE axis and remains the hard iterated-reasoning residual (partial here).
    assert longs > 0.85, f"did not generalize to LONGER sentences (the surface-length axis): {longs}"
    assert deep > 0.45, f"depth (reasoning-step-count) generalization regressed below the known partial: {deep}"
    print("langtext selftest OK (variable-length punctuation-delimited text: LONGER sentences generalize strongly "
          "via the one-hop-per-sentence unit; deeper chains are the separate, residual step-count axis)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftrain", action="store_true", help="recursive self-training over text (both length axes)")
    ap.add_argument("--target", type=int, default=22)
    ap.add_argument("--steps", type=int, default=14000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.selftrain:
        m, reached = self_train(device=a.device, target=a.target)
        print(f"self-training over text reached horizon depth {reached}. BOTH axes together (deep x long):")
        for depth in (5, 10, 15, reached):
            print(f"  depth {depth:2d}, sentences mu5 : {evaluate(m, depth, device=a.device):.3f}")
        for mu in (5, 7, 9):
            print(f"  depth {reached}, sentences mu{mu} (deep + long): "
                  f"{evaluate(m, reached, mu=float(mu), device=a.device):.3f}")
        print(f"  depth {reached}, sentences mu11 (deep + EXTRAPOLATED length): "
              f"{evaluate(m, reached, mu=11.0, device=a.device):.3f}")
        return 0
    m = train(a.steps, device=a.device)
    print("text length-gen (train depth 3-8, sentence length ~ Normal(5,2)):")
    for depth in (5, 8, 12, 16, 20):
        print(f"  depth {depth:2d} (mu5): {evaluate(m, depth, device=a.device):.3f}")
    for mu in (5, 7, 9, 11):
        print(f"  depth 8  (mu{mu:2d}): {evaluate(m, 8, mu=float(mu), device=a.device):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
