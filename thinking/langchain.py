#!/usr/bin/env python3
"""Stage 2 of learn-any-language-in-context: multi-hop REASONING by scratchpad generation.

Multi-hop reasoning = iterated single-hop lookup. Given facts that form a chain (a->b, b->c, c->d ...) in a
random-symbol language, the model is asked for a's chain and GENERATES it step by step -- each step a single
pointer-lookup (the proven stage-1 op), composed into a multi-hop walk to the top. The facts are shuffled, so
position is useless: it must look up each hop. Trained across random languages => language-agnostic chaining.

  facts (shuffled):  c d . a b . b c .      query: a   ->  generates: b c d   (a->b->c->d, the conclusion is d)

  python -m thinking.langchain --selftest
  python -m thinking.langchain --steps 7000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from scratchpad_model import ScratchpadLM

NSYM = 50
PAD, SEP, END = 0, 1, 2
VOCAB = 3 + NSYM


def gen(rng, n=None):
    """A random-language chain a1->a2->...->aN given as SHUFFLED (child,parent) pairs, then query a1 and the
    chain to generate. Returns (sequence, query_start_index, chain)."""
    n = n or int(rng.integers(3, 6))
    chain = ((rng.permutation(NSYM)[:n]) + 3).tolist()
    pairs = [[chain[i], chain[i + 1]] for i in range(n - 1)]
    rng.shuffle(pairs)
    seq = []
    for c, p in pairs:
        seq += [c, p, SEP]
    qstart = len(seq) + 1                                # index right after [a1] -> first generated token
    seq += [chain[0]] + chain[1:] + [END]               # query a1, then the chain to predict, then END
    return seq, qstart, chain


def _batch(rng, bs):
    eps = [gen(rng) for _ in range(bs)]; L = max(len(s) for s, _, _ in eps)
    x = torch.full((bs, L), PAD, dtype=torch.long)
    for i, (s, _, _) in enumerate(eps):
        x[i, :len(s)] = torch.tensor(s)
    return x


def train(steps, d=128, layers=3, heads=4, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = ScratchpadLM(VOCAB, d=d, layers=layers, heads=heads, max_len=48, pad=PAD,
                     pos_mode="rope", causal=True, pointer=True, tie=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(steps):
        x = _batch(rng, 64); out = m(x)
        loss = F.nll_loss(out[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def walk(m, seq, qstart, max_steps=8):
    """Generate the chain autoregressively from the facts + query a1 (the scratchpad reasoning)."""
    cur = seq[:qstart]; out = []
    for _ in range(max_steps):
        nxt = int(m(torch.tensor([cur]))[0, -1].argmax())
        if nxt == END:
            break
        out.append(nxt); cur.append(nxt)
    return out


def evaluate(m, n=400, seed=999):
    rng = np.random.default_rng(seed); m.eval(); top_ok = chain_ok = 0
    with torch.no_grad():
        for _ in range(n):
            seq, qstart, chain = gen(rng)
            got = walk(m, seq, qstart)
            chain_ok += (got == chain[1:])               # full correct multi-hop walk
            top_ok += (bool(got) and got[-1] == chain[-1])   # reached the correct top (conclusion)
    return top_ok / n, chain_ok / n


def selftest():
    m = train(steps=4000)
    top, full = evaluate(m, 400)
    print(f"langchain selftest: top(conclusion) {top:.3f} | full-chain {full:.3f} on unseen random languages")
    assert top > 0.9, f"too low: {top}"
    print("langchain selftest OK (multi-hop reasoning via scratchpad walk; composes single-hop lookups)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    m = train(steps=a.steps)
    top, full = evaluate(m)
    print(f"multi-hop on unseen random languages -> reached correct conclusion {top:.3f}, full chain {full:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
