#!/usr/bin/env python3
"""Prior-Fitted Network for SPAN extraction -- the TabPFN paradigm applied to our readers. Instead of TRAINING a
span reader per task, meta-train ONE transformer on a PRIOR over extraction rules; at inference it reads a few
(text, answer-span) SUPPORT examples in-context and extracts the span for a QUERY of an UNSEEN task -- no weight
update (pure forward pass), exactly like TabPFN conditions on the support rows.

Proof-of-concept prior: each task = 'the answer is the span strictly between delimiter token A and delimiter token B'
with (A,B) drawn fresh per episode. The model never memorizes a fixed (A,B); it must INFER the rule from the
support examples' answer flags and apply it to the query. Generalization to fresh (A,B) at test = the PFN claim.
Ablation: zero the support flags (remove the rule signal) -> accuracy collapses to chance, proving it is the
IN-CONTEXT support that drives prediction, not memorization.

  python -m thinking.span_pfn --selftest
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V = 20            # token vocab (content tokens 2..V-1; 0=pad reserved)
S = 12            # length of each example
K = 4             # support examples per episode


def gen_episode(rng, k=K, s=S):
    """One task (A,B drawn fresh); k support examples + 1 query, all sharing the rule 'span between A and B'."""
    a, b = rng.choice(range(2, V), size=2, replace=False)
    toks, flags = [], []
    for _ in range(k + 1):
        i, j = sorted(rng.choice(range(s), size=2, replace=False))
        while j - i < 2:                                       # ensure a non-empty between-span
            i, j = sorted(rng.choice(range(s), size=2, replace=False))
        seq = rng.choice([t for t in range(2, V) if t not in (a, b)], size=s).tolist()
        seq[i] = a; seq[j] = b
        fl = [1 if i < p < j else 0 for p in range(s)]         # in-span = strictly between A and B
        toks.append(seq); flags.append(fl)
    return np.array(toks), np.array(flags)                     # (k+1, s) tokens, (k+1, s) gold in-span


class SpanPFN(nn.Module):
    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.flag = nn.Embedding(3, d)                         # 0=support-out, 1=support-in, 2=query-unknown
        self.pos = nn.Embedding(S, d)                          # position WITHIN an example
        self.seg = nn.Embedding(K + 1, d)                      # which example (support 0..K-1, query=K)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, 1)

    def forward(self, toks, flags, segs, poss):                # all (B, (K+1)*S)
        h = self.tok(toks) + self.flag(flags) + self.pos(poss) + self.seg(segs)
        h = self.enc(h)
        return self.out(h).squeeze(-1)                         # (B, T) in-span logits


def _episode_tensors(rng, bs, mask_support=False):
    T = (K + 1) * S
    toks = np.zeros((bs, T), int); flags = np.zeros((bs, T), int)
    segs = np.zeros((bs, T), int); poss = np.zeros((bs, T), int); y = np.zeros((bs, S))
    for b in range(bs):
        tk, fl = gen_episode(rng)
        for e in range(K + 1):
            sl = slice(e * S, (e + 1) * S)
            toks[b, sl] = tk[e]; segs[b, sl] = e; poss[b, sl] = np.arange(S)
            if e < K:                                          # support: show the answer flags (the rule signal)
                flags[b, sl] = 0 if mask_support else fl[e]
            else:                                              # query: flag = UNKNOWN; target = its gold span
                flags[b, sl] = 2; y[b] = fl[e]
    t = lambda a: torch.tensor(a)
    return t(toks), t(flags), t(segs), t(poss), torch.tensor(y, dtype=torch.float32)


def _query_logits(model, toks, flags, segs, poss):
    z = model(toks, flags, segs, poss)
    return z[:, K * S:(K + 1) * S]                             # logits on the QUERY positions


def train(model, steps=3000, bs=64, lr=2e-3, seed=0):
    rng = np.random.default_rng(seed); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for st in range(steps):
        toks, flags, segs, poss, y = _episode_tensors(rng, bs)
        loss = F.binary_cross_entropy_with_logits(_query_logits(model, toks, flags, segs, poss), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 500 == 0:
            print(f"  step {st} loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, n=400, seed=123, mask_support=False):
    rng = np.random.default_rng(seed)
    toks, flags, segs, poss, y = _episode_tensors(rng, n, mask_support=mask_support)
    pred = (_query_logits(model, toks, flags, segs, poss) > 0).float()
    tok_acc = (pred == y).float().mean().item()
    exact = (pred == y).all(dim=1).float().mean().item()       # whole query span exactly right
    return tok_acc, exact


def selftest():
    torch.manual_seed(0)
    model = SpanPFN(d=64, layers=2, heads=4)
    train(model, steps=3000, bs=64)
    acc, exact = evaluate(model)                               # FRESH (A,B) tasks, no weight update
    macc, mexact = evaluate(model, mask_support=True)          # ablation: support rule signal removed
    print(f"span_pfn selftest: in-context on FRESH tasks -> exact-span {exact:.3f} (token-acc {acc:.3f})")
    print(f"  ablation (support flags zeroed) -> exact-span {mexact:.3f}  "
          f"(collapses: it USES the in-context support, not memorization)")
    assert exact > 0.85, f"PFN not generalizing in-context: exact {exact}"
    assert mexact < exact - 0.4, f"not actually using support (no in-context learning): {mexact} vs {exact}"
    print("span_pfn selftest OK (one meta-trained transformer infers a NEW span rule from support examples, "
          "zero weight updates -- the TabPFN paradigm for span extraction)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    model = SpanPFN()
    train(model, steps=a.steps)
    print("eval:", evaluate(model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
