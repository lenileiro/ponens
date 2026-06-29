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


FAMILIES = ["delim", "after", "gt", "eq"]                      # diverse rule TYPES (eq held out by default)


def _apply_rule(seq, fam, p):
    s = len(seq)
    if fam == "delim":                                         # tokens strictly between A and B
        a, b = p; ia = seq.index(a) if a in seq else 0; ib = seq.index(b) if b in seq else s - 1
        i, j = min(ia, ib), max(ia, ib)
        return [1 if i < q < j else 0 for q in range(s)]
    if fam == "after":                                         # the L tokens after the first marker M
        m, L = p; i = seq.index(m) if m in seq else s
        return [1 if i < q <= i + L else 0 for q in range(s)]
    if fam == "gt":                                            # all tokens with value > T
        return [1 if seq[q] > p else 0 for q in range(s)]
    if fam == "eq":                                            # all tokens equal to X
        return [1 if seq[q] == p else 0 for q in range(s)]
    raise ValueError(fam)


def gen_episode(rng, k=K, s=S, family="delim"):
    """One task (params drawn fresh) from `family`; k support + 1 query sharing the SAME rule. The model must infer
    the rule (and its params) from the support answer-flags and apply it to the query."""
    for _ in range(50):
        if family == "delim":
            p = tuple(rng.choice(range(2, V), size=2, replace=False))
        elif family == "after":
            p = (int(rng.integers(2, V)), int(rng.integers(1, 4)))
        elif family == "gt":
            p = int(rng.integers(V // 2, V - 2))
        else:  # eq
            p = int(rng.integers(2, V))
        toks, flags, ok = [], [], True
        for _ in range(k + 1):
            seq = rng.integers(2, V, size=s).tolist()
            if family == "delim":
                i, j = sorted(rng.choice(range(s), size=2, replace=False))
                seq[i], seq[j] = int(p[0]), int(p[1])
            elif family == "after":
                seq[int(rng.integers(0, s - 1))] = p[0]
            elif family == "eq":                              # plant the target value so the span is non-empty
                seq[int(rng.integers(0, s))] = int(p)
            elif family == "gt":                              # plant a token above threshold
                seq[int(rng.integers(0, s))] = int(rng.integers(p + 1, V))
            fl = _apply_rule(seq, family, p)
            if sum(fl) == 0:                                   # require a non-empty answer span
                ok = False; break
            toks.append(seq); flags.append(fl)
        if ok:
            return np.array(toks), np.array(flags)
    return np.array(toks), np.array(flags)


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


HELDIN = ["delim", "after", "gt"]; HELDOUT = "eq"              # train on 3 rule families, hold one out


def _episode_tensors(rng, bs, mask_support=False, family="delim"):
    T = (K + 1) * S
    toks = np.zeros((bs, T), int); flags = np.zeros((bs, T), int)
    segs = np.zeros((bs, T), int); poss = np.zeros((bs, T), int); y = np.zeros((bs, S))
    for b in range(bs):
        fam = rng.choice(HELDIN) if family == "mix" else family
        tk, fl = gen_episode(rng, family=fam)
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


def train(model, steps=3000, bs=64, lr=2e-3, seed=0, family="delim"):
    rng = np.random.default_rng(seed); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for st in range(steps):
        toks, flags, segs, poss, y = _episode_tensors(rng, bs, family=family)
        loss = F.binary_cross_entropy_with_logits(_query_logits(model, toks, flags, segs, poss), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 500 == 0:
            print(f"  step {st} loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, n=400, seed=123, mask_support=False, family="delim"):
    rng = np.random.default_rng(seed)
    toks, flags, segs, poss, y = _episode_tensors(rng, n, mask_support=mask_support, family=family)
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


def multitest(steps=6000):
    """Train on a MIX of rule families (delim/after/gt); test in-context generalization to (a) NEW PARAMS of seen
    families [in-distribution] and (b) a HELD-OUT family [out-of-distribution] -- the real test of 'handle any new
    span task from a few examples'."""
    torch.manual_seed(0)
    model = SpanPFN(d=96, layers=3, heads=4)
    train(model, steps=steps, bs=64, family="mix")
    print("span_pfn MULTI-FAMILY (meta-trained on delim/after/gt, in-context, zero weight updates):")
    for fam in HELDIN:
        _, ex = evaluate(model, family=fam); print(f"  held-IN  {fam:6} new-params exact-span {ex:.3f}")
    _, exo = evaluate(model, family=HELDOUT)
    print(f"  held-OUT {HELDOUT:6} (NEVER trained on this rule type) exact-span {exo:.3f}")
    print("  -> strong in-distribution = generalizes to new params of seen task types; held-out shows the prior's reach.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--multitest", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.multitest:
        return multitest()
    model = SpanPFN()
    train(model, steps=a.steps)
    print("eval:", evaluate(model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
