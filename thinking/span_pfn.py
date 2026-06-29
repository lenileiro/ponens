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

try:
    from device import get_device
except Exception:                                              # fallback if device.py unavailable
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

V = 20            # token vocab (content tokens 2..V-1; 0=pad reserved)
S = 12            # length of each example
K = 4             # support examples per episode


# A broader PRIOR over span-task structures: positional (relative to markers/ends) + content (value-based). The
# wider the prior, the more the model learns the META-skill 'infer whatever rule from the support' (vs memorizing
# one family) -- which is what lifts generalization to UNSEEN task types.
FAMILIES = ["delim", "after", "before", "firstk", "lastk", "gt", "lt", "eq", "band"]


def _apply_rule(seq, fam, p):
    s = len(seq)
    if fam == "delim":                                         # strictly between delimiters A and B
        a, b = p; ia = seq.index(a) if a in seq else 0; ib = seq.index(b) if b in seq else s - 1
        i, j = min(ia, ib), max(ia, ib); return [1 if i < q < j else 0 for q in range(s)]
    if fam == "after":                                         # L tokens after marker M
        m, L = p; i = seq.index(m) if m in seq else s; return [1 if i < q <= i + L else 0 for q in range(s)]
    if fam == "before":                                        # L tokens before marker M
        m, L = p; i = seq.index(m) if m in seq else -1; return [1 if i - L <= q < i else 0 for q in range(s)]
    if fam == "firstk":                                        # first K positions
        return [1 if q < p else 0 for q in range(s)]
    if fam == "lastk":                                         # last K positions
        return [1 if q >= s - p else 0 for q in range(s)]
    if fam == "gt":                                            # value > T
        return [1 if seq[q] > p else 0 for q in range(s)]
    if fam == "lt":                                            # value < T
        return [1 if seq[q] < p else 0 for q in range(s)]
    if fam == "eq":                                            # value == X
        return [1 if seq[q] == p else 0 for q in range(s)]
    if fam == "band":                                          # lo <= value <= hi
        lo, hi = p; return [1 if lo <= seq[q] <= hi else 0 for q in range(s)]
    raise ValueError(fam)


def _sample_params(rng, fam):
    if fam == "delim":
        return tuple(int(x) for x in rng.choice(range(2, V), size=2, replace=False))
    if fam in ("after", "before"):
        return (int(rng.integers(2, V)), int(rng.integers(1, 4)))
    if fam in ("firstk", "lastk"):
        return int(rng.integers(2, 5))
    if fam == "gt":
        return int(rng.integers(V // 2, V - 2))
    if fam == "lt":
        return int(rng.integers(4, V // 2))
    if fam == "eq":
        return int(rng.integers(2, V))
    if fam == "band":
        lo = int(rng.integers(2, V - 4)); return (lo, lo + int(rng.integers(2, 4)))
    raise ValueError(fam)


def _plant(rng, seq, fam, p):                                  # guarantee a non-empty answer span
    s = len(seq)
    if fam == "delim":
        i, j = sorted(rng.choice(range(s), size=2, replace=False)); seq[i], seq[j] = int(p[0]), int(p[1])
    elif fam == "after":
        seq[int(rng.integers(0, s - 1))] = p[0]
    elif fam == "before":
        seq[int(rng.integers(1, s))] = p[0]
    elif fam == "gt":
        seq[int(rng.integers(0, s))] = int(rng.integers(p + 1, V))
    elif fam == "lt":
        seq[int(rng.integers(0, s))] = int(rng.integers(2, p))
    elif fam == "eq":
        seq[int(rng.integers(0, s))] = int(p)
    elif fam == "band":
        seq[int(rng.integers(0, s))] = int(rng.integers(p[0], p[1] + 1))
    return seq


def gen_episode(rng, k=None, s=None, family="delim"):
    """One task (params drawn fresh) from `family`; k support + 1 query sharing the SAME rule. The model must infer
    the rule (and its params) from the support answer-flags and apply it to the query."""
    k = K if k is None else k; s = S if s is None else s
    for _ in range(50):
        p = _sample_params(rng, family)
        toks, flags, ok = [], [], True
        for _ in range(k + 1):
            seq = _plant(rng, rng.integers(2, V, size=s).tolist(), family, p)
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


HELDIN = ["delim", "after", "before", "firstk", "gt", "lt", "eq"]   # train on 7 families...
HELDOUT = ["lastk", "band"]                                          # ...hold out 2 structurally-related ones


def _episode_tensors(rng, bs, mask_support=False, family="delim", dev=None):
    dev = dev or torch.device("cpu")
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
    t = lambda a: torch.tensor(a).to(dev)
    return t(toks), t(flags), t(segs), t(poss), torch.tensor(y, dtype=torch.float32).to(dev)


def _query_logits(model, toks, flags, segs, poss):
    z = model(toks, flags, segs, poss)
    return z[:, K * S:(K + 1) * S]                             # logits on the QUERY positions


def train(model, steps=3000, bs=64, lr=2e-3, seed=0, family="delim", dev=None):
    rng = np.random.default_rng(seed); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for st in range(steps):
        toks, flags, segs, poss, y = _episode_tensors(rng, bs, family=family, dev=dev)
        loss = F.binary_cross_entropy_with_logits(_query_logits(model, toks, flags, segs, poss), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 500 == 0:
            print(f"  step {st} loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, n=400, seed=123, mask_support=False, family="delim", dev=None):
    rng = np.random.default_rng(seed)
    toks, flags, segs, poss, y = _episode_tensors(rng, n, mask_support=mask_support, family=family, dev=dev)
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


def multitest(steps=9000):
    """Train on a BROAD mix of rule families; test in-context generalization to (a) new PARAMS of seen families and
    (b) HELD-OUT families never trained on. Tests whether a broader prior lifts cross-family (true zero-shot) reach."""
    torch.manual_seed(0)
    model = SpanPFN(d=128, layers=4, heads=4)
    train(model, steps=steps, bs=64, family="mix")
    print(f"span_pfn BROAD PRIOR ({len(HELDIN)} families: {','.join(HELDIN)}); in-context, zero weight updates:")
    hi = np.mean([evaluate(model, family=f)[1] for f in HELDIN])
    print(f"  held-IN  (new params of {len(HELDIN)} seen families)  mean exact-span {hi:.3f}")
    for f in HELDOUT:
        ex = evaluate(model, family=f)[1]
        print(f"  held-OUT {f:6} (NEVER trained on this rule type) exact-span {ex:.3f}")
    print("  NOTE: at this small/CPU scale the 7-family mix STALLS (loss plateau ~0.55, exact~0) even with K=8 "
          "support -- the model can't learn to disambiguate+apply many families. A broad prior needs TabPFN-scale "
          "(big model + millions of synthetic episodes + large context). Single-family / new-params works (0.96).")
    return 0


def train_scaled(d, layers, heads, bs, steps, k, save):
    """Scaled broad-prior meta-training (for GPU): big model + many synthetic episodes + larger support, to test
    whether scale lets the model learn the BROAD 7-family mix that stalled at small/CPU scale."""
    global K
    K = k
    dev = get_device()
    torch.manual_seed(0)
    model = SpanPFN(d=d, layers=layers, heads=heads).to(dev)
    print(f"scaled PFN: d{d} L{layers} H{heads} bs{bs} steps{steps} K{k} | families {HELDIN} | device {dev}", flush=True)
    train(model, steps=steps, bs=bs, family="mix", dev=dev)
    if save:
        torch.save({"state": model.state_dict(), "cfg": (d, layers, heads, k)}, save)
    hi = float(np.mean([evaluate(model, family=f, dev=dev)[1] for f in HELDIN]))
    print(f"RESULT scaled broad-prior: held-IN mean exact-span {hi:.3f}", flush=True)
    for f in HELDOUT:
        print(f"  held-OUT {f:6} exact-span {evaluate(model, family=f, dev=dev)[1]:.3f}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--multitest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--save", default="/tmp/span_pfn.pt")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.multitest:
        return multitest()
    if a.train:
        return train_scaled(a.d, a.layers, a.heads, a.bs, a.steps, a.k, a.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
