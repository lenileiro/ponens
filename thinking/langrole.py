#!/usr/bin/env python3
"""Phase 2 -- cracking the VARIABLE-WORD-ORDER wall (frontier B): infer a per-episode grammar in-context and
generalize to a word order NEVER trained (langgrammar's cross_grammar = 0.00 at every scale).

Minimal, fast testbed: ONE-HOP lookup in an invented language whose grammar (where the relation MARKER sits among
its two arguments) varies per episode. A fact is 3 tokens -- subject, object, and a recurring marker -- in order
g: g0=[M s o], g1=[s M o], g2=[s o M] (subject precedes object). Query a subject; answer its object. Train on
marker positions {first, last} = {g0, g2}; TEST on the never-trained {middle} = g1.

The wall: a positional reader learns "the marker is at position 0 or 2" and cannot parse g1. The fix (relational
bottleneck, Abstractor/arXiv:2309.06629): identify the marker by RECURRENCE (it is the value that repeats across
facts) via a content-only relational pass, BEFORE position -- so the rule "the recurring token is the marker; the
others are subj,obj in order" transfers to any marker position. Recurrence is LEARNED (a content-keyed attention
head), not hardcoded.

  python -m thinking.langrole --selftest
  python -m thinking.langrole --steps 12000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NSYM = 60
PAD, SEP, QSEP = 0, 1, 2
V0 = 3                                                      # first real symbol id
VOCAB = V0 + NSYM


def render(M, s, o, g):
    return [M, s, o] if g == 0 else ([s, M, o] if g == 1 else [s, o, M])


def gen(rng, grammars, nfacts=None):
    g = int(grammars[int(rng.integers(len(grammars)))])
    nfacts = nfacts or int(rng.integers(3, 6))
    pool = (rng.permutation(NSYM) + V0).tolist()
    M = pool.pop()
    ents = pool[:2 * nfacts]
    facts = [(ents[2 * i], ents[2 * i + 1]) for i in range(nfacts)]   # (subj, obj)
    rng.shuffle(facts)
    seq = []
    for s, o in facts:
        seq += render(M, s, o, g) + [SEP]
    qi = int(rng.integers(nfacts)); qsubj, qobj = facts[qi]
    seq += [QSEP, qsubj]
    return seq, qobj, g


def _batch(rng, bs, grammars, device):
    eps = [gen(rng, grammars) for _ in range(bs)]; L = max(len(s) for s, _, _ in eps)
    x = torch.full((bs, L), PAD, dtype=torch.long)
    ans = torch.zeros(bs, dtype=torch.long)
    for i, (s, a, _) in enumerate(eps):
        x[i, :len(s)] = torch.tensor(s); ans[i] = a
    return x.to(device), ans.to(device), (x != PAD).to(device)


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.ln = nn.LayerNorm(d); self.a = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, kpm):
        z = self.ln(x); x = x + self.a(z, z, z, key_padding_mask=kpm, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class Reader(nn.Module):
    """Reads facts+query, predicts the queried object. arch='pos' = token+absolute-position (baseline that keys on
    where the marker sits). arch='rel' = a content-only RELATIONAL pass first (each token mixes with SAME-VALUE
    tokens -> markers, which recur across facts, get tagged by recurrence) THEN position -> parses any grammar."""
    def __init__(self, arch="pos", d=128, h=4, layers=3, max_len=64, vocab=VOCAB):
        super().__init__()
        self.arch = arch
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        if arch == "rel":
            self.rel_q = nn.Linear(d, d, bias=False)       # content-keyed relation pass: attend to same-value tokens
            self.rel_k = nn.Linear(d, d, bias=False)
            self.rel_v = nn.Linear(d, d, bias=False)       # value = content (so the answer is still copyable)
            self.rel_ln = nn.LayerNorm(d)
            self.gate = nn.Parameter(torch.zeros(1))       # ZERO-INIT: start as the working baseline, learn to use recurrence
        self.blocks = nn.ModuleList([Attn(d, h) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, vocab))

    def forward(self, x):
        kpm = (x == PAD)
        e = self.tok(x)
        h = e + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        if self.arch == "rel":
            # recurrence side-channel: each token attends to SAME-VALUE tokens (content-keyed, position-free) and
            # summarizes them -> a marker (its value recurs across facts) gets a strong, distinctive signal; an
            # entity (appears ~once) gets almost none. Gated in from zero so it never breaks the base lookup.
            q, k = self.rel_q(e), self.rel_k(e)
            att = (q @ k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
            att = att.masked_fill(kpm[:, None, :], -1e9).softmax(-1)
            rel = self.rel_ln(att @ self.rel_v(e))
            h = h + torch.tanh(self.gate) * rel
        for b in self.blocks:
            h = b(h, kpm)
        return self.head(h[:, -1])                          # readout at the query-subject position -> predict object


FILL = 3                                                   # constant filler token (distinct from PAD/SEP/QSEP)
WV0 = 4                                                     # first real symbol id in the windowed task
WVOCAB = WV0 + NSYM


def gen_window(rng, marker_slots, K=7, nfacts=None):
    """FORCING task: each fact is a K-slot window with the marker at one of `marker_slots` and the two entities at
    two other random slots (subject before object), rest FILLER. The marker slot varies, so NO fixed position
    identifies it -- the only grammar-invariant cue is RECURRENCE (the marker value repeats across facts). Query a
    subject; answer its object = the other entity in its window. Train on most slots, hold one out."""
    nfacts = nfacts or int(rng.integers(3, 6))
    pool = (rng.permutation(NSYM) + WV0).tolist()
    M = pool.pop()
    ents = pool[:2 * nfacts]
    facts = [(ents[2 * i], ents[2 * i + 1]) for i in range(nfacts)]
    rng.shuffle(facts)
    seq = []
    for s, o in facts:
        m_slot = int(marker_slots[int(rng.integers(len(marker_slots)))])
        rest = [j for j in range(K) if j != m_slot]
        a, b = sorted(rng.choice(rest, 2, replace=False).tolist())   # subject slot < object slot
        win = [FILL] * K
        win[m_slot] = M; win[a] = s; win[b] = o
        seq += win + [SEP]
    qi = int(rng.integers(nfacts)); qsubj, qobj = facts[qi]
    seq += [QSEP, qsubj]
    return seq, qobj


def _batch_window(rng, bs, marker_slots, device):
    eps = [gen_window(rng, marker_slots) for _ in range(bs)]; L = max(len(s) for s, _ in eps)
    x = torch.full((bs, L), PAD, dtype=torch.long); ans = torch.zeros(bs, dtype=torch.long)
    for i, (s, a) in enumerate(eps):
        x[i, :len(s)] = torch.tensor(s); ans[i] = a
    return x.to(device), ans.to(device)


def train_w(arch, steps, marker_slots, d=128, h=4, layers=3, bs=64, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Reader(arch, d=d, h=h, layers=layers, max_len=80, vocab=WVOCAB).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        x, ans = _batch_window(rng, bs, marker_slots, device)
        loss = F.cross_entropy(m(x), ans)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def eval_w(m, marker_slots, n=400, seed=999, device="cpu"):
    rng = np.random.default_rng(seed); m.eval(); ok = 0
    with torch.no_grad():
        for _ in range(0, n, 100):
            b = min(100, n - _)
            x, ans = _batch_window(rng, b, marker_slots, device)
            ok += int((m(x).argmax(-1) == ans).sum())
    return ok / n


def experiment_forced(steps=12000, device="cpu", K=7, heldout=3):
    train_slots = tuple(j for j in range(K) if j != heldout)
    print(f"  forcing: marker slot varies; train slots {train_slots}, HELD-OUT slot {heldout}")
    out = {}
    for arch in ("pos",):
        m = train_w(arch, steps, train_slots, device=device)
        seen = eval_w(m, train_slots, device=device)
        held = eval_w(m, (heldout,), device=device)
        out[arch] = (seen, held)
        print(f"  {arch:3s}: trained slots {seen:.3f} | HELD-OUT slot {held:.3f}", flush=True)
    return out


def train(arch, steps, grammars=(0, 2), d=128, h=4, layers=3, bs=64, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Reader(arch, d=d, h=h, layers=layers).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        x, ans, _ = _batch(rng, bs, grammars, device)
        loss = F.cross_entropy(m(x), ans)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def evaluate(m, grammars, n=400, seed=999, device="cpu"):
    rng = np.random.default_rng(seed); m.eval(); ok = 0
    with torch.no_grad():
        for _ in range(0, n, 100):
            b = min(100, n - _)
            x, ans, _ = _batch(rng, b, grammars, device)
            ok += int((m(x).argmax(-1) == ans).sum())
    return ok / n


def experiment(steps=12000, device="cpu"):
    out = {}
    for arch in ("pos", "rel"):
        m = train(arch, steps, grammars=(0, 2), device=device)
        ind = evaluate(m, (0, 2), device=device)            # trained word orders
        cross = evaluate(m, (1,), device=device)            # NEVER-trained marker position (the wall)
        out[arch] = (ind, cross)
        print(f"  {arch:3s}: in-distribution {ind:.3f} | cross-grammar(unseen order) {cross:.3f}", flush=True)
    return out


def selftest():
    # (1) THE WALL: train two discrete word orders -> perfect in-distribution, collapse on the unseen third.
    mw = train("pos", 7000, grammars=(0, 2), device="cpu")
    wall_ind = evaluate(mw, (0, 2), device="cpu"); wall_cross = evaluate(mw, (1,), device="cpu")
    # (2) THE FIX (force recurrence): train the SAME plain reader on MANY marker positions so the position
    # shortcut is infeasible -> it must learn 'marker = recurring token' -> generalizes to a held-out position.
    # (Needs enough steps: the general rule emerges via a LATE phase transition, like arXiv:2505.20896's phase 3.)
    train_slots = (0, 1, 2, 4, 5, 6)
    mf = train_w("pos", 12000, train_slots, device="cpu")
    seen = eval_w(mf, train_slots, device="cpu"); held = eval_w(mf, (3,), device="cpu")
    print(f"langrole selftest: WALL train2-orders in-dist {wall_ind:.3f} / unseen-order {wall_cross:.3f}  ||  "
          f"FIX train-6-positions seen {seen:.3f} / HELD-OUT position {held:.3f}")
    assert wall_ind > 0.9 and wall_cross < 0.1, f"wall did not reproduce: ind {wall_ind} cross {wall_cross}"
    assert held > 0.75, f"forcing-recurrence fix did not generalize to the held-out position: {held}"
    print("langrole selftest OK (the variable-word-order wall is an artifact of too-few training orders; "
          "marker-position DIVERSITY forces the recurrence rule and generalizes to an unseen order)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    print("variable-word-order inference (train marker {first,last}, test unseen {middle}):")
    experiment(steps=a.steps, device=a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
