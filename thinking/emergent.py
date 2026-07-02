#!/usr/bin/env python3
"""emergent -- the agent DISCOVERS language on its own via a self-play referential game.

A different paradigm from the supervised stack (thinking/STACK.md): no hand-authored grammar, no
labeled NL<->meaning. Two agents INVENT a shared discrete code to communicate structured meanings; the
only pressure is communicative success + generalization. Compositionality is not given -- it EMERGES
(or doesn't), and we MEASURE it.

Setup (Lewis signaling / referential game, the standard emergent-communication design):
  * MEANINGS: structured objects = n_attr attributes x n_val values (this structure is what *lets* a
    compositional code exist -- a symbol per attribute-value).
  * SPEAKER: sees a target meaning -> emits a discrete MESSAGE (length L over vocab V) via
    Gumbel-softmax straight-through (end-to-end differentiable; no REINFORCE variance).
  * LISTENER: gets the message + the target among distractors -> must pick the target. Success trains
    BOTH jointly. The agents invent the code; nothing about it is given.
  * PRESSURE: hold out a fraction of attribute-value COMBINATIONS -> the emergence test is zero-shot
    communication accuracy on UNSEEN combos.
  * METRICS (not loss): comm accuracy (train vs held-out) + TOPOGRAPHIC SIMILARITY (Spearman corr of
    meaning-distance vs message-distance -- the standard compositionality measure).

Decisions (locked): self-play language game; the agent's OWN emergent code first (English alignment
deferred); prove EMERGENCE first (C2 deferred).

  python -m thinking.emergent --selftest
  python -m thinking.emergent --steps 3000
"""
import argparse
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================================================
# Meaning space + train/held-out COMBINATION split
# ===================================================================================================
def all_meanings(n_attr, n_val):
    import itertools
    return [tuple(m) for m in itertools.product(range(n_val), repeat=n_attr)]


def split_meanings(n_attr, n_val, held_frac=0.2, seed=0):
    ms = all_meanings(n_attr, n_val)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ms)); rng.shuffle(idx)
    cut = int(len(ms) * (1 - held_frac))
    return [ms[i] for i in idx[:cut]], [ms[i] for i in idx[cut:]]


def onehot(meaning, n_val):
    v = torch.zeros(len(meaning) * n_val)
    for a, val in enumerate(meaning):
        v[a * n_val + val] = 1.0
    return v


# ===================================================================================================
# Agents
# ===================================================================================================
class Speaker(nn.Module):
    """meaning -> message (L symbols over vocab V), emitted via Gumbel-softmax (straight-through)."""
    def __init__(self, n_attr, n_val, L, V, d=128):
        super().__init__()
        self.L, self.V = L, V
        self.enc = nn.Sequential(nn.Linear(n_attr * n_val, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.heads = nn.Linear(d, L * V)

    def forward(self, x, tau=1.0, hard=True):
        h = self.enc(x)
        logits = self.heads(h).view(-1, self.L, self.V)
        if self.training:
            msg = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)   # (B, L, V) ~ one-hot
        else:
            idx = logits.argmax(-1)                                      # greedy at eval
            msg = F.one_hot(idx, self.V).float()
        return msg, logits


class Listener(nn.Module):
    """message + candidate meanings -> pick the target. Embeds symbols, scores each candidate."""
    def __init__(self, n_attr, n_val, L, V, d=128):
        super().__init__()
        self.sym = nn.Linear(V, d, bias=False)
        self.msg = nn.GRU(d, d, batch_first=True)
        self.obj = nn.Sequential(nn.Linear(n_attr * n_val, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, msg, cands):
        # msg (B, L, V) ; cands (B, C, n_attr*n_val)
        e = self.sym(msg)                                # (B, L, d)
        _, hn = self.msg(e)                              # (1, B, d)
        m = hn.squeeze(0)                                # (B, d)
        o = self.obj(cands)                              # (B, C, d)
        return torch.einsum("bd,bcd->bc", m, o)         # (B, C) scores


# ===================================================================================================
# Game: a batch of targets + distractors; listener must pick the target
# ===================================================================================================
def make_batch(rng, meanings, n_val, batch, n_cand):
    B = batch
    tgt_i = rng.integers(0, len(meanings), size=B)
    targets = [meanings[i] for i in tgt_i]
    x = torch.stack([onehot(m, n_val) for m in targets])                # (B, A*V)
    cand = torch.zeros(B, n_cand, x.shape[1])
    pos = rng.integers(0, n_cand, size=B)                               # target slot per row
    for b in range(B):
        used = {targets[b]}
        cand[b, pos[b]] = x[b]
        for c in range(n_cand):
            if c == pos[b]:
                continue
            while True:
                dm = meanings[rng.integers(0, len(meanings))]
                if dm not in used:
                    used.add(dm); cand[b, c] = onehot(dm, n_val); break
    return x, cand, torch.tensor(pos, dtype=torch.long), targets


def train(spk, lis, meanings, n_val, steps, n_cand, batch=128, lr=1e-3, seed=0, tau=1.5, log=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.Adam(list(spk.parameters()) + list(lis.parameters()), lr=lr)
    spk.train(); lis.train()
    for step in range(steps):
        x, cand, pos, _ = make_batch(rng, meanings, n_val, batch, n_cand)
        msg, _ = spk(x, tau=tau, hard=True)
        scores = lis(msg, cand)
        loss = F.cross_entropy(scores, pos)
        opt.zero_grad(); loss.backward(); opt.step()
        if log and (step % log == 0 or step == steps - 1):
            acc = (scores.argmax(-1) == pos).float().mean().item()
            print(f"  step {step:5d}  loss {loss.item():.3f}  train-acc {acc:.3f}", flush=True)
    return spk, lis


# ===================================================================================================
# Eval: communication accuracy (greedy) + topographic similarity
# ===================================================================================================
@torch.no_grad()
def comm_acc(spk, lis, meanings, n_val, n_cand, seed=1, trials=2000):
    spk.eval(); lis.eval()
    rng = np.random.default_rng(seed)
    x, cand, pos, _ = make_batch(rng, meanings, n_val, min(trials, 2000), n_cand)
    msg, _ = spk(x)
    scores = lis(msg, cand)
    return (scores.argmax(-1) == pos).float().mean().item()


@torch.no_grad()
def messages_for(spk, meanings, n_val):
    spk.eval()
    x = torch.stack([onehot(m, n_val) for m in meanings])
    msg, _ = spk(x)
    return msg.argmax(-1).cpu().numpy()                # (N, L) symbol ids


def topsim(spk, meanings, n_val, seed=2, pairs=3000):
    """Spearman corr between meaning-distance (Hamming over attributes) and message-distance (Hamming
    over symbols). High -> similar meanings get similar messages -> COMPOSITIONAL."""
    msgs = messages_for(spk, meanings, n_val)
    rng = np.random.default_rng(seed)
    md, sd = [], []
    n = len(meanings)
    for _ in range(pairs):
        i, j = rng.integers(0, n), rng.integers(0, n)
        if i == j:
            continue
        md.append(sum(a != b for a, b in zip(meanings[i], meanings[j])))
        sd.append(int((msgs[i] != msgs[j]).sum()))
    md, sd = np.array(md, float), np.array(sd, float)

    def rank(a):
        order = a.argsort(); r = np.empty_like(order, float); r[order] = np.arange(len(a)); return r
    rm, rs = rank(md), rank(sd)
    if rm.std() < 1e-9 or rs.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rm, rs)[0, 1])


def chance(n_cand):
    return 1.0 / n_cand


# ===================================================================================================
# selftest + run
# ===================================================================================================
def selftest():
    torch.set_num_threads(2)
    n_attr, n_val, L, V, n_cand = 2, 4, 2, 6, 5
    tr, te = split_meanings(n_attr, n_val, 0.2, 0)
    assert tr and te
    spk = Speaker(n_attr, n_val, L, V, d=64)
    lis = Listener(n_attr, n_val, L, V, d=64)
    train(spk, lis, tr, n_val, steps=400, n_cand=n_cand, batch=64, seed=0)
    acc = comm_acc(spk, lis, tr, n_val, n_cand)
    assert acc > chance(n_cand) + 0.1, ("agents should learn to communicate above chance", acc)
    ts = topsim(spk, tr, n_val)
    assert -1.0 <= ts <= 1.0
    # messages are deterministic per meaning at eval
    m1 = messages_for(spk, tr[:3], n_val); m2 = messages_for(spk, tr[:3], n_val)
    assert (m1 == m2).all(), "greedy messages must be deterministic"
    print("emergent selftest OK")


def run(steps, seed=0, n_attr=4, n_val=8, L=4, V=10, n_cand=10, d=128, verbose=True):
    tr, te = split_meanings(n_attr, n_val, 0.2, seed)
    spk = Speaker(n_attr, n_val, L, V, d=d)
    lis = Listener(n_attr, n_val, L, V, d=d)
    if verbose:
        print(f"meanings: {n_attr}x{n_val}={n_val**n_attr} (train {len(tr)} / held-out {len(te)}) | "
              f"msg L={L} V={V} | candidates={n_cand} (chance {chance(n_cand):.3f}) | steps {steps}",
              flush=True)
    train(spk, lis, tr, n_val, steps=steps, n_cand=n_cand, batch=128, seed=seed,
          log=(max(1, steps // 8) if verbose else 0))
    tr_acc = comm_acc(spk, lis, tr, n_val, n_cand)
    te_acc = comm_acc(spk, lis, te, n_val, n_cand)              # ZERO-SHOT on unseen combinations
    ts_tr = topsim(spk, tr, n_val)
    ts_all = topsim(spk, tr + te, n_val)
    if verbose:
        print("\n== EMERGENCE (the agent's invented language) ==")
        print(f"  communication accuracy : train {tr_acc:.3f} | HELD-OUT (zero-shot combos) {te_acc:.3f}"
              f"  (chance {chance(n_cand):.3f})")
        print(f"  topographic similarity : {ts_all:.3f}  (>0 => compositional: similar meanings -> "
              f"similar messages)")
        print("  -> emergence = held-out accuracy >> chance AND topsim > 0 (generalizes by composing)")
    return dict(train_acc=tr_acc, heldout_acc=te_acc, chance=chance(n_cand),
                topsim=ts_all, topsim_train=ts_tr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-attr", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--msg-len", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=10)
    ap.add_argument("--cands", type=int, default=10)
    ap.add_argument("--d", type=int, default=128)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    run(args.steps, seed=args.seed, n_attr=args.n_attr, n_val=args.n_val, L=args.msg_len,
        V=args.vocab, n_cand=args.cands, d=args.d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
