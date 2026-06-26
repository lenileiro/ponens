#!/usr/bin/env python3
"""Stage 1 of learn-any-language-in-context: reliable IN-CONTEXT LOOKUP across random-symbol languages, built on
the project's proven induction machinery (ScratchpadLM's pointer/copy head + dense LM loss -- the combination
that broke the in-context recall floor).

A vanilla transformer sits at chance on this (it learns 'the answer is a value slot' but not WHICH -- the
induction/binding wall). The pointer head copies the value bound to the queried key. Every episode is a fresh
random language (symbols permuted), so the model cannot memorize content -- it must LEARN to bind key->value
in-context, which generalizes to any unseen symbol set ('any language').

  episode:  k1 v1 k2 v2 ... kN vN  kq   -> predict vq   (the value bound to the queried key)

  python -m thinking.langlearn --selftest
  python -m thinking.langlearn --steps 6000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from scratchpad_model import ScratchpadLM

NSYM = 50                                                # symbol pool a language draws from
PAD = 0
VOCAB = 1 + NSYM


def gen(rng, n=None, max_len=40):
    """One random-language MQAR episode: n key->value pairs (distinct keys), then a queried key; target is its
    value. Symbols are permuted fresh => content can't be memorized, only the in-context binding."""
    n = n or int(rng.integers(4, 8))
    syms = (rng.permutation(NSYM) + 1).tolist()          # this language's symbols
    keys, vals = syms[:n], syms[n:2 * n]
    seq = []
    for k, v in zip(keys, vals):
        seq += [k, v]
    qi = int(rng.integers(n)); seq += [keys[qi], vals[qi]]   # query key, then its value (the target to predict)
    return seq[:max_len]


def _batch(rng, bs, max_len=40):
    eps = [gen(rng, max_len=max_len) for _ in range(bs)]; L = max(len(e) for e in eps)
    x = torch.full((bs, L), PAD, dtype=torch.long); qpos = []
    for i, e in enumerate(eps):
        x[i, :len(e)] = torch.tensor(e); qpos.append(len(e) - 2)   # position of the query key (predict next)
    return x, qpos


def train(steps, d=128, layers=3, heads=4, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = ScratchpadLM(VOCAB, d=d, layers=layers, heads=heads, max_len=48, pad=PAD,
                     pos_mode="rope", causal=True, pointer=True, tie=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(steps):
        x, _ = _batch(rng, 64)
        out = m(x)                                        # log-probs (B, L, V)
        loss = F.nll_loss(out[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1), ignore_index=PAD)  # dense LM loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def evaluate(m, n=500, seed=999):
    rng = np.random.default_rng(seed); m.eval(); ok = 0
    with torch.no_grad():
        for _ in range(n):
            e = gen(rng); x = torch.tensor([e])
            qpos = len(e) - 2                            # predict the value after the query key
            pred = int(m(x)[0, qpos].argmax())
            ok += (pred == e[-1])
    return ok / n


def selftest():
    m = train(steps=3000)
    acc = evaluate(m, 500)
    print(f"langlearn selftest: {acc:.3f} in-context lookup on unseen random languages")
    assert acc > 0.9, f"too low: {acc}"
    print("langlearn selftest OK (learns key->value binding in-context; generalizes to any symbol set)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--steps", type=int, default=5000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    m = train(steps=a.steps)
    print(f"in-context lookup accuracy on unseen random languages: {evaluate(m):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
