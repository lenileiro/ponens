#!/usr/bin/env python3
"""Stage 3 of learn-any-language-in-context: REASON, then RESPOND in the prompt's own language.

The agent reads facts written in an INVENTED language -- a random per-episode grammar: content entities joined by
a learned CONNECTIVE in a sentence frame ("<conn> SUBJ OBJ", a prefix/VSO frame, e.g. "isa a b"). Both the
connective and the entities are random symbols permuted fresh each episode, so nothing is memorized: the model
must INFER the language's structure in-context, REASON over it, and then SPEAK it. Asked about an entity, the model:

  1. THINKs -- a scratchpad walk up the chain (bare nodes a2 a3 ... aN, the proven stage-2 reasoning that
     terminates cleanly when a node has no parent), then
  2. RESPONDs -- one well-formed sentence in the language's own frame stating the conclusion: "a1 <conn> aN".

Distractor facts (extra edges) force real lookup, not fact-dumping. Trained dense-LM across random languages, so
the whole skill -- infer the grammar, chain, answer in that grammar -- is language-agnostic. Frame note: the
object immediately follows the subject so the induction/copy head can walk it (a connective placed BETWEEN
subject and object lands exactly where the copy head reads and collapses the walk -- prefix it instead).

  facts (invented frame, shuffled, + distractors):  f a b .  f x y .  f b c .   question: q a
  THINK: b c   RESPOND (in the same frame, transitive conclusion):  f a c

  python -m thinking.langrespond --selftest
  python -m thinking.langrespond --steps 9000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from scratchpad_model import ScratchpadLM

NSYM = 60
PAD, SEP, END, ANSW = 0, 1, 2, 3                          # ANSW = end-of-thinking, start of the framed response
VOCAB = 4 + NSYM


def gen(rng):
    """One invented-language episode. Returns (sequence, gen_start, walk, answer, chain) where walk = the
    scratchpad nodes to think and answer = the framed response sentence "<conn> a1 aN" to speak."""
    pool = ((rng.permutation(NSYM)) + 4).tolist()
    def take(k):
        out = pool[:k]; del pool[:k]; return out
    Fc = take(1)                                                                   # the learned connective ("conn X Y")
    Qpost = take(1)                                                                # the question marker ("a1 ?")
    depth = int(rng.integers(3, 6))
    chain = take(depth)                                                            # the reasoning path a1..aN
    dist = take(int(rng.integers(3, 7)))                                           # distractor entities

    def stmt(s, o):
        return Fc + [s, o]                                                         # connective first; object follows subject

    path_edges = [(chain[i], chain[i + 1]) for i in range(depth - 1)]
    # distractor edges originate ONLY from distractor entities => every chain node keeps a unique parent
    others = dist + chain
    dist_edges = [(s, others[int(rng.integers(len(others)))]) for s in dist for _ in range(int(rng.integers(0, 2)))]
    sents = [stmt(a, b) for a, b in path_edges] + [stmt(a, b) for a, b in dist_edges]
    rng.shuffle(sents)

    seq = []
    for st in sents:
        seq += st + [SEP]
    seq += [chain[0]] + Qpost + [SEP]                                              # the question about a1
    gen_start = len(seq)
    walk = chain[1:]                                                               # think: a2 a3 ... aN
    answer = Fc + [chain[0], chain[-1]]                                            # respond: <conn> a1 aN
    seq += walk + [ANSW] + answer + [END]
    return seq, gen_start, walk, answer, chain


def _batch(rng, bs):
    eps = [gen(rng) for _ in range(bs)]; L = max(len(s) for s, *_ in eps)
    x = torch.full((bs, L), PAD, dtype=torch.long)
    for i, (s, *_) in enumerate(eps):
        x[i, :len(s)] = torch.tensor(s)
    return x


def train(steps, d=192, layers=4, heads=6, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = ScratchpadLM(VOCAB, d=d, layers=layers, heads=heads, max_len=160, pad=PAD,
                     pos_mode="rope", causal=True, pointer=True, tie=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(steps):
        x = _batch(rng, 48); out = m(x)
        loss = F.nll_loss(out[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def speak(m, seq, gen_start, max_steps=40):
    """Generate think + response autoregressively (reason, then respond in the language)."""
    cur = seq[:gen_start]; out = []
    for _ in range(max_steps):
        nxt = int(m(torch.tensor([cur]))[0, -1].argmax())
        if nxt == END:
            break
        out.append(nxt); cur.append(nxt)
    return out


def evaluate(m, n=300, seed=999):
    rng = np.random.default_rng(seed); m.eval(); resp_ok = walk_ok = 0
    with torch.no_grad():
        for _ in range(n):
            seq, gen_start, walk, answer, chain = gen(rng)
            got = speak(m, seq, gen_start)
            if ANSW in got:
                gi = got.index(ANSW)
                walk_ok += (got[:gi] == walk)
                resp_ok += (got[gi + 1:] == answer)       # well-formed response sentence with correct conclusion
    return resp_ok / n, walk_ok / n


def selftest():
    m = train(steps=7000)
    resp, walk = evaluate(m, 300)
    print(f"langrespond selftest: well-formed-correct response {resp:.3f} | reasoning-walk {walk:.3f} "
          f"on unseen invented languages")
    assert resp > 0.85, f"too low: {resp}"
    print("langrespond selftest OK (reasons then responds in the prompt's own language; "
          "connective learned in-context, conclusion by reasoning)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--steps", type=int, default=9000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    m = train(steps=a.steps)
    resp, walk = evaluate(m)
    print(f"respond-in-language on unseen invented languages -> response {resp:.3f}, reasoning-walk {walk:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
