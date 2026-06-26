#!/usr/bin/env python3
"""Stage 4 of learn-any-language-in-context: MULTIPLE relations + INHERITANCE in an invented language.

A real language has more than one relation. Here each random per-episode language has TWO learned connectives:
a taxonomic 'isa' (transitive backbone) and a 'hasprop' (a property a node carries). Properties INHERIT down the
isa chain -- the classic "robin isa bird, bird hasprop fly => robin hasprop fly", but with random connectives and
entities the model must INFER in-context, and the answer is the PROPERTY itself (generated), not yes/no.

Asked what property an entity has, the model COMPOSES the two relations: it walks the isa chain up to the ancestor
that carries a property (the proven scratchpad walk + the proven single-hop lookup), switching relation at the
top, then RESPONDs with the inherited property in the language's own frame. Distractor isa edges and distractor
property facts (other entities, other properties) force real composition, not grabbing any property.

  facts (invented, shuffled):  I a b .  I b c .  P c p .  I x y .  P y q .   question: a ?
  THINK (isa-walk a->b->c, then c's property):  b c p     RESPOND (framed):  P a p

  python -m thinking.langrel --selftest
  python -m thinking.langrel --steps 11000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from scratchpad_model import ScratchpadLM

NSYM = 70
PAD, SEP, END, ANSW = 0, 1, 2, 3
VOCAB = 4 + NSYM


def gen(rng):
    """One invented two-relation episode. Returns (sequence, gen_start, think, answer, query_node, prop)."""
    pool = ((rng.permutation(NSYM)) + 4).tolist()
    def take(k):
        out = pool[:k]; del pool[:k]; return out
    Cisa, Cprop, Qmark = take(1), take(1), take(1)                                 # two relations + a question marker
    depth = int(rng.integers(3, 6))
    chain = take(depth)                                                            # the isa backbone a1..aN
    prop = take(1)[0]                                                              # the property carried by the top aN
    dist = take(int(rng.integers(3, 6)))                                           # distractor entities
    dist_props = take(int(rng.integers(2, 5)))                                     # distractor property values

    def fact(c, s, o):
        return c + [s, o]                                                          # connective first; object follows subject

    isa_edges = [(chain[i], chain[i + 1]) for i in range(depth - 1)]
    others = dist + chain
    dist_isa = [(s, others[int(rng.integers(len(others)))]) for s in dist for _ in range(int(rng.integers(0, 2)))]
    prop_facts = [(chain[-1], prop)]                                              # the top of the chain carries a property
    dist_prop_facts = [(dist[int(rng.integers(len(dist)))], p) for p in dist_props]   # other entities carry other props

    sents = ([fact(Cisa, a, b) for a, b in isa_edges] + [fact(Cisa, a, b) for a, b in dist_isa]
             + [fact(Cprop, s, p) for s, p in prop_facts] + [fact(Cprop, s, p) for s, p in dist_prop_facts])
    rng.shuffle(sents)

    seq = []
    for st in sents:
        seq += st + [SEP]
    qi = int(rng.integers(depth))                                                 # ask about any node at/below the top
    qnode = chain[qi]
    seq += [qnode] + Qmark + [SEP]
    gen_start = len(seq)
    think = chain[qi + 1:] + [prop]                                               # isa-walk up to the top, then its property
    answer = Cprop + [qnode, prop]                                                # respond: "<hasprop> qnode prop"
    seq += think + [ANSW] + answer + [END]
    return seq, gen_start, think, answer, qnode, prop


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
    cur = seq[:gen_start]; out = []
    for _ in range(max_steps):
        nxt = int(m(torch.tensor([cur]))[0, -1].argmax())
        if nxt == END:
            break
        out.append(nxt); cur.append(nxt)
    return out


def evaluate(m, n=300, seed=999):
    rng = np.random.default_rng(seed); m.eval(); resp_ok = prop_ok = 0
    with torch.no_grad():
        for _ in range(n):
            seq, gen_start, think, answer, qnode, prop = gen(rng)
            got = speak(m, seq, gen_start)
            if ANSW in got:
                resp = got[got.index(ANSW) + 1:]
                resp_ok += (resp == answer)                  # well-formed framed response with the inherited property
                prop_ok += (len(resp) >= 1 and resp[-1] == prop)   # got the right inherited property (looser)
    return resp_ok / n, prop_ok / n


def selftest():
    m = train(steps=9000)
    resp, prop = evaluate(m, 300)
    print(f"langrel selftest: well-formed-correct response {resp:.3f} | inherited-property {prop:.3f} "
          f"on unseen invented languages")
    assert resp > 0.85, f"too low: {resp}"
    print("langrel selftest OK (composes two learned relations + inheritance; responds with the inherited property)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--steps", type=int, default=11000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    m = train(steps=a.steps)
    resp, prop = evaluate(m)
    print(f"multi-relation inheritance on unseen invented languages -> response {resp:.3f}, inherited-property {prop:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
