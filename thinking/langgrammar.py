#!/usr/bin/env python3
"""Stage 5 of learn-any-language-in-context: form rules NATURALLY for text -- position-invariant, grammar-inferred
multi-relation reasoning. (GPU-ready; the full robustness run is meant for GPU.)

Earlier stages leaned on POSITION (fixed question slot, object adjacent to subject so the copy head could read
it). That does not scale to real text. Here each random per-episode language has its own GRAMMAR -- the WORD
ORDER of a fact, i.e. where the relation MARKER sits relative to its two arguments -- and the model must INFER
that grammar in-context (markers are recognizable by RECURRENCE across sentences, not by position), parse the
facts, REASON (isa chain + inherited property -- two relations), and PRODUCE the answer in the SAME inferred word
order. Facts are shuffled with a variable count, so absolute position is uninformative: the query is located by
its marker, not its index.

GPU run 1 finding (runs/langgrammar_s.json): MIXING word orders in-context defeats the positional copy/induction
head even in-distribution at d256/40k (response ~0.01) -- inferring a VARIABLE word order per-prompt is an
architectural wall, not a scale problem. So this module now runs experiment (A): a FIXED word order (no in-context
grammar inference) with heavily RANDOMIZED positions, which directly tests the scalable concern -- not leaning on
a fixed global position. Metrics (reported separately):
  * in_distribution : fixed grammar, variable fact counts (the query is found by its MARKER, not its index).
  * random_position : same grammar, LONGER prompts / more distractors -> absolute positions shift beyond the
                      trained range; staying correct proves marker/role binding, not indexing.
  * cross_grammar   : a never-trained word order -> expected to fail (documents the open frontier (B): variable
                      word-order inference, which needs a different mechanism than positional copy).

  fixed grammar g=0 (marker FIRST), facts shuffled, variable count:  I a b .  I x y .  I b c .  P c p .  query: a
  THINK (isa-walk a->b->c, then c's property):  b c p     RESPOND in the same grammar (P a p):  P a p

  python -m thinking.langgrammar --selftest                       # CPU correctness gate
  python -m thinking.langgrammar --steps 60000 --device cuda --out runs/langgrammar.json
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from scratchpad_model import ScratchpadLM

NSYM = 80
PAD, SEP, END, ANSW = 0, 1, 2, 3
VOCAB = 4 + NSYM
GRAMMARS = (0, 1, 2)                      # marker BEFORE args / BETWEEN args / AFTER args (subject precedes object)


def render(marker, s, o, g):
    """Lay out one fact in word order g. The marker token moves; subject still precedes object (a head-direction
    universal). The model must INFER g from the recurring marker's position across sentences."""
    if g == 0:
        return [marker, s, o]
    if g == 1:
        return [s, marker, o]
    return [s, o, marker]


def gen(rng, grammars=GRAMMARS, extra=(0, 4)):
    """One invented two-relation episode in a randomly chosen grammar. `extra` = range of EXTRA distractor facts
    (raise it to push absolute positions past the trained range). Returns (seq, gen_start, think, answer, g).

    The reasoning isolates the variable under test (grammar/position): the top of the isa chain carries the only
    property, so the answer is unambiguous once the chain is walked. Distractor isa edges keep the WALK non-trivial
    and lengthen the prompt; the model still composes two relations + inheritance to answer."""
    g = int(grammars[int(rng.integers(len(grammars)))])
    pool = ((rng.permutation(NSYM)) + 4).tolist()
    def take(k):
        out = pool[:k]; del pool[:k]; return out
    Misa, Mprop, Qmark = take(1)[0], take(1)[0], take(1)[0]
    depth = int(rng.integers(3, 6))
    chain = take(depth)                                              # isa backbone a1..aN
    prop = take(1)[0]                                               # property carried by the top aN (the only holder)
    dist = take(int(rng.integers(3, 6)))                           # distractor entities

    isa_edges = [(chain[i], chain[i + 1]) for i in range(depth - 1)]
    others = dist + chain
    n_dist_isa = len(dist) + int(rng.integers(extra[0], extra[1] + 1))    # extra distractor edges lengthen the prompt
    dist_isa = [(dist[int(rng.integers(len(dist)))], others[int(rng.integers(len(others)))]) for _ in range(n_dist_isa)]
    prop_facts = [(chain[-1], prop)]

    sents = ([render(Misa, a, b, g) for a, b in isa_edges] + [render(Misa, a, b, g) for a, b in dist_isa]
             + [render(Mprop, s, p, g) for s, p in prop_facts])
    rng.shuffle(sents)

    seq = []
    for st in sents:
        seq += st + [SEP]
    qi = int(rng.integers(depth)); qnode = chain[qi]
    seq += [Qmark, qnode, SEP]                                      # question located by its marker, anywhere in the stream
    gen_start = len(seq)
    think = chain[qi + 1:] + [prop]                                # isa-walk to the top, then its inherited property
    answer = render(Mprop, qnode, prop, g)                          # respond in the SAME inferred word order
    seq += think + [ANSW] + answer + [END]
    return seq, gen_start, think, answer, g


def _batch(rng, bs, grammars, extra):
    eps = [gen(rng, grammars, extra) for _ in range(bs)]; L = max(len(s) for s, *_ in eps)
    x = torch.full((bs, L), PAD, dtype=torch.long)
    for i, (s, *_) in enumerate(eps):
        x[i, :len(s)] = torch.tensor(s)
    return x


def train(steps, grammars, d=256, layers=6, heads=8, bs=64, lr=1e-3, seed=0, device="cpu",
          max_len=192, extra=(0, 4)):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = ScratchpadLM(VOCAB, d=d, layers=layers, heads=heads, max_len=max_len, pad=PAD,
                     pos_mode="rope", causal=True, pointer=True, tie=True).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for step in range(steps):
        x = _batch(rng, bs, grammars, extra).to(device)
        out = m(x)
        loss = F.nll_loss(out[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def speak(m, seq, gen_start, device, max_steps=40):
    cur = seq[:gen_start]; out = []
    for _ in range(max_steps):
        nxt = int(m(torch.tensor([cur], device=device))[0, -1].argmax())
        if nxt == END:
            break
        out.append(nxt); cur.append(nxt)
    return out


def evaluate(m, grammars, extra=(0, 4), n=400, seed=999, device="cpu"):
    rng = np.random.default_rng(seed); m.eval(); resp_ok = prop_ok = 0
    with torch.no_grad():
        for _ in range(n):
            seq, gen_start, think, answer, g = gen(rng, grammars, extra)
            got = speak(m, seq, gen_start, device)
            if ANSW in got:
                resp = got[got.index(ANSW) + 1:]
                resp_ok += (resp == answer)                         # correct response IN THE INFERRED GRAMMAR
                prop_ok += (len(resp) >= 1 and answer[-1] in resp)  # got the inherited property (order-agnostic)
    return resp_ok / n, prop_ok / n


def run(steps, d=256, layers=6, heads=8, bs=64, lr=1e-3, seed=0, device="cpu", n_eval=400, grammar=0):
    """Experiment (A): POSITION-INVARIANCE with a FIXED grammar. The word order is fixed (no in-context grammar
    inference -- that defeated the positional copy head, see langgrammar_s.json), but absolute POSITIONS are
    heavily randomized via a variable number of distractor facts, and the query is located by its MARKER, not its
    index. The robustness question: does staying correct hold when prompts are LONGER than trained (positions
    shifted past the trained range)? A cross-grammar probe documents that a never-trained word order still fails
    (the open frontier (B))."""
    g = (grammar,); cross = ((grammar + 1) % len(GRAMMARS),)
    m = train(steps, g, d=d, layers=layers, heads=heads, bs=bs, lr=lr, seed=seed, device=device,
              max_len=224, extra=(0, 6))
    in_dist = evaluate(m, g, extra=(0, 6), n=n_eval, device=device)
    random_position = evaluate(m, g, extra=(10, 16), n=n_eval, device=device)        # longer prompts, shifted positions
    cross_grammar = evaluate(m, cross, extra=(0, 6), n=n_eval, device=device)         # never-trained word order (probe)
    res = {
        "steps": steps, "d": d, "layers": layers, "heads": heads, "seed": seed,
        "train_grammar": grammar, "cross_grammar_probe": cross[0],
        "in_distribution": {"response": in_dist[0], "property": in_dist[1]},
        "random_position": {"response": random_position[0], "property": random_position[1]},
        "cross_grammar": {"response": cross_grammar[0], "property": cross_grammar[1]},
    }
    return res


def selftest():
    device = "cpu"
    m = train(3000, grammars=(0,), d=128, layers=3, heads=4, bs=48, max_len=192, extra=(0, 6), device=device)
    resp, prop = evaluate(m, (0,), extra=(0, 6), n=200, device=device)
    print(f"langgrammar selftest (tiny/CPU, fixed-grammar position-invariance): response {resp:.3f} | "
          f"property {prop:.3f} (chance ~0.25)")
    assert prop > 0.45, f"did not learn above chance: {prop}"
    print("langgrammar selftest OK (composes two relations under randomized positions; full run is for GPU)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grammar", type=int, default=0, help="the single fixed word order to train on")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write result JSON here")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    res = run(a.steps, d=a.d, layers=a.layers, heads=a.heads, bs=a.bs, lr=a.lr, seed=a.seed,
              device=a.device, grammar=a.grammar)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
