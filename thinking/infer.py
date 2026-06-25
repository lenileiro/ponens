#!/usr/bin/env python3
"""Inference over sentences with a LEARNED semantic parser (no hardcoded word matching, no word dictionaries,
no baked-in example text). A small model LEARNS the function of the logical words (every / is / can ...) and
points to the subject & object terms, classifying the relation (is-a / can) and statement TYPE (rule / fact /
query). Crucially it only ever sees the LOGICAL words and a CASE feature -- every content word is mapped to a
single UNKNOWN slot in training AND at inference -- so it CANNOT memorize content and must reason from
structure alone; it generalizes to any real, unseen words. The parsed facts/rules go to the datalog engine and
the answer is whatever it PROVES.

Operates on text YOU pass (CLI or infer()); it has no example sentences of its own.

  python -m thinking.infer "Every dog is a mammal. Rex is a dog." "Is Rex a mammal?"
  python -m thinking.infer --selftest        # verifies on randomly-generated multi-hop problems
"""
import argparse
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datalog import Datalog

GWORDS = "every all a an the is are can could does do ? .".split()    # the only words the model knows; rest -> UNK
RELS = ["isa", "can"]; TYPES = ["rule", "fact", "query"]
PAD, UNK = 0, 1


def build_vocab():
    toks = {"[PAD]": PAD, "[UNK]": UNK}
    for w in GWORDS:
        toks.setdefault(w, len(toks))
    return toks


def _toks(s):
    return re.findall(r"[A-Za-z][A-Za-z0-9]*|\?", s)


def _enc(toks, vocab):
    """token -> (id, case). Logical words keep their id; ALL content words collapse to UNK -> the model reasons
    from structure (logical words + position + case), never from content identity."""
    ids = [vocab.get(w.lower(), UNK) for w in toks]
    case = [1 if w[:1].isupper() else 0 for w in toks]                # proper-noun cue (instance vs class)
    return ids, case


def gen_item(rng):
    """A random statement/question. Content terms are placeholders (the model sees them only as UNK); we keep
    their positions for the pointer targets. Returns (tokens, subj_pos, obj_pos, rel, type)."""
    c1, c2, a = (f"c{i}" for i in rng.choice(900, 3, replace=False)); e = f"E{rng.integers(900)}"
    k = rng.integers(6)
    if k == 0:   t = _toks(f"{rng.choice(['every','a'])} {c1} is a {c2}"); r, ty, s, o = 0, 0, c1, c2
    elif k == 1: t = _toks(f"{e} is a {c2}"); r, ty, s, o = 0, 1, e, c2
    elif k == 2: t = _toks(f"every {c1} can {a}"); r, ty, s, o = 1, 0, c1, a
    elif k == 3: t = _toks(f"{e} can {a}"); r, ty, s, o = 1, 1, e, a
    elif k == 4: x = rng.choice([e, c1]); t = _toks(f"is {x} a {c2} ?"); r, ty, s, o = 0, 2, x, c2
    else:        x = rng.choice([e, c1]); t = _toks(f"can {x} {a} ?"); r, ty, s, o = 1, 2, x, a
    return t, t.index(s), t.index(o), r, ty


class Parser(nn.Module):
    def __init__(self, vocab, d=128, layers=2, heads=4, max_len=12):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD); self.pos = nn.Embedding(max_len, d); self.case = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.subj = nn.Linear(d, 1); self.obj = nn.Linear(d, 1); self.rel = nn.Linear(d, len(RELS)); self.typ = nn.Linear(d, len(TYPES))

    def forward(self, ids, case, pad):
        h = self.enc(self.emb(ids) + self.pos(torch.arange(ids.shape[1])[None]) + self.case(case), src_key_padding_mask=~pad)
        sj = self.subj(h).squeeze(-1).masked_fill(~pad, -1e9); ob = self.obj(h).squeeze(-1).masked_fill(~pad, -1e9)
        pooled = (h * pad[..., None]).sum(1) / pad.sum(1, keepdim=True).clamp(min=1)
        return sj, ob, self.rel(pooled), self.typ(pooled)


def train(steps, d=128, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); vocab = build_vocab()
    m = Parser(len(vocab), d=d); opt = torch.optim.AdamW(m.parameters(), lr=8e-4)
    for _ in range(steps):
        it = [gen_item(rng) for _ in range(64)]; L = max(len(t) for t, *_ in it)
        ids = torch.zeros(len(it), L, dtype=torch.long); case = torch.zeros(len(it), L, dtype=torch.long); pad = torch.zeros(len(it), L, dtype=torch.bool)
        sp = torch.tensor([s for _, s, _, _, _ in it]); op = torch.tensor([o for _, _, o, _, _ in it])
        rl = torch.tensor([r for *_, r, _ in it]); tp = torch.tensor([t for *_, t in it])
        for i, (tk, *_) in enumerate(it):
            e, c = _enc(tk, vocab); n = len(e); ids[i, :n] = torch.tensor(e); case[i, :n] = torch.tensor(c); pad[i, :n] = True
        sj, ob, rel, typ = m(ids, case, pad)
        loss = F.cross_entropy(sj, sp) + F.cross_entropy(ob, op) + F.cross_entropy(rel, rl) + F.cross_entropy(typ, tp)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m, vocab


def _val(m, vocab, n=300, seed=999):
    rng = np.random.default_rng(seed); ok = 0
    for _ in range(n):
        t, sp, op, rl, tp = gen_item(rng); s, o, r, ty = parse_tokens(m, vocab, t)
        ok += (t[sp] == s and t[op] == o and r == RELS[rl] and ty == TYPES[tp])
    return ok / n


def fit(steps=2500, d=96, seed=0, tries=6, verbose=False):
    for k in range(tries):
        m, vocab = train(steps, d, seed + k); v = _val(m, vocab)
        if verbose:
            print(f"  fit try {k}: parse-val {v:.3f}", flush=True)
        if v > 0.97:
            return m, vocab
    return m, vocab


def parse_tokens(m, vocab, toks):
    ids, case = _enc(toks, vocab); it = torch.tensor([ids]); ct = torch.tensor([case]); pad = torch.ones_like(it, dtype=torch.bool)
    with torch.no_grad():
        sj, ob, rel, typ = m(it, ct, pad)
    return toks[int(sj[0].argmax())], toks[int(ob[0].argmax())], RELS[int(rel[0].argmax())], TYPES[int(typ[0].argmax())]


def _logic(parsed):
    s, o, rel, ty = parsed; s, o = s.lower(), o.lower()
    if ty == "rule":
        return "rule", ((rel, ("?x", o)), [("isa", ("?x", s))])
    return ("query" if ty == "query" else "fact"), (rel, (s, o))


def infer(m, vocab, text, question, verbose=False):
    facts, rules = [], []
    for sent in re.split(r"[.!?]", text):
        if not sent.strip():
            continue
        kind, val = _logic(parse_tokens(m, vocab, _toks(sent)))
        (rules if kind == "rule" else facts).append(val)
    _, goal = _logic(parse_tokens(m, vocab, _toks(question)))
    dl = Datalog(rules); closure, prov = dl.closure(facts)
    yes = goal in closure
    return ("yes" if yes else "no"), (dl.proof_tree(prov, goal) if (yes and verbose) else None)


def _gen_problem(rng):
    """A random multi-hop inference problem (chain of universal rules + an instance) -> (text, question, gold).
    Used only to VERIFY the engine; not example content (random terms each time)."""
    chain = [f"k{rng.integers(99999)}" for _ in range(rng.integers(2, 5))]; e = f"E{rng.integers(99999)}"
    stmts = [f"every {chain[i]} is a {chain[i + 1]}" for i in range(len(chain) - 1)] + [f"{e} is a {chain[0]}"]
    rng.shuffle(stmts)
    target = chain[-1] if rng.random() < 0.5 else f"z{rng.integers(99999)}"   # half reachable, half not
    gold = "yes" if target == chain[-1] else "no"
    return ". ".join(stmts) + " .", f"is {e} a {target} ?", gold


def selftest():
    m, vocab = fit(steps=2500, d=96)
    rng = np.random.default_rng(7); ok = 0; n = 200
    for _ in range(n):
        text, q, gold = _gen_problem(rng)
        ok += (infer(m, vocab, text, q)[0] == gold)
    acc = ok / n
    print(f"infer selftest: parse-val {_val(m, vocab):.3f} | end-to-end multi-hop {acc:.3f} ({n} random problems)")
    assert acc > 0.95, f"too low: {acc}"
    print("infer selftest OK (LEARNED parser, structure-only -> datalog-verified multi-hop chaining)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("text", nargs="?"); ap.add_argument("question", nargs="?")
    a = ap.parse_args(argv)
    if a.selftest or not (a.text and a.question):
        return selftest()
    m, vocab = fit(steps=a.steps)
    ans, proof = infer(m, vocab, a.text, a.question, verbose=True)
    print(f"A: {ans}")
    if proof:
        print(f"proof: {proof}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
