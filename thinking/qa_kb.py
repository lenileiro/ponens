#!/usr/bin/env python3
"""Symbolic-KB reading-comprehension QA -- the binding-problem-free route for MULTI-ENTITY passages.

The neural span-pointer (thinking/qa.py) cannot bind an answer to the RIGHT entity among several (the documented
neural binding problem -- 5 architectures incl. BiDAF stuck at chance, even on seen tokens). The fix is
"model proposes, brain proves" applied to text: a LEARNED per-sentence tagger extracts (subject, relation,
object) TRIPLES (a local, structural task with no binding -- it generalizes to unseen tokens like the
single-entity reader did), then a SYMBOLIC KB keyed by (subject, relation) answers the question by LOOKUP.
Binding becomes an exact dict-key match on the entity token -> trivial, works for any number of entities, no
hardcoded word matching (the tagger is learned; only the lookup is symbolic).

  python -m thinking.qa_kb --selftest
  python -m thinking.qa_kb            # train tagger + answer a multi-entity held-out passage
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.qa import POOL_TR, POOL_TE, READABLE, TEMPLATES, build_vocab  # noqa

RELS = ["place", "year", "job", "count"]; RID = {r: i for i, r in enumerate(RELS)}


def gen_tag(rng, test=False):
    """One training item: a FACT sentence (subject+object+relation) or a QUESTION (subject+relation, no object).
    Returns (tokens, subject_pos, object_pos_or_-100, relation_id). Values are random tokens (structural task)."""
    pool = POOL_TE if test else POOL_TR
    name, val = (pool[i] for i in rng.choice(len(pool), 2, replace=False))
    rel = RELS[rng.integers(len(RELS))]; ptemps, qtemps = TEMPLATES[rel]
    if rng.random() < 0.5:                                # a fact sentence -> subject + object
        toks = ptemps[rng.integers(len(ptemps))].format(e=name, v=val).split()
        return toks, toks.index(name), toks.index(val), RID[rel]
    toks = qtemps[rng.integers(len(qtemps))].format(e=name).split()   # a question -> subject only
    return toks, toks.index(name), -100, RID[rel]


class Tagger(nn.Module):
    """Bidirectional transformer with two POINTERS (subject position, object position) + a RELATION classifier.
    Pointers (softmax over positions, exactly one selection) are stable -- unlike a per-token role head, which
    can collapse on the rare OBJECT class. Local structural task (no cross-entity binding) -> generalizes."""
    def __init__(self, vocab, d=96, layers=2, heads=4, max_len=24):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0); self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.subj = nn.Linear(d, 1); self.obj = nn.Linear(d, 1); self.rel = nn.Linear(d, len(RELS))

    def forward(self, ids, pad):
        h = self.enc(self.emb(ids) + self.pos(torch.arange(ids.shape[1])[None]), src_key_padding_mask=~pad)
        ssc = self.subj(h).squeeze(-1).masked_fill(~pad, -1e9)        # subject-position pointer
        osc = self.obj(h).squeeze(-1).masked_fill(~pad, -1e9)         # object-position pointer
        pooled = (h * pad[..., None]).sum(1) / pad.sum(1, keepdim=True).clamp(min=1)
        return ssc, osc, self.rel(pooled)


def _vec(toks, vocab):
    return [vocab.get(t, 0) for t in toks]


def train_tagger(steps=2500, d=96, seed=0, verbose=True):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); vocab = build_vocab()
    m = Tagger(len(vocab), d=d); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for step in range(steps):
        items = [gen_tag(rng) for _ in range(64)]; L = max(len(t) for t, _, _, _ in items)
        ids = torch.tensor([_vec(t, vocab) + [0] * (L - len(t)) for t, _, _, _ in items])
        pad = torch.tensor([[1] * len(t) + [0] * (L - len(t)) for t, _, _, _ in items], dtype=torch.bool)
        sp = torch.tensor([s for _, s, _, _ in items]); op = torch.tensor([o for _, _, o, _ in items])
        rels = torch.tensor([rl for _, _, _, rl in items])
        ssc, osc, rellog = m(ids, pad)
        loss = (F.cross_entropy(ssc, sp) + F.cross_entropy(osc, op, ignore_index=-100)   # obj only for sentences
                + F.cross_entropy(rellog, rels))
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % max(1, steps // 6) == 0:
            print(f"  tagger step {step}: loss {loss.item():.3f}", flush=True)
    return m, vocab


def parse(m, vocab, toks):
    """Parse a token sequence -> (subject_token, relation, object_token) via the two pointers + relation head."""
    ids = torch.tensor([_vec(toks, vocab)]); pad = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        ssc, osc, rellog = m(ids, pad)
    return toks[int(ssc[0].argmax())], RELS[int(rellog[0].argmax())], toks[int(osc[0].argmax())]


def build_kb(m, vocab, passage_toks):
    """Split the passage into sentences (on '.'), parse each into a triple, key the KB by (subject, relation)."""
    kb, sent = {}, []
    for t in passage_toks + ["."]:
        if t == ".":
            if sent:
                s, r, o = parse(m, vocab, sent)
                if s is not None and o is not None:
                    kb[(s, r)] = o
            sent = []
        else:
            sent.append(t)
    return kb


def answer_kb(m, vocab, passage, question):
    kb = build_kb(m, vocab, passage.split())
    s, r, _ = parse(m, vocab, question.split())
    return kb.get((s, r))


def gen_passage(rng, test=False, n_entities=2):
    """A multi-entity passage + (question, answer) pairs (reuses qa.py's templates)."""
    pool = POOL_TE if test else POOL_TR
    idxs = rng.choice(len(pool), n_entities * 5, replace=False); toks = [pool[i] for i in idxs]
    sents, qa = [], []
    for e in range(n_entities):
        name = toks[e * 5]; vals = toks[e * 5 + 1:e * 5 + 5]
        for ft, v in zip(RELS, vals):
            pt, qt = TEMPLATES[ft]
            sents.append(pt[rng.integers(len(pt))].format(e=name, v=v))
            qa.append((qt[rng.integers(len(qt))].format(e=name), v))
    order = rng.permutation(len(sents))
    return " ".join(sents[i] for i in order), qa


def evaluate(m, vocab, n=300, test=True, n_entities=3, seed=123):
    rng = np.random.default_rng(seed); ok = tot = 0
    for _ in range(n):
        passage, qa = gen_passage(rng, test, n_entities)
        for q, ans in qa:
            ok += (answer_kb(m, vocab, passage, q) == ans); tot += 1
    return ok / tot


def selftest():
    m, vocab = train_tagger(steps=1500, d=64, seed=0, verbose=False)
    for ne in (2, 3):
        acc = evaluate(m, vocab, n=150, test=True, n_entities=ne)     # MULTI-entity, unseen tokens
        print(f"  KB-QA {ne}-entity held-out: {acc:.3f}")
        assert acc > 0.9, f"{ne}-entity too low: {acc}"
    print("qa_kb selftest OK (multi-entity QA via extract-to-KB + symbolic lookup; binding by entity key)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    m, vocab = train_tagger(steps=a.steps)
    for ne in (2, 3, 4):
        print(f"\nheld-out {ne}-entity KB-QA accuracy: {evaluate(m, vocab, n_entities=ne):.3f}")
    rng = np.random.default_rng(1); passage, qa = gen_passage(rng, test=True, n_entities=3)
    print("\nMULTI-ENTITY PASSAGE (held-out tokens):\n  " + passage)
    for q, ans in qa[:6]:
        got = answer_kb(m, vocab, passage, q)
        print(f"  Q: {q}\n  A: {got}   {'OK' if got == ans else '(wrong, gold ' + str(ans) + ')'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
