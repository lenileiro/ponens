#!/usr/bin/env python3
"""Reading-comprehension QA that LEARNS to answer -- no hardcoded word matching (no stopword lists, no regex,
no wh-word->type rules). A small bidirectional transformer reads [question | passage] and a span-POINTER head
points to the answer's position in the passage; the answer is the token it points at. The model learns the
question->answer-role mapping from self-generated comprehension data.

It is trained on one pool of names/places/years and TESTED on a DISJOINT pool (unseen tokens), so it cannot
memorize answers -- it must learn the STRUCTURE ('where ...' -> the place slot, 'when ...' -> the year slot,
'who ...' -> the subject) and point to the right position regardless of the actual value. In-house, from
scratch, no pretrained model.

  python -m thinking.qa --selftest
  python -m thinking.qa --steps 4000        # train + answer a held-out passage
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Slot VALUES are generic tokens drawn from disjoint train/test halves of a large pool. No token is inherently
# a 'place' or 'year' -> the ONLY way to answer is to point by FRAME STRUCTURE ('after in', 'after during',
# 'after a', 'after had'); memorizing which tokens are answers is impossible. This forces structural reading.
POOL = [f"w{i}" for i in range(400)]
READABLE = "Trent Cairo Lima Berlin poet sailor banker 1931 1955 1908 five three seven Maria".split()  # held-out, readable demo
POOL_TR, POOL_TE = POOL[:300], POOL[300:] + READABLE
FRAME = "was born in during worked as a had children where when what did do how many have ? .".split()


def gen_example(rng, test=False):
    """A passage about an entity (place/year/job/count facts) in RANDOM sentence order + (question, answer)
    pairs. All slot values are random generic tokens, so the answer can only be found by frame structure."""
    pool = POOL_TE if test else POOL_TR
    name, place, year, job, n = (pool[i] for i in rng.choice(len(pool), 5, replace=False))
    facts = {                                                    # structurally DISTINCT frames (in/during/as/had)
        "born_place": (f"{name} was born in {place} .", place, f"where was {name} born ?"),
        "born_year":  (f"{name} was born during {year} .", year,  f"when was {name} born ?"),
        "job":        (f"{name} worked as a {job} .", job,    f"what did {name} do ?"),
        "children":   (f"{name} had {n} children .", n,       f"how many children did {name} have ?"),
    }
    keys = list(facts); order = rng.permutation(len(keys))
    passage = " ".join(facts[keys[i]][0] for i in order)
    qa = [(facts[k][2].split(), facts[k][1]) for k in keys]
    return passage.split(), qa


def build_vocab():
    toks = {"[PAD]": 0, "[SEP]": 1}
    for w in POOL + READABLE + FRAME:
        if w not in toks:
            toks[w] = len(toks)
    return toks


class SpanReader(nn.Module):
    """Bidirectional transformer encoder + a per-position pointer score. Reads [question [SEP] passage] and
    scores every PASSAGE position; softmax over positions = where the answer is. Pure learned attention --
    no lexical rules."""
    def __init__(self, vocab, d=96, layers=2, heads=4, max_len=80):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.ptr = nn.Linear(d, 1)

    def forward(self, ids, pad_mask, pass_mask):
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        h = self.enc(x, src_key_padding_mask=~pad_mask)
        score = self.ptr(h).squeeze(-1)                          # (B, L) per-position pointer logit
        score = score.masked_fill(~pass_mask, -1e9)             # only passage positions are valid answers
        return score


def _encode(qtoks, ptoks, vocab, max_len=80):
    ids = [vocab.get(t, 0) for t in qtoks] + [vocab["[SEP]"]] + [vocab.get(t, 0) for t in ptoks]
    pstart = len(qtoks) + 1
    passpos = list(range(pstart, pstart + len(ptoks)))
    ids = ids[:max_len]
    return ids, pstart, passpos


def _batch(rng, vocab, bs, test=False, max_len=80):
    IDS, PAD, PASS, GOLD = [], [], [], []
    for _ in range(bs):
        ptoks, qa = gen_example(rng, test)
        qtoks, ans = qa[rng.integers(len(qa))]
        ids, pstart, passpos = _encode(qtoks, ptoks, vocab, max_len)
        gold = pstart + ptoks.index(ans)                        # absolute position of the answer token
        if gold >= len(ids):
            continue
        IDS.append(ids); GOLD.append(gold)
        PAD.append([1] * len(ids)); PASS.append([(pstart <= i < pstart + len(ptoks)) for i in range(len(ids))])
    L = max(len(x) for x in IDS)
    def pad(rows, v): return [r + [v] * (L - len(r)) for r in rows]
    return (torch.tensor(pad(IDS, 0)), torch.tensor(pad(PAD, 0), dtype=torch.bool),
            torch.tensor(pad(PASS, 0), dtype=torch.bool), torch.tensor(GOLD))


def train(steps=4000, d=96, seed=0, verbose=True):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    vocab = build_vocab(); model = SpanReader(len(vocab), d=d)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(steps):
        ids, pad, pas, gold = _batch(rng, vocab, 64)
        score = model(ids, pad, pas)
        loss = F.cross_entropy(score, gold)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % max(1, steps // 6) == 0:
            print(f"  step {step}: loss {loss.item():.3f}", flush=True)
    return model, vocab


def evaluate(model, vocab, n=400, test=True, seed=123):
    rng = np.random.default_rng(seed); model.eval(); ok = 0; tot = 0
    with torch.no_grad():
        for _ in range(n):
            ptoks, qa = gen_example(rng, test)
            for qtoks, ans in qa:
                ids, pstart, passpos = _encode(qtoks, ptoks, vocab)
                idt = torch.tensor([ids]); pad = torch.ones_like(idt, dtype=torch.bool)
                pas = torch.tensor([[(pstart <= i < pstart + len(ptoks)) for i in range(len(ids))]])
                pos = int(model(idt, pad, pas)[0].argmax())
                ok += (ptoks[pos - pstart] == ans); tot += 1
    return ok / tot


def answer(model, vocab, passage, question):
    """Answer a free question about a passage by pointing to the span. Returns the pointed token."""
    ptoks = passage.split(); qtoks = question.lower().replace("?", " ?").replace(".", " .").split()
    ids, pstart, _ = _encode(qtoks, ptoks, vocab)
    idt = torch.tensor([ids]); pad = torch.ones_like(idt, dtype=torch.bool)
    pas = torch.tensor([[(pstart <= i < pstart + len(ptoks)) for i in range(len(ids))]])
    with torch.no_grad():
        pos = int(model(idt, pad, pas)[0].argmax())
    return ptoks[pos - pstart]


def selftest():
    model, vocab = train(steps=1500, d=64, seed=0, verbose=False)
    seen = evaluate(model, vocab, n=200, test=False)
    held = evaluate(model, vocab, n=200, test=True)           # disjoint names/places/years -> structure, not memory
    print(f"qa selftest: train-pool {seen:.3f} | HELD-OUT pool {held:.3f} (chance ~0.25-0.5)")
    assert held > 0.85, f"held-out QA too low: {held}"
    print("qa selftest OK (learned to point to the answer span; generalizes to unseen tokens, no hardcoded rules)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    model, vocab = train(steps=a.steps)
    print(f"\nheld-out-pool QA accuracy: {evaluate(model, vocab, test=True):.3f}")
    # answer a fresh passage (unseen entity), to show it reads + answers
    p = "Maria was born in Lima . Maria worked as a banker . Maria had three children . Maria was born during 1955 ."
    print("\nPASSAGE (held-out entity/words, never seen in training): " + p)
    for q in ["where was Maria born ?", "what did Maria do ?", "how many children did Maria have ?", "when was Maria born ?"]:
        print(f"  Q: {q}\n  A: {answer(model, vocab, p, q)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
