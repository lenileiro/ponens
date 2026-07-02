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

# Each fact has MULTIPLE passage templates (paraphrase) and MULTIPLE question phrasings; a per-fact distinctive
# preposition (in / during / as / had) keeps place vs year vs job vs count answerable with random tokens, while
# everything else varies -> the model must learn the QUESTION->ROLE mapping, not a single frame.
TEMPLATES = {
    "place": (["{e} was born in {v} .", "{e} grew up in {v} .", "{e} settled in {v} ."],
              ["where was {e} born ?", "what is the birthplace of {e} ?", "where did {e} grow up ?"]),
    "year":  (["{e} was born during {v} .", "{e} was alive during {v} ."],
              ["when was {e} born ?", "in what year was {e} born ?", "when did {e} live ?"]),
    "job":   (["{e} worked as a {v} .", "{e} served as a {v} .", "{e} trained as a {v} ."],
              ["what did {e} do ?", "what was the job of {e} ?", "what profession did {e} have ?"]),
    "count": (["{e} had {v} children .", "{e} raised {v} children ."],
              ["how many children did {e} have ?", "what was the number of children of {e} ?"]),
}
FRAME = sorted({w for pt, qt in TEMPLATES.values() for s in pt + qt for w in s.split()
                if not w.startswith("{")} | {"?", "."})


def gen_example(rng, test=False, n_entities=1):
    """A passage about SEVERAL entities (each with place/year/job/count facts), sentences shuffled together +
    (question, answer) pairs. The question must be matched to the RIGHT entity AND the right fact frame; all
    values are random tokens so only structure + entity-matching can answer. Paraphrased questions/sentences."""
    pool = POOL_TE if test else POOL_TR
    idxs = rng.choice(len(pool), n_entities * 5, replace=False); toks = [pool[i] for i in idxs]
    sents, qa = [], []
    for e in range(n_entities):
        name = toks[e * 5]; vals = toks[e * 5 + 1:e * 5 + 5]
        for ft, v in zip(("place", "year", "job", "count"), vals):
            ptemps, qtemps = TEMPLATES[ft]
            sents.append(ptemps[rng.integers(len(ptemps))].format(e=name, v=v))
            qa.append((qtemps[rng.integers(len(qtemps))].format(e=name).split(), v))
    order = rng.permutation(len(sents))
    passage = " ".join(sents[i] for i in order)
    return passage.split(), qa


def build_vocab():
    toks = {"[PAD]": 0, "[SEP]": 1}
    for w in POOL + READABLE + FRAME:
        if w not in toks:
            toks[w] = len(toks)
    return toks


class SpanReader(nn.Module):
    """Bidirectional transformer encoder + a QUESTION-CONDITIONED pointer (pointer-network attention): the
    question is summarized into a query vector, and each PASSAGE position is scored by learned compatibility
    (query . key) with it. This lets the model COMPARE each candidate to what the question asks -- needed to
    bind the answer to the RIGHT entity in a multi-entity passage. Pure learned attention, no lexical rules."""
    def __init__(self, vocab, d=128, layers=4, heads=4, max_len=96):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.q_proj = nn.Linear(d, d); self.k_proj = nn.Linear(d, d); self.scale = d ** 0.5

    def forward(self, ids, pad_mask, pass_mask):
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        h = self.enc(x, src_key_padding_mask=~pad_mask)
        qmask = (pad_mask & ~pass_mask).float()[..., None]      # question tokens (incl SEP), not passage/pad
        qsum = (h * qmask).sum(1) / qmask.sum(1).clamp(min=1.0)  # (B, d) question summary
        q = self.q_proj(qsum)[:, None]; k = self.k_proj(h)      # (B,1,d), (B,L,d)
        score = (q * k).sum(-1) / self.scale                    # (B, L) question-conditioned pointer logit
        return score.masked_fill(~pass_mask, -1e9)              # only passage positions are valid answers


def _encode(qtoks, ptoks, vocab, max_len=96):
    ids = [vocab.get(t, 0) for t in qtoks] + [vocab["[SEP]"]] + [vocab.get(t, 0) for t in ptoks]
    pstart = len(qtoks) + 1
    passpos = list(range(pstart, pstart + len(ptoks)))
    ids = ids[:max_len]
    return ids, pstart, passpos


def _batch(rng, vocab, bs, test=False, max_len=96):
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
    model, vocab = train(steps=2500, d=96, seed=0, verbose=False)
    seen = evaluate(model, vocab, n=200, test=False)
    held = evaluate(model, vocab, n=200, test=True)           # disjoint pool -> structure+paraphrase, not memory
    print(f"qa selftest: train-pool {seen:.3f} | HELD-OUT pool {held:.3f} (chance ~0.3)")
    assert held > 0.85, f"held-out QA too low: {held}"
    print("qa selftest OK (learned to answer paraphrased questions on unseen tokens; no hardcoded rules)")
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
    p = "Maria settled in Lima . Maria served as a banker . Maria raised three children . Maria was alive during 1955 ."
    print("\nPASSAGE (held-out entity/words, never seen in training): " + p)
    for q in ["what is the birthplace of Maria ?", "what profession did Maria have ?",
              "how many children did Maria have ?", "in what year was Maria born ?"]:   # paraphrased questions
        print(f"  Q: {q}\n  A: {answer(model, vocab, p, q)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
