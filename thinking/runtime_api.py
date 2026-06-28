#!/usr/bin/env python3
"""Runtime text API -- the B2B product surface. Every call is PURE RUNTIME INFERENCE over raw text: zero training,
no per-task model, language-agnostic, stateless (statistics are computed per request from the inputs). One shared
reasoning core (recurrence/IDF + sentence relevance) powers the PUBLIC capabilities:

  classify(query, examples)      -> label        (nearest class over the caller's labeled examples; char n-grams)
  answer(question, passage)      -> answer span   (locate the span the question points at)
  summarize(text, k)             -> summary       (most central sentences)

INTERNAL component (NOT a public endpoint): context pruning -- `_prune_context(question, passage, tau)` drops
question-irrelevant sentences (the Provence task). It is used inside the pipeline to focus/reduce retrieved
context before answering or before handing context to a downstream generator; callers don't invoke it directly.

Nothing is trained on the caller's data; examples/passages live in the request. Works for any language/script.

  python -m thinking.runtime_api --selftest
"""
import argparse
import sys
from collections import Counter

import numpy as np

import thinking.langzero as Z
import thinking.squadqa as Q
import thinking.summ as SU
import thinking.prune as PR


def classify(query, examples):
    """Label `query` by nearest class over the caller's (text, label) examples -- char-n-gram TF-IDF (IDF from the
    examples), language-agnostic, zero training."""
    labels = sorted(set(l for _, l in examples))
    cnts = [(Z.ngram_counts(t), l) for t, l in examples]
    df = {}
    for c, _ in cnts:
        for b in c:
            df[b] = df.get(b, 0) + 1
    nsup = len(cnts)
    idf = {b: np.log((nsup + 1) / (v + 1)) + 1.0 for b, v in df.items()}
    def tfidf(c):
        v = np.zeros(Z.D, dtype=np.float32)
        for b, x in c.items():
            v[b] = x * idf.get(b, 1.0)
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v
    protos = {}
    for l in labels:
        protos[l] = np.mean([tfidf(c) for c, ll in cnts if ll == l], axis=0)
    qv = tfidf(Z.ngram_counts(query))
    return max(labels, key=lambda l: float(qv @ (protos[l] / (np.linalg.norm(protos[l]) + 1e-9))))


def _prune_context(question, passage, tau=0.4):
    """INTERNAL: drop sentences irrelevant to `question`, keeping the answer-bearing ones. Used to focus/reduce
    context before answering or before a downstream generator. Not a public endpoint."""
    ex = {"q": Q.toks(question), "c": Q.toks(passage), "golds": [""]}
    kept, *_ = PR.prune(ex, {}, 1.0, tau)
    return kept


def answer(question, passage, prune_first=True):
    """Extract the answer span `question` points at within `passage` (runtime span reasoning, per-passage IDF).
    Internally PRUNES the passage to the relevant sentences first (focuses reasoning, and reduces long/multi-passage
    contexts) -- prune is an internal stage, not called by the user."""
    if prune_first:
        focused = _prune_context(question, passage, tau=0.4)
        if focused.strip():
            passage = focused
    ex = {"q": Q.toks(question), "c": Q.toks(passage)}
    return Q.answer(ex, {}, 1.0)


def _doc_idf(sents_tok):
    df = Counter()
    for ws in sents_tok:
        for w in set(ws):
            df[w] += 1
    m = max(1, len(sents_tok))
    return {w: np.log((m + 1) / (c + 1)) + 1.0 for w, c in df.items()}, np.log(m + 1) + 1.0


def summarize(text, k=3):
    """Extractive summary: the k most central sentences (cosine to the document TF-IDF centroid)."""
    asents = SU.sents(text)
    if not asents:
        return ""
    idf, idf_default = _doc_idf([SU.toks(s) for s in asents])
    return SU.summarize_centroid(asents, idf, idf_default, min(k, len(asents)))


def selftest():
    # classify (any language / no training): label by the caller's examples
    ex = [("how do I cancel my order", "cancel"), ("stop my subscription please", "cancel"),
          ("where is my package", "track"), ("track my delivery", "track"),
          ("I want a refund", "refund"), ("give me my money back", "refund")]
    pred = classify("please cancel the order I made", ex)
    print("classify -> ", pred)
    assert pred == "cancel"

    passage = ("The Eiffel Tower is located in Paris. It was completed in 1889 for the World's Fair. "
               "The tower is made of wrought iron and stands 330 metres tall.")
    a = answer("When was the Eiffel Tower completed?", passage)
    print("answer   -> ", repr(a))
    assert "1889" in a

    s = summarize(passage, k=1)
    print("summarize-> ", repr(s[:80]))
    assert len(s) > 0

    # internal pruning (not a public call): focuses long context before answering
    focused = _prune_context("How tall is the tower?", passage, tau=0.5)
    print("[internal prune] -> ", repr(focused[:90]))
    assert "330" in focused and len(focused) < len(passage)

    print("runtime_api selftest OK (public: classify / answer / summarize; prune used INTERNALLY -- all "
          "zero-training runtime inference)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
