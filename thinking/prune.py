#!/usr/bin/env python3
"""Runtime CONTEXT PRUNING for RAG (the Provence task -- https://huggingface.co/blog/nadiinchi/provence -- but
with ZERO training and language-agnostic). Given a question + retrieved passage, keep only the relevant sentences
(drop the noise) so a downstream LLM gets a short, clean context. Provence trains a DeBERTa cross-encoder on
synthetic Llama labels; we do the SAME task by pure RUNTIME reasoning: score each sentence's relevance to the
question (passage-discriminative IDF + rarest distinctive match -- the recurrence principle, no stoplist), then
auto-threshold relative to the top sentence (no fixed k). No model, no training, any language.

Eval like Provence: COMPRESSION (fraction of context removed) vs ANSWER-RETENTION (is the answer still present
after pruning?) on SQuAD. A good pruner removes most of the context while keeping the answer.

  python -m thinking.prune --selftest
  python -m thinking.prune --dev /tmp/squad/dev-v1.1.json
"""
import argparse
import sys
from collections import Counter

import numpy as np

from thinking import squadqa as Q


def _sentences(ctoks):
    return Q.sentences(ctoks)                               # abbreviation-aware shared splitter


def relevance(ex, idf, idf_default):
    """Per-sentence relevance to the question: passage-discriminative IDF (a word in many sentences of this passage
    is non-discriminative) x global IDF, scored by the rarest distinctive matched word. Returns (sentences, scores)."""
    c = ex["c"]; cl = [t.lower() for t in c]
    qset = set(t.lower() for t in ex["q"])
    sents = _sentences(c)
    msent = len(sents); sdf = Counter()
    for s, e in sents:
        for t in set(cl[s:e]):
            sdf[t] += 1
    def w(t):
        return (np.log((msent + 1) / (sdf.get(t, 0) + 1)) + 1.0) * idf.get(t, idf_default)
    scores = []
    for s, e in sents:
        matched = [w(t) for t in set(cl[s:e]) if t in qset]
        scores.append((max(matched) + 0.3 * sum(matched)) if matched else 0.0)
    return sents, scores


def prune(ex, idf, idf_default, tau=0.5):
    """Keep sentences scoring >= tau * top-score (auto-adapts how many; >=1 always kept)."""
    sents, scores = relevance(ex, idf, idf_default)
    mx = max(scores) if scores else 0.0
    keep = [i for i, sc in enumerate(scores) if mx > 0 and sc >= tau * mx]
    if not keep:
        keep = [int(np.argmax(scores))] if scores else []
    kset = set(keep)
    kept = " ".join(" ".join(ex["c"][s:e]) for i, (s, e) in enumerate(sents) if i in kset)
    n_tot = sum(e - s for s, e in sents)
    n_kept = sum(e - s for i, (s, e) in enumerate(sents) if i in kset)
    return kept, n_kept, n_tot, len(keep), len(sents)


def evaluate(exs, tau, n=None):
    idf, idf_default = Q.build_idf(exs)
    sub = exs[:n] if n else exs
    comp = ret = 0.0; ksent = tsent = 0
    for e in sub:
        kept, nk, nt, nks, nts = prune(e, idf, idf_default, tau)
        comp += 1.0 - nk / max(1, nt)
        ret += int(any(Q._norm(g) in Q._norm(kept) for g in e["golds"]))
        ksent += nks; tsent += nts
    m = len(sub)
    return {"tau": tau, "compression": comp / m, "answer_retention": ret / m,
            "kept_sents": ksent / m, "total_sents": tsent / m}


def selftest():
    exs = Q.read_squad(Q.DEV)
    print("prune selftest: runtime RAG context pruning on SQuAD (ZERO training) -- compression vs answer-retention")
    rows = [evaluate(exs, tau, n=1500) for tau in (0.3, 0.5, 0.7)]
    for r in rows:
        print(f"  tau {r['tau']}: removed {r['compression']*100:4.1f}% of context | answer kept "
              f"{r['answer_retention']*100:4.1f}% | sents {r['kept_sents']:.1f}/{r['total_sents']:.1f}")
    mid = rows[1]
    assert mid["compression"] > 0.4 and mid["answer_retention"] > 0.75, f"poor compression/retention tradeoff: {mid}"
    print("prune selftest OK (runtime context pruning: cuts most of the passage while keeping the answer; "
          "no training, language-agnostic)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dev", default=Q.DEV)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    exs = Q.read_squad(a.dev)
    print(f"RAG context pruning on SQuAD ({len(exs[:a.n] if a.n else exs)} q, RUNTIME, zero training):")
    for tau in (0.2, 0.4, 0.6, 0.8):
        r = evaluate(exs, tau, n=a.n)
        print(f"  tau {tau}: removed {r['compression']*100:4.1f}% of context | answer retained "
              f"{r['answer_retention']*100:4.1f}% | avg {r['kept_sents']:.1f}/{r['total_sents']:.1f} sentences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
