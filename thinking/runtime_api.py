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

import re

import thinking.langzero as Z
import thinking.squadqa as Q
import thinking.summ as SU
import thinking.prune as PR
import thinking.kb as KB


def _kb_expand(text):
    """Expand text with KB synonyms so matching bridges MEANING (car~automobile), not just surface. Runtime KB
    lookup -- no pretraining, no hardcoded words."""
    ws = re.findall(r"[a-z]+", text.lower())
    extra = set()
    for w in ws:
        extra |= {s for s in KB.synonyms(w) if " " not in s}
    return text + " " + " ".join(extra)


def classify(query, examples, use_kb=False):
    """Label `query` by nearest class over the caller's (text, label) examples -- char-n-gram TF-IDF (IDF from the
    examples), language-agnostic, zero training. use_kb=True consults the runtime knowledge base to also match by
    MEANING (synonyms), bridging paraphrases that share no characters."""
    labels = sorted(set(l for _, l in examples))
    prep = _kb_expand if use_kb else (lambda t: t)
    cnts = [(Z.ngram_counts(prep(t)), l) for t, l in examples]
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
    qv = tfidf(Z.ngram_counts(prep(query)))
    return max(labels, key=lambda l: float(qv @ (protos[l] / (np.linalg.norm(protos[l]) + 1e-9))))


def _prune_context(question, passage, tau=0.4):
    """INTERNAL: drop sentences irrelevant to `question`, keeping the answer-bearing ones. Used to focus/reduce
    context before answering or before a downstream generator. Not a public endpoint."""
    ex = {"q": Q.toks(question), "c": Q.toks(passage), "golds": [""]}
    kept, *_ = PR.prune(ex, {}, 1.0, tau)
    return kept


def _build_ex(question, passage, prune_first=True, use_kb=True):
    if prune_first:
        focused = _prune_context(question, passage, tau=0.4)
        if focused.strip():
            passage = focused
    qtoks = Q.toks(question)
    ex = {"q": qtoks, "c": Q.toks(passage)}
    if use_kb:
        redundant = set()
        for w in qtoks:
            if re.match(r"[a-z]", w.lower()):
                redundant |= {s for s in KB.related(w.lower()) if " " not in s}
        ex["redundant"] = redundant
        at = Q.answer_type(qtoks)                            # LAT: 'quantity'/'time' -> number; 'person'/'location'/
        if at:                                              # 'group'/'entity'/'isa' -> proper noun / is-a candidate
            ex["want_type"] = at
            foc = Q.lat_focus(qtoks)                          # focus NOUN -> is-a matching + POS-agnostic candidates
            if foc and foc[1] == "NN" and at != "quantity":
                ex["focus_word"] = foc[0]
    return ex


def answer(question, passage, prune_first=True, use_kb=True, model=None):
    """Extract the answer span `question` points at within `passage` (runtime span reasoning, per-passage IDF).
    Internally PRUNES the passage to the relevant sentences first. use_kb consults the runtime KB to exclude words
    that merely RESTATE the question (synonyms/hypernyms of question words, e.g. 'payment' for 'refund') -- the
    answer is the NEW information, so those redundant words are not candidate answers. No hardcoding, no training.

    `model` (optional): a pre-trained reranker loaded via squad_rank.load_model -- trained ONCE on public data and
    shipped as weights; if given, it scores the candidates instead of the hand-tuned formula (the customer never
    trains). None -> the pure rule pipeline."""
    ex = _build_ex(question, passage, prune_first, use_kb)
    if model is not None:
        from thinking import squad_rank as R
        return R.predict(ex, model, {}, 1.0)
    return Q.answer(ex, {}, 1.0)


def explain(question, passage, prune_first=True, use_kb=True):
    """Like `answer`, but returns (answer, trace) -- an AUDIT TRAIL of why: the sentence chosen, the grounded
    answer-type the question was mapped to, and which signals the winning span satisfied (number / proper-noun /
    is-a-the-focus / supersense-match / passive-agent). This interpretability is the differentiator over an LLM."""
    tr = {}
    ans = Q.answer(_build_ex(question, passage, prune_first, use_kb), {}, 1.0, trace=tr)
    fired = [k for k, v in tr.get("signals", {}).items() if v]
    return ans, {"answer": ans, "expected_type": tr.get("answer_type"), "focus": tr.get("focus_word"),
                 "chosen_sentence": tr.get("sentence"), "why": fired or ["nearest high-IDF noun phrase (no type signal)"]}


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
