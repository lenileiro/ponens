#!/usr/bin/env python3
"""Extractive QA on SQuAD by PURE RUNTIME REASONING -- NO training, ever. (Per the project principle: a reasoning
model solves any prompt at runtime; it never trains on the dataset.)

For each (passage, question) we reason out the answer span: the answer is the stretch of text where the question's
informative words CLUSTER in the passage, but which is NOT itself made of question words (the answer is the new
information the question is pointing at). Word informativeness = IDF computed at runtime over the passage
collection (the recurrence principle -- common words like 'the' carry no signal; no hardcoded stoplist). No
answer-type rules (no 'when->date'), no model, no fitting -- just runtime scoring. Metric: SQuAD EM / token-F1.

  python -m thinking.squadqa --selftest
  python -m thinking.squadqa --dev /tmp/squad/dev-v1.1.json
"""
import argparse
import json
import re
import sys
from collections import Counter

import numpy as np

DEV = "/tmp/squad/dev-v1.1.json"


def toks(text):
    return re.findall(r"\w+|[^\w\s]", text)


def read_squad(path):
    data = json.load(open(path))["data"]
    out = []
    for art in data:
        for para in art["paragraphs"]:
            ct = toks(para["context"])
            for qa in para["qas"]:
                out.append({"q": toks(qa["question"]), "c": ct, "golds": [a["text"] for a in qa["answers"]]})
    return out


def build_idf(exs):
    """Runtime IDF over the passage collection (a corpus statistic, not training): common words -> low weight."""
    df = Counter()
    seen_ctx = set()
    docs = 0
    for e in exs:
        key = id(e["c"])
        if key in seen_ctx:
            continue
        seen_ctx.add(key); docs += 1
        for w in set(t.lower() for t in e["c"]):
            df[w] += 1
    return {w: np.log((docs + 1) / (c + 1)) + 1.0 for w, c in df.items()}, np.log(docs + 1) + 1.0


def answer(ex, idf, idf_default, max_len=8):
    """Runtime reasoning, no training: (1) pick the SENTENCE where the question's informative (high-IDF) words
    concentrate; (2) within it, the highest-IDF contiguous run of non-question content tokens (the answer the
    question points at), discounted by distance to the question's words. (Char-n-gram fuzzy matching was tried and
    slightly hurt -- loose matches add sentence/span-selection noise -- so exact IDF-weighted matching is used.)"""
    c = ex["c"]; cl = [t.lower() for t in c]
    qset = set(t.lower() for t in ex["q"])
    n = len(c)
    if n == 0:
        return ""
    def w(t):
        return idf.get(t, idf_default)
    word = [bool(re.match(r"\w", t)) for t in c]
    sents = []; start = 0
    for i, t in enumerate(c):
        if t in (".", "?", "!"):
            sents.append((start, i + 1)); start = i + 1
    if start < n:
        sents.append((start, n))
    if not sents:
        sents = [(0, n)]
    def sscore(se):
        s, e = se
        return sum(w(t) for t in set(cl[s:e]) if t in qset)
    bs, be = max(sents, key=sscore)
    qpos = [k for k in range(bs, be) if cl[k] in qset]
    best, bi, bj = -1.0, bs, bs
    i = bs
    while i < be:
        if word[i] and cl[i] not in qset:
            j = i
            while j < be and word[j] and cl[j] not in qset and (j - i) < max_len:
                j += 1
            mass = sum(w(cl[k]) for k in range(i, j))
            d = min((min(abs(i - p), abs(j - 1 - p)) for p in qpos), default=0)
            sc = mass / (1.0 + 0.25 * d)
            if sc > best:
                best, bi, bj = sc, i, j
            i = j
        else:
            i += 1
    span = c[bi:bj]
    while span and not re.match(r"\w", span[0]):
        span = span[1:]
    while span and not re.match(r"\w", span[-1]):
        span = span[:-1]
    return " ".join(span)


def _norm(s):
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def evaluate(exs, n=None):
    idf, idf_default = build_idf(exs)
    sub = exs[:n] if n else exs
    em = f1 = 0
    for e in sub:
        pred = answer(e, idf, idf_default)
        em += max(int(_norm(pred) == _norm(g)) for g in e["golds"])
        f1 += max(_f1(pred, g) for g in e["golds"])
    return em / len(sub), f1 / len(sub)


def selftest():
    exs = read_squad(DEV)
    em, f1 = evaluate(exs, n=1500)
    print(f"squadqa selftest: SQuAD dev (RUNTIME reasoning, ZERO training) -- EM {em:.3f} | token-F1 {f1:.3f} "
          f"(unsupervised sliding-window baseline range ~0.13-0.20)")
    assert f1 > 0.14, f"runtime span reasoning too weak: {f1}"
    print("squadqa selftest OK (extractive QA by pure runtime reasoning -- no training, no model, no hardcoded "
          "answer-type rules)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dev", default=DEV)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    exs = read_squad(a.dev)
    em, f1 = evaluate(exs, n=a.n)
    print(f"SQuAD dev ({len(exs[:a.n] if a.n else exs)} q, RUNTIME reasoning, zero training): "
          f"EM {em:.3f} | token-F1 {f1:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
