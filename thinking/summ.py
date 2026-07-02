#!/usr/bin/env python3
"""Extractive summarization by PURE RUNTIME REASONING -- NO training. Pick the most CENTRAL sentences (closest to
the document's TF-IDF centroid -- the recurrence principle: a sentence is central if it shares informative,
non-common words with the rest of the document). Non-classification, generative-by-selection; scored by ROUGE.

Runtime, no model, no fitting, no labels. Compared against the lead-k baseline (first k sentences -- strong for
news). IDF is a runtime corpus statistic over the article collection.

  python -m thinking.summ --selftest
  python -m thinking.summ --base "/tmp/bbc/BBC News Summary"
"""
import argparse
import os
import re
import sys
from collections import Counter

import numpy as np

BASE = "/tmp/bbc/BBC News Summary"


def sents(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(s.strip()) > 10]


def toks(s):
    return re.findall(r"\w+", s.lower())


def read_pairs(base, cap=None):
    pairs = []
    art_root = os.path.join(base, "News Articles"); sum_root = os.path.join(base, "Summaries")
    for cat in sorted(os.listdir(art_root)):
        ad = os.path.join(art_root, cat); sd = os.path.join(sum_root, cat)
        if not os.path.isdir(ad):
            continue
        for fn in sorted(os.listdir(ad)):
            ap, sp = os.path.join(ad, fn), os.path.join(sd, fn)
            if not os.path.exists(sp):
                continue
            art = open(ap, encoding="utf-8", errors="ignore").read()
            ref = open(sp, encoding="utf-8", errors="ignore").read()
            asents = sents(art)
            if len(asents) >= 3:
                pairs.append((asents, ref))
            if cap and len(pairs) >= cap:
                return pairs
    return pairs


def build_idf(pairs):
    df = Counter(); n = 0
    for asents, _ in pairs:
        n += 1
        words = set(w for s in asents for w in toks(s))
        for w in words:
            df[w] += 1
    return {w: np.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}, np.log(n + 1) + 1.0


def _vec(s, idf, idf_default):
    v = Counter(toks(s))
    return {w: c * idf.get(w, idf_default) for w, c in v.items()}


def _cos(a, b):
    if not a or not b:
        return 0.0
    dot = sum(a[w] * b.get(w, 0.0) for w in a)
    na = np.sqrt(sum(x * x for x in a.values())); nb = np.sqrt(sum(x * x for x in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def summarize_centroid(asents, idf, idf_default, k):
    """Central sentences = highest cosine to the document TF-IDF centroid. Returns k sentences in original order."""
    vecs = [_vec(s, idf, idf_default) for s in asents]
    centroid = Counter()
    for v in vecs:
        for w, x in v.items():
            centroid[w] += x
    scores = [(_cos(v, centroid), i) for i, v in enumerate(vecs)]
    pick = sorted(i for _, i in sorted(scores, reverse=True)[:k])
    return " ".join(asents[i] for i in pick)


def lead(asents, k):
    return " ".join(asents[:k])


def _ngrams(ws, n):
    return Counter(tuple(ws[i:i + n]) for i in range(len(ws) - n + 1))


def rouge_n(pred, ref, n):
    p, r = _ngrams(toks(pred), n), _ngrams(toks(ref), n)
    if not p or not r:
        return 0.0
    same = sum((p & r).values())
    if same == 0:
        return 0.0
    prec, rec = same / max(1, sum(p.values())), same / max(1, sum(r.values()))
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred, ref):
    a, b = toks(pred), toks(ref)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(a)][len(b)]
    prec, rec = lcs / len(a), lcs / len(b)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def evaluate(pairs, n=None):
    idf, idf_default = build_idf(pairs)
    sub = pairs[:n] if n else pairs
    agg = {"centroid": [0.0, 0.0, 0.0], "lead": [0.0, 0.0, 0.0]}
    for asents, ref in sub:
        k = max(1, min(len(asents), len(sents(ref))))         # match the reference length (fair)
        for name, pred in (("centroid", summarize_centroid(asents, idf, idf_default, k)), ("lead", lead(asents, k))):
            agg[name][0] += rouge_n(pred, ref, 1); agg[name][1] += rouge_n(pred, ref, 2); agg[name][2] += rouge_l(pred, ref)
    m = len(sub)
    return {k: [v / m for v in vs] for k, vs in agg.items()}


def selftest():
    pairs = read_pairs(BASE, cap=400)
    res = evaluate(pairs)
    c, l = res["centroid"], res["lead"]
    print(f"summ selftest: extractive summarization (RUNTIME, zero training) on {len(pairs)} BBC articles")
    print(f"  centroid-centrality : ROUGE-1 {c[0]:.3f} | ROUGE-2 {c[1]:.3f} | ROUGE-L {c[2]:.3f}")
    print(f"  lead-k baseline     : ROUGE-1 {l[0]:.3f} | ROUGE-2 {l[1]:.3f} | ROUGE-L {l[2]:.3f}")
    assert c[0] > 0.3, f"centroid ROUGE-1 too low: {c[0]}"
    assert c[0] > l[0] and c[2] > l[2], f"centrality did not beat the lead baseline: {c} vs {l}"
    print("summ selftest OK (runtime extractive summarization via centroid centrality BEATS lead-k; zero training)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    pairs = read_pairs(a.base)
    res = evaluate(pairs, n=a.n)
    print(f"BBC extractive summarization ({len(pairs)} articles, RUNTIME, zero training):")
    for name in ("centroid", "lead"):
        v = res[name]
        print(f"  {name:9s}: ROUGE-1 {v[0]:.3f} | ROUGE-2 {v[1]:.3f} | ROUGE-L {v[2]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
