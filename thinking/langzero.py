#!/usr/bin/env python3
"""LANGUAGE-AGNOSTIC zero-shot text classification for the B2B API: no word vocabulary, no language assumption, no
training on customer data. Classify a query by nearest class over the customer's IN-PROMPT examples, using
CHARACTER N-GRAM features (sub-word units that exist in any language/script) with IDF estimated FROM THE PROMPT
itself (the recurrence principle, computed per request -- common n-grams downweighted, no hardcoded stoplist).

Why char n-grams: a word vocabulary assumes a known language; characters/bytes do not. Char n-grams capture
morphology and are robust to typos, and -- crucially -- the method is INVARIANT to the alphabet: apply any
consistent character substitution (an unknown script) and the n-gram overlap structure is preserved, so accuracy
is unchanged. We verify exactly that.

  python -m thinking.langzero --selftest
  python -m thinking.langzero            # Bitext English + character-permuted (unknown-script) -- same accuracy
"""
import argparse
import csv
import sys
import zlib

import numpy as np

DATA = "kaggle_data/bitext_customer_support.csv"
D = 8192                                                    # hashed char-n-gram feature dimension
NS = (3, 4, 5)                                              # character n-gram sizes


def ngram_counts(text, ns=NS):
    """Hashed character n-gram counts -- pure structure, no words, no language. Returns {bucket: count}."""
    s = " " + text + " "                                    # boundary marker; no lowercasing (stays language-agnostic)
    c = {}
    for n in ns:
        for i in range(len(s) - n + 1):
            b = zlib.crc32(s[i:i + n].encode("utf-8")) % D
            c[b] = c.get(b, 0.0) + 1.0
    return c


def _dense(counts):
    v = np.zeros(D, dtype=np.float32)
    for b, x in counts.items():
        v[b] = x
    return v


def load(path=DATA, kmin=8):
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["intent"], []).append(r["instruction"])
    return {i: v for i, v in by.items() if len(v) >= kmin}


def _perm_map(seed=0):
    """A consistent random substitution over printable characters -- simulates an UNKNOWN language/script."""
    rng = np.random.default_rng(seed)
    chars = [chr(c) for c in range(32, 127)]
    shuf = list(chars); rng.shuffle(shuf)
    return str.maketrans("".join(chars), "".join(shuf))


def zeroshot(by, C=5, k=5, n=400, seed=0, perm=None):
    """ZERO-SHOT: k support examples per class -> TF-IDF (IDF from the prompt's support set) class prototype; query
    -> nearest cosine prototype. perm applies a character substitution to ALL text (unknown-script test)."""
    intents = list(by); rng = np.random.default_rng(seed); ok = 0
    tr = perm

    def cnt(t):
        return ngram_counts(t.translate(tr) if tr else t)

    for _ in range(n):
        chosen = list(rng.choice(intents, C, replace=False))
        sup_counts = [[] for _ in range(C)]; q_counts = []; qy = []
        df = {}
        for ci, it in enumerate(chosen):
            idx = rng.choice(len(by[it]), k + 1, replace=False)
            for j in idx[:k]:
                c = cnt(by[it][j]); sup_counts[ci].append(c)
                for b in c:
                    df[b] = df.get(b, 0) + 1
            q_counts.append(cnt(by[it][idx[k]])); qy.append(ci)
        nsup = C * k
        idf = {b: np.log((nsup + 1) / (v + 1)) + 1.0 for b, v in df.items()}

        def tfidf(c):
            v = np.zeros(D, dtype=np.float32)
            for b, x in c.items():
                v[b] = x * idf.get(b, 1.0)
            nrm = np.linalg.norm(v)
            return v / nrm if nrm > 0 else v

        P = np.stack([np.mean([tfidf(c) for c in sup_counts[ci]], axis=0) for ci in range(C)])
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        Qm = np.stack([tfidf(c) for c in q_counts])
        pred = (Qm @ P.T).argmax(1)
        ok += int((pred == np.array(qy)).sum())
    return ok / (n * C)


def selftest():
    by = load()
    eng = zeroshot(by, C=5, k=5, n=200, seed=0)
    perm = zeroshot(by, C=5, k=5, n=200, seed=0, perm=_perm_map(1))
    print(f"langzero selftest: ZERO-SHOT Bitext 5-way (chance 0.20, NO training, NO word vocab) -- "
          f"English {eng:.3f} | character-permuted/unknown-script {perm:.3f}")
    assert eng > 0.75, f"char-n-gram zero-shot too low: {eng}"
    assert abs(eng - perm) < 0.05, f"NOT language-agnostic: english {eng:.3f} vs permuted {perm:.3f}"
    print("langzero selftest OK (char-n-gram TF-IDF retrieval: strong zero-shot AND alphabet-invariant -> works "
          "for any language/script, no word training)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    by = load(a.data)
    print("ZERO-SHOT, language-agnostic (char n-grams, IDF from prompt, no training, no word vocab):")
    perm = _perm_map(1)
    for C in (3, 5, 10):
        e = zeroshot(by, C=C, k=5, n=300, seed=0)
        p = zeroshot(by, C=C, k=5, n=300, seed=0, perm=perm)
        print(f"  {C:2d}-way (chance {1/C:.3f}): English {e:.3f} | unknown-script(permuted) {p:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
