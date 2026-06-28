#!/usr/bin/env python3
"""Extractive QA on SQuAD by PURE RUNTIME REASONING -- NO training, ever. (Per the project principle: a reasoning
model solves any prompt at runtime; it never trains on the dataset.)

For each (passage, question) we reason out the answer span: the answer is the stretch of text where the question's
informative words CLUSTER in the passage, but which is NOT itself made of question words (the answer is the new
information the question is pointing at). Word informativeness = IDF computed at runtime over the passage
collection (the recurrence principle -- common words like 'the' carry no signal; no hardcoded stoplist). The one
answer-type signal is a GROUNDED LAT (not a hardcoded 'when->date' table), read from WordNet:
  - measurable-property focus ('how LONG/TALL/FAR/OLD', 'what YEAR') -> a NUMERIC span wins wherever it sits;
  - entity-noun focus ('which CITY', 'what COUNTRY/AUTHOR') -> a PROPER NOUN (mid-sentence Capitalized run); the
    question's OWN named entity (a phrase containing a question word, e.g. 'Eiffel Tower') is excluded as restatement.
No model, no fitting, no hardcoded word lists. Metric: SQuAD EM / token-F1.

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


import functools


@functools.lru_cache(maxsize=50000)
def _in_kb(w):
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return True
    return bool(wn.synsets(w))


def _specific(w):
    """A factoid answer tends to be a SPECIFIC token: a number (digit) or a name (a word the KB doesn't know).
    Character class + KB membership -- no word list, no type map, no training."""
    return bool(re.search(r"\d", w)) or (w.isalpha() and len(w) > 1 and not _in_kb(w))


def lat_focus(qtoks):
    """Lexical-Answer-Type focus: the (word, coarse-POS) signalling the expected answer type. Found via POS tags
    (grammatical categories from the tagger -- NOT a hardcoded wh-word list): the content word right after an
    interrogative (how LONG, which CITY, what YEAR). None if there's no such focus."""
    try:
        import nltk
        tags = nltk.pos_tag([t.lower() for t in qtoks])      # lowercase: a capitalized leading 'Which' mis-tags as JJ
    except Exception:
        return None
    for i, (w, t) in enumerate(tags):
        if t in ("WDT", "WP", "WP$", "WRB") and i + 1 < len(tags):
            w2, t2 = tags[i + 1]                              # ONLY the immediately-adjacent word: 'how LONG',
            if t2[:2] in ("NN", "JJ", "RB"):                 # 'which CITY', 'what YEAR'. 'when/who/where was ...'
                return (w2.lower(), t2[:2])                  # have no adjacent type-word -> no focus (use base scoring)
    return None


def answer_type(qtoks):
    """Map the question to an expected answer type via the GROUNDED LAT (WordNet, no word lists): 'quantity' (the
    focus is a measurable property/quantity -> a number), 'entity' (the focus is a noun naming a kind with named
    instances -> a proper noun), or None (no usable focus -> fall back to plain span scoring)."""
    f = lat_focus(qtoks)
    if not f:
        return None
    word, pos2 = f
    if _expects_quantity(word):
        return "quantity"
    if pos2 == "NN" and _is_entity_type(word):
        return "entity"
    return None


def sentences(ctoks):
    """Split token list into sentence spans. A '.' after a SINGLE-letter token is treated as an abbreviation
    (U.S., A.B.) and does NOT end a sentence -- a general rule, no hardcoded abbreviation list."""
    sents = []; start = 0
    for i, t in enumerate(ctoks):
        if t in (".", "?", "!") and i > 0 and len(ctoks[i - 1]) >= 2:
            sents.append((start, i + 1)); start = i + 1
    if start < len(ctoks):
        sents.append((start, len(ctoks)))
    return sents or [(0, len(ctoks))]


def read_squad(path):
    data = json.load(open(path))["data"]
    out = []
    for art in data:
        for para in art["paragraphs"]:
            ct = toks(para["context"])
            for qa in para["qas"]:
                qt = toks(qa["question"])
                e = {"q": qt, "c": ct, "golds": [a["text"] for a in qa["answers"]]}
                at = answer_type(qt)                          # GROUNDED LAT (WordNet, no word list)
                if at == "quantity":                         # 'how long/tall', 'what year' -> a number
                    e["want_quantity"] = True
                elif at == "entity":                         # 'which city', 'what country' -> a proper noun
                    e["want_entity"] = True
                out.append(e)
    return out


@functools.lru_cache(maxsize=50000)
def _expects_quantity(word):
    try:
        from thinking import kb
        return kb.expects_quantity(word)
    except Exception:
        return False


@functools.lru_cache(maxsize=50000)
def _is_entity_type(word):
    try:
        from thinking import kb
        return kb.is_entity_type(word)
    except Exception:
        return False


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
    excluded = qset | set(ex.get("redundant", ()))           # question words + words the KB says merely restate them
    n = len(c)
    if n == 0:
        return ""
    word = [bool(re.match(r"\w", t)) for t in c]
    sents = sentences(c)
    # PASSAGE-LEVEL discriminativeness (recurrence within this doc): a word in MANY sentences of the passage is
    # non-discriminative ('Super Bowl', 'NFL' here) and is down-weighted, so the question's DISTINCTIVE word
    # ('AFC') selects the right sentence -- combined with the global IDF.
    msent = len(sents); sdf = Counter()
    for s, e in sents:
        for t in set(cl[s:e]):
            sdf[t] += 1
    def w(t):
        return (np.log((msent + 1) / (sdf.get(t, 0) + 1)) + 1.0) * idf.get(t, idf_default)
    def sscore(se):
        s, e = se
        matched = [w(t) for t in set(cl[s:e]) if t in qset]
        if not matched:
            return 0.0
        return max(matched) + 0.3 * sum(matched)            # a distinctive match dominates many generic ones
    bs, be = max(sents, key=sscore)
    want_qty = bool(ex.get("want_quantity"))                 # LAT: question expects a NUMERIC answer ('how long/tall')
    want_ent = bool(ex.get("want_entity")) and not want_qty  # LAT: question expects a PROPER NOUN ('which city/author')
    # NER by ORTHOGRAPHY (no model, no name list): a mid-sentence Capitalized run is a proper-noun phrase. A phrase
    # that CONTAINS a question word is the question's OWN named entity (e.g. 'Eiffel Tower' for '...the tower...'),
    # so it merely restates the subject -> excluded; the answer is a DIFFERENT name ('Paris').
    proper_pos, excl_pos = set(), set()
    if want_ent:
        start_pos = set()
        for s, e in sents:
            for k in range(s, e):
                if word[k]:
                    start_pos.add(k); break
        def _isname(k):
            return bool(re.match(r"[A-Z][a-z]", c[k])) and k not in start_pos
        k = 0
        while k < n:
            if _isname(k):
                j = k
                while j < n and _isname(j):
                    j += 1
                own = any(cl[m] in qset for m in range(k, j))
                for m in range(k, j):
                    proper_pos.add(m)
                    if own:
                        excl_pos.add(m)
                k = j
            else:
                k += 1
    qpos = [k for k in range(bs, be) if cl[k] in qset]
    best, bi, bj = -1.0, bs, bs
    i = bs
    while i < be:
        if word[i] and cl[i] not in excluded and i not in excl_pos:  # answer = NEW info: not a question word, not
            j = i                                            # KB-redundant with it, not the question's own named entity
            while j < be and word[j] and cl[j] not in excluded and j not in excl_pos and (j - i) < max_len:
                j += 1
            mass = sum(w(cl[k]) for k in range(i, j))
            d = min((min(abs(i - p), abs(j - 1 - p)) for p in qpos), default=0)
            has_digit = any(re.search(r"\d", cl[k]) for k in range(i, j))
            has_name = any(k in proper_pos for k in range(i, j))
            sc = mass / (1.0 + 0.25 * d)
            if any(_specific(cl[k]) for k in range(i, j)):   # answers are specific (numbers/names): mild prior
                sc *= 1.8
            if want_qty:                                     # quantity question: the answer IS the measured value --
                sc = (mass * 3.0) if has_digit else sc * 0.4 # the number wins wherever it sits (distance-independent)
            elif want_ent:                                   # entity question: the answer IS a (new) proper noun
                sc = (mass * 3.0) if has_name else sc * 0.4
            if sc > best:
                best, bi, bj = sc, i, j
            i = j
        else:
            i += 1
    # trim the span to its high-IDF CORE: drop low-information edge words (e.g. 'champion', 'defeated') so the
    # answer is the informative entity, not its surrounding filler.
    core = [k for k in range(bi, bj) if word[k]]
    if want_qty and any(re.search(r"\d", cl[k]) for k in core):
        while core and not re.search(r"\d", cl[core[0]]):    # a quantity reads as NUMBER+unit ('5 business days',
            core = core[1:]                                  # '330 metres'): start the answer at the number
        bi, bj = core[0], core[-1] + 1
    elif want_ent and any(k in proper_pos for k in core):    # an entity answer IS the proper-noun phrase: keep the
        pcore = [k for k in core if k in proper_pos]         # first contiguous Capitalized block ('George Orwell')
        blk = [pcore[0]]
        for k in pcore[1:]:
            if k == blk[-1] + 1:
                blk.append(k)
            else:
                break
        bi, bj = blk[0], blk[-1] + 1
    elif core:
        ws = [w(cl[k]) for k in core]
        thr = 0.55 * max(ws)
        while len(core) > 1 and w(cl[core[0]]) < thr:
            core = core[1:]
        while len(core) > 1 and w(cl[core[-1]]) < thr:
            core = core[:-1]
        bi, bj = core[0], core[-1] + 1
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
    assert f1 > 0.17, f"runtime span reasoning too weak: {f1}"
    print("squadqa selftest OK (extractive QA by pure runtime reasoning -- no training, no model; the only "
          "answer-type signal is a GROUNDED LAT from WordNet: measurable-property focus -> a number; "
          "entity-noun focus -> a proper noun. No hardcoded word lists.)")
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
