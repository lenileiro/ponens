#!/usr/bin/env python3
"""AGENT-EDITABLE comprehension strategy (the only file an agent modifies -- like autoresearch's train.py).

Improve propose() to RAISE the harness metric (exact-parent over all held). Rules (enforced by the
harness): you see ONLY the concept's `gloss` plus the READ-ONLY `brain`; you may NOT access the held
concept's true parent/ancestors. The brain VERIFIES every proposal (keeps the provable ancestors, picks
the most specific), so rank likely true ancestors high AND surface more of them (coverage).

In-house tools only (WordNet dictionary + Porter stemmer -- no pretrained LM). brain API:
  brain.candidates / brain.parents_with_word(word) / brain.lemmas(name) / brain.gloss_of(name) /
  brain.depth(name) / brain.toks(text)
"""
from collections import Counter
from functools import lru_cache

NAME = "lexical + gloss-overlap + Porter-stem + synonyms"

_GIDX = _STEMIDX = None
_STOP = set("a an the of or and to in is are was for with that this these those as by from on at it its "
            "their his her any some such other being been have has had not no which who whose various "
            "esp especially used relating".split())


def _ps():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


def _build(brain):
    ps = _ps()
    g, stem = {}, {}
    for c in brain.candidates:
        for w in brain.toks(brain.gloss_of(c)):
            g.setdefault(w, set()).add(c)
        for lem in brain.lemmas(c):
            for tok in lem.split():
                if len(tok) >= 3:
                    stem.setdefault(ps.stem(tok), set()).add(c)
    g = {w: s for w, s in g.items() if w not in _STOP and len(s) <= 300}
    return g, stem


@lru_cache(maxsize=None)
def _syns(word):
    from nltk.corpus import wordnet as wn
    return {l.replace("_", " ").lower() for s in wn.synsets(word) for l in s.lemma_names()}


def propose(gloss, brain):
    global _GIDX, _STEMIDX
    if _GIDX is None:
        _GIDX, _STEMIDX = _build(brain)
    ps = _ps()
    words = brain.toks(gloss)
    # (1) NAME match in the gloss ('a breed of DOG' -> dog), ranked most-specific first
    named = set()
    for w in words:
        named |= brain.parents_with_word(w)
    ranked = sorted(named, key=lambda c: -brain.depth(c))
    # (2) GLOSS-OVERLAP: parents whose own gloss overlaps this gloss (+coverage)
    cnt = Counter()
    for w in words:
        for c in _GIDX.get(w, ()):
            cnt[c] += 1
    ov = sorted([c for c, k in cnt.most_common(40) if k >= 2 and c not in named],
                key=lambda c: (-cnt[c], -brain.depth(c)))
    # (3) MORPHOLOGY (Porter stem): catch 'running'~'run', 'canine'~'canid' (+coverage)
    stem = set()
    for w in words:
        if len(w) >= 3:
            stem |= _STEMIDX.get(ps.stem(w), set())
    stem = sorted(stem - named, key=lambda c: -brain.depth(c))[:25]
    # (4) SYNONYMS: expand gloss words to WordNet synonyms, match parent names (+coverage)
    syn = set()
    for w in words:
        for sw in _syns(w):
            syn |= brain.parents_with_word(sw)
    syn = sorted(syn - named, key=lambda c: -brain.depth(c))
    return ranked + ov + stem + syn
