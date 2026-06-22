#!/usr/bin/env python3
"""AGENT-EDITABLE comprehension strategy (the only file an agent modifies -- like autoresearch's train.py).

Improve propose() to RAISE the harness metric (exact-parent over all held). Rules (enforced by the
harness): you see ONLY the concept's `gloss` plus the READ-ONLY `brain`; you may NOT access the held
concept's true parent or ancestors. The brain VERIFIES your proposals (keeps the provable ancestors and
picks the most specific), so rank likely true ancestors high AND surface more of them (coverage).

brain API:  brain.candidates                -> all candidate parent synset-names
            brain.parents_with_word(word)    -> candidate parents whose lemma == word
            brain.lemmas(name) / brain.gloss_of(name) / brain.depth(name)
            brain.toks(text)                 -> set of lowercase word tokens
"""
from collections import Counter

NAME = "lexical + gloss-overlap"

_GIDX = None            # inverted index: gloss-word -> candidate parents whose GLOSS contains it (cached)
_STOP = set("a an the of or and to in is are was for with that this these those as by from on at it "
            "its their his her any some such other being been have has had not no which who whose".split())


def _gloss_index(brain):
    idx = {}
    for c in brain.candidates:
        for w in brain.toks(brain.gloss_of(c)):
            idx.setdefault(w, set()).add(c)
    return {w: s for w, s in idx.items() if w not in _STOP and len(s) <= 300}   # drop common/huge postings


def propose(gloss, brain):
    global _GIDX
    if _GIDX is None:
        _GIDX = _gloss_index(brain)
    words = brain.toks(gloss)
    # (1) STRONG signal: a parent whose NAME appears in the gloss ('a breed of DOG' -> dog)
    named = set()
    for w in words:
        named |= brain.parents_with_word(w)
    named_ranked = sorted(named, key=lambda c: -brain.depth(c))
    # (2) COVERAGE signal: parents whose own GLOSS overlaps this gloss (semantic-ish, lexical) -- raises
    #     the chance a true ancestor is surfaced even when it isn't named outright.
    cnt = Counter()
    for w in words:
        for c in _GIDX.get(w, ()):
            cnt[c] += 1
    overlap = [c for c, k in cnt.most_common(40) if k >= 2 and c not in named]
    overlap.sort(key=lambda c: (-cnt[c], -brain.depth(c)))
    return named_ranked + overlap
