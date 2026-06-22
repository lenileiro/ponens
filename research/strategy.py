#!/usr/bin/env python3
"""AGENT-EDITABLE comprehension strategy (the only file an agent modifies -- like autoresearch's train.py).

Improve propose() to RAISE the harness metric (exact-parent over all held). Rules (enforced by the
harness): you see ONLY the concept's `gloss` plus the READ-ONLY `brain`; you may NOT access the held
concept's true parent/ancestors. The brain VERIFIES every proposal (keeps the provable ancestors, picks
the most specific), so rank likely true ancestors high AND surface more of them (coverage).

brain API:  brain.candidates / brain.parents_with_word(word) / brain.lemmas(name) /
            brain.gloss_of(name) / brain.depth(name) / brain.toks(text)
"""
from collections import Counter

NAME = "lexical + gloss-overlap + morphology"

_GIDX = None    # gloss-word -> candidate parents whose GLOSS contains it
_PIDX = None    # 4-char stem -> candidate parents whose lemma starts with it (crude morphology)
_STOP = set("a an the of or and to in is are was for with that this these those as by from on at it its "
            "their his her any some such other being been have has had not no which who whose various "
            "esp especially used relating".split())


def _build(brain):
    g, p = {}, {}
    for c in brain.candidates:
        for w in brain.toks(brain.gloss_of(c)):
            g.setdefault(w, set()).add(c)
        for lem in brain.lemmas(c):
            if len(lem) >= 4:
                p.setdefault(lem[:4], set()).add(c)
    g = {w: s for w, s in g.items() if w not in _STOP and len(s) <= 300}
    return g, p


def propose(gloss, brain):
    global _GIDX, _PIDX
    if _GIDX is None:
        _GIDX, _PIDX = _build(brain)
    words = brain.toks(gloss)
    # (1) NAME match: a parent named in the gloss ('a breed of DOG' -> dog)
    named = set()
    for w in words:
        named |= brain.parents_with_word(w)
    ranked = sorted(named, key=lambda c: -brain.depth(c))
    # (2) GLOSS-OVERLAP: parents whose own gloss overlaps this gloss (semantic-ish; +coverage)
    cnt = Counter()
    for w in words:
        for c in _GIDX.get(w, ()):
            cnt[c] += 1
    ov = sorted([c for c, k in cnt.most_common(40) if k >= 2 and c not in named],
                key=lambda c: (-cnt[c], -brain.depth(c)))
    # (3) MORPHOLOGY: parents whose lemma shares a 4-char stem with a gloss word ('canine'~'canid'; +cov)
    morph = set()
    for w in words:
        if len(w) >= 4:
            morph |= _PIDX.get(w[:4], set())
    morph = sorted(morph - named, key=lambda c: -brain.depth(c))[:20]
    return ranked + ov + morph
