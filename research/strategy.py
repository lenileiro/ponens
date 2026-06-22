#!/usr/bin/env python3
"""AGENT-EDITABLE comprehension strategy (the only file an agent modifies -- like autoresearch's train.py).

Improve propose() to RAISE the harness metric (exact-parent comprehension). Rules (enforced by the
harness): you see ONLY the concept's `gloss` plus the READ-ONLY `brain`; you may NOT access the held
concept's true parent or ancestors. The brain VERIFIES your proposals (keeps the provable ancestors and
picks the most specific), so rank likely true ancestors high. Return parent synset-names, best-first.

brain API:  brain.candidates                -> all candidate parent synset-names
            brain.parents_with_word(word)    -> candidate parents whose lemma == word
            brain.lemmas(name) / brain.gloss_of(name) / brain.depth(name)
            brain.toks(text)                 -> set of lowercase word tokens
"""

NAME = "baseline-lexical"


def propose(gloss, brain):
    """BASELINE: a parent is a candidate if its NAME appears as a word in the gloss ('a breed of DOG'
    -> dog); rank most-specific first (the brain then keeps the provable ones)."""
    words = brain.toks(gloss)
    cands = set()
    for w in words:
        cands |= brain.parents_with_word(w)
    return sorted(cands, key=lambda c: -brain.depth(c))
