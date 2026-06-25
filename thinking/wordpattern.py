#!/usr/bin/env python3
"""Reasoning that UNDERSTANDS WORDS and FINDS THE PATTERN -- on REAL words with REAL meanings (WordNet), not a
toy template set. Given labeled real words, the system reads each word's MEANING (its WordNet hypernym chain --
what it IS) and DISCOVERS the verified semantic pattern that separates the labels, then answers about NEW words
by their meaning. No memorization, no hand-coded word lists: the pattern is induced and the answer is the
meaning-grounded rule applied.

  read words -> ground to meaning (WordNet is-a chain) -> azr_win verified rule discovery -> pattern -> answer

Example: from {robin, sparrow, eagle}=yes, {salmon, shark, trout}=no it DISCOVERS the rule "is-a bird"
(not the surface words), then correctly answers penguin=yes, tuna=no -- generalizing by meaning to unseen words.

  python -m thinking.wordpattern --selftest
  python -m thinking.wordpattern            # run the demos
"""
import sys

import numpy as np
from nltk.corpus import wordnet as wn

from thinking.azr_win import _enum_pure_conjuncts, apply_rule


def _ancestors(syn):
    return {syn.name().split(".")[0]} | {h.name().split(".")[0] for h in syn.closure(lambda x: x.hypernyms())}


def _meaning(word, context=None):
    """A word's MEANING = the things it IS (is-a ancestors). With a `context` set (the pattern's semantic space),
    DISAMBIGUATE by picking the sense whose ancestors best fit the context -- so 'van' reads as the vehicle, not
    the vanguard, once we know we're talking about vehicles. Without context, use the primary (commonest) sense."""
    ss = wn.synsets(word.replace(" ", "_"))
    if not ss:
        return set()
    if context:
        return _ancestors(max(ss, key=lambda s: len(_ancestors(s) & context)))   # sense most consistent w/ context
    return _ancestors(ss[0])


def _featurize(words, vocab=None):
    """Binary meaning-feature matrix: column j = 'is-a vocab[j]'. vocab built from the words if not given; when
    given (classify time), each word's sense is disambiguated against vocab (the learned semantic space)."""
    ctx = set(vocab) if vocab is not None else None
    meanings = [_meaning(w, ctx) for w in words]
    if vocab is None:
        vocab = sorted(set().union(*meanings)) if meanings else []
    X = np.array([[1.0 if v in m else 0.0 for v in vocab] for m in meanings],
                 dtype=float).reshape(len(words), len(vocab))         # always 2-D (handles empty)
    return X, vocab


def discover(train_words, labels):
    """Find the verified MEANING pattern separating the labels: the is-a feature(s) present in EVERY positive
    and NO negative -- a pure semantic discriminator (e.g. 'bird'). If no single is-a works, fall back to an
    exhaustive pure conjunction of is-a features (azr_win). Prefer the MOST GENERAL such hypernym (simplicity)."""
    y = np.asarray(labels, int); X, vocab = _featurize(train_words)
    pos, neg = X[y == 1], X[y == 0]
    lits = [(j, "==", 1.0) for j in range(len(vocab))]
    single = [[(j, "==", 1.0)] for j in range(len(vocab)) if pos[:, j].all() and not neg[:, j].any()]
    if single:                                                        # most general = covers the most words overall
        single.sort(key=lambda c: -float(X[:, c[0][0]].sum()))
        return [single[0]], vocab
    return _enum_pure_conjuncts(X, y, lits, max_lit=3, min_precision=1.0, min_support=1), vocab


def render(rule, vocab):
    if not rule:
        return "(no pattern found)"
    return "  OR  ".join("(" + " AND ".join(f"is-a {vocab[j]}" if t == 1 else f"NOT is-a {vocab[j]}"
                                            for j, o, t in c) + ")" for c in rule)


def classify(rule, vocab, words):
    X, _ = _featurize(words, vocab)                                    # same meaning-vocab as training
    return apply_rule(rule, X).astype(int)


def solve(train_words, labels, test_words, verbose=True):
    rule, vocab = discover(train_words, labels)
    if verbose:
        print(f"  discovered pattern: {render(rule, vocab)}")
    pred = classify(rule, vocab, test_words)
    return rule, vocab, pred


DEMOS = [
    ("birds vs fish", ["robin", "sparrow", "eagle", "owl"], ["salmon", "shark", "trout", "tuna"],
     [("penguin", 1), ("crow", 1), ("herring", 0), ("cod", 0)]),
    ("mammals vs birds", ["dog", "cat", "horse", "cow"], ["robin", "eagle", "owl", "crow"],
     [("lion", 1), ("wolf", 1), ("sparrow", 0), ("hawk", 0)]),
    ("vehicles vs animals", ["car", "truck", "bus", "boat"], ["dog", "cat", "horse", "cow"],
     [("van", 1), ("ship", 1), ("lion", 0), ("eagle", 0)]),
]


def _run_demo(name, pos, neg, tests):
    words = pos + neg; labels = [1] * len(pos) + [0] * len(neg)
    rule, vocab, _ = solve(words, labels, [], verbose=False)
    print(f"\n[{name}]  from {pos}=yes / {neg}=no")
    print(f"  PATTERN: {render(rule, vocab)}")
    ok = 0
    for w, gold in tests:
        p = int(classify(rule, vocab, [w])[0]); ok += (p == gold)
        print(f"  {w:9s} -> {'yes' if p else 'no':3s}  (expected {'yes' if gold else 'no'}) {'OK' if p == gold else 'X'}")
    return ok, len(tests)


def selftest():
    ok = tot = 0
    for name, pos, neg, tests in DEMOS:
        o, t = _run_demo(name, pos, neg, tests); ok += o; tot += t
    print(f"\nwordpattern selftest: {ok}/{tot} held-out words correct by meaning-pattern")
    assert ok >= tot - 1, f"too many wrong: {ok}/{tot}"
    print("wordpattern selftest OK (understands words via WordNet meaning, finds the verified pattern, generalizes)")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    for name, pos, neg, tests in DEMOS:
        _run_demo(name, pos, neg, tests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
