#!/usr/bin/env python3
"""FIXED research harness -- adapted from karpathy/autoresearch for OUR brain/comprehension objective.

autoresearch: agent edits train.py, fixed 5-min budget, single metric val_bpb. OURS: agent edits
research/strategy.py, fixed held-out WordNet eval, single metric = EXACT-parent comprehension, with the
BRAIN guaranteeing FALSE-assertion == 0 (it verifies every proposal).

The loop: strategy.propose(gloss, brain) returns is-a parent candidates ranked best-first, using ONLY
the concept's gloss + READ-ONLY brain knowledge (candidate parents and their lemmas/glosses/depth). The
harness applies the BRAIN GATE -- keeps the provable ancestors, picks the most specific -- and scores
exact-parent on the held-out set. NO CHEATING: the strategy never sees the held concept's true parent;
the brain (not the strategy) does the verification. This file is FIXED; agents improve strategy.py.

  python -m research.harness            # score the current strategy on the fixed metric
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thinking.comprehend as C  # noqa: E402  (load_wordnet / split -- the dataset)


class Brain:
    """Read-only brain knowledge the strategy may use (no answer leakage): the candidate parent pool and,
    for each candidate, its lemma words / gloss / is-a depth, plus an inverted word->parents index."""
    def __init__(self, cands, lemmas, depth, gloss, by_word):
        self.candidates = cands
        self._lem, self._depth, self._gloss, self._byw = lemmas, depth, gloss, by_word

    def lemmas(self, name):
        return self._lem.get(name, set())

    def depth(self, name):
        return self._depth.get(name, 0)

    def gloss_of(self, name):
        return self._gloss.get(name, "")

    def parents_with_word(self, word):
        return self._byw.get(word, set())                    # candidate parents whose lemma == word

    def toks(self, text):
        return set(re.findall(r"[a-z]+", text.lower()))


_CACHE = {}


def load(max_concepts=20000, seed=0):
    key = (max_concepts, seed)
    if key in _CACHE:
        return _CACHE[key]
    from nltk.corpus import wordnet as wn
    rows, gloss = C.load_wordnet(max_concepts, seed)
    _, te = C.split(rows, 0.1, seed)
    cands = sorted({r["parent"] for r in rows})
    lemmas, depth, byw = {}, {}, {}
    for p in cands:
        depth[p] = len(wn.synset(p).hypernym_paths()[0])
        for w in wn.synset(p).lemma_names():
            w = w.replace("_", " ").lower()
            lemmas.setdefault(p, set()).add(w)
            byw.setdefault(w, set()).add(p)
    out = (te, Brain(cands, lemmas, depth, gloss, byw))
    _CACHE[key] = out
    return out


def score(propose, max_concepts=20000, seed=0, budget_s=180, verbose=True):
    """Score a strategy's propose() on the fixed metric. The brain GATES (keep provable, most specific)."""
    te, brain = load(max_concepts, seed)
    t = time.time()
    exact = cov = false = asserted = 0
    n = len(te)
    for r in te:
        ranked = propose(r["deftext"], brain)                # strategy sees ONLY the gloss + brain
        prov = [c for c in ranked if c in r["ancestors"]]    # BRAIN GATE: provable ancestors
        if prov:
            pick = max(prov, key=lambda c: brain.depth(c))   # most specific provable
            asserted += 1
            exact += int(pick == r["parent"])
            false += int(pick not in r["ancestors"])
        cov += int(bool(prov))
    dt = time.time() - t
    res = dict(metric=exact / max(1, n), coverage=cov / max(1, n),
               false=false / max(1, asserted), n=n, seconds=dt,
               valid=(dt <= budget_s and false == 0))
    if verbose:
        flag = "OK" if res["valid"] else ("OVER-BUDGET" if dt > budget_s else "FALSE-ASSERTION!")
        print(f"  METRIC (exact-parent) = {res['metric']:.4f} | coverage {res['coverage']:.3f} | "
              f"false {res['false']:.3f} | {n} held | {dt:.1f}s/{budget_s}s  [{flag}]", flush=True)
    return res


def main(argv=None):
    import argparse
    import importlib
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-concepts", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=180)
    args = ap.parse_args(argv)
    strat = importlib.import_module("research.strategy")
    print(f"strategy: {getattr(strat, 'NAME', 'strategy')}", flush=True)
    score(strat.propose, max_concepts=args.max_concepts, seed=args.seed, budget_s=args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
