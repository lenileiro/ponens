#!/usr/bin/env python3
"""Dev explorer for the autonomous loop: load the harness data ONCE, score several propose() variants,
print a leaderboard. The winner gets written into strategy.py (the agent keeps improvements). Not the
fixed harness -- just fast multi-experiment iteration in one process."""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import research.harness as H  # noqa: E402

STOP = set("a an the of or and to in is are was for with that this these those as by from on at it its "
           "their his her any some such other being been have has had not no which who whose various "
           "esp especially used relating".split())
PREP = {"of", "genus", "family", "order", "class", "phylum", "kingdom", "division"}

te, brain = H.load(20000, 0)
# shared gloss inverted index (candidate gloss-word -> candidate parents)
GIDX = {}
for c in brain.candidates:
    for w in brain.toks(brain.gloss_of(c)):
        GIDX.setdefault(w, set()).add(c)
GIDX = {w: s for w, s in GIDX.items() if w not in STOP and len(s) <= 300}
# prefix index for crude morphology (4-char stem -> candidate parents whose lemma starts with it)
PIDX = {}
for c in brain.candidates:
    for lem in brain.lemmas(c):
        if len(lem) >= 4:
            PIDX.setdefault(lem[:4], set()).add(c)


def named(words):
    out = set()
    for w in words:
        out |= brain.parents_with_word(w)
    return out


def overlap(words, named_set, k=40):
    cnt = Counter()
    for w in words:
        for c in GIDX.get(w, ()):
            cnt[c] += 1
    ov = [c for c, n in cnt.most_common(k) if n >= 2 and c not in named_set]
    ov.sort(key=lambda c: (-cnt[c], -brain.depth(c)))
    return ov


def head_boost(wl, named_set):
    """Candidates appearing right AFTER 'of/genus/family/...': the gloss head ('a breed of DOG')."""
    heads = []
    for i, w in enumerate(wl[:-1]):
        if w in PREP:
            heads += [c for c in brain.parents_with_word(wl[i + 1]) if c in named_set]
    return heads


def morph(words, named_set, k=20):
    out = []
    for w in words:
        if len(w) >= 4:
            out += [c for c in PIDX.get(w[:4], ()) if c not in named_set]
    return sorted(set(out), key=lambda c: -brain.depth(c))[:k]


def V_baseline(gloss):
    return sorted(named(brain.toks(gloss)), key=lambda c: -brain.depth(c))


def V_overlap(gloss):
    w = brain.toks(gloss); nm = named(w)
    return sorted(nm, key=lambda c: -brain.depth(c)) + overlap(w, nm)


def V_head(gloss):
    wl = re.findall(r"[a-z]+", gloss.lower()); w = set(wl); nm = named(w)
    hb = head_boost(wl, nm)
    rest = sorted([c for c in nm if c not in hb], key=lambda c: -brain.depth(c))
    return hb + rest + overlap(w, nm)


def V_head_morph(gloss):
    wl = re.findall(r"[a-z]+", gloss.lower()); w = set(wl); nm = named(w)
    hb = head_boost(wl, nm)
    rest = sorted([c for c in nm if c not in hb], key=lambda c: -brain.depth(c))
    return hb + rest + overlap(w, nm) + morph(w, nm)


def score(propose):
    exact = cov = false = asserted = 0
    n = len(te)
    for r in te:
        ranked = propose(r["deftext"])
        prov = [c for c in ranked if c in r["ancestors"]]
        if prov:
            pick = max(prov, key=lambda c: brain.depth(c))
            asserted += 1; exact += int(pick == r["parent"]); false += int(pick not in r["ancestors"])
        cov += int(bool(prov))
    return dict(metric=exact / n, coverage=cov / n, false=false / max(1, asserted))


for name, fn in [("baseline", V_baseline), ("overlap", V_overlap), ("head+overlap", V_head),
                 ("head+overlap+morph", V_head_morph)]:
    import time
    t = time.time(); r = score(fn)
    print(f"  {name:>22}: metric {r['metric']:.4f} | coverage {r['coverage']:.3f} | false {r['false']:.3f}"
          f" | {time.time()-t:.1f}s", flush=True)


# ---- iteration 3+: synonym expansion + Porter stemming ----
from functools import lru_cache
from nltk.corpus import wordnet as _wn
from nltk.stem import PorterStemmer
_ps = PorterStemmer()


@lru_cache(maxsize=None)
def _syns(word):
    out = set()
    for s in _wn.synsets(word):
        for l in s.lemma_names():
            out.add(l.replace("_", " ").lower())
    return out


# stem index: Porter-stem of each candidate lemma token -> candidates
_STEMIDX = {}
for c in brain.candidates:
    for lem in brain.lemmas(c):
        for tok in lem.split():
            if len(tok) >= 3:
                _STEMIDX.setdefault(_ps.stem(tok), set()).add(c)


def syn_named(words, named_set):
    out = set()
    for w in words:
        for sw in _syns(w):
            out |= brain.parents_with_word(sw)
    return sorted(out - named_set, key=lambda c: -brain.depth(c))


def stem_morph(words, named_set, k=25):
    out = set()
    for w in words:
        if len(w) >= 3:
            out |= _STEMIDX.get(_ps.stem(w), set())
    return sorted(out - named_set, key=lambda c: -brain.depth(c))[:k]


def V_syn(gloss):       # current best (overlap+prefix-morph) + synonyms
    w = brain.toks(gloss); nm = named(w)
    return V_head_morph(gloss) + syn_named(w, nm)


def V_stem(gloss):      # overlap + Porter-stem morph (replaces 4-char prefix)
    w = brain.toks(gloss); nm = named(w)
    return sorted(nm, key=lambda c: -brain.depth(c)) + overlap(w, nm) + stem_morph(w, nm)


def V_syn_stem(gloss):  # everything
    w = brain.toks(gloss); nm = named(w)
    return (sorted(nm, key=lambda c: -brain.depth(c)) + overlap(w, nm)
            + stem_morph(w, nm) + syn_named(w, nm))


for name, fn in [("cur (ov+prefix)", V_head_morph), ("+syn", V_syn), ("+porter-stem", V_stem),
                 ("+syn+stem", V_syn_stem)]:
    import time
    t = time.time(); r = score(fn)
    print(f"  {name:>18}: metric {r['metric']:.4f} | coverage {r['coverage']:.3f} | false {r['false']:.3f}"
          f" | {time.time()-t:.1f}s", flush=True)
