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
