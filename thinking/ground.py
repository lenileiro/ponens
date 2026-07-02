#!/usr/bin/env python3
"""ground -- MEANING-BASED grounding: map a free-text query to the WordNet concept it MEANS, over ALL
~117k concepts, TRAINING-FREE, then report KERNEL-VERIFIED facts about it.

Fixes the grounding failure seen in the trained roundtrip chat: a small char-CNN over only 2400
concepts snapped queries to SURFACE form ("a large striped wild cat" -> "strip", "happy" -> "happen").
The lesson from the research loop was that the LEXICAL signal (lemmas + gloss overlap) + the BRAIN beats
the from-scratch encoder (0.72 vs 0.33) and needs no training. So grounding here is:

  1. LEMMA match -- a query word that NAMES a concept grounds to that concept (most-frequent sense
     first). "dog" -> dog.n.01, "running" -> run.v.01.
  2. GLOSS overlap -- for a description, the concept whose gloss/lemmas share the most (IDF-weighted,
     Porter-stemmed) content words with the query. "a large striped wild cat" -> a feline, not "strip".

Then the BRAIN (datalog closure + kernel proof terms over the concept's WordNet neighborhood) reports
only KERNEL-PROVEN facts -- model proposes the grounding, brain proves the facts.

    python -m thinking.ground --selftest
    python -m thinking.ground --ask "a large striped wild cat"
    python -m thinking.ground --chat
"""
import argparse
import math
import os
import pickle
import re
import sys
from functools import lru_cache

_STOP = set("a an the of or and to in is are was for with that this these those as by from on at it its "
            "their his her any some such other being been have has had not no which who whose various "
            "esp especially used relating kind type sort form way thing something someone any".split())


def _toks(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 1]


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


def _stems(words):
    ps = _stemmer()
    return [ps.stem(w) for w in words]


# ===================================================================================================
# Inverted index over ALL of WordNet (content-stem -> concepts, + doc frequencies). Cached to disk:
# the index is a deterministic function of the WordNet DB, and building it is ~one WordNet load.
# ===================================================================================================
def build_index(cache_dir="runs/ground_cache"):
    cache = os.path.join(cache_dir, "wn_index.pkl") if cache_dir else None
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    from nltk.corpus import wordnet as wn
    inv = {}                                                 # content stem -> set(synset name)
    df = {}                                                  # content stem -> doc frequency
    ps = _stemmer()
    for s in wn.all_synsets():
        name = s.name()
        words = set(_toks(s.definition()))
        for lem in s.lemma_names():
            words |= set(_toks(lem.replace("_", " ")))
        stems = {ps.stem(w) for w in words}
        for st in stems:
            inv.setdefault(st, set()).add(name)
            df[st] = df.get(st, 0) + 1
    n = len(list(wn.all_synsets()))
    idx = {"inv": inv, "df": df, "n": n}
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(idx, f)
    return idx


def _idf(idx, stem):
    return math.log(1 + idx["n"] / (1 + idx["df"].get(stem, 0)))


# ===================================================================================================
# Ground a query to concepts: lemma-name match (the query NAMES a concept) + IDF gloss overlap.
# ===================================================================================================
LEMMA_BONUS = 14.0                                           # naming a concept is a strong grounding signal


def ground(query, idx, topk=5, lemma_lookup=True):
    qwords = _toks(query)
    qstems = set(_stems(qwords))
    score = {}
    why = {}
    if lemma_lookup:                                         # (1) LEMMA match (needs real WordNet)
        from nltk.corpus import wordnet as wn
        # a leading article ("a/an/the X") makes this a NOUN PHRASE denoting a THING -> the head is a
        # noun and the other words ('large','striped','wild') are modifiers, so weight noun lemma
        # matches above verb/adjective ones (fixes "a large striped wild cat" -> cat, not stripe.v).
        raw = query.strip().lower().split()
        is_np = bool(raw) and raw[0] in ("a", "an", "the")
        # the HEAD of the NP is the LAST content word BEFORE any relative-clause / preposition marker.
        # This skips leading adjectives (head is last: "large striped wild CAT") AND trailing modifying
        # clauses (head is before the marker: "FRUIT that monkeys eat", "TOOL used for cutting wood").
        # WordNet's synsets() is POS-ordered (nouns first), not frequency-ordered, so we cannot rely on
        # sense order to spot the head -- this positional rule is robust to that.
        MARK = {"that", "which", "who", "whom", "whose", "where", "when", "used", "for", "with",
                "of", "in", "on", "to", "from", "by", "made", "having", "containing"}
        seg = raw[1:]                                        # drop the leading article
        cut = next((i for i, w in enumerate(seg) if w in MARK), len(seg))
        head = next((w for w in reversed(seg[:cut]) if w not in _STOP and len(w) > 1), None) if is_np else None

        def word_w(w, s):
            if not is_np:
                return 1.0
            base = 1.0 if s.pos() == "n" else 0.3           # NP denotes a thing -> prefer nouns
            return base * (2.5 if (w == head and s.pos() == "n") else 1.0)   # emphasize the head noun

        # each query word's synsets (most-frequent sense first) get a rank-decayed, weighted bonus
        for w in qwords:
            for rank, s in enumerate(wn.synsets(w)):
                b = LEMMA_BONUS * word_w(w, s) / (1 + rank)
                score[s.name()] = score.get(s.name(), 0.0) + b
                why.setdefault(s.name(), set()).add(f"names '{w}'")
        # whole phrase as a lemma ("wild cat" -> wildcat) -- strongest if it exists
        phrase = "_".join(qwords)
        for rank, s in enumerate(wn.synsets(phrase)):
            w = 1.0 if not is_np else (1.0 if s.pos() == "n" else 0.3)
            score[s.name()] = score.get(s.name(), 0.0) + LEMMA_BONUS * 1.5 * w / (1 + rank)
            why.setdefault(s.name(), set()).add(f"names '{' '.join(qwords)}'")
    # (2) GLOSS overlap: concepts sharing IDF-weighted content stems with the query
    for st in qstems:
        w = _idf(idx, st)
        for name in idx["inv"].get(st, ()):
            score[name] = score.get(name, 0.0) + w
            why.setdefault(name, set()).add(f"gloss~{st}")
    ranked = sorted(score, key=lambda c: -score[c])[:topk]
    return [(c, round(score[c], 2), sorted(why[c])[:4]) for c in ranked]


# ===================================================================================================
# Brain over the grounded concept's WordNet neighborhood -> KERNEL-VERIFIED facts.
# ===================================================================================================
def neighborhood(name):
    """is-a path + typed relations for a concept (and the concepts on its path), shaped for build_brain."""
    from nltk.corpus import wordnet as wn
    s = wn.synset(name)
    RELS = {"has_part": lambda x: x.part_meronyms(), "made_of": lambda x: x.substance_meronyms(),
            "member_of": lambda x: x.member_holonyms(), "entails": lambda x: x.entailments(),
            "causes": lambda x: x.causes(), "similar": lambda x: x.similar_tos()}
    parent, relations = {}, {k: set() for k in RELS}
    relations["antonym"] = set(); relations["derived"] = set()
    path = s.hypernym_paths()[0] if s.hypernym_paths() else [s]
    for x in path + [s]:
        ns = [y.name() for y in (x.hypernym_paths()[0] if x.hypernym_paths() else [x])]
        for ch, pa in zip(ns[1:], ns):
            parent[ch] = pa
        for k, fn in RELS.items():
            for o in fn(x):
                relations[k].add((x.name(), o.name()))
        for lem in x.lemmas():
            for a in lem.antonyms():
                relations["antonym"].add((x.name(), a.synset().name()))
            for d in lem.derivationally_related_forms():
                relations["derived"].add((x.name(), d.synset().name()))

    def anc(nm):
        out, cur, seen = [], nm, set()
        while cur in parent and cur not in seen:
            seen.add(cur); cur = parent[cur]; out.append(cur)
        return out

    names = {s.name()} | set(parent) | set(parent.values())
    concepts = [dict(name=n, ancestors=anc(n)) for n in sorted(names)]
    relations = {k: v for k, v in relations.items() if v}
    return concepts, relations


def proven_facts(name, k=10):
    """Build the neighborhood brain and return KERNEL-PROVEN facts about `name` (verify each)."""
    import thinking.meaning as M
    concepts, relations = neighborhood(name)
    brain = M.build_brain(concepts, relations)
    out = []
    for (p, a) in brain.known:
        if len(a) == 2 and a[0] == name:
            ok, _ = brain.verify((p, a))                     # kernel-checked proof term
            if ok:
                out.append((p, a[1]))
    # most-specific is-a first (deepest), then other relations
    isa = sorted([o for (p, o) in out if p == "isa"], key=lambda o: -len(M.brain_ancestor_index(brain).get(o, ())))
    rest = [(p, o) for (p, o) in out if p != "isa"]
    return isa[:k], rest[:k], len([1 for _ in out])


def respond(query, idx):
    g = ground(query, idx, topk=5)
    if not g:
        return f"  '{query}': could not ground to any concept."
    top, sc, why = g[0]
    import thinking.meaning as M
    isa, rest, n = proven_facts(top)
    lines = [f"  query        : {query}",
             f"  grounded as  : {M.parent_name_text(top)}  ({top}, score {sc}, {', '.join(why)})",
             f"  is-a (proven): {' -> '.join(M.parent_name_text(x) for x in isa[:6]) or '(none)'}"]
    if rest:
        lines.append("  also proven  : "
                     + ", ".join(f"{p} {M.parent_name_text(o)}" for p, o in rest[:6]))
    alt = "; ".join(f"{M.parent_name_text(c)} ({s})" for c, s, _w in g[1:4])
    if alt:
        lines.append(f"  alternatives : {alt}")
    return "\n".join(lines)


# ===================================================================================================
def selftest():
    """CPU-only: index a tiny hand-set, ground by lemma + gloss overlap, kernel-verify facts."""
    # tiny synthetic index so selftest needs no full WordNet build
    idx = {"inv": {}, "df": {}, "n": 100}
    for name, words in [("tiger.n.02", "large feline cat stripe asia coat"),
                        ("strip.n.01", "long narrow piece land cloth")]:
        for st in set(_stems(_toks(words))):
            idx["inv"].setdefault(st, set()).add(name); idx["df"][st] = idx["df"].get(st, 0) + 1
    g = ground("a large striped cat", idx, topk=2, lemma_lookup=False)   # gloss-overlap mechanism only
    assert g and g[0][0] == "tiger.n.02", f"gloss grounding wrong: {g}"
    # lemma grounding + kernel-verified facts over real WordNet
    from nltk.corpus import wordnet as wn
    assert wn.synsets("dog"), "wordnet not available"
    isa, rest, n = proven_facts("dog.n.01", k=12)
    assert any(M_name in ("animal.n.01", "carnivore.n.01", "mammal.n.01") for M_name in isa), \
        f"dog is-a not proven: {isa}"
    print(f"  ground(cat-desc)->tiger OK | dog.n.01 proven is-a {len(isa)} facts (e.g. {isa[:3]})")
    print("ground selftest OK")
    return 0


def main(argv=None):
    import thinking.meaning  # noqa: F401  (ensure importable)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ask", default=None)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    idx = build_index()
    if args.ask:
        print(respond(args.ask, idx), flush=True)
        return 0
    if args.chat:
        print("== GROUND chat -- type a word or description; blank to quit ==", flush=True)
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if not line:
                break
            print(respond(line, idx), flush=True)
        return 0
    print("nothing to do; use --selftest / --ask / --chat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
