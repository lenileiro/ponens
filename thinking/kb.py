#!/usr/bin/env python3
"""A runtime KNOWLEDGE BASE the agent queries to find MEANING -- no pretraining, no meaning baked into weights, and
NO hardcoded words in code. Meaning lives in an external resource (WordNet); the agent looks words up generically
on demand. This is "grounding in meaning" (allowed) rather than hardcoded word lists (forbidden) or training.

The agent uses it to bridge SURFACE gaps with MEANING -- e.g. match a query to examples/passages by synonyms and
is-a relations, not just shared characters. Capabilities:

  meaning(word)      -> definition (gloss)
  synonyms(word)     -> synonymous terms
  isa(word)          -> hypernym (is-a) chain  [types/categories, from the KB hierarchy, not a hand list]
  related(word)      -> synonyms + hypernyms + hyponyms + derivations (the word's meaning-neighborhood)
  similar(a, b)      -> do a and b share meaning? (synonym / shared neighborhood / shared is-a ancestor)
  expects_quantity(w)-> is w a measurable property? ('how long/tall/far' -> numeric answer; via attribute relation)

  python -m thinking.kb --selftest
  python -m thinking.kb refund          # inspect a word's KB meaning
"""
import argparse
import functools
import sys


def _wn():
    from nltk.corpus import wordnet as wn
    return wn


@functools.lru_cache(maxsize=50000)
def synonyms(word):
    wn = _wn()
    out = set()
    for s in wn.synsets(word):
        for l in s.lemmas():
            out.add(l.name().lower().replace("_", " "))
    out.discard(word.lower())
    return frozenset(out)


@functools.lru_cache(maxsize=50000)
def isa(word):
    """The is-a (hypernym) ancestors of the word's primary sense -- types from the KB hierarchy, not hand-listed."""
    wn = _wn()
    ss = wn.synsets(word)
    if not ss:
        return tuple()
    return tuple(h.name() for h in ss[0].closure(lambda x: x.hypernyms()))


@functools.lru_cache(maxsize=50000)
def related(word):
    """The word's meaning-neighborhood: synonyms + direct hypernyms/hyponyms + derivationally related forms."""
    wn = _wn()
    out = set(synonyms(word))
    for s in wn.synsets(word):
        for rel in (s.hypernyms() + s.hyponyms()):
            for l in rel.lemmas():
                out.add(l.name().lower().replace("_", " "))
        for l in s.lemmas():
            for d in l.derivationally_related_forms():
                out.add(d.name().lower().replace("_", " "))
    out.discard(word.lower())
    return frozenset(out)


def meaning(word):
    wn = _wn()
    ss = wn.synsets(word)
    return ss[0].definition() if ss else None


@functools.lru_cache(maxsize=50000)
def expects_quantity(word):
    """Does a question focused on `word` expect a NUMERIC answer? Read PURELY from WordNet's ATTRIBUTE relation --
    no hardcoded words, no category list. A word is a measurable DIMENSION iff it participates in the attribute
    relation (which is symmetric in WordNet): adjectives grade a dimension ('how LONG/TALL/FAR/OLD' -> the synset
    has an attribute noun), and the dimension noun points back ('what LENGTH/AGE/DISTANCE/SIZE/SPEED' -> the noun
    synset has attribute adjectives). Plain qualities (good, red) and entity nouns (city, author) participate in no
    attribute relation -> False (the answer is a name, not a number)."""
    wn = _wn()
    for s in wn.synsets(word):                                # any POS
        if s.attributes():
            return True
    return False


@functools.lru_cache(maxsize=50000)
def is_entity_type(word):
    """Is `word` a noun-type with NAMED instances, so a question on it expects a PROPER NOUN ('which CITY/COUNTRY/
    AUTHOR' -> Paris/France/Orwell)? Read PURELY from WordNet's INSTANCE relation -- no hardcoded list: True iff the
    type (or one of its kinds) has instance_hyponyms (proper-named individuals). city/author/river have them;
    year/number/length/color do not."""
    wn = _wn()
    for s in wn.synsets(word, pos=wn.NOUN)[:3]:
        if s.instance_hyponyms():
            return True
        for h in s.hyponyms():
            if h.instance_hyponyms():
                return True
    return False


def similar(a, b):
    """Do a and b share meaning, per the KB? Synonyms, or one in the other's neighborhood, or a shared is-a ancestor."""
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    if b in synonyms(a) or a in synonyms(b) or b in related(a) or a in related(b):
        return True
    ia, ib = set(isa(a)), set(isa(b))
    return bool(ia and ib and (a in {x.split(".")[0] for x in ib} or b in {x.split(".")[0] for x in ia}))


def kb_match(qword, words):
    """Does any word in `words` share meaning with qword (per the KB)? Used to match by MEANING, not just surface."""
    ql = qword.lower()
    return any(similar(ql, w.lower()) for w in words)


def selftest():
    # meaning lookup at runtime (not pretrained, not hardcoded)
    print("meaning(refund):", meaning("refund"))
    print("synonyms(buy):", sorted(synonyms("buy"))[:6])
    print("isa(day)[:3]:", isa("day")[:3])
    # bridge surface gaps via MEANING
    pairs_same = [("buy", "purchase"), ("car", "automobile"), ("doctor", "physician"), ("big", "large")]
    pairs_diff = [("buy", "rocket"), ("car", "happiness")]
    for a, b in pairs_same:
        print(f"  similar({a},{b}) = {similar(a, b)}")
        assert similar(a, b), f"KB should relate {a},{b}"
    for a, b in pairs_diff:
        assert not similar(a, b), f"KB should NOT relate {a},{b}"
    # the agent uses kb_match to find meaning beyond surface: 'physician' matches a doc mentioning 'doctor'
    assert kb_match("physician", ["the", "doctor", "examined", "her"])
    assert not kb_match("rocket", ["the", "doctor", "examined", "her"])
    # answer-typing read PURELY from WordNet relations (no hardcoded word/category lists):
    # quantity = participates in the ATTRIBUTE relation (gradable dimension); entity = has named INSTANCES.
    for w in ("long", "tall", "far", "old", "wide", "length", "age", "distance"):
        assert expects_quantity(w), f"{w} should expect a numeric answer"
    for w in ("city", "author", "country"):
        assert not expects_quantity(w), f"{w} should NOT expect a numeric answer"
    for w in ("city", "author", "country", "river"):
        assert is_entity_type(w), f"{w} should expect a named entity"
    for w in ("year", "length", "distance"):
        assert not is_entity_type(w), f"{w} should NOT expect a named entity"
    print("kb selftest OK (runtime meaning lookup + meaning-based matching; no pretraining, no hardcoded words)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("word", nargs="?")
    a = ap.parse_args(argv)
    if a.selftest or not a.word:
        return selftest()
    print(f"meaning : {meaning(a.word)}")
    print(f"synonyms: {sorted(synonyms(a.word))}")
    print(f"is-a    : {list(isa(a.word))[:6]}")
    print(f"related : {sorted(related(a.word))[:15]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
