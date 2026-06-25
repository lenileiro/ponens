#!/usr/bin/env python3
"""Inference over proper sentences: read English statements, turn them into logical facts + rules, and answer a
question by CHAINING them -- verified by the datalog engine (the 'brain'), with the proof shown.

  "Every dog is a mammal.  Rex is a dog."   ->  Is Rex a mammal?  -> yes   (Rex is-a dog, dog -> mammal)
  "Birds can fly.  Every robin is a bird.  Tweety is a robin."  ->  Can Tweety fly?  -> yes (3-hop)

No hardcoded CONTENT matching: dog/mammal/Rex/fly are extracted as terms, never compared to any word list. Only
the closed-class LOGICAL grammar is recognized -- quantifiers (every/all/a), the copula (is/are), modal (can),
possession (has/have) -- which is what turns a sentence into logic. The answer is whatever the datalog closure
PROVES (and 'no' when it cannot), so nothing is asserted without a derivation.

  python -m thinking.infer --selftest
  python -m thinking.infer
"""
import re
import sys

from datalog import Datalog

QUANT = {"every", "all", "any", "each"}            # -> universal RULE
ART = {"a", "an", "the"}
COPULA = {"is", "are", "was", "were"}              # -> is-a relation
MODAL = {"can", "could"}                           # -> ability relation
POSS = {"has", "have", "had"}                      # -> possession relation
GRAMMAR = QUANT | ART | COPULA | MODAL | POSS | {"does", "do", "did", "no", "not"}


def _sing(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _rel(word):
    w = word.lower()
    return "isa" if w in COPULA else ("can" if w in MODAL else ("has" if w in POSS else None))


def _split(words):
    """Find the relation word; return (subject-tokens, relation, object-tokens). Pure grammar, no content list."""
    for i, w in enumerate(words):
        r = _rel(w)
        if r:
            return words[:i], r, words[i + 1:]
    return words, None, []


def _term(tokens):
    """The content term = the last non-grammar word (singularised, lower-cased)."""
    content = [t for t in tokens if t.lower() not in GRAMMAR]
    return _sing(content[-1].lower()) if content else None


def parse_statement(sent):
    """A statement -> a datalog FACT (instance) or RULE (universal). Universal if it is quantified or its
    subject is plural/generic; an instance fact if the subject is a singular proper noun."""
    words = re.findall(r"[A-Za-z]+", sent)
    subj_toks, rel, obj_toks = _split(words)
    if not rel or not subj_toks or not obj_toks:
        return None
    subj, obj = _term(subj_toks), _term(obj_toks)
    if subj is None or obj is None:
        return None
    sc = [t for t in subj_toks if t.lower() not in GRAMMAR]
    quantified = words[0].lower() in QUANT or words[0].lower() in ART
    plural = sc and sc[-1].endswith("s") and not sc[-1].endswith("ss")
    proper = sc and sc[-1][0].isupper() and not plural
    if quantified or (plural and not proper):                        # universal: subj-kind -> (rel) obj
        return ("rule", (rel, ("?x", obj)), [("isa", ("?x", subj))])
    return ("fact", (rel, (subj, obj)))                              # instance


def parse_query(q):
    """A question fronts the verb ('Is Rex a mammal?', 'Can Tweety fly?'): relation = the leading verb, then the
    subject and object are the remaining content words."""
    words = re.findall(r"[A-Za-z]+", q)
    if not words:
        return None
    rel = _rel(words[0])
    content = [t for t in words[1:] if t.lower() not in GRAMMAR]
    if rel is None or len(content) < 2:
        return None
    return (rel, (_sing(content[0].lower()), _sing(content[-1].lower())))


def read(text):
    """Parse all sentences -> (facts, rules)."""
    facts, rules = [], []
    for sent in re.split(r"[.!?]", text):
        if not sent.strip():
            continue
        p = parse_statement(sent)
        if p and p[0] == "fact":
            facts.append(p[1])
        elif p and p[0] == "rule":
            rules.append((p[1], p[2]))
    return facts, rules


def infer(text, question, verbose=False):
    """Answer the question by datalog closure over the parsed sentences. Returns 'yes'/'no' (+ proof if yes)."""
    facts, rules = read(text)
    goal = parse_query(question)
    if goal is None:
        return "?", None
    dl = Datalog(rules); closure, prov = dl.closure(facts)
    yes = goal in closure
    proof = None
    if yes and verbose:
        proof = dl.proof_tree(prov, goal)
    return ("yes" if yes else "no"), proof


DEMOS = [
    ("Every dog is a mammal. Rex is a dog.",
     [("Is Rex a mammal?", "yes"), ("Is Rex a dog?", "yes"), ("Is Rex a fish?", "no")]),
    ("Birds can fly. Every robin is a bird. Tweety is a robin.",
     [("Can Tweety fly?", "yes"), ("Is Tweety a bird?", "yes"), ("Can Rex fly?", "no")]),
    ("Every mammal is an animal. Every dog is a mammal. Spot is a dog.",
     [("Is Spot an animal?", "yes"), ("Is Spot a mammal?", "yes")]),     # 3-hop chain
]


def _run(text, qs):
    print(f"\nTEXT: {text}")
    f, r = read(text)
    print(f"  parsed {len(f)} facts, {len(r)} rules")
    ok = 0
    for q, gold in qs:
        a, _ = infer(text, q); ok += (a == gold)
        print(f"  Q: {q:22s} A: {a:3s} (expected {gold}) {'OK' if a == gold else 'X'}")
    return ok, len(qs)


def selftest():
    ok = tot = 0
    for text, qs in DEMOS:
        o, t = _run(text, qs); ok += o; tot += t
    print(f"\ninfer selftest: {ok}/{tot} correct by verified chaining")
    assert ok == tot, f"wrong: {ok}/{tot}"
    print("infer selftest OK (reads sentences -> logic -> datalog-verified multi-hop inference; no content hardcoding)")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    for text, qs in DEMOS:
        _run(text, qs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
