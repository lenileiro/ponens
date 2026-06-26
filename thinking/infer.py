#!/usr/bin/env python3
"""Inference over a (longer) shared text by DISCOVERING its patterns -- no hardcoded word lists, no predefined
relations, no synthetic templates, no training, no example text.

Idea: in a longer text the STRUCTURAL words (the connectives like 'is a') recur across many sentences, while
the ENTITIES vary. So we DISCOVER the connectives statistically (high document-frequency tokens), treat the
remaining varying tokens as entities, and read each sentence as an EDGE entity->entity. The discovered relation
graph is closed transitively by the datalog engine, and a question is answered by whether its two entities are
connected -- chaining the patterns the text itself revealed.

The longer / more repetitive the text, the more reliably the patterns separate connectives from entities.

  python -m thinking.infer "A robin is a bird. A bird is an animal. Rex is a dog. A dog is a mammal. A mammal is an animal." "Is Rex an animal?"
  python -m thinking.infer --selftest
"""
import argparse
import re
import sys
from collections import Counter

from datalog import Datalog


def _sents(text):
    return [s for s in re.split(r"[.!?]", text) if s.strip()]


def _toks(s):
    return re.findall(r"[A-Za-z][A-Za-z0-9]*", s.lower())


def discover(sents, df_frac=0.85):
    """Discover the structural CONNECTIVES (tokens recurring across ALMOST EVERY sentence -- true function words,
    not merely frequent hub entities) vs the ENTITIES (varying tokens). Each sentence -> an edge
    (first entity -> last entity). Returns (edges, entities, connectives)."""
    rows = [_toks(s) for s in sents]
    n = len(rows); df = Counter()
    for ts in rows:
        df.update(set(ts))
    conn = {w for w, c in df.items() if c >= max(2, df_frac * n)}     # recurring across the text => structural
    edges, ents = [], set()
    for ts in rows:
        content = [w for w in ts if w not in conn]                   # the varying tokens = entities
        if len(content) >= 2:
            a, b = content[0], content[-1]
            edges.append((a, b)); ents.update((a, b))
    return edges, ents, conn


def _engine(edges):
    facts = [("r", (a, b)) for a, b in edges]
    rules = [(("r", ("?x", "?z")), [("r", ("?x", "?y")), ("r", ("?y", "?z"))])]   # transitive closure
    return facts, rules


def answer(text, question, verbose=False):
    """Answer by discovering the text's relation graph and checking if the question's two entities connect."""
    edges, ents, conn = discover(_sents(text))
    qc = [w for w in _toks(question) if w not in conn]               # question entities = its non-connective tokens
    if len(qc) < 2:
        return "?", None
    a, b = qc[0], qc[-1]                                             # first->last (an unknown target -> 'no')
    facts, rules = _engine(edges)
    dl = Datalog(rules); closure, prov = dl.closure(facts)
    yes = ("r", (a, b)) in closure
    proof = dl.proof_tree(prov, ("r", (a, b))) if (yes and verbose) else None
    if verbose:
        print(f"  discovered connectives: {sorted(conn)}")
        print(f"  discovered edges: {edges}")
    return ("yes" if yes else "no"), proof


def _gen_text(rng):
    """A random, longer taxonomy text (a chain with branches) + (question, gold). Random terms -> verification
    only, not example content."""
    depth = rng.integers(3, 6)
    chain = [f"k{rng.integers(99999)}" for _ in range(depth)]
    stmts = [f"a {chain[i]} is a {chain[i + 1]}" for i in range(depth - 1)]
    insts = []
    for _ in range(rng.integers(3, 6)):                              # several instances hung off the chain -> longer
        e = f"E{rng.integers(99999)}"; lvl = rng.integers(depth)
        stmts.append(f"{e} is a {chain[lvl]}"); insts.append((e, lvl))
    for _ in range(rng.integers(2, 5)):                             # sibling branches -> longer, varied
        stmts.append(f"a x{rng.integers(99999)} is a {chain[rng.integers(depth)]}")
    rng.shuffle(stmts)
    e, lvl = insts[rng.integers(len(insts))]
    if rng.random() < 0.5:
        tgt = chain[rng.integers(lvl, depth)]; gold = "yes"        # reachable ancestor
    else:
        tgt = f"z{rng.integers(99999)}"; gold = "no"               # unrelated
    return ". ".join(stmts) + " .", f"is {e} a {tgt} ?", gold


def selftest():
    rng = __import__("numpy").random.default_rng(0); ok = 0; n = 300
    for _ in range(n):
        text, q, gold = _gen_text(rng)
        ok += (answer(text, q)[0] == gold)
    acc = ok / n
    print(f"infer selftest: {acc:.3f} on {n} random longer-text problems (patterns discovered, datalog-chained)")
    assert acc > 0.95, f"too low: {acc}"
    print("infer selftest OK (discovers connectives from the text, chains by reachability; no hardcoded words)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("text", nargs="?"); ap.add_argument("question", nargs="?")
    a = ap.parse_args(argv)
    if a.selftest or not (a.text and a.question):
        return selftest()
    ans, proof = answer(a.text, a.question, verbose=True)
    print(f"A: {ans}")
    if proof:
        print(f"proof: {proof}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
