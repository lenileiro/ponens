#!/usr/bin/env python3
"""Inference over a (longer) shared text by DISCOVERING its patterns -- no hardcoded word lists, no predefined
relations, no synthetic templates, no training, no example text.

Idea: in a longer text the STRUCTURAL words (the connectives like 'is a') recur across many sentences, while
the ENTITIES vary. So we DISCOVER the connectives statistically (high document-frequency tokens), treat the
remaining varying tokens as entities, and read each sentence as an EDGE entity->entity. The discovered relation
graph is closed transitively by the datalog engine, and a wh-question is answered by RETURNING the set of
entities connected to its known entity -- 'what is Rex?' -> dog, mammal, animal -- which scales far beyond
yes/no (open-ended retrieval, not naming pairs to check). Direction is read off the graph: things the entity
relates TO (its categories) or things that relate to it (its members).

The longer / more repetitive the text, the more reliably the patterns separate connectives from entities.

  python -m thinking.infer "A robin is a bird. A bird is an animal. Rex is a dog. A dog is a mammal. A mammal is an animal." "What is Rex?"
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


def _by_distance(e, targets, edges, forward):
    """Order an answer set by graph distance from e (most specific / closest first)."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a if forward else b, []).append(b if forward else a)
    dist = {e: 0}; q = [e]
    while q:
        u = q.pop(0)
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return sorted(targets, key=lambda t: dist.get(t, 1e9))


def answer(text, question, verbose=False):
    """Answer a wh-question with the SET of entities related to the question's known entity (scales beyond
    yes/no): forward gives 'what X is' (its categories), backward gives 'what is an X' (its members). The
    relation graph + transitive closure are discovered from the text; the answer is read off the closure."""
    edges, ents, conn = discover(_sents(text))
    known = [w for w in _toks(question) if w in ents]                # the entity the question is ABOUT
    if not known:
        return [], None
    e = known[0]
    facts, rules = _engine(edges)
    closure, _ = Datalog(rules).closure(facts)
    fwd = [f[1][1] for f in closure if f[0] == "r" and f[1][0] == e]   # things e relates TO (categories)
    bwd = [f[1][0] for f in closure if f[0] == "r" and f[1][1] == e]   # things that relate to e (members)
    res = _by_distance(e, fwd, edges, True) if fwd else _by_distance(e, bwd, edges, False)
    if verbose:
        print(f"  discovered connectives: {sorted(conn)}  | about: {e}  | direction: {'categories' if fwd else 'members'}")
    return res, None


def _gen_text(rng):
    """A random, longer taxonomy text (a chain with branches) + a wh-question + the expected ANSWER SET.
    Random terms -> verification only, not example content."""
    depth = int(rng.integers(3, 6))
    chain = [f"k{rng.integers(99999)}" for _ in range(depth)]
    stmts = [f"a {chain[i]} is a {chain[i + 1]}" for i in range(depth - 1)]
    insts = []
    for _ in range(int(rng.integers(3, 6))):                         # several instances -> longer
        e = f"E{rng.integers(99999)}"; lvl = int(rng.integers(depth))
        stmts.append(f"{e} is a {chain[lvl]}"); insts.append((e, lvl))
    for _ in range(int(rng.integers(2, 5))):                         # sibling branches -> longer, varied
        stmts.append(f"a x{rng.integers(99999)} is a {chain[int(rng.integers(depth))]}")
    rng.shuffle(stmts)
    e, lvl = insts[int(rng.integers(len(insts)))]
    return ". ".join(stmts) + " .", f"what is {e} ?", set(chain[lvl:])   # answer = e's categories up the chain


def selftest():
    rng = __import__("numpy").random.default_rng(0); ok = 0; n = 300
    for _ in range(n):
        text, q, expect = _gen_text(rng)
        ok += (set(answer(text, q)[0]) == expect)
    acc = ok / n
    print(f"infer selftest: {acc:.3f} on {n} random longer-text problems (answer SET, not yes/no)")
    assert acc > 0.95, f"too low: {acc}"
    print("infer selftest OK (discovers patterns from the text, RETURNS the related-entity set; no hardcoded words)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("text", nargs="?"); ap.add_argument("question", nargs="?")
    a = ap.parse_args(argv)
    if a.selftest or not (a.text and a.question):
        return selftest()
    ans, _ = answer(a.text, a.question, verbose=True)
    print(f"A: {', '.join(ans) if ans else '(nothing the text supports)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
