#!/usr/bin/env python3
"""Read a (longer) prompt, LEARN its patterns, and RESPOND IN NATURAL LANGUAGE -- no hardcoded word lists, no
predefined relations, no templates, no model, no example text.

From a longer prompt the STRUCTURAL words (connectives like 'is a') recur across many sentences while the
ENTITIES vary. We DISCOVER the connectives statistically, read each sentence as an EDGE entity->entity (keeping
the original sentence), and close the relation graph transitively (datalog). To answer a question we find the
reasoning PATH for its entity and respond in natural language by (1) restating the prompt's own sentences along
that path and (2) generating a conclusion by reusing the prompt's sentence pattern with the question's subject.

The longer / more repetitive the prompt, the more reliably patterns separate connectives from entities and the
more the response can reuse the prompt's phrasing.

  python -m thinking.infer "A robin is a bird. A bird is an animal. Rex is a dog. A dog is a mammal. A mammal is an animal." "What is Rex?"
    -> "Rex is a dog. A dog is a mammal. A mammal is an animal. So Rex is an animal."
  python -m thinking.infer --selftest
"""
import argparse
import re
import sys
from collections import Counter, deque

from datalog import Datalog


def _sents(text):
    return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]


def _toks(s):
    return re.findall(r"[A-Za-z][A-Za-z0-9]*", s.lower())


def discover(text, df_frac=0.85):
    """Discover connectives (recur in ~every sentence) vs entities (vary). Each sentence -> an edge
    (first entity -> last entity), KEEPING the original sentence text. Returns (edges, entities, connectives)."""
    sents = _sents(text); rows = [_toks(s) for s in sents]; n = len(rows); df = Counter()
    for ts in rows:
        df.update(set(ts))
    conn = {w for w, c in df.items() if c >= max(2, df_frac * n)}
    edges, ents = [], set()
    for s, ts in zip(sents, rows):
        content = [w for w in ts if w not in conn]
        if len(content) >= 2:
            a, b = content[0], content[-1]
            edges.append((a, b, s)); ents.update((a, b))
    return edges, ents, conn


def _closure(edges):
    facts = [("r", (a, b)) for a, b, _ in edges]
    rules = [(("r", ("?x", "?z")), [("r", ("?x", "?y")), ("r", ("?y", "?z"))])]
    return Datalog(rules).closure(facts)[0]


def answer_set(text, question):
    """The set of entities related to the question's known entity (categories, most-specific first)."""
    edges, ents, conn = discover(text)
    known = [w for w in _toks(question) if w in ents]
    if not known:
        return []
    e = known[0]; closure = _closure(edges)
    fwd = [f[1][1] for f in closure if f[0] == "r" and f[1][0] == e]
    bwd = [f[1][0] for f in closure if f[0] == "r" and f[1][1] == e]
    targets, forward = (fwd, True) if fwd else (bwd, False)
    adj = {}
    for a, b, _ in edges:
        adj.setdefault(a if forward else b, []).append(b if forward else a)
    dist = {e: 0}; q = deque([e])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return sorted(set(targets), key=lambda t: dist.get(t, 1e9))


def _path(e, edges):
    """Reasoning path from e to the farthest entity it reaches: list of (u, v, sentence) edges, in order."""
    adj = {}
    for a, b, s in edges:
        adj.setdefault(a, []).append((b, s))
    par = {e: None}; order = [e]; q = deque([e])
    while q:
        u = q.popleft()
        for v, s in adj.get(u, []):
            if v not in par:
                par[v] = (u, s); order.append(v); q.append(v)
    if len(order) < 2:
        return []
    far = order[-1]; path = []; cur = far
    while par[cur] is not None:
        u, s = par[cur]; path.append((u, cur, s)); cur = u
    path.reverse()
    return path


def respond(text, question):
    """Respond in NATURAL LANGUAGE: restate the prompt's sentences along the reasoning path, then a generated
    conclusion (the prompt's last sentence pattern rewritten with the question's subject)."""
    edges, ents, conn = discover(text)
    known = [w for w in _toks(question) if w in ents]
    if not known:
        return "The text doesn't mention that."
    e = known[0]; path = _path(e, edges)
    if not path:
        return f"The text doesn't say anything about {e}."
    parts = [s.strip().rstrip(".").capitalize() + "." for (_, _, s) in path]    # the prompt's own chain, in NL
    u, top, last = path[-1]                                                      # generate the conclusion sentence
    lt = re.findall(r"[A-Za-z]+", last); ci = [i for i, w in enumerate(lt) if w.lower() not in conn]
    tail = lt[ci[0] + 1:]                                                        # everything after the subject term
    concl = " ".join([e.capitalize()] + tail).strip()
    if len(path) > 1:
        parts.append(f"So {concl}.")
    return " ".join(parts)


def _gen_prompt(rng):
    """A random, LONGER taxonomy prompt + wh-question + the entity set the response must contain."""
    depth = int(rng.integers(4, 7))
    chain = [f"k{rng.integers(99999)}" for _ in range(depth)]
    stmts = [f"a {chain[i]} is a {chain[i + 1]}" for i in range(depth - 1)]
    insts = []
    for _ in range(int(rng.integers(4, 8))):
        ev = f"E{rng.integers(99999)}"; lvl = int(rng.integers(depth)); stmts.append(f"{ev} is a {chain[lvl]}"); insts.append((ev, lvl))
    for _ in range(int(rng.integers(3, 7))):
        stmts.append(f"a x{rng.integers(99999)} is a {chain[int(rng.integers(depth))]}")
    rng.shuffle(stmts)
    ev, lvl = insts[int(rng.integers(len(insts)))]
    return ". ".join(stmts) + " .", f"what is {ev} ?", set(chain[lvl:])


def selftest():
    rng = __import__("numpy").random.default_rng(0); ok = 0; n = 200
    for _ in range(n):
        text, q, expect = _gen_prompt(rng)
        words = set(_toks(respond(text, q)))
        ok += expect.issubset(words)                                            # the NL response covers all categories
    acc = ok / n
    print(f"infer selftest: {acc:.3f} on {n} random longer prompts (NL response covers the inferred categories)")
    assert acc > 0.95, f"too low: {acc}"
    print("infer selftest OK (learns the prompt's patterns, responds in natural language; no hardcoded words)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("text", nargs="?"); ap.add_argument("question", nargs="?")
    a = ap.parse_args(argv)
    if a.selftest or not (a.text and a.question):
        return selftest()
    print(respond(a.text, a.question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
