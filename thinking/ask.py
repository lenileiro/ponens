#!/usr/bin/env python3
"""ask -- a Q&A reasoning core over REAL meaning, answering with SOUND, LEARNED-GUIDED proofs. Ties the
whole arc to interaction: a natural question about concepts is grounded to WordNet synsets, posed as a
relational goal, and solved by the learned-guided kernel-verified proof search (thinking/azr_kmeaning).
The agent answers YES only with a kernel-checked proof term, and ABSTAINS when it cannot prove -- it never
asserts the unprovable ("model proposes, brain proves", now with learned proof search over real meaning).

    python -m thinking.ask                      # train on a neighborhood, answer a demo set
    python -m thinking.ask --ask "does a robin have a feather"
"""
import argparse
import re
import sys

import thinking.lota_kernel as LK
from thinking.azr_meaning import RULES, base_tree
from thinking.azr_kmeaning import (PREDS, PID, wordnet_kb, selfplay, solve, exhaustive_prove, tree_atoms)


def ground(word, ents, pos="n"):
    """Map a word to a WordNet synset present in the KB (most-frequent in-KB sense)."""
    from nltk.corpus import wordnet as wn
    for s in wn.synsets(word.replace(" ", "_"), pos=pos):
        if s.name() in ents:
            return s.name()
    for s in wn.synsets(word.replace(" ", "_")):
        if s.name() in ents:
            return s.name()
    return None


def parse(q):
    """Tiny pattern parser -> (relation, subject_word, object_word|None)."""
    q = q.strip().lower().rstrip("?")
    m = re.match(r"(?:is|are)\s+(?:an|a|the)?\s*(.+?)\s+(?:an|a|the)?\s*(.+)", q)
    if q.startswith("is ") or q.startswith("are "):
        if m:
            return "isa", m.group(1).strip(), m.group(2).strip()
    m = re.match(r"(?:does|do|can)\s+(?:an|a|the)?\s*(.+?)\s+have\s+(?:an|a|the)?\s*(.+)", q)
    if m:
        return "has_part", m.group(1).strip(), m.group(2).strip()
    m = re.match(r"what\s+is\s+(?:an|a|the)?\s*(.+)", q)
    if m:
        return "whatis", m.group(1).strip(), None
    return None, None, None


def proof_chain(tree):
    """Readable is-a/relation chain from a proof tree."""
    leaves = []

    def walk(t):
        if t["rule"] is None:
            leaves.append(t["fact"])
        else:
            walk(t["from"][0]); leaves.append(t["from"][1]["fact"])
    walk(tree)
    parts = [f"{a.split('.')[0]} -{p}-> {b.split('.')[0]}" for (p, (a, b)) in leaves]
    return " ; ".join(parts)


def answer(query, solve_fn, facts, ents):
    """solve_fn(goal) -> (proof_tree|None, kernel_checks). Works with either the learned-guided solver or
    the exhaustive (complete) prover -- both kernel-sound."""
    rel, x, y = parse(query)
    if rel is None:
        return f"  Q: {query}\n  (couldn't parse -- try 'is a X a Y' / 'does a X have Y' / 'what is a X')"
    sx = ground(x, ents)
    if sx is None:
        return f"  Q: {query}\n  A: I don't know the concept '{x}'."
    if rel == "whatis":
        isa = {(a, b) for (p, (a, b)) in facts if p == "isa"}
        anc = []
        seen, fr = set(), [sx]
        while fr:
            u = fr.pop()
            for (a, b) in isa:
                if a == u and b not in seen:
                    seen.add(b); anc.append(b); fr.append(b)
        proven = []
        for z in anc:
            t, _c = solve_fn(("isa", (sx, z)))
            if t is not None:
                proven.append(z.split(".")[0])
        return f"  Q: {query}\n  A: a {x} is a " + ", ".join(proven[:6]) + "  (each kernel-proven)"
    sy = ground(y, ents, pos="n")
    if sy is None:
        return f"  Q: {query}\n  A: I don't know the concept '{y}'."
    goal = (rel, (sx, sy))
    tree, ck = solve_fn(goal)
    if tree is not None:
        verb = "is a" if rel == "isa" else "has a"
        return (f"  Q: {query}\n  A: YES -- a {x} {verb} {y}.  [kernel-verified proof, {ck} checks]\n"
                f"     proof: {proof_chain(tree)}")
    return (f"  Q: {query}\n  A: I can't prove that -- abstaining (not asserting the unprovable).")


def setup(cap=90, rounds=350, d=96, device="cpu", verbose=True):
    # seed roots with the query concepts so they're guaranteed in the KB neighborhood
    facts, ents, concepts = wordnet_kb(cap=cap, roots=("robin.n.01", "dog.n.01", "bird.n.01"))
    if verbose:
        print(f"  meaning KB: {len(facts)} facts over {len(ents)} real concepts; training the proof "
              f"proposer (learned-guided, kernel-sound)...", flush=True)
    prop, eid, env, _lem = selfplay(facts, ents, concepts, d=d, rounds=rounds, beam=3, budget=6,
                                    device=device, verbose=False)
    return prop, facts, ents, eid, env


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ask", default=None)
    ap.add_argument("--cap", type=int, default=90)
    ap.add_argument("--rounds", type=int, default=350)
    ap.add_argument("--exhaustive", action="store_true",
                    help="answer with the complete (exhaustive) kernel-sound prover -- fast, no training")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    facts, ents, concepts = wordnet_kb(cap=args.cap, roots=("robin.n.01", "dog.n.01", "bird.n.01"))
    env = LK.build_env(ents, PREDS, facts, RULES)
    print(f"  meaning KB: {len(facts)} facts over {len(ents)} real WordNet concepts", flush=True)
    if args.exhaustive:
        def solve_fn(goal):
            t, _closure = exhaustive_prove(goal, facts, env)
            n = 0
            if t is not None:
                atoms = set(); tree_atoms(t, atoms); n = len(atoms)
            return t, n
        mode = "exhaustive kernel-sound prover"
    else:
        prop, facts, ents, eid, env = setup(cap=args.cap, rounds=args.rounds, device=args.device)
        def solve_fn(goal):
            t, ck, _u = solve(prop, goal, facts, eid, env, 6, 3, args.device, guided=True)
            return t, ck
        mode = "learned-guided kernel-sound prover"
    queries = [args.ask] if args.ask else [
        "is a robin a bird", "is a robin an animal", "does a robin have a feather",
        "is a robin a mammal", "what is a robin",
    ]
    print(f"\n== ASK ({mode}; answers only with a kernel-verified proof, else abstains) ==", flush=True)
    for q in queries:
        print(answer(q, solve_fn, facts, ents), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
