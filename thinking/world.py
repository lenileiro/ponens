"""Worlds: rule schema + ground-truth fact generation, with the Datalog engine as oracle.

A World produces Problem instances (EDB facts + query head + oracle answer). ChainWorld is the
current benchmark (linear r-chains, transitive `far` closure); new worlds (distractor chains,
multi-relation, WordNet KB) implement the same two methods and everything downstream — trace
rendering, checking, the flow runtime, evaluation — works unchanged, because all of it is driven
by the RULES, not by this file.
"""
import sys
import os
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datalog import Datalog

# transitive closure, LEFT-recursive: the proof tree linearizes to a forward walk from the query
# head, so every content prediction in the trace is a verbatim induction lookup.
RULES = [(("far", ("?x", "?y")), [("r", ("?x", "?y"))]),
         (("far", ("?x", "?z")), [("far", ("?x", "?y")), ("r", ("?y", "?z"))])]
ENGINE = Datalog(RULES)


@dataclass
class Problem:
    head: str                         # query subject
    edb: tuple                        # ground facts, e.g. (('r', ('a', 'b')), ...)
    goal: tuple                       # the queried fact, e.g. ('far', ('a', 'e'))
    answer: str                       # oracle answer (engine-verified at construction)
    k: int                            # derivation depth
    extra_rules: tuple = ()           # rules defined IN THE QUESTION (novel-relation problems)
    question: tuple = ()              # problem-specific question surface (overrides the bank)


N_SLOTS = 160                                              # entity slot pool (covers deep-50 trees)


def anonymize(problem, lines, rng):
    """Rename every PERSON in the example to a slot token (p0..pN, random assignment per
    example). All slot embeddings are trained; 'held-out names' stop existing as a concept --
    rung 0 showed unseen-name embeddings derail the model's structural circuits even though
    the pointer copies them fine. Numbers (years/ages) are left untouched."""
    ents = []
    seen = set()
    for _, args in problem.edb:
        for a in args:
            if not str(a).isdigit() and a not in seen:
                seen.add(a)
                ents.append(a)
    slots = [f"p{i}" for i in rng.permutation(N_SLOTS)[:len(ents)]]
    mp = dict(zip(ents, slots))
    f = lambda a: mp.get(a, a)
    ren = lambda atom: (atom[0], tuple(f(a) for a in atom[1]))
    from dataclasses import replace
    p2 = replace(
        problem,
        head=f(problem.head),
        edb=tuple(ren(x) for x in problem.edb),
        goal=ren(problem.goal),
        answer=f(problem.answer),
        extra_rules=tuple((h, [b for b in body]) for h, body in problem.extra_rules),
        question=problem.question,
    )
    l2 = [(typ, ren(head), tuple(ren(b) for b in body)) for typ, head, body in lines]
    return p2, l2


def entity_pools(n_train, n_test, seed):
    """Disjoint train/test entity pools (CVCV pseudo-words), deterministic across processes."""
    rng = np.random.default_rng(seed)
    c, v = "bdfgklmnprstv", "aeiou"
    pool = set()
    while len(pool) < n_train + n_test:
        pool.add(c[rng.integers(13)] + v[rng.integers(5)] + c[rng.integers(13)] + v[rng.integers(5)])
    ents = sorted(pool)               # sorted first: set order is not reproducible
    rng.shuffle(ents)
    return ents[:n_train], ents[n_train:]


class ChainWorld:
    """Linear r-chains over an entity pool; query = the far-endpoint of the chain head."""

    def __init__(self, entities, seed=0):
        self.entities = list(entities)
        self.rng = np.random.default_rng(seed)

    def sample(self, k, rng=None):
        rng = rng or self.rng
        e = list(rng.choice(self.entities, k + 1, replace=False))
        edb = [("r", (e[i], e[i + 1])) for i in range(k)]
        rng.shuffle(edb)
        goal = ("far", (e[0], e[k]))
        assert ENGINE.entails(set(edb), goal), "oracle disagrees with construction"
        return Problem(head=e[0], edb=tuple(edb), goal=goal, answer=e[k], k=k)

    def trace_steps(self, problem):
        """Gold derivation for supervision: [(head_atom, body_atoms)] in topological order,
        straight from the engine's closure provenance (EDB facts skipped)."""
        _, prov = ENGINE.closure(set(problem.edb))
        steps, seen = [], set()

        def rec(f):
            ri, body, _ = prov[f]
            if ri is None or f in seen:
                return
            for b in body:
                rec(b)
            seen.add(f)
            steps.append((f, body))
        rec(problem.goal)
        return steps
