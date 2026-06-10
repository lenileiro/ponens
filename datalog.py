#!/usr/bin/env python3
"""Minimal Datalog engine (facts + Horn rules -> least-fixpoint deductive closure, with PROVENANCE).

For our controlled relational curriculum this gives, for free and consistently-by-construction:
  - the deductive CLOSURE (all derivable facts) -> contradiction-free data + ground-truth far-hop labels
  - a PROOF TREE + hop-DEPTH for every derived fact -> proof supervision + a length axis
  - a VERIFIER / ORACLE: is a fact entailed? is a claimed derivation a valid proof?

Atoms: (pred, terms) where a term is a constant (str) or a variable (str starting with '?').
Rule:  (head_atom, [body_atoms]).  Facts: (pred, (const, ...)).

  python datalog.py   # self-check on transitive ancestor
"""


def _unify(pat, fact, sub):
    """Match pattern atom against a ground fact under substitution `sub`; return extended sub or None."""
    if pat[0] != fact[0] or len(pat[1]) != len(fact[1]):
        return None
    s = dict(sub)
    for t, a in zip(pat[1], fact[1]):
        if isinstance(t, str) and t.startswith("?"):
            if t in s and s[t] != a:
                return None
            s[t] = a
        elif t != a:
            return None
    return s


def _inst(atom, sub):
    return (atom[0], tuple(sub.get(t, t) if isinstance(t, str) and t.startswith("?") else t for t in atom[1]))


def _isvar(t):
    return isinstance(t, str) and t.startswith("?")


def _walk(t, sub):
    while _isvar(t) and t in sub:
        t = sub[t]
    return t


def _ground(atom, sub):
    return (atom[0], tuple(_walk(t, sub) for t in atom[1]))


def _unify2(a, b, sub):
    """General unification of two atoms (either may contain variables), under `sub`."""
    if a[0] != b[0] or len(a[1]) != len(b[1]):
        return None
    s = dict(sub)
    for x, y in zip(a[1], b[1]):
        x, y = _walk(x, s), _walk(y, s)
        if x == y:
            continue
        if _isvar(x):
            s[x] = y
        elif _isvar(y):
            s[y] = x
        else:
            return None
    return s


def _rename(atom, n):
    return (atom[0], tuple(f"{t}~{n}" if _isvar(t) else t for t in atom[1]))


class Datalog:
    def __init__(self, rules):
        self.rules = rules

    def _join(self, body, known, sub, used):
        """Conjunctive join of body atoms over `known` facts; yield (substitution, used-facts-tuple)."""
        if not body:
            yield sub, used
            return
        atom, rest = body[0], body[1:]
        for f in known:
            s = _unify(atom, f, sub)
            if s is not None:
                yield from self._join(rest, known, s, used + (f,))

    def closure(self, facts):
        """Least fixpoint. Returns (all_facts, prov) where prov[f] = (rule_idx|None, body_facts, depth)."""
        known = set(facts)
        prov = {f: (None, (), 0) for f in known}            # EDB facts: depth 0, no rule
        changed = True
        while changed:
            changed = False
            new = []                                         # collect this round; add after (no mutation mid-join)
            for ri, (head, body) in enumerate(self.rules):
                for sub, used in self._join(body, known, {}, ()):
                    hf = _inst(head, sub)
                    if hf not in known:
                        new.append((hf, ri, used))
            for hf, ri, used in new:
                if hf not in known:
                    known.add(hf)
                    prov[hf] = (ri, used, 1 + max((prov[u][2] for u in used), default=0))
                    changed = True
        return known, prov

    def entails(self, facts, query):
        """Verifier/oracle: is `query` (a ground fact) in the deductive closure of `facts`?"""
        return query in self.closure(facts)[0]

    def proof_tree(self, prov, fact):
        """Reconstruct the proof tree of a derived fact from provenance (for proof supervision)."""
        ri, body, _ = prov[fact]
        if ri is None:
            return {"fact": fact, "rule": None}
        return {"fact": fact, "rule": ri, "from": [self.proof_tree(prov, b) for b in body]}

    # ---- backward chaining (GOAL-DIRECTED: explores only what is relevant to the query) -----------
    def prove(self, facts, goal, depth=64):
        """SLD resolution: yield ground instances of `goal` provable from facts+rules. Unlike
        closure() (exhaustive forward fixpoint), this starts at the GOAL and recurses on subgoal
        obligations -- the thinking-flow order. Left-recursive rules are handled by a per-path
        repeated-subgoal check; `depth` bounds rule applications."""
        self._rn = 0
        seen = set()
        for s in self._prove(tuple(facts), (goal,), {}, depth, frozenset()):
            g = _ground(goal, s)
            if g not in seen:
                seen.add(g)
                yield g

    def _prove(self, facts, goals, sub, depth, path):
        if not goals:
            yield sub
            return
        g0, rest = goals[0], goals[1:]
        g = _ground(g0, sub)
        if g in path:                                    # loop check (left recursion)
            return
        for f in facts:                                  # resolve against EDB facts
            s = _unify2(g, f, sub)
            if s is not None:
                yield from self._prove(facts, rest, s, depth, path)
        if depth <= 0:
            return
        for head, body in self.rules:                    # resolve against rules (fresh variables)
            self._rn += 1
            s = _unify2(_rename(head, self._rn), g, sub)
            if s is not None:
                yield from self._prove(facts, tuple(_rename(a, self._rn) for a in body) + rest,
                                       s, depth - 1, path | {g})

    def options(self, goal, sub=None):
        """The CHOICE POINTS at `goal`: [(rule_idx, instantiated_body)] -- the subgoal obligations a
        thinking flow picks among (the model chooses, the engine validates the choice)."""
        sub = sub or {}
        out = []
        for ri, (head, body) in enumerate(self.rules):
            self._rn = getattr(self, "_rn", 0) + 1
            s = _unify2(_rename(head, self._rn), _ground(goal, sub), sub)
            if s is not None:
                out.append((ri, tuple(_ground(_rename(a, self._rn), s) for a in body)))
        return out


if __name__ == "__main__":
    # transitive ancestor: anc(X,Y):-par(X,Y).  anc(X,Z):-par(X,Y),anc(Y,Z).
    rules = [(("anc", ("?x", "?y")), [("par", ("?x", "?y"))]),
             (("anc", ("?x", "?z")), [("par", ("?x", "?y")), ("anc", ("?y", "?z"))])]
    dl = Datalog(rules)
    facts = {("par", ("a", "b")), ("par", ("b", "c")), ("par", ("c", "d")), ("par", ("d", "e"))}
    closure, prov = dl.closure(facts)
    anc = sorted([f for f in closure if f[0] == "anc"], key=lambda f: prov[f][2])
    print("derived ancestors (fact -> depth):")
    for f in anc:
        print(f"  {f[1][0]}->{f[1][1]}  depth {prov[f][2]}")
    print("entails anc(a,e)?", dl.entails(facts, ("anc", ("a", "e"))), "(should be True, depth 4)")
    print("entails anc(e,a)?", dl.entails(facts, ("anc", ("e", "a"))), "(should be False)")
    # backward chaining: goal-directed, enumerates bindings, agrees with the forward oracle
    print("prove anc(a,?x):", sorted(g[1][1] for g in dl.prove(facts, ("anc", ("a", "?x")))),
          "(should be b c d e)")
    print("prove anc(a,e):", list(dl.prove(facts, ("anc", ("a", "e")))), "(one ground proof)")
    print("prove anc(e,a):", list(dl.prove(facts, ("anc", ("e", "a")))), "(should be [])")
    print("options at anc(a,?x):", dl.options(("anc", ("a", "?x"))), "(two rule choices)")
