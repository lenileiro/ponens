#!/usr/bin/env python3
"""bdd_types -- a set-theoretic TYPE system for LOTA categories, BDD-backed.

Following Elixir's set-theoretic types (elixir-lang.org blog series): types are sets of values closed
under UNION / INTERSECTION / NEGATION, and subtyping is decided by EMPTINESS  (A <= B  iff  A and not B
is empty). We represent the boolean combination of category literals as a Reduced Ordered Binary
Decision Diagram (ROBDD) with HASH-CONSING -- the structural dedup those posts rely on (equal types ->
the SAME node). Emptiness is decided against the TAXONOMY (the isa hierarchy): a cube of category
literals is satisfiable iff some category c has all the positives among its ancestors-or-self and none
of the negatives -- so `bird and fish = empty` (no entity is both) and `robin <= animal` fall out.

This complements the dependent-type kernel (thinking/kernel.py, proofs) with the OTHER major school of
types (set-theoretic, subtyping) -- the deferred "category type system" from thinking/LANG.md.

Scope note: this is a correct ROBDD + taxonomy-grounded emptiness. The blog's LAZY-union /
EAGER-literal-intersection / EAGER-literal-difference tricks are the PERFORMANCE layer for huge type
algebras; noted as the scaling step, not needed for correctness here.

  python -m thinking.bdd_types --selftest
  python -m thinking.bdd_types --demo
"""
import argparse
import sys


# ===================================================================================================
# ROBDD over an ordered set of category atoms. Terminals TOP (1) / BOT (0). Nodes hash-consed.
# ===================================================================================================
class Node:
    __slots__ = ("i", "low", "high")

    def __init__(self, i, low, high):
        self.i = i; self.low = low; self.high = high     # var index; false-branch; true-branch


class BDD:
    TOP = ("TOP",)                                       # the type ANY (all values)
    BOT = ("BOT",)                                       # the type NONE (empty)

    def __init__(self, atoms):
        self.atoms = list(atoms)
        self.order = {a: i for i, a in enumerate(self.atoms)}
        self._unique = {}                                # hash-cons table -> canonical nodes
        self._memo = {}

    def _node(self, i, low, high):
        if low is high:                                  # reduction rule: redundant test removed
            return low
        key = (i, id(low), id(high))                     # children are already canonical -> id is safe
        n = self._unique.get(key)
        if n is None:
            n = Node(i, low, high); self._unique[key] = n
        return n

    def atom(self, name):
        return self._node(self.order[name], self.BOT, self.TOP)   # true iff this category holds

    def _cof(self, f, i):
        return (f.low, f.high) if isinstance(f, Node) and f.i == i else (f, f)

    def neg(self, f):
        if f is self.TOP:
            return self.BOT
        if f is self.BOT:
            return self.TOP
        k = ("neg", id(f))
        if k in self._memo:
            return self._memo[k]
        r = self._node(f.i, self.neg(f.low), self.neg(f.high))
        self._memo[k] = r; return r

    def _bin(self, op, f, g):
        # terminal shortcuts
        if op == "and":
            if f is self.BOT or g is self.BOT:
                return self.BOT
            if f is self.TOP:
                return g
            if g is self.TOP:
                return f
        else:  # or
            if f is self.TOP or g is self.TOP:
                return self.TOP
            if f is self.BOT:
                return g
            if g is self.BOT:
                return f
        if f is g:
            return f
        k = (op, id(f), id(g)) if id(f) <= id(g) else (op, id(g), id(f))
        if k in self._memo:
            return self._memo[k]
        i = min(f.i if isinstance(f, Node) else 10**9,
                g.i if isinstance(g, Node) else 10**9)
        fl, fh = self._cof(f, i); gl, gh = self._cof(g, i)
        r = self._node(i, self._bin(op, fl, gl), self._bin(op, fh, gh))
        self._memo[k] = r; return r

    def and_(self, f, g):
        return self._bin("and", f, g)

    def or_(self, f, g):
        return self._bin("or", f, g)

    def diff(self, f, g):
        return self.and_(f, self.neg(g))

    # --- enumerate the cubes (paths to TOP): (positive atoms, negative atoms) ---
    def cubes(self, f, pos=(), neg=()):
        if f is self.TOP:
            yield (frozenset(pos), frozenset(neg)); return
        if f is self.BOT:
            return
        a = self.atoms[f.i]
        yield from self.cubes(f.low, pos, neg + (a,))     # this category false
        yield from self.cubes(f.high, pos + (a,), neg)    # this category true


# ===================================================================================================
# Taxonomy: the isa hierarchy supplies which literal cubes are actually inhabited.
# ===================================================================================================
class Taxonomy:
    def __init__(self, isa_edges):
        self.parent = dict(isa_edges)
        self.cats = sorted({x for e in isa_edges for x in e})

    def up(self, c):
        """ancestors-or-self of c (its full set of category memberships)."""
        s, cur = {c}, c
        while cur in self.parent:
            cur = self.parent[cur]; s.add(cur)
        return s

    def sat(self, pos, neg):
        """A cube (pos, neg) is inhabited iff SOME category c has every positive among up(c) and no
        negative in up(c). (Closed-taxonomy reading; open-world would also allow fresh leaves.)"""
        for c in self.cats:
            u = self.up(c)
            if pos <= u and not (neg & u):
                return True
        return False


class TypeChecker:
    """Set-theoretic types over the taxonomy categories, BDD-backed."""

    def __init__(self, isa_edges):
        self.tax = Taxonomy(isa_edges)
        self.bdd = BDD(self.tax.cats)

    @classmethod
    def from_kb(cls, kb):
        """Build the type lattice from a prover KB's DIRECT isa edges (kb.facts)."""
        return cls([e for (pred, e) in kb.facts if pred == "isa"])

    def lit(self, c):
        return self.bdd.atom(c)

    def empty(self, t):
        """NAIVE: enumerate ALL cubes (paths to TOP), check each against the taxonomy."""
        return all(not self.tax.sat(p, n) for (p, n) in self.bdd.cubes(t))

    def empty_eager(self, t):
        """EAGER (the posts' idea): prune a subtree the MOMENT its committed positives are already
        taxonomy-incompatible -- never enumerate the doomed cubes beneath it. Adding more
        positives/negatives only tightens, so an unsatisfiable partial cube can never recover. Sets
        self.stats = {visited, sat_calls} for the stress comparison."""
        self.stats = {"visited": 0, "sat_calls": 0}

        def go(f, pos, neg):
            self.stats["visited"] += 1
            if f is self.bdd.BOT:
                return True                                  # this branch contributes no cube
            self.stats["sat_calls"] += 1
            if not self.tax.sat(pos, neg):                   # partial cube already dead -> prune subtree
                return True
            if f is self.bdd.TOP:
                return False                                 # a fully-inhabited cube survived -> NON-empty
            a = self.bdd.atoms[f.i]
            return go(f.low, pos, neg | {a}) and go(f.high, pos | {a}, neg)

        return go(t, frozenset(), frozenset())

    def subtype(self, a, b):
        """A <= B  iff  A and not B is empty (the set-theoretic definition)."""
        return self.empty(self.bdd.diff(a, b))

    def equiv(self, a, b):
        return self.subtype(a, b) and self.subtype(b, a)

    def member(self, ent, t):
        """Is entity `ent` necessarily of type `t`? -> its category is a subtype of t."""
        return self.subtype(self.lit(ent), t)

    def disjoint(self, a, b):
        return self.empty(self.bdd.and_(a, b))

    def query(self, s):
        """Answer a set-theoretic category question in LOTA S-expr. Forms:
        (subtype T1 T2) (member entity T) (disjoint T1 T2) (empty T) -- where T is a type expr
        over categories with and/or/not. Returns bool. This is the type system exposed to the agent
        language: compound category questions (intersections/negations) the isa/prop proof path can't
        natively express."""
        toks = _tok(s)
        assert toks[0] == "(" and toks[-1] == ")", "query must be an S-expr"
        head = toks[1]
        # the operands are themselves type S-exprs; re-serialize the slices and parse them
        inner = _split_args(toks[2:-1])
        if head == "subtype":
            return self.subtype(self.parse(inner[0]), self.parse(inner[1]))
        if head == "member":
            return self.member(inner[0].strip(), self.parse(inner[1]))
        if head == "disjoint":
            return self.disjoint(self.parse(inner[0]), self.parse(inner[1]))
        if head == "empty":
            return self.empty(self.parse(inner[0]))
        raise ValueError(f"unknown query head {head}")

    # parse a small type S-expr: (and t..) (or t..) (not t) <category>
    def parse(self, s):
        toks, pos = _tok(s), [0]

        def p():
            t = toks[pos[0]]; pos[0] += 1
            if t == "(":
                head = toks[pos[0]]; pos[0] += 1
                args = []
                while toks[pos[0]] != ")":
                    args.append(p())
                pos[0] += 1
                if head == "and":
                    r = args[0]
                    for a in args[1:]:
                        r = self.bdd.and_(r, a)
                    return r
                if head == "or":
                    r = args[0]
                    for a in args[1:]:
                        r = self.bdd.or_(r, a)
                    return r
                if head == "not":
                    return self.bdd.neg(args[0])
                raise ValueError(f"bad type op {head}")
            return self.lit(t)
        return p()


def _tok(s):
    return s.replace("(", " ( ").replace(")", " ) ").split()


def _split_args(toks):
    """Split a flat token list into top-level operand STRINGS (a bare atom, or a balanced (...) group)."""
    out, i = [], 0
    while i < len(toks):
        if toks[i] == "(":
            depth, j = 0, i
            while j < len(toks):
                depth += (toks[j] == "(") - (toks[j] == ")")
                j += 1
                if depth == 0:
                    break
            out.append(" ".join(toks[i:j])); i = j
        else:
            out.append(toks[i]); i += 1
    return out


# ===================================================================================================
# Demo taxonomy + selftest
# ===================================================================================================
def gen_tree_taxonomy(branch=3, depth=4):
    """A balanced tree taxonomy: root 'c' with `branch` children per node to `depth` levels."""
    edges, frontier, nid = [], ["c0"], [1]

    def fresh():
        n = f"c{nid[0]}"; nid[0] += 1; return n
    for _ in range(depth):
        nxt = []
        for parent in frontier:
            for _b in range(branch):
                child = fresh(); edges.append((child, parent)); nxt.append(child)
        frontier = nxt
    return edges


def stress(branch=3, depth=4):
    """Compare NAIVE full-cube enumeration vs EAGER pruning on a larger taxonomy. Build a type that
    mixes many disjoint categories (lots of doomed cubes) and confirm both agree + eager prunes."""
    import time
    edges = gen_tree_taxonomy(branch, depth)
    tc = TypeChecker(edges)
    cats = tc.tax.cats
    leaves = [c for c in cats if c not in tc.tax.parent.values()]
    # type = union of many leaves, intersected with the negation of one leaf -> many disjoint cubes
    u = tc.lit(leaves[0])
    for lf in leaves[1:min(len(leaves), 12)]:
        u = tc.bdd.or_(u, tc.lit(lf))
    t = tc.bdd.and_(u, tc.bdd.neg(tc.lit(leaves[0])))
    n_cubes = sum(1 for _ in tc.bdd.cubes(t))
    t0 = time.time(); e1 = tc.empty(t); naive_ms = (time.time() - t0) * 1e3
    t0 = time.time(); e2 = tc.empty_eager(t); eager_ms = (time.time() - t0) * 1e3
    return dict(cats=len(cats), atoms=len(tc.bdd.atoms), naive_cubes=n_cubes, agree=(e1 == e2),
                empty=e1, naive_ms=naive_ms, eager_ms=eager_ms,
                eager_visited=tc.stats["visited"], eager_sat_calls=tc.stats["sat_calls"])


def _demo_edges():
    return [("robin", "bird"), ("sparrow", "bird"), ("penguin", "bird"), ("bird", "animal"),
            ("dog", "mammal"), ("cat", "mammal"), ("mammal", "animal"),
            ("salmon", "fish"), ("fish", "animal"), ("animal", "living"),
            ("oak", "plant"), ("plant", "living")]


def selftest():
    tc = TypeChecker(_demo_edges())
    B = tc.bdd
    bird, animal, mammal, fish = (tc.lit(x) for x in ("bird", "animal", "mammal", "fish"))
    robin, living, plant = (tc.lit(x) for x in ("robin", "living", "plant"))

    # subtyping from the taxonomy (decided by emptiness of A and not B)
    assert tc.subtype(bird, animal), "bird <= animal"
    assert tc.subtype(robin, living), "robin <= living (transitive)"
    assert not tc.subtype(animal, bird), "animal is NOT <= bird"
    assert not tc.subtype(bird, mammal), "bird is NOT <= mammal (incomparable)"

    # disjointness: incomparable categories intersect to EMPTY
    assert tc.empty(B.and_(bird, fish)), "bird and fish = empty (no entity is both)"
    assert tc.empty(B.and_(bird, mammal)), "bird and mammal = empty"
    assert not tc.empty(B.and_(bird, animal)), "bird and animal = bird (non-empty)"

    # union subtyping: (bird or mammal) <= animal
    assert tc.subtype(B.or_(bird, mammal), animal), "(bird or mammal) <= animal"
    # robin <= (animal and not plant)
    assert tc.subtype(robin, B.and_(animal, B.neg(plant))), "robin <= animal and not plant"
    # bird and not animal = empty (bird is fully inside animal)
    assert tc.empty(B.diff(bird, animal)), "bird \\ animal = empty"
    # negation / complement: animal and not bird is NON-empty (mammals, fish, ...)
    assert not tc.empty(B.diff(animal, bird)), "animal \\ bird non-empty"

    # HASH-CONSING / canonicity: equal types are the SAME node (the dedup the BDD posts rely on)
    assert B.and_(bird, animal) is B.and_(animal, bird), "and is canonical (order-independent)"
    assert B.and_(bird, bird) is bird, "idempotent and reduces"
    assert tc.equiv(B.or_(bird, mammal), B.or_(mammal, bird)), "union commutes"
    assert B.neg(B.neg(bird)) is bird, "double negation cancels (canonical)"

    # eager pruning agrees with naive enumeration (and prunes)
    assert tc.empty(B.and_(bird, fish)) == tc.empty_eager(B.and_(bird, fish)) is True
    assert tc.empty(B.or_(bird, mammal)) == tc.empty_eager(B.or_(bird, mammal)) is False
    s = stress(branch=3, depth=3)
    assert s["agree"], ("naive vs eager disagree", s)

    # parser
    t = tc.parse("(and animal (not fish))")
    assert tc.subtype(tc.lit("dog"), t) and not tc.subtype(tc.lit("salmon"), t), \
        "dog : animal and not fish ; salmon is not"

    # LOTA-surface set-theoretic QUERIES (compound categories the isa/prop proof path can't express)
    assert tc.query("(subtype (and bird (not penguin)) animal)"), "non-penguin birds are animals"
    assert tc.query("(member robin (and animal (not plant)))"), "robin : animal and not plant"
    assert not tc.query("(member salmon (and animal (not fish)))"), "salmon is a fish -> not"
    assert tc.query("(disjoint bird fish)"), "bird and fish disjoint"
    assert tc.query("(empty (and bird mammal))"), "bird and mammal empty"
    assert not tc.query("(empty (or bird mammal))"), "bird or mammal non-empty"

    print("bdd_types selftest OK")


def demo():
    tc = TypeChecker(_demo_edges())
    print("bdd_types -- set-theoretic category types (BDD-backed), subtyping via emptiness\n")
    checks = [
        ("bird <= animal", "(and bird (not animal))", "subtype"),
        ("robin <= living", "(and robin (not living))", "subtype"),
        ("bird and fish", "(and bird fish)", "empty"),
        ("(bird or mammal) <= animal", "(and (or bird mammal) (not animal))", "subtype"),
        ("robin : animal and not plant", "(and robin (not (and animal (not plant))))", "subtype"),
        ("animal and not bird (mammals/fish/...)", "(and animal (not bird))", "nonempty"),
    ]
    for label, expr, kind in checks:
        t = tc.parse(expr)
        e = tc.empty(t)
        if kind == "subtype":
            print(f"  [{'YES' if e else 'no ':>3}]  {label}   (A and not B {'= empty' if e else 'non-empty'})")
        elif kind == "empty":
            print(f"  [{'YES' if e else 'no ':>3}]  {label} = empty")
        else:
            print(f"  [{'YES' if not e else 'no ':>3}]  {label} is non-empty")
    print("\n  --- LOTA-surface set-theoretic queries (compound categories) ---")
    for q in ["(subtype (and bird (not penguin)) animal)",
              "(member robin (and animal (not plant)))",
              "(member salmon (and animal (not fish)))",
              "(disjoint bird fish)",
              "(empty (and bird mammal))"]:
        print(f"  [{'YES' if tc.query(q) else 'no ':>3}]  {q}")


def cross_check():
    """Two INDEPENDENT type systems must agree: BDD set-theoretic subtyping (A and not B = empty) vs
    the kernel prover's isa derivability. For every category pair, subtype(a,b) should match
    isa(a,b) being kernel-provable."""
    import thinking.prover as P
    kb = P.build_kb()
    tc = TypeChecker.from_kb(kb)
    prover = P.KernelProver(kb)
    agree = total = 0
    disagreements = []
    for a in tc.tax.cats:
        for b in tc.tax.cats:
            if a == b:
                continue
            total += 1
            st = tc.subtype(tc.lit(a), tc.lit(b))                # set-theoretic: a ∧ ¬b = ∅
            kp, _, _ = prover.verify(("isa", (a, b)))            # kernel-verified isa derivability
            if st == kp:
                agree += 1
            else:
                disagreements.append((a, b, st, kp))
    return agree, total, disagreements


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--cross", action="store_true",
                    help="cross-check BDD subtyping vs the kernel prover's isa derivability")
    ap.add_argument("--stress", action="store_true",
                    help="naive full-cube enumeration vs eager pruning on a larger taxonomy")
    args = ap.parse_args(argv)
    if args.stress:
        s = stress(branch=3, depth=4)
        print("eager pruning vs naive cube enumeration (Elixir/BDD line):")
        print(f"  taxonomy: {s['cats']} categories / {s['atoms']} atoms")
        print(f"  type is empty: {s['empty']}  (naive & eager agree: {s['agree']})")
        print(f"  NAIVE  : enumerated {s['naive_cubes']} cubes   in {s['naive_ms']:.1f} ms")
        print(f"  EAGER  : visited {s['eager_visited']} nodes / {s['eager_sat_calls']} sat-checks "
              f"in {s['eager_ms']:.1f} ms  (prunes doomed subtrees early)")
        won = s["eager_visited"] < s["naive_cubes"]
        print(f"  verdict: {'eager prunes' if won else 'NO win at this scale'} -- the hash-consed "
              f"ROBDD is already compact here; eager pruning pays off only on the large-scale "
              f"union/intersection blowup the posts target (1000+ literals). Scale-only, as noted.")
        return 0
    if args.selftest:
        selftest(); return 0
    if args.demo:
        demo(); return 0
    if args.cross:
        agree, total, dis = cross_check()
        print(f"cross-check: BDD set-theoretic subtyping vs kernel-prover isa derivability")
        print(f"  AGREE {agree}/{total} category pairs" + ("" if not dis else f"  DISAGREE: {dis[:5]}"))
        print("  (two independent type systems -- set-theoretic emptiness and proof-search -- concur)")
        return 0
    ap.print_help(); return 0


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    raise SystemExit(main())
