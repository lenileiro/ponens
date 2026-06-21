#!/usr/bin/env python3
"""lota_kernel -- elaborate LOTA reasoning onto the trusted kernel (thinking/kernel.py).

The unification we've been building toward, now with a REAL trust split (Lean's design):

    datalog  =  proof SEARCH   (untrusted: finds a derivation)
    kernel   =  proof CHECKING (trusted: re-verifies the derivation as a typed term)

A LOTA atom (p a1 .. ak) becomes a kernel TYPE (a Prop): `p a1 .. ak`. A datalog derivation becomes a
kernel proof TERM:
  * each base fact      -> an AXIOM constant of its atom-type,
  * each rule h :- b1.. -> a UNIVERSALLY-QUANTIFIED function axiom  (forall x.., b1 -> .. -> h),
  * a derivation        -> that rule axiom APPLIED to the entity instantiation + the sub-proofs.
The kernel then type-checks the whole term against the goal type. If it checks, the fact is PROVEN --
not by closed-world assumption, but by a verified proof. A false goal has no such term (and a bogus
term is rejected by the kernel). This retires the hollow CWA gate.

  python -m thinking.lota_kernel --selftest
  python -m thinking.lota_kernel --demo
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thinking.kernel as K  # noqa: E402
import thinking.lang as L  # noqa: E402  (the LOTA surface reader/checker/desugarer)
import thinking.bdd_types as BT  # noqa: E402  (set-theoretic types -> disjointness)
from datalog import Datalog  # noqa: E402


def _excl_type(A, B):
    """The exclusion axiom type:  forall x:Ent, isa x A -> isa x B -> False  (valid iff A,B disjoint).
    Set-theoretic types JUSTIFY this axiom (they prove A and B disjoint); the kernel then uses it to
    prove negations -- a real disproof, replacing closed-world 'not derivable'."""
    return K.n_pi("x", K.n_c("Ent"),
                  K.n_arr(K.n_app(K.n_c("isa"), K.n_v("x"), K.n_c(A)),
                          K.n_arr(K.n_app(K.n_c("isa"), K.n_v("x"), K.n_c(B)), K.n_c("False"))))

ENT = K.Const("Ent")          # the type of entities
PROP = K.Sort(0)
TYPE = K.Sort(1)


def is_var(x):
    return isinstance(x, str) and x.startswith("?")


# ---------------------------------------------------------------------------------------------------
# atom (p, (args...))  ->  kernel TYPE  `p a1 .. ak`  (a Prop). In a rule, args may be variables that
# refer to Pi-bound Ent binders -> de Bruijn index = depth - 1 - var_level.
# ---------------------------------------------------------------------------------------------------
def atom_type(atom, depth=0, var_level=None):
    p, args = atom
    t = K.Const(p)
    for a in args:
        if var_level is not None and a in var_level:
            t = K.App(t, K.Var(depth - 1 - var_level[a]))
        else:
            t = K.App(t, K.Const(a))
    return t


def rule_vars(rule):
    head, body = rule
    seen, out = set(), []
    for (_p, args) in [head] + list(body):
        for a in args:
            if is_var(a) and a not in seen:
                seen.add(a); out.append(a)
    return out


def rule_axiom_type(rule):
    """rule  h :- b1,..,bN  over vars v0..v(m-1)  ==>
       (forall v0:Ent) .. (forall v(m-1):Ent), b1 -> .. -> bN -> h          (all Props)."""
    head, body = rule
    vs = rule_vars(rule)
    m, N = len(vs), len(body)
    lvl = {v: i for i, v in enumerate(vs)}
    t = atom_type(head, m + N, lvl)                       # head sits under m Ent + N proof binders
    for k in reversed(range(N)):                          # proof binders b_k at depth m+k
        t = K.Pi(atom_type(body[k], m + k, lvl), t)
    for _ in range(m):                                    # m Ent binders (outermost)
        t = K.Pi(ENT, t, "e")
    return t


# ---------------------------------------------------------------------------------------------------
# Build the kernel environment for a KB, and convert a datalog proof tree -> kernel proof term.
# ---------------------------------------------------------------------------------------------------
def ax_name(fact):
    p, args = fact
    return "ax_" + "_".join([p] + list(args))


def build_env(entities, preds, facts, rules):
    env = {"Ent": TYPE}
    for e in entities:
        env[e] = ENT
    for p, arity in preds.items():                        # p : Ent -> .. -> Prop
        ty = PROP
        for _ in range(arity):
            ty = K.Pi(ENT, ty, "e")
        env[p] = ty
    for f in facts:                                       # base facts as axioms of their atom-type
        env[ax_name(f)] = atom_type(f)
    for i, r in enumerate(rules):                         # rules as forall-function axioms
        env[f"rule{i}"] = rule_axiom_type(r)
    return env


def _unify(pattern, ground, sub):
    (p1, a1), (p2, a2) = pattern, ground
    if p1 != p2 or len(a1) != len(a2):
        return False
    for x, c in zip(a1, a2):
        if is_var(x):
            if sub.get(x, c) != c:
                return False
            sub[x] = c
        elif x != c:
            return False
    return True


def proof_term(tree, rules):
    """datalog proof tree -> kernel proof TERM (untrusted construction; the kernel will re-check it)."""
    fact = tree["fact"]
    if tree["rule"] is None:
        return K.Const(ax_name(fact))
    ri = tree["rule"]
    head, body = rules[ri]
    sub = {}
    _unify(head, fact, sub)
    for bp, child in zip(body, tree["from"]):
        _unify(bp, child["fact"], sub)
    term = K.Const(f"rule{ri}")
    for v in rule_vars(rules[ri]):                        # apply the entity instantiation ...
        term = K.App(term, K.Const(sub[v]))
    for child in tree["from"]:                            # ... then the sub-proofs (body order)
        term = K.App(term, proof_term(child, rules))
    return term


class KB:
    def __init__(self, entities, preds, facts, rules, build_types=True, eager_closure=True):
        # build_types/eager_closure default ON (full LOTA brain: closure + set-theoretic types ->
        # disjointness -> kernel EXCLUSION axioms, so negations get real proofs). Both are OPTIMIZED to
        # scale: closure is semi-naive (datalog.py); the type lattice is built cheaply and exclusion
        # axioms are computed LAZILY on demand (no upfront O(cats^2) enumeration). Flags still allow
        # turning either off for the leanest possible KB.
        self.entities, self.preds, self.facts, self.rules = entities, preds, facts, rules
        self.env = build_env(entities, preds, facts, rules)
        self.env.update(K.logic_env())                   # And/Or/Eq/Ex/False available for proofs
        self.dl = Datalog(rules)
        self.known, self.prov = self.dl.closure(facts) if eager_closure else (set(), {})
        self.tc = self._cats = None
        self._disjoint = {}                              # (a,b) -> bool, memoized lazily
        if build_types and any(p == "isa" for (p, _e) in facts):
            self.tc = BT.TypeChecker.from_kb(self)       # build the lattice only; pairs done on demand
            self._cats = self.tc.tax.cats

    def _ensure_excl(self, a, b):
        """LAZY disjointness + exclusion axiom for the pair (a,b): compute it the first time it is
        needed (e.g. by a disproof), memoize, and compile the kernel axiom into env if disjoint. Avoids
        the upfront O(cats^2) enumeration -- only the pairs a query actually touches are ever computed."""
        if (a, b) in self._disjoint:
            return self._disjoint[(a, b)]
        # a category outside the taxonomy can't be proven disjoint from anything -> not disjoint (robust:
        # never crash on an unknown category; the brain just declines to disprove it)
        order = getattr(self.tc, "bdd", None) and self.tc.bdd.order
        if not self.tc or a not in order or b not in order:
            self._disjoint[(a, b)] = False
            return False
        # empty_eager prunes doomed cubes -> far cheaper than the naive emptiness check
        dis = self.tc.empty_eager(self.tc.bdd.and_(self.tc.lit(a), self.tc.lit(b)))
        self._disjoint[(a, b)] = dis
        if dis:
            self.env[f"excl_{a}_{b}"] = K.compile(_excl_type(a, b))
        return dis

    def _atom_term(self, goal):
        """(ok, kernel Term) for an atomic goal -- the kernel-checked datalog derivation."""
        if goal not in self.known:
            return False, None
        term = proof_term(self.dl.proof_tree(self.prov, goal), self.rules)
        return K.has_type([], term, atom_type(goal), self.env), term

    def verify(self, goal):
        """Returns (proven: bool, detail). proven=True ONLY if the KERNEL type-checks a proof term."""
        ok, term = self._atom_term(goal)
        if term is None:
            return False, "no derivation (datalog search found none)"
        return ok, K.show(term)

    # ----- one wired path: LOTA SURFACE s-expr -> core -> datalog search -> kernel-checked proof -----
    def prove_surface(self, surface):
        """Take a LOTA surface string, parse+check+desugar it (thinking/lang.py), and route the
        supported fragment (ground atoms + conjunctions) through the kernel. Returns a dict describing
        the outcome. Quantifiers/events/modality/negation are reported as deferred (need inductives /
        domain iteration -- the next kernel extensions)."""
        node = L.read(surface)
        errs = L.check(node)
        if errs:
            return {"status": "invalid", "detail": errs}
        return self._prove_core(L.desugar(node))

    def _core_to_type(self, f, varmap):
        """Core LOTA formula -> kernel TYPE (Prop), with LOTA vars mapped to de Bruijn indices.
        Mirrors atom_type / the And-composition so beta-conversion lines up at the kernel."""
        tag = f[0]
        if tag == "atom":
            t = K.Const(f[1])
            for a in f[2]:
                t = K.App(t, K.Var(varmap[a]) if a in varmap else K.Const(a))
            return t
        if tag == "and":
            parts = [self._core_to_type(c, varmap) for c in f[1:]]
            t = parts[-1]
            for x in reversed(parts[:-1]):
                t = K.App(K.App(K.Const("And"), x), t)
            return t
        raise KernelError("predicate body must be atom/and in v1")

    def _prove_core(self, f):
        """Returns a dict; when status=='proven' it includes a kernel-checked proof TERM ('kterm') of
        type 'ktype'. Conjunctions build a real And_intro term (the inductive And from logic_env);
        existentials build Ex_intro (a witness from the domain); restricted universals build an
        And-fold of implication proofs over the known domain (closed-domain reading)."""
        tag = f[0]
        if tag == "atom":
            pred, args = f[1], f[2]
            if any(isinstance(a, str) and a.startswith("?") for a in args):
                return {"status": "unsupported", "detail": "free variable (needs quantifier handling)"}
            if pred not in self.env or any(a not in self.env for a in args):
                return {"status": "unknown", "goal": (pred, args),
                        "detail": "predicate/entity not declared in this KB"}
            ok, term = self._atom_term((pred, args))
            if ok:
                return {"status": "proven", "goal": (pred, args), "kterm": term,
                        "ktype": atom_type((pred, args)), "term": K.show(term)}
            return {"status": "unprovable", "goal": (pred, args)}
        if tag == "and":
            parts = [self._prove_core(c) for c in f[1:]]
            if not all(p["status"] == "proven" for p in parts):
                return {"status": "unprovable", "conjuncts": parts}
            kterm, ktype = parts[-1]["kterm"], parts[-1]["ktype"]          # right-fold And_intro
            for p in reversed(parts[:-1]):
                A, B = p["ktype"], ktype
                kterm = K.App(K.App(K.App(K.App(K.Const("And_intro"), A), B), p["kterm"]), kterm)
                ktype = K.App(K.App(K.Const("And"), A), B)
            ok = K.has_type([], kterm, ktype, self.env)                    # kernel re-checks the whole
            return {"status": "proven" if ok else "unprovable", "kterm": kterm, "ktype": ktype,
                    "term": K.show(kterm), "conjuncts": parts}
        if tag == "not":
            inner = f[1]
            # NEGATION of an isa atom via an exclusion axiom: prove  isa(r,B) -> False  by finding a
            # category A with r:A and A disjoint-from B, then  fun h:isa(r,B) => excl_A_B r <r:A> h.
            if (inner[0] == "atom" and inner[1] == "isa" and len(inner[2]) == 2
                    and not any(isinstance(a, str) and a.startswith("?") for a in inner[2])):
                r, Bcat = inner[2]
                # only the categories r ACTUALLY has can disprove isa(r,B) -> check those, lazily, not
                # all cats (so disproof is O(|r's categories|) with on-demand disjointness)
                cats_of_r = [A for A in (self._cats or []) if ("isa", (r, A)) in self.known]
                for A in cats_of_r:
                    if self._ensure_excl(A, Bcat):
                        ok, pr = self._atom_term(("isa", (r, A)))
                        if not ok:
                            continue
                        isaRB = K.App(K.App(K.Const("isa"), K.Const(r)), K.Const(Bcat))
                        body = K.App(K.App(K.App(K.Const(f"excl_{A}_{Bcat}"), K.Const(r)), pr),
                                     K.Var(0))                       # h : isa(r,B)
                        kterm = K.Lam(isaRB, body, "h")
                        ktype = K.Pi(isaRB, K.Const("False"), "h")   # isa(r,B) -> False  ==  not isa(r,B)
                        if K.has_type([], kterm, ktype, self.env):
                            return {"status": "proven", "kterm": kterm, "ktype": ktype,
                                    "term": K.show(kterm), "note": f"negation via excl_{A}_{Bcat}"}
                return {"status": "unprovable", "detail": "no disjoint supertype to disprove isa"}
            return {"status": "unsupported", "detail": "v1 negation: only isa atoms (exclusion axioms)"}
        if tag == "must":
            # necessity (closed setting) = phi is kernel-provable; the proof of phi witnesses it.
            r = self._prove_core(f[1])
            if r["status"] == "proven":
                return {**r, "note": "necessity = kernel-provable"}
            return {"status": "unprovable", "detail": "not necessary (phi not provable)"}
        if tag in ("can", "may"):
            # possibility = NOT refutable: we cannot kernel-prove not(phi). This is a CONSISTENCY
            # DECISION, not a constructed proof term -- a true modal proof needs possible-worlds
            # semantics (deferred). Refutation (impossible) DOES carry a kernel disproof.
            neg = self._prove_core(("not", f[1]))
            if neg["status"] == "proven":
                return {"status": "impossible", "detail": "refuted: not(phi) is kernel-proven",
                        "disproof": neg.get("term")}
            return {"status": "possible",
                    "detail": "consistent: not(phi) not provable (modal decision, not a proof term)"}
        if tag == "if":
            ra = self._prove_core(f[1])
            if ra["status"] != "proven":
                return {"status": "vacuous"}                                # antecedent unprovable -> vacuous
            rb = self._prove_core(f[2])
            if rb["status"] != "proven":
                return {"status": "unprovable", "detail": "consequent fails"}
            A, B = ra["ktype"], rb["ktype"]                                 # fun _:A => <proof of B> : A->B
            kterm = K.Lam(A, K.shift(rb["kterm"], 1), "h")
            ktype = K.Pi(A, K.shift(B, 1), "_")
            ok = K.has_type([], kterm, ktype, self.env)
            return {"status": "proven" if ok else "unprovable", "kterm": kterm, "ktype": ktype,
                    "term": K.show(kterm)}
        if tag == "exists":
            var, body = f[1], f[2]
            P = K.Lam(ENT, self._core_to_type(body, {var: 0}), "x")        # the predicate
            for c in self.entities:                                         # search the domain for a witness
                r = self._prove_core(L._subst(body, var, c))
                if r["status"] == "proven":
                    kterm = K.App(K.App(K.App(K.App(K.Const("Ex_intro"), ENT), P),
                                        K.Const(c)), r["kterm"])
                    ktype = K.App(K.App(K.Const("Ex"), ENT), P)
                    if K.has_type([], kterm, ktype, self.env):
                        return {"status": "proven", "kterm": kterm, "ktype": ktype,
                                "term": K.show(kterm), "witness": c}
            return {"status": "unprovable", "detail": "no witness in the known domain"}
        if tag == "forall":
            var, body = f[1], f[2]
            proofs = []
            for c in self.entities:                                         # closed-domain: over known entities
                r = self._prove_core(L._subst(body, var, c))
                if r["status"] == "vacuous":
                    continue
                if r["status"] != "proven":
                    return {"status": "unprovable", "detail": f"fails for {c}"}
                proofs.append(r)
            if not proofs:
                return {"status": "proven (closed-domain)", "kterm": K.Const("trivial"),
                        "ktype": K.Const("True"), "term": "trivial", "note": "vacuously (no instances)"}
            kterm, ktype = proofs[-1]["kterm"], proofs[-1]["ktype"]
            for p in reversed(proofs[:-1]):
                A, B = p["ktype"], ktype
                kterm = K.App(K.App(K.App(K.App(K.Const("And_intro"), A), B), p["kterm"]), kterm)
                ktype = K.App(K.App(K.Const("And"), A), B)
            ok = K.has_type([], kterm, ktype, self.env)
            return {"status": "proven (closed-domain)" if ok else "unprovable", "kterm": kterm,
                    "ktype": ktype, "term": K.show(kterm),
                    "note": "holds for all KNOWN entities (closed-domain), not a true forall"}
        return {"status": "unsupported",
                "detail": f"kernel path supports atoms/and/if/exists/forall; got '{tag}' "
                          f"(modality/negation-from-CWA: next)"}


# ---------------------------------------------------------------------------------------------------
# Demo KB: is-a chain + transitivity + property inheritance (real multi-hop reasoning).
# ---------------------------------------------------------------------------------------------------
def _demo_kb():
    # a taxonomy WITH sibling branches (bird / fish / mammal under animal) so disjointness -- and thus
    # negation via exclusion axioms -- is meaningful.
    entities = ["robin", "sparrow", "bird", "salmon", "fish", "cat", "mammal",
                "animal", "living_thing", "fly", "move"]
    preds = {"isa": 2, "prop": 2}
    facts = [("isa", ("robin", "bird")), ("isa", ("sparrow", "bird")), ("isa", ("bird", "animal")),
             ("isa", ("salmon", "fish")), ("isa", ("fish", "animal")),
             ("isa", ("cat", "mammal")), ("isa", ("mammal", "animal")),
             ("isa", ("animal", "living_thing")),
             ("prop", ("animal", "move"))]                # animals move (inherited)
    rules = [
        (("isa", ("?x", "?z")), [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))]),    # transitivity
        (("prop", ("?x", "?p")), [("isa", ("?x", "?y")), ("prop", ("?y", "?p"))]),  # inheritance
    ]
    return KB(entities, preds, facts, rules)


def selftest():
    kb = _demo_kb()
    # multi-hop is-a: robin -> bird -> animal -> living_thing (kernel-checked proof term)
    ok, term = kb.verify(("isa", ("robin", "living_thing")))
    assert ok, ("3-hop isa should be kernel-proven", term)
    # inherited property: robin moves (robin isa ... animal, animal has prop move)
    ok2, _ = kb.verify(("prop", ("robin", "move")))
    assert ok2, "inherited property should be kernel-proven"
    # a FALSE goal: no derivation, hence no proof
    ok3, why = kb.verify(("isa", ("robin", "fly")))
    assert not ok3, "false goal must not be provable"
    # SOUNDNESS: a real proof term must NOT type-check against a DIFFERENT goal type
    tree = kb.dl.proof_tree(kb.prov, ("isa", ("robin", "animal")))
    good_term = proof_term(tree, kb.rules)
    assert K.has_type([], good_term, atom_type(("isa", ("robin", "animal"))), kb.env)
    assert not K.has_type([], good_term, atom_type(("isa", ("robin", "fly"))), kb.env), \
        "kernel must reject a proof of A used as a proof of B"
    # the base facts and rule axioms are themselves well-typed declarations
    for name, ty in kb.env.items():
        K.infer([], ty, kb.env)                          # every declared type is itself type-correct

    # ----- the WIRED PATH: LOTA surface string -> kernel-checked proof -----
    r = kb.prove_surface("(isa robin living_thing)")
    assert r["status"] == "proven", ("surface isa should prove", r)
    r = kb.prove_surface("(prop robin move)")
    assert r["status"] == "proven", ("surface inherited prop should prove", r)
    r = kb.prove_surface("(and (isa robin animal) (prop robin move))")
    assert r["status"] == "proven", ("surface conjunction should prove", r)
    # the conjunction proof is a REAL And_intro term, re-checked by the kernel against And(...)
    assert K.has_type([], r["kterm"], r["ktype"], kb.env), "composite And proof must kernel-check"
    assert isinstance(r["ktype"], K.App), "conjunction goal type should be an And application"
    r = kb.prove_surface("(isa robin fly)")
    assert r["status"] == "unprovable", ("false surface goal", r)
    r = kb.prove_surface("(and (isa robin animal) (isa robin fly))")
    assert r["status"] == "unprovable", "conjunction with a false conjunct must fail"

    # EXISTS: a real witnessed existential proof (Ex_intro), kernel-checked
    r = kb.prove_surface("(exists ?x (prop ?x move))")
    assert r["status"] == "proven" and "witness" in r, ("exists should prove with a witness", r)
    assert K.has_type([], r["kterm"], r["ktype"], kb.env), "Ex_intro proof must kernel-check"
    r = kb.prove_surface("(exists ?x (isa ?x fly))")
    assert r["status"] == "unprovable", "existential with no witness must fail"

    # restricted FORALL (closed-domain): every known thing that is-an-animal moves -> And-fold of impls
    r = kb.prove_surface("(forall ?x (if (isa ?x animal) (prop ?x move)))")
    assert r["status"] == "proven (closed-domain)", ("closed-domain forall should prove", r)
    assert K.has_type([], r["kterm"], r["ktype"], kb.env), "forall And-fold proof must kernel-check"
    # a closed-domain forall that does NOT hold for some known entity
    r = kb.prove_surface("(forall ?x (if (isa ?x animal) (isa ?x fly)))")
    assert r["status"] == "unprovable", "forall must fail when an instance fails"

    # NEGATION via exclusion axioms (set-theoretic disjointness -> kernel disproof). robin is a bird;
    # bird and fish are disjoint -> isa(robin,fish) is FALSE, with a real kernel proof.
    r = kb.prove_surface("(not (isa robin fish))")
    assert r["status"] == "proven", ("disjoint-based negation should prove", r)
    assert K.has_type([], r["kterm"], r["ktype"], kb.env), "negation proof must kernel-check"
    # but a negation that ISN'T forced by disjointness is not provable this way (sound: robins DO fly)
    r = kb.prove_surface("(not (isa robin bird))")
    assert r["status"] == "unprovable", "must not disprove a TRUE isa"

    # MODALITY: must = necessity (provable) ; can/may = possibility (not refutable)
    assert kb.prove_surface("(must (isa robin animal))")["status"] == "proven", "robin must be an animal"
    assert kb.prove_surface("(must (prop robin fly))")["status"] == "unprovable", "robin need not fly"
    assert kb.prove_surface("(can (isa robin fish))")["status"] == "impossible", \
        "robin can't be a fish (refuted via exclusion)"
    assert kb.prove_surface("(can (prop robin fly))")["status"] == "possible", \
        "robin flying is consistent (not refutable)"
    print("lota_kernel selftest OK")


def demo():
    kb = _demo_kb()
    print("lota_kernel -- datalog SEARCHES, the kernel CHECKS the proof term\n")
    goals = [("isa", ("robin", "animal")), ("isa", ("robin", "living_thing")),
             ("prop", ("robin", "move")), ("isa", ("robin", "fly"))]
    for g in goals:
        ok, detail = kb.verify(g)
        gt = K.show(atom_type(g))
        if ok:
            print(f"  [KERNEL-PROVEN ] {gt}")
            print(f"        proof term : {detail}")
        else:
            print(f"  [unprovable    ] {gt}   ({detail})")
    print("\n  (each PROVEN line is a typed term re-checked by the trusted kernel -- not CWA.)")
    print("\n  --- wired path: LOTA SURFACE string -> kernel-checked proof ---")
    for s in ["(isa robin living_thing)", "(and (isa robin animal) (prop robin move))",
              "(exists ?x (prop ?x move))", "(forall ?x (if (isa ?x animal) (prop ?x move)))",
              "(isa robin fly)", "(forall ?x (if (isa ?x animal) (isa ?x fly)))"]:
        r = kb.prove_surface(s)
        st = r["status"]
        tag = ("KERNEL-PROVEN" if st.startswith("proven") else
               {"unprovable": "unprovable", "unsupported": "deferred",
                "unknown": "unknown", "invalid": "invalid"}.get(st, st))
        extra = f"  [witness {r['witness']}]" if r.get("witness") else (
                "  [closed-domain]" if "closed-domain" in st else "")
        print(f"  [{tag:>13}] {s}{extra}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    if args.demo:
        demo(); return 0
    ap.print_help(); return 0


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    raise SystemExit(main())
