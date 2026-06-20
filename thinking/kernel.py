#!/usr/bin/env python3
"""kernel -- a MINIMAL, SOUND dependent type-theory kernel for the agent's language.

Design borrowed from Lean (lean-lang.org), kept tiny and in-house (no Lean dependency):

  * TRUST ARCHITECTURE. The kernel here is the ONLY trusted code. Anything that PRODUCES terms -- an
    elaborator, a tactic, or a (neural) model -- is UNTRUSTED; whatever it emits is re-checked by this
    kernel. This is exactly our project thesis ("model proposes, engine verifies"); Lean formalizes it
    as kernel + untrusted tactics. It also fixes the hollow-verifier problem we hit with the datalog
    closed-world gate: a claim counts as true ONLY if accompanied by a proof TERM the kernel accepts.

  * CURRY-HOWARD. A proposition is a TYPE; a proof is a TERM of that type; checking a proof IS
    type-checking. Implication A -> B and universal quantification (forall x:A, B x) are both the
    dependent function type Pi. No separate "logic" layer is trusted -- just the type-checker.

  * DEPENDENT TYPES from a SMALL CORE (Lean's "small + expressive"). The whole trusted core is:
    Sort (universes), Var (de Bruijn), Pi (dependent function type), Lam, App. A predicative universe
    hierarchy Sort 0 : Sort 1 : Sort 2 : ... keeps it SOUND (no Type:Type paradox).

This module is the trusted core onto which the friendly LOTA surface (thinking/lang.py) will elaborate
in v2 -- replacing the datalog CWA gate with real, kernel-checked proof terms. See thinking/LANG.md.

  python -m thinking.kernel --selftest
  python -m thinking.kernel --demo
"""
import argparse
import sys


# ===================================================================================================
# Terms (de Bruijn indices: Var(0) is the innermost binder). Plain immutable-ish nodes.
# ===================================================================================================
class Term:
    pass


class Sort(Term):
    __slots__ = ("n",)

    def __init__(self, n):
        self.n = n                       # universe level (0 = Prop-like, 1, 2, ...)


class Var(Term):
    __slots__ = ("i",)

    def __init__(self, i):
        self.i = i                       # de Bruijn index


class Pi(Term):
    __slots__ = ("dom", "cod", "name")

    def __init__(self, dom, cod, name="_"):
        self.dom = dom                   # type of the argument
        self.cod = cod                   # result type, in context extended by `dom`
        self.name = name


class Lam(Term):
    __slots__ = ("dom", "body", "name")

    def __init__(self, dom, body, name="x"):
        self.dom = dom
        self.body = body                 # in context extended by `dom`
        self.name = name


class App(Term):
    __slots__ = ("fn", "arg")

    def __init__(self, fn, arg):
        self.fn = fn
        self.arg = arg


class Const(Term):
    """A declared name (entity / predicate / AXIOM) resolved against an Env. Axioms are OPAQUE (no
    value), so the kernel treats them as trusted givens -- this is how a datalog derivation's base
    facts and rules enter as proof primitives that the kernel then composes and re-checks."""
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


class KernelError(Exception):
    pass


# ===================================================================================================
# de Bruijn machinery: shift (re-index free vars) and subst (capture-avoiding substitution)
# ===================================================================================================
def shift(t, d, c=0):
    if isinstance(t, (Sort, Const)):
        return t
    if isinstance(t, Var):
        return Var(t.i + d) if t.i >= c else t
    if isinstance(t, Pi):
        return Pi(shift(t.dom, d, c), shift(t.cod, d, c + 1), t.name)
    if isinstance(t, Lam):
        return Lam(shift(t.dom, d, c), shift(t.body, d, c + 1), t.name)
    if isinstance(t, App):
        return App(shift(t.fn, d, c), shift(t.arg, d, c))
    raise KernelError(f"shift: unknown term {t}")


def subst(t, j, s):
    """Replace Var(j) with s (s lives in the OUTER context; caller/recursion shifts as binders pass)."""
    if isinstance(t, (Sort, Const)):
        return t
    if isinstance(t, Var):
        return s if t.i == j else t
    if isinstance(t, Pi):
        return Pi(subst(t.dom, j, s), subst(t.cod, j + 1, shift(s, 1)), t.name)
    if isinstance(t, Lam):
        return Lam(subst(t.dom, j, s), subst(t.body, j + 1, shift(s, 1)), t.name)
    if isinstance(t, App):
        return App(subst(t.fn, j, s), subst(t.arg, j, s))
    raise KernelError(f"subst: unknown term {t}")


def subst_top(s, t):
    """Beta substitution: replace Var(0) in t by s, then drop the binder level."""
    return shift(subst(t, 0, shift(s, 1)), -1)


# ===================================================================================================
# Reduction + definitional equality (conversion up to beta)
# ===================================================================================================
def whnf(t):
    """Weak head normal form (beta)."""
    while isinstance(t, App):
        f = whnf(t.fn)
        if isinstance(f, Lam):
            t = subst_top(t.arg, f.body)
        else:
            return App(f, t.arg)
    return t


def equal(a, b):
    """Definitional equality: convertible up to beta (structural after whnf)."""
    a, b = whnf(a), whnf(b)
    if isinstance(a, Sort) and isinstance(b, Sort):
        return a.n == b.n
    if isinstance(a, Var) and isinstance(b, Var):
        return a.i == b.i
    if isinstance(a, Const) and isinstance(b, Const):
        return a.name == b.name
    if isinstance(a, App) and isinstance(b, App):
        return equal(a.fn, b.fn) and equal(a.arg, b.arg)
    if isinstance(a, Pi) and isinstance(b, Pi):
        return equal(a.dom, b.dom) and equal(a.cod, b.cod)
    if isinstance(a, Lam) and isinstance(b, Lam):
        return equal(a.dom, b.dom) and equal(a.body, b.body)
    return False


# ===================================================================================================
# THE TYPE-CHECKER (the trusted kernel). ctx: list of types, ctx[0] = innermost binder.
# ===================================================================================================
def _ensure_sort(ctx, t, env):
    w = whnf(infer(ctx, t, env))
    if not isinstance(w, Sort):
        raise KernelError("expected a type (Sort), got a non-sort")
    return w.n


def infer(ctx, t, env=None):
    """Infer the type of t in ctx (env = declared constants). Soundness lives here and nowhere else."""
    env = env or {}
    if isinstance(t, Sort):
        return Sort(t.n + 1)                                  # Sort n : Sort (n+1)  (predicative)
    if isinstance(t, Const):
        if t.name not in env:
            raise KernelError(f"unknown constant '{t.name}'")
        return env[t.name]
    if isinstance(t, Var):
        if t.i < 0 or t.i >= len(ctx):
            raise KernelError(f"variable index {t.i} out of range (ctx depth {len(ctx)})")
        return shift(ctx[t.i], t.i + 1)                       # lift the recorded type over the binders
    if isinstance(t, Pi):
        i = _ensure_sort(ctx, t.dom, env)
        j = _ensure_sort([t.dom] + ctx, t.cod, env)
        return Sort(max(i, j))                                # predicative product rule
    if isinstance(t, Lam):
        _ensure_sort(ctx, t.dom, env)
        body_ty = infer([t.dom] + ctx, t.body, env)
        return Pi(t.dom, body_ty, t.name)
    if isinstance(t, App):
        ft = whnf(infer(ctx, t.fn, env))
        if not isinstance(ft, Pi):
            raise KernelError("application of a non-function")
        at = infer(ctx, t.arg, env)
        if not equal(at, ft.dom):
            raise KernelError("argument type mismatch")
        return subst_top(t.arg, ft.cod)
    raise KernelError(f"infer: unknown term {t}")


def check(ctx, term, ty, env=None):
    """Check that `term` has type `ty`. Returns True or raises KernelError. THIS is 'verify a proof'."""
    got = infer(ctx, term, env)
    if not equal(got, ty):
        raise KernelError("type mismatch: term does not have the claimed type")
    return True


def has_type(ctx, term, ty, env=None):
    try:
        return check(ctx, term, ty, env)
    except KernelError:
        return False


# ===================================================================================================
# Pretty-printer (names reconstructed for readability; not trusted)
# ===================================================================================================
def show(t, names=None):
    names = names or []

    def fresh(base):
        nm, k = base, 0
        used = set(names)
        while nm in used:
            k += 1; nm = f"{base}{k}"
        return nm

    if isinstance(t, Sort):
        return "Prop" if t.n == 0 else (f"Type{t.n - 1}" if t.n > 1 else "Type")
    if isinstance(t, Const):
        return t.name
    if isinstance(t, Var):
        return names[t.i] if t.i < len(names) else f"#{t.i}"
    if isinstance(t, Pi):
        nm = fresh(t.name if t.name != "_" else "a")
        d = show(t.dom, names)
        c = show(t.cod, [nm] + names)
        return f"(forall {nm}:{d}, {c})" if _occurs(t.cod) else f"({d} -> {c})"
    if isinstance(t, Lam):
        nm = fresh(t.name)
        return f"(fun {nm}:{show(t.dom, names)} => {show(t.body, [nm] + names)})"
    if isinstance(t, App):
        return f"({show(t.fn, names)} {show(t.arg, names)})"
    return "?"


def _occurs(cod, depth=0):
    if isinstance(cod, Var):
        return cod.i == depth
    if isinstance(cod, Pi):
        return _occurs(cod.dom, depth) or _occurs(cod.cod, depth + 1)
    if isinstance(cod, Lam):
        return _occurs(cod.dom, depth) or _occurs(cod.body, depth + 1)
    if isinstance(cod, App):
        return _occurs(cod.fn, depth) or _occurs(cod.arg, depth)
    return False


# ===================================================================================================
# Named-binder builder -> de Bruijn (so we author types/terms with NAMES; no manual index arithmetic).
# Named term forms: ('sort',n) ('var',name) ('const',name) ('pi',name,dom,cod) ('lam',name,dom,body)
# ('app',f,a). Use the helpers below.
# ===================================================================================================
def n_sort(n): return ("sort", n)
def n_v(name): return ("var", name)
def n_c(name): return ("const", name)
def n_pi(name, dom, cod): return ("pi", name, dom, cod)
def n_arr(dom, cod): return ("pi", "_", dom, cod)          # non-dependent function
def n_lam(name, dom, body): return ("lam", name, dom, body)


def n_app(f, *args):
    t = f
    for a in args:
        t = ("app", t, a)
    return t


def compile(t, scope=None):
    """Compile a named term to a de Bruijn kernel Term. scope = bound names, innermost first."""
    scope = scope or []
    tag = t[0]
    if tag == "sort":
        return Sort(t[1])
    if tag == "const":
        return Const(t[1])
    if tag == "var":
        if t[1] not in scope:
            raise KernelError(f"unbound variable '{t[1]}'")
        return Var(scope.index(t[1]))                     # first match = innermost binder
    if tag == "pi":
        _, name, dom, cod = t
        return Pi(compile(dom, scope), compile(cod, [name] + scope), name)
    if tag == "lam":
        _, name, dom, body = t
        return Lam(compile(dom, scope), compile(body, [name] + scope), name)
    if tag == "app":
        return App(compile(t[1], scope), compile(t[2], scope))
    raise KernelError(f"compile: bad named term {t}")


# ===================================================================================================
# LOGIC PRELUDE: the standard connectives as a SOUND set of declared axioms (constructor/eliminator
# types). This is "inductive types" realized for PROOF-CHECKING: the types are exactly those a real
# inductive would give, so building/checking logic proofs is sound. (No iota-computation -- proofs that
# need to COMPUTE with eliminators, e.g. arithmetic, await full computational inductives.)
# ===================================================================================================
_PROP, _TYPE = n_sort(0), n_sort(1)


def logic_env():
    A, B, C, P, a, b, w = (n_v(x) for x in "ABCPabw")
    decls = {
        "False": _PROP,
        "False_elim": n_pi("C", _PROP, n_arr(n_c("False"), n_v("C"))),
        "True": _PROP,
        "trivial": n_c("True"),
        "And": n_arr(_PROP, n_arr(_PROP, _PROP)),
        "And_intro": n_pi("A", _PROP, n_pi("B", _PROP,
                     n_arr(A, n_arr(B, n_app(n_c("And"), A, B))))),
        "And_left": n_pi("A", _PROP, n_pi("B", _PROP, n_arr(n_app(n_c("And"), A, B), A))),
        "And_right": n_pi("A", _PROP, n_pi("B", _PROP, n_arr(n_app(n_c("And"), A, B), B))),
        "Or": n_arr(_PROP, n_arr(_PROP, _PROP)),
        "Or_inl": n_pi("A", _PROP, n_pi("B", _PROP, n_arr(A, n_app(n_c("Or"), A, B)))),
        "Or_inr": n_pi("A", _PROP, n_pi("B", _PROP, n_arr(B, n_app(n_c("Or"), A, B)))),
        "Or_elim": n_pi("A", _PROP, n_pi("B", _PROP, n_pi("C", _PROP,
                   n_arr(n_app(n_c("Or"), A, B),
                   n_arr(n_arr(A, C), n_arr(n_arr(B, C), C)))))),
        "Eq": n_pi("A", _TYPE, n_arr(A, n_arr(A, _PROP))),
        "Eq_refl": n_pi("A", _TYPE, n_pi("a", A, n_app(n_c("Eq"), A, a, a))),
        "Eq_elim": n_pi("A", _TYPE, n_pi("a", A, n_pi("b", A, n_pi("P", n_arr(A, _PROP),
                   n_arr(n_app(n_c("Eq"), A, a, b),
                   n_arr(n_app(P, a), n_app(P, b))))))),
        "Ex": n_pi("A", _TYPE, n_arr(n_arr(A, _PROP), _PROP)),
        "Ex_intro": n_pi("A", _TYPE, n_pi("P", n_arr(A, _PROP), n_pi("w", A,
                    n_arr(n_app(P, w), n_app(n_c("Ex"), A, P))))),
        "Ex_elim": n_pi("A", _TYPE, n_pi("P", n_arr(A, _PROP), n_pi("C", _PROP,
                   n_arr(n_app(n_c("Ex"), A, P),
                   n_arr(n_pi("w", A, n_arr(n_app(P, w), C)), C))))),
    }
    return {name: compile(ty) for name, ty in decls.items()}


# ===================================================================================================
# Selftest + demo
# ===================================================================================================
PROP = Sort(0)


def selftest():
    # identity:  (fun A:Type x:A => x)  :  (forall A:Type, A -> A)
    idterm = Lam(Sort(1), Lam(Var(0), Var(0), "x"), "A")
    idtype = Pi(Sort(1), Pi(Var(0), Var(1), "x"), "A")
    assert check([], idterm, idtype), "identity should type-check"

    # K / weakening:  A -> B -> A  (a real propositional proof, Curry-Howard)
    #   proof: fun (A B:Prop) (a:A) (b:B) => a
    K = Lam(PROP, Lam(PROP, Lam(Var(1), Lam(Var(1), Var(1), "b"), "a"), "B"), "A")
    Kty = Pi(PROP, Pi(PROP, Pi(Var(1), Pi(Var(1), Var(3), "b"), "a"), "B"), "A")
    assert check([], K, Kty), "A->B->A should be provable"

    # SOUNDNESS: a WRONG proof of A->B->A (returns b, i.e. proves A->B->B) must be REJECTED
    wrong = Lam(PROP, Lam(PROP, Lam(Var(1), Lam(Var(1), Var(0), "b"), "a"), "B"), "A")
    assert not has_type([], wrong, Kty), "kernel must reject the bogus proof"
    assert has_type([], wrong, Pi(PROP, Pi(PROP, Pi(Var(1), Pi(Var(1), Var(2), "b"), "a"), "B"), "A")), \
        "the bogus term genuinely proves A->B->B"

    # ill-typed term (apply a non-function) must be rejected by inference
    try:
        infer([], App(PROP, PROP)); raise AssertionError("should have raised")
    except KernelError:
        pass

    # dependent: forall (A:Type)(P:A->Prop)(x:A), P x -> P x   (identity on a dependent proof)
    A = Sort(1)
    Ptype = Pi(Var(0), PROP, "x")                                  # P : A -> Prop
    dterm = Lam(A, Lam(Ptype, Lam(Var(1), Lam(App(Var(1), Var(0)), Var(0), "h"), "x"), "P"), "A")
    dtype = Pi(A, Pi(Ptype, Pi(Var(1), Pi(App(Var(1), Var(0)),
                                          App(Var(2), Var(1)), "h"), "x"), "P"), "A")
    assert check([], dterm, dtype), "dependent forall proof should type-check"

    # beta / definitional equality: (fun x:Prop => x) Prop  ==  Prop
    assert equal(App(Lam(PROP, Var(0), "x"), PROP), PROP), "beta-conversion failed"

    # ---- LOGIC PRELUDE: every declared axiom type is itself well-typed (catches authoring errors) ----
    env = logic_env()
    for name, ty in env.items():
        infer([], ty, env)

    def proves(named_term, named_type):
        return has_type([], compile(named_term), compile(named_type), env)

    # non-contradiction:  forall A:Prop, A -> (A -> False) -> False   (fun A a f => f a)
    assert proves(
        n_lam("A", _PROP, n_lam("a", n_v("A"), n_lam("f", n_arr(n_v("A"), n_c("False")),
              n_app(n_v("f"), n_v("a"))))),
        n_pi("A", _PROP, n_arr(n_v("A"), n_arr(n_arr(n_v("A"), n_c("False")), n_c("False"))))), \
        "non-contradiction A -> ~A -> False should be provable"

    # conjunction:  forall A B:Prop, A -> B -> And A B   (via And_intro)
    assert proves(
        n_lam("A", _PROP, n_lam("B", _PROP, n_lam("a", n_v("A"), n_lam("b", n_v("B"),
              n_app(n_c("And_intro"), n_v("A"), n_v("B"), n_v("a"), n_v("b")))))),
        n_pi("A", _PROP, n_pi("B", _PROP, n_arr(n_v("A"), n_arr(n_v("B"),
             n_app(n_c("And"), n_v("A"), n_v("B"))))))), "And introduction should prove"

    # And is commutative on proofs:  And A B -> And B A
    assert proves(
        n_lam("A", _PROP, n_lam("B", _PROP, n_lam("h", n_app(n_c("And"), n_v("A"), n_v("B")),
              n_app(n_c("And_intro"), n_v("B"), n_v("A"),
                    n_app(n_c("And_right"), n_v("A"), n_v("B"), n_v("h")),
                    n_app(n_c("And_left"), n_v("A"), n_v("B"), n_v("h")))))),
        n_pi("A", _PROP, n_pi("B", _PROP, n_arr(n_app(n_c("And"), n_v("A"), n_v("B")),
             n_app(n_c("And"), n_v("B"), n_v("A")))))), "And commutativity should prove"

    # disjunction intro:  forall A B:Prop, A -> Or A B
    assert proves(
        n_lam("A", _PROP, n_lam("B", _PROP, n_lam("a", n_v("A"),
              n_app(n_c("Or_inl"), n_v("A"), n_v("B"), n_v("a"))))),
        n_pi("A", _PROP, n_pi("B", _PROP, n_arr(n_v("A"),
             n_app(n_c("Or"), n_v("A"), n_v("B")))))), "Or introduction should prove"

    # equality reflexivity:  forall A:Type a:A, Eq A a a
    assert proves(
        n_lam("A", _TYPE, n_lam("a", n_v("A"), n_app(n_c("Eq_refl"), n_v("A"), n_v("a")))),
        n_pi("A", _TYPE, n_pi("a", n_v("A"), n_app(n_c("Eq"), n_v("A"), n_v("a"), n_v("a"))))), \
        "Eq refl should prove"

    # existential intro:  forall A:Type P:A->Prop w:A, P w -> Ex A P
    assert proves(
        n_lam("A", _TYPE, n_lam("P", n_arr(n_v("A"), _PROP), n_lam("w", n_v("A"),
              n_lam("h", n_app(n_v("P"), n_v("w")),
                    n_app(n_c("Ex_intro"), n_v("A"), n_v("P"), n_v("w"), n_v("h")))))),
        n_pi("A", _TYPE, n_pi("P", n_arr(n_v("A"), _PROP), n_pi("w", n_v("A"),
             n_arr(n_app(n_v("P"), n_v("w")), n_app(n_c("Ex"), n_v("A"), n_v("P"))))))), \
        "Ex introduction should prove"

    # SOUNDNESS: a bogus 'proof' of And A B from only a:A (no b) must be REJECTED
    assert not proves(
        n_lam("A", _PROP, n_lam("B", _PROP, n_lam("a", n_v("A"),
              n_app(n_c("And_intro"), n_v("A"), n_v("B"), n_v("a"), n_v("a"))))),
        n_pi("A", _PROP, n_pi("B", _PROP, n_arr(n_v("A"),
             n_app(n_c("And"), n_v("A"), n_v("B")))))), "kernel must reject the bogus And proof"

    print("kernel selftest OK")


def demo():
    print("kernel demo -- a tiny SOUND dependent type theory (Lean's design, in-house)\n")
    cases = [
        ("identity  fun A:Type x:A => x",
         Lam(Sort(1), Lam(Var(0), Var(0), "x"), "A"),
         Pi(Sort(1), Pi(Var(0), Var(1), "x"), "A")),
        ("weakening A -> B -> A   (fun A B a b => a)",
         Lam(PROP, Lam(PROP, Lam(Var(1), Lam(Var(1), Var(1), "b"), "a"), "B"), "A"),
         Pi(PROP, Pi(PROP, Pi(Var(1), Pi(Var(1), Var(3), "b"), "a"), "B"), "A")),
    ]
    for label, term, ty in cases:
        ok = has_type([], term, ty)
        print(f"  [{'PROVED ' if ok else 'REJECT '}] {label}")
        print(f"            proof : {show(term)}")
        print(f"            type  : {show(ty)}")
    # a rejected bogus proof
    bogus = Lam(PROP, Lam(PROP, Lam(Var(1), Lam(Var(1), Var(0), "b"), "a"), "B"), "A")
    claim = Pi(PROP, Pi(PROP, Pi(Var(1), Pi(Var(1), Var(3), "b"), "a"), "B"), "A")
    print(f"\n  [{'PROVED ' if has_type([], bogus, claim) else 'REJECT '}] bogus 'proof' of A->B->A "
          f"(actually returns b) -- kernel rejects it")


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
