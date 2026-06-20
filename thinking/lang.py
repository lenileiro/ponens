#!/usr/bin/env python3
"""LOTA -- a small, expressive Language Of Thought & Action for the agent (working name; rename freely).

This is THE FOUNDATION step (see thinking/C2_ROADMAP.md): before learning any natural language, give
the agent its own language -- one that is SMALL (a handful of orthogonal primitives), EXPRESSIVE
(events, actions, quantifiers, modality, tense, attitudes -- all by COMPOSITION), and -- decisively --
EXECUTABLE/VERIFIABLE. Our evidence says the verifiable, compositional, typed core is the part that
GENERALIZES; raw surface English is the part that doesn't. So we make the generalizing thing the
substrate. Natural language then becomes a MAPPING problem (surface <-> LOTA), not from-scratch
comprehension: parse (NL->LOTA) . reason (in LOTA, proof-checked) . generate (LOTA->NL, faithful).

Design (locked):
  * grounding   : UNIFIED knowledge + action -- events subsume facts AND acts.
  * syntax      : S-expressions (trivial to parse, regular, easy for a small model to emit).
  * expressivity: FULL -- relations, events, actions, not/and/or/if/iff, forall/exists (+restrictors),
                  modality (can/must/may), attitudes (believe/want/goal), tense/aspect modifiers.

Everything desugars to ONE first-order logical core over ground atoms, evaluated against a WORLD
(atoms + Datalog rules) under closed-world negation-as-failure -- reusing datalog.py (the verified
reasoning engine). Actions execute via schemas (precondition + effects) that mutate the world.

  python -m thinking.lang --selftest
  python -m thinking.lang --demo

Surface examples:
  (isa robin bird)                                  ; a relation / fact
  (can (fly :agent robin))                          ; modality over an event
  (not (can (fly :agent penguin)))                  ; negation-as-failure
  (event chase :agent (the cat) :patient (a mouse) :loc (the garden) :tense past)
  (forall (?x bird) (can (fly :agent ?x)))          ; restricted universal
  (do pickup :agent self :patient (the block :color red))   ; an action to execute
  (goal (holding :agent self :patient blockA))      ; an attitude
"""
import argparse
import itertools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datalog import Datalog  # noqa: E402  (the verified reasoning engine: closure / entails / proofs)


# ===================================================================================================
# 1. READER : S-expression text -> nested Python lists (atoms are str tokens; sublists are lists)
# ===================================================================================================
def tokenize(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\r\n":
            i += 1
        elif c == ";":                          # line comment
            while i < n and s[i] != "\n":
                i += 1
        elif c in "()":
            out.append(c); i += 1
        elif c == '"':                          # quoted literal -> kept with quotes as one token
            j = i + 1
            while j < n and s[j] != '"':
                j += 1
            out.append(s[i:j + 1]); i = j + 1
        else:
            j = i
            while j < n and s[j] not in " \t\r\n();":
                j += 1
            out.append(s[i:j]); i = j
    return out


def read(s):
    """Parse exactly one S-expression; error on trailing junk. Returns nested lists / str atoms."""
    toks = tokenize(s)
    pos = 0

    def parse():
        nonlocal pos
        if pos >= len(toks):
            raise SyntaxError("unexpected end of input")
        t = toks[pos]; pos += 1
        if t == "(":
            lst = []
            while pos < len(toks) and toks[pos] != ")":
                lst.append(parse())
            if pos >= len(toks):
                raise SyntaxError("missing ')'")
            pos += 1                              # consume ')'
            return lst
        if t == ")":
            raise SyntaxError("unexpected ')'")
        return t

    node = parse()
    if pos != len(toks):
        raise SyntaxError(f"trailing tokens after expression: {toks[pos:]}")
    return node


def write(node):
    """Canonical S-expression text for a parsed node (round-trippable)."""
    if isinstance(node, list):
        return "(" + " ".join(write(x) for x in node) + ")"
    return node


# ===================================================================================================
# 2. VOCABULARY of constructs (the SMALL set; everything else is a user predicate / entity)
# ===================================================================================================
DETS = {"a", "an", "some", "the", "all", "every", "each", "no"}
EXIST_DETS = {"a", "an", "some", "the"}
UNIV_DETS = {"all", "every", "each"}
CONNECTIVES = {"and", "or", "not", "if", "iff"}
QUANTS = {"forall", "exists"}
MODALS = {"can", "must", "may"}
ATTITUDES = {"believe", "want", "goal"}
EVENTISH = {"event", "do"}
MODIFIER_ROLES = {"tense", "aspect", "time"}     # recorded; not truth-evaluated in v1
RESERVED = (DETS | CONNECTIVES | QUANTS | MODALS | ATTITUDES | EVENTISH)


def is_var(t):
    return isinstance(t, str) and t.startswith("?")


def is_role(t):
    return isinstance(t, str) and t.startswith(":")


# ===================================================================================================
# 3. CHECKER : light structural / category validation (Term vs Formula). Returns a list of errors.
# ===================================================================================================
def check(node):
    errs = []

    def is_term(x):
        if isinstance(x, str):
            return not is_role(x)
        return isinstance(x, list) and len(x) >= 2 and x[0] in DETS

    def chk_formula(x):
        if isinstance(x, str):
            errs.append(f"bare atom '{x}' used where a formula (clause) is expected"); return
        if not x:
            errs.append("empty () is not a formula"); return
        h = x[0]
        if h in ("and", "or"):
            if len(x) < 2:
                errs.append(f"({h} ...) needs >=1 clause")
            for c in x[1:]:
                chk_formula(c)
        elif h == "not":
            if len(x) != 2:
                errs.append("(not phi) takes exactly one clause")
            else:
                chk_formula(x[1])
        elif h in ("if", "iff"):
            if len(x) != 3:
                errs.append(f"({h} a b) takes two clauses")
            else:
                chk_formula(x[1]); chk_formula(x[2])
        elif h in QUANTS:
            if len(x) != 3:
                errs.append(f"({h} VAR body) or ({h} (VAR KIND) body)")
            else:
                b = x[1]
                if isinstance(b, list):
                    if len(b) != 2 or not is_var(b[0]):
                        errs.append(f"{h} restrictor must be (?var KIND)")
                elif not is_var(b):
                    errs.append(f"{h} binder must be a ?variable, got {b!r}")
                chk_formula(x[2])
        elif h in MODALS:
            if len(x) != 2:
                errs.append(f"({h} phi) takes one clause")
            else:
                chk_formula(x[1])
        elif h == "goal":
            if len(x) != 2:
                errs.append("(goal phi) takes one clause")
            else:
                chk_formula(x[1])
        elif h in ("believe", "want"):
            roles = _roles(x[1:])
            if ":agent" not in roles or ":that" not in roles:
                errs.append(f"({h} :agent A :that phi) needs :agent and :that")
            else:
                chk_formula(roles[":that"])
        elif h in EVENTISH:
            if len(x) < 2 or isinstance(x[1], list):
                errs.append(f"({h} TYPE :role term ...) needs a type symbol")
            _chk_roles(x[2:], errs, is_term)
        else:                                     # plain predication: (pred term ...) or role-form
            if any(is_role(t) for t in x[1:]):
                _chk_roles(x[1:], errs, is_term)   # role-form: roles start right after the head
            else:
                for t in x[1:]:
                    if not is_term(t):
                        errs.append(f"predicate '{h}': '{write(t)}' is not a valid term")

    chk_formula(node)
    return errs


def _chk_roles(rest, errs, is_term):
    i = 0
    while i < len(rest):
        if not is_role(rest[i]):
            errs.append(f"expected a :role, got {write(rest[i])}"); i += 1; continue
        if i + 1 >= len(rest):
            errs.append(f"role {rest[i]} has no value"); break
        i += 2


def _roles(rest):
    """Ordered {':role': value} from a flat [:r v :r v ...] list."""
    out = {}
    i = 0
    while i < len(rest):
        if is_role(rest[i]) and i + 1 < len(rest):
            out[rest[i]] = rest[i + 1]; i += 2
        else:
            i += 1
    return out


# ===================================================================================================
# 4. DESUGAR : surface S-expr -> first-order CORE formula over ground atoms.
#    Core forms (tuples):
#      ('atom', pred, (arg, ...))
#      ('and'|'or', f, ...)   ('not', f)   ('if', a, b)   ('iff', a, b)
#      ('exists'|'forall', '?v', f)
#      ('must'|'can'|'may', f)
#      ('goal', f)   ('believe'|'want', agentConst, f)
#    Determiner NPs are quantifier-RAISED; events introduce an existential event variable.
# ===================================================================================================
class _Fresh:
    def __init__(self):
        self.n = 0

    def __call__(self, prefix="?x"):
        self.n += 1
        return f"{prefix}{self.n}"


def desugar(node):
    fresh = _Fresh()
    return _form(node, fresh)


def _form(x, fresh):
    if isinstance(x, str):
        raise SyntaxError(f"bare atom '{x}' is not a formula")
    h = x[0]
    if h in ("and", "or"):
        return (h, *[_form(c, fresh) for c in x[1:]])
    if h == "not":
        return ("not", _form(x[1], fresh))
    if h in ("if", "iff"):
        return (h, _form(x[1], fresh), _form(x[2], fresh))
    if h in QUANTS:
        b = x[1]
        if isinstance(b, list):                   # restricted: (forall (?x KIND) body)
            var, kind = b[0], b[1]
            restr = ("atom", kind, (var,))
            body = _form(x[2], fresh)
            return ("forall", var, ("if", restr, body)) if h == "forall" \
                else ("exists", var, ("and", restr, body))
        return (h, b, _form(x[2], fresh))
    if h in MODALS:
        return (h, _form(x[1], fresh))
    if h == "goal":
        return ("goal", _form(x[1], fresh))
    if h in ("believe", "want"):
        r = _roles(x[1:])
        return (h, r[":agent"], _form(r[":that"], fresh))
    if h in EVENTISH:
        return _event(x, fresh)
    return _pred(x, fresh)


def _term(t, binders, fresh):
    """Return a core term (const/var); NP determiners append (det, kind, var, restr_atoms) to binders."""
    if isinstance(t, str):
        return t                                  # const or ?var or "literal"
    if isinstance(t, list) and t and t[0] in DETS:
        det, kind = t[0], t[1]
        v = fresh("?e" if kind in ("event",) else "?x")
        extra = []                                # NP modifiers: (the block :color red) -> color(v,red)
        for rname, val in _roles(t[2:]).items():
            extra.append(("atom", rname[1:], (v, _term(val, binders, fresh))))
        binders.append((det, kind, v, extra))
        return v
    raise SyntaxError(f"not a valid term: {write(t)}")


def _wrap(body, binders):
    """Apply collected NP quantifiers around a body (innermost binder = last collected)."""
    for det, kind, v, extra in reversed(binders):
        restr = ("and", ("atom", kind, (v,)), *extra) if extra else ("atom", kind, (v,))
        if det in EXIST_DETS:
            body = ("exists", v, ("and", restr, body))
        elif det in UNIV_DETS:
            body = ("forall", v, ("if", restr, body))
        elif det == "no":
            body = ("not", ("exists", v, ("and", restr, body)))
    return body


def _pred(x, fresh):
    head, rest = x[0], x[1:]
    if any(is_role(t) for t in rest):             # role-form predication -> treat like an event
        return _event(["event", head, *rest], fresh)
    binders = []
    args = tuple(_term(t, binders, fresh) for t in rest)
    return _wrap(("atom", head, args), binders)


def _event(x, fresh):
    etype, rest = x[1], x[2:]
    evar = fresh("?e")
    binders = []
    conj = [("atom", etype, (evar,))]
    for rname, val in _roles(rest).items():
        name = rname[1:]
        filler = val if (name in MODIFIER_ROLES and isinstance(val, str)) \
            else _term(val, binders, fresh)
        conj.append(("atom", name, (evar, filler)))
    body = ("and", *conj) if len(conj) > 1 else conj[0]
    return ("exists", evar, _wrap(body, binders))


# ===================================================================================================
# 5. WORLD + EVALUATOR : closed-world truth of a core formula against atoms + Datalog rules.
# ===================================================================================================
class World:
    def __init__(self, atoms=(), rules=(), schemas=None, goals=(), attitudes=None, extra_domain=()):
        self.atoms = set(atoms)                   # ground facts: (pred, (args...))
        self.rules = list(rules)                  # Datalog rules: (head, [body...])
        self.schemas = dict(schemas or {})        # action type -> schema dict
        self.goals = set(goals)                   # core formulas registered as goals
        self.attitudes = dict(attitudes or {})    # ('believe'|'want', agent) -> set(core formulas)
        self.extra_domain = set(extra_domain)
        self._recompute()

    def _recompute(self):
        self.closure = Datalog(self.rules).closure(self.atoms)[0] if self.rules else set(self.atoms)
        dom = set(self.extra_domain)
        for _p, args in self.closure:
            dom.update(a for a in args if not is_var(a))
        self.domain = dom

    def add(self, atom):
        self.atoms.add(atom); self._recompute()

    def remove(self, atom):
        self.atoms.discard(atom); self._recompute()


def _subst(f, var, const):
    tag = f[0]
    if tag == "atom":
        return ("atom", f[1], tuple(const if a == var else a for a in f[2]))
    if tag in ("exists", "forall"):
        if f[1] == var:                           # shadowed
            return f
        return (tag, f[1], _subst(f[2], var, const))
    if tag in ("believe", "want"):
        return (tag, f[1], _subst(f[2], var, const))
    return (tag, *[_subst(c, var, const) for c in f[1:]])


def evaluate(f, world):
    """Closed-world truth value (bool). Unprovable == false (negation-as-failure)."""
    tag = f[0]
    if tag == "atom":
        if any(is_var(a) for a in f[1]):
            raise ValueError(f"free variable in atom at eval time: {f}")
        return (f[1], f[2]) in world.closure
    if tag == "and":
        return all(evaluate(c, world) for c in f[1:])
    if tag == "or":
        return any(evaluate(c, world) for c in f[1:])
    if tag == "not":
        return not evaluate(f[1], world)
    if tag == "if":
        return (not evaluate(f[1], world)) or evaluate(f[2], world)
    if tag == "iff":
        return evaluate(f[1], world) == evaluate(f[2], world)
    if tag == "exists":
        return any(evaluate(_subst(f[2], f[1], d), world) for d in world.domain)
    if tag == "forall":
        return all(evaluate(_subst(f[2], f[1], d), world) for d in world.domain)
    if tag == "must":                             # necessity: provable in the closure
        return evaluate(f[1], world)
    if tag in ("can", "may"):                     # possibility: not explicitly ruled out (open-world)
        return _possible(f[1], world)
    if tag == "goal":
        return f[1] in world.goals
    if tag in ("believe", "want"):
        return f[2] in world.attitudes.get((tag, f[1]), set())
    raise ValueError(f"cannot evaluate core form: {f}")


def _possible(f, world):
    """Open-world POSSIBILITY: φ is possible unless EXPLICITLY ruled out. Explicit impossibility of an
    atom (p, args) is recorded as the fact ('~p', args) in the world. This is genuinely distinct from
    closed-world negation-as-failure: a merely-UNKNOWN φ is still possible (can), but its plain
    negation (not φ) is true by failure. That contrast is the whole point of having modality."""
    tag = f[0]
    if tag == "atom":
        return ("~" + f[1], f[2]) not in world.closure
    if tag == "not":
        return not evaluate(f[1], world)          # ¬φ possible iff φ is not necessary (unprovable)
    if tag == "and":
        return all(_possible(c, world) for c in f[1:])
    if tag == "or":
        return any(_possible(c, world) for c in f[1:])
    if tag == "if":
        return _possible(("or", ("not", f[1]), f[2]), world)
    if tag == "iff":
        return True
    if tag == "exists":
        return any(_possible(_subst(f[2], f[1], d), world) for d in world.domain)
    if tag == "forall":
        return all(_possible(_subst(f[2], f[1], d), world) for d in world.domain)
    return evaluate(f, world) or True             # nested must/goal/etc.: if not known-true, allow


def truth(surface, world):
    """Parse + check + desugar + evaluate a surface string against the world."""
    node = read(surface)
    errs = check(node)
    if errs:
        raise ValueError("invalid LOTA: " + "; ".join(errs))
    return evaluate(desugar(node), world)


# ===================================================================================================
# 6. ACTIONS : execute a (do TYPE :role filler ...) against the world via a schema (pre + effects).
#    A schema: {"params": [role,...], "pre": <surface str using ?role>, "add": [..], "del": [..]}
#    where add/del are atom templates (pred, (args using ?role or const)).
# ===================================================================================================
def execute(surface, world):
    node = read(surface)
    if not node or node[0] != "do":
        raise ValueError("execute expects a (do TYPE ...) action")
    atype = node[1]
    if atype not in world.schemas:
        raise ValueError(f"no schema for action '{atype}'")
    sch = world.schemas[atype]
    roles = {r[1:]: v for r, v in _roles(node[2:]).items()}
    bind = {f"?{k}": (v if isinstance(v, str) else None) for k, v in roles.items()}
    if any(v is None for v in bind.values()):
        raise ValueError("v1 actions require constant role fillers (no NP fillers yet)")

    def fill(args):
        return tuple(bind.get(a, a) for a in args)

    pre = sch.get("pre")
    if pre is not None:
        pf = desugar(read(pre))
        for k, v in bind.items():                 # substitute role vars in the precondition
            pf = _subst(pf, k, v)
        if not evaluate(pf, world):
            return False, world, "precondition failed"
    for (p, args) in sch.get("del", []):
        world.remove((p, fill(args)))
    for (p, args) in sch.get("add", []):
        world.add((p, fill(args)))
    return True, world, "ok"


# ===================================================================================================
# 7. SELFTEST + DEMO
# ===================================================================================================
def _toy_world():
    rules = [(("animal", ("?x",)), [("bird", ("?x",))]),     # birds are animals
             (("animal", ("?x",)), [("mammal", ("?x",))]),   # mammals are animals
             (("mammal", ("?x",)), [("cat", ("?x",))])]      # cats are mammals
    atoms = [("bird", ("robin",)), ("bird", ("penguin",)), ("bird", ("kiwi",)),
             ("cat", ("c1",)), ("mouse", ("m1",)),           # c1 is a cat (entity), m1 a mouse
             ("can_fly", ("robin",)),                        # robin flies; penguin merely UNKNOWN
             ("~can_fly", ("kiwi",)),                        # kiwi is EXPLICITLY unable to fly
             # event e1: the cat (c1) chased a mouse (m1)
             ("chase", ("e1",)), ("agent", ("e1", "c1")), ("patient", ("e1", "m1")),
             ("block", ("blockA",)), ("clear", ("blockA",)), ("red", ("blockA",))]
    schemas = {"pickup": {"params": ["agent", "patient"],
                          "pre": "(clear ?patient)",
                          "add": [("holding", ("?agent", "?patient"))],
                          "del": [("clear", ("?patient",))]}}
    return World(atoms=atoms, rules=rules, schemas=schemas)


def selftest():
    # --- reader round-trips ---
    for s in ["(isa robin bird)",
              "(event chase :agent (the cat) :patient (a mouse) :tense past)",
              "(forall (?x bird) (can (fly :agent ?x)))"]:
        assert write(read(s)) == s, ("round-trip", s, write(read(s)))

    # --- checker catches malformed input ---
    assert check(read("(and robin bird)")), "checker should reject bare atoms as clauses"
    assert check(read("(forall x (bird x))")), "checker should reject non-?var binder"
    assert check(read("(isa robin bird)")) == [], "valid form must pass the checker"

    w = _toy_world()

    # --- relations + Datalog closure (reuses the verified engine) ---
    assert truth("(animal robin)", w)             # derived: bird -> animal
    assert truth("(animal c1)", w)                # derived: cat -> mammal -> animal (2 hops)
    assert not truth("(animal nobody)", w)        # unknown entity -> false (CWA)

    # --- quantifiers (+ restrictor) ---
    assert truth("(exists ?x (bird ?x))", w)
    assert truth("(forall (?x bird) (animal ?x))", w)        # every bird is an animal
    assert not truth("(forall (?x animal) (bird ?x))", w)    # not every animal is a bird (c1 is a cat)

    # --- negation-as-failure ---
    assert truth("(not (can_fly penguin))", w)               # unprovable -> false by failure
    assert truth("(can_fly robin)", w)

    # --- modality: necessity=provable; possibility=NOT explicitly ruled out (open-world) ---
    assert truth("(must (animal robin))", w)                 # provable -> necessary
    assert not truth("(must (can_fly penguin))", w)          # not provable -> not necessary
    assert truth("(can (can_fly penguin))", w)               # merely UNKNOWN -> still possible
    assert not truth("(can (can_fly kiwi))", w)              # EXPLICITLY ruled out (~can_fly) -> impossible
    # the crucial contrast: penguin is can_fly-FALSE by failure, yet can_fly is POSSIBLE for it
    assert truth("(not (can_fly penguin))", w) and truth("(can (can_fly penguin))", w)

    # --- events (existential over the event variable + quantified roles) ---
    assert truth("(event chase :agent (the cat) :patient (a mouse))", w)   # e1 matches
    assert not truth("(event chase :agent (the mouse) :patient (a cat))", w)

    # --- attitudes (structural: is this registered as a goal?) ---
    w.goals.add(desugar(read("(holding self blockA)")))
    assert truth("(goal (holding self blockA))", w)
    assert not truth("(goal (holding self blockB))", w)

    # --- ACTION execution mutates the world (grounding) ---
    assert not truth("(holding self blockA)", w)
    ok, w, msg = execute("(do pickup :agent self :patient blockA)", w)
    assert ok, ("pickup failed", msg)
    assert truth("(holding self blockA)", w), "effect not applied"
    assert not truth("(clear blockA)", w), "deleted precondition still true"
    ok2, w, msg2 = execute("(do pickup :agent self :patient blockA)", w)
    assert not ok2, "precondition (clear) should now fail"

    print("lang selftest OK")


def demo():
    w = _toy_world()
    print("LOTA demo -- a small, expressive, EXECUTABLE agent language\n")
    rows = [
        ("(animal robin)", "derived fact (bird->animal via the engine)"),
        ("(forall (?x bird) (animal ?x))", "restricted universal"),
        ("(not (can_fly penguin))", "negation-as-failure"),
        ("(must (animal robin))", "necessity = provable"),
        ("(can (can_fly penguin))", "possibility = CWA-consistent"),
        ("(event chase :agent (the cat) :patient (a mouse))", "event w/ quantified roles"),
        ("(can (can_fly kiwi))", "impossible: explicitly ruled out (~can_fly)"),
    ]
    for s, why in rows:
        print(f"  {truth(s, w)!s:>5}  {s:<48} ; {why}")
    print("\n  action: (do pickup :agent self :patient blockA)")
    ok, w, msg = execute("(do pickup :agent self :patient blockA)", w)
    print(f"    executed -> {ok} ({msg});  now (holding self blockA) = {truth('(holding self blockA)', w)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--eval", metavar="SEXPR", help="evaluate one form against the toy world")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    if args.demo:
        demo(); return 0
    if args.eval:
        print(truth(args.eval, _toy_world())); return 0
    ap.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
