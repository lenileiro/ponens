"""StepChecker: the proof CHECKER gating the decode loop.

Strictly local validation — body membership in the known set plus rule instantiation — never a
closure call, so the engine can't leak the answer. The checker is generic over any Datalog
ruleset: nothing here knows about 'far' or chains.
"""
import sys
import os
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datalog import _ground as _ground_sub
from datalog import _rename, _unify, _unify2


def _ground(atom):
    return not any(str(a).startswith("?") for a in atom[1])


def _is_var_term(term):
    return isinstance(term, str) and term.startswith("?")


def _subst(atom, subst):
    return (atom[0], tuple(subst.get(a, a) for a in atom[1]))


def _body_instances(patterns, facts_by_pred):
    """All ground body instances matching rule body patterns against known facts."""
    partials = [({}, [])]
    for pat in patterns:
        nxt = []
        for subst, body in partials:
            for fact in facts_by_pred.get(pat[0], ()):
                s = _unify(pat, fact, dict(subst))
                if s is not None:
                    nxt.append((s, body + [fact]))
        partials = nxt
        if not partials:
            break
    return partials


def instantiates(rules, head, body):
    """Is (head <- body) a ground instance of some rule? (Pure shape check; no fact membership.)"""
    for hp, bps in rules:
        if len(bps) != len(body):
            continue
        s0 = _unify(hp, head, {})
        if s0 is None:
            continue
        for perm in permutations(body):
            s, ok = dict(s0), True
            for pat, f in zip(bps, perm):
                s = _unify(pat, f, s)
                if s is None:
                    ok = False
                    break
            if ok:
                return True
    return False


class StepChecker:
    def __init__(self, rules):
        self.rules = rules

    def valid_step(self, head, body, known):
        """Is (head <- body) a ground instance of some rule with every body atom already known?"""
        return all(f in known for f in body) and instantiates(self.rules, head, body)

    def valid_answer(self, goal_pred, head, answer, known, edb):
        """Local completeness: the trace derived (goal_pred head answer) and `answer` has no
        outgoing base-relation edge left to follow (premature-stop guard)."""
        if (goal_pred, (head, answer)) not in known:
            return False
        return not any(f[1][0] == answer for f in edb if f[0] != goal_pred)


class GoalChecker:
    """FORWARD (evidence-first) checker for kinship traces.

    State: the KNOWN set (checked EDB facts + derived atoms). 'check F' grounds an EDB fact; 'think H
    needs B' derives H when every body atom is already known and (H <- B) instantiates a rule
    (or an arithmetic builtin verifies the value). The answer is valid only when a matching fact
    about the question pair has been DERIVED -- conclusions come LAST (the goal-directed/
    preorder variant put the answer at token 3 of line 1 = answer-only supervision; v7/v8
    showed it unlearnable). All local -- the checker never proves anything itself."""

    def __init__(self, rules, answer_preds, builtins=None):
        self.rules = rules
        self.answer_preds = set(answer_preds)
        self.builtins = builtins or {}                     # pred -> fn(head, body, pair) -> bool
        #                                                    (arithmetic VERIFIED, never computed)
        self.recursive = {h[0] for h, b in rules if any(a[0] == h[0] for a in b)}

    def new_state(self, pair, edb, goal_pred=None, extra_rules=()):
        return {"known": set(), "edb": set(edb), "derived": [], "pair": tuple(pair),
                "goal_pred": goal_pred, "extra_rules": list(extra_rules), "seen": set(),
                "support_atoms": "pending", "support_ranks": {},
                "support_plan": None, "support_pos": 0}

    def valid_step_state(self, st, typ, head, body):
        """Non-mutating version of step() for candidate generation. Rejections record their
        cause in st['why'] (the taxonomy that tells us WHERE per-line error lives)."""
        body = tuple(body)
        if (typ, head, body) in st["seen"]:
            st["why"] = "duplicate"
            return False
        if typ == "check":
            if head in st["edb"]:
                return True
            st["why"] = "check-not-edb"
            return False
        if typ != "think":
            st["why"] = "bad-line-type"
            return False
        if not all(f in st["known"] for f in body):
            st["why"] = "premise-unknown"
            return False
        if head[0] in self.builtins:
            if not self.builtins[head[0]](head, body, st["pair"]):
                st["why"] = "builtin-reject"
                return False
        elif not instantiates(self.rules + st["extra_rules"], head, body):
            st["why"] = "no-rule"
            return False
        elif (head[0] in self.answer_preds and head[0] not in self.recursive
              and tuple(head[1]) != st["pair"]):
            st["why"] = "goal-anchor"
            return False
        return True

    def answer_candidates(self, st):
        """Generic valid answer words for the current verified state."""
        x, _z = st["pair"]
        gp = st["goal_pred"]
        if gp in self.builtins:
            vals = sorted({h[1][1] for h in st["derived"] if h[0] == gp and h[1][0] == x})
            return [v for v in vals if self.valid_answer(st, v)]
        extra_heads = {r[0][0] for r in st["extra_rules"]}
        if gp in extra_heads:
            vals = sorted({h[1][1] for h in st["known"] if h[0] == gp and h[1][0] == x})
            return [v for v in vals if self.valid_answer(st, v)]
        return [a for a in sorted(self.answer_preds) if self.valid_answer(st, a)]

    def relevant_predicates(self, st):
        """Backward predicate dependency slice for the current goal predicate."""
        if st["goal_pred"] in self.builtins:
            return None                                    # builtin dependencies are opaque here
        rel = {st["goal_pred"]}
        changed = True
        while changed:
            changed = False
            for hp, bps in self.rules + st["extra_rules"]:
                if hp[0] not in rel:
                    continue
                for b in bps:
                    if b[0] not in rel:
                        rel.add(b[0])
                        changed = True
        return rel

    def support_atoms(self, st):
        """Ground atoms on one lazy proof-support path for the current goal.

        This is the proof analogue of a narrowed type environment: it does not know relation
        names or trace grammar, but it uses the loaded Datalog rules lazily to remove atoms that
        cannot contribute to the requested ground goal. Builtin predicates fall back to
        unconstrained candidate generation because their dependencies are Python-side verifier
        functions.
        """
        cached = st.get("support_atoms", "pending")
        if cached != "pending":
            return cached
        if st["goal_pred"] in self.builtins:
            st["support_atoms"] = None
            return None
        target = (st["goal_pred"], tuple(st["pair"]))
        facts_by_pred = {}
        for fact in st["edb"]:
            facts_by_pred.setdefault(fact[0], []).append(fact)
        for facts in facts_by_pred.values():
            facts.sort()
        fact_index = self._fact_index(facts_by_pred)

        self._support_rn = 0
        max_depth = max(8, len(st["edb"]) + len(self.rules) + len(st["extra_rules"]) + 4)
        old_recursion_limit = sys.getrecursionlimit()
        if old_recursion_limit < max_depth * 4 + 100:
            sys.setrecursionlimit(max_depth * 4 + 100)
        try:
            proof = self._prove_support(
                target, {}, facts_by_pred, fact_index, self.rules + st["extra_rules"],
                max_depth, frozenset())
        finally:
            if sys.getrecursionlimit() != old_recursion_limit:
                sys.setrecursionlimit(old_recursion_limit)
        if proof is None:
            st["support_atoms"] = None
            st["support_ranks"] = {}
            return None
        _fact, _subst, support, ranks, plan = proof
        st["support_atoms"] = support
        st["support_ranks"] = ranks
        st["support_plan"] = self._dedupe_support_plan(plan)
        return support

    def _fact_index(self, facts_by_pred):
        """Predicate-local inverted index for generic fact lookup during proof search."""
        out = {}
        for pred, facts in facts_by_pred.items():
            by_pos = {}
            for fact in facts:
                for i, term in enumerate(fact[1]):
                    by_pos.setdefault(i, {}).setdefault(term, []).append(fact)
            out[pred] = by_pos
        return out

    def _fact_candidates(self, atom, facts_by_pred, fact_index):
        """Facts that can still unify with atom, using the narrowest bound argument index."""
        facts = facts_by_pred.get(atom[0], ())
        by_pos = fact_index.get(atom[0], {})
        narrowed = None
        for i, term in enumerate(atom[1]):
            if _is_var_term(term):
                continue
            cur = by_pos.get(i, {}).get(term, ())
            if narrowed is None or len(cur) < len(narrowed):
                narrowed = cur
                if not narrowed:
                    break
        return narrowed if narrowed is not None else facts

    def _prove_support(self, goal, subst, facts_by_pred, fact_index, rules, depth, path):
        """Find one proof path lazily; return (ground_fact, subst, support_atoms, ranks, plan)."""
        g = _ground_sub(goal, subst)
        for fact in self._fact_candidates(g, facts_by_pred, fact_index):
            s = _unify2(g, fact, subst)
            if s is not None:
                gf = _ground_sub(g, s)
                if _ground(gf):
                    return gf, s, {fact}, {fact: 0}, [("check", fact, ())]
        if depth <= 0:
            return None
        key = g
        if key in path:
            return None
        for head, body in rules:
            self._support_rn += 1
            rn = self._support_rn
            rhead = _rename(head, rn)
            s = _unify2(rhead, g, subst)
            if s is None:
                continue
            cur = s
            support, ranks, body_facts, plan = set(), {}, [], []
            ok = True
            for atom in tuple(_rename(a, rn) for a in body):
                proof = self._prove_support(atom, cur, facts_by_pred, fact_index,
                                            rules, depth - 1,
                                            path | {key})
                if proof is None:
                    ok = False
                    break
                bf, cur, bs, br, bp = proof
                body_facts.append(bf)
                support.update(bs)
                ranks.update(br)
                plan.extend(bp)
            if not ok:
                continue
            hf = _ground_sub(rhead, cur)
            if not _ground(hf):
                continue
            support.add(hf)
            ranks[hf] = 1 + max((ranks.get(bf, 0) for bf in body_facts), default=0)
            plan.append(("think", hf, tuple(body_facts)))
            return hf, cur, support, ranks, plan
        return None

    def _dedupe_support_plan(self, plan):
        """Remove repeated proof obligations while preserving the forward proof order."""
        out, seen = [], set()
        for typ, head, body in plan:
            step = (typ, head, tuple(body))
            if step in seen:
                continue
            seen.add(step)
            out.append(step)
        return out

    def _support_frontier(self, st):
        """Next executable steps from the cached support proof, if a proof plan exists."""
        if self.support_atoms(st) is None:
            return None
        plan = st.get("support_plan") or ()
        pos = st.get("support_pos", 0)
        while pos < len(plan):
            typ, head, body = plan[pos]
            if (typ, head, body) in st["seen"] or head in st["known"]:
                pos += 1
                continue
            break
        st["support_pos"] = pos

        out = []
        for typ, head, body in plan[pos:]:
            if (typ, head, body) in st["seen"] or head in st["known"]:
                continue
            if typ == "think" and not all(f in st["known"] for f in body):
                break
            if self.valid_step_state(st, typ, head, body):
                out.append((typ, head, body))
            break
        return out

    def candidate_rank(self, st, step):
        """Generic proof-frontier rank: lower atoms are closer to checked evidence."""
        return st.get("support_ranks", {}).get(step[1], 10 ** 9)

    def candidate_steps(self, st, include_checks=True, goal_pruned=True,
                        relevance_pruned=True):
        """Generic legal local proof steps from the current checker state.

        The generator is rule-driven: it enumerates unchecked EDB facts and ground instances of
        the loaded Datalog rules whose bodies are already known. It does not know about kinship,
        ancestor, or any task-specific grammar.

        relevance_pruned=False enumerates by VALIDITY ONLY (no goal-dependency slice, no proof
        support): the honest candidate set for masked decoding, where choosing the relevant
        step among all legal ones must remain the model's job."""
        if goal_pruned and relevance_pruned:
            frontier = self._support_frontier(st)
            if frontier is not None:
                return frontier

        candidates = []
        relevant = self.relevant_predicates(st) if relevance_pruned else None
        support = (self.support_atoms(st)
                   if goal_pruned and relevance_pruned else None)
        if include_checks:
            for fact in sorted(st["edb"] - st["known"]):
                if relevant is not None and fact[0] not in relevant:
                    continue
                if support is not None and fact not in support:
                    continue
                body = ()
                if self.valid_step_state(st, "check", fact, body):
                    candidates.append(("check", fact, body))

        facts_by_pred = {}
        for fact in st["known"]:
            facts_by_pred.setdefault(fact[0], []).append(fact)
        for facts in facts_by_pred.values():
            facts.sort()

        for hp, bps in self.rules + st["extra_rules"]:
            if relevant is not None and hp[0] not in relevant:
                continue
            for subst, body in _body_instances(bps, facts_by_pred):
                head = _subst(hp, subst)
                body = tuple(body)
                if not _ground(head) or head in st["known"]:
                    continue
                if support is not None and (head not in support
                                            or any(f not in support for f in body)):
                    continue
                if self.valid_step_state(st, "think", head, body):
                    candidates.append(("think", head, body))
        candidates.sort(key=lambda x: (self.candidate_rank(st, x), x[0], x[1], x[2]))
        return candidates

    def step(self, st, typ, head, body):
        """Validate one line against the state; mutate ONLY when valid. Returns bool.
        Duplicate lines are rejected: gold traces never repeat, and looping on a valid check
        was eating the whole line budget (rung A-anon demo)."""
        body = tuple(body)
        if not self.valid_step_state(st, typ, head, body):
            return False
        if typ == "check":
            st["known"].add(head)
            st["seen"].add((typ, head, body))
            return True
        st["known"].add(head)
        st["derived"].append(head)
        st["seen"].add((typ, head, body))
        return True

    def valid_answer(self, st, ans):
        x, z = st["pair"]
        gp = st["goal_pred"]
        if gp in self.builtins:                            # value query: ans = a derived value
            return any(h[0] == gp and h[1][0] == x and h[1][1] == ans for h in st["derived"])
        if gp in {r[0][0] for r in st["extra_rules"]}:     # NOVEL relation: ans = the linked NOUN
            return (gp, (x, ans)) in st["known"]
        return ans in self.answer_preds and (ans, (x, z)) in st["known"]
