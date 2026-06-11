"""StepChecker: the proof CHECKER gating the decode loop.

Strictly local validation — body membership in the known set plus rule instantiation — never a
closure call, so the engine can't leak the answer. The checker is generic over any Datalog
ruleset: nothing here knows about 'far' or chains.
"""
import sys
import os
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datalog import _unify


def _ground(atom):
    return not any(str(a).startswith("?") for a in atom[1])


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
                "goal_pred": goal_pred, "extra_rules": list(extra_rules), "seen": set()}

    def valid_step_state(self, st, typ, head, body):
        """Non-mutating version of step() for candidate generation."""
        body = tuple(body)
        if (typ, head, body) in st["seen"]:
            return False
        if typ == "check":
            return head in st["edb"]
        if typ != "think":
            return False
        if not all(f in st["known"] for f in body):
            return False
        if head[0] in self.builtins:
            if not self.builtins[head[0]](head, body, st["pair"]):
                return False
        elif not instantiates(self.rules + st["extra_rules"], head, body):
            return False
        elif (head[0] in self.answer_preds and head[0] not in self.recursive
              and tuple(head[1]) != st["pair"]):
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

    def candidate_steps(self, st, include_checks=True):
        """Generic legal local proof steps from the current checker state.

        The generator is rule-driven: it enumerates unchecked EDB facts and ground instances of
        the loaded Datalog rules whose bodies are already known. It does not know about kinship,
        ancestor, or any task-specific grammar.
        """
        candidates = []
        if include_checks:
            for fact in sorted(st["edb"] - st["known"]):
                body = ()
                if self.valid_step_state(st, "check", fact, body):
                    candidates.append(("check", fact, body))

        facts_by_pred = {}
        for fact in st["known"]:
            facts_by_pred.setdefault(fact[0], []).append(fact)
        for facts in facts_by_pred.values():
            facts.sort()

        for hp, bps in self.rules + st["extra_rules"]:
            for subst, body in _body_instances(bps, facts_by_pred):
                head = _subst(hp, subst)
                body = tuple(body)
                if not _ground(head) or head in st["known"]:
                    continue
                if self.valid_step_state(st, "think", head, body):
                    candidates.append(("think", head, body))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
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
