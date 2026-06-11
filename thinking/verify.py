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

    State: the KNOWN set (EDB facts + derived atoms). 'check F' grounds an EDB fact; 'think H
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
        return {"known": set(edb), "edb": set(edb), "derived": [], "pair": tuple(pair),
                "goal_pred": goal_pred, "extra_rules": list(extra_rules), "seen": set()}

    def step(self, st, typ, head, body):
        """Validate one line against the state; mutate ONLY when valid. Returns bool.
        Duplicate lines are rejected: gold traces never repeat, and looping on a valid check
        was eating the whole line budget (rung A-anon demo)."""
        if (typ, head, body) in st["seen"]:
            return False
        if typ == "check":
            if head in st["edb"]:
                st["seen"].add((typ, head, body))
                return True
            return False
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
            return False                                   # GOAL ANCHOR: non-recursive answerable
            #                                                conclusions must be about the ASKED
            #                                                pair (gold traces never violate this;
            #                                                recursive preds need intermediates)
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
