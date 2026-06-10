"""RULE INDUCTION: the agent discovers the rules from raw observations -- no hand-rolled rules.

Popper-style generate-test (Cropper & Morel, "learning from failures"): enumerate a typed
hypothesis space of Horn rules + arithmetic forms, test every candidate against OBSERVATIONS
(facts, question type, answer -- never traces, never the generator's rules), keep a minimal
consistent set by greedy cover. The world's hand-coded rules play the role of NATURE: they
generate the data and are never shown to the learner.

Hypothesis space:
  relations  A: h(X,Z) <- b1(X,Z)                       (alias)
             B: h(X,Z) <- b1(X,Y), b2(Y,Z)              (chain: grandparents, aunts, in-laws)
             C: h(X,Z) <- b1(X,W), b2(P,X), b3(P,Z)     (witness: nephew/niece)
             D: h(X,Z) <- b1(P,X), b2(P,Q), b3(Q,Z)     (two-branch: cousins)
             E: h(X,Z) <- b1(X,Y), h(Y,Z)               (recursive: ancestor)
     each body atom may be direction-flipped (spouse symmetry, child-of inverses).
  values     VAL = s2 - s1 over slots {born(x), died(x), born(z), died(z), year-in-question}
  choices    answer = argmin/argmax over {born, died} of the question pair

  python -m thinking.cli induce            # induce + report + write rules.json
"""
import itertools
import json
import logging

log = logging.getLogger("thinking")

BASE = ("mother", "father", "sister", "brother", "spouse")


# ---- observations (what nature shows the agent) ---------------------------------------------------
def gather_observations(world, n_rel, n_val, rng, depths=(2, 3), vdepths=(4, 5, 6)):
    """(edb, qpred, pair, answer) tuples. qpred is the question TYPE (the NL surface names it);
    the gold trace and the generating rules are withheld."""
    obs = []
    for _ in range(n_rel):
        k = depths[int(rng.integers(len(depths)))]
        p, _lines = world.sample(k, rng)
        obs.append((p.edb, p.goal[0], p.goal[1], p.answer))
    from .kinship import VALUE_PREDS
    for _ in range(n_val):
        k = vdepths[int(rng.integers(len(vdepths)))]
        p, _lines = world.sample_deep(k, rng, include=VALUE_PREDS + ("ancestor",))
        obs.append((p.edb, p.goal[0], p.goal[1], p.answer))
    return obs


# ---- relation hypotheses ---------------------------------------------------------------------------
def _atoms(edb):
    idx = {}
    for pred, args in edb:
        idx.setdefault(pred, set()).add(args)
    return idx


def _dirs(pred):
    return ((pred, False), (pred, True))                  # (pred, flipped?)


def _holds(idx, pred, flip, a, b):
    return (b, a) in idx.get(pred, ()) if flip else (a, b) in idx.get(pred, ())


def _ends(idx, pred, flip, a):
    """All b with pred(a,b) (or flipped)."""
    out = []
    for (h, t) in idx.get(pred, ()):
        if flip and t == a:
            out.append(h)
        elif not flip and h == a:
            out.append(t)
    return out


def rel_hypotheses():
    """Enumerate (pattern, body) hypotheses. body = tuple of (pred, flipped)."""
    H = []
    for b1 in itertools.product(BASE, (False, True)):
        H.append(("A", (b1,)))
        H.append(("E", (b1,)))
        for b2 in itertools.product(BASE, (False, True)):
            H.append(("B", (b1, b2)))
            H.append(("E2", (b1, b2)))                    # recursion over a UNION of step preds
            #                                               (minimal predicate invention: 'parent')
            for b3 in itertools.product(BASE, (False, True)):
                H.append(("C", (b1, b2, b3)))
                H.append(("D", (b1, b2, b3)))
                H.append(("F", (b1, b2, b3)))             # plain 3-chain (great-grandparents)
    return H


def rel_covers(pattern, body, idx, x, z):
    """Does this hypothesis derive (x, z) from the facts?"""
    if pattern == "A":
        (p1, f1), = body
        return _holds(idx, p1, f1, x, z)
    if pattern == "B":
        (p1, f1), (p2, f2) = body
        return any(_holds(idx, p2, f2, y, z) for y in _ends(idx, p1, f1, x))
    if pattern == "C":                                    # b1(X,W) witness; b2(P,X); b3(P,Z)
        (p1, f1), (p2, f2), (p3, f3) = body
        if not _ends(idx, p1, f1, x):
            return False
        ps = _ends(idx, p2, not f2, x)                    # P with b2(P,X): reverse lookup
        return any(_holds(idx, p3, f3, pp, z) for pp in ps)
    if pattern == "D":                                    # b1(P,X); b2(P,Q); b3(Q,Z)
        (p1, f1), (p2, f2), (p3, f3) = body
        for pp in _ends(idx, p1, not f1, x):
            for qq in _ends(idx, p2, f2, pp):
                if _holds(idx, p3, f3, qq, z):
                    return True
        return False
    if pattern == "F":                                    # 3-chain
        (p1, f1), (p2, f2), (p3, f3) = body
        for y in _ends(idx, p1, f1, x):
            for w in _ends(idx, p2, f2, y):
                if _holds(idx, p3, f3, w, z):
                    return True
        return False
    if pattern in ("E", "E2"):                            # transitive closure of the step preds
        steps = body
        seen, frontier = set(), [x]
        while frontier:
            cur = frontier.pop()
            for p1, f1 in steps:
                for nxt in _ends(idx, p1, f1, cur):
                    if nxt == z:
                        return True
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)
        return False
    return False


# ---- value / choice hypotheses ---------------------------------------------------------------------
SLOTS = ("born_x", "died_x", "born_z", "died_z", "year_q")


def _slot(idx, slot, x, z, pair):
    pred, who = slot.split("_")
    if who == "q":
        try:
            return int(pair[1])                          # the year in the question
        except (ValueError, TypeError):
            return None
    person = x if who == "x" else z
    for (h, t) in idx.get(pred, ()):
        if h == person:
            try:
                return int(t)
            except ValueError:
                return None
    return None


def val_hypotheses():
    return [(a, b) for a in SLOTS for b in SLOTS if a != b]      # VAL = b - a


def choice_hypotheses():
    return [(pred, pick) for pred in ("born", "died") for pick in ("min", "max")]


# ---- generate-test with greedy cover ----------------------------------------------------------------
def induce(obs):
    """Learn a rule set per observed question type. Returns dict + a coverage report."""
    by_pred = {}
    for edb, qpred, pair, ans in obs:
        by_pred.setdefault(qpred, []).append((_atoms(edb), pair, ans))
    learned, report = {}, {}
    rel_H = rel_hypotheses()
    for qpred, items in by_pred.items():
        numeric = all(a.lstrip("-").isdigit() for _, _, a in items)
        if numeric:                                       # arithmetic discovery
            cands = []
            for a, b in val_hypotheses():
                ok = True
                for idx, pair, ans in items:
                    v1 = _slot(idx, a, pair[0], pair[1], pair)
                    v2 = _slot(idx, b, pair[0], pair[1], pair)
                    if v1 is None or v2 is None or int(ans) != v2 - v1:
                        ok = False
                        break
                if ok:
                    cands.append(("sub", a, b))
            learned[qpred] = cands[:1]
            report[qpred] = f"{len(cands)} consistent arithmetic forms; kept {cands[:1]}"
            continue
        if all(ans in pair for _, pair, ans in items):
            cands = []                                    # choice discovery (who_older-style)
            for pred, pick in choice_hypotheses():
                ok = True
                for idx, pair, ans in items:
                    v = {p: _slot(idx, f"{pred}_x", p, p, pair) for p in pair}
                    if None in v.values():
                        ok = False
                        break
                    win = min(pair, key=v.get) if pick == "min" else max(pair, key=v.get)
                    if win != ans:
                        ok = False
                        break
                if ok:
                    cands.append(("choice", pred, pick))
            learned[qpred] = cands[:1]
            report[qpred] = f"{len(cands)} consistent choice forms; kept {cands[:1]}"
            continue
        # relation rules: greedy cover with zero false positives across OTHER preds' pairs
        pos = [(idx, pair) for idx, pair, ans in items if ans == qpred]
        neg = [(idx, pair) for p2, it2 in by_pred.items() if p2 != qpred
               for idx, pair, ans in it2
               if not all(a.lstrip("-").isdigit() for _, _, a in it2)]
        scored = []                                       # (coverage, strict?, pattern, body)
        for pattern, body in rel_H:
            cov = {i for i, (idx, (x, z)) in enumerate(pos) if rel_covers(pattern, body, idx, x, z)}
            if not cov:
                continue
            strict = not any(rel_covers(pattern, body, idx, x, z) for idx, (x, z) in neg)
            scored.append((cov, strict, pattern, body))
        # pass 1: zero-false-positive rules (learning from failures); pass 2: LENIENT completion
        # for what strictness leaves uncovered -- answers are MOST-SPECIFIC, so a general rule
        # (ancestor) legitimately covers pairs answered with a more specific relation.
        chosen, covered = [], set()
        for lenient in (False, True):
            for cov, strict, pattern, body in sorted(scored, key=lambda c: -len(c[0])):
                if (strict or lenient) and cov - covered:
                    chosen.append((pattern, body))
                    covered |= cov
                if len(covered) == len(pos):
                    break
            if len(covered) == len(pos):
                break
        learned[qpred] = [("rel", pattern, body) for pattern, body in chosen]
        report[qpred] = (f"{len(pos)} obs, covered {len(covered)} "
                         f"with {len(chosen)} rules: "
                         + "; ".join(_show(qpred, pat, b) for pat, b in chosen))
    return learned, report


def _show(h, pattern, body):
    v = {"A": "({0} X Z)", "B": "({0} X Y)({1} Y Z)", "C": "({0} X W)({1} P X)({2} P Z)",
         "D": "({0} P X)({1} P Q)({2} Q Z)", "F": "({0} X Y)({1} Y W)({2} W Z)",
         "E": "({0} X Y)(%s Y Z)" % h, "E2": "[{0}|{1}]* X Z"}[pattern]
    return f"{h}(X,Z) <- " + v.format(*[f"{'~' if f else ''}{p}" for p, f in body])


def held_out_accuracy(learned, obs):
    """Score the induced theory on UNSEEN observations (the honest metric)."""
    ok = tot = 0
    for edb, qpred, pair, ans in obs:
        rules = learned.get(qpred)
        if not rules:
            continue
        tot += 1
        idx = _atoms(edb)
        x, z = pair
        kind = rules[0][0] if rules else None
        if kind == "sub":
            _, a, b = rules[0]
            v1, v2 = _slot(idx, a, x, z, pair), _slot(idx, b, x, z, pair)
            ok += (v1 is not None and v2 is not None and str(v2 - v1) == ans)
        elif kind == "choice":
            _, pred, pick = rules[0]
            v = {p: _slot(idx, f"{pred}_x", p, p, pair) for p in pair}
            if None not in v.values():
                win = min(pair, key=v.get) if pick == "min" else max(pair, key=v.get)
                ok += (win == ans)
        else:
            ok += any(rel_covers(pat, body, idx, x, z) for _, pat, body in rules)
    return ok / max(1, tot)


def save_rules(learned, path):
    enc = {k: [list(r if r[0] != "rel" else (r[0], r[1], [list(b) for b in r[2]]))
               for r in v] for k, v in learned.items()}
    with open(path, "w") as f:
        json.dump(enc, f, indent=1)
    log.info("induced rules -> %s", path)
