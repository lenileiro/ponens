"""KinshipWorld: CLUTRR-style family trees with interacting rules and branching proofs.

The question is natural-language style ("how is X related to Y ?") and the ANSWER IS A RELATION,
not an entity — the flow must derive a kinship fact linking the pair and name its predicate.
Derivations compose multiple rules (gendered base facts -> parent/sibling abstraction -> nested
kinship), so proofs are trees, not walks: this is the nested-thinking testbed.

Base EDB predicates: mother, father, sister, brother, spouse (one direction; symmetry is a rule).
"""
from dataclasses import dataclass
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datalog import Datalog

from .world import Problem

RULES = [
    (("parent", ("?x", "?y")), [("mother", ("?x", "?y"))]),
    (("parent", ("?x", "?y")), [("father", ("?x", "?y"))]),
    (("sibling", ("?x", "?y")), [("sister", ("?x", "?y"))]),
    (("sibling", ("?x", "?y")), [("brother", ("?x", "?y"))]),
    (("married", ("?x", "?y")), [("spouse", ("?x", "?y"))]),
    (("married", ("?x", "?y")), [("spouse", ("?y", "?x"))]),
    (("grandmother", ("?x", "?z")), [("mother", ("?x", "?y")), ("parent", ("?y", "?z"))]),
    (("grandfather", ("?x", "?z")), [("father", ("?x", "?y")), ("parent", ("?y", "?z"))]),
    (("grandparent", ("?x", "?z")), [("parent", ("?x", "?y")), ("parent", ("?y", "?z"))]),
    (("great_grandmother", ("?x", "?z")), [("mother", ("?x", "?y")), ("grandparent", ("?y", "?z"))]),
    (("great_grandfather", ("?x", "?z")), [("father", ("?x", "?y")), ("grandparent", ("?y", "?z"))]),
    (("aunt", ("?x", "?z")), [("sister", ("?x", "?y")), ("parent", ("?y", "?z"))]),
    (("uncle", ("?x", "?z")), [("brother", ("?x", "?y")), ("parent", ("?y", "?z"))]),
    (("cousin", ("?x", "?y")), [("parent", ("?p", "?x")), ("sibling", ("?p", "?q")),
                                ("parent", ("?q", "?y"))]),
    (("mother_in_law", ("?x", "?z")), [("mother", ("?x", "?y")), ("married", ("?y", "?z"))]),
    (("father_in_law", ("?x", "?z")), [("father", ("?x", "?y")), ("married", ("?y", "?z"))]),
    # nephew/niece: gender of the subject witnessed by their own sibling fact (no gender atoms)
    (("nephew", ("?x", "?z")), [("brother", ("?x", "?w")), ("parent", ("?p", "?x")),
                                ("sibling", ("?p", "?z"))]),
    (("niece", ("?x", "?z")), [("sister", ("?x", "?w")), ("parent", ("?p", "?x")),
                               ("sibling", ("?p", "?z"))]),
    # DEEP recursion: the relation that spans arbitrarily many generations. Right-recursive so the
    # goal-directed decomposition walks DOWN one generation per think-line (the new person first
    # appears in the parent body atom -- a context lookup).
    (("ancestor", ("?x", "?y")), [("parent", ("?x", "?y"))]),
    (("ancestor", ("?x", "?z")), [("parent", ("?x", "?y")), ("ancestor", ("?y", "?z"))]),
]
ENGINE = Datalog(RULES)

BASE_PREDS = ("mother", "father", "sister", "brother", "spouse", "born", "died")

# ---- arithmetic (value) queries: the checker VERIFIES the subtraction, never computes it -------
VALUE_PREDS = ("age_at_death", "age_when", "age_when_died", "age_in_year",
               "older_by", "who_older", "who_younger")


def _years(body, pred, person):
    for b in body:
        if b[0] == pred and b[1][0] == person:
            return int(b[1][1])
    return None


def _bi_age_at_death(head, body, pair):
    """think x age_at_death A needs x born Y1 and x died Y2 .  valid iff A == Y2 - Y1."""
    x = pair[0]
    if head[1][0] != x or len(body) != 2:
        return False
    y1, y2 = _years(body, "born", x), _years(body, "died", x)
    try:
        return y1 is not None and y2 is not None and int(head[1][1]) == y2 - y1
    except ValueError:
        return False


def _bi_age_when(head, body, pair):
    """think x age_when A needs x born Y1 and z born Y2 .  valid iff A == Y2 - Y1 (z from the
    question pair -- 'how old was x when z was born')."""
    x, z = pair
    if head[1][0] != x or len(body) != 2:
        return False
    y1, y2 = _years(body, "born", x), _years(body, "born", z)
    try:
        return y1 is not None and y2 is not None and int(head[1][1]) == y2 - y1
    except ValueError:
        return False


def _bi_age_when_died(head, body, pair):
    """think x age_when_died A needs x born Y1 and z died Y2 .  valid iff A == Y2 - Y1."""
    x, z = pair
    if head[1][0] != x or len(body) != 2:
        return False
    y1, y2 = _years(body, "born", x), _years(body, "died", z)
    try:
        return y1 is not None and y2 is not None and int(head[1][1]) == y2 - y1
    except ValueError:
        return False


def _bi_age_in_year(head, body, pair):
    """HYPOTHETICAL: 'how old would x be in YEAR ?' -- the anchor year comes from the QUESTION
    (pair[1]), not from any fact. think x age_in_year A needs x born Y1 .  A == YEAR - Y1."""
    x, year = pair
    if head[1][0] != x or len(body) != 1:
        return False
    y1 = _years(body, "born", x)
    try:
        return y1 is not None and int(head[1][1]) == int(year) - y1
    except ValueError:
        return False


def _bi_older_by(head, body, pair):
    """RELATIVITY (magnitude): 'how much older is x than z' == 'how many years younger is z than
    x' -- the same fact from either frame. valid iff VAL == born(z) - born(x)."""
    x, z = pair
    if head[1][0] != x or len(body) != 2:
        return False
    y1, y2 = _years(body, "born", x), _years(body, "born", z)
    try:
        return y1 is not None and y2 is not None and int(head[1][1]) == y2 - y1
    except ValueError:
        return False


def _bi_who(older):
    """RELATIVITY (direction): who_older / who_younger flip the answer for the same pair.
    head carries the claimed winner: think x who_older WINNER needs x born Y1 and z born Y2 ."""
    def f(head, body, pair):
        x, z = pair
        if head[1][0] != x or len(body) != 2:
            return False
        y1, y2 = _years(body, "born", x), _years(body, "born", z)
        if y1 is None or y2 is None or y1 == y2:
            return False
        winner = (x if (y1 < y2) == older else z)
        return head[1][1] == winner
    return f


AGE_BUILTINS = {"age_at_death": _bi_age_at_death, "age_when": _bi_age_when,
                "age_when_died": _bi_age_when_died, "age_in_year": _bi_age_in_year,
                "older_by": _bi_older_by,
                "who_older": _bi_who(True), "who_younger": _bi_who(False)}
# what counts as an ANSWER to "how is x related to y" (specific kinship, not the abstractions;
# 'ancestor' is the deep answer -- its derivation length is the depth axis)
ANSWER_PREDS = ("grandmother", "grandfather", "great_grandmother", "great_grandfather",
                "aunt", "uncle", "cousin", "nephew", "niece",
                "mother_in_law", "father_in_law", "ancestor")

# Templated natural-language surface (the reading layer; thinking stays canonical). Each predicate
# has SYNONYM variants, including INVERSE phrasings ({t} first) that stay gender-sound by stating
# the gender explicitly ("a child of X , who is a woman" == mother). The model must normalize all
# of them to the same canonical check-line.
TEMPLATES = {
    "mother": [
        ("{h}", "is", "the", "mother", "of", "{t}", "."),
        ("{h}", "is", "the", "mom", "of", "{t}", "."),
        ("{t}", "is", "a", "child", "of", "{h}", ",", "who", "is", "a", "woman", "."),
    ],
    "father": [
        ("{h}", "is", "the", "father", "of", "{t}", "."),
        ("{h}", "is", "the", "dad", "of", "{t}", "."),
        ("{t}", "is", "a", "child", "of", "{h}", ",", "who", "is", "a", "man", "."),
    ],
    "sister": [
        ("{h}", "is", "a", "sister", "of", "{t}", "."),
        ("{h}", "and", "{t}", "are", "siblings", "and", "{h}", "is", "a", "woman", "."),
    ],
    "brother": [
        ("{h}", "is", "a", "brother", "of", "{t}", "."),
        ("{h}", "and", "{t}", "are", "siblings", "and", "{h}", "is", "a", "man", "."),
    ],
    "spouse": [
        ("{h}", "is", "married", "to", "{t}", "."),
        ("{h}", "is", "the", "spouse", "of", "{t}", "."),
        ("{t}", "and", "{h}", "are", "a", "married", "couple", "."),
        ("{h}", "and", "{t}", "tied", "the", "knot", "years", "ago", "."),
    ],
    "born": [
        ("{h}", "was", "born", "in", "{t}", "."),
        ("{h}", "came", "into", "the", "world", "in", "{t}", "."),
        ("in", "{t}", ",", "{h}", "was", "born", "."),
        ("{h}", "arrived", "in", "the", "year", "{t}", "."),
    ],
    "died": [
        ("{h}", "died", "in", "{t}", "."),
        ("{h}", "passed", "away", "in", "{t}", "."),
        ("in", "{t}", ",", "{h}", "passed", "away", "."),
        ("{h}", "was", "laid", "to", "rest", "in", "{t}", "."),
    ],
}
# question variants: (tokens, applicable preds). None = any RELATION query; value queries (ages)
# use only their own phrasings. The ancestor variant poses the ANTONYM CONTRAST (ancestor or
# descendant) which the flow must resolve by direction.
QUESTION = [
    (("how", "is", "{h}", "related", "to", "{t}", "?"), ANSWER_PREDS),
    (("what", "is", "{h}", "to", "{t}", "?"), ANSWER_PREDS),
    (("what", "relation", "is", "{h}", "to", "{t}", "?"), ANSWER_PREDS),
    (("is", "{h}", "an", "ancestor", "or", "a", "descendant", "of", "{t}", "?"), ("ancestor",)),
    (("how", "old", "was", "{h}", "when", "they", "died", "?"), ("age_at_death",)),
    (("what", "age", "did", "{h}", "reach", "?"), ("age_at_death",)),
    (("how", "many", "years", "did", "{h}", "live", "?"), ("age_at_death",)),
    (("how", "old", "was", "{h}", "when", "{t}", "was", "born", "?"), ("age_when",)),
    (("what", "was", "the", "age", "of", "{h}", "at", "the", "birth", "of", "{t}", "?"),
     ("age_when",)),
    (("how", "old", "was", "{h}", "when", "{t}", "died", "?"), ("age_when_died",)),
    (("what", "was", "the", "age", "of", "{h}", "when", "{t}", "passed", "away", "?"),
     ("age_when_died",)),
    (("how", "old", "would", "{h}", "be", "in", "{t}", "?"), ("age_in_year",)),
    (("if", "{h}", "were", "alive", "in", "{t}", ",", "how", "old", "would", "they", "be", "?"),
     ("age_in_year",)),
    (("how", "old", "will", "{h}", "be", "in", "the", "year", "{t}", "?"), ("age_in_year",)),
    # relativity of magnitude: same fact, either frame, same number
    (("how", "much", "older", "is", "{h}", "than", "{t}", "?"), ("older_by",)),
    (("how", "many", "years", "younger", "is", "{t}", "than", "{h}", "?"), ("older_by",)),
    (("by", "how", "many", "years", "does", "{h}", "predate", "{t}", "?"), ("older_by",)),
    # relativity of direction: who_older / who_younger flip the answer for the same pair
    (("who", "is", "older", ",", "{h}", "or", "{t}", "?"), ("who_older",)),
    (("who", "was", "born", "first", ",", "{h}", "or", "{t}", "?"), ("who_older",)),
    (("who", "is", "younger", ",", "{h}", "or", "{t}", "?"), ("who_younger",)),
    (("of", "{h}", "and", "{t}", ",", "who", "was", "born", "last", "?"), ("who_younger",)),
]

def bank_levels():
    """Ordered curriculum levels available in the distilled bank ([] when no bank)."""
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "surfaces.json")
    return list(json.load(open(path))["templates"].keys()) if os.path.exists(path) else []


def definitions(level="mix", split="train"):
    """{word: [definition token-lists]} from the bank ({} when absent)."""
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "surfaces.json")
    if not os.path.exists(path):
        return {}
    b = json.load(open(path))
    sec = b.get("definitions" if split == "train" else "eval_definitions", {})
    levels = list(sec) if level == "mix" else [level]
    out = {}
    for lv in levels:
        for w, vs in sec.get(lv, {}).items():
            out.setdefault(w, []).extend(tuple(v) for v in vs)
    return out


def surfaces(level="mix", split="train"):
    """(templates, question) from the DISTILLED bank (surfaces.json) when present, else the
    built-ins. level: kindergarten | midschool | mix (the language curriculum). split: train |
    eval -- eval phrasings are HELD OUT, so reading them is the language-understanding test."""
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "surfaces.json")
    if level == "canonical":                               # staircase rung A0: facts share the
        tmpl = {p: [("{h}", p, "{t}", ".")] for p in BASE_PREDS}   # trace's surface exactly
        qs = [(("query", "rel", "{h}", "{t}", "?"), ANSWER_PREDS)]
        qs += [(("query", vp, "{h}", "{t}", "?"), (vp,)) for vp in VALUE_PREDS]
        return tmpl, qs
    if level == "builtin" or not os.path.exists(path):     # staircase rung A: no bank
        return TEMPLATES, QUESTION
    b = json.load(open(path))
    tsec = b["templates" if split == "train" else "eval_templates"]
    qsec = b["questions" if split == "train" else "eval_questions"]
    levels = list(tsec) if level == "mix" else [level]
    tmpl, qs = {}, []
    for lv in levels:
        for pred, vs in tsec.get(lv, {}).items():
            tmpl.setdefault(pred, []).extend(tuple(v) for v in vs)
        for qk, vs in qsec.get(lv, {}).items():
            preds = ANSWER_PREDS if qk == "rel" else (qk,)
            qs.extend((tuple(v), preds) for v in vs)
    for pred, vs in TEMPLATES.items():                     # safety: bank gaps fall back
        if not tmpl.get(pred):
            tmpl[pred] = list(vs)
    covered = set()                                        # question-side gaps fall back too
    for _, ps in qs:
        covered |= set(ps) if isinstance(ps, tuple) else {ps}
    for v, ps in QUESTION:
        pset = set(ps) if isinstance(ps, tuple) else {ps}
        if pset - covered:
            qs.append((v, ps))
    return tmpl, (qs or QUESTION)


NAMES = """ada alan alba aldo alma amos anna aris asha axel bart beth bill bora carl cleo cora dale
dana dara dave dina dora earl edna ella elsa emil enzo erik esme etta evan ezra faye fern finn
gail gene gina glen greg hana hank hugo iggy ines iris ivan jack jade jane jeff joan joel john
jude july kane kara kate kira kobe kyle lara leah lena leon lila lisa lola luca luke lyle mara
meg mia milo mina nash nell nico nina noel nora olga omar opal oren otto pam pearl penn piper
quin rene rhea rita rosa ross ruby rudy ruth ryan sage sara seth tess thea tina toby todd tona
ula uma una vada vera vick wade walt wren yara york yuri zane zara zeke zora""".split()


def name_pools(n_train, n_test, seed):
    """Real names first; CVCV pseudo-names as overflow (50-generation trees need hundreds)."""
    rng = np.random.default_rng(seed)
    names = sorted(set(NAMES))
    need = n_train + n_test
    if need > len(names):
        c, v = "bdfgklmnprstv", "aeiou"
        extra = set()
        while len(names) + len(extra) < need:
            w = (c[rng.integers(13)] + v[rng.integers(5)] + c[rng.integers(13)] + v[rng.integers(5)])
            if w not in names:
                extra.add(w)
        names = names + sorted(extra)
    rng.shuffle(names)
    return names[:n_train], names[n_train:n_train + n_test]


@dataclass
class Person:
    name: str
    female: bool


class FamilyWorld:
    """Random clean family trees (no remarriage, spouses are fresh people) -> EDB facts; queries
    sampled from the deductive closure at a target derivation depth, restricted to pairs whose
    answerable kinship is UNIQUE (so the oracle answer is well-defined)."""

    def __init__(self, names, seed=0):
        self.names = list(names)
        self.rng = np.random.default_rng(seed)

    def _tree(self, rng):
        names = [str(n) for n in rng.permutation(self.names)]
        take = lambda female: Person(names.pop(), female)
        edb = []
        gen = [(take(False), take(True))]                  # founder couple
        for _ in range(int(rng.integers(2, 4))):           # generations
            nxt = []
            for dad, mom in gen:
                edb.append(("spouse", (dad.name, mom.name)))
                kids = [take(bool(rng.integers(2))) for _ in range(int(rng.integers(2, 4)))]
                if len(names) < 12:
                    break
                for k in kids:
                    edb.append(("mother", (mom.name, k.name)))
                    edb.append(("father", (dad.name, k.name)))
                for a, b in zip(kids, kids[1:]):           # adjacent pairs only (quadratic bloat
                    edb.append(("sister" if a.female else "brother", (a.name, b.name)))
                    edb.append(("sister" if b.female else "brother", (b.name, a.name)))
                for k in kids:
                    if rng.random() < 0.7 and len(names) > 4:   # marry in a fresh spouse
                        sp = take(not k.female)
                        couple = (k, sp) if not k.female else (sp, k)
                        nxt.append(couple)
            gen = nxt
            if not gen:
                break
        return edb

    def sample(self, k, rng=None, include=None, exclude=None):
        """A Problem + its gold goal-directed proof lines. k <= 3: named-relation queries sampled
        from the engine's closure (CLUTRR regime). k >= 4: DEEP regime -- a k-generation ancestor
        spine with distractor branches, proof constructed generatively (closure over a 50-level
        tree is infeasible; every line is still GoalChecker-validated, see selftest)."""
        rng = rng or self.rng
        if k >= 4:
            return self.sample_deep(k, rng, include, exclude)
        return self._sample_named(k, rng, include, exclude)

    DEEP_QTYPES = ("ancestor", "grandmother", "grandfather", "great_grandmother",
                   "great_grandfather", "aunt", "uncle", "cousin", "nephew", "niece",
                   "mother_in_law", "father_in_law", "age_at_death", "age_when",
                   "age_when_died", "age_in_year", "older_by", "who_older", "who_younger")

    def sample_deep(self, depth, rng=None, include=None, exclude=None):
        """A `depth`-generation spine with the FULL relationship web embedded (spouses, sibling
        branches with their own children) -- everything not on the gold derivation is a
        distractor. The query is either the depth-long 'ancestor' chain or a named relation
        (grand*/aunt/uncle/cousin/nephew/niece/in-law) nested INSIDE the deep tree. Gold proof
        lines are constructed generatively (closure at this scale is infeasible); the selftest
        validates every query type against the GoalChecker AND the engine on small instances."""
        rng = rng or self.rng
        qs = [q for q in self.DEEP_QTYPES
              if (include is None or q in include) and (exclude is None or q not in exclude)]
        assert qs, "no deep query types left after include/exclude"
        q = ("ancestor" if "ancestor" in qs and rng.random() < 0.4
             else qs[int(rng.integers(len(qs)))])
        names = [str(n) for n in rng.permutation(self.names)]
        assert len(names) >= 2 * depth + 10, "name pool too small for this depth"
        take = lambda: names.pop()

        spine = [take() for _ in range(depth + 1)]
        genders = [bool(rng.integers(2)) for _ in spine]   # True = female
        site = int(rng.integers(0, max(1, depth - 3)))     # anchor for named queries
        # force what the chosen query needs (genders / enrichments at the site)
        if q in ("grandmother", "great_grandmother", "mother_in_law"):
            genders[site] = True
        if q in ("grandfather", "great_grandfather", "father_in_law"):
            genders[site] = False
        spouses, branches = {}, {}                         # i -> name / i -> (sib, sib_female, child|None)
        if q in ("mother_in_law", "father_in_law"):
            spouses[site + 1] = take()
        if q in ("aunt", "uncle", "cousin", "nephew", "niece"):
            branches[site] = (take(), q in ("aunt", "niece") or
                              (q in ("cousin", "nephew") and bool(rng.integers(2))),
                              take() if q == "cousin" else None)
        if q in ("nephew", "niece"):
            genders[site + 1] = (q == "niece")
            branches[site + 1] = (take(), bool(rng.integers(2)), None)   # gender witness sibling
        # random extra enrichment (distractors) wherever names remain
        for i in range(depth):
            if i not in spouses and len(names) > 4 and rng.random() < 0.15:
                spouses[i + 1] = take()
            if i not in branches and len(names) > 4 and rng.random() < 0.15:
                branches[i] = (take(), bool(rng.integers(2)), take() if rng.random() < 0.4 else None)

        edb = []
        P = lambda i: "mother" if genders[i] else "father"
        S = lambda fem: "sister" if fem else "brother"
        for i in range(depth):
            edb.append((P(i), (spine[i], spine[i + 1])))
        for i, sp in spouses.items():
            edb.append(("spouse", (spine[i], sp)))
        for i, (sib, fem, child) in branches.items():
            edb.append((S(fem), (sib, spine[i])))          # both directions: each subject's gender
            edb.append((S(genders[i]), (spine[i], sib)))
            if child is not None:
                edb.append(("mother" if fem else "father", (sib, child)))
        # CONSISTENT CHRONOLOGY: birth years flow down the generations; everyone gets born+died
        # facts (a historical tree) -- temporal facts are distractors for relation queries and
        # the substance of age queries.
        byear = {spine[0]: 1500 + int(rng.integers(300))}
        for i in range(depth):
            byear[spine[i + 1]] = byear[spine[i]] + 18 + int(rng.integers(23))
        for i, sp in spouses.items():
            byear[sp] = byear[spine[i]] + int(rng.integers(-6, 7))
        for i, (sib, fem, child) in branches.items():
            byear[sib] = byear[spine[i]] + int(rng.integers(-8, 9))
            if child is not None:
                byear[child] = byear[sib] + 18 + int(rng.integers(23))
        dyear = {p: y + 45 + int(rng.integers(50)) for p, y in byear.items()}
        for p in byear:
            edb.append(("born", (p, str(byear[p]))))
            edb.append(("died", (p, str(dyear[p]))))
        rng.shuffle(edb)

        def dparent(i):                                    # decompose parent(spine[i], spine[i+1])
            par = ("parent", (spine[i], spine[i + 1]))
            f = (P(i), (spine[i], spine[i + 1]))
            return [("check", f, ()), ("think", par, (f,))], par

        lines = []
        if q == "ancestor":
            x, z = spine[0], spine[-1]
            for i in range(depth - 1, -1, -1):           # EVIDENCE-FIRST: derive upward
                pl, par = dparent(i)
                lines += pl
                anc = ("ancestor", (spine[i], z))
                if i < depth - 1:
                    lines.append(("think", anc, (par, ("ancestor", (spine[i + 1], z)))))
                else:
                    lines.append(("think", anc, (par,)))
        elif q in ("grandmother", "grandfather"):
            x, z = spine[site], spine[site + 2]
            f = (P(site), (spine[site], spine[site + 1]))
            pl, par = dparent(site + 1)
            lines = [("check", f, ())] + pl + [("think", (q, (x, z)), (f, par))]
        elif q in ("great_grandmother", "great_grandfather"):
            x, z = spine[site], spine[site + 3]
            f = (P(site), (spine[site], spine[site + 1]))
            gp = ("grandparent", (spine[site + 1], z))
            pl1, par1 = dparent(site + 1)
            pl2, par2 = dparent(site + 2)
            lines = ([("check", f, ())] + pl1 + pl2 +
                     [("think", gp, (par1, par2)), ("think", (q, (x, z)), (f, gp))])
        elif q in ("aunt", "uncle"):
            sib, fem, _ = branches[site]
            x, z = sib, spine[site + 1]
            f = (S(fem), (sib, spine[site]))
            pl, par = dparent(site)
            lines = [("check", f, ())] + pl + [("think", (q, (x, z)), (f, par))]
        elif q == "cousin":
            sib, fem, child = branches[site]
            x, z = child, spine[site + 1]
            fp = ("mother" if fem else "father", (sib, child))
            pc = ("parent", (sib, child))
            fs = (S(fem), (sib, spine[site]))
            sg = ("sibling", (sib, spine[site]))
            pl, par = dparent(site)
            lines = ([("check", fp, ()), ("think", pc, (fp,)),
                      ("check", fs, ()), ("think", sg, (fs,))] + pl +
                     [("think", ("cousin", (x, z)), (pc, sg, par))])
        elif q in ("nephew", "niece"):
            sib, _, _ = branches[site]                     # the aunt/uncle side
            wit, _, _ = branches[site + 1]                 # gender witness for spine[site+1]
            x, z = spine[site + 1], sib
            fw = (S(genders[site + 1]), (spine[site + 1], wit))
            fs = (S(genders[site]), (spine[site], sib))
            sg = ("sibling", (spine[site], sib))
            pl, par = dparent(site)
            lines = ([("check", fw, ())] + pl +
                     [("check", fs, ()), ("think", sg, (fs,)),
                      ("think", (q, (x, z)), (fw, par, sg))])
        elif q == "age_at_death":                          # value query: checker verifies Y2-Y1
            x = z = spine[int(rng.integers(len(spine)))]
            a = dyear[x] - byear[x]
            fb, fd = ("born", (x, str(byear[x]))), ("died", (x, str(dyear[x])))
            lines = [("check", fb, ()), ("check", fd, ()),
                     ("think", ("age_at_death", (x, str(a))), (fb, fd))]
            p = Problem(head=x, edb=tuple(edb), goal=("age_at_death", (x, x)),
                        answer=str(a), k=depth)
            return p, lines
        elif q == "age_when":                              # how old was x when z was born
            i = int(rng.integers(0, depth))
            j = int(rng.integers(i + 1, depth + 1))
            x, z = spine[i], spine[j]
            a = byear[z] - byear[x]
            fx, fz = ("born", (x, str(byear[x]))), ("born", (z, str(byear[z])))
            lines = [("check", fx, ()), ("check", fz, ()),
                     ("think", ("age_when", (x, str(a))), (fx, fz))]
            p = Problem(head=x, edb=tuple(edb), goal=("age_when", (x, z)),
                        answer=str(a), k=depth)
            return p, lines
        elif q == "age_when_died":                         # how old was x when z died
            i = int(rng.integers(0, depth))
            j = int(rng.integers(i, depth + 1))            # z's death is always after x's birth
            x, z = spine[i], spine[j]
            if x == z:                                     # that would be age_at_death
                z = spine[min(j + 1, depth)] if j < depth else spine[max(0, i - 1)]
            a = dyear[z] - byear[x]
            fx, fz = ("born", (x, str(byear[x]))), ("died", (z, str(dyear[z])))
            lines = [("check", fx, ()), ("check", fz, ()),
                     ("think", ("age_when_died", (x, str(a))), (fx, fz))]
            p = Problem(head=x, edb=tuple(edb), goal=("age_when_died", (x, z)),
                        answer=str(a), k=depth)
            return p, lines
        elif q == "age_in_year":                           # HYPOTHETICAL anchor year from the question
            x = spine[int(rng.integers(len(spine)))]
            year = byear[x] + 1 + int(rng.integers(400))   # may be far beyond death ('would be')
            a = year - byear[x]
            fx = ("born", (x, str(byear[x])))
            lines = [("check", fx, ()), ("think", ("age_in_year", (x, str(a))), (fx,))]
            p = Problem(head=x, edb=tuple(edb), goal=("age_in_year", (x, str(year))),
                        answer=str(a), k=depth)
            return p, lines
        elif q == "older_by":                              # relative magnitude (either frame)
            i = int(rng.integers(0, depth))
            j = int(rng.integers(i + 1, depth + 1))
            x, z = spine[i], spine[j]                      # x strictly older -> positive answer
            a = byear[z] - byear[x]
            fx, fz = ("born", (x, str(byear[x]))), ("born", (z, str(byear[z])))
            lines = [("check", fx, ()), ("check", fz, ()),
                     ("think", ("older_by", (x, str(a))), (fx, fz))]
            p = Problem(head=x, edb=tuple(edb), goal=("older_by", (x, z)),
                        answer=str(a), k=depth)
            return p, lines
        elif q in ("who_older", "who_younger"):            # relative direction (answer = a NAME)
            i = int(rng.integers(0, depth))
            j = int(rng.integers(i + 1, depth + 1))
            pr = [spine[i], spine[j]]
            rng.shuffle(pr)                                # either order in the question
            x, z = pr
            winner = (spine[i] if q == "who_older" else spine[j])
            fx, fz = ("born", (x, str(byear[x]))), ("born", (z, str(byear[z])))
            lines = [("check", fx, ()), ("check", fz, ()),
                     ("think", (q, (x, winner)), (fx, fz))]
            p = Problem(head=x, edb=tuple(edb), goal=(q, (x, z)), answer=winner, k=depth)
            return p, lines
        else:                                              # mother_in_law / father_in_law
            sp = spouses[site + 1]
            x, z = spine[site], sp
            f = (P(site), (spine[site], spine[site + 1]))
            fsp = ("spouse", (spine[site + 1], sp))
            mar = ("married", (spine[site + 1], sp))
            lines = [("check", f, ()), ("check", fsp, ()), ("think", mar, (fsp,)),
                     ("think", (q, (x, z)), (f, mar))]
        p = Problem(head=x, edb=tuple(edb), goal=(q, (x, z)), answer=q, k=depth)
        return p, lines

    def _sample_named(self, k, rng, include=None, exclude=None):
        for _ in range(60):                                # rejection-sample worlds
            edb = self._tree(rng)
            closure, prov = ENGINE.closure(set(edb))
            byp = {}
            for f in closure:
                if f[0] in ANSWER_PREDS and f[0] != "ancestor":   # deep regime owns 'ancestor'
                    byp.setdefault(f[1], []).append(f)
            # the queried LINK must be computable ONLY through rules: no EDB fact may directly
            # connect the pair (in either direction) -- reading alone can never answer it
            direct = {frozenset(f[1]) for f in edb}
            cands = [fs[0] for pair, fs in byp.items()
                     if len(fs) == 1 and prov[fs[0]][2] == k
                     and frozenset(pair) not in direct
                     and (include is None or fs[0][0] in include)
                     and (exclude is None or fs[0][0] not in exclude)]
            if cands:
                goal = cands[int(rng.integers(len(cands)))]
                p = Problem(head=goal[1][0], edb=tuple(edb), goal=goal, answer=goal[0], k=k)
                return p, self.proof_lines(p, prov)
        raise RuntimeError(f"could not sample a depth-{k} kinship problem")

    # ---- NOVEL relations: defined IN THE QUESTION, never in the dataset ------------------------
    # The skill (read a definition -> use it as a rule -> link nouns to the answer) is trained
    # with random compositions bound to a small set of relation-SLOT tokens; eval uses HELD-OUT
    # compositions, so the relationship itself is never in the training data.
    NOVEL_SLOTS = tuple(f"rel{c}" for c in "abcdefghijklmnopqrst")
    NOVEL_STEPS = ("mother", "father", "parent", "brother", "sister", "spouse")
    NOVEL_HOLDOUT = (("mother", "brother"), ("father", "sister"), ("spouse", "mother"),
                     ("brother", "spouse"), ("parent", "spouse"), ("sister", "parent"))

    def sample_novel(self, rng=None, train=True):
        """A relationship that exists ONLY in the question: 'a relX of someone is the P1 of a P2
        of that person . who is the relX of H ?' -> answer is the linked PERSON. Gold trace
        derives the chain bottom-up; the checker validates against the in-question rule."""
        rng = rng or self.rng
        for _ in range(80):
            p1 = self.NOVEL_STEPS[int(rng.integers(len(self.NOVEL_STEPS)))]
            p2 = self.NOVEL_STEPS[int(rng.integers(len(self.NOVEL_STEPS)))]
            if (train and (p1, p2) in self.NOVEL_HOLDOUT) or \
               (not train and (p1, p2) not in self.NOVEL_HOLDOUT):
                continue
            base, _ = self.sample_deep(4 + int(rng.integers(3)), rng, include=("ancestor",))
            edb = base.edb
            idx = {}
            for pred, (h, t) in edb:
                idx.setdefault(pred, []).append((h, t))
            idx["parent"] = idx.get("mother", []) + idx.get("father", [])

            def steps_for(pred):                           # concrete fact + optional unfold
                for (h, t) in idx.get(pred, []):
                    yield h, t

            found = None
            for y, z in steps_for(p2):                     # x --p1--> y --p2--> z
                for x, y2 in steps_for(p1):
                    if y2 == y and x != z and not any(set(f[1]) == {x, z} for f in edb):
                        found = (x, y, z)
                        break
                if found:
                    break
            if not found:
                continue
            x, y, z = found
            name = self.NOVEL_SLOTS[int(rng.integers(len(self.NOVEL_SLOTS)))]
            rule = ((name, ("?x", "?z")),
                    [(p1, ("?x", "?y")), (p2, ("?y", "?z"))])
            lines = []

            def ground(pred, a, b):                        # check the fact; unfold 'parent'
                if pred == "parent":
                    real = next(pr for pr in ("mother", "father") if (a, b) in idx.get(pr, []))
                    lines.append(("check", (real, (a, b)), ()))
                    lines.append(("think", ("parent", (a, b)), ((real, (a, b)),)))
                    return ("parent", (a, b))
                lines.append(("check", (pred, (a, b)), ()))
                return (pred, (a, b))
            f1 = ground(p1, x, y)
            f2 = ground(p2, y, z)
            lines.append(("think", (name, (x, z)), (f1, f2)))
            qdef = ("a", name, "of", "someone", "is", "the", p1, "of", "a", p2, "of",
                    "that", "person", ".", "who", "is", "the", name, "of", "{h}", "?")
            p = Problem(head=x, edb=edb, goal=(name, (x, z)), answer=z, k=2,
                        extra_rules=(rule,), question=((qdef, (name,)),))
            return p, lines
        raise RuntimeError("could not sample a novel-relation problem")

    def proof_lines(self, problem, prov):
        """EVIDENCE-FIRST trace: POSTORDER of the proof tree -- ground the facts, derive upward,
        conclude last. (Preorder/goal-directed put the ANSWER at token 3 of line 1 = answer-only
        supervision for the hardest decision; v7/v8 showed it unlearnable at our scale.)
        [('check', fact)] leaves before each [('think', head, body)] derivation."""
        tree = ENGINE.proof_tree(prov, problem.goal)
        lines = []

        def rec(node):
            if node["rule"] is None:
                lines.append(("check", node["fact"], ()))
            else:
                for c in node["from"]:
                    rec(c)
                lines.append(("think", node["fact"], tuple(c["fact"] for c in node["from"])))
        rec(tree)
        return lines
