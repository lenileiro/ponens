#!/usr/bin/env python3
"""NEGATION FIX (fork of reason_lang2): make SUPPORTED NEGATION (EXCLUDE) learnable via a STRUCTURED,
GROUNDED negation-as-failure proof that is PARALLEL in form to the positive INHERIT proof, so the
model is not biased toward "yes".

THE BUG (in reason_lang2): for a "no" answer (EXCLUDE), the proof was just the plain isa-walk with no
"note" leaf and no terminal "so" verdict. Positives emit a distinctive `note <holder> can <prop> .` +
`so <subj> can <prop> . answer yes .`; negatives emitted neither marker. The decoder's `verdict` could
only ever flip to "yes" (on an `in_so` line that matched the query); "no" was just the fallback. The
model had no structural way to represent "not derivable" and defaulted to the positive pattern -> EXCLUDE
train acc ~0.23 (below chance).

THE FIX (negation-as-failure, grounded + parallel):
  For a supported "no", we render:
     <full isa walk the subject DOES have>           (same `_isa_sent` form as the positive bridge)
     note no <subject> can <property> .              (the FAILURE marker: no rule gives it the property)
     so <subject> cannot <property> .                (terminal verdict -> answer no)
     answer no .
  The `note no ...` absence is VERIFIED against the closure: has_prop(subject, property) is NOT derivable.
  The decoder recognizes a grounded `so <subj> cannot <prop> .` line (checked: query NOT in closure) ->
  verdict "no", mirroring the positive `so <subj> can <prop> .` (checked: query IN closure) -> verdict
  "yes". yes/no stay balanced; both verdicts now have a learnable, engine-checkable terminal step.

ORIGINAL DOC (reason_lang2) below:
BROADEN natural-language verified reasoning toward RICHER, LESS-TEMPLATED language: a model that
READS NL facts and answers NL queries by composing them, trained with engine-derived PROOF supervision
rendered as a natural-language reasoning chain (CoT). The real test: does it reason from MEANING, not
from surface templates?  We test generalization on TWO axes:
  (a) CONFIG-holdout  -- novel fresh-random entity configurations (as reason_lang.py did), AND
  (b) TEMPLATE-holdout -- UNSEEN SURFACE PHRASINGS of the same facts/queries. We train on a SUBSET of
      surface templates and test on phrasings the model NEVER saw. (b) is the meaning-vs-template test.

Self-contained; reuses ONLY the shared engine (datalog.Datalog) + ScratchpadLM. Does not modify
reason_lang.py. Keeps the reason_lang fix that beats AR exposure-bias drift: for inherited properties,
emit the recalled property LEAF FIRST (a distinct "note" token) before the isa bridge.

BROADENING over reason_lang.py:
  1. MORE RELATIONS / RULES (datalog Horn rules = ground-truth oracle):
       R0 isa transitivity:        isa(X,Z)       :- isa(X,Y), isa(Y,Z)
       R1 part_of transitivity:    part_of(X,Z)   :- part_of(X,Y), part_of(Y,Z)
       R2 located_in transitivity: located_in(X,Z):- located_in(X,Y), located_in(Y,Z)
       R3 has_prop INHERITANCE:    has_prop(X,P)  :- isa(X,Y), has_prop(Y,P)
     Plus a NEGATION/EXCLUSION query notion: an answer "no" is SUPPORTED -- a queried property that the
     subject does NOT inherit (engine confirms it is NOT in the closure). yes/no balanced by the engine.
  2. BIGGER VOCAB: ~70 nouns/categories, ~40 properties, ~30 parts, ~30 places -- fresh-random per
     example so nothing is memorized.
  3. PARAPHRASE VARIETY: every fact AND query type has MULTIPLE surface templates (e.g. "a robin is a
     bird ." / "robins are birds ." / "every robin is a bird ."). The engine parse handles all of them
     (we own the templates -> EXACTLY invertible). Held-out-TEMPLATE split: template ids are partitioned
     train vs test; we ASSERT held-out templates never appear in any train example.

  python -m thinking.reason_lang2 --sanity   # PROOF train-acc gate (>=3200 steps)
  python -m thinking.reason_lang2            # >=3 seeds, ANSWER-ONLY vs PROOF, config- & template-holdout
"""
import argparse
import os
import sys
import random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datalog import Datalog
from scratchpad_model import ScratchpadLM

# ---------------------------------------------------------------------------------------------------
# ENGINE: four transitive/inheritance Horn rules. The engine is the oracle for EVERY label (yes & no).
# ---------------------------------------------------------------------------------------------------
RULES = [
    (("isa", ("?x", "?z")),        [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))]),          # R0
    (("part_of", ("?x", "?z")),    [("part_of", ("?x", "?y")), ("part_of", ("?y", "?z"))]),  # R1
    (("located_in", ("?x", "?z")), [("located_in", ("?x", "?y")), ("located_in", ("?y", "?z"))]),  # R2
    (("has_prop", ("?x", "?p")),   [("isa", ("?x", "?y")), ("has_prop", ("?y", "?p"))]),      # R3 inherit
]
DL = Datalog(RULES)

# --- BIGGER vocabulary pools (fresh-random per example -> no memorization) --------------------------
NOUNS = ["robin", "sparrow", "eagle", "salmon", "trout", "shark", "tiger", "lion", "wolf", "fox",
         "bird", "fish", "mammal", "animal", "reptile", "insect", "spider", "snake", "lizard",
         "creature", "beast", "organism", "vertebrate", "predator", "rodent", "feline", "canine",
         "primate", "ape", "monkey", "whale", "dolphin", "otter", "seal", "bat", "owl", "hawk",
         "frog", "toad", "newt", "crab", "lobster", "beetle", "moth", "wasp", "bee", "ant",
         "heron", "crane", "swan", "goose", "duck", "deer", "elk", "moose", "goat", "sheep",
         "horse", "zebra", "camel", "llama", "rabbit", "hare", "mouse", "rat", "vole", "shrew",
         "mole", "badger", "weasel", "stoat"]                                       # 70
PROPS = ["breathe", "move", "grow", "eat", "sleep", "swim", "fly", "hunt", "rest", "feed",
         "wander", "roam", "hide", "climb", "dig", "sing", "crawl", "leap", "dive", "glide",
         "burrow", "forage", "graze", "perch", "nest", "molt", "shed", "bask", "float", "drift",
         "soar", "pounce", "stalk", "scurry", "scamper", "wade", "paddle", "flap", "hover", "sprint"]  # 40
PARTS = ["wing", "tail", "beak", "claw", "fin", "scale", "feather", "fur", "horn", "antler",
         "hoof", "paw", "snout", "whisker", "gill", "tooth", "tongue", "spine", "shell", "tusk",
         "trunk", "mane", "hump", "pouch", "talon", "fang", "crest", "barb", "frill", "ridge"]   # 30
PLACES = ["forest", "river", "valley", "mountain", "desert", "meadow", "swamp", "tundra", "reef",
          "cave", "canyon", "delta", "lagoon", "savanna", "jungle", "marsh", "prairie", "glacier",
          "estuary", "woodland", "grassland", "wetland", "highland", "lowland", "shoreline",
          "den", "thicket", "hollow", "ravine", "gully"]                            # 30

STRUCT = ["<pad>", "a", "an", "is", "can", "cannot", "has", "in", "are", "every", "all", "lives",
          "live", "have", "inside", "located", "true", "it", "that", "of", "part", "the",
          "so", "note", "answer", "yes", "no", "query", ".", "?"]

# query types: transitive-relation queries + property inherit (positive) + property exclusion (negative)
QTYPES = ("ISA", "PARTOF", "LOCIN", "INHERIT", "EXCLUDE")


def article(word):
    return "an" if word and word[0] in "aeiou" else "a"


def plural(word):
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def build_vocab():
    base = list(STRUCT) + list(NOUNS) + list(PROPS) + list(PARTS) + list(PLACES)
    assert len(base) == len(set(base)), "duplicate vocab token: " + str(
        [t for t in base if base.count(t) > 1][:5])
    # plural surfaces appear in templates ("robins are birds .") -> add a plural token for every
    # pluralizable label (NOUNS/PARTS/PLACES). de-dup against base (e.g. a plural may equal a label).
    plurals = []
    for w in NOUNS + PARTS + PLACES:
        pw = plural(w)
        if pw not in base and pw not in plurals:
            plurals.append(pw)
    itos = base + plurals
    assert len(itos) == len(set(itos)), "duplicate vocab token (with plurals)"
    return itos, {t: i for i, t in enumerate(itos)}


# ---------------------------------------------------------------------------------------------------
# PARAPHRASE TEMPLATES.  Each (relation, kind) -> list of templates, each a callable (x, y)->token list.
# kind: "fact" for asserting a fact, "query" for the question.  We OWN every template so parsing the
# rendered surface back to the relation is EXACT (see parse_sentence / parse_query).
# Template ids are partitioned train vs test (TEMPLATE-holdout). Multiple phrasings of the SAME fact.
# ---------------------------------------------------------------------------------------------------
def _isa_fact_templates(x, y):
    return [
        [article(x), x, "is", article(y), y, "."],            # "a robin is a bird ."
        [plural(x), "are", plural(y), "."],                   # "robins are birds ."
        ["every", x, "is", article(y), y, "."],               # "every robin is a bird ."
        ["all", plural(x), "are", plural(y), "."],            # "all robins are birds ."
    ]


def _part_fact_templates(x, y):
    # x is a part of y  (e.g. "a wing is part of a bird .")
    return [
        [article(x), x, "is", "part", "of", article(y), y, "."],   # "a wing is part of a bird ."
        [plural(x), "are", "part", "of", plural(y), "."],          # "wings are part of birds ."
        [article(y), y, "has", article(x), x, "."],                # "a bird has a wing ."
        [plural(y), "have", plural(x), "."],                       # "birds have wings ."
    ]


def _locin_fact_templates(x, y):
    # x is located in y (e.g. "a robin lives in a forest .")
    return [
        [article(x), x, "lives", "in", article(y), y, "."],        # "a robin lives in a forest ."
        [plural(x), "live", "in", plural(y), "."],                 # "robins live in forests ."
        [article(x), x, "is", "located", "in", article(y), y, "."],  # "a robin is located in a forest ."
        [article(x), x, "is", "inside", article(y), y, "."],       # "a robin is inside a forest ."
    ]


def _prop_fact_templates(x, p):
    return [
        [article(x), x, "can", p, "."],                            # "a bird can fly ."
        [plural(x), "can", p, "."],                                # "birds can fly ."
        ["every", x, "can", p, "."],                               # "every bird can fly ."
    ]


def _isa_query_templates(x, y):
    return [
        ["query", "is", article(x), x, article(y), y, "?"],            # "is a robin a bird ?"
        ["query", "are", plural(x), plural(y), "?"],                   # "are robins birds ?"
        ["query", "is", "it", "true", "that", article(x), x, "is", article(y), y, "?"],
    ]


def _part_query_templates(x, y):
    return [
        ["query", "is", article(x), x, "part", "of", article(y), y, "?"],   # "is a wing part of a bird?"
        ["query", "does", article(y), y, "have", article(x), x, "?"] if False else
        ["query", "is", article(x), x, "inside", article(y), y, "?"],       # alt phrasing (no new word)
        ["query", "is", "it", "true", "that", article(x), x, "is", "part", "of", article(y), y, "?"],
    ]


def _locin_query_templates(x, y):
    return [
        ["query", "is", article(x), x, "in", article(y), y, "?"],              # "is a robin in a forest?"
        ["query", "is", article(x), x, "located", "in", article(y), y, "?"],
        ["query", "is", "it", "true", "that", article(x), x, "lives", "in", article(y), y, "?"],
    ]


def _prop_query_templates(x, p):
    return [
        ["query", "can", article(x), x, p, "?"],                               # "can a robin fly ?"
        ["query", "is", "it", "true", "that", article(x), x, "can", p, "?"],
        ["query", "can", plural(x), p, "?"],                                   # "can robins fly ?"
    ]


# registry: relation -> (fact_templates_fn, query_templates_fn)
FACT_TPL = {"isa": _isa_fact_templates, "part_of": _part_fact_templates,
            "located_in": _locin_fact_templates, "has_prop": _prop_fact_templates}
QUERY_TPL = {"isa": _isa_query_templates, "part_of": _part_query_templates,
             "located_in": _locin_query_templates, "has_prop": _prop_query_templates}


def n_fact_tpls(rel):
    return len(FACT_TPL[rel]("robin", "bird"))


def n_query_tpls(rel):
    return len(QUERY_TPL[rel]("robin", "bird"))


# ---------------------------------------------------------------------------------------------------
# TEMPLATE-HOLDOUT split: for each relation, reserve the LAST fact-template-id and LAST query-template-id
# for the test set. Train uses only the remaining ids. We assert no train example uses a held-out id.
# ---------------------------------------------------------------------------------------------------
def heldout_fact_ids(rel):
    return {n_fact_tpls(rel) - 1}


def heldout_query_ids(rel):
    return {n_query_tpls(rel) - 1}


def allowed_fact_ids(rel, template_holdout):
    ids = list(range(n_fact_tpls(rel)))
    if template_holdout:
        ids = [i for i in ids if i not in heldout_fact_ids(rel)]
    return ids


def allowed_query_ids(rel, template_holdout):
    ids = list(range(n_query_tpls(rel)))
    if template_holdout:
        ids = [i for i in ids if i not in heldout_query_ids(rel)]
    return ids


def render_fact(rng, atom, fact_ids):
    """Render a fact using a template id allowed by `fact_ids` (dict rel->list of ids)."""
    pred, (a, b) = atom
    tpls = FACT_TPL[pred](a, b)
    tid = rng.choice(fact_ids[pred])
    return tpls[tid]


def render_query(rng, q, query_ids):
    pred, (a, b) = q
    tpls = QUERY_TPL[pred](a, b)
    tid = rng.choice(query_ids[pred])
    return tpls[tid]


# ---------------------------------------------------------------------------------------------------
# PARSE (EXACT inversion -- we own the templates). Strip articles/determiners + de-pluralize for the
# engine check. We keep a per-example singular map so plural surfaces invert to the canonical label.
# ---------------------------------------------------------------------------------------------------
DETS = {"a", "an", "the", "every", "all"}


def _strip(words):
    return [w for w in words if w not in DETS]


def parse_sentence(words, sing):
    """Invert a rendered FACT sentence (without trailing '.') -> atom, or None. `sing` maps any plural
    surface form back to its singular label (built per example)."""
    w = [sing.get(t, t) for t in _strip(words)]
    # isa:  [x, is, y]  |  [x, are, y]
    if len(w) == 3 and w[1] in ("is", "are"):
        return ("isa", (w[0], w[2]))
    # part_of: [x, is, part, of, y] | [x, are, part, of, y]  -> part_of(x,y)
    if len(w) == 5 and w[1] in ("is", "are") and w[2] == "part" and w[3] == "of":
        return ("part_of", (w[0], w[4]))
    # has form: [y, has, x] | [y, have, x]  -> part_of(x,y)
    if len(w) == 3 and w[1] in ("has", "have"):
        return ("part_of", (w[2], w[0]))
    # located_in: [x, lives, in, y] | [x, live, in, y] | [x, is, located, in, y] | [x, is, inside, y]
    if len(w) == 4 and w[1] in ("lives", "live") and w[2] == "in":
        return ("located_in", (w[0], w[3]))
    if len(w) == 5 and w[1] == "is" and w[2] == "located" and w[3] == "in":
        return ("located_in", (w[0], w[4]))
    if len(w) == 4 and w[1] == "is" and w[2] == "inside":
        return ("located_in", (w[0], w[3]))
    # has_prop: [x, can, p]
    if len(w) == 3 and w[1] == "can":
        return ("has_prop", (w[0], w[2]))
    return None


# ---------------------------------------------------------------------------------------------------
def config_key(edb, q):
    return (frozenset(edb), q)


def _build_chain(rng, pool, n):
    return rng.sample(pool, n)


def make_example(rng, qtype):
    """Fresh random NL relation config + a (query, gold) of the requested type. All labels sampled fresh
    per example -> no memorization. Transitive queries need >=2 hops for a positive; INHERIT needs the
    isa bridge + R3; EXCLUDE is a SUPPORTED negative (property the subject provably does NOT inherit)."""
    edb = set()
    meta = {}

    # build an isa chain (used by ISA, INHERIT, EXCLUDE) and parallel part_of / located_in chains
    n_isa = rng.randint(4, 5)
    isa_chain = _build_chain(rng, NOUNS, n_isa + 2)
    spares = isa_chain[n_isa:]
    isa_chain = isa_chain[:n_isa]
    for i in range(n_isa - 1):
        edb.add(("isa", (isa_chain[i], isa_chain[i + 1])))

    # property facts: attach a property to a category at index >=1 so members below inherit through isa
    props = rng.sample(PROPS, 3)
    cat_idx = rng.randint(1, n_isa - 1)
    cat = isa_chain[cat_idx]
    p_inh = props[0]
    edb.add(("has_prop", (cat, p_inh)))
    edb.add(("has_prop", (spares[0], props[1])))
    meta.update(isa_chain=isa_chain, cat_idx=cat_idx, cat=cat, p_inh=p_inh, props=props, spares=spares)

    if qtype in ("PARTOF", "LOCIN"):
        rel = "part_of" if qtype == "PARTOF" else "located_in"
        pool = PARTS if qtype == "PARTOF" else PLACES
        # part_of: a part chain (part -> bigger part -> whole); located_in: place containment chain
        n = rng.randint(4, 5)
        chain = _build_chain(rng, pool if rel == "located_in" else (PARTS + NOUNS), n + 2)
        ch = chain[:n]
        sp = chain[n:]
        for i in range(n - 1):
            edb.add((rel, (ch[i], ch[i + 1])))
        if rng.random() < 0.5:                       # positive: transitive pair distance >=2
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            q = (rel, (ch[i], ch[j]))
        else:                                        # negative: head -> a non-ancestor (supported no)
            tgt = sp[0]
            q = (rel, (ch[0], tgt))
        return edb, q, DL.entails(edb, q), meta

    if qtype == "ISA":
        if rng.random() < 0.5:
            i = rng.randint(0, n_isa - 3)
            j = rng.randint(i + 2, n_isa - 1)
            q = ("isa", (isa_chain[i], isa_chain[j]))
        else:
            tgt = spares[1]
            q = ("isa", (isa_chain[0], tgt))
        return edb, q, DL.entails(edb, q), meta

    if qtype == "INHERIT":                            # positive cross-relation composition (member<cat)
        member = isa_chain[rng.randint(0, cat_idx - 1)]
        q = ("has_prop", (member, p_inh))
        return edb, q, DL.entails(edb, q), meta

    # EXCLUDE: SUPPORTED negative. member exists, query a property the member does NOT inherit.
    member = isa_chain[rng.randint(0, cat_idx - 1)]
    neg_p = props[2]                                  # a property held by no one in this config
    q = ("has_prop", (member, neg_p))
    gold = DL.entails(edb, q)
    assert gold is False, "EXCLUDE must be a supported negative"
    return edb, q, gold, meta


# ---------------------------------------------------------------------------------------------------
# Proof rendered as an NL reasoning chain. Transitive relations walk their chain; INHERIT emits the
# recalled property leaf FIRST (reason_lang's AR-drift fix), then the isa bridge; negatives walk with
# no "so" so the trace-grounded read-out returns "no".
# ---------------------------------------------------------------------------------------------------
def _succ(edb, rel):
    return {a: b for (p, (a, b)) in edb if p == rel}


def _walk(edb, rel, x):
    succ = _succ(edb, rel)
    out, cur = [], x
    while cur in succ:
        out.append((cur, succ[cur]))
        cur = succ[cur]
    return out


# canonical proof-chain sentence renderers (always the SAME canonical surface in the proof, so the
# model produces a consistent CoT; held-out-template variety is in the PROMPT, not the CoT).
def _isa_sent(a, b):
    return [article(a), a, "is", article(b), b, "."]


def _part_sent(a, b):
    return [article(a), a, "is", "part", "of", article(b), b, "."]


def _loc_sent(a, b):
    return [article(a), a, "is", "located", "in", article(b), b, "."]


def _prop_sent(a, p):
    return [article(a), a, "can", p, "."]


SENT = {"isa": _isa_sent, "part_of": _part_sent, "located_in": _loc_sent, "has_prop": _prop_sent}


# negative ("cannot") sentence renderers -- parallel form to the positive SENT, but using "cannot"/
# negated phrasing so a grounded terminal `so ... cannot ...` line yields verdict "no".
def _isa_neg_sent(a, b):
    return [article(a), a, "is", "no", article(b), b, "."]


def _part_neg_sent(a, b):
    return [article(a), a, "is", "no", "part", "of", article(b), b, "."]


def _loc_neg_sent(a, b):
    return [article(a), a, "is", "no", "located", "in", article(b), b, "."]


def _prop_neg_sent(a, p):
    return [article(a), a, "cannot", p, "."]


SENT_NEG = {"isa": _isa_neg_sent, "part_of": _part_neg_sent,
            "located_in": _loc_neg_sent, "has_prop": _prop_neg_sent}


def render_proof_chain(edb, q, gold):
    """Render a PROOF chain that is PARALLEL for yes and no.

    Positive: walk/bridge the supporting facts, then `so <q> .` -> answer yes.
    Negative (NEGATION-AS-FAILURE, grounded): emit the relevant walk the subject DOES have, then a
    `note no <q-negated> .` failure marker (verified: q NOT in closure), then a terminal
    `so <q-negated> .` -> answer no. Both verdicts get a distinctive `note`+`so` structure so the model
    is not biased toward "yes".
    """
    pred, (a, b) = q
    rows = []
    reached = False
    if pred in ("isa", "part_of", "located_in"):
        walk = _walk(edb, pred, a)
        for (ca, cc) in walk:
            rows.append(SENT[pred](ca, cc))
        if b in {cc for (_ca, cc) in walk}:
            reached = True
    else:  # has_prop INHERIT: recalled property leaf FIRST, then isa bridge subject->holder
        holders = {x for (p, (x, px)) in edb if p == "has_prop" and px == b}
        bridge, cur, target = [], a, None
        succ = _succ(edb, "isa")
        while cur in succ:
            nxt = succ[cur]
            bridge.append((cur, nxt))
            if nxt in holders:
                target = nxt
                break
            cur = nxt
        if target is not None:
            rows.append(["note"] + _prop_sent(target, b))
            for (ca, cc) in bridge:
                rows.append(_isa_sent(ca, cc))
            reached = True
        else:
            for (ca, cc) in _walk(edb, "isa", a):     # negative: full isa walk the subject DOES have
                rows.append(_isa_sent(ca, cc))
    if reached:
        rows.append(["so"] + SENT[pred](a, b))          # positive verdict (grounded: q in closure)
    else:
        # NEGATION-AS-FAILURE: failure marker + grounded terminal negative verdict.
        rows.append(["note", "no"] + SENT_NEG[pred](a, b))
        rows.append(["so"] + SENT_NEG[pred](a, b))
    assert reached == gold, ("render/oracle disagree", q, gold, rows)
    return rows


# ---------------------------------------------------------------------------------------------------
def _build_sing(edb, q):
    """Per-example plural->singular map covering every label that appears, so plural surfaces invert."""
    labels = set()
    for (_p, (x, y)) in edb:
        labels.add(x); labels.add(y)
    labels.add(q[1][0]); labels.add(q[1][1])
    sing = {}
    for lab in labels:
        sing[plural(lab)] = lab
    return sing


def render_prompt(rng, edb, q, fact_ids, query_ids):
    facts = list(edb)
    rng.shuffle(facts)
    toks = []
    for atom in facts:
        toks += render_fact(rng, atom, fact_ids)
    toks += render_query(rng, q, query_ids)
    return toks


def make_batch(rng, stoi, proof, fact_ids, query_ids, device, batch=64, block=240,
               assert_no_holdout=None):
    seqs, masks, configs, qtypes = [], [], [], []
    fill = stoi["<pad>"]
    tries = 0
    while len(seqs) < batch and tries < batch * 6:
        tries += 1
        qtype = rng.choice(QTYPES)
        edb, q, gold, meta = make_example(rng, qtype)
        ans = "yes" if gold else "no"
        prompt = render_prompt(rng, edb, q, fact_ids, query_ids)
        if assert_no_holdout is not None:
            # verify NO held-out template surface leaked into a train prompt (meaning-test integrity)
            assert _no_heldout_template(prompt, edb, q), "held-out template leaked into TRAIN"
        toks = list(prompt)
        if proof:
            rows = render_proof_chain(edb, q, gold)
            sup_start = len(toks)
            for r in rows:
                toks += r
            toks += ["answer", ans, "."]
        else:
            sup_start = len(toks)
            toks += ["answer", ans, "."]
        ids = [stoi[t] for t in toks]
        if len(ids) > block:
            continue
        m = [0] * len(ids)
        if proof:
            for i in range(sup_start, len(ids)):
                m[i] = 1
        else:
            m[sup_start + 1] = 1
        seqs.append(ids)
        masks.append(m)
        configs.append(config_key(edb, q))
        qtypes.append(qtype)
    if not seqs:
        return None
    L = min(max(len(s) for s in seqs), block)
    ids_b = torch.full((len(seqs), L), fill, dtype=torch.long)
    mask_b = torch.zeros((len(seqs), L), dtype=torch.float)
    for r, (s, m) in enumerate(zip(seqs, masks)):
        ids_b[r, :len(s)] = torch.tensor(s)
        mask_b[r, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids_b.to(device), mask_b.to(device), configs, qtypes


def _no_heldout_template(prompt, edb, q):
    """A train prompt must contain ZERO held-out template surfaces. We re-render each fact/query with
    ALL its templates and confirm any held-out template surface is NOT a substring of the prompt."""
    ptoks = " ".join(prompt)
    for atom in edb:
        pred = atom[0]
        a, b = atom[1]
        for tid in heldout_fact_ids(pred):
            surf = " ".join(FACT_TPL[pred](a, b)[tid])
            if surf in ptoks:
                return False
    pred = q[0]
    a, b = q[1]
    for tid in heldout_query_ids(pred):
        surf = " ".join(QUERY_TPL[pred](a, b)[tid])
        if surf in ptoks:
            return False
    return True


def parse_neg_sentence(words, sing):
    """Invert a rendered NEGATIVE terminal sentence (without trailing '.') -> atom whose ABSENCE the
    line asserts, or None. Mirrors parse_sentence but for the SENT_NEG surfaces. A leading bare 'no'
    (from the `note no ...` failure marker) is tolerated/stripped."""
    raw = list(words)
    if raw and raw[0] == "no":
        raw = raw[1:]
    w = [sing.get(t, t) for t in _strip(raw)]
    # has_prop negative: [x, cannot, p]
    if len(w) == 3 and w[1] == "cannot":
        return ("has_prop", (w[0], w[2]))
    # isa negative: [x, is, no, y]
    if len(w) == 4 and w[1] == "is" and w[2] == "no":
        return ("isa", (w[0], w[3]))
    # part_of negative: [x, is, no, part, of, y]
    if len(w) == 6 and w[1] == "is" and w[2] == "no" and w[3] == "part" and w[4] == "of":
        return ("part_of", (w[0], w[5]))
    # located_in negative: [x, is, no, located, in, y]
    if len(w) == 6 and w[1] == "is" and w[2] == "no" and w[3] == "located" and w[4] == "in":
        return ("located_in", (w[0], w[5]))
    return None


def loss_fn(model, ids, mask):
    logits = model(ids)[:, :-1]
    target = ids[:, 1:]
    tgt_mask = (mask[:, 1:] > 0)
    sel_logits = logits[tgt_mask]
    sel_target = target[tgt_mask]
    if sel_logits.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(sel_logits, sel_target)


# ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def greedy_answer(model, prompt_toks, q, edb, stoi, itos, proof, device, max_new=90, block=240):
    pred, (a, b) = q
    ids = [stoi[t] for t in prompt_toks]
    yes_id, no_id = stoi["yes"], stoi["no"]
    answer_id, dot_id, so_id = stoi["answer"], stoi["."], stoi["so"]

    if not proof:
        ctx = torch.tensor([ids[-block:]], device=device)
        nxt = int(model(ctx)[0, -1].argmax())
        ids.append(nxt if nxt == answer_id else answer_id)
        ctx = torch.tensor([ids[-block:]], device=device)
        logits = model(ctx)[0, -1]
        return "yes" if float(logits[yes_id]) >= float(logits[no_id]) else "no"

    closure = DL.closure(set(edb))[0]
    sing = _build_sing(edb, q)
    verdict, cur, sents, saw_answer, in_so = None, [], 0, False, False
    for _ in range(max_new):
        ctx = torch.tensor([ids[-block:]], device=device)
        logits = model(ctx)[0, -1]
        nxt = int(logits.argmax())
        if saw_answer:
            # NEGATION-AS-FAILURE / closed-world: "yes" must be EARNED by a grounded positive `so`
            # proof of the exact query (verdict=="yes"). If the chain grounded a negative verdict, or
            # produced no grounded positive at all, the sound answer is "no". We do NOT fall back to the
            # bare yes/no token -- that fallback let a hallucinated positive proof leak a "yes" for an
            # EXCLUDE query (the residual failure). Absence of a verified positive derivation == no.
            return "yes" if verdict == "yes" else "no"
        ids.append(nxt)
        if nxt == answer_id:
            saw_answer = True
            continue
        if nxt == so_id:
            in_so = True
            cur = []
            continue
        if itos[nxt] == "note":
            cur = []
            continue
        if nxt == dot_id:
            w = [itos[t] for t in cur]
            parsed = parse_sentence(w, sing)
            if parsed is not None and parsed in closure:          # ENGINE CHECK -> reject hallucination
                if in_so and parsed == (pred, (a, b)):
                    verdict = "yes"
            else:
                # NEGATION-AS-FAILURE: a grounded `so <q-negated> .` line. ENGINE CHECK: the asserted
                # absence must hold (query NOT in closure), mirroring the positive grounding above.
                neg = parse_neg_sentence(w, sing)
                if neg is not None and neg == (pred, (a, b)) and neg not in closure:
                    if in_so:
                        verdict = "no"
            cur, in_so = [], False
            sents += 1
            if sents > 4 * len(edb) + 8:
                break
        else:
            cur.append(nxt)
    return verdict if verdict is not None else "no"


@torch.no_grad()
def evaluate(model, rng, stoi, itos, proof, device, fact_ids, query_ids, n_eval=80, seen=None,
             expect_heldout=False):
    """If expect_heldout: at least one of fact/query uses a held-out template id (template-holdout eval).
    We verify by asserting a held-out template surface IS present in the prompt for that axis."""
    model.eval()
    per = {qt: [0, 0] for qt in QTYPES}
    for _ in range(n_eval):
        for qtype in QTYPES:
            edb, q, gold, meta = make_example(rng, qtype)
            if seen is not None:
                assert config_key(edb, q) not in seen, "held-out config overlaps train!"
            gold_s = "yes" if gold else "no"
            prompt = render_prompt(rng, edb, q, fact_ids, query_ids)
            pred = greedy_answer(model, prompt, q, edb, stoi, itos, proof, device)
            per[qtype][1] += 1
            per[qtype][0] += int(pred == gold_s)
    model.train()
    acc = {qt: per[qt][0] / max(1, per[qt][1]) for qt in QTYPES}
    acc["OVERALL"] = sum(per[qt][0] for qt in QTYPES) / max(1, sum(per[qt][1] for qt in QTYPES))
    # hop breakdown: ISA/PARTOF/LOCIN/INHERIT positives are multi-hop; treat transitive+inherit as multi
    acc["1HOP"] = acc["ISA"]                       # ISA includes distance-2 (multi) but is the simplest
    return acc


def _all_ids(template_holdout):
    """Build fact_ids/query_ids dicts for train (subset) or 'all' for test-with-heldout."""
    rels_f = ["isa", "part_of", "located_in", "has_prop"]
    rels_q = ["isa", "part_of", "located_in", "has_prop"]
    fids = {r: allowed_fact_ids(r, template_holdout) for r in rels_f}
    qids = {r: allowed_query_ids(r, template_holdout) for r in rels_q}
    return fids, qids


def _heldout_only_ids():
    """For TEMPLATE-holdout TEST: use ONLY the held-out template ids (unseen phrasings)."""
    rels = ["isa", "part_of", "located_in", "has_prop"]
    fids = {r: list(heldout_fact_ids(r)) for r in rels}
    qids = {r: list(heldout_query_ids(r)) for r in rels}
    return fids, qids


def train_one(proof, seed, steps, lr, device, vocab,
              dim=192, layers=4, heads=6, batch=64, max_len=240):
    itos, stoi = vocab
    assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
    assert (dim // heads) % 2 == 0, f"head dim ({dim // heads}) must be even for RoPE"
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = ScratchpadLM(vocab=len(itos), d=dim, layers=layers, heads=heads, max_len=max_len,
                         pos_mode="rope", pointer=True, tie=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    # TRAIN uses only NON-held-out templates (template_holdout=True restricts ids)
    tr_fids, tr_qids = _all_ids(template_holdout=True)
    seen, last = set(), 0.0
    for step in range(steps):
        out = make_batch(rng, stoi, proof, tr_fids, tr_qids, device, batch=batch,
                         block=max_len, assert_no_holdout=True)
        ids, mask, configs, _ = out
        seen.update(configs)
        loss = loss_fn(model, ids, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = loss.item()
        if (step + 1) % 200 == 0:
            print(f"    [{'PROOF' if proof else 'ANSWER'} seed{seed}] step {step+1}/{steps} "
                  f"loss {last:.4f}", flush=True)

    # TRAIN-acc gate: seen templates, seen-ish configs
    train_acc = evaluate(model, random.Random(seed + 1000), stoi, itos, proof, device,
                         tr_fids, tr_qids, n_eval=60)
    # (a) CONFIG-holdout: novel configs, SEEN templates
    cfg_acc = evaluate(model, random.Random(seed + 7000), stoi, itos, proof, device,
                       tr_fids, tr_qids, n_eval=80, seen=seen)
    # (b) TEMPLATE-holdout: novel configs AND UNSEEN phrasings (held-out template ids only)
    ho_fids, ho_qids = _heldout_only_ids()
    tpl_acc = evaluate(model, random.Random(seed + 13000), stoi, itos, proof, device,
                       ho_fids, ho_qids, n_eval=80, seen=seen, expect_heldout=True)
    return train_acc, cfg_acc, tpl_acc, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--which", choices=["A", "B", "both"], default="both")
    # GPU + scale knobs (defaults equal the previous hardcoded values -> CPU behavior unchanged)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=240)
    ap.add_argument("--device", default="auto",
                    help="auto -> cuda if available else cpu; or pass cpu/cuda explicitly")
    ap.add_argument("--out", default=None, help="optional path to write per-seed JSON results")
    args = ap.parse_args()

    assert args.dim % args.heads == 0, f"--dim ({args.dim}) must be divisible by --heads ({args.heads})"
    assert (args.dim // args.heads) % 2 == 0, \
        f"head dim ({args.dim // args.heads}) must be even for RoPE"

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    # Coexist politely under a heavily-oversubscribed CPU (concurrent agents): allow capping the
    # intra-op thread pool via env (fewer threads = less thrash when all cores are already taken).
    _nt = os.environ.get("RLN_THREADS")
    torch.set_num_threads(int(_nt) if _nt else max(1, os.cpu_count() or 1))
    vocab = build_vocab()
    itos, stoi = vocab

    mk = dict(dim=args.dim, layers=args.layers, heads=args.heads, batch=args.batch,
              max_len=args.max_len)

    print(f"REASON-LANG2 | broader NL | rules: isa/part_of/located_in TRANS + has_prop INHERIT + "
          f"EXCLUDE-neg | nouns={len(NOUNS)} props={len(PROPS)} parts={len(PARTS)} places={len(PLACES)} "
          f"| qtypes={QTYPES} | vocab={len(itos)} | steps={args.steps} lr={args.lr}")
    print(f"  model: dim={args.dim} layers={args.layers} heads={args.heads} batch={args.batch} "
          f"max_len={args.max_len} | device={device}")
    print("  template counts (fact/query): " + ", ".join(
        f"{r}:{n_fact_tpls(r)}/{n_query_tpls(r)}" for r in ["isa", "part_of", "located_in", "has_prop"]))
    print("  TEMPLATE-holdout: reserve last fact-id and last query-id per relation for TEST (unseen).")

    # show paraphrase variety + a held-out template example
    rng = random.Random(2)
    print("\n  paraphrase examples (same isa fact robin/bird, all train+test templates):")
    for t in _isa_fact_templates("robin", "bird"):
        print("    -", " ".join(t))
    print("  held-out (TEST-only) isa fact template:",
          " ".join(_isa_fact_templates("robin", "bird")[list(heldout_fact_ids("isa"))[0]]))

    # show a rendered NL proof example (INHERIT)
    while True:
        edb, q, gold, meta = make_example(rng, "INHERIT")
        if gold and meta["cat_idx"] >= 2:
            break
    tr_fids, tr_qids = _all_ids(template_holdout=True)
    print("\n  INHERIT example (multi-hop cross-relation):")
    print("    facts:", " ".join(render_prompt(random.Random(3), edb, q, tr_fids, tr_qids)))
    rows = render_proof_chain(edb, q, gold)
    print("    proof chain:", "  ".join(" ".join(r) for r in rows), " answer yes .")

    if args.sanity:
        print("\n=== SANITY GATE: PROOF must LEARN TRAIN (>0.85 overall) ===")
        tr, cfg, tpl, ls = train_one(True, 0, args.steps, args.lr, device, vocab, **mk)
        print(f"PROOF seed0 TRAIN   overall {tr['OVERALL']:.3f} | per-q " +
              " ".join(f"{k} {tr[k]:.2f}" for k in QTYPES))
        print(f"PROOF seed0 CONFIG  overall {cfg['OVERALL']:.3f} | per-q " +
              " ".join(f"{k} {cfg[k]:.2f}" for k in QTYPES))
        print(f"PROOF seed0 TEMPLT  overall {tpl['OVERALL']:.3f} | per-q " +
              " ".join(f"{k} {tpl[k]:.2f}" for k in QTYPES))
        gate_pass = tr['OVERALL'] > 0.85
        print(f"GATE {'PASS' if gate_pass else 'FAIL'} (train overall {tr['OVERALL']:.3f}) "
              f"@ {args.steps} steps")
        if args.out:
            _write_results(args.out, args, device, mode="sanity", payload={
                "loss": ls, "gate_pass": gate_pass,
                "seed0": {"train": tr, "config": cfg, "template": tpl}})
            print(f"  wrote results -> {args.out}")
        return

    print(f"\n=== A (answer-only) vs B (proof) : {args.seeds}-seed, config- & template-holdout ===")
    res = {"A": {"cfg": [], "tpl": []}, "B": {"cfg": [], "tpl": []}}
    trn = {"A": [], "B": []}
    per_seed = []                                       # for --out JSON (full A/B/train/cfg/tpl)
    for seed in range(args.seeds):
        print(f"\n-- seed {seed} --")
        rec = {"seed": seed}
        if args.which in ("A", "both"):
            trA, cfgA, tplA, _ = train_one(False, seed, args.steps, args.lr, device, vocab, **mk)
            print(f"  (A) ANSWER train {trA['OVERALL']:.3f} | CFG {cfgA['OVERALL']:.3f} "
                  f"TPL {tplA['OVERALL']:.3f}")
            res["A"]["cfg"].append(cfgA); res["A"]["tpl"].append(tplA); trn["A"].append(trA["OVERALL"])
            rec["A"] = {"train": trA, "config": cfgA, "template": tplA}
        if args.which in ("B", "both"):
            trB, cfgB, tplB, _ = train_one(True, seed, args.steps, args.lr, device, vocab, **mk)
            print(f"  (B) PROOF  train {trB['OVERALL']:.3f} | CFG {cfgB['OVERALL']:.3f} "
                  f"TPL {tplB['OVERALL']:.3f}")
            res["B"]["cfg"].append(cfgB); res["B"]["tpl"].append(tplB); trn["B"].append(trB["OVERALL"])
            rec["B"] = {"train": trB, "config": cfgB, "template": tplB}
        per_seed.append(rec)
        if args.out:                                  # incremental save: survive mid-run death
            _write_results(args.out, args, device, mode="AB", payload={"per_seed": per_seed})
            print(f"  [saved {len(per_seed)} seed(s) -> {args.out}]", flush=True)

    def ms(xs):
        m = sum(xs) / len(xs)
        return m, (max(xs) - min(xs)) / 2

    keys = ["OVERALL"] + list(QTYPES)
    print("\n================= SUMMARY (held-out accuracy, chance=0.50) =================")
    ta = f"{sum(trn['A'])/len(trn['A']):.3f}" if trn['A'] else "n/a"
    tb = f"{sum(trn['B'])/len(trn['B']):.3f}" if trn['B'] else "n/a"
    print(f"{args.seeds} seeds, mean +/- half-spread | train overall: A {ta} B {tb}")
    for holdout in ("cfg", "tpl"):
        label = "CONFIG-HOLDOUT" if holdout == "cfg" else "TEMPLATE-HOLDOUT (unseen phrasings)"
        print(f"\n  [{label}]")
        print(f"  {'method':<12s} | " + " | ".join(f"{k:^11s}" for k in keys))
        for tag, lab in (("A", "ANSWER-ONLY"), ("B", "PROOF-SUPER")):
            if not res[tag][holdout]:
                continue
            cells = []
            for k in keys:
                m, s = ms([r[k] for r in res[tag][holdout]])
                cells.append(f"{m:.2f}±{s:.2f}")
            print(f"  {lab:<12s} | " + " | ".join(f"{c:^11s}" for c in cells))
    print("=" * 76)

    if args.out:
        _write_results(args.out, args, device, mode="AB", payload={"per_seed": per_seed})
        print(f"\nwrote results -> {args.out}")


def _write_results(path, args, device, mode, payload):
    """Persist a run's numbers so a GPU run's results are retrievable as JSON."""
    import json
    out = {
        "mode": mode,
        "config": {"steps": args.steps, "lr": args.lr, "seeds": args.seeds,
                   "which": args.which, "dim": args.dim, "layers": args.layers,
                   "heads": args.heads, "batch": args.batch, "max_len": args.max_len,
                   "device": device},
        "results": payload,
    }
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
