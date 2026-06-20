#!/usr/bin/env python3
"""PHASE 1 (language-reasoning roadmap): read English facts about REAL entities with REAL word
meanings and answer queries by COMPOSING them -- so the model learns what words actually MEAN and
their associations, not random synthetic label bindings.

This is a fork-in-spirit of reason_lang_neg.py (DO NOT modify that file). It reuses, unchanged:
  - the shared Datalog engine (datalog.Datalog) as the ground-truth oracle for every yes/no label,
  - the NL proof-chain rendering idea (positive walk/bridge + grounded negation-as-failure),
  - the closed-world trace-grounded greedy read-out (a "yes" must be EARNED by an engine-verified
    positive derivation of the exact query; absence of one == "no"),
  - ScratchpadLM (pointer head, RoPE, tied) as the model.

THE NEW THING: instead of fresh-random labels per example (which can't carry meaning), there is ONE
small, CURATED, semantically-TRUE knowledge base of REAL common entities. "robin isa bird",
"bird isa animal", "wing part_of bird", "bird can fly", "fish can swim" -- facts a person agrees
with. Because the labels are real English words reused across many examples, learning to compose
them IS learning word-meaning association (a robin inherits "can move" from animal because the model
has bound the *meaning* of robin/bird/animal/move, not a one-off random symbol).

HELD-OUT COMPOSITION (the meaningful generalization test, enforced here for the lead to measure):
  We designate a set of (entity, derived-fact) MULTI-HOP queries as held-out. For those, the prompt
  states ONLY the base 1-hop facts (e.g. "robin isa bird", "bird isa animal", "animal can move") and
  asks "can a robin move?". The derived answer (robin can move / robin isa animal) is NEVER stated
  verbatim in the prompt -- we ASSERT this. The model must COMPOSE real facts it was never directly
  shown the answer for. Train examples never use a held-out (subject, query) pair.

  python -m thinking.reason_realtext --sanity --steps 3000   # TRAIN-acc gate (>0.85), CPU
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
# ENGINE: identical Horn rules to reason_lang_neg -- transitivity + has_prop inheritance. The engine
# is the oracle for every label (yes & no) over the REAL knowledge base.
# ---------------------------------------------------------------------------------------------------
RULES = [
    (("isa", ("?x", "?z")),        [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))]),          # transitive
    (("part_of", ("?x", "?z")),    [("part_of", ("?x", "?y")), ("part_of", ("?y", "?z"))]),  # transitive
    (("has_prop", ("?x", "?p")),   [("isa", ("?x", "?y")), ("has_prop", ("?y", "?p"))]),      # inherit
]
DL = Datalog(RULES)

# ---------------------------------------------------------------------------------------------------
# THE REAL MINI-KB. Every fact is semantically TRUE common-sense English. Base (1-hop) facts only;
# everything multi-hop is DERIVED by the engine. Words are reused across examples -> meaning, not
# random labels.
# ---------------------------------------------------------------------------------------------------
# is-a hierarchy (specific -> general), all base 1-hop edges.
ISA = [
    # birds
    ("robin", "bird"), ("sparrow", "bird"), ("eagle", "bird"), ("owl", "bird"),
    ("hawk", "bird"), ("penguin", "bird"), ("duck", "bird"),
    ("bird", "animal"),
    # fish
    ("salmon", "fish"), ("trout", "fish"), ("shark", "fish"), ("tuna", "fish"),
    ("fish", "animal"),
    # mammals
    ("dog", "mammal"), ("cat", "mammal"), ("wolf", "mammal"), ("lion", "mammal"),
    ("tiger", "mammal"), ("whale", "mammal"), ("bat", "mammal"), ("horse", "mammal"),
    ("mammal", "animal"),
    # reptiles / amphibians / insects
    ("snake", "reptile"), ("lizard", "reptile"), ("turtle", "reptile"),
    ("reptile", "animal"),
    ("frog", "amphibian"), ("toad", "amphibian"),
    ("amphibian", "animal"),
    ("bee", "insect"), ("ant", "insect"), ("beetle", "insect"),
    ("insect", "animal"),
    # top level
    ("animal", "living_thing"),
    # plants
    ("oak", "tree"), ("pine", "tree"), ("maple", "tree"),
    ("tree", "plant"),
    ("rose", "flower"), ("daisy", "flower"), ("tulip", "flower"),
    ("flower", "plant"),
    ("plant", "living_thing"),
]

# part_of (part -> larger part / whole), all base 1-hop edges, semantically true.
PART_OF = [
    ("feather", "wing"), ("wing", "bird"),
    ("beak", "bird"), ("talon", "bird"),
    ("fin", "fish"), ("gill", "fish"), ("scale", "fish"),
    ("paw", "leg"), ("leg", "mammal"), ("fur", "mammal"), ("tail", "mammal"),
    ("petal", "flower"), ("stem", "flower"),
    ("leaf", "branch"), ("branch", "tree"), ("root", "tree"), ("trunk", "tree"),
]

# has_prop: a property attached to a category; members inherit it through isa. All true.
HAS_PROP = [
    ("animal", "move"), ("animal", "breathe"), ("animal", "grow"),
    ("bird", "fly"), ("bird", "lay_eggs"),
    ("fish", "swim"),
    ("mammal", "nurse"),
    ("reptile", "crawl"),
    ("insect", "crawl"),
    ("plant", "grow"), ("plant", "need_sunlight"),
    ("flower", "bloom"),
    ("tree", "grow_tall"),
    ("living_thing", "live"),
]

EDB = set()
for (a, b) in ISA:
    EDB.add(("isa", (a, b)))
for (a, b) in PART_OF:
    EDB.add(("part_of", (a, b)))
for (a, p) in HAS_PROP:
    EDB.add(("has_prop", (a, p)))

CLOSURE, PROV = DL.closure(EDB)

# all vocabulary words that ever appear as a label
LABELS = set()
for (_p, (x, y)) in EDB:
    LABELS.add(x); LABELS.add(y)
LABELS = sorted(LABELS)

STRUCT = ["<pad>", "a", "an", "is", "can", "cannot", "has", "in", "are", "every", "all",
          "have", "located", "true", "it", "that", "of", "part", "the",
          "so", "note", "answer", "yes", "no", "query", ".", "?"]

QTYPES = ("ISA", "PARTOF", "INHERIT", "EXCLUDE")


def article(word):
    return "an" if word and word[0] in "aeiou" else "a"


def plural(word):
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def build_vocab():
    base = list(STRUCT) + list(LABELS)
    assert len(base) == len(set(base)), "duplicate base vocab token: " + str(
        [t for t in base if base.count(t) > 1][:5])
    plurals = []
    for w in LABELS:
        pw = plural(w)
        if pw not in base and pw not in plurals:
            plurals.append(pw)
    itos = base + plurals
    assert len(itos) == len(set(itos)), "duplicate vocab token (with plurals)"
    return itos, {t: i for i, t in enumerate(itos)}


# ---------------------------------------------------------------------------------------------------
# PARAPHRASE TEMPLATES -- multiple real-English surfaces per fact/query (reused idea from
# reason_lang_neg). We own every template so parsing back to the relation is EXACT.
# ---------------------------------------------------------------------------------------------------
def _isa_fact_templates(x, y):
    return [
        [article(x), x, "is", article(y), y, "."],
        [plural(x), "are", plural(y), "."],
        ["every", x, "is", article(y), y, "."],
        ["all", plural(x), "are", plural(y), "."],
    ]


def _part_fact_templates(x, y):
    return [
        [article(x), x, "is", "part", "of", article(y), y, "."],
        [plural(x), "are", "part", "of", plural(y), "."],
        [article(y), y, "has", article(x), x, "."],
        [plural(y), "have", plural(x), "."],
    ]


def _prop_fact_templates(x, p):
    return [
        [article(x), x, "can", p, "."],
        [plural(x), "can", p, "."],
        ["every", x, "can", p, "."],
    ]


def _isa_query_templates(x, y):
    return [
        ["query", "is", article(x), x, article(y), y, "?"],
        ["query", "are", plural(x), plural(y), "?"],
        ["query", "is", "it", "true", "that", article(x), x, "is", article(y), y, "?"],
    ]


def _part_query_templates(x, y):
    return [
        ["query", "is", article(x), x, "part", "of", article(y), y, "?"],
        ["query", "is", article(x), x, "in", article(y), y, "?"],
        ["query", "is", "it", "true", "that", article(x), x, "is", "part", "of", article(y), y, "?"],
    ]


def _prop_query_templates(x, p):
    return [
        ["query", "can", article(x), x, p, "?"],
        ["query", "is", "it", "true", "that", article(x), x, "can", p, "?"],
        ["query", "can", plural(x), p, "?"],
    ]


FACT_TPL = {"isa": _isa_fact_templates, "part_of": _part_fact_templates, "has_prop": _prop_fact_templates}
QUERY_TPL = {"isa": _isa_query_templates, "part_of": _part_query_templates, "has_prop": _prop_query_templates}


def n_fact_tpls(rel):
    return len(FACT_TPL[rel]("robin", "bird"))


def n_query_tpls(rel):
    return len(QUERY_TPL[rel]("robin", "bird"))


def render_fact(rng, atom):
    pred, (a, b) = atom
    return rng.choice(FACT_TPL[pred](a, b))


def render_query(rng, q):
    pred, (a, b) = q
    return rng.choice(QUERY_TPL[pred](a, b))


# ---------------------------------------------------------------------------------------------------
# PARSE (exact inversion). Strip determiners, de-pluralize via a global singular map.
# ---------------------------------------------------------------------------------------------------
DETS = {"a", "an", "the", "every", "all"}
SING = {plural(w): w for w in LABELS}


def _strip(words):
    return [w for w in words if w not in DETS]


def parse_sentence(words):
    w = [SING.get(t, t) for t in _strip(words)]
    if len(w) == 3 and w[1] in ("is", "are"):
        return ("isa", (w[0], w[2]))
    if len(w) == 5 and w[1] in ("is", "are") and w[2] == "part" and w[3] == "of":
        return ("part_of", (w[0], w[4]))
    if len(w) == 3 and w[1] in ("has", "have"):
        return ("part_of", (w[2], w[0]))
    if len(w) == 3 and w[1] == "can":
        return ("has_prop", (w[0], w[2]))
    return None


def parse_neg_sentence(words):
    raw = list(words)
    if raw and raw[0] == "no":
        raw = raw[1:]
    w = [SING.get(t, t) for t in _strip(raw)]
    if len(w) == 3 and w[1] == "cannot":
        return ("has_prop", (w[0], w[2]))
    if len(w) == 4 and w[1] == "is" and w[2] == "no":
        return ("isa", (w[0], w[3]))
    if len(w) == 6 and w[1] == "is" and w[2] == "no" and w[3] == "part" and w[4] == "of":
        return ("part_of", (w[0], w[5]))
    return None


# ---------------------------------------------------------------------------------------------------
# QUERY BANK: enumerate every possible (qtype, query, gold) over the REAL KB, then split held-out.
# A held-out query is a MULTI-HOP DERIVED query (distance >= 2 / inheritance through >=1 isa edge):
# for these the prompt will contain only the base 1-hop facts on the relevant path, never the answer.
# ---------------------------------------------------------------------------------------------------
def _succ(edb, rel):
    return {a: b for (p, (a, b)) in edb if p == rel}


def _chain(edb, rel, x):
    """ordered list of (a,b) edges from x upward along rel."""
    succ = _succ(edb, rel)
    out, cur = [], x
    while cur in succ:
        out.append((cur, succ[cur]))
        cur = succ[cur]
    return out


def _ancestors(edb, rel, x):
    return [b for (_a, b) in _chain(edb, rel, x)]


ISA_SUCC = _succ(EDB, "isa")
PART_SUCC = _succ(EDB, "part_of")
# leaves / interior nodes for each relation
ISA_NODES = set(ISA_SUCC) | {b for b in ISA_SUCC.values()}
PART_NODES = set(PART_SUCC) | {b for b in PART_SUCC.values()}


def hops_isa(x, y):
    """number of isa edges from x to y, or None if y is not an ancestor."""
    anc = _ancestors(EDB, "isa", x)
    if y in anc:
        return anc.index(y) + 1
    return None


def hops_part(x, y):
    anc = _ancestors(EDB, "part_of", x)
    if y in anc:
        return anc.index(y) + 1
    return None


def inherit_hops(member, prop):
    """min isa hops to a category that directly has prop (>=1 means inherited), or None."""
    holders = {a for (a, p) in HAS_PROP if p == prop}
    chain = [member] + _ancestors(EDB, "isa", member)
    for i, node in enumerate(chain):
        if node in holders:
            return i  # 0 = directly stated, >=1 = inherited through isa
    return None


def build_query_bank():
    """All (qtype, query, gold, hops) over the real KB."""
    bank = {qt: [] for qt in QTYPES}
    # ISA: every (descendant, ancestor) reachable; positives. Also supported negatives.
    for x in ISA_NODES:
        anc = _ancestors(EDB, "isa", x)
        for i, y in enumerate(anc):
            bank["ISA"].append(("isa", (x, y), True, i + 1))
        # negatives: x vs an isa node that is NOT an ancestor and not x
        for y in ISA_NODES:
            if y != x and y not in anc and DL.entails(EDB, ("isa", (x, y))) is False:
                bank["ISA"].append(("isa", (x, y), False, 0))
    # PARTOF: same.
    for x in PART_NODES:
        anc = _ancestors(EDB, "part_of", x)
        for i, y in enumerate(anc):
            bank["PARTOF"].append(("part_of", (x, y), True, i + 1))
        for y in PART_NODES:
            if y != x and y not in anc and DL.entails(EDB, ("part_of", (x, y))) is False:
                bank["PARTOF"].append(("part_of", (x, y), False, 0))
    # INHERIT: member can <prop> where prop is inherited (hops >= 1). positives only.
    all_props = sorted({p for (_a, p) in HAS_PROP})
    for x in ISA_NODES:
        for prop in all_props:
            h = inherit_hops(x, prop)
            if h is not None and h >= 1:
                bank["INHERIT"].append(("has_prop", (x, prop), True, h))
    # EXCLUDE: supported negative -- member does NOT have/inherit prop.
    for x in ISA_NODES:
        for prop in all_props:
            if DL.entails(EDB, ("has_prop", (x, prop))) is False:
                bank["EXCLUDE"].append(("has_prop", (x, prop), False, 0))
    return bank


QBANK = build_query_bank()


# HELD-OUT COMPOSITION SET: pick specific multi-hop derived queries (hops >= 2 for ISA/PARTOF,
# inherit hops >= 2 for INHERIT) whose answer must be COMPOSED. Train never uses these (subj,query).
def build_heldout(seed=0):
    rng = random.Random(seed)
    held = set()
    # ISA: deep inheritance, e.g. "robin isa animal" / "robin isa living_thing"
    deep_isa = [q for q in QBANK["ISA"] if q[2] and q[3] >= 2]
    # PARTOF deep
    deep_part = [q for q in QBANK["PARTOF"] if q[2] and q[3] >= 2]
    # INHERIT deep: property inherited through >=2 isa edges, e.g. "can a robin move?"
    deep_inh = [q for q in QBANK["INHERIT"] if q[3] >= 2]
    # EXCLUDE (negation-as-failure): reserve a fraction of supported-negative queries so negation
    # GENERALIZATION is actually measured (previously 0 EXCLUDE were held out, so held-out negation
    # was untested). The model is still trained on the OTHER EXCLUDE examples, then must answer
    # "no" on these unseen (subject, prop) negatives via the same grounded-failure proof.
    neg = list(QBANK["EXCLUDE"])
    rng.shuffle(deep_isa); rng.shuffle(deep_part); rng.shuffle(deep_inh); rng.shuffle(neg)
    for q in deep_isa[: max(4, len(deep_isa) // 3)]:
        held.add((q[0], q[1]))
    for q in deep_part[: max(2, len(deep_part) // 3)]:
        held.add((q[0], q[1]))
    for q in deep_inh[: max(6, len(deep_inh) // 3)]:
        held.add((q[0], q[1]))
    for q in neg[: max(8, len(neg) // 4)]:
        held.add((q[0], q[1]))
    return held


HELDOUT = build_heldout()


def train_queries():
    """flat list of (qtype, query, gold) NOT in the held-out composition set."""
    out = []
    for qt in QTYPES:
        for (pred, args_, gold, _h) in QBANK[qt]:
            if (pred, args_) in HELDOUT:
                continue
            out.append((qt, (pred, args_), gold))
    return out


def heldout_queries():
    out = []
    for qt in QTYPES:
        for (pred, args_, gold, _h) in QBANK[qt]:
            if (pred, args_) in HELDOUT:
                out.append((qt, (pred, args_), gold))
    return out


# ---------------------------------------------------------------------------------------------------
# MINIMAL SUPPORTING EDB for a query: only the BASE 1-hop facts on the relevant path(s), plus a few
# distractor base facts. The derived answer is NEVER among them (we assert). This is what enforces
# held-out composition: prompt has "robin isa bird", "bird isa animal", "animal can move" -- not
# "robin can move".
# ---------------------------------------------------------------------------------------------------
def support_facts(q):
    """The minimal set of base EDB facts that supports q (or, for a negative, the relevant chain the
    subject DOES have). Returns a set of base atoms, none of which equals the derived query."""
    pred, (a, b) = q
    facts = set()
    if pred == "isa":
        for e in _chain(EDB, "isa", a):
            facts.add(("isa", e))
            if e[1] == b:
                break
        if b not in {y for (_x, y) in _chain(EDB, "isa", a)}:
            # negative: just give the subject's full isa chain (subject does NOT reach b)
            for e in _chain(EDB, "isa", a):
                facts.add(("isa", e))
    elif pred == "part_of":
        for e in _chain(EDB, "part_of", a):
            facts.add(("part_of", e))
            if e[1] == b:
                break
        if b not in {y for (_x, y) in _chain(EDB, "part_of", a)}:
            for e in _chain(EDB, "part_of", a):
                facts.add(("part_of", e))
    else:  # has_prop (INHERIT positive or EXCLUDE negative)
        holders = {h for (h, p) in HAS_PROP if p == b}
        chain = _chain(EDB, "isa", a)
        # walk isa chain up to (and including the edge into) the holder
        reached = False
        for e in chain:
            facts.add(("isa", e))
            if e[0] in holders:  # a itself is a holder (shouldn't be for hops>=1) -- guard
                reached = True
                break
            if e[1] in holders:
                facts.add(("has_prop", (e[1], b)))
                reached = True
                break
        if not reached:
            # negative: full isa chain, no has_prop edge for b -> supported "no"
            for e in chain:
                facts.add(("isa", e))
    return facts


def add_distractors(rng, facts, q, n=3):
    """Add a few unrelated TRUE base facts as distractors (still real KB facts). Never add a fact that
    flips the gold of q, and never add the derived query itself. FAST: gold is the precomputed global
    oracle (q in CLOSURE); we add candidate distractors then do a SINGLE local closure check on the
    final set, dropping the batch only if it changed the gold (rare -- random unrelated facts)."""
    pred, (a, b) = q
    gold = q in CLOSURE
    pool = [atom for atom in EDB if atom not in facts and atom != (pred, (a, b))]
    rng.shuffle(pool)
    out = set(facts) | set(pool[:n])
    # one closure check; if a distractor accidentally enabled q (gold was no, now derivable), retreat.
    if (q in DL.closure(out)[0]) != gold:
        return set(facts)
    return out


# ---------------------------------------------------------------------------------------------------
# PROOF chain (identical structure to reason_lang_neg: positive walk/bridge + grounded NAF negative).
# ---------------------------------------------------------------------------------------------------
def _isa_sent(a, b):
    return [article(a), a, "is", article(b), b, "."]


def _part_sent(a, b):
    return [article(a), a, "is", "part", "of", article(b), b, "."]


def _prop_sent(a, p):
    return [article(a), a, "can", p, "."]


SENT = {"isa": _isa_sent, "part_of": _part_sent, "has_prop": _prop_sent}


def _isa_neg_sent(a, b):
    return [article(a), a, "is", "no", article(b), b, "."]


def _part_neg_sent(a, b):
    return [article(a), a, "is", "no", "part", "of", article(b), b, "."]


def _prop_neg_sent(a, p):
    return [article(a), a, "cannot", p, "."]


SENT_NEG = {"isa": _isa_neg_sent, "part_of": _part_neg_sent, "has_prop": _prop_neg_sent}


def render_proof_chain(facts, q, gold):
    pred, (a, b) = q
    rows = []
    reached = False
    if pred in ("isa", "part_of"):
        walk = _chain(facts, pred, a)
        for (ca, cc) in walk:
            rows.append(SENT[pred](ca, cc))
        if b in {cc for (_ca, cc) in walk}:
            reached = True
    else:  # has_prop inherit
        holders = {x for (p, (x, px)) in facts if p == "has_prop" and px == b}
        bridge, cur, target = [], a, None
        succ = _succ(facts, "isa")
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
            for (ca, cc) in _chain(facts, "isa", a):
                rows.append(_isa_sent(ca, cc))
    if reached:
        rows.append(["so"] + SENT[pred](a, b))
    else:
        rows.append(["note", "no"] + SENT_NEG[pred](a, b))
        rows.append(["so"] + SENT_NEG[pred](a, b))
    assert reached == gold, ("render/oracle disagree", q, gold, rows)
    return rows


# ---------------------------------------------------------------------------------------------------
def render_prompt(rng, facts, q):
    fl = list(facts)
    rng.shuffle(fl)
    toks = []
    for atom in fl:
        toks += render_fact(rng, atom)
    toks += render_query(rng, q)
    return toks


def is_heldout(q):
    return (q[0], q[1]) in HELDOUT


def assert_answer_not_verbatim(facts, q):
    """For a HELD-OUT composition query, the answer must NEVER be stated as a base fact in `facts`
    (the model must compose it). 1-hop direct-lookup TRAIN queries legitimately state the fact. In
    all cases the support's closure must agree with the global oracle for this query (sound label)."""
    if is_heldout(q):
        assert q not in facts, ("held-out query stated verbatim!", q)
    # global gold is precomputed (q in CLOSURE); local closure must agree (one closure, small set).
    assert (q in DL.closure(facts)[0]) == (q in CLOSURE), ("support changed gold", q)


def make_example(rng, item):
    """item = (qtype, query, gold). Build minimal-support EDB + distractors + prompt + proof."""
    qtype, q, gold = item
    facts = support_facts(q)
    facts = add_distractors(rng, facts, q, n=rng.randint(2, 4))  # verifies gold unchanged (1 closure)
    if is_heldout(q):
        assert q not in facts, ("held-out query stated verbatim!", q)
    return facts, q, gold, qtype


def make_batch(rng, stoi, proof, pool, device, batch=64, block=260):
    seqs, masks, qtypes = [], [], []
    fill = stoi["<pad>"]
    tries = 0
    while len(seqs) < batch and tries < batch * 8:
        tries += 1
        item = rng.choice(pool)
        facts, q, gold, qtype = make_example(rng, item)
        ans = "yes" if gold else "no"
        prompt = render_prompt(rng, facts, q)
        toks = list(prompt)
        if proof:
            rows = render_proof_chain(facts, q, gold)
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
        qtypes.append(qtype)
    if not seqs:
        return None
    L = min(max(len(s) for s in seqs), block)
    ids_b = torch.full((len(seqs), L), fill, dtype=torch.long)
    mask_b = torch.zeros((len(seqs), L), dtype=torch.float)
    for r, (s, m) in enumerate(zip(seqs, masks)):
        ids_b[r, :len(s)] = torch.tensor(s)
        mask_b[r, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids_b.to(device), mask_b.to(device), qtypes


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
def greedy_answer(model, prompt_toks, q, facts, stoi, itos, proof, device, max_new=90, block=260):
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

    closure = DL.closure(set(facts))[0]
    verdict, cur, sents, saw_answer, in_so = None, [], 0, False, False
    for _ in range(max_new):
        ctx = torch.tensor([ids[-block:]], device=device)
        logits = model(ctx)[0, -1]
        nxt = int(logits.argmax())
        if saw_answer:
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
            parsed = parse_sentence(w)
            if parsed is not None and parsed in closure:
                if in_so and parsed == (pred, (a, b)):
                    verdict = "yes"
            else:
                neg = parse_neg_sentence(w)
                if neg is not None and neg == (pred, (a, b)) and neg not in closure:
                    if in_so:
                        verdict = "no"
            cur, in_so = [], False
            sents += 1
            if sents > 4 * len(facts) + 8:
                break
        else:
            cur.append(nxt)
    return verdict if verdict is not None else "no"


@torch.no_grad()
def evaluate(model, rng, stoi, itos, proof, device, pool, n_eval=240):
    model.eval()
    per = {qt: [0, 0] for qt in QTYPES}
    items = list(pool)
    for _ in range(n_eval):
        item = rng.choice(items)
        facts, q, gold, qtype = make_example(rng, item)
        gold_s = "yes" if gold else "no"
        prompt = render_prompt(rng, facts, q)
        pred = greedy_answer(model, prompt, q, facts, stoi, itos, proof, device)
        per[qtype][1] += 1
        per[qtype][0] += int(pred == gold_s)
    model.train()
    acc = {qt: (per[qt][0] / per[qt][1] if per[qt][1] else 0.0) for qt in QTYPES}
    tot_c = sum(per[qt][0] for qt in QTYPES)
    tot_n = sum(per[qt][1] for qt in QTYPES)
    acc["OVERALL"] = tot_c / max(1, tot_n)
    return acc


def train_one(proof, seed, steps, lr, device, vocab,
              dim=192, layers=4, heads=6, batch=64, max_len=260):
    itos, stoi = vocab
    assert dim % heads == 0
    assert (dim // heads) % 2 == 0
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = ScratchpadLM(vocab=len(itos), d=dim, layers=layers, heads=heads, max_len=max_len,
                         pos_mode="rope", pointer=True, tie=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    pool = train_queries()
    last = 0.0
    for step in range(steps):
        out = make_batch(rng, stoi, proof, pool, device, batch=batch, block=max_len)
        ids, mask, _ = out
        loss = loss_fn(model, ids, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = loss.item()
        if (step + 1) % 200 == 0:
            print(f"    [{'PROOF' if proof else 'ANSWER'} seed{seed}] step {step+1}/{steps} "
                  f"loss {last:.4f}", flush=True)
    train_acc = evaluate(model, random.Random(seed + 1000), stoi, itos, proof, device,
                         pool, n_eval=240)
    held = evaluate(model, random.Random(seed + 7000), stoi, itos, proof, device,
                    heldout_queries(), n_eval=160)
    return model, train_acc, held, last


def show_kb_summary():
    nb = len(LABELS)
    print(f"REAL MINI-KB | entities/categories(labels)={nb} | "
          f"isa={len(ISA)} part_of={len(PART_OF)} has_prop={len(HAS_PROP)} base facts "
          f"({len(EDB)} total) | closure={len(CLOSURE)} derived-incl")
    print("  example REAL facts:")
    for s in ["a robin is a bird .", "a bird is an animal .", "an animal can move .",
              "a wing is part of a bird .", "a fish can swim .", "a rose is a flower ."]:
        print("    -", s)
    print(f"  query bank sizes: " + ", ".join(f"{qt}={len(QBANK[qt])}" for qt in QTYPES))
    print(f"  HELD-OUT composition pairs (subject,derived-query) reserved for lead: {len(HELDOUT)}")
    ex = list(HELDOUT)[:5]
    for (pred, (a, b)) in ex:
        print(f"    - held: {pred}({a},{b})  [multi-hop derived; prompt states only base facts]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=260)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device)
    _nt = os.environ.get("RLN_THREADS")
    torch.set_num_threads(int(_nt) if _nt else max(1, os.cpu_count() or 1))
    vocab = build_vocab()
    itos, stoi = vocab

    show_kb_summary()
    print(f"  vocab={len(itos)} | model dim={args.dim} layers={args.layers} heads={args.heads} "
          f"batch={args.batch} | device={device} | steps={args.steps}")

    # show a held-out composition example: prompt (base only) + proof chain + asserted no-verbatim
    rng = random.Random(2)
    held = heldout_queries()
    inh = [it for it in held if it[0] == "INHERIT"]
    demo = rng.choice(inh) if inh else rng.choice(held)
    facts, q, gold, qtype = make_example(rng, demo)
    assert_answer_not_verbatim(facts, q)
    print(f"\n  HELD-OUT composition demo ({qtype}, gold={'yes' if gold else 'no'}):")
    print("    prompt (base facts only):", " ".join(render_prompt(random.Random(3), facts, q)))
    rows = render_proof_chain(facts, q, gold)
    print("    proof chain:", "  ".join(" ".join(r) for r in rows),
          f" answer {'yes' if gold else 'no'} .")
    print(f"    derived query {q} NOT stated verbatim in prompt: OK")

    if args.sanity:
        print("\n=== SANITY GATE: PROOF must LEARN TRAIN (>0.85 overall) ===")
        model, tr, held_acc, ls = train_one(True, 0, args.steps, args.lr, device, vocab,
                                             dim=args.dim, layers=args.layers, heads=args.heads,
                                             batch=args.batch, max_len=args.max_len)
        print(f"PROOF seed0 TRAIN   overall {tr['OVERALL']:.3f} | per-q " +
              " ".join(f"{k} {tr[k]:.2f}" for k in QTYPES))
        print(f"PROOF seed0 HELDOUT overall {held_acc['OVERALL']:.3f} | per-q " +
              " ".join(f"{k} {held_acc[k]:.2f}" for k in QTYPES) +
              "   [for-reference; lead to validate]")
        gate = tr['OVERALL'] > 0.85
        print(f"GATE {'PASS' if gate else 'FAIL'} (train overall {tr['OVERALL']:.3f}) @ {args.steps} steps")

        # inspect 2-3 generated proof chains on TRAIN items to confirm real-fact reasoning
        print("\n  sample generated proof chains (greedy decode, TRAIN items):")
        ins_rng = random.Random(999)
        pool = train_queries()
        for _ in range(3):
            item = ins_rng.choice(pool)
            f2, q2, g2, qt2 = make_example(ins_rng, item)
            prompt = render_prompt(ins_rng, f2, q2)
            _show_decode(model, prompt, q2, f2, stoi, itos, device, qt2, g2)
        return

    print("(no --sanity: nothing else to do; pass --sanity to run the gate)")


@torch.no_grad()
def _show_decode(model, prompt_toks, q, facts, stoi, itos, device, qtype, gold, block=260, max_new=90):
    """Decode and print the model's proof chain (for human inspection of real-fact reasoning)."""
    model.eval()
    ids = [stoi[t] for t in prompt_toks]
    answer_id = stoi["answer"]
    out_toks = []
    saw_answer = False
    for _ in range(max_new):
        ctx = torch.tensor([ids[-block:]], device=device)
        nxt = int(model(ctx)[0, -1].argmax())
        ids.append(nxt)
        out_toks.append(itos[nxt])
        if nxt == answer_id:
            saw_answer = True
        elif saw_answer:
            break
    model.train()
    pred = greedy_answer(model, prompt_toks, q, facts, stoi, itos, True, device)
    print(f"    [{qtype}] q={q[0]}{q[1]} gold={'yes' if gold else 'no'} pred={pred}")
    print(f"       chain: {' '.join(out_toks)}")


if __name__ == "__main__":
    main()
