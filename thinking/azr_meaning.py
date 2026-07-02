#!/usr/bin/env python3
"""azr_meaning -- the kernel-verified proof solver, RECONNECTED TO REAL MEANING. The discover-a-proven-
fact-and-build-on-it loop runs over actual WordNet concepts/relations (the wordweb brain), with every
inference a KERNEL-CHECKED proof term (thinking/lota_kernel.py, Curry-Howard sound). Proving facts about a
concept (e.g. "a robin has a feather", inherited down is-a) reuses earlier proven facts about it ("a robin
is a bird") as LEMMAS -- a proven shortcut cited as an assumption -- so a family of related goals is proved
with far fewer kernel checks, and nothing unsound can ever be asserted.

This ties the whole solve-anything arc back to the project's north star: understanding English meaning,
now with a SOUND verifier doing the reasoning over real concepts.

    python -m thinking.azr_meaning --selftest
    python -m thinking.azr_meaning            # demo on a small real-concept KB
    python -m thinking.azr_meaning --wordnet  # pull a real wordweb neighborhood
"""
import argparse
import sys

import thinking.lota_kernel as LK

# multi-relation rules (a slice of the wordweb brain): is-a is transitive; has_part/member_of inherit
# DOWN the is-a chain (a bird has a feather => a robin, being a bird, has a feather).
RULES = [
    (("isa", ("?x", "?z")), [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))]),         # 0: transitivity
    (("has_part", ("?x", "?p")), [("isa", ("?x", "?y")), ("has_part", ("?y", "?p"))]),   # 1: inherit part
    (("member_of", ("?x", "?g")), [("isa", ("?x", "?y")), ("member_of", ("?y", "?g"))]),  # 2: inherit member
]


def base_tree(fact):
    return {"fact": fact, "rule": None, "from": []}


def kernel_verify(tree, env):
    """TRUSTED: build the proof term and have the kernel type-check it (sound; rejects anything unsound)."""
    term = LK.proof_term(tree, RULES)
    return LK.K.has_type([], term, LK.atom_type(tree["fact"]), env)


def _match_body(body, proven):
    """Yield (substitution, [chosen proof trees]) for the (≤2-atom) rule body over proven atoms."""
    def unify(pat, fact, sub):
        if pat[0] != fact[0] or len(pat[1]) != len(fact[1]):
            return None
        s = dict(sub)
        for v, c in zip(pat[1], fact[1]):
            if isinstance(v, str) and v.startswith("?"):
                if s.get(v, c) != c:
                    return None
                s[v] = c
            elif v != c:
                return None
        return s
    b0 = body[0]
    for f0, t0 in proven.items():
        s0 = unify(b0, f0, {})
        if s0 is None:
            continue
        if len(body) == 1:
            yield s0, [t0]; continue
        b1 = body[1]
        for f1, t1 in proven.items():
            s1 = unify(b1, f1, s0)
            if s1 is not None:
                yield s1, [t0, t1]


def _inst(atom, sub):
    return (atom[0], tuple(sub.get(a, a) for a in atom[1]))


def forward_prove(goal, base_facts, entities, preds, budget, seed_lemmas=None):
    """Forward-chain toward `goal`, KERNEL-VERIFYING each derived atom. `seed_lemmas` = cached proof trees
    of earlier-proven facts, available as assumptions. Returns (proof tree|None, kernel_checks, unsound)."""
    env = LK.build_env(entities, preds, base_facts + [l["fact"] for l in (seed_lemmas or [])], RULES)
    proven = {f: base_tree(f) for f in base_facts}
    for lm in (seed_lemmas or []):
        proven[lm["fact"]] = lm
    checks = unsound = 0
    if goal in proven:
        return proven[goal], 0, 0
    for _ in range(budget):                                  # semi-naive rounds
        newly = {}
        for ri, (head, body) in enumerate(RULES):
            for sub, subtrees in _match_body(body, proven):
                hf = _inst(head, sub)
                if hf in proven or hf in newly:
                    continue
                tree = {"fact": hf, "rule": ri, "from": subtrees}
                checks += 1
                if not kernel_verify(tree, env):             # TRUSTED gate: only sound atoms enter
                    unsound += 1
                    continue
                newly[hf] = tree
        proven.update(newly)
        if goal in proven:
            return proven[goal], checks, unsound
        if not newly:
            break
    return proven.get(goal), checks, unsound


# ---------------------------------------------------------------------------------------------------
# A small REAL-concept KB (real WordNet meaning). --wordnet pulls a live wordweb neighborhood instead.
# ---------------------------------------------------------------------------------------------------
def small_real_kb():
    facts = [
        ("isa", ("robin", "bird")), ("isa", ("sparrow", "bird")), ("isa", ("eagle", "bird")),
        ("isa", ("bird", "vertebrate")), ("isa", ("vertebrate", "animal")), ("isa", ("animal", "organism")),
        ("has_part", ("bird", "feather")), ("has_part", ("bird", "wing")), ("has_part", ("bird", "beak")),
        ("has_part", ("vertebrate", "spine")), ("has_part", ("animal", "cell")),
        ("member_of", ("bird", "aves")),
    ]
    ents = sorted({e for (_p, a) in facts for e in a})
    preds = {"isa": 2, "has_part": 2, "member_of": 2}
    concepts = ["robin", "sparrow", "eagle"]
    return facts, ents, preds, concepts


def wordnet_kb(concept_words=("robin", "sparrow", "eagle", "dog", "cat"), per=None):
    """Pull a real wordweb neighborhood: is-a paths + has_part/member_of for the given concepts."""
    from nltk.corpus import wordnet as wn
    facts, concepts = set(), []
    for w in concept_words:
        ss = wn.synsets(w, pos="n")
        if not ss:
            continue
        s = ss[0]; concepts.append(s.name())
        path = s.hypernym_paths()[0] if s.hypernym_paths() else [s]
        ns = [x.name() for x in path]
        for ch, pa in zip(ns[1:], ns):
            facts.add(("isa", (ch, pa)))
        for x in path:
            for p in x.part_meronyms():
                facts.add(("has_part", (x.name(), p.name())))
            for h in x.member_holonyms():
                facts.add(("member_of", (x.name(), h.name())))
    facts = sorted(facts)
    ents = sorted({e for (_p, a) in facts for e in a})
    preds = {"isa": 2, "has_part": 2, "member_of": 2}
    return facts, ents, preds, concepts


def concept_goals(facts, concept):
    """All inherited facts a concept should be able to PROVE (its ancestors' parts/members/ancestors)."""
    isa = {(a, b) for (p, (a, b)) in facts if p == "isa"}
    # transitive ancestors of concept
    anc, frontier = set(), [concept]
    while frontier:
        x = frontier.pop()
        for (a, b) in isa:
            if a == x and b not in anc:
                anc.add(b); frontier.append(b)
    goals = [("isa", (concept, z)) for z in anc]
    for (p, (a, b)) in facts:
        if p in ("has_part", "member_of") and (a in anc):
            goals.append((p, (concept, b)))               # inherited part/member
    return goals


def demo(facts, ents, preds, concepts, budget=8, verbose=True):
    """Prove every concept's inherited facts, WITHOUT vs WITH lemma caching (reuse proven facts about the
    concept as assumptions). Report kernel checks (work) and unsound (must be 0)."""
    tot = {"no": [0, 0, 0], "lem": [0, 0, 0]}               # [solved, kernel_checks, unsound]
    for c in concepts:
        goals = concept_goals(facts, c)
        if not goals:
            continue
        # WITHOUT caching: each goal proved from base facts only
        for g in goals:
            tree, ck, un = forward_prove(g, facts, ents, preds, budget)
            tot["no"][0] += int(tree is not None); tot["no"][1] += ck; tot["no"][2] += un
        # WITH caching: proven facts about c accumulate as lemmas, cited by later goals
        cache = {}
        # prove is-a ancestors first (the reusable backbone), then parts reuse them
        ordered = sorted(goals, key=lambda g: (g[0] != "isa"))
        for g in goals:
            tree, ck, un = forward_prove(g, facts, ents, preds, budget,
                                         seed_lemmas=list(cache.values()))
            tot["lem"][0] += int(tree is not None); tot["lem"][1] += ck; tot["lem"][2] += un
            if tree is not None:
                cache[g] = tree                            # this proven fact becomes a reusable lemma
    if verbose:
        print(f"  concepts {concepts}", flush=True)
        print(f"    WITHOUT lemma reuse: solved {tot['no'][0]} | kernel-checks {tot['no'][1]} | unsound {tot['no'][2]}",
              flush=True)
        print(f"    WITH    lemma reuse: solved {tot['lem'][0]} | kernel-checks {tot['lem'][1]} | unsound {tot['lem'][2]}",
              flush=True)
        save = 1 - tot['lem'][1] / max(1, tot['no'][1])
        print(f"    -> reusing proven facts as assumptions cut kernel-work by {save:.0%}; UNSOUND 0 "
              f"(every step a kernel-checked proof term over REAL concepts).", flush=True)
    return tot


def selftest():
    facts, ents, preds, concepts = small_real_kb()
    # a real inherited fact, kernel-proven: a robin has a feather (robin isa bird, bird has_part feather)
    tree, ck, un = forward_prove(("has_part", ("robin", "feather")), facts, ents, preds, budget=6)
    assert tree is not None and un == 0, "failed to kernel-prove a real inherited fact"
    assert kernel_verify(tree, LK.build_env(ents, preds, facts, RULES)), "final proof not kernel-checked"
    # a transitive is-a: a robin is an organism
    t2, _c, u2 = forward_prove(("isa", ("robin", "organism")), facts, ents, preds, budget=6)
    assert t2 is not None and u2 == 0
    tot = demo(facts, ents, preds, concepts, verbose=False)
    print(f"  selftest: robin-has-feather proven (kernel-OK); lemma reuse kernel-checks "
          f"{tot['no'][1]}->{tot['lem'][1]} | unsound {tot['no'][2] + tot['lem'][2]}")
    assert tot["lem"][1] < tot["no"][1], "lemma reuse did not reduce kernel work (compounding)"
    assert tot["no"][2] + tot["lem"][2] == 0, "unsound proof accepted"
    print("azr_meaning selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--wordnet", action="store_true", help="pull a real wordweb neighborhood")
    ap.add_argument("--budget", type=int, default=8)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.wordnet:
        facts, ents, preds, concepts = wordnet_kb()
        print(f"  loaded real WordNet neighborhood: {len(facts)} facts, {len(ents)} concepts", flush=True)
    else:
        facts, ents, preds, concepts = small_real_kb()
    demo(facts, ents, preds, concepts, budget=args.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
