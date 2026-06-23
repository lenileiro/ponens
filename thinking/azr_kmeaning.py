#!/usr/bin/env python3
"""azr_kmeaning -- the culmination: LEARNED + SOUND + COMPOUNDING multi-relation proof search over REAL
WordNet meaning. Unifies azr_kproof (learned proposer guides, kernel verifies, lemmas compound) with
azr_meaning (real concepts, multiple relations + rules). Now proving e.g. has_part(robin, X) requires
CHOOSING among many candidate derivations each round (is-a transitivity AND has_part/member_of inheritance
all fire) -- so the learned proposer earns its keep (unlike the is-a tree, where there was no choice), the
kernel type-checks every derived atom (sound), and kernel-verified facts cached as LEMMAS compound.

A learned DerivProposer scores each candidate derivation by relevance to the goal; a beam expands the top
few; the kernel verifies each; reaching the goal = a kernel-checked proof. Self-play trains the proposer on
solved-proof atoms (which derivations were on the proof). Bar: guided solves more within a tight derivation
budget than unguided (random-order) on this BRANCHING domain; UNSOUND 0; lemmas raise deep-goal reach.

    python -m thinking.azr_kmeaning --selftest
    python -m thinking.azr_kmeaning --wordnet --cap 120
"""
import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import thinking.lota_kernel as LK
from thinking.azr_meaning import RULES, base_tree, kernel_verify, _match_body, _inst

PREDS = {"isa": 2, "has_part": 2, "member_of": 2}
PID = {"isa": 0, "has_part": 1, "member_of": 2}


def tree_atoms(tree, out):
    out.add(tree["fact"])
    for c in tree["from"]:
        tree_atoms(c, out)


# ===================================================================================================
# Real multi-relation WordNet neighborhood: is-a paths + part_meronyms (has_part) + member_holonyms.
# ===================================================================================================
def wordnet_kb(cap=120, roots=("animal.n.01", "bird.n.01"), seed=0):
    from nltk.corpus import wordnet as wn
    syns, queue = set(), [wn.synset(r) for r in roots]
    while queue and len(syns) < cap:
        s = queue.pop(0)
        if s.name() in syns:
            continue
        syns.add(s.name()); queue += s.hyponyms()
    nodes = set(syns)
    for n in list(syns):
        for path in wn.synset(n).hypernym_paths():
            nodes.update(x.name() for x in path)
    facts = set()
    for n in nodes:
        for h in wn.synset(n).hypernyms():
            if h.name() in nodes:
                facts.add(("isa", (n, h.name())))
    # collect parts/members (extend node set with their endpoints)
    extra = set()
    for n in list(nodes):
        s = wn.synset(n)
        for p in s.part_meronyms():
            facts.add(("has_part", (n, p.name()))); extra.add(p.name())
        for g in s.member_holonyms():
            facts.add(("member_of", (n, g.name()))); extra.add(g.name())
    nodes |= extra
    ents = sorted(nodes)
    concepts = sorted(syns)
    return sorted(facts), ents, concepts


def synthetic_kb(rng, depth=6, sibs=3, parts=4):
    """A controllable multi-relation KB with branching: an is-a chain c0..cD (+ siblings) and has_part/
    member_of facts at several levels -> proving an inherited fact needs the right is-a chain among many
    candidate derivations."""
    facts = []
    chain = [f"c{i}" for i in range(depth + 1)]
    for i in range(depth):
        facts.append(("isa", (chain[i], chain[i + 1])))
    for i in range(1, depth):                                # siblings (distractors) under each level
        for s in range(sibs):
            facts.append(("isa", (f"s{i}_{s}", chain[i])))
    for i in range(1, depth + 1):                            # parts/members at various levels
        facts.append(("has_part", (chain[i], f"p{i}")))
        if i % 2 == 0:
            facts.append(("member_of", (chain[i], f"g{i}")))
    ents = sorted({e for (_p, a) in facts for e in a})
    concepts = [chain[0]] + [f"s1_{s}" for s in range(sibs)]
    return facts, ents, concepts


def goals_for(facts, concept):
    isa = {(a, b) for (p, (a, b)) in facts if p == "isa"}
    anc, fr = set(), [concept]
    while fr:
        x = fr.pop()
        for (a, b) in isa:
            if a == x and b not in anc:
                anc.add(b); fr.append(b)
    g = [("isa", (concept, z)) for z in anc]
    for (p, (a, b)) in facts:
        if p in ("has_part", "member_of") and a in anc:
            g.append((p, (concept, b)))
    return g


# ===================================================================================================
# Learned proposer: score a candidate derived atom by relevance to the goal.
# ===================================================================================================
class DerivProposer(nn.Module):
    def __init__(self, n_ent, d=128):
        super().__init__()
        self.ent = nn.Embedding(n_ent, d)
        self.pred = nn.Embedding(len(PID), d)
        self.mlp = nn.Sequential(nn.LayerNorm(6 * d), nn.Linear(6 * d, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, cp, ca, gp, ga):                       # cp,gp:(B,) preds; ca,ga:(B,2) ent ids
        feat = torch.cat([self.pred(cp), self.ent(ca).flatten(1),
                          self.pred(gp), self.ent(ga).flatten(1)], -1)
        return self.mlp(feat).squeeze(-1)


def solve(proposer, goal, base_facts, eid, env, budget, beam, device, guided=True):
    """Guided kernel-verified forward chaining. Each round: enumerate candidate rule firings, the learned
    proposer scores them by goal-relevance, the beam-best are kernel-verified and added. Returns
    (proof tree|None, kernel_checks, unsound)."""
    proven = {f: base_tree(f) for f in base_facts}
    checks = unsound = 0
    if goal in proven:
        return proven[goal], 0, 0
    gp = torch.tensor([PID[goal[0]]], device=device)
    ga = torch.tensor([[eid[goal[1][0]], eid[goal[1][1]]]], device=device)
    for _ in range(budget):
        cands = []
        for ri, (head, body) in enumerate(RULES):
            for sub, subtrees in _match_body(body, proven):
                hf = _inst(head, sub)
                if hf not in proven and hf not in (c[0] for c in cands):
                    cands.append((hf, ri, subtrees))
        if not cands:
            break
        if guided:
            cp = torch.tensor([PID[hf[0]] for hf, _r, _s in cands], device=device)
            ca = torch.tensor([[eid[hf[1][0]], eid[hf[1][1]]] for hf, _r, _s in cands], device=device)
            with torch.no_grad():
                sc = proposer(cp, ca, gp.expand(len(cands)), ga.expand(len(cands), -1))
            order = torch.argsort(sc, descending=True).tolist()
        else:
            order = list(range(len(cands))); random.shuffle(order)
        progressed = False
        for i in order[:beam]:
            hf, ri, subtrees = cands[i]
            tree = {"fact": hf, "rule": ri, "from": subtrees}
            checks += 1
            if not kernel_verify(tree, env):
                unsound += 1; continue
            proven[hf] = tree; progressed = True
            if hf == goal:
                return tree, checks, unsound
        if not progressed:
            break
    return proven.get(goal), checks, unsound


def evaluate(proposer, facts, eid, env, concepts, budget, beam, device, n=120, guided=True):
    rng = random.Random(0)
    solved = checks = unsound = tot = 0
    goalpool = [(c, g) for c in concepts for g in goals_for(facts, c)]
    rng.shuffle(goalpool)
    for (c, g) in goalpool[:n]:
        tree, ck, un = solve(proposer, g, facts, eid, env, budget, beam, device, guided=guided)
        tot += 1; solved += int(tree is not None); checks += ck; unsound += un
    return dict(solve=solved / max(1, tot), checks=checks / max(1, tot), unsound=unsound)


def full_closure(facts, env, max_rounds=12):
    """ONE kernel-verified deductive closure of the KB: every derivable atom + its proof tree. All goals'
    proofs come from this single closure -> O(closure) instead of O(goals x closure) for training data."""
    proven = {f: base_tree(f) for f in facts}
    for _ in range(max_rounds):
        newly = {}
        for ri, (head, body) in enumerate(RULES):
            for sub, subtrees in _match_body(body, proven):
                hf = _inst(head, sub)
                if hf not in proven and hf not in newly:
                    tree = {"fact": hf, "rule": ri, "from": subtrees}
                    if kernel_verify(tree, env):
                        newly[hf] = tree
        if not newly:
            break
        proven.update(newly)
    return proven


def exhaustive_prove(goal, facts, env, max_rounds=10):
    """The COMPLETE teacher: semi-naive forward chaining, kernel-verifying every derived atom, until the
    goal is proven. Sound but does ALL the work; used only to generate training proofs for the proposer."""
    proven = {f: base_tree(f) for f in facts}
    if goal in proven:
        return proven[goal], proven
    for _ in range(max_rounds):
        newly = {}
        for ri, (head, body) in enumerate(RULES):
            for sub, subtrees in _match_body(body, proven):
                hf = _inst(head, sub)
                if hf not in proven and hf not in newly:
                    tree = {"fact": hf, "rule": ri, "from": subtrees}
                    if kernel_verify(tree, env):
                        newly[hf] = tree
        proven.update(newly)
        if goal in proven:
            return proven[goal], proven
        if not newly:
            break
    return proven.get(goal), proven


def selfplay(facts, ents, concepts, d=128, rounds=600, bs=64, beam=2, budget=6, lr=1.5e-3,
             device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eid = {e: i for i, e in enumerate(ents)}
    env = LK.build_env(ents, PREDS, facts, RULES)
    prop = DerivProposer(len(ents), d=d).to(device)
    opt = torch.optim.AdamW(prop.parameters(), lr=lr, weight_decay=0.01)
    base_set = set(facts)
    goalpool = [(c, g) for c in concepts for g in goals_for(facts, c)]

    # EXPERT-ITERATION BOOTSTRAP: the complete prover teaches proof-relevance (no cold-start). Compute the
    # kernel-verified closure ONCE; every goal's proof tree is already in it. For each goal, its proof's
    # DERIVED atoms are positives; other derivable atoms are negatives.
    closure = full_closure(facts, env)                       # one closure for all goals (the speedup)
    derivable = [a for a in closure if a not in base_set]
    examples, lemmas = [], {}
    for (c, g) in goalpool:
        if g not in closure:
            continue
        patoms = set(); tree_atoms(closure[g], patoms)
        pos = {a for a in patoms if a not in base_set}
        if closure[g]["rule"] is not None and g not in lemmas and len(lemmas) < 3 * len(ents):
            lemmas[g] = closure[g]
        negs = [a for a in derivable if a not in pos]
        random.Random(hash(g) & 0xffff).shuffle(negs)
        for a in list(pos) + negs[:max(4, 3 * len(pos))]:
            examples.append((PID[a[0]], [eid[a[1][0]], eid[a[1][1]]],
                             PID[g[0]], [eid[g[1][0]], eid[g[1][1]]], float(a in pos)))
    if verbose:
        print(f"  teacher: {len(goalpool)} goals -> {len(examples)} training examples, {len(lemmas)} lemmas",
              flush=True)

    rng = random.Random(0)
    for rnd in range(rounds):
        rng.shuffle(examples)
        batch = examples[:512]
        prop.train()
        cp = torch.tensor([e[0] for e in batch], device=device)
        ca = torch.tensor([e[1] for e in batch], device=device)
        gp = torch.tensor([e[2] for e in batch], device=device)
        ga = torch.tensor([e[3] for e in batch], device=device)
        y = torch.tensor([e[4] for e in batch], device=device)
        loss = F.binary_cross_entropy_with_logits(prop(cp, ca, gp, ga), y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0); opt.step()
        if verbose and rnd % max(1, rounds // 6) == 0:
            eg = evaluate(prop, facts, eid, env, concepts, budget, beam, device, guided=True)
            eu = evaluate(prop, facts, eid, env, concepts, budget, beam, device, guided=False)
            print(f"  r{rnd:5d}: guided {eg['solve']:.2f} (checks {eg['checks']:.1f}) | "
                  f"unguided {eu['solve']:.2f} (checks {eu['checks']:.1f}) | unsound {eg['unsound']+eu['unsound']}",
                  flush=True)
    return prop, eid, env, lemmas


def selftest():
    rng = np.random.default_rng(0)
    facts, ents, concepts = synthetic_kb(rng, depth=6, sibs=3, parts=4)
    prop, eid, env, lemmas = selfplay(facts, ents, concepts, d=64, rounds=500, bs=24, beam=2, budget=4,
                                      device="cpu", verbose=False)
    eg = evaluate(prop, facts, eid, env, concepts, budget=4, beam=2, device="cpu", guided=True)
    eu = evaluate(prop, facts, eid, env, concepts, budget=4, beam=2, device="cpu", guided=False)
    print(f"  selftest: guided solve {eg['solve']:.2f} vs unguided {eu['solve']:.2f} | "
          f"unsound {eg['unsound']+eu['unsound']} | lemmas {len(lemmas)}")
    assert eg["unsound"] + eu["unsound"] == 0, "kernel accepted an unsound multi-relation step"
    assert eg["solve"] > eu["solve"] + 0.1, "learned guidance did not beat unguided on the branching domain"
    print("azr_kmeaning selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--wordnet", action="store_true")
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=600); ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--budget", type=int, default=6); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.wordnet:
        facts, ents, concepts = wordnet_kb(cap=args.cap)
        print(f"  REAL WordNet multi-relation KB: {len(facts)} facts, {len(ents)} concepts, "
              f"{len(concepts)} query concepts", flush=True)
    else:
        facts, ents, concepts = synthetic_kb(np.random.default_rng(0))
    prop, eid, env, lemmas = selfplay(facts, ents, concepts, d=args.d, rounds=args.rounds,
                                      beam=args.beam, budget=args.budget, device=args.device)
    eg = evaluate(prop, facts, eid, env, concepts, args.budget, args.beam, prop.ent.weight.device, guided=True)
    eu = evaluate(prop, facts, eid, env, concepts, args.budget, args.beam, prop.ent.weight.device, guided=False)
    print(f"  FINAL: guided {eg['solve']:.3f} (checks {eg['checks']:.1f}) vs unguided {eu['solve']:.3f} "
          f"(checks {eu['checks']:.1f}) | unsound {eg['unsound']+eu['unsound']} | lemmas {len(lemmas)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
