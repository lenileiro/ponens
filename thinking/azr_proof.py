#!/usr/bin/env python3
"""azr_proof -- WIRE THE SOLVER TO THE REAL KERNEL. The iterate-execute-residual idea, but the domain is
the kernel's native one (relational proof search) and verification is a KERNEL-CHECKED PROOF TERM
(Curry-Howard, thinking/lota_kernel.py) -- not I/O testing. A "program" is a PROOF; each step is one
inference (transitivity) whose proof term the kernel type-checks before it is accepted. Sound by
construction: an unsound step cannot enter the proven set.

This is the "discover a proven fact / shortcut and build on it as an assumption" loop, literally:
  - build a proof of the goal one KERNEL-VERIFIED step at a time (compose proven atoms via transitivity);
  - a step budget caps proof length (composition depth);
  - LEMMAS = kernel-verified long-range facts cached from past proofs; seeding them shortens future proofs
    so goals that EXCEED the budget without lemmas become provable WITH them = COMPOUNDING.

    python -m thinking.azr_proof --selftest
    python -m thinking.azr_proof --hubs 8 --budget 3
"""
import argparse
import sys
from collections import deque

import thinking.lota_kernel as LK

REL = "r"                                                    # one transitive relation (isa-like)
RULES = [((REL, ("?x", "?z")), [(REL, ("?x", "?y")), (REL, ("?y", "?z"))])]   # transitivity, rule 0


def base_tree(fact):
    return {"fact": fact, "rule": None, "from": []}


def compose_tree(t_xy, t_yz):
    """Proof tree for r(x,z) from proofs of r(x,y) and r(y,z) via the transitivity rule (rule 0)."""
    (_, (x, _y1)), (_, (_y2, z)) = t_xy["fact"], t_yz["fact"]
    return {"fact": (REL, (x, z)), "rule": 0, "from": [t_xy, t_yz]}


def kernel_verify(tree, env):
    """The TRUSTED check: build the proof term and have the kernel type-check it against the atom type."""
    term = LK.proof_term(tree, RULES)
    return LK.K.has_type([], term, LK.atom_type(tree["fact"]), env)


def prove(goal, base_facts, entities, budget, lemmas=None, env=None):
    """Prove r(src,tgt) by composing proven atoms into a chain, KERNEL-VERIFYING each transitivity step.
    `lemmas` = extra kernel-verified atoms (long-range edges) seeded into the proven set -> shorter chains.
    Returns (proof_tree or None, n_steps, all_steps_verified)."""
    src, tgt = goal[1]
    if env is None:
        env = LK.build_env(entities, {REL: 2}, base_facts + [l["fact"] for l in (lemmas or [])], RULES)
    proven = {f[1]: base_tree(f) for f in base_facts}        # (x,y) -> proof tree (base facts = axioms)
    for lm in (lemmas or []):                                # seed kernel-verified lemmas (long edges)
        proven[lm["fact"][1]] = lm
    # shortest chain src->tgt over proven-atom edges (each edge already kernel-proven)
    adj = {}
    for (x, y) in proven:
        adj.setdefault(x, []).append(y)
    prev, dq = {src: None}, deque([src])
    while dq:
        u = dq.popleft()
        if u == tgt:
            break
        for v in adj.get(u, []):
            if v not in prev:
                prev[v] = u; dq.append(v)
    if tgt not in prev:
        return None, 0, True                                 # unreachable with current proven edges
    chain = []                                               # [src, n1, ..., tgt]
    n = tgt
    while n is not None:
        chain.append(n); n = prev[n]
    chain.reverse()
    steps = len(chain) - 2                                   # transitivity applications to fold the chain
    if steps > budget:
        return None, steps, True                             # too deep for the budget (no lemmas help)
    # left-fold the chain into r(src,tgt), kernel-verifying each composition
    acc = proven[(chain[0], chain[1])]
    verified = True
    for i in range(1, len(chain) - 1):
        nxt = proven[(chain[i], chain[i + 1])]
        acc = compose_tree(acc, nxt)
        verified = verified and kernel_verify(acc, env)
        if not verified:
            return None, i, False
    return acc, max(0, steps), verified


# ===================================================================================================
# A structured KB: a fixed HUB BACKBONE (recurs across tasks) + per-task peripheral edges. Backbone
# segments are the reusable LEMMAS whose discovery makes deep goals provable within a small budget.
# ===================================================================================================
def backbone(m):
    hubs = [f"h{i}" for i in range(m)]
    edges = [(REL, (hubs[i], hubs[i + 1])) for i in range(m - 1)]    # h0->h1->...->h_{m-1}
    return hubs, edges


def task(rng, hubs, backbone_edges, i=None, j=None):
    """src -> h_i -> ...(backbone)... -> h_j -> tgt. Goal: prove r(src,tgt)."""
    m = len(hubs)
    if i is None:
        i = int(rng.integers(0, m - 1)); j = int(rng.integers(i + 1, m))
    src, tgt = "src", "tgt"
    facts = list(backbone_edges) + [(REL, (src, hubs[i])), (REL, (hubs[j], tgt))]
    ents = hubs + [src, tgt]
    return (REL, (src, tgt)), facts, ents, (i, j)


def derive_lemma(hubs, backbone_edges, i, j):
    """Kernel-verified backbone segment r(h_i, h_j) -- a reusable lemma (proven once, cited thereafter)."""
    ents = hubs + ["src", "tgt"]
    env = LK.build_env(ents, {REL: 2}, backbone_edges, RULES)
    proven = {f[1]: base_tree(f) for f in backbone_edges}
    acc = proven[(hubs[i], hubs[i + 1])]
    for k in range(i + 1, j):
        acc = compose_tree(acc, proven[(hubs[k], hubs[k + 1])])
        assert kernel_verify(acc, env), "lemma derivation failed kernel check"
    return acc                                               # proof tree for r(h_i, h_j)


def selftest():
    """CPU-only: (1) a multi-hop goal is solved with a KERNEL-VERIFIED proof; (2) a goal that EXCEEDS the
    step budget without lemmas becomes provable WITH a kernel-verified backbone lemma = COMPOUNDING;
    (3) zero unsound steps ever accepted."""
    import numpy as np
    rng = np.random.default_rng(0)
    hubs, bb = backbone(7)
    # (1) plain multi-hop proof, generous budget
    goal, facts, ents, (i, j) = task(rng, hubs, bb, i=1, j=4)         # src->h1->h2->h3->h4->tgt (4 steps)
    tree, steps, ok = prove(goal, facts, ents, budget=10)
    assert tree is not None and ok, "failed to prove a reachable multi-hop goal"
    assert kernel_verify(tree, LK.build_env(ents, {REL: 2}, facts, RULES)), "final proof not kernel-checked"
    # (2) COMPOUNDING: budget=2 is too small for the 4-step proof...
    no_lemma, st_nl, _ = prove(goal, facts, ents, budget=2)
    # ...but a kernel-verified backbone lemma r(h1,h4) collapses it to src->h1, [h1->h4], h4->tgt = 2 steps
    lemma = derive_lemma(hubs, bb, 1, 4)
    with_lemma, st_wl, ok2 = prove(goal, facts, ents, budget=2, lemmas=[lemma])
    print(f"  selftest: plain proof steps={steps} kernel-OK={ok} | budget2 no-lemma solved={no_lemma is not None}"
          f" | budget2 +lemma solved={with_lemma is not None} (steps {st_wl}, kernel-OK={ok2})")
    assert no_lemma is None, "control failed: deep goal should NOT fit budget 2 without a lemma"
    assert with_lemma is not None and ok2, "lemma did not enable the deep goal within budget (compounding)"
    print("azr_proof selftest OK")
    return 0


def run(hubs_n=8, budget=3, n_tasks=200, verbose=True):
    """Solve-rate at increasing backbone span, within a fixed step budget, WITHOUT vs WITH cached lemmas.
    Every accepted step is a kernel-checked proof term (sound by construction)."""
    import numpy as np
    rng = np.random.default_rng(0)
    hubs, bb = backbone(hubs_n)
    # lemma cache = all kernel-verified backbone segments (the recurring structure)
    cache = [derive_lemma(hubs, bb, i, j) for i in range(hubs_n - 1) for j in range(i + 1, hubs_n)]
    no_lem = {"solved": 0, "unsound": 0}
    with_lem = {"solved": 0, "unsound": 0}
    spans = []
    for _ in range(n_tasks):
        goal, facts, ents, (i, j) = task(rng, hubs, bb)
        spans.append(j - i)
        t0, _s, ok0 = prove(goal, facts, ents, budget=budget)
        t1, _s1, ok1 = prove(goal, facts, ents, budget=budget, lemmas=cache)
        no_lem["solved"] += int(t0 is not None); no_lem["unsound"] += int(not ok0)
        with_lem["solved"] += int(t1 is not None); with_lem["unsound"] += int(not ok1)
    if verbose:
        print(f"  backbone {hubs_n} hubs, budget {budget} steps, {n_tasks} tasks "
              f"(mean span {sum(spans)/len(spans):.1f}):", flush=True)
        print(f"    WITHOUT lemmas: solved {no_lem['solved']/n_tasks:.3f} | unsound {no_lem['unsound']}", flush=True)
        print(f"    WITH lemmas   : solved {with_lem['solved']/n_tasks:.3f} | unsound {with_lem['unsound']}", flush=True)
        print("    -> lemmas (kernel-verified, cached) let deep goals fit the budget = COMPOUNDING; "
              "0 unsound = every step is a kernel-checked proof term.", flush=True)
    return dict(without=no_lem["solved"] / n_tasks, with_lemmas=with_lem["solved"] / n_tasks,
                unsound=no_lem["unsound"] + with_lem["unsound"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hubs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=200)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    run(hubs_n=args.hubs, budget=args.budget, n_tasks=args.tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
