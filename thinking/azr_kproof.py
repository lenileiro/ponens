#!/usr/bin/env python3
"""azr_kproof -- LEARNED + SOUND: the learned StepProposer (azr_iter) guides the KERNEL-VERIFIED proof
search (azr_proof). The proposer is a navigation policy that learns WHICH inference to make next; the
kernel type-checks the proof term of EVERY step (Curry-Howard sound) before it is accepted. So search is
learned and fast, while correctness is guaranteed -- the AlphaProof-shaped pattern (neural proposer +
sound verifier), built on our own kernel.

Domain: prove r(src,tgt) over a fixed relational graph by composing proven atoms via transitivity, each
composition kernel-checked. A NavProposer scores candidate next nodes given (current node, goal); beam
search follows it. Self-play trains the proposer on solved proof paths; kernel-verified LEMMAS (long-range
proven atoms) are cached so deeper goals fit the step budget = compounding. Bar: learned guidance solves
more within a fixed expansion budget than unguided search; UNSOUND 0; lemmas raise reach.

    python -m thinking.azr_kproof --selftest
    python -m thinking.azr_kproof --nodes 40 --rounds 1500
"""
import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.azr_proof import REL, RULES, base_tree, compose_tree, kernel_verify
from thinking.azr_iter import StepProposer            # the SAME learned proposer, reused literally
import thinking.lota_kernel as LK


def gen_graph(rng, N, branch=3):
    """Fixed random DAG over entities 0..N-1; edge i->j only for j>i (acyclic). r is transitive."""
    edges = set()
    for i in range(N - 1):
        ks = rng.choice(range(i + 1, N), size=min(branch, N - 1 - i), replace=False)
        for j in ks:
            edges.add((i, int(j)))
    # ensure connectivity of the spine so deep goals exist
    for i in range(N - 1):
        edges.add((i, i + 1))
    return sorted(edges)


def wordnet_graph(cap=600, roots=("animal.n.01", "plant.n.02", "artifact.n.01"), seed=0):
    """The REAL WordNet is-a DAG (a neighborhood): entities = synsets, edges = direct hypernym links.
    Proving r(child, ancestor) here = proving a real transitive is-a fact about actual concepts."""
    from nltk.corpus import wordnet as wn
    syns, queue = set(), [wn.synset(r) for r in roots]
    while queue and len(syns) < cap:                          # BFS down hyponyms from the roots
        s = queue.pop(0)
        if s.name() in syns:
            continue
        syns.add(s.name())
        queue += s.hyponyms()
    nodes = set(syns)
    for n in list(syns):                                      # add hypernym paths up to the roots
        for path in wn.synset(n).hypernym_paths():
            nodes.update(x.name() for x in path)
    nodes = sorted(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    edges = set()
    for n in nodes:
        for h in wn.synset(n).hypernyms():
            if h.name() in idx:
                edges.add((idx[n], idx[h.name()]))            # child -> direct parent (is-a)
    return len(nodes), sorted(edges), {i: n for n, i in idx.items()}


def reachable(edges, src):
    adj = {}
    for (a, b) in edges:
        adj.setdefault(a, []).append(b)
    seen, dist, dq = {src}, {src: 0}, [src]
    i = 0
    while i < len(dq):
        u = dq[i]; i += 1
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v); dist[v] = dist[u] + 1; dq.append(v)
    return dist


def ename(i):
    return f"e{i}"


# ===================================================================================================
# The proposer guiding the kernel proof search IS azr_iter's StepProposer, reused literally: a proof
# state is one "demo" (K=1) of length L=1 -- the current node -- with the goal node as the target, over
# an alphabet/op-vocab of the N entities. StepProposer(A=N, L=1, vocab=N) then maps (current, goal) -> a
# distribution over the next node. Same learned single-step proposer as azr_iter; the kernel verifies.
# ===================================================================================================
class NavProposer(nn.Module):
    def __init__(self, N, d=128):
        super().__init__()
        self.sp = StepProposer(A=N, L=1, vocab=N, d=d)

    def forward(self, cur, tgt):                          # cur,tgt: (B,) entity ids
        return self.sp(cur.view(-1, 1, 1), tgt.view(-1, 1, 1))   # -> (B, N) next-node logits


def solve(proposer, src, tgt, proven_edges, env, budget, beam, device, guided=True):
    """Beam search composing proven atoms src->...->tgt, KERNEL-VERIFYING each step. proven_edges: dict
    (x,y)->proof tree (base edges + cached lemmas). Returns (proof tree|None, n_steps, unsound)."""
    adj = {}
    for (x, y) in proven_edges:
        adj.setdefault(x, []).append(y)
    unsound = 0
    # frontier: (cur, proof tree for r(src,cur)). seed with src's direct edges.
    if src == tgt:
        return None, 0, 0
    frontier = [(y, proven_edges[(src, y)]) for y in adj.get(src, [])]
    for y, _ in frontier:
        if y == tgt:
            return proven_edges[(src, tgt)], 1, 0
    for step in range(budget):
        # score candidate frontier nodes by the proposer (or uniformly if unguided)
        if guided and frontier:
            cur_ids = torch.tensor([src] * len(frontier), device=device)
            tg = torch.tensor([tgt] * len(frontier), device=device)
            with torch.no_grad():
                sc = proposer(cur_ids, tg)
            scores = [float(sc[i, cur]) for i, (cur, _t) in enumerate(frontier)]
        else:
            scores = [random.random() for _ in frontier]     # unguided = RANDOM (fair baseline)
        order = sorted(range(len(frontier)), key=lambda i: -scores[i])[:beam]
        nxt_frontier, seen = [], set()
        for i in order:
            cur, tree = frontier[i]
            for nb in adj.get(cur, []):
                if nb in seen:
                    continue
                comp = compose_tree(tree, proven_edges[(cur, nb)])
                if not kernel_verify(comp, env):            # TRUSTED gate
                    unsound += 1; continue
                if nb == tgt:
                    return comp, step + 2, unsound
                seen.add(nb); nxt_frontier.append((nb, comp))
        if not nxt_frontier:
            break
        # keep the beam-best next nodes by proposer score toward tgt
        if guided:
            ci = torch.tensor([src] * len(nxt_frontier), device=device)
            tg = torch.tensor([tgt] * len(nxt_frontier), device=device)
            with torch.no_grad():
                sc = proposer(ci, tg)
            nxt_frontier.sort(key=lambda nt: -float(sc[0, nt[0]]))
        else:
            random.shuffle(nxt_frontier)                      # unguided keeps a RANDOM beam
        frontier = nxt_frontier[:beam]
    return None, budget, unsound


def chain_of(tree):
    """Recover the entity-id chain [src, ..., tgt] from a left-folded proof tree."""
    edges = []

    def collect(t):
        if t["rule"] is None:
            edges.append(t["fact"][1])                       # (x,y) base/lemma edge
        else:
            collect(t["from"][0]); edges.append(t["from"][1]["fact"][1])
    collect(tree)
    return [int(edges[0][0][1:])] + [int(e[1][1:]) for e in edges]    # "e5"->5


def path_pairs(tree, src):
    """HINDSIGHT navigation supervision: every downstream node on the path is a valid sub-goal ->
    (cur, sub-goal, next-step). Densely trains the navigator from one solved path (fixes cold-start)."""
    nodes = chain_of(tree)
    return [(nodes[k], nodes[m], nodes[k + 1])
            for k in range(len(nodes) - 1) for m in range(k + 1, len(nodes))]


def selfplay(N=40, branch=3, d=128, rounds=1500, bs=48, beam=4, budget=8, lr=1.5e-3,
             device=None, seeds=1, verbose=True, graph=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = []
    for seed in range(seeds):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        edges = graph[1] if graph is not None else gen_graph(rng, N, branch)   # real taxonomy or synthetic
        if graph is not None:
            N = graph[0]
        ents = [ename(i) for i in range(N)]
        base_facts = [(REL, (ename(a), ename(b))) for (a, b) in edges]
        env = LK.build_env(ents, {REL: 2}, base_facts, RULES)
        proven_edges = {(a, b): base_tree((REL, (ename(a), ename(b)))) for (a, b) in edges}
        prop = NavProposer(N, d=d).to(device)
        opt = torch.optim.AdamW(prop.parameters(), lr=lr, weight_decay=0.01)

        def sample_task():
            src = int(rng.integers(0, N - 2))
            dist = reachable(edges, src)
            far = [v for v, dd in dist.items() if dd >= 2]
            tgt = int(rng.choice(far)) if far else src + 1
            return src, tgt

        lemmas, lemma_count = {}, {}                          # kernel-verified long-range facts (the library)
        for rnd in range(rounds):
            pairs, solved = [], 0
            pe = {**proven_edges, **lemmas}                   # base edges + cached lemmas
            for _ in range(bs):
                src, tgt = sample_task()
                tree, _st, _un = solve(prop, src, tgt, pe, env, budget, beam, device, guided=True)
                if tree is not None:
                    solved += 1; pairs += path_pairs(tree, src)
                    # cache long-range (>=3 base hops) solved facts as reusable LEMMAS (kernel-verified
                    # long edges) -> shorten future proofs so deeper goals fit the step budget
                    if len(chain_of(tree)) - 1 >= 3 and (src, tgt) not in lemmas and len(lemmas) < 4 * N:
                        lemmas[(src, tgt)] = tree
            if pairs:
                prop.train()
                cur = torch.tensor([p[0] for p in pairs], device=device)
                tg = torch.tensor([p[1] for p in pairs], device=device)
                nx = torch.tensor([p[2] for p in pairs], device=device)
                loss = F.cross_entropy(prop(cur, tg), nx)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0); opt.step()
            if verbose and rnd % max(1, rounds // 8) == 0:
                ev = evaluate(prop, edges, proven_edges, env, N, budget, 1, device, n=150)
                print(f"  s{seed} r{rnd:5d}: train-solve {solved/bs:.2f} | guided {ev['guided']:.3f} "
                      f"unguided {ev['unguided']:.3f} | unsound {ev['unsound']}", flush=True)
        ev = evaluate(prop, edges, proven_edges, env, N, budget, 1, device, n=400)
        # COMPOUNDING: deep goals at a TIGHT budget, base edges vs base+lemmas (kernel-verified long edges)
        pe_lem = {**proven_edges, **lemmas}
        ev["lemmas"] = len(lemmas)
        ev["deep_base"] = evaluate(prop, edges, proven_edges, env, N, 2, 1, device, n=200, min_depth=4)["guided"]
        ev["deep_lem"] = evaluate(prop, edges, pe_lem, env, N, 2, 1, device, n=200, min_depth=4)["guided"]
        res.append(ev)
        if verbose:
            print(f"  s{seed} FINAL: guided {ev['guided']:.3f} vs random {ev['unguided']:.3f} | unsound "
                  f"{ev['unsound']} | lemmas {ev['lemmas']} | deep@budget2 base {ev['deep_base']:.3f} -> "
                  f"+lemmas {ev['deep_lem']:.3f} (COMPOUNDING)", flush=True)
    return res


def evaluate(prop, edges, proven_edges, env, N, budget, beam, device, n=300, min_depth=3):
    rng = np.random.default_rng(999)
    g = u = unsound = tot = 0
    for _ in range(n):
        src = int(rng.integers(0, N - 2))
        dist = reachable(edges, src)
        far = [v for v, dd in dist.items() if dd >= min_depth]   # deep enough that navigation matters
        if not far:
            continue
        tgt = int(rng.choice(far)); tot += 1
        tg, _s, un1 = solve(prop, src, tgt, proven_edges, env, budget, beam, device, guided=True)
        tu, _s2, un2 = solve(prop, src, tgt, proven_edges, env, budget, beam, device, guided=False)
        g += int(tg is not None); u += int(tu is not None); unsound += un1 + un2
    return dict(guided=g / max(1, tot), unguided=u / max(1, tot), unsound=unsound, n=tot)


def selftest():
    # beam=1 (greedy): the policy MUST pick the right next node each step -> a trained proposer navigates
    # to the goal, an unguided greedy walk (arbitrary choice) cannot. The cleanest test that guidance
    # works, with the kernel guaranteeing every accepted step is sound.
    res = selfplay(N=24, branch=4, d=64, rounds=900, bs=32, beam=3, budget=6, seeds=1,
                   device="cpu", verbose=False)
    ev = res[0]
    print(f"  selftest: guided {ev['guided']:.2f} vs random {ev['unguided']:.2f} | unsound {ev['unsound']} | "
          f"lemmas {ev['lemmas']} | deep@budget2 base {ev['deep_base']:.2f} -> +lemmas {ev['deep_lem']:.2f}")
    assert ev["unsound"] == 0, "kernel accepted an unsound step (not sound)"
    assert ev["guided"] > ev["unguided"] + 0.15, "learned guidance did not beat random (not learned)"
    assert ev["deep_lem"] > ev["deep_base"] + 0.1, "lemmas did not enable deeper proofs (no compounding)"
    print("azr_kproof selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--nodes", type=int, default=40); ap.add_argument("--branch", type=int, default=6)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--beam", type=int, default=4); ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--device", default=None)
    ap.add_argument("--wordnet", action="store_true", help="run over the REAL WordNet is-a graph")
    ap.add_argument("--cap", type=int, default=500)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    graph = None
    if args.wordnet:
        N, edges, names = wordnet_graph(cap=args.cap)
        graph = (N, edges)
        print(f"  REAL WordNet is-a graph: {N} synsets, {len(edges)} direct is-a links "
              f"(e.g. {names[edges[0][0]]} -> {names[edges[0][1]]})", flush=True)
    selfplay(N=args.nodes, branch=args.branch, d=args.d, rounds=args.rounds, beam=args.beam,
             budget=args.budget, seeds=args.seeds, device=args.device, graph=graph)
    return 0


if __name__ == "__main__":
    sys.exit(main())
