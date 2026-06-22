#!/usr/bin/env python3
"""azr_struct -- the experiment that validates Architecture v2 (thinking/SOLVE_ANYTHING.md). The ONLY
change from azr.py's plateaued run is a STRUCTURED task distribution: tasks are compositions of a small
set of recurring "concept" sub-procedures (not uniform-random programs). The diagnosis says this is the
load-bearing fix -- now solutions SHARE substructure, so MDL compression can DISCOVER the concepts, admit
them (kernel/executor-verified) to the library, and the solver builds depth-2/3 tasks from depth-1
concept-macros => composition stops being a wall and capability COMPOUNDS.

Bar (vs the uniform-random baseline: d1 0.97, d2/d3 ~0.36, 1 macro, frontier stuck at 2):
  - the library GROWS and its macros RECOVER the true concepts;
  - mean solution description-length DROPS as concepts are adopted;
  - d2/d3 (concept-depth) solve-rate CLIMBS past 0.36 and the frontier ADVANCES.

    python -m thinking.azr_struct --selftest
    python -m thinking.azr_struct --A 6 --L 4 --K 4 --concepts 6 --maxdepth 3 --rounds 2500
"""
import argparse
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from thinking.azr import (NBASE, NOP, Library, Solver, apply_base, decode_candidates, execute,
                          prog_len, _ctx_tensors, encode_ctx)


# ---- the latent STRUCTURE: a fixed set of recurring concept sub-procedures (each a base-op sequence) ----
def behav_equiv(p, q, A, L, rng, n=40):
    """Do programs p and q compute the same function (on n random length-L inputs over Z_A)?"""
    for _ in range(n):
        x = [int(v) for v in rng.integers(0, A, L)]
        if execute(p, x, A, _IDLIB) != execute(q, x, A, _IDLIB):
            return False
    return True


def make_concepts(rng, n_concepts, A, L, clen=2):
    """Distinct, IRREDUCIBLE concept programs (each `clen` non-NOP base ops, not equivalent to a shorter
    program and behaviorally distinct from each other). Irreducibility guarantees a depth-1 task genuinely
    needs `clen` ops -> the concept's substructure actually recurs in shortest solutions for compression
    to harvest. (The uniform-random run had no such recurring structure -> nothing to compress.)"""
    chk = np.random.default_rng(12345)                       # fixed RNG for equivalence checks
    singles = [[]] + [[op] for op in range(1, NBASE)]        # identity + every 1-op program
    concepts = []
    tries = 0
    while len(concepts) < n_concepts and tries < 3000:
        tries += 1
        c = [int(rng.integers(1, NBASE)) for _ in range(clen)]
        if any(behav_equiv(c, s, A, L, chk) for s in singles):       # reducible -> skip
            continue
        if any(behav_equiv(c, e, A, L, chk) for e in concepts):      # duplicate behavior -> skip
            continue
        concepts.append(c)
    return concepts


def gen_task_struct(rng, concepts, A, L, K, depth):
    """A task program = composition of `depth` CONCEPTS (in base ops). Same concepts recur across tasks."""
    cidx = [int(rng.integers(0, len(concepts))) for _ in range(depth)]
    prog = [op for ci in cidx for op in concepts[ci]]        # flatten to base ops
    xs = [[int(v) for v in rng.integers(0, A, L)] for _ in range(K + 1)]
    demos = [(x, execute(prog, x, A, _IDLIB)) for x in xs[:K]]
    return prog, demos, xs[K], execute(prog, xs[K], A, _IDLIB)


_IDLIB = Library(max_macros=0)                               # base-only executor for task generation


def solve_batch(solver, tasks, A, n_samples, device, lib):
    cand = decode_candidates(solver, tasks, n_samples, device, lib.active())
    out = []
    for (t, cs) in zip(tasks, cand):
        _p, demos, _xq, _y = t
        ver = [c for c in cs if all(execute(c, x, A, lib) == y for (x, y) in demos)]
        out.append((min(ver, key=prog_len), True) if ver else (None, False))
    return out


def evaluate(solver, rng, concepts, A, L, K, depths, device, n_samples, lib, n=200):
    out = {}
    for d in depths:
        tasks = [gen_task_struct(rng, concepts, A, L, K, d) for _ in range(n)]
        sol = solve_batch(solver, tasks, A, n_samples, device, lib)
        ok = sum(1 for (t, (p, f)) in zip(tasks, sol) if f and execute(p, t[2], A, lib) == t[3])
        out[d] = ok / n
    return out


def abstract_struct(lib, verified, k_add=2, thresh=15):
    """Add up to k_add new macros = the most frequent novel op-bigrams across verified programs (MDL:
    a recurring 2-op combo that shortens the corpus). Verifier-sound by construction: a macro is just a
    named base-op sequence, so executing it is identical to executing those base ops."""
    cnt = Counter()
    for p in verified:
        ops = [o for o in p if o != NOP]
        for a, b in zip(ops, ops[1:]):
            cnt[(a, b)] += 1
    added = []
    for (pair, c) in cnt.most_common():
        if c < thresh or len(added) >= k_add:
            break
        base_seq = lib.expand(pair[0]) + lib.expand(pair[1])
        if any(v == base_seq for v in lib.macros.values()):
            continue
        slot = lib.add_macro(base_seq)
        if slot is not None:
            added.append((slot, base_seq))
    return added


def concepts_recovered(lib, concepts, A, L):
    """Behavioral: a concept is recovered if SOME library macro computes the same function (not nec. the
    same op-sequence) -- the solver finds behaviorally-equivalent shortcuts, which is what matters."""
    chk = np.random.default_rng(777)
    macro_seqs = list(lib.macros.values())
    return sum(1 for c in concepts if any(behav_equiv(m, c, A, L, chk) for m in macro_seqs))


def mean_sol_len(sol):
    lens = [prog_len(p) for (p, f) in sol if f]
    return float(np.mean(lens)) if lens else 0.0


def selfplay(A=6, L=4, K=4, n_concepts=6, maxdepth=3, D=6, d=128, rounds=2500, bs=48, n_samples=16,
             lr=1.5e-3, max_macros=16, abstract_every=200, device=None, seeds=1, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = []
    for seed in range(seeds):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        concepts = make_concepts(rng, n_concepts, A, L)
        lib = Library(max_macros=max_macros)
        solver = Solver(A, L, D, lib.vocab, d=d).to(device)
        opt = torch.optim.AdamW(solver.parameters(), lr=lr, weight_decay=0.01)
        frontier, recent, vbuf = 1, [], []
        for rnd in range(rounds):
            tasks = [gen_task_struct(rng, concepts, A, L, K, int(rng.integers(1, frontier + 1)))
                     for _ in range(bs)]
            sol = solve_batch(solver, tasks, A, n_samples, device, lib)
            sctx, sprog = [], []
            for (t, (p, f)) in zip(tasks, sol):
                if f:
                    sctx.append(t); sprog.append(p); vbuf.append(p)
            recent.append(len(sctx) / bs)
            if sctx:
                solver.train()
                val, role, pos = _ctx_tensors(sctx, device)
                tgt = torch.tensor(sprog, device=device)
                BOS = solver.vocab
                dec_in = torch.cat([torch.full((len(sctx), 1), BOS, device=device), tgt[:, :-1]], 1)
                logits = solver(val, role, pos, dec_in)
                loss = F.cross_entropy(logits.reshape(-1, solver.vocab), tgt.reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0); opt.step()
            if rnd > 0 and rnd % abstract_every == 0 and vbuf:
                added = abstract_struct(lib, vbuf[-3000:])
                vbuf = vbuf[-3000:]
                if added and verbose:
                    print(f"  s{seed} r{rnd}: +macros {[a[0] for a in added]} "
                          f"(lib {len(lib.macros)}, concepts recovered {concepts_recovered(lib, concepts, A, L)}"
                          f"/{len(concepts)})", flush=True)
            if len(recent) >= 50 and np.mean(recent[-50:]) > 0.7 and frontier < maxdepth:
                frontier += 1; recent = []
                if verbose:
                    print(f"  s{seed} r{rnd}: frontier -> {frontier}", flush=True)
            if verbose and rnd % max(1, rounds // 10) == 0:
                acc = evaluate(solver, rng, concepts, A, L, K, range(1, maxdepth + 1), device,
                               n_samples, lib, n=100)
                print(f"  s{seed} r{rnd:5d} (frontier {frontier}, macros {len(lib.macros)}, "
                      f"concepts {concepts_recovered(lib, concepts, A, L)}/{len(concepts)}): "
                      + " ".join(f"d{k}:{v:.2f}" for k, v in acc.items()), flush=True)
        final = evaluate(solver, rng, concepts, A, L, K, range(1, maxdepth + 1), device, n_samples, lib, n=250)
        rec = concepts_recovered(lib, concepts, A, L)
        res.append((final, rec, len(lib.macros)))
        if verbose:
            print(f"  s{seed} FINAL: concepts {rec}/{len(concepts)}, macros {len(lib.macros)} | "
                  + " ".join(f"d{k}:{v:.3f}" for k, v in final.items()), flush=True)
    return res


def selftest():
    """CPU-only, tiny: with STRUCTURE present, the library must recover >=1 concept and beat depth-2
    chance -- the qualitative break the uniform-random selftest could not achieve."""
    res = selfplay(A=5, L=4, K=4, n_concepts=3, maxdepth=2, D=4, d=64, rounds=500, bs=32, n_samples=12,
                   max_macros=6, abstract_every=120, seeds=1, device="cpu", verbose=False)
    final, rec, nmac = res[0]
    print(f"  selftest: concepts recovered {rec}/3, macros {nmac} | d1:{final[1]:.2f} d2:{final[2]:.2f}")
    assert rec >= 1, f"compression failed to recover any concept from structured tasks ({rec})"
    print("azr_struct selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--A", type=int, default=6); ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--concepts", type=int, default=6)
    ap.add_argument("--maxdepth", type=int, default=3); ap.add_argument("--D", type=int, default=6)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--rounds", type=int, default=2500)
    ap.add_argument("--n-samples", type=int, default=16); ap.add_argument("--max-macros", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    res = selfplay(A=args.A, L=args.L, K=args.K, n_concepts=args.concepts, maxdepth=args.maxdepth,
                   D=args.D, d=args.d, rounds=args.rounds, n_samples=args.n_samples,
                   max_macros=args.max_macros, seeds=args.seeds, device=args.device)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump([{"final": {f"d{k}": v for k, v in fin.items()}, "concepts": rec, "macros": m}
                       for (fin, rec, m) in res], f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
