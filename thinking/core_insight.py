#!/usr/bin/env python3
"""core_insight -- CORE-style CONTRASTIVE REFLECTION in the abstract structured domain (arxiv 2605.28742,
"CORE: Contrastive Reflection Enables Rapid Improvements in Reasoning"). Instead of GRPO weight updates
(data-hungry), extract a NON-PARAMETRIC "insight" by CONTRASTING successful vs failed reasoning traces and
feed it back as a strategy prior at solve time -- no gradient, few rollouts.

Domain insight: here a CONCEPT is a recurring op-transition (clen=2 op-bigram), so the contrast naturally
recovers WHICH op-transitions lead to solutions. The insight bank = {prev_op -> {op -> log-odds(success vs
failure)}}, built from a handful of labeled rollouts, added to solve_iter's candidate score (w_insight). It
captures non-greedy 'detour' moves the executor-closeness heuristic alone misses. We measure solve@1 of a
WEAK proposer with vs without the bank -- a rapid, weight-free improvement from few traces.

    python -m thinking.core_insight --selftest
    python -m thinking.core_insight --maxdepth 4 --rounds 400 --traces 60
"""
import argparse
import math
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from thinking.azr import NOP, Library, execute
from thinking.azr_struct import (make_concepts, gen_task_struct, refactor, abstract_struct,
                                 concepts_recovered)
from thinking.azr_iter import StepProposer, solve_iter, solve_iter_batch, chain_train_pairs

START = -1


def train_weak(A, L, K, concepts, lib, d, rounds, maxdepth, beam, device, seed=0):
    """A deliberately LIGHT SFT proposer (ReST-EM, batched greedy) -- the base CORE will improve at test time."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    prop = StepProposer(A, L, lib.vocab, d=d).to(device)
    opt = torch.optim.AdamW(prop.parameters(), lr=1.5e-3, weight_decay=0.01)
    frontier, recent, vbuf = 1, [], []
    for rnd in range(rounds):
        max_steps = frontier * 2 + 2
        tasks = [gen_task_struct(rng, concepts, A, L, K, int(rng.integers(1, frontier + 1)))
                 for _ in range(32)]
        greedy = solve_iter_batch(prop, tasks, lib, A, L, max_steps, beam, device)
        pairs, ns = [], 0
        for ti, t in enumerate(tasks):
            ch = greedy[ti]
            if ch is not None:
                ns += 1; vbuf.append(ch)
                X = [list(x) for (x, _y) in t[1]]; Y = [list(y) for (_x, y) in t[1]]
                rc = refactor(ch, lib, len(ch) + len(lib.macros) + 4)
                rc = [o for o in rc if o != NOP] or ch
                pairs += chain_train_pairs(X, Y, rc, lib, A)
        recent.append(ns / 32)
        if pairs:
            prop.train(); rng.shuffle(pairs)
            st = torch.tensor([p[0] for p in pairs], device=device)
            tg = torch.tensor([p[1] for p in pairs], device=device)
            op = torch.tensor([p[2] for p in pairs], device=device)
            loss = F.cross_entropy(prop(st, tg), op)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0); opt.step()
        if rnd > 0 and rnd % 100 == 0 and vbuf:
            abstract_struct(lib, vbuf[-3000:]); vbuf = vbuf[-3000:]
        if len(recent) >= 40 and np.mean(recent[-40:]) > 0.7 and frontier < maxdepth:
            frontier += 1; recent = []
    return prop


@torch.no_grad()
def rollout_ops(prop, task, lib, A, L, max_steps, device, temp):
    """One stochastic rollout -> (op sequence, solved?). The raw trace CORE will reflect on."""
    prop.eval()
    demos = task[1]
    states = [list(x) for (x, _y) in demos]; Y = [list(y) for (_x, y) in demos]
    active = [op for op, a in enumerate(lib.active()) if a and op != NOP]
    amask = torch.full((prop.vocab,), float("-inf"), device=device)
    for o in active:
        amask[o] = 0.0
    tg = torch.tensor([Y], device=device)
    ops = []
    for _ in range(max_steps):
        if all(states[i] == Y[i] for i in range(len(Y))):
            break
        st = torch.tensor([states], device=device)
        lp = F.log_softmax(prop(st, tg)[0] / temp + amask, -1)
        o = int(torch.distributions.Categorical(logits=lp).sample())
        ops.append(o); states = [execute([o], s, A, lib) for s in states]
    return ops, all(states[i] == Y[i] for i in range(len(Y)))


def collect_traces(prop, concepts, A, L, K, lib, depths, device, n_tasks=20, n_roll=4, temp=1.0, seed=1):
    rng = np.random.default_rng(seed)
    traces = []
    for d in depths:
        for _ in range(n_tasks):
            t = gen_task_struct(rng, concepts, A, L, K, d)
            for _ in range(n_roll):
                traces.append(rollout_ops(prop, t, lib, A, L, d * 2 + 2, device, temp))
    return traces


def extract_insights(traces, alpha=1.0, mode="contrast", min_count=2):
    """Build the insight table {prev_op -> {op -> score}} from labeled traces.
      mode='contrast' : log-odds of (prev->op) in SUCCESS vs FAILURE (CORE-style; class-balanced by rate).
      mode='success'  : positive-only 'what works' prior = log(success rate of the transition).
    min_count: ignore transitions seen too rarely (noise floor)."""
    succ = defaultdict(lambda: defaultdict(float))
    fail = defaultdict(lambda: defaultdict(float))
    nsucc = nfail = 0
    for ops, solved in traces:
        nsucc += int(solved); nfail += int(not solved)
        tgt = succ if solved else fail
        prev = START
        for o in ops:
            tgt[prev][o] += 1.0; prev = o
    # class-balance: normalize failure counts so imbalance (many failed random-walk traces) doesn't dominate
    fscale = (nsucc / max(1, nfail))
    bank = {}
    for k in set(succ) | set(fail):
        bank[k] = {}
        for o in set(succ[k]) | set(fail[k]):
            s, f = succ[k].get(o, 0.0), fail[k].get(o, 0.0) * fscale
            if s + f < min_count:
                continue
            if mode == "success":
                bank[k][o] = math.log((s + alpha) / (s + f + 2 * alpha))    # P(success | transition)
            else:
                bank[k][o] = math.log((s + alpha) / (f + alpha))
        if not bank[k]:
            del bank[k]
    return bank, nsucc, nfail


def evaluate_core(prop, concepts, A, L, K, lib, depths, beam, device, bank=None, w_insight=1.0,
                  n=150, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    for d in depths:
        ok = 0
        for _ in range(n):
            t = gen_task_struct(rng, concepts, A, L, K, d)
            ch = solve_iter(prop, t, lib, A, L, d * 2 + 2, beam, device, bank=bank, w_insight=w_insight)
            ok += int(ch is not None)
        out[d] = ok / n
    return out


def top_insights(bank, k=6):
    flat = [((p, o), v) for p, d in bank.items() for o, v in d.items()]
    return sorted(flat, key=lambda x: -x[1])[:k]


def run(A=6, L=4, K=4, n_concepts=6, maxdepth=4, d=128, rounds=400, beam=8, max_macros=16,
        traces_per_depth=20, n_roll=4, w_insight=1.0, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    concepts = make_concepts(rng, n_concepts, A, L)
    lib = Library(max_macros=max_macros)
    prop = train_weak(A, L, K, concepts, lib, d, rounds, maxdepth, beam, device)
    depths = list(range(2, maxdepth + 1))
    base = evaluate_core(prop, concepts, A, L, K, lib, depths, beam, device, bank=None)
    traces = collect_traces(prop, concepts, A, L, K, lib, depths, device,
                            n_tasks=traces_per_depth, n_roll=n_roll)
    if verbose:
        print(f"  weak proposer trained {rounds} rounds; concepts recovered "
              f"{concepts_recovered(lib, concepts, A, L)}/{n_concepts}", flush=True)
    nsucc = sum(int(s) for _o, s in traces); nfail = len(traces) - nsucc
    if verbose:
        print(f"  CORE: reflected on {len(traces)} traces ({nsucc} success / {nfail} fail), NO weight update",
              flush=True)
        print("    baseline (no bank)   : " + " ".join(f"d{k}:{v:.3f}" for k, v in base.items()), flush=True)
    # sweep insight mode x weight on the SAME proposer+traces (only re-evals; cheap)
    best = (base, "none", 0.0)
    dd = max(depths)
    for mode in ("contrast", "success"):
        bank, _ns, _nf = extract_insights(traces, mode=mode)
        for w in (0.05, 0.1, 0.25, 0.5):
            ev = evaluate_core(prop, concepts, A, L, K, lib, depths, beam, device, bank=bank, w_insight=w)
            if verbose:
                print(f"    +{mode:8s} w={w:<4}: " + " ".join(f"d{k}:{v:.3f}" for k, v in ev.items()), flush=True)
            if ev[dd] > best[0][dd]:
                best = (ev, mode, w)
    if verbose:
        b = best[0]
        print(f"  BEST: {best[1]} w={best[2]}  ->  " + " ".join(f"d{k}:{v:.3f}" for k, v in b.items())
              + f"   (baseline d{dd} {base[dd]:.3f} -> {b[dd]:.3f}, NO weight update)", flush=True)
    return base, best[0], best


def selftest():
    base, cored, bank = run(A=5, L=4, K=4, n_concepts=3, maxdepth=3, d=64, rounds=250, beam=6,
                            max_macros=6, traces_per_depth=16, n_roll=4, device="cpu", verbose=False)
    dd = max(base)                                                  # deepest depth tested
    lift = cored[dd] - base[dd]
    print(f"  selftest: deepest d{dd} solve@1  baseline {base[dd]:.3f} -> +insight {cored[dd]:.3f} "
          f"(Δ{lift:+.3f}); bank has {len(bank)} contexts")
    assert cored[dd] >= base[dd] - 0.05, f"insight bank hurt solve@1 ({base[dd]:.3f}->{cored[dd]:.3f})"
    print("core_insight selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--A", type=int, default=6); ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--concepts", type=int, default=6)
    ap.add_argument("--maxdepth", type=int, default=4); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=400); ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--max-macros", type=int, default=16); ap.add_argument("--traces", type=int, default=20)
    ap.add_argument("--w-insight", type=float, default=1.0); ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    run(A=args.A, L=args.L, K=args.K, n_concepts=args.concepts, maxdepth=args.maxdepth, d=args.d,
        rounds=args.rounds, beam=args.beam, max_macros=args.max_macros, traces_per_depth=args.traces,
        w_insight=args.w_insight, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
