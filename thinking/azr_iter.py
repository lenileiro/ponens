#!/usr/bin/env python3
"""azr_iter -- the ITERATE-EXECUTE-RESIDUAL solver (thinking/SOLVE_ANYTHING.md "The Solver"). The fix for
the composition wall: instead of decoding a whole program blind, SOLVE by building it one verified op at a
time. A learned single-step proposer looks at the CURRENT executed state + the target and proposes the next
op; the executor applies it (forward), and a shallow BEAM search over executed states (scored by closeness
to target) finds a chain that reproduces ALL demos -- which is the sound verification. This turns a depth-d
task into d single-step inductions (the regime our associative inducer aces) and lets a wrong first step be
recovered by the beam (greedy can't). Trained on verified chains (+ macro-refactored for adoption); library
grows via MDL abstraction so recurring chains become one-step ops -> the depth ceiling rises (compounding).

Bar (vs whole-program decode: d1 0.97 but d2/d3 ~0.36, frontier stuck): d2/d3 climb past 0.36 + frontier
advances -- composition as interpolation over verified single steps.

    python -m thinking.azr_iter --selftest
    python -m thinking.azr_iter --A 6 --L 4 --K 4 --concepts 6 --maxdepth 3 --rounds 1500
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.azr import NBASE, NOP, Library, execute, apply_ops_np
from thinking.reasoner import Block
from thinking.azr_struct import (make_concepts, gen_task_struct, concepts_recovered, abstract_struct,
                                 refactor, _IDLIB)


# ===================================================================================================
# Single-step proposer: given the K demos' CURRENT states + targets, propose the next op.
# ===================================================================================================
class StepProposer(nn.Module):
    def __init__(self, A, L, vocab, d=128, h=4, layers=3):
        super().__init__()
        self.A, self.L, self.vocab = A, L, vocab
        self.val = nn.Embedding(A, d)
        self.role = nn.Embedding(2, d)                       # 0 = current state, 1 = target
        self.pos = nn.Embedding(L, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, vocab))

    def forward(self, state, target):
        """state, target: (B, K, L) int. Returns (B, vocab) next-op logits."""
        B, K, L = state.shape
        ar = torch.arange(L, device=state.device)
        s = self.val(state) + self.role.weight[0] + self.pos(ar)          # (B,K,L,d)
        t = self.val(target) + self.role.weight[1] + self.pos(ar)
        x = torch.cat([s, t], 2).reshape(B, K * 2 * L, s.size(-1))         # flatten demos
        h = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        for blk in self.blocks:
            h = blk(h, None)
        return self.head(h[:, 0])                            # pool via CLS token


def closeness(states, targets, L):
    """Fraction of positions matching across all demos (the execution-guided heuristic / value)."""
    tot = match = 0
    for s, y in zip(states, targets):
        for a, b in zip(s, y):
            match += int(a == b); tot += 1
    return match / max(1, tot)


@torch.no_grad()
def solve_iter(proposer, task, lib, A, L, max_steps, beam, device, w_policy=0.5, temp=0.0,
               bank=None, w_insight=1.0):
    """Beam search over single ops, execution-guided. Returns a verified op-chain (reproduces all demos)
    or None. temp>0 samples the kept beam stochastically (for best-of-N diversity); temp=0 is greedy.
    bank (optional): a CONTRASTIVE-INSIGHT table {prev_op -> {op -> log-odds}} added to the candidate score
    as a non-parametric strategy prior (no weight update); START prev-op key = -1."""
    proposer.eval()
    demos = task[1]
    X = [list(x) for (x, _y) in demos]
    Y = [list(y) for (_x, y) in demos]
    active = [op for op, a in enumerate(lib.active()) if a and op != NOP]
    frontier = [(X, [])]
    for _step in range(max_steps):
        for states, ops in frontier:
            if all(states[i] == Y[i] for i in range(len(Y))):
                return ops
        st = torch.tensor([states for states, _ in frontier], device=device)
        tg = torch.tensor([Y] * len(frontier), device=device)
        logp = F.log_softmax(proposer(st, tg), -1).detach().cpu().numpy()   # (F, vocab)
        Yarr = np.array(Y)
        aops = np.array(active)
        nA, Fn = len(active), len(frontier)
        # apply EVERY active op to EVERY frontier node's states in one grouped-by-op vectorized pass
        tiled = np.repeat(np.array([s for s, _ in frontier]), nA, axis=0)   # (Fn*nA, K, L)
        nss = apply_ops_np(tiled, np.tile(aops, Fn), lib, A)
        close = (nss == Yarr).reshape(Fn * nA, -1).mean(1)
        cands = []
        for fi in range(Fn):
            ops = frontier[fi][1]
            pri = bank.get(ops[-1] if ops else -1, {}) if bank is not None else None
            for j, o in enumerate(active):
                idx = fi * nA + j
                score = float(close[idx]) + w_policy * float(logp[fi, o])
                if pri is not None:
                    score += w_insight * pri.get(o, 0.0)       # contrastive-insight strategy prior
                cands.append((score, nss[idx].tolist(), ops + [o]))
        if temp > 0 and cands:                               # stochastic ordering for best-of-N diversity
            sc = np.array([c[0] for c in cands], dtype=float)
            pr = np.exp((sc - sc.max()) / temp); pr /= pr.sum()
            ordered = [cands[i] for i in np.random.choice(len(cands), size=len(cands), replace=False, p=pr)]
        else:
            ordered = sorted(cands, key=lambda c: -c[0])     # greedy (temp=0)
        seen, frontier = set(), []
        for score, ns, ops in ordered:
            key = tuple(tuple(s) for s in ns)
            if key in seen:
                continue
            seen.add(key); frontier.append((ns, ops))
            if len(frontier) >= beam:
                break
    for states, ops in frontier:
        if all(states[i] == Y[i] for i in range(len(Y))):
            return ops
    return None


@torch.no_grad()
def solve_iter_batch(proposer, tasks, lib, A, L, max_steps, beam, device, w_policy=0.5):
    """GREEDY (temp=0) beam search for MANY tasks at once: all tasks' frontier nodes go through ONE proposer
    forward per step (B = sum of beam sizes) instead of one forward per task. Same scoring/dedup/beam as
    solve_iter's greedy path. Returns a list of verified op-chains (or None) aligned with `tasks`."""
    proposer.eval()
    n = len(tasks)
    Xs = [[list(x) for (x, _y) in t[1]] for t in tasks]
    Ys = [[list(y) for (_x, y) in t[1]] for t in tasks]
    active = [op for op, a in enumerate(lib.active()) if a and op != NOP]
    aops = np.array(active); nA = len(active)
    frontiers = [[(Xs[i], [])] for i in range(n)]
    sol = [None] * n
    for i in range(n):
        if all(Xs[i][k] == Ys[i][k] for k in range(len(Ys[i]))):
            sol[i] = []
    for _step in range(max_steps):
        rows = [(i, fi) for i in range(n) if sol[i] is None
                for fi in range(len(frontiers[i]))]
        if not rows:
            break
        statemat = [frontiers[i][fi][0] for (i, fi) in rows]
        targmat = [Ys[i] for (i, _fi) in rows]
        st = torch.tensor(np.array(statemat), device=device)
        tg = torch.tensor(np.array(targmat), device=device)
        logp = F.log_softmax(proposer(st, tg), -1).detach().cpu().numpy()   # (R, vocab)
        R = len(rows)
        nss = apply_ops_np(np.repeat(np.array(statemat), nA, axis=0), np.tile(aops, R), lib, A)
        close = (nss == np.repeat(np.array(targmat), nA, axis=0)).reshape(R * nA, -1).mean(1)
        bytask = {i: [] for (i, _fi) in rows}
        for r, (i, fi) in enumerate(rows):
            ops = frontiers[i][fi][1]
            for j, o in enumerate(active):
                idx = r * nA + j
                bytask[i].append((float(close[idx]) + w_policy * float(logp[r, o]),
                                  nss[idx].tolist(), ops + [o]))
        for i, cands in bytask.items():
            seen, newf = set(), []
            for score, ns, ops in sorted(cands, key=lambda c: -c[0]):
                key = tuple(tuple(s) for s in ns)
                if key in seen:
                    continue
                seen.add(key); newf.append((ns, ops))
                if len(newf) >= beam:
                    break
            frontiers[i] = newf
            for states, ops in newf:
                if sol[i] is None and all(states[k] == Ys[i][k] for k in range(len(Ys[i]))):
                    sol[i] = ops
    return sol


def _apply_base_torch(op, s, A):
    """Vectorized base op on a torch int tensor (..., L) -- same semantics as apply_base, ON DEVICE."""
    if op == 0:  return s
    if op == 1:  return (s + 1) % A
    if op == 2:  return (s + 2) % A
    if op == 3:  return (s - 1) % A
    if op == 4:  return torch.flip(s, [-1])
    if op == 5:  return torch.roll(s, -1, dims=-1)
    if op == 6:  return torch.roll(s, 1, dims=-1)
    if op == 7:  return torch.sort(s, dim=-1).values
    raise ValueError(op)


@torch.no_grad()
def solve_iter_gpu(proposer, tasks, lib, A, L, max_steps, beam, device, w_policy=0.5):
    """FULLY TENSORIZED greedy beam search (keeps the GPU fed): the executor runs as torch ops on-device, and
    candidate-expand -> score -> top-beam are batched tensor ops -- NO Python per-candidate loop and NO
    per-step GPU->CPU sync. Fixed (n, beam) frontier (padded/masked). Skips exact state-dedup (top-k by score
    instead), which slightly lowers beam diversity but removes the Python that starved the GPU. Returns a list
    of verified op-chains (or None) aligned with `tasks` -- matches solve_iter_batch's solve set closely."""
    proposer.eval()
    n = len(tasks); B = beam; K = len(tasks[0][1])
    X = torch.tensor([[list(x) for (x, _y) in t[1]] for t in tasks], device=device)   # (n,K,L)
    Y = torch.tensor([[list(y) for (_x, y) in t[1]] for t in tasks], device=device)
    active = [op for op, a in enumerate(lib.active()) if a and op != NOP]
    nA = len(active)
    op_ids = torch.tensor(active, device=device)                                      # (nA,)
    expanded = [lib.expand(o) for o in active]                                        # base-seq per op
    F_ = X[:, None].expand(n, B, K, L).clone()                                        # (n,B,K,L) frontier
    valid = torch.zeros(n, B, dtype=torch.bool, device=device); valid[:, 0] = True
    chains = torch.full((n, B, max_steps), -1, dtype=torch.long, device=device)
    clen = torch.zeros(n, B, dtype=torch.long, device=device)
    solved = (X == Y).reshape(n, -1).all(1)
    sol = [[] if solved[i] else None for i in range(n)]
    arn = torch.arange(n, device=device)[:, None].expand(n, B)
    arB = torch.arange(B, device=device)[None, :].expand(n, B)
    Yb = Y[:, None]                                                                   # (n,1,K,L)
    for _step in range(max_steps):
        if bool(solved.all()):
            break
        logp = F.log_softmax(proposer(F_.reshape(n * B, K, L),
                                      Y[:, None].expand(n, B, K, L).reshape(n * B, K, L)), -1)
        logp = logp.reshape(n, B, -1)[:, :, active]                                   # (n,B,nA)
        cand = torch.stack([_reduce_ops(expanded[j], F_, A) for j in range(nA)], 2)   # (n,B,nA,K,L)
        close = (cand == Yb[:, :, None]).reshape(n, B, nA, -1).float().mean(-1)       # (n,B,nA)
        score = (close + w_policy * logp).masked_fill(~valid[:, :, None], float("-inf"))
        topv, topi = score.reshape(n, B * nA).topk(B, dim=1)                          # (n,B)
        parent, opj = topi // nA, topi % nA
        F_ = cand[arn, parent, opj]                                                   # (n,B,K,L)
        chains = chains[arn, parent].clone()
        clen = clen[arn, parent].clone()
        chains[arn, arB, clen.clamp(max=max_steps - 1)] = op_ids[opj]
        clen = clen + 1
        valid = topv > float("-inf")
        match = (F_ == Yb).reshape(n, B, -1).all(-1) & valid                          # (n,B)
        newly = (~solved) & match.any(1)
        if bool(newly.any()):
            first = match.float().argmax(1)                                           # first matching slot
            for i in torch.nonzero(newly).flatten().tolist():
                b = int(first[i]); sol[i] = chains[i, b, :int(clen[i, b])].tolist()
            solved = solved | newly
    return sol


def _reduce_ops(seq, s, A):
    for b in seq:
        s = _apply_base_torch(b, s, A)
    return s


def chain_train_pairs(X, Y, chain, lib, A):
    """A verified chain -> per-step (state, target, next-op) training tuples (state before each op)."""
    pairs, states = [], [list(x) for x in X]
    for op in chain:
        pairs.append((([list(s) for s in states]), Y, op))
        states = [execute([op], s, A, lib) for s in states]
    return pairs


def evaluate(proposer, rng, concepts, A, L, K, depths, lib, max_steps, beam, device, n=150,
             samples=1, temp=0.9):
    """solve-rate per depth. samples>1 = BEST-OF-N: try N stochastic rollouts, succeed if ANY verifies.
    Sound: solve_iter only returns executor-verified chains, so pass@N IS accuracy (no false positives)."""
    out = {}
    for d in depths:
        tasks = [gen_task_struct(rng, concepts, A, L, K, d) for _ in range(n)]
        gsolve = solve_iter_gpu if "cuda" in str(device) else solve_iter_batch
        greedy = gsolve(proposer, tasks, lib, A, L, max_steps, beam, device)            # s=0 greedy, batched
        ok = 0
        for ti, t in enumerate(tasks):
            if greedy[ti] is not None:
                ok += 1; continue
            for _s in range(samples - 1):                               # best-of-N: sampled fallback
                if solve_iter(proposer, t, lib, A, L, max_steps, beam, device, temp=temp) is not None:
                    ok += 1; break
        out[d] = ok / n
    return out


def collect_solve(prop, t, lib, A, L, max_steps, beam, device, collect_n, collect_temp):
    """ReST-EM collection: try greedy first, then up to collect_n-1 sampled rollouts; return the first
    verified chain. Banks solutions to HARD tasks the model can only reach by sampling -> training on them
    pulls those into greedy (solve@1) reach over rounds."""
    for s in range(max(1, collect_n)):
        chain = solve_iter(prop, t, lib, A, L, max_steps, beam, device,
                           temp=(0.0 if s == 0 else collect_temp))
        if chain is not None:
            return chain
    return None


def selfplay(A=6, L=4, K=4, n_concepts=6, maxdepth=3, d=128, rounds=1500, bs=32, beam=6,
             lr=1.5e-3, max_macros=16, abstract_every=100, collect_n=1, collect_temp=0.9,
             device=None, seeds=1, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = []
    for seed in range(seeds):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        concepts = make_concepts(rng, n_concepts, A, L)
        lib = Library(max_macros=max_macros)
        prop = StepProposer(A, L, lib.vocab, d=d).to(device)
        opt = torch.optim.AdamW(prop.parameters(), lr=lr, weight_decay=0.01)
        frontier_depth, recent, vbuf = 1, [], []
        for rnd in range(rounds):
            max_steps = frontier_depth * 2 + 2                # base ops needed = depth*clen, + slack
            tasks = [gen_task_struct(rng, concepts, A, L, K, int(rng.integers(1, frontier_depth + 1)))
                     for _ in range(bs)]
            pairs, nsolved = [], 0
            for t in tasks:
                chain = collect_solve(prop, t, lib, A, L, max_steps, beam, device, collect_n, collect_temp)
                if chain is not None:
                    nsolved += 1; vbuf.append(chain)
                    X = [list(x) for (x, _y) in t[1]]; Y = [list(y) for (_x, y) in t[1]]
                    rc = refactor(chain, lib, len(chain) + len(lib.macros) + 4)   # adopt macros
                    rc = [o for o in rc if o != NOP] or chain
                    pairs += chain_train_pairs(X, Y, rc, lib, A)
            recent.append(nsolved / bs)
            if pairs:
                prop.train()
                rng.shuffle(pairs)
                st = torch.tensor([p[0] for p in pairs], device=device)
                tg = torch.tensor([p[1] for p in pairs], device=device)
                op = torch.tensor([p[2] for p in pairs], device=device)
                loss = F.cross_entropy(prop(st, tg), op)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0); opt.step()
            if rnd > 0 and rnd % abstract_every == 0 and vbuf:
                added = abstract_struct(lib, vbuf[-3000:]); vbuf = vbuf[-3000:]
                if added and verbose:
                    print(f"  s{seed} r{rnd}: +macros {[a[0] for a in added]} (lib {len(lib.macros)}, "
                          f"concepts {concepts_recovered(lib, concepts, A, L)}/{len(concepts)})", flush=True)
            if len(recent) >= 40 and np.mean(recent[-40:]) > 0.7 and frontier_depth < maxdepth:
                frontier_depth += 1; recent = []
                if verbose:
                    print(f"  s{seed} r{rnd}: frontier -> {frontier_depth}", flush=True)
            if verbose and rnd % max(1, rounds // 10) == 0:
                acc = evaluate(prop, rng, concepts, A, L, K, range(1, maxdepth + 1),
                               lib, maxdepth * 2 + 2, beam, device, n=80)
                print(f"  s{seed} r{rnd:5d} (frontier {frontier_depth}, macros {len(lib.macros)}, "
                      f"concepts {concepts_recovered(lib, concepts, A, L)}/{len(concepts)}): "
                      + " ".join(f"d{k}:{v:.2f}" for k, v in acc.items()), flush=True)
        final = evaluate(prop, rng, concepts, A, L, K, range(1, maxdepth + 1), lib,
                         maxdepth * 2 + 2, beam, device, n=200)
        bestN = evaluate(prop, rng, concepts, A, L, K, range(1, maxdepth + 1), lib,
                         maxdepth * 2 + 2, beam, device, n=200, samples=16, temp=0.9)
        rec = concepts_recovered(lib, concepts, A, L)
        res.append((final, rec, len(lib.macros), bestN))
        if verbose:
            print(f"  s{seed} FINAL: concepts {rec}/{len(concepts)}, macros {len(lib.macros)}", flush=True)
            print("    solve@1   : " + " ".join(f"d{k}:{v:.3f}" for k, v in final.items()), flush=True)
            print("    best-of-16: " + " ".join(f"d{k}:{v:.3f}" for k, v in bestN.items())
                  + "   (verifier-sound test-time scaling)", flush=True)
    return res


def selftest():
    """CPU-only, tiny: the iterate-execute-residual solver must solve DEPTH-2 above chance -- the
    composition the whole-program decoder plateaued on."""
    res = selfplay(A=5, L=4, K=4, n_concepts=3, maxdepth=2, d=64, rounds=300, bs=24, beam=5,
                   max_macros=6, abstract_every=120, seeds=1, device="cpu", verbose=False)
    final, rec, nmac, bestN = res[0]
    print(f"  selftest: concepts {rec}/3, macros {nmac} | solve@1 d2:{final[2]:.2f} | "
          f"best-of-16 d2:{bestN[2]:.2f} (verifier-sound test-time scaling)")
    assert final[2] > 0.4, f"iterate-execute-residual failed to compose depth-2 ({final[2]:.2f})"
    assert bestN[2] >= final[2], "best-of-N should not hurt (it can only add verified solutions)"
    print("azr_iter selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--A", type=int, default=6); ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--concepts", type=int, default=6)
    ap.add_argument("--maxdepth", type=int, default=3); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=1500); ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--max-macros", type=int, default=16); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--collect-n", type=int, default=1, help="ReST-EM: best-of-N rollouts to COLLECT a "
                    "verified solution per task during training (1 = greedy only)")
    ap.add_argument("--device", default=None); ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    res = selfplay(A=args.A, L=args.L, K=args.K, n_concepts=args.concepts, maxdepth=args.maxdepth,
                   d=args.d, rounds=args.rounds, beam=args.beam, max_macros=args.max_macros,
                   collect_n=args.collect_n, seeds=args.seeds, device=args.device)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump([{"solve_at_1": {f"d{k}": v for k, v in fin.items()},
                        "best_of_16": {f"d{k}": v for k, v in bN.items()}, "concepts": rec, "macros": m}
                       for (fin, rec, m, bN) in res], f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
