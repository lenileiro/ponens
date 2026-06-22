#!/usr/bin/env python3
"""azr -- ABSOLUTE-ZERO-style self-generated curriculum over VERIFIED PROGRAM SYNTHESIS. Zero external
data: the model invents its own tasks, an executor verifies them, and it learns to SOLVE by inducing the
program from examples -- the realizable shadow of Solomonoff induction (programs = hypotheses, executor =
sound verifier, shortest verified program = simplicity prior). See thinking/SOLVE_ANYTHING.md.

A "task" IS a program. The loop, with NO human data:
  1. PROPOSE  -- sample a program P (depth d at the current frontier) + K+1 input lists.
  2. EXECUTE  -- the EXECUTOR (sound verifier) computes outputs -> verified demos (x, P(x)) + a query.
  3. SOLVE    -- the neural solver reads the demos and emits candidate programs; we keep only those the
                 executor confirms reproduce ALL demos (VERIFIED), and take the SHORTEST (Occam/MDL).
  4. LEARN    -- train the solver (cross-entropy) to emit that verified program from those demos
                 (ReST-EM / STaR; the supervision is self-generated + executor-verified).
  5. CURRICULUM -- raise the frontier depth when the solve-rate is high; the solver + task space co-evolve.

The bar: solve-rate on HELD-OUT self-generated tasks climbs from ~0 with zero external data, and the
LEARNED solver beats an equal search budget with an untrained solver (it learned to INDUCE, not just search).

    python -m thinking.azr --selftest
    python -m thinking.azr --A 7 --L 5 --K 4 --maxdepth 3 --rounds 4000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.reasoner import Block

# ---- the DSL: ops on a list v of L ints mod A. NOP first (lets programs be shorter than D slots). ----
OPS = ["NOP", "ADD1", "ADD2", "SUB1", "REV", "ROL", "ROR", "SORT"]
NOP = 0


def apply_op(op, v, A):
    if op == 0:    return list(v)                      # NOP
    if op == 1:    return [(x + 1) % A for x in v]     # ADD1
    if op == 2:    return [(x + 2) % A for x in v]     # ADD2
    if op == 3:    return [(x - 1) % A for x in v]     # SUB1
    if op == 4:    return list(v[::-1])                # REV
    if op == 5:    return v[1:] + v[:1]                # ROL
    if op == 6:    return v[-1:] + v[:-1]              # ROR
    if op == 7:    return sorted(v)                    # SORT
    raise ValueError(op)


def execute(prog, v, A):
    for op in prog:
        v = apply_op(op, v, A)
    return v


def prog_len(prog):
    return sum(1 for op in prog if op != NOP)           # non-NOP ops = description length (simplicity)


# ---- task proposer (self-generated, zero external data) ----
def gen_task(rng, A, L, K, depth):
    """A hidden program of EXACTLY `depth` non-NOP ops + K demo inputs + 1 query input."""
    prog = [int(rng.integers(1, len(OPS))) for _ in range(depth)]      # non-NOP ops
    xs = [[int(v) for v in rng.integers(0, A, L)] for _ in range(K + 1)]
    demos = [(x, execute(prog, x, A)) for x in xs[:K]]
    xq = xs[K]
    return prog, demos, xq, execute(prog, xq, A)


ROLE = {"demo_in": 0, "demo_out": 1, "query_in": 2, "slot": 3}


def encode_ctx(demos, xq, L):
    val, role, pos = [], [], []
    for (x, y) in demos:
        for i, s in enumerate(x): val.append(s); role.append(ROLE["demo_in"]); pos.append(i)
        for i, s in enumerate(y): val.append(s); role.append(ROLE["demo_out"]); pos.append(i)
    for i, s in enumerate(xq):     val.append(s); role.append(ROLE["query_in"]); pos.append(i)
    return val, role, pos


def batch_ctx(tasks, L, device):
    enc = [encode_ctx(d, q, L) for (_p, d, q, _y) in tasks]
    val, role, pos = zip(*enc)
    return (torch.tensor(val, device=device), torch.tensor(role, device=device),
            torch.tensor(pos, device=device))


# ---- the solver: reads demos+query, emits D program slots (op distribution per slot) ----
class Solver(nn.Module):
    def __init__(self, A, L, D, d=128, h=4, layers=3):
        super().__init__()
        self.D = D
        self.val = nn.Embedding(A, d)
        self.role = nn.Embedding(len(ROLE), d)
        self.pos = nn.Embedding(L, d)
        self.slot = nn.Embedding(D, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, len(OPS)))

    def forward(self, val, role, pos):
        B = val.size(0)
        x = self.val(val) + self.role(role) + self.pos(pos)           # (B,Tc,d)
        slots = self.slot(torch.arange(self.D, device=val.device)).unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([x, slots], 1)
        for b in self.blocks:
            h = b(h, None)
        return self.head(h[:, -self.D:])                              # (B,D,|OPS|) per-slot op logits


@torch.no_grad()
def solve(solver, task, A, L, n_samples, device, greedy_first=True):
    """Sample candidate programs from the solver, keep executor-VERIFIED ones (match all demos), return
    the SHORTEST verified program (Occam). Returns (prog or None, found_bool)."""
    _p, demos, _xq, _y = task
    val, role, pos = batch_ctx([task], L, device)
    logits = solver(val, role, pos)[0]                                # (D,|OPS|)
    cands = []
    if greedy_first:
        cands.append([int(a) for a in logits.argmax(-1)])
    probs = F.softmax(logits, -1)
    for _ in range(n_samples):
        cands.append([int(torch.multinomial(probs[s], 1)) for s in range(solver.D)])
    verified = [c for c in cands if all(execute(c, x, A) == y for (x, y) in demos)]
    if not verified:
        return None, False
    return min(verified, key=prog_len), True


def evaluate(solver, rng, A, L, K, D, depths, device, n_samples, n=200):
    """Solve-rate (query output exactly correct) on fresh held-out self-generated tasks, per depth."""
    out = {}
    for d in depths:
        ok = 0
        for _ in range(n):
            task = gen_task(rng, A, L, K, d)
            prog, found = solve(solver, task, A, L, n_samples, device)
            if found and execute(prog, task[2], A) == task[3]:
                ok += 1
        out[d] = ok / n
    return out


def selfplay(A=7, L=5, K=4, maxdepth=3, D=4, d=128, rounds=4000, bs=64, n_samples=24, lr=1.5e-3,
             device=None, seeds=1, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = []
    for seed in range(seeds):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        solver = Solver(A, L, D, d=d).to(device)
        opt = torch.optim.AdamW(solver.parameters(), lr=lr, weight_decay=0.01)
        frontier = 1
        recent = []
        for rnd in range(rounds):
            # 1-2. PROPOSE + EXECUTE: a batch of self-generated, executor-verified tasks at the frontier
            tasks = [gen_task(rng, A, L, K, int(rng.integers(1, frontier + 1))) for _ in range(bs)]
            # 3. SOLVE via sampling + executor verification; collect (ctx -> shortest verified program)
            solved_ctx, solved_prog, nsolved = [], [], 0
            for t in tasks:
                prog, found = solve(solver, t, A, L, n_samples, device)
                if found:
                    solved_ctx.append(t); solved_prog.append(prog); nsolved += 1
            recent.append(nsolved / bs)
            # 4. LEARN: train solver to emit the verified program from those demos (self-supervised)
            if solved_ctx:
                val, role, pos = batch_ctx(solved_ctx, L, device)
                tgt = torch.tensor(solved_prog, device=device)         # (n,D)
                solver.train()
                logits = solver(val, role, pos)
                loss = F.cross_entropy(logits.reshape(-1, len(OPS)), tgt.reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
                opt.step()
            # 5. CURRICULUM: raise frontier when recently solving the current frontier well
            if len(recent) >= 50 and np.mean(recent[-50:]) > 0.7 and frontier < maxdepth:
                frontier += 1; recent = []
                if verbose:
                    print(f"  seed {seed} round {rnd}: frontier -> depth {frontier}", flush=True)
            if verbose and rnd % max(1, rounds // 10) == 0:
                acc = evaluate(solver, rng, A, L, K, D, range(1, frontier + 1), device, n_samples, n=100)
                print(f"  seed {seed} round {rnd:5d} (frontier {frontier}): solve-rate "
                      + " ".join(f"d{k}:{v:.2f}" for k, v in acc.items()), flush=True)
        final = evaluate(solver, rng, A, L, K, D, range(1, maxdepth + 1), device, n_samples, n=200)
        res.append((solver, final))
        if verbose:
            print(f"  seed {seed} FINAL solve-rate: "
                  + " ".join(f"d{k}:{v:.3f}" for k, v in final.items()), flush=True)
    return res


def selftest():
    """CPU-only, tiny: the self-play loop runs with ZERO external data and the LEARNED solver beats an
    equal-budget UNTRAINED solver on held-out depth-1/2 tasks (it learned to induce, not just search)."""
    A, L, K, D = 5, 4, 4, 3
    res = selfplay(A=A, L=L, K=K, maxdepth=2, D=D, d=64, rounds=400, bs=48, n_samples=12, seeds=1,
                   device="cpu", verbose=False)
    solver, final = res[0]
    rng = np.random.default_rng(123)
    untrained = Solver(A, L, D, d=64)
    base = evaluate(untrained, rng, A, L, K, D, [1, 2], "cpu", n_samples=12, n=200)
    print(f"  selftest: LEARNED solve-rate d1:{final[1]:.2f} d2:{final[2]:.2f} | "
          f"UNTRAINED(=search only) d1:{base[1]:.2f} d2:{base[2]:.2f}")
    assert final[1] > base[1] + 0.1, f"learned solver did not beat search-only at d1 ({final[1]} vs {base[1]})"
    print("azr selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--A", type=int, default=7)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--maxdepth", type=int, default=3)
    ap.add_argument("--D", type=int, default=4, help="program slots (>= maxdepth)")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    res = selfplay(A=args.A, L=args.L, K=args.K, maxdepth=args.maxdepth, D=args.D, d=args.d,
                   rounds=args.rounds, n_samples=args.n_samples, seeds=args.seeds, device=args.device)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({f"d{k}": v for _s, fin in res for k, v in fin.items()}, f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
