#!/usr/bin/env python3
"""azr -- ABSOLUTE-ZERO-style self-generated curriculum over VERIFIED PROGRAM SYNTHESIS, zero external
data, now with an AUTOREGRESSIVE solver + LIBRARY LEARNING (DreamCoder-style abstraction). See
thinking/SOLVE_ANYTHING.md.

A task IS a program. Loop, with NO human data:
  PROPOSE   sample a program (frontier depth) + inputs.
  EXECUTE   the executor (sound verifier) computes outputs -> verified demos.
  SOLVE     the solver AUTOREGRESSIVELY decodes candidate programs (op_t conditioned on op_<t); keep
            executor-verified ones (reproduce all demos); take the SHORTEST (Occam/MDL).
  LEARN     train the solver to emit that verified program (ReST-EM, self-generated supervision).
  ABSTRACT  periodically compress the most frequent op-combo in verified programs into a NEW library
            primitive (macro). Depth-2 combos become depth-1 -> the composition wall recedes.
  CURRICULUM raise frontier depth as solve-rate rises.

Two earlier walls this targets: (1) per-slot INDEPENDENT emission couldn't model op2|op1 -> fixed by the
autoregressive decoder; (2) composition plateau -> attacked by library learning (a learned, growing
simplicity prior). Bar: d2/d3 solve-rate climbs past the ~0.36 plateau, frontier advances.

    python -m thinking.azr --selftest
    python -m thinking.azr --A 6 --L 4 --K 4 --maxdepth 4 --rounds 2500
"""
import argparse
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_OPS = ["NOP", "ADD1", "ADD2", "SUB1", "REV", "ROL", "ROR", "SORT"]
NOP = 0
NBASE = len(BASE_OPS)


def apply_base(op, v, A):
    if op == 0:  return list(v)
    if op == 1:  return [(x + 1) % A for x in v]
    if op == 2:  return [(x + 2) % A for x in v]
    if op == 3:  return [(x - 1) % A for x in v]
    if op == 4:  return list(v[::-1])
    if op == 5:  return v[1:] + v[:1]
    if op == 6:  return v[-1:] + v[:-1]
    if op == 7:  return sorted(v)
    raise ValueError(op)


class Library:
    """Base ops + a growing list of MACROS. macro[k] expands to a flat list of BASE-op ids. The op
    vocabulary is pre-sized (NBASE + max_macros); macros are inactive until defined, then usable."""
    def __init__(self, max_macros=12):
        self.max_macros = max_macros
        self.vocab = NBASE + max_macros
        self.macros = {}                                    # op-id (>=NBASE) -> [base ops]

    def active(self):
        return [True] * NBASE + [(NBASE + k) in self.macros for k in range(self.max_macros)]

    def expand(self, op):
        return self.macros[op] if op >= NBASE else [op]

    def add_macro(self, base_seq):
        slot = NBASE + len(self.macros)
        if slot >= self.vocab:
            return None
        self.macros[slot] = list(base_seq)
        return slot


def execute(prog, v, A, lib):
    for op in prog:
        for b in lib.expand(op):
            v = apply_base(b, v, A)
    return v


def prog_len(prog):
    return sum(1 for op in prog if op != NOP)               # macro counts as 1 -> Occam prefers macros


def gen_task(rng, A, L, K, depth, lib):
    prog = [int(rng.integers(1, NBASE)) for _ in range(depth)]      # base-op program of given depth
    xs = [[int(v) for v in rng.integers(0, A, L)] for _ in range(K + 1)]
    demos = [(x, execute(prog, x, A, lib)) for x in xs[:K]]
    return prog, demos, xs[K], execute(prog, xs[K], A, lib)


ROLE = {"demo_in": 0, "demo_out": 1, "query_in": 2}


def encode_ctx(demos, xq):
    val, role, pos = [], [], []
    for (x, y) in demos:
        for i, s in enumerate(x): val.append(s); role.append(0); pos.append(i)
        for i, s in enumerate(y): val.append(s); role.append(1); pos.append(i)
    for i, s in enumerate(xq):     val.append(s); role.append(2); pos.append(i)
    return val, role, pos


# ===================================================================================================
# Autoregressive solver: encode context (demos+query), causally decode a program of D ops.
# ===================================================================================================
class Solver(nn.Module):
    def __init__(self, A, L, D, vocab, d=128, h=4, layers=3):
        super().__init__()
        self.D, self.vocab, self.d = D, vocab, d
        self.val = nn.Embedding(A, d); self.role = nn.Embedding(3, d); self.cpos = nn.Embedding(L, d)
        self.optok = nn.Embedding(vocab + 1, d)             # +1 = BOS (id == vocab)
        self.opos = nn.Embedding(D, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, h, 4 * d, batch_first=True, norm_first=True, activation="gelu")
            for _ in range(layers)])
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)

    def _mask(self, Tc, P, device):
        T = Tc + P
        m = torch.zeros(T, T, dtype=torch.bool, device=device)
        m[:Tc, Tc:] = True                                  # context can't see program
        m[Tc:, Tc:] = torch.triu(torch.ones(P, P, dtype=torch.bool, device=device), 1)  # causal program
        return m

    def forward(self, val, role, pos, progtok):
        """progtok: (B,P) decoder input (starts with BOS). Returns logits (B,P,vocab) predicting next op."""
        cx = self.val(val) + self.role(role) + self.cpos(pos)        # (B,Tc,d)
        Tc, P = cx.size(1), progtok.size(1)
        pe = self.optok(progtok) + self.opos(torch.arange(P, device=val.device))[None]
        h = torch.cat([cx, pe], 1)
        mask = self._mask(Tc, P, val.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.head(self.ln(h[:, Tc:]))                # (B,P,vocab)


def _ctx_tensors(tasks, device):
    enc = [encode_ctx(d, q) for (_p, d, q, _y) in tasks]
    val, role, pos = zip(*enc)
    return (torch.tensor(val, device=device), torch.tensor(role, device=device),
            torch.tensor(pos, device=device))


@torch.no_grad()
def decode_candidates(solver, tasks, n_samples, device, active_mask):
    """Batched autoregressive decode: for each task, 1 greedy + (n_samples-1) sampled programs of D ops.
    Returns a list (len=len(tasks)) of lists of candidate programs (each a length-D op list)."""
    solver.eval()
    bs = len(tasks)
    val, role, pos = _ctx_tensors(tasks, device)
    R = n_samples
    val = val.repeat_interleave(R, 0); role = role.repeat_interleave(R, 0); pos = pos.repeat_interleave(R, 0)
    N = bs * R
    BOS = solver.vocab
    seq = torch.full((N, 1), BOS, dtype=torch.long, device=device)
    block = torch.tensor([not a for a in active_mask], device=device)        # True = inactive op
    for t in range(solver.D):
        logits = solver(val, role, pos, seq)[:, -1]          # (N,vocab)
        logits = logits.masked_fill(block, -1e9)
        if t == 0:
            pass
        nxt = torch.empty(N, dtype=torch.long, device=device)
        # row 0 of each task's R block = greedy; rest = sampled
        greedy = logits.argmax(-1)
        probs = F.softmax(logits, -1)
        samp = torch.multinomial(probs, 1).squeeze(-1)
        is_greedy = (torch.arange(N, device=device) % R == 0)
        nxt = torch.where(is_greedy, greedy, samp)
        seq = torch.cat([seq, nxt[:, None]], 1)
    progs = seq[:, 1:].tolist()                              # drop BOS -> (N,D)
    return [progs[i * R:(i + 1) * R] for i in range(bs)]


def solve_batch(solver, tasks, A, n_samples, device, lib):
    """Return per-task (shortest verified program or None, found_bool)."""
    cand = decode_candidates(solver, tasks, n_samples, device, lib.active())
    out = []
    for (t, cs) in zip(tasks, cand):
        _p, demos, _xq, _y = t
        verified = [c for c in cs if all(execute(c, x, A, lib) == y for (x, y) in demos)]
        out.append((min(verified, key=prog_len), True) if verified else (None, False))
    return out


def evaluate(solver, rng, A, L, K, depths, device, n_samples, lib, n=200):
    out = {}
    for dpth in depths:
        tasks = [gen_task(rng, A, L, K, dpth, lib) for _ in range(n)]
        sol = solve_batch(solver, tasks, A, n_samples, device, lib)
        ok = sum(1 for (t, (prog, f)) in zip(tasks, sol)
                 if f and execute(prog, t[2], A, lib) == t[3])
        out[dpth] = ok / n
    return out


# ===================================================================================================
# Library learning: compress the most frequent op-bigram in verified programs into a new macro.
# ===================================================================================================
def abstract(lib, verified_progs):
    """Find the most frequent adjacent (a,b) op pair (both non-NOP) across verified programs; if it
    recurs enough, add macro = expand(a)+expand(b) (flattened to base ops)."""
    cnt = Counter()
    for p in verified_progs:
        ops = [o for o in p if o != NOP]
        for a, b in zip(ops, ops[1:]):
            cnt[(a, b)] += 1
    for (pair, c) in cnt.most_common():
        if c < 20:
            return None
        base_seq = lib.expand(pair[0]) + lib.expand(pair[1])
        # skip if an identical macro already exists
        if any(v == base_seq for v in lib.macros.values()):
            continue
        return lib.add_macro(base_seq)
    return None


def selfplay(A=6, L=4, K=4, maxdepth=4, D=4, d=128, rounds=2500, bs=48, n_samples=16, lr=1.5e-3,
             max_macros=12, abstract_every=300, device=None, seeds=1, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = []
    for seed in range(seeds):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        lib = Library(max_macros=max_macros)
        solver = Solver(A, L, D, lib.vocab, d=d).to(device)
        opt = torch.optim.AdamW(solver.parameters(), lr=lr, weight_decay=0.01)
        frontier, recent, vbuf = 1, [], []
        for rnd in range(rounds):
            tasks = [gen_task(rng, A, L, K, int(rng.integers(1, frontier + 1)), lib) for _ in range(bs)]
            sol = solve_batch(solver, tasks, A, n_samples, device, lib)
            sctx, sprog = [], []
            for (t, (prog, f)) in zip(tasks, sol):
                if f:
                    sctx.append(t); sprog.append(prog); vbuf.append(prog)
            recent.append(len(sctx) / bs)
            if sctx:
                solver.train()
                val, role, pos = _ctx_tensors(sctx, device)
                tgt = torch.tensor(sprog, device=device)                  # (n,D)
                BOS = solver.vocab
                dec_in = torch.cat([torch.full((len(sctx), 1), BOS, device=device), tgt[:, :-1]], 1)
                logits = solver(val, role, pos, dec_in)                   # (n,D,vocab)
                loss = F.cross_entropy(logits.reshape(-1, solver.vocab), tgt.reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0); opt.step()
            # ABSTRACT: grow the library from recent verified programs
            if rnd > 0 and rnd % abstract_every == 0 and vbuf:
                slot = abstract(lib, vbuf[-2000:])
                if slot is not None and verbose:
                    print(f"  seed {seed} round {rnd}: + macro {slot} = {lib.macros[slot]} "
                          f"(lib size {len(lib.macros)})", flush=True)
                vbuf = vbuf[-2000:]
            # CURRICULUM
            if len(recent) >= 50 and np.mean(recent[-50:]) > 0.7 and frontier < maxdepth:
                frontier += 1; recent = []
                if verbose:
                    print(f"  seed {seed} round {rnd}: frontier -> depth {frontier}", flush=True)
            if verbose and rnd % max(1, rounds // 10) == 0:
                acc = evaluate(solver, rng, A, L, K, range(1, maxdepth + 1), device, n_samples, lib, n=100)
                print(f"  seed {seed} round {rnd:5d} (frontier {frontier}, macros {len(lib.macros)}): "
                      + " ".join(f"d{k}:{v:.2f}" for k, v in acc.items()), flush=True)
        final = evaluate(solver, rng, A, L, K, range(1, maxdepth + 1), device, n_samples, lib, n=250)
        res.append((solver, lib, final))
        if verbose:
            print(f"  seed {seed} FINAL (macros {len(lib.macros)}): "
                  + " ".join(f"d{k}:{v:.3f}" for k, v in final.items()), flush=True)
    return res


def selftest():
    """CPU-only, tiny: autoregressive loop runs with zero external data, learns d1, and a macro is added."""
    A, L, K, D = 5, 4, 3, 3
    res = selfplay(A=A, L=L, K=K, maxdepth=2, D=D, d=64, rounds=400, bs=32, n_samples=12,
                   abstract_every=150, max_macros=6, seeds=1, device="cpu", verbose=False)
    solver, lib, final = res[0]
    rng = np.random.default_rng(123)
    untrained = Solver(A, L, D, lib.vocab, d=64)
    base = evaluate(untrained, rng, A, L, K, [1, 2], "cpu", 12, lib, n=200)
    print(f"  selftest: LEARNED d1:{final[1]:.2f} d2:{final[2]:.2f} | UNTRAINED d1:{base[1]:.2f} "
          f"d2:{base[2]:.2f} | macros learned: {len(lib.macros)}")
    assert final[1] > base[1] + 0.1, f"autoregressive solver didn't learn d1 ({final[1]} vs {base[1]})"
    print("azr selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--A", type=int, default=6); ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--maxdepth", type=int, default=4)
    ap.add_argument("--D", type=int, default=5, help="decoder program slots (>= maxdepth)")
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--rounds", type=int, default=2500)
    ap.add_argument("--n-samples", type=int, default=16); ap.add_argument("--max-macros", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    res = selfplay(A=args.A, L=args.L, K=args.K, maxdepth=args.maxdepth, D=args.D, d=args.d,
                   rounds=args.rounds, n_samples=args.n_samples, max_macros=args.max_macros,
                   seeds=args.seeds, device=args.device)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({f"d{k}": v for _s, _l, fin in res for k, v in fin.items()}, f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
