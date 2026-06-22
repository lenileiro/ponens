#!/usr/bin/env python3
"""induce -- a model that FINDS THE RULE ON ITS OWN from the input and solves it. No language training,
no hand-coded rules: each problem hides a DIFFERENT rule, shown only through a few input->output demos,
and the model must INDUCE the rule in-context and apply it to a query. Generalization is tested on rules
NEVER SEEN in training -- so it can only succeed by discovering the pattern, not memorizing.

This is in-context rule induction (the ARC-AGI / meta-learning regime; HRM/TRM territory): the recurrent
"thinking" core (one weight-shared block, unrolled with INPUT INJECTION) reads the demos + query and
fills the answer. Symbols are abstract ids -- nothing linguistic. A symbolic rule engine GENERATES the
data and VERIFIES outputs (exact match); the model never calls it.

A rule = a hidden symbol substitution (a random bijection on the alphabet) optionally composed with a
structural transform (reverse / cyclic shift). The model must read the demos, figure out BOTH the symbol
remapping and the positional transform, and apply them to the query.

    python -m thinking.induce --selftest
    python -m thinking.induce --A 10 --L 6 --K 5 --steps 6000 --seeds 3
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.reasoner import Block                          # reuse the proven recurrent block

ROLE = {"demo_in": 0, "demo_out": 1, "query_in": 2, "answer": 3, "sep": 4}


# ===================================================================================================
# Rules + episodes. A rule is a fresh random program per episode -> the model can NEVER memorize it;
# it must be induced from the demos. The engine also VERIFIES (it computes the true output).
# ===================================================================================================
def make_rule(rng, A, allow_reverse, shift_set, substitute=True):
    # STRUCTURED value rule: add a hidden offset c (mod A) -- a discoverable PATTERN (one number to
    # induce), not a random lookup table. The model infers c from any demo pair and applies it.
    c = int(rng.integers(0, A)) if substitute else 0
    sigma = (np.arange(A) + c) % A
    rev = bool(allow_reverse and rng.random() < 0.5)
    shift = int(rng.choice(shift_set))

    def apply(seq):
        s = [int(sigma[x]) for x in seq]                     # 1) relabel symbols
        if rev:
            s = s[::-1]                                       # 2) structural transform...
        if shift:
            s = s[shift:] + s[:shift]                         #    (cyclic shift)
        return s
    return apply, dict(rev=rev, shift=shift)


def gen_episode(rng, A, L, K, allow_reverse, shift_set, substitute=True):
    rule, _desc = make_rule(rng, A, allow_reverse, shift_set, substitute)
    demos = []
    for _ in range(K):
        x = [int(v) for v in rng.integers(0, A, size=L)]
        demos.append((x, rule(x)))
    xq = [int(v) for v in rng.integers(0, A, size=L)]
    return demos, xq, rule(xq)


def encode(episode, L):
    """Flatten demos + query into fixed-length (sym, role, pos) token streams. We predict the output
    DIRECTLY at the query-input positions (the L tokens just before the final sep) -- no placeholder
    indirection: the query symbol is right there, the model just transforms it."""
    demos, xq, yq = episode
    sym, role, pos = [], [], []

    def put(seq, r):
        for i, s in enumerate(seq):
            sym.append(s); role.append(ROLE[r]); pos.append(i)

    def sep():
        sym.append(0); role.append(ROLE["sep"]); pos.append(0)

    for (x, y) in demos:
        put(x, "demo_in"); sep(); put(y, "demo_out"); sep()
    put(xq, "query_in"); sep()                               # query_in = positions [-(L+1):-1]
    return sym, role, pos, yq


def make_batch(rng, A, L, K, bs, allow_reverse, shift_set, substitute=True):
    eps = [encode(gen_episode(rng, A, L, K, allow_reverse, shift_set, substitute), L) for _ in range(bs)]
    sym, role, pos, yq = zip(*eps)
    return (torch.tensor(sym), torch.tensor(role), torch.tensor(pos), torch.tensor(yq))


# ===================================================================================================
# Model: embed (symbol + role + within-segment position); unroll ONE shared block T steps with input
# injection; read the answer slots -> predict symbols. Position lets it align demo-in vs demo-out
# positions (to discover reverse/shift); symbol embedding lets it discover the substitution.
# ===================================================================================================
class Inducer(nn.Module):
    def __init__(self, A, L, d=128, h=4):
        super().__init__()
        self.A, self.L = A, L
        self.sym = nn.Embedding(A, d)
        self.role = nn.Embedding(len(ROLE), d)
        self.pos = nn.Embedding(L, d)
        # TWO stacked layers per thinking step (untied within the step) -> enough depth for the
        # match-and-copy (induction-head) circuit that substitution lookup needs; the whole 2-layer
        # unit is then LOOPED with input injection for the iterative/structural part of the rule.
        self.block1 = Block(d, h)
        self.block2 = Block(d, h)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, A))

    def forward(self, sym, role, pos, T):
        x0 = self.sym(sym) + self.role(role) + self.pos(pos)
        h = x0
        for _t in range(T):
            h = self.block2(self.block1(h + x0, None), None)  # input injection each thinking step
        ans = h[:, -(self.L + 1):-1, :]                       # predict at the query_in positions
        return self.head(ans)                                 # (B, L, A) symbol logits


# ===================================================================================================
def train(model, rng, A, L, K, T, steps, device, allow_reverse, shift_set, substitute_off=False,
          bs=64, lr=1.2e-3, warmup_frac=0.1, log=0):
    import math
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(steps * warmup_frac))

    def lr_at(s):
        if s < warm:
            return lr * (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        sym, role, pos, yq = make_batch(rng, A, L, K, bs, allow_reverse, shift_set, not substitute_off)
        sym, role, pos, yq = [t.to(device) for t in (sym, role, pos, yq)]
        logits = model(sym, role, pos, T)                    # (B,L,A)
        loss = F.cross_entropy(logits.reshape(-1, A), yq.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and (step % log == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {loss.item():.4f}  lr {lr_at(step):.2e}", flush=True)


@torch.no_grad()
def evaluate(model, rng, A, L, K, T, device, allow_reverse, shift_set, substitute_off=False, n=2000):
    model.eval()
    seq_ok = sym_ok = tot = symtot = 0
    while tot < n:
        sym, role, pos, yq = make_batch(rng, A, L, K, min(256, n), allow_reverse, shift_set,
                                        not substitute_off)
        sym, role, pos = [t.to(device) for t in (sym, role, pos)]
        pred = model(sym, role, pos, T).argmax(-1).cpu()     # (B,L)
        match = (pred == yq)
        seq_ok += (match.all(1)).sum().item()
        sym_ok += match.sum().item()
        tot += len(yq); symtot += yq.numel()
    return dict(exact=seq_ok / tot, per_symbol=sym_ok / symtot)


def run(A=10, L=6, K=5, d=128, h=4, T=10, steps=6000, seeds=1, device=None, chunks=10, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # STRUCTURED rule space: value rule add-c (mod A, any c) + positional reverse + shift. TRAIN shifts
    # {0,1,2}; HELD-OUT unseen shifts {3,4} -> the positional parameter at test was never trained, only
    # inducible from demos. Train in CHUNKS and eval each -> we SEE the induction circuit form (grokking).
    train_shifts, heldout_shifts = [0, 1, 2], [3, 4]
    per = max(1, steps // chunks)
    rows = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = Inducer(A, L, d=d, h=h).to(device)
        done = 0
        while done < steps:
            n = min(per, steps - done)
            train(model, rng, A, L, K, T, n, device, allow_reverse=True, shift_set=train_shifts, log=0)
            done += n
            if verbose:
                se = evaluate(model, rng, A, L, K, T, device, True, train_shifts, n=1000)
                sh = evaluate(model, rng, A, L, K, T, device, True, heldout_shifts, n=1000)
                print(f"  seed {seed} @ {done:6d}: SEEN exact {se['exact']:.3f} sym {se['per_symbol']:.3f}"
                      f" | HELD-OUT(shift{heldout_shifts}) exact {sh['exact']:.3f} sym {sh['per_symbol']:.3f}",
                      flush=True)
        seen = evaluate(model, rng, A, L, K, T, device, True, train_shifts, n=3000)
        held = evaluate(model, rng, A, L, K, T, device, True, heldout_shifts, n=3000)
        rows.append((seen, held))
    se, sh = np.mean([r[0]['exact'] for r in rows]), np.mean([r[1]['exact'] for r in rows])
    sse, ssh = np.mean([r[0]['per_symbol'] for r in rows]), np.mean([r[1]['per_symbol'] for r in rows])
    if verbose:
        print(f"\n  MEAN/{seeds}: SEEN exact {se:.3f} (sym {sse:.3f}) | HELD-OUT exact {sh:.3f} "
              f"(sym {ssh:.3f}) | chance exact {(1/A)**L:.1e} sym {1/A:.3f}", flush=True)
        print("  -> learned to INDUCE RULES iff HELD-OUT (unseen shift) accuracy is high -- discovering "
              "the value+position pattern from demos, not memorizing.", flush=True)
    return dict(A=A, L=L, K=K, T=T, steps=steps, seeds=seeds, heads=h,
                seen_exact=float(se), seen_sym=float(sse), heldout_exact=float(sh), heldout_sym=float(ssh))


def selftest():
    """CPU-only, tiny: verify the PIPELINE end-to-end on the copy rule (identity), which needs no
    induction circuit and must be learned to ~perfect. (Value/positional rule INDUCTION is the open
    frontier we scale on GPU -- it needs far more training than a CPU selftest, so it is NOT asserted
    here; do not fake a pass.)"""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    A, L, K = 6, 4, 5
    model = Inducer(A, L, d=64)
    train(model, rng, A, L, K, T=6, steps=400, device="cpu", allow_reverse=False, shift_set=[0],
          substitute_off=True, log=0)
    r = evaluate(model, rng, A, L, K, T=6, device="cpu", allow_reverse=False, shift_set=[0],
                 substitute_off=True, n=1000)
    print(f"  selftest: identity-copy per-symbol {r['per_symbol']:.3f} (chance {1/A:.3f}) -- pipeline check")
    assert r["per_symbol"] > 0.9, f"pipeline broken: identity copy not learned ({r['per_symbol']:.3f})"
    print("induce selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--A", type=int, default=10, help="alphabet size")
    ap.add_argument("--L", type=int, default=6, help="sequence length")
    ap.add_argument("--K", type=int, default=5, help="number of demos per episode")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--T", type=int, default=10, help="recurrent thinking steps")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    res = run(A=args.A, L=args.L, K=args.K, d=args.d, h=args.heads, T=args.T, steps=args.steps,
              seeds=args.seeds, device=args.device)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
