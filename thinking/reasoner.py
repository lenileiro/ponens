#!/usr/bin/env python3
"""reasoner -- a model that LEARNS TO THINK (iterated deduction in latent space), taught + verified by
the symbolic kernel, instead of CALLING a hand-coded inference engine.

The critique that started this: our datalog closure + proof kernel is hand-coded -- the model wasn't
reasoning, it was calling our engine. The literature (Universal Transformer / recurrent-depth /
TRM/HRM / Looped-Transformers + input injection; AlphaProof-style proposer+verifier; implicit-CoT)
converges on one design: a small WEIGHT-SHARED RECURRENT block, unrolled at test time, learns the
iterated computation; a sound checker TEACHES and VERIFIES it. This is the smallest module that proves
"the model learned to reason": it learns multi-hop ENTAILMENT (is q derivable from these edges by
transitive closure?) and must GENERALIZE TO UNSEEN DEPTH -- train on <=K hops, test on >K -- with NO
call to the engine at inference.

Why this task: transitive-closure entailment is exactly the iterated computation our own length-gen
work cracked (recurrence + INPUT INJECTION generalized k<=20 -> k=40). Here each iteration does ONE hop
of deduction (Bellman-Ford / NBFNet style); the datalog kernel supplies sound labels + per-hop depth as
dense per-step supervision; depth-generalization is the honest test of learned reasoning vs memorizing.

    python -m thinking.reasoner --selftest
    python -m thinking.reasoner --n-ent 16 --k-train 4 --max-depth 8 --steps 4000 --seeds 3
"""
import argparse
import sys
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROLE = {"pad": 0, "edge_src": 1, "edge_dst": 2, "qsrc": 3, "qdst": 4}


# ===================================================================================================
# Episodes: random is-a graphs; the KERNEL (datalog transitive closure) is the TEACHER (sound labels),
# BFS gives the per-hop depth schedule. Entity ids are PERMUTED per episode so identity carries no
# signal -- the model must reason over STRUCTURE, not memorize which slot is the root.
# ===================================================================================================
def gen_graph(rng, n_ent, chain_p=0.6):
    """Random forest of is-a edges (child, parent). chain_p biases toward long chains (deep queries)."""
    edges = []
    for i in range(1, n_ent):
        p = i - 1 if rng.random() < chain_p else int(rng.integers(0, i))
        edges.append((i, p))                                 # i isa p
    return edges


_RULES = [(("isa", ("?x", "?z")), [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))])]   # transitivity


def kernel_closure(edges):
    """The TEACHER: datalog transitive closure (sound). Returns the set of entailed isa pairs."""
    import datalog as D
    facts = [("isa", (a, b)) for (a, b) in edges]
    known, _ = D.Datalog(_RULES).closure(facts)
    return {a for a in known if a[0] == "isa"}


def bfs_dist(edges, src):
    g = defaultdict(list)
    for a, b in edges:
        g[a].append(b)
    dist, dq = {src: 0}, deque([src])
    while dq:
        u = dq.popleft()
        for v in g[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def gen_episode(rng, n_ent, depth_lo, depth_hi, chain_p=0.6):
    """One (edges, query, label, hops). Query is a POSITIVE at hop-distance in [lo,hi] (~half the time)
    or a NEGATIVE (non-ancestor). Entity ids permuted so the model can't memorize identities."""
    edges = gen_graph(rng, n_ent, chain_p)
    perm = rng.permutation(n_ent)
    edges = [(int(perm[a]), int(perm[b])) for (a, b) in edges]
    closure = kernel_closure(edges)                          # KERNEL = teacher (sound labels)
    # pick a source with descendants and gather candidates by hop distance
    want_pos = rng.random() < 0.5
    srcs = list({a for (a, _b) in edges})
    rng.shuffle(srcs)
    for s in srcs:
        dist = bfs_dist(edges, s)
        if want_pos:
            cands = [t for t, d in dist.items() if depth_lo <= d <= depth_hi]
            if cands:
                qd = int(rng.choice(cands))
                assert ("isa", (s, qd)) in closure           # teacher agrees with BFS reachability
                return edges, (s, qd), 1, dist[qd]
        else:
            non = [t for t in range(n_ent) if t != s and t not in dist]
            if non:
                qd = int(rng.choice(non))
                assert ("isa", (s, qd)) not in closure
                return edges, (s, qd), 0, 0
    return gen_episode(rng, n_ent, depth_lo, depth_hi, chain_p)   # retry on a degenerate draw


def encode(episode, max_len):
    """(edges, query) -> token arrays. Each edge = [src, dst]; query = [qsrc, qdst] at the end."""
    edges, (qs, qd), label, hops = episode
    ent, role = [], []
    for (a, b) in edges:
        ent += [a, b]; role += [ROLE["edge_src"], ROLE["edge_dst"]]
    qpos = (len(ent), len(ent) + 1)
    ent += [qs, qd]; role += [ROLE["qsrc"], ROLE["qdst"]]
    L = len(ent)
    pad = max_len - L
    ent += [0] * pad; role += [ROLE["pad"]] * pad
    mask = [False] * L + [True] * pad                        # True = padding (ignored by attention)
    return ent, role, qpos, mask, label, hops


def make_batch(rng, n_ent, depth_lo, depth_hi, bs, chain_p=0.6):
    eps = [gen_episode(rng, n_ent, depth_lo, depth_hi, chain_p) for _ in range(bs)]
    max_len = max(2 * len(e[0]) + 2 for e in eps)
    ent, role, qpos, mask, label, hops = zip(*[encode(e, max_len) for e in eps])
    return (torch.tensor(ent), torch.tensor(role), torch.tensor(qpos),
            torch.tensor(mask), torch.tensor(label, dtype=torch.float32), torch.tensor(hops))


# ===================================================================================================
# The recurrent reasoner: ONE weight-shared block, unrolled T steps with INPUT INJECTION. A readout at
# every step gives DENSE per-step supervision: "after t iterations, is q derivable within t hops?"
# -> the model learns to do exactly one hop of deduction per iteration, so running MORE steps at test
# time reasons DEEPER (the learned analog of iterating the closure to fixpoint).
# ===================================================================================================
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, kpm):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, key_padding_mask=kpm, need_weights=False)[0]
        x = x + self.ff(self.ln2(x))
        return x


class Reasoner(nn.Module):
    def __init__(self, n_ent, d=128, h=4):
        super().__init__()
        self.ent = nn.Embedding(n_ent, d)
        self.role = nn.Embedding(len(ROLE), d)
        self.block = Block(d, h)
        self.read = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.d = d

    def forward(self, ent, role, qpos, mask, T):
        x0 = self.ent(ent) + self.role(role)                 # (B,L,d)
        h = x0
        b = torch.arange(ent.size(0), device=ent.device)
        outs = []
        for _t in range(T):
            h = self.block(h + x0, mask)                     # INPUT INJECTION: re-add the problem
            qs = h[b, qpos[:, 0]]
            qd = h[b, qpos[:, 1]]
            outs.append(self.read(torch.cat([qs, qd], -1)).squeeze(-1))
        return torch.stack(outs, 1)                          # (B,T) per-step entailment logits


def step_targets(label, hops, T):
    """y_t = 1 iff the query is entailed AND derivable within t hops (t=1..T); negatives are 0 always."""
    t = torch.arange(1, T + 1, device=label.device).unsqueeze(0)         # (1,T)
    return (label.unsqueeze(1) * (t >= hops.unsqueeze(1)).float())       # (B,T)


# ===================================================================================================
def train(model, rng, n_ent, k_train, T_train, steps, device, bs=64, lr=1.2e-3, warmup_frac=0.1, log=0):
    import math
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(steps * warmup_frac))

    def lr_at(s):                                            # linear warmup -> cosine decay (fixes the
        if s < warm:                                        # seed-fragile cold-start stall we saw)
            return lr * (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ent, role, qpos, mask, label, hops = make_batch(rng, n_ent, 1, k_train, bs)
        ent, role, qpos, mask, label, hops = [t.to(device) for t in (ent, role, qpos, mask, label, hops)]
        logits = model(ent, role, qpos, mask, T_train)       # (B,T)
        y = step_targets(label, hops, T_train)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and (step % log == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {loss.item():.4f}  lr {lr_at(step):.2e}", flush=True)


@torch.no_grad()
def evaluate(model, rng, n_ent, buckets, T_eval, device, n=400):
    """Accuracy of the UNAIDED model (last-step readout) per hop-depth bucket; chance ~0.5 (balanced)."""
    model.eval()
    out = {}
    for (lo, hi) in buckets:
        correct = tot = 0
        while tot < n:
            ent, role, qpos, mask, label, hops = make_batch(rng, n_ent, lo, hi, min(128, n))
            ent, role, qpos, mask = [t.to(device) for t in (ent, role, qpos, mask)]
            pred = (model(ent, role, qpos, mask, T_eval)[:, -1] > 0).float().cpu()
            correct += (pred == label).sum().item(); tot += len(label)
        out[(lo, hi)] = correct / tot
    return out


def run(n_ent=16, k_train=4, max_depth=8, d=128, T_train=8, T_eval=14, steps=4000, seeds=1,
        device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    buckets = [(1, k_train), (k_train + 1, max_depth)]       # SEEN depths vs HELD-OUT deeper
    rows = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = Reasoner(n_ent, d=d).to(device)
        train(model, rng, n_ent, k_train, T_train, steps, device,
              log=(max(1, steps // 5) if verbose else 0))
        acc = evaluate(model, rng, n_ent, buckets, T_eval, device)
        rows.append(acc)
        if verbose:
            seen = acc[(1, k_train)]; gen = acc[(k_train + 1, max_depth)]
            print(f"  seed {seed}: SEEN depth<={k_train} acc {seen:.3f} | "
                  f"HELD-OUT depth {k_train+1}-{max_depth} acc {gen:.3f}  "
                  f"(unaided model vs kernel, T_eval={T_eval})", flush=True)
    seen = float(np.mean([r[(1, k_train)] for r in rows]))
    gen = float(np.mean([r[(k_train + 1, max_depth)] for r in rows]))
    if verbose:
        print(f"\n  MEAN over {seeds} seed(s): SEEN {seen:.3f} | HELD-OUT-DEPTH {gen:.3f} | chance 0.5",
              flush=True)
        print("  -> learned to reason iff HELD-OUT-DEPTH accuracy stays high (generalizes past trained "
              "hops with NO kernel call at inference)", flush=True)
    return dict(seen=seen, generalize=gen, n_ent=n_ent, k_train=k_train, max_depth=max_depth,
                T_train=T_train, T_eval=T_eval, steps=steps, seeds=seeds)


def selftest():
    """CPU-only, tiny: the teacher labels are sound, the model learns trained-depth deduction."""
    rng = np.random.default_rng(0)
    # teacher soundness: BFS reachability == datalog closure on a few graphs
    for _ in range(20):
        edges = gen_graph(rng, 7)
        clo = kernel_closure(edges)
        for s in {a for (a, _b) in edges}:
            dist = bfs_dist(edges, s)
            for t in range(7):
                assert (("isa", (s, t)) in clo) == (t in dist and dist[t] > 0), \
                    "teacher (datalog) disagrees with BFS reachability"
    r = run(n_ent=7, k_train=3, max_depth=5, d=32, T_train=6, T_eval=8, steps=500, seeds=1,
            device="cpu", verbose=False)
    print(f"  selftest: SEEN depth<=3 acc {r['seen']:.3f} (chance 0.5) | "
          f"HELD-OUT depth 4-5 acc {r['generalize']:.3f}")
    assert r["seen"] > 0.8, f"failed to learn trained-depth deduction: {r['seen']:.3f}"
    print("reasoner selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-ent", type=int, default=16)
    ap.add_argument("--k-train", type=int, default=4, help="max hop-depth seen in training")
    ap.add_argument("--max-depth", type=int, default=8, help="max hop-depth tested (held-out > k-train)")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--t-train", type=int, default=8, help="recurrent unroll steps in training")
    ap.add_argument("--t-eval", type=int, default=14, help="recurrent steps at test (more = deeper)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    res = run(n_ent=args.n_ent, k_train=args.k_train, max_depth=args.max_depth, d=args.d,
              T_train=args.t_train, T_eval=args.t_eval, steps=args.steps, seeds=args.seeds,
              device=args.device)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
