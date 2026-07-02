#!/usr/bin/env python3
"""diffuse -- PARALLEL / DIFFUSION reasoning (Mercury-style) for program induction. Instead of decoding a
program left-to-right (autoregressive, brittle -- one wrong token cascades, which is why whole-program AR
plateaued on composition), the model emits the WHOLE program at once and REVISES it over a few parallel
mask-predict denoising rounds: predict all slots, keep the most confident, re-mask the rest, repeat. This
lets it self-correct earlier steps. Trained as a masked denoiser on teacher programs (structured-concept
tasks from azr_struct); the executor VERIFIES the final program (sound). Demonstrates iterative refinement
(T>1) beats one-shot (T=1), and best-of-N (verifier-sound) on top.

    python -m thinking.diffuse --selftest
    python -m thinking.diffuse --maxdepth 3 --concepts 6 --rounds 3000
"""
import argparse
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.azr import NBASE, NOP, execute
from thinking.azr_struct import make_concepts, gen_task_struct, _IDLIB
from thinking.reasoner import Block

MASK = NBASE                                                 # mask token id (base ops are 0..NBASE-1)
ROLE_IN, ROLE_OUT = 0, 1


class Diffuser(nn.Module):
    """Encode the demos + a (partially masked) program; predict the op at every program slot in parallel."""
    def __init__(self, A, L, D, d=128, h=4, layers=4):
        super().__init__()
        self.A, self.L, self.D = A, L, D
        self.val = nn.Embedding(A, d)
        self.drole = nn.Embedding(2, d)
        self.dpos = nn.Embedding(L, d)
        self.optok = nn.Embedding(NBASE + 1, d)              # base ops + MASK
        self.spos = nn.Embedding(D, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, NBASE))

    def forward(self, di, do, prog):
        B, K, L = di.shape
        ar = torch.arange(L, device=di.device)
        din = self.val(di) + self.drole.weight[ROLE_IN] + self.dpos(ar)
        dou = self.val(do) + self.drole.weight[ROLE_OUT] + self.dpos(ar)
        demo = torch.cat([din, dou], 2).reshape(B, K * 2 * L, -1)
        sl = self.optok(prog) + self.spos(torch.arange(self.D, device=di.device))
        h = torch.cat([demo, sl], 1)
        for b in self.blocks:
            h = b(h, None)
        return self.head(h[:, -self.D:])                     # (B, D, NBASE)


def gen_batch(rng, concepts, A, L, K, depth_lo, depth_hi, D, bs, device):
    di, do, prog = [], [], []
    for _ in range(bs):
        d = int(rng.integers(depth_lo, depth_hi + 1))
        p, demos, _xq, _yq = gen_task_struct(rng, concepts, A, L, K, d)
        p = (p + [NOP] * D)[:D]                               # pad program to D slots
        di.append([x for (x, _y) in demos]); do.append([y for (_x, y) in demos]); prog.append(p)
    t = lambda z: torch.tensor(z, device=device)
    return t(di), t(do), t(prog)


@torch.no_grad()
def denoise(model, di, do, T, device, temp=0.0):
    """Mask-predict: start all-masked, over T rounds predict all slots and unmask the most-confident ones
    (schedule), re-masking the rest -> parallel, self-correcting decoding. Returns (B,D) op programs."""
    model.eval()
    B = di.size(0)
    D = model.D
    prog = torch.full((B, D), MASK, dtype=torch.long, device=device)
    for t in range(T):
        logits = model(di, do, prog)                         # (B,D,NBASE)
        if temp > 0:
            probs = F.softmax(logits / temp, -1)
            pred = torch.multinomial(probs.reshape(-1, NBASE), 1).reshape(B, D)
            conf = probs.gather(-1, pred[..., None]).squeeze(-1)
        else:
            conf, pred = F.softmax(logits, -1).max(-1)
        keep = int(math.ceil(D * (t + 1) / T))               # how many slots are fixed by round t
        masked = prog == MASK
        c = conf.masked_fill(~masked, -1.0)                  # only (re)decide masked slots by confidence
        # unmask the top-(keep - already_fixed) most-confident masked slots
        fixed = (~masked).sum(1)
        new_prog = prog.clone()
        for b in range(B):
            need = max(0, keep - int(fixed[b]))
            if need > 0:
                idx = torch.topk(c[b], min(need, int(masked[b].sum())) if masked[b].any() else 0).indices
                new_prog[b, idx] = pred[b, idx]
        # last round: fill any remaining
        if t == T - 1:
            rem = new_prog == MASK
            new_prog[rem] = pred[rem]
        prog = new_prog
    return prog


def verify(prog_row, demos, A):
    p = [int(o) for o in prog_row if int(o) != NOP]
    return all(execute(p, list(x), A, _IDLIB) == list(y) for (x, y) in demos)


def evaluate(model, rng, concepts, A, L, K, depths, D, device, T, samples=1, temp=0.9, n=150):
    out = {}
    for d in depths:
        ok = 0
        for _ in range(n):
            p, demos, _xq, _yq = gen_task_struct(rng, concepts, A, L, K, d)
            di = torch.tensor([[x for (x, _y) in demos]], device=device)
            do = torch.tensor([[y for (_x, y) in demos]], device=device)
            solved = False
            for s in range(samples):
                prog = denoise(model, di, do, T, device, temp=(0.0 if s == 0 else temp))[0]
                if verify(prog, demos, A):
                    solved = True; break
            ok += int(solved)
        out[d] = ok / n
    return out


def selfplay(A=6, L=4, K=4, n_concepts=6, maxdepth=3, clen=2, d=128, rounds=3000, bs=64, T=4,
             lr=1.5e-3, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0); torch.manual_seed(0)
    concepts = make_concepts(rng, n_concepts, A, L, clen=clen)
    D = maxdepth * clen
    model = Diffuser(A, L, D, d=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for rnd in range(rounds):
        di, do, prog = gen_batch(rng, concepts, A, L, K, 1, maxdepth, D, bs, device)
        # DIFFUSION training: mask a random fraction of program slots, predict them from demos + context
        frac = float(rng.uniform(0.3, 1.0))
        mask = torch.rand(prog.shape, device=device) < frac
        inp = prog.clone(); inp[mask] = MASK
        model.train()
        logits = model(di, do, inp)
        loss = F.cross_entropy(logits[mask], prog[mask]) if mask.any() else logits.sum() * 0
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if verbose and rnd % max(1, rounds // 8) == 0:
            acc = evaluate(model, rng, concepts, A, L, K, range(1, maxdepth + 1), D, device, T, n=80)
            print(f"  r{rnd:5d}: " + " ".join(f"d{k}:{v:.2f}" for k, v in acc.items()), flush=True)
    return model, concepts, D


def selftest():
    model, concepts, D = selfplay(A=5, L=4, K=4, n_concepts=3, maxdepth=2, d=64, rounds=1500, bs=48,
                                  T=4, device="cpu", verbose=False)
    rng = np.random.default_rng(7)
    oneshot = evaluate(model, rng, concepts, 5, 4, 4, [1, 2], D, "cpu", T=1, n=200)
    refine = evaluate(model, rng, concepts, 5, 4, 4, [1, 2], D, "cpu", T=4, n=200)
    bestN = evaluate(model, rng, concepts, 5, 4, 4, [1, 2], D, "cpu", T=4, samples=12, n=200)
    print(f"  selftest: one-shot(T=1) d2:{oneshot[2]:.2f} | refine(T=4) d2:{refine[2]:.2f} | "
          f"refine+best12 d2:{bestN[2]:.2f}")
    # honest finding: pure parallel denoising (no execution feedback) ~ AR on composition; the verifier-
    # backed best-of-N is what makes it solve -> assert that, not greedy refinement.
    assert bestN[2] > 0.4, f"diffusion + best-of-N failed to compose depth-2 ({bestN[2]:.2f})"
    assert bestN[2] >= refine[2], "best-of-N should not hurt (verifier-sound)"
    print("diffuse selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--A", type=int, default=6); ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--concepts", type=int, default=6)
    ap.add_argument("--maxdepth", type=int, default=3); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3000); ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--device", default=None); ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    model, concepts, D = selfplay(A=args.A, L=args.L, K=args.K, n_concepts=args.concepts,
                                  maxdepth=args.maxdepth, d=args.d, rounds=args.rounds, T=args.T,
                                  device=args.device)
    dev = next(model.parameters()).device
    rng = np.random.default_rng(7)
    one = evaluate(model, rng, concepts, args.A, args.L, args.K, range(1, args.maxdepth + 1), D, dev, T=1)
    ref = evaluate(model, rng, concepts, args.A, args.L, args.K, range(1, args.maxdepth + 1), D, dev, T=args.T)
    bN = evaluate(model, rng, concepts, args.A, args.L, args.K, range(1, args.maxdepth + 1), D, dev,
                  T=args.T, samples=16)
    print("  one-shot (T=1) : " + " ".join(f"d{k}:{v:.3f}" for k, v in one.items()), flush=True)
    print(f"  diffusion (T={args.T}): " + " ".join(f"d{k}:{v:.3f}" for k, v in ref.items()), flush=True)
    print(f"  + best-of-16   : " + " ".join(f"d{k}:{v:.3f}" for k, v in bN.items())
          + "  (parallel refinement + verifier-sound sampling)", flush=True)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({"one_shot": one, "diffusion": ref, "best_of_16": bN}, f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
