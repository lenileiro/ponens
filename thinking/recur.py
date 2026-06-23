#!/usr/bin/env python3
"""recur -- WEIGHT-TIED INTERNAL RECURRENCE, a SECOND reasoning mode alongside the external-execution loop
(azr_win). Following "Looped Transformers are Better at Learning Learning Algorithms" (arxiv 2311.12424) and
our own computation-lengthgen finding (input injection is decisive): ONE transformer block is applied T times
to an internal latent with INPUT INJECTION (h <- block(h + x), x = the task encoding re-fed every iteration),
then the program is decoded from the final latent. The "thinking" is the internal recurrence -- iterate the
SAME weights to think longer -- rather than the per-op executor loop. Trained with a curriculum over the loop
count T (so it works at any T); the executor VERIFIES the final program (sound) and best-of-N keeps any
verified one.

Contrast of the two modes (both end in the same sound verifier):
  * EXTERNAL loop (azr_win): propose op -> execute -> re-encode. State lives in the world; sound per-step.
  * INTERNAL loop (this):    iterate a tied block on a latent (input injection); decode whole program; verify.
The looped-transformer claim we test: MORE test-time loops -> better composition, and train-shallow loops can
generalize deeper (think longer for harder tasks).

    python -m thinking.recur --selftest
    python -m thinking.recur --maxdepth 3 --concepts 6 --rounds 4000
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinking.azr import NBASE, NOP, execute
from thinking.azr_struct import make_concepts, gen_task_struct, _IDLIB
from thinking.reasoner import Block
from thinking.diffuse import gen_batch, verify


class LoopedReasoner(nn.Module):
    """Encode the demos + D program slots once -> x; iterate ONE tied block T times with input injection;
    decode the op at every slot from the final latent."""
    def __init__(self, A, L, D, d=128, h=4):
        super().__init__()
        self.A, self.L, self.D = A, L, D
        self.val = nn.Embedding(A, d)
        self.drole = nn.Embedding(2, d)
        self.dpos = nn.Embedding(L, d)
        self.slot0 = nn.Parameter(torch.zeros(1, 1, d))
        self.spos = nn.Embedding(D, d)
        self.block = Block(d, h)                              # ONE tied block, reused every loop
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, NBASE))

    def encode(self, di, do):
        B, K, L = di.shape
        ar = torch.arange(L, device=di.device)
        din = self.val(di) + self.drole.weight[0] + self.dpos(ar)
        dou = self.val(do) + self.drole.weight[1] + self.dpos(ar)
        demo = torch.cat([din, dou], 2).reshape(B, K * 2 * L, -1)
        sl = (self.slot0 + self.spos(torch.arange(self.D, device=di.device))).expand(B, -1, -1)
        return torch.cat([demo, sl], 1)                      # (B, 2KL + D, d)

    def forward(self, di, do, T):
        x = self.encode(di, do)
        h = torch.zeros_like(x)
        for _ in range(max(1, T)):
            h = self.block(h + x, None)                      # INPUT INJECTION + weight-tied recurrence
        return self.head(h[:, -self.D:])                     # (B, D, NBASE)


@torch.no_grad()
def decode(model, di, do, T, temp=0.0):
    logits = model(di, do, T)
    if temp > 0:
        probs = F.softmax(logits / temp, -1)
        B, D, _ = probs.shape
        return torch.multinomial(probs.reshape(-1, NBASE), 1).reshape(B, D)
    return logits.argmax(-1)


def evaluate(model, rng, concepts, A, L, K, depths, D, device, T, samples=1, temp=0.9, n=150):
    model.eval()
    out = {}
    for d in depths:
        ok = 0
        for _ in range(n):
            _p, demos, _xq, _yq = gen_task_struct(rng, concepts, A, L, K, d)
            di = torch.tensor([[x for (x, _y) in demos]], device=device)
            do = torch.tensor([[y for (_x, y) in demos]], device=device)
            for s in range(samples):
                prog = decode(model, di, do, T, temp=(0.0 if s == 0 else temp))[0]
                if verify(prog, demos, A):
                    ok += 1; break
        out[d] = ok / n
    return out


def selfplay(A=6, L=4, K=4, n_concepts=6, maxdepth=3, clen=2, d=128, rounds=4000, bs=64, Tmax=None,
             lr=1.5e-3, curriculum=False, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0); torch.manual_seed(0)
    concepts = make_concepts(rng, n_concepts, A, L, clen=clen)
    D = maxdepth * clen
    Tmax = Tmax or (maxdepth + 2)
    model = LoopedReasoner(A, L, D, d=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for rnd in range(rounds):
        di, do, prog = gen_batch(rng, concepts, A, L, K, 1, maxdepth, D, bs, device)
        # FIXED T forces the model to actually USE the recurrence (random/curriculum T incl. T=1 lets it learn
        # a 1-shot solution that ignores extra loops -> the block collapses to ~idempotent).
        T = int(rng.integers(2, Tmax + 1)) if curriculum else Tmax
        model.train()
        logits = model(di, do, T)
        loss = F.cross_entropy(logits.reshape(-1, NBASE), prog.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if verbose and rnd % max(1, rounds // 8) == 0:
            acc = evaluate(model, rng, concepts, A, L, K, range(1, maxdepth + 1), D, device, Tmax, n=80)
            print(f"  r{rnd:5d} (T={Tmax}): " + " ".join(f"d{k}:{v:.2f}" for k, v in acc.items()), flush=True)
    return model, concepts, D, Tmax


def selftest():
    model, concepts, D, Tmax = selfplay(A=5, L=4, K=4, n_concepts=3, maxdepth=2, d=64, rounds=1500,
                                        bs=48, device="cpu", verbose=False)
    rng = np.random.default_rng(7)
    one = evaluate(model, rng, concepts, 5, 4, 4, [2], D, "cpu", T=1, samples=12, n=200)[2]
    many = evaluate(model, rng, concepts, 5, 4, 4, [2], D, "cpu", T=2 * Tmax, samples=12, n=200)[2]
    print(f"  selftest: depth-2 best-of-12  T=1:{one:.2f}  T={2 * Tmax}:{many:.2f}  "
          f"(weight-tied internal recurrence; THINKING LONGER via more loops helps)")
    assert many > 0.4, f"looped reasoner failed to compose depth-2 ({many:.2f})"
    assert many >= one + 0.05, f"more internal loops should help ({one:.2f}->{many:.2f})"
    print("recur selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--A", type=int, default=6); ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--concepts", type=int, default=6)
    ap.add_argument("--maxdepth", type=int, default=3); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=4000); ap.add_argument("--Tmax", type=int, default=None)
    ap.add_argument("--device", default=None); ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    model, concepts, D, Tmax = selfplay(A=args.A, L=args.L, K=args.K, n_concepts=args.concepts,
                                        maxdepth=args.maxdepth, d=args.d, rounds=args.rounds,
                                        Tmax=args.Tmax, device=args.device)
    dev = next(model.parameters()).device
    rng = np.random.default_rng(7)
    depths = range(1, args.maxdepth + 1)
    print("  test-time loop scaling (best-of-16, verifier-sound):", flush=True)
    for T in (1, 2, Tmax, Tmax * 2):                         # more loops at test time = think longer
        ev = evaluate(model, rng, concepts, args.A, args.L, args.K, depths, D, dev, T=T, samples=16)
        print(f"    T={T:<3}: " + " ".join(f"d{k}:{v:.3f}" for k, v in ev.items()), flush=True)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({"Tmax": Tmax, "scaling": {str(T): evaluate(
                model, rng, concepts, args.A, args.L, args.K, depths, D, dev, T=T, samples=16)
                for T in (1, Tmax, Tmax * 2)}}, f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
