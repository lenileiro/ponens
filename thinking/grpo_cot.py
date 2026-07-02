#!/usr/bin/env python3
"""grpo_cot -- CHAIN-OF-THOUGHT TRAINED BY RL (GRPO / Dr.GRPO) with a SOUND verifier as the reward.

The model (prove.Prover) emits a chain-of-thought: an ordered sequence of derived atoms ending at the goal --
its proof. We do NOT imitate teacher proofs here (that's SFT, prove.py). Instead the policy SAMPLES G whole
proofs per goal, the KERNEL verifies each (verify_proof -> 1/0, sound -> a wrong chain scores 0, never a false
positive), and we do a group-relative policy-gradient update (GRPO): advantage = reward - group_mean (Dr.GRPO:
no std/length normalization, which removes the length & difficulty bias). High-reward chains are up-weighted,
low-reward chains down-weighted. The verifier reward is FREE and DENSE-enough at the outcome level.

Why this matters beyond SFT: RL needs NO labels -- only the (sound) verifier -- so it can train on goals that
have NO teacher proof at all. We warm-start a weak policy with brief SFT (cold-start: GRPO needs the policy to
hit nonzero reward to get gradient), then GRPO LIFTS the unaided proof rate, including on HELD-OUT goals the
SFT split never saw. Bar: SFT+GRPO model-proves-alone (held-out) > SFT-only, 0 unsound (kernel-gated).

    python -m thinking.grpo_cot --selftest
    python -m thinking.grpo_cot --wordnet --cap 120
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import thinking.lota_kernel as LK
from thinking.azr_meaning import RULES
from thinking.azr_kmeaning import PREDS, PID, wordnet_kb, full_closure, goals_for, synthetic_kb
from thinking.prove import (Prover, NPRED, encode, verify_proof, generate, evaluate, proof_seq)

PNAMES = ("isa", "has_part", "member_of")


# ===================================================================================================
# SFT warm-start (brief) -- gives the policy a nonzero base rate so GRPO has gradient signal.
# ===================================================================================================
def sft_warmstart(model, train_goals, facts, env, eid, device, rounds=300, bs=48, lr=1.2e-3, maxlen=10):
    closure = full_closure(facts, env)
    data = []
    for (c, g) in train_goals:
        if g not in closure:
            continue
        seq = proof_seq(closure[g])
        if seq and len(seq) <= maxlen:
            data.append((encode(g, eid), [encode(a, eid) for a in seq]))
    if not data:
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(0)
    for rnd in range(rounds):
        idx = rng.integers(0, len(data), size=min(bs, len(data)))
        batch = [data[i] for i in idx]
        T = max(len(s) for _g, s in batch)
        goal = torch.tensor([g for g, _s in batch], device=device)
        steps = torch.zeros((len(batch), T, 3), dtype=torch.long, device=device)
        tp = torch.full((len(batch), T + 1), NPRED, dtype=torch.long, device=device)
        te1 = torch.zeros((len(batch), T + 1), dtype=torch.long, device=device)
        te2 = torch.zeros((len(batch), T + 1), dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), T + 1), dtype=torch.bool, device=device)
        for b, (_g, s) in enumerate(batch):
            for t, a in enumerate(s):
                steps[b, t] = torch.tensor(a, device=device)
                tp[b, t] = a[0]; te1[b, t] = a[1]; te2[b, t] = a[2]; mask[b, t] = True
            mask[b, len(s)] = True
        model.train()
        lp, le1, le2 = model(goal, steps)
        loss = F.cross_entropy(lp[mask], tp[mask])
        am = mask & (tp != NPRED)
        if am.any():
            loss = loss + F.cross_entropy(le1[am], te1[am]) + F.cross_entropy(le2[am], te2[am])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


# ===================================================================================================
# GRPO rollout: SAMPLE G proof chains for one goal, score each with the kernel verifier.
# ===================================================================================================
@torch.no_grad()
def sample_chain(model, goal, eid, inv, device, maxlen, temp):
    """Sample ONE proof chain from the policy (stochastic). Returns (atoms_enc, stopped) where atoms_enc is
    a list of [pred,e1,e2] and stopped indicates a STOP action was emitted (vs hit maxlen)."""
    g = torch.tensor([encode(goal, eid)], device=device)
    steps = torch.zeros((1, 0, 3), dtype=torch.long, device=device)
    atoms, stopped = [], False
    for _ in range(maxlen):
        lp, le1, le2 = model(g, steps)
        p = int(torch.distributions.Categorical(logits=lp[0, -1] / temp).sample())
        if p == NPRED:
            stopped = True
            break
        e1 = int(torch.distributions.Categorical(logits=le1[0, -1] / temp).sample())
        e2 = int(torch.distributions.Categorical(logits=le2[0, -1] / temp).sample())
        atoms.append([p, e1, e2])
        steps = torch.cat([steps, torch.tensor([[[p, e1, e2]]], device=device)], 1)
    return atoms, stopped


def chain_reward(atoms, inv, facts, env, goal):
    """SOUND reward: the kernel checks the emitted chain. 1.0 iff every step is derivable AND the goal is
    reached -- a bogus chain scores 0 (no false positive)."""
    seq = [(PNAMES[a[0]], (inv[a[1]], inv[a[2]])) for a in atoms]
    return 1.0 if verify_proof(seq, facts, env, goal) else 0.0


def chain_logp(model, goal, atoms, stopped, eid, device):
    """logp of a sampled chain under the CURRENT policy (teacher-forced recompute). Includes the terminal
    STOP action's logp when the chain stopped."""
    g = torch.tensor([encode(goal, eid)], device=device)
    n = len(atoms)
    steps = torch.tensor([atoms], dtype=torch.long, device=device) if n else \
        torch.zeros((1, 0, 3), dtype=torch.long, device=device)
    lp, le1, le2 = model(g, steps)                            # (1, n+1, *)
    lsp = F.log_softmax(lp[0], -1); ls1 = F.log_softmax(le1[0], -1); ls2 = F.log_softmax(le2[0], -1)
    total = g.new_zeros((), dtype=torch.float)
    for t, a in enumerate(atoms):                            # each atom: pred + e1 + e2
        total = total + lsp[t, a[0]] + ls1[t, a[1]] + ls2[t, a[2]]
    if stopped:                                              # terminal STOP action at position n
        total = total + lsp[n, NPRED]
    return total


def grpo_finetune(model, train_goals, facts, env, eid, inv, device, steps=400, goals_per=8, G=8,
                  lr=4e-4, temp=1.0, maxlen=10, test_goals=None, verbose=True, log_every=None):
    """GRPO / Dr.GRPO: for each goal, sample G chains, reward = kernel-verified (1/0), advantage =
    reward - group_mean (no std normalization). Policy-gradient update up-weights verified chains. Uses ONLY
    the verifier -- no teacher proofs -- so it learns on goals with no labelled proof."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(1)
    log_every = log_every or max(1, steps // 6)
    for it in range(steps):
        gidx = rng.integers(0, len(train_goals), size=min(goals_per, len(train_goals)))
        batch_loss = torch.zeros((), device=device)
        nseen = 0
        solved_any = 0
        for gi in gidx:
            _c, goal = train_goals[gi]
            model.eval()
            chains = [sample_chain(model, goal, eid, inv, device, maxlen, temp) for _ in range(G)]
            rewards = np.array([chain_reward(a, inv, facts, env, goal) for (a, _s) in chains])
            solved_any += int(rewards.max() > 0)
            if rewards.std() < 1e-6:                          # all same reward -> no relative signal, skip
                continue
            adv = rewards - rewards.mean()                    # Dr.GRPO: mean-baseline, no std/length norm
            model.train()
            for (atoms, stopped), a in zip(chains, adv):
                if abs(a) < 1e-6:
                    continue
                lp = chain_logp(model, goal, atoms, stopped, eid, device)
                batch_loss = batch_loss - a * lp              # -advantage * logp
                nseen += 1
        if nseen > 0:
            loss = batch_loss / nseen
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if verbose and (it % log_every == 0 or it == steps - 1):
            msg = f"  grpo it{it:4d}: rollout-solve {solved_any}/{len(gidx)}"
            if test_goals is not None:
                ev = evaluate(model, test_goals, eid, inv, facts, env, device, n=len(test_goals))
                msg += f" | HELD-OUT proves-alone {ev['proved']:.3f}"
            print(msg, flush=True)
    return model


def split_goals(goalpool, frac=0.6, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(goalpool))
    k = int(len(goalpool) * frac)
    return [goalpool[i] for i in idx[:k]], [goalpool[i] for i in idx[k:]]


def run(facts, ents, concepts, d=128, sft_rounds=300, grpo_steps=400, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eid = {e: i for i, e in enumerate(ents)}
    inv = {i: e for e, i in eid.items()}
    env = LK.build_env(ents, PREDS, facts, RULES)
    goalpool = [(c, g) for c in concepts for g in goals_for(facts, c)]
    train_goals, test_goals = split_goals(goalpool, frac=0.6)
    torch.manual_seed(0)
    model = Prover(len(ents), d=d).to(device)
    if verbose:
        print(f"  {len(goalpool)} goals -> SFT-train {len(train_goals)} / HELD-OUT {len(test_goals)}", flush=True)
    sft_warmstart(model, train_goals, facts, env, eid, device, rounds=sft_rounds)
    base = evaluate(model, test_goals, eid, inv, facts, env, device, n=len(test_goals))
    if verbose:
        print(f"  after SFT warm-start: HELD-OUT proves-alone {base['proved']:.3f}", flush=True)
    grpo_finetune(model, train_goals, facts, env, eid, inv, device, steps=grpo_steps,
                  test_goals=test_goals, verbose=verbose)
    after = evaluate(model, test_goals, eid, inv, facts, env, device, n=len(test_goals))
    return base["proved"], after["proved"], model


def selftest():
    rng = np.random.default_rng(0)
    facts, ents, concepts = synthetic_kb(rng, depth=6, sibs=3, parts=4)
    base, after, _m = run(facts, ents, concepts, d=96, sft_rounds=250, grpo_steps=250,
                          device="cpu", verbose=False)
    print(f"  selftest: HELD-OUT proves-alone  SFT {base:.3f} -> SFT+GRPO {after:.3f}  (0 unsound, kernel-gated)")
    assert after >= base + 0.02, f"GRPO (CoT-by-RL) did not lift the held-out proof rate ({base:.3f}->{after:.3f})"
    print("grpo_cot selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--wordnet", action="store_true"); ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--sft-rounds", type=int, default=300); ap.add_argument("--grpo-steps", type=int, default=400)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.wordnet:
        facts, ents, concepts = wordnet_kb(cap=args.cap)
        print(f"  REAL WordNet KB: {len(facts)} facts, {len(ents)} concepts", flush=True)
    else:
        facts, ents, concepts = synthetic_kb(np.random.default_rng(0))
    base, after, _m = run(facts, ents, concepts, d=args.d, sft_rounds=args.sft_rounds,
                          grpo_steps=args.grpo_steps, device=args.device)
    print(f"  FINAL: HELD-OUT proves-alone  SFT {base:.3f} -> SFT+GRPO {after:.3f}  "
          f"(CoT trained by RL, kernel-verified, 0 unsound)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
