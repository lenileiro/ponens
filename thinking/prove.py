#!/usr/bin/env python3
"""prove -- THE MODEL ALONE PROVES (the kernel only verifies). Unlike azr_kmeaning/azr_kproof, where a
symbolic search engine does the derivation and the model only scores candidates, here the model
AUTOREGRESSIVELY GENERATES the entire proof -- the ordered sequence of derived atoms ending at the goal --
and the kernel checks each emitted step is soundly derivable. The model is the prover; the kernel is the
sound gate. A bogus emitted atom is rejected, so it can never assert the unprovable (0 unsound), but every
ACCEPTED proof was produced by the model on its own.

Trained on the complete prover's proofs (kernel-verified), it learns to emit valid derivations; at test it
proves UNAIDED (greedy generation, no search). Bar: high model-proves-alone rate over real WordNet meaning,
UNSOUND 0.

    python -m thinking.prove --selftest
    python -m thinking.prove --wordnet --cap 120
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import thinking.lota_kernel as LK
from thinking.azr_meaning import RULES, base_tree, kernel_verify, _match_body, _inst
from thinking.azr_kmeaning import PREDS, PID, wordnet_kb, full_closure, tree_atoms, goals_for, synthetic_kb

NPRED = len(PID)


def proof_seq(tree):
    """Linearize a proof tree to its DERIVED atoms in derivation order (children before parent, goal last)."""
    seq = []

    def visit(t):
        if t["rule"] is None:
            return
        for c in t["from"]:
            visit(c)
        if t["fact"] not in seq:
            seq.append(t["fact"])
    visit(tree)
    return seq


def verify_proof(seq, facts, env, goal):
    """KERNEL-CHECK the model's emitted derivation: each atom must be soundly derivable from base + prior
    atoms via a rule. Returns True iff every step checks AND the goal is reached. Bogus steps -> reject."""
    proven = {f: base_tree(f) for f in facts}
    for atom in seq:
        if atom in proven:
            continue
        made = None
        for ri, (head, body) in enumerate(RULES):
            for sub, subtrees in _match_body(body, proven):
                if _inst(head, sub) == atom:
                    tree = {"fact": atom, "rule": ri, "from": subtrees}
                    if kernel_verify(tree, env):
                        made = tree
                        break
            if made:
                break
        if made is None:
            return False                                     # model emitted a non-derivable atom -> reject
        proven[atom] = made
    return goal in proven


# ===================================================================================================
# Generative prover: condition on the goal atom, autoregressively emit proof atoms (pred, e1, e2) + STOP.
# ===================================================================================================
class Prover(nn.Module):
    def __init__(self, n_ent, d=160, h=4, layers=4, maxlen=10):
        super().__init__()
        self.ent = nn.Embedding(n_ent, d)
        self.pred = nn.Embedding(NPRED, d)
        self.role = nn.Embedding(2, d)                       # 0 = goal token, 1 = proof-step token
        self.posn = nn.Embedding(maxlen + 1, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, h, 4 * d, batch_first=True, norm_first=True, activation="gelu")
            for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.h_pred = nn.Linear(d, NPRED + 1)                # +1 = STOP
        self.h_e1 = nn.Linear(d, n_ent)
        self.h_e2 = nn.Linear(d, n_ent)
        self.d = d

    def _tok(self, pred, e1, e2, role, pos):
        return self.pred(pred) + self.ent(e1) + self.ent(e2) + self.role.weight[role] + self.posn(pos)

    def forward(self, goal, steps):
        """goal: (B,3) [pred,e1,e2]; steps: (B,T,3). Returns logits (pred (+STOP), e1, e2) at each of the
        T+1 positions (the goal token + each step token), predicting the NEXT step."""
        B = goal.size(0)
        dev = goal.device
        gt = self._tok(goal[:, 0], goal[:, 1], goal[:, 2],
                       0, torch.zeros(B, dtype=torch.long, device=dev))[:, None]
        toks = [gt]
        T = steps.size(1)
        for t in range(T):
            pos = torch.full((B,), t + 1, dtype=torch.long, device=dev)
            toks.append(self._tok(steps[:, t, 0], steps[:, t, 1], steps[:, t, 2], 1, pos)[:, None])
        x = torch.cat(toks, 1)                               # (B, T+1, d)
        mask = torch.triu(torch.ones(T + 1, T + 1, dtype=torch.bool, device=dev), 1)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        x = self.ln(x)
        return self.h_pred(x), self.h_e1(x), self.h_e2(x)


def encode(atom, eid):
    return [PID[atom[0]], eid[atom[1][0]], eid[atom[1][1]]]


@torch.no_grad()
def generate(model, goal, eid, inv, facts, env, device, maxlen=10):
    """Model emits the proof greedily (no search); returns the atom sequence it produced."""
    model.eval()
    g = torch.tensor([encode(goal, eid)], device=device)
    steps = torch.zeros((1, 0, 3), dtype=torch.long, device=device)
    seq = []
    for _ in range(maxlen):
        lp, le1, le2 = model(g, steps)
        p = int(lp[0, -1].argmax())
        if p == NPRED:                                       # STOP
            break
        a = (("isa", "has_part", "member_of")[p], (inv[int(le1[0, -1].argmax())], inv[int(le2[0, -1].argmax())]))
        seq.append(a)
        steps = torch.cat([steps, torch.tensor([[encode(a, eid)]], device=device)], 1)
    return seq


def evaluate(model, goalpool, eid, inv, facts, env, device, n=200):
    proved = unsound = tot = 0
    for (c, g) in goalpool[:n]:
        seq = generate(model, g, eid, inv, facts, env, device)
        ok = verify_proof(seq, facts, env, g)               # kernel checks the MODEL's proof
        proved += int(ok); tot += 1
        # "unsound" would be: we accept a proof the kernel rejects -- impossible here (we gate on verify)
    return dict(proved=proved / max(1, tot), n=tot)


def train_set(facts, env, goalpool, eid, maxlen=10):
    closure = full_closure(facts, env)
    data = []
    for (c, g) in goalpool:
        if g not in closure:
            continue
        seq = proof_seq(closure[g])
        if not seq or len(seq) > maxlen:
            continue
        data.append((encode(g, eid), [encode(a, eid) for a in seq]))
    return data


def selfplay(facts, ents, concepts, d=160, rounds=500, bs=64, lr=1.2e-3, maxlen=10,
             device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eid = {e: i for i, e in enumerate(ents)}
    inv = {i: e for e, i in eid.items()}
    env = LK.build_env(ents, PREDS, facts, RULES)
    goalpool = [(c, g) for c in concepts for g in goals_for(facts, c)]
    data = train_set(facts, env, goalpool, eid, maxlen)      # teacher proofs (kernel-verified)
    if verbose:
        print(f"  {len(goalpool)} goals -> {len(data)} teacher proofs; training the generative prover", flush=True)
    model = Prover(len(ents), d=d, maxlen=maxlen).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(0)
    for rnd in range(rounds):
        idx = rng.integers(0, len(data), size=min(bs, len(data)))
        batch = [data[i] for i in idx]
        T = max(len(s) for _g, s in batch)
        goal = torch.tensor([g for g, _s in batch], device=device)
        steps = torch.zeros((len(batch), T, 3), dtype=torch.long, device=device)
        tp = torch.full((len(batch), T + 1), NPRED, dtype=torch.long, device=device)   # target pred (STOP default)
        te1 = torch.zeros((len(batch), T + 1), dtype=torch.long, device=device)
        te2 = torch.zeros((len(batch), T + 1), dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), T + 1), dtype=torch.bool, device=device)
        for b, (_g, s) in enumerate(batch):
            for t, a in enumerate(s):
                steps[b, t] = torch.tensor(a, device=device)
                tp[b, t] = a[0]; te1[b, t] = a[1]; te2[b, t] = a[2]; mask[b, t] = True
            mask[b, len(s)] = True                            # the STOP position is supervised too
        model.train()
        lp, le1, le2 = model(goal, steps)
        loss = F.cross_entropy(lp[mask], tp[mask])
        am = mask & (tp != NPRED)                             # entity heads only where there's a real atom
        if am.any():
            loss = loss + F.cross_entropy(le1[am], te1[am]) + F.cross_entropy(le2[am], te2[am])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if verbose and rnd % max(1, rounds // 6) == 0:
            ev = evaluate(model, goalpool, eid, inv, facts, env, device, n=150)
            print(f"  r{rnd:5d}: model-proves-alone {ev['proved']:.3f} (kernel-verified)", flush=True)
    return model, eid, inv, env, goalpool


def selftest():
    rng = np.random.default_rng(0)
    facts, ents, concepts = synthetic_kb(rng, depth=6, sibs=3, parts=4)
    model, eid, inv, env, goalpool = selfplay(facts, ents, concepts, d=96, rounds=500, bs=48,
                                              device="cpu", verbose=False)
    ev = evaluate(model, goalpool, eid, inv, facts, env, "cpu", n=len(goalpool))
    print(f"  selftest: model-proves-alone {ev['proved']:.3f} (each kernel-verified; unsound impossible)")
    assert ev["proved"] > 0.6, f"model did not learn to generate proofs alone ({ev['proved']:.2f})"
    print("prove selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--wordnet", action="store_true"); ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=600); ap.add_argument("--d", type=int, default=160)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.wordnet:
        facts, ents, concepts = wordnet_kb(cap=args.cap)
        print(f"  REAL WordNet KB: {len(facts)} facts, {len(ents)} concepts", flush=True)
    else:
        facts, ents, concepts = synthetic_kb(np.random.default_rng(0))
    model, eid, inv, env, goalpool = selfplay(facts, ents, concepts, d=args.d, rounds=args.rounds,
                                              device=args.device)
    ev = evaluate(model, goalpool, eid, inv, facts, env, model.ent.weight.device, n=400)
    print(f"  FINAL: model-proves-alone {ev['proved']:.3f} over {ev['n']} goals (kernel-verified, 0 unsound)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
