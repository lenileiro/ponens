#!/usr/bin/env python3
"""DEPTH-GENERALIZATION scaling test for engine-verified per-step reasoning.

Builds directly on thinking/reason_demo.py (which showed PROOF-SUPERVISED transitive inference
generalizes to held-out CONFIGS at ~0.76-0.82 vs answer-only chance 0.50). Here we change the
generalization axis to the HARDEST one: DEPTH / proof-length.

  TRAIN: queries at rank-distance 1..K_TRAIN  (proof chains up to ~K_TRAIN steps).
  TEST : queries at rank-distance > K_TRAIN on FRESH configs -- STRICTLY DEEPER proofs / LONGER
         verified reasoning chains than ever trained. Train/test depths are DISJOINT by construction
         (distance is bucketed, asserted at eval time).

Proof-supervision is IDENTICAL to reason_demo (engine-derived forward-walk chain rendered as
dense-supervised `step gt a b .` tokens; the answer is read off the ENGINE-VERIFIED trace).

Two architectures, everything else identical (size/seed/steps/lr/supervision):
  (1) STANDARD : ScratchpadLM(loop=False, pointer=True)  -- the config that won the config-holdout.
  (2) RECURRENT: ScratchpadLM(loop=True, loop_inject=True, mhc=True, attn_window=W, pointer=True)
                 forward(..., loops=R, loop_noise=0.03)   -- the project's length-gen recipe.

  python -m thinking.reason_scale --sanity        # sanity gate: both archs learn TRAIN
  python -m thinking.reason_scale                  # full depth-gen comparison, >=3 seeds
"""
import argparse
import os
import sys
import random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datalog import Datalog
from scratchpad_model import ScratchpadLM

RULES = [(("gt", ("?x", "?z")), [("gt", ("?x", "?y")), ("gt", ("?y", "?z"))])]
DL = Datalog(RULES)

POOL = [f"e{i}" for i in range(40)]
STRUCT = ["<pad>", "gt", "query", ">", "<", ":", "step", "answer", "."]


def build_vocab():
    itos = list(STRUCT) + list(POOL)
    return itos, {t: i for i, t in enumerate(itos)}


def config_key(edb, query_pair):
    return (frozenset(edb), tuple(query_pair))


def make_example(rng, pool, n):
    names = rng.sample(pool, n)
    order = list(names)
    rng.shuffle(order)                       # order[0] > order[1] > ... > order[n-1]
    edb = {("gt", (order[i], order[i + 1])) for i in range(n - 1)}
    return edb, order


def render_prompt(edb, query_pair, rng):
    facts = list(edb)
    rng.shuffle(facts)
    toks = []
    for _pred, (a, b) in facts:
        toks += ["gt", a, b, "."]
    x, z = query_pair
    toks += ["query", x, z, ":"]
    return toks


def forward_walk(edb, x):
    """Canonical x-rooted forward walk: derived (x, c) pairs (excluding restated EDB edges)."""
    succ = {a: b for (_p, (a, b)) in edb}
    out, cur = [], x
    while cur in succ:
        nxt = succ[cur]
        if ("gt", (x, nxt)) not in edb:
            out.append((x, nxt))
        cur = nxt
    return out


def pick_query_at_distance(rng, order, dist):
    """Pick a queried pair whose rank-distance == dist, randomized orientation. order is gt-desc."""
    i = rng.randint(0, len(order) - 1 - dist)
    a, b = order[i], order[i + dist]          # a > b, distance dist
    return (a, b) if rng.random() < 0.5 else (b, a)


def make_batch(rng, pool, n, stoi, device, dist_lo, dist_hi, batch=64, block=256):
    """Proof-supervised batch (engine-derived chain, dense loss over chain+answer). Queries are
    drawn at rank-distance in [dist_lo, dist_hi]. Returns (ids, mask, configs, dists).

    PADDING uses a NON-pad FILLER token (not stoi['<pad>']), and supervision/loss never touches the
    padded tail. Why a filler instead of '<pad>': the windowed-attention recurrent path makes a
    right-pad TAIL query row attend only to in-window keys that are themselves pad -- and pad keys
    are masked out, so that row softmaxes over an all-(-inf) span = NaN, which then spreads to ALL
    positions across loop iterations via `0 * NaN` in the value aggregation (verified). With a real
    filler id there is no pad mask, so every tail row at least attends to itself (no all-masked row,
    no NaN). CAUSALITY guarantees the trailing filler NEVER leaks into the real positions (a real
    query can only attend to keys at <= its own position; filler sits strictly after) -- verified
    bit-identical (diff 0.0) to the unpadded single-sequence forward. The model's pad token is kept
    OUT of the batch, so model.pad never appears and the would-be-NaN path is never triggered."""
    seqs, masks, configs, dists = [], [], [], []
    fill = stoi["."]                                     # real token used purely as inert right-fill
    for _ in range(batch):
        edb, order = make_example(rng, pool, n)
        dist = rng.randint(dist_lo, min(dist_hi, len(order) - 1))
        x, z = pick_query_at_distance(rng, order, dist)
        gt_xz = DL.entails(edb, ("gt", (x, z)))
        ans = ">" if gt_xz else "<"
        prompt = render_prompt(edb, (x, z), rng)
        toks = list(prompt)
        chain = forward_walk(edb, x)
        reached = {c for (_a, c) in chain} | {b for (_p, (a, b)) in edb if a == x}
        assert (z in reached) == gt_xz, "walk/engine disagree"
        sup_start = len(toks)
        if gt_xz:
            for (ca, cc) in chain:
                toks += ["step", "gt", ca, cc, "."]
            if ("gt", (x, z)) not in edb:
                toks += ["step", "gt", x, z, "."]
        else:
            chain2 = forward_walk(edb, z)
            for (ca, cc) in chain2:
                toks += ["step", "gt", ca, cc, "."]
            if ("gt", (z, x)) not in edb:
                toks += ["step", "gt", z, x, "."]
        toks += ["answer", ans, "."]
        ids = [stoi[t] for t in toks]
        if len(ids) > block:
            continue
        m = [0] * len(ids)
        for i in range(sup_start, len(ids)):
            m[i] = 1
        seqs.append(ids)
        masks.append(m)
        configs.append(config_key(edb, (x, z)))
        dists.append(dist)
    L = min(max(len(s) for s in seqs), block)
    ids_b = torch.full((len(seqs), L), fill, dtype=torch.long)   # filler, NOT model.pad (see above)
    mask_b = torch.zeros((len(seqs), L), dtype=torch.float)
    for r, (s, m) in enumerate(zip(seqs, masks)):
        ids_b[r, :len(s)] = torch.tensor(s)
        mask_b[r, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids_b.to(device), mask_b.to(device), configs, dists


def model_logits(model, ids, recurrent, loops, loop_noise):
    if recurrent:
        return model(ids, loops=loops, loop_noise=loop_noise)
    return model(ids)


def loss_fn(model, ids, mask, recurrent, loops, loop_noise):
    logits = model_logits(model, ids, recurrent, loops, loop_noise)
    logits = logits[:, :-1]
    target = ids[:, 1:]
    tgt_mask = (mask[:, 1:] > 0)
    # CE over the supervised chain+answer span only (the filler-padded tail is never supervised).
    sel_logits = logits[tgt_mask]                       # (S, V)
    sel_target = target[tgt_mask]                       # (S,)
    if sel_logits.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(sel_logits, sel_target)


@torch.no_grad()
def greedy_answer(model, prompt_toks, query_pair, edb, stoi, itos, device,
                  recurrent, loops, max_new=80, block=256):
    """Generate the chain, ENGINE-CHECK each emitted `step gt a b .`, read the queried pair's
    order off the verified trace ('>' if it derived gt(x,z), '<' if gt(z,x)); bare-answer fallback."""
    x, z = query_pair
    ids = [stoi[t] for t in prompt_toks]
    answer_id, step_id = stoi["answer"], stoi["step"]
    gt_id, lt_id, dot_id = stoi[">"], stoi["<"], stoi["."]
    closure = DL.closure(set(edb))[0]
    # Adjacent (distance-1) queries are a GIVEN fact: the pair is an EDB edge that forward_walk and
    # make_batch deliberately EXCLUDE from the rendered chain (only DERIVED gt(x,c) hops are steps),
    # so the verified trace never restates it and the trace-grounded verdict can never be set for
    # them. Resolving such a query is just READING the presented edge, not reasoning -- so the
    # legitimate fallback reads the queried pair's order off the PROMPT EDB directly (never the
    # transitive closure, which would let deeper queries skip the chain). The model's bare answer
    # token is unreliable (it collapses to a constant symbol), so this replaces it as the fallback.
    edb_verdict = (">" if ("gt", (x, z)) in edb else
                   ("<" if ("gt", (z, x)) in edb else None))
    verdict, cur, depth, saw_answer = None, [], 0, False
    for _ in range(max_new):
        ctx = torch.tensor([ids[-block:]], device=device)
        logits = model_logits(model, ctx, recurrent, loops, 0.0)[0, -1]
        nxt = int(logits.argmax())
        if saw_answer:
            bare = ">" if nxt == gt_id else ("<" if nxt == lt_id else
                   (">" if float(logits[gt_id]) >= float(logits[lt_id]) else "<"))
            return verdict if verdict is not None else (edb_verdict if edb_verdict is not None else bare)
        ids.append(nxt)
        if nxt == answer_id:
            saw_answer = True
            continue
        if nxt == dot_id:
            w = [itos[t] for t in cur]
            if len(w) == 4 and w[0] == "step" and w[1] == "gt":
                a, b = w[2], w[3]
                if ("gt", (a, b)) in closure:
                    if (a, b) == (x, z):
                        verdict = ">"
                    elif (a, b) == (z, x):
                        verdict = "<"
            cur = []
            depth += 1
            if depth > 4 * len(edb) + 8:
                break
        else:
            cur.append(nxt)
    return verdict if verdict is not None else edb_verdict


@torch.no_grad()
def eval_at_distance(model, rng, pool, n, stoi, itos, device, recurrent, loops, dist,
                     n_eval=200, seen_configs=None, assert_depth_gt=None):
    model.eval()
    correct = 0
    for _ in range(n_eval):
        edb, order = make_example(rng, pool, n)
        x, z = pick_query_at_distance(rng, order, dist)
        if assert_depth_gt is not None:
            assert dist > assert_depth_gt, "eval depth not strictly deeper than train"
        if seen_configs is not None:
            assert config_key(edb, (x, z)) not in seen_configs, "held-out config overlaps train!"
        gt_xz = DL.entails(edb, ("gt", (x, z)))
        gold = ">" if gt_xz else "<"
        prompt = render_prompt(edb, (x, z), rng)
        pred = greedy_answer(model, prompt, (x, z), edb, stoi, itos, device, recurrent, loops)
        correct += int(pred == gold)
    model.train()
    return correct / n_eval


def build_model(recurrent, n, vocab, device, attn_window, loops):
    itos, _ = vocab
    if recurrent:
        return ScratchpadLM(vocab=len(itos), d=192, layers=4, heads=6, max_len=256,
                            pos_mode="rope", pointer=True, tie=True,
                            loop=True, loops=loops, loop_inject=True, mhc=True,
                            attn_window=attn_window).to(device)
    return ScratchpadLM(vocab=len(itos), d=192, layers=4, heads=6, max_len=256,
                        pos_mode="rope", pointer=True, tie=True, loop=False).to(device)


def train_one(recurrent, seed, n, steps, lr, device, vocab, k_train, attn_window, loops,
              loop_noise, test_depths, n_eval=80):
    itos, stoi = vocab
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = build_model(recurrent, n, vocab, device, attn_window, loops)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    seen, last = set(), 0.0
    for step in range(steps):
        ids, mask, configs, _ = make_batch(rng, POOL, n, stoi, device, 1, k_train, batch=64)
        seen.update(configs)
        loss = loss_fn(model, ids, mask, recurrent, loops, loop_noise)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = loss.item()
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"    [{'RECUR' if recurrent else 'STAND'} seed{seed}] step {step+1}/{steps} "
                  f"loss {last:.4f}", flush=True)
    # TRAIN-depth accuracy (in-distribution sanity: queries at distance 1..k_train)
    tr_rng = random.Random(seed + 1000)
    train_acc = 0.0
    for d in range(1, k_train + 1):
        train_acc += eval_at_distance(model, tr_rng, POOL, n, stoi, itos, device,
                                      recurrent, loops, d, n_eval=n_eval)
    train_acc /= k_train
    # DEEPER-than-trained held-out accuracy, per test depth (configs asserted disjoint from train)
    depth_acc = {}
    for d in test_depths:
        depth_acc[d] = eval_at_distance(model, random.Random(seed + 5000 + d), POOL, n, stoi,
                                        itos, device, recurrent, loops, d, n_eval=n_eval,
                                        seen_configs=seen, assert_depth_gt=k_train)
    return train_acc, depth_acc, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9)           # entities -> distances up to n-1
    ap.add_argument("--k_train", type=int, default=3)     # train distances 1..3
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--loops", type=int, default=8)       # recurrent iteration count
    ap.add_argument("--attn_window", type=int, default=24)
    ap.add_argument("--loop_noise", type=float, default=0.03)
    ap.add_argument("--n_eval", type=int, default=80)     # examples per (depth) eval cell
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    device = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    vocab = build_vocab()
    itos, stoi = vocab
    test_depths = list(range(args.k_train + 1, args.n))    # strictly deeper than trained

    print(f"DEPTH-GEN split | N={args.n} entities | TRAIN dist 1..{args.k_train} | "
          f"TEST dist {test_depths} (strictly deeper, fresh configs, asserted disjoint) | "
          f"steps={args.steps} lr={args.lr} | RECUR loops={args.loops} "
          f"attn_window={args.attn_window} loop_noise={args.loop_noise} | device={device}")

    if args.sanity:
        print("\n=== SANITY GATE: both archs must LEARN TRAIN (high train acc) ===")
        for recurrent in (False, True):
            tr, da, ls = train_one(recurrent, 0, args.n, args.steps, args.lr, device, vocab,
                                   args.k_train, args.attn_window, args.loops, args.loop_noise,
                                   test_depths[:2], n_eval=args.n_eval)
            tag = "RECURRENT" if recurrent else "STANDARD"
            print(f"({tag}) seed0: TRAIN-depth acc {tr:.3f}  final_loss {ls:.4f}  "
                  f"first deeper depths {da}")
        return

    print("\n=== DEPTH-GEN: STANDARD vs RECURRENT, %d seeds ===" % args.seeds)
    res = {"STANDARD": {"train": [], "depth": {d: [] for d in test_depths}},
           "RECURRENT": {"train": [], "depth": {d: [] for d in test_depths}}}
    for seed in range(args.seeds):
        print(f"\n-- seed {seed} --")
        for recurrent in (False, True):
            tag = "RECURRENT" if recurrent else "STANDARD"
            tr, da, ls = train_one(recurrent, seed, args.n, args.steps, args.lr, device, vocab,
                                   args.k_train, args.attn_window, args.loops, args.loop_noise,
                                   test_depths, n_eval=args.n_eval)
            res[tag]["train"].append(tr)
            for d in test_depths:
                res[tag]["depth"][d].append(da[d])
            print(f"  ({tag}) TRAIN {tr:.3f} loss {ls:.4f} | deeper " +
                  " ".join(f"d{d}={da[d]:.2f}" for d in test_depths), flush=True)

    def ms(xs):
        m = sum(xs) / len(xs)
        return m, (max(xs) - min(xs)) / 2

    print("\n===================== DEPTH-GEN SUMMARY (held-out DEEPER-chain answer acc) =====================")
    print(f"train distances 1..{args.k_train}; chance = 0.50; >=3 seeds mean +/- half-spread")
    header = "arch       | TRAIN  | " + " | ".join(f"d={d:<2d}" for d in test_depths)
    print(header)
    print("-" * len(header))
    for tag in ("STANDARD", "RECURRENT"):
        trm, trs = ms(res[tag]["train"])
        cells = []
        for d in test_depths:
            m, s = ms(res[tag]["depth"][d])
            cells.append(f"{m:.2f}±{s:.2f}")
        print(f"{tag:<10s} | {trm:.3f} | " + " | ".join(cells))
    print("=" * len(header))


if __name__ == "__main__":
    main()
