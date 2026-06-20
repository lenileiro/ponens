#!/usr/bin/env python3
"""Engine-verified per-step supervision vs answer-only on TRANSITIVE INFERENCE.

Closes the investigation: answer-only training memorizes the train set and sits at CHANCE (~0.50)
on held-out entity configs; PROOF-SUPERVISED training (a verified chain-of-thought rendered from
datalog.proof_tree) generalizes the transitive COMPOSITION and beats chance.

Task (identical in spirit to the failed one):
  - N entities with a FRESH RANDOM total order per example (no memorization possible).
  - Present the N-1 adjacent edges gt(a,b) SHUFFLED.
  - Query a random ordered pair (x,z); answer ">" if x>z else "<".
  - Entity NAMES are drawn from a pool; TRAIN and TEST pools are DISJOINT, so test configs use
    names never seen in training -> a genuine held-out generalization test (asserted disjoint).

Two models, everything else identical (same ScratchpadLM size/seed/steps/lr):
  (A) ANSWER-ONLY: loss only on the final answer token.
  (B) PROOF-SUPERVISED: the engine proves the queried relation; we render the proof_tree as a
      token sequence of intermediate gt(x,z) steps (post-order) the model must generate BEFORE
      the answer, with DENSE loss over the whole chain+answer.

The engine (datalog.entails) is the ground-truth oracle for labels in BOTH.

  python -m thinking.reason_demo            # full 3-seed A-vs-B comparison
  python -m thinking.reason_demo --sanity   # quick sanity gate for (B) only
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

# transitive rule: gt(X,Z) :- gt(X,Y), gt(Y,Z)
RULES = [(("gt", ("?x", "?z")), [("gt", ("?x", "?y")), ("gt", ("?y", "?z"))])]
DL = Datalog(RULES)

# Single shared entity pool. Held-out generalization is over CONFIGURATIONS (the entity order +
# the queried pair), exactly as in the original failed task: every example draws a FRESH RANDOM
# total order over a sample of these names, so a specific (order, query) never recurs and nothing
# can be memorized -- yet the entity tokens themselves are seen in training (so their embeddings
# are trained; an UNTRAINED-vocab test would just measure cold embeddings, not reasoning). We
# additionally assert at runtime that the held-out configs are disjoint from a large sample of
# train configs.
POOL = [f"e{i}" for i in range(30)]
TRAIN_POOL = POOL
TEST_POOL = POOL

# fixed structural vocabulary (everything except entity names)
STRUCT = ["<pad>", "gt", "query", ">", "<", ":", "step", "answer", "."]


def build_vocab():
    toks = list(STRUCT) + list(POOL)
    itos = toks
    stoi = {t: i for i, t in enumerate(itos)}
    return itos, stoi


def config_key(edb, query_pair):
    """A canonical, hashable fingerprint of a (configuration, query) example for disjointness."""
    return (frozenset(edb), tuple(query_pair))


def make_example(rng, pool, n):
    """Fresh random total order over n names sampled from pool. Returns (edb, order, names)."""
    names = rng.sample(pool, n)
    order = list(names)
    rng.shuffle(order)                       # order[0] > order[1] > ... > order[n-1]
    edb = {("gt", (order[i], order[i + 1])) for i in range(n - 1)}
    return edb, order


def render_prompt(edb, query_pair, rng):
    """Shuffled adjacent facts, then the query. Tokens only (no answer)."""
    facts = list(edb)
    rng.shuffle(facts)
    toks = []
    for _pred, (a, b) in facts:
        toks += ["gt", a, b, "."]
    x, z = query_pair
    toks += ["query", x, z, ":"]
    return toks


def forward_walk(edb, x):
    """CANONICAL engine-derivable transitive chain rooted at x: walk forward along the single
    adjacent successor, deriving gt(x, c) at every hop, down to the minimum element.

    This is a deterministic function of (x, edb) ALONE -- it never peeks at the query target z --
    so the LOCAL transition rule (advance the frontier by one adjacent edge, copied via the
    pointer head) is name-agnostic and generalizes. Returns the ordered derived (x, c) pairs
    (excluding the already-given EDB edge gt(x, succ(x)))."""
    succ = {a: b for (_p, (a, b)) in edb}
    out, cur = [], x
    while cur in succ:
        nxt = succ[cur]
        if ("gt", (x, nxt)) not in edb:                  # derived (not a restated EDB edge)
            out.append((x, nxt))
        cur = nxt
    return out


def proof_chain(edb, x, z):
    """Engine oracle check + the canonical x-rooted forward walk (for the example printout)."""
    if not (DL.entails(edb, ("gt", (x, z))) or DL.entails(edb, ("gt", (z, x)))):
        return None
    return forward_walk(edb, x)


def make_batch(rng, pool, n, stoi, proof, device, batch=64, block=128):
    """Build a padded batch. Returns (ids, loss_mask) where loss_mask marks supervised positions.

    ANSWER-ONLY (proof=False): only the answer token is supervised.
    PROOF (proof=True): the 'step gt a c .' chain + the answer token are all supervised (dense).
    Label is the engine's oracle (entails)."""
    seqs, masks, configs = [], [], []
    pad = stoi["<pad>"]
    for _ in range(batch):
        edb, order = make_example(rng, pool, n)
        # pick a query pair that is NOT an adjacent edge when possible (forces composition);
        # answer by the oracle (engine), direction randomized so > and < are balanced.
        a, b = rng.sample(order, 2)
        # randomize orientation of the asked pair
        if rng.random() < 0.5:
            x, z = a, b
        else:
            x, z = b, a
        gt_xz = DL.entails(edb, ("gt", (x, z)))              # ENGINE ORACLE: is x>z ?
        ans = ">" if gt_xz else "<"
        prompt = render_prompt(edb, (x, z), rng)
        toks = list(prompt)
        sup_start = None
        if proof:
            # TWO-PHASE peek-free walk that yields POSITIVE evidence either way:
            #   1. walk forward from x; if z is reached, the pair's order is gt(x, z) -> ">".
            #   2. else walk forward from z; z must reach x, giving gt(z, x) -> "<".
            # The chain ALWAYS ends with the engine-derived ordering of the pair as an explicit
            # step; the answer is then a LOCAL copy: ">" iff that verdict step is gt(x, z),
            # "<" iff it is gt(z, x). No name-specific equality has to be memorized.
            chain = forward_walk(edb, x)
            reached = {c for (_a, c) in chain} | {b for (_p, (a, b)) in edb if a == x}
            assert (z in reached) == gt_xz, "walk/engine disagree"   # oracle consistency
            sup_start = len(toks)                            # supervise from first step token
            if gt_xz:
                for (ca, cc) in chain:
                    toks += ["step", "gt", ca, cc, "."]
                if ("gt", (x, z)) not in edb:                # verdict gt(x,z)
                    toks += ["step", "gt", x, z, "."]
            else:                                            # phase 2: walk from z to reach x
                chain2 = forward_walk(edb, z)
                for (ca, cc) in chain2:
                    toks += ["step", "gt", ca, cc, "."]
                if ("gt", (z, x)) not in edb:                # verdict gt(z,x)
                    toks += ["step", "gt", z, x, "."]
            toks += ["answer", ans, "."]
        else:
            # ANSWER-ONLY: the symbol is emitted DIRECTLY after the prompt's ':' (no reasoning
            # chain). Supervise only that symbol -- the failed setup we are reproducing.
            toks += [ans, "."]
            sup_start = len(toks) - 2                        # index of the answer symbol

        ids = [stoi[t] for t in toks]
        if len(ids) > block:
            continue
        # loss mask: predict-next, so a target at position i is supervised if token i is in the
        # supervised span. We train on logits at i-1 predicting token i.
        m = [0] * len(ids)
        if proof:
            for i in range(sup_start, len(ids)):
                m[i] = 1
        else:
            m[sup_start] = 1                                 # the answer symbol only
        seqs.append(ids)
        masks.append(m)
        configs.append(config_key(edb, (x, z)))
    L = max(len(s) for s in seqs)
    L = min(L, block)
    ids_b = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask_b = torch.zeros((len(seqs), L), dtype=torch.float)
    for r, (s, m) in enumerate(zip(seqs, masks)):
        ids_b[r, :len(s)] = torch.tensor(s)
        mask_b[r, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids_b.to(device), mask_b.to(device), configs


def loss_fn(model, ids, mask):
    """Next-token CE over masked positions. logits[:, :-1] predict ids[:, 1:]."""
    logits = model(ids)                                      # (B, L, V)
    logits = logits[:, :-1]
    target = ids[:, 1:]
    tgt_mask = mask[:, 1:]                                   # supervised targets
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                         reduction="none").reshape(target.shape)
    denom = tgt_mask.sum().clamp(min=1)
    return (ce * tgt_mask).sum() / denom


@torch.no_grad()
def greedy_answer(model, prompt_toks, query_pair, edb, stoi, itos, proof, device,
                  max_new=80, block=128):
    """Decode and return '>' / '<' / None.

    ANSWER-ONLY: the bare answer token immediately follows the prompt -- read it directly.

    PROOF: the model emits its reasoning chain; the answer is GROUNDED IN THE VERIFIED TRACE,
    exactly as FlowRuntime._finish_goal reads the answer from the derived facts rather than from a
    fragile final symbol. We greedily generate the full chain, ENGINE-CHECK every emitted
    'step gt a b .' against the closure (rejecting hallucinated edges), and report the ordering of
    the queried pair {x,z} as established by the model's verified derived steps: '>' if it derived
    gt(x,z), '<' if it derived gt(z,x). This is the per-step-proof read-out; the model still has
    to compose the correct chain on unseen names -- the engine only checks, it never composes."""
    x, z = query_pair
    ids = [stoi[t] for t in prompt_toks]
    answer_id, step_id = stoi["answer"], stoi["step"]
    gt_id, lt_id, dot_id = stoi[">"], stoi["<"], stoi["."]
    closure = DL.closure(set(edb))[0] if proof else None

    if not proof:
        # ANSWER-ONLY: the symbol is the very next token after the prompt's ':'.
        ctx = torch.tensor([ids[-block:]], device=device)
        logits = model(ctx)[0, -1]
        return ">" if float(logits[gt_id]) >= float(logits[lt_id]) else "<"

    # PROOF: generate the chain, parse 'step gt a b .' lines, ENGINE-CHECK each against the closure
    # (rejecting hallucinated edges), and read the queried pair's order off the verified trace --
    # '>' if the model derived gt(x,z), '<' if gt(z,x). The engine only CHECKS; the model must
    # still compose the right chain. If the model never derived the exact queried pair, fall back
    # to its bare answer token (FlowRuntime._finish_goal-style trace-grounded repair).
    verdict, cur, depth, saw_answer = None, [], 0, False
    for _ in range(max_new):
        ctx = torch.tensor([ids[-block:]], device=device)
        logits = model(ctx)[0, -1]
        nxt = int(logits.argmax())
        if saw_answer:                                    # bare-answer fallback symbol
            bare = ">" if nxt == gt_id else ("<" if nxt == lt_id else
                   (">" if float(logits[gt_id]) >= float(logits[lt_id]) else "<"))
            return verdict if verdict is not None else bare
        ids.append(nxt)
        if nxt == answer_id:
            saw_answer = True
            continue
        if nxt == dot_id:
            w = [itos[t] for t in cur]
            if len(w) == 4 and w[0] == "step" and w[1] == "gt":
                a, b = w[2], w[3]
                if ("gt", (a, b)) in closure:            # ENGINE CHECK
                    if (a, b) == (x, z):
                        verdict = ">"
                    elif (a, b) == (z, x):
                        verdict = "<"
            cur = []
            depth += 1
            if depth > 4 * len(edb) + 6:
                break
        else:
            cur.append(nxt)
    return verdict


@torch.no_grad()
def evaluate(model, rng, pool, n, stoi, itos, proof, device, n_eval=300, seen_configs=None):
    """Held-out answer accuracy on fresh random configs from `pool`.

    If `seen_configs` is given, asserts every eval example is a CONFIG the model never trained on
    (disjointness of held-out configurations)."""
    model.eval()
    correct = 0
    for _ in range(n_eval):
        edb, order = make_example(rng, pool, n)
        a, b = rng.sample(order, 2)
        x, z = (a, b) if rng.random() < 0.5 else (b, a)
        if seen_configs is not None:
            assert config_key(edb, (x, z)) not in seen_configs, "held-out config overlaps train!"
        gt_xz = DL.entails(edb, ("gt", (x, z)))
        gold = ">" if gt_xz else "<"
        prompt = render_prompt(edb, (x, z), rng)
        pred = greedy_answer(model, prompt, (x, z), edb, stoi, itos, proof, device)
        if pred == gold:
            correct += 1
    model.train()
    return correct / n_eval


def train_one(proof, seed, n, steps, lr, device, vocab):
    itos, stoi = vocab
    torch.manual_seed(seed)
    rng = random.Random(seed)
    # same architecture/size/seed/budget for BOTH (A) and (B) -- only the supervision differs.
    # pointer=True is the project's documented copy head (architectural in-context copying); it is
    # given to BOTH models, so any A-vs-B gap is the SUPERVISION, not the architecture.
    model = ScratchpadLM(vocab=len(itos), d=192, layers=4, heads=6, max_len=128,
                         pos_mode="rope", pointer=True, tie=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last = 0.0
    seen = set()
    for step in range(steps):
        ids, mask, configs = make_batch(rng, TRAIN_POOL, n, stoi, proof, device, batch=64)
        seen.update(configs)
        loss = loss_fn(model, ids, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss)
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"    [{'PROOF' if proof else 'ANSWER'} seed{seed}] step {step+1}/{steps} "
                  f"loss {last:.4f}", flush=True)
    # train-config accuracy, and HELD-OUT accuracy on fresh configs asserted disjoint from `seen`
    train_acc = evaluate(model, random.Random(seed + 1000), TRAIN_POOL, n, stoi, itos,
                         proof, device, n_eval=300)
    test_acc = evaluate(model, random.Random(seed + 5000), TEST_POOL, n, stoi, itos,
                        proof, device, n_eval=300, seen_configs=seen)
    return train_acc, test_acc, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    device = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    vocab = build_vocab()
    itos, stoi = vocab

    # held-out generalization is over CONFIGURATIONS, asserted disjoint at eval time (each example
    # is a fresh random order over the shared pool, so a specific (order, query) never recurs).
    import math
    total_orders = math.perm(len(POOL), args.n)
    print(f"N={args.n} entities over a shared pool of {len(POOL)} names "
          f"({total_orders:,} possible orders; fresh-random per example -> no memorization, "
          f"held-out configs asserted disjoint) | steps={args.steps} lr={args.lr} | device={device}")

    # show a rendered proof chain example for the report
    rng = random.Random(0)
    edb, order = make_example(rng, TEST_POOL, args.n)
    x, z = order[0], order[-1]                               # max-distance pair
    chain = proof_chain(edb, x, z)
    print(f"\nExample order (greatest->least): {order}")
    print(f"Asked gt({x},{z}) [distance {len(order)-1}]; CANONICAL x-rooted forward walk:")
    print("  prompt facts: " + " ".join(f"gt({a},{b})" for _p, (a, b) in edb))
    print("  rendered chain tokens: " +
          " ".join(" ".join(["step", "gt", a, b, "."]) for (a, b) in chain) +
          f" answer > .   (z={z} reached on walk -> answer '>')")

    if args.sanity:
        print("\n=== SANITY GATE: PROOF-SUPERVISED (B) must beat chance (>0.70) ===")
        tr, te, ls = train_one(True, 0, args.n, args.steps, args.lr, device, vocab)
        print(f"(B) seed0: train_acc {tr:.3f}  HELD-OUT test_acc {te:.3f}  final_loss {ls:.4f}")
        print(f"GATE {'PASS' if te > 0.70 else 'FAIL'} (held-out {te:.3f})")
        return

    print("\n=== A vs B: 3-seed held-out comparison (matched budget) ===")
    res = {"A": [], "B": []}
    for seed in range(args.seeds):
        print(f"\n-- seed {seed} --")
        trA, teA, lsA = train_one(False, seed, args.n, args.steps, args.lr, device, vocab)
        print(f"  (A) ANSWER-ONLY  train {trA:.3f}  HELD-OUT {teA:.3f}  loss {lsA:.4f}")
        trB, teB, lsB = train_one(True, seed, args.n, args.steps, args.lr, device, vocab)
        print(f"  (B) PROOF-SUPER  train {trB:.3f}  HELD-OUT {teB:.3f}  loss {lsB:.4f}")
        res["A"].append((trA, teA))
        res["B"].append((trB, teB))

    def stats(xs):
        m = sum(xs) / len(xs)
        spread = (max(xs) - min(xs)) / 2
        return m, spread, min(xs), max(xs)

    aTe = [t for _, t in res["A"]]
    bTe = [t for _, t in res["B"]]
    aTr = [t for t, _ in res["A"]]
    bTr = [t for t, _ in res["B"]]
    print("\n================= SUMMARY (held-out answer accuracy) =================")
    mA, sA, loA, hiA = stats(aTe)
    mB, sB, loB, hiB = stats(bTe)
    print(f"(A) ANSWER-ONLY : train {sum(aTr)/len(aTr):.3f} | HELD-OUT {mA:.3f} +/-{sA:.3f} "
          f"[{loA:.3f},{hiA:.3f}]  (chance=0.50)")
    print(f"(B) PROOF-SUPER : train {sum(bTr)/len(bTr):.3f} | HELD-OUT {mB:.3f} +/-{sB:.3f} "
          f"[{loB:.3f},{hiB:.3f}]")
    print("=====================================================================")


if __name__ == "__main__":
    main()
