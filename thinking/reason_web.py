#!/usr/bin/env python3
"""Scale engine-verified per-step reasoning from ONE relation to a RELATION WEB.

Builds on thinking/reason_demo.py (proof-supervised transitive inference generalizes to held-out
configs where answer-only sits at chance). Here we move from a SINGLE transitive relation to a small
WEB of relation types whose derivations require CROSS-RELATION COMPOSITION:

  RULES (Horn clauses, run by the datalog engine = the ground-truth oracle for ALL labels):
    R0 isa transitivity:       isa(?X,?Z)      :- isa(?X,?Y),      isa(?Y,?Z)
    R1 part_of transitivity:   part_of(?X,?Z)  :- part_of(?X,?Y),  part_of(?Y,?Z)
    R2 INHERITANCE (cross-rel): has_prop(?X,?P) :- isa(?X,?Y),      has_prop(?Y,?P)

  R2 is the key cross-relation step: a property declared on a CATEGORY is inherited by a member
  through the isa chain. A multi-hop has_prop query (isa(x->y1->...->yk), has_prop(yk,p)) REQUIRES
  composing a k-step isa transitive chain (R0) and THEN the inheritance step (R2).

EXAMPLE GENERATION (fresh random labels per example -> no memorization):
  - sample distinct entity labels from a shared pool; build an isa chain x0 isa x1 isa ... isa xK,
    a separate part_of chain, and declare some has_prop(category, property) base facts.
  - present all base facts SHUFFLED; query a DERIVED fact; answer yes/no = datalog.entails.
  - query types: ISA (transitive isa), PART (transitive part_of), INHERIT (cross-relation has_prop).
  - negatives: same query predicate but a pair/property NOT entailed (engine-checked).

PROOF RENDERING (dense-supervised `step <pred> <args> .` tokens, answer read off the VERIFIED trace):
  - ISA / PART: engine forward-walk along the chain, emitting each derived transitive edge.
  - INHERIT: emit the derived isa(x, cat) edges that bridge x to the property-bearing category,
    THEN the cross-relation step has_prop(x, p). This renders the actual rule composition
    (R0* then R2). The answer ("yes") is grounded in whether the verified trace derived the
    queried fact; negatives have no such derivation -> "no" (read off the absence in the trace,
    with an engine-checked closure guard).

HELD-OUT generalization is over fresh random CONFIGS (asserted disjoint from train).

  python -m thinking.reason_web --sanity   # train-acc gate for PROOF (B) only
  python -m thinking.reason_web            # 3-seed A(answer-only) vs B(proof) by query type
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

RULES = [
    (("isa", ("?x", "?z")), [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))]),        # R0
    (("part_of", ("?x", "?z")), [("part_of", ("?x", "?y")), ("part_of", ("?y", "?z"))]),  # R1
    (("has_prop", ("?x", "?p")), [("isa", ("?x", "?y")), ("has_prop", ("?y", "?p"))]),     # R2
]
DL = Datalog(RULES)

POOL = [f"e{i}" for i in range(40)]          # entity / category labels
PROPS = [f"p{i}" for i in range(20)]         # property labels
STRUCT = ["<pad>", "isa", "part_of", "has_prop", "query", "yes", "no", ":", "step", "answer", "."]

QTYPES = ("ISA", "PART", "INHERIT")


def build_vocab():
    itos = list(STRUCT) + list(POOL) + list(PROPS)
    return itos, {t: i for i, t in enumerate(itos)}


def config_key(edb, q):
    return (frozenset(edb), q)


def make_example(rng, qtype, isa_len=5, part_len=5):
    """Build a fresh random relation-web config + a (query, gold) of the requested type.

    Returns (edb_set, query_atom, gold_bool, meta). meta carries the chain info for proof rendering.
    Distances/lengths are randomized so a fresh (config, query) never recurs.
    """
    # disjoint label sets for the isa chain, the part chain, and (some) distractor categories
    n_isa = rng.randint(4, isa_len)          # >=4 labels -> always a distance-2+ transitive pair
    n_part = rng.randint(4, part_len)
    need = n_isa + n_part + 2                 # +2 spare distractor labels
    labels = rng.sample(POOL, need)
    isa_chain = labels[:n_isa]               # isa_chain[0] isa isa_chain[1] isa ...
    part_chain = labels[n_isa:n_isa + n_part]
    spares = labels[n_isa + n_part:]

    edb = set()
    for i in range(n_isa - 1):
        edb.add(("isa", (isa_chain[i], isa_chain[i + 1])))
    for i in range(n_part - 1):
        edb.add(("part_of", (part_chain[i], part_chain[i + 1])))

    # property facts: attach a property to a category somewhere up the isa chain
    props = rng.sample(PROPS, 3)
    prop_on = {}                              # category -> property declared directly on it
    # declare one property on a category at a random height in the isa chain (index >=1 so a member
    # below inherits it through >=1 isa hop); plus a couple of unrelated property facts
    cat_idx = rng.randint(1, n_isa - 1)
    cat = isa_chain[cat_idx]
    p_inh = props[0]
    edb.add(("has_prop", (cat, p_inh)))
    prop_on[cat] = p_inh
    # an unrelated property on a spare category (distractor; not inherited by isa_chain[0])
    if spares:
        edb.add(("has_prop", (spares[0], props[1])))

    meta = {"isa_chain": isa_chain, "part_chain": part_chain, "cat_idx": cat_idx,
            "cat": cat, "p_inh": p_inh, "spares": spares, "props": props}

    if qtype == "ISA":
        if rng.random() < 0.5:               # positive: a transitive isa pair (distance >= 2)
            i = rng.randint(0, n_isa - 3)    # n_isa>=4 guarantees a valid i with j>=i+2
            j = rng.randint(i + 2, n_isa - 1)
            q = ("isa", (isa_chain[i], isa_chain[j]))
        else:                                # negative: subject = root (non-empty walk), false target
            # ask isa(root, T) where T is NOT an isa-ancestor of root: a part-chain label or a spare.
            tgt = part_chain[-1] if rng.random() < 0.5 else (spares[0] if spares else part_chain[0])
            q = ("isa", (isa_chain[0], tgt))
        return edb, q, DL.entails(edb, q), meta

    if qtype == "PART":
        if rng.random() < 0.5:
            i = rng.randint(0, n_part - 3)
            j = rng.randint(i + 2, n_part - 1)
            q = ("part_of", (part_chain[i], part_chain[j]))
        else:                                # negative: subject = root, false target
            tgt = isa_chain[-1] if rng.random() < 0.5 else (spares[0] if spares else isa_chain[0])
            q = ("part_of", (part_chain[0], tgt))
        return edb, q, DL.entails(edb, q), meta

    # INHERIT: cross-relation composition. Positive REQUIRES isa-chain (member -> cat) + R2.
    if rng.random() < 0.5:                    # positive: member below cat inherits p_inh
        member = isa_chain[rng.randint(0, cat_idx - 1)]   # strictly below the property-bearing cat
        q = ("has_prop", (member, p_inh))
    else:                                     # negative: member with a property it does NOT inherit
        member = isa_chain[0]
        # a property that is declared somewhere but not on this member's isa-ancestry
        neg_p = meta["props"][2]
        q = ("has_prop", (member, neg_p))
    return edb, q, DL.entails(edb, q), meta


# ---------------------------------------------------------------------------------------------------
# Engine-derived proof chains (post-order rule composition rendered as `step <pred> args .` tokens).
# ---------------------------------------------------------------------------------------------------
def isa_walk(edb, x):
    """Derived isa(x, c) hops along the isa chain (excluding the restated EDB edge)."""
    succ = {a: b for (p, (a, b)) in edb if p == "isa"}
    out, cur = [], x
    while cur in succ:
        nxt = succ[cur]
        if ("isa", (x, nxt)) not in edb:
            out.append((x, nxt))
        cur = nxt
    return out


def part_walk(edb, x):
    succ = {a: b for (p, (a, b)) in edb if p == "part_of"}
    out, cur = [], x
    while cur in succ:
        nxt = succ[cur]
        if ("part_of", (x, nxt)) not in edb:
            out.append((x, nxt))
        cur = nxt
    return out


def render_proof_chain(edb, q, gold):
    """Return list of step-token-lists (each a `step <pred> a b .` row): the ENGINE-DERIVED forward
    EXPLORATION from the query subject `a`, rendered IDENTICALLY for positives and negatives so the
    post-`:` structure is invariant (always begins with `step`; the answer-collapse failure -- where
    negatives have no chain so the model learns to skip straight to `answer` -- is thereby removed).

    The queried fact (pred, (a,b)) appears as a verified step IFF it is entailed (gold=yes); a
    negative renders the SAME exhaustive derivation from `a` but the queried fact simply never shows
    up in it -> trace-grounded read-out returns "no" from its absence. For INHERIT the chain renders
    the isa bridge (R0*) THEN every inherited has_prop(a, p) (R2) -- the cross-relation composition.
    `gold` is asserted consistent with the rendered trace below."""
    pred, (a, b) = q
    rows = []
    if pred == "isa":
        for (ca, cc) in isa_walk(edb, a):                 # all derived isa(a, c) up the chain
            rows.append(["step", "isa", ca, cc, "."])
    elif pred == "part_of":
        for (ca, cc) in part_walk(edb, a):
            rows.append(["step", "part_of", ca, cc, "."])
    elif pred == "has_prop":
        # 1) derived isa(a, c) bridge for every strict ancestor c of a (R0*)
        ancestors = [cc for (_aa, cc) in isa_walk(edb, a)]
        succ = {x: y for (p, (x, y)) in edb if p == "isa"}
        first = succ.get(a)
        anc_set = set(ancestors) | ({first} if first else set())   # includes the direct parent
        for (ca, cc) in isa_walk(edb, a):
            rows.append(["step", "isa", ca, cc, "."])
        # 2) every property inherited by `a` through an ancestor that bears it (R2, cross-relation)
        inherited = []
        for (p, (cx, px)) in edb:
            if p == "has_prop" and cx in anc_set and cx != a:
                inherited.append(px)
        for px in sorted(set(inherited)):
            rows.append(["step", "has_prop", a, px, "."])
    # consistency: queried fact is rendered as a step iff gold (engine truth)
    rendered = (["step", pred, a, b, "."] in rows)
    assert rendered == gold, ("render/oracle disagree", q, gold, rows)
    return rows


# ---------------------------------------------------------------------------------------------------
def render_prompt(edb, q, rng):
    facts = list(edb)
    rng.shuffle(facts)
    toks = []
    for pred, (a, b) in facts:
        toks += [pred, a, b, "."]
    pred, (a, b) = q
    toks += ["query", pred, a, b, ":"]
    return toks


def make_batch(rng, stoi, proof, device, batch=64, block=160):
    seqs, masks, configs, qtypes = [], [], [], []
    fill = stoi["."]
    for _ in range(batch):
        qtype = rng.choice(QTYPES)
        edb, q, gold, meta = make_example(rng, qtype)
        ans = "yes" if gold else "no"
        prompt = render_prompt(edb, q, rng)
        toks = list(prompt)
        if proof:
            rows = render_proof_chain(edb, q, gold)
            sup_start = len(toks)
            for r in rows:
                toks += r
            toks += ["answer", ans, "."]
        else:
            toks += [ans, "."]
            sup_start = len(toks) - 2
        ids = [stoi[t] for t in toks]
        if len(ids) > block:
            continue
        m = [0] * len(ids)
        if proof:
            for i in range(sup_start, len(ids)):
                m[i] = 1
        else:
            m[sup_start] = 1
        seqs.append(ids)
        masks.append(m)
        configs.append(config_key(edb, q))
        qtypes.append(qtype)
    L = min(max(len(s) for s in seqs), block)
    ids_b = torch.full((len(seqs), L), fill, dtype=torch.long)
    mask_b = torch.zeros((len(seqs), L), dtype=torch.float)
    for r, (s, m) in enumerate(zip(seqs, masks)):
        ids_b[r, :len(s)] = torch.tensor(s)
        mask_b[r, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids_b.to(device), mask_b.to(device), configs, qtypes


def loss_fn(model, ids, mask):
    logits = model(ids)[:, :-1]
    target = ids[:, 1:]
    tgt_mask = (mask[:, 1:] > 0)
    sel_logits = logits[tgt_mask]
    sel_target = target[tgt_mask]
    if sel_logits.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(sel_logits, sel_target)


@torch.no_grad()
def greedy_answer(model, prompt_toks, q, edb, stoi, itos, proof, device, max_new=90, block=160):
    """ANSWER-ONLY: read the bare yes/no token. PROOF: generate the chain, ENGINE-CHECK each emitted
    `step <pred> a b .` against the closure, and ground the verdict in the verified trace -- "yes"
    iff the model derived the EXACT queried fact via a verified step, else "no" (trace absence)."""
    pred, (a, b) = q
    ids = [stoi[t] for t in prompt_toks]
    yes_id, no_id = stoi["yes"], stoi["no"]
    answer_id, dot_id = stoi["answer"], stoi["."]
    if not proof:
        ctx = torch.tensor([ids[-block:]], device=device)
        logits = model(ctx)[0, -1]
        return "yes" if float(logits[yes_id]) >= float(logits[no_id]) else "no"

    closure = DL.closure(set(edb))[0]
    verdict, cur, depth, saw_answer = None, [], 0, False
    for _ in range(max_new):
        ctx = torch.tensor([ids[-block:]], device=device)
        logits = model(ctx)[0, -1]
        nxt = int(logits.argmax())
        if saw_answer:
            bare = "yes" if float(logits[yes_id]) >= float(logits[no_id]) else "no"
            return verdict if verdict is not None else bare
        ids.append(nxt)
        if nxt == answer_id:
            saw_answer = True
            continue
        if nxt == dot_id:
            w = [itos[t] for t in cur]
            if len(w) == 4 and w[0] == "step" and w[1] in ("isa", "part_of", "has_prop"):
                sp, sa, sb = w[1], w[2], w[3]
                if (sp, (sa, sb)) in closure:                 # ENGINE CHECK (reject hallucinations)
                    if (sp, sa, sb) == (pred, a, b):
                        verdict = "yes"
            cur = []
            depth += 1
            if depth > 4 * len(edb) + 8:
                break
        else:
            cur.append(nxt)
    # trace-grounded read-out: derived the exact fact -> yes; never derived it -> no
    return verdict if verdict is not None else "no"


@torch.no_grad()
def evaluate(model, rng, stoi, itos, proof, device, n_eval=120, seen=None):
    """Held-out accuracy, broken down by query type."""
    model.eval()
    per = {qt: [0, 0] for qt in QTYPES}     # [correct, total]
    for _ in range(n_eval):
        for qtype in QTYPES:
            edb, q, gold, meta = make_example(rng, qtype)
            if seen is not None:
                assert config_key(edb, q) not in seen, "held-out config overlaps train!"
            gold_s = "yes" if gold else "no"
            prompt = render_prompt(edb, q, rng)
            pred = greedy_answer(model, prompt, q, edb, stoi, itos, proof, device)
            per[qtype][1] += 1
            per[qtype][0] += int(pred == gold_s)
    model.train()
    acc = {qt: per[qt][0] / max(1, per[qt][1]) for qt in QTYPES}
    acc["OVERALL"] = sum(per[qt][0] for qt in QTYPES) / max(1, sum(per[qt][1] for qt in QTYPES))
    return acc


def train_one(proof, seed, steps, lr, device, vocab):
    itos, stoi = vocab
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = ScratchpadLM(vocab=len(itos), d=192, layers=4, heads=6, max_len=160,
                         pos_mode="rope", pointer=True, tie=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    seen, last = set(), 0.0
    for step in range(steps):
        ids, mask, configs, _ = make_batch(rng, stoi, proof, device, batch=64)
        seen.update(configs)
        loss = loss_fn(model, ids, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = loss.item()
        if (step + 1) % max(1, steps // 6) == 0:
            print(f"    [{'PROOF' if proof else 'ANSWER'} seed{seed}] step {step+1}/{steps} "
                  f"loss {last:.4f}", flush=True)
    train_acc = evaluate(model, random.Random(seed + 1000), stoi, itos, proof, device, n_eval=80)
    test_acc = evaluate(model, random.Random(seed + 7000), stoi, itos, proof, device,
                        n_eval=120, seen=seen)
    return train_acc, test_acc, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    device = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    vocab = build_vocab()
    itos, stoi = vocab

    print(f"RELATION WEB | rules: R0 isa-trans, R1 part_of-trans, R2 has_prop INHERITANCE "
          f"(cross-rel) | pool={len(POOL)} labels {len(PROPS)} props | query types {QTYPES} | "
          f"steps={args.steps} lr={args.lr} device={device}")

    # show a rendered cross-relation proof example
    rng = random.Random(2)
    while True:
        edb, q, gold, meta = make_example(rng, "INHERIT")
        if gold:
            break
    print("\nINHERIT example (cross-relation composition):")
    print("  isa chain:", " -> ".join(meta["isa_chain"]),
          f"| property declared: has_prop({meta['cat']},{meta['p_inh']})")
    print(f"  query: has_prop({q[1][0]},{q[1][1]})  gold={'yes' if gold else 'no'}")
    rows = render_proof_chain(edb, q, gold)
    print("  rendered proof chain: " + "  ".join(" ".join(r) for r in rows) + "  answer yes .")
    print("  (renders the isa bridge R0* then the cross-relation has_prop step R2)")

    if args.sanity:
        print("\n=== SANITY GATE: PROOF (B) must LEARN TRAIN (>0.85 overall) ===")
        tr, te, ls = train_one(True, 0, args.steps, args.lr, device, vocab)
        print(f"(B) seed0 TRAIN acc: overall {tr['OVERALL']:.3f} "
              f"ISA {tr['ISA']:.3f} PART {tr['PART']:.3f} INHERIT {tr['INHERIT']:.3f} | loss {ls:.4f}")
        print(f"(B) seed0 HELD-OUT: overall {te['OVERALL']:.3f} "
              f"ISA {te['ISA']:.3f} PART {te['PART']:.3f} INHERIT {te['INHERIT']:.3f}")
        print(f"GATE {'PASS' if tr['OVERALL'] > 0.85 else 'FAIL'} (train overall {tr['OVERALL']:.3f})")
        return

    print("\n=== A (answer-only) vs B (proof) : %d-seed held-out by query type ===" % args.seeds)
    res = {"A": [], "B": []}
    trn = {"A": [], "B": []}
    for seed in range(args.seeds):
        print(f"\n-- seed {seed} --")
        trA, teA, lsA = train_one(False, seed, args.steps, args.lr, device, vocab)
        print(f"  (A) ANSWER  train {trA['OVERALL']:.3f} | HELD-OUT overall {teA['OVERALL']:.3f} "
              f"ISA {teA['ISA']:.2f} PART {teA['PART']:.2f} INHERIT {teA['INHERIT']:.2f}")
        trB, teB, lsB = train_one(True, seed, args.steps, args.lr, device, vocab)
        print(f"  (B) PROOF   train {trB['OVERALL']:.3f} | HELD-OUT overall {teB['OVERALL']:.3f} "
              f"ISA {teB['ISA']:.2f} PART {teB['PART']:.2f} INHERIT {teB['INHERIT']:.2f}")
        res["A"].append(teA); res["B"].append(teB)
        trn["A"].append(trA["OVERALL"]); trn["B"].append(trB["OVERALL"])

    def ms(xs):
        m = sum(xs) / len(xs)
        return m, (max(xs) - min(xs)) / 2

    keys = ["OVERALL", "ISA", "PART", "INHERIT"]
    print("\n===================== SUMMARY (held-out accuracy, chance=0.50) =====================")
    print(f">=3 seeds, mean +/- half-spread | train overall: A {sum(trn['A'])/len(trn['A']):.3f} "
          f"B {sum(trn['B'])/len(trn['B']):.3f}")
    print(f"{'cell':<12s} | " + " | ".join(f"{k:^13s}" for k in keys))
    for tag, label in (("A", "ANSWER-ONLY"), ("B", "PROOF-SUPER ")):
        cells = []
        for k in keys:
            m, s = ms([r[k] for r in res[tag]])
            cells.append(f"{m:.2f}±{s:.2f}")
        print(f"{label:<12s} | " + " | ".join(f"{c:^13s}" for c in cells))
    print("=" * 78)


if __name__ == "__main__":
    main()
