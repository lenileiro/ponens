#!/usr/bin/env python3
"""Bridge LANGUAGE COMPREHENSION and VERIFIED REASONING: a model that READS natural-language facts
and answers NL queries by COMPOSING them, trained with engine-derived PROOF supervision rendered as
a natural-language reasoning chain (CoT).

Same underlying logic as thinking/reason_web.py (isa transitivity + property INHERITANCE), but
EVERYTHING the model sees is TEMPLATED NATURAL LANGUAGE, word-tokenized. The Datalog engine is the
ORACLE: we control the templates, so we parse the rendered facts back to relations EXACTLY, compute
the deductive closure + proof_tree, and from that proof render a natural-language reasoning chain.

  RELATIONS / RULES (run by the datalog engine = ground-truth oracle for every label):
    R0 isa transitivity:        isa(?X,?Z)      :- isa(?X,?Y), isa(?Y,?Z)
    R1 INHERITANCE (cross-rel):  has_prop(?X,?P) :- isa(?X,?Y), has_prop(?Y,?P)

  NL TEMPLATES (word-tokenized; a/an chosen by leading vowel of the rendered word):
    isa(x,y)      ->  "a robin is a bird ."            (subject + "is" + a/an + category)
    has_prop(x,p) ->  "a bird can breathe ."           (subject + "can" + property)
    query isa     ->  "is a robin an animal ?"
    query has_prop->  "can a robin breathe ?"

  PROOF rendered as an NL reasoning chain (the model generates this, then the answer). ISA queries
  walk the isa chain; INHERIT queries use QUERY-DRIVEN proof-tree order -- the recalled property leaf
  FIRST (led by a distinct "note" token), then the isa bridge, then the conclusion:
    ISA:     "a robin is a bird . a bird is an animal . so a robin is an animal . answer yes ."
    INHERIT: "note an animal can breathe . a robin is a bird . a bird is an animal .
              so a robin can breathe . answer yes ."
  Each rendered fact-sentence corresponds to ONE engine-derived edge of the proof; the final "so ..."
  sentence restates the queried fact; "answer yes/no ." is the verdict. Dense loss over chain+answer.
  At test the model GENERATES the chain then the answer; we read the answer off the generated chain
  and (when proof) engine-CHECK each emitted fact-sentence against the closure, grounding the verdict
  in the verified trace (bare-answer fallback if nothing verified).

  GENERALIZATION: held-out configs are fresh random noun/category/property assignments asserted
  DISJOINT from anything seen in training; INHERIT queries need >=2 isa hops + the inheritance rule,
  so multi-hop composition is required.

  python -m thinking.reason_lang --sanity   # train-acc gate for PROOF only (>=1200 steps)
  python -m thinking.reason_lang            # 3-seed ANSWER-ONLY vs PROOF, held-out, by query type
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
    (("isa", ("?x", "?z")), [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))]),            # R0 isa-trans
    (("has_prop", ("?x", "?p")), [("isa", ("?x", "?y")), ("has_prop", ("?y", "?p"))]),  # R1 inheritance
]
DL = Datalog(RULES)

# --- natural-language vocabulary (concrete words so the surface form is real English) ---------------
# nouns/categories are drawn from one shared pool (a chain is an arbitrary stacking of these), so a
# fresh random assignment per example prevents memorization.
NOUNS = ["robin", "sparrow", "eagle", "salmon", "trout", "shark", "tiger", "lion", "wolf", "fox",
         "bird", "fish", "mammal", "animal", "reptile", "insect", "spider", "snake", "lizard",
         "creature", "beast", "organism", "vertebrate", "predator", "rodent", "feline", "canine",
         "primate", "ape", "monkey", "whale", "dolphin", "otter", "seal", "bat", "owl", "hawk",
         "frog", "toad", "newt"]
PROPS = ["breathe", "move", "grow", "eat", "sleep", "swim", "fly", "hunt", "rest", "feed",
         "wander", "roam", "hide", "climb", "dig", "sing", "crawl", "leap", "dive", "glide"]

# structural / template words the model must produce verbatim in the chain
STRUCT = ["<pad>", "a", "an", "is", "can", "so", "note", "answer", "yes", "no", "query", ".", "?"]

QTYPES = ("ISA", "INHERIT")        # direct/1-hop transitive isa  vs  multi-hop cross-relation inherit


def article(word):
    return "an" if word[0] in "aeiou" else "a"


def build_vocab():
    itos = list(STRUCT) + list(NOUNS) + list(PROPS)
    assert len(itos) == len(set(itos)), "duplicate vocab token"
    return itos, {t: i for i, t in enumerate(itos)}


# ---------------------------------------------------------------------------------------------------
# NL rendering of a single fact / query (word-token lists). These are EXACTLY invertible.
# ---------------------------------------------------------------------------------------------------
def render_isa_fact(x, y):
    return [article(x), x, "is", article(y), y, "."]


def render_prop_fact(x, p):
    return [article(x), x, "can", p, "."]


def render_fact(atom):
    pred, (a, b) = atom
    return render_isa_fact(a, b) if pred == "isa" else render_prop_fact(a, b)


def render_query(q):
    pred, (a, b) = q
    if pred == "isa":
        return ["query", "is", article(a), a, article(b), b, "?"]
    return ["query", "can", article(a), a, b, "?"]


# ---------------------------------------------------------------------------------------------------
def config_key(edb, q):
    return (frozenset(edb), q)


def make_example(rng, qtype, isa_len=4):
    """Fresh random NL relation config + a (query, gold) of the requested type.

    Returns (edb_set, query_atom, gold_bool, meta). All labels sampled fresh per example -> no
    memorization. INHERIT positives require >=2 isa hops + the inheritance rule (multi-hop)."""
    n_isa = rng.randint(4, isa_len)            # >=4 -> a distance-2+ transitive pair always exists
    need = n_isa + 2                           # +2 spare distractor labels
    labels = rng.sample(NOUNS, need)
    isa_chain = labels[:n_isa]                 # isa_chain[0] is_a isa_chain[1] is_a ...
    spares = labels[n_isa:]

    edb = set()
    for i in range(n_isa - 1):
        edb.add(("isa", (isa_chain[i], isa_chain[i + 1])))

    # property facts: attach a property to a category at index >=1 (so a member below inherits it
    # through >=1 isa hop), plus an unrelated distractor property on a spare category.
    props = rng.sample(PROPS, 3)
    cat_idx = rng.randint(1, n_isa - 1)
    cat = isa_chain[cat_idx]
    p_inh = props[0]
    edb.add(("has_prop", (cat, p_inh)))
    edb.add(("has_prop", (spares[0], props[1])))

    meta = {"isa_chain": isa_chain, "cat_idx": cat_idx, "cat": cat, "p_inh": p_inh,
            "spares": spares, "props": props}

    if qtype == "ISA":
        if rng.random() < 0.5:                 # positive: transitive isa pair (distance >= 2)
            i = rng.randint(0, n_isa - 3)
            j = rng.randint(i + 2, n_isa - 1)
            q = ("isa", (isa_chain[i], isa_chain[j]))
        else:                                  # negative: isa(root, T) with T not an ancestor
            tgt = spares[1] if rng.random() < 0.5 else spares[0]
            q = ("isa", (isa_chain[0], tgt))
        return edb, q, DL.entails(edb, q), meta

    # INHERIT: cross-relation composition. Positive REQUIRES isa-chain (member -> cat) + R1.
    if rng.random() < 0.5:                      # positive: member strictly below cat inherits p_inh
        member = isa_chain[rng.randint(0, cat_idx - 1)]
        q = ("has_prop", (member, p_inh))
    else:                                       # negative: member with a property it does NOT inherit
        member = isa_chain[0]
        neg_p = meta["props"][2]
        q = ("has_prop", (member, neg_p))
    return edb, q, DL.entails(edb, q), meta


# ---------------------------------------------------------------------------------------------------
# Engine-derived proof rendered as a NATURAL-LANGUAGE reasoning chain.
# ---------------------------------------------------------------------------------------------------
def isa_succ(edb):
    return {a: b for (p, (a, b)) in edb if p == "isa"}


def chain_to(edb, x):
    """The ordered list of base isa edges (a,b) starting from x up the chain (real EDB sentences)."""
    succ = isa_succ(edb)
    out, cur = [], x
    while cur in succ:
        out.append((cur, succ[cur]))
        cur = succ[cur]
    return out


def render_proof_chain(edb, q, gold):
    """Render the engine-derived proof of `q` as a list of NL sentences (each a word-token list).

    POLICY (chosen to make free-running generation faithful -- see below):
      - ISA query: walk the FULL isa chain from the subject to the chain end, restating each base
        isa sentence; append "so a x is a/an y ." iff the target y was reached (entailed).
      - INHERIT query: QUERY-DRIVEN proof-tree order. Emit the recalled property leaf FIRST --
        "note a/an c can p ." for the ancestor c that bears the queried property p -- THEN the isa
        bridge from the subject up to c, THEN "so a/an x can p .".
    Negatives render the isa walk with no note/so sentence -> trace-grounded read-out returns "no".

    WHY the INHERIT order matters (measured): rendering the property check as a MID-WALK gating
    decision (walk isa, switch to the property when you hit the holder) does NOT free-run reliably --
    the model learns the dominant "continue the isa walk" continuation and never emits the property
    (train INHERIT stuck ~0.47-0.62, ISA fine at ~0.97). Emitting the recalled property leaf FIRST,
    while the query property is still fresh in context, turns the hard step into a clean associative
    recall the pointer head handles -> train INHERIT -> 1.00, held-out -> ~0.98."""
    pred, (a, b) = q
    rows = []
    reached = False
    walk = chain_to(edb, a)                                 # FULL isa walk from subject to chain end
    if pred == "isa":
        for (ca, cc) in walk:
            rows.append(render_isa_fact(ca, cc))
        if b in {cc for (_ca, cc) in walk}:                # target reached on the walk -> entailed
            reached = True
    else:  # has_prop (INHERIT): QUERY-DRIVEN order -- emit the relevant property fact FIRST (the
        # model recalls it while the query property is fresh in context), THEN the isa bridge from the
        # subject up to that holder, THEN the conclusion. This decouples the hard property-recall from
        # the isa walk: the recall is the first thing emitted, not a mid-walk gating decision.
        holders = {x for (p, (x, px)) in edb if p == "has_prop" and px == b}
        bridge, cur, target = [], a, None
        succ = isa_succ(edb)
        while cur in succ:
            nxt = succ[cur]
            bridge.append((cur, nxt))
            if nxt in holders:
                target = nxt
                break
            cur = nxt
        if target is not None:
            rows.append(["note"] + render_prop_fact(target, b))   # the recalled property leaf, FIRST
            for (ca, cc) in bridge:                                # then the isa bridge subject->holder
                rows.append(render_isa_fact(ca, cc))
            reached = True
        else:
            for (ca, cc) in walk:                                  # negative: full isa walk, no recall
                rows.append(render_isa_fact(ca, cc))
    # final "so ..." restating the queried fact iff entailed
    if reached:
        rows.append(["so"] + render_fact(q))
    assert reached == gold, ("render/oracle disagree", q, gold, rows)
    return rows


# ---------------------------------------------------------------------------------------------------
def render_prompt(edb, q, rng):
    facts = list(edb)
    rng.shuffle(facts)
    toks = []
    for atom in facts:
        toks += render_fact(atom)
    toks += render_query(q)
    return toks


def make_batch(rng, stoi, proof, device, batch=64, block=200):
    seqs, masks, configs, qtypes = [], [], [], []
    fill = stoi["<pad>"]
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
            sup_start = len(toks)
            toks += ["answer", ans, "."]
        ids = [stoi[t] for t in toks]
        if len(ids) > block:
            continue
        m = [0] * len(ids)
        if proof:
            for i in range(sup_start, len(ids)):
                m[i] = 1
        else:
            # answer-only: supervise just the yes/no token (index of `ans`)
            m[sup_start + 1] = 1
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


# ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def greedy_answer(model, prompt_toks, q, edb, stoi, itos, proof, device, max_new=80, block=200):
    """ANSWER-ONLY: read the bare yes/no token directly after the query. PROOF: generate the NL chain,
    ENGINE-CHECK each emitted fact-sentence against the closure, ground the verdict in whether the
    model derived the EXACT queried fact (read off the generated "so ..." sentence / its absence),
    with a bare-answer fallback if nothing verified."""
    pred, (a, b) = q
    ids = [stoi[t] for t in prompt_toks]
    yes_id, no_id = stoi["yes"], stoi["no"]
    answer_id, dot_id = stoi["answer"], stoi["."]
    so_id = stoi["so"]

    if not proof:
        # answer-only sequence is "answer yes/no ." ; the model must emit "answer" then yes/no.
        # generate one token (should be "answer"), then read the yes/no logit at the next position.
        ctx = torch.tensor([ids[-block:]], device=device)
        nxt = int(model(ctx)[0, -1].argmax())
        ids.append(nxt if nxt == answer_id else answer_id)
        ctx = torch.tensor([ids[-block:]], device=device)
        logits = model(ctx)[0, -1]
        return "yes" if float(logits[yes_id]) >= float(logits[no_id]) else "no"

    closure = DL.closure(set(edb))[0]
    verdict, cur, sents, saw_answer, in_so = None, [], 0, False, False
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
        if nxt == so_id:
            in_so = True
            cur = []
            continue
        if itos[nxt] == "note":                                  # marker: start of an inherited-prop step
            cur = []
            continue
        if nxt == dot_id:
            w = [itos[t] for t in cur]
            parsed = _parse_sentence(w)
            if parsed is not None and parsed in closure:        # ENGINE CHECK (reject hallucinations)
                if in_so and parsed == (pred, (a, b)):
                    verdict = "yes"
            cur, in_so = [], False
            sents += 1
            if sents > 4 * len(edb) + 8:
                break
        else:
            cur.append(nxt)
    return verdict if verdict is not None else "no"


def _parse_sentence(words):
    """Invert a rendered fact-sentence (article-stripped) -> atom, or None. EXACT (we own templates).
      [art, x, 'is', art, y]   -> ('isa', (x, y))
      [art, x, 'can', p]       -> ('has_prop', (x, p))"""
    w = [t for t in words if t not in ("a", "an")]
    if len(w) == 3 and w[1] == "is":
        return ("isa", (w[0], w[2]))
    if len(w) == 3 and w[1] == "can":
        return ("has_prop", (w[0], w[2]))
    return None


@torch.no_grad()
def evaluate(model, rng, stoi, itos, proof, device, n_eval=120, seen=None):
    model.eval()
    per = {qt: [0, 0] for qt in QTYPES}
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
    model = ScratchpadLM(vocab=len(itos), d=192, layers=4, heads=6, max_len=200,
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
        if (step + 1) % 150 == 0:
            print(f"    [{'PROOF' if proof else 'ANSWER'} seed{seed}] step {step+1}/{steps} "
                  f"loss {last:.4f}", flush=True)
    train_acc = evaluate(model, random.Random(seed + 1000), stoi, itos, proof, device, n_eval=80)
    test_acc = evaluate(model, random.Random(seed + 7000), stoi, itos, proof, device,
                        n_eval=120, seen=seen)
    return train_acc, test_acc, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3200)   # >=3200: seeds converge at different rates
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--which", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    device = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    vocab = build_vocab()
    itos, stoi = vocab

    print(f"REASON-LANG | NL-templated facts, word-tokenized | rules: R0 isa-trans, "
          f"R1 has_prop INHERITANCE | nouns={len(NOUNS)} props={len(PROPS)} | query types {QTYPES} | "
          f"steps={args.steps} lr={args.lr} device={device}")

    # show a rendered NL proof example (INHERIT, multi-hop)
    rng = random.Random(2)
    while True:
        edb, q, gold, meta = make_example(rng, "INHERIT")
        if gold and meta["cat_idx"] >= 2:        # show a genuinely multi-hop one
            break
    print("\nINHERIT example (multi-hop cross-relation composition):")
    print("  facts (shuffled):", " ".join(render_prompt(edb, q, random.Random(3))[:-7]))
    print("  query:", " ".join(render_query(q)), f" gold={'yes' if gold else 'no'}")
    rows = render_proof_chain(edb, q, gold)
    chain = "  ".join(" ".join(r) for r in rows)
    print("  rendered NL reasoning chain:", chain, " answer yes .")

    if args.sanity:
        print("\n=== SANITY GATE: PROOF must LEARN TRAIN (>0.85 overall) ===")
        tr, te, ls = train_one(True, 0, args.steps, args.lr, device, vocab)
        print(f"PROOF seed0 TRAIN acc: overall {tr['OVERALL']:.3f} "
              f"ISA {tr['ISA']:.3f} INHERIT {tr['INHERIT']:.3f} | loss {ls:.4f}")
        print(f"PROOF seed0 HELD-OUT:  overall {te['OVERALL']:.3f} "
              f"ISA {te['ISA']:.3f} INHERIT {te['INHERIT']:.3f}")
        print(f"GATE {'PASS' if tr['OVERALL'] > 0.85 else 'FAIL'} (train overall {tr['OVERALL']:.3f})")
        return

    print("\n=== A (answer-only) vs B (proof) : %d-seed held-out by query type ===" % args.seeds)
    res = {"A": [], "B": []}
    trn = {"A": [], "B": []}
    for seed in range(args.seeds):
        print(f"\n-- seed {seed} --")
        if args.which in ("A", "both"):
            trA, teA, lsA = train_one(False, seed, args.steps, args.lr, device, vocab)
            print(f"  (A) ANSWER  train {trA['OVERALL']:.3f} | HELD-OUT overall {teA['OVERALL']:.3f} "
                  f"ISA {teA['ISA']:.2f} INHERIT {teA['INHERIT']:.2f}")
            res["A"].append(teA); trn["A"].append(trA["OVERALL"])
        if args.which in ("B", "both"):
            trB, teB, lsB = train_one(True, seed, args.steps, args.lr, device, vocab)
            print(f"  (B) PROOF   train {trB['OVERALL']:.3f} | HELD-OUT overall {teB['OVERALL']:.3f} "
                  f"ISA {teB['ISA']:.2f} INHERIT {teB['INHERIT']:.2f}")
            res["B"].append(teB); trn["B"].append(trB["OVERALL"])

    def ms(xs):
        m = sum(xs) / len(xs)
        return m, (max(xs) - min(xs)) / 2

    keys = ["OVERALL", "ISA", "INHERIT"]
    print("\n===================== SUMMARY (held-out accuracy, chance=0.50) =====================")
    ta = f"{sum(trn['A'])/len(trn['A']):.3f}" if trn['A'] else "n/a"
    tb = f"{sum(trn['B'])/len(trn['B']):.3f}" if trn['B'] else "n/a"
    print(f">={args.seeds} seeds, mean +/- half-spread | train overall: A {ta} B {tb}")
    print(f"{'cell':<12s} | " + " | ".join(f"{k:^13s}" for k in keys))
    for tag, label in (("A", "ANSWER-ONLY"), ("B", "PROOF-SUPER ")):
        if not res[tag]:
            continue
        cells = []
        for k in keys:
            m, s = ms([r[k] for r in res[tag]])
            cells.append(f"{m:.2f}±{s:.2f}")
        print(f"{label:<12s} | " + " | ".join(f"{c:^13s}" for c in cells))
    print("=" * 78)


if __name__ == "__main__":
    main()
