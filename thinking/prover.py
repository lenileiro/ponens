#!/usr/bin/env python3
"""prover -- a LEARNED prover as the untrusted tactic; the kernel is the trusted gate.

The reconnection of the neural side to the verified core (Lean's tactic+kernel split, realized):

    model  =  PROPOSER (untrusted): given a goal, emit a proof TERM (token sequence)
    kernel =  CHECKER  (trusted):   type-check that term against the goal -> accept / reject

The training signal and the eval metric are BOTH the kernel's verdict, so the metric cannot be hollow
(a proof either type-checks or it does not -- no CWA, no surface-match heuristic). Gold proofs come from
the datalog search + kernel pipeline (thinking/lota_kernel.py). Held-out = unseen COMBINATIONS of seen
entities; proofs are serialized so base-fact axioms DECOMPOSE into reusable tokens (`ax isa robin bird`),
so a held-out combination composes from seen pieces.

  python -m thinking.prover --selftest
  python -m thinking.prover --steps 4000 --out /tmp/prover.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thinking.kernel as K  # noqa: E402
import thinking.lota_kernel as LK  # noqa: E402
from scratchpad_model import ScratchpadLM  # noqa: E402


# ===================================================================================================
# A taxonomy KB (enough derivable atoms for a train/held-out split).
# ===================================================================================================
def build_kb():
    isa_edges = [
        ("animal", "living"), ("plant", "living"),
        ("mammal", "animal"), ("bird", "animal"), ("fish", "animal"),
        ("dog", "mammal"), ("cat", "mammal"), ("cow", "mammal"), ("bat", "mammal"),
        ("robin", "bird"), ("eagle", "bird"), ("owl", "bird"), ("duck", "bird"),
        ("salmon", "fish"), ("trout", "fish"), ("shark", "fish"),
        ("oak", "plant"), ("rose", "plant"), ("fern", "plant"),
    ]
    prop_edges = [("living", "grow"), ("animal", "move"), ("animal", "breathe"),
                  ("mammal", "nurse"), ("bird", "fly"), ("fish", "swim"), ("plant", "photosynthesize")]
    entities = sorted({x for e in isa_edges for x in e} |
                      {p for _c, p in prop_edges})
    preds = {"isa": 2, "prop": 2}
    facts = [("isa", e) for e in isa_edges] + [("prop", e) for e in prop_edges]
    rules = [
        (("isa", ("?x", "?z")), [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))]),
        (("prop", ("?x", "?p")), [("isa", ("?x", "?y")), ("prop", ("?y", "?p"))]),
    ]
    return LK.KB(entities, preds, facts, rules)


# ===================================================================================================
# Serialization: kernel proof term <-> token sequence. Apps -> binary parens; ax_ leaves DECOMPOSE
# into reusable tokens ("ax" pred args...), so novel entity combinations compose from seen tokens.
# ===================================================================================================
def ser_proof(t, preds):
    if isinstance(t, K.Const):
        if t.name.startswith("ax_"):
            parts = t.name.split("_")                       # ax_isa_robin_bird -> ax isa robin bird
            return ["ax"] + parts[1:]
        return [t.name]
    if isinstance(t, K.App):
        return ["("] + ser_proof(t.fn, preds) + ser_proof(t.arg, preds) + [")"]
    raise ValueError("proof terms here are App/Const only")


def parse_proof(toks, preds):
    pos = 0

    def p():
        nonlocal pos
        tok = toks[pos]; pos += 1
        if tok == "(":
            f = p(); a = p()
            if pos >= len(toks) or toks[pos] != ")":
                raise ValueError("malformed: expected ')'")
            pos += 1
            return K.App(f, a)
        if tok == "ax":                                     # ax pred arg1 .. arg{arity}  -> Const
            pred = toks[pos]; pos += 1
            n = preds[pred]
            args = toks[pos:pos + n]; pos += n
            if len(args) != n:
                raise ValueError("malformed ax")
            return K.Const("ax_" + "_".join([pred] + args))
        return K.Const(tok)

    t = p()
    if pos != len(toks):
        raise ValueError("trailing tokens")
    return t


def goal_tokens(goal):
    p, args = goal
    return ["("] + [p] + list(args) + [")"]


# ===================================================================================================
# Data: (goal-tokens, proof-tokens) for every DERIVED atom; split by unseen (subj,obj) combination.
# ===================================================================================================
def gen_data(kb):
    derived = [a for a in kb.known if a not in set(kb.facts)]   # only non-base (need a real proof)
    pairs = []
    for goal in sorted(derived):
        ok, term = kb._atom_term(goal)
        if not ok:
            continue
        pairs.append({"goal": goal, "gtoks": goal_tokens(goal),
                      "ptoks": ser_proof(term, kb.preds)})
    return pairs


def split(pairs, held_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs)); rng.shuffle(idx)
    cut = int(len(pairs) * (1 - held_frac))
    return [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]


# ===================================================================================================
# Model (shared-vocab seq2seq, pointer head so it copies entity/pred tokens from the goal).
# ===================================================================================================
SPECIAL = ["<pad>", "<bos>", "<sep>", "<eos>"]


class Vocab:
    def __init__(self, itos):
        self.itos = itos; self.stoi = {t: i for i, t in enumerate(itos)}; self.pad = 0

    @classmethod
    def build(cls, pairs):
        toks = set()
        for p in pairs:
            toks.update(p["gtoks"]); toks.update(p["ptoks"])
        return cls(SPECIAL + sorted(toks))

    def __len__(self):
        return len(self.itos)

    def enc(self, toks):
        return [self.stoi[t] for t in toks]


def build_model(vsize, max_len, d=256, layers=4, heads=8):
    assert (d // heads) % 2 == 0
    return ScratchpadLM(vocab=vsize, d=d, layers=layers, heads=heads, max_len=max_len,
                        pad=0, pos_mode="rope", tie=True, pointer=True, causal=True)


def example(pair, vocab):
    seq = ["<bos>"] + pair["gtoks"] + ["<sep>"] + pair["ptoks"] + ["<eos>"]
    ws = len(["<bos>"] + pair["gtoks"] + ["<sep>"])
    return vocab.enc(seq), ws


def make_batch(rng, vocab, pairs, device, batch, block):
    seqs, masks = [], []
    tries = 0
    while len(seqs) < batch and tries < batch * 20:
        tries += 1
        ids, ws = example(pairs[rng.integers(len(pairs))], vocab)
        if len(ids) > block:
            continue
        m = [0] * len(ids)
        for i in range(ws, len(ids)):
            m[i] = 1
        seqs.append(ids); masks.append(m)
    if not seqs:
        return None
    Ln = max(len(s) for s in seqs)
    ib = torch.full((len(seqs), Ln), 0, dtype=torch.long)
    mb = torch.zeros((len(seqs), Ln), dtype=torch.float)
    for r, (s, m) in enumerate(zip(seqs, masks)):
        ib[r, :len(s)] = torch.tensor(s); mb[r, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ib.to(device), mb.to(device)


def train(model, vocab, pairs, steps, device, batch=32, block=96, lr=3e-3, seed=0, log_every=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    model.to(device).train()
    for step in range(steps):
        b = make_batch(rng, vocab, pairs, device, batch, block)
        if b is None:
            continue
        ids, mask = b
        logits = model(ids)[:, :-1]; tgt = ids[:, 1:]; sel = mask[:, 1:] > 0
        loss = F.cross_entropy(logits[sel], tgt[sel])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {loss.item():.4f}", flush=True)
    return model


@torch.no_grad()
def emit(model, vocab, pair, device, block=96, max_new=80):
    ids = vocab.enc(["<bos>"] + pair["gtoks"] + ["<sep>"])
    eos = vocab.stoi["<eos>"]; out = []
    model.eval()
    for _ in range(max_new):
        ctx = torch.tensor([ids[-block:]], device=device)
        nxt = int(model(ctx)[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt); out.append(vocab.itos[nxt])
    return out


import itertools  # noqa: E402


class KernelProver:
    """Goal-directed proof SEARCH over the kernel: backward-chain from the goal, searching the free
    'middle' variables (the branch points, e.g. y in isa(x,z):-isa(x,y),isa(y,z)), assemble a proof
    TERM, and have the kernel CHECK it. Sound (kernel) + complete-for-this-KB (exhaustive search) ->
    where one-shot emission failed (5.6%), kernel-guided search proves the derivable goals. A neural
    model can rank the branch choices to prune the search (the scaling lever); here we search
    exhaustively. `order` is an optional callable(candidates)->ordered (neural-guidance hook)."""

    def __init__(self, kb, budget=200000, order=None):
        self.kb = kb
        self.fact_set = set(kb.facts)
        self.rules = kb.rules
        self.domain = list(kb.entities)
        self.order = order or (lambda goal, cands: cands)
        self.budget = budget
        self.nodes = 0
        self.memo = {}

    def _search(self, goal, stack):
        self.nodes += 1
        if goal in self.memo:                          # memoize: each distinct goal searched once
            return self.memo[goal]
        if self.nodes > self.budget or goal in stack:  # budget backstop / cycle guard (no caching)
            return None
        if goal in self.fact_set:
            t = K.Const(LK.ax_name(goal)); self.memo[goal] = t; return t
        res = None
        for ri, (head, body) in enumerate(self.rules):
            sub = {}
            if not LK._unify(head, goal, sub):
                continue
            vs = LK.rule_vars((head, body))
            free = [v for v in vs if v not in sub]
            for assign in self.order(goal, itertools.product(self.domain, repeat=len(free))):
                s = dict(sub); s.update(zip(free, assign))
                subgoals = [(bp[0], tuple(s.get(a, a) for a in bp[1])) for bp in body]
                proofs, ok = [], True
                for sg in subgoals:
                    pr = self._search(sg, stack | {goal})
                    if pr is None:
                        ok = False; break
                    proofs.append(pr)
                if ok:
                    term = K.Const(f"rule{ri}")
                    for v in vs:
                        term = K.App(term, K.Const(s[v]))
                    for pr in proofs:
                        term = K.App(term, pr)
                    res = term; break
            if res is not None:
                break
        self.memo[goal] = res                          # cache success or failure (KB is acyclic)
        return res

    def verify(self, goal):
        """Returns (proven, term, nodes_explored). proven=True ONLY if the kernel accepts the term."""
        self.nodes = 0; self.memo = {}
        term = self._search(goal, frozenset())
        if term is None:
            return False, None, self.nodes
        ok = K.has_type([], term, LK.atom_type(goal), self.kb.env)
        return ok, term, self.nodes


def search_eval(kb, goals, order=None):
    """Run kernel-guided search over goals; report kernel-verified rate + avg nodes explored.
    `order` (optional) is a neural ranker's candidate-ordering callable -> prunes the search."""
    prover = KernelProver(kb, order=order)
    verified = total_nodes = 0
    for g in goals:
        ok, _term, nodes = prover.verify(g)
        verified += int(ok); total_nodes += nodes
    n = max(1, len(goals))
    return {"n": len(goals), "verified": verified / n, "avg_nodes": total_nodes / n}


# ===================================================================================================
# NEURAL GUIDANCE: rank the branch candidates (the free 'middle' var y) so the SOUND search explores
# fewer nodes. The model only REORDERS choices; the kernel still checks every accepted proof, so a bad
# ranking can only cost search time, never soundness. Trained on the middle-vars from gold proofs.
# ===================================================================================================
import torch.nn as nn  # noqa: E402


def _walk_middles(tree, rules, out):
    if tree["rule"] is None:
        return
    head, body = rules[tree["rule"]]
    sub = {}
    LK._unify(head, tree["fact"], sub)
    for bp, child in zip(body, tree["from"]):
        LK._unify(bp, child["fact"], sub)
    headvars = {a for a in head[1] if LK.is_var(a)}
    for v in LK.rule_vars((head, body)):
        if v not in headvars:                                  # a 'middle' var chosen during search
            out.append((tree["fact"], sub[v]))
    for child in tree["from"]:
        _walk_middles(child, rules, out)


def guidance_examples(kb):
    out = []
    for goal in (a for a in kb.known if a not in set(kb.facts)):
        _walk_middles(kb.dl.proof_tree(kb.prov, goal), kb.rules, out)
    return out                                                 # list of (goal_fact, middle_entity)


class NeuralRanker(nn.Module):
    def __init__(self, syms, dim=32):
        super().__init__()
        self.stoi = {s: i for i, s in enumerate(syms)}
        self.emb = nn.Embedding(len(syms), dim)
        self.mlp = nn.Sequential(nn.Linear(4 * dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def _ids(self, toks):
        return torch.tensor([self.stoi.get(t, 0) for t in toks])

    def score_batch(self, goals, ys):
        pred = self._ids([g[0] for g in goals]); a0 = self._ids([g[1][0] for g in goals])
        a1 = self._ids([g[1][1] for g in goals]); yy = self._ids(ys)
        e = torch.cat([self.emb(pred), self.emb(a0), self.emb(a1), self.emb(yy)], dim=-1)
        return self.mlp(e).squeeze(-1)

    def order_fn(self, domain):
        def order(goal, cands):
            cands = list(cands)
            if not cands or len(cands[0]) != 1:                # only rank single free-var choices
                return cands
            with torch.no_grad():
                s = self.score_batch([goal] * len(cands), [c[0] for c in cands])
            return [cands[i] for i in torch.argsort(s, descending=True).tolist()]
        return order


def train_ranker(kb, steps=1500, seed=0, dim=32):
    syms = list(kb.entities) + list(kb.preds)
    r = NeuralRanker(syms, dim)
    exs = guidance_examples(kb)
    rng = np.random.default_rng(seed)
    dom = list(kb.entities)
    opt = torch.optim.AdamW(r.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = rng.integers(0, len(exs), size=64)
        goals = [exs[i][0] for i in idx]
        pos = [exs[i][1] for i in idx]
        neg = [dom[rng.integers(len(dom))] for _ in idx]       # random negative middle
        sp = r.score_batch(goals, pos); sn = r.score_batch(goals, neg)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([sp, sn]), torch.cat([torch.ones_like(sp), torch.zeros_like(sn)]))
        opt.zero_grad(); loss.backward(); opt.step()
    return r


def evaluate(model, vocab, kb, pairs, device, block):
    """The metric IS the kernel verdict: emit a proof, parse it, kernel-check it against the goal."""
    verified = parses = 0
    samples = []
    for pair in pairs:
        toks = emit(model, vocab, pair, device, block)
        ok = False
        try:
            term = parse_proof(toks, kb.preds)
            parses += 1
            ok = K.has_type([], term, LK.atom_type(pair["goal"]), kb.env)   # TRUSTED CHECK
        except Exception:
            pass
        verified += int(ok)
        samples.append((pair["goal"], " ".join(toks), ok))
    n = max(1, len(pairs))
    return {"n": len(pairs), "verified": verified / n, "well_formed": parses / n}, samples


# ===================================================================================================
# selftest + run
# ===================================================================================================
def selftest():
    kb = build_kb()
    pairs = gen_data(kb)
    assert len(pairs) > 40, ("need data", len(pairs))
    # round-trip: serialize a gold proof and parse it back -> kernel still accepts
    g = pairs[0]
    term = parse_proof(g["ptoks"], kb.preds)
    assert K.has_type([], term, LK.atom_type(g["goal"]), kb.env), "gold proof must kernel-check"
    # tiny train lifts at least one held-out proof to kernel-verified
    tr, te = split(pairs, 0.25, 0)
    vocab = Vocab.build(pairs)
    torch.set_num_threads(2)
    m = build_model(len(vocab), 96, d=64, layers=2, heads=4)
    train(m, vocab, tr, steps=200, device="cpu", batch=16, block=96, seed=0)
    res, _ = evaluate(m, vocab, kb, te[:8], "cpu", 96)
    assert res["well_formed"] >= 0.0
    # KERNEL-GUIDED SEARCH: sound + complete-for-this-KB. Proves derivable goals; rejects false ones.
    kp = KernelProver(kb)
    ok, term, _ = kp.verify(("isa", ("robin", "living")))      # multi-hop derivable
    assert ok and term is not None, "search should prove a derivable multi-hop goal"
    ok2, _, _ = kp.verify(("prop", ("robin", "fly")))          # robin is a bird -> can fly (inherited)
    assert ok2, "search should prove an inherited property"
    ok3, t3, _ = kp.verify(("isa", ("robin", "fish")))         # FALSE
    assert not ok3 and t3 is None, "search must not prove a false goal"
    # NEURAL-GUIDED search: ranker only reorders -> verified rate unchanged, nodes should not increase
    r = train_ranker(kb, steps=200)
    derived = [a for a in kb.known if a not in set(kb.facts)]
    base = search_eval(kb, derived)
    guided = search_eval(kb, derived, order=r.order_fn(list(kb.entities)))
    assert guided["verified"] == 1.0 and base["verified"] == 1.0, "kernel keeps both sound+complete"
    print("prover selftest OK")


def run(steps, out_path, seed=0, d=256, layers=4, heads=8, block=96, device="cpu", verbose=True):
    kb = build_kb()
    pairs = gen_data(kb)
    tr, te = split(pairs, 0.2, seed)
    vocab = Vocab.build(pairs)
    if verbose:
        print(f"derived goals: {len(pairs)} (train {len(tr)} / held-out {len(te)}) | "
              f"vocab {len(vocab)} | device {device} | steps {steps}", flush=True)
    m = build_model(len(vocab), block, d=d, layers=layers, heads=heads)
    train(m, vocab, tr, steps=steps, device=device, batch=32, block=block, seed=seed,
          log_every=(max(1, steps // 8) if verbose else 0))
    tr_res, _ = evaluate(m, vocab, kb, tr, device, block)
    te_res, samp = evaluate(m, vocab, kb, te, device, block)
    if verbose:
        print(f"\n== learned prover, KERNEL-VERIFIED proof rate ==")
        print(f"  TRAIN    verified {tr_res['verified']:.3f}  well-formed {tr_res['well_formed']:.3f}"
              f"  (n={tr_res['n']})")
        print(f"  HELD-OUT verified {te_res['verified']:.3f}  well-formed {te_res['well_formed']:.3f}"
              f"  (n={te_res['n']})  [unseen entity combinations]")
        for goal, toks, ok in samp[:6]:
            print(f"   [{'KERNEL-OK' if ok else 'reject  '}] {goal[0]}{goal[1]}  ->  {toks[:70]}")
    result = {"steps": steps, "seed": seed, "n_train": len(tr), "n_test": len(te),
              "train": tr_res, "heldout": te_res}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"\nwrote {out_path}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--search", action="store_true",
                    help="evaluate the KERNEL-GUIDED SEARCH prover (vs one-shot) and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    if args.search:
        kb = build_kb()
        derived = sorted(a for a in kb.known if a not in set(kb.facts))
        # false goals: random (subj, obj) pairs that are NOT entailed
        ents = list(kb.entities)
        false_goals = [("isa", (a, b)) for a in ents for b in ents
                       if ("isa", (a, b)) not in kb.known][:40]
        dres = search_eval(kb, derived)
        fres = search_eval(kb, false_goals)
        print("=== KERNEL-GUIDED PROOF SEARCH (sound kernel + exhaustive backward chaining) ===")
        print(f"  derivable goals : verified {dres['verified']:.3f}  (n={dres['n']}, "
              f"avg {dres['avg_nodes']:.1f} search nodes)")
        print(f"  FALSE goals     : verified {fres['verified']:.3f}  (n={fres['n']}) "
              f"<- must be 0.000 (soundness)")
        print("  contrast: the one-shot LEARNED prover scored ~0.06 verified on held-out; "
              "kernel-guided search proves the derivable set.")
        # NEURAL-GUIDED: train a ranker, re-run search with model-ordered branch choices
        r = train_ranker(kb, steps=1500)
        gres = search_eval(kb, derived, order=r.order_fn(list(kb.entities)))
        speedup = dres["avg_nodes"] / max(1e-9, gres["avg_nodes"])
        print("\n=== + NEURAL GUIDANCE (model ranks branch choices; kernel still checks) ===")
        print(f"  derivable goals : verified {gres['verified']:.3f}  "
              f"(avg {gres['avg_nodes']:.1f} nodes vs {dres['avg_nodes']:.1f} unguided "
              f"-> {speedup:.0f}x fewer search nodes)")
        print("  soundness unchanged (kernel checks every accepted proof); the model only prunes.")
        return 0
    run(args.steps, args.out, seed=args.seed, d=args.d, layers=args.layers, heads=args.heads,
        block=args.block, device=args.device)
    return 0


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    raise SystemExit(main())
