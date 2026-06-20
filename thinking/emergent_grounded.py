#!/usr/bin/env python3
"""emergent_grounded -- emergent language GROUNDED in the provable brain.

Marries the two directions: the agent DISCOVERS its own code via self-play ([[emergent-language]]), but
the meanings are GROUNDED in the verified brain we built (datalog closure + kernel, thinking/STACK.md),
so the code means DERIVABLE TRUTH and its correctness is machine-CHECKABLE -- "we can prove it".

Game (reconstruction, grounded):
  * MEANING of an entity e = its DERIVED fact-set F(e) from the datalog closure: every isa(e,*) and
    prop(e,*) that is DERIVABLE -- including INHERITED facts the brain reasons out (robin can move
    because animals move). That derived truth is the brain's unique contribution (a non-reasoning
    grader couldn't supply it).
  * SPEAKER sees F(e) -> invents a discrete MESSAGE (Gumbel-softmax). LISTENER decodes -> predicts F(e).
  * THE BRAIN GRADES IT: every predicted fact is checked against derivable truth -> FAITHFULNESS =
    fraction of conveyed facts that are PROVABLY TRUE of e (ground truth, kernel-backed). Plus recall.
  * PRESSURE: hold out whole ENTITIES -> zero-shot: convey a NEVER-SEEN entity's derived facts by
    composing the code over shared categories/properties.

  python -m thinking.emergent_grounded --selftest
  python -m thinking.emergent_grounded --steps 3000
"""
import argparse
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thinking.prover as P  # noqa: E402  (build_kb: taxonomy + inheritance -> the brain)
from thinking.emergent import Speaker  # noqa: E402


# ===================================================================================================
# The grounded meaning space: entities -> derived fact vectors (from the brain's closure)
# ===================================================================================================
def fact_universe(kb):
    """All (pred, obj) templates that hold for some entity in the closure (any predicate -- isa/prop/
    color/size/...). The entity's fact VECTOR marks which are DERIVABLE for it."""
    ents = set(kb.entities)
    atoms = {(p, args[1]) for (p, args) in kb.known if len(args) == 2 and args[0] in ents}
    return sorted(atoms)


def rich_kb(seed=0, n_entities=200):
    """A RICHER grounded world: entities = combinations of INDEPENDENT factors (taxonomic category +
    color + size + habitat + diet). The isa-chain and inherited category-props are DERIVED by the brain
    (its contribution); color/size/habitat/diet are direct. Held-out entities are unseen COMBINATIONS,
    so a compositional code must COMPOSE to convey them -- where compositionality finally bites."""
    import itertools
    isa_tax = [("dog", "mammal"), ("cat", "mammal"), ("cow", "mammal"),
               ("robin", "bird"), ("eagle", "bird"), ("duck", "bird"),
               ("salmon", "fish"), ("trout", "fish"),
               ("mammal", "animal"), ("bird", "animal"), ("fish", "animal"), ("animal", "living")]
    cat_props = [("animal", "move"), ("animal", "breathe"), ("mammal", "nurse"),
                 ("bird", "fly"), ("fish", "swim"), ("living", "grow")]
    leaves = ["dog", "cat", "cow", "robin", "eagle", "duck", "salmon", "trout"]
    colors = ["red", "blue", "green", "brown", "white", "black"]
    sizes = ["small", "medium", "large"]
    habitats = ["forest", "water", "desert", "farm"]
    diets = ["herbivore", "carnivore", "omnivore"]
    combos = list(itertools.product(leaves, colors, sizes, habitats, diets))
    np.random.default_rng(seed).shuffle(combos)
    combos = combos[:n_entities]
    ents, facts = [], [("isa", e) for e in isa_tax] + [("prop", e) for e in cat_props]
    for i, (lf, co, sz, ha, di) in enumerate(combos):
        e = f"e{i}"; ents.append(e)
        facts += [("isa", (e, lf)), ("color", (e, co)), ("size", (e, sz)),
                  ("habitat", (e, ha)), ("diet", (e, di))]
    all_ents = sorted(set(ents) | {x for pair in isa_tax for x in pair})
    preds = {"isa": 2, "color": 2, "size": 2, "habitat": 2, "diet": 2, "prop": 2}
    rules = [(("isa", ("?x", "?z")), [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))]),
             (("prop", ("?x", "?p")), [("isa", ("?x", "?y")), ("prop", ("?y", "?p"))])]
    import thinking.lota_kernel as LK
    kb = LK.KB(all_ents, preds, facts, rules)
    kb._combo_ents = ents                                   # the combination entities (the meanings)
    return kb


def entities_with_facts(kb, atoms, min_facts=3):
    """Entities (leaves) whose DERIVED fact-set is non-trivial -- the meanings to communicate."""
    ents = []
    for e in kb.entities:
        vec = fact_vec(kb, atoms, e)
        if vec.sum() >= min_facts:
            ents.append(e)
    return sorted(ents)


def fact_vec(kb, atoms, e):
    """Binary vector: 1 where the (pred, e, obj) atom is DERIVABLE for entity e (brain closure)."""
    v = np.zeros(len(atoms), dtype=np.float32)
    for i, (pred, obj) in enumerate(atoms):
        if (pred, (e, obj)) in kb.known:
            v[i] = 1.0
    return v


def split_entities(ents, held_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ents)); rng.shuffle(idx)
    cut = max(1, int(len(ents) * (1 - held_frac)))
    return [ents[i] for i in idx[:cut]], [ents[i] for i in idx[cut:]]


# ===================================================================================================
# Listener: message -> predicted fact-set (reconstruction)
# ===================================================================================================
class ReconListener(nn.Module):
    def __init__(self, L, V, n_facts, d=128):
        super().__init__()
        self.sym = nn.Linear(V, d, bias=False)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_facts))

    def forward(self, msg):
        _, hn = self.gru(self.sym(msg))
        return self.out(hn.squeeze(0))                  # (B, n_facts) logits


# ===================================================================================================
# Train (speaker + listener jointly; BCE reconstruction of the brain-derived fact-set)
# ===================================================================================================
def make_batch(rng, vecs, batch):
    idx = rng.integers(0, len(vecs), size=batch)
    return torch.from_numpy(np.stack([vecs[i] for i in idx]))


def train(spk, lis, vecs, steps, batch=64, lr=1e-3, seed=0, tau=1.5, log=0,
          reset_every=0, make_lis=None):
    """Joint training. ITERATED LEARNING (ease-of-teaching): if reset_every>0 and make_lis is given,
    re-initialize the LISTENER every reset_every steps -> the speaker must keep its code easy to teach
    a fresh learner, a transmission bottleneck that favors COMPOSITIONAL (compressible) codes."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt_s = torch.optim.Adam(spk.parameters(), lr=lr)
    opt_l = torch.optim.Adam(lis.parameters(), lr=lr)
    spk.train(); lis.train()
    for step in range(steps):
        if reset_every and make_lis and step > 0 and step % reset_every == 0:
            lis = make_lis(); lis.train()                       # fresh listener (new generation)
            opt_l = torch.optim.Adam(lis.parameters(), lr=lr)
        x = make_batch(rng, vecs, batch)
        msg, _ = spk(x, tau=tau, hard=True)
        logits = lis(msg)
        loss = F.binary_cross_entropy_with_logits(logits, x)
        opt_s.zero_grad(); opt_l.zero_grad(); loss.backward(); opt_s.step(); opt_l.step()
        if log and (step % log == 0 or step == steps - 1):
            acc = ((logits > 0).float() == x).float().mean().item()
            print(f"  step {step:5d}  loss {loss.item():.3f}  bit-acc {acc:.3f}", flush=True)
    return spk, lis


# ===================================================================================================
# Eval: reconstruction + BRAIN-VERIFIED faithfulness/recall + topsim, on train and HELD-OUT entities
# ===================================================================================================
@torch.no_grad()
def evaluate(spk, lis, kb, atoms, ents, thresh=0.5):
    spk.eval(); lis.eval()
    vecs = np.stack([fact_vec(kb, atoms, e) for e in ents])
    x = torch.from_numpy(vecs)
    msg, _ = spk(x)
    pred = (torch.sigmoid(lis(msg)) > thresh).numpy().astype(np.float32)
    exact = float((pred == vecs).all(axis=1).mean())            # whole fact-set exactly right
    # BRAIN-VERIFIED faithfulness: of the facts the message conveys, how many are PROVABLY TRUE of e?
    faith_ok = faith_tot = recall_ok = recall_tot = 0
    for r, e in enumerate(ents):
        for i in np.where(pred[r] > 0)[0]:
            faith_tot += 1
            pr, obj = atoms[i]
            if (pr, (e, obj)) in kb.known:                      # the brain confirms it (kernel-backed)
                faith_ok += 1
        recall_ok += int((pred[r] * vecs[r]).sum()); recall_tot += int(vecs[r].sum())
    faith = faith_ok / max(1, faith_tot)
    recall = recall_ok / max(1, recall_tot)
    return dict(n=len(ents), exact=exact, faithfulness=faith, recall=recall)


def topsim(spk, kb, atoms, ents, seed=2, pairs=2000):
    spk.eval()
    vecs = np.stack([fact_vec(kb, atoms, e) for e in ents])
    msgs = spk(torch.from_numpy(vecs))[0].argmax(-1).numpy()
    rng = np.random.default_rng(seed); md, sd = [], []
    for _ in range(pairs):
        i, j = rng.integers(0, len(ents)), rng.integers(0, len(ents))
        if i == j:
            continue
        md.append(int(np.abs(vecs[i] - vecs[j]).sum()))         # fact-set Hamming
        sd.append(int((msgs[i] != msgs[j]).sum()))              # message Hamming
    md, sd = np.array(md, float), np.array(sd, float)

    def rank(a):
        o = a.argsort(); r = np.empty_like(o, float); r[o] = np.arange(len(a)); return r
    if md.std() < 1e-9 or sd.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rank(md), rank(sd))[0, 1])


# ===================================================================================================
# selftest + run
# ===================================================================================================
def _setup(L, V, d, seed, rich=False):
    kb = rich_kb(seed=seed) if rich else P.build_kb()
    atoms = fact_universe(kb)
    ents = getattr(kb, "_combo_ents", None) or entities_with_facts(kb, atoms)
    tr, te = split_entities(ents, 0.25, seed)
    spk = Speaker(len(atoms), 1, L, V, d=d)             # speaker input = the fact vector (n_attr*n_val
    spk.enc[0] = nn.Linear(len(atoms), d)              # ... here that product == len(atoms))
    lis = ReconListener(L, V, len(atoms), d=d)
    return kb, atoms, ents, tr, te, spk, lis


def selftest():
    torch.set_num_threads(2)
    kb, atoms, ents, tr, te, spk, lis = _setup(L=4, V=8, d=64, seed=0)
    assert len(atoms) > 5 and len(tr) >= 3 and len(te) >= 1, (len(atoms), len(tr), len(te))
    tr_vecs = [fact_vec(kb, atoms, e) for e in tr]
    train(spk, lis, tr_vecs, steps=400, batch=32, seed=0)
    r = evaluate(spk, lis, kb, atoms, tr)
    assert r["faithfulness"] >= 0.0 and r["recall"] > 0.3, ("should convey some true facts", r)
    # the grounding is REAL: a conveyed fact is checked against the brain's closure (kernel-backed)
    assert ("isa", ("robin", "animal")) in kb.known, "brain derives inherited/transitive facts"
    print("emergent_grounded selftest OK")


def run(steps, seed=0, L=6, V=12, d=128, verbose=True):
    kb, atoms, ents, tr, te, spk, lis = _setup(L, V, d, seed)
    if verbose:
        print(f"brain KB: {len(ents)} entities w/ derived facts | fact-universe {len(atoms)} atoms | "
              f"train {len(tr)} / held-out {len(te)} entities | msg L={L} V={V} | steps {steps}",
              flush=True)
    tr_vecs = [fact_vec(kb, atoms, e) for e in tr]
    train(spk, lis, tr_vecs, steps=steps, batch=64, seed=seed,
          log=(max(1, steps // 8) if verbose else 0))
    rtr = evaluate(spk, lis, kb, atoms, tr)
    rte = evaluate(spk, lis, kb, atoms, te)                     # ZERO-SHOT: never-seen entities
    ts = topsim(spk, kb, atoms, ents)
    if verbose:
        print("\n== GROUNDED EMERGENCE (code grounded in the provable brain) ==")
        print(f"  TRAIN    : exact-factset {rtr['exact']:.3f} | brain-verified faithfulness "
              f"{rtr['faithfulness']:.3f} | recall {rtr['recall']:.3f}")
        print(f"  HELD-OUT : exact-factset {rte['exact']:.3f} | brain-verified faithfulness "
              f"{rte['faithfulness']:.3f} | recall {rte['recall']:.3f}   (never-seen entities)")
        print(f"  topsim {ts:.3f}  (>0 => compositional). Faithfulness is GROUND TRUTH: each conveyed "
              f"fact is checked against the brain's closure (kernel-backed).")
    return dict(train=rtr, heldout=rte, topsim=ts)


def _consolidate(spk, kb, atoms, tr, L, V, d, steps=2000, seed=0):
    """Freeze the (now-compositional) speaker; train a FRESH listener to convergence on its code, so
    held-out reconstruction reflects the code's structure, not an undertrained decoder."""
    for p in spk.parameters():
        p.requires_grad_(False)
    spk.eval()
    lis = ReconListener(L, V, len(atoms), d=d); lis.train()
    opt = torch.optim.Adam(lis.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed + 7)
    vecs = [fact_vec(kb, atoms, e) for e in tr]
    for _ in range(steps):
        x = make_batch(rng, vecs, 64)
        with torch.no_grad():
            msg, _ = spk(x, tau=1.0, hard=True)
        loss = F.binary_cross_entropy_with_logits(lis(msg), x)
        opt.zero_grad(); loss.backward(); opt.step()
    return lis


def iterated_compare(steps, seed=0, L=6, V=12, d=128, reset_every=800, rich=False):
    """Baseline (no reset) vs ITERATED LEARNING (periodic listener reset), same total steps. The
    iterated arm then CONSOLIDATES (fresh listener trained to convergence on the compositional code)
    so held-out reconstruction reflects the code, not an undertrained decoder."""
    out = {}
    for tag, rev in [("baseline", 0), ("iterated", reset_every)]:
        kb, atoms, ents, tr, te, spk, lis = _setup(L, V, d, seed, rich=rich)
        if tag == "baseline":
            print(f"  world: {'RICH' if rich else 'toy'} | {len(ents)} meaning-entities "
                  f"(train {len(tr)}/held-out {len(te)}) | fact-universe {len(atoms)} | L{L} V{V}",
                  flush=True)
        mk = (lambda: ReconListener(L, V, len(atoms), d=d)) if rev else None
        train(spk, lis, [fact_vec(kb, atoms, e) for e in tr], steps=steps, batch=64, seed=seed,
              reset_every=rev, make_lis=mk)
        if rev:                                              # consolidate the iterated code's decoder
            lis = _consolidate(spk, kb, atoms, tr, L, V, d, steps=2000, seed=seed)
        rte = evaluate(spk, lis, kb, atoms, te)
        out[tag] = dict(heldout_exact=rte["exact"], heldout_faith=rte["faithfulness"],
                        heldout_recall=rte["recall"], topsim=topsim(spk, kb, atoms, ents))
    print("ITERATED LEARNING (ease-of-teaching: reset listener) vs baseline:")
    for tag in ("baseline", "iterated"):
        o = out[tag]
        print(f"  {tag:>9}: held-out exact {o['heldout_exact']:.3f} | faith {o['heldout_faith']:.3f} "
              f"| recall {o['heldout_recall']:.3f} | topsim {o['topsim']:.3f}")
    d_ts = out["iterated"]["topsim"] - out["baseline"]["topsim"]
    d_ex = out["iterated"]["heldout_exact"] - out["baseline"]["heldout_exact"]
    print(f"  delta (iterated - baseline): topsim {d_ts:+.3f}, held-out exact {d_ex:+.3f}")
    return out


@torch.no_grad()
def teacher_messages(spk, vecs):
    spk.eval()
    return spk(torch.from_numpy(np.stack(vecs)))[0].argmax(-1)        # (N, L) symbol ids


def imitate(new_spk, vecs, tmsg, steps, lr=1e-3, seed=0):
    """A fresh speaker learns the previous speaker's code on a SUBSET (the transmission bottleneck)."""
    rng = np.random.default_rng(seed); opt = torch.optim.Adam(new_spk.parameters(), lr=lr)
    X = torch.from_numpy(np.stack(vecs)); new_spk.train()
    for _ in range(steps):
        idx = rng.integers(0, len(vecs), size=min(64, len(vecs)))
        _, logits = new_spk(X[idx], tau=1.0, hard=True)              # (B, L, V) logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tmsg[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return new_spk


def iterated_bottleneck(steps_per_gen, gens=5, rich=True, L=8, V=16, d=128, seed=0, frac=0.5):
    """TRUE iterated learning (Kirby transmission bottleneck): each generation a FRESH speaker imitates
    the previous one on a SUBSET, then plays the game. Only compositional codes survive learning from a
    subset + generalizing. Track topsim + held-out recall across generations on the rich world."""
    kb, atoms, ents, tr, te, spk, lis = _setup(L, V, d, seed, rich=rich)
    tr_vecs = [fact_vec(kb, atoms, e) for e in tr]
    print(f"  world: {'RICH' if rich else 'toy'} | {len(ents)} entities (train {len(tr)}/held {len(te)})"
          f" | bottleneck frac {frac} | {gens} generations x {steps_per_gen} steps", flush=True)
    train(spk, lis, tr_vecs, steps=steps_per_gen, seed=seed)

    def ev():
        r = evaluate(spk, lis, kb, atoms, te)
        return topsim(spk, kb, atoms, ents), r["recall"], r["faithfulness"], r["exact"]
    ts, rc, fa, ex = ev()
    print(f"  gen 0 : topsim {ts:.3f} | held-out recall {rc:.3f} | faith {fa:.3f} | exact {ex:.3f}")
    rng = np.random.default_rng(seed + 1)
    for g in range(1, gens + 1):
        k = max(2, int(frac * len(tr_vecs)))
        sub = [tr_vecs[i] for i in rng.choice(len(tr_vecs), k, replace=False)]
        tmsg = teacher_messages(spk, sub)
        new = Speaker(len(atoms), 1, L, V, d=d); new.enc[0] = nn.Linear(len(atoms), d)
        imitate(new, sub, tmsg, steps=max(1, steps_per_gen // 2), seed=seed + g)   # bottleneck
        lis = ReconListener(L, V, len(atoms), d=d)
        train(new, lis, tr_vecs, steps=steps_per_gen, seed=seed + g)               # game fine-tune
        spk = new
        ts, rc, fa, ex = ev()
        print(f"  gen {g} : topsim {ts:.3f} | held-out recall {rc:.3f} | faith {fa:.3f} | exact {ex:.3f}")
    return dict(topsim=ts, recall=rc, faith=fa, exact=ex)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--msg-len", type=int, default=6)
    ap.add_argument("--vocab", type=int, default=12)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--iterated", action="store_true",
                    help="compare iterated learning (listener resets) vs baseline")
    ap.add_argument("--reset-every", type=int, default=800)
    ap.add_argument("--rich", action="store_true",
                    help="use the richer grounded world (independent factors; large held-out)")
    ap.add_argument("--il-bottleneck", action="store_true",
                    help="TRUE iterated learning (Kirby transmission bottleneck) across generations")
    ap.add_argument("--gens", type=int, default=5)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    if args.il_bottleneck:
        iterated_bottleneck(args.steps, gens=args.gens, rich=args.rich, L=args.msg_len, V=args.vocab,
                            d=args.d, seed=args.seed); return 0
    if args.iterated:
        iterated_compare(args.steps, seed=args.seed, L=args.msg_len, V=args.vocab, d=args.d,
                         reset_every=args.reset_every, rich=args.rich); return 0
    run(args.steps, seed=args.seed, L=args.msg_len, V=args.vocab, d=args.d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
