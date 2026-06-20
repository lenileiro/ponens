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


def wordnet_kb(seed=0, per_cat=25):
    """REAL grounded data across MANY WordNet relational DIMENSIONS (not just is-a): a concept's meaning
    spans is-a (hypernym), has-part (meronym), made-of (substance) and member-of (holonym). The brain
    holds the RULES over these dimensions -- is-a transitivity PLUS has-part/made-of/member-of being
    INHERITED down the is-a hierarchy (a bird has a wing => a robin has a wing) -- so a concept's full
    multi-relation fact-set is DERIVED and kernel/closure-VERIFIED from a small core. Using all the
    dimensions makes the core MULTI-HOT (not the one-hot is-a chain that collapsed precision) and gives
    the agent real WordNet structure to learn the rules over."""
    from nltk.corpus import wordnet as wn
    seeds = ["animal.n.01", "plant.n.02", "food.n.01", "vehicle.n.01", "tool.n.01",
             "furniture.n.01", "clothing.n.01", "bird.n.01"]
    rng = np.random.default_rng(seed)

    def descendants(syn, cap=400):
        seen, frontier = set(), [syn]
        while frontier and len(seen) < cap:
            nxt = []
            for s in frontier:
                for h in s.hyponyms():
                    if h not in seen:
                        seen.add(h); nxt.append(h)
            frontier = nxt
        return seen
    leaves = []
    for sd in seeds:
        desc = [s for s in descendants(wn.synset(sd)) if not s.hyponyms()]
        rng.shuffle(desc)
        leaves += desc[:per_cat]
    # one extractor per relational DIMENSION (each maps a synset -> related synsets along that axis)
    relmap = {"has_part": lambda s: s.part_meronyms(),
              "made_of": lambda s: s.substance_meronyms(),
              "member_of": lambda s: s.member_holonyms()}
    nodes, isa = set(), set()
    for lf in leaves:
        path = lf.hypernym_paths()[0]                         # root..leaf
        nodes.update(path)
        for parent, child in zip(path, path[1:]):
            isa.add((child.name(), parent.name()))
    facts = [("isa", e) for e in isa]
    # attach each relation's edges where they HANG (ancestor or leaf) -> the brain inherits them down
    for pred, fn in relmap.items():
        for s in nodes:
            for o in fn(s):
                facts.append((pred, (s.name(), o.name())))
    ents = {x for (_p, pair) in facts for x in pair}
    preds = {"isa": 2, "has_part": 2, "made_of": 2, "member_of": 2}
    rules = [(("isa", ("?x", "?z")), [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))])]
    for pred in relmap:                                       # cross-dimension rule: inherit down is-a
        rules.append(((pred, ("?x", "?p")), [("isa", ("?x", "?y")), (pred, ("?y", "?p"))]))
    import thinking.lota_kernel as LK
    kb = LK.KB(sorted(ents), preds, facts, rules)
    kb._combo_ents = sorted({lf.name() for lf in leaves})
    return kb


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
    # structured speaker+listener round-trip (the boundary-cracking architecture) wires up
    rk = rich_kb(seed=0); ca, _st = core_universe(rk)
    ss = SegmentedSpeaker(ca, 8, 32); sl = SegmentedListener(build_groups(ca), len(ca), ss.spg, 8, 32)
    xv = torch.from_numpy(np.stack([core_vec(set(rk.facts), ca, e) for e in rk._combo_ents[:4]]))
    assert sl(ss(xv)[0]).shape == (4, len(ca)), "segmented speaker+listener shapes must align"
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


class SlotSpeaker(nn.Module):
    """DISCOVERS the factorization (no predicate labels given): K latent slots; atoms COMPETE to be
    assigned to slots (softmax over slots, slot-attention style), so slots self-organize into factor
    groups. Each slot -> one symbol. The factor structure is learned, not handed in."""
    def __init__(self, n_atoms, K, V, dim=64, assign_tau=0.5):
        super().__init__()
        self.atom_emb = nn.Embedding(n_atoms, dim)
        self.slot_q = nn.Parameter(torch.randn(K, dim) * 0.2)
        self.head = nn.Linear(dim, V)
        self.K, self.V, self.L, self.assign_tau = K, V, K, assign_tau
        self.register_buffer("aidx", torch.arange(n_atoms))

    def assignment(self):
        E = self.atom_emb(self.aidx)
        return torch.softmax(torch.einsum("nd,kd->nk", E, self.slot_q) / self.assign_tau, dim=-1)

    def assign_entropy(self):
        a = self.assignment().clamp_min(1e-9)
        return (-(a * a.log()).sum(-1)).mean()                   # mean per-atom entropy (push -> hard)

    def forward(self, x, tau=1.0, hard=True):
        E = self.atom_emb(self.aidx)                              # (n, dim)
        assign = self.assignment()                               # (n,K) sharpened competition (fixed)
        w = x.unsqueeze(-1) * assign.unsqueeze(0)                 # (B,n,K) weighted by presence
        slots = torch.einsum("bnk,nd->bkd", w, E)                # (B,K,dim)
        logits = self.head(slots)                                # (B,K,V)
        if self.training:
            msg = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        else:
            msg = F.one_hot(logits.argmax(-1), self.V).float()
        return msg, None


class SlotListener(nn.Module):
    """Each slot-symbol asserts a subset of atoms; aggregate over slots (permutation-invariant). No
    predicate grouping given -- mirrors the discovered factorization."""
    def __init__(self, K, V, n_atoms, dim=64):
        super().__init__()
        self.sym = nn.Linear(V, dim, bias=False)
        self.dec = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, n_atoms))

    def forward(self, msg):
        return self.dec(self.sym(msg)).sum(1)                    # (B, n_atoms) aggregate slot assertions


def discover_run(steps, seed=0, K=16, V=16, dim=64, rich=True, verbose=True,
                 assign_tau=1.0, ent_lam=0.0, kb=None):
    """REALISM push: discover the factorization (slot attention) instead of being given predicates;
    listener predicts the core; the BRAIN derives the rest. To CLOSE THE DISCOVERY GAP, sharpen the
    (input-independent) atom->slot partition: low assignment temperature + an ENTROPY regularizer push
    each atom into ONE slot -> clean factor groups approaching the given-structure ceiling. Pass kb to
    ground in REAL data (WordNet)."""
    kb = kb if kb is not None else (rich_kb(seed=seed) if rich else P.build_kb())
    full_atoms = fact_universe(kb)
    core, static = core_universe(kb)
    ents = getattr(kb, "_combo_ents", None) or entities_with_facts(kb, full_atoms)
    tr, te = split_entities(ents, 0.25, seed)
    fs = set(kb.facts)
    spk = SlotSpeaker(len(core), K, V, dim, assign_tau=assign_tau)
    lis = SlotListener(K, V, len(core), dim)
    tr_vecs = [core_vec(fs, core, e) for e in tr]
    # custom loop: BCE reconstruction + entropy regularizer on the atom->slot partition
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.Adam(list(spk.parameters()) + list(lis.parameters()), lr=1e-3)
    spk.train(); lis.train()
    for step in range(steps):
        x = make_batch(rng, tr_vecs, 64)
        msg, _ = spk(x, tau=1.5, hard=True)
        loss = F.binary_cross_entropy_with_logits(lis(msg), x) + ent_lam * spk.assign_entropy()
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % max(1, steps // 6) == 0:
            print(f"  step {step:5d}  loss {loss.item():.3f}  assign-entropy "
                  f"{spk.assign_entropy().item():.3f}", flush=True)

    def ev(es):
        spk.eval(); lis.eval()
        x = torch.from_numpy(np.stack([core_vec(fs, core, e) for e in es]))
        with torch.no_grad():
            pred = (torch.sigmoid(lis(spk(x)[0])) > 0.5).numpy()
        f_ok = f_tot = r_ok = r_tot = exact = 0
        for r, e in enumerate(es):
            base_pred = [(p, (e, o)) for k, (p, o) in enumerate(core) if pred[r, k]]
            got = {(p, a[1]) for (p, a) in kb.dl.closure(static + base_pred)[0]
                   if len(a) == 2 and a[0] == e}
            true = {(p, o) for (p, o) in full_atoms if (p, (e, o)) in kb.known}
            for f in got:
                f_tot += 1; f_ok += int((f[0], (e, f[1])) in kb.known)
            r_ok += len(got & true); r_tot += len(true); exact += int(got == true)
        n = max(1, len(es))
        return dict(exact=exact / n, faith=f_ok / max(1, f_tot), recall=r_ok / max(1, r_tot))
    rtr, rte = ev(tr), ev(te)
    if verbose:
        print(f"\n== DISCOVERED factorization (slot attention) + brain closure ==")
        print(f"  world: RICH | core {len(core)} | full {len(full_atoms)} | train {len(tr)}/held {len(te)}"
              f" | {K} slots (factors NOT given)")
        print(f"  TRAIN    : full-exact {rtr['exact']:.3f} | faith {rtr['faith']:.3f} | recall {rtr['recall']:.3f}")
        print(f"  HELD-OUT : full-exact {rte['exact']:.3f} | faith {rte['faith']:.3f} | recall "
              f"{rte['recall']:.3f}   (never-seen combos; factorization self-discovered)")
    return dict(train=rtr, heldout=rte)


def build_groups(atoms):
    """Ordered list of core-atom index groups, one per typed predicate (the factor structure)."""
    groups = {}
    for i, (pred, _obj) in enumerate(atoms):
        groups.setdefault(pred, []).append(i)
    return [torch.tensor(v) for v in groups.values()]


def mandatory_groups(core, groups, ents, facts_set):
    """A relation-dimension is MANDATORY-FUNCTIONAL if every entity has EXACTLY ONE direct value in it
    (e.g. is-a: exactly one direct parent). Such a dimension must be decoded by ARGMAX (pick the single
    value), not by independent thresholding -- thresholding a near-one-hot target over-predicts, and the
    brain's closure then amplifies each false value into a whole false chain. Returns a bool per group."""
    out = []
    for g in groups:
        gi = g.tolist()
        counts = [sum(1 for k in gi if (core[k][0], (e, core[k][1])) in facts_set) for e in ents]
        out.append(bool(counts) and min(counts) == 1 and max(counts) == 1)
    return out


def structured_decode(prob, groups, mand):
    """Decode per relation-dimension: mandatory-functional groups by argmax (exactly one), the rest by
    threshold. prob: (B, n_core). Returns a boolean prediction matrix matched to the factor structure."""
    pred = np.zeros(prob.shape, dtype=bool)
    rows = np.arange(prob.shape[0])
    for g, is_mand in zip(groups, mand):
        gi = g.numpy()
        if is_mand:
            pred[rows, gi[prob[:, gi].argmax(1)]] = True
        else:
            pred[:, gi] = prob[:, gi] > 0.5
    return pred


def index_closure(closed_static):
    """Index the once-closed taxonomy for O(ancestors) per-entity derivation. Returns (anc_of, rel_of):
    anc_of[x] = all z with isa(x,z); rel_of[pred][x] = all o with pred(x,o) (pred != isa). Because the
    taxonomy is already a fixpoint, an entity's full fact-set composes from one is-a hop -- no re-running
    the datalog loop per entity (which re-scans the whole graph). Equivalent to closure, far faster."""
    anc_of, rel_of = {}, {}
    for (p, a) in closed_static:
        if len(a) != 2:
            continue
        x, o = a
        if p == "isa":
            anc_of.setdefault(x, set()).add(o)
        else:
            rel_of.setdefault(p, {}).setdefault(x, set()).add(o)
    return anc_of, rel_of


def derive_entity(e, base_pred, anc_of, rel_of):
    """The BRAIN's rules applied to a leaf's predicted core: is-a transitivity + each relation inherited
    down the is-a chain. base_pred = the predicted core atoms (pred,(e,o)). Returns the derived full
    fact-set {(pred, o)} for e -- identical to the datalog closure, computed by composition."""
    parents = {o for (p, (_, o)) in base_pred if p == "isa"}
    anc = set(parents)
    for par in parents:
        anc |= anc_of.get(par, set())                        # isa transitivity (parent already closed)
    got = {("isa", a) for a in anc}
    got |= {(p, o) for (p, (_, o)) in base_pred if p != "isa"}   # the leaf's DIRECT relations
    for pred, idx in rel_of.items():                         # inherit each relation down the is-a chain
        for y in anc:
            got |= {(pred, o) for o in idx.get(y, ())}
    return got


class SegmentedListener(nn.Module):
    """Mirrors the segmented speaker: each message SEGMENT decodes ONLY its own predicate's atoms ->
    fully disentangled decode (no GRU entanglement), so each factor is read independently and novel
    combinations compose. Scatters per-group logits back into the core-atom vector."""
    def __init__(self, group_idx, n_core, spg, V, d=128):
        super().__init__()
        self.group_idx, self.n_core, self.spg = group_idx, n_core, spg
        self.sym = nn.Linear(V, d, bias=False)
        self.dec = nn.ModuleList(
            nn.Sequential(nn.Linear(spg * d, d), nn.ReLU(), nn.Linear(d, len(idx)))
            for idx in group_idx)

    def forward(self, msg):
        B = msg.shape[0]
        e = self.sym(msg)
        out = torch.zeros(B, self.n_core)
        pos = 0
        for idx, dec in zip(self.group_idx, self.dec):
            out[:, idx] = dec(e[:, pos:pos + self.spg].reshape(B, -1))
            pos += self.spg
        return out


class SegmentedSpeaker(nn.Module):
    """Structured speaker: one message SEGMENT per typed predicate (isa/color/size/...), computed ONLY
    from that predicate's facts -> positionally DISENTANGLED by construction, so each factor's code is
    stable across the others and novel combinations COMPOSE. The agent still discovers the per-factor
    code; we only give the inductive bias that the world is factored into typed relations."""
    def __init__(self, atoms, V, d=128, syms_per_group=1):
        super().__init__()
        groups = {}
        for i, (pred, _obj) in enumerate(atoms):
            groups.setdefault(pred, []).append(i)
        self.group_idx = [torch.tensor(v) for v in groups.values()]
        self.V = V
        self.L = len(self.group_idx) * syms_per_group
        self.spg = syms_per_group
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(len(idx), d), nn.ReLU(), nn.Linear(d, syms_per_group * V))
            for idx in self.group_idx)

    def forward(self, x, tau=1.0, hard=True):
        segs = []
        for idx, head in zip(self.group_idx, self.heads):
            lg = head(x[:, idx]).view(-1, self.spg, self.V)
            if self.training:
                segs.append(F.gumbel_softmax(lg, tau=tau, hard=hard, dim=-1))
            else:
                segs.append(F.one_hot(lg.argmax(-1), self.V).float())
        msg = torch.cat(segs, dim=1)                          # (B, L, V)
        return msg, None


def arch_compare(steps, seed=0, V=16, d=128, rich=True, L_holistic=8):
    """Holistic speaker (shared bottleneck) vs SEGMENTED speaker (per-predicate, disentangled) on the
    rich world. Does structured architecture crack compositional generalization (the boundary)?"""
    out = {}
    kb, atoms, ents, tr, te, _, _ = _setup(L_holistic, V, d, seed, rich=rich)
    tr_vecs = [fact_vec(kb, atoms, e) for e in tr]
    print(f"  world: {'RICH' if rich else 'toy'} | {len(ents)} entities (train {len(tr)}/held {len(te)})"
          f" | fact-universe {len(atoms)}", flush=True)
    builders = {
        "holistic": lambda: (lambda s: (s, ReconListener(L_holistic, V, len(atoms), d)))(
            _mk_holistic(atoms, L_holistic, V, d)),
        "segmented": lambda: (lambda s: (s, ReconListener(s.L, V, len(atoms), d)))(
            SegmentedSpeaker(atoms, V, d)),
    }
    for tag, build in builders.items():
        spk, lis = build()
        train(spk, lis, tr_vecs, steps=steps, batch=64, seed=seed)
        r = evaluate(spk, lis, kb, atoms, te)
        out[tag] = dict(exact=r["exact"], faith=r["faithfulness"], recall=r["recall"],
                        topsim=topsim(spk, kb, atoms, ents), L=spk.L)
        print(f"  {tag:>9} (L={spk.L}): held-out exact {r['exact']:.3f} | faith {r['faithfulness']:.3f}"
              f" | recall {r['recall']:.3f} | topsim {out[tag]['topsim']:.3f}", flush=True)
    return out


def _mk_holistic(atoms, L, V, d):
    s = Speaker(len(atoms), 1, L, V, d=d)
    s.enc[0] = nn.Linear(len(atoms), d)
    return s


def core_universe(kb):
    """The IRREDUCIBLE base facts about the combo entities (directly asserted: leaf isa + attributes),
    NOT the derived isa-chain / inherited props -- those the BRAIN derives by closure."""
    ce = set(getattr(kb, "_combo_ents", []))
    base = set(kb.facts)
    core = sorted({(p, args[1]) for (p, args) in base if len(args) == 2 and args[0] in ce})
    static = [f for f in kb.facts if not (len(f[1]) == 2 and f[1][0] in ce)]   # taxonomy + cat-props
    return core, static


def core_vec(kb_facts, core, e):
    return np.array([1.0 if (p, (e, o)) in kb_facts else 0.0 for (p, o) in core], dtype=np.float32)


def brain_close_run(steps, seed=0, V=16, d=128, rich=True, verbose=True, kb=None):
    """Segmented speaker over the IRREDUCIBLE factors; listener predicts those; the BRAIN derives the
    rest by closure. Eval faithfulness/recall/exact on the FULL fact-set -- the derived part is exact
    when the core is right. This leverages the provable brain to crack the 'exact' boundary. Pass kb to
    ground in REAL multi-relation data (WordNet). V auto-grows to cover the largest factor's values, and
    each dimension is decoded by its arity (functional=argmax, multi-valued=threshold)."""
    kb = kb if kb is not None else (rich_kb(seed=seed) if rich else P.build_kb())
    full_atoms = fact_universe(kb)
    core, static = core_universe(kb)
    ents = getattr(kb, "_combo_ents", None) or entities_with_facts(kb, full_atoms)
    tr, te = split_entities(ents, 0.25, seed)
    facts_set = set(kb.facts)
    groups = build_groups(core)
    mand = mandatory_groups(core, groups, ents, facts_set)
    # a high-cardinality dimension (e.g. WordNet is-a: 139 parents) needs a MULTI-SYMBOL code, not one
    # huge categorical (untrainable under Gumbel). Size symbols/group so V**spg covers the widest factor.
    import math
    spg = max(1, math.ceil(math.log(max(2, max(len(g) for g in groups))) / math.log(V)))
    spk = SegmentedSpeaker(core, V, d, syms_per_group=spg)
    lis = SegmentedListener(groups, len(core), spk.spg, V, d)   # structured decode
    tr_vecs = [core_vec(facts_set, core, e) for e in tr]
    train(spk, lis, tr_vecs, steps=steps, batch=64, seed=seed,
          log=(max(1, steps // 6) if verbose else 0))
    # close the SHARED taxonomy ONCE (the brain), then derive each entity by composition over it --
    # closure is monotone+idempotent, so this equals closure(static + core_e) but is O(ancestors)/entity
    anc_of, rel_of = index_closure(kb.dl.closure(static)[0])
    # a held-out concept is COMMUNICABLE only if its core VALUES were seen in training (a fixed symbol
    # code cannot encode a value it never observed) -- so split held-out by that to separate the
    # compositional generalization from the concept-level open-vocab wall.
    train_vals = {(p, o) for e in tr for (p, o) in core if (p, (e, o)) in facts_set}

    def records(es):
        spk.eval(); lis.eval()
        x = torch.from_numpy(np.stack([core_vec(facts_set, core, e) for e in es]))
        with torch.no_grad():
            prob = torch.sigmoid(lis(spk(x)[0])).numpy()
        pred = structured_decode(prob, groups, mand)          # arity-aware: argmax vs threshold
        recs = []
        for r, e in enumerate(es):
            base_pred = [(p, (e, o)) for k, (p, o) in enumerate(core) if pred[r, k]]
            got = derive_entity(e, base_pred, anc_of, rel_of)        # BRAIN derives the rest
            true = {(p, o) for (p, o) in full_atoms if (p, (e, o)) in kb.known}
            tcore = {(p, o) for (p, o) in core if (p, (e, o)) in facts_set}
            recs.append(dict(
                exact=int(got == true), faith_ok=len(got & true), faith_tot=len(got),
                rec_ok=len(got & true), rec_tot=len(true),
                core_exact=int({core[k] for k in np.where(pred[r])[0]} == tcore),
                seen=(tcore <= train_vals)))
        return recs

    def agg(recs):
        n = max(1, len(recs))
        return dict(n=len(recs), exact=sum(r["exact"] for r in recs) / n,
                    faith=sum(r["faith_ok"] for r in recs) / max(1, sum(r["faith_tot"] for r in recs)),
                    recall=sum(r["rec_ok"] for r in recs) / max(1, sum(r["rec_tot"] for r in recs)),
                    core_exact=sum(r["core_exact"] for r in recs) / n)
    rtr, te_recs = agg(records(tr)), records(te)
    rte = agg(te_recs)
    rte_seen = agg([r for r in te_recs if r["seen"]])
    rte_unseen = agg([r for r in te_recs if not r["seen"]])
    if verbose:
        print(f"\n== SEGMENTED speaker + BRAIN CLOSURE (message conveys core factors; brain derives rest) ==")
        dims = ", ".join(f"{p}{'*' if m else ''}" for p, m in zip(
            dict.fromkeys(p for p, _ in core), mand))
        is_wn = bool(kb._combo_ents) and "." in kb._combo_ents[0]
        print(f"  world: {'WordNet' if is_wn else 'RICH'} | "
              f"dimensions [{dims}] (*=functional->argmax) | core factors {len(core)} | "
              f"full atoms {len(full_atoms)} | train {len(tr)}/held {len(te)} | msg L={spk.L} V={V}")
        print(f"  TRAIN    : full-exact {rtr['exact']:.3f} | faith {rtr['faith']:.3f} | recall "
              f"{rtr['recall']:.3f} | core-exact {rtr['core_exact']:.3f}")
        print(f"  HELD-OUT : full-exact {rte['exact']:.3f} | faith {rte['faith']:.3f} | recall "
              f"{rte['recall']:.3f} | core-exact {rte['core_exact']:.3f}   (never-seen concepts)")
        if rte_seen["n"] and rte_unseen["n"]:
            print(f"    +-- core values SEEN in train (n={rte_seen['n']:3d}): exact {rte_seen['exact']:.3f}"
                  f" | faith {rte_seen['faith']:.3f} | recall {rte_seen['recall']:.3f}   <- GENERALIZES")
            print(f"    +-- core value UNSEEN (n={rte_unseen['n']:3d}): exact {rte_unseen['exact']:.3f}"
                  f" | faith {rte_unseen['faith']:.3f} | recall {rte_unseen['recall']:.3f}   <- open-vocab wall")
        print("  the DERIVED facts are exact when the core is right -- the brain computes them, proved.")
    return dict(train=rtr, heldout=rte, heldout_seen=rte_seen, heldout_unseen=rte_unseen)


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
    ap.add_argument("--arch", action="store_true",
                    help="compare holistic vs segmented (disentangled) speaker architecture")
    ap.add_argument("--brain-close", action="store_true",
                    help="segmented speaker over core factors + brain derives the rest by closure")
    ap.add_argument("--discover", action="store_true",
                    help="DISCOVER the factorization (slot attention) -- predicates NOT given")
    ap.add_argument("--wordnet", action="store_true",
                    help="REAL grounded data: WordNet hypernym chains (brain closes is-a)")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--assign-tau", type=float, default=1.0)
    ap.add_argument("--ent-lam", type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    if args.arch:
        arch_compare(args.steps, seed=args.seed, V=args.vocab, d=args.d, rich=args.rich,
                     L_holistic=args.msg_len); return 0
    if args.brain_close:
        brain_close_run(args.steps, seed=args.seed, V=args.vocab, d=args.d, rich=True); return 0
    if args.wordnet:
        kb = wordnet_kb(seed=args.seed)
        print(f"  REAL grounded data: WordNet | {len(kb._combo_ents)} real concepts | "
              f"{len(kb.entities)} synsets | multi-relation closure derived by the brain", flush=True)
        if args.discover:                                      # self-discover the dimensions (harder)
            discover_run(args.steps, seed=args.seed, K=args.slots, V=max(32, args.vocab), kb=kb,
                         assign_tau=args.assign_tau, ent_lam=args.ent_lam)
        else:                                                  # typed dimensions + arity-aware decode
            brain_close_run(args.steps, seed=args.seed, V=args.vocab, d=args.d, kb=kb)
        return 0
    if args.discover:
        discover_run(args.steps, seed=args.seed, K=args.slots, V=args.vocab, rich=True,
                     assign_tau=args.assign_tau, ent_lam=args.ent_lam); return 0
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
