#!/usr/bin/env python3
"""define -- learn MEANING by READING DEFINITIONS (the path THROUGH the concept-level open-vocab wall).

The relational emergent code ([[emergent-language]], emergent_grounded.py) conveys a concept by a SYMBOL
for its identity, so a NEVER-SEEN concept with a near-unique parent is unguessable (held-out exact 0.000
on unseen-parent concepts -- a fixed symbol code cannot encode a value it never observed).

WordNet gives a far richer signal: every concept has a natural-language DEFINITION (gloss), built from
a SHARED vocabulary. `brabancon_griffon` is unguessable as a symbol, but its gloss -- "a variety of
Brussels griffon having a short smooth coat" -- is ordinary words. So we PAIR each concept's definition
with its brain-verified relational fact-set and learn  definition-text -> meaning:

  * a CHAR-level encoder reads the gloss (char-level => novel definition words still encode -> no UNK,
    open-vocab-proof on the input side);
  * it predicts the concept's relational CORE (direct parent + direct parts/members), decoded ARITY-aware
    (functional dims by argmax, multi-valued by threshold);
  * the BRAIN derives the rest by closure and VERIFIES every conveyed fact (kernel/datalog ground truth).

Eval on NEVER-SEEN concepts, split by whether the concept's parent was seen: the symbol code scored
exact 0.000 on unseen-parent concepts; reading the definition should cross that wall. A NAME-only
baseline (encode the concept's name/synonyms, not its gloss) shows the understanding comes from the
DEFINITION, not the surface name.

  python -m thinking.define --selftest
  python -m thinking.define --steps 4000 --input def
  python -m thinking.define --steps 4000 --input name     # baseline: name string only
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thinking.emergent_grounded as EG  # noqa: E402


# ===================================================================================================
# Data: concept -> (definition text, synonyms) + brain-verified relational core
# ===================================================================================================
def concept_text(name, mode="def"):
    """The natural-language signal for a concept. mode: 'def' = gloss (the meaning), 'name' = synonyms
    only (baseline -- the surface name, deliberately NOT the meaning), 'both' = synonyms + gloss."""
    from nltk.corpus import wordnet as wn
    s = wn.synset(name)
    syn = " ".join(w.replace("_", " ") for w in s.lemma_names())
    gloss = s.definition()
    return {"def": gloss, "name": syn, "both": syn + " : " + gloss}[mode]


class CharVocab:
    """Char-level vocab built from TRAINING texts only (+ <pad>=0, <unk>=1). Char-level is the point:
    a never-seen definition WORD is still made of seen chars, so novel concepts still encode."""
    def __init__(self, texts):
        chars = sorted({c for t in texts for c in t.lower()})
        self.itos = ["<pad>", "<unk>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, t, max_len=200):
        ids = [self.stoi.get(c, 1) for c in t.lower()[:max_len]]
        return ids or [1]

    def __len__(self):
        return len(self.itos)


def batch_pad(seqs):
    m = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), m, dtype=torch.long)
    lens = torch.tensor([len(s) for s in seqs])
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x, lens


# ===================================================================================================
# Model: char encoder -> relational core logits
# ===================================================================================================
class CharMeaning(nn.Module):
    """Read the definition char-by-char (bi-GRU) -> a meaning vector -> predict the relational core.
    Dropout (embedding + feature) fights the tiny-data memorization that makes a high-capacity encoder
    overfit each gloss instead of learning the shared-word -> meaning RULE."""
    def __init__(self, n_chars, n_core, d=128, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_chars, d, padding_idx=0)
        self.edrop = nn.Dropout(drop)
        self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * d, d), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(d, n_core))

    def forward(self, ids, lens):
        e = self.edrop(self.emb(ids))
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)                              # h: (2, B, d)
        return self.head(torch.cat([h[0], h[1]], dim=-1))   # (B, n_core)


# ===================================================================================================
# Train + brain-verified eval
# ===================================================================================================
def train(model, X, lens, Y, steps, batch=64, lr=1e-3, seed=0, log=0, wd=1e-4, device="cpu"):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for step in range(steps):
        idx = rng.integers(0, len(Y), size=min(batch, len(Y)))
        bx, bl = batch_pad([X[i] for i in idx])
        logits = model(bx.to(device), bl)
        loss = F.binary_cross_entropy_with_logits(logits, Y[idx].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        if log and (step % log == 0 or step == steps - 1):
            with torch.no_grad():
                acc = ((torch.sigmoid(logits) > 0.5) == (Y[idx].to(device) > 0.5)).float().mean().item()
            print(f"  step {step:5d}  loss {loss.item():.3f}  bit-acc {acc:.3f}", flush=True)


def run(steps=4000, seed=0, mode="def", d=128, per_cat=25, verbose=True, device="cpu",
        batch=64, out=None, cap=400):
    kb = EG.wordnet_kb(seed=seed, per_cat=per_cat, cap=cap)
    core, static = EG.core_universe(kb)
    ents = kb._combo_ents
    tr, te = EG.split_entities(ents, 0.25, seed)
    facts_set = set(kb.facts)
    groups = EG.build_groups(core)
    mand = EG.mandatory_groups(core, groups, ents, facts_set)
    anc_of, rel_of = EG.index_closure(kb.dl.closure(static)[0])
    train_vals = {(p, o) for e in tr for (p, o) in core if (p, (e, o)) in facts_set}

    texts = {e: concept_text(e, mode) for e in ents}
    vocab = CharVocab([texts[e] for e in tr])               # vocab from TRAIN texts only
    X = {e: vocab.encode(texts[e]) for e in ents}
    Ytr = torch.from_numpy(np.stack([EG.core_vec(facts_set, core, e) for e in tr]))
    model = CharMeaning(len(vocab), len(core), d=d).to(device)
    train(model, [X[e] for e in tr], None, Ytr, steps, batch=batch, seed=seed, device=device,
          log=(max(1, steps // 6) if verbose else 0))

    def records(es):
        model.eval()
        recs_prob = []
        for i in range(0, len(es), 256):                    # chunk so big held-out sets fit
            bx, bl = batch_pad([X[e] for e in es[i:i + 256]])
            with torch.no_grad():
                recs_prob.append(torch.sigmoid(model(bx.to(device), bl)).cpu().numpy())
        prob = np.concatenate(recs_prob, axis=0)
        pred = EG.structured_decode(prob, groups, mand)
        recs = []
        for r, e in enumerate(es):
            base_pred = [(p, (e, o)) for k, (p, o) in enumerate(core) if pred[r, k]]
            got = EG.derive_entity(e, base_pred, anc_of, rel_of)
            true = EG.true_factset(e, core, facts_set, anc_of, rel_of)   # brain-derived ground truth
            tcore = {(p, o) for (p, o) in core if (p, (e, o)) in facts_set}
            recs.append(dict(exact=int(got == true), faith_ok=len(got & true), faith_tot=len(got),
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

    rtr = agg(records(tr))
    te_recs = records(te)
    rte = agg(te_recs)
    seen = agg([r for r in te_recs if r["seen"]])
    unseen = agg([r for r in te_recs if not r["seen"]])
    if verbose:
        print(f"\n== MEANING from {'DEFINITION' if mode != 'name' else 'NAME (baseline)'} "
              f"(read text -> relational core; brain derives + verifies) ==")
        print(f"  WordNet | {len(ents)} concepts | core {len(core)} | "
              f"train {len(tr)}/held {len(te)} | char-vocab {len(vocab)} | input='{mode}'")
        print(f"  TRAIN    : exact {rtr['exact']:.3f} | faith {rtr['faith']:.3f} | recall "
              f"{rtr['recall']:.3f} | core-exact {rtr['core_exact']:.3f}")
        print(f"  HELD-OUT : exact {rte['exact']:.3f} | faith {rte['faith']:.3f} | recall "
              f"{rte['recall']:.3f} | core-exact {rte['core_exact']:.3f}   (never-seen concepts)")
        if seen["n"] and unseen["n"]:
            print(f"    +-- parent SEEN   (n={seen['n']:3d}): exact {seen['exact']:.3f} | "
                  f"faith {seen['faith']:.3f} | recall {seen['recall']:.3f}")
            print(f"    +-- parent UNSEEN (n={unseen['n']:3d}): exact {unseen['exact']:.3f} | "
                  f"faith {unseen['faith']:.3f} | recall {unseen['recall']:.3f}   <- the open-vocab wall")
        print("  faith is GROUND TRUTH: every conveyed fact checked against the brain's closure.")
    result = dict(mode=mode, n_concepts=len(ents), n_core=len(core), per_cat=per_cat, steps=steps,
                  train=rtr, heldout=rte, heldout_seen=seen, heldout_unseen=unseen)
    if out:
        import json
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"  wrote {out}", flush=True)
    return result


def selftest():
    torch.set_num_threads(2)
    v = CharVocab(["a dog", "a cat"])
    assert v.encode("a dog") and v.stoi.get("z", 1) == 1   # unseen char -> <unk>
    r = run(steps=400, seed=0, mode="def", d=64, verbose=False)
    assert r["train"]["recall"] > 0.2, ("should read SOME meaning from definitions on train", r)
    print("define selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", choices=["def", "name", "both"], default="def")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--per-cat", type=int, default=25, help="concepts sampled per seed category (scale)")
    ap.add_argument("--cap", type=int, default=400, help="BFS cap per category (raise to scale concepts)")
    ap.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    device = args.device
    if device is None:
        try:
            from device import get_device
            device = get_device()
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    run(args.steps, seed=args.seed, mode=args.input, d=args.d, per_cat=args.per_cat,
        device=device, batch=args.batch, out=args.out, cap=args.cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
