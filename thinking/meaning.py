#!/usr/bin/env python3
"""meaning -- capture MEANING by CONTRASTIVE pair-training over ALL of WordNet's signals (all POS),
plus a BRAIN-VERIFIED relational probe (the locked direction: "both -- contrastive + verified probe").

WordNet gives many SIGNALS per concept -- word class (POS), names/synonyms, definition (gloss), example
sentences, antonyms, and relations (is-a, ...). Each is a different VIEW of the same meaning. We:

  1. CONTRASTIVE pair-training (capture meaning): a shared char-level encoder embeds every text view
     (definition / synonyms / example). Views of the SAME concept are pulled together and different
     concepts pushed apart (InfoNCE, in-batch negatives -- CLIP-style). Meaning is captured from all
     angles at once, across nouns + verbs + adjectives + adverbs.
  2. VERIFIED relational probe (understanding = PROVABLE relations): a POINTER head scores each candidate
     is-a parent by how well its NAME matches the concept's meaning embedding (query=concept, key=parent
     name). The predicted parent is closed by the BRAIN (is-a transitivity) and every conveyed fact is
     verified. Because a parent is identified by its NAME (not a fixed output slot), a NEVER-SEEN parent
     is still scorable if its name-words were seen -- crossing the open-vocab wall a fixed head can't.
  3. a POS probe (word classification) -- a cheap checkable auxiliary.

Eval on NEVER-SEEN concepts: zero-shot retrieval (does a held-out definition retrieve its own
synonyms?), brain-verified is-a faith/recall (split by seen/unseen parent), POS accuracy.

  python -m thinking.meaning --selftest
  python -m thinking.meaning --steps 4000 --per-pos 150
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================================================
# Data: sample concepts across ALL POS; gather every signal/view
# ===================================================================================================
def gather(per_pos=150, seed=0, max_anc=64):
    from nltk.corpus import wordnet as wn
    rng = np.random.default_rng(seed)
    syns = list(wn.all_synsets())
    rng.shuffle(syns)
    picked = {"n": [], "v": [], "a": [], "r": []}
    for s in syns:
        p = "a" if s.pos() in ("a", "s") else s.pos()
        if p in picked and len(picked[p]) < per_pos:
            picked[p].append(s)
    chosen = [s for v in picked.values() for s in v]

    parent = {}                                              # child name -> direct is-a parent name
    node_text = {}                                            # any synset name -> its MEANING text
    def meaning_text(x):
        syn = ", ".join(w.replace("_", " ") for w in x.lemma_names())
        return f"means {x.definition()} also called {syn}"
    for s in chosen:
        paths = s.hypernym_paths()
        if paths:
            objs = paths[0]
            for x in objs:                                    # cache the MEANING of every node on path
                node_text.setdefault(x.name(), meaning_text(x))
            ns = [x.name() for x in objs]
            for ch, pa in zip(ns[1:], ns):                   # ns is root..leaf; (child, parent)
                parent[ch] = pa

    def ancestors(name):
        out, cur, seen = [], name, set()
        while cur in parent and cur not in seen and len(out) < max_anc:
            seen.add(cur); cur = parent[cur]; out.append(cur)
        return out

    concepts = []
    for s in chosen:
        pos = "a" if s.pos() in ("a", "s") else s.pos()
        syn = ", ".join(w.replace("_", " ") for w in s.lemma_names())
        views = [f"means {s.definition()}"]
        if len(s.lemma_names()) > 0:
            views.append(f"also called {syn}")
        if s.examples():
            views.append(f"as in {s.examples()[0]}")
        ants = sorted({a.name() for l in s.lemmas() for a in l.antonyms()})
        node_text.setdefault(s.name(), views[0])
        concepts.append(dict(
            name=s.name(), pos=pos, syn=syn, views=views,
            parent=parent.get(s.name()), ancestors=ancestors(s.name()), antonyms=ants))
    return concepts, parent, node_text


def split(concepts, held_frac=0.25, seed=0):
    idx = np.arange(len(concepts)); np.random.default_rng(seed).shuffle(idx)
    cut = int(len(concepts) * (1 - held_frac))
    return [concepts[i] for i in idx[:cut]], [concepts[i] for i in idx[cut:]]


def parent_name_text(name):
    return name.split(".")[0].replace("_", " ")             # synset id -> readable lemma for matching


# ===================================================================================================
# Shared char-level text encoder -> normalized meaning embedding
# ===================================================================================================
class CharVocab:
    def __init__(self, texts):
        chars = sorted({c for t in texts for c in t.lower()})
        self.itos = ["<pad>", "<unk>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, t, max_len=160):
        return [self.stoi.get(c, 1) for c in t.lower()[:max_len]] or [1]

    def __len__(self):
        return len(self.itos)


def pad(seqs):
    m = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), m, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x, torch.tensor([len(s) for s in seqs])


class Encoder(nn.Module):
    def __init__(self, n_chars, d=256, drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_chars, d, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d, d)

    def forward(self, ids, lens):
        e = self.drop(self.emb(ids))
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)
        z = self.proj(torch.cat([h[0], h[1]], dim=-1))
        return F.normalize(z, dim=-1)                        # unit embedding (cosine space)


# ===================================================================================================
# Joint training: contrastive (all POS) + is-a pointer probe (n/v) + POS probe
# ===================================================================================================
POS_IDS = {"n": 0, "v": 1, "a": 2, "r": 3}


def encode_texts(enc, vocab, texts, device):
    x, l = pad([vocab.encode(t) for t in texts])
    return enc(x.to(device), l)


def gloss_words(text, maxw=24):
    return [w for w in text.lower().replace("means ", "").replace("also called ", "").split()][:maxw] \
        or ["?"]


def encode_words(enc, vocab, word_lists, device, maxw=24):
    """Encode each concept's gloss WORDS individually (char-encoded -> open-vocab-safe). Returns
    (B, maxw, d) normalized word embeddings + a (B, maxw) validity mask."""
    flat, lens = [], []
    for ws in word_lists:
        ws = ws[:maxw]; lens.append(len(ws)); flat.extend(ws)
    emb = encode_texts(enc, vocab, flat, device)             # (total_words, d)
    out = torch.zeros(len(word_lists), maxw, emb.shape[-1], device=device)
    mask = torch.zeros(len(word_lists), maxw, dtype=torch.bool, device=device)
    i = 0
    for b, n in enumerate(lens):
        out[b, :n] = emb[i:i + n]; mask[b, :n] = True; i += n
    return F.normalize(out, dim=-1), mask


def parent_keys(enc, p_head, vocab, node_text, pnames, device, isa_mode):
    """Key embeddings for candidate parents. 'meaning' -> parent gloss via p_head; else parent NAME."""
    if isa_mode == "meaning":
        txt = [node_text.get(p, parent_name_text(p)) for p in pnames]
        return F.normalize(p_head(encode_texts(enc, vocab, txt, device)), dim=-1)
    return F.normalize(encode_texts(enc, vocab, [parent_name_text(p) for p in pnames], device), dim=-1)


def concept_scores(enc, c_head, vocab, rows, keys, device, isa_mode, maxw=24):
    """(len(rows), len(keys)) is-a scores. 'copy' (attention/copy): a parent scores as the BEST-matching
    gloss WORD -> exploits 'the parent word appears in the gloss', which pooling+argmax cannot. 'name'/
    'meaning': pooled concept embedding (optionally projected) dotted with the parent keys."""
    if isa_mode == "copy":
        we, mask = encode_words(enc, vocab, [gloss_words(c["views"][0], maxw) for c in rows], device, maxw)
        s = torch.einsum("bwd,pd->bwp", we, keys)           # (B, W, P)
        return s.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values    # best gloss word per parent
    q = encode_texts(enc, vocab, [c["views"][0] for c in rows], device)
    if isa_mode == "meaning":
        q = F.normalize(c_head(q), dim=-1)
    return q @ keys.t()


def train(enc, pos_head, c_head, p_head, vocab, node_text, tr, steps, device, batch=128, lr=1e-3,
          temp=0.07, lam_isa=1.0, lam_pos=0.3, seed=0, log=0, isa_mode="name"):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    params = (list(enc.parameters()) + list(pos_head.parameters())
              + list(c_head.parameters()) + list(p_head.parameters()))
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    multi = [c for c in tr if len(c["views"]) >= 2]
    enc.train(); pos_head.train(); c_head.train(); p_head.train()
    for step in range(steps):
        bc = [multi[i] for i in rng.integers(0, len(multi), size=min(batch, len(multi)))]
        a = encode_texts(enc, vocab, [c["views"][0] for c in bc], device)            # definition view
        b = encode_texts(enc, vocab, [c["views"][rng.integers(1, len(c["views"]))] for c in bc],
                         device)                                                      # another view
        logits = a @ b.t() / temp                            # InfoNCE (CLIP-style, in-batch negatives)
        labels = torch.arange(len(bc), device=device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        # is-a probe (mode: name | meaning | copy), in-batch parents as candidates
        nv = [c for c in bc if c["parent"]]
        if nv:
            keys = parent_keys(enc, p_head, vocab, node_text, [c["parent"] for c in nv], device, isa_mode)
            isa_logits = concept_scores(enc, c_head, vocab, nv, keys, device, isa_mode) / temp
            lbl = torch.arange(len(nv), device=device)
            loss = loss + lam_isa * 0.5 * (F.cross_entropy(isa_logits, lbl)
                                           + F.cross_entropy(isa_logits.t(), lbl))
        # POS probe (word classification)
        pos_lbl = torch.tensor([POS_IDS[c["pos"]] for c in bc], device=device)
        loss = loss + lam_pos * F.cross_entropy(pos_head(a), pos_lbl)
        opt.zero_grad(); loss.backward(); opt.step()
        if log and (step % log == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {loss.item():.3f}", flush=True)


# ===================================================================================================
# Eval: zero-shot retrieval + brain-verified is-a (seen/unseen parent) + POS accuracy
# ===================================================================================================
@torch.no_grad()
def evaluate(enc, pos_head, c_head, p_head, vocab, node_text, tr, te, parent, device, isa_mode="name"):
    enc.eval(); pos_head.eval(); c_head.eval(); p_head.eval()

    # --- (1) zero-shot meaning retrieval: held-out definition -> its OWN synonyms among all held syn ---
    held = [c for c in te if len(c["views"]) >= 2 and c["syn"]]
    defs = encode_texts(enc, vocab, [c["views"][0] for c in held], device)
    syns = encode_texts(enc, vocab, [f"also called {c['syn']}" for c in held], device)
    sim = defs @ syns.t()
    rank1 = (sim.argmax(1) == torch.arange(len(held), device=device)).float().mean().item()
    ranks = (sim >= sim.gather(1, torch.arange(len(held), device=device)[:, None]).expand_as(sim)
             ).sum(1).float()
    mrr = (1.0 / ranks).mean().item()

    # --- (2) brain-verified is-a via the POINTER over candidate parent NAMES (incl. held parents,
    #         so unseen parents are scorable by name -> the open-vocab-wall test) ---
    def anc_set(name):
        out, cur, seen = set(), name, set()
        while cur in parent and cur not in seen:
            seen.add(cur); cur = parent[cur]; out.add(cur)
        return out
    cand = sorted({c["parent"] for c in tr + te if c["parent"]})
    cand_emb = parent_keys(enc, p_head, vocab, node_text, cand, device, isa_mode)
    train_parents = {c["parent"] for c in tr if c["parent"]}

    def isa_eval(group):
        rows = [c for c in group if c["parent"]]
        if not rows:
            return None
        pick = concept_scores(enc, c_head, vocab, rows, cand_emb, device, isa_mode).argmax(1).tolist()
        recs = []
        for c, pi in zip(rows, pick):
            pred = cand[pi]
            got = {pred} | anc_set(pred)
            true = {c["parent"]} | set(c["ancestors"])
            recs.append(dict(parent_ok=int(pred == c["parent"]),
                             faith=len(got & true) / max(1, len(got)),
                             recall=len(got & true) / max(1, len(true)),
                             seen=c["parent"] in train_parents))
        return recs

    te_isa = isa_eval(te) or []
    seen = [r for r in te_isa if r["seen"]]
    unseen = [r for r in te_isa if not r["seen"]]
    agg = lambda rs, k: (sum(r[k] for r in rs) / len(rs)) if rs else 0.0

    # --- (3) POS classification accuracy on held-out ---
    pos_acc = 0.0
    if te:
        emb = encode_texts(enc, vocab, [c["views"][0] for c in te], device)
        pred = pos_head(emb).argmax(1).cpu().numpy()
        gold = np.array([POS_IDS[c["pos"]] for c in te])
        pos_acc = float((pred == gold).mean())

    return dict(
        retr_at1=rank1, retr_mrr=mrr, n_retr=len(held),
        isa_parent=agg(te_isa, "parent_ok"), isa_faith=agg(te_isa, "faith"),
        isa_recall=agg(te_isa, "recall"), n_isa=len(te_isa),
        isa_seen=dict(n=len(seen), parent=agg(seen, "parent_ok"), faith=agg(seen, "faith"),
                      recall=agg(seen, "recall")),
        isa_unseen=dict(n=len(unseen), parent=agg(unseen, "parent_ok"), faith=agg(unseen, "faith"),
                        recall=agg(unseen, "recall")),
        pos_acc=pos_acc)


def run(steps=4000, seed=0, per_pos=150, d=256, device="cpu", batch=128, verbose=True, out=None,
        isa_mode="name"):
    concepts, parent, node_text = gather(per_pos=per_pos, seed=seed)
    tr, te = split(concepts, 0.25, seed)
    vocab = CharVocab([v for c in tr for v in c["views"]]
                      + [node_text.get(c["parent"], "") for c in tr if c["parent"]])
    enc = Encoder(len(vocab), d=d).to(device)
    pos_head = nn.Linear(d, 4).to(device)
    mlp = lambda: nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)).to(device)
    c_head, p_head = mlp(), mlp()                            # is-a projection heads (learned hypernymy)
    if verbose:
        from collections import Counter
        pc = Counter(c["pos"] for c in concepts)
        print(f"  WordNet ALL-POS | {len(concepts)} concepts {dict(pc)} | train {len(tr)}/held {len(te)}"
              f" | char-vocab {len(vocab)} | views=def/syn/example | device {device}", flush=True)
    train(enc, pos_head, c_head, p_head, vocab, node_text, tr, steps, device, batch=batch, seed=seed,
          isa_mode=isa_mode, log=(max(1, steps // 6) if verbose else 0))
    r = evaluate(enc, pos_head, c_head, p_head, vocab, node_text, tr, te, parent, device, isa_mode)
    if verbose:
        print("\n== MEANING: contrastive multi-view embedding + brain-verified relational probe ==")
        print(f"  CONTRASTIVE retrieval (held-out def -> its own synonyms, {r['n_retr']} concepts): "
              f"recall@1 {r['retr_at1']:.3f} | MRR {r['retr_mrr']:.3f}   (chance {1.0/max(1,r['n_retr']):.3f})")
        print(f"  POS classification (word class) held-out accuracy: {r['pos_acc']:.3f}  (chance 0.25)")
        print(f"  BRAIN-VERIFIED is-a pointer ({r['n_isa']} n/v held-out): parent {r['isa_parent']:.3f} "
              f"| faith {r['isa_faith']:.3f} | recall {r['isa_recall']:.3f}")
        s, u = r["isa_seen"], r["isa_unseen"]
        if s["n"]:
            print(f"    +-- parent SEEN   (n={s['n']:3d}): parent {s['parent']:.3f} | faith {s['faith']:.3f}"
                  f" | recall {s['recall']:.3f}")
        if u["n"]:
            print(f"    +-- parent UNSEEN (n={u['n']:3d}): parent {u['parent']:.3f} | faith {u['faith']:.3f}"
                  f" | recall {u['recall']:.3f}   <- name-pointer vs the open-vocab wall")
        print("  is-a facts brain-verified by closure; retrieval/POS measure captured meaning.")
    if out:
        import json
        with open(out, "w") as f:
            json.dump(dict(per_pos=per_pos, steps=steps, n=len(concepts), **r), f, indent=2)
        if verbose:
            print(f"  wrote {out}", flush=True)
    return r


def selftest():
    torch.set_num_threads(2)
    r = run(steps=300, seed=0, per_pos=40, d=64, device="cpu", verbose=False)
    assert r["retr_at1"] > 1.5 / max(1, r["n_retr"]), ("retrieval should beat chance", r)
    print("meaning selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-pos", type=int, default=150)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--isa-mode", choices=["name", "meaning", "copy"], default="name",
                    help="is-a probe: 'name' (pooled gloss vs parent name), 'meaning' (parent gloss), "
                         "'copy' (attention/copy -- best-matching gloss WORD vs parent name)")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args.steps, seed=args.seed, per_pos=args.per_pos, d=args.d, device=device,
        batch=args.batch, out=args.out, isa_mode=args.isa_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
