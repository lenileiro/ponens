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

    # MORE LANGUAGE DIMENSIONS -> relations the brain can reason over (collected for every node on the
    # is-a paths, so inheritable relations attach at ancestors too). Synset-level + lemma-level edges.
    RELS = {"has_part": lambda s: s.part_meronyms(), "made_of": lambda s: s.substance_meronyms(),
            "member_of": lambda s: s.member_holonyms(), "entails": lambda s: s.entailments(),
            "causes": lambda s: s.causes(), "similar": lambda s: s.similar_tos()}
    relations = {k: set() for k in RELS}
    relations["antonym"] = set(); relations["derived"] = set()
    seen_nodes = set()

    def collect(x):
        if x.name() in seen_nodes:
            return
        seen_nodes.add(x.name())
        for k, fn in RELS.items():
            for o in fn(x):
                relations[k].add((x.name(), o.name()))
        for lem in x.lemmas():
            for a in lem.antonyms():
                relations["antonym"].add((x.name(), a.synset().name()))
            for dform in lem.derivationally_related_forms():
                relations["derived"].add((x.name(), dform.synset().name()))

    concepts = []
    for s in chosen:
        pos = "a" if s.pos() in ("a", "s") else s.pos()
        syn = ", ".join(w.replace("_", " ") for w in s.lemma_names())
        views = [f"means {s.definition()}"]
        if len(s.lemma_names()) > 0:
            views.append(f"also called {syn}")
        if s.examples():
            views.append(f"as in {s.examples()[0]}")
        ants = sorted({a.name() for lem in s.lemmas() for a in lem.antonyms()})
        node_text.setdefault(s.name(), views[0])
        collect(s)                                           # relations on the concept (incl. adj/adv)
        for x in (s.hypernym_paths()[0] if s.hypernym_paths() else []):
            collect(x)                                       # + on each is-a ancestor (for inheritance)
        concepts.append(dict(
            name=s.name(), pos=pos, syn=syn, views=views,
            parent=parent.get(s.name()), ancestors=ancestors(s.name()), antonyms=ants))
    return concepts, parent, node_text, relations


def split(concepts, held_frac=0.25, seed=0):
    idx = np.arange(len(concepts)); np.random.default_rng(seed).shuffle(idx)
    cut = int(len(concepts) * (1 - held_frac))
    return [concepts[i] for i in idx[:cut]], [concepts[i] for i in idx[cut:]]


def build_brain(concepts, relations=None):
    """The provable BRAIN (THE purpose of the project): datalog closure + KERNEL proofs + BDD types.
    Many LANGUAGE DIMENSIONS map to RELATIONS + RULES the brain reasons over:
      is-a         (hypernym)            -- TRANSITIVE
      has_part/made_of/member_of         -- INHERITED DOWN is-a (a bird has wings => a robin has wings)
      entails      (verb entailment)     -- TRANSITIVE (walk => move => ...)
      causes       (verb causation)      -- direct
      similar / antonym / derived        -- SYMMETRIC
    Meaning counts only when the brain returns a KERNEL-CHECKED proof term; negations ("X is NOT a Y")
    are proved from category disjointness via exclusion axioms. The model PROPOSES, the brain PROVES."""
    import thinking.lota_kernel as LK
    facts = []
    for c in concepts:
        chain = [c["name"]] + c["ancestors"]                 # name -> parent -> ... -> root
        for ch, pa in zip(chain, chain[1:]):
            facts.append(("isa", (ch, pa)))
    preds = {"isa": 2}
    rules = [(("isa", ("?x", "?z")), [("isa", ("?x", "?y")), ("isa", ("?y", "?z"))])]   # transitivity
    for pred, edges in (relations or {}).items():
        if not edges:
            continue
        preds[pred] = 2
        facts += [(pred, e) for e in edges]
        if pred in ("has_part", "made_of", "member_of"):     # inherited down the is-a chain
            rules.append(((pred, ("?x", "?p")), [("isa", ("?x", "?y")), (pred, ("?y", "?p"))]))
        elif pred == "entails":                              # transitive verb implication
            rules.append((("entails", ("?x", "?z")), [("entails", ("?x", "?y")), ("entails", ("?y", "?z"))]))
        elif pred in ("similar", "antonym", "derived"):      # symmetric
            rules.append(((pred, ("?x", "?y")), [(pred, ("?y", "?x"))]))
    ents = sorted({x for (_p, e) in facts for x in e})
    return LK.KB(ents, preds, facts, rules, build_types=True, eager_closure=True)


def brain_ancestor_index(brain):
    """node -> set of is-a ancestors, read off the brain's verified closure (kb.known)."""
    idx = {}
    for (p, a) in brain.known:
        if p == "isa" and len(a) == 2:
            idx.setdefault(a[0], set()).add(a[1])
    return idx


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

    @classmethod
    def from_itos(cls, itos):
        v = cls.__new__(cls)
        v.itos = list(itos); v.stoi = {c: i for i, c in enumerate(v.itos)}
        return v


def pad(seqs):
    m = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), m, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x, torch.tensor([len(s) for s in seqs])


class Encoder(nn.Module):
    """char bi-GRU (sequential -> under-uses the GPU; kept as a baseline option)."""
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


class CNNEncoder(nn.Module):
    """char 1D-CNN: multi-width conv n-gram detectors + masked global max-pool. Fully PARALLEL (high
    GPU utilization, much faster than the RNN) and its n-gram features suit lexical matching/copy."""
    def __init__(self, n_chars, d=256, drop=0.2, kernels=(3, 4, 5, 7)):
        super().__init__()
        self.emb = nn.Embedding(n_chars, d, padding_idx=0)
        self.drop = nn.Dropout(drop)
        nf = max(1, d // len(kernels))
        self.convs = nn.ModuleList([nn.Conv1d(d, nf, k, padding=k // 2) for k in kernels])
        self.proj = nn.Linear(nf * len(kernels), d)

    def forward(self, ids, lens):
        L = ids.size(1)
        mask = (ids != 0).unsqueeze(1)                       # (B,1,L) real-token mask
        e = self.drop(self.emb(ids)).transpose(1, 2)         # (B, d, L)
        feats = []
        for c in self.convs:
            h = F.relu(c(e))[..., :L]                        # (B, nf, L); crop (even kernels add 1)
            h = h.masked_fill(~mask, -1e4)                   # ignore padding in the max
            feats.append(h.max(dim=-1).values)              # global max-pool -> strongest n-gram
        z = self.proj(torch.cat(feats, dim=-1))
        return F.normalize(z, dim=-1)


def build_encoder(kind, n_chars, d):
    return CNNEncoder(n_chars, d=d) if kind == "cnn" else Encoder(n_chars, d=d)


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


def pad_to_tensor(id_lists, device, lmax=None):
    """ONE-TIME pre-pad: a list of char-id lists -> a fixed-width (N, Lmax) LongTensor on device.
    Done once at setup so per-step work is pure GPU indexing (no Python pad() in the loop)."""
    lmax = lmax or max((len(s) for s in id_lists), default=1)
    t = torch.zeros(len(id_lists), lmax, dtype=torch.long)
    for i, s in enumerate(id_lists):
        s = s[:lmax]
        t[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return t.to(device)


def pad_words_to_tensor(word_id_lists_per_concept, device, W=24, Lw=18):
    """ONE-TIME pre-pad of gloss WORDS -> (N, W, Lw) ids + (N, W) word-validity mask, on device."""
    N = len(word_id_lists_per_concept)
    t = torch.zeros(N, W, Lw, dtype=torch.long)
    m = torch.zeros(N, W, dtype=torch.bool)
    for i, ws in enumerate(word_id_lists_per_concept):
        for j, w in enumerate(ws[:W]):
            w = w[:Lw]; t[i, j, :len(w)] = torch.tensor(w, dtype=torch.long); m[i, j] = True
    return t.to(device), m.to(device)


def enc_rows(enc, T, idx):
    """Encode rows `idx` of a pre-padded id-tensor T -> normalized embeddings. Pure GPU indexing; lens
    derived on-device. No per-step Python pad."""
    ids = T[idx]
    lens = (ids != 0).sum(1).clamp(min=1)
    return enc(ids, lens)                                    # encoder normalizes


def enc_word_rows(enc, Tw, Twm, idx):
    """Encode gloss words for rows `idx` from the (N,W,Lw) tensor -> (B,W,d) normalized + (B,W) mask.
    All B*W words go through the encoder in ONE forward."""
    w = Tw[idx]; B, W, Lw = w.shape
    ids = w.reshape(B * W, Lw)
    lens = (ids != 0).sum(1).clamp(min=1)
    e = enc(ids, lens).reshape(B, W, -1)
    return F.normalize(e, dim=-1), Twm[idx]


def build_cache(concepts, vocab, device, W=24, Lw=18, Lmax=160):
    """Encode EVERY text into fixed-width id TENSORS once (def / positive-view / synonyms / gloss-words /
    parent names) -> training/eval is pure GPU indexing afterward (no per-step Python pad/encode)."""
    def posview(c):                                          # positive contrastive view: example>syn>def
        ex = [v for v in c["views"] if v.startswith("as in")]
        sy = [v for v in c["views"] if v.startswith("also called")]
        return (ex or sy or [c["views"][0]])[0]
    pnames = sorted({c["parent"] for c in concepts if c["parent"]})
    pidx = {p: i for i, p in enumerate(pnames)}
    word_lists = [[vocab.encode(w, Lw) for w in gloss_words(c["views"][0], W)] for c in concepts]
    words_T, words_M = pad_words_to_tensor(word_lists, device, W, Lw)
    return dict(
        def_T=pad_to_tensor([vocab.encode(c["views"][0], Lmax) for c in concepts], device),
        pos_T=pad_to_tensor([vocab.encode(posview(c), Lmax) for c in concepts], device),
        syn_T=pad_to_tensor([vocab.encode(f"also called {c['syn']}", Lmax) if c["syn"] else [0]
                             for c in concepts], device),
        words_T=words_T, words_M=words_M,
        pname_T=pad_to_tensor([vocab.encode(parent_name_text(p), Lmax) for p in pnames], device),
        par_idx=torch.tensor([pidx.get(c["parent"], -1) for c in concepts], dtype=torch.long, device=device),
        pos_lab=torch.tensor([POS_IDS[c["pos"]] for c in concepts], dtype=torch.long, device=device),
        syn_present=torch.tensor([c["syn"] is not None for c in concepts], device=device),
        pnames=pnames, ancestors=[({c["parent"]} | set(c["ancestors"])) if c["parent"] else set()
                                  for c in concepts])


def isa_scores(enc, cache, sel, keys, isa_mode, temp):
    """(len(sel), len(keys)) is-a scores from pre-padded tensors. copy: best-matching gloss WORD."""
    if isa_mode == "copy":
        we, m = enc_word_rows(enc, cache["words_T"], cache["words_M"], sel)
        s = torch.einsum("bwd,pd->bwp", we, keys)
        return s.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values / temp
    return (enc_rows(enc, cache["def_T"], sel) @ keys.t()) / temp


def train(enc, pos_head, cache, tr_idx, steps, device, batch=128, lr=1e-3,
          temp=0.07, lam_isa=1.0, lam_pos=0.3, seed=0, log=0, isa_mode="copy"):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pos_head.parameters()), lr=lr, weight_decay=1e-5)
    use_amp = device == "cuda"                               # mixed precision -> ~2x throughput on H100
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    enc.train(); pos_head.train()
    n = len(tr_idx)
    for step in range(steps):
        sel = tr_idx[torch.randint(0, n, (min(batch, n),), device=device)]             # concept indices
        with torch.autocast(device_type="cuda", enabled=use_amp):
            a = enc_rows(enc, cache["def_T"], sel)           # definition view
            b = enc_rows(enc, cache["pos_T"], sel)           # positive view (example/synonyms)
            logits = a @ b.t() / temp                        # InfoNCE (CLIP-style, in-batch negatives)
            labels = torch.arange(len(sel), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            # is-a probe (name | copy): concept -> its parent (in-batch parents as candidates)
            pi = cache["par_idx"][sel]
            nv = pi >= 0
            if nv.any():
                sub, kidx = sel[nv], pi[nv]
                keys = F.normalize(enc_rows(enc, cache["pname_T"], kidx), dim=-1)
                sc = isa_scores(enc, cache, sub, keys, isa_mode, temp)
                lbl = torch.arange(len(sub), device=device)
                loss = loss + lam_isa * 0.5 * (F.cross_entropy(sc, lbl) + F.cross_entropy(sc.t(), lbl))
            loss = loss + lam_pos * F.cross_entropy(pos_head(a), cache["pos_lab"][sel])  # POS probe
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if log and (step % log == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {loss.item():.3f}", flush=True)


# ===================================================================================================
# Eval: zero-shot retrieval + brain-verified is-a (seen/unseen parent) + POS accuracy
# ===================================================================================================
@torch.no_grad()
def evaluate(enc, pos_head, cache, tr_idx, te_idx, parent, device, isa_mode="copy",
             brain=None, names=None, kp_cap=200, neg_cap=60):
    enc.eval(); pos_head.eval()
    te_set = te_idx.tolist()

    # --- (1) zero-shot meaning retrieval: held-out definition -> its OWN synonyms among all held syn ---
    held = te_idx[cache["syn_present"][te_idx]]
    defs = enc_rows(enc, cache["def_T"], held)
    syns = enc_rows(enc, cache["syn_T"], held)
    sim = defs @ syns.t()
    ar = torch.arange(len(held), device=device)
    rank1 = (sim.argmax(1) == ar).float().mean().item()
    ranks = (sim >= sim.gather(1, ar[:, None]).expand_as(sim)).sum(1).float()
    mrr = (1.0 / ranks).mean().item()

    # --- (2) brain-verified is-a via the POINTER over ALL candidate parent NAMES (incl. held parents,
    #         so unseen parents are scorable by name -> the open-vocab-wall test) ---
    def anc_set(name):
        out, cur, seen = set(), name, set()
        while cur in parent and cur not in seen:
            seen.add(cur); cur = parent[cur]; out.add(cur)
        return out
    pnames = cache["pnames"]
    cand_emb = F.normalize(enc_rows(enc, cache["pname_T"],
                                    torch.arange(len(pnames), device=device)), dim=-1)
    train_parents = {pnames[i] for i in cache["par_idx"][tr_idx].tolist() if i >= 0}

    def isa_eval(idx_list):
        rows = [i for i in idx_list if cache["par_idx"][i].item() >= 0]
        if not rows:
            return []
        sel = torch.tensor(rows, dtype=torch.long, device=device)
        pick = isa_scores(enc, cache, sel, cand_emb, isa_mode, temp=1.0).argmax(1).tolist()
        recs = []
        for ci, pj in zip(rows, pick):
            pred = pnames[pj]
            got = {pred} | anc_set(pred)
            true = cache["ancestors"][ci]
            recs.append(dict(parent_ok=int(pred == pnames[cache["par_idx"][ci].item()]),
                             faith=len(got & true) / max(1, len(got)),
                             recall=len(got & true) / max(1, len(true)),
                             seen=pnames[cache["par_idx"][ci].item()] in train_parents))
        return recs

    te_isa = isa_eval(te_set)
    seen = [r for r in te_isa if r["seen"]]
    unseen = [r for r in te_isa if not r["seen"]]
    agg = lambda rs, k: (sum(r[k] for r in rs) / len(rs)) if rs else 0.0

    # --- (2b) BRAIN INTEGRATION: verify the model's is-a claims with KERNEL PROOFS (the project's
    #          purpose -- model proposes, the brain proves), and PROVE negations via the BDD type
    #          disjointness/exclusion axioms. faith here = fraction of conveyed facts the kernel proves.
    kp = dict(proven=0, conveyed=0, recovered=0, true=0, n=0, neg_proven=0, neg_n=0)
    if brain is not None and names is not None:
        banc = brain_ancestor_index(brain)
        rows = [i for i in te_set if cache["par_idx"][i].item() >= 0][:kp_cap]
        if rows:
            sel = torch.tensor(rows, dtype=torch.long, device=device)
            pick = isa_scores(enc, cache, sel, cand_emb, isa_mode, temp=1.0).argmax(1).tolist()
            kp["n"] = len(rows)
            for ci, pj in zip(rows, pick):
                C, P = names[ci], pnames[pj]
                conveyed = {P} | banc.get(P, set())          # brain derives the chain above P
                pv = [x for x in conveyed if brain._atom_term(("isa", (C, x)))[0]]   # KERNEL-checked
                truec = banc.get(C, set())
                kp["proven"] += len(pv); kp["conveyed"] += len(conveyed)
                kp["recovered"] += len(set(pv) & truec); kp["true"] += len(truec)
            # negation: prove "C is NOT B" from category disjointness (exclusion axioms)
            allcats = sorted({a for s in banc.values() for a in s})
            rng = np.random.default_rng(0)
            for ci in rows[:neg_cap]:
                C, truec = names[ci], banc.get(names[ci], set())
                cands = [b for b in allcats if b not in truec and b != names[ci]]
                if not cands:
                    continue
                B = cands[int(rng.integers(0, len(cands)))]
                kp["neg_n"] += 1
                st = brain._prove_core(("not", ("atom", "isa", (C, B))))
                kp["neg_proven"] += int(st.get("status") == "proven")

    # --- (3) POS classification accuracy on held-out ---
    pred = pos_head(enc_rows(enc, cache["def_T"], te_idx)).argmax(1)
    pos_acc = (pred == cache["pos_lab"][te_idx]).float().mean().item()

    return dict(
        retr_at1=rank1, retr_mrr=mrr, n_retr=len(held),
        isa_parent=agg(te_isa, "parent_ok"), isa_faith=agg(te_isa, "faith"),
        isa_recall=agg(te_isa, "recall"), n_isa=len(te_isa),
        isa_seen=dict(n=len(seen), parent=agg(seen, "parent_ok"), faith=agg(seen, "faith"),
                      recall=agg(seen, "recall")),
        isa_unseen=dict(n=len(unseen), parent=agg(unseen, "parent_ok"), faith=agg(unseen, "faith"),
                        recall=agg(unseen, "recall")),
        pos_acc=pos_acc,
        kernel_faith=kp["proven"] / max(1, kp["conveyed"]),
        kernel_recall=kp["recovered"] / max(1, kp["true"]),
        kernel_n=kp["n"], neg_proven=kp["neg_proven"] / max(1, kp["neg_n"]), neg_n=kp["neg_n"])


def run(steps=4000, seed=0, per_pos=150, d=256, device="cpu", batch=128, verbose=True, out=None,
        isa_mode="copy", encoder="cnn"):
    concepts, parent, _node_text, relations = gather(per_pos=per_pos, seed=seed)
    tr, te = split(concepts, 0.25, seed)
    idx = {id(c): i for i, c in enumerate(concepts)}
    names = [c["name"] for c in concepts]
    vocab = CharVocab([v for c in tr for v in c["views"]]
                      + [parent_name_text(c["parent"]) for c in tr if c["parent"]])
    # VECTORIZE: encode every text into fixed-width id TENSORS ONCE; train/eval are pure GPU indexing.
    cache = build_cache(concepts, vocab, device)
    brain = build_brain(concepts, relations)                 # provable BRAIN: many relations + rules
    tr_idx = torch.tensor([idx[id(c)] for c in tr], dtype=torch.long, device=device)
    te_idx = torch.tensor([idx[id(c)] for c in te], dtype=torch.long, device=device)
    enc = build_encoder(encoder, len(vocab), d).to(device)
    pos_head = nn.Linear(d, 4).to(device)
    if verbose:
        from collections import Counter
        pc = Counter(c["pos"] for c in concepts)
        rels = ", ".join(f"{k}:{len(v)}" for k, v in relations.items() if v)
        print(f"  WordNet ALL-POS | {len(concepts)} concepts {dict(pc)} | train {len(tr)}/held {len(te)}"
              f" | char-vocab {len(vocab)} | enc={encoder} isa={isa_mode} | device {device}", flush=True)
        print(f"  BRAIN relations->rules: [{rels}] | {len(brain.known)} kernel-closed facts", flush=True)
    train(enc, pos_head, cache, tr_idx, steps, device, batch=batch, seed=seed,
          isa_mode=isa_mode, log=(max(1, steps // 6) if verbose else 0))
    r = evaluate(enc, pos_head, cache, tr_idx, te_idx, parent, device, isa_mode,
                 brain=brain, names=names)
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
        print(f"  BRAIN (KERNEL-PROVEN, n={r['kernel_n']}): faith {r['kernel_faith']:.3f} | recall "
              f"{r['kernel_recall']:.3f}   -- each conveyed fact carries a kernel-checked proof term")
        print(f"  BRAIN NEGATION (n={r['neg_n']}): proved 'X is NOT a Y' in {r['neg_proven']:.3f} via BDD "
              f"type disjointness -> exclusion axioms (understanding what things are NOT)")
        print("  the model PROPOSES; the BRAIN PROVES (datalog closure + kernel + set-theoretic types).")
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
    ap.add_argument("--isa-mode", choices=["name", "copy"], default="copy",
                    help="is-a probe: 'name' (pooled gloss vs parent name) or "
                         "'copy' (attention/copy -- best-matching gloss WORD vs parent name)")
    ap.add_argument("--encoder", choices=["cnn", "gru"], default="cnn",
                    help="text encoder: 'cnn' (parallel, fast, default) or 'gru' (sequential baseline)")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args.steps, seed=args.seed, per_pos=args.per_pos, d=args.d, device=device,
        batch=args.batch, out=args.out, isa_mode=args.isa_mode, encoder=args.encoder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
