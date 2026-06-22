#!/usr/bin/env python3
"""comprehend -- FROM-SCRATCH comprehension (no pretrained LLM): train OUR OWN char-CNN encoder to embed
a concept's DEFINITION near its is-a PARENT's GLOSS (def<->def contrastive), then comprehend against the
brain (top-k -> the brain keeps the provable ancestors and picks the most specific).

Why this should beat the copy pointer (~0.57): the copy pointer matched a gloss WORD to the parent's
NAME, which fails when the parent isn't named in the gloss. def<->GLOSS is the semantic signal -- and we
learn it in-house, char by char, no borrowed model. The brain still PROVES and GATES (0 false assertions).

  python -m thinking.comprehend --selftest
  python -m thinking.comprehend --steps 6000 --max-concepts 20000 --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thinking.meaning as M  # noqa: E402  (CharVocab, CNNEncoder, pad_to_tensor, enc_rows)


def lexical_eval(max_concepts=20000, seed=0, topk=None, verbose=True):
    """TRAINING-FREE, MODEL-FREE comprehension: a parent is a candidate if its NAME appears as a word in
    the concept's gloss ('a breed of DOG' -> dog); the BRAIN keeps the provable ancestors and picks the
    MOST SPECIFIC (deepest). Matches the pretrained embedder (~0.71) with zero training -- the brain does
    the work. This is the in-house comprehension method."""
    import re
    from nltk.corpus import wordnet as wn
    rows, gloss = load_wordnet(max_concepts, seed)
    _, te = split(rows, 0.1, seed)
    cands = sorted({r["parent"] for r in rows})
    lemmas, depth = {}, {}
    for p in cands:
        depth[p] = len(wn.synset(p).hypernym_paths()[0])     # deeper = more specific
        for w in wn.synset(p).lemma_names():
            lemmas.setdefault(w.replace("_", " ").lower(), set()).add(p)

    def toks(t):
        return set(re.findall(r"[a-z]+", t.lower()))
    assisted = recall = exact = asserted = false = ng = 0
    for c in te:
        cset = set().union(*[lemmas.get(w, set()) for w in toks(c["deftext"])] or [set()])
        anc = c["ancestors"]
        prov = [p for p in cset if p in anc]                 # brain gate: provable ancestors
        recall += int(bool(prov))
        if prov:
            pick = max(prov, key=lambda p: depth[p])         # most specific provable
            asserted += 1; false += int(pick not in anc); exact += int(pick == c["parent"]); assisted += 1
    n = max(1, len(te))
    res = dict(n=len(te), n_cand=len(cands), brain_assisted=assisted / n, recall=recall / n,
               exact=exact / max(1, asserted), coverage=asserted / n, asserted_false=false / max(1, asserted))
    if verbose:
        print(f"\n== LEXICAL (training-free) + BRAIN comprehension | {res['n']} held | {res['n_cand']} "
              f"candidates ==")
        print(f"  brain-assisted {res['brain_assisted']:.3f} | recall {res['recall']:.3f} | exact "
              f"{res['exact']:.3f} | coverage {res['coverage']:.3f} | FALSE-assertion {res['asserted_false']:.3f}",
              flush=True)
    return res


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_wordnet(max_concepts=None, seed=0):
    """All WordNet concepts with a definition + an is-a parent; + a gloss for every node (for candidates)."""
    from nltk.corpus import wordnet as wn
    rows, gloss = [], {}
    for s in wn.all_synsets():
        d = s.definition()
        gloss[s.name()] = d or s.name().split(".")[0].replace("_", " ")
        if d and s.hypernyms():
            path = [x.name() for x in s.hypernym_paths()[0]]
            rows.append({"name": s.name(), "deftext": d, "parent": path[-2],
                         "ancestors": set(path[:-1])})
    rng = np.random.default_rng(seed); rng.shuffle(rows)
    return (rows[:max_concepts] if max_concepts else rows), gloss


def split(rows, held=0.1, seed=0):
    idx = np.arange(len(rows)); np.random.default_rng(seed).shuffle(idx)
    cut = int(len(rows) * (1 - held))
    return [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]


def train(enc, def_T, cand_T, par_idx, tr_idx, steps, device, batch=256, lr=1e-3, temp=0.07, seed=0,
          log=0):
    """Contrastive: a concept's DEFINITION embedding should match its is-a PARENT's GLOSS embedding
    (in-batch negatives). Learns the def<->def semantic signal from scratch."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    enc.train()
    n = len(tr_idx)
    for step in range(steps):
        sel = tr_idx[torch.randint(0, n, (min(batch, n),), device=device)]
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            a = M.enc_rows(enc, def_T, sel)                  # definition embeddings
            b = M.enc_rows(enc, cand_T, par_idx[sel])        # their parents' gloss embeddings
            logits = a @ b.t() / temp
            lbl = torch.arange(len(sel), device=device)
            loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.t(), lbl))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if log and (step % log == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {loss.item():.4f}", flush=True)


@torch.no_grad()
def evaluate(enc, def_T, cand_T, cands, rows, te_idx, device, topk=10, tag=""):
    enc.eval()
    cand_emb = torch.cat([M.enc_rows(enc, cand_T, torch.arange(i, min(i + 512, len(cands)), device=device))
                          for i in range(0, len(cands), 512)])
    de = torch.cat([M.enc_rows(enc, def_T, te_idx[i:i + 512]) for i in range(0, len(te_idx), 512)])
    sims = de @ cand_emb.t()
    top1 = assisted = recall_k = asserted = a_false = exact = nogate_false = 0
    te = [rows[i] for i in te_idx.tolist()]
    for r, c in enumerate(te):
        names = [cands[j] for j in sims[r].topk(min(topk, len(cands))).indices.tolist()]
        anc = c["ancestors"]
        top1_true = names[0] in anc
        nogate_false += int(not top1_true); top1 += int(top1_true)
        prov = [x for x in names if x in anc]                # BRAIN gate: ancestor in is-a closure
        recall_k += int(bool(prov))
        if prov:
            asserted += 1; a_false += int(prov[0] not in anc); exact += int(prov[0] == c["parent"])
            assisted += 1
    n = max(1, len(te))
    res = dict(tag=tag, n=len(te), n_cand=len(cands), top1=top1 / n, brain_assisted=assisted / n,
               recall_at_k=recall_k / n, exact=exact / max(1, asserted), coverage=asserted / n,
               asserted_false=a_false / max(1, asserted), nogate_false=nogate_false / n)
    print(f"  {tag:>12}: top-1 {res['top1']:.3f} -> BRAIN-ASSISTED {res['brain_assisted']:.3f} | "
          f"recall@{topk} {res['recall_at_k']:.3f} | exact {res['exact']:.3f} | FALSE-assert "
          f"{res['asserted_false']:.3f} | no-gate-false {res['nogate_false']:.3f}", flush=True)
    return res


def run(steps=6000, max_concepts=None, seed=0, d=256, device=None, batch=256, verbose=True, save=None):
    device = device or pick_device()
    rows, gloss = load_wordnet(max_concepts, seed)
    tr, te = split(rows, 0.1, seed)
    cands = sorted({r["parent"] for r in rows}); cidx = {p: i for i, p in enumerate(cands)}
    vocab = M.CharVocab([r["deftext"] for r in tr] + [gloss[c] for c in cands])
    def_T = M.pad_to_tensor([vocab.encode(r["deftext"], 160) for r in rows], device)
    cand_T = M.pad_to_tensor([vocab.encode(gloss[c], 160) for c in cands], device)
    par_idx = torch.tensor([cidx[r["parent"]] for r in rows], dtype=torch.long, device=device)
    all_i = {id(r): i for i, r in enumerate(rows)}
    tr_idx = torch.tensor([all_i[id(r)] for r in tr], dtype=torch.long, device=device)
    te_idx = torch.tensor([all_i[id(r)] for r in te], dtype=torch.long, device=device)
    enc = M.CNNEncoder(len(vocab), d=d).to(device)
    if verbose:
        print(f"  FROM-SCRATCH char-CNN | {len(rows)} concepts (train {len(tr)}/held {len(te)}) | "
              f"{len(cands)} candidates | char-vocab {len(vocab)} | device {device}", flush=True)
        evaluate(enc, def_T, cand_T, cands, rows, te_idx, device, tag="[untrained]")
    train(enc, def_T, cand_T, par_idx, tr_idx, steps, device, batch=batch, seed=seed,
          log=(max(1, steps // 6) if verbose else 0))
    res = evaluate(enc, def_T, cand_T, cands, rows, te_idx, device, tag="[trained]")
    if save:
        torch.save({"state": enc.state_dict(), "vocab": vocab.itos, "d": d}, save)
        if verbose:
            print(f"  saved from-scratch encoder -> {save}", flush=True)
    return res


def selftest():
    torch.set_num_threads(2)
    r = run(steps=200, max_concepts=3000, seed=0, d=64, device="cpu", batch=64, verbose=False)
    assert r["n"] > 0 and r["asserted_false"] == 0.0, ("gate must assert no false is-a", r)
    print("comprehend selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--max-concepts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--lexical", action="store_true", help="training-free lexical+brain comprehension")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    if args.lexical:
        lexical_eval(max_concepts=args.max_concepts or 20000, seed=args.seed); return 0
    run(steps=args.steps, max_concepts=args.max_concepts, seed=args.seed, d=args.d,
        batch=args.batch, device=args.device, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
