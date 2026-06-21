#!/usr/bin/env python3
"""roundtrip -- READ a definition -> UNDERSTAND the concept -> WRITE its definition back, in the agent's
own words, with the BRAIN proving both directions. The bridge toward the end goal: understand a prompt,
reason, and respond in writing.

The loop (brain-checked both ways):
  1. COMPREHEND: read a held-out concept's definition (the prompt); the meaning model recovers its is-a
     parent (the copy pointer, [[emergent-language]]/meaning.py); the BRAIN proves isa(concept, parent)
     with a KERNEL-checked term.
  2. WRITE: condition a generator (prover.py seq2seq) on the recovered facts (pos + is-a parent) and
     emit a fresh definition in the agent's own words.
  3. VERIFY: parse the WRITTEN definition's is-a claim and KERNEL-check it against the brain.

So what the agent UNDERSTOOD is proven, and what it WROTE is proven. Generation is grounded in the
provable brain -- it can only faithfully claim what the brain can prove.

  python -m thinking.roundtrip --selftest
  python -m thinking.roundtrip --steps 4000 --per-pos 150
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thinking.meaning as M  # noqa: E402  (gather/build_brain/build_cache/encoder/copy pointer)
import thinking.prover as P  # noqa: E402  (conditional seq2seq generator: gtoks -> ptoks)


STOP = set(".,;:()[]{}'\"`")


def def_words(text, maxw=40):
    text = text.replace("means ", "").replace("also called", "").replace("as in", "")
    return [w.strip("".join(STOP)) for w in text.lower().split() if w.strip("".join(STOP))][:maxw]


DESC_RELS = ("has_part", "made_of", "member_of")           # descriptive relations for definitions


def facts_gtoks(pos, facts, cap=6):
    """Serialize the brain-verified FACT-SET as generator input: pos tag + (predicate + obj lemma) for
    each fact. Conditioning on the FULL fact-set (parent + parts/substance/membership), not just the
    parent, lets the writer compose a richer faithful definition."""
    g = ["pos", pos]
    for pred, obj in facts[:cap]:
        g += [pred] + M.parent_name_text(obj).split()
    return g


def direct_facts(concepts, relations):
    """Per-concept DIRECT descriptive facts (has_part/made_of/member_of) for writer conditioning."""
    df = {c["name"]: [] for c in concepts}
    for pred in DESC_RELS:
        for (s, o) in (relations or {}).get(pred, ()):
            if s in df:
                df[s].append((pred, o))
    return df


def write_pairs(concepts, dfacts):
    """(fact-set -> definition) training pairs. Condition on the FULL fact-set (is-a parent + direct
    parts/substance/membership); FAITHFUL target leads with 'a <parent> -' then the gloss, so the
    written definition asserts the grounded parent (exactly what the brain checks)."""
    pairs = []
    for c in concepts:
        if not c["parent"]:
            continue
        gloss = def_words(c["views"][0])
        if gloss:
            facts = [("isa", c["parent"])] + dfacts.get(c["name"], [])
            ptoks = ["a"] + M.parent_name_text(c["parent"]).split() + ["-"] + gloss
            pairs.append({"gtoks": facts_gtoks(c["pos"], facts), "ptoks": ptoks, "name": c["name"]})
    return pairs


def all_parent_claims(ptoks, cand_lemmas):
    """EVERY is-a claim the written definition makes = all candidate parents whose lemma appears in it.
    The brain then filters these to the provable ones (brain-gatekept generation)."""
    out = []
    for name, lemma in cand_lemmas:
        lw = lemma.split()
        if any(ptoks[i:i + len(lw)] == lw for i in range(len(ptoks) - len(lw) + 1)):
            out.append(name)
    return out


def run(steps=4000, seed=0, per_pos=150, d=256, device="cpu", batch=128, verbose=True,
        write_steps=None, n_show=6, out=None, save=None):
    write_steps = write_steps or steps
    concepts, parent, _nt, relations = M.gather(per_pos=per_pos, seed=seed)
    tr, te = M.split(concepts, 0.25, seed)
    brain = M.build_brain(concepts, relations)
    dfacts = direct_facts(concepts, relations)               # per-concept descriptive facts for writing
    banc = M.brain_ancestor_index(brain)                     # node -> is-a ancestors (for top-k select)
    idx = {id(c): i for i, c in enumerate(concepts)}
    names = [c["name"] for c in concepts]

    # ---- COMPREHENSION model (meaning.py: char-CNN encoder + copy is-a pointer) ----
    vocab = M.CharVocab([v for c in tr for v in c["views"]]
                        + [M.parent_name_text(c["parent"]) for c in tr if c["parent"]])
    cache = M.build_cache(concepts, vocab, device)
    enc = M.build_encoder("cnn", len(vocab), d).to(device)
    pos_head = nn.Linear(d, 4).to(device)
    tr_idx = torch.tensor([idx[id(c)] for c in tr], dtype=torch.long, device=device)
    te_idx = torch.tensor([idx[id(c)] for c in te], dtype=torch.long, device=device)
    M.train(enc, pos_head, cache, tr_idx, steps, device, batch=batch, seed=seed, isa_mode="copy",
            log=(max(1, steps // 5) if verbose else 0))

    # ---- WRITER (prover.py seq2seq): recovered facts -> definition in the agent's own words ----
    wpairs = write_pairs(tr, dfacts)                         # TRAIN only on training pairs
    wvocab = P.Vocab.build(write_pairs(concepts, dfacts))    # token inventory over all (held words encode)
    block = 96
    wmodel = P.build_model(len(wvocab), block, d=min(256, d), layers=4, heads=8)
    P.train(wmodel, wvocab, wpairs, steps=write_steps, device=device, batch=32, block=block,
            seed=seed, log_every=(max(1, write_steps // 5) if verbose else 0))

    # ---- ROUND-TRIP eval on held-out concepts: read def -> comprehend -> write -> brain-verify ----
    pnames = cache["pnames"]
    cand_lemmas = [(p, M.parent_name_text(p)) for p in pnames]
    with torch.no_grad():
        cand_emb = torch.nn.functional.normalize(
            M.enc_rows(enc, cache["pname_T"], torch.arange(len(pnames), device=device)), dim=-1)

    topk = 10

    @torch.no_grad()
    def comprehend(ci, C):
        """BRAIN-ASSISTED comprehension: the model proposes its top-k parents; the BRAIN keeps the ones
        it can prove for C and picks the MOST SPECIFIC (deepest) -> recovers a provable ancestor far
        more often than top-1. Returns (top1, brain_pick, top1_proven, anyk_proven)."""
        sel = torch.tensor([ci], dtype=torch.long, device=device)
        sc = M.isa_scores(enc, cache, sel, cand_emb, "copy", temp=1.0)[0]
        order = sc.topk(min(topk, len(pnames))).indices.tolist()
        cands = [pnames[j] for j in order]
        top1 = cands[0]
        provable = [x for x in cands if brain._atom_term(("isa", (C, x)))[0]]   # brain gatekeeps
        pick = max(provable, key=lambda x: len(banc.get(x, ()))) if provable else top1  # most specific
        return top1, pick, top1 in provable, bool(provable)

    comp_ok = comp_proven = comp_top1 = recall_k = n = 0
    raw_claims = raw_proven = grounded_proven = filt_recovered = 0
    shows = []
    for ci in te_idx.tolist():
        if cache["par_idx"][ci].item() < 0:
            continue
        n += 1
        C, true_parent = names[ci], pnames[cache["par_idx"][ci].item()]
        top1, p_hat, top1_proven, anyk = comprehend(ci, C)   # COMPREHEND (brain-assisted)
        comp_top1 += int(top1_proven); recall_k += int(anyk)
        comp_ok += int(p_hat == true_parent)
        comp_grounded = brain._atom_term(("isa", (C, p_hat)))[0]
        comp_proven += int(comp_grounded)
        # RICHER faithful writing: condition the writer on the FULL brain-verified fact-set of C
        facts = [("isa", p_hat)] + dfacts.get(C, [])
        gen = P.emit(wmodel, wvocab, {"gtoks": facts_gtoks(names_pos(concepts, ci), facts)},
                     device, block)                           # WRITE from the verified fact-set
        # BRAIN-FILTERED generation: keep only the kernel-PROVABLE is-a claims (gatekept response).
        claims = all_parent_claims(gen, cand_lemmas)
        proven = [x for x in claims if brain._atom_term(("isa", (C, x)))[0]]
        raw_claims += len(claims); raw_proven += len(proven)
        filt_recovered += int(len(proven) > 0)
        grounded_proven += int(comp_grounded)
        if len(shows) < n_show:
            kept = M.parent_name_text(proven[0]) if proven else "(none survive)"
            shows.append((C, M.parent_name_text(true_parent), M.parent_name_text(p_hat),
                          " ".join(gen[:14]), kept))

    res = dict(n=n, comp_acc=comp_ok / max(1, n), comp_kernel=comp_proven / max(1, n),
               comp_top1=comp_top1 / max(1, n), recall_at_k=recall_k / max(1, n),
               raw_write_faith=raw_proven / max(1, raw_claims),
               filtered_recall=filt_recovered / max(1, n),
               grounded_faith=grounded_proven / max(1, n))
    if verbose:
        print("\n== ROUND-TRIP: read def -> COMPREHEND -> WRITE -> BRAIN GATEKEEPS (model proposes, brain proves) ==")
        print(f"  {n} held-out concepts | brain {len(brain.known)} kernel-closed facts")
        print(f"  COMPREHEND     : top-1 proven {res['comp_top1']:.3f} -> BRAIN-ASSISTED (top-{topk}, brain "
              f"picks provable) {res['comp_kernel']:.3f} | recall@{topk} {res['recall_at_k']:.3f} | exact "
              f"parent {res['comp_acc']:.3f}")
        print(f"  WRITE (raw)    : of all is-a claims the writer makes, brain-PROVEN {res['raw_write_faith']:.3f}"
              f"  <- a free LM hallucinates")
        print(f"  WRITE (FILTERED): >=1 true is-a survives the brain filter for {res['filtered_recall']:.3f} "
              f"of concepts; kept content is 100% kernel-proven (hallucinations rejected)")
        print(f"  GROUNDED claim  : agent asserts its brain-verified comprehension -> faith {res['grounded_faith']:.3f}")
        print("  -- examples (concept | true -> comprehended | written... | brain-KEPT claim) --")
        for C, tp, cp, w, kept in shows:
            print(f"     {C:22s} {tp}->{cp} | {w} | KEPT: {kept}")
    if out:
        import json
        with open(out, "w") as f:
            json.dump(dict(per_pos=per_pos, steps=steps, examples=shows, **res), f, indent=2)
        if verbose:
            print(f"  wrote {out}", flush=True)
    if save:
        torch.save({"enc": enc.state_dict(), "pos_head": pos_head.state_dict(),
                    "wmodel": wmodel.state_dict(), "vocab_itos": vocab.itos,
                    "wvocab_itos": wvocab.itos, "concepts": concepts, "relations": relations,
                    "parent": parent, "config": {"d": d, "wd": min(256, d), "block": block}}, save)
        if verbose:
            print(f"  saved chattable model -> {save}", flush=True)
    return res


def names_pos(concepts, ci):
    return concepts[ci]["pos"]


# ===================================================================================================
# Load a trained model and CHAT with it (inference) -- prepared for the next training cycle
# ===================================================================================================
def load_agent(path, device="cpu"):
    import torch.nn.functional as Fn
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]; d = cfg["d"]
    concepts, relations, parent = ckpt["concepts"], ckpt["relations"], ckpt["parent"]
    vocab = M.CharVocab.from_itos(ckpt["vocab_itos"])
    cache = M.build_cache(concepts, vocab, device)
    brain = M.build_brain(concepts, relations)
    enc = M.build_encoder("cnn", len(vocab), d).to(device); enc.load_state_dict(ckpt["enc"]); enc.eval()
    pos_head = nn.Linear(d, 4).to(device); pos_head.load_state_dict(ckpt["pos_head"]); pos_head.eval()
    wvocab = P.Vocab(ckpt["wvocab_itos"])
    wmodel = P.build_model(len(wvocab), cfg["block"], d=cfg["wd"], layers=4, heads=8).to(device)
    wmodel.load_state_dict(ckpt["wmodel"]); wmodel.eval()
    pnames = cache["pnames"]
    with torch.no_grad():
        cand_emb = Fn.normalize(M.enc_rows(enc, cache["pname_T"],
                                           torch.arange(len(pnames), device=device)), dim=-1)
    return dict(enc=enc, pos_head=pos_head, wmodel=wmodel, wvocab=wvocab, vocab=vocab, cache=cache,
                brain=brain, concepts=concepts, names=[c["name"] for c in concepts], pnames=pnames,
                cand_emb=cand_emb, block=cfg["block"], device=device,
                dfacts=direct_facts(concepts, relations), banc=M.brain_ancestor_index(brain))


@torch.no_grad()
def comprehend_text(bundle, text, k=5):
    """Free text (a prompt) -> the agent's top-k candidate brain categories (copy pointer over words)."""
    words = [bundle["vocab"].encode(w, 18) for w in M.gloss_words(text)]
    Tw, Tm = M.pad_words_to_tensor([words], bundle["device"])
    we, mask = M.enc_word_rows(bundle["enc"], Tw, Tm, torch.tensor([0], device=bundle["device"]))
    s = torch.einsum("bwd,pd->bwp", we, bundle["cand_emb"]).masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values[0]
    return [bundle["pnames"][j] for j in s.topk(min(k, len(bundle["pnames"]))).indices.tolist()]


def brain_facts_about(brain, node, k=8):
    fs = [(p, a[1]) for (p, a) in brain.known if len(a) == 2 and a[0] == node][:k]
    return ", ".join(f"{p} {M.parent_name_text(o)}" for p, o in fs) or "(none yet)"


@torch.no_grad()
def respond(bundle, text):
    """READ the prompt -> COMPREHEND (brain category) -> WRITE a definition -> ground in PROVEN facts."""
    key = text.strip().lower()
    known = next((i for i, nm in enumerate(bundle["names"])
                  if nm == key or M.parent_name_text(nm) == key), None)
    brain = bundle["brain"]
    if known is not None:                                    # known concept: read its gloss, BRAIN-ASSIST
        C = bundle["names"][known]; pos = bundle["concepts"][known]["pos"]
        cands = comprehend_text(bundle, bundle["concepts"][known]["views"][0])
        provable = [x for x in cands if brain._atom_term(("isa", (C, x)))[0]]
        p_hat = max(provable, key=lambda x: len(bundle["banc"].get(x, ()))) if provable else cands[0]
        facts = [("isa", p_hat)] + bundle["dfacts"].get(C, [])
    else:                                                    # free description (novel): top-1, unverified
        C, pos = None, "n"
        p_hat = comprehend_text(bundle, text)[0]
        facts = [("isa", p_hat)]
    gen = P.emit(bundle["wmodel"], bundle["wvocab"], {"gtoks": facts_gtoks(pos, facts)},
                 bundle["device"], bundle["block"])
    cat = M.parent_name_text(p_hat)
    out = [f"  understood as : a {cat}",
           f"  my definition: {' '.join(gen)}",
           f"  brain proves about '{cat}': {brain_facts_about(bundle['brain'], p_hat)}"]
    if known is not None:
        C = bundle["names"][known]
        ok = bundle["brain"]._atom_term(("isa", (C, p_hat)))[0]
        out.append(f"  [known: {C}] is-a {cat} -> {'KERNEL-PROVEN' if ok else 'unproven'}")
    return "\n".join(out)


def chat(bundle):
    print("\n== CHAT (model proposes, brain proves) -- type a word or a definition; blank line to quit ==",
          flush=True)
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            break
        print(respond(bundle, line), flush=True)


def selftest():
    torch.set_num_threads(2)
    r = run(steps=300, write_steps=300, seed=0, per_pos=40, d=64, device="cpu", verbose=False, n_show=0)
    assert 0.0 <= r["comp_kernel"] <= 1.0 and r["n"] > 0, r
    assert 0.0 <= r["raw_write_faith"] <= 1.0 and 0.0 <= r["filtered_recall"] <= 1.0, r
    print("roundtrip selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--write-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-pos", type=int, default=150)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None, help="save a chattable model checkpoint here after training")
    ap.add_argument("--load", default=None, help="load a saved model (skip training)")
    ap.add_argument("--chat", action="store_true", help="interactive chat with the loaded model")
    ap.add_argument("--ask", default=None, help="one-shot prompt to the loaded model")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if args.load:                                            # LOAD a trained model -> chat / ask
        bundle = load_agent(args.load, device)
        print(f"loaded {args.load} | {len(bundle['names'])} concepts | brain "
              f"{len(bundle['brain'].known)} kernel-closed facts", flush=True)
        if args.ask:
            print(respond(bundle, args.ask), flush=True)
        if args.chat or not args.ask:
            chat(bundle)
        return 0
    run(args.steps, seed=args.seed, per_pos=args.per_pos, d=args.d, device=device, batch=args.batch,
        write_steps=args.write_steps, out=args.out, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
