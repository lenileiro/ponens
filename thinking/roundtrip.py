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


def facts_gtoks(pos, parent_name):
    """Serialize the recovered FACTS as generator input: pos tag + is-a parent lemma words."""
    return ["pos", pos, "isa"] + M.parent_name_text(parent_name).split()


def write_pairs(concepts):
    """(facts -> definition) training pairs. FAITHFUL target: lead with the explicit is-a claim
    'a <parent> ...' then the gloss, so the written definition ASSERTS the parent it is grounded on
    (and the claim is exactly what the brain checks) rather than burying/omitting it."""
    pairs = []
    for c in concepts:
        if not c["parent"]:
            continue
        gloss = def_words(c["views"][0])
        if gloss:
            ptoks = ["a"] + M.parent_name_text(c["parent"]).split() + ["-"] + gloss
            pairs.append({"gtoks": facts_gtoks(c["pos"], c["parent"]), "ptoks": ptoks, "name": c["name"]})
    return pairs


def written_parent_claim(ptoks, cand_lemmas):
    """The is-a claim the WRITTEN definition actually makes = the EARLIEST-positioned candidate parent
    lemma appearing in it (the faithful writer leads with 'a <parent> ...'). Returns its name or None."""
    best, best_pos = None, len(ptoks) + 1
    for name, lemma in cand_lemmas:
        lw = lemma.split()
        for i in range(len(ptoks) - len(lw) + 1):
            if ptoks[i:i + len(lw)] == lw:
                if i < best_pos:
                    best, best_pos = name, i
                break
    return best


def run(steps=4000, seed=0, per_pos=150, d=256, device="cpu", batch=128, verbose=True,
        write_steps=None, n_show=6, out=None):
    write_steps = write_steps or steps
    concepts, parent, _nt, relations = M.gather(per_pos=per_pos, seed=seed)
    tr, te = M.split(concepts, 0.25, seed)
    brain = M.build_brain(concepts, relations)
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
    wpairs = write_pairs(tr)                                 # TRAIN only on training pairs
    wvocab = P.Vocab.build(write_pairs(concepts))            # token inventory over all (held words encode)
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

    @torch.no_grad()
    def comprehend(ci):                                      # def -> predicted is-a parent (copy pointer)
        sel = torch.tensor([ci], dtype=torch.long, device=device)
        pj = M.isa_scores(enc, cache, sel, cand_emb, "copy", temp=1.0).argmax(1).item()
        return pnames[pj]

    comp_ok = comp_proven = wrote = write_proven = n = 0
    shows = []
    for ci in te_idx.tolist():
        if cache["par_idx"][ci].item() < 0:
            continue
        n += 1
        C, true_parent = names[ci], pnames[cache["par_idx"][ci].item()]
        p_hat = comprehend(ci)                               # COMPREHEND
        comp_ok += int(p_hat == true_parent)
        comp_proven += int(brain._atom_term(("isa", (C, p_hat)))[0])      # BRAIN proves comprehension
        gen = P.emit(wmodel, wvocab, {"gtoks": facts_gtoks(names_pos(concepts, ci), p_hat)},
                     device, block)                           # WRITE from recovered facts
        claim = written_parent_claim(gen, cand_lemmas)       # parse the WRITTEN def's is-a claim
        if claim is not None:
            wrote += 1
            write_proven += int(brain._atom_term(("isa", (C, claim)))[0])  # BRAIN proves the WRITTEN claim
        if len(shows) < n_show:
            shows.append((C, M.parent_name_text(true_parent), M.parent_name_text(p_hat), " ".join(gen[:18])))

    res = dict(n=n, comp_acc=comp_ok / max(1, n), comp_kernel=comp_proven / max(1, n),
               wrote_frac=wrote / max(1, n), write_kernel=write_proven / max(1, wrote))
    if verbose:
        print("\n== ROUND-TRIP: read def -> COMPREHEND -> WRITE -> brain-verify (model proposes, brain proves) ==")
        print(f"  {n} held-out concepts | brain {len(brain.known)} kernel-closed facts")
        print(f"  COMPREHEND: parent acc {res['comp_acc']:.3f} | brain-PROVEN {res['comp_kernel']:.3f}")
        print(f"  WRITE     : produced an is-a claim in {res['wrote_frac']:.3f} | of those brain-PROVEN "
              f"{res['write_kernel']:.3f}")
        print("  -- examples (concept | true parent -> comprehended -> WRITTEN def) --")
        for C, tp, cp, w in shows:
            print(f"     {C:24s} {tp} -> {cp} :: {w}")
    if out:
        import json
        with open(out, "w") as f:
            json.dump(dict(per_pos=per_pos, steps=steps, examples=shows, **res), f, indent=2)
        if verbose:
            print(f"  wrote {out}", flush=True)
    return res


def names_pos(concepts, ci):
    return concepts[ci]["pos"]


def selftest():
    torch.set_num_threads(2)
    r = run(steps=300, write_steps=300, seed=0, per_pos=40, d=64, device="cpu", verbose=False, n_show=0)
    assert 0.0 <= r["comp_kernel"] <= 1.0 and r["n"] > 0, r
    assert r["wrote_frac"] >= 0.0, r
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
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    run(args.steps, seed=args.seed, per_pos=args.per_pos, d=args.d, device=device, batch=args.batch,
        write_steps=args.write_steps, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
