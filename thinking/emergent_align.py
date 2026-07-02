#!/usr/bin/env python3
"""emergent_align -- align the agent's DISCOVERED grounded code to ENGLISH (the deferred bridge).

The agent invented a grounded, provable code ([[emergent-language]] / emergent_grounded.py): a message
encodes an entity's brain-derived fact-set. Now we connect that self-invented language to human words.

Both the emergent MESSAGE and an English DESCRIPTION ground to the SAME brain fact-set, so we learn a
translator  message-symbols -> English-words  and GRADE it by the brain: parse the produced English
back to facts and check each is DERIVABLE for the entity (kernel/datalog-verified). Held out on
never-seen ENTITIES => zero-shot alignment (translate a novel concept's invented message to faithful
English by composing).

  python -m thinking.emergent_align --selftest
  python -m thinking.emergent_align --steps 3000
"""
import argparse
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thinking.prover as P  # noqa: E402  (build_kb + seq2seq: Vocab/build_model/train/emit)
import thinking.emergent_grounded as EG  # noqa: E402
from thinking.emergent import Speaker  # noqa: E402
from thinking.emergent_grounded import ReconListener  # noqa: E402
import torch.nn as nn  # noqa: E402


# ---- English describer / parser over the brain fact-set (deterministic, compositional template) ----
def english(kb, atoms, e):
    toks = []
    for (pred, obj) in atoms:
        if (pred, (e, obj)) in kb.known:
            toks += (["is", "a", obj, "."] if pred == "isa" else ["can", obj, "."])
    return toks


def parse_english(toks):
    facts, i = set(), 0
    while i < len(toks):
        if toks[i] == "is" and i + 2 < len(toks) and toks[i + 1] == "a":
            facts.add(("isa", toks[i + 2])); i += 4
        elif toks[i] == "can" and i + 1 < len(toks):
            facts.add(("prop", toks[i + 1])); i += 3
        else:
            i += 1
    return facts


@torch.no_grad()
def message_tokens(spk, vec):
    spk.eval()
    m = spk(torch.from_numpy(vec)[None])[0].argmax(-1)[0].tolist()
    return [f"s{j}" for j in m]


def true_factset(kb, atoms, e):
    return {(pred, obj) for (pred, obj) in atoms if (pred, (e, obj)) in kb.known}


# ---- build the grounded speaker (so it invents the code), then the translation dataset ----
def build(steps_spk, L, V, d, seed):
    kb = P.build_kb()
    atoms = EG.fact_universe(kb)
    ents = EG.entities_with_facts(kb, atoms)
    tr_e, te_e = EG.split_entities(ents, 0.25, seed)
    spk = Speaker(len(atoms), 1, L, V, d=d)
    spk.enc[0] = nn.Linear(len(atoms), d)
    lis = ReconListener(L, V, len(atoms), d=d)
    EG.train(spk, lis, [EG.fact_vec(kb, atoms, e) for e in tr_e], steps=steps_spk, batch=64, seed=seed)
    def pair(e):
        return {"gtoks": message_tokens(spk, EG.fact_vec(kb, atoms, e)), "ptoks": english(kb, atoms, e)}
    return kb, atoms, spk, tr_e, te_e, [pair(e) for e in tr_e], [pair(e) for e in te_e]


def evaluate(model, vocab, kb, atoms, ents, pairs, device, block):
    """Translate each entity's invented message -> English; BRAIN-verify the English's facts."""
    faith_ok = faith_tot = rec_ok = rec_tot = exact = 0
    for e, pr in zip(ents, pairs):
        out = P.emit(model, vocab, pr, device, block)
        got = parse_english(out)
        true = true_factset(kb, atoms, e)
        for f in got:
            faith_tot += 1; faith_ok += int((f[0], (e, f[1])) in kb.known)   # brain confirms
        rec_ok += len(got & true); rec_tot += len(true)
        exact += int(got == true)
    n = max(1, len(ents))
    return dict(n=len(ents), exact=exact / n, faithfulness=faith_ok / max(1, faith_tot),
                recall=rec_ok / max(1, rec_tot))


def selftest():
    torch.set_num_threads(2)
    kb, atoms, spk, tr_e, te_e, tr_p, te_p = build(steps_spk=300, L=4, V=8, d=64, seed=0)
    assert tr_p and te_p and all(p["ptoks"] for p in tr_p)
    assert parse_english(["is", "a", "bird", ".", "can", "move", "."]) == {("isa", "bird"), ("prop", "move")}
    vocab = P.Vocab.build(tr_p + te_p)
    m = P.build_model(len(vocab), 80, d=64, layers=2, heads=4)
    P.train(m, vocab, tr_p, steps=300, device="cpu", batch=16, block=80, seed=0)
    r = evaluate(m, vocab, kb, atoms, tr_e, tr_p, "cpu", 80)
    assert r["recall"] > 0.3, ("translation should recover some facts", r)
    print("emergent_align selftest OK")


def run(steps, seed=0, L=6, V=12, d=128, steps_spk=4000, block=96, verbose=True):
    kb, atoms, spk, tr_e, te_e, tr_p, te_p = build(steps_spk, L, V, d, seed)
    vocab = P.Vocab.build(tr_p + te_p)
    if verbose:
        print(f"entities: train {len(tr_e)} / held-out {len(te_e)} | emergent msg L={L} V={V} | "
              f"translate symbols->English | vocab {len(vocab)} | steps {steps}", flush=True)
    m = P.build_model(len(vocab), block, d=d, layers=4, heads=8)
    P.train(m, vocab, tr_p, steps=steps, device="cpu", batch=32, block=block, seed=seed,
            log_every=(max(1, steps // 6) if verbose else 0))
    rtr = evaluate(m, vocab, kb, atoms, tr_e, tr_p, "cpu", block)
    rte = evaluate(m, vocab, kb, atoms, te_e, te_p, "cpu", block)
    if verbose:
        print("\n== ALIGNMENT: invented code -> English (brain-verified) ==")
        print(f"  TRAIN    : exact-factset {rtr['exact']:.3f} | brain-verified faithfulness "
              f"{rtr['faithfulness']:.3f} | recall {rtr['recall']:.3f}")
        print(f"  HELD-OUT : exact-factset {rte['exact']:.3f} | brain-verified faithfulness "
              f"{rte['faithfulness']:.3f} | recall {rte['recall']:.3f}   (never-seen entities)")
        # show a couple of translations
        for e, pr in list(zip(te_e, te_p))[:3]:
            out = P.emit(m, vocab, pr, "cpu", block)
            print(f"   [{e}]  {' '.join(pr['gtoks'])}  ->  {' '.join(out)}")
        print("  faithfulness is GROUND TRUTH: the English's facts are checked against the brain's closure.")
    return dict(train=rtr, heldout=rte)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--msg-len", type=int, default=6)
    ap.add_argument("--vocab", type=int, default=12)
    ap.add_argument("--d", type=int, default=128)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    run(args.steps, seed=args.seed, L=args.msg_len, V=args.vocab, d=args.d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
