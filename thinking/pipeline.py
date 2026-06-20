#!/usr/bin/env python3
"""pipeline -- the capstone: natural-language QUESTION -> verified-reasoning ANSWER.

    English question  --parse (learned)-->  LOTA goal  --neural-guided proof SEARCH-->
                       --kernel CHECK (trusted)-->  answer (yes + proof | not proven)

Ties together everything built this session:
  * a learned NL->LOTA parser (compositional; held-out = unseen entity combinations),
  * the kernel-guided proof search (thinking/prover.py) with the neural branch-ranker,
  * the SOUND kernel (thinking/kernel.py) as the final gate.

Because the prover is sound (kernel-checked) AND complete for this KB, a CORRECTLY PARSED question is
answered with a GUARANTEED-correct verdict -- a 'yes' always carries a kernel-checked proof, and the
prover never asserts a false 'yes'. So end-to-end correctness reduces to parse correctness, with the
reasoning step carrying a machine-checkable certificate. This is "understand language -> reason with
proof", in-house and dependency-free.

  python -m thinking.pipeline --selftest
  python -m thinking.pipeline --steps 4000 --demo
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thinking.kernel as K  # noqa: E402
import thinking.lota_kernel as LK  # noqa: E402
import thinking.prover as P  # noqa: E402  (KB, seq2seq machinery, KernelProver, NeuralRanker)


def _art(w):
    return "an" if w and w[0] in "aeiou" else "a"


def gen_questions(kb):
    """Templated English questions over the KB ontology, paired with their LOTA goal. Truth is NOT
    encoded -- the prover decides it. isa: 'is a robin a bird ?' ; prop: 'can a robin fly ?'."""
    facts = list(kb.facts)
    subjects = sorted({e[0] for (p, e) in facts})
    isa_objs = sorted({e[1] for (p, e) in facts if p == "isa"})
    prop_vals = sorted({e[1] for (p, e) in facts if p == "prop"})
    pairs = []
    for x in subjects:
        for y in isa_objs:
            if x == y:
                continue
            nl = f"is {_art(x)} {x} {_art(y)} {y} ?"
            pairs.append({"gtoks": nl.split(), "ptoks": P.goal_tokens(("isa", (x, y))),
                          "goal": ("isa", (x, y))})
        for pv in prop_vals:
            nl = f"can {_art(x)} {x} {pv} ?"
            pairs.append({"gtoks": nl.split(), "ptoks": P.goal_tokens(("prop", (x, pv))),
                          "goal": ("prop", (x, pv))})
    return pairs


def split(pairs, held_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs)); rng.shuffle(idx)
    cut = int(len(pairs) * (1 - held_frac))
    return [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]


def parse_goal(toks):
    if len(toks) >= 4 and toks[0] == "(" and toks[-1] == ")":
        inner = toks[1:-1]
        return (inner[0], tuple(inner[1:]))
    raise ValueError("not a goal")


def answer(model, vocab, prover, kb, pair, device, block):
    """Full pipeline on one question: parse -> prove -> verdict. Returns a dict."""
    gtoks = P.emit(model, vocab, pair, device, block)
    try:
        goal = parse_goal(gtoks)
    except Exception:
        return {"parse_ok": False, "verdict": None, "goal": None}
    parse_ok = (goal == pair["goal"])
    if goal[0] not in kb.preds or any(a not in kb.env for a in goal[1]):
        return {"parse_ok": parse_ok, "verdict": "unknown-symbol", "goal": goal}
    ok, term, _ = prover.verify(goal)
    return {"parse_ok": parse_ok, "verdict": "yes" if ok else "no", "goal": goal,
            "proof": K.show(term) if term is not None else None}


def evaluate(model, vocab, prover, kb, pairs, device, block):
    parse_correct = end_to_end = 0
    samples = []
    for pair in pairs:
        a = answer(model, vocab, prover, kb, pair, device, block)
        truth = pair["goal"] in kb.known                          # ground truth from the KB
        verdict_true = (a["verdict"] == "yes")
        correct = a["parse_ok"] and (verdict_true == truth)       # right parse AND right answer
        parse_correct += int(a["parse_ok"]); end_to_end += int(correct)
        samples.append((pair, a, truth, correct))
    n = max(1, len(pairs))
    return {"n": len(pairs), "parse_acc": parse_correct / n, "end_to_end_acc": end_to_end / n}, samples


def run(steps, seed=0, d=256, layers=4, heads=8, block=64, device="cpu", demo=False, verbose=True):
    kb = P.build_kb()
    pairs = gen_questions(kb)
    tr, te = split(pairs, 0.2, seed)
    vocab = P.Vocab.build(pairs)
    if verbose:
        print(f"questions: {len(pairs)} (train {len(tr)} / held-out {len(te)}) | vocab {len(vocab)} "
              f"| device {device} | steps {steps}", flush=True)
    model = P.build_model(len(vocab), block, d=d, layers=layers, heads=heads)
    P.train(model, vocab, tr, steps=steps, device=device, batch=32, block=block, seed=seed,
            log_every=(max(1, steps // 6) if verbose else 0))
    ranker = P.train_ranker(kb, steps=1500, seed=seed)
    prover = P.KernelProver(kb, order=ranker.order_fn(list(kb.entities)))
    res, samples = evaluate(model, vocab, prover, kb, te, device, block)
    if verbose:
        print(f"\n== END-TO-END: English question -> kernel-verified answer (held-out) ==")
        print(f"  parse accuracy     {res['parse_acc']:.3f}   (NL -> correct LOTA goal)")
        print(f"  end-to-end accuracy {res['end_to_end_acc']:.3f}   (right answer, reasoning "
              f"kernel-certified)  (n={res['n']})")
        if demo:
            print("\n  examples (held-out):")
            shown = 0
            for pair, a, truth, correct in samples:
                if shown >= 6:
                    break
                q = " ".join(pair["gtoks"])
                v = a["verdict"]
                mark = "OK " if correct else "xx "
                print(f"   [{mark}] {q!r} -> goal {a['goal']} -> {v!s:>3} (truth={truth})")
                if v == "yes" and a.get("proof"):
                    print(f"            proof: {a['proof'][:80]}")
                shown += 1
    return res


def selftest():
    kb = P.build_kb()
    pairs = gen_questions(kb)
    assert len(pairs) > 100
    # goal serialization round-trips through parse_goal
    g = ("isa", ("robin", "bird"))
    assert parse_goal(P.goal_tokens(g)) == g
    tr, te = split(pairs, 0.25, 0)
    vocab = P.Vocab.build(pairs)
    torch.set_num_threads(2)
    model = P.build_model(len(vocab), 64, d=64, layers=2, heads=4)
    P.train(model, vocab, tr, steps=200, device="cpu", batch=16, block=64, seed=0)
    ranker = P.train_ranker(kb, steps=150)
    prover = P.KernelProver(kb, order=ranker.order_fn(list(kb.entities)))
    res, _ = evaluate(model, vocab, prover, kb, te[:10], "cpu", 64)
    assert 0.0 <= res["end_to_end_acc"] <= 1.0
    # the reasoning step is sound regardless of the parser: a true goal proves, a false one does not
    okt, _, _ = prover.verify(("isa", ("robin", "animal")))
    okf, _, _ = prover.verify(("isa", ("robin", "fish")))
    assert okt and not okf, "kernel reasoning must be sound+complete behind the parser"
    print("pipeline selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return 0
    run(args.steps, seed=args.seed, d=args.d, layers=args.layers, heads=args.heads,
        block=args.block, device=args.device, demo=args.demo)
    return 0


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    raise SystemExit(main())
