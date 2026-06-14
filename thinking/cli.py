"""CLI entry points.

  python -m thinking.cli selftest                          # correctness core, no GPU, <1s
  python -m thinking.cli train --out runs/x [--sup steps|path] [--arch ...] [--seed N] [--neg]
  python -m thinking.cli eval runs/x [--mode verified|free|path] [--hops 2,6,10]
  python -m thinking.cli demo runs/x [--k 6]
  python -m thinking.cli sweep --out runs/grid [--seeds 0,1,2] [--archs standard,relational]
"""
import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("thinking")


def cmd_selftest(_args):
    import numpy as np
    from .world import ChainWorld, RULES, ENGINE, entity_pools
    from .trace import (render_example, parse_line, build_vocab, pack_batch,
                        build_rule_vocab, pack_batch_with_meta, render_goal_line)
    from .verify import StepChecker

    # entity pools: disjoint + deterministic
    a1, b1 = entity_pools(50, 10, seed=3)
    a2, b2 = entity_pools(50, 10, seed=3)
    assert a1 == a2 and b1 == b2 and not set(a1) & set(b1)

    # world + oracle + gold trace
    w = ChainWorld(a1, seed=0)
    p = w.sample(4)
    steps = w.trace_steps(p)
    assert len(steps) == 4 and steps[-1][0] == p.goal

    # checker accepts the gold trace, line by line, and the verified frontier reaches the answer
    chk = StepChecker(RULES)
    known = set(p.edb)
    for head, body in steps:
        assert chk.valid_step(head, body, known), f"gold step rejected: {head} <- {body}"
        known.add(head)
    assert chk.valid_answer(p.goal[0], p.head, p.answer, known, set(p.edb))
    assert not chk.valid_answer(p.goal[0], p.head, p.head, known, set(p.edb))  # premature stop

    # checker rejects corruptions: unsupported body, non-rule shape, unknown derived fact
    h, b = steps[-1]
    assert not chk.valid_step(h, (("r", ("zz", "yy")),), set(p.edb))
    assert not chk.valid_step(("far", ("a", "a")), (("far", ("a", "a")),), known)

    # render/parse roundtrip on every think line
    ex = render_example(p, steps, "steps")
    toks, i = ex.tokens, 0
    nthink = 0
    while i < len(toks):
        if toks[i] == "think":
            j = toks.index(".", i)
            ps = parse_line(toks[i:j])
            assert ps is not None, f"gold line unparseable: {toks[i:j]}"
            nthink += 1
            i = j
        i += 1
    assert nthink == len(steps)
    assert parse_line(["think", "a", "far", "b"]) is None          # malformed -> None
    assert parse_line(["junk", "a", "far", "b", "so", "a", "far", "b"]) is None

    # aux alignment: token AFTER each predictor position == token at the gold context position
    for pred_pos, gold_pos in ex.aux:
        assert toks[pred_pos + 1] == toks[gold_pos], (pred_pos, gold_pos)
    exp = render_example(p, steps, "path")
    for pred_pos, gold_pos in exp.aux:
        assert exp.tokens[pred_pos + 1] == exp.tokens[gold_pos]

    # packing: rows are example-aligned, aux survives the offset shift
    vocab = build_vocab([ex, exp])
    seqs = [(vocab.enc(e.tokens), e.aux) for e in (ex, exp)]
    rng = np.random.default_rng(0)
    x, sup = pack_batch(seqs, 128, 4, vocab.pad, rng)
    assert x.shape == (4, 129)
    for r, pp, cc in sup:
        assert x[r, pp + 1] == x[r, cc], "aux misaligned after packing"

    # cached inference must be numerically equivalent to full-prefix inference
    import torch
    from scratchpad_model import ScratchpadLM
    torch.manual_seed(0)
    m = ScratchpadLM(len(vocab), d=32, layers=2, heads=4, max_len=64, pad=vocab.pad,
                     pointer=True).eval()
    ids = torch.tensor([vocab.enc(ex.tokens[:20])])
    with torch.no_grad():
        full = m(ids)
        cache, outs = None, []
        for j in range(ids.shape[1]):
            logits, cache = m.forward_step(ids[:, j:j + 1], cache)
            outs.append(logits)
        inc = torch.stack(outs, 1)
    assert torch.allclose(full, inc, atol=1e-5), "cached decode diverges from full forward"
    torch.manual_seed(0)
    mp = ScratchpadLM(len(vocab), d=32, layers=2, heads=4, max_len=64, pad=vocab.pad,
                      pointer=False).eval()
    prefix = torch.randn(1, 3, 32)
    with torch.no_grad():
        pref = mp(ids, prefix=prefix)
    assert pref.shape == (1, ids.shape[1] + 3, len(vocab)), "continuous prefix shape mismatch"
    try:
        m(ids, prefix=prefix)
        raise AssertionError("pointer model accepted continuous prefix")
    except ValueError:
        pass

    # backward chaining agrees with the forward oracle
    edb = set(p.edb)
    fwd = {f[1][1] for f in ENGINE.closure(edb)[0] if f[0] == "far" and f[1][0] == p.head}
    bwd = {g[1][1] for g in ENGINE.prove(edb, ("far", (p.head, "?x")))}
    assert fwd == bwd, (fwd, bwd)

    # ---- kinship: nested proofs, goal-directed traces, agenda checker -------------------------
    from .kinship import (FamilyWorld, RULES as KR, ANSWER_PREDS, TEMPLATES, QUESTION, name_pools)
    from .trace import render_goal_example, parse_goal_line
    from .verify import GoalChecker
    tr, te = name_pools(90, 30, seed=1)
    assert not set(tr) & set(te)
    fw = FamilyWorld(tr, seed=0)
    for depth in (2, 3):
        kp, lines = fw.sample(depth)
        assert lines[0][0] == "check", "evidence-first: traces must open with grounding"
        assert lines[-1][0] == "think" and lines[-1][1] == kp.goal, "conclusion must come LAST"
        # forward checker accepts the gold trace and the gold answer
        chk = GoalChecker(KR, ANSWER_PREDS)
        st = chk.new_state(kp.goal[1], kp.edb, goal_pred=kp.goal[0])
        for typ, head, body in lines:
            assert chk.step(st, typ, head, body), f"gold line rejected: {typ} {head} {body}"
        assert chk.valid_answer(st, kp.answer)
        assert not chk.valid_answer(st, "cousin" if kp.answer != "cousin" else "aunt")
        # rejects: checking a non-fact, deriving from unknown bodies, answering with no derivation
        st2 = chk.new_state(kp.goal[1], kp.edb, goal_pred=kp.goal[0])
        assert not chk.step(st2, "check", kp.goal, ())              # the goal is not an EDB fact
        first_think = next(ln for ln in lines if ln[0] == "think")
        assert not chk.step(st2, *first_think), "think before check must not see oracle facts"
        assert not chk.step(st2, *lines[-1])                        # conclusion before its premises
        assert not chk.valid_answer(st2, kp.answer)                 # nothing derived yet
        # render/parse roundtrip incl. aux alignment under the NL surface
        ex = render_goal_example(kp, lines, TEMPLATES, QUESTION)
        i, nlines = 0, 0
        while i < len(ex.tokens):
            if ex.tokens[i] in ("think", "check"):
                j = ex.tokens.index(".", i)
                assert parse_goal_line(ex.tokens[i:j]) is not None, ex.tokens[i:j]
                nlines += 1
                i = j
            i += 1
        assert nlines == len(lines)
        for pp, cc in ex.aux:
            assert ex.tokens[pp + 1] == ex.tokens[cc], (pp, cc)
    # composition holdout: include/exclude filters work
    kp, _ = fw.sample(3, include=("great_grandmother", "great_grandfather"))
    assert kp.answer.startswith("great_")

    # NOVEL relations: rule exists only IN THE QUESTION; the answer is a linked noun
    import numpy as _np0
    from .kinship import AGE_BUILTINS
    for train_flag in (True, False):
        np_, nl = fw.sample_novel(_np0.random.default_rng(17), train=train_flag)
        chain = tuple(b[0] for b in np_.extra_rules[0][1])
        assert (chain in FamilyWorld.NOVEL_HOLDOUT) != train_flag, "novel split leak"
        chk = GoalChecker(KR, ANSWER_PREDS, builtins=AGE_BUILTINS)
        st = chk.new_state(np_.goal[1], np_.edb, goal_pred=np_.goal[0],
                           extra_rules=np_.extra_rules)
        for ln in nl:
            assert chk.step(st, *ln), f"novel gold line rejected: {ln}"
        assert chk.valid_answer(st, np_.answer)
        assert not chk.valid_answer(st, np_.head), "wrong noun accepted"
        ex = render_goal_example(np_, nl, TEMPLATES, np_.question)
        assert np_.goal[0] in ex.tokens, "novel name missing from surface"

    # DEEP world: every query type's constructed gold trace is (a) GoalChecker-valid line by
    # line with the agenda fully discharged, and (b) ENGINE-entailed (small instances, depth 5)
    from .kinship import ENGINE as KENGINE, AGE_BUILTINS, VALUE_PREDS
    import numpy as _np
    for q in FamilyWorld.DEEP_QTYPES:
        kp, lines = fw.sample_deep(5, _np.random.default_rng(11), include=(q,))
        assert kp.answer == q or q in VALUE_PREDS          # value queries answer with a NUMBER
        chk = GoalChecker(KR, ANSWER_PREDS, builtins=AGE_BUILTINS)
        st = chk.new_state(kp.goal[1], kp.edb, goal_pred=kp.goal[0])
        for ln in lines:
            assert chk.step(st, *ln), f"deep {q}: gold line rejected: {ln}"
        assert chk.valid_answer(st, kp.answer), f"deep {q}: gold answer rejected"
        assert not chk.valid_answer(st, "9999"), f"deep {q}: wrong answer accepted"
        if q not in VALUE_PREDS:                           # engine has no arithmetic builtins
            assert KENGINE.entails(set(kp.edb), kp.goal), f"deep {q}: engine disagrees"
            # the queried link is RULE-ONLY: no EDB fact directly connects the pair
            assert not any(frozenset(f[1]) == frozenset(kp.goal[1]) for f in kp.edb), \
                f"deep {q}: queried pair directly linked by a stated fact"
        ex = render_goal_example(kp, lines, TEMPLATES, QUESTION)
        for pp, cc in ex.aux:
            assert ex.tokens[pp + 1] == ex.tokens[cc]
        # synonym/inverse surface variants: aux alignment must survive ANY variant choice
        for vseed in (1, 2, 3):
            exv = render_goal_example(kp, lines, TEMPLATES, QUESTION, _np.random.default_rng(vseed))
            for pp, cc in exv.aux:
                assert exv.tokens[pp + 1] == exv.tokens[cc], f"variant aux misaligned ({q})"
    # surface bank (when present): EVERY level must cover EVERY queryable predicate's questions
    from .kinship import surfaces as _surf, bank_levels as _bl
    for _lv in (_bl() or ["mix"]):
        for _sp in ("train", "eval"):
            _t, _q = _surf(_lv, _sp)
            _cov = set()
            for _, _ps in _q:
                _cov |= set(_ps) if isinstance(_ps, tuple) else {_ps}
            for _pred in ANSWER_PREDS + VALUE_PREDS:
                assert _pred in _cov, f"bank gap: {_lv}/{_sp} lacks questions for {_pred}"
            for _pred in ("mother", "father", "born", "died"):
                assert _t.get(_pred), f"bank gap: {_lv}/{_sp} lacks templates for {_pred}"

    # PHASE-1 extraction: render -> parse roundtrip recovers the gold EDB exactly, aux aligned
    from .trace import render_extraction_example, parse_fact_line
    kp, _l = fw.sample(3, _np.random.default_rng(21))
    exx = render_extraction_example(kp, TEMPLATES, QUESTION, _np.random.default_rng(3))
    got, i = set(), exx.tokens.index("extract") + 1
    while exx.tokens[i] != "done":
        j = exx.tokens.index(".", i)
        f = parse_fact_line(exx.tokens[i:j])
        assert f is not None, exx.tokens[i:j]
        got.add(f)
        i = j + 1
    assert got == set(kp.edb), "extraction roundtrip mismatch"
    for pp, cc in exx.aux:
        assert exx.tokens[pp + 1] == exx.tokens[cc]

    # deep ancestor at depth 30: checker-valid + fits the scaling assumptions
    train_names, test_names = name_pools(1500, 3216, seed=2)
    assert len(train_names) == 1500 and len(test_names) == 3216
    assert len(set(train_names) | set(test_names)) == 4716
    big = FamilyWorld(name_pools(120, 80, seed=2)[0], seed=0)
    kp, lines = big.sample_deep(30, _np.random.default_rng(5), include=("ancestor",))
    chk = GoalChecker(KR, ANSWER_PREDS, builtins=AGE_BUILTINS)
    st = chk.new_state(kp.goal[1], kp.edb, goal_pred="ancestor")
    for ln in lines:
        assert chk.step(st, *ln)
    assert chk.valid_answer(st, "ancestor")
    anc = [ln for ln in lines if ln[0] == "think" and ln[1][0] == "ancestor"]
    assert len(anc) == 30 and anc[-1][1] == kp.goal, "ancestor trace must reach target linearly"
    assert all(ln[1][1][0] == kp.goal[1][0] for ln in anc), \
        "ancestor trace must keep the question head as the forward frontier anchor"
    assert anc[0][2][0][0] == "parent" and all(ln[2][0][0] == "ancestor" and
                                                ln[2][1][0] == "parent" for ln in anc[1:]), \
        "ancestor trace must be forward-recursive: ancestor(x,y) + parent(y,z)"
    ex = render_goal_example(kp, lines, TEMPLATES, QUESTION, _np.random.default_rng(9))
    assert len(ex.tokens) <= 96 * 30, f"depth-30 example {len(ex.tokens)} tokens > block budget"
    meta = ex.meta.get("lines", [])
    proof_meta = [m for m in meta if m["kind"] != "answer"]
    assert len(proof_meta) == len(lines), "trace metadata must cover every proof line"
    assert [m["rule_id"] for m in proof_meta if m["rule_id"].startswith("ancestor")] == \
        ["ancestor_base"] + ["ancestor_forward"] * 29
    assert [m["depth_index"] for m in proof_meta if m["rule_id"] == "ancestor_forward"] == \
        list(range(1, 30)), "ancestor_forward metadata must preserve step index"
    rv = build_rule_vocab([ex])
    assert {"ancestor_base", "ancestor_forward"} <= set(rv)
    mv = build_vocab([ex])
    xb, _, rt, spans = pack_batch_with_meta(
        [(mv.enc(ex.tokens), ex.aux, ex.meta)], len(ex.tokens), 1, mv.pad,
        _np.random.default_rng(0), rv)
    assert xb.shape == (1, len(ex.tokens) + 1)
    assert rt.ne(-100).any() and spans, "rule metadata must survive packing"
    stc = chk.new_state(kp.goal[1], kp.edb, goal_pred="ancestor")
    support = chk.support_atoms(stc)
    assert support is not None and kp.goal in support, "frontier support must include the goal"
    assert stc["support_plan"] and len(stc["support_plan"]) <= len(lines), \
        "frontier should cache a compact executable proof plan"
    checks0 = [ln for ln in chk.candidate_steps(stc) if ln[0] == "check"]
    assert 0 < len(checks0) <= 30, "frontier should prune deep distractor facts"
    assert all(ln[1] in support for ln in checks0), "frontier emitted unsupported checks"
    for ln in lines[:8]:
        assert ln in chk.candidate_steps(stc, goal_pruned=False), \
            f"generic candidates missed gold line {ln}"
        assert render_goal_line(*ln)                       # renderer owns line grammar
        assert chk.step(stc, *ln)
    std = chk.new_state(kp.goal[1], kp.edb, goal_pred="ancestor")
    for _ in range(len(lines) + 8):
        if chk.valid_answer(std, "ancestor"):
            break
        cands = chk.candidate_steps(std)
        assert cands, "frontier ran out before the answer"
        assert chk.step(std, *cands[0])
    assert chk.valid_answer(std, "ancestor"), "frontier proof did not reach the answer"
    # chronology is consistent: every child born after its parent, death after birth
    born = {a[1][0]: int(a[1][1]) for a in kp.edb if a[0] == "born"}
    died = {a[1][0]: int(a[1][1]) for a in kp.edb if a[0] == "died"}
    for pred, (h, t) in kp.edb:
        if pred in ("mother", "father"):
            assert born[h] < born[t], "child born before parent"
    assert all(died[p] > born[p] for p in died), "death before birth"

    # ---- MASKED decoding: eager validity pruning at the token level ------------------------
    # 1) completeness: every gold line is in the validity-only candidate set at its state
    from .trace import render_goal_line
    for depth in (2, 3):
        kp, lines = fw.sample(depth)
        chk = GoalChecker(KR, ANSWER_PREDS)
        st = chk.new_state(kp.goal[1], kp.edb, goal_pred=kp.goal[0])
        for typ, head, body in lines:
            cands = chk.candidate_steps(st, goal_pruned=False, relevance_pruned=False)
            assert (typ, head, tuple(body)) in cands, f"gold step not in candidates: {typ} {head}"
            assert chk.step(st, typ, head, body)
        assert kp.answer in chk.answer_candidates(st)
    # 2) rejection taxonomy: causes are recorded
    st2 = chk.new_state(kp.goal[1], kp.edb, goal_pred=kp.goal[0])
    assert not chk.step(st2, "check", kp.goal, ())
    assert st2["why"] == "check-not-edb", st2["why"]
    # 3) masked decode with an UNTRAINED model: every line checker-valid by construction
    from .config import Config as _Cfg
    from .flow import FlowRuntime
    from .train import Trainer as _Tr
    _cfg = _Cfg(world="kinship", train_hops=(2, 3), block=2048, anonymize=False)
    _tr = _Tr(_cfg)
    kp2, _ = _tr.world.sample(2, _np0.random.default_rng(5))
    exs = _tr.build_examples(2, _np0.random.default_rng(5))
    from .trace import Vocab
    toks = [t for e in exs for t in e.tokens]
    toks += [tok for f in kp2.edb for tok in (f[1][0], f[0], f[1][1])]
    toks += [w for v, _ps in QUESTION for w in v if not w.startswith("{")]
    toks += list(ANSWER_PREDS)
    vv = Vocab(toks)
    torch.manual_seed(0)
    mm = ScratchpadLM(len(vv), d=32, layers=2, heads=4, max_len=2048, pad=vv.pad,
                      pointer=True).eval()
    rt = FlowRuntime(mm, vv, GoalChecker(KR, ANSWER_PREDS, builtins=AGE_BUILTINS), _cfg, "cpu")
    rr = rt.run_goal(kp2, TEMPLATES, QUESTION, rng=_np0.random.default_rng(7),
                     decode="masked-full")
    assert rr.n_invalid == 0 and rr.n_resampled == 0, "pure masking must never reject"
    assert all(s in ("ok", "answer", "repair") for _, s in rr.lines), rr.lines
    # hybrid: model proposes freely; a rejected line is REPLACED by one masked line (no temp
    # resampling, nothing rejected enters the context)
    rr = rt.run_goal(kp2, TEMPLATES, QUESTION, rng=_np0.random.default_rng(7), decode="masked")
    assert rr.n_resampled == 0, "masked repair replaces the resample loop"
    assert all(s in ("ok", "masked", "answer", "repair") for _, s in rr.lines), rr.lines
    # ranked action decode: the model may choose irrelevant legal actions, but syntax and local
    # verifier validity are deterministic, and oracle candidate ranks are recorded.
    rr = rt.run_goal(kp2, TEMPLATES, QUESTION, rng=_np0.random.default_rng(7), decode="ranker")
    assert rr.n_resampled == 0, "ranker must not use text resampling"
    assert rr.rank_positions, "ranker should record oracle next-step ranks"
    assert all(s in ("ranked", "answer", "repair") for _, s in rr.lines), rr.lines

    # ---- LANG-0: hybrid byte-BPE vocab keeps the canonical layer atomic --------------------
    try:
        from model_io import BPEVocab
        from .hybrid_vocab import canonical_specials
    except Exception:
        BPEVocab = None
    if BPEVocab is not None:
        specials = canonical_specials()
        corpus = ("the cat sat on the warm mat near the door . a small boy ran to the hill "
                  "and laughed with his mother every bright morning by the river .") * 30
        Vb = BPEVocab(texts=[corpus], vocab_size=1500, specials=specials)
        frag = [s for s in specials
                if len(Vb.enc(" " + s)) != 1 and len(Vb.enc(s)) != 1]
        assert not frag, f"canonical tokens fragmented under BPE: {frag[:5]}"
        from .world import anonymize
        kp3, l3 = fw.sample(3)
        kp3, l3 = anonymize(kp3, l3, _np0.random.default_rng(11))
        ex3 = render_goal_example(kp3, l3, TEMPLATES, QUESTION, _np0.random.default_rng(11))
        trace_toks = ex3.tokens[ex3.tokens.index("check"):]
        back = [Vb.tk.id_to_token(Vb.enc(" " + t)[0] if len(Vb.enc(" " + t)) == 1
                                  else Vb.enc(t)[0]).lstrip("Ġ") for t in trace_toks]
        assert back == trace_toks, "trace must round-trip token-for-token under hybrid vocab"

    # FER/UFR probe scoring: reusable same-rule vectors should score better than fractured ones.
    from .probes import pairwise_probe_metrics, score_probe_metrics
    good = [
        {"rule_id": "r1", "problem_depth": 4, "depth_index": 0,
         "vector": _np0.array([1.0, 0.0, 0.0])},
        {"rule_id": "r1", "problem_depth": 8, "depth_index": 1,
         "vector": _np0.array([0.98, 0.05, 0.0])},
        {"rule_id": "r2", "problem_depth": 4, "depth_index": 0,
         "vector": _np0.array([0.0, 1.0, 0.0])},
        {"rule_id": "r2", "problem_depth": 8, "depth_index": 1,
         "vector": _np0.array([0.0, 0.98, 0.05])},
    ]
    gm = pairwise_probe_metrics(good)
    gs = score_probe_metrics(gm, n_vectors=len(good))
    assert gm["rule_reuse_margin"] > 0.8
    assert gs["verdict"] in ("low_fer_risk", "medium_fer_risk")
    bad = [
        {"rule_id": "r1", "problem_depth": 4, "depth_index": 0,
         "vector": _np0.array([1.0, 0.0])},
        {"rule_id": "r1", "problem_depth": 8, "depth_index": 1,
         "vector": _np0.array([0.0, 1.0])},
        {"rule_id": "r2", "problem_depth": 4, "depth_index": 0,
         "vector": _np0.array([0.9, 0.1])},
        {"rule_id": "r2", "problem_depth": 8, "depth_index": 1,
         "vector": _np0.array([0.1, 0.9])},
    ]
    bm = pairwise_probe_metrics(bad)
    bs = score_probe_metrics(bm, n_vectors=len(bad))
    assert bm["rule_reuse_margin"] < 0.1
    assert "low_same_vs_different_rule_margin" in bs["risk_flags"]

    # Vision understanding learns schema-free visual concepts from manifest images.
    from .vision_understanding import selftest as vision_understanding_selftest
    vision_understanding_selftest()
    from .image_data import selftest as image_data_selftest
    image_data_selftest()
    from .image_embed import selftest as image_embed_selftest
    image_embed_selftest()
    from .image_curate import selftest as image_curate_selftest
    image_curate_selftest()
    from .image_latent import selftest as image_latent_selftest
    image_latent_selftest()
    from .audio import selftest as audio_selftest
    audio_selftest()
    from .text import selftest as text_selftest
    text_selftest()
    from .multimodal import selftest as multimodal_selftest
    multimodal_selftest()
    print("selftest OK")


def cmd_induce(args):
    """The agent DISCOVERS the rules from raw (facts, question, answer) observations --
    generate-test over a typed hypothesis space, scored on held-out questions."""
    import numpy as np
    from .kinship import FamilyWorld, name_pools
    from .induce import gather_observations, induce, held_out_accuracy, save_rules
    rng = np.random.default_rng(args.seed)
    tr, te = name_pools(300, 120, args.seed)
    obs = gather_observations(FamilyWorld(tr, seed=args.seed), args.n_rel, args.n_val, rng)
    learned, report = induce(obs)
    print(f"observed {len(obs)} (facts, question, answer) triples -- no rules, no traces\n")
    for k in sorted(report):
        print(f"  {k}: {report[k]}")
    obs2 = gather_observations(FamilyWorld(te, seed=args.seed + 1), 80, 60,
                               np.random.default_rng(args.seed + 9))
    acc = held_out_accuracy(learned, obs2)
    print(f"\nheld-out accuracy of the INDUCED theory: {acc:.2f} (n={len(obs2)})")
    save_rules(learned, args.out)


def cmd_ablate(args):
    """Loop-regression ablation: identical data/objectives across recursion settings.
    Isolates the mHC write-gate cold-start (zeros-init bug, fixed) from recursion itself."""
    from .config import Config
    from .train import Trainer
    for tag, loop, loops, mhc in (("loop=False", False, 8, True),
                                  ("loops=4/mhc-fixed", True, 4, True),
                                  ("loops=4/no-mhc", True, 4, False),
                                  ("loops=8/mhc-fixed", True, 8, True)):
        cfg = Config(world="kinship", train_hops=(2, 3), block=512, steps=args.steps,
                     n_examples=800, loop=loop, loops=loops, mhc=mhc,
                     log_every=args.steps // 4)
        log.info("== ablation arm %s ==", tag)
        Trainer(cfg).train()


def cmd_train(args):
    from .config import Config
    from .train import Trainer
    cfg = Config.from_dict(vars(args))
    cfg.sup, cfg.seed, cfg.arch, cfg.world = args.sup, args.seed, args.arch, args.world
    if args.world == "kinship":                            # nested proofs: depths 2-3 exist
        cfg.train_hops, cfg.test_hops = (2, 3), (2, 3)
        cfg.block = 512                                    # family trees need the full window
    if args.simple:                                        # STAIRCASE RUNG A: minimal world --
        cfg.extract_frac = cfg.write_frac = 0.0            # shallow QA only, built-in surfaces,
        cfg.math_frac = cfg.def_frac = cfg.novel_frac = 0.0   # no curriculum, no exams
        cfg.curriculum = False
        cfg.lang_level = "canonical" if args.canon else "builtin"
        cfg.deep_depth = args.deep_depth                   # rung C: --simple + explicit deep
        if args.bank:                                      # RUNG B: + distilled surface bank
            cfg.lang_level = "mix"
            cfg.curriculum = not args.no_curriculum
            cfg.block = max(cfg.block, 1024)               # never shrink a deep-sized block
        if args.deep_depth:                                # DEEP regime: 50%-mix ancestor spines
            n = cfg.deep_depth = args.deep_depth
            cfg.test_hops = (2, 3, max(4, n // 2), n)
            cfg.block = max(768, 144 * n)                  # distilled scholar surfaces run to 32
            #                                                tokens/fact; born/died double the count
            cfg.batch = 8 if cfg.block > 1024 else cfg.batch
            cfg.n_test_entities = max(cfg.n_test_entities, 2 * n + 16)  # sampler asserts 2k+10
    if args.steps:
        cfg.steps = args.steps
    if args.batch:
        cfg.batch = args.batch
    if args.dim:
        cfg.d, cfg.heads = args.dim, max(cfg.heads, args.dim // 32)
    if args.examples:
        cfg.n_examples = args.examples
    if args.loops:
        cfg.loops = args.loops
    if args.deep_frac:
        cfg.deep_frac = args.deep_frac
    if args.contrastive:
        cfg.contrastive_frac = args.contrastive
    if args.rule_w:
        cfg.rule_w = args.rule_w
    if args.rule_contrast_w:
        cfg.rule_contrast_w = args.rule_contrast_w
    if args.rule_contrast_temp:
        cfg.rule_contrast_temp = args.rule_contrast_temp
    if args.trace_rank_w:
        cfg.trace_rank_w = args.trace_rank_w
    if args.trace_rank_batch:
        cfg.trace_rank_batch = args.trace_rank_batch
    if args.trace_rank_candidates:
        cfg.trace_rank_candidates = args.trace_rank_candidates
    if args.trace_rank_states:
        cfg.trace_rank_states = args.trace_rank_states
    if args.trace_dagger_frac is not None:
        cfg.trace_dagger_frac = args.trace_dagger_frac
    cfg.deep_preds = tuple(p.strip() for p in args.deep_preds.split(",") if p.strip()) \
        if args.deep_preds else ()
    if args.pos:
        cfg.pos_mode = args.pos
    if args.test_names_n:
        cfg.n_test_entities = args.test_names_n
    if args.no_loop:
        cfg.loop = False
    t = Trainer(cfg)
    m, vocab = t.train()
    if args.neg and cfg.sup == "steps":
        t.train_negatives(m, vocab, t.mine_negatives(m, vocab))
    t.save(args.out, m, vocab)


def cmd_eval(args):
    from .train import load_run
    from .evaluate import evaluate, save_results
    cfg, m, vocab, trainer, runtime = load_run(args.run)
    hops = tuple(int(h) for h in args.hops.split(",")) if args.hops else None
    mode = args.mode or ("path" if cfg.sup == "path" else "verified")
    preds = tuple(args.preds.split(",")) if args.preds else None
    if args.block:
        cfg.block = runtime.cfg.block = args.block         # length-gen: eval beyond training ctx
    ents = trainer.train_ents if args.train_names else None
    level = args.level if args.level != "mix" else cfg.lang_level   # respect --simple runs
    res = evaluate(runtime, trainer, mode=mode, hops=hops, split=args.split, n=args.n or None,
                   preds=preds, phrasings=args.phrasings, lang_level=level, entities=ents)
    tag = f"{mode}/{args.split}" + (f"/{args.preds}" if args.preds else "") + \
          (f"/k{args.hops}" if args.hops else "") + \
          ("/train-names" if args.train_names else "") + \
          (f"/{args.phrasings}-phrasings" if args.phrasings != "train" else "") + \
          (f"/{args.level}" if args.level != "mix" else "")
    save_results(args.run, {tag: res})


def cmd_deep_eval(args):
    import json
    from .train import load_run
    from .deep_eval import deep_eval, parse_ints, save_deep_report
    cfg, _m, _vocab, trainer, runtime = load_run(args.run)
    if args.block:
        cfg.block = runtime.cfg.block = args.block
    depths = parse_ints(args.depths, default=(4, 8, 16, 32, 64))
    preds = tuple(p.strip() for p in args.preds.split(",") if p.strip())
    level = args.level if args.level != "mix" else cfg.lang_level
    report = deep_eval(runtime, trainer, depths=depths, n=args.n or None, preds=preds,
                       mode=args.mode, phrasings=args.phrasings, lang_level=level,
                       train_names=args.train_names, decode=args.decode)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        path = args.out
    else:
        path = save_deep_report(args.run, report)
    print(json.dumps(report["by_depth"], indent=1))
    log.info("deep-eval -> %s", path)


def cmd_probe(args):
    import json
    from .train import load_run
    from .deep_eval import parse_ints
    from .probes import probe_report, save_probe_report
    cfg, _m, _vocab, trainer, runtime = load_run(args.run)
    if args.block:
        cfg.block = runtime.cfg.block = args.block
    depths = parse_ints(args.depths, default=(4, 8, 16, 32))
    preds = tuple(p.strip() for p in args.preds.split(",") if p.strip())
    level = args.level if args.level != "mix" else cfg.lang_level
    report = probe_report(runtime, trainer, depths=depths, n=args.n, preds=preds,
                          layer=args.layer, phrasings=args.phrasings, lang_level=level,
                          train_names=args.train_names)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        path = args.out
    else:
        path = save_probe_report(args.run, report)
    print(json.dumps({
        "n_vectors": report["n_vectors"],
        "same_rule_cos": report["same_rule_cos"],
        "different_rule_cos": report["different_rule_cos"],
        "rule_reuse_margin": report["rule_reuse_margin"],
        "same_rule_cross_depth_cos": report["same_rule_cross_depth_cos"],
        "cross_depth_reuse_margin": report["cross_depth_reuse_margin"],
        "cross_depth_reuse_gap": report["cross_depth_reuse_gap"],
        "depth_index_leakage": report["depth_index_leakage"],
        "ufr_score": report["ufr_score"],
        "verdict": report["verdict"],
        "weak_rules": report["weak_rules"],
        "risk_flags": report["risk_flags"],
    }, indent=1))
    log.info("probe -> %s", path)


def cmd_demo(args):
    from .train import load_run
    from .evaluate import demo
    _, _, _, trainer, runtime = load_run(args.run)
    demo(runtime, trainer, k=args.k)


def cmd_sweep(args):
    """The comparison grid: sup x arch x seed, each as its own run dir + a combined table."""
    import json
    from .config import Config
    from .train import Trainer
    from .evaluate import evaluate, save_results
    from .flow import FlowRuntime
    from .verify import StepChecker
    from .world import RULES
    from .train import DEV
    import numpy as np
    seeds = [int(s) for s in args.seeds.split(",")]
    archs = args.archs.split(",")
    table = {}
    for sup in ("path", "steps"):
        for arch in archs:
            for seed in seeds:
                cfg = Config(sup=sup, arch=arch, seed=seed)
                if args.steps:
                    cfg.steps = args.steps
                run = os.path.join(args.out, f"{sup}_{arch}_s{seed}")
                t = Trainer(cfg)
                log.info("== train %s ==", run)
                m, vocab = t.train()
                t.save(run, m, vocab)
                runtime = FlowRuntime(m, vocab, StepChecker(RULES), cfg, DEV)
                modes = ("path",) if sup == "path" else ("free", "verified")
                for mode in modes:
                    res = evaluate(runtime, t, mode=mode)
                    save_results(run, {mode: res})
                    table.setdefault(f"{sup}/{arch}/{mode}", []).append(res)
                if sup == "steps" and args.neg:
                    t.train_negatives(m, vocab, t.mine_negatives(m, vocab))
                    res = evaluate(runtime, t, mode="verified")
                    save_results(run, {"verified+neg": res})
                    table.setdefault(f"{sup}/{arch}/verified+neg", []).append(res)
    summary = {cond: {k: float(np.mean([r[k]["acc"] for r in rs])) for k in rs[0]}
               for cond, rs in table.items()}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="thinking", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("induce")
    p.add_argument("--out", default="rules.json")
    p.add_argument("--n-rel", type=int, default=240, dest="n_rel")
    p.add_argument("--n-val", type=int, default=160, dest="n_val")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("train")
    p.add_argument("--out", required=True)
    p.add_argument("--world", default="chain", choices=("chain", "kinship"))
    p.add_argument("--deep-depth", type=int, default=0, dest="deep_depth")
    p.add_argument("--simple", action="store_true",
                   help="staircase rung A: shallow QA, built-in surfaces, no exams")
    p.add_argument("--canon", action="store_true",
                   help="rung A0: canonical fact surfaces (chain-world conditions)")
    p.add_argument("--bank", action="store_true",
                   help="rung B: distilled 8-level surface bank + curriculum")
    p.add_argument("--no-curriculum", action="store_true", dest="no_curriculum",
                   help="bank without ladder phases (B6 showed fixed phase pools memorize)")
    p.add_argument("--sup", default="steps", choices=("steps", "path"))
    p.add_argument("--arch", default="standard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--dim", type=int, default=0, help="model width (256 = the proven point)")
    p.add_argument("--examples", type=int, default=0)
    p.add_argument("--loops", type=int, default=0)
    p.add_argument("--deep-frac", type=float, default=0.0, dest="deep_frac")
    p.add_argument("--contrastive", type=float, default=0.0, help="contrastive triplet share")
    p.add_argument("--rule-w", type=float, default=0.0, dest="rule_w",
                   help="auxiliary rule/action classifier weight")
    p.add_argument("--rule-contrast-w", type=float, default=0.0, dest="rule_contrast_w",
                   help="same-rule contrastive alignment weight")
    p.add_argument("--rule-contrast-temp", type=float, default=0.1,
                   dest="rule_contrast_temp", help="contrastive temperature override")
    p.add_argument("--trace-rank-w", type=float, default=0.0, dest="trace_rank_w",
                   help="next verifier-action ranking loss weight")
    p.add_argument("--trace-rank-batch", type=int, default=0, dest="trace_rank_batch",
                   help="ranking states per optimizer step")
    p.add_argument("--trace-rank-candidates", type=int, default=0,
                   dest="trace_rank_candidates", help="candidate cap for rank loss/decode")
    p.add_argument("--trace-rank-states", type=int, default=0, dest="trace_rank_states",
                   help="max support/on-policy steps before a rank target")
    p.add_argument("--trace-dagger-frac", type=float, default=None, dest="trace_dagger_frac",
                   help="fraction of rank states reached by model-ranked rollout")
    p.add_argument("--deep-preds", default="", dest="deep_preds",
                   help="restrict DEEP-regime query types, e.g. ancestor or ancestor,older_by")
    p.add_argument("--pos", choices=("rope", "none"), help="position mode (none = NoPE)")
    p.add_argument("--test-names", type=int, default=0, dest="test_names_n",
                   help="test name-pool size (length-gen: must cover EVAL depth, 2k+16)")
    p.add_argument("--no-loop", action="store_true", dest="no_loop")
    p.add_argument("--neg", action="store_true")
    p = sub.add_parser("ablate")
    p.add_argument("--steps", type=int, default=800)
    p = sub.add_parser("eval")
    p.add_argument("run")
    p.add_argument("--mode",
                   choices=("free", "verified", "masked", "path", "extract", "self", "write",
                            "math", "define"))
    p.add_argument("--split", default="iid", choices=("iid", "holdout", "novel"))
    p.add_argument("--hops")
    p.add_argument("--n", type=int, default=0, help="examples per depth (default cfg.n_eval)")
    p.add_argument("--preds", help="restrict query types, e.g. older_by,who_older,who_younger")
    p.add_argument("--block", type=int, default=0, help="eval context override (length-gen)")
    p.add_argument("--phrasings", default="train", choices=("train", "eval"),
                   help="eval = HELD-OUT surface patterns (language-understanding test)")
    p.add_argument("--train-names", action="store_true", dest="train_names",
                   help="evaluate on TRAINING-pool names (harness/generalization discriminator)")
    p.add_argument("--level", default="mix", help="education register: preschool..scholar|mix")
    p = sub.add_parser("deep-eval")
    p.add_argument("run")
    p.add_argument("--depths", default="4,8,16,32,64",
                   help="comma-separated deep depths to test")
    p.add_argument("--n", type=int, default=0, help="examples per depth (default cfg.n_eval)")
    p.add_argument("--preds", default="ancestor",
                   help="restrict deep query types, e.g. ancestor or ancestor,older_by")
    p.add_argument("--mode", default="verified", choices=("free", "verified"))
    p.add_argument("--decode", default="sample",
                   choices=("sample", "hybrid", "constrained", "ranker"),
                   help="sample = token generation with verifier retries; constrained = follow "
                        "the generic checker frontier when available; ranker = model-rank legal "
                        "verifier actions; hybrid = sample until the verifier stalls, then "
                        "switch to that generic frontier")
    p.add_argument("--block", type=int, default=0, help="eval context override")
    p.add_argument("--phrasings", default="train", choices=("train", "eval"))
    p.add_argument("--train-names", action="store_true", dest="train_names",
                   help="use training names instead of held-out/depth-sized eval names")
    p.add_argument("--level", default="mix", help="education register: preschool..scholar|mix")
    p.add_argument("--out", default="", help="write report to this JSON file")
    p = sub.add_parser("probe")
    p.add_argument("run")
    p.add_argument("--depths", default="4,8,16,32",
                   help="comma-separated deep depths to probe")
    p.add_argument("--n", type=int, default=4, help="teacher-forced examples per depth")
    p.add_argument("--preds", default="ancestor")
    p.add_argument("--layer", type=int, default=-1, help="transformer block to capture")
    p.add_argument("--block", type=int, default=0, help="context override for fit checks")
    p.add_argument("--phrasings", default="train", choices=("train", "eval"))
    p.add_argument("--train-names", action="store_true", dest="train_names")
    p.add_argument("--level", default="mix", help="education register: preschool..scholar|mix")
    p.add_argument("--out", default="", help="write report to this JSON file")
    p = sub.add_parser("demo")
    p.add_argument("run")
    p.add_argument("--k", type=int, default=6)
    p = sub.add_parser("sweep")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--archs", default="standard,relational")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--neg", action="store_true", default=True)
    args = ap.parse_args(argv)
    {"selftest": cmd_selftest, "induce": cmd_induce, "train": cmd_train, "eval": cmd_eval,
     "deep-eval": cmd_deep_eval, "probe": cmd_probe, "demo": cmd_demo, "sweep": cmd_sweep,
     "ablate": cmd_ablate}[args.cmd](args)


if __name__ == "__main__":
    main()
