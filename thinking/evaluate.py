"""Evaluation: held-out worlds across depths, three decode modes, JSON-serializable results."""
import json
import logging
import os
import numpy as np

from .world import ChainWorld
from .flow import FlowRuntime

log = logging.getLogger("thinking")


def evaluate(runtime: FlowRuntime, trainer, mode="verified", hops=None, n=None, entities=None,
             split="iid", preds=None, phrasings="train", lang_level="mix"):
    """mode: 'free' | 'verified' | 'path' (path-sup models). Held-out entities by default.
    split (kinship): 'iid' = trained goal predicates | 'holdout' = composition-holdout predicates.
    preds: restrict query types (e.g. relativity set). phrasings='eval' uses HELD-OUT surface
    patterns (the language-understanding test); lang_level picks the education register."""
    cfg = runtime.cfg
    kin = cfg.world == "kinship"
    if kin:
        from .kinship import FamilyWorld, surfaces
        TEMPLATES, QUESTION = surfaces(lang_level, phrasings)
        world = FamilyWorld(entities if entities is not None else trainer.test_ents)
        inc, exc = ((cfg.holdout_preds, None) if split == "holdout"
                    else (None, cfg.holdout_preds))
        if preds:
            inc, exc = tuple(preds), None
    else:
        world = ChainWorld(entities if entities is not None else trainer.test_ents)
    out = {}
    for k in (hops or cfg.test_hops):
        rng = np.random.default_rng(cfg.eval_seed + k)
        correct = inval = res = skipped = 0
        N = n or cfg.n_eval
        for _ in range(N):
            try:
                if kin and split == "novel":           # relationship NOT in the dataset:
                    p, _ = world.sample_novel(rng, train=False)   # held-out compositions
                elif kin:
                    for _try in range(20):             # world must fit the context window
                        p, _ = world.sample(k, rng, include=inc, exclude=exc)
                        plen = sum(max(len(v) for v in TEMPLATES[pred]) for pred, _ in p.edb) + 12
                        budget = cfg.block - (26 * k + 96 if k >= 4 else 96)   # room for the trace
                        if plen <= budget:
                            break
                    else:
                        skipped += 1
                        continue
                else:
                    p = world.sample(k, rng)
            except RuntimeError:                       # no problem at this depth/split
                skipped += 1
                continue
            if cfg.anonymize:
                from .world import anonymize
                p, _ = anonymize(p, [], rng)
            if mode == "path":
                pred = runtime.path_answer(p)
            elif kin and mode == "extract":                # READING exam: extraction F1
                _, m = runtime.extract(p, TEMPLATES, QUESTION, rng=rng)
                correct += m["f1"]
                pred = None
                continue
            elif kin and mode == "math":                   # MATH exam: unseen operand pairs
                y1 = 1500 + int(rng.integers(900))
                y2 = y1 + int(rng.integers(601))
                correct += (runtime.compute(y1, y2) == str(y2 - y1))
                pred = None
                continue
            elif kin and mode == "define":                 # VOCABULARY exam: definition match
                from .kinship import definitions as _defs, bank_levels
                lvls = bank_levels() or ["mix"]
                lv = lvls[int(rng.integers(len(lvls)))]
                dfs = {**_defs(lv, "train"), **{}}
                if not dfs:
                    skipped += 1
                    continue
                w = sorted(dfs)[int(rng.integers(len(dfs)))]
                words = runtime.define(w, lv)
                ok = ({tuple(v[:-1]) for v in dfs.get(w, [])} |
                      {tuple(v[:-1]) for v in _defs(lv, "eval").get(w, [])})
                correct += (tuple(words) in ok)
                pred = None
                continue
            elif kin and mode == "write":                  # WRITING exam: level-pattern match
                from .kinship import surfaces as _sf, bank_levels
                lvls = bank_levels() or ["mix"]
                lv = lvls[int(rng.integers(len(lvls)))]
                fact = p.edb[int(rng.integers(len(p.edb)))]
                words = runtime.write(fact, lv)
                ok_pats = (_sf(lv, "train")[0].get(fact[0], []) +
                           list(_sf(lv, "eval")[0].get(fact[0], [])))
                h, t = fact[1]
                filled = {tuple(h if w == "{h}" else t if w == "{t}" else w for w in v)[:-1]
                          for v in ok_pats}                # patterns sans the final '.'
                correct += (tuple(words) in filled)
                pred = None
                continue
            elif kin and mode == "self":                   # reason over the model's OWN extraction
                facts, _ = runtime.extract(p, TEMPLATES, QUESTION, rng=rng)
                r = runtime.run_goal(p, TEMPLATES, QUESTION, verify=True, rng=rng, edb=facts)
                pred, inval, res = r.answer, inval + r.n_invalid, res + r.n_resampled
            elif kin:
                r = runtime.run_goal(p, TEMPLATES, QUESTION, verify=(mode == "verified"), rng=rng)
                pred, inval, res = r.answer, inval + r.n_invalid, res + r.n_resampled
            else:
                r = runtime.run(p, verify=(mode == "verified"))
                pred, inval, res = r.answer, inval + r.n_invalid, res + r.n_resampled
            correct += (pred == p.answer)
        done = N - skipped
        if done == 0:
            out[k] = {"acc": 0.0, "invalid_per_ex": 0.0, "resampled_per_ex": 0.0,
                      "n": 0, "skipped": skipped}
            log.info("k=%-3d %s/%s skipped all %d examples (context budget too small)",
                     k, mode, split, skipped)
            continue
        out[k] = {"acc": correct / done, "invalid_per_ex": inval / done,
                  "resampled_per_ex": res / done, "n": done, "skipped": skipped}
        log.info("k=%-3d %s/%s acc %.2f (inval/ex %.1f, resamp/ex %.1f, n=%d)",
                 k, mode, split, out[k]["acc"], out[k]["invalid_per_ex"],
                 out[k]["resampled_per_ex"], done)
    return out


def save_results(out_dir, results):
    path = os.path.join(out_dir, "results.json")
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update(results)
    with open(path, "w") as f:
        json.dump(existing, f, indent=1)
    log.info("results -> %s", path)


def demo(runtime, trainer, k=6, seed=7):
    """Print one annotated thinking flow on a held-out world."""
    rng = np.random.default_rng(seed)
    if runtime.cfg.world == "kinship":
        from .kinship import FamilyWorld, surfaces
        from .trace import render_prompt
        from .world import anonymize
        TEMPLATES, QUESTION = surfaces(runtime.cfg.lang_level)   # match the trained register
        world = FamilyWorld(trainer.test_ents)
        p, _ = world.sample(k, rng)
        if runtime.cfg.anonymize:
            p, _ = anonymize(p, [], rng)
        prompt, _, _ = render_prompt(p, TEMPLATES, QUESTION, rng)
        print(" ".join(prompt) + f"   (oracle: {p.answer})")
        r = runtime.run_goal(p, TEMPLATES, QUESTION, verify=True, prompt=prompt)
    else:
        world = ChainWorld(trainer.test_ents)
        p = world.sample(k, rng)
        if runtime.cfg.anonymize:
            from .world import anonymize
            p, _ = anonymize(p, [], rng)
        print("world:", " ".join(" ".join((h, pred, t, ".")) for pred, (h, t) in p.edb))
        print(f"query: {p.goal[0]} {p.head} ?   (oracle: {p.answer})")
        r = runtime.run(p, verify=True)
    for words, status in r.lines:
        print(f"  {' '.join(words)} .   [{status}]")
    print(f"flow answer: {r.answer}  ({'correct' if r.answer == p.answer else 'WRONG'})")
    return r
