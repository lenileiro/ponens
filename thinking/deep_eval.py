"""Depth-scaling evaluation for deep kinship reasoning.

This is stricter than the generic evaluator: it samples the explicit deep generator, records
trace-validity metrics, and skips examples whose full gold trace cannot fit the active context.
"""
import json
import os
import time
from collections import Counter

import numpy as np


def parse_ints(spec, default=()):
    if spec is None or spec == "":
        return tuple(default)
    if isinstance(spec, (list, tuple)):
        return tuple(int(x) for x in spec)
    return tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())


def _deep_entities(cfg, trainer, max_depth, train_names=False):
    if train_names:
        return trainer.train_ents
    need = 2 * max_depth + 16
    if len(trainer.test_ents) >= need:
        return trainer.test_ents
    if cfg.world != "kinship":
        return trainer.test_ents
    from .kinship import name_pools
    _, ents = name_pools(cfg.n_train_entities, need, cfg.world_seed)
    return ents


def _first_bad(lines):
    bad = {"invalid", "drop", "skip", "reject", "repair"}
    for i, (_words, status) in enumerate(lines):
        if status in bad:
            return i
    return None


def _status_counts(lines):
    c = Counter(status for _words, status in lines)
    return {k: int(c[k]) for k in sorted(c)}


def _line_tokens(lines):
    return sum((len(words) if words else 0) + 1 for words, _status in lines)


def _rank_metrics(result):
    ranks = list(getattr(result, "rank_positions", []) or [])
    counts = list(getattr(result, "rank_candidate_counts", []) or [])
    if not ranks:
        return {
            "rank_steps": 0,
            "candidate_top1": None,
            "candidate_top5": None,
            "candidate_mrr": None,
            "avg_candidates": None,
            "oracle_missing": int(getattr(result, "rank_oracle_missing", 0) or 0),
        }
    return {
        "rank_steps": len(ranks),
        "candidate_top1": sum(r == 1 for r in ranks) / len(ranks),
        "candidate_top5": sum(r <= 5 for r in ranks) / len(ranks),
        "candidate_mrr": sum(1.0 / r for r in ranks) / len(ranks),
        "avg_candidates": (sum(counts) / len(counts)) if counts else None,
        "oracle_missing": int(getattr(result, "rank_oracle_missing", 0) or 0),
    }


def deep_eval(runtime, trainer, depths=(4, 8, 16, 32, 64), n=None, preds=("ancestor",),
              mode="verified", phrasings="train", lang_level=None, entities=None,
              train_names=False, decode="sample"):
    """Return a JSON-serializable depth report for the explicit deep generator."""
    cfg = runtime.cfg
    if cfg.world != "kinship":
        raise ValueError("deep-eval currently targets the kinship deep generator")
    if mode not in {"free", "verified"}:
        raise ValueError("deep-eval mode must be 'free' or 'verified'")

    from .kinship import FamilyWorld, surfaces
    from .trace import render_goal_example, render_prompt
    from .world import anonymize

    depths = tuple(int(d) for d in depths)
    preds = tuple(preds) if preds else ("ancestor",)
    ents = entities or _deep_entities(cfg, trainer, max(depths), train_names=train_names)
    world = FamilyWorld(ents, seed=cfg.world_seed + 17)
    level = lang_level or cfg.lang_level
    templates, question = surfaces(level, phrasings)
    N = n or cfg.n_eval
    report = {
        "world": cfg.world,
        "mode": mode,
        "decode": decode,
        "depths": list(depths),
        "preds": list(preds),
        "n_requested": int(N),
        "block": int(cfg.block),
        "phrasings": phrasings,
        "lang_level": level,
        "by_depth": {},
    }

    for depth in depths:
        rng = np.random.default_rng(cfg.eval_seed + 7919 * depth)
        rows = []
        skipped = 0
        for _ in range(N):
            try:
                problem, gold_lines = world.sample_deep(depth, rng, include=preds)
            except (AssertionError, RuntimeError):
                skipped += 1
                continue
            if cfg.anonymize:
                problem, gold_lines = anonymize(problem, gold_lines, rng)
            surface_seed = int(rng.integers(2 ** 31))
            srng = np.random.default_rng(surface_seed)
            gold = render_goal_example(problem, gold_lines, templates, question, srng)
            if len(gold.tokens) > cfg.block + 1:
                skipped += 1
                continue
            prompt, _, _ = render_prompt(
                problem, templates, problem.question or question,
                np.random.default_rng(surface_seed))
            t0 = time.perf_counter()
            result = runtime.run_goal(
                problem, templates, question, verify=(mode == "verified"), prompt=prompt,
                decode=decode)
            elapsed = time.perf_counter() - t0
            first_bad = _first_bad(result.lines)
            rankm = _rank_metrics(result)
            rows.append({
                "correct": result.answer == problem.answer,
                "answer": result.answer,
                "gold_answer": problem.answer,
                "valid_trace": first_bad is None,
                "first_invalid": first_bad,
                "invalid": int(result.n_invalid),
                "resampled": int(result.n_resampled),
                "n_lines": len(result.lines),
                "n_tokens": _line_tokens(result.lines),
                "latency_ms": elapsed * 1000.0,
                "statuses": _status_counts(result.lines),
                **rankm,
            })

        done = len(rows)
        if done == 0:
            report["by_depth"][str(depth)] = {
                "n": 0,
                "skipped": int(skipped),
                "acc": 0.0,
                "valid_trace_rate": 0.0,
                "invalid_per_ex": 0.0,
                "resampled_per_ex": 0.0,
                "avg_lines": 0.0,
                "avg_tokens": 0.0,
                "avg_latency_ms": 0.0,
                "first_invalid_mean": None,
                "first_invalid_hist": {},
                "status_counts": {},
                "rank_steps_per_ex": 0.0,
                "candidate_top1": None,
                "candidate_top5": None,
                "candidate_mrr": None,
                "avg_candidates": None,
                "oracle_missing_per_ex": 0.0,
            }
            continue

        firsts = [r["first_invalid"] for r in rows if r["first_invalid"] is not None]
        first_hist = Counter(str(i) for i in firsts)
        statuses = Counter()
        for r in rows:
            statuses.update(r["statuses"])
        rank_rows = [r for r in rows if r["rank_steps"]]
        rank_steps = sum(r["rank_steps"] for r in rows)
        report["by_depth"][str(depth)] = {
            "n": int(done),
            "skipped": int(skipped),
            "acc": sum(r["correct"] for r in rows) / done,
            "valid_trace_rate": sum(r["valid_trace"] for r in rows) / done,
            "invalid_per_ex": sum(r["invalid"] for r in rows) / done,
            "resampled_per_ex": sum(r["resampled"] for r in rows) / done,
            "avg_lines": sum(r["n_lines"] for r in rows) / done,
            "avg_tokens": sum(r["n_tokens"] for r in rows) / done,
            "avg_latency_ms": sum(r["latency_ms"] for r in rows) / done,
            "first_invalid_mean": (sum(firsts) / len(firsts)) if firsts else None,
            "first_invalid_hist": {k: int(v) for k, v in sorted(first_hist.items())},
            "status_counts": {k: int(statuses[k]) for k in sorted(statuses)},
            "rank_steps_per_ex": rank_steps / done,
            "candidate_top1": (
                sum(r["candidate_top1"] * r["rank_steps"] for r in rank_rows) / rank_steps
                if rank_steps else None),
            "candidate_top5": (
                sum(r["candidate_top5"] * r["rank_steps"] for r in rank_rows) / rank_steps
                if rank_steps else None),
            "candidate_mrr": (
                sum(r["candidate_mrr"] * r["rank_steps"] for r in rank_rows) / rank_steps
                if rank_steps else None),
            "avg_candidates": (
                sum(r["avg_candidates"] * r["rank_steps"] for r in rank_rows) / rank_steps
                if rank_steps else None),
            "oracle_missing_per_ex": sum(r["oracle_missing"] for r in rows) / done,
        }
    return report


def save_deep_report(run_dir, report, name="deep_eval.json"):
    path = os.path.join(run_dir, name)
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    return path
