"""FER/UFR representation probes for typed reasoning traces.

The probes run the trained model on gold traces and compare hidden states for equivalent rule
steps across depths and contexts. High same-rule similarity with a positive same-vs-different
margin is evidence for reusable structure; weak margins are a FER risk signal.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from .deep_eval import _deep_entities, parse_ints


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def _unit_interval(x, low, high):
    if x is None:
        return None
    if high <= low:
        return 0.0
    return _clamp01((float(x) - low) / (high - low))


def _cos(a, b):
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return 0.0
    return float(np.dot(a, b) / den)


def _mean(xs):
    return float(np.mean(xs)) if xs else None


def _stats(xs):
    if not xs:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _empty_pairwise():
    names = (
        "same_rule",
        "different_rule",
        "same_rule_within_depth",
        "same_rule_cross_depth",
        "same_rule_same_depth_index",
        "same_rule_different_depth_index",
    )
    return {name: [] for name in names}


def pairwise_probe_metrics(vectors):
    """Aggregate generic FER/UFR pairwise metrics over labeled trace vectors.

    This deliberately depends only on trace metadata labels, problem depth, and occurrence index.
    It does not encode any kinship rule grammar.
    """
    buckets = _empty_pairwise()
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            c = _cos(a["vector"], b["vector"])
            same_rule = a["rule_id"] == b["rule_id"]
            buckets["same_rule" if same_rule else "different_rule"].append(c)
            if not same_rule:
                continue
            if a.get("problem_depth") == b.get("problem_depth"):
                buckets["same_rule_within_depth"].append(c)
            else:
                buckets["same_rule_cross_depth"].append(c)
            ai, bi = a.get("depth_index"), b.get("depth_index")
            if ai is not None and bi is not None:
                if ai == bi:
                    buckets["same_rule_same_depth_index"].append(c)
                else:
                    buckets["same_rule_different_depth_index"].append(c)

    pairwise = {name: _stats(vals) for name, vals in buckets.items()}
    same_mean = pairwise["same_rule"]["mean"]
    diff_mean = pairwise["different_rule"]["mean"]
    cross_mean = pairwise["same_rule_cross_depth"]["mean"]
    within_mean = pairwise["same_rule_within_depth"]["mean"]
    same_idx_mean = pairwise["same_rule_same_depth_index"]["mean"]
    diff_idx_mean = pairwise["same_rule_different_depth_index"]["mean"]
    return {
        "pairwise": pairwise,
        "same_rule_cos": same_mean,
        "different_rule_cos": diff_mean,
        "rule_reuse_margin": (
            same_mean - diff_mean if same_mean is not None and diff_mean is not None else None),
        "same_rule_cross_depth_cos": cross_mean,
        "cross_depth_reuse_margin": (
            cross_mean - diff_mean if cross_mean is not None and diff_mean is not None else None),
        "cross_depth_reuse_gap": (
            within_mean - cross_mean if within_mean is not None and cross_mean is not None else None),
        "depth_index_leakage": (
            same_idx_mean - diff_idx_mean
            if same_idx_mean is not None and diff_idx_mean is not None else None),
    }


def score_probe_metrics(metrics, rules=None, n_vectors=0):
    """Return heuristic UFR score + FER risk flags from pairwise probe metrics."""
    rules = rules or {}
    margin = metrics.get("rule_reuse_margin")
    cross_margin = metrics.get("cross_depth_reuse_margin")
    drift_gap = metrics.get("cross_depth_reuse_gap")
    idx_leak = metrics.get("depth_index_leakage")

    margin_score = _unit_interval(margin, 0.02, 0.25)
    cross_score = _unit_interval(cross_margin, 0.02, 0.25)
    if cross_score is None:
        cross_score = margin_score
    leak_penalty = max(
        _unit_interval(drift_gap, 0.05, 0.30) or 0.0,
        _unit_interval(idx_leak, 0.03, 0.20) or 0.0,
    )
    usable = [s for s in (margin_score, cross_score) if s is not None]
    if usable:
        reuse_score = 0.55 * usable[0] + 0.45 * usable[-1]
        ufr_score = _clamp01(0.8 * reuse_score + 0.2 * (1.0 - leak_penalty))
    else:
        ufr_score = None

    risk_flags = []
    if n_vectors < 8:
        risk_flags.append("insufficient_probe_vectors")
    if margin is not None and margin < 0.05:
        risk_flags.append("low_same_vs_different_rule_margin")
    if cross_margin is not None and cross_margin < 0.05:
        risk_flags.append("weak_cross_depth_rule_reuse")
    if drift_gap is not None and drift_gap > 0.15:
        risk_flags.append("depth_tied_representation")
    if idx_leak is not None and idx_leak > 0.10:
        risk_flags.append("index_tied_representation")
    weak_rules = sorted(
        rid for rid, row in rules.items()
        if row.get("n", 0) >= 2
        and row.get("mean_same_rule_cos") is not None
        and row["mean_same_rule_cos"] < 0.2)
    if weak_rules:
        risk_flags.append("weak_rule_reuse")
    if ufr_score is None:
        verdict = "unknown"
    elif ufr_score < 0.35:
        verdict = "high_fer_risk"
    elif ufr_score < 0.55:
        verdict = "medium_fer_risk"
    else:
        verdict = "low_fer_risk"
    return {
        "ufr_score": ufr_score,
        "verdict": verdict,
        "weak_rules": weak_rules,
        "risk_flags": sorted(set(risk_flags)),
    }


def _capture_block(runtime, ids, layer=-1):
    """Return hidden states from one transformer block for a single teacher-forced sequence."""
    m = runtime.m
    idx = layer if layer >= 0 else len(m.blocks) + layer
    if idx < 0 or idx >= len(m.blocks):
        raise ValueError(f"layer {layer} out of range for {len(m.blocks)} blocks")
    captured = {}

    def hook(_module, _inputs, output):
        captured["x"] = output.detach().float().cpu()

    handle = m.blocks[idx].register_forward_hook(hook)
    was_training = m.training
    m.eval()
    try:
        with torch.no_grad():
            _ = m(ids)
    finally:
        handle.remove()
        if was_training:
            m.train()
    if "x" not in captured:
        raise RuntimeError("activation hook did not fire")
    return captured["x"][0].numpy()


def collect_trace_vectors(runtime, vocab, example, layer=-1, kinds=("think",)):
    """Collect one vector per proof line from a rendered Example with line metadata."""
    ids = torch.tensor([vocab.enc(example.tokens[:-1])], device=runtime.dev)
    hidden = _capture_block(runtime, ids, layer=layer)
    rows = []
    for line_index, meta in enumerate(example.meta.get("lines", [])):
        if kinds and meta["kind"] not in kinds:
            continue
        start = int(meta["start"])
        end = min(int(meta["end"]), hidden.shape[0])
        if end <= start:
            continue
        vec = hidden[start:end].mean(0)
        rows.append({**meta, "line_index": line_index, "vector": vec})
    return rows


def probe_report(runtime, trainer, depths=(4, 8, 16, 32), n=4, preds=("ancestor",),
                 layer=-1, phrasings="train", lang_level=None, train_names=False):
    """Build a JSON report of same-rule reuse and depth drift."""
    cfg = runtime.cfg
    if cfg.world != "kinship":
        raise ValueError("FER probes currently target the kinship trace format")

    from .kinship import FamilyWorld, surfaces
    from .trace import render_goal_example
    from .world import anonymize

    depths = tuple(int(d) for d in depths)
    preds = tuple(preds) if preds else ("ancestor",)
    ents = _deep_entities(cfg, trainer, max(depths), train_names=train_names)
    world = FamilyWorld(ents, seed=cfg.world_seed + 41)
    level = lang_level or cfg.lang_level
    templates, question = surfaces(level, phrasings)
    vectors = []
    skipped = 0

    for depth in depths:
        rng = np.random.default_rng(cfg.eval_seed + 104729 * depth)
        for _ in range(n):
            try:
                problem, lines = world.sample_deep(depth, rng, include=preds)
            except (AssertionError, RuntimeError):
                skipped += 1
                continue
            if cfg.anonymize:
                problem, lines = anonymize(problem, lines, rng)
            ex = render_goal_example(
                problem, lines, templates, problem.question or question,
                np.random.default_rng(int(rng.integers(2 ** 31))))
            if len(ex.tokens) > cfg.block + 1:
                skipped += 1
                continue
            for row in collect_trace_vectors(runtime, runtime.v, ex, layer=layer):
                row["problem_depth"] = depth
                vectors.append(row)

    by_rule = defaultdict(list)
    for v in vectors:
        by_rule[v["rule_id"]].append(v)

    rules = {}
    for rid, rows in sorted(by_rule.items()):
        sims = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                sims.append(_cos(rows[i]["vector"], rows[j]["vector"]))
        rules[rid] = {
            "n": len(rows),
            "mean_same_rule_cos": _mean(sims),
            "depths": sorted({int(r["problem_depth"]) for r in rows}),
        }

    depth_cos = {}
    anc = by_rule.get("ancestor_forward", [])
    if anc:
        by_idx = defaultdict(list)
        for row in anc:
            by_idx[int(row["depth_index"])].append(row["vector"])
        means = {i: np.mean(vs, axis=0) for i, vs in by_idx.items()}
        first = min(means)
        depth_cos = {
            "to_first": {str(i): _cos(means[first], means[i]) for i in sorted(means)},
            "adjacent": {
                f"{a}->{b}": _cos(means[a], means[b])
                for a, b in zip(sorted(means), sorted(means)[1:])
            },
        }

    metrics = pairwise_probe_metrics(vectors)
    score = score_probe_metrics(metrics, rules=rules, n_vectors=len(vectors))

    counts = Counter(v["rule_id"] for v in vectors)
    return {
        "probe": "fer_ufr_trace_reuse",
        "layer": int(layer),
        "depths": list(depths),
        "preds": list(preds),
        "n_per_depth": int(n),
        "skipped": int(skipped),
        "n_vectors": len(vectors),
        "rule_counts": {k: int(counts[k]) for k in sorted(counts)},
        **metrics,
        **score,
        "rules": rules,
        "ancestor_forward_depth_cos": depth_cos,
    }


def save_probe_report(run_dir, report, name="fer_probe.json"):
    path = os.path.join(run_dir, name)
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    return path


__all__ = [
    "parse_ints",
    "collect_trace_vectors",
    "pairwise_probe_metrics",
    "score_probe_metrics",
    "probe_report",
    "save_probe_report",
]
