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


def _cos(a, b):
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return 0.0
    return float(np.dot(a, b) / den)


def _mean(xs):
    return float(np.mean(xs)) if xs else None


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
    for meta in example.meta.get("lines", []):
        if kinds and meta["kind"] not in kinds:
            continue
        start = int(meta["start"])
        end = min(int(meta["end"]), hidden.shape[0])
        if end <= start:
            continue
        vec = hidden[start:end].mean(0)
        rows.append({**meta, "vector": vec})
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

    same, diff = [], []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            c = _cos(vectors[i]["vector"], vectors[j]["vector"])
            if vectors[i]["rule_id"] == vectors[j]["rule_id"]:
                same.append(c)
            else:
                diff.append(c)

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

    same_mean, diff_mean = _mean(same), _mean(diff)
    risk_flags = []
    if same_mean is not None and diff_mean is not None and same_mean - diff_mean < 0.05:
        risk_flags.append("low_same_vs_different_rule_margin")
    if rules.get("ancestor_forward", {}).get("mean_same_rule_cos") is not None:
        if rules["ancestor_forward"]["mean_same_rule_cos"] < 0.2:
            risk_flags.append("weak_ancestor_forward_reuse")

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
        "same_rule_cos": same_mean,
        "different_rule_cos": diff_mean,
        "rule_reuse_margin": (same_mean - diff_mean
                              if same_mean is not None and diff_mean is not None else None),
        "rules": rules,
        "ancestor_forward_depth_cos": depth_cos,
        "risk_flags": risk_flags,
    }


def save_probe_report(run_dir, report, name="fer_probe.json"):
    path = os.path.join(run_dir, name)
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    return path


__all__ = ["parse_ints", "collect_trace_vectors", "probe_report", "save_probe_report"]
