"""Model-selection helpers for self-study loops."""

import math


def signal_regression_report(before_scores, after_scores, score_keys,
                             skip_keys=None, tolerance=0.0,
                             signals=None):
    """Report whether active score signals regressed beyond a small tolerance."""
    before_scores = before_scores if isinstance(before_scores, dict) else {}
    after_scores = after_scores if isinstance(after_scores, dict) else {}
    score_keys = dict(score_keys or {})
    skip_keys = dict(skip_keys or {})
    if signals is None:
        signals = tuple(score_keys)
    tolerance = max(0.0, float(tolerance))
    active = []
    deltas = {}
    regressions = {}
    nonfinite = []
    for signal in signals:
        if signal not in score_keys:
            continue
        score_key = score_keys[signal]
        if score_key not in before_scores or score_key not in after_scores:
            continue
        skip_key = skip_keys.get(signal, f"{signal}_skipped")
        before_skipped = bool(before_scores.get(skip_key, False))
        after_skipped = bool(after_scores.get(skip_key, False))
        if before_skipped and after_skipped:
            continue
        before_value = float(before_scores.get(score_key, 0.0) or 0.0)
        after_value = float(after_scores.get(score_key, 0.0) or 0.0)
        if not math.isfinite(after_value):
            active.append(str(signal))
            deltas[str(signal)] = -1.0
            regressions[str(signal)] = max(1.0, tolerance + 1e-12)
            nonfinite.append(str(signal))
            continue
        if not math.isfinite(before_value):
            before_value = 0.0
        delta = after_value - before_value
        active.append(str(signal))
        deltas[str(signal)] = float(delta)
        regression = max(0.0, -delta)
        if regression > tolerance:
            regressions[str(signal)] = float(regression)
    max_regression = max(regressions.values()) if regressions else 0.0
    return {
        "enabled": True,
        "tolerance": float(tolerance),
        "allowed": not bool(regressions),
        "active_signals": active,
        "signal_deltas": deltas,
        "regressions": regressions,
        "regressed_signals": list(regressions),
        "nonfinite_signals": nonfinite,
        "max_regression": float(max_regression),
    }


def concept_round_selection_decision(
        score_delta_from_best, score_min_delta, insight_delta=0.0,
        insight_allowed=True, insight_gate=False, insight_accept_w=0.25,
        insight_min_delta=0.0):
    score_delta_from_best = float(score_delta_from_best)
    score_min_delta = float(score_min_delta)
    insight_delta = float(insight_delta)
    insight_accept_w = float(insight_accept_w)
    insight_min_delta = float(insight_min_delta)
    selected_by_score = (
        bool(insight_allowed) and score_delta_from_best > score_min_delta)
    insight_boost = 0.0
    if (bool(insight_allowed) and bool(insight_gate)
            and insight_accept_w > 0.0 and insight_delta > insight_min_delta):
        insight_boost = insight_accept_w * max(0.0, insight_delta)
    effective_delta = score_delta_from_best + insight_boost
    selected_by_insight = (
        bool(insight_allowed) and insight_boost > 0.0
        and effective_delta > score_min_delta)
    return {
        "selected": bool(selected_by_score or selected_by_insight),
        "selected_by_score": bool(selected_by_score),
        "selected_by_insight": bool(
            selected_by_insight and not selected_by_score),
        "insight_score_boost": float(insight_boost),
        "insight_effective_delta": float(effective_delta),
    }
