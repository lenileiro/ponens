"""Model-selection helpers for self-study loops."""


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
