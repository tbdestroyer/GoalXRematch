from __future__ import annotations

import math

from goalx.models import GoalXRequest, GoalXResponse, TacticalSuggestion


def _distance(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
    return math.hypot(a_x - b_x, a_y - b_y)


def _suggest_spacing(request: GoalXRequest) -> TacticalSuggestion | None:
    min_dist = 100.0
    players = request.teammates
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            dist = _distance(players[i].x, players[i].y, players[j].x, players[j].y)
            if dist < min_dist:
                min_dist = dist
    if min_dist < 9.0:
        return TacticalSuggestion(
            title="Increase spacing between nearest teammates",
            detail="Two teammates are too close. Expand passing lanes to improve shot buildup quality.",
            priority="high",
            metric_impact=0.24,
        )
    return None


def _suggest_width(request: GoalXRequest) -> TacticalSuggestion | None:
    ys = [p.y for p in request.teammates]
    team_width = max(ys) - min(ys)
    if request.phase in ("attack", "transition") and team_width < 38.0:
        return TacticalSuggestion(
            title="Use full width in possession",
            detail="Stretch opponents by placing a wide option on each side before final-third entry.",
            priority="medium",
            metric_impact=0.17,
        )
    return None


def _suggest_half_space_occupation(request: GoalXRequest) -> TacticalSuggestion | None:
    in_left_half_space = any(20.0 <= p.y <= 35.0 and p.x >= 55.0 for p in request.teammates)
    in_right_half_space = any(65.0 >= p.y >= 50.0 and p.x >= 55.0 for p in request.teammates)
    if request.phase == "attack" and not (in_left_half_space and in_right_half_space):
        return TacticalSuggestion(
            title="Occupy both half-spaces near box",
            detail="Position one attacker in each half-space to improve passing angles and shot probability.",
            priority="high",
            metric_impact=0.26,
        )
    return None


def _suggest_defensive_compactness(request: GoalXRequest) -> TacticalSuggestion | None:
    if request.phase != "defense" or len(request.teammates) < 4:
        return None
    ys = [p.y for p in request.teammates]
    xs = [p.x for p in request.teammates]
    team_height = max(xs) - min(xs)
    team_width = max(ys) - min(ys)
    if team_height > 42.0 or team_width > 48.0:
        return TacticalSuggestion(
            title="Recover compact defensive block",
            detail="Defensive unit is too stretched. Compress lines to reduce central shots and through balls.",
            priority="high",
            metric_impact=0.28,
        )
    return None


def analyze_goalx(request: GoalXRequest) -> GoalXResponse:
    suggestions: list[TacticalSuggestion] = []
    checks = (
        _suggest_spacing(request),
        _suggest_width(request),
        _suggest_half_space_occupation(request),
        _suggest_defensive_compactness(request),
    )
    for suggestion in checks:
        if suggestion is not None:
            suggestions.append(suggestion)

    impact_penalty = sum(s.metric_impact for s in suggestions)
    base_score = 78 if request.phase == "attack" else 74
    goalx_score = max(10, min(100, round(base_score - impact_penalty * 100)))

    suggestions.sort(key=lambda x: (x.priority != "high", -x.metric_impact))
    return GoalXResponse(goalx_score=goalx_score, suggestions=suggestions)

