from __future__ import annotations

import math

from goalx.models import BlueLockStats, LivePitchScene, PlayerGoalXSnapshot

AXIS_ORDER = ("speed", "defense", "passing", "dribble", "shoot", "offense")
AXIS_LABELS = ("SPEED", "DEFENSE", "PASS", "DRIBBLE", "SHOOT", "OFFENSE")


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _per90(value: float, minutes: float) -> float:
    if minutes <= 0.0:
        return 0.0
    return value * 90.0 / minutes


def grade_for(score: float) -> str:
    if score >= 90:
        return "S"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 50:
        return "D"
    return "E"


def compute_blue_lock_stats(totals: dict[str, float]) -> BlueLockStats:
    minutes = totals["minutes"]
    matches = max(1.0, totals["matches"])
    pass_accuracy = totals["completed_passes"] / totals["attempted_passes"] if totals["attempted_passes"] > 0 else 0.0
    dribble_accuracy = totals["successful_dribbles"] / totals["dribble_attempts"] if totals["dribble_attempts"] > 0 else 0.0
    duel_rate = totals["duels_won"] / totals["duels_total"] if totals["duels_total"] > 0 else 0.0

    speed = _clamp(
        (totals["distance_m"] / matches / 11000.0) * 48
        + (_per90(totals["sprints"], minutes) / 26.0) * 52
    )
    offense = _clamp(
        _per90(totals["goals"], minutes) * 22
        + _per90(totals["assists"], minutes) * 16
        + _per90(totals["key_passes"], minutes) * 8
        + pass_accuracy * 18
    )
    shoot = _clamp(
        _per90(totals["goals"], minutes) * 24
        + _per90(totals["shots_on_target"], minutes) * 10
        + _per90(totals["shots"], minutes) * 5
    )
    dribble = _clamp(
        _per90(totals["successful_dribbles"], minutes) * 14
        + dribble_accuracy * 42
        + (_per90(totals["sprints"], minutes) / 30.0) * 18
    )
    passing = _clamp(
        pass_accuracy * 42
        + _per90(totals["key_passes"], minutes) * 8
        + _per90(totals["assists"], minutes) * 16
        + _per90(totals["completed_passes"], minutes) / 50.0 * 12
    )
    defense = _clamp(
        _per90(totals["tackles_won"], minutes) * 14
        + _per90(totals["interceptions"], minutes) * 16
        + duel_rate * 32
    )
    return BlueLockStats(
        speed=round(speed),
        offense=round(offense),
        shoot=round(shoot),
        dribble=round(dribble),
        passing=round(passing),
        defense=round(defense),
    )


def compute_goalx_score(blue_lock: BlueLockStats) -> int:
    values = [
        blue_lock.speed,
        blue_lock.offense,
        blue_lock.shoot,
        blue_lock.dribble,
        blue_lock.passing,
        blue_lock.defense,
    ]
    return round(_clamp(sum(values) / 6.0))


def build_snapshot(player_id: str, totals: dict[str, float]) -> PlayerGoalXSnapshot:
    blue_lock = compute_blue_lock_stats(totals)
    score = compute_goalx_score(blue_lock)
    return PlayerGoalXSnapshot(
        player_id=player_id,
        matches_analyzed=round(totals["matches"]),
        goalx_score=score,
        grade=grade_for(score),
        blue_lock=blue_lock,
    )


def live_axes_from_capture(
    motion_score: float,
    hud_score: float,
    scene: LivePitchScene | None,
) -> dict[str, float]:
    speed = _clamp(38 + motion_score * 420 + hud_score * 12)
    offense = 48.0
    shoot = 46.0
    dribble = _clamp(40 + motion_score * 280)
    passing = 50.0
    defense = 48.0
    if scene is None:
        return {
            "speed": speed,
            "offense": offense,
            "shoot": shoot,
            "dribble": dribble,
            "passing": passing,
            "defense": defense,
        }

    ball_x = scene.ball_x
    teammates = scene.teammates or []
    if scene.phase == "attack" or ball_x >= 58:
        offense = _clamp(58 + (ball_x - 50) * 0.7 + min(12.0, len(teammates) * 1.5))
    elif scene.phase == "defense" or ball_x <= 42:
        offense = _clamp(36 + ball_x * 0.2)
    else:
        offense = _clamp(50 + (ball_x - 50) * 0.25)

    if ball_x >= 72:
        shoot = _clamp(62 + (ball_x - 72) * 1.1 + motion_score * 80)
    elif ball_x >= 58:
        shoot = _clamp(52 + (ball_x - 58) * 0.8)
    else:
        shoot = _clamp(38 + ball_x * 0.12)

    if teammates:
        xs = [player.x for player in teammates]
        ys = [player.y for player in teammates]
        spread = (max(xs) - min(xs) + max(ys) - min(ys)) / 2.0
        passing = _clamp(38 + spread * 0.55 + (12 if len(teammates) >= 3 else 0))
        nearest = 100.0
        for index, left in enumerate(teammates):
            for right in teammates[index + 1 :]:
                dist = math.hypot(left.x - right.x, left.y - right.y)
                nearest = min(nearest, dist)
        dribble = _clamp(42 + motion_score * 260 + min(18.0, nearest * 0.35))

    if scene.phase == "defense" or ball_x <= 40:
        defense = _clamp(62 + (40 - min(ball_x, 40)) * 0.55)
    elif ball_x >= 70:
        defense = _clamp(36 + (100 - ball_x) * 0.2)
    else:
        defense = _clamp(50 + (50 - ball_x) * 0.15)

    return {
        "speed": speed,
        "offense": offense,
        "shoot": shoot,
        "dribble": dribble,
        "passing": passing,
        "defense": defense,
    }


def _logistic(value: float, center: float, scale: float, invert: bool = False) -> float:
    """Map a raw metric to 0..1 with 0.5 at ``center``; ``scale`` is roughly one grade step."""
    if scale <= 0.0:
        return 0.5
    z = (value - center) / scale
    if invert:
        z = -z
    return 1.0 / (1.0 + math.exp(-z))


def _volume(count: float, saturate_at: float) -> float:
    """0..1 confidence that a rate built from ``count`` events is meaningful."""
    if saturate_at <= 0.0:
        return 1.0
    return _clamp(count / saturate_at, 0.0, 1.0)


def _rate_term(rate: float, center: float, scale: float, count: float, saturate_at: float) -> float:
    """Logistic rate score shrunk toward neutral when the sample is tiny."""
    conf = _volume(count, saturate_at)
    return 0.5 + (_logistic(rate, center, scale) - 0.5) * conf


# Baselines approximate an average Rematch 5v5 player; 0.5 == grade C/D border.
TRACKING_BASELINES: dict[str, tuple[float, float]] = {
    "top_speed_mps": (6.0, 0.9),
    "sprint_share": (0.25, 0.10),
    "distance_per_min_m": (90.0, 25.0),
    "pace_duel_rate": (0.5, 0.18),
    "pace_duels_won_per_min": (0.30, 0.20),
    "pass_rate": (0.72, 0.10),
    "forward_pass_share": (0.45, 0.15),
    "passes_per_min": (1.5, 0.7),
    "key_passes_per_min": (0.15, 0.12),
    "shots_per_min": (0.35, 0.20),
    "on_target_rate": (0.50, 0.18),
    "avg_shot_quality": (0.45, 0.15),
    "goals_per_min": (0.12, 0.10),
    "carry_distance_per_min_m": (25.0, 12.0),
    "take_on_rate": (0.50, 0.18),
    "retention": (0.65, 0.12),
    "involvement": (0.35, 0.12),
    "attacking_third_share": (0.30, 0.12),
    "forward_runs_per_min": (0.60, 0.35),
    "box_touches_per_min": (0.30, 0.20),
    "recoveries_per_min": (0.50, 0.30),
    "goal_side_share": (0.70, 0.12),
    "avg_closing_speed_mps": (1.0, 0.8),
    "avg_nearest_mate_m": (12.0, 5.0),
}


def axes_from_tracking(metrics: dict[str, float]) -> dict[str, float]:
    """Six Blue Lock axes (0-100) from ``MatchTracker.metrics()`` output.

    Rates are shrunk toward neutral when the event count is small, velocity-based
    terms are shrunk when capture cadence was too slow to track motion, and the
    whole result is shrunk toward 50 until a few minutes of gameplay exist.
    """
    b = TRACKING_BASELINES
    minutes = max(0.0, float(metrics.get("gameplay_minutes", 0.0)))
    quality = _clamp(float(metrics.get("tracking_quality", 0.0)), 0.0, 1.0)
    per_min = max(minutes, 0.5)

    def motion(term: float) -> float:
        return 0.5 + (term - 0.5) * quality

    def L(key: str, value: float) -> float:
        center, scale = b[key]
        return _logistic(value, center, scale)

    def R(key: str, value: float, count: float, saturate_at: float) -> float:
        center, scale = b[key]
        return _rate_term(value, center, scale, count, saturate_at)

    speed = (
        0.40 * motion(R("pace_duel_rate", metrics.get("pace_duel_rate", 0.0), metrics.get("pace_duels", 0), 4.0) * 0.5
                      + L("pace_duels_won_per_min", metrics.get("pace_duels_won", 0) / per_min) * 0.5)
        + 0.25 * motion(L("top_speed_mps", metrics.get("top_speed_mps", 0.0)))
        + 0.20 * motion(L("sprint_share", metrics.get("sprint_share", 0.0)))
        + 0.15 * motion(L("distance_per_min_m", metrics.get("distance_per_min_m", 0.0)))
    )
    # Prefer the game's own "PASS" confirmation (balls played to a teammate that the HUD
    # acknowledged) over the geometric estimate when we have any confirmed samples.
    hud_kicks = float(metrics.get("pass_kicks", 0) or 0)
    if hud_kicks >= 1:
        pass_rate_term = R("pass_rate", metrics.get("pass_rate_hud", 0.0), hud_kicks, 6.0)
        pass_volume = max(hud_kicks, float(metrics.get("passes", 0) or 0))
    else:
        pass_rate_term = R("pass_rate", metrics.get("pass_rate", 0.0), metrics.get("passes", 0), 6.0)
        pass_volume = float(metrics.get("passes", 0) or 0)
    passing = (
        0.40 * pass_rate_term
        + 0.20 * R("forward_pass_share", metrics.get("forward_pass_share", 0.0), metrics.get("passes_completed", 0), 5.0)
        + 0.20 * L("passes_per_min", pass_volume / per_min)
        + 0.20 * L("key_passes_per_min", metrics.get("key_passes", 0) / per_min)
    )
    shoot = (
        0.35 * L("shots_per_min", metrics.get("shots", 0) / per_min)
        + 0.25 * R("on_target_rate", metrics.get("on_target_rate", 0.0), metrics.get("shots", 0), 3.0)
        + 0.20 * R("avg_shot_quality", metrics.get("avg_shot_quality", 0.0), metrics.get("shots", 0), 2.0)
        + 0.20 * L("goals_per_min", metrics.get("goals", 0) / per_min)
    )
    dribble = (
        0.35 * motion(L("carry_distance_per_min_m", metrics.get("carry_distance_per_min_m", 0.0)))
        + 0.35 * motion(R("take_on_rate", metrics.get("take_on_rate", 0.0), metrics.get("take_ons", 0), 4.0))
        + 0.30 * R("retention", metrics.get("retention", 0.0), metrics.get("carries", 0), 5.0)
    )
    offense = (
        0.30 * L("involvement", metrics.get("involvement", 0.0))
        + 0.20 * L("attacking_third_share", metrics.get("attacking_third_share", 0.0))
        + 0.25 * motion(L("forward_runs_per_min", metrics.get("forward_runs_per_min", 0.0)))
        + 0.25 * L("box_touches_per_min", metrics.get("box_touches", 0) / per_min)
    )
    defense = (
        0.30 * L("recoveries_per_min", metrics.get("recoveries", 0) / per_min)
        + 0.25 * motion(L("avg_closing_speed_mps", metrics.get("avg_closing_speed_mps", 0.0)))
        + 0.25 * L("goal_side_share", metrics.get("goal_side_share", 0.0))
        + 0.20 * _logistic(metrics.get("avg_nearest_mate_m", 12.0), *b["avg_nearest_mate_m"], invert=True)
    )

    reliability = _clamp(minutes / 3.0, 0.0, 1.0)

    def finish(value01: float) -> float:
        return _clamp(50.0 + (value01 * 100.0 - 50.0) * reliability)

    return {
        "speed": finish(speed),
        "offense": finish(offense),
        "shoot": finish(shoot),
        "dribble": finish(dribble),
        "passing": finish(passing),
        "defense": finish(defense),
    }


SCOREBOARD_AXIS_WEIGHTS: dict[str, dict[str, float]] = {
    "passing": {"passes": 0.7, "assists": 0.3},
    "shoot": {"goals": 0.6, "shots": 0.4},
    "offense": {"goals": 0.4, "assists": 0.3, "shots": 0.3},
    "dribble": {"dribbles": 1.0},
    "defense": {"tackles": 0.45, "interceptions": 0.35, "blocks": 0.1, "saves": 0.1},
}


def axes_from_scoreboard(comparisons: dict[str, object]) -> dict[str, float]:
    """Axes (0-100) from end-of-match lobby percentiles; only axes with data are returned.

    ``comparisons`` is ``ScoreboardResult.comparisons`` (objects or dicts with a
    ``percentile`` field). Median of the lobby maps to 50, best to ~85, worst to ~15.
    """
    def percentile_of(key: str) -> float | None:
        item = comparisons.get(key)
        if item is None:
            return None
        value = item.get("percentile") if isinstance(item, dict) else getattr(item, "percentile", None)
        return None if value is None else float(value)

    axes: dict[str, float] = {}
    for axis, weights in SCOREBOARD_AXIS_WEIGHTS.items():
        total = 0.0
        weighted = 0.0
        for key, weight in weights.items():
            p = percentile_of(key)
            if p is None:
                continue
            total += weight
            weighted += weight * p
        if total > 0.0:
            axes[axis] = _clamp(50.0 + (weighted / total - 0.5) * 70.0)
    return axes


def merge_axes(primary: dict[str, float], secondary: dict[str, float], secondary_weight: float = 0.5) -> dict[str, float]:
    """Blend two partial axis dicts; axes missing from one side take the other's value."""
    merged: dict[str, float] = {}
    weight = _clamp(secondary_weight, 0.0, 1.0)
    for key in AXIS_ORDER:
        a = primary.get(key)
        b = secondary.get(key)
        if a is None and b is None:
            continue
        if a is None:
            merged[key] = _clamp(b)  # type: ignore[arg-type]
        elif b is None:
            merged[key] = _clamp(a)
        else:
            merged[key] = _clamp(a * (1.0 - weight) + b * weight)
    return merged


def blue_lock_from_axes(axes: dict[str, float]) -> BlueLockStats:
    return BlueLockStats(**{key: round(_clamp(axes.get(key, 50.0))) for key in AXIS_ORDER})


def blend_blue_lock(career: BlueLockStats | None, live: BlueLockStats | None, live_weight: float = 0.42) -> BlueLockStats:
    if career is None and live is None:
        return BlueLockStats(speed=0, offense=0, shoot=0, dribble=0, passing=0, defense=0)
    if career is None:
        return live
    if live is None:
        return career
    weight = _clamp(live_weight, 0.0, 1.0)
    merged: dict[str, int] = {}
    for key in ("speed", "offense", "shoot", "dribble", "passing", "defense"):
        merged[key] = round(_clamp(getattr(career, key) * (1.0 - weight) + getattr(live, key) * weight))
    return BlueLockStats(**merged)


def snapshot_from_stats(player_id: str, stats: BlueLockStats, matches_analyzed: int = 0) -> PlayerGoalXSnapshot:
    score = compute_goalx_score(stats)
    return PlayerGoalXSnapshot(
        player_id=player_id,
        matches_analyzed=matches_analyzed,
        goalx_score=score,
        grade=grade_for(score),
        blue_lock=stats,
    )


def build_radar_svg(snapshot: PlayerGoalXSnapshot) -> str:
    labels = AXIS_LABELS
    values = [
        snapshot.blue_lock.speed,
        snapshot.blue_lock.defense,
        snapshot.blue_lock.passing,
        snapshot.blue_lock.dribble,
        snapshot.blue_lock.shoot,
        snapshot.blue_lock.offense,
    ]
    grades = [grade_for(value) for value in values]
    name = "".join(ch for ch in snapshot.player_id.upper() if ch.isalnum() or ch in "-_")[:14] or "PLAYER"

    center_x = 640.0
    center_y = 360.0
    radius = 188.0
    start_deg = -90.0
    step = 60.0

    def point(scale: float, idx: int) -> tuple[float, float]:
        deg = start_deg + idx * step
        rad = math.radians(deg)
        return center_x + math.cos(rad) * radius * scale, center_y + math.sin(rad) * radius * scale

    def poly(scale: float) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(scale, i) for i in range(6)))

    stat_points = [point(max(0.08, value / 100.0), i) for i, value in enumerate(values)]
    stat_poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in stat_points)

    spokes = []
    label_svg = []
    grade_svg = []
    value_svg = []
    marks = []
    for i, label in enumerate(labels):
        ox, oy = point(1.0, i)
        spokes.append(
            f'<line x1="{center_x}" y1="{center_y}" x2="{ox:.2f}" y2="{oy:.2f}" stroke="#222" stroke-width="1.2"/>'
        )
        lx, ly = point(1.46, i)
        gx, gy = point(1.22, i)
        vx, vy = stat_points[i]
        label_svg.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" fill="#111" font-size="26" font-weight="700" '
            f'font-family="Arial Black, Impact, sans-serif" text-anchor="middle">{label}</text>'
        )
        grade_svg.append(
            f'<text x="{gx:.2f}" y="{gy + 10:.2f}" fill="#111" font-size="42" font-weight="700" '
            f'font-family="Arial Black, Impact, sans-serif" text-anchor="middle">{grades[i]}</text>'
        )
        value_svg.append(
            f'<text x="{vx:.2f}" y="{vy + 4:.2f}" fill="#111" font-size="13" '
            f'font-family="Consolas, monospace" text-anchor="middle">{values[i]}</text>'
        )
        mx, my = point(1.0, i)
        marks.append(
            f'<text x="{mx:.2f}" y="{my + 4:.2f}" fill="#444" font-size="11" '
            f'font-family="Consolas, monospace" text-anchor="middle">100</text>'
        )

    pentagon = "118,86 182,110 168,186 68,186 54,110"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="720" viewBox="0 0 980 720">'
        '<rect width="100%" height="100%" fill="#f4f4f1"/>'
        '<g stroke="#d7d7d2" stroke-width="1">'
        + "".join(f'<line x1="0" y1="{y}" x2="980" y2="{y}"/>' for y in range(20, 720, 20))
        + "".join(f'<line x1="{x}" y1="0" x2="{x}" y2="720"/>' for x in range(20, 980, 20))
        + "</g>"
        '<rect x="28" y="28" width="214" height="430" fill="none" stroke="#111" stroke-width="3"/>'
        f'<text x="135" y="64" text-anchor="middle" font-size="28" font-weight="700" '
        f'font-family="Arial Black, Impact, sans-serif">{name}</text>'
        f'<polygon points="{pentagon}" fill="#ecece8" stroke="#111" stroke-width="3"/>'
        '<circle cx="118" cy="128" r="5" fill="#111"/>'
        '<circle cx="142" cy="128" r="5" fill="#111"/>'
        '<path d="M112 148 Q130 162 148 148" fill="none" stroke="#111" stroke-width="2"/>'
        '<text x="135" y="230" text-anchor="middle" font-size="14" fill="#333" '
        'font-family="Arial, sans-serif">OVERALL</text>'
        f'<text x="135" y="292" text-anchor="middle" font-size="64" font-weight="700" '
        f'font-family="Arial Black, Impact, sans-serif">{snapshot.goalx_score}</text>'
        f'<text x="135" y="372" text-anchor="middle" font-size="92" font-weight="700" '
        f'font-family="Arial Black, Impact, sans-serif">{snapshot.grade}</text>'
        f'<circle cx="{center_x}" cy="{center_y}" r="268" fill="none" stroke="#111" stroke-width="5"/>'
        f'<circle cx="{center_x}" cy="{center_y}" r="248" fill="none" stroke="#111" stroke-width="2"/>'
        f'{"".join(spokes)}'
        f'<polygon points="{poly(0.5)}" fill="none" stroke="#666" stroke-width="1"/>'
        f'<polygon points="{poly(1.0)}" fill="none" stroke="#111" stroke-width="3"/>'
        f'<polygon points="{stat_poly}" fill="#bdbdbd" fill-opacity="0.72" stroke="#111" stroke-width="2"/>'
        f'{"".join(marks)}'
        f'{"".join(grade_svg)}'
        f'{"".join(label_svg)}'
        f'{"".join(value_svg)}'
        "</svg>"
    )
