from __future__ import annotations

from goalx.models import LivePitchScene, PlayerPosition
from goalx.performance import axes_from_tracking, blue_lock_from_axes
from goalx.tracking import MatchTracker

DT = 0.25  # seconds between frames (4 fps capture)
UNITS_PER_M_X = 100.0 / 60.0
UNITS_PER_M_Y = 100.0 / 36.0


def _scene(ball, me, mates=(), opps=(), phase="transition") -> LivePitchScene:
    teammates = [PlayerPosition(player_id="me", role="fw", x=me[0], y=me[1])]
    teammates += [PlayerPosition(player_id=f"t{i}", role="mf", x=x, y=y) for i, (x, y) in enumerate(mates)]
    opponents = [PlayerPosition(player_id=f"o{i}", role="df", x=x, y=y) for i, (x, y) in enumerate(opps)]
    return LivePitchScene(
        phase=phase,
        ball_x=max(0.0, min(100.0, ball[0])),
        ball_y=max(0.0, min(100.0, ball[1])),
        teammates=teammates,
        opponents=opponents,
        confidence=0.8,
        detection_count=len(teammates) + len(opponents),
    )


def _run(tracker: MatchTracker, frames, t0: float = 0.0) -> dict[str, float]:
    metrics = {}
    for index, (ball, me, mates, opps) in enumerate(frames):
        metrics = tracker.observe(_scene(ball, me, mates, opps), t0 + index * DT, self_xy=me)
    return metrics


def test_pace_duel_counts_when_outrunning_defender_with_ball() -> None:
    tracker = MatchTracker()
    frames = []
    me_x = 30.0
    opp_x = 32.0
    me_step = 7.0 * DT * UNITS_PER_M_X  # 7 m/s
    opp_step = 3.5 * DT * UNITS_PER_M_X
    for _ in range(14):
        frames.append(((me_x + 0.5, 50.0), (me_x, 50.0), [(10.0, 50.0)], [(opp_x, 50.5)]))
        me_x += me_step
        opp_x += opp_step
    metrics = _run(tracker, frames)
    assert metrics["pace_duels_won"] >= 1
    assert metrics["pace_duel_rate"] == 1.0
    assert metrics["take_ons"] == 0
    assert metrics["top_speed_mps"] > 6.0
    assert metrics["carry_distance_per_min_m"] > 0


def test_slow_take_on_is_dribble_not_pace() -> None:
    tracker = MatchTracker()
    frames = []
    me_x = 60.0
    opp_x = 61.5
    me_step = 2.6 * DT * UNITS_PER_M_X  # jog past a standing defender
    for _ in range(14):
        frames.append(((me_x + 0.4, 50.0), (me_x, 50.0), [], [(opp_x, 50.5)]))
        me_x += me_step
    metrics = _run(tracker, frames)
    assert metrics["take_ons_won"] >= 1
    assert metrics["pace_duels"] == 0


def test_pass_and_shot_events() -> None:
    tracker = MatchTracker()
    frames = []
    # Hold the ball for a moment, then pass forward to a teammate 20 units up the pitch.
    for _ in range(4):
        frames.append(((40.5, 50.0), (40.0, 50.0), [(60.0, 52.0)], [(80.0, 48.0)]))
    frames.append(((50.0, 51.0), (40.0, 50.0), [(60.0, 52.0)], [(80.0, 48.0)]))
    for _ in range(4):
        frames.append(((60.3, 52.0), (40.0, 50.0), [(60.0, 52.0)], [(80.0, 48.0)]))
    metrics = _run(tracker, frames)
    assert metrics["passes"] == 1
    assert metrics["passes_completed"] == 1
    assert metrics["forward_pass_share"] == 1.0

    # Now we receive it back near the box and shoot fast toward goal.
    tracker2 = MatchTracker()
    frames2 = []
    for _ in range(4):
        frames2.append(((82.5, 50.0), (82.0, 50.0), [], [(95.0, 50.0)]))
    frames2.append(((88.0, 50.5), (82.0, 50.0), [], [(95.0, 50.0)]))  # ~13 m/s
    frames2.append(((94.0, 51.0), (82.0, 50.0), [], [(95.0, 50.0)]))
    frames2.append(((99.0, 51.5), (82.0, 50.0), [], [(95.0, 50.0)]))
    metrics2 = _run(tracker2, frames2)
    assert metrics2["shots"] == 1
    assert metrics2["shots_on_target"] == 1
    assert metrics2["losses"] == 0


def test_hud_pass_popup_confirms_kicks_toward_teammates() -> None:
    tracker = MatchTracker()
    frames = []
    # Kick 1: ball played toward teammate at (60, 52) -> game says PASS.
    for _ in range(3):
        frames.append(((40.5, 50.0), (40.0, 50.0), [(60.0, 52.0)], [(80.0, 48.0)]))
    frames.append(((50.0, 51.0), (40.0, 50.0), [(60.0, 52.0)], [(80.0, 48.0)]))
    for _ in range(3):
        frames.append(((60.3, 52.0), (40.0, 50.0), [(60.0, 52.0)], [(80.0, 48.0)]))
    t = 0.0
    for ball, me, mates, opps in frames:
        tracker.observe(_scene(ball, me, mates, opps), t, self_xy=me)
        t += DT
    tracker.record_popup("pass", t)
    m = tracker.metrics()
    assert m["pass_kicks"] == 1
    assert m["pass_kicks_confirmed"] == 1
    assert m["pass_rate_hud"] == 1.0

    # Kick 2: played toward a teammate but no PASS popup -> unconfirmed attempt.
    frames2 = []
    for _ in range(3):
        frames2.append(((40.5, 50.0), (40.0, 50.0), [(60.0, 52.0)], [(55.0, 51.0)]))
    frames2.append(((48.0, 50.5), (40.0, 50.0), [(60.0, 52.0)], [(55.0, 51.0)]))
    for _ in range(3):
        frames2.append(((55.2, 51.0), (40.0, 50.0), [(60.0, 52.0)], [(55.0, 51.0)]))
    t += 4.0
    for ball, me, mates, opps in frames2:
        tracker.observe(_scene(ball, me, mates, opps), t, self_xy=me)
        t += DT
    m = tracker.metrics()
    assert m["pass_kicks"] == 2
    assert m["pass_kicks_confirmed"] == 1
    assert m["pass_rate_hud"] == 0.5
    assert m["losses"] == 1

    # A PASS popup with no kick seen (sparse frames) still counts as a completed pass.
    tracker.record_popup("pass", t + 10.0)
    m = tracker.metrics()
    assert m["pass_kicks"] == 3 and m["pass_kicks_confirmed"] == 2
    assert m["popups"]["pass"] == 2

    hud = {"gameplay_minutes": 5.0, "tracking_quality": 1.0, "pass_kicks": 10, "pass_kicks_confirmed": 9, "pass_rate_hud": 0.9, "passes": 4, "pass_rate": 0.5}
    geo = {"gameplay_minutes": 5.0, "tracking_quality": 1.0, "passes": 4, "pass_rate": 0.5}
    assert axes_from_tracking(hud)["passing"] > axes_from_tracking(geo)["passing"] + 8


def test_interception_and_tackle_are_recoveries() -> None:
    tracker = MatchTracker()
    frames = []
    # Opponent carries, we are far away; ball squirts to us -> interception.
    for _ in range(3):
        frames.append(((30.5, 50.0), (45.0, 50.0), [], [(30.0, 50.0)]))
    frames.append(((38.0, 50.0), (45.0, 50.0), [], [(30.0, 50.0)]))
    for _ in range(3):
        frames.append(((45.3, 50.0), (45.0, 50.0), [], [(30.0, 50.0)]))
    metrics = _run(tracker, frames)
    assert metrics["recoveries"] == 1
    assert metrics["interceptions"] == 1

    tracker2 = MatchTracker()
    frames2 = []
    for _ in range(3):
        frames2.append(((30.5, 50.0), (33.0, 50.0), [], [(30.0, 50.0)]))
    for _ in range(3):
        frames2.append(((33.1, 50.0), (33.0, 50.0), [], [(30.0, 50.0)]))
    metrics2 = _run(tracker2, frames2)
    assert metrics2["recoveries"] == 1
    assert metrics2["tackles"] == 1
    assert metrics2["goal_side_share"] < 0.5  # we were in front of the ball, not goal side


def test_sparse_capture_disables_motion_metrics() -> None:
    tracker = MatchTracker()
    me_x = 30.0
    for index in range(6):
        tracker.observe(_scene((me_x + 0.5, 50.0), (me_x, 50.0), [], [(50.0, 50.0)]), index * 2.2, self_xy=(me_x, 50.0))
        me_x += 10.0
    metrics = tracker.metrics()
    assert metrics["tracking_quality"] == 0.0
    assert metrics["top_speed_mps"] == 0.0
    axes = axes_from_tracking(metrics)
    assert 40.0 <= axes["speed"] <= 60.0


def test_cutscene_frames_do_not_feed_tracking() -> None:
    tracker = MatchTracker()
    for index in range(6):
        tracker.observe(_scene((50.0, 50.0), (10.0 + index * 15.0, 50.0), [], []), index * DT, self_xy=(10.0 + index * 15.0, 50.0), gameplay=False)
    assert tracker.metrics()["gameplay_minutes"] == 0.0
    assert tracker.counters.frames == 0


def test_axes_reward_the_right_criteria() -> None:
    minutes = 6.0
    striker = {
        "gameplay_minutes": minutes, "tracking_quality": 1.0,
        "top_speed_mps": 7.6, "sprint_share": 0.4, "distance_per_min_m": 110.0,
        "pace_duels": 8, "pace_duels_won": 7, "pace_duel_rate": 0.875,
        "passes": 6, "passes_completed": 4, "pass_rate": 0.67, "forward_pass_share": 0.5, "key_passes": 1,
        "shots": 6, "shots_on_target": 4, "on_target_rate": 0.67, "avg_shot_quality": 0.6, "goals": 2,
        "carries": 10, "carry_distance_per_min_m": 40.0, "take_ons": 6, "take_ons_won": 4, "take_on_rate": 0.67, "retention": 0.7,
        "involvement": 0.5, "attacking_third_share": 0.55, "forward_runs_per_min": 1.2, "box_touches": 6,
        "recoveries": 1, "goal_side_share": 0.3, "avg_closing_speed_mps": 0.2, "avg_nearest_mate_m": 18.0,
    }
    anchor = {
        "gameplay_minutes": minutes, "tracking_quality": 1.0,
        "top_speed_mps": 5.2, "sprint_share": 0.12, "distance_per_min_m": 80.0,
        "pace_duels": 1, "pace_duels_won": 0, "pace_duel_rate": 0.0,
        "passes": 20, "passes_completed": 18, "pass_rate": 0.9, "forward_pass_share": 0.4, "key_passes": 2,
        "shots": 0, "shots_on_target": 0, "on_target_rate": 0.0, "avg_shot_quality": 0.0, "goals": 0,
        "carries": 6, "carry_distance_per_min_m": 10.0, "take_ons": 1, "take_ons_won": 0, "take_on_rate": 0.0, "retention": 0.9,
        "involvement": 0.2, "attacking_third_share": 0.1, "forward_runs_per_min": 0.1, "box_touches": 0,
        "recoveries": 7, "goal_side_share": 0.9, "avg_closing_speed_mps": 1.8, "avg_nearest_mate_m": 10.0,
    }
    striker_axes = axes_from_tracking(striker)
    anchor_axes = axes_from_tracking(anchor)
    assert striker_axes["speed"] > anchor_axes["speed"] + 15
    assert striker_axes["shoot"] > anchor_axes["shoot"] + 20
    assert striker_axes["offense"] > anchor_axes["offense"] + 15
    assert striker_axes["dribble"] > anchor_axes["dribble"] + 10
    assert anchor_axes["passing"] > striker_axes["passing"] + 10
    assert anchor_axes["defense"] > striker_axes["defense"] + 20
    stats = blue_lock_from_axes(striker_axes)
    assert 0 <= stats.speed <= 100


def test_tiny_sample_stays_near_neutral() -> None:
    axes = axes_from_tracking({"gameplay_minutes": 0.2, "tracking_quality": 1.0, "shots": 1, "shots_on_target": 1, "on_target_rate": 1.0, "goals": 1})
    assert 45.0 <= axes["shoot"] <= 62.0
