from __future__ import annotations

from goalx.models import BlueLockStats, LivePitchScene, PlayerPosition
from goalx.performance import blend_blue_lock, build_radar_svg, compute_blue_lock_stats, live_axes_from_capture, snapshot_from_stats


def test_shoot_axis_tracks_shots_more_than_passing() -> None:
    totals = {
        "matches": 1.0,
        "minutes": 90.0,
        "shots": 8.0,
        "shots_on_target": 5.0,
        "goals": 3.0,
        "assists": 0.0,
        "key_passes": 1.0,
        "successful_dribbles": 1.0,
        "dribble_attempts": 4.0,
        "completed_passes": 12.0,
        "attempted_passes": 20.0,
        "tackles_won": 0.0,
        "interceptions": 0.0,
        "duels_won": 1.0,
        "duels_total": 4.0,
        "sprints": 10.0,
        "distance_m": 8000.0,
    }
    stats = compute_blue_lock_stats(totals)
    assert stats.shoot > stats.passing
    assert stats.offense >= 40


def test_live_attacking_scene_raises_offense_and_shoot() -> None:
    scene = LivePitchScene(
        phase="attack",
        ball_x=82.0,
        ball_y=48.0,
        teammates=[
            PlayerPosition(player_id="t1", role="fw", x=78, y=40),
            PlayerPosition(player_id="t2", role="mf", x=62, y=55),
            PlayerPosition(player_id="t3", role="df", x=40, y=50),
        ],
        opponents=[],
        confidence=0.6,
        detection_count=3,
    )
    live = live_axes_from_capture(0.12, 0.5, scene)
    assert live["offense"] > 60
    assert live["shoot"] > 55


def test_radar_svg_matches_blue_lock_axes() -> None:
    snapshot = snapshot_from_stats(
        "isagi",
        BlueLockStats(speed=75, offense=87, shoot=77, dribble=65, passing=69, defense=72),
    )
    svg = build_radar_svg(snapshot)
    assert "SPEED" in svg
    assert "DEFENSE" in svg
    assert "PASS" in svg
    assert "DRIBBLE" in svg
    assert "SHOOT" in svg
    assert "OFFENSE" in svg
    assert snapshot.grade == "B"
    assert ">B<" in svg or ">76<" in svg


def test_blend_moves_career_stats_toward_live_capture() -> None:
    career = BlueLockStats(speed=50, offense=50, shoot=50, dribble=50, passing=50, defense=50)
    live = BlueLockStats(speed=80, offense=80, shoot=80, dribble=80, passing=80, defense=80)
    blended = blend_blue_lock(career, live, live_weight=0.5)
    assert blended.speed == 65
