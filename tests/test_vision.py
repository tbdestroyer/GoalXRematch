from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

from goalx.models import GoalXRequest
from goalx.tactics import analyze_goalx
from goalx.vision import classify_frame, estimate_pitch_scene, rematch_visibility_score

FIXTURE = Path(__file__).parent / "fixtures" / "rematch_live_frame.png"
STADIUM = Path(__file__).parent / "fixtures" / "rematch_stadium_frame.png"


def test_estimate_pitch_scene_from_synthetic_match_frame() -> None:
    image = Image.new("RGB", (640, 360), (28, 128, 42))
    draw = ImageDraw.Draw(image)
    for x, y in ((90, 170), (150, 110), (150, 230), (260, 180), (340, 120)):
        draw.ellipse((x, y, x + 18, y + 18), fill=(30, 70, 220))
    for x, y in ((500, 170), (430, 110), (430, 240), (360, 190)):
        draw.ellipse((x, y, x + 18, y + 18), fill=(220, 50, 50))
    draw.ellipse((300, 175, 314, 189), fill=(245, 245, 245))

    scene = estimate_pitch_scene(image)
    assert 0.0 <= scene.ball_x <= 100.0
    assert 0.0 <= scene.ball_y <= 100.0
    assert len(scene.teammates) >= 2
    assert scene.phase in {"attack", "defense", "transition"}

    analysis = analyze_goalx(
        GoalXRequest(
            phase=scene.phase,
            ball_x=scene.ball_x,
            ball_y=scene.ball_y,
            teammates=scene.teammates,
            opponents=scene.opponents,
        )
    )
    assert 0 <= analysis.goalx_score <= 100


def test_estimate_pitch_scene_from_rematch_teal_court() -> None:
    image = Image.new("RGB", (860, 360), (40, 110, 135))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 860, 70), fill=(170, 190, 175))
    for x, y in ((180, 210), (260, 160), (340, 230), (430, 190)):
        draw.ellipse((x, y, x + 22, y + 34), fill=(28, 70, 170))
    for x, y in ((620, 150), (700, 210), (760, 170)):
        draw.ellipse((x, y, x + 20, y + 32), fill=(235, 232, 228))
    draw.ellipse((390, 175, 404, 189), fill=(220, 70, 200))
    draw.ellipse((720, 230, 850, 355), fill=(24, 48, 78))
    draw.ellipse((748, 268, 760, 280), fill=(30, 80, 190))
    draw.ellipse((800, 250, 812, 262), fill=(30, 80, 190))
    draw.ellipse((790, 300, 800, 310), fill=(240, 240, 240))

    scene = estimate_pitch_scene(image)
    assert scene.detection_count >= 2
    assert len(scene.view_teammates) + len(scene.view_opponents) >= 2
    assert rematch_visibility_score(image) >= 0.18


def test_desktop_window_is_not_treated_as_a_match() -> None:
    image = Image.new("RGB", (860, 360), (18, 24, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((180, 40, 680, 310), fill=(248, 247, 246))
    scene = estimate_pitch_scene(image)
    assert scene.detection_count == 0
    assert scene.view_teammates == []
    assert rematch_visibility_score(image) < 0.18


def test_real_rematch_capture_finds_players() -> None:
    if not FIXTURE.exists():
        return
    image = Image.open(FIXTURE).convert("RGB")
    scene = estimate_pitch_scene(image)
    assert rematch_visibility_score(image) >= 0.18
    assert scene.detection_count >= 2 or len(scene.view_teammates) >= 1
    assert scene.confidence > 0.15


def test_stadium_match_tracks_foreground_player() -> None:
    if not STADIUM.exists():
        return
    image = Image.open(STADIUM).convert("RGB")
    scene = estimate_pitch_scene(image)
    assert rematch_visibility_score(image) >= 0.3
    assert any(player.y >= 40.0 for player in scene.view_teammates)


def test_stadium_match_detects_minimap() -> None:
    if not STADIUM.exists():
        return
    image = Image.open(STADIUM).convert("RGB")
    scene = estimate_pitch_scene(image)
    assert scene.map_detected
    assert scene.map_region.get("width", 0) > 5
    assert len(scene.map_teammates) + len(scene.map_opponents) >= 2
    assert scene.self_x is not None and scene.self_y is not None


def test_stadium_match_fuses_camera_and_minimap() -> None:
    if not STADIUM.exists():
        return
    image = Image.open(STADIUM).convert("RGB")
    scene = estimate_pitch_scene(image)
    assert scene.source == "fused"
    assert scene.fusion_matches >= 1
    players = scene.teammates + scene.opponents
    assert 2 <= len(players) <= 14
    assert len(scene.teammates) <= 7 and len(scene.opponents) <= 7
    for index, first in enumerate(players):
        for second in players[index + 1 :]:
            assert math.hypot(first.x - second.x, first.y - second.y) >= 1.0
    assert scene.view_teammates or scene.view_opponents  # raw camera kept
    assert scene.map_teammates  # raw minimap kept (frame %)
    assert scene.confidence >= 0.6


def test_stadium_frame_classifies_as_gameplay() -> None:
    if not STADIUM.exists():
        return
    image = Image.open(STADIUM).convert("RGB")
    assert classify_frame(image) == "gameplay"
    scene = estimate_pitch_scene(image)
    assert scene.gameplay_detected is True
    assert scene.frame_kind == "gameplay"


def test_letterboxed_replay_is_not_gameplay() -> None:
    image = Image.new("RGB", (860, 360), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 45, 860, 315), fill=(36, 132, 48))
    for x, y in ((200, 150), (420, 200), (600, 130)):
        draw.ellipse((x, y, x + 22, y + 40), fill=(240, 240, 240))
    kind = classify_frame(image)
    assert kind in {"cutscene", "menu"}
    scene = estimate_pitch_scene(image)
    assert scene.gameplay_detected is False


def test_dark_menu_is_not_gameplay() -> None:
    image = Image.new("RGB", (860, 360), (8, 10, 14))
    draw = ImageDraw.Draw(image)
    draw.rectangle((300, 150, 560, 190), fill=(40, 44, 60))
    assert classify_frame(image) == "menu"
    scene = estimate_pitch_scene(image)
    assert scene.gameplay_detected is False
    assert scene.frame_kind == "menu"
