from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from goalx.kits import KitCalibrator
from goalx.vision import classify_frame, estimate_pitch_scene

STADIUM = Path(__file__).parent / "fixtures" / "rematch_stadium_frame.png"

ORANGE = (235, 96, 30)
WHITE = (238, 238, 240)
BLUE = (34, 70, 200)
RED = (220, 40, 40)


def _synthetic_match_frame(team_rgb: tuple[int, int, int], opp_rgb: tuple[int, int, int], shift: int = 0) -> Image.Image:
    image = Image.new("RGB", (860, 360), (40, 128, 48))
    draw = ImageDraw.Draw(image)
    # HUD: dark scoreboard with white digits top-left, stamina bar bottom-centre.
    draw.rectangle((10, 8, 150, 34), fill=(18, 20, 28))
    draw.rectangle((30, 14, 60, 28), fill=(245, 245, 245))
    draw.rectangle((90, 14, 110, 28), fill=(245, 245, 245))
    draw.rectangle((360, 322, 500, 332), fill=(60, 230, 120))
    # Controlled player: big kit block near centre-bottom.
    draw.rectangle((375 + shift, 170, 405 + shift, 230), fill=team_rgb)
    # Teammates and opponents out on the pitch (far = small).
    for x, y in ((230 + shift, 120), (610 - shift, 135)):
        draw.rectangle((x, y, x + 10, y + 18), fill=team_rgb)
    for x, y in ((300 + shift, 110), (470 - shift, 125), (700 - shift, 140)):
        draw.rectangle((x, y, x + 10, y + 18), fill=opp_rgb)
    return image


def test_calibrator_learns_orange_vs_white_kits() -> None:
    calibrator = KitCalibrator()
    for shift in range(0, 24, 6):
        frame = _synthetic_match_frame(ORANGE, WHITE, shift)
        assert classify_frame(frame) == "gameplay"
        calibrator.observe(frame, "gameplay")
    profile = calibrator.profile()
    assert profile is not None
    assert profile.team_label in {"orange", "red"}
    assert profile.opp_label == "white"
    assert profile.samples == 4
    assert profile.confidence > 0.2


def test_calibrator_learns_blue_vs_red_kits_in_a_new_game() -> None:
    calibrator = KitCalibrator()
    for shift in range(0, 24, 6):
        calibrator.observe(_synthetic_match_frame(BLUE, RED, shift), "gameplay")
    profile = calibrator.profile()
    assert profile is not None
    assert profile.team_label == "blue"
    assert profile.opp_label == "red"

    # The profile drives kit splitting in the scene parser.
    scene = estimate_pitch_scene(_synthetic_match_frame(BLUE, RED, 12), kit_profile=profile)
    assert len(scene.view_opponents) >= 2
    assert len(scene.view_teammates) >= 1


def test_calibrator_locks_after_enough_samples_and_resets() -> None:
    calibrator = KitCalibrator()
    for index in range(16):
        calibrator.observe(_synthetic_match_frame(ORANGE, WHITE, index % 4 * 5), "gameplay")
    profile = calibrator.profile()
    assert profile is not None
    assert profile.locked is True
    assert profile.samples >= 15
    calibrator.reset()
    assert calibrator.profile() is None


def test_calibrator_ignores_cutscene_frames() -> None:
    calibrator = KitCalibrator()
    frame = _synthetic_match_frame(ORANGE, WHITE)
    calibrator.observe(frame, "cutscene")
    calibrator.observe(frame, "menu")
    assert calibrator.profile() is None
    calibrator.observe(frame, "gameplay")
    assert calibrator.profile() is not None


def test_override_forces_labels() -> None:
    calibrator = KitCalibrator()
    profile = calibrator.override(team_label="white", opp_label="blue")
    assert profile is not None
    assert profile.team_label == "white"
    assert profile.opp_label == "blue"
    assert profile.locked is True
    # Later observations must not drift an overridden profile.
    calibrator.observe(_synthetic_match_frame(ORANGE, WHITE), "gameplay")
    after = calibrator.profile()
    assert after is not None and after.team_label == "white" and after.opp_label == "blue"


def test_stadium_fixture_calibrates_orange_team_vs_white_enemy() -> None:
    if not STADIUM.exists():
        return
    image = Image.open(STADIUM).convert("RGB")
    calibrator = KitCalibrator()
    for _ in range(3):
        calibrator.observe(image, "gameplay")
    profile = calibrator.profile()
    assert profile is not None
    assert profile.team_label in {"orange", "red"}
    assert profile.team_hue is not None and (profile.team_hue <= 42 or profile.team_hue >= 345)
    assert profile.opp_label in {"white", "grey"}
    assert profile.opp_saturation < 0.25
    assert profile.opp_brightness > 130
