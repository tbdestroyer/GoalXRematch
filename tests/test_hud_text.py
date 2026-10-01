from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from goalx.hud_text import PopupTemplateStore, detect_action_popup, signature_similarity

GO_FRAME = Path(__file__).parent.parent / "data" / "live-capture" / "frame_20261001_052703_560713.png"
STADIUM = Path(__file__).parent / "fixtures" / "rematch_stadium_frame.png"


def _popup_frame(text: str, size=(1720, 720)) -> Image.Image:
    image = Image.new("RGB", size, (60, 140, 70))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arialbd.ttf", 150)
    except OSError:
        font = ImageFont.load_default()
    # Centre band, pale cyan block letters with spacing so glyphs separate.
    x = size[0] // 2 - 60 * len(text)
    for ch in text:
        draw.text((x, size[1] * 0.3), ch, fill=(214, 250, 248), font=font)
        x += 120
    return image


def test_synthetic_pass_popup_is_detected_with_four_glyphs() -> None:
    detection = detect_action_popup(_popup_frame("PASS"))
    assert detection is not None
    assert detection.glyphs == 4
    assert detection.label == "pass"
    assert 20.0 <= detection.bbox["x"] <= 60.0


def test_plain_pitch_frame_has_no_popup() -> None:
    assert detect_action_popup(Image.new("RGB", (860, 360), (40, 120, 50))) is None
    if STADIUM.exists():
        assert detect_action_popup(Image.open(STADIUM).convert("RGB")) is None


def test_real_go_popup_detected() -> None:
    if not GO_FRAME.exists():
        return
    detection = detect_action_popup(Image.open(GO_FRAME).convert("RGB"))
    assert detection is not None
    assert detection.glyphs == 3
    assert 40.0 <= detection.bbox["x"] <= 50.0


def test_template_store_learns_and_classifies(tmp_path: Path) -> None:
    store = PopupTemplateStore(tmp_path / "templates.json")
    pass_det = detect_action_popup(_popup_frame("PASS"))
    goal_det = detect_action_popup(_popup_frame("GOAL"))
    assert pass_det is not None and goal_det is not None
    store.learn("pass", pass_det)
    store.learn("goal", goal_det)
    assert store.labels() == {"pass": 1, "goal": 1}

    again = detect_action_popup(_popup_frame("PASS"))
    assert again is not None
    classified = store.classify(again)
    assert classified.label == "pass"
    assert classified.confidence >= 0.78
    assert signature_similarity(pass_det.signature, goal_det.signature) < signature_similarity(pass_det.signature, again.signature)

    reloaded = PopupTemplateStore(tmp_path / "templates.json")
    assert reloaded.labels() == {"pass": 1, "goal": 1}
