from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from goalx.performance import axes_from_scoreboard, merge_axes
from goalx.scoreboard import OcrWord, ScoreboardRow, compare_rows, ocr_words, parse_scoreboard, read_scoreboard, rows_to_match_stats

HEADERS = ["PLAYER", "GOALS", "ASSISTS", "SHOTS", "PASSES", "TACKLES", "INTERCEPTIONS"]
TEAM_A = [
    ("selkea", [1, 2, 4, 18, 3, 2]),
    ("JokerStoli", [2, 0, 5, 9, 1, 0]),
    ("Star_platinum", [0, 1, 1, 12, 4, 3]),
    ("Nova", [0, 0, 2, 7, 2, 1]),
    ("Kepler", [1, 0, 3, 10, 0, 2]),
]
TEAM_B = [
    ("Ace", [1, 1, 6, 14, 2, 1]),
    ("Bolt", [0, 0, 1, 20, 5, 4]),
    ("Cyan", [2, 1, 4, 6, 1, 0]),
    ("Dex", [0, 2, 2, 11, 3, 2]),
    ("Echo", [0, 0, 0, 5, 2, 1]),
]


def _words_from_table(x_cols, header_y=100, row_h=60, gap=120) -> list[OcrWord]:
    words = [OcrWord(h, x - 40, header_y, 80, 24) for h, x in zip(HEADERS, x_cols)]
    y = header_y + row_h
    for team in (TEAM_A, TEAM_B):
        for name, stats in team:
            words.append(OcrWord(name, x_cols[0] - 40, y, 120, 24))
            for value, x in zip(stats, x_cols[1:]):
                words.append(OcrWord(str(value), x - 12, y, 24, 24))
            y += row_h
        y += gap
    return words


def test_parse_scoreboard_assigns_columns_teams_and_self() -> None:
    x_cols = [200, 700, 850, 1000, 1150, 1300, 1480]
    result = parse_scoreboard(_words_from_table(x_cols), self_name="selkea")
    assert result.detected
    assert set(result.columns) >= {"goals", "assists", "shots", "passes", "tackles", "interceptions"}
    assert len(result.rows) == 10
    me = next(row for row in result.rows if row.is_self)
    assert me.name == "selkea"
    assert me.stats["passes"] == 18
    assert sum(1 for row in result.rows if row.team == "self") == 5
    assert sum(1 for row in result.rows if row.team == "opp") == 5

    passes = result.comparisons["passes"]
    assert passes.rank == 2  # only Bolt (20) passed more
    assert passes.players == 10
    assert passes.percentile > 0.8
    assert passes.best == 20
    assert result.comparisons["goals"].rank == 3  # behind the two players on 2


def test_fuzzy_self_name_survives_ocr_slips() -> None:
    x_cols = [200, 700, 850, 1000, 1150, 1300, 1480]
    result = parse_scoreboard(_words_from_table(x_cols), self_name="Selkea_")
    assert result.self_name == "selkea"


def test_axes_from_scoreboard_rewards_lobby_rank() -> None:
    rows = [
        ScoreboardRow("me", {"passes": 18, "goals": 1, "tackles": 3, "shots": 4}, y=1, team="self", is_self=True),
        ScoreboardRow("a", {"passes": 9, "goals": 2, "tackles": 1, "shots": 5}, y=2, team="self"),
        ScoreboardRow("b", {"passes": 12, "goals": 0, "tackles": 4, "shots": 1}, y=3, team="opp"),
        ScoreboardRow("c", {"passes": 7, "goals": 0, "tackles": 2, "shots": 2}, y=4, team="opp"),
    ]
    comparisons = compare_rows(rows)
    axes = axes_from_scoreboard(comparisons)
    assert axes["passing"] > 70
    assert 40 <= axes["shoot"] <= 70
    assert "speed" not in axes
    merged = merge_axes({"speed": 60.0, "passing": 50.0}, axes, secondary_weight=0.5)
    assert merged["speed"] == 60.0
    assert merged["passing"] > 60.0


def test_rows_to_match_stats_uses_hud_pass_rate_for_attempts() -> None:
    rows = [
        ScoreboardRow("selkea", {"passes": 18, "goals": 1, "shots": 4, "tackles": 3, "interceptions": 2, "assists": 2}, y=1, team="self", is_self=True),
        ScoreboardRow("Bolt", {"passes": 20, "goals": 0, "shots": 1, "tackles": 5, "interceptions": 4, "assists": 0}, y=2, team="opp"),
    ]
    stats = rows_to_match_stats(rows, minutes=6.0, pass_rate_hint=0.75)
    me = stats[0]
    assert me.player_id == "selkea"
    assert me.completed_passes == 18
    assert me.attempted_passes == 24
    assert stats[1].attempted_passes == 20
    assert me.goals == 1 and me.tackles_won == 3 and me.interceptions == 2


def _render_results_screen() -> Image.Image:
    image = Image.new("RGB", (1720, 900), (18, 20, 28))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arialbd.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    x_cols = [120, 620, 790, 940, 1090, 1250, 1430]
    for header, x in zip(HEADERS, x_cols):
        draw.text((x, 90), header, fill=(235, 235, 235), font=font)
    y = 160
    for team in (TEAM_A, TEAM_B):
        for name, stats in team:
            draw.text((x_cols[0], y), name, fill=(240, 240, 240), font=font)
            for value, x in zip(stats, x_cols[1:]):
                draw.text((x + 30, y), str(value), fill=(240, 240, 240), font=font)
            y += 56
        y += 110
    return image


def test_windows_ocr_reads_rendered_results_screen() -> None:
    image = _render_results_screen()
    words, backend = ocr_words(image)
    if backend == "none":
        return  # OCR backend not available on this machine
    result = read_scoreboard(image, self_name="selkea")
    assert result.detected
    assert result.ocr_backend == "winocr"
    me = next((row for row in result.rows if row.is_self), None)
    assert me is not None
    assert me.stats.get("passes") == 18
    assert len(result.rows) == 10
    # Windows OCR skips single digits; the digit matcher must fill every cell.
    expected = {name: stats for name, stats in TEAM_A + TEAM_B}
    keys = ["goals", "assists", "shots", "passes", "tackles", "interceptions"]
    correct = 0
    for row in result.rows:
        truth = expected.get(row.name.replace(" ", "")) or expected.get(row.name)
        if truth is None:
            continue
        correct += sum(1 for key, value in zip(keys, truth) if row.stats.get(key) == value)
    assert correct >= 54
    assert result.comparisons["passes"].rank == 2
