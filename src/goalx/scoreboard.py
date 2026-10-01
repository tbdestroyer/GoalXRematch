"""End-of-match scoreboard reader.

Rematch shows a per-player stats table when a match ends. We OCR the frame
(Windows built-in OCR via ``winocr`` when available), locate the stat header
row, assign every number on each player row to the nearest header column, pick
out the local player's row, and compare it against everyone else in the lobby.

The parsed rows can be converted to ``MatchPlayerStats`` so they feed the same
career totals the exported match files do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from PIL import Image

from goalx.models import MatchPlayerStats

# Canonical stat -> header spellings seen on the Rematch results screen (lower-case, fuzzy).
HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "goals": ("goals", "goal", "buts", "but"),
    "assists": ("assists", "assist", "decisives"),
    "shots": ("shots", "shot", "tirs", "tir"),
    "passes": ("passes", "pass", "passe"),
    "tackles": ("tackles", "tackle", "tacles", "tacle"),
    "interceptions": ("interceptions", "interception", "intercepts", "intercept"),
    "saves": ("saves", "save", "arrets", "arret"),
    "dribbles": ("dribbles", "dribble"),
    "blocks": ("blocks", "block"),
    "score": ("score", "rating", "mvp", "points", "pts"),
}

MIN_HEADER_MATCHES = 3
NUMERIC_RE = re.compile(r"^[0-9]{1,3}(?:[.,][0-9]{1,2})?%?$")


@dataclass
class OcrWord:
    text: str
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class ScoreboardRow:
    name: str
    stats: dict[str, float]
    y: float
    team: str = "unknown"  # "self" | "opp" | "unknown"
    is_self: bool = False

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "team": self.team, "is_self": self.is_self, "stats": dict(self.stats)}


@dataclass
class StatComparison:
    value: float
    lobby_avg: float
    team_avg: float
    opp_avg: float
    rank: int
    players: int
    percentile: float
    best: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "value": self.value,
            "lobby_avg": round(self.lobby_avg, 2),
            "team_avg": round(self.team_avg, 2),
            "opp_avg": round(self.opp_avg, 2),
            "rank": self.rank,
            "players": self.players,
            "percentile": round(self.percentile, 3),
            "best": self.best,
        }


@dataclass
class ScoreboardResult:
    detected: bool
    columns: list[str] = field(default_factory=list)
    rows: list[ScoreboardRow] = field(default_factory=list)
    self_name: str | None = None
    comparisons: dict[str, StatComparison] = field(default_factory=dict)
    ocr_backend: str = "none"
    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "columns": list(self.columns),
            "rows": [row.as_dict() for row in self.rows],
            "self_name": self.self_name,
            "comparisons": {key: value.as_dict() for key, value in self.comparisons.items()},
            "ocr_backend": self.ocr_backend,
            "captured_at": self.captured_at,
        }


# ------------------------------------------------------------------------ OCR
def ocr_words(image: Image.Image) -> tuple[list[OcrWord], str]:
    """Return OCR words with pixel boxes and the backend name ("winocr" | "none")."""
    try:
        import winocr  # type: ignore
    except Exception:  # pragma: no cover - platform dependent
        return [], "none"
    rgb = image.convert("RGB")
    try:
        result = winocr.recognize_pil_sync(rgb, "en")
    except Exception:  # pragma: no cover - platform dependent
        return [], "none"
    def get(obj: object, key: str, default: object = None) -> object:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    words: list[OcrWord] = []
    for line in get(result, "lines", []) or []:
        for word in get(line, "words", []) or []:
            rect = get(word, "bounding_rect")
            if rect is None:
                continue
            words.append(
                OcrWord(
                    str(get(word, "text", "")),
                    float(get(rect, "x", 0.0)),
                    float(get(rect, "y", 0.0)),
                    float(get(rect, "width", 0.0)),
                    float(get(rect, "height", 0.0)),
                )
            )
    return words, "winocr"


# ---------------------------------------------------------------------- parse
def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9% ]", "", text.lower().replace("é", "e").replace("ê", "e")).strip()


def _header_key(text: str) -> str | None:
    norm = _norm(text)
    if not norm or len(norm) < 3:
        return None
    for key, aliases in HEADER_ALIASES.items():
        if norm in aliases:
            return key
    for key, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            if len(alias) >= 4 and norm.startswith(alias):
                return key
            if len(norm) >= 5 and alias.startswith(norm):
                return key
    return None


def _to_number(text: str) -> float | None:
    cleaned = text.strip().replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
    if not NUMERIC_RE.match(cleaned):
        return None
    cleaned = cleaned.rstrip("%").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _cluster_lines(words: list[OcrWord]) -> list[list[OcrWord]]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: w.cy)
    lines: list[list[OcrWord]] = [[ordered[0]]]
    for word in ordered[1:]:
        current = lines[-1]
        avg_cy = sum(w.cy for w in current) / len(current)
        avg_h = sum(w.h for w in current) / len(current)
        if abs(word.cy - avg_cy) <= max(6.0, avg_h * 0.6):
            current.append(word)
        else:
            lines.append([word])
    for line in lines:
        line.sort(key=lambda w: w.x)
    return lines


def parse_scoreboard(
    words: list[OcrWord],
    self_name: str | None = None,
    image: Image.Image | None = None,
) -> ScoreboardResult:
    """Parse OCR words into a scoreboard.

    When ``image`` is given, numeric cells OCR missed (Windows OCR drops isolated
    single digits) are read with the template ``DigitMatcher``, which first learns
    the game's digit shapes from the numbers OCR did recognise.
    """
    result = ScoreboardResult(detected=False, self_name=self_name)
    lines = _cluster_lines(words)
    if not lines:
        return result

    # Header = the line with the most recognised stat labels.
    best_header: list[OcrWord] | None = None
    best_columns: dict[str, float] = {}
    for line in lines:
        columns: dict[str, float] = {}
        for word in line:
            key = _header_key(word.text)
            if key and key not in columns:
                columns[key] = word.cx
        if len(columns) > len(best_columns):
            best_header, best_columns = line, columns
    if best_header is None or len(best_columns) < MIN_HEADER_MATCHES:
        return result

    header_y = sum(w.cy for w in best_header) / len(best_header)
    column_items = sorted(best_columns.items(), key=lambda item: item[1])
    column_xs = [x for _, x in column_items]
    gaps = [b - a for a, b in zip(column_xs, column_xs[1:])]
    tolerance = (min(gaps) / 2.0) if gaps else 60.0

    header_h = sum(w.h for w in best_header) / len(best_header)
    matcher = None
    if image is not None:
        from goalx.digits import DigitMatcher

        matcher = DigitMatcher()

    rows: list[ScoreboardRow] = []
    numeric_cells: list[tuple[str, OcrWord]] = []
    for line in lines:
        line_y = sum(w.cy for w in line) / len(line)
        if line_y <= header_y + header_h * 0.5:
            continue
        line_h = sum(w.h for w in line) / len(line)
        stats: dict[str, float] = {}
        name_parts: list[str] = []
        for word in line:
            number = _to_number(word.text)
            if number is None:
                if word.cx < column_xs[0] - tolerance * 0.5 and _header_key(word.text) is None:
                    name_parts.append(word.text)
                continue
            key, cx = min(column_items, key=lambda item: abs(item[1] - word.cx))
            if abs(cx - word.cx) <= tolerance and key not in stats:
                stats[key] = number
                numeric_cells.append((word.text, word))
        has_name = bool(name_parts)
        if len(stats) >= 2 or (has_name and matcher is not None):
            name = " ".join(name_parts).strip() or f"player_{len(rows) + 1}"
            row = ScoreboardRow(name=name, stats=stats, y=line_y)
            row_h = max(line_h, header_h)
            rows.append(row)
            row._h = row_h  # type: ignore[attr-defined]

    if matcher is not None and image is not None and rows:
        # Teach the matcher the in-game digit font from numbers OCR already read.
        for text, word in numeric_cells:
            if text.strip().isdigit():
                pad = word.h * 0.3
                matcher.learn(text.strip(), image.crop((int(word.x - pad), int(word.y - pad), int(word.x + word.w + pad), int(word.y + word.h + pad))))
        # Value columns: where OCR'd numbers actually sit (header text may be off-centre).
        value_cx: dict[str, float] = {}
        per_col: dict[str, list[float]] = {}
        for text, word in numeric_cells:
            key, cx = min(column_items, key=lambda item: abs(item[1] - word.cx))
            per_col.setdefault(key, []).append(word.cx)
        for key, cx in column_items:
            xs = sorted(per_col.get(key, []))
            value_cx[key] = xs[len(xs) // 2] if xs else cx
        ordered = sorted(column_items, key=lambda item: value_cx[item[0]])
        bounds: dict[str, tuple[float, float]] = {}
        for index, (key, _) in enumerate(ordered):
            cx = value_cx[key]
            left = (value_cx[ordered[index - 1][0]] + cx) / 2.0 if index > 0 else cx - tolerance
            right = (value_cx[ordered[index + 1][0]] + cx) / 2.0 if index + 1 < len(ordered) else cx + tolerance
            # Header words can extend far right of the values; don't let a cell swallow them.
            bounds[key] = (left + 2.0, right - 2.0)
        for row in rows:
            row_h = getattr(row, "_h", header_h)
            for key, cx in column_items:
                if key in row.stats:
                    continue
                left, right = bounds[key]
                box = (int(left), int(row.y - row_h * 0.9), int(right), int(row.y + row_h * 0.9))
                cell = image.crop(box)
                read = matcher.read(cell)
                if read is not None and read.value is not None:
                    row.stats[key] = float(read.value)
        rows = [row for row in rows if len(row.stats) >= 2]

    if len(rows) < 2:
        return result

    result.detected = True
    result.columns = [key for key, _ in column_items]
    result.rows = rows
    _assign_teams_and_self(result, self_name)
    result.comparisons = compare_rows(result.rows)
    return result


def _assign_teams_and_self(result: ScoreboardResult, self_name: str | None) -> None:
    rows = result.rows
    # Teams: the biggest vertical gap between consecutive rows separates the two tables.
    if len(rows) >= 4:
        ordered = sorted(rows, key=lambda r: r.y)
        gaps = [(b.y - a.y, index) for index, (a, b) in enumerate(zip(ordered, ordered[1:]))]
        median_gap = sorted(g for g, _ in gaps)[len(gaps) // 2]
        biggest, split_index = max(gaps)
        if biggest >= median_gap * 1.6:
            first = {id(r) for r in ordered[: split_index + 1]}
            for row in rows:
                row.team = "A" if id(row) in first else "B"
    # Self: fuzzy name match on the linked account / HUD name.
    me: ScoreboardRow | None = None
    if self_name:
        target = _norm(self_name).replace(" ", "")
        scored = []
        for row in rows:
            candidate = _norm(row.name).replace(" ", "")
            if not candidate or not target:
                continue
            if candidate == target:
                score = 1.0
            elif target in candidate or candidate in target:
                score = 0.8
            else:
                score = _similarity(candidate, target)
            scored.append((score, row))
        if scored:
            score, row = max(scored, key=lambda item: item[0])
            if score >= 0.6:
                me = row
    if me is not None:
        me.is_self = True
        result.self_name = me.name
        my_team = me.team
        for row in rows:
            if my_team in {"A", "B"}:
                row.team = "self" if row.team == my_team else "opp"
            else:
                row.team = "unknown"
        me.team = "self"


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    # Dice coefficient on character bigrams; cheap and tolerant to OCR slips.
    def bigrams(s: str) -> list[str]:
        return [s[i : i + 2] for i in range(len(s) - 1)] or [s]

    ba, bb = bigrams(a), bigrams(b)
    overlap = 0
    pool = list(bb)
    for gram in ba:
        if gram in pool:
            pool.remove(gram)
            overlap += 1
    return 2.0 * overlap / (len(ba) + len(bb))


def compare_rows(rows: list[ScoreboardRow]) -> dict[str, StatComparison]:
    me = next((row for row in rows if row.is_self), None)
    if me is None or len(rows) < 2:
        return {}
    comparisons: dict[str, StatComparison] = {}
    for key, value in me.stats.items():
        values = [row.stats[key] for row in rows if key in row.stats]
        if len(values) < 2:
            continue
        team_vals = [row.stats[key] for row in rows if key in row.stats and row.team == "self"]
        opp_vals = [row.stats[key] for row in rows if key in row.stats and row.team == "opp"]
        higher = sum(1 for v in values if v > value)
        lower = sum(1 for v in values if v < value)
        rank = higher + 1
        percentile = (lower + 0.5 * (len(values) - higher - lower - 1)) / max(1, len(values) - 1)
        comparisons[key] = StatComparison(
            value=value,
            lobby_avg=sum(values) / len(values),
            team_avg=sum(team_vals) / len(team_vals) if team_vals else sum(values) / len(values),
            opp_avg=sum(opp_vals) / len(opp_vals) if opp_vals else sum(values) / len(values),
            rank=rank,
            players=len(values),
            percentile=max(0.0, min(1.0, percentile)),
            best=max(values),
        )
    return comparisons


def read_scoreboard(image: Image.Image, self_name: str | None = None) -> ScoreboardResult:
    words, backend = ocr_words(image)
    result = parse_scoreboard(words, self_name=self_name, image=image)
    result.ocr_backend = backend
    return result


# --------------------------------------------------------------------- ingest
def rows_to_match_stats(
    rows: list[ScoreboardRow],
    minutes: float,
    pass_rate_hint: float | None = None,
) -> list[MatchPlayerStats]:
    """Convert scoreboard rows into ``MatchPlayerStats`` for career storage.

    The results screen lists completed passes; attempted passes are estimated from
    the live HUD pass rate when known, otherwise assumed equal to completed.
    """
    minutes = max(1.0, min(130.0, minutes))
    players: list[MatchPlayerStats] = []
    for row in rows:
        passes = int(row.stats.get("passes", 0))
        rate = pass_rate_hint if pass_rate_hint and 0.2 <= pass_rate_hint <= 1.0 and row.is_self else None
        attempted = int(round(passes / rate)) if rate else passes
        shots = int(row.stats.get("shots", 0))
        goals = int(row.stats.get("goals", 0))
        dribbles = int(row.stats.get("dribbles", 0))
        tackles = int(row.stats.get("tackles", 0))
        interceptions = int(row.stats.get("interceptions", 0))
        player_id = re.sub(r"[^A-Za-z0-9_\-]", "_", row.name)[:32] or "player"
        players.append(
            MatchPlayerStats(
                player_id=player_id,
                minutes=minutes,
                shots=min(40, shots),
                shots_on_target=min(40, max(goals, int(round(shots * 0.5)))),
                goals=min(20, goals),
                assists=min(20, int(row.stats.get("assists", 0))),
                key_passes=min(80, int(row.stats.get("assists", 0))),
                successful_dribbles=min(80, dribbles),
                dribble_attempts=min(120, max(dribbles, int(round(dribbles * 1.4)))),
                completed_passes=min(400, passes),
                attempted_passes=min(500, max(passes, attempted)),
                tackles_won=min(80, tackles),
                interceptions=min(80, interceptions),
                duels_won=min(80, tackles + interceptions),
                duels_total=min(120, int(round((tackles + interceptions) * 1.5))),
                sprints=0,
                distance_m=0.0,
            )
        )
    return players
