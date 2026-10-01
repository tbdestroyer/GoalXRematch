from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from goalx.models import MatchIngestRequest, MatchPlayerStats

CSV_REQUIRED_COLUMNS = (
    "match_id",
    "played_at",
    "player_id",
    "minutes",
    "shots",
    "shots_on_target",
    "goals",
    "assists",
    "key_passes",
    "successful_dribbles",
    "dribble_attempts",
    "completed_passes",
    "attempted_passes",
    "tackles_won",
    "interceptions",
    "duels_won",
    "duels_total",
    "sprints",
    "distance_m",
)


def parse_match_file(file_path: Path) -> list[MatchIngestRequest]:
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return [parse_json_match(file_path)]
    if suffix == ".csv":
        return parse_csv_matches(file_path)
    raise RuntimeError(f"Unsupported export format: {file_path.name}")


def parse_json_match(file_path: Path) -> MatchIngestRequest:
    content = file_path.read_text(encoding="utf-8")
    try:
        payload = MatchIngestRequest.model_validate(json.loads(content))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in file: {file_path.name}") from exc
    except ValidationError as exc:
        raise RuntimeError(f"Invalid match schema in file: {file_path.name}") from exc
    return payload


def parse_csv_matches(file_path: Path) -> list[MatchIngestRequest]:
    content = file_path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(content.splitlines()))
    if not rows:
        return []
    missing = [column for column in CSV_REQUIRED_COLUMNS if column not in rows[0]]
    if missing:
        raise RuntimeError(
            f"CSV file missing required columns ({', '.join(missing)}): {file_path.name}"
        )

    grouped: dict[str, dict[str, object]] = defaultdict(lambda: {"played_at": None, "players": []})
    for row_index, row in enumerate(rows, start=2):
        match_id = str(row["match_id"]).strip()
        if not match_id:
            raise RuntimeError(f"CSV row with empty match_id in file: {file_path.name}")
        try:
            played_at_raw = str(row["played_at"]).strip()
            played_at = datetime.fromisoformat(played_at_raw) if played_at_raw else None
            player = MatchPlayerStats(
                player_id=str(row["player_id"]).strip(),
                minutes=float(row["minutes"]),
                shots=int(row["shots"]),
                shots_on_target=int(row["shots_on_target"]),
                goals=int(row["goals"]),
                assists=int(row["assists"]),
                key_passes=int(row["key_passes"]),
                successful_dribbles=int(row["successful_dribbles"]),
                dribble_attempts=int(row["dribble_attempts"]),
                completed_passes=int(row["completed_passes"]),
                attempted_passes=int(row["attempted_passes"]),
                tackles_won=int(row["tackles_won"]),
                interceptions=int(row["interceptions"]),
                duels_won=int(row["duels_won"]),
                duels_total=int(row["duels_total"]),
                sprints=int(row["sprints"]),
                distance_m=float(row["distance_m"]),
            )
        except (ValueError, ValidationError) as exc:
            raise RuntimeError(
                f"Invalid CSV values at row {row_index} in file: {file_path.name}"
            ) from exc
        grouped[match_id]["players"].append(player)
        if played_at is not None:
            grouped[match_id]["played_at"] = played_at

    matches: list[MatchIngestRequest] = []
    for match_id, grouped_data in grouped.items():
        matches.append(
            MatchIngestRequest(
                match_id=match_id,
                played_at=grouped_data["played_at"],
                players=grouped_data["players"],
            )
        )
    return matches
