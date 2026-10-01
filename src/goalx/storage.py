from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from goalx.models import MatchPlayerStats

DB_ENV_KEY = "GOALX_DB_PATH"


def _resolve_db_path() -> Path:
    raw = os.getenv(DB_ENV_KEY, "data/goalx.db")
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_resolve_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS match_player_stats (
                match_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                played_at TEXT NOT NULL,
                minutes REAL NOT NULL,
                shots INTEGER NOT NULL,
                shots_on_target INTEGER NOT NULL,
                goals INTEGER NOT NULL,
                assists INTEGER NOT NULL,
                key_passes INTEGER NOT NULL,
                successful_dribbles INTEGER NOT NULL,
                dribble_attempts INTEGER NOT NULL,
                completed_passes INTEGER NOT NULL,
                attempted_passes INTEGER NOT NULL,
                tackles_won INTEGER NOT NULL,
                interceptions INTEGER NOT NULL,
                duels_won INTEGER NOT NULL,
                duels_total INTEGER NOT NULL,
                sprints INTEGER NOT NULL,
                distance_m REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (match_id, player_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rematch_connection (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                account_name TEXT NOT NULL,
                export_directory TEXT NOT NULL,
                linked_player_id TEXT NOT NULL DEFAULT '',
                poll_seconds INTEGER NOT NULL,
                auto_sync INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_match_files (
                file_path TEXT PRIMARY KEY,
                match_id TEXT NOT NULL,
                imported_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auto_export_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                enabled INTEGER NOT NULL,
                source_directory TEXT NOT NULL,
                log_directory TEXT NOT NULL,
                poll_seconds INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS copied_source_files (
                source_file_path TEXT PRIMARY KEY,
                copied_file_name TEXT NOT NULL,
                copied_at TEXT NOT NULL
            )
            """
        )
        column_names = [row["name"] for row in conn.execute("PRAGMA table_info(rematch_connection)").fetchall()]
        if "linked_player_id" not in column_names:
            conn.execute("ALTER TABLE rematch_connection ADD COLUMN linked_player_id TEXT NOT NULL DEFAULT ''")


def upsert_match_stats(
    match_id: str,
    players: Iterable[MatchPlayerStats],
    played_at: datetime | None,
) -> list[str]:
    played_at_iso = (played_at or datetime.now(timezone.utc)).isoformat()
    updated_at = datetime.now(timezone.utc).isoformat()
    touched: list[str] = []
    with _connect() as conn:
        for player in players:
            touched.append(player.player_id)
            conn.execute(
                """
                INSERT INTO match_player_stats (
                    match_id, player_id, played_at, minutes, shots, shots_on_target, goals, assists,
                    key_passes, successful_dribbles, dribble_attempts, completed_passes,
                    attempted_passes, tackles_won, interceptions, duels_won, duels_total,
                    sprints, distance_m, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(match_id, player_id) DO UPDATE SET
                    played_at=excluded.played_at,
                    minutes=excluded.minutes,
                    shots=excluded.shots,
                    shots_on_target=excluded.shots_on_target,
                    goals=excluded.goals,
                    assists=excluded.assists,
                    key_passes=excluded.key_passes,
                    successful_dribbles=excluded.successful_dribbles,
                    dribble_attempts=excluded.dribble_attempts,
                    completed_passes=excluded.completed_passes,
                    attempted_passes=excluded.attempted_passes,
                    tackles_won=excluded.tackles_won,
                    interceptions=excluded.interceptions,
                    duels_won=excluded.duels_won,
                    duels_total=excluded.duels_total,
                    sprints=excluded.sprints,
                    distance_m=excluded.distance_m,
                    updated_at=excluded.updated_at
                """,
                (
                    match_id,
                    player.player_id,
                    played_at_iso,
                    player.minutes,
                    player.shots,
                    player.shots_on_target,
                    player.goals,
                    player.assists,
                    player.key_passes,
                    player.successful_dribbles,
                    player.dribble_attempts,
                    player.completed_passes,
                    player.attempted_passes,
                    player.tackles_won,
                    player.interceptions,
                    player.duels_won,
                    player.duels_total,
                    player.sprints,
                    player.distance_m,
                    updated_at,
                ),
            )
    return sorted(set(touched))


def get_player_totals(player_id: str, recent_matches: int = 5) -> dict[str, float] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS matches,
                COALESCE(SUM(minutes), 0) AS minutes,
                COALESCE(SUM(shots), 0) AS shots,
                COALESCE(SUM(shots_on_target), 0) AS shots_on_target,
                COALESCE(SUM(goals), 0) AS goals,
                COALESCE(SUM(assists), 0) AS assists,
                COALESCE(SUM(key_passes), 0) AS key_passes,
                COALESCE(SUM(successful_dribbles), 0) AS successful_dribbles,
                COALESCE(SUM(dribble_attempts), 0) AS dribble_attempts,
                COALESCE(SUM(completed_passes), 0) AS completed_passes,
                COALESCE(SUM(attempted_passes), 0) AS attempted_passes,
                COALESCE(SUM(tackles_won), 0) AS tackles_won,
                COALESCE(SUM(interceptions), 0) AS interceptions,
                COALESCE(SUM(duels_won), 0) AS duels_won,
                COALESCE(SUM(duels_total), 0) AS duels_total,
                COALESCE(SUM(sprints), 0) AS sprints,
                COALESCE(SUM(distance_m), 0) AS distance_m
            FROM (
                SELECT *
                FROM match_player_stats
                WHERE player_id = ?
                ORDER BY played_at DESC, updated_at DESC
                LIMIT ?
            )
            """,
            (player_id, recent_matches),
        ).fetchone()
    if row is None:
        return None
    result = {key: float(row[key]) for key in row.keys()}
    if result["matches"] <= 0:
        return None
    return result


def get_player_match_rows(player_id: str) -> list[dict[str, object]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT match_id, player_id, played_at, minutes, shots, shots_on_target, goals, assists,
                   key_passes, successful_dribbles, dribble_attempts, completed_passes,
                   attempted_passes, tackles_won, interceptions, duels_won, duels_total,
                   sprints, distance_m, updated_at
            FROM match_player_stats
            WHERE player_id = ?
            ORDER BY played_at DESC, updated_at DESC
            """,
            (player_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def export_player_match_data(player_id: str, export_dir: str | None = None) -> dict[str, object]:
    rows = get_player_match_rows(player_id)
    totals = get_player_totals(player_id)
    if totals is None:
        raise ValueError(f"No match data found for player_id={player_id!r}.")

    from goalx.performance import build_snapshot

    snapshot = build_snapshot(player_id, totals)
    payload = {
        "player_id": player_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matches_analyzed": len(rows),
        "goalx_snapshot": snapshot.model_dump(),
        "match_rows": rows,
    }

    if export_dir:
        directory = Path(export_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_name = f"rematch_match_export_{player_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        file_path = directory / file_name
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["export_file"] = str(file_path)
    return payload


def export_player_tactical_data(player_id: str, export_dir: str | None = None) -> dict[str, object]:
    totals = get_player_totals(player_id)
    if totals is None:
        raise ValueError(f"No tactical data found for player_id={player_id!r}.")

    from goalx.models import GoalXRequest
    from goalx.performance import build_snapshot, build_radar_svg
    from goalx.tactics import analyze_goalx

    snapshot = build_snapshot(player_id, totals)
    suggestions = analyze_goalx(
        GoalXRequest(
            phase="attack",
            ball_x=50.0,
            ball_y=50.0,
            teammates=[
                {"player_id": player_id, "role": "cm", "x": 51.0, "y": 52.0},
                {"player_id": f"{player_id}_a", "role": "st", "x": 70.0, "y": 47.0},
                {"player_id": f"{player_id}_b", "role": "mf", "x": 58.0, "y": 58.0},
                {"player_id": f"{player_id}_c", "role": "df", "x": 42.0, "y": 49.0},
            ],
            opponents=[],
        )
    )
    payload = {
        "player_id": player_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "goalx_snapshot": snapshot.model_dump(),
        "radar_svg": build_radar_svg(snapshot),
        "tactical_suggestions": [s.model_dump() for s in suggestions.suggestions],
        "goalx_score": suggestions.goalx_score,
        "low_data_mode": suggestions.low_data_mode,
    }

    if export_dir:
        directory = Path(export_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_name = f"rematch_tactical_export_{player_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        file_path = directory / file_name
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["export_file"] = str(file_path)
    return payload


def export_player_match_csv(player_id: str) -> str:
    rows = get_player_match_rows(player_id)
    if not rows:
        raise ValueError(f"No match rows found for player_id={player_id!r}.")
    fieldnames = list(rows[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def save_rematch_connection(
    account_name: str,
    export_directory: str,
    linked_player_id: str,
    poll_seconds: int,
    auto_sync: bool,
) -> None:
    updated_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO rematch_connection (
                id, account_name, export_directory, linked_player_id, poll_seconds, auto_sync, updated_at
            )
            VALUES (1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                account_name = excluded.account_name,
                export_directory = excluded.export_directory,
                linked_player_id = excluded.linked_player_id,
                poll_seconds = excluded.poll_seconds,
                auto_sync = excluded.auto_sync,
                updated_at = excluded.updated_at
            """,
            (account_name, export_directory, linked_player_id, poll_seconds, int(auto_sync), updated_at),
        )


def get_rematch_connection() -> dict[str, str | int] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT account_name, export_directory, linked_player_id, poll_seconds, auto_sync
            FROM rematch_connection
            WHERE id = 1
            """
        ).fetchone()
    if row is None:
        return None
    return {
        "account_name": row["account_name"],
        "export_directory": row["export_directory"],
        "linked_player_id": row["linked_player_id"],
        "poll_seconds": int(row["poll_seconds"]),
        "auto_sync": int(row["auto_sync"]),
    }


def is_file_processed(file_path: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM processed_match_files
            WHERE file_path = ?
            """,
            (file_path,),
        ).fetchone()
    return row is not None


def mark_file_processed(file_path: str, match_id: str) -> None:
    imported_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO processed_match_files (file_path, match_id, imported_at)
            VALUES (?, ?, ?)
            """,
            (file_path, match_id, imported_at),
        )


def count_processed_files() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM processed_match_files").fetchone()
    return int(row["count"])


def discover_players_from_recent_matches(recent_matches: int = 20) -> list[tuple[str, int]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT player_id, COUNT(*) AS matches_seen
            FROM (
                SELECT match_id, player_id
                FROM match_player_stats
                ORDER BY played_at DESC, updated_at DESC
                LIMIT ?
            )
            GROUP BY player_id
            ORDER BY matches_seen DESC, player_id ASC
            """,
            (recent_matches * 22,),
        ).fetchall()
    return [(str(row["player_id"]), int(row["matches_seen"])) for row in rows]


def save_auto_export_config(
    enabled: bool,
    source_directory: str,
    log_directory: str,
    poll_seconds: int,
) -> None:
    updated_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO auto_export_config (id, enabled, source_directory, log_directory, poll_seconds, updated_at)
            VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                enabled = excluded.enabled,
                source_directory = excluded.source_directory,
                log_directory = excluded.log_directory,
                poll_seconds = excluded.poll_seconds,
                updated_at = excluded.updated_at
            """,
            (int(enabled), source_directory, log_directory, poll_seconds, updated_at),
        )


def get_auto_export_config() -> dict[str, str | int] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT enabled, source_directory, log_directory, poll_seconds
            FROM auto_export_config
            WHERE id = 1
            """
        ).fetchone()
    if row is None:
        return None
    return {
        "enabled": int(row["enabled"]),
        "source_directory": str(row["source_directory"]),
        "log_directory": str(row["log_directory"]),
        "poll_seconds": int(row["poll_seconds"]),
    }


def is_source_file_copied(source_file_path: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM copied_source_files
            WHERE source_file_path = ?
            """,
            (source_file_path,),
        ).fetchone()
    return row is not None


def mark_source_file_copied(source_file_path: str, copied_file_name: str) -> None:
    copied_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO copied_source_files (source_file_path, copied_file_name, copied_at)
            VALUES (?, ?, ?)
            """,
            (source_file_path, copied_file_name, copied_at),
        )


def count_copied_source_files() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM copied_source_files").fetchone()
    return int(row["count"])
