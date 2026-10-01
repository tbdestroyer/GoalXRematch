from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

os.environ["GOALX_DB_PATH"] = str(tempfile.gettempdir() + f"\\goalx_test_{uuid4().hex}.db")

from goalx.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_suggestions() -> None:
    payload = {
        "phase": "attack",
        "ball_x": 63,
        "ball_y": 52,
        "teammates": [
            {"player_id": "p1", "role": "fw", "x": 70, "y": 50},
            {"player_id": "p2", "role": "fw", "x": 69, "y": 52},
            {"player_id": "p3", "role": "mf", "x": 58, "y": 49},
            {"player_id": "p4", "role": "mf", "x": 57, "y": 51},
            {"player_id": "p5", "role": "df", "x": 43, "y": 53},
        ],
        "opponents": [],
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["low_data_mode"] is True
    assert isinstance(body["goalx_score"], int)
    assert body["suggestions"]


def test_ingest_updates_goalx_and_radar() -> None:
    ingest_payload = {
        "match_id": "m_1001",
        "players": [
            {
                "player_id": "bluelock_9",
                "minutes": 90,
                "shots": 6,
                "shots_on_target": 4,
                "goals": 2,
                "assists": 1,
                "key_passes": 5,
                "successful_dribbles": 4,
                "dribble_attempts": 6,
                "completed_passes": 31,
                "attempted_passes": 38,
                "tackles_won": 2,
                "interceptions": 2,
                "duels_won": 5,
                "duels_total": 9,
                "sprints": 24,
                "distance_m": 10850,
            }
        ],
    }
    ingest_response = client.post("/ingest/match-stats", json=ingest_payload)
    assert ingest_response.status_code == 200
    assert ingest_response.json()["rows_processed"] == 1

    snapshot_response = client.get("/players/bluelock_9/goalx")
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()
    assert snapshot["goalx_score"] >= 0
    assert snapshot["goalx_score"] <= 100
    assert snapshot["blue_lock"]["speed"] >= 0
    assert snapshot["blue_lock"]["shoot"] >= 0
    assert snapshot["grade"] in {"S", "A", "B", "C", "D", "E"}

    radar_response = client.get("/players/bluelock_9/blue-lock-radar.svg")
    assert radar_response.status_code == 200
    assert radar_response.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in radar_response.text
    assert "SPEED" in radar_response.text
    assert "OFFENSE" in radar_response.text
    assert "DRIBBLE" in radar_response.text


def test_connect_and_sync_from_export_folder() -> None:
    with tempfile.TemporaryDirectory(prefix="goalx_exports_") as export_dir:
        export_path = Path(export_dir)
        match_payload = {
            "match_id": "m_2001",
            "players": [
                {
                    "player_id": "linked_10",
                    "minutes": 90,
                    "shots": 4,
                    "shots_on_target": 2,
                    "goals": 1,
                    "assists": 2,
                    "key_passes": 6,
                    "successful_dribbles": 3,
                    "dribble_attempts": 5,
                    "completed_passes": 43,
                    "attempted_passes": 52,
                    "tackles_won": 1,
                    "interceptions": 2,
                    "duels_won": 4,
                    "duels_total": 7,
                    "sprints": 22,
                    "distance_m": 10320,
                }
            ],
        }
        (export_path / "match_2001.json").write_text(json.dumps(match_payload), encoding="utf-8")

        connect_payload = {
            "account_name": "rematch-pc-user",
            "export_directory": str(export_path),
            "linked_player_id": "linked_10",
            "poll_seconds": 15,
            "auto_sync": True,
        }
        connect_response = client.post("/connect/rematch-account", json=connect_payload)
        assert connect_response.status_code == 200
        assert connect_response.json()["connected"] is True

        sync_response = client.post("/sync/rematch-now")
        assert sync_response.status_code == 200
        assert sync_response.json()["matches_ingested"] == 1
        assert sync_response.json()["rows_processed"] == 1

        player_response = client.get("/players/linked_10/goalx")
        assert player_response.status_code == 200
        assert player_response.json()["player_id"] == "linked_10"

        status_response = client.get("/connect/rematch-account/status")
        assert status_response.status_code == 200
        status_body = status_response.json()
        assert status_body["connected"] is True
        assert status_body["linked_player_id"] == "linked_10"
        assert status_body["files_processed"] >= 1

        me_response = client.get("/me/goalx")
        assert me_response.status_code == 200
        assert me_response.json()["player_id"] == "linked_10"

        discover_response = client.get("/connect/rematch-account/discover")
        assert discover_response.status_code == 200
        players = discover_response.json()["discovered_players"]
        assert any(item["player_id"] == "linked_10" for item in players)


def test_sync_from_csv_export_file() -> None:
    with tempfile.TemporaryDirectory(prefix="goalx_exports_csv_") as export_dir:
        export_path = Path(export_dir)
        csv_content = """match_id,played_at,player_id,minutes,shots,shots_on_target,goals,assists,key_passes,successful_dribbles,dribble_attempts,completed_passes,attempted_passes,tackles_won,interceptions,duels_won,duels_total,sprints,distance_m
m_3001,2026-09-29T06:15:00+00:00,csv_7,90,5,3,1,1,4,3,5,34,41,2,2,5,8,23,10700
m_3001,2026-09-29T06:15:00+00:00,csv_8,90,2,1,0,1,3,2,4,29,36,3,4,6,10,20,10400
"""
        (export_path / "match_3001.csv").write_text(csv_content, encoding="utf-8")

        connect_payload = {
            "account_name": "rematch-pc-user",
            "export_directory": str(export_path),
            "linked_player_id": "csv_7",
            "poll_seconds": 15,
            "auto_sync": True,
        }
        connect_response = client.post("/connect/rematch-account", json=connect_payload)
        assert connect_response.status_code == 200

        sync_response = client.post("/sync/rematch-now")
        assert sync_response.status_code == 200
        body = sync_response.json()
        assert body["matches_ingested"] == 1
        assert body["rows_processed"] == 2

        me_response = client.get("/me/goalx")
        assert me_response.status_code == 200
        assert me_response.json()["player_id"] == "csv_7"


def test_export_match_and_tactical_data() -> None:
    ingest_payload = {
        "match_id": "m_export_1",
        "players": [
            {
                "player_id": "export_77",
                "minutes": 90,
                "shots": 5,
                "shots_on_target": 3,
                "goals": 2,
                "assists": 1,
                "key_passes": 4,
                "successful_dribbles": 4,
                "dribble_attempts": 6,
                "completed_passes": 33,
                "attempted_passes": 40,
                "tackles_won": 2,
                "interceptions": 2,
                "duels_won": 6,
                "duels_total": 10,
                "sprints": 25,
                "distance_m": 11000,
            }
        ],
    }
    ingest_response = client.post("/ingest/match-stats", json=ingest_payload)
    assert ingest_response.status_code == 200

    exports_response = client.get("/export/rematch-match-data", params={"player_id": "export_77"})
    assert exports_response.status_code == 200
    body = exports_response.json()
    assert body["player_id"] == "export_77"
    assert body["match_rows"]
    assert body["goalx_snapshot"]["goalx_score"] >= 0

    tactical_response = client.get("/export/rematch-tactical-data", params={"player_id": "export_77"})
    assert tactical_response.status_code == 200
    tactical_body = tactical_response.json()
    assert tactical_body["player_id"] == "export_77"
    assert "radar_svg" in tactical_body
    assert tactical_body["tactical_suggestions"]

    with tempfile.TemporaryDirectory(prefix="goalx_exports_final_") as export_dir:
        connect_response = client.post(
            "/connect/rematch-account",
            json={
                "account_name": "export_test_account",
                "export_directory": export_dir,
                "linked_player_id": "export_77",
                "poll_seconds": 15,
                "auto_sync": True,
            },
        )
        assert connect_response.status_code == 200

        write_response = client.post("/export/rematch-match-data", params={"player_id": "export_77"})
        assert write_response.status_code == 200
        payload = write_response.json()
        assert payload["status"] == "ok"
        assert payload["export_file"]


def test_auto_export_helper_copies_and_ingests() -> None:
    with (
        tempfile.TemporaryDirectory(prefix="goalx_exports_auto_") as export_dir,
        tempfile.TemporaryDirectory(prefix="goalx_source_auto_") as source_dir,
        tempfile.TemporaryDirectory(prefix="goalx_logs_auto_") as log_dir,
    ):
        export_path = Path(export_dir)
        source_path = Path(source_dir)
        log_path = Path(log_dir)

        connect_payload = {
            "account_name": "rematch-pc-user",
            "export_directory": str(export_path),
            "linked_player_id": "auto_11",
            "poll_seconds": 15,
            "auto_sync": True,
        }
        connect_response = client.post("/connect/rematch-account", json=connect_payload)
        assert connect_response.status_code == 200

        source_csv = """match_id,played_at,player_id,minutes,shots,shots_on_target,goals,assists,key_passes,successful_dribbles,dribble_attempts,completed_passes,attempted_passes,tackles_won,interceptions,duels_won,duels_total,sprints,distance_m
m_4001,2026-09-29T06:30:00+00:00,auto_11,90,4,2,1,1,4,3,5,33,40,2,3,5,8,22,10600
"""
        (source_path / "source_match_4001.csv").write_text(source_csv, encoding="utf-8")
        (log_path / "rematch.log").write_text("INFO: final whistle reached\n", encoding="utf-8")

        auto_export_payload = {
            "enabled": True,
            "source_directory": str(source_path),
            "log_directory": str(log_path),
            "poll_seconds": 5,
        }
        auto_export_response = client.post("/auto-export/config", json=auto_export_payload)
        assert auto_export_response.status_code == 200
        assert auto_export_response.json()["enabled"] is True

        run_response = client.post("/auto-export/run-once")
        assert run_response.status_code == 200
        run_body = run_response.json()
        assert run_body["match_end_detected"] is True
        assert run_body["copied_files"]

        copied_file_path = export_path / run_body["copied_files"][0]
        assert copied_file_path.exists()

        sync_response = client.post("/sync/rematch-now")
        assert sync_response.status_code == 200
        assert sync_response.json()["matches_ingested"] == 1
        assert sync_response.json()["rows_processed"] == 1

        me_response = client.get("/me/goalx")
        assert me_response.status_code == 200
        assert me_response.json()["player_id"] == "auto_11"

        auto_status_response = client.get("/auto-export/status")
        assert auto_status_response.status_code == 200
        auto_status = auto_status_response.json()
        assert auto_status["enabled"] is True
        assert auto_status["copied_files"] >= 1


def test_live_capture_window_target_and_loop_controls() -> None:
    windows_response = client.get("/capture/windows", params={"query": "rematch"})
    assert windows_response.status_code == 200
    assert isinstance(windows_response.json(), list)

    target_response = client.post("/capture/window-target", json={"title_contains": "Rematch"})
    assert target_response.status_code == 200
    target_body = target_response.json()
    assert target_body["active"] is True
    assert target_body["title_contains"] == "Rematch"

    start_response = client.post("/capture/start", json={"interval_seconds": 0.5})
    assert start_response.status_code == 200
    assert start_response.json()["running"] is True

    status_response = client.get("/live/capture/status")
    assert status_response.status_code == 200
    status_body = status_response.json()
    assert "window_target" in status_body
    assert "window_detected" in status_body
    assert "capture_interval_seconds" in status_body

    stop_response = client.post("/capture/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["running"] is False

    clear_response = client.delete("/capture/window-target")
    assert clear_response.status_code == 200
    assert clear_response.json()["active"] is False

    live_status = client.get("/live/capture/status")
    assert live_status.status_code == 200
    live_body = live_status.json()
    assert "pitch_scene" in live_body
    assert "live_frame_url" in live_body
    assert "motion_score" in live_body
    assert "hud_score" in live_body
    assert "kit_profile" in live_body
    assert "frame_kind" in live_body
    assert "gameplay_detected" in live_body


def test_live_kit_override_and_reset() -> None:
    override = client.post("/live/kits/override", json={"team_label": "blue", "opp_label": "white"})
    assert override.status_code == 200
    body = override.json()
    assert body["kit_profile"]["team_label"] == "blue"
    assert body["kit_profile"]["opp_label"] == "white"
    assert body["kit_profile"]["locked"] is True

    status = client.get("/live/capture/status").json()
    assert status["kit_profile"]["team_label"] == "blue"

    bad = client.post("/live/kits/override", json={})
    assert bad.status_code == 400

    reset = client.post("/live/kits/reset")
    assert reset.status_code == 200
    assert reset.json()["kit_profile"] is None
    assert client.get("/live/kits").json()["kit_profile"] is None
