from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from goalx.auto_export import AutoExportService
from goalx.connector import RematchSyncService
from goalx.screen_capture import screen_capture_service
from goalx.models import (
    AutoExportConfigRequest,
    AutoExportRunResult,
    AutoExportStatus,
    CaptureLoopRequest,
    CaptureLoopStatus,
    CaptureWindowCandidate,
    CaptureWindowTargetRequest,
    CaptureWindowTargetResponse,
    GoalXRequest,
    GoalXResponse,
    KitOverrideRequest,
    KitProfileResponse,
    LivePitchScene,
    MatchIngestRequest,
    MatchIngestResponse,
    LiveCaptureStatus,
    BlueLockStats,
    PlayerDiscoveryItem,
    PlayerDiscoveryResponse,
    PlayerGoalXSnapshot,
    PopupLearnRequest,
    PopupLearnResponse,
    RematchAccountConnectRequest,
    RematchAccountStatus,
    ScoreboardIngestResponse,
    SyncRunResult,
    TacticalRecognitionResult,
    WindowCaptureConfigRequest,
    WindowCaptureConfigResponse,
)
from goalx.performance import blend_blue_lock, build_radar_svg, build_snapshot, snapshot_from_stats
from goalx.scoreboard import ScoreboardResult, rows_to_match_stats
from goalx.storage import (
    count_copied_source_files,
    count_processed_files,
    discover_players_from_recent_matches,
    export_player_match_csv,
    export_player_match_data,
    export_player_tactical_data,
    get_auto_export_config,
    get_player_totals,
    get_rematch_connection,
    init_db,
    save_auto_export_config,
    save_rematch_connection,
    upsert_match_stats,
)
from goalx.tactics import analyze_goalx

init_db()
sync_service = RematchSyncService()
auto_export_service = AutoExportService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    sync_service.start()
    auto_export_service.start()
    try:
        yield
    finally:
        auto_export_service.stop()
        sync_service.stop()


app = FastAPI(
    title="GoalX for Rematch",
    version="0.1.0",
    description="Low-data tactical suggestion API for improving positioning and GoalX.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/live-capture")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=GoalXResponse)
def analyze(request: GoalXRequest) -> GoalXResponse:
    return analyze_goalx(request)


@app.post("/ingest/match-stats", response_model=MatchIngestResponse)
def ingest_match_stats(payload: MatchIngestRequest) -> MatchIngestResponse:
    player_ids = upsert_match_stats(payload.match_id, payload.players, payload.played_at)
    return MatchIngestResponse(
        match_id=payload.match_id,
        rows_processed=len(payload.players),
        players_updated=player_ids,
    )


@app.get("/players/{player_id}/goalx", response_model=PlayerGoalXSnapshot)
def player_goalx(player_id: str) -> PlayerGoalXSnapshot:
    totals = get_player_totals(player_id)
    if totals is None:
        raise HTTPException(status_code=404, detail="No match data found for player.")
    return build_snapshot(player_id, totals)


@app.get("/players/{player_id}/blue-lock-radar.svg")
def player_radar_svg(player_id: str) -> Response:
    totals = get_player_totals(player_id)
    if totals is None:
        raise HTTPException(status_code=404, detail="No match data found for player.")
    snapshot = build_snapshot(player_id, totals)
    svg = build_radar_svg(snapshot)
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/connect/rematch-account", response_model=RematchAccountStatus)
def connect_rematch_account(payload: RematchAccountConnectRequest) -> RematchAccountStatus:
    save_rematch_connection(
        account_name=payload.account_name,
        export_directory=payload.export_directory,
        linked_player_id=payload.linked_player_id,
        poll_seconds=payload.poll_seconds,
        auto_sync=payload.auto_sync,
    )
    cfg = get_rematch_connection()
    if cfg is None:
        raise HTTPException(status_code=500, detail="Connection setup failed.")
    return RematchAccountStatus(
        account_name=str(cfg["account_name"]),
        export_directory=str(cfg["export_directory"]),
        linked_player_id=str(cfg["linked_player_id"]),
        poll_seconds=int(cfg["poll_seconds"]),
        auto_sync=bool(cfg["auto_sync"]),
        connected=True,
        last_sync_at=sync_service.last_sync_at,
        last_error=sync_service.last_error,
        files_processed=count_processed_files(),
    )


@app.get("/connect/rematch-account/status", response_model=RematchAccountStatus)
def rematch_account_status() -> RematchAccountStatus:
    cfg = get_rematch_connection()
    if cfg is None:
        return RematchAccountStatus(
            account_name="",
            export_directory="",
            linked_player_id="",
            poll_seconds=15,
            auto_sync=False,
            connected=False,
            last_sync_at=sync_service.last_sync_at,
            last_error=sync_service.last_error,
            files_processed=count_processed_files(),
        )
    return RematchAccountStatus(
        account_name=str(cfg["account_name"]),
        export_directory=str(cfg["export_directory"]),
        linked_player_id=str(cfg["linked_player_id"]),
        poll_seconds=int(cfg["poll_seconds"]),
        auto_sync=bool(cfg["auto_sync"]),
        connected=True,
        last_sync_at=sync_service.last_sync_at,
        last_error=sync_service.last_error,
        files_processed=count_processed_files(),
    )


@app.post("/sync/rematch-now", response_model=SyncRunResult)
def sync_rematch_now() -> SyncRunResult:
    try:
        return sync_service.run_sync_once()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/auto-export/config", response_model=AutoExportStatus)
def configure_auto_export(payload: AutoExportConfigRequest) -> AutoExportStatus:
    save_auto_export_config(
        enabled=payload.enabled,
        source_directory=payload.source_directory,
        log_directory=payload.log_directory,
        poll_seconds=payload.poll_seconds,
    )
    cfg = get_auto_export_config()
    if cfg is None:
        raise HTTPException(status_code=500, detail="Auto-export configuration save failed.")
    return AutoExportStatus(
        enabled=bool(cfg["enabled"]),
        source_directory=str(cfg["source_directory"]),
        log_directory=str(cfg["log_directory"]),
        poll_seconds=int(cfg["poll_seconds"]),
        running=auto_export_service.running,
        last_match_end_at=auto_export_service.last_match_end_at,
        last_copy_at=auto_export_service.last_copy_at,
        copied_files=count_copied_source_files(),
        last_error=auto_export_service.last_error,
    )


@app.get("/auto-export/status", response_model=AutoExportStatus)
def auto_export_status() -> AutoExportStatus:
    cfg = get_auto_export_config()
    if cfg is None:
        return AutoExportStatus(
            enabled=False,
            source_directory="",
            log_directory="",
            poll_seconds=5,
            running=auto_export_service.running,
            last_match_end_at=auto_export_service.last_match_end_at,
            last_copy_at=auto_export_service.last_copy_at,
            copied_files=count_copied_source_files(),
            last_error=auto_export_service.last_error,
        )
    return AutoExportStatus(
        enabled=bool(cfg["enabled"]),
        source_directory=str(cfg["source_directory"]),
        log_directory=str(cfg["log_directory"]),
        poll_seconds=int(cfg["poll_seconds"]),
        running=auto_export_service.running,
        last_match_end_at=auto_export_service.last_match_end_at,
        last_copy_at=auto_export_service.last_copy_at,
        copied_files=count_copied_source_files(),
        last_error=auto_export_service.last_error,
    )


@app.post("/auto-export/run-once", response_model=AutoExportRunResult)
def auto_export_run_once() -> AutoExportRunResult:
    try:
        return auto_export_service.run_once()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/connect/rematch-account/discover", response_model=PlayerDiscoveryResponse)
def discover_linkable_players() -> PlayerDiscoveryResponse:
    items = [
        PlayerDiscoveryItem(player_id=player_id, matches_seen=matches_seen)
        for player_id, matches_seen in discover_players_from_recent_matches()
    ]
    return PlayerDiscoveryResponse(discovered_players=items)


@app.get("/me/goalx", response_model=PlayerGoalXSnapshot)
def my_goalx() -> PlayerGoalXSnapshot:
    snapshot = _compose_player_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No match or live capture data yet.")
    return snapshot


@app.get("/me/blue-lock-radar.svg")
def my_radar_svg() -> Response:
    snapshot = _compose_player_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No match or live capture data yet.")
    svg = build_radar_svg(snapshot)
    return Response(content=svg, media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.get("/live/blue-lock-radar.svg")
def live_radar_svg() -> Response:
    snapshot = _compose_player_snapshot()
    if snapshot is None:
        snapshot = snapshot_from_stats(
            "REMATCH",
            BlueLockStats(speed=0, offense=0, shoot=0, dribble=0, passing=0, defense=0),
        )
    svg = build_radar_svg(snapshot)
    return Response(content=svg, media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.get("/export/rematch-match-data", response_model=None)
def export_rematch_match_data(player_id: str | None = None, format: str = "json"):
    cfg = get_rematch_connection()
    target_player = player_id or (str(cfg["linked_player_id"]) if cfg else None)
    if target_player is None:
        raise HTTPException(status_code=400, detail="Provide player_id or link a Rematch account first.")
    if format.lower() == "csv":
        csv_payload = export_player_match_csv(str(target_player))
        return Response(content=csv_payload, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=rematch_match_data_{target_player}.csv"})
    payload = export_player_match_data(str(target_player))
    return payload


@app.post("/export/rematch-match-data")
def export_rematch_match_data_file(player_id: str | None = None) -> dict[str, object]:
    cfg = get_rematch_connection()
    export_dir = str(cfg["export_directory"]) if cfg else "exports"
    target_player = player_id or (str(cfg["linked_player_id"]) if cfg and str(cfg["linked_player_id"]).strip() else None)
    if target_player is None:
        raise HTTPException(status_code=400, detail="Provide player_id or link a Rematch account first.")
    payload = export_player_match_data(str(target_player), export_dir)
    return {"status": "ok", "player_id": target_player, "export_file": payload.get("export_file"), "matches_analyzed": payload["matches_analyzed"]}


@app.get("/export/rematch-tactical-data", response_model=None)
def export_rematch_tactical_data(player_id: str | None = None, format: str = "json"):
    cfg = get_rematch_connection()
    target_player = player_id or (str(cfg["linked_player_id"]) if cfg else None)
    if target_player is None:
        raise HTTPException(status_code=400, detail="Provide player_id or link a Rematch account first.")
    payload = export_player_tactical_data(str(target_player))
    if format.lower() == "csv":
        rows = [
            {"player_id": payload["player_id"], "goalx_score": payload["goalx_score"], "generated_at": payload["generated_at"]}
        ]
        import csv as csv_module
        output = __import__("io").StringIO()
        writer = csv_module.DictWriter(output, fieldnames=["player_id", "goalx_score", "generated_at"])
        writer.writeheader()
        writer.writerows(rows)
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=rematch_tactical_data_{target_player}.csv"})
    return payload


@app.get("/ui/demo", response_class=HTMLResponse)
def demo_ui() -> HTMLResponse:
    ui_path = Path(__file__).resolve().parent / "ui" / "demo.html"
    html = ui_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/ui/live-capture", response_class=HTMLResponse)
def live_capture_ui() -> HTMLResponse:
    ui_path = Path(__file__).resolve().parent / "ui" / "live_capture.html"
    html = ui_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/live/capture/status", response_model=LiveCaptureStatus)
def live_capture_status() -> LiveCaptureStatus:
    return _build_live_capture_status()


@app.get("/live/kits", response_model=KitProfileResponse)
def live_kits() -> KitProfileResponse:
    profile = screen_capture_service.kit_profile()
    return KitProfileResponse(kit_profile=profile, message="calibrating" if profile is None or not profile.locked else "locked")


@app.post("/live/kits/reset", response_model=KitProfileResponse)
def reset_live_kits() -> KitProfileResponse:
    screen_capture_service.reset_kits()
    return KitProfileResponse(kit_profile=None, message="Kit profile reset; recalibrating from the next gameplay frames.")


@app.post("/live/kits/override", response_model=KitProfileResponse)
def override_live_kits(payload: KitOverrideRequest) -> KitProfileResponse:
    if payload.team_hue is None and payload.opp_hue is None and not payload.team_label and not payload.opp_label:
        raise HTTPException(status_code=400, detail="Provide team_hue/opp_hue or team_label/opp_label.")
    profile = screen_capture_service.override_kits(
        team_hue=payload.team_hue,
        opp_hue=payload.opp_hue,
        team_label=payload.team_label,
        opp_label=payload.opp_label,
    )
    if profile is None:
        raise HTTPException(status_code=400, detail="Unknown kit label; use e.g. white, red, orange, blue, black.")
    return KitProfileResponse(kit_profile=profile, message="Kit profile overridden and locked.")


@app.get("/capture/windows", response_model=list[CaptureWindowCandidate])
def list_capture_windows(query: str = "") -> list[CaptureWindowCandidate]:
    windows = screen_capture_service.list_windows(query=query)
    return [
        CaptureWindowCandidate(
            title=str(item["title"]),
            x=int(item["x"]),
            y=int(item["y"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in windows
    ]


@app.post("/capture/window-target", response_model=CaptureWindowTargetResponse)
def set_capture_window_target(payload: CaptureWindowTargetRequest) -> CaptureWindowTargetResponse:
    try:
        result = screen_capture_service.set_window_target(payload.title_contains)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CaptureWindowTargetResponse(
        active=bool(result["active"]),
        title_contains=str(result["title_contains"]),
        matched=bool(result["matched"]),
        region=result["region"],
    )


@app.delete("/capture/window-target", response_model=CaptureWindowTargetResponse)
def clear_capture_window_target() -> CaptureWindowTargetResponse:
    result = screen_capture_service.clear_window_target()
    return CaptureWindowTargetResponse(
        active=bool(result["active"]),
        title_contains=str(result["title_contains"]),
        matched=bool(result["matched"]),
        region=result["region"],
    )


@app.post("/capture/start", response_model=CaptureLoopStatus)
def start_capture_loop(payload: CaptureLoopRequest) -> CaptureLoopStatus:
    try:
        status = screen_capture_service.start_live_capture(payload.interval_seconds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CaptureLoopStatus(
        running=bool(status["running"]),
        interval_seconds=float(status["interval_seconds"]),
        target_window=str(status["target_window"]) if status["target_window"] else None,
    )


@app.post("/capture/stop", response_model=CaptureLoopStatus)
def stop_capture_loop() -> CaptureLoopStatus:
    status = screen_capture_service.stop_live_capture()
    return CaptureLoopStatus(
        running=bool(status["running"]),
        interval_seconds=float(status["interval_seconds"]),
        target_window=str(status["target_window"]) if status["target_window"] else None,
    )


@app.post("/capture/config", response_model=WindowCaptureConfigResponse)
def set_capture_config(payload: WindowCaptureConfigRequest) -> WindowCaptureConfigResponse:
    region = screen_capture_service.set_crop_region(payload.x, payload.y, payload.width, payload.height)
    return WindowCaptureConfigResponse(
        enabled=payload.enabled,
        x=region[0],
        y=region[1],
        width=region[2],
        height=region[3],
        source="game-window-crop" if payload.enabled else "fullscreen",
    )


@app.post("/capture/frame")
def capture_frame() -> dict[str, object]:
    metadata = screen_capture_service.capture_latest()
    return {
        "status": "ok",
        "captured_at": metadata["captured_at"],
        "motion_score": metadata["motion_score"],
        "file_name": metadata["file_name"],
        "file_path": metadata["file_path"],
        "crop_region": metadata["region"],
    }


@app.post("/capture/recognize", response_model=TacticalRecognitionResult)
def recognize_tactical_frame() -> TacticalRecognitionResult:
    screen_capture_service.capture_latest()
    live = _build_live_capture_status()
    return TacticalRecognitionResult(
        match_state=live.match_state,
        phase=live.phase,
        game_window_detected=live.window_detected or bool(live.live_frame_url),
        hud_detected=live.hud_detected,
        tactical_focus=live.tactical_focus,
        goalx_score=live.goalx_score,
        pitch_scene=live.pitch_scene,
    )


@app.get("/capture/latest")
def latest_capture() -> FileResponse:
    frame_path = screen_capture_service.latest_frame_path()
    if not frame_path:
        raise HTTPException(status_code=404, detail="No captured frame found yet.")
    media_type = "image/jpeg" if frame_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return FileResponse(frame_path, media_type=media_type)


@app.get("/live/tracking")
def live_tracking() -> dict[str, object]:
    axes, sources = screen_capture_service.session_axes_with_sources()
    return {
        "metrics": screen_capture_service.tracking_metrics(),
        "axes": axes or {},
        "axis_sources": sources,
        "action_popup": screen_capture_service.last_popup(),
        "popup_templates": screen_capture_service.popup_templates(),
    }


@app.post("/live/tracking/reset")
def reset_live_tracking() -> dict[str, object]:
    screen_capture_service.reset_match_tracking()
    return {"status": "ok", "metrics": screen_capture_service.tracking_metrics()}


@app.post("/live/popups/learn", response_model=PopupLearnResponse)
def learn_live_popup(payload: PopupLearnRequest) -> PopupLearnResponse:
    label = payload.label.strip().lower()
    count, detected = screen_capture_service.learn_popup(label)
    if detected is None:
        return PopupLearnResponse(label=label, templates=count, detected=None, message="No popup text visible on the latest frame. Try again while the word is on screen.")
    return PopupLearnResponse(label=label, templates=count, detected=detected, message=f"Learned '{label}' from the current popup ({count} template(s)).")


@app.get("/live/scoreboard")
def live_scoreboard() -> dict[str, object]:
    result = screen_capture_service.last_scoreboard()
    return {"scoreboard": None if result is None else result.as_dict()}


@app.post("/live/scoreboard/read", response_model=ScoreboardIngestResponse)
def read_live_scoreboard(ingest: bool = True) -> ScoreboardIngestResponse:
    """Force a scoreboard read on the latest frame (use on the end-of-match results screen)."""
    result = screen_capture_service.read_scoreboard_now()
    if result is None or not result.detected:
        return ScoreboardIngestResponse(detected=False, message="No results table recognised on the latest frame.", scoreboard=None if result is None else result.as_dict())
    match_id = _ingest_scoreboard(result) if ingest else None
    return ScoreboardIngestResponse(
        detected=True,
        rows=len(result.rows),
        self_name=result.self_name,
        match_id=match_id,
        message="Scoreboard read" + (" and stored." if match_id else "."),
        scoreboard=result.as_dict(),
    )


def _ingest_scoreboard(result: ScoreboardResult) -> str | None:
    """Store a results screen as a match for every player on it; returns the match id."""
    if not result.detected or len(result.rows) < 2:
        return None
    metrics = screen_capture_service.tracking_metrics()
    minutes = max(1.0, float(metrics.get("gameplay_minutes", 0.0) or 0.0))
    if minutes < 2.0:
        minutes = 6.0  # typical Rematch match length when tracking was short/absent
    pass_rate = float(metrics.get("pass_rate_hud", 0.0) or 0.0) or None
    cfg = get_rematch_connection()
    linked = str(cfg["linked_player_id"]).strip() if cfg and str(cfg["linked_player_id"]).strip() else ""
    players = rows_to_match_stats(result.rows, minutes=minutes, pass_rate_hint=pass_rate)
    if linked:
        for row, player in zip(result.rows, players):
            if row.is_self:
                player.player_id = linked[:32]
    match_id = f"scoreboard_{result.captured_at.replace(':', '').replace('-', '')[:15]}"
    upsert_match_stats(match_id, players, _parse_iso(result.captured_at))
    return match_id


def _auto_ingest_scoreboard() -> None:
    result = screen_capture_service.take_scoreboard_for_ingest()
    if result is not None:
        try:
            _ingest_scoreboard(result)
        except Exception:
            pass


def _build_live_capture_status() -> LiveCaptureStatus:
    capture_meta = screen_capture_service.latest_metadata()
    capture_loop = screen_capture_service.capture_loop_status()
    cfg = get_rematch_connection()
    linked_player_id = str(cfg["linked_player_id"]) if cfg and str(cfg["linked_player_id"]).strip() else ""
    capture_enabled = bool(capture_loop["running"] or screen_capture_service.latest_frame_path())
    crop = screen_capture_service.get_crop_region()
    crop_region = {"x": crop[0], "y": crop[1], "width": crop[2], "height": crop[3]}
    meta_region = capture_meta.get("region")
    if crop_region["width"] <= 0 and isinstance(meta_region, dict) and int(meta_region.get("width") or 0) > 0:
        crop_region = {
            "x": int(meta_region.get("x") or 0),
            "y": int(meta_region.get("y") or 0),
            "width": int(meta_region.get("width") or 0),
            "height": int(meta_region.get("height") or 0),
        }
    image_size_meta = capture_meta.get("image_size") if isinstance(capture_meta.get("image_size"), dict) else {}
    image_size = {
        "width": int(image_size_meta.get("width") or 0),
        "height": int(image_size_meta.get("height") or 0),
    }
    hud_score = float(capture_meta.get("hud_score", 0.0) or 0.0)
    motion_score = float(capture_meta.get("motion_score", 0.0) or 0.0)
    hud_detected = hud_score >= 0.15
    window_detected = bool(capture_meta.get("window_detected", False))
    frame_url = "/capture/latest" if screen_capture_service.latest_frame_path() else ""
    captured_at = _parse_iso(capture_meta.get("captured_at"))
    last_error = str(capture_meta.get("last_error") or "") or None
    pitch_scene = _scene_from_meta(capture_meta.get("pitch_scene"))
    live_analysis = _analyze_scene(pitch_scene) if pitch_scene else None
    snapshot = _compose_player_snapshot()
    frame_kind = screen_capture_service.last_frame_kind()
    gameplay_detected = bool(pitch_scene and pitch_scene.gameplay_detected and screen_capture_service.gameplay_active())
    gated = bool(pitch_scene) and not gameplay_detected

    if live_analysis is not None:
        match_state = "match_live" if hud_detected or (pitch_scene and pitch_scene.confidence >= 0.2) else "match_not_visible"
        phase = pitch_scene.phase if pitch_scene else "transition"
        goalx_score = live_analysis.goalx_score
        tactical_focus = [f"{item.title}: {item.detail}" for item in live_analysis.suggestions] or [
            "Shape looks stable. Keep the next support pass on time."
        ]
    elif not linked_player_id:
        match_state = "waiting_for_match"
        phase = "build-up"
        goalx_score = int(motion_score * 100)
        tactical_focus = ["Set the Rematch window target, then start capture to populate the live pitch."]
    elif snapshot is None:
        match_state = "waiting_for_match"
        phase = "build-up"
        goalx_score = int(motion_score * 100)
        tactical_focus = ["Waiting for first valid match export"]
    else:
        match_state = "match_live" if hud_detected else "match_not_visible"
        phase = "attack" if snapshot.goalx_score >= 60 else "build-up"
        goalx_score = snapshot.goalx_score
        tactical_focus = [
            "Press the left half-space early",
            "Keep the defensive line compact",
            "Increase support depth after turnover",
        ]

    if gated and capture_enabled:
        label = {"cutscene": "Cutscene", "menu": "Menu / loading", "unknown": "Non-gameplay frame"}.get(frame_kind, "Cutscene")
        tactical_focus = [f"{label} — holding last tactical read"] + tactical_focus

    screen_capture_service.note_match_state(match_state)
    status = LiveCaptureStatus(
        enabled=capture_enabled or bool(linked_player_id),
        recording=bool(capture_loop["running"]),
        source="screen+export",
        match_state=match_state,
        phase=phase,
        goalx_score=goalx_score,
        tactical_focus=tactical_focus,
        hud_detected=hud_detected,
        window_detected=window_detected,
        window_target=str(capture_meta.get("window_target", "") or ""),
        capture_interval_seconds=float(capture_loop["interval_seconds"]),
        crop_region=crop_region,
        motion_score=round(motion_score, 4),
        hud_score=round(hud_score, 4),
        captured_at=captured_at,
        last_error=last_error,
        live_frame_url=frame_url,
        pitch_scene=pitch_scene,
        image_size=image_size,
        kit_profile=screen_capture_service.kit_profile(),
        frame_kind=frame_kind,
        gameplay_detected=gameplay_detected,
    )
    if snapshot is not None:
        status.blue_lock = snapshot.blue_lock
        status.goalx_score = snapshot.goalx_score
    return status


def _compose_player_snapshot() -> PlayerGoalXSnapshot | None:
    cfg = get_rematch_connection()
    linked_player_id = str(cfg["linked_player_id"]).strip() if cfg and str(cfg["linked_player_id"]).strip() else ""
    career = None
    if linked_player_id:
        totals = get_player_totals(linked_player_id)
        if totals is not None:
            career = build_snapshot(linked_player_id, totals)
    live = screen_capture_service.session_blue_lock()
    if career is None and live is None:
        return None
    blended = blend_blue_lock(career.blue_lock if career else None, live)
    return snapshot_from_stats(
        linked_player_id or "REMATCH",
        blended,
        matches_analyzed=career.matches_analyzed if career else 0,
    )


def _parse_iso(value: object) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _scene_from_meta(value: object) -> LivePitchScene | None:
    if not isinstance(value, dict):
        return None
    try:
        return LivePitchScene.model_validate(value)
    except Exception:
        return None


def _analyze_scene(scene: LivePitchScene) -> GoalXResponse | None:
    if len(scene.teammates) < 2:
        return None
    request = GoalXRequest(
        phase=scene.phase,
        ball_x=scene.ball_x,
        ball_y=scene.ball_y,
        teammates=scene.teammates,
        opponents=scene.opponents,
    )
    return analyze_goalx(request)

