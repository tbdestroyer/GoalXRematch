from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Phase = Literal["attack", "defense", "transition"]


class PlayerPosition(BaseModel):
    player_id: str = Field(min_length=1, max_length=32)
    role: str = Field(min_length=1, max_length=24)
    x: float = Field(ge=0.0, le=100.0, description="Pitch length percent")
    y: float = Field(ge=0.0, le=100.0, description="Pitch width percent")


class GoalXRequest(BaseModel):
    phase: Phase
    ball_x: float = Field(ge=0.0, le=100.0)
    ball_y: float = Field(ge=0.0, le=100.0)
    teammates: list[PlayerPosition] = Field(min_length=2, max_length=11)
    opponents: list[PlayerPosition] = Field(default_factory=list, max_length=11)


class TacticalSuggestion(BaseModel):
    title: str
    detail: str
    priority: Literal["high", "medium", "low"]
    metric_impact: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated impact on GoalX from 0 to 1",
    )


class GoalXResponse(BaseModel):
    goalx_score: int = Field(ge=0, le=100)
    suggestions: list[TacticalSuggestion]
    low_data_mode: bool = True


class MatchPlayerStats(BaseModel):
    player_id: str = Field(min_length=1, max_length=32)
    minutes: float = Field(gt=0.0, le=130.0)
    shots: int = Field(ge=0, le=40)
    shots_on_target: int = Field(ge=0, le=40)
    goals: int = Field(ge=0, le=20)
    assists: int = Field(ge=0, le=20)
    key_passes: int = Field(ge=0, le=80)
    successful_dribbles: int = Field(ge=0, le=80)
    dribble_attempts: int = Field(ge=0, le=120)
    completed_passes: int = Field(ge=0, le=400)
    attempted_passes: int = Field(ge=0, le=500)
    tackles_won: int = Field(ge=0, le=80)
    interceptions: int = Field(ge=0, le=80)
    duels_won: int = Field(ge=0, le=80)
    duels_total: int = Field(ge=0, le=120)
    sprints: int = Field(ge=0, le=200)
    distance_m: float = Field(ge=0.0, le=25000.0)


class MatchIngestRequest(BaseModel):
    match_id: str = Field(min_length=1, max_length=64)
    played_at: datetime | None = None
    players: list[MatchPlayerStats] = Field(min_length=1, max_length=22)


class MatchIngestResponse(BaseModel):
    match_id: str
    rows_processed: int
    players_updated: list[str]


class BlueLockStats(BaseModel):
    speed: int = Field(ge=0, le=100)
    offense: int = Field(ge=0, le=100)
    shoot: int = Field(ge=0, le=100)
    dribble: int = Field(ge=0, le=100)
    passing: int = Field(ge=0, le=100)
    defense: int = Field(ge=0, le=100)


class PlayerGoalXSnapshot(BaseModel):
    player_id: str
    matches_analyzed: int = Field(ge=0)
    goalx_score: int = Field(ge=0, le=100)
    grade: str = "E"
    blue_lock: BlueLockStats


class RematchAccountConnectRequest(BaseModel):
    account_name: str = Field(min_length=1, max_length=64)
    export_directory: str = Field(min_length=1, max_length=300)
    linked_player_id: str = Field(min_length=1, max_length=32)
    poll_seconds: int = Field(ge=5, le=300, default=15)
    auto_sync: bool = True


class RematchAccountStatus(BaseModel):
    account_name: str
    export_directory: str
    linked_player_id: str
    poll_seconds: int
    auto_sync: bool
    connected: bool
    last_sync_at: datetime | None = None
    last_error: str | None = None
    files_processed: int = Field(ge=0)


class SyncRunResult(BaseModel):
    matches_ingested: int = Field(ge=0)
    rows_processed: int = Field(ge=0)
    players_updated: list[str]
    files_processed: list[str]


class PlayerDiscoveryItem(BaseModel):
    player_id: str
    matches_seen: int = Field(ge=0)


class PlayerDiscoveryResponse(BaseModel):
    discovered_players: list[PlayerDiscoveryItem]


class AutoExportConfigRequest(BaseModel):
    enabled: bool = True
    source_directory: str = Field(min_length=1, max_length=300)
    log_directory: str = Field(min_length=1, max_length=300)
    poll_seconds: int = Field(ge=3, le=120, default=5)


class AutoExportStatus(BaseModel):
    enabled: bool
    source_directory: str
    log_directory: str
    poll_seconds: int
    running: bool
    last_match_end_at: datetime | None = None
    last_copy_at: datetime | None = None
    copied_files: int = Field(ge=0)
    last_error: str | None = None


class AutoExportRunResult(BaseModel):
    match_end_detected: bool
    copied_files: list[str]
    message: str


class LivePitchScene(BaseModel):
    phase: Phase
    ball_x: float = Field(ge=0.0, le=100.0)
    ball_y: float = Field(ge=0.0, le=100.0)
    teammates: list[PlayerPosition] = Field(default_factory=list)
    opponents: list[PlayerPosition] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    detection_count: int = Field(ge=0, default=0)
    view_ball_x: float | None = Field(default=None, ge=0.0, le=100.0)
    view_ball_y: float | None = Field(default=None, ge=0.0, le=100.0)
    view_teammates: list[PlayerPosition] = Field(default_factory=list)
    view_opponents: list[PlayerPosition] = Field(default_factory=list)
    map_detected: bool = False
    map_ball_x: float | None = Field(default=None, ge=0.0, le=100.0)
    map_ball_y: float | None = Field(default=None, ge=0.0, le=100.0)
    map_teammates: list[PlayerPosition] = Field(default_factory=list)
    map_opponents: list[PlayerPosition] = Field(default_factory=list)
    map_region: dict[str, float] = Field(default_factory=dict)
    self_x: float | None = Field(default=None, ge=0.0, le=100.0)
    self_y: float | None = Field(default=None, ge=0.0, le=100.0)
    facing_deg: float | None = None
    source: Literal["fused", "minimap", "camera", "none"] = "none"
    fusion_matches: int = Field(ge=0, default=0)
    gameplay_detected: bool = True
    frame_kind: Literal["gameplay", "cutscene", "menu", "unknown"] = "unknown"


class KitProfile(BaseModel):
    team_hue: float | None = None
    team_saturation: float = Field(ge=0.0, le=1.0, default=0.0)
    team_brightness: float = Field(ge=0.0, le=255.0, default=0.0)
    opp_hue: float | None = None
    opp_saturation: float = Field(ge=0.0, le=1.0, default=0.0)
    opp_brightness: float = Field(ge=0.0, le=255.0, default=0.0)
    team_label: str = "unknown"
    opp_label: str = "unknown"
    samples: int = Field(ge=0, default=0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    locked: bool = False


class KitOverrideRequest(BaseModel):
    team_hue: float | None = Field(default=None, ge=0.0, le=360.0)
    opp_hue: float | None = Field(default=None, ge=0.0, le=360.0)
    team_label: str | None = Field(default=None, max_length=24)
    opp_label: str | None = Field(default=None, max_length=24)


class KitProfileResponse(BaseModel):
    kit_profile: KitProfile | None = None
    message: str = ""


class LiveCaptureStatus(BaseModel):
    enabled: bool = False
    recording: bool = False
    source: str = "local-export"
    match_state: str = "waiting_for_match"
    phase: str = "build-up"
    goalx_score: int = Field(ge=0, le=100, default=0)
    blue_lock: BlueLockStats = Field(default_factory=lambda: BlueLockStats(
        speed=0,
        offense=0,
        shoot=0,
        dribble=0,
        passing=0,
        defense=0,
    ))
    tactical_focus: list[str] = Field(default_factory=list)
    hud_detected: bool = False
    window_detected: bool = False
    window_target: str = ""
    capture_interval_seconds: float = Field(ge=0.2, le=10.0, default=1.5)
    crop_region: dict[str, int] = Field(default_factory=lambda: {"x": 0, "y": 0, "width": 0, "height": 0})
    motion_score: float = Field(ge=0.0, le=1.0, default=0.0)
    hud_score: float = Field(ge=0.0, le=1.0, default=0.0)
    captured_at: datetime | None = None
    last_error: str | None = None
    live_frame_url: str = ""
    pitch_scene: LivePitchScene | None = None
    image_size: dict[str, int] = Field(default_factory=lambda: {"width": 0, "height": 0})
    kit_profile: KitProfile | None = None
    frame_kind: str = "unknown"
    gameplay_detected: bool = False
    tracking: dict[str, object] = Field(default_factory=dict)
    action_popup: dict[str, object] | None = None
    scoreboard: dict[str, object] | None = None
    axis_sources: dict[str, str] = Field(default_factory=dict)


class PopupLearnRequest(BaseModel):
    label: str = Field(min_length=1, max_length=24)


class PopupLearnResponse(BaseModel):
    label: str
    templates: int
    detected: dict[str, object] | None = None
    message: str = ""


class ScoreboardIngestResponse(BaseModel):
    detected: bool
    rows: int = 0
    self_name: str | None = None
    match_id: str | None = None
    message: str = ""
    scoreboard: dict[str, object] | None = None


class WindowCaptureConfigRequest(BaseModel):
    x: int = Field(ge=0, default=0)
    y: int = Field(ge=0, default=0)
    width: int = Field(ge=100, default=1280)
    height: int = Field(ge=100, default=720)
    enabled: bool = True


class WindowCaptureConfigResponse(BaseModel):
    enabled: bool
    x: int
    y: int
    width: int
    height: int
    source: str = "fullscreen"


class TacticalRecognitionResult(BaseModel):
    match_state: str
    phase: str
    game_window_detected: bool
    hud_detected: bool
    tactical_focus: list[str]
    goalx_score: int = Field(ge=0, le=100)
    pitch_scene: LivePitchScene | None = None


class CaptureWindowTargetRequest(BaseModel):
    title_contains: str = Field(min_length=2, max_length=120)


class CaptureWindowTargetResponse(BaseModel):
    active: bool
    title_contains: str
    matched: bool
    region: dict[str, int] | None = None


class CaptureWindowCandidate(BaseModel):
    title: str
    x: int
    y: int
    width: int
    height: int


class CaptureLoopRequest(BaseModel):
    interval_seconds: float = Field(ge=0.2, le=10.0, default=0.5)


class CaptureLoopStatus(BaseModel):
    running: bool
    interval_seconds: float = Field(ge=0.2, le=10.0)
    target_window: str | None = None
