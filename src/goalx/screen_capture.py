from __future__ import annotations

import ctypes
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from ctypes import wintypes

from mss import mss
from PIL import Image, ImageStat

from goalx.hud_text import PopupDetection, PopupTemplateStore, detect_action_popup
from goalx.kits import KitCalibrator
from goalx.models import BlueLockStats, KitProfile, LivePitchScene
from goalx.scoreboard import ScoreboardResult, read_scoreboard
from goalx.tracking import MatchTracker
from goalx.vision import estimate_pitch_scene, rematch_visibility_score
from goalx.performance import axes_from_tracking, live_axes_from_capture, axes_from_scoreboard, merge_axes

GAMEPLAY_STREAK_REQUIRED = 2
MENU_FRAMES_FOR_NEW_MATCH = 12
SCOREBOARD_CHECK_EVERY_SECONDS = 2.0
SCOREBOARD_MIN_GAMEPLAY_SECONDS = 45.0
# Below this many minutes of tracked play the legacy scene heuristics still carry weight.
TRACKING_TAKEOVER_MINUTES = 2.0


class ScreenCaptureService:
    def __init__(self, base_dir: str | os.PathLike[str] = "data/live-capture") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_frame_path: Path | None = None
        self._last_motion_score: float = 0.0
        self._last_hud_score: float = 0.0
        self._last_timestamp: datetime | None = None
        self._last_window_match: dict[str, int | str] | None = None
        self._last_error: str | None = None
        self._previous_gray_bytes: bytes | None = None
        self._crop_region: tuple[int, int, int, int] | None = None
        self._target_window_title_contains: str | None = None
        self._capture_interval_seconds: float = 0.5
        self._capture_thread: threading.Thread | None = None
        self._capture_stop_event = threading.Event()
        self._last_scene: dict[str, object] | None = None
        self._kept_frames = 8
        self._session_axes: dict[str, float] | None = None
        self._session_samples = 0
        self._last_region: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._kit_calibrator = KitCalibrator()
        self._gameplay_streak = 0
        self._menu_streak = 0
        self._last_frame_kind = "unknown"
        self._last_match_state = "waiting_for_match"
        self._tracker = MatchTracker()
        self._popup_store = PopupTemplateStore(self.base_dir.parent / "popup-templates.json")
        self._last_popup: PopupDetection | None = None
        self._last_popup_label_at: tuple[str, float] | None = None
        self._last_image: Image.Image | None = None
        self._last_scoreboard: ScoreboardResult | None = None
        self._scoreboard_checked_at: float = 0.0
        self._scoreboard_ingested_for: str | None = None
        self._self_name: str | None = None
        self._non_gameplay_streak = 0

    # -- player tracking / popups / scoreboard ------------------------------------

    def set_self_name(self, name: str | None) -> None:
        self._self_name = (name or "").strip() or None

    def tracking_metrics(self) -> dict[str, object]:
        return self._tracker.metrics()

    def tracking_axes(self) -> dict[str, float] | None:
        metrics = self._tracker.metrics()
        if float(metrics.get("gameplay_minutes", 0.0)) <= 0.0:
            return None
        return axes_from_tracking(metrics)

    def last_popup(self) -> dict[str, object] | None:
        return None if self._last_popup is None else self._last_popup.as_dict()

    def learn_popup(self, label: str) -> tuple[int, dict[str, object] | None]:
        """Learn the popup currently on screen under ``label`` (e.g. "pass")."""
        detection = self._last_popup
        if self._last_image is not None:
            fresh = detect_action_popup(self._last_image)
            if fresh is not None:
                detection = fresh
        if detection is None:
            return len(self._popup_store.templates.get(label.lower(), [])), None
        count = self._popup_store.learn(label, detection)
        return count, detection.as_dict()

    def popup_templates(self) -> dict[str, int]:
        return self._popup_store.labels()

    def last_scoreboard(self) -> ScoreboardResult | None:
        return self._last_scoreboard

    def read_scoreboard_now(self) -> ScoreboardResult | None:
        if self._last_image is None:
            return None
        result = read_scoreboard(self._last_image, self_name=self._self_name)
        if result.detected:
            self._last_scoreboard = result
        return result

    def take_scoreboard_for_ingest(self) -> ScoreboardResult | None:
        """Return a detected scoreboard once (so a results screen is stored a single time)."""
        result = self._last_scoreboard
        if result is None or not result.detected:
            return None
        if self._scoreboard_ingested_for == result.captured_at:
            return None
        self._scoreboard_ingested_for = result.captured_at
        return result

    def scoreboard_axes(self) -> dict[str, float] | None:
        if self._last_scoreboard is None or not self._last_scoreboard.detected:
            return None
        axes = axes_from_scoreboard(self._last_scoreboard.comparisons)
        return axes or None

    def reset_match_tracking(self) -> None:
        self._tracker.reset()
        self._last_scoreboard = None
        self._scoreboard_ingested_for = None

    # -- kit colours -------------------------------------------------------------

    def kit_profile(self) -> KitProfile | None:
        return self._kit_calibrator.profile()

    def reset_kits(self) -> KitProfile | None:
        self._kit_calibrator.reset()
        return None

    def override_kits(
        self,
        team_hue: float | None = None,
        opp_hue: float | None = None,
        team_label: str | None = None,
        opp_label: str | None = None,
    ) -> KitProfile | None:
        return self._kit_calibrator.override(team_hue, opp_hue, team_label, opp_label)

    def note_match_state(self, match_state: str) -> None:
        """Called by the status builder; a return to the lobby resets the per-match kit profile."""
        previous = self._last_match_state
        self._last_match_state = match_state
        if previous == "match_live" and match_state == "waiting_for_match":
            self._kit_calibrator.reset()
            self._tracker.reset()

    def last_frame_kind(self) -> str:
        return self._last_frame_kind

    def gameplay_active(self) -> bool:
        return self._gameplay_streak >= GAMEPLAY_STREAK_REQUIRED

    def set_crop_region(self, x: int, y: int, width: int, height: int) -> tuple[int, int, int, int]:
        self._crop_region = (x, y, width, height)
        return self._crop_region

    def get_crop_region(self) -> tuple[int, int, int, int]:
        if self._crop_region is not None and self._crop_region[2] > 0 and self._crop_region[3] > 0:
            return self._crop_region
        if self._last_region[2] > 0 and self._last_region[3] > 0:
            return self._last_region
        if self._last_window_match is not None:
            return _window_region(self._last_window_match)
        return (0, 0, 0, 0)

    def list_windows(self, query: str | None = None, limit: int = 50) -> list[dict[str, int | str]]:
        items = _list_visible_windows()
        if query and query.strip():
            lower = query.strip().lower()
            items = [item for item in items if lower in str(item["title"]).lower()]
        return items[:limit]

    def set_window_target(self, title_contains: str) -> dict[str, object]:
        title = title_contains.strip()
        if not title:
            raise ValueError("title_contains cannot be blank.")
        self._target_window_title_contains = title
        window = _find_window_by_title(title)
        self._last_window_match = window
        region = None
        if window is not None:
            capture_region = _window_region(window)
            self._crop_region = capture_region
            self._last_region = capture_region
            region = _region_to_dict(capture_region)
        return {
            "active": True,
            "title_contains": title,
            "matched": window is not None,
            "region": region,
        }

    def clear_window_target(self) -> dict[str, object]:
        self._target_window_title_contains = None
        self._last_window_match = None
        return {"active": False, "title_contains": "", "matched": False, "region": None}

    def get_window_target(self) -> str | None:
        return self._target_window_title_contains

    def start_live_capture(self, interval_seconds: float = 0.5) -> dict[str, object]:
        if interval_seconds < 0.2 or interval_seconds > 10.0:
            raise ValueError("interval_seconds must be between 0.2 and 10.0.")
        with self._lock:
            self._capture_interval_seconds = interval_seconds
            if self._capture_thread is not None and self._capture_thread.is_alive():
                return self.capture_loop_status()
            self._capture_stop_event.clear()
            self._session_axes = None
            self._session_samples = 0
            self._kit_calibrator.reset()
            self._tracker.reset()
            self._last_scoreboard = None
            self._scoreboard_ingested_for = None
            self._gameplay_streak = 0
            self._menu_streak = 0
            self._non_gameplay_streak = 0
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="goalx-live-capture")
            self._capture_thread.start()
        return self.capture_loop_status()

    def stop_live_capture(self) -> dict[str, object]:
        with self._lock:
            thread = self._capture_thread
            self._capture_stop_event.set()
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            self._capture_thread = None
        return self.capture_loop_status()

    def capture_loop_status(self) -> dict[str, object]:
        running = bool(self._capture_thread and self._capture_thread.is_alive() and not self._capture_stop_event.is_set())
        return {
            "running": running,
            "interval_seconds": self._capture_interval_seconds,
            "target_window": self._target_window_title_contains,
        }

    def capture_latest(self, region: tuple[int, int, int, int] | None = None) -> dict[str, object]:
        with self._lock:
            if self._target_window_title_contains:
                match = _find_window_by_title(self._target_window_title_contains)
                self._last_window_match = match
            with mss() as sct:
                monitor = sct.monitors[1]
                capture_region = self._resolve_capture_region(region, monitor)
                image = None
                hwnd = int(self._last_window_match["hwnd"]) if self._last_window_match and "hwnd" in self._last_window_match else 0
                if hwnd:
                    image = _capture_window_image(hwnd)
                if image is None:
                    monitor_region = {
                        "left": capture_region[0],
                        "top": capture_region[1],
                        "width": capture_region[2],
                        "height": capture_region[3],
                    }
                    screenshot = sct.grab(monitor_region)
                    image = Image.frombytes("RGBA", screenshot.size, screenshot.bgra, "raw", "BGRA").convert("RGB")
            self._last_region = capture_region
            timestamp = datetime.now(timezone.utc)
            # JPEG keeps the save under ~40 ms at 3440x1440 (PNG was ~400 ms), which
            # is what allows the 2-4 fps cadence the movement metrics need.
            file_name = f"frame_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            target_path = self.base_dir / file_name
            image.save(target_path, quality=88)

            motion_score = self._compute_motion_score(image)
            hud_score = self._estimate_hud_score(image)
            scene = estimate_pitch_scene(image, kit_profile=self._kit_calibrator.profile())
            self._last_image = image
            self._apply_frame(image, scene, motion_score, hud_score, timestamp.timestamp())
            self._last_frame_path = target_path
            self._last_motion_score = motion_score
            self._last_hud_score = hud_score
            self._last_timestamp = timestamp
            self._last_error = None
            self._prune_old_frames()

            return {
                "captured_at": timestamp.isoformat(),
                "file_name": file_name,
                "file_path": str(target_path),
                "motion_score": round(motion_score, 4),
                "hud_score": round(hud_score, 4),
                "image_size": {"width": image.width, "height": image.height},
                "region": {
                    "x": capture_region[0],
                    "y": capture_region[1],
                    "width": capture_region[2],
                    "height": capture_region[3],
                },
                "window_target": self._target_window_title_contains or "",
                "window_detected": self._last_window_match is not None,
                "pitch_scene": self._last_scene,
                "session_samples": self._session_samples,
            }

    def latest_frame_path(self) -> str:
        if self._last_frame_path is None:
            return ""
        return str(self._last_frame_path)

    def latest_metadata(self) -> dict[str, object]:
        if self._last_timestamp is None:
            return {
                "captured_at": None,
                "motion_score": 0.0,
                "hud_score": 0.0,
                "window_detected": False,
                "window_target": self._target_window_title_contains or "",
                "image_size": {"width": 0, "height": 0},
                "last_error": self._last_error,
                "pitch_scene": None,
                "session_samples": self._session_samples,
                "region": _region_to_dict(self._last_region),
            }
        return {
            "captured_at": self._last_timestamp.isoformat(),
            "motion_score": round(self._last_motion_score, 4),
            "hud_score": round(self._last_hud_score, 4),
            "window_detected": self._last_window_match is not None,
            "window_target": self._target_window_title_contains or "",
            "image_size": self._image_size_for_latest(),
            "last_error": self._last_error,
            "pitch_scene": self._last_scene,
            "session_samples": self._session_samples,
            "region": _region_to_dict(self._last_region),
        }

    def latest_scene(self) -> dict[str, object] | None:
        return self._last_scene

    def session_blue_lock(self) -> BlueLockStats | None:
        axes, _sources = self.session_axes_with_sources()
        if axes is None:
            return None
        return BlueLockStats(**{key: round(max(0.0, min(100.0, axes[key]))) for key in axes})

    def session_axes_with_sources(self) -> tuple[dict[str, float] | None, dict[str, str]]:
        """Live axes = tracked events (primary) + scoreboard lobby rank (when a results screen was read).

        The legacy scene heuristics only fill in while the tracker has under
        ``TRACKING_TAKEOVER_MINUTES`` of play.
        """
        keys = ("speed", "offense", "shoot", "dribble", "passing", "defense")
        legacy = None
        if self._session_samples >= 2 and self._session_axes is not None:
            legacy = {key: float(self._session_axes[key]) for key in keys}
        tracked = self.tracking_axes()
        minutes = float(self._tracker.metrics().get("gameplay_minutes", 0.0)) if tracked else 0.0
        sources: dict[str, str] = {}
        if tracked is None and legacy is None:
            return None, sources
        if tracked is None:
            axes = dict(legacy)  # type: ignore[arg-type]
            sources = {key: "scene" for key in keys}
        elif legacy is None:
            axes = dict(tracked)
            sources = {key: "tracking" for key in keys}
        else:
            takeover = max(0.0, min(1.0, minutes / TRACKING_TAKEOVER_MINUTES))
            axes = merge_axes(legacy, tracked, secondary_weight=0.35 + 0.65 * takeover)
            sources = {key: "tracking" if takeover >= 0.5 else "tracking+scene" for key in keys}
        board = self.scoreboard_axes()
        if board:
            axes = merge_axes(axes, board, secondary_weight=0.5)
            for key in board:
                sources[key] = sources.get(key, "tracking") + "+scoreboard"
        return axes, sources

    def _apply_frame(
        self,
        image: Image.Image,
        scene: LivePitchScene,
        motion_score: float,
        hud_score: float,
        timestamp: float | None = None,
    ) -> None:
        """Cutscene gate: only consecutive gameplay frames feed kits, session stats and the tactical scene."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp()
        self._last_frame_kind = scene.frame_kind
        if scene.frame_kind == "gameplay":
            self._gameplay_streak += 1
            self._menu_streak = 0
            self._non_gameplay_streak = 0
        else:
            self._gameplay_streak = 0
            self._non_gameplay_streak += 1
            if scene.frame_kind == "menu":
                self._menu_streak += 1
                if self._menu_streak == MENU_FRAMES_FOR_NEW_MATCH:
                    self._kit_calibrator.reset()
            else:
                self._menu_streak = 0
            self._maybe_read_scoreboard(image, timestamp)

        if self._gameplay_streak >= GAMEPLAY_STREAK_REQUIRED:
            self._kit_calibrator.observe(image, scene.frame_kind)
            self._ingest_session_stats(motion_score, hud_score, scene)
            self._observe_popup(image, timestamp)
            self_xy = (scene.self_x, scene.self_y) if scene.self_x is not None and scene.self_y is not None else None
            weight = 1.0 if scene.source == "fused" else 0.6 if scene.source == "minimap" else 0.0
            if weight > 0.0 and (scene.teammates or scene.opponents):
                self._tracker.observe(scene, timestamp, self_xy=self_xy, gameplay=True, weight=weight)
            self._last_scene = scene.model_dump()
            return

        # Non-gameplay frame: break trajectories so cutscene jumps never count as movement.
        self._tracker.observe(scene, timestamp, gameplay=False)

        if self._last_scene is not None:
            held = dict(self._last_scene)
            held["gameplay_detected"] = False
            held["frame_kind"] = scene.frame_kind
            self._last_scene = held
        else:
            stale = scene.model_copy(update={"gameplay_detected": False})
            self._last_scene = stale.model_dump()

    def _observe_popup(self, image: Image.Image, timestamp: float) -> None:
        detection = detect_action_popup(image)
        if detection is None:
            self._last_popup = None
            return
        detection = self._popup_store.classify(detection)
        self._last_popup = detection
        if detection.confidence < 0.4:
            return
        # A popup stays on screen for several frames; count each label once per burst.
        last = self._last_popup_label_at
        if last is not None and last[0] == detection.label and timestamp - last[1] <= 1.8:
            self._last_popup_label_at = (detection.label, timestamp)
            return
        self._last_popup_label_at = (detection.label, timestamp)
        self._tracker.record_popup(detection.label, timestamp)

    def _maybe_read_scoreboard(self, image: Image.Image, timestamp: float) -> None:
        """On non-gameplay frames after a reasonable spell of play, look for the results table."""
        if self._non_gameplay_streak < GAMEPLAY_STREAK_REQUIRED:
            return
        if timestamp - self._scoreboard_checked_at < SCOREBOARD_CHECK_EVERY_SECONDS:
            return
        played = float(self._tracker.metrics().get("gameplay_minutes", 0.0)) * 60.0
        if played < SCOREBOARD_MIN_GAMEPLAY_SECONDS and self._last_scoreboard is None:
            return
        self._scoreboard_checked_at = timestamp
        try:
            result = read_scoreboard(image, self_name=self._self_name)
        except Exception:
            return
        if result.detected:
            previous = self._last_scoreboard
            # Prefer the read with the most rows/stats (the table may animate in).
            if previous is None or _scoreboard_richness(result) >= _scoreboard_richness(previous):
                self._last_scoreboard = result

    def _ingest_session_stats(self, motion_score: float, hud_score: float, scene: object) -> None:
        if hud_score < 0.15:
            return
        pitch_scene = scene if isinstance(scene, LivePitchScene) else None
        if pitch_scene is not None and pitch_scene.confidence < 0.2 and pitch_scene.detection_count < 2:
            return
        live = live_axes_from_capture(motion_score, hud_score, pitch_scene)
        if self._session_axes is None:
            self._session_axes = live
        else:
            for key, value in live.items():
                previous = self._session_axes[key]
                self._session_axes[key] = previous * 0.86 + value * 0.14
        self._session_samples += 1

    def _capture_loop(self) -> None:
        while not self._capture_stop_event.is_set():
            try:
                self.capture_latest()
            except Exception as exc:
                self._last_error = str(exc)
            self._capture_stop_event.wait(self._capture_interval_seconds)

    def _resolve_capture_region(
        self,
        explicit_region: tuple[int, int, int, int] | None,
        monitor: dict[str, int] | object,
    ) -> tuple[int, int, int, int]:
        if explicit_region is not None:
            return explicit_region
        if self._crop_region is not None and self._crop_region[2] > 0 and self._crop_region[3] > 0:
            return self._crop_region
        if self._target_window_title_contains:
            match = _find_window_by_title(self._target_window_title_contains)
            self._last_window_match = match
            if match is not None:
                return _window_region(match)
        if isinstance(monitor, dict):
            return (monitor["left"], monitor["top"], monitor["width"], monitor["height"])
        return monitor

    def _compute_motion_score(self, image: Image.Image) -> float:
        current = image.convert("L").resize((160, 90)).tobytes()
        if self._previous_gray_bytes is None:
            self._previous_gray_bytes = current
            return 0.0
        previous = self._previous_gray_bytes
        self._previous_gray_bytes = current
        diffs = 0
        for a, b in zip(current, previous):
            diffs += abs(a - b)
        max_possible = len(current) * 255
        if max_possible == 0:
            return 0.0
        return max(0.0, min(1.0, diffs / max_possible))

    def _estimate_hud_score(self, image: Image.Image) -> float:
        rematch_score = rematch_visibility_score(image)
        if rematch_score >= 0.18:
            return rematch_score
        width, height = image.size
        top_band_height = max(50, int(height * 0.16))
        top_band = image.crop((0, 0, width, top_band_height)).convert("L")
        stat = ImageStat.Stat(top_band)
        variance = float(stat.var[0]) if stat.var else 0.0
        normalized_variance = min(1.0, variance / 1800.0)
        sample = top_band.resize((120, 24))
        pixels = list(sample.getdata())
        edge_sum = 0.0
        for row in range(24):
            row_start = row * 120
            for col in range(1, 120):
                edge_sum += abs(pixels[row_start + col] - pixels[row_start + col - 1])
        edge_score = min(1.0, edge_sum / (24 * 119 * 255 * 0.30))
        generic = (normalized_variance * 0.45) + (edge_score * 0.55)
        return max(0.0, min(1.0, min(generic * 0.35, rematch_score + 0.08)))

    def _prune_old_frames(self) -> None:
        frames = sorted(list(self.base_dir.glob("frame_*.png")) + list(self.base_dir.glob("frame_*.jpg")))
        extra = frames[: max(0, len(frames) - self._kept_frames)]
        for path in extra:
            try:
                path.unlink()
            except OSError:
                continue

    def _image_size_for_latest(self) -> dict[str, int]:
        if self._last_frame_path is None:
            return {"width": 0, "height": 0}
        with Image.open(self._last_frame_path) as img:
            return {"width": img.width, "height": img.height}


def _scoreboard_richness(result: ScoreboardResult) -> int:
    return sum(len(row.stats) for row in result.rows) + (5 if any(row.is_self for row in result.rows) else 0)


def _list_visible_windows() -> list[dict[str, int | str]]:
    if os.name != "nt":
        return []
    user32 = ctypes.windll.user32
    windows: list[dict[str, int | str]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        title = title_buffer.value.strip()
        if not title:
            return True
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width < 300 or height < 200:
            return True
        windows.append(
            {
                "title": title,
                "hwnd": int(hwnd) if hwnd is not None else 0,
                "x": int(rect.left),
                "y": int(rect.top),
                "width": int(width),
                "height": int(height),
            }
        )
        return True

    user32.EnumWindows(enum_callback, 0)
    return sorted(windows, key=lambda item: str(item["title"]).lower())


def _find_window_by_title(title_contains: str) -> dict[str, int | str] | None:
    lower = title_contains.strip().lower()
    ranked: list[tuple[int, dict[str, int | str]]] = []
    for window in _list_visible_windows():
        title = str(window["title"]).strip().lower()
        area = int(window["width"]) * int(window["height"])
        if title == lower:
            score = 1_000_000 + area
        elif title.startswith(lower):
            score = 500_000 + area
        elif lower in title:
            score = 100_000 + area
        else:
            continue
        ranked.append((score, window))
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[0])[1]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def _capture_window_image(hwnd: int) -> Image.Image | None:
    if os.name != "nt" or hwnd <= 0:
        return None
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width < 32 or height < 32:
        return None
    hdc = user32.GetWindowDC(hwnd)
    if not hdc:
        return None
    memdc = gdi32.CreateCompatibleDC(hdc)
    hbmp = gdi32.CreateCompatibleBitmap(hdc, width, height)
    previous = gdi32.SelectObject(memdc, hbmp)
    printed = user32.PrintWindow(hwnd, memdc, 2)
    if printed == 0:
        printed = user32.PrintWindow(hwnd, memdc, 0)
    bmi = _BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0
    buffer = ctypes.create_string_buffer(width * height * 4)
    copied = gdi32.GetDIBits(memdc, hbmp, 0, height, buffer, ctypes.byref(bmi), 0)
    gdi32.SelectObject(memdc, previous)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(memdc)
    user32.ReleaseDC(hwnd, hdc)
    if printed == 0 or copied == 0:
        return None
    image = Image.frombuffer("RGBA", (width, height), buffer, "raw", "BGRA", 0, 1).convert("RGB")
    sample = image.resize((48, 27))
    mean = sum(sum(pixel) for pixel in sample.getdata()) / (48 * 27 * 3)
    if mean < 5:
        return None
    return image


def _window_region(window: dict[str, int | str]) -> tuple[int, int, int, int]:
    return (int(window["x"]), int(window["y"]), int(window["width"]), int(window["height"]))


def _region_to_dict(region: tuple[int, int, int, int]) -> dict[str, int]:
    return {"x": region[0], "y": region[1], "width": region[2], "height": region[3]}


screen_capture_service = ScreenCaptureService()
