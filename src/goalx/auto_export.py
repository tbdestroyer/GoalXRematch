from __future__ import annotations

import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from goalx.models import AutoExportRunResult
from goalx.storage import (
    get_auto_export_config,
    get_rematch_connection,
    is_source_file_copied,
    mark_source_file_copied,
)

MATCH_END_MARKERS = (
    "match ended",
    "postmatch",
    "game over",
    "final whistle",
)


class AutoExportService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_log_file: Path | None = None
        self._last_log_pos: int = 0
        self.last_error: str | None = None
        self.last_match_end_at: datetime | None = None
        self.last_copy_at: datetime | None = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="goalx-auto-export", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._stop_event.set()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=2)

    @property
    def running(self) -> bool:
        return self._running

    def run_once(self) -> AutoExportRunResult:
        cfg = get_auto_export_config()
        if cfg is None or not bool(cfg["enabled"]):
            raise RuntimeError("Auto-export is not configured or disabled.")

        rematch_cfg = get_rematch_connection()
        if rematch_cfg is None:
            raise RuntimeError("Rematch account connection is not configured.")

        source_dir = Path(str(cfg["source_directory"]))
        log_dir = Path(str(cfg["log_directory"]))
        export_dir = Path(str(rematch_cfg["export_directory"]))
        if not source_dir.exists() or not source_dir.is_dir():
            raise RuntimeError("Auto-export source directory does not exist.")
        if not log_dir.exists() or not log_dir.is_dir():
            raise RuntimeError("Auto-export log directory does not exist.")
        if not export_dir.exists() or not export_dir.is_dir():
            raise RuntimeError("GoalX export directory does not exist.")

        match_end_detected = self._consume_log_match_end(log_dir)
        copied_files: list[str] = []
        if match_end_detected:
            copied_files = self._copy_latest_new_exports(source_dir, export_dir)

        message = "No match-end signal found in logs."
        if match_end_detected and copied_files:
            message = "Match end detected and export file copied."
        elif match_end_detected:
            message = "Match end detected but no new source export file found."

        return AutoExportRunResult(
            match_end_detected=match_end_detected,
            copied_files=copied_files,
            message=message,
        )

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            cfg = get_auto_export_config()
            if cfg is None or not bool(cfg["enabled"]):
                self._stop_event.wait(5)
                continue
            poll_seconds = int(cfg["poll_seconds"])
            if poll_seconds < 3:
                poll_seconds = 3
            try:
                self.run_once()
                self.last_error = None
            except RuntimeError as exc:
                self.last_error = str(exc)
            self._stop_event.wait(poll_seconds)

    def _consume_log_match_end(self, log_dir: Path) -> bool:
        log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
        if not log_files:
            raise RuntimeError("No .log files found in auto-export log directory.")
        latest = log_files[-1]
        if self._last_log_file is None or latest.resolve() != self._last_log_file.resolve():
            self._last_log_file = latest
            self._last_log_pos = 0

        with latest.open("r", encoding="utf-8", errors="ignore") as file_handle:
            file_handle.seek(self._last_log_pos)
            chunk = file_handle.read()
            self._last_log_pos = file_handle.tell()
        if not chunk:
            return False
        lowered = chunk.lower()
        if any(marker in lowered for marker in MATCH_END_MARKERS):
            self.last_match_end_at = datetime.now(timezone.utc)
            return True
        return False

    def _copy_latest_new_exports(self, source_dir: Path, export_dir: Path) -> list[str]:
        source_files = sorted(
            [*source_dir.glob("*.csv"), *source_dir.glob("*.json")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        copied: list[str] = []
        for source_file in source_files:
            source_resolved = str(source_file.resolve())
            if is_source_file_copied(source_resolved):
                continue
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            destination_name = f"auto_{timestamp}_{source_file.name}"
            destination_path = export_dir / destination_name
            shutil.copy2(source_file, destination_path)
            mark_source_file_copied(source_resolved, destination_name)
            self.last_copy_at = datetime.now(timezone.utc)
            copied.append(destination_name)
            break
        return copied

