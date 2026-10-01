from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from goalx.importers import parse_match_file
from goalx.models import SyncRunResult
from goalx.storage import (
    get_rematch_connection,
    is_file_processed,
    mark_file_processed,
    upsert_match_stats,
)


class RematchSyncService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_sync_at: datetime | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="goalx-rematch-sync", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._stop_event.set()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=2)

    def run_sync_once(self) -> SyncRunResult:
        cfg = get_rematch_connection()
        if cfg is None:
            raise RuntimeError("No Rematch account connection configured.")
        directory = Path(str(cfg["export_directory"]))
        if not directory.exists() or not directory.is_dir():
            raise RuntimeError("Configured export directory does not exist.")

        files = sorted(
            [*directory.glob("*.json"), *directory.glob("*.csv")],
            key=lambda p: p.name,
        )
        total_rows = 0
        match_count = 0
        updated_players: set[str] = set()
        processed_files: list[str] = []

        for file_path in files:
            full_path = str(file_path.resolve())
            if is_file_processed(full_path):
                continue

            payloads = parse_match_file(file_path)
            if not payloads:
                mark_file_processed(full_path, "__empty__")
                processed_files.append(file_path.name)
                continue
            last_match_id = payloads[-1].match_id
            for payload in payloads:
                players = upsert_match_stats(payload.match_id, payload.players, payload.played_at)
                match_count += 1
                total_rows += len(payload.players)
                updated_players.update(players)
            mark_file_processed(full_path, last_match_id)
            processed_files.append(file_path.name)

        self.last_sync_at = datetime.now(timezone.utc)
        self.last_error = None
        return SyncRunResult(
            matches_ingested=match_count,
            rows_processed=total_rows,
            players_updated=sorted(updated_players),
            files_processed=processed_files,
        )

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            cfg = get_rematch_connection()
            if cfg is None or not bool(cfg["auto_sync"]):
                self._stop_event.wait(5)
                continue
            poll_seconds = int(cfg["poll_seconds"])
            if poll_seconds < 5:
                poll_seconds = 5
            try:
                self.run_sync_once()
            except RuntimeError as exc:
                self.last_error = str(exc)
            self._stop_event.wait(poll_seconds)
