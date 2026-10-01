# GoalX for Rematch (MVP)

Lightweight GoalX tactical assistant for **Rematch on PC**, focused on **positioning improvements** with **low data usage**.

## Why this is low-data
- Rule-based tactical engine (no heavy model downloads)
- Compact request/response JSON over a single API endpoint
- Stateless processing suitable for lightweight desktop tooling

## Current MVP features
- Analyze team snapshot and return a GoalX score
- Ingest per-match player stats and update rolling GoalX
- Generate a Blue-Lock-style radar graph (SVG)
- Connect a Rematch account profile + local export folder
- Auto-sync new game files and update after each match
- Windows auto-export helper (log-triggered copy from source exporter folder)
- Tactical suggestions based on:
  - teammate spacing
  - attacking width
  - half-space occupation
  - defensive compactness

## Tech stack
- Python 3.11+
- FastAPI
- Pytest

## Run locally
```powershell
python -m pip install -e .[dev]
python -m uvicorn goalx.main:app --reload --app-dir src
```

Open:
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/ui/live-capture` for the live match-capture overlay dashboard

## Test
```powershell
python -m pytest
```

## API
### `POST /analyze`
Input: phase, ball position, teammates, opponents  
Output: goalx_score (0-100), prioritized tactical suggestions, low_data_mode flag.

### `POST /ingest/match-stats`
Ingest one match of per-player stats (JSON).  
This updates each player's rolling GoalX basis using latest matches.

### `GET /players/{player_id}/goalx`
Returns Blue-Lock axis stats + overall GoalX score.

### `GET /players/{player_id}/blue-lock-radar.svg`
Returns an SVG radar chart like the reference style.

### `POST /connect/rematch-account`
Connects your Rematch PC profile to a local export folder, polling interval, and your linked `player_id`.

### `GET /connect/rematch-account/status`
Shows current connection state, last sync, and processed file count.

### `GET /connect/rematch-account/discover`
Lists player IDs discovered from recent ingested matches (to help you choose your linked `player_id`).

### `POST /sync/rematch-now`
Runs an immediate sync pass over new match export files in your export folder (`.json` or `.csv`).

### `GET /me/goalx`
Returns GoalX snapshot for the linked account player.

### `GET /me/blue-lock-radar.svg`
Returns Blue-Lock radar SVG for the linked account player.

### `GET /ui/demo`
Example UI screen for connection, sync, GoalX snapshot, and radar graph.

### `POST /auto-export/config`
Configures the built-in Windows auto-export helper:
- `source_directory`: folder where your external exporter writes files
- `log_directory`: folder containing `.log` files with match-end lines
- `poll_seconds`: helper polling interval

### `GET /auto-export/status`
Returns helper status, last match-end detection time, and copied-file count.

### `POST /auto-export/run-once`
Runs one immediate helper pass: detect match-end in logs and copy latest new `.csv`/`.json` from source folder into GoalX export folder.

### `GET /capture/windows`
Lists visible desktop windows (optional `query` filter). Use this to confirm your Rematch window title is discoverable.

### `POST /capture/window-target`
Sets live capture to follow a window title substring (example: `{"title_contains":"Rematch"}`).

### `POST /capture/start` and `POST /capture/stop`
Starts/stops continuous capture loop from the target Rematch window (or crop/fullscreen fallback).

## How to get data from Rematch on PC
Use a local desktop pipeline:
1. Connect once with `POST /connect/rematch-account` (`account_name`, `export_directory`, `linked_player_id`, `poll_seconds`).
2. After each game, have your exporter write a match file into `export_directory` (`.csv` recommended, `.json` also supported).
3. Auto-sync watcher detects new files and ingests them.
4. Optional immediate update: call `POST /sync/rematch-now`.
5. Read updated GoalX from `/me/goalx` (or `/players/{player_id}/goalx`).
6. Render/share graph from `/me/blue-lock-radar.svg` (or `/players/{player_id}/blue-lock-radar.svg`).

This keeps data usage low because only compact numeric summaries are stored/transmitted (no video upload required).

## Make it work with your actual live game
1. Start Rematch on your PC and leave it visible (windowed/fullscreen borderless both work).
2. Open `/ui/live-capture`.
3. Use **List windows** / the dropdown, then **Set window** with `Rematch` in the title field.
4. Click **Start capture** (backend live loop; the UI polls status and draws the overlay).
5. The pitch panel shows the latest captured frame plus estimated teammate (blue), opponent (red), and ball markers.
6. Coach suggestions now come from GoalX `/analyze` using those estimated positions.
7. Check `/live/capture/status`:
   - `window_detected: true` means GoalX found the Rematch window.
   - `hud_detected: true` means frame quality is good enough for live HUD recognition.
   - `pitch_scene` contains live coordinates and `confidence`.
8. Keep match exports enabled too (`/sync/rematch-now` or auto-sync) so Blue Lock radar stays grounded in real match stats.

## Windows custom helper workflow (option 3)
If Rematch itself does not provide direct API export, use:
1. Any local exporter that writes `.csv`/`.json` to a source folder.
2. Point GoalX auto-export helper at that source folder + a log folder containing match-end markers (`match ended`, `final whistle`, `postmatch`, `game over`).
3. Helper copies the newest not-yet-copied export file into [exports/](C:/RematchGoalX/exports).
4. Existing GoalX sync loop ingests it and updates `/me/goalx` automatically.

## About "real account connection"
At the moment, this connector is PC-local and production-safe: it links your account label to your player ID and continuously imports post-match stat files.  
If Rematch exposes an official account/stats API in the future, this architecture can swap the file-ingest source for API pulls without changing GoalX scoring endpoints.

## Example "after every game" files
CSV (recommended for auto-export tools):
```csv
match_id,played_at,player_id,minutes,shots,shots_on_target,goals,assists,key_passes,successful_dribbles,dribble_attempts,completed_passes,attempted_passes,tackles_won,interceptions,duels_won,duels_total,sprints,distance_m
m_3001,2026-09-29T06:15:00+00:00,bluelock_9,90,5,3,1,1,4,3,5,34,41,2,2,5,8,23,10700
```

JSON:
```json
{
  "match_id": "m_3001",
  "players": [
    {
      "player_id": "bluelock_9",
      "minutes": 90,
      "shots": 5,
      "shots_on_target": 3,
      "goals": 1,
      "assists": 1,
      "key_passes": 4,
      "successful_dribbles": 3,
      "dribble_attempts": 5,
      "completed_passes": 34,
      "attempted_passes": 41,
      "tackles_won": 2,
      "interceptions": 2,
      "duels_won": 5,
      "duels_total": 8,
      "sprints": 23,
      "distance_m": 10700
    }
  ]
}
```

## Research references used for direction
- socceraction (MIT): event-based action valuation primitives
- kloppy (BSD-3): provider-agnostic soccer data normalization
- LaurieOnTracking (MIT): tactical/pitch-control concepts
- best_lineup (MIT): formation optimization concept

These informed architecture choices; code in this repo is original MVP code.

## PC-first direction (Rematch)
- Prioritize desktop workflows (local match/session exports, keyboard-first controls)
- Keep runtime light so analysis can run alongside the game on PC
- Defer mobile/PWA-specific work
"# GoalXRematch" 
