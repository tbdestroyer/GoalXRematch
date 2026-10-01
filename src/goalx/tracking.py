"""Trajectory tracking for live Rematch capture.

Consumes fused ``LivePitchScene`` frames (pitch coordinates 0-100) over time and
derives player-centric events for the local player: passes, losses, shots,
recoveries, carries, take-ons, pace duels, sprints, pressing and shape metrics.

All distances are converted from pitch units to metres using the configured
pitch size so thresholds are stated in real-world terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from goalx.models import LivePitchScene, PlayerPosition

# Rematch 5v5 pitches are small; defaults approximate the arena/stadium size.
DEFAULT_PITCH_LENGTH_M = 60.0
DEFAULT_PITCH_WIDTH_M = 36.0

# Capture cadence above this makes velocity-based metrics meaningless.
MAX_TRACK_DT_SECONDS = 1.0
MIN_TRACK_DT_SECONDS = 0.05


@dataclass
class _Point:
    x: float
    y: float
    t: float


@dataclass
class _Carrier:
    side: str  # "self" | "team" | "opp" | "loose"
    player_id: str
    x: float
    y: float


@dataclass
class TrackingCounters:
    gameplay_seconds: float = 0.0
    tracked_seconds: float = 0.0
    frames: int = 0
    tracked_frames: int = 0
    self_frames: int = 0

    # SPEED
    distance_m: float = 0.0
    sprint_frames: int = 0
    moving_frames: int = 0
    speed_samples: list[float] = field(default_factory=list)
    pace_duels: int = 0
    pace_duels_won: int = 0
    runs_in_behind: int = 0

    # PASS
    passes_attempted: int = 0
    passes_completed: int = 0
    pass_kicks: int = 0  # balls played toward a teammate (HUD-confirmable)
    pass_kicks_confirmed: int = 0  # the game flashed "PASS"
    popups: dict[str, int] = field(default_factory=dict)
    forward_passes: int = 0
    switches: int = 0
    key_passes: int = 0
    pass_length_m: float = 0.0

    # SHOOT
    shots: int = 0
    shots_on_target: int = 0
    shot_quality_sum: float = 0.0
    goals: int = 0

    # DRIBBLE
    carries: int = 0
    carry_distance_m: float = 0.0
    carries_retained: int = 0
    carries_lost: int = 0
    take_ons: int = 0
    take_ons_won: int = 0
    losses: int = 0

    # OFFENSE
    team_possession_frames: int = 0
    involvement_frames: int = 0
    attacking_third_frames: int = 0
    forward_runs: int = 0
    box_touches: int = 0

    # DEFENSE
    opp_possession_frames: int = 0
    goal_side_frames: int = 0
    pressing_samples: list[float] = field(default_factory=list)
    carrier_distance_samples: list[float] = field(default_factory=list)
    compactness_samples: list[float] = field(default_factory=list)
    recoveries: int = 0
    tackles: int = 0
    interceptions: int = 0


class MatchTracker:
    """Accumulates player-centric events from a stream of pitch scenes."""

    def __init__(
        self,
        pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
        pitch_width_m: float = DEFAULT_PITCH_WIDTH_M,
        attack_direction: int = 1,
        sprint_speed_mps: float = 6.0,
        pace_duel_speed_mps: float = 5.0,
        carry_radius_m: float = 2.4,
        duel_radius_m: float = 3.6,
        shot_speed_mps: float = 11.0,
    ) -> None:
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.attack_direction = 1 if attack_direction >= 0 else -1
        self.sprint_speed_mps = sprint_speed_mps
        self.pace_duel_speed_mps = pace_duel_speed_mps
        self.carry_radius_m = carry_radius_m
        self.duel_radius_m = duel_radius_m
        self.shot_speed_mps = shot_speed_mps
        self.counters = TrackingCounters()
        self.reset_trajectory()

    # ------------------------------------------------------------------ state
    def reset_trajectory(self) -> None:
        self._prev_t: float | None = None
        self._prev_self: _Point | None = None
        self._prev_ball: _Point | None = None
        self._self_velocity: tuple[float, float] = (0.0, 0.0)
        self._ball_velocity: tuple[float, float] = (0.0, 0.0)
        self._carrier: _Carrier | None = None
        self._carrier_candidate: _Carrier | None = None
        self._carrier_candidate_frames = 0
        self._prev_opponents: list[PlayerPosition] = []
        self._prev_opp_velocity: dict[str, tuple[float, float]] = {}
        self._carry_active = False
        self._carry_distance_m = 0.0
        self._duel_targets: dict[str, tuple[float, float, float]] = {}
        self._forward_run_frames = 0
        self._pending_key_pass_until: float | None = None
        self._pending_shot_until: float | None = None
        self._pending_carry_outcome = False
        self._loose_origin: _Carrier | None = None
        self._loose_since: float | None = None
        self._loose_origin_me_close = False
        self._prev_carrier_distance_m: float | None = None
        self._last_pass_kick_t: float | None = None
        self._last_pass_kick_confirmed = False

    def reset(self) -> None:
        self.counters = TrackingCounters()
        self.reset_trajectory()

    def record_goal(self, scored_by_self_team: bool, timestamp: float | None = None) -> None:
        if scored_by_self_team and (
            self._pending_shot_until is not None and (timestamp is None or timestamp <= self._pending_shot_until)
        ):
            self.counters.goals += 1
            self._pending_shot_until = None

    def record_popup(self, label: str, timestamp: float, window_seconds: float = 3.0) -> None:
        """Feed an on-screen action popup (e.g. "pass") detected by ``hud_text``."""
        label = label.strip().lower()
        if not label or label in {"unknown", "long"}:
            return
        self.counters.popups[label] = self.counters.popups.get(label, 0) + 1
        if label != "pass":
            return
        c = self.counters
        recent = self._last_pass_kick_t is not None and timestamp - self._last_pass_kick_t <= window_seconds
        if recent and not self._last_pass_kick_confirmed:
            c.pass_kicks_confirmed += 1
            self._last_pass_kick_confirmed = True
        elif not recent:
            # The game confirmed a pass we did not see leave the foot (sparse frames): count both.
            c.pass_kicks += 1
            c.pass_kicks_confirmed += 1
            self._last_pass_kick_t = timestamp
            self._last_pass_kick_confirmed = True

    def _register_pass_kick(self, timestamp: float) -> None:
        self.counters.pass_kicks += 1
        self._last_pass_kick_t = timestamp
        self._last_pass_kick_confirmed = False

    def _teammate_on_line(self, ball: _Point, teammates: list[PlayerPosition], perp_m: float = 4.0) -> bool:
        vx, vy = self._ball_velocity
        speed = math.hypot(vx, vy)
        if speed < 2.0:
            return False
        for mate in teammates:
            dx_m, dy_m = self._to_m(mate.x - ball.x, mate.y - ball.y)
            along = (dx_m * vx + dy_m * vy) / speed
            perp = abs(dx_m * vy - dy_m * vx) / speed
            if along > 0.0 and perp <= perp_m:
                return True
        return False

    # --------------------------------------------------------------- geometry
    def _to_m(self, dx_units: float, dy_units: float) -> tuple[float, float]:
        return dx_units * self.pitch_length_m / 100.0, dy_units * self.pitch_width_m / 100.0

    def _dist_m(self, ax: float, ay: float, bx: float, by: float) -> float:
        dx, dy = self._to_m(ax - bx, ay - by)
        return math.hypot(dx, dy)

    def _goal_x(self, attacking: bool) -> float:
        forward = self.attack_direction > 0
        return 100.0 if (forward == attacking) else 0.0

    def _forward(self, dx_units: float) -> float:
        return dx_units * self.attack_direction

    # ----------------------------------------------------------------- update
    def observe(
        self,
        scene: LivePitchScene,
        timestamp: float,
        self_xy: tuple[float, float] | None = None,
        gameplay: bool = True,
        weight: float = 1.0,
    ) -> dict[str, float]:
        """Feed one scene. ``timestamp`` is seconds (monotonic or epoch)."""
        c = self.counters
        if not gameplay or weight <= 0.0:
            self.reset_trajectory()
            return self.metrics()

        c.frames += 1
        dt = None if self._prev_t is None else timestamp - self._prev_t
        tracked = dt is not None and MIN_TRACK_DT_SECONDS <= dt <= MAX_TRACK_DT_SECONDS
        if dt is not None and dt > MAX_TRACK_DT_SECONDS:
            self.reset_trajectory()
            dt = None
        if dt is not None:
            c.gameplay_seconds += dt
        if tracked:
            c.tracked_seconds += dt  # type: ignore[arg-type]
            c.tracked_frames += 1

        me = self._resolve_self(scene, self_xy)
        ball = _Point(scene.ball_x, scene.ball_y, timestamp)
        teammates = [p for p in scene.teammates if me is None or not _same_spot(p, me)]
        opponents = list(scene.opponents)

        if me is not None:
            c.self_frames += 1

        # --- velocities
        self_speed = 0.0
        if tracked and me is not None and self._prev_self is not None:
            vx, vy = self._velocity(self._prev_self, me, dt)  # type: ignore[arg-type]
            self._self_velocity = (vx * 0.5 + self._self_velocity[0] * 0.5, vy * 0.5 + self._self_velocity[1] * 0.5)
            self_speed = math.hypot(*self._self_velocity)
            step_m = self._dist_m(me.x, me.y, self._prev_self.x, self._prev_self.y)
            c.distance_m += step_m
            c.speed_samples.append(self_speed)
            if len(c.speed_samples) > 4000:
                del c.speed_samples[: len(c.speed_samples) - 4000]
            if self_speed >= 1.5:
                c.moving_frames += 1
            if self_speed >= self.sprint_speed_mps:
                c.sprint_frames += 1
        if tracked and self._prev_ball is not None:
            self._ball_velocity = self._velocity(self._prev_ball, ball, dt)  # type: ignore[arg-type]
        opp_velocity: dict[str, tuple[float, float]] = {}
        if tracked:
            prev_by_id = {p.player_id: p for p in self._prev_opponents}
            for opp in opponents:
                prev = prev_by_id.get(opp.player_id)
                if prev is None:
                    prev = _nearest(opp, self._prev_opponents, self._dist_m, max_m=8.0)
                if prev is not None:
                    opp_velocity[opp.player_id] = self._velocity(_Point(prev.x, prev.y, 0.0), _Point(opp.x, opp.y, 0.0), dt)  # type: ignore[arg-type]

        # --- carrier
        carrier = self._update_carrier(me, teammates, opponents, ball, tracked)
        prev_carrier = self._carrier
        if carrier is not None and (prev_carrier is None or carrier.side != prev_carrier.side or carrier.player_id != prev_carrier.player_id):
            self._on_carrier_change(prev_carrier, carrier, me, ball, timestamp, teammates)
        self._carrier = carrier if carrier is not None else self._carrier

        side = self._carrier.side if self._carrier else "loose"
        team_possession = side in {"self", "team"}
        opp_possession = side == "opp"

        # --- offense presence
        if me is not None and team_possession:
            c.team_possession_frames += 1
            if self._forward(ball.x - 50.0) >= 0 and self._dist_m(me.x, me.y, ball.x, ball.y) <= 9.0:
                c.involvement_frames += 1
            if self._forward(me.x - 50.0) >= 17.0:
                c.attacking_third_frames += 1
            if side == "team" and tracked:
                fwd_speed = self._forward(self._self_velocity[0])
                if fwd_speed >= 3.0:
                    self._forward_run_frames += 1
                elif self._forward_run_frames >= 2:
                    if self._forward(me.x - 50.0) >= 17.0:
                        c.forward_runs += 1
                    self._forward_run_frames = 0
                else:
                    self._forward_run_frames = 0

        # --- carry / dribble
        if me is not None and side == "self":
            if tracked and self_speed >= 1.5:
                if not self._carry_active:
                    self._carry_active = True
                    self._carry_distance_m = 0.0
                    c.carries += 1
                self._carry_distance_m += self._dist_m(me.x, me.y, self._prev_self.x, self._prev_self.y) if self._prev_self else 0.0
                c.carry_distance_m += self._dist_m(me.x, me.y, self._prev_self.x, self._prev_self.y) if self._prev_self else 0.0
        elif self._carry_active and side not in {"self", "loose"}:
            self._carry_active = False

        # --- duels (pace & take-on) while carrying or chasing a loose ball
        if me is not None and tracked and (side == "self" or (side == "loose" and self._dist_m(me.x, me.y, ball.x, ball.y) <= 8.0)):
            self._update_duels(me, opponents, opp_velocity, self_speed, timestamp)
        elif me is not None and tracked and side == "team":
            self._update_runs_in_behind(me, opponents, opp_velocity, self_speed)
        else:
            self._duel_targets.clear()

        # --- defense
        if me is not None and opp_possession and self._carrier is not None:
            c.opp_possession_frames += 1
            own_goal_x = self._goal_x(attacking=False)
            if _between(me.x, ball.x, own_goal_x, margin=1.0):
                c.goal_side_frames += 1
            carrier_d = self._dist_m(me.x, me.y, self._carrier.x, self._carrier.y)
            c.carrier_distance_samples.append(carrier_d)
            if tracked and self._prev_carrier_distance_m is not None:
                closing = (self._prev_carrier_distance_m - carrier_d) / dt  # type: ignore[operator]
                if carrier_d <= 15.0:
                    c.pressing_samples.append(closing)
            self._prev_carrier_distance_m = carrier_d
            if teammates:
                nearest_mate = min(self._dist_m(me.x, me.y, p.x, p.y) for p in teammates)
                c.compactness_samples.append(nearest_mate)
            for samples in (c.carrier_distance_samples, c.pressing_samples, c.compactness_samples):
                if len(samples) > 4000:
                    del samples[: len(samples) - 4000]
        else:
            self._prev_carrier_distance_m = None

        # --- pending windows
        if self._pending_key_pass_until is not None and timestamp > self._pending_key_pass_until:
            self._pending_key_pass_until = None
        if self._pending_shot_until is not None and timestamp > self._pending_shot_until:
            self._pending_shot_until = None
        if self._loose_since is not None and timestamp - self._loose_since > 3.0:
            self._loose_origin = None
            self._loose_since = None
            self._pending_carry_outcome = False

        self._prev_t = timestamp
        self._prev_self = me
        self._prev_ball = ball
        self._prev_opponents = opponents
        self._prev_opp_velocity = opp_velocity
        return self.metrics()

    # -------------------------------------------------------------- internals
    def _resolve_self(self, scene: LivePitchScene, self_xy: tuple[float, float] | None) -> _Point | None:
        t = self._prev_t if self._prev_t is not None else 0.0
        if self_xy is not None:
            return _Point(float(self_xy[0]), float(self_xy[1]), t)
        sx = getattr(scene, "self_x", None)
        sy = getattr(scene, "self_y", None)
        if sx is not None and sy is not None:
            return _Point(float(sx), float(sy), t)
        if self._prev_self is not None and scene.teammates:
            nearest = _nearest(PlayerPosition(player_id="me", role="mf", x=self._prev_self.x, y=self._prev_self.y), scene.teammates, self._dist_m, max_m=7.0)
            if nearest is not None:
                return _Point(nearest.x, nearest.y, t)
        return None

    def _velocity(self, prev: _Point, cur: _Point, dt: float) -> tuple[float, float]:
        dx, dy = self._to_m(cur.x - prev.x, cur.y - prev.y)
        return dx / dt, dy / dt

    def _update_carrier(
        self,
        me: _Point | None,
        teammates: list[PlayerPosition],
        opponents: list[PlayerPosition],
        ball: _Point,
        tracked: bool,
    ) -> _Carrier | None:
        candidates: list[tuple[float, _Carrier]] = []
        if me is not None:
            candidates.append((self._dist_m(me.x, me.y, ball.x, ball.y), _Carrier("self", "self", me.x, me.y)))
        for p in teammates:
            candidates.append((self._dist_m(p.x, p.y, ball.x, ball.y), _Carrier("team", p.player_id, p.x, p.y)))
        for p in opponents:
            candidates.append((self._dist_m(p.x, p.y, ball.x, ball.y), _Carrier("opp", p.player_id, p.x, p.y)))
        if not candidates:
            return None
        dist, nearest = min(candidates, key=lambda item: item[0])
        candidate = nearest if dist <= self.carry_radius_m else _Carrier("loose", "loose", ball.x, ball.y)
        if self._carrier is not None and candidate.side == self._carrier.side and candidate.player_id == self._carrier.player_id:
            self._carrier_candidate = None
            self._carrier_candidate_frames = 0
            return _Carrier(candidate.side, candidate.player_id, candidate.x, candidate.y)
        # Require persistence when frames are dense; accept immediately when sparse
        # or when the ball is clearly struck away from the previous carrier.
        needed = 2 if tracked and self._prev_t is not None else 1
        if self._carrier is not None and tracked:
            ball_speed = math.hypot(*self._ball_velocity)
            prev_pos = self._locate(self._carrier, me, teammates, opponents)
            if ball_speed >= 6.0 and prev_pos is not None and self._dist_m(prev_pos[0], prev_pos[1], ball.x, ball.y) > self.carry_radius_m * 1.5:
                needed = 1
        # Any run of frames disagreeing with the current carrier counts toward the switch;
        # the most recent candidate wins so loose/opponent flicker still resolves.
        if self._carrier_candidate is not None:
            self._carrier_candidate_frames += 1
        else:
            self._carrier_candidate_frames = 1
        self._carrier_candidate = candidate
        if self._carrier_candidate_frames >= needed or self._carrier is None:
            self._carrier_candidate = None
            self._carrier_candidate_frames = 0
            return candidate
        if self._carrier is not None:
            return _Carrier(self._carrier.side, self._carrier.player_id, self._carrier.x, self._carrier.y)
        return candidate

    def _locate(
        self,
        carrier: _Carrier,
        me: _Point | None,
        teammates: list[PlayerPosition],
        opponents: list[PlayerPosition],
    ) -> tuple[float, float] | None:
        if carrier.side == "self":
            return (me.x, me.y) if me is not None else (carrier.x, carrier.y)
        pool = teammates if carrier.side == "team" else opponents if carrier.side == "opp" else []
        for p in pool:
            if p.player_id == carrier.player_id:
                return (p.x, p.y)
        if pool:
            near = _nearest(PlayerPosition(player_id="c", role="mf", x=carrier.x, y=carrier.y), pool, self._dist_m, max_m=6.0)
            if near is not None:
                return (near.x, near.y)
        return (carrier.x, carrier.y) if carrier.side != "loose" else None

    def _on_carrier_change(
        self,
        prev: _Carrier | None,
        cur: _Carrier,
        me: _Point | None,
        ball: _Point,
        timestamp: float,
        teammates: list[PlayerPosition],
    ) -> None:
        c = self.counters
        prev_side = prev.side if prev else "loose"

        # Resolve where a loose ball came from (expires after a few seconds).
        origin: _Carrier | None = None
        origin_me_close = False
        if prev_side == "loose" and self._loose_origin is not None and self._loose_since is not None:
            if timestamp - self._loose_since <= 3.0:
                origin = self._loose_origin
                origin_me_close = self._loose_origin_me_close
            self._loose_origin = None
            self._loose_since = None
        elif prev is not None and prev_side != "loose":
            origin = prev
            origin_me_close = me is not None and self._dist_m(me.x, me.y, prev.x, prev.y) <= self.duel_radius_m

        # Ball is now loose: remember who released it.
        if cur.side == "loose":
            if prev is not None and prev_side != "loose":
                release_pos = self._locate_prev(prev, me)
                self._loose_origin = _Carrier(prev.side, prev.player_id, release_pos[0], release_pos[1])
                self._loose_since = timestamp
                self._loose_origin_me_close = me is not None and self._dist_m(me.x, me.y, release_pos[0], release_pos[1]) <= self.duel_radius_m
                if prev_side == "self":
                    shot = self._is_shot(ball, teammates)
                    if self._carry_active:
                        self._carry_active = False
                        if shot:
                            c.carries_retained += 1
                        else:
                            self._pending_carry_outcome = True
                    if shot:
                        self._register_shot(me or _Point(prev.x, prev.y, timestamp), ball, timestamp)
                        self._loose_origin = None  # a shot is resolved; don't count it as a pass later
                        self._loose_since = None
                    elif self._teammate_on_line(ball, teammates):
                        self._register_pass_kick(timestamp)
                elif prev_side == "team" and self._pending_key_pass_until is not None and timestamp <= self._pending_key_pass_until:
                    if self._is_shot(ball, teammates):
                        c.key_passes += 1
                        self._pending_key_pass_until = None
            return

        origin_side = origin.side if origin is not None else "none"

        # Ball left us and arrived somewhere.
        if origin_side == "self":
            if prev_side == "self" and cur.side == "team":
                self._register_pass_kick(timestamp)  # arrived at a teammate between frames
            if cur.side == "team":
                self._register_pass(origin, cur, timestamp, completed=True)
                if self._carry_active or self._pending_carry_outcome:
                    c.carries_retained += 1
            elif cur.side == "opp":
                direct_tackle = prev_side == "self" and origin_me_close
                if not direct_tackle:
                    self._register_pass(origin, cur, timestamp, completed=False)
                c.losses += 1
                if self._carry_active or self._pending_carry_outcome:
                    c.carries_lost += 1
            self._carry_active = False
            self._pending_carry_outcome = False
            return

        if cur.side == "self":
            if me is not None and self._forward(me.x - 50.0) >= 34.0 and abs(me.y - 50.0) <= 22.0:
                c.box_touches += 1
            if origin_side == "opp":
                c.recoveries += 1
                if origin_me_close:
                    c.tackles += 1
                else:
                    c.interceptions += 1
            return

    def _locate_prev(self, carrier: _Carrier, me: _Point | None) -> tuple[float, float]:
        if carrier.side == "self" and self._prev_self is not None:
            return (self._prev_self.x, self._prev_self.y)
        return (carrier.x, carrier.y)

    def _register_pass(self, origin: _Carrier, target: _Carrier, timestamp: float, completed: bool) -> None:
        c = self.counters
        c.passes_attempted += 1
        length = self._dist_m(origin.x, origin.y, target.x, target.y)
        c.pass_length_m += length
        if completed:
            c.passes_completed += 1
            if self._forward(target.x - origin.x) >= 5.0:
                c.forward_passes += 1
            if abs(target.y - origin.y) >= 30.0:
                c.switches += 1
            self._pending_key_pass_until = timestamp + 3.0

    def _is_shot(self, ball: _Point, teammates: list[PlayerPosition] | None = None) -> bool:
        vx, vy = self._ball_velocity
        speed = math.hypot(vx, vy)
        if speed < self.shot_speed_mps:
            return False
        if self._forward(vx) <= 0.0:
            return False
        goal_x = self._goal_x(attacking=True)
        if abs(goal_x - ball.x) * self.pitch_length_m / 100.0 > 32.0:
            return False  # too far out to be a shot in a 5v5 arena
        hit_y = self._goal_intersection_y(ball)
        if hit_y is None or not 25.0 <= hit_y <= 75.0:
            return False
        # A fast ball with a teammate on its line before the goal is a pass, not a shot.
        for mate in teammates or []:
            if self._forward(mate.x - ball.x) <= 0.0:
                continue
            dx_m, dy_m = self._to_m(mate.x - ball.x, mate.y - ball.y)
            along = (dx_m * vx + dy_m * vy) / speed
            perp = abs(dx_m * vy - dy_m * vx) / speed
            if along > 0.0 and perp <= 2.5:
                return False
        return True

    def _goal_intersection_y(self, ball: _Point) -> float | None:
        vx, vy = self._ball_velocity
        if abs(vx) < 1e-6:
            return None
        goal_x = self._goal_x(attacking=True)
        dx_m = (goal_x - ball.x) * self.pitch_length_m / 100.0
        t = dx_m / vx
        if t <= 0.0:
            return None
        dy_units = (vy * t) / (self.pitch_width_m / 100.0)
        return ball.y + dy_units

    def _register_shot(self, me: _Point, ball: _Point, timestamp: float) -> None:
        c = self.counters
        c.shots += 1
        hit_y = self._goal_intersection_y(ball)
        if hit_y is not None and 38.0 <= hit_y <= 62.0:
            c.shots_on_target += 1
        goal_x = self._goal_x(attacking=True)
        dist = self._dist_m(me.x, me.y, goal_x, 50.0)
        angle = abs(me.y - 50.0)
        quality = max(0.05, min(1.0, 1.15 - dist / 28.0 - angle / 120.0))
        c.shot_quality_sum += quality
        self._pending_shot_until = timestamp + 3.0

    def _update_duels(
        self,
        me: _Point,
        opponents: list[PlayerPosition],
        opp_velocity: dict[str, tuple[float, float]],
        self_speed: float,
        timestamp: float,
    ) -> None:
        c = self.counters
        vx, vy = self._self_velocity
        if self_speed < 1.0:
            self._duel_targets.clear()
            return
        ux, uy = vx / self_speed, vy / self_speed
        seen: set[str] = set()
        for opp in opponents:
            dx_m, dy_m = self._to_m(opp.x - me.x, opp.y - me.y)
            along = dx_m * ux + dy_m * uy
            dist = math.hypot(dx_m, dy_m)
            key = opp.player_id
            if key in self._duel_targets:
                start_along, start_t, start_speed = self._duel_targets[key]
                if along <= -1.8:
                    ovx, ovy = opp_velocity.get(key, (0.0, 0.0))
                    opp_speed = math.hypot(ovx, ovy)
                    if self_speed >= self.pace_duel_speed_mps and self_speed > opp_speed + 0.8:
                        c.pace_duels += 1
                        c.pace_duels_won += 1
                    else:
                        c.take_ons += 1
                        c.take_ons_won += 1
                    del self._duel_targets[key]
                    continue
                if dist > 10.0 or timestamp - start_t > 2.5:
                    # Lost or abandoned: still ahead of us after the window.
                    if along > 0.0:
                        if start_speed >= self.pace_duel_speed_mps:
                            c.pace_duels += 1
                        else:
                            c.take_ons += 1
                    del self._duel_targets[key]
                    continue
                seen.add(key)
                continue
            if dist <= self.duel_radius_m * 1.6 and along >= -0.5:
                self._duel_targets[key] = (along, timestamp, self_speed)
                seen.add(key)
        for key in list(self._duel_targets):
            if key not in seen and key not in {o.player_id for o in opponents}:
                del self._duel_targets[key]

    def _update_runs_in_behind(
        self,
        me: _Point,
        opponents: list[PlayerPosition],
        opp_velocity: dict[str, tuple[float, float]],
        self_speed: float,
    ) -> None:
        if self_speed < self.pace_duel_speed_mps:
            self._duel_targets.clear()
            return
        vx, vy = self._self_velocity
        ux, uy = vx / self_speed, vy / self_speed
        for opp in opponents:
            dx_m, dy_m = self._to_m(opp.x - me.x, opp.y - me.y)
            along = dx_m * ux + dy_m * uy
            dist = math.hypot(dx_m, dy_m)
            key = f"run:{opp.player_id}"
            if key in self._duel_targets:
                if along <= -1.8:
                    self.counters.runs_in_behind += 1
                    self.counters.pace_duels += 1
                    self.counters.pace_duels_won += 1
                    del self._duel_targets[key]
                elif dist > 10.0:
                    del self._duel_targets[key]
            elif dist <= self.duel_radius_m * 1.6 and along >= -0.5:
                self._duel_targets[key] = (along, 0.0, self_speed)

    # ---------------------------------------------------------------- metrics
    def metrics(self) -> dict[str, float]:
        c = self.counters
        minutes = max(c.gameplay_seconds, 1.0) / 60.0
        tracked_minutes = max(c.tracked_seconds, 1.0) / 60.0
        speeds = sorted(c.speed_samples)
        top_speed = speeds[int(len(speeds) * 0.95) - 1] if len(speeds) >= 5 else (speeds[-1] if speeds else 0.0)
        sprint_share = c.sprint_frames / c.moving_frames if c.moving_frames else 0.0
        pass_rate = c.passes_completed / c.passes_attempted if c.passes_attempted else 0.0
        pace_rate = c.pace_duels_won / c.pace_duels if c.pace_duels else 0.0
        take_on_rate = c.take_ons_won / c.take_ons if c.take_ons else 0.0
        retention = c.carries_retained / (c.carries_retained + c.carries_lost) if (c.carries_retained + c.carries_lost) else 0.0
        tracking_quality = c.tracked_frames / c.frames if c.frames else 0.0
        return {
            "gameplay_minutes": round(c.gameplay_seconds / 60.0, 3),
            "tracking_quality": round(tracking_quality, 3),
            "self_visible": round(c.self_frames / c.frames, 3) if c.frames else 0.0,
            # speed
            "top_speed_mps": round(top_speed, 2),
            "sprint_share": round(sprint_share, 3),
            "distance_per_min_m": round(c.distance_m / tracked_minutes, 1),
            "pace_duels": c.pace_duels,
            "pace_duels_won": c.pace_duels_won,
            "pace_duel_rate": round(pace_rate, 3),
            "pace_duels_won_per_min": round(c.pace_duels_won / minutes, 3),
            "runs_in_behind": c.runs_in_behind,
            # pass
            "passes": c.passes_attempted,
            "passes_completed": c.passes_completed,
            "pass_rate": round(pass_rate, 3),
            "pass_kicks": c.pass_kicks,
            "pass_kicks_confirmed": c.pass_kicks_confirmed,
            "pass_rate_hud": round(c.pass_kicks_confirmed / c.pass_kicks, 3) if c.pass_kicks else 0.0,
            "popups": dict(c.popups),
            "forward_pass_share": round(c.forward_passes / c.passes_completed, 3) if c.passes_completed else 0.0,
            "passes_per_min": round(c.passes_attempted / minutes, 3),
            "avg_pass_length_m": round(c.pass_length_m / c.passes_attempted, 1) if c.passes_attempted else 0.0,
            "key_passes": c.key_passes,
            "switches": c.switches,
            # shoot
            "shots": c.shots,
            "shots_on_target": c.shots_on_target,
            "shots_per_min": round(c.shots / minutes, 3),
            "on_target_rate": round(c.shots_on_target / c.shots, 3) if c.shots else 0.0,
            "avg_shot_quality": round(c.shot_quality_sum / c.shots, 3) if c.shots else 0.0,
            "goals": c.goals,
            # dribble
            "carries": c.carries,
            "carry_distance_per_min_m": round(c.carry_distance_m / tracked_minutes, 1),
            "take_ons": c.take_ons,
            "take_ons_won": c.take_ons_won,
            "take_on_rate": round(take_on_rate, 3),
            "retention": round(retention, 3),
            "losses": c.losses,
            # offense
            "involvement": round(c.involvement_frames / c.team_possession_frames, 3) if c.team_possession_frames else 0.0,
            "attacking_third_share": round(c.attacking_third_frames / c.team_possession_frames, 3) if c.team_possession_frames else 0.0,
            "forward_runs_per_min": round(c.forward_runs / minutes, 3),
            "box_touches": c.box_touches,
            # defense
            "recoveries": c.recoveries,
            "tackles": c.tackles,
            "interceptions": c.interceptions,
            "recoveries_per_min": round(c.recoveries / minutes, 3),
            "goal_side_share": round(c.goal_side_frames / c.opp_possession_frames, 3) if c.opp_possession_frames else 0.0,
            "avg_closing_speed_mps": round(sum(c.pressing_samples) / len(c.pressing_samples), 2) if c.pressing_samples else 0.0,
            "avg_carrier_distance_m": round(sum(c.carrier_distance_samples) / len(c.carrier_distance_samples), 1) if c.carrier_distance_samples else 0.0,
            "avg_nearest_mate_m": round(sum(c.compactness_samples) / len(c.compactness_samples), 1) if c.compactness_samples else 0.0,
        }


# --------------------------------------------------------------------- helpers
def _same_spot(a: PlayerPosition, b: _Point) -> bool:
    return abs(a.x - b.x) < 0.6 and abs(a.y - b.y) < 0.6


def _nearest(target: PlayerPosition, pool: list[PlayerPosition], dist_fn, max_m: float) -> PlayerPosition | None:
    best = None
    best_d = max_m
    for p in pool:
        d = dist_fn(target.x, target.y, p.x, p.y)
        if d <= best_d:
            best, best_d = p, d
    return best


def _between(value: float, a: float, b: float, margin: float = 0.0) -> bool:
    low, high = (a, b) if a <= b else (b, a)
    return low - margin <= value <= high + margin
