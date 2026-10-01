from __future__ import annotations

import math
import threading

from PIL import Image

from goalx.models import KitProfile
from goalx.vision import Blob, color_label, kit_distance, kit_samples, minimap_kit_samples

LOCK_SAMPLES = 15
FAST_ALPHA = 0.3
LOCKED_ALPHA = 0.05
MINIMAP_WEIGHT = 0.6


class _Cluster:
    def __init__(self, hue: float | None, sat: float, br: float) -> None:
        self.hx = 0.0 if hue is None else math.cos(math.radians(hue)) * sat
        self.hy = 0.0 if hue is None else math.sin(math.radians(hue)) * sat
        self.sat = sat
        self.br = br

    @property
    def hue(self) -> float | None:
        if self.sat < 0.05:
            return None if math.hypot(self.hx, self.hy) < 0.02 else (math.degrees(math.atan2(self.hy, self.hx)) + 360.0) % 360.0
        return (math.degrees(math.atan2(self.hy, self.hx)) + 360.0) % 360.0

    def distance(self, other: "_Cluster") -> float:
        return math.hypot(self.hx - other.hx, self.hy - other.hy) + abs(self.br - other.br) / 255.0 * 0.6

    def blend(self, other: "_Cluster", alpha: float) -> None:
        self.hx = self.hx * (1.0 - alpha) + other.hx * alpha
        self.hy = self.hy * (1.0 - alpha) + other.hy * alpha
        self.sat = self.sat * (1.0 - alpha) + other.sat * alpha
        self.br = self.br * (1.0 - alpha) + other.br * alpha

    def label(self) -> str:
        return color_label(self.hue if self.sat >= 0.28 else 0.0, self.sat, self.br)

    def copy(self) -> "_Cluster":
        clone = _Cluster(None, self.sat, self.br)
        clone.hx, clone.hy = self.hx, self.hy
        return clone


def _blob_cluster(blob: Blob) -> _Cluster:
    return _Cluster(blob["hue"], blob["saturation"], blob["brightness"])


def _weighted_mean(items: list[tuple[_Cluster, float]]) -> _Cluster | None:
    total = sum(weight for _c, weight in items)
    if total <= 0:
        return None
    mean = _Cluster(None, 0.0, 0.0)
    mean.hx = sum(c.hx * w for c, w in items) / total
    mean.hy = sum(c.hy * w for c, w in items) / total
    mean.sat = sum(c.sat * w for c, w in items) / total
    mean.br = sum(c.br * w for c, w in items) / total
    return mean


def two_cluster(samples: list[tuple[_Cluster, float]], seed: _Cluster | None) -> tuple[_Cluster | None, _Cluster | None]:
    """Weighted 2-means on hue-circle*saturation + brightness. `seed` anchors the first cluster."""
    if not samples:
        return None, None
    if seed is not None:
        c1 = seed.copy()
    else:
        mean = _weighted_mean(samples)
        c1 = max(samples, key=lambda item: item[0].distance(mean))[0].copy()  # type: ignore[arg-type]
    c2 = max(samples, key=lambda item: item[0].distance(c1))[0].copy()
    if c2.distance(c1) < 0.18:
        return _weighted_mean(samples), None
    for _ in range(6):
        group1 = [item for item in samples if item[0].distance(c1) <= item[0].distance(c2)]
        group2 = [item for item in samples if item[0].distance(c1) > item[0].distance(c2)]
        new1 = _weighted_mean(group1) if group1 else None
        new2 = _weighted_mean(group2) if group2 else None
        if seed is not None and new1 is not None:
            # Keep the anchor dominant so the local player's kit defines "team".
            anchored = seed.copy()
            anchored.blend(new1, 0.5)
            new1 = anchored
        if new1 is None or new2 is None:
            break
        if new1.distance(c1) < 0.005 and new2.distance(c2) < 0.005:
            c1, c2 = new1, new2
            break
        c1, c2 = new1, new2
    return c1, c2


class KitCalibrator:
    """Learns 'my team' vs 'enemy' kit colours per match from gameplay frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._team: _Cluster | None = None
        self._opp: _Cluster | None = None
        self._samples = 0
        self._locked = False
        self._override = False
        self._stability: list[float] = []

    def reset(self) -> None:
        with self._lock:
            self._team = None
            self._opp = None
            self._samples = 0
            self._locked = False
            self._override = False
            self._stability = []

    def override(
        self,
        team_hue: float | None = None,
        opp_hue: float | None = None,
        team_label: str | None = None,
        opp_label: str | None = None,
    ) -> KitProfile | None:
        with self._lock:
            team = _cluster_from_override(team_hue, team_label) or self._team
            opp = _cluster_from_override(opp_hue, opp_label) or self._opp
            if team is None and opp is None:
                return self._profile_locked()
            self._team = team
            self._opp = opp
            self._samples = max(self._samples, LOCK_SAMPLES)
            self._locked = True
            self._override = True
            return self._profile_locked()

    def observe(self, image: Image.Image, frame_kind: str = "gameplay") -> KitProfile | None:
        if frame_kind != "gameplay":
            return self.profile()
        local, others = kit_samples(image)
        map_markers = minimap_kit_samples(image)
        samples: list[tuple[_Cluster, float]] = []
        for blob in others:
            samples.append((_blob_cluster(blob), max(1.0, math.sqrt(blob["size"]))))
        for blob in map_markers:
            samples.append((_blob_cluster(blob), MINIMAP_WEIGHT * max(1.0, math.sqrt(blob["size"]))))
        seed = _blob_cluster(local) if local is not None else None
        with self._lock:
            if self._override:
                return self._profile_locked()
            if seed is None and not samples:
                return self._profile_locked()
            self._ingest(seed, samples)
            return self._profile_locked()

    def profile(self) -> KitProfile | None:
        with self._lock:
            return self._profile_locked()

    # -- internals -----------------------------------------------------------------

    def _ingest(self, seed: _Cluster | None, samples: list[tuple[_Cluster, float]]) -> None:
        anchor = seed if seed is not None else (self._team.copy() if self._team is not None else None)
        c1, c2 = two_cluster(samples, anchor)
        if anchor is not None:
            # The local player's kit (or the running team estimate) defines "team"; anything far from it is "enemy".
            team_obs = anchor.copy()
            candidates = [cluster for cluster in (c1, c2) if cluster is not None]
            near = [cluster for cluster in candidates if cluster.distance(anchor) < 0.35]
            far = [cluster for cluster in candidates if cluster.distance(anchor) >= 0.35]
            for cluster in near:
                team_obs.blend(cluster, 0.4)
            opp_obs = max(far, key=lambda cluster: cluster.distance(anchor)) if far else None
        else:
            team_obs, opp_obs = _heuristic_sides(c1, c2)

        alpha = LOCKED_ALPHA if self._locked else FAST_ALPHA
        if team_obs is not None:
            if self._team is None:
                self._team = team_obs.copy()
            else:
                self._stability.append(self._team.distance(team_obs))
                self._stability = self._stability[-6:]
                self._team.blend(team_obs, alpha)
        if opp_obs is not None:
            if self._opp is None:
                self._opp = opp_obs.copy()
            else:
                self._opp.blend(opp_obs, alpha)
        self._samples += 1
        if not self._locked and self._samples >= LOCK_SAMPLES and self._team is not None and self._opp is not None:
            if not self._stability or max(self._stability[-4:]) < 0.25:
                self._locked = True

    def _profile_locked(self) -> KitProfile | None:
        if self._team is None and self._opp is None:
            return None
        separation = 1.0
        if self._team is not None and self._opp is not None:
            separation = min(1.0, self._team.distance(self._opp) / 0.5)
        completeness = 1.0 if (self._team is not None and self._opp is not None) else 0.5
        confidence = min(1.0, self._samples / LOCK_SAMPLES) * (0.5 + 0.5 * separation) * completeness
        return KitProfile(
            team_hue=None if self._team is None else _rounded_hue(self._team),
            team_saturation=0.0 if self._team is None else round(max(0.0, min(1.0, self._team.sat)), 3),
            team_brightness=0.0 if self._team is None else round(max(0.0, min(255.0, self._team.br)), 1),
            opp_hue=None if self._opp is None else _rounded_hue(self._opp),
            opp_saturation=0.0 if self._opp is None else round(max(0.0, min(1.0, self._opp.sat)), 3),
            opp_brightness=0.0 if self._opp is None else round(max(0.0, min(255.0, self._opp.br)), 1),
            team_label="unknown" if self._team is None else self._team.label(),
            opp_label="unknown" if self._opp is None else self._opp.label(),
            samples=self._samples,
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            locked=self._locked,
        )


def _rounded_hue(cluster: _Cluster) -> float | None:
    hue = cluster.hue
    if hue is None:
        return 0.0 if cluster.sat < 0.28 else None
    return round(hue, 1)


def _heuristic_sides(c1: _Cluster | None, c2: _Cluster | None) -> tuple[_Cluster | None, _Cluster | None]:
    """Without the local player: saturated kit is 'team', the pale kit is 'enemy' (Rematch default pairing)."""
    if c1 is None:
        return None, None
    if c2 is None:
        return c1, None
    if c1.sat >= c2.sat:
        return c1, c2
    return c2, c1


def _cluster_from_override(hue: float | None, label: str | None) -> _Cluster | None:
    if hue is not None:
        return _Cluster(hue, 0.75, 120.0)
    if label is None:
        return None
    presets: dict[str, tuple[float | None, float, float]] = {
        "white": (0.0, 0.05, 230.0),
        "grey": (0.0, 0.08, 140.0),
        "black": (0.0, 0.1, 30.0),
        "red": (2.0, 0.85, 110.0),
        "orange": (25.0, 0.85, 130.0),
        "yellow": (55.0, 0.85, 170.0),
        "green": (120.0, 0.7, 120.0),
        "teal": (180.0, 0.6, 120.0),
        "blue": (225.0, 0.75, 110.0),
        "purple": (280.0, 0.6, 110.0),
        "pink": (325.0, 0.6, 160.0),
    }
    preset = presets.get(label.strip().lower())
    if preset is None:
        return None
    return _Cluster(*preset)


def profile_distance(profile: KitProfile, blob: Blob) -> tuple[float, float]:
    d_team = kit_distance(blob["hue"], blob["saturation"], blob["brightness"], profile.team_hue, profile.team_saturation, profile.team_brightness)
    d_opp = kit_distance(blob["hue"], blob["saturation"], blob["brightness"], profile.opp_hue, profile.opp_saturation, profile.opp_brightness)
    return d_team, d_opp
