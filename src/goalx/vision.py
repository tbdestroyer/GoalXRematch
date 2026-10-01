from __future__ import annotations

import math
from typing import Callable

from PIL import Image

from goalx.models import KitProfile, LivePitchScene, Phase, PlayerPosition

Blob = dict[str, float]
KeepFn = Callable[[int, int, int, int, int], bool]

# Rematch anchors its circular minimap bottom-right; measured on 3440x1440 and 860x360 captures.
MINIMAP_CENTER_X = 0.893
MINIMAP_CENTER_Y = 0.74
MINIMAP_RADIUS_H = 0.2
MINIMAP_SAMPLE = 200
MINIMAP_MASK = 0.46

# Camera projection anchors (percent of frame).
CAMERA_LOCAL_Y = 62.0
CAMERA_HORIZON_Y = 27.0
CAMERA_FORWARD_GAIN = 5.0
CAMERA_LATERAL_GAIN = 11.0
FUSION_MATCH_TOLERANCE = 14.0
DUPLICATE_DISTANCE = 1.0


def estimate_pitch_scene(image: Image.Image, kit_profile: KitProfile | None = None) -> LivePitchScene:
    scan = _minimap_markers(image)
    frame_kind = classify_frame(image, scan)
    camera = _parse_camera_view(image, kit_profile)
    minimap = _parse_minimap(image, kit_profile, scan)

    map_ok = minimap is not None and minimap.map_detected and minimap.detection_count >= 1
    cam_ok = camera is not None and camera.detection_count >= 1

    if map_ok and cam_ok and minimap.self_x is not None:
        tactical, matches = _fuse_scenes(camera, minimap)
        source = "fused"
    elif map_ok and cam_ok:
        # No self ring to anchor the projection: keep whichever source saw more players.
        if minimap.detection_count >= camera.detection_count:
            tactical, matches, source = minimap, 0, "minimap"
        else:
            tactical, matches, source = camera, 0, "camera"
    elif map_ok:
        tactical, matches, source = minimap, 0, "minimap"
    elif camera is not None and len(camera.teammates) >= 2:
        tactical, matches, source = camera, 0, "camera"
    elif cam_ok:
        tactical, matches, source = camera, 0, "camera"
    else:
        tactical, matches, source = _empty_scene(), 0, "none"

    view = camera if cam_ok else None
    confidence = tactical.confidence
    if view is not None and source != "fused":
        confidence = max(confidence, view.confidence)
    return LivePitchScene(
        phase=tactical.phase,
        ball_x=tactical.ball_x,
        ball_y=tactical.ball_y,
        teammates=tactical.teammates[:7],
        opponents=tactical.opponents[:7],
        confidence=round(min(1.0, confidence), 3),
        detection_count=len(tactical.teammates[:7]) + len(tactical.opponents[:7]),
        view_ball_x=None if view is None else view.ball_x,
        view_ball_y=None if view is None else view.ball_y,
        view_teammates=[] if view is None else view.teammates[:7],
        view_opponents=[] if view is None else view.opponents[:7],
        map_detected=bool(minimap is not None and minimap.map_detected),
        map_ball_x=None if minimap is None else minimap.map_ball_x,
        map_ball_y=None if minimap is None else minimap.map_ball_y,
        map_teammates=[] if minimap is None else minimap.map_teammates[:7],
        map_opponents=[] if minimap is None else minimap.map_opponents[:7],
        map_region={} if minimap is None else minimap.map_region,
        self_x=None if minimap is None else minimap.self_x,
        self_y=None if minimap is None else minimap.self_y,
        facing_deg=tactical.facing_deg if tactical.facing_deg is not None else (None if minimap is None else minimap.facing_deg),
        source=source,
        fusion_matches=matches,
        gameplay_detected=frame_kind == "gameplay",
        frame_kind=frame_kind,
    )


# --------------------------------------------------------------------------- frame classification


def frame_cues(image: Image.Image, scan: "MinimapScan | None" = None) -> dict[str, float | bool]:
    sample = _analysis_image(image, max_width=240)
    width, height = sample.size
    pixels = list(sample.getdata())
    total = max(1, width * height)

    grass = 0
    dark = 0
    sky = 0
    quantized: dict[tuple[int, int, int], int] = {}
    for r, g, b in pixels:
        if _is_pitch(r, g, b) or _is_rematch_court(r, g, b):
            grass += 1
        if (r + g + b) / 3.0 < 20:
            dark += 1
        if _is_sky(r, g, b):
            sky += 1
        key = (r // 32, g // 32, b // 32)
        quantized[key] = quantized.get(key, 0) + 1
    dominant = max(quantized.values()) / total if quantized else 0.0

    band = max(2, int(height * 0.08))
    top_rows = pixels[: band * width]
    bottom_rows = pixels[(height - band) * width :]
    top_dark = sum(1 for r, g, b in top_rows if (r + g + b) / 3.0 < 18) / max(1, len(top_rows))
    bottom_dark = sum(1 for r, g, b in bottom_rows if (r + g + b) / 3.0 < 18) / max(1, len(bottom_rows))
    letterbox = top_dark > 0.9 and bottom_dark > 0.9

    # Scoreboard / timer block top-left: bright digits on a dark box.
    tl_w = int(width * 0.24)
    tl_h = max(2, int(height * 0.12))
    tl_bright = tl_dark = tl_total = 0
    for y in range(tl_h):
        for x in range(tl_w):
            r, g, b = pixels[y * width + x]
            br = (r + g + b) / 3.0
            tl_total += 1
            if br >= 200 and _saturation(r, g, b) < 0.25:
                tl_bright += 1
            if br < 60:
                tl_dark += 1
    hud_top_left = tl_total > 0 and 0.01 <= tl_bright / tl_total <= 0.5 and tl_dark / tl_total >= 0.03

    # Stamina bar bottom-center: saturated green/teal segment.
    bar = 0
    for y in range(int(height * 0.84), int(height * 0.97)):
        for x in range(int(width * 0.36), int(width * 0.64)):
            r, g, b = pixels[y * width + x]
            hue = _hue(r, g, b)
            if 90 <= hue <= 185 and _saturation(r, g, b) >= 0.45 and (r + g + b) / 3.0 >= 70 and g >= r + 30:
                bar += 1
    stamina_bar = bar >= 8

    minimap_present = _minimap_present(image, scan)

    return {
        "grass_fraction": round(grass / total, 4),
        "dark_fraction": round(dark / total, 4),
        "sky_fraction": round(sky / total, 4),
        "dominant_fraction": round(dominant, 4),
        "letterbox": letterbox,
        "hud_top_left": hud_top_left,
        "stamina_bar": stamina_bar,
        "minimap_present": minimap_present,
    }


def classify_frame(image: Image.Image, scan: "MinimapScan | None" = None) -> str:
    cues = frame_cues(image, scan)
    grass = float(cues["grass_fraction"])
    dark = float(cues["dark_fraction"])
    dominant = float(cues["dominant_fraction"])
    hud_votes = int(bool(cues["hud_top_left"])) + int(bool(cues["stamina_bar"])) + int(bool(cues["minimap_present"]))

    if dark >= 0.8:
        return "menu"
    if grass < 0.06 and (dominant >= 0.55 or dark >= 0.45):
        return "menu"
    if cues["letterbox"] and not cues["minimap_present"]:
        return "cutscene"
    if hud_votes >= 2:
        return "gameplay"
    if hud_votes == 1 and grass >= 0.12 and (cues["minimap_present"] or cues["hud_top_left"]):
        return "gameplay"
    if hud_votes == 0 and grass >= 0.12:
        return "cutscene"
    if grass < 0.06:
        return "menu"
    return "unknown"


def _minimap_present(image: Image.Image, scan: "MinimapScan | None" = None) -> bool:
    if scan is None:
        scan = _minimap_markers(image)
    return bool(scan is not None and scan.present)


# --------------------------------------------------------------------------- visibility


def rematch_visibility_score(image: Image.Image) -> float:
    sample = _analysis_image(image, max_width=240)
    width, height = sample.size
    pixels = list(sample.getdata())
    court = 0
    sky = 0
    white_center = 0
    scoreboard = 0
    center_box = 0
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[y * width + x]
            if _is_rematch_court(r, g, b) or _is_pitch(r, g, b):
                court += 1
            if _is_sky(r, g, b):
                sky += 1
            if y < int(height * 0.18) and x < int(width * 0.42):
                brightness = (r + g + b) / 3.0
                if brightness > 170 and _saturation(r, g, b) < 0.18:
                    scoreboard += 1
                if brightness < 55 and b >= r:
                    scoreboard += 1
            if int(width * 0.3) <= x <= int(width * 0.7) and int(height * 0.3) <= y <= int(height * 0.7):
                center_box += 1
                if min(r, g, b) > 210:
                    white_center += 1
    total = max(1, width * height)
    court_frac = court / total
    if white_center > center_box * 0.45:
        return 0.05
    score = 0.0
    score += min(0.55, court_frac * 1.8)
    score += min(0.25, sky / total * 0.8)
    score += min(0.25, scoreboard / max(1, int(width * 0.42) * int(height * 0.18)) * 0.9)
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------- camera view


def _camera_keep_factory(width: int, height: int) -> KeepFn:
    top_skip = int(height * 0.24)
    bottom_skip = int(height * 0.88)
    minimap_left = int(width * (MINIMAP_CENTER_X - 0.12))
    minimap_top = int(height * (MINIMAP_CENTER_Y - MINIMAP_RADIUS_H - 0.03))

    def keep(x: int, y: int, r: int, g: int, b: int) -> bool:
        if y < top_skip or y >= bottom_skip:
            return False
        if x >= minimap_left and y >= minimap_top:
            return False
        if x < int(width * 0.08):
            return False
        brightness = (r + g + b) / 3.0
        sat = _saturation(r, g, b)
        hue = _hue(r, g, b)
        if brightness < 40:
            return False
        if _is_pitch(r, g, b) or _is_rematch_court(r, g, b):
            return False
        # Pastel sky / ice walls are background, but pure whites in the play area are white kits.
        if _is_sky(r, g, b) and not (sat <= 0.16 and brightness >= 150):
            return False
        if y < int(height * 0.34) and 190 <= hue <= 255:
            return False
        if 155 <= hue <= 200 and brightness >= 110 and g >= r + 12:
            return False
        if sat < 0.22 and brightness < 150:
            return False
        return sat >= 0.28 or brightness >= 150

    return keep


def _parse_camera_view(image: Image.Image, kit_profile: KitProfile | None = None) -> LivePitchScene | None:
    sample = _analysis_image(image, max_width=360)
    width, height = sample.size
    pixels = list(sample.getdata())
    keep = _camera_keep_factory(width, height)

    all_blobs = _collect_blobs(width, height, pixels, keep, min_size=10, max_size=50_000)
    local, local_members = _pick_local_player(all_blobs, width, height)
    blobs = [blob for blob in all_blobs if blob not in local_members and _is_player_candidate(blob)]
    if not blobs:
        if local is None:
            return None
        self_pos = PlayerPosition(player_id="self", role="mf", x=round(max(0.0, min(100.0, local["x"])), 1), y=round(max(0.0, min(100.0, local["y"])), 1))
        return LivePitchScene(
            phase="transition",
            ball_x=50.0,
            ball_y=50.0,
            teammates=[self_pos],
            opponents=[],
            confidence=0.22,
            detection_count=1,
            self_x=self_pos.x,
            self_y=self_pos.y,
        )

    ball = _pick_ball(blobs)
    player_blobs = [blob for blob in blobs if blob is not ball]
    team, opp = _split_rematch_kits(player_blobs, kit_profile=kit_profile)
    if len(team) + len(opp) < 1:
        return None
    if kit_profile is None and local is None and len(team) < 1 and len(opp) >= 2:
        team, opp = opp, []

    teammates = _to_players(team, "team")
    opponents = _to_players(opp, "opp")
    if local is not None:
        teammates.insert(
            0,
            PlayerPosition(player_id="self", role="mf", x=round(max(0.0, min(100.0, local["x"])), 1), y=round(max(0.0, min(100.0, local["y"])), 1)),
        )
    detections = len(teammates) + len(opponents)
    scene = LivePitchScene(
        phase=_phase_from_ball(ball["x"]),
        ball_x=round(ball["x"], 1),
        ball_y=round(ball["y"], 1),
        teammates=teammates[:7],
        opponents=opponents[:7],
        confidence=round(max(0.22, min(0.9, 0.2 + detections * 0.08)), 3),
        detection_count=detections,
    )
    if local is not None:
        scene.self_x = round(local["x"], 1)
        scene.self_y = round(local["y"], 1)
    return scene


def _is_player_candidate(blob: Blob) -> bool:
    if blob["size"] > 1600 or blob["size"] < 10:
        return False
    if blob["saturation"] < 0.16 and blob["size"] > 70:
        return False
    if blob["brightness"] > 210 and blob["saturation"] < 0.10 and blob["size"] > 35:
        return False
    if blob["w"] > 3.5 * max(1.0, blob["h"]):
        return False  # ad boards, pitch lines
    if blob["fill"] < 0.25:
        return False  # diagonal pitch lines
    if blob["saturation"] < 0.12 and blob["brightness"] > 170 and blob["h"] <= 5 and blob["w"] >= 2 * blob["h"]:
        return False  # name tags / number badges
    return True


def _pick_local_player(blobs: list[Blob], width: int, height: int) -> tuple[Blob | None, list[Blob]]:
    """The controlled player is drawn large near centre-bottom; its kit fragments are grouped by proximity."""
    zone = [
        blob
        for blob in blobs
        if 32.0 <= blob["x"] <= 68.0
        and 36.0 <= blob["y"] <= 86.0
        and 8 <= blob["size"] <= 2500
        and blob["w"] <= width * 0.16
        and blob["h"] <= height * 0.5
        and not (blob["brightness"] > 235 and blob["saturation"] < 0.08)  # flat white UI panels
    ]
    if not zone:
        return None, []

    def px(blob: Blob) -> tuple[float, float]:
        return blob["x"] / 100.0 * (width - 1), blob["y"] / 100.0 * (height - 1)

    groups: list[list[Blob]] = []
    for blob in sorted(zone, key=lambda item: -item["size"]):
        bx, by = px(blob)
        for group in groups:
            if any(abs(bx - px(other)[0]) <= 14 and abs(by - px(other)[1]) <= 24 for other in group):
                group.append(blob)
                break
        else:
            groups.append([blob])
    best = max(groups, key=lambda group: sum(item["size"] for item in group))
    total = sum(item["size"] for item in best)
    ys = [px(item)[1] for item in best]
    extent = max(ys) - min(ys) + max(item["h"] for item in best)
    if total < 60 or extent < 10:
        return None, []
    cos_sum = sin_sum = sat = br = 0.0
    xs = 0.0
    yy = 0.0
    for item in best:
        weight = item["size"]
        cos_sum += math.cos(math.radians(item["hue"])) * weight
        sin_sum += math.sin(math.radians(item["hue"])) * weight
        sat += item["saturation"] * weight
        br += item["brightness"] * weight
        xs += item["x"] * weight
        yy += item["y"] * weight
    merged: Blob = {
        "x": xs / total,
        "y": yy / total,
        "size": float(total),
        "hue": (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0,
        "saturation": sat / total,
        "brightness": br / total,
        "blue_bias": sum(item["blue_bias"] * item["size"] for item in best) / total,
        "w": max(item["w"] for item in best),
        "h": float(extent),
        "fill": 1.0,
    }
    return merged, best


def kit_samples(image: Image.Image) -> tuple[Blob | None, list[Blob]]:
    """Return (local player blob, other candidate player blobs) in camera space for kit clustering."""
    sample = _analysis_image(image, max_width=360)
    width, height = sample.size
    pixels = list(sample.getdata())
    keep = _camera_keep_factory(width, height)
    all_blobs = _collect_blobs(width, height, pixels, keep, min_size=10, max_size=50_000)
    local, members = _pick_local_player(all_blobs, width, height)
    others = [
        blob
        for blob in all_blobs
        if blob not in members
        and _is_player_candidate(blob)
        and not _looks_like_ball(blob)
        and not (42 <= blob["hue"] <= 70 and blob["saturation"] >= 0.5 and blob["brightness"] >= 150)
    ]
    return local, others


# --------------------------------------------------------------------------- minimap


def _minimap_box(width: int, height: int) -> tuple[int, int, int, int]:
    cx = width * MINIMAP_CENTER_X
    cy = height * MINIMAP_CENTER_Y
    half = height * MINIMAP_RADIUS_H
    left = int(max(0, cx - half))
    top = int(max(0, cy - half))
    right = int(min(width, cx + half))
    bottom = int(min(height, cy + half))
    return left, top, right, bottom


def _minimap_markers(image: Image.Image) -> "MinimapScan | None":
    """Return (self ring blob, candidate marker blobs, sample pixels) from the circular minimap crop."""
    width, height = image.size
    box = _minimap_box(width, height)
    left, top, right, bottom = box
    if right - left < 40 or bottom - top < 40:
        return None
    size = MINIMAP_SAMPLE
    sample = image.convert("RGB").crop(box).resize((size, size))
    pixels = list(sample.getdata())
    c = (size - 1) / 2.0
    radius = size * MINIMAP_MASK

    def inside(x: int, y: int) -> bool:
        return (x - c) ** 2 + (y - c) ** 2 <= radius * radius

    def is_yellow(r: int, g: int, b: int) -> bool:
        hue = _hue(r, g, b)
        return 40 <= hue <= 70 and _saturation(r, g, b) >= 0.5 and (r + g + b) / 3.0 >= 100

    bg_hue, bg_fraction, light_fraction = _minimap_background(pixels, size, inside)

    def is_background(r: int, g: int, b: int) -> bool:
        if _is_pitch(r, g, b):
            return True
        sat = _saturation(r, g, b)
        if sat < 0.15:
            return False
        return _hue_diff(_hue(r, g, b), bg_hue) <= 22.0

    def is_marker(r: int, g: int, b: int) -> bool:
        if is_yellow(r, g, b):
            return False
        sat = _saturation(r, g, b)
        br = (r + g + b) / 3.0
        saturated = sat >= 0.45 and br >= 40
        light = sat <= 0.30 and br >= 95
        return saturated or light

    def keep_class(predicate: Callable[[int, int, int], bool]) -> KeepFn:
        def keep(x: int, y: int, r: int, g: int, b: int) -> bool:
            if not inside(x, y):
                return False
            if is_background(r, g, b):
                return False
            return predicate(r, g, b)

        return keep

    yellows = _collect_blobs(size, size, pixels, keep_class(is_yellow), min_size=6, max_size=400)
    raw_markers = _collect_blobs(size, size, pixels, keep_class(is_marker), min_size=4, max_size=400)
    self_blob = max(yellows, key=lambda blob: blob["size"]) if yellows else None
    self_px = None if self_blob is None else (self_blob["x"] / 100.0 * (size - 1), self_blob["y"] / 100.0 * (size - 1))

    def near_self(blob: Blob, limit: float) -> bool:
        if self_px is None:
            return False
        bx = blob["x"] / 100.0 * (size - 1)
        by = blob["y"] / 100.0 * (size - 1)
        return math.hypot(bx - self_px[0], by - self_px[1]) <= limit

    markers: list[Blob] = []
    for blob in raw_markers:
        longest = max(blob["w"], blob["h"])
        shortest = min(blob["w"], blob["h"])
        if longest > 24 or shortest < 3 or longest > 2.6 * shortest:
            continue  # pitch lines and line fragments
        if blob["saturation"] <= 0.30:
            if blob["fill"] < 0.4 or blob["size"] < 12:
                continue
        elif blob["size"] < 6:
            continue
        if blob["saturation"] > 0.30 and _hue_diff(blob["hue"], bg_hue) <= 35.0:
            continue  # grass / court shaded patches
        if near_self(blob, 7.0):
            continue
        markers.append(blob)
    return MinimapScan(
        self_blob=self_blob,
        markers=_dedupe_blobs(markers, 4.0),
        pixels=pixels,
        background_fraction=bg_fraction,
        light_fraction=light_fraction,
    )


class MinimapScan:
    def __init__(
        self,
        self_blob: Blob | None,
        markers: list[Blob],
        pixels: list[tuple[int, int, int]],
        background_fraction: float,
        light_fraction: float,
    ) -> None:
        self.self_blob = self_blob
        self.markers = markers
        self.pixels = pixels
        self.background_fraction = background_fraction
        self.light_fraction = light_fraction

    @property
    def present(self) -> bool:
        if self.self_blob is not None and self.background_fraction >= 0.2:
            return True
        return self.background_fraction >= 0.4 and self.light_fraction >= 0.01 and len(self.markers) >= 1


def _minimap_background(
    pixels: list[tuple[int, int, int]],
    size: int,
    inside: Callable[[int, int], bool],
) -> tuple[float, float, float]:
    """Dominant saturated hue inside the circle (grass or court), its fraction, and light-line fraction."""
    cos_sum = sin_sum = 0.0
    total = light = 0
    saturated = 0
    for y in range(size):
        for x in range(size):
            if not inside(x, y):
                continue
            r, g, b = pixels[y * size + x]
            total += 1
            sat = _saturation(r, g, b)
            br = (r + g + b) / 3.0
            if br >= 150 and sat <= 0.3:
                light += 1
            if sat >= 0.15 and br >= 25:
                hue = math.radians(_hue(r, g, b))
                cos_sum += math.cos(hue)
                sin_sum += math.sin(hue)
                saturated += 1
    if total == 0 or saturated == 0:
        return 0.0, 0.0, 0.0
    bg_hue = (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0
    near = 0
    for y in range(size):
        for x in range(size):
            if not inside(x, y):
                continue
            r, g, b = pixels[y * size + x]
            if _saturation(r, g, b) >= 0.15 and _hue_diff(_hue(r, g, b), bg_hue) <= 22.0:
                near += 1
    return bg_hue, near / total, light / total


def _hue_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def minimap_kit_samples(image: Image.Image) -> list[Blob]:
    scan = _minimap_markers(image)
    if scan is None:
        return []
    return scan.markers


def _split_minimap_markers(markers: list[Blob], kit_profile: KitProfile | None) -> tuple[list[Blob], list[Blob]]:
    """Minimap markers are drawn in flat kit colours: one saturated group and/or one pale group.

    Without a kit profile Rematch's usual pairing applies (saturated = you, pale = enemy). With a profile the
    *groups* are assigned to sides by their mean colour, which is more stable than per-marker matching.
    """
    saturated = [blob for blob in markers if blob["saturation"] >= 0.35]
    pale = [blob for blob in markers if blob["saturation"] < 0.35]
    usable = kit_profile is not None and kit_profile.samples > 0 and (kit_profile.team_hue is not None or kit_profile.opp_hue is not None)
    if not usable or kit_profile is None:
        return saturated, pale

    def group_distance(group: list[Blob], hue: float | None, sat: float, br: float, missing: float) -> float:
        if not group:
            return 0.0
        if hue is None:
            return missing
        return sum(kit_distance(blob["hue"], blob["saturation"], blob["brightness"], hue, sat, br) for blob in group) / len(group)

    team_kit = (kit_profile.team_hue, kit_profile.team_saturation, kit_profile.team_brightness)
    opp_kit = (kit_profile.opp_hue, kit_profile.opp_saturation, kit_profile.opp_brightness)
    straight = group_distance(saturated, *team_kit, 1.0) + group_distance(pale, *opp_kit, 1.0)
    swapped = group_distance(saturated, *opp_kit, 1.0) + group_distance(pale, *team_kit, 1.0)
    if swapped < straight:
        return pale, saturated
    return saturated, pale


def _parse_minimap(
    image: Image.Image,
    kit_profile: KitProfile | None = None,
    scan: MinimapScan | None = None,
) -> LivePitchScene | None:
    width, height = image.size
    if scan is None:
        scan = _minimap_markers(image)
    if scan is None or not scan.present:
        return None
    self_blob, markers, pixels = scan.self_blob, scan.markers, scan.pixels
    box = _minimap_box(width, height)
    left, top, right, bottom = box
    size = MINIMAP_SAMPLE
    self_xy = None if self_blob is None else (self_blob["x"], self_blob["y"])
    self_px = None if self_blob is None else (self_blob["x"] / 100.0 * (size - 1), self_blob["y"] / 100.0 * (size - 1))

    team_blobs, opp_blobs = _split_minimap_markers(markers, kit_profile)
    ball_blob = None

    team_count = len(team_blobs) + (1 if self_blob is not None else 0)
    if team_count + len(opp_blobs) < 1:
        return None

    facing = _estimate_facing(pixels, size, self_px) if self_px is not None else None

    def to_pitch(blob_x: float, blob_y: float) -> tuple[float, float]:
        return _map_to_pitch(blob_x, blob_y)

    teammates: list[PlayerPosition] = []
    if self_blob is not None:
        px, py = to_pitch(self_blob["x"], self_blob["y"])
        teammates.append(PlayerPosition(player_id="map_self", role="mf", x=px, y=py))
    for index, blob in enumerate(sorted(team_blobs, key=lambda item: item["x"])):
        px, py = to_pitch(blob["x"], blob["y"])
        teammates.append(PlayerPosition(player_id=f"map_team_{index + 1}", role="mf", x=px, y=py))
    opponents: list[PlayerPosition] = []
    for index, blob in enumerate(sorted(opp_blobs, key=lambda item: item["x"])):
        px, py = to_pitch(blob["x"], blob["y"])
        opponents.append(PlayerPosition(player_id=f"map_opp_{index + 1}", role="mf", x=px, y=py))
    teammates = _dedupe_players(teammates)[:7]
    opponents = _dedupe_players(opponents)[:7]

    if ball_blob is not None:
        ball_src = (ball_blob["x"], ball_blob["y"])
    elif self_xy is not None:
        ball_src = self_xy
    elif teammates or opponents:
        first = (teammates + opponents)[0]
        ball_src = (first.y, 100.0 - first.x)
    else:
        ball_src = (50.0, 50.0)
    ball_x, ball_y = to_pitch(*ball_src)

    frame_team = [
        _blob_to_frame_player(blob, f"map_team_{index + 1}", box, width, height)
        for index, blob in enumerate(([self_blob] if self_blob is not None else []) + team_blobs)
    ]
    frame_opp = [_blob_to_frame_player(blob, f"map_opp_{index + 1}", box, width, height) for index, blob in enumerate(opp_blobs)]
    map_ball = _blob_to_frame_xy({"x": ball_src[0], "y": ball_src[1]}, box, width, height)
    detections = len(teammates) + len(opponents)
    self_pitch = None if self_xy is None else to_pitch(*self_xy)
    return LivePitchScene(
        phase=_phase_from_ball(ball_x),
        ball_x=ball_x,
        ball_y=ball_y,
        teammates=teammates,
        opponents=opponents,
        confidence=round(max(0.35, min(0.94, 0.32 + detections * 0.07 + (0.08 if self_blob is not None else 0.0))), 3),
        detection_count=detections,
        map_detected=True,
        map_ball_x=map_ball[0],
        map_ball_y=map_ball[1],
        map_teammates=frame_team[:7],
        map_opponents=frame_opp[:7],
        map_region={
            "x": round(left / max(1, width) * 100.0, 2),
            "y": round(top / max(1, height) * 100.0, 2),
            "width": round((right - left) / max(1, width) * 100.0, 2),
            "height": round((bottom - top) / max(1, height) * 100.0, 2),
        },
        self_x=None if self_pitch is None else self_pitch[0],
        self_y=None if self_pitch is None else self_pitch[1],
        facing_deg=facing,
        source="minimap",
    )


def _map_to_pitch(map_x: float, map_y: float) -> tuple[float, float]:
    """Minimap is drawn with the attacking goal at the top: pitch length runs up the map."""
    px = max(0.0, min(100.0, 100.0 - map_y))
    py = max(0.0, min(100.0, map_x))
    return round(px, 1), round(py, 1)


def _estimate_facing(pixels: list[tuple[int, int, int]], size: int, self_px: tuple[float, float]) -> float:
    """Direction (degrees, 0=right, 90=up) of the brighter camera cone drawn from the self marker."""
    sectors = 16
    sums = [0.0] * sectors
    counts = [0] * sectors
    sx, sy = self_px
    for y in range(size):
        for x in range(size):
            dx = x - sx
            dy = sy - y
            dist = math.hypot(dx, dy)
            if dist < 8 or dist > size * 0.42:
                continue
            r, g, b = pixels[y * size + x]
            if not (_is_pitch(r, g, b) or _is_rematch_court(r, g, b)):
                continue
            angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
            index = int(angle / (360.0 / sectors)) % sectors
            sums[index] += (r + g + b) / 3.0
            counts[index] += 1
    means = [sums[i] / counts[i] if counts[i] >= 12 else None for i in range(sectors)]
    valid = sorted(m for m in means if m is not None)
    if len(valid) < 4:
        return 90.0
    median = valid[len(valid) // 2]
    if valid[-1] - median < 4.0:
        return 90.0
    threshold = median + 0.45 * (valid[-1] - median)
    bright = [m is not None and m >= threshold for m in means]
    # Largest contiguous run of bright sectors (circular); its centre is the camera cone direction.
    best_run: list[int] = []
    for start in range(sectors):
        if not bright[start] or bright[(start - 1) % sectors]:
            continue
        run = []
        idx = start
        while bright[idx % sectors] and len(run) < sectors:
            run.append(idx % sectors)
            idx += 1
        if len(run) > len(best_run):
            best_run = run
    if not best_run:
        best_run = [max(range(sectors), key=lambda i: means[i] if means[i] is not None else -1.0)]
    step = 360.0 / sectors
    cos_sum = sum(math.cos(math.radians((i + 0.5) * step)) for i in best_run)
    sin_sum = sum(math.sin(math.radians((i + 0.5) * step)) for i in best_run)
    return round((math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0, 1)


# --------------------------------------------------------------------------- fusion


def _project_camera_point(cam_x: float, cam_y: float, self_pitch: tuple[float, float], facing_deg: float) -> tuple[float, float, float]:
    """Project a camera detection (frame %) into pitch coords anchored on the local player's minimap position."""
    y = max(cam_y, CAMERA_HORIZON_Y + 2.5)
    ratio = (CAMERA_LOCAL_Y - CAMERA_HORIZON_Y) / (y - CAMERA_HORIZON_Y)
    forward = max(-4.0, min(48.0, CAMERA_FORWARD_GAIN * (ratio - 1.0)))
    lateral = (cam_x - 50.0) / 50.0 * CAMERA_LATERAL_GAIN * (1.0 + forward / 15.0)
    theta = math.radians(facing_deg)
    # Minimap image coords (y down): forward = (cos t, -sin t), right = (sin t, cos t).
    fx, fy = math.cos(theta), -math.sin(theta)
    rx, ry = math.sin(theta), math.cos(theta)
    # self_pitch -> map coords
    map_x = self_pitch[1]
    map_y = 100.0 - self_pitch[0]
    mx = map_x + fx * forward + rx * lateral
    my = map_y + fy * forward + ry * lateral
    px, py = _map_to_pitch(mx, my)
    return px, py, forward


def _best_facing(
    camera: LivePitchScene,
    minimap: LivePitchScene,
    self_pitch: tuple[float, float],
    prior: float,
) -> float:
    """Pick the camera heading that best overlays camera detections onto minimap markers."""
    cam_points = [(p.x, p.y) for p in camera.teammates + camera.opponents if p.player_id != "self"]
    map_points = [(p.x, p.y) for p in minimap.teammates + minimap.opponents if p.player_id != "map_self"]
    if len(cam_points) < 1 or len(map_points) < 1:
        return prior
    best_angle = prior
    best_score = -1e9
    for step in range(24):
        angle = step * 15.0
        matches = 0
        total_dist = 0.0
        for cx, cy in cam_points:
            px, py, _depth = _project_camera_point(cx, cy, self_pitch, angle)
            nearest = min(math.hypot(px - mx, py - my) for mx, my in map_points)
            if nearest <= FUSION_MATCH_TOLERANCE:
                matches += 1
                total_dist += nearest
            else:
                total_dist += FUSION_MATCH_TOLERANCE * 1.5
        prior_bonus = max(0.0, 1.0 - _hue_diff(angle, prior) / 180.0) * 2.0
        score = matches * 10.0 - total_dist / len(cam_points) + prior_bonus
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def _fuse_scenes(camera: LivePitchScene, minimap: LivePitchScene) -> tuple[LivePitchScene, int]:
    if minimap.self_x is None or minimap.self_y is None:
        return minimap, 0
    self_pitch = (minimap.self_x, minimap.self_y)
    prior = minimap.facing_deg if minimap.facing_deg is not None else 90.0
    facing = _best_facing(camera, minimap, self_pitch, prior)

    markers: list[dict[str, object]] = []
    for player in minimap.teammates:
        markers.append({"side": "team", "x": player.x, "y": player.y, "id": player.player_id, "used": False})
    for player in minimap.opponents:
        markers.append({"side": "opp", "x": player.x, "y": player.y, "id": player.player_id, "used": False})

    matches = 0
    if camera.self_x is not None:
        matches += 1  # local player seen by camera and by the minimap self ring.

    projected: list[tuple[str, float, float, float]] = []
    for player in camera.teammates:
        if player.player_id == "self":
            continue
        px, py, depth = _project_camera_point(player.x, player.y, self_pitch, facing)
        projected.append(("team", px, py, depth))
    for player in camera.opponents:
        px, py, depth = _project_camera_point(player.x, player.y, self_pitch, facing)
        projected.append(("opp", px, py, depth))

    # Greedy nearest-neighbour matching, closest pairs first.
    pairs: list[tuple[float, int, int]] = []
    for ci, (_side, px, py, _depth) in enumerate(projected):
        for mi, marker in enumerate(markers):
            if marker["id"] == "map_self":
                continue
            dist = math.hypot(px - float(marker["x"]), py - float(marker["y"]))
            if dist <= FUSION_MATCH_TOLERANCE:
                pairs.append((dist, ci, mi))
    pairs.sort()
    used_cam: set[int] = set()
    for _dist, ci, mi in pairs:
        marker = markers[mi]
        if ci in used_cam or marker["used"]:
            continue
        _side, px, py, depth = projected[ci]
        cam_weight = 0.6 if depth < 10.0 else 0.35
        marker["x"] = round(cam_weight * px + (1.0 - cam_weight) * float(marker["x"]), 1)
        marker["y"] = round(cam_weight * py + (1.0 - cam_weight) * float(marker["y"]), 1)
        marker["used"] = True
        used_cam.add(ci)
        matches += 1

    team: list[PlayerPosition] = []
    opp: list[PlayerPosition] = []
    for marker in markers:
        target = team if marker["side"] == "team" else opp
        target.append(PlayerPosition(player_id=str(marker["id"]), role="mf", x=float(marker["x"]), y=float(marker["y"])))

    if camera.confidence >= 0.5:
        for ci, (side, px, py, depth) in enumerate(projected):
            if ci in used_cam or depth >= 15.0:
                continue
            target = team if side == "team" else opp
            if len(target) >= 7:
                continue
            if any(math.hypot(px - p.x, py - p.y) < 4.0 for p in team + opp):
                continue
            target.append(PlayerPosition(player_id=f"cam_{side}_{ci + 1}", role="mf", x=px, y=py))

    team = _dedupe_players(team)[:7]
    opp = _dedupe_players(opp)[:7]
    markers_total = len(minimap.teammates) + len(minimap.opponents)
    confidence = min(0.97, 0.5 + 0.06 * matches + 0.02 * markers_total)
    fused = LivePitchScene(
        phase=minimap.phase,
        ball_x=minimap.ball_x,
        ball_y=minimap.ball_y,
        teammates=team,
        opponents=opp,
        confidence=round(confidence, 3),
        detection_count=len(team) + len(opp),
        source="fused",
        fusion_matches=matches,
        self_x=minimap.self_x,
        self_y=minimap.self_y,
        facing_deg=facing,
    )
    return fused, matches


def _dedupe_players(players: list[PlayerPosition]) -> list[PlayerPosition]:
    kept: list[PlayerPosition] = []
    for player in players:
        if any(math.hypot(player.x - other.x, player.y - other.y) < DUPLICATE_DISTANCE for other in kept):
            continue
        kept.append(player)
    return kept


def _dedupe_blobs(blobs: list[Blob], limit: float) -> list[Blob]:
    kept: list[Blob] = []
    for blob in sorted(blobs, key=lambda item: -item["size"]):
        if any(math.hypot(blob["x"] - other["x"], blob["y"] - other["y"]) * (MINIMAP_SAMPLE / 100.0) < limit for other in kept):
            continue
        kept.append(blob)
    return kept


# --------------------------------------------------------------------------- helpers


def _analysis_image(image: Image.Image, max_width: int) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= max_width:
        return rgb
    target_h = max(90, int(height * max_width / width))
    return rgb.resize((max_width, target_h))


def _blob_to_frame_xy(
    blob: Blob,
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    left, top, right, bottom = box
    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    x = (left + blob["x"] / 100.0 * crop_w) / max(1, frame_width) * 100.0
    y = (top + blob["y"] / 100.0 * crop_h) / max(1, frame_height) * 100.0
    return round(max(0.0, min(100.0, x)), 1), round(max(0.0, min(100.0, y)), 1)


def _blob_to_frame_player(
    blob: Blob,
    player_id: str,
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> PlayerPosition:
    x, y = _blob_to_frame_xy(blob, box, frame_width, frame_height)
    return PlayerPosition(player_id=player_id, role="mf", x=x, y=y)


def _collect_blobs(
    width: int,
    height: int,
    pixels: list[tuple[int, int, int]],
    keep: KeepFn,
    min_size: int,
    max_size: int,
) -> list[Blob]:
    """Two-pass connected components (4-neighbour) with union-find."""
    labels = [-1] * (width * height)
    parent: list[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for y in range(height):
        row = y * width
        for x in range(width):
            i = row + x
            r, g, b = pixels[i]
            if not keep(x, y, r, g, b):
                continue
            left_label = labels[i - 1] if x > 0 else -1
            up_label = labels[i - width] if y > 0 else -1
            if left_label < 0 and up_label < 0:
                labels[i] = len(parent)
                parent.append(labels[i])
            elif left_label >= 0 and up_label >= 0:
                labels[i] = min(left_label, up_label)
                union(left_label, up_label)
            else:
                labels[i] = max(left_label, up_label)

    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        if label < 0:
            continue
        root = find(label)
        groups.setdefault(root, []).append(i)

    blobs: list[Blob] = []
    for pixel_ids in groups.values():
        count = len(pixel_ids)
        if count < min_size or count > max_size:
            continue
        xs = [pid % width for pid in pixel_ids]
        ys = [pid // width for pid in pixel_ids]
        cos_sum = sin_sum = 0.0
        brightness = saturation = blue_bias = 0.0
        for pid in pixel_ids:
            r, g, b = pixels[pid]
            hue = math.radians(_hue(r, g, b))
            cos_sum += math.cos(hue)
            sin_sum += math.sin(hue)
            brightness += (r + g + b) / 3.0
            saturation += _saturation(r, g, b)
            blue_bias += b - r
        mean_hue = (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0
        w = max(xs) - min(xs) + 1
        h = max(ys) - min(ys) + 1
        blobs.append(
            {
                "x": (sum(xs) / count) / max(1, width - 1) * 100.0,
                "y": (sum(ys) / count) / max(1, height - 1) * 100.0,
                "size": float(count),
                "hue": mean_hue,
                "brightness": brightness / count,
                "saturation": saturation / count,
                "blue_bias": blue_bias / count,
                "w": float(w),
                "h": float(h),
                "fill": count / float(w * h),
            }
        )
    return blobs


def _looks_like_ball(blob: Blob) -> bool:
    magenta = blob["hue"] >= 280 or blob["hue"] <= 20
    bright_small = blob["size"] <= 24 and blob["brightness"] >= 140
    return blob["size"] <= 28 and (bright_small or (magenta and blob["brightness"] >= 90))


def _pick_ball(blobs: list[Blob]) -> Blob:
    ranked = sorted(blobs, key=lambda blob: (blob["size"], -blob["brightness"], -abs(blob["x"] - 50.0)))
    for blob in ranked:
        if _looks_like_ball(blob):
            return blob
    return min(blobs, key=lambda blob: blob["size"])


def kit_distance(blob_hue: float, blob_sat: float, blob_br: float, hue: float | None, sat: float, br: float) -> float:
    """Distance in a hue-circle * saturation disc plus brightness; ~0.45 separates distinct kits."""
    if hue is None:
        hx = hy = 0.0
    else:
        hx = math.cos(math.radians(hue)) * sat
        hy = math.sin(math.radians(hue)) * sat
    bx = math.cos(math.radians(blob_hue)) * blob_sat
    by = math.sin(math.radians(blob_hue)) * blob_sat
    return math.hypot(bx - hx, by - hy) + abs(blob_br - br) / 255.0 * 0.6


def _split_by_profile(blobs: list[Blob], profile: KitProfile, limit: float = 0.5) -> tuple[list[Blob], list[Blob]]:
    team: list[Blob] = []
    opp: list[Blob] = []
    for blob in blobs:
        d_team = kit_distance(blob["hue"], blob["saturation"], blob["brightness"], profile.team_hue, profile.team_saturation, profile.team_brightness)
        d_opp = kit_distance(blob["hue"], blob["saturation"], blob["brightness"], profile.opp_hue, profile.opp_saturation, profile.opp_brightness)
        if profile.team_hue is None:
            d_team = 9.0
        if profile.opp_hue is None:
            d_opp = 9.0
        if min(d_team, d_opp) > limit:
            continue
        (team if d_team <= d_opp else opp).append(blob)
    return team, opp


def _split_rematch_kits(
    blobs: list[Blob],
    strict: bool = False,
    kit_profile: KitProfile | None = None,
) -> tuple[list[Blob], list[Blob]]:
    if kit_profile is not None and kit_profile.samples > 0 and (kit_profile.team_hue is not None or kit_profile.opp_hue is not None):
        return _split_by_profile(blobs, kit_profile)
    team: list[Blob] = []
    opp: list[Blob] = []
    for blob in blobs:
        if _is_team_kit(blob):
            team.append(blob)
        elif _is_opp_kit(blob):
            opp.append(blob)
    if team or opp:
        return team, [blob for blob in opp if blob not in team]
    if strict or len(blobs) < 2:
        return ([], []) if strict else (blobs, [])
    hues = sorted(blob["hue"] for blob in blobs)
    median = hues[len(hues) // 2]
    left = [blob for blob in blobs if blob["hue"] <= median]
    right = [blob for blob in blobs if blob["hue"] > median]
    if not left or not right:
        by_x = sorted(blobs, key=lambda blob: blob["x"])
        mid = max(1, len(by_x) // 2)
        return by_x[:mid], by_x[mid:]
    return left, right


def _is_team_kit(blob: Blob) -> bool:
    hue = blob["hue"]
    if 185 <= hue <= 265 and blob["saturation"] >= 0.18 and blob["brightness"] <= 160:
        return True
    if hue <= 35 or hue >= 345:
        return blob["saturation"] >= 0.28 and blob["brightness"] >= 45
    if 80 <= hue <= 150 and blob["saturation"] >= 0.25:
        return True
    return blob["blue_bias"] >= 18 and blob["brightness"] <= 140


def _is_opp_kit(blob: Blob) -> bool:
    return blob["brightness"] >= 145 and blob["saturation"] <= 0.42


def _to_players(blobs: list[Blob], prefix: str) -> list[PlayerPosition]:
    roles = ["gk", "df", "df", "mf", "mf", "fw", "fw"]
    ordered = sorted(blobs, key=lambda blob: blob["x"])
    players: list[PlayerPosition] = []
    for index, blob in enumerate(ordered):
        players.append(
            PlayerPosition(
                player_id=f"{prefix}_{index + 1}",
                role=roles[min(index, len(roles) - 1)],
                x=round(max(0.0, min(100.0, blob["x"])), 1),
                y=round(max(0.0, min(100.0, blob["y"])), 1),
            )
        )
    return players


def _phase_from_ball(ball_x: float) -> Phase:
    if ball_x >= 62.0:
        return "attack"
    if ball_x <= 38.0:
        return "defense"
    return "transition"


def _empty_scene() -> LivePitchScene:
    return LivePitchScene(
        phase="transition",
        ball_x=50.0,
        ball_y=50.0,
        teammates=[],
        opponents=[],
        confidence=0.0,
        detection_count=0,
    )


def _is_background(r: int, g: int, b: int) -> bool:
    if (r + g + b) / 3.0 < 16:
        return True
    return _is_pitch(r, g, b) or _is_rematch_court(r, g, b) or _is_sky(r, g, b)


def _is_pitch(r: int, g: int, b: int) -> bool:
    return g > 70 and g > r * 1.12 and g > b * 1.05 and _saturation(r, g, b) > 0.18


def _is_rematch_court(r: int, g: int, b: int) -> bool:
    hue = _hue(r, g, b)
    sat = _saturation(r, g, b)
    brightness = (r + g + b) / 3.0
    teal = 165.0 <= hue <= 220.0 and sat >= 0.22 and 18.0 <= brightness <= 165.0
    return teal and b >= r + 6 and g >= r


def _is_sky(r: int, g: int, b: int) -> bool:
    brightness = (r + g + b) / 3.0
    sat = _saturation(r, g, b)
    hue = _hue(r, g, b)
    pastel = brightness >= 145 and sat <= 0.48 and 20.0 <= hue <= 210.0
    ice = brightness >= 120 and sat <= 0.28
    return pastel or ice


def _hue(r: int, g: int, b: int) -> float:
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    mx = max(rf, gf, bf)
    mn = min(rf, gf, bf)
    delta = mx - mn
    if delta < 1e-6:
        return 0.0
    if mx == rf:
        hue = ((gf - bf) / delta) % 6.0
    elif mx == gf:
        hue = (bf - rf) / delta + 2.0
    else:
        hue = (rf - gf) / delta + 4.0
    return hue * 60.0


def _saturation(r: int, g: int, b: int) -> float:
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx == 0:
        return 0.0
    return (mx - mn) / mx


def color_label(hue: float | None, sat: float, brightness: float) -> str:
    if hue is None:
        return "unknown"
    if sat < 0.28:
        if brightness >= 170:
            return "white"
        if brightness >= 90:
            return "grey"
        return "black"
    if brightness < 45:
        return "black"
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 42:
        return "orange"
    if hue < 68:
        return "yellow"
    if hue < 160:
        return "green"
    if hue < 200:
        return "teal"
    if hue < 262:
        return "blue"
    if hue < 300:
        return "purple"
    return "pink"
