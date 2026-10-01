"""Detection of Rematch action feedback popups ("PASS", "GO!", "TACKLE", ...).

Rematch flashes large pale-cyan/white block text above the local player when an
action is confirmed. We find that text as a line of bright glyph blobs in the
centre band of the frame, build a small binary signature of the line, and
classify it either by learned templates (``PopupTemplateStore``) or, when no
templates exist yet, by a glyph-count heuristic.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ANALYSIS_WIDTH = 860
SIGNATURE_W = 64
SIGNATURE_H = 16

# Centre band where popups appear (fractions of the frame).
BAND_X = (0.26, 0.74)
BAND_Y = (0.18, 0.66)


@dataclass
class PopupDetection:
    label: str
    confidence: float
    glyphs: int
    aspect: float
    bbox: dict[str, float]
    signature: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "glyphs": self.glyphs,
            "aspect": round(self.aspect, 2),
            "bbox": self.bbox,
        }


def _is_popup_pixel(r: int, g: int, b: int) -> bool:
    # Pale cyan/white glyph body: bright, low red deficit, green/blue high.
    if g < 185 or b < 175:
        return False
    if r > g + 10:
        return False
    brightness = (r + g + b) / 3.0
    return brightness >= 180 and (g - r) >= -5


def _components(mask: list[bool], width: int, height: int, min_size: int) -> list[dict[str, int]]:
    seen = [False] * (width * height)
    comps: list[dict[str, int]] = []
    for start in range(width * height):
        if not mask[start] or seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        xs_min = xs_max = start % width
        ys_min = ys_max = start // width
        size = 0
        while queue:
            idx = queue.popleft()
            size += 1
            x, y = idx % width, idx // width
            xs_min, xs_max = min(xs_min, x), max(xs_max, x)
            ys_min, ys_max = min(ys_min, y), max(ys_max, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    nidx = ny * width + nx
                    if mask[nidx] and not seen[nidx]:
                        seen[nidx] = True
                        queue.append(nidx)
        if size >= min_size:
            comps.append({"x0": xs_min, "x1": xs_max, "y0": ys_min, "y1": ys_max, "size": size})
    return comps


def detect_action_popup(image: Image.Image) -> PopupDetection | None:
    rgb = image.convert("RGB")
    full_w, full_h = rgb.size
    scale = ANALYSIS_WIDTH / full_w if full_w > ANALYSIS_WIDTH else 1.0
    sample = rgb.resize((int(full_w * scale), max(1, int(full_h * scale)))) if scale != 1.0 else rgb
    width, height = sample.size
    x0, x1 = int(width * BAND_X[0]), int(width * BAND_X[1])
    y0, y1 = int(height * BAND_Y[0]), int(height * BAND_Y[1])
    band = sample.crop((x0, y0, x1, y1))
    bw, bh = band.size
    pixels = list(band.getdata())
    mask = [_is_popup_pixel(r, g, b) for (r, g, b) in pixels]
    if sum(mask) < bw * bh * 0.004:
        return None

    min_glyph_h = max(8, int(height * 0.045))
    max_glyph_h = int(height * 0.30)
    comps = [
        c
        for c in _components(mask, bw, bh, min_size=max(12, min_glyph_h * 2))
        if min_glyph_h <= (c["y1"] - c["y0"] + 1) <= max_glyph_h and (c["x1"] - c["x0"] + 1) <= max_glyph_h * 1.4
    ]
    if len(comps) < 2:
        return None

    # Group glyphs into one text line: pick the tallest glyph, keep those overlapping it vertically.
    anchor = max(comps, key=lambda c: c["y1"] - c["y0"])
    a_h = anchor["y1"] - anchor["y0"] + 1
    line = [
        c
        for c in comps
        if min(c["y1"], anchor["y1"]) - max(c["y0"], anchor["y0"]) >= a_h * 0.45
    ]
    line.sort(key=lambda c: c["x0"])
    # Drop glyphs far from their neighbours (stray white kit pixels).
    merged: list[dict[str, int]] = []
    for c in line:
        if merged and c["x0"] - merged[-1]["x1"] > a_h * 1.3:
            if len(merged) >= 2:
                break
            merged = [c]
            continue
        merged.append(c)
    if len(merged) < 2:
        return None
    lx0 = min(c["x0"] for c in merged)
    lx1 = max(c["x1"] for c in merged)
    ly0 = min(c["y0"] for c in merged)
    ly1 = max(c["y1"] for c in merged)
    line_w = lx1 - lx0 + 1
    line_h = ly1 - ly0 + 1
    if line_h <= 0 or line_w < line_h * 1.2:
        return None

    signature = _signature(mask, bw, lx0, ly0, lx1, ly1)
    bbox = {
        "x": round((x0 + lx0) / width * 100.0, 2),
        "y": round((y0 + ly0) / height * 100.0, 2),
        "width": round(line_w / width * 100.0, 2),
        "height": round(line_h / height * 100.0, 2),
    }
    glyphs = len(merged)
    aspect = line_w / line_h
    label, confidence = _heuristic_label(glyphs, aspect)
    return PopupDetection(label=label, confidence=confidence, glyphs=glyphs, aspect=aspect, bbox=bbox, signature=signature)


def _signature(mask: list[bool], bw: int, x0: int, y0: int, x1: int, y1: int) -> list[int]:
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    sig: list[int] = []
    for sy in range(SIGNATURE_H):
        ys = y0 + int(sy * h / SIGNATURE_H)
        ye = y0 + max(ys - y0 + 1, int((sy + 1) * h / SIGNATURE_H))
        for sx in range(SIGNATURE_W):
            xs = x0 + int(sx * w / SIGNATURE_W)
            xe = x0 + max(xs - x0 + 1, int((sx + 1) * w / SIGNATURE_W))
            total = 0
            hits = 0
            for y in range(ys, min(ye, y1 + 1)):
                row = y * bw
                for x in range(xs, min(xe, x1 + 1)):
                    total += 1
                    if mask[row + x]:
                        hits += 1
            sig.append(1 if total and hits * 2 >= total else 0)
    return sig


def _heuristic_label(glyphs: int, aspect: float) -> tuple[str, float]:
    if glyphs == 4 and 2.0 <= aspect <= 4.6:
        return "pass", 0.45
    if glyphs == 3 and 1.4 <= aspect <= 2.6:
        return "go", 0.35
    if glyphs >= 6:
        return "long", 0.2
    return "unknown", 0.1


def signature_similarity(a: list[int], b: list[int]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    agree = sum(1 for x, y in zip(a, b) if x == y)
    ink = sum(1 for x, y in zip(a, b) if x == 1 or y == 1)
    overlap = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    jaccard = overlap / ink if ink else 0.0
    return 0.5 * (agree / len(a)) + 0.5 * jaccard


class PopupTemplateStore:
    """Learned popup signatures per label, persisted as JSON."""

    def __init__(self, path: str | Path = "data/popup-templates.json") -> None:
        self.path = Path(path)
        self.templates: dict[str, list[list[int]]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.templates = {str(k): [list(map(int, sig)) for sig in v] for k, v in raw.items()}
            except (OSError, ValueError):
                self.templates = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.templates), encoding="utf-8")

    def learn(self, label: str, detection: PopupDetection, max_per_label: int = 6) -> int:
        label = label.strip().lower()
        sigs = self.templates.setdefault(label, [])
        sigs.append(list(detection.signature))
        if len(sigs) > max_per_label:
            del sigs[: len(sigs) - max_per_label]
        self.save()
        return len(sigs)

    def forget(self, label: str | None = None) -> None:
        if label is None:
            self.templates = {}
        else:
            self.templates.pop(label.strip().lower(), None)
        self.save()

    def classify(self, detection: PopupDetection, threshold: float = 0.78) -> PopupDetection:
        best_label = None
        best_score = 0.0
        for label, sigs in self.templates.items():
            for sig in sigs:
                score = signature_similarity(detection.signature, sig)
                if score > best_score:
                    best_label, best_score = label, score
        if best_label is not None and best_score >= threshold:
            detection.label = best_label
            detection.confidence = round(best_score, 3)
        elif best_label is not None and self.templates:
            # Templates exist but none match well: downgrade heuristic confidence.
            detection.confidence = min(detection.confidence, 0.3)
        return detection

    def labels(self) -> dict[str, int]:
        return {label: len(sigs) for label, sigs in self.templates.items()}
