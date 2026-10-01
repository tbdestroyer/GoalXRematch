"""Small-number reader for scoreboard cells.

Windows OCR ignores isolated single digits, which is most of a 5v5 results
table. This module binarises a cell, splits it into glyphs, and matches each
glyph against digit templates by normalised correlation. Templates come from
bold system fonts and are extended at runtime with glyphs harvested from
multi-digit numbers OCR did recognise on the same screen, so the matcher adapts
to the game's font.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GLYPH_W = 16
GLYPH_H = 24
FONT_CANDIDATES = ("arialbd.ttf", "segoeuib.ttf", "bahnschrift.ttf", "impact.ttf", "verdanab.ttf", "arial.ttf")


@dataclass
class DigitRead:
    text: str
    confidence: float
    glyphs: int

    @property
    def value(self) -> int | None:
        return int(self.text) if self.text.isdigit() else None


def _to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Return boolean ink mask (True = text) regardless of polarity."""
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < 40.0:
        return np.zeros_like(gray, dtype=bool)
    # Otsu threshold.
    hist, edges = np.histogram(gray, bins=64, range=(lo, hi + 1e-3))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    best_t, best_var = (lo + hi) / 2.0, -1.0
    w0 = 0.0
    sum0 = 0.0
    sum_all = float((hist * centers).sum())
    for i in range(len(hist)):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * centers[i]
        m0 = sum0 / w0
        m1 = (sum_all - sum0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, centers[i]
    bright = gray > best_t
    # Text is the minority class touching the border least.
    border = np.concatenate([bright[0, :], bright[-1, :], bright[:, 0], bright[:, -1]])
    background_is_bright = border.mean() > 0.5
    return ~bright if background_is_bright else bright


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """4-connected components -> list of (x0, y0, x1, y1) inclusive boxes."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    boxes: list[tuple[int, int, int, int]] = []
    current = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x]:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            x0 = x1 = x
            y0 = y1 = y
            while stack:
                cy, cx = stack.pop()
                x0, x1 = min(x0, cx), max(x1, cx)
                y0, y1 = min(y0, cy), max(y1, cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
            boxes.append((x0, y0, x1, y1))
    return boxes


def extract_glyphs(image: Image.Image) -> list[np.ndarray]:
    """Split a cell into normalised glyph arrays (GLYPH_H x GLYPH_W, 0..1), left to right."""
    gray = _to_gray(image)
    mask = _binarize(gray)
    if not mask.any():
        return []
    boxes = _components(mask)
    if not boxes:
        return []
    heights = [b[3] - b[1] + 1 for b in boxes]
    max_h = max(heights)
    if max_h < 6:
        return []
    # Keep components that are tall enough to be digits (drops dots, noise, underscores).
    kept = [b for b, h in zip(boxes, heights) if h >= max_h * 0.55]
    kept.sort(key=lambda b: b[0])
    # Merge horizontally overlapping boxes (e.g. broken strokes).
    merged: list[list[int]] = []
    for b in kept:
        if merged and b[0] <= merged[-1][2] - 1:
            m = merged[-1]
            m[0], m[1], m[2], m[3] = min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])
        else:
            merged.append(list(b))
    # Use a common vertical band so all glyphs share baseline scaling.
    top = min(m[1] for m in merged)
    bottom = max(m[3] for m in merged)
    glyphs: list[np.ndarray] = []
    for x0, _, x1, _ in merged:
        crop = mask[top : bottom + 1, x0 : x1 + 1].astype(np.uint8) * 255
        glyphs.append(_normalise(Image.fromarray(crop)))
    return glyphs


def _normalise(glyph: Image.Image) -> np.ndarray:
    w, h = glyph.size
    # Preserve aspect: fit into GLYPH_H tall, pad width to GLYPH_W (narrow "1" stays narrow).
    scale = GLYPH_H / max(1, h)
    new_w = max(1, min(GLYPH_W, int(round(w * scale))))
    resized = glyph.resize((new_w, GLYPH_H), Image.BILINEAR)
    canvas = Image.new("L", (GLYPH_W, GLYPH_H), 0)
    canvas.paste(resized, ((GLYPH_W - new_w) // 2, 0))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 1e-6:
        return 0.0
    return float((a * b).sum() / denom)


class DigitMatcher:
    def __init__(self, use_fonts: bool = True) -> None:
        self.templates: dict[str, list[np.ndarray]] = {str(d): [] for d in range(10)}
        self.learned: dict[str, int] = {str(d): 0 for d in range(10)}
        if use_fonts:
            self._load_font_templates()

    def _load_font_templates(self) -> None:
        for name in FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(name, 64)
            except OSError:
                continue
            for d in range(10):
                img = Image.new("L", (96, 110), 0)
                ImageDraw.Draw(img).text((16, 10), str(d), fill=255, font=font)
                glyphs = extract_glyphs(img)
                if len(glyphs) == 1:
                    self.templates[str(d)].append(glyphs[0])

    def learn(self, text: str, image: Image.Image, max_per_digit: int = 8) -> bool:
        """Harvest glyph templates from a cell whose text OCR already recognised."""
        if not text.isdigit():
            return False
        glyphs = extract_glyphs(image)
        if len(glyphs) != len(text):
            return False
        for ch, glyph in zip(text, glyphs):
            bucket = self.templates[ch]
            if any(_correlation(glyph, t) > 0.97 for t in bucket):
                continue
            bucket.append(glyph)
            self.learned[ch] += 1
            if len(bucket) > max_per_digit + 6:
                del bucket[: len(bucket) - (max_per_digit + 6)]
        return True

    def match_glyph(self, glyph: np.ndarray) -> tuple[str, float]:
        best_digit, best_score = "", -1.0
        for digit, bucket in self.templates.items():
            for template in bucket:
                score = _correlation(glyph, template)
                # Learned (in-game font) templates get a small bonus.
                if self.learned[digit] and score > 0.5:
                    score += 0.03
                if score > best_score:
                    best_digit, best_score = digit, score
        return best_digit, best_score

    def read(self, image: Image.Image, min_confidence: float = 0.55, max_digits: int = 3) -> DigitRead | None:
        glyphs = extract_glyphs(image)
        if not glyphs or len(glyphs) > max_digits:
            return None
        text = ""
        scores: list[float] = []
        for glyph in glyphs:
            digit, score = self.match_glyph(glyph)
            if not digit:
                return None
            text += digit
            scores.append(score)
        confidence = min(scores) if scores else 0.0
        if confidence < min_confidence:
            return None
        return DigitRead(text=text, confidence=confidence, glyphs=len(glyphs))
