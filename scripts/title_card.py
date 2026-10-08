"""Title card for the cinema bars at the end of a video: a short text in wide-spaced capitals whose letters LEAN in a wave.

The effect (2026-10-04, from the user's reference thumbnail "KNOWLEDGE AND WISDOM"): bold capitals with a wide tracking, and the slant of the
letters changes along the line, as if the text were printed on a gently rippling surface. At the left end of a line the letters lean one way, in
the middle they stand straight, at the right end they lean the other way, and the second line is the mirror of the first; the ripple travels
slowly so the title lives. The text FADES IN, then holds.

`render_title_frames` writes one transparent PNG per frame (RGBA, the size of the bar it goes on); scripts/trend_philosopher.py feeds them to
ffmpeg as an image sequence and overlays them. Only Pillow and numpy are needed.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TITLE_EFFECTS = ("shear_wave", "plain")
DEFAULT_TITLE_EFFECT = "shear_wave"
TITLE_POSITIONS = ("top", "bottom", "center")
DEFAULT_TITLE_POSITION = "top"
TITLE_SECONDS = 5.0  # the title is on screen for the last this-many seconds of the video
TITLE_FADE_SECONDS = 0.6
TITLE_TRACKING = 0.16  # extra space between letters, in em
TITLE_LEAN = 0.32  # peak slant of the letters: tan(18 degrees)
TITLE_WAVE_PERIOD_SECONDS = 4.0  # the ripple travels once along the line in this long
_FONT_DIRS = (Path(r"C:\Windows\Fonts"), Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts")


def find_font(name_or_path: str | Path) -> Path:
    """A font file from a path, or from a file name looked up in $TREND_CAPTION_FONTSDIR and the usual font folders."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    folders = [Path(os.environ["TREND_CAPTION_FONTSDIR"])] if os.environ.get("TREND_CAPTION_FONTSDIR") else []
    for folder in [*folders, *_FONT_DIRS]:
        if folder.is_dir():
            for found in folder.rglob(candidate.name):
                return found
    raise FileNotFoundError(f"font not found: {name_or_path}")


def font_family(path: str | Path) -> str:
    """The family name of a font file (what an ASS style or fontconfig calls it): 'Constantia' for constanb.ttf."""
    return ImageFont.truetype(str(path), 40).getname()[0]


def title_lines(text: str) -> list[str]:
    """The text as capital lines ('\\n' or '|' separate them)."""
    lines = [line.strip().upper() for line in text.replace("|", "\n").split("\n") if line.strip()]
    if not lines:
        raise ValueError("the end title is empty")
    return lines


def _spaced_width(font: ImageFont.FreeTypeFont, line: str, tracking_px: float) -> float:
    return sum(font.getlength(ch) for ch in line) + tracking_px * (len(line) - 1)


def layout_mask(lines: list[str], font_path: Path, *, width: int, height: int, max_fraction: float = 0.86,
                tracking: float = TITLE_TRACKING, max_size: int = 150) -> tuple[Image.Image, list[tuple[int, int]]]:
    """The text block as an L mask of width x height (the bar), centred, with the (top, bottom) rows of every line. The font size is the
    largest that keeps the widest line within max_fraction of the width (and the block within 80 % of the height)."""
    size = max_size
    while size > 24:
        font = ImageFont.truetype(str(font_path), size)
        widest = max(_spaced_width(font, line, tracking * size) for line in lines)
        asc, desc = font.getmetrics()
        block = (asc + desc) * 1.18 * len(lines)
        if widest <= width * max_fraction and block <= height * 0.8:
            break
        size -= 2
    font = ImageFont.truetype(str(font_path), size)
    asc, desc = font.getmetrics()
    line_h = (asc + desc) * 1.18
    top = (height - line_h * len(lines)) / 2
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    rows = []
    for i, line in enumerate(lines):
        x = (width - _spaced_width(font, line, tracking * size)) / 2
        y = top + i * line_h
        for ch in line:
            draw.text((x, y), ch, font=font, fill=255)
            x += font.getlength(ch) + tracking * size
        rows.append((int(y + asc * 0.12), int(y + asc * 1.02)))  # roughly the cap height of this line
    return mask, rows


def shear_wave(mask: np.ndarray, rows: list[tuple[int, int]], t: float, *, lean: float = TITLE_LEAN,
               period: float = TITLE_WAVE_PERIOD_SECONDS) -> np.ndarray:
    """Slant each line of the mask by an amount that changes along x: output(x, y) = mask(x - s(x) * (y_centre - y), y), s(x) = lean * sin(phase).
    Positive s shifts the TOP of a letter to the right (a forward slash). Half a period of the wave spans the line, so one end leans forward, the
    middle stands straight and the other end leans back; each next line is the mirror of the previous one."""
    height, width = mask.shape
    out = np.zeros_like(mask, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    covered = np.zeros(height, bool)
    for i, (top, bottom) in enumerate(rows):
        centre = (top + bottom) / 2
        left = np.nonzero(mask[max(top, 0):bottom].max(axis=0))[0]
        if left.size == 0:
            continue
        x0, x1 = float(left.min()), float(left.max())
        u = np.clip((xs - x0) / max(x1 - x0, 1.0), 0.0, 1.0)  # 0 at the left end of the line, 1 at the right end
        phase = math.pi * u + math.pi / 2 + (math.pi if i % 2 else 0.0) - 2 * math.pi * t / period
        slope = lean * np.cos(phase - math.pi / 2)  # = lean * sin(phase)
        # the band of rows this line owns: from half-way to the previous line to half-way to the next one
        band_top = 0 if i == 0 else (rows[i - 1][1] + top) // 2
        band_bottom = height if i == len(rows) - 1 else (bottom + rows[i + 1][0]) // 2
        for y in range(max(band_top, 0), min(band_bottom, height)):
            source_x = xs - slope * (centre - y)
            out[y] = np.interp(source_x, xs, mask[y].astype(np.float32), left=0.0, right=0.0)
            covered[y] = True
    out[~covered] = mask[~covered]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_title_frames(
    text: str, font: str | Path, out_dir: Path, *, width: int, height: int, fps: int, seconds: float = TITLE_SECONDS,
    fade_seconds: float = TITLE_FADE_SECONDS, effect: str = DEFAULT_TITLE_EFFECT, tracking: float = TITLE_TRACKING,
    lean: float = TITLE_LEAN, wave_period: float = TITLE_WAVE_PERIOD_SECONDS,
) -> int:
    """Write out_dir/title_0000.png ... (transparent RGBA, width x height, white text, fading in over `fade_seconds`) for `seconds` of video
    and return the number of frames. effect "shear_wave" is the leaning wave above, "plain" the same text standing straight."""
    if effect not in TITLE_EFFECTS:
        raise ValueError(f"title effect must be one of {TITLE_EFFECTS}, got {effect!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = title_lines(text)
    mask, rows = layout_mask(lines, find_font(font), width=width, height=height, tracking=tracking)
    base = np.asarray(mask, np.uint8)
    frames = max(1, round(seconds * fps))
    white = np.full((height, width, 3), 255, np.uint8)
    for k in range(frames):
        t = k / fps
        shaped = shear_wave(base, rows, t, lean=lean, period=wave_period) if effect == "shear_wave" else base
        fade = min(1.0, t / fade_seconds) if fade_seconds > 0 else 1.0
        fade = fade * fade * (3 - 2 * fade)  # smoothstep
        alpha = (shaped.astype(np.float32) * fade).astype(np.uint8)
        Image.fromarray(np.dstack([white, alpha]), "RGBA").save(out_dir / f"title_{k:04d}.png", compress_level=1)
    return frames
