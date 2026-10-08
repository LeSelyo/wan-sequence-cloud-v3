"""Tools of the motion LAB: put the variants of an experiment side by side in ONE video with their settings written under each panel (so the choice is made by looking), and measure a few things in
the clips (does the person blink? how much does the picture move?).

    compose(panels, out, title)      panels = [{"clip": Path, "label": "S2V + LoRA", "lines": ["steps 4", "prompt: ..."]}, ...]
    blink_series(clip, face)         eye openness per frame -> blinks counted (face = {"cx","cy","w","h"} in fractions of the frame, from scripts/face_tools.py)
    motion_amount(clip)              how much the picture changes from frame to frame (a frozen pose is near 0)
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FONT_TITLE = ROOT / "assets" / "fonts" / "Anton-Regular.ttf"
FONT_TEXT = Path(r"C:\Windows\Fonts\arial.ttf")
PANEL = (480, 832)
CAPTION_HEIGHT = 250
HEAD = 70


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    for candidate in (path, Path(r"C:\Windows\Fonts\arialbd.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def caption_image(panels: list[dict], title: str) -> Image.Image:
    """The overlay (RGBA, the size of the whole video): a title bar on top, under every panel its label (big) and its settings (small, wrapped)."""
    width, height = PANEL[0] * len(panels), HEAD + PANEL[1] + CAPTION_HEIGHT
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # transparent over the panels, opaque bars for the title and the captions
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, HEAD), fill=(14, 14, 14, 255))
    draw.text((16, 12), title, font=_font(FONT_TITLE, 40), fill=(255, 122, 0, 255))
    for index, panel in enumerate(panels):
        x = index * PANEL[0]
        draw.rectangle((x, HEAD + PANEL[1], x + PANEL[0] - 4, height), fill=(24, 24, 24, 255))
        draw.text((x + 12, HEAD + PANEL[1] + 8), f"{chr(65 + index)}  {panel['label']}", font=_font(FONT_TITLE, 30), fill=(255, 255, 255, 255))
        y = HEAD + PANEL[1] + 52
        for line in panel.get("lines", []):
            for wrapped in textwrap.wrap(line, width=52) or [""]:
                if y > height - 22:
                    break
                draw.text((x + 12, y), wrapped, font=_font(FONT_TEXT, 17), fill=(215, 215, 215, 255))
                y += 21
    return image


def compose(panels: list[dict], out: Path, title: str, seconds: float = 3.2, fps: int = 30) -> Path:
    """Side-by-side video of the clips of the panels (each cut to `seconds`, held on its last frame if short), a still picture works too, captions burned in."""
    out.parent.mkdir(parents=True, exist_ok=True)
    overlay = out.with_suffix(".captions.png")
    caption_image(panels, title).save(overlay)
    inputs, filters = [], []
    for index, panel in enumerate(panels):
        path = Path(panel["clip"])
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            inputs += ["-loop", "1", "-t", f"{seconds:.2f}", "-i", str(path)]
        else:
            inputs += ["-i", str(path)]
        filters.append(f"[{index}:v]fps={fps},scale={PANEL[0]}:{PANEL[1]}:force_original_aspect_ratio=increase,crop={PANEL[0]}:{PANEL[1]},tpad=stop_mode=clone:stop_duration=3,trim=duration={seconds:.2f},setpts=PTS-STARTPTS[p{index}]")
    stack = "".join(f"[p{i}]" for i in range(len(panels)))
    count = len(panels)
    inputs += ["-i", str(overlay)]
    graph = ";".join(filters) + f";{stack}hstack=inputs={count}[row];[row]pad={PANEL[0] * count}:{HEAD + PANEL[1] + CAPTION_HEIGHT}:0:{HEAD}[padded];[padded][{count}:v]overlay=0:0:format=auto[v]"
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]", "-t", f"{seconds:.2f}", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", str(out)],
                   check=True)
    overlay.unlink(missing_ok=True)
    return out


def gray_frames(clip: Path, size: tuple[int, int] = (480, 832)) -> np.ndarray:
    raw = subprocess.run([mt.ffmpeg_binary(), "-v", "error", "-i", str(clip), "-vf", f"fps=16,scale={size[0]}:{size[1]}:flags=lanczos,format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, size[1], size[0]).astype(float)


def motion_amount(clip: Path) -> float:
    """Mean absolute change between consecutive frames, 0-255 scale (a frozen pose is below 0.3, a walking person in a corridor is above 1.5)."""
    frames = gray_frames(clip, (240, 416))
    return round(float(np.abs(np.diff(frames, axis=0)).mean()), 2) if len(frames) > 1 else 0.0


def blink_series(clip: Path, face: dict, size: tuple[int, int] = (480, 832)) -> dict:
    """Eye openness per frame = the contrast (standard deviation) inside the band of the eyes: open eyes (dark iris and lashes against skin and white) have a lot, a closed eyelid very little.
    A blink = a dip below 70 % of the usual openness for 1 to 6 frames at 16 fps. Returns {"blinks", "openness", "series"}."""
    frames = gray_frames(clip, size)
    x0, x1 = int((face["cx"] - 0.40 * face["w"]) * size[0]), int((face["cx"] + 0.40 * face["w"]) * size[0])
    y0, y1 = int((face["cy"] - 0.22 * face["h"]) * size[1]), int((face["cy"] + 0.04 * face["h"]) * size[1])
    series = np.array([f[y0:y1, x0:x1].std() for f in frames])
    if len(series) < 6 or series.max() == 0:
        return {"blinks": 0, "openness": 0.0, "series": []}
    baseline = float(np.percentile(series, 70))
    closed = series < 0.7 * baseline
    blinks, run = 0, 0
    for flag in closed:
        if flag:
            run += 1
        else:
            if 1 <= run <= 6:
                blinks += 1
            run = 0
    if 1 <= run <= 6:
        blinks += 1
    return {"blinks": blinks, "openness": round(baseline, 1), "series": [round(float(v), 1) for v in series]}


def before_after_sheet(columns: list[dict], out: Path, title: str, cell: tuple[int, int] = (288, 512)) -> Path:
    """One picture: a column per shot, BEFORE on top and AFTER under it, with the shot id and what the shot is. columns = [{"label": "s039 corridor", "before": Path, "after": Path}, ...]"""
    gap, head, label_h = 6, 64, 54
    width = len(columns) * (cell[0] + gap)
    height = head + 2 * (cell[1] + label_h)
    sheet = Image.new("RGB", (width, height), (14, 14, 14))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, font=_font(FONT_TITLE, 38), fill=(255, 122, 0))
    for index, column in enumerate(columns):
        x = index * (cell[0] + gap)
        for row, key in enumerate(("before", "after")):
            y = head + row * (cell[1] + label_h)
            path = column.get(key)
            if path and Path(path).exists():
                sheet.paste(Image.open(path).convert("RGB").resize(cell, Image.LANCZOS), (x, y + label_h))
            draw.rectangle((x, y, x + cell[0], y + label_h), fill=(24, 24, 24))
            draw.text((x + 8, y + 6), f"{key.upper()}  {column['label']}", font=_font(FONT_TITLE, 24), fill=(255, 255, 255) if key == "after" else (170, 170, 170))
            draw.text((x + 8, y + 34), column.get(f"{key}_note", ""), font=_font(FONT_TEXT, 14), fill=(200, 200, 200))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def blinks_on_box(clips: list[Path], box) -> dict[str, dict]:
    """Blinks counted ON THE BOX with face landmarks (scripts/box/eyes_ear.py, the eye aspect ratio): the pixel-contrast measure of `blink_series` missed closed eyes (0 counted while the eyelids
    closed), so use this one. `box` = story_produce.Box. Returns {clip name: {blinks, long_closures, longest_closed_s, blinks_per_10s, face_found, ear}}."""
    import json
    remote = "/root/lab_eyes"
    box.run(f"mkdir -p {remote}")
    box.put(ROOT / "scripts" / "box" / "eyes_ear.py", f"{remote}/eyes_ear.py")
    for clip in clips:
        box.put(Path(clip), f"{remote}/{Path(clip).name}")
    names = " ".join(f"{remote}/{Path(c).name}" for c in clips)
    output = box.run(f"cd {remote} && PYTHONPATH=/root/eyelibs /root/ttsenv/bin/python eyes_ear.py {names} 2>/dev/null", timeout=900)
    found = {Path(json.loads(line)["clip"]).name: json.loads(line) for line in output.splitlines() if line.startswith("{")}
    box.run(f"rm -f {remote}/*.mp4")
    return found
