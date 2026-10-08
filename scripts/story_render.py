"""The CPU half of a story video: from the plan, the animated clips and the voice lines, build the finished 9:16 video (720x1280, 30 fps).

    python scripts/story_render.py PLAN.json --cards runs/last_city_v3 --run OUT_run --out story.mp4

Per shot (one file each, then joined, so a bad shot is re-made alone):
  every shot but the choice  its clip <run>/post/<id>.mp4 (face-detail pass: scripts/clip_post.py) or else <run>/clips/<id>.mp4 (raw S2V), filled to 720x1280, with the montage effects of the shot:
                             a slow ZOOM (push-in), a SHAKE (impacts, chases), a white FLASH on a shock. A missing clip falls back to a push-in on the shot's picture and is REPORTED (fallback: true).
  choice                     the two characters' waiting clips side by side, two ORANGE cards "A name" / "B name" and a COUNTDOWN RING that empties
  rewind                     the shot played BACKWARDS with a colour split: the story rewinds to the choice
Words on screen: every spoken line shows ONE WORD AT A TIME (white, bold, black outline, a short GLITCH when the word enters, then perfectly clean), at the time the word is spoken (Whisper word times of
<run>/voices/align.json); a TITLE (the hook: the words TikTok reads) over the first seconds; a TAG at the start of each branch (CASE A / CASE B, ENDING A / ENDING B); a closing question after the last line.
The sound background (scripts/story_sound.py) is ducked under the voices. No film grain is added: it showed on the eyelids. Everything here is Pillow + numpy + ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SIZE = (720, 1280)
K = SIZE[1] / 1024  # everything on screen was designed on a 576x1024 frame: sizes are multiplied by K
FPS = 30
ORANGE = (255, 122, 0)
FONT_BOLD = ROOT / "assets" / "fonts" / "Anton-Regular.ttf"  # open licence (SIL OFL), the same file on the PC and on the box: the montage looks the same wherever it runs
FONT_FALLBACK = Path(r"C:\Windows\Fonts\impact.ttf")
TAIL = {"talk": 0.25, "choice": 0.0, "narration": 0.4, "pov": 0.4, "twist": 2.4, "rewind": 0.5}  # silence after the line, per kind of shot (the twist keeps the screen: the impact rings, the card shows)
CHOICE_MIN_SECONDS = 3.2
RING_SECONDS_BEFORE_END = 0.35
RING_START_SECONDS = 0.5  # the voice says "now you must choose" first, then the ring empties
RING_CENTER_Y = 0.12  # top of the frame: the faces stay free
SUB_PIXELS = round(84 * K)  # the size of the spoken word
SUB_Y = 0.71  # its height in the frame: lower, clear of the faces, above the TikTok caption zone
GLITCH_STRENGTHS = (1.0, 0.7, 0.45, 0.2)  # one glitch frame each (1/30 s), then the clean word: the word hits, then it is calm and easy to read
# the look every shot gets: cold, shadows lifted (the pictures are very dark), dark corners. NO grain: the grain of the first grade showed on the eyelids
GRADE = "eq=contrast=1.05:saturation=0.9:gamma=1.22,unsharp=5:5:0.4:5:5:0.0,vignette=angle=PI/4.5"
SOURCE = (738, 1280)  # a clip after the face-detail pass; a raw clip is scaled to the same height
TAG_SECONDS = 2.6


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


END_CARD_SECONDS = 2.8  # the last shot keeps the screen for the closing question


def shot_seconds(kind: str, voice_seconds: float, last: bool = False) -> float:
    base = voice_seconds + TAIL.get(kind, 0.3) + (END_CARD_SECONDS if last else 0.0)
    return round(max(base, CHOICE_MIN_SECONDS) if kind == "choice" else base, 3)


def word_windows(text: str, seconds: float, lead: float = 0.08) -> list[tuple[str, float, float]]:
    """Each word of the line with the time it is on screen, ESTIMATED from the length of the audio only: the time of a word is proportional to its letters, a word that ends a sentence or a clause
    lasts longer (the voice pauses there). The real times (Whisper) are used when there are some: see aligned_windows."""
    words = re.findall(r"[\w'’\-]+[.,!?;:…]*", text)
    if not words:
        return []
    weights = [len(re.sub(r"\W", "", w)) + 2 + (3 if w[-1] in ".!?…" else 1.5 if w[-1] in ",;:" else 0) for w in words]
    usable = max(0.2, seconds - lead)
    cursor, windows = lead, []
    for word, weight in zip(words, weights):
        length = usable * weight / sum(weights)
        windows.append((word.rstrip(".,!?;:…") or word, round(cursor, 3), round(cursor + length, 3)))
        cursor += length
    return windows


def aligned_windows(text: str, aligned: list[dict] | None, seconds: float) -> list[tuple[str, float, float]]:
    """The words of the SCRIPT with the times they are really spoken: same count as the recognised words -> one to one, else spread between the first and last recognised time."""
    script = [w.rstrip(".,!?;:…") or w for w in re.findall(r"[\w'’\-]+[.,!?;:…]*", text)]
    if not aligned or not script:
        return word_windows(text, seconds)
    starts = [max(0.0, float(w["start"]) - 0.03) for w in aligned]
    ends = [float(w["end"]) for w in aligned]
    if len(aligned) != len(script):
        first, last = starts[0], ends[-1]
        weights = [len(w) + 1 for w in script]
        cursor, starts, ends = first, [], []
        for weight in weights:
            starts.append(cursor)
            cursor += (last - first) * weight / sum(weights)
            ends.append(cursor)
    windows = []
    for index, word in enumerate(script):
        end = starts[index + 1] if index + 1 < len(script) else ends[index] + 0.2
        windows.append((word, round(starts[index], 3), round(max(end, starts[index] + 0.1), 3)))
    return windows


def _font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (FONT_BOLD, FONT_FALLBACK, Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), int(size))
    return ImageFont.load_default()


def word_image(word: str, size: tuple[int, int] = SIZE, y_fraction: float = SUB_Y, pixels: int | None = None) -> Image.Image:
    """One transparent frame with the text: white capitals, thick black outline, centred, shrunk if it is wider than the frame; several lines (separated by a newline) are stacked."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    lines = word.upper().split(chr(10))
    pixels = int(pixels or (SUB_PIXELS if len(lines) == 1 else round(76 * K)))
    font = _font(pixels)
    while max(draw.textlength(line, font=font) for line in lines) > size[0] * 0.9 and pixels > 24:
        pixels -= 4
        font = _font(pixels)
    top = size[1] * y_fraction - (len(lines) - 1) * pixels * 1.1 / 2
    for index, line in enumerate(lines):
        draw.text((size[0] / 2, top + index * pixels * 1.1), line, font=font, fill=(255, 255, 255, 255), anchor="mm", stroke_width=max(3, pixels // 11), stroke_fill=(0, 0, 0, 255))
    return image


def _tint(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """The same shape (same transparency) in another colour."""
    layer = Image.new("RGBA", image.size, color + (0,))
    layer.putalpha(image.getchannel("A"))
    return layer


def glitch_frame(clean: Image.Image, strength: float, rnd: random.Random) -> Image.Image:
    """The word hit by a glitch: a red and a cyan copy pulled apart (RGB split) behind the clean word, and a few horizontal bands torn sideways. strength 1 = hard, 0 = clean.
    The clean word stays on top of the split so it is always readable."""
    if strength <= 0:
        return clean
    bbox = clean.getbbox()
    if bbox is None:
        return clean
    shift = max(1, int(round(16 * K * strength)))
    out = Image.new("RGBA", clean.size, (0, 0, 0, 0))
    for color, dx in (((255, 30, 70), shift), ((0, 230, 255), -shift)):
        out.alpha_composite(ImageChops.offset(_tint(clean, color), dx, 0))
    out.alpha_composite(clean)
    for _ in range(2 + int(2 * strength)):
        band_h = rnd.randint(int(6 * K), int(16 * K))
        y0 = rnd.randint(bbox[1], max(bbox[1], bbox[3] - band_h))
        band = out.crop((0, y0, out.width, y0 + band_h))
        out.paste((0, 0, 0, 0), (0, y0, out.width, y0 + band_h))
        out.alpha_composite(ImageChops.offset(band, rnd.choice((-1, 1)) * int(rnd.randint(8, 26) * K * max(1, int(round(strength * 1.5)))), 0), (0, y0))
    return out


def glitch_sequence(clean: Image.Image, seed: str) -> list[Image.Image]:
    """The frames of a word's entrance: glitch frames (strong to light) and, last, the clean word."""
    rnd = random.Random(seed)
    return [glitch_frame(clean, strength, rnd) for strength in GLITCH_STRENGTHS] + [clean]


# ---------------------------------------------------------------- camera effects of a shot (zoom / shake / flash), with sub-pixel crops
def ease(progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return progress * progress * (3 - 2 * progress)


SHAKE_STYLE = "impact"  # "impact" = a short irregular jolt that dies out (default), "legacy" = the constant regular wobble of the first versions (too much, too regular: kept only to compare)


def shake_offset(t: float, shake: float, phase: float, style: str = "impact") -> tuple[float, float]:
    """The camera shake at time t, in pixels of the output frame. 'impact': a jolt of a few pixels that dies out in about a second, made of three frequencies that do not repeat (so it never
    looks like a regular wobble). 'legacy': the old constant 7-9 Hz oscillation."""
    if shake <= 0:
        return 0.0, 0.0
    if style == "legacy":
        amplitude = shake * 9.0 * K * (0.35 + 0.65 * math.exp(-t / 0.7))
        return amplitude * math.sin(2 * math.pi * 7.3 * t + phase), amplitude * math.sin(2 * math.pi * 9.1 * t + phase * 1.7)
    amplitude = shake * 3.5 * K * math.exp(-t / 0.45)
    dx = sum(w * math.sin(2 * math.pi * f * t + phase * k) for k, (f, w) in enumerate(((4.1, 0.5), (7.7, 0.3), (12.9, 0.2)), start=1))
    dy = sum(w * math.sin(2 * math.pi * f * t + phase * (k + 2.3)) for k, (f, w) in enumerate(((3.3, 0.5), (8.9, 0.3), (14.1, 0.2)), start=1))
    return amplitude * dx, amplitude * dy


def fx_box(t: float, seconds: float, fx: dict, source: tuple[int, int] = SOURCE, out: tuple[int, int] = SIZE, phase: float = 0.0, style: str | None = None) -> tuple[float, float, float, float]:
    """Crop box (x0, y0, x1, y1), in pixels of the source frame, at time t: the full output-sized window of the source, shrunk by the zoom (a push-in) and moved by the shake."""
    zoom = 1.0 + float(fx.get("zoom", 0.0)) * ease(t / max(seconds, 1e-6))
    width, height = out[0] / zoom, out[1] / zoom
    dx, dy = shake_offset(t, float(fx.get("shake", 0.0)), phase, style or SHAKE_STYLE)
    cx, cy = source[0] / 2 + dx, source[1] / 2 + dy
    x0 = min(max(0.0, cx - width / 2), source[0] - width)
    y0 = min(max(0.0, cy - height / 2), source[1] - height)
    return (x0, y0, x0 + width, y0 + height)


FLASH_ALPHAS = (0.75, 0.5, 0.3, 0.15)


def apply_flash(frame: Image.Image, index: int) -> Image.Image:
    if index >= len(FLASH_ALPHAS):
        return frame
    return Image.blend(frame, Image.new("RGB", frame.size, (255, 255, 255)), FLASH_ALPHAS[index])


def kb_box(t: float, seconds: float, size: tuple[int, int] = SIZE, zoom_end: float = 1.14, drift: tuple[float, float] = (0.0, -0.02)) -> tuple[float, float, float, float]:
    """Crop box (x0, y0, x1, y1) of the push-in at time t: the zoom grows smoothly from 1 to zoom_end, the centre drifts a little (the fallback of a missing clip)."""
    eased = ease(t / max(seconds, 1e-6))
    zoom = 1 + (zoom_end - 1) * eased
    w, h = size[0] / zoom, size[1] / zoom
    cx = size[0] / 2 + drift[0] * size[0] * eased
    cy = size[1] / 2 + drift[1] * size[1] * eased
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def cover(image: Image.Image, size: tuple[int, int] = SIZE) -> Image.Image:
    """Scale and crop to fill the frame (the card is already 9:16, a portrait may not be)."""
    scale = max(size[0] / image.width, size[1] / image.height)
    resized = image.resize((math.ceil(image.width * scale), math.ceil(image.height * scale)), Image.LANCZOS)
    left, top = (resized.width - size[0]) // 2, (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def choice_frame(t: float, seconds: float, portraits: list[Image.Image], names: list[str], size: tuple[int, int] = SIZE, background: Image.Image | None = None) -> Image.Image:
    """The choice card at time t: the characters side by side (`portraits` as stills, or `background`, an already composed frame of moving clips), the countdown ring, the orange A / B cards."""
    w, h = size
    k = h / 1024
    if background is not None:
        frame = background.convert("RGB").resize(size)
    else:
        frame = Image.new("RGB", size, (0, 0, 0))
        half = w // len(portraits)
        for index, portrait in enumerate(portraits):
            frame.paste(cover(portrait, (half, h)), (index * half, 0))
    ramp = np.clip((np.arange(h) - h * 0.55) / (h * 0.25), 0, 1) * 170  # a soft dark gradient under the cards, no hard edge
    shade = Image.fromarray(np.dstack([np.zeros((h, w, 3), dtype=np.uint8), np.repeat(ramp[:, None], w, axis=1).astype(np.uint8)]), "RGBA")
    frame = Image.alpha_composite(frame.convert("RGBA"), shade)
    ring_start, ring_end = RING_START_SECONDS, seconds - RING_SECONDS_BEFORE_END
    remaining = 1.0 - min(1.0, max(0.0, (t - ring_start) / max(ring_end - ring_start, 1e-6)))
    scale = 3  # supersample the ring: smooth edges
    ring_h = int(260 * k)
    ring = Image.new("RGBA", (w * scale, ring_h * scale), (0, 0, 0, 0))
    cx, cy, radius = w * scale // 2, ring_h * scale // 2, int(78 * k * scale)
    d = ImageDraw.Draw(ring)
    d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(255, 255, 255, 70), width=int(10 * k * scale))
    if remaining > 0.001:
        d.arc((cx - radius, cy - radius, cx + radius, cy + radius), start=-90, end=-90 + 360 * remaining, fill=ORANGE + (255,), width=int(10 * k * scale))
    ring = ring.resize((w, ring_h), Image.LANCZOS)
    frame.alpha_composite(ring, (0, int(h * RING_CENTER_Y) - ring_h // 2))
    draw = ImageDraw.Draw(frame)
    draw.text((w / 2, int(h * RING_CENTER_Y)), str(max(1, math.ceil(remaining * 3))) if remaining > 0.001 else "0", font=_font(84 * k), fill=(255, 255, 255, 255), anchor="mm", stroke_width=5, stroke_fill=(0, 0, 0, 255))
    for index, name in enumerate(names):
        x0 = int(28 * k) + index * (w // 2)
        card = (x0, int(h * 0.80), x0 + w // 2 - int(56 * k), int(h * 0.80) + int(118 * k))
        draw.rounded_rectangle(card, radius=int(22 * k), fill=ORANGE + (255,))
        draw.text((card[0] + 44 * k, (card[1] + card[3]) / 2), "AB"[index], font=_font(72 * k), fill=(255, 255, 255, 255), anchor="mm", stroke_width=3, stroke_fill=(120, 50, 0, 255))
        label = name.upper()
        size_px = 46 * k
        while draw.textlength(label, font=_font(size_px)) > card[2] - card[0] - 100 * k and size_px > 20:
            size_px -= 4
        draw.text(((card[0] + 90 * k + card[2]) / 2, (card[1] + card[3]) / 2), label, font=_font(size_px), fill=(255, 255, 255, 255), anchor="mm", stroke_width=2, stroke_fill=(120, 50, 0, 255))
    return frame.convert("RGB")


def _encode_frames(frames, out: Path, seconds: float, audio: Path | None) -> None:
    """Pipe raw RGB frames to ffmpeg (h264 yuv420p) and mux the audio, padded with silence to the exact length of the shot."""
    command = [mt.ffmpeg_binary(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "-"]
    if audio:
        command += ["-i", str(audio), "-af", f"aresample=48000,apad=whole_dur={seconds:.3f}"]
    else:
        command += ["-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={seconds:.3f}"]
    command += ["-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ac", "2", str(out)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed encoding {out.name}")


def kenburns_frames(card: Image.Image, seconds: float, **kwargs):
    base = cover(card.convert("RGB"))
    for index in range(int(round(seconds * FPS))):
        yield base.resize(SIZE, Image.BICUBIC, box=kb_box(index / FPS, seconds, **kwargs))


def decode_frames(clip: Path, source: tuple[int, int] = SOURCE):
    """The frames of a clip at 30 fps and at the height of SOURCE (a post-processed clip already is; a raw 16 fps clip is scaled and its frames are repeated): PIL RGB pictures, one at a time."""
    process = subprocess.Popen([mt.ffmpeg_binary(), "-v", "error", "-i", str(clip), "-vf", f"fps={FPS},scale={source[0]}:{source[1]}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               stdout=subprocess.PIPE)
    size = source[0] * source[1] * 3
    try:
        while True:
            chunk = process.stdout.read(size)
            if len(chunk) < size:
                break
            yield Image.frombytes("RGB", source, chunk)
    finally:
        process.stdout.close()
        process.wait()


def clip_frames(clip: Path, seconds: float, fx: dict, rewind: bool = False, seed: int = 0):
    """The `seconds` of a shot from its clip: the effects of the shot applied (zoom, shake, flash), the last frame held if the clip is short; a rewind plays it backwards with a colour split."""
    wanted = int(round(seconds * FPS))
    phase = (seed % 97) / 97 * math.tau
    frames = list(decode_frames(clip)) if rewind else None
    if rewind and frames:
        frames = frames[::-1]
    source = iter(frames) if frames is not None else decode_frames(clip)
    last = None
    for index in range(wanted):
        frame = next(source, None)
        if frame is not None:
            last = frame
        if last is None:
            raise RuntimeError(f"no frame in {clip}")
        picture = last.resize(SIZE, Image.BICUBIC, box=fx_box(index / FPS, seconds, fx, source=last.size, phase=phase))
        if rewind:
            r, g, b = picture.split()
            shift = int(round((5 + 3 * math.sin(index * 0.9)) * K))
            picture = Image.merge("RGB", (ImageChops.offset(r, shift, 0), g, ImageChops.offset(b, -shift, 0)))
            picture = Image.blend(picture, Image.new("RGB", SIZE, (0, 60, 90)), 0.22)
        if fx.get("flash"):
            picture = apply_flash(picture, index)
        yield picture


def window_frames(window: tuple, index: int) -> list[tuple[float, Image.Image]]:
    """(start time, picture) of every change of one on-screen text: the glitch frames of its entrance, then the clean text until its end."""
    text, start, end = window[0], window[1], window[2]
    options = window[3] if len(window) > 3 else {}
    clean = word_image(text, y_fraction=options.get("y", SUB_Y), pixels=options.get("pixels"))
    images = glitch_sequence(clean, f"{text}-{index}") if options.get("glitch", True) else [clean]
    step = 1.0 / FPS
    return [(start + k * step, image) for k, image in enumerate(images) if start + k * step < end]


def subtitle_overlay(windows: list, seconds: float, work: Path) -> Path | None:
    """The on-screen texts of a shot as a list for ffmpeg's concat demuxer: one transparent picture per change (texts can overlap in time: a title above the spoken word), a blank one in the gaps.
    A window is (text, start, end) or (text, start, end, options): options = {"y": fraction of the height, "pixels": font size, "glitch": False}. Every text ENTERS with a glitch of a few frames and is then clean."""
    if not windows:
        return None
    work.mkdir(parents=True, exist_ok=True)
    tracks = [window_frames(window, index) for index, window in enumerate(windows)]
    ends = [window[2] for window in windows]
    starts = [window[1] for window in windows]
    changes = sorted({0.0, *(t for track in tracks for t, _ in track), *ends})
    changes = [t for t in changes if t < seconds + 0.2]
    blank_path = work / "blank.png"
    Image.new("RGBA", SIZE, (0, 0, 0, 0)).save(blank_path)
    lines, written = [], {}
    for position, t in enumerate(changes):
        stack = []
        for index, track in enumerate(tracks):
            if starts[index] <= t + 1e-6 < ends[index]:
                shown = [(s, image) for s, image in track if s <= t + 1e-6]
                if shown:
                    stack.append((index, len(shown) - 1))
        key = tuple(stack)
        if key not in written:
            if not stack:
                written[key] = blank_path
            else:
                composed = Image.new("RGBA", SIZE, (0, 0, 0, 0))
                for index, frame_index in stack:
                    composed.alpha_composite(tracks[index][frame_index][1])
                written[key] = work / f"c{len(written):04d}.png"
                composed.save(written[key])
        duration = (changes[position + 1] - t) if position + 1 < len(changes) else 0.2
        lines += [f"file '{written[key].resolve().as_posix()}'", f"duration {max(duration, 0.001):.4f}"]
    lines.append(f"file '{blank_path.resolve().as_posix()}'")  # the concat demuxer ignores the duration of the last entry
    listing = work / "words.ffconcat"
    listing.write_text("\n".join(lines), encoding="utf-8")
    return listing


def burn_subtitles(video: Path, listing: Path | None, out: Path, grade: str | None = GRADE) -> None:
    """Grade the shot, then lay the on-screen texts on top (the text is never graded)."""
    chain = (grade + ",") if grade else ""
    if listing is None:
        graph = f"[0:v]{chain}format=yuv420p[v]"
        inputs = ["-i", str(video)]
    else:
        graph = f"[0:v]{chain}format=yuv420p[g];[1:v]fps={FPS},format=rgba[sub];[g][sub]overlay=0:0:eof_action=pass,format=yuv420p[v]"
        inputs = ["-i", str(video), "-f", "concat", "-safe", "0", "-i", str(listing)]
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-maxrate", "10M",
                    "-bufsize", "20M", "-c:a", "copy", "-shortest", str(out)], check=True)


def choice_clip_frames(clips: list[Path], names: list[str], seconds: float):
    """Two characters waiting, side by side (the centre of each clip), under the choice card."""
    wanted = int(round(seconds * FPS))
    tracks = [list(decode_frames(clip)) for clip in clips]
    half = SIZE[0] // len(clips)
    for index in range(wanted):
        frame = Image.new("RGB", SIZE)
        for k, track in enumerate(tracks):
            picture = track[min(index, len(track) - 1)].resize(SIZE, Image.BICUBIC, box=((SOURCE[0] - SIZE[0]) / 2, 0, (SOURCE[0] + SIZE[0]) / 2, SIZE[1]))
            left = (SIZE[0] - half) // 2
            frame.paste(picture.crop((left, 0, left + half, SIZE[1])), (k * half, 0))
        yield choice_frame(index / FPS, seconds, [], names, background=frame)


def shot_starts(timings: dict) -> dict:
    cursor, starts = 0.0, {}
    for sid, info in timings.items():
        starts[sid] = round(cursor, 3)
        cursor += info["seconds"]
    return starts


def choice_ids(plan: dict, shot: dict) -> list[str]:
    """The two characters of the choice card: those the shot lists, else the two main characters of the story (a model may leave the list empty: a choice is always between the two of them)."""
    ids = [i for i in shot.get("in_shot", []) if i in {c["id"] for c in plan["characters"]}][:2]
    return ids if len(ids) == 2 else [c["id"] for c in plan["characters"][:2]]


def find_clip(run: Path, sid: str) -> Path | None:
    """The best clip of a shot: after the face-detail pass if there is one, else the raw clip."""
    for folder in ("post", "clips"):
        path = run / folder / f"{sid}.mp4"
        if path.exists():
            return path
    return None


def overlay_windows(plan: dict, shot: dict, index: int, seconds: float, spoken: float, words: list) -> list:
    """Everything written on a shot: the spoken words, the hook title (first shot), the tag of a branch, the closing card (last shot)."""
    windows = list(words)
    if index == 0 and plan.get("title_overlay"):
        title = plan["title_overlay"]
        windows.append((title["text"], title.get("start", 0.15), min(title.get("end", 3.8), seconds), {"y": title.get("y", 0.15), "pixels": round(title.get("pixels", 62) * K)}))
    if shot.get("tag"):
        windows.append((shot["tag"], 0.1, min(TAG_SECONDS, seconds - 0.05), {"y": 0.075, "pixels": round(40 * K)}))
    if index == len(plan["shots"]) - 1:
        if plan.get("end_card"):
            windows.append((plan["end_card"], round(spoken + 0.35, 3), round(seconds - 0.05, 3), {"y": 0.42, "pixels": round(80 * K)}))
        if plan.get("end_card_small"):
            windows.append((plan["end_card_small"], round(spoken + 0.8, 3), round(seconds - 0.05, 3), {"y": 0.82, "pixels": round(46 * K)}))
    return windows


def render_story(plan: dict, cards_dir: Path, run: Path, out: Path, sound: bool = True) -> dict:
    cards = json.loads((cards_dir / "cards.json").read_text(encoding="utf-8"))
    voices_dir = run / "voices"
    aligned = json.loads((voices_dir / "align.json").read_text(encoding="utf-8")) if (voices_dir / "align.json").exists() else {}
    work = run / "render"
    work.mkdir(parents=True, exist_ok=True)
    characters = {c["id"]: c for c in plan["characters"]}
    timings, parts = {}, []
    for index, shot in enumerate(plan["shots"]):
        started = time.time()
        sid, kind = shot["id"], shot["kind"]
        voice = voices_dir / f"{sid}.wav"
        seconds = shot_seconds(kind, wav_seconds(voice), last=index == len(plan["shots"]) - 1)
        raw, burned = work / f"{sid}_raw.mp4", work / f"{sid}.mp4"
        fallback = False
        fx = shot.get("fx") or {}
        if kind == "choice":
            ids = choice_ids(plan, shot)
            names = [characters[i]["name"] for i in ids]
            waiting = [find_clip(run, f"{sid}_{i}") for i in ids]
            if all(waiting):
                _encode_frames(choice_clip_frames(waiting, names, seconds), raw, seconds, voice)
            else:  # no waiting clips: the portraits stand still under the card, and say so
                fallback = True
                portraits = [Image.open(ROOT / (cards["characters"][i].get("closeup") or cards["characters"][i]["portrait"])["file"]) for i in ids]
                _encode_frames((choice_frame(k / FPS, seconds, portraits, names) for k in range(int(round(seconds * FPS)))), raw, seconds, voice)
                print(f"{sid}: NO WAITING CLIPS, still portraits", flush=True)
        elif find_clip(run, sid):
            _encode_frames(clip_frames(find_clip(run, sid), seconds, fx, rewind=kind == "rewind", seed=index), raw, seconds, voice)
        else:  # no clip: a push-in on the picture of the shot, and say so
            fallback = True
            picture = run / "stills" / f"{sid}.png"
            if not picture.exists():
                entry = cards["characters"][shot["speaker"]]
                picture = ROOT / (entry.get("closeup") or entry["portrait"])["file"] if kind == "talk" else ROOT / cards["locations"][shot["location"]]["file"]
            _encode_frames(kenburns_frames(Image.open(picture), seconds), raw, seconds, voice)
            print(f"{sid}: NO CLIP, push-in on {picture.name}", flush=True)
        spoken = seconds - TAIL.get(kind, 0.3) - (END_CARD_SECONDS if index == len(plan["shots"]) - 1 else 0.0)
        words = aligned_windows(shot["text"], aligned.get(sid), spoken) if aligned.get(sid) else word_windows(shot["text"], spoken if kind != "choice" else 1.3)
        burn_subtitles(raw, subtitle_overlay(overlay_windows(plan, shot, index, seconds, spoken, words), seconds, work / f"{sid}_words"), burned)
        parts.append(burned)
        timings[sid] = {"kind": kind, "seconds": seconds, "voice_seconds": round(wav_seconds(voice), 3), "render_seconds": round(time.time() - started, 1),
                        "words_timed_by": "whisper" if aligned.get(sid) else "estimate", "fallback": fallback}
        print(f"{sid} {kind}: {seconds:.2f} s", flush=True)
    listing = work / "all.txt"
    listing.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in parts), encoding="utf-8")
    joined = work / "joined.mp4"
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)], check=True)
    total = round(sum(t["seconds"] for t in timings.values()), 2)
    if sound:
        import story_sound as so
        track = so.build_soundtrack(total, sound_cues(plan, timings), work / "soundtrack.wav")
        so.mix_with_voices(joined, track, out)
    else:
        joined.replace(out)
    return {"file": str(out), "shots": timings, "total_seconds": total, "sound": sound, "fallback_shots": [s for s, t in timings.items() if t["fallback"]]}


def sound_cues(plan: dict, timings: dict) -> dict:
    """When the sound background must DO something: the heartbeat of the choice, the impact on each twist (and a bright pad + the end of the rain on a GOOD ending), the reverse swell into a rewind,
    thunder on the big impacts."""
    starts = shot_starts(timings)
    cues: dict = {"choice": None, "twists": [], "rewind": None, "uplift": None, "rain_stop": None, "thunder": []}
    last_thunder = -99.0
    for shot in plan["shots"]:
        begin, length = starts[shot["id"]], timings[shot["id"]]["seconds"]
        if shot["kind"] == "choice":
            cues["choice"] = [begin + RING_START_SECONDS, begin + length - RING_SECONDS_BEFORE_END]
        elif shot["kind"] == "twist":
            cues["twists"].append(begin)
            if shot.get("ending") == "good":
                cues["uplift"], cues["rain_stop"] = begin, begin
        elif shot["kind"] == "rewind":
            cues["rewind"] = begin
        elif (shot.get("fx") or {}).get("shake", 0) >= 0.8 and begin > 3.0 and begin - last_thunder > 14.0:
            cues["thunder"].append(begin)
            last_thunder = begin
    return cues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="the production folder: voices/, clips/ or post/, stills/ inside")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-sound", action="store_true")
    args = parser.parse_args()
    result = render_story(json.loads(args.plan.read_text(encoding="utf-8")), args.cards.resolve(), args.run, args.out, sound=not args.no_sound)
    print("RESULT " + json.dumps(result))  # one line: the remote render is read back from the log


if __name__ == "__main__":
    main()
