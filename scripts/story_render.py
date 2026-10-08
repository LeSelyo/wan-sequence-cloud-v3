"""The CPU half of a story video: from the plan, the cards, the voice lines and the talking clips, build the finished 9:16 video.

    python scripts/story_render.py PLAN.json --cards runs/test20_en_night --voices voices_dir --talk talk_dir --out story.mp4 [--seed 1]

Per shot (one file each, then joined, so a bad shot is re-made alone):
  narration / pov / twist  the CARD of the place with a slow push-in (Ken-Burns) -- the fallback while no image-to-video clip exists (a clip named <shot id>.mp4 in --motion replaces it)
  talk                     the S2V clip of the character (scripts/s2v_talk.py) in <talk>/<shot id>.mp4
  choice                   the two portraits side by side, two ORANGE cards "A name" / "B name" and a COUNTDOWN RING that empties
Every spoken line shows ONE WORD AT A TIME (white, bold, black outline), timed on the line's audio. A quiet low rumble sits under everything (--no-bed to drop it).
Voice lines are <voices>/<shot id>.wav (scripts/box/story_voices.py). Everything here is Pillow + numpy + ffmpeg: light on the PC.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402

SIZE = (576, 1024)
FPS = 30
ORANGE = (255, 122, 0)
FONT_BOLD = Path(r"C:\Windows\Fonts\impact.ttf")
FONT_FALLBACK = Path(r"C:\Windows\Fonts\arialbd.ttf")
TAIL = {"talk": 0.15, "choice": 0.0, "narration": 0.2, "pov": 0.2, "twist": 0.7}  # silence after the line, per kind of shot
CHOICE_MIN_SECONDS = 3.2
RING_SECONDS_BEFORE_END = 0.35
RING_START_SECONDS = 0.5  # the voice says "now you must choose" first, then the ring empties
RING_CENTER_Y = 0.12  # top of the frame: the faces stay free


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def shot_seconds(kind: str, voice_seconds: float) -> float:
    base = voice_seconds + TAIL.get(kind, 0.3)
    return round(max(base, CHOICE_MIN_SECONDS) if kind == "choice" else base, 3)


def word_windows(text: str, seconds: float, lead: float = 0.08) -> list[tuple[str, float, float]]:
    """Each word of the line with the time it is on screen. The time of a word is proportional to its letters, a word that ends a sentence or a clause lasts longer (the voice pauses there).
    Timed on the length of the audio only: a later version can use forced alignment (faster-whisper) for exact times."""
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


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if FONT_BOLD.exists() else FONT_FALLBACK), size)


def word_image(word: str, size: tuple[int, int] = SIZE, y_fraction: float = 0.64) -> Image.Image:
    """One transparent frame with the word: white capitals, thick black outline, centred, shrunk if the word is wider than the frame."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    text = word.upper()
    pixels = 118
    font = _font(pixels)
    while draw.textlength(text, font=font) > size[0] * 0.9 and pixels > 30:
        pixels -= 6
        font = _font(pixels)
    draw.text((size[0] / 2, size[1] * y_fraction), text, font=font, fill=(255, 255, 255, 255), anchor="mm", stroke_width=max(4, pixels // 11), stroke_fill=(0, 0, 0, 255))
    return image


def kb_box(t: float, seconds: float, size: tuple[int, int] = SIZE, zoom_end: float = 1.14, drift: tuple[float, float] = (0.0, -0.02)) -> tuple[float, float, float, float]:
    """Crop box (x0, y0, x1, y1) of the push-in at time t: the zoom grows smoothly from 1 to zoom_end, the centre drifts a little."""
    progress = min(1.0, max(0.0, t / max(seconds, 1e-6)))
    eased = progress * progress * (3 - 2 * progress)
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


def choice_frame(t: float, seconds: float, portraits: list[Image.Image], names: list[str], size: tuple[int, int] = SIZE) -> Image.Image:
    """The choice card at time t: the portraits side by side (darkened), the countdown ring, the orange A / B cards at the bottom."""
    w, h = size
    frame = Image.new("RGB", size, (0, 0, 0))
    half = w // len(portraits)
    for index, portrait in enumerate(portraits):
        panel = cover(portrait, (half, h))
        frame.paste(panel, (index * half, 0))
    ramp = np.clip((np.arange(h) - h * 0.55) / (h * 0.25), 0, 1) * 170  # a soft dark gradient under the cards, no hard edge
    shade = Image.fromarray(np.dstack([np.zeros((h, w, 3), dtype=np.uint8), np.repeat(ramp[:, None], w, axis=1).astype(np.uint8)]), "RGBA")
    frame = Image.alpha_composite(frame.convert("RGBA"), shade)
    ring_start, ring_end = RING_START_SECONDS, seconds - RING_SECONDS_BEFORE_END
    remaining = 1.0 - min(1.0, max(0.0, (t - ring_start) / max(ring_end - ring_start, 1e-6)))
    scale = 3  # supersample the ring: smooth edges
    ring = Image.new("RGBA", (w * scale, 260 * scale), (0, 0, 0, 0))
    cx, cy, radius = w * scale // 2, 130 * scale, 78 * scale
    d = ImageDraw.Draw(ring)
    d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(255, 255, 255, 70), width=10 * scale)
    if remaining > 0.001:
        d.arc((cx - radius, cy - radius, cx + radius, cy + radius), start=-90, end=-90 + 360 * remaining, fill=ORANGE + (255,), width=10 * scale)
    ring = ring.resize((w, 260), Image.LANCZOS)
    frame.alpha_composite(ring, (0, int(h * RING_CENTER_Y) - 130))
    draw = ImageDraw.Draw(frame)
    draw.text((w / 2, int(h * RING_CENTER_Y)), str(max(1, math.ceil(remaining * 3))) if remaining > 0.001 else "0", font=_font(84), fill=(255, 255, 255, 255), anchor="mm", stroke_width=5, stroke_fill=(0, 0, 0, 255))
    for index, name in enumerate(names):
        x0 = 28 + index * (w // 2)
        card = (x0, int(h * 0.80), x0 + w // 2 - 56, int(h * 0.80) + 118)
        draw.rounded_rectangle(card, radius=22, fill=ORANGE + (255,))
        draw.text((card[0] + 44, (card[1] + card[3]) / 2), "AB"[index], font=_font(72), fill=(255, 255, 255, 255), anchor="mm", stroke_width=3, stroke_fill=(120, 50, 0, 255))
        label = name.upper()
        size_px = 46
        while draw.textlength(label, font=_font(size_px)) > card[2] - card[0] - 100 and size_px > 20:
            size_px -= 4
        draw.text(((card[0] + 90 + card[2]) / 2, (card[1] + card[3]) / 2), label, font=_font(size_px), fill=(255, 255, 255, 255), anchor="mm", stroke_width=2, stroke_fill=(120, 50, 0, 255))
    return frame.convert("RGB")


def _encode_frames(frames, out: Path, seconds: float, audio: Path | None) -> None:
    """Pipe raw RGB frames to ffmpeg (h264 yuv420p) and mux the audio, padded with silence to the exact length of the shot."""
    command = [mt.ffmpeg_binary(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "-"]
    if audio:
        command += ["-i", str(audio), "-af", f"aresample=48000,apad=whole_dur={seconds:.3f}"]
    else:
        command += ["-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={seconds:.3f}"]
    command += ["-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ac", "2", str(out)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed encoding {out.name}")


def kenburns_frames(card: Image.Image, seconds: float, **kwargs):
    base = cover(card.convert("RGB"))
    for index in range(int(round(seconds * FPS))):
        box = kb_box(index / FPS, seconds, **kwargs)
        yield base.resize(SIZE, Image.BICUBIC, box=box)


def choice_frames(portraits: list[Image.Image], names: list[str], seconds: float):
    for index in range(int(round(seconds * FPS))):
        yield choice_frame(index / FPS, seconds, portraits, names)


def subtitle_overlay(windows: list[tuple[str, float, float]], seconds: float, work: Path) -> Path | None:
    """The one-word subtitles of a shot as a list for ffmpeg's concat demuxer (an image per word, a transparent image in the gaps); returns the list file."""
    if not windows:
        return None
    work.mkdir(parents=True, exist_ok=True)
    blank = work / "blank.png"
    Image.new("RGBA", SIZE, (0, 0, 0, 0)).save(blank)
    lines, cursor = [], 0.0
    for index, (word, start, end) in enumerate(windows):
        if start > cursor + 1e-3:
            lines += [f"file '{blank.resolve().as_posix()}'", f"duration {start - cursor:.3f}"]
        path = work / f"w{index:02d}.png"
        word_image(word).save(path)
        lines += [f"file '{path.resolve().as_posix()}'", f"duration {end - start:.3f}"]
        cursor = end
    lines += [f"file '{blank.resolve().as_posix()}'", f"duration {max(0.04, seconds - cursor + 0.2):.3f}"]
    lines.append(f"file '{blank.resolve().as_posix()}'")  # the concat demuxer ignores the duration of the last entry
    listing = work / "words.ffconcat"
    listing.write_text("\n".join(lines), encoding="utf-8")
    return listing


def burn_subtitles(video: Path, listing: Path | None, out: Path) -> None:
    if listing is None:
        video.replace(out)
        return
    command = [mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(video), "-f", "concat", "-safe", "0", "-i", str(listing), "-filter_complex",
               f"[1:v]fps={FPS},format=rgba[sub];[0:v][sub]overlay=0:0:eof_action=pass,format=yuv420p[v]", "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "medium",
               "-crf", "18", "-c:a", "copy", "-shortest", str(out)]
    subprocess.run(command, check=True)


def talk_clip_to_frame_size(clip: Path, out: Path, seconds: float, audio: Path) -> None:
    """The S2V clip (480x832) filled to 576x1024 (scale then crop the 15 px of width), its own audio replaced by the clean voice line, padded/cut to the shot length."""
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(clip), "-i", str(audio), "-filter_complex",
                    f"[0:v]scale=-2:{SIZE[1]}:flags=lanczos,crop={SIZE[0]}:{SIZE[1]},fps={FPS},tpad=stop_mode=clone:stop_duration=2,trim=duration={seconds:.3f},setpts=PTS-STARTPTS[v];"
                    f"[1:a]aresample=48000,apad=whole_dur={seconds:.3f}[a]", "-map", "[v]", "-map", "[a]", "-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ac", "2", str(out)], check=True)


def add_bed(video: Path, out: Path, level_db: float = -26.0) -> None:
    """A low rumble (brown noise, low-passed) under the voices, a slow fade in/out. Makes the silence between words feel like the world is still there."""
    seconds = float(subprocess.run([mt.ffmpeg_binary(), "-hide_banner", "-i", str(video)], capture_output=True, text=True).stderr.split("Duration: ")[1].split(",")[0].split(":")[2]) \
        + 60 * int(subprocess.run([mt.ffmpeg_binary(), "-hide_banner", "-i", str(video)], capture_output=True, text=True).stderr.split("Duration: ")[1].split(",")[0].split(":")[1])
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(video), "-f", "lavfi", "-i", f"anoisesrc=color=brown:amplitude=0.5:r=48000:d={seconds:.3f}", "-filter_complex",
                    f"[1:a]lowpass=f=420,volume={level_db}dB,afade=t=in:d=0.8,afade=t=out:st={max(0.0, seconds - 1.2):.3f}:d=1.2,aformat=channel_layouts=stereo[bed];"
                    f"[0:a][bed]amix=inputs=2:duration=first:normalize=0[a]", "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", str(out)], check=True)


def render_story(plan: dict, cards_dir: Path, voices_dir: Path, talk_dir: Path, out: Path, motion_dir: Path | None = None, bed: bool = True) -> dict:
    cards = json.loads((cards_dir / "cards.json").read_text(encoding="utf-8"))
    root = cards_dir.parents[3]  # results/story_trend/runs/<run> -> repository root (card paths in cards.json are relative to it)
    work = out.parent / f"{out.stem}_work"
    work.mkdir(parents=True, exist_ok=True)
    characters = {c["id"]: c for c in plan["characters"]}
    timings, parts = {}, []
    for shot in plan["shots"]:
        started = time.time()
        sid, kind = shot["id"], shot["kind"]
        voice = voices_dir / f"{sid}.wav"
        seconds = shot_seconds(kind, wav_seconds(voice))
        raw, burned = work / f"{sid}_raw.mp4", work / f"{sid}.mp4"
        location = cards["locations"][shot["location"]]
        place = Image.open(root / location["file"])
        if kind == "talk" and (talk_dir / f"{sid}.mp4").exists():
            talk_clip_to_frame_size(talk_dir / f"{sid}.mp4", raw, seconds, voice)
        elif (motion_dir / f"{sid}.mp4" if motion_dir else Path("/nonexistent")).exists():
            talk_clip_to_frame_size(motion_dir / f"{sid}.mp4", raw, seconds, voice)
        elif kind == "choice":
            ids = shot["in_shot"][:2]
            portraits = [Image.open(root / cards["characters"][i]["portrait"]["file"]) for i in ids]
            _encode_frames(choice_frames(portraits, [characters[i]["name"] for i in ids], seconds), raw, seconds, voice)
        else:
            card = Image.open(root / cards["characters"][shot["in_shot"][0]]["portrait"]["file"]) if kind == "talk" else place
            _encode_frames(kenburns_frames(card, seconds), raw, seconds, voice)
        burn_subtitles(raw, subtitle_overlay(word_windows(shot["text"], seconds - TAIL.get(kind, 0.3) if kind != "choice" else 1.3), seconds, work / f"{sid}_words"), burned)
        parts.append(burned)
        timings[sid] = {"kind": kind, "seconds": seconds, "voice_seconds": round(wav_seconds(voice), 3), "render_seconds": round(time.time() - started, 1)}
        print(f"{sid} {kind}: {seconds:.2f} s", flush=True)
    listing = work / "all.txt"
    listing.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in parts), encoding="utf-8")
    joined = work / "joined.mp4"
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)], check=True)
    if bed:
        add_bed(joined, out)
    else:
        joined.replace(out)
    return {"file": str(out), "shots": timings, "total_seconds": round(sum(t["seconds"] for t in timings.values()), 2), "bed": bed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--talk", type=Path, required=True)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-bed", action="store_true")
    args = parser.parse_args()
    result = render_story(json.loads(args.plan.read_text(encoding="utf-8")), args.cards.resolve(), args.voices, args.talk, args.out, args.motion, bed=not args.no_bed)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
