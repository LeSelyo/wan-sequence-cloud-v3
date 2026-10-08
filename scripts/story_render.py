"""The CPU half of a story video: from the plan, the animated clips and the voice lines, build the finished 9:16 video.

    python scripts/story_render.py PLAN.json --cards runs/test20_en_night --run OUT_run --out story.mp4

Per shot (one file each, then joined, so a bad shot is re-made alone):
  every shot but the choice  its S2V clip <run>/clips/<shot id>.mp4 (scripts/s2v_talk.py), filled to 576x1024, graded (cold, a little grain, vignette). A missing clip falls back to a slow
                             push-in on the shot's picture and is REPORTED (fallback: true), never silently.
  choice                     the two characters' waiting clips side by side (each one half of the frame), two ORANGE cards "A name" / "B name" and a COUNTDOWN RING that empties
Every spoken line shows ONE WORD AT A TIME (white, bold, black outline), at the time the word is spoken (Whisper word times of <run>/voices/align.json, estimated from the audio length when absent).
The sound background (scripts/story_sound.py: rain, water, drone, heartbeat on the choice, riser and impact on the twist, thunder) is ducked under the voices.
Voice lines are <run>/voices/<shot id>.wav (scripts/box/story_voices.py). Everything here is Pillow + numpy + ffmpeg: light on the PC.
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
SIZE = (576, 1024)
FPS = 30
ORANGE = (255, 122, 0)
FONT_BOLD = Path(r"C:\Windows\Fonts\impact.ttf")
FONT_FALLBACK = Path(r"C:\Windows\Fonts\arialbd.ttf")
TAIL = {"talk": 0.15, "choice": 0.0, "narration": 0.2, "pov": 0.2, "twist": 2.4}  # the twist keeps the screen after the line: the impact rings and the end card shows  # silence after the line, per kind of shot
CHOICE_MIN_SECONDS = 3.2
RING_SECONDS_BEFORE_END = 0.35
RING_START_SECONDS = 0.5  # the voice says "now you must choose" first, then the ring empties
RING_CENTER_Y = 0.12  # top of the frame: the faces stay free
SUB_PIXELS = 84  # the size of the spoken word (it was 118: too big)
SUB_Y = 0.71  # and its height in the frame (it was 0.64): lower, clear of the faces, above the TikTok caption zone
GLITCH_STRENGTHS = (1.0, 0.7, 0.45, 0.2)  # one glitch frame each (1/30 s), then the clean word: the word hits, then it is calm and easy to read
GRADE = "eq=contrast=1.05:saturation=0.9:gamma=1.22,unsharp=5:5:0.6:5:5:0.0,noise=alls=5:allf=t,vignette=angle=PI/4.5"  # the look every shot gets: cold, shadows lifted (the pictures are very dark), a little grain


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def shot_seconds(kind: str, voice_seconds: float) -> float:
    base = voice_seconds + TAIL.get(kind, 0.3)
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
    return ImageFont.truetype(str(FONT_BOLD if FONT_BOLD.exists() else FONT_FALLBACK), size)


def word_image(word: str, size: tuple[int, int] = SIZE, y_fraction: float = SUB_Y, pixels: int | None = None) -> Image.Image:
    """One transparent frame with the word: white capitals, thick black outline, centred, shrunk if the word is wider than the frame."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    lines = word.upper().split(chr(10))
    pixels = pixels or (SUB_PIXELS if len(lines) == 1 else 76)
    font = _font(pixels)
    while max(draw.textlength(line, font=font) for line in lines) > size[0] * 0.9 and pixels > 30:
        pixels -= 6
        font = _font(pixels)
    top = size[1] * y_fraction - (len(lines) - 1) * pixels * 1.1 / 2
    for index, line in enumerate(lines):
        draw.text((size[0] / 2, top + index * pixels * 1.1), line, font=font, fill=(255, 255, 255, 255), anchor="mm", stroke_width=max(4, pixels // 11), stroke_fill=(0, 0, 0, 255))
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
    shift = max(1, int(round(16 * strength)))
    out = Image.new("RGBA", clean.size, (0, 0, 0, 0))
    for color, dx in (((255, 30, 70), shift), ((0, 230, 255), -shift)):
        out.alpha_composite(ImageChops.offset(_tint(clean, color), dx, 0))
    out.alpha_composite(clean)
    for _ in range(2 + int(2 * strength)):
        band_h = rnd.randint(6, 16)
        y0 = rnd.randint(bbox[1], max(bbox[1], bbox[3] - band_h))
        band = out.crop((0, y0, out.width, y0 + band_h))
        out.paste((0, 0, 0, 0), (0, y0, out.width, y0 + band_h))
        out.alpha_composite(ImageChops.offset(band, rnd.choice((-1, 1)) * rnd.randint(8, 26) * max(1, int(round(strength * 1.5))) // 1, 0), (0, y0))
    return out


def glitch_sequence(clean: Image.Image, seed: str) -> list[Image.Image]:
    """The frames of a word's entrance: glitch frames (strong to light) and, last, the clean word."""
    rnd = random.Random(seed)
    return [glitch_frame(clean, strength, rnd) for strength in GLITCH_STRENGTHS] + [clean]


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


def choice_frame(t: float, seconds: float, portraits: list[Image.Image], names: list[str], size: tuple[int, int] = SIZE, background: Image.Image | None = None) -> Image.Image:
    """The choice card at time t: the characters side by side (`portraits` as stills, or `background`, an already composed frame of moving clips), the countdown ring, the orange A / B cards."""
    w, h = size
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
    command += ["-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ac", "2", str(out)]
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


def subtitle_overlay(windows: list, seconds: float, work: Path) -> Path | None:
    """The on-screen words of a shot as a list for ffmpeg's concat demuxer (an image per frame of change, a transparent image in the gaps); returns the list file.
    A window is (text, start, end) or (text, start, end, options): options = {"y": fraction of the height, "pixels": font size, "glitch": False} (a title can be placed and sized apart).
    Every word ENTERS with a glitch (see glitch_frame) of a few frames and is then perfectly clean until the next word."""
    if not windows:
        return None
    work.mkdir(parents=True, exist_ok=True)
    blank = work / "blank.png"
    Image.new("RGBA", SIZE, (0, 0, 0, 0)).save(blank)
    frame = 1.0 / FPS
    lines, cursor = [], 0.0

    def entry(path: Path, duration: float) -> None:
        lines.extend([f"file '{path.resolve().as_posix()}'", f"duration {max(duration, 0.001):.4f}"])

    for index, window in enumerate(windows):
        word, start, end = window[0], window[1], window[2]
        options = window[3] if len(window) > 3 else {}
        if start > cursor + 1e-3:
            entry(blank, start - cursor)
        clean = word_image(word, y_fraction=options.get("y", SUB_Y), pixels=options.get("pixels"))
        frames = glitch_sequence(clean, f"{word}-{index}") if options.get("glitch", True) else [clean]
        remaining = end - start
        for k, image in enumerate(frames):
            path = work / f"w{index:03d}_{k}.png"
            image.save(path)
            last = k == len(frames) - 1
            duration = remaining if last else min(frame, remaining)
            if duration <= 0:
                break
            entry(path, duration)
            remaining -= duration
        cursor = max(end, start)
    entry(blank, max(0.04, seconds - cursor + 0.2))
    lines.append(f"file '{blank.resolve().as_posix()}'")  # the concat demuxer ignores the duration of the last entry
    listing = work / "words.ffconcat"
    listing.write_text("\n".join(lines), encoding="utf-8")
    return listing


def burn_subtitles(video: Path, listing: Path | None, out: Path, grade: str | None = GRADE) -> None:
    """Grade the shot, then lay the one-word subtitles on top (the text is never graded: no grain on the letters)."""
    chain = (grade + ",") if grade else ""
    if listing is None:
        graph = f"[0:v]{chain}format=yuv420p[v]"
        inputs = ["-i", str(video)]
    else:
        graph = f"[0:v]{chain}format=yuv420p[g];[1:v]fps={FPS},format=rgba[sub];[g][sub]overlay=0:0:eof_action=pass,format=yuv420p[v]"
        inputs = ["-i", str(video), "-f", "concat", "-safe", "0", "-i", str(listing)]
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-maxrate", "6M", "-bufsize", "12M", "-c:a", "copy",
                    "-shortest", str(out)], check=True)


def clip_to_shot(clip: Path, out: Path, seconds: float, audio: Path) -> None:
    """The S2V clip (480x832) filled to 576x1024 (scale then crop the 15 px of width), its own audio replaced by the clean voice line, held on its last frame if the clip is short."""
    graph = (f"[0:v]scale=-2:{SIZE[1]}:flags=lanczos,crop={SIZE[0]}:{SIZE[1]},fps={FPS},tpad=stop_mode=clone:stop_duration=2,trim=duration={seconds:.3f},setpts=PTS-STARTPTS[v];"
             f"[1:a]aresample=48000,apad=whole_dur={seconds:.3f}[a]")
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(clip), "-i", str(audio), "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-t", f"{seconds:.3f}", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ac", "2", str(out)], check=True)


def read_frames(clip: Path, seconds: float) -> list[Image.Image]:
    """The frames of a clip as 576x1024 pictures at 30 fps, exactly seconds * 30 of them (the last frame is held if the clip is short)."""
    raw = subprocess.run([mt.ffmpeg_binary(), "-v", "error", "-i", str(clip), "-vf", f"scale=-2:{SIZE[1]}:flags=lanczos,crop={SIZE[0]}:{SIZE[1]},fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    size = SIZE[0] * SIZE[1] * 3
    frames = [Image.frombytes("RGB", SIZE, raw[i:i + size]) for i in range(0, len(raw) - size + 1, size)]
    wanted = int(round(seconds * FPS))
    return (frames + [frames[-1]] * wanted)[:wanted]


def choice_clip_frames(clips: list[Path], names: list[str], seconds: float):
    """Two characters waiting, side by side (the centre of each clip), under the choice card."""
    panels = [read_frames(clip, seconds) for clip in clips]
    half = SIZE[0] // len(clips)
    for index in range(len(panels[0])):
        frame = Image.new("RGB", SIZE)
        for k, track in enumerate(panels):
            left = (SIZE[0] - half) // 2
            frame.paste(track[index].crop((left, 0, left + half, SIZE[1])), (k * half, 0))
        yield choice_frame(index / FPS, seconds, [], names, background=frame)


def shot_starts(timings: dict) -> dict:
    cursor, starts = 0.0, {}
    for sid, info in timings.items():
        starts[sid] = round(cursor, 3)
        cursor += info["seconds"]
    return starts


def render_story(plan: dict, cards_dir: Path, run: Path, out: Path, sound: bool = True) -> dict:
    cards = json.loads((cards_dir / "cards.json").read_text(encoding="utf-8"))
    voices_dir, clips_dir = run / "voices", run / "clips"
    aligned = json.loads((voices_dir / "align.json").read_text(encoding="utf-8")) if (voices_dir / "align.json").exists() else {}
    work = run / "render"
    work.mkdir(parents=True, exist_ok=True)
    characters = {c["id"]: c for c in plan["characters"]}
    timings, parts = {}, []
    for shot in plan["shots"]:
        started = time.time()
        sid, kind = shot["id"], shot["kind"]
        voice = voices_dir / f"{sid}.wav"
        seconds = shot_seconds(kind, wav_seconds(voice))
        raw, burned = work / f"{sid}_raw.mp4", work / f"{sid}.mp4"
        fallback = False
        if kind == "choice":
            ids = shot["in_shot"][:2]
            names = [characters[i]["name"] for i in ids]
            waiting = [clips_dir / f"{sid}_{i}.mp4" for i in ids]
            if all(c.exists() for c in waiting):
                _encode_frames(choice_clip_frames(waiting, names, seconds), raw, seconds, voice)
            else:  # no waiting clips: the portraits stand still under the card, and say so
                fallback = True
                portraits = [Image.open(ROOT / cards["characters"][i]["portrait"]["file"]) for i in ids]
                _encode_frames((choice_frame(k / FPS, seconds, portraits, names) for k in range(int(round(seconds * FPS)))), raw, seconds, voice)
                print(f"{sid}: NO WAITING CLIPS, still portraits", flush=True)
        elif (clips_dir / f"{sid}.mp4").exists():
            clip_to_shot(clips_dir / f"{sid}.mp4", raw, seconds, voice)
        else:  # no clip: a push-in on the picture of the shot, and say so
            fallback = True
            picture = run / "stills" / f"{sid}.png"
            if not picture.exists():
                picture = ROOT / cards["characters"][shot["speaker"]]["portrait"]["file"] if shot["kind"] == "talk" else ROOT / cards["locations"][shot["location"]]["file"]
            _encode_frames(kenburns_frames(Image.open(picture), seconds), raw, seconds, voice)
            print(f"{sid}: NO CLIP, push-in on {picture.name}", flush=True)
        spoken = seconds - TAIL.get(kind, 0.3)
        windows = aligned_windows(shot["text"], aligned.get(sid), spoken) if aligned.get(sid) else word_windows(shot["text"], spoken if kind != "choice" else 1.3)
        if plan.get("end_card") and shot is plan["shots"][-1]:  # a closing question over the last frames, after the last word
            windows = [*windows, (plan["end_card"], round(spoken + 0.35, 3), round(seconds - 0.1, 3))]
        burn_subtitles(raw, subtitle_overlay(windows, seconds, work / f"{sid}_words"), burned)
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
        starts = shot_starts(timings)
        ids = [s["id"] for s in plan["shots"]]
        choice = next((s for s in plan["shots"] if s["kind"] == "choice"), None)
        twist = next((s for s in plan["shots"] if s["kind"] == "twist"), None)
        cues = {"choice": [starts[choice["id"]] + RING_START_SECONDS, starts[choice["id"]] + timings[choice["id"]]["seconds"] - RING_SECONDS_BEFORE_END] if choice else None,
                "twist": starts[twist["id"]] if twist else None, "thunder": [starts[ids[3]], starts[ids[-3]]] if len(ids) > 6 else []}
        track = so.build_soundtrack(total, cues, work / "soundtrack.wav")
        so.mix_with_voices(joined, track, out)
    else:
        joined.replace(out)
    return {"file": str(out), "shots": timings, "total_seconds": total, "sound": sound, "fallback_shots": [s for s, t in timings.items() if t["fallback"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="the production folder: voices/, clips/, stills/ inside")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-sound", action="store_true")
    args = parser.parse_args()
    result = render_story(json.loads(args.plan.read_text(encoding="utf-8")), args.cards.resolve(), args.run, args.out, sound=not args.no_sound)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
