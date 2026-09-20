#!/usr/bin/env python3
"""Check ffmpeg/ffprobe and the app's own concatenate() on synthetic portrait clips.

Runs inside the built image (no GPU needed):
    docker run --rm --entrypoint python IMAGE /app/scripts/smoke_concat.py

Builds three 480x832 / 16 fps H.264 clips like the Wan CreateVideo output, then joins them
with transition "cut" (stream copy) and "crossfade" (xfade re-encode) and checks the
resulting duration, size and frame rate.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

WIDTH, HEIGHT, FPS, SECONDS, FADE = 480, 832, 16, 2, 0.25

workdir = Path(tempfile.mkdtemp(prefix="concat_smoke_"))
os.environ.setdefault("DATA_ROOT", str(workdir / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.orchestrator import concatenate  # noqa: E402  (needs DATA_ROOT set first)


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,codec_name",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    return {**info["streams"][0], "duration": float(info["format"]["duration"])}


def check(name: str, ok: bool, detail: object) -> None:
    print(("PASS" if ok else "FAIL"), name, detail, flush=True)
    if not ok:
        sys.exit(1)


clips = []
for index, color in enumerate(("red", "green", "blue")):
    clip = workdir / f"clip{index}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c={color}:s={WIDTH}x{HEIGHT}:r={FPS}:d={SECONDS}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    clips.append(clip)

cut = workdir / "cut.mp4"
concatenate(clips, cut, "cut", 0.0)
info = probe(cut)
check("cut: duration", abs(info["duration"] - 3 * SECONDS) < 0.2, info)
check("cut: size", (info["width"], info["height"]) == (WIDTH, HEIGHT), info)
check("cut: frames", int(info["nb_read_frames"]) == 3 * SECONDS * FPS, info)

fade = workdir / "fade.mp4"
concatenate(clips, fade, "crossfade", FADE)
info = probe(fade)
expected = 3 * SECONDS - 2 * FADE
check("crossfade: duration", abs(info["duration"] - expected) < 0.2, info)
check("crossfade: size", (info["width"], info["height"]) == (WIDTH, HEIGHT), info)
check("crossfade: fps", info["r_frame_rate"] == f"{FPS}/1", info)
print("concat smoke OK")
