"""Driving-video preparation for Wan 2.2 Animate.

What the official template does with the driving clip (all confirmed live on
2026-10-02, see README "Animate driving clips"):

* LoadVideo -> ImageScale(lanczos, crop=center) to a fixed 640x640 square, so a
  16:9 clip silently loses its left/right thirds -- fighters standing at the
  sides are cut out of the pose and mask.
* Frames are consumed 1:1 and written at the shot's fps, so a 25 fps clip played
  at 16 fps runs in slow motion.
* Only the first 77-frame segment is clean; anything the driving clip does not
  cover afterwards freezes on its last pose (and the extension stage hard-cuts).

normalize_driving_video() fixes the first two by handing the template exactly
what it wants (a square, at the output fps); trim_video() fixes the third.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal

# The template's ImageScale and SAM2 PointsEditor both work on a 640x640 canvas.
TEMPLATE_CANVAS = 640
DrivingFit = Literal["crop", "pad"]


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise ValueError(f"{path} has no video stream")
    stream = streams[0]
    numerator, _, denominator = str(stream.get("r_frame_rate", "0/1")).partition("/")
    fps = float(numerator) / float(denominator or 1) if float(denominator or 1) else 0.0
    duration = float(stream["duration"]) if stream.get("duration") not in (None, "N/A") else 0.0
    frames = int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else round(duration * fps)
    return {"width": int(stream["width"]), "height": int(stream["height"]), "fps": fps, "frames": frames, "duration": duration}


def square_filter(fit: DrivingFit, fps: int, size: int = TEMPLATE_CANVAS) -> str:
    """ffmpeg -vf chain that makes a size x size clip at `fps`.

    "crop" reproduces the template's own center crop (so nothing changes for
    clips that already work); "pad" letterboxes instead, keeping the whole frame
    at lower effective resolution -- the right choice when the action reaches
    the sides of a wide clip.
    """
    if fit == "pad":
        fit_chain = (
            f"scale={size}:{size}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:black"
        )
    elif fit == "crop":
        fit_chain = f"scale={size}:{size}:force_original_aspect_ratio=increase:flags=lanczos,crop={size}:{size}"
    else:
        raise ValueError(f"unknown driving_fit {fit!r}")
    return f"fps={fps},{fit_chain},setsar=1"


def normalize_driving_video(src: Path, dst: Path, *, fps: int, fit: DrivingFit = "crop", size: int = TEMPLATE_CANVAS) -> dict:
    """Re-encodes the driving clip as a square, constant-fps, audio-less mp4 and
    returns probe info of the result. The caller keeps `src`'s own probe if it
    needs the original geometry (see to_square_coords)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(src),
            "-vf", square_filter(fit, fps, size), "-an",
            "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(dst),
        ],
        check=True,
    )
    return probe_video(dst)


def to_square_coords(x: float, y: float, width: int, height: int, fit: DrivingFit) -> tuple[float, float]:
    """Maps a point given in 0..1 coordinates of the ORIGINAL driving frame (what
    a person sees when picking the subject) into 0..1 coordinates of the
    normalized square clip. May fall outside 0..1 when "crop" cut the subject
    out; callers must treat that as an error rather than clamping."""
    if fit == "crop":
        if width >= height:
            return (x * width - (width - height) / 2) / height, y
        return x, (y * height - (height - width) / 2) / width
    if width >= height:
        return x, 0.5 + (y - 0.5) * height / width
    return 0.5 + (x - 0.5) * width / height, y


def trim_video(src: Path, dst: Path, frames: int) -> None:
    """Keeps the first `frames` frames. Used to cut the extension stage's frozen
    tail: the clip is only worth as many frames as the driving video had."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(src), "-frames:v", str(frames),
            "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-an", str(dst),
        ],
        check=True,
    )
