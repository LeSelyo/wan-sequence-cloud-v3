"""FFmpeg montage on the server: the jump cut shot (schemas.JumpCutSpec, Mode.JUMPCUT).

There is deliberately NO route that takes raw FFmpeg arguments: an arbitrary command line reads any file the service can read and fetches
any URL. What the API exposes are typed montage options (schemas.JumpCutSpec); this module turns them into FFmpeg runs, using the
same code as the trend videos (scripts/trend_philosopher.py: jumpcut_lengths, build_jumpcut, shake_filter).

FFMPEG_BIN / FFPROBE_BIN select the binary (the Docker image installs a recent static build and points them at it; see the Dockerfile).
`ffmpeg_status()` reports what is actually installed, and `validate_jumpcut()` refuses a shot at admission when a filter it needs is missing,
instead of failing halfway through a job.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .schemas import JUMPCUT_MIN_IMAGE_SECONDS, JumpCutSpec

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
# Filters every jump cut render needs, plus the ones each option adds (see trend_philosopher.build_phased_background / shake_filter).
BASE_FILTERS = ("fps", "format", "setsar", "scale", "crop")
TRANSITION_FILTERS = {"cut": (), "shake": (), "wipe_left": ("xfade",), "wipe_right": ("xfade",), "swing": ("xfade",)}
SHAKE_FILTERS = {
    "jitter": ("scale", "crop"),
    "roll": ("scale", "rotate", "crop"),
    "handheld": ("scale", "rotate", "crop"),
    "sway": ("scale", "rotate", "crop"),
    "rock": ("pad", "fillborders", "perspective", "crop"),
    "drift": ("pad", "fillborders", "perspective", "crop"),
}
REQUIRED_ENCODERS = ("libx264",)


def ffmpeg_binary() -> str:
    """FFMPEG_BIN, else the ffmpeg on PATH, else (a developer laptop) the pip package static-ffmpeg's build."""
    configured = os.environ.get("FFMPEG_BIN")
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import static_ffmpeg

        return static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()[0]
    except Exception:
        return "ffmpeg"


def ffprobe_binary() -> str:
    configured = os.environ.get("FFPROBE_BIN")
    if configured:
        return configured
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg = Path(ffmpeg_binary())
    sibling = ffmpeg.with_name("ffprobe" + ffmpeg.suffix)
    return str(sibling) if sibling.is_file() else "ffprobe"


def _run_capture(binary: str, *args: str) -> str:
    result = subprocess.run([binary, "-hide_banner", *args], capture_output=True, text=True, timeout=30)
    return result.stdout


def parse_filter_names(text: str) -> frozenset[str]:
    """Filter names out of `ffmpeg -filters`. The flag column is 3 characters in FFmpeg 4.x (" T.C acrusher") and 2 in 8.x (" TS aap")."""
    return frozenset(m.group(1) for m in re.finditer(r"^\s*[TSC.]{2,3}\s+(\w+)\s", text, re.MULTILINE))


def parse_encoder_names(text: str) -> frozenset[str]:
    """Encoder names out of `ffmpeg -encoders` (" V....D libx264 ...")."""
    return frozenset(m.group(1) for m in re.finditer(r"^\s*[VAS][A-Z.]{5}\s+(\S+)\s", text, re.MULTILINE))


@functools.lru_cache(maxsize=4)
def _inventory(binary: str) -> tuple[str, frozenset[str], frozenset[str]]:
    """(version banner, filter names, encoder names) of one ffmpeg binary. Cached: the binary does not change while the server runs."""
    banner = _run_capture(binary, "-version")
    filters_text = _run_capture(binary, "-filters")
    encoders_text = _run_capture(binary, "-encoders")
    return banner, parse_filter_names(filters_text), parse_encoder_names(encoders_text)


def ffmpeg_status(binary: str | None = None) -> dict[str, Any]:
    """What this FFmpeg is: its version, and whether it has everything a jump cut can need."""
    binary = binary or ffmpeg_binary()
    try:
        banner, filters, encoders = _inventory(binary)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ready": False, "binary": binary, "error": f"{type(exc).__name__}: {exc}"}
    version_match = re.search(r"ffmpeg version (\S+)", banner)
    version = version_match.group(1) if version_match else None
    major_match = re.match(r"n?(\d+)\.", version or "")
    every_filter = set(BASE_FILTERS) | {name for names in TRANSITION_FILTERS.values() for name in names} | {
        name for names in SHAKE_FILTERS.values() for name in names
    }
    missing = sorted(every_filter - filters) + sorted(set(REQUIRED_ENCODERS) - encoders)
    return {
        "ready": bool(version) and not missing,
        "binary": binary,
        "version": version,
        "major": int(major_match.group(1)) if major_match else None,
        "missing": missing,
    }


def required_filters(spec: JumpCutSpec) -> list[str]:
    needed = set(BASE_FILTERS) | set(TRANSITION_FILTERS[spec.transition])
    if spec.constant_shake:
        needed |= set(SHAKE_FILTERS[spec.constant_shake])
    return sorted(needed)


def validate_jumpcut(spec: JumpCutSpec) -> None:
    """Admission check (raises ValueError -> HTTP 422): this server's FFmpeg must have every filter the chosen options use."""
    status = ffmpeg_status()
    if not status.get("version"):
        raise ValueError(f"ffmpeg is not usable on this server ({status.get('error', 'no version reported')})")
    _, filters, encoders = _inventory(status["binary"])
    missing = [name for name in required_filters(spec) if name not in filters] + [
        name for name in REQUIRED_ENCODERS if name not in encoders
    ]
    if missing:
        raise ValueError(
            f"this server's ffmpeg {status['version']} lacks {missing}, needed by the chosen jump cut options; "
            "use a recent FFmpeg build (see the Dockerfile) or pick another transition / shake"
        )


@functools.lru_cache(maxsize=1)
def _trend_module():
    """scripts/trend_philosopher.py, loaded once and pointed at this server's ffmpeg."""
    module = sys.modules.get("trend_philosopher")
    if module is None:
        spec = importlib.util.spec_from_file_location("trend_philosopher", SCRIPTS_DIR / "trend_philosopher.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["trend_philosopher"] = module
        spec.loader.exec_module(module)
    module.FFMPEG, module.FFPROBE = ffmpeg_binary(), ffprobe_binary()
    return module


def render_jumpcut(
    spec: JumpCutSpec, images: list[Path], out_path: Path, *, width: int, height: int, fps: int, seconds: float, seed: int,
) -> Path:
    """The jump cut of `images` (spec.images, already fetched to local files, same order) lasting `seconds`, written to out_path.
    Everything it writes besides the video goes in out_path's folder, so give each shot its own folder."""
    if len(images) != len(spec.images):
        raise ValueError("render_jumpcut needs one local file per spec image")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    module = _trend_module()
    return module.build_jumpcut(
        images, out_path, total_seconds=seconds, ratios=spec.ratios, width=width, height=height, fps=fps,
        transition=spec.transition, transition_seconds=spec.transition_seconds,
        constant_shake=spec.constant_shake, constant_shake_amount=spec.constant_shake_amount,
        cut_range=(spec.cut_min_seconds, spec.cut_max_seconds), rng_seed=seed,
    )


def options() -> dict[str, Any]:
    """What a client can choose for a jump cut shot (GET /v1/montage): the values, the defaults and this server's FFmpeg."""
    defaults = JumpCutSpec.model_construct(images=[])
    return {
        "ffmpeg": ffmpeg_status(),
        "jumpcut": {
            "mode": "jumpcut",
            "transitions": list(TRANSITION_FILTERS),
            "constant_shakes": list(SHAKE_FILTERS),
            "defaults": {
                "transition": defaults.transition,
                "transition_seconds": defaults.transition_seconds,
                "constant_shake": defaults.constant_shake,
                "constant_shake_amount": defaults.constant_shake_amount,
                "cut_min_seconds": defaults.cut_min_seconds,
                "cut_max_seconds": defaults.cut_max_seconds,
            },
            "limits": {"images": [2, 40], "min_image_seconds": JUMPCUT_MIN_IMAGE_SECONDS},
            "example": {
                "id": "demo",
                "shots": [{
                    "id": "burst", "mode": "jumpcut", "frames": 121, "fps": 24,
                    "jumpcut": {
                        "images": [{"image_id": "img_<32 hex>"}, {"image_id": "img_<32 hex>"}, {"image_id": "img_<32 hex>"}],
                        "ratios": [1, 1, 2], "transition": "swing", "constant_shake": "rock",
                    },
                }],
            },
        },
    }
