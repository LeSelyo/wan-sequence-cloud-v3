#!/usr/bin/env python3
"""Reusable "kung-fu cat" trend toolkit, built from the 2026-10-02 experiments.

Everything goes through this app's own HTTP API (WAN_BASE_URL / WAN_API_TOKEN, same
contract as scripts/trend_philosopher.py), never straight to ComfyUI, so it works the
same against a local container or a rented GPU. ffmpeg/ffprobe must be on PATH.

Recipes (see README "Trend videos: kung-fu cat"):

  transform   a real-looking cat sits on a couch, its paws turn into human hands
              (Krea2 pair with the same seed -> Wan FLF2V morph), then it does
              kung fu (Animate move mode driven by a human performer's pose video).
  background  the same hands-cat in several settings, one pose clip, side by side
              (Animate move mode takes the scene from the reference image).
  fight       cat-vs-cat in two Animate passes over one two-person clip: pass A swaps
              fighter A, pass B replays pass A's output and swaps fighter B. SAM2 gets a
              green point on the fighter to replace and a red one on the other.
  sfx-download  fetch the Mixkit martial-arts sound effects used for hits and cries.

Pure helpers (motion peaks -> sound placement, joins, SFX mix) work on any video:
  python scripts/trend_cat_kungfu.py sfx-mix clip.mp4 out.mp4 --sfx-dir sfx

Driving clips: a person doing the moves, ideally frontal and upper-body for a face-cam
look, the subject visible in frame 0. The app squares/resamples them (see
app/driving_video.py); give a server-side `--driving-path` (relative to the mounted
videos dir) or a `--driving-url`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

try:  # optional: a pip-provided static ffmpeg when the host has none
    import static_ffmpeg

    static_ffmpeg.add_paths()
except Exception:  # pragma: no cover - environment dependent
    pass

BASE_URL = os.environ.get("WAN_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.environ.get("WAN_API_TOKEN", "")
SQUARE = 512  # validated Animate size (the template squares the driving clip anyway)

# --------------------------------------------------------------------------
# Prompts that worked (Krea2 turbo, seed 2: same cat and couch in both images)
# --------------------------------------------------------------------------
SCENES = {
    "couch": ("on a beige fabric couch in a cosy living room", "soft natural window light"),
    "dojo": ("on a wooden floor in a traditional japanese dojo with paper sliding doors behind", "soft warm window light"),
    "roof": ("on a rooftop ledge at night with a glowing city skyline behind", "neon city lights and soft rim light"),
    "forest": ("on a mossy rock in a misty bamboo forest", "soft green morning light"),
}
_CAT = (
    "candid webcam photo of {kind} fluffy ginger tabby cat sitting upright like a person {where}, centered, "
    "facing the camera directly and looking into the lens, {light}, realistic photo, sharp focus, shallow depth of field"
)
PLAIN_SUFFIX = ", relaxed, front paws resting on its lap"
# Krea2 rarely draws human hands on a cat; spelling out fingers/thumbs/bare arms with
# "anthropomorphic" is what gave an actual cat with human-like hands.
HANDS_SUFFIX = ", with realistic human hands with fingers and thumbs, hands open in front in a kung fu pose, bare furry arms"
MORPH_PROMPT = (
    "a ginger tabby cat sitting on a couch facing the camera, its front paws slowly transform into human hands "
    "with fingers while it raises them in a kung fu guard, smooth realistic transformation, static camera"
)
KUNGFU_PROMPT = (
    "a ginger tabby cat with human hands sitting upright facing the camera, performing fast kung fu hand strikes "
    "and blocks, serious face"
)


def plain_cat_prompt(scene: str = "couch") -> str:
    where, light = SCENES[scene]
    return _CAT.format(kind="a", where=where, light=light) + PLAIN_SUFFIX


def hands_cat_prompt(scene: str = "couch") -> str:
    where, light = SCENES[scene]
    return _CAT.format(kind="an anthropomorphic", where=where, light=light) + HANDS_SUFFIX


# --------------------------------------------------------------------------
# ffmpeg helpers
# --------------------------------------------------------------------------
def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def probe_fps(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1", str(video)],
        check=True, capture_output=True, text=True,
    )
    numerator, _, denominator = out.stdout.strip().partition("/")
    return float(numerator) / float(denominator or 1)


def motion_energy(video: Path) -> list[float]:
    """Mean absolute difference between consecutive frames, one value per frame:
    where the fighter hits, kicks or strikes, this spikes."""
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(video),
            "-vf", "tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
            "-an", "-f", "null", "-",
        ],
        check=True, capture_output=True, text=True,
    )
    return [float(line.split("=", 1)[1]) for line in result.stdout.splitlines() if "YAVG=" in line]


def strongest_hits(energy: list[float], fps: float, *, count: int = 4, min_gap: float = 0.55, skip: float = 0.3) -> list[float]:
    """Times (seconds) of the `count` strongest motion peaks that are at least
    `min_gap` apart; the first `skip` seconds (the frame-0 transient) are ignored."""
    ranked = sorted(range(len(energy)), key=lambda index: energy[index], reverse=True)
    chosen: list[float] = []
    for index in ranked:
        t = index / fps
        if t < skip or any(abs(t - other) < min_gap for other in chosen):
            continue
        chosen.append(t)
        if len(chosen) == count:
            break
    return sorted(chosen)


# Mixkit "Sound Effects Free License" previews (no attribution required). `peak` is where the
# transient lands inside the file, so a sound can be started `peak` seconds before the hit;
# `gain` compensates the quiet voice samples (measured 2026-10-02).
SFX = {
    "strike_effort": {"id": 2162, "peak": 0.08, "gain": 1.0},
    "kick": {"id": 2163, "peak": 0.40, "gain": 0.9},
    "scream": {"id": 2175, "peak": 0.35, "gain": 2.5},
    "effort": {"id": 2174, "peak": 0.24, "gain": 5.0},
    "karate_hit": {"id": 2154, "peak": 0.13, "gain": 4.0},
    "punch_fast": {"id": 2047, "peak": 0.26, "gain": 1.0},
    "punch": {"id": 2052, "peak": 0.15, "gain": 1.0},
}


def download_sfx(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, spec in SFX.items():
        target = dest / f"{name}.mp3"
        url = f"https://assets.mixkit.co/active_storage/sfx/{spec['id']}/{spec['id']}-preview.mp3"
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
        target.write_bytes(response.content)
        paths.append(target)
    return paths


LEAD = {"scream": 0.1}  # the battle cry starts a beat before the kick connects


def plan_sfx(hits: list[float], duration: float) -> list[tuple[str, float, float]]:
    """(sfx name, start time, gain) for each hit: punches and karate hits, with a
    short effort grunt on most of them, and on the LAST hit (the finisher, when there
    are several) a kick + battle scream + strike-with-effort. Starts are shifted by
    each sample's own transient position so the sound lands on the motion peak."""
    events: list[tuple[str, float, float]] = []
    cycle = ["punch_fast", "punch", "karate_hit"]
    for index, t in enumerate(hits):
        if index == len(hits) - 1 and len(hits) > 1:
            names = ["kick", "scream", "strike_effort"]
        else:
            names = [cycle[index % 3]] + (["effort"] if index % 3 != 1 else [])
        for name in names:
            spec = SFX[name]
            events.append((name, max(0.0, t - spec["peak"] - LEAD.get(name, 0.0)), spec["gain"]))
    return [event for event in events if event[1] < duration]


def _amix(inputs: int) -> str:
    """`amix` at full level per input: `normalize=0` needs FFmpeg >= 4.4, older builds get
    dropout_transition=0 plus a gain that undoes the 1/N scaling."""
    help_text = subprocess.run(["ffmpeg", "-hide_banner", "-h", "filter=amix"], capture_output=True, text=True)
    if "normalize" in help_text.stdout + help_text.stderr:
        return f"amix=inputs={inputs}:normalize=0"
    return f"amix=inputs={inputs}:dropout_transition=0,volume={inputs}"


def mix_sfx(video: Path, out: Path, events: list[tuple[str, float, float]], sfx_dir: Path, *, upscale: int | None = 1024, fade: float = 0.25) -> Path:
    """Adds the planned sounds to a (silent) video, optionally lanczos-upscales it,
    and fades both out. `upscale` is the output width (square clips -> square)."""
    duration = probe_duration(video)
    scale = f"scale={upscale}:-2:flags=lanczos,unsharp=5:5:0.8:5:5:0.0," if upscale else ""
    if not events:  # nothing to place: just upscale/fade and keep the clip silent
        _run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(video),
                "-vf", f"{scale}fade=t=out:st={max(0.0, duration - fade):.3f}:d={fade}", "-an",
                "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out),
            ]
        )
        return out
    inputs = ["-i", str(video)]
    parts = []
    for index, (name, start, gain) in enumerate(events, start=1):
        inputs += ["-i", str(sfx_dir / f"{name}.mp3")]
        ms = int(start * 1000)
        parts.append(f"[{index}:a]volume={gain},adelay={ms}|{ms}[a{index}]")
    labels = "".join(f"[a{index}]" for index in range(1, len(events) + 1))
    parts.append(
        f"{labels}{_amix(len(events))},alimiter=limit=0.95,atrim=0:{duration:.3f},"
        f"afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade}[aout]"
    )
    parts.append(f"[0:v]{scale}fade=t=out:st={max(0.0, duration - fade):.3f}:d={fade}[vout]")
    _run(
        [
            "ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(parts),
            "-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", str(out),
        ]
    )
    return out


def join_clips(clips: list[Path], out: Path, xfade: float = 0.2) -> Path:
    """Video-only crossfade chain (all clips must share size and fps)."""
    if len(clips) == 1:
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(clips[0]), "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out)])
        return out
    durations = [probe_duration(clip) for clip in clips]
    filters, previous, offset = [], "[0:v]", durations[0] - xfade
    for index in range(1, len(clips)):
        label = f"[v{index}]"
        filters.append(f"{previous}[{index}:v]xfade=transition=fade:duration={xfade}:offset={offset:.3f}{label}")
        previous = label
        offset += durations[index] - xfade
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for clip in clips:
        cmd += ["-i", str(clip)]
    _run(cmd + ["-filter_complex", ";".join(filters), "-map", previous, "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out)])
    return out


def stack_side_by_side(clips: list[Path], out: Path) -> Path:
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for clip in clips:
        cmd += ["-i", str(clip)]
    inputs = "".join(f"[{index}:v]" for index in range(len(clips)))
    _run(cmd + ["-filter_complex", f"{inputs}hstack=inputs={len(clips)}[v]", "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out)])
    return out


def last_frame(video: Path, out_png: Path) -> Path:
    _run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.1", "-i", str(video), "-frames:v", "1", "-update", "1", str(out_png)])
    return out_png


def cut_to(video: Path, out: Path, seconds: float) -> Path:
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-t", f"{seconds:.3f}", "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out)])
    return out


# --------------------------------------------------------------------------
# Request builders (validated against app.schemas in tests)
# --------------------------------------------------------------------------
def driving_source(*, path: str | None = None, url: str | None = None, job_id: str | None = None) -> dict:
    chosen = {key: value for key, value in (("path", path), ("url", url), ("job_id", job_id)) if value}
    if len(chosen) != 1:
        raise ValueError("give exactly one of --driving-path, --driving-url or a previous job")
    return chosen


def image_request(prompt: str, seed: int, size: int = 1024) -> dict:
    return {"engine": "krea2", "prompt": prompt, "width": size, "height": size, "steps": 8, "seed": seed}


def morph_shot(plain_image_id: str, hands_image_id: str, *, shot_id: str = "morph", seed: int = 5, prompt: str = MORPH_PROMPT) -> dict:
    # Wan 2.2 FLF2V: 81 frames at 16 fps = 5 s; the non-turbo recipe is 20 steps (verified 235 s on a 4090).
    return {
        "id": shot_id, "mode": "t+i(keyframe)2v", "prompt": prompt,
        "start_image": {"image_id": plain_image_id}, "end_image": {"image_id": hands_image_id},
        "width": SQUARE, "height": SQUARE, "frames": 81, "fps": 16, "steps": 20, "cfg": 4.0, "seed": seed,
    }


def animate_shot(
    reference_image_id: str, driving: dict, *, mode: str = "animate_move", shot_id: str = "animate", seed: int = 9,
    prompt: str = KUNGFU_PROMPT, driving_fit: str = "crop", subject_point: tuple[float, float] | None = None,
    exclude_points: list[tuple[float, float]] | None = None,
) -> dict:
    shot = {
        "id": shot_id, "mode": mode, "prompt": prompt, "start_image": {"image_id": reference_image_id},
        "driving_video": driving, "width": SQUARE, "height": SQUARE, "fps": 16, "seed": seed, "driving_fit": driving_fit,
    }
    if subject_point is not None:
        shot["subject_point"] = list(subject_point)  # SAM2 green point: the fighter to replace
    if exclude_points:
        shot["exclude_points"] = [list(point) for point in exclude_points]  # SAM2 red points: leave these out
    return shot


def to_padded_square(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    """A point in 0..1 of the ORIGINAL driving frame -> 0..1 of the same frame once
    the app letterboxed it into a square (driving_fit="pad"); pass A's output has
    that layout, so pass B's points live there."""
    x, y = point
    if width >= height:
        return x, 0.5 + (y - 0.5) * height / width
    return 0.5 + (x - 0.5) * width / height, y


# --------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------
class Api:
    def __init__(self, base_url: str = BASE_URL, token: str = API_TOKEN):
        self.base_url, self.token = base_url.rstrip("/"), token

    def _call(self, method: str, path: str, **kwargs) -> httpx.Response:
        headers = kwargs.pop("headers", {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        with httpx.Client(timeout=kwargs.pop("timeout", 120), follow_redirects=True) as client:
            response = client.request(method, self.base_url + path, headers=headers, **kwargs)
        response.raise_for_status()
        return response

    def _wait(self, status_url: str, *, poll: float = 5.0, timeout: float = 7200.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self._call("GET", status_url).json()
            if job["status"] == "completed":
                return job
            if job["status"] == "failed":
                raise RuntimeError(f"{status_url} failed: {job.get('error')}")
            time.sleep(poll)
        raise TimeoutError(status_url)

    def image(self, prompt: str, seed: int, size: int = 1024) -> str:
        accepted = self._call("POST", "/v1/images", json=image_request(prompt, seed, size)).json()
        self._wait(accepted["status_url"], poll=3.0)
        return accepted["image_id"]

    def upload_image(self, path: Path) -> str:
        with path.open("rb") as handle:
            return self._call("POST", "/v1/images/upload", files={"file": (path.name, handle, "image/png")}).json()["image_id"]

    def job(self, job_id: str, shots: list[dict], out: Path) -> Path:
        accepted = self._call("POST", "/v1/jobs", json={"id": job_id, "shots": shots}).json()
        self._wait(accepted["status_url"])
        out.write_bytes(self._call("GET", f"/v1/jobs/{job_id}/output", timeout=600).content)
        return out


# --------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------
def run_transform(api: Api, work: Path, driving: dict, *, scene: str = "couch", seed: int = 2, sfx_dir: Path | None = None) -> Path:
    """Normal cat -> paws turn into hands -> kung fu, as one video."""
    work.mkdir(parents=True, exist_ok=True)
    tag = f"{scene}_{seed}_{int(time.time())}"
    plain = api.image(plain_cat_prompt(scene), seed)
    hands = api.image(hands_cat_prompt(scene), seed)
    morph = api.job(f"morph_{tag}", [morph_shot(plain, hands)], work / "morph.mp4")
    reference = api.upload_image(last_frame(morph, work / "morph_last.png"))
    kungfu = api.job(f"kungfu_{tag}", [animate_shot(reference, driving)], work / "kungfu.mp4")
    joined = join_clips([morph, kungfu], work / "joined.mp4")
    return finish_with_sfx(joined, work / "transform_final.mp4", sfx_dir)


def run_background_change(api: Api, work: Path, driving: dict, scenes: list[str], *, seed: int = 2, sfx_dir: Path | None = None) -> Path:
    """One hands-cat, one pose clip, several settings, side by side."""
    work.mkdir(parents=True, exist_ok=True)
    tag = f"{seed}_{int(time.time())}"
    clips = []
    for scene in scenes:
        reference = api.image(hands_cat_prompt(scene), seed)
        clips.append(api.job(f"bg_{scene}_{tag}", [animate_shot(reference, driving, shot_id=scene)], work / f"{scene}.mp4"))
    return finish_with_sfx(stack_side_by_side(clips, work / "stacked.mp4"), work / "background_final.mp4", sfx_dir, upscale=None)


def run_fight(
    api: Api, work: Path, driving: dict, first_ref: Path, second_ref: Path, first_point: tuple[float, float],
    second_point: tuple[float, float], *, driving_size: tuple[int, int], max_seconds: float = 2.9,
    sfx_dir: Path | None = None,
) -> Path:
    """Cat vs cat in two Animate passes over one two-fighter clip.

    Both points are (x, y) in 0..1 of the ORIGINAL driving frame at frame 0, with both
    fighters visible. SAM2 gets two kinds of points: GREEN on the fighter to replace
    and RED on the other one (and, in pass B, on the first cat) so the mask does not
    bleed from one fighter onto the other. Pass A replaces the first fighter; pass B
    replays pass A's output (a padded square: `driving_size` maps the points into it)
    and replaces the second. `driving_fit=pad` keeps both fighters of a wide clip;
    `max_seconds` cuts before the source's own shot change."""
    work.mkdir(parents=True, exist_ok=True)
    tag = str(int(time.time()))
    width, height = driving_size
    a_ref, b_ref = api.upload_image(first_ref), api.upload_image(second_ref)
    pass_a = f"fight_a_{tag}"
    api.job(pass_a, [animate_shot(
        a_ref, driving, mode="animate_mix", shot_id="a", driving_fit="pad", subject_point=first_point,
        exclude_points=[second_point],
        prompt="a grey tabby cat with a clearly visible face and head, white karate gi, fighting")], work / "pass_a.mp4")
    api.job(f"fight_b_{tag}", [animate_shot(
        b_ref, {"job_id": pass_a}, mode="animate_mix", shot_id="b",
        subject_point=to_padded_square(second_point, width, height),
        exclude_points=[to_padded_square(first_point, width, height)],
        prompt="an orange cat with a clearly visible face in a dark vest, fighting, taking hits")], work / "pass_b.mp4")
    return finish_with_sfx(cut_to(work / "pass_b.mp4", work / "pass_b_cut.mp4", max_seconds), work / "fight_final.mp4", sfx_dir)


def finish_with_sfx(video: Path, out: Path, sfx_dir: Path | None, *, upscale: int | None = 1024) -> Path:
    if sfx_dir is None:
        return video
    fps = probe_fps(video)
    hits = strongest_hits(motion_energy(video), fps)
    return mix_sfx(video, out, plan_sfx(hits, probe_duration(video)), sfx_dir, upscale=upscale)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _point(text: str) -> tuple[float, float]:
    x, y = (float(part) for part in text.split(","))
    return x, y


def _size(text: str) -> tuple[int, int]:
    width, height = (int(part) for part in text.lower().split("x"))
    return width, height


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def driving_args(p):
        p.add_argument("--driving-path")
        p.add_argument("--driving-url")

    p = sub.add_parser("sfx-download")
    p.add_argument("dest", type=Path)
    p = sub.add_parser("sfx-mix")
    p.add_argument("video", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--sfx-dir", type=Path, required=True)
    p.add_argument("--hits", type=int, default=4)
    p = sub.add_parser("transform")
    driving_args(p)
    p.add_argument("--work", type=Path, default=Path("trend_cat_work"))
    p.add_argument("--scene", choices=sorted(SCENES), default="couch")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--sfx-dir", type=Path)
    p = sub.add_parser("background")
    driving_args(p)
    p.add_argument("--work", type=Path, default=Path("trend_cat_work"))
    p.add_argument("--scenes", default="dojo,roof,forest")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--sfx-dir", type=Path)
    p = sub.add_parser("fight")
    driving_args(p)
    p.add_argument("--work", type=Path, default=Path("trend_cat_work"))
    p.add_argument("--first-ref", type=Path, required=True, help="character image replacing the first fighter")
    p.add_argument("--second-ref", type=Path, required=True, help="character image replacing the second fighter")
    p.add_argument("--first-point", type=_point, required=True, help="x,y (0..1) of the first fighter in the driving clip's frame 0")
    p.add_argument("--second-point", type=_point, required=True, help="x,y (0..1) of the second fighter, same frame")
    p.add_argument("--driving-size", type=_size, required=True, help="WxH of the driving clip, e.g. 1280x720")
    p.add_argument("--max-seconds", type=float, default=2.9)
    p.add_argument("--sfx-dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "sfx-download":
        print("\n".join(str(path) for path in download_sfx(args.dest)))
    elif args.command == "sfx-mix":
        hits = strongest_hits(motion_energy(args.video), probe_fps(args.video), count=args.hits)
        print(mix_sfx(args.video, args.out, plan_sfx(hits, probe_duration(args.video)), args.sfx_dir))
    else:
        api = Api()
        driving = driving_source(path=args.driving_path, url=args.driving_url)
        if args.command == "transform":
            print(run_transform(api, args.work, driving, scene=args.scene, seed=args.seed, sfx_dir=args.sfx_dir))
        elif args.command == "background":
            print(run_background_change(api, args.work, driving, args.scenes.split(","), seed=args.seed, sfx_dir=args.sfx_dir))
        else:
            print(run_fight(api, args.work, driving, args.first_ref, args.second_ref, args.first_point, args.second_point,
                            driving_size=args.driving_size, max_seconds=args.max_seconds, sfx_dir=args.sfx_dir))


if __name__ == "__main__":
    main(sys.argv[1:])
