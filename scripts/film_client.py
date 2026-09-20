#!/usr/bin/env python3
"""End-to-end remote client: Flux keyframes (+LoRA) -> chained FLF2V film, through the public API.

Talks to the service exactly like a remote user would (bearer token over HTTP), never to ComfyUI.

    WAN_BASE_URL=https://host WAN_API_TOKEN=... python scripts/film_client.py all \
        --film-dir results/film_artemis_portrait

Stages (run one or several; `all` runs them in order):
    smoke      health, readiness, catalog contents
    auth       missing / wrong / valid bearer token
    uploads    PNG + JPEG accepted; WebP, garbage and oversized images rejected
    flux       same seed with and without the film's Flux LoRA (proves the LoRA is applied)
    keyframes  every keyframe of film.json (Flux Schnell + LoRA), saved to keyframes/
    reupload   re-upload one keyframe as JPEG and use that image_id (exercises the upload route)
    pilot      2 short FLF2V shots with a crossfade (exercises FLF2V + ffmpeg xfade cheaply)
    film       the full FLF2V job (transition "cut"), sequence saved as film_sequence.mp4
    clips      per-shot clips fetched over scp (WAN_SSH=user@host, WAN_SSH_PORT, WAN_SSH_KEY)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

STAGES = ["smoke", "auth", "uploads", "flux", "keyframes", "reupload", "pilot", "film", "clips"]
BASE = os.environ.get("WAN_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TOKEN = os.environ.get("WAN_API_TOKEN", "")
T0 = time.time()
FILM_DIR = Path(".")


def log(event: str, **fields) -> None:
    record = {"t": round(time.time() - T0, 1), "event": event, **fields}
    print(json.dumps(record, ensure_ascii=False), flush=True)
    with (FILM_DIR / "run_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def request(method: str, path: str, *, token: str | None = TOKEN, retries: int = 4, **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    timeout = kwargs.pop("timeout", 60)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                return client.request(method, BASE + path, headers=headers, **kwargs)
        except httpx.TransportError as exc:  # proxy hiccup / cold start: retry
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{method} {path} unreachable after {retries} attempts: {last!r}")


def expect(name: str, ok: bool, detail: str = "") -> None:
    log("check", name=name, result="PASS" if ok else "FAIL", detail=detail)
    if not ok:
        raise SystemExit(f"check failed: {name} {detail}")


def load_film() -> dict:
    return json.loads((FILM_DIR / "film.json").read_text(encoding="utf-8"))


def state_path() -> Path:
    return FILM_DIR / "state.json"


def load_state() -> dict:
    if state_path().exists():
        return json.loads(state_path().read_text(encoding="utf-8"))
    return {"image_ids": {}}


def save_state(state: dict) -> None:
    state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------- stages
def stage_smoke(film: dict, state: dict) -> None:
    expect("live", request("GET", "/health/live", token=None).status_code == 200)
    ready = request("GET", "/health/ready", token=None)
    log("ready", status=ready.status_code, body=ready.json())
    expect("ready", ready.status_code == 200, ready.text[:500])
    catalog = request("GET", "/v1/catalog")
    expect("catalog", catalog.status_code == 200, str(catalog.status_code))
    items = catalog.json().get("items", {})
    expect("catalog has flux niji lora", film["flux"]["lora_id"] in items)


def stage_auth(film: dict, state: dict) -> None:
    for label, token in (("missing", None), ("wrong", "not-the-token")):
        response = request("GET", "/v1/catalog", token=token)
        expect(f"auth {label} token rejected", response.status_code in (401, 403), str(response.status_code))
    post = request("POST", "/v1/jobs", token=None, json={})
    expect("auth POST /v1/jobs without token rejected", post.status_code in (401, 403), str(post.status_code))
    expect("auth valid token", request("GET", "/v1/catalog").status_code == 200)


def _png_bytes(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (40, 90, 160)).save(buffer, "PNG")
    return buffer.getvalue()


def stage_uploads(film: dict, state: dict) -> None:
    from PIL import Image

    fmt = film["format"]
    png = _png_bytes(fmt["width"], fmt["height"])
    jpeg_buffer = io.BytesIO()
    Image.open(io.BytesIO(png)).save(jpeg_buffer, "JPEG", quality=92)
    webp_buffer = io.BytesIO()
    Image.open(io.BytesIO(png)).save(webp_buffer, "WEBP")
    cases = [
        ("png", "a.png", "image/png", png, {201}),
        ("jpeg", "a.jpg", "image/jpeg", jpeg_buffer.getvalue(), {201}),
        ("webp", "a.webp", "image/webp", webp_buffer.getvalue(), {400, 415, 422}),
        ("garbage", "a.png", "image/png", b"not an image at all", {400, 415, 422}),
        ("oversized-pixels", "big.png", "image/png", _png_bytes(4200, 4200), {400, 413, 422}),
    ]
    for name, filename, mime, data, accepted in cases:
        response = request("POST", "/v1/images/upload", files={"file": (filename, data, mime)}, timeout=120)
        expect(f"upload {name}", response.status_code in accepted, f"{response.status_code} {response.text[:200]}")


def _wait_image(image_id: str, timeout: int = 900) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = request("GET", f"/v1/images/{image_id}").json()
        if job["status"] == "completed":
            return job
        if job["status"] == "failed":
            raise SystemExit(f"image {image_id} failed: {job.get('error')}")
        time.sleep(2)
    raise SystemExit(f"image {image_id} timed out")


def _generate_image(prompt: str, seed: int, width: int, height: int, loras: list[dict], guidance: float, steps: int) -> tuple[str, float]:
    started = time.time()
    response = request(
        "POST",
        "/v1/images",
        json={"engine": "flux_schnell", "prompt": prompt, "seed": seed, "width": width, "height": height,
              "steps": steps, "guidance": guidance, "loras": loras},
    )
    if response.status_code != 202:
        raise SystemExit(f"POST /v1/images -> {response.status_code}: {response.text[:500]}")
    image_id = response.json()["image_id"]
    _wait_image(image_id)
    return image_id, round(time.time() - started, 1)


def _download(path: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=600, follow_redirects=True) as client:
        with client.stream("GET", BASE + path, headers={"Authorization": f"Bearer {TOKEN}"}) as response:
            if response.status_code != 200:
                raise SystemExit(f"GET {path} -> {response.status_code}")
            with target.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)


def stage_flux(film: dict, state: dict) -> None:
    from PIL import Image, ImageChops, ImageStat

    flux, fmt = film["flux"], film["format"]
    prompt = "niji, " + "a rocket lifting off at dawn over the ocean, vertical composition, anime illustration"
    out = FILM_DIR / "flux_lora_test"
    out.mkdir(exist_ok=True)
    results = {}
    for label, loras in (
        ("without_lora", []),
        ("with_lora", [{"id": flux["lora_id"], "weight": flux["lora_weight"], "target": "keyframe"}]),
    ):
        image_id, seconds = _generate_image(prompt, 777, fmt["width"], fmt["height"], loras, flux["guidance"], flux["steps"])
        _download(f"/v1/images/{image_id}/output", out / f"{label}.png")
        results[label] = image_id
        log("flux_image", label=label, image_id=image_id, seconds=seconds)
    a = Image.open(out / "without_lora.png").convert("RGB")
    b = Image.open(out / "with_lora.png").convert("RGB")
    expect("flux images have requested size", a.size == b.size == (fmt["width"], fmt["height"]), f"{a.size} {b.size}")
    diff = sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3
    expect("flux LoRA changes the image (mean abs diff > 3/255)", diff > 3, f"diff={diff:.2f}")


def stage_keyframes(film: dict, state: dict) -> None:
    flux, fmt = film["flux"], film["format"]
    folder = FILM_DIR / "keyframes"
    folder.mkdir(exist_ok=True)
    loras = [{"id": flux["lora_id"], "weight": flux["lora_weight"], "target": "keyframe"}]
    for keyframe in film["keyframes"]:
        file = folder / f"{keyframe['id']}.png"
        key = str(keyframe["n"])
        if file.exists() and key in state["image_ids"]:
            continue
        image_id, seconds = _generate_image(keyframe["prompt"], keyframe["seed"], fmt["width"], fmt["height"], loras, flux["guidance"], flux["steps"])
        _download(f"/v1/images/{image_id}/output", file)
        state["image_ids"][key] = image_id
        save_state(state)
        log("keyframe", n=keyframe["n"], id=keyframe["id"], image_id=image_id, seconds=seconds)


def stage_reupload(film: dict, state: dict) -> None:
    from PIL import Image

    n = film["reupload_keyframe"]
    keyframe = film["keyframes"][n]
    buffer = io.BytesIO()
    Image.open(FILM_DIR / "keyframes" / f"{keyframe['id']}.png").convert("RGB").save(buffer, "JPEG", quality=95)
    response = request("POST", "/v1/images/upload", files={"file": (f"{keyframe['id']}.jpg", buffer.getvalue(), "image/jpeg")}, timeout=120)
    expect("reupload jpeg", response.status_code == 201, f"{response.status_code} {response.text[:200]}")
    state["image_ids"][str(n)] = response.json()["image_id"]
    state["reuploaded"] = n
    save_state(state)


def _shot(film: dict, clip: dict, state: dict, frames: int | None = None, shot_id: str | None = None) -> dict:
    fmt, wan, ids = film["format"], film["wan"], state["image_ids"]
    return {
        "id": shot_id or clip["id"],
        "mode": wan["mode"],
        "prompt": clip["prompt"],
        "negative_prompt": film["negative_prompt"],
        "start_image": {"image_id": ids[str(clip["start_keyframe"])]},
        "end_image": {"image_id": ids[str(clip["end_keyframe"])]},
        "width": fmt["width"],
        "height": fmt["height"],
        "frames": frames or clip["frames"],
        "fps": fmt["fps"],
        "seed": clip["seed"],
        "steps": wan["steps"],
        "cfg": wan["cfg"],
        "turbo_mode": False,
        "loras": [],
    }


def _run_job(job: dict, download_to: Path, timeout: int) -> dict:
    response = request("POST", "/v1/jobs", json=job)
    if response.status_code != 202:
        raise SystemExit(f"POST /v1/jobs -> {response.status_code}: {response.text[:800]}")
    log("job_accepted", **response.json())
    started, last = time.time(), None
    while time.time() - started < timeout:
        status = request("GET", f"/v1/jobs/{job['id']}").json()
        view = (status["status"], status.get("current_shot"), round(status.get("progress") or 0, 3))
        if view != last:
            log("job_progress", job_id=job["id"], status=status["status"], current_shot=status.get("current_shot"), progress=view[2])
            last = view
        if status["status"] == "completed":
            _download(f"/v1/jobs/{job['id']}/output", download_to)
            log("job_done", job_id=job["id"], seconds=round(time.time() - started, 1), bytes=download_to.stat().st_size, file=str(download_to))
            return status
        if status["status"] == "failed":
            raise SystemExit(f"job {job['id']} failed: {status.get('error')}")
        time.sleep(10)
    raise SystemExit(f"job {job['id']} timed out after {timeout}s")


def _job_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%m%d_%H%M%S')}"


def stage_pilot(film: dict, state: dict) -> None:
    clips = film["clips"][:2]
    job = {
        "id": _job_id("pilot"),
        "shots": [_shot(film, c, state, frames=33, shot_id=f"pilot{i + 1}") for i, c in enumerate(clips)],
        "transition": {"type": "crossfade", "duration_seconds": 0.25},
    }
    _run_job(job, FILM_DIR / "pilot_crossfade.mp4", 1800)
    state["pilot_job"] = job["id"]
    save_state(state)
    _probe(FILM_DIR / "pilot_crossfade.mp4", 2 * 33 / film["format"]["fps"] - 0.25)


def stage_film(film: dict, state: dict) -> None:
    job = {
        "id": _job_id("film"),
        "shots": [_shot(film, c, state) for c in film["clips"]],
        "transition": {"type": "cut", "duration_seconds": 0.0},
    }
    (FILM_DIR / "film_request.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    _run_job(job, FILM_DIR / "film_sequence.mp4", 3 * 3600)
    state["film_job"] = job["id"]
    save_state(state)
    _probe(FILM_DIR / "film_sequence.mp4", film["total_seconds"])


def _probe(path: Path, expected_seconds: float | None) -> None:
    if not shutil.which("ffprobe") or os.environ.get("WAN_SKIP_PROBE"):
        log("probe_skipped", reason="ffprobe not installed locally or WAN_SKIP_PROBE set", file=str(path))
        return
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,codec_name,nb_frames",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    duration = float(info["format"]["duration"])
    log("probe", file=str(path), duration=duration, stream=info["streams"][0])
    if expected_seconds is not None:
        expect(f"{path.name} duration ~{expected_seconds:.2f}s", abs(duration - expected_seconds) < 1.0, f"{duration:.2f}")


def stage_clips(film: dict, state: dict) -> None:
    target, port, key = os.environ.get("WAN_SSH"), os.environ.get("WAN_SSH_PORT", "22"), os.environ.get("WAN_SSH_KEY")
    if not target or "film_job" not in state:
        log("clips_skipped", reason="WAN_SSH not set or film not run; fetch /workspace/outputs/<job>/*.mp4 by hand")
        return
    folder = FILM_DIR / "clips"
    folder.mkdir(exist_ok=True)
    for clip in film["clips"]:
        name = f"{clip['n']:02d}_{clip['id'].split('_', 1)[1]}.mp4"
        command = ["scp", "-P", port, "-o", "StrictHostKeyChecking=no"]
        if key:
            command += ["-i", key]
        command += [f"{target}:/workspace/outputs/{state['film_job']}/{clip['id']}.mp4", str(folder / name)]
        subprocess.run(command, check=True)
        log("clip_saved", n=clip["n"], file=str(folder / name))


DISPATCH = {name: globals()[f"stage_{name}"] for name in STAGES}


def main() -> None:
    global FILM_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=STAGES + ["all"])
    parser.add_argument("--film-dir", default="results/film_artemis_portrait")
    args = parser.parse_args()
    FILM_DIR = Path(args.film_dir)
    FILM_DIR.mkdir(parents=True, exist_ok=True)
    film, state = load_film(), load_state()
    for stage in (STAGES if "all" in args.stages else args.stages):
        log("stage_start", stage=stage, base=BASE)
        DISPATCH[stage](film, state)
        state = load_state()
        log("stage_end", stage=stage)


if __name__ == "__main__":
    sys.exit(main())
