"""The "rainy-sunny anime short shots" trend: a RANDOMISED sequence of short painted-anime shots (rain, sun, reflections in puddles, flowers, birds, trains,
calm streets, gardens), each a Krea2 still (anime style + LoRA) brought to life by Wan 2.2 image-to-video, rotated 90 degrees to fill a 576x1024 frame,
cut on the beat of a piece of music (scripts/music_timing.py).

What is random (seeded, so a video can be replayed): the CONTEXT (weather, time of day, setting, palette), the LIST of shots (archetypes drawn with weights and
constraints: never the same archetype twice in a row, a "wow" puddle-reflection shot when the weather allows it, a mix of macro / animal / vehicle / people /
wide shots) and every slot of every prompt (flower, bird, surface, light...). What is not: the style (anime painting, rich colour), the calm mood.

    python scripts/trend_rain_anime.py plan --seed 7 --shots 8 > plan.json
    python scripts/trend_rain_anime.py stills plan.json --style enhance --out stills/        (needs the API with the krea2 profile)
    python scripts/trend_rain_anime.py animate plan.json stills/ --out clips/                 (needs the API with the i2v-turbo profile)
    python scripts/trend_rain_anime.py assemble plan.json clips/ --music track.mp3 --timing track.timing.json --out trend.mp4

The API is read from WAN_BASE_URL (default http://127.0.0.1:8000) and WAN_API_TOKEN, like scripts/film_client.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402

BASE_URL = os.environ.get("WAN_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.environ.get("WAN_API_TOKEN", "")

# --- the picture format: shots are made LANDSCAPE and turned 90 degrees, so the whole width of the painting fills the tall frame ---
STILL_SIZE = (1024, 576)  # landscape, multiples of 16
FINAL_SIZE = (576, 1024)  # after the rotation (the reference video: 576x1024, 30 fps)
FINAL_FPS = 30
CLIP_FRAMES = 49  # 3 s at Wan's native 16 fps (4k + 1 frames)
CLIP_FPS = 16
ROTATION = "clockwise"  # which way the landscape picture is turned: "clockwise" | "counterclockwise"

# --- the styles to compare (all Krea2). weight 0 = prompt only ---
STYLE_BASE = ("Makoto Shinkai style anime background painting, hand-painted, extremely detailed, rich and varied vivid colours "
              "(emerald and leaf greens, teal, cobalt blue, warm gold, soft pink and lavender), luminous volumetric light, cinematic composition, "
              "clean crisp painted shapes, no text, no watermark")
STYLES = {
    "shinkai": {"label": "prompt only (Shinkai painting)", "loras": [], "prefix": ""},
    "enhance": {"label": "Anime Enhance LoRA 0.8", "loras": [("krea2_turbo_anime_enhance", 0.8)], "prefix": "anime style, "},
    "enhance_strong": {"label": "Anime Enhance LoRA 1.0", "loras": [("krea2_turbo_anime_enhance", 1.0)], "prefix": "anime style, "},
    "gurren": {"label": "Gurren Lagann LoRA 1.0", "loras": [("krea2_gurren_lagann_style", 1.0)], "prefix": "GurrenLagannStyle, "},
    "gurren_strong": {"label": "Gurren Lagann LoRA 1.6", "loras": [("krea2_gurren_lagann_style", 1.6)], "prefix": "GurrenLagannStyle, "},
    "enhance_gurren_18": {"label": "Anime Enhance 0.7 + Gurren Lagann 1.8", "loras": [("krea2_turbo_anime_enhance", 0.7), ("krea2_gurren_lagann_style", 1.8)],
                          "prefix": "anime style, GurrenLagannStyle, "},
    "enhance_gurren": {"label": "Anime Enhance 0.7 + Gurren Lagann 1.5", "loras": [("krea2_turbo_anime_enhance", 0.7), ("krea2_gurren_lagann_style", 1.5)],
                       "prefix": "anime style, GurrenLagannStyle, "},
}
DEFAULT_STYLE = "enhance"

# --- the random world ---
WEATHERS = {  # key: (weight, words used in every still prompt, words for motion, allows the puddle-reflection "wow")
    "rain_to_sun": (0.40, "light rain falling while bright sun breaks through the clouds, everything wet and glistening, sparkling reflections", "light rain falls, sunlight glitters", True),
    "light_rain": (0.20, "gentle steady rain, soft grey-blue sky with luminous clouds, wet surfaces", "rain falls gently", True),
    "after_rain": (0.28, "just after the rain, sun shining through clearing clouds, puddles everywhere, dripping leaves, fresh saturated colours", "water drips, light shimmers", True),
    "heavy_rain": (0.12, "heavy rain, mist and rain streaks, dramatic cool light", "heavy rain streaks fall", False),
}
TIMES = {"morning": "soft morning light", "afternoon": "bright afternoon light", "golden hour": "warm golden hour light", "dusk": "blue dusk light with warm lamps glowing"}
SETTINGS = {"nature": 0.45, "calm city": 0.35, "mixed": 0.20}
FLOWERS = ["pink cosmos", "blue hydrangea", "cherry blossom branches", "morning glory", "white lotus", "red spider lilies", "purple irises", "yellow rapeseed blossoms"]
BIRDS = ["small sparrow", "blue kingfisher", "white heron", "swallow", "wagtail", "dove"]
SURFACES = ["an old wooden bridge", "mossy stone steps", "wet cobblestones", "a wooden veranda", "a garden path", "an asphalt road"]
TRAINS = ["a green local train", "a yellow commuter train", "a vintage blue train"]
CITY_DETAILS = ["a pedestrian traffic light glowing green", "a red vending machine", "a bus stop with a shelter", "a street sign in a Japanese town", "a railway crossing with red lights"]
PLACES_NATURE = ["a japanese garden with a pond", "a bamboo grove", "a riverside path", "a green valley village", "a hillside with terraced fields", "a quiet park"]
PLACES_CITY = ["a quiet japanese street", "a small neighbourhood crossing", "a calm station platform", "a narrow alley with shops", "a riverside town"]


def _pick(rnd, seq):
    return seq[rnd.randrange(len(seq))]


# archetype: (weight, family, needs, still prompt (slots in {}), motion prompts)
ARCHETYPES = {
    "raindrop_puddle": dict(weight=1.0, family="macro", needs="any", wow=True, slots=["surface"],
        still="extreme close-up of a single raindrop hitting a puddle on {surface}, concentric ripples, the water reflecting the bright sky, white clouds and the sun",
        motion=["a raindrop falls into the puddle and concentric ripples spread outward, the reflection of the sky and clouds wobbles and glitters, static camera"]),
    "puddle_sky": dict(weight=1.0, family="wide", needs="any", wow=True, slots=["place"],
        still="low angle shot of a large puddle on the ground in {place}, a perfect mirror reflection of blue sky, white clouds, the sun and the surroundings upside down",
        motion=["soft ripples spread across the puddle, clouds drift slowly in the reflection, sparkling light on the water, slow gentle camera push-in"]),
    "dew_web": dict(weight=0.8, family="macro", needs="nature", wow=False, slots=[],
        still="a spider web covered in dew drops and raindrops, backlit by the sun, creamy bokeh of green foliage, tiny rainbow glints",
        motion=["water droplets glisten and tremble on the web, the web sways slightly in the breeze, light shimmers, tiny macro camera drift"]),
    "flowers": dict(weight=1.0, family="macro", needs="nature", wow=False, slots=["flower"],
        still="{flower} covered in raindrops, soft sunbeams and bokeh, dripping petals, delicate and colourful",
        motion=["the flowers sway gently in the breeze, raindrops slide down the petals and drip, slow camera drift"]),
    "bubbles_water": dict(weight=0.7, family="macro", needs="nature", wow=False, slots=[],
        still="clear shallow water in a stream with bubbles rising to the surface, sun rays piercing the water, pebbles and green plants below, sparkling light",
        motion=["bubbles rise and shimmer to the surface, ripples and sun rays dance on the stream bed, gentle camera drift"]),
    "bird": dict(weight=1.0, family="animal", needs="any", wow=False, slots=["bird"],
        still="a {bird} perched on a wet branch, seen from below against a bright sky, droplets falling, green leaves",
        motion=["the bird spreads its wings and takes flight, leaves tremble, droplets scatter", "the bird shakes off the water, ruffles its feathers and looks around, droplets fall"]),
    "train": dict(weight=0.9, family="vehicle", needs="city", wow=False, slots=["train"],
        still="{train} passing through a quiet suburb in the rain, a railway crossing with warning lights, wet road reflections, lush green hills behind",
        motion=["the train passes slowly from right to left, warning lights blink, rain falls, static camera"]),
    "umbrella_street": dict(weight=1.0, family="people", needs="city", wow=False, slots=["place"],
        still="{place} in the rain, a few people walking with colourful umbrellas seen from far away, wet ground reflecting the lights and sky",
        motion=["people walk slowly with their umbrellas, rain falls, reflections shimmer on the wet street, slow camera drift"]),
    "umbrella_back": dict(weight=0.7, family="people", needs="any", wow=False, slots=["place"],
        still="a person with a red umbrella seen from behind walking away along a wet path in {place}, serene and quiet",
        motion=["the person walks slowly away, rain falls softly, the umbrella sways gently"]),
    "aerial": dict(weight=0.8, family="aerial", needs="nature", wow=False, slots=["place"],
        still="bird's eye view of {place} with winding paths, lush trees and a small rainbow in the mist, lit by sun after rain",
        motion=["slow camera drift forward over the trees, the leaves sway, mist moves gently"]),
    "leaves": dict(weight=0.9, family="macro", needs="nature", wow=False, slots=[],
        still="large glossy green leaves with raindrops, sunlight filtering through, shallow depth of field, bright glowing foliage",
        motion=["the leaves sway in the breeze and raindrops fall and slide off, light flickers through the foliage"]),
    "city_detail": dict(weight=0.9, family="detail", needs="city", wow=False, slots=["detail"],
        still="{detail}, a wet street in a quiet Japanese town, glowing colours reflected on the ground, tall buildings and trees",
        motion=["the lights glow and flicker softly, rain drips, a car passes slowly in the background"]),
    "valley": dict(weight=0.9, family="wide", needs="nature", wow=False, slots=[],
        still="wide landscape of green hills and a river valley with a small town, low clouds and mist, shafts of sunlight",
        motion=["clouds drift slowly, mist rises from the valley, the grass sways, slow camera push-in"]),
    "window_rain": dict(weight=0.6, family="detail", needs="city", wow=False, slots=[],
        still="rain drops running down a window glass with blurred colourful city lights and green trees outside, cosy interior glow",
        motion=["raindrops slide down the glass, the blurred lights shimmer, static camera"]),
}
MIN_FAMILIES = 3  # a sequence mixes at least this many families


def plan_context(rnd: random.Random, setting: str | None = None, weather: str | None = None) -> dict:
    weathers = list(WEATHERS)
    weather = weather or rnd.choices(weathers, weights=[WEATHERS[w][0] for w in weathers])[0]
    setting = setting or rnd.choices(list(SETTINGS), weights=list(SETTINGS.values()))[0]
    time_of_day = _pick(rnd, list(TIMES))
    return {"weather": weather, "weather_words": WEATHERS[weather][1], "motion_weather": WEATHERS[weather][2], "wow_allowed": WEATHERS[weather][3],
            "time_of_day": time_of_day, "light_words": TIMES[time_of_day], "setting": setting}


def _allowed(name: str, context: dict) -> bool:
    spec = ARCHETYPES[name]
    if spec["needs"] == "city" and context["setting"] == "nature":
        return False
    if spec["needs"] == "nature" and context["setting"] == "calm city":
        return False
    if spec.get("wow") and not context["wow_allowed"]:
        return False
    return True


def _fill(rnd: random.Random, slot: str, context: dict) -> str:
    if slot == "surface":
        return _pick(rnd, SURFACES)
    if slot == "flower":
        return _pick(rnd, FLOWERS)
    if slot == "bird":
        return _pick(rnd, BIRDS)
    if slot == "train":
        return _pick(rnd, TRAINS)
    if slot == "detail":
        return _pick(rnd, CITY_DETAILS)
    if slot == "place":
        pool = PLACES_NATURE if context["setting"] == "nature" else PLACES_CITY if context["setting"] == "calm city" else PLACES_NATURE + PLACES_CITY
        return _pick(rnd, pool)
    raise KeyError(slot)


def plan_trend(seed: int, shots: int = 8, *, setting: str | None = None, weather: str | None = None, force: list[str] | None = None) -> dict:
    """A reproducible random plan: the context, then `shots` shots. `force` pins the first archetypes (tests, or a hand-made sequence)."""
    if shots < 1:
        raise ValueError("a trend needs at least one shot")
    rnd = random.Random(seed)
    context = plan_context(rnd, setting, weather)
    pool = [n for n in ARCHETYPES if _allowed(n, context)]
    chosen: list[str] = list(force or [])
    for name in chosen:
        if name not in ARCHETYPES:
            raise ValueError(f"unknown archetype {name!r}")
    wow = [n for n in pool if ARCHETYPES[n].get("wow")]
    while len(chosen) < shots:
        weights = [ARCHETYPES[n]["weight"] * (0.0 if chosen and n == chosen[-1] else 1.0) * (0.45 if n in chosen[-3:] else 1.0) for n in pool]
        chosen.append(rnd.choices(pool, weights=weights)[0])
    # the "wow" shot (a puddle reflecting the sky) is guaranteed when the weather allows it, in the second half of the sequence
    if wow and not any(n in wow for n in chosen) and shots >= 3:
        position = rnd.randrange(shots // 2, shots)
        chosen[position] = _pick(rnd, wow)
    # a sequence mixes families: swap repeated ones until there are enough kinds (when the pool has them)
    for _ in range(40):
        if len({ARCHETYPES[n]["family"] for n in chosen}) >= min(MIN_FAMILIES, len({ARCHETYPES[n]["family"] for n in pool}), shots):
            break
        position = rnd.randrange(shots)
        used = {ARCHETYPES[n]["family"] for n in chosen}
        options = [n for n in pool if ARCHETYPES[n]["family"] not in used]
        if options and not (force and position < len(force)):
            chosen[position] = _pick(rnd, options)
    plan_shots = []
    for index, name in enumerate(chosen):
        spec = ARCHETYPES[name]
        slots = {slot: _fill(rnd, slot, context) for slot in spec["slots"]}
        scene = spec["still"].format(**slots)
        plan_shots.append({
            "id": f"s{index + 1:02d}", "archetype": name, "family": spec["family"], "slots": slots,
            "scene": f"{scene}, {context['weather_words']}, {context['light_words']}",
            "motion": f"{_pick(rnd, spec['motion'])}, {context['motion_weather']}",
            "seed": rnd.randrange(2**31),
        })
    return {"seed": seed, "context": context, "shots": plan_shots}


def still_request(shot: dict, style: str = DEFAULT_STYLE, size: tuple[int, int] = STILL_SIZE) -> dict:
    """The POST /v1/images body for one shot in one style."""
    spec = STYLES[style]
    body = {"engine": "krea2", "prompt": f"{spec['prefix']}{shot['scene']}, {STYLE_BASE}", "width": size[0], "height": size[1], "seed": shot["seed"]}
    if spec["loras"]:
        body["loras"] = [{"id": lora, "weight": weight, "target": "keyframe"} for lora, weight in spec["loras"]]
    return body


def clip_job(shot: dict, image_id: str, job_id: str, size: tuple[int, int] = STILL_SIZE, frames: int = CLIP_FRAMES, fps: int = CLIP_FPS) -> dict:
    """The POST /v1/jobs body that animates one still (Wan 2.2 image-to-video, turbo)."""
    return {"id": job_id, "shots": [{
        "id": shot["id"], "mode": "i2v", "prompt": shot["motion"], "start_image": {"image_id": image_id},
        "width": size[0], "height": size[1], "frames": frames, "fps": fps, "seed": shot["seed"], "turbo_mode": True,
    }]}


# --- detail shots with as little shadow as possible (2026-10-05): a lighting phrase for the prompt, and a post-process that lifts the shadows ---
LOW_SHADOW = "soft even diffused daylight, no cast shadows, shadowless flat high-key lighting, bright clean colours, every detail clearly visible"


def lift_shadows(src: Path, dst: Path, strength: float = 0.6, sharpen: float = 0.4) -> Path:
    """Open the shadows of a picture to keep its details: a gamma below 1 (dark tones rise a lot, whites stay white), then a light unsharp mask.
    strength 0 = untouched; 0.6 = a clear lift; 1.0 = a strong, flat look."""
    import numpy as np
    from PIL import Image, ImageFilter

    if strength < 0:
        raise ValueError("strength must be >= 0")
    image = Image.open(src).convert("RGB")
    data = np.asarray(image, np.float32) / 255.0
    data = np.power(data, 1.0 / (1.0 + strength))
    lifted = Image.fromarray(np.clip(data * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
    if sharpen > 0:
        lifted = lifted.filter(ImageFilter.UnsharpMask(radius=2, percent=int(100 * sharpen), threshold=2))
    dst.parent.mkdir(parents=True, exist_ok=True)
    lifted.save(dst)
    return dst


def upload_still(path: Path) -> str:
    """Send a picture to the API (POST /v1/images/upload) and return its image_id, to animate a picture that was not generated there (a post-processed one)."""
    import httpx

    with httpx.Client(base_url=BASE_URL, headers=_headers(), timeout=120) as client:
        response = client.post("/v1/images/upload", files={"file": (Path(path).name, Path(path).read_bytes(), "image/png")})
        response.raise_for_status()
        return response.json()["image_id"]


# ------------------------------------------------------------------ the register of every generation (its seed above all)
def record_generation(registry: Path | None, entry: dict) -> None:
    """Append one generation to the register (a JSON list, rewritten atomically): kind, id, file, SEED, style, LoRAs with their weights, prompt, size,
    frames, duration. Nothing secret is ever in an entry. A generation can be replayed exactly with the same seed, prompt, LoRAs and size."""
    if registry is None:
        return
    registry.parent.mkdir(parents=True, exist_ok=True)
    items = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
    items.append({"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry})
    temp = registry.with_suffix(".tmp")
    temp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    temp.replace(registry)


def _headers() -> dict:
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


def _log(path: Path | None, **fields) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}) + "\n")


def run_still(request: dict, out_path: Path, *, log: Path | None = None, poll: float = 3.0, timeout: float = 1800.0,
              registry: Path | None = None, label: str | None = None) -> dict:
    """Generate one image through the API and save it; returns {"image_id", "seconds"}. Never gives up on a job just because it is slow: it
    raises only if the job FAILS or `timeout` (a long, generous bound) passes."""
    import httpx

    started = time.time()
    with httpx.Client(base_url=BASE_URL, headers=_headers(), timeout=120) as client:
        queued = client.post("/v1/images", json=request)
        queued.raise_for_status()
        image_id = queued.json()["image_id"]
        last_note = 0.0
        while True:
            job = client.get(f"/v1/images/{image_id}").json()
            if time.time() - last_note >= 30:
                last_note = time.time()
                _log(log, kind="poll", image_id=image_id, status=job["status"], progress=job.get("progress"), elapsed=round(time.time() - started))
            if job["status"] == "completed":
                break
            if job["status"] == "failed":
                _log(log, kind="still", image_id=image_id, status="failed", error=job.get("error"), request={k: v for k, v in request.items() if k != "prompt"})
                raise RuntimeError(f"image job {image_id} failed: {job.get('error')}")
            if time.time() - started > timeout:
                raise TimeoutError(f"image job {image_id} still {job['status']} after {timeout:.0f} s")
            time.sleep(poll)
        data = client.get(f"/v1/images/{image_id}/output")
        data.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data.content)
    seconds = round(time.time() - started, 1)
    _log(log, kind="still", image_id=image_id, status="completed", seconds=seconds, file=str(out_path), seed=request.get("seed"))
    record_generation(registry, {"kind": "still", "id": label or out_path.stem, "image_id": image_id, "file": str(out_path), "seed": request.get("seed"),
                                 "loras": request.get("loras", []), "prompt": request["prompt"], "width": request["width"], "height": request["height"],
                                 "engine": request.get("engine"), "steps": request.get("steps"), "seconds": seconds})
    return {"image_id": image_id, "seconds": seconds}


def run_clip(job: dict, out_path: Path, *, log: Path | None = None, poll: float = 5.0, timeout: float = 3600.0,
             registry: Path | None = None, label: str | None = None, source_image: str | None = None) -> dict:
    import httpx

    started = time.time()
    with httpx.Client(base_url=BASE_URL, headers=_headers(), timeout=120) as client:
        queued = client.post("/v1/jobs", json=job)
        queued.raise_for_status()
        job_id = job["id"]
        last_note = 0.0
        while True:
            state = client.get(f"/v1/jobs/{job_id}").json()
            if time.time() - last_note >= 30:  # a heartbeat every 30 s: the status and progress of the job, as the box's own monitor sees it
                last_note = time.time()
                _log(log, kind="poll", job_id=job_id, status=state["status"], progress=state.get("progress"), shot=state.get("current_shot"),
                     elapsed=round(time.time() - started))
            if state["status"] == "completed":
                break
            if state["status"] == "failed":
                _log(log, kind="clip", job_id=job_id, status="failed", error=state.get("error"))
                raise RuntimeError(f"job {job_id} failed: {state.get('error')}")
            if time.time() - started > timeout:
                raise TimeoutError(f"job {job_id} still {state['status']} after {timeout:.0f} s")
            time.sleep(poll)
        data = client.get(f"/v1/jobs/{job_id}/output")
        data.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data.content)
    seconds = round(time.time() - started, 1)
    shot = job["shots"][0]
    _log(log, kind="clip", job_id=job_id, status="completed", seconds=seconds, file=str(out_path), seed=shot.get("seed"))
    record_generation(registry, {"kind": "clip", "id": label or out_path.stem, "job_id": job_id, "file": str(out_path), "seed": shot.get("seed"),
                                 "source_image": source_image or (shot.get("start_image") or {}).get("image_id"), "prompt": shot.get("prompt"),
                                 "width": shot["width"], "height": shot["height"], "frames": shot["frames"], "fps": shot["fps"],
                                 "turbo_mode": shot.get("turbo_mode"), "seconds": seconds})
    return {"job_id": job_id, "seconds": seconds}


# ------------------------------------------------------------------ finishing and assembling (ffmpeg)
def rotate_filter(direction: str = ROTATION) -> str:
    if direction not in ("clockwise", "counterclockwise"):
        raise ValueError("rotation must be clockwise or counterclockwise")
    return "transpose=1" if direction == "clockwise" else "transpose=2"


def finish_filter(*, interpolate: bool = True, fps: int = FINAL_FPS, size: tuple[int, int] = FINAL_SIZE, direction: str = ROTATION) -> str:
    """landscape clip -> portrait: motion-interpolated to `fps`, turned 90 degrees, scaled to `size`, a touch of colour and a little grain"""
    steps = []
    if interpolate:
        steps.append(f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1:me_mode=bidir")
    else:
        steps.append(f"fps={fps}")
    steps += [rotate_filter(direction), f"scale={size[0]}:{size[1]}:flags=lanczos", "eq=saturation=1.08:contrast=1.04", "noise=alls=3:allf=t", "format=yuv420p"]
    return ",".join(steps)


def finish_clip(src: Path, dst: Path, *, seconds: float | None = None, **kwargs) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = [mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(src)]
    if seconds:
        command += ["-t", f"{seconds:.3f}"]
    command += ["-vf", finish_filter(**kwargs), "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium", str(dst)]
    subprocess.run(command, check=True)
    return dst


def shot_durations(cuts: list[float], total: float) -> list[float]:
    """Shot lengths from cut times: shot i lasts from cut i-1 to cut i (the first starts at 0, the last ends at `total`)."""
    edges = [0.0] + [c for c in sorted(cuts) if 0 < c < total] + [total]
    return [round(b - a, 4) for a, b in zip(edges, edges[1:]) if b - a > 1e-3]


def assemble(clips: list[Path], durations: list[float], out_path: Path, *, music: Path | None = None, music_offset: float = 0.0,
             speed: float = 1.0, work: Path | None = None) -> Path:
    """Cut `clips` (already finished: portrait, 30 fps) to `durations` one after the other (clips are reused in order when there are more shots than
    clips; a clip shorter than its slot holds its last frame) and lay the music under them, starting `music_offset` seconds into the track."""
    work = work or out_path.parent / "assemble_work"
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    for index, duration in enumerate(durations):
        src = clips[index % len(clips)]
        part = work / f"part_{index:03d}.mp4"
        subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(src), "-vf", f"tpad=stop_mode=clone:stop_duration={duration:.3f},fps={FINAL_FPS}",
                        "-t", f"{duration:.3f}", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(part)], check=True)
        parts.append(part)
    listing = work / "list.txt"
    listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    silent = work / "silent.mp4"
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(silent)], check=True)
    total = sum(durations)
    if music is None:
        shutil.copy2(silent, out_path)
    else:
        subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(silent), "-ss", f"{music_offset:.3f}", "-i", str(music),
                        "-filter_complex", f"[1:a]atempo={speed:.4f},afade=t=out:st={max(0.0, total - 1.0):.3f}:d=1.0[a]", "-map", "0:v", "-map", "[a]",
                        "-t", f"{total:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out_path)], check=True)
    return out_path


RECIPES_FILE = Path(__file__).resolve().parent / "assets" / "rain_anime_recipes.json"


def load_recipes() -> dict:
    """The frozen recipes: styles, finishing, post-process levels and every shot that was validated (or rejected) with all its seeds."""
    return json.loads(RECIPES_FILE.read_text(encoding="utf-8"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="print a random plan (JSON)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shots", type=int, default=8)
    p.add_argument("--setting", choices=list(SETTINGS))
    p.add_argument("--weather", choices=list(WEATHERS))
    s = sub.add_parser("stills", help="generate the stills of a plan through the API")
    s.add_argument("plan", type=Path)
    s.add_argument("--style", choices=list(STYLES), default=DEFAULT_STYLE)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--log", type=Path)
    c = sub.add_parser("animate", help="animate the stills of a plan (image to video through the API)")
    c.add_argument("plan", type=Path)
    c.add_argument("stills", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--log", type=Path)
    c.add_argument("--no-finish", action="store_true", help="keep the raw landscape 16 fps clips")
    a = sub.add_parser("assemble", help="cut the finished clips on the music's timing")
    a.add_argument("plan", type=Path)
    a.add_argument("clips", type=Path)
    a.add_argument("--music", type=Path)
    a.add_argument("--timing", type=Path, help="timing JSON saved by scripts/music_timing_server.py")
    a.add_argument("--seconds", type=float, default=20.0)
    a.add_argument("--speed", type=float, default=1.0)
    a.add_argument("--music-offset", type=float, default=None)
    a.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("recipes", help="list the frozen recipes (validated / rejected / reference) of scripts/assets/rain_anime_recipes.json")
    r.add_argument("--status", choices=["validated", "rejected", "reference"])
    y = sub.add_parser("replay", help="regenerate a frozen recipe EXACTLY (same prompt, LoRAs, size, seeds) through the API; --seed/--clip-seed try a new seed (recorded)")
    y.add_argument("name")
    y.add_argument("--out", type=Path, required=True)
    y.add_argument("--seed", type=int, help="new still seed (default: the recipe's)")
    y.add_argument("--clip-seed", type=int, help="new clip seed (default: the recipe's)")
    y.add_argument("--log", type=Path)
    y.add_argument("--registry", type=Path, help="generation register to append to (seeds, LoRAs, prompt)")
    q = sub.add_parser("post", help="drops + calmer far ripples + speed-up on a finished clip (scripts/rain_drop_post.py); defaults to the reference clip's impacts")
    q.add_argument("clip", type=Path)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--level", choices=["normal", "exaggerated"], default="normal")
    q.add_argument("--impacts", help="impacts JSON, or 'none' when the clip has no impact map")
    q.add_argument("--seed", type=int, help="seed of the drops (default 1)")
    args = parser.parse_args(argv)
    if args.command == "recipes":
        for name, item in load_recipes()["recipes"].items():
            if args.status in (None, item["status"]):
                print(f"{item['status']:10} {name:34} still seed {item['still']['seed']:>10}  clip seed {item['clip']['seed']:>10}  {item['style']}  {item['note'][:70]}")
        return
    if args.command == "replay":
        recipe = load_recipes()["recipes"][args.name]
        body = {**recipe["still"], **({"seed": args.seed} if args.seed is not None else {})}
        label = args.name + (f"_s{args.seed}" if args.seed is not None else "") + (f"_c{args.clip_seed}" if args.clip_seed is not None else "")
        still = run_still(body, args.out / f"{label}.png", log=args.log, registry=args.registry, label=label)
        clip = recipe["clip"]
        shot = {"id": "s01", "motion": clip["prompt"], "seed": args.clip_seed if args.clip_seed is not None else clip["seed"]}
        raw = args.out / f"{label}_raw.mp4"
        run_clip(clip_job(shot, still["image_id"], f"replay-{label}-{int(time.time())}", frames=clip["frames"], fps=clip["fps"]), raw, log=args.log,
                 registry=args.registry, label=label + "_clip", source_image=still["image_id"])
        print("wrote", finish_clip(raw, args.out / f"{label}.mp4"))
        return
    if args.command == "post":
        import rain_drop_post as rp
        overrides = {"level": args.level, **({"seed": args.seed} if args.seed is not None else {})}
        impacts = None if args.impacts == "none" else Path(args.impacts or rp.REFERENCE_IMPACTS)
        print(json.dumps({k: v for k, v in rp.process(args.clip, args.out, impacts, overrides).items() if k not in ("drops", "impacts")}, indent=1))
        return
    if args.command == "plan":
        print(json.dumps(plan_trend(args.seed, args.shots, setting=args.setting, weather=args.weather), indent=1))
        return
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.command == "stills":
        for shot in plan["shots"]:
            result = run_still(still_request(shot, args.style), args.out / f"{shot['id']}_{args.style}.png", log=args.log)
            print(shot["id"], shot["archetype"], result, flush=True)
    elif args.command == "animate":
        for shot in plan["shots"]:
            still = next(args.stills.glob(f"{shot['id']}_*.json"), None)
            image_id = json.loads(still.read_text())["image_id"] if still else None
            if not image_id:
                raise SystemExit(f"no image id recorded for {shot['id']} in {args.stills}")
            raw = args.out / f"{shot['id']}_raw.mp4"
            result = run_clip(clip_job(shot, image_id, f"{plan['seed']}-{shot['id']}-{int(time.time())}"), raw, log=args.log)
            if not args.no_finish:
                finish_clip(raw, args.out / f"{shot['id']}.mp4")
            print(shot["id"], result, flush=True)
    else:
        timing = json.loads(args.timing.read_text(encoding="utf-8")) if args.timing else None
        cuts = mt.expand_cut_times(timing, args.seconds, speed=args.speed, music_offset=args.music_offset) if timing else [i * args.seconds / len(plan["shots"]) for i in range(1, len(plan["shots"]))]
        durations = shot_durations(cuts, args.seconds)
        clips = sorted(args.clips.glob("s*.mp4"))
        clips = [c for c in clips if not c.stem.endswith("_raw")]
        offset = args.music_offset if args.music_offset is not None else (timing["loop"]["start"] if timing and timing.get("loop") else 0.0)
        assemble(clips, durations, args.out, music=args.music, music_offset=offset, speed=args.speed)
        print("wrote", args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
