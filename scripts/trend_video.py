"""A short rain-and-sun anime video built like the reference trend (about 20 shots of 0.4-2.1 s, jump cuts on the beat, lots of scenery, rain, views from above that move away), PLANNED AT RANDOM
from what is available and feasible, then made phase by phase (one model family at a time on the server) and assembled on the music. The system is a catalogue + a seeded planner + idempotent phases,
so any new archetype, camera, style, library clip or source can be added to the catalogue and will be drawn by the planner.

SOURCES of a shot (what is feasible, with its GPU cost):
  wan       Krea2 still in one of the validated styles + Wan 2.2 image-to-video turbo (about 17 s + 53 s)
  zoomout   Krea2 still only; a view from above that MOVES AWAY is made on the PC by jump cuts at decreasing zooms (about 17 s, no video model, always works)
  light     the same place at another hour: the still of a `wan` shot edited by Qwen-Image-Edit (dawn / sunset / night, composition kept), then animated (about 30 s + 53 s)
  library   a clip already made and validated (no GPU): the reference rain clips, the puddle with impacts, the scenes of earlier phases
and a CPU `rain` overlay (falling glassy drops with trails, scripts/rain_drop_post.py style general) on some of them.

    python scripts/trend_video.py plan --seed 7 [--seconds 20] [--setting mixed] [--library-share 0.35] [--gpu-budget 1500]      # writes the plan JSON, prints it
    python scripts/trend_video.py status PLAN                                                                                    # what is missing, per phase / model family
    python scripts/trend_video.py run PLAN --phase stills|qwen|clips|local|assemble [--comfy-url http://127.0.0.1:18188]
    scripts/trend_video_driver.sh PLAN                                                                                           # restarts the families in order and runs the phases
Seeds, prompts and LoRAs of every generation go in results/trend_rain_anime/generations.json (done by the helpers of trend_rain_anime.py / comfy_edit.py).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402
import trend_rain_anime as tr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "trend_rain_anime"
REGISTRY = RESULTS / "generations.json"
MUSIC = RESULTS / "music" / "reference_trend.mov"
FPS = 30
SIZE = (576, 1024)
EXCLUDED_FAMILIES = {"people", "animal"}  # no people in this video, no bird (the user's choices); a parameter, not a rule of the system

# ----------------------------------------------------------------------------- the catalogue
POV = "high angle point of view looking down diagonally toward the lower right, camera tilted, the puddle fills the whole picture"
PUDDLE = ("a close-up of a puddle on wet pavement, only the puddle and its reflection of bright blue sky, white and pink clouds and the sun, bright sunlight breaking through the rain, "
          "sparkling reflections")
DEPTH = ("the splashes are small and get smaller and less detailed the farther they are from the camera, the close ones sharp, the far ones tiny and soft, strong depth of field, the water "
         "between them stays calm with only faint thin ripples")
CRATERS_MOTION = ("many small raindrops keep falling into the puddle, small craters pop up and fade with only faint thin ripples, the reflection of the sky and clouds shimmers softly, "
                  "static camera, light rain")
EXTRA = {  # added to the archetypes of trend_rain_anime.py; `camera`: how the shot moves (wan = the video model, zoomout = jump cuts on the PC); `style`: a style that is known to work for it
    "puddle_craters": dict(weight=1.5, family="puddle", needs="any", wow=True, slots=[], camera="wan", style="enhance_gurren",
                           still=f"{POV}, {PUDDLE}, many small round raindrop craters with a dark hollow centre and a bright highlight, glossy drops frozen in the air, {DEPTH}",
                           motion=[CRATERS_MOTION]),
    "aerial_away": dict(weight=1.3, family="aerial", needs="any", wow=False, slots=["place"], camera="zoomout",
                        still="straight-down bird's eye view from very high above of {place}, winding paths, lush trees, small rooftops and a faint rainbow in the mist, lit by sun after the rain",
                        motion=["the camera rises straight up and pulls far away from above, the landscape shrinking below"]),
    "street_empty": dict(weight=0.9, family="detail", needs="city", wow=False, slots=[], camera="wan",
                         still="a quiet city street crossing with shops, wet pavement reflecting the sky, deserted, nobody around",
                         motion=["rain falls softly, reflections shimmer on the wet street, the leaves of the trees tremble slightly, static camera"]),
    "station_empty": dict(weight=0.8, family="detail", needs="city", wow=False, slots=[], camera="wan",
                          still="an empty railway station platform with a train waiting and puddles reflecting the sky, deserted, nobody around",
                          motion=["rain falls, the lights of the train glow, reflections ripple in the puddles, static camera"]),
}
CATALOG = {**{k: {**v, "camera": v.get("camera", "wan")} for k, v in tr.ARCHETYPES.items()}, **EXTRA}
STYLE_BY_FAMILY = {  # the styles that worked, per kind of shot (the user's verdicts), drawn with these weights
    "macro": [("enhance", 0.5), ("enhance_gurren", 0.3), ("enhance_gurren_18", 0.2)],
    "wide": [("enhance", 0.5), ("enhance_gurren", 0.3), ("shinkai", 0.2)],
    "aerial": [("enhance", 0.6), ("shinkai", 0.2), ("enhance_gurren", 0.2)],
    "detail": [("enhance_gurren_18", 0.6), ("enhance", 0.25), ("enhance_gurren", 0.15)],
    "vehicle": [("enhance_gurren_18", 0.6), ("enhance", 0.25), ("enhance_gurren", 0.15)],
    "puddle": [("enhance_gurren", 0.65), ("gurren", 0.15), ("enhance", 0.2)],
}
LIBRARY = [  # clips already made (finished, 576x1024, 30 fps) and validated by the user or kept as references; used when the file exists
    ("k_craters", "phaseK/k_craters.mp4", "puddle", "the reference rain clip (small craters scaled by depth)"),
    ("c_big", "phaseC/c_big_enhance_gurren_2.mp4", "puddle", "the first reference puddle (big craters with a highlight)"),
    ("union_craters_crown", "phaseQ/q_k_craters_crown_burst_blue.mp4", "puddle", "k_craters + crown impacts with pencil outline and burst"),
    ("train_rain", "phaseF/e_train_enhance_gurren_18_1.mp4", "vehicle", "a train in the rain"),
    ("sign", "phaseF/e_sign_enhance_gurren_18_1.mp4", "detail", "a very detailed street sign in the rain"),
    ("corner", "phaseF/e2_corner_overcast.mp4", "detail", "a street corner in the rain"),
    ("street_empty", "phaseS/s_rainy_street.mp4", "detail", "a deserted rainy street crossing"),
    ("station_empty", "phaseS/s_station.mp4", "detail", "a deserted rainy station platform"),
    ("dew_web", "phaseS/s_dew_web.mp4", "macro", "a dewy spider web"),
    ("cosmos", "phaseS/s_cosmos.mp4", "macro", "cosmos flowers in the rain"),
    ("leaves", "phaseS/s_leaves.mp4", "macro", "wet leaves with sun rays"),
    ("valley", "phaseS/s_valley.mp4", "wide", "a river valley after the rain"),
]
LIBRARY_ALIAS = {"train_rain": "train", "sign": "city_detail", "corner": "city_detail", "cosmos": "flowers", "dew_web": "dew_web", "valley": "valley", "leaves": "leaves",
                 "street_empty": "street_empty", "station_empty": "station_empty", "k_craters": "puddle_craters", "c_big": "puddle_craters", "union_craters_crown": "puddle_craters",
                 "rain_chain": "puddle_craters"}  # which archetype of the catalogue a library clip stands for (so the planner does not draw the same kind twice in a row)
LIGHT_EDITS = ["dawn", "sunset", "night"]
COST = {"still": 17.0, "clip": 53.0, "qwen": 30.0, "restart": 120.0}  # seconds of GPU, measured on the 4090 (novram)


def library_available() -> list[dict]:
    items = []
    for item_id, rel, family, note in LIBRARY:
        path = RESULTS / rel
        if path.exists() and path.stat().st_size > 100_000:
            items.append({"id": item_id, "file": rel, "family": family, "note": note})
    return items


# ----------------------------------------------------------------------------- the plan
def beat_grid(music: Path = MUSIC) -> dict:
    """Tempo, beat length and the first downbeat of the music (automatic analysis: confirm by ear or tap the real timing with scripts/music_timing_server.py)."""
    if music.exists():
        analysis = mt.analyze_music(str(music))
        first = (analysis["downbeats"] or analysis["beats"] or [0.0])[0]
        return {"file": str(music.relative_to(ROOT)).replace("\\", "/"), "bpm": analysis["bpm"], "beat": 60.0 / analysis["bpm"], "offset": float(first)}
    return {"file": None, "bpm": 114.0, "beat": 60.0 / 114.0, "offset": 0.0}


def cut_frames(rnd: random.Random, seconds: float, grid: dict) -> list[int]:
    """Frames of every cut: on the beat grid like the reference (a few longer shots at the start, then bursts), the sum is exactly seconds * FPS."""
    total_beats = max(8, round(seconds / grid["beat"]))
    beats = [2, 2, 2, 4]  # the reference opens with 1.0, 1.0, 1.3, 2.1 s then cuts faster
    while sum(beats) < total_beats:
        beats.append(rnd.choices([1, 2, 3, 4], weights=[0.55, 0.30, 0.10, 0.05])[0])
    while sum(beats) > total_beats:  # trim the end
        over = sum(beats) - total_beats
        if beats[-1] > over:
            beats[-1] -= over
        else:
            beats.pop()
    scale = seconds / (total_beats * grid["beat"])
    edges, cumulative = [0], 0.0
    for count in beats:
        cumulative += count * grid["beat"] * scale
        edges.append(round(cumulative * FPS))
    edges[-1] = round(seconds * FPS)
    return [b - a for a, b in zip(edges, edges[1:])]


def _allowed(name: str, context: dict) -> bool:
    spec = CATALOG[name]
    if spec["family"] in EXCLUDED_FAMILIES:
        return False
    if spec["needs"] == "city" and context["setting"] == "nature":
        return False
    if spec["needs"] == "nature" and context["setting"] == "calm city":
        return False
    if spec.get("wow") and not context["wow_allowed"]:
        return False
    return True


def _style(rnd: random.Random, name: str) -> str:
    spec = CATALOG[name]
    if spec.get("style") and rnd.random() < 0.7:  # the style known to work for this shot, most of the time
        return spec["style"]
    options = STYLE_BY_FAMILY[spec["family"]]
    return rnd.choices([s for s, _ in options], weights=[w for _, w in options])[0]


def _shot(rnd: random.Random, name: str, context: dict, index: int) -> dict:
    spec = CATALOG[name]
    slots = {slot: tr._fill(rnd, slot, context) for slot in spec["slots"]}
    scene = spec["still"].format(**slots)
    return {"id": f"u{index:02d}", "source": "zoomout" if spec["camera"] == "zoomout" else "wan", "archetype": name, "family": spec["family"], "slots": slots,
            "style": _style(rnd, name), "scene": f"{scene}, {context['weather_words']}, {context['light_words']}",
            "motion": f"{tr._pick(rnd, spec['motion'])}, {context['motion_weather']}", "seed": rnd.randrange(2**31)}


def estimate(shots: list[dict]) -> dict:
    stills = sum(1 for s in shots if s["source"] in ("wan", "zoomout"))
    clips = sum(1 for s in shots if s["source"] in ("wan", "light"))
    qwen = sum(1 for s in shots if s["source"] == "light")
    families = 1 + (1 if qwen else 0) + (1 if clips else 0)
    seconds = stills * COST["still"] + clips * COST["clip"] + qwen * COST["qwen"] + families * COST["restart"]
    return {"stills": stills, "clips": clips, "qwen_edits": qwen, "model_families": families, "gpu_seconds": round(seconds)}


def make_plan(seed: int, seconds: float = 20.0, *, setting: str = "mixed", weather: str | None = None, library_share: float = 0.35, light_variants: int = 2, max_chains: int = 2,
              gpu_budget: float = 1500.0, music: Path = MUSIC) -> dict:
    """A reproducible random plan: music grid and cut lengths, context (weather, hour, setting), then the shots drawn from the catalogue and the library, the jump-cut chains seen from above
    that move away, the light twins, the rain overlays, kept inside the GPU budget."""
    rnd = random.Random(seed)
    grid = beat_grid(music)
    frames = cut_frames(rnd, seconds, grid)
    n = len(frames)
    context = tr.plan_context(rnd, setting, weather)
    pool = [name for name in CATALOG if _allowed(name, context) and CATALOG[name]["camera"] != "zoomout"]
    zoom_names = [name for name in CATALOG if _allowed(name, context) and CATALOG[name]["camera"] == "zoomout"]
    library = library_available()
    rnd.shuffle(library)
    chain_len = [rnd.choice([3, 4]) for _ in range(min(max_chains, 2 if zoom_names else 0))]
    starts = []
    if chain_len:
        starts.append(rnd.randint(3, max(3, n // 2 - chain_len[0])))
    if len(chain_len) > 1:
        starts.append(rnd.randint(n // 2 + 1, max(n // 2 + 1, n - chain_len[1] - 1)))
    shots, cuts, index, last = [], [], 1, None
    used: Counter = Counter()
    while len(cuts) < n:
        position = len(cuts)
        remaining = n - position
        if starts and position >= starts[0] and remaining >= chain_len[0]:
            length = chain_len.pop(0)
            starts.pop(0)
            shot = _shot(rnd, rnd.choice(zoom_names), context, index)
            k_levels = np.geomspace(rnd.uniform(2.0, 2.6), 1.12, length)  # each cut is a JUMP to a wider view; the last one ends on the widest picture (zoom 1.0)
            shot["focus"] = [round(rnd.uniform(-0.12, 0.12), 3), round(rnd.uniform(-0.12, 0.12), 3)]
            shot["levels"] = [round(float(z), 3) for z in k_levels]
            shots.append(shot)
            for k in range(length):
                level = shot["levels"][k]
                cuts.append({"shot": shot["id"], "frames": frames[len(cuts)], "zoom": [level, round(level * 0.92, 3)] if k < length - 1 else [level, 1.0]})
            used[shot["archetype"]] += 1
            index += 1
            continue
        pick_library = library and rnd.random() < library_share
        if pick_library:
            fresh = [x for x in library if used[LIBRARY_ALIAS.get(x["id"], x["id"])] == 0] or library  # not the same kind of shot as one already drawn, while there is a choice
            item = fresh[-1]
            library.remove(item)
            shot = {"id": f"u{index:02d}", "source": "library", "archetype": item["id"], "family": item["family"], "library": item["file"], "note": item["note"], "seed": None}
            used[LIBRARY_ALIAS.get(item["id"], item["id"])] += 1
        else:
            choices = [n_ for n_ in pool if n_ != last]
            weights = [CATALOG[n_]["weight"] * (0.4 ** used[n_]) for n_ in choices]  # what was already drawn comes back less often
            shot = _shot(rnd, rnd.choices(choices, weights=weights)[0], context, index)
            last = shot["archetype"]
            used[last] += 1
            if shot["family"] in ("wide", "aerial", "detail", "vehicle") and rnd.random() < 0.5:
                shot["rain_overlay"] = True
        shots.append(shot)
        cuts.append({"shot": shot["id"], "frames": frames[len(cuts)]})
        index += 1
    # the "wow" of the reference: a puddle reflecting the sky, guaranteed in the second half
    if not any(s["family"] == "puddle" for s in shots) and context["wow_allowed"]:
        choice = rnd.choice([s for s in shots[n // 2:] if s["source"] != "zoomout"] or shots)
        replacement = _shot(rnd, "puddle_craters", context, int(choice["id"][1:]))
        shots[shots.index(choice)] = replacement
        for cut in cuts:
            if cut["shot"] == choice["id"]:
                cut["shot"] = replacement["id"]
    # light twins: the same place at another hour, right after the shot, taking the place of a one-cut shot
    twins = 0
    for i in range(len(shots) - 1):  # the LIVE list: a twin is never made from a shot that was replaced
        if twins >= light_variants:
            break
        shot, nxt = shots[i], shots[i + 1]
        if shot["source"] == "wan" and shot["family"] in ("wide", "detail", "vehicle", "puddle") and nxt["source"] in ("wan", "library") and rnd.random() < 0.7:
            twin = {"id": shot["id"] + "L", "source": "light", "archetype": shot["archetype"], "family": shot["family"], "base": shot["id"], "edit": rnd.choice(LIGHT_EDITS),
                    "style": shot["style"], "motion": shot["motion"], "seed": rnd.randrange(2**31)}
            for cut in cuts:
                if cut["shot"] == nxt["id"]:
                    cut["shot"] = twin["id"]
            shots[i + 1] = twin
            twins += 1
    # GPU budget: the most expensive generated shots become library clips while the library has some left
    spare = [x for x in library_available() if x["file"] not in {s.get("library") for s in shots}]
    while estimate(shots)["gpu_seconds"] > gpu_budget and spare:
        victims = [s for s in shots if s["source"] in ("wan", "light") and s["family"] != "puddle"]
        if not victims:
            break
        victim = rnd.choice(victims)
        item = spare.pop()
        shots[shots.index(victim)] = {"id": victim["id"], "source": "library", "archetype": item["id"], "family": item["family"], "library": item["file"], "note": item["note"], "seed": None}
    for cut in cuts:  # where in its clip a cut starts (the clips last 3 s)
        shot = next(s for s in shots if s["id"] == cut["shot"])
        if shot["source"] != "zoomout":
            length = 3.0 if shot["source"] != "library" else _clip_seconds(RESULTS / shot["library"])
            cut["t0"] = round(rnd.uniform(0.0, max(0.0, length - cut["frames"] / FPS - 0.1)), 2)
    return {"name": f"trend_{seed}", "seed": seed, "seconds": seconds, "fps": FPS, "size": list(SIZE), "context": context, "music": grid, "cuts": cuts, "shots": shots,
            "estimate": estimate(shots), "excluded_families": sorted(EXCLUDED_FAMILIES), "styles_used": sorted({s["style"] for s in shots if s.get("style")})}


def _clip_seconds(path: Path) -> float:
    text = subprocess.run([mt.ffmpeg_binary(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    stamp = text.split("Duration:")[1].split(",")[0].strip()
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ----------------------------------------------------------------------------- the phases
def run_dir(plan: dict) -> Path:
    path = RESULTS / "video_runs" / plan["name"]
    (path / "cuts").mkdir(parents=True, exist_ok=True)
    return path


def needed(plan: dict) -> dict[str, list[str]]:
    """What is still missing, per phase (a phase = one model family, so the driver knows which restarts it needs)."""
    d = run_dir(plan)
    missing: dict[str, list[str]] = {"stills": [], "qwen": [], "clips": [], "local": [], "assemble": []}
    for s in plan["shots"]:
        sid = s["id"]
        if s["source"] in ("wan", "zoomout") and not (d / f"{sid}.png").exists():
            missing["stills"].append(sid)
        if s["source"] == "light" and not (d / f"{sid}.png").exists():
            missing["qwen"].append(sid)
        if s["source"] in ("wan", "light") and not (d / f"{sid}.mp4").exists():
            missing["clips"].append(sid)
        if s.get("rain_overlay") and not (d / f"{sid}_rain.mp4").exists() and (d / f"{sid}.mp4").exists():
            missing["local"].append(sid)
    for index, cut in enumerate(plan["cuts"]):
        shot = next(s for s in plan["shots"] if s["id"] == cut["shot"])
        if shot["source"] == "zoomout" and not (d / "cuts" / f"c{index:02d}.mp4").exists() and (d / f"{shot['id']}.png").exists():
            missing["local"].append(f"c{index:02d}")
    if not (d / f"{plan['name']}.mp4").exists():
        missing["assemble"].append(plan["name"])
    return missing


def phase_stills(plan: dict) -> list[str]:
    d = run_dir(plan)
    failed: list[str] = []
    for s in plan["shots"]:
        out = d / f"{s['id']}.png"
        if s["source"] not in ("wan", "zoomout") or out.exists():
            continue
        try:
            result = tr.run_still(tr.still_request(s, s["style"]), out, log=RESULTS / "logs" / "jobs.jsonl", registry=REGISTRY, label=f"{plan['name']}_{s['id']}")
        except Exception as exc:  # a failed still is reported, the rest goes on (the plan can be rerun)
            print("FAILED still", s["id"], type(exc).__name__, str(exc)[:200], flush=True)
            failed.append(s["id"])
            continue
        (d / f"{s['id']}.json").write_text(json.dumps({**result, "shot": s}), encoding="utf-8")
        print(f"still {s['id']} {s['archetype']} [{s['style']}] {result['seconds']} s seed {s['seed']}", flush=True)
    return failed


def phase_qwen(plan: dict, comfy_url: str) -> list[str]:
    import comfy_edit as ce
    d = run_dir(plan)
    failed: list[str] = []
    for s in plan["shots"]:
        out = d / f"{s['id']}.png"
        base = d / f"{s.get('base', '')}.png"
        if s["source"] != "light" or out.exists() or not base.exists():
            continue
        try:
            entry = ce.run_edit(base, out, ce.LIGHT[s["edit"]], seed=1, registry=REGISTRY, label=f"{plan['name']}_{s['id']}", base_url=comfy_url)
        except Exception as exc:
            print("FAILED qwen", s["id"], type(exc).__name__, str(exc)[:200], flush=True)
            failed.append(s["id"])
            continue
        print(f"qwen {s['id']} ({s['edit']}) {entry['seconds']} s", flush=True)
    return failed


def phase_clips(plan: dict) -> list[str]:
    d = run_dir(plan)
    failed: list[str] = []
    for s in plan["shots"]:
        png, final = d / f"{s['id']}.png", d / f"{s['id']}.mp4"
        if s["source"] not in ("wan", "light") or final.exists() or not png.exists():
            continue
        raw = d / f"{s['id']}_raw.mp4"
        try:
            image_id = tr.upload_still(png)
            result = tr.run_clip(tr.clip_job(s, image_id, f"{plan['name']}-{s['id']}-{int(time.time())}"), raw, log=RESULTS / "logs" / "jobs.jsonl", registry=REGISTRY,
                                 label=f"{plan['name']}_{s['id']}_clip", source_image=image_id)
            tr.finish_clip(raw, final)
        except Exception as exc:
            print("FAILED clip", s["id"], type(exc).__name__, str(exc)[:200], flush=True)
            failed.append(s["id"])
            continue
        print(f"clip {s['id']} {s['archetype']} {result['seconds']} s seed {s['seed']}", flush=True)
    return failed


def render_zoom_cut(still: Path, out: Path, frames: int, z0: float, z1: float, focus: tuple[float, float] = (0.0, 0.0)) -> None:
    """A view from above that moves away: the still (turned like every shot) is cropped around its centre, the crop grows from zoom z0 to z1 during the cut (z0 > z1 = moving away)."""
    image = Image.open(still).convert("RGB").transpose(Image.Transpose.ROTATE_270)
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    width, height = image.size
    command = [mt.ffmpeg_binary(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "-", "-vf",
               "eq=saturation=1.08:contrast=1.04,noise=alls=3:allf=t,format=yuv420p", "-c:v", "libx264", "-crf", "14", "-preset", "veryfast", str(out)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for f in range(frames):
        t = f / max(1, frames - 1)
        zoom = max(1.0, math.exp(math.log(z0) + (math.log(z1) - math.log(z0)) * t))  # never wider than the picture
        cw, ch = width / zoom, height / zoom
        cx = width / 2 + focus[0] * width * (1 - 1 / zoom)
        cy = height / 2 + focus[1] * height * (1 - 1 / zoom)
        left, top = min(max(0.0, cx - cw / 2), width - cw), min(max(0.0, cy - ch / 2), height - ch)
        frame = image.resize(SIZE, Image.BICUBIC, box=(left, top, left + cw, top + ch))
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()


def phase_local(plan: dict) -> list[str]:
    import rain_drop_post as rp
    d = run_dir(plan)
    for s in plan["shots"]:
        src, out = d / f"{s['id']}.mp4", d / f"{s['id']}_rain.mp4"
        if s.get("rain_overlay") and src.exists() and not out.exists():
            rp.process(src, out, None, {"style": "general", "speed": 1.0, "seed": s["seed"] % 1000 or 1})
            print(f"rain overlay {s['id']}", flush=True)
    for index, cut in enumerate(plan["cuts"]):
        shot = next(s for s in plan["shots"] if s["id"] == cut["shot"])
        out = d / "cuts" / f"c{index:02d}.mp4"
        if shot["source"] == "zoomout" and not out.exists() and (d / f"{shot['id']}.png").exists():
            render_zoom_cut(d / f"{shot['id']}.png", out, cut["frames"], cut["zoom"][0], cut["zoom"][1], tuple(shot["focus"]))
            print(f"zoom cut {index:02d} {shot['id']} {cut['zoom']}", flush=True)
    return []


def _source_of(plan: dict, cut: dict, index: int) -> tuple[Path | None, float]:
    d = run_dir(plan)
    shot = next(s for s in plan["shots"] if s["id"] == cut["shot"])
    if shot["source"] == "zoomout":
        return d / "cuts" / f"c{index:02d}.mp4", 0.0
    if shot["source"] == "library":
        return RESULTS / shot["library"], cut["t0"]
    rain = d / f"{shot['id']}_rain.mp4"
    return (rain if rain.exists() else d / f"{shot['id']}.mp4"), cut["t0"]


def phase_assemble(plan: dict, allow_fallback: bool = False) -> dict:
    """Cut every shot (start t0, exact number of frames), join them, add the music from the first downbeat with a fade out. A shot whose clip is missing is replaced by a library clip."""
    d = run_dir(plan)
    segments = []
    spare = [RESULTS / x["file"] for x in library_available()]
    for index, cut in enumerate(plan["cuts"]):
        src, t0 = _source_of(plan, cut, index)
        if src is None or not src.exists():
            if not allow_fallback:
                missing = [f"cut {i:02d} ({c['shot']})" for i, c in enumerate(plan["cuts"]) if not (_source_of(plan, c, i)[0] or Path("/nonexistent")).exists()]
                raise SystemExit(f"assembly refused: {len(missing)} clip(s) missing, a video is never built from stand-ins unless --allow-fallback: {', '.join(missing[:8])}")
            src, t0 = spare[index % len(spare)], 0.2
            print(f"cut {index:02d}: clip of {cut['shot']} missing, library clip used (--allow-fallback)", flush=True)
        seg = d / "cuts" / f"seg{index:02d}.mp4"
        subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-ss", f"{t0}", "-i", str(src), "-vf", f"fps={FPS},scale={SIZE[0]}:{SIZE[1]}:flags=lanczos,setsar=1,tpad=stop_mode=clone:stop_duration=2,format=yuv420p",
                        "-frames:v", str(cut["frames"]), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", str(seg)], check=True)
        segments.append(seg)
    listing = d / "cuts" / "list.txt"
    listing.write_text("".join(f"file '{s.resolve().as_posix()}'\n" for s in segments), encoding="utf-8")
    silent = d / f"{plan['name']}_silent.mp4"
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(silent)], check=True)
    final = d / f"{plan['name']}.mp4"
    music = ROOT / plan["music"]["file"] if plan["music"]["file"] else None
    if music and music.exists():
        fade = max(0.0, plan["seconds"] - 0.8)
        subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(silent), "-ss", f"{plan['music']['offset']:.3f}", "-i", str(music), "-t", f"{plan['seconds']}", "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-af", f"afade=t=out:st={fade}:d=0.8", "-shortest", str(final)], check=True)
    else:
        silent.replace(final)
    entry = {"kind": "video", "id": plan["name"], "operation": "trend_video", "plan": str((d / "plan.json").relative_to(ROOT)).replace("\\", "/"), "seed": plan["seed"], "seconds": plan["seconds"],
             "cuts": len(plan["cuts"]), "shots": len(plan["shots"]), "music": plan["music"], "output": str(final), "silent": str(silent), "estimate": plan["estimate"]}
    tr.record_generation(REGISTRY, entry)
    return entry


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--setting", choices=list(tr.SETTINGS), default="mixed")
    p.add_argument("--weather", choices=list(tr.WEATHERS))
    p.add_argument("--library-share", type=float, default=0.35)
    p.add_argument("--light-variants", type=int, default=2)
    p.add_argument("--gpu-budget", type=float, default=1500.0)
    p.add_argument("--out", type=Path)
    s = sub.add_parser("status")
    s.add_argument("plan", type=Path)
    s.add_argument("--json", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("plan", type=Path)
    r.add_argument("--phase", choices=["stills", "qwen", "clips", "local", "assemble"], required=True)
    r.add_argument("--comfy-url", default="http://127.0.0.1:18188")
    r.add_argument("--allow-fallback", action="store_true", help="assembly only: replace a missing clip by a library clip (never the default)")
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = make_plan(args.seed, args.seconds, setting=args.setting, weather=args.weather, library_share=args.library_share, light_variants=args.light_variants, gpu_budget=args.gpu_budget)
        out = args.out or (run_dir(plan) / "plan.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: plan[k] for k in ("name", "context", "estimate", "styles_used")}, indent=1))
        print(f"{len(plan['cuts'])} cuts, {len(plan['shots'])} shots, plan written to {out}")
        return
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.command == "status":
        missing = needed(plan)
        print(json.dumps({k: v for k, v in missing.items()} if args.json else {k: len(v) for k, v in missing.items()}, indent=None if args.json else 1))
        return
    failed = {"stills": lambda: phase_stills(plan), "qwen": lambda: phase_qwen(plan, args.comfy_url), "clips": lambda: phase_clips(plan), "local": lambda: phase_local(plan),
              "assemble": lambda: print(json.dumps(phase_assemble(plan, args.allow_fallback), indent=1))}[args.phase]()
    if failed:
        print("FAILED:", failed, flush=True)
        sys.exit(2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
