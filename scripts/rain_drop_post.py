"""Post-process for a puddle-impact clip (no generation, so it costs no GPU time):
  1. calm the far ripples: farther than `reach` from the core of each impact, the thin ring strokes are erased in a soft fade and the reflection is steadied
     over a few frames, so the wave no longer breaks up the reflected painting. The cores (the volcano-shaped crater with its rim) and their first rings are
     untouched, and nothing is warped: lines stay straight and the horizon is not bent;
  2. add a GENERAL rain over the whole picture, not only over the puddle: artificial glassy drops with a trail drawn as parallel lines (mostly white, a light
     blue line stuck to it for the reflection of the drop, only a little dark line); the drops that land on the water make a small faint ripple, never on a real
     impact. No real impact is created or changed;
  3. speed the clip up (frames are skipped, the clip gets shorter, nothing is generated).
Several RAIN STYLES coexist (--style); none replaces another, they are all kept as choices (see RAIN_STYLES): "general" (above, the newest), "calm_targeted" (v3: far
ripples calmed, faint drops that each aim at a real impact and stop outside its core) and "pinch" (v2: the ripples are pulled in toward each impact by a warp, kept
for the record: on the reference clip it bent lines), and the "steady*" family (the reflected painting is held steady in time everywhere outside the cores, so the waves no
longer wobble it; only a share of the thin ring strokes is kept, more of it near the impacts). The tool never overwrites an existing file unless --overwrite is given.
Every parameter goes in a JSON file next to the output (and in the generation register) so the result can be replayed exactly.

    python scripts/rain_drop_post.py in.mp4 out.mp4 [--impacts impacts.json|none] [--level normal|exaggerated] [--speed 2] [--reach 30] [--rain-rate 150] [--seed 1]
impacts.json: {"impacts": [{"x": 195, "y": 528, "core": 55}, ...], "aspect": 0.85}   (pixel positions in the clip, core = radius kept intact)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402

DEFAULTS = {
    "speed": 2.0,  # whole numbers only (2 = 3 s -> 1.5 s): frames are skipped at an even rhythm; 1.5 made an uneven, jerky cadence
    "reach": 30.0,  # pixels of rings kept around the core of each impact; farther rings are calmed (0 = ripples stay inside the core only, <0 = off)
    "feather": 80.0,  # width of the soft edge of that reach
    "kernel": 9,  # thickness (px) of the ring strokes that are removed far from the impacts
    "reflect_window": 3,  # the far reflection is the median of the ring-free frames i-3..i+3 (steadier painting); 0 = this frame only
    "aspect": 0.85,  # width / height of the ripple ellipses (the pond is seen at an angle)
    "rain_rate": 150.0,  # artificial drops per second over the whole picture
    "water_share": 0.55,  # share of the drops that land on the water (and make a small ripple); the others land elsewhere (leaves, rock, ground)
    "opacity": 0.45,  # peak opacity of a drop trail; ripples use 0.8 of it
    "length": 70.0,  # trail length in pixels (before the depth factor)
    "fall_speed": 1800.0,  # pixels per second in the final (sped up) clip
    "flight": 0.11,  # seconds a drop is visible before it lands
    "ripple_life": 0.5,  # seconds a small ripple lives
    "angle": 12.0,  # degrees, direction the drops come FROM, seen from where they land (0 = from the right edge, positive = from slightly above)
    "black_share": 0.3,  # the dark line is only this share of the opacity (less black, more white)
    "white": [255, 255, 255],
    "blue_day": [190, 228, 255], "blue_night": [110, 160, 235],  # lighter blue by day
    "dark_day": [70, 85, 130], "dark_night": [5, 5, 25],
    "night": None,  # None = decided from the brightness of the clip; True/False to force
    "seed": 1,
    # the older styles' extra knobs (unused by "general")
    "targeted": 0,  # number of faint drops that each aim at a real impact and stop outside its core (styles calm_targeted and pinch)
    "stop_margin": 10.0,
    "t_white": [255, 255, 255], "t_black": [10, 8, 30], "t_blue": [90, 170, 255],  # the first trail colours: black + white + saturated blue lines
    "steady": False,  # hold the reflected painting steady: base = temporal median of the ring-free frames, plus a share of the ring strokes (styles steady*)
    "reflect_ref_frames": 0,  # >0: the steady painting is the median of the ring-free FIRST n frames (before the waves spread) for the whole clip, instead of a sliding window
    "ring_strength": 0.35,  # share of the thin ring strokes kept far from the impacts (1 = all, 0 = none); near the impacts they are kept in full
    "warp": None,  # "pinch" = pull the rings toward each impact with a radial warp (style pinch)
    "shrink": 65.0, "ramp": 50.0, "fade_warp": 140.0,  # the pinch: pixels pulled, build-up distance after the core, fade distance
    "style": "general",
}

# every rain style that was put forward stays available. Each entry only lists what differs from DEFAULTS.
RAIN_STYLES = {
    "general": {"label": "v4 general rain: glassy drops with a mostly white trail over the whole picture, small ripples on the water, steadier reflection, far rings calmed",
                "set": {}},
    "calm_targeted": {"label": "v3: far ripples calmed, faint black/white/blue drops each aimed at a real impact (they stop outside its core)",
                      "set": {"rain_rate": 0.0, "targeted": 14, "reflect_window": 0, "opacity": 0.3, "length": 70.0, "fall_speed": 1500.0}},
    "pinch": {"label": "v2: ripples pulled toward each impact by a radial warp, 1.5x speed (kept for the record: bent lines on the reference clip)",
              "set": {"rain_rate": 0.0, "targeted": 14, "reflect_window": 0, "opacity": 0.3, "length": 70.0, "fall_speed": 1500.0, "warp": "pinch", "speed": 1.5, "reach": -1.0}},
}

RAIN_STYLES["steady"] = {"label": "steady reflection (balanced): the reflected painting is held steady (+-12 frames), 35 percent of the far ring strokes kept, the impact cores untouched, general rain on top",
                         "set": {"steady": True, "reflect_window": 12, "ring_strength": 0.35}}
RAIN_STYLES["steady_still"] = {"label": "steady reflection (very calm): +-20 frames, only 12 percent of the far ring strokes, rings stay close to the impacts",
                               "set": {"steady": True, "reflect_window": 20, "ring_strength": 0.12, "reach": 10.0, "feather": 60.0}}
RAIN_STYLES["steady_keep"] = {"label": "steady reflection (lively): +-8 frames, 70 percent of the far ring strokes kept, the water still moves but the painting is steadier",
                              "set": {"steady": True, "reflect_window": 8, "ring_strength": 0.7, "reach": 60.0, "feather": 90.0}}

RAIN_STYLES["steady_first"] = {"label": "steady reflection (frozen painting): the reflected painting is the one of the first 6 frames, before the waves spread, for the whole clip; "
                                        "30 percent of the far ring strokes kept; impact cores untouched",
                               "set": {"steady": True, "reflect_ref_frames": 6, "ring_strength": 0.3}}

# named intensities. "exaggerated" is the demo level: only the cores keep their rings, a dense visible rain, faster.
LEVELS = {
    "normal": {},
    "exaggerated": {"speed": 3.0, "reach": 0.0, "feather": 60.0, "kernel": 11, "reflect_window": 4, "rain_rate": 320.0, "opacity": 0.75, "length": 120.0, "fall_speed": 2200.0},
}
# the reference clip: its seed (still and clip), and the impacts read on it, shipped next to the recipes
REFERENCE_SEED = 1372175472
REFERENCE_IMPACTS = Path(__file__).resolve().parent / "assets" / "impacts_c_big_enhance_gurren_2.json"


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def reach_weight(width: int, height: int, impacts: list[dict], p: dict) -> np.ndarray:
    """1 near an impact (its core and the first `reach` pixels of rings around it), 0 farther away, with a soft edge. Nothing is moved: the picture keeps its
    geometry, so no bent lines and no curved horizon."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    keep = np.zeros_like(xx)
    for item in impacts:
        rho = np.sqrt(((xx - item["x"]) / p["aspect"]) ** 2 + (yy - item["y"]) ** 2)  # elliptical radius
        keep = np.maximum(keep, 1.0 - smoothstep((rho - float(item["core"]) - p["reach"]) / p["feather"]))
    return keep


def ring_free(frame: np.ndarray, size: int) -> np.ndarray:
    """The same picture without its thin ring strokes (dark and bright lines narrower than `size`): grey closing then opening with a round footprint (a square one
    leaves blocky corners on the clouds). Big shapes (clouds, sun, rocks, leaves) stay, so the water looks calm there instead of broken."""
    out = np.empty_like(frame)
    axis = np.arange(size) - (size - 1) / 2.0
    disk = (axis[:, None] ** 2 + axis[None, :] ** 2) <= (size / 2.0) ** 2
    for channel in range(3):
        plane = ndi.grey_closing(frame[..., channel], footprint=disk)
        out[..., channel] = ndi.grey_opening(plane, footprint=disk)
    return ndi.gaussian_filter(out, sigma=(1.2, 1.2, 0))


def water_mask(frame_paths: list[Path]) -> np.ndarray:
    """1 where the picture moves during the clip (the water: rings and moving reflections), 0 where it stays put (foliage, rocks, big painted shapes):
    from the temporal standard deviation of the luminance. The calming only touches the water, so static details keep every pixel."""
    stack = np.stack([np.asarray(Image.open(path).convert("L")).astype(np.float32) for path in frame_paths[::2]])
    moving = smoothstep((stack.std(axis=0) - 6.0) / 12.0)
    return ndi.gaussian_filter(moving, 3.0)


def calm_far(frame: np.ndarray, background: np.ndarray, keep: np.ndarray, water: np.ndarray | None) -> np.ndarray:
    """Blend toward the calm background (ring-free, steadied in time) far from the impacts, on the water only."""
    far = 1.0 - keep
    if water is not None:
        far = far * water
    far = far[..., None]
    if far.max() < 1e-3:
        return frame
    return (frame.astype(np.float32) * (1 - far) + background.astype(np.float32) * far).clip(0, 255).astype(np.uint8)


def steady_frame(frame: np.ndarray, own_free: np.ndarray, background: np.ndarray, keep: np.ndarray, core: np.ndarray, water: np.ndarray, ring_strength: float) -> np.ndarray:
    """Hold the reflected painting steady on the water: the calm background (temporal median of the ring-free frames) plus the ring strokes of this frame
    (frame minus its own ring-free version) weighted by ring_strength far from the impacts and fully near them. The cores are copied from the frame untouched."""
    detail = frame.astype(np.float32) - own_free.astype(np.float32)
    weight = (ring_strength + (1.0 - ring_strength) * keep)[..., None]
    steady = background.astype(np.float32) + detail * weight
    inside = core[..., None]
    candidate = frame.astype(np.float32) * inside + steady * (1.0 - inside)
    moving = water[..., None]  # only the water: foliage and rocks keep every pixel
    return (frame.astype(np.float32) * (1 - moving) + candidate * moving).clip(0, 255).astype(np.uint8)


def plan_rain(width: int, height: int, duration: float, p: dict, water: np.ndarray | None = None, keep: np.ndarray | None = None) -> list[dict]:
    """Seeded general rain: each drop lands somewhere on the picture at time `arrive`; the ones landing on the water (outside every real impact) get a small ripple."""
    rnd = random.Random(p["seed"])
    candidates = None
    if water is not None:
        mask = water > 0.6
        if keep is not None:
            mask &= keep < 0.2  # never on a real impact
        points = np.argwhere(mask[::4, ::4]) * 4
        candidates = points if len(points) else None
    angle = math.radians(p["angle"])
    drops = []
    for k in range(int(p["rain_rate"] * duration)):
        if candidates is not None and rnd.random() < p["water_share"]:
            y, x = candidates[rnd.randrange(len(candidates))]
            x, y, on_water = float(x) + rnd.uniform(0, 4), float(y) + rnd.uniform(0, 4), True
        else:
            x, y = rnd.uniform(0, width), rnd.uniform(0, height)
            on_water = bool(water is not None and water[int(min(y, height - 1)), int(min(x, width - 1))] > 0.6 and (keep is None or keep[int(min(y, height - 1)), int(min(x, width - 1))] < 0.2))
        a = angle + math.radians(rnd.uniform(-6, 6))
        drops.append({"id": k, "arrive": round(rnd.uniform(0.0, duration + p["flight"]), 3), "land": [round(x, 1), round(y, 1)],
                      "direction": [round(math.cos(a), 4), round(-math.sin(a), 4)], "depth": round(rnd.uniform(0.55, 1.0), 3), "ripple": on_water})
    return drops


def _line(draw: ImageDraw.ImageDraw, a: tuple[float, float], b: tuple[float, float], colour, alpha: float, width: float, scale: int) -> None:
    if alpha > 0.004:
        draw.line((a[0] * scale, a[1] * scale, b[0] * scale, b[1] * scale), fill=(*colour, int(255 * min(1.0, alpha))), width=max(1, round(width * scale)))


def draw_rain(frame: np.ndarray, t: float, drops: list[dict], p: dict, night: bool) -> np.ndarray:
    """Alpha-blend the drops and small ripples alive at time t (seconds of the final clip), drawn at 2x then reduced for clean lines."""
    height, width = frame.shape[:2]
    scale = 2
    blue = p["blue_night" if night else "blue_day"]
    dark = p["dark_night" if night else "dark_day"]
    layer = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    drawn = False
    for drop in drops:
        age = t - drop["arrive"]
        if age < -p["flight"] or age > p["ripple_life"] * 1.3:
            continue
        depth = drop["depth"]
        ux, uy = drop["direction"]
        nx, ny = -uy, ux
        lx, ly = drop["land"]
        if age <= 0:  # in flight: a glassy bead and its trail
            ahead = -age * p["fall_speed"] * depth
            hx, hy = lx + ux * ahead, ly + uy * ahead
            length = p["length"] * depth
            fade = min(1.0, (age + p["flight"]) / 0.03)
            for name, colour, offset, width_px, share in (("dark", dark, -1.9, 1.0, p["black_share"]), ("white", p["white"], 0.0, 1.9 * depth, 1.0), ("blue", blue, 2.5, 1.3, 0.85)):
                for s in range(6):
                    a0, a1 = s / 6, (s + 1) / 6
                    alpha = p["opacity"] * depth * share * (1 - 0.9 * a0) * fade
                    _line(draw, (hx + ux * length * a0 + nx * offset, hy + uy * length * a0 + ny * offset),
                          (hx + ux * length * a1 + nx * offset, hy + uy * length * a1 + ny * offset), colour, alpha, width_px, scale)
            r = 1.4 + 1.8 * depth  # the bead: white with a light blue rim
            box = ((hx - r) * scale, (hy - r) * scale, (hx + r) * scale, (hy + r) * scale)
            draw.ellipse(box, fill=(*p["white"], int(255 * p["opacity"] * 1.1 * depth * fade)), outline=(*blue, int(255 * p["opacity"] * fade)), width=scale)
            drawn = True
        elif drop["ripple"]:  # small faint ripple on the water
            for delay, weight in ((0.0, 1.0), (0.14, 0.6)):
                life = (age - delay) / p["ripple_life"]
                if 0 <= life <= 1.0:
                    radius = 3.0 + 24.0 * depth * life
                    alpha = p["opacity"] * 0.8 * weight * (1 - life) ** 1.3
                    for colour, extra, width_px, share in ((p["white"], 0.0, 1.4, 1.0), (blue, 2.4, 1.0, 0.8)):
                        rr = radius + extra
                        draw.ellipse(((lx - rr * p["aspect"]) * scale, (ly - rr) * scale, (lx + rr * p["aspect"]) * scale, (ly + rr) * scale),
                                     outline=(*colour, int(255 * min(1.0, alpha * share))), width=max(1, round(width_px * scale)))
                    drawn = True
    if not drawn:
        return frame
    layer = layer.resize((width, height), Image.LANCZOS)
    rgba = np.asarray(layer).astype(np.float32)
    alpha = rgba[..., 3:4] / 255.0
    return (frame.astype(np.float32) * (1 - alpha) + rgba[..., :3] * alpha).clip(0, 255).astype(np.uint8)


def pinch_map(width: int, height: int, impacts: list[dict], p: dict) -> tuple[np.ndarray, np.ndarray]:
    """(style pinch) Source coordinates for every output pixel: identity inside each core (protected from every neighbour), then the picture is read farther away so the
    rings land closer to the impact, fading out smoothly."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx_total, dy_total = np.zeros_like(xx), np.zeros_like(yy)
    influence, protect = np.zeros_like(xx), np.zeros_like(xx)
    for item in impacts:
        dx, dy = xx - item["x"], yy - item["y"]
        rho = np.sqrt((dx / p["aspect"]) ** 2 + dy ** 2) + 1e-6
        r0 = float(item["core"])
        delta = p["shrink"] * smoothstep((rho - r0) / p["ramp"]) * (1.0 - smoothstep((rho - r0 - p["ramp"]) / p["fade_warp"]))
        dx_total += dx / rho * delta
        dy_total += dy / rho * delta
        influence += delta / max(p["shrink"], 1e-6)
        protect = np.maximum(protect, 1.0 - smoothstep((rho - r0) / 18.0))
    keep = (1.0 - protect) / np.maximum(1.0, influence)
    return xx + dx_total * keep, yy + dy_total * keep


def warp(frame: np.ndarray, coords: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    out = np.empty_like(frame)
    for channel in range(3):
        out[..., channel] = ndi.map_coordinates(frame[..., channel], [coords[1], coords[0]], order=1, mode="nearest")
    return out


def plan_targeted(impacts: list[dict], duration: float, p: dict) -> list[dict]:
    """(styles calm_targeted, pinch) Seeded faint drops, each aimed at the edge of an impact core and arriving at a chosen time, never inside the core."""
    rnd = random.Random(p["seed"])
    weights = [float(i["core"]) + 1.0 for i in impacts]
    drops = []
    for k in range(int(p["targeted"])):
        target = rnd.choices(impacts, weights)[0]
        side = math.radians(p["angle"] + rnd.uniform(-8, 8))
        direction = (math.cos(side), -math.sin(side))
        around = rnd.uniform(-0.6, 0.6)
        radius = float(target["core"]) + p["stop_margin"]
        rotated = (math.cos(around) * direction[0] - math.sin(around) * direction[1], math.sin(around) * direction[0] + math.cos(around) * direction[1])
        end = (target["x"] + radius * p["aspect"] * rotated[0], target["y"] + radius * rotated[1])
        drops.append({"id": k, "arrive": round(rnd.uniform(0.2, max(0.3, duration - 0.15)), 3), "end": [round(end[0], 1), round(end[1], 1)],
                      "direction": [round(direction[0], 4), round(direction[1], 4)], "target": [target["x"], target["y"]]})
    return drops


def draw_targeted(frame: np.ndarray, t: float, drops: list[dict], p: dict) -> np.ndarray:
    """(styles calm_targeted, pinch) Three parallel lines per drop (black, white, blue next to the white one = the reflection of the drop), drawn at 2x."""
    height, width = frame.shape[:2]
    scale = 2
    layer = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    drawn = False
    for drop in drops:
        ahead = (drop["arrive"] - t) * p["fall_speed"]
        if ahead < 0 or ahead > p["fall_speed"] * 0.25:
            continue
        ux, uy = drop["direction"]
        nx, ny = -uy, ux
        hx, hy = drop["end"][0] + ux * ahead, drop["end"][1] + uy * ahead
        fade_in = min(1.0, (drop["arrive"] - t) / 0.03 + 0.2)
        for name, colour, offset, width_px in (("black", p["t_black"], -2.2, 2.0), ("white", p["t_white"], 0.0, 2.2), ("blue", p["t_blue"], 2.8, 1.6)):
            for k in range(8):
                a0, a1 = k / 8, (k + 1) / 8
                alpha = p["opacity"] * (1 - a0) * fade_in * (0.85 if name == "blue" else 1.0)
                _line(draw, (hx + ux * p["length"] * a0 + nx * offset, hy + uy * p["length"] * a0 + ny * offset),
                      (hx + ux * p["length"] * a1 + nx * offset, hy + uy * p["length"] * a1 + ny * offset), colour, alpha, width_px, scale)
        drawn = True
    if not drawn:
        return frame
    layer = layer.resize((width, height), Image.LANCZOS)
    rgba = np.asarray(layer).astype(np.float32)
    alpha = rgba[..., 3:4] / 255.0
    return (frame.astype(np.float32) * (1 - alpha) + rgba[..., :3] * alpha).clip(0, 255).astype(np.uint8)


def process(src: Path, dst: Path, impacts_file: Path | None, overrides: dict | None = None, *, overwrite: bool = False) -> dict:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists: every rain version is kept, choose another name (or pass --overwrite)")
    spec = json.loads(Path(impacts_file).read_text(encoding="utf-8")) if impacts_file else {}
    level = (overrides or {}).get("level", "normal")
    style = (overrides or {}).get("style") or "general"
    p = {**DEFAULTS, **{k: v for k, v in spec.items() if k in DEFAULTS}, **RAIN_STYLES[style]["set"], **LEVELS[level],
         **{k: v for k, v in (overrides or {}).items() if k in DEFAULTS and v is not None}}
    p["level"], p["style"] = level, style
    impacts = spec.get("impacts", [])
    ffmpeg = mt.ffmpeg_binary()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(src), str(tmp / "in_%04d.png")], check=True)
        frames = sorted(tmp.glob("in_*.png"))
        probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(src)], capture_output=True, text=True).stderr
        fps = 30.0
        for part in probe.split(","):
            if part.strip().endswith(" fps"):
                fps = float(part.strip().split()[0])
        first = np.asarray(Image.open(frames[0]).convert("RGB"))
        height, width = first.shape[:2]
        night = bool(p["night"]) if p["night"] is not None else float(first.astype(np.float32).mean()) < 55.0
        p["night"] = night
        keep = reach_weight(width, height, impacts, p) if p["reach"] >= 0 and impacts else None
        water = water_mask(frames) if (impacts and (keep is not None or p["rain_rate"] > 0)) else None
        coords = pinch_map(width, height, impacts, p) if p["warp"] == "pinch" and impacts else None
        core = reach_weight(width, height, impacts, {**p, "reach": 0.0, "feather": 25.0}) if (p["steady"] and impacts) else None
        free = None
        if keep is not None:
            free = np.stack([ring_free(np.asarray(Image.open(f).convert("RGB")), int(p["kernel"])) for f in frames])
        first_background = np.median(free[:max(1, int(p["reflect_ref_frames"]))], axis=0).astype(np.uint8) if (free is not None and p["reflect_ref_frames"] > 0) else None
        out_count = int(len(frames) / p["speed"])
        duration = out_count / fps
        drops = plan_rain(width, height, duration, p, water, keep) if p["rain_rate"] > 0 else []
        targeted = plan_targeted(impacts, duration, p) if p["targeted"] > 0 and impacts else []
        window = int(p["reflect_window"])
        for j in range(out_count):
            index = min(len(frames) - 1, int(round(j * p["speed"])))
            frame = np.asarray(Image.open(frames[index]).convert("RGB"))
            if coords is not None:
                frame = warp(frame, coords)
            if keep is not None:
                lo, hi = max(0, index - window), min(len(frames), index + window + 1)
                if p["reflect_ref_frames"] > 0:
                    background = first_background
                else:
                    background = np.median(free[lo:hi], axis=0).astype(np.uint8) if window > 0 else free[index]
                if p["steady"]:
                    frame = steady_frame(frame, free[index], background, keep, core, water, p["ring_strength"])
                else:
                    frame = calm_far(frame, background, keep, water)
            if drops:
                frame = draw_rain(frame, j / fps, drops, p, night)
            if targeted:
                frame = draw_targeted(frame, j / fps, targeted, p)
            Image.fromarray(frame).save(tmp / f"out_{j:04d}.png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-v", "error", "-framerate", f"{fps:g}", "-i", str(tmp / "out_%04d.png"), "-c:v", "libx264", "-crf", "14",
                        "-pix_fmt", "yuv420p", str(dst)], check=True)
    record = {"kind": "post", "id": dst.stem, "source": str(src), "operation": "rain_drop_post", "output": str(dst), "frames_in": len(frames), "frames_out": out_count,
              "fps": fps, "duration": round(duration, 3), "impacts": impacts, "params": p, "style_label": RAIN_STYLES[style]["label"], "drops": len(drops) + len(targeted),
              "drops_file": str(dst.with_suffix(".drops.json"))}
    dst.with_suffix(".post.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    dst.with_suffix(".drops.json").write_text(json.dumps(drops or targeted), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--impacts", help="impacts JSON (default: the reference clip's); pass 'none' for a clip with no impact map")
    parser.add_argument("--level", choices=list(LEVELS), default="normal")
    parser.add_argument("--style", choices=list(RAIN_STYLES), default="general", help="; ".join(f"{k}: {v['label']}" for k, v in RAIN_STYLES.items()))
    parser.add_argument("--overwrite", action="store_true", help="replace the output if it exists (never done by default: all versions are kept)")
    for key in ("speed", "reach", "feather", "kernel", "reflect_window", "rain_rate", "water_share", "opacity", "length", "fall_speed", "angle", "black_share"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=float)
    parser.add_argument("--ring-strength", dest="ring_strength", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--night", dest="night", action="store_true", default=None, help="force the night palette (darker blue)")
    parser.add_argument("--day", dest="night", action="store_false", help="force the day palette (light blue)")
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if (k in DEFAULTS or k in ("level", "style")) and v is not None}
    started = time.time()
    record = process(Path(args.src), Path(args.dst), None if args.impacts == "none" else Path(args.impacts or REFERENCE_IMPACTS), overrides, overwrite=args.overwrite)
    print(f"{record['frames_in']} -> {record['frames_out']} frames, {record['duration']} s, {record['drops']} drops, in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
