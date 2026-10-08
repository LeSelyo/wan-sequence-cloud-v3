"""Isolate ONE splash event of a clip and MULTIPLY it (no generation, no GPU):
  1. calm water: the splashes of the still the clip was made from are rebuilt as calm water (diffusion from the pixels around them);
  2. the event: for each frame only the INK of the splash is kept, inside a soft elliptical window around the chosen splash, for the first `event_frames` frames: the pixels that are
     much darker than the calm water (crater, outlines) or much brighter and whitish (highlights, crown, thin ring lines). Reflections (clouds, sun) are not coloured ink, so they stay
     behind. The window cuts the wide spreading ripples, so the rings stay SMALL;
  3. copies: that ink is pasted with its own colours at several places of the water, delayed in time, scaled by depth (near = big, far = small and less detailed), with a short fade in and out.
The result is a clip of `--seconds` with many drops of the SAME kind as the isolated one. All the parameters and every copy (place, scale, start) go in a JSON next to the output.

    python scripts/drop_multiply.py clip.mp4 still.png out.mp4 --spec spec.json [--seconds 10] [--copies 22] [--seed 1]
spec.json: {"event": {"x": 150, "y": 295, "rx": 70, "ry": 100}, "inpaint": [{"x":..,"y":..,"rx":..,"ry":..}, ...], "allowed": [[x0,y0,x1,y1], ...],
            "depth": {"near_weight": [0.6, 0.4], "far_scale": 0.3}}     (pixel positions in the finished clip, 576x1024; the still is turned 90 degrees clockwise like the clip)
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402

DEFAULTS = {"seconds": 10.0, "copies": 30, "event_frames": 40, "fade_in": 3, "fade_out_share": 0.3, "min_gap": 0.9, "seed": 1, "fps": 30, "window_feather": 0.35, "keep_original": True,
            "dark": 25.0, "dark_soft": 30.0, "bright": 50.0, "bright_soft": 40.0, "white_sat": 0.6}


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def calm_water(still: np.ndarray, ellipses: list[dict], margin: float = 1.25) -> np.ndarray:
    """Rebuild the water under every splash ellipse by multi-scale diffusion from its surroundings, with a light grain."""
    h, w = still.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    hole = np.zeros((h, w), bool)
    for e in ellipses:
        hole |= (((xx - e["x"]) / (e["rx"] * margin)) ** 2 + ((yy - e["y"]) / (e["ry"] * margin)) ** 2) <= 1.0
    hole = ndi.binary_dilation(hole, iterations=6)
    known = ~hole
    img = still.astype(np.float32)
    filled = img.copy()
    first = True
    for factor in (8, 4, 2, 1):
        sh, sw = h // factor, w // factor
        small = np.asarray(Image.fromarray(np.clip(filled, 0, 255).astype(np.uint8)).resize((sw, sh), Image.LANCZOS)).astype(np.float32)
        k = np.asarray(Image.fromarray((known * 255).astype(np.uint8)).resize((sw, sh), Image.NEAREST)) > 127
        cur = small.copy()
        if first:
            cur[~k] = small[k].mean(axis=0)
            first = False
        for _ in range(1500 if factor >= 8 else (400 if factor >= 4 else 120)):
            cur = np.where(k[..., None], small, ndi.uniform_filter(cur, size=(5, 5, 1), mode="nearest"))
        up = np.asarray(Image.fromarray(np.clip(cur, 0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)).astype(np.float32)
        filled = np.where(known[..., None], img, up)
    rng = np.random.default_rng(3)
    filled = np.where(known[..., None], filled, filled + rng.normal(0, 2.0, filled.shape))
    soft = ndi.gaussian_filter(hole.astype(np.float32), 5)[..., None]
    return np.clip(img * (1 - soft) + filled * soft, 0, 255)


def depth_scale(x: float, y: float, width: int, height: int, depth: dict) -> float:
    """1 near the camera down to far_scale at the far side (the far side is the sun side: right and bottom of the turned picture)."""
    wx, wy = depth.get("near_weight", [0.6, 0.4])
    far = wx * x / width + wy * y / height
    return 1.0 - (1.0 - depth.get("far_scale", 0.3)) * float(np.clip((far - 0.1) / 0.7, 0, 1))


def plan_copies(spec: dict, width: int, height: int, p: dict) -> list[dict]:
    """Seeded places inside the allowed rectangles, scaled by depth (relative to the source splash), spread apart, with start frames over the whole clip."""
    rnd = random.Random(p["seed"])
    ev = spec["event"]
    k_src = depth_scale(ev["x"], ev["y"], width, height, spec.get("depth", {}))
    total = int(p["seconds"] * p["fps"])
    copies = []
    if p["keep_original"]:
        copies.append({"id": 0, "x": ev["x"], "y": ev["y"], "scale": 1.0, "start": 6})
    rects = spec["allowed"]
    tries = 0
    while len(copies) < p["copies"] and tries < 5000:
        tries += 1
        x0, y0, x1, y1 = rnd.choice(rects)
        x, y = rnd.uniform(x0, x1), rnd.uniform(y0, y1)
        scale = depth_scale(x, y, width, height, spec.get("depth", {})) / k_src
        radius = max(ev["rx"], ev["ry"]) * scale * p["min_gap"]
        if any(np.hypot(x - c["x"], y - c["y"]) < (radius + max(ev["rx"], ev["ry"]) * c["scale"] * p["min_gap"]) * 0.8 and abs(rnd.random()) < 0.85 for c in copies):
            continue
        copies.append({"id": len(copies), "x": round(x, 1), "y": round(y, 1), "scale": round(scale, 3), "start": 0})
    # start times spread evenly over the clip (with a jitter) so there is no empty moment and no crowding
    span = max(1, total - p["event_frames"] // 2)
    rest = [c for c in copies if c["id"] != 0] if p["keep_original"] else copies
    for index, c in enumerate(rest):
        c["start"] = min(span - 1, int((index + rnd.random()) / max(1, len(rest)) * span))
    return copies


def window(rx: float, ry: float, feather: float) -> np.ndarray:
    """Soft elliptical window: 1 in the middle, smooth to 0 at the ellipse (this is what keeps the rings small)."""
    h, w = int(2 * ry) + 1, int(2 * rx) + 1
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - rx) / rx) ** 2 + ((yy - ry) / ry) ** 2)
    return 1.0 - smoothstep((r - (1.0 - feather)) / feather)


def extract_event(frames: list[Path], bg: np.ndarray, spec: dict, p: dict) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, int, int, int]]:
    """The ink of the chosen splash for the first `event_frames` frames: a list of (colour crop, alpha crop) and the crop box (x0, y0, x1, y1) in the clip."""
    height, width = bg.shape[:2]
    ev = spec["event"]
    x0, x1 = int(max(0, ev["x"] - ev["rx"])), int(min(width, ev["x"] + ev["rx"] + 1))
    y0, y1 = int(max(0, ev["y"] - ev["ry"])), int(min(height, ev["y"] + ev["ry"] + 1))
    win = window(ev["rx"], ev["ry"], p["window_feather"])[: y1 - y0, : x1 - x0]
    n_event = min(p["event_frames"], len(frames))
    layers = []  # (colour, alpha) of the ink of each frame
    lum_bg = (bg @ np.array([0.299, 0.587, 0.114], np.float32))[y0:y1, x0:x1]
    for i in range(n_event):
        f = np.asarray(Image.open(frames[i]).convert("RGB")).astype(np.float32)[y0:y1, x0:x1]
        lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
        whitish = (f.min(axis=2) / np.maximum(f.max(axis=2), 1.0)) > p["white_sat"]
        ink_dark = smoothstep((lum_bg - lum - p["dark"]) / p["dark_soft"])
        ink_bright = smoothstep((lum - lum_bg - p["bright"]) / p["bright_soft"]) * whitish
        layers.append((f, np.maximum(ink_dark, ink_bright) * win))
    return layers, (x0, y0, x1, y1)


def process(clip: Path, still: Path, out: Path, spec: dict, overrides: dict | None = None) -> dict:
    p = {**DEFAULTS, **(overrides or {})}
    ffmpeg = mt.ffmpeg_binary()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(clip), str(tmp / "c_%04d.png")], check=True)
        frames = sorted(tmp.glob("c_*.png"))
        first = np.asarray(Image.open(frames[0]).convert("RGB"))
        height, width = first.shape[:2]
        turned = np.asarray(Image.open(still).convert("RGB").transpose(Image.Transpose.ROTATE_270))  # clockwise, like the finished clip
        assert turned.shape[:2] == (height, width), f"still {turned.shape[:2]} does not match the clip {(height, width)}"
        bg = calm_water(turned, spec["inpaint"])
        layers, _box = extract_event(frames, bg, spec, p)
        n_event = len(layers)
        copies = plan_copies(spec, width, height, p)
        scaled = {}
        for c in copies:
            key = round(c["scale"], 2)
            if key not in scaled:
                scaled[key] = [(ndi.zoom(col, (key, key, 1), order=1), np.clip(ndi.zoom(al, (key, key), order=1), 0, 1)) for col, al in layers]
        total = int(p["seconds"] * p["fps"])
        fade_out = max(1, int(n_event * p["fade_out_share"]))
        for j in range(total):
            frame = bg.copy()
            for c in copies:
                age = j - c["start"]
                if not 0 <= age < n_event:
                    continue
                weight = min(1.0, (age + 1) / p["fade_in"]) * min(1.0, (n_event - age) / fade_out)
                col, al = scaled[round(c["scale"], 2)][age]
                al = al * weight
                h, w = al.shape[:2]
                cx0, cy0 = int(c["x"] - w / 2), int(c["y"] - h / 2)
                sx0, sy0 = max(0, -cx0), max(0, -cy0)
                ex0, ey0 = max(0, cx0), max(0, cy0)
                ex1, ey1 = min(width, cx0 + w), min(height, cy0 + h)
                if ex1 > ex0 and ey1 > ey0:
                    a = al[sy0:sy0 + (ey1 - ey0), sx0:sx0 + (ex1 - ex0), None]
                    frame[ey0:ey1, ex0:ex1] = frame[ey0:ey1, ex0:ex1] * (1 - a) + col[sy0:sy0 + (ey1 - ey0), sx0:sx0 + (ex1 - ex0)] * a
            Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).save(tmp / f"o_{j:04d}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-v", "error", "-framerate", str(p["fps"]), "-i", str(tmp / "o_%04d.png"), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out)], check=True)
        Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).save(out.with_suffix(".calm_water.png"))
    record = {"kind": "post", "id": out.stem, "operation": "drop_multiply", "source_clip": str(clip), "source_still": str(still), "output": str(out), "params": p, "spec": spec,
              "copies": copies, "frames_out": total, "fps": p["fps"]}
    out.with_suffix(".multiply.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", type=Path)
    parser.add_argument("still", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--copies", type=int)
    parser.add_argument("--event-frames", dest="event_frames", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None}
    started = time.time()
    record = process(args.clip, args.still, args.out, json.loads(args.spec.read_text(encoding="utf-8")), overrides)
    print(f"{len(record['copies'])} copies, {record['frames_out']} frames, in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
