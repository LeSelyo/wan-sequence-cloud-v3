"""Transplant ONE isolated impact (a crown, a volcano...) onto ANOTHER clip, the TARGET, whose water is not clean and keeps moving (no calm background is needed for the target: the
impact is drawn over its frames). Before it is added the impact is REWORKED so that it no longer reads as a static water bubble that never bursts:
  - its outline is traced with a pencil line (white, black, black + white, white + light blue parallel line), with a slightly wobbly width and a grain, the line follows the silhouette
    of the impact frame by frame and keeps a constant thin width whatever the depth scale;
  - optionally it BURSTS: the ink shrinks and fades after a few frames while thin pencil rings spread from its outline and fade, like the end of a bubble.
The copies are placed on free water of the target (not on its own drops or ripples, measured on its frames), scaled by depth (near = bigger, far = small), spread in time.
No generation, no GPU. Every parameter and every copy goes in a JSON next to the output.

    python scripts/drop_transplant.py render TARGET.mp4 SOURCE.mp4 SOURCE_STILL.png OUT.mp4 --source-spec crown.json --target-spec target.json --style burst [--copies 12] [--seed 1]
    python scripts/drop_transplant.py planche TARGET.mp4 SOURCE.mp4 SOURCE_STILL.png OUT.png --source-spec crown.json --target-spec target.json
    python scripts/drop_transplant.py union OUT [--method auto|qwen|transplant] [--style S] [--outline O] [--burst|--no-burst] [--copies N] [--seed K] [--target T --source S ...] [--show-defaults]
styles: none (raw ink) | white | black | double | blue | burst | burst_blue
`union --method auto` (the default) uses QWEN-IMAGE-EDIT when it is in the bundle (profile qwen-edit of config/base_models.json) AND present on the server: the two pictures (the base scene
k_craters, the crown impact) are given to the model with the instruction "mix the two" and OUT is a PNG keyframe of the union, to be animated by the usual pipeline. Otherwise (or with
--method transplant) the CPU transplant below is used and OUT is an MP4. `union` takes EVERYTHING from the defaults declared in scripts/assets/rain_anime_recipes.json ("rain_union": the base video k_craters with its generation seeds/prompts, the crown impact
isolated from m_crown_1, the pencil outline and burst of the last creation = burst_blue, 12 copies, seed 1); any option given on the command line replaces the default. The two ingredients stay
usable apart (the base clip alone, the crown multiplied over its own calm water) and their union is the default. --outline (none|white|black|double|blue) and --burst/--no-burst choose the two
parts of the reworked impact separately, in any combination.
target.json: {"allowed": [[x0,y0,x1,y1], ...], "scale": {"near": 0.55, "far": 0.17}, "depth": {"near_weight": [0.6, 0.4]}, "busy": {"threshold": 28, "max_fraction": 0.09}}
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
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drop_multiply as dm  # noqa: E402
import music_timing as mt  # noqa: E402

COLOURS = {"white": (255, 255, 255), "black": (12, 10, 34), "blue": (178, 222, 255)}
# lines: (colour, offset in line widths, opacity); offset > 0 = outside the silhouette. burst = the impact collapses and rings spread
STYLES = {
    "none": {"lines": [], "burst": False, "label": "brut (comme dans le clip multiplie)"},
    "white": {"lines": [("white", 0.0, 0.9)], "burst": False, "label": "crayon blanc"},
    "black": {"lines": [("black", 0.0, 0.85)], "burst": False, "label": "crayon noir"},
    "double": {"lines": [("black", 1.8, 0.8), ("white", 0.0, 0.9)], "burst": False, "label": "noir dehors + blanc"},
    "blue": {"lines": [("white", 0.0, 0.9), ("blue", 1.9, 0.8)], "burst": False, "label": "blanc + bleu clair parallele"},
    "burst": {"lines": [("white", 0.0, 0.9)], "burst": True, "label": "crayon blanc + eclatement"},
    "burst_blue": {"lines": [("white", 0.0, 0.9), ("blue", 1.9, 0.8)], "burst": True, "label": "blanc + bleu + eclatement"},
}
OUTLINES = {  # the pencil outline alone: lines of (colour, offset in line widths, opacity)
    "none": [], "white": [("white", 0.0, 0.9)], "black": [("black", 0.0, 0.85)],
    "double": [("black", 1.8, 0.8), ("white", 0.0, 0.9)], "blue": [("white", 0.0, 0.9), ("blue", 1.9, 0.8)],
}
ROOT = Path(__file__).resolve().parent.parent
RECIPES = ROOT / "scripts" / "assets" / "rain_anime_recipes.json"
DEFAULTS = {"copies": 12, "seed": 1, "fps": 30, "line_width": 1.9, "ring_max": 26.0, "ring_delays": [9, 14], "ring_life": 24, "collapse_start": 12, "tail": 38,
            "stagger_end": 26, "tries": 500}


def smoothstep(x):
    return dm.smoothstep(x)


def disk(radius: int) -> np.ndarray:
    axis = np.arange(-radius, radius + 1)
    return (axis[:, None] ** 2 + axis[None, :] ** 2) <= radius ** 2


def silhouette(alpha: np.ndarray, scale: float) -> np.ndarray:
    """The solid outline shape of the impact: strong ink only, fragments merged, holes filled, spray specks dropped, boundary smoothed so the pencil line is clean."""
    mask = alpha > 0.4
    if not mask.any():
        return mask
    close_r = max(2, int(round(4 * math.sqrt(scale))))
    closed = ndi.binary_fill_holes(ndi.binary_closing(mask, structure=disk(close_r)))
    opened = ndi.binary_opening(closed, structure=disk(max(1, close_r - 1)))
    shape = opened if opened.any() else closed
    labels, count = ndi.label(shape)
    if count > 1:  # keep the main body and the pieces that are not tiny next to it
        areas = ndi.sum(shape, labels, range(1, count + 1))
        keep = [i + 1 for i, area in enumerate(areas) if area >= 0.2 * areas.max()]
        shape = np.isin(labels, keep)
    smooth = ndi.gaussian_filter(shape.astype(np.float32), 1.5 + 2.0 * math.sqrt(scale)) > 0.5
    return smooth if smooth.any() else shape


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    field = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return field / (field.std() + 1e-6)


def add_line(pre: np.ndarray, alpha: np.ndarray, line_alpha: np.ndarray, colour) -> tuple[np.ndarray, np.ndarray]:
    la = np.clip(line_alpha, 0, 1)[..., None]
    pre = np.asarray(colour, np.float32) * la + pre * (1 - la)
    return pre, la[..., 0] + alpha * (1 - la[..., 0])


def resolve_style(style: str | None = None, outline: str | None = None, burst: bool | None = None) -> tuple[str, dict]:
    """A named style, or an outline (none|white|black|double|blue) and the burst chosen separately; an explicit outline/burst replaces the one of the named style."""
    base = STYLES[style] if style else {"lines": [], "burst": False, "label": ""}
    if outline is None and burst is None and style:
        return style, base
    lines = OUTLINES[outline] if outline is not None else base["lines"]
    is_burst = base["burst"] if burst is None else bool(burst)
    name = f"{outline if outline is not None else (style or 'custom')}{'+burst' if is_burst else ''}"
    return name, {"lines": lines, "burst": is_burst, "label": f"outline {outline if outline is not None else 'of ' + str(style)}, burst {'on' if is_burst else 'off'}"}


def load_union_defaults() -> dict:
    """The declared defaults of the union, with every path made absolute (relative paths are relative to the repository root)."""
    union = json.loads(RECIPES.read_text(encoding="utf-8"))["rain_union"]
    d = dict(union["defaults"])
    for key in ("target", "source", "source_still", "base_still", "source_spec", "target_spec"):
        if key in d:
            d[key] = str(ROOT / d[key])
    return d


class Impact:
    """One copy of the isolated impact: renders the (colour, alpha) layer of every age, reworked with the chosen style."""

    def __init__(self, layers, scale: float, style: str, seed: int, p: dict):
        self.layers, self.scale, self.style, self.seed, self.p = layers, scale, (STYLES[style] if isinstance(style, str) else style), seed, p
        self.n = len(layers)

    @property
    def life(self) -> int:
        return min(self.n, self.p["tail"]) if self.style["burst"] else self.n

    def envelope(self, age: int) -> tuple[float, float]:
        """(scale factor, ink opacity) of the ink at this age."""
        if not self.style["burst"]:
            fade_out = max(1, int(self.n * 0.3))
            return 1.0, min(1.0, (age + 1) / 3.0) * min(1.0, (self.n - age) / fade_out)
        c = self.p["collapse_start"]
        grow = 0.7 + 0.3 * float(smoothstep(age / 5.0))
        shrink = 1.0 - 0.45 * float(smoothstep((age - c) / 16.0))
        opacity = float(smoothstep((age + 1) / 4.0)) * (1.0 - float(smoothstep((age - c) / 18.0)))
        return (grow if age < 5 else shrink), opacity

    def render(self, age: int):
        p, k = self.p, self.scale
        col0, al0 = self.layers[min(age, self.n - 1)]
        factor, opacity = self.envelope(age)
        s = max(0.05, k * factor)
        col = ndi.zoom(col0, (s, s, 1), order=1)
        al = np.clip(ndi.zoom(al0, (s, s), order=1), 0, 1)
        width = float(np.clip(p["line_width"] * math.sqrt(k), 0.9, 2.0))
        pad = int(math.ceil(6 + 3.5 * width + (p["ring_max"] * k + 3 * width if self.style["burst"] else 0)))
        col = np.pad(col, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        al = np.pad(al, pad)
        pre = col * (al * opacity)[..., None]
        alpha = al * opacity
        if self.style["lines"]:
            rng = np.random.default_rng(self.seed * 1000 + age // 3)  # the line "boils" a little every 3 frames, like a hand drawing
            sil = silhouette(al, s)
            if sil.any():
                sd = ndi.distance_transform_edt(~sil) - ndi.distance_transform_edt(sil)  # > 0 outside the silhouette
                wobble = 0.6 * smooth_noise(rng, al.shape, 3.0)
                thickness = 1.0 + 0.3 * np.clip(smooth_noise(rng, al.shape, 5.0), -1.5, 1.5)
                grain = np.clip(0.88 + 0.18 * smooth_noise(rng, al.shape, 0.8), 0.55, 1.0)
                visible = float(np.clip(0.55 + k, 0.55, 1.0))  # far impacts: lighter lines (less visible, less detailed)
                rings = [(0.0, 1.0)]
                if self.style["burst"]:
                    rings = []
                    for delay in p["ring_delays"]:
                        v = (age - delay) / p["ring_life"]
                        if 0 <= v <= 1:
                            rings.append((p["ring_max"] * k * float(smoothstep(v)), (1 - v) ** 1.2))
                    rings.append((0.0, opacity if opacity > 0 else 0.0))
                for colour_name, offset, opacity_line in self.style["lines"]:
                    for ring_offset, ring_alpha in rings:
                        distance = sd - (offset * width + ring_offset) + wobble
                        line = np.clip((0.5 * width * thickness + 0.5 - np.abs(distance)), 0, 1)
                        pre, alpha = add_line(pre, alpha, line * grain * opacity_line * ring_alpha * visible, COLOURS[colour_name])
        return (pre / np.maximum(alpha, 1e-4)[..., None]).astype(np.float32), np.clip(alpha, 0, 1).astype(np.float32)


def paste(frame: np.ndarray, col: np.ndarray, al: np.ndarray, cx: float, cy: float) -> None:
    height, width = frame.shape[:2]
    h, w = al.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    sx, sy = max(0, -x0), max(0, -y0)
    ex0, ey0, ex1, ey1 = max(0, x0), max(0, y0), min(width, x0 + w), min(height, y0 + h)
    if ex1 > ex0 and ey1 > ey0:
        a = al[sy:sy + ey1 - ey0, sx:sx + ex1 - ex0, None]
        frame[ey0:ey1, ex0:ex1] = frame[ey0:ey1, ex0:ex1] * (1 - a) + col[sy:sy + ey1 - ey0, sx:sx + ex1 - ex0] * a


def busy_stack(frames: list[np.ndarray], threshold: float, every: int = 6) -> dict[int, np.ndarray]:
    """Where the target already has ink (its own drops, ring lines, sharp edges): luminance far from its local mean."""
    out = {}
    for i in range(0, len(frames), every):
        lum = frames[i].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        out[i] = np.abs(lum - ndi.gaussian_filter(lum, 9)) > threshold
    return out


def depth_value(x: float, y: float, width: int, height: int, target: dict) -> float:
    wx, wy = target.get("depth", {}).get("near_weight", [0.6, 0.4])
    far = wx * x / width + wy * y / height
    t = float(np.clip((far - 0.1) / 0.7, 0, 1))
    near, far_scale = target["scale"]["near"], target["scale"]["far"]
    return near + (far_scale - near) * t


def plan_on_target(source_spec: dict, target: dict, size: tuple[int, int], busy: dict[int, np.ndarray], total: int, life: int, p: dict) -> list[dict]:
    """Seeded places on free water: inside the allowed rectangles, away from the target's own ink during the whole life of the copy, spread in space and time."""
    width, height = size
    rnd = random.Random(p["seed"])
    ev = source_spec["event"]
    max_fraction = target.get("busy", {}).get("max_fraction", 0.09)
    n = p["copies"]
    span = max(1, total - p["stagger_end"])
    copies: list[dict] = []
    for index in range(n):
        start = min(span - 1, int((index + rnd.random() * 0.8) / n * span))
        frames = [t for t in busy if start - 6 <= t <= start + life]
        for _ in range(p["tries"]):
            x0, y0, x1, y1 = rnd.choices(target["allowed"], weights=[(r[2] - r[0]) * (r[3] - r[1]) for r in target["allowed"]])[0]  # zones weighted by their area
            x, y = rnd.uniform(x0, x1), rnd.uniform(y0, y1)
            scale = depth_value(x, y, width, height, target)
            rx, ry = ev["rx"] * scale, ev["ry"] * scale
            ya, yb, xa, xb = int(max(0, y - ry)), int(min(height, y + ry)), int(max(0, x - rx)), int(min(width, x + rx))
            if yb - ya < 4 or xb - xa < 4:
                continue
            if any(float(busy[t][ya:yb, xa:xb].mean()) > max_fraction for t in frames):
                continue
            radius = max(rx, ry)
            if any(abs(c["start"] - start) < life and math.hypot(x - c["x"], y - c["y"]) < (radius + c["radius"]) * 0.9 for c in copies):
                continue
            copies.append({"id": index, "x": round(x, 1), "y": round(y, 1), "scale": round(scale, 3), "start": start, "radius": round(radius, 1)})
            break
    return copies


def decode(clip: Path, folder: Path, prefix: str) -> list[Path]:
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(clip), str(folder / f"{prefix}_%04d.png")], check=True)
    return sorted(folder.glob(f"{prefix}_*.png"))


def load_event(source_clip: Path, source_still: Path, source_spec: dict, folder: Path):
    frames = decode(source_clip, folder, "s")
    first = np.asarray(Image.open(frames[0]).convert("RGB"))
    turned = np.asarray(Image.open(source_still).convert("RGB").transpose(Image.Transpose.ROTATE_270))
    assert turned.shape[:2] == first.shape[:2], f"still {turned.shape[:2]} does not match the clip {first.shape[:2]}"
    bg = dm.calm_water(turned, source_spec["inpaint"])
    layers, box = dm.extract_event(frames, bg, source_spec, {**dm.DEFAULTS})
    return layers


def render(target: Path, source_clip: Path, source_still: Path, out: Path, source_spec: dict, target_spec: dict, style, overrides: dict | None = None, extra: dict | None = None) -> dict:
    """`style` is a name of STYLES or a ready style dict (see resolve_style); `extra` (base video info, defaults used...) goes into the JSON record."""
    p = {**DEFAULTS, **(overrides or {})}
    style_name, style_spec = (style, STYLES[style]) if isinstance(style, str) else (style.get("name", "custom"), style)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        layers = load_event(source_clip, source_still, source_spec, tmp)
        frames = [np.asarray(Image.open(f).convert("RGB")) for f in decode(target, tmp, "t")]
        height, width = frames[0].shape[:2]
        total = len(frames)
        busy = busy_stack(frames, target_spec.get("busy", {}).get("threshold", 28.0))
        probe = Impact(layers, 1.0, style_spec, 0, p)
        copies = plan_on_target(source_spec, target_spec, (width, height), busy, total, probe.life, p)
        impacts = {c["id"]: Impact(layers, c["scale"], style_spec, p["seed"] * 100 + c["id"], p) for c in copies}
        cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for j in range(total):
            frame = frames[j].astype(np.float32)
            for c in copies:
                age = j - c["start"]
                if 0 <= age < impacts[c["id"]].life:
                    key = (c["id"], age)
                    if key not in cache:
                        cache[key] = impacts[c["id"]].render(age)
                    col, al = cache[key]
                    paste(frame, col, al, c["x"], c["y"])
            Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).save(tmp / f"o_{j:04d}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-framerate", str(p["fps"]), "-i", str(tmp / "o_%04d.png"), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out)], check=True)
    record = {"kind": "post", "id": out.stem, "operation": "drop_transplant", "target": str(target), "source_clip": str(source_clip), "source_still": str(source_still), "output": str(out),
              "style": style_name, "style_label": style_spec["label"], "style_spec": style_spec, "params": p, "source_spec": source_spec, "target_spec": target_spec, "copies": copies,
              "frames_out": total, "fps": p["fps"], **(extra or {})}
    out.with_suffix(".transplant.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


def planche(target: Path, source_clip: Path, source_still: Path, out_png: Path, source_spec: dict, target_spec: dict, styles: list[str], ages=(4, 12, 20, 30), seed: int = 1) -> None:
    """Contact sheet: one row per outline treatment of the SAME impact, columns = moments of its life, near copy on the left, far copy (magnified) on the right,
    drawn over real frames of the target."""
    p = {**DEFAULTS, "seed": seed}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        layers = load_event(source_clip, source_still, source_spec, tmp)
        frames = [np.asarray(Image.open(f).convert("RGB")) for f in decode(target, tmp, "t")]
        near_scale, far_scale = target_spec["scale"]["near"], target_spec["scale"]["far"]
        blocks = [("proche", near_scale, target_spec.get("planche", {}).get("near_at", [200, 360]), (190, 250), 1), ("loin", far_scale, target_spec.get("planche", {}).get("far_at", [330, 620]), (95, 125), 2)]
        tile_w, tile_h = 190, 250
        label_w, head_h = 150, 34
        cols = len(ages)
        sheet = Image.new("RGB", (label_w + 2 * cols * (tile_w + 4) + 12, head_h + len(styles) * (tile_h + 4)), (16, 16, 20))
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 14)
            font_small = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 12)
        except OSError:
            font = font_small = ImageFont.load_default()
        for r, name in enumerate(styles):
            y0 = head_h + r * (tile_h + 4)
            draw.text((6, y0 + 6), name, font=font, fill=(255, 214, 10))
            for wrap_i, line in enumerate(STYLES[name]["label"].split(" + ") if len(STYLES[name]["label"]) > 18 else [STYLES[name]["label"]]):
                draw.text((6, y0 + 28 + 16 * wrap_i), line, font=font_small, fill=(200, 200, 210))
        for b, (block_name, scale, at, (cw, ch), magnify) in enumerate(blocks):
            for c, age in enumerate(ages):
                x0 = label_w + (b * cols + c) * (tile_w + 4) + (12 if b else 0)
                draw.text((x0 + 2, 8), f"{block_name} - age {age}", font=font_small, fill=(255, 255, 255))
                for r, name in enumerate(styles):
                    impact = Impact(layers, scale, name, seed * 100, p)
                    frame = frames[min(len(frames) - 1, 14 + age)].astype(np.float32).copy()
                    if age < impact.life:
                        col, al = impact.render(age)
                        paste(frame, col, al, at[0], at[1])
                    cx0, cy0 = int(at[0] - cw / 2), int(at[1] - ch / 2)
                    crop = np.clip(frame[max(0, cy0):cy0 + ch, max(0, cx0):cx0 + cw], 0, 255).astype(np.uint8)
                    tile = Image.fromarray(crop).resize((crop.shape[1] * magnify, crop.shape[0] * magnify), Image.LANCZOS if magnify > 1 else Image.NEAREST)
                    sheet.paste(tile, (x0, head_h + r * (tile_h + 4)))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("render", "planche"):
        s = sub.add_parser(name)
        s.add_argument("target", type=Path)
        s.add_argument("source", type=Path)
        s.add_argument("source_still", type=Path)
        s.add_argument("out", type=Path)
        s.add_argument("--source-spec", type=Path, required=True)
        s.add_argument("--target-spec", type=Path, required=True)
        s.add_argument("--seed", type=int, default=1)
        if name == "render":
            s.add_argument("--style", choices=list(STYLES), default="burst")
            s.add_argument("--outline", choices=list(OUTLINES), help="pencil outline alone (replaces the one of --style)")
            s.add_argument("--burst", dest="burst", action="store_true", default=None)
            s.add_argument("--no-burst", dest="burst", action="store_false")
            s.add_argument("--copies", type=int)
        else:
            s.add_argument("--styles", default=",".join(STYLES))
    u = sub.add_parser("union", help="base video + crown impact with the declared defaults, any option replaces its default")
    u.add_argument("out", type=Path, nargs="?")
    u.add_argument("--target", type=Path)
    u.add_argument("--source", type=Path)
    u.add_argument("--source-still", type=Path)
    u.add_argument("--source-spec", type=Path)
    u.add_argument("--target-spec", type=Path)
    u.add_argument("--method", choices=["auto", "qwen", "transplant"])
    u.add_argument("--base-still", type=Path)
    u.add_argument("--url", help="ComfyUI of the server (through the ssh tunnel), default from the declared defaults")
    u.add_argument("--style", choices=list(STYLES))
    u.add_argument("--outline", choices=list(OUTLINES))
    u.add_argument("--burst", dest="burst", action="store_true", default=None)
    u.add_argument("--no-burst", dest="burst", action="store_false")
    u.add_argument("--copies", type=int)
    u.add_argument("--seed", type=int)
    u.add_argument("--show-defaults", action="store_true")
    args = parser.parse_args()
    started = time.time()
    if args.command == "union":
        defaults = load_union_defaults()
        if args.show_defaults or args.out is None:
            print(json.dumps(defaults, indent=1))
            return
        given = {k: v for k, v in vars(args).items() if k in defaults and v is not None}
        chosen = {**defaults, **{k: (str(v) if isinstance(v, Path) else v) for k, v in given.items()}}
        union_info = json.loads(RECIPES.read_text(encoding="utf-8"))["rain_union"]
        method = chosen.get("method", "auto")
        if method == "auto":
            import comfy_edit as ce
            method = "qwen" if ce.qwen_available(chosen.get("url", ce.BASE_URL)) else "transplant"
            print(f"method auto -> {method} ({'Qwen-Image-Edit is in the bundle and on the server' if method == 'qwen' else 'Qwen not available here: CPU transplant'})")
        if method == "qwen":
            import comfy_edit as ce
            out = args.out if args.out.suffix.lower() == ".png" else args.out.with_suffix(".png")
            entry = ce.run_edit(Path(chosen["base_still"]), out, ce.MIX, seed=chosen["seed"], registry=ROOT / "results" / "trend_rain_anime" / "generations.json", label=out.stem,
                                base_url=chosen.get("url", ce.BASE_URL), references=[Path(chosen["source_still"])])
            record = {"kind": "union", "method": "qwen", "output": str(out), "prompt": ce.MIX, "defaults_declared": defaults, "overridden": sorted(given),
                      "base_video": union_info["base_video"], "impact": union_info["impact"], "edit": entry}
            out.with_suffix(".union.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"union (qwen, seed {chosen['seed']}): {entry['seconds']} s -> {out}; overridden: {sorted(given) or 'nothing'}")
            return
        if args.out.suffix.lower() != ".mp4":
            args.out = args.out.with_suffix(".mp4")
        # an explicit outline or burst replaces the default style as a whole, so the default style does not leak into the combination
        style_name, style_spec = resolve_style(chosen["style"] if (args.outline is None and args.burst is None) else (args.style or None), args.outline, args.burst)
        extra = {"union": True, "method": "transplant", "defaults_declared": defaults, "overridden": sorted(given), "base_video": union_info["base_video"], "impact": union_info["impact"]}
        overrides = {"seed": chosen["seed"], "copies": chosen["copies"]}
        record = render(Path(chosen["target"]), Path(chosen["source"]), Path(chosen["source_still"]), args.out, json.loads(Path(chosen["source_spec"]).read_text(encoding="utf-8")),
                        json.loads(Path(chosen["target_spec"]).read_text(encoding="utf-8")), {**style_spec, "name": style_name}, overrides, extra)
        print(f"union ({style_name}): {len(record['copies'])} copies placed, {record['frames_out']} frames, in {time.time() - started:.1f} s; overridden: {sorted(given) or 'nothing'}")
        return
    source_spec = json.loads(args.source_spec.read_text(encoding="utf-8"))
    target_spec = json.loads(args.target_spec.read_text(encoding="utf-8"))
    if args.command == "render":
        overrides = {"seed": args.seed, **({"copies": args.copies} if args.copies else {})}
        style = args.style
        if args.outline is not None or args.burst is not None:
            name, spec = resolve_style(args.style, args.outline, args.burst)
            style = {**spec, "name": name}
        record = render(args.target, args.source, args.source_still, args.out, source_spec, target_spec, style, overrides)
        print(f"{len(record['copies'])} copies placed, {record['frames_out']} frames, in {time.time() - started:.1f} s")
    else:
        planche(args.target, args.source, args.source_still, args.out, source_spec, target_spec, args.styles.split(","), seed=args.seed)
        print("wrote", args.out, f"in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
