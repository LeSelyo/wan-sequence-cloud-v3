"""Draws the emblem shown in the CENTRE of the screen at the end of the philosopher video: a pair of scales in a GOTHIC / ROCK / ROMAN style.

Roman: an Ionic column (fluted shaft, volute capital, stepped base) carries the beam; laurel branches at its foot. Gothic: a lancet arch with a double
moulding and crockets frames it, a quatrefoil rose window with a red glass heart above, the beam ends in fleur-de-lis spikes, the pans hang from
chains and end in spikes. Rock: bone-and-black engraving with heavy outlines like a tattoo flash, crimson accents, flames. The two pans carry the
subject of the video: a radiant EYE (the gaze, "being seen") against a flaming HEART wrapped in thorns (a life).

Everything is drawn with Pillow (no external asset, no licence to carry), in painter's order at 3x and reduced for clean edges, with a soft dark halo
and a translucent dark arch panel so it reads on any picture. `python scripts/make_balance_png.py` writes scripts/assets/balance.png;
`draw_balance(height)` returns the image (transparent RGBA, cropped to the drawing, taller than wide)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 3  # supersampling
W, H = 1000, 1050  # the design grid
INK = (8, 8, 10, 255)
BONE = (228, 221, 202, 255)
ASH = (150, 145, 134, 255)
RED = (172, 22, 36, 255)
RED_DARK = (92, 10, 18, 255)
FLAME = (255, 146, 36, 255)
FLAME_Y = (255, 214, 92, 255)
PANEL = (0, 0, 0, 125)


def _pt(p):
    return (p[0] * S, p[1] * S)


def poly(d, pts, fill=BONE, ink=5):
    pts = list(pts)
    if fill:
        d.polygon([_pt(p) for p in pts], fill=fill)
    if ink:
        d.line([_pt(p) for p in pts + [pts[0]]], fill=INK, width=round(ink * S), joint="curve")


def line(d, pts, width, fill=INK):
    d.line([_pt(p) for p in pts], fill=fill, width=round(width * S), joint="curve")


def disc(d, c, r, fill=BONE, ink=5):
    d.ellipse([(c[0] - r) * S, (c[1] - r) * S, (c[0] + r) * S, (c[1] + r) * S], fill=fill, outline=INK if ink else None, width=round(ink * S))


def ellipse_pts(c, rx, ry, rot=0.0, n=28):
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = rx * math.cos(a), ry * math.sin(a)
        out.append((c[0] + x * math.cos(rot) - y * math.sin(rot), c[1] + x * math.sin(rot) + y * math.cos(rot)))
    return out


def mirror(pts):
    return [(W - x, y) for x, y in pts]


def arch_points(left, right, spring_y, r, n=48):
    """lancet arch: two arcs of radius r meeting at the apex; returns the left side, from the spring point to the apex"""
    cx, cy = left + r, spring_y
    apex_angle = math.acos((W / 2 - cx) / r)  # the angle where x = W / 2
    pts = []
    for i in range(n + 1):
        a = math.pi - (math.pi - apex_angle) * i / n
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    return pts, (cx, cy)


def draw_arch(d):
    spring, base = 690, 1005
    outer, centre = arch_points(100, 900, spring, 600)
    # translucent dark panel inside the arch (it makes the emblem readable over any picture)
    inner, _ = arch_points(122, 878, spring, 578)
    panel = inner + mirror(inner[::-1]) + [(878, base), (122, base)]
    d.polygon([_pt(p) for p in panel], fill=PANEL)
    # the moulding: ink band, bone band, a second thin line inside
    full = [(100, base)] + outer + mirror(outer[::-1]) + [(900, base)]
    line(d, full, 24, INK)
    line(d, full, 14, BONE)
    full_in = [(122, base)] + inner + mirror(inner[::-1]) + [(878, base)]
    line(d, full_in, 6, INK)
    # crockets: little leaf spikes on the outside of the arch
    cx, cy = centre
    for k in range(3, len(outer) - 1, 4):
        x, y = outer[k]
        nx, ny = (x - cx), (y - cy)
        length = math.hypot(nx, ny)
        nx, ny = nx / length, ny / length
        tx, ty = -ny, nx
        base_a = (x - tx * 8 + nx * 10, y - ty * 8 + ny * 10)
        base_b = (x + tx * 8 + nx * 10, y + ty * 8 + ny * 10)
        tip = (x + nx * 36, y + ny * 36)
        for sign in (1, -1):
            if sign == 1:
                poly(d, [base_a, base_b, tip], BONE, 4)
            else:
                poly(d, mirror([base_a, base_b, tip]), BONE, 4)
    # finial at the apex: a spike with a cross-piece
    ax, ay = outer[-1]
    poly(d, [(ax - 14, ay + 6), (ax + 14, ay + 6), (ax + 10, ay - 22), (ax, ay - 64), (ax - 10, ay - 22)], BONE, 5)
    poly(d, [(ax - 26, ay - 20), (ax + 26, ay - 20), (ax + 26, ay - 10), (ax - 26, ay - 10)], BONE, 4)


def draw_rose(d):
    """quatrefoil window with red glass above the beam"""
    cx, cy = 500, 232
    for dx, dy in ((-24, 0), (24, 0), (0, -24), (0, 24)):
        disc(d, (cx + dx, cy + dy), 25, RED_DARK, 6)
    disc(d, (cx, cy), 20, BONE, 5)
    disc(d, (cx, cy), 9, RED, 4)


def draw_laurels(d):
    def branch(p0, p1, c, leaves=9):
        pts = [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0], (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1])
               for t in [i / 30 for i in range(31)]]
        line(d, pts, 9, INK)
        line(d, pts, 4, BONE)
        for i in range(leaves):
            t = (i + 1) / (leaves + 1)
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1]
            dx = 2 * (1 - t) * (c[0] - p0[0]) + 2 * t * (p1[0] - c[0])
            dy = 2 * (1 - t) * (c[1] - p0[1]) + 2 * t * (p1[1] - c[1])
            ang = math.atan2(dy, dx)
            for side in (-1, 1):
                a = ang + side * 0.75
                lc = (x + math.cos(a) * 19, y + math.sin(a) * 19)
                poly(d, ellipse_pts(lc, 20, 8, a, 18), BONE, 3.5)
                line(d, [(x, y), (x + math.cos(a) * 38, y + math.sin(a) * 38)], 2.5, INK)

    branch((438, 986), (150, 924), (290, 1030))
    branch((562, 986), (850, 924), (710, 1030))


def draw_column(d):
    # shaft, tapering, with five flutes
    poly(d, [(466, 408), (534, 408), (528, 884), (472, 884)], BONE, 5)
    for k in range(-2, 3):
        line(d, [(500 + k * 11, 420), (500 + k * 9, 876)], 3, INK)
        line(d, [(505 + k * 11, 420), (505 + k * 9, 876)], 2, ASH)
    # capital: abacus, echinus, two volutes
    poly(d, [(432, 372), (568, 372), (568, 394), (432, 394)], BONE, 5)
    poly(d, [(450, 394), (550, 394), (532, 414), (468, 414)], BONE, 5)
    for sx in (-1, 1):
        c = (500 + sx * 66, 408)
        disc(d, c, 24, BONE, 5)
        spiral = []
        for i in range(60):
            a = i * 0.34 * (1 if sx == 1 else -1)
            r = 19 - i * 0.26
            spiral.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
        line(d, spiral, 3, INK)
    # base: tori and plinth
    poly(d, [(458, 884), (542, 884), (548, 906), (452, 906)], BONE, 5)
    poly(d, [(440, 906), (560, 906), (568, 934), (432, 934)], BONE, 5)
    poly(d, [(412, 934), (588, 934), (600, 970), (400, 970)], BONE, 5)
    # neck between the capital and the beam
    poly(d, [(488, 346), (512, 346), (512, 372), (488, 372)], BONE, 4)


def draw_beam(d):
    y = 332
    poly(d, [(228, y - 8), (500, y - 18), (772, y - 8), (772, y + 6), (500, y + 20), (228, y + 6)], BONE, 5)
    # fleur-de-lis spikes on the ends and the rings the chains hang from
    for x in (228, 772):
        poly(d, [(x, y - 52), (x - 11, y - 22), (x, y - 8), (x + 11, y - 22)], BONE, 4)
        poly(d, [(x - 22, y - 30), (x - 8, y - 22), (x - 10, y - 8)], BONE, 3.5)
        poly(d, [(x + 22, y - 30), (x + 8, y - 22), (x + 10, y - 8)], BONE, 3.5)
        disc(d, (x, y + 8), 15, BONE, 5)
        disc(d, (x, y + 8), 6, INK, 0)
    # the pivot: a boss with a red gem, and two horns
    poly(d, [(474, y - 14), (436, y - 46), (456, y - 8)], BONE, 4)
    poly(d, mirror([(474, y - 14), (436, y - 46), (456, y - 8)]), BONE, 4)
    disc(d, (500, y), 36, BONE, 5)
    disc(d, (500, y), 24, ASH, 4)
    disc(d, (500, y), 15, RED, 4)
    disc(d, (494, y - 6), 4, BONE, 0)


def chain(d, p0, p1, link=24):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    ang = math.atan2(dy, dx)
    n = int(length // link)
    for i in range(n):
        t = (i + 0.5) / n
        c = (p0[0] + dx * t, p0[1] + dy * t)
        poly(d, ellipse_pts(c, 13 if i % 2 == 0 else 6, 7, ang, 20), BONE if i % 2 == 0 else ASH, 3)


def draw_pan(d, x):
    rim_y = 706
    chain(d, (x, 340), (x - 90, rim_y - 4))
    chain(d, (x, 340), (x + 90, rim_y - 4))
    bowl = [(x - 96, rim_y)]
    for i in range(1, 24):
        a = math.pi * i / 24
        bowl.append((x - 96 * math.cos(a), rim_y + 70 * math.sin(a)))
    bowl.append((x + 96, rim_y))
    poly(d, bowl, BONE, 5)
    line(d, [(x - 80, rim_y + 22), (x + 80, rim_y + 22)], 3, INK)
    poly(d, [(x - 104, rim_y - 10), (x + 104, rim_y - 10), (x + 98, rim_y + 6), (x - 98, rim_y + 6)], BONE, 5)
    for k in (-1, 0, 1):  # three little spikes on the rim
        poly(d, [(x + k * 60 - 9, rim_y - 10), (x + k * 60 + 9, rim_y - 10), (x + k * 60, rim_y - 34)], BONE, 3.5)
    poly(d, [(x - 13, rim_y + 66), (x + 13, rim_y + 66), (x, rim_y + 104)], BONE, 4)  # the pendant spike


def draw_eye(d, c):
    cx, cy = c
    # radiating rays
    for k in range(9):
        a = math.radians(-168 + k * 19.5)
        r0, r1 = 52, 90 if k % 2 == 0 else 74
        tip = (cx + r1 * math.cos(a), cy + r1 * math.sin(a))
        l = (cx + r0 * math.cos(a - 0.09), cy + r0 * math.sin(a - 0.09))
        r = (cx + r0 * math.cos(a + 0.09), cy + r0 * math.sin(a + 0.09))
        poly(d, [l, tip, r], FLAME_Y, 3.5)
    upper = [(cx + u * 62, cy - 33 * (1 - u * u) ** 0.85) for u in [i / 12 - 1 for i in range(25)]]
    lower = [(cx + u * 62, cy + 33 * (1 - u * u) ** 0.85) for u in [1 - i / 12 for i in range(25)]]
    poly(d, upper + lower, BONE, 5)
    disc(d, c, 22, RED, 5)
    disc(d, c, 9, INK, 0)
    disc(d, (cx - 7, cy - 8), 4, BONE, 0)


def draw_heart(d, c):
    cx, cy = c
    k = 3.0
    pts = []
    for i in range(60):
        t = 2 * math.pi * i / 60
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * k, cy - y * k + 6))
    # flames behind and above the heart

    def flame(fx, base, h, w):
        right = []
        for i in range(25):
            t = math.pi * i / 24
            right.append((fx + (w / 2) * math.sin(t) * (1 - i / 24) ** 0.55, base - h * (0.5 - 0.5 * math.cos(t))))
        pts_f = right + [(2 * fx - x, y) for x, y in right[::-1]]
        poly(d, pts_f, FLAME, 4)
        inner = [(fx + (x - fx) * 0.5, base - (base - y) * 0.62) for x, y in pts_f]
        poly(d, inner, FLAME_Y, 0)
    flame(cx, cy - 20, 88, 56)
    flame(cx - 34, cy - 14, 60, 38)
    flame(cx + 34, cy - 14, 60, 38)
    poly(d, pts, RED, 5)
    poly(d, [(cx - 30 + i * 3.4, cy - 22 - 10 * math.sin(i / 3)) for i in range(5)] + [(cx - 8, cy - 30), (cx - 24, cy - 14)], BONE, 0)  # highlight
    # a thorny vine wound across the heart
    vine = [(cx - 54 + i * 4.5, cy + 10 + 9 * math.sin(i * 0.55)) for i in range(25)]
    line(d, vine, 11, INK)
    line(d, vine, 5, ASH)
    for i in range(1, 24, 3):
        x, y = vine[i]
        up = 1 if (i // 3) % 2 == 0 else -1
        poly(d, [(x - 4, y), (x + 4, y), (x + up * 2, y - up * 15)], INK, 0)


def draw_balance(height: int = 900) -> Image.Image:
    """The emblem as a transparent RGBA image `height` pixels tall (cropped to its drawing and its halo)."""
    big = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    draw_arch(d)
    draw_rose(d)
    draw_laurels(d)
    draw_column(d)
    draw_beam(d)
    draw_pan(d, 228)
    draw_pan(d, 772)
    draw_eye(d, (228, 640))
    draw_heart(d, (772, 648))
    art = np.asarray(big, np.float32).copy()
    # a light from above: the lower part of the bone is a little darker; ink and red keep their colour
    shade = np.linspace(1.0, 0.78, art.shape[0])[:, None, None]
    mask = (art[..., :3].sum(axis=2) > 120)[..., None]
    art[..., :3] = np.where(mask, art[..., :3] * shade, art[..., :3])
    art = Image.fromarray(np.clip(art, 0, 255).astype(np.uint8), "RGBA")
    # a soft dark halo under everything
    alpha = art.split()[3]
    halo_alpha = alpha.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(30 * S / 2)).point(lambda v: min(255, int(v * 0.9)))
    halo = Image.new("RGBA", art.size, (0, 0, 0, 0))
    halo.putalpha(halo_alpha)
    out = Image.alpha_composite(halo, art)
    box = out.getbbox()
    cropped = out.crop(box)
    width = round(cropped.width * height / cropped.height)
    return cropped.resize((width, height), Image.LANCZOS)


def main(argv=None) -> None:
    out = Path(argv[0]) if argv else Path(__file__).resolve().parent / "assets" / "balance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image = draw_balance()
    image.save(out)
    print(f"wrote {out} ({image.width}x{image.height})")


if __name__ == "__main__":
    main(sys.argv[1:])
