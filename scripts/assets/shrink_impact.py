# -*- coding: utf-8 -*-
"""Make a SMALLER version of an impact in a still (calm_a): the big crown is scaled down around its centre, the water it used to cover is rebuilt by diffusion from
the surrounding water, so the video model has room to generate other, similar impacts. usage: shrink_impact.py src dst scale [cx cy rx ry]"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

src, dst, scale = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
# default = the big crown of calm_a: ring ellipse centre and radii, plus the falling drop above it and the tail below it
cx, cy, rx, ry = (float(v) for v in sys.argv[4:8]) if len(sys.argv) >= 8 else (322.0, 420.0, 200.0, 105.0)
img = np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
h, w = img.shape[:2]
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
mask = (ellipse <= 1.0).astype(np.float32)
mask = np.maximum(mask, ((abs(xx - 300) < 38) & (yy > 215) & (yy < cy)).astype(np.float32))  # the falling drop and the splash column above
mask = np.maximum(mask, ((abs(xx - 318) < 30) & (yy > cy) & (yy < h)).astype(np.float32))  # the tail below
mask = ndi.binary_dilation(mask > 0, iterations=10).astype(np.float32)
soft = ndi.gaussian_filter(mask, 6)

# 1. rebuild the water under the old impact: multi-scale diffusion from the pixels around it
known = mask < 0.5
filled = img.copy()
first_level = True
for factor in (8, 4, 2, 1):
    small_h, small_w = h // factor, w // factor
    small = np.asarray(Image.fromarray(np.clip(filled, 0, 255).astype(np.uint8)).resize((small_w, small_h), Image.LANCZOS)).astype(np.float32)
    k = np.asarray(Image.fromarray((known * 255).astype(np.uint8)).resize((small_w, small_h), Image.NEAREST)) > 127
    cur = small.copy()
    if first_level:  # coarsest level starts from the mean colour; finer levels start from the previous (coarser) fill
        cur[~k] = small[k].mean(axis=0)
        first_level = False
    for _ in range(1500 if factor >= 8 else (400 if factor >= 4 else 120)):
        blurred = ndi.uniform_filter(cur, size=(5, 5, 1), mode="nearest")
        cur = np.where(k[..., None], small, blurred)
    up = np.asarray(Image.fromarray(np.clip(cur, 0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)).astype(np.float32)
    filled = np.where(known[..., None], img, up)
# a light grain so the rebuilt water does not look smeared
rng = np.random.default_rng(3)
filled = np.where(known[..., None], filled, filled + rng.normal(0, 2.0, filled.shape))
base = img * (1 - soft[..., None]) + filled * soft[..., None]

# 2. paste the old impact scaled down around its centre, with a soft edge
m = int(max(rx, ry) * 1.9)
x0, y0, x1, y1 = int(max(0, cx - rx - 20)), int(215), int(min(w, cx + rx + 20)), h
patch = img[y0:y1, x0:x1]
patch_mask = soft[y0:y1, x0:x1]
new_w, new_h = int((x1 - x0) * scale), int((y1 - y0) * scale)
patch_small = np.asarray(Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)).astype(np.float32)
mask_small = np.asarray(Image.fromarray((patch_mask * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)).astype(np.float32) / 255.0
mask_small = ndi.gaussian_filter(mask_small, 2) * ndi.gaussian_filter((np.ones_like(mask_small) * 0 + 1), 1)
# the shrunk patch is centred on the old ring centre
px = int(cx - (cx - x0) * scale)
py = int(cy - (cy - y0) * scale)
out = base.copy()
ys, xs = slice(py, py + new_h), slice(px, px + new_w)
region = out[ys, xs]
alpha = mask_small[: region.shape[0], : region.shape[1], None]
out[ys, xs] = region * (1 - alpha) + patch_small[: region.shape[0], : region.shape[1]] * alpha
dst.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(dst)
print("wrote", dst, "scale", scale, "centre", (cx, cy), "radii", (rx, ry))
Path(str(dst) + ".json").write_text(json.dumps({"source": str(src), "operation": "shrink_impact", "scale": scale, "centre": [cx, cy], "radii": [rx, ry]}), encoding="utf-8")
