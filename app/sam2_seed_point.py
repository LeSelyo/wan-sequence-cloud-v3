"""Computes the SAM2 seed point that Wan 2.2 Animate's official template
needs for its PointsEditor node.

The pinned template (video_wan2_2_14B_animate.json) ships a PointsEditor
with a hardcoded UI-default point ({"x":256,"y":256} on a 640x640 canvas) --
meant for a human to click on the subject in ComfyUI's web UI before
running the graph. Submitted headlessly through this app's API, that point
is never updated, so SAM2's video segmentor seeds from whatever happens to
be at (256,256) in the driving video's first frame -- for most shots that
misses the subject, producing a broken/partial character mask and a near
pass-through of the original driving video (confirmed live on 2026-10-01:
the resulting clip barely replaced the character at all, until the seed
point was corrected by hand).

This module replaces that hardcoded point with one computed from a real
YOLO person-detection pass on the driving video's first frame, run through
the same vendored detector ComfyUI-WanAnimatePreprocess's own
PoseAndFaceDetection node uses -- so the seed point actually lands on the
subject.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# cv2/numpy aren't in requirements.txt: this module only runs in the shipped
# container, where app/ and ComfyUI share one venv (both launched from
# /opt/venv/bin/python, confirmed 2026-10-01) and cv2 is already pulled in
# by ComfyUI-WanAnimatePreprocess's own dependencies. Don't import this
# module in a context that doesn't have that venv.
import cv2
import numpy as np

CUSTOM_NODE_PKG_PATH = Path("/opt/ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess")
_PKG_NAME = "wan_animate_preprocess_pkg"

# Confirmed on the pinned template (video_wan2_2_14B_animate.json, fetched
# 2026-09-27): the PointsEditor node declares its own canvas as a literal
# 640x640 (not bound to any upstream width/height). If a future template
# revision changes this, build_bindings() asserts on it rather than silently
# computing a point for the wrong canvas size -- see build_bindings in
# scripts/prepare_workflows.py.
POINTS_EDITOR_CANVAS = 640


def _load_yolo_class():
    if _PKG_NAME in sys.modules:
        return sys.modules[_PKG_NAME + ".nodes"].Yolo
    if str(CUSTOM_NODE_PKG_PATH.parent) not in sys.path:
        sys.path.insert(0, "/opt/ComfyUI")
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME,
        CUSTOM_NODE_PKG_PATH / "__init__.py",
        submodule_search_locations=[str(CUSTOM_NODE_PKG_PATH)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_NAME] = pkg
    spec.loader.exec_module(pkg)
    nodes_spec = importlib.util.spec_from_file_location(
        f"{_PKG_NAME}.nodes",
        CUSTOM_NODE_PKG_PATH / "nodes.py",
        submodule_search_locations=[str(CUSTOM_NODE_PKG_PATH)],
    )
    nodes_mod = importlib.util.module_from_spec(nodes_spec)
    nodes_mod.__package__ = _PKG_NAME
    sys.modules[f"{_PKG_NAME}.nodes"] = nodes_mod
    nodes_spec.loader.exec_module(nodes_mod)
    return nodes_mod.Yolo


def compute_seed_point(driving_video_path: Path, yolo_model_path: Path) -> dict:
    """Returns the PointsEditor input values (points_store/coordinates/
    neg_coordinates, all JSON strings) for the given driving video, mapped
    into the template's fixed 640x640 PointsEditor canvas.

    Raises RuntimeError if no person is detected in the first frame -- a
    silently-wrong default point is worse than a clear failure here, since
    the animate pipeline has no other signal that the seed point was bad
    until a human reviews the output video.
    """
    cap = cv2.VideoCapture(str(driving_video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {driving_video_path}")

    h0, w0 = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    Yolo = _load_yolo_class()
    yolo = Yolo(str(yolo_model_path), "CUDAExecutionProvider")
    yolo.reinit()
    try:
        shape = np.array([h0, w0])[None]
        detection = yolo(cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None], shape)[0][0]
        bbox = detection.get("bbox")
    finally:
        yolo.cleanup()

    if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
        raise RuntimeError(
            f"no person detected in the first frame of {driving_video_path} "
            "-- cannot compute a SAM2 seed point for wan22_animate"
        )

    x1, y1, x2, y2 = bbox[:4]
    cx = (x1 + x2) / 2
    # Chest-level (not bbox-center) stays on the torso through arm/leg
    # motion better than a pure bbox centroid, since SAM2's video segmentor
    # only gets this one seed for the whole clip (confirmed empirically:
    # 2026-10-01 test used this same 0.35 offset successfully).
    chest_y = y1 + (y2 - y1) * 0.35

    # Replicate the template's own ImageScale(upscale_method=lanczos,
    # crop=center) transform into the PointsEditor's 640x640 canvas: scale
    # to cover, then center-crop.
    canvas = POINTS_EDITOR_CANVAS
    scale = max(canvas / w0, canvas / h0)
    scaled_w, scaled_h = w0 * scale, h0 * scale
    crop_x = (scaled_w - canvas) / 2
    crop_y = (scaled_h - canvas) / 2
    px = min(canvas - 1, max(0, cx * scale - crop_x))
    py = min(canvas - 1, max(0, chest_y * scale - crop_y))
    px, py = round(px), round(py)

    # A negative point anchored at a corner, mirroring the template's own
    # default negative point convention, keeps SAM2 from also claiming
    # background in that corner as foreground.
    neg = {"x": 5, "y": 5}
    positive = {"x": px, "y": py}
    return {
        "points_store": json.dumps({"positive": [positive], "negative": [neg]}),
        "coordinates": json.dumps([positive]),
        "neg_coordinates": json.dumps([neg]),
    }
