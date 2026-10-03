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

# cv2/numpy aren't in requirements.txt: detection only runs in the shipped
# container, where app/ and ComfyUI share one venv (both launched from
# /opt/venv/bin/python, confirmed 2026-10-01) and cv2 is already pulled in
# by ComfyUI-WanAnimatePreprocess's own dependencies. They are imported lazily
# inside compute_seed_point so that this module (and therefore app.orchestrator)
# still imports in a plain dev/test environment; only the YOLO pass needs them.

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


def _clamp_point(x: float, y: float) -> dict:
    canvas = POINTS_EDITOR_CANVAS
    return {"x": round(min(canvas - 1, max(0, x))), "y": round(min(canvas - 1, max(0, y)))}


def seed_point_values(
    x: float, y: float, negatives: list[tuple[float, float]] | None = None,
    extra_positives: list[tuple[float, float]] | None = None,
) -> dict:
    """PointsEditor input values (all JSON strings) for one positive click at
    (x, y) on the template's 640x640 canvas.

    PointsEditor has two kinds of points (red/green in its web UI): POSITIVE
    (green) marks the subject SAM2 must segment -- the person to replace -- and
    NEGATIVE (red) marks what it must leave out (background, the other fighter).
    The official template only wires the positive output into Sam2Segmentation
    (confirmed on the template JSON 2026-10-02: `coordinates_negative` has no
    link), which prepare_workflows.py now connects. `negatives` are canvas
    coordinates; without any, the template's own corner default is kept in the
    editor widgets (and, unless the shot asked for red points, stays unwired).
    """
    positives = [_clamp_point(x, y)] + [_clamp_point(px, py) for px, py in (extra_positives or [])]
    negative = [_clamp_point(nx, ny) for nx, ny in negatives] if negatives else [{"x": 5, "y": 5}]
    return {
        "points_store": json.dumps({"positive": positives, "negative": negative}),
        "coordinates": json.dumps(positives),
        "neg_coordinates": json.dumps(negative),
    }


def _canvas_points(points) -> list[tuple[float, float]]:
    out = []
    for x, y in points:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("points must be (x, y) with both values between 0 and 1")
        out.append((x * POINTS_EDITOR_CANVAS, y * POINTS_EDITOR_CANVAS))
    return out


def manual_seed_point(subject_point: tuple[float, float], exclude_points=()) -> dict:
    """Seed from an explicit (x, y) in 0..1 coordinates of the (already square,
    see app/driving_video.py) driving clip's first frame, plus optional red
    "exclude" points in the same coordinates.

    Needed whenever YOLO's "best person" is not the one to replace -- e.g. a
    two-fighter clip where the second fighter is the character to swap
    (2026-10-02 cat-vs-cat test: auto picked the wrong fighter; the point had to
    be set by hand). The subject must be visible in frame 0: SAM2 only gets this
    single click for the whole clip, so someone who enters the shot later is
    never segmented.
    """
    (x, y), = _canvas_points([subject_point])
    return seed_point_values(x, y, _canvas_points(exclude_points))


def compute_seed_point(driving_video_path: Path, yolo_model_path: Path, exclude_points=()) -> dict:
    """Returns the PointsEditor input values (points_store/coordinates/
    neg_coordinates, all JSON strings) for the given driving video, mapped
    into the template's fixed 640x640 PointsEditor canvas.

    Raises RuntimeError if no person is detected in the first frame -- a
    silently-wrong default point is worse than a clear failure here, since
    the animate pipeline has no other signal that the seed point was bad
    until a human reviews the output video.
    """
    import cv2
    import numpy as np

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
    return seed_point_values(cx * scale - crop_x, chest_y * scale - crop_y, _canvas_points(exclude_points))


def animal_clicks(x1: float, y1: float, x2: float, y2: float) -> list[tuple[float, float]]:
    """Green SAM2 clicks for an animal bbox (canvas coordinates) as (x, y) pairs:
    the centre first, then one a quarter of the way in from each side and one higher on the back.
    A single click on a quadruped made SAM2 segment only a patch of it (live, dog clip)."""
    width, height = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [(cx, cy), (x1 + 0.25 * width, cy), (x1 + 0.75 * width, cy), (cx, y1 + 0.3 * height)]


def compute_animal_seed_point(driving_video_path: Path, exclude_points=()) -> dict:
    """Green SAM2 point for a driving video whose subject is an animal.

    yolov10m (compute_seed_point) is a person detector, so it finds nothing on a cat.
    comfyui_controlnet_aux's own YOLOX detector, class-filtered to the COCO animal
    classes 14-23 (bird..giraffe, includes cat and dog), is what its AnimalPose
    estimator runs on; this reuses it on the first frame (live-verified on a cat clip
    2026-10-02) and puts the point at the bbox center, which for a quadruped is on
    the body (a human's chest offset would be wrong).
    """
    import cv2
    import numpy as np

    src = "/opt/ComfyUI/custom_nodes/comfyui_controlnet_aux/src"
    if src not in sys.path:
        sys.path.insert(0, src)
    if "/opt/ComfyUI" not in sys.path:
        sys.path.insert(0, "/opt/ComfyUI")
    from custom_controlnet_aux.dwpose.animalpose import AnimalPoseImage
    from custom_controlnet_aux.dwpose.dw_onnx.cv_ox_det import inference_detector
    from custom_controlnet_aux.util import custom_hf_download

    cap = cv2.VideoCapture(str(driving_video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {driving_video_path}")
    h0, w0 = frame.shape[:2]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detector = AnimalPoseImage(custom_hf_download("hr16/yolox-onnx", "yolox_l.onnx"), None, torchscript_device="cuda")
    found = inference_detector(detector.det, image, detect_classes=list(range(14, 24)), dtype=np.float32)
    if found is None or found.shape[0] == 0:
        raise RuntimeError(
            f"no animal detected in the first frame of {driving_video_path} "
            "-- cannot compute a SAM2 seed point (pass subject_point to set it by hand)"
        )
    x1, y1, x2, y2 = found[0][:4]
    canvas = POINTS_EDITOR_CANVAS
    scale = max(canvas / w0, canvas / h0)
    crop_x, crop_y = (w0 * scale - canvas) / 2, (h0 * scale - canvas) / 2
    clicks = animal_clicks(x1 * scale - crop_x, y1 * scale - crop_y, x2 * scale - crop_x, y2 * scale - crop_y)
    return seed_point_values(clicks[0][0], clicks[0][1], _canvas_points(exclude_points), extra_positives=clicks[1:])
