"""Where is the face in a picture? A light detector (scikit-image LBP cascade: no OpenCV, no download) to
  - choose among several close-up candidates the one whose face is CENTRED and big enough (the S2V model needs real pixels on the eyes);
  - give the face box that the quality checks (scripts/s2v_face_test.py) measure.

    detect_faces(image) -> [{"cx", "cy", "w", "h"}] as fractions of the picture, biggest first
    pick_closeup(paths) -> (best path, its face)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        from skimage import data
        from skimage.feature import Cascade
        _DETECTOR = Cascade(data.lbp_frontal_face_cascade_filename())
    return _DETECTOR


def detect_faces(image: Image.Image | Path | str, min_fraction: float = 0.10, max_fraction: float = 0.75) -> list[dict]:
    """Faces found, biggest first, as fractions of the picture (centre x/y, width, height). The cascade fires several times on one face: boxes that overlap a bigger one are dropped."""
    picture = image if isinstance(image, Image.Image) else Image.open(image)
    gray = np.asarray(picture.convert("L"))
    height, width = gray.shape
    found = _detector().detect_multi_scale(img=gray, scale_factor=1.15, step_ratio=1, min_size=(int(width * min_fraction), int(width * min_fraction)),
                                           max_size=(int(width * max_fraction), int(width * max_fraction)))
    faces = sorted(({"cx": (b["c"] + b["width"] / 2) / width, "cy": (b["r"] + b["height"] / 2) / height, "w": b["width"] / width, "h": b["height"] / height} for b in found),
                   key=lambda f: -f["w"] * f["h"])
    kept: list[dict] = []
    for face in faces:
        if not any(abs(face["cx"] - k["cx"]) < k["w"] * 0.5 and abs(face["cy"] - k["cy"]) < k["h"] * 0.5 for k in kept):
            kept.append(face)
    return kept


def face_score(face: dict, target: tuple[float, float] = (0.5, 0.40), wanted_width: float = 0.38) -> float:
    """Higher = better close-up: centred on the target point and about `wanted_width` of the frame wide (too small = no pixels for the eyes, too big = the head is cropped)."""
    return -3.0 * abs(face["cx"] - target[0]) - 2.0 * abs(face["cy"] - target[1]) - 2.5 * abs(face["w"] - wanted_width)


def pick_closeup(paths: list[Path]) -> tuple[Path, dict] | None:
    """The candidate whose biggest face scores best; None when no candidate shows a face."""
    best = None
    for path in paths:
        faces = detect_faces(path)
        if not faces:
            continue
        score = face_score(faces[0])
        if best is None or score > best[0]:
            best = (score, path, faces[0])
    return (best[1], best[2]) if best else None


def face_box(face: dict, margin: float = 0.15) -> tuple[float, float, float, float]:
    """x0, y0, x1, y1 (fractions) around a detected face, with a margin, clipped to the picture."""
    half_w, half_h = face["w"] / 2 * (1 + margin), face["h"] / 2 * (1 + margin)
    return (max(0.0, face["cx"] - half_w), max(0.0, face["cy"] - half_h), min(1.0, face["cx"] + half_w), min(1.0, face["cy"] + half_h))
