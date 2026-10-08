"""Blinks of a clip, counted on the BOX (CPU): MediaPipe Face Mesh gives the eyelid landmarks, the eye aspect ratio (EAR = height of the eye / its width) drops towards 0 when the lid closes.

    PYTHONPATH=/root/eyelibs /root/ttsenv/bin/python eyes_ear.py clip1.mp4 [clip2.mp4 ...]      -> one JSON line per clip on stdout

A BLINK = the EAR stays under 70 % of its usual open value for 1 frame up to 0.6 s. A longer closure is not a blink (the person shuts the eyes: counted apart as `long_closures`, `longest_closed_s`).
The first 0.5 s are ignored (the model's first frames settle). Needs: mediapipe 0.10.x, opencv, numpy.
"""
import json
import sys

import cv2
import numpy as np

LEFT = (33, 160, 158, 133, 153, 144)
RIGHT = (362, 385, 387, 263, 373, 380)
MAX_BLINK_S = 0.6
SKIP_S = 0.5


def ear(points, ids) -> float:
    p = [np.array(points[i]) for i in ids]
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]) + 1e-9))


def analyse(path: str) -> dict:
    import mediapipe as mp
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 16.0
    values, found = [], 0
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3, min_tracking_confidence=0.3) as mesh:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks:
                h, w = frame.shape[:2]
                points = [(l.x * w, l.y * h) for l in result.multi_face_landmarks[0].landmark]
                values.append((ear(points, LEFT) + ear(points, RIGHT)) / 2)
                found += 1
            else:
                values.append(None)
    series = np.array([v if v is not None else np.nan for v in values], dtype=float)
    skip = int(SKIP_S * fps)
    usable = series[skip:]
    good = usable[~np.isnan(usable)]
    if len(good) < 8:
        return {"clip": path, "frames": len(values), "face_found": round(found / max(len(values), 1), 2), "blinks": None, "note": "face not found"}
    open_level = float(np.percentile(good, 85))
    closed = np.where(np.isnan(usable), False, usable < 0.70 * open_level)
    blinks, long_closures, longest, run = 0, 0, 0, 0
    for flag in list(closed) + [False]:
        if flag:
            run += 1
            continue
        if run:
            seconds = run / fps
            longest = max(longest, seconds)
            if seconds <= MAX_BLINK_S:
                blinks += 1
            else:
                long_closures += 1
        run = 0
    seconds_total = len(usable) / fps
    return {"clip": path, "fps": fps, "frames": len(values), "face_found": round(found / len(values), 2), "open_level": round(open_level, 3), "blinks": blinks, "long_closures": long_closures,
            "longest_closed_s": round(longest, 2), "blinks_per_10s": round(10 * blinks / max(seconds_total, 1e-6), 1), "ear": [None if np.isnan(v) else round(float(v), 3) for v in series]}


if __name__ == "__main__":
    for clip in sys.argv[1:]:
        print(json.dumps(analyse(clip)), flush=True)
