"""Compare S2V settings on the FACE of one character: same portrait, same voice, same seed, one clip per setting, then numbers on the face area and a picture of the same frame side by side.

    python scripts/s2v_face_test.py PORTRAIT.png AUDIO.wav OUT_DIR --profiles i2v_low,i2v_1022_low,i2v_low_real [--size 480x832] [--seconds 3]

Numbers (face box = the central upper part of the frame, where the face of a portrait stands):
  detail   mean absolute difference between the face and its slightly blurred self: the fine detail AND the noise (higher = more of both)
  flicker  mean |frame - average of its two neighbours|: what shimmers from one frame to the next (eyelids, skin): LOWER is calmer
  calm     detail / flicker: the more fine detail per unit of shimmer, the better (a good face is detailed AND stable)
Needs ComfyUI reachable on --url (ssh tunnel local 18188 -> box 8188) started for the Wan family.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402
import s2v_talk as st  # noqa: E402

FACE_BOX = (0.26, 0.22, 0.74, 0.46)  # x0, y0, x1, y1 as fractions of the frame (default; give --face-box for a portrait whose face stands elsewhere)


def read_gray_frames(clip: Path, size: tuple[int, int]) -> np.ndarray:
    raw = subprocess.run([mt.ffmpeg_binary(), "-v", "error", "-i", str(clip), "-vf", f"scale={size[0]}:{size[1]}:flags=lanczos,format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, size[1], size[0]).astype(np.float64)


def face_metrics(clip: Path, size: tuple[int, int] = (480, 832), box: tuple[float, float, float, float] = FACE_BOX) -> dict:
    frames = read_gray_frames(clip, size)
    x0, y0, x1, y1 = (int(box[0] * size[0]), int(box[1] * size[1]), int(box[2] * size[0]), int(box[3] * size[1]))
    face = frames[:, y0:y1, x0:x1]
    blurred = np.stack([np.asarray(Image.fromarray(f.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float64) for f in face])
    detail = float(np.abs(face - blurred).mean())
    flicker = float(np.abs(face[1:-1] - (face[:-2] + face[2:]) / 2).mean())
    return {"detail": round(detail, 3), "flicker": round(flicker, 3), "calm": round(detail / max(flicker, 1e-6), 3)}


def frame_crop(clip: Path, index: int, size: tuple[int, int] = (480, 832), scale: int = 2, box: tuple[float, float, float, float] = FACE_BOX) -> Image.Image:
    raw = subprocess.run([mt.ffmpeg_binary(), "-v", "error", "-i", str(clip), "-vf", f"scale={size[0]}:{size[1]}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frame_bytes = size[0] * size[1] * 3
    index = min(index, len(raw) // frame_bytes - 1)
    frame = Image.frombytes("RGB", size, raw[index * frame_bytes:(index + 1) * frame_bytes])
    crop_box = (int(box[0] * size[0]), int(box[1] * size[1]), int(box[2] * size[0]), int(box[3] * size[1]))
    face = frame.crop(crop_box)
    return face.resize((face.width * scale, face.height * scale), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("portrait", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--profiles", default="i2v_low")
    parser.add_argument("--size", default="480x832")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--face-box", default=None, help="x0,y0,x1,y1 as fractions of the frame, around the face of THIS portrait")
    parser.add_argument("--url", default=st.BASE_URL)
    parser.add_argument("--prompt", default="a man in a dark captain coat speaks to the camera, small natural head movements, serious expression, cinematic, realistic skin, sharp eyes")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    size = tuple(int(v) for v in args.size.split("x"))
    box = tuple(float(v) for v in args.face_box.split(",")) if args.face_box else FACE_BOX
    results, crops = {}, []
    for name in args.profiles.split(","):
        target = args.out / f"{name}_{args.size}.mp4"
        started = time.time()
        if not target.exists():
            try:
                st.run_talk(args.portrait, args.audio, target, prompt=args.prompt, seed=args.seed, lora=name, seconds=args.seconds, base_url=args.url, size=size)
            except Exception as error:  # one setting that fails (out of memory, a missing LoRA) must not stop the comparison of the others
                print(name, "FAILED:", str(error)[:200].replace(chr(10), " "), flush=True)
                continue
        seconds = round(time.time() - started, 1)
        results[name] = {**face_metrics(target, size, box), "seconds": seconds, "file": str(target)}
        crops.append((name, frame_crop(target, 24, size, box=box)))
        print(name, results[name], flush=True)
    sheet = Image.new("RGB", (sum(c.width for _, c in crops) + 6 * (len(crops) - 1), crops[0][1].height), (30, 30, 30))
    x = 0
    for _, crop in crops:
        sheet.paste(crop, (x, 0))
        x += crop.width + 6
    sheet.save(args.out / "face_crops.png")
    (args.out / "metrics.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
