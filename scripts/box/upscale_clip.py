"""Runs ON THE BOX (system python of the app image: torch + spandrel + ffmpeg): the face-detail PASS of S2V clips.

    /opt/venv/bin/python upscale_clip.py OUT_DIR --model /workspace/models/upscale_models/RealESRGAN_x2.pth --size 738x1280 --denoise 1.5 --fps 30 [--no-interp] [--jobs 6] CLIP1.mp4 CLIP2.mp4 ...

For every clip: decode the 16 fps frames, enlarge each one with an ESRGAN model on the GPU (the GPU is used by one frame at a time, whatever the number of jobs), bring it to the final size,
and let ffmpeg (CPU, one process per clip, `--jobs` at a time) remove the shimmer left between frames (hqdn3d, strong in time), interpolate 16 -> 30 fps (motion-compensated), and sharpen the same
way on every frame. It does not use ComfyUI (a ComfyUI started with --novram keeps the upscaler on the CPU). Prints one line per clip with its seconds.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GPU_LOCK = threading.Lock()


def probe(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    width, height = out.split(",")[:2]
    return int(width), int(height)


def filter_chain(denoise: float, fps: int, interpolate: bool, sharp: float = 0.5, source_fps: int = 16) -> str:
    parts = []
    if denoise > 0:
        parts.append(f"hqdn3d={denoise * 0.6:.2f}:{denoise * 0.6:.2f}:{denoise * 3:.2f}:{denoise * 3:.2f}")
    if fps != source_fps:
        parts.append(f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" if interpolate else f"fps={fps}")
    parts.append(f"unsharp=5:5:{sharp}:5:5:0.0")
    return ",".join(parts)


def load_model(path: str):
    import torch
    from spandrel import ModelLoader
    loaded = ModelLoader().load_from_file(path)
    model = loaded.model.eval().cuda().half()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, int(loaded.scale), torch


def enlarge(model, torch, frame_bytes: bytes, width: int, height: int, size: tuple[int, int], tile: int = 0) -> bytes:
    """One RGB frame -> the model -> resized (bicubic, antialiased) to the final size, as RGB bytes."""
    import numpy as np
    import torch.nn.functional as F
    array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
    tensor = torch.from_numpy(array.copy()).cuda().half().permute(2, 0, 1).unsqueeze(0) / 255.0
    with GPU_LOCK, torch.no_grad():
        result = model(tensor)
        result = F.interpolate(result.float(), size=(size[1], size[0]), mode="bicubic", antialias=True, align_corners=False)
        out = (result.clamp(0, 1) * 255.0 + 0.5).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    return out.tobytes()


def process(clip: Path, out_dir: Path, model_tuple, size: tuple[int, int], denoise: float, fps: int, interpolate: bool, sharp: float = 0.5) -> str:
    model, scale, torch = model_tuple
    started = time.time()
    width, height = probe(clip)
    frame_size = width * height * 3
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(clip), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    target = out_dir / clip.name
    encoder = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", "16", "-i", "-", "-vf", filter_chain(denoise, fps, interpolate, sharp),
                                "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", str(target)], stdin=subprocess.PIPE)
    count = 0
    while True:
        chunk = decoder.stdout.read(frame_size)
        if len(chunk) < frame_size:
            break
        encoder.stdin.write(enlarge(model, torch, chunk, width, height, size))
        count += 1
    decoder.stdout.close()
    decoder.wait()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {clip.name}")
    return f"{clip.name}: {count} frames x{scale} -> {size[0]}x{size[1]} in {time.time() - started:.1f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--model", default="/workspace/models/upscale_models/RealESRGAN_x2.pth")
    parser.add_argument("--size", default="738x1280")
    parser.add_argument("--denoise", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-interp", action="store_true")
    parser.add_argument("--sharp", type=float, default=0.5, help="strength of the constant sharpening (unsharp luma amount)")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    size = tuple(int(v) for v in args.size.split("x"))
    model_tuple = load_model(args.model)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(lambda c: process(c, args.out_dir, model_tuple, size, args.denoise, args.fps, not args.no_interp, args.sharp), args.clips):
            print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
