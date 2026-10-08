"""POST-PROCESS of S2V clips on the box: ESRGAN enlargement of every frame, temporal denoise, motion-compensated interpolation 16 -> 30 fps, one h264 file per clip at the final size.
The PC only sends the clips and receives the result (scripts/box/upscale_clip.py does the work, on the GPU and the 32 CPU cores of the box).

    python scripts/clip_post.py OUT_DIR CLIP1.mp4 CLIP2.mp4 ... [--model RealESRGAN_x2.pth] [--size 738x1280] [--denoise 1.5] [--fps 30] [--no-interp] [--jobs 6]

Why: the S2V model draws at 480x832 and its eyes, eyelids and skin come out grainy; a detail pass makes the frame bigger and cleaner, the temporal denoise removes the shimmer that stays between
frames, the interpolation makes the 16 fps clips fluid. The box is reached over ssh (BOX_SSH_HOST / BOX_SSH_PORT / BOX_SSH_KEY).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from story_produce import Box  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"x2": "RealESRGAN_x2.pth", "x4plus": "RealESRGAN_x4plus.pth"}
MODEL_DIR = "/workspace/models/upscale_models"


def remote_command(run: str, names: list[str], model: str, size: tuple[int, int], denoise: float, fps: int, interpolate: bool, jobs: int, sharp: float = 0.5) -> str:
    """The command run on the box for a batch of clips already uploaded to /root/post_in/<run>."""
    files = " ".join(f"/root/post_in/{run}/{name}" for name in names)
    return (f"/opt/venv/bin/python /root/upscale_clip.py /root/post_out/{run} --model {MODEL_DIR}/{model} --size {size[0]}x{size[1]} --denoise {denoise} --fps {fps} --sharp {sharp} "
            f"--jobs {jobs}{'' if interpolate else ' --no-interp'} {files}")


def run_batch(clips: list[Path], out_dir: Path, box: Box, *, model: str = MODELS["x2"], size: tuple[int, int] = (738, 1280), denoise: float = 1.0, fps: int = 30, interpolate: bool = True,
              jobs: int = 6, sharp: float = 0.5, keep_remote: str | None = None) -> dict:
    started = time.time()
    run = uuid.uuid4().hex[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    box.run(f"mkdir -p /root/post_in/{run} /root/post_out/{run}")
    box.put(ROOT / "scripts" / "box" / "upscale_clip.py", "/root/upscale_clip.py")
    for clip in clips:
        box.put(clip, f"/root/post_in/{run}/{clip.name}")
    uploaded = time.time() - started
    log = box.run(remote_command(run, [c.name for c in clips], model, size, denoise, fps, interpolate, jobs, sharp), timeout=7200)
    worked = time.time() - started - uploaded
    for clip in clips:
        box.get(f"/root/post_out/{run}/{clip.name}", out_dir / clip.name)
    if keep_remote:  # the box keeps the face-detail clips where the montage on the box reads them: they never have to travel up again
        box.run(f"mkdir -p {keep_remote} && cp /root/post_out/{run}/*.mp4 {keep_remote}/")
    box.run(f"rm -rf /root/post_in/{run} /root/post_out/{run}")
    return {"clips": len(clips), "seconds": round(time.time() - started, 1), "upload_seconds": round(uploaded, 1), "box_seconds": round(worked, 1), "model": model, "size": list(size),
            "denoise": denoise, "fps": fps, "interpolate": interpolate, "log": log.strip().splitlines(), "kept_on_box": keep_remote}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--model", default=MODELS["x2"], choices=sorted(MODELS.values()))
    parser.add_argument("--size", default="738x1280")
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--sharp", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-interp", action="store_true")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--ssh-host", default=os.environ.get("BOX_SSH_HOST", "n1.de.clorecloud.net"))
    parser.add_argument("--ssh-port", type=int, default=int(os.environ.get("BOX_SSH_PORT", "1380")))
    parser.add_argument("--ssh-key", default=os.environ.get("BOX_SSH_KEY", "~/.ssh/id_ed25519_clore"))
    args = parser.parse_args()
    result = run_batch(args.clips, args.out_dir, Box(args.ssh_host, args.ssh_port, args.ssh_key), model=args.model, size=tuple(int(v) for v in args.size.split("x")), denoise=args.denoise,
                       fps=args.fps, interpolate=not args.no_interp, jobs=args.jobs, sharp=args.sharp)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
