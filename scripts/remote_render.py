"""The MONTAGE on the box, not on the PC: the PC only sends the small files (plan, voices, the clips the box does not have yet) and receives the finished mp4.

    python scripts/remote_render.py PLAN.json --cards CARDS_DIR --run RUN_DIR --out OUT.mp4 --name JOB

Why: the render is Pillow + ffmpeg (about 5 minutes of one CPU core at 720x1280 for 2 minutes of video). The box has 32 cores and is rented anyway, the PC stays free. The box keeps the clips of the
face-detail pass in /root/jobs/<name>/run/post, so after the first render only what changed travels again. The same `story_render.py` runs on both sides (same font: assets/fonts/Anton-Regular.ttf, open licence).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from story_produce import Box  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/story"
FFMPEG_ON_BOX = "/opt/venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
PYTHON_ON_BOX = "/opt/venv/bin/python"


def remote_listing(box: Box, remote_dir: str) -> dict[str, int]:
    """file name -> size of what the box already has in a folder."""
    out = box.run(f"mkdir -p {remote_dir} && cd {remote_dir} && find . -maxdepth 1 -type f -printf '%f %s\\n'")
    return {line.split()[0]: int(line.split()[1]) for line in out.splitlines() if line.strip()}


def missing_files(local_dir: Path, remote: dict[str, int], pattern: str = "*") -> list[Path]:
    """The local files of a folder that the box does not have (or has with another size)."""
    return [p for p in sorted(local_dir.glob(pattern)) if p.is_file() and remote.get(p.name) != p.stat().st_size] if local_dir.exists() else []


def render_command(job: str, plan_name: str = "plan.json") -> str:
    return (f"cd {REMOTE_ROOT} && FFMPEG_BIN={FFMPEG_ON_BOX} {PYTHON_ON_BOX} scripts/story_render.py jobs/{job}/{plan_name} --cards jobs/{job}/cards --run jobs/{job}/run --out jobs/{job}/out.mp4")


def pack(paths: list[tuple[Path, str]], archive: Path) -> Path:
    with tarfile.open(archive, "w") as tar:
        for path, arcname in paths:
            tar.add(path, arcname=arcname)
    return archive


def render_on_box(plan_path: Path, cards_dir: Path, run: Path, out: Path, box: Box, job: str, poll: float = 5.0, timeout: float = 3600.0) -> dict:
    started = time.time()
    base = f"{REMOTE_ROOT}/jobs/{job}"
    box.run(f"mkdir -p {REMOTE_ROOT}/scripts {REMOTE_ROOT}/assets/fonts {base}/cards {base}/run/voices {base}/run/post {base}/run/clips {base}/run/stills")
    # 1. the code and the font (small), the plan, the cards.json, the voices
    small = [(p, f"scripts/{p.name}") for p in sorted((ROOT / "scripts").glob("*.py"))] + [(p, f"assets/fonts/{p.name}") for p in (ROOT / "assets" / "fonts").glob("*.ttf")]
    small += [(plan_path, f"jobs/{job}/plan.json"), (cards_dir / "cards.json", f"jobs/{job}/cards/cards.json")]
    small += [(p, f"jobs/{job}/run/voices/{p.name}") for p in sorted((run / "voices").glob("*")) if p.is_file()]
    archive = pack(small, out.parent / f"{job}_small.tar")
    box.put(archive, f"{REMOTE_ROOT}/small.tar")
    box.run(f"cd {REMOTE_ROOT} && tar -xf small.tar && rm small.tar")
    archive.unlink()
    # 2. the clips: what the box does not have yet (post-processed clips stay on the box from one render to the next); raw clips only for shots without a post clip
    sent = 0
    post_remote = remote_listing(box, f"{base}/run/post")
    for clip in missing_files(run / "post", post_remote, "*.mp4"):
        box.put(clip, f"{base}/run/post/{clip.name}")
        sent += 1
    clips_remote = remote_listing(box, f"{base}/run/clips")
    raw_needed = [c for c in sorted((run / "clips").glob("*.mp4")) if "_v" not in c.stem and "_retry" not in c.stem and not (run / "post" / c.name).exists()]
    for clip in [c for c in raw_needed if clips_remote.get(c.name) != c.stat().st_size]:
        box.put(clip, f"{base}/run/clips/{clip.name}")
        sent += 1
    uploaded = time.time() - started
    # 3. the render, detached (an ssh line that lasts minutes can drop): poll a marker file
    log = f"{base}/render.log"
    box.run(f"rm -f {base}/out.mp4 {base}/render.done; cd {REMOTE_ROOT} && (setsid nohup bash -c '{render_command(job)} > {log} 2>&1; echo $? > {base}/render.done' > /dev/null 2>&1 < /dev/null &)")
    while True:
        done = box.run(f"cat {base}/render.done 2>/dev/null || true").strip()
        if done:
            break
        if time.time() - started > timeout:
            raise TimeoutError(f"the render on the box is still running after {timeout:.0f} s (log {log})")
        time.sleep(poll)
    tail = box.run(f"tail -n 40 {log}")
    if done != "0":
        raise RuntimeError(f"the render on the box failed (exit {done}):\n{tail[-2500:]}")
    rendered = time.time() - started - uploaded
    out.parent.mkdir(parents=True, exist_ok=True)
    box.get(f"{base}/out.mp4", out)
    line = next((l for l in reversed(tail.splitlines()) if l.startswith("RESULT ")), None)
    result = json.loads(line[len("RESULT "):]) if line else {}
    return {"file": str(out), "uploaded_clips": sent, "upload_seconds": round(uploaded, 1), "render_seconds": round(rendered, 1), "total_seconds": round(time.time() - started, 1), "render": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--ssh-host", default=os.environ.get("BOX_SSH_HOST", "n1.de.clorecloud.net"))
    parser.add_argument("--ssh-port", type=int, default=int(os.environ.get("BOX_SSH_PORT", "1380")))
    parser.add_argument("--ssh-key", default=os.environ.get("BOX_SSH_KEY", "~/.ssh/id_ed25519_clore"))
    args = parser.parse_args()
    result = render_on_box(args.plan, args.cards, args.run, args.out, Box(args.ssh_host, args.ssh_port, args.ssh_key), args.name)
    print(json.dumps({k: v for k, v in result.items() if k != "render"}, indent=1))


if __name__ == "__main__":
    main()
