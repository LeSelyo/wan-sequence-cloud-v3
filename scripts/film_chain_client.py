#!/usr/bin/env python3
"""Chained FLF2V film: each clip starts from the REAL last frame of the previous generated clip.

Reuses the plumbing of film_client.py (same env vars, same API). Needs ffmpeg/ffprobe on the client.

    WAN_BASE_URL=https://host WAN_API_TOKEN=... python scripts/film_chain_client.py keyframes chain assemble \
        --film-dir results/film_artemis_portrait_nolora_chained

Stages:
    keyframes  Flux Schnell keyframes WITHOUT LoRA (only kf00 is a start frame; the others are end frames)
    chain      one single-shot job per clip; start_image = uploaded last frame of the previous clip
    assemble   cut the clips together with ffmpeg, dropping the duplicated first frame of clips 2..N
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import film_client as fc  # noqa: E402

STAGES = ["keyframes", "chain", "assemble"]


def stem(clip: dict) -> str:
    return f"{clip['n']:02d}_{clip['id'].split('_', 1)[1]}"


def stage_keyframes(film: dict, state: dict) -> None:
    fmt, flux = film["format"], film["flux"]
    folder = fc.FILM_DIR / "keyframes"
    for k in film["keyframes"]:
        file, key = folder / f"{k['id']}.png", str(k["n"])
        if file.exists() and key in state["image_ids"]:
            continue
        image_id, seconds = fc._generate_image(k["prompt"], k["seed"], fmt["width"], fmt["height"], [], flux["guidance"], flux["steps"])
        fc._download(f"/v1/images/{image_id}/output", file)
        state["image_ids"][key] = image_id
        fc.save_state(state)
        fc.log("keyframe", n=k["n"], id=k["id"], image_id=image_id, seconds=seconds, lora=None)


def extract_last_frame(video: Path, frames: int, target: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", f"select=eq(n\\,{frames - 1})",
         "-vsync", "0", "-frames:v", "1", str(target)],
        check=True,
    )
    if not target.exists() or target.stat().st_size == 0:
        raise SystemExit(f"could not extract the last frame of {video}")


def upload_png(path: Path) -> str:
    response = fc.request("POST", "/v1/images/upload", files={"file": (path.name, path.read_bytes(), "image/png")}, timeout=120)
    fc.expect(f"upload {path.name}", response.status_code == 201, f"{response.status_code} {response.text[:200]}")
    return response.json()["image_id"]


def stage_chain(film: dict, state: dict) -> None:
    chain = state.setdefault("chain", {})
    previous = state["image_ids"]["0"]  # kf00 = start frame of clip 1 only
    for clip in film["clips"]:
        n, name = str(clip["n"]), stem(clip)
        video = fc.FILM_DIR / "clips" / f"{name}.mp4"
        if n in chain and video.exists():
            previous = chain[n]["last_frame_image_id"]
            continue
        shot = fc._shot(film, clip, state)
        shot["start_image"] = {"image_id": previous}
        job = {"id": fc._job_id(f"chain{clip['n']:02d}"), "shots": [shot], "transition": {"type": "cut", "duration_seconds": 0.0}}
        started = time.time()
        fc._run_job(job, video, 2400)
        last = fc.FILM_DIR / "lastframes" / f"{clip['n']:02d}_last.png"
        last.parent.mkdir(exist_ok=True)
        extract_last_frame(video, clip["frames"], last)
        previous = upload_png(last)
        chain[n] = {"job_id": job["id"], "seconds": round(time.time() - started, 1), "last_frame_image_id": previous}
        fc.save_state(state)
        fc.log("chain_clip_done", n=clip["n"], name=name, seconds=chain[n]["seconds"])


def stage_assemble(film: dict, state: dict) -> None:
    clips = [fc.FILM_DIR / "clips" / f"{stem(c)}.mp4" for c in film["clips"]]
    missing = [str(p) for p in clips if not p.exists()]
    if missing:
        raise SystemExit(f"missing clips: {missing}")
    parts = []
    for index in range(len(clips)):
        trim = "trim=start_frame=1," if index else ""
        parts.append(f"[{index}:v]{trim}setpts=PTS-STARTPTS[v{index}]")
    joined = "".join(f"[v{index}]" for index in range(len(clips)))
    graph = ";".join(parts) + f";{joined}concat=n={len(clips)}:v=1:a=0[out]"
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for path in clips:
        command += ["-i", str(path)]
    output = fc.FILM_DIR / "film_sequence_chained.mp4"
    command += ["-filter_complex", graph, "-map", "[out]", "-r", str(film["format"]["fps"]),
                "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(output)]
    subprocess.run(command, check=True)
    fc.log("assembled", file=str(output), bytes=output.stat().st_size)
    fc._probe(output, film["total_seconds"] - (len(clips) - 1) / film["format"]["fps"])


DISPATCH = {"keyframes": stage_keyframes, "chain": stage_chain, "assemble": stage_assemble}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=STAGES + ["all"])
    parser.add_argument("--film-dir", required=True)
    args = parser.parse_args()
    fc.FILM_DIR = Path(args.film_dir)
    film, state = fc.load_film(), fc.load_state()
    for stage in (STAGES if "all" in args.stages else args.stages):
        fc.log("stage_start", stage=stage, base=fc.BASE)
        DISPATCH[stage](film, state)
        state = fc.load_state()
        fc.log("stage_end", stage=stage)


if __name__ == "__main__":
    main()
