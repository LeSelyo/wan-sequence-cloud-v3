"""Produces a story video from its plan and cards, step by step, writing the measured time of every step in <run>/produce.json.

    python scripts/story_produce.py PLAN.json --cards runs/test20_en_night --out story.mp4 [--steps transcribe,voices,talk,render] [--seed 1]

  transcribe  the exact transcript (ref_text) of every recorded voice that has none, by Whisper on the box -> results/story_trend/voices/voices.json
  voices      one wav per shot (cloned recorded voice or a voice designed from text) on the box (Qwen3-TTS), copied back to <run>/voices/
  talk        one S2V clip per talking shot (scripts/s2v_talk.py through the ComfyUI tunnel), the clip as long as the voice line
  render      the finished 9:16 video (scripts/story_render.py), on this PC (light: Pillow + ffmpeg)

The box is reached over ssh (BOX_SSH_HOST / BOX_SSH_PORT / BOX_SSH_KEY or the flags); the app must run for the Wan family with the ComfyUI tunnel on --comfy-url for `talk`.
Voices need the voice environment of the box (/root/ttsenv) and the sample files in /root/voices. Nothing secret is ever written.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2v_talk as st  # noqa: E402
import story_engine as se  # noqa: E402
import story_render as sr  # noqa: E402
import story_voice_job as vj  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "results" / "story_trend" / "generations.json"
VOICES_JSON = se.VOICES_FILE
STEPS = ("transcribe", "voices", "talk", "render")


class Box:
    def __init__(self, host: str, port: int, key: str):
        self.target, self.port, self.key = f"root@{host}", str(port), os.path.expanduser(key)

    def _opts(self) -> list[str]:
        return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-i", self.key]

    def run(self, command: str, timeout: float = 3600) -> str:
        done = subprocess.run(["ssh", "-p", self.port, *self._opts(), self.target, command], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        if done.returncode != 0:
            raise RuntimeError(f"box command failed ({done.returncode}): {command[:120]}\n{done.stderr[-1500:]}\n{done.stdout[-800:]}")
        return done.stdout

    def put(self, local: Path, remote: str) -> None:
        subprocess.run(["scp", "-P", self.port, *self._opts(), str(local), f"{self.target}:{remote}"], check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=600)

    def get(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-P", self.port, *self._opts(), f"{self.target}:{remote}", str(local)], check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=600)


def step_transcribe(box: Box, language: str) -> dict:
    data = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
    done = {}
    for voice in data["voices"]:
        if voice.get("ref_text"):
            continue
        started = time.time()
        text = box.run(f"/root/ttsenv/bin/python /root/story_voices.py transcribe /root/voices/{voice['file']} --model small --language {language} 2>/dev/null | tail -n 1")
        voice["ref_text"] = text.strip()
        done[voice["id"]] = {"seconds": round(time.time() - started, 1), "words": len(voice["ref_text"].split())}
        VOICES_JSON.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return done


def step_voices(box: Box, plan: dict, run: Path, seed: int) -> dict:
    started = time.time()
    remote_out = f"/root/voices_out/{run.name}"
    job = vj.build_job(plan, se.load_voices(), "/root/voices", remote_out, seed)
    local_job = run / "voice_job.json"
    local_job.write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    box.put(local_job, f"/root/{run.name}_voice_job.json")
    box.run(f"/root/ttsenv/bin/python /root/story_voices.py speak /root/{run.name}_voice_job.json 2>&1 | grep -v Warning | tail -n 20")
    box.get(f"{remote_out}/manifest.json", run / "voices" / "manifest.json")
    for line in job["lines"]:
        box.get(f"{remote_out}/{line['id']}.wav", run / "voices" / f"{line['id']}.wav")
    return {"seconds": round(time.time() - started, 1), "lines": {l["id"]: l["voice"]["kind"] for l in job["lines"]}}


def step_talk(plan: dict, cards: dict, cards_dir: Path, run: Path, comfy_url: str, lora: str, seed: int) -> dict:
    timings = {}
    characters = {c["id"]: c for c in plan["characters"]}
    for shot in plan["shots"]:
        if shot["kind"] != "talk":
            continue
        character = characters[shot["speaker"]]
        voice = run / "voices" / f"{shot['id']}.wav"
        seconds = sr.wav_seconds(voice)  # the render holds the last frame for the short tail
        portrait = ROOT / cards["characters"][shot["speaker"]]["portrait"]["file"]
        prompt = (f"{character['name']}, a {character['age']}-year-old {'woman' if character['gender'] == 'f' else 'man'} {character['role']}, speaks to the camera, {shot['visual']}, "
                  "small natural head and hand movements, natural expression, cinematic, handheld")
        entry = st.run_talk(portrait, voice, run / "talk" / f"{shot['id']}.mp4", prompt=prompt, seed=seed, lora=lora, seconds=seconds, base_url=comfy_url, registry=REGISTRY,
                            label=f"{run.name}_{shot['id']}")
        timings[shot["id"]] = {"seconds": entry["seconds"], "clip_seconds": round(seconds, 2), "lora": lora, "steps": entry["steps"]}
        print(f"{shot['id']}: S2V {entry['seconds']} s", flush=True)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--cards", type=Path, required=True, help="the folder of story_cards.py (cards.json inside)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", default=",".join(STEPS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lora", choices=list(st.LORAS), default="i2v_low")
    parser.add_argument("--comfy-url", default=st.BASE_URL)
    parser.add_argument("--ssh-host", default=os.environ.get("BOX_SSH_HOST", "n1.de.clorecloud.net"))
    parser.add_argument("--ssh-port", type=int, default=int(os.environ.get("BOX_SSH_PORT", "1380")))
    parser.add_argument("--ssh-key", default=os.environ.get("BOX_SSH_KEY", "~/.ssh/id_ed25519_clore"))
    parser.add_argument("--no-bed", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    cards_dir = args.cards.resolve()
    cards = json.loads((cards_dir / "cards.json").read_text(encoding="utf-8"))
    run = args.out.resolve().parent / f"{args.out.stem}_run"
    run.mkdir(parents=True, exist_ok=True)
    report_path = run / "produce.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    box = Box(args.ssh_host, args.ssh_port, args.ssh_key)
    steps = [s for s in args.steps.split(",") if s]
    for step in steps:
        started = time.time()
        if step == "transcribe":
            report[step] = step_transcribe(box, plan["params"]["language"])
        elif step == "voices":
            report[step] = step_voices(box, plan, run, args.seed)
        elif step == "talk":
            report[step] = step_talk(plan, cards, cards_dir, run, args.comfy_url, args.lora, args.seed)
        elif step == "render":
            report[step] = sr.render_story(plan, cards_dir, run / "voices", run / "talk", args.out, bed=not args.no_bed)
        else:
            raise SystemExit(f"unknown step {step!r}; steps: {', '.join(STEPS)}")
        report.setdefault("step_seconds", {})[step] = round(time.time() - started, 1)
        report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{step}] {report['step_seconds'][step]} s", flush=True)


if __name__ == "__main__":
    main()
