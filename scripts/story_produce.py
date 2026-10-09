"""Produces a story video from its plan and cards, step by step, writing the measured time of every step in <run>/produce.json.

    python scripts/story_produce.py PLAN.json --cards runs/test20_en_night --out story.mp4 [--steps transcribe,voices,talk,render] [--seed 1]

  transcribe  the exact transcript (ref_text) of every recorded voice that has none, by Whisper on the box -> results/story_trend/voices/voices.json
  voices      one wav per shot (cloned recorded voice or a voice designed from text) on the box (Qwen3-TTS), copied back to <run>/voices/
  animate     one S2V clip per shot (scripts/s2v_talk.py through the ComfyUI tunnel): the speaker's portrait + the voice line for a talk shot, each character's portrait waiting for the choice, the
              shot's own still (scripts/story_stills.py) + silence for every other shot; the clip is as long as the shot
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
import i2v_action as i2v  # noqa: E402
import method_registry as mr  # noqa: E402
import s2v_talk as st  # noqa: E402
import story_identity as sid  # noqa: E402
import story_engine as se  # noqa: E402
import story_render as sr  # noqa: E402
import story_voice_job as vj  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "results" / "story_trend" / "generations.json"
VOICES_JSON = se.VOICES_FILE
STEPS = ("transcribe", "voices", "animate", "render")
OFFER_METHOD = None  # "i2v_offer_step" makes the choice moment the other way (the two step forward); None = plan A, the default the user chose
LOOK = "cinematic, photographic, moody, natural motion"  # the default; a plan with a brief gets the atmosphere and the hour of ITS world (see look_of)


def look_of(plan: dict) -> str:
    """The words added to every animation prompt: the atmosphere of the world of THIS story (a lunar dome has no rain)."""
    world = (plan.get("brief") or {}).get("world")
    if not world:
        return LOOK
    return f"cinematic, photographic, {world['atmosphere']}, {world['hour']}, natural motion"


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
    box.put(ROOT / "scripts" / "box" / "story_voices.py", "/root/story_voices.py")  # always the current version of the box script (an old copy once lacked `align`)
    box.run(f"/root/ttsenv/bin/python /root/story_voices.py speak /root/{run.name}_voice_job.json 2>&1 | grep -v Warning | tail -n 20")
    box.get(f"{remote_out}/manifest.json", run / "voices" / "manifest.json")
    for line in job["lines"]:
        box.get(f"{remote_out}/{line['id']}.wav", run / "voices" / f"{line['id']}.wav")
    spoken = time.time() - started
    box.run(f"/root/ttsenv/bin/python /root/story_voices.py align {remote_out} --language {plan['params']['language']} 2>&1 | grep -v Warning | tail -n 3")  # the real time of every word, for the subtitles
    box.get(f"{remote_out}/align.json", run / "voices" / "align.json")
    return {"seconds": round(time.time() - started, 1), "speak_seconds": round(spoken, 1), "align_seconds": round(time.time() - started - spoken, 1), "lines": {l["id"]: l["voice"]["kind"] for l in job["lines"]}}


def padded_wav(source: Path | None, seconds: float, out: Path) -> Path:
    """16 kHz mono wav of EXACTLY `seconds`: the voice line followed by silence (or only silence): what S2V is given for a shot."""
    import wave

    import numpy as np
    samples = np.zeros(int(round(seconds * 16000)), dtype=np.int16)
    if source is not None:
        with wave.open(str(source), "rb") as handle:
            data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float64)
            if handle.getnchannels() == 2:
                data = data[::2]
            rate = handle.getframerate()
        if rate != 16000:
            data = np.interp(np.linspace(0, len(data) - 1, int(len(data) * 16000 / rate)), np.arange(len(data)), data)
        samples[:min(len(samples), len(data))] = data[:len(samples)].astype(np.int16)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.tobytes())
    return out


def portrait_of(cards: dict, character_id: str) -> str:
    """The picture a character speaks from: the tight close-up when one was picked (more pixels on the face), else the first portrait."""
    entry = cards["characters"][character_id]
    return (entry.get("closeup") or entry["portrait"])["file"]


def last_frame(clip: Path, dst: Path) -> Path:
    """The last picture of a clip (the start of the next one: two clips of one scene follow each other without a jump)."""
    import music_timing as mt
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-sseof", "-0.2", "-i", str(clip), "-update", "1", "-frames:v", "1", str(dst)], check=True)
    return dst


def animate_jobs(plan: dict, cards: dict, run: Path, registry: "mr.Registry | None" = None) -> list[dict]:
    """Every clip to make. The METHOD of each shot comes from the registry (scripts/method_registry.py: what the lab and the user proved works for that kind of shot):
    S2V only when a character SPEAKS or shouts (the portrait + the voice line); everything else is I2V (people who walk, hands that act, the two who hold out a hand at the choice, a waiting face)
    or, for places without people, the S2V that is proven for them. Fields: id, engine (s2v | i2v), source picture, voice, seconds, prompt, profile and style for I2V, chain_from (the clip whose last picture starts this one)."""
    registry = registry or mr.Registry()
    characters = {c["id"]: c for c in plan["characters"]}
    jobs = []
    previous = None
    for index, shot in enumerate(plan["shots"]):
        voice = run / "voices" / f"{shot['id']}.wav"
        seconds = sr.shot_seconds(shot["kind"], sr.wav_seconds(voice), last=index == len(plan["shots"]) - 1)
        look = f"{shot['visual']}, {look_of(plan)}"
        route = mr.route(shot, plan, registry, OFFER_METHOD if shot["kind"] in ("offer", "choice") else None)
        people = [characters[i] for i in shot.get("in_shot", []) if i in characters]
        if shot["kind"] == "talk":
            who = characters[shot["speaker"]]
            portrait = ROOT / portrait_of(cards, shot["speaker"])
            jobs.append({"id": shot["id"], "engine": "s2v", "source": portrait, "voice": voice, "seconds": seconds,
                         "prompt": f"{who['name']}, a {who['age']}-year-old {'woman' if who['gender'] == 'f' else 'man'} {who['role']}, {look}"})
        elif shot["kind"] == "choice" and (run / "stills_id" / f"{sid.TWO_SHOT_ID}.png").exists():
            jobs.append({"id": shot["id"], "engine": "i2v", "source": sid.source_of(run, shot), "voice": None, "seconds": seconds, "profile": route["settings"]["profile"], "style": route["settings"]["style"],
                         "prompt": i2v.build_prompt(route["settings"]["style"], characters=[characters[i] for i in sr.choice_ids(plan, shot)]), "need": route["need"], "method": route["method"]})
        elif shot["kind"] == "choice":  # no two-shot yet (the identity pass has not run): each character waits in his or her own clip
            waiting = registry.choose("person_idle_closeup")
            for who_id in sr.choice_ids(plan, shot):
                who = characters[who_id]
                jobs.append({"id": f"{shot['id']}_{who_id}", "engine": "i2v", "source": ROOT / portrait_of(cards, who_id), "voice": None, "seconds": seconds, "profile": waiting["settings"]["profile"],
                             "style": waiting["settings"]["style"], "prompt": i2v.build_prompt(waiting["settings"]["style"], characters=[who]), "need": "person_idle_closeup", "method": waiting["method"]})
        elif route["engine"] == "i2v":
            style = route["settings"]["style"]
            job = {"id": shot["id"], "engine": "i2v", "source": sid.source_of(run, shot), "voice": None, "seconds": seconds, "profile": route["settings"]["profile"], "style": style, "need": route["need"],
                   "method": route["method"]}
            if shot["kind"] == "offer" and previous and previous["kind"] == "offer":  # the second proposal goes on from the last picture of the first one
                job.update(style="offer_next", chain_from=previous["id"])
            job["prompt"] = i2v.build_prompt(job["style"], shot.get("motion") or shot.get("visual", ""), people if shot["kind"] != "offer" else [characters[i] for i in ("c1", "c2") if i in characters])
            jobs.append(job)
        else:
            jobs.append({"id": shot["id"], "engine": "s2v", "source": sid.source_of(run, shot), "voice": None, "seconds": seconds, "prompt": look, "need": route["need"], "method": route["method"]})
        previous = shot
    return jobs


def step_animate(plan: dict, cards: dict, run: Path, comfy_url: str, lora: str, seed: int, only: set[str] | None = None, force: bool = False, skip: set[str] | None = None) -> dict:
    """One clip per job (see animate_jobs), into <run>/clips/<id>.mp4, by the engine of the job; a clip that exists is kept (re-run with --force or --only to make it again: the old one is kept as _vN)."""
    timings = {}
    for job in animate_jobs(plan, cards, run):
        if only and job["id"] not in only and job["id"].split("_")[0] not in only:
            continue
        if skip and (job["id"] in skip or job["id"].split("_")[0] in skip):
            continue
        target = run / "clips" / f"{job['id']}.mp4"
        if target.exists() and not (force or only):
            continue
        if target.exists():
            target.replace(target.with_name(f"{target.stem}_v{len(list(target.parent.glob(target.stem + '_v*.mp4'))) + 1}.mp4"))
        if job["engine"] == "i2v":
            start = job["source"]
            if job.get("chain_from") and (run / "clips" / f"{job['chain_from']}.mp4").exists():
                start = last_frame(run / "clips" / f"{job['chain_from']}.mp4", run / "stills_id" / f"{job['id']}_start.png")
            entry = i2v.run_i2v(start, target, prompt=job["prompt"], seed=seed, profile=job["profile"], seconds=min(job["seconds"], i2v.MAX_SECONDS), base_url=comfy_url, registry=REGISTRY,
                                label=f"{run.name}_{job['id']}")
            timings[job["id"]] = {"engine": "i2v", "method": job.get("method"), "seconds": entry["seconds"], "clip_seconds": round(min(job["seconds"], i2v.MAX_SECONDS), 2), "profile": job["profile"], "steps": entry["steps"]}
        else:
            audio = padded_wav(job["voice"], job["seconds"], run / "audio16k" / f"{job['id']}.wav")
            entry = st.run_talk(job["source"], audio, target, prompt=job["prompt"], seed=seed, lora=lora, seconds=job["seconds"], base_url=comfy_url, registry=REGISTRY, label=f"{run.name}_{job['id']}")
            timings[job["id"]] = {"engine": "s2v", "method": job.get("method", "s2v_voice"), "seconds": entry["seconds"], "clip_seconds": round(job["seconds"], 2), "lora": lora, "steps": entry["steps"]}
        print(f"{job['id']}: {job['engine'].upper()} {timings[job['id']]['seconds']} s ({job['seconds']:.1f} s clip)", flush=True)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--cards", type=Path, required=True, help="the folder of story_cards.py (cards.json inside)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", default=",".join(STEPS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lora", choices=list(st.LORAS), default="t2v_1217_low")
    parser.add_argument("--comfy-url", default=st.BASE_URL)
    parser.add_argument("--ssh-host", default=os.environ.get("BOX_SSH_HOST", "n1.de.clorecloud.net"))
    parser.add_argument("--ssh-port", type=int, default=int(os.environ.get("BOX_SSH_PORT", "1380")))
    parser.add_argument("--ssh-key", default=os.environ.get("BOX_SSH_KEY", "~/.ssh/id_ed25519_clore"))
    parser.add_argument("--no-sound", action="store_true", help="no sound background (only the voices)")
    parser.add_argument("--only", help="animate only these shot ids (comma-separated); the old clip is kept as _vN")
    parser.add_argument("--skip", help="do not animate these shot ids (comma-separated): a still that must be made again first")
    parser.add_argument("--force", action="store_true", help="animate again even if the clip exists")
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
        elif step == "animate":
            report[step] = {**report.get(step, {}), **step_animate(plan, cards, run, args.comfy_url, args.lora, args.seed, set(args.only.split(",")) if args.only else None, args.force,
                                                                set(args.skip.split(",")) if args.skip else None)}
        elif step == "render":
            report[step] = sr.render_story(plan, cards_dir, run, args.out, sound=not args.no_sound)
        else:
            raise SystemExit(f"unknown step {step!r}; steps: {', '.join(STEPS)}")
        report.setdefault("step_seconds", {})[step] = round(time.time() - started, 1)
        report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{step}] {report['step_seconds'][step]} s", flush=True)


if __name__ == "__main__":
    main()
