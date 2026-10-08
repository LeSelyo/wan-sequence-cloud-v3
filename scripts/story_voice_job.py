"""The voice job of a story: one line per shot, each with the voice of its speaker (a recorded voice to CLONE, or a voice DESIGNED from a text), ready for scripts/box/story_voices.py.

    python scripts/story_voice_job.py PLAN.json --out job.json [--samples-dir /root/voices] [--out-dir /root/voices_out/test20_en]

A recorded voice needs the exact transcript of its sample (ref_text in voices.json: `story_voices.py transcribe`); without it the voice falls back to a designed one, never to a wrong guess.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_engine as se  # noqa: E402

LANGUAGES = {"en": "English", "fr": "French"}


def voice_spec(voice_id: str, voices: list[dict], samples_dir: str, fallback_description: str) -> dict:
    """{"kind": "design", "instruct"} for design:..., {"kind": "clone", ...} for a recorded voice that has its transcript, a designed voice otherwise."""
    if voice_id.startswith("design:"):
        return {"kind": "design", "instruct": voice_id[len("design:"):]}
    voice = next((v for v in voices if v["id"] == voice_id), None)
    if voice and voice.get("ref_text"):
        return {"kind": "clone", "ref_audio": f"{samples_dir.rstrip('/')}/{voice['file']}", "ref_text": voice["ref_text"], "voice_id": voice_id}
    return {"kind": "design", "instruct": fallback_description, "note": f"voice '{voice_id}' has no transcript yet"}


def build_job(plan: dict, voices: list[dict], samples_dir: str, out_dir: str, seed: int = 1) -> dict:
    characters = {c["id"]: c for c in plan["characters"]}
    lines = []
    for index, shot in enumerate(plan["shots"]):
        speaker = shot["speaker"]
        fallback = se.voice_description(characters[speaker]) if speaker in characters else "a calm captivating narrator, natural realistic voice"
        lines.append({"id": shot["id"], "text": shot["text"], "voice": voice_spec(plan["voices"][speaker], voices, samples_dir, fallback), "seed": seed * 1000 + index})
    return {"out_dir": out_dir, "language": LANGUAGES.get(plan["params"]["language"], "English"), "lines": lines}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-dir", default="/root/voices")
    parser.add_argument("--out-dir", default="/root/voices_out/story")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    job = build_job(plan, se.load_voices(), args.samples_dir, args.out_dir, args.seed)
    args.out.write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(job['lines'])} lines -> {args.out}")


if __name__ == "__main__":
    main()
