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


BREAK_BEFORE = {"choice", "rewind"}  # a new passage starts here: the narrator draws breath for the choice and for the rewind
BREAK_AFTER = {"twist"}  # a twist closes its passage
MAX_PASSAGE_WORDS = 30  # about 12 s of speech: long enough for one pace and one intonation, short enough not to drift
MAX_PASSAGE_LINES = 6


def group_lines(plan: dict, lines: list[dict]) -> list[dict]:
    """The lines that are SPOKEN AS ONE PASSAGE: consecutive lines of the same voice and the same branch, cut before a choice or a rewind and after a twist, at most MAX_PASSAGE_WORDS words. Lines of 3 to 5 words
    generated alone had a pace between 1.3 and 4.9 words per second and levels 10 dB apart; read together they keep one pace and one intonation (box side: the passage is cut back into its lines)."""
    groups: list[dict] = []
    current: dict | None = None
    for shot, line in zip(plan["shots"], lines):
        key = json.dumps(line["voice"], sort_keys=True)
        words = len(line["text"].split())
        joins = (current is not None and current["key"] == key and shot.get("branch") == current["branch"] and shot.get("kind") not in BREAK_BEFORE and not current["closed"]
                 and current["words"] + words <= MAX_PASSAGE_WORDS and len(current["line_ids"]) < MAX_PASSAGE_LINES)
        if not joins:
            current = {"key": key, "branch": shot.get("branch"), "line_ids": [], "words": 0, "closed": False, "voice": line["voice"]}
            groups.append(current)
        current["line_ids"].append(line["id"])
        current["words"] += words
        if shot.get("kind") in BREAK_AFTER:
            current["closed"] = True
    return [{"id": f"g{n + 1:02d}", "line_ids": g["line_ids"], "voice": g["voice"]} for n, g in enumerate(groups)]


def build_job(plan: dict, voices: list[dict], samples_dir: str, out_dir: str, seed: int = 1, passages: bool = True) -> dict:
    characters = {c["id"]: c for c in plan["characters"]}
    lines = []
    for index, shot in enumerate(plan["shots"]):
        speaker = shot["speaker"]
        fallback = se.voice_description(characters[speaker]) if speaker in characters else "a calm captivating narrator, natural realistic voice"
        lines.append({"id": shot["id"], "text": shot["text"], "voice": voice_spec(plan["voices"][speaker], voices, samples_dir, fallback), "seed": seed * 1000 + index})
    job = {"out_dir": out_dir, "language": LANGUAGES.get(plan["params"]["language"], "English"), "lines": lines}
    groups = group_lines(plan, lines) if passages else []
    if any(len(g["line_ids"]) > 1 for g in groups):  # nothing to read together: the lines stay alone
        job["groups"] = [{**g, "seed": seed * 1000 + 500 + n} for n, g in enumerate(groups)]
    return job


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
