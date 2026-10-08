"""PICTURE JUDGE: a vision model LOOKS at every generated still and says whether it belongs to the story, so that the pipeline notices a wrong picture by itself and makes it again with another seed.
(auto_ab_1: a station at night, freezing, and its corridor came out as a sunlit stone arcade with a parapet; nothing in the pipeline looked at the pictures.)

    judge_still(ask, image, shot, plan)       -> {"answer": {...}, "major": [...], "minor": [...], "ok": bool}
    judge_stills(plan, stills_dir, ask, only) -> {shot id: judge_still(...)}
    python scripts/still_judge.py PROJECT [--stills DIR] [--shots s039,s033]       (the model on the box answers on 127.0.0.1:11434; the GPU must not be busy with another job)

MAJOR problems make the picture again (another seed): sunlight or a blue sky in a world without daylight, a place that does not belong to the world, the action of the shot not visible, a collage of
panels, an impossible or glitched geometry (a window inside a window). MINOR ones are only reported (invented letters on a sign: almost every picture has some). The model answers in a JSON schema, temperature 0, thinking off.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = {
    "type": "object",
    "properties": {
        "place_seen": {"type": "string", "maxLength": 140},
        "sunlight_or_blue_sky": {"type": "boolean"},
        "matches_world": {"type": "integer", "minimum": 1, "maximum": 5},
        "action_visible": {"type": "boolean"},
        "collage_or_split_panels": {"type": "boolean"},
        "impossible_geometry": {"type": "boolean"},
        "garbled_text": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string", "maxLength": 140}, "maxItems": 4},
    },
    "required": ["place_seen", "sunlight_or_blue_sky", "matches_world", "action_visible", "collage_or_split_panels", "impossible_geometry", "garbled_text", "problems"],
}
PROMPT = ("You check ONE picture made for a short cinematic story. Every person in it is fictional.\n"
          "THE WORLD of the story: {setting}. Atmosphere: {atmosphere}. Hour: {hour}.\n"
          "THE PLACE of this shot: {place}\n"
          "THE SHOT should show: {still}\n\n"
          "Look at the picture and answer strictly and honestly:\n"
          "- place_seen: in at most 12 words, what place the picture really shows.\n"
          "- sunlight_or_blue_sky: true if the light looks like the sun or daylight (sunbeams, bright warm patches on the floor, a blue daytime sky). A window that shows space, stars or a dark night sky is false.\n"
          "- matches_world: 1 to 5. 5 = clearly this world and this place; 1 = another world (for example an old stone arcade, a village or a garden for a space station).\n"
          "- action_visible: true if what the shot should show (the action, the object, the people) can be seen.\n"
          "- collage_or_split_panels: true if the picture is made of several panels or a collage.\n"
          "- impossible_geometry: true if something is physically impossible or glitched: a window inside a window or a second room stacked above the first, duplicated or melted objects, merged bodies, extra limbs or fingers, an emblem repeated everywhere.\n"
          "- garbled_text: true if a sign or a label has unreadable or invented letters.\n"
          "- problems: a short list of what is wrong with the picture for THIS story (empty if nothing).")
DARK_HOURS = {"night", "midnight", "nightfall", "dusk", "evening", "twilight"}


def world_of(plan: dict) -> dict:
    world = (plan.get("brief") or {}).get("world") or {}
    return {"setting": world.get("setting") or plan.get("context", "")[:200], "atmosphere": world.get("atmosphere", "dark and tense"), "hour": world.get("hour", "night")}


def daylight_forbidden(plan: dict, shot: dict) -> bool:
    """A night world has no sun, except in a shot that tells its own daylight hour (the dawn of a good ending)."""
    import story_style as ss
    hour = world_of(plan)["hour"]
    if ss.changes_the_hour(shot.get("time"), hour) and shot.get("time"):
        return not set(re.findall(r"[a-z]+", shot["time"].lower())) & ss.DAYLIGHT_WORDS
    return bool(set(re.findall(r"[a-z]+", hour.lower())) & DARK_HOURS)


def verdict(answer: dict, no_daylight: bool) -> tuple[list[str], list[str]]:
    major, minor = [], []
    if no_daylight and answer["sunlight_or_blue_sky"]:
        major.append("sunlight or a blue sky in a world without daylight")
    if answer["matches_world"] <= 2:
        major.append(f"the place does not belong to the world ({answer['place_seen']})")
    if not answer["action_visible"]:
        major.append("the action of the shot is not visible")
    if answer["collage_or_split_panels"]:
        major.append("a collage of panels")
    if answer.get("impossible_geometry"):
        major.append("something in the picture is physically impossible or glitched")
    if answer["garbled_text"]:
        minor.append("a sign has invented letters")
    minor += [p for p in answer.get("problems", []) if p not in minor]
    return major, minor


def judge_still(ask, image: Path, shot: dict, plan: dict, seed: int = 0) -> dict:
    """`ask(prompt, schema, image_path, seed) -> dict` is the vision model (see `ollama_vision`)."""
    location = next((l for l in plan["locations"] if l["id"] == shot["location"]), {})
    prompt = PROMPT.format(**world_of(plan), place=location.get("description") or shot["location"], still=shot["still"])
    answer = ask(prompt, SCHEMA, image, seed)
    major, minor = verdict(answer, daylight_forbidden(plan, shot))
    return {"answer": answer, "major": major, "minor": minor, "ok": not major}


def judge_stills(plan: dict, stills_dir: Path, ask, only: set[str] | None = None) -> dict[str, dict]:
    results = {}
    for shot in plan["shots"]:
        image = stills_dir / f"{shot['id']}.png"
        if not shot.get("still") or not image.exists() or (only and shot["id"] not in only):
            continue
        started = time.time()
        results[shot["id"]] = {**judge_still(ask, image, shot, plan), "seconds": round(time.time() - started, 1)}
        print(f"{shot['id']}: {'ok' if results[shot['id']]['ok'] else 'WRONG ' + '; '.join(results[shot['id']]['major'])}", flush=True)
    return results


def ollama_vision(model: str = "qwen3.6:27b", host: str = "http://127.0.0.1:11434"):
    """ask(prompt, schema, image, seed) through Ollama (images in base64), structured output, thinking off, temperature 0; `.unload()` frees the GPU for the picture jobs."""
    import httpx

    def ask(prompt: str, schema: dict, image: Path, seed: int = 0) -> dict:
        encoded = base64.b64encode(Path(image).read_bytes()).decode()
        response = httpx.post(host.rstrip("/") + "/api/chat", json={"model": model, "stream": False, "think": False, "format": schema, "keep_alive": "10m",
                                                                  "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
                                                                  "options": {"num_ctx": 8192, "temperature": 0, "seed": seed, "num_predict": 600}}, timeout=1800)
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])

    def unload() -> None:
        try:
            httpx.post(host.rstrip("/") + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        except Exception:
            pass
    ask.unload = unload
    return ask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path)
    parser.add_argument("--stills", type=Path, help="a folder of <shot id>.png (default: PROJECT/run/stills)")
    parser.add_argument("--shots")
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.project / "plan.json").read_text(encoding="utf-8"))
    ask = ollama_vision(args.model)
    try:
        results = judge_stills(plan, args.stills or args.project / "run" / "stills", ask, set(args.shots.split(",")) if args.shots else None)
    finally:
        ask.unload()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    wrong = [i for i, r in results.items() if not r["ok"]]
    print(f"{len(results)} pictures judged, {len(wrong)} wrong: {wrong}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
