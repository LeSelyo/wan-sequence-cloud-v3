"""The PICTURE of every shot that is not a close-up of a character: one Krea2 still per shot (its own prompt `still`, the SAME style prompt as the cards of the video), which the S2V model then brings to life.
Talking shots and the choice card start from the character portraits instead (cards.json).

    python scripts/story_stills.py PLAN.json --cards runs/test20_en_night --out OUT_DIR [--only s003,s004] [--seed-shift 0]

Needs the app on the box started for the Krea2 family (reachable on 127.0.0.1:8000, WAN_API_TOKEN in the environment). Seeds, prompts and files go to OUT_DIR/stills.json and the register.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_style as ss  # noqa: E402
import trend_rain_anime as tr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "story_trend"
REGISTRY = RESULTS / "generations.json"
SIZE = (576, 1024)


STILL_PROMPT_VERSION = "2026-10-09.1"  # 2026-10-09.1: the place of the shot is WRITTEN in the prompt; a shot's own hour no longer turns the light to "warm natural light" unless it is a daylight hour


def shot_prompt(shot: dict, plan: dict, style: dict) -> str:
    """The picture prompt: what the shot shows + WHERE it is (the description of its location, written by the world: "the freezing metal walkways leading to the airlock", so the picture cannot drift to a
    generic sunlit arcade) + the art direction of the video. A shot with its own `time` (a dawn at the end of a night story) changes the hour of the style only when that hour is a daylight one."""
    location = next((l for l in plan["locations"] if l["id"] == shot["location"]), {})
    style = ss.shot_hour_style(style, shot.get("time"), ((plan.get("brief") or {}).get("world") or {}).get("hour"))
    where = (location.get("description") or "").strip().rstrip(".")
    place = f", set in {where[:1].lower() + where[1:]}" if where else ""
    return f"{shot['still'].strip().rstrip('. ')}{place}, vertical composition, {ss.style_prompt(style, location.get('tags', []))}"


def make_stills(plan: dict, cards: dict, out: Path, only: set[str] | None = None, seed_shift: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    book_path = out / "stills.json"
    book = json.loads(book_path.read_text(encoding="utf-8")) if book_path.exists() else {}
    rnd = random.Random(plan["params"]["seed"] * 977 + 13)
    seeds = {shot["id"]: rnd.randrange(2**31) for shot in plan["shots"]}  # one seed per shot, always drawn in the same order so a re-run gives the same picture
    for shot in plan["shots"]:
        if not shot.get("still") or (only and shot["id"] not in only):
            continue
        seed = seeds[shot["id"]] + seed_shift
        prompt = shot_prompt(shot, plan, cards["style"])
        path = out / f"{shot['id']}.png"
        if path.exists():  # never overwrite: the earlier picture is kept next to the new one
            path.replace(out / f"{shot['id']}_v{len(book.get(shot['id'], {}).get('earlier', [])) + 1}_seed{book.get(shot['id'], {}).get('seed', 0)}.png")
        result = tr.run_still({"engine": "krea2", "prompt": prompt, "width": SIZE[0], "height": SIZE[1], "seed": seed}, path, log=RESULTS / "logs" / "jobs.jsonl", registry=REGISTRY,
                              label=f"{plan.get('title', 'story')}_{shot['id']}")
        earlier = book.get(shot["id"], {}).get("earlier", []) + ([{"seed": book[shot["id"]]["seed"]}] if shot["id"] in book else [])
        book[shot["id"]] = {"file": str(path.resolve().relative_to(ROOT)).replace("\\", "/"), "seed": seed, "prompt": prompt, "seconds": result["seconds"], "earlier": earlier, "prompt_version": STILL_PROMPT_VERSION}
        book_path.write_text(json.dumps(book, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{shot['id']}: {result['seconds']} s seed {seed}", flush=True)
    return book


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", help="comma-separated shot ids")
    parser.add_argument("--seed-shift", type=int, default=0, help="added to every seed (another picture for the same shot)")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    cards = json.loads((args.cards / "cards.json").read_text(encoding="utf-8"))
    book = make_stills(plan, cards, args.out, set(args.only.split(",")) if args.only else None, args.seed_shift)
    print(f"{len(book)} stills in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
