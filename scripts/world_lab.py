"""WORLD lab: does the decor of the pictures belong to the world of the story? (auto_ab_1: a station at night, freezing, and its corridor shot came out as a sunlit stone arcade with a parapet.)

Re-makes the pictures of chosen shots with the CORRECTED prompt (the places of the story decide the style, the hour of the brief, the atmosphere, the place written in the prompt, no more "warm natural
light" on night shots) with the SAME seeds as the video, so the only change is the prompt; puts BEFORE and AFTER side by side in one sheet and writes the two prompts of every shot.

    python scripts/world_lab.py results/story_trend/auto/auto_ab_1 [--shots s039,s033,...]

Needs the app on the box (Krea2 engine) reachable on 127.0.0.1:8000. The token is read over ssh into memory only. Output: results/story_trend/lab/<project>/world/ (never overwrites).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_tools as lt  # noqa: E402
import story_produce as sp  # noqa: E402
import story_stills as sst  # noqa: E402
import story_style as ss  # noqa: E402
import trend_rain_anime as tr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "results" / "story_trend" / "lab"
REGISTRY = ROOT / "results" / "story_trend" / "generations.json"
DEFAULT_SHOTS = "s039,s033,s037,s004,s019,s003,s042,s043"


def corrected_style(plan: dict, seed: int = 0) -> dict:
    world = (plan.get("brief") or {}).get("world") or {}
    tags = [tag for loc in plan["locations"] for tag in loc.get("tags", [])]
    return ss.style_from_context(plan["context"], random.Random(seed), tags=tags, hour=world.get("hour"), atmosphere=" ".join([world.get("atmosphere", ""), world.get("setting", "")]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path)
    parser.add_argument("--shots", default=DEFAULT_SHOTS)
    parser.add_argument("--style-seed", type=int, default=0)
    parser.add_argument("--ssh", default="n1.de.clorecloud.net:1380:~/.ssh/id_ed25519_clore")
    args = parser.parse_args()
    project = args.project
    plan = json.loads((project / "plan.json").read_text(encoding="utf-8"))
    cards = json.loads((project / "cards" / "cards.json").read_text(encoding="utf-8"))
    book = json.loads((project / "run" / "stills" / "stills.json").read_text(encoding="utf-8"))
    host, port, key = args.ssh.split(":")
    tr.API_TOKEN = sp.Box(host, int(port), key).run("cat /root/.api_token").strip()  # in memory only
    out = LAB / project.name / "world"
    out.mkdir(parents=True, exist_ok=True)
    after_style = corrected_style(plan, args.style_seed)
    (out / "style_before.json").write_text(json.dumps(cards["style"], indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "style_after.json").write_text(json.dumps(after_style, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["# World lab: BEFORE (prompt of the video) and AFTER (corrected prompt), same seeds", "",
             "| | before | after |", "|---|---|---|",
             f"| lighting | {cards['style']['lighting']} | {after_style['lighting']} |", f"| grade | {cards['style']['grade']} | {after_style['grade']} |",
             f"| palette | {', '.join(cards['style']['palette'])} | {', '.join(after_style['palette'])} |", f"| materials | {', '.join(cards['style']['materials'])} | {', '.join(after_style['materials'])} |",
             f"| kinds of place | {', '.join(cards['style']['tags'])} | {', '.join(after_style['tags'])} |", ""]
    columns = []
    for shot_id in args.shots.split(","):
        shot = next(s for s in plan["shots"] if s["id"] == shot_id)
        before = ROOT / book[shot_id]["file"]
        prompt = sst.shot_prompt(shot, plan, after_style)
        dst = out / f"{shot_id}_after_seed{book[shot_id]['seed']}.png"
        if not dst.exists():
            started = time.time()
            tr.run_still({"engine": "krea2", "prompt": prompt, "width": sst.SIZE[0], "height": sst.SIZE[1], "seed": book[shot_id]["seed"]}, dst, log=ROOT / "results" / "story_trend" / "logs" / "jobs.jsonl",
                         registry=REGISTRY, label=f"world_lab_{project.name}_{shot_id}")
            print(f"{shot_id}: {time.time() - started:.0f} s", flush=True)
        location = next((l for l in plan["locations"] if l["id"] == shot["location"]), {})
        columns.append({"label": f"{shot_id} {shot['location']}", "before": before, "after": dst, "before_note": f"light: {cards['style']['lighting']}", "after_note": f"light: {after_style['lighting']}"})
        lines += [f"## {shot_id} ({shot['location']}: {location.get('description', '')})", f"- seed {book[shot_id]['seed']}", f"- BEFORE: {book[shot_id]['prompt']}", f"- AFTER: {prompt}", ""]
    sheet = lt.before_after_sheet(columns, out / f"world_before_after_{len(columns)}shots.png", "WORLD - same shot, same seed: before | after")
    (out / "COMMENTS_world.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"sheet: {sheet}")


if __name__ == "__main__":
    main()
