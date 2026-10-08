"""Phase 1 of a story video: the STYLE of the video, a CARD for every location (establishing picture that fixes how the place looks) and a CHARACTER SHEET + a hero portrait for every character,
all with Krea2 (no style LoRA: realistic), all with the SAME style prompt, so that every later shot is edited from these cards and keeps the look of its place and the face of its character.
Seeds, prompts and files go in cards.json (and every picture in the register results/story_trend/generations.json).

    python scripts/story_cards.py PLAN.json --out results/story_trend/runs/test20_en [--style-seed 3] [--only locations|characters]
Needs the app on the box reachable on 127.0.0.1:8000 (ssh tunnel) with WAN_API_TOKEN in the environment, started for the Krea2 family.
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
PORTRAIT = (576, 1024)  # 9:16
SHEET = (1024, 576)


def location_prompt(loc: dict, style: dict) -> str:
    return (f"{loc['description']}, empty of people, wide establishing shot, vertical composition, {ss.style_prompt(style, loc.get('tags', []))}")


def character_text(c: dict) -> str:
    return f"{c['name']}, a {c['age']}-year-old {c['gender'] == 'f' and 'woman' or 'man'}, {c['role']}, {c['look']}, wearing {c['wardrobe']}"


def sheet_prompt(c: dict, style: dict) -> str:
    return (f"character reference sheet of {character_text(c)}, three views of the SAME person side by side on a plain neutral grey studio background: front view, three-quarter view and side profile, "
            f"full body, photorealistic, sharp detailed face, natural skin, {ss.style_prompt(style)}")


def portrait_prompt(c: dict, style: dict, loc: dict | None) -> str:
    place = f", in {loc['description']}" if loc else ""
    return (f"cinematic medium close-up of {character_text(c)}{place}, looking straight at the camera, mouth slightly open as if speaking, photorealistic, sharp detailed face, natural skin, "
            f"{ss.style_prompt(style, (loc or {}).get('tags', []))}")


def build_cards(plan: dict, out: Path, style_seed: int = 0, only: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    library = ss.StyleLibrary()
    style = ss.style_from_context(plan["context"], random.Random(style_seed))
    style_id = library.add(style, origin="auto", context=plan["context"], seed=style_seed)
    cards = {"style_id": style_id, "style": style, "style_seed": style_seed, "negative_prompt_style": ss.negative_prompt(style), "locations": {}, "characters": {}, "plan": plan.get("title"),
             "note": "Krea2 has no negative prompt: the style forbids things through its wording only."}
    rnd = random.Random(plan["params"]["seed"] * 31 + 7)
    seeds = {}

    def make(label: str, prompt: str, size: tuple[int, int]) -> dict:
        seeds.setdefault(label, rnd.randrange(2**31))
        request = {"engine": "krea2", "prompt": prompt, "width": size[0], "height": size[1], "seed": seeds[label]}
        path = out / f"{label}.png"
        result = tr.run_still(request, path, log=RESULTS / "logs" / "jobs.jsonl", registry=REGISTRY, label=f"{plan.get('title', 'story')}_{label}")
        print(f"{label}: {result['seconds']} s seed {seeds[label]}", flush=True)
        return {"file": str(path.resolve().relative_to(ROOT)).replace("\\", "/"), "seed": seeds[label], "prompt": prompt, "size": list(size), "image_id": result["image_id"], "seconds": result["seconds"]}

    first_loc = plan["locations"][0]
    if only in (None, "locations"):
        for loc in plan["locations"]:
            cards["locations"][loc["id"]] = {**make(f"loc_{loc['id']}", location_prompt(loc, style), PORTRAIT), "kind": loc["kind"], "tags": loc.get("tags", []), "variant_of": loc.get("variant_of")}
    if only in (None, "characters"):
        for c in plan["characters"]:
            cards["characters"][c["id"]] = {"name": c["name"], "sheet": make(f"char_{c['id']}_sheet", sheet_prompt(c, style), SHEET),
                                            "portrait": make(f"char_{c['id']}_portrait", portrait_prompt(c, style, first_loc), PORTRAIT)}
    library.record_use(style_id, plan.get("title", "story"), [loc["id"] for loc in plan["locations"]])
    (out / "cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8")
    return cards


def redo_location(plan: dict, out: Path, loc_id: str, seed: int) -> dict:
    """Make ONE location card again with another seed (a card came out wrong): the earlier picture is kept as <name>_v<N>_<seed>.png, cards.json points to the new one and remembers the old ones."""
    cards = json.loads((out / "cards.json").read_text(encoding="utf-8"))
    loc = next(l for l in plan["locations"] if l["id"] == loc_id)
    style = cards["style"]
    path = out / f"loc_{loc_id}.png"
    old = cards["locations"][loc_id]
    kept = out / f"loc_{loc_id}_v{len(old.get('earlier', [])) + 1}_seed{old['seed']}.png"
    if path.exists():
        path.replace(kept)
    request = {"engine": "krea2", "prompt": location_prompt(loc, style), "width": PORTRAIT[0], "height": PORTRAIT[1], "seed": seed}
    result = tr.run_still(request, path, log=RESULTS / "logs" / "jobs.jsonl", registry=REGISTRY, label=f"{plan.get('title', 'story')}_loc_{loc_id}_seed{seed}")
    print(f"loc_{loc_id}: {result['seconds']} s seed {seed} (the earlier one is kept as {kept.name})", flush=True)
    earlier = [*old.get("earlier", []), {"file": str(kept.resolve().relative_to(ROOT)).replace("\\", "/"), "seed": old["seed"], "why": "replaced by another seed"}]
    cards["locations"][loc_id] = {**old, "seed": seed, "image_id": result["image_id"], "seconds": result["seconds"], "earlier": earlier}
    (out / "cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8")
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--style-seed", type=int, default=0)
    parser.add_argument("--only", choices=["locations", "characters"])
    parser.add_argument("--redo", metavar="LOCATION_ID", help="make this one location card again with --seed (the old picture is kept)")
    parser.add_argument("--seed", type=int, help="seed for --redo")
    args = parser.parse_args()
    started = time.time()
    if args.redo:
        redo_location(json.loads(args.plan.read_text(encoding="utf-8")), args.out, args.redo, args.seed if args.seed is not None else random.randrange(2**31))
        return
    cards = build_cards(json.loads(args.plan.read_text(encoding="utf-8")), args.out, args.style_seed, args.only)
    print(f"style {cards['style_id']}; {len(cards['locations'])} locations, {len(cards['characters'])} characters in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
