"""The IDENTITY pass: the SAME person in every shot. A picture made from a text description gives a stranger; here Qwen-Image-Edit-2511 receives the picture of the shot (Picture 1) and the close-up of the
character (Picture 2: his or her real face) and puts the person of the close-up into the scene, wearing the wardrobe of the character. The scene of the choice (both characters holding out a hand) is made the
same way with the two close-ups.

    jobs(plan, cards, run)            -> what to make: {"id", "scene", "refs", "prompt", "shots"}
    run_jobs(plan, cards, run, url)   -> makes the missing pictures into <run>/stills_id/<id>.png (an earlier one is kept as _vN); timings per job
    source_of(run, shot)              -> the picture an animation starts from: the identity picture if there is one, else the still

Lab 2026-10-08: the SHORT prompt keeps the face; the clothes of the character must be WRITTEN in it (without them Elara ran in black instead of her white chef's coat). The app must run WITHOUT SageAttention
(black pictures otherwise: scripts/comfy_edit.py checks it and says so).
"""
from __future__ import annotations

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TWO_SHOT_ID = "two_shot"  # the picture of the offers and of the choice: both characters, c1 on the left, c2 on the right
SINGLE = ("Put the {who} from Picture 2 into the scene of Picture 1 in place of the person who is there, with the same pose and the same action. "
          "{pronoun} is wearing {wardrobe}. Keep the place, the light, the camera angle and exactly the same framing and size of the person as in Picture 1 (a close-up stays a close-up).")
TWO = ("Put the {first} from Picture 2 on the left and the {second} from Picture 3 on the right, side by side in the place of Picture 1, close together, both facing the camera, "
       "each one holding out one open hand toward the camera. Keep their faces, hair and clothes exactly as in Pictures 2 and 3. Same place, same light. "
       "On the left is {first_name} ({first_detail}); on the right is {second_name} ({second_detail}). Never swap them and never mix their faces or their clothes.")
WITH_PEOPLE = ("narration", "pov", "twist", "rewind")


def person_word(character: dict) -> str:
    return "woman" if character.get("gender") == "f" else "man"


def reference_of(cards: dict, character_id: str) -> Path:
    entry = cards["characters"][character_id]
    return ROOT / (entry.get("closeup") or entry["portrait"])["file"]


def prompt_single(character: dict) -> str:
    wardrobe = (character.get("wardrobe") or "the clothes of Picture 2").rstrip(". ")
    return SINGLE.format(who=person_word(character), pronoun="She" if character.get("gender") == "f" else "He", wardrobe=wardrobe[:1].lower() + wardrobe[1:])


def detail_of(character: dict) -> str:
    """What tells the two people apart for the edit model (two men looked alike and got each other's face and costume): age, face, wardrobe."""
    words = lambda text, n: " ".join((text or "").rstrip(". ").split()[:n]).lower()  # noqa: E731
    return f"{character.get('age', '?')} years old, {words(character.get('look'), 10)}, wearing {words(character.get('wardrobe'), 10)}"


def prompt_two(first: dict, second: dict) -> str:
    return TWO.format(first=person_word(first), second=person_word(second), first_name=first.get("name", "the first"), second_name=second.get("name", "the second"),
                      first_detail=detail_of(first), second_detail=detail_of(second))


def faces_seen(run: Path) -> dict[str, bool]:
    """What the vision model saw in the stills (run/stills_judge_N.json, the later rounds win): is a face or a whole body in the picture? A picture of a hand on a rung or of a holster does not get a face."""
    import json
    seen: dict[str, bool] = {}
    for path in sorted(run.glob("stills_judge_*.json")):
        for shot_id, verdict in json.loads(path.read_text(encoding="utf-8")).items():
            if "person_visible" in verdict.get("answer", {}):
                seen[shot_id] = bool(verdict["answer"]["person_visible"])
    return seen


def jobs(plan: dict, cards: dict, run: Path) -> list[dict]:
    characters = {c["id"]: c for c in plan["characters"]}
    seen = faces_seen(run)
    made: list[dict] = []
    offer_ids = [s["id"] for s in plan["shots"] if s["kind"] in ("offer", "choice")]
    first_offer = next((s for s in plan["shots"] if s["kind"] == "offer" and s.get("still")), None)
    if first_offer and offer_ids:
        c1, c2 = plan["characters"][0], plan["characters"][1]
        made.append({"id": TWO_SHOT_ID, "scene": run / "stills" / f"{first_offer['id']}.png", "refs": [reference_of(cards, c1["id"]), reference_of(cards, c2["id"])], "prompt": prompt_two(c1, c2), "shots": offer_ids})
    for shot in plan["shots"]:
        people = [i for i in shot.get("in_shot", []) if i in characters]
        if shot["kind"] not in WITH_PEOPLE or not shot.get("still") or not people or seen.get(shot["id"]) is False:
            continue
        if len(people) >= 2:
            first, second = characters[people[0]], characters[people[1]]
            made.append({"id": shot["id"], "scene": run / "stills" / f"{shot['id']}.png", "refs": [reference_of(cards, first["id"]), reference_of(cards, second["id"])], "prompt": prompt_two(first, second), "shots": [shot["id"]]})
        else:
            who = characters[people[0]]
            made.append({"id": shot["id"], "scene": run / "stills" / f"{shot['id']}.png", "refs": [reference_of(cards, who["id"])], "prompt": prompt_single(who), "shots": [shot["id"]]})
    return made


def missing(plan: dict, cards: dict, run: Path) -> list[str]:
    return [j["id"] for j in jobs(plan, cards, run) if not (run / "stills_id" / f"{j['id']}.png").exists()]


def run_jobs(plan: dict, cards: dict, run: Path, base_url: str, seed: int = 1, only: set[str] | None = None, force: bool = False) -> dict:
    import comfy_edit as ce
    out = run / "stills_id"
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    for index, job in enumerate(jobs(plan, cards, run)):
        if only and job["id"] not in only:
            continue
        target = out / f"{job['id']}.png"
        if target.exists() and not (force or only):
            continue
        if not job["scene"].exists():
            timings[job["id"]] = {"skipped": f"no scene picture {job['scene'].name}"}
            continue
        if target.exists():
            target.replace(target.with_name(f"{target.stem}_v{len(list(out.glob(target.stem + '_v*.png'))) + 1}.png"))
        started = time.time()
        ce.run_edit(job["scene"], target, job["prompt"], seed=seed + index, steps=4, references=job["refs"], base_url=base_url, label=f"{run.name}_identity_{job['id']}")
        timings[job["id"]] = {"seconds": round(time.time() - started, 1), "prompt": job["prompt"], "seed": seed + index}
        print(f"identity {job['id']}: {timings[job['id']]['seconds']} s", flush=True)
    return timings


def source_of(run: Path, shot: dict) -> Path:
    """The picture a clip starts from: the identity picture of the shot, the two-shot for the offers and the choice, else the still."""
    own = run / "stills_id" / f"{shot['id']}.png"
    if own.exists():
        return own
    two = run / "stills_id" / f"{TWO_SHOT_ID}.png"
    if shot["kind"] in ("offer", "choice") and two.exists():
        return two
    return run / "stills" / f"{shot['id']}.png"
