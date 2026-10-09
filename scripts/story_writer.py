"""The AUTOMATIC STORY of the "you must choose" trend: from THREE LINES OF CONTEXT (and optional parameters) a local model (Ollama + Qwen, on the rented box) writes the whole video as a list of SHOTS,
each with its spoken line, its own picture prompt, its own camera move and montage effects; the plan is validated, repaired (the model is told what is wrong) and, if the model never gets it right,
a template story with the same structure is used: the pipeline never stops for lack of a model.

    python scripts/story_writer.py "A flooded city at night. Two strangers on a rescue boat each offer to save you: a ship captain and a doctor." --seconds 150 --out plan.json [--seed 1] [--language en] [--llm]

Structure written (branches=2): HOOK (a grandiose picture + the words TikTok reads) -> SETUP -> OFFERS (the two characters speak, with clues for both) -> CHOICE -> branch A -> (rewind) -> branch B -> closing question.
`endings` says how each branch ends: "bad" for one, "good" for the other (drawn by the seed unless given). The spoken text is in the chosen language; picture and motion prompts stay in English (the image and video
models are English).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_engine as se  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PLAN = ROOT / "results" / "story_trend" / "plans" / "the_last_city_150_en.json"  # the shots of the first hand-written 2m30 story show the model what a good plan looks like
KINDS = ["narration", "talk", "pov", "choice", "twist", "rewind", "offer"]
MAX_TALK_BEATS = 4  # the characters speak (a mouth that follows the voice) only when the scene calls for it
CAMERAS = ["wide", "medium", "close", "pov"]
MAX_SPOKEN_WORDS = 14
SECONDS_PER_SHOT = 3.1  # the average length of a shot (voice + a short tail): sets how many shots a video of N seconds needs

SHOT_SCHEMA = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": KINDS}, "branch": {"type": "string", "enum": ["main", "A", "B"]}, "speaker": {"type": "string"}, "text": {"type": "string"},
    "location": {"type": "string"}, "in_shot": {"type": "array", "items": {"type": "string"}}, "still": {"type": "string"}, "motion": {"type": "string"},
    "camera": {"type": "string", "enum": CAMERAS},
    "fx": {"type": "object", "properties": {"zoom": {"type": "number"}, "shake": {"type": "number"}, "flash": {"type": "boolean"}}, "required": ["zoom", "shake", "flash"]},
    "tag": {"type": "string"}, "time": {"type": "string"}, "ending": {"type": "string", "enum": ["", "bad", "good"]},
    "choice": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}},
    "required": ["kind", "branch", "speaker", "text", "location", "in_shot", "still", "motion", "camera", "fx"]}
STORY_SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"}, "hook_title": {"type": "string"}, "logline": {"type": "string"}, "substitutions": {"type": "array", "items": {"type": "string"}},
    "characters": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "name": {"type": "string"}, "role": {"type": "string"}, "gender": {"type": "string", "enum": ["m", "f"]},
                                                                                "age": {"type": "integer"}, "look": {"type": "string"}, "wardrobe": {"type": "string"}},
                                                 "required": ["id", "name", "role", "gender", "age", "look", "wardrobe"]}},
    "shots": {"type": "array", "items": SHOT_SCHEMA}, "end_card": {"type": "string"}, "end_card_small": {"type": "string"}, "caption": {"type": "string"}},
    "required": ["title", "hook_title", "characters", "shots", "end_card", "end_card_small", "caption"]}


def target_shots(seconds: float, shot_scale: float = 1.0) -> int:
    return max(10, int(round(seconds / (SECONDS_PER_SHOT * shot_scale))))


def draw_endings(seed: int) -> dict:
    """Which branch ends badly and which well (one of each), drawn by the seed."""
    return {"A": "bad", "B": "good"} if random.Random(seed * 7 + 3).random() < 0.5 else {"A": "good", "B": "bad"}


def example_shots(limit: int = 17) -> list[dict]:
    """A short extract of the first hand-written story as the model's example: the hook, two offers, the choice, the end of a branch, the rewind and the start of the other."""
    if not EXAMPLE_PLAN.exists():
        return []
    shots = json.loads(EXAMPLE_PLAN.read_text(encoding="utf-8"))["shots"]
    picks = [0, 1, 2, 3, 4, 6, 7, 9, 13, 14, 15, 16, 28, 29, 30, 31, 32, 33, 44, 45]
    keys = ("kind", "branch", "speaker", "text", "location", "in_shot", "still", "motion", "camera", "fx", "tag", "time", "ending", "choice", "offer_of")
    out = []
    for index in picks[:limit]:
        if index < len(shots):
            shot = {k: shots[index][k] for k in keys if k in shots[index] and shots[index][k] not in (None, "", {})}
            shot.setdefault("branch", "main" if index < 15 else ("A" if index < 30 else "B"))
            out.append(shot)
    return out


def build_prompt(context: str, params: dict, endings: dict, location_ids: list[str], example: bool = True) -> str:
    language = {"en": "English", "fr": "French"}[params["language"]]
    n = target_shots(params["target_seconds"])
    two = params["branches"] >= 2
    words = int(params["target_seconds"] * se.WORDS_PER_SECOND[params["language"]] * 0.9)
    structure = ("1. HOOK (3 shots): the FIRST shot must be a GRANDIOSE establishing picture (epic scale, awe) that puts the viewer in the world instantly, spoken line contains the key words of the story "
                 "(the disaster, the place, 'last', 'you'); hook_title = two short UPPERCASE lines starting with 'POV:' that contain the words people search for.\n"
                 "2. SETUP (3 shots): you are in danger, help arrives.\n3. OFFERS (6-8 shots): the two main characters (c1, c2) each speak SHORT lines (kind 'talk'), accuse each other, each one leaves a CLUE; "
                 "include at least one narration shot that shows a clue.\n4. CHOICE (1 shot, kind 'choice', speaker narrator, text like 'Now you must choose.', choice={a: name of c1, b: name of c2}).\n")
    if two:
        structure += (f"5. BRANCH A (branch 'A', you follow c1): about {int(n * 0.28)} shots, ends {endings['A'].upper()} ; its LAST shot is kind 'twist' with ending '{endings['A']}' and a tag 'ENDING A: ...'; the FIRST shot of the branch has tag 'CASE A: <name>'.\n"
                      f"6. REWIND (1 shot, kind 'rewind', branch 'B', narrator line like 'Rewind. What if you had chosen <name of c2>?').\n"
                      f"7. BRANCH B (branch 'B', you follow c2): about {int(n * 0.30)} shots, ends {endings['B'].upper()} ; its LAST shot is kind 'twist' with ending '{endings['B']}' and a tag 'ENDING B: ...'; "
                      "the FIRST shot of the branch has tag 'CASE B: <name>'. Branch B has at least 3 FAST dynamic action shots (chase, collapse, escape) and at least 2 talk lines.\n"
                      "8. end_card = a closing question (2 short UPPERCASE lines), end_card_small = the two endings + 'FOLLOW FOR PART 2'.\n")
    else:
        structure += (f"5. BRANCH A (branch 'A', you follow c1): about {int(n * 0.5)} shots, ends {endings['A'].upper()}; the last shot is kind 'twist' with ending '{endings['A']}'.\n"
                      "6. end_card = a closing question, end_card_small = 'FOLLOW FOR PART 2'.\n")
    sentences = len(re.findall(r"[.!?]+(?:\s|$)", context.strip())) or 1  # outside the f-string: a backslash in an f-string expression is a SyntaxError before Python 3.12 (the image runs 3.10)
    return (f"You write a vertical TikTok video, 'you must choose' trend: a second-person interactive story ('you') with a choice and a twist ending, as a JSON list of SHOTS.\n"
            f"CONTEXT ({sentences} sentences): {context}\n\nLANGUAGE of the spoken text, hook_title, end cards and tags: {language}. The 'still', 'motion' and 'time' fields are ALWAYS English.\n"
            f"TARGET: about {params['target_seconds']} seconds = about {n} shots, about {words} spoken words in total. Tone: {params['tone']}. Everything fictional: invent new names; if the context names a real public figure, "
            f"replace them by an invented archetype and list the replacement in 'substitutions'.\n\nSTRUCTURE (in this order):\n{structure}\n"
            f"RULES: every shot has a spoken 'text' of at most {MAX_SPOKEN_WORDS} words (short punchy sentences). kind 'talk' = a main character says it IN the scene (speaker c1 or c2); every other kind is the narrator "
            f"(speaker 'narrator'). 'location' is one of {location_ids}. 'in_shot' lists the character ids visible. For every shot but talk and choice, 'still' = ONE concrete filmable picture description (what is in the frame, "
            "the angle, the light) WITHOUT any text, letters or style words, 20-45 words, and 'motion' = the camera move and what moves in the picture (10-25 words) - vary the camera: aerial flight, whip pan, low angle tilt-up, "
            "handheld run, slow push-in, crane, dolly. For talk and choice shots 'still' is empty and 'motion' describes the acting. fx = {zoom 0.03-0.08, shake 0-1.0 (high only for impacts, chases, crashes), flash true only on "
            "a shock}. camera in [wide, medium, close, pov]. A shot may set 'time' (English, e.g. 'it is dawn: golden sunrise light') only when the hour changes. Characters: exactly two main ones, ids c1 and c2, realistic, "
            "with a profession, an age, a face description and a wardrobe. Make the ending of one branch BAD for you and the other GOOD, with a surprising twist in each.\n\n"
            + (f"EXAMPLE of shots from another story (follow this level of detail and this format; do NOT reuse its plot):\n{json.dumps(example_shots(), ensure_ascii=False)}\n\n" if example else "")
            + "Answer with the JSON of the story only.")


def validate_story(story: dict, params: dict, endings: dict, location_ids: list[str]) -> list[str]:
    """The problems of a story (empty = valid). Everything the later steps rely on is checked here."""
    problems: list[str] = []
    characters = {c.get("id"): c for c in story.get("characters", [])}
    if set(characters) != {"c1", "c2"}:
        problems.append("exactly two main characters with ids c1 and c2")
    shots = story.get("shots", [])
    n = target_shots(params["target_seconds"], params.get("shot_scale", 1.0))
    if not (0.7 * n <= len(shots) <= 1.35 * n):
        problems.append(f"{len(shots)} shots: about {n} are needed for {params['target_seconds']} s")
    two = params["branches"] >= 2
    if not str(story.get("hook_title", "")).strip() or "\n" not in story["hook_title"].strip():
        problems.append("hook_title must be two lines separated by a newline")
    if not story.get("end_card") or not story.get("caption"):
        problems.append("end_card and caption are required")
    choice_index = next((i for i, s in enumerate(shots) if s.get("kind") == "choice"), None)
    if sum(1 for s in shots if s.get("kind") == "choice") != 1:
        problems.append("exactly one choice shot")
    names = {c.get("name") for c in story.get("characters", [])}
    for i, shot in enumerate(shots):
        where = f"shot {i + 1}"
        kind = shot.get("kind")
        if kind not in KINDS:
            problems.append(f"{where}: unknown kind {kind}")
            continue
        words = se.count_words(shot.get("text", ""))
        if words == 0 or words > MAX_SPOKEN_WORDS:
            problems.append(f"{where}: the spoken text must have 1-{MAX_SPOKEN_WORDS} words (has {words})")
        if kind == "talk" and shot.get("speaker") not in ("c1", "c2"):
            problems.append(f"{where}: a talk shot is spoken by c1 or c2")
        if kind == "offer" and shot.get("offer_of", shot.get("speaker")) not in ("c1", "c2"):
            problems.append(f"{where}: an offer shot is the proposal of c1 or c2")
        if kind not in ("talk", "offer") and shot.get("speaker") != "narrator":
            problems.append(f"{where}: {kind} is spoken by the narrator")
        if shot.get("location") not in location_ids:
            problems.append(f"{where}: location must be one of {location_ids}")
        if any(who not in characters for who in shot.get("in_shot", [])):
            problems.append(f"{where}: in_shot has an unknown character")
        if kind == "offer" and not {"c1", "c2"} <= set(shot.get("in_shot", [])):
            problems.append(f"{where}: both characters are in the offer shot")
        if kind not in ("talk", "choice") and se.count_words(shot.get("still", "")) < 8:
            problems.append(f"{where}: a still description of at least 8 words is needed")
        if se.count_words(shot.get("motion", "")) < 4:
            problems.append(f"{where}: a motion description is needed")
        if kind == "choice" and not ({shot.get("choice", {}).get("a"), shot.get("choice", {}).get("b")} <= names):
            problems.append(f"{where}: the choice must name the two characters")
    twists = [s for s in shots if s.get("kind") == "twist"]
    if len(twists) != (2 if two else 1):
        problems.append(f"{2 if two else 1} twist shot(s) are needed (one per branch)")
    for twist in twists:
        branch = twist.get("branch")
        if branch in endings and twist.get("ending") != endings[branch]:
            problems.append(f"the twist of branch {branch} must have ending '{endings[branch]}'")
    if choice_index is not None and shots:
        if any(s.get("branch") != "main" for s in shots[:choice_index + 1]):
            problems.append("every shot up to the choice has branch 'main'")
        if shots[0].get("kind") not in ("narration", "pov"):
            problems.append("the first shot is a narration or pov hook")
        if two and not any(s.get("kind") == "rewind" for s in shots):
            problems.append("a rewind shot between the branches")
    talks = [s for s in shots if s.get("kind") == "talk"]
    if len(talks) > MAX_TALK_BEATS:
        problems.append(f"{len(talks)} talk shots: the characters speak only when they give an order or shout, at most {MAX_TALK_BEATS}")
    offers = [i for i, s in enumerate(shots) if s.get("kind") == "offer"]
    if offers and (len(offers) != 2 or offers[1] != offers[0] + 1 or [shots[i].get("offer_of", shots[i].get("speaker")) for i in offers] != ["c1", "c2"]):
        problems.append("exactly two consecutive offer shots, c1 then c2: the ONLY moment where both are in the picture and speak")
    if offers and any(shots[i].get("kind") != "choice" for i in [offers[-1] + 1] if i < len(shots)):
        problems.append("the choice comes immediately after the second offer")
    by_id = {c.get("id"): c.get("name", "") for c in story.get("characters", [])}
    for i, shot in enumerate(shots):
        chosen = {"A": "c1", "B": "c2"}.get(shot.get("branch")) if shot.get("kind") != "rewind" else None
        if not chosen:
            continue
        other = "c2" if chosen == "c1" else "c1"
        if other in shot.get("in_shot", []) or (by_id.get(other) and re.search(rf"\b{re.escape(by_id[other].lower())}\b", shot.get("text", "").lower())):
            problems.append(f"shot {i + 1}: branch {shot['branch']} is about you and {by_id.get(chosen)} only: {by_id.get(other)} is neither shown nor named")
    total = estimate_seconds(shots, params["language"])
    if not (0.75 * params["target_seconds"] <= total <= 1.3 * params["target_seconds"]):
        problems.append(f"the spoken text lasts about {total:.0f} s, the target is {params['target_seconds']} s")
    return problems


def estimate_seconds(shots: list[dict], language: str = "en") -> float:
    wps = se.WORDS_PER_SECOND[language]
    return round(sum(2.6 if s.get("kind") == "choice" else max(1.6, se.count_words(s.get("text", "")) / wps + 0.35) for s in shots), 1)


def location_kit(context: str) -> tuple[list[dict], list[str]]:
    kind, tags, rooms = se.kit_for(context)
    locations = [{"id": rid, "kind": kind, "tags": tags, "description": desc, "variant_of": var} for rid, desc, var in rooms]
    return locations, [loc["id"] for loc in locations]


def write_story_llm(context: str, params: dict, endings: dict, location_ids: list[str], llm_json, attempts: int = 3, example: bool = True) -> tuple[dict | None, list[str]]:
    """The model writes the story with STORY_SCHEMA; it is told the problems of its previous attempt (repairs). Returns (story or None, the problems of the last attempt)."""
    prompt = build_prompt(context, params, endings, location_ids, example)
    problems: list[str] = []
    for attempt in range(attempts):
        try:
            story = llm_json(prompt + (f"\n\nYour previous answer had these problems, write the whole story again and fix them: {problems[:12]}" if problems else ""), STORY_SCHEMA)
        except Exception as error:  # the model is down, too slow or answered outside the schema: the template takes over
            return None, [f"model error: {error}"]
        problems = validate_story(story, params, endings, location_ids)
        if not problems:
            return story, []
    return None, problems


# ---------------------------------------------------------------- the story without a model: the template story, upgraded to shots with their own picture and move
def template_story(context: str, params: dict, endings: dict) -> dict:
    """The old template (se.write_plan_template) gives the beats; each beat becomes a shot with a picture prompt and a camera move made from its visual and its place."""
    plan = se.write_plan_template(context, params)
    shots = []
    cameras = ["slow push-in, handheld", "slow lateral glide", "slow tilt up", "slow dolly forward"]
    for i, shot in enumerate(se.compile_shots(plan)):
        location = next(loc for loc in plan["locations"] if loc["id"] == shot["location"])
        kind = shot["kind"]
        branch = "main" if kind in ("narration", "talk", "choice", "pov") and not shots else shots[-1]["branch"] if shots else "main"
        shots.append({"kind": kind, "branch": branch, "speaker": shot["speaker"], "text": shot["text"], "location": shot["location"], "in_shot": shot["in_shot"],
                      "still": "" if kind in ("talk", "choice") else f"{shot['visual']}, {location['description']}", "motion": shot["visual"] + ", " + cameras[i % len(cameras)], "camera": shot["camera"],
                      "fx": {"zoom": 0.05, "shake": 0.0, "flash": False}, "choice": shot.get("choice")})
    seen_choice = False
    for shot in shots:  # after the choice the first branch is A, the second B (the template writes A then B, each ending with a twist)
        if shot["kind"] == "choice":
            seen_choice = True
        elif seen_choice:
            shot["branch"] = "A" if not any(s["kind"] == "twist" and s["branch"] == "A" for s in shots[:shots.index(shot)]) else "B"
    for shot in shots:
        if shot["kind"] == "twist":
            shot["ending"] = endings.get(shot["branch"], "")
    plan_names = [c["name"] for c in plan["characters"]]
    return {"title": plan["title"], "hook_title": "POV: YOU MUST CHOOSE\nTHE END OF THE WORLD", "logline": plan["logline"], "substitutions": [], "characters": plan["characters"][:2],
            "shots": shots, "end_card": "WHICH ONE\nWOULD YOU CHOOSE?", "end_card_small": "FOLLOW FOR PART 2", "caption": f"{plan['logline']} Who would you choose, {plan_names[0]} or {plan_names[1]}? #youmustchoose #pov #apocalypse #fyp",
            "locations": plan["locations"]}


def make_story_plan(context: str, given: dict | None = None, seed: int | None = None, llm_json=None, voices: list[dict] | None = None, example: bool = True) -> dict:
    """The whole automatic planning: parameters (with provenance), the story (model if it answers validly, else template), the plan in the format of the production steps."""
    given = dict(given or {})
    endings_given = given.pop("endings", None)  # not a parameter of the engine: it says how each branch ends
    params, provenance = se.resolve_params(given, seed)
    endings = endings_given or draw_endings(params["seed"])
    provenance["endings"] = "given" if endings_given else "rng"
    locations, location_ids = location_kit(context)
    story, problems, source = None, [], "template"
    if llm_json is not None and params["story_source"] in ("auto", "llm"):
        story, problems = write_story_llm(context, params, endings, location_ids, llm_json, example=example)
        source = "llm" if story else f"template (the model never gave a valid story: {problems[:3]})"
    if story is None:
        story = template_story(context, params, endings)
        locations = story.pop("locations")
    shots = []
    for index, shot in enumerate(story["shots"]):
        clean = {k: v for k, v in shot.items() if v not in (None, "", {}) or k in ("still", "text")}
        shots.append({"id": f"s{index + 1:03d}", "visual": shot["motion"], **clean, "fx": shot.get("fx") or {}, "choice": shot.get("choice") or None})
    plan = {"title": story["title"], "logline": story.get("logline") or context.strip()[:200], "language": params["language"], "substitutions": story.get("substitutions", []), "characters": story["characters"],
            "locations": locations, "beats": [], "context": context, "params": params, "provenance": provenance, "endings": endings, "story_source": source, "shots": shots,
            "title_overlay": {"text": story["hook_title"].strip(), "start": 0.15, "end": 3.8, "y": 0.15, "pixels": 62}, "end_card": story["end_card"], "end_card_small": story["end_card_small"],
            "caption": story["caption"], "estimated_seconds": estimate_seconds(shots, params["language"]), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    plan["voices"] = se.assign_voices(plan, voices if voices is not None else se.load_voices(), params["seed"])
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("context")
    parser.add_argument("--seconds", type=int, default=150)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--language", choices=sorted(se.WORDS_PER_SECOND))
    parser.add_argument("--branches", type=int, choices=[1, 2])
    parser.add_argument("--llm", action="store_true", help="ask the model (Ollama on 127.0.0.1:11434 through the tunnel); without it the template story is used")
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    given = {k: v for k, v in {"language": args.language, "target_seconds": args.seconds, "branches": args.branches}.items() if v is not None}
    llm = (lambda prompt, schema: ollama_story(prompt, schema, args.model, args.seed or 0)) if args.llm else None
    plan = make_story_plan(args.context, given, args.seed, llm)
    text = json.dumps(plan, indent=1, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(f"{plan['title']} | {plan['story_source']} | {len(plan['shots'])} shots, about {plan['estimated_seconds']} s of speech | endings {plan['endings']}")
    for shot in plan["shots"]:
        print(f"{shot['id']} {shot['branch']:4} {shot['kind']:9} {shot['speaker']:8} {shot['text'][:80]}")


def ollama_story(prompt: str, schema: dict, model: str, seed: int, host: str = "http://127.0.0.1:11434") -> dict:
    """Structured output through Ollama's `format` (the JSON schema): the model cannot answer outside it. Thinking is off (a reasoning model would spend minutes before the JSON)."""
    import httpx
    resp = httpx.post(host.rstrip("/") + "/api/chat", json={"model": model, "stream": False, "think": False, "format": schema, "messages": [{"role": "user", "content": prompt}],
                                                          "options": {"num_ctx": 14336, "temperature": 0.8, "top_p": 0.95, "seed": seed, "num_predict": 9000}}, timeout=1800)
    resp.raise_for_status()
    return json.loads(resp.json()["message"]["content"])


if __name__ == "__main__":
    main()
