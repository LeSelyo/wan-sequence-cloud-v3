"""Story engine for the "you must choose" trend (second-person story, a choice A / B, branches, a twist, characters who speak a short line). It turns THREE LINES OF CONTEXT (or full parameters)
into a validated plan of shots that the production phases (images, edits, videos, voices, subtitles, montage) consume. Everything has a DEFAULT that is drawn at random from a seed and every
default can be overridden: `resolve_params` says, for each parameter, whether it was given, a fixed default or drawn by the seed (provenance), and that is written in the plan.

    python scripts/story_engine.py plan "an apocalypse on the moon, a spaceship, two people offer you help" --seed 5 [--language fr] [--seconds 75] [--out plan.json]

Story sources, best first: `write_plan_llm` (a local model answers with a JSON schema, the plan is validated and repaired, 2 repairs) and `write_plan_template` (no model: the same structure from
templates and the seed, always valid: the pipeline never depends on a model being up). All characters are FICTIONAL: real public figures named in the context are replaced by invented archetypes
and the replacement is reported in plan["substitutions"].
Shot kinds: narration (voice-over on a picture), talk (a character says a short line IN the scene: voice tied to the image), pov (first-person hands), choice (the A / B card with a countdown),
twist (the last line). The language of the voice-over and of the lines is a parameter (en by default, fr available): the structure is the same.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES_FILE = ROOT / "results" / "story_trend" / "voices" / "voices.json"
WORDS_PER_SECOND = {"en": 2.6, "fr": 2.4}
SHOT_KINDS = ("narration", "talk", "pov", "choice", "twist")
CAMERAS = ("wide", "medium", "close", "pov")
TONES = ["tense", "bleak", "paranoid", "desperate", "cold"]
ENDINGS = ["twist", "bleak", "hopeful"]
FICTIONAL_NAMES = ["Kael", "Maren", "Oren", "Talia", "Brandt", "Sera", "Vik", "Nadia", "Corin", "Ilse", "Rhys", "Mara", "Dorian", "Lena", "Joss", "Anika"]
GENDERS = {"Kael": "m", "Maren": "f", "Oren": "m", "Talia": "f", "Brandt": "m", "Sera": "f", "Vik": "m", "Nadia": "f", "Corin": "m", "Ilse": "f", "Rhys": "m", "Mara": "f", "Dorian": "m", "Lena": "f",
           "Joss": "m", "Anika": "f"}
# parameters: key -> (default, how it is drawn when no default is fixed)
DEFAULTS = {"language": "en", "aspect": "9:16", "size": [576, 1024], "fps": 30, "target_seconds": 75, "choices": 1, "branches": 2, "pov": "second person", "world": "real world",
            "characters": None, "talking_lines": None, "tone": None, "ending": None, "voices": None, "style": None, "seed": None, "llm_model": "qwen3.6:27b", "story_source": "auto"}
FIXED = {"language", "aspect", "size", "fps", "target_seconds", "choices", "branches", "pov", "world", "llm_model", "story_source"}  # "default" provenance; the others are drawn by the seed


def resolve_params(given: dict | None = None, seed: int | None = None) -> tuple[dict, dict]:
    """(params, provenance). provenance[k] = "given" | "default" | "rng". A parameter drawn at random is reproducible: same seed, same draw."""
    given = {k: v for k, v in (given or {}).items() if v is not None}
    unknown = set(given) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    seed = given.get("seed", seed)
    seed = random.Random().randrange(2**31) if seed is None else seed
    rnd = random.Random(seed)
    drawn = {"characters": rnd.choice([2, 2, 3]), "talking_lines": rnd.choice([1, 2]), "tone": rnd.choice(TONES), "ending": rnd.choice(ENDINGS), "voices": "auto", "style": "auto"}
    params, provenance = {}, {}
    for key, default in DEFAULTS.items():
        if key == "seed":
            params[key], provenance[key] = seed, "given" if "seed" in given else "rng"
        elif key in given:
            params[key], provenance[key] = given[key], "given"
        elif key in FIXED:
            params[key], provenance[key] = default, "default"
        else:
            params[key], provenance[key] = drawn[key], "rng"
    if params["language"] not in WORDS_PER_SECOND:
        raise ValueError(f"language must be one of {sorted(WORDS_PER_SECOND)}")
    return params, provenance


# ------------------------------------------------------------------ the plan
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"}, "logline": {"type": "string"}, "language": {"type": "string"}, "substitutions": {"type": "array", "items": {"type": "string"}},
        "characters": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "name": {"type": "string"}, "role": {"type": "string"}, "gender": {"type": "string", "enum": ["m", "f"]}, "age": {"type": "integer"},
            "look": {"type": "string"}, "wardrobe": {"type": "string"}}, "required": ["id", "name", "role", "gender", "age", "look", "wardrobe"]}},
        "locations": {"type": "array", "minItems": 2, "maxItems": 8, "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "kind": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}, "description": {"type": "string"}, "variant_of": {"type": ["string", "null"]}},
            "required": ["id", "kind", "tags", "description"]}},
        "beats": {"type": "array", "minItems": 8, "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": list(SHOT_KINDS)}, "text": {"type": "string"}, "speaker": {"type": "string"}, "location": {"type": "string"},
            "in_shot": {"type": "array", "items": {"type": "string"}}, "visual": {"type": "string"}, "camera": {"type": "string", "enum": list(CAMERAS)},
            "choice": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}}, "required": ["kind", "text", "speaker", "location", "visual", "camera"]}},
    },
    "required": ["title", "logline", "language", "characters", "locations", "beats"],
}


def count_words(text: str) -> int:
    return len(re.findall(r"[\w'’-]+", text))


def validate_plan(plan: dict) -> list[str]:
    """Problems of a plan (empty = valid): references resolve, every talking shot has a character and a short line, there is at least one choice, a variant points to an existing location."""
    problems = []
    characters = {c["id"]: c for c in plan.get("characters", [])}
    locations = {loc["id"]: loc for loc in plan.get("locations", [])}
    if len(characters) < 2:
        problems.append("at least two characters")
    for loc in locations.values():
        if loc.get("variant_of") and loc["variant_of"] not in locations:
            problems.append(f"location {loc['id']} is a variant of an unknown location")
    beats = plan.get("beats", [])
    if not any(b["kind"] == "choice" for b in beats):
        problems.append("no choice beat")
    if not any(b["kind"] == "talk" for b in beats):
        problems.append("no talking beat (a character must say a line)")
    for i, beat in enumerate(beats):
        where = f"beat {i}"
        if beat["kind"] not in SHOT_KINDS:
            problems.append(f"{where}: unknown kind {beat['kind']}")
        if beat.get("location") not in locations:
            problems.append(f"{where}: unknown location {beat.get('location')}")
        if beat["kind"] == "talk":
            if beat.get("speaker") not in characters:
                problems.append(f"{where}: a talking beat needs a speaker who is a character")
            if count_words(beat["text"]) > 18:
                problems.append(f"{where}: a spoken line must stay short (18 words max)")
        elif beat["kind"] in ("narration", "twist", "pov") and beat.get("speaker", "narrator") != "narrator":
            problems.append(f"{where}: {beat['kind']} is spoken by the narrator")
        for who in beat.get("in_shot", []):
            if who not in characters:
                problems.append(f"{where}: {who} is not a character")
        if beat["kind"] == "choice" and not (beat.get("choice") or {}).get("a") or beat["kind"] == "choice" and not beat["choice"].get("b"):
            problems.append(f"{where}: a choice needs a and b")
    return problems


def estimate_seconds(plan: dict, language: str | None = None) -> float:
    wps = WORDS_PER_SECOND[language or plan.get("language", "en")]
    total = 0.0
    for beat in plan["beats"]:
        total += (3.0 if beat["kind"] == "choice" else max(1.8, count_words(beat["text"]) / wps + 0.3))
    return round(total, 1)


def compile_shots(plan: dict, max_words: int = 16) -> list[dict]:
    """Beats -> shots: a long narration is cut at sentence ends into several shots (same place, another camera) so no picture stays more than a few seconds; a talking line, a choice and the
    twist stay in one shot. Each shot has an id, a kind, the text to speak, who speaks it, where, who is in the picture, the visual and the camera."""
    shots, cycle = [], 0
    for beat in plan["beats"]:
        chunks = [beat["text"]]
        if beat["kind"] in ("narration", "pov") and count_words(beat["text"]) > max_words:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", beat["text"]) if s.strip()]
            chunks, current = [], ""
            for sentence in sentences:
                if current and count_words(current + " " + sentence) > max_words:
                    chunks.append(current)
                    current = sentence
                else:
                    current = (current + " " + sentence).strip()
            if current:
                chunks.append(current)
        for k, chunk in enumerate(chunks):
            camera = beat["camera"] if k == 0 else ("wide", "close", "medium")[cycle % 3]
            cycle += k > 0
            shots.append({"id": f"s{len(shots) + 1:03d}", "kind": beat["kind"], "text": chunk, "speaker": beat.get("speaker", "narrator"), "location": beat["location"],
                          "in_shot": beat.get("in_shot", []), "visual": beat["visual"], "camera": camera, "choice": beat.get("choice")})
    return shots


# ------------------------------------------------------------------ writing the story
def write_plan_llm(context: str, params: dict, llm_json, attempts: int = 3) -> dict | None:
    """The local model writes the plan with PLAN_SCHEMA, the plan is validated, the model is told the problems (2 repairs). None when it never gets valid (the caller uses the template)."""
    language = {"en": "English", "fr": "French"}[params["language"]]
    words = int(params["target_seconds"] * WORDS_PER_SECOND[params["language"]])
    prompt = (f"Write a second-person interactive story for a vertical video, in {language}, about {words} words of narration in total.\nContext (3 lines): {context}\n"
              f"Tone: {params['tone']}. Ending: {params['ending']}. {params['characters']} named characters, {params['choices']} choice with {params['branches']} branches (A and B), "
              f"{params['talking_lines']} short spoken line(s) per main character (kind 'talk', 18 words max, said IN the scene, e.g. 'No, don't listen to him, come with me'). "
              "Beats kinds: narration (voice-over), talk, pov (first-person hands), choice (give a and b), twist (last beat). Short punchy sentences. Locations may have variants (variant_of) "
              "when the same big place has different rooms; every beat names its location id. 'visual' = one concrete filmable sentence without any style words. "
              "ALL characters are fictional. If the context names real public figures, replace each with an invented archetype with a made-up name and list the replacement in 'substitutions'.")
    last: list[str] = []
    for _ in range(attempts):
        try:
            plan = llm_json(prompt + (f"\nYour previous plan had these problems, fix them: {last}" if last else ""), PLAN_SCHEMA)
        except Exception:
            return None
        plan.setdefault("substitutions", [])
        plan["language"] = params["language"]
        last = validate_plan(plan)
        if not last:
            return plan
    return None


def ollama_json(prompt: str, schema: dict, model: str = "qwen3.6:27b", host: str = "http://127.0.0.1:11434", seed: int = 0) -> dict:
    """Structured output through Ollama's `format` (a JSON schema): the model cannot answer outside it. Needs ollama on the box (`ollama pull qwen3.6:27b`, 17 GB)."""
    import httpx
    resp = httpx.post(host.rstrip("/") + "/api/chat", json={"model": model, "stream": False, "think": False, "format": schema, "messages": [{"role": "user", "content": prompt}],
                                                          "options": {"num_ctx": 8192, "temperature": 0.7, "seed": seed}}, timeout=900)
    resp.raise_for_status()
    return json.loads(resp.json()["message"]["content"])


TEMPLATES = {
    "en": {
        "hook": ["You look up. {event}.", "{event}, and you are the only one who sees it.", "The sky changes first. {event}."],
        "offer_a": "{a} reaches you first. \"{line_a}\"", "offer_b": "{b} blocks the way. \"{line_b}\"",
        "lines_a": ["Come with me. I can get you out of here.", "No, don't listen to him. Come with me.", "Trust me. I have the only safe place left."],
        "lines_b": ["Don't go with him. He lies about everything.", "Stay with me. I will keep you alive.", "Listen to me, I have what you need."],
        "choose": "Now you must choose.", "branch_a": ["You follow {a}.", "At first it feels like safety.", "{a} gives the orders and everyone obeys.", "Then people start to disappear.",
                                                       "You discover why. {a} decides who stays."], "branch_b": ["You follow {b}.", "At first {b} shares everything.", "But supplies run low.",
                                                                                                                 "Someone must be left behind.", "You are the one {b} chooses."],
        "twist": ["In the end you understand: it was never about survival.", "In the end you understand: you were the one they were waiting for.", "In the end you understand: nobody was coming."],
    },
    "fr": {
        "hook": ["Tu lèves les yeux. {event}.", "{event}, et tu es le seul à le voir.", "Le ciel change d'abord. {event}."],
        "offer_a": "{a} te rejoint en premier. « {line_a} »", "offer_b": "{b} te barre la route. « {line_b} »",
        "lines_a": ["Viens avec moi. Je peux te sortir d'ici.", "Non, ne l'écoute pas. Viens avec moi.", "Fais-moi confiance. J'ai le seul abri sûr."],
        "lines_b": ["Ne pars pas avec lui. Il ment sur tout.", "Reste avec moi. Je te garderai en vie.", "Écoute-moi, j'ai ce qu'il te faut."],
        "choose": "Maintenant tu dois choisir.", "branch_a": ["Tu suis {a}.", "Au début, c'est la sécurité.", "{a} donne les ordres et tout le monde obéit.", "Puis des gens disparaissent.",
                                                       "Tu découvres pourquoi. {a} décide qui reste."], "branch_b": ["Tu suis {b}.", "Au début, {b} partage tout.", "Mais les vivres baissent.",
                                                                                                                 "Quelqu'un doit être laissé derrière.", "C'est toi que {b} choisit."],
        "twist": ["À la fin tu comprends : ce n'était jamais une question de survie.", "À la fin tu comprends : c'est toi qu'ils attendaient.", "À la fin tu comprends : personne ne viendrait."],
    },
}
EVENTS = {  # the opening event must fit the kind of place (a moon story never opens on a city blackout)
    "spaceship": {"en": ["The moon splits in two", "The planet below turns dark", "The stars begin to go out one by one"],
                  "fr": ["La lune se fissure", "La planète en dessous s'assombrit", "Les étoiles s'éteignent une à une"]},
    "shelter": {"en": ["The ground shakes and the lights die", "The alarm sounds and the doors seal", "The sky outside turns red"],
                "fr": ["Le sol tremble et les lumières meurent", "L'alarme sonne et les portes se scellent", "Le ciel dehors devient rouge"]},
    "flood": {"en": ["The river breaks through the dam", "The water rises in the streets in minutes", "The sirens stop and the water keeps coming"],
              "fr": ["Le fleuve rompt le barrage", "L'eau monte dans les rues en quelques minutes", "Les sirènes s'arrêtent et l'eau continue d'arriver"]},
    "city": {"en": ["The power goes out across the whole city", "The sky turns red", "The sea pulls back from the coast"],
             "fr": ["Le courant s'éteint dans toute la ville", "Le ciel devient rouge", "La mer se retire de la côte"]},
}
PLACE_KITS = {  # keywords of the context -> (kind, tags, [(id, description, variant_of)])
    "space": ("spaceship", ["space", "window"], [("deck", "the main deck of a huge spaceship, long walls and a viewport", None), ("corridor", "a narrow corridor of the same spaceship", "deck"),
                                                 ("quarters", "a small private room of the same spaceship", "deck"), ("cargo", "the cargo bay of the same spaceship, crates and bare floor", "deck")]),
    "shelter": ("shelter", ["shelter"], [("hall", "the main hall of an underground shelter, crowded", None), ("storage", "the food storage room of the same shelter", "hall"),
                                          ("quarters", "a cramped sleeping room of the same shelter", "hall"), ("tunnel", "a long service tunnel of the same shelter", "hall")]),
    "flood": ("flood", ["city", "sea", "window"], [("boat", "the open deck of a small rescue boat at night, rails and a spotlight", None), ("street", "a flooded city street seen from the water, half-submerged cars and dark windows", "boat"),
                                                   ("cabin", "the small cabin of the same rescue boat, a bunk and a lamp", "boat"), ("roof", "a rooftop above the flood water of the same city", "boat"),
                                                   ("sea", "the open black sea at night under fog, far from any coast", "boat")]),
    "city": ("city", ["city", "window"], [("street", "a empty city street at dusk", None), ("office", "an office with a view of the same street", "street"),
                                           ("stairs", "a concrete stairwell of the same building", "street"), ("roof", "the roof of the same building", "street")]),
}


def kit_for(context: str) -> tuple:
    """The kind of place from WHOLE WORDS of the context (a substring test took 'ship captain' for a spaceship)."""
    words = set(re.findall(r"[a-z]+", context.lower()))
    if words & {"space", "moon", "spaceship", "starship", "planet", "orbit", "alien", "aliens"}:
        return PLACE_KITS["space"]
    if words & {"flood", "flooded", "boat", "river", "dam", "tsunami", "storm"}:
        return PLACE_KITS["flood"]
    if words & {"bunker", "shelter", "underground", "survivor", "survivors", "war"}:
        return PLACE_KITS["shelter"]
    return PLACE_KITS["city"]


def write_plan_template(context: str, params: dict) -> dict:
    """The no-model story: the same structure (hook, two characters who offer help and say a line each, the choice, branch A, branch B, twist) from templates and the seed. Always valid."""
    rnd = random.Random(params["seed"])
    t = TEMPLATES[params["language"]]
    names = rnd.sample(FICTIONAL_NAMES, params["characters"])
    a, b = names[0], names[1]
    kind, tags, rooms = kit_for(context)
    locations = [{"id": rid, "kind": kind, "tags": tags, "description": desc, "variant_of": var} for rid, desc, var in rooms]
    ids = [r["id"] for r in locations]
    characters = [{"id": f"c{i + 1}", "name": n, "role": "offers help" if i < 2 else "bystander", "gender": GENDERS[n], "age": rnd.randint(28, 62),
                   "look": rnd.choice(["weathered face, short dark hair", "sharp eyes, grey at the temples", "tired eyes, scar on the chin", "tall, calm, direct stare"]),
                   "wardrobe": rnd.choice(["worn dark coat", "utility jacket with patches", "old uniform without insignia", "plain dark sweater"])} for i, n in enumerate(names)]
    beats = [{"kind": "narration", "text": rnd.choice(t["hook"]).format(event=rnd.choice(EVENTS[kind][params["language"]])), "speaker": "narrator", "location": ids[0], "in_shot": [],
              "visual": "the viewer looks at the sky or the horizon through a window as something enormous breaks apart", "camera": "wide"},
             {"kind": "talk", "text": rnd.choice(t["lines_a"]), "speaker": "c1", "location": ids[0], "in_shot": ["c1"], "visual": f"{a} speaks straight to the viewer, urgent, hand held out", "camera": "close"},
             {"kind": "talk", "text": rnd.choice(t["lines_b"]), "speaker": "c2", "location": ids[0], "in_shot": ["c2"], "visual": f"{b} steps in front of the viewer, firm, one hand raised", "camera": "close"},
             {"kind": "choice", "text": t["choose"], "speaker": "narrator", "location": ids[0], "in_shot": ["c1", "c2"], "visual": "the two characters face the viewer side by side", "camera": "medium",
              "choice": {"a": a, "b": b}}]
    keep = 2 if params["target_seconds"] < 30 else (3 if params["target_seconds"] < 45 else len(t["branch_a"]))  # a short video keeps only the first lines of a branch
    branches = (("a", a, "branch_a", "c1"), ("b", b, "branch_b", "c2"))[: max(1, min(2, params["branches"]))]
    for branch, who, key, character in branches:
        for i, line in enumerate(t[key][:keep]):
            room = ids[1 + i % (len(ids) - 1)]
            beats.append({"kind": "pov" if i == 1 else "narration", "text": line.format(a=a, b=b), "speaker": "narrator", "location": room, "in_shot": [character] if i in (0, 2, 4) else [],
                          "visual": f"first-person view in the {room}: the viewer's own hands and {who} nearby" if i == 1 else f"{who} in the {room}, tense atmosphere", "camera": "pov" if i == 1 else rnd.choice(CAMERAS[:3])})
        beats.append({"kind": "twist", "text": rnd.choice(t["twist"]), "speaker": "narrator", "location": ids[-1], "in_shot": [], "visual": "the viewer alone, the place empty and silent", "camera": "wide"})
    return {"title": slug_title(context), "logline": context.strip()[:200], "language": params["language"], "substitutions": [], "characters": characters, "locations": locations, "beats": beats}


def slug_title(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().split(".")[0])[:60].capitalize()


# ------------------------------------------------------------------ voices
def load_voices(path: Path = VOICES_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["voices"] if path.exists() else []


def voice_description(character: dict) -> str:
    """The text given to Qwen3-TTS VoiceDesign when no recorded voice fits the character."""
    decade = max(2, min(8, character["age"] // 10)) * 10
    return f"{'a woman' if character['gender'] == 'f' else 'a man'} in {'her' if character['gender'] == 'f' else 'his'} {decade}s, natural realistic conversational voice, {character['role']}, calm and urgent"


def assign_voices(plan: dict, voices: list[dict], seed: int = 0) -> dict[str, str]:
    """narrator -> a voice whose default role is narrator; each character -> a distinct recorded voice of ITS gender (a hint that is None fits nobody); when there is none, "design:<description>"
    (a voice made from a text description by Qwen3-TTS VoiceDesign). Never a voice of the wrong gender. Drawn by the seed."""
    rnd = random.Random(seed)
    narrator = next((v for v in voices if v.get("role_default") == "narrator"), voices[0] if voices else None)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    if narrator:
        mapping["narrator"], used = narrator["id"], {narrator["id"]}
    else:
        mapping["narrator"] = "design:a calm captivating narrator, natural realistic voice"
    for c in plan["characters"]:
        pool = [v for v in voices if v["id"] not in used and v.get("gender_hint") == c["gender"]]
        if pool:
            pick = rnd.choice(pool)
            mapping[c["id"]] = pick["id"]
            used.add(pick["id"])
        else:
            mapping[c["id"]] = "design:" + voice_description(c)
    return mapping


def make_plan(context: str, given: dict | None = None, seed: int | None = None, llm_json=None, voices: list[dict] | None = None) -> dict:
    """The whole planning step: resolve parameters, write the story (model if given and valid, else the template), validate, estimate, assign voices. Returns the plan with its provenance."""
    params, provenance = resolve_params(given, seed)
    plan, source = None, "template"
    if llm_json is not None and params["story_source"] in ("auto", "llm"):
        plan = write_plan_llm(context, params, llm_json)
        source = "llm" if plan else "template (the model never gave a valid plan)"
    if plan is None:
        plan = write_plan_template(context, params)
    problems = validate_plan(plan)
    if problems:
        raise ValueError(f"the plan is invalid: {problems}")
    plan.update({"context": context, "params": params, "provenance": provenance, "story_source": source, "estimated_seconds": estimate_seconds(plan, params["language"]), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "shots": compile_shots(plan)})
    plan["voices"] = assign_voices(plan, voices if voices is not None else load_voices(), params["seed"])
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("context")
    p.add_argument("--seed", type=int)
    p.add_argument("--language", choices=sorted(WORDS_PER_SECOND))
    p.add_argument("--seconds", type=int)
    p.add_argument("--llm", action="store_true", help="use the local model (ollama) when it answers")
    p.add_argument("--out", type=Path)
    args = parser.parse_args()
    given = {k: v for k, v in {"language": args.language, "target_seconds": args.seconds}.items() if v is not None}
    plan = make_plan(args.context, given, args.seed, llm_json=(lambda prompt, schema: ollama_json(prompt, schema)) if args.llm else None)
    text = json.dumps(plan, indent=1, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(f"{plan['title']} | {plan['story_source']} | {len(plan['shots'])} shots, about {plan['estimated_seconds']} s | provenance: "
          + ", ".join(f"{k}={v}" for k, v in plan["provenance"].items() if v != "default"))
    for shot in plan["shots"]:
        print(f"{shot['id']} {shot['kind']:9} {shot['camera']:6} {shot['location']:9} {shot['speaker']:9} {shot['text'][:70]}")


if __name__ == "__main__":
    main()
