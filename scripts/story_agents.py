"""THE CHAIN OF AGENTS that writes the story of a trend video from a free context (three sentences or more), with constant prompts and a deterministic checker after every agent.

    ANALYST   context  -> BRIEF   (world, hour, places, the two characters with their promise and their HIDDEN TRUTH, keywords, facts that must appear)
    PLANNER   brief    -> OUTLINE (the beats act by act, the clues and the twists, the title, the closing question, the caption)
    WRITER    outline  -> LINES   (the spoken line of every beat, chunk by chunk, with the previous lines for continuity)
    DIRECTOR  lines    -> PICTURES (the still, the camera move and the effects of every shot, chunk by chunk)
    CODE      rhythm pass (no camera move three times in a row, shake on impacts), tags, endings, choice card, ids
    CHECKER   scripts/story_writer.validate_story on the assembled story; JUDGE (optional) scores it 1-10 on seven criteria and picks the best of several candidates.

What keeps it STABLE: the prompts live in scripts/trends/<trend>.py and never change (only their `{{slots}}` do), each agent answers inside a JSON schema, its answer is checked by code and the agent is told
what is wrong (repairs), a stage that never succeeds falls back to a plain template for THAT stage only (the report says which), and the examples put in the prompts come from the library of APPROVED stories
(scripts/story_library.py), so the quality follows what the user approved.

    python scripts/story_agents.py "context of three sentences or more" --seconds 150 --out plan.json [--llm] [--candidates 2] [--judge]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_engine as se  # noqa: E402
import story_library as lib  # noqa: E402
import story_style as ss  # noqa: E402
import story_writer as sw  # noqa: E402

IMPACT_WORDS = {"crash", "crashes", "collapse", "collapses", "falls", "fall", "explodes", "explosion", "scream", "screams", "shouts", "chase", "run", "runs", "racing", "guns", "speeding", "storm", "wave", "roars",
                "shoves", "jumps", "pulls"}


def load_trend(name: str):
    return importlib.import_module(f"trends.{name}")


# ---------------------------------------------------------------------------------------------- slots: constant prompt + variable fields
def fill(template: str, slots: dict) -> str:
    """Replace every {{slot}}; a slot that is missing or an unknown one left over is an error (the constant text must never go out half filled)."""
    names = set(re.findall(r"\{\{(\w+)\}\}", template))
    missing = names - set(slots)
    if missing:
        raise KeyError(f"slots not filled: {sorted(missing)}")
    text = template
    for name in names:
        text = text.replace("{{" + name + "}}", str(slots[name]))
    return text


def constant_hash(template: str) -> str:
    """A fingerprint of the constant part of a prompt (the template with its slots emptied): the same for every video of the trend, changes only when somebody edits the prompt."""
    return hashlib.sha1(re.sub(r"\{\{\w+\}\}", "{{}}", template).encode("utf-8")).hexdigest()[:10]


def prompt_fingerprint(trend) -> dict:
    return {"version": trend.PROMPT_VERSION, **{n.lower().replace("_prompt", ""): constant_hash(getattr(trend, n)) for n in ("ANALYST_PROMPT", "PLANNER_PROMPT", "WRITER_PROMPT", "DIRECTOR_PROMPT", "JUDGE_PROMPT", "LENGTH_PROMPT", "LOOK_PROMPT", "PREMISE_PROMPT") if hasattr(trend, n)}}


# ---------------------------------------------------------------------------------------------- one agent: ask, check, repair
def sized(schema: dict, key: str, count: int) -> dict:
    """The schema of an answer whose list `key` must have EXACTLY `count` items (measured on Ollama: the structured output enforces minItems/maxItems, so 'one line per beat' can no longer be missed: a chunk
    of the lab came back with 19 lines for 18 beats three times and fell back to a template)."""
    import copy
    out = copy.deepcopy(schema)
    out["properties"][key]["minItems"] = out["properties"][key]["maxItems"] = count
    return out


def clean_strings(value):
    """The model sometimes returns a broken apostrophe (U+FFFD: "Kael?s"): it is turned back into an apostrophe, a lone one is dropped."""
    if isinstance(value, str):
        return re.sub(r"(?<=\w)\ufffd(?=\w)", "'", value).replace("\ufffd", "")
    if isinstance(value, list):
        return [clean_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: clean_strings(v) for k, v in value.items()}
    return value


def call_agent(llm, prompt: str, schema: dict, check, attempts: int = 3, seed: int = 0, soft=None) -> tuple[dict | None, list[str], int]:
    """Ask the model; check the answer with code; tell the model what is wrong and ask again. Returns (answer or None, the problems of the last attempt, attempts used).
    `soft(problem)` marks the problems that are a matter of taste (a line a little long, an ending that reads mixed): when the attempts are used up and the BEST answer has only those, it is kept
    (with its problems listed) instead of falling back to a template, which is made of truncated purposes and reads like notes."""
    problems: list[str] = []
    best: tuple[dict, list[str], tuple[int, int]] | None = None
    for attempt in range(1, attempts + 1):
        text = prompt + (f"\n\nYour previous answer had these problems. Answer again, the whole JSON, and fix them: {problems[:10]}" if problems else "")
        try:
            answer = llm(text, schema, seed=seed + attempt)
        except Exception as error:  # the model is down, too slow or answered outside the schema
            return None, [f"model error: {str(error)[:200]}"], attempt
        answer = clean_strings(answer)
        problems = check(answer)
        if not problems:
            return answer, [], attempt
        key = (sum(1 for p in problems if not (soft and soft(p))), len(problems))  # the answer with the fewest HARD problems wins, then the fewest problems (a tie used to keep the first one, even
        if best is None or key < best[2]:  # when it had a hard problem and the last one only had remarks of taste: the whole chunk then fell back to a template)
            best = (answer, problems, key)
    if soft and best and all(soft(p) for p in best[1]):
        return best[0], best[1], attempts
    return None, problems, attempts


# ---------------------------------------------------------------------------------------------- stage 0: the IDEA agent (a context out of nothing)
def draw_seed_elements(params: dict, trend) -> dict:
    """The bones of a story drawn by the seed from the pools of the trend: a place and what happens to it, an hour, two roles, a tone. The same seed always draws the same bones."""
    rnd = random.Random(params["seed"] * 31 + 5)
    setting, premise, hour = rnd.choice(trend.IDEA_WORLDS)
    role_a, role_b = rnd.choice(trend.IDEA_ROLES)
    if rnd.random() < 0.5:
        role_a, role_b = role_b, role_a
    bones = {"setting": setting, "premise": premise, "hour": hour, "role_a": role_a, "role_b": role_b, "tone": rnd.choice(trend.IDEA_TONES), "mode": "pool"}
    if getattr(trend, "IDEA_INVENT", False):  # the model INVENTS the place from two ingredients drawn by the seed; the pool world stays as the fallback without a model
        bones.update(mode="invent", ingredients=rnd.sample(trend.IDEA_INGREDIENTS, 2), hour=rnd.choice(trend.HOURS), fallback_world=(setting, premise), kind=rnd.choice(trend.IDEA_KINDS),
                     genre=rnd.choice(trend.IDEA_GENRES), names=rnd.sample(trend.IDEA_NAMES, 2))
    return bones


def context_size(context: str) -> dict:
    """The size of a context, given to the analyst: its sentences and its words (a longer context is a richer brief, not a shorter one)."""
    return {"sentences": max(1, len(re.findall(r"[.!?]+(?:\s|$)", context.strip()))), "words": se.count_words(context)}


def is_auto(params: dict) -> bool:
    """The length of the video is decided by the creation (the length agent), not fixed by whoever asks."""
    return params.get("length_mode") == "auto"


def context_sentences(params: dict, trend) -> int:
    """How many sentences the idea agent is asked for: more for a longer video (about one per 20 s), between the bounds of the trend. More information = a more stable, more precise generation."""
    low, high = getattr(trend, "IDEA_SENTENCES", (4, 12))
    return max(low, min(high, round(params.get("target_seconds", 150) / 20)))


def validate_idea(answer: dict, bones: dict, sentences: int = 4) -> list[str]:
    context = answer.get("context", "")
    problems = []
    minimum = max(3, sentences - 2)
    if len(re.findall(r"[.!?]", context)) < minimum or se.count_words(context) < 12 * minimum:
        problems.append(f"the context needs about {sentences} sentences (at least {minimum} sentences and {12 * minimum} words)")
    for role in ((bones["role_a"], bones["role_b"]) if bones.get("mode") != "invent" else ()):  # when the model invents the two characters it is free to name them as it likes
        if role.split()[-1].lower() not in context.lower():
            problems.append(f"the role '{role}' must be named in the context")
    if not answer.get("title", "").strip():
        problems.append("a short title is needed")
    return problems


def run_idea(params: dict, trend, llm) -> tuple[str, dict]:
    """FROM NOTHING: the context of the video is invented. Returns (context, report). Without a model the bones become a plain three-sentence context."""
    bones = draw_seed_elements(params, trend)
    world = bones.get("fallback_world") or (bones["setting"], bones["premise"])
    plain = f"{world[0].capitalize()}, {bones['hour']}. {world[1].capitalize()}. Two strangers each offer to save you: a {bones['role_a']} and a {bones['role_b']}."
    if llm is None:
        return plain, {"source": "template", "bones": bones}
    count = context_sentences(params, trend)
    sentences = f"{count} sentences (between {count - 1} and {count + 2})"
    if is_auto(params):  # no length was fixed: the story decides how big its context is, and the length agent reads it afterwards
        low, high = getattr(trend, "IDEA_AUTO_SENTENCES", (5, 14))
        count = low + 2
        sentences = f"as many sentences as THIS story needs, between {low} and {high} (a simple story needs few; a story with several places, reversals or rules of its own needs more)"
    if bones["mode"] == "invent":
        prompt = fill(trend.IDEA_INVENT_PROMPT, {"ingredients": " + ".join(bones["ingredients"]), "hour": bones["hour"], "tone": bones["tone"], "avoid": "; ".join(w[0] for w in trend.IDEA_WORLDS),
                                                 "sentences": sentences, "kind": bones["kind"], "genre": bones["genre"], "names": " and ".join(bones["names"])})
    else:
        prompt = fill(trend.IDEA_PROMPT, {"setting": bones["setting"], "premise": bones["premise"], "hour": bones["hour"], "role_a": bones["role_a"], "role_b": bones["role_b"], "tone": bones["tone"],
                                          "sentences": sentences})
    answer, problems, attempts = call_agent(llm, prompt, trend.IDEA_SCHEMA, lambda a: validate_idea(a, bones, count), seed=params["seed"])
    if answer is None:
        return plain, {"source": "template", "bones": bones, "attempts": attempts, "problems": problems[:3]}
    return answer["context"].strip(), {"source": "llm", "bones": bones, "attempts": attempts, "title": answer["title"], "invented": {"setting": answer.get("setting"), "premise": answer.get("premise")} if bones["mode"] == "invent" else None}


def validate_length(answer: dict, trend) -> list[str]:
    """What the length agent answered must be usable: a few key events, a number of seconds (out of the bounds it is clamped, not refused), a reason."""
    problems = []
    events = [e for e in answer.get("events", []) if isinstance(e, str) and e.strip()]
    if len(events) < 4:
        problems.append("list 6 to 20 key events of the story (at least 4)")
    if not isinstance(answer.get("seconds"), int) or answer["seconds"] <= 0:
        problems.append("seconds must be a whole number of seconds")
    if se.count_words(answer.get("why", "")) < 4:
        problems.append("say in one sentence why this length")
    return problems


def fallback_length(context: str, trend) -> int:
    """Without a model: the length follows the size of the context (a longer context tells a bigger story): the minimum for 4 sentences, 30 s more for each further sentence, clamped to the bounds."""
    low, high = trend.LENGTH_RANGE
    sentences = context_size(context)["sentences"]
    return int(max(low, min(high, round((low + 30 * max(0, sentences - 4)) / 10) * 10)))


def run_length(context: str, params: dict, trend, llm) -> tuple[int, dict]:
    """AUTO LENGTH: nobody fixed how long the video is, the creation decides it from the key events of the context, between the bounds of the trend. Returns (seconds, report)."""
    low, high = trend.LENGTH_RANGE
    per_shot = trend.SECONDS_PER_SHOT * params.get("shot_scale", 1.0)
    prompt = fill(trend.LENGTH_PROMPT, {"context": context.strip(), "shot_seconds": f"{per_shot:.1f}", "shots_2min": round(120 / per_shot), "shots_per_minute": round(60 / per_shot),
                                        "min_seconds": low, "max_seconds": high})
    answer, problems, attempts = (None, ["no model"], 0) if llm is None else call_agent(llm, prompt, trend.LENGTH_SCHEMA, lambda a: validate_length(a, trend), seed=params["seed"] + 3)
    if answer is None:
        return fallback_length(context, trend), {"source": "template", "attempts": attempts, "problems": problems[:3]}
    chosen = int(max(low, min(high, round(answer["seconds"] / 10) * 10)))
    return chosen, {"source": "llm", "attempts": attempts, "asked": answer["seconds"], "chosen": chosen, "events": answer["events"], "why": answer["why"]}


# ---------------------------------------------------------------------------------------------- stage 1: the analyst
def validate_brief(brief: dict, trend) -> list[str]:
    problems = []
    characters = {c.get("id"): c for c in brief.get("characters", [])}
    if set(characters) != {"c1", "c2"}:
        problems.append("exactly two characters with ids c1 and c2")
    for c in characters.values():
        if se.count_words(c.get("hidden_truth", "")) < 5:
            problems.append(f"{c.get('id')}: the hidden truth needs at least 5 words")
        if not c.get("role") or not c.get("name"):
            problems.append(f"{c.get('id')}: name and role are required")
    if len(brief.get("keywords", [])) < 6:
        problems.append("at least 6 keywords")
    locations = brief.get("locations", [])
    ids = [loc.get("id") for loc in locations]
    if len(locations) < 4 or len(set(ids)) != len(ids):
        problems.append("4 to 6 places with distinct ids")
    if brief.get("world", {}).get("hour") not in trend.HOURS:
        problems.append(f"hour must be one of {trend.HOURS}")
    if se.count_words(brief.get("world", {}).get("scale_image", "")) < 8:
        problems.append("scale_image: a grandiose picture of at least 8 words")
    return problems


def heuristic_brief(context: str, params: dict, trend) -> dict:
    """The analyst's fallback: the kit of places of the engine, two invented characters whose roles are guessed from 'a X and a Y', everything else generic. Enough to keep the pipeline alive."""
    kind, tags, rooms = se.kit_for(context)
    roles = re.findall(r"\ba ([a-z]+(?: [a-z]+)?)(?: and (?:a|an) ([a-z]+(?: [a-z]+)?))?[.,]?\s*$", context.strip().lower().split(":")[-1].strip())
    first, second = (roles[0] if roles and roles[0][0] else ("survivor", "stranger")) if roles else ("survivor", "stranger")
    rnd = random.Random(params["seed"])
    names = rnd.sample(se.FICTIONAL_NAMES, 2)
    hour = "night" if re.search(r"\bnight|midnight|moon", context.lower()) else "dusk" if re.search(r"dusk|evening|sunset", context.lower()) else "day"
    return {"title": se.slug_title(context), "substitutions": [],
            "world": {"setting": rooms[0][1], "premise": context.strip()[:200], "hour": hour, "atmosphere": "tense, dark, wet", "scale_image": f"an epic wide view of {rooms[0][1]} under a huge sky, awe-inspiring scale"},
            "locations": [{"id": rid, "description": desc, "tags": [t for t in tags if t in trend.LOCATION_TAGS]} for rid, desc, _ in rooms],
            "characters": [{"id": f"c{i + 1}", "name": names[i], "role": role, "gender": se.GENDERS[names[i]], "age": rnd.randint(30, 60), "look": "weathered face, direct stare", "wardrobe": "worn dark clothes",
                            "public_promise": "I can get you out of here", "hidden_truth": "is not telling you the whole truth about where they are going", "voice_style": "calm and urgent"} for i, role in enumerate((first, second))],
            "viewer": "you", "stakes": "your life", "keywords": [w for w in dict.fromkeys(re.findall(r"[a-z]{5,}", context.lower()))][:10] + ["survival", "apocalypse", "pov"], "must_include": []}


def run_analyst(context: str, params: dict, trend, llm) -> tuple[dict, dict]:
    size = context_size(context)
    n_locations = trend.locations_asked(params.get("target_seconds", 150), size["sentences"]) if hasattr(trend, "locations_asked") else "5 or 6" if size["sentences"] < 7 else "6 or 7" if size["sentences"] < 10 else "7 or 8"
    prompt = fill(trend.ANALYST_PROMPT, {"context": context.strip(), "context_size": f"{size['sentences']} sentences, {size['words']} words", "n_locations": n_locations, "language": {"en": "English", "fr": "French"}[params["language"]], "tone": params["tone"], "hours": trend.HOURS,
                                         "location_tags": trend.LOCATION_TAGS})
    brief, problems, attempts = (None, ["no model"], 0) if llm is None else call_agent(llm, prompt, trend.BRIEF_SCHEMA, lambda b: validate_brief(b, trend), seed=params["seed"])
    if brief is None:
        return heuristic_brief(context, params, trend), {"source": "template", "attempts": attempts, "problems": problems[:5]}
    return brief, {"source": "llm", "attempts": attempts}


# ---------------------------------------------------------------------------------------------- the LOOK agent: the art direction comes from the invented WORLD
def look_tags(brief: dict) -> list[str]:
    return sorted({t for loc in brief["locations"] for t in loc.get("tags", [])})


def look_world_text(brief: dict) -> str:
    world = brief["world"]
    places = "\n".join(f"- {loc['id']} (tags: {', '.join(loc.get('tags', [])) or 'none'}): {loc['description']}" for loc in brief["locations"])
    return f"setting: {world['setting']}\npremise: {world['premise']}\nhour: {world['hour']}\natmosphere: {world['atmosphere']}\nthe grandiose picture: {world['scale_image']}\nplaces:\n{places}"


def validate_look(style: dict, brief: dict) -> list[str]:
    """What makes an art direction unusable: an invalid axis, a motif that a picture model would PAINT as text (a number, a letter, a word: 'a stencilled number on the crates' put 912 on every door), a lighting
    that contradicts the hour of the world."""
    problems = ss.validate_style(style)
    if re.search(r"\d|\b(?:numbers?|numerals?|letters?|words?|text|signs?|signage|writing|inscriptions?|labels?|stencil\w*)\b", str(style.get("motif", "")), re.I):
        problems.append("the motif must be a SHAPE or an object (no number, letter, word, sign or text: a picture model paints them on the walls)")
    hour = brief["world"].get("hour")
    if style.get("lighting") in ss.WRONG_LIGHTING.get(hour, set()):
        problems.append(f"the lighting '{style.get('lighting')}' contradicts the hour of this world ({hour})")
    world_words = set(re.findall(r"[a-z]+", look_world_text(brief).lower()))
    for lighting, needs in ss.LIGHTING_NEEDS.items():  # a light needs a source in the world: tubes, neon, a fire (a sun-drenched creature was given fluorescent tubes)
        if style.get("lighting") == lighting and not world_words & needs:
            problems.append(f"the lighting '{lighting}' needs one of {sorted(needs)[:6]} in this world: choose the lighting of the sun, the sky or the fire this world really has")
    return problems


def run_look(brief: dict, params: dict, trend, llm) -> tuple[dict | None, dict]:
    """The art direction of the video (palette, materials, lighting, grade, motif, what each kind of place always shows), written by the model from the WORLD it invented. None (the rules of
    scripts/story_style.py are used instead) when there is no model or it never gives a valid one."""
    if llm is None or not hasattr(trend, "LOOK_PROMPT"):
        return None, {"source": "template", "attempts": 0, "problems": ["no model"]}
    tags = look_tags(brief)
    prompt = fill(trend.LOOK_PROMPT, {"hour": brief["world"].get("hour", "day"), "tags": ", ".join(tags) or "none", "world": look_world_text(brief)})
    answer, problems, attempts = call_agent(llm, prompt, ss.STYLE_SCHEMA, lambda s: validate_look(s, brief), seed=params["seed"] + 5, soft=lambda p: p.startswith("the lighting"))
    if answer is None:
        return None, {"source": "template", "attempts": attempts, "problems": problems[:3]}
    style = dict(answer)
    if any(p.startswith("the lighting") for p in validate_look(style, brief)):  # one enum the model keeps getting wrong must not cost the whole art direction (it fell back on a concrete bunker): the lighting of the hour
        words = set(re.findall(r"[a-z]+", look_world_text(brief).lower()))
        sunny = bool(words & {"sun", "sunlit", "sunlight", "golden", "sunny", "amber", "honey", "sun-drenched"})
        style["lighting"] = {"day": "golden hour backlight" if sunny else "overcast flat daylight", "night": "cold moonlit", "dusk": "dusk blue hour", "dawn": "overcast flat daylight"}.get(brief["world"].get("hour"), "overcast flat daylight")
    style["name"] = ss.slug(style["name"])
    forced = {k: v for k, v in (style.get("forced_elements") or {}).items() if k in tags}  # a key that is no tag of the story is never used
    if brief["world"].get("hour") in ss.HOUR_SENTENCES:
        forced["time"] = ss.HOUR_SENTENCES[brief["world"]["hour"]]  # the hour of the story, by code, in every picture
    style["forced_elements"], style["tags"] = forced, tags
    return style, {"source": "llm", "attempts": attempts}


# ---------------------------------------------------------------------------------------------- stage 2: the planner
def structure_text(counts: dict[str, int]) -> str:
    return "\n".join(f"- {act}: {n} beat{'s' if n > 1 else ''}" for act, n in counts.items())


def compact_brief(brief: dict, characters_only: bool = False) -> str:
    world = brief["world"]
    lines = [] if characters_only else [f"TITLE: {brief['title']}", f"WORLD: {world['setting']} | {world['premise']} | hour {world['hour']} | {world['atmosphere']}", f"YOU ARE: {brief['viewer']} | STAKES: {brief['stakes']}",
                                       "PLACES: " + "; ".join(f"{loc['id']} = {loc['description']}" for loc in brief["locations"]), f"KEYWORDS: {', '.join(brief['keywords'])}"]
    for c in brief["characters"]:
        lines.append(f"{c['id']} {c['name']} ({c['role']}, {c['gender']}, {c['age']}): {c['look']}; wears {c['wardrobe']}; promises: {c['public_promise']}; HIDDEN TRUTH: {c['hidden_truth']}; speaks: {c['voice_style']}")
    return "\n".join(lines)


def expected_branch(act: str) -> str:
    return "main" if act in ("hook", "setup", "offers", "choice") else ("A" if act == "branch_a" else "B")


def validate_outline(outline: dict, counts: dict[str, int], endings: dict, trend, needs: set[str] | None = None) -> list[str]:
    """`needs` = the header and footer fields this answer must carry (None = all of them): a part of a long outline is only asked for the fields that belong to it."""
    problems = []
    beats = outline.get("beats", [])
    if abs(len(beats) - sum(counts.values())) > max(6, round(0.08 * sum(counts.values()))):
        problems.append(f"{len(beats)} beats in total, {sum(counts.values())} are needed (a video of the right length)")
    order = [act for act in trend.ACT_PURPOSE if act in counts]
    sequence = [b.get("act") for b in beats]
    expected = [act for act in order for _ in range(counts[act])]
    for act in order:
        got = sequence.count(act)
        if abs(got - counts[act]) > (0 if counts[act] == 1 else max(3, round(0.12 * counts[act]))):  # the counts are a guide (the length is decided by the lines); the order and the kinds are what must be right
            problems.append(f"act {act}: {got} beats, {counts[act]} are needed")
    if [a for a in sequence if a] != sorted((a for a in sequence if a), key=lambda a: order.index(a) if a in order else 99):
        problems.append(f"the acts must come in this order: {order}")
    for i, beat in enumerate(beats):
        where = f"beat {i + 1}"
        if beat.get("branch") != expected_branch(beat.get("act", "")) and not (beat.get("act") == "rewind" and beat.get("branch") == "B"):
            problems.append(f"{where}: branch of act {beat.get('act')} must be {expected_branch(beat.get('act', ''))}")
        if (beat.get("kind") in ("talk", "offer")) != (beat.get("speaker") in ("c1", "c2")):
            problems.append(f"{where}: only a talk or offer beat is spoken by c1 or c2")
        if beat.get("kind") == "offer" and beat.get("act") != "offers":
            problems.append(f"{where}: the offer beats belong to the offers act")
        if se.count_words(beat.get("purpose", "")) < 4:
            problems.append(f"{where}: the purpose needs at least 4 words")
    if "offers" in counts:
        offers = [i for i, b in enumerate(beats) if b.get("act") == "offers"]
        pair = [i for i in offers if beats[i].get("kind") == "offer"]
        if len(pair) != 2 or pair[1] != pair[0] + 1 or [beats[i].get("speaker") for i in pair] != ["c1", "c2"]:
            problems.append("the offers act needs exactly two consecutive beats of kind offer, speaker c1 then c2 (both hold out a hand in ONE shot)")
        if len(offers) != 2:
            problems.append("the offers act is EXACTLY the two offer beats: the context and the clues belong to the setup act, nothing happens between the offers and the choice")
        if pair and pair[-1] + 1 < len(beats) and beats[pair[-1] + 1].get("kind") != "choice":
            problems.append("the choice beat comes immediately after the second offer")
    for act, own_speakers in (("branch_a", ("narrator", "c1")), ("branch_b", ("narrator", "c2"))):
        if act not in counts:
            continue
        own = [b for b in beats if b.get("act") == act]
        if any(b.get("speaker") not in own_speakers for b in own):
            problems.append(f"{act}: the other character does not speak or appear any more, only {own_speakers[1]} and the narrator")
        share = sum(1 for b in own if b.get("kind") == "pov") / max(1, len(own))
        if own and share < getattr(trend, "MIN_POV_SHARE", 0) - 0.08:
            problems.append(f"{act}: only {share:.0%} of its beats are first-person action (kind pov), at least {getattr(trend, 'MIN_POV_SHARE', 0):.0%} are needed")
    for act, kind in (("choice", "choice"), ("rewind", "rewind")):
        if act in counts and not any(b.get("act") == act and b.get("kind") == kind for b in beats):
            problems.append(f"the {act} act needs a beat of kind {kind}")
    for act in ("branch_a", "branch_b"):
        if act in counts:
            tail = [b for b in beats if b.get("act") == act][-2:]  # the twist is the last beat of the branch, or the one before an epilogue
            if not any(b.get("kind") == "twist" for b in tail):
                problems.append(f"a twist must close {act} (one of its last two beats)")
    if beats and "hook" in counts and beats[0].get("kind") not in ("narration", "pov"):
        problems.append("the first beat is a narration or pov hook")
    def wanted(name: str) -> bool:
        return needs is None or name in needs  # a part of a long outline is only asked for the fields that belong to it

    title = outline.get("hook_title", "").strip()
    if wanted("hook_title") and ("\n" not in title or not title.upper().startswith(("POV", "POINT DE VUE", "TON POV"))):
        problems.append("hook_title: two lines, the first starts with POV:")
    if wanted("closing_question") and "\n" not in outline.get("closing_question", "").strip():
        problems.append("closing_question: two lines")
    if wanted("caption") and outline.get("caption", "").count("#") < 5:
        problems.append("caption needs at least 5 hashtags")
    if (wanted("ending_label_a") and not outline.get("ending_label_a")) or (wanted("ending_label_b") and "branch_b" in counts and not outline.get("ending_label_b")):
        problems.append("the ending labels are required")
    return problems


def template_outline(brief: dict, counts: dict[str, int], endings: dict, params: dict, trend) -> dict:
    """The planner's fallback: every act gets generic beats made from the brief."""
    c1, c2 = brief["characters"][0], brief["characters"][1]
    beats = []
    kinds = {"hook": ["narration", "narration", "pov"], "setup": ["narration"], "offers": ["narration", "pov"], "choice": ["choice"], "rewind": ["rewind"],
             "branch_a": ["pov", "narration", "pov"], "branch_b": ["pov", "narration", "pov"]}
    for act, n in counts.items():
        for i in range(n):
            kind = "twist" if (act in ("branch_a", "branch_b") and i == n - 1) else "offer" if (act == "offers" and i < 2) else kinds[act][(i - 2 if act == "offers" else i) % len(kinds[act])]
            speaker = (c1["id"] if i % 2 == 0 else c2["id"]) if kind in ("talk", "offer") else "narrator"
            beats.append({"act": act, "branch": "B" if act == "rewind" else expected_branch(act), "kind": kind, "speaker": speaker,
                          "purpose": f"{act.replace('_', ' ')}: step {i + 1} of the story of {c1['name']} and {c2['name']} in {brief['world']['setting']}"})
    keywords = brief["keywords"]
    return {"hook_title": f"POV: {keywords[0].upper()}\n{keywords[1].upper()}", "beats": beats, "ending_label_a": "THE END" if endings["A"] == "bad" else "SAVED", "ending_label_b": "THE END" if endings["B"] == "bad" else "SAVED",
            "closing_question": "WHICH ENDING\nDID YOU GET?", "caption": f"{brief['world']['premise'][:120]} " + " ".join("#" + re.sub(r"\W", "", k) for k in keywords[:9])}


def outline_score(outline: dict, brief: dict) -> float:
    """A cheap score to choose between candidate outlines: distinct purposes, keywords of the brief used in the beats, a mix of kinds."""
    beats = outline["beats"]
    words = " ".join(b["purpose"].lower() for b in beats)
    distinct = len({b["purpose"].lower() for b in beats}) / max(1, len(beats))
    keyword_hits = sum(1 for k in brief["keywords"] if k.lower() in words) / max(1, len(brief["keywords"]))
    kinds = len({b["kind"] for b in beats}) / 6
    return round(distinct + keyword_hits + kinds, 3)


def plan_parts(counts: dict[str, int]) -> list[dict[str, int]]:
    """The acts of a long outline grouped into parts planned one after the other: up to the choice, the branch A, the rewind and the branch B."""
    groups = (("hook", "setup", "offers", "choice"), ("branch_a",), ("rewind", "branch_b"))
    parts = [{act: counts[act] for act in group if act in counts} for group in groups]
    return [part for part in parts if part]


def part_needs(part_counts: dict[str, int], last: bool) -> set[str]:
    """The header and footer fields of the outline that a part must carry: the title and caption come with the hook, an ending label with its branch, the closing question with the last part."""
    needs: set[str] = set()
    if "hook" in part_counts:
        needs |= {"hook_title", "caption"}
    if "branch_a" in part_counts:
        needs.add("ending_label_a")
    if "branch_b" in part_counts:
        needs.add("ending_label_b")
    if last:
        needs.add("closing_question")
    return needs


def planned_so_far(beats: list[dict]) -> str:
    if not beats:
        return "BEATS ALREADY PLANNED: (none, this part starts the video)"
    return "BEATS ALREADY PLANNED (continue the story from them, never repeat them):\n" + "\n".join(f"{n + 1}. [{b['act']}/{b['kind']}/{b['speaker']}] {' '.join(b['purpose'].split()[:22])}" for n, b in enumerate(beats))


def planner_prompt(brief: dict, counts: dict, endings: dict, params: dict, trend) -> str:
    return fill(trend.PLANNER_PROMPT, {"structure": structure_text(counts), "act_purposes": "\n".join(f"- {act}: {trend.ACT_PURPOSE[act]}" for act in counts), "max_talk": getattr(trend, "MAX_TALK_BEATS", 4),
                                       "mood_a": trend.ENDING_MOOD[endings["A"]], "mood_b": trend.ENDING_MOOD[endings["B"]] if "B" in endings else "(there is no branch B in this video)",
                                       "language": {"en": "English", "fr": "French"}[params["language"]], "brief": compact_brief(brief)})


def soft_outline_problem(problem: str) -> bool:
    """The counts of beats are a guide (the length is decided by the lines): an act with a few beats more or less than asked, or a total a little off, does not make an outline unusable; the order,
    the offers, the choice, the branches and the twists are what must be right."""
    return (problem.startswith("act ") and " beats, " in problem) or " beats in total, " in problem


def run_planner_in_parts(brief: dict, counts: dict, endings: dict, params: dict, trend, llm) -> tuple[dict, dict]:
    """A long video (more beats than ONE answer of the model can hold) is planned part by part: up to the choice, the branch A, the rewind and the branch B. Each part is shown the beats already planned and
    is checked on its own acts; a part the model never gets right falls back to the template beats for THAT part only."""
    parts = plan_parts(counts)
    fallback = template_outline(brief, counts, endings, params, trend)
    beats: list[dict] = []
    fields: dict[str, str] = {}
    report: dict = {"candidates": [], "parts": []}
    for number, part_counts in enumerate(parts, 1):
        needs = part_needs(part_counts, last=number == len(parts))
        prompt = planner_prompt(brief, part_counts, endings, params, trend) + "\n\n" + fill(trend.PLANNER_PART_NOTE, {"parts": len(parts), "part": number, "already": planned_so_far(beats)})
        outline, problems, attempts = call_agent(llm, prompt, trend.OUTLINE_SCHEMA, lambda o, c=part_counts, n=needs: validate_outline(o, c, endings, trend, n), seed=params["seed"] + 100 * number, soft=soft_outline_problem)
        if outline is None:
            part_beats, source = [b for b in fallback["beats"] if b["act"] in part_counts], "template"
        else:
            part_beats, source = outline["beats"], "llm"
            fields.update({name: outline.get(name, "") for name in needs})
        beats += part_beats
        report["parts"].append({"acts": list(part_counts), "beats": len(part_beats), "source": source, "attempts": attempts, "problems": problems[:3]})
    merged = {**{name: fallback[name] for name in ("hook_title", "ending_label_a", "ending_label_b", "closing_question", "caption")}, **{k: v for k, v in fields.items() if v}, "beats": beats}
    sources = {part["source"] for part in report["parts"]}
    return merged, {**report, "source": "llm" if sources == {"llm"} else "template" if sources == {"template"} else "partly template"}


def run_planner(brief: dict, counts: dict, endings: dict, params: dict, trend, llm, candidates: int = 1) -> tuple[dict, dict]:
    if llm is not None and sum(counts.values()) > getattr(trend, "PLANNER_SPLIT_BEATS", 10**6):  # a long video: one answer would be cut, the outline is planned part by part
        return run_planner_in_parts(brief, counts, endings, params, trend, llm)
    prompt = fill(trend.PLANNER_PROMPT, {"structure": structure_text(counts), "act_purposes": "\n".join(f"- {act}: {trend.ACT_PURPOSE[act]}" for act in counts),
                                        "max_talk": getattr(trend, "MAX_TALK_BEATS", 4), "mood_a": trend.ENDING_MOOD[endings["A"]], "mood_b": trend.ENDING_MOOD[endings["B"]] if "B" in endings else "(there is no branch B in this video)", "language": {"en": "English", "fr": "French"}[params["language"]], "brief": compact_brief(brief)})
    best, report = None, {"candidates": []}
    for k in range(candidates if llm else 0):
        outline, problems, attempts = call_agent(llm, prompt, trend.OUTLINE_SCHEMA, lambda o: validate_outline(o, counts, endings, trend), seed=params["seed"] + 100 * k, soft=soft_outline_problem)
        report["candidates"].append({"ok": outline is not None, "attempts": attempts, "problems": problems[:4], "score": outline_score(outline, brief) if outline else None})
        if outline and (best is None or outline_score(outline, brief) > outline_score(best, brief)):
            best = outline
    if best is None:
        return template_outline(brief, counts, endings, params, trend), {**report, "source": "template"}
    return best, {**report, "source": "llm"}


# ---------------------------------------------------------------------------------------------- stage 3: the writer
def chunks_of(beats: list[dict], max_size: int | None = None) -> list[list[int]]:
    """The beats are written in chunks that follow the story: the first acts up to the choice, the branch A, the rewind + the branch B. With `max_size` a longer chunk is cut into equal parts (a 10-minute
    story has about 45 beats in a branch: one list of lines that long loses its thread)."""
    groups: list[list[int]] = [[], [], []]
    for i, beat in enumerate(beats):
        act = beat["act"]
        groups[0 if act in ("hook", "setup", "offers", "choice") else 1 if act == "branch_a" else 2].append(i)
    chunks = [g for g in groups if g]
    if not max_size:
        return chunks
    out: list[list[int]] = []
    for chunk in chunks:
        pieces = -(-len(chunk) // max_size)
        size = -(-len(chunk) // pieces)
        out += [chunk[i:i + size] for i in range(0, len(chunk), size)]
    return out


def offer_rule(trend) -> str:
    """What the writer is told about the two offer beats, from the voice chosen for them."""
    if getattr(trend, "OFFER_VOICE", "narrator") == "narrator":
        return (f"kind offer = the NARRATOR tells the proposal of the character named in the beat (c1 or c2) in the third person, at most {getattr(trend, 'MAX_OFFER_WORDS', 9)} words, "
                "e.g. 'Brandt offers a dry harbor.'")
    return f"kind offer = the character SAYS his or her proposal in the first person, at most {getattr(trend, 'MAX_OFFER_WORDS', 9)} words."


def validate_independence(lines: list[dict], beats: list[dict], brief: dict) -> list[str]:
    """ONE character per branch: in branch A c2 is never named, in branch B c1 is never named. The rewind line may name the other one."""
    names = {c["id"]: c["name"] for c in brief["characters"]}
    problems = []
    for i, (line, beat) in enumerate(zip(lines, beats)):
        other = "c2" if beat.get("act") == "branch_a" else "c1" if beat.get("act") == "branch_b" else None
        if other and re.search(rf"\b{re.escape(names[other].lower())}\b", line.get("text", "").lower()):
            problems.append(f"line {i + 1}: this branch is only about you and {names['c1' if other == 'c2' else 'c2']}: {names[other]} must not be named")
    return problems


def validate_lines(answer: dict, beats: list[dict], location_ids: list[str], max_words: int, avg_words: float | None = None, offer_words: int | None = None) -> list[str]:
    lines = answer.get("lines", [])
    problems = []
    if len(lines) != len(beats):
        return [f"{len(lines)} lines but {len(beats)} beats: one line per beat, same order"]
    if avg_words:
        mean = sum(se.count_words(l.get("text", "")) for l in lines) / len(lines)
        if mean > 1.2 * avg_words:
            problems.append(f"the lines are too long on average ({mean:.1f} words): aim for about {avg_words:.0f} words per line, shorten most of them")
    seen = set()
    for i, (line, beat) in enumerate(zip(lines, beats)):
        words = se.count_words(line.get("text", ""))
        limit = offer_words if (offer_words and beat.get("kind") == "offer") else max_words
        if not 1 <= words <= limit:
            problems.append(f"line {i + 1}: {words} words, the limit is {limit}")
        if line.get("location") not in location_ids:
            problems.append(f"line {i + 1}: location must be one of {location_ids}")
        if line.get("text", "").strip().lower() in seen:
            problems.append(f"line {i + 1}: repeats an earlier line")
        seen.add(line.get("text", "").strip().lower())
    return problems


def verify_clarity(llm, trend, picked: list[dict], lines: list[dict], previous: str, seed: int = 0) -> list[str]:
    """The script doctor: a viewer who sees the lines once must understand them (nothing refers to what was never shown, no nonsense phrase). A verifier that cannot answer never blocks the story."""
    numbered = "\n".join(f"{n + 1}. {l.get('text', '')}" for n, l in enumerate(lines))
    try:
        verdict = llm(fill(trend.CLARITY_PROMPT, {"previous": previous, "lines": numbered}), trend.CLARITY_SCHEMA, seed=seed)
    except Exception:
        return []
    return [f"line {item['n']} is confusing for a viewer who sees it once ({item['why'][:120]}): rewrite it so that it is clear, introduce what it refers to" for item in verdict.get("confusing", [])
            if isinstance(item.get("n"), int) and 1 <= item["n"] <= len(lines)][:4]


def verify_premise(llm, trend, lines: list[dict], premise: str, seed: int = 0) -> list[str]:
    """The viewer hears the first lines knowing nothing: do they say WHAT is happening and WHY it is dangerous? (the stacked test said "Choose before stairs collapse" and never that a creature was waking.)
    A yes/no question to a second call, its answer is told to the writer; a verifier that cannot answer never blocks the story."""
    numbered = "\n".join(f"{n + 1}. {l.get('text', '')}" for n, l in enumerate(lines))
    try:
        verdict = llm(fill(trend.PREMISE_PROMPT, {"premise": premise, "lines": numbered}), trend.PREMISE_SCHEMA, seed=seed)
    except Exception:
        return []
    problems = []
    if not verdict.get("situation", True):
        problems.append(f"the first lines never say WHAT is happening ({str(verdict.get('missing', ''))[:140]}): say it in plain words in the first three lines")
    if not verdict.get("cause", True):
        problems.append(f"the first lines never say WHY it is dangerous ({str(verdict.get('missing', ''))[:140]}): tell the cause in plain words in the first three lines, before any detail of atmosphere")
    return problems


def soft_writer_problem(problem: str) -> bool:
    """The problems of a written chunk that do not make it unusable: lines a little long on average, a line one or two words over its limit, an ending that reads MIXED (a teaser question left open),
    an ending that closes everything."""
    over = re.match(r"line \d+: (\d+) words, the limit is (\d+)$", problem)
    if over and 0 < int(over.group(1)) - int(over.group(2)) <= 2:
        return True
    if re.match(r"line \d+: (location must be one of|repeats an earlier line)", problem):  # a place the model invented is replaced by code, a repeated line is a matter of taste
        return True
    return (problem.startswith(("the lines are too long on average", "the first lines never say", "this branch must end", "the last line closes everything")) or " is confusing for a viewer" in problem
            or " this branch is only about you and " in problem)  # the ending a verifier reads differently from the one drawn, the other character named: asked again, then kept (the QA report says it)


def verify_polarity(llm, trend, lines: list[dict], expected: str, seed: int = 0) -> list[str]:
    """Ask a small separate question: how do the last lines of this branch END for the viewer? A branch that must end good and reads bad (or the reverse) is a problem the writer is told about."""
    tail = "\n".join(f"- {l['text']}" for l in lines[-6:])
    try:
        verdict = llm(fill(trend.VERIFIER_PROMPT, {"lines": tail}), trend.VERIFIER_SCHEMA, seed=seed)
    except Exception:
        return []  # a verifier that cannot answer never blocks the story
    problems = []
    if verdict["polarity"] != expected:
        problems.append(f"this branch must end {expected.upper()} for you but its last lines read {verdict['polarity'].upper()} ({verdict['reason'][:140]}): rewrite the last lines so that they end {expected.upper()}")
    if not verdict.get("suspense", True):
        problems.append("the last line closes everything: end on ONE open question (an unexplained sound, door, name or detail) so that the viewer wants to know what comes next")
    return problems


def before_the_offers(lines: list[dict], beats: list[dict]) -> list[dict]:
    """The lines a viewer hears BEFORE the offers (the hook and the setup): where the premise must have been told."""
    cut = next((i for i, b in enumerate(beats) if b.get("kind") == "offer"), len(beats))
    return lines[:max(3, min(cut, 14))]


def run_writer(brief: dict, outline: dict, params: dict, trend, llm, library_examples: str, endings: dict | None = None) -> tuple[list[dict], dict]:
    beats = outline["beats"]
    average = params["target_seconds"] * se.WORDS_PER_SECOND[params["language"]] * 0.85 / max(1, len(beats))
    location_ids = [loc["id"] for loc in brief["locations"]]
    lines: list[dict | None] = [None] * len(beats)
    report = {"chunks": []}
    scale = params.get("shot_scale", 1.0)
    max_words = trend.max_words(scale) if hasattr(trend, "max_words") else trend.MAX_SPOKEN_WORDS
    offer_words = trend.max_offer_words(scale) if hasattr(trend, "max_offer_words") else getattr(trend, "MAX_OFFER_WORDS", None)
    last_index = {b["act"]: i for i, b in enumerate(beats)}  # the last beat of every act: a branch only ENDS in the chunk that holds its last beat
    for chunk in chunks_of(beats, getattr(trend, "WRITER_MAX_CHUNK", None)):
        picked = [beats[i] for i in chunk]
        acts = {b["act"] for b in picked}
        branch_letter = "A" if "branch_a" in acts else "B" if "branch_b" in acts else None
        closes_branch = branch_letter is not None and chunk[-1] == last_index["branch_a" if branch_letter == "A" else "branch_b"]
        expected = (endings or {}).get(branch_letter) if closes_branch else None
        mood = trend.ENDING_MOOD[expected] if expected else trend.OPEN_MOOD
        previous = " / ".join(l["text"] for l in lines[:chunk[0]][-5:] if l) or "(this is the start)"
        prompt = fill(trend.WRITER_PROMPT, {"language": {"en": "English", "fr": "French"}[params["language"]], "max_words": max_words, "must_include": "; ".join(brief["must_include"]) or "(none)",
                                           "location_ids": location_ids, "brief": compact_brief(brief), "previous_lines": previous, "count": len(picked), "examples": library_examples or "(none yet)",
                                           "offer_rule": offer_rule(trend), "premise_rule": getattr(trend, "PREMISE_FIRST" if chunk[0] == 0 else "PREMISE_NEXT", "").replace("<<premise>>", brief["world"].get("premise", "")), "avg_words": f"{average:.0f}", "total_words": int(params["target_seconds"] * se.WORDS_PER_SECOND[params["language"]] * 0.85), "ending_mood": mood,
                                           "beats": "\n".join(f"{n + 1}. [{b['kind']}, speaker {b['speaker']}] {b['purpose']}" for n, b in enumerate(picked))})
        def check(a, picked=picked, expected=expected, previous=previous, first=chunk[0] == 0):
            basic = validate_lines(a, picked, location_ids, max_words, average, offer_words)
            found = basic + validate_independence(a.get("lines", []), picked, brief)
            if any(not soft_writer_problem(p) for p in basic):  # the lines themselves are unusable (a wrong count, a line far over its limit): no use asking the verifiers
                return found
            if expected:
                found += verify_polarity(llm, trend, a.get("lines", []), expected, seed=params["seed"])
            if hasattr(trend, "CLARITY_PROMPT"):
                found += verify_clarity(llm, trend, picked, a.get("lines", []), previous, seed=params["seed"])
            if first and hasattr(trend, "PREMISE_PROMPT"):
                found += verify_premise(llm, trend, before_the_offers(a.get("lines", []), picked), brief["world"].get("premise", ""), seed=params["seed"])
            return found

        answer, problems, attempts = (None, ["no model"], 0) if llm is None else call_agent(
            llm, prompt, sized(trend.LINES_SCHEMA, "lines", len(picked)), check, seed=params["seed"] + 7 * chunk[0], soft=soft_writer_problem, attempts=5 if closes_branch else 3)
        from_model = answer is not None
        if answer is not None:
            for line in answer["lines"]:
                if line.get("location") not in location_ids:  # a place the model invented: the main set (the QA report does not count it as a problem)
                    line["location"] = location_ids[0]
        if answer is None:
            answer = {"lines": [{"text": " ".join(b["purpose"].split()[:max_words - 2]), "location": location_ids[0], "in_shot": []} for b in picked]}
        report["chunks"].append({"beats": len(picked), "source": "llm" if from_model else "template", "attempts": attempts,
                                 "problems": [p for p in problems if "location must be one of" not in p][:4]})  # a place the model invented was repaired by code: not a problem to report
        for i, line in zip(chunk, answer["lines"]):
            lines[i] = line
    return lines, report


# ---------------------------------------------------------------------------------------------- stage 4: the director
def viewer_words(text: str) -> str:
    """The viewer is never "the narrator" in a picture or a movement (the video model took it for a person): it is "you" / "the viewer"."""
    text = re.sub(r"\bthe narrator['\u2019]s\b", "your", text, flags=re.I)
    return re.sub(r"\bthe narrator\b", "the viewer", text, flags=re.I)


def soft_director_problem(problem: str) -> bool:
    return "NAME" in problem or "never write 'the narrator'" in problem


def validate_directions(answer: dict, picked: list[dict], trend, names: dict[str, str] | None = None) -> list[str]:
    items = answer.get("directions", [])
    if len(items) != len(picked):
        return [f"{len(items)} directions but {len(picked)} shots: one per shot, same order"]
    problems = []
    for i, (item, shot) in enumerate(zip(items, picked)):
        if item.get("n") != i + 1:  # a list shifted by one shot puts every picture on the line before it
            problems.append(f"direction {i + 1} says n={item.get('n')}: the directions come in the order of the shots and each one repeats the number of its shot")
        if se.count_words(item.get("motion", "")) < 4:  # every shot needs its acting or camera move, a talking shot too
            problems.append(f"shot {i + 1}: a motion of at least 4 words is needed (for a talk shot: how the character acts while speaking)")
        written = f"{item.get('still', '')} {item.get('motion', '')}".lower()
        if "narrator" in written:
            problems.append(f"direction {i + 1}: never write 'the narrator' in a picture or a movement: the viewer is 'you' (first-person view), 'your hands' or 'the camera'")
        if names and shot["kind"] != "offer":
            for cid in shot.get("in_shot", []):
                name = names.get(cid, "")
                where = item.get("motion", "") if shot["kind"] in ("talk", "choice") else written
                if name and not re.search(rf"\b{re.escape(name.lower())}\b", where.lower()):
                    problems.append(f"direction {i + 1}: {name} is in this shot, write {name}'s NAME and what {name} wears or does in the still and the motion (not he / she / the figure)")
        if shot["kind"] in ("talk", "choice"):
            continue
        words = se.count_words(item.get("still", ""))
        if not 14 <= words <= 60:
            problems.append(f"shot {i + 1}: the still has {words} words, 20 to 45 are needed")
    return problems


def clean_still(text: str, trend) -> str:
    """What a picture model must never be asked for (text, letters, a collage, panels) is taken out by code: even a negation ('not a collage') makes such a picture more likely, so the words go.
    A short clause that is only about it ('no text', 'split screen') is dropped whole; in a longer clause the word and its article are removed."""
    forbidden = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in trend.FORBIDDEN_IN_PICTURES) + r")s?\b", re.I)
    original = text
    text = re.sub(r"\b\d{1,2}\s*:\s*\d{1,2}\b", "", text)  # a ratio such as 9:16 was painted as text on crates and walls, and even as a phone clock
    text = re.sub(r"\b(?:vertical|portrait|tall)\s+(?:composition|format|framing|orientation|frame|aspect(?:\s+ratio)?)\b|\baspect ratio\b", "", text, flags=re.I)
    clauses = [c.strip() for c in text.split(",") if c.strip()]
    kept = []
    for clause in clauses:
        if forbidden.search(clause):
            if len(clause.split()) <= 4:
                continue
            clause = forbidden.sub("", clause)
            clause = re.sub(r"\b(?:not|no|without)\s+(?:a|an)?\s*(?=,|$)", "", clause)
        kept.append(re.sub(r"\s{2,}", " ", clause).strip())
    cleaned = ", ".join(c for c in kept if c)
    previous = None
    while previous != cleaned:  # no dangling "with a" / "and" left behind by a removed word
        previous = cleaned
        cleaned = re.sub(r"\b(?:with|of|and|or|a|an|the|on|in)\b\s*(?=,|\.|$)", "", cleaned)
        cleaned = re.sub(r"\b(?:with|of|and|or|a|an|the)\b\s+(?=(?:and|or|with|on|in|of)\b)", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", re.sub(r"\s+,", ",", cleaned)).strip(" ,")
    return cleaned or original


def run_director(brief: dict, shots: list[dict], params: dict, trend, llm, examples: str) -> tuple[list[dict], dict]:
    directions: list[dict | None] = [None] * len(shots)
    report = {"chunks": []}
    size = 12
    names_by_id = {c["id"]: c["name"] for c in brief["characters"]}
    for start in range(0, len(shots), size):
        picked = shots[start:start + size]
        prompt = fill(trend.DIRECTOR_PROMPT, {"hour": brief["world"]["hour"], "atmosphere": brief["world"]["atmosphere"], "camera_moves": ", ".join(trend.CAMERA_MOVES), "cameras": trend.CAMERAS,
                                             "scale_image": brief["world"]["scale_image"], "brief": compact_brief(brief), "count": len(picked), "examples": examples or "(none yet)",
                                             "shots": "\n".join(f"{n + 1}. [{s['kind']}, {s['branch']}, place {s['location']}, visible {', '.join(names_by_id.get(i, i) for i in s['in_shot']) or 'nobody'}] {s['text']}" for n, s in enumerate(picked))})
        answer, problems, attempts = (None, ["no model"], 0) if llm is None else call_agent(llm, prompt, sized(trend.DIRECTIONS_SCHEMA, "directions", len(picked)), lambda a: validate_directions(a, picked, trend, names_by_id), seed=params["seed"] + 13 * start, soft=soft_director_problem)
        from_model = answer is not None  # a kept answer with soft problems is still the model's
        if answer is None:
            locations = {loc["id"]: loc["description"] for loc in brief["locations"]}
            answer = {"directions": [{"still": "" if s["kind"] in ("talk", "choice") else f"{s['text']}, {locations.get(s['location'], '')}", "motion": f"{trend.CAMERA_MOVES[(start + n) % len(trend.CAMERA_MOVES)]}, slow, cinematic",
                                      "camera": s.get("camera_hint", "wide"), "fx": {"zoom": 0.05, "shake": 0.0, "flash": False}} for n, s in enumerate(picked)]}
        report["chunks"].append({"shots": len(picked), "source": "llm" if from_model else "template", "attempts": attempts, "problems": problems[:3]})
        for n, item in enumerate(answer["directions"]):
            item["still"] = clean_still(item.get("still", ""), trend)
            directions[start + n] = item
    return directions, report


STRONG_IMPACT_WORDS = {"crash", "crashes", "collapse", "collapses", "explodes", "explosion", "slams", "shatters", "crumbles", "erupts"}


def rhythm_pass(shots: list[dict], trend) -> dict:
    """Done by code, not by the model: clamp the effects, shake ONLY on a real impact (and at most one shot in six), never the same camera move three times in a row, a flash on every twist.
    Too much, too regular shaking was the first complaint about the videos: the default is calm."""
    changed = {"clamped": 0, "shake_added": 0, "shake_removed": 0, "camera_changed": 0, "flash_added": 0}
    for shot in shots:
        fx = shot.setdefault("fx", {})
        before = dict(fx)
        fx["zoom"] = min(0.08, max(0.03, float(fx.get("zoom", 0.05))))
        fx["shake"] = min(0.7, max(0.0, float(fx.get("shake", 0.0))))
        fx["flash"] = bool(fx.get("flash", False))
        if fx != before:
            changed["clamped"] += 1
        words = set(re.findall(r"[a-z]+", shot.get("text", "").lower())) | set(re.findall(r"[a-z]+", shot.get("motion", "").lower()))
        if words & STRONG_IMPACT_WORDS and fx["shake"] < 0.5 and shot["kind"] not in ("talk", "choice", "offer"):
            fx["shake"] = 0.6
            changed["shake_added"] += 1
        if shot["kind"] == "twist" and not fx["flash"]:
            fx["flash"] = True
            changed["flash_added"] += 1
    shaken = [s for s in shots if s["fx"]["shake"] > 0]
    allowed = max(1, len(shots) // 6)
    for shot in sorted(shaken, key=lambda s: s["fx"]["shake"])[: max(0, len(shaken) - allowed)]:  # keep only the strongest ones
        shot["fx"]["shake"] = 0.0
        changed["shake_removed"] += 1
    for i in range(2, len(shots)):
        if shots[i]["camera"] == shots[i - 1]["camera"] == shots[i - 2]["camera"] and shots[i]["kind"] not in ("talk", "choice", "offer"):
            shots[i]["camera"] = [c for c in trend.CAMERAS if c != shots[i]["camera"]][i % 3]
            changed["camera_changed"] += 1
    return changed


def shot_text(shot: dict) -> str:
    return (shot.get("text", "") + " " + shot.get("motion", "")).lower()


# ---------------------------------------------------------------------------------------------- assembly, judge, the whole chain
def offer_still(brief: dict, location: str) -> str:
    """The picture of the choice moment, the same for both offer shots: both characters side by side facing the camera, c1 on the left, each holding out an open hand. It is the scene the identity pass puts the two
    real faces into."""
    first, second = brief["characters"][0], brief["characters"][1]
    place = next((l["description"] for l in brief.get("locations", []) if l["id"] == location), "")

    def who(c: dict) -> str:
        return f"{'a woman' if c['gender'] == 'f' else 'a man'} ({' '.join(c['wardrobe'].rstrip('. ').split()[:9])})"
    return f"Two people stand side by side, close together, facing the camera: on the left {who(first)}, on the right {who(second)}; both hold out one open hand toward the camera, serious faces, {' '.join(place.rstrip('. ').split()[:22])}"


def infer_in_shot(shots: list[dict], brief: dict) -> int:
    """Who is in the picture, from what the shot SAYS: the writer often leaves `in_shot` empty for 'Kael steps forward' or 'close-up of Elara's face', and a shot with nobody listed gets no identity pass and is
    taken for a place. A character named in the line, the picture or the motion is in the shot; in a branch the other character (c2 in A, c1 in B) never is (one character per branch; the chosen one may be out of frame).
    Returns how many shots changed."""
    changed = 0
    for shot in shots:
        text = f"{shot.get('text', '')} {shot.get('still', '')} {shot.get('motion', '')}".lower()
        found = [c["id"] for c in brief["characters"][:2] if re.search(rf"\b{re.escape(c['name'].lower())}\b", text)]
        merged = [i for i in ("c1", "c2") if i in set(shot.get("in_shot", [])) | set(found)]
        other = {"A": "c2", "B": "c1"}.get(shot.get("branch")) if shot.get("kind") != "rewind" else None
        if other:
            merged = [i for i in merged if i != other]
        if shot.get("kind") in ("talk", "choice", "offer"):
            continue  # decided by their kind
        if merged != shot.get("in_shot", []):
            shot["in_shot"] = merged
            changed += 1
    return changed


def unify_offer_scene(shots: list[dict], brief: dict) -> None:
    """The two offers and the choice are ONE scene in ONE place: the place of the choice (the model gave each offer the place of what it promises). The two offer shots get the same two-shot picture."""
    choice = next((s for s in shots if s["kind"] == "choice"), None)
    if choice:
        for shot in shots:
            if shot["kind"] == "offer":
                shot["location"] = choice["location"]
                shot["still"] = offer_still(brief, choice["location"])


def assemble_story(brief: dict, outline: dict, lines: list[dict], directions: list[dict], endings: dict, trend) -> dict:
    c1, c2 = brief["characters"][0], brief["characters"][1]
    shots = []
    seen_branch: set[str] = set()
    for beat, line, direction in zip(outline["beats"], lines, directions):
        shot = {"kind": beat["kind"], "branch": beat["branch"], "speaker": beat["speaker"], "text": line["text"].strip(), "location": line["location"], "in_shot": [i for i in line.get("in_shot", []) if i in ("c1", "c2")],
                "still": "" if beat["kind"] in ("talk", "choice") else direction["still"].strip(), "motion": direction["motion"].strip(), "camera": direction["camera"], "fx": dict(direction["fx"])}
        shot["still"], shot["motion"] = viewer_words(shot["still"]), viewer_words(shot["motion"])
        shot["beat"] = {"act": beat["act"], "purpose": beat["purpose"]}  # what the planner wanted from this shot, kept in the plan (it was lost before)
        if ss.changes_the_hour(direction.get("time"), brief["world"].get("hour")):  # the same hour as the story is not a change of hour
            shot["time"] = direction["time"].strip()
        if beat["kind"] == "choice":
            shot["choice"] = {"a": c1["name"], "b": c2["name"]}
            shot["in_shot"] = ["c1", "c2"]  # a choice is always between the two of them, whatever the model listed
        if beat["kind"] == "offer":
            shot["in_shot"] = ["c1", "c2"]  # the ONLY moment where both are in the picture, both holding out a hand
            shot["offer_of"] = beat["speaker"]  # whose proposal it is
            shot["still"] = offer_still(brief, line["location"])  # the same two-shot picture for both offers, made by code
            if getattr(trend, "OFFER_VOICE", "narrator") == "narrator":
                shot["speaker"] = "narrator"  # the narrator quotes it: no mouth has to follow a voice
            if shots and shots[-1]["kind"] == "offer" and getattr(trend, "OFFER_OVERLAP", 0.0) > 0:
                shot["overlap"] = trend.OFFER_OVERLAP  # seconds the second line starts before the first one ends
        if beat["act"] in ("branch_a", "branch_b") and beat["act"] not in seen_branch:
            seen_branch.add(beat["act"])
            shot["tag"] = f"CASE {'A' if beat['act'] == 'branch_a' else 'B'}: {(c1 if beat['act'] == 'branch_a' else c2)['name'].upper()}"
        if beat["kind"] == "twist":
            letter = "A" if beat["act"] == "branch_a" else "B"
            shot["ending"] = endings[letter]
            shot["tag"] = f"ENDING {letter}: {outline['ending_label_' + letter.lower()].upper()}"
        shots.append(shot)
    infer_in_shot(shots, brief)
    unify_offer_scene(shots, brief)
    return {"title": brief["title"], "hook_title": outline["hook_title"].strip(), "logline": brief["world"]["premise"], "substitutions": brief.get("substitutions", []),
            "characters": [{k: c[k] for k in ("id", "name", "role", "gender", "age", "look", "wardrobe")} for c in (c1, c2)], "shots": shots, "end_card": outline["closing_question"].strip(),
            "end_card_small": f"A: {outline['ending_label_a'].upper()}  ·  B: {outline.get('ending_label_b', '').upper()}\nFOLLOW FOR PART 2".strip(), "caption": outline["caption"].strip()}


def story_text(plan: dict) -> str:
    return "\n".join(f"[{s['branch']}/{s['kind']}/{s['speaker']}] {s['text']} | {(s.get('still') or s.get('motion', ''))[:90]}" for s in plan["shots"])


def judge_plan(context: str, plan: dict, trend, llm) -> dict | None:
    prompt = fill(trend.JUDGE_PROMPT, {"context": context.strip(), "story": story_text(plan)})
    try:
        scores = llm(prompt, trend.JUDGE_SCHEMA, seed=1)
    except Exception:
        return None
    numeric = [scores[k] for k in ("hook", "coherence", "clues", "twist", "voice", "faithfulness", "variety")]
    return {**scores, "mean": round(sum(numeric) / len(numeric), 2)}


def make_plan(context: str | None = None, given: dict | None = None, seed: int | None = None, llm=None, trend_name: str = "you_must_choose", candidates: int = 1, judge: bool = False,
              outline_candidates: int = 1, voices: list[dict] | None = None, library: lib.StoryLibrary | None = None, exclude_examples: set[str] | None = None) -> dict:
    """The whole chain for one context. `llm(prompt, schema, seed=...) -> dict`; None = no model (every stage uses its fallback). With candidates > 1 the whole chain runs several times and the
    judge picks the best."""
    trend = load_trend(trend_name)
    given = dict(given or {})
    endings_given = given.pop("endings", None)
    params, provenance = se.resolve_params({"target_seconds": trend.DEFAULTS["target_seconds"], **given}, seed)
    endings = endings_given or sw.draw_endings(params["seed"])
    if params["branches"] < 2:
        endings = {"A": endings["A"]}
    provenance["endings"] = "given" if endings_given else "rng"
    library = library if library is not None else lib.StoryLibrary()
    idea_report = None
    if not (context or "").strip():  # NOTHING was given: the idea agent invents the subject from the seed
        context, idea_report = run_idea(params, trend, llm)
        provenance["context"] = "rng"
    else:
        provenance["context"] = "given"
    best = None
    unload = getattr(llm, "unload", None)
    for k in range(max(1, candidates)):
        started = time.time()
        run_params = {**params, "seed": params["seed"] + 1000 * k}
        length_report = None
        if is_auto(run_params):  # no fixed length: the creation decides it from the context, between the bounds of the trend
            run_params["target_seconds"], length_report = run_length(context, run_params, trend, llm)
        brief, report_a = run_analyst(context, run_params, trend, llm)
        style, report_look = run_look(brief, run_params, trend, llm)
        counts = trend.budget(run_params["target_seconds"], run_params["branches"], run_params.get("shot_scale", 1.0))
        outline, report_p = run_planner(brief, counts, endings, run_params, trend, llm, outline_candidates)
        examples = library.examples_text(brief, k=2, exclude=exclude_examples)
        lines, report_w = run_writer(brief, outline, run_params, trend, llm, examples, endings)
        provisional = [{"kind": b["kind"], "branch": b["branch"], "text": l["text"], "location": l["location"], "in_shot": l.get("in_shot", [])} for b, l in zip(outline["beats"], lines)]
        infer_in_shot(provisional, brief)  # the director is told who is in each shot, by name
        directions, report_d = run_director(brief, provisional, run_params, trend, llm, library.director_examples_text(brief, k=2, exclude=exclude_examples))
        story = assemble_story(brief, outline, lines, directions, endings, trend)
        rhythm = rhythm_pass(story["shots"], trend)
        location_ids = [loc["id"] for loc in brief["locations"]]
        problems = sw.validate_story(story, run_params, endings, location_ids)
        plan = finish_plan(story, brief, context, run_params, {**provenance, **({"target_seconds": "auto"} if length_report else {})}, endings, location_ids, voices)
        plan["idea"] = idea_report
        plan["style"] = style  # the art direction of THIS world (None = the pictures step uses the rules of story_style.py)
        plan["agents"] = {"trend": trend.NAME, "prompts": prompt_fingerprint(trend), "length": length_report, "analyst": report_a, "look": report_look, "planner": report_p, "writer": report_w, "director": report_d, "rhythm": rhythm,
                          "problems_left": problems, "seconds": round(time.time() - started, 1)}
        plan["story_source"] = "chain: " + ", ".join(f"{n}={plan['agents'][n]['source']}" for n in ("analyst", "planner")) + ", writer=" + ("llm" if all(c["source"] == "llm" for c in report_w["chunks"]) else "partly template") \
            + ", director=" + ("llm" if all(c["source"] == "llm" for c in report_d["chunks"]) else "partly template")
        if judge and llm is not None:
            plan["agents"]["judge"] = judge_plan(context, plan, trend, llm)
        key = (plan["agents"].get("judge") or {}).get("mean", 0) - len(problems)
        if best is None or key > best[0]:
            best = (key, plan)
    if unload:
        unload()
    return best[1]


def finish_plan(story: dict, brief: dict, context: str, params: dict, provenance: dict, endings: dict, location_ids: list[str], voices) -> dict:
    shots = []
    for index, shot in enumerate(story["shots"]):
        clean = {k: v for k, v in shot.items() if v not in (None, "", {}) or k in ("still", "text")}
        shots.append({"id": f"s{index + 1:03d}", "visual": shot["motion"], **clean, "fx": shot.get("fx") or {}, "choice": shot.get("choice") or None})
    locations = [{"id": loc["id"], "kind": "custom", "tags": [t for t in loc.get("tags", [])], "description": loc["description"], "variant_of": None} for loc in brief["locations"]]
    plan = {"title": story["title"], "logline": story["logline"], "language": params["language"], "substitutions": story.get("substitutions", []), "characters": story["characters"], "locations": locations,
            "beats": [], "context": context, "params": params, "provenance": provenance, "endings": endings, "shots": shots, "brief": brief,
            "title_overlay": {"text": story["hook_title"], "start": 0.15, "end": 3.8, "y": 0.15, "pixels": 62}, "end_card": story["end_card"], "end_card_small": story["end_card_small"], "caption": story["caption"],
            "estimated_seconds": sw.estimate_seconds(shots, params["language"]), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    plan["voices"] = se.assign_voices(plan, voices if voices is not None else se.load_voices(), params["seed"])
    return plan


def ollama(model: str = "qwen3.6:27b", host: str = "http://127.0.0.1:11434", temperature: float = 0.8):
    """llm(prompt, schema, seed) through Ollama's structured output; thinking off (a reasoning model would spend minutes before the JSON)."""
    import httpx

    def call(prompt: str, schema: dict, seed: int = 0) -> dict:
        response = httpx.post(host.rstrip("/") + "/api/chat", json={"model": model, "stream": False, "think": False, "format": schema, "keep_alive": "10m", "messages": [{"role": "user", "content": prompt}],
                                                                 "options": {"num_ctx": 12288, "temperature": temperature, "top_p": 0.95, "seed": seed, "num_predict": 7000}}, timeout=1800)
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])

    def unload() -> None:
        """Free the GPU for the image and video jobs: the model stays loaded between the calls of one story (a reload costs a minute) and is released at the end of it."""
        try:
            httpx.post(host.rstrip("/") + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        except Exception:
            pass
    call.unload = unload
    return call


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("context")
    parser.add_argument("--seconds", type=int, default=150)
    parser.add_argument("--shot-scale", type=float, help="TEST ONLY: multiplies the time a shot stays on screen (0.4 = about 1.5 s per shot: many more shots in the same minute); not used by default")
    parser.add_argument("--auto-length", action="store_true", help="the creation decides how long the video is (2 to 10 minutes, from the context); --seconds is then ignored")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--language", choices=sorted(se.WORDS_PER_SECOND))
    parser.add_argument("--branches", type=int, choices=[1, 2])
    parser.add_argument("--trend", default="you_must_choose")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--outline-candidates", type=int, default=1)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    given = {k: v for k, v in {"language": args.language, "target_seconds": args.seconds, "branches": args.branches, "shot_scale": args.shot_scale, "length_mode": "auto" if args.auto_length else None}.items() if v is not None}
    plan = make_plan(args.context, given, args.seed, ollama(args.model) if args.llm else None, args.trend, args.candidates, args.judge, args.outline_candidates)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{plan['title']} | {plan['story_source']} | {len(plan['shots'])} shots, about {plan['estimated_seconds']} s of speech | {plan['agents']['seconds']} s | problems left: {plan['agents']['problems_left'][:3]}")
    for shot in plan["shots"]:
        print(f"{shot['id']} {shot['branch']:4} {shot['kind']:9} {shot['speaker']:8} {shot['text'][:90]}")


if __name__ == "__main__":
    main()
