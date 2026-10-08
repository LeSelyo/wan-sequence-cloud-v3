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
    return {"version": trend.PROMPT_VERSION, **{n.lower().replace("_prompt", ""): constant_hash(getattr(trend, n)) for n in ("ANALYST_PROMPT", "PLANNER_PROMPT", "WRITER_PROMPT", "DIRECTOR_PROMPT", "JUDGE_PROMPT")}}


# ---------------------------------------------------------------------------------------------- one agent: ask, check, repair
def call_agent(llm, prompt: str, schema: dict, check, attempts: int = 3, seed: int = 0) -> tuple[dict | None, list[str], int]:
    """Ask the model; check the answer with code; tell the model what is wrong and ask again. Returns (answer or None, the problems of the last attempt, attempts used)."""
    problems: list[str] = []
    for attempt in range(1, attempts + 1):
        text = prompt + (f"\n\nYour previous answer had these problems. Answer again, the whole JSON, and fix them: {problems[:10]}" if problems else "")
        try:
            answer = llm(text, schema, seed=seed + attempt)
        except Exception as error:  # the model is down, too slow or answered outside the schema
            return None, [f"model error: {str(error)[:200]}"], attempt
        problems = check(answer)
        if not problems:
            return answer, [], attempt
    return None, problems, attempts


# ---------------------------------------------------------------------------------------------- stage 0: the IDEA agent (a context out of nothing)
def draw_seed_elements(params: dict, trend) -> dict:
    """The bones of a story drawn by the seed from the pools of the trend: a place and what happens to it, an hour, two roles, a tone. The same seed always draws the same bones."""
    rnd = random.Random(params["seed"] * 31 + 5)
    setting, premise, hour = rnd.choice(trend.IDEA_WORLDS)
    role_a, role_b = rnd.choice(trend.IDEA_ROLES)
    if rnd.random() < 0.5:
        role_a, role_b = role_b, role_a
    return {"setting": setting, "premise": premise, "hour": hour, "role_a": role_a, "role_b": role_b, "tone": rnd.choice(trend.IDEA_TONES)}


def validate_idea(answer: dict, bones: dict) -> list[str]:
    context = answer.get("context", "")
    problems = []
    if len(re.findall(r"[.!?]", context)) < 3 or se.count_words(context) < 30:
        problems.append("the context needs three or four sentences (at least 30 words)")
    for role in (bones["role_a"], bones["role_b"]):
        if role.split()[-1].lower() not in context.lower():
            problems.append(f"the role '{role}' must be named in the context")
    if not answer.get("title", "").strip():
        problems.append("a short title is needed")
    return problems


def run_idea(params: dict, trend, llm) -> tuple[str, dict]:
    """FROM NOTHING: the context of the video is invented. Returns (context, report). Without a model the bones become a plain three-sentence context."""
    bones = draw_seed_elements(params, trend)
    plain = f"{bones['setting'].capitalize()}, {bones['hour']}. {bones['premise'].capitalize()}. Two strangers each offer to save you: a {bones['role_a']} and a {bones['role_b']}."
    if llm is None:
        return plain, {"source": "template", "bones": bones}
    prompt = fill(trend.IDEA_PROMPT, {"setting": bones["setting"], "premise": bones["premise"], "hour": bones["hour"], "role_a": bones["role_a"], "role_b": bones["role_b"], "tone": bones["tone"]})
    answer, problems, attempts = call_agent(llm, prompt, trend.IDEA_SCHEMA, lambda a: validate_idea(a, bones), seed=params["seed"])
    if answer is None:
        return plain, {"source": "template", "bones": bones, "attempts": attempts, "problems": problems[:3]}
    return answer["context"].strip(), {"source": "llm", "bones": bones, "attempts": attempts, "title": answer["title"]}


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
    prompt = fill(trend.ANALYST_PROMPT, {"context": context.strip(), "language": {"en": "English", "fr": "French"}[params["language"]], "tone": params["tone"], "hours": trend.HOURS,
                                         "location_tags": trend.LOCATION_TAGS})
    brief, problems, attempts = (None, ["no model"], 0) if llm is None else call_agent(llm, prompt, trend.BRIEF_SCHEMA, lambda b: validate_brief(b, trend), seed=params["seed"])
    if brief is None:
        return heuristic_brief(context, params, trend), {"source": "template", "attempts": attempts, "problems": problems[:5]}
    return brief, {"source": "llm", "attempts": attempts}


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


def validate_outline(outline: dict, counts: dict[str, int], endings: dict, trend) -> list[str]:
    problems = []
    beats = outline.get("beats", [])
    if abs(len(beats) - sum(counts.values())) > 6:
        problems.append(f"{len(beats)} beats in total, {sum(counts.values())} are needed (a video of the right length)")
    order = [act for act in trend.ACT_PURPOSE if act in counts]
    sequence = [b.get("act") for b in beats]
    expected = [act for act in order for _ in range(counts[act])]
    for act in order:
        got = sequence.count(act)
        if abs(got - counts[act]) > (0 if counts[act] == 1 else 3):  # the counts are a guide (the length is decided by the lines); the order and the kinds are what must be right
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
        if any(beats[i].get("kind") == "talk" for i in offers):
            problems.append("no talk beat in the offers act: c1 and c2 speak only in the two offer beats")
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
    if beats and beats[0].get("kind") not in ("narration", "pov"):
        problems.append("the first beat is a narration or pov hook")
    title = outline.get("hook_title", "").strip()
    if "\n" not in title or not title.upper().startswith(("POV", "POINT DE VUE", "TON POV")):
        problems.append("hook_title: two lines, the first starts with POV:")
    if "\n" not in outline.get("closing_question", "").strip():
        problems.append("closing_question: two lines")
    if outline.get("caption", "").count("#") < 5:
        problems.append("caption needs at least 5 hashtags")
    if not outline.get("ending_label_a") or ("branch_b" in counts and not outline.get("ending_label_b")):
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


def run_planner(brief: dict, counts: dict, endings: dict, params: dict, trend, llm, candidates: int = 1) -> tuple[dict, dict]:
    prompt = fill(trend.PLANNER_PROMPT, {"structure": structure_text(counts), "act_purposes": "\n".join(f"- {act}: {trend.ACT_PURPOSE[act]}" for act in counts),
                                        "max_talk": getattr(trend, "MAX_TALK_BEATS", 4), "mood_a": trend.ENDING_MOOD[endings["A"]], "mood_b": trend.ENDING_MOOD[endings["B"]] if "B" in endings else "(there is no branch B in this video)", "language": {"en": "English", "fr": "French"}[params["language"]], "brief": compact_brief(brief)})
    best, report = None, {"candidates": []}
    for k in range(candidates if llm else 0):
        outline, problems, attempts = call_agent(llm, prompt, trend.OUTLINE_SCHEMA, lambda o: validate_outline(o, counts, endings, trend), seed=params["seed"] + 100 * k)
        report["candidates"].append({"ok": outline is not None, "attempts": attempts, "problems": problems[:4], "score": outline_score(outline, brief) if outline else None})
        if outline and (best is None or outline_score(outline, brief) > outline_score(best, brief)):
            best = outline
    if best is None:
        return template_outline(brief, counts, endings, params, trend), {**report, "source": "template"}
    return best, {**report, "source": "llm"}


# ---------------------------------------------------------------------------------------------- stage 3: the writer
def chunks_of(beats: list[dict]) -> list[list[int]]:
    """The beats are written in chunks that follow the story: the first acts up to the choice, the branch A, the rewind + the branch B."""
    groups: list[list[int]] = [[], [], []]
    for i, beat in enumerate(beats):
        act = beat["act"]
        groups[0 if act in ("hook", "setup", "offers", "choice") else 1 if act == "branch_a" else 2].append(i)
    return [g for g in groups if g]


def offer_rule(trend) -> str:
    """What the writer is told about the two offer beats, from the voice chosen for them."""
    if getattr(trend, "OFFER_VOICE", "narrator") == "narrator":
        return (f"kind offer = the NARRATOR tells the proposal of the character named in the beat (c1 or c2) in the third person, at most {getattr(trend, 'MAX_OFFER_WORDS', 9)} words, "
                "e.g. 'Brandt offers a dry harbor.'")
    return f"kind offer = the character SAYS his or her proposal in the first person, at most {getattr(trend, 'MAX_OFFER_WORDS', 9)} words."


def validate_independence(lines: list[dict], beats: list[dict], brief: dict) -> list[str]:
    """After the choice the two possibilities are no longer linked: in branch A c2 is never named (and never in the picture), in branch B c1 is never named. The rewind line may name the other one."""
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
        if mean > 1.3 * avg_words:
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


def run_writer(brief: dict, outline: dict, params: dict, trend, llm, library_examples: str, endings: dict | None = None) -> tuple[list[dict], dict]:
    beats = outline["beats"]
    average = params["target_seconds"] * se.WORDS_PER_SECOND[params["language"]] * 0.85 / max(1, len(beats))
    location_ids = [loc["id"] for loc in brief["locations"]]
    lines: list[dict | None] = [None] * len(beats)
    report = {"chunks": []}
    for chunk in chunks_of(beats):
        picked = [beats[i] for i in chunk]
        acts = {b["act"] for b in picked}
        branch_letter = "A" if "branch_a" in acts else "B" if "branch_b" in acts else None
        expected = (endings or {}).get(branch_letter) if branch_letter else None
        mood = trend.ENDING_MOOD[expected] if expected else trend.OPEN_MOOD
        previous = " / ".join(l["text"] for l in lines[:chunk[0]][-5:] if l) or "(this is the start)"
        prompt = fill(trend.WRITER_PROMPT, {"language": {"en": "English", "fr": "French"}[params["language"]], "max_words": trend.MAX_SPOKEN_WORDS, "must_include": "; ".join(brief["must_include"]) or "(none)",
                                           "location_ids": location_ids, "brief": compact_brief(brief), "previous_lines": previous, "count": len(picked), "examples": library_examples or "(none yet)",
                                           "offer_rule": offer_rule(trend), "avg_words": f"{average:.0f}", "total_words": int(params["target_seconds"] * se.WORDS_PER_SECOND[params["language"]] * 0.85), "ending_mood": mood,
                                           "beats": "\n".join(f"{n + 1}. [{b['kind']}, speaker {b['speaker']}] {b['purpose']}" for n, b in enumerate(picked))})
        answer, problems, attempts = (None, ["no model"], 0) if llm is None else call_agent(
            llm, prompt, trend.LINES_SCHEMA,
            lambda a: validate_lines(a, picked, location_ids, trend.MAX_SPOKEN_WORDS, average, getattr(trend, "MAX_OFFER_WORDS", None)) + validate_independence(a.get("lines", []), picked, brief)
            + (verify_polarity(llm, trend, a.get("lines", []), expected, seed=params["seed"]) if expected and not validate_lines(a, picked, location_ids, trend.MAX_SPOKEN_WORDS) else []),
            seed=params["seed"] + 7 * chunk[0])
        if answer is None:
            answer = {"lines": [{"text": " ".join(b["purpose"].split()[:trend.MAX_SPOKEN_WORDS - 2]), "location": location_ids[0], "in_shot": []} for b in picked]}
        report["chunks"].append({"beats": len(picked), "source": "llm" if llm and not problems else "template", "attempts": attempts, "problems": problems[:3]})
        for i, line in zip(chunk, answer["lines"]):
            lines[i] = line
    return lines, report


# ---------------------------------------------------------------------------------------------- stage 4: the director
def validate_directions(answer: dict, picked: list[dict], trend) -> list[str]:
    items = answer.get("directions", [])
    if len(items) != len(picked):
        return [f"{len(items)} directions but {len(picked)} shots: one per shot, same order"]
    problems = []
    for i, (item, shot) in enumerate(zip(items, picked)):
        if se.count_words(item.get("motion", "")) < 4:  # every shot needs its acting or camera move, a talking shot too
            problems.append(f"shot {i + 1}: a motion of at least 4 words is needed (for a talk shot: how the character acts while speaking)")
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
    return cleaned or text


def run_director(brief: dict, shots: list[dict], params: dict, trend, llm, examples: str) -> tuple[list[dict], dict]:
    directions: list[dict | None] = [None] * len(shots)
    report = {"chunks": []}
    size = 12
    for start in range(0, len(shots), size):
        picked = shots[start:start + size]
        prompt = fill(trend.DIRECTOR_PROMPT, {"hour": brief["world"]["hour"], "atmosphere": brief["world"]["atmosphere"], "camera_moves": ", ".join(trend.CAMERA_MOVES), "cameras": trend.CAMERAS,
                                             "scale_image": brief["world"]["scale_image"], "brief": compact_brief(brief), "count": len(picked), "examples": examples or "(none yet)",
                                             "shots": "\n".join(f"{n + 1}. [{s['kind']}, {s['branch']}, place {s['location']}, visible {','.join(s['in_shot']) or 'nobody'}] {s['text']}" for n, s in enumerate(picked))})
        answer, problems, attempts = (None, ["no model"], 0) if llm is None else call_agent(llm, prompt, trend.DIRECTIONS_SCHEMA, lambda a: validate_directions(a, picked, trend), seed=params["seed"] + 13 * start)
        if answer is None:
            locations = {loc["id"]: loc["description"] for loc in brief["locations"]}
            answer = {"directions": [{"still": "" if s["kind"] in ("talk", "choice") else f"{s['text']}, {locations.get(s['location'], '')}", "motion": f"{trend.CAMERA_MOVES[(start + n) % len(trend.CAMERA_MOVES)]}, rain, handheld",
                                      "camera": s.get("camera_hint", "wide"), "fx": {"zoom": 0.05, "shake": 0.0, "flash": False}} for n, s in enumerate(picked)]}
        report["chunks"].append({"shots": len(picked), "source": "llm" if llm and not problems else "template", "attempts": attempts, "problems": problems[:3]})
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
def assemble_story(brief: dict, outline: dict, lines: list[dict], directions: list[dict], endings: dict, trend) -> dict:
    c1, c2 = brief["characters"][0], brief["characters"][1]
    shots = []
    seen_branch: set[str] = set()
    for beat, line, direction in zip(outline["beats"], lines, directions):
        shot = {"kind": beat["kind"], "branch": beat["branch"], "speaker": beat["speaker"], "text": line["text"].strip(), "location": line["location"], "in_shot": [i for i in line.get("in_shot", []) if i in ("c1", "c2")],
                "still": "" if beat["kind"] in ("talk", "choice") else direction["still"].strip(), "motion": direction["motion"].strip(), "camera": direction["camera"], "fx": dict(direction["fx"])}
        if ss.changes_the_hour(direction.get("time"), brief["world"].get("hour")):  # the same hour as the story is not a change of hour
            shot["time"] = direction["time"].strip()
        if beat["kind"] == "choice":
            shot["choice"] = {"a": c1["name"], "b": c2["name"]}
            shot["in_shot"] = ["c1", "c2"]  # a choice is always between the two of them, whatever the model listed
        if beat["kind"] == "offer":
            shot["in_shot"] = ["c1", "c2"]  # the ONLY moment where both are in the picture, both holding out a hand
            shot["offer_of"] = beat["speaker"]  # whose proposal it is
            if getattr(trend, "OFFER_VOICE", "narrator") == "narrator":
                shot["speaker"] = "narrator"  # the narrator quotes it: no mouth has to follow a voice
            if shots and shots[-1]["kind"] == "offer" and getattr(trend, "OFFER_OVERLAP", 0.0) > 0:
                shot["overlap"] = trend.OFFER_OVERLAP  # seconds the second line starts before the first one ends
        if beat["act"] in ("branch_a", "branch_b"):
            gone = "c2" if beat["act"] == "branch_a" else "c1"  # after the choice the other character is never in a picture
            shot["in_shot"] = [i for i in shot["in_shot"] if i != gone]
        if beat["act"] in ("branch_a", "branch_b") and beat["act"] not in seen_branch:
            seen_branch.add(beat["act"])
            shot["tag"] = f"CASE {'A' if beat['act'] == 'branch_a' else 'B'}: {(c1 if beat['act'] == 'branch_a' else c2)['name'].upper()}"
        if beat["kind"] == "twist":
            letter = "A" if beat["act"] == "branch_a" else "B"
            shot["ending"] = endings[letter]
            shot["tag"] = f"ENDING {letter}: {outline['ending_label_' + letter.lower()].upper()}"
        shots.append(shot)
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
        brief, report_a = run_analyst(context, run_params, trend, llm)
        counts = trend.budget(params["target_seconds"], params["branches"])
        outline, report_p = run_planner(brief, counts, endings, run_params, trend, llm, outline_candidates)
        examples = library.examples_text(brief, k=2, exclude=exclude_examples)
        lines, report_w = run_writer(brief, outline, run_params, trend, llm, examples, endings)
        provisional = [{"kind": b["kind"], "branch": b["branch"], "text": l["text"], "location": l["location"], "in_shot": l.get("in_shot", [])} for b, l in zip(outline["beats"], lines)]
        directions, report_d = run_director(brief, provisional, run_params, trend, llm, library.director_examples_text(brief, k=2, exclude=exclude_examples))
        story = assemble_story(brief, outline, lines, directions, endings, trend)
        rhythm = rhythm_pass(story["shots"], trend)
        location_ids = [loc["id"] for loc in brief["locations"]]
        problems = sw.validate_story(story, run_params, endings, location_ids)
        plan = finish_plan(story, brief, context, run_params, provenance, endings, location_ids, voices)
        plan["idea"] = idea_report
        plan["agents"] = {"trend": trend.NAME, "prompts": prompt_fingerprint(trend), "analyst": report_a, "planner": report_p, "writer": report_w, "director": report_d, "rhythm": rhythm,
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
    given = {k: v for k, v in {"language": args.language, "target_seconds": args.seconds, "branches": args.branches}.items() if v is not None}
    plan = make_plan(args.context, given, args.seed, ollama(args.model) if args.llm else None, args.trend, args.candidates, args.judge, args.outline_candidates)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{plan['title']} | {plan['story_source']} | {len(plan['shots'])} shots, about {plan['estimated_seconds']} s of speech | {plan['agents']['seconds']} s | problems left: {plan['agents']['problems_left'][:3]}")
    for shot in plan["shots"]:
        print(f"{shot['id']} {shot['branch']:4} {shot['kind']:9} {shot['speaker']:8} {shot['text'][:90]}")


if __name__ == "__main__":
    main()
