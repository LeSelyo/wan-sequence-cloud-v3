"""The character bible AS OBSERVED: what the reference close-up of a character REALLY shows, written by the vision model, and used everywhere instead of the free text of the brief.

Why: the brief says "milky white eyes" or "scarred knuckles", the picture model draws a face without either, and a later prompt that repeats the free text makes the edit model give the character pale eyes or a
scar on the cheek that the reference never had (what the user saw between two scenes). The close-up is the source of truth: its traits are looked at ONCE, after it is chosen, and every identity prompt,
every check of "is it the same person" and the quality report use THAT description.

    describe(ask, image, character)  -> {"eyes", "hair", "facial_hair", "scars_or_marks", "skin", "head_cover"}      (ask = still_judge.ollama_vision)
    traits_text(traits)              -> "dark brown eyes, short grey hair, grey stubble, olive skin, a beige hood"   (positive facts only: a negation makes the thing more likely to be drawn)
    save(cards_dir, {id: traits})    -> cards.json: characters[id]["traits"] and ["traits_text"]
"""
from __future__ import annotations

import json
from pathlib import Path

FIELDS = ("eyes", "hair", "facial_hair", "scars_or_marks", "skin", "head_cover")
NOTHING = {"", "none", "no", "not visible", "n/a", "no scar", "no marks", "nothing", "unknown", "not applicable", "no hood"}
SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in FIELDS}, "required": list(FIELDS)}
PROMPT = ("You look at ONE reference portrait of a fictional character made for a short story ({name}, {role}). Describe ONLY what you can SEE on the face and the head, in a few plain words each, so that another "
          "artist can draw exactly the same person again:\n"
          "- eyes: the colour of the eyes (and anything unusual you see in them)\n- hair: colour, length, style\n- facial_hair: a beard, a moustache, stubble, or \"clean-shaven\"\n"
          "- scars_or_marks: a scar, a tattoo or a mark ON THE FACE, and where; the word \"none\" when there is none\n- skin: the skin tone\n- head_cover: a hood, a scarf, a hat or a helmet, or \"none\"\n"
          "Each answer is at most five words. Never guess: write \"not visible\" for what you cannot see.")


def describe(ask, image: Path, character: dict, seed: int = 0) -> dict:
    """The traits of the close-up, as the vision model sees them. A model that cannot answer gives {} (the pipeline goes on without)."""
    try:
        answer = ask(PROMPT.format(name=character.get("name", "the character"), role=character.get("role", "")), SCHEMA, image, seed)
    except Exception:
        return {}
    return {k: str(answer.get(k, "")).strip() for k in FIELDS}


def traits_text(traits: dict | None) -> str:
    """Positive facts only, in a fixed order: 'dark brown eyes, short grey hair, grey stubble, a scar across the left cheek, olive skin, a beige hood'."""
    if not traits:
        return ""
    parts = []
    for key in FIELDS:
        value = " ".join(str(traits.get(key, "")).strip().rstrip(".").split()[:7])  # a short phrase: the model is asked for five words and sometimes writes a sentence
        if value.lower() in NOTHING:
            continue
        suffix = {"eyes": "eyes", "hair": "hair", "skin": "skin"}.get(key)
        parts.append(f"{value} {suffix}" if suffix and suffix not in value.lower() and not (key == "eyes" and "eye" in value.lower()) else value)
    return ", ".join(parts)


def save(cards_dir: Path, traits: dict[str, dict]) -> None:
    path = Path(cards_dir) / "cards.json"
    cards = json.loads(path.read_text(encoding="utf-8"))
    for character_id, found in traits.items():
        if found and character_id in cards["characters"]:
            cards["characters"][character_id]["traits"] = found
            cards["characters"][character_id]["traits_text"] = traits_text(found)
    path.write_text(json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8")


def of(cards: dict, character_id: str) -> str:
    """The observed traits of a character, '' when they were never looked at."""
    return ((cards.get("characters") or {}).get(character_id) or {}).get("traits_text", "")
