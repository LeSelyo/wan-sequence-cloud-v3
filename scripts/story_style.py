"""Style bibles for story videos ("you must choose" trend): ONE art direction per video, created automatically from the context (or provided), CLASSIFIED when it is created and kept in a
library on the PC (results/story_trend/style_library, never on the rented server) so that a good style can be selected again for another story and another location.

A style is regulated, not free text: every axis is an ENUM (lighting, lens, grade, texture) or a short bounded list (palette, materials, forbidden), plus a recurring motif and `forced_elements`
(what a kind of place must always show: e.g. every window of a spaceship shows the black void of space, never a landscape). The same style gives the same look to every location of a video:
`style_prompt(style, location_tags)` is appended to every image prompt, `negative_prompt(style)` to every negative.

How a style is made (best first):  LLM with a JSON schema (`style_via_llm`, validated, 2 repairs)  ->  rules from the context keywords (`style_from_context`, deterministic, no model needed).
Library entry = style + metadata: origin (auto | provided), method (prompt_only | lora), seed_policy (rng by default | fixed), quality (None until tested), tests, reusable (None until decided), uses.

    python scripts/story_style.py make "an apocalypse on the moon, a spaceship" --seed 3        # create (rules), print, add to the library
    python scripts/story_style.py list                                                          # what the library holds
    python scripts/story_style.py select "spaceship,void"                                        # pick a reusable good style for these tags
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_DIR = ROOT / "results" / "story_trend" / "style_library"

LIGHTING = ["low-key warm practical lights", "cold moonlit", "golden hour backlight", "overcast flat daylight", "neon night", "firelight and embers", "harsh fluorescent", "dusk blue hour"]
LENSES = ["35mm cinematic", "24mm wide handheld", "50mm shallow depth of field", "anamorphic 40mm", "documentary 28mm"]
GRADES = ["teal and amber", "desaturated bleach bypass", "warm bronze monochrome", "cold steel blue", "rich earth tones", "high contrast black and gold"]
TEXTURES = ["photoreal 3D render, fine film grain", "photoreal practical set, subtle grain", "gritty documentary realism", "polished high-budget cinema still"]
ALWAYS_NEGATIVE = ["cartoon", "anime", "illustration", "painting", "text", "watermark", "logo", "deformed hands", "extra fingers", "blurry face"]
# what a kind of place must always show, by keyword of the context (the style FORCES it; the decor must also stay consistent from shot to shot)
FORCED_BY_KIND = {
    "window": "the windows always show the real outside of this world (never a generic landscape)",
    "space": "every window and viewport shows the black void of space with stars and the planet or moon, never a landscape",
    "city": "the city beyond the windows keeps the same skyline and the same sky colour",
    "shelter": "no daylight comes in: only artificial or fire light, concrete and metal",
    "sea": "the horizon stays on the same side with the same water colour",
}
CONTEXT_RULES = [  # the FIRST rule with a matching whole word wins, so the specific ones (space, shelter, sea/flood) come before the general one (city). (keywords, lighting, lens, grade, texture, motif, tags)
    (("space", "moon", "spaceship", "planet", "alien", "orbit", "orbiting", "orbital", "spacecraft", "starship", "vacuum", "airlock", "satellite"), "low-key warm practical lights", "35mm cinematic", "warm bronze monochrome", "photoreal 3D render, fine film grain",
     "one small winged emblem on the uniforms and on a few doors", ["space", "window"]),
    (("bunker", "shelter", "underground", "siege", "war", "survivor"), "firelight and embers", "documentary 28mm", "desaturated bleach bypass", "gritty documentary realism",
     "a stencilled number on a few crates and doors", ["shelter"]),
    (("sea", "ship", "island", "storm", "ocean", "flood", "flooded", "boat", "river", "dam"), "overcast flat daylight", "24mm wide handheld", "cold steel blue", "photoreal practical set, subtle grain",
     "a recurring rope-knot symbol", ["sea", "window"]),
    (("city", "office", "politic", "news", "election", "street"), "dusk blue hour", "50mm shallow depth of field", "teal and amber", "photoreal practical set, subtle grain",
     "one recurring colour on a few signs and uniforms", ["city", "window"]),
]
TIME_RULES = [  # (whole words of the context, lighting, what the light must always be): the hour told by the story beats the lighting of the kind of place
    ({"night", "midnight", "nightfall", "moonlit"}, "cold moonlit", "it is night in every shot: dark sky, the only light comes from lamps, windows and spotlights, never daylight"),
    ({"dusk", "evening", "twilight", "sunset"}, "dusk blue hour", "it is dusk in every shot: deep blue sky with the last warm glow on the horizon, lamps coming on"),
    ({"dawn", "sunrise", "daybreak"}, "overcast flat daylight", "it is dawn in every shot: pale grey-pink sky, low cold light"),
]
DAYLIGHT_WORDS = {"dawn", "sunrise", "daybreak", "morning", "day", "daylight", "noon", "sunlight", "sunny", "midday"}
COLD_WORDS = {"freezing", "frozen", "frost", "icy", "ice", "cold", "frigid", "arctic", "hypothermia"}  # the atmosphere of the brief says it is cold: the grade follows it, not the keyword rule
COLD_PALETTES = [["steel blue", "ash grey", "sodium yellow", "black"], ["midnight blue", "silver", "crimson accent", "concrete grey"]]
MATERIALS_BY_KIND = {  # a station is metal and glass, not carved wood and dusty cloth: the materials come from the kind of world (the first tag of its rule)
    "space": ["scratched steel", "glass and chrome", "brushed bronze", "rough concrete", "worn leather"],
    "shelter": ["rough concrete", "scratched steel", "worn leather", "dusty fabric"],
    "sea": ["wet asphalt", "scratched steel", "worn leather", "rough concrete"],
}
FANTASY_WORDS = ("magic", "magical", "fairy", "elf", "elves", "dragon", "wizard", "witch", "spell", "enchanted", "kingdom", "castle", "goblin", "sorcerer", "spirit", "ghost", "haunted")
DEFAULT_RULE = ("cold moonlit", "35mm cinematic", "rich earth tones", "polished high-budget cinema still", "one recurring symbol repeated on props", ["window"])
PALETTES = [["bronze", "brass", "charcoal", "ember orange"], ["steel blue", "ash grey", "sodium yellow", "black"], ["deep teal", "amber", "rust", "off-white"],
            ["olive", "mud brown", "tungsten orange", "soot"], ["midnight blue", "silver", "crimson accent", "concrete grey"]]
MATERIALS = ["brushed bronze", "worn leather", "rough concrete", "scratched steel", "dusty fabric", "cracked stone", "wet asphalt", "carved wood", "glass and chrome"]

STYLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 40}, "logline": {"type": "string", "maxLength": 200},
        "palette": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
        "materials": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
        "lighting": {"type": "string", "enum": LIGHTING}, "lens": {"type": "string", "enum": LENSES}, "grade": {"type": "string", "enum": GRADES}, "texture": {"type": "string", "enum": TEXTURES},
        "motif": {"type": "string", "maxLength": 120},
        "forced_elements": {"type": "object", "additionalProperties": {"type": "string"}},
        "forbidden": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
    "required": ["name", "logline", "palette", "materials", "lighting", "lens", "grade", "texture", "motif", "forced_elements", "forbidden", "tags"],
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "style"


def validate_style(style: dict) -> list[str]:
    """Problems of a style (empty list = valid): every enum axis in its list, bounded lists, a motif, forced elements as short strings."""
    problems = []
    for key, allowed in (("lighting", LIGHTING), ("lens", LENSES), ("grade", GRADES), ("texture", TEXTURES)):
        if style.get(key) not in allowed:
            problems.append(f"{key} must be one of {allowed}")
    if not 3 <= len(style.get("palette", [])) <= 5:
        problems.append("palette needs 3 to 5 colours")
    if not 2 <= len(style.get("materials", [])) <= 5:
        problems.append("materials needs 2 to 5 entries")
    if not str(style.get("motif", "")).strip():
        problems.append("a recurring motif is required")
    if not isinstance(style.get("forced_elements"), dict) or any(not isinstance(v, str) or len(v) > 200 for v in style["forced_elements"].values()):
        problems.append("forced_elements must map a tag to a short sentence")
    if not style.get("name"):
        problems.append("name is required")
    return problems


def style_from_context(context: str, rng: random.Random, name: str | None = None, tags=(), hour: str | None = None, atmosphere: str = "") -> dict:
    """The deterministic style: the places of the story (their tags, the most frequent kind wins), the keywords of the context, the hour and the atmosphere of the brief choose the axes, the seed chooses
    palette and materials. Same input + same seed = same style. The TAGS of the locations come first: a story whose places are a deck, a corridor and an airlock is a spaceship even when its text
    only says "the last city lights on the Earth below" (the keyword "city" alone made it a city story: warm wood and bronze, dusk, then a sunlit stone arcade in a station)."""
    words = set(re.findall(r"[a-z]+", context.lower()))  # whole words: a substring test took "relationship" for a ship
    votes = {rule[6][0]: sum(1 for t in tags if t == rule[6][0]) for rule in CONTEXT_RULES}  # the first tag of a rule names its kind of world
    voted = max(CONTEXT_RULES, key=lambda r: votes[r[6][0]]) if any(votes.values()) else None
    rule = voted if voted and votes[voted[6][0]] > 0 else next((r for r in CONTEXT_RULES if words & set(r[0])), None)
    if rule is None and words & set(FANTASY_WORDS):  # a fantasy world with no other kind of place: warm light, a wide lens, earth tones, a rune as its motif (a photoreal RENDER of a fantastic world)
        rule = (FANTASY_WORDS, "golden hour backlight", "anamorphic 40mm", "rich earth tones", "polished high-budget cinema still", "one recurring rune carved on a few stones and doors", ["window"])
    lighting, lens, grade, texture, motif, rule_tags = rule[1:] if rule else DEFAULT_RULE
    kinds = {*rule_tags, *tags}
    forced = {tag: FORCED_BY_KIND[tag] for tag in sorted(kinds) if tag in FORCED_BY_KIND}  # every kind of place of the story keeps what it must show (the shelter has no daylight)
    hour_words = set(re.findall(r"[a-z]+", (hour or "").lower())) | words
    for time_words, hour_lighting, hour_sentence in TIME_RULES:
        if hour_words & time_words:
            if rule and rule[6][0] == "shelter":  # underground the hour changes nothing: the lamps and the fire keep the light of the place (a moonlit bunker is nonsense)
                forced["time"] = "there is no daylight and no sky in any shot: the only light comes from lamps, emergency lights, screens and fire"
            else:
                lighting, forced["time"] = hour_lighting, hour_sentence
            break
    palette = list(rng.choice(PALETTES))
    if set(re.findall(r"[a-z]+", atmosphere.lower())) & COLD_WORDS:
        grade, palette = "cold steel blue", list(rng.choice(COLD_PALETTES))
    primary = rule_tags[0] if rule else None
    return {"name": name or slug(f"{(rule[0][0] if rule else 'world')}-{rng.randrange(1000, 9999)}"), "logline": context.strip()[:200], "palette": palette,
            "materials": rng.sample(MATERIALS_BY_KIND.get(primary, MATERIALS), 3), "lighting": lighting, "lens": lens, "grade": grade, "texture": texture, "motif": motif, "forced_elements": forced,
            "forbidden": ["bright cheerful colours", "flat studio lighting"], "tags": sorted(set(rule_tags) | set(tags))}


def changes_the_hour(shot_time: str | None, story_hour: str | None = None) -> bool:
    """A shot's own `time` is a change of hour only when it is a daylight hour or an hour other than the one of the story ("night" on a shot of a night story changes nothing)."""
    if not shot_time:
        return False
    said = set(re.findall(r"[a-z]+", shot_time.lower()))
    return bool(said & DAYLIGHT_WORDS) or not said & set(re.findall(r"[a-z]+", (story_hour or "").lower()))


def shot_hour_style(style: dict, shot_time: str | None, story_hour: str | None = None) -> dict:
    """The style of ONE shot that tells its own hour (a dawn at the end of a night story). Warm light only when that hour IS a daylight one: the director also wrote "night" on shots of a night story,
    and the old code turned every `time` into "warm natural light", which put the sun into a station at night. The same hour as the story changes nothing; a dusk changes the sentence only."""
    if not changes_the_hour(shot_time, story_hour):
        return style
    forced = {**style.get("forced_elements", {}), "time": shot_time}
    if set(re.findall(r"[a-z]+", shot_time.lower())) & DAYLIGHT_WORDS:
        return {**style, "lighting": "warm natural light", "forced_elements": forced}
    return {**style, "forced_elements": forced}


def style_via_llm(context: str, llm_json, seed: int = 0, attempts: int = 3) -> dict | None:
    """Ask a language model for a style: it must answer with the JSON schema, every axis an enum, then the answer is VALIDATED and the model is told what is wrong (2 repairs).
    `llm_json(prompt, schema) -> dict`. Returns None when the model never produces a valid style (the caller falls back to style_from_context)."""
    prompt = ("You are a film production designer. Create ONE complete art direction for a short cinematic story of ANY genre (fantasy and supernatural included). All characters and places are fictional.\n"
              f"Story context: {context}\n"
              "Choose lighting, lens, grade and texture ONLY from the allowed lists of the schema. The motif is one recurring emblem or object shape that appears on props and walls. "
              "forced_elements maps a kind of place (e.g. window, space, shelter) to what it must ALWAYS show, so every shot of that place looks the same world (for a spaceship: the windows always "
              "show the black void and stars). Keep it concrete, filmable, and RENDERED as a believable photographic image, whatever the genre. No cartoon.")
    last: list[str] = []
    for _ in range(attempts):
        try:
            style = llm_json(prompt + (f"\nYour previous answer had these problems, fix them: {last}" if last else ""), STYLE_SCHEMA)
        except Exception:
            return None
        last = validate_style(style)
        if not last:
            style["name"] = slug(style["name"])
            return style
    return None


def style_prompt(style: dict, location_tags=()) -> str:
    """The text appended to EVERY image prompt of the video: the same art direction in every shot, plus what the kind of place must always show."""
    forced = [style["forced_elements"][t] for t in (*location_tags, "time") if t in style.get("forced_elements", {})]  # "time" applies to every shot
    parts = [style["texture"], style["lens"], style["lighting"], f"{style['grade']} colour grade", "palette of " + ", ".join(style["palette"]), "materials: " + ", ".join(style["materials"]),
             "recurring motif: " + style["motif"], *forced]
    return ", ".join(parts)


def negative_prompt(style: dict) -> str:
    return ", ".join([*style.get("forbidden", []), *ALWAYS_NEGATIVE])


# ------------------------------------------------------------------ measuring a style on its test pictures
def colour_signature(path: Path) -> np.ndarray:
    """A small signature of a picture: 12-bin saturation-weighted hue histogram + mean and spread of the luminance."""
    image = Image.open(path).convert("RGB").resize((96, 96))
    hsv = np.asarray(image.convert("HSV")).astype(float) / 255.0
    hist, _ = np.histogram(hsv[..., 0], bins=12, range=(0, 1), weights=hsv[..., 1])
    hist = hist / max(hist.sum(), 1e-6)
    lum = np.asarray(image.convert("L")).astype(float) / 255.0
    return np.concatenate([hist, [lum.mean(), lum.std()]])


def palette_consistency(paths: list[Path]) -> float:
    """0..1: how alike the colour signatures of the test pictures are (1 = the same look). A style that keeps its look across different locations scores high."""
    if len(paths) < 2:
        return 1.0
    sigs = [colour_signature(p) for p in paths]
    distances = [0.5 * np.abs(a[:12] - b[:12]).sum() + 0.5 * np.abs(a[12:] - b[12:]).sum() for i, a in enumerate(sigs) for b in sigs[i + 1:]]
    return float(max(0.0, 1.0 - np.mean(distances)))


# ------------------------------------------------------------------ the library
class StyleLibrary:
    """results/story_trend/style_library/index.json (+ one JSON per style). Lives on the PC: it survives the closing of any rented server."""

    def __init__(self, root: Path = LIBRARY_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def _read(self) -> dict:
        return json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else {}

    def _write(self, index: dict) -> None:
        temp = self.index_path.with_suffix(".tmp")
        temp.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.index_path)

    def all(self) -> dict[str, dict]:
        return self._read()

    def add(self, style: dict, *, origin: str = "auto", context: str = "", method: str = "prompt_only", lora: dict | None = None, seed_policy: str = "rng", seed: int | None = None) -> str:
        problems = validate_style(style)
        if problems:
            raise ValueError(f"invalid style: {problems}")
        digest = hashlib.sha1(json.dumps(style, sort_keys=True).encode()).hexdigest()[:6]
        style_id = f"{style['name']}-{digest}"
        index = self._read()
        if style_id in index:
            return style_id
        index[style_id] = {"id": style_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "origin": origin, "method": method, "lora": lora, "seed_policy": seed_policy, "seed": seed,
                           "quality": None, "reusable": None, "tests": [], "uses": [], "tags": style.get("tags", []), "context": context, "style": style}
        self._write(index)
        return style_id

    def get(self, style_id: str) -> dict:
        return self._read()[style_id]

    def record_test(self, style_id: str, *, location_kind: str, images: list[str], consistency: float | None = None, verdict: str | None = None, note: str = "") -> dict:
        """A test = pictures of the style on one KIND of place. Quality and reusability are decided from the tests (see classify)."""
        index = self._read()
        entry = index[style_id]
        entry["tests"].append({"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "location_kind": location_kind, "images": images, "consistency": consistency, "verdict": verdict, "note": note})
        entry["quality"], entry["reusable"] = classify(entry)
        index[style_id] = entry
        self._write(index)
        return entry

    def record_use(self, style_id: str, video: str, locations: list[str]) -> None:
        index = self._read()
        index[style_id]["uses"].append({"video": video, "locations": locations, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        self._write(index)

    def set_verdict(self, style_id: str, verdict: str, note: str = "") -> dict:
        """The user's word ("good" / "bad") is the strongest test."""
        return self.record_test(style_id, location_kind="(user)", images=[], verdict=verdict, note=note)

    def select(self, tags=(), rng: random.Random | None = None, min_quality: float = 0.6) -> str | None:
        """A reusable style whose quality is good enough and whose tags overlap the wanted ones (any when no tags), chosen at random weighted by quality and by being less used."""
        rng = rng or random.Random()
        pool = [(sid, e) for sid, e in self._read().items() if e.get("reusable") and (e.get("quality") or 0) >= min_quality and (not tags or set(tags) & set(e.get("tags", [])))]
        if not pool:
            return None
        weights = [(e["quality"] ** 2) / (1 + len(e["uses"])) for _, e in pool]
        return rng.choices([sid for sid, _ in pool], weights=weights)[0]


def classify(entry: dict) -> tuple[float | None, bool | None]:
    """quality = mean of the measured consistencies and the user's verdicts (good 1.0, ok 0.6, bad 0.0); a bad verdict from the user makes the style non reusable whatever the numbers say.
    reusable = quality >= 0.7 AND tested on at least 2 kinds of place (a style must work 'across locations'), None while there is not enough evidence."""
    scores, kinds = [], set()
    for test in entry["tests"]:
        if test.get("consistency") is not None:
            scores.append(test["consistency"])
            kinds.add(test["location_kind"])
        if test.get("verdict") in ("good", "ok", "bad"):
            scores.append({"good": 1.0, "ok": 0.6, "bad": 0.0}[test["verdict"]])
    if not scores:
        return None, None
    quality = round(float(np.mean(scores)), 3)
    if any(t.get("verdict") == "bad" for t in entry["tests"]):
        return quality, False
    if quality >= 0.7 and len(kinds) >= 2:
        return quality, True
    return quality, None if len(kinds) < 2 else False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make")
    m.add_argument("context")
    m.add_argument("--seed", type=int, default=0)
    sub.add_parser("list")
    s = sub.add_parser("select")
    s.add_argument("tags", nargs="?", default="")
    args = parser.parse_args()
    library = StyleLibrary()
    if args.command == "make":
        style = style_from_context(args.context, random.Random(args.seed))
        style_id = library.add(style, origin="auto", context=args.context, seed=args.seed)
        print(style_id)
        print(json.dumps(style, indent=1, ensure_ascii=False))
    elif args.command == "list":
        for sid, e in library.all().items():
            print(f"{sid:36} {e['origin']:8} {e['method']:12} quality={e['quality']} reusable={e['reusable']} tests={len(e['tests'])} uses={len(e['uses'])} tags={e['tags']}")
    else:
        print(library.select([t for t in args.tags.split(",") if t]))


if __name__ == "__main__":
    main()
