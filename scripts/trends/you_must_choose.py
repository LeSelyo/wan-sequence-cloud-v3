"""The TREND "you must choose": everything that is specific to it lives HERE (the constant prompts, the structure recipe, the schemas of what each agent answers, the render preset), so another trend is another
file next to this one and the chain (scripts/story_agents.py) does not change.

A video of this trend: a hook with a GRANDIOSE picture and the words TikTok reads, a setup, two characters who offer help and each hide a clue, THE CHOICE (A / B card with a countdown), branch A, a rewind,
branch B (one branch ends BAD for you, the other GOOD, each with a twist), a closing question.

THE CONSTANT PROMPTS. Each agent has ONE prompt text that never changes (the same words for every video of the trend: that is what keeps the quality stable). Only the SLOTS `{{name}}` are filled, from
the BRIEF that the analyst agent extracts from the context. `PROMPT_VERSION` changes when a constant text changes, and the report of a story records it.
"""
from __future__ import annotations

NAME = "you_must_choose"
PROMPT_VERSION = "2026-10-08.2"
KINDS = ["narration", "talk", "pov", "choice", "twist", "rewind"]
CAMERAS = ["wide", "medium", "close", "pov"]
LOCATION_TAGS = ["space", "sea", "city", "shelter", "window", "forest", "desert", "ice", "underground"]  # the tags that make the style prompt force what a kind of place must always show
HOURS = ["night", "dusk", "dawn", "day"]
MAX_SPOKEN_WORDS = 14
SECONDS_PER_SHOT = 3.1
SHARES = {"hook": 0.07, "setup": 0.09, "offers": 0.20, "branch_a": 0.27, "branch_b": 0.30}  # share of the shots of each act (two branches); one branch gets branch_a + branch_b
CAMERA_MOVES = ("aerial flight forward", "whip pan", "low angle tilt up", "handheld run", "slow push-in", "crane up", "dolly forward", "orbit around the subject", "lateral glide", "fast zoom-in", "tilt down")
FORBIDDEN_IN_PICTURES = ("text", "letters", "caption", "subtitle", "watermark", "logo", "collage", "split screen", "panels", "triptych", "diptych")

DEFAULTS = {"language": "en", "target_seconds": 150, "branches": 2, "tone": "tense", "aspect": "9:16", "fps": 30}

RENDER_PRESET = {"size": [720, 1280], "fps": 30, "subtitle": "one word at a time, glitch entrance (4 frames) then clean", "choice_ui": "orange A/B cards + countdown ring", "tags": ["CASE A", "CASE B", "ENDING A", "ENDING B"],
                 "sound": "rain, water, drone, heartbeat on the choice, riser+impact on each twist, reverse swell on the rewind, bright pad on a good ending"}


def budget(seconds: float, branches: int) -> dict[str, int]:
    """How many shots each act has for a video of `seconds` (about one shot every 3.1 s)."""
    total = max(12, round(seconds / SECONDS_PER_SHOT))
    two = branches >= 2
    shares = dict(SHARES) if two else {"hook": 0.09, "setup": 0.11, "offers": 0.28, "branch_a": 0.52, "branch_b": 0.0}
    counts = {act: max(3, round(total * share)) for act, share in shares.items() if share > 0}
    counts["choice"] = 1
    if two:
        counts["rewind"] = 1
    return counts


ACT_PURPOSE = {
    "hook": "GRANDIOSE first picture (epic scale, awe) that puts the viewer in the world at once; the spoken lines contain the key words of the story; the third shot puts YOU in danger.",
    "setup": "help arrives; the two main characters come into view.",
    "offers": "c1 and c2 each speak SHORT lines (kind 'talk'), accuse each other, and each leaves one CLUE (one narration shot shows a clue).",
    "choice": "the narrator says it is time to choose (kind 'choice').",
    "branch_a": "you follow c1; the clues come true; ends with a twist (kind 'twist').",
    "rewind": "the story rewinds: the narrator asks what if you had chosen c2 (kind 'rewind').",
    "branch_b": "you follow c2; FAST action shots (chase, collapse, escape) and at least two talk lines; ends with a twist (kind 'twist').",
}

# ---------------------------------------------------------------------------------------------- the constant prompts
ANALYST_PROMPT = """[[STAGE:analyst]]
You are the ANALYST of a short-video studio. The trend is "you must choose": a second-person interactive story ("you") in which two characters offer to save you, the viewer chooses, and each choice
leads to a different ending with a twist. Read the CONTEXT below (three sentences or more) and extract the BRIEF the other writers will work from.

RULES
- Keep EVERY concrete fact of the context (places, roles, objects, numbers, the hour, the weather). Invent only what is missing, and make it fit.
- Exactly two main characters, ids "c1" and "c2": the two people named or described in the context (their roles come from it); if the context names a real public figure, invent an archetype with a made-up
  name instead and list the replacement in "substitutions". Give each a name, a gender (m or f), an age, a face description, a wardrobe, what they PUBLICLY promise the viewer, and a HIDDEN TRUTH
  (what they are really doing or hiding) that a clue can reveal later.
- "locations": 5 or 6 places of this world, id = short lowercase word, the first one is the main set; tags only from {{location_tags}}.
- "scale_image": ONE grandiose picture of this world (epic scale) that will open the video.
- "keywords": 8 to 12 words or short phrases that people search for about this story (the disaster, the place, the roles, the feeling), lowercase.
- "hour" is one of {{hours}}; if the context gives none, choose the most dramatic one.

LANGUAGE of the text fields: {{language}}. TONE: {{tone}}.

CONTEXT:
{{context}}

Answer with the JSON brief only."""

PLANNER_PROMPT = """[[STAGE:planner]]
You are the PLANNER of a short-video studio, trend "you must choose". From the BRIEF, plan the whole video as a list of BEATS (one beat = one shot), in this order and with exactly these counts:
{{structure}}

WHAT EACH ACT MUST DO
{{act_purposes}}

RULES
- The clues: c1's hidden truth and c2's hidden truth must each be hinted at in the OFFERS act (one clue each) and CONFIRMED in their branch (branch A confirms c1's, branch B confirms c2's).
- Branch A ends {{ending_a}} for you, branch B ends {{ending_b}} for you; each last beat is a twist that RECASTS what you believed, with a short UPPERCASE ending label ("COLLECTED", "THE CURE"...).
- Every beat has: act, branch (main for hook/setup/offers/choice, A or B after the choice; the rewind has branch B), kind, speaker (narrator, c1 or c2; only kind talk is spoken by c1 or c2), and "purpose" = ONE sentence
  saying what happens and what the picture shows.
- The first beat is a hook (kind narration or pov). Make the hook's spoken idea contain the biggest keyword of the brief. Vary the beats: no two consecutive beats with the same purpose.
- "hook_title": exactly two UPPERCASE lines separated by a newline, the first starts with "POV:", both contain keywords of the brief. "closing_question": two short UPPERCASE lines. "caption": one sentence + 8 to 10
  hashtags built from the keywords.
LANGUAGE of hook_title, closing_question, caption, ending labels: {{language}}.

BRIEF:
{{brief}}

Answer with the JSON outline only."""

WRITER_PROMPT = """[[STAGE:writer]]
You are the WRITER of a short-video studio, trend "you must choose". Write the SPOKEN LINES of the beats below, in order, one line per beat, in {{language}}.

RULES
- At most {{max_words}} words per line, short punchy sentences, natural speech, no stage directions, no quotation marks. Second person for the narrator ("you").
- kind talk = the character SAYS it to you in the scene (c1 and c2 do not sound alike: use their voice_style and their public promise). The other kinds are the narrator.
- Respect each beat's purpose, keep the story continuous with the previous lines, never repeat a line or an opening word three times in a row.
- Facts that must appear somewhere: {{must_include}}.
- "location" of each beat is one of {{location_ids}}; "in_shot" = the character ids visible in it.

APPROVED STORIES (examples of STYLE only, other plots; do not copy their words):
{{examples}}

BRIEF (compact):
{{brief}}

PREVIOUS LINES (for continuity): {{previous_lines}}

BEATS TO WRITE ({{count}}):
{{beats}}

Answer with the JSON list of lines only, same order, same count."""

DIRECTOR_PROMPT = """[[STAGE:director]]
You are the DIRECTOR of a short-video studio, trend "you must choose": vertical 9:16, realistic dark cinematic look, ONE art direction for the whole video (hour: {{hour}}; atmosphere: {{atmosphere}}).
For each shot below (its spoken line is given), write the PICTURE and the CAMERA MOVE.

RULES
- "still" = ONE concrete filmable photograph (what is in the frame, the angle, the light), 20 to 45 words, in English, a single continuous scene. NEVER ask for text, letters, captions, logos, a collage, panels or a
  split screen. Characters: use their look and wardrobe from the brief. For kind talk and choice leave "still" empty (the portrait of the character is used).
- "motion" = the camera move and what moves in the picture, 10 to 25 words, English. Use a DIFFERENT camera move from the neighbouring shots, from this list or similar: {{camera_moves}}. At least one third of
  the shots must be dynamic (fast flight, chase, whip pan, impact).
- "fx": zoom 0.03 to 0.08; shake 0 to 1 (above 0.7 only on impacts, chases, crashes); flash true only on a shock.
- "camera" in {{cameras}}. "time": only when the hour of the picture differs from "{{hour}}" (e.g. the dawn of a good ending), English, e.g. "it is dawn: golden light".
- The grandiose picture of the world, for the first shot: {{scale_image}}.

APPROVED EXAMPLES of pictures and camera moves (other plots, copy only the level of detail):
{{examples}}

BRIEF (compact):
{{brief}}

SHOTS ({{count}}):
{{shots}}

Answer with the JSON list of directions only, same order, same count."""

JUDGE_PROMPT = """[[STAGE:judge]]
You are a strict EDITOR of TikTok storytelling videos, trend "you must choose". Score the STORY below from 1 (poor) to 10 (excellent) on each criterion, and name the single biggest weakness in one sentence.
Criteria: hook (a grandiose first image and a spoken line that make you stay), coherence (the story makes sense shot to shot), clues (each hidden truth is hinted then paid off), twist (each ending recasts what
you believed), voice (the lines sound like spoken, distinct characters), faithfulness (it keeps every fact of the context), variety (the shots differ: places, camera, rhythm).

CONTEXT:
{{context}}

STORY (branch, kind, speaker: spoken line | picture):
{{story}}

Answer with the JSON scores only."""

# ---------------------------------------------------------------------------------------------- the schemas of what each agent answers
CHARACTER_SCHEMA = {"type": "object", "properties": {"id": {"type": "string", "enum": ["c1", "c2"]}, "name": {"type": "string"}, "role": {"type": "string"}, "gender": {"type": "string", "enum": ["m", "f"]},
                                                      "age": {"type": "integer"}, "look": {"type": "string"}, "wardrobe": {"type": "string"}, "public_promise": {"type": "string"},
                                                      "hidden_truth": {"type": "string"}, "voice_style": {"type": "string"}},
                    "required": ["id", "name", "role", "gender", "age", "look", "wardrobe", "public_promise", "hidden_truth", "voice_style"]}
BRIEF_SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"}, "substitutions": {"type": "array", "items": {"type": "string"}},
    "world": {"type": "object", "properties": {"setting": {"type": "string"}, "premise": {"type": "string"}, "hour": {"type": "string", "enum": HOURS}, "atmosphere": {"type": "string"},
                                                  "scale_image": {"type": "string"}}, "required": ["setting", "premise", "hour", "atmosphere", "scale_image"]},
    "locations": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "description": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
                                              "required": ["id", "description", "tags"]}},
    "characters": {"type": "array", "items": CHARACTER_SCHEMA}, "viewer": {"type": "string"}, "stakes": {"type": "string"},
    "keywords": {"type": "array", "items": {"type": "string"}}, "must_include": {"type": "array", "items": {"type": "string"}}},
    "required": ["title", "world", "locations", "characters", "viewer", "stakes", "keywords", "must_include"]}
BEAT_SCHEMA = {"type": "object", "properties": {"act": {"type": "string", "enum": list(ACT_PURPOSE)}, "branch": {"type": "string", "enum": ["main", "A", "B"]}, "kind": {"type": "string", "enum": KINDS},
                                                "speaker": {"type": "string"}, "purpose": {"type": "string"}}, "required": ["act", "branch", "kind", "speaker", "purpose"]}
OUTLINE_SCHEMA = {"type": "object", "properties": {"hook_title": {"type": "string"}, "beats": {"type": "array", "items": BEAT_SCHEMA}, "ending_label_a": {"type": "string"}, "ending_label_b": {"type": "string"},
                                                   "closing_question": {"type": "string"}, "caption": {"type": "string"}},
                  "required": ["hook_title", "beats", "ending_label_a", "ending_label_b", "closing_question", "caption"]}
LINES_SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "location": {"type": "string"},
                                                                                                                       "in_shot": {"type": "array", "items": {"type": "string"}}},
                                                                                         "required": ["text", "location", "in_shot"]}}}, "required": ["lines"]}
DIRECTIONS_SCHEMA = {"type": "object", "properties": {"directions": {"type": "array", "items": {"type": "object", "properties": {
    "still": {"type": "string"}, "motion": {"type": "string"}, "camera": {"type": "string", "enum": CAMERAS},
    "fx": {"type": "object", "properties": {"zoom": {"type": "number"}, "shake": {"type": "number"}, "flash": {"type": "boolean"}}, "required": ["zoom", "shake", "flash"]}, "time": {"type": "string"}},
    "required": ["still", "motion", "camera", "fx"]}}}, "required": ["directions"]}
JUDGE_SCHEMA = {"type": "object", "properties": {k: {"type": "integer"} for k in ("hook", "coherence", "clues", "twist", "voice", "faithfulness", "variety")} | {"weakness": {"type": "string"}},
                "required": ["hook", "coherence", "clues", "twist", "voice", "faithfulness", "variety", "weakness"]}
