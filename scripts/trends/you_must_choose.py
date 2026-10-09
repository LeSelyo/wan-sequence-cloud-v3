"""The TREND "you must choose": everything that is specific to it lives HERE (the constant prompts, the structure recipe, the schemas of what each agent answers, the render preset), so another trend is another
file next to this one and the chain (scripts/story_agents.py) does not change.

A video of this trend: a hook with a GRANDIOSE picture and the words TikTok reads, a setup, two characters who offer help and each hide a clue, THE CHOICE (A / B card with a countdown), branch A, a rewind,
branch B (one branch ends BAD for you, the other GOOD, each with a twist), a closing question.

THE CONSTANT PROMPTS. Each agent has ONE prompt text that never changes (the same words for every video of the trend: that is what keeps the quality stable). Only the SLOTS `{{name}}` are filled, from
the BRIEF that the analyst agent extracts from the context. `PROMPT_VERSION` changes when a constant text changes, and the report of a story records it.
"""
from __future__ import annotations

NAME = "you_must_choose"
PROMPT_VERSION = "2026-10-09.6"  # .6: the LENGTH agent decides how long a video is when none is fixed (2 to 10 minutes), the outline of a long video is planned by parts, its lines are written in smaller chunks; the other prompts are unchanged
PROMPT_VERSION_PREVIOUS = "2026-10-09.5"  # .5: the characters are introduced by name BEFORE the offer, the offer is immediately followed by the choice, both characters may appear in a branch, the director names them (.4: characters speak rarely, the narrator quotes the offers; .3: ONE offer shot, independent branches, suspense)
KINDS = ["narration", "talk", "pov", "choice", "twist", "rewind", "offer"]
CAMERAS = ["wide", "medium", "close", "pov"]
LOCATION_TAGS = ["space", "sea", "city", "shelter", "window", "forest", "desert", "ice", "underground"]  # the tags that make the style prompt force what a kind of place must always show
HOURS = ["night", "dusk", "dawn", "day"]
MAX_SPOKEN_WORDS = 14
MAX_OFFER_WORDS = 9  # the two lines of the offer shot are said in about 4 s together, the second one starting over the end of the first
OFFER_OVERLAP = 0.0  # seconds the second line of the offer may start before the first one ends (0 = one after the other: "or not"; the render does not mix overlapping lines yet)
OFFER_VOICE = "narrator"  # who says the two proposals: "narrator" (quotes them, the reference video does this) or "characters" (their own voices over the clip, the mouths do not move)
MAX_TALK_BEATS = 4  # characters SPEAK (a close-up whose mouth follows the voice) only when the scene calls for it: an order, a shout
MIN_POV_SHARE = 0.3  # share of the beats of a branch that are first-person action shots
SECONDS_PER_SHOT = 3.8  # measured on the finished video "The Last Breath": 173.5 s for 46 shots = 3.77 s per shot (voice + pauses + the end card). With 3.1 a "2 minute" video came out near 3 minutes
LENGTH_RANGE = (120, 600)  # AUTO LENGTH (no fixed length): what the length agent may decide, in seconds: never under 2 minutes, never over 10
IDEA_AUTO_SENTENCES = (5, 14)  # the idea agent of an auto-length video writes as many sentences as the story needs, between these bounds
PLANNER_SPLIT_BEATS = 60  # beyond this many beats the outline is planned act by act: one answer of the model is limited (about 7000 tokens) and a list of 150 beats would be cut
WRITER_MAX_CHUNK = 28  # the lines are written in chunks of at most this many beats (a branch of a 10-minute story has about 45); a shorter video never reaches it
SHARES = {"hook": 0.07, "setup": 0.09, "offers": 0.20, "branch_a": 0.27, "branch_b": 0.30}  # share of the shots of each act (two branches); one branch gets branch_a + branch_b
CAMERA_MOVES = ("aerial flight forward", "whip pan", "low angle tilt up", "handheld run", "slow push-in", "crane up", "dolly forward", "orbit around the subject", "lateral glide", "fast zoom-in", "tilt down")
FORBIDDEN_IN_PICTURES = ("text", "letters", "caption", "subtitle", "watermark", "logo", "collage", "split screen", "panels", "triptych", "diptych")

DEFAULTS = {"language": "en", "target_seconds": 150, "branches": 2, "tone": "tense", "aspect": "9:16", "fps": 30}

RENDER_PRESET = {"size": [720, 1280], "fps": 30, "subtitle": "one word at a time, glitch entrance (4 frames) then clean", "choice_ui": "orange A/B cards + countdown ring", "tags": ["CASE A", "CASE B", "ENDING A", "ENDING B"],
                 "sound": "rain, water, drone, heartbeat on the choice, riser+impact on each twist, reverse swell on the rewind, bright pad on a good ending"}


def budget(seconds: float, branches: int, shot_scale: float = 1.0) -> dict[str, int]:
    """How many shots each act has for a video of `seconds` (about one shot every SECONDS_PER_SHOT s). `shot_scale` < 1 is TEST MODE (never the default): shorter shots, so more of them in the same time."""
    total = max(12, round(seconds / (SECONDS_PER_SHOT * shot_scale)))
    two = branches >= 2
    shares = dict(SHARES) if two else {"hook": 0.09, "setup": 0.11, "offers": 0.28, "branch_a": 0.52, "branch_b": 0.0}
    counts = {act: max(3, round(total * share)) for act, share in shares.items() if share > 0}
    counts["setup"] += max(0, counts["offers"] - 2)  # the offers are exactly the two proposals: everything that gave context goes BEFORE them (the characters introduced, the clues)
    counts["offers"] = 2
    counts["choice"] = 1
    if two:
        counts["rewind"] = 1
    return {act: counts[act] for act in ACT_PURPOSE if act in counts}  # always in the order of the story: the planner is told this order and the checker expects it


def max_words(shot_scale: float = 1.0) -> int:
    """The longest spoken line of a shot: the normal limit, shorter when the shots are (TEST mode: a shot of 1.5 s holds about 4 words, not 14)."""
    return MAX_SPOKEN_WORDS if shot_scale >= 1 else max(5, round(MAX_SPOKEN_WORDS * shot_scale))


def max_offer_words(shot_scale: float = 1.0) -> int:
    return MAX_OFFER_WORDS if shot_scale >= 1 else max(4, round(MAX_OFFER_WORDS * shot_scale))


def locations_asked(seconds: float, sentences: int) -> str:
    """How many places the analyst is asked for: the size of the context decides (more sentences, more places) and a long video gets more of them, so that a hundred shots do not stay in six rooms
    (one more place per 90 s above 2:30, at most 10 or 11)."""
    base = 5 if sentences < 7 else 6 if sentences < 10 else 7
    low = min(base + max(0, round((seconds - 150) / 90)), 10)
    return f"{low} or {low + 1}"


ACT_PURPOSE = {
    "hook": "GRANDIOSE first picture (epic scale, awe) that puts the viewer in the world at once; the spoken lines contain the key words of the story; the third shot puts YOU in danger.",
    "setup": "CONTEXT before the offer. First the place and the danger, then the two main characters are INTRODUCED one AFTER the other, c1 first then c2: each in a clear picture of his or her own, named, with what he or she wears "
             "and carries, and ONE visible CLUE (an object or a detail that hints at his or her hidden truth). Every beat names the character it is about. Nothing else happens before the offer.",
    "offers": "EXACTLY two beats and nothing else: kind 'offer', first c1 then c2 (the speaker field says whose proposal it is). Both stand side by side and hold out a hand to you; each beat is the proposal of that "
              "character in ONE very short line told by the narrator; they never accuse each other.",
    "choice": "IMMEDIATELY after the second offer (no beat in between): the narrator says it is time to choose (kind 'choice'), the same scene, c1 and c2 still hold out their hands.",
    "branch_a": "the WHOLE first point of view: you follow c1 (the one you chose) from start to end; c2 never appears and is never named; real ACTION of your own hands and body (first-person 'pov' beats); the clues come true; ends with a twist (kind 'twist') and a last line that leaves a question open.",
    "rewind": "the story rewinds: the narrator asks what if you had chosen c2 (kind 'rewind').",
    "branch_b": "the WHOLE second point of view: you follow c2 (the one you chose) from start to end; c1 never appears and is never named; FAST action shots (chase, collapse, escape) and real ACTION of your own hands and body (first-person 'pov' beats); ends with a twist (kind 'twist') and a last line that leaves a question open.",
}

# ---------------------------------------------------------------------------------------------- the constant prompts
ANALYST_PROMPT = """[[STAGE:analyst]]
You are the ANALYST of a short-video studio. The trend is "you must choose": a second-person interactive story ("you") in which two characters offer to save you, the viewer chooses, and each choice
leads to a different ending with a twist. Read the CONTEXT below ({{context_size}}) and extract the BRIEF the other writers will work from. The GENRE is the one of the context (realistic, science fiction, fantasy,
supernatural...): never make the story more realistic or more ordinary than the context says.

RULES
- Keep EVERY concrete fact of the context (places, roles, objects, numbers, the hour, the weather, the rules of the world, its magic). Invent only what is missing, and make it fit. The longer the context, the more
  of it must be found in the brief: every named place becomes a location (up to the number asked), every object or rule goes into "premise", "stakes" or "must_include".
- Exactly two main characters, ids "c1" and "c2": the two beings named or described in the context (people or not: a fairy, an elf thief, a ghost, a robot, a knight...; their roles come from it). If the context
  does not describe them, INVENT them, fitting its world and genre. If the context names a real public figure, invent an archetype with a made-up name instead and list the replacement in "substitutions".
  Give each a name, a gender (m or f: the closest one, it is used for the voice), an age, a face or body description, a wardrobe, what they PUBLICLY promise the viewer, and a HIDDEN TRUTH
  (what they are really doing or hiding) that a clue can reveal later.
- "locations": {{n_locations}} places of this world, id = short lowercase word, the first one is the main set; tags only from {{location_tags}}.
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
- STRUCTURE: setup (the characters are introduced) -> offers (EXACTLY the two beats of kind offer, speaker c1 then c2, both in the same picture holding out a hand, the narrator tells each proposal, nobody accuses the other) -> choice (the very next beat) -> the whole branch A -> rewind -> the whole branch B. NOTHING happens between the second offer and the choice. The context and the clues are told BEFORE the offers.
- THE CHARACTERS SPEAK RARELY: kind talk (a close-up whose mouth follows the voice) ONLY when a character gives an ORDER or shouts to someone in the scene, at most {{max_talk}} in the whole video. Everything else is told by the narrator, who may quote what they say.
- ONE CHARACTER PER BRANCH: branch A is you and c1 from start to end, branch B is you and c2 from start to end. The other one never appears and is never named (except by the rewind narrator). Anything that helps the viewer understand the story can happen around you, in or out of the frame of the chosen character (an explosion, a door closing, a countdown).
- ACTION: at least a third of the beats of each branch are kind pov: YOUR hands or body DO something (turn, push, pull, climb, grab, lift, run, open, hold on to something that moves) - never just "holding an arm".
- SUSPENSE: the last beat of each branch leaves ONE question open (a sound, a door, a name, a detail that was not explained); it never closes the story with a full explanation.
- The clues: c1's hidden truth and c2's hidden truth must each be hinted at in the SETUP act (one clue each, when the character is introduced) and CONFIRMED in their branch (branch A confirms c1's, branch B confirms c2's).
- ENDINGS. Branch A: {{mood_a}}
  Branch B: {{mood_b}}
  Each branch has a twist beat that RECASTS what you believed, with a short UPPERCASE ending label ("COLLECTED", "THE CURE"...).
- Every beat has: act, branch (main for hook/setup/offers/choice, A or B after the choice; the rewind has branch B), kind, speaker (narrator, c1 or c2; only kind talk is spoken by c1 or c2), and "purpose" = ONE sentence
  saying what happens and what the picture shows.
- The first beat is a hook (kind narration or pov). Make the hook's spoken idea contain the biggest keyword of the brief. Vary the beats: no two consecutive beats with the same purpose.
- "hook_title": exactly two UPPERCASE lines separated by a newline, the first starts with "POV:", both contain keywords of the brief. "closing_question": two short UPPERCASE lines. "caption": one sentence + 8 to 10
  hashtags built from the keywords.
LANGUAGE of hook_title, closing_question, caption, ending labels: {{language}}.

BRIEF:
{{brief}}

Answer with the JSON outline only."""

LENGTH_PROMPT = """[[STAGE:length]]
You are the PRODUCER of a short-video studio, trend "you must choose": a second-person story in which two characters each offer to save you, you choose, you live the WHOLE story of the character you chose, then
the story rewinds and you live the whole story of the other one; each ends with a twist. Nobody fixed the length of this video: YOU decide it, from the CONTEXT below, as the story NEEDS: as long as it takes
to tell it clearly and well, never padded, never rushed.

HOW TO DECIDE
- A shot lasts about {{shot_seconds}} seconds, so 2:00 is about {{shots_2min}} shots and every additional minute adds about {{shots_per_minute}} shots.
- First list the KEY EVENTS the story contains: the places we go through, the dangers, the rules or objects of the world, the clues, the reversals, what each of the two characters does in his or her own
  branch (6 to 20 short items). Then decide the length from them.
- A simple story (one place, one danger, one clue and one twist per branch) needs 2:00 to 2:30. A story with several places, a real journey in each branch and a twist that needs a set-up needs 3:00 to 5:00:
  this is the usual case. Only a vast world with many places, rules and reversals deserves 6:00 to 10:00.
- Never less than {{min_seconds}} seconds, never more than {{max_seconds}} seconds.

CONTEXT:
{{context}}

Answer with the JSON only: "events", "seconds" (a whole number of seconds) and "why" (one sentence)."""
LENGTH_SCHEMA = {"type": "object", "properties": {"events": {"type": "array", "items": {"type": "string"}}, "seconds": {"type": "integer"}, "why": {"type": "string"}}, "required": ["events", "seconds", "why"]}

PLANNER_PART_NOTE = """THIS VIDEO IS LONG, so its plan is written in {{parts}} parts, one after the other. You are writing PART {{part}} of {{parts}}: plan ONLY the acts listed in the structure above, with their counts.
The other parts are planned separately: ignore every rule about an act that is not in your structure (the hook beat, the title, the closing question and the endings concern only the part that contains them),
but always answer every field of the JSON.
{{already}}"""

WRITER_PROMPT = """[[STAGE:writer]]
You are the WRITER of a short-video studio, trend "you must choose". Write the SPOKEN LINES of the beats below, in order, one line per beat, in {{language}}.

RULES
- At most {{max_words}} words per line and about {{avg_words}} words on AVERAGE (the whole video must stay near {{total_words}} words): most lines are short, only a twist may be longer. Short punchy sentences, natural
  speech, no stage directions, no quotation marks. Second person for the narrator ("you").
- THE ENDING OF THIS PART: {{ending_mood}}
- kind talk = the character gives an ORDER or shouts it in the scene (c1 and c2 do not sound alike: use their voice_style). {{offer_rule}} The other kinds are the narrator, who may quote what a character says.
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
You are the DIRECTOR of a short-video studio, trend "you must choose": vertical 9:16, a clean, believable photographic cinematic look (a photoreal RENDER applied to ANY world, fantasy and supernatural ones included), ONE art direction for the whole video (hour: {{hour}}; atmosphere: {{atmosphere}}).
For each shot below (its spoken line is given), write the PICTURE and the CAMERA MOVE.

RULES
- "still" = ONE concrete filmable photograph (what is in the frame, the angle, the light), 20 to 45 words, in English, a single continuous scene. NEVER ask for text, letters, captions, logos, a collage, panels or a
  split screen. Characters: use their look and wardrobe from the brief. For kind talk and choice leave "still" empty (the portrait of the character is used) but ALWAYS write the "motion": how the character acts while speaking (gesture, glance, expression, what moves around).
- "motion" = the camera move and what moves in the picture, 10 to 25 words, English. Use a DIFFERENT camera move from the neighbouring shots, from this list or similar: {{camera_moves}}. At least one third of
  the shots must be dynamic (fast flight, chase, whip pan, impact).
- WHO IS WHO: whenever a main character is in a picture write his or her NAME in the still AND in the motion, with what he or she wears; never "he", "she", "the figure", "the subject" alone. The viewer is never "the narrator": write "you" (first-person view), "your hands" or "the camera".
- Kind pov = FIRST PERSON: the picture shows YOUR hands and forearms (or what you see ahead of you) in the middle of a physical action, and the "motion" says what MOVES: the hands crank, push, pull or climb and the object turns, opens or gives way, the camera moves with you. Describe ONLY what is in the picture and what happens to it. Never a person who just stands and trembles.
- Every shot with a character in it names a physical verb (walks, runs, climbs, turns, opens, reaches, falls); the kind offer shows c1 and c2 side by side, both facing you and holding out an open hand toward the camera.
- Keep every picture consistent with the rules of THIS world (its magic and fantasy too): no rain, wind or sea where this world has none (a lunar dome, a vacuum, a desert...); use its own light, dust, steam, sparks, snow, glow.
- "fx": zoom 0.03 to 0.08; shake 0 to 1 (above 0.7 only on impacts, chases, crashes); flash true only on a shock.
- "camera" in {{cameras}}. "time": only when the hour of the picture differs from "{{hour}}" (e.g. the dawn of a good ending), English, e.g. "it is dawn: golden light".
- Every direction starts with "n": the NUMBER of its shot in the list below (1, 2, 3...), so that the pictures stay aligned with the lines: direction n is for shot n, never for another one.
- The grandiose picture of the world is for the FIRST shot only, never reuse it: {{scale_image}}. Every other shot has its OWN picture, made from ITS line.

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
    "n": {"type": "integer"}, "still": {"type": "string"}, "motion": {"type": "string"}, "camera": {"type": "string", "enum": CAMERAS},
    "fx": {"type": "object", "properties": {"zoom": {"type": "number"}, "shake": {"type": "number"}, "flash": {"type": "boolean"}}, "required": ["zoom", "shake", "flash"]}, "time": {"type": "string"}},
    "required": ["n", "still", "motion", "camera", "fx"]}}}, "required": ["directions"]}
JUDGE_SCHEMA = {"type": "object", "properties": {k: {"type": "integer"} for k in ("hook", "coherence", "clues", "twist", "voice", "faithfulness", "variety")} | {"weakness": {"type": "string"}},
                "required": ["hook", "coherence", "clues", "twist", "voice", "faithfulness", "variety", "weakness"]}


# ---------------------------------------------------------------------------------------------- the moods of the endings (what "bad" and "good" must mean in the words)
SUSPENSE = " SUSPENSE: the very last line is a TEASER that leaves ONE question open (a sound, a door, a name, a detail nobody explained) without changing the outcome: the viewer is still saved or still lost; never wrap everything up, never say the end."
ENDING_MOOD = {
    "bad": "it ends BAD for you: the twist shows that the person you chose lied to you or used you, and you lose your freedom, your people or your life. Not a happy ending, no last-second rescue." + SUSPENSE,
    "good": "it ends GOOD for you: the twist shows that the person you chose was really on your side, or that you were the hero all along; you are saved and the ending feels earned. NO death, NO betrayal, NO horror at the end." + SUSPENSE,
}
OPEN_MOOD = "(the story is still open here: nobody is saved or lost yet)"

# ---------------------------------------------------------------------------------------------- FROM NOTHING: the IDEA agent invents the context of a video from a random seed
IDEA_WORLDS = [  # (the place, what is happening to it, the hour)
    ("a flooded megacity", "the sea has swallowed the streets and keeps rising", "night"),
    ("a drifting spaceship", "the sun it orbits is dying and the hull is cracking", "dusk"),
    ("a mountain village", "an avalanche has buried the road and the last cable car is dead", "night"),
    ("an underground bunker", "the air filters are failing after a war on the surface", "night"),
    ("an oil platform in the ocean", "a monstrous storm is tearing it apart", "dusk"),
    ("a desert city", "a black sun has risen and the heat melts the streets", "day"),
    ("a space station above a dark Earth", "the lights of every city below have gone out", "night"),
    ("a burning forest town", "a wall of fire is closing in from every side", "dusk"),
    ("a frozen harbor", "an endless winter has locked the ships in the ice", "dawn"),
    ("a clifftop monastery", "the sea is rising to the walls and a fog hides the coast", "dusk"),
    ("a train crossing a ruined continent", "the rails ahead are gone and something follows the train", "night"),
    ("a lunar colony", "the moon is breaking apart under the domes", "night"),
    ("a research base in a jungle", "a strange fog has swallowed the camp and the radio is silent", "dawn"),
    ("an island city", "a tsunami warning has sounded and the bridges are collapsing", "day"),
    ("a skyscraper in a sandstorm", "the whole city is buried in a wall of red dust", "day"),
    ("a lighthouse on a dying coast", "the tide never goes out any more and the sky is green", "dusk"),
]
IDEA_ROLES = [("ship captain", "doctor"), ("engineer", "priest"), ("soldier", "nurse"), ("smuggler", "scientist"), ("pilot", "teacher"), ("firefighter", "journalist"),
              ("mechanic", "mayor"), ("hunter", "botanist"), ("police officer", "street musician"), ("monk", "geologist"), ("radio operator", "chef"), ("guide", "surgeon")]
IDEA_TONES = ["tense", "bleak", "paranoid", "desperate", "cold", "eerie"]
IDEA_WORLDS += [
    ("a cable-car station on a glacier", "a crack is splitting the ice under the pylons and the cabins are stuck above the void", "dawn"),
    ("a floating market city on stilts", "the water is draining away and the whole city leans toward the abyss", "dusk"),
    ("a deep-sea mining rig", "the pressure doors are failing one after the other and the lights go out level by level", "night"),
    ("a hospital in a besieged city", "the generators are dying and the front line reaches the lower floors", "night"),
    ("a mountain observatory", "a solar storm has fried every machine and an avalanche is coming", "day"),
    ("an abandoned amusement park at the edge of a swamp", "a toxic tide is rising and the rides are starting to move by themselves", "dusk"),
    ("a salt desert caravan", "the horizon has disappeared in a white storm and the water is nearly gone", "day"),
    ("a high-speed train in a tunnel under a sea", "the tunnel is flooding behind the last carriage", "night"),
    ("a floating greenhouse city in the clouds", "the balloons that hold it up are leaking one by one", "dawn"),
    ("a cathedral turned refuge", "the walls are cracking from an earthquake that does not stop", "night"),
    ("a polar research base", "the aurora has cut every radio and something is walking on the ice around the camp", "night"),
    ("a stranded cruise ship in a dead port", "the lower decks are filling with a red tide and the lifeboats are gone", "dusk"),
    ("a mining town under an ash cloud", "the volcano above it woke up and the only road is closing", "day"),
    ("an orbital elevator", "the cable has snapped above the middle platform and the cabin is falling slowly", "dawn"),
    ("a submarine under the ice cap", "the reactor is overheating and the ice above is closing", "night"),
    ("a night market in a flooded old town", "the dams have broken upstream and the first wave is five minutes away", "night"),
    ("a wind farm in a hurricane", "the towers are folding one by one toward the control hut", "dusk"),
    ("a library-fortress in a burning city", "the fire is eating the stacks and the exits are walled by collapsed shelves", "night"),
    ("a mountain pass monastery", "a blizzard has erased the path and the bell keeps ringing though nobody rings it", "dusk"),
    ("a desert pipeline station", "a sandstorm the size of a country is erasing the track and the pumps are about to explode", "day"),
]
IDEA_ROLES += [("lighthouse keeper", "surgeon"), ("bounty hunter", "librarian"), ("diver", "judge"), ("train conductor", "astronomer"), ("beekeeper", "general"), ("cartographer", "locksmith"),
               ("midwife", "bodyguard"), ("clockmaker", "paramedic"), ("fisherman", "archivist"), ("stunt pilot", "veterinarian"), ("pastor", "demolition expert"), ("courier", "hacker"),
               ("park ranger", "arms dealer"), ("magician", "forensic doctor"), ("chef", "mountain guide"), ("translator", "sapper"), ("farmer", "pharmacist"), ("sailor", "violinist")]
IDEA_TONES += ["claustrophobic", "melancholic", "feverish", "ominous", "ruthless", "dreamlike", "grim", "restless"]
# the ingredients the model is given to INVENT a place of its own (two are drawn by the seed, the place must be new and must not be one of the worlds above)
IDEA_INGREDIENTS = ["salt", "glass", "ice", "rust", "moss", "neon", "sand", "ash", "steam", "fog", "copper", "bones", "silk", "oil", "lava", "coral", "concrete", "mirrors", "rope", "bells", "honey", "feathers", "amber", "antlers", "tapestries", "orchids", "obsidian", "pearls", "kites", "thunder",
                    "lace", "clay", "moonlight", "embers", "willows", "mosaics", "gears", "horses", "ink", "stained glass",
                    "cables", "wax", "tar", "snow", "tide", "thorns", "lanterns", "dust", "marble", "wire", "rails", "sails", "chimneys", "bridges", "tunnels", "silos", "balloons", "antennas",
                    "greenhouses", "stairs", "elevators", "cranes", "locks", "orchards", "cliffs", "caves", "canals", "domes", "wells", "ladders", "pylons", "reefs", "ruins", "storms",
                    "pipes", "ferries", "windmills", "scaffolding", "dams", "railways"]
IDEA_KINDS = ["a natural wonder", "a moving vehicle", "an underground world", "a city district", "a sacred or ancient site", "a water world", "a frozen land", "a place in the sky", "a desert", "a forest",
              "the inside of a gigantic creature", "a ruined future city", "a dreamlike landscape", "a market or a festival", "a ship at sea", "a mountain village", "an island", "a castle or a palace",
              "a train or a road", "a farm or a village in a valley"]
IDEA_GENRES = ["realistic disaster", "science fiction", "high fantasy", "dark fantasy", "folk tale", "supernatural horror", "post-apocalyptic", "steampunk", "cosmic space opera", "fairy tale", "mythology", "gothic"]
IDEA_NAMES = ["Mira", "Tobias", "Ines", "Rurik", "Amara", "Dov", "Salome", "Bram", "Noor", "Casimir", "Yuki", "Odalys", "Leif", "Zainab", "Teodor", "Anselm", "Imogen", "Kasimir", "Saoirse", "Ravi",
              "Philippa", "Okoye", "Bastien", "Mei", "Gideon", "Thalia", "Anouk", "Jasper", "Esme", "Florian", "Nadia", "Cormac", "Hana", "Matthias", "Lucinda", "Idris", "Wren", "Benedikt", "Soraya", "Tomas",
              "Marguerite", "Ansel", "Delphine", "Kofi", "Perrin", "Yara", "Ignatius", "Juno", "Oskar", "Vesna", "Lorcan", "Pilar", "Ezra", "Calliope", "Dmitri", "Rosalind", "Hiro", "Ottoline", "Bertrand", "Zuri"]
IDEA_INVENT = True  # False = the place is drawn from IDEA_WORLDS as before
IDEA_PROMPT = """[[STAGE:idea]]
You are the IDEA writer of a short-video studio, trend "you must choose" (a second-person story: two characters each offer to save you, you choose, each choice has an ending with a twist). Nobody gave you a
subject: a random seed gives the bones, you invent the rest.

SEED (follow it): place = {{setting}} | what is happening = {{premise}} | hour = {{hour}} | the two characters who offer to save you = a {{role_a}} and a {{role_b}} | tone = {{tone}}

Write the CONTEXT of ONE video: three or four sentences in English that say WHERE and WHEN we are, WHAT is happening to the world, who YOU are and what you need, and that two strangers each offer to save you
(name their two roles). Be concrete and vivid, add one surprising detail of your own, no real people or brands, do not tell how it ends. Any genre is welcome (realistic, science fiction, fantasy,
supernatural). The context has about {{sentences}}. Also give a short title (2 to 4 words).

Answer with the JSON only."""
IDEA_INVENT_PROMPT = """[[STAGE:idea]]
You are the IDEA writer of a short-video studio, trend "you must choose" (a second-person story: two characters each offer to save you, you choose, each choice has an ending with a twist). Nobody gave you a
subject: you INVENT the place and the catastrophe, from the seed below.

SEED (follow it): kind of place = {{kind}} | genre = {{genre}} | ingredients to build the place around = {{ingredients}} | hour = {{hour}} | tone = {{tone}}
The place must be NEW: never one of these (already made): {{avoid}}.

Invent a filmable place of that kind (a place with room to move through and several distinct areas) and what is happening to it right now. Follow the genre of the seed: whatever it is, the pictures will be
rendered as clean, believable photographic scenes. Do not default to industrial structures (towers, catwalks, refineries, dams) unless the seed asks for them.
THE TWO CHARACTERS: invent them yourself, any kind of beings (people or not: a fairy, an elf thief, a ghost, a robot, a knight, an old fisherman...), each with a name or a title, a few words of look and clothes,
and what he or she offers you. Two strangers each offer to save you. NAMES: use these names (or names in the same spirit, never Kael, Elara, Silas, Lyra or Aria): {{names}}. Introduce the two in two
DIFFERENT ways (never "to your left ... to your right").
Write the CONTEXT of ONE video: about {{sentences}} in English that say WHERE and WHEN we are, WHAT is happening, who YOU are and what you need, who the two strangers are and what each one offers. Be vivid, add
one surprising detail of your own, no real people or brands, do not tell how it ends. Also give the place in a few words ("setting"), what is happening to it ("premise") and a short title (2 to 4 words).

Answer with the JSON only."""
IDEA_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}, "context": {"type": "string"}, "setting": {"type": "string"}, "premise": {"type": "string"}}, "required": ["title", "context"]}

VERIFIER_PROMPT = """[[STAGE:verifier]]
You are the VERIFIER of a short-video studio. Read the LAST LINES of one branch of a "you must choose" story and say how the story ENDS FOR THE VIEWER ("you"):
- "good": you are saved, you win, or you turn out to be the hero, and the tone is hopeful;
- "bad": you die, are captured, used, betrayed or lose something essential, and the tone is dark;
- "mixed": neither clearly.
Judge the OUTCOME that the twist and the lines before the last one show for the viewer. The very last line may add a teaser (an unexplained sound, door, name): a teaser does NOT make a good ending mixed and does not make a bad one worse.
Also say whether the LAST LINE leaves a question open (suspense = true: an unexplained sound, door, name or detail, the viewer wants to know what comes next) or closes everything (suspense = false).

LAST LINES:
{{lines}}

Answer with the JSON only."""
CLARITY_PROMPT = """[[STAGE:clarity]]
You are the SCRIPT DOCTOR of a short-video studio. A viewer sees ONE part of a short story told to him or her ("you"), line by line, once, with no pause. Find the lines this viewer cannot understand: a reference
to someone or something never introduced, a phrase that makes no sense, a sudden change nobody caused. Be strict but fair: short punchy lines are fine, nonsense is not.

THE LINES BEFORE THIS PART (context): {{previous}}

THE LINES OF THIS PART (numbered):
{{lines}}

Answer with the JSON only: the numbers of the confusing lines with a short reason (an empty list when every line is clear)."""
CLARITY_SCHEMA = {"type": "object", "properties": {"confusing": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "why": {"type": "string"}}, "required": ["n", "why"]}}},
                  "required": ["confusing"]}
VERIFIER_SCHEMA = {"type": "object", "properties": {"polarity": {"type": "string", "enum": ["good", "bad", "mixed"]}, "reason": {"type": "string"}, "suspense": {"type": "boolean"}}, "required": ["polarity", "reason", "suspense"]}
