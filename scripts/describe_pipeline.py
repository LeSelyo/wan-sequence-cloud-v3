"""Writes docs/PIPELINE_SCRIPTING_PROMPTING.json: HOW the script (story text) and ALL the prompts of the choice-story pipeline are generated, built from the live code (the prompts are the real constants, not
copies), with one real example taken from a finished project.

    python scripts/describe_pipeline.py [PROJECT_DIR]      (default: results/story_trend/auto/bunker_final_b)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2v_action as i2v  # noqa: E402
import method_registry as mr  # noqa: E402
import s2v_talk as st  # noqa: E402
import still_judge as sj  # noqa: E402
import story_agents as ag  # noqa: E402
import story_cards as sc  # noqa: E402
import story_identity as sid  # noqa: E402
import story_style as ss  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TREND = ag.load_trend("you_must_choose")


def retry_policy() -> dict:
    return {"attempts": 3, "how": "the answer is checked by CODE; the problems found are appended to the prompt ('Your previous answer had these problems. Answer again, the whole JSON, and fix them: [...]') and the model is asked again; "
            "the best answer is kept when only SOFT problems remain (a line a little long, an ending that reads mixed, a line judged confusing), never a template; a hard problem (a line over 14 words, a wrong count, a wrong order) "
            "after 3 attempts falls back to a template", "seed": "params.seed + attempt (+ an offset per chunk / candidate), so the same seed gives the same text"}


def build(project: Path) -> dict:
    plan = json.loads((project / "plan.json").read_text(encoding="utf-8")) if (project / "plan.json").exists() else None
    cards = json.loads((project / "cards" / "cards.json").read_text(encoding="utf-8")) if (project / "cards" / "cards.json").exists() else None
    example = {}
    if plan:
        shot = next(s for s in plan["shots"] if s["kind"] == "pov" and s["branch"] == "A")
        example = {"project": project.name, "context_given_by_the_user": plan["context"], "analyst_brief_excerpt": {k: plan["brief"][k] for k in ("title", "world") if k in plan["brief"]},
                   "shot": shot, "still_prompt_sent_to_krea2": (json.loads((project / "run" / "stills" / "stills.json").read_text(encoding="utf-8")).get(shot["id"], {}) or {}).get("prompt"),
                   "i2v_prompt_sent_to_wan": next((j["prompt"] for j in __import__("story_produce").animate_jobs(plan, cards, project / "run") if j["id"] == shot["id"]), None) if cards else None}
    return {
        "overview": "A CHAIN of constant prompts, one per role, each answering in a JSON SCHEMA through Ollama (model qwen3.6:27b, structured output, thinking off). Code (not the model) checks every answer, tells the model what is wrong "
                    "and asks again. Prompts never change from story to story: only their {{slots}} are filled (brief, counts, ending moods, previous lines...). PROMPT_VERSION is recorded in the plan.",
        "model": {"name": "qwen3.6:27b", "runtime": "Ollama on the rented box", "request": {"format": "the JSON schema of the stage", "think": False, "stream": False, "keep_alive": "10m (unloaded at the end of the story)",
                                                                                          "options": {"num_ctx": 12288, "temperature": 0.8, "top_p": 0.95, "num_predict": 7000, "seed": "see retry_policy"}}},
        "prompt_version": TREND.PROMPT_VERSION,
        "retry_policy": retry_policy(),
        "scripting": {
            "stages": [
                {"order": 0, "role": "IDEA (only when no context is given)", "input": "bones drawn by the seed from pools: place + what happens + hour (IDEA_WORLDS), two roles (IDEA_ROLES), tone (IDEA_TONES)", "prompt": TREND.IDEA_PROMPT,
                 "output_schema": TREND.IDEA_SCHEMA, "code_checks": "the context names the place, the event and both roles, 3-4 sentences, no ending told"},
                {"order": 1, "role": "ANALYST", "input": "the context (>= 3 sentences, given or invented)", "prompt": TREND.ANALYST_PROMPT, "output_schema": TREND.BRIEF_SCHEMA,
                 "code_checks": "exactly 2 characters c1/c2 with gender, age, look, wardrobe, public_promise, hidden_truth, voice_style; 5-6 locations with allowed tags; hour in HOURS; keywords; fallback = heuristic_brief()"},
                {"order": 2, "role": "PLANNER (best of N outlines)", "input": "the brief + the budget of beats per act (budget(): hook, setup, offers, choice, branch_a, rewind, branch_b; about one shot per 3.1 s) + the ending moods "
                                                                                 "(A and B drawn bad/good by the seed)", "prompt": TREND.PLANNER_PROMPT, "act_purposes": TREND.ACT_PURPOSE, "output_schema": TREND.OUTLINE_SCHEMA,
                 "code_checks": ["acts in order, counts within tolerance", "exactly two consecutive beats of kind offer (c1 then c2) in the offers act, no talk there", "branch A only narrator/c1, branch B only narrator/c2",
                                 f"first-person (pov) share of each branch >= {TREND.MIN_POV_SHARE:.0%} - 8 points", "a twist among the last two beats of each branch", "hook title 2 lines starting POV:", ">= 5 hashtags"],
                 "selection": "outline_score(): distinct purposes + brief keywords used + variety of kinds; the best candidate is kept (--outline-candidates)"},
                {"order": 3, "role": "WRITER (one call per chunk: hook..choice / branch A / rewind+branch B)", "input": "the beats of the chunk, the compact brief, the previous lines, the ending mood of the branch "
                                                                                                                  "(ENDING_MOOD + SUSPENSE teaser), average words per line, total words, approved example stories (the library, never the story under test)",
                 "prompt": TREND.WRITER_PROMPT, "offer_rule_filled_from_OFFER_VOICE": ag.offer_rule(TREND), "ending_moods": TREND.ENDING_MOOD, "suspense_rule": TREND.SUSPENSE, "output_schema": TREND.LINES_SCHEMA,
                 "code_checks": [f"1 line per beat, 1..{TREND.MAX_SPOKEN_WORDS} words (offer lines <= {TREND.MAX_OFFER_WORDS})", "average length <= 1.2 x the target (soft)", "location in the brief, no repeated line",
                                 "branch lines never name the other character (validate_independence; to be REVERSED in the next version)"],
                 "verifiers_second_model_calls": [{"name": "verify_polarity", "prompt": TREND.VERIFIER_PROMPT, "schema": TREND.VERIFIER_SCHEMA, "rule": "the outcome of the branch must read good/bad as drawn; a teaser does not make it mixed; the last line must leave a question open (suspense)"},
                                                  {"name": "verify_clarity (script doctor)", "prompt": TREND.CLARITY_PROMPT, "schema": TREND.CLARITY_SCHEMA, "rule": "lines a first-time viewer cannot understand are sent back (soft)"}]},
                {"order": 4, "role": "DIRECTOR (chunks of 12 shots)", "input": "the shots with their line, kind, branch, place, who is visible; the style (hour, atmosphere), camera moves list, the scale image of the world (first shot only)",
                 "prompt": TREND.DIRECTOR_PROMPT, "output_schema": TREND.DIRECTIONS_SCHEMA, "code_checks": ["one direction per shot, each repeating its number n (a list shifted by one shot is refused)", "motion >= 4 words", "still 14-60 words",
                                                                                                           "clean_still(): words a picture model must never get (text, letters, collage, panels...) are removed by code"]},
                {"order": 5, "role": "ASSEMBLY + RHYTHM (code only, no model)", "does": ["merges beat + line + direction into shots s001..s0NN", "offer shots: narrator voice, offer_of = c1/c2, the two-shot still built by offer_still() in the place of the choice",
                                                                                            "infer_in_shot(): a character named in the line/still/motion is in the shot (the other one never, after the choice)", "rhythm_pass(): zoom 0.03-0.08, shake clamped to 0.7 and only on impact words, "
                                                                                                                                                                                                                      "at most 1 shot in 6, never the same camera 3 times in a row, flash on every twist",
                                                                                            "a shot's own `time` is kept only if it is a daylight hour or another hour than the story's"]},
                {"order": 6, "role": "JUDGE (optional --judge, best of --candidates whole stories)", "input": "the whole story", "prompt": TREND.JUDGE_PROMPT, "output_schema": TREND.JUDGE_SCHEMA,
                 "criteria": ["hook", "coherence", "clues", "twist", "voice", "faithfulness", "variety"], "output": "scores 1-10 + the weakness in one sentence; the mean picks the candidate"},
            ],
            "final_checker": "story_writer.validate_story() runs on the assembled plan (kinds, words, twists with the right ending, one offer pair, <= 4 talk shots, branches independent, length 0.75-1.3 x target)",
            "plan_json_shape": {"title": "str", "hook_title": "2 lines", "characters": "[{id,name,role,gender,age,look,wardrobe}]", "locations": "[{id,description,tags}]",
                                "shots": "[{id,kind(narration|pov|talk|offer|choice|twist|rewind),branch(main|A|B),speaker,text,offer_of?,location,in_shot,still,motion,camera,fx{zoom,shake,flash},time?,tag?,ending?,choice?}]",
                                "end_card": "str", "caption": "str + hashtags", "brief": "...", "agents": "report of every stage (attempts, problems, sources)"},
        },
        "prompting": {
            "style_bible": {"how": "style_from_context(context, rng, tags=tags of the locations, hour, atmosphere): the kind of world is the most frequent location tag (space / shelter / sea / city) else a keyword rule; "
                                  "lighting, lens, grade, texture from that rule, the hour of the brief, 'freezing' words -> cold steel blue; palette and materials drawn by the seed from lists fitted to the kind of world; "
                                  "forced_elements = what each kind of place must always show (windows show the void of space, a shelter has no daylight...)", "prompt_suffix": "style_prompt(style, location_tags)",
                            "example": cards["style"] if cards else None, "example_suffix": ss.style_prompt(cards["style"], ["shelter"]) if cards else None},
            "krea2_still": {"template": "{still}, set in {location description}, vertical composition, {style_prompt}", "size": "576x1024", "seed": "one per shot, drawn from the plan seed",
                            "checked_by": "still_judge (vision model): sunlight in a night world, a place that does not belong, action not visible, collage, impossible geometry; up to 2 retries with another seed"},
            "character_cards": {"sheet": "character_text = '{name}, a {age}-year-old {woman|man}, {role}, {look}, wearing {wardrobe}'; sheet_prompt = character reference sheet ... three views ...; "
                                         "portrait_prompt = cinematic medium close-up ...; closeup_prompt = tight head-and-shoulders portrait, face filling a third, sharp eyes; 5 seeds, the one whose face the detector finds is kept",
                                "style": "same style_prompt suffix"},
            "identity_qwen_image_edit": {"single": sid.SINGLE, "two_shot": sid.TWO, "inputs": "Picture 1 = the Krea2 still of the shot; Picture 2 (and 3) = the close-up(s) of the character(s)",
                                         "wardrobe": "the character's wardrobe is written into the prompt", "applies_to": "shots with a character in them whose picture shows a face or a whole body (judge: person_visible)",
                                         "checked_by": "vision model compares with the close-up(s) (same_person), retries with another seed"},
            "i2v_wan22": {"engine": "Wan 2.2 image-to-video, two experts (high/low noise), lightx2v 4-step LoRAs, 480x832, 16 fps, <= 5 s", "negative": i2v.NEGATIVE, "styles": i2v.STYLES,
                          "routing": "method_registry.route(shot): need = classify_shot(kind, who is in the shot, motion words) -> method chosen by status (works > partial > untested; a failing method is never chosen)",
                          "needs_and_methods": {n: [r["method"] + ":" + r["status"] for r in mr.Registry().records if r["need"] == n] for n in mr.NEEDS}},
            "s2v_wan22": {"used_for": "ONLY a character who speaks or shouts (talk shots): the close-up portrait + the voice line", "negative": st.NEGATIVE, "lora": "t2v_1217_low, 4 steps"},
            "voices": {"narrator": "recorded sample cloned (Qwen3-TTS), the characters: cloned if a recorded voice has a transcript else a voice DESIGNED from a text description (age, gender, role, voice_style)",
                       "offer_lines": f"OFFER_VOICE = {TREND.OFFER_VOICE}: the narrator quotes the two proposals"},
            "vision_judge": {"prompt": sj.PROMPT, "schema": sj.SCHEMA, "same_person_rule": sj.SAME_PERSON_REFS},
            "render": "story_render.py (Pillow + ffmpeg on the box): one-word glitch subtitles, choice card with A/B + countdown ring over the two-shot clip, sound background from nothing (rain only in a wet world)",
        },
        "real_example": example,
    }


def main() -> None:
    project = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "story_trend" / "auto" / "bunker_final_b"
    out = ROOT / "docs" / "PIPELINE_SCRIPTING_PROMPTING.json"
    out.write_text(json.dumps(build(project), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
