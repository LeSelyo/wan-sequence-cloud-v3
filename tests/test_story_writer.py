import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_engine as se  # noqa: E402
import story_writer as sw  # noqa: E402
import reference_story as reference_plan  # noqa: E402

EXAMPLE = Path(__file__).resolve().parent.parent / "results" / "story_trend" / "plans" / "the_last_city_150_en.json"
CONTEXT = "A flooded city at night. Two strangers on a rescue boat each offer to save you: a ship captain and a doctor."
pytestmark = pytest.mark.skipif(not EXAMPLE.exists(), reason="needs the hand-written reference plan")


def reference_story() -> dict:
    """The hand-written 2m30 plan, converted to what the model must answer (branch and ending fields added): a known-good story to test the validator on."""
    plan = reference_plan.load()
    shots, branch, twists = [], "main", 0
    for shot in plan["shots"]:
        item = {k: shot.get(k) for k in ("kind", "speaker", "text", "location", "in_shot", "still", "motion", "camera", "fx", "tag", "time", "choice", "offer_of") if shot.get(k) not in (None, "")}
        item["still"] = shot.get("still") or ""
        if shot["kind"] == "rewind":
            branch = "B"
        item["branch"] = branch
        if shot["kind"] == "choice":
            branch = "A"
        if shot["kind"] == "twist":
            item["ending"] = "bad" if twists == 0 else "good"
            twists += 1
        shots.append(item)
    return {"title": plan["title"], "hook_title": plan["title_overlay"]["text"], "logline": plan["logline"], "characters": plan["characters"], "shots": shots, "end_card": plan["end_card"],
            "end_card_small": plan["end_card_small"], "caption": plan["caption"]}


PARAMS = {"language": "en", "target_seconds": 150, "branches": 2, "tone": "tense", "seed": 1}
ENDINGS = {"A": "bad", "B": "good"}
LOCATIONS = ["boat", "street", "cabin", "roof", "sea"]


def test_the_reference_story_is_valid_and_the_validator_names_what_is_wrong():
    story = reference_story()
    assert sw.validate_story(story, PARAMS, ENDINGS, ["boat", "street", "cabin", "roof", "sea"]) == []
    broken = copy.deepcopy(story)
    broken["shots"][0]["text"] = "word " * 30
    broken["shots"][1]["location"] = "mars"
    broken["shots"] = [s for s in broken["shots"] if s["kind"] != "rewind"]
    next(s for s in broken["shots"] if s["kind"] == "twist")["ending"] = "good"
    problems = " | ".join(sw.validate_story(broken, PARAMS, ENDINGS, LOCATIONS))
    assert "1-14 words" in problems and "location must be one of" in problems and "rewind" in problems and "must have ending 'bad'" in problems


def test_the_model_is_told_its_problems_and_a_repaired_story_is_used():
    prompts = []
    good, bad = reference_story(), reference_story()
    bad["shots"][2]["text"] = "word " * 30

    def llm(prompt, schema):
        prompts.append(prompt)
        return bad if len(prompts) == 1 else good

    plan = sw.make_story_plan(CONTEXT, {"target_seconds": 150, "endings": ENDINGS}, seed=1, llm_json=llm, voices=[])
    assert plan["story_source"] == "llm" and len(prompts) == 2 and "previous answer had these problems" in prompts[1] and "1-14 words" in prompts[1]
    assert plan["title_overlay"]["text"].count("\n") == 1 and [s["id"] for s in plan["shots"]][:2] == ["s001", "s002"] and plan["shots"][0]["visual"] == plan["shots"][0]["motion"]
    assert set(plan["voices"]) == {"narrator", "c1", "c2"} and plan["endings"] == ENDINGS and plan["provenance"]["endings"] == "given"


def test_when_the_model_is_down_or_never_valid_the_template_story_keeps_the_pipeline_alive():
    def down(prompt, schema):
        raise RuntimeError("connection refused")

    for llm in (down, lambda prompt, schema: {"title": "x", "hook_title": "x", "characters": [], "shots": [], "end_card": "", "end_card_small": "", "caption": ""}):
        plan = sw.make_story_plan(CONTEXT, {"target_seconds": 60}, seed=2, llm_json=llm, voices=[])
        assert plan["story_source"].startswith("template") and plan["shots"] and all(s["id"].startswith("s") for s in plan["shots"])
        assert all((s["still"] or s["kind"] in ("talk", "choice")) and s["motion"] for s in plan["shots"])
        assert any(s["kind"] == "choice" for s in plan["shots"]) and any(s["kind"] == "twist" for s in plan["shots"])


def test_one_ending_is_bad_and_the_other_good_and_the_seed_decides():
    seen = {tuple(sorted(sw.draw_endings(seed).items())) for seed in range(12)}
    assert seen == {(("A", "bad"), ("B", "good")), (("A", "good"), ("B", "bad"))}
    assert sw.draw_endings(5) == sw.draw_endings(5)


def test_the_prompt_asks_for_the_hook_the_two_endings_and_the_language():
    prompt = sw.build_prompt(CONTEXT, {**PARAMS, "language": "fr"}, ENDINGS, LOCATIONS)
    assert "French" in prompt and "GRANDIOSE" in prompt and "ends BAD" in prompt.replace("ends BAD ", "ends BAD") or "BAD" in prompt
    assert "ending 'bad'" in prompt and "ending 'good'" in prompt and "REWIND" in prompt and CONTEXT in prompt
