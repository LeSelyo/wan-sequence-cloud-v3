import json
import random

import numpy as np
import pytest
from PIL import Image

from scripts import story_engine as se
from scripts import story_style as ss

CONTEXT = "an apocalypse on the moon, a spaceship, two people offer you help"
VOICES = [{"id": "dan", "role_default": "narrator", "gender_hint": "m"}, {"id": "siren", "role_default": "character", "gender_hint": "f"},
          {"id": "jessica", "role_default": "character", "gender_hint": "f"}]


# ---------------------------------------------------------------- parameters
def test_every_parameter_has_a_default_and_says_where_it_comes_from():
    params, provenance = se.resolve_params({"language": "fr", "target_seconds": 40}, seed=9)
    assert params["language"] == "fr" and provenance["language"] == "given" and provenance["target_seconds"] == "given"
    assert provenance["aspect"] == "default" and params["aspect"] == "9:16" and params["size"] == [576, 1024]
    assert provenance["tone"] == "rng" and provenance["characters"] == "rng"
    again, _ = se.resolve_params({"language": "fr", "target_seconds": 40}, seed=9)
    assert again == params  # the draws are reproducible from the seed
    assert se.resolve_params({"tone": "cold"}, seed=9)[0]["tone"] == "cold"  # a drawn default can be overridden


def test_unknown_parameter_and_unknown_language_are_refused():
    with pytest.raises(ValueError):
        se.resolve_params({"colour": "red"})
    with pytest.raises(ValueError):
        se.resolve_params({"language": "de"})


# ---------------------------------------------------------------- the story
def test_template_plan_is_valid_reproducible_and_has_the_trend_structure():
    a, b = se.make_plan(CONTEXT, seed=5, voices=VOICES), se.make_plan(CONTEXT, seed=5, voices=VOICES)
    assert a["shots"] == b["shots"] and a["characters"] == b["characters"]
    assert se.validate_plan(a) == []
    kinds = [s["kind"] for s in a["shots"]]
    assert kinds.count("choice") == 1 and kinds.count("talk") >= 2 and kinds.count("twist") == 2  # one twist at the end of each branch
    talk = [s for s in a["shots"] if s["kind"] == "talk"]
    assert all(s["speaker"].startswith("c") and se.count_words(s["text"]) <= 18 for s in talk)  # characters speak short lines in the scene
    assert next(s for s in a["shots"] if s["kind"] == "choice")["choice"]["a"]
    assert se.make_plan(CONTEXT, seed=6, voices=VOICES)["characters"] != a["characters"]


def test_the_opening_event_fits_the_place_and_both_languages_exist():
    plan = se.make_plan(CONTEXT, seed=5, voices=VOICES)
    assert all(l["kind"] == "spaceship" and "space" in l["tags"] for l in plan["locations"])
    assert "city" not in plan["shots"][0]["text"].lower()
    fr = se.make_plan(CONTEXT, {"language": "fr"}, seed=5, voices=VOICES)
    assert fr["language"] == "fr" and se.validate_plan(fr) == []
    assert [s["kind"] for s in fr["shots"]] == [s["kind"] for s in plan["shots"]]  # same structure in both languages
    assert fr["shots"][0]["text"] != plan["shots"][0]["text"]


def test_variants_of_a_big_place_point_to_an_existing_location():
    plan = se.make_plan(CONTEXT, seed=2, voices=VOICES)
    ids = {l["id"] for l in plan["locations"]}
    variants = [l for l in plan["locations"] if l["variant_of"]]
    assert variants and all(l["variant_of"] in ids for l in variants)
    broken = json.loads(json.dumps(plan))
    broken["locations"][1]["variant_of"] = "nowhere"
    assert any("unknown location" in p for p in se.validate_plan(broken))


def test_validator_catches_a_bad_plan():
    plan = se.make_plan(CONTEXT, seed=3, voices=VOICES)
    bad = json.loads(json.dumps(plan))
    talk = next(b for b in bad["beats"] if b["kind"] == "talk")
    talk["speaker"], talk["text"] = "narrator", "word " * 30
    problems = se.validate_plan(bad)
    assert any("speaker" in p for p in problems) and any("short" in p for p in problems)


def test_a_model_answer_is_validated_and_repaired_then_falls_back_to_the_template():
    good = se.make_plan(CONTEXT, seed=4, voices=VOICES)
    calls = []

    def repairing(prompt, schema):
        calls.append(prompt)
        plan = {k: good[k] for k in ("title", "logline", "language", "characters", "locations", "beats")}
        if len(calls) == 1:
            plan = json.loads(json.dumps(plan))
            plan["beats"] = [b for b in plan["beats"] if b["kind"] != "choice"]  # the first answer has no choice
        return plan

    plan = se.make_plan(CONTEXT, seed=4, llm_json=repairing, voices=VOICES)
    assert plan["story_source"] == "llm" and len(calls) == 2 and "no choice beat" in calls[1]
    broken = se.make_plan(CONTEXT, seed=4, llm_json=lambda p, s: {"title": "x"}, voices=VOICES)
    assert broken["story_source"].startswith("template") and se.validate_plan(broken) == []
    down = se.make_plan(CONTEXT, seed=4, llm_json=lambda p, s: (_ for _ in ()).throw(OSError("model down")), voices=VOICES)
    assert down["story_source"].startswith("template")


def test_long_narration_is_cut_into_shots_and_talk_lines_stay_whole():
    plan = {"characters": [{"id": "c1"}], "beats": [
        {"kind": "narration", "text": "One two three four five six seven eight. Nine ten eleven twelve thirteen fourteen fifteen sixteen. Seventeen eighteen.",
         "speaker": "narrator", "location": "x", "visual": "v", "camera": "wide"},
        {"kind": "talk", "text": "Come with me now, please.", "speaker": "c1", "location": "x", "visual": "v", "camera": "close"}]}
    shots = se.compile_shots(plan, max_words=9)
    assert len([s for s in shots if s["kind"] == "narration"]) == 3 and len([s for s in shots if s["kind"] == "talk"]) == 1
    assert shots[0]["camera"] == "wide" and shots[1]["camera"] in ("wide", "close", "medium")


def test_voices_are_distinct_and_the_narrator_gets_the_narrator_voice():
    plan = se.make_plan(CONTEXT, seed=5, voices=VOICES)
    assert plan["voices"]["narrator"] == "dan"
    speakers = [plan["voices"][c["id"]] for c in plan["characters"][:2]]
    assert len(set(speakers)) == 2 and "dan" not in speakers


def test_the_model_is_told_to_make_every_character_fictional_and_to_report_substitutions():
    seen = []

    def spy(prompt, schema):
        seen.append(prompt)
        raise OSError("stop")

    se.write_plan_llm("a politician offers help", se.resolve_params({}, 1)[0], spy)
    assert "fictional" in seen[0] and "substitutions" in seen[0]


# ---------------------------------------------------------------- styles
def test_style_is_regulated_deterministic_and_forces_the_void_behind_spaceship_windows():
    a, b = ss.style_from_context(CONTEXT, random.Random(1)), ss.style_from_context(CONTEXT, random.Random(1))
    assert a == b and ss.validate_style(a) == []
    assert "void of space" in ss.style_prompt(a, ["space", "window"]) and "void of space" not in ss.style_prompt(a, ["shelter"])
    assert a["lighting"] in ss.LIGHTING and "cartoon" in ss.negative_prompt(a)
    assert any("lighting" in p for p in ss.validate_style(dict(a, lighting="disco")))


def test_style_from_the_model_is_validated_and_repaired():
    good = ss.style_from_context(CONTEXT, random.Random(2))
    answers = [dict(good, grade="rainbow"), good]
    calls = []

    def llm(prompt, schema):
        calls.append(prompt)
        return answers[len(calls) - 1]

    assert ss.style_via_llm(CONTEXT, llm)["name"] == good["name"] and len(calls) == 2 and "grade" in calls[1]
    assert ss.style_via_llm(CONTEXT, lambda p, s: dict(good, motif="")) is None


def test_library_keeps_styles_classifies_them_and_selects_only_good_reusable_ones(tmp_path):
    lib = ss.StyleLibrary(tmp_path / "lib")
    good = lib.add(ss.style_from_context(CONTEXT, random.Random(1)), origin="auto", context=CONTEXT)
    bad = lib.add(ss.style_from_context("a flooded city", random.Random(2)), origin="auto")
    assert lib.add(ss.style_from_context(CONTEXT, random.Random(1)), origin="auto") == good  # same style = same entry
    assert lib.get(good)["quality"] is None and lib.get(good)["seed_policy"] == "rng" and lib.get(good)["method"] == "prompt_only"
    assert lib.select() is None  # nothing tested yet: nothing is offered for reuse
    lib.record_test(good, location_kind="spaceship", images=["a.png"], consistency=0.9)
    assert lib.get(good)["reusable"] is None  # one kind of place is not enough evidence
    lib.record_test(good, location_kind="shelter", images=["b.png"], consistency=0.85)
    assert lib.get(good)["reusable"] is True and lib.get(good)["quality"] >= 0.8
    lib.record_test(bad, location_kind="city", images=[], consistency=0.95)
    lib.set_verdict(bad, "bad", "ugly")
    assert lib.get(bad)["reusable"] is False
    assert {lib.select(rng=random.Random(i)) for i in range(5)} == {good}
    assert lib.select(["space"]) == good and lib.select(["sea"]) is None
    assert ss.StyleLibrary(tmp_path / "lib").get(good)["tests"]  # persisted on disk, readable by a new session


def test_palette_consistency_is_high_for_the_same_look_and_low_for_different_looks(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for i, base in enumerate([(200, 120, 40), (190, 115, 45), (60, 90, 200)]):
        arr = np.clip(np.array(base) + rng.normal(0, 8, (96, 96, 3)), 0, 255).astype(np.uint8)
        paths.append(tmp_path / f"{i}.png")
        Image.fromarray(arr).save(paths[-1])
    assert ss.palette_consistency(paths[:2]) > 0.9
    assert ss.palette_consistency([paths[0], paths[2]]) < ss.palette_consistency(paths[:2])


def test_the_kind_of_place_comes_from_whole_words_and_a_short_video_is_short():
    flood = "A flooded city at night. Two strangers on a rescue boat each offer to save you: a ship captain and a doctor."
    assert se.kit_for(flood)[0] == "flood"  # 'ship captain' is not a spaceship
    assert se.kit_for("a relationship drama in an office")[0] == "city"
    assert se.kit_for("the moon is breaking apart")[0] == "spaceship"
    short = se.make_plan(flood, {"target_seconds": 20, "branches": 1}, seed=1, voices=VOICES)
    kinds = [s["kind"] for s in short["shots"]]
    assert se.validate_plan(short) == [] and kinds.count("twist") == 1 and kinds.count("choice") == 1 and kinds.count("talk") == 2
    assert 14 <= short["estimated_seconds"] <= 24
    full = se.make_plan(flood, {"target_seconds": 75, "branches": 2}, seed=1, voices=VOICES)
    assert full["estimated_seconds"] > short["estimated_seconds"] * 1.8 and [s["kind"] for s in full["shots"]].count("twist") == 2


def test_style_keywords_are_whole_words_and_a_flood_gets_the_sea_rule():
    flood = ss.style_from_context("A flooded city at night, a rescue boat", random.Random(1))
    assert "sea" in flood["tags"] and flood["grade"] == "cold steel blue"
    plain = ss.style_from_context("a relationship drama in an office", random.Random(1))
    assert "sea" not in plain["tags"]


def test_a_character_never_gets_a_voice_of_the_wrong_gender_and_gets_a_described_voice_instead():
    plan = {"characters": [{"id": "c1", "gender": "m", "age": 32, "role": "ship captain"}, {"id": "c2", "gender": "f", "age": 59, "role": "doctor"}]}
    voices = se.assign_voices(plan, VOICES, seed=1)
    assert voices["narrator"] == "dan" and voices["c2"] in ("siren", "jessica")
    assert voices["c1"].startswith("design:") and "man in his 30s" in voices["c1"] and "ship captain" in voices["c1"]  # no recorded male voice left: Dan is the narrator
    only_dan = se.assign_voices(plan, VOICES[:1], seed=1)
    assert only_dan["c1"].startswith("design:") and only_dan["c2"].startswith("design:") and "woman in her 50s" in only_dan["c2"]


def test_the_hour_of_the_story_beats_the_lighting_of_the_place_and_applies_to_every_shot():
    night = ss.style_from_context("A flooded city at night. A rescue boat.", random.Random(1))
    assert night["lighting"] == "cold moonlit" and "time" in night["forced_elements"]
    for tags in ([], ["sea"], ["window"]):
        assert "it is night in every shot" in ss.style_prompt(night, tags)
    day = ss.style_from_context("A flooded city. A rescue boat.", random.Random(1))
    assert day["lighting"] == "overcast flat daylight" and "time" not in day["forced_elements"]
    assert ss.style_from_context("The moon is breaking apart at dusk", random.Random(1))["lighting"] == "dusk blue hour"


def test_the_places_of_the_story_decide_its_world_before_a_stray_keyword():
    """auto_ab_1: a station story whose text says 'the last city lights' was styled as a CITY (olive, carved wood, dusk); its places (deck, corridor, airlock = space) must win."""
    context = "The observation deck of the Oort Station shudders as the last city lights on the darkened Earth flicker out."
    tags = ["space", "window", "shelter", "underground", "shelter", "underground", "space", "space"]
    style = ss.style_from_context(context, random.Random(0), tags=tags, hour="night", atmosphere="Silence, freezing, smell of recycled fear and ozone")
    assert "space" in style["tags"] and "city" not in style["tags"] and "void of space" in ss.style_prompt(style, ["space"])
    assert style["lighting"] == "cold moonlit" and "never daylight" in style["forced_elements"]["time"]  # the hour of the brief, not only the words of the context
    assert style["grade"] == "cold steel blue"  # the atmosphere says freezing
    assert not set(style["materials"]) & {"carved wood", "dusty fabric", "cracked stone", "wet asphalt"}  # metal and glass, not an old town
    assert "no daylight comes in" in ss.style_prompt(style, ["shelter"])  # every kind of place of the story keeps its rule, not only the winning one
    keyword_only = ss.style_from_context(context, random.Random(0))
    assert "city" in keyword_only["tags"]  # without places the old behaviour is kept


def test_a_shot_hour_is_a_change_only_when_it_is_daylight_or_another_hour():
    assert not ss.changes_the_hour("night", "night") and not ss.changes_the_hour("it is night: dark", "night") and not ss.changes_the_hour(None, "night")
    assert ss.changes_the_hour("it is dawn: golden light", "night") and ss.changes_the_hour("it is dusk", "night") and ss.changes_the_hour("daylight", "daylight")
