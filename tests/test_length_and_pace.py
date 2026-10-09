"""The test multiplier of the time of a shot (never the default), the length decided by the creation (2 to 10 minutes), a long outline planned by parts, long chunks of lines cut in smaller ones."""
import ast
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_agents as sa  # noqa: E402
import story_engine as se  # noqa: E402
import story_library as lib  # noqa: E402
import story_render as sr  # noqa: E402
import story_text_lab as lab  # noqa: E402
import story_writer as sw  # noqa: E402

TREND = sa.load_trend("you_must_choose")
CONTEXT = "A flooded city at night. Two strangers on a rescue boat each offer to save you: a ship captain and a doctor."
ENDINGS = {"A": "bad", "B": "good"}


# ------------------------------------------------------------------------------------------------ the test multiplier of the time of a shot
def test_the_budget_is_unchanged_by_default_and_the_test_scale_stacks_more_shorter_shots_in_the_same_time():
    assert TREND.budget(150, 2) == TREND.budget(150, 2, 1.0)
    assert TREND.budget(60, 2, 0.4) == TREND.budget(150, 2)  # a minute of shots of 0.4 holds as many shots as 2:30 of normal ones
    assert sum(TREND.budget(60, 2, 0.4).values()) >= 1.9 * sum(TREND.budget(60, 2).values())
    assert list(TREND.budget(60, 2, 0.4)) == list(TREND.ACT_PURPOSE)  # the acts stay in the order of the story


def test_the_lines_get_shorter_with_the_shots_and_never_longer_than_the_default():
    assert (TREND.max_words(1.0), TREND.max_offer_words(1.0)) == (14, 9)
    assert (TREND.max_words(0.4), TREND.max_offer_words(0.4)) == (6, 4)  # "Brandt offers a dry harbor." still fits
    assert TREND.max_words(2.0) == 14 and TREND.max_words(0.25) == 5 and TREND.max_offer_words(0.25) == 4


def test_the_silence_after_a_line_shrinks_with_the_test_scale_the_default_is_untouched_and_the_choice_stays_readable():
    assert sr.shot_seconds("narration", 1.0) == 1.4 and sr.shot_seconds("narration", 1.0, scale=0.4) == 1.16
    assert sr.tail_of("twist", 1.0) == 2.4 and sr.tail_of("twist", 0.4) == 0.96 and sr.tail_of("talk", 0.4) == 0.15 and sr.tail_of("choice", 0.4) == 0.0
    assert sr.shot_seconds("choice", 0.5, scale=0.4) == sr.CHOICE_MIN_SECONDS  # the countdown ring needs its time whatever the pace


def test_the_scale_and_the_length_mode_are_parameters_with_safe_defaults():
    params, provenance = se.resolve_params({}, seed=1)
    assert params["shot_scale"] == 1.0 and params["length_mode"] == "fixed" and provenance["shot_scale"] == "default" and provenance["length_mode"] == "default"
    params, provenance = se.resolve_params({"shot_scale": 0.4, "length_mode": "auto"}, seed=1)
    assert params["shot_scale"] == 0.4 and params["length_mode"] == "auto" and provenance["shot_scale"] == "given"
    for bad in ({"shot_scale": 0.1}, {"shot_scale": 5}, {"length_mode": "whatever"}):
        with pytest.raises(ValueError):
            se.resolve_params(bad, seed=1)


def test_the_validator_expects_as_many_shots_as_the_scale_says():
    assert sw.target_shots(60, 0.4) > 2 * sw.target_shots(60)
    assert sw.target_shots(150) == sw.target_shots(150, 1.0)


def test_a_stacked_test_video_has_a_shot_every_second_and_a_half_with_short_lines(tmp_path):
    plan = sa.make_plan(CONTEXT, {"target_seconds": 60, "shot_scale": 0.4}, seed=3, llm=None, library=lib.StoryLibrary(tmp_path / "library", seed=False), voices=[])
    assert plan["params"]["shot_scale"] == 0.4 and len(plan["shots"]) == sum(TREND.budget(60, 2, 0.4).values())
    assert all(len(s["text"].split()) <= TREND.max_words(0.4) for s in plan["shots"])
    assert not any("shots: about" in p for p in plan["agents"]["problems_left"])  # the checker knows the pace of the test too


# ------------------------------------------------------------------------------------------------ the length decided by the creation
def length_llm(*answers):
    calls = []

    def llm(prompt, schema, seed=0):
        calls.append(prompt)
        return answers[min(len(calls), len(answers)) - 1]
    llm.calls = calls
    return llm


GOOD = {"events": ["the flooded streets", "the boat", "the two offers", "a secret", "a cure"], "why": "it needs this much time"}


@pytest.mark.parametrize("asked,expected", [(50, 120), (119, 120), (187, 190), (300, 300), (601, 600), (900, 600)])
def test_the_length_agent_is_clamped_to_two_and_ten_minutes_and_rounded_to_ten_seconds(asked, expected):
    seconds, report = sa.run_length(CONTEXT, {"seed": 1}, TREND, length_llm({**GOOD, "seconds": asked}))
    assert seconds == expected and report["asked"] == asked and report["source"] == "llm" and report["events"] == GOOD["events"]


def test_a_length_answer_without_events_or_reason_is_sent_back_and_the_second_try_is_kept():
    llm = length_llm({"events": ["one"], "seconds": 200, "why": ""}, {**GOOD, "seconds": 240})
    seconds, report = sa.run_length(CONTEXT, {"seed": 1}, TREND, llm)
    assert seconds == 240 and report["attempts"] == 2 and "Your previous answer had these problems" in llm.calls[1]


def test_the_length_prompt_says_what_a_shot_and_a_minute_cost_in_shots_and_the_test_scale_changes_it():
    llm = length_llm({**GOOD, "seconds": 200})
    sa.run_length(CONTEXT, {"seed": 1}, TREND, llm)
    assert "{{" not in llm.calls[0] and "A shot lasts about 3.8 seconds, so 2:00 is about 32 shots and every additional minute adds about 16 shots" in llm.calls[0]
    assert "Never less than 120 seconds, never more than 600 seconds" in llm.calls[0] and CONTEXT in llm.calls[0]
    llm = length_llm({**GOOD, "seconds": 200})
    sa.run_length(CONTEXT, {"seed": 1, "shot_scale": 0.4}, TREND, llm)
    assert "A shot lasts about 1.5 seconds, so 2:00 is about 79 shots" in llm.calls[0]


def test_without_a_model_the_length_follows_the_size_of_the_context_inside_the_bounds():
    sentences = lambda n: " ".join(f"This is sentence number {i} of the context." for i in range(n))  # noqa: E731
    lengths = [sa.fallback_length(sentences(n), TREND) for n in (2, 4, 6, 9, 14, 40)]
    assert lengths == sorted(lengths) and lengths[0] == 120 and lengths[1] == 120 and lengths[-1] == 600 and all(120 <= x <= 600 and x % 10 == 0 for x in lengths)
    seconds, report = sa.run_length(CONTEXT, {"seed": 1}, TREND, None)
    assert report["source"] == "template" and 120 <= seconds <= 600


def test_a_longer_video_asks_the_analyst_for_more_places_and_the_default_is_unchanged():
    assert [TREND.locations_asked(150, n) for n in (4, 8, 12)] == ["5 or 6", "6 or 7", "7 or 8"]  # exactly what was asked before the auto length existed
    assert TREND.locations_asked(120, 4) == "5 or 6" and TREND.locations_asked(300, 4) == "7 or 8" and TREND.locations_asked(600, 12) == "10 or 11"


# ------------------------------------------------------------------------------------------------ a long outline is planned by parts
def fake_beats(counts: dict) -> list[dict]:
    """An outline that obeys every rule of the planner for the given acts (what a cooperative model would write)."""
    beats = []
    for act, n in counts.items():
        for i in range(n):
            kind = {"hook": "narration", "setup": "narration", "offers": "offer", "choice": "choice", "rewind": "rewind"}.get(act)
            speaker = "narrator"
            if act == "offers":
                speaker = f"c{i + 1}"
            elif act in ("branch_a", "branch_b"):
                kind = "twist" if i == n - 1 else "pov" if i % 2 == 0 else "narration"
            branch = "main" if act in ("hook", "setup", "offers", "choice") else "B" if act in ("rewind", "branch_b") else "A"
            beats.append({"act": act, "branch": branch, "kind": kind, "speaker": speaker, "purpose": f"{act} step {i + 1}: something specific and visible happens here"})
    return beats


class PartPlanner:
    """Answers the planner from the structure written in the prompt; remembers the prompts."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, schema, seed=0):
        self.prompts.append(prompt)
        counts = {act: int(n) for act, n in re.findall(r"^- (\w+): (\d+) beats?$", prompt, flags=re.M)}
        return {"hook_title": "POV: THE LAST HARBOR\nYOU MUST CHOOSE", "beats": fake_beats(counts), "ending_label_a": "COLLECTED", "ending_label_b": "THE CURE", "closing_question": "WHICH ENDING\nDID YOU GET?",
                "caption": "A story #pov #flood #choice #apocalypse #survival #twist"}


def test_a_video_of_ten_minutes_is_planned_in_three_parts_each_told_what_was_planned_before():
    brief = sa.heuristic_brief(CONTEXT, {"seed": 1}, TREND)
    counts = TREND.budget(600, 2)
    assert sum(counts.values()) > TREND.PLANNER_SPLIT_BEATS
    model = PartPlanner()
    outline, report = sa.run_planner(brief, counts, ENDINGS, {"seed": 1, "language": "en"}, TREND, model)
    assert report["source"] == "llm" and [p["acts"] for p in report["parts"]] == [["hook", "setup", "offers", "choice"], ["branch_a"], ["rewind", "branch_b"]]
    assert len(model.prompts) == 3 and len(outline["beats"]) == sum(counts.values())
    assert sa.validate_outline(outline, counts, ENDINGS, TREND) == []  # the merged outline passes the checks of a whole outline: order, offers, choice, twists, header and footer fields
    assert outline["hook_title"].startswith("POV") and outline["closing_question"].count("\n") == 1 and outline["ending_label_a"] == "COLLECTED" and outline["ending_label_b"] == "THE CURE"
    assert "PART 1 of 3" in model.prompts[0] and "(none, this part starts the video)" in model.prompts[0]
    assert "PART 2 of 3" in model.prompts[1] and "BEATS ALREADY PLANNED (continue the story from them" in model.prompts[1] and "hook step 1" in model.prompts[1]
    assert "PART 3 of 3" in model.prompts[2] and "branch_a step 1" in model.prompts[2] and "- rewind: 1 beat" in model.prompts[2]
    assert "- branch_a:" not in model.prompts[0].split("RULES")[0]  # a part lists only its own acts in the structure


def test_a_part_the_model_never_gets_right_falls_back_to_the_template_for_that_part_only():
    brief = sa.heuristic_brief(CONTEXT, {"seed": 1}, TREND)
    counts = TREND.budget(600, 2)
    good = PartPlanner()

    def model(prompt, schema, seed=0):
        if "PART 2 of 3" in prompt:
            return {"hook_title": "x", "beats": [], "ending_label_a": "", "ending_label_b": "", "closing_question": "x", "caption": "x"}  # never valid
        return good(prompt, schema, seed)

    outline, report = sa.run_planner(brief, counts, ENDINGS, {"seed": 1, "language": "en"}, TREND, model)
    assert [p["source"] for p in report["parts"]] == ["llm", "template", "llm"] and report["source"] == "partly template"
    assert len(outline["beats"]) == sum(counts.values()) and sa.validate_outline(outline, counts, ENDINGS, TREND) == [] and outline["ending_label_a"]


def test_a_normal_video_is_still_planned_in_one_answer_and_without_a_model_by_the_template():
    brief = sa.heuristic_brief(CONTEXT, {"seed": 1}, TREND)
    counts = TREND.budget(150, 2)
    model = PartPlanner()
    outline, report = sa.run_planner(brief, counts, ENDINGS, {"seed": 1, "language": "en"}, TREND, model)
    assert len(model.prompts) == 1 and "parts" not in report and "PART" not in model.prompts[0] and len(outline["beats"]) == sum(counts.values())
    outline, report = sa.run_planner(brief, TREND.budget(600, 2), ENDINGS, {"seed": 1, "language": "en"}, TREND, None)
    assert report["source"] == "template" and len(outline["beats"]) == sum(TREND.budget(600, 2).values())


def test_the_tolerance_on_the_number_of_beats_grows_with_a_long_outline_and_stays_the_same_for_a_short_one():
    brief = sa.heuristic_brief(CONTEXT, {"seed": 1}, TREND)
    short, long = TREND.budget(150, 2), TREND.budget(600, 2)
    outline = {"hook_title": "POV: X\nY", "beats": fake_beats(short), "ending_label_a": "A", "ending_label_b": "B", "closing_question": "A\nB", "caption": "#a #b #c #d #e"}
    assert sa.validate_outline(outline, short, ENDINGS, TREND) == []
    outline["beats"] = outline["beats"][:-8]  # 8 beats missing from 40 is too many ...
    assert any("beats in total" in p for p in sa.validate_outline(outline, short, ENDINGS, TREND))
    big = {**outline, "beats": fake_beats(long)}
    big["beats"] = [b for i, b in enumerate(big["beats"]) if i % 40 != 7]  # ... 4 missing from 158 is within the tolerance of a long one
    assert not any("beats in total" in p for p in sa.validate_outline(big, long, ENDINGS, TREND))


# ------------------------------------------------------------------------------------------------ long chunks of lines are cut
def test_a_long_story_is_written_in_chunks_of_at_most_28_beats_and_a_short_one_in_the_usual_three():
    short, long = [{"act": a} for a, n in TREND.budget(150, 2).items() for _ in range(n)], [{"act": a} for a, n in TREND.budget(600, 2).items() for _ in range(n)]
    assert len(sa.chunks_of(short, TREND.WRITER_MAX_CHUNK)) == 3 == len(sa.chunks_of(short))
    chunks = sa.chunks_of(long, TREND.WRITER_MAX_CHUNK)
    assert len(chunks) > 3 and max(len(c) for c in chunks) <= TREND.WRITER_MAX_CHUNK and [i for c in chunks for i in c] == list(range(len(long)))
    first, second, third = {"hook", "setup", "offers", "choice"}, {"branch_a"}, {"rewind", "branch_b"}
    for chunk in chunks:
        acts = {long[i]["act"] for i in chunk}
        assert acts <= first or acts <= second or acts <= third  # a chunk never mixes the story before the choice with a branch


def test_only_the_last_chunk_of_a_branch_is_checked_for_its_ending_and_the_others_are_open():
    brief = sa.heuristic_brief(CONTEXT, {"seed": 1}, TREND)
    counts = TREND.budget(600, 2)
    beats = fake_beats(counts)
    seen = {"verifier": 0, "moods": []}
    place = [loc["id"] for loc in brief["locations"]]

    def model(prompt, schema, seed=0):
        stage = re.search(r"\[\[STAGE:(\w+)\]\]", prompt).group(1)
        if stage == "verifier":
            seen["verifier"] += 1
            return {"polarity": "bad" if seen["verifier"] == 1 else "good", "reason": "canned", "suspense": True}
        if stage == "clarity":
            return {"confusing": []}
        assert stage == "writer"
        seen["moods"].append("open" if TREND.OPEN_MOOD in prompt else "ending")
        count = int(re.search(r"BEATS TO WRITE \((\d+)\)", prompt).group(1))
        start = len(seen.setdefault("lines", []))
        seen["lines"] += [None] * count
        return {"lines": [{"text": f"Step {start + i} goes on", "location": place[0], "in_shot": []} for i in range(count)]}

    lines, report = sa.run_writer(brief, {"beats": beats}, {"seed": 1, "language": "en", "target_seconds": 600}, TREND, model, "", ENDINGS)
    chunks = sa.chunks_of(beats, TREND.WRITER_MAX_CHUNK)
    assert len(lines) == len(beats) and len(report["chunks"]) == len(chunks) and all(c["source"] == "llm" for c in report["chunks"])
    assert seen["verifier"] == 2  # one question about the ending per branch, asked about the chunk that holds its last beat
    assert seen["moods"].count("ending") == 2 and seen["moods"].count("open") == len(chunks) - 2


# ------------------------------------------------------------------------------------------------ the text lab's length summary
def fake_story(seed: int, chosen: int | None, shots: int = 50):
    agents = {"length": {"chosen": chosen, "asked": chosen + 3, "source": "llm", "why": "because", "events": ["a", "b"]}} if chosen else {}
    return {"seed": seed, "seconds": 300, "plan": {"title": f"T{seed}", "params": {"shot_scale": 1.0}, "shots": [{}] * shots, "estimated_seconds": 140.0, "agents": agents}}


def test_the_lab_sums_up_the_lengths_the_creation_decided_and_checks_the_promised_bounds():
    summary = lab.length_summary([fake_story(1, 180), fake_story(2, 420), fake_story(3, None)])  # a story with a fixed length is not part of it
    assert len(summary["rows"]) == 2 and (summary["min"], summary["max"], summary["mean"]) == (180, 420, 300)
    assert summary["within_bounds"] and summary["mean_at_least_2m30"] and lab.mmss(300) == "5:00" and lab.mmss(125) == "2:05"
    assert not lab.length_summary([fake_story(1, 100), fake_story(2, 140)])["within_bounds"] and not lab.length_summary([fake_story(1, 120), fake_story(2, 140)])["mean_at_least_2m30"]
    assert lab.length_summary([fake_story(1, None)]) is None
    text = "\n".join(lab.length_lines(summary))
    assert "average 5:00" in text and "| 1 | T1 | 3:00 | 50 |" in text and "key events it counted: a; b" in text


def test_the_scripts_of_the_cli_accept_the_two_new_flags():
    for name in ("story_agents", "story_auto", "story_text_lab"):
        source = (Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        ast.parse(source)
        assert '"--shot-scale"' in source and ("--auto-length" in source)
