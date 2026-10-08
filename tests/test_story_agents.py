import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_agents as sa  # noqa: E402
import story_library as lib  # noqa: E402

REFERENCE = Path(__file__).resolve().parent.parent / "results" / "story_trend" / "plans" / "the_last_city_150_en.json"
CONTEXT = "A flooded city at night. Two strangers on a rescue boat each offer to save you: a ship captain and a doctor."
pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="needs the hand-written reference plan")
TREND = sa.load_trend("you_must_choose")


class CannedModel:
    """A stand-in for the model that answers every stage from the hand-written reference story (so the MECHANICS of the chain are tested without a GPU)."""

    def __init__(self, fail_first_writer: bool = False):
        self.plan = json.loads(REFERENCE.read_text(encoding="utf-8"))
        self.calls: list[str] = []
        self.fail_first_writer = fail_first_writer
        self.acts = self.acts_of(self.plan["shots"])
        self.by_text = {s["text"]: s for s in self.plan["shots"]}

    @staticmethod
    def acts_of(shots: list[dict]) -> list[str]:
        """The act of every shot of the reference story, read from its structure (3 hook shots, 3 of setup, then the offers up to the choice, branch A up to its twist, the rewind, branch B)."""
        acts, phase = [], "offers"
        for index, shot in enumerate(shots):
            if shot["kind"] == "choice":
                acts.append("choice")
                phase = "branch_a"
            elif shot["kind"] == "rewind":
                acts.append("rewind")
                phase = "branch_b"
            else:
                acts.append("hook" if index < 3 else "setup" if index < 6 else phase)
        return acts

    @classmethod
    def counts(cls, shots: list[dict]) -> dict:
        acts = cls.acts_of(shots)
        return {act: acts.count(act) for act in TREND.ACT_PURPOSE if act in acts}

    def __call__(self, prompt: str, schema: dict, seed: int = 0) -> dict:
        stage = re.search(r"\[\[STAGE:(\w+)\]\]", prompt).group(1)
        self.calls.append(stage)
        shots = self.plan["shots"]
        if stage == "analyst":
            characters = [{**{k: c[k] for k in ("id", "name", "role", "gender", "age", "look", "wardrobe")}, "public_promise": "I can save you", "hidden_truth": "is hiding what really happens to the people they save",
                           "voice_style": "calm"} for c in self.plan["characters"]]
            return {"title": "The last city", "substitutions": [], "world": {"setting": "a drowned megacity", "premise": "the flood has covered the world", "hour": "night", "atmosphere": "wet, dark",
                                                                          "scale_image": "a colossal aerial view of a glowing megacity under a storm"},
                    "locations": [{"id": l["id"], "description": l["description"], "tags": l["tags"]} for l in self.plan["locations"]], "viewer": "a survivor", "stakes": "your life",
                    "characters": characters, "keywords": ["flood", "last city", "rescue boat", "captain", "doctor", "apocalypse", "survival"], "must_include": ["a ship captain", "a doctor"]}
        if stage == "planner":
            beats = []
            for act, shot in zip(self.acts, shots):
                branch = "B" if act == "rewind" else ("main" if act in ("hook", "setup", "offers", "choice") else "A" if act == "branch_a" else "B")
                beats.append({"act": act, "branch": branch, "kind": shot["kind"], "speaker": shot["speaker"], "purpose": f"the narrator says: {shot['text']}"})
            return {"hook_title": self.plan["title_overlay"]["text"], "beats": beats, "ending_label_a": "COLLECTED", "ending_label_b": "THE CURE", "closing_question": "WHICH ENDING\nDID YOU GET?",
                    "caption": self.plan["caption"]}
        if stage == "writer":
            purposes = re.findall(r"^\d+\. \[[^\]]+\] the narrator says: (.*)$", prompt, flags=re.M)
            if self.fail_first_writer and self.calls.count("writer") == 1:
                return {"lines": [{"text": "too long " * 20, "location": "boat", "in_shot": []} for _ in purposes]}
            return {"lines": [{"text": self.by_text[t]["text"], "location": self.by_text[t]["location"], "in_shot": self.by_text[t]["in_shot"]} for t in purposes]}
        if stage == "director":
            texts = re.findall(r"^\d+\. \[[^\]]+\] (.*)$", prompt.split("SHOTS (")[1], flags=re.M)
            return {"directions": [{"still": self.by_text[t].get("still") or "", "motion": self.by_text[t]["motion"], "camera": self.by_text[t]["camera"], "fx": {"zoom": 0.05, "shake": 0.0, "flash": False}} for t in texts]}
        if stage == "judge":
            return {"hook": 8, "coherence": 8, "clues": 7, "twist": 8, "voice": 7, "faithfulness": 9, "variety": 8, "weakness": "none"}
        raise AssertionError(stage)


@pytest.fixture()
def library(tmp_path):
    return lib.StoryLibrary(tmp_path / "library", seed=False)


def test_the_chain_with_a_cooperative_model_gives_a_valid_plan_with_both_endings_and_every_stage_from_the_model(library, monkeypatch):
    model = CannedModel()
    monkeypatch.setattr(TREND, "budget", lambda seconds, branches: CannedModel.counts(model.plan["shots"]))
    plan = sa.make_plan(CONTEXT, {"target_seconds": 150, "endings": {"A": "bad", "B": "good"}}, seed=1, llm=model, library=library, voices=[], judge=True)
    assert plan["agents"]["problems_left"] == [] and plan["story_source"] == "chain: analyst=llm, planner=llm, writer=llm, director=llm"
    assert len(plan["shots"]) == 46 and [s["kind"] for s in plan["shots"]].count("twist") == 2 and any(s["kind"] == "rewind" for s in plan["shots"])
    twists = [s for s in plan["shots"] if s["kind"] == "twist"]
    assert [t["ending"] for t in twists] == ["bad", "good"] and twists[0]["tag"] == "ENDING A: COLLECTED" and twists[1]["tag"] == "ENDING B: THE CURE"
    assert plan["shots"][0]["id"] == "s001" and next(s for s in plan["shots"] if s["kind"] == "choice")["choice"] == {"a": "Brandt", "b": "Ilse"}
    assert sum(1 for s in plan["shots"] if s.get("tag", "").startswith("CASE")) == 2
    assert model.calls[:2] == ["analyst", "planner"] and model.calls.count("writer") == 3 and model.calls.count("director") == 4 and model.calls[-1] == "judge"  # 3 chunks of lines, 46 shots in chunks of 12
    assert plan["agents"]["judge"]["mean"] > 7 and set(plan["voices"]) == {"narrator", "c1", "c2"}


def test_a_chunk_of_lines_that_breaks_the_rules_is_sent_back_with_the_problems_and_repaired(library, monkeypatch):
    model = CannedModel(fail_first_writer=True)
    monkeypatch.setattr(TREND, "budget", lambda seconds, branches: CannedModel.counts(model.plan["shots"]))
    plan = sa.make_plan(CONTEXT, {"target_seconds": 150, "endings": {"A": "bad", "B": "good"}}, seed=1, llm=model, library=library, voices=[])
    assert plan["agents"]["writer"]["chunks"][0]["attempts"] == 2 and plan["agents"]["problems_left"] == [] and model.calls.count("writer") == 4


def test_without_a_model_every_stage_falls_back_to_its_template_and_the_plan_is_still_complete(library):
    plan = sa.make_plan(CONTEXT, {"target_seconds": 60}, seed=3, llm=None, library=library, voices=[])
    assert plan["agents"]["analyst"]["source"] == "template" and plan["agents"]["planner"]["source"] == "template" and "partly template" in plan["story_source"]
    assert plan["shots"] and plan["characters"] and plan["title_overlay"]["text"].count("\n") == 1 and all(s["id"].startswith("s") and s["motion"] for s in plan["shots"])
    assert any(s["kind"] == "choice" for s in plan["shots"]) and sum(1 for s in plan["shots"] if s["kind"] == "twist") == 2


def test_the_prompts_are_constant_only_the_slots_change_and_a_missing_slot_is_an_error():
    template = TREND.ANALYST_PROMPT
    one = sa.fill(template, {"context": "AAA", "language": "English", "tone": "tense", "hours": TREND.HOURS, "location_tags": TREND.LOCATION_TAGS})
    two = sa.fill(template, {"context": "BBB", "language": "French", "tone": "bleak", "hours": TREND.HOURS, "location_tags": TREND.LOCATION_TAGS})
    assert one.replace("AAA", "X").replace("English", "L").replace("tense", "T") == two.replace("BBB", "X").replace("French", "L").replace("bleak", "T")  # the same text
    assert "{{" not in one
    with pytest.raises(KeyError):
        sa.fill(template, {"context": "x"})
    fingerprint = sa.prompt_fingerprint(TREND)
    assert fingerprint["version"] == TREND.PROMPT_VERSION and len(set(fingerprint.values())) == len(fingerprint)


def test_the_rhythm_pass_clamps_effects_adds_shake_on_impacts_and_breaks_three_equal_camera_moves():
    shots = [{"kind": "narration", "camera": "wide", "text": "the tower collapses", "motion": "a crash", "fx": {"zoom": 0.5, "shake": 3, "flash": 0}},
             {"kind": "narration", "camera": "wide", "text": "calm", "motion": "slow", "fx": {}}, {"kind": "narration", "camera": "wide", "text": "calm", "motion": "slow", "fx": {}},
             {"kind": "twist", "camera": "wide", "text": "end", "motion": "slow", "fx": {}}]
    changes = sa.rhythm_pass(shots, TREND)
    assert shots[0]["fx"]["zoom"] == 0.08 and shots[0]["fx"]["shake"] == 1.0 and shots[3]["fx"]["flash"] is True and shots[2]["camera"] != "wide" and changes["camera_changed"] >= 1


def test_the_library_gives_the_most_similar_approved_story_and_never_an_unapproved_one(library):
    plan = json.loads(REFERENCE.read_text(encoding="utf-8"))
    library.add(plan, approved=True, author="claude", entry_id="flood")
    other = {**plan, "title": "bunker", "caption": "#bunker #war", "context": "an underground bunker war", "characters": [{"role": "soldier"}, {"role": "nurse"}]}
    library.add(other, approved=False, entry_id="bunker")
    brief = {"keywords": ["flood", "rescue boat", "captain"], "world": {"setting": "a flooded city", "premise": "a flood"}, "characters": [{"role": "captain"}, {"role": "doctor"}]}
    similar = library.similar(brief, k=2)
    assert [e["id"] for e in similar] == ["flood"]  # the unapproved one is never used
    text = library.examples_text(brief, k=2)
    assert "Day nine of the great flood" in text and "talk/c1" in text and "still:" in library.director_examples_text(brief, k=1)


def test_the_acts_always_come_in_the_order_of_the_story_in_the_budget_the_prompt_and_the_template_outline():
    """A bug found by the first real run: the budget listed choice and rewind LAST, the planner was told that order and the checker expected the other one, so it never succeeded."""
    counts = TREND.budget(150, 2)
    assert list(counts) == ["hook", "setup", "offers", "choice", "branch_a", "rewind", "branch_b"]
    assert [line.split(":")[0].lstrip("- ") for line in sa.structure_text(counts).splitlines()] == list(counts)
    brief = sa.heuristic_brief(CONTEXT, {"seed": 1, "language": "en", "tone": "tense"}, TREND)
    endings = {"A": "bad", "B": "good"}
    outline = sa.template_outline(brief, counts, endings, {"seed": 1, "language": "en"}, TREND)
    assert sa.validate_outline(outline, counts, endings, TREND) == []  # the fallback outline satisfies the very checker that rejects the model's
    assert list(TREND.budget(60, 1)) == ["hook", "setup", "offers", "choice", "branch_a"]


def test_a_talking_shot_needs_a_motion_too_so_the_director_is_asked_again():
    shots = [{"kind": "talk", "text": "x"}, {"kind": "narration", "text": "y"}]
    answer = {"directions": [{"still": "", "motion": "", "camera": "close", "fx": {"zoom": 0.05, "shake": 0, "flash": False}},
                             {"still": "a wide photograph of a flooded street at night with a boat and a light, rain falling hard", "motion": "slow push-in on the boat, rain, handheld", "camera": "wide",
                              "fx": {"zoom": 0.05, "shake": 0, "flash": False}}]}
    problems = sa.validate_directions(answer, shots, TREND)
    assert len(problems) == 1 and problems[0].startswith("shot 1: a motion")
