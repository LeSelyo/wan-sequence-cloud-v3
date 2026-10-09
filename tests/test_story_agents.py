import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_agents as sa  # noqa: E402
import story_library as lib  # noqa: E402
import reference_story  # noqa: E402

REFERENCE = Path(__file__).resolve().parent.parent / "results" / "story_trend" / "plans" / "the_last_city_150_en.json"
CONTEXT = "A flooded city at night. Two strangers on a rescue boat each offer to save you: a ship captain and a doctor."
pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="needs the hand-written reference plan")
TREND = sa.load_trend("you_must_choose")


class CannedModel:
    """A stand-in for the model that answers every stage from the hand-written reference story (so the MECHANICS of the chain are tested without a GPU)."""

    def __init__(self, fail_first_writer: bool = False):
        self.plan = reference_story.load()
        self.calls: list[str] = []
        self.fail_first_writer = fail_first_writer
        self.acts = self.acts_of(self.plan["shots"])
        self.by_text = {s["text"]: s for s in self.plan["shots"]}

    @staticmethod
    def acts_of(shots: list[dict]) -> list[str]:
        """The act of every shot of the reference story, read from its structure (3 hook shots, 3 of setup, then the offers up to the choice, branch A up to its twist, the rewind, branch B)."""
        acts, phase = [], "setup"
        for index, shot in enumerate(shots):
            if shot["kind"] == "choice":
                acts.append("choice")
                phase = "branch_a"
            elif shot["kind"] == "rewind":
                acts.append("rewind")
                phase = "branch_b"
            elif shot["kind"] == "offer":
                acts.append("offers")
            else:
                acts.append("hook" if index < 3 else phase)
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
                beats.append({"act": act, "branch": branch, "kind": shot["kind"], "speaker": shot.get("offer_of") or shot["speaker"], "purpose": f"the narrator says: {shot['text']}"})
            return {"hook_title": self.plan["title_overlay"]["text"], "beats": beats, "ending_label_a": "COLLECTED", "ending_label_b": "THE CURE", "closing_question": "WHICH ENDING\nDID YOU GET?",
                    "caption": self.plan["caption"]}
        if stage == "writer":
            purposes = re.findall(r"^\d+\. \[[^\]]+\] the narrator says: (.*)$", prompt, flags=re.M)
            if self.fail_first_writer and self.calls.count("writer") == 1:
                return {"lines": [{"text": "too long " * 20, "location": "boat", "in_shot": []} for _ in purposes]}
            return {"lines": [{"text": self.by_text[t]["text"], "location": self.by_text[t]["location"], "in_shot": self.by_text[t]["in_shot"]} for t in purposes]}
        if stage == "director":
            texts = re.findall(r"^\d+\. \[[^\]]+\] (.*)$", prompt.split("SHOTS (")[1], flags=re.M)
            return {"directions": [{"n": i + 1, "still": self.by_text[t].get("still") or "", "motion": self.by_text[t]["motion"], "camera": self.by_text[t]["camera"], "fx": {"zoom": 0.05, "shake": 0.0, "flash": False}}
                                   for i, t in enumerate(texts)]}
        if stage == "clarity":
            return {"confusing": []}
        if stage == "verifier":
            lines = prompt.lower()
            return {"polarity": "bad" if ("collected" in lines or "never rescued" in lines) else "good", "reason": "canned"}
        if stage == "idea":
            roles = re.search(r"a (.+?) and a (.+?) \| tone", prompt)
            first, second = roles.groups() if roles else ("fairy", "elf thief")  # when the model invents the two characters the prompt gives no roles
            return {"title": "Flood", "context": f"The water is rising fast at night and the old city bells ring by themselves. You are alone on a roof and the whole city is lost beneath the waves. Two strangers each offer to save you: a {first} and a {second}, and each one promises a boat. The {first} glows softly and speaks of a hidden harbor. The {second} grins and speaks of a secret canal. A small light moves far away on the black water. You have until the tide reaches the bell tower to decide."}
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
    one = sa.fill(template, {"context": "AAA", "language": "English", "tone": "tense", "hours": TREND.HOURS, "location_tags": TREND.LOCATION_TAGS, "context_size": "4 sentences, 60 words", "n_locations": "5 or 6"})
    two = sa.fill(template, {"context": "BBB", "language": "French", "tone": "bleak", "hours": TREND.HOURS, "location_tags": TREND.LOCATION_TAGS, "context_size": "4 sentences, 60 words", "n_locations": "5 or 6"})
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
    assert shots[0]["fx"]["zoom"] == 0.08 and shots[0]["fx"]["shake"] == 0.7 and shots[3]["fx"]["flash"] is True and shots[2]["camera"] != "wide" and changes["camera_changed"] >= 1


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
    answer = {"directions": [{"n": 1, "still": "", "motion": "", "camera": "close", "fx": {"zoom": 0.05, "shake": 0, "flash": False}},
                             {"n": 2, "still": "a wide photograph of a flooded street at night with a boat and a light, rain falling hard", "motion": "slow push-in on the boat, rain, handheld", "camera": "wide",
                              "fx": {"zoom": 0.05, "shake": 0, "flash": False}}]}
    problems = sa.validate_directions(answer, shots, TREND)
    assert len(problems) == 1 and problems[0].startswith("shot 1: a motion")


def test_from_nothing_the_idea_agent_draws_the_bones_by_seed_and_invents_the_context(library, monkeypatch):
    model = CannedModel()
    monkeypatch.setattr(TREND, "budget", lambda seconds, branches: CannedModel.counts(model.plan["shots"]))
    plan = sa.make_plan(None, {"target_seconds": 150, "endings": {"A": "bad", "B": "good"}}, seed=7, llm=model, library=library, voices=[])
    assert model.calls[0] == "idea" and model.calls[1] == "analyst" and plan["provenance"]["context"] == "rng" and plan["idea"]["source"] == "llm" and plan["context"]
    first = sa.draw_seed_elements({"seed": 7}, TREND)
    assert first == sa.draw_seed_elements({"seed": 7}, TREND) and plan["idea"]["bones"] == first
    assert len({json.dumps(sa.draw_seed_elements({"seed": n}, TREND), sort_keys=True) for n in range(30)}) >= 20  # different seeds, different worlds
    nothing = sa.make_plan(None, {"target_seconds": 60}, seed=7, llm=None, library=library, voices=[])
    assert nothing["idea"]["source"] == "template" and first["role_a"] in nothing["context"] and first["role_b"] in nothing["context"] and nothing["shots"]


def test_a_branch_that_ends_the_wrong_way_is_sent_back_to_the_writer_with_the_reason(library, monkeypatch):
    class WrongEnding(CannedModel):
        def __call__(self, prompt, schema, seed=0):
            answer = super().__call__(prompt, schema, seed)
            stage = re.search(r"\[\[STAGE:(\w+)\]\]", prompt).group(1)
            if stage == "writer" and "it ends GOOD for you" in prompt and not getattr(self, "spoiled", False):
                self.spoiled = True  # the first answer for the good branch ends in death: the verifier says bad
                answer["lines"][-1]["text"] = "You were collected."
                answer["lines"][-2]["text"] = "You were never rescued."
            return answer

    model = WrongEnding()
    monkeypatch.setattr(TREND, "budget", lambda seconds, branches: CannedModel.counts(model.plan["shots"]))
    plan = sa.make_plan(CONTEXT, {"target_seconds": 150, "endings": {"A": "bad", "B": "good"}}, seed=1, llm=model, library=library, voices=[])
    chunk = plan["agents"]["writer"]["chunks"][2]
    assert getattr(model, "spoiled", False) and chunk["attempts"] == 2 and chunk["source"] == "llm" and "polarity" not in json.dumps(chunk)
    assert model.calls.count("verifier") >= 3


def test_the_writer_is_asked_for_short_lines_on_average_and_told_when_a_chunk_is_too_long():
    beats = [{}, {}, {}]
    long_lines = {"lines": [{"text": "one two three four five six seven eight nine ten eleven", "location": "boat", "in_shot": []} for _ in beats]}
    problems = sa.validate_lines(long_lines, beats, ["boat"], 14, avg_words=7)
    assert len(problems) >= 1 and "too long on average" in problems[0]
    ok = {"lines": [{"text": f"short line {n}", "location": "boat", "in_shot": []} for n in range(3)]}
    assert sa.validate_lines(ok, beats, ["boat"], 14, avg_words=7) == []


def test_a_picture_description_never_keeps_text_collage_or_panels_even_as_a_negation():
    for dirty in ("a boat at night, not a collage, rain falling", "a city, no text, split screen, with letters on the wall", "wide view of the sea with a caption and a logo, dramatic"):
        cleaned = sa.clean_still(dirty, TREND)
        assert not any(word in cleaned.lower() for word in TREND.FORBIDDEN_IN_PICTURES) and cleaned and not cleaned.endswith(("with", "and", "a"))
    assert sa.clean_still("a plain clear sentence about a boat", TREND) == "a plain clear sentence about a boat"
    assert sa.clean_still("a boat at night, not a collage, rain falling", TREND) == "a boat at night, rain falling"


def test_the_look_of_the_animation_comes_from_the_world_of_the_story_not_from_a_fixed_rain():
    import story_produce as sp
    plan = {"brief": {"world": {"atmosphere": "dry vacuum, violet glow", "hour": "night"}}}
    assert "rain" not in sp.look_of(plan) and "violet glow" in sp.look_of(plan) and "rain" not in sp.look_of({})


def outline_of(counts: dict, extra_offers: int = 2) -> dict:
    """A valid outline of the NEW structure: the offers act = the offer pair + narration, the branches with a third of first-person action, a twist closing each branch."""
    acts = [a for a in counts for _ in range(counts[a] + (extra_offers if a == "setup" else 0))]  # the context beats are in the setup act, the offers act is EXACTLY the two offers
    kinds = {"choice": "choice", "rewind": "rewind"}
    beats = [{"act": a, "branch": "B" if a in ("rewind", "branch_b") else "A" if a == "branch_a" else "main", "kind": kinds.get(a, "narration"), "speaker": "narrator", "purpose": "something happens here now"} for a in acts]
    pair = [b for b in beats if b["act"] == "offers"][:2]
    pair[0].update(kind="offer", speaker="c1")
    pair[1].update(kind="offer", speaker="c2")
    for act in ("branch_a", "branch_b"):
        own = [b for b in beats if b["act"] == act]
        for b in own[::2]:
            b["kind"] = "pov"
        own[-1]["kind"] = "twist"
    return {"hook_title": "POV: A\nB", "beats": beats, "ending_label_a": "X", "ending_label_b": "Y", "closing_question": "A\nB", "caption": "#a #b #c #d #e"}


def test_the_planner_is_not_rejected_for_a_few_beats_too_many_but_is_for_a_wrong_order():
    counts = TREND.budget(150, 2)
    outline = outline_of(counts)
    beats = outline["beats"]
    assert sa.validate_outline(outline, counts, {"A": "bad", "B": "good"}, TREND) == []
    outline["beats"] = beats[10:] + beats[:10]
    assert any("order" in p for p in sa.validate_outline(outline, counts, {"A": "bad", "B": "good"}, TREND))


def test_the_offers_act_is_exactly_the_pair_of_offers_and_the_choice_follows_at_once():
    counts = TREND.budget(150, 2)
    endings = {"A": "bad", "B": "good"}
    assert counts["offers"] == 2 and counts["setup"] > 3  # the context beats of the old offers act went to the setup
    outline = outline_of(counts)
    first_offer = next(i for i, b in enumerate(outline["beats"]) if b["kind"] == "offer")
    assert sa.validate_outline(outline, counts, endings, TREND) == []
    outline["beats"][first_offer + 1].update(kind="narration", speaker="narrator")
    assert any("exactly two consecutive beats of kind offer" in p for p in sa.validate_outline(outline, counts, endings, TREND))
    outline = outline_of(counts)
    outline["beats"].insert(first_offer + 2, {"act": "offers", "branch": "main", "kind": "narration", "speaker": "narrator", "purpose": "a clue after the offer which adds nothing"})
    problems = sa.validate_outline(outline, counts, endings, TREND)
    assert any("EXACTLY the two offer beats" in p for p in problems) and any("immediately after the second offer" in p for p in problems)
    outline = outline_of(counts)
    outline["beats"][first_offer].update(speaker="c2")
    outline["beats"][first_offer + 1].update(speaker="c1")
    assert any("c1 then c2" in p for p in sa.validate_outline(outline, counts, endings, TREND))


def test_a_branch_belongs_to_its_character_and_is_made_of_real_action():
    counts = TREND.budget(150, 2)
    endings = {"A": "bad", "B": "good"}
    outline = outline_of(counts)
    next(b for b in outline["beats"] if b["act"] == "branch_a" and b["kind"] == "narration").update(kind="talk", speaker="c2")
    assert any("branch_a" in p and "other character" in p for p in sa.validate_outline(outline, counts, endings, TREND))
    outline = outline_of(counts)
    for b in outline["beats"]:
        if b["act"] == "branch_b" and b["kind"] == "pov":
            b["kind"] = "narration"
    assert any("branch_b" in p and "first-person action" in p for p in sa.validate_outline(outline, counts, endings, TREND))


def test_the_lines_of_a_branch_never_name_the_other_character():
    brief = {"characters": [{"id": "c1", "name": "Brandt"}, {"id": "c2", "name": "Ilse"}]}
    beats = [{"act": "branch_a", "kind": "narration"}, {"act": "branch_b", "kind": "narration"}, {"act": "rewind", "kind": "rewind"}]
    lines = [{"text": "Far behind you, Ilse's light goes out."}, {"text": "Brandt screams your name."}, {"text": "What if you had chosen Ilse?"}]
    problems = sa.validate_independence(lines, beats, brief)
    assert len(problems) == 2 and "Ilse must not be named" in problems[0] and "Brandt must not be named" in problems[1]  # the rewind may name them


def test_the_offer_lines_are_short():
    offer = {"lines": [{"text": "I am Captain Brandt and I know a dry harbor now.", "location": "boat"}]}
    assert sa.validate_lines(offer, [{"kind": "offer"}], ["boat"], 14, None, 9) == ["line 1: 11 words, the limit is 9"]
    assert sa.validate_lines(offer, [{"kind": "narration"}], ["boat"], 14, None, 9) == []


def test_a_branch_that_closes_everything_is_sent_back_to_the_writer_for_a_lack_of_suspense():
    def llm(prompt, schema, seed=0):
        return {"polarity": "bad", "reason": "captured", "suspense": False}
    problems = sa.verify_polarity(llm, TREND, [{"text": "You were collected. The end."}], "bad")
    assert len(problems) == 1 and "ONE open question" in problems[0]
    assert sa.verify_polarity(lambda prompt, schema, seed=0: {"polarity": "bad", "reason": "x", "suspense": True}, TREND, [{"text": "A door opens."}], "bad") == []


def test_the_offer_shot_puts_both_in_the_picture_and_a_branch_shows_only_its_character():
    brief = {"title": "T", "world": {"premise": "p", "hour": "night"}, "characters": [{"id": "c1", "name": "Brandt", "role": "r", "gender": "m", "age": 40, "look": "l", "wardrobe": "w"},
                                                                                   {"id": "c2", "name": "Ilse", "role": "r", "gender": "f", "age": 40, "look": "l", "wardrobe": "w"}]}
    base = {"fx": {"zoom": 0.05, "shake": 0.0, "flash": False}, "camera": "medium", "motion": "m", "still": "s"}
    beats = [{"act": "offers", "branch": "main", "kind": "offer", "speaker": "c1", "purpose": "p"}, {"act": "offers", "branch": "main", "kind": "offer", "speaker": "c2", "purpose": "p"},
             {"act": "branch_a", "branch": "A", "kind": "pov", "speaker": "narrator", "purpose": "p"}, {"act": "branch_b", "branch": "B", "kind": "pov", "speaker": "narrator", "purpose": "p"}]
    lines = [{"text": "I know a dry harbor.", "location": "boat", "in_shot": ["c1"]}, {"text": "I have a hospital.", "location": "boat", "in_shot": ["c2"]},
             {"text": "You run.", "location": "boat", "in_shot": ["c1", "c2"]}, {"text": "You climb.", "location": "boat", "in_shot": ["c1", "c2"]}]
    outline = {"hook_title": "x", "beats": beats, "ending_label_a": "A", "ending_label_b": "B", "closing_question": "q", "caption": "c"}
    shots = sa.assemble_story(brief, outline, lines, [dict(base) for _ in beats], {"A": "bad", "B": "good"}, TREND)["shots"]
    assert [s["in_shot"] for s in shots[:2]] == [["c1", "c2"], ["c1", "c2"]] and "overlap" not in shots[1]  # one after the other
    assert [s["speaker"] for s in shots[:2]] == ["narrator", "narrator"] and [s["offer_of"] for s in shots[:2]] == ["c1", "c2"]  # the narrator quotes the two proposals
    assert shots[2]["in_shot"] == ["c1"] and shots[3]["in_shot"] == ["c2"]  # the chosen one is in the shot, the other never


def test_the_other_character_of_a_branch_is_removed_and_the_chosen_one_is_not_forced_into_the_picture():
    brief = {"characters": [{"id": "c1", "name": "Kael"}, {"id": "c2", "name": "Elara"}]}
    shots = [{"id": "s1", "kind": "narration", "branch": "main", "text": "Kael steps forward.", "still": "close-up of a scarred face", "motion": "push-in", "in_shot": []},
             {"id": "s2", "kind": "narration", "branch": "main", "text": "Two figures.", "still": "medium shot of Elara standing", "motion": "x", "in_shot": ["c1"]},
             {"id": "s3", "kind": "pov", "branch": "A", "text": "You climb.", "still": "your hands on a rusted grate, Elara far below", "motion": "x", "in_shot": ["c2"]},
             {"id": "s4", "kind": "pov", "branch": "B", "text": "A door slams somewhere.", "still": "a steel door shaking in its frame", "motion": "x", "in_shot": []},
             {"id": "s5", "kind": "rewind", "branch": "B", "text": "What if you chose Elara instead of Kael?", "still": "x", "motion": "x", "in_shot": []},
             {"id": "s6", "kind": "talk", "branch": "main", "text": "Kael", "still": "", "motion": "x", "in_shot": ["c1"]}]
    assert sa.infer_in_shot(shots, brief) == 4
    assert [s["in_shot"] for s in shots] == [["c1"], ["c1", "c2"], [], [], ["c1", "c2"], ["c1"]]  # c2 is removed from branch A; nobody is forced in (an event elsewhere is allowed); the rewind shows both


def test_the_director_must_name_the_characters_and_never_write_the_narrator():
    shots = [{"kind": "narration", "text": "a", "in_shot": ["c1"]}, {"kind": "narration", "text": "b", "in_shot": []}]
    item = {"still": "a wide photograph of a flooded street at night with a boat and a light, rain falling hard", "motion": "slow push-in on the boat now", "camera": "wide", "fx": {"zoom": 0.05, "shake": 0, "flash": False}}
    answer = {"directions": [{**item, "n": 1}, {**item, "n": 2, "motion": "slow push-in toward the narrator's face"}]}
    problems = sa.validate_directions(answer, shots, TREND, {"c1": "Kael", "c2": "Elara"})
    assert len(problems) == 2 and "Kael is in this shot, write Kael's NAME" in problems[0] and "never write 'the narrator'" in problems[1]
    answer["directions"][0]["still"] = "Kael in his torn olive-drab uniform walks down a flooded street at night with a boat and a light behind him"
    assert len(sa.validate_directions(answer, shots, TREND, {"c1": "Kael"})) == 1
    assert sa.viewer_words("push-in toward the narrator's face, the narrator stands") == "push-in toward your face, the viewer stands"
    assert sa.clean_strings({"a": ["Kael�s hand", "x�"]}) == {"a": ["Kael's hand", "x"]}


def test_the_offers_and_the_choice_are_one_scene_in_the_place_of_the_choice():
    brief = {"title": "T", "world": {"premise": "p", "hour": "night"}, "locations": [{"id": "vent", "description": "A vent shaft."}, {"id": "corridor", "description": "A red-lit corridor."}],
             "characters": [{"id": "c1", "name": "Kael", "role": "r", "gender": "m", "age": 40, "look": "l", "wardrobe": "Fatigues."}, {"id": "c2", "name": "Elara", "role": "r", "gender": "f", "age": 40, "look": "l", "wardrobe": "A white coat."}]}
    base = {"fx": {"zoom": 0.05, "shake": 0.0, "flash": False}, "camera": "medium", "motion": "m", "still": "s"}
    beats = [{"act": "offers", "branch": "main", "kind": "offer", "speaker": "c1", "purpose": "p"}, {"act": "offers", "branch": "main", "kind": "offer", "speaker": "c2", "purpose": "p"},
             {"act": "choice", "branch": "main", "kind": "choice", "speaker": "narrator", "purpose": "p"}]
    lines = [{"text": "Kael offers the vent.", "location": "vent", "in_shot": []}, {"text": "Elara offers the clean room.", "location": "vent", "in_shot": []}, {"text": "Choose now.", "location": "corridor", "in_shot": []}]
    outline = {"hook_title": "x", "beats": beats, "ending_label_a": "A", "ending_label_b": "B", "closing_question": "q", "caption": "c"}
    shots = sa.assemble_story(brief, outline, lines, [dict(base) for _ in beats], {"A": "bad", "B": "good"}, TREND)["shots"]
    assert [s["location"] for s in shots] == ["corridor", "corridor", "corridor"] and "red-lit corridor" in shots[0]["still"] and shots[0]["still"] == shots[1]["still"]


def test_a_chunk_with_only_soft_problems_keeps_the_best_answer_of_the_model_never_a_template():
    answers = iter([{"n": 1}, {"n": 2}, {"n": 3}])

    def llm(prompt, schema, seed=0):
        return next(answers)
    problems_of = {1: ["the lines are too long on average (9.3 words)", "x"], 2: ["the lines are too long on average (9.0 words)"], 3: ["this branch must end GOOD for you but its last lines read MIXED (teaser)"]}
    answer, problems, attempts = sa.call_agent(llm, "p", {}, lambda a: problems_of[a["n"]], soft=lambda p: sa.soft_writer_problem(p) or p == "x")
    assert answer == {"n": 2} and problems == ["the lines are too long on average (9.0 words)"] and attempts == 3  # the best of the three, listed as imperfect
    answers = iter([{"n": 1}, {"n": 1}, {"n": 1}])
    hard = sa.call_agent(llm, "p", {}, lambda a: ["line 3: 40 words, the limit is 14"], soft=sa.soft_writer_problem)
    assert hard[0] is None  # a line over the limit is not a matter of taste
    assert sa.soft_writer_problem("this branch must end GOOD for you but its last lines read BAD (x)") is False  # the opposite outcome is a real failure
    assert sa.soft_writer_problem("this branch must end GOOD for you but its last lines read MIXED (x)") is True


def test_a_list_of_directions_shifted_by_one_shot_is_refused():
    """Found in the final run: the pictures of the last shots of a branch were the ones of the line before. Each direction repeats the number of its shot."""
    shots = [{"kind": "narration", "text": "a"}, {"kind": "narration", "text": "b"}, {"kind": "narration", "text": "c"}]
    item = {"still": "a wide photograph of a flooded street at night with a boat and a light, rain falling hard", "motion": "slow push-in on the boat now", "camera": "wide", "fx": {"zoom": 0.05, "shake": 0, "flash": False}}
    right = {"directions": [{**item, "n": 1}, {**item, "n": 2}, {**item, "n": 3}]}
    assert sa.validate_directions(right, shots, TREND) == []
    shifted = {"directions": [{**item, "n": 1}, {**item, "n": 3}, {**item, "n": 4}]}
    problems = sa.validate_directions(shifted, shots, TREND)
    assert len(problems) == 2 and "direction 2 says n=3" in problems[0]
    assert "reuse it" in TREND.DIRECTOR_PROMPT  # the grandiose picture of the world is for the first shot only


def test_the_script_doctor_sends_a_confusing_line_back_to_the_writer_as_a_soft_problem():
    def llm(prompt, schema, seed=0):
        assert "[[STAGE:clarity]]" in prompt and "2. Blood on cuff? No, a bird." in prompt
        return {"confusing": [{"n": 2, "why": "a bird has never been mentioned"}, {"n": 9, "why": "out of range"}]}
    lines = [{"text": "She preps the kit."}, {"text": "Blood on cuff? No, a bird."}]
    problems = sa.verify_clarity(llm, TREND, [{}, {}], lines, "(the start)")
    assert problems == ["line 2 is confusing for a viewer who sees it once (a bird has never been mentioned): rewrite it so that it is clear, introduce what it refers to"]
    assert sa.soft_writer_problem(problems[0]) is True  # a matter of taste after three attempts: the best answer of the model is kept
    assert sa.verify_clarity(lambda prompt, schema, seed=0: 1 / 0, TREND, [{}], lines[:1], "x") == []  # a doctor that cannot answer never blocks the story


def test_the_two_offer_shots_get_the_same_two_shot_picture_made_by_code():
    brief = {"locations": [{"id": "corridor", "description": "A narrow red-lit corridor of the bunker."}],
             "characters": [{"id": "c1", "gender": "m", "wardrobe": "Military fatigues, dog tags."}, {"id": "c2", "gender": "f", "wardrobe": "White coat, stethoscope."}]}
    still = sa.offer_still(brief, "corridor")
    assert "on the left a man (Military fatigues, dog tags)" in still and "on the right a woman (White coat, stethoscope)" in still and "both hold out one open hand toward the camera" in still
    assert "narrow red-lit corridor" in still and len(still.split()) <= 48


def test_the_idea_agent_invents_the_place_from_two_drawn_ingredients_and_never_repeats_a_known_world():
    bones = sa.draw_seed_elements({"seed": 7}, TREND)
    assert bones["mode"] == "invent" and len(set(bones["ingredients"])) == 2 and bones["hour"] in TREND.HOURS and bones["fallback_world"][0] in [w[0] for w in TREND.IDEA_WORLDS]
    assert len(TREND.IDEA_WORLDS) >= 30 and len(TREND.IDEA_ROLES) >= 28 and len(TREND.IDEA_TONES) >= 12
    seen = {}

    def llm(prompt, schema, seed=0):
        seen["prompt"] = prompt
        return {"title": "Salt Bells", "context": "The salt flats ring at dusk while every bell of the old station rings by itself. You are a courier stuck on a cable car high above the white plain with the last water almost gone. The wind smells of burnt sugar. Far below, a caravan of lanterns is walking away from you. Two strangers reach your cabin along the cable and each offers to save you. A " + bones["role_a"] + " and a " + bones["role_b"] + " each offer to save you. One of them is lying and you have until the last lantern disappears.", "setting": "a salt cable car", "premise": "the cable is singing"}
    context, report = sa.run_idea({"seed": 7}, TREND, llm)
    assert " + ".join(bones["ingredients"]) in seen["prompt"] and "never one of these (already made)" in seen["prompt"] and "a flooded megacity" in seen["prompt"]
    assert report["invented"] == {"setting": "a salt cable car", "premise": "the cable is singing"} and "[[STAGE:idea]]" in seen["prompt"]


def test_the_idea_prompt_lets_the_model_invent_both_characters_any_genre_and_asks_for_a_size_that_follows_the_video():
    assert "{{role_a}}" not in TREND.IDEA_INVENT_PROMPT and "invent them yourself" in TREND.IDEA_INVENT_PROMPT and "fairy, an elf thief" in TREND.IDEA_INVENT_PROMPT and "Follow the genre of the seed" in TREND.IDEA_INVENT_PROMPT
    assert sa.context_sentences({"target_seconds": 150}, TREND) == 8 and sa.context_sentences({"target_seconds": 30}, TREND) == 4 and sa.context_sentences({"target_seconds": 600}, TREND) == 12
    short = sa.validate_idea({"title": "t", "context": "One. Two. Three."}, {"mode": "invent", "role_a": "x", "role_b": "y"}, sentences=8)
    assert short and "about 8 sentences" in short[0]  # a longer video asks a richer context
    assert sa.context_size("The sea rises. A fairy and an elf thief appear! You run?") == {"sentences": 3, "words": 12}


def test_the_analyst_is_told_the_size_of_the_context_and_may_invent_the_characters_of_any_genre():
    seen = {}

    def llm(prompt, schema, seed=0):
        seen["prompt"] = prompt
        raise RuntimeError("stop here")
    sa.run_analyst("A fairy and an elf thief offer you a boat. The sea rises. The city sleeps beneath it.", {"language": "en", "tone": "tense", "seed": 1}, TREND, llm)
    prompt = seen["prompt"]
    assert "(3 sentences, 18 words)" in prompt or "(3 sentences, 17 words)" in prompt and "5 or 6 places" in prompt and "INVENT them" in prompt and "fantasy" in prompt and "never make the story more realistic" in prompt
    assert "realistic dark" not in TREND.DIRECTOR_PROMPT and "photoreal RENDER applied to ANY world" in TREND.DIRECTOR_PROMPT
