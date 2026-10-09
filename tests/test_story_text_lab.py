import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_story  # noqa: E402
import story_text_lab as tl  # noqa: E402

pytestmark = pytest.mark.skipif(not reference_story.REFERENCE.exists(), reason="needs the hand-written reference plan")


def test_the_idea_range_records_what_was_drawn_and_what_the_model_invented():
    def llm(prompt, schema, seed=0):
        return {"title": "Salt Bells", "context": " ".join(f"Sentence number {i} of a long enough invented context about the salt flats and the bells." for i in range(8)), "setting": "a salt cable car", "premise": "the cable sings"}
    rows = tl.idea_range([3, 4], 150, llm)
    assert [r["seed"] for r in rows] == [3, 4] and rows[0]["source"] == "llm" and rows[0]["asked_sentences"] == 8 and rows[0]["size"]["sentences"] == 8
    assert rows[0]["invented"] == {"setting": "a salt cable car", "premise": "the cable sings"} and len(rows[0]["bones"]["ingredients"]) == 2 and rows[0]["bones"]["ingredients"] != rows[1]["bones"]["ingredients"]


def test_the_structure_checks_read_the_new_rules_on_a_plan(tmp_path):
    plan = reference_story.load()
    plan["characters"] = [{"id": "c1", "name": "Brandt"}, {"id": "c2", "name": "Ilse"}]
    checks = tl.structure_checks(plan)
    assert checks["offers_adjacent"] and checks["choice_right_after_offers"] and checks["shots_before_offers"] == 12 and checks["talk_shots"] <= 4
    assert checks["other_character_named_in_branch"] == {"A": [], "B": []} and len(checks["twists"]) == 2
    broken = reference_story.load()
    broken["characters"] = plan["characters"]
    broken["shots"].insert(14, dict(broken["shots"][0], id="sx", kind="narration"))
    assert tl.structure_checks(broken)["choice_right_after_offers"] is False
