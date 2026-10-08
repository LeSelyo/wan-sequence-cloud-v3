import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_agents as sa  # noqa: E402
import story_eval as ev  # noqa: E402

REFERENCE = Path(__file__).resolve().parent.parent / "results" / "story_trend" / "plans" / "the_last_city_150_en.json"
CONTEXT = "A flooded city at night. Two strangers on a rescue boat each offer to save you: a ship captain and a doctor."
pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="needs the hand-written reference plan")


def test_the_measures_tell_a_written_story_from_a_template_one():
    hand = json.loads(REFERENCE.read_text(encoding="utf-8"))
    good = ev.plan_metrics(hand, CONTEXT, 150)
    template = sa.make_plan(CONTEXT, {"target_seconds": 150}, seed=1, llm=None, voices=[])
    bad = ev.plan_metrics(template, CONTEXT, 150)
    assert good["problems_left"] == 0 and good["has_choice"] and good["rewind"] and [t[0] for t in good["twists"]] == ["bad", "good"]
    assert good["repeated_word_pairs"] < bad["repeated_word_pairs"] and good["distinct_first_words"] > bad["distinct_first_words"] and good["forbidden_in_pictures"] == 0
    assert 130 <= good["speech_seconds"] <= 160 and good["talk_lines"]["c1"] >= 2 and good["talk_lines"]["c2"] >= 2
