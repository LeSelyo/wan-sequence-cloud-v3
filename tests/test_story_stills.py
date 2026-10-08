import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_stills as sst  # noqa: E402

STYLE = {"texture": "film", "lens": "35mm", "lighting": "cold moonlit", "grade": "teal", "palette": ["black"], "materials": ["water"], "motif": "rain",
         "forced_elements": {"sea": "black water", "time": "it is night in every shot: dark sky"}}
PLAN = {"locations": [{"id": "sea", "tags": ["sea"]}]}


def test_a_night_style_applies_to_every_shot_but_a_shot_can_set_its_own_hour():
    night = sst.shot_prompt({"still": "a boat", "location": "sea"}, PLAN, STYLE)
    assert "it is night in every shot" in night and "cold moonlit" in night and night.startswith("a boat, vertical composition")
    dawn = sst.shot_prompt({"still": "a city", "location": "sea", "time": "it is dawn: golden light"}, PLAN, STYLE)
    assert "it is dawn: golden light" in dawn and "it is night" not in dawn and "cold moonlit" not in dawn and "black water" in dawn  # the place still shows its forced element
