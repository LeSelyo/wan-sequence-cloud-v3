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


def test_the_place_of_the_shot_is_written_in_the_prompt():
    plan = {"locations": [{"id": "corridor", "tags": ["space"], "description": "The freezing metal walkways leading to the airlock."}]}
    prompt = sst.shot_prompt({"still": "Debris falls around the narrator.", "location": "corridor"}, plan, STYLE)
    assert "Debris falls around the narrator, set in the freezing metal walkways leading to the airlock, vertical composition" in prompt


def test_a_night_shot_of_a_night_story_never_turns_the_light_to_the_sun():
    """auto_ab_1: the director wrote time 'night' on the shots of a night story and the old code answered 'warm natural light': a sunlit stone arcade with a parapet in a station."""
    plan = {"locations": [{"id": "sea", "tags": ["sea"]}], "brief": {"world": {"hour": "night"}}}
    night = sst.shot_prompt({"still": "a boat", "location": "sea", "time": "night"}, plan, STYLE)
    assert "warm natural light" not in night and "cold moonlit" in night
    dusk = sst.shot_prompt({"still": "a boat", "location": "sea", "time": "it is dusk: the last glow"}, plan, STYLE)
    assert "warm natural light" not in dusk and "it is dusk: the last glow" in dusk
    dawn = sst.shot_prompt({"still": "a boat", "location": "sea", "time": "it is dawn: golden sunrise light"}, plan, STYLE)
    assert "warm natural light" in dawn and "cold moonlit" not in dawn
