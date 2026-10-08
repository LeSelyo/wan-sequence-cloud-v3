import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import still_judge as sj  # noqa: E402

PLAN = {"brief": {"world": {"setting": "Oort Station, orbiting a darkened Earth", "atmosphere": "freezing, silence", "hour": "night"}},
        "locations": [{"id": "corridor", "description": "The freezing metal walkways leading to the airlock"}],
        "shots": [{"id": "s039", "location": "corridor", "still": "Debris falls around the narrator", "kind": "narration"},
                  {"id": "s050", "location": "corridor", "still": "A sunrise over the Earth", "time": "it is dawn: golden sunrise light", "kind": "narration"},
                  {"id": "s001", "location": "corridor", "still": "", "kind": "talk"}]}
GOOD = {"place_seen": "a metal corridor with viewports on space", "sunlight_or_blue_sky": False, "matches_world": 5, "action_visible": True, "collage_or_split_panels": False, "garbled_text": False, "problems": []}


def test_a_sunlit_stone_arcade_in_a_night_station_is_wrong_and_a_sign_with_odd_letters_is_only_a_note():
    arcade = dict(GOOD, place_seen="a sunlit stone arcade with a parapet", sunlight_or_blue_sky=True, matches_world=1)
    major, minor = sj.verdict(arcade, no_daylight=True)
    assert len(major) == 2 and "sunlight" in major[0] and "stone arcade" in major[1]
    major, minor = sj.verdict(dict(GOOD, garbled_text=True), no_daylight=True)
    assert major == [] and minor == ["a sign has invented letters"]
    assert sj.verdict(dict(GOOD, action_visible=False, collage_or_split_panels=True), True)[0] == ["the action of the shot is not visible", "a collage of panels"]
    assert sj.verdict(dict(GOOD, sunlight_or_blue_sky=True), no_daylight=False)[0] == []  # a dawn is allowed to be sunny


def test_daylight_is_forbidden_in_a_night_world_except_in_a_shot_that_tells_its_own_dawn():
    assert sj.daylight_forbidden(PLAN, PLAN["shots"][0]) is True
    assert sj.daylight_forbidden(PLAN, PLAN["shots"][1]) is False
    assert sj.daylight_forbidden(PLAN, dict(PLAN["shots"][0], time="night")) is True


def test_the_model_is_given_the_world_the_place_and_the_shot_with_the_picture(tmp_path):
    seen = {}

    def ask(prompt, schema, image, seed):
        seen.update(prompt=prompt, schema=schema, image=image)
        return dict(GOOD)

    picture = tmp_path / "s039.png"
    picture.write_bytes(b"png")
    result = sj.judge_still(ask, picture, PLAN["shots"][0], PLAN)
    assert result["ok"] and seen["image"] == picture and set(seen["schema"]["required"]) <= set(seen["schema"]["properties"])
    assert "Oort Station" in seen["prompt"] and "freezing metal walkways" in seen["prompt"] and "Debris falls around the narrator" in seen["prompt"] and "Hour: night" in seen["prompt"]


def test_only_the_shots_with_a_picture_are_judged(tmp_path):
    for name in ("s039", "s050"):
        (tmp_path / f"{name}.png").write_bytes(b"png")
    calls = []
    results = sj.judge_stills(PLAN, tmp_path, lambda prompt, schema, image, seed: calls.append(image.name) or dict(GOOD))
    assert sorted(results) == ["s039", "s050"] and calls == ["s039.png", "s050.png"]  # the talking shot has no still
