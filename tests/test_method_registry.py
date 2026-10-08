import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import method_registry as mr  # noqa: E402


def test_each_shot_is_classified_by_what_it_is_and_what_moves():
    assert mr.classify_shot({"kind": "talk"}) == "talking_closeup" and mr.classify_shot({"kind": "choice"}) == "two_person_offer" and mr.classify_shot({"kind": "offer"}) == "two_person_offer"
    assert mr.classify_shot({"kind": "narration", "in_shot": ["c2"], "motion": "she sprints down the corridor toward the camera"}) == "person_locomotion"
    assert mr.classify_shot({"kind": "narration", "in_shot": ["c1"], "motion": "he stands still and looks"}) == "person_in_scene"
    assert mr.classify_shot({"kind": "pov", "motion": "your hands crank the valve wheel"}) == "pov_action"
    assert mr.classify_shot({"kind": "pov", "motion": "the view drifts over the dark sea"}) == "scene_camera_move"
    assert mr.classify_shot({"kind": "narration", "in_shot": [], "still": "colossal aerial view of a megacity under a storm"}) == "establishing_epic"
    assert mr.classify_shot({"kind": "narration", "in_shot": [], "motion": "rows of silent people turn their heads"}) == "crowd"
    assert mr.classify_shot({"kind": "narration", "in_shot": [], "motion": "slow push-in on a chart under a lamp"}) == "scene_camera_move"


def test_the_registry_picks_what_works_never_what_fails_and_marks_an_untested_pick_as_a_trial(tmp_path):
    registry = mr.Registry(tmp_path / "registry.json")
    walk = registry.choose("person_locomotion")
    assert walk["method"] == "i2v_action" and walk["status"] == "works" and not walk["trial"] and walk["settings"]["style"] == "action"
    assert all(r["method"] != "s2v_silence" for r in registry.candidates("person_locomotion"))  # the method that makes people freeze is never offered
    pov = registry.choose("pov_action")
    assert pov["method"] == "i2v_pov" and pov["trial"] is False and pov["settings"]["engine"] == "i2v"
    assert registry.choose("scene_camera_move")["method"] == "s2v_silence" and registry.choose("talking_closeup")["settings"]["audio"] == "voice"


def test_a_result_or_a_choice_of_the_user_changes_what_the_pipeline_picks_and_is_kept_with_its_proof(tmp_path):
    registry = mr.Registry(tmp_path / "registry.json")
    registry.record("pov_action", "i2v_lightx4", "works", "hands crank the valve, steam bursts", metrics={"motion_amount": 2.4}, source="lab pov_valve, chosen by the user")
    again = mr.Registry(tmp_path / "registry.json")  # reloaded from the file
    assert again.choose("pov_action")["status"] == "works" and not again.choose("pov_action")["trial"]
    assert again.records[[r["method"] for r in again.records].index("i2v_lightx4")]["evidence"]
    registry.record("person_locomotion", "i2v_lightx4", "fails", "slides without moving the legs in this setting")
    assert registry.candidates("person_locomotion")[0]["method"] != "i2v_lightx4"


def test_a_measure_can_decide_the_status_and_unknown_names_are_refused():
    assert mr.verdict_from_metrics("person_locomotion", {"motion_amount": 2.0}) == "works" and mr.verdict_from_metrics("person_locomotion", {"motion_amount": 0.3}) == "fails"
    assert mr.verdict_from_metrics("talking_closeup", {"blinks": 2}) == "works" and mr.verdict_from_metrics("talking_closeup", {"blinks": 0}) == "fails"
    assert mr.verdict_from_metrics("scene_camera_move", {"motion_amount": 2.0}) is None
    import pytest
    with pytest.raises(ValueError):
        mr.Registry(Path("x.json")).record("nonsense", "i2v_lightx4", "works", "x")


def test_the_route_of_a_shot_follows_the_choices_of_the_user(tmp_path):
    registry = mr.Registry(tmp_path / "registry.json")
    assert mr.route({"kind": "talk"}, None, registry)["method"] == "s2v_voice"  # S2V only when the character speaks
    offer = mr.route({"kind": "choice"}, None, registry)
    assert offer["need"] == "two_person_offer" and offer["method"] == "i2v_offer_hands" and offer["settings"]["style"] == "offer_hands"  # the choice moment: always plan A
    assert all(r["method"] != "s2v_voice" for r in registry.candidates("two_person_offer"))  # plan C was dropped for this scene
    walk = mr.route({"kind": "narration", "in_shot": ["c2"], "motion": "she runs, the camera follows at the same speed"}, None, registry)
    assert walk["method"] == "i2v_follow"
    assert mr.route({"kind": "narration", "in_shot": ["c2"], "motion": "she sprints down the corridor"}, None, registry)["method"] == "i2v_action"
    assert mr.route({"kind": "narration", "in_shot": ["c1"], "motion": "he nods slowly"}, None, registry)["method"] == "i2v_action"
    assert mr.route({"kind": "pov", "motion": "your hands crank the valve wheel"}, None, registry)["method"] == "i2v_pov"
