import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_identity as sid  # noqa: E402

PEOPLE = [{"id": "c1", "name": "Kael", "gender": "m", "wardrobe": "Heavy thermal suit, grease-stained, frayed headset."}, {"id": "c2", "name": "Elara", "gender": "f", "wardrobe": "Pristine white chef's uniform, sealed apron."}]
CARDS = {"characters": {"c1": {"portrait": {"file": "p1.png"}, "closeup": {"file": "c1.png"}}, "c2": {"portrait": {"file": "p2.png"}}}}
PLAN = {"characters": PEOPLE, "shots": [
    {"id": "s001", "kind": "narration", "in_shot": [], "still": "a huge station above a dark Earth"},
    {"id": "s002", "kind": "offer", "in_shot": ["c1", "c2"], "still": "Two people side by side hold out a hand toward you"},
    {"id": "s003", "kind": "offer", "in_shot": ["c1", "c2"], "still": "Two people side by side hold out a hand toward you"},
    {"id": "s004", "kind": "choice", "in_shot": ["c1", "c2"], "still": ""},
    {"id": "s005", "kind": "narration", "in_shot": ["c2"], "still": "A woman runs down a metal corridor"},
    {"id": "s006", "kind": "talk", "in_shot": ["c1"], "still": ""}]}


def test_the_identity_jobs_cover_the_people_shots_and_the_one_scene_of_the_choice(tmp_path):
    jobs = {j["id"]: j for j in sid.jobs(PLAN, CARDS, tmp_path)}
    assert set(jobs) == {"two_shot", "s005"}  # a place, a talking close-up: nothing to do; the offers and the choice share ONE picture
    assert jobs["two_shot"]["shots"] == ["s002", "s003", "s004"] and jobs["two_shot"]["scene"] == tmp_path / "stills" / "s002.png"
    assert [r.name for r in jobs["two_shot"]["refs"]] == ["c1.png", "p2.png"]  # the close-up when there is one, else the portrait; c1 first = on the left
    assert "man from Picture 2 on the left and the woman from Picture 3 on the right" in jobs["two_shot"]["prompt"] and "holding out one open hand" in jobs["two_shot"]["prompt"]


def test_a_picture_without_a_face_or_a_body_does_not_get_one(tmp_path):
    """The vision model said s005 shows only a hand: the identity pass would paint a face where there is a holster or a wrist."""
    import json
    (tmp_path / "stills_judge_1.json").write_text(json.dumps({"s005": {"answer": {"person_visible": False}}, "s002": {"answer": {"person_visible": True}}}), encoding="utf-8")
    assert {j["id"] for j in sid.jobs(PLAN, CARDS, tmp_path)} == {"two_shot"}
    (tmp_path / "stills_judge_2.json").write_text(json.dumps({"s005": {"answer": {"person_visible": True}}}), encoding="utf-8")  # a later round saw a face after the picture was made again
    assert {j["id"] for j in sid.jobs(PLAN, CARDS, tmp_path)} == {"two_shot", "s005"}


def test_the_wardrobe_of_the_character_is_written_in_the_prompt(tmp_path):
    """Lab 2026-10-08: without it Elara ran in black instead of her white chef's coat."""
    prompt = {j["id"]: j for j in sid.jobs(PLAN, CARDS, tmp_path)}["s005"]["prompt"]
    assert "the woman from Picture 2" in prompt and "She is wearing pristine white chef's uniform, sealed apron" in prompt
    assert "He is wearing heavy thermal suit" in sid.prompt_single(PEOPLE[0])


def test_an_animation_starts_from_the_identity_picture_when_there_is_one(tmp_path):
    shot = {"id": "s005", "kind": "narration"}
    assert sid.source_of(tmp_path, shot) == tmp_path / "stills" / "s005.png"
    (tmp_path / "stills_id").mkdir()
    (tmp_path / "stills_id" / "two_shot.png").write_bytes(b"x")
    assert sid.source_of(tmp_path, {"id": "s004", "kind": "choice"}).name == "two_shot.png" and sid.source_of(tmp_path, shot).name == "s005.png"
    (tmp_path / "stills_id" / "s005.png").write_bytes(b"x")
    assert sid.source_of(tmp_path, shot) == tmp_path / "stills_id" / "s005.png"
    assert sid.missing(PLAN, CARDS, tmp_path) == []


def test_the_two_shot_prompt_names_each_person_with_age_face_and_wardrobe_so_two_men_are_not_mixed():
    import story_identity as sid_
    tomas = {"id": "c1", "name": "Tomas", "gender": "m", "age": 45, "look": "Weathered face, milky white eyes, hands stained with beeswax.", "wardrobe": "Heavy, layered robes made of thick beeswax."}
    ravi = {"id": "c2", "name": "Ravi", "gender": "m", "age": 28, "look": "Sharp features, scarred knuckles, curly black hair.", "wardrobe": "Light flexible gear scaled in iridescent chitin."}
    prompt = sid_.prompt_two(tomas, ravi)
    assert "man from Picture 2 on the left and the man from Picture 3 on the right" in prompt
    assert "On the left is Tomas (45 years old, weathered face, milky white eyes, hands stained with beeswax, wearing heavy, layered robes made of thick beeswax)" in prompt
    assert "on the right is Ravi (28 years old, sharp features" in prompt and "Never swap them" in prompt
    import still_judge as sj_
    assert "BOTH must match their own portrait" in sj_.SAME_PERSON_REFS and "{people}" in sj_.SAME_PERSON_REFS
