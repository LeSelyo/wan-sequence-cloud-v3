"""The reference close-up is the source of truth for a character's face: observed once, used by every identity prompt (never the free text of the brief)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_identity as sid  # noqa: E402
import story_traits as stt  # noqa: E402

TOMAS = {"id": "c1", "name": "Tomas", "gender": "m", "age": 45, "look": "Weathered face, milky white eyes, hands stained with beeswax.", "wardrobe": "Heavy layered robes made of thick beeswax."}
RAVI = {"id": "c2", "name": "Ravi", "gender": "m", "age": 28, "look": "Sharp features, scarred knuckles, curly black hair.", "wardrobe": "Light flexible gear scaled in iridescent chitin."}


def test_traits_are_positive_facts_in_a_fixed_order_and_nothing_stands_for_nothing():
    found = {"eyes": "dark brown", "hair": "short grey", "facial_hair": "grey stubble", "scars_or_marks": "None", "skin": "olive", "head_cover": "a beige hood"}
    assert stt.traits_text(found) == "dark brown eyes, short grey hair, grey stubble, olive skin, a beige hood"  # no "no scar": a negation makes the scar more likely to be drawn
    assert stt.traits_text({"eyes": "blue eyes", "scars_or_marks": "a scar across the left cheek", "hair": "not visible", "skin": ""}) == "blue eyes, a scar across the left cheek"
    assert stt.traits_text({}) == "" and stt.traits_text(None) == ""


def test_the_vision_model_looks_at_the_close_up_once_and_a_model_that_cannot_answer_gives_nothing(tmp_path):
    seen = {}

    def ask(prompt, schema, image, seed=0):
        seen.update(prompt=prompt, image=image, required=schema["required"])
        return {"eyes": "dark brown", "hair": "grey", "facial_hair": "stubble", "scars_or_marks": "none", "skin": "olive", "head_cover": "hood"}
    found = stt.describe(ask, tmp_path / "close.png", TOMAS)
    assert found["eyes"] == "dark brown" and "Tomas" in seen["prompt"] and "Never guess" in seen["prompt"] and set(seen["required"]) == set(stt.FIELDS)

    def broken(prompt, schema, image, seed=0):
        raise RuntimeError("model down")
    assert stt.describe(broken, tmp_path / "close.png", TOMAS) == {}


def test_the_traits_are_kept_in_the_cards_and_read_back_and_a_character_never_looked_at_has_none(tmp_path):
    (tmp_path / "cards.json").write_text(json.dumps({"characters": {"c1": {"name": "Tomas"}, "c2": {"name": "Ravi"}}}), encoding="utf-8")
    stt.save(tmp_path, {"c1": {"eyes": "dark brown", "hair": "grey", "facial_hair": "none", "scars_or_marks": "none", "skin": "olive", "head_cover": "none"}, "c2": {}})
    cards = json.loads((tmp_path / "cards.json").read_text(encoding="utf-8"))
    assert stt.of(cards, "c1") == "dark brown eyes, grey hair, olive skin" and stt.of(cards, "c2") == "" and "traits" not in cards["characters"]["c2"]


def test_the_identity_prompts_use_the_observed_traits_and_never_the_free_text_that_gave_a_character_pale_eyes_and_a_scar():
    two = sid.prompt_two(TOMAS, RAVI, "dark brown eyes, grey stubble", "short black hair, clean-shaven")
    assert "On the left is Tomas (45 years old, dark brown eyes, grey stubble, wearing heavy layered robes made of thick beeswax)" in two
    assert "on the right is Ravi (28 years old, short black hair, clean-shaven, wearing light flexible gear scaled in iridescent chitin)" in two
    assert "milky" not in two and "scar" not in two and "Sharp features" not in two  # the brief's words are not in the prompt
    one = sid.prompt_single(TOMAS, "dark brown eyes, grey stubble")
    assert one.endswith("Keep exactly the face and the head of Picture 2: dark brown eyes, grey stubble.") and "milky" not in one
    assert "Keep exactly the face" not in sid.prompt_single(TOMAS)  # a project without observed traits still works


def test_the_quality_report_says_when_the_characters_were_never_observed(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_story_qa as helper
    import story_qa as qa
    root = helper.project(tmp_path)
    (root / "cards").mkdir()
    (root / "cards" / "cards.json").write_text(json.dumps({"characters": {"c1": {"name": "Tomas"}, "c2": {"name": "Ravi", "traits_text": "short black hair"}}}), encoding="utf-8")
    warnings = "\n".join(qa.collect(root)["warnings"])
    assert "['Tomas'] were never looked at" in warnings and "Ravi" not in warnings.split("never looked at")[0].split("FACES")[-1]
    assert "short black hair" in qa.markdown(qa.collect(root))
