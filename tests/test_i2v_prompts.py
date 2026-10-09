import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import i2v_action as i2v  # noqa: E402

KAEL = {"name": "Kael", "gender": "m", "wardrobe": "Torn olive-drab uniform, heavy combat boots, gas mask hanging around neck."}
ELARA = {"name": "Elara", "gender": "f", "wardrobe": "Immaculate medical scrubs, white coat, latex gloves, carrying a heavy medical kit."}


def test_the_video_prompt_says_who_is_who_by_name_and_clothes():
    prompt = i2v.build_prompt("pov", "he pulls you hard, the camera jerks forward", [KAEL])
    assert prompt.startswith("Kael, the man in torn olive-drab uniform, heavy combat boots, gas mask hanging. he pulls you hard")
    two = i2v.build_prompt("offer_hands", characters=[KAEL, ELARA])
    assert two.startswith("Kael, the man in torn olive-drab uniform") and ", on the left, and Elara, the woman in immaculate medical scrubs" in two and "on the right, slowly stretch their open hands" in two


def test_real_steps_are_asked_only_for_a_movement_through_the_place():
    close = i2v.build_prompt("action", "fast zoom-in on his terrified eyes, his head jerks slightly", [KAEL])
    walk = i2v.build_prompt("action", "Kael runs down the corridor toward the camera", [KAEL])
    assert "the whole body in motion" not in close and "keeps their place" in close  # a close-up is no longer told to swing its arms
    assert "real steps and body movement" in walk and "keeps their place" not in walk


def test_the_viewer_is_never_the_narrator_in_a_video_prompt_and_a_place_has_no_anchor():
    prompt = i2v.build_prompt("scene", "slow tilt down to the narrator's hesitant face")
    assert prompt == "slow tilt down to your hesitant face"
