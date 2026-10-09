"""The hand-written 2m30 story of the first video (results/story_trend/plans/the_last_city_150_en.json) has the OLD structure (talk lines and accusations before the choice, the other character named in
the branches). The tests use it rewritten into the NEW structure: ONE offer moment (two offer shots, c1 then c2, both in the picture), no talk before the choice, independent branches, more first-person action."""
import json
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent.parent / "results" / "story_trend" / "plans" / "the_last_city_150_en.json"


def new_structure(plan: dict) -> dict:
    shots = plan["shots"]
    both = "Two strangers stand side by side on the deck of the rescue boat, close together, both facing you and holding out an open hand toward the camera, black flooded city behind them"
    for index, who, text in ((6, "c1", "Brandt offers a dry harbor."), (7, "c2", "Ilse offers a rooftop hospital.")):
        shots[index].update(kind="offer", speaker="narrator", offer_of=who, text=text, in_shot=["c1", "c2"], still=both, motion="both step forward and hold out their open hands toward the camera, slow push-in",
                            camera="medium")
    shots[26].update(kind="narration", speaker="narrator", text="He says you were the last on his list.", in_shot=[], still="An open metal drawer full of paper lists under a dim lamp in a ship cabin at night, a hand turning the pages, cold light on the names", motion="slow push-in on the list")
    shots[39].update(kind="narration", speaker="narrator", text="She says the water carries a fever.", in_shot=[], still="A makeshift hospital room on a flooded rooftop at night, camp beds in rows and one lamp swinging above a table of vials and bandages", motion="slow push-in over the beds")
    for index, text in ((8, "The big boat looks empty."), (9, "A small inflatable rocks beside it."), (11, "Two hands stay out, waiting for you."), (13, "No time. You must choose now.")):
        shots[index].update(kind="narration", speaker="narrator", text=text, in_shot=[], still="The big rescue boat and a small orange inflatable rock side by side on the black water of the flooded city at night, lit by one cold lamp on the deck", motion="slow push-in over the dark water")
    shots[25]["text"] = "Far behind you, a light goes out."
    shots[32].update(text="Someone screams your name across the water.", in_shot=[])
    for index in (17, 19, 23, 27, 34, 35, 37, 41):
        shots[index].update(kind="pov", still=shots[index].get("still") or "First-person view: your two hands grip the wet railing of the boat while it slams through the black waves between the drowned towers of the flooded city at night", camera="pov")
    # the context comes BEFORE the offer: the two characters are introduced (shots 8-13), then the two offers, then the choice at once
    plan["shots"] = shots[:6] + shots[8:14] + shots[6:8] + shots[14:]
    names = {"c1": "Brandt", "c2": "Ilse"}
    for number, shot in enumerate(plan["shots"], 1):
        shot["id"] = f"s{number:03d}"
        # the director NAMES the characters of a shot in its still and its motion (the video model is not told who is who otherwise)
        who = [cid for cid, name in names.items() if cid in shot.get("in_shot", []) or name.lower() in shot["text"].lower()]
        for cid in who:
            if shot["kind"] != "offer":
                if shot.get("still") and names[cid].lower() not in shot["still"].lower():
                    shot["still"] += f", {names[cid]} in the picture"
                if names[cid].lower() not in shot["motion"].lower():
                    shot["motion"] += f", {names[cid]} moves"
        shot["still"] = (shot.get("still") or "").replace("narrator", "viewer")
        shot["motion"] = shot["motion"].replace("narrator", "viewer")
    return plan


def load() -> dict:
    return new_structure(json.loads(REFERENCE.read_text(encoding="utf-8")))
