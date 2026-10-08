"""The METHODS REGISTRY: for every NEED of a shot, which method WORKS, which FAILS and the proof, so that the automatic pipeline picks the right method by itself from the nature of the shot
(and learns from every test and every choice of the user).

    classify_shot(shot, plan)  -> the need ("talking_closeup", "person_locomotion", "pov_action", "scene_camera_move", ...)
    Registry().choose(need)    -> the best method known for it (status works > partial > untested; a method that FAILS is never chosen) with its settings
    Registry().record(need, method, "works" | "fails" | "partial", note, metrics=None, source=None)
    python scripts/method_registry.py show | verdict NEED METHOD STATUS "note"

The registry is a JSON file (results/story_trend/methods/registry.json) seeded below with what has been MEASURED or SEEN so far (each seed carries its evidence); the motion lab (scripts/motion_lab.py)
and the choices of the user write to it, and a measured threshold can decide a status by itself (see `verdict_from_metrics`).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_FILE = ROOT / "results" / "story_trend" / "methods" / "registry.json"

NEEDS = {
    "talking_closeup": "a character speaks to the camera, close-up, with his voice line",
    "person_idle_closeup": "a character waits or listens, close-up, no voice (the two waiting faces of the choice card)",
    "person_locomotion": "a character walks, runs, climbs or moves through a place (wide or medium shot)",
    "two_person_offer": "both main characters in one shot holding out a hand to the viewer",
    "pov_action": "first-person shot where the viewer's own hands or body do something",
    "scene_camera_move": "a place or an object, the camera moves, nobody in the frame",
    "establishing_epic": "a huge landscape, city or vehicle seen from far (the grandiose hook)",
    "crowd": "many people (a crowd, rows of people)",
    "wide_with_character": "a still that must show a main character in a wide or medium shot (the face must stay the same)",
}
# the methods and their settings (what the pipeline runs)
METHODS = {
    "s2v_voice": {"engine": "s2v", "lora": "t2v_1217_low", "audio": "voice", "note": "Wan 2.2 Sound-to-Video, 4-step LoRA t2v 1217, the voice line drives the mouth"},
    "s2v_silence": {"engine": "s2v", "lora": "t2v_1217_low", "audio": "silence", "note": "the same model with silence: audio-driven, so a person keeps the starting pose"},
    "s2v_blink_prompt": {"engine": "s2v", "lora": "t2v_1217_low", "audio": "voice", "prompt_extra": "blinks naturally every two or three seconds, the eyelids close and open fully, breathes",
                         "negative_extra": "static eyes, frozen face, eyes always open, no blinking", "note": "S2V with a prompt that asks for blinks and a negative prompt against frozen eyes"},
    "i2v_lightx4": {"engine": "i2v", "profile": "lightx2v4", "note": "Wan 2.2 Image-to-Video, two experts, lightx2v v1 4 steps: made for movement"},
    "i2v_lightx6": {"engine": "i2v", "profile": "lightx2v6", "note": "the same with 6 steps"},
    "i2v_lightx4_hi": {"engine": "i2v", "profile": "lightx2v4_hi", "note": "the same with the LoRAs at 0.8: more motion and detail"},
    "text_still": {"engine": "krea2", "note": "a picture made from the text description only"},
    "qwen_identity": {"engine": "qwen_edit", "note": "Qwen-Image-Edit-2511 puts the person of the close-up (reference) into the scene: the face stays the same"},
}
RANK = {"works": 3, "partial": 2, "untested": 1, "fails": 0}

# (need, method, status, note): what is KNOWN today (dates are 2026-10-08)
SEED = [
    ("talking_closeup", "s2v_voice", "works", "tight close-up (face ~40 % of the frame) + LoRA t2v 1217: clean eyes, face shimmer -24 % against the v1 LoRA (eye test, Ilse); BUT no blinking yet (user feedback)"),
    ("talking_closeup", "s2v_blink_prompt", "untested", "to measure in the blink experiment of the lab"),
    ("person_idle_closeup", "s2v_silence", "partial", "stable face but the eyes never blink (user feedback)"),
    ("person_idle_closeup", "i2v_lightx4", "untested", "no audio: the face is free to blink; to measure"),
    ("person_locomotion", "s2v_silence", "fails", "people keep the starting pose, 'they never walk' (user feedback on auto_ab_1)"),
    ("person_locomotion", "i2v_lightx4", "works", "smoke test: a man walks toward the camera, strides and weight shifts visible (67 s per 3 s clip)"),
    ("person_locomotion", "i2v_lightx4_hi", "untested", "to compare in the motion experiment"),
    ("two_person_offer", "i2v_lightx4", "untested", "to test in the offer experiment (two-shot made with Qwen identity pass)"),
    ("two_person_offer", "s2v_voice", "untested", "to test: do both mouths move?"),
    ("pov_action", "s2v_silence", "fails", "the arm trembles in place, nothing happens (user feedback: 'tres pauvre')"),
    ("pov_action", "i2v_lightx4", "untested", "to test in the POV experiment"),
    ("scene_camera_move", "s2v_silence", "works", "camera push and moving water / rain / light on places without people, several runs"),
    ("scene_camera_move", "i2v_lightx4", "untested", "to compare"),
    ("establishing_epic", "s2v_silence", "works", "aerial push toward a megacity, storm, huge hull: fine"),
    ("crowd", "s2v_silence", "partial", "faces turn slowly; no real group motion"),
    ("crowd", "i2v_lightx4", "untested", "to compare"),
    ("wide_with_character", "text_still", "fails", "a wide still described by text gives a stranger, not the character of the close-up (user feedback + auto_ab_1)"),
    ("wide_with_character", "qwen_identity", "untested", "to test in the identity experiment"),
]

LOCOMOTION = {"walk", "walks", "walking", "run", "runs", "running", "sprint", "sprints", "sprinting", "climb", "climbs", "climbing", "jump", "jumps", "crawl", "crawls", "swim", "swims", "dive", "dives",
              "chase", "chases", "flee", "flees", "rush", "rushes", "step", "steps", "stride", "strides", "race", "races", "racing", "dash", "dashes", "pulls", "drags", "follows", "dodge", "dodges"}
EPIC = {"aerial", "colossal", "gigantic", "vast", "megacity", "skyline", "horizon", "epic", "towering", "enormous", "storm", "planet", "hull"}
CROWD = {"crowd", "crowds", "rows", "survivors", "people", "faces", "passengers", "villagers"}
HANDS = {"hand", "hands", "grip", "grips", "grab", "grabs", "turn", "turns", "crank", "cranks", "push", "pushes", "shove", "shoves", "pull", "pulls", "press", "presses", "open", "opens", "lift", "lifts"}


def words_of(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def classify_shot(shot: dict, plan: dict | None = None) -> str:
    """The NEED of a shot from what it is (kind, who is in it, what it shows and what moves)."""
    kind = shot.get("kind")
    if kind == "talk":
        return "talking_closeup"
    if kind == "choice":
        return "person_idle_closeup"
    if kind == "offer":
        return "two_person_offer"
    text = words_of(" ".join([shot.get("motion", ""), shot.get("still", ""), shot.get("visual", "")]))
    people = [i for i in shot.get("in_shot", []) if i]
    if kind == "pov" or "first-person" in (shot.get("still") or "").lower():
        return "pov_action" if text & (HANDS | LOCOMOTION) else "scene_camera_move"
    if people:
        return "person_locomotion" if text & LOCOMOTION else "person_idle_closeup"
    if text & CROWD and not text & EPIC:
        return "crowd"
    if text & EPIC:
        return "establishing_epic"
    return "scene_camera_move"


def verdict_from_metrics(need: str, metrics: dict) -> str | None:
    """A status decided by a MEASURE when the need has a threshold: motion for moving people, blinks for faces. None = a person has to look."""
    if need in ("person_locomotion", "pov_action") and "motion_amount" in metrics:
        return "works" if metrics["motion_amount"] >= 1.2 else "partial" if metrics["motion_amount"] >= 0.6 else "fails"
    if need in ("talking_closeup", "person_idle_closeup") and "blinks" in metrics:
        return "works" if metrics["blinks"] >= 1 else "fails"
    return None


class Registry:
    def __init__(self, path: Path = REGISTRY_FILE):
        self.path = path
        self.records: list[dict] = json.loads(path.read_text(encoding="utf-8"))["records"] if path.exists() else []
        known = {(r["need"], r["method"]) for r in self.records}
        for need, method, status, note in SEED:
            if (need, method) not in known:
                self.records.append({"need": need, "method": method, "status": status, "evidence": [{"when": "2026-10-08", "source": "seed", "note": note}]})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"needs": NEEDS, "methods": METHODS, "records": self.records}, indent=1, ensure_ascii=False), encoding="utf-8")

    def record(self, need: str, method: str, status: str, note: str, metrics: dict | None = None, source: str | None = None) -> dict:
        if need not in NEEDS or method not in METHODS or status not in RANK:
            raise ValueError(f"unknown need / method / status: {need} {method} {status}")
        evidence = {"when": time.strftime("%Y-%m-%d"), "source": source or "manual", "note": note, **({"metrics": metrics} if metrics else {})}
        for record in self.records:
            if record["need"] == need and record["method"] == method:
                record["status"] = status
                record["evidence"].append(evidence)
                break
        else:
            record = {"need": need, "method": method, "status": status, "evidence": [evidence]}
            self.records.append(record)
        self.save()
        return record

    def candidates(self, need: str) -> list[dict]:
        found = [r for r in self.records if r["need"] == need and r["status"] != "fails"]
        return sorted(found, key=lambda r: (-RANK[r["status"]], -len(r["evidence"])))

    def choose(self, need: str, prefer: str | None = None) -> dict:
        """The best method for a need: 'works' first, then 'partial', then an 'untested' one (marked trial: the result is recorded afterwards). A method that fails is never chosen."""
        options = self.candidates(need)
        if not options:
            raise LookupError(f"no method that does not fail is known for {need}")
        picked = next((r for r in options if r["method"] == prefer), options[0]) if prefer else options[0]
        return {"need": need, "method": picked["method"], "status": picked["status"], "trial": picked["status"] == "untested", "settings": METHODS[picked["method"]],
                "why": picked["evidence"][-1]["note"]}

    def table(self) -> str:
        lines = []
        for need in NEEDS:
            lines.append(f"{need}: {NEEDS[need]}")
            for r in sorted((r for r in self.records if r["need"] == need), key=lambda r: -RANK[r["status"]]):
                lines.append(f"   {r['status']:8} {r['method']:18} {r['evidence'][-1]['note'][:110]}")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show")
    verdict = sub.add_parser("verdict")
    verdict.add_argument("need")
    verdict.add_argument("method")
    verdict.add_argument("status", choices=list(RANK))
    verdict.add_argument("note")
    args = parser.parse_args()
    registry = Registry()
    if args.command == "verdict":
        registry.record(args.need, args.method, args.status, args.note, source="user choice")
    registry.save()
    print(registry.table())


if __name__ == "__main__":
    main()
