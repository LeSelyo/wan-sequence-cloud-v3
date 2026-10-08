"""The MOTION LAB: variants of the same shot made with different settings (engine, prompt, LoRA, steps, nodes), put side by side with their settings written under each panel, so that the best one is
CHOSEN BY LOOKING. Every experiment writes its videos in results/story_trend/lab/<project>/<experiment>/ and a COMMENTS.md that says what each variant is and what to look at.

    python scripts/motion_lab.py PROJECT_DIR [--only identity,motion,blink,offer,pov] [--url http://127.0.0.1:18188]

PROJECT_DIR = a project of scripts/story_auto.py (results/story_trend/auto/NAME): its plan, its close-ups (cards/closeups, the face of every character) and its stills are the material.
The app on the box must run with SageAttention OFF (Qwen-Image-Edit gives black pictures with it) and ComfyUI is reached through the tunnel on --url.

Experiments
  identity  the CHARACTER STAYS THE SAME in a wide / action shot: Qwen-Image-Edit puts the person of the close-up (Picture 2) into a scene (Picture 1); two prompt styles; also the two-person shot.
  motion    people WALK / RUN: S2V (audio-driven: people stay in their pose) against Wan I2V (made for movement) with three prompt styles.
  blink     do the eyes BLINK? prompts / engines compared, blinks counted in the clips.
  offer     the shot where BOTH characters hold out a hand toward you.
  pov       first-person shots where YOUR hands do something (real action instead of a trembling arm).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy_edit as ce  # noqa: E402
import face_tools as ft  # noqa: E402
import i2v_action as i2v  # noqa: E402
import lab_tools as lt  # noqa: E402
import s2v_talk as st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "results" / "story_trend" / "lab"
REGISTRY = ROOT / "results" / "story_trend" / "generations.json"
SILENCE_SECONDS = 3.2

# ---------------------------------------------------------------------------------------------- the prompts compared (kept here so that the report can quote them)
EDIT_SIMPLE = "Put the {who} from Picture 2 into the scene of Picture 1, {pose}."
EDIT_STRICT = ("Replace the person in Picture 1 by the {who} from Picture 2: exactly the same face, the same hair, the same clothes and the same age as in Picture 2. Keep the place, the light and the camera "
               "angle of Picture 1. {pose}. Full body visible, natural proportions.")
EDIT_TWO = ("Put the man from Picture 2 and the woman from Picture 3 side by side in the place of Picture 1, close together, both facing the camera, each one holding out one open hand toward the camera. "
            "Keep their faces, hair and clothes exactly as in Pictures 2 and 3. Same place, same light.")
MOTION_PLAIN = "{subject} {action}."
MOTION_ACTION = "{subject} {action}, real steps and body movement, arms swinging, clothes and hair moving, natural weight and balance, the whole body in motion."
MOTION_FOLLOW = "{subject} {action}, real steps and body movement, arms swinging, clothes and hair moving; the camera follows the movement smoothly at the same speed, steady, cinematic."
BLINK_BASE = "{who} speaks to the camera, small natural head movements, cinematic, realistic skin, sharp eyes"
BLINK_PROMPT = "{who} speaks to the camera, blinks naturally every two or three seconds (the eyelids close and open fully), breathes, small head and eyebrow movements, eyes moving slightly, cinematic, realistic skin"
BLINK_NEGATIVE = "static eyes, frozen face, eyes always open, no blinking, blurry, deformed face, extra fingers, distorted mouth, subtitles, text, watermark, worst quality, low quality"


class Lab:
    def __init__(self, project: Path, url: str, world: bool = False):
        """world=True: the pictures of the shots come from the world lab (`lab/<project>/world/<shot>_after_seed*.png`, made with the corrected style) and everything goes to `lab/<project>_world`."""
        self.project, self.url = project, url
        self.world = LAB / project.name / "world" if world else None
        self.plan = json.loads((project / "plan.json").read_text(encoding="utf-8"))
        self.cards = json.loads((project / "cards" / "cards.json").read_text(encoding="utf-8"))
        self.run = project / "run"
        self.out = LAB / (project.name + ("_world" if world else ""))
        self.out.mkdir(parents=True, exist_ok=True)
        self.report: list[str] = []

    # ------------------------------------------------------------ material
    def character(self, character_id: str) -> dict:
        entry = self.cards["characters"][character_id]
        info = next(c for c in self.plan["characters"] if c["id"] == character_id)
        closeup = ROOT / (entry.get("closeup") or entry["portrait"])["file"]
        return {"id": character_id, "name": info["name"], "who": f"{'woman' if info['gender'] == 'f' else 'man'} ({info['role']})", "closeup": closeup, "face": (ft.detect_faces(closeup) or [None])[0], "info": info}

    def still(self, shot_id: str) -> Path:
        if self.world:
            made = sorted(self.world.glob(f"{shot_id}_after_seed*.png"))
            if made:
                return made[-1]
        return self.run / "stills" / f"{shot_id}.png"

    def note(self, text: str) -> None:
        self.report.append(text)

    def silent_wav(self) -> Path:
        import story_produce as sp
        return sp.padded_wav(None, SILENCE_SECONDS, self.out / "silence.wav")

    # ------------------------------------------------------------ engines
    def edit(self, scene: Path, refs: list[Path], prompt: str, name: str, seed: int = 1) -> Path:
        dst = self.out / "stills" / f"{name}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            ce.run_edit(scene, dst, prompt, seed=seed, steps=4, references=refs, base_url=self.url, registry=REGISTRY, label=f"lab_{self.project.name}_{name}")
        return dst

    def make_i2v(self, start: Path, prompt: str, name: str, profile: str = "lightx2v4", seed: int = 1, seconds: float = 3.0) -> dict:
        dst = self.out / "clips" / f"{name}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        entry = {"seconds": None}
        if not dst.exists():
            entry = i2v.run_i2v(start, dst, prompt=prompt, seed=seed, profile=profile, seconds=seconds, base_url=self.url, registry=REGISTRY, label=f"lab_{self.project.name}_{name}")
        return {"clip": dst, "gen_seconds": entry.get("seconds")}

    def make_s2v(self, start: Path, prompt: str, name: str, audio: Path | None = None, lora: str = "t2v_1217_low", seed: int = 1, seconds: float = 3.0, negative: str | None = None) -> dict:
        dst = self.out / "clips" / f"{name}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        entry = {"seconds": None}
        if not dst.exists():
            kwargs = {"negative": negative} if negative else {}
            entry = st.run_talk(start, audio or self.silent_wav(), dst, prompt=prompt, seed=seed, lora=lora, seconds=seconds, base_url=self.url, registry=REGISTRY, label=f"lab_{self.project.name}_{name}", **kwargs)
        return {"clip": dst, "gen_seconds": entry.get("seconds")}

    # ------------------------------------------------------------ experiments
    def identity(self) -> None:
        """Same person, wide shot: Qwen-Image-Edit with the close-up as reference, two prompt styles."""
        kael, elara = self.character("c1"), self.character("c2")
        cases = [("elara_runs", elara, "s039", "she is running toward the camera in mid-stride, one foot off the ground"),
                 ("kael_walks", kael, "s026", "he is walking toward the camera in mid-stride, arms swinging")]
        lines = ["# Identity pass: the SAME person in a wide shot", "",
                 "Problem: a wide still made from the text description gives a stranger (the close-up has the real face). Fix tested: Qwen-Image-Edit-2511 (4 steps, Lightning LoRA) receives the scene as Picture 1 and the "
                 "close-up as Picture 2 and swaps the person. Two prompt styles: A = short, B = strict (same face, hair, clothes, keep light and angle).", ""]
        for name, who, shot, pose in cases:
            scene = self.still(shot)
            a = self.edit(scene, [who["closeup"]], EDIT_SIMPLE.format(who=who["who"].split(" (")[0], pose=pose), f"{name}_A")
            b = self.edit(scene, [who["closeup"]], EDIT_STRICT.format(who=who["who"].split(" (")[0], pose=pose[0].upper() + pose[1:]), f"{name}_B")
            panels = [{"clip": scene, "label": "scene (before)", "lines": [f"still {shot} made from the text description", "the person is a stranger"]},
                      {"clip": who["closeup"], "label": f"reference: {who['name']}", "lines": ["the close-up used for the voice and the face", "this face must appear in every shot"]},
                      {"clip": a, "label": "A: short prompt", "lines": [EDIT_SIMPLE.format(who="person", pose=pose)]},
                      {"clip": b, "label": "B: strict prompt", "lines": ["same face/hair/clothes as Picture 2, keep light and angle of Picture 1, full body"]}]
            lt.compose(panels, self.out / f"identity_{name}.mp4", f"IDENTITY - {who['name']} in a wide shot", seconds=2.0)
            lines.append(f"* `identity_{name}.mp4`: {who['name']} ({shot}). Look at: is it the same face, hair and clothes as the reference? A or B?")
        scene, refs = self.still("s004"), [kael["closeup"], elara["closeup"]]
        two = self.edit(scene, refs, EDIT_TWO, "two_shot_hands")
        lt.compose([{"clip": scene, "label": "scene (before)", "lines": ["still s004: two strangers made from the text"]}, {"clip": kael["closeup"], "label": f"reference: {kael['name']}"},
                    {"clip": elara["closeup"], "label": f"reference: {elara['name']}"},
                    {"clip": two, "label": "two-shot, hands out", "lines": ["Pictures 2 and 3 = the two close-ups", EDIT_TWO[:160]]}], self.out / "identity_two_shot.mp4", "IDENTITY - both characters in ONE shot", seconds=2.0)
        lines.append("* `identity_two_shot.mp4`: both characters side by side holding out a hand (the shot of the offer). Look at: are BOTH faces right?")
        (self.out / "COMMENTS_identity.md").write_text("\n".join(lines), encoding="utf-8")

    def motion(self) -> None:
        """People who walk and run: S2V (audio-driven) against Wan I2V with three prompt styles, from the identity-fixed stills."""
        cases = [("elara_runs", "the woman", "runs down the corridor toward the camera"), ("kael_walks", "the man", "walks steadily forward through the corridor toward the camera")]
        lines = ["# Motion: do people really MOVE?", "",
                 "S2V (Wan 2.2 Sound-to-Video, 4-step LoRA) is driven by AUDIO: with silence the person keeps the starting pose. I2V (Wan 2.2 Image-to-Video, two experts high/low noise, lightx2v 4-step LoRAs) is "
                 "built for movement. Prompt styles: plain / action (steps, arms, hair, weight) / follow (the camera tracks the movement).", ""]
        for name, subject, action in cases:
            start = self.out / "stills" / f"{name}_A.png"  # the SHORT edit prompt kept the clothes of the character (the strict one kept the old outfit)
            if not start.exists():
                continue
            variants = [("S2V + silence (the current method)", self.make_s2v(start, f"{subject} {action}, cinematic", f"{name}_s2v"), ["engine S2V, LoRA t2v_1217 (4 steps), silent audio", "prompt: plain"]),
                        ("I2V, plain prompt", self.make_i2v(start, MOTION_PLAIN.format(subject=subject.capitalize(), action=action), f"{name}_i2v_plain"), ["engine I2V two experts, lightx2v v1 4 steps", f"prompt: {MOTION_PLAIN.format(subject=subject.capitalize(), action=action)}"]),
                        ("I2V, action prompt", self.make_i2v(start, MOTION_ACTION.format(subject=subject.capitalize(), action=action), f"{name}_i2v_action"), ["same engine", "prompt: + real steps, arms swinging, clothes and hair moving, weight"]),
                        ("I2V, action + camera follows", self.make_i2v(start, MOTION_FOLLOW.format(subject=subject.capitalize(), action=action), f"{name}_i2v_follow"), ["same engine", "prompt: + the camera follows at the same speed"])]
            panels = [{"clip": v["clip"], "label": label, "lines": [*notes, f"motion amount {lt.motion_amount(v['clip'])} | made in {v['gen_seconds']} s"]} for label, v, notes in variants]
            lt.compose(panels, self.out / f"motion_{name}.mp4", f"MOTION - {name.replace('_', ' ')}", seconds=3.0)
            lines.append(f"* `motion_{name}.mp4`: look at: does the person walk/run (feet, arms, weight)? Does the face stay the same? Which prompt style looks best?")
        (self.out / "COMMENTS_motion.md").write_text("\n".join(lines), encoding="utf-8")

    def blink(self) -> None:
        """Do the eyes blink? Prompts and engines compared on a speaking close-up and a listening close-up."""
        names_done: list[str] = []
        for cid in ("c2", "c1"):
            who = self.character(cid)
            if who["face"] is None:
                continue
            first = next((s for s in self.plan["shots"] if s["kind"] == "talk" and s["speaker"] == cid), None)
            audio = None
            if first:
                import story_produce as sp
                wav = self.run / "voices" / f"{first['id']}.wav"
                audio = sp.padded_wav(wav, 3.0, self.out / f"audio_{cid}.wav") if wav.exists() else None
            name = who["name"].lower()
            variants = [("S2V, current prompt", self.make_s2v(who["closeup"], BLINK_BASE.format(who=who["name"]), f"blink_{name}_base", audio), ["LoRA t2v_1217, 4 steps", f"prompt: {BLINK_BASE.format(who='X')[2:]}"]),
                        ("S2V + blink prompt", self.make_s2v(who["closeup"], BLINK_PROMPT.format(who=who["name"]), f"blink_{name}_prompt", audio, negative=BLINK_NEGATIVE), ["same, prompt asks for blinks", "negative prompt: no blinking, frozen eyes, eyes always open"]),
                        ("S2V + blink prompt, other seed", self.make_s2v(who["closeup"], BLINK_PROMPT.format(who=who["name"]), f"blink_{name}_prompt2", audio, seed=2, negative=BLINK_NEGATIVE), ["same as B, seed 2", "(is the blink a matter of luck?)"]),
                        ("I2V, blink prompt (no voice)", self.make_i2v(who["closeup"], BLINK_PROMPT.format(who=who["name"]).replace("speaks to the camera", "looks at the camera, lips slightly moving"), f"blink_{name}_i2v"),
                         ["I2V two experts, lightx2v 4 steps", "no audio: the face is free to blink"])]
            names_done.append(name)
        self.blink_report(names_done)

    BLINK_LABELS = {"base": ("S2V, current prompt", ["LoRA t2v_1217, 4 steps", "prompt: speaks, small natural head movements"]),
                    "prompt": ("S2V + blink prompt", ["same, prompt asks for blinks", "negative prompt: no blinking, frozen eyes, eyes always open"]),
                    "prompt2": ("S2V + blink prompt, other seed", ["same as B, seed 2", "(is the blink a matter of luck?)"]),
                    "i2v": ("I2V, blink prompt (no voice)", ["I2V two experts, lightx2v 4 steps", "no audio: the face is free to blink"])}

    def blink_report(self, names: list[str], suffix: str = "_v2") -> None:
        """Blinks COUNTED ON THE BOX with face landmarks (eye aspect ratio): the first version of this lab counted the contrast of the eye band and said 0 everywhere while the eyelids closed. A blink lasts
        under 0.6 s; a longer closure is the eyes shut (not a blink). The videos of this report end with `suffix` so the first ones are kept."""
        import story_produce as sp
        box = sp.Box("n1.de.clorecloud.net", 1380, "~/.ssh/id_ed25519_clore")
        lines = ["# Blinking (counted on the box with face landmarks)", "", "A blink = the eye aspect ratio under 70 % of its open value for up to 0.6 s. A person blinks about once every 3-4 s (15-20 per minute); the clips last 3 s.", ""]
        for name in names:
            clips = {key: self.out / "clips" / f"blink_{name}_{key}.mp4" for key in self.BLINK_LABELS}
            measured = lt.blinks_on_box([c for c in clips.values() if c.exists()], box)
            panels = []
            for key, (label, notes) in self.BLINK_LABELS.items():
                clip = clips[key]
                m = measured.get(clip.name, {})
                if not clip.exists() or m.get("blinks") is None:
                    continue
                shut = f" | EYES SHUT {m['longest_closed_s']} s" if m["long_closures"] else ""
                panels.append({"clip": clip, "label": label, "lines": [*notes, f"BLINKS: {m['blinks']} in 3 s (natural: 1 per 3-4 s){shut}"]})
                lines.append(f"* {name} / {label}: {m['blinks']} blink(s), longest closed {m['longest_closed_s']} s, long closures {m['long_closures']}")
            lt.compose(panels, self.out / f"blink_{name}{suffix}.mp4", f"BLINK - {name.capitalize()}", seconds=3.0)
        (self.out / f"COMMENTS_blink{suffix}.md").write_text("\n".join(lines), encoding="utf-8")

    def offer(self) -> None:
        """The shot where both characters hold out a hand: I2V prompts on the two-shot, S2V with a voice line."""
        two = self.out / "stills" / "two_shot_hands.png"
        if not two.exists():
            return
        lines = ["# The offer: both characters hold out a hand", ""]
        variants = [("I2V, hands slowly toward you", self.make_i2v(two, "The man and the woman slowly stretch their open hands toward the camera, their fingers reaching, they look at the viewer, blink, small head movements", "offer_i2v_a"),
                     ["I2V two experts, 4 steps", "prompt: both stretch their hands toward the camera, blink, small head movements"]),
                    ("I2V, they step forward", self.make_i2v(two, "The man and the woman step forward together toward the camera and hold out their open hands, urgent faces, the camera pushes slowly in", "offer_i2v_b"),
                     ["same engine", "prompt: they take a step forward while holding out the hands, slow push-in"])]
        first = next((s for s in self.plan["shots"] if s["kind"] == "talk" and s["speaker"] == "c1"), None)
        if first and (self.run / "voices" / f"{first['id']}.wav").exists():
            import story_produce as sp
            audio = sp.padded_wav(self.run / "voices" / f"{first['id']}.wav", 3.0, self.out / "audio_offer.wav")
            variants.append(("S2V with the first line", self.make_s2v(two, "The man and the woman hold out their hands toward the camera, the man speaks, blinking, small head movements", "offer_s2v", audio), ["S2V, LoRA t2v_1217", f"audio: the line of {first['id']}"]))
        panels = [{"clip": v["clip"], "label": label, "lines": [*notes, f"motion amount {lt.motion_amount(v['clip'])} | made in {v['gen_seconds']} s"]} for label, v, notes in variants]
        lt.compose(panels, self.out / "offer.mp4", "THE OFFER - both characters hold out a hand", seconds=3.0)
        lines.append("* `offer.mp4`: look at: do BOTH hands come toward the camera? Are both faces stable? Is a step forward better than a static pose?")
        (self.out / "COMMENTS_offer.md").write_text("\n".join(lines), encoding="utf-8")

    def pov(self) -> None:
        """First-person shots with a REAL action of the hands."""
        lines = ["# POV: real action of YOUR hands", ""]
        cases = [("valve", "s002", "Your gloved hands grip the valve wheel and crank it hard, the wheel turns, steam bursts out, the camera pushes in", "your hands crank the valve; steam"),
                 ("push", "s019", "Your hands shove the person ahead through the narrow hatch, you climb through behind him, the camera moves forward with you", "you push and climb through the hatch")]
        for name, shot, action, note in cases:
            start = self.still(shot)
            if not start.exists():
                continue
            variants = [("S2V + silence (current)", self.make_s2v(start, action, f"pov_{name}_s2v"), ["engine S2V 4 steps", "the arm trembles, nothing happens"]),
                        ("I2V, action prompt", self.make_i2v(start, action, f"pov_{name}_i2v"), ["I2V two experts, 4 steps", f"prompt: {action}"]),
                        ("I2V, 6 steps", self.make_i2v(start, action, f"pov_{name}_i2v6", profile="lightx2v6"), ["I2V 6 steps", "more time for the action to develop"])]
            panels = [{"clip": v["clip"], "label": label, "lines": [*notes, f"motion amount {lt.motion_amount(v['clip'])} | made in {v['gen_seconds']} s"]} for label, v, notes in variants]
            lt.compose(panels, self.out / f"pov_{name}.mp4", f"POV - {note}", seconds=3.0)
            lines.append(f"* `pov_{name}.mp4`: {note}")
        (self.out / "COMMENTS_pov.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path)
    parser.add_argument("--only", default="identity,motion,blink,offer,pov")
    parser.add_argument("--url", default="http://127.0.0.1:18188")
    parser.add_argument("--world", action="store_true", help="start from the pictures of the world lab (corrected decor) and write to lab/<project>_world")
    args = parser.parse_args()
    lab = Lab(args.project.resolve(), args.url, args.world)
    for name in args.only.split(","):
        started = time.time()
        print(f"[{name}] ...", flush=True)
        try:
            getattr(lab, name)()
        except Exception as error:  # one failing experiment never stops the others
            print(f"[{name}] FAILED: {str(error)[:300]}", flush=True)
            continue
        print(f"[{name}] done in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
