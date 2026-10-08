"""THE PROCEDURE: three lines of context in, a finished "you must choose" TikTok video out (story, characters, pictures, voices, animated clips, face-detail pass, montage, sound, caption), every step resumable,
every number measured and written to report.json. Nothing is hand-made: the story is written by the model on the box (scripts/story_writer.py), everything else by the steps below.

    python scripts/story_auto.py run NAME                              (FROM NOTHING: no context at all, the idea agent invents the subject from the seed: the trend A/B is the only input)
    python scripts/story_auto.py run NAME "A flooded city at night. Two strangers on a rescue boat each offer to save you: a ship captain and a doctor." --seconds 150 [--language en] [--seed 1] [--branches 2]
    python scripts/story_auto.py run NAME --plan my_plan.json          (a plan written elsewhere: the story step is skipped)
    python scripts/story_auto.py status NAME                           (which steps are done, what is missing)

Steps, in order, each skipped when its outputs are complete (a crash or a closed laptop loses nothing: run the same command again):
  story     the plan (model on the box -> validation and repairs -> template story if the model never gets it right)           [llm]
  cards     the style of the video + a character sheet and portrait per character                                              [krea2]
  closeups  tight portraits of each character (the face needs pixels: wide portraits gave grainy eyes), the best centred one is kept  [krea2]
  stills    one picture per shot that is not a close-up                                                                        [krea2]
  voices    one voice line per shot (cloned voices, or a voice made from a text description) + the time of every word            [tts]
  animate   one S2V clip per shot (the character speaks his line; a scene moves)                                                [s2v]
  qc        the face of every talking clip is measured; a clip that shimmers more than the others is made again with another seed [s2v]
  post      face-detail pass on the box: bigger frames, denoise in time, 16 -> 30 fps                                           [s2v]
  render    the video: glitch words, title, tags, effects, sound background, closing question (on this PC: light)
  caption   the text and hashtags to post with the video
The box app is switched between families ([krea2] <-> [s2v], one at a time) by this script. The box is reached over ssh (BOX_SSH_HOST / BOX_SSH_PORT / BOX_SSH_KEY); the tunnels
(8000 app, 18188 ComfyUI, 11434 model) are opened when missing. No token is ever written to a file.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clip_post as cp  # noqa: E402
import remote_render as rr  # noqa: E402
import face_tools as ft  # noqa: E402
import lab_tools as lt  # noqa: E402
import s2v_face_test as sft  # noqa: E402
import s2v_talk as st  # noqa: E402
import story_cards as sc  # noqa: E402
import story_identity as sid  # noqa: E402
import story_agents as ag  # noqa: E402
import story_engine as se  # noqa: E402
import story_library as lib  # noqa: E402
import story_produce as sp  # noqa: E402
import story_render as sr  # noqa: E402
import still_judge as sj  # noqa: E402
import story_stills as sst  # noqa: E402
import story_writer as sw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AUTO = ROOT / "results" / "story_trend" / "auto"
STEPS = ("story", "cards", "closeups", "stills", "identity", "voices", "animate", "qc", "post", "render", "caption")
FAMILY = {"cards": "lab", "closeups": "lab", "stills": "lab", "identity": "lab", "animate": "lab", "qc": "lab", "post": "lab"}  # the app family each step needs ("voices" and "story" do not depend on it)
CLOSEUP_SEEDS = (11, 22, 33, 44, 55)
TUNNELS = {8000: 8000, 18188: 8188, 11434: 11434}  # local port -> box port
IDENTITY_MAJOR = ("the person is not the character", "something in the picture is physically impossible")  # the only judge verdicts that make an identity picture again (the place and the action come from the still, already judged)
PICTURE_RETRIES = 2  # a picture the judge finds wrong is made again this many times at most (another seed each time, the earlier ones are kept)
MIN_MOTION = 1.0  # an I2V clip of a person who moves or of hands that act with less picture change than this (frozen) is made again
MIN_SECONDS_TO_BLINK = 2.5  # a talking clip at least this long with no blink is made again
FLICKER_OUTLIER = 1.35  # a talking clip whose face shimmers more than this times the median of its characters is made again


class Pipeline:
    def __init__(self, name: str, ssh_host: str, ssh_port: int, ssh_key: str):
        self.name = name
        self.root = AUTO / name
        self.run = self.root / "run"
        self.cards_dir = self.root / "cards"
        self.plan_path = self.root / "plan.json"
        self.out = self.root / f"{name}.mp4"
        self.report_path = self.root / "report.json"
        self.box = sp.Box(ssh_host, ssh_port, ssh_key)
        self.ssh = (ssh_host, ssh_port, os.path.expanduser(ssh_key))
        self.report = json.loads(self.report_path.read_text(encoding="utf-8")) if self.report_path.exists() else {"steps": {}}
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ what is there
    def plan(self) -> dict:
        return json.loads(self.plan_path.read_text(encoding="utf-8"))

    def cards(self) -> dict:
        return json.loads((self.cards_dir / "cards.json").read_text(encoding="utf-8"))

    def status(self) -> dict[str, tuple[bool, str]]:
        """step -> (done, detail)."""
        result: dict[str, tuple[bool, str]] = {}
        has_plan = self.plan_path.exists()
        result["story"] = (has_plan, "plan.json" if has_plan else "no plan yet")
        if not has_plan:
            for step in STEPS[1:]:
                result[step] = (False, "waiting for the plan")
            return result
        plan = self.plan()
        shots = plan["shots"]
        cards_ok = (self.cards_dir / "cards.json").exists()
        result["cards"] = (cards_ok, "cards.json" if cards_ok else "missing")
        characters = self.cards()["characters"] if cards_ok else {}
        picked = [c for c in plan["characters"] if "closeup" in characters.get(c["id"], {})]
        result["closeups"] = (cards_ok and len(picked) == len(plan["characters"]), f"{len(picked)}/{len(plan['characters'])} close-ups picked")
        need = [s["id"] for s in shots if s.get("still")]
        have = [i for i in need if (self.run / "stills" / f"{i}.png").exists()]
        result["stills"] = (len(have) == len(need), f"{len(have)}/{len(need)} stills")
        todo_identity = sid.missing(plan, self.cards(), self.run) if cards_ok and len(have) == len(need) else None
        result["identity"] = (todo_identity == [], "waiting for the stills" if todo_identity is None else f"{len(sid.jobs(plan, self.cards(), self.run)) - len(todo_identity)}/{len(sid.jobs(plan, self.cards(), self.run))} identity pictures")
        voices = [s["id"] for s in shots if (self.run / "voices" / f"{s['id']}.wav").exists()]
        aligned = (self.run / "voices" / "align.json").exists()
        result["voices"] = (len(voices) == len(shots) and aligned, f"{len(voices)}/{len(shots)} lines, words {'timed' if aligned else 'not timed'}")
        jobs = sp.animate_jobs(plan, self.cards(), self.run) if cards_ok and len(voices) == len(shots) else []
        clips = [j["id"] for j in jobs if (self.run / "clips" / f"{j['id']}.mp4").exists()]
        result["animate"] = (bool(jobs) and len(clips) == len(jobs), f"{len(clips)}/{len(jobs)} clips")
        result["qc"] = (self.report["steps"].get("qc") is not None, "measured" if self.report["steps"].get("qc") else "not run")
        posted = [j["id"] for j in jobs if (self.run / "post" / f"{j['id']}.mp4").exists()]
        result["post"] = (bool(jobs) and len(posted) == len(jobs), f"{len(posted)}/{len(jobs)} clips post-processed")
        result["render"] = (self.out.exists(), self.out.name if self.out.exists() else "not rendered")
        result["caption"] = ((self.root / "caption.txt").exists(), "caption.txt" if (self.root / "caption.txt").exists() else "missing")
        return result

    # ------------------------------------------------------------ the box
    def ensure_tunnels(self) -> None:
        host, port, key = self.ssh
        missing = []
        for local in TUNNELS:
            with socket.socket() as probe:
                probe.settimeout(0.5)
                if probe.connect_ex(("127.0.0.1", local)) != 0:
                    missing.append(local)
        if missing:
            forwards = [arg for local in missing for arg in ("-L", f"{local}:127.0.0.1:{TUNNELS[local]}")]
            subprocess.run(["ssh", "-f", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6", "-o", "BatchMode=yes", "-i", key, "-p", str(port), *forwards,
                            f"root@{host}"], check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)

    def api_token(self) -> str:
        """The token of the app on the box, kept in memory only: the picture client reads it from a module variable (it reads the environment once, at import), so it is set there."""
        import trend_rain_anime as tr
        token = self.box.run("cat /root/.api_token").strip()
        tr.API_TOKEN = token
        os.environ["WAN_API_TOKEN"] = token
        return token

    def switch(self, family: str) -> None:
        """Stop the app and start it for another family (krea2 | s2v). One family at a time: the boxes have one GPU and the models do not share it."""
        state_path = AUTO / "box_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        if state.get("family") == family and self.healthy(family):
            return
        if family == "lab" and self.healthy(family) and self.sage_is_off():  # the app already runs the way the lab did: no restart
            AUTO.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({"family": family, "label": "already running", "at": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
            return
        started = time.time()
        label = f"{self.name}_{family}_{int(started) % 100000}"
        sage = "COMFY_USE_SAGE_ATTENTION=0 " if family in ("qwen", "lab") else ""  # Qwen-Image-Edit gives black pictures with SageAttention; the lab (identity, I2V, S2V, Krea2) ran without it
        self.box.run(f"/root/stop_app.sh; {sage}/root/start_app.sh {label} novram")
        for _ in range(90):
            if self.healthy(family):
                break
            time.sleep(4)
        else:
            raise RuntimeError(f"the app did not come up for {family}")
        state_path.write_text(json.dumps({"family": family, "label": label, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
        self.report.setdefault("switches", []).append({"to": family, "seconds": round(time.time() - started, 1)})

    @staticmethod
    def healthy(family: str) -> bool:
        """The picture app (8000) for krea2, ComfyUI (18188) for the video families, both for the one family of the lab."""
        import httpx
        urls = {"krea2": ["http://127.0.0.1:8000/health/live"], "lab": ["http://127.0.0.1:8000/health/live", "http://127.0.0.1:18188/system_stats"]}.get(family, ["http://127.0.0.1:18188/system_stats"])
        try:
            return all(httpx.get(url, timeout=5).status_code == 200 for url in urls)
        except Exception:
            return False

    @staticmethod
    def sage_is_off() -> bool:
        import comfy_edit as ce
        try:
            return ce.sage_attention_on(st.BASE_URL) is False
        except Exception:
            return False

    # ------------------------------------------------------------ the steps
    def step_story(self, args) -> dict:
        if args.plan:
            self.plan_path.write_text(Path(args.plan).read_text(encoding="utf-8"), encoding="utf-8")
            return {"source": f"given plan {args.plan}"}
        given = {k: v for k, v in {"language": args.language, "target_seconds": args.seconds, "branches": args.branches}.items() if v is not None}
        llm = None if args.no_llm else ag.ollama(args.model)
        if args.story_method == "single":  # one call writes the whole plan (kept to compare with the chain)
            plan = sw.make_story_plan(args.context, given, args.seed, (lambda prompt, schema: llm(prompt, schema, seed=args.seed or 0)) if llm else None)
        else:  # the chain of agents of the trend (analyst, planner, writer, director + the checks)
            plan = ag.make_plan(args.context or None, given, args.seed, llm, trend_name=args.trend, candidates=args.candidates, judge=args.judge, outline_candidates=args.outline_candidates)
        self.plan_path.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
        judged = (plan.get("agents") or {}).get("judge")
        lib.StoryLibrary().add(plan, approved=False, author="llm" if llm else "template", scores=judged or {}, entry_id=self.name)  # kept, but only an APPROVED story is ever used as an example
        return {"source": plan["story_source"], "shots": len(plan["shots"]), "estimated_speech_seconds": plan["estimated_seconds"], "endings": plan["endings"], "judge": judged,
                "problems_left": (plan.get("agents") or {}).get("problems_left")}

    def step_cards(self, args) -> dict:
        plan = self.plan()
        os.environ["WAN_API_TOKEN"] = self.api_token()
        self.cards_dir.mkdir(parents=True, exist_ok=True)
        cards = sc.build_cards(plan, self.cards_dir, style_seed=args.style_seed, only="characters")
        return {"style": cards["style_id"], "characters": len(cards["characters"])}

    def step_closeups(self, args) -> dict:
        plan = self.plan()
        os.environ["WAN_API_TOKEN"] = self.api_token()
        sc.make_closeups(plan, self.cards_dir, list(CLOSEUP_SEEDS))
        picks = {}
        for character in plan["characters"]:
            candidates = sorted((self.cards_dir / "closeups").glob(f"char_{character['id']}_close_*.png"))
            best = ft.pick_closeup(candidates)
            chosen = best[0] if best else candidates[0]  # no face found anywhere: the first one, and the report says so
            seed = int(chosen.stem.rsplit("_", 1)[1])
            sc.set_closeup(self.cards_dir, character["id"], seed)
            picks[character["id"]] = {"seed": seed, "face_found": best is not None, "face": {k: round(v, 2) for k, v in best[1].items()} if best else None}
        return picks

    def step_stills(self, args) -> dict:
        os.environ["WAN_API_TOKEN"] = self.api_token()
        book = sst.make_stills(self.plan(), self.cards(), self.run / "stills", only=set(self.missing_stills()) or None)
        result = {"stills": len(book)}
        if not args.no_picture_check:
            result["picture_check"] = self.check_stills(args)
        return result

    def check_stills(self, args) -> dict:
        """The vision model LOOKS at every still (does it belong to the world, is the action there, no sun in a night world, no collage); a wrong one is made again with another seed, up to PICTURE_RETRIES
        times. The model and the picture jobs do not share the GPU: the model is unloaded before every new batch of pictures."""
        plan, cards, stills = self.plan(), self.cards(), self.run / "stills"
        ask = sj.ollama_vision(args.model)
        rounds, todo, wrong = [], None, []
        for attempt in range(PICTURE_RETRIES + 1):
            try:
                results = sj.judge_stills(plan, stills, ask, todo)
            finally:
                ask.unload()
            wrong = sorted(i for i, r in results.items() if not r["ok"])
            rounds.append({"judged": len(results), "wrong": {i: results[i]["major"] for i in wrong}, "minor_notes": sum(1 for r in results.values() if r["minor"])})
            (self.run / f"stills_judge_{attempt + 1}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
            if not wrong or attempt == PICTURE_RETRIES:
                break
            sst.make_stills(plan, cards, stills, only=set(wrong), seed_shift=1000 * (attempt + 1))
            todo = set(wrong)
        return {"rounds": rounds, "problems_left": wrong}

    def missing_stills(self) -> list[str]:
        return [s["id"] for s in self.plan()["shots"] if s.get("still") and not (self.run / "stills" / f"{s['id']}.png").exists()]

    def step_identity(self, args) -> dict:
        """The same person in every shot: Qwen-Image-Edit puts the close-up of the character (and his or her wardrobe) into the picture of the shot; both characters hold out a hand in the scene of the choice."""
        timings = sid.run_jobs(self.plan(), self.cards(), self.run, st.BASE_URL, args.seed or 1)
        result = {"pictures": len(timings), "timings": timings}
        if not args.no_picture_check:
            result["picture_check"] = self.check_identity(args)
        return result

    def check_identity(self, args) -> dict:
        """The vision model compares every identity picture with the close-up(s) of the character(s): same face and hair, the wardrobe of the character, no glitched head, the place of the world. A wrong one is made
        again with another seed (PICTURE_RETRIES at most, the earlier ones are kept as _vN)."""
        plan, cards = self.plan(), self.cards()
        characters = {c["id"]: c for c in plan["characters"]}
        ask = sj.ollama_vision(args.model)
        rounds, todo, wrong = [], None, []
        for attempt in range(PICTURE_RETRIES + 1):
            items = []
            for job in sid.jobs(plan, cards, self.run):
                picture = self.run / "stills_id" / f"{job['id']}.png"
                if not picture.exists() or (todo and job["id"] not in todo):
                    continue
                shot = next(sh for sh in plan["shots"] if sh["id"] == job["shots"][0])
                who = [characters[i] for i in (plan["characters"][0]["id"], plan["characters"][1]["id"])] if job["id"] == sid.TWO_SHOT_ID else [characters[i] for i in shot["in_shot"] if i in characters][:2]
                people = "; ".join(f"{'on the left ' if k == 0 and len(who) == 2 else 'on the right ' if len(who) == 2 else ''}a {sid.person_word(c)} wearing {c.get('wardrobe', 'the clothes of the portrait').rstrip('. ')}" for k, c in enumerate(who))
                judged_shot = {**shot, "id": job["id"], "still": "Both characters side by side, facing the camera, each one holding out an open hand toward the viewer" if job["id"] == sid.TWO_SHOT_ID else shot["still"]}
                items.append((picture, judged_shot, job["refs"], people))
            try:
                results = sj.judge_images(plan, items, ask)
            finally:
                ask.unload()
            wrong = sorted(i for i, r in results.items() if any(m.startswith(IDENTITY_MAJOR) for m in r["major"]))  # only what a new identity picture can fix: the wrong person, a glitched head
            rounds.append({"judged": len(results), "wrong": {i: results[i]["major"] for i in wrong}, "other_notes": {i: r["major"] for i, r in results.items() if i not in wrong and r["major"]}})
            (self.run / f"identity_judge_{attempt + 1}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
            if not wrong or attempt == PICTURE_RETRIES:
                break
            sid.run_jobs(plan, cards, self.run, st.BASE_URL, seed=(args.seed or 1) + 1000 * (attempt + 1), only=set(wrong), force=True)
            todo = set(wrong)
        return {"rounds": rounds, "problems_left": wrong}

    def step_voices(self, args) -> dict:
        report = {}
        if any(not v.get("ref_text") for v in json.loads(se.VOICES_FILE.read_text(encoding="utf-8"))["voices"]):
            report["transcribe"] = sp.step_transcribe(self.box, self.plan()["params"]["language"])
        report["voices"] = sp.step_voices(self.box, self.plan(), self.run, args.seed or 1)
        return report

    def step_animate(self, args) -> dict:
        return sp.step_animate(self.plan(), self.cards(), self.run, st.BASE_URL, args.lora, args.seed or 1)

    def step_qc(self, args) -> dict:
        """What is measured on the clips, and made again (another seed, the old clip kept as _vN) when it is wrong:
        talking clips (S2V): the shimmer of the face (a clip far above the median) and the BLINKS (counted with face landmarks on the box: none in a clip of 2.5 s or more = made again);
        I2V clips of people who move and hands that act: the motion amount (a frozen picture, under MIN_MOTION, is made again)."""
        plan, cards = self.plan(), self.cards()

        def face_box_of(speaker: str):
            faces = ft.detect_faces(ROOT / sp.portrait_of(cards, speaker))
            return ft.face_box(faces[0], 0.05) if faces else sft.FACE_BOX

        def blinks_of(ids: list[str]) -> dict[str, int | None]:
            clips = [self.run / "clips" / f"{i}.mp4" for i in ids if (self.run / "clips" / f"{i}.mp4").exists()]
            found = lt.blinks_on_box(clips, self.box) if clips else {}
            return {i: (found.get(f"{i}.mp4") or {}).get("blinks") for i in ids}

        measured: dict[str, dict] = {}
        for shot in plan["shots"]:
            clip = self.run / "clips" / f"{shot['id']}.mp4"
            if shot["kind"] != "talk" or not clip.exists():
                continue
            measured[shot["id"]] = {"speaker": shot["speaker"], "seconds": sr.wav_seconds(self.run / "voices" / f"{shot['id']}.wav") + 0.25, **sft.face_metrics(clip, (480, 832), face_box_of(shot["speaker"]))}
        for shot_id, count in blinks_of(list(measured)).items():
            measured[shot_id]["blinks"] = count
        median = statistics.median(m["flicker"] for m in measured.values()) if measured else 0.0
        outliers = {i for i, m in measured.items() if median and m["flicker"] > FLICKER_OUTLIER * median}
        outliers |= {i for i, m in measured.items() if m.get("blinks") == 0 and m["seconds"] >= MIN_SECONDS_TO_BLINK}
        retried = {}
        if outliers:
            retry = sp.step_animate(plan, cards, self.run, st.BASE_URL, args.lora, (args.seed or 1) + 17, set(outliers), force=True)
            new_blinks = blinks_of(sorted(outliers))
            for shot_id in sorted(outliers):
                clip = self.run / "clips" / f"{shot_id}.mp4"
                new = sft.face_metrics(clip, (480, 832), face_box_of(measured[shot_id]["speaker"]))
                before, after = measured[shot_id], {**new, "blinks": new_blinks.get(shot_id)}
                kept_new = ((after["blinks"] or 0) > 0, -after["flicker"]) > ((before.get("blinks") or 0) > 0, -before["flicker"])  # a clip that blinks first, then the calmer face
                if not kept_new:  # the retry is not better: the first clip comes back
                    self.restore_first(clip, shot_id)
                retried[shot_id] = {"before": {"flicker": before["flicker"], "blinks": before.get("blinks")}, "after": {"flicker": after["flicker"], "blinks": after["blinks"]}, "kept": "retry" if kept_new else "first",
                                    "seconds": retry.get(shot_id, {}).get("seconds")}
        motion, frozen = {}, []
        for job in sp.animate_jobs(plan, cards, self.run):
            clip = self.run / "clips" / f"{job['id']}.mp4"
            if job["engine"] == "i2v" and job.get("need") in ("person_locomotion", "pov_action") and clip.exists():
                motion[job["id"]] = lt.motion_amount(clip)
                if motion[job["id"]] < MIN_MOTION:
                    frozen.append(job["id"])
        moved = {}
        if frozen:
            sp.step_animate(plan, cards, self.run, st.BASE_URL, args.lora, (args.seed or 1) + 31, set(frozen), force=True)
            for shot_id in frozen:
                clip = self.run / "clips" / f"{shot_id}.mp4"
                amount = lt.motion_amount(clip)
                if amount <= motion[shot_id]:
                    self.restore_first(clip, shot_id)
                moved[shot_id] = {"before": motion[shot_id], "after": amount, "kept": "retry" if amount > motion[shot_id] else "first"}
        return {"median_flicker": round(median, 3), "clips": measured, "retried": retried, "motion_amount": motion, "frozen_retried": moved}

    @staticmethod
    def restore_first(clip: Path, shot_id: str) -> None:
        """A retry that is not better: the retry is set aside (_retry_rejected) and the first clip, kept as _vN, comes back."""
        old = sorted(clip.parent.glob(f"{shot_id}_v*.mp4"))[-1]
        clip.replace(clip.with_name(f"{shot_id}_retry_rejected.mp4"))
        old.replace(clip)

    def step_post(self, args) -> dict:
        todo = [c for c in sorted((self.run / "clips").glob("*.mp4")) if "_v" not in c.stem and "_retry" not in c.stem and not (self.run / "post" / c.name).exists()]
        if not todo:
            return {"clips": 0}
        result = cp.run_batch(todo, self.run / "post", self.box, keep_remote=f"{rr.REMOTE_ROOT}/jobs/{self.name}/run/post")
        result.pop("log", None)
        return result

    def step_render(self, args) -> dict:
        """The montage runs on the box (32 cores, the PC stays free) unless --render-on pc."""
        if args.render_on == "pc":
            result = sr.render_story(self.plan(), self.cards_dir, self.run, self.out)
            return {"where": "pc", "seconds_of_video": result["total_seconds"], "fallback_shots": result["fallback_shots"], "file": str(self.out)}
        result = rr.render_on_box(self.plan_path, self.cards_dir, self.run, self.out, self.box, self.name)
        inner = result.get("render", {})
        return {"where": "box", "seconds_of_video": inner.get("total_seconds"), "fallback_shots": inner.get("fallback_shots"), "uploaded_clips": result["uploaded_clips"],
                "upload_seconds": result["upload_seconds"], "render_seconds": result["render_seconds"], "file": str(self.out)}

    def step_caption(self, args) -> dict:
        plan = self.plan()
        (self.root / "caption.txt").write_text(plan["caption"] + "\n", encoding="utf-8")
        return {"words": len(plan["caption"].split())}

    # ------------------------------------------------------------ the run
    def go(self, args) -> None:
        sp.OFFER_METHOD = None if getattr(args, "offer_method", "i2v_offer_hands") == "i2v_offer_hands" else args.offer_method
        wanted = [s for s in (args.steps.split(",") if args.steps else STEPS) if s]
        self.ensure_tunnels()
        for step in wanted:
            done, detail = self.status()[step]
            if done and not args.force:
                print(f"[{step}] already done ({detail})", flush=True)
                continue
            if step in FAMILY:
                self.switch(FAMILY[step])
            started = time.time()
            print(f"[{step}] ...", flush=True)
            result = getattr(self, f"step_{step}")(args)
            self.report["steps"][step] = {**(result if isinstance(result, dict) else {}), "seconds": round(time.time() - started, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self.report_path.write_text(json.dumps(self.report, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"[{step}] done in {self.report['steps'][step]['seconds']} s", flush=True)
        print(json.dumps({s: d for s, (ok, d) in self.status().items()}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("name")
    run.add_argument("context", nargs="?", default="")
    run.add_argument("--plan", help="a plan written elsewhere (skips the story step)")
    run.add_argument("--seconds", type=int, default=150)
    run.add_argument("--language", choices=sorted(se.WORDS_PER_SECOND))
    run.add_argument("--branches", type=int, choices=[1, 2])
    run.add_argument("--seed", type=int)
    run.add_argument("--style-seed", type=int, default=0)
    run.add_argument("--lora", choices=list(st.LORAS), default="t2v_1217_low")
    run.add_argument("--model", default="qwen3.6:27b")
    run.add_argument("--no-llm", action="store_true", help="template story, no model")
    run.add_argument("--trend", default="you_must_choose", help="the trend: its prompts, structure and render preset live in scripts/trends/<trend>.py")
    run.add_argument("--story-method", choices=["chain", "single"], default="chain")
    run.add_argument("--candidates", type=int, default=1, help="write the whole story this many times and keep the best (the judge scores them)")
    run.add_argument("--outline-candidates", type=int, default=2, help="outlines tried by the planner, the best one is kept")
    run.add_argument("--offer-method", choices=["i2v_offer_hands", "i2v_offer_step"], default="i2v_offer_hands", help="the choice moment: the two hold out their hands slowly (plan A, default) or step forward (plan B)")
    run.add_argument("--no-picture-check", action="store_true", help="do not let the vision model look at the stills (saves a few minutes)")
    run.add_argument("--judge", action="store_true", help="the model scores the finished story 1-10 on seven criteria")
    run.add_argument("--render-on", choices=["box", "pc"], default="box", help="where the montage (Pillow + ffmpeg) runs: the box by default, the PC stays free")
    run.add_argument("--steps", help="only these steps (comma-separated)")
    run.add_argument("--force", action="store_true", help="make the steps again even when their outputs exist")
    status = sub.add_parser("status")
    status.add_argument("name")
    approve = sub.add_parser("approve", help="the story of this project is good: it becomes an example for the next stories of the trend")
    approve.add_argument("name")
    for p in (run, status):
        p.add_argument("--ssh-host", default=os.environ.get("BOX_SSH_HOST", "n1.de.clorecloud.net"))
        p.add_argument("--ssh-port", type=int, default=int(os.environ.get("BOX_SSH_PORT", "1380")))
        p.add_argument("--ssh-key", default=os.environ.get("BOX_SSH_KEY", "~/.ssh/id_ed25519_clore"))
    args = parser.parse_args()
    if args.command == "approve":
        plan = json.loads((AUTO / args.name / "plan.json").read_text(encoding="utf-8"))
        lib.StoryLibrary().add(plan, approved=True, author="user", entry_id=args.name, scores=(plan.get("agents") or {}).get("judge") or {})
        print(f"{args.name} is now an approved example of the trend")
        return
    pipeline = Pipeline(args.name, args.ssh_host, args.ssh_port, args.ssh_key)
    if args.command == "status":
        for step, (done, detail) in pipeline.status().items():
            print(f"{'OK ' if done else '-- '} {step:9} {detail}")
        return
    pipeline.go(args)


if __name__ == "__main__":
    main()
