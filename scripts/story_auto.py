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
import s2v_face_test as sft  # noqa: E402
import s2v_talk as st  # noqa: E402
import story_cards as sc  # noqa: E402
import story_agents as ag  # noqa: E402
import story_engine as se  # noqa: E402
import story_library as lib  # noqa: E402
import story_produce as sp  # noqa: E402
import story_render as sr  # noqa: E402
import story_stills as sst  # noqa: E402
import story_writer as sw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AUTO = ROOT / "results" / "story_trend" / "auto"
STEPS = ("story", "cards", "closeups", "stills", "voices", "animate", "qc", "post", "render", "caption")
FAMILY = {"cards": "krea2", "closeups": "krea2", "stills": "krea2", "animate": "s2v", "qc": "s2v", "post": "s2v"}  # the app family each step needs ("voices" and "story" do not depend on it)
CLOSEUP_SEEDS = (11, 22, 33, 44, 55)
TUNNELS = {8000: 8000, 18188: 8188, 11434: 11434}  # local port -> box port
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
        started = time.time()
        label = f"{self.name}_{family}_{int(started) % 100000}"
        self.box.run(f"/root/stop_app.sh; /root/start_app.sh {label} novram")
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
        import httpx
        try:
            url = "http://127.0.0.1:8000/health/live" if family == "krea2" else "http://127.0.0.1:18188/system_stats"
            return httpx.get(url, timeout=5).status_code == 200
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
        return {"stills": len(book)}

    def missing_stills(self) -> list[str]:
        return [s["id"] for s in self.plan()["shots"] if s.get("still") and not (self.run / "stills" / f"{s['id']}.png").exists()]

    def step_voices(self, args) -> dict:
        report = {}
        if any(not v.get("ref_text") for v in json.loads(se.VOICES_FILE.read_text(encoding="utf-8"))["voices"]):
            report["transcribe"] = sp.step_transcribe(self.box, self.plan()["params"]["language"])
        report["voices"] = sp.step_voices(self.box, self.plan(), self.run, args.seed or 1)
        return report

    def step_animate(self, args) -> dict:
        return sp.step_animate(self.plan(), self.cards(), self.run, st.BASE_URL, args.lora, args.seed or 1)

    def step_qc(self, args) -> dict:
        """Measure the face of every talking clip (same numbers as scripts/s2v_face_test.py); a clip far above the median shimmer is made again with another seed (the old one is kept as _vN)."""
        plan, cards = self.plan(), self.cards()
        measured: dict[str, dict] = {}
        for shot in plan["shots"]:
            clip = self.run / "clips" / f"{shot['id']}.mp4"
            if shot["kind"] != "talk" or not clip.exists():
                continue
            portrait = ROOT / sp.portrait_of(cards, shot["speaker"])
            faces = ft.detect_faces(portrait)
            box = ft.face_box(faces[0], 0.05) if faces else sft.FACE_BOX
            measured[shot["id"]] = {"speaker": shot["speaker"], **sft.face_metrics(clip, (480, 832), box)}
        median = statistics.median(m["flicker"] for m in measured.values()) if measured else 0.0
        outliers = [i for i, m in measured.items() if median and m["flicker"] > FLICKER_OUTLIER * median]
        retried = {}
        if outliers:
            retry = sp.step_animate(plan, cards, self.run, st.BASE_URL, args.lora, (args.seed or 1) + 17, set(outliers), force=True)
            for shot_id in outliers:
                clip = self.run / "clips" / f"{shot_id}.mp4"
                new = sft.face_metrics(clip, (480, 832), ft.face_box(ft.detect_faces(ROOT / sp.portrait_of(cards, measured[shot_id]["speaker"]))[0], 0.05) if ft.detect_faces(ROOT / sp.portrait_of(cards, measured[shot_id]["speaker"])) else sft.FACE_BOX)
                kept_new = new["flicker"] < measured[shot_id]["flicker"]
                if not kept_new:  # the retry is not calmer: the first clip comes back
                    old = sorted(clip.parent.glob(f"{shot_id}_v*.mp4"))[-1]
                    clip.replace(clip.with_name(f"{shot_id}_retry_rejected.mp4"))
                    old.replace(clip)
                retried[shot_id] = {"before": measured[shot_id]["flicker"], "after": new["flicker"], "kept": "retry" if kept_new else "first", "seconds": retry.get(shot_id, {}).get("seconds")}
        return {"median_flicker": round(median, 3), "clips": measured, "retried": retried}

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
