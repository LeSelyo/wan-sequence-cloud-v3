"""QUALITY REPORT of a finished project: what the pipeline can measure by itself, so that a defect is SIGNALLED at the end of the run instead of being found by eye on the video.

    python scripts/story_qa.py results/story_trend/auto/NAME        -> NAME/QA.md and NAME/qa.json (also written at the end of `story_auto.py run`)

Every check is objective (counts, durations, levels, what the speech recogniser heard), never a taste. A WARNING is something a person should look at before publishing; an INFO is a number.
  script    a stage that fell back on its template (cut sentences, stage directions read aloud), the premise check that never passed, repeated lines, a branch of first-person shots only
  voice     the pace and the level of the lines, how many lines the recogniser heard exactly as written, the words it heard differently
  pictures  the pictures the vision judge still flagged after its last round (places and identities)
  video     the length against the target, the time of every step
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACE_STD_MAX = 0.9  # words per second
LEVEL_STD_MAX = 1.5  # dB
EXACT_TEXT_MIN = 0.7  # share of lines heard exactly as written
FLAGGED_PICTURES_MAX = 0.3
LENGTH_DEVIATION_MAX = 0.35
POV_RUN_MAX = 4


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def script_facts(plan: dict) -> dict:
    shots = plan["shots"]
    agents = plan.get("agents") or {}
    chunks = (agents.get("writer") or {}).get("chunks", [])
    texts = [" ".join(words(s["text"])) for s in shots]
    exact = len(texts) - len(set(texts))
    near = sum(1 for i in range(len(texts)) for j in range(i + 1, len(texts)) if texts[i] != texts[j] and difflib.SequenceMatcher(None, texts[i], texts[j]).ratio() >= 0.85)
    branches = {}
    for letter in ("A", "B"):
        kinds = [s["kind"] for s in shots if s.get("branch") == letter and s["kind"] != "rewind"]
        run = longest = 0
        for kind in kinds:
            run = run + 1 if kind == "pov" else 0
            longest = max(longest, run)
        branches[letter] = {"shots": len(kinds), "pov_share": round(kinds.count("pov") / max(1, len(kinds)), 2), "longest_pov_run": longest}
    return {"shots": len(shots), "average_words": round(sum(len(s["text"].split()) for s in shots) / max(1, len(shots)), 1), "template_chunks": [i for i, c in enumerate(chunks) if c.get("source") == "template"],
            "chunk_problems": [(i, c["problems"]) for i, c in enumerate(chunks) if c.get("problems")], "planner": (agents.get("planner") or {}).get("source"), "director_templates": [i for i, c in enumerate((agents.get("director") or {}).get("chunks", [])) if c.get("source") == "template"],
            "look": (agents.get("look") or {}).get("source"), "repeated_lines": exact, "near_repeated_lines": near, "branches": branches, "problems_left": agents.get("problems_left", []),
            "target_seconds": plan["params"].get("target_seconds"), "estimated_speech_seconds": plan.get("estimated_seconds"), "shot_scale": plan["params"].get("shot_scale", 1.0)}


def voice_facts(project: Path, plan: dict) -> dict | None:
    folder = project / "run" / "voices"
    align = load(folder / "align.json")
    if not align:
        return None
    import numpy as np
    rows, differences, exact = [], [], 0
    for shot in plan["shots"]:
        path = folder / f"{shot['id']}.wav"
        if not path.exists():
            continue
        with wave.open(str(path), "rb") as handle:
            rate, frames = handle.getframerate(), handle.getnframes()
            x = np.frombuffer(handle.readframes(frames), dtype=np.int16).astype("float64") / 32768
        heard = align.get(shot["id"], [])
        span = (heard[-1]["end"] - heard[0]["start"]) if heard else 0
        hop = int(rate * 0.02)
        voiced = [x[i:i + hop] for i in range(0, len(x) - hop, hop) if float(np.sqrt(np.mean(x[i:i + hop] ** 2))) > 0.015]
        level = 20 * np.log10(float(np.sqrt(np.mean(np.concatenate(voiced) ** 2))) + 1e-9) if voiced else None
        rows.append((len(heard) / span if span > 0.2 else None, level, frames / rate))
        if words(" ".join(w["word"] for w in heard)) == words(shot["text"]):
            exact += 1
        else:
            differences.append({"id": shot["id"], "written": shot["text"], "heard": " ".join(w["word"] for w in heard)})
    paces = [r[0] for r in rows if r[0]]
    levels = [r[1] for r in rows if r[1] is not None]
    manifest = load(folder / "manifest.json") or {}
    return {"lines": len(rows), "seconds": round(sum(r[2] for r in rows), 1), "pace_mean": round(float(np.mean(paces)), 2) if paces else None, "pace_std": round(float(np.std(paces)), 2) if paces else None,
            "pace_min": round(min(paces), 2) if paces else None, "pace_max": round(max(paces), 2) if paces else None, "level_mean": round(float(np.mean(levels)), 1) if levels else None,
            "level_std": round(float(np.std(levels)), 2) if levels else None, "exact_lines": exact, "different_lines": differences, "passages": len(manifest.get("passages", []))}


def judge_facts(project: Path, pattern: str) -> dict | None:
    rounds = sorted((project / "run").glob(pattern), key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
    if not rounds:
        return None
    last = load(rounds[-1])
    flagged = sorted(i for i, v in last.items() if isinstance(v, dict) and v.get("major"))
    return {"rounds": len(rounds), "judged_last_round": len(last), "flagged_after_last_round": flagged, "first_round_flagged": sum(1 for v in (load(rounds[0]) or {}).values() if isinstance(v, dict) and v.get("major"))}


def collect(project: Path) -> dict:
    plan = load(project / "plan.json")
    report = load(project / "report.json") or {"steps": {}}
    facts = {"project": project.name, "script": script_facts(plan), "voice": voice_facts(project, plan), "stills": judge_facts(project, "stills_judge_*.json"),
             "identity": judge_facts(project, "identity_judge_*.json"), "steps_seconds": {k: v.get("seconds") for k, v in report["steps"].items()},
             "video_seconds": (report["steps"].get("render") or {}).get("seconds_of_video"), "clips": len(list((project / "run" / "clips").glob("*.mp4"))) if (project / "run" / "clips").exists() else 0}
    cards = load(project / "cards" / "cards.json") or {"characters": {}}
    facts["characters"] = {cid: {"name": c.get("name"), "observed_traits": c.get("traits_text", "")} for cid, c in cards["characters"].items()}
    facts["total_pipeline_seconds"] = round(sum(v for v in facts["steps_seconds"].values() if v), 1)
    facts["warnings"] = warnings_of(facts)
    return facts


def warnings_of(facts: dict) -> list[str]:
    out = []
    script, voice = facts["script"], facts["voice"]
    if script["template_chunks"]:
        out.append(f"SCRIPT: the lines of chunk(s) {script['template_chunks']} are the TEMPLATE (cut sentences, stage directions read aloud): the model never gave a usable text")
    if script["planner"] == "template" or script["planner"] == "partly template":
        out.append("SCRIPT: the outline (the beats) is partly or wholly the template")
    if script["director_templates"]:
        out.append(f"PICTURES: the picture descriptions of chunk(s) {script['director_templates']} are the template")
    if script["look"] == "template":
        out.append("STYLE: the art direction is the rule-based one (tags, keywords), not written from the world: it can give a world the wrong materials")
    serious = ("the first lines never say", "this branch must end", "the last line closes everything")  # what the writer was asked five times and never gave (the remarks of clarity are only listed)
    for index, problems in script["chunk_problems"]:
        for problem in problems:
            if problem.startswith(serious) or " this branch is only about you and " in problem:
                out.append(f"SCRIPT: chunk {index} kept with an unresolved problem: {problem[:150]}")
    if script["repeated_lines"] or script["near_repeated_lines"]:
        out.append(f"SCRIPT: {script['repeated_lines']} line(s) repeated and {script['near_repeated_lines']} nearly repeated")
    for letter, branch in script["branches"].items():
        if branch["longest_pov_run"] > POV_RUN_MAX:
            out.append(f"SCRIPT: branch {letter} has {branch['longest_pov_run']} first-person shots in a row ({branch['pov_share']:.0%} of the branch): monotonous pictures")
    if script["problems_left"]:
        out.append(f"SCRIPT: the checker leaves {len(script['problems_left'])} problem(s): {script['problems_left'][0][:120]}")
    if voice:
        if voice["pace_std"] and voice["pace_std"] > PACE_STD_MAX:
            out.append(f"VOICE: the pace jumps from line to line ({voice['pace_min']} to {voice['pace_max']} words/s, spread {voice['pace_std']})")
        if voice["level_std"] and voice["level_std"] > LEVEL_STD_MAX:
            out.append(f"VOICE: the level jumps from line to line (spread {voice['level_std']} dB)")
        if voice["lines"] and voice["exact_lines"] / voice["lines"] < EXACT_TEXT_MIN:
            out.append(f"VOICE: only {voice['exact_lines']} of {voice['lines']} lines are heard exactly as written")
    unseen = [c["name"] for c in facts.get("characters", {}).values() if not c["observed_traits"]]
    if unseen:
        out.append(f"FACES: the reference close-ups of {unseen} were never looked at (no observed traits): the identity prompts have the wardrobe only, eyes, scars and hair can drift between scenes")
    for key, label in (("stills", "PICTURES"), ("identity", "FACES")):
        info = facts[key]
        if info and info["judged_last_round"] and len(info["flagged_after_last_round"]) / info["judged_last_round"] > FLAGGED_PICTURES_MAX:
            out.append(f"{label}: the vision judge still flags {len(info['flagged_after_last_round'])} of {info['judged_last_round']} pictures after {info['rounds']} round(s): {info['flagged_after_last_round'][:8]}")
    if facts["video_seconds"] and facts["script"]["target_seconds"]:
        target = float(facts["script"]["target_seconds"])
        if abs(facts["video_seconds"] - target) / target > LENGTH_DEVIATION_MAX:
            out.append(f"LENGTH: the video lasts {facts['video_seconds']:.0f} s for a target of {target:.0f} s")
    return out


def markdown(facts: dict) -> str:
    script, voice = facts["script"], facts["voice"]
    lines = [f"# Quality report: {facts['project']}", "", "## To look at before publishing", ""]
    lines += [f"- **{w}**" for w in facts["warnings"]] or ["- nothing measurable is wrong"]
    lines += ["", "## Script", "", f"- {script['shots']} shots, {script['average_words']} words per line, target {script['target_seconds']} s (speech estimated {script['estimated_speech_seconds']} s), shot scale {script['shot_scale']}",
              f"- planner: {script['planner']}; writer chunks on the template: {script['template_chunks'] or 'none'}; director chunks on the template: {script['director_templates'] or 'none'}; art direction: {script['look']}",
              f"- first-person share: " + ", ".join(f"branch {k}: {v['pov_share']:.0%} (longest run {v['longest_pov_run']})" for k, v in script["branches"].items())]
    if voice:
        lines += ["", "## Voice", "", f"- {voice['lines']} lines, {voice['seconds']} s, {voice['passages']} passage(s); pace {voice['pace_mean']} words/s (spread {voice['pace_std']}, from {voice['pace_min']} to {voice['pace_max']}); level {voice['level_mean']} dB (spread {voice['level_std']} dB)",
                  f"- heard exactly as written: {voice['exact_lines']} of {voice['lines']}"]
        lines += [f"  - {d['id']}: written \"{d['written']}\" / heard \"{d['heard']}\"" for d in voice["different_lines"]]
    if facts.get("characters"):
        lines += ["", "## Characters (as observed on their reference close-up)", ""] + [f"- {c['name']}: {c['observed_traits'] or 'not observed'}" for c in facts["characters"].values()]
    for key, title in (("stills", "Pictures (vision judge)"), ("identity", "Faces (vision judge)")):
        info = facts[key]
        if info:
            lines += ["", f"## {title}", "", f"- {info['rounds']} round(s); flagged in the first: {info['first_round_flagged']}; still flagged after the last of {info['judged_last_round']}: {info['flagged_after_last_round']}"]
    lines += ["", "## Time", "", f"- video: {facts['video_seconds']} s, clips: {facts['clips']}, pipeline: {facts['total_pipeline_seconds']} s ({facts['total_pipeline_seconds'] / 60:.0f} min)", "- " + ", ".join(f"{k} {v:.0f} s" for k, v in facts["steps_seconds"].items() if v)]
    return "\n".join(lines) + "\n"


def write(project: Path) -> Path:
    facts = collect(project)
    (project / "qa.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    out = project / "QA.md"
    out.write_text(markdown(facts), encoding="utf-8")
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else sys.exit(__doc__)
    target = target if target.is_absolute() else ROOT / target
    print(write(target))
    print(markdown(collect(target)).split("## Script")[0])
