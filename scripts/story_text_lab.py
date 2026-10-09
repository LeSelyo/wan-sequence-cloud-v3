"""TEXT LAB of the story route: ONLY the text agents (idea -> analyst -> planner -> writer -> director), with the model used today (qwen3.6:27b through Ollama), no picture or video model. It tests the NEW route:
the model INVENTS the place and the two characters (any genre, fantasy included) from the drawn ingredients, the size of the context follows the length of the video, the structure is context -> offer -> choice
at once -> whole branch A -> whole branch B.

    python scripts/story_text_lab.py --out results/story_trend/text_lab_2026-10-09 --ideas 12 --stories 4 --seconds 120 [--model qwen3.6:27b] [--seed-start 1]

Ollama must answer on 127.0.0.1:11434 (a tunnel to the box). Writes <out>/report.json and <out>/REPORT.md (everything a person has to read to judge: the bones, the invented place and characters, the context, the beats
with their purposes, the lines, the pictures prompts, the checks).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_agents as ag  # noqa: E402
import story_eval as ev  # noqa: E402
import story_library as lib  # noqa: E402

TREND = ag.load_trend("you_must_choose")


def idea_range(seeds: list[int], seconds: int, llm, given: dict | None = None) -> list[dict]:
    """The idea agent alone: what it invents for each seed (place, premise, context, title) and what the seed drew."""
    rows = []
    for seed in seeds:
        started = time.time()
        context, report = ag.run_idea({"seed": seed, "target_seconds": seconds, **(given or {})}, TREND, llm)
        size = ag.context_size(context)
        rows.append({"seed": seed, "seconds": round(time.time() - started, 1), "source": report["source"], "bones": report["bones"], "title": report.get("title"), "invented": report.get("invented"),
                     "context": context, "size": size, "asked_sentences": ag.context_sentences({"target_seconds": seconds, **(given or {})}, TREND), "attempts": report.get("attempts"), "problems": report.get("problems")})
        print(f"idea seed {seed}: {rows[-1]['seconds']} s, {size['sentences']} sentences, {report['source']}: {(report.get('invented') or {}).get('setting') or report['bones'].get('setting')}", flush=True)
    return rows


def structure_checks(plan: dict) -> dict:
    """The rules of the new structure measured on the finished plan (by code, nothing is judged by a model here)."""
    shots = plan["shots"]
    kinds = [s["kind"] for s in shots]
    offers = [i for i, k in enumerate(kinds) if k == "offer"]
    choice = kinds.index("choice") if "choice" in kinds else None
    names = {c["id"]: c["name"] for c in plan["characters"]}
    branch = {b: [s for s in shots if s["branch"] == b and s["kind"] != "rewind"] for b in ("A", "B")}
    other_named = {b: [s["id"] for s in shots_ if names[("c2" if b == "A" else "c1")].lower() in s["text"].lower()] for b, shots_ in branch.items()}
    return {"offers_adjacent": len(offers) == 2 and offers[1] == offers[0] + 1, "choice_right_after_offers": bool(offers) and choice == offers[-1] + 1,
            "shots_before_offers": offers[0] if offers else None, "talk_shots": kinds.count("talk"), "pov_share": round(kinds.count("pov") / max(1, len(kinds)), 2),
            "branch_sizes": {b: len(v) for b, v in branch.items()}, "other_character_named_in_branch": other_named,
            "twists": [(s.get("ending"), s["text"]) for s in shots if s["kind"] == "twist"], "last_lines": {b: [s["text"] for s in v[-2:]] for b, v in branch.items()},
            "characters_in_shots": {i: sum(1 for s in shots if i in s.get("in_shot", [])) for i in names}}


def story_range(seeds: list[int], seconds: int, llm, outline_candidates: int = 2, library=None, given: dict | None = None) -> list[dict]:
    rows = []
    for seed in seeds:
        started = time.time()
        plan = ag.make_plan(None, {"target_seconds": seconds, **(given or {})}, seed, llm, outline_candidates=outline_candidates, voices=[], library=library)
        elapsed = round(time.time() - started)
        metrics = ev.plan_metrics(plan, plan["context"], plan["params"]["target_seconds"])
        rows.append({"seed": seed, "seconds": elapsed, "plan": plan, "metrics": metrics, "structure": structure_checks(plan)})
        print(f"story seed {seed}: {elapsed} s, {len(plan['shots'])} shots, problems left {plan['agents']['problems_left']}", flush=True)
    return rows


def mmss(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def length_summary(stories: list[dict]) -> dict | None:
    """What the creation decided about the length of each story (None when the stories had a fixed length): the rows, and the bounds the trend promises (never under 2:00, never over 10:00, 2:30 on average or more)."""
    rows = []
    for r in stories:
        plan = r["plan"]
        length = (plan.get("agents") or {}).get("length")
        if not length:
            continue
        scale = plan["params"].get("shot_scale", 1.0)
        rows.append({"seed": r["seed"], "title": plan["title"], "chosen": length["chosen"], "asked": length.get("asked"), "source": length["source"], "why": length.get("why"), "events": length.get("events", []),
                     "shots": len(plan["shots"]), "speech_seconds": plan["estimated_seconds"], "video_estimate": round(len(plan["shots"]) * TREND.SECONDS_PER_SHOT * scale), "text_seconds": r["seconds"]})
    if not rows:
        return None
    chosen = [row["chosen"] for row in rows]
    low, high = TREND.LENGTH_RANGE
    return {"rows": rows, "min": min(chosen), "max": max(chosen), "mean": round(sum(chosen) / len(chosen)), "within_bounds": all(low <= c <= high for c in chosen), "mean_at_least_2m30": sum(chosen) / len(chosen) >= 150}


def length_lines(summary: dict) -> list[str]:
    lines = ["## 0. Length decided by the creation (nobody fixed it)", "",
             f"{len(summary['rows'])} stories: shortest {mmss(summary['min'])}, longest {mmss(summary['max'])}, **average {mmss(summary['mean'])}** (the bounds are {mmss(TREND.LENGTH_RANGE[0])} to {mmss(TREND.LENGTH_RANGE[1])}, "
             f"within bounds: {summary['within_bounds']}, average at least 2:30: {summary['mean_at_least_2m30']}).", "",
             "| seed | title | decided | shots | speech (estimate) | video (shots x 3.8 s) | text generation | source |", "|---|---|---|---|---|---|---|---|"]
    for row in summary["rows"]:
        lines.append(f"| {row['seed']} | {row['title']} | {mmss(row['chosen'])} | {row['shots']} | {mmss(row['speech_seconds'])} | {mmss(row['video_estimate'])} | {mmss(row['text_seconds'])} | {row['source']} |")
    lines.append("")
    for row in summary["rows"]:
        lines += [f"- **seed {row['seed']}, {mmss(row['chosen'])}** (the model asked for {row['asked']} s): {row['why']}", f"  - key events it counted: {'; '.join(row['events'])}"]
    return lines + [""]


def briefs_range(ideas: list[dict], seconds: int, llm) -> list[dict]:
    """The ANALYST on each invented context: the characters it creates (name, role, look, wardrobe, promise, hidden truth, voice), the world and the places. This is the 'character creation' step."""
    rows = []
    for idea in ideas:
        started = time.time()
        brief, report = ag.run_analyst(idea["context"], {"language": "en", "tone": idea["bones"]["tone"], "seed": idea["seed"], "target_seconds": seconds}, TREND, llm)
        rows.append({"seed": idea["seed"], "seconds": round(time.time() - started, 1), "source": report["source"], "attempts": report.get("attempts"), "problems": report.get("problems"), "brief": brief})
        print(f"brief seed {idea['seed']}: {rows[-1]['seconds']} s, {report['source']}: {[c['name'] + ' (' + c['role'] + ')' for c in brief['characters']]}", flush=True)
    return rows


def character_lines(brief: dict) -> list[str]:
    out = []
    for ch in brief.get("characters", []):
        out += [f"- **{ch['id']} {ch['name']}**, {ch['role']} ({ch['gender']}, {ch['age']} years)", f"  - look: {ch['look']}", f"  - wardrobe: {ch['wardrobe']}", f"  - promises you: {ch['public_promise']}",
                f"  - hidden truth: {ch['hidden_truth']}", f"  - voice: {ch['voice_style']}"]
    return out


def write_characters_report(out: Path, ideas: list[dict], briefs: list[dict], stories: list[dict], settings: dict) -> None:
    """A report that STARTS with the characters: for every idea the analyst's creation, then the whole story chains."""
    by_seed = {b["seed"]: b for b in briefs}
    md = [f"# Characters, places and stories created by the model ({settings['model']}, {settings['seconds']} s videos)", "",
          "Sections: **1. Characters and world created by the analyst for each invented idea** (below) · 2. the ideas (full contexts) · 3. the whole text chains (beats and lines).", "", "## 1. Characters created, idea by idea", ""]
    for r in ideas:
        b = by_seed.get(r["seed"])
        md += [f"### seed {r['seed']}: {r.get('title')}  (genre {r['bones'].get('genre')}, kind of place: {r['bones'].get('kind')}, suggested names {r['bones'].get('names')})"]
        if b:
            world = b["brief"].get("world", {})
            md += [f"- place invented by the idea agent: **{(r.get('invented') or {}).get('setting')}**", f"- world kept by the analyst: {world.get('setting')} | hour {world.get('hour')} | {world.get('atmosphere')}", ""]
            md += character_lines(b["brief"])
            md += ["", "- places: " + "; ".join(f"{loc['id']} ({', '.join(loc['tags'])})" for loc in b["brief"].get("locations", [])), ""]
        else:
            md += ["(analyst not run for this seed)", ""]
    md += ["## 2. The ideas (full contexts)", ""]
    for r in ideas:
        md += [f"### seed {r['seed']}: {r.get('title')}  ({r['size']['sentences']} sentences / {r['size']['words']} words)", r["context"], ""]
    if stories:
        md += ["## 3. Whole text chains", ""]
        for r in stories:
            plan = r["plan"]
            md += [f"### seed {r['seed']}: {plan['title']}  ({len(plan['shots'])} shots, ~{plan['estimated_seconds']} s of speech, problems left {plan['agents']['problems_left']})", "", "**Characters:**", *character_lines(plan["brief"]), "",
                   "**Beats and lines:**", *beat_lines(plan), ""]
    (out / "REPORT_CHARACTERS.md").write_text("\n".join(md), encoding="utf-8")


def beat_lines(plan: dict) -> list[str]:
    lines = []
    for s in plan["shots"]:
        beat = s.get("beat") or {}
        tag = f"{s['kind']}{'/' + s['offer_of'] if s.get('offer_of') else ''}"
        lines.append(f"- **{s['id']}** `{s['branch']}/{tag}/{s['speaker']}` ({beat.get('act', '?')}) {s['text']}  \n  _purpose:_ {beat.get('purpose', '-')}  \n  _still:_ {s.get('still') or '-'}  \n  _motion:_ {s.get('motion')}")
    return lines


def write_report(out: Path, ideas: list[dict], stories: list[dict], settings: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"settings": settings, "ideas": ideas, "stories": stories}, indent=1, ensure_ascii=False), encoding="utf-8")
    auto = settings.get("auto_length")
    md = [f"# Text lab: {settings['model']}, {'length decided by the creation (2:00 to 10:00)' if auto else str(settings['seconds']) + ' s videos'}{', shots at ' + str(settings['shot_scale']) + ' of their normal time (TEST)' if settings.get('shot_scale') else ''}, prompt version {TREND.PROMPT_VERSION}", ""]
    summary = length_summary(stories)
    if summary:
        md += length_lines(summary)
    md += ["## 1. The idea agent alone (what the model invents from the drawn ingredients)", "",
           f"The context is asked at about {ideas[0]['asked_sentences']} sentences for {settings['seconds']} s." if ideas else "", ""]
    for r in ideas:
        b = r["bones"]
        md += [f"### seed {r['seed']}: {r.get('title') or '(template)'}  ({r['source']}, {r['seconds']} s, {r['size']['sentences']} sentences / {r['size']['words']} words)",
               f"- drawn: ingredients {b.get('ingredients')}, hour {b['hour']}, tone {b['tone']}", f"- invented: place **{(r.get('invented') or {}).get('setting')}**, premise: {(r.get('invented') or {}).get('premise')}", "", r["context"], ""]
    md += ["## 2. Whole text chains (idea -> analyst -> planner -> writer -> director)", ""]
    for r in stories:
        p, m, c = r["plan"], r["metrics"], r["structure"]
        brief = p.get("brief", {})
        md += [f"### seed {r['seed']}: {p['title']}  ({r['seconds']} s, {len(p['shots'])} shots, ~{p['estimated_seconds']} s of speech)", f"- source: {p['story_source']}; problems left: {p['agents']['problems_left']}; judge: {p['agents'].get('judge')}",
               f"- endings {p['endings']}; checks: offers adjacent {c['offers_adjacent']}, choice right after {c['choice_right_after_offers']}, shots before the offers {c['shots_before_offers']}, talk shots {c['talk_shots']}, "
               f"pov share {c['pov_share']}, branch sizes {c['branch_sizes']}, other character named in a branch {c['other_character_named_in_branch']}", "",
               f"**Context given to the analyst ({ag.context_size(p['context'])['sentences']} sentences):** {p['context']}", "",
               f"**World:** {brief.get('world', {}).get('setting')} | hour {brief.get('world', {}).get('hour')} | {brief.get('world', {}).get('atmosphere')}", "", "**The two characters made by the model:**"]
        for ch in brief.get("characters", []):
            md += [f"- **{ch['id']} {ch['name']}**, {ch['role']} ({ch['gender']}, {ch['age']}): {ch['look']} | wears: {ch['wardrobe']} | promises: {ch['public_promise']} | hidden: {ch['hidden_truth']}"]
        md += ["", "**Places:** " + "; ".join(f"{loc['id']} ({', '.join(loc['tags'])}): {loc['description']}" for loc in brief.get("locations", [])), "", "**Beats and lines:**", *beat_lines(p), ""]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ideas", type=int, default=12)
    parser.add_argument("--stories", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--auto-length", action="store_true", help="nobody fixes the length: the creation decides it (2 to 10 minutes); --seconds is ignored")
    parser.add_argument("--shot-scale", type=float, help="TEST ONLY: multiplies the time a shot stays on screen (0.4 = about 1.5 s per shot)")
    parser.add_argument("--outline-candidates", type=int, default=2, help="outlines tried by the planner for a video that is not planned by parts (1 saves time)")
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--briefs-from", type=Path, help="reuse the ideas of an earlier run (its report.json), run the ANALYST on each and write REPORT_CHARACTERS.md there")
    args = parser.parse_args()
    llm = ag.ollama(args.model, args.host)
    if args.briefs_from:
        earlier = json.loads((args.briefs_from / "report.json").read_text(encoding="utf-8"))
        try:
            briefs = briefs_range(earlier["ideas"], earlier["settings"]["seconds"], llm)
        finally:
            llm.unload()
        write_characters_report(args.briefs_from, earlier["ideas"], briefs, earlier["stories"], earlier["settings"])
        (args.briefs_from / "briefs.json").write_text(json.dumps(briefs, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"report: {args.briefs_from / 'REPORT_CHARACTERS.md'}")
        return
    ideas, stories = [], []
    started = time.time()
    given = {k: v for k, v in {"length_mode": "auto" if args.auto_length else None, "shot_scale": args.shot_scale}.items() if v is not None}
    try:
        ideas = idea_range(list(range(args.seed_start, args.seed_start + args.ideas)), args.seconds, llm, given) if args.ideas else []
        write_report(args.out, ideas, stories, vars(args) | {"out": str(args.out)})
        empty = lib.StoryLibrary(args.out / "no_examples", seed=False)  # no approved story is shown to the writer: the test measures the prompts, not the examples
        stories = story_range(list(range(args.seed_start, args.seed_start + args.stories)), args.seconds, llm, outline_candidates=args.outline_candidates, library=empty, given=given)
    finally:
        llm.unload()
        if ideas or stories:
            write_report(args.out, ideas, stories, vars(args) | {"out": str(args.out), "total_seconds": round(time.time() - started)})
    print(f"report: {args.out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
