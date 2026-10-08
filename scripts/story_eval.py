"""PROOF that the automatic story works: the same contexts are written by several METHODS and every plan gets the same automatic measures, then everything is put side by side in a report.

    python scripts/story_eval.py REPORT_DIR --contexts contexts.json [--methods chain,single,hand] [--judge]

Methods: `chain` (scripts/story_agents.py, the four agents + checks), `single` (scripts/story_writer.py: one call writes the whole plan), `hand` (a plan written by a person, here by Claude: the reference
the automatic ones are compared with; given as {"hand": "path/to/plan.json"} in the context entry), `template` (no model).
Measures (no model needed): structure problems left by the checker, shots, spoken words, seconds of speech against the target, lines per character, the keywords of the context that the HOOK uses, how much of the
context's vocabulary survives in the story, variety (distinct first words, repeated word pairs), pictures (length, forbidden words, distinct camera moves), the two endings, the language, and the seconds it took.
With --judge the same model also scores each plan 1-10 on seven criteria (it judges its own writing, so the number is a hint, not a verdict: the report says so).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_agents as sa  # noqa: E402
import story_engine as se  # noqa: E402
import story_writer as sw  # noqa: E402

TREND = sa.load_trend("you_must_choose")
STOP_EN = {"the", "a", "an", "and", "you", "to", "of", "in", "is", "it", "was", "are", "on", "for", "that", "with", "your", "i", "he", "she", "they", "we", "this", "at", "no", "not", "but", "me", "my"}
STOP_FR = {"le", "la", "les", "un", "une", "et", "tu", "vous", "de", "des", "du", "est", "il", "elle", "ne", "pas", "que", "qui", "dans", "pour", "sur", "je", "nous", "ce", "se", "au", "en"}


def content_words(text: str, minimum: int = 5) -> set[str]:
    return {w for w in re.findall(r"[a-zà-ÿ]+", text.lower()) if len(w) >= minimum}


def stem(word: str) -> str:
    return word[:5]


def plan_metrics(plan: dict, context: str, target_seconds: float, language: str = "en") -> dict:
    shots = plan["shots"]
    spoken = [s["text"] for s in shots]
    words = [w for t in spoken for w in re.findall(r"[\w'’-]+", t.lower())]
    stop = STOP_EN if language == "en" else STOP_FR
    hook_text = " ".join(spoken[:3] + [plan.get("title_overlay", {}).get("text", "")] + [(plan.get("caption") or "")]).lower()
    keywords = (plan.get("brief") or {}).get("keywords") or sorted(content_words(context))
    hook_hits = [k for k in keywords if stem(k.lower()) in {stem(w) for w in content_words(hook_text, 4)} or k.lower() in hook_text]
    story_words = {stem(w) for w in content_words(" ".join(spoken) + " " + " ".join((s.get("still") or "") + " " + (s.get("motion") or "") for s in shots), 4)}
    context_words = [w for w in content_words(context, 5)]
    kept = [w for w in context_words if stem(w) in story_words]
    firsts = [t.split()[0].lower() for t in spoken if t.split()]
    bigrams = Counter(zip(words, words[1:]))
    repeated = sum(c - 1 for c in bigrams.values() if c > 1) / max(1, len(words))
    stills = [s["still"] for s in shots if s.get("still")]
    moves = [" ".join(re.findall(r"[a-z]+", (s.get("motion") or "").lower())[:3]) for s in shots]
    forbidden = sum(1 for t in stills for w in TREND.FORBIDDEN_IN_PICTURES if re.search(rf"\b{w}\b", t.lower()))
    talk = Counter(s["speaker"] for s in shots if s["kind"] == "talk")
    twists = [s for s in shots if s["kind"] == "twist"]
    stopword_share = sum(1 for w in words if w in stop) / max(1, len(words))
    problems = sw.validate_story({"characters": plan["characters"], "shots": shots, "hook_title": (plan.get("title_overlay") or {}).get("text", ""), "end_card": plan.get("end_card", ""), "caption": plan.get("caption", "")},
                                 plan["params"], plan.get("endings", {}), [loc["id"] for loc in plan["locations"]])
    return {"shots": len(shots), "spoken_words": len(words), "speech_seconds": sw.estimate_seconds(shots, language), "target_seconds": target_seconds,
            "problems_left": len(problems), "problems": problems[:4], "talk_lines": dict(talk), "twists": [(t.get("ending"), t.get("tag")) for t in twists],
            "hook_keywords": f"{len(hook_hits)}/{len(keywords)}", "context_words_kept": f"{len(kept)}/{len(context_words)}", "distinct_first_words": round(len(set(firsts)) / max(1, len(firsts)), 2),
            "repeated_word_pairs": round(repeated, 3), "mean_line_words": round(len(words) / max(1, len(spoken)), 1), "still_words_mean": round(sum(se.count_words(t) for t in stills) / max(1, len(stills)), 1),
            "forbidden_in_pictures": forbidden, "distinct_camera_moves": f"{len(set(moves))}/{len(moves)}", "dynamic_shots": sum(1 for s in shots if (s.get("fx") or {}).get("shake", 0) >= 0.5),
            "language_stopword_share": round(stopword_share, 2), "rewind": any(s["kind"] == "rewind" for s in shots), "has_choice": any(s["kind"] == "choice" for s in shots)}


def make(method: str, entry: dict, llm, judge: bool, seed: int) -> tuple[dict, float]:
    started = time.time()
    given = {"target_seconds": entry.get("seconds", 150), **({"language": entry["language"]} if entry.get("language") else {})}
    if method == "hand":
        plan = json.loads(Path(entry["hand"]).read_text(encoding="utf-8"))
        plan.setdefault("agents", {})
    elif method == "chain":
        plan = sa.make_plan(entry["context"], given, seed, llm, judge=judge, outline_candidates=entry.get("outline_candidates", 1))
    elif method == "single":
        plan = sw.make_story_plan(entry["context"], given, seed, (lambda prompt, schema: llm(prompt, schema, seed=seed)) if llm else None)
        plan.setdefault("agents", {})
        if judge and llm:
            plan["agents"]["judge"] = sa.judge_plan(entry["context"], plan, TREND, llm)
    elif method == "template":
        plan = sa.make_plan(entry["context"], given, seed, None)
    else:
        raise SystemExit(f"unknown method {method}")
    return plan, round(time.time() - started, 1)


def table(rows: list[dict], keys: list[str]) -> str:
    head = "| " + " | ".join(["method", *keys]) + " |\n|" + "---|" * (len(keys) + 1) + "\n"
    return head + "\n".join("| " + " | ".join([r["method"], *[str(r["metrics"].get(k, r.get(k, ""))) for k in keys]]) + " |" for r in rows)


def excerpt(plan: dict) -> str:
    shots = plan["shots"]
    pick = shots[:3] + [s for s in shots if s["kind"] == "talk"][:3] + [s for s in shots if s["kind"] == "twist"]
    return "\n".join(f"- `{s.get('branch', '?')}/{s['kind']}/{s['speaker']}` {s['text']}" for s in pick)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--methods", default="chain,single,hand")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    llm = None if args.no_llm else sa.ollama(args.model)
    entries = json.loads(args.contexts.read_text(encoding="utf-8"))
    sections = [f"# Automatic story: the chain compared with one call and with a story written by hand\n\nModel `{args.model}`, prompts `{TREND.PROMPT_VERSION}`, seed {args.seed}. The judge (when on) is the same model: a hint, not a verdict.\n"]
    summary = []
    for entry in entries:
        rows = []
        for method in args.methods.split(","):
            if method == "hand" and "hand" not in entry:
                continue
            if (method in ("chain", "single")) and llm is None:
                continue
            if method == "single" and entry.get("skip_single"):
                continue
            print(f"[{entry['id']}] {method} ...", flush=True)
            plan, seconds = make(method, entry, llm, args.judge, args.seed)
            (args.out / f"{entry['id']}_{method}.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
            metrics = plan_metrics(plan, entry["context"], entry.get("seconds", 150), entry.get("language", "en"))
            rows.append({"method": method, "metrics": metrics, "seconds": seconds, "source": plan.get("story_source", "hand-written"), "judge": (plan.get("agents") or {}).get("judge"), "plan": plan})
            print(f"   {seconds} s | {metrics['shots']} shots | problems left {metrics['problems_left']} | {plan.get('story_source', '')[:90]}", flush=True)
        keys = ["shots", "speech_seconds", "problems_left", "hook_keywords", "context_words_kept", "distinct_first_words", "repeated_word_pairs", "mean_line_words", "distinct_camera_moves", "dynamic_shots",
                "forbidden_in_pictures", "language_stopword_share"]
        judged = [r for r in rows if r["judge"]]
        sections.append(f"## {entry['id']}\n\n> {entry['context']}\n\n{table(rows, ['seconds', 'source', *keys])}\n")
        if judged:
            sections.append("Judge (1-10): " + "; ".join(f"**{r['method']}** hook {r['judge']['hook']}, coherence {r['judge']['coherence']}, clues {r['judge']['clues']}, twist {r['judge']['twist']}, voice {r['judge']['voice']}, "
                                                         f"faithfulness {r['judge']['faithfulness']}, variety {r['judge']['variety']} = **{r['judge']['mean']}** ({r['judge']['weakness']})" for r in judged) + "\n")
        for r in rows:
            sections.append(f"### {entry['id']} / {r['method']}\n\nendings: {r['metrics']['twists']} | problems: {r['metrics']['problems']}\n\n{excerpt(r['plan'])}\n")
        summary.append({"id": entry["id"], "rows": [{"method": r["method"], "seconds": r["seconds"], **r["metrics"], "judge": r["judge"]} for r in rows]})
    (args.out / "REPORT.md").write_text("\n".join(sections), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"report: {args.out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
