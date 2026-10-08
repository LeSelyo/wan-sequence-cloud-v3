"""The LIBRARY OF APPROVED STORIES of a trend: every plan the user approves is stored here with its keywords, and the agents of scripts/story_agents.py get the two most similar approved stories as
examples of STYLE (how the hook, the offers, the twists and the pictures are written) in their prompts. The prompts stay constant: only these examples change, and they improve with what the user validates.

    lib = StoryLibrary()                      # results/story_trend/story_library
    lib.add(plan, approved=True, author="user")
    lib.examples_text(brief, k=2)             # a few lines of text for the WRITER prompt
    lib.director_examples_text(brief, k=2)    # a few picture/motion pairs for the DIRECTOR prompt

The first entry is the story Claude wrote by hand for the first 2m30 video (author "claude"): the seed, so that the chain has an example from the first run. A story the model wrote and nobody approved is
stored with approved=False and never used as an example.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "results" / "story_trend" / "story_library"
SEED_PLAN = ROOT / "results" / "story_trend" / "plans" / "the_last_city_150_en.json"
STOP = {"the", "and", "you", "are", "with", "that", "this", "from", "have", "your", "them", "they", "their", "what", "when", "into", "each", "one", "two", "who", "for", "not", "but"}


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


def plan_keywords(plan: dict) -> set[str]:
    base = tokens(plan.get("context", "") + " " + plan.get("title", "") + " " + plan.get("logline", ""))
    base |= {w.lstrip("#") for w in re.findall(r"#\w+", plan.get("caption", ""))}
    base |= tokens(" ".join(c.get("role", "") for c in plan.get("characters", [])))
    if plan.get("brief"):
        base |= tokens(" ".join(plan["brief"].get("keywords", [])))
    return base


class StoryLibrary:
    def __init__(self, root: Path = LIBRARY, seed: bool = True):
        self.root = root
        self.index_path = root / "index.json"
        self.entries: list[dict] = json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else []
        if seed and SEED_PLAN.exists() and not any(e["id"] == "the_last_city_150_en" for e in self.entries):
            self.add(json.loads(SEED_PLAN.read_text(encoding="utf-8")), approved=True, author="claude", entry_id="the_last_city_150_en")

    def add(self, plan: dict, approved: bool = False, author: str = "llm", scores: dict | None = None, entry_id: str | None = None, save: bool = True) -> dict:
        entry_id = entry_id or re.sub(r"\W+", "_", plan.get("title", "story").lower())[:40] + time.strftime("_%Y%m%d%H%M%S")
        self.entries = [e for e in self.entries if e["id"] != entry_id]
        stored = (self.root / "plans" / f"{entry_id}.json")
        entry = {"id": entry_id, "title": plan.get("title"), "approved": approved, "author": author, "scores": scores or {}, "keywords": sorted(plan_keywords(plan)), "plan": str(stored.relative_to(ROOT)) if stored.is_relative_to(ROOT) else str(stored),
                 "language": plan.get("language", "en"), "added": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.entries.append(entry)
        if save:
            stored.parent.mkdir(parents=True, exist_ok=True)
            stored.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
            self.index_path.write_text(json.dumps(self.entries, indent=1, ensure_ascii=False), encoding="utf-8")
        return entry

    def approve(self, entry_id: str, scores: dict | None = None) -> None:
        for entry in self.entries:
            if entry["id"] == entry_id:
                entry["approved"] = True
                entry["scores"] = scores or entry["scores"]
        self.index_path.write_text(json.dumps(self.entries, indent=1, ensure_ascii=False), encoding="utf-8")

    def similar(self, brief: dict, k: int = 2, exclude: set[str] | None = None) -> list[dict]:
        """The k approved stories whose keywords overlap most with the brief's (keywords, roles, setting)."""
        wanted = tokens(" ".join(brief.get("keywords", [])) + " " + brief["world"]["setting"] + " " + brief["world"]["premise"] + " " + " ".join(c["role"] for c in brief["characters"]))
        scored = []
        for entry in self.entries:
            if not entry["approved"] or (exclude and entry["id"] in exclude):
                continue
            overlap = len(wanted & set(entry["keywords"])) / max(1, len(wanted | set(entry["keywords"])))
            scored.append((overlap, entry))
        scored.sort(key=lambda pair: -pair[0])
        return [e for _, e in scored[:k]]

    def load_plan(self, entry: dict) -> dict:
        path = Path(entry["plan"])
        return json.loads((path if path.is_absolute() else ROOT / path).read_text(encoding="utf-8"))

    def examples_text(self, brief: dict, k: int = 2, exclude: set[str] | None = None) -> str:
        """Spoken lines of approved stories, as STYLE examples: the hook, an exchange of the offers, the twist of each ending (about 20 short lines per story)."""
        blocks = []
        for entry in self.similar(brief, k, exclude):
            shots = self.load_plan(entry)["shots"]
            hook = shots[:3]
            talks = [s for s in shots if s["kind"] == "talk"][:4]
            twists = [s for s in shots if s["kind"] == "twist"]
            lines = [f"  {s['kind']}/{s['speaker']}: {s['text']}" for s in (*hook, *talks, *twists)]
            blocks.append(f"Approved story \"{entry['title']}\" (other plot, copy only the STYLE):\n" + "\n".join(lines))
        return "\n".join(blocks)

    def director_examples_text(self, brief: dict, k: int = 2, exclude: set[str] | None = None) -> str:
        """A few picture + camera pairs of approved stories: the first (grandiose) shot, a pov, a fast action shot, a twist."""
        blocks = []
        for entry in self.similar(brief, k, exclude):
            shots = [s for s in self.load_plan(entry)["shots"] if s.get("still")]
            picks = [shots[0], next((s for s in shots if s["kind"] == "pov"), shots[1]), next((s for s in shots if (s.get("fx") or {}).get("shake", 0) >= 0.9), shots[2]), next((s for s in shots if s["kind"] == "twist"), shots[-1])]
            blocks.append(f"Approved story \"{entry['title']}\":\n" + "\n".join(f"  still: {s['still']}\n  motion: {s['motion']}" for s in picks))
        return "\n".join(blocks)
