"""The quality report measures by itself what a person used to find by eye on the video."""
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_qa as qa  # noqa: E402


def write_wav(path: Path, seconds: float, amplitude: float, rate: int = 24000) -> None:
    t = np.arange(int(seconds * rate)) / rate
    data = (amplitude * np.sin(2 * np.pi * 200 * t) * 32767).astype("int16")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())


def project(tmp_path, *, template=False, repeat=False, pov_run=False, level_gap=False, heard_ok=True):
    shots = []
    sentences = ["The ribcage glows", "Wax weeps down the stairs", "You climb a cracked vertebra", "Ozone burns your throat", "Mirrors show your regrets", "A hand reaches from the dark",
                 "He offers a safe harbor", "She offers an endless dream", "Choose before the floor breaks", "Heat washes over your face", "The beast opens one eye", "Silence swallows every sound"]
    for i in range(12):
        kind = "pov" if (pov_run and i >= 2) else ("pov" if i % 2 else "narration")
        shots.append({"id": f"s{i + 1:03d}", "kind": kind, "branch": "A" if i < 6 else "B", "text": "The same line again" if (repeat and i in (3, 4)) else sentences[i]})
    chunks = [{"beats": 12, "source": "template" if template else "llm", "attempts": 3, "problems": []}]
    plan = {"shots": shots, "params": {"target_seconds": 30, "shot_scale": 1.0}, "estimated_speech_seconds": 30,
            "agents": {"planner": {"source": "llm"}, "writer": {"chunks": chunks}, "director": {"chunks": [{"source": "llm"}]}, "look": {"source": "llm"}, "problems_left": []}}
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "report.json").write_text(json.dumps({"steps": {"story": {"seconds": 100}, "render": {"seconds": 50, "seconds_of_video": 31.0}}}), encoding="utf-8")
    voices = tmp_path / "run" / "voices"
    voices.mkdir(parents=True)
    align = {}
    for i, shot in enumerate(shots):
        write_wav(voices / f"{shot['id']}.wav", 2.0, 0.05 if (level_gap and i % 2) else 0.3)
        words = shot["text"].split() if heard_ok else ["blah"]
        align[shot["id"]] = [{"word": w, "start": 0.1 + k * 0.3, "end": 0.35 + k * 0.3} for k, w in enumerate(words)]
    (voices / "align.json").write_text(json.dumps(align), encoding="utf-8")
    (voices / "manifest.json").write_text(json.dumps({"passages": [{"id": "g01"}]}), encoding="utf-8")
    (tmp_path / "run" / "stills_judge_1.json").write_text(json.dumps({s["id"]: {"major": ["x"] if i < 2 else [], "ok": i >= 2} for i, s in enumerate(shots)}), encoding="utf-8")
    return tmp_path


def test_a_clean_project_has_no_warning_and_the_report_is_written(tmp_path):
    facts = qa.collect(project(tmp_path))
    assert facts["warnings"] == [] and facts["voice"]["exact_lines"] == 12 and facts["voice"]["passages"] == 1 and facts["stills"]["first_round_flagged"] == 2
    out = qa.write(tmp_path)
    assert out.name == "QA.md" and "nothing measurable is wrong" in out.read_text(encoding="utf-8") and (tmp_path / "qa.json").exists()


def test_every_defect_the_user_found_by_eye_is_a_warning(tmp_path):
    facts = qa.collect(project(tmp_path, template=True, repeat=True, pov_run=True, level_gap=True, heard_ok=False))
    text = "\n".join(facts["warnings"])
    assert "TEMPLATE" in text and "repeated" in text and "first-person shots in a row" in text and "level jumps" in text and "heard exactly as written" in text
    assert qa.markdown(facts).startswith("# Quality report") and "## To look at before publishing" in qa.markdown(facts)


def test_a_video_far_from_its_target_length_and_pictures_still_flagged_are_warnings(tmp_path):
    root = project(tmp_path)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    plan["params"]["target_seconds"] = 100
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    flagged = {f"s{i + 1:03d}": {"major": ["wrong place"], "ok": False} for i in range(8)}
    (root / "run" / "stills_judge_2.json").write_text(json.dumps(flagged), encoding="utf-8")
    text = "\n".join(qa.collect(root)["warnings"])
    assert "the video lasts 31 s for a target of 100 s" in text and "still flags 8 of 8 pictures after 2 round(s)" in text


def test_an_unresolved_problem_the_writer_kept_is_listed_but_the_remarks_of_clarity_are_only_numbers(tmp_path):
    root = project(tmp_path)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    plan["agents"]["writer"]["chunks"][0]["problems"] = ["line 1 is confusing for a viewer who sees it once (x)", "this branch must end GOOD for you but its last lines read BAD (x)"]
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    warnings = qa.collect(root)["warnings"]
    assert any("must end GOOD" in w for w in warnings) and not any("confusing" in w for w in warnings)
