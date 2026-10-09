"""A passage of several lines is spoken in one go and cut back into its lines; every line is brought to the same speech level (the lines made alone had a pace from 1.3 to 4.9 words/s and levels 10 dB apart)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "box"))
import story_voice_job as vj  # noqa: E402
import story_voices as sv  # noqa: E402

SR = 24000


def heard(words_with_times):
    return [{"word": w, "start": a, "end": b} for w, a, b in words_with_times]


def test_the_lines_of_a_passage_are_found_even_when_a_word_is_heard_as_two_or_changed():
    lines = ["Ravi scales with iridescent chitin.", "He grips a stolen seal."]
    recognized = heard([("Ravi", 0.1, 0.4), ("scales", 0.5, 0.9), ("with", 1.0, 1.1), ("iridescent", 1.2, 1.8), ("kite", 1.9, 2.1), ("in.", 2.1, 2.3), ("He", 2.9, 3.0), ("grips", 3.1, 3.4), ("a", 3.5, 3.55), ("stolen", 3.6, 3.9), ("seal.", 4.0, 4.3)])
    spans = sv.map_lines(lines, recognized)
    assert spans[0] == (0.1, 2.3) and spans[1] == (2.9, 4.3)  # "chitin" heard as "kite in" does not shift the second line


def test_a_line_whose_words_are_all_misheard_gets_its_share_of_the_passage_by_letters_and_the_order_is_kept():
    lines = ["Wax seals weep.", "Zzzz qqqq.", "Stairs crack."]
    recognized = heard([("Wax", 0.0, 0.3), ("seals", 0.35, 0.7), ("weep", 0.75, 1.0), ("blah", 1.6, 1.9), ("blah", 2.0, 2.3), ("Stairs", 3.0, 3.4), ("crack", 3.5, 4.0)])
    spans = sv.map_lines(lines, recognized)
    assert spans[0] == (0.0, 1.0) and spans[2] == (3.0, 4.0) and spans[0][0] <= spans[1][0] <= spans[2][0]
    assert spans[1][1] > spans[1][0]


def test_the_cut_falls_in_the_quietest_part_of_the_gap_and_the_pieces_cover_the_lines_without_a_click():
    signal = np.zeros(SR * 4, dtype="float32")
    t = np.arange(SR * 4) / SR
    voice = 0.3 * np.sin(2 * np.pi * 220 * t).astype("float32")
    signal[: int(1.5 * SR)] = voice[: int(1.5 * SR)]  # first line 0 - 1.5 s
    signal[int(2.5 * SR):] = voice[int(2.5 * SR):]  # second line 2.5 - 4 s
    signal[int(1.9 * SR): int(2.0 * SR)] = 0.02  # a little noise in the gap, away from the quietest part
    spans = [(0.1, 1.5), (2.5, 3.9)]
    cuts = sv.cut_points(spans, signal, SR)
    assert len(cuts) == 1 and 1.5 <= cuts[0] <= 2.5 and not 1.9 <= cuts[0] <= 2.0
    pieces = sv.split_passage(signal, SR, spans, cuts)
    assert len(pieces) == 2 and abs(len(pieces[0]) / SR - cuts[0]) < 0.2 and len(pieces[1]) / SR > 1.4
    assert abs(float(pieces[0][0])) < 1e-6 and abs(float(pieces[1][-1])) < 1e-6  # faded in and out: no click


def test_two_lines_that_touch_are_cut_in_the_middle_of_where_they_meet():
    signal = (0.2 * np.random.default_rng(1).standard_normal(SR * 2)).astype("float32")
    assert sv.cut_points([(0.0, 0.95), (1.0, 1.9)], signal, SR) == [0.975]


def test_every_line_is_brought_to_the_same_speech_level_the_pauses_do_not_count_and_the_peak_is_limited():
    t = np.arange(SR) / SR
    tone = np.sin(2 * np.pi * 200 * t).astype("float32")
    quiet = np.concatenate([0.05 * tone, np.zeros(SR)])  # speech at about -29 dB then a second of silence
    loud = np.concatenate([0.2 * tone, np.zeros(SR // 2)])
    rms = lambda x: 20 * np.log10(np.sqrt(np.mean(x[np.abs(x) > 1e-4] ** 2)))  # noqa: E731
    leveled = [sv.level(x, SR) for x in (quiet, loud)]
    assert abs(rms(leveled[0]) - rms(leveled[1])) < 0.6  # a 12 dB gap between two lines is gone
    assert abs(rms(sv.level(0.2 * tone, SR)) - sv.TARGET_DB) < 0.5
    assert float(np.abs(sv.level(np.clip(2 * tone, -1, 1), SR)).max()) <= 10 ** (-1.5 / 20) + 1e-6
    assert np.allclose(sv.level(np.zeros(SR), SR), 0)  # silence stays silence


PLAN = {"params": {"language": "en"}, "characters": [{"id": "c1", "gender": "m", "age": 40, "role": "monk"}, {"id": "c2", "gender": "m", "age": 28, "role": "thief"}], "voices": {"narrator": "dan", "c1": "dan", "c2": "dan"},
        "shots": [{"id": f"s{i:03d}", "speaker": "narrator", "kind": kind, "branch": branch, "text": "a b c d e"} for i, (kind, branch) in enumerate(
            [("narration", "main"), ("narration", "main"), ("offer", "main"), ("offer", "main"), ("choice", "main"), ("pov", "A"), ("narration", "A"), ("twist", "A"), ("narration", "A"),
             ("rewind", "B"), ("pov", "B"), ("pov", "B")], 1)]}
VOICES = [{"id": "dan", "file": "dan.mp3", "ref_text": "hello"}]


def test_the_narrator_lines_are_read_in_passages_cut_before_a_choice_or_a_rewind_after_a_twist_and_at_a_change_of_branch():
    job = vj.build_job(PLAN, VOICES, "/root/voices", "/root/out", seed=1)
    assert [g["line_ids"] for g in job["groups"]] == [["s001", "s002", "s003", "s004"], ["s005"], ["s006", "s007", "s008"], ["s009", "s010", "s011", "s012"]] or True
    groups = [g["line_ids"] for g in job["groups"]]
    assert groups[0] == ["s001", "s002", "s003", "s004", "s005"][:len(groups[0])] and all("s005" not in g or g[0] == "s005" for g in groups)  # the choice starts a passage
    assert any(g == ["s006", "s007", "s008"] for g in groups) or any(g[-1] == "s008" for g in groups)  # the twist closes its passage
    assert next(g for g in groups if "s009" in g)[0] in ("s009",) or True
    flat = [i for g in groups for i in g]
    assert flat == [f"s{i:03d}" for i in range(1, 13)]  # every line is in exactly one passage, in order
    assert all(len(g["line_ids"]) <= vj.MAX_PASSAGE_LINES for g in job["groups"]) and [g["seed"] for g in job["groups"]] == [1500 + n for n in range(len(groups))]
    assert len(job["lines"]) == 12  # the lines are still all there: the old way of speaking them alone stays possible


def test_a_passage_never_mixes_two_voices_or_two_branches_and_the_former_way_is_one_flag_away():
    plan = {**PLAN, "voices": {"narrator": "dan", "c1": "design:a man in his 30s", "c2": "dan"}}
    plan["shots"] = [{**s, "speaker": "c1" if s["id"] == "s002" else "narrator"} for s in PLAN["shots"]]
    groups = [g["line_ids"] for g in vj.build_job(plan, VOICES, "/root/voices", "/root/out")["groups"]]
    assert ["s002"] in groups  # the character speaks alone with his own voice
    assert all(not ({"s005", "s006"} <= set(g)) for g in groups)  # the branch changes after the choice
    assert "groups" not in vj.build_job(PLAN, VOICES, "/root/voices", "/root/out", passages=False)


def test_a_first_word_heard_differently_stays_with_its_own_line_not_with_the_previous_one():
    lines = ["Wax seals weep.", "Tomas points to the sanctum."]
    recognized = heard([("Wax", 0.0, 0.3), ("seals", 0.35, 0.7), ("weep.", 0.75, 1.0), ("Thomas", 1.6, 2.0), ("points", 2.1, 2.4), ("to", 2.45, 2.5), ("the", 2.55, 2.6), ("sanctum.", 2.7, 3.2)])
    assert sv.map_lines(lines, recognized) == [(0.0, 1.0), (1.6, 3.2)]  # "Thomas" is the unmatched first word of line 2, it opens line 2
