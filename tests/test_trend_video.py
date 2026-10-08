import json
import subprocess

import numpy as np
from PIL import Image

from scripts import music_timing as mt
from scripts import trend_video as tv


def _plan(seed=7, **kw):
    return tv.make_plan(seed, 20.0, **kw)


def test_plan_is_reproducible_and_changes_with_the_seed():
    assert _plan(7) == _plan(7)
    assert _plan(7)["cuts"] != _plan(8)["cuts"] or _plan(7)["shots"] != _plan(8)["shots"]


def test_cuts_fill_exactly_the_requested_seconds_on_the_beat_like_the_reference():
    plan = _plan(7)
    frames = [c["frames"] for c in plan["cuts"]]
    assert sum(frames) == 20 * tv.FPS
    assert all(f >= 10 for f in frames)  # the reference's fastest cuts are about 0.4 s
    assert min(frames[:4]) >= 28 and max(frames[:3]) <= 36 and 55 <= frames[3] <= 70  # it opens with longer shots (1.0, 1.0, 1.0, 2.1 s)
    assert sum(1 for f in frames[4:] if f <= 17) >= 4  # then bursts


def test_the_plan_keeps_the_user_choices_no_people_no_bird_and_mixes_families():
    plan = _plan(7)
    assert not {s["family"] for s in plan["shots"]} & tv.EXCLUDED_FAMILIES
    assert len({s["family"] for s in plan["shots"]}) >= 3
    assert plan["estimate"]["gpu_seconds"] <= 1500 or not tv.library_available()


def test_jump_cuts_seen_from_above_move_away_and_never_go_wider_than_the_picture():
    for seed in (7, 11, 3):
        plan = _plan(seed)
        zoom_shots = [s for s in plan["shots"] if s["source"] == "zoomout"]
        assert zoom_shots, "the planner always draws views from above that move away when the catalogue has them"
        for shot in zoom_shots:
            cuts = [c for c in plan["cuts"] if c["shot"] == shot["id"]]
            assert 3 <= len(cuts) <= 4
            starts = [c["zoom"][0] for c in cuts]
            assert starts == sorted(starts, reverse=True) and cuts[-1]["zoom"][1] == 1.0  # each cut JUMPS to a wider view, the last one ends on the whole picture
            assert all(z >= 1.0 for c in cuts for z in c["zoom"])


def test_light_twins_come_from_a_shot_that_is_really_in_the_plan_and_library_files_exist():
    for seed in (7, 11, 5):
        plan = _plan(seed)
        ids = {s["id"] for s in plan["shots"]}
        for shot in plan["shots"]:
            if shot["source"] == "light":
                assert shot["base"] in ids and shot["edit"] in tv.LIGHT_EDITS
            if shot["source"] == "library":
                assert (tv.RESULTS / shot["library"]).exists()
        assert {c["shot"] for c in plan["cuts"]} <= ids
        for cut in plan["cuts"]:
            shot = next(s for s in plan["shots"] if s["id"] == cut["shot"])
            if shot["source"] != "zoomout":
                assert cut["t0"] >= 0 and cut["t0"] + cut["frames"] / tv.FPS <= 6.5


def test_status_lists_what_each_model_family_still_has_to_do(tmp_path, monkeypatch):
    monkeypatch.setattr(tv, "RESULTS", tmp_path)
    plan = _plan(7)
    missing = tv.needed(plan)
    assert len(missing["stills"]) == sum(1 for s in plan["shots"] if s["source"] in ("wan", "zoomout"))
    assert len(missing["qwen"]) == sum(1 for s in plan["shots"] if s["source"] == "light")
    assert missing["assemble"] == [plan["name"]] and missing["local"] == []  # nothing local to do before the stills and clips exist


def test_a_zoom_cut_is_rendered_with_the_exact_number_of_frames(tmp_path):
    still = tmp_path / "still.png"
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((90, 160, 3)) * 255).astype(np.uint8)).save(still)
    out = tmp_path / "cut.mp4"
    tv.render_zoom_cut(still, out, 15, 2.0, 1.0)
    text = subprocess.run([mt.ffmpeg_binary(), "-hide_banner", "-i", str(out), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"], capture_output=True, text=True).stderr
    frames = int([line for line in text.splitlines() if line.startswith("frame=")][-1].split("frame=")[1].split()[0])
    assert frames == 15 and f"{tv.SIZE[0]}x{tv.SIZE[1]}" in text
