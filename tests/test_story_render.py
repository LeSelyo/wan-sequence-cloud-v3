import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_render as sr  # noqa: E402


def test_each_word_gets_a_window_in_order_and_the_last_one_ends_with_the_audio():
    windows = sr.word_windows("In the end you understand: it was never about survival.", 3.5)
    assert [w[0] for w in windows] == "In the end you understand it was never about survival".split()
    assert windows[0][1] == 0.08 and abs(windows[-1][2] - 3.5) < 0.01
    assert all(a[2] <= b[1] + 1e-6 for a, b in zip(windows, windows[1:])) and all(w[2] > w[1] for w in windows)


def test_a_word_that_ends_a_sentence_lasts_longer_than_the_same_word_inside_it():
    first, second = sr.word_windows("Go. Go", 4.0)
    assert (first[2] - first[1]) > (second[2] - second[1])


def test_shot_length_is_voice_plus_tail_and_a_choice_never_shorter_than_the_countdown():
    assert sr.shot_seconds("talk", 2.0) == 2.25 and sr.shot_seconds("twist", 3.0) == 5.4 and sr.shot_seconds("narration", 2.0, last=True) == round(2.0 + sr.TAIL["narration"] + sr.END_CARD_SECONDS, 3)
    assert sr.shot_seconds("choice", 1.2) == sr.CHOICE_MIN_SECONDS and sr.shot_seconds("choice", 5.0) == 5.0


def test_word_and_choice_frames_have_the_video_size_and_the_ring_empties():
    assert sr.word_image("SURVIVAL").size == sr.SIZE
    portraits = [Image.new("RGB", (576, 1024), (40, 40, 60)), Image.new("RGB", (576, 1024), (60, 40, 40))]
    first, last = sr.choice_frame(0.5, 3.5, portraits, ["A", "B"]), sr.choice_frame(3.2, 3.5, portraits, ["A", "B"])
    assert first.size == last.size == sr.SIZE
    orange = lambda im: sum(1 for px in im.crop((0, 0, 576, 220)).getdata() if px[0] > 220 and 80 < px[1] < 160 and px[2] < 40)
    assert orange(first) > 3 * orange(last)


def test_subtitle_words_follow_the_real_times_when_the_counts_match_and_are_spread_when_they_do_not():
    aligned = [{"word": "You", "start": 0.2, "end": 0.4}, {"word": "choose", "start": 0.6, "end": 1.1}, {"word": "Brandt", "start": 1.3, "end": 1.9}]
    windows = sr.aligned_windows("You choose Brandt.", aligned, 2.0)
    assert [w[0] for w in windows] == ["You", "choose", "Brandt"] and windows[1][1] == 0.57 and windows[0][2] == windows[1][1]  # a word stays until the next one starts
    spread = sr.aligned_windows("You choose Brandt now.", aligned, 2.0)  # the recogniser heard 3 words, the script has 4
    assert [w[0] for w in spread] == ["You", "choose", "Brandt", "now"] and spread[0][1] == 0.17 and all(a[2] <= b[1] + 1e-6 for a, b in zip(spread, spread[1:]))
    assert sr.aligned_windows("Hi there.", None, 1.0) == sr.word_windows("Hi there.", 1.0)


def test_a_two_line_end_card_fits_the_frame():
    card = sr.word_image("Would you have chosen" + chr(10) + "Ilse?", y_fraction=0.5)
    assert card.size == sr.SIZE and card.getbbox() is not None and card.getbbox()[0] > 0 and card.getbbox()[2] < sr.SIZE[0]


def test_a_word_enters_with_a_glitch_that_dies_out_and_ends_perfectly_clean():
    import random
    clean = sr.word_image("RAIN")
    frames = sr.glitch_sequence(clean, "RAIN-0")
    assert len(frames) == len(sr.GLITCH_STRENGTHS) + 1 and frames[-1] is clean and all(f.size == sr.SIZE for f in frames)
    hard, light = frames[0], frames[len(sr.GLITCH_STRENGTHS) - 1]
    assert list(hard.getdata()) != list(clean.getdata())  # the first frame is hit
    wide = lambda im: (im.getbbox()[2] - im.getbbox()[0])
    assert wide(hard) > wide(clean) and wide(hard) >= wide(light)  # the colour split widens the word, less and less
    assert sr.glitch_frame(clean, 0, random.Random(1)) is clean


def test_push_in_starts_on_the_whole_frame_and_stays_inside_it():
    start, end = sr.kb_box(0, 4), sr.kb_box(4, 4)
    assert start == (0.0, 0.0, float(sr.SIZE[0]), float(sr.SIZE[1]))
    assert end[2] - end[0] < sr.SIZE[0] and end[0] >= 0 and end[2] <= sr.SIZE[0] and end[1] >= 0 and end[3] <= sr.SIZE[1]


def test_the_subtitle_is_smaller_and_lower_than_the_first_version_and_a_title_can_be_placed_apart():
    assert sr.SUB_PIXELS / sr.SIZE[1] < 118 / 1024 and sr.SUB_Y > 0.66  # relative to the frame: smaller than the 118 px of a 1024-tall frame, lower than 0.64
    word = sr.word_image("FLOOD")
    top = sr.word_image("TITLE", y_fraction=0.18, pixels=70)
    assert word.getbbox()[1] > sr.SIZE[1] * 0.64 and top.getbbox()[3] < sr.SIZE[1] * 0.30


def test_a_title_over_a_spoken_word_is_composed_in_one_picture_and_no_duration_is_negative(tmp_path):
    windows = [("RAIN", 0.1, 0.9), ("FALLS", 0.9, 1.4), ("POV: THE LAST CITY" + chr(10) + "IS DROWNING", 0.0, 1.2, {"y": 0.15, "pixels": 60})]
    listing = sr.subtitle_overlay(windows, 1.5, tmp_path)
    text = listing.read_text(encoding="utf-8")
    durations = [float(line.split()[1]) for line in text.splitlines() if line.startswith("duration")]
    assert durations and all(d > 0 for d in durations) and abs(sum(durations) - 1.5) < 0.5
    assert len(set(line for line in text.splitlines() if line.startswith("file"))) >= 6  # glitch frames of each text + the blank


def test_zoom_shake_and_flash_effects_stay_inside_the_frame_and_the_flash_fades():
    fx = {"zoom": 0.06, "shake": 1.0}
    for t in (0.0, 0.3, 1.0, 2.5):
        x0, y0, x1, y1 = sr.fx_box(t, 3.0, fx)
        assert 0 <= x0 < x1 <= sr.SOURCE[0] and 0 <= y0 < y1 <= sr.SOURCE[1]
    assert sr.fx_box(2.9, 3.0, {"zoom": 0.08})[2] - sr.fx_box(2.9, 3.0, {"zoom": 0.08})[0] < sr.fx_box(0.0, 3.0, {"zoom": 0.08})[2] - sr.fx_box(0.0, 3.0, {"zoom": 0.08})[0]
    white = Image.new("RGB", (8, 8), (0, 0, 0))
    levels = [sr.apply_flash(white, i).getpixel((0, 0))[0] for i in range(6)]
    assert levels == sorted(levels, reverse=True) and levels[0] > 150 and levels[-1] == 0


def test_sound_cues_follow_the_story_choice_twists_rewind_good_ending_and_impacts():
    plan = {"shots": [{"id": "s001", "kind": "narration", "fx": {}}, {"id": "s002", "kind": "choice"}, {"id": "s003", "kind": "narration", "fx": {"shake": 1.0}}, {"id": "s004", "kind": "twist", "ending": "bad"},
                      {"id": "s005", "kind": "rewind"}, {"id": "s006", "kind": "twist", "ending": "good"}]}
    timings = {sid: {"seconds": 4.0} for sid in ("s001", "s002", "s003", "s004", "s005", "s006")}
    cues = sr.sound_cues(plan, timings)
    assert cues["choice"] == [4.5, 7.65] and cues["twists"] == [12.0, 20.0] and cues["rewind"] == 16.0 and cues["uplift"] == 20.0 and cues["rain_stop"] == 20.0 and cues["thunder"] == [8.0]


def test_a_choice_shot_whose_character_list_is_empty_still_shows_the_two_main_characters():
    plan = {"characters": [{"id": "c1"}, {"id": "c2"}]}
    assert sr.choice_ids(plan, {"in_shot": []}) == ["c1", "c2"] and sr.choice_ids(plan, {"in_shot": ["c2", "c1"]}) == ["c2", "c1"]
    assert sr.choice_ids(plan, {"in_shot": ["c1"]}) == ["c1", "c2"] and sr.choice_ids(plan, {"in_shot": ["zz"]}) == ["c1", "c2"]


def test_the_default_shake_is_a_short_irregular_jolt_not_a_constant_regular_wobble():
    new = [sr.shake_offset(i / 30, 1.0, 0.7, "impact") for i in range(90)]
    old = [sr.shake_offset(i / 30, 1.0, 0.7, "legacy") for i in range(90)]
    amplitude = lambda series, a, b: max(abs(x) + abs(y) for x, y in series[a:b])
    assert amplitude(new, 0, 15) > 3 * amplitude(new, 60, 90)  # it dies out within about a second
    assert amplitude(new, 0, 15) < 0.5 * amplitude(old, 0, 15) and amplitude(new, 60, 90) < 0.1 * amplitude(old, 60, 90)  # and it is far smaller than the old one
    assert sr.shake_offset(1.0, 0.0, 0.3) == (0.0, 0.0)
    xs = [x for x, _ in new[:30]]
    crossings = sum(1 for a, b in zip(xs, xs[1:]) if a * b < 0)
    assert crossings >= 2  # it does move, with several frequencies
