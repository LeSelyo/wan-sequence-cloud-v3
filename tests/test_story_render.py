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
    assert sr.shot_seconds("talk", 2.0) == 2.15 and sr.shot_seconds("twist", 3.0) == 5.4
    assert sr.shot_seconds("choice", 1.2) == sr.CHOICE_MIN_SECONDS and sr.shot_seconds("choice", 5.0) == 5.0


def test_push_in_starts_on_the_whole_frame_and_stays_inside_it():
    start, end = sr.kb_box(0, 4), sr.kb_box(4, 4)
    assert start == (0.0, 0.0, 576.0, 1024.0)
    assert end[2] - end[0] < 576 and end[0] >= 0 and end[2] <= 576 and end[1] >= 0 and end[3] <= 1024


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


def test_the_subtitle_is_smaller_and_lower_than_before_and_a_title_can_be_placed_apart():
    assert sr.SUB_PIXELS < 100 and sr.SUB_Y > 0.66
    word = sr.word_image("FLOOD")
    top = sr.word_image("TITLE", y_fraction=0.18, pixels=70)
    assert word.getbbox()[1] > sr.SIZE[1] * 0.64 and top.getbbox()[3] < sr.SIZE[1] * 0.30


def test_overlay_list_has_glitch_frames_then_the_clean_word_and_never_a_negative_duration(tmp_path):
    listing = sr.subtitle_overlay([("RAIN", 0.1, 0.9), ("FALLS", 0.9, 0.95)], 1.5, tmp_path)
    text = listing.read_text(encoding="utf-8")
    assert text.count("w000_") == len(sr.GLITCH_STRENGTHS) + 1 and "w001_0" in text
    assert all(float(line.split()[1]) > 0 for line in text.splitlines() if line.startswith("duration"))
