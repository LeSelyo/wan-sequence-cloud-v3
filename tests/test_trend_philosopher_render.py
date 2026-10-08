"""2026-10-03 philosopher fixes: a real full-frame black beat (captions, voice and music keep
going), music beat grid, beat-snapped cuts, and a tiny end-to-end render on synthetic inputs."""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import trend_philosopher as tp  # noqa: E402


def _tool_available(name: str) -> bool:
    return bool(shutil.which(str(name)) or Path(str(name)).exists())


HAS_FFMPEG = _tool_available(tp.FFMPEG) and _tool_available(tp.FFPROBE)


def click_track(path: Path, *, bpm: float, offset: float, seconds: float, sr: int = 22050) -> Path:
    import numpy as np

    rng = np.random.default_rng(0)
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    burst = (rng.standard_normal(int(0.03 * sr)) * np.exp(-np.linspace(0, 8, int(0.03 * sr)))).astype(np.float32)
    t = offset
    while t < seconds - 0.05:
        i = int(t * sr)
        y[i:i + len(burst)] += burst
        t += 60.0 / bpm
    y /= max(1e-6, float(np.abs(y).max()))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((y * 30000).astype("<i2").tobytes())
    return path


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not available")
class BeatAnalysisTests(unittest.TestCase):
    def test_click_track_tempo_and_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            info = tp.analyze_beats(click_track(Path(temporary) / "clicks.wav", bpm=120, offset=0.2, seconds=30))
        self.assertAlmostEqual(info["bpm"], 120, delta=3)
        beats = info["beats"]
        self.assertAlmostEqual(beats[0] % 0.5, 0.2, delta=0.06)
        gaps = [b - a for a, b in zip(beats, beats[1:])]
        self.assertAlmostEqual(sum(gaps) / len(gaps), 0.5, delta=0.01)

    def test_a_slower_pulse_is_not_reported_as_double_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            info = tp.analyze_beats(click_track(Path(temporary) / "clicks.wav", bpm=75, offset=0.1, seconds=40))
        self.assertAlmostEqual(info["bpm"], 75, delta=3)

    def test_too_short_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                tp.analyze_beats(click_track(Path(temporary) / "tiny.wav", bpm=120, offset=0.0, seconds=0.2))


class ImpactSizeTests(unittest.TestCase):
    def test_sizes_follow_the_impact_of_the_word(self):
        small = tp.impact_scale("le")
        plain = tp.impact_scale("nuit")
        key = tp.impact_scale("lumiere")  # power word, 7 letters
        medium = tp.impact_scale("blanche")  # 7 letters, not a power word
        long_ = tp.impact_scale("transforme")
        power = tp.impact_scale("jamais")
        self.assertLess(small, plain)
        self.assertLess(plain, medium)
        self.assertLess(medium, long_)
        self.assertLess(long_, power)
        self.assertEqual(power, key)
        self.assertEqual(small, 0.88)
        self.assertEqual(plain, 1.0)

    def test_the_last_word_and_punchy_sentences_hit_harder_and_the_size_is_capped(self):
        self.assertAlmostEqual(tp.impact_scale("blanche", last_in_sentence=True), tp.impact_scale("blanche") + 0.1)
        self.assertAlmostEqual(tp.impact_scale("blanche", sentence_words=2), tp.impact_scale("blanche") + 0.1)
        self.assertEqual(tp.impact_scale("jamais", last_in_sentence=True, sentence_words=1), 1.65)
        self.assertEqual(tp.impact_scale("le", last_in_sentence=True, sentence_words=1), 0.88)  # small words never grow

    def test_accents_do_not_hide_a_power_word(self):
        self.assertEqual(tp.impact_scale("Vérité."), tp.impact_scale("verite"))
        self.assertEqual(tp.impact_scale("liberté,"), 1.5)

    def test_power_words_are_the_coloured_key_word(self):
        words = "la lumière revient lentement".split()
        self.assertEqual(words[tp.pick_key_word(words, [0, 1, 2, 3])], "lumière")

    def test_a_chunk_never_needs_more_width_than_the_frame(self):
        width, font = 1080, 106
        words = "Elle cherche toujours lentement une vérité immuable.".split()
        scales = [tp.impact_scale(w, last_in_sentence=i == len(words) - 1, sentence_words=len(words)) for i, w in enumerate(words)]
        chunks = tp.chunk_words(words, scales=scales, max_units=width * 0.92 / (font * tp.EM_PER_CAP))
        self.assertEqual([i for chunk in chunks for i in chunk], list(range(len(words))))
        for chunk in chunks:
            self.assertGreaterEqual(tp.fit_factor([words[i] for i in chunk], [scales[i] for i in chunk], font, width), 0.9, chunk)

    def test_without_a_budget_chunking_is_unchanged(self):
        words = "Le temps n'efface rien, il transforme lentement ce que nous croyons immuable.".split()
        self.assertEqual(tp.chunk_words(words), tp.chunk_words(words, scales=None, max_units=None))

    def test_the_black_beat_boosts_the_caption_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            normal, boosted = Path(temporary) / "a.ass", Path(temporary) / "b.ass"
            tp.build_captions(["Un chat noir dort."], [(0.0, 2.0)], normal, width=1080, height=1920)
            tp.build_captions(["Un chat noir dort."], [(0.0, 2.0)], boosted, width=1080, height=1920, force_visible=(0.5, 1.5))
            import re

            def sizes(path):
                return [int(v) for v in re.findall(r"\\t\(90,170,\\fscx(\d+)", path.read_text(encoding="utf-8"))]

            self.assertGreater(max(sizes(boosted)), max(sizes(normal)))


class FasterBackgroundsTests(unittest.TestCase):
    def test_the_default_cut_range_is_faster_than_the_previous_one(self):
        self.assertEqual(tp.BG_CUT_RANGE, tp.BG_CUT_RANGE_V3)
        self.assertLess(tp.BG_CUT_RANGE_V3[1], tp.BG_CUT_RANGE_V2[0] + 0.001)  # even the slowest cut beats the old fastest

    def test_beats_are_split_into_eighths_when_that_stays_above_a_quarter_second(self):
        beats = [0.27 + 0.5614 * k for k in range(40)]
        grid = tp.subdivide_beats(beats)
        gaps = [b - a for a, b in zip(grid, grid[1:])]
        self.assertAlmostEqual(sum(gaps) / len(gaps), 0.2807, delta=0.002)
        self.assertEqual(len(tp.subdivide_beats([0.0, 0.3, 0.6, 0.9])), 4)  # 0.3 s beats are not split
        self.assertEqual(tp.subdivide_beats([1.0]), [1.0])

    def test_every_background_is_used_and_none_repeats_back_to_back(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(tp, "_run"):
            images = [Path(temporary) / f"{n}.png" for n in "abcde"]
            tp.build_background_video(images, Path(temporary) / "bg.mp4", duration=30.0, width=270, height=480, rng_seed=5)
            files = [l.split("'")[1] for l in (Path(temporary) / "bg.txt").read_text(encoding="utf-8").splitlines() if l.startswith("file")]
        shown = [Path(f).name for f in files]
        self.assertEqual(set(shown), {"a.png", "b.png", "c.png", "d.png", "e.png"})
        self.assertTrue(all(x != y for x, y in zip(shown[:-1], shown[1:-1])))  # the trailing entry repeats the last image on purpose
        self.assertGreater(len(shown), 60)  # a cut every 0.25-0.5 s

    def test_relative_paths_are_written_absolute_for_the_concat_demuxer(self):
        import os

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(tp, "_run"):
            previous = os.getcwd()
            os.chdir(temporary)
            try:
                Path("work").mkdir()
                images = [Path("work") / "a.png", Path("work") / "b.png"]
                tp.build_background_video(images, Path("work") / "bg.mp4", duration=3.0, width=270, height=480)
                listed = [
                    line.split("'")[1]
                    for line in Path("work/bg.txt").read_text(encoding="utf-8").splitlines()
                    if line.startswith("file")
                ]
            finally:
                os.chdir(previous)
        self.assertTrue(listed)
        self.assertTrue(all(os.path.isabs(p) for p in listed), listed)

    def test_the_default_pool_of_extra_backgrounds_is_big_and_people_free(self):
        self.assertGreaterEqual(len(tp.DEFAULT_BACKGROUND_PROMPTS), 12)
        self.assertTrue(all("no people" in p for p in tp.DEFAULT_BACKGROUND_PROMPTS))


class AmixCompatibilityTests(unittest.TestCase):
    def test_modern_ffmpeg_uses_normalize_zero(self):
        with mock.patch.object(tp, "_filter_has_option", return_value=True):
            self.assertEqual(tp.amix_filter(2, duration="first"), "amix=inputs=2:duration=first:normalize=0")

    def test_old_ffmpeg_gets_the_same_level_from_a_gain(self):
        with mock.patch.object(tp, "_filter_has_option", return_value=False):
            self.assertEqual(
                tp.amix_filter(2, duration="first"), "amix=inputs=2:duration=first:dropout_transition=0,volume=2"
            )
            self.assertEqual(tp.amix_filter(3), "amix=inputs=3:dropout_transition=0,volume=3")

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not available")
    def test_the_real_ffmpeg_answers(self):
        tp._filter_has_option.cache_clear()
        self.assertIsInstance(tp._filter_has_option("amix", "normalize"), bool)


class HormoziCaptionTests(unittest.TestCase):
    WORDS = "Le temps n'efface rien, il transforme lentement ce que nous croyons immuable.".split()

    def test_chunks_are_short_and_end_at_punctuation(self):
        chunks = tp.chunk_words(self.WORDS)
        self.assertEqual([i for chunk in chunks for i in chunk], list(range(len(self.WORDS))))  # nothing lost
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 3)
            self.assertLessEqual(len(" ".join(self.WORDS[i] for i in chunk)), 17 + 4)
        comma = self.WORDS.index("rien,")
        self.assertTrue(any(chunk[-1] == comma for chunk in chunks))  # the comma closes a chunk

    def test_key_word_is_the_longest_meaningful_one_and_small_words_are_skipped(self):
        words = "il transforme lentement".split()
        self.assertEqual(words[tp.pick_key_word(words, [0, 1, 2])], "transforme")
        self.assertIsNone(tp.pick_key_word("et le pas".split(), [0, 1, 2]))
        stop = "rien toujours".split()  # 'rien' is a stop word even though it has 4 letters; 'toujours' is long
        self.assertEqual(stop[tp.pick_key_word(stop, [0, 1])], "toujours")

    def test_word_times_cover_the_sentence_and_give_pauses_to_punctuation(self):
        times = tp.estimate_word_times(["Le", "temps,", "passe."], 10.0, 13.0)
        self.assertAlmostEqual(times[0][0], 10.0)
        self.assertAlmostEqual(times[-1][1], 13.0)
        for (_, end), (start, _) in zip(times, times[1:]):
            self.assertAlmostEqual(end, start)
        self.assertGreater(times[2][1] - times[2][0], times[0][1] - times[0][0])

    def _events(self, sentences, windows, **kwargs):
        with tempfile.TemporaryDirectory() as temporary:
            ass = Path(temporary) / "c.ass"
            tp.build_captions(sentences, windows, ass, width=1080, height=1920, **kwargs)
            text = ass.read_text(encoding="utf-8")
        return text, [line for line in text.splitlines() if line.startswith("Dialogue")]

    def test_bold_uppercase_style_with_coloured_key_words(self):
        header, events = self._events(["Le temps transforme lentement toutes choses."], [(0.0, 4.0)])
        self.assertIn(tp.CAPTION_FONT, header)
        self.assertIn(",-1,", header.split("Style: Quote,")[1])  # bold
        joined = " ".join(events)
        self.assertIn("TEMPS", joined)
        self.assertNotIn("temps", joined)  # uppercase
        palette = {tp._ass_colour(hex_) for _, hex_, _ in tp.CAPTION_KEY_COLORS}
        used = {c for c in palette if f"\\c{c}" in joined}
        self.assertEqual(len(used), 1, "one key-word colour per sentence")
        import re

        finals = [int(v) for v in re.findall(r"\\t\(90,170,\\fscx(\d+)", joined)]
        self.assertGreater(max(finals), 110)  # impact words are bigger than the base size...
        self.assertLess(min(finals), 100)  # ...and small words recede
        for event in events:
            self.assertLessEqual(event.count(f"\\c{key_colour}"), 1) if (key_colour := next(iter(used))) else None

    def test_key_word_colours_change_across_sentences(self):
        sentences = [f"Le grand silence numero {i} traverse longtemps." for i in range(40)]
        windows = [(2.0 * i, 2.0 * i + 1.9) for i in range(40)]
        _, events = self._events(sentences, windows)
        palette = {tp._ass_colour(hex_): name for name, hex_, _ in tp.CAPTION_KEY_COLORS}
        seen = {name for colour, name in palette.items() if any(f"\\c{colour}" in e for e in events)}
        self.assertEqual(seen, {"yellow", "red", "blue"})
        yellow = sum(f"\\c{tp._ass_colour('FFD400')}" in e for e in events)
        self.assertGreater(yellow, len(events) * 0.3)  # mostly yellow, sometimes red or blue

    def test_captions_bridge_the_pause_between_sentences(self):
        _, events = self._events(["Un grand silence.", "Une longue nuit."], [(0.0, 2.0), (2.3, 4.0)])

        def seconds(text):
            h, m, rest = text.split(":")
            return int(h) * 3600 + int(m) * 60 + float(rest)

        ends = sorted(seconds(e.split(",")[2]) for e in events)
        starts = sorted(seconds(e.split(",")[1]) for e in events)
        self.assertAlmostEqual(max(e for e in ends if e < 3.0), 2.28, delta=0.02)  # held to just before the next sentence
        # a long pause is not bridged for ever
        _, long_pause = self._events(["Un grand silence.", "Une longue nuit."], [(0.0, 2.0), (6.0, 8.0)])
        first_sentence_end = max(seconds(e.split(",")[2]) for e in long_pause if seconds(e.split(",")[1]) < 3.0)
        self.assertAlmostEqual(first_sentence_end, 2.9, delta=0.02)
        del starts

    def test_colours_are_deterministic(self):
        first = self._events(["Le grand silence."], [(0.0, 2.0)])[1]
        second = self._events(["Le grand silence."], [(0.0, 2.0)])[1]
        self.assertEqual(first, second)

    def test_given_word_times_win_over_the_estimate(self):
        _, events = self._events(["Un grand silence."], [(0.0, 3.0)], word_times=[[(0.0, 0.2), (1.0, 1.5), (2.5, 3.0)]])
        # "silence." is a power word set at the maximum size: it gets a chunk of its own, which must
        # start at the time the voice gave for that word (2.5 s, minus the small caption lead), not at an estimate
        self.assertEqual(len(events), 2)
        self.assertIn("0:00:00.00", events[0])
        self.assertIn(tp._fmt_ts(2.5 - tp.CAPTION_LEAD), events[1])


class BackgroundPhaseTests(unittest.TestCase):
    """One still grand background, a short quick-cut burst, then the black screen (2026-10-03 review)."""

    def test_the_black_screen_targets_eight_seconds(self):
        self.assertEqual(tp.BLACK_SCREEN_SECONDS, 8.0)  # "the timing of the 8 seconds": the end then snaps to the nearest sentence end

    def test_burst_ends_where_the_black_screen_starts_and_is_capped_at_ten_seconds(self):
        phases = tp.plan_background_phases(108.0, (31.7, 38.7), flash_seconds=8.0)
        self.assertEqual([p[0] for p in phases], ["grand", "flash", "grand"])
        self.assertEqual(phases[0][1:], (0.0, 23.7))
        self.assertEqual(phases[1][1:], (23.7, 31.7))
        self.assertEqual(phases[2][1:], (31.7, 108.0))
        capped = tp.plan_background_phases(108.0, (40.0, 47.0), flash_seconds=60.0)
        self.assertAlmostEqual(capped[1][2] - capped[1][1], tp.BG_FLASH_MAX_SECONDS, places=3)

    def test_the_black_screen_keeps_its_planned_length_on_an_awkward_beat_grid(self):
        beats = [0.3 + 0.561 * k for k in range(300)]  # ~107 BPM: a beat every 0.56 s, so snapping alone could be +-0.28 s off
        for seed in range(30):
            start, end = tp.plan_black_screen(108.0, beats=beats, rng_seed=seed)
            self.assertAlmostEqual(end - start, tp.BLACK_SCREEN_SECONDS, delta=0.15)

    def test_burst_starts_on_a_beat_when_there_is_music(self):
        beats = [0.37 + 0.5 * k for k in range(240)]
        flash = tp.plan_background_phases(108.0, (31.87, 38.87), flash_seconds=8.0, beats=beats)[1]
        self.assertTrue(any(abs(flash[1] - b) < 1e-3 for b in beats))
        self.assertLessEqual(flash[2] - flash[1], tp.BG_FLASH_MAX_SECONDS + 1e-6)

    def test_without_a_black_screen_or_when_it_starts_at_zero_nothing_flashes(self):
        self.assertEqual(tp.plan_background_phases(50.0, None), [("grand", 0.0, 50.0)])
        self.assertEqual([p[0] for p in tp.plan_background_phases(50.0, (0.0, 7.0))], ["grand"])

    def test_every_flash_cut_stays_between_0_2_and_0_4_seconds(self):
        lo, hi = tp.BG_FLASH_CUT_RANGE
        for seed in range(25):
            for beats in (None, [0.1 + 0.25 * k for k in range(400)]):
                lengths = tp.flash_cut_lengths(10.0, 18.0, beat_times=beats, rng_seed=seed)
                self.assertAlmostEqual(sum(lengths), 8.0, places=3)
                self.assertTrue(all(lo - 1e-6 <= x <= hi + 1e-6 for x in lengths), lengths)

    def test_cut_point_is_the_earliest_sentence_end_after_the_time(self):
        windows = [(51.52, 53.92), (54.22, 55.58), (56.38, 59.74), (60.04, 65.0), (65.3, 67.14)]
        self.assertEqual(tp.cut_point(windows, 60.0), 65.0)  # 59.74 is just under the minute
        self.assertEqual(tp.cut_point(windows, 59.0), 59.74)
        self.assertIsNone(tp.cut_point(windows, 70.0))

    def test_blend_opacity_is_the_weight_of_the_sharp_picture_not_of_the_blur(self):
        """Measured with two flat colours: all_mode=normal:all_opacity=0.25 over white (A) / black (B) gives 25 % of A.
        The ramp must therefore pass 1 - weight, otherwise the blur starts nearly full and CLEARS toward the black."""
        import numpy as np

        def grey(opacity):
            out = subprocess.run(
                [tp.FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=white:s=16x16:d=0.1,format=gray", "-f", "lavfi", "-i",
                 "color=c=black:s=16x16:d=0.1,format=gray", "-filter_complex", f"[0][1]blend=all_mode=normal:all_opacity={opacity}",
                 "-frames:v", "1", "-f", "rawvideo", "-"], capture_output=True).stdout
            return float(np.frombuffer(out, np.uint8)[0])

        self.assertAlmostEqual(grey(0.25), 255 * 0.25, delta=2)  # 25 % of the first input
        phases = tp.plan_background_phases(108.0, (31.7, 38.7))
        graph = tp.blur_graph(phases, (31.7, 38.7), width=1080, style="mix")
        opac = [float(w) for w in re.findall(r"all_opacity=([0-9.]+)", graph)]
        self.assertGreaterEqual(min(opac), 0.3 - 1e-9)  # light: never more than 70 % blurred
        into_black = [float(w) for w, x in re.findall(r"all_opacity=([0-9.]+):enable='between\(t,([0-9.]+),", graph) if 31.3 < float(x) < 31.7]
        self.assertEqual(into_black, sorted(into_black, reverse=True))  # opacity of the sharp input FALLS as the black approaches
        self.assertLess(into_black[-1], into_black[0])

    def test_blur_is_light_and_only_exists_around_the_burst_and_the_black_screen(self):
        phases = tp.plan_background_phases(108.0, (31.7, 38.7))
        graph = tp.blur_graph(phases, (31.7, 38.7), width=1080, style="mix")
        self.assertIn("gblur=sigma=6.48", graph)
        # the blurred copy is only computed from the burst start to the end of the clearing ramp after the black
        enable = re.search(r"gblur=sigma=[0-9.]+:enable='between\(t,([0-9.]+),([0-9.]+)\)'", graph)
        self.assertAlmostEqual(float(enable.group(1)), 23.7, delta=0.06)
        self.assertAlmostEqual(float(enable.group(2)), 38.7 + tp.BLUR_RAMP_SECONDS, delta=0.06)
        # blend's all_opacity is the weight of the FIRST (sharp) input: the blurred weight is 1 - opacity (v7 had it inverted)
        steps = [(1 - float(w), float(x), float(y)) for w, x, y in re.findall(r"all_opacity=([0-9.]+):enable='between\(t,([0-9.]+),([0-9.]+)\)'", graph)]
        self.assertTrue(steps)
        self.assertLessEqual(max(w for w, _, _ in steps), 0.7 + 1e-9)  # light: never more than 70 % of the blurred copy
        # transitions only: after the burst-start pulse and before the ramp into the black the backgrounds stay sharp
        ramp_in = 31.7 - tp.BLUR_RAMP_SECONDS
        self.assertFalse(any(x < ramp_in - 1e-6 and y > 24.1 for _, x, y in steps), steps)
        self.assertTrue(graph.rstrip().endswith("[bl]"))
        half = 0.5 / 24  # windows are centred on frames, so an edge may sit half a frame off the black's edge
        self.assertTrue(all(not (31.7 + half <= x < 38.7 - half) for _, x, _ in steps))  # nothing is blended while the screen is black
        self.assertIsNone(tp.blur_graph(phases, (31.7, 38.7), width=1080, strength=0, style="mix"))
        self.assertIsNone(tp.blur_graph(tp.plan_background_phases(50.0, None), None, width=1080, style="mix"))

    def test_the_blur_around_the_black_screen_is_short_and_progressive(self):
        self.assertLessEqual(tp.BLUR_RAMP_SECONDS, 0.4)  # shorter than the first version's 0.6 s
        graph = tp.blur_graph(tp.plan_background_phases(108.0, (31.7, 38.7)), (31.7, 38.7), width=1080, style="mix")
        steps = [(1 - float(w), float(x)) for w, x in re.findall(r"all_opacity=([0-9.]+):enable='between\(t,([0-9.]+),", graph)]
        build_up = [w for w, x in steps if 31.7 - tp.BLUR_RAMP_SECONDS - 1e-6 <= x < 31.7]
        clearing = [w for w, x in steps if 38.7 - 1e-6 <= x < 38.7 + tp.BLUR_RAMP_SECONDS]
        self.assertGreaterEqual(len(build_up), 6)  # about one step per frame: a gradual build-up, not 2-3 big jumps
        self.assertGreaterEqual(len(clearing), 6)
        self.assertEqual(build_up, sorted(build_up))  # grows monotonically into the black...
        self.assertEqual(clearing, sorted(clearing, reverse=True))  # ...and clears monotonically after it
        self.assertLess(build_up[0], 0.1)
        self.assertLess(clearing[-1], 0.1)
        jumps = [abs(y - x) for x, y in zip(build_up, build_up[1:])]
        self.assertLess(max(jumps), 0.15)  # no abrupt step


class ProductionDefaultsTests(unittest.TestCase):
    """The base production route: the rules validated on 2026-10-03 are the DEFAULTS, no flag to remember."""

    def parse(self, *extra):
        return tp.build_parser().parse_args(["--script", "s.txt", "--out", "o.mp4", "--work-dir", "w", *extra])

    def test_cli_defaults_are_the_production_rules(self):
        args = self.parse()
        self.assertEqual(args.black_screen_seconds, 8.0)
        self.assertTrue(args.black_end_on_sentence)  # and it ends on a sentence end
        self.assertEqual((args.after_black_shake, args.exchange_backgrounds, args.exchange_directions), (None, None, None))  # exchanges are opt-in
        self.assertEqual((args.flash_seconds, tp.BG_FLASH_CUT_RANGE), (8.0, (0.2, 0.4)))
        self.assertLessEqual(args.flash_seconds, tp.BG_FLASH_MAX_SECONDS)
        self.assertEqual(args.cut_after, 60.0)
        self.assertIsNone(args.align_words)  # automatic: on whenever faster-whisper is installed
        self.assertEqual(args.blur, 1.0)
        self.assertFalse(args.keep_character_after_black)
        self.assertFalse(args.permanent_cuts)

    def test_builder_defaults_match_the_cli(self):
        import inspect

        defaults = {k: v.default for k, v in inspect.signature(tp.build_philosopher_trend).parameters.items()}
        self.assertEqual(defaults["black_screen_seconds"], 8.0)
        self.assertTrue(defaults["black_end_on_sentence"])
        self.assertEqual(defaults["flash_seconds"], 8.0)
        self.assertEqual(defaults["cut_after"], 60.0)
        self.assertIsNone(defaults["align_words"])
        self.assertEqual(defaults["blur"], 1.0)
        self.assertFalse(defaults["keep_character_after_black"])
        self.assertFalse(defaults["permanent_cuts"])

    def test_each_rule_can_still_be_turned_off(self):
        args = self.parse("--no-align-words", "--cut-after", "0", "--blur", "0", "--keep-character-after-black", "--permanent-cuts")
        self.assertIs(args.align_words, False)
        self.assertEqual((args.cut_after, args.blur), (0.0, 0.0))
        self.assertTrue(args.keep_character_after_black and args.permanent_cuts)
        self.assertIs(self.parse("--align-words").align_words, True)

    def test_grand_background_is_generated_by_default_then_reused(self):
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            args = self.parse()
            args.work_dir = work

            def fake(path, prompt, **kwargs):
                path.write_bytes(b"png")
                fake.calls.append((prompt, kwargs["width"], kwargs["height"]))
                return path

            fake.calls = []
            with mock.patch.object(tp, "generate_krea2_image", fake):
                first = tp.resolve_grand_background(args)
                second = tp.resolve_grand_background(args)  # cached in the work dir: no second generation
            self.assertEqual(first, work / "grand_background.png")
            self.assertEqual(second, first)
            self.assertEqual(fake.calls, [(tp.DEFAULT_GRAND_BACKGROUND_PROMPT, 832, 1472)])

    def test_grand_background_falls_back_quietly_when_the_api_is_down_and_respects_explicit_choices(self):
        import tempfile
        from unittest import mock

        import httpx

        with tempfile.TemporaryDirectory() as temporary:
            args = self.parse()
            args.work_dir = Path(temporary) / "work"
            with mock.patch.object(tp, "generate_krea2_image", side_effect=httpx.ConnectError("down")):
                self.assertIsNone(tp.resolve_grand_background(args))  # the builder then uses the first burst background
            explicit = self.parse("--grand-background", "mine.png")
            self.assertEqual(tp.resolve_grand_background(explicit), Path("mine.png"))
            legacy = self.parse("--permanent-cuts")
            self.assertIsNone(tp.resolve_grand_background(legacy))


class BlurStyleTests(unittest.TestCase):
    """Two blur styles compared in results/blur_lab: "mix" (light gaussian mix, hard cut to black) and "softfocus"
    (screen-blended glow + a fade to / from the black)."""

    PHASES = tp.plan_background_phases(65.0, (31.7, 38.7))

    def test_the_default_style_is_softfocus_and_unknown_styles_are_refused(self):
        self.assertEqual(tp.DEFAULT_BLUR_STYLE, "softfocus")  # chosen by the user on 2026-10-04
        self.assertEqual(set(tp.BLUR_STYLES), {"mix", "softfocus"})
        with self.assertRaises(ValueError):
            tp.blur_graph(self.PHASES, (31.7, 38.7), width=1080, style="sparkle")
        self.assertEqual(tp.build_parser().parse_args(["--script", "s", "--out", "o", "--work-dir", "w"]).blur_style, "softfocus")
        self.assertEqual(tp.build_parser().parse_args(["--script", "s", "--out", "o", "--work-dir", "w", "--blur-style", "mix"]).blur_style, "mix")

    def test_softfocus_is_a_screen_blended_glow_on_rgb_planes_around_the_black_only(self):
        graph = tp.blur_graph(self.PHASES, (31.7, 38.7), width=1080, style="softfocus")
        self.assertIn("all_mode=screen", graph)
        self.assertNotIn("all_mode=normal", graph)
        self.assertIn("format=gbrp", graph)  # a screen blend in YUV tints the picture purple
        self.assertTrue(graph.rstrip().endswith("[g]format=yuv420p[bl]"))
        self.assertIn("gblur=sigma=21.60", graph)  # 2 % of the 1080 px width
        steps = [(float(w), float(x)) for w, x in re.findall(r"all_opacity=([0-9.]+):enable='between\(t,([0-9.]+),", graph)]
        # screen opacity is the strength of the glow (not inverted): it grows into the black, clears after it, peak 0.9
        into = [w for w, x in steps if x < 31.7]
        after = [w for w, x in steps if x > 38.0]
        self.assertEqual(into, sorted(into))
        self.assertEqual(after, sorted(after, reverse=True))
        self.assertLessEqual(max(w for w, _ in steps), tp.SOFTFOCUS_PEAK + 1e-9)
        self.assertGreaterEqual(len(into), 8)  # about one step per frame over 0.45 s
        half = 0.5 / 24
        self.assertTrue(all(not (31.7 + half <= x < 38.7 - half) for _, x in steps))
        self.assertTrue(all(x > 31.7 - tp.SOFTFOCUS_RAMP_SECONDS - 0.1 for _, x in steps))  # nothing during the burst: transitions only

    def test_softfocus_comes_with_a_fade_to_and_from_the_black(self):
        fade = tp.fade_graph((31.7, 38.7))
        self.assertIn("fade=t=out:st=31.2500:d=0.4500:enable='lt(t,38.7000)'", fade)
        self.assertIn("fade=t=in:st=38.7000:d=0.4500:enable='gte(t,38.6995)'", fade)

    def test_the_default_graph_is_the_softfocus_one(self):
        self.assertEqual(tp.blur_graph(self.PHASES, (31.7, 38.7), width=1080), tp.blur_graph(self.PHASES, (31.7, 38.7), width=1080, style="softfocus"))


class OpeningTransitionsAndCharacterEntryTests(unittest.TestCase):
    """2026-10-04: burst transitions (shake / wipes / swing), the eyelid / oval opening, and the character's entry knobs."""

    def test_defaults_are_the_documented_ones(self):
        self.assertEqual((tp.DEFAULT_BURST_TRANSITION, tp.BURST_TRANSITION_SECONDS), ("cut", 0.10))
        self.assertEqual((tp.DEFAULT_INTRO_STYLE, tp.INTRO_SECONDS, tp.INTRO_HOLD_SECONDS), ("eyelid", 1.2, 0.15))
        self.assertEqual(tp.BURST_TRANSITIONS, ("cut", "shake", "wipe_left", "wipe_right", "swing"))
        self.assertEqual(tp.INTRO_STYLES, ("none", "eyelid", "oval"))
        import inspect

        sig = inspect.signature(tp.build_philosopher_trend).parameters
        self.assertEqual((sig["character_rise_seconds"].default, sig["character_rise_delay"].default,
                          sig["character_rise_from"].default, sig["character_final_y"].default), (2.0, 0.0, "bottom", tp.DEFAULT_CHARACTER_FINAL_Y))
        self.assertEqual(tp.DEFAULT_CHARACTER_FINAL_Y, 0.55)  # raised from 0.62: a bit more toward the centre of the screen
        args = tp.build_parser().parse_args(["--script", "s", "--out", "o", "--work-dir", "w"])
        self.assertEqual((args.burst_transition, args.intro_style, args.intro_seconds, args.character_from), ("cut", "eyelid", 1.2, "bottom"))
        args = tp.build_parser().parse_args(["--script", "s", "--out", "o", "--work-dir", "w", "--burst-transition", "swing",
                                             "--intro-style", "oval", "--character-from", "top", "--character-rise-delay", "0.8"])
        self.assertEqual((args.burst_transition, args.intro_style, args.character_from, args.character_rise_delay), ("swing", "oval", "top", 0.8))

    def test_unknown_styles_are_refused(self):
        with self.assertRaises(ValueError):
            tp.intro_graph("[b]", "[v]", "wink", width=100, height=100, fps=24)
        with self.assertRaises(ValueError):
            tp.build_phased_background(Path("g.png"), [Path("a.png")], [("grand", 0.0, 1.0)], Path("o.mp4"), width=10, height=10, transition="spin")
        with self.assertRaises(ValueError):
            tp.build_character_overlay(Path("a.mp4"), Path("c.png"), Path("o.mp4"), duration=1.0, rise_from="left")

    def test_none_means_no_opening(self):
        self.assertIsNone(tp.intro_graph("[b]", "[v]", "none", width=100, height=100, fps=24))
        self.assertIsNone(tp.intro_graph("[b]", "[v]", "eyelid", width=100, height=100, fps=24, seconds=0))

    def test_the_opening_graphs_draw_an_alpha_layer_that_ends_with_the_opening(self):
        for style in ("eyelid", "oval"):
            graph = tp.intro_graph("[b]", "[v]", style, width=1080, height=1920, fps=24, seconds=1.2, hold=0.15)
            self.assertIn("color=c=black:s=1080x1920:r=24:d=1.350", graph)  # hold + opening
            self.assertIn("geq=lum=0:cb=128:cr=128:a=", graph)
            self.assertTrue(graph.endswith("[b][lid]overlay=format=auto:eof_action=pass[v]") or "overlay=format=auto:eof_action=pass[v]" in graph)
        self.assertIn("pow(2*X/W-1,2)", tp.intro_graph("[b]", "[v]", "eyelid", width=10, height=10, fps=24))
        self.assertIn("hypot(", tp.intro_graph("[b]", "[v]", "oval", width=10, height=10, fps=24))

    def test_wipe_names_follow_the_style_and_swing_alternates(self):
        self.assertEqual(tp._wipe_names("wipe_left", 3), ["wipeleft"] * 3)
        self.assertEqual(tp._wipe_names("wipe_right", 2), ["wiperight"] * 2)
        self.assertEqual(tp._wipe_names("swing", 5), ["wipeleft", "wiperight", "wipeleft", "wiperight", "wipeleft"])

    def test_shake_crops_with_a_decaying_offset_on_every_cut_time(self):
        graph = tp.shake_filter(1080, 1920, impulses=[1.0, 1.3])
        self.assertIn("scale=1134:2016,crop=1080:1920", graph)  # enlarged by SHAKE_ZOOM so no edge shows
        self.assertEqual(graph.count("exp(-16*(t-"), 4)  # two cut times in x and in y
        self.assertIn("if(gte(t,1.000)", graph)
        self.assertIn("if(gte(t,1.300)", graph)

    def test_the_character_slides_from_the_chosen_edge_after_the_chosen_delay(self):
        from unittest import mock

        seen = {}
        with mock.patch.object(tp, "_run", side_effect=lambda cmd: seen.update(cmd=cmd)):
            tp.build_character_overlay(Path("bg.mp4"), Path("c.png"), Path("o.mp4"), duration=5.0, rise_duration=1.5, rise_delay=0.8, rise_from="top", dest_y_fraction=0.5)
            top = " ".join(seen["cmd"])
            tp.build_character_overlay(Path("bg.mp4"), Path("c.png"), Path("o.mp4"), duration=5.0)
            default = " ".join(seen["cmd"])
        self.assertIn("y='-h+((H*0.5-h/2)-(-h))*clip((t-0.800)/1.500,0,1)'", top)  # from the top, 0.8 s late, 1.5 s long, ends mid-screen
        self.assertIn(f"y='H+((H*{tp.DEFAULT_CHARACTER_FINAL_Y}-h/2)-(H))*clip((t-0.000)/2.000,0,1)'", default)  # default: from the bottom, 2 s, 55 %


class JumpCutAndConstantShakeTests(unittest.TestCase):
    """The jump cut as a callable (a time ratio per image + a total duration) and the constant camera shake presets."""

    def test_ratios_split_the_total_duration_in_proportion(self):
        self.assertEqual(tp.jumpcut_lengths(5.0, ratios=[1, 1, 2, 1]), [1.0, 1.0, 2.0, 1.0])
        lengths = tp.jumpcut_lengths(3.0, ratios=[3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertAlmostEqual(sum(lengths), 3.0)
        self.assertAlmostEqual(lengths[0] / lengths[1], 3.0)
        self.assertEqual(len(lengths), 15)  # the number of images is len(ratios)

    def test_without_ratios_it_is_the_automatic_beat_snapped_burst(self):
        auto = tp.jumpcut_lengths(8.0, rng_seed=3)
        self.assertAlmostEqual(sum(auto), 8.0, places=3)
        lo, hi = tp.BG_FLASH_CUT_RANGE
        self.assertTrue(all(lo - 1e-6 <= x <= hi + 1e-6 for x in auto))
        self.assertEqual(auto, tp.flash_cut_lengths(0.0, 8.0, rng_seed=3))

    def test_bad_ratios_are_refused(self):
        for bad in ([], [1, 0], [1, -2]):
            with self.assertRaises(ValueError):
                tp.jumpcut_lengths(5.0, ratios=bad)
        with self.assertRaises(ValueError):
            tp.jumpcut_lengths(0.0, ratios=[1])
        with self.assertRaises(ValueError):
            tp.jumpcut_lengths(1.0, ratios=[1000, 1])  # the second image would last 1 ms

    def test_every_constant_shake_preset_builds_a_filter_and_tilts_only_when_it_should(self):
        self.assertEqual(set(tp.CONSTANT_SHAKES), {"jitter", "roll", "handheld", "sway", "rock", "drift"})
        for name in tp.CONSTANT_SHAKES:
            graph = tp.shake_filter(1080, 1920, windows=[(3.0, 5.0)], preset=name)
            self.assertTrue(graph.startswith("pad=" if name in ("rock", "drift") else "scale=") and "crop=1080:1920" in graph, name)  # rock/drift pad (mirror), the others enlarge
            tilts = "rotate=" in graph or "perspective=" in graph  # rock tilts inside its sub-pixel perspective filter
            self.assertEqual(tilts, tp.SHAKE_PRESETS[name]["rot"] > 0, name)  # jitter never tilts, the others do
            self.assertTrue("clip((t-3.000)/0.12,0,1)" in graph or "clip(((in/24)-3.000)/0.12,0,1)" in graph, name)  # fades in at the start of the burst window (rock reads the frame count: in / fps)
        self.assertIsNone(tp.shake_filter(1080, 1920, windows=[(3.0, 5.0)]))  # no preset, no impulses: nothing
        self.assertIsNone(tp.shake_filter(1080, 1920, windows=[(3.0, 5.0)], preset="roll", amount=0))
        self.assertIsNone(tp.shake_filter(1080, 1920, windows=[], preset="roll"))
        with self.assertRaises(ValueError):
            tp.shake_filter(1080, 1920, windows=[(3.0, 5.0)], preset="earthquake")

    def test_rock_is_a_pure_slow_rocking_that_does_not_enlarge_the_picture(self):
        import re

        cfg = tp.SHAKE_PRESETS["rock"]
        self.assertEqual((cfg["f"], cfg["fr"]), ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))  # one 1 Hz wave, no fast tremor
        self.assertTrue(cfg["subpixel"])
        graph = tp.shake_filter(1080, 1920, windows=[(3.0, 5.0)], preset="rock")
        self.assertNotIn("scale=", graph)  # the framing is the original one
        self.assertTrue(graph.startswith("pad="))
        self.assertIn("fillborders=", graph)
        self.assertIn("mode=mirror", graph)  # the uncovered edges are the picture's own border, mirrored
        self.assertIn("perspective=", graph)  # moved and tilted by an INTERPOLATING filter: no whole-pixel stepping
        self.assertIn("eval=frame", graph)
        self.assertIn("interpolation=cubic", graph)
        self.assertNotIn("rotate=", graph)
        self.assertNotIn("x='(in_w-", graph)  # no crop whose offset snaps to whole pixels: the final crop is exactly centred
        self.assertTrue(re.search(r"crop=1080:1920:x=(\d+):y=\1$", graph), graph[-60:])
        self.assertEqual(set(re.findall(r"2\*PI\*([0-9.]+)\*", graph)), {"1.0"})  # one wave of 1 Hz, nothing faster

    def test_drift_is_as_slow_as_rock_but_its_direction_and_angle_never_settle(self):
        import re

        rock, drift = tp.SHAKE_PRESETS["rock"], tp.SHAKE_PRESETS["drift"]
        self.assertEqual(drift["amp"], rock["amp"])  # the same reach as the validated slow rocking
        self.assertTrue(drift["subpixel"] and drift["edges"] == "mirror")  # same smooth, framing-preserving machinery
        slow = lambda cfg: max(max(cfg["f"]), max(cfg.get("fy", cfg["f"])), max(cfg["fr"]))  # noqa: E731
        self.assertLessEqual(slow(drift), slow(rock))  # nothing faster than rock's 1 Hz
        self.assertEqual(sum(1 for f in drift["f"] if f), 3)  # three unrelated waves per axis, the tilt too: it never repeats along one line
        self.assertEqual(sum(1 for f in drift["fr"] if f), 3)
        self.assertNotEqual(drift["f"], drift["fy"])  # x and y wander on different rhythms, so the direction keeps turning
        graph = tp.shake_filter(1080, 1920, windows=[(0.0, 60.0)], preset="drift")
        self.assertNotIn("scale=", graph)
        self.assertIn("perspective=", graph)
        freqs = {float(f) for f in re.findall(r"2\*PI\*([0-9.]+)\*", graph)}
        self.assertEqual(freqs, {0.9, 0.55, 0.3, 0.7, 0.43, 0.26, 0.8, 0.45, 0.27})
        self.assertLessEqual(max(freqs), 1.0)

    def test_drift_moves_smoothly_and_keeps_the_picture_whole(self):
        import tempfile
        import numpy as np
        from PIL import Image, ImageDraw

        w, h, fps = 180, 320, 24
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            image = tmp / "bar.png"
            canvas = Image.new("L", (w, h), 20)
            ImageDraw.Draw(canvas).rectangle((w // 2 - 6, 0, w // 2 + 6, h), fill=235)
            canvas.convert("RGB").save(image)
            out = tmp / "drift.mp4"
            tp.build_phased_background(image, [image], [("grand", 0.0, 1.0), ("flash", 1.0, 9.0), ("grand", 9.0, 10.0)], out, width=w, height=h,
                                       fps=fps, constant_shake="drift", constant_shake_amount=3.0)
            raw = subprocess.run([tp.FFMPEG, "-v", "error", "-i", str(out), "-vf", "format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
            frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w)[int(1.4 * fps):int(8.6 * fps)]  # 7 s of drift
            xs = np.arange(w, dtype=np.float64)
            path = []
            for f in frames:
                weights = np.clip(f[h // 2 + 60].astype(np.float64) - 100, 0, None)
                path.append(float((weights * xs).sum() / weights.sum()))
            path = np.array(path)
            t = np.arange(len(path)) / fps
            cols = [np.ones_like(t)]
            for k in np.arange(0.1, 2.0 + 1e-9, 0.1):  # a smooth path of <= 2 Hz (finer grid: the drift is not periodic)
                cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
            coef, *_ = np.linalg.lstsq(np.stack(cols, axis=1), path, rcond=None)
            residual = float(np.sqrt(np.mean((path - np.stack(cols, axis=1) @ coef) ** 2)))
            self.assertGreater(float(path.max() - path.min()), 4.0)  # it really moves
            self.assertLess(residual, 0.15)  # and without whole-pixel stepping
            # the direction keeps changing: the horizontal velocity changes sign many times in 7 s (a 1 Hz rocking would give ~14)
            velocity = np.diff(path)
            self.assertGreaterEqual(int((np.diff(np.sign(velocity[np.abs(velocity) > 0.02])) != 0).sum()), 6)

    def test_rock_moves_smoothly_without_micro_jerks(self):
        import tempfile
        import numpy as np
        from PIL import Image, ImageDraw

        w, h, fps = 180, 320, 24
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            image = tmp / "bar.png"
            canvas = Image.new("L", (w, h), 20)
            ImageDraw.Draw(canvas).rectangle((w // 2 - 6, 0, w // 2 + 6, h), fill=235)  # a vertical bar: its x is the horizontal motion
            canvas.convert("RGB").save(image)
            out = tmp / "rock.mp4"
            tp.build_phased_background(image, [image], [("grand", 0.0, 1.0), ("flash", 1.0, 4.0), ("grand", 4.0, 5.0)], out, width=w, height=h,
                                       fps=fps, constant_shake="rock", constant_shake_amount=3.0)
            raw = subprocess.run([tp.FFMPEG, "-v", "error", "-i", str(out), "-vf", "format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
            frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w)[int(1.4 * fps):int(3.6 * fps)]
            xs = np.arange(w, dtype=np.float64)
            path = []
            for f in frames:
                weights = np.clip(f[h // 2 + 60].astype(np.float64) - 100, 0, None)
                path.append(float((weights * xs).sum() / weights.sum()))
            path = np.array(path)
            t = np.arange(len(path)) / fps
            cols = [np.ones_like(t)]
            for k in np.arange(0.25, 2.0 + 1e-9, 0.25):  # everything a smooth path of <= 2 Hz can do
                cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
            basis = np.stack(cols, axis=1)
            coef, *_ = np.linalg.lstsq(basis, path, rcond=None)
            residual = float(np.sqrt(np.mean((path - basis @ coef) ** 2)))
            self.assertGreater(float(path.max() - path.min()), 4.0)  # it really rocks (the amplitude is ~6 px)
            self.assertLess(residual, 0.15)  # whole-pixel stepping (crop) leaves ~0.5 px of residual: the micro-jerks

    def test_rock_keeps_the_size_and_never_shows_a_black_corner(self):
        import tempfile
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            image = tmp / "white.png"
            Image.new("RGB", (180, 320), (250, 250, 250)).save(image)
            out = tmp / "rock.mp4"
            tp.build_phased_background(image, [image], [("grand", 0.0, 0.5), ("flash", 0.5, 3.5), ("grand", 3.5, 4.0)], out, width=180, height=320,
                                       fps=24, constant_shake="rock", constant_shake_amount=3.0)
            raw = subprocess.run([tp.FFMPEG, "-v", "error", "-i", str(out), "-vf", "format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
            frames = np.frombuffer(raw, np.uint8).reshape(-1, 320, 180)  # 180x320: exactly the requested size
            self.assertGreater(int(frames.min()), 200)  # a white picture stays white everywhere, even at the corners of every tilted frame

    def test_the_amount_scales_the_shake_and_the_margin_hides_every_edge(self):
        small = tp.shake_filter(1080, 1920, windows=[(0.0, 1.0)], preset="handheld", amount=0.5)
        big = tp.shake_filter(1080, 1920, windows=[(0.0, 1.0)], preset="handheld", amount=2.0)
        enlarged = lambda graph: int(graph.split("scale=")[1].split(":")[0])  # noqa: E731
        self.assertLess(enlarged(small), enlarged(big))
        self.assertGreater(enlarged(small), 1080)  # always enlarged: the cropped frame never shows an edge

    def test_build_jumpcut_is_one_call_with_a_ratio_per_image_and_a_total_duration(self):
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            images = []
            for index, colour in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                path = tmp / f"i{index}.png"
                Image.new("RGB", (90, 160), colour).save(path)
                images.append(path)
            out = tmp / "jump.mp4"
            tp.build_jumpcut(images, out, total_seconds=4.0, ratios=[1, 2, 1], width=90, height=160, fps=24)
            self.assertAlmostEqual(tp._audio_duration(out) if False else float(subprocess.run(
                [tp.FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout), 4.0, delta=0.1)
            # image 0 red for 1 s, image 1 green for 2 s, image 2 blue for 1 s, in the order given
            def centre(t):
                png = tmp / f"f{t}.png"
                subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", str(png)], check=True)
                return Image.open(png).convert("RGB").getpixel((45, 80))

            red, green, blue = centre(0.5), centre(2.0), centre(3.5)
            self.assertTrue(red[0] > 200 and red[1] < 60, red)
            self.assertTrue(green[1] > 200 and green[0] < 60, green)
            self.assertTrue(blue[2] > 200 and blue[0] < 60, blue)
            self.assertTrue(centre(1.2)[1] > 200)  # still green at 1.2 s: image 1 starts at 1 s
            self.assertTrue(centre(2.9)[1] > 200)  # and lasts until 3 s

    def test_the_constant_shake_moves_every_frame_of_the_burst_and_nothing_outside_it(self):
        import tempfile
        import numpy as np
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            image = tmp / "detail.png"  # a picture with structure: a still frame would be identical, a shaken one would not
            canvas = Image.new("RGB", (180, 320), (30, 30, 30))
            draw = ImageDraw.Draw(canvas)
            for k in range(12):
                draw.rectangle((10 + k * 13, 30 + (k % 4) * 60, 18 + k * 13, 80 + (k % 4) * 60), fill=(240, 240 - k * 15, 60 + k * 10))
            canvas.save(image)
            out = tmp / "shaken.mp4"
            tp.build_phased_background(image, [image], [("grand", 0.0, 1.0), ("flash", 1.0, 3.0), ("grand", 3.0, 4.0)], out,
                                       width=180, height=320, fps=24, constant_shake="jitter", constant_shake_amount=5.0)  # 0.9 px x 5: the crop offset is whole pixels
            raw = subprocess.run([tp.FFMPEG, "-v", "error", "-i", str(out), "-vf", "format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
            frames = np.frombuffer(raw, np.uint8).reshape(-1, 320, 180).astype(np.float32)
            motion = [float(np.abs(frames[i + 1] - frames[i]).mean()) for i in range(len(frames) - 1)]
            inside = motion[int(1.3 * 24):int(2.7 * 24)]   # well inside the burst (past the fade-in)
            outside = motion[2:int(0.8 * 24)] + motion[int(3.3 * 24):-2]
            moving = [m > 0.05 for m in inside]
            self.assertGreaterEqual(sum(moving) / len(moving), 0.8)  # constant: almost every frame pair differs...
            longest, run = 0, 0
            for flag in moving:
                run = 0 if flag else run + 1
                longest = max(longest, run)
            self.assertLessEqual(longest, 2)  # ...and it never freezes for long
            self.assertLess(max(outside), 0.2)  # still picture before and after the burst (encoder noise only, ~0.07 on the first frames)
            self.assertGreater(float(np.median(inside)), 1.0)  # while the burst really moves, 10x more than any noise


class BlackScreenWipeTests(unittest.TestCase):
    """image -> black -> image as a shutter wipe with a speed curve; the other side of the shutter is another image, dimmed."""

    BLACK = (1.5, 2.1)

    def test_defaults_and_refusals(self):
        self.assertEqual((tp.DEFAULT_BLACK_TRANSITION, tp.BLACK_WIPE_SECONDS, tp.TRANSITION_SHAKE_SECONDS), ("blur", 0.8, 1.2))
        self.assertEqual((tp.DEFAULT_BLACK_WIPE_EASING, tp.BLACK_WIPE_PANEL_VISIBILITY, tp.BLACK_WIPE_PANEL_FADE_SECONDS), ("exponential", 0.5, 0.3))
        self.assertEqual(tp.BLACK_WIPE_EASINGS, ("linear", "smooth", "exponential", "logarithmic"))
        args = tp.build_parser().parse_args(["--script", "s", "--out", "o", "--work-dir", "w"])
        self.assertEqual((args.black_transition, args.black_wipe_seconds, args.black_wipe_easing, args.black_wipe_panel_visibility, args.transition_shake),
                         ("blur", 0.8, "exponential", 0.5, None))
        with self.assertRaises(ValueError):
            tp.black_wipes_graph("[a]", "[b]", self.BLACK, "spin", width=90, height=160, fps=24)
        with self.assertRaises(ValueError):
            tp.black_wipes_graph("[a]", "[b]", self.BLACK, "swing", width=90, height=160, fps=24, easing="bouncy")
        self.assertIsNone(tp.black_wipes_graph("[a]", "[b]", self.BLACK, "blur", width=90, height=160, fps=24))
        self.assertIsNone(tp.black_wipes_graph("[a]", "[b]", None, "swing", width=90, height=160, fps=24))

    def test_the_speed_curves(self):
        for name in tp.BLACK_WIPE_EASINGS:
            values = [tp.ease_value(name, None, u / 20) for u in range(21)]
            self.assertEqual((values[0], values[-1]), (0.0, 1.0), name)  # the edge starts and ends where it must
            self.assertEqual(values, sorted(values), name)  # and never goes back
        mid = {name: tp.ease_value(name, None, 0.5) for name in tp.BLACK_WIPE_EASINGS}
        self.assertAlmostEqual(mid["linear"], 0.5)
        self.assertAlmostEqual(mid["smooth"], 0.5)
        self.assertLess(mid["exponential"], 0.2)  # slow start: far less than half of the way at half the time
        self.assertGreater(mid["logarithmic"], 0.7)  # fast start: most of the way at half the time
        # exponential ACCELERATES (the distance covered per step keeps growing), logarithmic BRAKES
        steps = lambda name: [tp.ease_value(name, None, (i + 1) / 10) - tp.ease_value(name, None, i / 10) for i in range(10)]  # noqa: E731
        self.assertEqual(steps("exponential"), sorted(steps("exponential")))
        self.assertEqual(steps("logarithmic"), sorted(steps("logarithmic"), reverse=True))
        # a stronger exponential starts even slower
        self.assertLess(tp.ease_value("exponential", 8.0, 0.5), tp.ease_value("exponential", 2.0, 0.5))

    def test_the_expression_matches_the_python_curve(self):
        import math

        for name in ("exponential", "logarithmic", "smooth"):
            expr = tp.wipe_progress_expr(name, None, "0.4")
            value = eval(expr.replace("exp(", "math.exp(").replace("log(", "math.log("))  # noqa: S307  (a plain arithmetic expression of our own)
            self.assertAlmostEqual(value, tp.ease_value(name, None, 0.4), places=9, msg=name)

    def test_graph_slides_a_shutter_over_the_picture_with_the_curve_in_it(self):
        graph = tp.black_wipes_graph("[a]", "[b]", self.BLACK, "swing", width=90, height=160, fps=24, seconds=0.5, easing="exponential")
        self.assertIn("overlay=x='main_w*(1-((exp(4.0*clip((t-1.0000)/0.5000,0,1))-1)/(exp(4.0)-1)))'", graph)  # closing: in from the right
        self.assertIn("enable='between(t,1.0000,1.5000)'", graph)  # only while the wipe lasts
        self.assertIn("overlay=x='main_w*((exp(4.0*clip((t-2.1000)/0.5000,0,1))-1)/(exp(4.0)-1))'", graph)  # opening: leaves toward the right
        self.assertIn("enable='between(t,2.1000,2.6000)'", graph)
        self.assertIn("color=c=black", graph)  # no panel image: a plain black shutter
        self.assertNotIn("xfade", graph)
        log = tp.black_wipes_graph("[a]", "[b]", self.BLACK, "wipe_left", width=90, height=160, fps=24, seconds=0.5, easing="logarithmic")
        self.assertIn("log(1+9.0*", log)
        self.assertIn("-main_w*(", log.split("[wc];")[1])  # wipe_left opens toward the left
        right = tp.black_wipes_graph("[a]", "[b]", self.BLACK, "wipe_right", width=90, height=160, fps=24, seconds=0.5, easing="linear")
        self.assertIn("-main_w*(1-(", right)  # wipe_right closes from the left

    def test_the_other_side_of_the_shutter_can_be_another_dimmed_image(self):
        graph = tp.black_wipes_graph("[a]", "[b]", self.BLACK, "swing", width=90, height=160, fps=24, seconds=0.5, panel="[3:v]", panel_visibility=0.35)
        self.assertTrue(graph.startswith("[3:v]scale=90:160"))
        self.assertIn("lutrgb=r='val*0.35':g='val*0.35':b='val*0.35'", graph)  # the image, dimmed
        # the closing wipe ends 0.3 s before the black screen; the image fades to black in that gap, and fades up from black before the opening wipe
        self.assertIn("fade=t=out:st=1.2000:d=0.3000", graph)
        self.assertIn("fade=t=in:st=2.1000:d=0.3000", graph)
        self.assertIn("clip((t-0.7000)/0.5000,0,1)", graph)  # closing wipe: 0.7 -> 1.2 s
        self.assertIn("clip((t-2.4000)/0.5000,0,1)", graph)  # opening wipe: 2.4 -> 2.9 s
        self.assertIn("enable='between(t,0.7000,1.5000)'", graph)  # the shutter layer stays (full cover) until the black screen
        self.assertIn("enable='between(t,2.1000,2.9000)'", graph)
        self.assertNotIn("color=c=black", graph)

    def test_a_wipe_that_passes_once_keeps_the_other_image_and_never_returns(self):
        graph = tp.black_wipes_graph("[a]", "[b]", (1.5, 9.0), "swing", width=90, height=160, fps=24, seconds=0.5, panel="[1:v]", once=True)
        self.assertEqual(graph.count("overlay="), 1)  # one shutter, no return wipe
        self.assertIn("enable='gte(t,1.0000)'", graph)  # it stays after the wipe
        self.assertNotIn("fade=", graph)  # and does not fade to black
        self.assertNotIn("split", graph)
        plain = tp.black_wipes_graph("[a]", "[b]", (1.5, 9.0), "swing", width=90, height=160, fps=24, seconds=0.5, once=True)
        self.assertIn("color=c=black", plain)

    def test_the_other_image_gets_the_same_camera_shake_as_the_first(self):
        shake = tp.shake_filter(90, 160, windows=[(0.2, 3.0)], preset="rock", fps=24)
        graph = tp.black_wipes_graph("[a]", "[b]", (1.5, 9.0), "swing", width=90, height=160, fps=24, seconds=0.5, panel="[1:v]", once=True, panel_filters=shake)
        self.assertIn(shake, graph)  # the very same filter string: identical movement, so it carries on across the two images
        self.assertIn(f"format=yuv420p,{shake}[sc2]", graph)
        both = tp.black_wipes_graph("[a]", "[b]", self.BLACK, "swing", width=90, height=160, fps=24, seconds=0.5, panel="[1:v]", panel_filters=shake)
        self.assertIn(f"format=yuv420p,{shake},split[sc][so]", both)

    def test_the_once_wipe_really_leaves_the_other_image_on_screen(self):
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            Image.new("RGB", (90, 160), (250, 250, 250)).save(tmp / "white.png")
            Image.new("RGB", (90, 160), (220, 30, 30)).save(tmp / "panel.png")
            graph = "[0:v]scale=90:160,fps=24,format=yuv420p[p];" + tp.black_wipes_graph(
                "[p]", "[v]", (1.5, 9.0), "wipe_left", width=90, height=160, fps=24, seconds=0.5, easing="linear", panel="[1:v]", panel_visibility=0.8, once=True)
            out = tmp / "once.mp4"
            subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-loop", "1", "-framerate", "24", "-t", "4", "-i", str(tmp / "white.png"),
                            "-loop", "1", "-framerate", "24", "-i", str(tmp / "panel.png"), "-filter_complex", graph, "-map", "[v]", "-t", "4", "-pix_fmt", "yuv420p", str(out)], check=True)

            def pixel(t, x):
                png = tmp / f"o_{t}_{x}.png"
                subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", str(png)], check=True)
                return Image.open(png).convert("RGB").getpixel((x, 80))

            self.assertGreater(sum(pixel(0.5, 45)), 600)  # the white picture before the wipe
            red, green, blue = pixel(3.5, 45)  # long after: the other image, dimmed, still there (not black, not white)
            self.assertTrue(red > 100 and green < 80, (red, green, blue))
            self.assertTrue(pixel(3.9, 8)[0] > 100)  # and it is still the same on both sides: no return of the first picture

    def _render(self, tmp, style, easing, panel_colour=None, seconds=0.5):
        """A 3.2 s clip: a flat white picture (never changes) wiped to black and back; returns a function t -> (left pixel, right pixel)."""
        from PIL import Image

        image = tmp / "white.png"
        Image.new("RGB", (90, 160), (250, 250, 250)).save(image)
        extra, panel = [], None
        if panel_colour:
            Image.new("RGB", (90, 160), panel_colour).save(tmp / "panel.png")
            extra, panel = ["-loop", "1", "-framerate", "24", "-i", str(tmp / "panel.png")], "[1:v]"
        graph = "[0:v]scale=90:160,fps=24,format=yuv420p[p];" + tp.black_wipes_graph(
            "[p]", "[w]", self.BLACK, style, width=90, height=160, fps=24, seconds=seconds, easing=easing, panel=panel, panel_visibility=0.8)
        # the caller draws the black between b0 and b1, as the pipeline does
        graph += f";[w]drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,{self.BLACK[0]},{self.BLACK[1]})'[v]"
        out = tmp / f"{style}_{easing}_{bool(panel_colour)}.mp4"
        subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-loop", "1", "-framerate", "24", "-t", "3.2", "-i", str(image), *extra, "-filter_complex", graph,
                        "-map", "[v]", "-t", "3.2", "-pix_fmt", "yuv420p", str(out)], check=True)

        def pixels(t, y=80):
            png = tmp / f"{out.stem}_{t}.png"
            subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", str(png)], check=True)
            return Image.open(png).convert("RGB")

        return pixels

    @staticmethod
    def _covered(image):
        """Fraction of the row that is no longer the white picture (the shutter's reach)."""
        gray = image.convert("L")
        return sum(gray.getpixel((x, 80)) < 200 for x in range(gray.width)) / gray.width

    def test_the_speed_curve_is_what_the_viewer_sees(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            at = {}  # coverage of the screen by the shutter, half-way through the closing wipe (b0 - 0.5 s ... b0 = 1.0 s ... 1.5 s)
            for easing in ("linear", "exponential", "logarithmic"):
                pixels = self._render(tmp, "wipe_left", easing, seconds=0.5)
                at[easing] = self._covered(pixels(1.25))
                self.assertEqual(self._covered(pixels(0.5)), 0.0, easing)  # before the wipe: the untouched picture
            self.assertAlmostEqual(at["linear"], 0.5, delta=0.15)
            self.assertLess(at["exponential"], 0.3)  # slow start: the shutter has barely entered
            self.assertGreater(at["logarithmic"], 0.65)  # fast start: it is already most of the way

    def test_directions_the_picture_is_untouched_and_it_comes_back(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            for style, black_side_closing, black_side_opening in (("wipe_left", "right", "left"), ("wipe_right", "left", "right"), ("swing", "right", "right")):
                pixels = self._render(tmp, style, "linear", seconds=0.5)
                left = lambda im: im.convert("L").getpixel((8, 80))  # noqa: E731
                right = lambda im: im.convert("L").getpixel((82, 80))  # noqa: E731
                closing = pixels(1.25)  # half-way through the closing wipe
                side = (right if black_side_closing == "right" else left)(closing)
                other = (left if black_side_closing == "right" else right)(closing)
                self.assertLess(side, 40, (style, "closing"))  # the shutter arrives from this side...
                self.assertGreater(other, 200, (style, "closing"))  # ...the picture is still on the other
                self.assertLess(left(pixels(1.8)) + right(pixels(1.8)), 40)  # fully black
                opening = pixels(2.35)  # half-way through the opening wipe: the shutter retreats toward black_side_opening
                side = (right if black_side_opening == "right" else left)(opening)
                other = (left if black_side_opening == "right" else right)(opening)
                self.assertLess(side, 40, (style, "opening"))  # the shutter retreats toward this side: it is still black there
                self.assertGreater(other, 200, (style, "opening"))  # the picture is already back on the other
                self.assertGreater(left(pixels(3.0)) + right(pixels(3.0)), 400)  # the same picture is back: the background never changed

    def test_the_other_side_of_the_shutter_shows_a_dimmed_image_not_flat_black(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            pixels = self._render(tmp, "wipe_left", "linear", panel_colour=(220, 30, 30), seconds=0.5)
            mid = pixels(0.95)  # half-way through the closing wipe (0.7 -> 1.2 s): the shutter shows the DIMMED red image, not black
            red, green, _ = mid.getpixel((82, 80))
            self.assertGreater(red, 40)
            self.assertGreater(red, green * 2)  # it is the red image
            self.assertLess(red, 200)  # but dimmed
            done = pixels(1.2)  # the wipe is over: the red image still covers the screen at full dimmed brightness...
            self.assertGreater(done.getpixel((8, 80))[0], 40)
            end = pixels(1.52)  # ...and has faded to black when the black screen starts (1.5 s)
            self.assertLess(sum(end.getpixel((82, 80))), 60)


class WordAlignmentTests(unittest.TestCase):
    SENTENCES = ["Le temps passe vite.", "Que reste alors ?"]
    WINDOWS = [(0.0, 2.0), (2.5, 5.0)]

    def test_times_follow_the_speech_model_and_ignore_accents_case_and_punctuation(self):
        spoken = [("le", 0.0, 0.2), ("Temps", 0.3, 0.7), ("passe", 0.8, 1.2), ("vite", 1.3, 1.9),
                  ("Que", 2.6, 2.8), ("reste", 2.85, 3.3), ("alors", 4.0, 4.6)]  # the lone "?" is not a spoken word
        times = tp.align_word_times(None, self.SENTENCES, self.WINDOWS, spoken=spoken)
        self.assertEqual(len(times), 2)
        self.assertEqual([len(t) for t in times], [4, 4])
        self.assertAlmostEqual(times[0][1][0], 0.3, places=2)  # "temps"
        self.assertAlmostEqual(times[1][2][0], 4.0, places=2)  # "alors"
        flat = [x for t in times for x in t]
        self.assertTrue(all(b > a for a, b in flat))

    def test_missed_words_are_interpolated_between_their_neighbours(self):
        spoken = [("le", 0.0, 0.2), ("vite", 1.3, 1.9)]  # "temps passe" was not heard
        first = tp.align_word_times(None, ["Le temps passe vite."], [(0.0, 2.0)], spoken=spoken)[0]
        self.assertEqual(len(first), 4)
        self.assertGreaterEqual(first[1][0], 0.2 - 1e-6)
        self.assertLessEqual(first[2][1], 1.3 + 1e-6)
        self.assertLess(first[1][0], first[2][0])

    def test_a_badly_matched_sentence_falls_back_to_the_estimate(self):
        spoken = [("completely", 0.0, 0.5), ("different", 0.5, 1.0), ("words", 1.0, 1.5), ("here", 1.5, 2.0)]
        self.assertEqual(tp.align_word_times(None, ["Le temps passe vite."], [(0.0, 2.0)], spoken=spoken), [[]])

    def test_without_faster_whisper_the_function_degrades_to_nothing(self):
        import sys
        from unittest import mock

        with mock.patch.dict(sys.modules, {"faster_whisper": None}):
            self.assertEqual(tp.align_word_times(Path("x.wav"), ["Un."], [(0.0, 1.0)]), [])


class BeatGridAndPlanTests(unittest.TestCase):
    BEATS = [0.2 + 0.5 * k for k in range(240)]  # 120 BPM, bars every 2 s from 0.2, covers a 2-minute video

    def test_beats_for_video_offsets_and_loops_the_track(self):
        one = tp.beats_for_video(self.BEATS, music_start=1.2, duration=5.0)
        self.assertAlmostEqual(one[0], 0.0)
        self.assertTrue(all(b < 5.0 for b in one))
        # only the section [0, 2) repeats, so the grid stays continuous across the seam
        looped = tp.beats_for_video([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], music_start=0.0, duration=7.0, loop_length=2.0)
        self.assertEqual(looped[:6], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        self.assertTrue(all(b < 7.0 for b in looped))
        self.assertEqual(tp.beats_for_video([0.1], music_start=5.0, duration=3.0), [])

    def test_loop_section_is_whole_bars_before_the_tracks_fade_out(self):
        info = {"bpm": 120.0, "duration": 92.0, "tail": 3.0}  # a bar is 2 s
        self.assertIsNone(tp.music_loop_length(info, music_start=0.0, duration=80.0))
        length = tp.music_loop_length(info, music_start=0.0, duration=108.0)
        self.assertAlmostEqual(length, 88.0)  # 89 s usable -> 44 whole bars
        self.assertAlmostEqual(length % 2.0, 0.0)
        shifted = tp.music_loop_length(info, music_start=1.0, duration=108.0)
        self.assertLessEqual(1.0 + shifted, 89.0)
        with self.assertRaises(ValueError):
            tp.music_loop_length({"bpm": 120.0, "duration": 4.0, "tail": 3.0}, music_start=0.0, duration=20.0)

    def test_black_screen_always_lands_inside_the_clip(self):
        for duration in (6.0, 12.0, 25.0, 45.0, 108.0):
            window = tp.plan_black_screen(duration, rng_seed=7)
            self.assertIsNotNone(window, duration)
            start, end = window
            self.assertGreater(start, 0.0)
            self.assertLess(end, duration)
            self.assertGreater(end - start, 1.0)

    def test_a_long_clip_gets_it_between_30_and_40_seconds(self):
        for seed in range(20):
            start, end = tp.plan_black_screen(108.0, rng_seed=seed)
            self.assertTrue(30.0 <= start <= 40.0, (seed, start))
            self.assertAlmostEqual(end - start, tp.BLACK_SCREEN_SECONDS, places=3)

    def test_it_can_be_disabled_or_pinned(self):
        self.assertIsNone(tp.plan_black_screen(108.0, at=0))
        self.assertIsNone(tp.plan_black_screen(108.0, at=-1))
        self.assertIsNone(tp.plan_black_screen(108.0, length=0))
        self.assertEqual(tp.plan_black_screen(108.0, at=50.0, length=3.0), (50.0, 53.0))

    def test_snapping_never_pulls_the_black_screen_before_30_seconds(self):
        beats = [0.27 + 0.5614 * k for k in range(200)]  # 106.9 BPM: a bar every 2.2456 s
        for seed in range(40):
            start, _ = tp.plan_black_screen(108.0, beats=beats, rng_seed=seed)
            self.assertGreaterEqual(start, 30.0, seed)

    def test_both_audio_inputs_get_one_format_for_old_ffmpeg(self):
        self.assertIn("aresample=44100", tp.COMMON_AUDIO)
        self.assertIn("channel_layouts=stereo", tp.COMMON_AUDIO)

    def test_music_snaps_the_start_to_a_bar_and_the_end_to_a_beat(self):
        for seed in range(10):
            start, end = tp.plan_black_screen(108.0, beats=self.BEATS, rng_seed=seed)
            self.assertAlmostEqual((start - 0.2) % 2.0, 0.0, delta=0.001)
            self.assertAlmostEqual((end - 0.2) % 0.5, 0.0, delta=0.001)
            self.assertGreater(end - start, 1.4)

    def test_background_cuts_follow_the_beat(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(tp, "_run") as run:
            images = [Path(temporary) / "a.png", Path(temporary) / "b.png"]
            tp.build_background_video(
                images, Path(temporary) / "bg.mp4", duration=10.0, width=270, height=480,
                cut_range=(0.5, 1.0), beat_times=[0.25 * k for k in range(1, 100)], rng_seed=3,
            )
            lines = (Path(temporary) / "bg.txt").read_text(encoding="utf-8").splitlines()
        run.assert_called_once()
        durations = [float(line.split()[1]) for line in lines if line.startswith("duration")]
        total = 0.0
        for length in durations:
            total += length
            self.assertAlmostEqual(total / 0.25, round(total / 0.25), delta=0.01)
            self.assertGreaterEqual(length, 0.3)
        self.assertGreaterEqual(total, 10.0)

    def test_captions_overlapping_the_black_screen_are_shown_without_gaps_centered(self):
        with tempfile.TemporaryDirectory() as temporary:
            ass = Path(temporary) / "c.ass"
            tp.build_captions(
                ["Un.", "Le silence profond.", "Et la nuit tombe doucement sur nous."],
                [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)], ass, width=270, height=480, force_visible=(3.0, 5.0),
            )
            events = [line for line in ass.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue")]

        def span(event):
            def seconds(text):
                h, m, rest = text.split(":")
                return int(h) * 3600 + int(m) * 60 + float(rest)

            fields = event.split(",")
            return seconds(fields[1]), seconds(fields[2])

        def y_of(event):
            return int(event.split("\\pos(")[1].split(",")[1].split(")")[0])

        forced = [e for e in events if span(e)[0] >= 1.9]  # every chunk of the two sentences under the black
        self.assertTrue(forced)
        self.assertTrue(all(y_of(e) == 240 for e in forced))  # half of the 480 px height
        spans = sorted(span(e) for e in forced)
        self.assertAlmostEqual(spans[0][0], 2.0 - tp.CAPTION_LEAD, delta=0.02)  # starts with the sentence (a hair early)
        self.assertAlmostEqual(spans[-1][1], 6.0, delta=0.02)  # holds until the end of the last sentence
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(start - end, 0.35)  # no hole in the black



def _can_burn_subtitles() -> bool:
    if not HAS_FFMPEG:
        return False
    with tempfile.TemporaryDirectory() as temporary:
        ass = Path(temporary) / "t.ass"
        tp.build_captions(["Bonjour le monde."], [(0.0, 2.0)], ass, width=270, height=480)
        escaped = str(ass).replace("\\", "/").replace(":", "\\:")
        result = subprocess.run(
            [tp.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=270x480:d=1", "-vf",
             f"ass='{escaped}'", "-frames:v", "1", str(Path(temporary) / "o.png")],
            capture_output=True,
        )
        return result.returncode == 0


@unittest.skipUnless(_can_burn_subtitles(), "this ffmpeg cannot burn ASS subtitles")
class CinemaModeTests(unittest.TestCase):
    """The cinema bars (black bars top and bottom, at chosen moments only) and a camera shake that never moves the philosopher."""

    W, H, FPS = 270, 480, 12

    def test_the_bars_leave_a_band_of_the_requested_aspect_ratio(self):
        self.assertEqual(tp.cinema_bar_height(1080, 1920), 656)  # 16:9 -> a 1080 x 607 band (a PC screen), 656 px of bar above and below
        self.assertEqual(tp.cinema_bar_height(1080, 1920, 2.39), 734)  # scope: a thinner band
        self.assertEqual(tp.CINEMA_ASPECT, 16 / 9)
        for aspect in (16 / 9, 2.39, 1.85):
            self.assertEqual(tp.cinema_bar_height(1080, 1920, aspect) % 2, 0)
        with self.assertRaisesRegex(ValueError, "no bars"):
            tp.cinema_bar_height(1080, 1920, 0.5)  # taller than the frame: nothing to letterbox

    def test_the_default_moments_are_never_the_whole_video(self):
        phases = [("grand", 0.0, 23.846), ("flash", 23.846, 31.704), ("grand", 31.704, 65.25)]
        windows = tp.plan_cinema_windows(65.25, (31.704, 38.704), phases)
        self.assertEqual(windows[0], (23.846, 38.704))  # the burst, through the black screen: the bars leave under it
        self.assertAlmostEqual(windows[1][0], 55.25, places=3)  # and the last 10 s
        self.assertGreaterEqual(windows[1][1], 65.25)  # which stay to the end
        covered = sum(min(b, 65.25) - a for a, b in windows)
        self.assertLess(covered / 65.25, 0.45)  # portrait most of the time
        self.assertEqual(tp.plan_cinema_windows(8.0, None, []), [])  # a short clip with no burst: no moment, no bars
        self.assertEqual(len(tp.plan_cinema_windows(65.25, None, phases)), 2)

    def test_the_slide_follows_the_requested_speed_curve(self):
        windows = [(2.0, 8.0)]
        expo = [tp.cinema_progress(windows, 2.0 + 1.2 * k / 10, easing="exponential") for k in range(11)]
        logar = [tp.cinema_progress(windows, 2.0 + 1.2 * k / 10, easing="logarithmic") for k in range(11)]
        self.assertEqual((expo[0], expo[-1], logar[0], logar[-1]), (0.0, 1.0, 0.0, 1.0))
        self.assertEqual(expo, sorted(expo))
        self.assertEqual(logar, sorted(logar))
        self.assertLess(expo[5], 0.2)  # slow at first: 12 % at half time
        self.assertGreater(logar[5], 0.6)  # fast at first: 74 % at half time
        self.assertEqual(tp.cinema_progress(windows, 5.0), 1.0)  # in place in the middle of the window
        self.assertEqual(tp.cinema_progress(windows, 8.5), 0.0)  # gone afterwards
        self.assertEqual(tp.cinema_progress(windows, 1.0), 0.0)  # and not there before
        self.assertLess(tp.cinema_progress(windows, 7.4), 1.0)  # leaving
        with self.assertRaises(ValueError):
            tp.cinema_graph("[a]", "[b]", windows, "spin", width=270, height=480, fps=12)
        with self.assertRaises(ValueError):
            tp.cinema_graph("[a]", "[b]", [(5.0, 5.0)], "slide", width=270, height=480, fps=12)
        self.assertIsNone(tp.cinema_graph("[a]", "[b]", [], "slide", width=270, height=480, fps=12))

    def _black_rows(self, mode, windows, easing="exponential", seconds=1.0, length=9.0):
        """Render a white video through the cinema graph and count, frame by frame, the rows of the middle column that are black."""
        import numpy as np

        graph = f"color=c=white:s={self.W}x{self.H}:r={self.FPS},format=yuv420p[s];" + tp.cinema_graph(
            "[s]", "[v]", windows, mode, width=self.W, height=self.H, fps=self.FPS, seconds=seconds, easing=easing)
        raw = subprocess.run([tp.FFMPEG, "-v", "error", "-filter_complex", graph, "-map", "[v]", "-t", str(length), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, self.H, self.W)
        return [int((frame[:, self.W // 2] < 60).sum()) for frame in frames]

    def test_instant_bars_are_in_place_for_the_whole_window_and_only_then(self):
        bar = tp.cinema_bar_height(self.W, self.H)
        rows = self._black_rows("instant", [(2.0, 4.0), (6.0, 7.0)])
        at = lambda t: rows[int(round(t * self.FPS))]  # noqa: E731
        self.assertEqual((at(0.5), at(1.9)), (0, 0))  # portrait before
        self.assertEqual((at(2.05), at(3.0), at(3.9)), (2 * bar,) * 3)  # the bars are simply there, both of them, at full height
        self.assertEqual((at(4.5), at(5.5)), (0, 0))  # and gone
        self.assertEqual(at(6.5), 2 * bar)  # a second moment
        self.assertEqual(at(8.0), 0)
        self.assertLess(self.H - 2 * bar, self.W * 9 / 16 + 3)  # what is left is a 16:9 band

    def test_sliding_bars_come_in_progressively_and_leave_the_same_way(self):
        bar = tp.cinema_bar_height(self.W, self.H)
        windows = [(2.0, 7.0)]
        rows = self._black_rows("slide", windows, seconds=1.5)
        for k, count in enumerate(rows):
            expected = 2 * bar * tp.cinema_progress(windows, k / self.FPS, seconds=1.5)
            self.assertAlmostEqual(count, expected, delta=6, msg=f"frame {k} (t={k / self.FPS:.2f})")  # whole-pixel positions: a few px
        self.assertEqual(rows[int(1.9 * self.FPS)], 0)
        self.assertEqual(rows[int(4.5 * self.FPS)], 2 * bar)
        self.assertEqual(rows[int(8.0 * self.FPS)], 0)
        entering = rows[int(2.0 * self.FPS):int(3.5 * self.FPS) + 1]
        self.assertEqual(entering, sorted(entering))  # monotonic
        self.assertTrue(0 < entering[len(entering) // 2] < 2 * bar)  # genuinely in between: progressive, not instant

    def test_exponential_and_logarithmic_really_differ_on_screen(self):
        windows = [(1.0, 8.0)]
        expo = self._black_rows("slide", windows, easing="exponential", seconds=2.0)
        logar = self._black_rows("slide", windows, easing="logarithmic", seconds=2.0)
        mid = int(2.0 * self.FPS)  # half-way through the slide
        self.assertLess(expo[mid], logar[mid] / 2)  # exponential is still slow, logarithmic has almost arrived
        self.assertEqual(expo[int(4.0 * self.FPS)], logar[int(4.0 * self.FPS)])  # both end in the same place

    def test_a_window_that_starts_at_zero_begins_in_place(self):
        windows = [(0.0, 4.0)]
        self.assertEqual(tp.cinema_progress(windows, 0.0), 1.0)  # no entrance: the bars are there on the very first frame
        self.assertEqual(tp.cinema_progress(windows, 1.0), 1.0)
        self.assertLess(tp.cinema_progress(windows, 3.5, seconds=1.2), 1.0)  # and they leave with the curve
        self.assertEqual(tp.cinema_progress(windows, 4.5), 0.0)
        expression = tp.cinema_progress_expr(windows, 1.2, "logarithmic", None)
        self.assertTrue(expression.startswith("(1)*(1-("))  # the entrance term is the constant 1
        self.assertNotIn("clip((t-0.0000)", expression)
        rows = self._black_rows("slide", windows, easing="logarithmic", seconds=1.2, length=6.0)
        bar = tp.cinema_bar_height(self.W, self.H)
        self.assertEqual(rows[0], 2 * bar)  # frame 0 is already letterboxed
        self.assertEqual(rows[int(2.0 * self.FPS)], 2 * bar)
        self.assertEqual(rows[int(5.0 * self.FPS)], 0)
        self.assertEqual(self._black_rows("instant", windows, length=6.0)[0], 2 * bar)

    def test_the_default_plan_can_open_in_landscape(self):
        phases = [("grand", 0.0, 23.846), ("flash", 23.846, 31.704), ("grand", 31.704, 65.25)]
        windows = tp.plan_cinema_windows(65.25, (31.704, 38.704), phases, start_seconds=tp.LANDSCAPE_START_SECONDS)
        self.assertEqual(windows[0], (0.0, tp.LANDSCAPE_START_SECONDS))
        self.assertEqual(len(windows), 3)  # landscape start, the burst, the ending
        self.assertEqual(tp.plan_cinema_windows(65.25, (31.704, 38.704), phases)[0][0] > 0, True)  # no start window unless asked for

    def _visible_extent(self, band_height):
        """Render a white picture through the oval opening at mid-way and measure how far the visible ellipse reaches (horizontally, vertically)."""
        import numpy as np

        graph = f"color=c=white:s={self.W}x{self.H}:r={self.FPS},format=yuv420p[s];" + tp.intro_graph(
            "[s]", "[v]", "oval", width=self.W, height=self.H, fps=self.FPS, seconds=1.0, hold=0.0, band_height=band_height)
        raw = subprocess.run([tp.FFMPEG, "-v", "error", "-filter_complex", graph, "-map", "[v]", "-frames:v", "8", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True).stdout
        frame = np.frombuffer(raw, np.uint8).reshape(-1, self.H, self.W)[4]  # about half-way through the opening
        horizontal = int((frame[self.H // 2, :] > 128).sum())
        vertical = int((frame[:, self.W // 2] > 128).sum())
        return horizontal, vertical

    def test_the_oval_opens_as_a_wide_ellipse_inside_the_landscape_band(self):
        bar = tp.cinema_bar_height(self.W, self.H)
        portrait_h, portrait_v = self._visible_extent(None)
        band_h, band_v = self._visible_extent(self.H - 2 * bar)
        self.assertGreater(portrait_v, portrait_h)  # over the whole portrait frame the ellipse is TALL
        self.assertGreater(band_h, band_v)  # in the landscape band it is WIDE: the shape of a landscape scene
        self.assertGreater(band_h / band_v, 1.3)

    def test_the_philosopher_never_shakes_but_the_background_does(self):
        import tempfile
        import numpy as np
        from PIL import Image

        w, h, fps = 180, 320, 24
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            background = tmp / "bg.mp4"
            subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=0x141414:s={w}x{h}:r={fps}:d=4", "-vf",
                            "drawbox=x=84:y=0:w=12:h=320:color=0xebebeb:t=fill", "-pix_fmt", "yuv420p", str(background)], check=True)
            character = tmp / "character.png"
            Image.new("RGBA", (100, 200), (200, 20, 20, 255)).save(character)
            out = tmp / "overlay.mp4"
            shake = tp.shake_filter(w, h, windows=[(0.0, 4.0)], preset="jitter", amount=3.0, fps=fps)
            tp.build_character_overlay(background, character, out, duration=4.0, character_width=100, rise_duration=0.1, background_filters=shake)
            raw = subprocess.run([tp.FFMPEG, "-v", "error", "-i", str(out), "-vf", "format=rgb24", "-f", "rawvideo", "-"], capture_output=True).stdout
            frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)[int(0.6 * fps):int(3.8 * fps)].astype(np.int32)
            character_edges, bar_centres = [], []
            for frame in frames:
                row = frame[int(h * 0.62)]  # across the philosopher
                red = np.where((row[:, 0] > 150) & (row[:, 1] < 90))[0]
                character_edges.append((int(red.min()), int(red.max())))
                top = frame[40, :, 0].astype(np.float64)  # above him: the bar of the background
                weights = np.clip(top - 100, 0, None)
                bar_centres.append(float((weights * np.arange(w)).sum() / weights.sum()))
            self.assertEqual(len(set(character_edges)), 1, character_edges[:5])  # not one pixel of movement
            self.assertGreater(max(bar_centres) - min(bar_centres), 1.5)  # while the background moves

    def test_a_filter_on_the_background_leaves_the_plain_overlay_untouched(self):
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            background = tmp / "bg.mp4"
            subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x303030:s=90x160:r=12:d=1", "-pix_fmt", "yuv420p", str(background)], check=True)
            character = tmp / "c.png"
            Image.new("RGBA", (40, 80), (200, 20, 20, 255)).save(character)
            for name, filters in (("plain", None), ("filtered", "hflip")):
                tp.build_character_overlay(background, character, tmp / f"{name}.mp4", duration=1.0, character_width=40, rise_duration=0.1, background_filters=filters)
                self.assertTrue((tmp / f"{name}.mp4").is_file())


class BlackSyncAndExchangeTests(unittest.TestCase):
    """The black screen ends on a sentence end near its 8 s target; after it the grandiose backgrounds are exchanged by sliding wipes."""

    ENDS = [33.78, 40.72, 43.44, 45.58, 47.96, 51.22, 53.92, 55.58, 59.74, 65.0]  # sentence ends of the real voice-over

    def test_the_black_screen_ends_on_the_sentence_end_nearest_to_eight_seconds(self):
        self.assertEqual(tp.snap_black_end_to_sentence((31.704, 39.704), self.ENDS, target_seconds=8.0, duration=65.25), (31.704, 40.72))
        self.assertEqual(tp.snap_black_end_to_sentence((31.704, 39.704), [33.78, 38.1, 41.9, 50.0], target_seconds=8.0, duration=65.25), (31.704, 38.1))  # 1.6 s vs 2.2 s from the 39.704 target
        self.assertEqual(tp.snap_black_end_to_sentence((31.704, 39.704), [33.78, 38.1, 41.2, 50.0], target_seconds=8.0, duration=65.25), (31.704, 41.2))  # 1.6 s vs 1.5 s: the later one is nearer
        # the start never moves; ends too close to the start (< half the target) or too close to the end of the video are never chosen
        self.assertEqual(tp.snap_black_end_to_sentence((31.704, 39.704), [33.78, 35.0, 63.0], target_seconds=8.0, duration=65.25), (31.704, 39.704))  # nothing eligible: left as planned
        self.assertEqual(tp.snap_black_end_to_sentence((31.704, 39.704), [], target_seconds=8.0, duration=65.25), (31.704, 39.704))

    def test_an_exact_end_stays_exact_when_the_sync_is_off_or_a_sentence_ends_there(self):
        self.assertEqual(tp.snap_black_end_to_sentence((10.0, 18.0), [18.0], target_seconds=8.0, duration=60.0), (10.0, 18.0))

    def test_the_exchanges_are_spread_evenly_and_land_on_sentence_ends_when_one_is_near(self):
        free = tp.plan_exchange_times(5, 40.0, 64.0)
        self.assertEqual([round(b - a, 3) for a, b in zip(free, free[1:])], [4.0] * 4)  # no sentences: 4 s apart
        arrivals = tp.plan_exchange_times(5, 40.72, 65.25, sentence_ends=self.ENDS)
        self.assertEqual(len(arrivals), 5)
        self.assertEqual(arrivals, sorted(arrivals))
        synced = [a for a in arrivals if any(abs(a - e) < 0.002 for e in self.ENDS)]
        self.assertGreaterEqual(len(synced), 3)  # the ones with a sentence end within a second land on it
        for a, b in zip(arrivals, arrivals[1:]):
            self.assertGreaterEqual(b - a, tp.EXCHANGE_SECONDS + 0.5 - 1e-6)  # never two wipes on top of each other
        self.assertTrue(40.72 + tp.EXCHANGE_SECONDS < arrivals[0] and arrivals[-1] < 65.25)
        self.assertEqual(tp.plan_exchange_times(0, 40.0, 64.0), [])
        with self.assertRaisesRegex(ValueError, "too short"):
            tp.plan_exchange_times(5, 40.0, 44.0)

    def test_every_direction_moves_the_new_background_the_way_it_is_named(self):
        graph = tp.exchange_graph(["right", "up", "down", "left", "right"], [45.0, 48.0, 51.0, 54.0, 57.0], fps=24)
        self.assertEqual(graph.count("overlay="), 5)
        moves = [part for part in graph.split(";") if "overlay=" in part]
        self.assertIn("x='-main_w*(1-(", moves[0])  # right: starts left of the frame
        self.assertIn("y='main_h*(1-(", moves[1])  # up: starts below the frame
        self.assertIn("y='-main_h*(1-(", moves[2])  # down: starts above the frame
        self.assertIn("x='main_w*(1-(", moves[3])  # left: starts right of the frame
        self.assertIn("enable='between(t,44.0000,48.0500)'", moves[0])  # stays under the next wipe until it has covered the frame
        self.assertIn("enable='gte(t,56.0000)'", moves[4])  # the last one stays
        self.assertEqual(tp.EXCHANGE_DIRECTIONS, ("right", "left", "up", "down"))
        self.assertEqual(tp.DEFAULT_EXCHANGE_DIRECTIONS, ("right", "up", "down", "left", "right"))  # the order the user asked for
        with self.assertRaisesRegex(ValueError, "direction"):
            tp.exchange_graph(["sideways"], [5.0], fps=24)
        with self.assertRaises(ValueError):
            tp.exchange_graph(["right"], [5.0, 6.0], fps=24)

    def test_the_wipes_slide_in_the_named_directions_on_screen(self):
        import tempfile
        from PIL import Image

        W, H, fps = 270, 480, 12
        colours = {"red": (230, 20, 20), "green": (20, 200, 20), "blue": (20, 20, 230), "yellow": (230, 220, 20), "white": (250, 250, 250)}
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            base = tmp / "base.mp4"
            subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=white:s={W}x{H}:r={fps}:d=14", "-pix_fmt", "yuv420p", str(base)], check=True)
            images = []
            for name in ("red", "green", "blue", "yellow"):
                path = tmp / f"{name}.png"
                Image.new("RGB", (W, H), colours[name]).save(path)
                images.append(path)
            out = tmp / "exchanged.mp4"
            directions, arrivals = ["right", "up", "down", "left"], [3.0, 6.0, 9.0, 12.0]
            tp.build_background_exchanges(base, out, images, directions, arrivals, width=W, height=H, fps=fps, seconds=1.0, easing="linear")

            def dominant(t, xy):
                png = tmp / f"f_{t}_{xy[0]}_{xy[1]}.png"
                subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", str(png)], check=True)
                pixel = Image.open(png).convert("RGB").getpixel(xy)
                return min(colours, key=lambda name: sum((a - b) ** 2 for a, b in zip(colours[name], pixel)))

            left, right, top, bottom = (W // 4, H // 2), (3 * W // 4, H // 2), (W // 2, H // 4), (W // 2, 3 * H // 4)
            self.assertEqual((dominant(1.0, left), dominant(1.0, right)), ("white", "white"))  # before: the base
            self.assertEqual((dominant(2.5, left), dominant(2.5, right)), ("red", "white"))  # right: half-way, it covers the left half (it came from the left)
            self.assertEqual((dominant(3.5, left), dominant(3.5, right)), ("red", "red"))  # arrived
            self.assertEqual((dominant(5.5, top), dominant(5.5, bottom)), ("red", "green"))  # up: it covers the bottom half (it came from below)
            self.assertEqual((dominant(8.5, top), dominant(8.5, bottom)), ("blue", "green"))  # down: it covers the top half (it came from above)
            self.assertEqual((dominant(11.5, left), dominant(11.5, right)), ("blue", "yellow"))  # left: it covers the right half (it came from the right)
            self.assertEqual((dominant(13.5, left), dominant(13.5, right)), ("yellow", "yellow"))
            self.assertAlmostEqual(float(_probe_duration(out)), 14.0, delta=0.2)


def _probe_duration(video: Path) -> str:
    return subprocess.run([tp.FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
                          capture_output=True, text=True).stdout.strip()


class BlackScreenRenderTests(unittest.TestCase):
    """Tiny synthetic render: the black beat is a real black screen (character included)
    while captions, voice and music continue."""

    WIDTH, HEIGHT, SECONDS = 270, 480, 8.0

    def setUp(self):
        from PIL import Image

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.backgrounds = []
        for index, colour in enumerate(((230, 230, 200), (200, 220, 240), (240, 210, 210))):
            path = self.dir / f"bg{index}.png"
            Image.new("RGB", (self.WIDTH, self.HEIGHT), colour).save(path)
            self.backgrounds.append(path)
        self.character = self.dir / "character.png"
        Image.new("RGBA", (200, 400), (180, 30, 30, 255)).save(self.character)
        self.voice = self.dir / "voice.wav"
        subprocess.run(
            [tp.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=220:duration={self.SECONDS}", str(self.voice)],
            check=True,
        )
        self.windows = [
            {"start": 0.0, "end": 2.0, "text": "Premier silence."},
            {"start": 2.0, "end": 4.0, "text": "Deuxieme silence."},
            {"start": 4.0, "end": 6.0, "text": "Troisieme silence."},
            {"start": 6.0, "end": 8.0, "text": "Quatrieme silence."},
        ]

    def _build(self, name, **kwargs):
        out = self.dir / f"{name}.mp4"
        kwargs.setdefault("black_end_on_sentence", False)  # these fixtures place the black screen exactly; the sentence sync has its own tests
        tp.build_philosopher_trend(
            " ".join(w["text"] for w in self.windows), out, work_dir=self.dir / f"work_{name}",
            character_image_path=self.character, matte=False, background_images=self.backgrounds,
            precomputed_voice=(self.voice, self.windows), width=self.WIDTH, height=self.HEIGHT, fps=12, **kwargs,
        )
        return out

    def _plan(self, name):
        return json.loads((self.dir / f"work_{name}" / "plan.json").read_text(encoding="utf-8"))

    def _frame(self, video: Path, t: float):
        from PIL import Image

        png = self.dir / f"frame_{video.stem}_{t}.png"
        subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1", str(png)], check=True)
        return Image.open(png).convert("RGB")

    @staticmethod
    def _brightest(image, box):
        return max(high for _, high in image.crop(box).getextrema())

    def _has_audio(self, video: Path) -> bool:
        out = subprocess.run(
            [tp.FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True,
        ).stdout
        return "audio" in out

    def test_black_screen_hides_everything_but_the_captions(self):
        out = self._build("plain", black_screen_at=3.0, black_screen_seconds=2.0)
        before, during = self._frame(out, 2.5), self._frame(out, 3.5)
        corner = (0, 0, 40, 40)
        self.assertGreater(self._brightest(before, corner), 150)  # normal background
        self.assertLess(self._brightest(during, corner), 12)  # black
        self.assertLess(self._brightest(during, (60, 300, 210, 400)), 12)  # the character is hidden too
        self.assertGreater(self._brightest(during, (0, 200, self.WIDTH, 280)), 150)  # centered caption still there
        self.assertTrue(self._has_audio(out))
        self.assertEqual(self._plan("plain")["black_screen"], [3.0, 5.0])

    def test_one_grand_background_then_a_quick_burst_then_black_then_the_grand_one_again(self):
        self._build("phases", black_screen_at=5.0, black_screen_seconds=2.0, flash_seconds=2.0)
        self.assertEqual(self._plan("phases")["background_phases"], [["grand", 0.0, 3.0], ["flash", 3.0, 5.0], ["grand", 5.0, 8.0]])
        stream = self.dir / "work_phases" / "bg_video.mp4"  # the background alone: the character covers the final frame's corner
        corner = lambda t: self._frame(stream, t).getpixel((5, 5))  # noqa: E731
        far = lambda a, b: max(abs(x - y) for x, y in zip(a, b)) > 15  # noqa: E731
        grand = corner(0.5)
        self.assertFalse(any(far(grand, corner(t)) for t in (1.0, 1.5, 2.0, 2.5)))  # still: no permanent cutting
        burst = [corner(t) for t in (3.1, 3.4, 3.7, 4.0, 4.3, 4.6)]
        self.assertTrue(all(far(grand, c) for c in burst))  # the burst only shows the OTHER images...
        self.assertTrue(any(far(a, b) for a, b in zip(burst, burst[1:])))  # ...and they change quickly
        self.assertFalse(far(grand, corner(7.5)))  # the grand background is back after the black screen

    def test_the_character_disappears_after_the_black_screen(self):
        out = self._build("nochar", black_screen_at=3.0, black_screen_seconds=2.0)
        plan = self._plan("nochar")
        self.assertEqual(plan["character_until"], 3.0)
        red = lambda t: self._frame(out, t).getpixel((5, 5))[1] < 80  # noqa: E731  (the test character is dark red, the background is light)
        self.assertTrue(red(2.5))  # before the black: the character covers the corner
        self.assertFalse(red(6.5))  # after it: only the grand background
        kept = self._build("kept", black_screen_at=3.0, black_screen_seconds=2.0, keep_character_after_black=True)
        self.assertTrue(self._frame(kept, 6.5).getpixel((5, 5))[1] < 80)

    def test_the_video_ends_at_the_end_of_the_first_sentence_after_the_asked_time(self):
        out = self._build("cut", black_screen_at=0, cut_after=3.0)  # sentences end at 2, 4, 6, 8 s
        plan = self._plan("cut")
        self.assertEqual(plan["cut"]["sentence_end"], 4.0)
        self.assertAlmostEqual(plan["duration"], 4.0 + 0.25, places=2)
        self.assertAlmostEqual(tp._audio_duration(out), 4.25, delta=0.25)
        ass = (self.dir / "work_cut" / "captions.ass").read_text(encoding="utf-8")
        self.assertIn("SILENCE", ass.upper())
        self.assertNotIn("TROISIEME", ass.upper())  # the sentences after the cut are gone
        full = self._build("nocut", black_screen_at=0)
        self.assertAlmostEqual(tp._audio_duration(full), self.SECONDS, delta=0.3)

    def test_by_default_a_long_video_stops_at_the_end_of_the_first_sentence_after_one_minute(self):
        long_voice = self.dir / "long.wav"
        subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=72", str(long_voice)], check=True)
        windows = [{"start": 5.0 * i, "end": 5.0 * i + 4.0, "text": f"Phrase numero {i}."} for i in range(14)]  # ends at 4, 9, ..., 64, 69
        out = self.dir / "long.mp4"
        tp.build_philosopher_trend(
            " ".join(w["text"] for w in windows), out, work_dir=self.dir / "work_long", character_image_path=self.character,
            matte=False, background_images=self.backgrounds, precomputed_voice=(long_voice, windows),
            width=self.WIDTH, height=self.HEIGHT, fps=8, black_screen_at=0,
        )
        plan = json.loads((self.dir / "work_long" / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["cut"]["sentence_end"], 64.0)  # 59.0 is under the minute, 64.0 is the first end after it
        self.assertAlmostEqual(plan["duration"], 64.25, places=2)
        self.assertAlmostEqual(tp._audio_duration(out), 64.25, delta=0.3)

    def test_softfocus_style_renders_a_fade_into_the_black_and_out_of_it(self):
        from unittest import mock

        with mock.patch.object(tp, "align_word_times", side_effect=AssertionError("given word_times must skip the speech model")):
            out = self._build("soft", black_screen_at=3.0, black_screen_seconds=2.0, blur_style="softfocus", word_times=[[(0.0, 1.0), (1.0, 2.0)]], align_words=True)
        self.assertEqual(self._plan("soft")["blur_style"], "softfocus")
        stream_corner = lambda t: self._brightest(self._frame(out, t), (0, 0, 40, 40))  # noqa: E731
        self.assertGreater(stream_corner(2.4), 150)  # before the fade
        self.assertLess(stream_corner(5.0), 12)  # inside the black
        self.assertTrue(12 < stream_corner(2.85) < stream_corner(2.4))  # a fade, not a hard cut: partly dark just before the black
        self.assertTrue(12 < stream_corner(5.2) < stream_corner(7.0))  # and the picture comes back progressively after it

    def test_the_opening_starts_black_and_opens_onto_the_picture(self):
        for style in ("eyelid", "oval"):
            out = self._build(f"intro_{style}", black_screen_at=0, intro_style=style, intro_seconds=1.0)
            first, last = self._frame(out, 0.02), self._frame(out, 2.0)
            self.assertLess(self._brightest(first, (0, 0, 40, 40)), 12)  # closed: the corner is black (the captions are drawn over the lids)
            self.assertGreater(self._brightest(last, (0, 0, 40, 40)), 150)  # open: the picture is there
            mid = self._frame(out, 0.6)  # half open: the centre is lit before the corner
            self.assertGreater(self._brightest(mid, (self.WIDTH // 2 - 20, self.HEIGHT // 2 - 20, self.WIDTH // 2 + 20, self.HEIGHT // 2 + 20)),
                               self._brightest(mid, (0, 0, 20, 20)))
        none = self._build("intro_none", black_screen_at=0, intro_style="none")
        self.assertGreater(self._brightest(self._frame(none, 0.02), (0, 0, 40, 40)), 150)  # no opening: there from frame 0
        self.assertEqual(self._plan("intro_eyelid")["intro"], {"style": "eyelid", "seconds": 1.0, "landscape": False, "band_px": None})

    def test_every_burst_transition_renders_and_keeps_the_planned_length(self):
        for style in tp.BURST_TRANSITIONS:
            out = self._build(f"burst_{style}", black_screen_at=5.0, black_screen_seconds=2.0, flash_seconds=2.0,
                              intro_style="none", burst_transition=style)
            plan = self._plan(f"burst_{style}")
            self.assertEqual(plan["burst_transition"], style)
            stream = self.dir / f"work_burst_{style}" / "bg_video.mp4"
            self.assertAlmostEqual(tp._audio_duration(stream) if False else float(subprocess.run(
                [tp.FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(stream)], capture_output=True, text=True).stdout),
                self.SECONDS, delta=0.25)

    def test_a_wipe_shows_two_images_at_once_in_the_middle_of_a_cut_and_a_plain_cut_never_does(self):
        palette = [(230, 230, 200), (200, 220, 240), (240, 210, 210)]  # the three flat test backgrounds

        def near(pixel, colour):
            return max(abs(a - b) for a, b in zip(pixel, colour)) < 14

        def mixed_frames(style):
            stream = self.dir / f"bg_{style}.mp4"
            tp.build_phased_background(self.backgrounds[0], self.backgrounds[1:], [("grand", 0.0, 1.0), ("flash", 1.0, 3.0), ("grand", 3.0, 4.0)],
                                       stream, width=self.WIDTH, height=self.HEIGHT, fps=24, transition=style, transition_seconds=0.12)
            # decode EVERY frame (a wipe lasts ~3 frames and ends on the planned cut time); 2 rows: yuv420 crops are even-sized
            raw = subprocess.run([tp.FFMPEG, "-v", "error", "-i", str(stream), "-vf", f"crop={self.WIDTH}:2:0:100", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                 capture_output=True).stdout
            frame_bytes = self.WIDTH * 2 * 3
            count = 0
            for i in range(0, len(raw) - frame_bytes + 1, frame_bytes):
                pixels = [tuple(raw[i + x * 3:i + x * 3 + 3]) for x in range(self.WIDTH)]
                seen = [sum(near(px, colour) for px in pixels) for colour in palette]
                count += sum(n >= 3 for n in seen) >= 2
            return count

        self.assertEqual(mixed_frames("cut"), 0)
        self.assertEqual(mixed_frames("shake"), 0)
        for style in ("wipe_left", "wipe_right", "swing"):
            self.assertGreater(mixed_frames(style), 0, style)

    def test_the_character_can_wait_and_come_from_the_top(self):
        out = self._build("char_top", black_screen_at=0, intro_style="none", character_rise_delay=1.5, character_rise_seconds=1.0, character_rise_from="top")
        self.assertEqual(self._plan("char_top")["character_entry"], {"seconds": 1.0, "delay": 1.5, "from": "top", "final_y": tp.DEFAULT_CHARACTER_FINAL_Y, "width": tp.DEFAULT_CHARACTER_WIDTH})
        dark_red = lambda t, box: self._frame(out, t).crop(box).getpixel((0, 0))[1] < 80  # noqa: E731  (the test character is dark red)
        self.assertFalse(dark_red(1.0, (5, 5, 6, 6)))  # still waiting: nothing in the corner
        self.assertTrue(dark_red(3.5, (5, 5, 6, 6)))  # arrived

    def test_swing_wipes_and_a_shake_through_the_pipeline(self):
        out = self._build("wipes", black_screen_at=3.0, black_screen_seconds=2.0, intro_style="none", black_transition="swing",
                          black_wipe_seconds=0.4, black_wipe_easing="logarithmic", transition_shake="rock", transition_shake_seconds=0.6)
        plan = self._plan("wipes")
        self.assertEqual((plan["black_transition"], plan["transition_shake"]), ("swing", "rock"))
        self.assertEqual(plan["black_wipe"], {"seconds": 0.4, "easing": "logarithmic", "panel": True})  # another burst image on the other side
        self.assertAlmostEqual(tp._audio_duration(out), self.SECONDS, delta=0.3)
        corner = lambda t: self._brightest(self._frame(out, t), (0, 0, 40, 40))  # noqa: E731
        self.assertGreater(corner(2.0), 150)  # before the wipe
        self.assertLess(corner(4.0), 12)  # inside the black screen
        self.assertGreater(corner(6.5), 150)  # the picture is back

    def test_a_permanent_shake_runs_through_the_whole_pipeline(self):
        out = self._build("perm_shake", black_screen_at=3.0, black_screen_seconds=2.0, intro_style="none", permanent_shake="drift",
                          permanent_shake_amount=1.0, transition_shake="rock")  # the permanent one replaces the transition-only one
        plan = self._plan("perm_shake")
        self.assertEqual((plan["permanent_shake"], plan["black_transition"]), ("drift", "blur"))  # soft focus (blur) stays the default
        self.assertAlmostEqual(tp._audio_duration(out), self.SECONDS, delta=0.3)
        corner = lambda t: self._brightest(self._frame(out, t), (0, 0, 40, 40))  # noqa: E731
        self.assertGreater(corner(2.0), 150)  # the picture, whole, with no black corner even though it is shaking
        self.assertLess(corner(4.0), 12)  # the black screen
        self.assertGreater(corner(6.5), 150)
        args = tp.build_parser().parse_args(["--script", "s.txt", "--out", "o.mp4", "--work-dir", "w", "--permanent-shake", "drift",
                                             "--permanent-shake-amount", "0.8"])
        self.assertEqual((args.permanent_shake, args.permanent_shake_amount), ("drift", 0.8))
        self.assertIsNone(tp.build_parser().parse_args(["--script", "s.txt", "--out", "o.mp4", "--work-dir", "w"]).permanent_shake)  # off by default

    def test_the_cinema_bars_come_in_at_the_chosen_moments_only(self):
        out = self._build("cinema", black_screen_at=5.0, black_screen_seconds=2.0, intro_style="none", cinema_windows=[(1.0, 3.0)],
                          cinema_mode="instant")
        plan = self._plan("cinema")
        self.assertEqual(plan["cinema"]["windows"], [[1.0, 3.0]])
        self.assertEqual((plan["cinema"]["mode"], plan["cinema"]["bar_px"]), ("instant", tp.cinema_bar_height(self.WIDTH, self.HEIGHT)))
        bar = plan["cinema"]["bar_px"]
        top, bottom, middle = (0, 0, 40, 40), (0, self.HEIGHT - 40, 40, self.HEIGHT), (0, self.HEIGHT // 2 - 20, 40, self.HEIGHT // 2 + 20)
        before, during, after = self._frame(out, 0.4), self._frame(out, 2.0), self._frame(out, 3.8)
        self.assertGreater(self._brightest(before, top), 150)  # portrait before
        self.assertLess(self._brightest(during, top), 12)  # bar at the top
        self.assertLess(self._brightest(during, bottom), 12)  # and at the bottom
        self.assertGreater(self._brightest(during, middle), 150)  # the picture between them
        self.assertEqual(self._brightest(during, (0, bar + 2, 40, bar + 12)) > 150, True)  # which starts right under the top bar
        self.assertGreater(self._brightest(after, top), 150)  # portrait again
        self.assertLess(self._brightest(self._frame(out, 5.5), top), 12)  # the black screen is still black

    def test_the_default_cinema_plan_uses_the_burst_and_a_sliding_exponential(self):
        self._build("cinema_auto", black_screen_at=4.0, black_screen_seconds=2.0, cinema=True)
        cinema = self._plan("cinema_auto")["cinema"]
        self.assertEqual((cinema["mode"], cinema["easing"], cinema["aspect"]), ("slide", "exponential", round(16 / 9, 4)))
        self.assertTrue(cinema["windows"])
        for start, end in cinema["windows"]:
            self.assertTrue(0 <= start < end)
        self.assertIsNone(self._plan("legacy" if False else "cinema_auto").get("missing"))
        self._build("no_cinema", black_screen_at=4.0, black_screen_seconds=2.0)
        self.assertIsNone(self._plan("no_cinema")["cinema"])  # off unless asked for

    def test_the_cinema_options_are_validated(self):
        with self.assertRaises(ValueError):
            self._build("bad_mode", cinema=True, cinema_mode="spin")
        with self.assertRaises(ValueError):
            self._build("bad_ease", cinema=True, cinema_easing="bounce")
        with self.assertRaises(ValueError):
            self._build("bad_aspect", cinema=True, cinema_aspect=0.4)

    def test_every_shake_is_recorded_as_moving_the_background_only(self):
        self._build("shake_target", black_screen_at=3.0, black_screen_seconds=2.0, intro_style="none", permanent_shake="drift")
        self.assertEqual(self._plan("shake_target")["shake_target"], "background")

    def test_the_shake_is_applied_to_the_background_before_the_philosopher_is_laid_on_it(self):
        calls, real_run = [], tp._run

        def spy(command, *args, **kwargs):
            calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        with mock.patch.object(tp, "_run", side_effect=spy):
            self._build("shake_order", black_screen_at=3.0, black_screen_seconds=2.0, intro_style="none", permanent_shake="drift",
                        transition_shake="rock")
        graph_of = lambda command: command[command.index("-filter_complex") + 1]  # noqa: E731
        composite = next(c for c in calls if c[-1].endswith("composite.mp4"))
        self.assertRegex(graph_of(composite), r"\[0:v\]pad=.*perspective=.*\[bgs\];\[bgs\]\[char\]overlay")  # background shaken, THEN the character on top
        final = next(c for c in calls if c[-1].endswith("shake_order.mp4"))
        self.assertNotIn("perspective=", graph_of(final))  # the last pass (blur, wipes, bars, captions) never moves the picture again
        self.assertNotIn("[sk]", graph_of(final))

    def test_the_video_can_open_in_landscape_and_the_philosopher_enters_when_the_bars_leave(self):
        out = self._build("landscape", black_screen_at=6.0, black_screen_seconds=1.5, intro_style="oval", landscape_start_seconds=3.0,
                          cinema_mode="instant", character_rise_seconds=0.2)
        plan = self._plan("landscape")
        bar = tp.cinema_bar_height(self.WIDTH, self.HEIGHT)
        self.assertEqual(plan["cinema"]["windows"], [[0.0, 3.0]])  # only the landscape start: no burst / ending moments unless cinema=True
        self.assertEqual((plan["landscape_start"], plan["intro"]["landscape"], plan["intro"]["band_px"]), (3.0, True, self.HEIGHT - 2 * bar))
        self.assertEqual(plan["character_entry"]["delay"], 3.0)  # instant bars: he enters when they are gone
        top = (0, 0, 40, 40)
        self.assertLess(self._brightest(self._frame(out, 0.3), top), 12)  # bar in place from the start (and the intro is still black anyway)
        self.assertLess(self._brightest(self._frame(out, 2.0), top), 12)  # the landscape scene: the bar is there
        self.assertGreater(self._brightest(self._frame(out, 2.0), (0, self.HEIGHT // 2 - 20, 40, self.HEIGHT // 2 + 20)), 150)  # with the picture in the band
        self.assertGreater(self._brightest(self._frame(out, 4.0), top), 150)  # portrait afterwards

    def test_a_sliding_landscape_start_lets_the_philosopher_enter_as_the_bars_start_to_leave(self):
        self._build("landscape_slide", black_screen_at=7.0, black_screen_seconds=1.0, intro_style="oval", landscape_start_seconds=4.0, cinema_seconds=1.0)
        plan = self._plan("landscape_slide")
        self.assertEqual(plan["character_entry"]["delay"], 3.0)  # 4.0 s portrait again, the bars leave over the second before
        self._build("landscape_late", black_screen_at=7.0, black_screen_seconds=1.0, landscape_start_seconds=4.0, cinema_seconds=1.0, character_rise_delay=3.5)
        self.assertEqual(self._plan("landscape_late")["character_entry"]["delay"], 3.5)  # an explicit later delay wins
        self._build("portrait", black_screen_at=7.0, black_screen_seconds=1.0)
        self.assertEqual((self._plan("portrait")["landscape_start"], self._plan("portrait")["character_entry"]["delay"]), (None, 0.0))

    def test_the_oval_outside_a_landscape_start_is_flagged(self):
        with mock.patch("sys.stderr") as err:
            self._build("oval_portrait", black_screen_at=0, intro_style="oval")
        self.assertIn("landscape", "".join(call.args[0] for call in err.write.call_args_list))
        with mock.patch("sys.stderr") as err:
            self._build("oval_landscape", black_screen_at=0, intro_style="oval", landscape_start_seconds=3.0)
        self.assertNotIn("landscape scene", "".join(call.args[0] for call in err.write.call_args_list))

    def test_the_philosopher_is_smaller_and_his_width_can_be_chosen(self):
        self.assertEqual(tp.DEFAULT_CHARACTER_WIDTH, 540)  # was 760
        self.assertEqual(tp.build_parser().parse_args(["--script", "s", "--out", "o", "--work-dir", "w"]).character_width, 540)
        self.assertEqual(tp.build_parser().parse_args(["--script", "s", "--out", "o", "--work-dir", "w", "--character-width", "480"]).character_width, 480)
        self.assertEqual(tp.build_parser().parse_args(["--script", "s", "--out", "o", "--work-dir", "w", "--landscape-start"]).landscape_start, tp.LANDSCAPE_START_SECONDS)
        self.assertEqual(tp.build_parser().parse_args(["--script", "s", "--out", "o", "--work-dir", "w", "--landscape-start", "5"]).landscape_start, 5.0)
        self.assertIsNone(tp.build_parser().parse_args(["--script", "s", "--out", "o", "--work-dir", "w"]).landscape_start)
        calls, real_run = [], tp._run

        def spy(command, *args, **kwargs):
            calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        with mock.patch.object(tp, "_run", side_effect=spy):
            self._build("narrow", black_screen_at=0, intro_style="none", character_width=120)
            self._build("default_width", black_screen_at=0, intro_style="none")
        graph_of = lambda command: command[command.index("-filter_complex") + 1]  # noqa: E731
        composites = [c for c in calls if c[-1].endswith("composite.mp4")]
        self.assertIn("scale=120:-1", graph_of(composites[0]))
        self.assertIn(f"scale={tp.DEFAULT_CHARACTER_WIDTH}:-1", graph_of(composites[1]))
        self.assertEqual(self._plan("narrow")["character_entry"]["width"], 120)

    def test_the_black_screen_ends_on_a_sentence_end_in_the_pipeline(self):
        out = self._build("sync", black_screen_at=1.0, black_screen_seconds=2.6, intro_style="none", black_end_on_sentence=True)
        plan = self._plan("sync")
        self.assertEqual(plan["black_screen"], [1.0, 4.0])  # target 3.6 s -> the sentence ends are 2, 4, 6, 8 -> 4.0
        self.assertEqual(plan["black_end"], {"on_sentence": True, "planned": [1.0, 3.6], "target_seconds": 2.6, "seconds": 3.0})
        top = (0, 0, 40, 40)
        self.assertLess(self._brightest(self._frame(out, 3.8), top), 12)  # still black just before the sentence ends
        self.assertGreater(self._brightest(self._frame(out, 4.8), top), 150)  # back as the next sentence starts
        self._build("nosync", black_screen_at=1.0, black_screen_seconds=2.6, intro_style="none", black_end_on_sentence=False)
        self.assertEqual(self._plan("nosync")["black_screen"], [1.0, 3.6])

    def test_the_grandiose_backgrounds_are_exchanged_after_the_black_screen(self):
        out = self._build("exchange", black_screen_at=1.0, black_screen_seconds=2.6, intro_style="none", black_end_on_sentence=True,
                          exchange_backgrounds=[self.backgrounds[1], self.backgrounds[2]], exchange_directions=["right", "up"], exchange_seconds=0.5,
                          blur=0)
        plan = self._plan("exchange")
        exchange = plan["exchange"]
        self.assertEqual((exchange["directions"], exchange["images"]), (["right", "up"], ["bg0.png", "bg1.png", "bg2.png"]))  # the grand one, then the others
        first, second = exchange["arrivals"]
        self.assertTrue(plan["black_screen"][1] < first < second < self.SECONDS)
        near = lambda pixel, colour: max(abs(a - b) for a, b in zip(pixel, colour)) < 16  # noqa: E731
        spot = (self.WIDTH // 2, 60)
        self.assertTrue(near(self._frame(out, plan["black_screen"][1] + 0.3).getpixel(spot), (230, 230, 200)))  # the grand background comes back
        self.assertTrue(near(self._frame(out, min(first + 0.4, second - 0.3)).getpixel(spot), (200, 220, 240)))  # then the first exchanged one
        self.assertTrue(near(self._frame(out, second + 0.4).getpixel(spot), (240, 210, 210)))  # then the second
        self.assertTrue((self.dir / "work_exchange" / "bg_video_exchange.mp4").is_file())
        none = self._build("no_exchange", black_screen_at=1.0, black_screen_seconds=2.6)
        self.assertIsNone(self._plan("no_exchange")["exchange"])  # opt-in
        self.assertTrue(none.is_file())

    def test_another_shake_takes_over_after_the_black_screen(self):
        calls, real_run = [], tp._run

        def spy(command, *args, **kwargs):
            calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        with mock.patch.object(tp, "_run", side_effect=spy):
            self._build("two_shakes", black_screen_at=3.0, black_screen_seconds=2.0, intro_style="none", permanent_shake="drift", after_black_shake="rock")
        plan = self._plan("two_shakes")
        self.assertEqual((plan["permanent_shake"], plan["after_black_shake"]), ("drift", "rock"))
        composite = next(c for c in calls if c[-1].endswith("composite.mp4"))
        graph = composite[composite.index("-filter_complex") + 1]
        self.assertEqual(graph.count("perspective="), 2)  # drift, then rock, both on the background before the philosopher
        self.assertIn("2*PI*0.9*", graph)  # drift's waves
        self.assertIn("2*PI*1.0*", graph)  # rock's wave
        self.assertIn("(5.000-(in/12))", graph)  # drift stops at the end of the black screen (5.0 s)
        self.assertIn("((in/12)-5.000)", graph)  # rock starts there

    def test_the_exchange_and_black_sync_flags_on_the_command_line(self):
        parse = lambda *extra: tp.build_parser().parse_args(["--script", "s.txt", "--out", "o.mp4", "--work-dir", "w", *extra])  # noqa: E731
        args = parse("--no-black-end-on-sentence", "--after-black-shake", "rock", "--exchange-backgrounds", "a.png", "b.png", "--exchange-directions",
                     "right", "up", "down", "--exchange-seconds", "0.8", "--exchange-easing", "logarithmic")
        self.assertEqual((args.black_end_on_sentence, args.after_black_shake, args.exchange_backgrounds, args.exchange_directions, args.exchange_seconds,
                          args.exchange_easing), (False, "rock", [Path("a.png"), Path("b.png")], ["right", "up", "down"], 0.8, "logarithmic"))
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            parse("--exchange-directions", "diagonal")

    def test_the_cinema_flags_on_the_command_line(self):
        parse = lambda *extra: tp.build_parser().parse_args(["--script", "s.txt", "--out", "o.mp4", "--work-dir", "w", *extra])  # noqa: E731
        default = parse()
        self.assertEqual((default.cinema, default.cinema_windows, default.cinema_mode, default.cinema_easing, default.cinema_seconds, default.cinema_aspect),
                         (False, None, "slide", "exponential", 1.2, 16 / 9))
        args = parse("--cinema", "--cinema-windows", "23.8-38.7", "55-65", "--cinema-mode", "instant", "--cinema-easing", "logarithmic",
                     "--cinema-seconds", "0.8", "--cinema-aspect", "2.39")
        self.assertEqual((args.cinema, args.cinema_windows, args.cinema_mode, args.cinema_easing, args.cinema_seconds, args.cinema_aspect),
                         (True, [(23.8, 38.7), (55.0, 65.0)], "instant", "logarithmic", 0.8, 2.39))
        self.assertAlmostEqual(parse("--cinema-aspect", "16:9").cinema_aspect, 16 / 9)
        for bad in (("--cinema-windows", "10"), ("--cinema-windows", "9-3"), ("--cinema-aspect", "wide"), ("--cinema-mode", "fade")):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                parse(*bad)

    def test_permanent_cuts_can_still_be_asked_for(self):
        self._build("legacy", black_screen_at=5.0, black_screen_seconds=2.0, permanent_cuts=True)
        self.assertEqual(self._plan("legacy")["background_phases"], "permanent_cuts")

    def test_a_short_clip_still_gets_its_black_screen(self):
        out = self._build("auto")
        start, end = self._plan("auto")["black_screen"]
        self.assertTrue(0 < start < end < self.SECONDS)
        self.assertLess(self._brightest(self._frame(out, (start + end) / 2), (0, 0, 40, 40)), 12)

    def test_music_is_mixed_and_the_black_screen_snaps_to_its_bars(self):
        music = click_track(self.dir / "music.wav", bpm=120, offset=0.2, seconds=20)
        out = self._build("music", music_path=music)
        plan = self._plan("music")
        self.assertAlmostEqual(plan["bpm"], 120, delta=3)
        off_bar = abs(((plan["black_screen"][0] - 0.2) + 1.0) % 2.0 - 1.0)  # distance to the nearest bar line
        self.assertLess(off_bar, 0.08)
        self.assertTrue(self._has_audio(out))
        self.assertAlmostEqual(tp._audio_duration(out), self.SECONDS, delta=0.3)

    def test_music_shorter_than_the_video_loops_instead_of_going_silent(self):
        music = click_track(self.dir / "short.wav", bpm=120, offset=0.2, seconds=6)
        out = self._build("loop", music_path=music, black_screen_at=0)
        late = self.dir / "late.wav"
        subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-ss", "6", "-t", "1.5", "-i", str(out), "-vn", str(late)], check=True)
        stats = subprocess.run(
            [tp.FFMPEG, "-v", "info", "-i", str(late), "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True
        ).stderr
        self.assertNotIn("max_volume: -91", stats)  # the late part of the clip still carries audio

    def test_no_black_screen_when_disabled(self):
        out = self._build("off", black_screen_at=0)
        self.assertIsNone(self._plan("off")["black_screen"])
        self.assertGreater(self._brightest(self._frame(out, 3.5), (0, 0, 40, 40)), 150)


if __name__ == "__main__":
    unittest.main()
