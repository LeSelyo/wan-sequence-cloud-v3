"""2026-10-03 philosopher fixes: a real full-frame black beat (captions, voice and music keep
going), music beat grid, beat-snapped cuts, and a tiny end-to-end render on synthetic inputs."""
import json
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
        self.assertIn("\\fscx114", joined)  # the key word is a bit bigger
        for event in events:
            self.assertLessEqual(event.count("\\c&H00FFFFFF&"), 1)

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
        words = ["Un", "grand", "silence."]
        _, events = self._events(["Un grand silence."], [(0.0, 3.0)], word_times=[[(0.0, 0.2), (1.0, 1.5), (2.5, 3.0)]])
        # one chunk of three words, from the first word's start
        self.assertEqual(len(events), 1)
        self.assertIn("0:00:00.00", events[0])
        del words


class BeatGridAndPlanTests(unittest.TestCase):
    BEATS = [0.2 + 0.5 * k for k in range(60)]  # 120 BPM, bars every 2 s from 0.2

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

        forced = [e for e in events if span(e)[0] >= 1.99]  # every chunk of the two sentences under the black
        self.assertTrue(forced)
        self.assertTrue(all(y_of(e) == 240 for e in forced))  # half of the 480 px height
        spans = sorted(span(e) for e in forced)
        self.assertAlmostEqual(spans[0][0], 2.0, delta=0.02)  # starts with the sentence
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
