"""The end title card: leaning-letters wave on the cinema bars (scripts/title_card.py) and its place in the video."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import title_card as tc  # noqa: E402
import trend_philosopher as tp  # noqa: E402

FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\DejaVuSans-Bold.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"), Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
)
FONT = next((path for path in FONT_CANDIDATES if path.is_file()), None)


class TitleTextTests(unittest.TestCase):
    def test_lines_are_capitals_and_the_separators_work(self):
        self.assertEqual(tc.title_lines("Une vie|exister sans être vue ?"), ["UNE VIE", "EXISTER SANS ÊTRE VUE ?"])
        self.assertEqual(tc.title_lines("a\nb\n\n c "), ["A", "B", "C"])
        with self.assertRaises(ValueError):
            tc.title_lines(" | \n ")

    def test_the_constants_match_the_pipeline(self):
        self.assertEqual((tp.END_TITLE_EFFECTS, tp.END_TITLE_POSITIONS, tp.END_TITLE_SECONDS, tp.DEFAULT_END_TITLE_EFFECT),
                         (tc.TITLE_EFFECTS, tc.TITLE_POSITIONS, tc.TITLE_SECONDS, tc.DEFAULT_TITLE_EFFECT))

    def test_a_missing_font_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            tc.find_font("no-such-font-here.ttf")

    @unittest.skipUnless(FONT, "no bold font file found")
    def test_a_font_is_found_by_path_and_by_file_name(self):
        self.assertEqual(tc.find_font(FONT), FONT)
        self.assertTrue(tc.find_font(FONT.name).is_file())


class ShearWaveTests(unittest.TestCase):
    """Vertical bars make the slant measurable: a bar leaning forward has its top to the right of its bottom."""

    def _bars(self, lines=1):
        mask = np.zeros((200 * lines, 1000), np.uint8)
        rows = []
        for i in range(lines):
            top, bottom = 50 + 200 * i, 150 + 200 * i
            for x in (20, 495, 972):  # left end, middle, right end of the line
                mask[top:bottom, x:x + 8] = 255
            rows.append((top, bottom))
        return mask, rows

    @staticmethod
    def _lean(shaped, x, top, bottom):
        """(x of the bar near the top) - (x of the bar near the bottom), in pixels, around column x"""
        def centre(row):
            cols = np.arange(max(x - 60, 0), min(x + 70, shaped.shape[1]))
            weights = shaped[row, cols].astype(float)
            return float((weights * cols).sum() / weights.sum())
        return centre(top + 5) - centre(bottom - 5)

    def test_one_end_leans_forward_the_middle_stands_and_the_other_end_leans_back(self):
        mask, rows = self._bars()
        shaped = tc.shear_wave(mask, rows, 0.0)
        left, middle, right = (self._lean(shaped, x, 50, 150) for x in (24, 499, 976))
        self.assertGreater(left, 20)  # forward slash: the top is well to the right of the bottom (0.32 x 90 rows = ~29 px)
        self.assertLess(abs(middle), 5)  # straight
        self.assertLess(right, -20)  # back slash

    def test_the_next_line_is_the_mirror_of_the_first(self):
        mask, rows = self._bars(lines=2)
        shaped = tc.shear_wave(mask, rows, 0.0)
        first_left = self._lean(shaped, 24, 50, 150)
        second_left = self._lean(shaped, 24, 250, 350)
        self.assertGreater(first_left, 20)
        self.assertLess(second_left, -20)

    def test_the_wave_travels(self):
        mask, rows = self._bars()
        a, b = tc.shear_wave(mask, rows, 0.0), tc.shear_wave(mask, rows, 1.0)  # a quarter of the 4 s period later
        self.assertGreater(abs(self._lean(a, 24, 50, 150) - self._lean(b, 24, 50, 150)), 15)
        full = tc.shear_wave(mask, rows, tc.TITLE_WAVE_PERIOD_SECONDS)  # one whole period: back where it started
        self.assertTrue(np.allclose(a.astype(int), full.astype(int), atol=3))

    def test_blank_rows_and_empty_lines_are_left_alone(self):
        empty = np.zeros((100, 200), np.uint8)
        self.assertEqual(int(tc.shear_wave(empty, [(10, 90)], 0.5).sum()), 0)


@unittest.skipUnless(FONT, "no bold font file found")
class TitleFramesTests(unittest.TestCase):
    def test_frames_fade_in_and_hold_and_keep_the_bar_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            count = tc.render_title_frames("Une vie|exister ?", FONT, out, width=540, height=164, fps=12, seconds=2.0, fade_seconds=0.5)
            self.assertEqual(count, 24)
            self.assertEqual(len(list(out.glob("title_*.png"))), 24)
            first, middle, last = (Image.open(out / f"title_{k:04d}.png") for k in (0, 12, 23))
            self.assertEqual((first.mode, first.size), ("RGBA", (540, 164)))
            self.assertEqual(int(np.asarray(first)[..., 3].max()), 0)  # invisible on the first frame
            self.assertEqual(int(np.asarray(middle)[..., 3].max()), 255)  # fully there once the fade is over
            self.assertEqual(int(np.asarray(last)[..., 3].max()), 255)
            alpha = np.asarray(last)[..., 3]
            ys, xs = np.nonzero(alpha > 40)
            self.assertLess(xs.max() - xs.min(), 540 * 0.92)  # the text fits inside the bar...
            self.assertGreater(xs.max() - xs.min(), 540 * 0.4)  # ... and fills a good part of it
            self.assertTrue(ys.min() > 5 and ys.max() < 158)

    def test_the_shear_wave_moves_and_the_plain_effect_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            tc.render_title_frames("Une vie|exister ?", FONT, tmp / "wave", width=540, height=164, fps=12, seconds=3.0, fade_seconds=0.2)
            tc.render_title_frames("Une vie|exister ?", FONT, tmp / "plain", width=540, height=164, fps=12, seconds=3.0, fade_seconds=0.2, effect="plain")
            alpha = lambda folder, k: np.asarray(Image.open(tmp / folder / f"title_{k:04d}.png"))[..., 3].astype(int)  # noqa: E731
            self.assertGreater(int(np.abs(alpha("wave", 8) - alpha("wave", 20)).sum()), 20000)  # the letters change shape over time
            self.assertEqual(int(np.abs(alpha("plain", 8) - alpha("plain", 20)).sum()), 0)  # a straight title is the same frame after frame
        with self.assertRaises(ValueError):
            tc.render_title_frames("x", FONT, Path(tempfile.gettempdir()) / "unused_title", width=100, height=60, fps=12, effect="spin")


class CaptionsKeepClearTests(unittest.TestCase):
    def _anchors(self, avoid):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "c.ass"
            sentences = [f"phrase numéro {i} du test" for i in range(40)]
            windows = [(i * 2.0, i * 2.0 + 1.8) for i in range(40)]
            tp.build_captions(sentences, windows, out, width=1080, height=1920, avoid=avoid)
            text = out.read_text(encoding="utf-8")
        import re

        return [(float(h) * 3600 + float(m) * 60 + float(s), int(y)) for h, m, s, y in re.findall(r"Dialogue: 0,(\d+):(\d+):([\d.]+),.*?\\pos\(\d+,(\d+)\)", text)]

    def test_no_caption_is_anchored_in_the_zone_while_the_title_is_up(self):
        free = self._anchors(None)
        self.assertTrue(any(y < 0.34 * 1920 and 20 <= t <= 60 for t, y in free))  # without the rule some captions do sit in the top rows
        kept = self._anchors((20.0, 60.0, 0.0, 656 / 1920))
        self.assertFalse(any(y < 0.34 * 1920 + 0.05 * 1920 and 20.0 <= t <= 60.0 for t, y in kept))
        outside = lambda rows: [(t, y) for t, y in rows if t < 18 or t > 62]  # noqa: E731
        self.assertEqual(outside(free), outside(kept))  # everything else is untouched


class TitleInThePipelineTests(unittest.TestCase):
    WIDTH, HEIGHT, SECONDS = 270, 480, 8.0

    def setUp(self):
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
        subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=220:duration={self.SECONDS}", str(self.voice)], check=True)
        self.windows = [{"start": 2.0 * i, "end": 2.0 * i + 2.0, "text": f"Phrase {i}."} for i in range(4)]

    def _build(self, name, **kwargs):
        out = self.dir / f"{name}.mp4"
        tp.build_philosopher_trend(
            " ".join(w["text"] for w in self.windows), out, work_dir=self.dir / f"work_{name}", character_image_path=self.character, matte=False,
            background_images=self.backgrounds, precomputed_voice=(self.voice, self.windows), width=self.WIDTH, height=self.HEIGHT, fps=12,
            black_screen_at=0, intro_style="none", **kwargs)
        return out

    def _frame(self, video, t):
        png = self.dir / f"frame_{video.stem}_{t}.png"
        subprocess.run([tp.FFMPEG, "-y", "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1", str(png)], check=True)
        return Image.open(png).convert("RGB")

    @unittest.skipUnless(FONT, "no bold font file found")
    def test_the_title_appears_on_the_top_bar_for_the_last_seconds_only(self):
        calls, real_run = [], tp._run

        def spy(command, *args, **kwargs):
            calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        with mock.patch.object(tp, "_run", side_effect=spy):
            out = self._build("title", cinema_windows=[(0.0, 9.0)], cinema_mode="instant", end_title="Une vie|exister ?", end_title_font=FONT,
                              end_title_seconds=3.0)
        plan = json_plan(self.dir / "work_title" / "plan.json")
        bar = tp.cinema_bar_height(self.WIDTH, self.HEIGHT)
        self.assertEqual((plan["end_title"]["start"], plan["end_title"]["seconds"], plan["end_title"]["position"], plan["end_title"]["bar_px"], plan["end_title"]["y"]),
                         (5.0, 3.0, "top", bar, 0))
        self.assertEqual(len(list((self.dir / "work_title" / "title").glob("title_*.png"))), 36)  # 3 s at 12 fps
        final = next(c for c in calls if c[-1].endswith("title.mp4"))
        graph = final[final.index("-filter_complex") + 1]
        self.assertIn("setpts=PTS+5.0000/TB", graph)
        self.assertIn("eof_action=pass", graph)
        self.assertTrue(any(part.endswith("title_%04d.png") for part in final))  # fed as an image sequence
        text_zone = (20, 20, self.WIDTH - 20, bar - 20)

        def brightest(t):
            return max(high for _, high in self._frame(out, t).crop(text_zone).getextrema())

        self.assertLess(brightest(2.0), 25)  # before: the bar is plain black
        self.assertGreater(brightest(7.0), 230)  # at the end: white letters on it

    @unittest.skipUnless(FONT, "no bold font file found")
    def test_every_subtitle_and_the_title_use_the_caption_font(self):
        calls, real_run = [], tp._run

        def spy(command, *args, **kwargs):
            calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        with mock.patch.object(tp, "_run", side_effect=spy):
            self._build("capfont", caption_font=FONT, end_title="Une vie|exister ?", cinema_windows=[(0.0, 9.0)], cinema_mode="instant", end_title_seconds=3.0)
        work = self.dir / "work_capfont"
        family = tc.font_family(FONT)
        self.assertIn(f"Style: Quote,{family},", (work / "captions.ass").read_text(encoding="utf-8"))  # every subtitle: the family of that file
        self.assertTrue((work / "fonts" / FONT.name).is_file())  # staged for libass
        final = next(c for c in calls if c[-1].endswith("capfont.mp4"))
        graph = final[final.index("-filter_complex") + 1]
        self.assertIn("fontsdir='" + str(work / "fonts").replace("\\", "/").replace(":", "\\:") + "'", graph)
        plan = json_plan(work / "plan.json")
        self.assertEqual(plan["caption_font"], str(FONT))
        self.assertEqual(plan["end_title"]["font"], str(FONT))  # the title took the caption font: no end_title_font was given
        other = self._build("origfont")
        self.assertIn(f"Style: Quote,{tp.CAPTION_FONT},", (self.dir / "work_origfont" / "captions.ass").read_text(encoding="utf-8"))
        self.assertIsNone(json_plan(self.dir / "work_origfont" / "plan.json")["caption_font"])
        self.assertTrue(other.is_file())

    def test_the_end_image_fades_in_on_the_bottom_bar_and_the_captions_stay_in_the_middle(self):
        import re

        png = self.dir / "symbol.png"
        Image.new("RGBA", (200, 200), (255, 255, 255, 255)).save(png)
        out = self._build("image", cinema_windows=[(0.0, 9.0)], cinema_mode="instant", end_image=png, end_image_fade=0.4, end_image_start=4.0,
                          end_image_position="bottom")
        plan = json_plan(self.dir / "work_image" / "plan.json")
        bar = tp.cinema_bar_height(self.WIDTH, self.HEIGHT)
        info = plan["end_image"]
        self.assertEqual((info["position"], info["start"], info["bar_y"], info["bar_px"]), ("bottom", 4.0, self.HEIGHT - bar, bar))
        self.assertEqual(info["height"], int(bar * 0.7) - int(bar * 0.7) % 2)
        zone = (self.WIDTH // 2 - 30, self.HEIGHT - 80, self.WIDTH // 2 + 30, self.HEIGHT - 30)  # low in the bar, under the rows where a lower-anchored caption sits
        brightest = lambda t: max(high for _, high in self._frame(out, t).crop(zone).getextrema())  # noqa: E731
        self.assertLess(brightest(2.0), 25)  # before: the bottom bar is black
        self.assertGreater(brightest(6.0), 230)  # after the fade: the symbol is there
        self.assertLess(brightest(4.1), 200)  # and it fades in, it does not pop
        captions = (self.dir / "work_image" / "captions.ass").read_text(encoding="utf-8")
        rows = [(float(m) * 60 + float(sec), int(y)) for m, sec, y in re.findall(r"Dialogue: 0,\d+:(\d+):([\d.]+),.*?\\pos\(\d+,(\d+)\)", captions)]
        after = [y for t_, y in rows if t_ >= 4.0]
        self.assertTrue(after and all(y < self.HEIGHT - bar for y in after))  # none on the bottom bar while the symbol is up

    def test_the_end_image_sits_in_the_centre_of_the_screen_by_default_and_the_captions_go_to_the_bars(self):
        import re

        png = self.dir / "emblem.png"
        Image.new("RGBA", (120, 150), (200, 20, 20, 255)).save(png)  # dark red: the light picture band behind it is (230, 230, 200)
        out = self._build("centre", cinema_windows=[(0.0, 9.0)], cinema_mode="instant", end_image=png, end_image_fade=0.2, end_image_start=2.0)
        info = json_plan(self.dir / "work_centre" / "plan.json")["end_image"]
        bar = tp.cinema_bar_height(self.WIDTH, self.HEIGHT)
        band = self.HEIGHT - 2 * bar
        self.assertEqual(info["position"], "center")
        self.assertEqual(info["height"], int(band * 0.92) - int(band * 0.92) % 2)  # most of the visible band
        self.assertAlmostEqual(info["y"] + info["height"] / 2, self.HEIGHT / 2, delta=1.5)  # centred on the screen, like the philosopher
        self.assertLess(info["y"] + info["height"], self.HEIGHT - bar)  # entirely inside the picture band: not on a bar
        self.assertGreater(info["y"], bar)
        middle = (self.WIDTH // 2, self.HEIGHT // 2)
        self.assertGreater(self._frame(out, 1.0).getpixel(middle)[1], 200)  # before: the picture band, light
        self.assertLess(self._frame(out, 5.0).getpixel(middle)[1], 60)  # after the fade: the emblem is in the middle of the screen
        captions = (self.dir / "work_centre" / "captions.ass").read_text(encoding="utf-8")
        rows = [(float(m) * 60 + float(sec), int(y)) for m, sec, y in re.findall(r"Dialogue: 0,\d+:(\d+):([\d.]+),.*?\\pos\(\d+,(\d+)\)", captions)]
        after = [y for t_, y in rows if t_ >= 2.0]
        self.assertTrue(after and all(abs(y - self.HEIGHT // 2) > 0.1 * self.HEIGHT for y in after), sorted(set(after)))  # on the bars, not over the emblem

    def test_the_end_image_is_validated_and_found_automatically_after_the_last_bars_come_in(self):
        png = self.dir / "symbol.png"
        Image.new("RGBA", (50, 50), (255, 255, 255, 255)).save(png)
        with self.assertRaises(FileNotFoundError):
            self._build("noimage", end_image=self.dir / "missing.png")
        with self.assertRaisesRegex(ValueError, "position"):
            self._build("badimgpos", end_image=png, end_image_position="left")
        self._build("autoimage", cinema_windows=[(2.0, 9.0)], cinema_mode="slide", cinema_seconds=1.0, end_image=png)
        self.assertEqual(json_plan(self.dir / "work_autoimage" / "plan.json")["end_image"]["start"], 3.0)  # window start 2.0 + the 1.0 s slide
        self._build("fallbackimage", end_image=png)  # no cinema window: the last END_IMAGE_FALLBACK_SECONDS (8) of an 8 s video
        self.assertEqual(json_plan(self.dir / "work_fallbackimage" / "plan.json")["end_image"]["start"], 0.0)
        self.assertIsNone(json_plan(self.dir / "work_fallbackimage" / "plan.json")["end_title"])
        parse = lambda *extra: tp.build_parser().parse_args(["--script", "s.txt", "--out", "o.mp4", "--work-dir", "w", *extra])  # noqa: E731
        default = parse()
        self.assertEqual((default.caption_font, default.end_image, default.end_image_position, default.end_image_height, default.end_image_start, default.end_image_fade),
                         (None, None, "center", None, None, 0.8))
        args = parse("--caption-font", "constanb.ttf", "--end-image", "b.png", "--end-image-position", "center", "--end-image-height", "400",
                     "--end-image-start", "58", "--end-image-fade", "1.2")
        self.assertEqual((args.caption_font, args.end_image, args.end_image_position, args.end_image_height, args.end_image_start, args.end_image_fade),
                         ("constanb.ttf", Path("b.png"), "center", 400, 58.0, 1.2))

    def test_the_title_options_are_validated_and_off_by_default(self):
        with self.assertRaisesRegex(ValueError, "end_title_font"):
            self._build("nofont", end_title="Hello")
        with self.assertRaisesRegex(ValueError, "effect"):
            self._build("badfx", end_title="Hello", end_title_font="x.ttf", end_title_effect="spin")
        with self.assertRaisesRegex(ValueError, "position"):
            self._build("badpos", end_title="Hello", end_title_font="x.ttf", end_title_position="left")
        self._build("notitle")
        self.assertIsNone(json_plan(self.dir / "work_notitle" / "plan.json")["end_title"])
        parse = lambda *extra: tp.build_parser().parse_args(["--script", "s.txt", "--out", "o.mp4", "--work-dir", "w", *extra])  # noqa: E731
        default = parse()
        self.assertEqual((default.end_title, default.end_title_font, default.end_title_seconds, default.end_title_effect, default.end_title_position),
                         (None, None, 5.0, "shear_wave", "top"))
        args = parse("--end-title", "A|B", "--end-title-font", "constanb.ttf", "--end-title-seconds", "4", "--end-title-effect", "plain", "--end-title-position", "bottom")
        self.assertEqual((args.end_title, args.end_title_font, args.end_title_seconds, args.end_title_effect, args.end_title_position),
                         ("A|B", "constanb.ttf", 4.0, "plain", "bottom"))


class CaptionFontTests(unittest.TestCase):
    def _ass(self, **kwargs):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "c.ass"
            tp.build_captions(["Une phrase de test."], [(0.0, 2.0)], out, width=1080, height=1920, **kwargs)
            return out.read_text(encoding="utf-8")

    def test_the_captions_use_the_given_family_and_the_original_font_by_default(self):
        self.assertIn(f"Style: Quote,{tp.CAPTION_FONT},", self._ass())
        self.assertIn("Style: Quote,Constantia,", self._ass(font="Constantia"))
        self.assertIn(",-1,0,0,0,", self._ass(font="Constantia"))  # still bold

    @unittest.skipUnless(FONT, "no bold font file found")
    def test_a_font_file_gives_its_family_name(self):
        self.assertEqual(tc.font_family(FONT), "DejaVu Sans" if "DejaVu" in FONT.name else tc.font_family(FONT))
        self.assertTrue(tc.font_family(FONT))

    def test_two_zones_leave_only_the_middle_band_to_the_captions(self):
        import re

        sentences = [f"phrase numéro {i} du test" for i in range(40)]
        windows = [(i * 2.0, i * 2.0 + 1.8) for i in range(40)]
        zones = [(20.0, 70.0, 0.0, 656 / 1920), (20.0, 70.0, 1 - 656 / 1920, 1.0)]
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "c.ass"
            tp.build_captions(sentences, windows, out, width=1080, height=1920, avoid=zones)
            rows = [(float(h) * 3600 + float(m) * 60 + float(sec), int(y)) for h, m, sec, y in
                    re.findall(r"Dialogue: 0,(\d+):(\d+):([\d.]+),.*?\\pos\(\d+,(\d+)\)", out.read_text(encoding="utf-8"))]
        inside = [y for t_, y in rows if 20.0 <= t_ <= 68.0]
        self.assertTrue(inside)
        self.assertTrue(all(abs(y - 0.5 * 1920) < 2 for y in inside), sorted(set(inside)))  # the middle anchor, over the picture band
        outside = [y for t_, y in rows if t_ < 18]
        self.assertGreater(len(set(outside)), 1)  # elsewhere the captions still move around


class BalanceGeneratorTests(unittest.TestCase):
    def test_the_emblem_is_a_transparent_symmetric_gothic_drawing_with_red_accents(self):
        import make_balance_png as mb

        image = mb.draw_balance(300)
        self.assertEqual((image.mode, image.height), ("RGBA", 300))
        rgba = np.asarray(image)
        alpha = rgba[..., 3]
        self.assertEqual(int(alpha[0, 0]), 0)  # transparent corners: it is an overlay
        self.assertGreater(int((alpha > 200).sum()), 5000)  # and there is a real drawing
        self.assertTrue(0.8 < image.width / image.height < 1.0)  # a lancet arch: taller than wide
        left, right = alpha[:, : image.width // 2], alpha[:, image.width - image.width // 2:][:, ::-1]
        both = (left > 128) & (right > 128)
        self.assertGreater(float(both.sum()) / float(np.maximum(left > 128, right > 128).sum()), 0.8)  # arch, column, beam: mirror images
        rgb = rgba[..., :3].astype(int)
        red = (rgb[..., 0] > 140) & (rgb[..., 1] < 60) & (rgb[..., 2] < 70) & (alpha > 200)
        self.assertGreater(int(red.sum()), 150)  # the crimson accents: the eye's iris, the heart, the gems
        bone = (rgb.min(axis=2) > 150) & (alpha > 200)
        self.assertGreater(int(bone.sum()), 4000)  # the bone-coloured engraving itself
        self.assertGreater(int(((alpha > 0) & (alpha < 200)).sum()), 3000)  # the soft dark halo that lets it read on any picture

    def test_the_shipped_png_is_the_generated_one(self):
        shipped = Path(__file__).resolve().parent.parent / "scripts" / "assets" / "balance.png"
        self.assertTrue(shipped.is_file())
        image = Image.open(shipped)
        self.assertEqual((image.mode, image.height), ("RGBA", 900))
        self.assertTrue(0.8 < image.width / image.height < 1.0)


def json_plan(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
