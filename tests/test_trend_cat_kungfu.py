"""scripts/trend_cat_kungfu.py: request payloads must validate against the real
schemas, hit/sound planning is deterministic, and the three recipes run end to end
against a fake API (synthetic clips instead of GPU renders)."""
import subprocess
import sys
import tempfile
import unittest
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import trend_cat_kungfu as tck  # noqa: E402

from app.schemas import SequenceRequest  # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
IMG_A, IMG_B = "img_" + "a" * 32, "img_" + "b" * 32


def _sequence(*shots):
    return SequenceRequest.model_validate({"id": "trend_test", "shots": list(shots)})


class PayloadTests(unittest.TestCase):
    def test_morph_shot_is_a_valid_flf2v_shot(self):
        request = _sequence(tck.morph_shot(IMG_A, IMG_B))
        shot = request.shots[0]
        self.assertEqual((shot.width, shot.height, shot.frames, shot.fps), (512, 512, 81, 16))
        self.assertEqual(shot.end_image.image_id, IMG_B)

    def test_animate_shots_validate_for_every_driving_source(self):
        for driving in (
            tck.driving_source(path="monk.mp4"),
            tck.driving_source(url="https://example.com/monk.mp4"),
            tck.driving_source(job_id="fight_a_1"),
        ):
            for mode in ("animate_move", "animate_mix"):
                with self.subTest(driving=driving, mode=mode):
                    shot = _sequence(tck.animate_shot(IMG_A, driving, mode=mode)).shots[0]
                    self.assertEqual((shot.width, shot.height), (512, 512))

    def test_subject_point_and_pad_for_the_fight_passes(self):
        shot = _sequence(
            tck.animate_shot(IMG_A, tck.driving_source(path="duo.mp4"), mode="animate_mix", driving_fit="pad", subject_point=(0.66, 0.53))
        ).shots[0]
        self.assertEqual(shot.driving_fit, "pad")
        self.assertEqual(shot.subject_point, (0.66, 0.53))

    def test_exactly_one_driving_source(self):
        with self.assertRaises(ValueError):
            tck.driving_source()
        with self.assertRaises(ValueError):
            tck.driving_source(path="a.mp4", url="https://example.com/a.mp4")

    def test_image_request_matches_the_images_route(self):
        from app.schemas import ImageGenerationRequest

        request = ImageGenerationRequest.model_validate(tck.image_request(tck.hands_cat_prompt("dojo"), 2))
        self.assertEqual((request.engine, request.steps, request.width), ("krea2", 8, 1024))

    def test_prompts_keep_the_findings_that_made_them_work(self):
        self.assertIn("realistic human hands with fingers and thumbs", tck.hands_cat_prompt())
        self.assertIn("anthropomorphic", tck.hands_cat_prompt())
        self.assertNotIn("anthropomorphic", tck.plain_cat_prompt())
        self.assertIn("japanese dojo", tck.hands_cat_prompt("dojo"))
        # the pair differs only in the cat description and the tail: same scene text
        self.assertIn(tck.SCENES["couch"][0], tck.plain_cat_prompt("couch"))
        self.assertIn(tck.SCENES["couch"][0], tck.hands_cat_prompt("couch"))


class PlanningTests(unittest.TestCase):
    def test_strongest_hits_are_spaced_and_skip_the_first_frames(self):
        energy = [30.0] + [1.0] * 59
        for frame, value in ((10, 9.0), (11, 8.5), (30, 7.0), (50, 6.0)):
            energy[frame] = value
        hits = tck.strongest_hits(energy, 16, count=3)
        self.assertEqual(hits, [10 / 16, 30 / 16, 50 / 16])  # 11 is too close to 10; frame 0 is ignored

    def test_finisher_gets_kick_scream_and_strike(self):
        events = tck.plan_sfx([0.6, 1.4, 2.5], duration=3.0)
        names = [name for name, _, _ in events]
        self.assertEqual(names.count("kick"), 1)
        self.assertEqual(names.count("scream"), 1)
        self.assertEqual(names.count("strike_effort"), 1)
        kick_start = next(start for name, start, _ in events if name == "kick")
        self.assertAlmostEqual(kick_start, 2.5 - tck.SFX["kick"]["peak"])
        scream_start = next(start for name, start, _ in events if name == "scream")
        self.assertLess(scream_start, kick_start, "the cry leads the kick")

    def test_a_single_hit_is_a_plain_impact(self):
        names = [name for name, _, _ in tck.plan_sfx([1.0], duration=3.0)]
        self.assertNotIn("kick", names)

    def test_no_event_starts_before_zero_or_after_the_clip(self):
        for start in (tck.plan_sfx([0.05, 0.9], duration=1.0)):
            self.assertGreaterEqual(start[1], 0.0)
            self.assertLess(start[1], 1.0)


def _make_clip(path: Path, *, frames: int, size: int = 64, fps: int = 16, source: str = "testsrc") -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"{source}=size={size}x{size}:rate={fps}",
            "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def _make_sfx_dir(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    for name in tck.SFX:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.6",
                "-c:a", "libmp3lame", str(directory / f"{name}.mp3"),
            ]
        )
        if result.returncode != 0:
            return False
    return True


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
class FfmpegHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_motion_energy_has_one_value_per_frame(self):
        clip = self.dir / "a.mp4"
        _make_clip(clip, frames=20)
        self.assertAlmostEqual(len(tck.motion_energy(clip)), 20, delta=1)

    def test_join_and_last_frame(self):
        first, second = self.dir / "1.mp4", self.dir / "2.mp4"
        _make_clip(first, frames=32)
        _make_clip(second, frames=32, source="mandelbrot")
        joined = tck.join_clips([first, second], self.dir / "joined.mp4", xfade=0.25)
        self.assertAlmostEqual(tck.probe_duration(joined), 2.0 + 2.0 - 0.25, delta=0.15)
        png = tck.last_frame(first, self.dir / "last.png")
        self.assertGreater(png.stat().st_size, 100)

    def test_stack_and_cut(self):
        a, b = self.dir / "a.mp4", self.dir / "b.mp4"
        _make_clip(a, frames=16)
        _make_clip(b, frames=16)
        stacked = tck.stack_side_by_side([a, b], self.dir / "s.mp4")
        info = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(stacked)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(info, "128,64")
        cut = tck.cut_to(stacked, self.dir / "c.mp4", 0.5)
        self.assertAlmostEqual(tck.probe_duration(cut), 0.5, delta=0.1)

    def test_mix_sfx_adds_an_audio_track(self):
        sfx_dir = self.dir / "sfx"
        if not _make_sfx_dir(sfx_dir):
            self.skipTest("ffmpeg has no mp3 encoder")
        clip = self.dir / "fight.mp4"
        _make_clip(clip, frames=48)
        out = tck.mix_sfx(clip, self.dir / "mixed.mp4", tck.plan_sfx([0.5, 1.2, 2.2], 3.0), sfx_dir, upscale=128)
        streams = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width", "-of", "csv=p=0", str(out)],
            check=True, capture_output=True, text=True,
        ).stdout
        self.assertIn("audio", streams)
        self.assertIn("128", streams)
        silent = tck.mix_sfx(clip, self.dir / "silent.mp4", [], sfx_dir, upscale=None)
        self.assertTrue(silent.is_file())


class FakeApi(tck.Api):
    """No network, no GPU: renders are tiny synthetic clips, ids are counters."""

    def __init__(self, directory: Path):
        super().__init__("http://unused", "")
        self.directory, self.counter, self.jobs, self.images = directory, 0, [], []

    def image(self, prompt, seed, size=1024):
        self.counter += 1
        self.images.append(prompt)
        return f"img_{self.counter:032x}"

    def upload_image(self, path):
        self.counter += 1
        assert path.is_file()
        return f"img_{self.counter:032x}"

    def job(self, job_id, shots, out):
        SequenceRequest.model_validate({"id": job_id, "shots": shots})  # whatever we send must be valid
        self.jobs.append((job_id, shots))
        _make_clip(out, frames=81 if shots[0]["mode"] == "t+i(keyframe)2v" else 40)
        return out


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.api = FakeApi(self.dir)

    def test_transform_chains_morph_then_animate_from_the_last_frame(self):
        final = tck.run_transform(self.api, self.dir / "w", tck.driving_source(path="monk.mp4"))
        self.assertTrue(final.is_file())
        morph, animate = self.api.jobs[0][1][0], self.api.jobs[1][1][0]
        self.assertEqual(morph["mode"], "t+i(keyframe)2v")
        self.assertEqual(animate["mode"], "animate_move")
        # the morph pair uses two different prompts but the same seed, and the animate
        # reference is the uploaded last frame (a third image id), not either still
        self.assertEqual(len(self.api.images), 2)
        self.assertNotIn(animate["start_image"]["image_id"], (morph["start_image"]["image_id"], morph["end_image"]["image_id"]))
        self.assertTrue((self.dir / "w" / "morph_last.png").is_file())

    def test_background_change_runs_one_animate_per_scene_with_the_same_pose_clip(self):
        driving = tck.driving_source(path="monk.mp4")
        out = tck.run_background_change(self.api, self.dir / "w", driving, ["dojo", "roof", "forest"])
        self.assertTrue(out.is_file())
        shots = [shots[0] for _, shots in self.api.jobs]
        self.assertEqual(len(shots), 3)
        self.assertTrue(all(shot["driving_video"] == driving for shot in shots))
        self.assertTrue(any("japanese dojo" in prompt for prompt in self.api.images))

    def test_fight_replays_pass_a_for_pass_b(self):
        ref = self.dir / "ref.png"
        _make_clip(self.dir / "ref.mp4", frames=1)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(self.dir / "ref.mp4"), "-frames:v", "1", str(ref)], check=True)
        out = tck.run_fight(
            self.api, self.dir / "w", tck.driving_source(path="duo.mp4"), ref, ref, (0.66, 0.53), (0.19, 0.58),
            driving_size=(1280, 720), max_seconds=1.0,
        )
        self.assertTrue(out.is_file())
        (pass_a_id, (shot_a,)), (_, (shot_b,)) = self.api.jobs
        # pass A: green on the first fighter, red on the second, both in the original frame
        self.assertEqual((shot_a["driving_fit"], shot_a["subject_point"]), ("pad", [0.66, 0.53]))
        self.assertEqual(shot_a["exclude_points"], [[0.19, 0.58]])
        # pass B replays pass A, so its points are mapped into the letterboxed square: x stays,
        # y is squeezed toward the middle by 720/1280; green on the second fighter, red on the first cat
        self.assertEqual(shot_b["driving_video"], {"job_id": pass_a_id})
        self.assertAlmostEqual(shot_b["subject_point"][0], 0.19)
        self.assertAlmostEqual(shot_b["subject_point"][1], 0.5 + 0.08 * 720 / 1280)
        self.assertAlmostEqual(shot_b["exclude_points"][0][0], 0.66)
        self.assertAlmostEqual(shot_b["exclude_points"][0][1], 0.5 + 0.03 * 720 / 1280)

    def test_padded_square_mapping_matches_the_apps(self):
        from app.driving_video import to_square_coords

        for point in ((0.1, 0.9), (0.5, 0.5), (0.95, 0.05)):
            for size in ((1280, 720), (720, 1280)):
                self.assertEqual(
                    tuple(round(v, 9) for v in tck.to_padded_square(point, *size)),
                    tuple(round(v, 9) for v in to_square_coords(*point, *size, "pad")),
                )

    def test_animate_shot_carries_green_and_red_points_into_a_valid_request(self):
        shot = tck.animate_shot(
            IMG_A, tck.driving_source(path="duo.mp4"), mode="animate_mix", driving_fit="pad",
            subject_point=(0.7, 0.5), exclude_points=[(0.2, 0.5)],
        )
        validated = _sequence(shot).shots[0]
        self.assertEqual(validated.exclude_points, [(0.2, 0.5)])


if __name__ == "__main__":
    unittest.main()
