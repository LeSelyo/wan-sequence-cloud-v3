"""render_shot() for Animate shots with ComfyUI mocked out: checks what the app
hands the template (square clip at the shot fps, SAM2 seed only for Mix, Move
without mask) and that the returned video is cut to the driving motion."""
import asyncio
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app import orchestrator
from app.driving_video import probe_video
from app.schemas import InputImage, InputVideo, Mode, Shot

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
FAKE_SEED = {"points_store": "ps", "coordinates": "co", "neg_coordinates": "ne"}


def _make_clip(path: Path, *, size: str, rate: int, seconds: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={seconds}",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
class AnimateRenderShotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.fake_settings = SimpleNamespace(
            driving_video_dir=root / "videos",
            inputs_dir=root / "inputs",
            comfy_input_dir=root / "comfy_in",
            comfy_output_dir=root / "comfy_out",
            detection_dir=root / "detection",
        )
        for directory in (
            self.fake_settings.driving_video_dir,
            self.fake_settings.inputs_dir,
            self.fake_settings.comfy_output_dir,
        ):
            directory.mkdir(parents=True)
        # 1 s of 25 fps wide footage -> 25 source frames -> 16 frames at 16 fps
        _make_clip(self.fake_settings.driving_video_dir / "performer.mp4", size="160x90", rate=25, seconds=1)
        _make_clip(self.fake_settings.comfy_output_dir / "extended.mp4", size="64x64", rate=16, seconds=3)
        (self.fake_settings.inputs_dir / "character.png").write_bytes(b"png")
        self.job_dir = root / "job"
        self.job_dir.mkdir()
        self.bound: dict = {}

    def _render(self, **shot_overrides):
        values = {
            "id": "s1",
            "mode": Mode.ANIMATE_MIX,
            "start_image": InputImage(path="character.png"),
            "driving_video": InputVideo(path="performer.mp4"),
            "fps": 16,
        }
        values.update(shot_overrides)
        shot = Shot(**values)

        def fake_bind(template, bound_values):
            self.bound = dict(bound_values)
            return {"1": {"class_type": "X", "inputs": {}}}, {}

        async def fake_queue(prompt):
            return {"outputs": {"243": {"videos": [{"filename": "extended.mp4", "subfolder": ""}]}}}

        with mock.patch.object(orchestrator, "settings", self.fake_settings), mock.patch.object(
            orchestrator, "bind_workflow", fake_bind
        ), mock.patch.object(orchestrator, "queue_and_wait", fake_queue), mock.patch.object(
            orchestrator, "compute_seed_point", return_value=FAKE_SEED
        ) as auto_seed:
            result = asyncio.run(orchestrator.render_shot(shot, self.job_dir))
        return result, auto_seed

    def test_mix_stages_a_square_clip_at_the_shot_fps_and_seeds_sam2(self):
        result, auto_seed = self._render()
        staged = self.fake_settings.comfy_input_dir / self.bound["driving_video"]
        info = probe_video(staged)
        self.assertEqual((info["width"], info["height"]), (640, 640))
        self.assertAlmostEqual(info["fps"], 16, delta=0.01)
        auto_seed.assert_called_once()
        self.assertEqual(self.bound["sam2_points_store"], "ps")
        self.assertTrue(self.bound["keep_background"])
        self.assertTrue(result.is_file())

    def test_move_has_no_seed_point_and_drops_the_background(self):
        _, auto_seed = self._render(mode=Mode.ANIMATE_MOVE)
        auto_seed.assert_not_called()
        self.assertFalse(self.bound["keep_background"])
        self.assertNotIn("sam2_points_store", self.bound)

    def test_extended_output_is_cut_to_the_driving_motion(self):
        result, _ = self._render()
        driving_frames = probe_video(self.fake_settings.comfy_input_dir / self.bound["driving_video"])["frames"]
        self.assertAlmostEqual(driving_frames, 16, delta=1)
        self.assertEqual(probe_video(result)["frames"], driving_frames)
        self.assertFalse(result.with_name(result.stem + "_trim.mp4").exists())

    def test_manual_subject_point_replaces_the_automatic_pick(self):
        # pad keeps the 16:9 frame inside the square: x stays, y is squeezed toward the middle
        _, auto_seed = self._render(subject_point=(0.25, 0.5), driving_fit="pad")
        auto_seed.assert_not_called()
        coordinates = json.loads(self.bound["sam2_coordinates"])
        self.assertEqual(coordinates, [{"x": 160, "y": 320}])

    def test_animal_pose_uses_the_animal_detector_for_the_seed_point(self):
        with mock.patch.object(orchestrator, "compute_animal_seed_point", return_value=FAKE_SEED) as animal:
            _, human = self._render(pose_source="animal")
        animal.assert_called_once()
        human.assert_not_called()
        self.assertEqual(self.bound["pose_source"], "animal")
        self.assertEqual(self.bound["sam2_points_store"], "ps")

    def test_human_pose_is_the_default_and_uses_the_person_detector(self):
        self._render()
        self.assertEqual(self.bound["pose_source"], "human")

    def test_red_points_are_mapped_sent_and_switch_the_negative_link_on(self):
        self._render(subject_point=(0.75, 0.5), exclude_points=[(0.25, 0.5)], driving_fit="pad")
        self.assertTrue(self.bound["use_negative_points"])
        self.assertEqual(json.loads(self.bound["sam2_coordinates"]), [{"x": 480, "y": 320}])
        self.assertEqual(json.loads(self.bound["sam2_neg_coordinates"]), [{"x": 160, "y": 320}])

    def test_no_red_points_means_the_validated_graph(self):
        self._render()
        self.assertFalse(self.bound["use_negative_points"])

    def test_red_point_cropped_out_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "falls outside"):
            self._render(subject_point=(0.5, 0.5), exclude_points=[(0.01, 0.5)], driving_fit="crop")

    def test_subject_cropped_out_is_an_error_not_a_silent_miss(self):
        with self.assertRaisesRegex(ValueError, "falls outside"):
            self._render(subject_point=(0.02, 0.5), driving_fit="crop")


if __name__ == "__main__":
    unittest.main()
