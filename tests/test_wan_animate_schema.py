import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.schemas import InputImage, InputVideo, Mode, Shot


REPO_ROOT = Path(__file__).resolve().parent.parent


def _shot(**overrides):
    defaults = {"id": "s1", "mode": Mode.ANIMATE_MIX}
    defaults.update(overrides)
    return Shot(**defaults)


class InputVideoTests(unittest.TestCase):
    def test_exactly_one_source_required(self):
        with self.assertRaisesRegex(ValidationError, "exactly one of"):
            InputVideo()
        with self.assertRaisesRegex(ValidationError, "exactly one of"):
            InputVideo(path="a.mp4", url="https://example.com/a.mp4")

    def test_path_traversal_rejected(self):
        with self.assertRaisesRegex(ValidationError, "relative to the mounted"):
            InputVideo(path="../escape.mp4")
        with self.assertRaisesRegex(ValidationError, "relative to the mounted"):
            InputVideo(path="/etc/passwd")

    def test_valid_path_accepted(self):
        InputVideo(path="clips/performer.mp4")


class AnimateShotValidationTests(unittest.TestCase):
    START = InputImage(path="character.png")
    DRIVING = InputVideo(path="performer.mp4")

    def test_mix_requires_driving_video(self):
        with self.assertRaisesRegex(ValidationError, "requires driving_video"):
            _shot(mode=Mode.ANIMATE_MIX, start_image=self.START)

    def test_move_requires_driving_video(self):
        with self.assertRaisesRegex(ValidationError, "requires driving_video"):
            _shot(mode=Mode.ANIMATE_MOVE, start_image=self.START)

    def test_animate_requires_start_image(self):
        with self.assertRaisesRegex(ValidationError, "requires start_image"):
            _shot(mode=Mode.ANIMATE_MIX, driving_video=self.DRIVING)

    def test_valid_mix_shot(self):
        shot = _shot(mode=Mode.ANIMATE_MIX, start_image=self.START, driving_video=self.DRIVING)
        self.assertEqual(shot.mode, Mode.ANIMATE_MIX)

    def test_valid_move_shot(self):
        shot = _shot(mode=Mode.ANIMATE_MOVE, start_image=self.START, driving_video=self.DRIVING)
        self.assertEqual(shot.mode, Mode.ANIMATE_MOVE)

    def test_driving_video_rejected_outside_animate_modes(self):
        with self.assertRaisesRegex(ValidationError, "only accepted for animate_mix/animate_move"):
            _shot(mode=Mode.I2V, start_image=self.START, driving_video=self.DRIVING)

    def test_turbo_mode_rejected_for_animate(self):
        with self.assertRaisesRegex(ValidationError, "supported only for T2V/I2V"):
            _shot(
                mode=Mode.ANIMATE_MIX,
                start_image=self.START,
                driving_video=self.DRIVING,
                turbo_mode=True,
            )


class WanAnimateCatalogTests(unittest.TestCase):
    """Pure JSON-shape checks: no network, no GPU, no live ComfyUI needed."""

    def setUp(self):
        self.catalog = json.loads((REPO_ROOT / "config/base_models.json").read_text(encoding="utf-8"))

    def test_profile_lists_expected_items(self):
        profile = self.catalog["profiles"]["wan22-animate"]
        self.assertEqual(
            set(profile),
            {
                "wan22_animate_diffusion",
                "umt5_xxl",
                "wan21_vae",
                "clip_vision_h",
                "wan22_animate_lightx",
                "wan22_animate_relight_lora",
            },
        )

    def test_capability_maps_to_animate(self):
        self.assertEqual(self.catalog["profile_capabilities"]["wan22-animate"], ["animate"])

    def test_new_items_have_verified_sizes_and_urls(self):
        items = self.catalog["items"]
        expected_sizes = {
            "wan22_animate_diffusion": 18401760586,
            "wan22_animate_lightx": 738005744,
            "wan22_animate_relight_lora": 1436672440,
            "clip_vision_h": 1264219396,
        }
        for item_id, size in expected_sizes.items():
            entry = items[item_id]
            self.assertEqual(entry["size_bytes"], size, item_id)
            self.assertTrue(entry["url"].startswith("https://huggingface.co/"), item_id)
            self.assertEqual(entry["family"], "wan22_animate", item_id)

    def test_lora_items_are_not_silently_excluded_from_readiness(self):
        """readiness.check_models() used to drop every 'loras/...' item for ANY
        profile, including non-turbo ones; wan22-animate's turbo/relight LoRAs
        are core requirements, not an optional turbo add-on, so they must stay."""
        from app.readiness import check_models
        from app.settings import Settings

        settings = Settings(
            data_root=REPO_ROOT / "does-not-exist",
            port=8000,
            comfyui_host="127.0.0.1",
            comfyui_port=8188,
            gpu_concurrency=1,
            app_test_mode=True,
            workflow_dir=REPO_ROOT / "does-not-exist",
            ui_workflow_dir=REPO_ROOT / "does-not-exist",
            readiness_profile="wan22-animate",
            max_shots_per_job=8,
            max_pending_jobs=2,
            max_total_frames=968,
            max_output_disk_gb=50,
            min_free_disk_gb=10,
            max_job_cost_units=8,
            max_input_image_mb=50,
            max_input_image_pixels=16777216,
        )
        _, details = check_models(settings, "wan22-animate")
        self.assertEqual(
            set(details),
            {
                "wan22_animate_diffusion",
                "umt5_xxl",
                "wan21_vae",
                "clip_vision_h",
                "wan22_animate_lightx",
                "wan22_animate_relight_lora",
            },
        )


if __name__ == "__main__":
    unittest.main()
