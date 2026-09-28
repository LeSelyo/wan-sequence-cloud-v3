import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.schemas import InputImage, Mode, Shot, TimedKeyframe

REPO_ROOT = Path(__file__).resolve().parent.parent


def _keyframe(frame: int) -> TimedKeyframe:
    return TimedKeyframe(image=InputImage(path=f"kf{frame}.png"), frame=frame)


def _vace_shot(**overrides):
    defaults = {
        "id": "s1",
        "mode": Mode.VACE,
        "prompt": "a rocket lifting off",
        "frames": 81,
        "keyframes": [_keyframe(0), _keyframe(80)],
    }
    defaults.update(overrides)
    return Shot(**defaults)


class VaceShotValidationTests(unittest.TestCase):
    def test_valid_shot(self):
        shot = _vace_shot()
        self.assertEqual(shot.mode, Mode.VACE)
        self.assertEqual(len(shot.keyframes), 2)

    def test_requires_prompt(self):
        with self.assertRaisesRegex(ValidationError, "requires a prompt"):
            _vace_shot(prompt="")

    def test_requires_at_least_two_keyframes(self):
        with self.assertRaisesRegex(ValidationError, "at least 2 keyframes"):
            _vace_shot(keyframes=[_keyframe(0)])

    def test_rejects_duplicate_frame_indices(self):
        with self.assertRaisesRegex(ValidationError, "distinct frame indices"):
            _vace_shot(keyframes=[_keyframe(0), _keyframe(0)])

    def test_rejects_frame_index_out_of_range(self):
        with self.assertRaisesRegex(ValidationError, "out of range"):
            _vace_shot(frames=81, keyframes=[_keyframe(0), _keyframe(81)])

    def test_keyframes_rejected_outside_vace_mode(self):
        with self.assertRaisesRegex(ValidationError, "only accepted for vace shots"):
            Shot(id="s1", mode=Mode.T2V, prompt="x", keyframes=[_keyframe(0), _keyframe(5)])

    def test_turbo_mode_rejected_for_vace(self):
        with self.assertRaisesRegex(ValidationError, "supported only for T2V/I2V"):
            _vace_shot(turbo_mode=True)


class VaceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((REPO_ROOT / "config/base_models.json").read_text(encoding="utf-8"))

    def test_profile_and_capability(self):
        self.assertEqual(
            set(self.catalog["profiles"]["wan22-vace"]),
            {"wan22_vace_high", "wan22_vace_low", "umt5_xxl", "wan21_vae"},
        )
        self.assertEqual(self.catalog["profile_capabilities"]["wan22-vace"], ["vace"])

    def test_diffusion_models_have_verified_sizes(self):
        for item_id in ("wan22_vace_high", "wan22_vace_low"):
            entry = self.catalog["items"][item_id]
            self.assertEqual(entry["size_bytes"], 17346056104, item_id)
            self.assertTrue(entry["sha256"])
            self.assertEqual(entry["family"], "wan22_vace", item_id)
            self.assertIn("/resolve/", entry["url"])
            self.assertNotIn("/resolve/main/", entry["url"], "must pin an exact commit, not a floating branch")


class VaceWorkflowBindingTests(unittest.TestCase):
    """The hand-authored workflows/wan22_vace.api.json resolves with no dangling
    node references. This does NOT prove ComfyUI accepts it -- only a live GPU
    pass can -- but it catches transcription mistakes in the JSON pointers."""

    def test_bindings_resolve_with_no_missing_references(self):
        import os

        from app.comfy import bind_workflow

        with tempfile.TemporaryDirectory() as temporary:
            shutil.copy(REPO_ROOT / "workflows/wan22_vace.api.json", Path(temporary) / "wan22_vace.api.json")
            previous = os.environ.get("WORKFLOW_DIR")
            os.environ["WORKFLOW_DIR"] = temporary
            try:
                prompt, bindings = bind_workflow(
                    "wan22_vace",
                    {
                        "positive_prompt": "test",
                        "negative_prompt": "",
                        "seed": 1,
                        "width": 480,
                        "height": 832,
                        "frames": 81,
                        "fps": 16,
                        "steps": 20,
                        "cfg": 5.0,
                        "turbo_mode": False,
                        "control_video": "cv.mp4",
                        "control_masks": "cm.mp4",
                        "output_prefix": "job/shot_video",
                    },
                )
            finally:
                if previous is None:
                    os.environ.pop("WORKFLOW_DIR", None)
                else:
                    os.environ["WORKFLOW_DIR"] = previous

        for node_id, node in prompt.items():
            for field, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], prompt, f"{node_id}.{field} references missing node {value[0]}")
        self.assertEqual(prompt["14"]["inputs"]["width"], 480)
        self.assertEqual(prompt["14"]["inputs"]["length"], 81)
        self.assertEqual(prompt["9"]["inputs"]["file"], "cv.mp4")
        self.assertEqual(prompt["11"]["inputs"]["file"], "cm.mp4")
        self.assertEqual(
            set(bindings),
            {
                "positive_prompt", "negative_prompt", "width", "height", "frames", "fps",
                "output_prefix", "control_video", "control_masks", "steps", "cfg", "seed",
                "high_model_target", "low_model_target",
            },
        )


class VaceControlAssetBuilderTests(unittest.TestCase):
    """Exercises build_vace_control_assets() with a real (static) ffmpeg -- fully
    offline, no GPU, no ComfyUI needed. Confirms keyframes land at the requested
    frame index and the mask polarity matches what WanVaceToVideo expects
    (white/1 = generate, black/0 = keep the keyframe as given)."""

    def test_control_video_and_mask_are_correct(self):
        import static_ffmpeg

        static_ffmpeg.add_paths()
        import asyncio

        from PIL import Image

        from app.orchestrator import build_vace_control_assets
        from app.schemas import InputImage, Shot, TimedKeyframe

        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary) / "job"
            job_dir.mkdir()
            inputs_dir = Path(temporary) / "inputs"
            inputs_dir.mkdir()

            red = inputs_dir / "red.png"
            blue = inputs_dir / "blue.png"
            Image.new("RGB", (64, 64), (200, 0, 0)).save(red)
            Image.new("RGB", (64, 64), (0, 0, 200)).save(blue)

            import app.orchestrator as orchestrator_module

            original_settings = orchestrator_module.settings
            try:
                orchestrator_module.settings = type(
                    "S", (), {"inputs_dir": inputs_dir, "image_outputs_dir": Path(temporary) / "img_out"}
                )()
                shot = Shot(
                    id="s1",
                    mode=Mode.VACE,
                    prompt="x",
                    width=256,
                    height=256,
                    frames=17,
                    fps=8,
                    keyframes=[
                        TimedKeyframe(image=InputImage(path="red.png"), frame=0),
                        TimedKeyframe(image=InputImage(path="blue.png"), frame=16),
                    ],
                )
                control_video, control_masks = asyncio.run(
                    build_vace_control_assets(shot, job_dir)
                )
            finally:
                orchestrator_module.settings = original_settings

            def frame_color(video_path: Path, index: int) -> tuple[int, int, int]:
                out = job_dir / f"probe_{video_path.stem}_{index}.png"
                import subprocess

                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
                     "-vf", f"select=eq(n\\,{index})", "-vsync", "0", "-frames:v", "1", str(out)],
                    check=True,
                )
                with Image.open(out) as image:
                    return image.convert("RGB").getpixel((32, 32))

            def close(actual, expected, tolerance=4):
                return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))

            # H.264/YUV420 is lossy: allow a few levels of rounding drift, real
            # colors/polarity must still be unambiguous.
            self.assertTrue(close(frame_color(control_video, 0), (200, 0, 0)))
            self.assertTrue(close(frame_color(control_video, 16), (0, 0, 200)))
            mid = frame_color(control_video, 8)
            self.assertTrue(100 < mid[0] < 155 and mid[0] == mid[1] == mid[2], mid)  # mid-grey filler

            self.assertTrue(close(frame_color(control_masks, 0), (0, 0, 0)))  # keyframe: keep as given
            self.assertTrue(close(frame_color(control_masks, 16), (0, 0, 0)))
            self.assertTrue(close(frame_color(control_masks, 8), (255, 255, 255)))  # filler: generate


if __name__ == "__main__":
    unittest.main()
