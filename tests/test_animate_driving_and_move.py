"""Animate findings of 2026-10-02 turned into code: square driving clips at the
output fps, a manual SAM2 subject point, Move mode (reference-image background)
and trimming of the frozen tail. No GPU, no ComfyUI; the ffmpeg-backed tests skip
themselves where ffmpeg/ffprobe are not installed."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from app.comfy import bind_workflow
from app.driving_video import (
    normalize_driving_video,
    probe_video,
    square_filter,
    to_square_coords,
    trim_video,
)
from app.sam2_seed_point import manual_seed_point, seed_point_values
from app.schemas import InputImage, InputVideo, Mode, Shot

REPO_ROOT = Path(__file__).resolve().parent.parent
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _animate(**overrides):
    values = {
        "id": "s1",
        "mode": Mode.ANIMATE_MIX,
        "start_image": InputImage(path="character.png"),
        "driving_video": InputVideo(path="performer.mp4"),
    }
    values.update(overrides)
    return Shot(**values)


class AnimateShotSizeTests(unittest.TestCase):
    def test_default_size_is_the_validated_512_square(self):
        shot = _animate()
        self.assertEqual((shot.width, shot.height), (512, 512))

    def test_one_explicit_side_sets_the_other(self):
        self.assertEqual((_animate(width=640).width, _animate(width=640).height), (640, 640))
        self.assertEqual((_animate(height=576).width, _animate(height=576).height), (576, 576))

    def test_non_square_size_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "must be square"):
            _animate(width=832, height=480)

    def test_non_animate_shots_keep_their_own_default_size(self):
        shot = Shot(id="t", mode=Mode.T2V, prompt="x")
        self.assertEqual((shot.width, shot.height), (832, 480))


class SubjectPointSchemaTests(unittest.TestCase):
    def test_accepted_for_mix(self):
        self.assertEqual(_animate(subject_point=(0.2, 0.7)).subject_point, (0.2, 0.7))

    def test_rejected_for_move_which_has_no_mask(self):
        with self.assertRaisesRegex(ValidationError, "only accepted for animate_mix"):
            _animate(mode=Mode.ANIMATE_MOVE, subject_point=(0.5, 0.5))

    def test_must_be_normalized(self):
        with self.assertRaisesRegex(ValidationError, "between 0 and 1"):
            _animate(subject_point=(1.2, 0.5))

    def test_driving_fit_defaults_to_the_templates_own_crop(self):
        self.assertEqual(_animate().driving_fit, "crop")
        self.assertEqual(_animate(driving_fit="pad").driving_fit, "pad")


class ExcludePointSchemaTests(unittest.TestCase):
    def test_red_points_accepted_for_mix(self):
        shot = _animate(subject_point=(0.7, 0.5), exclude_points=[(0.2, 0.5), (0.9, 0.1)])
        self.assertEqual(shot.exclude_points, [(0.2, 0.5), (0.9, 0.1)])

    def test_default_is_no_red_points(self):
        self.assertEqual(_animate().exclude_points, [])

    def test_rejected_for_move_and_out_of_range(self):
        with self.assertRaisesRegex(ValidationError, "only accepted for animate_mix"):
            _animate(mode=Mode.ANIMATE_MOVE, exclude_points=[(0.2, 0.5)])
        with self.assertRaisesRegex(ValidationError, "between 0 and 1"):
            _animate(exclude_points=[(0.2, 1.5)])


class ManualSeedPointTests(unittest.TestCase):
    def test_red_points_are_the_negative_list_and_default_stays_the_corner(self):
        values = manual_seed_point((0.75, 0.5), [(0.25, 0.5), (0.1, 0.9)])
        self.assertEqual(json.loads(values["coordinates"]), [{"x": 480, "y": 320}])
        self.assertEqual(json.loads(values["neg_coordinates"]), [{"x": 160, "y": 320}, {"x": 64, "y": 576}])
        self.assertEqual(json.loads(values["points_store"])["negative"], json.loads(values["neg_coordinates"]))
        default = seed_point_values(100, 100)
        self.assertEqual(json.loads(default["neg_coordinates"]), [{"x": 5, "y": 5}])

    def test_red_points_must_be_normalized(self):
        with self.assertRaises(ValueError):
            manual_seed_point((0.5, 0.5), [(0.5, 2.0)])

    def test_maps_normalized_point_onto_the_640_canvas(self):
        values = manual_seed_point((0.5, 0.25))
        self.assertEqual(json.loads(values["coordinates"]), [{"x": 320, "y": 160}])
        self.assertEqual(json.loads(values["neg_coordinates"]), [{"x": 5, "y": 5}])
        store = json.loads(values["points_store"])
        self.assertEqual(store["positive"], [{"x": 320, "y": 160}])

    def test_edges_stay_inside_the_canvas(self):
        self.assertEqual(json.loads(manual_seed_point((1.0, 1.0))["coordinates"]), [{"x": 639, "y": 639}])

    def test_out_of_range_is_an_error(self):
        with self.assertRaises(ValueError):
            manual_seed_point((-0.1, 0.5))


class SquareCoordinateTests(unittest.TestCase):
    def test_crop_of_a_wide_clip_drops_the_sides(self):
        # 1280x720 -> the visible square spans x in [280, 1000]
        x, y = to_square_coords(0.5, 0.5, 1280, 720, "crop")
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.5)
        x, _ = to_square_coords(0.1, 0.5, 1280, 720, "crop")
        self.assertLess(x, 0.0, "a subject at the left edge is cropped out")

    def test_pad_of_a_wide_clip_keeps_everything(self):
        x, y = to_square_coords(0.1, 0.0, 1280, 720, "pad")
        self.assertAlmostEqual(x, 0.1)
        self.assertAlmostEqual(y, 0.5 - 0.5 * 720 / 1280)
        for corner in ((0, 0), (1, 1), (0, 1), (1, 0)):
            nx, ny = to_square_coords(*corner, 1280, 720, "pad")
            self.assertTrue(0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0)

    def test_tall_clips_mirror_the_wide_case(self):
        x, y = to_square_coords(0.5, 0.5, 720, 1280, "pad")
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.5)
        _, y = to_square_coords(0.5, 0.05, 720, 1280, "crop")
        self.assertLess(y, 0.0)

    def test_filter_chains(self):
        self.assertIn("crop=640:640", square_filter("crop", 16))
        self.assertIn("pad=640:640", square_filter("pad", 16))
        self.assertTrue(square_filter("pad", 16).startswith("fps=16,"))
        with self.assertRaises(ValueError):
            square_filter("zoom", 16)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
class FfmpegDrivingVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.wide = self.dir / "wide.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=128x72:rate=25:duration=2",
                "-pix_fmt", "yuv420p", str(self.wide),
            ],
            check=True,
        )

    def test_normalizes_to_a_square_at_the_requested_fps(self):
        for fit in ("crop", "pad"):
            out = self.dir / f"{fit}.mp4"
            info = normalize_driving_video(self.wide, out, fps=16, fit=fit, size=64)
            self.assertEqual((info["width"], info["height"]), (64, 64), fit)
            self.assertAlmostEqual(info["fps"], 16, delta=0.01)
            # 2 s of 25 fps resampled to 16 fps -> 32 frames (+-1 for rounding)
            self.assertAlmostEqual(info["frames"], 32, delta=1)

    def test_trim_keeps_only_the_requested_frames(self):
        out = self.dir / "trimmed.mp4"
        trim_video(self.wide, out, 10)
        self.assertEqual(probe_video(out)["frames"], 10)


class AnimateExamplesTests(unittest.TestCase):
    def test_example_payloads_are_valid_requests(self):
        from app.schemas import SequenceRequest

        for name in ("sequence-animate-move.json", "sequence-animate-mix-fight.json"):
            with self.subTest(name=name):
                payload = json.loads((REPO_ROOT / "examples" / name).read_text(encoding="utf-8"))
                request = SequenceRequest.model_validate(payload)
                self.assertTrue(all(shot.mode in {Mode.ANIMATE_MIX, Mode.ANIMATE_MOVE} for shot in request.shots))


class AnimalPoseTests(unittest.TestCase):
    """pose_source="animal": AP10K skeleton instead of DWPose, opt-in, human path untouched."""

    OBJECT_INFO_OLD = {
        "AnimalPosePreprocessor": {
            "input": {
                "required": {
                    "image": ["IMAGE"],
                    "bbox_detector": [["None", "yolox_l.torchscript.pt", "yolox_l.onnx"], {"default": "yolox_l.torchscript.pt"}],
                    "pose_estimator": [["rtmpose-m_ap10k_256.onnx", "rtmpose-m_ap10k_256_bs5.torchscript.pt"], {}],
                    "resolution": ["INT", {}],
                }
            }
        }
    }

    def _graph(self):
        return {
            "212": {"class_type": "ImageScale", "inputs": {}},
            "158": {"class_type": "PixelPerfectResolution", "inputs": {}},
            "101": {
                "class_type": "DWPreprocessor",
                "inputs": {"image": ["212", 0], "detect_body": "enable", "detect_hand": "enable", "resolution": ["158", 0]},
            },
            "100": {
                "class_type": "DWPreprocessor",
                "inputs": {"image": ["212", 0], "detect_body": "disable", "detect_face": "enable", "resolution": ["158", 0]},
            },
        }

    def test_schema_default_and_scope(self):
        self.assertEqual(_animate().pose_source, "human")
        self.assertEqual(_animate(pose_source="animal").pose_source, "animal")
        self.assertEqual(_animate(mode=Mode.ANIMATE_MOVE, pose_source="animal").pose_source, "animal")
        with self.assertRaisesRegex(ValidationError, "only accepted for animate"):
            Shot(id="t", mode=Mode.T2V, prompt="x", pose_source="animal")

    def test_branch_shares_the_squared_frames_and_resolution_of_the_human_pose(self):
        import sys

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import prepare_workflows

        prompt = self._graph()
        self.assertTrue(prepare_workflows.add_animal_pose_branch(prompt, self.OBJECT_INFO_OLD))
        node = prompt[prepare_workflows.ANIMAL_POSE_NODE_ID]
        self.assertEqual(node["class_type"], "AnimalPosePreprocessor")
        self.assertEqual(node["inputs"]["image"], ["212", 0])
        self.assertEqual(node["inputs"]["resolution"], ["158", 0])
        self.assertEqual(node["inputs"]["bbox_detector"], "yolox_l.onnx")  # the one the human pipeline already uses
        self.assertEqual(node["inputs"]["pose_estimator"], "rtmpose-m_ap10k_256_bs5.torchscript.pt")

    def test_branch_understands_the_new_combo_format_and_skips_missing_nodes(self):
        import sys

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import prepare_workflows

        info = {
            "AnimalPosePreprocessor": {
                "input": {
                    "required": {
                        "bbox_detector": ["COMBO", {"options": ["yolox_l.torchscript.pt"]}],
                        "pose_estimator": ["COMBO", {"options": ["rtmpose-m_ap10k_256_bs5.torchscript.pt"]}],
                    }
                }
            }
        }
        prompt = self._graph()
        self.assertTrue(prepare_workflows.add_animal_pose_branch(prompt, info))
        self.assertEqual(prompt[prepare_workflows.ANIMAL_POSE_NODE_ID]["inputs"]["bbox_detector"], "yolox_l.torchscript.pt")
        untouched = self._graph()
        self.assertFalse(prepare_workflows.add_animal_pose_branch(untouched, {}))
        self.assertEqual(untouched, self._graph())
        bad = {"AnimalPosePreprocessor": {"input": {"required": {"bbox_detector": [["x"], {}], "pose_estimator": [["y"], {}]}}}}
        with self.assertRaisesRegex(RuntimeError, "no known YOLOX"):
            prepare_workflows.add_animal_pose_branch(self._graph(), bad)

    def _template(self, with_animal=True):
        stage = lambda: {  # noqa: E731
            "class_type": "WanAnimateToVideo",
            "inputs": {"width": 640, "pose_video": ["101", 0], "face_video": ["100", 0], "reference_image": ["10", 0]},
        }
        nodes = {"subgraph:1:1": stage(), "subgraph:2:2": stage(), "101": {"class_type": "DWPreprocessor", "inputs": {}},
                 "100": {"class_type": "DWPreprocessor", "inputs": {}},
                 "10": {"class_type": "LoadImage", "inputs": {}},
                 "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "lightx2v_I2V_14B.safetensors"}},
                 "animal_pose_ap10k": {"class_type": "AnimalPosePreprocessor", "inputs": {}}}
        bindings = {"width": ["/subgraph:1:1/inputs/width", "/subgraph:2:2/inputs/width"]}
        if with_animal:
            bindings.update(
                pose_video_inputs=["/subgraph:1:1/inputs/pose_video", "/subgraph:2:2/inputs/pose_video"],
                face_video_inputs=["/subgraph:1:1/inputs/face_video", "/subgraph:2:2/inputs/face_video"],
                animal_pose_output=["animal_pose_ap10k", 0],
            )
        return {**nodes, "_bindings": bindings}

    def _bind(self, template, **values):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"WORKFLOW_DIR": temporary}):
            (Path(temporary) / "wan22_animate.api.json").write_text(json.dumps(template), encoding="utf-8")
            return bind_workflow("wan22_animate", {"width": 512, **values})[0]

    def test_animal_pose_replaces_the_skeleton_and_drops_the_human_face_crops(self):
        prompt = self._bind(self._template(), pose_source="animal")
        for stage in ("subgraph:1:1", "subgraph:2:2"):
            inputs = prompt[stage]["inputs"]
            self.assertEqual(inputs["pose_video"], ["animal_pose_ap10k", 0])
            self.assertNotIn("face_video", inputs)
            self.assertIn("reference_image", inputs)

    def test_human_pose_is_untouched_by_default(self):
        for values in ({}, {"pose_source": "human"}):
            prompt = self._bind(self._template(), **values)
            for stage in ("subgraph:1:1", "subgraph:2:2"):
                self.assertEqual(prompt[stage]["inputs"]["pose_video"], ["101", 0])
                self.assertEqual(prompt[stage]["inputs"]["face_video"], ["100", 0])

    def test_animal_pose_with_a_workflow_that_has_no_branch_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "prepare_workflows.py --force"):
            self._bind(self._template(with_animal=False), pose_source="animal")


class JobOutputAsDrivingVideoTests(unittest.TestCase):
    """Chaining Animate passes (cat-vs-cat: pass B replays pass A) without any
    upload route: a completed job's output is a valid driving-video source."""

    def test_job_id_is_a_source_and_exclusive(self):
        self.assertEqual(InputVideo(job_id="fight_pass_a").job_id, "fight_pass_a")
        with self.assertRaisesRegex(ValidationError, "exactly one of"):
            InputVideo(job_id="a", path="b.mp4")
        with self.assertRaises(ValidationError):
            InputVideo(job_id="../etc")

    def _store(self, job):
        return mock.Mock(get=mock.Mock(return_value=job))

    def test_resolves_a_completed_job_output(self):
        from app import orchestrator

        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary) / "outputs"
            (outputs / "pass_a").mkdir(parents=True)
            video = outputs / "pass_a" / "sequence.mp4"
            video.write_bytes(b"x")
            fake_settings = mock.Mock(outputs_dir=outputs)
            job = {"status": "completed", "output": str(video), "metadata": {}}
            with mock.patch.object(orchestrator, "store", self._store(job)), mock.patch.object(
                orchestrator, "settings", fake_settings
            ):
                self.assertEqual(orchestrator.resolve_job_video("pass_a"), video.resolve())

    def test_rejects_unknown_unfinished_image_and_escaping_jobs(self):
        from app import orchestrator

        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary) / "outputs"
            outputs.mkdir()
            outside = Path(temporary) / "elsewhere.mp4"
            outside.write_bytes(b"x")
            fake_settings = mock.Mock(outputs_dir=outputs)
            bad_jobs = [
                None,
                {"status": "running", "output": str(outside), "metadata": {}},
                {"status": "completed", "output": str(outside), "metadata": {"kind": "image"}},
                {"status": "completed", "output": str(outside), "metadata": {}},
            ]
            for job in bad_jobs:
                with self.subTest(job=job), mock.patch.object(orchestrator, "store", self._store(job)), mock.patch.object(
                    orchestrator, "settings", fake_settings
                ), self.assertRaises(ValueError):
                    orchestrator.resolve_job_video("pass_a")


class MoveModeBindingTests(unittest.TestCase):
    """animate_move = animate_mix minus the SAM2 mask and blacked-out background
    inputs of every WanAnimateToVideo (live-verified 2026-10-02)."""

    def _template(self, with_binding=True):
        nodes = {
            "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "template"}},
            "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "lightx2v_I2V_14B.safetensors"}},
            "subgraph:1:2": {
                "class_type": "WanAnimateToVideo",
                "inputs": {"width": 640, "background_video": ["30", 0], "character_mask": ["31", 0], "pose_video": ["30", 0]},
            },
            "subgraph:3:4": {
                "class_type": "WanAnimateToVideo",
                "inputs": {"width": 640, "background_video": ["30", 0], "character_mask": ["31", 0], "pose_video": ["30", 0]},
            },
            "30": {"class_type": "ImageScale", "inputs": {}},
            "31": {"class_type": "GrowMask", "inputs": {}},
            "40": {"class_type": "PointsEditor", "inputs": {"coordinates": "[]", "neg_coordinates": "[]"}},
            "41": {"class_type": "Sam2Segmentation", "inputs": {"coordinates_positive": ["40", 0], "coordinates_negative": ["40", 1]}},
        }
        bindings = {"sam2_negative_input": ["/41/inputs/coordinates_negative"], "positive_prompt": "/10/inputs/text", "width": ["/subgraph:1:2/inputs/width", "/subgraph:3:4/inputs/width"]}
        if with_binding:
            bindings["background_inputs"] = [
                f"/{node}/inputs/{name}"
                for node in ("subgraph:1:2", "subgraph:3:4")
                for name in ("background_video", "character_mask")
            ]
        return {**nodes, "_bindings": bindings}

    def _bind(self, keep_background, with_binding=True, **extra):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"WORKFLOW_DIR": temporary}):
            (Path(temporary) / "wan22_animate.api.json").write_text(
                json.dumps(self._template(with_binding)), encoding="utf-8"
            )
            prompt, _ = bind_workflow(
                "wan22_animate", {"positive_prompt": "cat", "width": 512, "keep_background": keep_background, **extra}
            )
        return prompt

    def test_move_drops_background_and_mask_from_every_stage(self):
        prompt = self._bind(False)
        for node in ("subgraph:1:2", "subgraph:3:4"):
            inputs = prompt[node]["inputs"]
            self.assertNotIn("background_video", inputs)
            self.assertNotIn("character_mask", inputs)
            self.assertIn("pose_video", inputs, "the pose conditioning must stay")
            self.assertEqual(inputs["width"], 512)

    def test_mix_keeps_them(self):
        prompt = self._bind(True)
        for node in ("subgraph:1:2", "subgraph:3:4"):
            self.assertIn("background_video", prompt[node]["inputs"])
            self.assertIn("character_mask", prompt[node]["inputs"])

    def test_move_with_a_stale_workflow_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "prepare_workflows.py --force"):
            self._bind(False, with_binding=False)

    def test_red_points_are_wired_only_when_the_shot_asks_for_them(self):
        default = self._bind(True)
        self.assertNotIn("coordinates_negative", default["41"]["inputs"], "default graph = the live-validated one")
        self.assertIn("coordinates_positive", default["41"]["inputs"])
        wired = self._bind(True, use_negative_points=True)
        self.assertEqual(wired["41"]["inputs"]["coordinates_negative"], ["40", 1])

    def test_red_points_with_a_stale_workflow_fail_loudly(self):
        # a workflow converted before the binding existed has no 'sam2_negative_input'
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"WORKFLOW_DIR": temporary}):
            template = self._template()
            del template["_bindings"]["sam2_negative_input"]
            (Path(temporary) / "wan22_animate.api.json").write_text(json.dumps(template), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "sam2_negative_input"):
                bind_workflow("wan22_animate", {"keep_background": True, "use_negative_points": True})
            bind_workflow("wan22_animate", {"keep_background": True})  # without red points a stale file still works

    def test_converter_links_the_red_output_to_sam2(self):
        import sys

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import prepare_workflows

        def graph():
            return {
                "229": {"class_type": "PointsEditor", "inputs": {}},
                "107": {"class_type": "Sam2Segmentation", "inputs": {"coordinates_positive": ["229", 0]}},
            }

        info = {"PointsEditor": {"output_name": ["positive_coords", "negative_coords", "bbox", "bbox_mask", "cropped_image"]}}
        prompt = graph()
        prepare_workflows.wire_sam2_negative_points(prompt, info)
        self.assertEqual(prompt["107"]["inputs"]["coordinates_negative"], ["229", 1])
        self.assertEqual(prompt["107"]["inputs"]["coordinates_positive"], ["229", 0])
        renamed = {"PointsEditor": {"output_name": ["positive_coords", "other"]}}
        with self.assertRaises(ValueError):
            prepare_workflows.wire_sam2_negative_points(graph(), renamed)
        untouched = {"1": {"class_type": "CLIPTextEncode", "inputs": {}}}
        prepare_workflows.wire_sam2_negative_points(untouched, info)  # non-animate graphs are left alone
        self.assertEqual(untouched, {"1": {"class_type": "CLIPTextEncode", "inputs": {}}})

    def test_converter_collects_the_pointers_and_schema_version_was_bumped(self):
        import sys

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import prepare_workflows

        self.assertGreaterEqual(prepare_workflows.CONVERTER_SCHEMA_VERSION, 6)
        prompt = {
            "a": {"class_type": "WanAnimateToVideo", "inputs": {"width": 1, "height": 1, "background_video": ["x", 0], "character_mask": ["y", 0]}},
            "b": {"class_type": "WanAnimateToVideo", "inputs": {"width": 1, "height": 1, "background_video": ["x", 0], "character_mask": ["y", 0]}},
            "c": {"class_type": "Sam2Segmentation", "inputs": {"coordinates_negative": ["d", 1]}},
        }
        with self.assertRaises(RuntimeError) as raised:
            prepare_workflows.build_bindings(prompt, "wan22_animate")
        # Everything else is missing from this toy graph, but the new binding exists.
        self.assertNotIn("background_inputs", str(raised.exception))
        self.assertNotIn("sam2_negative_input", str(raised.exception))
        self.assertNotIn("'width'", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
