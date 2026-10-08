"""The jump cut shot (Mode.JUMPCUT): schema, admission, the FFmpeg inventory, a real render, and the whole thing through the HTTP API."""
from __future__ import annotations

import asyncio
import dataclasses
import importlib
import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app import limits, montage
from app.schemas import JumpCutSpec, Mode, SequenceRequest, Shot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import trend_philosopher as tp  # noqa: E402

FFMPEG_READY = bool(montage.ffmpeg_status().get("ready"))
RED, GREEN, BLUE = (230, 20, 20), (20, 200, 20), (20, 20, 230)


def _images(count: int = 3) -> list[dict]:
    return [{"path": f"still{i}.png"} for i in range(count)]


def _shot(**overrides) -> Shot:
    jumpcut = {"images": _images(), **overrides.pop("jumpcut", {})}
    return Shot(**{"id": "burst", "mode": "jumpcut", "frames": 96, "fps": 24, "jumpcut": jumpcut, **overrides})


def _png(path: Path, colour: tuple[int, int, int], size: tuple[int, int] = (256, 256)) -> Path:
    Image.new("RGB", size, colour).save(path)
    return path


def _probe(video: Path) -> dict:
    out = subprocess.run(
        [montage.ffprobe_binary(), "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,nb_read_frames:format=duration", "-of", "default=nw=1", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.split() if "=" in line)


def _pixel(video: Path, at: float, tmp: Path, xy: tuple[int, int] = (128, 128)) -> tuple[int, int, int]:
    png = tmp / f"frame_{at}.png"
    subprocess.run([montage.ffmpeg_binary(), "-y", "-v", "error", "-ss", str(at), "-i", str(video), "-frames:v", "1", str(png)], check=True)
    return Image.open(png).convert("RGB").getpixel(xy)


def _dominant(pixel: tuple[int, int, int]) -> int:
    return max(range(3), key=lambda channel: pixel[channel])


class JumpCutSchemaTests(unittest.TestCase):
    def test_images_alone_make_a_complete_shot(self):
        shot = _shot()
        self.assertEqual(shot.mode, Mode.JUMPCUT)
        spec = shot.jumpcut
        self.assertEqual((spec.transition, spec.constant_shake, spec.ratios), ("cut", None, None))
        self.assertEqual((spec.cut_min_seconds, spec.cut_max_seconds), tp.BG_FLASH_CUT_RANGE)
        self.assertEqual(spec.transition_seconds, tp.BURST_TRANSITION_SECONDS)

    def test_the_options_can_be_chosen(self):
        spec = _shot(jumpcut={"ratios": [1, 1, 2], "transition": "swing", "constant_shake": "rock", "constant_shake_amount": 0.5}).jumpcut
        self.assertEqual((spec.ratios, spec.transition, spec.constant_shake, spec.constant_shake_amount), ([1, 1, 2], "swing", "rock", 0.5))

    def test_needs_the_jumpcut_options_and_two_images(self):
        with self.assertRaisesRegex(ValidationError, "requires the jumpcut options"):
            Shot(id="s", mode="jumpcut")
        with self.assertRaises(ValidationError):
            _shot(jumpcut={"images": _images(1)})

    def test_gpu_inputs_are_refused(self):
        for extra in ({"start_image": {"path": "a.png"}}, {"prompt": "x", "loras": [{"id": "l"}]}, {"turbo_mode": True},
                      {"generate_start_image": {"engine": "krea2", "prompt": "x"}}):
            if "prompt" in extra:
                extra = {"loras": extra["loras"]}
            with self.assertRaisesRegex(ValidationError, "rendered by FFmpeg"):
                _shot(**extra)

    def test_ratios_that_leave_an_image_under_50_ms_are_refused(self):
        with self.assertRaisesRegex(ValidationError, "would stay only"):
            _shot(frames=96, jumpcut={"ratios": [1, 1000]})  # 4 s: the first image would stay 4 ms
        _shot(frames=96, jumpcut={"ratios": [1, 10]})  # 4 s / 11 = 0.36 s each: fine

    def test_bad_values_are_refused(self):
        for bad in ({"ratios": [1, 0, 2]}, {"ratios": [1, -1]}, {"transition": "spin"}, {"constant_shake": "earthquake"},
                    {"cut_min_seconds": 0.5, "cut_max_seconds": 0.2}, {"transition_seconds": 0}, {"unknown_field": 1}):
            with self.assertRaises(ValidationError, msg=str(bad)):
                _shot(jumpcut=bad)

    def test_jumpcut_options_are_refused_on_other_modes(self):
        with self.assertRaisesRegex(ValidationError, "only accepted for jumpcut shots"):
            Shot(id="s", mode="t2v", prompt="a bird", jumpcut={"images": _images()})

    def test_the_choices_mirror_the_script(self):
        from typing import get_args

        from app.schemas import JumpCutShake, JumpCutTransition

        self.assertEqual(set(get_args(JumpCutTransition)), set(tp.BURST_TRANSITIONS))
        self.assertEqual(set(get_args(JumpCutShake)), set(tp.CONSTANT_SHAKES))
        self.assertEqual(set(montage.TRANSITION_FILTERS), set(tp.BURST_TRANSITIONS))
        self.assertEqual(set(montage.SHAKE_FILTERS), set(tp.CONSTANT_SHAKES))


class AdmissionTests(unittest.TestCase):
    def _settings(self):
        return SimpleNamespace(max_shots_per_job=8, max_total_frames=968, max_job_cost_units=8, max_pending_jobs=2, min_free_disk_gb=0,
                               max_output_disk_gb=50, outputs_dir=Path("."), data_root=Path("."))

    def test_a_jumpcut_costs_almost_nothing_compared_with_a_diffusion_shot(self):
        jump = SequenceRequest(id="j", shots=[_shot()])
        wan = SequenceRequest(id="w", shots=[Shot(id="w", mode="t2v", prompt="a bird")])
        self.assertEqual(limits.estimate_cost_units(jump), limits.JUMPCUT_COST_UNITS)
        self.assertGreaterEqual(limits.estimate_cost_units(wan), 50 * limits.estimate_cost_units(jump))  # a default Wan shot is 1.0 unit

    def test_admitted_like_any_sequence(self):
        request = SequenceRequest(id="j", shots=[_shot(id="a"), _shot(id="b")])
        store = SimpleNamespace(pending_count=lambda: 0)
        cost = limits.admit_job(request, self._settings(), store, disk_usage=lambda _: SimpleNamespace(free=10**12), output_size=lambda _: 0)
        self.assertAlmostEqual(cost, 2 * limits.JUMPCUT_COST_UNITS)

    def test_the_frames_still_count_and_sizes_must_match_other_shots(self):
        request = SequenceRequest(id="j", shots=[_shot(id="a"), _shot(id="b", width=512)])
        with self.assertRaisesRegex(limits.AdmissionError, "same width and height"):
            limits.admit_job(request, self._settings(), SimpleNamespace(pending_count=lambda: 0),
                             disk_usage=lambda _: SimpleNamespace(free=10**12), output_size=lambda _: 0)


class FfmpegInventoryTests(unittest.TestCase):
    FILTERS_4 = " T.C acrusher          A->A       Reduce audio bit resolution.\n ... xfade             VV->V      Cross fade one video with another video.\n T.. perspective       V->V       Correct the perspective of video.\n"
    FILTERS_8 = " TS aap               AA->A      Apply Affine Projection algorithm to first audio stream.\n .. xfade             VV->V      Cross fade one video with another video.\n T. overlay           VV->V      Overlay a video source on top of the input.\n"
    ENCODERS = " V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)\n A....D aac                  AAC (Advanced Audio Coding)\n"

    def test_filter_names_are_read_from_ffmpeg_4_and_8_listings(self):
        self.assertEqual(montage.parse_filter_names(self.FILTERS_4), {"acrusher", "xfade", "perspective"})
        self.assertEqual(montage.parse_filter_names(self.FILTERS_8), {"aap", "xfade", "overlay"})
        self.assertEqual(montage.parse_encoder_names(self.ENCODERS), {"libx264", "aac"})

    def test_a_missing_filter_refuses_the_shot_with_a_clear_message(self):
        inventory = ("ffmpeg version 4.4.2-0ubuntu0.22.04.1", frozenset(montage.BASE_FILTERS), frozenset({"libx264"}))
        with mock.patch.object(montage, "_inventory", return_value=inventory), \
             mock.patch.object(montage, "ffmpeg_status", return_value={"version": "4.4.2", "binary": "ffmpeg"}):
            montage.validate_jumpcut(_shot().jumpcut)  # plain cuts need nothing exotic
            with self.assertRaisesRegex(ValueError, r"lacks \['xfade'\]"):
                montage.validate_jumpcut(_shot(jumpcut={"transition": "swing"}).jumpcut)
            with self.assertRaisesRegex(ValueError, "perspective"):
                montage.validate_jumpcut(_shot(jumpcut={"constant_shake": "rock"}).jumpcut)

    def test_required_filters_follow_the_chosen_options(self):
        self.assertEqual(montage.required_filters(_shot().jumpcut), sorted(montage.BASE_FILTERS))
        self.assertIn("xfade", montage.required_filters(_shot(jumpcut={"transition": "wipe_left"}).jumpcut))
        self.assertTrue({"pad", "fillborders", "perspective"} <= set(montage.required_filters(_shot(jumpcut={"constant_shake": "rock"}).jumpcut)))

    @unittest.skipUnless(FFMPEG_READY, "no usable ffmpeg")
    def test_this_machines_ffmpeg_can_render_every_option(self):
        status = montage.ffmpeg_status()
        self.assertTrue(status["ready"], status)
        self.assertEqual(status["missing"], [])
        self.assertIsNotNone(status["version"])


@unittest.skipUnless(FFMPEG_READY, "no usable ffmpeg")
class JumpCutRenderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.stills = [_png(self.tmp / f"still{i}.png", colour) for i, colour in enumerate((RED, GREEN, BLUE))]

    def test_ratios_give_each_image_its_share_of_the_shot(self):
        spec = _shot(jumpcut={"ratios": [1, 1, 2]}).jumpcut  # 4 s -> 1 s, 1 s, 2 s
        out = montage.render_jumpcut(spec, self.stills, self.tmp / "work" / "jc.mp4", width=256, height=256, fps=24, seconds=4.0, seed=1)
        info = _probe(out)
        self.assertAlmostEqual(float(info["duration"]), 4.0, delta=0.15)
        self.assertEqual((info["width"], info["height"]), ("256", "256"))
        self.assertEqual([_dominant(_pixel(out, at, self.tmp)) for at in (0.5, 1.5, 3.0)], [0, 1, 2])

    def test_the_automatic_mode_cuts_fast(self):
        spec = _shot().jumpcut
        out = montage.render_jumpcut(spec, self.stills, self.tmp / "auto" / "jc.mp4", width=256, height=256, fps=24, seconds=2.0, seed=3)
        self.assertAlmostEqual(float(_probe(out)["duration"]), 2.0, delta=0.15)
        seen = {_dominant(_pixel(out, at, self.tmp)) for at in (0.1, 0.5, 0.9, 1.3, 1.7)}
        self.assertGreaterEqual(len(seen), 2)  # 0.2-0.4 s per image: several different images within 2 s

    def test_wipes_and_the_smooth_rock_shake_render_on_this_ffmpeg(self):
        spec = _shot(jumpcut={"ratios": [1, 1, 1], "transition": "swing", "transition_seconds": 0.2, "constant_shake": "rock"}).jumpcut
        out = montage.render_jumpcut(spec, self.stills, self.tmp / "fancy" / "jc.mp4", width=256, height=256, fps=24, seconds=3.0, seed=5)
        self.assertAlmostEqual(float(_probe(out)["duration"]), 3.0, delta=0.15)
        self.assertEqual(_dominant(_pixel(out, 0.5, self.tmp)), 0)
        self.assertEqual(_dominant(_pixel(out, 2.5, self.tmp)), 2)

    def test_each_shot_keeps_its_own_work_folder(self):
        import app.orchestrator as orchestrator

        inputs = self.tmp / "inputs"
        inputs.mkdir()
        for path in self.stills:
            (inputs / path.name).write_bytes(path.read_bytes())
        local = dataclasses.replace(orchestrator.settings, data_root=self.tmp)
        job_dir = self.tmp / "job"
        job_dir.mkdir()
        shots = [_shot(id="one", jumpcut={"ratios": [1, 1, 1]}), _shot(id="two", jumpcut={"transition": "wipe_left"})]
        with mock.patch.object(orchestrator, "settings", local):
            videos = [asyncio.run(orchestrator.render_shot(shot, job_dir)) for shot in shots]
        self.assertEqual([video.name for video in videos], ["one.mp4", "two.mp4"])
        self.assertTrue(all(video.is_file() for video in videos))
        self.assertFalse((job_dir / "one_jumpcut").exists(), "intermediate files are cleaned up")
        final = job_dir / "sequence.mp4"
        orchestrator.concatenate(videos, final, "cut", 0.25, reencode=True)
        self.assertAlmostEqual(float(_probe(final)["duration"]), 8.0, delta=0.3)  # 96 frames at 24 fps, twice


class JumpCutApiTests(unittest.TestCase):
    """The service end to end: upload stills, POST /v1/jobs with a jumpcut shot, wait, download the mp4."""

    def _start(self, test_mode: bool):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        updates = {"DATA_ROOT": self.temporary.name, "API_TOKEN": "test-token", "REQUIRE_API_TOKEN": "1", "GPU_CONCURRENCY": "1",
                   "MIN_FREE_DISK_GB": "0", "APP_TEST_MODE": "1" if test_mode else "0"}
        previous = {key: os.environ.get(key) for key in updates}
        os.environ.update(updates)

        def restore():
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore)
        import app.main as main_module
        import app.orchestrator as orchestrator_module

        importlib.reload(orchestrator_module)
        main = importlib.reload(main_module)
        context = TestClient(main.app)
        self.client = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.headers = {"Authorization": "Bearer test-token"}

    def _upload(self, colour) -> str:
        data = io.BytesIO()
        Image.new("RGB", (256, 256), colour).save(data, format="PNG")
        response = self.client.post("/v1/images/upload", files={"file": ("a.png", data.getvalue(), "image/png")}, headers=self.headers)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["image_id"]

    def _job(self, images: list[str], **jumpcut) -> dict:
        return {"id": "jc1", "shots": [{"id": "burst", "mode": "jumpcut", "frames": 48, "fps": 24,
                                       "jumpcut": {"images": [{"image_id": i} for i in images], **jumpcut}}]}

    def test_the_options_route_lists_the_choices_and_the_ffmpeg(self):
        self._start(test_mode=True)
        body = self.client.get("/v1/montage", headers=self.headers).json()
        self.assertEqual(set(body["jumpcut"]["transitions"]), set(tp.BURST_TRANSITIONS))
        self.assertEqual(set(body["jumpcut"]["constant_shakes"]), set(tp.CONSTANT_SHAKES))
        self.assertEqual(body["jumpcut"]["defaults"]["transition"], "cut")
        self.assertIn("version", body["ffmpeg"])
        self.assertEqual(self.client.get("/v1/montage").status_code, 401)
        ready = self.client.get("/health/ready").json()
        self.assertIn("ffmpeg", ready["capabilities"]["montage"])

    def test_a_valid_job_is_accepted_and_a_bad_one_is_refused(self):
        self._start(test_mode=True)
        images = [self._upload(RED), self._upload(GREEN)]
        accepted = self.client.post("/v1/jobs", json=self._job(images, ratios=[1, 2]), headers=self.headers)
        self.assertEqual(accepted.status_code, 202, accepted.text)
        self.assertAlmostEqual(accepted.json()["estimated_cost_units"], limits.JUMPCUT_COST_UNITS)
        bad = self._job(images, ratios=[1, 1000])
        bad["id"] = "jc2"
        self.assertEqual(self.client.post("/v1/jobs", json=bad, headers=self.headers).status_code, 422)
        unknown = self._job(["img_" + "0" * 32])
        unknown["id"] = "jc3"
        self.assertEqual(self.client.post("/v1/jobs", json=unknown, headers=self.headers).status_code, 422)

    @unittest.skipUnless(FFMPEG_READY, "no usable ffmpeg")
    def test_a_real_job_renders_the_video_on_the_server(self):
        self._start(test_mode=False)
        images = [self._upload(RED), self._upload(GREEN), self._upload(BLUE)]
        response = self.client.post("/v1/jobs", json=self._job(images, ratios=[1, 1, 2], transition="swing", constant_shake="rock"),
                                    headers=self.headers)
        self.assertEqual(response.status_code, 202, response.text)
        deadline = time.time() + 120
        job = {}
        while time.time() < deadline:
            job = self.client.get("/v1/jobs/jc1", headers=self.headers).json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.5)
        self.assertEqual(job["status"], "completed", job)
        output = self.client.get("/v1/jobs/jc1/output", headers=self.headers)
        self.assertEqual(output.status_code, 200)
        video = Path(self.temporary.name) / "downloaded.mp4"
        video.write_bytes(output.content)
        info = _probe(video)
        self.assertAlmostEqual(float(info["duration"]), 2.0, delta=0.15)  # 48 frames at 24 fps
        self.assertEqual((info["width"], info["height"]), ("832", "480"))


if __name__ == "__main__":
    unittest.main()
