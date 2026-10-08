"""The rainy-sunny anime trend: random plans, requests valid for the API, finishing (rotation, 30 fps) and cutting on a music timing."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import music_timing as mt  # noqa: E402
import trend_rain_anime as tr  # noqa: E402

from app.schemas import ImageGenerationRequest, SequenceRequest  # noqa: E402


def probe(path: Path) -> dict:
    out = subprocess.run([mt.ffmpeg_binary().replace("ffmpeg.exe", "ffprobe.exe") if mt.ffmpeg_binary().endswith("ffmpeg.exe") else "ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-count_frames", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames:format=duration", "-of", "default=nw=1", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.split() if "=" in line)


class PlanTests(unittest.TestCase):
    def test_the_same_seed_gives_the_same_plan_and_other_seeds_differ(self):
        self.assertEqual(tr.plan_trend(7, 8), tr.plan_trend(7, 8))
        plans = {json.dumps(tr.plan_trend(seed, 8)["shots"]) for seed in range(12)}
        self.assertGreater(len(plans), 10)  # really random from one video to the next

    def test_a_plan_has_the_requested_shots_with_unique_ids_and_seeds_and_prompts(self):
        plan = tr.plan_trend(3, 10)
        self.assertEqual([s["id"] for s in plan["shots"]], [f"s{i:02d}" for i in range(1, 11)])
        self.assertEqual(len({s["seed"] for s in plan["shots"]}), 10)
        for shot in plan["shots"]:
            self.assertNotIn("{", shot["scene"])  # every slot is filled
            self.assertIn(plan["context"]["weather_words"], shot["scene"])  # the whole sequence shares the weather and the light
            self.assertIn(plan["context"]["light_words"], shot["scene"])
            self.assertTrue(shot["motion"])

    def test_never_the_same_archetype_twice_in_a_row_and_several_families(self):
        for seed in range(40):
            plan = tr.plan_trend(seed, 10)
            names = [s["archetype"] for s in plan["shots"]]
            self.assertTrue(all(a != b for a, b in zip(names, names[1:])), (seed, names))
            self.assertGreaterEqual(len({s["family"] for s in plan["shots"]}), 3, (seed, names))

    def test_the_wow_puddle_shot_is_there_when_the_weather_allows_it_and_never_in_heavy_rain(self):
        for seed in range(40):
            plan = tr.plan_trend(seed, 8)
            has_wow = any(tr.ARCHETYPES[s["archetype"]].get("wow") for s in plan["shots"])
            if plan["context"]["wow_allowed"]:
                self.assertTrue(has_wow, (seed, plan["context"]["weather"]))
            else:
                self.assertFalse(has_wow, (seed, plan["context"]["weather"]))
        heavy = tr.plan_trend(1, 8, weather="heavy_rain")
        self.assertFalse(any("puddle" in s["scene"] and "reflect" in s["scene"] for s in heavy["shots"]))

    def test_the_setting_decides_which_shots_exist(self):
        for seed in range(15):
            nature = tr.plan_trend(seed, 10, setting="nature")
            self.assertTrue(all(tr.ARCHETYPES[s["archetype"]]["needs"] != "city" for s in nature["shots"]), seed)
            city = tr.plan_trend(seed, 10, setting="calm city")
            self.assertTrue(all(tr.ARCHETYPES[s["archetype"]]["needs"] != "nature" for s in city["shots"]), seed)

    def test_a_hand_made_start_can_be_forced_and_bad_input_is_refused(self):
        plan = tr.plan_trend(5, 6, force=["flowers", "bubbles_water"], setting="nature")
        self.assertEqual([s["archetype"] for s in plan["shots"][:2]], ["flowers", "bubbles_water"])
        with self.assertRaises(ValueError):
            tr.plan_trend(5, 6, force=["nope"])
        with self.assertRaises(ValueError):
            tr.plan_trend(5, 0)


class RequestTests(unittest.TestCase):
    def test_every_style_gives_a_request_the_api_accepts(self):
        shot = tr.plan_trend(2, 4)["shots"][0]
        for style in tr.STYLES:
            body = tr.still_request(shot, style)
            request = ImageGenerationRequest.model_validate(body)  # the real schema of the service
            self.assertEqual((request.engine, request.width, request.height, request.seed), ("krea2", 1024, 576, shot["seed"]))
            self.assertEqual([(item.id, item.weight) for item in request.loras], list(tr.STYLES[style]["loras"]))
            self.assertIn(shot["scene"], request.prompt)

    def test_the_lora_ids_exist_in_the_catalog_for_the_krea2_engine(self):
        from app.catalog import lora_entry, validate_lora_for_stage

        for style, spec in tr.STYLES.items():
            for lora_id, _ in spec["loras"]:
                validate_lora_for_stage(lora_entry(lora_id), "krea2", "keyframe")

    def test_the_trigger_words_are_in_the_prompts_of_their_styles(self):
        from app.catalog import lora_entry

        shot = tr.plan_trend(2, 3)["shots"][0]
        for style, spec in tr.STYLES.items():
            for lora_id, _ in spec["loras"]:
                for trigger in lora_entry(lora_id)["trigger_words"]:
                    self.assertIn(trigger, tr.still_request(shot, style)["prompt"], (style, trigger))

    def test_the_clip_job_is_a_valid_turbo_image_to_video_sequence(self):
        shot = tr.plan_trend(2, 4)["shots"][1]
        request = SequenceRequest.model_validate(tr.clip_job(shot, "img_" + "a" * 32, "job-1"))
        only = request.shots[0]
        self.assertEqual((only.mode.value, only.turbo_mode, only.steps, only.cfg), ("i2v", True, 4, 1.0))
        self.assertEqual((only.width, only.height, only.frames, only.fps), (1024, 576, 49, 16))
        self.assertEqual(only.start_image.image_id, "img_" + "a" * 32)
        self.assertAlmostEqual(only.frames / only.fps, 3.0, delta=0.1)  # a 3 s shot


class ShadowTests(unittest.TestCase):
    def test_lifting_the_shadows_opens_the_dark_tones_and_keeps_the_whites(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (32, 1))
            Image.fromarray(np.dstack([ramp] * 3)).save(tmp / "ramp.png")
            out = np.asarray(Image.open(tr.lift_shadows(tmp / "ramp.png", tmp / "out" / "lift.png", 0.6, sharpen=0)).convert("L"), np.int32)
            before = ramp.astype(np.int32)
            self.assertGreater(int(out[0, 40]), int(before[0, 40]) + 25)  # a dark tone rises clearly
            self.assertEqual(int(out[0, 255]), 255)  # white stays white
            self.assertEqual(int(out[0, 0]), 0)  # black stays black
            self.assertTrue((out[0, 1:] >= before[0, 1:]).all())  # nothing gets darker
            self.assertTrue((np.diff(out[0]) >= 0).all())  # the tones keep their order
            same = np.asarray(Image.open(tr.lift_shadows(tmp / "ramp.png", tmp / "same.png", 0.0, sharpen=0)).convert("L"), np.int32)
            self.assertLessEqual(int(np.abs(same - before).max()), 1)  # strength 0 = untouched
        with self.assertRaises(ValueError):
            tr.lift_shadows(Path("x.png"), Path("y.png"), -1)

    def test_the_new_style_and_the_low_shadow_phrase_reach_the_request(self):
        from app.schemas import ImageGenerationRequest

        shot = {"id": "x", "scene": f"a sparrow on a branch, {tr.LOW_SHADOW}", "seed": 5}
        body = tr.still_request(shot, "enhance_gurren_18")
        request = ImageGenerationRequest.model_validate(body)
        self.assertEqual([(i.id, i.weight) for i in request.loras], [("krea2_turbo_anime_enhance", 0.7), ("krea2_gurren_lagann_style", 1.8)])
        self.assertIn("no cast shadows", request.prompt)
        self.assertIn("GurrenLagannStyle", request.prompt)


class RegisterTests(unittest.TestCase):
    def test_every_generation_is_written_down_with_its_seed_and_nothing_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "sub" / "generations.json"
            shot = tr.plan_trend(4, 3)["shots"][0]
            request = tr.still_request(shot, "gurren")
            tr.record_generation(registry, {"kind": "still", "id": "x", "seed": request["seed"], "loras": request["loras"], "prompt": request["prompt"]})
            tr.record_generation(registry, {"kind": "clip", "id": "y", "seed": shot["seed"]})
            items = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual([i["id"] for i in items], ["x", "y"])
            self.assertEqual(items[0]["seed"], shot["seed"])
            self.assertEqual(items[0]["loras"], [{"id": "krea2_gurren_lagann_style", "weight": 1.0, "target": "keyframe"}])
            self.assertIn("when", items[0])
            self.assertNotIn("token", registry.read_text(encoding="utf-8").lower())
            tr.record_generation(None, {"kind": "still"})  # no register: nothing happens

    def test_the_replayable_request_keeps_the_same_seed(self):
        shot = tr.plan_trend(9, 5)["shots"][2]
        first, second = tr.still_request(shot, "enhance"), tr.still_request(shot, "enhance")
        self.assertEqual(first, second)  # the same shot, style and seed: the same request, so the same picture
        self.assertEqual(first["seed"], shot["seed"])
        self.assertEqual(tr.clip_job(shot, "img_" + "b" * 32, "j")["shots"][0]["seed"], shot["seed"])


class FinishAndAssembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        for i, colour in enumerate(("red", "green", "blue")):
            subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=256x144:rate=16:duration=1.5,hue=h={i * 100}",
                            "-pix_fmt", "yuv420p", str(cls.dir / f"raw{i}.mp4")], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_landscape_clip_becomes_a_portrait_30_fps_clip(self):
        out = tr.finish_clip(self.dir / "raw0.mp4", self.dir / "fin0.mp4", size=(144, 256))
        info = probe(out)
        self.assertEqual((info["width"], info["height"]), ("144", "256"))
        self.assertEqual(info["r_frame_rate"], "30/1")
        self.assertAlmostEqual(float(info["duration"]), 1.5, delta=0.2)  # motion interpolation ends on the last source frame

    def test_the_rotation_direction_is_chosen_and_checked(self):
        from PIL import Image

        for direction in ("clockwise", "counterclockwise"):
            src = self.dir / f"half_{direction}.png"
            image = Image.new("RGB", (256, 144), (0, 0, 0))
            image.paste((255, 255, 255), (0, 0, 128, 144))  # the LEFT half of the landscape picture is white
            image.save(src)
            clip = self.dir / f"half_{direction}.mp4"
            subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-loop", "1", "-framerate", "16", "-t", "1", "-i", str(src), "-pix_fmt", "yuv420p", str(clip)], check=True)
            out = tr.finish_clip(clip, self.dir / f"rot_{direction}.mp4", size=(144, 256), interpolate=False, direction=direction)
            png = self.dir / f"rot_{direction}.png"
            subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(out), "-frames:v", "1", str(png)], check=True)
            top_is_white = Image.open(png).convert("L").getpixel((72, 20)) > 200
            # turned clockwise the left edge of the picture ends up on TOP; counterclockwise, at the bottom
            self.assertEqual(top_is_white, direction == "clockwise", direction)
        with self.assertRaises(ValueError):
            tr.rotate_filter("sideways")

    def test_shot_durations_follow_the_cuts(self):
        self.assertEqual(tr.shot_durations([2.0, 3.5, 7.0], 10.0), [2.0, 1.5, 3.5, 3.0])
        self.assertEqual(tr.shot_durations([], 5.0), [5.0])
        self.assertEqual(tr.shot_durations([0.0, 5.0, 12.0, 2.0], 5.0), [2.0, 3.0])  # cuts outside (0, total) are ignored

    def test_the_assembly_cuts_on_a_recorded_timing_and_lays_the_music(self):
        clips = [tr.finish_clip(self.dir / f"raw{i}.mp4", self.dir / f"f{i}.mp4", size=(144, 256), interpolate=False) for i in range(3)]
        music = self.dir / "music.wav"
        subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=12", str(music)], check=True)
        timing = {"mode": "loop", "loop": {"start": 1.0, "end": 5.0}, "marks": [1.0, 2.0, 3.5]}
        cuts = mt.expand_cut_times(timing, 8.0)  # the loop's marks, repeated every 4 s, from the loop start
        self.assertEqual(cuts, [1.0, 2.5, 4.0, 5.0, 6.5])
        durations = tr.shot_durations(cuts, 8.0)
        out = tr.assemble(clips, durations, self.dir / "trend.mp4", music=music, music_offset=1.0)
        info = probe(out)
        self.assertEqual((info["width"], info["height"]), ("144", "256"))
        self.assertAlmostEqual(float(info["duration"]), 8.0, delta=0.2)
        audio = subprocess.run([mt.ffmpeg_binary().replace("ffmpeg.exe", "ffprobe.exe") if mt.ffmpeg_binary().endswith("ffmpeg.exe") else "ffprobe", "-v", "error",
                                "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
        self.assertIn("aac", audio)


if __name__ == "__main__":
    unittest.main()
