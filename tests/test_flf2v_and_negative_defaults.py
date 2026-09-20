import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.catalog import lora_entry, validate_lora_for_stage
from app.comfy import bind_workflow


class Flf2vLoraCompatibilityTests(unittest.TestCase):
    """FLF2V shares the Wan 2.2 I2V model family, so I2V LoRAs must be accepted there."""

    def test_i2v_lora_is_accepted_on_the_flf2v_workflow(self):
        validate_lora_for_stage(
            lora_entry("spatial_magic_v2_wan22_i2v_high"),
            "wan22_i2v",
            "high",
            workflow="wan22_flf2v",
        )

    def test_i2v_lora_is_still_accepted_on_the_i2v_workflow(self):
        validate_lora_for_stage(
            lora_entry("spatial_magic_v2_wan22_i2v_high"),
            "wan22_i2v",
            "high",
            workflow="wan22_i2v",
        )

    def test_t2v_lora_is_still_rejected_on_flf2v(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_lora_for_stage(
                lora_entry("pixel_gamegirl_wan22_t2v_high"),
                "wan22_i2v",
                "high",
                workflow="wan22_flf2v",
            )

    def test_alias_is_not_symmetric_for_unrelated_workflows(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_lora_for_stage(
                lora_entry("spatial_magic_v2_wan22_i2v_high"),
                "wan22_i2v",
                "high",
                workflow="wan22_t2v",
            )


class NegativePromptDefaultTests(unittest.TestCase):
    OFFICIAL = "official default negative prompt"

    def _bind(self, negative_prompt):
        workflow = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": self.OFFICIAL}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "template positive"}},
            "_bindings": {
                "positive_prompt": "/2/inputs/text",
                "negative_prompt": "/1/inputs/text",
            },
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"WORKFLOW_DIR": temporary}
        ):
            (Path(temporary) / "wan22_t2v.api.json").write_text(
                json.dumps(workflow), encoding="utf-8"
            )
            prompt, _ = bind_workflow(
                "wan22_t2v",
                {"positive_prompt": "a bird", "negative_prompt": negative_prompt},
            )
        return prompt

    def test_empty_negative_prompt_keeps_the_workflow_default(self):
        prompt = self._bind("")
        self.assertEqual(prompt["1"]["inputs"]["text"], self.OFFICIAL)
        self.assertEqual(prompt["2"]["inputs"]["text"], "a bird")

    def test_blank_negative_prompt_keeps_the_workflow_default(self):
        self.assertEqual(self._bind("  \n ")["1"]["inputs"]["text"], self.OFFICIAL)

    def test_explicit_negative_prompt_still_overrides(self):
        self.assertEqual(self._bind("blurry")["1"]["inputs"]["text"], "blurry")


class FluxNijiLoraCatalogTests(unittest.TestCase):
    LORA_ID = "flux_schnell_niji_better_art"

    def test_is_accepted_for_flux_keyframes(self):
        for target in ("auto", "keyframe"):
            validate_lora_for_stage(lora_entry(self.LORA_ID), "flux_schnell", target)

    def test_is_rejected_on_wan_stages(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_lora_for_stage(
                lora_entry(self.LORA_ID), "wan22_i2v", "high", workflow="wan22_flf2v"
            )

    def test_is_rejected_for_video_targets(self):
        with self.assertRaisesRegex(ValueError, "cannot target"):
            validate_lora_for_stage(lora_entry(self.LORA_ID), "flux_schnell", "high")

    def test_download_is_pinned_by_size_and_sha256(self):
        entry = lora_entry(self.LORA_ID)
        self.assertEqual(entry["size_bytes"], 343805518)
        self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(entry["trigger_words"], ["niji"])
        self.assertTrue(entry["url"].startswith("https://civitai.com/api/download/models/"))


if __name__ == "__main__":
    unittest.main()
