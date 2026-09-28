import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.schemas import ImageGenerationRequest, KeyframeStage

REPO_ROOT = Path(__file__).resolve().parent.parent


class Krea2SchemaTests(unittest.TestCase):
    def test_negative_prompt_rejected(self):
        with self.assertRaisesRegex(ValidationError, "krea2 does not support a negative prompt"):
            ImageGenerationRequest(engine="krea2", prompt="x", negative_prompt="no")
        with self.assertRaisesRegex(ValidationError, "krea2 does not support a negative prompt"):
            KeyframeStage(engine="krea2", prompt="x", negative_prompt="no")

    def test_default_steps_is_krea2s_own_eight_not_flux_four(self):
        request = ImageGenerationRequest(engine="krea2", prompt="x")
        self.assertEqual(request.steps, 8)
        stage = KeyframeStage(engine="krea2", prompt="x")
        self.assertEqual(stage.steps, 8)

    def test_explicit_steps_are_not_overridden(self):
        request = ImageGenerationRequest(engine="krea2", prompt="x", steps=2)
        self.assertEqual(request.steps, 2)

    def test_flux_default_unaffected(self):
        request = ImageGenerationRequest(engine="flux_schnell", prompt="x")
        self.assertEqual(request.steps, 4)

    def test_krea2_steps_capped_by_the_shared_field_bound(self):
        # 8 is both krea2's own per-engine cap and the shared Field(le=8) bound,
        # so there is no value that trips the per-engine check without already
        # tripping pydantic's own bound first; this just confirms 9 is rejected.
        with self.assertRaisesRegex(ValidationError, "less_than_equal"):
            ImageGenerationRequest(engine="krea2", prompt="x", steps=9)

    def test_flux_steps_still_capped_at_four(self):
        with self.assertRaisesRegex(ValidationError, "flux_schnell supports at most 4 steps"):
            ImageGenerationRequest(engine="flux_schnell", prompt="x", steps=8)


class Krea2CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((REPO_ROOT / "config/base_models.json").read_text(encoding="utf-8"))

    def test_image_engine_entry(self):
        spec = self.catalog["image_engines"]["krea2"]
        self.assertTrue(spec["supported"])
        self.assertEqual(spec["workflow"], "image_krea2.api.json")
        self.assertEqual(
            set(spec["model_ids"]), {"krea2_turbo_diffusion", "krea2_text_encoder", "krea2_vae"}
        )

    def test_profile_and_capability(self):
        self.assertEqual(
            set(self.catalog["profiles"]["krea2"]),
            {"krea2_turbo_diffusion", "krea2_text_encoder", "krea2_vae"},
        )
        self.assertEqual(self.catalog["profile_capabilities"]["krea2"], ["image_generation"])

    def test_model_files_have_verified_sizes_and_pinned_commits(self):
        expected = {
            "krea2_turbo_diffusion": 13141730784,
            "krea2_text_encoder": 5242467968,
            "krea2_vae": 253806246,
        }
        for item_id, size in expected.items():
            entry = self.catalog["items"][item_id]
            self.assertEqual(entry["size_bytes"], size, item_id)
            self.assertTrue(entry["sha256"], item_id)
            self.assertNotIn("/resolve/main/", entry["url"], "must pin an exact commit")


class Krea2WorkflowBindingTests(unittest.TestCase):
    def test_bindings_resolve_with_no_missing_references(self):
        from app.comfy import bind_workflow

        with tempfile.TemporaryDirectory() as temporary:
            shutil.copy(REPO_ROOT / "workflows/image_krea2.api.json", Path(temporary) / "image_krea2.api.json")
            previous = os.environ.get("WORKFLOW_DIR")
            os.environ["WORKFLOW_DIR"] = temporary
            try:
                prompt, bindings = bind_workflow(
                    "image_krea2",
                    {
                        "positive_prompt": "test",
                        "seed": 7,
                        "width": 832,
                        "height": 480,
                        "steps": 8,
                        "guidance": 3.5,
                        "output_prefix": "images/x/krea2",
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
        self.assertEqual(prompt["7"]["inputs"]["steps"], 8)
        self.assertEqual(prompt["6"]["inputs"]["width"], 832)
        self.assertEqual(
            set(bindings),
            {"positive_prompt", "seed", "width", "height", "steps", "output_prefix", "main_model_target"},
        )


if __name__ == "__main__":
    unittest.main()
