import asyncio
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from app.comfy import bind_workflow
from app.schemas import ImageGenerationRequest, SequenceRequest, Shot
from scripts.materialize_bundled_models import materialize_models
from scripts.prepare_bundled_models import DEFAULT_CHUNK_SIZE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLUX_IDS = ["flux_schnell_diffusion", "flux_clip_l", "flux_t5xxl_fp8", "flux_ae"]
FLUX_METADATA = {
    "flux_schnell_diffusion": (
        "diffusion_models/flux1-schnell.safetensors",
        "f757664cb3ff8b19ff99e064a2387a5f547ad9e8",
        23_782_506_688,
        "9403429e0052277ac2a87ad800adece5481eecefd9ed334e1f348723621d2a0a",
    ),
    "flux_clip_l": (
        "text_encoders/clip_l.safetensors",
        "2f74b39c0606dae3b2196d79c18c2a40b71f3250",
        246_144_152,
        "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
    ),
    "flux_t5xxl_fp8": (
        "text_encoders/t5xxl_fp8_e4m3fn.safetensors",
        "2f74b39c0606dae3b2196d79c18c2a40b71f3250",
        4_893_934_904,
        "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09",
    ),
    "flux_ae": (
        "vae/ae.safetensors",
        "817f3ba14d96a306cf0cbed49ef9eea037991381",
        335_304_388,
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
    ),
}


class FluxCatalogTests(unittest.TestCase):
    def test_flux_assets_are_verified_base_models_and_all_extends_all_video(self):
        catalog = json.loads((PROJECT_ROOT / "config/base_models.json").read_text(encoding="utf-8"))
        self.assertEqual(catalog["profiles"]["flux-schnell"], FLUX_IDS)
        self.assertEqual(catalog["profiles"]["image"], FLUX_IDS)
        self.assertNotEqual(catalog["profiles"]["all"], catalog["profiles"]["all-video"])
        self.assertEqual(catalog["profiles"]["all"], catalog["profiles"]["all-video"] + FLUX_IDS)
        for item_id in FLUX_IDS:
            spec = catalog["items"][item_id]
            self.assertEqual(spec["kind"], "base_model")
            self.assertEqual(spec["engine"], "flux_schnell")
            self.assertEqual(spec["source"], "huggingface")
            self.assertGreater(spec["size_bytes"], 0)
            self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")
            self.assertNotIn("/loras/", spec["url"])
            self.assertFalse(spec["relative_path"].startswith("loras/"))
            self.assertEqual(
                (spec["relative_path"], spec["revision"], spec["size_bytes"], spec["sha256"]),
                FLUX_METADATA[item_id],
            )
        self.assertFalse(any("lightx" in item for item in catalog["profiles"]["all"]))
        self.assertEqual(
            catalog["profile_capabilities"]["all-video"],
            ["t2v", "i2v", "flf2v"],
        )
        self.assertEqual(
            catalog["profile_capabilities"]["all"],
            ["t2v", "i2v", "flf2v", "image_generation"],
        )

    def test_flux_diffusion_uses_the_existing_four_gib_split_policy(self):
        self.assertEqual(DEFAULT_CHUNK_SIZE, 4 * 1024**3)
        diffusion_size = FLUX_METADATA["flux_schnell_diffusion"][2]
        self.assertEqual(math.ceil(diffusion_size / DEFAULT_CHUNK_SIZE), 6)

    def test_wan_umt5_is_not_reused_as_flux_t5(self):
        catalog = json.loads((PROJECT_ROOT / "config/base_models.json").read_text(encoding="utf-8"))
        self.assertNotEqual(
            catalog["items"]["umt5_xxl"]["relative_path"],
            catalog["items"]["flux_t5xxl_fp8"]["relative_path"],
        )


class FluxWorkflowTests(unittest.TestCase):
    def test_native_api_workflow_has_required_loaders_sampling_decode_and_save(self):
        workflow = json.loads((PROJECT_ROOT / "workflows/image_flux_schnell.api.json").read_text(encoding="utf-8"))
        classes = {node["class_type"] for key, node in workflow.items() if not key.startswith("_")}
        for expected in (
            "UNETLoader", "DualCLIPLoader", "VAELoader", "CLIPTextEncodeFlux",
            "ConditioningZeroOut", "EmptySD3LatentImage", "KSampler", "VAEDecode", "SaveImage",
        ):
            self.assertIn(expected, classes)
        self.assertNotIn("CheckpointLoaderSimple", classes)
        sampler = next(node for node in workflow.values() if isinstance(node, dict) and node.get("class_type") == "KSampler")
        self.assertEqual((sampler["inputs"]["steps"], sampler["inputs"]["cfg"]), (4, 1.0))
        bindings = workflow["_bindings"]
        for name in ("positive_prompt", "seed", "width", "height", "steps", "guidance", "output_prefix"):
            self.assertIn(name, bindings)
        self.assertNotIn("negative_prompt", bindings)

    def test_bind_workflow_applies_non_default_flux_values(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"WORKFLOW_DIR": temporary}
        ):
            target = Path(temporary) / "image_flux_schnell.api.json"
            target.write_bytes((PROJECT_ROOT / "workflows/image_flux_schnell.api.json").read_bytes())
            prompt, _ = bind_workflow("image_flux_schnell", {
                "positive_prompt": "coastal city",
                "seed": 123,
                "width": 768,
                "height": 512,
                "steps": 3,
                "guidance": 2.25,
                "output_prefix": "images/test",
            })
        self.assertEqual(prompt["4"]["inputs"]["clip_l"], "coastal city")
        self.assertEqual(prompt["4"]["inputs"]["t5xxl"], "coastal city")
        self.assertEqual(prompt["7"]["inputs"]["seed"], 123)
        self.assertEqual(prompt["7"]["inputs"]["steps"], 3)
        self.assertEqual(prompt["7"]["inputs"]["cfg"], 1.0)
        self.assertEqual((prompt["6"]["inputs"]["width"], prompt["6"]["inputs"]["height"]), (768, 512))

    def test_flux_model_categories_materialize_byte_for_byte(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            models = {}
            for item_id, relative, data in (
                ("model", "diffusion_models/flux.bin", b"flux"),
                ("clip", "text_encoders/clip.bin", b"clip"),
                ("t5", "text_encoders/t5.bin", b"t5"),
                ("ae", "vae/ae.bin", b"ae"),
            ):
                part = bundle / "models" / item_id / "part.000"
                part.parent.mkdir(parents=True, exist_ok=True)
                part.write_bytes(data)
                models[item_id] = {
                    "catalog_id": item_id,
                    "target": relative,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "parts": [{"path": str(part.resolve()), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
                }
            manifest = bundle / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "profile": "image", "models": models}), encoding="utf-8")
            data_root = root / "workspace"
            data_root.mkdir()
            materialize_models(manifest, data_root, reserve_bytes=0)
            self.assertEqual((data_root / "models/diffusion_models/flux.bin").read_bytes(), b"flux")
            self.assertEqual((data_root / "models/text_encoders/clip.bin").read_bytes(), b"clip")
            self.assertEqual((data_root / "models/text_encoders/t5.bin").read_bytes(), b"t5")
            self.assertEqual((data_root / "models/vae/ae.bin").read_bytes(), b"ae")


class FluxApiContractTests(unittest.TestCase):
    def test_flux_request_contract_and_arbitrary_model_url_rejection(self):
        request = ImageGenerationRequest.model_validate({"prompt": "city", "steps": 4, "loras": []})
        self.assertEqual(request.engine, "flux_schnell")
        with self.assertRaises(ValidationError):
            ImageGenerationRequest.model_validate({"prompt": "city", "negative_prompt": "bad"})
        with self.assertRaises(ValidationError):
            ImageGenerationRequest.model_validate({"prompt": "city", "model_url": "https://evil.invalid/model"})
        with self.assertRaises(ValidationError):
            ImageGenerationRequest.model_validate({
                "prompt": "city",
                "loras": [{"id": "flux-style", "target": "high"}],
            })

    def test_standalone_flux_lora_is_validated_downloaded_and_injected(self):
        from app.orchestrator import generate_image_file

        request = ImageGenerationRequest.model_validate({
            "prompt": "city",
            "loras": [{"id": "flux-style", "weight": 0.7}],
        })
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            comfy_output = output_dir / "comfy"
            comfy_output.mkdir()
            (comfy_output / "generated.png").write_bytes(b"png")
            fake_settings = mock.Mock()
            fake_settings.comfy_output_dir = comfy_output
            prompt = {"1": {"class_type": "UNETLoader", "inputs": {}}}
            history = {"outputs": {"9": {"images": [{"filename": "generated.png", "subfolder": ""}]}}}
            with (
                mock.patch("app.orchestrator.settings", fake_settings),
                mock.patch("app.orchestrator.validate_image_generation_compatibility"),
                mock.patch("app.orchestrator.lora_entry", return_value={"filename": "flux-style.safetensors"}),
                mock.patch("app.orchestrator.validate_lora_for_stage") as validate,
                mock.patch("app.orchestrator.ensure_lora") as ensure,
                mock.patch("app.orchestrator.bind_workflow", return_value=(prompt, {})),
                mock.patch("app.orchestrator.inject_loras") as inject,
                mock.patch("app.orchestrator.queue_and_wait", new=mock.AsyncMock(return_value=history)),
            ):
                target, _ = asyncio.run(generate_image_file(request, output_dir, "images/test"))
                target_bytes = target.read_bytes()
        validate.assert_called_once_with(mock.ANY, "flux_schnell", "keyframe")
        ensure.assert_called_once()
        inject.assert_called_once_with(
            prompt,
            {},
            [{"filename": "flux-style.safetensors", "weight": 0.7}],
            "main",
        )
        self.assertEqual(target_bytes, b"png")

    def test_i2v_external_generated_and_ambiguous_inputs(self):
        external = Shot.model_validate({"id": "external", "mode": "i2v", "start_image": {"path": "input.png"}})
        generated = Shot.model_validate({"id": "generated", "mode": "i2v", "generate_start_image": {"engine": "flux_schnell", "prompt": "cat"}})
        self.assertEqual(external.start_image.path, "input.png")
        self.assertEqual(generated.generate_start_image.engine, "flux_schnell")
        with self.assertRaises(ValidationError):
            Shot.model_validate({"id": "missing", "mode": "i2v"})
        with self.assertRaises(ValidationError):
            Shot.model_validate({
                "id": "ambiguous", "mode": "i2v",
                "start_image": {"path": "input.png"},
                "generate_start_image": {"engine": "flux_schnell", "prompt": "cat"},
            })

    def test_internal_image_reference_rejects_traversal_and_arbitrary_paths(self):
        with self.assertRaises(ValidationError):
            Shot.model_validate({"id": "bad", "mode": "i2v", "start_image": {"image_id": "../../etc/passwd"}})
        with self.assertRaises(ValidationError):
            Shot.model_validate({"id": "bad", "mode": "i2v", "start_image": {"path": "../../etc/passwd"}})

    def test_generate_start_image_calls_flux_generator(self):
        from app.orchestrator import _generate_keyframe

        shot = Shot.model_validate({
            "id": "generated", "mode": "i2v", "width": 768, "height": 512,
            "generate_start_image": {"engine": "flux_schnell", "prompt": "cat", "steps": 3},
        })
        expected = Path("controlled/generated_start.png")
        with mock.patch(
            "app.orchestrator.generate_image_file",
            new=mock.AsyncMock(return_value=(expected, 42)),
        ) as generate:
            result = asyncio.run(_generate_keyframe(shot, Path("controlled")))
        self.assertEqual(result, expected)
        self.assertEqual(generate.await_args.kwargs["width"], 768)
        self.assertEqual(generate.await_args.kwargs["height"], 512)
        self.assertEqual(generate.await_args.args[0].engine, "flux_schnell")


if __name__ == "__main__":
    unittest.main()
