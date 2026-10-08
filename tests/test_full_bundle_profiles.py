import asyncio
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.readiness import check_capabilities
from app.schemas import Shot
from app.settings import ensure_data_directories, get_settings
from scripts.materialize_bundled_models import materialize_models
from scripts.prepare_bundled_models import generate_dockerfile, prepare_bundle


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_installed(settings, items: dict[str, dict], payloads: dict[str, bytes]) -> None:
    records = {}
    for item_id, data in payloads.items():
        spec = items[item_id]
        target = settings.models_dir / spec["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records[item_id] = {
            "size_bytes": len(data),
            "sha256": sha(data),
            "revision": spec.get("revision"),
            "verified_mtime_ns": target.stat().st_mtime_ns,
        }
    settings.downloads_dir.mkdir(parents=True, exist_ok=True)
    (settings.downloads_dir / "installed-files.json").write_text(
        json.dumps({"sections": {"base_models": records}}), encoding="utf-8"
    )


class CatalogProfileTests(unittest.TestCase):
    def test_video_profiles_are_deduplicated_and_exclude_images_and_loras(self):
        catalog = json.loads((PROJECT_ROOT / "config/base_models.json").read_text(encoding="utf-8"))
        profiles = catalog["profiles"]
        expected = list(dict.fromkeys(profiles["t2v"] + profiles["i2v"] + profiles["flf2v"]))
        self.assertEqual(set(profiles["all-video"]), set(expected))
        self.assertEqual(len(profiles["all-video"]), len(set(profiles["all-video"])))
        self.assertTrue(all(not catalog["items"][item]["relative_path"].startswith("checkpoints/") for item in profiles["all-video"]))
        self.assertTrue(all(not catalog["items"][item]["relative_path"].startswith("loras/") for item in profiles["all-video"]))

    def test_all_is_maximum_actually_supported_profile_without_dynamic_loras(self):
        catalog = json.loads((PROJECT_ROOT / "config/base_models.json").read_text(encoding="utf-8"))
        supported_image_ids = [
            item_id
            for engine in catalog["image_engines"].values()
            if engine["supported"]
            for item_id in engine["model_ids"]
        ]
        expected = list(dict.fromkeys(catalog["profiles"]["all-video"] + supported_image_ids))
        self.assertEqual(catalog["profiles"]["all"], expected)
        self.assertTrue(all(not catalog["items"][item]["relative_path"].startswith("loras/") for item in expected))
        self.assertFalse(any("lightx" in item for item in expected))
        self.assertEqual(
            supported_image_ids,
            [
                "flux_schnell_diffusion", "flux_clip_l", "flux_t5xxl_fp8", "flux_ae",
                "krea2_turbo_diffusion", "krea2_text_encoder", "krea2_vae",
            ],
        )


class GenericImageBundleTests(unittest.TestCase):
    def test_t2v_all_video_and_all_generate_statically_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps({
                "profiles": {"t2v": ["t"], "all-video": ["t", "i"], "image": ["image"], "all": ["t", "i", "image"]},
                "items": {
                    "t": {"relative_path": "diffusion_models/t.bin", "url": "https://example.invalid/t", "size_bytes": 1, "sha256": sha(b"t"), "revision": "r"},
                    "i": {"relative_path": "diffusion_models/i.bin", "url": "https://example.invalid/i", "size_bytes": 1, "sha256": sha(b"i"), "revision": "r"},
                    "image": {"relative_path": "checkpoints/image.safetensors", "url": "https://example.invalid/image", "size_bytes": 1, "sha256": sha(b"x"), "revision": "r"},
                },
            }), encoding="utf-8")
            base = root / "Dockerfile"
            base.write_text("FROM scratch\n", encoding="utf-8")
            for profile, selected in (("t2v", {"t": b"t"}), ("all-video", {"t": b"t", "i": b"i"}), ("image", {"image": b"x"}), ("all", {"t": b"t", "i": b"i", "image": b"x"})):
                output = root / profile
                for item_id, data in selected.items():
                    name = "image.safetensors" if item_id == "image" else f"{item_id}.bin"
                    source = output / "work" / item_id / name
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_bytes(data)
                generated = root / f"Dockerfile.{profile}"
                with mock.patch("scripts.prepare_bundled_models.download_resumable") as download:
                    manifest = prepare_bundle(catalog_path, profile, output, 1, base, generated)
                download.assert_not_called()
                self.assertEqual(list(manifest["models"]), list(selected))
                self.assertTrue(generated.is_file())

    def test_all_profile_can_split_and_materialize_a_declared_image_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video, checkpoint = b"video", b"image-checkpoint"
            output = root / ".bundled-models"
            for item, data, name in (("video", video, "video.bin"), ("image", checkpoint, "image.safetensors")):
                source = output / "work" / item / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(data)
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "profiles": {"all": ["video", "image"]},
                "items": {
                    "video": {"relative_path": "diffusion_models/video.bin", "url": "https://example.invalid/video", "size_bytes": len(video), "sha256": sha(video), "revision": "r"},
                    "image": {"relative_path": "checkpoints/image.safetensors", "url": "https://example.invalid/image", "size_bytes": len(checkpoint), "sha256": sha(checkpoint), "revision": "r"},
                },
            }), encoding="utf-8")
            base, generated = root / "Dockerfile", root / "Dockerfile.bundled"
            base.write_text("FROM scratch\n", encoding="utf-8")
            with mock.patch("scripts.prepare_bundled_models.download_resumable") as download:
                manifest = prepare_bundle(catalog, "all", output, 4, base, generated)
            download.assert_not_called()
            # Runtime manifests use immutable in-image paths; point this fixture at its local chunks.
            for item_id, model in manifest["models"].items():
                for part in model["parts"]:
                    part["path"] = str((output / "parts" / item_id / part["name"]).resolve())
            manifest_path = output / "manifest.runtime.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            data_root = root / "workspace"
            data_root.mkdir()
            materialize_models(manifest_path, data_root, reserve_bytes=0)
            self.assertEqual((data_root / "models/checkpoints/image.safetensors").read_bytes(), checkpoint)
            dockerfile = generated.read_text(encoding="utf-8")
            copy_lines = [line for line in dockerfile.splitlines() if line.startswith("COPY --link") and "part." in line]
            self.assertEqual(len(copy_lines), sum(len(model["parts"]) for model in manifest["models"].values()))
            self.assertNotIn("COPY --link .bundled-models/parts/ ", dockerfile)


class CapabilityAndKeyframeTests(unittest.TestCase):
    def test_readiness_distinguishes_t2v_and_image_and_supports_declared_all(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"DATA_ROOT": temporary}):
            settings = get_settings()
            ensure_data_directories(settings)
            payloads = {"t": b"t", "i": b"i", "f": b"f", "image": b"checkpoint"}
            items = {
                key: {"relative_path": ("checkpoints/image.safetensors" if key == "image" else f"diffusion_models/{key}.bin"), "size_bytes": len(data), "sha256": sha(data), "revision": "r"}
                for key, data in payloads.items()
            }
            catalog = {
                "profiles": {"t2v": ["t"], "i2v": ["i"], "flf2v": ["f"], "all": ["t", "i", "f", "image"]},
                "profile_capabilities": {"t2v": ["t2v"], "all": ["t2v", "i2v", "flf2v", "image_generation"]},
                "image_engines": {"sd15": {"workflow": "image_sd15.api.json", "model_ids": ["image"], "supported": True}},
                "items": items,
            }
            write_installed(settings, items, {"t": payloads["t"]})
            (settings.workflow_dir / "wan22_t2v.api.json").write_text("{}", encoding="utf-8")
            with mock.patch("app.readiness._catalog", return_value=catalog):
                t2v = check_capabilities(settings, "t2v")
            self.assertTrue(t2v["t2v"]["ready"])
            self.assertFalse(t2v["image_generation"]["ready"])
            self.assertTrue(t2v["required_ready"])

            write_installed(settings, items, payloads)
            for workflow in ("wan22_i2v.api.json", "wan22_flf2v.api.json", "image_sd15.api.json"):
                (settings.workflow_dir / workflow).write_text("{}", encoding="utf-8")
            with mock.patch("app.readiness._catalog", return_value=catalog):
                complete = check_capabilities(settings, "all")
            self.assertTrue(complete["required_ready"])
            self.assertTrue(all(complete[name]["ready"] for name in ("t2v", "i2v", "flf2v", "image_generation")))

    def test_generate_keyframe_refuses_current_undeclared_engine_cleanly(self):
        from app.orchestrator import _generate_keyframe

        shot = Shot.model_validate({"id": "shot", "mode": "i2v", "generate_start_image": {"engine": "sd15", "prompt": "cat"}})
        with self.assertRaisesRegex(ValueError, "not supported by this build"):
            asyncio.run(_generate_keyframe(shot, Path("job")))

    def test_generate_keyframe_uses_a_materialized_declared_checkpoint(self):
        from app.orchestrator import _generate_keyframe

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"DATA_ROOT": temporary}):
            settings = get_settings()
            ensure_data_directories(settings)
            checkpoint = b"checkpoint"
            spec = {"relative_path": "checkpoints/image.safetensors", "size_bytes": len(checkpoint), "sha256": sha(checkpoint), "revision": "r"}
            write_installed(settings, {"image": spec}, {"image": checkpoint})
            output = settings.comfy_output_dir / "generated.png"
            output.write_bytes(b"png")
            job_dir = settings.outputs_dir / "job"
            job_dir.mkdir()
            (settings.workflow_dir / "image_flux_schnell.api.json").write_text("{}", encoding="utf-8")
            shot = Shot.model_validate({"id": "shot", "mode": "i2v", "generate_start_image": {"engine": "flux_schnell", "prompt": "cat"}})
            prompt = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "image.safetensors"}}}
            engine = {"workflow": "image_flux_schnell.api.json", "model_ids": ["image"], "supported": True}
            history = {"outputs": {"2": {"images": [{"filename": "generated.png", "subfolder": ""}]}}}
            with (
                mock.patch("app.orchestrator.settings", settings),
                mock.patch("app.orchestrator.image_engine_spec", return_value=engine),
                mock.patch("app.orchestrator.check_model_items", return_value=(True, {"image": "ready"})),
                mock.patch("app.orchestrator.bind_workflow", return_value=(prompt, {})),
                mock.patch("app.orchestrator.queue_and_wait", new=mock.AsyncMock(return_value=history)),
            ):
                result = asyncio.run(_generate_keyframe(shot, job_dir))
            self.assertEqual(result.read_bytes(), b"png")


if __name__ == "__main__":
    unittest.main()


class ModelFoldersAreDeclaredToComfyUI(unittest.TestCase):
    def test_every_folder_the_catalogue_downloads_into_is_declared_in_extra_model_paths(self):
        """A model in a folder ComfyUI does not know is downloaded but INVISIBLE (AudioEncoderLoader listed no model for wav2vec2 until `audio_encoders` was declared)."""
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        catalog = json.loads((root / "config" / "base_models.json").read_text(encoding="utf-8"))
        declared = {line.split(":", 1)[1].strip() for line in (root / "config" / "extra_model_paths.yaml").read_text(encoding="utf-8").splitlines()
                    if line.startswith("  ") and ":" in line and "base_path" not in line}
        folders = {spec["relative_path"].split("/")[0] for spec in catalog["items"].values()}
        self.assertEqual(sorted(folders - declared), [])
