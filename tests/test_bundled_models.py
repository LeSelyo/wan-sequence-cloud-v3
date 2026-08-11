import hashlib
import json
import os
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from app.readiness import check_model_items
from app.settings import get_settings
from scripts.materialize_bundled_models import materialize_models
from scripts.prepare_bundled_models import (
    BUNDLE_SCHEMA_VERSION,
    generate_dockerfile,
    prepare_bundle,
    split_model,
)


DiskUsage = namedtuple("DiskUsage", "total used free")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_manifest(root: Path, data: bytes, chunk_size: int = 5):
    bundle = root / "bundle"
    parts_dir = bundle / "models/fake"
    source = root / "source.bin"
    source.write_bytes(data)
    parts = split_model(source, parts_dir, chunk_size)
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "profile": "fixture",
        "chunk_size_bytes": chunk_size,
        "models": {
            "fake": {
                "catalog_id": "fake",
                "filename": "fake.safetensors",
                "target": "diffusion_models/fake.safetensors",
                "size_bytes": len(data),
                "sha256": digest(data),
                "revision": "fixture",
                "source_url": "https://example.invalid/fake",
                "parts": [
                    {**part, "path": str((parts_dir / part["name"]).resolve())}
                    for part in parts
                ],
            }
        },
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


class SplitAndBuildGenerationTests(unittest.TestCase):
    def test_split_manifest_order_and_lossless_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = bytes(range(31))
            manifest_path, manifest = fixture_manifest(root, data, 7)
            parts = manifest["models"]["fake"]["parts"]
            self.assertEqual([part["name"] for part in parts], ["part.000", "part.001", "part.002", "part.003", "part.004"])
            reconstructed = b"".join(Path(part["path"]).read_bytes() for part in parts)
            self.assertEqual(reconstructed, data)
            self.assertEqual(digest(reconstructed), manifest["models"]["fake"]["sha256"])
            self.assertEqual([part["size_bytes"] for part in parts], [7, 7, 7, 7, 3])
            self.assertEqual([part["sha256"] for part in parts], [digest(data[i:i + 7]) for i in range(0, len(data), 7)])
            self.assertTrue(manifest_path.is_file())

    def test_generated_dockerfile_has_one_copy_layer_per_chunk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, manifest = fixture_manifest(root, b"0123456789abcdef", 4)
            base = root / "Dockerfile"
            generated = root / "Dockerfile.bundled"
            base.write_text("FROM scratch\n", encoding="utf-8")
            generate_dockerfile(base, generated, manifest)
            text = generated.read_text(encoding="utf-8")
            chunk_lines = [line for line in text.splitlines() if line.startswith("COPY --link") and "part." in line]
            self.assertEqual(len(chunk_lines), 4)
            self.assertTrue(all(line.count("part.") == 2 for line in chunk_lines))
            self.assertNotIn("COPY --link .bundled-models/parts/ ", text)

    def test_prepare_bundle_reuses_catalog_profile_and_removes_complete_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = b"small model fixture"
            output = root / ".bundled-models"
            source = output / "work/fake/fake.safetensors"
            source.parent.mkdir(parents=True)
            source.write_bytes(data)
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "profiles": {"t2v": ["fake"]},
                "items": {"fake": {
                    "relative_path": "diffusion_models/fake.safetensors",
                    "url": "https://example.invalid/fake",
                    "size_bytes": len(data),
                    "sha256": digest(data),
                    "revision": "fixture",
                }},
            }), encoding="utf-8")
            base = root / "Dockerfile"
            generated = root / "Dockerfile.bundled"
            base.write_text("FROM scratch\n", encoding="utf-8")
            with mock.patch("scripts.prepare_bundled_models.download_resumable") as download:
                manifest = prepare_bundle(catalog, "t2v", output, 6, base, generated)
            download.assert_not_called()
            self.assertFalse((output / "work").exists())
            self.assertEqual(list(manifest["models"]), ["fake"])
            self.assertNotIn("fake.safetensors", generated.read_text(encoding="utf-8").split("# Generated", 1)[1])
            before = (output / "manifest.json").read_bytes()
            with mock.patch("scripts.prepare_bundled_models.download_resumable") as second_download:
                second = prepare_bundle(catalog, "t2v", output, 6, base, generated)
            second_download.assert_not_called()
            self.assertEqual(second, manifest)
            self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_registry_cache_never_sees_complete_model_source(self):
        project = Path(__file__).resolve().parent.parent
        dockerignore = (project / ".dockerignore").read_text(encoding="utf-8")
        build = (project / "scripts/build-and-push.sh").read_text(encoding="utf-8")
        self.assertIn(".bundled-models/*", dockerignore)
        self.assertIn("!.bundled-models/parts/**", dockerignore)
        self.assertNotIn("!.bundled-models/work", dockerignore)
        self.assertIn("scripts/prepare_bundled_models.py", build)
        self.assertIn("--file Dockerfile.bundled", build)
        self.assertIn("--cache-to", build)


class MaterializationTests(unittest.TestCase):
    def test_absent_target_is_materialized_atomically_and_second_run_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = b"lossless bundled model"
            manifest_path, _ = fixture_manifest(root, data)
            data_root = root / "workspace"
            data_root.mkdir()
            result = materialize_models(manifest_path, data_root, reserve_bytes=0)
            target = data_root / "models/diffusion_models/fake.safetensors"
            self.assertEqual(target.read_bytes(), data)
            self.assertFalse(target.with_suffix(".safetensors.part").exists())
            modified = target.stat().st_mtime_ns
            result2 = materialize_models(manifest_path, data_root, reserve_bytes=0)
            self.assertEqual(result, {"fake": "materialized"})
            self.assertEqual(result2, {"fake": "already_valid"})
            self.assertEqual(target.stat().st_mtime_ns, modified)
            installed = json.loads((data_root / "downloads/installed-files.json").read_text(encoding="utf-8"))
            self.assertEqual(installed["sections"]["base_models"]["fake"]["installed_from"], "bundled-image")

    def test_wrong_target_hash_is_reconstructed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = b"correct bytes"
            manifest_path, _ = fixture_manifest(root, data)
            data_root = root / "workspace"
            target = data_root / "models/diffusion_models/fake.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"X" * len(data))
            materialize_models(manifest_path, data_root, reserve_bytes=0)
            self.assertEqual(target.read_bytes(), data)

    def test_missing_or_corrupt_chunk_fails_clearly(self):
        for corruption in ("missing", "hash"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = fixture_manifest(root, b"chunk validation fixture")
                chunk = Path(manifest["models"]["fake"]["parts"][0]["path"])
                chunk.unlink() if corruption == "missing" else chunk.write_bytes(b"bad!!")
                data_root = root / "workspace"
                data_root.mkdir()
                with self.assertRaisesRegex(RuntimeError, "bundled chunk.*invalid"):
                    materialize_models(manifest_path, data_root, reserve_bytes=0)

    def test_insufficient_space_fails_before_partial_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = fixture_manifest(root, b"space fixture")
            data_root = root / "workspace"
            data_root.mkdir()
            with self.assertRaisesRegex(RuntimeError, "insufficient free disk space"):
                materialize_models(
                    manifest_path,
                    data_root,
                    reserve_bytes=10,
                    disk_usage=lambda _: DiskUsage(100, 99, 1),
                )
            self.assertFalse((data_root / "models/diffusion_models/fake.safetensors.part").exists())

    def test_copy_failure_never_promotes_partial_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = fixture_manifest(root, b"atomic failure fixture")
            data_root = root / "workspace"
            data_root.mkdir()
            def fail(_block):
                raise OSError("simulated write failure")
            with self.assertRaisesRegex(OSError, "simulated"):
                materialize_models(manifest_path, data_root, reserve_bytes=0, copy_hook=fail)
            target = data_root / "models/diffusion_models/fake.safetensors"
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".safetensors.part").exists())

    def test_readiness_rejects_same_size_wrong_hash(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"DATA_ROOT": temporary}):
            settings = get_settings()
            expected = b"good"
            spec = {
                "relative_path": "diffusion_models/fake.safetensors",
                "size_bytes": len(expected),
                "sha256": digest(expected),
                "revision": "fixture",
            }
            target = settings.models_dir / spec["relative_path"]
            target.parent.mkdir(parents=True)
            target.write_bytes(b"baad")
            settings.downloads_dir.mkdir(parents=True)
            (settings.downloads_dir / "installed-files.json").write_text(
                json.dumps({"sections": {"base_models": {"fake": {
                    "size_bytes": len(expected),
                    "sha256": digest(expected),
                    "revision": "fixture",
                }}}}),
                encoding="utf-8",
            )
            with mock.patch("app.readiness._catalog", return_value={"items": {"fake": spec}}):
                ready, details = check_model_items(settings, ["fake"])
            self.assertFalse(ready)
            self.assertEqual(details["fake"], "sha256_mismatch")


if __name__ == "__main__":
    unittest.main()
