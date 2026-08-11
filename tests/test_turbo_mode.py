import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from app.readiness import check_models, check_turbo_assets
from app.schemas import Shot
from app.settings import get_settings
from scripts import download_base_models


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TurboSchemaTests(unittest.TestCase):
    def test_turbo_defaults_are_fixture_safe_and_standard_is_unchanged(self):
        standard = Shot.model_validate({"id": "standard", "mode": "t2v", "prompt": "x"})
        turbo = Shot.model_validate({"id": "turbo", "mode": "t2v", "prompt": "x", "turbo_mode": True})
        self.assertFalse(standard.turbo_mode)
        self.assertEqual((standard.steps, standard.cfg), (20, 5.0))
        self.assertEqual((turbo.steps, turbo.cfg), (4, 1.0))

    def test_turbo_rejects_incompatible_overrides_and_flf2v(self):
        for payload in (
            {"id": "steps", "mode": "t2v", "prompt": "x", "turbo_mode": True, "steps": 5},
            {"id": "cfg", "mode": "t2v", "prompt": "x", "turbo_mode": True, "cfg": 2.0},
            {
                "id": "flf",
                "mode": "t+i(keyframe)2v",
                "prompt": "x",
                "turbo_mode": True,
                "start_image": {"path": "start.png"},
                "end_image": {"path": "end.png"},
            },
        ):
            with self.subTest(payload=payload["id"]), self.assertRaises(ValidationError):
                Shot.model_validate(payload)


class TurboCatalogTests(unittest.TestCase):
    def test_official_catalog_keeps_standard_and_turbo_profiles_separate(self):
        catalog = json.loads((PROJECT_ROOT / "config/base_models.json").read_text(encoding="utf-8"))
        for family in ("t2v", "i2v"):
            standard = catalog["profiles"][family]
            turbo = catalog["profiles"][f"{family}-turbo"]
            lightx = [item for item in turbo if "lightx" in item]
            self.assertEqual(len(lightx), 2)
            self.assertFalse(any("lightx" in item for item in standard))
            self.assertTrue(set(standard).issubset(turbo))
            for item_id in lightx:
                spec = catalog["items"][item_id]
                self.assertEqual(spec["size_bytes"], 1226977424)
                self.assertEqual(len(spec["sha256"]), 64)
                self.assertTrue(spec["relative_path"].startswith("loras/"))

    def test_downloader_selects_only_requested_profile_assets(self):
        catalog = {
            "profiles": {"t2v": ["base"], "t2v-turbo": ["base", "high", "low"]},
            "items": {"base": {}, "high": {}, "low": {}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            with mock.patch.object(download_base_models, "CATALOG", path):
                standard = download_base_models.parse_args(["--profile", "t2v", "--check"])
                turbo = download_base_models.parse_args(["--profile", "t2v-turbo", "--check"])
        self.assertEqual(download_base_models.select_ids(standard, catalog), ["base"])
        self.assertEqual(download_base_models.select_ids(turbo, catalog), ["base", "high", "low"])

    def test_turbo_readiness_validates_files_and_manifest_without_affecting_standard(self):
        catalog = {
            "profiles": {"t2v": ["base"], "t2v-turbo": ["base", "high", "low"]},
            "items": {
                "base": {"relative_path": "diffusion_models/base.bin", "size_bytes": 1, "revision": "r"},
                "high": {"relative_path": "loras/high.bin", "size_bytes": 1, "revision": "r"},
                "low": {"relative_path": "loras/low.bin", "size_bytes": 1, "revision": "r"},
            },
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"DATA_ROOT": temporary}):
            settings = get_settings()
            for relative in ("diffusion_models/base.bin", "loras/high.bin", "loras/low.bin"):
                path = settings.models_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            manifest = settings.downloads_dir / "installed-files.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"sections": {"base_models": {
                "base": {"size_bytes": 1, "revision": "r"}
            }, "loras": {
                item_id: {"size_bytes": 1, "revision": "r"} for item_id in ("high", "low")
            }}}), encoding="utf-8")
            lora_catalog = {"items": {
                "high": {"filename": "high.bin", "size_bytes": 1, "revision": "r"},
                "low": {"filename": "low.bin", "size_bytes": 1, "revision": "r"},
            }}
            with (
                mock.patch("app.readiness._catalog", return_value=catalog),
                mock.patch("app.readiness._lora_catalog", return_value=lora_catalog),
                mock.patch("app.readiness.TURBO_ITEMS", {"wan22_t2v": ("high", "low")}),
            ):
                self.assertTrue(check_models(settings, "t2v")[0])
                self.assertTrue(check_turbo_assets(settings, "wan22_t2v")[0])
                (settings.models_dir / "loras/high.bin").unlink()
                self.assertTrue(check_models(settings, "t2v")[0])
                ready, details = check_turbo_assets(settings, "wan22_t2v")
        self.assertFalse(ready)
        self.assertEqual(details["high"], "missing")


if __name__ == "__main__":
    unittest.main()
