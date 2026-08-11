import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.mini_bundle_preflight import MODEL_TARGETS, run_fixture
from scripts.prepare_bundled_models import DEFAULT_CHUNK_SIZE


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _msys_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    suffix = resolved.as_posix().split(":", 1)[-1]
    return f"/{drive}{suffix}" if drive else resolved.as_posix()


class BundleReleasePreflightTests(unittest.TestCase):
    def test_fake_full_bundle_uses_real_generator_and_materializes_idempotently(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            result = run_fixture(Path(temporary), docker=False)
        self.assertEqual(result["models"], 10)
        self.assertGreaterEqual(result["parts"], 20)
        self.assertEqual(set(result["first"].values()), {"materialized"})
        self.assertEqual(set(result["second"].values()), {"already_valid"})
        self.assertEqual(set(result["first"]), set(MODEL_TARGETS))

    def test_production_all_profile_is_29_independent_four_gib_chunks(self):
        catalog = json.loads(
            (PROJECT_ROOT / "config/base_models.json").read_text(encoding="utf-8")
        )
        expected = sum(
            math.ceil(catalog["items"][item]["size_bytes"] / DEFAULT_CHUNK_SIZE)
            for item in catalog["profiles"]["all"]
        )
        self.assertEqual(expected, 29)
        self.assertEqual(DEFAULT_CHUNK_SIZE, 4_294_967_296)

    def test_generated_dockerfile_repairs_directories_without_merging_chunk_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_fixture(root, docker=False)
            dockerfile = (root / "Dockerfile.bundled").read_text(encoding="utf-8")
        copies = [
            line
            for line in dockerfile.splitlines()
            if line.startswith("COPY --link") and "part." in line
        ]
        self.assertEqual(len(copies), result["parts"])
        self.assertTrue(all("--chmod=0444" in line for line in copies))
        self.assertIn(
            "find /opt/wan-model-parts -type d -exec chmod 0555", dockerfile
        )


class EntrypointReleasePreflightTests(unittest.TestCase):
    @unittest.skipUnless(
        Path(r"C:\Program Files\Git\bin\bash.exe").is_file(),
        "Git Bash is required for the arbitrary-CWD entrypoint preflight",
    )
    def test_entrypoint_imports_app_and_preserves_environment_from_arbitrary_cwd(self):
        bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        uvicorn = shutil.which("uvicorn")
        self.assertIsNotNone(uvicorn)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            data_root = Path(temporary) / "workspace"
            for relative in (
                "models/diffusion_models",
                "models/text_encoders",
                "models/vae",
                "models/checkpoints",
                "models/loras",
                "inputs/comfy",
                "outputs/images",
                "outputs/comfy",
                "jobs",
                "cache/workflows",
                "downloads",
            ):
                (data_root / relative).mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(
                {
                    "APP_ROOT": _msys_path(PROJECT_ROOT),
                    "DATA_ROOT": _msys_path(data_root),
                    "ENTRYPOINT_PYTHON_BIN": _msys_path(Path(sys.executable)),
                    "ENTRYPOINT_UVICORN_BIN": _msys_path(Path(uvicorn)),
                    "ENTRYPOINT_PREFLIGHT_ONLY": "1",
                    "ENTRYPOINT_PREFLIGHT_EXPECT_API_TOKEN": "test-secret",
                    "READINESS_PROFILE": "all",
                    "REQUIRE_API_TOKEN": "1",
                    "API_TOKEN": "test-secret",
                    "GPU_CONCURRENCY": "1",
                }
            )
            result = subprocess.run(
                [str(bash), _msys_path(PROJECT_ROOT / "scripts/entrypoint.sh")],
                cwd=temporary,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("python_app_import=ok", result.stdout)
        self.assertIn("environment_propagation=ok", result.stdout)
        self.assertIn("readiness_profile=all", result.stdout)
        self.assertIn("api_token=set", result.stdout)
        self.assertNotIn("test-secret", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
