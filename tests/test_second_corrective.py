import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from collections import namedtuple
from dataclasses import replace
from pathlib import Path

import httpx

from app.job_store import JobStore
from app.limits import AdmissionError, admit_job, estimate_cost_units, validate_sequence_consistency
from app.schemas import SequenceRequest
from app.readiness import check_models, check_workflows
from app.settings import ensure_data_directories, get_settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AMD64_DIGEST = "sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d"
DiskUsage = namedtuple("DiskUsage", "total used free")


def request_with(shots, transition=None):
    return SequenceRequest.model_validate(
        {
            "id": "limits_test",
            "shots": shots,
            "transition": transition or {"type": "cut"},
        }
    )


class DigestTests(unittest.TestCase):
    def test_amd64_digest_matches_all_documentation(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(f"ubuntu22.04@{AMD64_DIGEST}", dockerfile)
        for name in ("VERSIONS.md", "AUDIT_REPORT.md"):
            text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
            self.assertIn(AMD64_DIGEST, text, name)
            self.assertNotIn("sha256:db2b1629", text, name)


class SequenceAndCostTests(unittest.TestCase):
    def test_dimension_mismatch(self):
        request = request_with(
            [
                {"id": "a", "mode": "t2v", "prompt": "a"},
                {"id": "b", "mode": "t2v", "prompt": "b", "width": 848},
            ]
        )
        with self.assertRaisesRegex(AdmissionError, "width and height"):
            validate_sequence_consistency(request)

    def test_fps_mismatch(self):
        request = request_with(
            [
                {"id": "a", "mode": "t2v", "prompt": "a"},
                {"id": "b", "mode": "t2v", "prompt": "b", "fps": 25},
            ]
        )
        with self.assertRaisesRegex(AdmissionError, "same fps"):
            validate_sequence_consistency(request)

    def test_crossfade_too_long(self):
        request = request_with(
            [{"id": "a", "mode": "t2v", "prompt": "a", "frames": 17, "fps": 60}],
            {"type": "crossfade", "duration_seconds": 0.3},
        )
        with self.assertRaisesRegex(AdmissionError, "must be shorter"):
            validate_sequence_consistency(request)

    def test_valid_sequence_and_reference_cost(self):
        request = request_with(
            [
                {"id": "a", "mode": "t2v", "prompt": "a"},
                {"id": "b", "mode": "t2v", "prompt": "b", "frames": 60},
            ],
            {"type": "crossfade", "duration_seconds": 0.2},
        )
        validate_sequence_consistency(request)
        reference = request_with([{"id": "a", "mode": "t2v", "prompt": "a"}])
        self.assertAlmostEqual(estimate_cost_units(reference), 1.0)

    def test_queue_and_disk_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.environ.get("DATA_ROOT")
            os.environ["DATA_ROOT"] = temporary
            try:
                settings = get_settings()
                ensure_data_directories(settings)
                store = JobStore(settings)
                store.initialize()
                request = request_with([{"id": "a", "mode": "t2v", "prompt": "a"}])
                with self.assertRaisesRegex(AdmissionError, "estimated cost"):
                    admit_job(
                        request,
                        replace(settings, max_job_cost_units=0.5),
                        store,
                        disk_usage=lambda _: DiskUsage(100, 1, 99 * 1024**3),
                        output_size=lambda _: 0,
                    )
                store.create("one", {})
                store.create("two", {})
                with self.assertRaisesRegex(AdmissionError, "queue is full"):
                    admit_job(
                        request,
                        settings,
                        store,
                        disk_usage=lambda _: DiskUsage(100, 1, 99 * 1024**3),
                        output_size=lambda _: 0,
                    )
                store.update("one", status="completed")
                store.update("two", status="completed")
                with self.assertRaisesRegex(AdmissionError, "free disk"):
                    admit_job(
                        request,
                        settings,
                        store,
                        disk_usage=lambda _: DiskUsage(100, 100, 0),
                        output_size=lambda _: 0,
                    )
            finally:
                if previous is None:
                    os.environ.pop("DATA_ROOT", None)
                else:
                    os.environ["DATA_ROOT"] = previous


class RemoteRedirectTests(unittest.TestCase):
    @staticmethod
    def resolver(host, _port):
        address = "127.0.0.1" if host == "127.0.0.1" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]

    def test_private_redirect_is_rejected(self):
        from app.orchestrator import download_remote_image

        async def scenario():
            transport = httpx.MockTransport(
                lambda request: httpx.Response(
                    302, headers={"Location": "https://127.0.0.1/private"}, request=request
                )
            )
            async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaisesRegex(ValueError, "non-public"):
                        await download_remote_image(
                            "https://public.example/start",
                            Path(temporary) / "image.png",
                            1024,
                            client=client,
                            resolver=self.resolver,
                        )

        asyncio.run(scenario())

    def test_https_downgrade_is_rejected(self):
        from app.orchestrator import download_remote_image

        async def scenario():
            transport = httpx.MockTransport(
                lambda request: httpx.Response(
                    302,
                    headers={"Location": "http://public.example/image"},
                    request=request,
                )
            )
            async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaisesRegex(ValueError, "downgrade"):
                        await download_remote_image(
                            "https://public.example/start",
                            Path(temporary) / "image.png",
                            1024,
                            client=client,
                            resolver=self.resolver,
                        )

        asyncio.run(scenario())

    def test_public_redirect_to_image_succeeds(self):
        from app.orchestrator import download_remote_image

        def handler(request):
            if request.url.host == "public.example":
                return httpx.Response(
                    302, headers={"Location": "https://cdn.example/image"}, request=request
                )
            return httpx.Response(
                200,
                headers={"Content-Type": "image/png", "Content-Length": "3"},
                content=b"png",
                request=request,
            )

        async def scenario():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler), follow_redirects=False
            ) as client:
                with tempfile.TemporaryDirectory() as temporary:
                    target = Path(temporary) / "image.png"
                    await download_remote_image(
                        "https://public.example/start",
                        target,
                        1024,
                        client=client,
                        resolver=self.resolver,
                    )
                    self.assertEqual(target.read_bytes(), b"png")

        asyncio.run(scenario())


class OfflineOperationsTests(unittest.TestCase):
    def test_readiness_profiles_are_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.environ.get("DATA_ROOT")
            os.environ["DATA_ROOT"] = temporary
            try:
                settings = get_settings()
                ensure_data_directories(settings)
                self.assertEqual(check_models(settings, "backend"), (True, {}))
                self.assertEqual(check_workflows(settings, "backend"), (True, {}))
                models_ok, details = check_models(settings, "t2v")
                self.assertFalse(models_ok)
                self.assertEqual(set(details), {"wan22_t2v_high", "wan22_t2v_low", "umt5_xxl", "wan21_vae"})
            finally:
                if previous is None:
                    os.environ.pop("DATA_ROOT", None)
                else:
                    os.environ["DATA_ROOT"] = previous

    def test_profiles_and_estimate_do_not_download(self):
        catalog = json.loads((PROJECT_ROOT / "config/base_models.json").read_text(encoding="utf-8"))
        self.assertEqual(
            catalog["profiles"]["t2v"],
            ["wan22_t2v_high", "wan22_t2v_low", "umt5_xxl", "wan21_vae"],
        )
        self.assertTrue(all("size_bytes" in item and "sha256" in item for item in catalog["items"].values()))
        with tempfile.TemporaryDirectory() as temporary:
            env = {**os.environ, "DATA_ROOT": temporary}
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/download_base_models.py"),
                    "--profile",
                    "t2v",
                    "--estimate",
                ],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("remaining=", result.stdout)
            self.assertFalse((Path(temporary) / "models").exists())
            refused = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts/download_base_models.py")],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(refused.returncode, 2)

    def test_cleanup_defaults_to_dry_run(self):
        from scripts.cleanup_jobs import main as cleanup_main

        with tempfile.TemporaryDirectory() as temporary:
            previous = os.environ.get("DATA_ROOT")
            os.environ["DATA_ROOT"] = temporary
            try:
                settings = get_settings()
                ensure_data_directories(settings)
                store = JobStore(settings)
                store.initialize()
                store.create("old_job", {})
                store.update("old_job", status="completed")
                output = settings.outputs_dir / "old_job"
                output.mkdir()
                (output / "sequence.mp4").write_bytes(b"test")
                self.assertEqual(cleanup_main(["--older-than-hours", "0"]), 0)
                self.assertTrue(output.exists())
                self.assertIsNotNone(store.get("old_job"))
            finally:
                if previous is None:
                    os.environ.pop("DATA_ROOT", None)
                else:
                    os.environ["DATA_ROOT"] = previous


if __name__ == "__main__":
    unittest.main()
