import asyncio
import hashlib
import importlib
import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path

from app.job_store import JobStore
from app.settings import get_settings
from scripts.download_utils import update_manifest, validate_file


class PersistenceTests(unittest.TestCase):
    def test_running_job_becomes_interrupted_after_initialize(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.environ.get("DATA_ROOT")
            os.environ["DATA_ROOT"] = temporary
            try:
                store = JobStore(get_settings())
                store.initialize()
                store.create("job1", {"id": "job1", "shots": []})
                store.update("job1", status="running")
                JobStore(get_settings()).initialize()
                job = store.get("job1")
                self.assertEqual(job["status"], "interrupted")
                self.assertIn("restarted", job["error"])
            finally:
                if previous is None:
                    os.environ.pop("DATA_ROOT", None)
                else:
                    os.environ["DATA_ROOT"] = previous


class DownloadValidationTests(unittest.TestCase):
    def test_size_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.bin"
            path.write_bytes(b"known-content")
            spec = {
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(b"known-content").hexdigest(),
            }
            self.assertEqual(validate_file(path, spec), (True, "valid"))

    def test_unchanged_manifest_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            records = {"one": {"size_bytes": 10}}
            update_manifest(path, "models", records)
            first = path.stat().st_mtime_ns
            update_manifest(path, "models", records)
            self.assertEqual(first, path.stat().st_mtime_ns)


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is not installed locally")
class TestModeApiTests(unittest.TestCase):
    def test_auth_health_job_and_persistence(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as temporary:
            updates = {
                "DATA_ROOT": temporary,
                "APP_TEST_MODE": "1",
                "API_TOKEN": "test-token",
                "REQUIRE_API_TOKEN": "1",
                "GPU_CONCURRENCY": "1",
            }
            previous = {key: os.environ.get(key) for key in updates}
            os.environ.update(updates)
            try:
                import app.orchestrator as orchestrator_module
                import app.main as main_module

                importlib.reload(orchestrator_module)
                main_module = importlib.reload(main_module)
                app, store = main_module.app, main_module.store

                with TestClient(app) as client:
                    self.assertEqual(client.get("/health/live").status_code, 200)
                    self.assertEqual(client.get("/health/ready").status_code, 200)
                    self.assertEqual(client.get("/v1/catalog").status_code, 401)
                    headers = {"Authorization": "Bearer test-token"}
                    image_created = client.post(
                        "/v1/images",
                        json={
                            "engine": "flux_schnell",
                            "prompt": "cinematic coast at sunrise",
                            "width": 832,
                            "height": 480,
                            "seed": 42,
                            "steps": 4,
                        },
                        headers=headers,
                    )
                    self.assertEqual(image_created.status_code, 202)
                    image_id = image_created.json()["image_id"]
                    self.assertRegex(image_id, r"^img_[a-f0-9]{32}$")
                    for _ in range(50):
                        image_status = client.get(
                            f"/v1/images/{image_id}", headers=headers
                        ).json()
                        if image_status["status"] == "completed":
                            break
                        time.sleep(0.02)
                    self.assertEqual(image_status["status"], "completed")
                    image_output = client.get(
                        f"/v1/images/{image_id}/output", headers=headers
                    )
                    self.assertEqual(image_output.status_code, 200)
                    self.assertEqual(image_output.headers["content-type"], "image/png")
                    self.assertEqual(
                        client.get(f"/v1/jobs/{image_id}", headers=headers).status_code,
                        404,
                    )
                    payload = {
                        "id": "test_job",
                        "shots": [{
                            "id": "shot",
                            "mode": "i2v",
                            "start_image": {"image_id": image_id},
                        }],
                    }
                    created = client.post("/v1/jobs", json=payload, headers=headers)
                    self.assertEqual(created.status_code, 202)
                    self.assertAlmostEqual(created.json()["estimated_cost_units"], 1.0)
                    for _ in range(50):
                        result = client.get("/v1/jobs/test_job", headers=headers).json()
                        if result["status"] == "completed":
                            break
                        time.sleep(0.02)
                    self.assertEqual(result["status"], "completed")
                    self.assertAlmostEqual(result["estimated_cost_units"], 1.0)
                    self.assertEqual(
                        client.get("/v1/jobs/test_job/output", headers=headers).status_code,
                        200,
                    )
                reopened = JobStore(get_settings())
                reopened.initialize()
                self.assertEqual(reopened.get("test_job")["status"], "completed")
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
