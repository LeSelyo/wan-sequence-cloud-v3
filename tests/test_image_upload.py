from __future__ import annotations

import importlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image


def encoded_image(format_name: str, size: tuple[int, int] = (8, 8)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (20, 40, 60)).save(output, format=format_name)
    return output.getvalue()


class ImageUploadApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.updates = {
            "DATA_ROOT": self.temporary.name,
            "APP_TEST_MODE": "1",
            "API_TOKEN": "test-token",
            "REQUIRE_API_TOKEN": "1",
            "GPU_CONCURRENCY": "1",
            "MAX_INPUT_DOWNLOAD_MB": "1",
            "MAX_INPUT_IMAGE_PIXELS": "100",
        }
        self.previous = {key: os.environ.get(key) for key in self.updates}
        os.environ.update(self.updates)
        import app.orchestrator as orchestrator_module
        import app.main as main_module

        importlib.reload(orchestrator_module)
        self.main = importlib.reload(main_module)
        self.client_context = TestClient(self.main.app)
        self.client = self.client_context.__enter__()
        self.headers = {"Authorization": "Bearer test-token"}

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temporary.cleanup()

    def upload(self, data: bytes, filename: str = "image.png", content_type: str = "image/png"):
        return self.client.post(
            "/v1/images/upload",
            files={"file": (filename, data, content_type)},
            headers=self.headers,
        )

    def test_valid_png_ignores_malicious_filename_and_integrates_with_i2v(self):
        response = self.upload(encoded_image("PNG"), "../../etc/passwd", "text/plain")
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        image_id = body["image_id"]
        self.assertRegex(image_id, r"^img_[a-f0-9]{32}$")
        self.assertEqual(body["status"], "completed")
        expected = Path(self.temporary.name) / "outputs/images" / image_id / "image.png"
        self.assertTrue(expected.is_file())
        self.assertNotIn("passwd", str(expected))
        status = self.client.get(f"/v1/images/{image_id}", headers=self.headers)
        self.assertEqual(status.status_code, 200)
        self.assertNotIn("output", status.json())
        self.assertEqual(status.json()["output_url"], f"/v1/images/{image_id}/output")
        output = self.client.get(f"/v1/images/{image_id}/output", headers=self.headers)
        self.assertEqual(output.status_code, 200)
        self.assertEqual(output.headers["content-type"], "image/png")
        with Image.open(io.BytesIO(output.content)) as decoded:
            self.assertEqual((decoded.format, decoded.size), ("PNG", (8, 8)))

        job = self.client.post(
            "/v1/jobs",
            headers=self.headers,
            json={
                "id": "upload_i2v",
                "shots": [{"id": "shot", "mode": "i2v", "start_image": {"image_id": image_id}}],
            },
        )
        self.assertEqual(job.status_code, 202, job.text)

    def test_cleanup_jobs_removes_uploaded_image_and_database_row(self):
        from scripts.cleanup_jobs import main as cleanup

        response = self.upload(encoded_image("PNG"))
        image_id = response.json()["image_id"]
        target = Path(self.temporary.name) / "outputs/images" / image_id
        self.assertTrue(target.is_dir())
        self.assertEqual(
            cleanup(["--older-than-hours", "0", "--status", "completed", "--force"]),
            0,
        )
        self.assertFalse(target.exists())
        self.assertEqual(
            self.client.get(f"/v1/images/{image_id}", headers=self.headers).status_code,
            404,
        )

    def test_jpeg_is_reencoded_as_canonical_png_and_works_for_flf2v(self):
        response = self.upload(encoded_image("JPEG"), "photo.jpg", "image/jpeg")
        self.assertEqual(response.status_code, 201, response.text)
        image_id = response.json()["image_id"]
        status = self.client.get(f"/v1/images/{image_id}", headers=self.headers).json()
        self.assertEqual(status["metadata"]["original_format"], "jpeg")
        job = self.client.post(
            "/v1/jobs",
            headers=self.headers,
            json={
                "id": "upload_flf2v",
                "shots": [{
                    "id": "shot",
                    "mode": "i(keyframe)2v",
                    "start_image": {"image_id": image_id},
                    "end_image": {"image_id": image_id},
                }],
            },
        )
        self.assertEqual(job.status_code, 202, job.text)

    def test_auth_empty_invalid_oversized_and_pixel_limits(self):
        valid = encoded_image("PNG")
        self.assertEqual(
            self.client.post("/v1/images/upload", files={"file": ("x.png", valid, "image/png")}).status_code,
            401,
        )
        self.assertEqual(
            self.client.post(
                "/v1/images/upload",
                files={"file": ("x.png", valid, "image/png")},
                headers={"Authorization": "Bearer wrong"},
            ).status_code,
            401,
        )
        self.assertEqual(self.upload(b"").status_code, 422)
        self.assertEqual(self.upload(b"not an image", "fake.png").status_code, 422)
        self.assertEqual(self.upload(b"x" * (1024 * 1024 + 1)).status_code, 413)
        self.assertEqual(self.upload(encoded_image("PNG", (11, 10))).status_code, 413)
        self.assertEqual(self.upload(encoded_image("WEBP"), "x.webp", "image/webp").status_code, 415)
        self.assertEqual(
            list((Path(self.temporary.name) / "outputs/images").glob(".upload-*")),
            [],
        )


if __name__ == "__main__":
    unittest.main()
