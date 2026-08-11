import asyncio
import hashlib
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx

from app.comfy import ComfyPromptError, queue_and_wait
from app.settings import ensure_data_directories, get_settings
from scripts import download_base_models
from scripts.download_utils import download_resumable, installed_record


class FakeResponse:
    def __init__(self, chunks, *, status=200, headers=None, on_exit=None):
        self.status = status
        self.headers = headers or {}
        self.chunks = list(chunks)
        self.on_exit = on_exit

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        if self.on_exit:
            self.on_exit()

    def read(self, _size):
        if not self.chunks:
            return b""
        item = self.chunks.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class DownloaderRuntimeTests(unittest.TestCase):
    def test_normal_download_promotes_completed_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.bin"
            response = FakeResponse(
                [b"abcdef"], headers={"Content-Length": "6"}
            )
            with mock.patch(
                "scripts.download_utils.urllib.request.urlopen",
                return_value=response,
            ):
                download_resumable("https://example.invalid/model", target, {})
            self.assertEqual(target.read_bytes(), b"abcdef")
            self.assertFalse(target.with_suffix(".bin.part").exists())

    def test_existing_partial_is_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.bin"
            partial = target.with_suffix(".bin.part")
            partial.write_bytes(b"abc")
            response = FakeResponse(
                [b"def"],
                status=206,
                headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6"},
            )
            with mock.patch(
                "scripts.download_utils.urllib.request.urlopen",
                return_value=response,
            ) as opened:
                download_resumable("https://example.invalid/model", target, {})
            self.assertEqual(target.read_bytes(), b"abcdef")
            self.assertEqual(opened.call_args.args[0].headers["Range"], "bytes=3-")

    def test_server_ignoring_range_restarts_partial_cleanly(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.bin"
            target.with_suffix(".bin.part").write_bytes(b"stale-prefix")
            response = FakeResponse(
                [b"fresh"], status=200, headers={"Content-Length": "5"}
            )
            with mock.patch(
                "scripts.download_utils.urllib.request.urlopen",
                return_value=response,
            ):
                download_resumable("https://example.invalid/model", target, {})
            self.assertEqual(target.read_bytes(), b"fresh")

    def test_missing_partial_reports_download_error_not_filenotfound(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.bin"
            partial = target.with_suffix(".bin.part")
            response = FakeResponse(
                [b"abcdef"],
                headers={"Content-Length": "6"},
                on_exit=lambda: partial.unlink(missing_ok=True),
            )
            with mock.patch(
                "scripts.download_utils.urllib.request.urlopen",
                return_value=response,
            ):
                with self.assertRaisesRegex(RuntimeError, "without producing"):
                    download_resumable("https://example.invalid/model", target, {})

    def test_http_error_is_explicit_and_keeps_existing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.bin"
            target.write_bytes(b"known-good")
            failure = urllib.error.HTTPError(
                "https://example.invalid/model",
                503,
                "Service Unavailable",
                None,
                None,
            )
            with mock.patch(
                "scripts.download_utils.urllib.request.urlopen",
                side_effect=failure,
            ):
                with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                    download_resumable(
                        "https://example.invalid/model", target, {}, force=True
                    )
            failure.close()
            self.assertEqual(target.read_bytes(), b"known-good")

    def test_interruption_retains_partial_and_existing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.bin"
            target.write_bytes(b"known-good")
            response = FakeResponse(
                [b"abc", OSError("connection lost")],
                headers={"Content-Length": "6"},
            )
            with mock.patch(
                "scripts.download_utils.urllib.request.urlopen",
                return_value=response,
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    download_resumable(
                        "https://example.invalid/model", target, {}, force=True
                    )
            self.assertEqual(target.read_bytes(), b"known-good")
            self.assertEqual(target.with_suffix(".bin.part").read_bytes(), b"abc")

    def test_valid_model_and_manifest_skip_network_and_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "models/model.bin"
            target.parent.mkdir(parents=True)
            content = b"already-valid"
            target.write_bytes(content)
            spec = {
                "relative_path": "model.bin",
                "url": "https://example.invalid/model",
                "source": "fixture",
                "revision": "fixed",
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "items": {"one": spec},
                        "profiles": {"t2v": ["one"], "i2v": [], "flf2v": [], "all": ["one"]},
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "downloads/installed-files.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sections": {
                            "base_models": {"one": installed_record(target, spec)}
                        },
                    }
                ),
                encoding="utf-8",
            )
            before = manifest.read_bytes()
            with (
                mock.patch.dict(os.environ, {"DATA_ROOT": temporary}),
                mock.patch.object(download_base_models, "CATALOG", catalog),
                mock.patch.object(download_base_models, "download_resumable") as download,
            ):
                self.assertEqual(download_base_models.main(["--profile", "t2v"]), 0)
                self.assertEqual(
                    download_base_models.main(["--profile", "t2v", "--check"]),
                    0,
                )
            download.assert_not_called()
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(manifest.read_bytes(), before)


class ComfyErrorTests(unittest.TestCase):
    def test_prompt_http_400_preserves_useful_details_and_status(self):
        real_client = httpx.AsyncClient

        async def scenario():
            payload = {
                "error": {"type": "prompt_outputs_failed_validation", "token": "secret"},
                "node_errors": {"80": {"errors": [{"message": "Required input is missing: format"}]}},
            }
            client = real_client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(400, json=payload, request=request)
                )
            )
            with (
                mock.patch("app.comfy.httpx.AsyncClient", return_value=client),
                mock.patch(
                    "app.comfy.get_settings",
                    return_value=SimpleNamespace(comfy_url="http://comfy.test"),
                ),
            ):
                with self.assertRaises(ComfyPromptError) as raised:
                    await queue_and_wait({"80": {"class_type": "SaveVideo"}})
            error = raised.exception
            self.assertEqual(error.status_code, 400)
            self.assertIn("prompt_outputs_failed_validation", str(error))
            self.assertIn("Required input is missing: format", str(error))
            self.assertNotIn("secret", str(error))
            self.assertEqual(error.details["error"]["token"], "<redacted>")

        asyncio.run(scenario())

    def test_prompt_non_json_error_uses_redacted_text_fallback(self):
        real_client = httpx.AsyncClient

        async def scenario():
            client = real_client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        502,
                        text="upstream failed with Authorization: Bearer private-value",
                        request=request,
                    )
                )
            )
            with (
                mock.patch("app.comfy.httpx.AsyncClient", return_value=client),
                mock.patch(
                    "app.comfy.get_settings",
                    return_value=SimpleNamespace(comfy_url="http://comfy.test"),
                ),
            ):
                with self.assertRaises(ComfyPromptError) as raised:
                    await queue_and_wait({})
            self.assertEqual(raised.exception.status_code, 502)
            self.assertIn("upstream failed", str(raised.exception))
            self.assertNotIn("private-value", str(raised.exception))

        asyncio.run(scenario())


class FilesystemInitializationTests(unittest.TestCase):
    def test_required_directories_are_created_idempotently_without_data_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(os.environ, {"DATA_ROOT": temporary}):
                settings = get_settings()
                ensure_data_directories(settings)
                sentinel = settings.models_dir / "existing-model.safetensors"
                sentinel.write_bytes(b"keep")
                ensure_data_directories(settings)
                self.assertTrue(all(path.is_dir() for path in settings.required_dirs))
                self.assertEqual(sentinel.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
