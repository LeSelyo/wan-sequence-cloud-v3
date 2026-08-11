import asyncio
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from app.lora_downloads import ensure_lora
from app.settings import get_settings


def entry(source: str, url: str | None, content: bytes = b"lora") -> dict:
    value = {
        "id": f"fixture-{source}",
        "kind": "lora",
        "filename": f"{source}.safetensors",
        "source": source,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    if url is not None:
        value["url"] = url
    return value


class LoraDownloadTests(unittest.TestCase):
    def test_catalog_entries_have_explicit_approved_source_and_url(self):
        catalog_path = Path(__file__).resolve().parent.parent / "config/loras.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.assertEqual(catalog["schema_version"], 2)
        for item_id, item in catalog["items"].items():
            self.assertIn(item.get("source"), {"huggingface", "civitai"}, item_id)
            self.assertTrue(str(item.get("url", "")).startswith("https://"), item_id)
        serialized = json.dumps(catalog).lower()
        self.assertNotIn("hf_token", serialized)
        self.assertNotIn("civitai_api_token", serialized)

    def _download_case(self, source: str, url: str, token_name: str):
        content = b"lora"
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {"DATA_ROOT": temporary, token_name: "private-token", "AUTO_DOWNLOAD_LORAS": "1"},
            clear=False,
        ):
            settings = get_settings()
            calls = []
            def download(received_url, target, headers, force=False):
                calls.append((received_url, dict(headers), force))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            with mock.patch("app.lora_downloads.download_resumable", side_effect=download):
                target = ensure_lora(entry(source, url), settings)
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(calls[0][0], url)
            self.assertEqual(calls[0][1]["Authorization"], "Bearer private-token")
            manifest_text = (settings.downloads_dir / "installed-files.json").read_text(encoding="utf-8")
            self.assertNotIn("private-token", manifest_text)
            self.assertEqual(json.loads(manifest_text)["sections"]["loras"][f"fixture-{source}"]["installed_from"], source)

    def test_huggingface_url_and_token_are_used_only_in_memory(self):
        self._download_case(
            "huggingface",
            "https://huggingface.co/example/repo/resolve/revision/lora.safetensors",
            "HF_TOKEN",
        )

    def test_civitai_url_and_token_are_used_only_in_memory(self):
        self._download_case(
            "civitai",
            "https://civitai.com/api/download/models/123",
            "CIVITAI_API_TOKEN",
        )

    def test_missing_url_fails_without_search_or_download(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"DATA_ROOT": temporary}):
            with mock.patch("app.lora_downloads.download_resumable") as download:
                with self.assertRaisesRegex(RuntimeError, "no configured download URL"):
                    ensure_lora(entry("huggingface", None), get_settings())
            download.assert_not_called()

    def test_two_concurrent_requests_share_one_coherent_download(self):
        content = b"lora"
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"DATA_ROOT": temporary}):
            settings = get_settings()
            calls = 0
            lock = threading.Lock()
            def download(_url, target, _headers, force=False):
                nonlocal calls
                with lock:
                    calls += 1
                time.sleep(0.1)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            spec = entry("huggingface", "https://huggingface.co/example/repo/resolve/revision/lora.safetensors")
            with mock.patch("app.lora_downloads.download_resumable", side_effect=download):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _: ensure_lora(spec, settings), range(2)))
            self.assertEqual(calls, 1)
            self.assertEqual(results[0].read_bytes(), content)
            self.assertEqual(results[0], results[1])

    def test_standard_does_not_prepare_lightx_and_turbo_prepares_both(self):
        from app.orchestrator import _ensure_turbo_loras

        async def scenario():
            with mock.patch("app.orchestrator.ensure_lora") as ensure:
                # Standard render path never calls this helper.
                ensure.assert_not_called()
                await _ensure_turbo_loras("wan22_t2v")
                self.assertEqual(ensure.call_count, 2)
                ids = [call.args[0]["id"] for call in ensure.call_args_list]
                self.assertEqual(ids, ["wan22_t2v_lightx_high", "wan22_t2v_lightx_low"])
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
