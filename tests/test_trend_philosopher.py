"""scripts/trend_philosopher.py helpers that need neither GPU nor network: the
Krea2 image helper (API mocked), background generation, and the Ollama script writer."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import trend_philosopher as tp  # noqa: E402


class FakeResponse:
    def __init__(self, payload=None, content=b""):
        self._payload, self.content = payload, content

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class Krea2HelperTests(unittest.TestCase):
    def test_posts_polls_and_saves_the_png(self):
        calls = []

        def fake_api(method, path, **kwargs):
            calls.append((method, path, kwargs.get("json")))
            if method == "POST":
                return FakeResponse({"image_id": "img_1"})
            if path.endswith("/output"):
                return FakeResponse(content=b"PNGDATA")
            return FakeResponse({"status": "completed"})

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(tp, "_api", fake_api):
            out = tp.generate_krea2_image(Path(temporary) / "a.png", "a cat", width=480, height=832, seed=3)
            self.assertEqual(out.read_bytes(), b"PNGDATA")
        self.assertEqual(calls[0][2], {"engine": "krea2", "prompt": "a cat", "width": 480, "height": 832, "steps": 8, "seed": 3})

    def test_failed_generation_raises(self):
        def fake_api(method, path, **kwargs):
            if method == "POST":
                return FakeResponse({"image_id": "img_1"})
            return FakeResponse({"status": "failed", "error": "boom"})

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(tp, "_api", fake_api):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                tp.generate_krea2_image(Path(temporary) / "a.png", "x", width=480, height=832)

    def test_character_still_comes_from_the_same_helper(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(tp, "generate_krea2_image") as helper:
            tp.resolve_character_image(Path(temporary) / "c.png", character_prompt="a monk", seed=7)
            helper.assert_called_once()
            self.assertEqual(helper.call_args.kwargs["seed"], 7)
            with self.assertRaises(ValueError):
                tp.resolve_character_image(Path(temporary) / "c.png")

    def test_backgrounds_are_one_image_per_prompt_with_distinct_seeds_and_the_video_aspect(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(tp, "generate_krea2_image") as helper:
            tp.generate_background_images(["a", "b", "c"], Path(temporary) / "bg", seed=10)
        self.assertEqual(
            [call.args[0].name for call in helper.call_args_list], ["bg_00.png", "bg_01.png", "bg_02.png"]
        )
        self.assertEqual([call.kwargs["seed"] for call in helper.call_args_list], [10, 11, 12])
        self.assertTrue(all((call.kwargs["width"], call.kwargs["height"]) == (480, 832) for call in helper.call_args_list))


class OllamaScriptTests(unittest.TestCase):
    def test_cleans_reasoning_and_markdown_and_asks_for_thinking(self):
        reply = {"message": {"content": "<think>hmm</think>**Le temps** passe.\n\n# Titre\nTout change."}}
        with mock.patch.object(tp.httpx, "post", return_value=FakeResponse(reply)) as post:
            text = tp.write_script_ollama("le temps", model="qwen3.6:27b")
        self.assertNotIn("hmm", text)
        self.assertNotIn("*", text)
        self.assertNotIn("#", text)
        self.assertIn("Le temps passe.", text)
        body = post.call_args.kwargs["json"]
        self.assertEqual((body["model"], body["think"], body["stream"]), ("qwen3.6:27b", True, False))
        self.assertIn("le temps", body["messages"][0]["content"])
        self.assertTrue(post.call_args.args[0].endswith("/api/chat"))


if __name__ == "__main__":
    unittest.main()
