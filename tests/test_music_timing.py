"""Music timing: automatic tempo / downbeat / loop detection, the maths of recorded timings, and the local recording server."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import music_timing as mt  # noqa: E402
import music_timing_server as srv  # noqa: E402

SR = 22050


def drum_track(path: Path, bpm=120.0, bars=24, loop_bars=4, intro_bars=2, seed=3) -> dict:
    """A synthetic track: an unrelated intro, then a 4-bar loop (a kick on every beat, a snare on 2 and 4, a different chord per bar) repeated."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    total = int((intro_bars + bars) * 4 * beat * SR)
    y = np.zeros(total, np.float32)

    def tone(freq, length, amp, decay):
        t = np.arange(int(length * SR)) / SR
        return (amp * np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)).astype(np.float32)

    def add(sound, at):
        i = int(at * SR)
        y[i:i + len(sound)] += sound[: max(0, len(y) - i)]

    noise = rng.standard_normal(SR).astype(np.float32)
    chords = [(220, 277, 330), (196, 247, 294), (174, 220, 262), (165, 208, 247)]
    intro_notes = [(311, 392), (415, 523)]
    for bar in range(intro_bars + bars):
        t0 = bar * 4 * beat
        in_intro = bar < intro_bars
        for b in range(4):
            at = t0 + b * beat
            if in_intro:
                add(tone(80, 0.12, 0.35, 28), at) if b == 0 else None
                add(tone(intro_notes[bar % 2][b % 2], beat * 0.9, 0.25, 4), at)
            else:
                add(tone(60, 0.15, 0.8 if b in (0, 2) else 0.6, 22), at)  # a kick on every beat, with a click so that it shows in the spectrum
                add(noise[: int(0.012 * SR)] * 0.8, at)
                if b == 0:  # a crash on the first beat of the bar: what a downbeat usually carries
                    add(noise[: int(0.3 * SR)] * np.exp(-np.arange(int(0.3 * SR)) / SR * 9).astype(np.float32) * 1.0, at)
                if b in (1, 3):
                    add(noise[: int(0.1 * SR)] * np.exp(-np.arange(int(0.1 * SR)) / SR * 40).astype(np.float32) * 0.5, at)
        if not in_intro:
            for f in chords[(bar - intro_bars) % loop_bars]:
                add(tone(f, 4 * beat * 0.95, 0.12, 0.6), t0)
    y /= max(1.0, float(np.abs(y).max()))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((y * 32000).astype(np.int16).tobytes())
    return {"beat": beat, "loop_start": intro_bars * 4 * beat, "loop_length": loop_bars * 4 * beat, "duration": total / SR}


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.truth = drum_track(cls.dir / "loop.wav")
        cls.result = mt.analyze_music(cls.dir / "loop.wav")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_tempo_is_found_and_its_octaves_are_offered(self):
        self.assertAlmostEqual(self.result["bpm"], 120.0, delta=1.0)
        self.assertEqual(self.result["bpm_alternatives"], [round(self.result["bpm"] / 2, 2), round(self.result["bpm"] * 2, 2)])
        self.assertGreater(self.result["tempo_confidence"], 0.5)

    def test_the_beats_are_on_the_drum_hits(self):
        beats = np.array(self.result["beats"])
        self.assertAlmostEqual(float(np.diff(beats).mean()), self.truth["beat"], delta=0.004)
        phase = float(np.min(np.abs(((beats - 0.0 + self.truth["beat"] / 2) % self.truth["beat"]) - self.truth["beat"] / 2)))
        self.assertLess(phase, 0.04)  # a beat falls within 40 ms of the true grid

    def test_the_downbeats_are_the_bar_starts(self):
        downbeats = np.array(self.result["downbeats"])
        bar = 4 * self.truth["beat"]
        offset = np.abs(((downbeats + bar / 2) % bar) - bar / 2)
        self.assertLess(float(np.median(offset)), 0.05)

    def test_the_loop_start_and_restart_are_found(self):
        loop = self.result["loop"]
        self.assertIsNotNone(loop)
        self.assertTrue(self.result["repeats"])
        self.assertAlmostEqual(loop["length"], self.truth["loop_length"], delta=0.1)  # 4 bars = 8 s at 120 bpm
        self.assertAlmostEqual(loop["start"], self.truth["loop_start"], delta=0.1)  # after the 2-bar intro
        self.assertAlmostEqual(loop["end"] - loop["start"], loop["length"], places=3)
        self.assertEqual(loop["length_beats"], 16)
        self.assertTrue(all(loop["start"] - 1e-6 <= t < loop["end"] for t in self.result["suggested_cuts"]))

    def test_a_track_that_never_repeats_has_no_loop(self):
        rng = np.random.default_rng(9)
        beat = 0.5
        y = np.zeros(int(40 * SR), np.float32)
        for k in range(80):  # a steady kick, but a different melody every beat: nothing repeats
            i = int(k * beat * SR)
            t = np.arange(int(0.4 * SR)) / SR
            f = float(rng.uniform(150, 1800))
            y[i:i + len(t)] += (0.3 * np.sin(2 * np.pi * f * t) * np.exp(-t * 6) + 0.6 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 20) * (k % 2 == 0)).astype(np.float32)
        path = self.dir / "noloop.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SR)
            handle.writeframes((y / max(1.0, float(np.abs(y).max())) * 30000).astype(np.int16).tobytes())
        result = mt.analyze_music(path)
        self.assertIsNone(result["loop"])
        self.assertFalse(result["repeats"])
        self.assertAlmostEqual(result["bpm"], 120.0, delta=1.5)

    def test_a_clip_that_is_too_short_is_refused(self):
        path = self.dir / "short.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SR)
            handle.writeframes(np.zeros(1000, np.int16).tobytes())
        with self.assertRaises(ValueError):
            mt.analyze_music(path)


class ExpandCutTimesTests(unittest.TestCase):
    LOOP = {"mode": "loop", "loop": {"start": 2.0, "end": 6.0}, "marks": [2.0, 3.0, 4.5, 5.0]}

    def test_a_loop_repeats_its_marks_every_loop_length(self):
        # the video starts at the loop start: 0 is not a cut, the pattern (0, 1, 2.5, 3) repeats every 4 s
        self.assertEqual(mt.expand_cut_times(self.LOOP, 10.0), [1.0, 2.5, 3.0, 4.0, 5.0, 6.5, 7.0, 8.0, 9.0])

    def test_sped_up_music_shortens_the_times(self):
        faster = mt.expand_cut_times(self.LOOP, 4.0, speed=2.0)
        self.assertEqual(faster, [0.5, 1.25, 1.5, 2.0, 2.5, 3.25, 3.5])

    def test_a_music_offset_starts_the_video_later_in_the_loop(self):
        times = mt.expand_cut_times(self.LOOP, 5.0, music_offset=3.0)
        self.assertEqual(times, [1.5, 2.0, 3.0, 4.0])  # music 3 -> 8 s: marks at 4.5, 5, 6, 7 (8.5 is past the end)

    def test_the_moment_the_loop_starts_again_is_a_cut_unless_told_otherwise(self):
        timing = {"mode": "loop", "loop": {"start": 2.0, "end": 6.0}, "marks": [3.0, 4.5]}  # no mark on the loop start itself
        self.assertEqual(mt.expand_cut_times(timing, 10.0), [1.0, 2.5, 4.0, 5.0, 6.5, 8.0, 9.0])
        self.assertEqual(mt.expand_cut_times(timing, 10.0, loop_restart_is_a_cut=False), [1.0, 2.5, 5.0, 6.5, 9.0])

    def test_a_full_track_uses_its_marks_once(self):
        full = {"mode": "full", "loop": None, "marks": [0.0, 1.5, 4.0, 12.0]}
        self.assertEqual(mt.expand_cut_times(full, 10.0), [1.5, 4.0])  # 0 is the start, 12 is past the end
        self.assertEqual(mt.expand_cut_times(full, 10.0, music_offset=1.5), [2.5])

    def test_bad_input_is_refused(self):
        with self.assertRaises(ValueError):
            mt.expand_cut_times(self.LOOP, 5.0, speed=0)
        with self.assertRaises(ValueError):
            mt.expand_cut_times({"mode": "loop", "loop": {"start": 3.0, "end": 3.0}, "marks": []}, 5.0)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.truth = drum_track(cls.dir / "demo.wav", bars=12)
        cls.server = srv.serve(cls.dir, 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def get(self, path, headers=None):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", headers=headers or {})
        return urllib.request.urlopen(request, timeout=60)

    def post(self, path, data, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=body, headers={"Content-Type": content_type}, method="POST")
        try:
            return json.load(urllib.request.urlopen(request, timeout=60)), 200
        except urllib.error.HTTPError as exc:
            return json.load(exc), exc.code

    def test_the_page_and_the_track_list(self):
        html = self.get("/").read().decode()
        self.assertIn("Prise de timing musique", html)
        self.assertIn("keydown", html)  # the keyboard capture
        self.assertEqual(json.load(self.get("/api/tracks"))["tracks"], ["demo.wav"])

    def test_the_analysis_route_finds_the_tempo_on_its_own(self):
        result = json.load(self.get("/api/analyze?track=demo.wav"))
        self.assertAlmostEqual(result["bpm"], 120.0, delta=1.0)
        self.assertIn("suggested_cuts", result)

    def test_peaks_for_the_waveform(self):
        peaks = json.load(self.get("/api/peaks?track=demo.wav"))
        self.assertEqual(len(peaks["peaks"]), 2400)
        self.assertAlmostEqual(peaks["duration"], self.truth["duration"], delta=0.1)
        self.assertEqual(max(peaks["peaks"]), 1.0)

    def test_audio_is_served_with_range_requests_for_seeking(self):
        size = (self.dir / "demo.wav").stat().st_size
        whole = self.get("/audio/demo.wav")
        self.assertEqual((whole.status, int(whole.headers["Content-Length"])), (200, size))
        part = self.get("/audio/demo.wav", {"Range": "bytes=100-199"})
        self.assertEqual((part.status, part.headers["Content-Range"], len(part.read())), (206, f"bytes 100-199/{size}", 100))

    def test_file_names_cannot_escape_the_folder(self):
        for bad in ("/audio/..%2F..%2Fetc%2Fpasswd", "/api/analyze?track=../scripts/music_timing.py", "/api/peaks?track=/etc/passwd", "/audio/nothing.wav"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.get(bad)
            self.assertEqual(caught.exception.code, 404, bad)

    def test_a_loop_timing_is_saved_and_loaded_back(self):
        marks = [self.truth["loop_start"] + x for x in (0.0, 2.0, 4.0, 6.0)]
        saved, status = self.post("/api/save", {"track": "demo.wav", "mode": "loop", "marks": marks + [30.0 * 0], "latency_ms": 20,
                                                "loop": {"start": self.truth["loop_start"], "end": self.truth["loop_start"] + 8.0}, "playback_rate": 1.25})
        self.assertEqual(status, 200, saved)
        self.assertEqual(saved["marks"], 4)  # the stray mark at 0 s is outside the loop and dropped
        self.assertEqual(saved["example_cuts_20s"][:3], [2.0, 4.0, 6.0])
        timing = json.load(self.get("/api/timing?track=demo.wav"))
        self.assertEqual((timing["mode"], timing["track"]["name"], timing["latency_ms"], timing["playback_rate_used"]), ("loop", "demo.wav", 20.0, 1.25))
        self.assertEqual(timing["marks_in_loop"], [0.0, 2.0, 4.0, 6.0])
        self.assertEqual(len(timing["track"]["sha256"]), 64)
        self.assertEqual(mt.expand_cut_times(timing, 20.0), saved["example_cuts_20s"])  # the saved file is what the video builder reads

    def test_a_full_track_timing_and_the_refusals(self):
        saved, status = self.post("/api/save", {"track": "demo.wav", "mode": "full", "marks": [1.0, 5.5, 9.0]})
        self.assertEqual((status, saved["marks"]), (200, 3))
        for bad in ({"track": "demo.wav", "mode": "loop", "marks": [1.0], "loop": {"start": 0.0, "end": None}},
                    {"track": "demo.wav", "mode": "loop", "marks": [1.0], "loop": {"start": 5.0, "end": 4.0}},
                    {"track": "demo.wav", "mode": "weird", "marks": []}, {"track": "demo.wav", "mode": "full", "marks": ["x"]},
                    {"track": "demo.wav", "mode": "full", "marks": [999.0]}, {"track": "../x.wav", "mode": "full", "marks": []}):
            body, status = self.post("/api/save", bad)
            self.assertEqual(status, 400, bad)
            self.assertIn("error", body)

    def test_upload_adds_a_track_and_refuses_other_files(self):
        body, status = self.post("/api/upload?name=my%20song.wav", (self.dir / "demo.wav").read_bytes(), "application/octet-stream")
        self.assertEqual((status, body["name"]), (200, "my song.wav"))
        self.assertIn("my song.wav", json.load(self.get("/api/tracks"))["tracks"])
        body, status = self.post("/api/upload?name=evil.exe", b"MZ", "application/octet-stream")
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
