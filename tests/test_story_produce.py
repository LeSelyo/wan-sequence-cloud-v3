import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_produce as sp  # noqa: E402


def write_wav(path: Path, seconds: float, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x10\x00" * int(seconds * rate))


def test_padded_wav_has_exactly_the_requested_length_and_keeps_the_voice_at_the_start(tmp_path):
    source = tmp_path / "line.wav"
    write_wav(source, 1.0)
    out = sp.padded_wav(source, 2.5, tmp_path / "padded.wav")
    with wave.open(str(out), "rb") as handle:
        assert handle.getframerate() == 16000 and handle.getnframes() == 40000
        data = handle.readframes(handle.getnframes())
    assert data[:2] != b"\x00\x00" and data[-2:] == b"\x00\x00"  # the voice first, silence after
    silent = sp.padded_wav(None, 1.0, tmp_path / "silence.wav")
    with wave.open(str(silent), "rb") as handle:
        assert handle.getnframes() == 16000 and set(handle.readframes(16000)) == {0}


def test_every_shot_gets_the_right_clip_job(tmp_path):
    plan = {"characters": [{"id": "c1", "name": "Brandt", "age": 32, "gender": "m", "role": "ship captain"}, {"id": "c2", "name": "Ilse", "age": 59, "gender": "f", "role": "doctor"}],
            "shots": [{"id": "s001", "kind": "narration", "speaker": "narrator", "in_shot": [], "visual": "the camera pushes in"},
                      {"id": "s002", "kind": "talk", "speaker": "c1", "in_shot": ["c1"], "visual": "speaks urgently"},
                      {"id": "s003", "kind": "choice", "speaker": "narrator", "in_shot": ["c1", "c2"], "visual": "waits tensely"}]}
    cards = {"characters": {"c1": {"portrait": {"file": "p1.png"}}, "c2": {"portrait": {"file": "p2.png"}}}}
    for shot in plan["shots"]:
        write_wav(tmp_path / "voices" / f"{shot['id']}.wav", 2.0)
    jobs = {j["id"]: j for j in sp.animate_jobs(plan, cards, tmp_path)}
    assert set(jobs) == {"s001", "s002", "s003_c1", "s003_c2"}
    assert jobs["s001"]["voice"] is None and jobs["s001"]["source"] == tmp_path / "stills" / "s001.png" and "the camera pushes in" in jobs["s001"]["prompt"]  # a scene: its own still, silence
    assert jobs["s002"]["voice"] == tmp_path / "voices" / "s002.wav" and jobs["s002"]["source"].name == "p1.png" and "Brandt" in jobs["s002"]["prompt"]  # a talk: the portrait + the line
    assert jobs["s002"]["seconds"] == 2.15 and jobs["s003_c2"]["voice"] is None and jobs["s003_c2"]["source"].name == "p2.png" and "mouth closed" in jobs["s003_c2"]["prompt"]
    assert jobs["s003_c1"]["seconds"] == 3.2  # the choice lasts at least the countdown
