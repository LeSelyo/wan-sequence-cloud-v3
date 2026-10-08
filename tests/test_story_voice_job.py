import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_voice_job as vj  # noqa: E402

VOICES = [{"id": "dan", "file": "dan.mp3", "ref_text": "hello there"}, {"id": "siren", "file": "siren.mp3", "ref_text": None}]
PLAN = {"params": {"language": "en"}, "characters": [{"id": "c1", "gender": "m", "age": 32, "role": "ship captain"}, {"id": "c2", "gender": "f", "age": 59, "role": "doctor"}],
        "voices": {"narrator": "dan", "c1": "design:a man in his 30s, ship captain", "c2": "siren"},
        "shots": [{"id": "s001", "speaker": "narrator", "text": "x"}, {"id": "s002", "speaker": "c1", "text": "y"}, {"id": "s003", "speaker": "c2", "text": "z"}]}


def test_recorded_voice_with_transcript_is_cloned_designed_voice_is_designed_and_a_voice_without_transcript_falls_back_to_a_designed_woman():
    job = vj.build_job(PLAN, VOICES, "/root/voices", "/root/out")
    voices = {line["id"]: line["voice"] for line in job["lines"]}
    assert voices["s001"] == {"kind": "clone", "ref_audio": "/root/voices/dan.mp3", "ref_text": "hello there", "voice_id": "dan"}
    assert voices["s002"] == {"kind": "design", "instruct": "a man in his 30s, ship captain"}
    assert voices["s003"]["kind"] == "design" and "woman in her 50s" in voices["s003"]["instruct"] and "no transcript" in voices["s003"]["note"]
    assert job["language"] == "English" and [l["seed"] for l in job["lines"]] == [1000, 1001, 1002]
