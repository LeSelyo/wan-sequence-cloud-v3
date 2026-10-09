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


PEOPLE = [{"id": "c1", "name": "Brandt", "age": 32, "gender": "m", "role": "ship captain", "wardrobe": "a heavy thermal suit"},
          {"id": "c2", "name": "Ilse", "age": 59, "gender": "f", "role": "doctor", "wardrobe": "a white coat"}]
CARDS = {"characters": {"c1": {"portrait": {"file": "p1.png"}}, "c2": {"portrait": {"file": "p2.png"}}}}


def registry_in(tmp_path):
    import method_registry as mr
    return mr.Registry(tmp_path / "registry.json")


def test_every_shot_gets_the_right_clip_job(tmp_path):
    plan = {"characters": PEOPLE,
            "shots": [{"id": "s001", "kind": "narration", "speaker": "narrator", "in_shot": [], "visual": "the camera pushes in"},
                      {"id": "s002", "kind": "talk", "speaker": "c1", "in_shot": ["c1"], "visual": "speaks urgently"},
                      {"id": "s003", "kind": "choice", "speaker": "narrator", "in_shot": ["c1", "c2"], "visual": "waits tensely"}]}
    for shot in plan["shots"]:
        write_wav(tmp_path / "voices" / f"{shot['id']}.wav", 2.0)
    jobs = {j["id"]: j for j in sp.animate_jobs(plan, CARDS, tmp_path, registry_in(tmp_path))}
    assert set(jobs) == {"s001", "s002", "s003_c1", "s003_c2"}
    assert jobs["s001"]["voice"] is None and jobs["s001"]["engine"] == "s2v" and jobs["s001"]["source"] == tmp_path / "stills" / "s001.png" and "the camera pushes in" in jobs["s001"]["prompt"]  # a place: S2V is proven
    assert jobs["s002"]["engine"] == "s2v" and jobs["s002"]["voice"] == tmp_path / "voices" / "s002.wav" and jobs["s002"]["source"].name == "p1.png" and "Brandt" in jobs["s002"]["prompt"]  # S2V only to speak
    assert jobs["s002"]["seconds"] == 2.25 and jobs["s003_c2"]["voice"] is None and jobs["s003_c2"]["source"].name == "p2.png"
    assert jobs["s003_c2"]["engine"] == "i2v" and "blinks naturally" in jobs["s003_c2"]["prompt"]  # no two-shot yet: each one waits in an I2V clip (it blinks)
    assert jobs["s003_c1"]["seconds"] == 4.8  # the last shot also keeps the screen for the closing question (2.8 s)


def test_people_move_with_i2v_the_choice_is_the_two_shot_and_the_second_offer_goes_on_from_the_first(tmp_path):
    plan = {"characters": PEOPLE,
            "shots": [{"id": "s001", "kind": "offer", "speaker": "narrator", "offer_of": "c1", "in_shot": ["c1", "c2"], "visual": "both hold out a hand", "motion": "both hold out a hand"},
                      {"id": "s002", "kind": "offer", "speaker": "narrator", "offer_of": "c2", "in_shot": ["c1", "c2"], "visual": "both hold out a hand", "motion": "both hold out a hand"},
                      {"id": "s003", "kind": "narration", "speaker": "narrator", "in_shot": ["c2"], "visual": "she runs", "motion": "she sprints down the corridor, the camera follows her"},
                      {"id": "s004", "kind": "pov", "speaker": "narrator", "in_shot": [], "visual": "x", "motion": "your hands crank the valve wheel, steam bursts out"},
                      {"id": "s005", "kind": "choice", "speaker": "narrator", "in_shot": ["c1", "c2"], "visual": "waits", "choice": {"a": "Brandt", "b": "Ilse"}}]}
    for shot in plan["shots"]:
        write_wav(tmp_path / "voices" / f"{shot['id']}.wav", 2.0)
    (tmp_path / "stills_id").mkdir()
    (tmp_path / "stills_id" / "two_shot.png").write_bytes(b"x")
    (tmp_path / "stills_id" / "s003.png").write_bytes(b"x")
    jobs = {j["id"]: j for j in sp.animate_jobs(plan, CARDS, tmp_path, registry_in(tmp_path))}
    assert set(jobs) == {"s001", "s002", "s003", "s004", "s005"}  # the choice is ONE clip now
    assert all(j["engine"] == "i2v" for j in jobs.values())
    assert jobs["s001"]["source"].name == "two_shot.png" and jobs["s001"]["prompt"].startswith("Brandt, the man in a heavy thermal suit, on the left, and Ilse, the woman in a white coat, on the right, slowly stretch their open hands") and "chain_from" not in jobs["s001"]
    assert jobs["s002"]["chain_from"] == "s001" and "Brandt, the man in" in jobs["s002"]["prompt"] and "keep holding out their open hands" in jobs["s002"]["prompt"]
    assert jobs["s003"]["source"].name == "s003.png" and jobs["s003"]["style"] == "follow" and "the camera follows the movement" in jobs["s003"]["prompt"] and "sprints down the corridor" in jobs["s003"]["prompt"] and jobs["s003"]["prompt"].startswith("Ilse, the woman in a white coat.") and "real steps" in jobs["s003"]["prompt"]
    assert jobs["s004"]["style"] == "pov" and jobs["s004"]["prompt"] == "your hands crank the valve wheel, steam bursts out"
    assert jobs["s005"]["source"].name == "two_shot.png" and jobs["s005"]["method"] == "i2v_offer_hands"  # the choice moment: always plan A


def test_each_job_is_made_by_its_engine_and_a_clip_that_exists_is_kept(tmp_path, monkeypatch):
    plan = {"characters": PEOPLE,
            "shots": [{"id": "s001", "kind": "talk", "speaker": "c1", "in_shot": ["c1"], "visual": "he shouts an order"},
                      {"id": "s002", "kind": "narration", "speaker": "narrator", "in_shot": ["c2"], "visual": "x", "motion": "she runs down the corridor"}]}
    for shot in plan["shots"]:
        write_wav(tmp_path / "voices" / f"{shot['id']}.wav", 2.0)
    (tmp_path / "stills").mkdir()
    (tmp_path / "stills" / "s002.png").write_bytes(b"x")
    made = []

    def write(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return {"seconds": 1.0, "steps": 4}
    monkeypatch.setattr(sp.st, "run_talk", lambda source, audio, target, **kw: made.append(("s2v", target.stem)) or write(target))
    monkeypatch.setattr(sp.i2v, "run_i2v", lambda start, target, **kw: made.append(("i2v", target.stem, kw["profile"], kw["seconds"])) or write(target))
    monkeypatch.setattr(sp, "animate_jobs", lambda plan, cards, run, registry=None, _real=sp.animate_jobs: _real(plan, cards, run, registry_in(tmp_path)))
    done = sp.step_animate(plan, CARDS, tmp_path, "http://x", "t2v_1217_low", 1)
    assert made == [("s2v", "s001"), ("i2v", "s002", "lightx2v4", 5.0)] and done["s002"]["engine"] == "i2v" and done["s001"]["engine"] == "s2v"
    assert sp.step_animate(plan, CARDS, tmp_path, "http://x", "t2v_1217_low", 1) == {}  # nothing is made twice


def test_the_other_way_of_making_the_choice_moment_is_kept_and_selectable(tmp_path, monkeypatch):
    plan = {"characters": PEOPLE, "shots": [{"id": "s001", "kind": "choice", "speaker": "narrator", "in_shot": ["c1", "c2"], "visual": "waits", "choice": {"a": "Brandt", "b": "Ilse"}}]}
    write_wav(tmp_path / "voices" / "s001.wav", 2.0)
    (tmp_path / "stills_id").mkdir()
    (tmp_path / "stills_id" / "two_shot.png").write_bytes(b"x")
    assert sp.animate_jobs(plan, CARDS, tmp_path, registry_in(tmp_path))[0]["method"] == "i2v_offer_hands"
    monkeypatch.setattr(sp, "OFFER_METHOD", "i2v_offer_step")
    step = sp.animate_jobs(plan, CARDS, tmp_path, registry_in(tmp_path))[0]
    assert step["method"] == "i2v_offer_step" and "step forward together" in step["prompt"]


class FakeVoiceBox:
    """What the voice samples step needs of a box: `run` (a listing of /root/voices) and `put`."""

    def __init__(self, listing: str = ""):
        self.listing, self.commands, self.sent = listing, [], []

    def run(self, command: str, timeout: float = 0) -> str:
        self.commands.append(command)
        return self.listing if "find" in command else ""

    def put(self, local: Path, remote: str) -> None:
        self.sent.append((local.name, remote))


def test_the_recorded_voices_the_box_lacks_are_sent_and_the_ones_it_has_are_not(tmp_path):
    (tmp_path / "dan.mp3").write_bytes(b"d" * 100)
    (tmp_path / "siren.mp3").write_bytes(b"s" * 200)
    voices = [{"id": "dan", "file": "dan.mp3"}, {"id": "siren", "file": "siren.mp3"}, {"id": "ghost", "file": "ghost.mp3"}, {"id": "design"}]
    box = FakeVoiceBox("dan.mp3 100\nsiren.mp3 7\n")  # dan is already on the box with the right size, siren has another size
    sent = sp.ensure_voice_samples(box, voices, local_dir=tmp_path)
    assert sent == ["siren.mp3"]
    assert box.sent == [("siren.mp3", "/root/voices/siren.mp3")]


def test_a_new_box_gets_every_recorded_voice_and_nothing_is_sent_without_a_sample(tmp_path):
    (tmp_path / "dan.mp3").write_bytes(b"d" * 100)
    assert sp.ensure_voice_samples(FakeVoiceBox(""), [{"id": "dan", "file": "dan.mp3"}], local_dir=tmp_path) == ["dan.mp3"]
    box = FakeVoiceBox("")
    assert sp.ensure_voice_samples(box, [{"id": "x", "file": "missing.mp3"}, {"id": "design"}], local_dir=tmp_path) == []
    assert box.commands == [] and box.sent == []  # no sample on the PC: the box is not even called
