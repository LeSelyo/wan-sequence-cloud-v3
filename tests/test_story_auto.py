import json
import sys
import types
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import story_auto as sa  # noqa: E402

PLAN = {"characters": [{"id": "c1", "name": "A", "age": 30, "gender": "m", "role": "captain"}, {"id": "c2", "name": "B", "age": 50, "gender": "f", "role": "doctor"}],
        "shots": [{"id": "s001", "kind": "narration", "speaker": "narrator", "in_shot": [], "visual": "push", "still": "a city", "text": "x"},
                  {"id": "s002", "kind": "talk", "speaker": "c1", "in_shot": ["c1"], "visual": "speaks", "still": "", "text": "y"},
                  {"id": "s003", "kind": "choice", "speaker": "narrator", "in_shot": ["c1", "c2"], "visual": "waits", "still": "", "text": "z", "choice": {"a": "A", "b": "B"}}],
        "caption": "hello #fyp"}


def wav(path: Path, seconds: float = 1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x01\x00" * int(24000 * seconds))


def make(tmp_path, monkeypatch) -> sa.Pipeline:
    monkeypatch.setattr(sa, "AUTO", tmp_path)
    return sa.Pipeline("demo", "box.invalid", 22, "key")


def test_a_new_project_waits_for_its_plan_then_each_step_reports_what_is_missing(tmp_path, monkeypatch):
    pipeline = make(tmp_path, monkeypatch)
    status = pipeline.status()
    assert status["story"][0] is False and all(not done for step, (done, _) in status.items())
    pipeline.plan_path.write_text(json.dumps(PLAN), encoding="utf-8")
    (pipeline.cards_dir).mkdir(parents=True)
    cards = {"characters": {"c1": {"portrait": {"file": "p1.png"}, "closeup": {"file": "c1.png"}}, "c2": {"portrait": {"file": "p2.png"}}}, "style": {}}
    (pipeline.cards_dir / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
    status = pipeline.status()
    assert status["story"][0] and status["cards"][0] and status["closeups"] == (False, "1/2 close-ups picked") and status["stills"] == (False, "0/1 stills")
    cards["characters"]["c2"]["closeup"] = {"file": "c2.png"}
    (pipeline.cards_dir / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
    (pipeline.run / "stills").mkdir(parents=True)
    (pipeline.run / "stills" / "s001.png").write_bytes(b"x")
    for shot in PLAN["shots"]:
        wav(pipeline.run / "voices" / f"{shot['id']}.wav")
    (pipeline.run / "voices" / "align.json").write_text("{}", encoding="utf-8")
    status = pipeline.status()
    assert status["closeups"][0] and status["stills"][0] and status["voices"][0] and status["animate"] == (False, "0/4 clips")  # s001, s002 and the two waiting clips of the choice
    for job in ("s001", "s002", "s003_c1", "s003_c2"):
        (pipeline.run / "clips").mkdir(exist_ok=True)
        (pipeline.run / "clips" / f"{job}.mp4").write_bytes(b"x")
    assert pipeline.status()["animate"][0] and not pipeline.status()["post"][0] and not pipeline.status()["render"][0]


def test_the_run_skips_what_is_done_switches_the_app_family_and_records_every_step(tmp_path, monkeypatch):
    pipeline = make(tmp_path, monkeypatch)
    calls, switches = [], []
    monkeypatch.setattr(pipeline, "ensure_tunnels", lambda: None)
    monkeypatch.setattr(pipeline, "switch", lambda family: switches.append(family))
    state = {step: (step != "stills" and step != "animate" and step != "caption", "") for step in sa.STEPS}
    monkeypatch.setattr(pipeline, "status", lambda: state)
    monkeypatch.setattr(pipeline, "step_stills", lambda args: calls.append("stills") or {"stills": 3}, raising=False)
    monkeypatch.setattr(pipeline, "step_animate", lambda args: calls.append("animate") or {"x": 1}, raising=False)
    monkeypatch.setattr(pipeline, "step_caption", lambda args: calls.append("caption") or {}, raising=False)
    pipeline.go(types.SimpleNamespace(steps=None, force=False))
    assert calls == ["stills", "animate", "caption"] and switches == ["krea2", "s2v"]  # the done steps (story, cards, ...) are not run again; caption needs no family
    report = json.loads(pipeline.report_path.read_text(encoding="utf-8"))
    assert set(report["steps"]) == {"stills", "animate", "caption"} and report["steps"]["stills"]["stills"] == 3 and "seconds" in report["steps"]["animate"]


def test_every_step_that_needs_the_gpu_family_is_mapped_and_the_order_is_the_documented_one():
    assert sa.STEPS == ("story", "cards", "closeups", "stills", "voices", "animate", "qc", "post", "render", "caption")
    assert sa.FAMILY == {"cards": "krea2", "closeups": "krea2", "stills": "krea2", "animate": "s2v", "qc": "s2v", "post": "s2v"}


def test_a_wrong_picture_is_made_again_with_another_seed_after_the_model_is_unloaded(tmp_path, monkeypatch):
    pipeline = make(tmp_path, monkeypatch)
    plan = {**PLAN, "brief": {"world": {"setting": "a station", "atmosphere": "cold", "hour": "night"}}, "locations": [{"id": "deck", "description": "the deck"}]}
    plan["shots"] = [dict(PLAN["shots"][0], location="deck")]
    pipeline.plan_path.write_text(json.dumps(plan), encoding="utf-8")
    pipeline.cards_dir.mkdir(parents=True)
    (pipeline.cards_dir / "cards.json").write_text(json.dumps({"characters": {}, "style": {}}), encoding="utf-8")
    (pipeline.run / "stills").mkdir(parents=True)
    (pipeline.run / "stills" / "s001.png").write_bytes(b"x")
    events = []
    answers = iter([{"place_seen": "a sunlit arcade", "sunlight_or_blue_sky": True, "matches_world": 1, "action_visible": True, "collage_or_split_panels": False, "garbled_text": False, "problems": []},
                    {"place_seen": "a metal deck", "sunlight_or_blue_sky": False, "matches_world": 5, "action_visible": True, "collage_or_split_panels": False, "garbled_text": False, "problems": []}])

    def fake_ask(prompt, schema, image, seed):
        events.append("judge")
        return next(answers)
    fake_ask.unload = lambda: events.append("unload")
    monkeypatch.setattr(sa.sj, "ollama_vision", lambda model: fake_ask)
    monkeypatch.setattr(sa.sst, "make_stills", lambda plan, cards, out, only=None, seed_shift=0: events.append(f"make {sorted(only)} +{seed_shift}"))
    result = pipeline.check_stills(types.SimpleNamespace(model="m"))
    assert events == ["judge", "unload", "make ['s001'] +1000", "judge", "unload"]  # the GPU is freed before the pictures are made
    assert result["problems_left"] == [] and result["rounds"][0]["wrong"] == {"s001": ["sunlight or a blue sky in a world without daylight", "the place does not belong to the world (a sunlit arcade)"]}
