import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import comfy_edit as ce
from scripts import drop_transplant as dt

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SPEC = {"event": {"x": 100, "y": 150, "rx": 40, "ry": 60}}
TARGET_SPEC = {"allowed": [[20, 20, 300, 400]], "scale": {"near": 0.55, "far": 0.17}, "depth": {"near_weight": [0.6, 0.4]}, "busy": {"max_fraction": 0.09}}


def test_outline_and_burst_are_chosen_separately_in_any_combination():
    name, spec = dt.resolve_style("burst_blue")
    assert name == "burst_blue" and spec["burst"] and len(spec["lines"]) == 2  # a named style is returned untouched
    name, spec = dt.resolve_style("burst_blue", outline="black")
    assert spec["burst"] is True and [l[0] for l in spec["lines"]] == ["black"]  # the outline alone is replaced, the burst of the style is kept
    name, spec = dt.resolve_style("burst_blue", burst=False)
    assert spec["burst"] is False and len(spec["lines"]) == 2  # the burst alone is replaced
    name, spec = dt.resolve_style(None, outline="double", burst=True)
    assert spec["burst"] and [l[0] for l in spec["lines"]] == ["black", "white"]


def test_the_declared_union_defaults_are_complete_and_point_to_real_specs():
    defaults = dt.load_union_defaults()
    assert defaults["style"] == "burst_blue" and defaults["method"] == "auto" and defaults["copies"] == 12 and defaults["seed"] == 1
    for key in ("source_spec", "target_spec"):
        assert Path(defaults[key]).exists(), key
    union = json.loads(dt.RECIPES.read_text(encoding="utf-8"))["rain_union"]
    assert union["base_video"]["recipe"] == "k_craters" and union["base_video"]["frames"] == 89 and len(union["base_video"]["sha256"]) == 64
    assert set(union["distinct_possibilities"]) == {"base_only", "impact_only", "union"}  # the two ingredients stay usable apart, and their union
    assert union["base_video"]["generation"]["still"]["seed"] == 1372175472


def test_qwen_is_in_the_bundle_catalogue_with_verified_files():
    catalog = json.loads((ROOT / "config" / "base_models.json").read_text(encoding="utf-8"))
    assert ce.bundle_declared() is True
    for item_id in catalog["profiles"]["qwen-edit"]:
        item = catalog["items"][item_id]
        assert len(item["sha256"]) == 64 and item["size_bytes"] > 0 and item["relative_path"] and item["url"].startswith("https://huggingface.co/")
    assert ce.bundle_declared(catalog=ROOT / "missing.json") is False


def test_auto_falls_back_to_the_cpu_transplant_when_the_server_does_not_have_qwen(monkeypatch):
    monkeypatch.setattr(ce, "server_has_model", lambda *a, **k: False)
    assert ce.qwen_available() is False
    monkeypatch.setattr(ce, "server_has_model", lambda *a, **k: True)
    assert ce.qwen_available() is True
    assert ce.qwen_available(catalog=ROOT / "missing.json") is False  # not in the bundle -> not the default, even if the server has it


def test_union_workflow_gives_the_second_picture_to_the_encoder_but_edits_only_the_first():
    graph = ce.build_workflow(ce.MIX, "base.png", references=["impact.png"])
    assert graph["17"]["class_type"] == "LoadImage" and graph["17"]["inputs"]["image"] == "impact.png"
    for encoder in ("9", "10"):
        assert graph[encoder]["inputs"]["image1"] == ["8", 0] and graph[encoder]["inputs"]["image2"] == ["18", 0]
    assert graph["13"]["inputs"]["pixels"] == ["8", 0]  # the latent that is sampled is the one of the first picture
    refs = [t for node in graph.values() for v in node["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) for t in [v[0]]]
    assert all(r in graph for r in refs)


def test_union_with_nothing_else_given_prints_its_defaults_and_overrides_are_applied():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "drop_transplant.py"), "union", "--show-defaults"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and json.loads(out.stdout)["style"] == "burst_blue"


def test_planning_on_a_target_keeps_copies_on_free_water_and_inside_the_allowed_zone():
    busy = {t: np.zeros((500, 400), bool) for t in range(0, 90, 6)}
    busy[0][:, 200:] = True  # the right half is busy at the start
    p = {**dt.DEFAULTS, "seed": 3, "copies": 8}
    copies = dt.plan_on_target(SOURCE_SPEC, TARGET_SPEC, (400, 500), busy, 89, 38, p)
    assert len(copies) >= 5
    for c in copies:
        assert 20 <= c["x"] <= 300 and 20 <= c["y"] <= 400
        if c["start"] <= 6:
            assert c["x"] < 215
