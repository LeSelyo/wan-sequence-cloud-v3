import numpy as np

from scripts import drop_multiply as dm

SPEC = {"event": {"x": 100, "y": 150, "rx": 40, "ry": 60}, "inpaint": [{"x": 100, "y": 150, "rx": 40, "ry": 60}],
        "allowed": [[20, 20, 200, 400]], "depth": {"near_weight": [0.6, 0.4], "far_scale": 0.3}}


def test_copies_are_seeded_inside_the_allowed_water_and_smaller_when_far():
    p = {**dm.DEFAULTS, "seed": 4}
    first, second = dm.plan_copies(SPEC, 300, 500, p), dm.plan_copies(SPEC, 300, 500, p)
    assert first == second and first != dm.plan_copies(SPEC, 300, 500, {**p, "seed": 5})
    assert first[0]["scale"] == 1.0 and first[0]["x"] == 100  # the original splash is kept
    for c in first[1:]:
        assert 20 <= c["x"] <= 200 and 20 <= c["y"] <= 400
    near = dm.depth_scale(10, 10, 300, 500, SPEC["depth"])
    far = dm.depth_scale(290, 490, 300, 500, SPEC["depth"])
    assert near > far and far >= SPEC["depth"]["far_scale"] - 1e-6


def test_starts_cover_the_whole_clip_without_gaps():
    p = {**dm.DEFAULTS, "seed": 2}
    starts = sorted(c["start"] for c in dm.plan_copies(SPEC, 300, 500, p))
    total = int(p["seconds"] * p["fps"])
    assert starts[-1] > total * 0.6 and max(b - a for a, b in zip(starts, starts[1:])) < total * 0.2


def test_window_is_one_in_the_middle_and_zero_at_the_edge():
    w = dm.window(30, 40, 0.35)
    assert w[40, 30] == 1.0 and w[0, 0] == 0.0 and w[40, 0] < 0.05


def test_calm_water_keeps_the_pixels_around_the_hole_and_fills_the_hole_smoothly():
    rng = np.random.default_rng(0)
    base = np.full((128, 96, 3), (40, 70, 130), np.float32)
    img = base.copy()
    img[40:90, 20:70] = (5, 5, 20)  # a dark splash
    out = dm.calm_water(img, [{"x": 45, "y": 65, "rx": 28, "ry": 28}], margin=1.0)
    assert np.abs(out[:20] - img[:20]).max() < 1e-3  # far from the hole: untouched
    assert np.abs(out[65, 45] - base[65, 45]).max() < 25  # the hole now looks like the water around it
