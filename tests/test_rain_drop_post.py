import numpy as np

from scripts import rain_drop_post as rp

IMPACTS = [{"x": 100, "y": 120, "core": 40}, {"x": 220, "y": 160, "core": 30}]


def test_cores_and_near_rings_are_kept_and_far_ones_are_calmed():
    p = {**rp.DEFAULTS, "reach": 20.0, "feather": 30.0}
    keep = rp.reach_weight(320, 240, IMPACTS, p)
    assert keep[120, 100] == 1.0 and keep[160, 220] == 1.0  # the two centres
    assert keep[0, 319] < 0.05  # far corner: calmed
    yy, xx = np.mgrid[0:240, 0:320]
    inside = np.sqrt(((xx - 100) / p["aspect"]) ** 2 + (yy - 120) ** 2) < 40 + p["reach"]
    assert keep[inside].min() > 0.99


def test_calming_never_moves_anything_and_removes_thin_strokes_far_away():
    flat = np.full((60, 80, 3), 120, np.uint8)
    line = flat.copy()
    line[:, 40] = 0  # a thin dark ring stroke
    free = rp.ring_free(line, 9)
    assert np.array_equal(rp.calm_far(line, free, np.ones((60, 80), np.float32), None), line)  # everything kept = untouched
    calmed = rp.calm_far(line, free, np.zeros((60, 80), np.float32), None)
    assert calmed[30, 40, 0] > 100  # the stroke is gone far from an impact
    assert abs(int(calmed[30, 10, 0]) - 120) <= 2  # the rest is unchanged: no geometry change


def test_rain_is_seeded_covers_the_whole_picture_and_spares_the_real_impacts():
    water = np.zeros((300, 200), np.float32)
    water[:, :120] = 1.0
    keep = rp.reach_weight(200, 300, [{"x": 60, "y": 150, "core": 40}], {**rp.DEFAULTS, "reach": 0.0, "feather": 20.0})
    p = {**rp.DEFAULTS, "seed": 5, "rain_rate": 200.0}
    first, second = rp.plan_rain(200, 300, 1.5, p, water, keep), rp.plan_rain(200, 300, 1.5, p, water, keep)
    assert first == second and first != rp.plan_rain(200, 300, 1.5, {**p, "seed": 6}, water, keep)
    assert len(first) == 300
    assert any(d["land"][0] > 150 for d in first)  # it rains beside the puddle too
    for d in first:
        if d["ripple"]:
            x, y = int(d["land"][0]), int(d["land"][1])
            assert water[min(y, 299), min(x, 199)] > 0.6 and keep[min(y, 299), min(x, 199)] < 0.2  # ripples only on water, never on a real impact


def test_drawing_leaves_the_frame_alone_when_nothing_is_alive_and_adds_white_when_a_drop_flies():
    frame = np.full((120, 80, 3), 90, np.uint8)
    far = [{"arrive": 5.0, "land": [20, 30], "direction": [1, 0], "depth": 1.0, "ripple": False}]
    assert np.array_equal(rp.draw_rain(frame, 0.0, far, rp.DEFAULTS, False), frame)
    flying = [{"arrive": 0.02, "land": [10, 60], "direction": [1, 0], "depth": 1.0, "ripple": False}]
    out = rp.draw_rain(frame, 0.0, flying, rp.DEFAULTS, False)
    assert out.astype(int).sum() > frame.astype(int).sum()  # mostly white added
    night = rp.draw_rain(frame, 0.0, flying, rp.DEFAULTS, True)
    assert night[..., 2].astype(int).sum() != out[..., 2].astype(int).sum()  # the night blue is not the day blue


def test_every_rain_style_is_kept_as_a_choice():
    assert {"general", "calm_targeted", "pinch"} <= set(rp.RAIN_STYLES)
    assert rp.RAIN_STYLES["pinch"]["set"]["warp"] == "pinch" and rp.RAIN_STYLES["pinch"]["set"]["speed"] == 1.5
    assert rp.RAIN_STYLES["calm_targeted"]["set"]["targeted"] == 14 and rp.RAIN_STYLES["calm_targeted"]["set"]["rain_rate"] == 0.0
    assert rp.RAIN_STYLES["general"]["set"] == {}


def test_pinch_style_still_protects_cores_and_targeted_drops_stay_outside_them():
    p = {**rp.DEFAULTS, **rp.RAIN_STYLES["pinch"]["set"]}
    xs, ys = rp.pinch_map(320, 240, IMPACTS, p)
    yy, xx = np.mgrid[0:240, 0:320]
    inside = np.sqrt(((xx - 100) / p["aspect"]) ** 2 + (yy - 120) ** 2) < 40 - 4
    assert np.abs(xs[inside] - xx[inside]).max() < 0.05
    for drop in rp.plan_targeted(IMPACTS, 2.0, p):
        target = next(i for i in IMPACTS if [i["x"], i["y"]] == drop["target"])
        distance = np.hypot((drop["end"][0] - target["x"]) / p["aspect"], drop["end"][1] - target["y"])
        assert distance >= target["core"] + p["stop_margin"] - 1.0


def test_process_refuses_to_overwrite_an_existing_version(tmp_path):
    import pytest
    out = tmp_path / "keep.mp4"
    out.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        rp.process(tmp_path / "in.mp4", out, None, {"style": "general"})
    assert out.read_bytes() == b"old"


def test_steady_styles_exist_and_the_steady_frame_keeps_cores_and_static_areas():
    assert {"steady", "steady_still", "steady_keep", "steady_first"} <= set(rp.RAIN_STYLES)
    rng = np.random.default_rng(1)
    frame = (rng.random((40, 60, 3)) * 255).astype(np.uint8)
    own = np.full_like(frame, 100)
    background = np.full_like(frame, 50)
    keep = np.zeros((40, 60), np.float32)
    core = np.zeros((40, 60), np.float32)
    core[:, :20] = 1.0  # a core on the left
    water = np.zeros((40, 60), np.float32)
    water[:, 10:] = 1.0  # static (not water) on the far left
    out = rp.steady_frame(frame, own, background, keep, core, water, 0.0)
    assert np.array_equal(out[:, :10], frame[:, :10])  # static area untouched
    assert np.array_equal(out[:, 10:20], frame[:, 10:20])  # core untouched even though it is water
    assert np.array_equal(out[:, 25:], background[:, 25:])  # elsewhere on the water: the steady painting, no ring strokes at strength 0
    kept = rp.steady_frame(frame, own, background, keep, core, water, 1.0)
    assert np.array_equal(kept[:, 25:], np.clip(background.astype(np.float32) + frame.astype(np.float32) - own.astype(np.float32), 0, 255).astype(np.uint8)[:, 25:])
