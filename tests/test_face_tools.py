import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import face_tools as ft  # noqa: E402


def astronaut(scale: float = 1.0) -> Image.Image:
    from skimage import data
    image = Image.fromarray(data.astronaut())
    return image.resize((int(image.width * scale), int(image.height * scale)))


def test_a_real_face_is_found_and_boxed_inside_the_picture():
    faces = ft.detect_faces(astronaut())
    assert faces and 0.0 < faces[0]["cx"] < 1.0 and 0.05 < faces[0]["w"] < 0.6
    x0, y0, x1, y1 = ft.face_box(faces[0])
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


def test_a_picture_without_a_face_gives_nothing(tmp_path):
    flat = Image.fromarray((np.random.default_rng(1).random((300, 300)) * 255).astype("uint8"))
    flat.save(tmp_path / "noise.png")
    assert ft.pick_closeup([tmp_path / "noise.png"]) is None


def test_the_centred_face_beats_the_off_centre_one_and_the_too_small_one():
    centred = {"cx": 0.5, "cy": 0.4, "w": 0.38, "h": 0.25}
    side = {"cx": 0.8, "cy": 0.4, "w": 0.38, "h": 0.25}
    tiny = {"cx": 0.5, "cy": 0.4, "w": 0.12, "h": 0.08}
    assert ft.face_score(centred) > ft.face_score(side) and ft.face_score(centred) > ft.face_score(tiny)
