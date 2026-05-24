import numpy as np
from PIL import Image

from manga_translator import inpaint
from manga_translator.models import TextRegion


def _region():
    return TextRegion(id=1, box=[[1, 1], [4, 1], [4, 4], [1, 4]], bbox=(1, 1, 4, 4))


def test_opencv_telea_uses_telea_flag(monkeypatch):
    flags = []

    def fake_inpaint(array, mask, radius, flag):
        flags.append(flag)
        return array

    monkeypatch.setattr(inpaint.cv2, "inpaint", fake_inpaint)
    image = Image.new("RGB", (8, 8), "white")

    inpaint.inpaint_text(image, [_region()], backend="opencv-telea")

    assert flags == [inpaint.cv2.INPAINT_TELEA]


def test_opencv_ns_uses_ns_flag(monkeypatch):
    flags = []

    def fake_inpaint(array, mask, radius, flag):
        flags.append(flag)
        return array

    monkeypatch.setattr(inpaint.cv2, "inpaint", fake_inpaint)
    image = Image.new("RGB", (8, 8), "white")

    inpaint.inpaint_text(image, [_region()], backend="opencv-ns")

    assert flags == [inpaint.cv2.INPAINT_NS]


def test_neural_inpainter_requires_downloaded_model(tmp_path):
    image = Image.new("RGB", (8, 8), "white")
    missing = tmp_path / "missing.pt"

    try:
        inpaint.inpaint_text(image, [_region()], backend="migan", model_path=str(missing))
    except RuntimeError as exc:
        assert "Download inpainter model first" in str(exc)
    else:
        raise AssertionError("Expected missing neural inpainter model to fail")


def test_make_text_mask_ignores_disabled_regions():
    region = _region()
    region.enabled = False

    mask = inpaint.make_text_mask((8, 8), [region], padding=2)

    assert np.count_nonzero(mask) == 0
