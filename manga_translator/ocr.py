from __future__ import annotations

from functools import lru_cache

from PIL import Image

from .config import default_hardware_acceleration


@lru_cache(maxsize=1)
def get_manga_ocr():
    try:
        from manga_ocr import MangaOcr
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("Manga OCR is not installed. Install manga-ocr and torch CPU.") from exc
    
    mocr = MangaOcr()
    if default_hardware_acceleration() == "intel_xpu":
        import intel_extension_for_pytorch as ipex
        mocr.model.to("xpu")
    return mocr


def recognize_japanese(image: Image.Image) -> str:
    mocr = get_manga_ocr()
    return str(mocr(image)).strip()
