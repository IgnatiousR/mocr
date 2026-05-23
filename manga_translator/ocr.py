from __future__ import annotations

from functools import lru_cache

from PIL import Image


@lru_cache(maxsize=1)
def get_manga_ocr():
    try:
        from manga_ocr import MangaOcr
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("Manga OCR is not installed. Install manga-ocr and torch CPU.") from exc
    return MangaOcr()


def recognize_japanese(image: Image.Image) -> str:
    mocr = get_manga_ocr()
    return str(mocr(image)).strip()
