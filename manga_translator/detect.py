from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .models import TextRegion


def _poly_to_bbox(poly: list[list[float]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _is_vertical(bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    return (y2 - y1) > (x2 - x1) * 1.4


def _sort_regions(regions: list[TextRegion]) -> list[TextRegion]:
    return sorted(regions, key=lambda r: (r.bbox[1] // 32, r.bbox[0]))


@lru_cache(maxsize=1)
def get_paddle_ocr():
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "PaddleOCR is not installed. Install PaddlePaddle CPU first, then paddleocr."
        ) from exc

    try:
        return PaddleOCR(lang="japan", use_doc_orientation_classify=False, use_doc_unwarping=False)
    except TypeError:  # PaddleOCR 2.x compatibility.
        return PaddleOCR(lang="japan", use_angle_cls=False, show_log=False)


def _extract_polys_and_scores(result) -> list[tuple[list[list[float]], float | None]]:
    extracted: list[tuple[list[list[float]], float | None]] = []
    for page in result or []:
        if isinstance(page, dict):
            boxes = page.get("dt_polys") or page.get("rec_polys") or []
            scores = page.get("dt_scores") or page.get("rec_scores") or []
            for idx, box in enumerate(boxes):
                poly = [[float(x), float(y)] for x, y in np.asarray(box).tolist()]
                score = float(scores[idx]) if idx < len(scores) else None
                extracted.append((poly, score))
            continue

        for item in page or []:
            box = item[0] if isinstance(item, (list, tuple)) and item else item
            score = None
            if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], (list, tuple)):
                if len(item[1]) > 1 and isinstance(item[1][1], (float, int)):
                    score = float(item[1][1])
            poly = [[float(x), float(y)] for x, y in np.asarray(box).tolist()]
            extracted.append((poly, score))
    return extracted


def _detect_text_regions_paddle(image: Image.Image) -> list[TextRegion]:
    ocr = get_paddle_ocr()
    rgb = image.convert("RGB")
    try:
        result = ocr.predict(np.array(rgb))
    except AttributeError:
        result = ocr.ocr(np.array(rgb), det=True, rec=False, cls=False)
    regions: list[TextRegion] = []

    for poly, score in _extract_polys_and_scores(result):
        bbox = _poly_to_bbox(poly)
        regions.append(
            TextRegion(
                id=len(regions) + 1,
                box=poly,
                bbox=bbox,
                polygon=poly,
                vertical=_is_vertical(bbox),
                confidence=score,
            )
        )

    sorted_regions = _sort_regions(regions)
    for idx, region in enumerate(sorted_regions, start=1):
        region.id = idx
    return sorted_regions


def _detect_text_regions_ctd(image: Image.Image) -> list[TextRegion]:
    try:
        from rusty_manga_image_translator import detect_text_regions as rust_detect
    except Exception as exc:  # pragma: no cover - optional quality bridge
        raise RuntimeError("CTD detector bridge is not installed; falling back to PaddleOCR.") from exc

    result = rust_detect(image.convert("RGB"))
    regions: list[TextRegion] = []
    for item in result or []:
        poly = item.get("polygon") or item.get("box") or []
        if not poly and "bbox" in item:
            x1, y1, x2, y2 = item["bbox"]
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if not poly:
            continue
        poly = [[float(x), float(y)] for x, y in poly]
        bbox = _poly_to_bbox(poly)
        regions.append(
            TextRegion(
                id=len(regions) + 1,
                box=poly,
                bbox=bbox,
                polygon=poly,
                vertical=_is_vertical(bbox),
                confidence=item.get("confidence"),
            )
        )
    return _sort_regions(regions)


def detect_text_regions(image: Image.Image, backend: str = "paddle") -> tuple[list[TextRegion], list[str]]:
    backend = (backend or "paddle").strip().lower()
    notes: list[str] = []
    if backend in {"ctd", "quality"}:
        try:
            regions = _detect_text_regions_ctd(image)
            for idx, region in enumerate(regions, start=1):
                region.id = idx
            return regions, notes
        except RuntimeError as exc:
            notes.append(str(exc))
    regions = _detect_text_regions_paddle(image)
    return regions, notes


def draw_region_overlay(image: Image.Image, regions: list[TextRegion]) -> Image.Image:
    out = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(out)
    for region in regions:
        points = [tuple(point) for point in region.box]
        draw.polygon(points, outline=(255, 64, 64, 255))
        x1, y1, _, _ = region.bbox
        draw.rectangle((x1, max(0, y1 - 18), x1 + 42, y1), fill=(255, 64, 64, 220))
        draw.text((x1 + 3, max(0, y1 - 17)), str(region.id), fill=(255, 255, 255, 255))
    return out


def crop_region(image: Image.Image, region: TextRegion, padding: int = 4) -> Image.Image:
    x1, y1, x2, y2 = region.bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)
    return image.crop((x1, y1, x2, y2))
