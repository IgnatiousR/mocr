from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .models import TextRegion


def make_text_mask(size: tuple[int, int], regions: list[TextRegion], padding: int) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        if not region.enabled:
            continue
        x1, y1, x2, y2 = region.bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def inpaint_text(image: Image.Image, regions: list[TextRegion], padding: int = 8, radius: int = 3) -> Image.Image:
    original_mode = image.mode
    rgb = image.convert("RGB")
    array = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    mask = make_text_mask(rgb.size, regions, padding)
    cleaned = cv2.inpaint(array, mask, radius, cv2.INPAINT_TELEA)
    cleaned_rgb = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    if original_mode == "RGBA":
        cleaned_rgb.putalpha(image.getchannel("A"))
    return cleaned_rgb
