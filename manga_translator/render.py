from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import TextRegion


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, size=size)
    for candidate in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3, align="center")
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_to_box(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if not text:
        return ""
    avg_char_width = max(1, int(draw.textlength("ABCDEFGHIJKLMNOPQRSTUVWXYZ", font=font) / 26))
    wrap_width = max(4, max_width // avg_char_width)
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        lines.extend(textwrap.wrap(paragraph, width=wrap_width) or [""])
    return "\n".join(lines)


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_width: int,
    box_height: int,
    font_path: str,
    preferred_size: int,
    auto_size: bool,
) -> tuple[str, ImageFont.ImageFont]:
    min_size = 10
    sizes = range(preferred_size, min_size - 1, -1) if auto_size else [preferred_size]
    for size in sizes:
        font = load_font(font_path, size)
        wrapped = wrap_to_box(draw, text, font, box_width)
        width, height = _measure(draw, wrapped, font)
        if width <= box_width and height <= box_height:
            return wrapped, font
    font = load_font(font_path, min_size)
    return wrap_to_box(draw, text, font, box_width), font


def render_translations(
    image: Image.Image,
    regions: list[TextRegion],
    font_path: str = "",
    font_size: int = 28,
    auto_font_size: bool = True,
) -> Image.Image:
    out = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(out)
    for region in regions:
        text = region.translated_text.strip()
        if not region.enabled or not text:
            continue
        x1, y1, x2, y2 = region.bbox
        pad = 4
        box_width = max(8, x2 - x1 - pad * 2)
        box_height = max(8, y2 - y1 - pad * 2)
        wrapped, font = fit_text(draw, text, box_width, box_height, font_path, font_size, auto_font_size)
        text_width, text_height = _measure(draw, wrapped, font)
        text_x = x1 + (x2 - x1 - text_width) / 2
        text_y = y1 + (y2 - y1 - text_height) / 2
        draw.multiline_text(
            (text_x, text_y),
            wrapped,
            fill=(24, 24, 24, 255),
            font=font,
            spacing=3,
            align="center",
            stroke_width=1,
            stroke_fill=(255, 255, 255, 180),
        )
    return out
