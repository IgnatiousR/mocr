from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import TextRegion

READABLE_MIN_FONT_SIZE = 12
ABSOLUTE_MIN_FONT_SIZE = 6
MIN_TEXT_SPACING = 5
RENDER_BOX_PAD = 6
SKIPPED_RENDER_NOTE = "translation skipped: text does not fit readable placement"
SQUEEZED_RENDER_NOTE = "translation squeezed below readable size to keep it visible"


@dataclass(frozen=True)
class FitResult:
    text: str
    font: ImageFont.ImageFont
    fits: bool
    vertical: bool = False
    readable: bool = True


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


def text_spacing(font: ImageFont.ImageFont, line_gap: int = 0) -> int:
    size = getattr(font, "size", READABLE_MIN_FONT_SIZE)
    return max(0, max(MIN_TEXT_SPACING, int(round(size * 0.32))) + line_gap)


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, line_gap: int = 0) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=text_spacing(font, line_gap), align="center")
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    if not text:
        return 0
    return draw.textlength(text, font=font)


def _split_long_word(draw: ImageDraw.ImageDraw, word: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for char in word:
        candidate = f"{current}{char}"
        if current and _text_width(draw, candidate, font) > max_width:
            pieces.append(current)
            current = char
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [word]


def _wrap_paragraph(draw: ImageDraw.ImageDraw, paragraph: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = paragraph.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        if _text_width(draw, word, font) <= max_width:
            current = word
            continue

        broken = _split_long_word(draw, word, font, max_width)
        lines.extend(broken[:-1])
        current = broken[-1]

    if current:
        lines.append(current)
    return lines


def wrap_to_box(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        lines.extend(_wrap_paragraph(draw, paragraph, font, max_width))
    return "\n".join(lines)


def _vertical_lines(text: str) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        if words:
            lines.extend(words)
        else:
            lines.append("")
    return lines


def stack_to_box(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    lines: list[str] = []
    for line in _vertical_lines(text):
        if _text_width(draw, line, font) <= max_width:
            lines.append(line)
            continue
        lines.extend(_split_long_word(draw, line, font, max_width))
    return "\n".join(lines)


def is_vertical_region(region: TextRegion) -> bool:
    if region.direction == "horizontal":
        return False
    if region.direction == "vertical":
        return True
    x1, y1, x2, y2 = region.bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return region.vertical or height > width * 1.4


def _parse_color(value: str, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    value = (value or "").strip()
    if not value:
        return fallback
    if value.startswith("#"):
        value = value[1:]
    if len(value) not in {6, 8}:
        return fallback
    try:
        channels = [int(value[idx : idx + 2], 16) for idx in range(0, len(value), 2)]
    except ValueError:
        return fallback
    if len(channels) == 3:
        channels.append(255)
    return tuple(channels)  # type: ignore[return-value]


def _fit_sizes(preferred_size: int, box_width: int, box_height: int, auto_size: bool, min_size: int = READABLE_MIN_FONT_SIZE) -> list[int]:
    if not auto_size:
        return [preferred_size]

    large_region_boost = min(6, max(0, min(box_width, box_height) // 48))
    max_size = max(preferred_size, preferred_size + large_region_boost)
    min_size = min(max_size, min_size)
    return list(range(max_size, min_size - 1, -1))


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def _is_too_dense(text: str, font: ImageFont.ImageFont, box_height: int, vertical: bool, line_gap: int = 0) -> bool:
    lines = _line_count(text)
    if lines <= 1:
        return False

    size = getattr(font, "size", READABLE_MIN_FONT_SIZE)
    spacing = text_spacing(font, line_gap)
    minimum_line_step = size + spacing
    if vertical and lines > max(3, box_height // max(1, minimum_line_step)):
        return True

    return lines * minimum_line_step > box_height * 0.92


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_width: int,
    box_height: int,
    font_path: str,
    preferred_size: int,
    auto_size: bool,
    vertical: bool = False,
    line_gap: int = 0,
    allow_unreadable: bool = False,
) -> FitResult:
    min_size = ABSOLUTE_MIN_FONT_SIZE if allow_unreadable else READABLE_MIN_FONT_SIZE
    sizes = _fit_sizes(preferred_size, box_width, box_height, auto_size, min_size=min_size)
    fallback_text = ""
    fallback_font = load_font(font_path, sizes[-1])
    for size in sizes:
        font = load_font(font_path, size)
        wrapped = stack_to_box(draw, text, font, box_width) if vertical else wrap_to_box(draw, text, font, box_width)
        width, height = _measure(draw, wrapped, font, line_gap)
        fallback_text = wrapped
        fallback_font = font
        readable = size >= READABLE_MIN_FONT_SIZE
        density_ok = not _is_too_dense(wrapped, font, box_height, vertical, line_gap) or allow_unreadable
        if width <= box_width and height <= box_height and density_ok:
            return FitResult(wrapped, font, True, vertical, readable)
    return FitResult(fallback_text, fallback_font, False, vertical, False)


def expanded_text_box(region: TextRegion, image_size: tuple[int, int], gap: int = RENDER_BOX_PAD) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x1, y1, x2, y2 = region.bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    expand_x = max(gap, int(width * 0.35))
    expand_y = max(gap, int(height * 0.18))
    if region.vertical or width < height * 0.65:
        target_width = min(image_width, max(width + expand_x * 2, int(min(height * 0.9, width * 3.0))))
        expand_x = max(expand_x, (target_width - width) // 2)
        expand_y = max(gap, int(height * 0.08))

    return (
        max(0, x1 - expand_x),
        max(0, y1 - expand_y),
        min(image_width, x2 + expand_x),
        min(image_height, y2 + expand_y),
    )


def _add_render_note(region: TextRegion, note: str) -> None:
    if note in region.notes:
        return
    region.notes = f"{region.notes}; {note}" if region.notes else note


def render_translations(
    image: Image.Image,
    regions: list[TextRegion],
    font_path: str = "",
    font_size: int = 28,
    auto_font_size: bool = True,
    text_box_gap: int = RENDER_BOX_PAD,
    line_gap: int = 0,
    overflow_text: bool = True,
    render_direction: str = "auto",
    font_color: str = "#181818",
    stroke_color: str = "#ffffff",
    stroke_width: int = 1,
) -> Image.Image:
    out = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(out)
    for region in regions:
        text = region.translated_text.strip()
        if not region.enabled or not text:
            continue
        x1, y1, x2, y2 = expanded_text_box(region, out.size, gap=text_box_gap)
        pad = text_box_gap
        box_width = max(8, x2 - x1 - pad * 2)
        box_height = max(8, y2 - y1 - pad * 2)
        region_vertical = is_vertical_region(region)
        if render_direction == "horizontal":
            region_vertical = False
        elif render_direction == "vertical":
            region_vertical = True
        fit = fit_text(
            draw,
            text,
            box_width,
            box_height,
            font_path,
            region.font_size or font_size,
            auto_font_size,
            vertical=region_vertical,
            line_gap=line_gap,
            allow_unreadable=overflow_text,
        )
        if not fit.fits:
            if not overflow_text:
                _add_render_note(region, SKIPPED_RENDER_NOTE)
                continue
            _add_render_note(region, SQUEEZED_RENDER_NOTE)
        elif not fit.readable:
            _add_render_note(region, SQUEEZED_RENDER_NOTE)
        text_width, text_height = _measure(draw, fit.text, fit.font, line_gap)
        text_x = x1 + (x2 - x1 - text_width) / 2
        text_y = y1 + (y2 - y1 - text_height) / 2
        draw.multiline_text(
            (text_x, text_y),
            fit.text,
            fill=_parse_color(region.font_color or font_color, (24, 24, 24, 255)),
            font=fit.font,
            spacing=text_spacing(fit.font, line_gap),
            align="center",
            stroke_width=region.stroke_width if region.stroke_width is not None else stroke_width,
            stroke_fill=_parse_color(region.stroke_color or stroke_color, (255, 255, 255, 180)),
        )
    return out
