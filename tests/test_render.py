from PIL import Image, ImageDraw

from manga_translator.models import TextRegion
from manga_translator.render import (
    ABSOLUTE_MIN_FONT_SIZE,
    READABLE_MIN_FONT_SIZE,
    SKIPPED_RENDER_NOTE,
    SQUEEZED_RENDER_NOTE,
    expanded_text_box,
    fit_text,
    is_vertical_region,
    load_font,
    render_translations,
    stack_to_box,
    text_spacing,
    wrap_to_box,
)


def _region(bbox=(20, 20, 80, 60), text="Hello"):
    x1, y1, x2, y2 = bbox
    return TextRegion(
        id=1,
        box=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        bbox=bbox,
        source_text="source",
        translated_text=text,
    )


def test_fit_text_does_not_shrink_below_readable_minimum():
    image = Image.new("RGB", (220, 120), "white")
    draw = ImageDraw.Draw(image)

    fit = fit_text(draw, "Readable text", 180, 70, "", preferred_size=20, auto_size=True)

    assert fit.fits is True
    assert fit.font.size >= READABLE_MIN_FONT_SIZE


def test_expanded_text_box_is_clamped_to_image_bounds():
    region = _region((2, 2, 18, 80), "Narrow bubble")
    region.vertical = True

    box = expanded_text_box(region, (64, 96))

    assert box[0] == 0
    assert box[1] >= 0
    assert box[2] <= 64
    assert box[3] <= 96
    assert box[2] - box[0] > region.bbox[2] - region.bbox[0]


def test_unreadable_long_translation_is_skipped_and_noted():
    region = _region((30, 30, 45, 45), "This translation is much too long to fit here clearly")
    image = Image.new("RGB", (80, 80), "white")

    render_translations(image, [region], font_size=20, auto_font_size=True, overflow_text=False)

    assert SKIPPED_RENDER_NOTE in region.notes


def test_overflow_text_keeps_tiny_translation_visible_and_noted():
    region = _region((30, 30, 45, 45), "This translation is much too long to fit here clearly")
    image = Image.new("RGB", (80, 80), "white")

    render_translations(image, [region], font_size=20, auto_font_size=True, overflow_text=True)

    assert SQUEEZED_RENDER_NOTE in region.notes


def test_wrap_to_box_preserves_manual_line_breaks():
    image = Image.new("RGB", (240, 100), "white")
    draw = ImageDraw.Draw(image)
    font = load_font("", 18)

    wrapped = wrap_to_box(draw, "First line\nSecond line", font, 220)

    assert "First line\nSecond line" == wrapped


def test_vertical_regions_use_stacked_words():
    image = Image.new("RGB", (120, 180), "white")
    draw = ImageDraw.Draw(image)
    font = load_font("", 16)

    stacked = stack_to_box(draw, "Like you sir", font, 80)

    assert stacked == "Like\nyou\nsir"


def test_fit_text_renders_vertical_region_above_slider_minimum():
    image = Image.new("RGB", (120, 180), "white")
    draw = ImageDraw.Draw(image)

    fit = fit_text(draw, "Like you sir", 80, 120, "", preferred_size=18, auto_size=True, vertical=True)

    assert fit.fits is True
    assert fit.vertical is True
    assert fit.font.size >= READABLE_MIN_FONT_SIZE
    assert fit.text == "Like\nyou\nsir"


def test_manual_line_breaks_are_preserved_in_vertical_layout():
    image = Image.new("RGB", (120, 180), "white")
    draw = ImageDraw.Draw(image)
    font = load_font("", 16)

    stacked = stack_to_box(draw, "First line\nSecond line", font, 90)

    assert stacked == "First\nline\nSecond\nline"


def test_tall_regions_are_treated_as_vertical_even_without_flag():
    region = _region((40, 10, 65, 100), "Tall text")

    assert is_vertical_region(region) is True


def test_line_spacing_scales_with_font_size():
    small = load_font("", 12)
    large = load_font("", 24)

    assert text_spacing(small) >= 5
    assert text_spacing(large) > text_spacing(small)


def test_dense_vertical_layout_is_rejected():
    image = Image.new("RGB", (120, 180), "white")
    draw = ImageDraw.Draw(image)

    fit = fit_text(
        draw,
        "one two three four five six seven eight nine ten",
        70,
        55,
        "",
        preferred_size=18,
        auto_size=True,
        vertical=True,
    )

    assert fit.fits is False


def test_auto_fit_finds_readable_size_when_spacing_allows():
    image = Image.new("RGB", (160, 180), "white")
    draw = ImageDraw.Draw(image)

    fit = fit_text(draw, "Like you sir", 100, 130, "", preferred_size=24, auto_size=True, vertical=True)

    assert fit.fits is True
    assert fit.font.size >= READABLE_MIN_FONT_SIZE


def test_allow_unreadable_can_fit_below_readable_minimum():
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)

    fit = fit_text(
        draw,
        "one two three four",
        45,
        35,
        "",
        preferred_size=18,
        auto_size=True,
        allow_unreadable=True,
    )

    assert fit.fits is True
    assert ABSOLUTE_MIN_FONT_SIZE <= fit.font.size < READABLE_MIN_FONT_SIZE
