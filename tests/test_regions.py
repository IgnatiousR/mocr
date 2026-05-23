from manga_translator.models import TextRegion
from manga_translator.pipeline import rows_to_regions


def test_rows_to_regions_applies_review_edits():
    source = "\u65e5\u672c\u8a9e"
    region = TextRegion(id=1, box=[[0, 0], [10, 0], [10, 10], [0, 10]], bbox=(0, 0, 10, 10))
    rows = [[1, False, source, "English", "", "skip"]]

    updated = rows_to_regions(rows, [region])

    assert updated[0].enabled is False
    assert updated[0].source_text == source
    assert updated[0].translated_text == "English"
    assert updated[0].notes == "skip"
