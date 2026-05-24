from pathlib import Path

from PIL import Image

from manga_translator.image_io import save_output_image, short_hash
from manga_translator.models import AppSettings, TextRegion
from manga_translator.pipeline import MangaTranslationPipeline, make_zip


def test_output_images_are_saved_in_category_folder(tmp_path):
    image = Image.new("RGB", (8, 8), "white")

    path = save_output_image(image, tmp_path, "final", "page_1", suffix=".png")

    assert Path(path).exists()
    assert Path(path).parent == tmp_path / "final"


def test_short_hash_distinguishes_same_named_files(tmp_path):
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "page.png"
    second = second_dir / "page.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert short_hash(first) != short_hash(second)


def test_composed_final_uses_original_name_translated(tmp_path, monkeypatch):
    source = tmp_path / "page one.png"
    Image.new("RGB", (16, 16), "white").save(source)
    pipeline = MangaTranslationPipeline()
    region = TextRegion(id=1, box=[[1, 1], [8, 1], [8, 8], [1, 8]], bbox=(1, 1, 8, 8), translated_text="Hi")

    monkeypatch.setattr("manga_translator.pipeline.inpaint_text", lambda image, regions, **kwargs: image)

    result = pipeline.compose_image(source, AppSettings(output_dir=str(tmp_path)), [region])

    assert Path(result.final_path).name == "page_one_translated.png"


def test_zip_entries_keep_translated_file_names(tmp_path):
    first = tmp_path / "first_translated.png"
    second = tmp_path / "second_translated.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    zip_path = make_zip([str(first), str(second)], tmp_path)

    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["first_translated.png", "second_translated.png"]
