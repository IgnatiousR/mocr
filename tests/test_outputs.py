from pathlib import Path

from PIL import Image

from manga_translator.image_io import save_output_image, short_hash


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
