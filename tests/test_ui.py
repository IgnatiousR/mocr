from pathlib import Path

from manga_translator import ui
from manga_translator.models import AppSettings, ImageJobResult, TextRegion
from manga_translator.ui import build_app, ram_usage_markdown, installed_font_choices


def test_build_app_with_dynamic_thread_slider(monkeypatch):
    monkeypatch.setattr("manga_translator.config.os.cpu_count", lambda: 4)

    app = build_app()

    assert app is not None


def test_installed_font_choices_finds_fonts_and_includes_extra_path(tmp_path):
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    arial = font_dir / "arial.ttf"
    custom = tmp_path / "custom.otf"
    ignored = font_dir / "readme.txt"
    arial.write_bytes(b"font")
    custom.write_bytes(b"font")
    ignored.write_text("not a font", encoding="utf-8")

    choices = installed_font_choices(extra_paths=[str(custom)], search_dirs=[font_dir])

    assert choices == [
        ("arial (arial.ttf)", str(arial)),
        ("custom (custom.otf)", str(custom)),
    ]


def test_ram_usage_markdown_includes_runtime_memory_fields():
    markdown = ram_usage_markdown()

    assert "### RAM Usage" in markdown
    assert "**State**" in markdown
    assert "**System**" in markdown
    assert "**Available**" in markdown
    assert "**This app**" in markdown


def test_download_translation_reports_missing_dependencies(monkeypatch):
    monkeypatch.setattr(ui, "missing_translation_dependencies", lambda backend: ["ctranslate2", "sentencepiece"])
    monkeypatch.setattr(ui, "download_translation_model", lambda backend: (_ for _ in ()).throw(AssertionError("should not download")))

    _, _, status = ui.download_translation_from_ui("sugoi", "model", "", "", "", "", "outputs")

    assert "Missing packages for sugoi: ctranslate2, sentencepiece" in status
    assert "requirements-translate.txt" in status


def test_download_translation_runs_when_dependencies_exist(monkeypatch, tmp_path):
    model = tmp_path / "model"
    monkeypatch.setattr(ui, "missing_translation_dependencies", lambda backend: [])
    monkeypatch.setattr(ui, "download_translation_model", lambda backend, destination=None: model)

    path, _, status = ui.download_translation_from_ui("sugoi", "", "", "", "", "", "outputs")

    assert path == str(model)
    assert f"Downloaded translation model: {model}" == status


def test_download_upscale_reports_missing_dependencies(monkeypatch):
    monkeypatch.setattr(ui, "missing_upscale_dependencies", lambda: ["realesrgan", "basicsr"])
    monkeypatch.setattr(ui, "download_realesrgan_model", lambda: (_ for _ in ()).throw(AssertionError("should not download")))

    _, _, status = ui.download_upscale_from_ui("sugoi", "model", "", "upscale.pth", "", "", "outputs")

    assert "Missing packages for Real-ESRGAN: realesrgan, basicsr" in status
    assert "requirements-upscale.txt" in status


def test_download_inpaint_reports_missing_dependencies(monkeypatch):
    monkeypatch.setattr(ui, "missing_inpaint_dependencies", lambda backend: ["torch"])
    monkeypatch.setattr(ui, "download_inpaint_model", lambda backend: (_ for _ in ()).throw(AssertionError("should not download")))

    _, _, status = ui.download_inpaint_from_ui("sugoi", "model", "", "", "migan", "migan.pt", "outputs")

    assert "Missing packages for migan: torch" in status
    assert "requirements-inpaint.txt" in status


def test_single_file_batch_returns_translated_image(monkeypatch, tmp_path):
    source = tmp_path / "page.png"
    source.write_bytes(b"image")
    translated = tmp_path / "final" / "page_translated.png"
    translated.parent.mkdir()
    translated.write_bytes(b"translated")

    monkeypatch.setattr(ui, "_settings", lambda *args, **kwargs: AppSettings(output_dir=str(tmp_path)))
    region = TextRegion(id=1, box=[[1, 1], [4, 1], [4, 4], [1, 4]], bbox=(1, 1, 4, 4), translated_text="Hi")
    analyzed = ImageJobResult(image_name="page.png", regions=[region])
    composed = ImageJobResult(
        image_name="page.png",
        regions=[region],
        original_path=str(tmp_path / "original.png"),
        overlay_path=str(tmp_path / "overlay.png"),
        cleaned_path=str(tmp_path / "cleaned.png"),
        final_path=str(translated),
    )

    monkeypatch.setattr(ui.PIPELINE, "analyze_image", lambda path, settings, run_translation=True: analyzed)
    monkeypatch.setattr(ui.PIPELINE, "compose_image", lambda path, settings, regions: composed)
    monkeypatch.setattr(ui, "make_zip", lambda paths, output_dir: (_ for _ in ()).throw(AssertionError("should not zip one file")))

    result = ui._batch_auto([str(source)], True, "sugoi", "", "", "", "", "", str(tmp_path), 1, 512, 64, 4, 2, 18, True, 6, 0, True, "PNG", False)

    assert len(result) == 10
    assert result[1] == [region.as_review_row()]
    assert result[2] == composed.original_path
    assert result[3] == composed.overlay_path
    assert result[4] == composed.cleaned_path
    assert result[5] == composed.final_path
    assert result[6] == composed.final_path
    assert result[7] == str(translated)
    assert result[8] == [(str(translated), "page_translated.png")]
    status = result[9]
    assert "OK: page.png" in status


def test_multi_file_batch_returns_zip(monkeypatch, tmp_path):
    sources = [tmp_path / "a.png", tmp_path / "b.png"]
    for source in sources:
        source.write_bytes(b"image")

    final_paths = [str(tmp_path / "final" / "a_translated.png"), str(tmp_path / "final" / "b_translated.png")]
    zip_path = tmp_path / "batches" / "translated_batch.zip"

    def fake_compose(path, settings, regions):
        index = 0 if str(path).endswith("a.png") else 1
        return ImageJobResult(
            image_name=sources[index].name,
            original_path=f"original-{index}.png",
            overlay_path=f"overlay-{index}.png",
            cleaned_path=f"cleaned-{index}.png",
            final_path=final_paths[index],
        )

    monkeypatch.setattr(ui, "_settings", lambda *args, **kwargs: AppSettings(output_dir=str(tmp_path)))
    monkeypatch.setattr(ui.PIPELINE, "analyze_image", lambda path, settings, run_translation=True: ImageJobResult(image_name=Path(path).name))
    monkeypatch.setattr(ui.PIPELINE, "compose_image", fake_compose)
    monkeypatch.setattr(ui, "make_zip", lambda paths, output_dir: str(zip_path))

    result = ui._batch_auto([str(path) for path in sources], True, "sugoi", "", "", "", "", "", str(tmp_path), 1, 512, 64, 4, 2, 18, True, 6, 0, True, "PNG", False)

    assert len(result) == 10
    assert result[5] == final_paths[1]
    assert result[7] == str(zip_path)
    assert result[8] == [(final_paths[0], "a_translated.png"), (final_paths[1], "b_translated.png")]
    status = result[9]
    assert "OK: a.png" in status
    assert "OK: b.png" in status
