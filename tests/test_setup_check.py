from manga_translator.setup_check import collect_setup_checks, dependency_status_message, setup_status_markdown


def test_setup_check_reports_missing_optional_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    checks = collect_setup_checks(
        translation_model_path=str(tmp_path / "missing.gguf"),
        font_path=str(tmp_path / "missing.ttf"),
        realesrgan_model_path=str(tmp_path / "missing.pth"),
        output_dir="outputs",
    )

    by_name = {check.name: check for check in checks}
    assert by_name["Translation model"].status == "Optional"
    assert by_name["Font"].status == "Optional"
    assert by_name["Real-ESRGAN model"].status == "Optional"


def test_setup_status_markdown_is_human_readable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    markdown = setup_status_markdown(output_dir="outputs")

    assert "Setup Status" in markdown
    assert "Output directory" in markdown


def test_setup_status_mentions_download_models_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    markdown = setup_status_markdown(
        translation_model_path="models/translation/missing.gguf",
        realesrgan_model_path="models/upscale/missing.pth",
    )

    assert "python scripts/download_models.py --translation" in markdown
    assert "python scripts/download_models.py --upscale" in markdown


def test_setup_status_mentions_neural_inpaint_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    markdown = setup_status_markdown(
        inpainter_backend="migan",
        inpaint_model_path="models/inpaint/missing.pt",
    )

    assert "Inpaint model" in markdown
    assert "python scripts/download_models.py --inpaint" in markdown


def test_dependency_status_message_uses_running_python():
    message = dependency_status_message("sugoi", ["ctranslate2", "sentencepiece"], "requirements-translate.txt")

    assert "Missing packages for sugoi: ctranslate2, sentencepiece" in message
    assert "-m pip install -r requirements-translate.txt" in message
