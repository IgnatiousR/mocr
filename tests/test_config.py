from pathlib import Path

from manga_translator.models import AppSettings


def test_env_defaults_load_when_ui_values_empty(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    font = tmp_path / "font.ttf"
    upscale = tmp_path / "upscale.pth"
    model.write_text("model", encoding="utf-8")
    font.write_text("font", encoding="utf-8")
    upscale.write_text("upscale", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                f"TRANSLATION_MODEL_PATH={model}",
                f"FONT_PATH={font}",
                f"REALESRGAN_MODEL_PATH={upscale}",
                "OUTPUT_DIR=my_outputs",
                "LLAMA_THREADS=6",
                "LLAMA_CONTEXT=1024",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = AppSettings.with_env_defaults()

    assert settings.translation_model_path == str(model)
    assert settings.font_path == str(font)
    assert settings.realesrgan_model_path == str(upscale)
    assert settings.output_dir == "my_outputs"
    assert settings.llama_threads == 6
    assert settings.llama_context == 1024


def test_ui_values_override_env_defaults(tmp_path, monkeypatch):
    env_model = tmp_path / "env.gguf"
    ui_model = tmp_path / "ui.gguf"
    env_model.write_text("env", encoding="utf-8")
    ui_model.write_text("ui", encoding="utf-8")
    (tmp_path / ".env").write_text(f"TRANSLATION_MODEL_PATH={env_model}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = AppSettings.with_env_defaults(translation_model_path=str(ui_model))

    assert settings.translation_model_path == str(ui_model)
    assert settings.model_file == ui_model


def test_realesrgan_model_path_validation(tmp_path):
    missing = tmp_path / "missing.pth"
    settings = AppSettings.with_env_defaults(realesrgan_model_path=str(missing))

    assert settings.realesrgan_model_file is None


def test_default_model_paths_are_project_local(monkeypatch):
    monkeypatch.delenv("TRANSLATION_MODEL_PATH", raising=False)
    monkeypatch.delenv("REALESRGAN_MODEL_PATH", raising=False)

    settings = AppSettings.with_env_defaults()

    assert "models" in settings.translation_model_path
    assert "models" in settings.realesrgan_model_path
