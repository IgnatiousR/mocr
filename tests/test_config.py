from pathlib import Path

from manga_translator.config import default_llama_threads
from manga_translator.models import AppSettings


def test_env_defaults_load_when_ui_values_empty(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    font = tmp_path / "font.ttf"
    upscale = tmp_path / "upscale.pth"
    inpaint = tmp_path / "migan_traced.pt"
    model.write_text("model", encoding="utf-8")
    font.write_text("font", encoding="utf-8")
    upscale.write_text("upscale", encoding="utf-8")
    inpaint.write_text("inpaint", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                f"TRANSLATION_MODEL_PATH={model}",
                f"FONT_PATH={font}",
                f"REALESRGAN_MODEL_PATH={upscale}",
                "INPAINTER_BACKEND=migan",
                f"INPAINT_MODEL_PATH={inpaint}",
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
    assert settings.inpainter_backend == "migan"
    assert settings.inpaint_model_path == str(inpaint)
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
    monkeypatch.delenv("INPAINTER_BACKEND", raising=False)
    monkeypatch.delenv("INPAINT_MODEL_PATH", raising=False)

    settings = AppSettings.with_env_defaults()

    assert "models" in settings.translation_model_path
    assert "models" in settings.realesrgan_model_path
    assert settings.processing_profile == "quality"
    assert settings.inpainter_backend == "anime-lama"
    assert "inpaint" in settings.inpaint_model_path


def test_neural_inpainter_default_model_path_is_project_local(monkeypatch):
    monkeypatch.delenv("INPAINT_MODEL_PATH", raising=False)

    settings = AppSettings.with_env_defaults(inpainter_backend="anime-lama")

    assert settings.inpainter_backend == "anime-lama"
    assert "models" in settings.inpaint_model_path
    assert "inpaint" in settings.inpaint_model_path


def test_llama_threads_default_to_detected_cpu_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLAMA_THREADS", raising=False)
    monkeypatch.setattr("manga_translator.config.os.cpu_count", lambda: 8)

    settings = AppSettings.with_env_defaults()

    assert settings.llama_threads == 8


def test_llama_threads_env_overrides_detected_cpu_count(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLAMA_THREADS=6", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLAMA_THREADS", raising=False)
    monkeypatch.setattr("manga_translator.config.os.cpu_count", lambda: 8)

    settings = AppSettings.with_env_defaults()

    assert settings.llama_threads == 6


def test_invalid_llama_threads_falls_back_to_detected_cpu_count(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLAMA_THREADS=not-a-number", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLAMA_THREADS", raising=False)
    monkeypatch.setattr("manga_translator.config.os.cpu_count", lambda: 8)

    settings = AppSettings.with_env_defaults()

    assert settings.llama_threads == 8


def test_llama_threads_detection_falls_back_when_cpu_count_unavailable(monkeypatch):
    monkeypatch.delenv("LLAMA_THREADS", raising=False)
    monkeypatch.setattr("manga_translator.config.os.cpu_count", lambda: None)

    assert default_llama_threads(env_path=Path("missing.env")) == 4
