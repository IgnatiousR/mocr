import sys
import types
from pathlib import Path

from manga_translator import model_manager
from manga_translator.model_manager import (
    DEFAULT_TRANSLATION_FILE,
    download_realesrgan_model,
    download_translation_model,
    ensure_model_dirs,
    realesrgan_missing_message,
    resolve_project_path,
    translation_missing_message,
)


def test_default_model_paths_resolve_under_models(monkeypatch):
    monkeypatch.delenv("TRANSLATION_MODEL_PATH", raising=False)
    monkeypatch.delenv("REALESRGAN_MODEL_PATH", raising=False)

    translation = model_manager.default_translation_model_path()
    upscale = model_manager.default_realesrgan_model_path()

    assert "models" in translation.parts
    assert "translation" in translation.parts
    assert "models" in upscale.parts
    assert "upscale" in upscale.parts


def test_missing_model_messages_are_actionable():
    assert "Download translation model first" in translation_missing_message("models/translation/missing.gguf")
    assert "Download Real-ESRGAN model first" in realesrgan_missing_message("models/upscale/missing.pth")


def test_ensure_model_dirs_creates_project_folders():
    ensure_model_dirs()

    assert model_manager.TRANSLATION_DIR.exists()
    assert model_manager.UPSCALE_DIR.exists()


def test_download_translation_model_can_be_mocked(tmp_path, monkeypatch):
    cached = tmp_path / DEFAULT_TRANSLATION_FILE
    cached.write_text("model", encoding="utf-8")
    destination = tmp_path / "models" / "translation" / DEFAULT_TRANSLATION_FILE

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.hf_hub_download = lambda repo_id, filename: str(cached)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setattr(model_manager, "TRANSLATION_DIR", tmp_path / "models" / "translation")
    monkeypatch.setenv("TRANSLATION_MODEL_PATH", str(destination))
    monkeypatch.delenv("TRANSLATION_MODEL_REPO", raising=False)
    monkeypatch.delenv("TRANSLATION_MODEL_FILE", raising=False)

    path = download_translation_model()

    assert path == destination
    assert path.read_text(encoding="utf-8") == "model"


def test_download_realesrgan_model_can_be_mocked(tmp_path, monkeypatch):
    destination = tmp_path / "models" / "upscale" / "RealESRGAN_x4plus_anime_6B.pth"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            if getattr(self, "_read", False):
                return b""
            self._read = True
            return b"weights"

    monkeypatch.setattr(model_manager, "UPSCALE_DIR", tmp_path / "models" / "upscale")
    monkeypatch.setenv("REALESRGAN_MODEL_PATH", str(destination))
    monkeypatch.setattr(model_manager.urllib.request, "urlopen", lambda url: FakeResponse())

    path = download_realesrgan_model()

    assert path == destination
    assert path.read_bytes() == b"weights"


def test_relative_project_path_resolves_from_project_root():
    path = resolve_project_path("models/translation/example.gguf")

    assert path.is_absolute()
    assert path.name == "example.gguf"
