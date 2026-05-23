from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from .config import get_config_value

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
TRANSLATION_DIR = MODELS_DIR / "translation"
UPSCALE_DIR = MODELS_DIR / "upscale"

DEFAULT_TRANSLATION_REPO = "webbigdata/gemma-2-2b-jpn-it-translate-gguf"
DEFAULT_TRANSLATION_FILE = "gemma-2-2b-jpn-it-translate-Q4_K_M.gguf"
DEFAULT_TRANSLATION_BACKEND = "llama"
DEFAULT_SUGOI_REPO = "entai2965/sugoi-v4-ja-en-ctranslate2"
DEFAULT_SUGOI_DIR = "sugoi-v4-ja-en-ctranslate2"
DEFAULT_REALESRGAN_FILE = "RealESRGAN_x4plus_anime_6B.pth"
DEFAULT_REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
    "RealESRGAN_x4plus_anime_6B.pth"
)


class ModelDownloadError(RuntimeError):
    pass


def ensure_model_dirs() -> None:
    TRANSLATION_DIR.mkdir(parents=True, exist_ok=True)
    UPSCALE_DIR.mkdir(parents=True, exist_ok=True)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _normalize_backend(backend: str | None = None) -> str:
    return (backend or get_translation_backend()).strip().lower()


def default_translation_model_path(backend: str | None = None) -> Path:
    backend = _normalize_backend(backend)
    configured = get_config_value("TRANSLATION_MODEL_PATH")
    if configured:
        configured_path = resolve_project_path(configured)
        if backend in {"ctranslate2", "sugoi"} and configured_path.name == DEFAULT_TRANSLATION_FILE:
            return TRANSLATION_DIR / DEFAULT_SUGOI_DIR
        return configured_path
    if backend in {"ctranslate2", "sugoi"}:
        return TRANSLATION_DIR / DEFAULT_SUGOI_DIR
    return TRANSLATION_DIR / get_translation_model_file()


def default_realesrgan_model_path() -> Path:
    configured = get_config_value("REALESRGAN_MODEL_PATH")
    if configured:
        return resolve_project_path(configured)
    return UPSCALE_DIR / DEFAULT_REALESRGAN_FILE


def get_translation_model_repo(backend: str | None = None) -> str:
    backend = _normalize_backend(backend)
    default_repo = DEFAULT_SUGOI_REPO if backend in {"ctranslate2", "sugoi"} else DEFAULT_TRANSLATION_REPO
    configured = get_config_value("TRANSLATION_MODEL_REPO", "")
    if backend in {"ctranslate2", "sugoi"} and configured == DEFAULT_TRANSLATION_REPO:
        return DEFAULT_SUGOI_REPO
    return configured or default_repo


def get_translation_model_file() -> str:
    return get_config_value("TRANSLATION_MODEL_FILE", DEFAULT_TRANSLATION_FILE)


def get_translation_backend() -> str:
    return get_config_value("TRANSLATION_BACKEND", DEFAULT_TRANSLATION_BACKEND).strip().lower()


def get_realesrgan_model_url() -> str:
    return get_config_value("REALESRGAN_MODEL_URL", DEFAULT_REALESRGAN_URL)


def translation_missing_message(path: str | Path | None = None) -> str:
    model_path = resolve_project_path(path) if path else default_translation_model_path()
    return f"Download translation model first: {model_path}"


def realesrgan_missing_message(path: str | Path | None = None) -> str:
    model_path = resolve_project_path(path) if path else default_realesrgan_model_path()
    return f"Download Real-ESRGAN model first: {model_path}"


def download_translation_model(backend: str | None = None) -> Path:
    ensure_model_dirs()
    backend = _normalize_backend(backend)
    repo_id = get_translation_model_repo(backend)
    destination = default_translation_model_path(backend)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if backend in {"ctranslate2", "sugoi"}:
            try:
                from huggingface_hub import snapshot_download
            except Exception as exc:  # pragma: no cover - depends on optional install
                raise ModelDownloadError("Install huggingface-hub first: python -m pip install -r requirements-base.txt") from exc
            snapshot_download(repo_id=repo_id, local_dir=destination)
        else:
            try:
                from huggingface_hub import hf_hub_download
            except Exception as exc:  # pragma: no cover - depends on optional install
                raise ModelDownloadError("Install huggingface-hub first: python -m pip install -r requirements-base.txt") from exc
            filename = get_translation_model_file()
            cached_path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
            if cached_path.resolve() != destination.resolve():
                shutil.copy2(cached_path, destination)
    except Exception as exc:
        model_ref = repo_id if backend in {"ctranslate2", "sugoi"} else f"{repo_id}/{get_translation_model_file()}"
        raise ModelDownloadError(f"Could not download translation model {model_ref}: {exc}") from exc
    return destination


def download_realesrgan_model() -> Path:
    ensure_model_dirs()
    url = get_realesrgan_model_url()
    destination = default_realesrgan_model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise ModelDownloadError(f"Could not download Real-ESRGAN model from {url}: {exc}") from exc
    return destination
