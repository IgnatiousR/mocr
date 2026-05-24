from __future__ import annotations

from dataclasses import dataclass
import shutil
import urllib.request
from pathlib import Path

from .config import get_config_value

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
TRANSLATION_DIR = MODELS_DIR / "translation"
UPSCALE_DIR = MODELS_DIR / "upscale"
INPAINT_DIR = MODELS_DIR / "inpaint"

DEFAULT_TRANSLATION_REPO = "webbigdata/gemma-2-2b-jpn-it-translate-gguf"
DEFAULT_TRANSLATION_FILE = "gemma-2-2b-jpn-it-translate-Q4_K_M.gguf"
DEFAULT_TRANSLATION_BACKEND = "llama"
DEFAULT_SUGOI_REPO = "entai2965/sugoi-v4-ja-en-ctranslate2"
DEFAULT_SUGOI_DIR = "sugoi-v4-ja-en-ctranslate2"
DEFAULT_FUGUMT_REPO = "staka/fugumt-ja-en"
DEFAULT_FUGUMT_DIR = "fugumt-ja-en"
DEFAULT_JPARACRAWL_BIG_REPO = "zyddnys/jparacrawl-big-ja-en-ctranslate2"
DEFAULT_JPARACRAWL_BIG_DIR = "jparacrawl-big-ja-en-ctranslate2"
DEFAULT_NLLB_REPO = "facebook/nllb-200-distilled-600M"
DEFAULT_NLLB_DIR = "nllb-200-distilled-600M"
DEFAULT_M2M100_REPO = "facebook/m2m100_418M"
DEFAULT_M2M100_DIR = "m2m100_418M"
DEFAULT_REALESRGAN_FILE = "RealESRGAN_x4plus_anime_6B.pth"
DEFAULT_REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
    "RealESRGAN_x4plus_anime_6B.pth"
)
DEFAULT_INPAINTER_BACKEND = "opencv-telea"
INPAINTER_MODEL_FILES = {
    "migan": "migan_traced.pt",
    "anime-lama": "anime-manga-big-lama.pt",
    "big-lama": "big-lama.pt",
}
INPAINTER_MODEL_URLS = {
    "migan": "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt",
    "anime-lama": "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt",
    "big-lama": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
}
OPENCV_INPAINTERS = {"opencv-telea", "opencv-ns"}
NEURAL_INPAINTERS = set(INPAINTER_MODEL_FILES)


class ModelDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranslationPreset:
    id: str
    label: str
    backend: str
    path: str
    repo: str
    file: str = ""


TRANSLATION_PRESETS: tuple[TranslationPreset, ...] = (
    TranslationPreset(
        id="gemma-gguf",
        label="Gemma 2 2B GGUF (llama.cpp)",
        backend="llama",
        path=f"models/translation/{DEFAULT_TRANSLATION_FILE}",
        repo=DEFAULT_TRANSLATION_REPO,
        file=DEFAULT_TRANSLATION_FILE,
    ),
    TranslationPreset(
        id="gemma-4-e2b-uncensored",
        label="Gemma 4 E2B Uncensored (llama.cpp)",
        backend="llama",
        path="models/translation/gemma-4-E2B-it-uncensored-Q4_K_M.gguf",
        repo="TrevorJS/gemma-4-E2B-it-uncensored-GGUF",
        file="gemma-4-E2B-it-uncensored-Q4_K_M.gguf",
    ),
    TranslationPreset(
        id="sugoi-v4",
        label="Sugoi V4 CTranslate2",
        backend="sugoi",
        path=f"models/translation/{DEFAULT_SUGOI_DIR}",
        repo=DEFAULT_SUGOI_REPO,
    ),
    TranslationPreset(
        id="fugumt",
        label="Fugu-MT Transformers",
        backend="fugumt",
        path=f"models/translation/{DEFAULT_FUGUMT_DIR}",
        repo=DEFAULT_FUGUMT_REPO,
    ),
    TranslationPreset(
        id="jparacrawl-big",
        label="JParaCrawl Big CTranslate2",
        backend="jparacrawl",
        path=f"models/translation/{DEFAULT_JPARACRAWL_BIG_DIR}",
        repo=DEFAULT_JPARACRAWL_BIG_REPO,
    ),
    TranslationPreset(
        id="nllb-600m",
        label="NLLB 600M Transformers",
        backend="nllb",
        path=f"models/translation/{DEFAULT_NLLB_DIR}",
        repo=DEFAULT_NLLB_REPO,
    ),
    TranslationPreset(
        id="m2m100-418m",
        label="M2M100 418M Transformers",
        backend="m2m100",
        path=f"models/translation/{DEFAULT_M2M100_DIR}",
        repo=DEFAULT_M2M100_REPO,
    ),
    TranslationPreset(
        id="sakura-1.5b-gguf",
        label="Sakura 1.5B Qwen2.5 GGUF",
        backend="llama",
        path="models/translation/sakura-1.5b-qwen2.5-v1.0-iq4_xs.gguf",
        repo="SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",
        file="sakura-1.5b-qwen2.5-v1.0-iq4_xs.gguf",
    ),
    TranslationPreset(
        id="hunyuan-mt-7b-gguf",
        label="Hunyuan-MT 7B GGUF",
        backend="llama",
        path="models/translation/hunyuan-mt-7b-q4_k_m.gguf",
        repo="bartowski/Hunyuan-MT-7B-GGUF",
        file="Hunyuan-MT-7B-Q4_K_M.gguf",
    ),
)
TRANSLATION_PRESETS_BY_ID = {preset.id: preset for preset in TRANSLATION_PRESETS}


def get_translation_preset(preset_id: str | None = None) -> TranslationPreset:
    normalized = (preset_id or "").strip().lower()
    if normalized in TRANSLATION_PRESETS_BY_ID:
        return TRANSLATION_PRESETS_BY_ID[normalized]
    backend = _normalize_backend(normalized or get_translation_backend())
    for preset in TRANSLATION_PRESETS:
        if preset.backend == backend:
            return preset
    return TRANSLATION_PRESETS_BY_ID["gemma-gguf"]


def translation_preset_choices() -> list[tuple[str, str]]:
    return [(preset.label, preset.id) for preset in TRANSLATION_PRESETS]


def default_translation_model_path_for_preset(preset_id: str | None = None) -> Path:
    return resolve_project_path(get_translation_preset(preset_id).path)


def ensure_model_dirs() -> None:
    TRANSLATION_DIR.mkdir(parents=True, exist_ok=True)
    UPSCALE_DIR.mkdir(parents=True, exist_ok=True)
    INPAINT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _normalize_backend(backend: str | None = None) -> str:
    value = (backend or get_translation_backend()).strip().lower()
    aliases = {
        "fugu": "fugumt",
        "fugu-mt": "fugumt",
        "fugumt-ja-en": "fugumt",
        "huggingface": "fugumt",
        "transformers": "fugumt",
        "jparacrawl-big": "jparacrawl",
        "nllb-600m": "nllb",
        "m2m100-418m": "m2m100",
    }
    return aliases.get(value, value)


def normalize_inpainter_backend(backend: str | None = None) -> str:
    value = (backend or get_inpainter_backend()).strip().lower()
    aliases = {
        "telea": "opencv-telea",
        "opencv": "opencv-telea",
        "opencv_ns": "opencv-ns",
        "opencv-navier-stokes": "opencv-ns",
        "navier-stokes": "opencv-ns",
        "lama": "big-lama",
        "anime-manga-lama": "anime-lama",
    }
    return aliases.get(value, value)


def default_translation_model_path(backend: str | None = None) -> Path:
    backend = _normalize_backend(backend)
    configured = get_config_value("TRANSLATION_MODEL_PATH")
    if configured:
        configured_path = resolve_project_path(configured)
        if backend == "llama" and configured_path.name in {DEFAULT_SUGOI_DIR, DEFAULT_FUGUMT_DIR}:
            return TRANSLATION_DIR / DEFAULT_TRANSLATION_FILE
        if backend in {"ctranslate2", "sugoi"} and configured_path.name == DEFAULT_TRANSLATION_FILE:
            return TRANSLATION_DIR / DEFAULT_SUGOI_DIR
        if backend == "fugumt" and configured_path.name in {DEFAULT_TRANSLATION_FILE, DEFAULT_SUGOI_DIR}:
            return TRANSLATION_DIR / DEFAULT_FUGUMT_DIR
        return configured_path
    if backend in {"ctranslate2", "sugoi"}:
        return TRANSLATION_DIR / DEFAULT_SUGOI_DIR
    if backend == "fugumt":
        return TRANSLATION_DIR / DEFAULT_FUGUMT_DIR
    if backend == "jparacrawl":
        return TRANSLATION_DIR / DEFAULT_JPARACRAWL_BIG_DIR
    if backend == "nllb":
        return TRANSLATION_DIR / DEFAULT_NLLB_DIR
    if backend == "m2m100":
        return TRANSLATION_DIR / DEFAULT_M2M100_DIR
    return TRANSLATION_DIR / get_translation_model_file()


def default_realesrgan_model_path() -> Path:
    configured = get_config_value("REALESRGAN_MODEL_PATH")
    if configured:
        return resolve_project_path(configured)
    return UPSCALE_DIR / DEFAULT_REALESRGAN_FILE


def default_inpaint_model_path(backend: str | None = None) -> Path:
    backend = normalize_inpainter_backend(backend)
    configured = get_config_value("INPAINT_MODEL_PATH")
    if configured:
        return resolve_project_path(configured)
    filename = INPAINTER_MODEL_FILES.get(backend, "")
    return INPAINT_DIR / filename if filename else Path("")


def get_translation_model_repo(backend: str | None = None) -> str:
    backend = _normalize_backend(backend)
    if backend in {"ctranslate2", "sugoi"}:
        default_repo = DEFAULT_SUGOI_REPO
    elif backend == "fugumt":
        default_repo = DEFAULT_FUGUMT_REPO
    elif backend == "jparacrawl":
        default_repo = DEFAULT_JPARACRAWL_BIG_REPO
    elif backend == "nllb":
        default_repo = DEFAULT_NLLB_REPO
    elif backend == "m2m100":
        default_repo = DEFAULT_M2M100_REPO
    else:
        default_repo = DEFAULT_TRANSLATION_REPO
    configured = get_config_value("TRANSLATION_MODEL_REPO", "")
    if backend in {"ctranslate2", "sugoi"} and configured == DEFAULT_TRANSLATION_REPO:
        return DEFAULT_SUGOI_REPO
    if backend == "fugumt" and configured in {DEFAULT_TRANSLATION_REPO, DEFAULT_SUGOI_REPO}:
        return DEFAULT_FUGUMT_REPO
    if backend == "llama" and configured in {DEFAULT_SUGOI_REPO, DEFAULT_FUGUMT_REPO}:
        return DEFAULT_TRANSLATION_REPO
    return configured or default_repo


def get_translation_model_file() -> str:
    return get_config_value("TRANSLATION_MODEL_FILE", DEFAULT_TRANSLATION_FILE) or DEFAULT_TRANSLATION_FILE


def get_translation_backend() -> str:
    return get_config_value("TRANSLATION_BACKEND", DEFAULT_TRANSLATION_BACKEND).strip().lower()


def get_realesrgan_model_url() -> str:
    return get_config_value("REALESRGAN_MODEL_URL", DEFAULT_REALESRGAN_URL)


def get_inpainter_backend() -> str:
    return get_config_value("INPAINTER_BACKEND", DEFAULT_INPAINTER_BACKEND).strip().lower()


def get_inpaint_model_url(backend: str | None = None) -> str:
    backend = normalize_inpainter_backend(backend)
    configured = get_config_value("INPAINT_MODEL_URL", "")
    return configured or INPAINTER_MODEL_URLS.get(backend, "")


def translation_missing_message(path: str | Path | None = None) -> str:
    model_path = resolve_project_path(path) if path else default_translation_model_path()
    return f"Download translation model first: {model_path}"


def realesrgan_missing_message(path: str | Path | None = None) -> str:
    model_path = resolve_project_path(path) if path else default_realesrgan_model_path()
    return f"Download Real-ESRGAN model first: {model_path}"


def inpaint_missing_message(backend: str | None = None, path: str | Path | None = None) -> str:
    backend = normalize_inpainter_backend(backend)
    model_path = resolve_project_path(path) if path else default_inpaint_model_path(backend)
    return f"Download inpainter model first for {backend}: {model_path}"


def download_translation_model(backend: str | None = None, destination: str | Path | None = None) -> Path:
    ensure_model_dirs()
    backend = _normalize_backend(backend)
    repo_id = get_translation_model_repo(backend)
    destination = resolve_project_path(destination) if destination else default_translation_model_path(backend)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if backend in {"ctranslate2", "sugoi", "fugumt", "jparacrawl", "nllb", "m2m100"}:
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
        model_ref = repo_id if backend in {"ctranslate2", "sugoi", "fugumt", "jparacrawl", "nllb", "m2m100"} else f"{repo_id}/{get_translation_model_file()}"
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


def download_inpaint_model(backend: str | None = None) -> Path:
    ensure_model_dirs()
    backend = normalize_inpainter_backend(backend)
    if backend in OPENCV_INPAINTERS:
        raise ModelDownloadError(f"{backend} does not need a model download.")
    if backend not in NEURAL_INPAINTERS:
        raise ModelDownloadError(f"Unknown inpainter backend: {backend}")

    url = get_inpaint_model_url(backend)
    destination = default_inpaint_model_path(backend)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise ModelDownloadError(f"Could not download inpainter model {backend} from {url}: {exc}") from exc
    return destination
