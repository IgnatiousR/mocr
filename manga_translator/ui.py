from __future__ import annotations

from contextlib import contextmanager
import os
import threading
from pathlib import Path

import gradio as gr
import psutil

from .config import MAX_LLAMA_THREADS, detected_cpu_count
from .model_manager import (
    NEURAL_INPAINTERS,
    ModelDownloadError,
    download_inpaint_model,
    download_realesrgan_model,
    download_translation_model,
    get_translation_preset,
    normalize_inpainter_backend,
    translation_preset_choices,
)
from .model_lifecycle import unload_cached_models, warm_load_models
from .models import AppSettings, ImageJobResult
from .pipeline import MangaTranslationPipeline, make_zip, rows_to_regions
from .profiles import PROFILE_CHOICES
from .setup_check import (
    dependency_status_message,
    missing_inpaint_dependencies,
    missing_translation_dependencies,
    missing_upscale_dependencies,
    setup_status_markdown,
)

PIPELINE = MangaTranslationPipeline()
FONT_EXTENSIONS = {".otc", ".otf", ".ttc", ".ttf"}
STONE_THEME = "theme-repo/STONE_Theme"
ACTIVE_JOBS = 0
ACTIVE_JOBS_LOCK = threading.Lock()
INPAINTER_CHOICES = [
    ("OpenCV Telea", "opencv-telea"),
    ("OpenCV Navier-Stokes", "opencv-ns"),
    ("MI-GAN", "migan"),
    ("Anime/Manga LaMa", "anime-lama"),
    ("Big-LaMa", "big-lama"),
]


def _format_bytes(value: float) -> str:
    gb = value / (1024**3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{value / (1024**2):.0f} MB"


def _app_memory_bytes() -> int:
    process = psutil.Process(os.getpid())
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _job_status() -> str:
    with ACTIVE_JOBS_LOCK:
        return "Running" if ACTIVE_JOBS else "Idle"


def ram_usage_markdown(*_unused_args: object) -> str:
    memory = psutil.virtual_memory()
    app_memory = _app_memory_bytes()
    return "\n".join(
        [
            "### RAM Usage",
            f"- **State**: {_job_status()}",
            f"- **System**: {_format_bytes(memory.used)} / {_format_bytes(memory.total)} ({memory.percent:.0f}%)",
            f"- **Available**: {_format_bytes(memory.available)}",
            f"- **This app**: {_format_bytes(app_memory)}",
        ]
    )


@contextmanager
def active_job():
    global ACTIVE_JOBS
    with ACTIVE_JOBS_LOCK:
        ACTIVE_JOBS += 1
    try:
        yield
    finally:
        with ACTIVE_JOBS_LOCK:
            ACTIVE_JOBS = max(0, ACTIVE_JOBS - 1)


def _font_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    windir = os.environ.get("WINDIR")
    local_app_data = os.environ.get("LOCALAPPDATA")
    home = Path.home()

    if windir:
        dirs.append(Path(windir) / "Fonts")
    if local_app_data:
        dirs.append(Path(local_app_data) / "Microsoft" / "Windows" / "Fonts")

    dirs.extend(
        [
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            home / "Library" / "Fonts",
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            home / ".local" / "share" / "fonts",
            home / ".fonts",
        ]
    )
    return dirs


def installed_font_choices(extra_paths: list[str] | None = None, search_dirs: list[Path] | None = None) -> list[tuple[str, str]]:
    font_paths: dict[str, Path] = {}
    for directory in search_dirs or _font_search_dirs():
        if not directory.exists():
            continue
        try:
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS:
                    resolved = str(path.resolve())
                    font_paths.setdefault(resolved.lower(), path)
        except OSError:
            continue

    for extra_path in extra_paths or []:
        if not extra_path:
            continue
        path = Path(extra_path)
        if path.exists() and path.suffix.lower() in FONT_EXTENSIONS:
            resolved = str(path.resolve())
            font_paths.setdefault(resolved.lower(), path)

    choices = [(f"{path.stem} ({path.name})", str(path)) for path in font_paths.values()]
    return sorted(choices, key=lambda choice: choice[0].lower())


def _settings(
    processing_profile: str,
    detector_backend: str,
    pre_upscale_ratio: int,
    revert_pre_upscale: bool,
    mask_refine: bool,
    mask_refine_dilation: int,
    mask_refine_blur: int,
    pre_dict_path: str,
    post_dict_path: str,
    render_direction: str,
    font_color: str,
    stroke_color: str,
    stroke_width: int,
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    text_box_gap: int,
    line_gap: int,
    overflow_text: bool,
    output_format: str,
    enable_upscale: bool,
) -> AppSettings:
    return AppSettings.with_env_defaults(
        processing_profile=processing_profile or "",
        detector_backend=detector_backend or "",
        translation_backend=translation_backend or "",
        translation_model_path=model_path or "",
        font_path=font_path or "",
        realesrgan_model_path=realesrgan_model_path or "",
        inpainter_backend=inpainter_backend or "",
        inpaint_model_path=inpaint_model_path or "",
        output_dir=output_dir or "",
        llama_threads=int(threads) if threads is not None else None,
        llama_context=int(context) if context is not None else None,
        max_translation_tokens=max_tokens,
        mask_padding=mask_padding,
        inpaint_radius=inpaint_radius,
        font_size=font_size,
        auto_font_size=auto_font_size,
        text_box_gap=text_box_gap,
        line_gap=line_gap,
        overflow_text=overflow_text,
        output_format=output_format,
        enable_upscale=enable_upscale,
        pre_upscale_ratio=pre_upscale_ratio,
        revert_pre_upscale=revert_pre_upscale,
        mask_refine=mask_refine,
        mask_refine_dilation=mask_refine_dilation,
        mask_refine_blur=mask_refine_blur,
        pre_dict_path=pre_dict_path or "",
        post_dict_path=post_dict_path or "",
        render_direction=render_direction or "auto",
        font_color=font_color or "#181818",
        stroke_color=stroke_color or "#ffffff",
        stroke_width=stroke_width,
    )


def _debug_gallery(result: ImageJobResult | None) -> list[tuple[str, str]]:
    if not result:
        return []
    gallery: list[tuple[str, str]] = []
    for label, path in result.debug_paths.items():
        if path and Path(path).exists() and Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            gallery.append((path, label))
    return gallery


def _file_path(file) -> str:
    return file.name if hasattr(file, "name") else str(file)


def analyze(
    files,
    auto_translate: bool,
    processing_profile: str,
    detector_backend: str,
    pre_upscale_ratio: int,
    revert_pre_upscale: bool,
    mask_refine: bool,
    mask_refine_dilation: int,
    mask_refine_blur: int,
    pre_dict_path: str,
    post_dict_path: str,
    render_direction: str,
    font_color: str,
    stroke_color: str,
    stroke_width: int,
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    text_box_gap: int,
    line_gap: int,
    overflow_text: bool,
    output_format: str,
    enable_upscale: bool,
):
    with active_job():
        return _analyze(
            files,
            auto_translate,
            processing_profile,
            detector_backend,
            pre_upscale_ratio,
            revert_pre_upscale,
            mask_refine,
            mask_refine_dilation,
            mask_refine_blur,
            pre_dict_path,
            post_dict_path,
            render_direction,
            font_color,
            stroke_color,
            stroke_width,
            translation_backend,
            model_path,
            font_path,
            realesrgan_model_path,
            inpainter_backend,
            inpaint_model_path,
            output_dir,
            threads,
            context,
            max_tokens,
            mask_padding,
            inpaint_radius,
            font_size,
            auto_font_size,
            text_box_gap,
            line_gap,
            overflow_text,
            output_format,
            enable_upscale,
        )


def _analyze(
    files,
    auto_translate: bool,
    processing_profile: str,
    detector_backend: str,
    pre_upscale_ratio: int,
    revert_pre_upscale: bool,
    mask_refine: bool,
    mask_refine_dilation: int,
    mask_refine_blur: int,
    pre_dict_path: str,
    post_dict_path: str,
    render_direction: str,
    font_color: str,
    stroke_color: str,
    stroke_width: int,
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    text_box_gap: int,
    line_gap: int,
    overflow_text: bool,
    output_format: str,
    enable_upscale: bool,
):
    if not files:
        raise gr.Error("Upload at least one image.")
    settings = _settings(
        processing_profile,
        detector_backend,
        pre_upscale_ratio,
        revert_pre_upscale,
        mask_refine,
        mask_refine_dilation,
        mask_refine_blur,
        pre_dict_path,
        post_dict_path,
        render_direction,
        font_color,
        stroke_color,
        stroke_width,
        translation_backend,
        model_path,
        font_path,
        realesrgan_model_path,
        inpainter_backend,
        inpaint_model_path,
        output_dir,
        threads,
        context,
        max_tokens,
        mask_padding,
        inpaint_radius,
        font_size,
        auto_font_size,
        text_box_gap,
        line_gap,
        overflow_text,
        output_format,
        enable_upscale,
    )
    first = files[0]
    result = PIPELINE.analyze_image(_file_path(first), settings, run_translation=auto_translate)
    status = f"Analyzed {result.image_name}: {len(result.regions)} region(s)."
    rows = [region.as_review_row() for region in result.regions]
    return result.model_dump_json(), rows, result.original_path, result.overlay_path, None, None, _debug_gallery(result), status


def compose(
    files,
    result_state,
    rows,
    processing_profile: str,
    detector_backend: str,
    pre_upscale_ratio: int,
    revert_pre_upscale: bool,
    mask_refine: bool,
    mask_refine_dilation: int,
    mask_refine_blur: int,
    pre_dict_path: str,
    post_dict_path: str,
    render_direction: str,
    font_color: str,
    stroke_color: str,
    stroke_width: int,
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    text_box_gap: int,
    line_gap: int,
    overflow_text: bool,
    output_format: str,
    enable_upscale: bool,
):
    with active_job():
        return _compose(
            files,
            result_state,
            rows,
            processing_profile,
            detector_backend,
            pre_upscale_ratio,
            revert_pre_upscale,
            mask_refine,
            mask_refine_dilation,
            mask_refine_blur,
            pre_dict_path,
            post_dict_path,
            render_direction,
            font_color,
            stroke_color,
            stroke_width,
            translation_backend,
            model_path,
            font_path,
            realesrgan_model_path,
            inpainter_backend,
            inpaint_model_path,
            output_dir,
            threads,
            context,
            max_tokens,
            mask_padding,
            inpaint_radius,
            font_size,
            auto_font_size,
            text_box_gap,
            line_gap,
            overflow_text,
            output_format,
            enable_upscale,
        )


def _compose(
    files,
    result_state,
    rows,
    processing_profile: str,
    detector_backend: str,
    pre_upscale_ratio: int,
    revert_pre_upscale: bool,
    mask_refine: bool,
    mask_refine_dilation: int,
    mask_refine_blur: int,
    pre_dict_path: str,
    post_dict_path: str,
    render_direction: str,
    font_color: str,
    stroke_color: str,
    stroke_width: int,
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    text_box_gap: int,
    line_gap: int,
    overflow_text: bool,
    output_format: str,
    enable_upscale: bool,
):
    if not files or not result_state:
        raise gr.Error("Analyze an image before composing.")
    settings = _settings(
        processing_profile,
        detector_backend,
        pre_upscale_ratio,
        revert_pre_upscale,
        mask_refine,
        mask_refine_dilation,
        mask_refine_blur,
        pre_dict_path,
        post_dict_path,
        render_direction,
        font_color,
        stroke_color,
        stroke_width,
        translation_backend,
        model_path,
        font_path,
        realesrgan_model_path,
        inpainter_backend,
        inpaint_model_path,
        output_dir,
        threads,
        context,
        max_tokens,
        mask_padding,
        inpaint_radius,
        font_size,
        auto_font_size,
        text_box_gap,
        line_gap,
        overflow_text,
        output_format,
        enable_upscale,
    )
    previous = ImageJobResult.model_validate_json(result_state)
    regions = rows_to_regions(rows, previous.regions)
    result = PIPELINE.compose_image(_file_path(files[0]), settings, regions)
    status = f"Composed {result.image_name}."
    return result.model_dump_json(), result.cleaned_path, result.final_path, result.final_path, _debug_gallery(result), status


def batch_auto(
    files,
    auto_translate: bool,
    processing_profile: str,
    detector_backend: str,
    pre_upscale_ratio: int,
    revert_pre_upscale: bool,
    mask_refine: bool,
    mask_refine_dilation: int,
    mask_refine_blur: int,
    pre_dict_path: str,
    post_dict_path: str,
    render_direction: str,
    font_color: str,
    stroke_color: str,
    stroke_width: int,
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    text_box_gap: int,
    line_gap: int,
    overflow_text: bool,
    output_format: str,
    enable_upscale: bool,
):
    with active_job():
        return _batch_auto(
            files,
            auto_translate,
            processing_profile,
            detector_backend,
            pre_upscale_ratio,
            revert_pre_upscale,
            mask_refine,
            mask_refine_dilation,
            mask_refine_blur,
            pre_dict_path,
            post_dict_path,
            render_direction,
            font_color,
            stroke_color,
            stroke_width,
            translation_backend,
            model_path,
            font_path,
            realesrgan_model_path,
            inpainter_backend,
            inpaint_model_path,
            output_dir,
            threads,
            context,
            max_tokens,
            mask_padding,
            inpaint_radius,
            font_size,
            auto_font_size,
            text_box_gap,
            line_gap,
            overflow_text,
            output_format,
            enable_upscale,
        )


def _batch_auto(
    files,
    auto_translate: bool,
    processing_profile: str,
    detector_backend: str,
    pre_upscale_ratio: int,
    revert_pre_upscale: bool,
    mask_refine: bool,
    mask_refine_dilation: int,
    mask_refine_blur: int,
    pre_dict_path: str,
    post_dict_path: str,
    render_direction: str,
    font_color: str,
    stroke_color: str,
    stroke_width: int,
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    text_box_gap: int,
    line_gap: int,
    overflow_text: bool,
    output_format: str,
    enable_upscale: bool,
):
    if not files:
        raise gr.Error("Upload at least one image.")
    settings = _settings(
        processing_profile,
        detector_backend,
        pre_upscale_ratio,
        revert_pre_upscale,
        mask_refine,
        mask_refine_dilation,
        mask_refine_blur,
        pre_dict_path,
        post_dict_path,
        render_direction,
        font_color,
        stroke_color,
        stroke_width,
        translation_backend,
        model_path,
        font_path,
        realesrgan_model_path,
        inpainter_backend,
        inpaint_model_path,
        output_dir,
        threads,
        context,
        max_tokens,
        mask_padding,
        inpaint_radius,
        font_size,
        auto_font_size,
        text_box_gap,
        line_gap,
        overflow_text,
        output_format,
        enable_upscale,
    )
    final_paths: list[str] = []
    messages: list[str] = []
    last_result: ImageJobResult | None = None
    for file in files:
        try:
            path = _file_path(file)
            analyzed = PIPELINE.analyze_image(path, settings, run_translation=auto_translate)
            composed = PIPELINE.compose_image(path, settings, analyzed.regions)
            final_paths.append(composed.final_path)
            last_result = composed
            messages.append(f"OK: {Path(path).name}")
        except Exception as exc:
            messages.append(f"FAILED: {Path(_file_path(file)).name}: {exc}")
    if not last_result:
        return "", [], None, None, None, None, None, None, [], [], "\n".join(messages)

    batch_file = final_paths[0] if len(final_paths) == 1 else make_zip(final_paths, settings.output_dir)
    result_gallery = [(path, Path(path).name) for path in final_paths]
    return (
        last_result.model_dump_json(),
        [region.as_review_row() for region in last_result.regions],
        last_result.original_path,
        last_result.overlay_path,
        last_result.cleaned_path,
        last_result.final_path,
        last_result.final_path,
        batch_file,
        _debug_gallery(last_result),
        result_gallery,
        "\n".join(messages),
    )


def refresh_setup_status(
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
) -> str:
    settings = AppSettings.with_env_defaults(
        translation_backend=translation_backend or "",
        translation_model_path=model_path or "",
        font_path=font_path or "",
        realesrgan_model_path=realesrgan_model_path or "",
        inpainter_backend=inpainter_backend or "",
        inpaint_model_path=inpaint_model_path or "",
        output_dir=output_dir or "",
    )
    return setup_status_markdown(
        translation_backend=settings.translation_backend,
        translation_model_path=settings.translation_model_path,
        font_path=settings.font_path,
        realesrgan_model_path=settings.realesrgan_model_path,
        inpainter_backend=settings.inpainter_backend,
        inpaint_model_path=settings.inpaint_model_path,
        output_dir=settings.output_dir,
    )


def download_translation_from_ui(
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
):
    downloaded_path = model_path
    missing = missing_translation_dependencies(translation_backend)
    if missing:
        status = dependency_status_message(translation_backend, missing, "requirements-translate.txt")
        return downloaded_path, refresh_setup_status(translation_backend, downloaded_path, font_path, realesrgan_model_path, inpainter_backend, inpaint_model_path, output_dir), status
    try:
        path = download_translation_model(translation_backend, model_path or None)
        downloaded_path = str(path)
        status = f"Downloaded translation model: {path}"
    except ModelDownloadError as exc:
        status = f"Download failed: {exc}"
    return downloaded_path, refresh_setup_status(translation_backend, downloaded_path, font_path, realesrgan_model_path, inpainter_backend, inpaint_model_path, output_dir), status


def select_translation_preset(
    preset_id: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
):
    preset = get_translation_preset(preset_id)
    setup_status = refresh_setup_status(
        preset.backend,
        preset.path,
        font_path,
        realesrgan_model_path,
        inpainter_backend,
        inpaint_model_path,
        output_dir,
    )
    status = f"Selected translation model: {preset.label}. Model path reset to {preset.path}."
    return preset.backend, preset.path, setup_status, status


def download_upscale_from_ui(
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
):
    missing = missing_upscale_dependencies()
    if missing:
        status = dependency_status_message("Real-ESRGAN", missing, "requirements-upscale.txt")
        return realesrgan_model_path, refresh_setup_status(translation_backend, model_path, font_path, realesrgan_model_path, inpainter_backend, inpaint_model_path, output_dir), status
    try:
        path = download_realesrgan_model()
        status = f"Downloaded Real-ESRGAN model: {path}"
    except ModelDownloadError as exc:
        status = f"Download failed: {exc}"
    return str(AppSettings.with_env_defaults().realesrgan_model_path), refresh_setup_status(translation_backend, model_path, font_path, realesrgan_model_path, inpainter_backend, inpaint_model_path, output_dir), status


def download_inpaint_from_ui(
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
):
    downloaded_path = inpaint_model_path
    resolved_backend = normalize_inpainter_backend(inpainter_backend)
    if resolved_backend not in NEURAL_INPAINTERS:
        status = f"{resolved_backend} does not need a model download."
        return downloaded_path, refresh_setup_status(translation_backend, model_path, font_path, realesrgan_model_path, resolved_backend, downloaded_path, output_dir), status
    missing = missing_inpaint_dependencies(resolved_backend)
    if missing:
        status = dependency_status_message(resolved_backend, missing, "requirements-inpaint.txt")
        return downloaded_path, refresh_setup_status(translation_backend, model_path, font_path, realesrgan_model_path, resolved_backend, downloaded_path, output_dir), status
    try:
        path = download_inpaint_model(resolved_backend)
        downloaded_path = str(path)
        status = f"Downloaded inpainter model: {path}"
    except ModelDownloadError as exc:
        status = f"Download failed: {exc}"
    return downloaded_path, refresh_setup_status(translation_backend, model_path, font_path, realesrgan_model_path, inpainter_backend, downloaded_path, output_dir), status


def warm_load_from_ui(
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    inpainter_backend: str,
    inpaint_model_path: str,
    output_dir: str,
):
    settings = AppSettings.with_env_defaults(
        translation_backend=translation_backend or "",
        translation_model_path=model_path or "",
        font_path=font_path or "",
        realesrgan_model_path=realesrgan_model_path or "",
        inpainter_backend=inpainter_backend or "",
        inpaint_model_path=inpaint_model_path or "",
        output_dir=output_dir or "",
    )
    return "\n".join(warm_load_models(settings))


def unload_models_from_ui():
    return "\n".join(unload_cached_models())


def _app_css() -> str:
    return """
    .app-shell { gap: 18px; align-items: flex-start; }
    .settings-sidebar {
        min-width: 300px;
        max-width: 360px;
        position: sticky;
        top: 12px;
        align-self: flex-start;
    }
    .main-workspace { min-width: 0; }
    .action-row button { min-width: 160px; }
    .gradio-container,
    .gradio-container > .contain {
        max-width: none !important;
        width: 100% !important;
    }
    .gradio-container {
        padding-left: 24px !important;
        padding-right: 24px !important;
    }
    .settings-sidebar .form,
    .settings-sidebar .block {
        min-width: 0 !important;
    }
    .settings-sidebar .wrap[data-testid="slider"] {
        min-width: 0 !important;
        overflow: visible !important;
    }
    .settings-sidebar .wrap[data-testid="slider"] label {
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        gap: 8px !important;
        min-width: 0 !important;
    }
    .settings-sidebar .wrap[data-testid="slider"] label > span {
        flex: 1 1 130px !important;
        min-width: 0 !important;
        white-space: normal !important;
    }
    .settings-sidebar .wrap[data-testid="slider"] input[type="number"] {
        flex: 0 0 88px !important;
        width: 88px !important;
        min-width: 88px !important;
        text-align: center !important;
    }
    .settings-sidebar .wrap[data-testid="slider"] input[type="range"] {
        flex: 1 0 100% !important;
        width: 100% !important;
        min-width: 0 !important;
    }
    @media (max-width: 900px) {
        .settings-sidebar {
            max-width: none;
            position: static;
        }
    }
    """


def _default_preset(defaults: AppSettings) -> str:
    return get_translation_preset(defaults.translation_backend).id


def _build_sidebar(defaults: AppSettings) -> dict[str, object]:
    font_choices = installed_font_choices(extra_paths=[defaults.font_path])
    thread_slider_max = min(MAX_LLAMA_THREADS, max(detected_cpu_count(), defaults.llama_threads))

    with gr.Column(scale=1, min_width=300, elem_classes=["settings-sidebar"]):
        with gr.Accordion("Models", open=True):
            processing_profile = gr.Dropdown(PROFILE_CHOICES, value=defaults.processing_profile, label="Processing profile")
            detector_backend = gr.Dropdown(
                [("Auto", "auto"), ("CTD quality", "ctd"), ("PaddleOCR", "paddle")],
                value=defaults.detector_backend,
                label="Text detector",
            )
            auto_translate = gr.Checkbox(label="Auto translate during analysis", value=True)
            translation_preset = gr.Dropdown(translation_preset_choices(), value=_default_preset(defaults), label="Translation model")
            translation_backend = gr.Textbox(value=defaults.translation_backend, label="translation backend", visible=False)
            model_path = gr.Textbox(value=defaults.translation_model_path, label="Translation model path", placeholder="models/translation/gemma-translate-q4.gguf")
            realesrgan_model_path = gr.Textbox(value=defaults.realesrgan_model_path, label="Real-ESRGAN model path", placeholder="C:/models/RealESRGAN_x4plus_anime_6B.pth")
            enable_upscale = gr.Checkbox(label="Real-ESRGAN anime upscale", value=defaults.enable_upscale)
            with gr.Row():
                download_translation_btn = gr.Button("Download translation model")
                download_upscale_btn = gr.Button("Download Real-ESRGAN model")
            with gr.Row():
                warm_load_btn = gr.Button("Warm load models")
                unload_models_btn = gr.Button("Unload cached models")

        with gr.Accordion("OCR and setup", open=False):
            setup_status = gr.Markdown(refresh_setup_status("", "", "", "", "", "", ""))
            refresh_status_btn = gr.Button("Refresh setup status")

        with gr.Accordion("CPU", open=False):
            threads = gr.Slider(1, thread_slider_max, value=defaults.llama_threads, step=1, label="llama.cpp CPU threads")
            context = gr.Slider(512, 8192, value=defaults.llama_context, step=512, label="llama context")
            max_tokens = gr.Slider(32, 1024, value=defaults.max_translation_tokens, step=32, label="max translation tokens")

        with gr.Accordion("RAM", open=True):
            ram_timer = gr.Timer(2)
            ram_usage = gr.Markdown(value=ram_usage_markdown())

        with gr.Accordion("Rendering and output", open=False):
            pre_upscale_ratio = gr.Slider(1, 4, value=defaults.pre_upscale_ratio, step=1, label="pre-detection upscale")
            revert_pre_upscale = gr.Checkbox(label="revert pre-upscale before compose", value=defaults.revert_pre_upscale)
            mask_refine = gr.Checkbox(label="refine text mask", value=defaults.mask_refine)
            mask_refine_dilation = gr.Slider(0, 40, value=defaults.mask_refine_dilation, step=1, label="mask refine dilation")
            mask_refine_blur = gr.Slider(0, 15, value=defaults.mask_refine_blur, step=1, label="mask refine blur")
            inpainter_backend = gr.Dropdown(
                choices=INPAINTER_CHOICES,
                value=defaults.inpainter_backend,
                label="Inpainter",
            )
            inpaint_model_path = gr.Textbox(
                value=defaults.inpaint_model_path,
                label="Inpaint model path",
                placeholder="models/inpaint/migan_traced.pt",
            )
            download_inpaint_btn = gr.Button("Download inpainter model")
            font_path = gr.Dropdown(
                choices=font_choices,
                value=defaults.font_path,
                label="Font",
                allow_custom_value=True,
                filterable=True,
            )
            output_dir = gr.Textbox(value=defaults.output_dir, label="Output directory", placeholder="outputs")
            mask_padding = gr.Slider(0, 40, value=defaults.mask_padding, step=1, label="mask padding", info="Expands the detected text area before removing text to catch stray pixels.")
            inpaint_radius = gr.Slider(1, 12, value=defaults.inpaint_radius, step=1, label="inpaint radius", info="Blur radius used by OpenCV inpainters to blend the background.")
            font_size = gr.Slider(8, 72, value=defaults.font_size, step=1, label="font size", info="Base font size for rendering translated text.")
            auto_font_size = gr.Checkbox(label="auto-fit font size", value=defaults.auto_font_size, info="Shrink font automatically to fit inside balloons.")
            text_box_gap = gr.Slider(0, 40, value=defaults.text_box_gap, step=1, label="text box gap", info="Extra bounding box expansion. High values cause text overlap! Keep at 0-6.")
            line_gap = gr.Slider(-4, 24, value=defaults.line_gap, step=1, label="line gap", info="Vertical spacing between lines of text.")
            overflow_text = gr.Checkbox(label="show all translated text", value=defaults.overflow_text, info="Force text to render even if it's too big for the bubble.")
            render_direction = gr.Dropdown([("Auto", "auto"), ("Horizontal", "horizontal"), ("Vertical", "vertical")], value=defaults.render_direction, label="render direction")
            font_color = gr.Textbox(value=defaults.font_color, label="font color")
            stroke_color = gr.Textbox(value=defaults.stroke_color, label="stroke color")
            stroke_width = gr.Slider(0, 8, value=defaults.stroke_width, step=1, label="stroke width")
            pre_dict_path = gr.Textbox(value=defaults.pre_dict_path, label="pre-translation dictionary")
            post_dict_path = gr.Textbox(value=defaults.post_dict_path, label="post-translation dictionary")
            output_format = gr.Dropdown(["PNG", "JPEG"], value=defaults.output_format, label="output format")

    return {
        "auto_translate": auto_translate,
        "processing_profile": processing_profile,
        "detector_backend": detector_backend,
        "translation_preset": translation_preset,
        "translation_backend": translation_backend,
        "model_path": model_path,
        "realesrgan_model_path": realesrgan_model_path,
        "enable_upscale": enable_upscale,
        "download_translation_btn": download_translation_btn,
        "download_upscale_btn": download_upscale_btn,
        "warm_load_btn": warm_load_btn,
        "unload_models_btn": unload_models_btn,
        "download_inpaint_btn": download_inpaint_btn,
        "setup_status": setup_status,
        "refresh_status_btn": refresh_status_btn,
        "threads": threads,
        "context": context,
        "max_tokens": max_tokens,
        "ram_timer": ram_timer,
        "ram_usage": ram_usage,
        "font_path": font_path,
        "inpainter_backend": inpainter_backend,
        "inpaint_model_path": inpaint_model_path,
        "output_dir": output_dir,
        "mask_padding": mask_padding,
        "pre_upscale_ratio": pre_upscale_ratio,
        "revert_pre_upscale": revert_pre_upscale,
        "mask_refine": mask_refine,
        "mask_refine_dilation": mask_refine_dilation,
        "mask_refine_blur": mask_refine_blur,
        "inpaint_radius": inpaint_radius,
        "font_size": font_size,
        "auto_font_size": auto_font_size,
        "text_box_gap": text_box_gap,
        "line_gap": line_gap,
        "overflow_text": overflow_text,
        "render_direction": render_direction,
        "font_color": font_color,
        "stroke_color": stroke_color,
        "stroke_width": stroke_width,
        "pre_dict_path": pre_dict_path,
        "post_dict_path": post_dict_path,
        "output_format": output_format,
    }


def _build_workspace() -> dict[str, object]:
    with gr.Column(scale=4, min_width=520, elem_classes=["main-workspace"]):
        files = gr.File(
            label="Images",
            file_count="multiple",
            file_types=["image"],
        )

        with gr.Row(elem_classes=["action-row"]):
            analyze_btn = gr.Button("Analyze first image", variant="primary")
            compose_btn = gr.Button("Compose reviewed image")
            batch_btn = gr.Button("Run full batch")

        status = gr.Textbox(label="Status", lines=4)
        review = gr.Dataframe(
            headers=["id", "enabled", "locked", "direction", "source_text", "translated_text", "font_size", "font_color", "stroke_color", "confidence", "notes"],
            datatype=["number", "bool", "bool", "str", "str", "str", "str", "str", "str", "str", "str"],
            type="array",
            interactive=True,
            label="Review and edit regions",
        )

        with gr.Tabs():
            with gr.Tab("Original"):
                original = gr.Image(label="Original", type="filepath")
            with gr.Tab("Detected boxes"):
                overlay = gr.Image(label="Detected boxes", type="filepath")
            with gr.Tab("Cleaned"):
                cleaned = gr.Image(label="Cleaned", type="filepath")
            with gr.Tab("Final composite"):
                final = gr.Image(label="Final composite", type="filepath")

        with gr.Row():
            final_file = gr.File(label="Final image")
            batch_zip = gr.File(label="Batch ZIP")
        debug_gallery = gr.Gallery(label="Stage debug artifacts", columns=4, object_fit="contain")
        batch_results = gr.Gallery(label="Batch results", columns=4, object_fit="contain")

    return {
        "files": files,
        "analyze_btn": analyze_btn,
        "compose_btn": compose_btn,
        "batch_btn": batch_btn,
        "status": status,
        "review": review,
        "original": original,
        "overlay": overlay,
        "cleaned": cleaned,
        "final": final,
        "final_file": final_file,
        "batch_zip": batch_zip,
        "debug_gallery": debug_gallery,
        "batch_results": batch_results,
    }


def _settings_inputs(sidebar: dict[str, object]) -> list[object]:
    return [
        sidebar["processing_profile"],
        sidebar["detector_backend"],
        sidebar["pre_upscale_ratio"],
        sidebar["revert_pre_upscale"],
        sidebar["mask_refine"],
        sidebar["mask_refine_dilation"],
        sidebar["mask_refine_blur"],
        sidebar["pre_dict_path"],
        sidebar["post_dict_path"],
        sidebar["render_direction"],
        sidebar["font_color"],
        sidebar["stroke_color"],
        sidebar["stroke_width"],
        sidebar["translation_backend"],
        sidebar["model_path"],
        sidebar["font_path"],
        sidebar["realesrgan_model_path"],
        sidebar["inpainter_backend"],
        sidebar["inpaint_model_path"],
        sidebar["output_dir"],
        sidebar["threads"],
        sidebar["context"],
        sidebar["max_tokens"],
        sidebar["mask_padding"],
        sidebar["inpaint_radius"],
        sidebar["font_size"],
        sidebar["auto_font_size"],
        sidebar["text_box_gap"],
        sidebar["line_gap"],
        sidebar["overflow_text"],
        sidebar["output_format"],
        sidebar["enable_upscale"],
    ]


def _bind_events(state: gr.State, sidebar: dict[str, object], workspace: dict[str, object]) -> None:
    settings_inputs = _settings_inputs(sidebar)
    common_setup_inputs = [
        sidebar["translation_backend"],
        sidebar["model_path"],
        sidebar["font_path"],
        sidebar["realesrgan_model_path"],
        sidebar["inpainter_backend"],
        sidebar["inpaint_model_path"],
        sidebar["output_dir"],
    ]

    workspace["analyze_btn"].click(
        analyze,
        inputs=[workspace["files"], sidebar["auto_translate"], *settings_inputs],
        outputs=[state, workspace["review"], workspace["original"], workspace["overlay"], workspace["cleaned"], workspace["final"], workspace["debug_gallery"], workspace["status"]],
        api_name=False,
        show_api=False,
    )
    workspace["compose_btn"].click(
        compose,
        inputs=[workspace["files"], state, workspace["review"], *settings_inputs],
        outputs=[state, workspace["cleaned"], workspace["final"], workspace["final_file"], workspace["debug_gallery"], workspace["status"]],
        api_name=False,
        show_api=False,
    )
    workspace["batch_btn"].click(
        batch_auto,
        inputs=[workspace["files"], sidebar["auto_translate"], *settings_inputs],
        outputs=[
            state,
            workspace["review"],
            workspace["original"],
            workspace["overlay"],
            workspace["cleaned"],
            workspace["final"],
            workspace["final_file"],
            workspace["batch_zip"],
            workspace["debug_gallery"],
            workspace["batch_results"],
            workspace["status"],
        ],
        api_name=False,
        show_api=False,
    )
    sidebar["refresh_status_btn"].click(
        refresh_setup_status,
        inputs=common_setup_inputs,
        outputs=[sidebar["setup_status"]],
        api_name=False,
        show_api=False,
    )
    sidebar["translation_preset"].change(
        select_translation_preset,
        inputs=[
            sidebar["translation_preset"],
            sidebar["font_path"],
            sidebar["realesrgan_model_path"],
            sidebar["inpainter_backend"],
            sidebar["inpaint_model_path"],
            sidebar["output_dir"],
        ],
        outputs=[sidebar["translation_backend"], sidebar["model_path"], sidebar["setup_status"], workspace["status"]],
        api_name=False,
        show_api=False,
    )
    sidebar["download_translation_btn"].click(
        download_translation_from_ui,
        inputs=common_setup_inputs,
        outputs=[sidebar["model_path"], sidebar["setup_status"], workspace["status"]],
        api_name=False,
        show_api=False,
    )
    sidebar["download_upscale_btn"].click(
        download_upscale_from_ui,
        inputs=common_setup_inputs,
        outputs=[sidebar["realesrgan_model_path"], sidebar["setup_status"], workspace["status"]],
        api_name=False,
        show_api=False,
    )
    sidebar["download_inpaint_btn"].click(
        download_inpaint_from_ui,
        inputs=common_setup_inputs,
        outputs=[sidebar["inpaint_model_path"], sidebar["setup_status"], workspace["status"]],
        api_name=False,
        show_api=False,
    )
    sidebar["warm_load_btn"].click(
        warm_load_from_ui,
        inputs=common_setup_inputs,
        outputs=[workspace["status"]],
        api_name=False,
        show_api=False,
    )
    sidebar["unload_models_btn"].click(
        unload_models_from_ui,
        outputs=[workspace["status"]],
        api_name=False,
        show_api=False,
    )
    sidebar["ram_timer"].tick(
        ram_usage_markdown,
        outputs=[sidebar["ram_usage"]],
        api_name=False,
        show_api=False,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="CPU Manga Translator", theme=STONE_THEME, css=_app_css()) as demo:
        defaults = AppSettings.with_env_defaults()
        state = gr.State("")
        gr.Markdown("# CPU Manga Translator")

        with gr.Row(elem_classes=["app-shell"]):
            sidebar = _build_sidebar(defaults)
            workspace = _build_workspace()

        _bind_events(state, sidebar, workspace)
    return demo


def main() -> None:
    build_app().queue(default_concurrency_limit=1).launch()
