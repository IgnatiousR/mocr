from __future__ import annotations

from pathlib import Path

import gradio as gr

from .model_manager import ModelDownloadError, download_realesrgan_model, download_translation_model
from .models import AppSettings, ImageJobResult
from .pipeline import MangaTranslationPipeline, make_zip, rows_to_regions
from .setup_check import setup_status_markdown

PIPELINE = MangaTranslationPipeline()


def _settings(
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    output_format: str,
    enable_upscale: bool,
) -> AppSettings:
    return AppSettings.with_env_defaults(
        translation_backend=translation_backend or "",
        translation_model_path=model_path or "",
        font_path=font_path or "",
        realesrgan_model_path=realesrgan_model_path or "",
        output_dir=output_dir or "",
        llama_threads=int(threads) if threads is not None else None,
        llama_context=int(context) if context is not None else None,
        max_translation_tokens=max_tokens,
        mask_padding=mask_padding,
        inpaint_radius=inpaint_radius,
        font_size=font_size,
        auto_font_size=auto_font_size,
        output_format=output_format,
        enable_upscale=enable_upscale,
    )


def _file_path(file) -> str:
    return file.name if hasattr(file, "name") else str(file)


def analyze(
    files,
    auto_translate: bool,
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    output_format: str,
    enable_upscale: bool,
):
    if not files:
        raise gr.Error("Upload at least one image.")
    settings = _settings(
        translation_backend,
        model_path,
        font_path,
        realesrgan_model_path,
        output_dir,
        threads,
        context,
        max_tokens,
        mask_padding,
        inpaint_radius,
        font_size,
        auto_font_size,
        output_format,
        enable_upscale,
    )
    first = files[0]
    result = PIPELINE.analyze_image(_file_path(first), settings, run_translation=auto_translate)
    status = f"Analyzed {result.image_name}: {len(result.regions)} region(s)."
    rows = [region.as_review_row() for region in result.regions]
    return result.model_dump_json(), rows, result.original_path, result.overlay_path, None, None, status


def compose(
    files,
    result_state,
    rows,
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    output_format: str,
    enable_upscale: bool,
):
    if not files or not result_state:
        raise gr.Error("Analyze an image before composing.")
    settings = _settings(
        translation_backend,
        model_path,
        font_path,
        realesrgan_model_path,
        output_dir,
        threads,
        context,
        max_tokens,
        mask_padding,
        inpaint_radius,
        font_size,
        auto_font_size,
        output_format,
        enable_upscale,
    )
    previous = ImageJobResult.model_validate_json(result_state)
    regions = rows_to_regions(rows, previous.regions)
    result = PIPELINE.compose_image(_file_path(files[0]), settings, regions)
    status = f"Composed {result.image_name}."
    return result.model_dump_json(), result.cleaned_path, result.final_path, result.final_path, status


def batch_auto(
    files,
    auto_translate: bool,
    translation_backend: str,
    model_path: str,
    font_path: str,
    realesrgan_model_path: str,
    output_dir: str,
    threads: int,
    context: int,
    max_tokens: int,
    mask_padding: int,
    inpaint_radius: int,
    font_size: int,
    auto_font_size: bool,
    output_format: str,
    enable_upscale: bool,
):
    if not files:
        raise gr.Error("Upload at least one image.")
    settings = _settings(
        translation_backend,
        model_path,
        font_path,
        realesrgan_model_path,
        output_dir,
        threads,
        context,
        max_tokens,
        mask_padding,
        inpaint_radius,
        font_size,
        auto_font_size,
        output_format,
        enable_upscale,
    )
    final_paths: list[str] = []
    messages: list[str] = []
    for file in files:
        try:
            path = _file_path(file)
            analyzed = PIPELINE.analyze_image(path, settings, run_translation=auto_translate)
            composed = PIPELINE.compose_image(path, settings, analyzed.regions)
            final_paths.append(composed.final_path)
            messages.append(f"OK: {Path(path).name}")
        except Exception as exc:
            messages.append(f"FAILED: {Path(_file_path(file)).name}: {exc}")
    zip_path = make_zip(final_paths, settings.output_dir)
    return zip_path, "\n".join(messages)


def refresh_setup_status(translation_backend: str, model_path: str, font_path: str, realesrgan_model_path: str, output_dir: str) -> str:
    settings = AppSettings.with_env_defaults(
        translation_backend=translation_backend or "",
        translation_model_path=model_path or "",
        font_path=font_path or "",
        realesrgan_model_path=realesrgan_model_path or "",
        output_dir=output_dir or "",
    )
    return setup_status_markdown(
        translation_backend=settings.translation_backend,
        translation_model_path=settings.translation_model_path,
        font_path=settings.font_path,
        realesrgan_model_path=settings.realesrgan_model_path,
        output_dir=settings.output_dir,
    )


def download_translation_from_ui(translation_backend: str, model_path: str, font_path: str, realesrgan_model_path: str, output_dir: str):
    downloaded_path = model_path
    try:
        path = download_translation_model(translation_backend)
        downloaded_path = str(path)
        status = f"Downloaded translation model: {path}"
    except ModelDownloadError as exc:
        status = f"Download failed: {exc}"
    return downloaded_path, refresh_setup_status(translation_backend, downloaded_path, font_path, realesrgan_model_path, output_dir), status


def download_upscale_from_ui(translation_backend: str, model_path: str, font_path: str, realesrgan_model_path: str, output_dir: str):
    try:
        path = download_realesrgan_model()
        status = f"Downloaded Real-ESRGAN model: {path}"
    except ModelDownloadError as exc:
        status = f"Download failed: {exc}"
    return str(AppSettings.with_env_defaults().realesrgan_model_path), refresh_setup_status(translation_backend, model_path, font_path, realesrgan_model_path, output_dir), status


def build_app() -> gr.Blocks:
    with gr.Blocks(title="CPU Manga Translator") as demo:
        state = gr.State("")
        gr.Markdown("# CPU Manga Translator")
        with gr.Row():
            files = gr.File(
                label="Images",
                file_count="multiple",
                file_types=["image"],
            )
            with gr.Column():
                auto_translate = gr.Checkbox(label="Auto translate during analysis", value=True)
                translation_backend = gr.Dropdown(["llama", "sugoi"], value="llama", label="translation backend")
                model_path = gr.Textbox(label="Translation model path", placeholder="models/translation/gemma-translate-q4.gguf")
                font_path = gr.Textbox(label="Font path", placeholder="C:/Windows/Fonts/arial.ttf")
                realesrgan_model_path = gr.Textbox(label="Real-ESRGAN model path", placeholder="C:/models/RealESRGAN_x4plus_anime_6B.pth")
                output_dir = gr.Textbox(label="Output directory", placeholder="outputs")

        with gr.Accordion("Models", open=False):
            with gr.Row():
                download_translation_btn = gr.Button("Download translation model")
                download_upscale_btn = gr.Button("Download Real-ESRGAN model")

        with gr.Accordion("Setup Status", open=False):
            setup_status = gr.Markdown(refresh_setup_status("", "", "", "", ""))
            refresh_status_btn = gr.Button("Refresh setup status")

        with gr.Accordion("CPU and rendering settings", open=False):
            with gr.Row():
                threads = gr.Slider(1, 16, value=4, step=1, label="llama.cpp CPU threads")
                context = gr.Slider(512, 8192, value=2048, step=512, label="llama context")
                max_tokens = gr.Slider(32, 1024, value=256, step=32, label="max translation tokens")
            with gr.Row():
                mask_padding = gr.Slider(0, 40, value=8, step=1, label="mask padding")
                inpaint_radius = gr.Slider(1, 12, value=3, step=1, label="inpaint radius")
                font_size = gr.Slider(8, 72, value=28, step=1, label="font size")
            with gr.Row():
                auto_font_size = gr.Checkbox(label="auto-fit font size", value=True)
                output_format = gr.Dropdown(["PNG", "JPEG"], value="PNG", label="output format")
                enable_upscale = gr.Checkbox(label="Real-ESRGAN anime upscale", value=False)

        with gr.Row():
            analyze_btn = gr.Button("Analyze first image", variant="primary")
            compose_btn = gr.Button("Compose reviewed image")
            batch_btn = gr.Button("Run full batch")

        status = gr.Textbox(label="Status", lines=4)
        review = gr.Dataframe(
            headers=["id", "enabled", "source_text", "translated_text", "confidence", "notes"],
            datatype=["number", "bool", "str", "str", "str", "str"],
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

        settings_inputs = [
            model_path,
            font_path,
            realesrgan_model_path,
            output_dir,
            threads,
            context,
            max_tokens,
            mask_padding,
            inpaint_radius,
            font_size,
            auto_font_size,
            output_format,
            enable_upscale,
        ]
        analyze_btn.click(
            analyze,
            inputs=[files, auto_translate, translation_backend, *settings_inputs],
            outputs=[state, review, original, overlay, cleaned, final, status],
            show_api=False,
        )
        compose_btn.click(
            compose,
            inputs=[files, state, review, translation_backend, *settings_inputs],
            outputs=[state, cleaned, final, final_file, status],
            show_api=False,
        )
        batch_btn.click(
            batch_auto,
            inputs=[files, auto_translate, translation_backend, *settings_inputs],
            outputs=[batch_zip, status],
            show_api=False,
        )
        refresh_status_btn.click(
            refresh_setup_status,
            inputs=[translation_backend, model_path, font_path, realesrgan_model_path, output_dir],
            outputs=[setup_status],
            show_api=False,
        )
        download_translation_btn.click(
            download_translation_from_ui,
            inputs=[translation_backend, model_path, font_path, realesrgan_model_path, output_dir],
            outputs=[model_path, setup_status, status],
            show_api=False,
        )
        download_upscale_btn.click(
            download_upscale_from_ui,
            inputs=[translation_backend, model_path, font_path, realesrgan_model_path, output_dir],
            outputs=[realesrgan_model_path, setup_status, status],
            show_api=False,
        )
    return demo


def main() -> None:
    build_app().queue(default_concurrency_limit=1).launch()
