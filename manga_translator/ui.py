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
    css = """
    :root {
        --radius-lg: 0 !important;
        --radius-md: 0 !important;
        --radius-sm: 0 !important;
        --radius-xs: 0 !important;
        --button-large-radius: 0 !important;
        --button-primary-radius: 0 !important;
        --button-secondary-radius: 0 !important;
        --input-radius: 0 !important;
        --block-radius: 0 !important;
        --block-border-width: 2px !important;
        --input-border-width: 2px !important;
        --button-border-width: 2px !important;
        --border-color-primary: #3a4658 !important;
        --border-color-accent: #465468 !important;
        --block-border-color: #3a4658 !important;
        --input-border-color: #3a4658 !important;
    }
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
    .gradio-container * {
        border-radius: 0 !important;
    }
    .gradio-container,
    .gradio-container > .contain {
        border: 0 !important;
        outline: 0 !important;
        max-width: none !important;
        width: 100% !important;
    }
    .gradio-container {
        padding-left: 24px !important;
        padding-right: 24px !important;
    }
    .gradio-container button,
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select,
    .gradio-container .wrap,
    .gradio-container .block,
    .gradio-container .block-label,
    .gradio-container .tab-nav,
    .gradio-container .tabitem,
    .gradio-container table,
    .gradio-container th,
    .gradio-container td {
        border-radius: 0 !important;
    }
    .gradio-container .block,
    .gradio-container .form,
    .gradio-container .wrap,
    .gradio-container .panel,
    .gradio-container .tabs,
    .gradio-container .tabitem,
    .gradio-container .accordion {
        border-width: 2px !important;
        border-color: #3a4658 !important;
        outline: 0 !important;
    }
    .gradio-container .block-label {
        border-color: #3a4658 !important;
    }
    .gradio-container .wrap,
    .gradio-container .block,
    .gradio-container .form,
    .gradio-container .tabitem,
    .gradio-container .tabs {
        box-shadow: none !important;
    }
    .gradio-container button {
        box-shadow: none !important;
        text-transform: none;
    }
    .gradio-container .primary {
        border: 2px solid #ff6a00 !important;
    }
    .gradio-container .secondary {
        border: 2px solid var(--border-color-primary) !important;
    }
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select {
        border-width: 2px !important;
        border-color: #3a4658 !important;
        box-shadow: none !important;
    }
    .gradio-container .upload-container,
    .gradio-container .file-preview,
    .gradio-container .file-preview-holder {
        border-color: #3a4658 !important;
        outline: 0 !important;
        box-shadow: none !important;
    }
    @media (max-width: 900px) {
        .settings-sidebar {
            max-width: none;
            position: static;
        }
    }
    """
    with gr.Blocks(title="CPU Manga Translator", css=css) as demo:
        defaults = AppSettings.with_env_defaults()
        default_backend = defaults.translation_backend if defaults.translation_backend in {"llama", "sugoi"} else "llama"
        state = gr.State("")
        gr.Markdown("# CPU Manga Translator")

        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=1, min_width=300, elem_classes=["settings-sidebar"]):
                with gr.Accordion("Models", open=True):
                    auto_translate = gr.Checkbox(label="Auto translate during analysis", value=True)
                    translation_backend = gr.Dropdown(["llama", "sugoi"], value=default_backend, label="translation backend")
                    model_path = gr.Textbox(value=defaults.translation_model_path, label="Translation model path", placeholder="models/translation/gemma-translate-q4.gguf")
                    realesrgan_model_path = gr.Textbox(value=defaults.realesrgan_model_path, label="Real-ESRGAN model path", placeholder="C:/models/RealESRGAN_x4plus_anime_6B.pth")
                    enable_upscale = gr.Checkbox(label="Real-ESRGAN anime upscale", value=defaults.enable_upscale)
                    with gr.Row():
                        download_translation_btn = gr.Button("Download translation model")
                        download_upscale_btn = gr.Button("Download Real-ESRGAN model")

                with gr.Accordion("OCR and setup", open=False):
                    setup_status = gr.Markdown(refresh_setup_status("", "", "", "", ""))
                    refresh_status_btn = gr.Button("Refresh setup status")

                with gr.Accordion("CPU", open=False):
                    threads = gr.Slider(1, 16, value=defaults.llama_threads, step=1, label="llama.cpp CPU threads")
                    context = gr.Slider(512, 8192, value=defaults.llama_context, step=512, label="llama context")
                    max_tokens = gr.Slider(32, 1024, value=defaults.max_translation_tokens, step=32, label="max translation tokens")

                with gr.Accordion("Rendering and output", open=False):
                    font_path = gr.Textbox(value=defaults.font_path, label="Font path", placeholder="C:/Windows/Fonts/arial.ttf")
                    output_dir = gr.Textbox(value=defaults.output_dir, label="Output directory", placeholder="outputs")
                    mask_padding = gr.Slider(0, 40, value=defaults.mask_padding, step=1, label="mask padding")
                    inpaint_radius = gr.Slider(1, 12, value=defaults.inpaint_radius, step=1, label="inpaint radius")
                    font_size = gr.Slider(8, 72, value=defaults.font_size, step=1, label="font size")
                    auto_font_size = gr.Checkbox(label="auto-fit font size", value=defaults.auto_font_size)
                    output_format = gr.Dropdown(["PNG", "JPEG"], value=defaults.output_format, label="output format")

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
