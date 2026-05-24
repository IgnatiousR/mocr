from __future__ import annotations

from .models import AppSettings


def warm_load_models(settings: AppSettings) -> list[str]:
    messages: list[str] = []
    if settings.model_file:
        try:
            from .translate import get_fugumt, get_llama, get_sugoi

            backend = settings.translation_backend
            if backend in {"sugoi", "ctranslate2", "jparacrawl", "jparacrawl-big"}:
                get_sugoi(str(settings.model_file))
            elif backend in {"fugumt", "fugu", "fugu-mt", "fugumt-ja-en", "transformers"}:
                get_fugumt(str(settings.model_file))
            else:
                get_llama(str(settings.model_file), settings.llama_threads, settings.llama_context)
            messages.append("Translation model warm-loaded.")
        except Exception as exc:
            messages.append(f"Translation warm-load skipped: {exc}")
    else:
        messages.append("Translation warm-load skipped: model path is missing.")

    try:
        from .ocr import get_manga_ocr

        get_manga_ocr()
        messages.append("Manga OCR warm-loaded.")
    except Exception as exc:
        messages.append(f"OCR warm-load skipped: {exc}")

    if settings.inpaint_model_file:
        try:
            from .inpaint import _load_torchscript_model

            _load_torchscript_model(str(settings.inpaint_model_file))
            messages.append("Inpainter warm-loaded.")
        except Exception as exc:
            messages.append(f"Inpainter warm-load skipped: {exc}")

    return messages


def unload_cached_models() -> list[str]:
    from .detect import get_paddle_ocr
    from .inpaint import _load_torchscript_model
    from .ocr import get_manga_ocr
    from .translate import get_fugumt, get_llama, get_sugoi, get_transformers_seq2seq

    get_paddle_ocr.cache_clear()
    get_manga_ocr.cache_clear()
    get_llama.cache_clear()
    get_sugoi.cache_clear()
    get_fugumt.cache_clear()
    get_transformers_seq2seq.cache_clear()
    _load_torchscript_model.cache_clear()
    return ["Cleared cached detector, OCR, translation, and inpainter models."]
