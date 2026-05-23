from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import get_config_value, get_int_config_value
from .model_manager import default_realesrgan_model_path, default_translation_model_path, resolve_project_path


class AppSettings(BaseModel):
    translation_backend: str = "llama"
    translation_model_path: str = ""
    font_path: str = ""
    realesrgan_model_path: str = ""
    output_dir: str = "outputs"
    llama_threads: int = Field(default=4, ge=1, le=32)
    llama_context: int = Field(default=2048, ge=512, le=8192)
    max_translation_tokens: int = Field(default=256, ge=32, le=2048)
    mask_padding: int = Field(default=8, ge=0, le=80)
    inpaint_radius: int = Field(default=3, ge=1, le=25)
    font_size: int = Field(default=28, ge=8, le=96)
    auto_font_size: bool = True
    output_format: str = "PNG"
    enable_upscale: bool = False

    @property
    def model_file(self) -> Path | None:
        if not self.translation_model_path:
            return None
        path = resolve_project_path(self.translation_model_path)
        return path if path.exists() else None

    @property
    def realesrgan_model_file(self) -> Path | None:
        if not self.realesrgan_model_path:
            return None
        path = resolve_project_path(self.realesrgan_model_path)
        return path if path.exists() else None

    @classmethod
    def with_env_defaults(
        cls,
        translation_model_path: str = "",
        translation_backend: str = "",
        font_path: str = "",
        realesrgan_model_path: str = "",
        output_dir: str = "",
        llama_threads: int | None = None,
        llama_context: int | None = None,
        **kwargs: Any,
    ) -> "AppSettings":
        resolved_backend = (translation_backend or get_config_value("TRANSLATION_BACKEND", "llama")).strip().lower()
        return cls(
            translation_backend=resolved_backend,
            translation_model_path=translation_model_path or str(default_translation_model_path(resolved_backend)),
            font_path=font_path or get_config_value("FONT_PATH"),
            realesrgan_model_path=realesrgan_model_path or get_config_value("REALESRGAN_MODEL_PATH") or str(default_realesrgan_model_path()),
            output_dir=output_dir or get_config_value("OUTPUT_DIR", "outputs"),
            llama_threads=llama_threads if llama_threads is not None else get_int_config_value("LLAMA_THREADS", 4),
            llama_context=llama_context if llama_context is not None else get_int_config_value("LLAMA_CONTEXT", 2048),
            **kwargs,
        )


class TextRegion(BaseModel):
    id: int
    box: list[list[float]]
    bbox: tuple[int, int, int, int]
    vertical: bool = False
    confidence: float | None = None
    source_text: str = ""
    translated_text: str = ""
    enabled: bool = True
    notes: str = ""

    def as_review_row(self) -> list[Any]:
        return [
            self.id,
            self.enabled,
            self.source_text,
            self.translated_text,
            self.confidence if self.confidence is not None else "",
            self.notes,
        ]


class ImageJobResult(BaseModel):
    image_name: str
    regions: list[TextRegion] = Field(default_factory=list)
    status: str = "pending"
    error: str = ""
    original_path: str = ""
    overlay_path: str = ""
    cleaned_path: str = ""
    final_path: str = ""
