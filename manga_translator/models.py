from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import default_llama_threads, get_config_value, get_int_config_value
from .model_manager import (
    NEURAL_INPAINTERS,
    default_inpaint_model_path,
    default_realesrgan_model_path,
    default_translation_model_path,
    normalize_inpainter_backend,
    resolve_project_path,
)


class AppSettings(BaseModel):
    processing_profile: str = "quality"
    detector_backend: str = "paddleocr vl 1.5"
    pre_upscale_ratio: int = Field(default=1, ge=1, le=4)
    revert_pre_upscale: bool = True
    mask_refine: bool = True
    mask_refine_dilation: int = Field(default=6, ge=0, le=80)
    mask_refine_blur: int = Field(default=3, ge=0, le=31)
    pre_dict_path: str = ""
    post_dict_path: str = ""
    render_direction: str = "auto"
    font_color: str = "#181818"
    stroke_color: str = "#ffffff"
    stroke_width: int = Field(default=1, ge=0, le=8)
    translation_backend: str = "llama"
    translation_model_path: str = ""
    font_path: str = ""
    realesrgan_model_path: str = ""
    inpainter_backend: str = "opencv-telea"
    inpaint_model_path: str = ""
    output_dir: str = "outputs"
    llama_threads: int = Field(default=4, ge=1, le=32)
    llama_context: int = Field(default=2048, ge=512, le=8192)
    max_translation_tokens: int = Field(default=256, ge=32, le=2048)
    mask_padding: int = Field(default=8, ge=0, le=80)
    inpaint_radius: int = Field(default=3, ge=1, le=25)
    font_size: int = Field(default=28, ge=8, le=96)
    auto_font_size: bool = True
    text_box_gap: int = Field(default=6, ge=0, le=40)
    line_gap: int = Field(default=0, ge=-4, le=24)
    overflow_text: bool = True
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

    @property
    def inpaint_model_file(self) -> Path | None:
        if self.inpainter_backend not in NEURAL_INPAINTERS or not self.inpaint_model_path:
            return None
        path = resolve_project_path(self.inpaint_model_path)
        return path if path.exists() else None

    @classmethod
    def with_env_defaults(
        cls,
        translation_model_path: str = "",
        processing_profile: str = "",
        detector_backend: str = "",
        translation_backend: str = "",
        font_path: str = "",
        realesrgan_model_path: str = "",
        inpainter_backend: str = "",
        inpaint_model_path: str = "",
        output_dir: str = "",
        llama_threads: int | None = None,
        llama_context: int | None = None,
        **kwargs: Any,
    ) -> "AppSettings":
        resolved_profile = (processing_profile or get_config_value("PROCESSING_PROFILE", "quality")).strip().lower()
        resolved_backend = (translation_backend or get_config_value("TRANSLATION_BACKEND", "llama")).strip().lower()
        resolved_inpainter = normalize_inpainter_backend(inpainter_backend or get_config_value("INPAINTER_BACKEND", "opencv-telea"))
        if resolved_profile == "quality" and resolved_inpainter == "opencv-telea":
            resolved_inpainter = "anime-lama"
        default_inpaint_path = str(default_inpaint_model_path(resolved_inpainter)) if resolved_inpainter in NEURAL_INPAINTERS else ""
        pre_upscale_ratio_value = kwargs.pop("pre_upscale_ratio", None)
        pre_dict_path_value = kwargs.pop("pre_dict_path", "")
        post_dict_path_value = kwargs.pop("post_dict_path", "")
        return cls(
            processing_profile=resolved_profile,
            detector_backend=detector_backend or get_config_value("DETECTOR_BACKEND", "paddleocr vl 1.5"),
            translation_backend=resolved_backend,
            translation_model_path=translation_model_path or str(default_translation_model_path(resolved_backend)),
            font_path=font_path or get_config_value("FONT_PATH"),
            realesrgan_model_path=realesrgan_model_path or get_config_value("REALESRGAN_MODEL_PATH") or str(default_realesrgan_model_path()),
            inpainter_backend=resolved_inpainter,
            inpaint_model_path=inpaint_model_path or get_config_value("INPAINT_MODEL_PATH") or default_inpaint_path,
            output_dir=output_dir or get_config_value("OUTPUT_DIR", "outputs"),
            llama_threads=llama_threads if llama_threads is not None else default_llama_threads(),
            llama_context=llama_context if llama_context is not None else get_int_config_value("LLAMA_CONTEXT", 2048),
            pre_upscale_ratio=pre_upscale_ratio_value if pre_upscale_ratio_value is not None else get_int_config_value("PRE_UPSCALE_RATIO", 1),
            pre_dict_path=pre_dict_path_value or get_config_value("PRE_DICT_PATH"),
            post_dict_path=post_dict_path_value or get_config_value("POST_DICT_PATH"),
            **kwargs,
        )


class TextRegion(BaseModel):
    id: int
    box: list[list[float]]
    bbox: tuple[int, int, int, int]
    polygon: list[list[float]] = Field(default_factory=list)
    merged_from: list[int] = Field(default_factory=list)
    raw_text: str = ""
    vertical: bool = False
    direction: str = "auto"
    font_size: int | None = None
    font_color: str = ""
    stroke_color: str = ""
    stroke_width: int | None = None
    confidence: float | None = None
    source_text: str = ""
    translated_text: str = ""
    enabled: bool = True
    locked: bool = False
    notes: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.polygon:
            self.polygon = [list(point) for point in self.box]
        if not self.merged_from:
            self.merged_from = [self.id]
        if not self.raw_text and self.source_text:
            self.raw_text = self.source_text

    def as_review_row(self) -> list[Any]:
        return [
            self.id,
            self.enabled,
            self.locked,
            self.direction,
            self.source_text,
            self.translated_text,
            self.font_size if self.font_size is not None else "",
            self.font_color,
            self.stroke_color,
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
    debug_paths: dict[str, str] = Field(default_factory=dict)
