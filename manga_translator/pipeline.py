from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .detect import crop_region, detect_text_regions, draw_region_overlay
from .image_io import image_hash, open_image, save_output_image, short_hash
from .inpaint import inpaint_text
from .model_manager import realesrgan_missing_message, translation_missing_message
from .models import AppSettings, ImageJobResult, TextRegion
from .ocr import recognize_japanese
from .render import render_translations
from .translate import translate_japanese_to_english
from .upscale import upscale_anime


def _suffix(settings: AppSettings) -> str:
    return ".jpg" if settings.output_format.upper() == "JPEG" else ".png"


def _output_stem(source: Path) -> str:
    return f"{source.stem}_{short_hash(source)}"


class MangaTranslationPipeline:
    def __init__(self) -> None:
        self.region_cache: dict[str, list[TextRegion]] = {}

    def analyze_image(self, path: str | Path, settings: AppSettings, run_translation: bool = True) -> ImageJobResult:
        source = Path(path)
        image = open_image(source)
        cache_key = f"{image_hash(source)}:{settings.mask_padding}"
        regions = [region.model_copy(deep=True) for region in self.region_cache.get(cache_key, [])]
        if not regions:
            regions = detect_text_regions(image)
            for region in regions:
                crop = crop_region(image, region)
                try:
                    region.source_text = recognize_japanese(crop)
                except Exception as exc:
                    region.notes = str(exc)
            self.region_cache[cache_key] = [region.model_copy(deep=True) for region in regions]

        if run_translation and not settings.model_file:
            for region in regions:
                if region.source_text:
                    region.notes = translation_missing_message(settings.translation_model_path)
        elif run_translation and settings.model_file:
            for region in regions:
                if region.source_text and not region.translated_text:
                    try:
                        region.translated_text = translate_japanese_to_english(
                            region.source_text,
                            str(settings.model_file),
                            backend=settings.translation_backend,
                            threads=settings.llama_threads,
                            context=settings.llama_context,
                            max_tokens=settings.max_translation_tokens,
                        )
                    except Exception as exc:
                        region.notes = str(exc)
        overlay = draw_region_overlay(image, regions)
        suffix = _suffix(settings)
        stem = _output_stem(source)
        overlay_path = save_output_image(overlay, settings.output_dir, "overlays", f"{stem}_overlay", suffix=suffix)
        original_path = save_output_image(image, settings.output_dir, "originals", f"{stem}_original", suffix=suffix)
        return ImageJobResult(
            image_name=source.name,
            regions=regions,
            status="analyzed",
            original_path=original_path,
            overlay_path=overlay_path,
        )

    def compose_image(
        self,
        path: str | Path,
        settings: AppSettings,
        regions: list[TextRegion],
    ) -> ImageJobResult:
        source = Path(path)
        image = open_image(source)
        cleaned = inpaint_text(image, regions, padding=settings.mask_padding, radius=settings.inpaint_radius)
        final = render_translations(
            cleaned,
            regions,
            font_path=settings.font_path,
            font_size=settings.font_size,
            auto_font_size=settings.auto_font_size,
        )
        if settings.enable_upscale:
            if not settings.realesrgan_model_file:
                raise RuntimeError(realesrgan_missing_message(settings.realesrgan_model_path))
            final = upscale_anime(final, model_path=str(settings.realesrgan_model_file))

        suffix = _suffix(settings)
        stem = _output_stem(source)
        cleaned_path = save_output_image(cleaned, settings.output_dir, "cleaned", f"{stem}_cleaned", suffix=suffix)
        final_path = save_output_image(final, settings.output_dir, "final", f"{stem}_translated", suffix=suffix)
        return ImageJobResult(
            image_name=source.name,
            regions=regions,
            status="composed",
            original_path=save_output_image(image, settings.output_dir, "originals", f"{stem}_original", suffix=suffix),
            overlay_path=save_output_image(draw_region_overlay(image, regions), settings.output_dir, "overlays", f"{stem}_overlay", suffix=suffix),
            cleaned_path=cleaned_path,
            final_path=final_path,
        )


def rows_to_regions(rows: list[list[object]], base_regions: list[TextRegion]) -> list[TextRegion]:
    by_id = {region.id: region.model_copy(deep=True) for region in base_regions}
    for row in rows or []:
        if not row:
            continue
        region_id = int(row[0])
        region = by_id.get(region_id)
        if not region:
            continue
        region.enabled = bool(row[1])
        region.source_text = str(row[2] or "")
        region.translated_text = str(row[3] or "")
        region.confidence = float(row[4]) if row[4] not in {"", None} else region.confidence
        region.notes = str(row[5] or "")
    return [by_id[key] for key in sorted(by_id)]


def make_zip(paths: list[str], output_dir: str | Path = "outputs") -> str:
    out_path = Path(output_dir).expanduser() / "batches" / "translated_batch.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path and Path(path).exists():
                archive.write(path, arcname=Path(path).name)
    return str(out_path)


def export_regions_json(result: ImageJobResult, output_dir: str | Path = "outputs") -> str:
    out_path = Path(output_dir).expanduser() / "regions" / f"{Path(result.image_name).stem}_regions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([region.model_dump() for region in result.regions], ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)
