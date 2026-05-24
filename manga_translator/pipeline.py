from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .detect import crop_region, detect_text_regions, draw_region_overlay
from .image_io import image_hash, open_image, save_output_image, short_hash
from .inpaint import inpaint_text, make_text_mask, refine_text_mask
from .model_manager import realesrgan_missing_message, translation_missing_message
from .models import AppSettings, ImageJobResult, TextRegion
from .ocr import recognize_japanese
from .profiles import apply_processing_profile
from .render import render_translations
from .text_processing import apply_replacements, clean_translation, load_replacement_dict, looks_repetitive
from .translate import translate_japanese_to_english
from .upscale import upscale_anime


def _suffix(settings: AppSettings) -> str:
    return ".jpg" if settings.output_format.upper() == "JPEG" else ".png"


def _output_stem(source: Path) -> str:
    return f"{source.stem}_{short_hash(source)}"


def _translated_output_stem(source: Path) -> str:
    return f"{source.stem}_translated"


def _save_output(image, settings: AppSettings, source: Path, category: str, name: str) -> str:
    return save_output_image(
        image,
        settings.output_dir,
        category,
        f"{_output_stem(source)}_{name}",
        suffix=_suffix(settings),
    )


def _save_translated_output(image, settings: AppSettings, source: Path) -> str:
    return save_output_image(
        image,
        settings.output_dir,
        "final",
        _translated_output_stem(source),
        suffix=_suffix(settings),
    )


class MangaTranslationPipeline:
    def __init__(self) -> None:
        self.region_cache: dict[str, list[TextRegion]] = {}

    def analyze_image(self, path: str | Path, settings: AppSettings, run_translation: bool = True) -> ImageJobResult:
        settings = apply_processing_profile(settings)
        source = Path(path)
        image = open_image(source)
        debug_paths: dict[str, str] = {}
        work_image = self._pre_upscale(image, settings)
        if work_image.size != image.size:
            debug_paths["pre_upscaled"] = _save_output(work_image, settings, source, "debug", "0_pre_upscaled")
        regions = self._detect_or_cached_regions(source, work_image, settings)
        if work_image.size != image.size and settings.revert_pre_upscale:
            self._scale_regions(regions, image.size[0] / work_image.size[0], image.size[1] / work_image.size[1])
            work_image = image
        debug_paths["raw_regions"] = self._export_regions_json(source, settings, regions, "1_raw_regions")
        regions = self._merge_regions(regions)
        debug_paths["merged_regions"] = self._export_regions_json(source, settings, regions, "2_merged_regions")
        if run_translation:
            self._translate_regions(regions, settings)
        debug_paths["translated_regions"] = self._export_regions_json(source, settings, regions, "3_translated_regions")

        overlay = draw_region_overlay(work_image, regions)
        return ImageJobResult(
            image_name=source.name,
            regions=regions,
            status="analyzed",
            original_path=_save_output(work_image, settings, source, "originals", "original"),
            overlay_path=_save_output(overlay, settings, source, "overlays", "overlay"),
            debug_paths=debug_paths,
        )

    def compose_image(
        self,
        path: str | Path,
        settings: AppSettings,
        regions: list[TextRegion],
    ) -> ImageJobResult:
        settings = apply_processing_profile(settings)
        source = Path(path)
        image = open_image(source)
        raw_mask = make_text_mask(image.size, regions, settings.mask_padding)
        refined_mask = refine_text_mask(raw_mask, settings.mask_refine_dilation, settings.mask_refine_blur) if settings.mask_refine else raw_mask
        debug_paths = {
            "raw_mask": self._save_mask(raw_mask, settings, source, "4_raw_mask"),
            "refined_mask": self._save_mask(refined_mask, settings, source, "5_refined_mask"),
        }
        cleaned = inpaint_text(
            image,
            regions,
            padding=settings.mask_padding,
            radius=settings.inpaint_radius,
            backend=settings.inpainter_backend,
            model_path=settings.inpaint_model_path,
            mask=refined_mask,
        )
        final = render_translations(
            cleaned,
            regions,
            font_path=settings.font_path,
            font_size=settings.font_size,
            auto_font_size=settings.auto_font_size,
            text_box_gap=settings.text_box_gap,
            line_gap=settings.line_gap,
            overflow_text=settings.overflow_text,
            render_direction=settings.render_direction,
            font_color=settings.font_color,
            stroke_color=settings.stroke_color,
            stroke_width=settings.stroke_width,
        )
        if settings.enable_upscale:
            if not settings.realesrgan_model_file:
                raise RuntimeError(realesrgan_missing_message(settings.realesrgan_model_path))
            final = upscale_anime(final, model_path=str(settings.realesrgan_model_file))

        return ImageJobResult(
            image_name=source.name,
            regions=regions,
            status="composed",
            original_path=_save_output(image, settings, source, "originals", "original"),
            overlay_path=_save_output(draw_region_overlay(image, regions), settings, source, "overlays", "overlay"),
            cleaned_path=_save_output(cleaned, settings, source, "cleaned", "cleaned"),
            final_path=_save_translated_output(final, settings, source),
            debug_paths=debug_paths,
        )

    def _detect_or_cached_regions(self, source: Path, image, settings: AppSettings) -> list[TextRegion]:
        cache_key = f"{image_hash(source)}:{settings.mask_padding}:{settings.detector_backend}:{settings.pre_upscale_ratio}"
        cached_regions = self.region_cache.get(cache_key, [])
        regions = [region.model_copy(deep=True) for region in cached_regions]
        if regions:
            return regions

        regions, detector_notes = detect_text_regions(image, settings.detector_backend)
        note = "; ".join(detector_notes)
        for region in regions:
            if note:
                region.notes = note
            crop = crop_region(image, region)
            try:
                region.source_text = recognize_japanese(crop)
                region.raw_text = region.source_text
            except Exception as exc:
                region.notes = f"{region.notes}; {exc}" if region.notes else str(exc)
        self.region_cache[cache_key] = [region.model_copy(deep=True) for region in regions]
        return regions

    def _translate_regions(self, regions: list[TextRegion], settings: AppSettings) -> None:
        if not settings.model_file:
            for region in regions:
                if region.source_text:
                    region.notes = translation_missing_message(settings.translation_model_path)
            return

        pre_replacements = load_replacement_dict(settings.pre_dict_path)
        post_replacements = load_replacement_dict(settings.post_dict_path)
        pending = [
            region
            for region in regions
            if region.source_text and not region.translated_text and not region.locked
        ]
        for region in pending:
            if not region.source_text:
                continue
            try:
                source_text = apply_replacements(region.source_text, pre_replacements)
                region.translated_text = translate_japanese_to_english(
                    source_text,
                    str(settings.model_file),
                    backend=settings.translation_backend,
                    threads=settings.llama_threads,
                    context=settings.llama_context,
                    max_tokens=settings.max_translation_tokens,
                )
                region.translated_text = clean_translation(apply_replacements(region.translated_text, post_replacements))
                if looks_repetitive(region.translated_text):
                    region.notes = f"{region.notes}; translation looks repetitive" if region.notes else "translation looks repetitive"
            except Exception as exc:
                region.notes = str(exc)

    def _pre_upscale(self, image, settings: AppSettings):
        if settings.pre_upscale_ratio <= 1:
            return image
        width = image.width * settings.pre_upscale_ratio
        height = image.height * settings.pre_upscale_ratio
        return image.resize((width, height))

    def _scale_regions(self, regions: list[TextRegion], sx: float, sy: float) -> None:
        for region in regions:
            region.box = [[point[0] * sx, point[1] * sy] for point in region.box]
            region.polygon = [[point[0] * sx, point[1] * sy] for point in region.polygon]
            x1, y1, x2, y2 = region.bbox
            region.bbox = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

    def _merge_regions(self, regions: list[TextRegion]) -> list[TextRegion]:
        if len(regions) < 2:
            return regions
        sorted_regions = sorted(regions, key=lambda r: (r.bbox[1] // 48, r.bbox[0]))
        merged: list[TextRegion] = []
        current: list[TextRegion] = []
        for region in sorted_regions:
            if current and not self._should_merge(current[-1], region):
                merged.append(self._make_merged_region(len(merged) + 1, current))
                current = []
            current.append(region)
        if current:
            merged.append(self._make_merged_region(len(merged) + 1, current))
        return merged

    def _should_merge(self, left: TextRegion, right: TextRegion) -> bool:
        lx1, ly1, lx2, ly2 = left.bbox
        rx1, ry1, rx2, ry2 = right.bbox
        avg_height = max(1, ((ly2 - ly1) + (ry2 - ry1)) // 2)
        vertical_gap = max(0, ry1 - ly2)
        horizontal_overlap = min(lx2, rx2) - max(lx1, rx1)
        same_column = horizontal_overlap > -avg_height
        same_orientation = left.vertical == right.vertical
        return same_orientation and same_column and vertical_gap <= avg_height * 0.85

    def _make_merged_region(self, region_id: int, parts: list[TextRegion]) -> TextRegion:
        if len(parts) == 1:
            region = parts[0].model_copy(deep=True)
            region.id = region_id
            return region
        x1 = min(region.bbox[0] for region in parts)
        y1 = min(region.bbox[1] for region in parts)
        x2 = max(region.bbox[2] for region in parts)
        y2 = max(region.bbox[3] for region in parts)
        box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        texts = [region.source_text for region in parts if region.source_text]
        confidences = [region.confidence for region in parts if region.confidence is not None]
        return TextRegion(
            id=region_id,
            box=box,
            polygon=box,
            bbox=(x1, y1, x2, y2),
            vertical=sum(1 for region in parts if region.vertical) > len(parts) / 2,
            merged_from=[region.id for region in parts],
            raw_text="\n".join(region.raw_text or region.source_text for region in parts if region.raw_text or region.source_text),
            source_text="\n".join(texts),
            confidence=sum(confidences) / len(confidences) if confidences else None,
            notes="; ".join(region.notes for region in parts if region.notes),
        )

    def _save_mask(self, mask, settings: AppSettings, source: Path, name: str) -> str:
        from PIL import Image

        return _save_output(Image.fromarray(mask), settings, source, "debug", name)

    def _export_regions_json(self, source: Path, settings: AppSettings, regions: list[TextRegion], name: str) -> str:
        out_path = Path(settings.output_dir).expanduser() / "debug" / f"{_output_stem(source)}_{name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([region.model_dump() for region in regions], ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out_path)


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
        if len(row) >= 11:
            region.locked = bool(row[2])
            region.direction = str(row[3] or "auto")
            region.source_text = str(row[4] or "")
            region.translated_text = str(row[5] or "")
            region.font_size = int(row[6]) if row[6] not in {"", None} else None
            region.font_color = str(row[7] or "")
            region.stroke_color = str(row[8] or "")
            region.confidence = float(row[9]) if row[9] not in {"", None} else region.confidence
            region.notes = str(row[10] or "")
        else:
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
