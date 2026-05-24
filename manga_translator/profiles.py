from __future__ import annotations

from .models import AppSettings


PROFILE_CHOICES = [
    ("Quality", "quality"),
    ("Fast", "fast"),
]


def apply_processing_profile(settings: AppSettings) -> AppSettings:
    profile = settings.processing_profile.strip().lower()
    updates: dict[str, object] = {"processing_profile": profile}
    if profile == "quality":
        if settings.detector_backend in {"", "auto"}:
            updates["detector_backend"] = "ctd"
        if settings.inpainter_backend == "opencv-telea":
            updates["inpainter_backend"] = "anime-lama"
            if not settings.inpaint_model_path:
                from .model_manager import default_inpaint_model_path

                updates["inpaint_model_path"] = str(default_inpaint_model_path("anime-lama"))
        updates["mask_refine"] = True
        updates["revert_pre_upscale"] = True
    elif profile == "fast":
        if settings.detector_backend in {"", "auto", "ctd"}:
            updates["detector_backend"] = "paddle"
        if settings.inpainter_backend in {"anime-lama", "big-lama", "migan"}:
            updates["inpainter_backend"] = "opencv-telea"
            updates["inpaint_model_path"] = ""
        updates["pre_upscale_ratio"] = 1
    return settings.model_copy(update=updates)
