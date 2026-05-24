from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .model_manager import NEURAL_INPAINTERS, OPENCV_INPAINTERS, inpaint_missing_message, normalize_inpainter_backend, resolve_project_path
from .models import TextRegion

OPENCV_FLAGS = {
    "opencv-telea": cv2.INPAINT_TELEA,
    "opencv-ns": cv2.INPAINT_NS,
}


def make_text_mask(size: tuple[int, int], regions: list[TextRegion], padding: int) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        if not region.enabled:
            continue
        x1, y1, x2, y2 = region.bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def _restore_alpha(result: Image.Image, original: Image.Image) -> Image.Image:
    if original.mode == "RGBA":
        result.putalpha(original.getchannel("A"))
    return result


def _opencv_inpaint(image: Image.Image, mask: np.ndarray, radius: int, flag: int) -> Image.Image:
    original_mode = image.mode
    rgb = image.convert("RGB")
    array = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    cleaned = cv2.inpaint(array, mask, radius, flag)
    cleaned_rgb = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    return _restore_alpha(cleaned_rgb, image.convert(original_mode))


@lru_cache(maxsize=3)
def _load_torchscript_model(model_path: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("Neural inpainters require torch. Install requirements-inpaint.txt first.") from exc

    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


def _pad_to_mod(array: np.ndarray, mod: int, value: float = 0) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = array.shape[-2:]
    pad_h = (mod - height % mod) % mod
    pad_w = (mod - width % mod) % mod
    if pad_h == 0 and pad_w == 0:
        return array, (height, width)
    padded = np.pad(array, [(0, 0), (0, pad_h), (0, pad_w)], mode="constant", constant_values=value)
    return padded, (height, width)


def _tensor_to_rgb_image(output) -> np.ndarray:
    array = output[0].detach().cpu().float().numpy()
    if array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.max() <= 2:
        array = array * 255
    return np.clip(array, 0, 255).astype("uint8")


def _run_lama_model(model, rgb_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    image = np.transpose(rgb_array.astype("float32") / 255.0, (2, 0, 1))
    mask_norm = (mask.astype("float32") / 255.0)[None, :, :]
    image, (height, width) = _pad_to_mod(image, 8)
    mask_norm, _ = _pad_to_mod(mask_norm, 8)
    with torch.no_grad():
        output = model(
            torch.from_numpy(image).unsqueeze(0),
            torch.from_numpy((mask_norm > 0).astype("float32")).unsqueeze(0),
        )
    return _tensor_to_rgb_image(output)[:height, :width]


def _run_migan_model(model, rgb_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    height, width = rgb_array.shape[:2]
    resized_image = cv2.resize(rgb_array, (512, 512), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    image = np.transpose(resized_image.astype("float32") / 255.0, (2, 0, 1))
    image = image * 2 - 1
    mask_norm = ((resized_mask > 120).astype("float32"))[None, :, :]
    erased_image = image * (1 - mask_norm)
    input_image = np.concatenate([0.5 - mask_norm, erased_image], axis=0)
    with torch.no_grad():
        output = model(torch.from_numpy(input_image).unsqueeze(0))
    result = _tensor_to_rgb_image(output)
    return cv2.resize(result, (width, height), interpolation=cv2.INTER_CUBIC)


def _neural_inpaint(image: Image.Image, mask: np.ndarray, backend: str, model_path: str) -> Image.Image:
    resolved_path = resolve_project_path(model_path)
    if not resolved_path.exists():
        raise RuntimeError(inpaint_missing_message(backend, resolved_path))

    rgb = image.convert("RGB")
    rgb_array = np.array(rgb)
    model = _load_torchscript_model(str(resolved_path))
    if backend == "migan":
        result = _run_migan_model(model, rgb_array, mask)
    else:
        result = _run_lama_model(model, rgb_array, mask)

    mask_pixels = mask > 0
    composed = rgb_array.copy()
    composed[mask_pixels] = result[mask_pixels]
    return _restore_alpha(Image.fromarray(composed), image)


def inpaint_text(
    image: Image.Image,
    regions: list[TextRegion],
    padding: int = 8,
    radius: int = 3,
    backend: str = "opencv-telea",
    model_path: str = "",
) -> Image.Image:
    backend = normalize_inpainter_backend(backend)
    mask = make_text_mask(image.size, regions, padding)
    if not mask.any():
        return image.copy()
    if backend in OPENCV_INPAINTERS:
        return _opencv_inpaint(image, mask, radius, OPENCV_FLAGS[backend])
    if backend in NEURAL_INPAINTERS:
        return _neural_inpaint(image, mask, backend, model_path)
    raise ValueError(f"Unknown inpainter backend: {backend}")
