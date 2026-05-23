from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def open_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    return image


def image_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def short_hash(path: str | Path) -> str:
    return image_hash(path)[:10]


def save_output_image(image: Image.Image, output_dir: str | Path, category: str, stem: str, suffix: str = ".png") -> str:
    out_dir = Path(output_dir).expanduser() / category
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe_stem(stem)}{suffix}"
    save_image = image
    if suffix.lower() in {".jpg", ".jpeg"} and image.mode == "RGBA":
        save_image = image.convert("RGB")
    save_image.save(path)
    return str(path)
