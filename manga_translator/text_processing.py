from __future__ import annotations

from pathlib import Path


def load_replacement_dict(path_value: str = "") -> list[tuple[str, str]]:
    if not path_value:
        return []
    path = Path(path_value).expanduser()
    if not path.exists():
        return []

    replacements: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            source, target = line.split("=>", 1)
        elif "\t" in line:
            source, target = line.split("\t", 1)
        elif "=" in line:
            source, target = line.split("=", 1)
        else:
            continue
        source = source.strip()
        if source:
            replacements.append((source, target.strip()))
    return replacements


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    out = text
    for source, target in replacements:
        out = out.replace(source, target)
    return out


def looks_repetitive(text: str, threshold: int = 20) -> bool:
    normalized = "".join(text.split())
    if len(normalized) < threshold:
        return False
    for size in range(1, min(12, len(normalized) // 2) + 1):
        unit = normalized[:size]
        repeats = len(normalized) // size
        if repeats >= 4 and unit * repeats == normalized[: size * repeats]:
            return True
    return False


def clean_translation(text: str) -> str:
    return " ".join(text.replace("<unk>", "").split()).strip()
