from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .model_manager import resolve_project_path, translation_missing_message


SENTENCE_RE = re.compile(r".+?[\u3002\uff01\uff1f!?]|.+$", re.DOTALL)


def split_for_translation(text: str, max_chars: int = 180) -> list[str]:
    text = re.sub(r"[ \t\r\f\v]+", " ", text.strip())
    if not text:
        return []
    line_parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            line_parts.extend(match.group(0).strip() for match in SENTENCE_RE.finditer(line))

    chunks: list[str] = []
    for part in line_parts or [text]:
        while len(part) > max_chars:
            chunks.append(part[:max_chars])
            part = part[max_chars:]
        if part:
            chunks.append(part)
    return chunks


@lru_cache(maxsize=1)
def get_llama(model_path: str, threads: int, context: int):
    resolved_path = resolve_project_path(model_path)
    if not resolved_path.exists():
        raise RuntimeError(translation_missing_message(resolved_path))
    try:
        from llama_cpp import Llama
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("llama-cpp-python is not installed.") from exc
    return Llama(
        model_path=str(resolved_path),
        n_ctx=context,
        n_threads=threads,
        n_gpu_layers=0,
        verbose=False,
    )


def translate_japanese_to_english(
    text: str,
    model_path: str,
    threads: int = 4,
    context: int = 2048,
    max_tokens: int = 256,
) -> str:
    chunks = split_for_translation(text)
    if not chunks:
        return ""
    llm = get_llama(model_path, threads, context)
    translated: list[str] = []
    for chunk in chunks:
        prompt = (
            "You are a professional Japanese to English manga translator. "
            "Translate naturally, preserve tone, and output only English.\n\n"
            f"Japanese:\n{chunk}\n\nEnglish:"
        )
        result = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            stop=["\nJapanese:", "\n\nJapanese:"],
        )
        translated.append(result["choices"][0]["text"].strip())
    return " ".join(part for part in translated if part).strip()
