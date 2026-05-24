from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .model_manager import resolve_project_path, translation_missing_message


SENTENCE_RE = re.compile(r".+?[\u3002\uff01\uff1f!?]|.+$", re.DOTALL)
SUGOI_REQUIRED_FILES = [
    "model.bin",
    "spm/spm.ja.nopretok.model",
    "spm/spm.en.nopretok.model",
]
FUGUMT_MODEL_FILES = ["pytorch_model.bin", "model.safetensors"]
FUGUMT_TOKENIZER_FILES = ["tokenizer_config.json", "source.spm", "target.spm", "vocab.json"]


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


def _missing_files(base_path: Path, relative_paths: list[str]) -> list[str]:
    return [path for path in relative_paths if not (base_path / path).exists()]


def validate_llama_model_path(model_path: str) -> Path:
    resolved_path = resolve_project_path(model_path)
    if not resolved_path.exists():
        raise RuntimeError(translation_missing_message(resolved_path))
    if resolved_path.is_dir():
        raise RuntimeError(f"Gemma/llama requires a .gguf file. Current path points to a directory: {resolved_path}. Select Gemma or update the model path.")
    if resolved_path.suffix.lower() != ".gguf":
        raise RuntimeError(f"Gemma/llama requires a .gguf file. Current path is: {resolved_path}. Select Gemma or update the model path.")
    return resolved_path


def validate_sugoi_model_path(model_path: str) -> Path:
    resolved_path = resolve_project_path(model_path)
    if not resolved_path.exists():
        raise RuntimeError(translation_missing_message(resolved_path))
    if not resolved_path.is_dir():
        raise RuntimeError(f"Sugoi V4 requires a CTranslate2 model directory. Current path is: {resolved_path}. Select Sugoi V4 or update the model path.")
    missing = _missing_files(resolved_path, SUGOI_REQUIRED_FILES)
    if missing:
        raise RuntimeError(f"Sugoi V4 model directory is incomplete: {resolved_path}. Missing: {', '.join(missing)}.")
    return resolved_path


def validate_fugumt_model_path(model_path: str) -> Path:
    resolved_path = resolve_project_path(model_path)
    if not resolved_path.exists():
        raise RuntimeError(translation_missing_message(resolved_path))
    if not resolved_path.is_dir():
        raise RuntimeError(f"Fugu-MT requires a Hugging Face model directory. Current path is: {resolved_path}. Select Fugu-MT or update the model path.")
    if not (resolved_path / "config.json").exists():
        raise RuntimeError(f"Fugu-MT model directory is incomplete: {resolved_path}. Missing: config.json.")
    if not any((resolved_path / filename).exists() for filename in FUGUMT_MODEL_FILES):
        raise RuntimeError(f"Fugu-MT model directory is incomplete: {resolved_path}. Missing model weights.")
    if not any((resolved_path / filename).exists() for filename in FUGUMT_TOKENIZER_FILES):
        raise RuntimeError(f"Fugu-MT model directory is incomplete: {resolved_path}. Missing tokenizer files.")
    return resolved_path


@lru_cache(maxsize=1)
def get_llama(model_path: str, threads: int, context: int):
    resolved_path = validate_llama_model_path(model_path)
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


@lru_cache(maxsize=1)
def get_sugoi(model_path: str):
    resolved_path = validate_sugoi_model_path(model_path)
    try:
        import ctranslate2
        import sentencepiece
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("ctranslate2 and sentencepiece are not installed.") from exc

    spm_path = resolved_path / "spm"
    source_spm = spm_path / "spm.ja.nopretok.model"
    target_spm = spm_path / "spm.en.nopretok.model"
    translator = ctranslate2.Translator(str(resolved_path), device="cpu")
    source_tokenizer = sentencepiece.SentencePieceProcessor(str(source_spm))
    target_tokenizer = sentencepiece.SentencePieceProcessor(str(target_spm))
    return translator, source_tokenizer, target_tokenizer


@lru_cache(maxsize=1)
def get_fugumt(model_path: str):
    resolved_path = validate_fugumt_model_path(model_path)
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("transformers and torch are not installed.") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(resolved_path), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(resolved_path), local_files_only=True)
    model.to("cpu")
    model.eval()
    return tokenizer, model


def translate_japanese_to_english_sugoi(
    text: str,
    model_path: str,
    max_tokens: int = 256,
) -> str:
    chunks = split_for_translation(text)
    if not chunks:
        return ""
    translator, source_tokenizer, target_tokenizer = get_sugoi(model_path)
    tokenized = [source_tokenizer.encode(chunk, out_type=str) for chunk in chunks]
    results = translator.translate_batch(
        source=tokenized,
        beam_size=5,
        max_decoding_length=max_tokens,
    )
    translated = [
        target_tokenizer.decode(result.hypotheses[0]).replace("<unk>", "").strip()
        for result in results
    ]
    return " ".join(part for part in translated if part).strip()


def translate_japanese_to_english_fugumt(
    text: str,
    model_path: str,
    max_tokens: int = 256,
) -> str:
    chunks = split_for_translation(text)
    if not chunks:
        return ""
    tokenizer, model = get_fugumt(model_path)
    translated: list[str] = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=4,
        )
        translated.append(tokenizer.decode(output_ids[0], skip_special_tokens=True).strip())
    return " ".join(part for part in translated if part).strip()


def translate_japanese_to_english(
    text: str,
    model_path: str,
    backend: str = "llama",
    threads: int = 4,
    context: int = 2048,
    max_tokens: int = 256,
) -> str:
    normalized_backend = backend.strip().lower()
    if normalized_backend in {"ctranslate2", "sugoi"}:
        return translate_japanese_to_english_sugoi(text, model_path, max_tokens=max_tokens)
    if normalized_backend in {"fugu", "fugu-mt", "fugumt", "fugumt-ja-en", "transformers"}:
        return translate_japanese_to_english_fugumt(text, model_path, max_tokens=max_tokens)

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
