import builtins

import pytest

from manga_translator import translate
from manga_translator.translate import (
    get_llama,
    get_sugoi,
    split_for_translation,
    translate_japanese_to_english_fugumt,
    validate_fugumt_model_path,
)


def test_split_for_translation_preserves_short_sentence():
    text = "\u3053\u3093\u306b\u3061\u306f\u3002"
    assert split_for_translation(text) == [text]


def test_split_for_translation_breaks_long_text():
    text = "\u3053\u308c\u306f\u9577\u3044\u6587\u7ae0\u3067\u3059\u3002" * 40
    chunks = split_for_translation(text, max_chars=60)
    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)


def test_fugumt_translation_joins_chunks(monkeypatch, tmp_path):
    class FakeInputs(dict):
        def __init__(self):
            super().__init__(input_ids=[1])

    class FakeTokenizer:
        def __call__(self, chunk, return_tensors, truncation):
            return FakeInputs()

        def decode(self, output_ids, skip_special_tokens):
            return "English"

    class FakeModel:
        def generate(self, **kwargs):
            return [[1, 2, 3]]

    model_dir = tmp_path / "fugumt"
    model_dir.mkdir()
    monkeypatch.setattr(translate, "get_fugumt", lambda model_path: (FakeTokenizer(), FakeModel()))

    assert translate_japanese_to_english_fugumt("\u3053\u3093\u306b\u3061\u306f\u3002", str(model_dir)) == "English"


def test_llama_rejects_directory_before_importing_llama_cpp(monkeypatch, tmp_path):
    get_llama.cache_clear()
    model_dir = tmp_path / "sugoi"
    model_dir.mkdir()
    original_import = builtins.__import__

    def fail_if_llama_imported(name, *args, **kwargs):
        if name == "llama_cpp":
            raise AssertionError("llama_cpp should not be imported for an invalid path")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_if_llama_imported)

    with pytest.raises(RuntimeError, match="requires a \\.gguf file.*directory"):
        get_llama(str(model_dir), 1, 512)


def test_llama_rejects_non_gguf_file_before_importing_llama_cpp(monkeypatch, tmp_path):
    get_llama.cache_clear()
    model_file = tmp_path / "model.bin"
    model_file.write_text("not gguf", encoding="utf-8")
    original_import = builtins.__import__

    def fail_if_llama_imported(name, *args, **kwargs):
        if name == "llama_cpp":
            raise AssertionError("llama_cpp should not be imported for an invalid path")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_if_llama_imported)

    with pytest.raises(RuntimeError, match="requires a \\.gguf file"):
        get_llama(str(model_file), 1, 512)


def test_sugoi_rejects_incomplete_directory_before_importing_ctranslate(monkeypatch, tmp_path):
    get_sugoi.cache_clear()
    model_dir = tmp_path / "sugoi"
    model_dir.mkdir()
    original_import = builtins.__import__

    def fail_if_ctranslate_imported(name, *args, **kwargs):
        if name in {"ctranslate2", "sentencepiece"}:
            raise AssertionError("translation dependencies should not be imported for an invalid path")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_if_ctranslate_imported)

    with pytest.raises(RuntimeError, match="Sugoi V4 model directory is incomplete"):
        get_sugoi(str(model_dir))


def test_fugumt_rejects_incomplete_directory(tmp_path):
    model_dir = tmp_path / "fugumt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Missing model weights"):
        validate_fugumt_model_path(str(model_dir))
