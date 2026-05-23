from manga_translator.translate import split_for_translation


def test_split_for_translation_preserves_short_sentence():
    text = "\u3053\u3093\u306b\u3061\u306f\u3002"
    assert split_for_translation(text) == [text]


def test_split_for_translation_breaks_long_text():
    text = "\u3053\u308c\u306f\u9577\u3044\u6587\u7ae0\u3067\u3059\u3002" * 40
    chunks = split_for_translation(text, max_chars=60)
    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)
