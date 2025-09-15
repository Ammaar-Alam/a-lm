from amlm.tokenizers.normalizer import iter_bytes, normalize_text


def test_normalize_whitespace_and_control_chars() -> None:
    raw = "Hello\u00A0World\r\nTab\tControl\x07"
    normalized = normalize_text(raw)
    assert normalized == "Hello World\nTab\tControl"


def test_iter_bytes_encodes_utf8() -> None:
    text = "Café"
    as_bytes = iter_bytes(text)
    assert isinstance(as_bytes, bytes)
    assert as_bytes.decode("utf-8") == normalize_text(text)
