from pathlib import Path

from alm.tokenizers import Tokenizer, Vocabulary, save_vocab


def test_tokenizer_round_trip(tmp_path: Path) -> None:
    vocab = Vocabulary.byte_fallback()
    vocab.add("hello")
    vocab.add("world")
    tok_file = tmp_path / "tok.json"
    save_vocab(vocab, tok_file)
    tokenizer = Tokenizer.from_file(tok_file)
    text = "hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text
