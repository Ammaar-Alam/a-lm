from pathlib import Path

from alm.tokenizers import train_unigram


def test_unigram_trainer_builds_vocab(tmp_path: Path) -> None:
    corpus = ["hello world", "hello there"]
    vocab = train_unigram(corpus, vocab_size=300)
    assert len(vocab) >= 256
    assert any(token == "hello" for token in vocab.id_to_token)
