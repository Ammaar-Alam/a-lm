from pathlib import Path

import json

from alm.tokenizers import bpe_trainer


def test_bpe_training_round_trip(tmp_path: Path) -> None:
    corpus = ["hello world", "hello there", "world of words"]
    vocab = bpe_trainer.train_bpe(corpus, vocab_size=300)
    assert len(vocab) >= 256
    out_path = tmp_path / "tokenizer.json"
    bpe_trainer.save_vocab(vocab, out_path)
    loaded = bpe_trainer.load_vocab(out_path)
    assert loaded.id_to_token[: len(vocab.id_to_token)] == vocab.id_to_token


def test_cli_train_tokenizer(tmp_path: Path) -> None:
    file1 = tmp_path / "file1.txt"
    file1.write_text("hello world")
    out = tmp_path / "tok.json"
    bpe_trainer.cli_train_tokenizer([str(file1)], vocab_size=300, output_path=str(out))
    data = json.loads(out.read_text())
    assert "tokens" in data
    assert len(data["tokens"]) >= 256
