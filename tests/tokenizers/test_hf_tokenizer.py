from pathlib import Path

import pytest

from alm.tokenizers import Tokenizer as AlmTokenizer

pytest.importorskip("tokenizers", reason="tokenizers not installed")


def test_hf_tokenizer_load_round_trip(tmp_path: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nhello there\n", encoding="utf-8")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=200,
        min_frequency=1,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]"],
    )
    tokenizer.train([str(corpus)], trainer=trainer)

    tok_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tok_path), pretty=True)

    loaded = AlmTokenizer.from_file(tok_path)
    text = "hello world"
    ids = loaded.encode(text)
    assert ids
    decoded = loaded.decode(ids)
    assert "hello" in decoded
    assert "world" in decoded
