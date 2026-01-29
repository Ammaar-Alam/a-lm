import json
from pathlib import Path

from alm.data.pack import iter_text_files, pack_tokens
from alm.tokenizers import Tokenizer, Vocabulary, save_vocab


def build_tokenizer(tmp_path: Path) -> Path:
    vocab = Vocabulary.byte_fallback()
    vocab.add("hello")
    vocab.add("world")
    path = tmp_path / "tok.json"
    save_vocab(vocab, path)
    return path


def test_pack_tokens(tmp_path: Path) -> None:
    tok_path = build_tokenizer(tmp_path)
    tokenizer = Tokenizer.from_file(tok_path)
    text_dir = tmp_path / "clean"
    text_dir.mkdir()
    (text_dir / "sample.txt").write_text("hello world\n")
    out_dir = tmp_path / "packed"
    pack_tokens(
        tokenizer,
        iter_text_files(sorted(text_dir.glob("*.txt"))),
        seq_len=4,
        shard_size=4,
        out_dir=out_dir,
        eos_token="\n",
        tokenizer_path=tok_path,
    )
    shards = list(out_dir.glob("shard_*.bin"))
    assert shards, "Expected at least one shard"
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["total_tokens"] > 0
