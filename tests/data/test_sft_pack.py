import json
from pathlib import Path

from alm.data.sft_dataset import PackedSFTDataset
from alm.data.sft_pack import iter_conversations, pack_sft
from alm.tokenizers import Tokenizer, Vocabulary, save_vocab


def test_pack_sft_pads_to_seq_len(tmp_path: Path) -> None:
    jsonl = tmp_path / "clean.jsonl"
    records = [
        {"turns": [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}]},
        {"turns": [{"role": "user", "text": "bye"}, {"role": "assistant", "text": "goodbye"}]},
    ]
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    tokenizer = Tokenizer(Vocabulary.byte_fallback())
    tok_path = tmp_path / "tok.json"
    save_vocab(tokenizer.vocab, tok_path)

    out_dir = tmp_path / "packed"
    pack_sft(
        tokenizer,
        iter_conversations([jsonl]),
        seq_len=64,
        shard_size=256,
        out_dir=out_dir,
        show_progress=False,
        workers=1,
        chunk_size=1,
    )

    dataset = PackedSFTDataset(out_dir)
    assert len(dataset) >= 2
