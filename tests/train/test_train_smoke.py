from pathlib import Path
from types import SimpleNamespace

import yaml

from alm.data.pack import iter_text_files, pack_tokens
from alm.tokenizers import Tokenizer, Vocabulary, save_vocab
from scripts import train_pretrain


def create_packed_dataset(tmp_path: Path) -> tuple[Path, Path]:
    text_dir = tmp_path / "clean"
    text_dir.mkdir()
    (text_dir / "sample.txt").write_text("hello world\nhello again\n", encoding="utf-8")
    tokenizer = Tokenizer(Vocabulary.byte_fallback())
    tok_path = tmp_path / "tok.json"
    save_vocab(tokenizer.vocab, tok_path)
    packed_dir = tmp_path / "packed"
    pack_tokens(
        tokenizer,
        iter_text_files(sorted(text_dir.glob("*.txt"))),
        seq_len=8,
        shard_size=8,
        out_dir=packed_dir,
        eos_token="\n",
        tokenizer_path=tok_path,
    )
    return packed_dir, tok_path


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_train_single_step(tmp_path: Path, monkeypatch) -> None:
    data_dir, tok_path = create_packed_dataset(tmp_path)
    model_cfg = {
        "model": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "ffn_hidden_size": 32,
            "vocab_size": 256,
            "max_position_embeddings": 128,
            "rope_theta": 10000,
            "dropout": 0.0,
            "alibi": False,
            "dual_ffn": {"enabled": False},
        }
    }
    train_cfg = {
        "optim": {"lr": 1e-3},
        "scheduler": {"warmup_steps": 0, "max_steps": 1},
        "training": {
            "micro_batch_size": 2,
            "gradient_accumulation": 1,
            "max_steps": 1,
            "checkpoint_interval": 1,
            "gradient_clip_norm": 1.0,
        },
        "logging": {"log_interval": 1},
    }
    model_path = write_yaml(tmp_path / "model.yaml", model_cfg)
    train_path = write_yaml(tmp_path / "train.yaml", train_cfg)
    out_dir = tmp_path / "runs"

    args = SimpleNamespace(
        model=str(model_path),
        train=str(train_path),
        data=str(data_dir),
        out=str(out_dir),
        device="cpu",
        resume=None,
        tokenizer=str(tok_path),
    )
    train_pretrain.train(args)
    assert (out_dir / "ckpt-last.pt").exists()
