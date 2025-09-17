import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from alm.data.sft_dataset import PackedSFTDataset
from alm.data.sft_pack import iter_conversations, pack_sft
from alm.tokenizers import Tokenizer, Vocabulary
from scripts import train_sft


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_sft_single_step(tmp_path: Path) -> None:
    jsonl = tmp_path / "clean.jsonl"
    conversation = {
        "turns": [
            {"role": "user", "text": "hello there kindly help me"},
            {"role": "assistant", "text": "sure here is a short reply"},
        ]
    }
    jsonl.write_text(json.dumps(conversation) + "\n")

    tokenizer = Tokenizer(Vocabulary.byte_fallback())
    packed_dir = tmp_path / "packed"
    pack_sft(
        tokenizer,
        iter_conversations([jsonl]),
        seq_len=8,
        shard_size=32,
        out_dir=packed_dir,
        show_progress=False,
        workers=1,
        chunk_size=1,
    )

    dataset = PackedSFTDataset(packed_dir)
    assert len(dataset) >= 1

    model_cfg = {
        "model": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "ffn_hidden_size": 32,
            "vocab_size": 256,
            "max_position_embeddings": 64,
            "dropout": 0.0,
            "alibi": False,
            "dual_ffn": {"enabled": False},
        }
    }
    train_cfg = {
        "optim": {"lr": 1e-3},
        "scheduler": {"warmup_steps": 0, "max_steps": 1},
        "training": {
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
            "max_steps": 1,
            "checkpoint_interval": 1,
            "gradient_clip_norm": 1.0,
        },
        "logging": {"log_interval": 1},
    }

    model_yaml = _write_yaml(tmp_path / "model.yaml", model_cfg)
    train_yaml = _write_yaml(tmp_path / "train.yaml", train_cfg)
    run_dir = tmp_path / "runs"

    args = SimpleNamespace(
        model=str(model_yaml),
        train=str(train_yaml),
        data=str(packed_dir),
        out=str(run_dir),
        device="cpu",
        resume=None,
        init=None,
    )

    train_sft.train(args)
    assert (run_dir / "ckpt-last.pt").exists()
