from pathlib import Path

import json
import numpy as np

from alm.data.dataset import PackedDataset, collate_tokens


def create_dummy_shard(tmp_path: Path, seq_len: int = 4) -> Path:
    shard = tmp_path / "shard_00000.bin"
    data = np.arange(seq_len * 4, dtype=np.uint32)
    data.tofile(shard)
    metadata = {
        "seq_len": seq_len,
        "shard_size": seq_len * 4,
        "total_tokens": int(data.size),
        "shards": [shard.name],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def test_packed_dataset_basic(tmp_path: Path) -> None:
    root = create_dummy_shard(tmp_path)
    dataset = PackedDataset(root)
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample.shape[0] == 4
    batch = collate_tokens([dataset[0], dataset[1]])
    assert batch.shape == (2, 4)
