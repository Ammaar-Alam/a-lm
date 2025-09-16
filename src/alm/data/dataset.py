"""Dataset utilities for packed token shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ShardInfo:
    path: Path
    sequences: int


class PackedDataset(Dataset[torch.Tensor]):
    """Dataset that loads token sequences from packed `.bin` shards."""

    def __init__(self, root: Path) -> None:
        root = Path(root)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {root}")
        metadata = json.loads(metadata_path.read_text())
        self.seq_len = int(metadata["seq_len"])
        shard_names = metadata.get("shards", [])
        if not shard_names:
            raise ValueError("No shards listed in metadata.json")

        self.shards: List[ShardInfo] = []
        self.memmap_arrays: List[np.memmap] = []
        self.offsets: List[Tuple[int, int]] = []

        offset = 0
        for name in shard_names:
            shard_path = root / name
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard {name} missing in {root}")
            raw = np.memmap(shard_path, dtype=np.uint32, mode="r")
            sequences = len(raw) // self.seq_len
            trimmed = raw[: sequences * self.seq_len]
            arr = trimmed.reshape(sequences, self.seq_len)
            self.memmap_arrays.append(arr)
            self.shards.append(ShardInfo(path=shard_path, sequences=sequences))
            self.offsets.append((offset, offset + sequences))
            offset += sequences

        if offset == 0:
            raise ValueError("Packed dataset contains zero sequences")
        self.total_sequences = offset

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(idx)
        for (start, end), arr in zip(self.offsets, self.memmap_arrays):
            if start <= idx < end:
                sample = arr[idx - start]
                return torch.from_numpy(np.array(sample, dtype=np.int64))
        raise IndexError(idx)


def collate_tokens(samples: List[torch.Tensor]) -> torch.Tensor:
    if not samples:
        raise ValueError("Empty batch provided to collate_tokens")
    return torch.stack(samples, dim=0)
