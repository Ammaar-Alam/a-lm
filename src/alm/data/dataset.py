"""Dataset utilities for packed token shards."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

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

        self.shards: list[ShardInfo] = []
        self.offsets: list[tuple[int, int]] = []
        self._cache: OrderedDict[int, np.memmap] = OrderedDict()
        self._max_cached_shards = 32
        itemsize = np.dtype(np.uint32).itemsize

        offset = 0
        for shard_idx, name in enumerate(shard_names):
            shard_path = root / name
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard {name} missing in {root}")
            tokens = shard_path.stat().st_size // itemsize
            sequences = tokens // self.seq_len
            if sequences == 0:
                continue
            self.shards.append(ShardInfo(path=shard_path, sequences=sequences))
            self.offsets.append((offset, offset + sequences))
            offset += sequences

        if offset == 0:
            raise ValueError("Packed dataset contains zero sequences")
        self.total_sequences = offset
        self._max_cached_shards = min(self._max_cached_shards, len(self.shards)) or 1

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(idx)
        for shard_index, ((start, end), info) in enumerate(zip(self.offsets, self.shards)):
            if start <= idx < end:
                arr = self._get_shard_memmap(shard_index, info)
                sample = arr[idx - start]
                return torch.from_numpy(np.array(sample, dtype=np.int64))
        raise IndexError(idx)

    def _get_shard_memmap(self, shard_index: int, info: ShardInfo) -> np.memmap:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached

        raw = np.memmap(info.path, dtype=np.uint32, mode="r")
        sequences = len(raw) // self.seq_len
        arr = raw[: sequences * self.seq_len].reshape(sequences, self.seq_len)
        self._cache[shard_index] = arr
        self._cache.move_to_end(shard_index)

        if len(self._cache) > self._max_cached_shards:
            old_index, old_arr = self._cache.popitem(last=False)
            try:
                old_arr._mmap.close()  # type: ignore[attr-defined]
            except AttributeError:
                pass
        return arr


def collate_tokens(samples: list[torch.Tensor]) -> torch.Tensor:
    if not samples:
        raise ValueError("Empty batch provided to collate_tokens")
    return torch.stack(samples, dim=0)
