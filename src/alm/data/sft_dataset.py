"""Dataset for SFT packed shards."""

from __future__ import annotations

import json
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class _ShardInfo:
    inputs: Path
    masks: Path
    sequences: int


class PackedSFTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Loads packed SFT shards with token and loss mask pairs."""

    def __init__(self, root: Path) -> None:
        root = Path(root)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {root}")
        metadata = json.loads(metadata_path.read_text())
        self.seq_len = int(metadata["seq_len"])
        input_files = metadata.get("inputs", [])
        mask_files = metadata.get("masks", [])
        if not input_files or len(input_files) != len(mask_files):
            raise ValueError("metadata.json must include matching inputs and masks")

        dtype_name = str(metadata.get("dtype", "uint32")).lower()
        if dtype_name not in {"uint16", "uint32"}:
            raise ValueError(f"Unsupported token dtype {dtype_name}")
        self._token_dtype = np.uint16 if dtype_name == "uint16" else np.uint32

        self._shards: list[_ShardInfo] = []
        self._offsets: list[tuple[int, int]] = []
        self._input_cache: OrderedDict[int, np.memmap] = OrderedDict()
        self._mask_cache: OrderedDict[int, np.memmap] = OrderedDict()
        self._max_cached = 16
        itemsize_tokens = np.dtype(self._token_dtype).itemsize
        itemsize_masks = np.dtype(np.uint8).itemsize

        offset = 0
        for idx, (input_name, mask_name) in enumerate(zip(input_files, mask_files)):
            input_path = root / input_name
            mask_path = root / mask_name
            if not input_path.exists() or not mask_path.exists():
                raise FileNotFoundError(f"Shard files missing for index {idx}")
            tokens = input_path.stat().st_size // itemsize_tokens
            sequences = tokens // self.seq_len
            mask_tokens = mask_path.stat().st_size // itemsize_masks
            if sequences == 0 or mask_tokens < sequences * self.seq_len:
                continue
            self._shards.append(_ShardInfo(input_path, mask_path, sequences))
            self._offsets.append((offset, offset + sequences))
            offset += sequences
        if offset == 0:
            raise ValueError("SFT dataset contains zero sequences")
        self.total_sequences = offset
        self._max_cached = min(self._max_cached, len(self._shards)) or 1

    def __len__(self) -> int:
        return self.total_sequences

    def _locate(self, idx: int) -> tuple[int, _ShardInfo, int]:
        for shard_index, ((start, end), shard) in enumerate(zip(self._offsets, self._shards)):
            if start <= idx < end:
                return shard_index, shard, idx - start
        raise IndexError(idx)

    def _get_array(
        self,
        cache: OrderedDict[int, np.memmap],
        shard_index: int,
        path: Path,
        dtype: np.dtype,
    ) -> np.memmap:
        cached = cache.get(shard_index)
        if cached is not None:
            cache.move_to_end(shard_index)
            return cached
        array = np.memmap(path, dtype=dtype, mode="r")
        cache[shard_index] = array
        cache.move_to_end(shard_index)
        while len(cache) > self._max_cached:
            old_index, old_array = cache.popitem(last=False)
            with suppress(AttributeError):
                old_array._mmap.close()  # type: ignore[attr-defined]
        return array

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(idx)
        shard_index, shard, local = self._locate(idx)
        token_array = self._get_array(
            self._input_cache, shard_index, shard.inputs, self._token_dtype
        )
        mask_array = self._get_array(self._mask_cache, shard_index, shard.masks, np.uint8)
        start = local * self.seq_len
        end = start + self.seq_len
        tokens = np.array(token_array[start:end], copy=True)
        mask = np.array(mask_array[start:end], copy=False)
        token_tensor = torch.from_numpy(tokens)
        mask_tensor = torch.from_numpy(mask.astype(np.bool_, copy=False))
        return token_tensor, mask_tensor
