#!/usr/bin/env python3
"""Inspect packed SFT shards by decoding a few sequences.

This is a debugging utility to verify:
- prompt formatting matches chat_cli (System/User/Assistant + optional EOT),
- loss masks align with assistant tokens,
- sequences are padded/truncated as expected.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from alm.tokenizers import Tokenizer


def _read_metadata(root: Path) -> dict:
    path = root / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found in {root}")
    return json.loads(path.read_text())


def _dtype_from_name(name: str) -> np.dtype:
    name = str(name).lower()
    if name == "uint16":
        return np.uint16
    if name == "uint32":
        return np.uint32
    raise ValueError(f"Unsupported dtype {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect packed SFT dataset sequences")
    parser.add_argument(
        "--data",
        required=True,
        help="Packed SFT directory (contains metadata.json)",
    )
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--num", type=int, default=3, help="Number of sequences to print")
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for sampling sequences",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="Max characters to print for decoded text",
    )
    parser.add_argument(
        "--index",
        type=int,
        action="append",
        default=[],
        help="Specific sequence index to print (repeatable). Defaults to random samples.",
    )
    return parser.parse_args()


def _pick_indices(total: int, requested: list[int], num: int, seed: int) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for idx in requested:
        if 0 <= idx < total and idx not in seen:
            indices.append(idx)
            seen.add(idx)
    if len(indices) >= num:
        return indices[:num]
    rng = random.Random(seed)
    while len(indices) < min(num, total):
        idx = rng.randrange(total)
        if idx in seen:
            continue
        indices.append(idx)
        seen.add(idx)
    return indices


def main() -> None:
    args = parse_args()
    root = Path(args.data)
    metadata = _read_metadata(root)
    seq_len = int(metadata["seq_len"])
    dtype = _dtype_from_name(str(metadata.get("dtype", "uint32")))
    tokenizer = Tokenizer.from_file(Path(args.tokenizer))

    input_files = metadata.get("inputs", [])
    mask_files = metadata.get("masks", [])
    if not input_files or len(input_files) != len(mask_files):
        raise ValueError("metadata.json must include matching inputs and masks")

    input_path = root / input_files[0]
    mask_path = root / mask_files[0]
    inputs = np.memmap(input_path, dtype=dtype, mode="r")
    masks = np.memmap(mask_path, dtype=np.uint8, mode="r")
    sequences = min(inputs.size // seq_len, masks.size // seq_len)
    if sequences <= 0:
        raise ValueError("No sequences available in first shard")
    inputs = inputs[: sequences * seq_len].reshape(sequences, seq_len)
    masks = masks[: sequences * seq_len].reshape(sequences, seq_len)

    print(f"data={root}")
    print(f"seq_len={seq_len} sequences(first_shard)={sequences} dtype={dtype}")
    print(
        f"default_system_prompt={metadata.get('default_system_prompt')!r} "
        f"eot_token={metadata.get('eot_token')!r}"
    )

    indices = _pick_indices(sequences, args.index, args.num, args.seed)
    for idx in indices:
        ids = inputs[idx].tolist()
        mask = masks[idx].astype(bool)
        mask_frac = float(mask.mean())

        full = tokenizer.decode(ids).replace("\x00", "�")
        assistant_only_ids = [tok for tok, keep in zip(ids, mask, strict=False) if bool(keep)]
        assistant = tokenizer.decode(assistant_only_ids).replace("\x00", "�")

        print("\n" + "=" * 80)
        print(f"sequence={idx} mask_frac={mask_frac:.2f} mask0={bool(mask[0])}")
        print("- FULL (decoded)")
        print(full[: args.max_chars].rstrip())
        print("- ASSISTANT-ONLY (decoded)")
        print(assistant[: args.max_chars].rstrip())


if __name__ == "__main__":
    main()
