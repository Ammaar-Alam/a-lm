#!/usr/bin/env python3
"""Pack cleaned text into mmap-friendly token shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from alm.data.pack import iter_text_files, pack_tokens
from alm.tokenizers import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack text files into token shards")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON")
    parser.add_argument(
        "--in", dest="input_dir", required=True, help="Directory with cleaned text files"
    )
    parser.add_argument("--out", required=True, help="Directory for binary shards")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for packing")
    parser.add_argument(
        "--shard-size", type=int, default=2048, help="Number of tokens per shard chunk"
    )
    parser.add_argument("--eos", default="\n", help="EoS string appended between documents")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live progress display",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(Path(args.tokenizer))
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise SystemExit("No .txt files found in input directory")
    metadata = pack_tokens(
        tokenizer,
        iter_text_files(files),
        seq_len=args.seq_len,
        shard_size=args.shard_size,
        out_dir=Path(args.out),
        eos_token=args.eos,
        show_progress=not args.no_progress,
    )
    total_shards = len(metadata["shards"])
    total_tokens = metadata["total_tokens"]
    print(
        f"Packed {total_tokens:,} tokens across {total_shards:,} shards (seq_len={args.seq_len})."
    )


if __name__ == "__main__":
    main()
