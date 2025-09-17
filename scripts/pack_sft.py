#!/usr/bin/env python3
"""Pack supervised fine-tuning data into mmap shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alm.data.sft_pack import iter_conversations, pack_sft
from alm.tokenizers import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack SFT conversations")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON")
    parser.add_argument("--jsonl", nargs="+", required=True, help="One or more JSONL files")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--shard-size", type=int, default=1_000_000, help="Tokens per shard")
    parser.add_argument("--workers", type=int, help="Number of encoding workers")
    parser.add_argument("--chunk-size", type=int, default=64, help="Conversations per worker batch")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(Path(args.tokenizer))
    conversations = iter_conversations([Path(path) for path in args.jsonl])
    metadata = pack_sft(
        tokenizer,
        conversations,
        seq_len=args.seq_len,
        shard_size=args.shard_size,
        out_dir=Path(args.out),
        show_progress=not args.no_progress,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
