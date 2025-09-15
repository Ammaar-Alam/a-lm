#!/usr/bin/env python3
"""CLI for training the byte fallback tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from amlm.tokenizers import cli_train_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the a-lm tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Input text files")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary size")
    parser.add_argument("--out", required=True, help="Output tokenizer JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli_train_tokenizer(args.input, args.vocab_size, args.out)


if __name__ == "__main__":
    main()
