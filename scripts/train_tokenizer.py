#!/usr/bin/env python3
"""CLI for training the tokenizer used by a-lm."""

from __future__ import annotations

import argparse

from alm.tokenizers import cli_train_tokenizer
from alm.tokenizers.hf_trainer import cli_train_hf_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the a-lm tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Input text files")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary size")
    parser.add_argument("--out", required=True, help="Output tokenizer JSON path")
    parser.add_argument(
        "--backend",
        choices=("hf", "python"),
        default="hf",
        help="Tokenizer training backend (default: hf)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        help="Uniformly sample up to this many lines from the corpus before training",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        help="Optional Bernoulli sampling ratio (0-1] applied before the max-lines cap",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="Print status every N merges (default: 500)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency for hf backend (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "python":
        cli_train_tokenizer(
            args.input,
            args.vocab_size,
            args.out,
            max_lines=args.max_lines,
            sample_ratio=args.sample_ratio,
            log_interval=args.log_interval,
            seed=args.seed,
        )
        return

    cli_train_hf_bpe(
        args.input,
        args.vocab_size,
        args.out,
        max_lines=args.max_lines,
        sample_ratio=args.sample_ratio,
        min_frequency=args.min_frequency,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
