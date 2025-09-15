#!/usr/bin/env python3
"""Prepare raw corpora according to YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from alm.data.config import load_corpus_config
from alm.data.prepare import prepare_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare text corpora for a-lm")
    parser.add_argument("--src", required=True, help="Path to corpus YAML config")
    parser.add_argument("--out", required=True, help="Directory to write cleaned text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_corpus_config(Path(args.src))
    prepare_all(config, Path(args.out))


if __name__ == "__main__":
    main()
