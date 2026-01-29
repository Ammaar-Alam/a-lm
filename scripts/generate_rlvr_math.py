#!/usr/bin/env python3
"""Generate synthetic math prompts for verifiable-reward training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alm.rlvr.math_tasks import generate_math_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RLVR math dataset (JSONL)")
    parser.add_argument("--out", default="data/rlvr/math.jsonl", help="Output JSONL path")
    parser.add_argument("--count", type=int, default=20000, help="Number of examples to generate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt used in the dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    examples = generate_math_examples(count=args.count, seed=args.seed, system=args.system)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps({"prompt": example.prompt, "answer": example.answer}, ensure_ascii=False)
                + "\n"
            )
    print(f"Wrote {len(examples):,} examples to {out_path}")


if __name__ == "__main__":
    main()
