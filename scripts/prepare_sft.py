#!/usr/bin/env python3
"""Prepare supervised fine-tuning conversations from public datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypedDict

from datasets import load_dataset

from alm.tokenizers.normalizer import normalize_text


class Turn(TypedDict):
    role: str
    text: str


class Conversation(TypedDict, total=False):
    system: str
    turns: list[Turn]


def write_jsonl(path: Path, records: Iterable[Conversation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_ultrachat(split: str, limit: int | None) -> Iterator[Conversation]:
    try:
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True)
    except ValueError as error:
        if split != "train_sft":
            dataset = load_dataset(
                "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True
            )
        else:  # pragma: no cover - defensive fallback
            raise error
    count = 0
    for sample in dataset:
        messages = (
            sample.get("messages")
            or sample.get("conversation")
            or sample.get("conversations")
            or []
        )
        turns: list[Turn] = []
        for message in messages:
            role = str(message.get("role") or message.get("from") or "").lower()
            if role not in {"user", "assistant"}:
                continue
            text = normalize_text(str(message.get("content") or message.get("text") or ""))
            if text:
                turns.append({"role": role, "text": text})
        if not turns or not any(turn["role"] == "assistant" for turn in turns):
            continue
        yield {"turns": turns}
        count += 1
        if limit is not None and count >= limit:
            return


def iter_oasst(split: str, limit: int | None) -> Iterator[Conversation]:
    dataset = load_dataset("OpenAssistant/oasst1", split=split, streaming=True)
    count = 0
    for sample in dataset:
        raw_turns = sample.get("conversation") or sample.get("messages") or []
        turns: list[Turn] = []
        for message in raw_turns:
            role = str(message.get("role") or message.get("author_role") or "").lower()
            if role not in {"user", "assistant"}:
                continue
            text = normalize_text(str(message.get("content") or message.get("text") or ""))
            if text:
                turns.append({"role": role, "text": text})
        if not turns or not any(turn["role"] == "assistant" for turn in turns):
            continue
        yield {"turns": turns}
        count += 1
        if limit is not None and count >= limit:
            return


def iter_dolly(split: str, limit: int | None) -> Iterator[Conversation]:
    dataset = load_dataset("databricks/databricks-dolly-15k", split=split)
    count = 0
    for sample in dataset:
        prompt = normalize_text(str(sample.get("instruction") or ""))
        response = normalize_text(str(sample.get("response") or ""))
        if not prompt or not response:
            continue
        turns: list[Turn] = [
            {"role": "user", "text": prompt},
            {"role": "assistant", "text": response},
        ]
        yield {"turns": turns}
        count += 1
        if limit is not None and count >= limit:
            return


def resolve_split(dataset: str, split: str | None) -> str:
    if not split or split.lower() == "auto":
        defaults = {
            "oasst1": "train",
            "ultrachat": "train_sft",
            "dolly": "train",
        }
        return defaults.get(dataset, "train")
    if dataset == "ultrachat" and split == "train":
        return "train_sft"
    return split


def build_conversations(args: argparse.Namespace) -> Iterator[Conversation]:
    include = {choice.lower() for choice in args.include}
    limit = args.max_per_source
    split = args.split
    if "oasst1" in include:
        yield from iter_oasst(resolve_split("oasst1", split), limit)
    if "ultrachat" in include:
        yield from iter_ultrachat(resolve_split("ultrachat", split), limit)
    if "dolly" in include:
        yield from iter_dolly(resolve_split("dolly", split), limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT conversations")
    parser.add_argument("--out", default="data/sft/clean.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["ultrachat", "oasst1"],
        help="Datasets to include (oasst1, ultrachat, dolly)",
    )
    parser.add_argument(
        "--split",
        default="auto",
        help="Dataset split (default: auto; picks best-known split per dataset)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        help="Optional maximum number of conversations to take from each dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conversations = build_conversations(args)
    write_jsonl(Path(args.out), conversations)


if __name__ == "__main__":
    main()
