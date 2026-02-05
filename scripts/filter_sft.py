#!/usr/bin/env python3
"""Filter SFT JSONL conversations for higher-quality training.

This is a lightweight post-processing step you can run after `prepare_sft.py`
to remove low-signal / refusal-heavy conversations that can dominate small
models during SFT.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_REFUSAL_PATTERNS = [
    r"\bas an ai language model\b",
    r"\bi'?m an ai language model\b",
    r"\bi'?m sorry[, ]+(?:but|i)\b",
    r"\bi (?:do not|don't) have access\b",
    r"\bi (?:do not|don't) have (?:the )?(?:ability|capability|capacity)\b",
    r"\bi am not capable\b",
    r"\bi (?:cannot|can't) (?:access|browse)\b",
    r"\bi am not able to\b",
    r"\bi do not have the ability to\b",
    r"\bthat (?:request|content) (?:violates|goes against)\b",
    r"\bgoes against (?:policy|policies|guidelines)\b",
    r"\bi (?:cannot|can't) help with that\b",
]


def _iter_jsonl(path: Path) -> tuple[int, dict[str, Any]]:  # pragma: no cover - CLI helper
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield line_no, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter SFT conversations JSONL")
    parser.add_argument("--in", dest="in_path", required=True, help="Input JSONL path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output JSONL path")
    parser.add_argument(
        "--drop-refusals",
        action="store_true",
        help="Drop conversations containing common refusal patterns.",
    )
    parser.add_argument(
        "--drop-assistant-regex",
        action="append",
        default=[],
        help="Drop conversation if any assistant turn matches this regex (repeatable).",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Minimum total turns to keep (default: 2).",
    )
    parser.add_argument(
        "--min-assistant-chars",
        type=int,
        default=8,
        help="Minimum assistant turn length to keep (default: 8).",
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.35,
        help=(
            "Minimum alphabetic-character ratio for assistant turns (default: 0.35). "
            "Useful for dropping empty enumerations like '1.\\n2.\\n3.'"
        ),
    )
    parser.add_argument(
        "--min-assistant-words",
        type=int,
        default=3,
        help="Minimum assistant word count per turn (default: 3).",
    )
    parser.add_argument(
        "--max-user-chars",
        type=int,
        default=0,
        help="Drop conversations with any user turn longer than this (0 disables).",
    )
    parser.add_argument(
        "--max-assistant-chars",
        type=int,
        default=0,
        help="Drop conversations with any assistant turn longer than this (0 disables).",
    )
    parser.add_argument(
        "--max-user-assistant-word-ratio",
        type=float,
        default=0.0,
        help=(
            "Drop conversation if user_words / assistant_words exceeds this ratio (<=0 disables)."
        ),
    )
    parser.add_argument(
        "--drop-repetition",
        action="store_true",
        help="Drop assistant turns that are highly repetitive (ngram heuristic).",
    )
    parser.add_argument(
        "--repetition-ngram",
        type=int,
        default=4,
        help="N-gram size for repetition heuristic (default: 4).",
    )
    parser.add_argument(
        "--min-unique-ngram-ratio",
        type=float,
        default=0.35,
        help="Minimum unique n-gram ratio to keep (default: 0.35).",
    )
    parser.add_argument(
        "--min-repetition-tokens",
        type=int,
        default=64,
        help="Minimum tokens before applying repetition heuristic (default: 64).",
    )
    return parser.parse_args()


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pat.search(text) for pat in patterns)


def _alpha_ratio(text: str) -> float:
    non_ws = 0
    alpha = 0
    for ch in text:
        if ch.isspace():
            continue
        non_ws += 1
        if ch.isalpha():
            alpha += 1
    if non_ws <= 0:
        return 0.0
    return alpha / non_ws


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _unique_ngram_ratio(text: str, n: int) -> float:
    """Compute unique/total n-gram ratio over word tokens (lower = more repetitive)."""
    if n <= 0:
        return 1.0
    tokens = re.findall(r"\w+", text.lower())
    total = len(tokens) - n + 1
    if total <= 0:
        return 1.0
    ngrams = {tuple(tokens[i : i + n]) for i in range(total)}
    return len(ngrams) / total


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patterns: list[str] = list(args.drop_assistant_regex)
    if args.drop_refusals:
        patterns.extend(DEFAULT_REFUSAL_PATTERNS)
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in patterns if p.strip()]

    total = 0
    kept = 0
    dropped = 0
    dropped_refusal = 0
    dropped_short = 0
    dropped_turns = 0
    dropped_repetition = 0
    dropped_alpha = 0
    dropped_words = 0
    dropped_length = 0
    dropped_balance = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for _, convo in _iter_jsonl(in_path):
            total += 1
            turns = convo.get("turns")
            if not isinstance(turns, list) or len(turns) < args.min_turns:
                dropped += 1
                dropped_turns += 1
                continue

            assistant_turns: list[str] = []
            user_turns: list[str] = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role", "")).lower()
                text = str(turn.get("text", ""))
                if role == "user":
                    user_turns.append(text)
                    continue
                if role != "assistant":
                    continue
                assistant_turns.append(text)

            if not assistant_turns:
                dropped += 1
                dropped_turns += 1
                continue

            if any(len(t.strip()) < args.min_assistant_chars for t in assistant_turns):
                dropped += 1
                dropped_short += 1
                continue

            if args.min_assistant_words > 0 and any(
                _word_count(t) < args.min_assistant_words for t in assistant_turns
            ):
                dropped += 1
                dropped_words += 1
                continue

            if args.max_user_chars > 0 and any(
                len(t.strip()) > args.max_user_chars for t in user_turns
            ):
                dropped += 1
                dropped_length += 1
                continue

            if args.max_assistant_chars > 0 and any(
                len(t.strip()) > args.max_assistant_chars for t in assistant_turns
            ):
                dropped += 1
                dropped_length += 1
                continue

            if args.max_user_assistant_word_ratio > 0:
                user_words = sum(_word_count(t) for t in user_turns)
                assistant_words = max(1, sum(_word_count(t) for t in assistant_turns))
                if (user_words / assistant_words) > args.max_user_assistant_word_ratio:
                    dropped += 1
                    dropped_balance += 1
                    continue

            min_alpha_ratio = float(args.min_alpha_ratio)
            if min_alpha_ratio > 0.0 and any(
                _alpha_ratio(t) < min_alpha_ratio for t in assistant_turns
            ):
                dropped += 1
                dropped_alpha += 1
                continue

            if args.drop_repetition:
                n = max(1, int(args.repetition_ngram))
                threshold = float(args.min_unique_ngram_ratio)
                min_tokens = max(0, int(args.min_repetition_tokens))
                is_repetitive = any(
                    len(re.findall(r"\w+", t)) >= min_tokens
                    and _unique_ngram_ratio(t, n) < threshold
                    for t in assistant_turns
                )
                if is_repetitive:
                    dropped += 1
                    dropped_repetition += 1
                    continue

            if compiled and any(_matches_any(t, compiled) for t in assistant_turns):
                dropped += 1
                dropped_refusal += 1
                continue

            handle.write(json.dumps(convo, ensure_ascii=False) + "\n")
            kept += 1

    print(
        json.dumps(
            {
                "in": str(in_path),
                "out": str(out_path),
                "total": total,
                "kept": kept,
                "dropped": dropped,
                "dropped_refusal": dropped_refusal,
                "dropped_short": dropped_short,
                "dropped_turns": dropped_turns,
                "dropped_repetition": dropped_repetition,
                "dropped_alpha": dropped_alpha,
                "dropped_words": dropped_words,
                "dropped_length": dropped_length,
                "dropped_balance": dropped_balance,
                "patterns": patterns,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
