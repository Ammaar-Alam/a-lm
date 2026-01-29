"""Reward helpers for verifiable tasks."""

from __future__ import annotations

import re

_INT_RE = re.compile(r"-?\d+")


def extract_int(text: str) -> str | None:
    matches = _INT_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def exact_int_reward(completion: str, expected: str) -> float:
    answer = extract_int(completion)
    if answer is None:
        return 0.0
    return 1.0 if answer == expected else 0.0


def dense_int_reward(completion: str, expected: str) -> float:
    answer = extract_int(completion)
    if answer is None:
        return 0.0
    try:
        predicted = int(answer)
        target = int(expected)
    except ValueError:
        return 0.0
    diff = abs(predicted - target)
    score = 1.0 / (1.0 + float(diff))
    if diff == 0:
        return 1.0
    return score
