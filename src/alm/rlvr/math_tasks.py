"""Synthetic math tasks for verifiable-reward training."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MathExample:
    prompt: str
    answer: str


def _expr_add(rng: random.Random) -> tuple[str, int]:
    a = rng.randint(0, 9999)
    b = rng.randint(0, 9999)
    return f"{a} + {b}", a + b


def _expr_sub(rng: random.Random) -> tuple[str, int]:
    a = rng.randint(0, 9999)
    b = rng.randint(0, 9999)
    if b > a:
        a, b = b, a
    return f"{a} - {b}", a - b


def _expr_mul(rng: random.Random) -> tuple[str, int]:
    a = rng.randint(0, 999)
    b = rng.randint(0, 999)
    return f"{a} * {b}", a * b


def _expr_mix(rng: random.Random) -> tuple[str, int]:
    a = rng.randint(0, 999)
    b = rng.randint(0, 999)
    c = rng.randint(0, 99)
    return f"({a} + {b}) * {c}", (a + b) * c


_GENERATORS = (_expr_add, _expr_sub, _expr_mul, _expr_mix)


def generate_math_examples(
    *,
    count: int,
    seed: int = 1337,
    system: str = "You are a helpful assistant.",
) -> list[MathExample]:
    rng = random.Random(seed)
    examples: list[MathExample] = []
    for _ in range(max(0, int(count))):
        expr_fn = rng.choice(_GENERATORS)
        expr, answer = expr_fn(rng)
        prompt = f"System: {system}\nUser: Compute: {expr}\nAssistant: "
        examples.append(MathExample(prompt=prompt, answer=str(answer)))
    return examples
