"""Vocabulary utilities for byte fallback tokenizers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

BYTE_SIZE = 256


@dataclass
class Token:
    text: str
    score: float = 0.0


class Vocabulary:
    """Simple vocabulary supporting byte fallback."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []

    def add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def extend(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.add(token)

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, token: str) -> int:
        return self.token_to_id[token]

    def decode(self, idx: int) -> str:
        return self.id_to_token[idx]

    @classmethod
    def byte_fallback(cls) -> Vocabulary:
        vocab = cls()
        vocab.extend([chr(i) for i in range(BYTE_SIZE)])
        return vocab

    def to_list(self) -> list[Token]:
        return [Token(text) for text in self.id_to_token]

    def from_pairs(self, pairs: Iterable[tuple[str, float]]) -> None:
        for token, _score in pairs:
            self.add(token)
