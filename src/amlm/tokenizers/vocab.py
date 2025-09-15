"""Vocabulary utilities for byte fallback tokenizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

BYTE_SIZE = 256


@dataclass
class Token:
    text: str
    score: float = 0.0


class Vocabulary:
    """Simple vocabulary supporting byte fallback."""

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []

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

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.id_to_token)

    def encode(self, token: str) -> int:
        return self.token_to_id[token]

    def decode(self, idx: int) -> str:
        return self.id_to_token[idx]

    @classmethod
    def byte_fallback(cls) -> "Vocabulary":
        vocab = cls()
        vocab.extend([chr(i) for i in range(BYTE_SIZE)])
        return vocab

    def to_list(self) -> List[Token]:
        return [Token(text) for text in self.id_to_token]

    def from_pairs(self, pairs: Iterable[Tuple[str, float]]) -> None:
        for token, score in pairs:
            self.add(token)
