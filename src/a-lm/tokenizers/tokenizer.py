"""Tokenizer utilities built around the trained vocabulary."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .normalizer import normalize_text
from .vocab import Vocabulary
from .bpe_trainer import load_vocab


class Tokenizer:
    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab
        # Greedy longest-match decode order
        self._sorted_tokens = sorted(
            self.vocab.id_to_token,
            key=lambda tok: len(tok),
            reverse=True,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        normalized = normalize_text(text)
        ids: List[int] = []
        i = 0
        length = len(normalized)
        while i < length:
            match = None
            for token in self._sorted_tokens:
                if normalized.startswith(token, i):
                    match = token
                    break
            if match is None or match == "":
                match = normalized[i]
            ids.append(self.vocab.encode(match))
            i += len(match)
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.vocab.decode(i) for i in ids)

    @classmethod
    def from_file(cls, path: Path) -> "Tokenizer":
        vocab = load_vocab(path)
        return cls(vocab)
