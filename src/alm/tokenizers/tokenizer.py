"""Tokenizer utilities built around the trained vocabulary."""

from __future__ import annotations

from pathlib import Path

from .bpe_trainer import load_vocab
from .normalizer import normalize_text
from .vocab import Vocabulary


class Tokenizer:
    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab
        self._sorted_tokens = sorted(
            self.vocab.id_to_token,
            key=len,
            reverse=True,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        normalized = normalize_text(text)
        ids: list[int] = []
        i = 0
        length = len(normalized)
        while i < length:
            match = None
            for token in self._sorted_tokens:
                if normalized.startswith(token, i):
                    match = token
                    break
            if not match:
                match = normalized[i]
            ids.append(self.vocab.encode(match))
            i += len(match)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab.decode(i) for i in ids)

    @classmethod
    def from_file(cls, path: Path) -> Tokenizer:
        vocab = load_vocab(path)
        return cls(vocab)
