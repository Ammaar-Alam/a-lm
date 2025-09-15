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
                if normalized.startswith(token, i) and token in self.vocab.token_to_id:
                    match = token
                    break

            if match is None:
                char = normalized[i]
                for b in char.encode("utf-8"):
                    ids.append(self.vocab.encode(chr(b)))
                i += 1
            else:
                ids.append(self.vocab.encode(match))
                i += len(match)
        return ids

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        byte_buffer = bytearray()

        for idx in ids:
            token = self.vocab.decode(idx)
            if len(token) == 1 and ord(token) < 256:
                byte_buffer.append(ord(token))
            else:
                if byte_buffer:
                    parts.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                parts.append(token)

        if byte_buffer:
            parts.append(byte_buffer.decode("utf-8", errors="replace"))

        return "".join(parts)

    @classmethod
    def from_file(cls, path: Path) -> Tokenizer:
        vocab = load_vocab(path)
        return cls(vocab)
