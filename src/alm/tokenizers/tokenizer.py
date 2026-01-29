"""Tokenizer utilities built around the trained vocabulary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from .normalizer import normalize_text
from .vocab import Vocabulary


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.token_id: int | None = None


class Tokenizer:
    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab
        self._root = _TrieNode()
        self._byte_ids: dict[str, int] = {}
        for i in range(256):
            char = chr(i)
            token_id = self.vocab.token_to_id.get(char)
            if token_id is not None:
                self._byte_ids[char] = token_id
        for token, idx in self.vocab.token_to_id.items():
            self._insert(token, idx)

    def _insert(self, token: str, token_id: int) -> None:
        node = self._root
        for char in token:
            node = node.children.setdefault(char, _TrieNode())
        node.token_id = token_id

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def fingerprint(self) -> str:
        joined = "\n".join(self.vocab.id_to_token).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()

    def encode(self, text: str) -> list[int]:
        normalized = normalize_text(text)
        ids: list[int] = []
        i = 0
        length = len(normalized)
        append = ids.append
        vocab_encode = self.vocab.encode
        byte_ids = self._byte_ids
        while i < length:
            node = self._root
            best_id: int | None = None
            best_pos = i
            j = i
            while j < length:
                next_char = normalized[j]
                child = node.children.get(next_char)
                if child is None:
                    break
                node = child
                j += 1
                if node.token_id is not None:
                    best_id = node.token_id
                    best_pos = j
            if best_id is not None:
                append(best_id)
                i = best_pos
                continue
            char = normalized[i]
            byte_token_id = byte_ids.get(char)
            if byte_token_id is not None:
                append(byte_token_id)
            else:
                for byte in char.encode("utf-8"):
                    append(vocab_encode(chr(byte)))
            i += 1
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

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    @classmethod
    def from_file(cls, path: Path):  # type: ignore[override]
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("tokens"), list):
            vocab = Vocabulary()
            vocab.extend(data["tokens"])
            return cls(vocab)
        from .hf_tokenizer import HFTokenizer

        return HFTokenizer.from_file(path)
