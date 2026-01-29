"""Hugging Face tokenizers wrapper used for large corpus runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from .normalizer import normalize_text

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from tokenizers import Tokenizer as HFTokenizerImpl


class HFTokenizer:
    def __init__(self, tokenizer: HFTokenizerImpl, fingerprint: str) -> None:
        self._tokenizer = tokenizer
        self._fingerprint = fingerprint

    @classmethod
    def from_file(cls, path: Path) -> HFTokenizer:
        raw = Path(path).read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        try:
            import tokenizers  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: install `tokenizers` (pip install tokenizers) "
                "or reinstall with `pip install -e '.[dev]'`."
            ) from exc
        tokenizer = tokenizers.Tokenizer.from_file(str(path))
        return cls(tokenizer, fingerprint)

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size(with_added_tokens=True))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def encode(self, text: str) -> list[int]:
        encoding = self._tokenizer.encode(normalize_text(text))
        return list(encoding.ids)

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        normalized = [normalize_text(text) for text in texts]
        encodings = self._tokenizer.encode_batch(normalized)
        return [list(enc.ids) for enc in encodings]

    def decode(self, ids: list[int]) -> str:
        return str(self._tokenizer.decode(ids))
