"""Byte-level normalization helpers used by the tokenizer."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

WHITESPACE_REPLACEMENTS = {
    "\u00A0": " ",  # non-breaking space
    "\u1680": " ",
    "\u2000": " ",
    "\u2001": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2004": " ",
    "\u2005": " ",
    "\u2006": " ",
    "\u2007": " ",
    "\u2008": " ",
    "\u2009": " ",
    "\u200A": " ",
    "\u202F": " ",
    "\u205F": " ",
    "\u3000": " ",
}

CONTROL_CHARS = {chr(i) for i in range(0, 32)} - {"\n", "\t"}
CONTROL_CHARS.add(chr(127))


def normalize_text(text: str) -> str:
    """Normalize Unicode text into a consistent byte-friendly form."""

    text = unicodedata.normalize("NFKC", text)
    text = text.translate({ord(k): v for k, v in WHITESPACE_REPLACEMENTS.items()})
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch not in CONTROL_CHARS)
    return text


def iter_bytes(text: str) -> bytes:
    """Encode text to UTF-8 bytes after normalization."""

    return normalize_text(text).encode("utf-8")


@dataclass(frozen=True)
class Symbol:
    """Container mapping between byte values and symbolic strings."""

    value: bytes

    def __str__(self) -> str:  # pragma: no cover - used rarely
        return self.value.decode("utf-8", errors="replace")
