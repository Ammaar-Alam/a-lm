"""Unigram language model tokenizer trainer."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

from .normalizer import normalize_text
from .vocab import Vocabulary

Token = tuple[str, float]


def initialize_vocab(corpus: Iterable[str]) -> list[Token]:
    freq: dict[str, int] = {}
    for line in corpus:
        line = normalize_text(line)
        for word in line.split():
            freq[word] = freq.get(word, 0) + 1
    total = sum(freq.values()) or 1
    return [(word, math.log(count / total)) for word, count in freq.items()]


def prune_vocab(candidates: list[Token], target_size: int) -> list[Token]:
    candidates.sort(key=lambda t: t[1], reverse=True)
    return candidates[:target_size]


def train_unigram(corpus: Iterable[str], vocab_size: int) -> Vocabulary:
    vocab = Vocabulary.byte_fallback()
    candidates = initialize_vocab(corpus)
    if not candidates:
        return vocab
    candidates = prune_vocab(candidates, vocab_size - len(vocab))
    vocab.from_pairs(candidates)
    return vocab


def train_from_files(files: list[Path], vocab_size: int) -> Vocabulary:
    def iter_corpus() -> Iterable[str]:
        for file in files:
            yield from Path(file).read_text(encoding="utf-8").splitlines()

    return train_unigram(iter_corpus(), vocab_size)


def save_vocab(vocab: Vocabulary, path: Path) -> None:
    data = {"tokens": vocab.id_to_token}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def cli_train(input_paths: list[str], vocab_size: int, output_path: str) -> None:
    files = [Path(p) for p in input_paths]
    vocab = train_from_files(files, vocab_size)
    save_vocab(vocab, Path(output_path))
