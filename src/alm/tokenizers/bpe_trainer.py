"""Byte pair encoding trainer with byte fallback."""

from __future__ import annotations

import collections
import json
from collections.abc import Iterable
from pathlib import Path

from .normalizer import normalize_text
from .vocab import Vocabulary

Pair = tuple[str, str]


def get_stats(tokens: Iterable[list[str]]) -> dict[Pair, int]:
    stats: dict[Pair, int] = collections.Counter()
    for token_list in tokens:
        for pair in zip(token_list, token_list[1:]):
            stats[pair] += 1
    return dict(stats)


def merge_vocab(tokens: Iterable[list[str]], pair: Pair) -> list[list[str]]:
    first, second = pair
    merged: list[list[str]] = []
    bigram = first + second
    for token_list in tokens:
        new_list: list[str] = []
        i = 0
        while i < len(token_list):
            if i < len(token_list) - 1 and token_list[i] == first and token_list[i + 1] == second:
                new_list.append(bigram)
                i += 2
            else:
                new_list.append(token_list[i])
                i += 1
        merged.append(new_list)
    return merged


def train_bpe(corpus: Iterable[str], vocab_size: int) -> Vocabulary:
    vocab = Vocabulary.byte_fallback()
    tokens = [[ch for ch in normalize_text(word)] for line in corpus for word in line.split()]
    if not tokens:
        return vocab

    while len(vocab) < vocab_size:
        stats = get_stats(tokens)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        merged_token = "".join(best_pair)
        vocab.add(merged_token)
        tokens = merge_vocab(tokens, best_pair)
    return vocab


def save_vocab(vocab: Vocabulary, path: Path) -> None:
    data = {"tokens": vocab.id_to_token}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_vocab(path: Path) -> Vocabulary:
    data = json.loads(path.read_text())
    vocab = Vocabulary()
    vocab.extend(data["tokens"])
    return vocab


def train_from_files(files: list[Path], vocab_size: int) -> Vocabulary:
    def iter_corpus() -> Iterable[str]:
        for file in files:
            yield from Path(file).read_text(encoding="utf-8").splitlines()

    return train_bpe(iter_corpus(), vocab_size)


def cli_train_tokenizer(input_paths: list[str], vocab_size: int, output_path: str) -> None:
    files = [Path(p) for p in input_paths]
    vocab = train_from_files(files, vocab_size)
    save_vocab(vocab, Path(output_path))
