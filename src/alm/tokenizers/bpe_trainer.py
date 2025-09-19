"""Byte pair encoding trainer with byte fallback."""

from __future__ import annotations

import collections
import json
import random
from collections.abc import Iterable, Sequence
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


def train_bpe(
    corpus: Iterable[str],
    vocab_size: int,
    *,
    log_interval: int | None = None,
) -> Vocabulary:
    vocab = Vocabulary.byte_fallback()
    tokens = [[ch for ch in normalize_text(line)] for line in corpus]
    if not tokens:
        return vocab

    base_size = len(vocab)
    merges_target = max(vocab_size - base_size, 0)
    if log_interval and log_interval > 0:
        print(
            f"[tokenizer] starting BPE merges: target={merges_target:,} vocab_size={vocab_size}",
            flush=True,
        )

    while len(vocab) < vocab_size:
        stats = get_stats(tokens)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        merged_token = "".join(best_pair)
        vocab.add(merged_token)
        tokens = merge_vocab(tokens, best_pair)

        merges_done = len(vocab) - base_size
        if (
            log_interval
            and log_interval > 0
            and (merges_done == 1 or merges_done % log_interval == 0 or len(vocab) == vocab_size)
        ):
            print(
                "[tokenizer] merges="
                f"{merges_done:,}/{merges_target:,} "
                f"pair={best_pair[0]!r}+{best_pair[1]!r} "
                f"freq={stats[best_pair]:,}",
                flush=True,
            )

    if log_interval and log_interval > 0:
        print(
            f"[tokenizer] finished training vocab of size {len(vocab):,}",
            flush=True,
        )
    return vocab


def save_vocab(vocab: Vocabulary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"tokens": vocab.id_to_token}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_vocab(path: Path) -> Vocabulary:
    data = json.loads(path.read_text())
    vocab = Vocabulary()
    vocab.extend(data["tokens"])
    return vocab


def train_from_files(
    files: Sequence[Path],
    vocab_size: int,
    *,
    max_lines: int | None = None,
    sample_ratio: float | None = None,
    log_interval: int | None = None,
    seed: int = 1337,
) -> Vocabulary:
    rng = random.Random(seed)
    selected: list[str] = []
    total_lines = 0
    cap = max_lines if max_lines and max_lines > 0 else None

    for file in files:
        with Path(file).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                total_lines += 1

                if (
                    sample_ratio is not None
                    and 0.0 < sample_ratio < 1.0
                    and rng.random() > sample_ratio
                ):
                    continue

                if cap is None:
                    selected.append(line)
                else:
                    if len(selected) < cap:
                        selected.append(line)
                    else:
                        j = rng.randint(0, total_lines - 1)
                        if j < cap:
                            selected[j] = line

        kept = len(selected) if cap is None else min(len(selected), cap)
        print(
            f"[tokenizer] scanned {total_lines:,} lines, keeping {kept:,} (file={file.name})",
            flush=True,
        )

        if cap is not None and len(selected) >= cap:
            continue

    if not selected:
        print("[tokenizer] no lines selected; emitting byte fallback vocabulary", flush=True)
        return Vocabulary.byte_fallback()

    return train_bpe(selected, vocab_size, log_interval=log_interval)


def cli_train_tokenizer(
    input_paths: list[str],
    vocab_size: int,
    output_path: str,
    *,
    max_lines: int | None = None,
    sample_ratio: float | None = None,
    log_interval: int | None = None,
    seed: int = 1337,
) -> None:
    files = [Path(p) for p in input_paths]
    vocab = train_from_files(
        files,
        vocab_size,
        max_lines=max_lines,
        sample_ratio=sample_ratio,
        log_interval=log_interval,
        seed=seed,
    )
    save_vocab(vocab, Path(output_path))
