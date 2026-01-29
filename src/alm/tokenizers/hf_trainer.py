"""Tokenizer trainer using Hugging Face tokenizers (Rust backend)."""

from __future__ import annotations

import random
from pathlib import Path


def _write_sampled_corpus(
    files: list[Path],
    out_path: Path,
    *,
    max_lines: int,
    sample_ratio: float | None,
    seed: int,
) -> None:
    rng = random.Random(seed)
    selected: list[str] = []
    total_lines = 0

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

                if len(selected) < max_lines:
                    selected.append(line)
                    continue

                j = rng.randint(0, total_lines - 1)
                if j < max_lines:
                    selected[j] = line

        kept = min(len(selected), max_lines)
        print(
            f"[tokenizer] scanned {total_lines:,} lines, keeping {kept:,} (file={file.name})",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for line in selected:
            handle.write(line + "\n")


def cli_train_hf_bpe(
    input_paths: list[str],
    vocab_size: int,
    output_path: str,
    *,
    max_lines: int | None = None,
    sample_ratio: float | None = None,
    min_frequency: int = 2,
    seed: int = 1337,
) -> None:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: install `tokenizers` (pip install tokenizers) "
            "or reinstall with `pip install -e '.[dev]'`."
        ) from exc

    files = [Path(path) for path in input_paths]
    train_files = files
    corpus_path: Path | None = None
    if max_lines and max_lines > 0:
        corpus_path = Path(output_path).with_name("tokenizer_corpus.txt")
        _write_sampled_corpus(
            files,
            corpus_path,
            max_lines=max_lines,
            sample_ratio=sample_ratio,
            seed=seed,
        )
        train_files = [corpus_path]
    elif sample_ratio is not None and 0.0 < sample_ratio < 1.0:
        corpus_path = Path(output_path).with_name("tokenizer_corpus.txt")
        _write_sampled_corpus(
            files,
            corpus_path,
            max_lines=500_000,
            sample_ratio=sample_ratio,
            seed=seed,
        )
        train_files = [corpus_path]

    print(
        "[tokenizer] hf bpe "
        f"vocab_size={vocab_size:,} "
        f"min_freq={min_frequency:,} "
        f"files={len(train_files)}",
        flush=True,
    )
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=max(1, int(min_frequency)),
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]"],
    )
    tokenizer.train([str(path) for path in train_files], trainer=trainer)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path), pretty=True)
