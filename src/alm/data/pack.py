"""Token packing utilities."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alm.tokenizers.tokenizer import Tokenizer
from alm.tokenizers.vocab import Vocabulary

try:  # pragma: no cover - optional dependency
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
except ModuleNotFoundError:  # pragma: no cover - rich optional
    Progress = None


def iter_text_files(paths: Sequence[Path]) -> Iterator[str]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                yield line.rstrip("\n")


_WORKER_TOKENIZER: Tokenizer | None = None


def _worker_init(tokens: list[str]) -> None:
    vocab = Vocabulary()
    vocab.extend(tokens)
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer(vocab)


def _worker_encode(chunk: list[str]) -> list[list[int]]:
    if _WORKER_TOKENIZER is None:  # pragma: no cover - sanity guard
        raise RuntimeError("Worker tokenizer not initialised")
    return _WORKER_TOKENIZER.encode_batch(chunk)


def _chunk_iterable(iterable: Iterable[str], chunk_size: int) -> Iterator[list[str]]:
    chunk: list[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def pack_tokens(
    tokenizer: Tokenizer,
    texts: Iterable[str],
    seq_len: int,
    shard_size: int,
    out_dir: Path,
    eos_token: str = "\n",
    show_progress: bool = False,
    workers: int | None = None,
    chunk_size: int = 512,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    eos_id = tokenizer.encode(eos_token)[-1]
    buffer: list[int] = []
    shard: list[int] = []
    shard_idx = 0
    total_tokens = 0
    shard_paths: list[str] = []

    if show_progress and Progress is None:
        print("Progress disabled: install 'rich' to enable live stats.")
        show_progress = False

    if workers is None:
        cpu_count = os.cpu_count() or 1
        workers = min(cpu_count, 6)
    workers = max(1, workers)
    chunk_size = max(1, chunk_size)

    progress: Progress | None = None
    task_id: int | None = None
    start_time = time.perf_counter()

    if show_progress and Progress is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            TextColumn("tok={task.fields[tokens]:,}"),
            TextColumn("tok/s={task.fields[rate]:.0f}"),
            TextColumn("shards={task.fields[shards]}"),
            refresh_per_second=5,
            transient=False,
        )
        progress.start()
        task_id = progress.add_task(
            "packing",
            total=None,
            tokens=0,
            rate=0.0,
            shards=0,
        )

    def update_progress() -> None:
        if progress and task_id is not None:
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            rate = total_tokens / elapsed
            progress.update(
                task_id,
                tokens=total_tokens,
                rate=rate,
                shards=shard_idx,
            )

    def flush_shard() -> None:
        nonlocal shard, shard_idx
        if not shard:
            return
        arr = np.array(shard, dtype=np.uint32)
        shard_path = out_dir / f"shard_{shard_idx:05d}.bin"
        arr.tofile(shard_path)
        shard_paths.append(str(shard_path.name))
        shard_idx += 1
        shard = []
        update_progress()

    text_iter = iter(texts)

    try:
        if workers <= 1:
            for chunk in _chunk_iterable(text_iter, chunk_size):
                for token_ids in tokenizer.encode_batch(chunk):
                    buffer.extend(token_ids)
                    buffer.append(eos_id)
                    while len(buffer) >= seq_len:
                        shard.extend(buffer[:seq_len])
                        buffer = buffer[seq_len:]
                        total_tokens += seq_len
                        if len(shard) >= shard_size:
                            flush_shard()
        else:
            tokens_snapshot = list(tokenizer.vocab.id_to_token)
            ctx = mp.get_context("spawn")
            chunk_iter = _chunk_iterable(text_iter, chunk_size)
            with ctx.Pool(
                processes=workers,
                initializer=_worker_init,
                initargs=(tokens_snapshot,),
            ) as pool:
                for encoded_chunk in pool.imap(_worker_encode, chunk_iter, chunksize=1):
                    for token_ids in encoded_chunk:
                        buffer.extend(token_ids)
                        buffer.append(eos_id)
                        while len(buffer) >= seq_len:
                            shard.extend(buffer[:seq_len])
                            buffer = buffer[seq_len:]
                            total_tokens += seq_len
                            if len(shard) >= shard_size:
                                flush_shard()
        if buffer:
            shard.extend(buffer)
            total_tokens += len(buffer)
        if shard:
            flush_shard()
        update_progress()
    finally:
        if progress:
            progress.stop()

    metadata = {
        "seq_len": seq_len,
        "shard_size": shard_size,
        "total_tokens": total_tokens,
        "shards": shard_paths,
        "workers": workers,
        "chunk_size": chunk_size,
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
