"""Utilities for packing SFT conversations into mmap shards."""

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

try:  # optional progress display
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
except ModuleNotFoundError:  # pragma: no cover - rich optional
    Progress = None  # type: ignore

_PROMPT_TOKENS = {
    "system": "System: {text}\n",
    "user": "User: {text}\nAssistant: ",
    "assistant": "{text}\n",
}

_WORKER_TOKENIZER: Any | None = None


def _worker_init(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_file(Path(tokenizer_path))


def _conversation_to_segments(conversation: dict[str, Any]) -> list[tuple[str, int]]:
    segments: list[tuple[str, int]] = []
    system = conversation.get("system")
    if system:
        segments.append((_PROMPT_TOKENS["system"].format(text=system), 0))
    for turn in conversation.get("turns", []):
        role = str(turn.get("role", "")).lower()
        text = str(turn.get("text", ""))
        if not text:
            continue
        if role == "user":
            segments.append((_PROMPT_TOKENS["user"].format(text=text), 0))
        elif role == "assistant":
            segments.append((_PROMPT_TOKENS["assistant"].format(text=text), 1))
    return segments


def _encode_segments(segments: list[tuple[str, int]]) -> tuple[list[int], list[int]]:
    if _WORKER_TOKENIZER is None:  # pragma: no cover
        raise RuntimeError("worker tokenizer not initialised")
    ids: list[int] = []
    mask: list[int] = []
    for text, label in segments:
        encoded = _WORKER_TOKENIZER.encode(text)
        if not encoded:
            continue
        ids.extend(encoded)
        mask.extend([label] * len(encoded))
    return ids, mask


def _encode_conversation(conversation: dict[str, Any]) -> tuple[list[int], list[int]]:
    segments = _conversation_to_segments(conversation)
    if not segments:
        return [], []
    ids, mask = _encode_segments(segments)
    if not ids or not any(mask):
        return [], []
    return ids, mask


def _iter_jsonl(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def pack_sft(
    tokenizer: Tokenizer,
    conversations: Iterable[dict[str, Any]],
    *,
    seq_len: int,
    shard_size: int,
    out_dir: Path,
    show_progress: bool = False,
    workers: int | None = None,
    chunk_size: int = 64,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    shard_idx = 0
    token_buffer: list[int] = []
    mask_buffer: list[int] = []
    input_paths: list[str] = []
    mask_paths: list[str] = []
    token_dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32

    if workers is None:
        workers = min(os.cpu_count() or 1, 6)
    workers = max(1, workers)
    chunk_size = max(1, chunk_size)

    progress = None
    task_id = None
    start = time.perf_counter()
    if show_progress and Progress is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            TextColumn("tok={task.fields[tok]:,}"),
            TextColumn("tok/s={task.fields[rate]:.0f}"),
            transient=False,
        )
        progress.start()
        task_id = progress.add_task("pack sft", total=None, tok=0, rate=0.0)

    def update_progress() -> None:
        if progress and task_id is not None:
            elapsed = max(time.perf_counter() - start, 1e-6)
            progress.update(task_id, tok=total_tokens, rate=total_tokens / elapsed)

    def flush() -> None:
        nonlocal token_buffer, mask_buffer, shard_idx
        if not token_buffer:
            return
        inputs = np.array(token_buffer, dtype=token_dtype)
        masks = np.array(mask_buffer, dtype=np.uint8)
        input_path = out_dir / f"inputs_{shard_idx:05d}.bin"
        mask_path = out_dir / f"mask_{shard_idx:05d}.bin"
        inputs.tofile(input_path)
        masks.tofile(mask_path)
        input_paths.append(input_path.name)
        mask_paths.append(mask_path.name)
        token_buffer = []
        mask_buffer = []
        shard_idx += 1
        update_progress()

    def add_sequence(ids: list[int], mask: list[int]) -> None:
        nonlocal token_buffer, mask_buffer, total_tokens
        if not ids or not any(mask):
            return
        start_idx = 0
        while start_idx < len(ids):
            end_idx = min(start_idx + seq_len, len(ids))
            token_buffer.extend(ids[start_idx:end_idx])
            mask_buffer.extend(mask[start_idx:end_idx])
            total_tokens += end_idx - start_idx
            start_idx = end_idx
            if len(token_buffer) >= shard_size:
                flush()

    def chunk_iterator(iterator: Iterator[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        for item in iterator:
            batch.append(item)
            if len(batch) >= chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch

    iterator = iter(conversations)
    if workers > 1:
        if tokenizer_path is None:
            raise ValueError("tokenizer_path required when workers > 1")
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers, initializer=_worker_init, initargs=(str(tokenizer_path),)
        ) as pool:
            for batch in chunk_iterator(iterator):
                for ids, mask in pool.imap(_encode_conversation, batch, chunksize=1):
                    if ids:
                        add_sequence(ids, mask)
    else:
        global _WORKER_TOKENIZER
        _WORKER_TOKENIZER = tokenizer
        for conversation in iterator:
            ids, mask = _encode_conversation(conversation)
            if ids:
                add_sequence(ids, mask)

    if token_buffer:
        flush()
    if progress:
        progress.stop()
    if not input_paths:
        raise ValueError("No sequences encoded; ensure conversations include assistant replies")

    metadata = {
        "seq_len": seq_len,
        "shard_size": shard_size,
        "total_tokens": total_tokens,
        "inputs": input_paths,
        "masks": mask_paths,
        "dtype": np.dtype(token_dtype).name,
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def iter_conversations(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    return _iter_jsonl(paths)


__all__ = ["pack_sft", "iter_conversations"]
