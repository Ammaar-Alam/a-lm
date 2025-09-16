"""Dataset preparation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from datasets import load_dataset

from alm.tokenizers.normalizer import normalize_text

from .config import CorpusConfig, SourceConfig


def iter_huggingface_source(cfg: SourceConfig, cache_dir: Optional[str]) -> Iterator[str]:
    kwargs = {}
    if cfg.config:
        kwargs["name"] = cfg.config
    dataset = load_dataset(
        cfg.dataset,
        split=cfg.split,
        streaming=cfg.streaming,
        cache_dir=cache_dir,
        **kwargs,
    )
    count_tokens = 0
    count_entries = 0
    for sample in dataset:
        text = extract_text(sample)
        if text is None:
            continue
        yield text
        count_entries += 1
        count_tokens += len(text.split())
        if cfg.sample_articles and count_entries >= cfg.sample_articles:
            break
        if cfg.sample_tokens and count_tokens >= cfg.sample_tokens:
            break


def iter_local_file(path: Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n")


def extract_text(sample: dict) -> Optional[str]:
    if "text" in sample:
        return sample["text"]
    if "content" in sample:
        return sample["content"]
    if "body" in sample:
        return sample["body"]
    for value in sample.values():
        if isinstance(value, str):
            return value
    return None


def prepare_source(cfg: SourceConfig, out_dir: Path, cache_dir: Optional[str] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{cfg.name}.txt"
    if cfg.kind == "huggingface":
        iterator = iter_huggingface_source(cfg, cache_dir)
    elif cfg.kind == "local":
        iterator = iter_local_file(Path(cfg.path))
    else:
        raise ValueError(f"Unsupported source kind: {cfg.kind}")

    total_lines = 0
    total_chars = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for line in iterator:
            cleaned = normalize_text(line)
            if not cleaned:
                continue
            writer.write(cleaned + "\n")
            total_lines += 1
            total_chars += len(cleaned)

    metadata = {
        "name": cfg.name,
        "kind": cfg.kind,
        "lines": total_lines,
        "chars": total_chars,
    }
    (out_dir / f"{cfg.name}.json").write_text(json.dumps(metadata, indent=2))
    return output_path


def prepare_all(config: CorpusConfig, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in config.sources:
        prepare_source(source, out_dir, cache_dir=config.cache_dir)
