"""Dataclasses and loader for corpus configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SourceConfig:
    name: str
    kind: str
    dataset: str | None = None
    split: str = "train"
    config: str | None = None
    data_files: str | list[str] | None = None
    streaming: bool = False
    # Optional list of columns to request from the dataset loader.
    # This is especially useful for streaming Parquet-backed datasets where
    # some shards may have schema drift across non-essential columns.
    columns: list[str] | None = None
    sample_tokens: int | None = None
    sample_articles: int | None = None
    path: str | None = None
    adapter: str | None = None
    filter_underage: bool = False
    notes: str | None = None


@dataclass
class CorpusConfig:
    sources: list[SourceConfig] = field(default_factory=list)
    cache_dir: str | None = None


def _expand_cache_dir(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("${") and value.endswith("}"):
        body = value[2:-1]
        if ":-" in body:
            var, default = body.split(":-", 1)
            return os.environ.get(var, default)
        return os.environ.get(body, "")
    return os.path.expanduser(os.path.expandvars(value))


def load_corpus_config(path: Path) -> CorpusConfig:
    data = yaml.safe_load(Path(path).read_text())
    sources_cfg: dict[str, dict] = data.get("sources", {})
    sources = [SourceConfig(name=name, **cfg) for name, cfg in sources_cfg.items()]
    cache_section = data.get("cache", {})
    cache_dir = _expand_cache_dir(cache_section.get("base_dir"))
    return CorpusConfig(sources=sources, cache_dir=cache_dir)
