"""Dataclasses and loader for corpus configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class SourceConfig:
    name: str
    kind: str
    dataset: Optional[str] = None
    split: str = "train"
    config: Optional[str] = None
    streaming: bool = False
    sample_tokens: Optional[int] = None
    sample_articles: Optional[int] = None
    path: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class CorpusConfig:
    sources: List[SourceConfig] = field(default_factory=list)
    cache_dir: Optional[str] = None


def _expand_cache_dir(value: Optional[str]) -> Optional[str]:
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
    sources_cfg: Dict[str, dict] = data.get("sources", {})
    sources = [
        SourceConfig(name=name, **cfg)
        for name, cfg in sources_cfg.items()
    ]
    cache_section = data.get("cache", {})
    cache_dir = _expand_cache_dir(cache_section.get("base_dir"))
    return CorpusConfig(sources=sources, cache_dir=cache_dir)
