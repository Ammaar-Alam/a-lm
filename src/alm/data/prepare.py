"""Dataset preparation utilities."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset

from alm.tokenizers.normalizer import normalize_text

from .config import CorpusConfig, SourceConfig

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# This is a deliberately conservative filter to avoid training on any content that
# might be sexual content involving minors. It is not perfect; it's a best-effort
# safety net for datasets that may contain unsafe samples.
_UNDERAGE_TERMS = (
    "loli",
    "lolicon",
    "shotacon",
    "preteen",
    "underage",
)


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


def _contains_underage_term(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _UNDERAGE_TERMS)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_role(value: str) -> str:
    role = value.strip().lower()
    if role in {"user", "human", "customer", "client", "speaker_1", "speaker1"}:
        return "user"
    if role in {"assistant", "bot", "gpt", "agent", "speaker_2", "speaker2"}:
        return "assistant"
    return role


def _extract_messages_generic(sample: dict[str, Any]) -> list[tuple[str, str]]:
    raw = (
        sample.get("messages")
        or sample.get("conversation")
        or sample.get("conversations")
        or sample.get("dialogue")
        or sample.get("turns")
        or []
    )
    if not isinstance(raw, list):
        return []
    messages: list[tuple[str, str]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from") or msg.get("speaker") or msg.get("author_role")
        if role is None and "is_human" in msg:
            role = "user" if bool(msg.get("is_human")) else "assistant"
        if role is None and "isHuman" in msg:
            role = "user" if bool(msg.get("isHuman")) else "assistant"
        role_norm = _normalize_role(_as_text(role))
        if role_norm not in {"user", "assistant"}:
            continue

        text = (
            msg.get("content")
            or msg.get("text")
            or msg.get("message")
            or msg.get("value")
            or msg.get("prompt")
            or ""
        )
        text_str = _strip_think_tags(_as_text(text)).strip()
        if not text_str:
            continue
        messages.append((role_norm, text_str))
    return messages


def _format_messages_as_text(
    messages: list[tuple[str, str]],
    *,
    system: str | None = None,
) -> list[str]:
    if not messages or not any(role == "assistant" for role, _ in messages):
        return []
    rendered: list[str] = []
    if system:
        system = system.strip()
        if system:
            rendered.append(f"System: {system}")
    for role, text in messages:
        if role == "user":
            rendered.append(f"User: {text} Assistant:")
        else:
            rendered.append(text)
    return rendered


def _extract_pippa(sample: dict[str, Any]) -> list[str]:
    system = sample.get("system") or sample.get("system_prompt")
    raw = sample.get("conversation") or sample.get("conversations") or sample.get("messages") or []
    if not isinstance(raw, list):
        return []
    messages: list[tuple[str, str]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = "user" if bool(msg.get("is_human")) else "assistant" if "is_human" in msg else None
        if role is None:
            role = msg.get("role") or msg.get("from")
        role_norm = _normalize_role(_as_text(role))
        if role_norm not in {"user", "assistant"}:
            continue
        text = msg.get("message") or msg.get("text") or msg.get("content") or ""
        text_str = _strip_think_tags(_as_text(text)).strip()
        if not text_str:
            continue
        messages.append((role_norm, text_str))
    return _format_messages_as_text(messages, system=_as_text(system) if system else None)


def _extract_rp_opus(sample: dict[str, Any]) -> list[str]:
    # rp-opus is shipped as JSONL; schemas vary. Prefer a messages-style parse, then
    # fall back to prompt/response style, then finally any single text field.
    system = sample.get("system") or sample.get("system_prompt")

    messages = _extract_messages_generic(sample)
    if messages:
        return _format_messages_as_text(messages, system=_as_text(system) if system else None)

    prompt = sample.get("prompt") or sample.get("instruction") or sample.get("input")
    response = sample.get("response") or sample.get("output") or sample.get("text")
    if prompt and response:
        pair = [("user", _as_text(prompt)), ("assistant", _as_text(response))]
        return _format_messages_as_text(pair, system=_as_text(system) if system else None)

    fallback = extract_text(sample)
    if fallback:
        return [_strip_think_tags(fallback)]
    return []


def extract_texts(sample: dict[str, Any], cfg: SourceConfig) -> list[str]:
    adapter = (cfg.adapter or "").strip().lower()
    if adapter == "pippa":
        texts = _extract_pippa(sample)
    elif adapter in {"rp_opus", "rp-opus", "rpopus"}:
        texts = _extract_rp_opus(sample)
    else:
        text = extract_text(sample)
        texts = [_strip_think_tags(text)] if text else []

    if cfg.filter_underage and any(_contains_underage_term(text) for text in texts):
        return []
    return [text for text in texts if text.strip()]


def iter_huggingface_source(cfg: SourceConfig, cache_dir: str | None) -> Iterator[str]:
    kwargs: dict = {}
    if cfg.config:
        kwargs["name"] = cfg.config
    if cfg.data_files:
        kwargs["data_files"] = cfg.data_files

    # For Parquet-backed streaming datasets, requesting only the needed columns
    # avoids hard failures when some shards have schema drift on non-essential
    # fields (e.g. a missing "date" column in certain FineWeb shards).
    columns = cfg.columns
    # Only apply the default "text" projection when we're in the simplest case:
    # a plain text dataset with no special adapter. Chat-style datasets (PIPPA,
    # rp-opus, etc.) typically don't have a `text` column.
    if columns is None and cfg.streaming and not (cfg.adapter or "").strip():
        columns = ["text"]

    def _load(with_columns: list[str] | None):
        base_kwargs = dict(
            split=cfg.split,
            streaming=cfg.streaming,
            cache_dir=cache_dir,
            **kwargs,
        )
        if with_columns:
            # 'columns' is supported for Parquet-backed streaming datasets.
            return load_dataset(cfg.dataset, columns=with_columns, **base_kwargs)
        return load_dataset(cfg.dataset, **base_kwargs)

    try:
        dataset = _load(columns)
    except TypeError:
        # Older datasets versions may not accept 'columns' in load_dataset().
        dataset = _load(None)
    except Exception:
        # If 'columns' selection is rejected for this dataset, fall back to
        # default behavior.
        if columns:
            dataset = _load(None)
        else:
            raise
    count_tokens = 0
    count_entries = 0
    for sample in dataset:
        if not isinstance(sample, dict):
            continue
        texts = extract_texts(sample, cfg)
        if not texts:
            continue
        for text in texts:
            if not text:
                continue
            yield text
            count_tokens += len(text.split())
            if cfg.sample_tokens and count_tokens >= cfg.sample_tokens:
                return
        count_entries += 1
        if cfg.sample_articles and count_entries >= cfg.sample_articles:
            break


def iter_local_file(path: Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n")


def extract_text(sample: dict) -> str | None:
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


def prepare_source(cfg: SourceConfig, out_dir: Path, cache_dir: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{cfg.name}.txt"
    metadata_path = out_dir / f"{cfg.name}.json"
    if output_path.exists() and metadata_path.exists():
        print(f"[prepare] skip (exists): {cfg.name}", flush=True)
        return output_path

    tmp_output = output_path.with_suffix(".txt.tmp")
    tmp_metadata = metadata_path.with_suffix(".json.tmp")
    if cfg.kind == "huggingface":
        iterator = iter_huggingface_source(cfg, cache_dir)
    elif cfg.kind == "local":
        iterator = iter_local_file(Path(cfg.path))
    else:
        raise ValueError(f"Unsupported source kind: {cfg.kind}")

    total_lines = 0
    total_chars = 0
    with tmp_output.open("w", encoding="utf-8") as writer:
        for line in iterator:
            cleaned = normalize_text(line)
            cleaned = cleaned.replace("\n", " ").replace("\t", " ").strip()
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
    tmp_metadata.write_text(json.dumps(metadata, indent=2))
    tmp_output.replace(output_path)
    tmp_metadata.replace(metadata_path)
    return output_path


def prepare_all(config: CorpusConfig, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in config.sources:
        prepare_source(source, out_dir, cache_dir=config.cache_dir)
