"""Rotary positional embeddings and helpers."""

from __future__ import annotations

import torch

_ROPE_CACHE: dict[tuple[int, int, float, int, str], tuple[torch.Tensor, torch.Tensor]] = {}

try:
    import torch._dynamo as _dynamo
except Exception:  # pragma: no cover - torch may be built without dynamo
    _dynamo = None  # type: ignore


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_angles(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    base: float | None = None,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if base is None:
        base = theta
    assert dim % 2 == 0, "RoPE dimension must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(offset, offset + seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)
    return cos, sin


def rope_angles_cached(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    base: float | None = None,
    offset: int = 0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (
        seq_len,
        dim,
        float(theta if base is None else base),
        offset,
        str(device) if device is not None else "cpu",
    )
    cached = _ROPE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cos, sin = rope_angles(seq_len, dim, theta=theta, base=base, offset=offset)
    if device is not None:
        cos = cos.to(device)
        sin = sin.to(device)
    _ROPE_CACHE[cache_key] = (cos, sin)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    for _ in range(x.ndim - cos.ndim):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


if _dynamo is not None:  # pragma: no cover - runtime-dependent
    # RoPE caches can confuse torch.compile + cudagraph reuse when tensors are cached across runs.
    # Keep this helper in eager mode while still allowing the surrounding model to compile.
    rope_angles_cached = _dynamo.disable(rope_angles_cached)
