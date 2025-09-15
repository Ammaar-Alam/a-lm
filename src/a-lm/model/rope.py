"""Rotary positional embeddings and helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_angles(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    base: Optional[float] = None,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if base is None:
        base = theta
    assert dim % 2 == 0, "RoPE dimension must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(offset, offset + seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)
