"""Dual-path feed-forward network with top-1 routing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .swiglu import SwiGLU


@dataclass
class RouterStats:
    routed_small: int
    routed_large: int
    dropped: int


class DualFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_small: int,
        hidden_large: int,
        router_temperature: float = 1.0,
        capacity_factor: float = 1.0,
        drop_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.router_temperature = router_temperature
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

        self.router = nn.Linear(dim, 2, bias=True)
        self.ffn_small = SwiGLU(dim, hidden_small)
        self.ffn_large = SwiGLU(dim, hidden_large)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RouterStats]:
        bsz, seq_len, dim = x.shape
        tokens = bsz * seq_len
        flat = x.reshape(tokens, dim)

        logits = self.router(flat) / max(self.router_temperature, 1e-6)
        routing = torch.argmax(logits, dim=-1)

        capacity = max(1, int(math.ceil(tokens / 2 * self.capacity_factor)))

        small_idx = torch.nonzero(routing == 0, as_tuple=False).flatten()
        large_idx = torch.nonzero(routing == 1, as_tuple=False).flatten()

        dropped = 0
        if large_idx.numel() > capacity:
            overflow = large_idx[capacity:]
            if self.drop_tokens:
                dropped = overflow.numel()
                large_idx = large_idx[:capacity]
            else:
                small_idx = torch.cat([small_idx, overflow], dim=0)
                large_idx = large_idx[:capacity]

        output = torch.zeros_like(flat)
        if small_idx.numel() > 0:
            output.index_copy_(0, small_idx, self.ffn_small(flat.index_select(0, small_idx)))
        if large_idx.numel() > 0:
            output.index_copy_(0, large_idx, self.ffn_large(flat.index_select(0, large_idx)))

        stats = RouterStats(
            routed_small=int(small_idx.numel()),
            routed_large=int(large_idx.numel()),
            dropped=int(dropped),
        )

        return output.reshape(bsz, seq_len, dim), stats
