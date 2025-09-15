"""SwiGLU feed-forward block."""

from __future__ import annotations

import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.w(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return self.out(torch.nn.functional.silu(x1) * x2)
