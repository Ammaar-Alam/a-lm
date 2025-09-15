"""Root Mean Square Layer Normalization."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Implementation of RMSNorm with optional bias."""

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias: Optional[nn.Parameter]
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        norm = torch.rsqrt(norm + self.eps)
        output = x * norm * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

    @staticmethod
    def scaled_init(std: float, dim: int) -> float:
        return std / math.sqrt(dim)
