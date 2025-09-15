"""ALiBi bias utilities."""

from __future__ import annotations

import math

import torch


def get_slopes(n_heads: int) -> torch.Tensor:
    def get_power_of_2_slopes(power: int) -> torch.Tensor:
        start = 2 ** (-(2 ** -(math.log2(power) - 3)))
        ratio = start
        return torch.tensor([start * ratio**i for i in range(power)], dtype=torch.float32)

    if math.log2(n_heads).is_integer():
        slopes = get_power_of_2_slopes(n_heads)
    else:
        closest_power = 2 ** math.floor(math.log2(n_heads))
        slopes = get_power_of_2_slopes(closest_power)
        extra = get_power_of_2_slopes(2 * closest_power)[0::2][: n_heads - closest_power]
        slopes = torch.cat([slopes, extra], dim=0)
    return slopes


def build_alibi_bias(
    n_heads: int,
    query_len: int,
    key_len: int,
    device: torch.device,
    past_len: int = 0,
) -> torch.Tensor:
    slopes = get_slopes(n_heads).to(device)
    q_pos = torch.arange(past_len, past_len + query_len, device=device)
    k_pos = torch.arange(0, key_len, device=device)
    bias = slopes[:, None, None] * (q_pos[None, :, None] - k_pos[None, None, :])
    return bias
