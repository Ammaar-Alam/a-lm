import torch

from a-lm.model import build_alibi_bias, get_slopes


def test_get_slopes_count() -> None:
    slopes = get_slopes(6)
    assert slopes.shape[0] == 6


def test_build_alibi_bias_shape() -> None:
    bias = build_alibi_bias(4, query_len=2, key_len=6, device=torch.device("cpu"), past_len=3)
    assert bias.shape == (4, 2, 6)
