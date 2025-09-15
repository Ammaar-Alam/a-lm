import torch

from alm.model.attention import MultiHeadAttention, repeat_kv


def test_repeat_kv() -> None:
    x = torch.randn(2, 2, 3, 4)
    out = repeat_kv(x, 2)
    assert out.shape == (2, 4, 3, 4)


def test_attention_shapes() -> None:
    mha = MultiHeadAttention(dim=16, n_heads=4, n_kv_heads=2)
    x = torch.randn(2, 3, 16)
    out, cache = mha(x)
    assert out.shape == (2, 3, 16)
    assert cache is None


def test_attention_cache() -> None:
    mha = MultiHeadAttention(dim=8, n_heads=2, n_kv_heads=1)
    x = torch.randn(1, 1, 8)
    out, cache = mha(x, use_cache=True)
    assert cache is not None
    next_x = torch.randn(1, 1, 8)
    out2, cache2 = mha(next_x, past_key_value=cache, use_cache=True)
    assert out2.shape == (1, 1, 8)
    assert cache2 is not None
    assert cache2[0].shape[-2] == 2
