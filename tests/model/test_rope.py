import torch

from alm.model import apply_rope, rope_angles


def test_rope_shapes() -> None:
    seq_len = 4
    dim = 8
    cos, sin = rope_angles(seq_len, dim, offset=3)
    x = torch.randn(1, seq_len, dim)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
