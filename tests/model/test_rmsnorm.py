import torch

from a-lm.model import RMSNorm


def test_rmsnorm_shapes() -> None:
    layer = RMSNorm(8)
    x = torch.randn(2, 4, 8)
    out = layer(x)
    assert out.shape == x.shape


def test_rmsnorm_stability() -> None:
    layer = RMSNorm(4, eps=1e-5)
    x = torch.zeros(1, 4)
    out = layer(x)
    assert torch.allclose(out, torch.zeros_like(out))
