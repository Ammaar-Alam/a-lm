import torch

from alm.model import SwiGLU


def test_swiglu_forward() -> None:
    layer = SwiGLU(dim=4, hidden_dim=8)
    x = torch.randn(2, 4)
    out = layer(x)
    assert out.shape == x.shape
