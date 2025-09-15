import torch

from amlm.model.dual_ffn import DualFFN


def test_dual_ffn_routes_tokens() -> None:
    dual = DualFFN(dim=8, hidden_small=16, hidden_large=32)
    x = torch.randn(2, 4, 8)
    out, stats = dual(x)
    assert out.shape == x.shape
    assert stats.routed_small + stats.routed_large >= x.numel() // x.shape[-1]


def test_dual_ffn_capacity_reroute() -> None:
    dual = DualFFN(dim=8, hidden_small=16, hidden_large=32, capacity_factor=0.1, drop_tokens=False)
    x = torch.randn(1, 10, 8)
    out, stats = dual(x)
    assert out.shape == x.shape
    assert stats.dropped == 0
