from types import SimpleNamespace

import amlm.utils.device as device


def test_detect_device_without_torch(monkeypatch) -> None:
    monkeypatch.setattr(device, "torch", None)
    assert device.is_mps_available() is False
    info = device.detect_device()
    assert info.type == "cpu"


def test_detect_device_prefers_mps(monkeypatch) -> None:
    fake_backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: True)
    )
    fake_cuda = SimpleNamespace(is_available=lambda: False, current_device=lambda: 0)
    fake_torch = SimpleNamespace(backends=fake_backends, cuda=fake_cuda)
    monkeypatch.setattr(device, "torch", fake_torch)
    assert device.is_mps_available() is True
    info = device.detect_device()
    assert info.type == "mps"
