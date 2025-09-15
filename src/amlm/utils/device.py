"""Helpers for selecting the best available compute device."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled in tests via mocking
    torch = None  # type: ignore


@dataclass(frozen=True)
class DeviceInfo:
    """Runtime view of the chosen compute device."""

    type: str
    index: Optional[int] = None

    def as_torch_device(self) -> str:
        if self.index is None:
            return self.type
        return f"{self.type}:{self.index}"


def is_mps_available() -> bool:
    """Return True if PyTorch reports that MPS is available."""

    if torch is None:
        return False
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def detect_device(preferred: str = "mps", fallback: str = "cpu") -> DeviceInfo:
    """Choose the best available device, preferring MPS on Apple Silicon."""

    if preferred == "mps" and is_mps_available():
        return DeviceInfo(type="mps")

    if torch is not None and torch.cuda.is_available():
        index = torch.cuda.current_device()
        return DeviceInfo(type="cuda", index=index)

    return DeviceInfo(type=fallback)


def _main() -> None:
    info = detect_device()
    print(f"Selected device: {info.as_torch_device()}")
    if info.type != "mps":
        print("MPS unavailable; falling back to", info.as_torch_device())


if __name__ == "__main__":
    _main()
