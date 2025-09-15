"""Configuration dataclasses for the transformer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DualFFNConfig:
    enabled: bool = True
    small_ratio: float = 0.5
    router_temperature: float = 1.0
    capacity_factor: float = 1.0
    drop_tokens: bool = False


@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden_size: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float = 10000.0
    dropout: float = 0.0
    alibi: bool = False
    dual_ffn: DualFFNConfig = field(default_factory=DualFFNConfig)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def small_hidden_size(self) -> int:
        if not self.dual_ffn.enabled:
            return self.ffn_hidden_size
        return max(1, int(self.ffn_hidden_size * self.dual_ffn.small_ratio))
