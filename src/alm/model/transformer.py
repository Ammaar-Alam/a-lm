"""Transformer model composed of attention and dual FFN blocks."""

from __future__ import annotations

import torch
from torch import nn

from .alibi import build_alibi_bias
from .attention import MultiHeadAttention
from .config import ModelConfig
from .dual_ffn import DualFFN, RouterStats
from .rmsnorm import RMSNorm
from .rope import rope_angles

KeyValueCache = tuple[torch.Tensor, torch.Tensor]


def _format_attention_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None or mask.ndim != 2:
        return mask
    formatted = mask[:, None, None, :]
    return (1.0 - formatted) * -1e9


class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, config.n_kv_heads)
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if config.dual_ffn.enabled:
            self.ffn = DualFFN(
                config.d_model,
                config.small_hidden_size,
                config.ffn_hidden_size,
                router_temperature=config.dual_ffn.router_temperature,
                capacity_factor=config.dual_ffn.capacity_factor,
                drop_tokens=config.dual_ffn.drop_tokens,
            )
        else:
            from .swiglu import SwiGLU

            self.ffn = SwiGLU(config.d_model, config.ffn_hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        alibi_bias: torch.Tensor | None,
        past_key_value: KeyValueCache | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, KeyValueCache | None, RouterStats | None]:
        residual = hidden_states
        normed = self.attn_norm(hidden_states)
        attn_out, present = self.attn(
            normed,
            attention_mask=attention_mask,
            position_cos=cos,
            position_sin=sin,
            alibi_bias=alibi_bias,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(attn_out)

        residual = hidden_states
        normed = self.ffn_norm(hidden_states)
        if isinstance(self.ffn, DualFFN):
            ff_out, stats = self.ffn(normed)
        else:
            ff_out = self.ffn(normed)
            stats = None
        hidden_states = residual + self.dropout(ff_out)
        return hidden_states, present, stats


class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[KeyValueCache | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[KeyValueCache | None], list[RouterStats | None]]:
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        hidden_states = self.embed_tokens(input_ids)
        past_len = 0
        if past_key_values and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[-2]

        caches: list[KeyValueCache | None] = []
        router_stats: list[RouterStats | None] = []

        cos, sin = rope_angles(
            input_ids.size(1),
            self.config.head_dim,
            theta=self.config.rope_theta,
            offset=past_len,
        )
        cos = cos.to(hidden_states.device)
        sin = sin.to(hidden_states.device)

        formatted_mask = _format_attention_mask(attention_mask)

        alibi_bias = None
        if self.config.alibi:
            total_len = past_len + input_ids.size(1)
            alibi_bias = build_alibi_bias(
                self.config.n_heads,
                query_len=input_ids.size(1),
                key_len=total_len,
                device=hidden_states.device,
                past_len=past_len,
            )

        for layer, past in zip(self.layers, past_key_values):
            hidden_states, present, stats = layer(
                hidden_states,
                formatted_mask,
                cos,
                sin,
                alibi_bias,
                past,
                use_cache,
            )
            caches.append(present if use_cache else None)
            router_stats.append(stats)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, caches, router_stats
