"""Attention module supporting grouped and multi-query attention."""

from __future__ import annotations

import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F

from .rope import apply_rope

_CAUSAL_MASK_CACHE: dict[tuple[int, int, int, str, int | None, torch.dtype], torch.Tensor] = {}
_SDPA_SUPPORTS_GQA = False
if hasattr(F, "scaled_dot_product_attention"):
    try:
        _SDPA_SUPPORTS_GQA = (
            "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
        )
    except (TypeError, ValueError):  # pragma: no cover - depends on torch build
        _SDPA_SUPPORTS_GQA = False


def get_causal_additive_mask(
    q_len: int,
    k_len: int,
    *,
    past_len: int = 0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (q_len, k_len, past_len, device.type, device.index, dtype)
    cached = _CAUSAL_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    neg = -1e4 if dtype in {torch.float16, torch.bfloat16} else -1e9
    mask = torch.full((q_len, k_len), neg, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=past_len + 1)
    _CAUSAL_MASK_CACHE[key] = mask
    return mask


KeyValueCache = tuple[torch.Tensor, torch.Tensor]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, n_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(bsz, n_heads, n_rep, seq_len, head_dim)
    return x.reshape(bsz, n_heads * n_rep, seq_len, head_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, backend: str = "math") -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by number of heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be multiple of n_kv_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.num_groups = n_heads // n_kv_heads
        self.backend = backend

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_cos: torch.Tensor | None = None,
        position_sin: torch.Tensor | None = None,
        alibi_bias: torch.Tensor | None = None,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KeyValueCache | None]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if position_cos is not None and position_sin is not None:
            q = apply_rope(q, position_cos, position_sin)
            k = apply_rope(k, position_cos, position_sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present: KeyValueCache | None = None
        if use_cache:
            present = (k, v)

        use_sdpa = (
            self.backend == "sdpa"
            and past_key_value is None
            and alibi_bias is None
            and attention_mask is None
            and hasattr(F, "scaled_dot_product_attention")
        )

        if use_sdpa:
            k_attn = k
            v_attn = v
            sdpa_kwargs: dict[str, bool] = {}
            if self.num_groups > 1:
                if _SDPA_SUPPORTS_GQA and hidden_states.device.type == "cuda":
                    sdpa_kwargs["enable_gqa"] = True
                else:
                    k_attn = repeat_kv(k_attn, self.num_groups)
                    v_attn = repeat_kv(v_attn, self.num_groups)
            attn_output = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                **sdpa_kwargs,
            )
        else:
            k = repeat_kv(k, self.num_groups)
            v = repeat_kv(v, self.num_groups)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            q_len = attn_scores.shape[-2]
            k_len = attn_scores.shape[-1]
            past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]

            attn_scores = attn_scores + get_causal_additive_mask(
                q_len,
                k_len,
                past_len=past_len,
                device=attn_scores.device,
                dtype=attn_scores.dtype,
            )

            if alibi_bias is not None:
                attn_scores = attn_scores + alibi_bias[:, None, :q_len, :k_len]

            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
            attn_output = torch.matmul(attn_weights.to(q.dtype), v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        output = self.out_proj(attn_output)
        return output, present
