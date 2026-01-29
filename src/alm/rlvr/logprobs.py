"""Utilities for computing completion log-probabilities."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from alm.model.transformer import TransformerModel


def completion_mean_logprobs(
    model: TransformerModel,
    *,
    sequences: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(sequences) != len(prompt_lens):
        raise ValueError("sequences and prompt_lens must have matching length")
    if not sequences:
        raise ValueError("sequences must be non-empty")

    max_len = max(len(seq) for seq in sequences)
    if max_len < 2:
        raise ValueError("all sequences must have at least 2 tokens")

    batch = len(sequences)
    seq_len = max_len - 1

    input_ids = torch.full((batch, seq_len), pad_id, dtype=torch.long)
    target_ids = torch.full((batch, seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, seq_len), dtype=torch.float32)

    lengths = torch.zeros((batch,), dtype=torch.long)
    for row, seq in enumerate(sequences):
        tokens = list(seq)
        if len(tokens) < 2:
            raise ValueError("sequence too short")
        inp = tokens[:-1]
        tgt = tokens[1:]
        length = len(inp)
        input_ids[row, :length] = torch.tensor(inp, dtype=torch.long)
        target_ids[row, :length] = torch.tensor(tgt, dtype=torch.long)
        attention_mask[row, :length] = 1.0
        lengths[row] = length

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    attention_mask = attention_mask.to(device)

    logits, _, _ = model(input_ids, attention_mask=attention_mask, use_cache=False)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    completion_logp = torch.zeros((batch,), device=device, dtype=torch.float32)
    completion_tokens = torch.zeros((batch,), device=device, dtype=torch.float32)
    for row, prompt_len in enumerate(prompt_lens):
        start = max(int(prompt_len) - 1, 0)
        end = int(lengths[row].item())
        if start >= end:
            continue
        completion_logp[row] = token_log_probs[row, start:end].sum()
        completion_tokens[row] = float(end - start)

    denom = completion_tokens.clamp(min=1.0)
    return completion_logp / denom, completion_tokens
