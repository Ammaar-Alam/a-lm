#!/usr/bin/env python3
"""Generate text from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from alm.model.config import ModelConfig
from alm.model.transformer import TransformerModel
from alm.tokenizers import Tokenizer


def resolve_device(name: str | None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(checkpoint: Path, device: torch.device) -> tuple[ModelConfig, dict, str | None]:
    payload = torch.load(checkpoint, map_location=device)
    config = ModelConfig.from_dict(payload["config"])
    state = payload["model"]
    return config, state, payload.get("tokenizer_fingerprint")


def sample_next_token(logits: torch.Tensor, top_k: int, top_p: float, temperature: float) -> int:
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(indices[choice])
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        if mask.shape[0] > 0:
            mask[..., 0] = False
        clipped = sorted_probs.masked_fill(mask, 0.0)
        total = clipped.sum()
        clipped = torch.ones_like(clipped) / clipped.size(-1) if total <= 0 else clipped / total
        choice = torch.multinomial(clipped, num_samples=1)
        return int(sorted_indices[choice])
    choice = torch.multinomial(probs, num_samples=1)
    return int(choice)


def generate(
    model: TransformerModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: torch.device,
) -> str:
    model.eval()
    with torch.inference_mode():
        prompt_tokens = tokenizer.encode(prompt)
        generated: list[int] = list(prompt_tokens)
        input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
        past = None
        for _ in range(max_tokens):
            logits, past, _ = model(input_ids, past_key_values=past, use_cache=True)
            next_logits = logits[:, -1, :].squeeze(0)
            if repetition_penalty > 1.0 and generated:
                window = generated[-128:]
                penalize = torch.tensor(sorted(set(window)), device=device, dtype=torch.long)
                if penalize.numel() > 0:
                    next_logits.index_copy_(0, penalize, next_logits[penalize] / repetition_penalty)
            next_token = sample_next_token(next_logits, top_k, top_p, temperature)
            generated.append(next_token)
            input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
        return tokenizer.decode(generated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from an a-lm checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (ckpt-last.pt)")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--prompt", default="Hello", help="Prompt string")
    parser.add_argument("--max-tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 for greedy)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling (0 disables)")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help=">1.0 discourages repeated tokens (1.0 disables)",
    )
    parser.add_argument("--device", default="auto", help="Device to run on (auto/mps/cuda/cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(Path(args.tokenizer))
    model_config, state_dict, checkpoint_fp = load_checkpoint(Path(args.checkpoint), device)
    if checkpoint_fp and checkpoint_fp != tokenizer.fingerprint:
        raise ValueError("Tokenizer fingerprint does not match checkpoint")
    model = TransformerModel(model_config).to(device)
    model.load_state_dict(state_dict)
    output = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        device,
    )
    print(output)


if __name__ == "__main__":
    main()
