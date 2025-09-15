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


def load_checkpoint(checkpoint: Path, device: torch.device) -> tuple[ModelConfig, dict]:
    payload = torch.load(checkpoint, map_location=device)
    config = ModelConfig.from_dict(payload["config"])
    state = payload["model"]
    return config, state


def sample_next_token(logits: torch.Tensor, top_k: int, temperature: float) -> int:
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(indices[choice])
    probs = torch.softmax(logits, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    return int(choice)


def generate(
    model: TransformerModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
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
            next_token = sample_next_token(next_logits, top_k, temperature)
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
    parser.add_argument("--device", default="auto", help="Device to run on (auto/mps/cuda/cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(Path(args.tokenizer))
    model_config, state_dict = load_checkpoint(Path(args.checkpoint), device)
    model = TransformerModel(model_config).to(device)
    model.load_state_dict(state_dict)
    output = generate(
        model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, device
    )
    print(output)


if __name__ == "__main__":
    main()
