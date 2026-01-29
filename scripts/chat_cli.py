#!/usr/bin/env python3
"""Interactive chat loop for a-lm checkpoints."""

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
    return config, payload["model"], payload.get("tokenizer_fingerprint")


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


def generate_reply(
    model: TransformerModel,
    tokenizer: Tokenizer,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stop_strings: list[str],
    device: torch.device,
) -> str:
    model.eval()
    with torch.inference_mode():
        generated = list(prompt_tokens)
        input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
        past = None
        prompt_len = len(prompt_tokens)
        for _ in range(max_tokens):
            logits, past, _ = model(input_ids, past_key_values=past, use_cache=True)
            next_logits = logits[:, -1, :].squeeze(0)
            if repetition_penalty > 1.0 and generated:
                window = generated[-128:]
                penalize = torch.tensor(sorted(set(window)), device=device, dtype=torch.long)
                if penalize.numel() > 0:
                    next_logits.index_copy_(0, penalize, next_logits[penalize] / repetition_penalty)
            token = sample_next_token(next_logits, top_k, top_p, temperature)
            generated.append(token)
            input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
            if stop_strings:
                reply_text = tokenizer.decode(generated[prompt_len:])
                for stop in stop_strings:
                    stop_idx = reply_text.find(stop)
                    if stop_idx != -1:
                        return reply_text[:stop_idx].strip()
        return tokenizer.decode(generated[prompt_len:]).strip()


def trim_to_context(tokens: list[int], limit: int) -> list[int]:
    if len(tokens) <= limit:
        return tokens
    return tokens[-limit:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with an a-lm checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (ckpt-last.pt)")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--device", default="auto", help="Device to run on (auto/mps/cuda/cpu)")
    parser.add_argument("--max-response", type=int, default=128, help="Max tokens per reply")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 for greedy)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling (0 disables)")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help=">1.0 discourages repeated tokens (1.0 disables)",
    )
    parser.add_argument(
        "--system", default="You are a helpful assistant.", help="System prompt prefix"
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Stop string (repeatable). Defaults to '\\nUser:' and '\\nSystem:' if omitted.",
    )
    return parser.parse_args()


def build_prompt(history: list[tuple[str, str]], user_message: str) -> str:
    system_text = ""
    turns: list[tuple[str, str]] = []
    for role, content in history:
        role_lower = role.strip().lower()
        text = content.strip()
        if not text:
            continue
        if role_lower == "system":
            system_text = text
        elif role_lower in {"user", "assistant"}:
            turns.append((role_lower, text))

    parts: list[str] = []
    if system_text:
        parts.append(f"System: {system_text}\n")

    for role, text in turns:
        if role == "user":
            parts.append(f"User: {text}\nAssistant: ")
        else:
            parts.append(f"{text}\n")

    parts.append(f"User: {user_message.strip()}\nAssistant: ")
    return "".join(parts)


def chat_loop(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(Path(args.tokenizer))
    config, weights, checkpoint_fp = load_checkpoint(Path(args.checkpoint), device)
    if checkpoint_fp and checkpoint_fp != tokenizer.fingerprint:
        raise ValueError("Tokenizer fingerprint does not match checkpoint")
    model = TransformerModel(config).to(device)
    model.load_state_dict(weights)

    max_context = config.max_position_embeddings
    history: list[tuple[str, str]] = [("System", args.system)]
    stop_strings = args.stop if args.stop else ["\nUser:", "\nSystem:"]

    print(f"Loaded model on {device}. Context window ~{config.max_position_embeddings} tokens.")
    print("Type /exit to quit. Press Ctrl+C to abort.")

    try:
        while True:
            try:
                user_message = input("you> ").strip()
            except EOFError:
                print()
                break
            if not user_message:
                continue
            if user_message.lower() in {"/exit", "quit", ":q"}:
                break

            prompt_text = build_prompt(history, user_message)
            prompt_tokens = tokenizer.encode(prompt_text)
            context_budget = max_context - args.max_response
            if context_budget <= 0:
                context_budget = max_context
            prompt_tokens = trim_to_context(prompt_tokens, context_budget)

            reply = generate_reply(
                model,
                tokenizer,
                prompt_tokens,
                args.max_response,
                args.temperature,
                args.top_k,
                args.top_p,
                args.repetition_penalty,
                stop_strings,
                device,
            )
            history.append(("User", user_message))
            history.append(("Assistant", reply))
            print(f"alm> {reply}")
    except KeyboardInterrupt:
        print("\nInterrupted.")


def main() -> None:
    args = parse_args()
    chat_loop(args)


if __name__ == "__main__":
    main()
