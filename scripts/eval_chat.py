#!/usr/bin/env python3
"""Run a small qualitative chat eval set against a checkpoint.

This uses the same prompt template and decoding logic as scripts/chat_cli.py,
but runs non-interactively for quick comparisons between checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from alm.model.config import ModelConfig
from alm.model.transformer import TransformerModel
from alm.tokenizers import Tokenizer

DEFAULT_PROMPTS = [
    "Hello! Introduce yourself in two sentences.",
    "What is your name?",
    "List three practical uses for a pocket-sized drone.",
    "Write a haiku about rain.",
    "Explain what gradient descent is in one paragraph.",
    "What is 17 * 23? Show the final answer only.",
]


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
    if isinstance(state, dict) and any(
        isinstance(key, str) and key.startswith("_orig_mod.") for key in state
    ):
        stripped: dict = {}
        for key, value in state.items():
            if isinstance(key, str) and key.startswith("_orig_mod."):
                stripped[key[len("_orig_mod.") :]] = value
            else:
                stripped[key] = value
        state = stripped
    return config, state, payload.get("tokenizer_fingerprint")


def sample_next_token(logits: torch.Tensor, top_k: int, top_p: float, temperature: float) -> int:
    scaled = logits / max(temperature, 1e-5)
    if top_k <= 0 and not (0.0 < top_p < 1.0):
        return int(torch.argmax(scaled).item())
    if top_k > 0:
        values, indices = torch.topk(scaled, top_k)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(indices[choice])
    probs = torch.softmax(scaled, dim=-1)
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
                    selected = next_logits[penalize]
                    adjusted = torch.where(
                        selected < 0,
                        selected * repetition_penalty,
                        selected / repetition_penalty,
                    )
                    try:
                        next_logits.index_copy_(0, penalize, adjusted)
                    except NotImplementedError:
                        next_logits[penalize] = adjusted
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


def build_prompt(history: list[tuple[str, str]], user_message: str, eot_token: str | None) -> str:
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
            if eot_token:
                parts.append(f"{eot_token}\n")

    parts.append(f"User: {user_message.strip()}\nAssistant: ")
    return "".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a small prompt set")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (ckpt-last.pt)")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--device", default="auto", help="Device to run on (auto/mps/cuda/cpu)")
    parser.add_argument("--system", default="", help="System prompt (must match SFT packing)")
    parser.add_argument("--eot-token", default="", help="Optional EOT marker (must match packing)")
    parser.add_argument("--max-response", type=int, default=128, help="Max tokens per reply")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 for greedy)")
    parser.add_argument("--top-p", type=float, default=0.0, help="Nucleus sampling (0 disables)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to run (repeatable). Uses a default set if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(Path(args.tokenizer))
    config, weights, checkpoint_fp = load_checkpoint(Path(args.checkpoint), device)
    if checkpoint_fp and checkpoint_fp != tokenizer.fingerprint:
        raise ValueError("Tokenizer fingerprint does not match checkpoint")
    model = TransformerModel(config).to(device)
    model.load_state_dict(weights)

    eot_token = args.eot_token.strip() or None
    stop_strings = ([eot_token] if eot_token else []) + ["\nUser:", "\nSystem:"]
    prompts = args.prompt or DEFAULT_PROMPTS

    max_context = config.max_position_embeddings
    context_budget = max_context - args.max_response
    if context_budget <= 0:
        context_budget = max_context

    history = []
    if args.system.strip():
        history = [("System", args.system.strip())]

    print(f"checkpoint={args.checkpoint}")
    print(
        f"decode: temp={args.temperature} top_k={args.top_k} top_p={args.top_p} "
        f"rep_pen={args.repetition_penalty} max_resp={args.max_response}"
    )
    for idx, user_message in enumerate(prompts, start=1):
        prompt_text = build_prompt(history, user_message, eot_token)
        prompt_tokens = tokenizer.encode(prompt_text)
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
        print("\n" + "=" * 80)
        print(f"PROMPT {idx}: {user_message}")
        print(f"REPLY  {idx}: {reply}")


if __name__ == "__main__":
    main()
