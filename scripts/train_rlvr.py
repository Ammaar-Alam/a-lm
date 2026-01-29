#!/usr/bin/env python3
"""Verifiable-reward RL (GRPO-style) for a-lm checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from alm.model.config import ModelConfig
from alm.model.transformer import TransformerModel
from alm.rlvr.logprobs import completion_mean_logprobs
from alm.rlvr.reward import dense_int_reward, exact_int_reward
from alm.tokenizers import Tokenizer


def resolve_device(name: str | None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, max_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(0, int(warmup_steps))
    max_steps = max(1, int(max_steps))

    def lr_lambda(step: int) -> float:
        step = min(step, max_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    config: ModelConfig,
    tokenizer_fingerprint: str | None,
    meta: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": dataclasses.asdict(config),
        "meta": meta,
    }
    if tokenizer_fingerprint:
        payload["tokenizer_fingerprint"] = tokenizer_fingerprint
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[int, str | None, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return (
        int(payload.get("step", 0)),
        payload.get("tokenizer_fingerprint"),
        payload.get("meta", {}),
    )


def load_init_checkpoint(path: Path) -> tuple[ModelConfig, dict[str, Any], int, str | None]:
    payload = torch.load(path, map_location="cpu")
    cfg = ModelConfig.from_dict(payload["config"])
    state = payload["model"]
    step = int(payload.get("step", 0))
    return cfg, state, step, payload.get("tokenizer_fingerprint")


def iter_jsonl(path: Path) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(record.get("prompt") or "")
            answer = str(record.get("answer") or "")
            if prompt and answer:
                examples.append({"prompt": prompt, "answer": answer})
    if not examples:
        raise ValueError(f"no usable examples found in {path}")
    return examples


def sample_next_token(logits: torch.Tensor, *, top_k: int, top_p: float, temperature: float) -> int:
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


def generate_completion(
    model: TransformerModel,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stop_token_id: int | None,
    device: torch.device,
) -> list[int]:
    generated: list[int] = []
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    past = None

    logits, past, _ = model(input_ids, past_key_values=past, use_cache=True)
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :].squeeze(0)
        if repetition_penalty > 1.0 and prompt_ids:
            window = prompt_ids[-128:] + generated[-128:]
            penalize = torch.tensor(sorted(set(window)), device=device, dtype=torch.long)
            if penalize.numel() > 0:
                next_logits.index_copy_(0, penalize, next_logits[penalize] / repetition_penalty)
        token = sample_next_token(next_logits, top_k=top_k, top_p=top_p, temperature=temperature)
        generated.append(token)
        if stop_token_id is not None and token == stop_token_id:
            break
        input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        logits, past, _ = model(input_ids, past_key_values=past, use_cache=True)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with verifiable reward signals (RLVR)")
    parser.add_argument("--init", required=True, help="Initial checkpoint (pretrain or SFT)")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--data", required=True, help="RLVR dataset JSONL (prompt/answer)")
    parser.add_argument("--out", required=True, help="Output directory for checkpoints")
    parser.add_argument("--device", default="auto", help="Device (auto/mps/cuda/cpu)")
    parser.add_argument("--resume", help="Optional checkpoint to resume from")

    parser.add_argument("--steps", type=int, default=2000, help="Optimizer steps")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm")

    parser.add_argument("--batch-prompts", type=int, default=4, help="Prompts per step")
    parser.add_argument("--group-size", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Tokens per completion")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help=">1.0 discourages repeats (1.0 disables)",
    )

    parser.add_argument(
        "--reward",
        choices=["dense", "exact"],
        default="dense",
        help="Reward function (dense helps bootstrap from weaker checkpoints)",
    )
    parser.add_argument(
        "--kl-beta",
        type=float,
        default=0.02,
        help="KL penalty coefficient (uses init checkpoint as reference)",
    )

    parser.add_argument("--checkpoint-interval", type=int, default=250, help="Save every N steps")
    parser.add_argument("--log-interval", type=int, default=25, help="Log every N steps")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")

    tokenizer = Tokenizer.from_file(Path(args.tokenizer))
    examples = iter_jsonl(Path(args.data))
    rng = random.Random(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_path = Path(args.init)
    config, state, init_step, init_fp = load_init_checkpoint(init_path)
    if init_fp and init_fp != tokenizer.fingerprint:
        raise ValueError("Tokenizer fingerprint does not match init checkpoint")

    model = TransformerModel(config).to(device)
    model.load_state_dict(state)

    ref_model = TransformerModel(config).to(device)
    ref_model.load_state_dict(state)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = build_optimizer(model, _as_float(args.lr, 3e-5))
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=args.steps)

    start_step = 0
    last_ckpt = out_dir / "ckpt-last.pt"
    ckpt_fp: str | None = tokenizer.fingerprint
    meta: dict[str, Any] = {
        "stage": "rlvr",
        "init_checkpoint": str(init_path),
        "init_step": init_step,
        "data": str(Path(args.data)),
        "reward": args.reward,
        "kl_beta": float(args.kl_beta),
    }

    if args.resume and Path(args.resume).exists():
        start_step, ckpt_fp, meta = load_checkpoint(
            Path(args.resume), model=model, optimizer=optimizer, scheduler=scheduler
        )
        print(f"Resumed from {args.resume} @ step {start_step}")
    elif last_ckpt.exists():
        start_step, ckpt_fp, meta = load_checkpoint(
            last_ckpt, model=model, optimizer=optimizer, scheduler=scheduler
        )
        print(f"Resumed from {last_ckpt} @ step {start_step}")

    if ckpt_fp and ckpt_fp != tokenizer.fingerprint:
        raise ValueError("Tokenizer fingerprint mismatch between checkpoint and tokenizer")

    reward_fn = dense_int_reward if args.reward == "dense" else exact_int_reward
    stop_token_id = tokenizer.encode("\n")[-1]

    model.train()
    grad_clip = float(args.grad_clip)
    batch_prompts = max(1, int(args.batch_prompts))
    group_size = max(1, int(args.group_size))
    max_new_tokens = max(1, int(args.max_new_tokens))

    start_time = time.perf_counter()
    step = start_step
    while step < int(args.steps):
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        batch = [rng.choice(examples) for _ in range(batch_prompts)]
        prompts: list[list[int]] = [tokenizer.encode(item["prompt"]) for item in batch]
        answers: list[str] = [item["answer"] for item in batch]

        sequences: list[list[int]] = []
        prompt_lens: list[int] = []
        rewards: list[float] = []

        model.eval()
        with torch.no_grad():
            for prompt_ids, expected in zip(prompts, answers):
                for _ in range(group_size):
                    completion_ids = generate_completion(
                        model,
                        prompt_ids=prompt_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        repetition_penalty=float(args.repetition_penalty),
                        stop_token_id=stop_token_id,
                        device=device,
                    )
                    completion_text = tokenizer.decode(completion_ids).strip()
                    reward = float(reward_fn(completion_text, expected))
                    sequences.append(prompt_ids + completion_ids)
                    prompt_lens.append(len(prompt_ids))
                    rewards.append(reward)
        model.train()

        mean_logp, token_counts = completion_mean_logprobs(
            model,
            sequences=sequences,
            prompt_lens=prompt_lens,
            pad_id=0,
            device=device,
        )
        if float(args.kl_beta) > 0.0:
            with torch.no_grad():
                ref_logp, _ = completion_mean_logprobs(
                    ref_model,
                    sequences=sequences,
                    prompt_lens=prompt_lens,
                    pad_id=0,
                    device=device,
                )
            kl_estimate = mean_logp - ref_logp
            kl_loss = float(args.kl_beta) * kl_estimate.mean()
        else:
            kl_loss = torch.tensor(0.0, device=device)

        reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        grouped = reward_tensor.view(batch_prompts, group_size)
        advantages = (grouped - grouped.mean(dim=1, keepdim=True)).view(-1).detach()

        pg_loss = -(advantages * mean_logp).mean()
        loss = pg_loss + kl_loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % int(args.log_interval) == 0 or step == start_step + 1:
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            iter_time = max(time.perf_counter() - step_start, 1e-6)
            tok = float(token_counts.sum().item())
            print(
                "step="
                f"{step}/{args.steps} "
                f"loss={float(loss.item()):.4f} "
                f"pg={float(pg_loss.item()):.4f} "
                f"kl={float(kl_loss.item()):.4f} "
                f"r_mean={float(reward_tensor.mean().item()):.3f} "
                f"tok/s={(tok / iter_time):.0f} "
                f"wall={elapsed:.0f}s"
            )

        if step % int(args.checkpoint_interval) == 0 or step == int(args.steps):
            save_checkpoint(
                out_dir / f"ckpt-step{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                config=config,
                tokenizer_fingerprint=tokenizer.fingerprint,
                meta=meta,
            )
            save_checkpoint(
                last_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                config=config,
                tokenizer_fingerprint=tokenizer.fingerprint,
                meta=meta,
            )

    print("RLVR training complete.")


if __name__ == "__main__":
    train(parse_args())
