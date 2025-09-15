#!/usr/bin/env python3
"""Pretraining loop for the Alam Language Model."""

from __future__ import annotations

import argparse
import dataclasses
import math
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from alm.data.dataset import PackedDataset, collate_tokens
from alm.model.config import DualFFNConfig, ModelConfig
from alm.model.transformer import TransformerModel


def resolve_device(name: str | None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_config(path: Path) -> ModelConfig:
    data = yaml.safe_load(path.read_text())
    model_data: dict[str, Any] = data.get("model", {})
    dual_cfg = model_data.get("dual_ffn", {})
    dual = DualFFNConfig(
        enabled=dual_cfg.get("enabled", True),
        small_ratio=dual_cfg.get("small_ratio", 0.5),
        router_temperature=dual_cfg.get("router_temperature", 1.0),
        capacity_factor=dual_cfg.get("capacity_factor", 1.0),
        drop_tokens=dual_cfg.get("drop_tokens", False),
    )
    return ModelConfig(
        d_model=model_data["d_model"],
        n_layers=model_data["n_layers"],
        n_heads=model_data["n_heads"],
        n_kv_heads=model_data["n_kv_heads"],
        ffn_hidden_size=model_data["ffn_hidden_size"],
        vocab_size=model_data["vocab_size"],
        max_position_embeddings=model_data["max_position_embeddings"],
        rope_theta=model_data.get("rope_theta", 10000.0),
        dropout=model_data.get("dropout", 0.0),
        alibi=model_data.get("alibi", False),
        dual_ffn=dual,
    )


def load_train_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 3e-4),
        betas=betas,  # type: ignore[arg-type]
        eps=cfg.get("eps", 1e-8),
        weight_decay=cfg.get("weight_decay", 0.1),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any]
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = cfg.get("warmup_steps", 0)
    max_steps = max(cfg.get("max_steps", 1), 1)

    def lr_lambda(step: int) -> float:
        step = min(step, max_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    config: ModelConfig,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": dataclasses.asdict(config),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> int:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("step", 0))


def collate_for_training(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    return inputs, targets


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = load_model_config(Path(args.model))
    train_config = load_train_config(Path(args.train))

    dataset = PackedDataset(Path(args.data))
    seq_len = dataset.seq_len
    if seq_len < 2:
        raise ValueError("Sequence length must be at least 2 for language modeling")

    training_cfg = train_config.get("training", {})
    micro_batch_size = int(training_cfg.get("micro_batch_size", 8))
    grad_accum = int(training_cfg.get("gradient_accumulation", 1))
    max_steps = int(training_cfg.get("max_steps", 1000))
    grad_clip = float(training_cfg.get("gradient_clip_norm", 1.0))

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_tokens,
    )
    data_iter = cycle(dataloader)

    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")

    model = TransformerModel(model_config).to(device)
    optimizer = build_optimizer(model, train_config.get("optim", {}))
    scheduler = build_scheduler(optimizer, train_config.get("scheduler", {}))

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    autocast_ctx = (
        torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else torch.autocast(device_type="mps", dtype=torch.float16)
        if device.type == "mps"
        else nullcontext()
    )

    criterion = nn.CrossEntropyLoss()
    log_cfg = train_config.get("logging", {})
    log_interval = int(log_cfg.get("log_interval", 100))
    ckpt_interval = int(training_cfg.get("checkpoint_interval", 500))

    start_step = 0
    last_ckpt = output_dir / "ckpt-last.pt"
    if args.resume and Path(args.resume).exists():
        start_step = load_checkpoint(Path(args.resume), model, optimizer, scheduler)
        print(f"Resumed from {args.resume} @ step {start_step}")
    elif last_ckpt.exists():
        start_step = load_checkpoint(last_ckpt, model, optimizer, scheduler)
        print(f"Resumed from {last_ckpt} @ step {start_step}")

    model.train()
    step = start_step
    while step < max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(grad_accum):
            batch = next(data_iter).to(device)
            inputs, targets = collate_for_training(batch)
            inputs = inputs.to(device)
            targets = targets.to(device)
            with autocast_ctx:
                logits, _, _ = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += float(loss.detach().item())
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        step += 1

        if step % log_interval == 0 or step == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"step={step} loss={accum_loss:.4f} lr={lr:.3e}")
        if step % ckpt_interval == 0 or step == max_steps:
            save_checkpoint(
                output_dir / f"ckpt-step{step:06d}.pt",
                model,
                optimizer,
                scheduler,
                step,
                model_config,
            )
            save_checkpoint(last_ckpt, model, optimizer, scheduler, step, model_config)

    print("Training complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the a-lm transformer")
    parser.add_argument("--model", required=True, help="Model config YAML path")
    parser.add_argument("--train", required=True, help="Training config YAML path")
    parser.add_argument("--data", required=True, help="Packed dataset directory")
    parser.add_argument("--out", required=True, help="Output directory for checkpoints")
    parser.add_argument("--device", default="auto", help="Device to train on (auto/mps/cuda/cpu)")
    parser.add_argument("--resume", help="Optional checkpoint path to resume from")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
