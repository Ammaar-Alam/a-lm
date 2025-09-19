#!/usr/bin/env python3
"""Pretraining loop for the Alam Language Model."""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
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
from alm.tokenizers import Tokenizer

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Console = None
    Progress = None


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
    attn_cfg = data.get("attention", {})
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
        attn_backend=attn_cfg.get("backend", "math"),
    )


def load_train_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    betas_cfg = cfg.get("betas", (0.9, 0.95))
    if isinstance(betas_cfg, (list, tuple)) and len(betas_cfg) == 2:
        betas = (float(betas_cfg[0]), float(betas_cfg[1]))
    else:
        betas = (0.9, 0.95)

    return torch.optim.AdamW(
        model.parameters(),
        lr=_as_float(cfg.get("lr", 3e-4), 3e-4),
        betas=betas,
        eps=_as_float(cfg.get("eps", 1e-8), 1e-8),
        weight_decay=_as_float(cfg.get("weight_decay", 0.1), 0.1),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any], total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(cfg.get("warmup_steps", 0))
    cfg_max_steps = int(cfg.get("max_steps", total_steps))
    max_steps = max(cfg_max_steps, total_steps, 1)

    def lr_lambda(step: int) -> float:
        step = min(step, max_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def override_scheduler_lr(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    base_lr: float,
) -> None:
    if not scheduler.base_lrs:
        for group in optimizer.param_groups:
            group["lr"] = base_lr
        return

    try:
        lambda_fn = scheduler.lr_lambdas[0]
        factor = float(lambda_fn(scheduler.last_epoch))
    except Exception:
        factor = 1.0

    desired_lr = base_lr * factor

    for group in optimizer.param_groups:
        group["lr"] = desired_lr

    scheduler.base_lrs = [base_lr for _ in scheduler.base_lrs]
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = [desired_lr for _ in scheduler.base_lrs]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    config: ModelConfig,
    tokenizer_fingerprint: str | None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": dataclasses.asdict(config),
    }
    if tokenizer_fingerprint:
        payload["tokenizer_fingerprint"] = tokenizer_fingerprint
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[int, str | None]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("step", 0)), payload.get("tokenizer_fingerprint")


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

    dataset_fingerprint = dataset.tokenizer_fingerprint
    if dataset_fingerprint:
        if not args.tokenizer:
            raise ValueError("Packed dataset encodes tokenizer fingerprint; provide --tokenizer")
        current_fp = Tokenizer.from_file(Path(args.tokenizer)).fingerprint
        if current_fp != dataset_fingerprint:
            raise ValueError("Tokenizer fingerprint mismatch between dataset and tokenizer")
    elif args.tokenizer:
        dataset_fingerprint = Tokenizer.from_file(Path(args.tokenizer)).fingerprint

    training_cfg = train_config.get("training", {})
    micro_batch_size = int(training_cfg.get("micro_batch_size", 8))
    grad_accum = int(training_cfg.get("gradient_accumulation", 1))
    max_steps = int(training_cfg.get("max_steps", 1000))
    grad_clip = float(training_cfg.get("gradient_clip_norm", 1.0))
    mixed_precision = str(training_cfg.get("mixed_precision", "fp32")).lower()

    device = resolve_device(args.device)

    num_workers = int(training_cfg.get("dataloader_workers", 0))
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_tokens,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    data_iter = cycle(dataloader)

    torch.set_float32_matmul_precision("high")

    def configure_precision(
        target: str,
    ) -> tuple[Any, torch.amp.GradScaler, str]:
        requested = target
        if requested == "auto":
            requested = "fp16" if device.type in {"cuda", "mps"} else "fp32"
        if requested not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported mixed_precision='{target}'")

        effective = requested
        scaler = torch.amp.GradScaler(device.type, enabled=False)
        ctx: Any

        if requested == "fp16":
            if device.type == "cuda":
                scaler = torch.amp.GradScaler("cuda", enabled=True)
                ctx = torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16)
            elif device.type == "mps":
                scaler = torch.amp.GradScaler("mps", enabled=True)
                ctx = torch.autocast(device_type="mps", dtype=torch.float16)
            else:
                effective = "fp32"
                ctx = nullcontext()
        elif requested == "bf16":
            if device.type == "cuda":
                ctx = torch.cuda.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            elif device.type == "cpu":
                ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            else:
                effective = "fp32"
                ctx = nullcontext()
        else:
            ctx = nullcontext()

        if effective == "fp32":
            scaler = torch.amp.GradScaler(device.type, enabled=False)
            ctx = nullcontext()

        return ctx, scaler, effective

    autocast_ctx, scaler, precision_used = configure_precision(mixed_precision)

    model = TransformerModel(model_config).to(device)
    if training_cfg.get("grad_checkpointing", False):
        model.enable_gradient_checkpointing()
    optim_cfg = train_config.get("optim", {})
    target_base_lr = _as_float(optim_cfg.get("lr", 3e-4), 3e-4)
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, train_config.get("scheduler", {}), max_steps)

    if mixed_precision != precision_used:
        print(
            f"Mixed precision '{mixed_precision}' not supported on {device.type}, "
            f"falling back to '{precision_used}'"
        )
    else:
        print(f"Using mixed precision '{precision_used}' on {device.type}")

    criterion = nn.CrossEntropyLoss()
    log_cfg = train_config.get("logging", {})
    log_interval = int(log_cfg.get("log_interval", 100))
    ckpt_interval = int(training_cfg.get("checkpoint_interval", 500))
    use_rich = bool(log_cfg.get("rich_progress", True) and Progress and Console)
    console = Console() if use_rich else (Console() if Console else None)
    progress: Progress | None = None
    progress_task: int | None = None
    tokens_per_step = micro_batch_size * grad_accum * seq_len
    loss_ema: float | None = None
    tps_ema: float | None = None

    start_step = 0
    last_ckpt = output_dir / "ckpt-last.pt"
    checkpoint_fp: str | None = None
    if args.resume and Path(args.resume).exists():
        start_step, checkpoint_fp = load_checkpoint(Path(args.resume), model, optimizer, scheduler)
        print(f"Resumed from {args.resume} @ step {start_step}")
    elif last_ckpt.exists():
        start_step, checkpoint_fp = load_checkpoint(last_ckpt, model, optimizer, scheduler)
        print(f"Resumed from {last_ckpt} @ step {start_step}")

    override_scheduler_lr(optimizer, scheduler, target_base_lr)
    group_lr = [group["lr"] for group in optimizer.param_groups]
    print(
        "Loaded scheduler: "
        f"last_epoch={scheduler.last_epoch} "
        f"base_lrs={scheduler.base_lrs} "
        f"group_lr={group_lr} "
        f"target_base_lr={target_base_lr}"
    )

    if checkpoint_fp and dataset_fingerprint and checkpoint_fp != dataset_fingerprint:
        raise ValueError(
            "Checkpoint tokenizer fingerprint does not match dataset/tokenizer fingerprint"
        )

    tokenizer_fingerprint = dataset_fingerprint

    if use_rich:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("loss={task.fields[loss]:.4f}", justify="right"),
            TextColumn("lr={task.fields[lr]:.2e}", justify="right"),
            TextColumn("tok/s={task.fields[tps]:.0f}", justify="right"),
            console=console,
            transient=False,
        )
        progress.start()
        progress_task = progress.add_task(
            "training",
            total=max_steps,
            completed=start_step,
            loss=0.0,
            lr=0.0,
            tps=0.0,
        )

    model.train()
    step = start_step
    interrupted = False
    if console and not use_rich and step:
        console.log(f"Resuming at step {step}")

    try:
        while step < max_steps:
            iter_start = time.perf_counter()
            optimizer.zero_grad()
            accum_loss = 0.0
            for _ in range(grad_accum):
                batch = next(data_iter).to(device)
                inputs, targets = collate_for_training(batch)
                inputs = inputs.to(device)
                targets = targets.to(device)
                with autocast_ctx:
                    logits, _, _ = model(inputs)
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )
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

            iter_time = max(time.perf_counter() - iter_start, 1e-8)
            tokens_per_sec = tokens_per_step / iter_time
            loss_ema = accum_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * accum_loss
            tps_ema = tokens_per_sec if tps_ema is None else 0.9 * tps_ema + 0.1 * tokens_per_sec
            lr = optimizer.param_groups[0]["lr"]

            if use_rich and progress and progress_task is not None:
                progress.update(
                    progress_task,
                    completed=step,
                    loss=loss_ema,
                    lr=lr,
                    tps=tps_ema,
                )

            if (step % log_interval == 0 or step == start_step + 1) and not use_rich:
                msg = (
                    f"step={step}/{max_steps} loss={accum_loss:.4f} "
                    f"lr={lr:.3e} tok/s={tokens_per_sec:.0f}"
                )
                if console:
                    console.log(msg)
                else:
                    print(msg)

            if step % ckpt_interval == 0 or step == max_steps:
                save_checkpoint(
                    output_dir / f"ckpt-step{step:06d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    model_config,
                    tokenizer_fingerprint,
                )
                save_checkpoint(
                    last_ckpt,
                    model,
                    optimizer,
                    scheduler,
                    step,
                    model_config,
                    tokenizer_fingerprint,
                )
                if console:
                    console.log(f"Checkpoint saved at step {step}")

    except KeyboardInterrupt:
        interrupted = True
        if console:
            console.print("[yellow]Training interrupted. Saving checkpoint...[/yellow]")
        else:
            print("Training interrupted. Saving checkpoint...")
    finally:
        if use_rich and progress:
            progress.stop()

    if interrupted:
        save_checkpoint(
            output_dir / f"ckpt-step{step:06d}-interrupt.pt",
            model,
            optimizer,
            scheduler,
            step,
            model_config,
            tokenizer_fingerprint,
        )
        save_checkpoint(
            last_ckpt,
            model,
            optimizer,
            scheduler,
            step,
            model_config,
            tokenizer_fingerprint,
        )
        if console:
            console.print(
                f"[green]Checkpoint saved at step {step}. Resume with --resume {last_ckpt}[/green]"
            )
        else:
            print(f"Checkpoint saved at step {step}. Resume with --resume {last_ckpt}")
    else:
        if console:
            console.print("[green]Training complete.[/green]")
        else:
            print("Training complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the a-lm transformer")
    parser.add_argument("--model", required=True, help="Model config YAML path")
    parser.add_argument("--train", required=True, help="Training config YAML path")
    parser.add_argument("--data", required=True, help="Packed dataset directory")
    parser.add_argument("--out", required=True, help="Output directory for checkpoints")
    parser.add_argument("--device", default="auto", help="Device to train on (auto/mps/cuda/cpu)")
    parser.add_argument("--resume", help="Optional checkpoint path to resume from")
    parser.add_argument("--tokenizer", help="Tokenizer JSON path for fingerprint validation")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
