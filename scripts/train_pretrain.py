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
from typing import Any, TextIO

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


def describe_device(device: torch.device) -> str:
    torch_version = getattr(torch, "__version__", "unknown")
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        mem_gib = props.total_memory / (1024**3)
        capability = f"{props.major}.{props.minor}"
        return (
            f"torch={torch_version} device=cuda:{index} "
            f"name={props.name} cc={capability} vram={mem_gib:.1f}GiB "
            f"cuda_runtime={torch.version.cuda} devices={torch.cuda.device_count()}"
        )
    if device.type == "mps":
        built = torch.backends.mps.is_built()
        available = torch.backends.mps.is_available()
        return f"torch={torch_version} device=mps built={built} available={available}"
    threads = torch.get_num_threads()
    return f"torch={torch_version} device=cpu threads={threads}"


def open_log_file(output_dir: Path, log_cfg: dict[str, Any]) -> tuple[TextIO | None, Path | None]:
    log_path = log_cfg.get("log_file")
    if not isinstance(log_path, str) or not log_path.strip():
        return None, None

    mode = str(log_cfg.get("log_file_mode", "a")).lower().strip()
    if mode not in {"a", "w"}:
        mode = "a"

    path = Path(log_path)
    if not path.is_absolute():
        path = output_dir / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode=mode, encoding="utf-8", buffering=1), path


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


def generate_sample(
    model: TransformerModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stop_strings: list[str],
    device: torch.device,
    autocast_ctx: Any,
) -> str:
    model_was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            prompt_tokens = tokenizer.encode(prompt)
            max_context = model.config.max_position_embeddings
            context_budget = max_context - max_tokens
            if context_budget <= 0:
                context_budget = max_context
            if len(prompt_tokens) > context_budget:
                prompt_tokens = prompt_tokens[-context_budget:]

            generated: list[int] = list(prompt_tokens)
            input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            past = None
            prompt_len = len(prompt_tokens)
            for _ in range(max_tokens):
                with autocast_ctx:
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
                if stop_strings:
                    completion = tokenizer.decode(generated[prompt_len:])
                    for stop in stop_strings:
                        stop_idx = completion.find(stop)
                        if stop_idx != -1:
                            return tokenizer.decode(prompt_tokens) + completion[:stop_idx]
            return tokenizer.decode(generated)
    finally:
        if model_was_training:
            model.train()


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
    log_cfg = train_config.get("logging", {})

    dataset = PackedDataset(Path(args.data))
    seq_len = dataset.seq_len
    if seq_len < 2:
        raise ValueError("Sequence length must be at least 2 for language modeling")

    dataset_fingerprint = dataset.tokenizer_fingerprint
    tokenizer: Tokenizer | None = None
    if dataset_fingerprint:
        if not args.tokenizer:
            raise ValueError("Packed dataset encodes tokenizer fingerprint; provide --tokenizer")
        tokenizer = Tokenizer.from_file(Path(args.tokenizer))
        if tokenizer.fingerprint != dataset_fingerprint:
            raise ValueError("Tokenizer fingerprint mismatch between dataset and tokenizer")
    elif args.tokenizer:
        tokenizer = Tokenizer.from_file(Path(args.tokenizer))
        dataset_fingerprint = tokenizer.fingerprint

    training_cfg = train_config.get("training", {})
    micro_batch_size = int(training_cfg.get("micro_batch_size", 8))
    grad_accum = int(training_cfg.get("gradient_accumulation", 1))
    max_steps = int(training_cfg.get("max_steps", 1000))
    grad_clip = float(training_cfg.get("gradient_clip_norm", 1.0))
    mixed_precision = str(training_cfg.get("mixed_precision", "fp32")).lower()

    device = resolve_device(args.device)
    log_file, log_file_path = open_log_file(output_dir, log_cfg)
    log_flush = bool(log_cfg.get("log_file_flush", True))

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
    log_interval = int(log_cfg.get("log_interval", 100))
    ckpt_interval = int(training_cfg.get("checkpoint_interval", 500))
    log_grad_norm = bool(log_cfg.get("log_grad_norm", False))
    ema_beta = _as_float(log_cfg.get("ema_beta", 0.9), 0.9)
    ema_beta = max(0.0, min(ema_beta, 0.999))
    loss_beta = _as_float(log_cfg.get("loss_ema_beta", ema_beta), ema_beta)
    loss_beta = max(0.0, min(loss_beta, 0.999))
    tps_beta = _as_float(log_cfg.get("tps_ema_beta", ema_beta), ema_beta)
    tps_beta = max(0.0, min(tps_beta, 0.999))

    sample_interval = int(log_cfg.get("sample_interval", 0))
    sample_at_start = bool(log_cfg.get("sample_at_start", False))
    sample_prompt = str(log_cfg.get("sample_prompt", "Hello"))
    sample_max_tokens = max(0, int(log_cfg.get("sample_max_tokens", 80)))
    sample_temperature = _as_float(log_cfg.get("sample_temperature", 0.8), 0.8)
    sample_top_k = int(log_cfg.get("sample_top_k", 40))
    sample_top_p = _as_float(log_cfg.get("sample_top_p", 0.95), 0.95)
    sample_repetition_penalty = _as_float(log_cfg.get("sample_repetition_penalty", 1.1), 1.1)
    stop_cfg = log_cfg.get("sample_stop", [])
    sample_stop_strings: list[str] = []
    if isinstance(stop_cfg, list):
        sample_stop_strings = [str(item) for item in stop_cfg if item is not None]

    use_rich = bool(log_cfg.get("rich_progress", True) and Progress and Console)
    console = Console() if use_rich else (Console() if Console else None)
    progress: Progress | None = None
    progress_task: int | None = None
    tokens_per_step = micro_batch_size * grad_accum * seq_len
    loss_ema: float | None = None
    tps_ema: float | None = None
    grad_norm: float | None = None

    def _timestamp() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def log_event(message: str) -> None:
        if log_file:
            log_file.write(f"{_timestamp()} {message}\n")
            if log_flush:
                log_file.flush()
        if console:
            console.log(message)
        else:
            print(message)

    def log_metrics(message: str) -> None:
        if log_file:
            log_file.write(f"{_timestamp()} {message}\n")
            if log_flush:
                log_file.flush()
        if not use_rich:
            if console:
                console.log(message)
            else:
                print(message)

    start_step = 0
    last_ckpt = output_dir / "ckpt-last.pt"
    checkpoint_fp: str | None = None
    if args.resume and Path(args.resume).exists():
        start_step, checkpoint_fp = load_checkpoint(Path(args.resume), model, optimizer, scheduler)
        log_event(f"Resumed from {args.resume} @ step {start_step}")
    elif last_ckpt.exists():
        start_step, checkpoint_fp = load_checkpoint(last_ckpt, model, optimizer, scheduler)
        log_event(f"Resumed from {last_ckpt} @ step {start_step}")

    override_scheduler_lr(optimizer, scheduler, target_base_lr)
    group_lr = [group["lr"] for group in optimizer.param_groups]
    log_event(
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

    if log_file_path:
        log_event(f"Logging to {log_file_path}")
    log_event(describe_device(device))
    log_event(
        "Run config: "
        f"seq_len={seq_len} micro_batch={micro_batch_size} grad_accum={grad_accum} "
        f"tokens_per_step={tokens_per_step} max_steps={max_steps} "
        f"precision={precision_used} ema_beta={ema_beta:.3f}"
    )

    if use_rich:
        columns: list[Any] = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("loss={task.fields[loss]:.4f}", justify="right"),
            TextColumn("lr={task.fields[lr]:.2e}", justify="right"),
            TextColumn("tok/s={task.fields[tps]:.0f}", justify="right"),
        ]
        if log_grad_norm:
            columns.append(TextColumn("gn={task.fields[gn]:.2f}", justify="right"))
        progress = Progress(
            *columns,
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
            gn=0.0,
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
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm = float(total_norm.detach().item() if isinstance(total_norm, torch.Tensor) else total_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            step += 1

            iter_time = max(time.perf_counter() - iter_start, 1e-8)
            tokens_per_sec = tokens_per_step / iter_time
            loss_ema = accum_loss if loss_ema is None else loss_beta * loss_ema + (1.0 - loss_beta) * accum_loss
            tps_ema = tokens_per_sec if tps_ema is None else tps_beta * tps_ema + (1.0 - tps_beta) * tokens_per_sec
            lr = optimizer.param_groups[0]["lr"]

            if use_rich and progress and progress_task is not None:
                progress.update(
                    progress_task,
                    completed=step,
                    loss=loss_ema,
                    lr=lr,
                    tps=tps_ema,
                    gn=grad_norm or 0.0,
                )

            if (step % log_interval == 0 or step == start_step + 1) and not use_rich:
                msg = f"step={step}/{max_steps} loss={loss_ema:.4f} lr={lr:.3e} tok/s={tps_ema:.0f}"
                if log_grad_norm and grad_norm is not None:
                    msg += f" gn={grad_norm:.2f}"
                log_metrics(msg)
            elif step % log_interval == 0 or step == start_step + 1:
                msg = f"step={step}/{max_steps} loss={loss_ema:.4f} lr={lr:.3e} tok/s={tps_ema:.0f}"
                if log_grad_norm and grad_norm is not None:
                    msg += f" gn={grad_norm:.2f}"
                if log_file:
                    log_metrics(msg)

            should_sample = sample_interval > 0 and step % sample_interval == 0
            if sample_interval > 0 and sample_at_start and step == start_step + 1:
                should_sample = True
            if should_sample:
                if tokenizer is None and args.tokenizer:
                    tokenizer = Tokenizer.from_file(Path(args.tokenizer))
                if tokenizer is None:
                    log_event("Sample generation requested but no --tokenizer provided; skipping.")
                else:
                    text = generate_sample(
                        model,
                        tokenizer,
                        sample_prompt,
                        sample_max_tokens,
                        sample_temperature,
                        sample_top_k,
                        sample_top_p,
                        sample_repetition_penalty,
                        sample_stop_strings,
                        device,
                        autocast_ctx,
                    )
                    header = f"[sample] step={step} prompt={sample_prompt!r}"
                    if log_file:
                        log_file.write(f"{_timestamp()} {header}\n")
                        for line in text.splitlines() or [""]:
                            log_file.write(f"{_timestamp()} > {line}\n")
                        if log_flush:
                            log_file.flush()
                    if console:
                        console.log(header)
                        console.print(text, markup=False)
                    else:
                        print(header)
                        print(text)

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
                log_event(f"Checkpoint saved at step {step}")

    except KeyboardInterrupt:
        interrupted = True
        log_event("Training interrupted. Saving checkpoint...")
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
        log_event(f"Checkpoint saved at step {step}. Resume with --resume {last_ckpt}")
    else:
        log_event("Training complete.")
    if log_file:
        log_file.close()


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
