#!/usr/bin/env python3
"""Supervised fine-tuning loop for a-lm."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import math
import os
import re
import shutil
import time
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from alm.data.sft_dataset import PackedSFTDataset
from alm.model.config import DualFFNConfig, ModelConfig
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


_ADAMW_SUPPORTS_FUSED = "fused" in inspect.signature(torch.optim.AdamW).parameters


def configure_torch_runtime(device: torch.device) -> None:
    """Best-effort performance knobs per device.

    Keep this conservative: enable speedups on CUDA without changing configs.
    """

    torch.set_float32_matmul_precision("high")
    if device.type != "cuda":
        return

    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None:
        matmul = getattr(cuda_backend, "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = True
        if matmul is not None and hasattr(matmul, "fp32_precision"):
            with suppress(Exception):  # pragma: no cover - depends on torch build
                matmul.fp32_precision = "tf32"

        for fn_name in (
            "enable_flash_sdp",
            "enable_mem_efficient_sdp",
            "enable_math_sdp",
            "enable_cudnn_sdp",
        ):
            fn = getattr(cuda_backend, fn_name, None)
            if callable(fn):
                with suppress(Exception):  # pragma: no cover - depends on torch build
                    fn(True)

    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32"):
        cudnn.allow_tf32 = True


def maybe_compile_model(model: nn.Module, device: torch.device) -> nn.Module:
    if device.type != "cuda":
        return model
    if os.getenv("ALM_TORCH_COMPILE", "1").lower() in {"0", "false", "no"}:
        return model
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        return model
    # Default to the safer Inductor mode. "reduce-overhead" enables CUDA graphs and can
    # crash when tensors escape the compiled region (common in training loops).
    mode = os.getenv("ALM_TORCH_COMPILE_MODE", "default").strip() or "default"
    try:
        compiled = compiler(model, backend="inductor", mode=mode)
    except Exception as error:  # pragma: no cover - depends on torch build
        print(f"[compile] disabled (failed): {error}")
        return model
    print(f"[compile] enabled backend=inductor mode={mode}")
    return compiled


def maybe_mark_step_begin(device: torch.device) -> None:
    if device.type != "cuda":
        return
    compiler = getattr(torch, "compiler", None)
    mark = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(mark):
        mark()


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    if device.type != "cuda":
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=True)


def _apply_adamw_perf_flags(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not _ADAMW_SUPPORTS_FUSED:
        return
    move_optimizer_state_to_device(optimizer, device)
    for group in optimizer.param_groups:
        group["fused"] = True


def build_optimizer(
    model: nn.Module, cfg: dict[str, Any], *, device: torch.device
) -> torch.optim.Optimizer:
    betas_cfg = cfg.get("betas", (0.9, 0.95))
    if isinstance(betas_cfg, (list, tuple)) and len(betas_cfg) == 2:
        betas = (float(betas_cfg[0]), float(betas_cfg[1]))
    else:
        betas = (0.9, 0.95)
    kwargs: dict[str, Any] = {}
    if device.type == "cuda" and _ADAMW_SUPPORTS_FUSED:
        kwargs["fused"] = True
    return torch.optim.AdamW(
        model.parameters(),
        lr=_as_float(cfg.get("lr", 5e-5), 5e-5),
        betas=betas,
        eps=_as_float(cfg.get("eps", 1e-8), 1e-8),
        weight_decay=_as_float(cfg.get("weight_decay", 0.1), 0.1),
        **kwargs,
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
            return (step + 1) / max(1, warmup_steps)
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
        if scheduler.last_epoch < 0:
            factor = 1.0
        else:
            factor = float(scheduler.lr_lambdas[0](scheduler.last_epoch))
    except Exception:  # pragma: no cover - defensive
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
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        config=config,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    atomic_torch_save(payload, path)


_CKPT_STEP_RE = re.compile(r"^ckpt-step(?P<step>\d{6})\.pt$")


def _unwrap_compiled_model(model: nn.Module) -> nn.Module:
    orig_mod = getattr(model, "_orig_mod", None)
    if isinstance(orig_mod, nn.Module):
        return orig_mod
    return model


def _strip_compile_prefix(state: dict[str, Any], prefix: str = "_orig_mod.") -> dict[str, Any]:
    if not state:
        return state
    if not any(isinstance(key, str) and key.startswith(prefix) for key in state):
        return state
    stripped: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(key, str) and key.startswith(prefix):
            stripped[key[len(prefix) :]] = value
        else:
            stripped[key] = value
    return stripped


def build_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    config: ModelConfig,
    tokenizer_fingerprint: str | None,
) -> dict[str, Any]:
    base_model = _unwrap_compiled_model(model)
    payload: dict[str, Any] = {
        "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": dataclasses.asdict(config),
    }
    if tokenizer_fingerprint:
        payload["tokenizer_fingerprint"] = tokenizer_fingerprint
    return payload


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        with suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def checkpoint_keep_steps() -> int:
    """How many step checkpoints to retain (negative = keep all)."""
    raw = os.getenv("ALM_KEEP_STEP_CHECKPOINTS", "").strip()
    if not raw:
        return 8
    try:
        return int(raw)
    except ValueError:
        return 8


def prune_step_checkpoints(output_dir: Path, *, keep: int) -> None:
    if keep < 0:
        return
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("ckpt-step*.pt"):
        match = _CKPT_STEP_RE.match(path.name)
        if match is None:
            continue
        checkpoints.append((int(match.group("step")), path))
    checkpoints.sort(key=lambda item: item[0])
    to_delete = checkpoints if keep == 0 else checkpoints[: max(0, len(checkpoints) - keep)]
    for _, path in to_delete:
        with suppress(FileNotFoundError):
            path.unlink()


def save_checkpoint_set(
    *,
    output_dir: Path,
    last_ckpt: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    config: ModelConfig,
    tokenizer_fingerprint: str | None,
) -> None:
    keep_steps = checkpoint_keep_steps()
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        config=config,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )

    if keep_steps >= 0:
        prune_step_checkpoints(output_dir, keep=max(0, keep_steps - 1))

    try:
        atomic_torch_save(payload, last_ckpt)
        if keep_steps != 0:
            step_ckpt = output_dir / f"ckpt-step{step:06d}.pt"
            with suppress(FileNotFoundError):
                step_ckpt.unlink()
            try:
                os.link(last_ckpt, step_ckpt)
            except OSError:
                atomic_torch_save(payload, step_ckpt)
    except Exception as error:
        try:
            free = shutil.disk_usage(str(output_dir)).free / (1024**3)
            print(f"[ckpt] save failed (free={free:.2f}GiB): {error}")
        except Exception:
            print(f"[ckpt] save failed: {error}")
        raise

    if keep_steps >= 0:
        prune_step_checkpoints(output_dir, keep=keep_steps)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[int, str | None]:
    payload = torch.load(path, map_location="cpu")
    model_state = payload["model"]
    if isinstance(model_state, dict):
        model_state = _strip_compile_prefix(model_state)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("step", 0)), payload.get("tokenizer_fingerprint")


def load_model_weights(path: Path, model: nn.Module) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model") if isinstance(payload, dict) else None
    if state is None:
        raise ValueError(f"Checkpoint at {path} missing 'model' state")
    if isinstance(state, dict):
        state = _strip_compile_prefix(state)
    model.load_state_dict(state)


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.stack([item[0] for item in batch], dim=0)
    mask = torch.stack([item[1] for item in batch], dim=0)
    return tokens, mask


def create_scaler(device: torch.device) -> torch.amp.GradScaler | None:
    try:
        return torch.amp.GradScaler(
            device.type,
            enabled=device.type in {"cuda", "mps"},
            init_scale=2.0**10,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=200,
        )
    except TypeError:  # pragma: no cover - fallback for older torch signature
        return torch.amp.GradScaler(device.type, enabled=device.type in {"cuda", "mps"})
    except AttributeError:  # pragma: no cover - very old torch without amp
        return None


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = load_model_config(Path(args.model))
    train_config = load_train_config(Path(args.train))

    device = resolve_device(args.device)
    configure_torch_runtime(device)

    dataset = PackedSFTDataset(Path(args.data))
    training_cfg = train_config.get("training", {})
    micro_batch_size = int(training_cfg.get("micro_batch_size", 8))
    grad_accum = int(training_cfg.get("gradient_accumulation", 8))
    max_steps = int(training_cfg.get("max_steps", 2000))
    grad_clip = float(training_cfg.get("gradient_clip_norm", 1.0))
    num_workers = int(training_cfg.get("dataloader_workers", 0))

    dataset_fingerprint = dataset.tokenizer_fingerprint
    if dataset.tokenizer_fingerprint:
        if not args.tokenizer:
            raise ValueError(
                "Packed dataset encodes tokenizer fingerprint; "
                "provide --tokenizer to verify compatibility."
            )
        current_fp = Tokenizer.from_file(Path(args.tokenizer)).fingerprint
        if current_fp != dataset.tokenizer_fingerprint:
            raise ValueError("Tokenizer fingerprint mismatch between dataset and current tokenizer")
    elif args.tokenizer:
        dataset_fingerprint = Tokenizer.from_file(Path(args.tokenizer)).fingerprint

    prefetch_factor = int(training_cfg.get("prefetch_factor", 2))
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": micro_batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_batch,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    dataloader = DataLoader(**loader_kwargs)
    warmup_batch: tuple[torch.Tensor, torch.Tensor] | None = None
    try:
        warmup_batch = next(iter(dataloader))
    except StopIteration:
        warmup_batch = None
    data_iter = iter(dataloader)

    model = TransformerModel(model_config).to(device)
    optim_cfg = train_config.get("optim", {})
    scheduler_cfg = train_config.get("scheduler", {})
    optimizer = build_optimizer(model, optim_cfg, device=device)
    scheduler = build_scheduler(optimizer, scheduler_cfg, max_steps)

    scaler = create_scaler(device)
    if device.type == "mps" and os.getenv("PYTORCH_MPS_FAST_MATH"):
        print("[warn] PYTORCH_MPS_FAST_MATH=1 detected; unset it for SFT stability on MPS.")
    print(f"[amp] device={device.type} scaler_enabled={bool(scaler and scaler.is_enabled())}")

    def _autocast_ctx() -> Any:
        amp = getattr(torch, "amp", None)
        if amp is not None and hasattr(amp, "autocast"):
            if device.type in {"cuda", "mps"}:
                return amp.autocast(device_type=device.type, dtype=torch.float16)
            return nullcontext()

        if device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        if device.type == "mps" and hasattr(torch, "autocast"):
            return torch.autocast(device_type="mps", dtype=torch.float16)
        return nullcontext()

    autocast_ctx = _autocast_ctx()

    last_ckpt = output_dir / "ckpt-last.pt"
    start_step = 0
    checkpoint_fp: str | None = None
    if args.resume and Path(args.resume).exists():
        start_step, checkpoint_fp = load_checkpoint(Path(args.resume), model, optimizer, scheduler)
        print(f"Resumed from {args.resume} @ step {start_step}")
    elif last_ckpt.exists():
        start_step, checkpoint_fp = load_checkpoint(last_ckpt, model, optimizer, scheduler)
        print(f"Resumed from {last_ckpt} @ step {start_step}")
    elif args.init:
        load_model_weights(Path(args.init), model)
        print(f"Loaded initial weights from {args.init}")

    _apply_adamw_perf_flags(optimizer, device)
    model = maybe_compile_model(model, device)

    if checkpoint_fp and dataset_fingerprint and checkpoint_fp != dataset_fingerprint:
        raise ValueError(
            "Checkpoint tokenizer fingerprint does not match dataset/tokenizer fingerprint"
        )

    base_lr = _as_float(optim_cfg.get("lr", 5e-5), 5e-5)
    override_scheduler_lr(optimizer, scheduler, base_lr)
    print(
        f"[lr] base={base_lr:.2e} last_epoch={scheduler.last_epoch} "
        f"group_lrs={[group['lr'] for group in optimizer.param_groups]}"
    )

    tokenizer_fingerprint = dataset_fingerprint

    if warmup_batch is not None and device.type in {"mps", "cuda"}:
        warm_tokens, _ = warmup_batch
        warm_tokens = warm_tokens[:1, : min(128, warm_tokens.size(1))]
        non_blocking = device.type == "cuda"
        warm_tokens = warm_tokens.to(device=device, dtype=torch.long, non_blocking=non_blocking)
        with torch.no_grad(), autocast_ctx:
            maybe_mark_step_begin(device)
            model(warm_tokens[:, :-1])

    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)
    model.train()

    ema_loss: float | None = None
    ema_tps: float | None = None
    step = start_step
    tokens_per_step = dataset.seq_len * micro_batch_size * grad_accum

    target_frac: float | None = None

    try:
        while step < max_steps:
            iter_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(grad_accum):
                try:
                    tokens, loss_mask = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    tokens, loss_mask = next(data_iter)

                non_blocking = device.type == "cuda"
                tokens = tokens.to(device, dtype=torch.long, non_blocking=non_blocking)
                loss_mask = loss_mask.to(device, non_blocking=non_blocking)
                target_mask = loss_mask[:, 1:]
                inputs = tokens[:, :-1]
                targets = tokens[:, 1:].masked_fill(~target_mask, -100)
                target_frac = float(target_mask.sum()) / max(1, target_mask.numel())

                with autocast_ctx:
                    maybe_mark_step_begin(device)
                    logits, _, _ = model(inputs)
                    if device.type == "cuda":
                        logits_for_loss = logits
                    else:
                        logits_for_loss = logits.float().clamp_(-30, 30)
                    loss = criterion(
                        logits_for_loss.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )
                loss = loss / grad_accum

                if scaler and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += float(loss.detach().item())

            if scaler and scaler.is_enabled():
                scaler.unscale_(optimizer)
            total_norm = nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=False
            )

            if scaler and scaler.is_enabled():
                prev_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                new_scale = float(scaler.get_scale())
                if new_scale < prev_scale:
                    print(
                        f"[amp] overflow detected; loss_scale {prev_scale:.0f} -> {new_scale:.0f}"
                    )
            else:
                if not torch.isfinite(total_norm):
                    print("Non-finite grad norm detected; skipping step (fp32 path)")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
            scheduler.step()
            step += 1

            iter_time = max(time.perf_counter() - iter_start, 1e-6)
            tokens_per_sec = tokens_per_step / iter_time
            ema_loss = accum_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * accum_loss
            ema_tps = tokens_per_sec if ema_tps is None else 0.9 * ema_tps + 0.1 * tokens_per_sec
            lr = optimizer.param_groups[0]["lr"]

            if (
                step % int(train_config.get("logging", {}).get("log_interval", 10)) == 0
                or step == start_step + 1
            ):
                extra = f" mask={target_frac:.2f}" if target_frac is not None else ""
                message = (
                    f"step={step}/{max_steps} loss={ema_loss:.4f} "
                    f"lr={lr:.3e} tok/s={ema_tps:.0f}{extra}"
                )
                print(message)

            ckpt_interval = int(training_cfg.get("checkpoint_interval", 1000))
            if step % ckpt_interval == 0 or step == max_steps:
                save_checkpoint_set(
                    output_dir=output_dir,
                    last_ckpt=last_ckpt,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    config=model_config,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                )
                print(f"Checkpoint saved at step {step}")

    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint...")
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            config=model_config,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        atomic_torch_save(payload, output_dir / f"ckpt-step{step:06d}-interrupt.pt")
        atomic_torch_save(payload, last_ckpt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the a-lm transformer")
    parser.add_argument("--model", required=True, help="Model config YAML path")
    parser.add_argument("--train", required=True, help="Training config YAML path")
    parser.add_argument("--data", required=True, help="Packed SFT dataset directory")
    parser.add_argument("--out", required=True, help="Output directory for checkpoints")
    parser.add_argument("--device", default="auto", help="Device to train on (auto/mps/cuda/cpu)")
    parser.add_argument("--resume", help="Checkpoint path to resume SFT training")
    parser.add_argument("--init", help="Checkpoint providing initial model weights")
    parser.add_argument("--tokenizer", help="Tokenizer JSON path for fingerprint validation")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
