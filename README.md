# a-lm (Alam Language Model)

(Disclaimer: AI-written README)

> *From-scratch small language model stack, built out of boredom and curiosity 🦭.*

## Table of Contents
1. [Project Summary](#project-summary)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Essential Commands](#essential-commands)
5. [Dataset Preparation](#dataset-preparation)
6. [Tokenizer Training](#tokenizer-training)
7. [Token Packing](#token-packing)
8. [Pretraining Loop](#pretraining-loop)
9. [Sampling From Checkpoints](#sampling-from-checkpoints)
10. [Repository Layout](#repository-layout)
11. [Testing & Linting](#testing--linting)
12. [Troubleshooting](#troubleshooting)
13. [Next Steps](#next-steps)

---

## Project Summary
`a-lm` walks through every layer required to build a compact decoder-only transformer: custom tokenizer, curated dataset pipeline, Apple-friendly training loop, weight-only quantization, and a streaming chat demo (coming in later phases). The code prioritizes clarity and reproducibility so you can showcase it on a resume and continue expanding the stack.

Highlights:
- Byte-fallback tokenizer (BPE + optional Unigram) trained on curated corpora.
- Transformer core with RMSNorm, SwiGLU, RoPE/ALiBi, grouped attention, dual FFN router.
- Packed dataset format for efficient streaming, simple PyTorch training loop with checkpoint/resume.
- Sampling CLI to inspect checkpoints mid-training.

---

## Prerequisites
- **Python** 3.11 or newer (use `pyenv`, `asdf`, or system Python).
- **Apple Silicon** with macOS 13+ recommended (MPS acceleration). CUDA works too.
- **Hugging Face account** with read token to access FineWeb-Edu and related datasets.

Optional but helpful:
- Homebrew packages (`git`, `llvm`, etc.).
- Adequate disk space for datasets/checkpoints (tens of GBs depending on corpora).

---

## Environment Setup
1. Clone the repository.
2. Create and activate a virtualenv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```
3. Install project + dev dependencies:
   ```bash
   pip install -e .[dev]
   pre-commit install
   ```
4. Log into Hugging Face (once per machine):
   ```bash
   huggingface-cli login
   # or export HF_TOKEN=... in your shell/profile
   ```

---

## Essential Commands
| Command | Purpose |
|---|---|
| `make dev` | Install dependencies and pre-commit hooks. |
| `make lint` / `ruff check --fix .` | Lint + autofix using Ruff. |
| `make test` / `pytest` | Run the test suite. |
| `ruff format .` | Apply code formatting. |
| `make check-mps` | Print which device (MPS/CUDA/CPU) PyTorch sees. |

> **Always run:** `ruff check --fix .`, `ruff format .`, and `pytest` before committing changes.

---

## Dataset Preparation
1. Ensure Hugging Face login is active (`huggingface-cli login`).
2. Inspect `configs/corpus.yaml`. By default it references:
   - FineWeb-Edu (quality web text slice)
   - Wikipedia snapshot (English)
   - TinyStories (tiny synthetic stories)
   - UltraChat/OASST (instruction data for later SFT)
3. Run the prep script to clean and normalize all active sources:
   ```bash
   python scripts/prepare_corpus.py \
     --src configs/corpus.yaml \
     --out data/clean
   ```
   - Output: `data/clean/*.txt` plus metadata JSON for each source.
   - You can comment out sources in the YAML if you want a smaller initial run.

---

## Tokenizer Training
4. Train a custom tokenizer (byte fallback BPE):
   ```bash
   python scripts/train_tokenizer.py \
     --input data/clean/*.txt \
     --vocab-size 32000 \
     --out artifacts/tokenizer.json
   ```
   - Produces `artifacts/tokenizer.json` (merge rules, vocab).
   - Unigram trainer exposed via `alm.tokenizers.train_unigram` for experimentation.

---

## Token Packing
5. Pack the cleaned text into fixed-length token shards:
   ```bash
   python scripts/pack_dataset.py \
     --tokenizer artifacts/tokenizer.json \
     --in data/clean \
     --out data/packed \
     --seq-len 512 \
     --shard-size 2048
   ```
   - `data/packed/` now holds `shard_*.bin` (uint32 tokens) and `metadata.json` describing seq length, shard list, and total tokens.
   - Adjust `--seq-len`/`--shard-size` to suit memory constraints.

---

## Pretraining Loop
6. Launch a pico-scale pretraining run (auto-detects device). Recommended for sanity-checks:
   ```bash
   python scripts/train_pretrain.py \
     --model configs/pico.yaml \
     --train configs/train.yaml \
     --data data/packed \
     --out runs/pico-pretrain \
     --device auto
   ```
   - Checkpoints saved to `runs/pico-pretrain/` (`ckpt-stepXXXXXX.pt`, `ckpt-last.pt`).
   - Resume automatically from `--resume path/to/ckpt.pt` or the default `ckpt-last.pt` inside `--out`.
   - Hyperparameters? Edit `configs/train.yaml` (micro batch size, accum steps, grad clip, max steps, logging interval, checkpoint cadence).

---

## Sampling From Checkpoints
7. Inspect generated text mid-training:
   ```bash
   python scripts/sample_text.py \
     --checkpoint runs/pico-pretrain/ckpt-last.pt \
     --tokenizer artifacts/tokenizer.json \
     --prompt "Hello" \
     --max-tokens 50 \
     --temperature 0.8 \
     --top-k 40 \
     --device auto
   ```
   - Supports top-k + temperature sampling; set `--top-k 0` for greedy decoding.
   - Swap prompts to gauge knowledge/fluency as training progresses.

---

## Repository Layout
```
.
├─ configs/          # YAML configs (corpus sources, model sizes, training hyperparams)
├─ scripts/          # CLI entrypoints (tokenizer, corpus prep, packing, training, sampling)
├─ src/alm/          # Library code: tokenizers, model, data, utils, etc.
├─ tests/            # Pytest suites (tokenizer/model/data/training)
├─ data/             # Generated cleaned corpora & packed shards (ignored)
├─ artifacts/        # Tokenizer artifacts, future checkpoints (ignored)
├─ runs/             # Training outputs (ignored)
├─ AGENTS.MD         # Agent playbook
├─ TODO_LIST.md      # Milestone tracker
└─ README.md         # You are here
```

---

## Testing & Linting
- Run locally before committing or pushing:
  ```bash
  ruff check --fix .
  ruff format .
  pytest
  ```
- CI (GitHub Actions) re-runs the same commands; keep them green by testing locally first.

---

## Troubleshooting
- **Hugging Face auth errors:** re-run `huggingface-cli login`, ensure the token has read access.
- **OOM during training:** lower `micro_batch_size`, increase `gradient_accumulation`, or reduce `--seq-len` when packing.
- **Slow/broken run:** verify `pip install -e .[dev]` succeeded, run `make check-mps` to confirm device.
- **Restart training:** delete or move `ckpt-last.pt` if you want a fresh start; otherwise training resumes automatically.

---

## Next Steps
- Phase 5 (coming soon): supervised fine-tuning on chat data + DPO alignment.
- Phase 6+: inference server (FastAPI), weight-only INT8 quantization, web chat widget.
- Keep `TODO_LIST.md` updated as milestones close; expand this README when new CLIs/configs appear.
