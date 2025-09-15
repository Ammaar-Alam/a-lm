# a-lm (Alam Language Model)

`a-lm` is a from-scratch small language model project that turns the blueprint in `PROJECT_OVERVIEW.md` into a working, resume-ready system. The end goal is to train a compact decoder-only Transformer on curated open datasets, align it for chat use, and ship a streaming FastAPI endpoint plus a lightweight web widget you can embed in a portfolio.

## What gets built
- Custom byte-fallback tokenizer (BPE + optional unigram) trained on a curated English corpus for readable output.
- Efficient Transformer stack (RMSNorm, SwiGLU, RoPE/ALiBi, grouped attention, dual FFN routing) optimized for Apple M2 hardware but portable to CPU.
- Data preparation scripts to download, clean, and pack small high-quality datasets (e.g., FineWeb-Edu slice, TinyStories, filtered UltraChat).
- Training pipeline for pretraining, supervised fine-tuning, and light preference tuning (DPO), with checkpoints saved for sharing.
- Inference runtime with constrained decoding, INT8 weight-only quantization, and a FastAPI streaming API, paired with a minimal chat widget.

## How development proceeds
Work follows the milestone ladder in `PROJECT_OVERVIEW.md`:
1. **M0 Scaffold:** repo tooling, packaging, Apple MPS detection.
2. **M1 Tokenizer → M8 Demo:** tokenizer, model core, data pipeline, training, alignment, inference, evaluation.
3. Optional stretch items (Core ML/GGUF export, speculative decoding, RAG) come after the MVP chat loop works.

Reference configurations now live under `configs/`:
- `configs/corpus.yaml` documents curated dataset sources and caching expectations.
- `configs/pico.yaml` and `configs/train.yaml` capture the initial model + training hyperparameters for the pico run.

Tokenizer work has begun:
- Train via `python scripts/train_tokenizer.py --input data/corpus/*.txt --vocab-size 32000 --out artifacts/tokenizer.json` (BPE default).
- Unigram variant available via `amlm.tokenizers.train_unigram` for experimentation.
- Tokenizer modules live in `src/amlm/tokenizers/` with pytest coverage in `tests/tokenizers/`.

Model core primitives are implemented under `src/amlm/model/`:
- RMSNorm, SwiGLU, RoPE/ALiBi utilities with unit tests in `tests/model/`.
- `attention.py` provides grouped/multi-query attention with KV-cache support.
- `dual_ffn.py` implements the lite/expert gate with routing stats; `transformer.py` wires layers and configuration (`config.py`).

## Current status
Planning + core scaffolding complete. Tokenizer and base Transformer components are implemented with tests; next phases add data prep, training loops, alignment, and serving.

## Contributing / running along
- Primary development target: Apple Silicon (M2) using PyTorch MPS; CPU-only fallbacks will be maintained so others can clone and experiment.
- Community contributors will be able to reproduce training locally by following the forthcoming `Makefile` and `scripts/` commands documented in the README.
- For questions about datasets or optional features, see the “Outstanding Questions” section in `TODO_LIST.md`.

This README will expand as phases complete (setup instructions, commands, evaluation results). For the full technical blueprint, read `PROJECT_OVERVIEW.md`.
