# a-lm (Alam Language Model)
(AI-assisted readme)

`a-lm` is a from-scratch small language model project; this was made just for fun/curiosity :). The end goal is to train a compact decoder-only Transformer on curated open datasets, align it for chat use, and ship a streaming FastAPI endpoint plus a lightweight web widget you can embed in a portfolio.

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

Progress and open questions live in `TODO_LIST.md`. Implementation practices, commands, and repo conventions are in `AGENTS.MD`.

## Current status
Planning stage. No code scaffold yet—first milestones will establish the Python package, tooling, and dataset acquisition flow before moving into tokenizer/model implementation.
