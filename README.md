# a-lm (Alam Language Model)

(Disclaimer: AI-written readme)

`a-lm` is a from-scratch small language model project that builds a custom tokenizer, efficient Transformer backbone, training pipeline, and inference demo suitable for showcasing on a portfolio. The focus is on understanding every piece – from text preprocessing through alignment and deployment – while keeping the stack lean enough to run on an Apple Silicon laptop.

## What gets built
- Custom byte-fallback tokenizer (BPE + optional unigram) trained on curated English corpora.
- Decoder-only Transformer with RMSNorm, SwiGLU, RoPE/ALiBi, grouped attention, and a dual FFN router.
- Data preparation scripts that download, clean, and pack high-quality open datasets (FineWeb-Edu slice, TinyStories, filtered UltraChat, etc.).
- Training loop for pretraining, supervised fine-tuning, and preference tuning (DPO) with checkpoints and sample generation.
- Inference runtime featuring constrained decoding, INT8 weight-only quantization, a FastAPI streaming API, and a lightweight chat widget.

## Roadmap snapshot
Development proceeds milestone by milestone:
1. **Scaffold & tooling** – package layout, CI, configuration. ✅
2. **Tokenizer** – byte normalization, BPE/Unigram trainers, CLI + tests. ✅
3. **Model core** – attention stack, dual FFN gate, Transformer assembly. ✅
4. **Data pipeline** – dataset download/cleaning, token packing. ⏳
5. **Pretraining loop** – optimizer, scheduler, gradient checkpointing, logging.
6. **Alignment** – SFT followed by lightweight DPO.
7. **Inference & quantization** – INT8 modules, constrained decoding, FastAPI server.
8. **Evaluation & demo** – lm-eval harness, streaming widget, documentation polish.

Reference configurations live under `configs/`:
- `configs/corpus.yaml` enumerates dataset sources and cache expectations.
- `configs/pico.yaml` and `configs/train.yaml` capture the initial model and training hyperparameters for a ~29M parameter “pico” run.

Tokenizer and model layers are available today:
- Train the tokenizer via `python scripts/train_tokenizer.py --input data/corpus/*.txt --vocab-size 32000 --out artifacts/tokenizer.json` (BPE default). A Unigram variant is exposed through `alm.tokenizers.train_unigram` for experiments.
- Tokenizer modules live in `src/alm/tokenizers/` with coverage in `tests/tokenizers/`.
- Core Transformer components (RMSNorm, SwiGLU, RoPE/ALiBi, grouped attention, dual FFN, `TransformerModel`) reside in `src/alm/model/` with shape and cache tests in `tests/model/`.

## Dataset notes
Initial data work focuses on high-quality, permissively licensed corpora. FineWeb-Edu (`https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu`) is the primary web slice; TinyStories and filtered chat datasets supplement it.

### One-time prerequisites
1. Create (or use) a Hugging Face account.
2. Generate a read token in your HF account settings.
3. Log in locally so the `datasets` library can stream private endpoints:
   ```bash
   huggingface-cli login
   # or export HF_TOKEN=... before running the scripts
   ```

### Prepare text corpora
```
# 1) Clean and normalize sources defined in configs/corpus.yaml
python scripts/prepare_corpus.py \
  --src configs/corpus.yaml \
  --out data/clean

# 2) Train (or reuse) the tokenizer
python scripts/train_tokenizer.py \
  --input data/clean/*.txt \
  --vocab-size 32000 \
  --out artifacts/tokenizer.json

# 3) Pack cleaned text into token shards
python scripts/pack_dataset.py \
  --tokenizer artifacts/tokenizer.json \
  --in data/clean \
  --out data/packed \
  --seq-len 512 \
  --shard-size 2048
```

Resulting directories:
- `data/clean/` – normalized `.txt` files + source metadata.
- `artifacts/tokenizer.json` – tokenizer vocab produced by step (2).
- `data/packed/` – contiguous `.bin` shards + `metadata.json` describing sequence length, shard count, and total tokens.

When you want to try a different dataset (e.g., your own export), create a new entry in `configs/corpus.yaml`, rerun the commands above, and point future training runs at the freshly packed directory.

## Current status
Planning and foundational code are in place. Tokenizer tooling and the Transformer backbone are implemented with tests; the next milestone will deliver data preparation scripts and sharded token packs so pretraining can begin.

## Contributing / running along
- Target hardware: Apple Silicon (M2) with PyTorch MPS acceleration; CPU fallbacks remain supported for collaborators.
- `Makefile` targets (`make dev`, `make lint`, `make test`, `make check-mps`) streamline setup.
- Keep `TODO_LIST.md` up to date as you progress through milestones; CI (`.github/workflows/ci.yml`) runs lint + tests on pushes and pull requests.
- Never commit secrets – dataset credentials and tokens belong in a local `.env` (ignored by git).

The README will expand with setup instructions, dataset steps, and evaluation results as each milestone lands.
