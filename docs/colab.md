# Google Colab quickstart

This walkthrough keeps outputs in Google Drive so runs survive session resets.

## 1) Create a Colab notebook
- Runtime → Change runtime type → Hardware accelerator: **GPU**

## 2) Mount Drive and clone the repo
```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/Ammaar-Alam/a-lm.git /content/drive/MyDrive/a-lm
%cd /content/drive/MyDrive/a-lm
```

If the repo is private, upload a zip to Drive and unzip:
```bash
%cd /content/drive/MyDrive
!unzip -q a-lm.zip
%cd /content/drive/MyDrive/a-lm
```

## 3) Install dependencies
```bash
%pip install -e . --no-deps
%pip install -U "huggingface_hub<1.0" "datasets>=2.19,<3" "pyarrow>=15.0.2,<19" \
  "fsspec>=2025.3.0" "gcsfs>=2025.3.0" "tokenizers>=0.22.0,<=0.23.0"
```

Notes:
- This intentionally does **not** downgrade `numpy` (Colab preinstalls expect `numpy>=2`).
- If `%pip` prints a warning about a package that was already imported, restart the runtime once.

## 4) Log into Hugging Face (required for datasets)
```python
from huggingface_hub import login
login()
```

## 5) Start a larger Colab run (nano model + bigger corpus)
If you already have a previous run on disk (or in a zip), you can skip pretraining and jump straight to SFT. The notebook’s “Start pretraining” cell will auto-detect an existing `runs/<RUN>/pretrain/ckpt-last.pt` + `artifacts/<RUN>/tokenizer.json` and reuse it.

```bash
!make colab-pretrain RUN=$(date +%Y%m%d-%H%M%S)
```

Optional knobs (use only if you want to change defaults):
```bash
!make colab-pretrain RUN=$(date +%Y%m%d-%H%M%S) SEQ_LEN=1024
!make colab-pretrain RUN=$(date +%Y%m%d-%H%M%S) TOKENIZER_MAX_LINES=200000
```

## 6) Chat with the checkpoint
```bash
!make chat RUN=20260127-161044
```

## 7) Instruction-tune (SFT) for chat
Pretraining teaches the model general language modeling, but not chat behavior. SFT is what teaches it to follow the `System/User/Assistant` prompt format used by `scripts/chat_cli.py`.

```bash
!python3 scripts/prepare_sft.py --out data/sft/20260127-161044/clean.jsonl
!python3 scripts/filter_sft.py \
  --in data/sft/20260127-161044/clean.jsonl \
  --out data/sft/20260127-161044/clean.filtered.jsonl \
  --drop-refusals
!python3 scripts/pack_sft.py \
  --tokenizer artifacts/20260127-161044/tokenizer.json \
  --jsonl data/sft/20260127-161044/clean.filtered.jsonl \
  --out data/sft_packed/20260127-161044 \
  --seq-len 2048 \
  --shard-size 1000000 \
  --workers 6 \
  --chunk-size 64 \
  --system-prompt "You are a helpful assistant." \
  --eot-token "<|eot|>"
!python3 scripts/train_sft.py \
  --model configs/nano.yaml \
  --train configs/sft_2048.yaml \
  --data data/sft_packed/20260127-161044 \
  --out runs/20260127-161044/sft \
  --device auto \
  --init runs/20260127-161044/pretrain/ckpt-last.pt \
  --tokenizer artifacts/20260127-161044/tokenizer.json
!make chat RUN=20260127-161044 CHECKPOINT=runs/20260127-161044/sft/ckpt-last.pt
```

## 8) Post-train with RLVR (verifiable rewards)
```bash
!make rlvr-data
!make rlvr-train RUN=20260127-161044 RLVR_INIT=runs/20260127-161044/sft/ckpt-last.pt
!make chat RUN=20260127-161044 CHECKPOINT=runs/20260127-161044/rlvr/ckpt-last.pt
```

## Notes
- `make colab-pretrain` uses:
  - `configs/nano.yaml` (≈60M params)
  - `configs/corpus_colab.yaml`
  - `configs/train_colab.yaml`
- To scale further, update `configs/corpus_colab.yaml` and `configs/train_colab.yaml`.
- If GPU memory is tight, keep `SEQ_LEN=512` and/or reduce `training.micro_batch_size`.
