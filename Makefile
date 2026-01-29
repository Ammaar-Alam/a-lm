PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: dev install lint format test clean check-mps train-pico fresh-pretrain colab-pretrain chat rlvr-data rlvr-train sft-prepare sft-pack sft-train

export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

LIBOMP_DIR ?= $(firstword $(wildcard /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib))
ifneq ($(LIBOMP_DIR),)
export DYLD_LIBRARY_PATH := $(LIBOMP_DIR):$(DYLD_LIBRARY_PATH)
endif

RUN ?= $(shell date +%Y%m%d-%H%M%S)
RUN := $(RUN)

CORPUS_CFG ?= configs/corpus_m2.yaml
MODEL_CFG ?= configs/pico.yaml
TRAIN_CFG ?= configs/train_m2.yaml

SEQ_LEN ?= 512
SHARD_SIZE ?= 1000000
WORKERS ?= 6
CHUNK_SIZE ?= 512
VOCAB_SIZE ?= 32000
TOKENIZER_BACKEND ?= hf
TOKENIZER_MAX_LINES ?=
TOKENIZER_SAMPLE_RATIO ?=
TOKENIZER_LOG_INTERVAL ?= 500
TOKENIZER_MIN_FREQUENCY ?= 2

CLEAN_DIR ?= data/clean/$(RUN)
TOKENIZER_PATH ?= artifacts/$(RUN)/tokenizer.json
PACKED_DIR ?= data/packed/$(RUN)
PRETRAIN_OUT ?= runs/$(RUN)/pretrain
RLVR_DATA ?= data/rlvr/math.jsonl
RLVR_OUT ?= runs/$(RUN)/rlvr
CHECKPOINT ?= runs/$(RUN)/pretrain/ckpt-last.pt
TOKENIZER ?= artifacts/$(RUN)/tokenizer.json

dev: install
	pre-commit install

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

lint:
	ruff check --fix .

format:
	ruff format .

test:
	pytest

check-mps:
	$(PYTHON) -m alm.utils.device

train-pico:
	$(PYTHON) scripts/train_pretrain.py \
		--model configs/pico.yaml \
		--train configs/train.yaml \
		--data data/packed \
		--out runs/pico-pretrain \
		--device auto \
		--tokenizer artifacts/tokenizer.json

fresh-pretrain:
	@echo "run: $(RUN)"
	@echo "stage: prepare corpus ($(CORPUS_CFG))"
	$(PYTHON) scripts/prepare_corpus.py --src $(CORPUS_CFG) --out $(CLEAN_DIR)
	@echo "stage: train tokenizer (backend=$(TOKENIZER_BACKEND) vocab=$(VOCAB_SIZE))"
	$(PYTHON) scripts/train_tokenizer.py \
		--input $(CLEAN_DIR)/*.txt \
		--vocab-size $(VOCAB_SIZE) \
		--out $(TOKENIZER_PATH) \
		--backend $(TOKENIZER_BACKEND) \
		$(if $(TOKENIZER_MAX_LINES),--max-lines $(TOKENIZER_MAX_LINES),) \
		$(if $(TOKENIZER_SAMPLE_RATIO),--sample-ratio $(TOKENIZER_SAMPLE_RATIO),) \
		--log-interval $(TOKENIZER_LOG_INTERVAL) \
		--min-frequency $(TOKENIZER_MIN_FREQUENCY)
	@echo "stage: pack dataset"
	$(PYTHON) scripts/pack_dataset.py \
		--tokenizer $(TOKENIZER_PATH) \
		--in $(CLEAN_DIR) \
		--out $(PACKED_DIR) \
		--seq-len $(SEQ_LEN) \
		--shard-size $(SHARD_SIZE) \
		--workers $(WORKERS) \
		--chunk-size $(CHUNK_SIZE)
	@echo "stage: pretrain (model=$(MODEL_CFG) train=$(TRAIN_CFG))"
	$(PYTHON) scripts/train_pretrain.py \
		--model $(MODEL_CFG) \
		--train $(TRAIN_CFG) \
		--data $(PACKED_DIR) \
		--out $(PRETRAIN_OUT) \
		--device auto \
		--tokenizer $(TOKENIZER_PATH)
	@echo "chat: make chat RUN=$(RUN)"

colab-pretrain: CORPUS_CFG = configs/corpus_colab.yaml
colab-pretrain: TRAIN_CFG = configs/train_colab.yaml
colab-pretrain: MODEL_CFG = configs/nano.yaml
colab-pretrain: SEQ_LEN = 512
colab-pretrain: TOKENIZER_BACKEND = hf
colab-pretrain:
	$(MAKE) fresh-pretrain \
		CORPUS_CFG=$(CORPUS_CFG) \
		TRAIN_CFG=$(TRAIN_CFG) \
		MODEL_CFG=$(MODEL_CFG) \
		SEQ_LEN=$(SEQ_LEN) \
		TOKENIZER_BACKEND=$(TOKENIZER_BACKEND)

chat:
	$(PYTHON) scripts/chat_cli.py \
		--checkpoint $(CHECKPOINT) \
		--tokenizer $(TOKENIZER) \
		--device auto

rlvr-data:
	$(PYTHON) scripts/generate_rlvr_math.py --out $(RLVR_DATA) --count 20000

rlvr-train:
	$(PYTHON) scripts/train_rlvr.py \
		--init runs/$(RUN)/pretrain/ckpt-last.pt \
		--tokenizer artifacts/$(RUN)/tokenizer.json \
		--data $(RLVR_DATA) \
		--out $(RLVR_OUT) \
		--device auto

sft-prepare:
	python scripts/prepare_sft.py --out data/sft/clean.jsonl

sft-pack:
	python scripts/pack_sft.py \
		--tokenizer artifacts/tokenizer.json \
		--jsonl data/sft/clean.jsonl \
		--out data/sft_packed \
		--seq-len 384 \
		--shard-size 1000000 \
		--workers 6 \
		--chunk-size 64

sft-train:
	unset PYTORCH_MPS_FAST_MATH || true; \
	python scripts/train_sft.py \
		--model configs/pico_sft.yaml \
		--train configs/sft.yaml \
		--data data/sft_packed \
		--out runs/pico-sft \
		--device auto \
		--init runs/pico-pretrain/ckpt-last.pt \
		--tokenizer artifacts/tokenizer.json

clean:
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
