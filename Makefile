PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: dev install lint format test clean check-mps train-pico sft-prepare sft-pack sft-train

dev: install
	pre-commit install

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	pytest

check-mps:
	$(PYTHON) -m alm.utils.device

train-pico:
	@echo "Training scripts not yet implemented. See TODO_LIST.md for roadmap."

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
	export PYTORCH_MPS_FAST_MATH=1
	python scripts/train_sft.py \
		--model configs/pico.yaml \
		--train configs/sft.yaml \
		--data data/sft_packed \
		--out runs/pico-sft \
		--device auto \
		--init runs/pico-pretrain/ckpt-last.pt

clean:
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
