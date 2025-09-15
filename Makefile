PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: dev install lint format test clean check-mps train-pico

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
	$(PYTHON) -m a-lm.utils.device

train-pico:
	@echo "Training scripts not yet implemented. See TODO_LIST.md for roadmap."

clean:
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
