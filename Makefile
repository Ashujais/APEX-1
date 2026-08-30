PYTHON ?= .venv/Scripts/python.exe
POWERSHELL ?= powershell
DATASET ?= examples/data/tiny-corpus.txt
RUN_DIR ?= checkpoints/apex-tiny-run

.PHONY: dev hardware status test test-all lint train train-small evaluate benchmark web-install web-lint web-build

dev:
	$(POWERSHELL) -NoProfile -ExecutionPolicy Bypass -File scripts/dev.ps1

hardware:
	$(PYTHON) scripts/detect_hardware.py

status:
	$(PYTHON) -m apex.cli status

test:
	$(PYTHON) scripts/run_tests.py

test-all:
	$(POWERSHELL) -NoProfile -ExecutionPolicy Bypass -File scripts/test.ps1

lint:
	$(PYTHON) -m ruff check .

train: train-small

train-small:
	$(PYTHON) -m apex.cli train-small --dataset "$(DATASET)" --output "$(RUN_DIR)"

evaluate:
	$(PYTHON) -m apex.cli evaluate --checkpoint "$(RUN_DIR)/checkpoint-final.pt" --tokenizer "$(RUN_DIR)/tokenizer.json" --dataset "$(DATASET)"

benchmark:
	$(PYTHON) -m apex.cli benchmark --checkpoint "$(RUN_DIR)/checkpoint-final.pt" --tokenizer "$(RUN_DIR)/tokenizer.json"

web-install:
	cd apps/web && npm ci

web-lint:
	cd apps/web && npm run lint

web-build:
	cd apps/web && npm run build
