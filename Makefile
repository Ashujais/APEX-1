PYTHON ?= .venv/Scripts/python.exe
DATASET ?= examples/data/tiny-corpus.txt
RUN_DIR ?= checkpoints/apex-tiny-run

.PHONY: hardware status test lint train train-small evaluate benchmark

hardware:
	$(PYTHON) scripts/detect_hardware.py

status:
	$(PYTHON) -m apex.cli status

test:
	$(PYTHON) scripts/run_tests.py

lint:
	$(PYTHON) -m ruff check .

train: train-small

train-small:
	$(PYTHON) -m apex.cli train-small --dataset "$(DATASET)" --output "$(RUN_DIR)"

evaluate:
	$(PYTHON) -m apex.cli evaluate --checkpoint "$(RUN_DIR)/checkpoint-final.pt" --tokenizer "$(RUN_DIR)/tokenizer.json" --dataset "$(DATASET)"

benchmark:
	$(PYTHON) -m apex.cli benchmark --checkpoint "$(RUN_DIR)/checkpoint-final.pt" --tokenizer "$(RUN_DIR)/tokenizer.json"
