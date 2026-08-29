# Training

## Experimental local pipeline

`apex train-small` performs real byte-BPE training, causal language-model optimization, checkpointing, and registry writes on a supplied UTF-8 text file. It is a pipeline-validation workload, not a useful pretrained model claim.

```powershell
.\.venv\Scripts\python.exe -m apex.cli train-small `
  --dataset examples/data/tiny-corpus.txt `
  --output checkpoints/apex-tiny-run `
  --steps 2 `
  --dataset-license project-authored
```

Resume with the same configuration and a higher total step target:

```powershell
.\.venv\Scripts\python.exe -m apex.cli train-small `
  --dataset examples/data/tiny-corpus.txt `
  --output checkpoints/apex-tiny-run-resumed `
  --steps 4 `
  --resume-from checkpoints/apex-tiny-run/checkpoint-final.pt `
  --dataset-license project-authored
```

Evaluation and timing use actual checkpoint execution:

```powershell
.\.venv\Scripts\python.exe -m apex.cli evaluate --checkpoint checkpoints/apex-tiny-run/checkpoint-final.pt --tokenizer checkpoints/apex-tiny-run/tokenizer.json --dataset examples/data/tiny-corpus.txt
.\.venv\Scripts\python.exe -m apex.cli benchmark --checkpoint checkpoints/apex-tiny-run/checkpoint-final.pt --tokenizer checkpoints/apex-tiny-run/tokenizer.json --prompt "APEX"
```

Generated checkpoints, datasets, and `.apex/registry` contents remain ignored by Git. Record only reproducible configs and measurements that do not contain private data.

## Capability boundaries

- `apex-tiny`: **EXPERIMENTAL**, suitable for CPU pipeline validation.
- `apex-100m`: **PLANNED / REQUIRES_GPU / REQUIRES_DATA**.
- Kaggle execution: **REQUIRES_KAGGLE** until a real accelerator session verifies it.
- Distributed training, LoRA/QLoRA, preference optimization, and production model serving: **MISSING**.
