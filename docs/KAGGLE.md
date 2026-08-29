# Kaggle research workflow

Status: **EXPERIMENTAL / REQUIRES_KAGGLE**

Kaggle is APEX-1's primary free GPU research environment. It is not a production inference backend. The adapter refuses to start Kaggle training unless it detects both a Kaggle runtime and a CUDA device through PyTorch; it never assumes a GPU model.

## Workflow

1. Push a clean commit to a private or public GitHub repository without datasets, checkpoints, or secrets.
2. Create a Kaggle notebook, enable an accelerator, and clone that commit.
3. Run `bash scripts/kaggle/setup_kaggle.sh`.
4. Put licensed input data in `/kaggle/input/...`; never commit it merely to feed a notebook.
5. Run `python scripts/kaggle/train.py --dataset /kaggle/input/<dataset>/<file> --output /kaggle/working/apex-artifacts/run --steps <n> --dataset-license <license>`.
6. Evaluate with `python scripts/kaggle/evaluate.py --checkpoint <checkpoint> --tokenizer <tokenizer.json> --dataset <validation-file>`.
7. Export `/kaggle/working/apex-artifacts` to durable storage before the runtime terminates.

The training checkpoint includes model, optimizer, scheduler, scaler, PyTorch RNG, CUDA RNG when present, optimizer/micro-step counts, dataset cursor, configuration, Git commit where discoverable, dataset/tokenizer hashes, and measured hardware metadata. Checkpoints are written at optimizer boundaries so no unpersisted accumulated gradients are implied.

## Current limitations

- The validated workload is the experimental `apex-tiny` text pipeline. `apex-100m` remains planned until a measured Kaggle allocation, licensed dataset, and run evidence exist.
- The local filesystem registry is single-process research storage. A shared production registry requires PostgreSQL and object storage.
- Kaggle runtime execution has not been performed from this laptop and remains `REQUIRES_KAGGLE`.
- Dataset streaming, sharding, DDP/FSDP, LoRA/QLoRA, and preference training are not implemented.
