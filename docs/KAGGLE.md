# Kaggle research workflow

Status: **APEX-100M PREPARATION READY / EXECUTION REQUIRES_KAGGLE**

Kaggle is APEX-1's primary free GPU research environment. It is not a production inference backend. The adapter refuses to start Kaggle training unless it detects both a Kaggle runtime and a CUDA device through PyTorch; it never assumes a GPU model.

## Workflow

1. Push a clean commit to a private or public GitHub repository without datasets, checkpoints, or secrets.
2. Create a Kaggle notebook, enable an accelerator, and clone that commit.
3. Run `bash scripts/kaggle/setup_kaggle.sh`.
4. Put licensed input data in `/kaggle/input/...`; never commit it merely to feed a notebook.
5. Run the non-training preflight first:
   `python scripts/kaggle/preflight.py --dataset /kaggle/input/<dataset>/<file> --dataset-license <license>`.
6. Run a one-step apex-tiny smoke with `scripts/kaggle/train.py`. This validates allocation,
   dataset loading, checkpointing, and artifacts; it is not APEX-100M training.
7. Evaluate with `python scripts/kaggle/evaluate.py --checkpoint <checkpoint> --tokenizer <tokenizer.json> --dataset <validation-file>`.
8. Export `/kaggle/working/apex-artifacts` to durable storage before the runtime terminates.

The training checkpoint includes model, optimizer, scheduler, scaler, PyTorch RNG, CUDA RNG when present, optimizer/micro-step counts, dataset cursor, configuration, Git commit where discoverable, dataset/tokenizer hashes, and measured hardware metadata. Checkpoints are written at optimizer boundaries so no unpersisted accumulated gradients are implied.

## Current limitations

- configs/apex-100m-kaggle.json is the planned Kaggle-first profile and resolves to 100,092,672
  parameters. Preflight validates runtime, CUDA, free VRAM, configuration, dataset license, and
  artifact storage without starting training.
- The executed workload remains the experimental `apex-tiny` text pipeline. `apex-100m`
  training remains planned until a measured Kaggle allocation, licensed dataset, and explicit
  user authorization exist.
- The local filesystem registry is single-process research storage. A shared production registry requires PostgreSQL and object storage.
- Kaggle runtime execution has not been performed from this laptop and remains `REQUIRES_KAGGLE`.
- Dataset streaming, sharding, DDP/FSDP, LoRA/QLoRA, and preference training are not implemented.
