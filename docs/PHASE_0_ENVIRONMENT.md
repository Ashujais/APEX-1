# Phase 0 environment inventory

Captured on 2026-08-29.

| Resource | Detected |
| --- | --- |
| OS | Windows 10 Pro 64-bit, 10.0.19045 |
| CPU | Intel Core i3-7020U, 2 cores / 4 logical processors |
| RAM | 3.8 GB |
| GPU | Intel HD Graphics 620, reported 1 GB shared adapter memory |
| NVIDIA CUDA | Not detected |
| Disk | F: 394.4 GB total / 382.8 GB free |
| Python | Bundled CPython 3.12.13 |
| Node | System 20.18.0; bundled 24.19.0 used for the web toolchain |
| Docker | CLI 29.5.2 installed; daemon not running |

## Recommendation

Use the machine for API/UI development, tokenizer validation, and tiny CPU-only Transformer smoke tests. Do not attempt APEX-100M training here. Practical pretraining begins only after measuring a suitable discrete GPU environment; multi-billion parameter work requires dedicated multi-GPU infrastructure.
