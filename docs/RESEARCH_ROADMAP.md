# Research roadmap

Each phase advances only with versioned data, measured hardware, reproducible configs, and evaluation evidence.

1. **Small model:** validate tokenizer, Transformer, checkpoints, loss curves, and contamination-aware evaluation at a compute-feasible scale.
2. **SFT:** curate licensed instruction data and compare full fine-tuning with LoRA/QLoRA.
3. **RAG:** measure retrieval recall, reranking quality, citation correctness, and latency.
4. **Tools:** add sandboxed, schema-validated tools with approval and audit boundaries.
5. **Agents:** evaluate planners, budgets, recovery, and verifier effectiveness.
6. **Reasoning:** compare candidate search and verifiers without exposing hidden chain-of-thought.
7. **Alignment:** benchmark modular preference methods and safety regressions.
8. **Multimodal:** integrate real encoders only when data, compute, and evaluation are available.
9. **Distributed training:** validate DDP/FSDP recovery, sharded checkpoints, and throughput.
10. **Large-scale training:** scale only after smaller runs establish data and architecture quality.
11. **Frontier experiments:** pursue novel architectures under reproducible evaluation and red-team gates.
