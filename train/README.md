# MolSys-AI Training (LoRA/QLoRA)

This directory groups together resources and scripts related to **training and fine-tuning**
models for MolSys-AI (for example, LoRA/QLoRA on top of Qwen2.5-7B).

The intent is to keep training logic clearly separated from:
- the runtime agent (`agent/`),
- the model server (`model_server/`),
- the RAG layer (`rag/`),
- and CLI/frontends.

In the deployment model described in the ADRs:
- training and experimentation happen on a **dedicated node** (Node B),
- serving happens on a **separate node** (Node A) that pulls models from Hugging Face Hub.

## Subdirectories

- `configs/`:
  - Experiment and training configuration files (YAML/JSON).
  - Examples: which base model to use, datasets, LoRA hyperparameters, evaluation settings.

- `scripts/`:
  - Reusable Python scripts and modules for:
    - data loading and preprocessing,
    - prompt construction,
    - training/evaluation loops,
    - exporting artifacts to Hugging Face Hub.

- `jobs/`:
  - Job launch scripts for the training node (e.g. SLURM, simple bash launchers).
  - These are expected to be tailored to the laboratory infrastructure.

- `notebooks/`:
  - Prototyping and exploratory analysis (Jupyter notebooks, if needed).
  - Not required for production use.

- `data/`:
  - This repository **should not** store raw training data.
  - Use this directory only for small sample snippets or metadata, and document
    where the real datasets live (local storage, object store, etc.).

At this stage, the training pipeline is not implemented yet. This skeleton is
provided to make it easier to add the first QLoRA/LoRA workflows without
restructuring the repository later.

