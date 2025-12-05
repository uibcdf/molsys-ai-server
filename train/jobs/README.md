# Training jobs and launch scripts

This directory is intended for job launch scripts that run training on the
dedicated training node (Node B).

Examples (to be added later):

- SLURM job scripts for QLoRA experiments.
- Simple `bash` wrappers that:
  - activate the appropriate conda/mamba environment,
  - call `python -m train.scripts.train_lora --config ...`,
  - capture logs and metrics.

These scripts will depend on the specific cluster or server setup used by the
laboratory and are therefore kept separate from the core Python packages.

