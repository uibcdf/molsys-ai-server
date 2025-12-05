# Training configs

This directory is intended to hold experiment configuration files for training
and fine-tuning MolSys-AI models.

Suggested conventions:

- Use human-readable formats such as YAML or TOML.
- Include, at minimum:
  - the base model identifier (e.g. a Hugging Face Hub repo),
  - LoRA/QLoRA hyperparameters,
  - dataset definitions (paths, filters, splits),
  - evaluation settings and output locations.

Example (outline only):

```yaml
experiment:
  name: "molsys-ai-qwen2p5-7b-lora-v0"

model:
  base_hub_id: "Qwen/Qwen2.5-7B-Instruct"
  lora_rank: 64
  lora_alpha: 16

data:
  train_manifest: "/path/to/train_manifest.jsonl"
  val_manifest: "/path/to/val_manifest.jsonl"

output:
  hub_org: "uibcdf"
  hub_repo: "molsys-ai-qwen2p5-7b-lora-v0"
```

Real configuration files can be added as the training pipeline is designed.

