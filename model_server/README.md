
# MolSys-AI Model Server (MVP)

This directory contains the FastAPI-based model server for MolSys-AI.

For the MVP, the server exposes a `/v1/chat` endpoint and uses a stub implementation.
Later it will call llama.cpp (or another backend) to run the chosen model.

Configuration will be read from `config.yaml` (see `config.example.yaml`).
