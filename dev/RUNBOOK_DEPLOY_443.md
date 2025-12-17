# Runbook: Production deployment on `443/tcp` (Caddy + systemd)

This runbook is the “day we get port 443” checklist to deploy:

- static docs on GitHub Pages (`https://uibcdf.org/...`)
- MolSys-AI API on this GPU host (`https://api.uibcdf.org`)

Target architecture (recommended):

- Caddy terminates TLS on `:443` and reverse-proxies:
  - `/v1/chat` → `127.0.0.1:8000` (public chat API; CORS enabled for the widget)
- `chat_api` and `model_server` run as systemd services bound to `127.0.0.1`.
  - `chat_api` calls the model engine server privately over `http://127.0.0.1:8001/v1/engine/chat`.

## 0) Prerequisites

- DNS: `api.uibcdf.org` points to this machine’s public IP.
- Firewall: inbound `443/tcp` is reachable from the Internet.
- CUDA Toolkit installed (nvcc available): see `dev/RUNBOOK_VLLM.md`.
- Model is downloaded locally (recommended): `models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`.

Python dependencies:

- The services require:
  - `pip install -e ".[server,rag]"` (FastAPI + RAG dependencies)
  - plus `sentence-transformers` if you want high-quality embeddings (recommended; included in `.[rag]`).

Quick checks (run from a remote machine):

```bash
dig +short api.uibcdf.org
nmap -Pn -p 443 api.uibcdf.org
```

## 1) Install Caddy (Ubuntu)

Follow the official Caddy installation instructions for your OS.

After installation:

```bash
sudo systemctl enable --now caddy
sudo systemctl status caddy --no-pager
```

## 2) Place configs

### 2.1 Caddyfile

Copy `dev/Caddyfile.example` to:

```bash
sudo cp dev/Caddyfile.example /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
```

### 2.2 MolSys-AI service environment

Create the directory and copy the env file:

```bash
sudo mkdir -p /etc/molsys-ai
sudo cp dev/molsys-ai.env.example /etc/molsys-ai/molsys-ai.env
sudo nano /etc/molsys-ai/molsys-ai.env
```

Set (at minimum):

- `MOLSYS_AI_MODEL_CONFIG` (path to the YAML file for `model_server`)
- `MOLSYS_AI_ENGINE_API_KEYS` (generate a strong key for the internal engine endpoint)
- `MOLSYS_AI_CHAT_API_KEYS` (optional: protect the public chat API)
- `MOLSYS_AI_ENGINE_API_KEY` (a key allowed by `MOLSYS_AI_ENGINE_API_KEYS`; used by `chat_api`)
- `MOLSYS_AI_CORS_ORIGINS` (include `https://uibcdf.org`)

### 2.3 Model server YAML config

Create the model config path referenced by `MOLSYS_AI_MODEL_CONFIG`, for example:

```bash
sudo nano /etc/molsys-ai/model_server.yaml
```

Use `server/model_server/config.example.yaml` as the template and set:

- `model.local_path` to your local model directory
- `tensor_parallel_size` to `1` initially (scale later if needed)
- keep the validated baseline:
  - `max_model_len: 8192`
  - `gpu_memory_utilization: 0.80`
  - `enforce_eager: true`

### 2.4 Prepare the docs RAG corpus (recommended)

The docs chatbot answers questions using a local snapshot of documentation files
(`*.md`, `*.rst`, `*.txt`) plus an embedding index.

If you keep the MolSysSuite repos checked out on this machine (recommended for frequent refresh),
you can generate/update the corpus with:

```bash
python dev/sync_rag_corpus.py --clean --build-index --build-anchors
```

For production, prefer storing generated artifacts under `/var/lib/molsys-ai` (outside the git checkout),
and set `MOLSYS_AI_DOCS_DIR` / `MOLSYS_AI_DOCS_INDEX` in `/etc/molsys-ai/molsys-ai.env`.

Recommended for production correctness/quality:

- Build the code-aware layers (API surface + symbol cards + recipes):

```bash
python dev/sync_rag_corpus.py --clean --build-api-surface --build-symbol-cards --build-recipes --build-index --build-project-indices --build-anchors
```

- Build a BM25 sidecar for stronger identifier matching:

```bash
python dev/sync_rag_corpus.py --clean --build-index --build-bm25 --build-project-indices
```

Then set `MOLSYS_AI_RAG_BM25_WEIGHT` in the service env (try `0.25`).

## 3) Install and enable systemd services

Copy the example units:

```bash
sudo cp dev/systemd/molsys-ai-model.service.example /etc/systemd/system/molsys-ai-model.service
sudo cp dev/systemd/molsys-ai-chat-api.service.example /etc/systemd/system/molsys-ai-chat-api.service
sudo systemctl daemon-reload
```

Edit both unit files and adjust:

- `WorkingDirectory` to your repo checkout path
- `ExecStart` Python path (venv/conda path on this host)
- optionally `User=` and `Group=` for a dedicated service account

Start services:

```bash
sudo systemctl enable --now molsys-ai-model
sudo systemctl enable --now molsys-ai-chat-api
sudo systemctl status molsys-ai-model --no-pager
sudo systemctl status molsys-ai-chat-api --no-pager
```

### 3.1 Optional: schedule periodic corpus refresh (weekly)

If the underlying docs repos change frequently, you can refresh the snapshot and index on a schedule.

Copy the refresh unit + timer:

```bash
sudo cp dev/systemd/molsys-ai-rag-refresh.service.example /etc/systemd/system/molsys-ai-rag-refresh.service
sudo cp dev/systemd/molsys-ai-rag-refresh.timer.example /etc/systemd/system/molsys-ai-rag-refresh.timer
sudo systemctl daemon-reload
sudo systemctl enable --now molsys-ai-rag-refresh.timer
sudo systemctl list-timers --all | grep molsys-ai-rag-refresh
```

This runs `dev/sync_rag_corpus.py` and then restarts `molsys-ai-chat-api` so it loads the new index.

## 4) Restart Caddy and verify end-to-end

```bash
sudo systemctl restart caddy
sudo systemctl status caddy --no-pager
```

From the server itself:

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8001/healthz
curl -fsS http://127.0.0.1:8000/docs >/dev/null
curl -fsS http://127.0.0.1:8001/docs >/dev/null
```

From a remote machine (public):

```bash
curl -fsS https://api.uibcdf.org/healthz
curl -fsS https://api.uibcdf.org/healthz/chat
curl -fsS https://api.uibcdf.org/healthz/model
curl -fsS -X POST https://api.uibcdf.org/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"ping","k":1}'
```

If `/v1/chat` is protected with API keys, test with a valid key:

```bash
curl -fsS -X POST https://api.uibcdf.org/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <YOUR_KEY>' \
  -d '{"messages":[{"role":"user","content":"Reply only: OK"}],"client":"cli","rag":"off","sources":"off"}'
```

## 5) Operational notes

- Keep `model_server` and `chat_api` bound to `127.0.0.1`.
- Only Caddy listens on public interfaces.
- `/v1/chat` is public by design; plan rate limiting at the proxy layer.
- If GPU memory gets stuck after crashes, check for stray `VLLM::EngineCore` processes.
