# Deploying `api.uibcdf.org` (docs chatbot + model server)

This guide describes a production-oriented deployment for:

- public docs hosted on GitHub Pages under `https://uibcdf.org/...`
- a dynamic API hosted on this server under `https://api.uibcdf.org`

Key idea: keep docs static (GH Pages) and run the chatbot API separately.

## 0) DNS

Create an `A` record in DreamHost DNS:

- `api.uibcdf.org` → `187.141.21.243`

Verify:

```bash
dig +short api.uibcdf.org
```

## 1) Open ports (must be reachable from the Internet)

Your `nmap` output showed `80/443` as `filtered`, so something is dropping inbound packets.

On the server, check firewall state:

```bash
sudo ufw status verbose
```

If `ufw` is active, allow HTTP/HTTPS:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw reload
```

Re-test from a remote machine:

```bash
nmap -Pn -p 80,443 187.141.21.243
```

### Current reality (data-center firewall)

If `ufw` is inactive and `iptables` policy is `ACCEPT`, but `nmap` still shows:

- `80/tcp filtered`
- `443/tcp filtered`

even while a local server is listening, then the filtering is happening upstream
(data-center firewall). In that case you must request port opening from the center.
An email/ticket template is available in:

- `dev/FIREWALL_PORT_REQUEST_TEMPLATE.md`

From the same scan we observed that `8080/tcp` is reachable from the Internet.
However, a remote `curl http://187.141.21.243:8080/` returned a Tomcat welcome
page (`Server: Apache-Coyote/1.1`), meaning `8080` is currently occupied by an
existing service (or upstream port mapping) and should be treated as **not
available** for MolSys-AI unless the center confirms you can repurpose it.

## 2) Run internal services on localhost

Run them bound to `127.0.0.1` (not directly exposed to the Internet):

- `model_server` (vLLM) on `127.0.0.1:8001` (source: `server/model_server/`)
- `docs_chat` on `127.0.0.1:8000` and allow the docs origin via CORS (source: `server/docs_chat/`)

Example env vars for `docs_chat`:

```bash
export MOLSYS_AI_MODEL_SERVER_URL=http://127.0.0.1:8001
export MOLSYS_AI_CORS_ORIGINS=https://uibcdf.org,https://www.uibcdf.org
export MOLSYS_AI_EMBEDDINGS=sentence-transformers
```

For a systemd-based deployment, see the example unit files:

- `dev/systemd/molsys-ai-model.service.example`
- `dev/systemd/molsys-ai-docs-chat.service.example`
- `dev/molsys-ai.env.example`

## 3) Reverse proxy + TLS (recommended)

Use a reverse proxy to terminate TLS and route:

- `https://api.uibcdf.org/v1/docs-chat` → `http://127.0.0.1:8000/v1/docs-chat`

Keep the model server bound to `127.0.0.1`. The public API should expose only `docs_chat`;
`docs_chat` calls the model server privately over `http://127.0.0.1:8001/v1/chat`.

### Option A: Caddy (simple)

Install Caddy and configure a site for `api.uibcdf.org`.

Example `Caddyfile`:

```caddyfile
api.uibcdf.org {
  encode zstd gzip

  reverse_proxy /v1/docs-chat* 127.0.0.1:8000

  # Optional: expose OpenAPI for docs_chat
  reverse_proxy /docs* 127.0.0.1:8000
  reverse_proxy /openapi.json 127.0.0.1:8000
}
```

An up-to-date example file is provided in:

- `dev/Caddyfile.example`

For a step-by-step “day we get port 443” checklist (Caddy + systemd), see:

- `dev/RUNBOOK_DEPLOY_443.md`

### Option B: nginx (more knobs, easy rate limiting)

nginx can enforce `limit_req` for public endpoints (recommended).

## 4) Security / abuse controls (important for public deployment)

CORS does not protect the API from non-browser clients. For a public endpoint, add:

- rate limiting (nginx `limit_req`, Cloudflare, or application-level)
- logging/monitoring
- optionally a “soft” proof-of-work / captcha at a higher layer if abused

Keep any CLI/agent endpoints (future) behind API keys; do not expose the raw model endpoint publicly.

### API key policy (recommended)

- Keep `POST /v1/docs-chat` public for the docs widget (CORS is a browser-only control).
- If you protect `POST /v1/docs-chat` with API keys (`MOLSYS_AI_DOCS_CHAT_API_KEYS`), treat the widget key as public and
  apply strict rate limiting at the proxy layer.
- Protect the internal model server endpoint (`POST /v1/chat`) with an API key allowlist:
  - set `MOLSYS_AI_CHAT_API_KEYS` on the `model_server`,
  - set `MOLSYS_AI_MODEL_SERVER_API_KEY` on `docs_chat` so it can call `http://127.0.0.1:8001/v1/chat`.

Clients can authenticate with:

- `Authorization: Bearer <key>`
- or `X-API-Key: <key>`

## 4.1 Temporary demo over HTTPS on a non-standard port (only if the port is yours)

If the data-center firewall exposes some non-standard port (e.g. `8443`) but
blocks `80/443`, you can still run a public demo at:

- `https://api.uibcdf.org:<PORT>/v1/docs-chat`

Requirements:

- A trusted TLS certificate for `api.uibcdf.org` (use DNS-01 validation).
- `docs_chat` (or a local reverse proxy) listening on `0.0.0.0:<PORT>` with TLS enabled.
- CORS allowing the docs origin: `https://uibcdf.org`.

Before using a port, verify it is actually bound on this host and reachable:

- locally: `sudo ss -ltnp | grep :<PORT>`
- remotely: `curl -v http://187.141.21.243:<PORT>/` (should hit your service, not a different one)

Practical approach:

1) Obtain a certificate using a DNS-01 ACME client (e.g. certbot manual DNS).
2) Run `docs_chat` with TLS on that port:

```bash
MOLSYS_AI_MODEL_SERVER_URL=http://127.0.0.1:8001 \
MOLSYS_AI_CORS_ORIGINS=https://uibcdf.org,https://www.uibcdf.org \
MOLSYS_AI_EMBEDDINGS=sentence-transformers \
uvicorn docs_chat.backend:app --host 0.0.0.0 --port <PORT> \
  --ssl-keyfile /path/to/privkey.pem \
  --ssl-certfile /path/to/fullchain.pem
```

3) Use the widget query-param override in docs during the demo:

`?molsys_ai_mode=backend&molsys_ai_backend_url=https://api.uibcdf.org:<PORT>/v1/docs-chat`

## 5) Temporary public demo without inbound 80/443 (recommended fallback)

If the data-center firewall currently blocks inbound `80/443` to this machine
(`nmap` shows `filtered` even when a local server listens), you can still run a
public demo by using an **external VPS** and an **SSH reverse tunnel**.

High-level idea:

- The VPS is reachable from the Internet on `443` and terminates TLS for a demo domain.
- The VPS reverse-proxies requests into an SSH tunnel that forwards to
  `docs_chat` running on this GPU machine.
- `docs_chat` continues to call `model_server` locally via `127.0.0.1`.

### 5.1 Create a demo subdomain

In DreamHost DNS, create an `A` record for a demo subdomain pointing to the VPS:

- `api-demo.uibcdf.org` → `<VPS_PUBLIC_IP>`

Keep `api.uibcdf.org` pointing to this machine for the final deployment later.

### 5.2 Run services locally on the GPU machine (no public ports)

Run both bound to `127.0.0.1`:

- `model_server` on `127.0.0.1:8001`
- `docs_chat` on `127.0.0.1:8000`

Use CORS for the public docs origin:

```bash
export MOLSYS_AI_MODEL_SERVER_URL=http://127.0.0.1:8001
export MOLSYS_AI_CORS_ORIGINS=https://uibcdf.org,https://www.uibcdf.org
```

### 5.3 Create the SSH reverse tunnel (GPU machine → VPS)

On the GPU machine, forward the VPS local port `9000` to local `docs_chat:8000`:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:9000:127.0.0.1:8000 \
  <VPS_USER>@<VPS_HOST>
```

For robustness, use `autossh` (recommended) to keep the tunnel alive.

### 5.4 Reverse proxy on the VPS (TLS on the VPS)

On the VPS, configure a reverse proxy for `api-demo.uibcdf.org` that forwards to
`127.0.0.1:9000` (the tunnel endpoint). With Caddy, this is minimal:

```caddyfile
api-demo.uibcdf.org {
  encode zstd gzip
  reverse_proxy 127.0.0.1:9000
}
```

### 5.5 Point the widget to the demo domain

During the demo, use the widget query param override:

`?molsys_ai_mode=backend&molsys_ai_backend_url=https://api-demo.uibcdf.org/v1/docs-chat`

This avoids changing the default `api.uibcdf.org` setting in the docs.
