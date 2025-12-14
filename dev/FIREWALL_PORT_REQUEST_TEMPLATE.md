# Firewall port opening request (template)

Subject: Request to open inbound TCP ports for MolSys-AI API host

Hello,

We are preparing a public demo service (MolSys-AI) and need inbound connectivity to a specific host. Could you please open the following inbound TCP ports from the Internet to this machine:

- Required: `443/tcp` (HTTPS)
- Optional: `80/tcp` (HTTP; useful for ACME HTTP-01 redirects, but not strictly required if we use DNS-01/TLS-ALPN)

Host identification:

- Public IP: `187.141.21.243`
- DNS name: `api.uibcdf.org`
- Hostname (as configured on the machine): `ixtlilton`
- Internal network interface (default route): `ens1` with `10.10.111.2/24`
- Default gateway: `10.10.111.1`
- MAC address (ens1): `84:16:f9:04:21:cb`

Service overview:

- We will run an HTTPS reverse proxy (Caddy or nginx) on `443/tcp`.
- Internal services will be bound to `127.0.0.1` only:
  - docs chatbot backend on `127.0.0.1:8000`
  - model server on `127.0.0.1:8001`
- Public endpoints:
  - `POST /v1/chat` (docs widget + CLI; CORS enabled for `https://uibcdf.org`)

Security measures (planned):

- TLS termination on the reverse proxy (Let’s Encrypt).
- The model server is not exposed directly to the Internet (only via reverse proxy).
- Optional API key authentication for `/v1/chat` (allowlist via environment variable).
- API key authentication for the internal model engine endpoint (`/v1/engine/chat`, bound to `127.0.0.1`).
- Access logs enabled; service logs retained for incident response.
- Rate limiting will be enforced at the reverse proxy layer when available.

If opening `443/tcp` is not possible, an acceptable fallback is a dedicated forwarded port such as `8443/tcp` routed to this host, but `443/tcp` is strongly preferred.

Thank you.
