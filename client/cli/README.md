# MolSys-AI CLI

The `molsys-ai` command line interface provides a minimal entrypoint to
interact with the MolSys-AI HTTP API from a terminal.

## Installation

From the repository root, in an active environment:

```bash
pip install -e .
```

This will expose the console script:

- `molsys-ai` (equivalent to `python -m cli.main`).

## Basic usage

Show version:

```bash
molsys-ai --version
```

## API keys and endpoint selection

The CLI uses an API key to authenticate to the public service.

Key prefixes are currently used as a routing hint:

- `u1_...`: lab/investigator keys (LAN-first, then fallback to the public API)
- `u2_...`: external keys (public API only)

The user experience is the same; the LAN-first behavior is automatic.

Store your API key locally:

```bash
molsys-ai login
```

Non-interactive login:

```bash
printf '%s' "$MOLSYS_AI_API_KEY" | molsys-ai login --stdin
```

Chat (interactive):

```bash
molsys-ai chat
```

By default, `molsys-ai chat` calls `POST /v1/chat` with `client="cli"` and lets the server decide when to use RAG
and when to show sources (citations and links).

Agent mode (local tools + remote LLM):

```bash
molsys-ai agent
```

Explicit local shell tool:

```bash
molsys-ai agent -m "!uname -a"
```

Inspect which local tools are available:

```bash
molsys-ai tools list
```

Check optional dependencies for local tools:

```bash
molsys-ai tools doctor
```

Note: tool execution is local. If you want the agent to execute MolSysSuite
workflows, install those tools in the same environment where you run `molsys-ai`.
To avoid CUDA stack conflicts with the vLLM server environment, a dedicated conda
environment for the agent is recommended.

Send a single message and exit:

```bash
molsys-ai chat -m "Reply only: OK"
```

Ask a documentation question (single-turn):

```bash
molsys-ai docs -m "How do I load a system in MolSysMT?"
```

Docs chat (interactive, multi-turn):

```bash
molsys-ai docs
```

`molsys-ai docs` is a convenience wrapper that forces RAG + sources on, so answers include citations like `[1]` and a list
of source URLs.

## Configuration and environment variables

- The CLI stores configuration under the user config directory:
  - `~/.config/molsys-ai/config.json` (or `$XDG_CONFIG_HOME/molsys-ai/config.json`)

- Override the API key without storing it:
  - `MOLSYS_AI_API_KEY=... molsys-ai chat -m "hello"`

- Override the server URL (advanced / internal):
  - `molsys-ai chat --server-url http://127.0.0.1:8001 -m "hello"`

## Notes and future work

- The current CLI is intentionally minimal and focused on API access.
- Future extensions may include:
  - interactive chat sessions,
  - authenticated chat access if needed,
  - richer output formatting.
