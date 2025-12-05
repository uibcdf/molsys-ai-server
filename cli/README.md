# MolSys-AI CLI

The `molsys-ai` command line interface provides a minimal entrypoint to
interact with the MolSys-AI agent and model server from a terminal.

## Installation

From the repository root, in an active environment:

```bash
pip install -e .
```

This will expose the console script:

- `molsys-ai` (equivalent to `python -m cli.main`).

## Basic usage

Print version information:

```bash
molsys-ai --version
```

Send a single message using the local echo stub:

```bash
molsys-ai --message "Hello from MolSys-AI"
```

Point the CLI to a running model server (stub or real):

```bash
molsys-ai --server-url http://127.0.0.1:8000 \
          --message "How do I load a system in MolSysMT?"
```

In all cases, the CLI prints a single reply to stdout and exits with:

- `0` on success,
- `1` if there was an error calling the model server (e.g. connection failure).

## Options

- `--version`
  - Print a short version string and exit.

- `-m, --message TEXT`
  - Send a single user message to MolSys-AI.
  - When `--server-url` is not provided, uses the local `EchoModelClient`.

- `--server-url URL`
  - Base URL of the MolSys-AI model server (see `model_server/README.md`).
  - When provided, the CLI uses `HTTPModelClient` to call `POST {URL}/v1/chat`.

## Notes and future work

- The current CLI is intentionally minimal and designed for wiring tests.
- Future extensions may include:
  - interactive chat sessions,
  - subcommands for running predefined workflows,
  - richer output formatting (e.g. via `rich`).

