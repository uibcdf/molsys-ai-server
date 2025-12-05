
# MolSys-AI

MolSys-AI is the AI assistant project for the UIBCDF ecosystem:

- MolSysMT
- MolSysViewer
- TopoMT
- and related computational tools (OpenMM, etc.)

MolSys-AI aims to provide:

- An **autonomous agent** that can design and execute workflows using the MolSys* tools.
- A **CLI interface** to interact with the agent from the terminal.
- A **documentation chatbot** embedded in Sphinx/GitHub Pages.
- A flexible **model serving** layer for self-hosted LLMs.

This repository currently contains the initial architecture, decisions and development roadmap.

See:

- `dev/ARCHITECTURE.md` for high-level architecture.
- `dev/ROADMAP.md` for the initial roadmap.
- `dev/CONSTRAINTS.md` for current constraints and assumptions.
- `dev/decisions/` for Architectural Decision Records (ADRs).

## Development environment

The recommended way to create a development environment is with `mamba` or `conda`:

```bash
mamba env create -f environment.yml
mamba activate molsys-ai
```

or:

```bash
conda env create -f environment.yml
conda activate molsys-ai
```

This environment uses Python 3.12, installs MolSys* ecosystem tools from the `uibcdf` channel,  
and includes the core Python dependencies used by this repository.  
Alternative setups using `venv` + `pip` are described in `dev/DEV_GUIDE.md`.

## Language policy

All text in this repository (source code, comments, documentation, development notes, and configuration files) must be written in **English**.
