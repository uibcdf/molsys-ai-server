# Guide for the Gemini Assistant

This document provides a quick context and operational guide for working efficiently in the `molsys-ai` repository.

## 1. Project Summary

**Objective:** To build an AI assistant for the `MolSys*` ecosystem of molecular simulation tools (MolSysMT, MolSysViewer, etc.).

**Key Components:**
1.  **Autonomous CLI Agent:** An agent that uses the MolSysSuite tools to perform workflows.
2.  **Documentation Chatbot:** A RAG-based chatbot to answer questions about using the tools, integrable into the web documentation.
3.  **Retrained Language Model:** An LLM specialized in the knowledge of the MolSys* ecosystem.

## 2. Development Environment

**IMPORTANT!** The working environment is managed exclusively with **Conda/Mamba** through the `environment.yml` file.

- **Preference for Micromamba:** The use of `micromamba` is preferred over `conda` for its greater speed and efficiency.

- **Conda Channels:** The channel priority is:
  1.  `uibcdf`: For the MolSys* ecosystem tools.
  2.  `conda-forge`: For most dependencies, aiming for stability.
  3.  `defaults`: As a last resort.

- **Local Dependencies (Under Development):** The tools `molsysmt`, `molsysviewer`, `topomt`, `elastnet`, and `pharmacophoremt` are installed manually from their local repositories. For this reason, they **must remain commented out** in the `environment.yml` file to avoid overwriting the development versions with the stable ones from the Conda channels.

- **Installation and Updates:** To set up or update the environment, use `micromamba` or `conda`. Ensure that the `environment.yml` file **does not contain the `name` field** so that the active environment is updated. To avoid terminal blocking, it is recommended to use the `-y` flag.
  ```bash
  # With micromamba (preferred)
  micromamba env update -f environment.yml -y

  # With conda
  conda env update --file environment.yml -y
  ```

- **Editable Mode:** To make commands like `molsys-ai` available, the package must be installed in editable mode. After activating the environment, run:
  ```bash
  pip install -e .
  ```

### 2.1. Running Servers in the Background

To run servers like `uvicorn` without blocking the terminal, the `nohup` command must be used. This ensures that the process continues to run even if the terminal session is closed and redirects all output to a log file.

- **Example Usage:**
  ```bash
  # Runs the server in the background and saves its log to 'server.log'
  nohup uvicorn model_server.server:app --reload > server.log 2>&1 &
  ```
- **To stop the server:**
  The `nohup` command returns a PID. To stop the server, the `kill` command must be used with the PGID (Process Group ID) provided by the `run_shell_command` tool to ensure that all child processes also terminate.

### 2.2. Compiling `llama-cpp-python` with GPU (CPU Fallback)

Compiling `llama-cpp-python` with GPU support (CUDA acceleration) is complex and sensitive to the environment configuration (version of `gcc`, `nvcc`, `cudatoolkit`, etc.). During development, we encountered multiple compilation problems that were irresolvable in an automated way.

- **Current Status:** To unblock development, the project operates with a **CPU-only** version of `llama-cpp-python` installed from `conda-forge`. This allows the RAG system to be functional, although model inference is slower.
- **Long-Term Solution:** To enable GPU support, the robust solution is to install the **full NVIDIA CUDA Toolkit at the system level** (not via `conda`) and force the compilation of `llama-cpp-python` to use that installation. This requires manual system configuration and is outside the scope of automated environment management.

Any attempt to reinstall `llama-cpp-python` with GPU support must take this complexity into account.

## 3. Key Execution Commands

- **Model Server (Stub):**
  ```bash
  uvicorn model_server.server:app --reload
  ```
- **CLI (pointing to the server):**
  ```bash
  molsys-ai --server-url http://127.0.0.1:8000 --message "Your question"
  ```
- **Documentation Chat Backend:**
  ```bash
  uvicorn docs_chat.backend:app --reload
  ```

## 4. Testing

- **Run Tests:**
  ```bash
  pytest
  ```
- **Convention for Mocks:** To avoid the `ModuleNotFoundError: No module named 'molsysmt'`, any test that directly or indirectly imports `molsysmt` must mock the module. The convention established in this project is to patch `sys.modules`. **See `tests/test_smoke.py` as a reference example**:
  ```python
  # At the beginning of the test function
  mocker.patch.dict("sys.modules", {"molsysmt": mocker.Mock()})

  # Import the modules that depend on molsysmt AFTER the patch
  from agent.core import MolSysAIAgent
  ```

## 5. RAG System (Retrieval-Augmented Generation)

- **Purpose:** It is the knowledge search engine for the chatbot. It is based on finding relevant documentation snippets for a question.
- **Data Source:** The documentation from the `molsysmt` repository, which is assumed to be in `../molsysmt/`.
- **Index:** The RAG process generates an index (`data/rag_index.pkl`) that contains the texts and their vector embeddings.
- **Index Construction:** To generate or update the index, a script must be run that calls `rag.build_index.build_index()`. The temporary script `_build_index_script.py` was created for this purpose:
  ```bash
  python _build_index_script.py
  ```
  **Note:** The first time, this command downloads a model from `sentence-transformers` and then processes the documentation. It may take several minutes and will use the GPU if available.

## 6. Current Status and Next Steps

To know the exact state in which the project was left and what the immediate next steps are, **always consult the `checkpoint.md` file**. This file is updated at the end of each work session to ensure a smooth transition.