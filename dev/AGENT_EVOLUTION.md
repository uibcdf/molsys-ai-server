# Agent Evolution: From Tool User to Autonomous Programmer

## Status

Proposed

## Context

The current MVP architecture for the MolSys-AI agent is based on a simple, custom-built ReAct-style loop. The agent, guided by a heuristic-based `SimplePlanner`, can select and execute high-level, pre-defined tools registered in a `ToolExecutor`.

This architecture is robust and sufficient for fulfilling the initial project roadmap (v0.1-v0.5), which focuses on validating the end-to-end execution of a few specific workflows.

However, this design has a key limitation: for every new complex workflow (e.g., "read a paper, run a benchmark, and report the results"), a new specialized, high-level tool must be manually coded and registered. This does not scale and limits the agent's autonomy and flexibility.

## Vision: The "Agent as Programmer"

The long-term vision is to evolve the agent from a mere "user of tools" to a "creator of workflows". The agent should be capable of handling complex, unseen tasks by dynamically generating and executing code that composes primitive building blocks.

In this paradigm, when faced with a task for which no single tool exists, the agent should reason like a programmer: decompose the problem, write a script using a library of low-level functions, execute it, and interpret the results.

## Proposed Evolution Roadmap

To achieve this vision, we propose a phased evolution building upon the existing architecture.

### Phase 1: Consolidate the MVP (Current Goal)

*   **Objective**: Fully implement the `v0.1 - v0.5` roadmap defined in `dev/ROADMAP.md`. This includes a functional agent with a few high-level tools, a working RAG system, and the first fine-tuned model via QLoRA.
*   **Outcome**: A stable, robust agent that can reliably use a small set of pre-defined tools. This validates the core ReAct loop and infrastructure.

### Phase 2: Implement the LLM-Powered Planner

*   **Objective**: Execute the plan documented in `dev/OPEN_QUESTIONS.md` (section 5). Replace the heuristic `SimplePlanner` with an LLM-based planner.
*   **Structural Changes**:
    *   The `MolSysAIAgent`'s `chat_with_planning` method will be updated to call the new planner.
    *   The new planner will receive the user query and a "menu" of available tools (formatted in the system prompt) and will output a structured tool call (e.g., JSON).
*   **Outcome**: An agent that can flexibly choose from a larger set of pre-defined tools using natural language, but is still limited to calling one pre-defined tool at a time.

### Phase 3: Granularize the Toolset & Introduce Code Execution

This is the most significant architectural leap.

*   **Objective**: Shift the philosophy from high-level tools to primitive "API-like" tools and give the agent the ability to execute code.
*   **Structural Changes**:
    *   **Granular Tools**: Decompose high-level tools into their primitive components. Instead of a single `run_topomt_benchmark` tool, the agent will be provided with a library of functions mirroring the `molsysmt` and `topomt` APIs (e.g., `molsysmt.load`, `topomt.detect_pockets`, `topomt.get_pocket_volume`).
    *   **Code Execution Tool**: Implement and register a new, critical tool: `python_interpreter(code: str)`. This tool must execute the provided Python code string in a secure, sandboxed environment to prevent security risks.
*   **Outcome**: The agent's available toolset now consists of low-level API functions and a powerful, general-purpose code interpreter.

### Phase 4: Evolve to "Agent as Programmer"

*   **Objective**: Leverage the new components to enable autonomous workflow generation.
*   **Structural Changes**:
    *   **Prompt Engineering**: The agent's main system prompt will be fundamentally changed. It will be instructed to "think like a programmer" and to write and execute Python code using the available API-like tools to solve multi-step problems for which no single tool exists.
    *   **Agent Loop Enhancement (Optional but Recommended)**: The ReAct loop in `MolSysAIAgent` could be enhanced to support debugging. If the `python_interpreter` returns an error, the error message should be fed back to the LLM in the next turn, with the instruction "The code failed with this error. Please fix it and try again."
*   **Outcome**: A truly autonomous agent that can fulfill complex requests, like the "benchmark a scientific paper" example, by dynamically writing, executing, and even debugging its own code.

---

### Vision Example: Autonomous Scientific Benchmark

To make the final goal concrete, consider the following user request:

> "Este pdf es un artículo científico de comparación entre herramientas de detección de pockets. Lee el artículo y extrae la batería de pdb ids de sistemas de prueba para el benchmark, así como los datos cuantitativos sobre la performance de cada librería comparada. Corre esos mismos tests en TopoMT (herramienta de MolSysSuite), saca los mismos datos cuantitativos para comparar y haz un reporte breve de comparación entre los resultados publicados y los obtenidos con TopoMT."

An agent that has completed the evolution to "Agent as Programmer" would handle this as follows:

1.  **Planning**: The agent's LLM planner determines that no single tool can fulfill this request. It decides to use the `python_interpreter` tool and formulates a multi-step plan.

2.  **Step-wise Execution & Code Generation**: The agent writes and executes code to solve each sub-task:
    *   It first uses a `read_pdf` tool to get the article's text.
    *   It processes this text to extract the list of PDB IDs and the performance data from the published tables.
    *   It then generates a Python script that iterates through the extracted PDB IDs. Inside the loop, it uses the primitive tools `molsysmt.load()` and `topomt.detect_pockets()` to run the analysis with the local library.
    *   The script aggregates the new results into a data structure (e.g., a dictionary).

3.  **Synthesis**: The agent receives the result from its script execution. It now has the data from the paper and the newly generated data for `TopoMT` in its context. It uses this complete information to reason about the comparison and generate the final, brief report in natural language as requested by the user.

This entire workflow is achieved autonomously, orchestrated by the agent's reasoning capabilities, without requiring a developer to first create a specific "benchmark" tool. This demonstrates the power and flexibility of the "Agent as Programmer" paradigm.
