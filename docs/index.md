# MolSys-AI Documentation Pilot

This is a small Sphinx-based documentation site used to prototype the
MolSys-AI chatbot integration.

For now, the chatbot does not answer real questions; it simply shows a
friendly placeholder message (or, in backend mode, calls the docs-chat
backend). In future iterations it will call the MolSys-AI backend to
answer questions about the MolSys\* ecosystem documentation and workflows.


## AI helper (pilot)

The box below is a placeholder for the MolSys-AI documentation assistant:

```{raw} html
<div id="molsys-ai-chat"></div>
```


## Getting started

To build this documentation locally, from the repository root:

```bash
sphinx-build -b html docs docs/_build/html
```

The generated HTML will live under `docs/_build/html`.

