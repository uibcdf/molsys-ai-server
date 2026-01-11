"""Configuration file for the MolSys-AI Sphinx documentation (pilot).

This small Sphinx project is intended as a proof of concept for embedding
the MolSys-AI documentation chatbot into Sphinx-generated HTML pages.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

project = "MolSys-AI Docs Pilot"
author = "UIBCDF"
year = datetime.now().year
copyright = f"{year}, {author}"

extensions = []

# Prefer myst-nb (supports MyST Markdown + notebooks). Fall back to myst-parser.
if importlib.util.find_spec("myst_nb") is not None:
    extensions.append("myst_nb")
else:
    extensions.append("myst_parser")

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "pydata_sphinx_theme"

# Static files (CSS, JS). We include the web widget from the repository so
# that Sphinx copies it into `_static` and can be referenced on the pages.
html_static_path = ["_static", "../server/web_widget"]
html_css_files = [
    "molsys_ai_pilot.css",
]
html_js_files = [
    "molsys_ai_config.js",
    "molsys_ai_widget.js",
]
