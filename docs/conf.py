from __future__ import annotations

from pathlib import Path

project = "Galaxy Collision Simulation"
author = "Ryan Charette"
copyright = "2026, Ryan Charette"

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "Galaxy Collision Simulation"
html_static_path: list[str] = []

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

autosectionlabel_prefix_document = True

nitpicky = False
suppress_warnings = [
    "myst.header",
]

repo_root = Path(__file__).resolve().parents[1]
