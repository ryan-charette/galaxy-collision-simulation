from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

project = "Galaxy Collision Simulation"
author = "Ryan Charette"
copyright = "2026, Ryan Charette"

extensions = [
    "breathe",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
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
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

doxygen_executable = os.environ.get("DOXYGEN_EXECUTABLE") or shutil.which("doxygen")
if doxygen_executable is None:
    raise RuntimeError(
        "Doxygen is required to build the C++ API reference. Install Doxygen "
        "or set DOXYGEN_EXECUTABLE to the doxygen binary, then rerun `nox -s docs`."
    )

doxygen_output = repo_root / "docs" / "_build" / "doxygen"
doxygen_output.mkdir(parents=True, exist_ok=True)

subprocess.run(
    [doxygen_executable, str(repo_root / "Doxyfile")],
    cwd=repo_root,
    check=True,
)

breathe_projects = {
    "fmmgalaxy": str(doxygen_output / "xml"),
}
breathe_default_project = "fmmgalaxy"

autodoc_typehints = "description"
autosummary_generate = True
