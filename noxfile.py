#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox>=2025.10.14"]
# ///

import nox


nox.needs_version = ">=2025.10.14"
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session(default=True)
def py_compile(session: nox.Session) -> None:
    """Compile-check Python source files without installing runtime dependencies."""
    session.run("python", "-m", "compileall", "-q", "scripts", "src/python", env={"PYTHONPATH": "src"})


@nox.session(default=True)
def tests(session: nox.Session) -> None:
    """Run the Python test suite."""
    session.install(".[test]")
    session.run("pytest", "src/python/tests", env={"PYTHONPATH": "src"})


@nox.session
def docs(session: nox.Session) -> None:
    """Build the Sphinx/MyST documentation."""
    session.install(".[docs]")
    session.run("sphinx-build", "-b", "html", "-W", "--keep-going", "docs", "docs/_build/html")


@nox.session
def precommit(session: nox.Session) -> None:
    """Run pre-commit hooks across the repository."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")


if __name__ == "__main__":
    nox.main()
