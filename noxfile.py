import nox


nox.options.sessions = ["py_compile", "tests"]


@nox.session
def py_compile(session: nox.Session) -> None:
    """Compile-check Python source files without installing runtime dependencies."""
    session.run("python", "-m", "compileall", "-q", "scripts", "python", "tests")


@nox.session
def tests(session: nox.Session) -> None:
    """Run the Python test suite."""
    session.install(".[test]")
    session.run("pytest", "tests", "python/tests")


@nox.session
def precommit(session: nox.Session) -> None:
    """Run pre-commit hooks across the repository."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")
