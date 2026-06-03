from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_quality_scaffolding_exists() -> None:
    assert (ROOT / ".pre-commit-config.yaml").is_file()
    assert (ROOT / "noxfile.py").is_file()
    assert (ROOT / "python" / "tests").is_dir()
