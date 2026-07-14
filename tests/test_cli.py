from pathlib import Path

from typer.testing import CliRunner

from autodoc_harness import __version__
from autodoc_harness.cli import app
from autodoc_harness.config import CONFIG_FILENAME

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_init_writes_starter_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--target", str(tmp_path)])
    assert result.exit_code == 0
    config_path = tmp_path / CONFIG_FILENAME
    assert config_path.exists()
    assert "entry_points" in config_path.read_text()


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text("target_repo: .\n")
    result = runner.invoke(app, ["init", "--target", str(tmp_path)])
    assert result.exit_code != 0
    result = runner.invoke(app, ["init", "--target", str(tmp_path), "--force"])
    assert result.exit_code == 0


def test_validate_reports_missing_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_validate_passes_for_scaffolded_config(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")
    runner.invoke(app, ["init", "--target", str(tmp_path)])
    result = runner.invoke(app, ["validate", "--config", str(tmp_path / CONFIG_FILENAME)])
    assert result.exit_code == 0
    assert "Config is valid" in result.output
