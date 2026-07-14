from pathlib import Path

import pytest

from autodoc_harness.config import (
    Config,
    ConfigError,
    load_config,
    validate_config_semantics,
)

VALID_YAML = """\
target_repo: {target_repo}
entry_points:
  - main.py
model:
  name: anthropic/claude-sonnet-4-5
  api_key_env: ANTHROPIC_API_KEY
"""


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / ".autodoc.yaml"
    config_path.write_text("target_repo: [unterminated")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(config_path)


def test_load_config_not_a_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / ".autodoc.yaml"
    config_path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(config_path)


def test_load_config_missing_required_field(tmp_path: Path) -> None:
    config_path = tmp_path / ".autodoc.yaml"
    config_path.write_text("entry_points:\n  - main.py\n")
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path)
    assert any("target_repo" in issue for issue in exc_info.value.issues)


def test_load_config_resolves_relative_target_repo(tmp_path: Path) -> None:
    (tmp_path / "repo").mkdir()
    config_path = tmp_path / ".autodoc.yaml"
    config_path.write_text(VALID_YAML.format(target_repo="./repo"))
    cfg = load_config(config_path)
    assert cfg.target_repo == (tmp_path / "repo").resolve()


def test_output_overwrite_false_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / ".autodoc.yaml"
    config_path.write_text(
        VALID_YAML.format(target_repo=str(tmp_path)) + "output:\n  overwrite: false\n"
    )
    with pytest.raises(ConfigError, match="not supported yet"):
        load_config(config_path)


def test_stage_override_falls_back_to_top_level() -> None:
    cfg = Config.model_validate(
        {
            "target_repo": "/tmp/repo",
            "entry_points": ["main.py"],
            "model": {
                "name": "anthropic/claude-sonnet-4-5",
                "api_key_env": "ANTHROPIC_API_KEY",
                "temperature": 0.2,
                "stage_overrides": {"reviewer": {"temperature": 0.0}},
            },
        }
    )
    reviewer_cfg = cfg.model.resolved_for_stage("reviewer")
    assert reviewer_cfg.temperature == 0.0
    assert reviewer_cfg.name == "anthropic/claude-sonnet-4-5"

    explorer_cfg = cfg.model.resolved_for_stage("master_explorer")
    assert explorer_cfg.temperature == 0.2


def test_validate_semantics_reports_missing_repo_and_entry_point_and_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = Config.model_validate(
        {
            "target_repo": str(tmp_path / "does-not-exist"),
            "entry_points": ["main.py"],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )
    issues = validate_config_semantics(cfg)
    assert any("target_repo does not exist" in issue for issue in issues)
    assert any("ANTHROPIC_API_KEY" in issue for issue in issues)


def test_validate_semantics_reports_missing_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = Config.model_validate(
        {
            "target_repo": str(tmp_path),
            "entry_points": ["missing.py"],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )
    issues = validate_config_semantics(cfg)
    assert any("entry point file not found" in issue for issue in issues)


def test_validate_semantics_passes_for_good_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    (tmp_path / "main.py").write_text("print('hi')\n")
    cfg = Config.model_validate(
        {
            "target_repo": str(tmp_path),
            "entry_points": ["main.py"],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )
    assert validate_config_semantics(cfg) == []


def test_local_provider_needs_no_api_key_env(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hi')\n")
    cfg = Config.model_validate(
        {
            "target_repo": str(tmp_path),
            "entry_points": ["main.py"],
            "model": {
                "name": "ollama_chat/gemma4:e4b",
                "api_base": "http://localhost:11434",
            },
        }
    )
    assert cfg.model.api_key_env is None
    assert validate_config_semantics(cfg) == []


def test_api_base_flows_through_stage_resolution() -> None:
    cfg = Config.model_validate(
        {
            "target_repo": "/tmp/repo",
            "entry_points": ["main.py"],
            "model": {"name": "ollama_chat/gemma4:e4b", "api_base": "http://localhost:11434"},
        }
    )
    resolved = cfg.model.resolved_for_stage("master_explorer")
    assert resolved.api_base == "http://localhost:11434"
    assert resolved.api_key_env is None


def test_entry_points_accept_bare_strings_and_rich_form() -> None:
    cfg = Config.model_validate(
        {
            "target_repo": "/tmp/repo",
            "entry_points": [
                "src/main.py",
                {"path": "src/api/server.py", "note": "HTTP server entry point"},
            ],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )
    assert cfg.entry_points[0].path == "src/main.py"
    assert cfg.entry_points[0].note is None
    assert cfg.entry_points[1].path == "src/api/server.py"
    assert cfg.entry_points[1].note == "HTTP server entry point"


def test_description_and_hints_default_to_absent() -> None:
    cfg = Config.model_validate(
        {
            "target_repo": "/tmp/repo",
            "entry_points": ["main.py"],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )
    assert cfg.description is None
    assert cfg.hints == []


def test_hints_accept_bare_strings_and_rich_form() -> None:
    cfg = Config.model_validate(
        {
            "target_repo": "/tmp/repo",
            "entry_points": ["main.py"],
            "hints": ["src/config.py", {"path": "src/schema/", "note": "DB schema directory"}],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )
    assert cfg.hints[0].path == "src/config.py"
    assert cfg.hints[1].note == "DB schema directory"


def test_validate_semantics_reports_missing_hint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    (tmp_path / "main.py").write_text("print('hi')\n")
    cfg = Config.model_validate(
        {
            "target_repo": str(tmp_path),
            "entry_points": ["main.py"],
            "hints": ["does-not-exist.py"],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )
    issues = validate_config_semantics(cfg)
    assert any("hint path not found" in issue for issue in issues)


def test_validate_semantics_allows_hint_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    (tmp_path / "main.py").write_text("print('hi')\n")
    (tmp_path / "config").mkdir()
    cfg = Config.model_validate(
        {
            "target_repo": str(tmp_path),
            "entry_points": ["main.py"],
            "hints": ["config"],
            "model": {"name": "anthropic/claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
        }
    )
    assert validate_config_semantics(cfg) == []
