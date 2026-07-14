"""Config schema, YAML loading, and semantic (env/filesystem) validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_core import ErrorDetails

CONFIG_FILENAME = ".autodoc.yaml"

STAGE_NAMES = ("master_explorer", "code_explorer", "synthesizer", "reviewer")

# Bare directory names (no trailing "/**") so the pattern matches the directory
# entry itself as well as everything beneath it - a leading/trailing "/**" only
# matches descendants, per git's own gitignore semantics.
DEFAULT_IGNORE_GLOBS: tuple[str, ...] = (
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    "vendor",
    "site-packages",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)


class PathNote(BaseModel):
    """A repo-relative path plus an optional note explaining its relevance.

    Used for both `entry_points` (traversal starting points) and `hints`
    (additional locations worth checking that aren't necessarily reachable by
    following imports from an entry point - e.g. config files, or anything else
    referenced only by dynamic/string paths rather than static imports).

    Accepts a bare string in YAML (`- src/main.py`) as shorthand for
    `{path: src/main.py, note: null}`, so a note is opt-in, not required.
    """

    path: str
    note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_string(cls, data: object) -> object:
        if isinstance(data, str):
            return {"path": data}
        return data


class ModelStageOverride(BaseModel):
    """Per-stage overrides layered on top of the top-level `model` config."""

    name: str | None = None
    api_key_env: str | None = None
    api_base: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None


class ResolvedModelConfig(BaseModel):
    """A `ModelConfig` with a specific stage's overrides already applied."""

    name: str
    api_key_env: str | None
    api_base: str | None = None
    temperature: float
    max_tokens: int
    timeout: float


class ModelConfig(BaseModel):
    name: str
    # None for keyless local/self-hosted providers (e.g. Ollama) - litellm needs no
    # API key for those. Providers that do need one should set this to the env var
    # holding it; `validate_config_semantics` only checks env vars that are named.
    api_key_env: str | None = None
    # e.g. "http://localhost:11434" for a local Ollama server. None lets litellm use
    # its own provider default.
    api_base: str | None = None
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: float = 120.0
    stage_overrides: dict[str, ModelStageOverride] = Field(default_factory=dict)

    def resolved_for_stage(self, stage: str) -> ResolvedModelConfig:
        override = self.stage_overrides.get(stage)
        return ResolvedModelConfig(
            name=(override.name if override and override.name else self.name),
            api_key_env=(
                override.api_key_env if override and override.api_key_env else self.api_key_env
            ),
            api_base=(override.api_base if override and override.api_base else self.api_base),
            temperature=(
                override.temperature
                if override and override.temperature is not None
                else self.temperature
            ),
            max_tokens=(
                override.max_tokens
                if override and override.max_tokens is not None
                else self.max_tokens
            ),
            timeout=(
                override.timeout if override and override.timeout is not None else self.timeout
            ),
        )


class GuardrailsConfig(BaseModel):
    max_files_per_component: int = 40
    max_tool_calls_master_explorer: int = 60
    max_tool_calls_per_code_explorer: int = 30
    max_tool_calls_reviewer_per_doc: int = 20
    max_extraction_attempts: int = 3
    max_file_bytes: int = 51_200
    max_total_cost_usd: float = 5.00
    max_run_seconds: int = 1800
    max_parallel_code_explorers: int = 5
    max_traversal_depth: int = 6


class OutputConfig(BaseModel):
    dir: str = "docs"
    overwrite: bool = True

    @model_validator(mode="after")
    def _overwrite_must_be_true(self) -> OutputConfig:
        if not self.overwrite:
            raise ValueError(
                "output.overwrite=false is not supported yet (this MVP is one-shot "
                "only); set it to true, or omit the field"
            )
        return self


class LoggingConfig(BaseModel):
    level: Literal["debug", "info", "warning", "error"] = "info"
    log_file: str | None = None


class Config(BaseModel):
    target_repo: Path
    # Free-text context about what the system does - without this, exploration
    # stages have nothing but file contents to infer intent from, which can lead
    # to correct-but-oddly-framed component names/summaries that don't match the
    # domain's own vocabulary.
    description: str | None = None
    entry_points: list[PathNote] = Field(min_length=1)
    # Locations worth checking that aren't traversal starting points themselves -
    # e.g. config files or anything else not reachable by following imports from
    # an entry point. Master Explorer decides whether/how they relate to what it
    # finds; unlike entry_points these may be directories, not just files.
    hints: list[PathNote] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)
    model: ModelConfig
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    ignore_globs: list[str] = Field(default_factory=list)
    honor_gitignore: bool = True
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def all_ignore_globs(self) -> list[str]:
        return [*DEFAULT_IGNORE_GLOBS, *self.ignore_globs]


class ConfigError(Exception):
    """Raised when a config file fails to load or validate.

    Collects every issue found rather than stopping at the first, so `validate`
    can report a complete punch list in one pass.
    """

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


def _format_pydantic_error(err: ErrorDetails) -> str:
    loc = ".".join(str(p) for p in err["loc"]) if err["loc"] else "<root>"
    return f"{loc}: {err['msg']}"


def load_config(path: Path) -> Config:
    """Parse and schema-validate a config file. Raises `ConfigError` on failure.

    Does not touch the filesystem beyond reading `path` itself, and does not check
    environment variables - see `validate_config_semantics` for those checks.
    """
    if not path.is_file():
        raise ConfigError([f"config file not found: {path}"])

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError([f"invalid YAML in {path}: {e}"]) from e

    if not isinstance(raw, dict):
        raise ConfigError([f"config file {path} must contain a YAML mapping at the top level"])

    if "target_repo" in raw:
        target_repo = Path(str(raw["target_repo"])).expanduser()
        if not target_repo.is_absolute():
            target_repo = (path.parent / target_repo).resolve()
        raw["target_repo"] = target_repo

    try:
        return Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError([_format_pydantic_error(err) for err in e.errors()]) from e


def validate_config_semantics(config: Config) -> list[str]:
    """Beyond-schema checks: does the target repo/entry points/env vars actually exist.

    Deliberately separate from `load_config` so tests and `--dry-run` can construct
    and use a `Config` without needing a real repo on disk or real env vars set.
    """
    issues: list[str] = []

    if not config.target_repo.is_dir():
        issues.append(f"target_repo does not exist or is not a directory: {config.target_repo}")
    else:
        for entry_point in config.entry_points:
            entry_point_path = config.target_repo / entry_point.path
            if not entry_point_path.is_file():
                issues.append(
                    f"entry point file not found: {entry_point.path} "
                    f"(resolved to {entry_point_path})"
                )
        for hint in config.hints:
            hint_path = config.target_repo / hint.path
            if not hint_path.exists():
                issues.append(f"hint path not found: {hint.path} (resolved to {hint_path})")

    env_vars_needed = {
        env_var
        for stage in STAGE_NAMES
        if (env_var := config.model.resolved_for_stage(stage).api_key_env) is not None
    }
    for env_var in sorted(env_vars_needed):
        if not os.environ.get(env_var):
            issues.append(f"environment variable {env_var} is not set (needed for model API key)")

    return issues
