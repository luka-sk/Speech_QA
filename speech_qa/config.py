"""Configuration: a TOML file plus environment overrides, validated at startup.

Resolution order (later wins):

1. built-in defaults (the same values as in ``speech_qa.toml``),
2. the configuration file,
3. environment variables named ``SPEECH_QA__<SECTION>__<KEY>``.

The configuration file is ``$SPEECH_QA_CONFIG`` if that variable is set,
otherwise ``speech_qa.toml`` in the current directory if it exists; without a
file the built-in defaults are used. Unknown sections and keys are rejected so
that typos do not go unnoticed. Relative paths are resolved against the
directory of the configuration file (or the current directory without one).
"""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from speech_qa.domain.errors import ConfigurationError

CONFIG_PATH_ENV_VAR = "SPEECH_QA_CONFIG"
"""Environment variable selecting the configuration file."""

ENV_OVERRIDE_PREFIX = "SPEECH_QA__"
"""Prefix of environment variables overriding single configuration values."""

DEFAULT_CONFIG_FILENAME = "speech_qa.toml"
"""Configuration file looked up in the current directory."""

_SECTION_AND_KEY = 2
"""Number of ``__``-separated parts after the override prefix."""


class _Section(BaseModel):
    """Base of all configuration sections: immutable, unknown keys rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class OllamaSettings(_Section):
    """Connection to the OpenAI-compatible Ollama endpoint."""

    base_url: str = "http://localhost:11434/v1"
    chat_model: str = Field(default="gemma4:12b", min_length=1)
    embedding_model: str = Field(default="bge-m3", min_length=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            msg = "must be an http:// or https:// URL"
            raise ValueError(msg)
        return value.rstrip("/")


class GenerationSettings(_Section):
    """How answers are generated."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none"
    max_answer_tokens: int = Field(default=800, gt=0)
    context_window_tokens: int = Field(default=4096, gt=0)

    @model_validator(mode="after")
    def _check_answer_fits_context(self) -> Self:
        if self.max_answer_tokens >= self.context_window_tokens:
            msg = "max_answer_tokens must be smaller than context_window_tokens"
            raise ValueError(msg)
        return self


class IndexSettings(_Section):
    """Location of the local index."""

    path: Path = Path(".speech_qa/index")
    collection: str = Field(default="speech_chunks", pattern=r"^[A-Za-z0-9_-]{1,64}$")


class IngestSettings(_Section):
    """Ingestion of the corpus."""

    embedding_batch_size: int = Field(default=32, ge=1, le=1024)


class ChunkingSettings(_Section):
    """Splitting of speeches into sentence windows."""

    target_chars: int = Field(default=1000, ge=100)
    max_chars: int = Field(default=1500, ge=100)
    overlap_sentences: int = Field(default=1, ge=0, le=10)

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        if self.target_chars > self.max_chars:
            msg = "target_chars must not be larger than max_chars"
            raise ValueError(msg)
        return self


class RetrievalSettings(_Section):
    """Search for relevant speeches."""

    candidate_chunks: int = Field(default=50, ge=1, le=1000)
    max_speeches: int = Field(default=8, ge=1, le=100)
    max_speeches_per_speaker: int = Field(default=2, ge=1)
    max_passages_per_speech: int = Field(default=2, ge=1, le=20)

    @model_validator(mode="after")
    def _check_limits(self) -> Self:
        if self.candidate_chunks < self.max_speeches:
            msg = "candidate_chunks must be at least max_speeches"
            raise ValueError(msg)
        return self


class QuerySettings(_Section):
    """Limits on user queries."""

    max_chars: int = Field(default=1000, ge=10, le=100_000)


class LoggingSettings(_Section):
    """Technical logging to stderr."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"


class Settings(_Section):
    """The complete, validated configuration of the application."""

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    query: QuerySettings = Field(default_factory=QuerySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_settings(
    config_file: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Settings:
    """Load and validate the configuration.

    Args:
        config_file: Explicit configuration file; must exist if given.
        environ: Environment variables; defaults to ``os.environ``.
        cwd: Directory used to look up the default file and to resolve relative
            paths without a file; defaults to the current directory.

    Returns:
        The validated settings with absolute paths.

    Raises:
        ConfigurationError: If the file is missing, unreadable or invalid, or if
            an environment override is malformed or invalid.
    """
    env = os.environ if environ is None else environ
    working_dir = Path.cwd() if cwd is None else cwd
    path = find_config_file(config_file, environ=env, cwd=working_dir)
    file_values = _read_config_file(path) if path else {}
    overrides = _environment_overrides(env)

    source = str(path) if path else "built-in defaults"
    if overrides:
        source += " + environment"
    try:
        settings = Settings.model_validate(_merge(file_values, overrides))
    except ValidationError as exc:
        raise ConfigurationError(source, _describe(exc)) from exc
    return _with_absolute_paths(settings, path.parent if path else working_dir)


def find_config_file(
    config_file: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Path | None:
    """Return the configuration file that :func:`load_settings` would read.

    Returns:
        The file, or ``None`` when the built-in defaults are used.

    Raises:
        ConfigurationError: If an explicitly selected file does not exist.
    """
    env = os.environ if environ is None else environ
    working_dir = Path.cwd() if cwd is None else cwd
    if config_file is not None:
        candidate, explicit = config_file, True
    elif env.get(CONFIG_PATH_ENV_VAR):
        candidate, explicit = Path(env[CONFIG_PATH_ENV_VAR]), True
    else:
        candidate, explicit = Path(DEFAULT_CONFIG_FILENAME), False
    if not candidate.is_absolute():
        candidate = working_dir / candidate
    if candidate.is_file():
        return candidate
    if explicit:
        raise ConfigurationError(str(candidate), ["the file does not exist"])
    return None


def _read_config_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(str(path), [f"invalid TOML: {exc}"]) from exc
    except OSError as exc:
        raise ConfigurationError(str(path), [f"the file cannot be read: {exc.strerror}"]) from exc


def _environment_overrides(env: Mapping[str, str]) -> dict[str, dict[str, str]]:
    overrides: dict[str, dict[str, str]] = {}
    for name, value in env.items():
        if not name.upper().startswith(ENV_OVERRIDE_PREFIX):
            continue
        parts = name[len(ENV_OVERRIDE_PREFIX) :].lower().split("__")
        if len(parts) != _SECTION_AND_KEY or not all(parts):
            raise ConfigurationError(
                "environment", [f"{name}: expected the form {ENV_OVERRIDE_PREFIX}<SECTION>__<KEY>"]
            )
        section, key = parts
        overrides.setdefault(section, {})[key] = value
    return overrides


def _merge(file_values: dict[str, Any], overrides: dict[str, dict[str, str]]) -> dict[str, Any]:
    merged = dict(file_values)
    for section, values in overrides.items():
        existing = merged.get(section)
        merged[section] = {**existing, **values} if isinstance(existing, dict) else values
    return merged


def _describe(error: ValidationError) -> list[str]:
    details = []
    for problem in error.errors():
        location = ".".join(str(part) for part in problem["loc"]) or "<root>"
        details.append(f"{location}: {problem['msg']}")
    return details


def _with_absolute_paths(settings: Settings, base_dir: Path) -> Settings:
    index_path = settings.index.path
    if index_path.is_absolute():
        return settings
    index = settings.index.model_copy(update={"path": (base_dir / index_path).resolve()})
    return settings.model_copy(update={"index": index})
