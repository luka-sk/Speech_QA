import os
import sys
from pathlib import Path

import pytest

from speech_qa.config import Settings, find_config_file, load_settings
from speech_qa.domain.errors import ConfigurationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def write_config(directory: Path, content: str, name: str = "speech_qa.toml") -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def test_defaults_are_used_without_a_config_file(tmp_path: Path) -> None:
    settings = load_settings(environ={}, cwd=tmp_path)

    assert settings.ollama.base_url == "http://localhost:11434/v1"
    assert settings.ollama.chat_model == "gemma4:12b"
    assert settings.ollama.embedding_model == "bge-m3"
    assert settings.generation.reasoning_effort == "none"
    assert settings.index.path == (tmp_path / ".speech_qa/index").resolve()


def test_shipped_config_file_matches_the_built_in_defaults() -> None:
    shipped = load_settings(REPOSITORY_ROOT / "speech_qa.toml", environ={})
    defaults = load_settings(environ={}, cwd=REPOSITORY_ROOT / "does-not-exist")

    assert shipped.model_dump(exclude={"index"}) == defaults.model_dump(exclude={"index"})
    assert shipped.index.path == (REPOSITORY_ROOT / ".speech_qa/index").resolve()


def test_config_file_in_working_directory_is_picked_up(tmp_path: Path) -> None:
    write_config(tmp_path, '[ollama]\nchat_model = "other-model"\n')

    settings = load_settings(environ={}, cwd=tmp_path)

    assert settings.ollama.chat_model == "other-model"
    assert settings.ollama.embedding_model == "bge-m3"


def test_config_file_can_be_selected_by_environment_variable(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[query]\nmax_chars = 500\n", name="custom.toml")

    settings = load_settings(environ={"SPEECH_QA_CONFIG": str(path)}, cwd=Path("/"))

    assert settings.query.max_chars == 500


def test_relative_index_path_is_resolved_against_the_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    path = write_config(config_dir, '[index]\npath = "data/index"\n')

    settings = load_settings(path, environ={}, cwd=tmp_path)

    assert settings.index.path == (config_dir / "data/index").resolve()


def test_absolute_index_path_is_kept(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    path = write_config(tmp_path, f"[index]\npath = '{target}'\n")

    assert load_settings(path, environ={}).index.path == target


def test_environment_overrides_file_values(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[ollama]\ntimeout_seconds = 30\n")

    settings = load_settings(
        path,
        environ={
            "SPEECH_QA__OLLAMA__TIMEOUT_SECONDS": "45",
            "SPEECH_QA__RETRIEVAL__MAX_SPEECHES": "5",
        },
    )

    assert settings.ollama.timeout_seconds == 45
    assert settings.retrieval.max_speeches == 5


def test_environment_override_names_are_case_insensitive(tmp_path: Path) -> None:
    settings = load_settings(environ={"speech_qa__logging__level": "DEBUG"}, cwd=tmp_path)

    assert settings.logging.level == "DEBUG"


def test_unrelated_environment_variables_are_ignored(tmp_path: Path) -> None:
    settings = load_settings(environ={"SPEECH_QA_OTHER": "x", "PATH": "/bin"}, cwd=tmp_path)

    assert settings == load_settings(environ={}, cwd=tmp_path)


@pytest.mark.parametrize(
    "name", ["SPEECH_QA__OLLAMA", "SPEECH_QA__OLLAMA__URL__X", "SPEECH_QA____"]
)
def test_malformed_environment_override_is_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_settings(environ={name: "x"}, cwd=tmp_path)

    assert caught.value.source == "environment"


@pytest.mark.parametrize(
    ("content", "expected_detail"),
    [
        ("[ollama]\nbse_url = 'http://x'\n", "ollama.bse_url: Extra inputs are not permitted"),
        ("[olama]\n", "olama: Extra inputs are not permitted"),
        ("[ollama]\ntimeout_seconds = 0\n", "ollama.timeout_seconds"),
        ("[ollama]\nbase_url = 'ftp://host'\n", "must be an http:// or https:// URL"),
        ("[generation]\nreasoning_effort = 'max'\n", "generation.reasoning_effort"),
        ("[generation]\nmax_answer_tokens = 5000\n", "smaller than context_window_tokens"),
        ("[chunking]\ntarget_chars = 2000\n", "target_chars must not be larger than max_chars"),
        ("[retrieval]\ncandidate_chunks = 3\n", "candidate_chunks must be at least max_speeches"),
        ("[index]\ncollection = 'bad name!'\n", "index.collection"),
        ("[logging]\nlevel = 'LOUD'\n", "logging.level"),
    ],
)
def test_invalid_values_are_reported_with_their_location(
    tmp_path: Path, content: str, expected_detail: str
) -> None:
    path = write_config(tmp_path, content)

    with pytest.raises(ConfigurationError) as caught:
        load_settings(path, environ={})

    assert caught.value.source == str(path)
    assert any(expected_detail in detail for detail in caught.value.details)


def test_invalid_environment_value_names_the_environment_as_source(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_settings(environ={"SPEECH_QA__QUERY__MAX_CHARS": "many"}, cwd=tmp_path)

    assert caught.value.source == "built-in defaults + environment"
    assert any(detail.startswith("query.max_chars") for detail in caught.value.details)


def test_invalid_toml_is_reported(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[ollama\n")

    with pytest.raises(ConfigurationError, match="invalid TOML"):
        load_settings(path, environ={})


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="file permissions are not enforced on Windows or for root",
)
def test_unreadable_config_file_is_reported(tmp_path: Path) -> None:
    path = write_config(tmp_path, "")
    path.chmod(0)
    try:
        with pytest.raises(ConfigurationError, match="cannot be read"):
            load_settings(path, environ={})
    finally:
        path.chmod(0o600)


def test_missing_explicit_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_settings(tmp_path / "missing.toml", environ={})


def test_missing_config_file_from_environment_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_settings(environ={"SPEECH_QA_CONFIG": "missing.toml"}, cwd=tmp_path)


def test_settings_are_immutable() -> None:
    settings = Settings()

    with pytest.raises(ValueError, match="frozen"):
        settings.query.max_chars = 5  # type: ignore[misc]


def test_find_config_file_reports_the_file_in_use(tmp_path: Path) -> None:
    default = write_config(tmp_path, "")
    custom = write_config(tmp_path, "", name="custom.toml")

    assert find_config_file(environ={}, cwd=tmp_path) == default
    assert find_config_file(custom, environ={}) == custom
    assert find_config_file(environ={"SPEECH_QA_CONFIG": "custom.toml"}, cwd=tmp_path) == custom
    assert find_config_file(environ={}, cwd=tmp_path / "nowhere") is None
