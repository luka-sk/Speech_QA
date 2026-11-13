from pathlib import Path

import pytest

from speech_qa.domain.errors import (
    ConfigurationError,
    IndexIncompatibleError,
    IndexLockedError,
    IndexNotFoundError,
    ModelNotAvailableError,
    ModelServiceUnavailableError,
    SpeechQAError,
)
from speech_qa.domain.models import ModelPurpose


@pytest.mark.parametrize(
    ("error", "expected_text"),
    [
        (ConfigurationError("speech_qa.toml", ["a: bad", "b: worse"]), "a: bad; b: worse"),
        (IndexNotFoundError(Path("var/index")), "var/index"),
        (IndexIncompatibleError("embedding model differs"), "embedding model differs"),
        (IndexLockedError(Path("var/index")), "another process"),
        (ModelServiceUnavailableError("http://localhost:11434/v1", "refused"), "refused"),
        (ModelNotAvailableError("bge-m3", ModelPurpose.EMBEDDING), "embedding model 'bge-m3'"),
    ],
)
def test_errors_are_speech_qa_errors_with_a_technical_description(
    error: SpeechQAError, expected_text: str
) -> None:
    assert isinstance(error, SpeechQAError)
    assert expected_text in str(error)


def test_model_not_available_error_keeps_structured_data() -> None:
    error = ModelNotAvailableError("gemma4:12b", ModelPurpose.CHAT)

    assert error.model == "gemma4:12b"
    assert error.purpose is ModelPurpose.CHAT
