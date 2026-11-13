"""Expected errors of the application.

Errors carry structured data, not user-facing text: the command-line layer
turns them into (Czech) messages in one place. ``str(error)`` gives an English
description intended for technical logs.
"""

from enum import StrEnum
from pathlib import Path

from speech_qa.domain.models import InputIssue, ModelPurpose


class SpeechQAError(Exception):
    """Base class of all expected errors; anything else is a bug."""


class ConfigurationError(SpeechQAError):
    """The configuration file or an environment override is invalid.

    Attributes:
        source: Where the invalid configuration came from (file path or "environment").
        details: Human-readable descriptions of the individual problems.
    """

    def __init__(self, source: str, details: list[str]) -> None:
        self.source = source
        self.details = details
        super().__init__(f"invalid configuration in {source}: " + "; ".join(details))


class InvalidInputError(SpeechQAError):
    """User input (a query or an input file) cannot be processed."""


class EmptyQueryError(InvalidInputError):
    """The query contains no text."""

    def __init__(self) -> None:
        super().__init__("the query is empty")


class QueryTooLongError(InvalidInputError):
    """The query is longer than allowed.

    Attributes:
        length: Length of the query in characters.
        limit: Maximum allowed length.
    """

    def __init__(self, length: int, limit: int) -> None:
        self.length = length
        self.limit = limit
        super().__init__(f"the query has {length} characters, the limit is {limit}")


class InputFileProblem(StrEnum):
    """Why an input file cannot be ingested."""

    NOT_FOUND = "not_found"
    NOT_A_FILE = "not_a_file"
    UNREADABLE = "unreadable"
    NO_VALID_SPEECHES = "no_valid_speeches"
    INVALID_LINE = "invalid_line"
    """A line is invalid and strict mode is on."""


class InputFileError(InvalidInputError):
    """An input file cannot be ingested.

    Attributes:
        path: The input file.
        problem: Why the file cannot be ingested.
        issues: Problems found in individual lines, if any.
        detail: Technical detail (e.g. the operating system error).
    """

    def __init__(
        self,
        path: Path,
        problem: InputFileProblem,
        issues: tuple[InputIssue, ...] = (),
        detail: str = "",
    ) -> None:
        self.path = path
        self.problem = problem
        self.issues = issues
        self.detail = detail
        super().__init__(f"cannot ingest {path}: {problem}" + (f" ({detail})" if detail else ""))


class EvaluationDataError(InvalidInputError):
    """An evaluation question file is missing or invalid.

    Attributes:
        path: The file.
        problems: Descriptions of the problems, with line numbers where possible.
    """

    def __init__(self, path: Path, problems: list[str]) -> None:
        self.path = path
        self.problems = problems
        super().__init__(f"invalid evaluation file {path}: " + "; ".join(problems))


class OutputExistsError(SpeechQAError):
    """An output file already exists and overwriting was not requested.

    Attributes:
        path: The existing file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"{path} already exists")


class IndexNotFoundError(SpeechQAError):
    """No index exists yet; the corpus has to be ingested first.

    Attributes:
        path: Location where the index was expected.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"no index found at {path}")


class IndexIncompatibleError(SpeechQAError):
    """The index was built with different settings and must be rebuilt.

    Attributes:
        reason: What does not match (e.g. the embedding model).
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"index is incompatible with the configuration: {reason}")


class IndexLockedError(SpeechQAError):
    """The local index is being used by another running command.

    Attributes:
        path: Location of the locked index.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"index at {path} is used by another process")


class ModelServiceUnavailableError(SpeechQAError):
    """The model server (Ollama) cannot be reached or keeps failing.

    Attributes:
        base_url: Endpoint that was contacted.
        detail: Technical description of the failure.
    """

    def __init__(self, base_url: str, detail: str) -> None:
        self.base_url = base_url
        self.detail = detail
        super().__init__(f"model server at {base_url} is unavailable: {detail}")


class ModelNotAvailableError(SpeechQAError):
    """A required model is not installed on the model server.

    Attributes:
        model: Name of the missing model.
        purpose: What the model is needed for.
    """

    def __init__(self, model: str, purpose: ModelPurpose) -> None:
        self.model = model
        self.purpose = purpose
        super().__init__(f"{purpose} model {model!r} is not available on the model server")


class ModelRequestError(SpeechQAError):
    """The model server rejected a request or returned an unusable response.

    Attributes:
        model: Model the request was made for.
        detail: Technical description of the problem.
    """

    def __init__(self, model: str, detail: str) -> None:
        self.model = model
        self.detail = detail
        super().__init__(f"request for model {model!r} failed: {detail}")
