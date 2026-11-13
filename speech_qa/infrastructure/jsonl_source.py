"""Reading speeches from a JSON Lines file.

Every line is validated on its own. Invalid lines are reported with their line
number and skipped; in strict mode the first invalid line stops the reading,
before anything is written to the index.
"""

import datetime as dt
import json
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from speech_qa.domain.errors import InputFileError, InputFileProblem
from speech_qa.domain.models import InputIssue, IssueKind, Speech

MAX_LINE_BYTES = 1_000_000
"""Longer lines are rejected; protects memory against malformed files."""

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}$"
"""Allowed speech ids: they appear in citations, so they must be simple tokens."""


class _SpeechRecord(BaseModel):
    """Schema of one line of the input file."""

    model_config = ConfigDict(extra="ignore", frozen=True, str_strip_whitespace=True)

    id: str = Field(pattern=ID_PATTERN)
    speaker: str = Field(min_length=1, max_length=200)
    party: str | None = Field(default=None, max_length=100)
    date: dt.date
    text: str = Field(min_length=1)

    @field_validator("speaker", "party", "text", mode="before")
    @classmethod
    def _normalize_unicode(cls, value: object) -> object:
        # Composed (NFC) form, so that "Doležal" typed in a query matches the data.
        return unicodedata.normalize("NFC", value) if isinstance(value, str) else value

    @field_validator("speaker")
    @classmethod
    def _collapse_spaces(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("party")
    @classmethod
    def _empty_party_means_none(cls, value: str | None) -> str | None:
        return value or None


@dataclass(frozen=True, slots=True)
class ReadResult:
    """Speeches read from a file and the problems with skipped lines."""

    speeches: tuple[Speech, ...]
    issues: tuple[InputIssue, ...]


def read_speeches(path: Path, *, strict: bool = False) -> ReadResult:
    """Read and validate speeches from a JSON Lines file.

    Args:
        path: The input file (UTF-8, one JSON object per line).
        strict: Stop at the first invalid line instead of skipping it.

    Returns:
        The valid speeches in file order and the problems with skipped lines.

    Raises:
        InputFileError: If the file is missing or unreadable, contains no valid
            speech, or (in strict mode) contains an invalid line.
    """
    if not path.exists():
        raise InputFileError(path, InputFileProblem.NOT_FOUND)
    if not path.is_file():
        raise InputFileError(path, InputFileProblem.NOT_A_FILE)

    speeches: list[Speech] = []
    issues: list[InputIssue] = []
    first_line_of_id: dict[str, int] = {}
    try:
        with path.open("rb") as handle:
            for line_number, raw in _numbered_lines(handle):
                speech, issue = _parse_line(line_number, raw, first_line_of_id)
                if issue is not None:
                    if strict:
                        raise InputFileError(path, InputFileProblem.INVALID_LINE, (issue,))
                    issues.append(issue)
                elif speech is not None:
                    first_line_of_id[speech.speech_id] = line_number
                    speeches.append(speech)
    except OSError as exc:
        raise InputFileError(path, InputFileProblem.UNREADABLE, detail=str(exc.strerror)) from exc

    if not speeches:
        raise InputFileError(path, InputFileProblem.NO_VALID_SPEECHES, tuple(issues))
    return ReadResult(speeches=tuple(speeches), issues=tuple(issues))


def _numbered_lines(handle: BinaryIO) -> Iterator[tuple[int, bytes | None]]:
    """Yield ``(line number, line)``; ``None`` stands for a line over the size limit."""
    line_number = 0
    while True:
        line = handle.readline(MAX_LINE_BYTES + 1)
        if not line:
            return
        line_number += 1
        if len(line) > MAX_LINE_BYTES:
            # Skip the rest of the over-long line without loading it into memory.
            while line and not line.endswith(b"\n"):
                line = handle.readline(MAX_LINE_BYTES)
            yield line_number, None
        else:
            yield line_number, line


def _parse_line(
    line_number: int, raw: bytes | None, first_line_of_id: dict[str, int]
) -> tuple[Speech | None, InputIssue | None]:
    """Parse one line; returns ``(None, None)`` for a blank line."""
    if raw is None:
        return None, InputIssue(line_number, IssueKind.LINE_TOO_LONG)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, InputIssue(line_number, IssueKind.NOT_UTF8)
    text = text.removeprefix("﻿").strip()
    if not text:
        return None, None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, InputIssue(line_number, IssueKind.INVALID_JSON, exc.msg)
    if not isinstance(data, dict):
        return None, InputIssue(line_number, IssueKind.NOT_AN_OBJECT)
    try:
        record = _SpeechRecord.model_validate(data)
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return None, InputIssue(line_number, IssueKind.INVALID_FIELDS, detail)
    if record.id in first_line_of_id:
        detail = f"{record.id} (first on line {first_line_of_id[record.id]})"
        return None, InputIssue(line_number, IssueKind.DUPLICATE_ID, detail)
    speech = Speech(
        speech_id=record.id,
        speaker=record.speaker,
        party=record.party,
        date=record.date,
        text=record.text,
    )
    return speech, None
