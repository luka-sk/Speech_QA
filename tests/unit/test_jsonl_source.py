import datetime as dt
import json
import unicodedata
from pathlib import Path

import pytest

from speech_qa.domain.errors import InputFileError, InputFileProblem
from speech_qa.domain.models import IssueKind
from speech_qa.infrastructure import jsonl_source
from speech_qa.infrastructure.jsonl_source import read_speeches
from tests.conftest import LONG_SPEECHES


def record(speech_id: str = "Q1", **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "id": speech_id,
        "speaker": "Lukáš Doležal",
        "party": "R7",
        "date": "2021-04-15",
        "text": "Kvalita rozhodnutí závisí na kvalitě informací.",
    }
    values.update(overrides)
    return values


def write_lines(tmp_path: Path, *lines: str | bytes) -> Path:
    path = tmp_path / "speeches.jsonl"
    with path.open("wb") as handle:
        for line in lines:
            handle.write(line if isinstance(line, bytes) else line.encode("utf-8"))
            handle.write(b"\n")
    return path


def test_reads_the_fixture_file() -> None:
    result = read_speeches(LONG_SPEECHES)

    assert [speech.speech_id for speech in result.speeches] == [f"F-{n:03}" for n in range(1, 11)]
    assert result.issues == ()
    first = result.speeches[0]
    assert first.speaker == "Lukáš Doležal"
    assert first.party == "R7"
    assert first.date == dt.date(2021, 4, 15)


def test_empty_party_means_a_speaker_without_party() -> None:
    speeches = {speech.speech_id: speech for speech in read_speeches(LONG_SPEECHES).speeches}

    assert speeches["F-009"].party is None


def test_missing_party_field_means_a_speaker_without_party(tmp_path: Path) -> None:
    data = record()
    del data["party"]
    path = write_lines(tmp_path, json.dumps(data))

    assert read_speeches(path).speeches[0].party is None


def test_ids_in_the_format_of_the_real_data_are_accepted(tmp_path: Path) -> None:
    real_id = "ParlaMint-CZ_2020-11-13-ps2017-062-10-007-503.u10"
    path = write_lines(tmp_path, json.dumps(record(real_id)))

    assert read_speeches(path).speeches[0].speech_id == real_id


def test_values_are_trimmed_and_normalised(tmp_path: Path) -> None:
    decomposed = unicodedata.normalize("NFD", "Lukáš  Doležal")
    path = write_lines(
        tmp_path, json.dumps(record(speaker=f" {decomposed} ", party=" R7 ", text=" Text. "))
    )

    speech = read_speeches(path).speeches[0]

    assert speech.speaker == "Lukáš Doležal"
    assert speech.party == "R7"
    assert speech.text == "Text."


def test_blank_lines_byte_order_mark_and_unknown_fields_are_accepted(tmp_path: Path) -> None:
    path = write_lines(
        tmp_path,
        "﻿" + json.dumps(record("Q1", extra="ignored")),
        "",
        "   ",
        json.dumps(record("Q2")),
    )

    result = read_speeches(path)

    assert [speech.speech_id for speech in result.speeches] == ["Q1", "Q2"]
    assert result.issues == ()


@pytest.mark.parametrize(
    ("line", "kind", "detail"),
    [
        (b'{"id": "Q2", ', IssueKind.INVALID_JSON, "Expecting"),
        (b"[1, 2]", IssueKind.NOT_AN_OBJECT, ""),
        (b"\xff\xfe", IssueKind.NOT_UTF8, ""),
        (json.dumps(record("Q2", date="2021-02-30")).encode(), IssueKind.INVALID_FIELDS, "date"),
        (json.dumps(record("Q2", text="")).encode(), IssueKind.INVALID_FIELDS, "text"),
        (json.dumps(record("Q 2")).encode(), IssueKind.INVALID_FIELDS, "id"),
        (json.dumps(record("Q2", speaker=None)).encode(), IssueKind.INVALID_FIELDS, "speaker"),
        (json.dumps(record("Q1")).encode(), IssueKind.DUPLICATE_ID, "first on line 1"),
    ],
)
def test_invalid_lines_are_skipped_and_reported(
    tmp_path: Path, line: bytes, kind: IssueKind, detail: str
) -> None:
    path = write_lines(tmp_path, json.dumps(record("Q1")), line, json.dumps(record("Q3")))

    result = read_speeches(path)

    assert [speech.speech_id for speech in result.speeches] == ["Q1", "Q3"]
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert (issue.line, issue.kind) == (2, kind)
    assert detail in issue.detail


def test_over_long_line_is_skipped_without_losing_the_next_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jsonl_source, "MAX_LINE_BYTES", 300)
    long_line = json.dumps(record("Q2", text="slovo " * 400))
    path = write_lines(tmp_path, json.dumps(record("Q1")), long_line, json.dumps(record("Q3")))

    result = read_speeches(path)

    assert [speech.speech_id for speech in result.speeches] == ["Q1", "Q3"]
    assert [(issue.line, issue.kind) for issue in result.issues] == [(2, IssueKind.LINE_TOO_LONG)]


def test_strict_mode_stops_at_the_first_invalid_line(tmp_path: Path) -> None:
    path = write_lines(tmp_path, json.dumps(record("Q1")), "not json", "also not json")

    with pytest.raises(InputFileError) as caught:
        read_speeches(path, strict=True)

    assert caught.value.problem is InputFileProblem.INVALID_LINE
    assert [issue.line for issue in caught.value.issues] == [2]


def test_file_without_valid_speech_is_an_error(tmp_path: Path) -> None:
    path = write_lines(tmp_path, "not json", "")

    with pytest.raises(InputFileError) as caught:
        read_speeches(path)

    assert caught.value.problem is InputFileProblem.NO_VALID_SPEECHES
    assert len(caught.value.issues) == 1


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(InputFileError) as caught:
        read_speeches(tmp_path / "missing.jsonl")

    assert caught.value.problem is InputFileProblem.NOT_FOUND


def test_directory_is_not_a_file(tmp_path: Path) -> None:
    with pytest.raises(InputFileError) as caught:
        read_speeches(tmp_path)

    assert caught.value.problem is InputFileProblem.NOT_A_FILE


def test_unreadable_file_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_lines(tmp_path, json.dumps(record()))

    def refuse(*args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "open", refuse)

    with pytest.raises(InputFileError) as caught:
        read_speeches(path)

    assert caught.value.problem is InputFileProblem.UNREADABLE
    assert caught.value.detail == "Permission denied"
