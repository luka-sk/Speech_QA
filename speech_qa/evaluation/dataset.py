"""Evaluation question sets stored as CSV files.

Columns: ``id;question;relevant_ids;expected_speakers;type;note``. Files are
written with a semicolon and UTF-8 with BOM, so that Czech Excel opens them
correctly; when reading, a comma delimiter and a missing BOM are accepted too.
Lists (relevant speech ids, expected speakers) are separated by commas inside
a cell. Empty ``relevant_ids`` means the archive has no answer to the question;
empty ``expected_speakers`` means the question names no speaker.
"""

import csv
import io
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from speech_qa.domain.errors import EvaluationDataError, OutputExistsError

COLUMNS = ("id", "question", "relevant_ids", "expected_speakers", "type", "note")
REQUIRED_COLUMNS = ("id", "question")


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """One evaluation question.

    Attributes:
        question_id: Unique id within the set.
        question: The question as a user would ask it.
        relevant_ids: Speeches that answer the question; empty if none does.
        expected_speakers: Speakers the question refers to; empty if none.
        kind: Free category (e.g. ``aggregate``, ``speaker``, ``synthetic``).
        note: Free note.
    """

    question_id: str
    question: str
    relevant_ids: tuple[str, ...] = ()
    expected_speakers: tuple[str, ...] = ()
    kind: str = ""
    note: str = ""

    @property
    def answerable(self) -> bool:
        """Whether the archive contains an answer."""
        return bool(self.relevant_ids)


@dataclass(frozen=True, slots=True)
class EvalSet:
    """A named set of evaluation questions read from a file."""

    name: str
    path: Path
    questions: tuple[EvalQuestion, ...]


def read_eval_set(path: Path, name: str) -> EvalSet:
    """Read and validate an evaluation set.

    Raises:
        EvaluationDataError: If the file is missing, unreadable or has invalid rows.
    """
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise EvaluationDataError(path, ["the file does not exist"]) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise EvaluationDataError(path, [f"the file cannot be read: {exc}"]) from exc

    first_line = content.split("\n", 1)[0]
    delimiter = ";" if first_line.count(";") >= first_line.count(",") else ","
    reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
    header = [column.strip() for column in reader.fieldnames or []]
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise EvaluationDataError(path, [f"missing columns: {', '.join(missing)}"])

    problems: list[str] = []
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for row in reader:
        line = reader.line_num
        values = {key.strip(): (value or "").strip() for key, value in row.items() if key}
        question_id, question = values.get("id", ""), values.get("question", "")
        if not question_id and not question:
            continue  # empty row, e.g. at the end of a spreadsheet
        if not question_id or not question:
            problems.append(f"line {line}: id and question are required")
            continue
        if question_id in seen:
            problems.append(f"line {line}: duplicate id {question_id}")
            continue
        seen.add(question_id)
        questions.append(
            EvalQuestion(
                question_id=question_id,
                question=question,
                relevant_ids=_split(values.get("relevant_ids", "")),
                expected_speakers=_split(values.get("expected_speakers", "")),
                kind=values.get("type", ""),
                note=values.get("note", ""),
            )
        )
    if problems:
        raise EvaluationDataError(path, problems)
    if not questions:
        raise EvaluationDataError(path, ["the file contains no questions"])
    return EvalSet(name=name, path=path, questions=tuple(questions))


def write_eval_set(path: Path, questions: Iterable[EvalQuestion], *, force: bool = False) -> None:
    """Write questions as a CSV file for Excel (semicolon, UTF-8 with BOM).

    Raises:
        OutputExistsError: If the file exists and ``force`` is false.
    """
    if path.exists() and not force:
        raise OutputExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(COLUMNS)
        for item in questions:
            writer.writerow(
                [
                    item.question_id,
                    item.question,
                    ",".join(item.relevant_ids),
                    ",".join(item.expected_speakers),
                    item.kind,
                    item.note,
                ]
            )


def _split(cell: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in cell.split(",") if part.strip())
