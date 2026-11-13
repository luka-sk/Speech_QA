from pathlib import Path

import pytest

from speech_qa.domain.errors import EvaluationDataError, OutputExistsError
from speech_qa.evaluation.dataset import EvalQuestion, read_eval_set, write_eval_set
from speech_qa.evaluation.loading import load_eval_sets
from tests.conftest import FIXTURES

TEMPLATE = Path(__file__).resolve().parents[2] / "eval" / "manual.example.csv"


def test_reads_the_fixture_set() -> None:
    eval_set = read_eval_set(FIXTURES / "eval_manual.csv", "manual")

    assert eval_set.name == "manual"
    assert len(eval_set.questions) == 7
    first = eval_set.questions[0]
    assert first.question_id == "q1"
    assert first.relevant_ids == ("F-001", "F-003")
    assert first.expected_speakers == ("Lukáš Doležal", "Ondřej Doležal")
    assert first.kind == "speaker"
    assert first.answerable
    assert not eval_set.questions[-1].answerable  # q7: no relevant speeches


def test_reads_the_template_shipped_for_users() -> None:
    assert [q.question_id for q in read_eval_set(TEMPLATE, "manual").questions] == [
        "man-01",
        "man-02",
    ]


def test_comma_delimiter_without_bom_and_optional_columns(tmp_path: Path) -> None:
    path = tmp_path / "set.csv"
    path.write_text(
        'id,question,relevant_ids\nq1,"Otázka, s čárkou?","A, B"\n,,\n', encoding="utf-8"
    )

    questions = read_eval_set(path, "x").questions

    assert questions == (EvalQuestion("q1", "Otázka, s čárkou?", ("A", "B")),)


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "out" / "set.csv"
    questions = [
        EvalQuestion(
            "s1", "Co říká Doležal; a proč?", ("F-001",), ("Lukáš Doležal",), "synthetic", "n"
        ),
        EvalQuestion("s2", "Téma?"),
    ]

    write_eval_set(path, questions)

    assert path.read_bytes().startswith(b"\xef\xbb\xbfid;question;")
    assert list(read_eval_set(path, "x").questions) == questions


def test_existing_file_is_not_overwritten_without_force(tmp_path: Path) -> None:
    path = tmp_path / "set.csv"
    path.write_text("keep", encoding="utf-8")

    with pytest.raises(OutputExistsError):
        write_eval_set(path, [EvalQuestion("a", "b")])
    write_eval_set(path, [EvalQuestion("a", "b")], force=True)

    assert read_eval_set(path, "x").questions == (EvalQuestion("a", "b"),)


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ("question;note\nOtázka;x\n", "missing columns: id"),
        ("id;question\nq1;\n", "line 2: id and question are required"),
        ("id;question\nq1;A\nq1;B\n", "line 3: duplicate id q1"),
        ("id;question\n", "no questions"),
    ],
)
def test_invalid_sets_are_reported(tmp_path: Path, content: str, problem: str) -> None:
    path = tmp_path / "set.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(EvaluationDataError) as caught:
        read_eval_set(path, "x")

    assert any(problem in item for item in caught.value.problems)


def test_missing_and_unreadable_files_are_reported(tmp_path: Path) -> None:
    (tmp_path / "binary.csv").write_bytes(b"\xff\xfe\x00")

    with pytest.raises(EvaluationDataError, match="does not exist"):
        read_eval_set(tmp_path / "missing.csv", "x")
    with pytest.raises(EvaluationDataError, match="cannot be read"):
        read_eval_set(tmp_path / "binary.csv", "x")


def test_load_eval_sets_skips_missing_files(tmp_path: Path) -> None:
    sets = load_eval_sets(FIXTURES / "eval_manual.csv", tmp_path / "missing.csv")

    assert [eval_set.name for eval_set in sets] == ["manual"]


def test_load_eval_sets_needs_at_least_one_file(tmp_path: Path) -> None:
    with pytest.raises(EvaluationDataError, match="no evaluation set found"):
        load_eval_sets(tmp_path / "a.csv", tmp_path / "b.csv")
