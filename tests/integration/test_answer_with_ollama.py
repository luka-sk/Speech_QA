"""End-to-end answers with the real Ollama (bge-m3 + gemma4:12b).

Run with: pytest -m integration --no-cov. Each answer takes tens of seconds.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from speech_qa import bootstrap
from speech_qa.application.ingestion import IngestionService
from speech_qa.application.qa import SpeechQA
from speech_qa.config import Settings
from speech_qa.diagnostics import run_checks
from speech_qa.domain.models import Answer
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex
from tests.conftest import LONG_SPEECHES

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qa(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SpeechQA]:
    settings = Settings()
    path: Path = tmp_path_factory.mktemp("answers") / "index"
    with QdrantSpeechIndex.open_local(path, settings.index.collection, create=True) as index:
        IngestionService(
            index, bootstrap.create_embedder(settings), bootstrap.create_chunker(settings)
        ).ingest(read_speeches(LONG_SPEECHES).speeches)
        yield bootstrap.create_qa(settings, index)


def cited(answer: Answer) -> set[str]:
    return {citation.evidence.speech.speech_id for citation in answer.citations}


def test_assignment_query_about_dolezal_cites_only_dolezals(qa: SpeechQA) -> None:
    answer = qa.ask("Co si myslí Doležal o vlivku kvality rozhodnutí na kvalitě informací?")

    assert answer.text
    assert cited(answer)
    assert cited(answer) <= {"F-001", "F-002", "F-003"}
    assert answer.unknown_citations == ()
    assert not answer.context_may_be_truncated


def test_assignment_query_about_decision_quality(qa: SpeechQA) -> None:
    answer = qa.ask("Co lidé diskutovali ke kvalitě rozhodování?")

    assert cited(answer) & {"F-001", "F-003", "F-004"}
    assert answer.unknown_citations == ()


def test_question_outside_the_archive_is_answered_without_sources(qa: SpeechQA) -> None:
    answer = qa.ask("Co si politici myslí o kryptoměnách?")

    assert answer.citations == ()


def test_check_finds_the_models() -> None:
    report = run_checks(Settings(), None)

    assert report.models_ok
