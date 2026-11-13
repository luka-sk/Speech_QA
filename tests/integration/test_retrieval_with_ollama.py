"""Retrieval against the real Ollama with bge-m3 (run with: pytest -m integration --no-cov)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from speech_qa import bootstrap
from speech_qa.application.ingestion import IngestionService
from speech_qa.application.retrieval import Retriever
from speech_qa.config import Settings
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex
from tests.conftest import LONG_SPEECHES

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def retriever(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Retriever]:
    settings = Settings()
    path: Path = tmp_path_factory.mktemp("retrieval") / "index"
    with QdrantSpeechIndex.open_local(path, settings.index.collection, create=True) as index:
        service = IngestionService(
            index, bootstrap.create_embedder(settings), bootstrap.create_chunker(settings)
        )
        service.ingest(read_speeches(LONG_SPEECHES).speeches)
        yield bootstrap.create_retriever(settings, index)


def speech_ids(retriever: Retriever, query: str) -> list[str]:
    return [evidence.speech.speech_id for evidence in retriever.retrieve(query).evidence]


def test_assignment_query_about_dolezal(retriever: Retriever) -> None:
    found = speech_ids(
        retriever, "Co si myslí Doležal o vlivku kvality rozhodnutí na kvalitě informací?"
    )

    assert found[0] in {"F-001", "F-003"}
    assert set(found) <= {"F-001", "F-002", "F-003"}


def test_assignment_query_about_decision_quality(retriever: Retriever) -> None:
    found = speech_ids(retriever, "Co lidé diskutovali ke kvalitě rozhodování?")

    assert len({"F-001", "F-003", "F-004"} & set(found[:3])) >= 2


def test_change_of_opinion_of_a_named_speaker(retriever: Retriever) -> None:
    assert set(speech_ids(retriever, "Změnil Svoboda názor na výstavbu dálnic?")) == {
        "F-006",
        "F-007",
    }


def test_party_and_topic(retriever: Retriever) -> None:
    assert speech_ids(retriever, "Co navrhovala strana VIZE v dopravě?")[0] == "F-005"


def test_year_and_topic(retriever: Retriever) -> None:
    assert speech_ids(retriever, "Co se v roce 2018 říkalo o cenách energií?") == ["F-010"]
