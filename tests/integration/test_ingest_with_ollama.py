"""Ingest against the real Ollama with bge-m3 (run with: pytest -m integration --no-cov)."""

from pathlib import Path

import pytest

from speech_qa import bootstrap
from speech_qa.application.ingestion import IngestionService
from speech_qa.config import OllamaSettings, Settings
from speech_qa.domain.errors import ModelNotAvailableError
from speech_qa.domain.models import SearchFilter
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex
from tests.conftest import LONG_SPEECHES

pytestmark = pytest.mark.integration

DOLEZALS = frozenset({"Lukáš Doležal", "Ondřej Doležal"})


def test_fixtures_are_ingested_and_searchable_with_real_embeddings(tmp_path: Path) -> None:
    settings = Settings()
    embedder = bootstrap.create_embedder(settings)
    speeches = read_speeches(LONG_SPEECHES).speeches

    with QdrantSpeechIndex.open_local(tmp_path / "index", "chunks", create=True) as index:
        service = IngestionService(index, embedder, bootstrap.create_chunker(settings))
        first = service.ingest(speeches)
        second = service.ingest(speeches)

        assert (first.new, first.chunks_written) == (10, index.chunk_count())
        assert first.manifest.embedding_dimension == 1024
        assert (second.unchanged, second.chunks_written) == (10, 0)

        query = "Co si myslí Doležal o vlivku kvality rozhodnutí na kvalitě informací?"
        hits = index.search(embedder.embed_query(query), SearchFilter(speakers=DOLEZALS), limit=5)
        assert hits[0].chunk.speech_id in {"F-001", "F-003"}

        query = "Co lidé diskutovali ke kvalitě rozhodování?"
        hits = index.search(embedder.embed_query(query), SearchFilter(), limit=3)
        assert hits[0].chunk.speech_id in {"F-001", "F-003", "F-004"}


def test_missing_embedding_model_is_reported() -> None:
    settings = Settings(ollama=OllamaSettings(embedding_model="speech-qa-missing-model"))

    with pytest.raises(ModelNotAvailableError) as caught:
        bootstrap.create_embedder(settings).embed_query("dotaz")

    assert caught.value.model == "speech-qa-missing-model"
