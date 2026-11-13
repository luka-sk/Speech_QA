import dataclasses
import datetime as dt

import pytest

from speech_qa.application.ingestion import SCHEMA_VERSION, IngestionService, content_hash
from speech_qa.domain.errors import IndexIncompatibleError
from speech_qa.domain.models import ChunkingParameters, InputIssue, IssueKind, SearchFilter, Speech
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex
from speech_qa.text.sentences import SentenceWindowChunker
from tests.conftest import LONG_SPEECHES, FakeEmbedder

NOW = dt.datetime(2026, 11, 12, 12, 0, tzinfo=dt.UTC)
CHUNKER = SentenceWindowChunker(target_chars=1000, max_chars=1500, overlap_sentences=1)
SPEECHES = read_speeches(LONG_SPEECHES).speeches


def make_service(
    index: QdrantSpeechIndex,
    embedder: FakeEmbedder,
    chunker: SentenceWindowChunker = CHUNKER,
    batch_size: int = 8,
) -> IngestionService:
    ticks = iter(range(100))
    return IngestionService(
        index,
        embedder,
        chunker,
        batch_size=batch_size,
        clock=lambda: NOW,
        timer=lambda: float(next(ticks)),
    )


def total_chunks(speeches: tuple[Speech, ...] = SPEECHES) -> int:
    return sum(len(CHUNKER.split(speech)) for speech in speeches)


def test_first_ingest_indexes_every_speech(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    issue = InputIssue(line=3, kind=IssueKind.INVALID_JSON)

    report = make_service(memory_index, fake_embedder).ingest(SPEECHES, [issue])

    assert (report.new, report.changed, report.unchanged) == (10, 0, 0)
    assert report.speeches_in_input == 10
    assert report.chunks_written == total_chunks() == memory_index.chunk_count()
    assert report.issues == (issue,)
    assert report.duration_seconds == 1.0
    assert not report.rebuilt


def test_manifest_describes_the_index(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    make_service(memory_index, fake_embedder).ingest(SPEECHES)

    manifest = memory_index.read_manifest()

    assert manifest is not None
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.embedding_model == "fake-embedder"
    assert manifest.embedding_dimension == fake_embedder.dimension
    assert manifest.chunking == ChunkingParameters(1000, 1500, 1)
    assert manifest.speakers == (
        "David Müller",
        "Eva Smith",
        "Lukáš Doležal",
        "Martin Svoboda",
        "Ondřej Doležal",
        "Petra Pokorná",
        "Tomáš Němec",
        "Veronika Doležalová",
    )
    assert manifest.parties == ("ABC", "K12", "LID", "NOVA", "PRO", "R7", "VIZE")
    assert manifest.speech_count == 10
    assert manifest.chunk_count == total_chunks()
    assert manifest.updated_at == NOW


def test_second_ingest_of_the_same_data_changes_nothing(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    service = make_service(memory_index, fake_embedder)
    service.ingest(SPEECHES)
    embedded_before = fake_embedder.embedded_texts

    report = service.ingest(SPEECHES)

    assert (report.new, report.changed, report.unchanged) == (0, 0, 10)
    assert report.chunks_written == 0
    assert fake_embedder.embedded_texts == embedded_before
    assert memory_index.chunk_count() == total_chunks()
    manifest = memory_index.read_manifest()
    assert manifest is not None
    assert manifest.embedding_dimension == fake_embedder.dimension


def test_only_new_and_changed_speeches_are_embedded(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    service = make_service(memory_index, fake_embedder)
    service.ingest(SPEECHES[:8])
    changed = dataclasses.replace(SPEECHES[0], text="Zcela nový text projevu. Druhá věta.")
    embedded_before = fake_embedder.embedded_texts

    report = service.ingest((changed, *SPEECHES[1:]))

    assert (report.new, report.changed, report.unchanged) == (2, 1, 7)
    expected = total_chunks((changed, SPEECHES[8], SPEECHES[9]))
    assert report.chunks_written == expected
    assert fake_embedder.embedded_texts - embedded_before == expected
    assert memory_index.chunk_count() == total_chunks((changed, *SPEECHES[1:]))


def test_changed_metadata_triggers_reindexing(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    service = make_service(memory_index, fake_embedder)
    service.ingest(SPEECHES)
    moved = dataclasses.replace(SPEECHES[0], party="K12")

    report = service.ingest((moved,))

    assert report.changed == 1
    hits = memory_index.search(
        fake_embedder.vector(moved.text), SearchFilter(parties=frozenset({"K12"})), limit=50
    )
    assert "F-001" in {hit.chunk.speech_id for hit in hits}


def test_speeches_not_in_the_input_stay_in_the_index(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    service = make_service(memory_index, fake_embedder)
    service.ingest(SPEECHES)

    report = service.ingest(SPEECHES[:1])

    assert report.manifest.speech_count == 10


def test_rebuild_starts_from_an_empty_index(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    service = make_service(memory_index, fake_embedder)
    service.ingest(SPEECHES)

    report = service.ingest(SPEECHES[:3], rebuild=True)

    assert report.rebuilt
    assert (report.new, report.unchanged) == (3, 0)
    assert report.manifest.speech_count == 3
    assert memory_index.chunk_count() == total_chunks(SPEECHES[:3])


def test_other_embedding_model_requires_a_rebuild(memory_index: QdrantSpeechIndex) -> None:
    make_service(memory_index, FakeEmbedder("model-a")).ingest(SPEECHES[:2])

    with pytest.raises(IndexIncompatibleError, match="model-a"):
        make_service(memory_index, FakeEmbedder("model-b")).ingest(SPEECHES[:2])

    report = make_service(memory_index, FakeEmbedder("model-b")).ingest(SPEECHES[:2], rebuild=True)
    assert report.manifest.embedding_model == "model-b"


def test_other_chunking_requires_a_rebuild(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    make_service(memory_index, fake_embedder).ingest(SPEECHES[:2])
    other = SentenceWindowChunker(target_chars=500, max_chars=800, overlap_sentences=0)

    with pytest.raises(IndexIncompatibleError, match="chunking"):
        make_service(memory_index, fake_embedder, chunker=other).ingest(SPEECHES[:2])


def test_other_schema_version_requires_a_rebuild(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    make_service(memory_index, fake_embedder).ingest(SPEECHES[:1])
    manifest = memory_index.read_manifest()
    assert manifest is not None
    memory_index.write_manifest(dataclasses.replace(manifest, schema_version=SCHEMA_VERSION + 1))

    with pytest.raises(IndexIncompatibleError, match="layout version"):
        make_service(memory_index, fake_embedder).ingest(SPEECHES[:1])


def test_chunks_are_embedded_in_batches(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    make_service(memory_index, fake_embedder, batch_size=5).ingest(SPEECHES)

    assert len(fake_embedder.calls) > 1
    assert fake_embedder.embedded_texts == total_chunks()


def test_unknown_dimension_is_zero_when_nothing_was_embedded(
    memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    report = make_service(memory_index, fake_embedder).ingest(())

    assert report.manifest.embedding_dimension == 0
    assert report.manifest.speech_count == 0


def test_content_hash_covers_metadata_and_text() -> None:
    speech = SPEECHES[0]
    variants = [
        dataclasses.replace(speech, speaker="Jiný Řečník"),
        dataclasses.replace(speech, party=None),
        dataclasses.replace(speech, date=dt.date(2000, 1, 1)),
        dataclasses.replace(speech, text=speech.text + " Dodatek."),
    ]

    assert content_hash(speech) == content_hash(dataclasses.replace(speech))
    assert len({content_hash(speech), *(content_hash(variant) for variant in variants)}) == 5
