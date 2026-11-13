import pytest

from speech_qa import bootstrap
from speech_qa.application.ingestion import IngestionService
from speech_qa.application.retrieval import RetrievalLimits, Retriever, ensure_queryable
from speech_qa.config import Settings
from speech_qa.domain.errors import IndexIncompatibleError, IndexNotFoundError
from speech_qa.domain.models import RetrievalResult
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex
from speech_qa.text.query_analysis import RuleBasedQueryAnalyzer
from speech_qa.text.sentences import SentenceWindowChunker
from tests.conftest import LONG_SPEECHES, FakeEmbedder

CHUNKER = SentenceWindowChunker(target_chars=1000, max_chars=1500, overlap_sentences=1)


@pytest.fixture
def ingested(memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder) -> QdrantSpeechIndex:
    IngestionService(memory_index, fake_embedder, CHUNKER).ingest(
        read_speeches(LONG_SPEECHES).speeches
    )
    return memory_index


def make_retriever(
    index: QdrantSpeechIndex, embedder: FakeEmbedder, limits: RetrievalLimits | None = None
) -> Retriever:
    manifest = index.read_manifest()
    assert manifest is not None
    analyzer = RuleBasedQueryAnalyzer(manifest.speakers, manifest.parties)
    return Retriever(analyzer, embedder, index, limits)


def speech_ids(result: RetrievalResult) -> list[str]:
    return [evidence.speech.speech_id for evidence in result.evidence]


def test_speaker_filter_keeps_only_the_named_speakers(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    query = "Co si myslí Doležal o vlivku kvality rozhodnutí na kvalitě informací?"

    result = make_retriever(ingested, fake_embedder).retrieve(query)

    assert set(speech_ids(result)) == {"F-001", "F-002", "F-003"}  # not Veronika Doležalová
    assert result.analysis.search_filter.speakers == {"Lukáš Doležal", "Ondřej Doležal"}


def test_query_is_embedded_without_the_recognised_names(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    make_retriever(ingested, fake_embedder).retrieve("Co si myslí Doležal o datech?")

    assert fake_embedder.calls[-1] == ["Co si myslí o datech?"]


def test_evidence_is_ranked_and_keeps_the_best_passages_in_document_order(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    result = make_retriever(ingested, fake_embedder).retrieve(
        "digitální identita a portál pro občany"
    )

    scores = [evidence.score for evidence in result.evidence]
    assert scores == sorted(scores, reverse=True)
    for evidence in result.evidence:
        assert 1 <= len(evidence.passages) <= 2
        indexes = [passage.chunk.index for passage in evidence.passages]
        assert indexes == sorted(indexes)
        assert evidence.score == max(passage.score for passage in evidence.passages)
        assert all(passage.speech == evidence.speech for passage in evidence.passages)


def test_broad_query_keeps_limited_speeches_per_speaker(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    limits = RetrievalLimits(max_speeches=10, max_speeches_per_speaker=1)

    result = make_retriever(ingested, fake_embedder, limits).retrieve(
        "Co zaznělo k dálnicím a datům?"
    )

    speakers = [evidence.speech.speaker for evidence in result.evidence]
    assert len(speakers) == len(set(speakers)) == 8  # 10 speeches by 8 speakers


def test_named_speaker_is_not_limited_by_diversity(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    limits = RetrievalLimits(max_speeches_per_speaker=1)

    result = make_retriever(ingested, fake_embedder, limits).retrieve(
        "Změnil Svoboda názor na dálnice?"
    )

    assert set(speech_ids(result)) == {"F-006", "F-007"}


def test_number_of_speeches_and_passages_is_limited(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    limits = RetrievalLimits(max_speeches=3, max_passages_per_speech=1)

    result = make_retriever(ingested, fake_embedder, limits).retrieve("rozhodování a informace")

    assert len(result.evidence) == 3
    assert all(len(evidence.passages) == 1 for evidence in result.evidence)


def test_filter_without_matches_gives_an_empty_result(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    result = make_retriever(ingested, fake_embedder).retrieve("Co se říkalo v roce 2030?")

    assert result.evidence == ()


def test_year_and_party_filters_are_applied(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    retriever = make_retriever(ingested, fake_embedder)

    assert speech_ids(retriever.retrieve("Co se v roce 2018 říkalo o cenách energií?")) == ["F-010"]
    assert set(speech_ids(retriever.retrieve("Co navrhovala VIZE?"))) == {"F-005", "F-008"}


@pytest.mark.parametrize(
    "limits",
    [
        {"candidate_chunks": 0},
        {"max_speeches": 0},
        {"max_speeches_per_speaker": 0},
        {"max_passages_per_speech": 0},
    ],
)
def test_limits_must_be_positive(limits: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        RetrievalLimits(**limits)


def test_index_built_with_another_embedding_model_cannot_be_queried(
    ingested: QdrantSpeechIndex,
) -> None:
    manifest = ingested.read_manifest()
    assert manifest is not None

    ensure_queryable(manifest, "fake-embedder")
    with pytest.raises(IndexIncompatibleError, match="bge-m3"):
        ensure_queryable(manifest, "bge-m3")


def test_bootstrap_requires_an_ingested_index(memory_index: QdrantSpeechIndex) -> None:
    with pytest.raises(IndexNotFoundError):
        bootstrap.create_retriever(Settings(), memory_index)


def test_bootstrap_checks_the_embedding_model(ingested: QdrantSpeechIndex) -> None:
    with pytest.raises(IndexIncompatibleError):
        bootstrap.create_retriever(Settings(), ingested)  # index was built by the fake embedder


def test_bootstrap_creates_a_retriever_for_a_compatible_index(
    memory_index: QdrantSpeechIndex,
) -> None:
    embedder = FakeEmbedder("bge-m3")
    IngestionService(memory_index, embedder, CHUNKER).ingest(read_speeches(LONG_SPEECHES).speeches)

    assert isinstance(bootstrap.create_retriever(Settings(), memory_index), Retriever)
