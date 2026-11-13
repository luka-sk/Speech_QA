import dataclasses
import datetime as dt
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from speech_qa.domain.errors import IndexIncompatibleError, IndexLockedError, IndexNotFoundError
from speech_qa.domain.models import (
    Chunk,
    ChunkingParameters,
    IndexedChunk,
    IndexedSpeech,
    IndexManifest,
    SearchFilter,
    SpeechInfo,
)
from speech_qa.infrastructure.qdrant_index import MANIFEST_FILENAME, QdrantSpeechIndex, point_id
from tests.conftest import FakeEmbedder

EMBEDDER = FakeEmbedder()


def speech_info(
    speech_id: str, speaker: str = "Lukáš Doležal", party: str | None = "R7", year: int = 2021
) -> SpeechInfo:
    return SpeechInfo(speech_id=speech_id, speaker=speaker, party=party, date=dt.date(year, 6, 1))


def indexed(info: SpeechInfo, index: int, text: str, digest: str = "hash") -> IndexedChunk:
    return IndexedChunk(
        chunk=Chunk(speech_id=info.speech_id, index=index, text=text),
        speech=info,
        content_hash=digest,
        vector=EMBEDDER.vector(text),
    )


def manifest(**overrides: object) -> IndexManifest:
    values: dict[str, object] = {
        "schema_version": 1,
        "embedding_model": "bge-m3",
        "embedding_dimension": 1024,
        "chunking": ChunkingParameters(1000, 1500, 1),
        "speakers": ("Lukáš Doležal", "Ondřej Doležal"),
        "parties": ("K12", "R7"),
        "speech_count": 2,
        "chunk_count": 5,
        "updated_at": dt.datetime(2026, 11, 12, 10, 30, tzinfo=dt.UTC),
    }
    values.update(overrides)
    return IndexManifest(**values)  # type: ignore[arg-type]


@pytest.fixture
def filled_index(memory_index: QdrantSpeechIndex) -> QdrantSpeechIndex:
    lukas = speech_info("S1", "Lukáš Doležal", "R7", 2021)
    ondrej = speech_info("S2", "Ondřej Doležal", "K12", 2023)
    david = speech_info("S3", "David Müller", None, 2026)
    memory_index.replace_speeches(
        ["S1", "S2", "S3"],
        [
            indexed(lukas, 0, "kvalita rozhodnutí závisí na kvalitě informací", "h1"),
            indexed(lukas, 1, "datový audit ministerstev", "h1"),
            indexed(ondrej, 0, "víc informací neznamená lepší rozhodnutí", "h2"),
            indexed(david, 0, "kybernetická bezpečnost nemocnic", "h3"),
        ],
    )
    return memory_index


def search_ids(index: QdrantSpeechIndex, query: str, search_filter: SearchFilter) -> list[str]:
    hits = index.search(EMBEDDER.vector(query), search_filter, limit=10)
    return [f"{hit.chunk.speech_id}#{hit.chunk.index}" for hit in hits]


def test_search_returns_chunks_with_speech_metadata(filled_index: QdrantSpeechIndex) -> None:
    hits = filled_index.search(EMBEDDER.vector("kybernetická bezpečnost"), SearchFilter(), limit=2)

    best = hits[0]
    assert best.chunk == Chunk(speech_id="S3", index=0, text="kybernetická bezpečnost nemocnic")
    assert best.speech == speech_info("S3", "David Müller", None, 2026)
    assert best.score > hits[1].score


def test_search_respects_the_limit(filled_index: QdrantSpeechIndex) -> None:
    assert len(filled_index.search(EMBEDDER.vector("informací"), SearchFilter(), limit=3)) == 3


@pytest.mark.parametrize(
    ("search_filter", "expected"),
    [
        (SearchFilter(speakers=frozenset({"Ondřej Doležal"})), {"S2#0"}),
        (
            SearchFilter(speakers=frozenset({"Lukáš Doležal", "Ondřej Doležal"})),
            {"S1#0", "S1#1", "S2#0"},
        ),
        (SearchFilter(parties=frozenset({"R7"})), {"S1#0", "S1#1"}),
        (SearchFilter(date_from=dt.date(2023, 1, 1)), {"S2#0", "S3#0"}),
        (SearchFilter(date_to=dt.date(2021, 12, 31)), {"S1#0", "S1#1"}),
        (SearchFilter(date_from=dt.date(2022, 1, 1), date_to=dt.date(2025, 12, 31)), {"S2#0"}),
        (SearchFilter(parties=frozenset({"R7"}), date_from=dt.date(2022, 1, 1)), set()),
        (SearchFilter(speakers=frozenset({"Neexistující"})), set()),
    ],
)
def test_search_filters(
    filled_index: QdrantSpeechIndex, search_filter: SearchFilter, expected: set[str]
) -> None:
    assert set(search_ids(filled_index, "informací rozhodnutí", search_filter)) == expected


def test_replacing_a_speech_removes_its_old_chunks(filled_index: QdrantSpeechIndex) -> None:
    lukas = speech_info("S1")

    filled_index.replace_speeches(["S1"], [indexed(lukas, 0, "nový jediný úsek", "h1-new")])

    assert search_ids(
        filled_index, "úsek", SearchFilter(speakers=frozenset({"Lukáš Doležal"}))
    ) == ["S1#0"]
    assert filled_index.chunk_count() == 3
    assert filled_index.indexed_speeches()["S1"].content_hash == "h1-new"


def test_reindexing_the_same_chunks_does_not_duplicate_them(
    filled_index: QdrantSpeechIndex,
) -> None:
    lukas = speech_info("S1")
    chunks = [indexed(lukas, 0, "kvalita", "h1"), indexed(lukas, 1, "audit", "h1")]

    filled_index.replace_speeches(["S1"], chunks)
    filled_index.replace_speeches(["S1"], chunks)

    assert filled_index.chunk_count() == 4


def test_indexed_speeches_summarise_every_speech(filled_index: QdrantSpeechIndex) -> None:
    assert filled_index.indexed_speeches() == {
        "S1": IndexedSpeech("h1", "Lukáš Doležal", "R7"),
        "S2": IndexedSpeech("h2", "Ondřej Doležal", "K12"),
        "S3": IndexedSpeech("h3", "David Müller", None),
    }


def test_indexed_speeches_page_through_large_collections(
    memory_index: QdrantSpeechIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("speech_qa.infrastructure.qdrant_index._SCROLL_PAGE", 2)
    infos = [speech_info(f"S{n}") for n in range(5)]
    memory_index.replace_speeches(
        [i.speech_id for i in infos], [indexed(i, 0, "text") for i in infos]
    )

    assert sorted(memory_index.indexed_speeches()) == ["S0", "S1", "S2", "S3", "S4"]


def test_empty_index_has_nothing(memory_index: QdrantSpeechIndex) -> None:
    assert memory_index.indexed_speeches() == {}
    assert memory_index.chunk_count() == 0
    assert memory_index.search(EMBEDDER.vector("cokoli"), SearchFilter(), limit=5) == []
    assert memory_index.read_manifest() is None


def test_replacing_with_no_chunks_on_an_empty_index_is_a_no_op(
    memory_index: QdrantSpeechIndex,
) -> None:
    memory_index.replace_speeches(["S1"], [])

    assert memory_index.chunk_count() == 0


def test_replacing_with_no_chunks_deletes_the_speech(filled_index: QdrantSpeechIndex) -> None:
    filled_index.replace_speeches(["S3"], [])

    assert "S3" not in filled_index.indexed_speeches()


def test_chunks_of_unlisted_speeches_are_rejected(memory_index: QdrantSpeechIndex) -> None:
    with pytest.raises(ValueError, match="S2"):
        memory_index.replace_speeches(["S1"], [indexed(speech_info("S2"), 0, "text")])


def test_manifest_round_trip(memory_index: QdrantSpeechIndex) -> None:
    stored = manifest()

    memory_index.write_manifest(stored)

    assert memory_index.read_manifest() == stored


def test_manifest_is_replaced_atomically(memory_index: QdrantSpeechIndex, tmp_path: Path) -> None:
    memory_index.write_manifest(manifest(speech_count=1))
    memory_index.write_manifest(manifest(speech_count=2))

    assert memory_index.read_manifest() == manifest(speech_count=2)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.json"]


def test_unreadable_manifest_makes_the_index_incompatible(
    memory_index: QdrantSpeechIndex, tmp_path: Path
) -> None:
    (tmp_path / "manifest.json").write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(IndexIncompatibleError, match="cannot be read"):
        memory_index.read_manifest()


def test_clear_removes_chunks_and_manifest(filled_index: QdrantSpeechIndex) -> None:
    filled_index.write_manifest(manifest())

    filled_index.clear()

    assert filled_index.chunk_count() == 0
    assert filled_index.read_manifest() is None


def test_index_without_manifest_path_keeps_no_manifest() -> None:
    with QdrantSpeechIndex(QdrantClient(":memory:"), "other", None) as index:
        index.write_manifest(manifest())
        index.clear()

        assert index.read_manifest() is None


def test_payload_indexes_are_created_on_request() -> None:
    client = QdrantClient(":memory:")
    with QdrantSpeechIndex(client, "chunks", None, create_payload_indexes=True) as index:
        # Local Qdrant accepts payload indexes but warns that they have no effect.
        with pytest.warns(UserWarning, match="Payload indexes have no effect"):
            index.replace_speeches(["S1"], [indexed(speech_info("S1"), 0, "text")])

        assert index.chunk_count() == 1


def test_point_ids_are_deterministic_and_distinct() -> None:
    assert point_id("S1", 0) == point_id("S1", 0)
    assert len({point_id("S1", 0), point_id("S1", 1), point_id("S10", 0)}) == 3


def test_local_index_persists_between_openings(tmp_path: Path) -> None:
    path = tmp_path / "index"
    with QdrantSpeechIndex.open_local(path, "chunks", create=True) as index:
        index.replace_speeches(["S1"], [indexed(speech_info("S1"), 0, "kvalita informací")])
        index.write_manifest(manifest())

    with QdrantSpeechIndex.open_local(path, "chunks", create=False) as reopened:
        assert reopened.chunk_count() == 1
        assert reopened.read_manifest() == manifest()
    assert (path / MANIFEST_FILENAME).is_file()


def test_missing_local_index_is_not_created_for_queries(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError):
        QdrantSpeechIndex.open_local(tmp_path / "missing", "chunks", create=False)

    assert not (tmp_path / "missing").exists()


def test_local_index_used_by_another_client_is_locked(tmp_path: Path) -> None:
    path = tmp_path / "index"
    with (
        QdrantSpeechIndex.open_local(path, "chunks", create=True),
        pytest.raises(IndexLockedError),
    ):
        QdrantSpeechIndex.open_local(path, "chunks", create=False)


def test_other_errors_when_opening_are_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        msg = "disk on fire"
        raise RuntimeError(msg)

    monkeypatch.setattr("speech_qa.infrastructure.qdrant_index.QdrantClient", fail)

    with pytest.raises(RuntimeError, match="disk on fire"):
        QdrantSpeechIndex.open_local(tmp_path, "chunks", create=False)


def test_manifest_dataclass_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest().speech_count = 3  # type: ignore[misc]


def test_iter_chunks_lists_every_stored_chunk(filled_index: QdrantSpeechIndex) -> None:
    chunks = sorted(
        filled_index.iter_chunks(), key=lambda item: (item.chunk.speech_id, item.chunk.index)
    )

    assert [(c.chunk.speech_id, c.chunk.index) for c in chunks] == [
        ("S1", 0),
        ("S1", 1),
        ("S2", 0),
        ("S3", 0),
    ]
    assert chunks[0].speech == speech_info("S1", "Lukáš Doležal", "R7", 2021)


def test_iter_chunks_of_an_empty_index(memory_index: QdrantSpeechIndex) -> None:
    assert list(memory_index.iter_chunks()) == []
