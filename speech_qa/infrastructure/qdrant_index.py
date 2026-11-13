"""Speech index stored in Qdrant.

The PoC runs Qdrant in local mode: the index is a directory on disk and only
one process can open it at a time. The same adapter works with a Qdrant server
(``QdrantClient(url=...)``), where payload indexes make filtering fast; in
local mode they have no effect, so they are created only for a server.

Each chunk is one point. Its id is derived from (speech id, chunk index), so
re-indexing a speech overwrites its points instead of duplicating them. The
manifest is a JSON file next to the Qdrant data.
"""

import datetime as dt
import json
import os
import tempfile
import uuid
from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import Any, Self

from qdrant_client import QdrantClient, models

from speech_qa.domain.errors import IndexIncompatibleError, IndexLockedError, IndexNotFoundError
from speech_qa.domain.models import (
    Chunk,
    ChunkingParameters,
    IndexedChunk,
    IndexedSpeech,
    IndexManifest,
    RetrievedChunk,
    SearchFilter,
    SpeechInfo,
    StoredChunk,
    Vector,
)
from speech_qa.domain.ports import SpeechIndex

MANIFEST_FILENAME = "speech_qa_manifest.json"

_POINT_ID_NAMESPACE = uuid.UUID("8f3c1f0e-6d2b-4b8e-9a51-2c7d0e4b9f11")
_SCROLL_PAGE = 1024
_LOCKED_MARKER = "already accessed by another instance"


def point_id(speech_id: str, chunk_index: int) -> str:
    """Deterministic Qdrant point id of a chunk."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{speech_id}\x00{chunk_index}"))


class QdrantSpeechIndex(SpeechIndex):
    """:class:`SpeechIndex` backed by a Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        manifest_path: Path | None,
        *,
        create_payload_indexes: bool = False,
    ) -> None:
        """Wrap an existing Qdrant client.

        Args:
            client: Connected Qdrant client (local, in-memory or server).
            collection: Name of the collection with speech chunks.
            manifest_path: Where the manifest is stored; ``None`` keeps no manifest.
            create_payload_indexes: Create payload indexes for filtered fields
                (useful only with a Qdrant server).
        """
        self._client = client
        self._collection = collection
        self._manifest_path = manifest_path
        self._create_payload_indexes = create_payload_indexes

    @classmethod
    def open_local(cls, path: Path, collection: str, *, create: bool) -> Self:
        """Open (or create) a local index directory.

        Args:
            path: Index directory.
            collection: Name of the collection with speech chunks.
            create: Create the directory if it does not exist.

        Raises:
            IndexNotFoundError: If the directory is missing and ``create`` is false.
            IndexLockedError: If another process has the index open.
        """
        if not path.is_dir():
            if not create:
                raise IndexNotFoundError(path)
            path.mkdir(parents=True)
        try:
            client = QdrantClient(path=str(path))
        except RuntimeError as exc:
            if _LOCKED_MARKER in str(exc):
                raise IndexLockedError(path) from exc
            raise
        return cls(client, collection, path / MANIFEST_FILENAME)

    # ----------------------------------------------------------------- manifest
    def read_manifest(self) -> IndexManifest | None:
        """Return the stored manifest, or ``None`` if there is none."""
        if self._manifest_path is None or not self._manifest_path.is_file():
            return None
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return _manifest_from_json(data)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            msg = f"the manifest {self._manifest_path} cannot be read ({exc})"
            raise IndexIncompatibleError(msg) from exc

    def write_manifest(self, manifest: IndexManifest) -> None:
        """Store the manifest atomically (write a temporary file, then rename)."""
        if self._manifest_path is None:
            return
        content = json.dumps(_manifest_to_json(manifest), ensure_ascii=False, indent=2)
        directory = self._manifest_path.parent
        fd, temporary = tempfile.mkstemp(dir=directory, prefix=".manifest-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            Path(temporary).replace(self._manifest_path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ content
    def indexed_speeches(self) -> dict[str, IndexedSpeech]:
        """Scan all points and summarise them per speech."""
        if not self._client.collection_exists(self._collection):
            return {}
        speeches: dict[str, IndexedSpeech] = {}
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=["speech_id", "content_hash", "speaker", "party"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                speeches.setdefault(
                    str(payload["speech_id"]),
                    IndexedSpeech(
                        content_hash=str(payload["content_hash"]),
                        speaker=str(payload["speaker"]),
                        party=_optional_str(payload.get("party")),
                    ),
                )
            if offset is None:
                return speeches

    def chunk_count(self) -> int:
        """Return the exact number of stored chunks."""
        if not self._client.collection_exists(self._collection):
            return 0
        return self._client.count(self._collection, exact=True).count

    def iter_chunks(self) -> Iterator[StoredChunk]:
        """Scroll through all stored chunks."""
        if not self._client.collection_exists(self._collection):
            return
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                self._collection, limit=_SCROLL_PAGE, offset=offset, with_vectors=False
            )
            for point in points:
                retrieved = _to_retrieved_chunk(point.payload or {}, 0.0)
                yield StoredChunk(chunk=retrieved.chunk, speech=retrieved.speech)
            if offset is None:
                return

    def replace_speeches(self, speech_ids: Collection[str], chunks: Sequence[IndexedChunk]) -> None:
        """Delete the speeches' old chunks and store the new ones."""
        unknown = {chunk.chunk.speech_id for chunk in chunks} - set(speech_ids)
        if unknown:
            msg = f"chunks of speeches not being replaced: {sorted(unknown)}"
            raise ValueError(msg)
        if chunks:
            self._ensure_collection(len(chunks[0].vector))
        elif not self._client.collection_exists(self._collection):
            return
        if speech_ids:
            self._client.delete(
                self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="speech_id", match=models.MatchAny(any=list(speech_ids))
                            )
                        ]
                    )
                ),
                wait=True,
            )
        if chunks:
            self._client.upsert(
                self._collection, points=[_to_point(chunk) for chunk in chunks], wait=True
            )

    def search(
        self, vector: Vector, search_filter: SearchFilter, limit: int
    ) -> list[RetrievedChunk]:
        """Cosine-similarity search restricted by the filter."""
        if not self._client.collection_exists(self._collection):
            return []
        response = self._client.query_points(
            self._collection,
            query=list(vector),
            query_filter=_to_qdrant_filter(search_filter),
            limit=limit,
            with_payload=True,
        )
        return [_to_retrieved_chunk(point.payload or {}, point.score) for point in response.points]

    def clear(self) -> None:
        """Delete the collection and the manifest."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        if self._manifest_path is not None:
            self._manifest_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Close the client and release the lock of a local index."""
        self._client.close()

    # ---------------------------------------------------------------- internals
    def _ensure_collection(self, dimension: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        if self._create_payload_indexes:
            keyword = models.PayloadSchemaType.KEYWORD
            for field_name, schema in (
                ("speech_id", keyword),
                ("speaker", keyword),
                ("party", keyword),
                ("date_ordinal", models.PayloadSchemaType.INTEGER),
            ):
                self._client.create_payload_index(
                    self._collection, field_name=field_name, field_schema=schema
                )


def _to_point(chunk: IndexedChunk) -> models.PointStruct:
    speech = chunk.speech
    return models.PointStruct(
        id=point_id(speech.speech_id, chunk.chunk.index),
        vector=list(chunk.vector),
        payload={
            "speech_id": speech.speech_id,
            "chunk_index": chunk.chunk.index,
            "text": chunk.chunk.text,
            "speaker": speech.speaker,
            "party": speech.party,
            "date": speech.date.isoformat(),
            # Integer copy of the date: range filters work the same in every Qdrant mode.
            "date_ordinal": speech.date.toordinal(),
            "content_hash": chunk.content_hash,
        },
    )


def _to_qdrant_filter(search_filter: SearchFilter) -> models.Filter | None:
    conditions: list[models.Condition] = []
    if search_filter.speakers:
        conditions.append(
            models.FieldCondition(
                key="speaker", match=models.MatchAny(any=sorted(search_filter.speakers))
            )
        )
    if search_filter.parties:
        conditions.append(
            models.FieldCondition(
                key="party", match=models.MatchAny(any=sorted(search_filter.parties))
            )
        )
    if search_filter.date_from or search_filter.date_to:
        conditions.append(
            models.FieldCondition(
                key="date_ordinal",
                range=models.Range(
                    gte=search_filter.date_from.toordinal() if search_filter.date_from else None,
                    lte=search_filter.date_to.toordinal() if search_filter.date_to else None,
                ),
            )
        )
    return models.Filter(must=conditions) if conditions else None


def _to_retrieved_chunk(payload: dict[str, Any], score: float) -> RetrievedChunk:
    speech_id = str(payload["speech_id"])
    return RetrievedChunk(
        chunk=Chunk(
            speech_id=speech_id, index=int(payload["chunk_index"]), text=str(payload["text"])
        ),
        speech=SpeechInfo(
            speech_id=speech_id,
            speaker=str(payload["speaker"]),
            party=_optional_str(payload.get("party")),
            date=dt.date.fromisoformat(str(payload["date"])),
        ),
        score=float(score),
    )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _manifest_to_json(manifest: IndexManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "embedding_model": manifest.embedding_model,
        "embedding_dimension": manifest.embedding_dimension,
        "chunking": {
            "target_chars": manifest.chunking.target_chars,
            "max_chars": manifest.chunking.max_chars,
            "overlap_sentences": manifest.chunking.overlap_sentences,
        },
        "speakers": list(manifest.speakers),
        "parties": list(manifest.parties),
        "speech_count": manifest.speech_count,
        "chunk_count": manifest.chunk_count,
        "updated_at": manifest.updated_at.isoformat(),
    }


def _manifest_from_json(data: dict[str, Any]) -> IndexManifest:
    chunking = data["chunking"]
    return IndexManifest(
        schema_version=int(data["schema_version"]),
        embedding_model=str(data["embedding_model"]),
        embedding_dimension=int(data["embedding_dimension"]),
        chunking=ChunkingParameters(
            target_chars=int(chunking["target_chars"]),
            max_chars=int(chunking["max_chars"]),
            overlap_sentences=int(chunking["overlap_sentences"]),
        ),
        speakers=tuple(str(name) for name in data["speakers"]),
        parties=tuple(str(name) for name in data["parties"]),
        speech_count=int(data["speech_count"]),
        chunk_count=int(data["chunk_count"]),
        updated_at=dt.datetime.fromisoformat(str(data["updated_at"])),
    )
