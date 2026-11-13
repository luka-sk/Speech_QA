"""Use case: ingest speeches into the index.

Ingestion is incremental. Every speech has a content hash; speeches whose hash
is already in the index are skipped, new and changed speeches are chunked,
embedded and stored. Nothing is deleted except with ``rebuild``, which starts
from an empty index. The manifest is written only after a successful run, so
an interrupted ingest can simply be started again.
"""

import datetime as dt
import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from speech_qa.domain.errors import IndexIncompatibleError
from speech_qa.domain.models import (
    Chunk,
    IndexedChunk,
    IndexedSpeech,
    IndexManifest,
    InputIssue,
    Speech,
)
from speech_qa.domain.ports import Embedder, SpeechIndex
from speech_qa.text.sentences import SentenceWindowChunker

SCHEMA_VERSION = 1
"""Version of the index layout; increase it when the stored payload changes."""

logger = logging.getLogger(__name__)


def content_hash(speech: Speech) -> str:
    """Hash of everything stored for a speech; a change means re-indexing."""
    canonical = json.dumps(
        [speech.speaker, speech.party, speech.date.isoformat(), speech.text],
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Outcome of an ingest run.

    Attributes:
        speeches_in_input: Valid speeches read from the input.
        new: Speeches added to the index.
        changed: Speeches whose content changed and were re-indexed.
        unchanged: Speeches skipped because they are already indexed.
        chunks_written: Chunks embedded and stored in this run.
        issues: Problems with skipped input lines.
        rebuilt: Whether the index was rebuilt from scratch.
        duration_seconds: Wall-clock duration of the run.
        manifest: The manifest of the index after the run.
    """

    speeches_in_input: int
    new: int
    changed: int
    unchanged: int
    chunks_written: int
    issues: tuple[InputIssue, ...]
    rebuilt: bool
    duration_seconds: float
    manifest: IndexManifest


class IngestionService:
    """Chunks, embeds and stores speeches."""

    def __init__(
        self,
        index: SpeechIndex,
        embedder: Embedder,
        chunker: SentenceWindowChunker,
        *,
        batch_size: int = 32,
        clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Create the service.

        Args:
            index: Where chunks are stored.
            embedder: Computes the chunk embeddings.
            chunker: Splits speeches into chunks.
            batch_size: Number of chunks embedded and stored together.
            clock: Source of the manifest timestamp (injectable for tests).
            timer: Source of the run duration (injectable for tests).
        """
        self._index = index
        self._embedder = embedder
        self._chunker = chunker
        self._batch_size = batch_size
        self._clock = clock
        self._timer = timer

    def ingest(
        self,
        speeches: Sequence[Speech],
        issues: Sequence[InputIssue] = (),
        *,
        rebuild: bool = False,
    ) -> IngestReport:
        """Bring the index up to date with the given speeches.

        Args:
            speeches: Valid speeches to index.
            issues: Problems found while reading the input; passed to the report.
            rebuild: Start from an empty index instead of updating it.

        Returns:
            A summary of the run.

        Raises:
            IndexIncompatibleError: If the existing index was built with another
                embedding model, other chunking parameters or another layout
                (use ``rebuild``).
        """
        started = self._timer()
        indexed: dict[str, IndexedSpeech]
        if rebuild:
            self._index.clear()
            indexed = {}
        else:
            self._check_compatible(self._index.read_manifest())
            indexed = self._index.indexed_speeches()

        new = changed = unchanged = 0
        pending: list[tuple[Speech, str]] = []
        for speech in speeches:
            digest = content_hash(speech)
            known = indexed.get(speech.speech_id)
            if known is not None and known.content_hash == digest:
                unchanged += 1
                continue
            if known is None:
                new += 1
            else:
                changed += 1
            pending.append((speech, digest))

        chunks_written, dimension = self._store(pending)
        manifest = self._build_manifest(dimension)
        self._index.write_manifest(manifest)
        return IngestReport(
            speeches_in_input=len(speeches),
            new=new,
            changed=changed,
            unchanged=unchanged,
            chunks_written=chunks_written,
            issues=tuple(issues),
            rebuilt=rebuild,
            duration_seconds=self._timer() - started,
            manifest=manifest,
        )

    def _check_compatible(self, manifest: IndexManifest | None) -> None:
        if manifest is None:
            return
        if manifest.schema_version != SCHEMA_VERSION:
            reason = f"index layout version {manifest.schema_version}, expected {SCHEMA_VERSION}"
            raise IndexIncompatibleError(reason)
        if manifest.embedding_model != self._embedder.model_name:
            reason = (
                f"embedding model {manifest.embedding_model!r}, "
                f"configured {self._embedder.model_name!r}"
            )
            raise IndexIncompatibleError(reason)
        if manifest.chunking != self._chunker.parameters:
            reason = f"chunking {manifest.chunking}, configured {self._chunker.parameters}"
            raise IndexIncompatibleError(reason)

    def _store(self, pending: Sequence[tuple[Speech, str]]) -> tuple[int, int | None]:
        """Chunk, embed and store speeches in batches.

        Returns:
            The number of stored chunks and the embedding dimension (``None`` if
            nothing was stored).
        """
        written = 0
        dimension: int | None = None
        batch: list[tuple[Speech, str, list[Chunk]]] = []
        batch_chunks = 0
        for position, (speech, digest) in enumerate(pending, start=1):
            chunks = self._chunker.split(speech)
            batch.append((speech, digest, chunks))
            batch_chunks += len(chunks)
            if batch_chunks >= self._batch_size or position == len(pending):
                stored, batch_dimension = self._store_batch(batch)
                written += stored
                dimension = batch_dimension or dimension
                logger.info("indexed %d/%d speeches (%d chunks)", position, len(pending), written)
                batch, batch_chunks = [], 0
        return written, dimension

    def _store_batch(
        self, batch: Sequence[tuple[Speech, str, list[Chunk]]]
    ) -> tuple[int, int | None]:
        items = [(speech, digest, chunk) for speech, digest, chunks in batch for chunk in chunks]
        vectors = self._embedder.embed_documents([chunk.text for _, _, chunk in items])
        indexed_chunks = [
            IndexedChunk(chunk=chunk, speech=speech.info, content_hash=digest, vector=vector)
            for (speech, digest, chunk), vector in zip(items, vectors, strict=True)
        ]
        self._index.replace_speeches([speech.speech_id for speech, _, _ in batch], indexed_chunks)
        return len(indexed_chunks), (len(vectors[0]) if vectors else None)

    def _build_manifest(self, dimension: int | None) -> IndexManifest:
        """Describe the index after the run; the dimension is 0 when unknown."""
        if dimension is None:
            previous = self._index.read_manifest()
            dimension = previous.embedding_dimension if previous is not None else 0
        indexed = self._index.indexed_speeches()
        return IndexManifest(
            schema_version=SCHEMA_VERSION,
            embedding_model=self._embedder.model_name,
            embedding_dimension=dimension,
            chunking=self._chunker.parameters,
            speakers=tuple(sorted({speech.speaker for speech in indexed.values()})),
            parties=tuple(sorted({s.party for s in indexed.values() if s.party is not None})),
            speech_count=len(indexed),
            chunk_count=self._index.chunk_count(),
            updated_at=self._clock(),
        )
