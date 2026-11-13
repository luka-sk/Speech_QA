"""Use case: find the speeches relevant to a query.

1. The query analyser turns names, party codes and years into a filter.
2. The rest of the query is embedded and the most similar chunks that satisfy
   the filter are fetched from the index.
3. Chunks are grouped by speech; a speech is ranked by its best chunk and keeps
   its best few chunks as passages.
4. Unless the query names a speaker, at most a few speeches per speaker are
   kept, so that a broad question is answered from several people.

A filter that matches nothing gives an empty result. The search is deliberately
not repeated without the filter: that would attribute other people's words to
the speaker the user asked about.
"""

import logging
from collections import Counter
from dataclasses import dataclass

from speech_qa.domain.errors import IndexIncompatibleError
from speech_qa.domain.models import Evidence, IndexManifest, RetrievalResult, RetrievedChunk
from speech_qa.domain.ports import Embedder, QueryAnalyzer, SpeechIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalLimits:
    """How much is fetched and kept.

    Attributes:
        candidate_chunks: Chunks fetched from the index.
        max_speeches: Speeches kept in the result.
        max_speeches_per_speaker: Speeches kept per speaker when the query names
            no speaker.
        max_passages_per_speech: Chunks kept per speech.
    """

    candidate_chunks: int = 50
    max_speeches: int = 8
    max_speeches_per_speaker: int = 2
    max_passages_per_speech: int = 2

    def __post_init__(self) -> None:
        if (
            min(
                self.candidate_chunks,
                self.max_speeches,
                self.max_speeches_per_speaker,
                self.max_passages_per_speech,
            )
            < 1
        ):
            msg = "all retrieval limits must be positive"
            raise ValueError(msg)


def ensure_queryable(manifest: IndexManifest, embedding_model: str) -> None:
    """Check that queries can be embedded compatibly with the stored chunks.

    Raises:
        IndexIncompatibleError: If the index was built with another embedding model.
    """
    if manifest.embedding_model != embedding_model:
        reason = f"embedding model {manifest.embedding_model!r}, configured {embedding_model!r}"
        raise IndexIncompatibleError(reason)


class Retriever:
    """Finds the speeches relevant to a query."""

    def __init__(
        self,
        analyzer: QueryAnalyzer,
        embedder: Embedder,
        index: SpeechIndex,
        limits: RetrievalLimits | None = None,
    ) -> None:
        """Create the retriever.

        Args:
            analyzer: Turns the query into a filter.
            embedder: Embeds the query (must be the model the index was built with).
            index: The speech index.
            limits: How much is fetched and kept.
        """
        self._analyzer = analyzer
        self._embedder = embedder
        self._index = index
        self._limits = limits or RetrievalLimits()

    def retrieve(self, query: str) -> RetrievalResult:
        """Find the speeches relevant to a query, best first."""
        analysis = self._analyzer.analyze(query)
        logger.info(
            "query filter: %s; search text: %r", analysis.search_filter, analysis.embedding_text
        )
        vector = self._embedder.embed_query(analysis.embedding_text)
        hits = self._index.search(vector, analysis.search_filter, self._limits.candidate_chunks)
        diversify = not analysis.search_filter.speakers
        evidence = self._select(hits, diversify=diversify)
        logger.info("retrieved %d chunks from %d speeches", len(hits), len(evidence))
        return RetrievalResult(analysis=analysis, evidence=evidence)

    def _select(self, hits: list[RetrievedChunk], *, diversify: bool) -> tuple[Evidence, ...]:
        by_speech: dict[str, list[RetrievedChunk]] = {}
        for hit in sorted(hits, key=lambda hit: hit.score, reverse=True):
            by_speech.setdefault(hit.chunk.speech_id, []).append(hit)

        selected: list[Evidence] = []
        per_speaker: Counter[str] = Counter()
        for chunks in by_speech.values():  # ordered by the best chunk of each speech
            speaker = chunks[0].speech.speaker
            if diversify and per_speaker[speaker] >= self._limits.max_speeches_per_speaker:
                continue
            per_speaker[speaker] += 1
            best = chunks[: self._limits.max_passages_per_speech]
            selected.append(
                Evidence(
                    speech=chunks[0].speech,
                    passages=tuple(sorted(best, key=lambda chunk: chunk.chunk.index)),
                    score=chunks[0].score,
                )
            )
            if len(selected) == self._limits.max_speeches:
                break
        return tuple(selected)
