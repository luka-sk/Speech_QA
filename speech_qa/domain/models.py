"""Entities and value objects of the speech Q&A domain.

All objects are immutable (frozen dataclasses), so they can be shared freely
between components without defensive copying.
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

type Vector = Sequence[float]
"""An embedding vector."""


# --------------------------------------------------------------------------- #
# Speeches and chunks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SpeechInfo:
    """Metadata of a speech, without its text.

    Attributes:
        speech_id: Stable unique identifier of the speech (used in citations).
        speaker: Full name of the speaker as it appears in the source data.
        party: Party of the speaker, or ``None`` for a speaker without a party.
        date: Date the speech was given.
    """

    speech_id: str
    speaker: str
    party: str | None
    date: dt.date


@dataclass(frozen=True, slots=True)
class Speech:
    """A speech as loaded from the source data.

    Attributes:
        speech_id: Stable unique identifier of the speech (used in citations).
        speaker: Full name of the speaker as it appears in the source data.
        party: Party of the speaker, or ``None`` for a speaker without a party.
        date: Date the speech was given.
        text: Full text of the speech.
    """

    speech_id: str
    speaker: str
    party: str | None
    date: dt.date
    text: str

    @property
    def info(self) -> SpeechInfo:
        """Metadata of the speech without its text."""
        return SpeechInfo(
            speech_id=self.speech_id, speaker=self.speaker, party=self.party, date=self.date
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous part of a speech; the unit of retrieval.

    Attributes:
        speech_id: Identifier of the speech the chunk belongs to.
        index: Zero-based position of the chunk within its speech.
        text: Text of the chunk.
    """

    speech_id: str
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """A chunk prepared for storage in the index.

    Attributes:
        chunk: The chunk itself.
        speech: Metadata of the speech the chunk belongs to.
        content_hash: Hash of the whole speech; used to detect changed speeches.
        vector: Embedding of the chunk text.
    """

    chunk: Chunk
    speech: SpeechInfo
    content_hash: str
    vector: Vector


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A chunk stored in the index, with the metadata of its speech."""

    chunk: Chunk
    speech: SpeechInfo


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by a similarity search.

    Attributes:
        chunk: The matching chunk.
        speech: Metadata of the speech the chunk belongs to.
        score: Similarity to the query; higher is more similar.
    """

    chunk: Chunk
    speech: SpeechInfo
    score: float


@dataclass(frozen=True, slots=True)
class IndexedSpeech:
    """What the index knows about a stored speech.

    Attributes:
        content_hash: Hash of the speech content when it was indexed.
        speaker: Speaker of the speech.
        party: Party of the speaker, or ``None``.
    """

    content_hash: str
    speaker: str
    party: str | None


# --------------------------------------------------------------------------- #
# Input data problems
# --------------------------------------------------------------------------- #
class IssueKind(StrEnum):
    """Kinds of problems found in a line of an input file."""

    LINE_TOO_LONG = "line_too_long"
    NOT_UTF8 = "not_utf8"
    INVALID_JSON = "invalid_json"
    NOT_AN_OBJECT = "not_an_object"
    INVALID_FIELDS = "invalid_fields"
    DUPLICATE_ID = "duplicate_id"


@dataclass(frozen=True, slots=True)
class InputIssue:
    """A problem with one line of an input file; the line is skipped.

    Attributes:
        line: One-based line number.
        kind: What is wrong.
        detail: Technical detail (e.g. which field is invalid).
    """

    line: int
    kind: IssueKind
    detail: str = ""


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SearchFilter:
    """Restrictions applied to a search. Empty fields mean "no restriction".

    Attributes:
        speakers: Only speeches by one of these speakers.
        parties: Only speeches by members of one of these parties.
        date_from: Only speeches given on or after this date.
        date_to: Only speeches given on or before this date.

    Raises:
        ValueError: If ``date_from`` is later than ``date_to``.
    """

    speakers: frozenset[str] = frozenset()
    parties: frozenset[str] = frozenset()
    date_from: dt.date | None = None
    date_to: dt.date | None = None

    def __post_init__(self) -> None:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            msg = f"date_from ({self.date_from}) is later than date_to ({self.date_to})"
            raise ValueError(msg)

    @property
    def is_empty(self) -> bool:
        """Whether the filter places no restriction at all."""
        return not (self.speakers or self.parties or self.date_from or self.date_to)


@dataclass(frozen=True, slots=True)
class SpeakerMention:
    """A speaker reference recognised in a query.

    Attributes:
        surface: The text as written in the query, e.g. ``"Doležala"``.
        speakers: All speakers in the archive the text can refer to. More than
            one speaker means the reference is ambiguous (shared surname).
    """

    surface: str
    speakers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """Result of analysing a user query.

    Attributes:
        query: The original query text.
        search_filter: Filter derived from the query.
        speaker_mentions: Speakers recognised in the query.
        party_mentions: Parties recognised in the query.
        search_text: The query without the recognised mentions (they are
            handled by the filter); empty to search with the whole query.
    """

    query: str
    search_filter: SearchFilter = field(default_factory=SearchFilter)
    speaker_mentions: tuple[SpeakerMention, ...] = ()
    party_mentions: tuple[str, ...] = ()
    search_text: str = ""

    @property
    def embedding_text(self) -> str:
        """Text used for the semantic search."""
        return self.search_text or self.query


@dataclass(frozen=True, slots=True)
class Evidence:
    """A speech found for a query, with its best matching chunks.

    Attributes:
        speech: Metadata of the speech.
        passages: The best matching chunks of the speech, in document order.
        score: Similarity of the best matching chunk.
    """

    speech: SpeechInfo
    passages: tuple[RetrievedChunk, ...]
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Speeches found for a query, best first.

    Attributes:
        analysis: How the query was understood.
        evidence: The found speeches; empty if nothing matches the filter.
    """

    analysis: QueryAnalysis
    evidence: tuple[Evidence, ...]


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Citation:
    """A source cited in an answer.

    Attributes:
        number: Number of the source as shown in the answer text (1, 2, ...).
        evidence: The cited speech and its passages.
    """

    number: int
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class AnswerTimings:
    """Duration of the steps of answering, in seconds."""

    retrieval_seconds: float
    generation_seconds: float


@dataclass(frozen=True, slots=True)
class Answer:
    """An answer to a question, with its sources.

    Attributes:
        question: The question as asked.
        text: The answer; citations refer to :attr:`citations` by number.
        citations: Cited sources in the order of their first citation.
        unknown_citations: Source numbers cited by the model that it was not given.
        sources: All sources given to the model.
        analysis: How the question was understood (filter, recognised names).
        found_nothing: No speech matched the question; the model was not asked.
        prompt_tokens: Prompt size reported by the model server, if known.
        context_may_be_truncated: The prompt came close to the context window,
            so the model server may have cut part of it.
        timings: Duration of retrieval and generation.
    """

    question: str
    text: str
    citations: tuple[Citation, ...]
    unknown_citations: tuple[str, ...]
    sources: tuple[Evidence, ...]
    analysis: QueryAnalysis
    found_nothing: bool
    prompt_tokens: int | None
    context_may_be_truncated: bool
    timings: AnswerTimings


# --------------------------------------------------------------------------- #
# Language model
# --------------------------------------------------------------------------- #
class ModelPurpose(StrEnum):
    """What a model is used for; lets errors tell the user what is missing."""

    EMBEDDING = "embedding"
    CHAT = "chat"


type ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single message of a chat conversation."""

    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    """Result of a chat completion.

    Attributes:
        text: Generated text.
        prompt_tokens: Prompt size reported by the server, if available.
        completion_tokens: Answer size reported by the server, if available.
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# --------------------------------------------------------------------------- #
# Index metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ChunkingParameters:
    """Parameters the index was chunked with; a change requires a rebuild."""

    target_chars: int
    max_chars: int
    overlap_sentences: int


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Metadata describing the content and compatibility of an index.

    Attributes:
        schema_version: Version of the index layout; a mismatch requires a rebuild.
        embedding_model: Embedding model the vectors were computed with.
        embedding_dimension: Length of the stored vectors.
        chunking: Chunking parameters the speeches were split with.
        speakers: All speakers in the index (used for query analysis).
        parties: All parties in the index (used for query analysis).
        speech_count: Number of indexed speeches.
        chunk_count: Number of indexed chunks.
        updated_at: Time of the last successful ingest.
    """

    schema_version: int
    embedding_model: str
    embedding_dimension: int
    chunking: ChunkingParameters
    speakers: tuple[str, ...]
    parties: tuple[str, ...]
    speech_count: int
    chunk_count: int
    updated_at: dt.datetime
