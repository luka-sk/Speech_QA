"""Ports: interfaces the application core needs from the outside world.

The application layer depends only on these abstractions; concrete adapters
(Ollama, Qdrant, ...) live in the infrastructure layer and are wired together in
the composition root. Ports exist only where there is more than one
implementation or where tests need a replacement (Dependency Inversion).
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator, Sequence
from typing import Self

from speech_qa.domain.models import (
    ChatCompletion,
    ChatMessage,
    IndexedChunk,
    IndexedSpeech,
    IndexManifest,
    QueryAnalysis,
    RetrievedChunk,
    SearchFilter,
    StoredChunk,
    Vector,
)


class Embedder(ABC):
    """Turns texts into embedding vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model; stored in the index manifest."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed texts that will be stored in the index.

        Args:
            texts: Texts to embed.

        Returns:
            One vector per text, in the same order.

        Raises:
            ModelServiceUnavailableError: If the model server cannot be reached.
            ModelNotAvailableError: If the embedding model is not installed.
        """

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """Embed a search query.

        Kept separate from :meth:`embed_documents` because some embedding models
        expect a different prefix or instruction for queries.

        Raises:
            ModelServiceUnavailableError: If the model server cannot be reached.
            ModelNotAvailableError: If the embedding model is not installed.
        """


class ChatModel(ABC):
    """A language model that generates text from a conversation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the language model."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        json_output: bool = False,
    ) -> ChatCompletion:
        """Generate the next assistant message.

        Args:
            messages: The conversation so far (system and user messages).
            max_tokens: Upper limit for the generated text; ``None`` uses the default.
            json_output: Ask the model to return a single JSON object.

        Raises:
            ModelServiceUnavailableError: If the model server cannot be reached.
            ModelNotAvailableError: If the language model is not installed.
        """


class SpeechIndex(ABC):
    """Persistent store of speech chunks with similarity search.

    Can be used as a context manager; :meth:`close` is called on exit.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @abstractmethod
    def read_manifest(self) -> IndexManifest | None:
        """Return the manifest, or ``None`` if no ingest has completed yet.

        Raises:
            IndexIncompatibleError: If the manifest exists but cannot be read.
        """

    @abstractmethod
    def write_manifest(self, manifest: IndexManifest) -> None:
        """Store the manifest after a successful ingest."""

    @abstractmethod
    def indexed_speeches(self) -> dict[str, IndexedSpeech]:
        """Return what the index knows about every stored speech, keyed by speech id."""

    @abstractmethod
    def chunk_count(self) -> int:
        """Return the number of stored chunks."""

    @abstractmethod
    def iter_chunks(self) -> Iterator[StoredChunk]:
        """Iterate over all stored chunks (in no particular order)."""

    @abstractmethod
    def replace_speeches(self, speech_ids: Collection[str], chunks: Sequence[IndexedChunk]) -> None:
        """Delete all chunks of the given speeches and store the new chunks.

        Args:
            speech_ids: Speeches whose existing chunks are removed first.
            chunks: New chunks; each must belong to one of ``speech_ids``.

        Raises:
            ValueError: If a chunk belongs to a speech not listed in ``speech_ids``.
        """

    @abstractmethod
    def search(
        self, vector: Vector, search_filter: SearchFilter, limit: int
    ) -> list[RetrievedChunk]:
        """Return the chunks most similar to ``vector`` that satisfy the filter.

        Args:
            vector: Embedding of the query.
            search_filter: Restrictions on speaker, party and date.
            limit: Maximum number of chunks to return.

        Returns:
            Matching chunks ordered from the most to the least similar.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks and the manifest (used by ``ingest --rebuild``)."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying storage (e.g. the lock of a local index)."""


class QueryAnalyzer(ABC):
    """Recognises speakers, parties and dates in a query and builds a filter."""

    @abstractmethod
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyse a user query.

        Args:
            query: The user's natural-language question.

        Returns:
            The recognised mentions and the search filter derived from them.
        """
