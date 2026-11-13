"""Shared test helpers: fixture files, a fake embedder and an in-memory index."""

import hashlib
import math
import re
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from speech_qa.domain.models import ChatCompletion, ChatMessage, Vector
from speech_qa.domain.ports import ChatModel, Embedder
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex

FIXTURES = Path(__file__).parent / "fixtures"
LONG_SPEECHES = FIXTURES / "speeches_long.jsonl"


def fold(text: str) -> str:
    """Lower-case text without diacritics."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words embeddings, good enough to test retrieval.

    Words are folded and cut to six characters (a crude stem) and hashed into a
    small vector, so texts sharing words get similar vectors.
    """

    def __init__(self, model_name: str = "fake-embedder", dimension: int = 64) -> None:
        self._model_name = model_name
        self.dimension = dimension
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(list(texts))
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        self.calls.append([text])
        return self.vector(text)

    @property
    def embedded_texts(self) -> int:
        return sum(len(call) for call in self.calls)

    def vector(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        values[-1] = 0.01  # never a zero vector
        for word in re.findall(r"\w+", fold(text)):
            digest = hashlib.sha256(word[:6].encode("utf-8")).digest()
            values[digest[0] % (self.dimension - 1)] += 1.0
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]


class FakeChatModel(ChatModel):
    """Language model that returns a scripted reply and records the requests."""

    def __init__(
        self,
        reply: str | Callable[[Sequence[ChatMessage]], str] = "Odpověď podle zdroje [1].",
        *,
        prompt_tokens: int | None = 100,
    ) -> None:
        self._reply = reply
        self._prompt_tokens = prompt_tokens
        self.calls: list[list[ChatMessage]] = []
        self.max_tokens: list[int | None] = []

    @property
    def model_name(self) -> str:
        return "fake-chat"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        json_output: bool = False,
    ) -> ChatCompletion:
        self.calls.append(list(messages))
        self.max_tokens.append(max_tokens)
        text = self._reply(messages) if callable(self._reply) else self._reply
        return ChatCompletion(text=text, prompt_tokens=self._prompt_tokens, completion_tokens=10)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def memory_index(tmp_path: Path) -> Iterator[QdrantSpeechIndex]:
    index = QdrantSpeechIndex(QdrantClient(":memory:"), "test_chunks", tmp_path / "manifest.json")
    yield index
    index.close()
