"""Adapters for an OpenAI-compatible model server (Ollama) via the OpenAI client.

The models run on-prem; no API key is needed. The OpenAI client nevertheless
requires a non-empty key, so a fixed placeholder is sent (Ollama ignores it).
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from itertools import batched
from typing import Literal

import httpx2
import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from speech_qa.config import OllamaSettings
from speech_qa.domain.errors import (
    ModelNotAvailableError,
    ModelRequestError,
    ModelServiceUnavailableError,
)
from speech_qa.domain.models import ChatCompletion, ChatMessage, ModelPurpose, Vector
from speech_qa.domain.ports import ChatModel, Embedder

type ReasoningEffort = Literal["none", "low", "medium", "high"]

_PLACEHOLDER_API_KEY = "not-needed"
"""Sent instead of an API key; not a secret (see the module docstring)."""


def create_client(
    settings: OllamaSettings, *, http_client: httpx2.Client | None = None
) -> openai.OpenAI:
    """Create an OpenAI client for the configured endpoint.

    Args:
        settings: Endpoint, timeout and retry settings.
        http_client: Custom HTTP client (used by tests); ``None`` for the default.
    """
    return openai.OpenAI(
        base_url=settings.base_url,
        api_key=_PLACEHOLDER_API_KEY,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        http_client=http_client,
    )


@contextmanager
def translate_errors(client: openai.OpenAI, model: str, purpose: ModelPurpose) -> Iterator[None]:
    """Turn OpenAI client exceptions into the application's errors.

    Args:
        client: The client making the request (for the endpoint in messages).
        model: Model the request is made for.
        purpose: What the model is used for.

    Raises:
        ModelServiceUnavailableError: Connection problems, timeouts, server errors,
            or an endpoint that does not exist.
        ModelNotAvailableError: The model is not installed.
        ModelRequestError: Any other rejected request.
    """
    base_url = str(client.base_url).rstrip("/")
    try:
        yield
    except openai.APIConnectionError as exc:  # includes timeouts
        raise ModelServiceUnavailableError(base_url, str(exc)) from exc
    except openai.NotFoundError as exc:
        # Ollama answers 404 both for a missing model and for a wrong URL path.
        if "model" in str(exc.message).lower():
            raise ModelNotAvailableError(model, purpose) from exc
        raise ModelServiceUnavailableError(base_url, f"endpoint not found: {exc.message}") from exc
    except openai.InternalServerError as exc:
        raise ModelServiceUnavailableError(
            base_url, f"HTTP {exc.status_code}: {exc.message}"
        ) from exc
    except openai.APIStatusError as exc:
        raise ModelRequestError(model, f"HTTP {exc.status_code}: {exc.message}") from exc


class OpenAICompatibleEmbedder(Embedder):
    """Embeddings from the ``/embeddings`` endpoint of an OpenAI-compatible server."""

    def __init__(self, client: openai.OpenAI, model: str, *, batch_size: int = 32) -> None:
        """Create the embedder.

        Args:
            client: OpenAI client pointing to the model server.
            model: Name of the embedding model (e.g. ``bge-m3``).
            batch_size: Maximum number of texts sent in one request.
        """
        if batch_size < 1:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        self._client = client
        self._model = model
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        """Name of the embedding model."""
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed texts in batches of at most ``batch_size``."""
        vectors: list[Vector] = []
        for batch in batched(texts, self._batch_size):
            vectors.extend(self._embed(list(batch)))
        return vectors

    def embed_query(self, text: str) -> Vector:
        """Embed a single query (bge-m3 needs no query prefix)."""
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[Vector]:
        with translate_errors(self._client, self._model, ModelPurpose.EMBEDDING):
            response = self._client.embeddings.create(
                model=self._model, input=texts, encoding_format="float"
            )
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(texts):
            detail = f"expected {len(texts)} embeddings, got {len(items)}"
            raise ModelRequestError(self._model, detail)
        return [list(item.embedding) for item in items]


class OpenAICompatibleChatModel(ChatModel):
    """A language model behind the ``/chat/completions`` endpoint."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        *,
        temperature: float = 0.0,
        reasoning_effort: ReasoningEffort = "none",
        default_max_tokens: int = 800,
    ) -> None:
        """Create the chat model.

        Args:
            client: OpenAI client pointing to the model server.
            model: Name of the language model (e.g. ``gemma4:12b``).
            temperature: Sampling temperature; 0 is the most deterministic.
            reasoning_effort: ``"none"`` disables the thinking mode of reasoning models.
            default_max_tokens: Answer length limit when the caller gives none.
        """
        self._client = client
        self._model = model
        self._temperature = temperature
        self._reasoning_effort: ReasoningEffort = reasoning_effort
        self._default_max_tokens = default_max_tokens

    @property
    def model_name(self) -> str:
        """Name of the language model."""
        return self._model

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        json_output: bool = False,
    ) -> ChatCompletion:
        """Generate the next assistant message."""
        response_format: openai.types.chat.completion_create_params.ResponseFormat | openai.Omit
        response_format = {"type": "json_object"} if json_output else openai.omit
        with translate_errors(self._client, self._model, ModelPurpose.CHAT):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[_to_message_param(message) for message in messages],
                temperature=self._temperature,
                max_tokens=max_tokens or self._default_max_tokens,
                reasoning_effort=self._reasoning_effort,
                response_format=response_format,
            )
        if not response.choices:
            raise ModelRequestError(self._model, "the response contains no answer")
        usage = response.usage
        return ChatCompletion(
            text=(response.choices[0].message.content or "").strip(),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )


def _to_message_param(message: ChatMessage) -> ChatCompletionMessageParam:
    match message.role:
        case "system":
            return ChatCompletionSystemMessageParam(role="system", content=message.content)
        case "user":
            return ChatCompletionUserMessageParam(role="user", content=message.content)
        case "assistant":
            return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)


def available_models(client: openai.OpenAI) -> frozenset[str]:
    """Return the names of the models installed on the model server.

    Raises:
        ModelServiceUnavailableError: If the model server cannot be reached.
    """
    with translate_errors(client, "(model list)", ModelPurpose.CHAT):
        return frozenset(model.id for model in client.models.list())


def is_model_available(model: str, installed: frozenset[str]) -> bool:
    """Whether a configured model name is installed (``bge-m3`` means ``bge-m3:latest``)."""
    return model in installed or (":" not in model and f"{model}:latest" in installed)
