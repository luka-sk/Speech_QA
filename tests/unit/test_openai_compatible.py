import json
from collections.abc import Callable

import httpx2
import pytest

from speech_qa.config import OllamaSettings
from speech_qa.domain.errors import (
    ModelNotAvailableError,
    ModelRequestError,
    ModelServiceUnavailableError,
)
from speech_qa.domain.models import ChatMessage, ModelPurpose
from speech_qa.infrastructure.openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleEmbedder,
    available_models,
    create_client,
    is_model_available,
)

type Handler = Callable[[httpx2.Request], httpx2.Response]


def make_embedder(handler: Handler, batch_size: int = 32) -> OpenAICompatibleEmbedder:
    settings = OllamaSettings(base_url="http://ollama.test/v1", timeout_seconds=5, max_retries=0)
    client = create_client(
        settings, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    return OpenAICompatibleEmbedder(client, "bge-m3", batch_size=batch_size)


def embeddings_response(texts: list[str], *, reverse: bool = False) -> httpx2.Response:
    data = [
        {"object": "embedding", "index": index, "embedding": [float(len(text)), float(index)]}
        for index, text in enumerate(texts)
    ]
    if reverse:
        data.reverse()
    body = {
        "object": "list",
        "model": "bge-m3",
        "data": data,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
    return httpx2.Response(200, json=body)


def error_response(status: int, message: str) -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"message": message, "type": "api_error"}})


def test_embeds_in_batches_and_keeps_the_order() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        return embeddings_response(body["input"], reverse=True)

    embedder = make_embedder(handler, batch_size=2)

    vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [len(body["input"]) for body in requests] == [2, 2, 1]  # type: ignore[arg-type]
    assert all(body["model"] == "bge-m3" for body in requests)
    assert all(body["encoding_format"] == "float" for body in requests)


def test_sends_requests_to_the_configured_endpoint_without_a_real_key() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return embeddings_response(json.loads(request.content)["input"])

    make_embedder(handler).embed_query("dotaz")

    assert str(seen[0].url) == "http://ollama.test/v1/embeddings"
    assert seen[0].headers["authorization"] == "Bearer not-needed"


def test_embed_query_returns_one_vector() -> None:
    embedder = make_embedder(
        lambda request: embeddings_response(json.loads(request.content)["input"])
    )

    assert embedder.embed_query("abc") == [3.0, 0.0]
    assert embedder.model_name == "bge-m3"


def test_missing_model_is_reported_as_not_available() -> None:
    embedder = make_embedder(
        lambda request: error_response(404, 'model "bge-m3" not found, try pulling it first')
    )

    with pytest.raises(ModelNotAvailableError) as caught:
        embedder.embed_query("dotaz")

    assert caught.value.model == "bge-m3"
    assert caught.value.purpose is ModelPurpose.EMBEDDING


def test_wrong_endpoint_path_is_reported_as_unavailable_service() -> None:
    embedder = make_embedder(lambda request: error_response(404, "404 page not found"))

    with pytest.raises(ModelServiceUnavailableError, match="endpoint not found"):
        embedder.embed_query("dotaz")


def test_connection_failure_is_reported_as_unavailable_service() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(ModelServiceUnavailableError) as caught:
        make_embedder(handler).embed_query("dotaz")

    assert caught.value.base_url == "http://ollama.test/v1"


def test_server_error_is_reported_as_unavailable_service() -> None:
    embedder = make_embedder(lambda request: error_response(500, "out of memory"))

    with pytest.raises(ModelServiceUnavailableError, match="HTTP 500"):
        embedder.embed_query("dotaz")


def test_rejected_request_is_reported_as_request_error() -> None:
    embedder = make_embedder(
        lambda request: error_response(400, "input exceeds maximum context length")
    )

    with pytest.raises(ModelRequestError, match="maximum context length"):
        embedder.embed_query("dotaz")


def test_wrong_number_of_embeddings_is_a_request_error() -> None:
    embedder = make_embedder(lambda request: embeddings_response(["only one"]))

    with pytest.raises(ModelRequestError, match="expected 2 embeddings, got 1"):
        embedder.embed_documents(["a", "b"])


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_embedder(lambda request: embeddings_response([]), batch_size=0)


# --------------------------------------------------------------------------- #
# Chat model and model list
# --------------------------------------------------------------------------- #
def make_chat(handler: Handler, **options: object) -> OpenAICompatibleChatModel:
    settings = OllamaSettings(base_url="http://ollama.test/v1", timeout_seconds=5, max_retries=0)
    client = create_client(
        settings, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    return OpenAICompatibleChatModel(client, "gemma4:12b", **options)  # type: ignore[arg-type]


def chat_response(content: str | None, *, choices: bool = True) -> httpx2.Response:
    body = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "gemma4:12b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ]
        if choices
        else [],
        "usage": {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290},
    }
    return httpx2.Response(200, json=body)


MESSAGES = [
    ChatMessage("system", "Pravidla."),
    ChatMessage("user", "Otázka?"),
    ChatMessage("assistant", "…"),
]


def test_chat_request_uses_the_configured_generation_settings() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return chat_response("  Odpověď [1].  ")

    chat = make_chat(handler, temperature=0.0, reasoning_effort="none", default_max_tokens=800)

    completion = chat.complete(MESSAGES)
    chat.complete(MESSAGES, max_tokens=50, json_output=True)

    first, second = bodies
    assert first["model"] == "gemma4:12b"
    assert first["messages"] == [
        {"role": "system", "content": "Pravidla."},
        {"role": "user", "content": "Otázka?"},
        {"role": "assistant", "content": "…"},
    ]
    assert first["temperature"] == 0.0
    assert first["reasoning_effort"] == "none"
    assert first["max_tokens"] == 800
    assert "response_format" not in first
    assert second["max_tokens"] == 50
    assert second["response_format"] == {"type": "json_object"}
    assert completion.text == "Odpověď [1]."
    assert (completion.prompt_tokens, completion.completion_tokens) == (1234, 56)
    assert chat.model_name == "gemma4:12b"


def test_empty_chat_content_becomes_empty_text() -> None:
    assert make_chat(lambda request: chat_response(None)).complete(MESSAGES).text == ""


def test_chat_response_without_choices_is_a_request_error() -> None:
    with pytest.raises(ModelRequestError, match="no answer"):
        make_chat(lambda request: chat_response("x", choices=False)).complete(MESSAGES)


def test_missing_chat_model_is_reported_as_not_available() -> None:
    chat = make_chat(lambda request: error_response(404, 'model "gemma4:12b" not found'))

    with pytest.raises(ModelNotAvailableError) as caught:
        chat.complete(MESSAGES)

    assert caught.value.purpose is ModelPurpose.CHAT


def test_available_models_lists_the_installed_models() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/models"
        models = [
            {"id": name, "object": "model", "created": 0, "owned_by": "library"}
            for name in ("bge-m3:latest", "gemma4:12b")
        ]
        return httpx2.Response(200, json={"object": "list", "data": models})

    settings = OllamaSettings(base_url="http://ollama.test/v1", max_retries=0)
    client = create_client(
        settings, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )

    assert available_models(client) == {"bge-m3:latest", "gemma4:12b"}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("bge-m3", True),
        ("bge-m3:latest", True),
        ("gemma4:12b", True),
        ("gemma4", False),
        ("x", False),
    ],
)
def test_is_model_available(model: str, expected: bool) -> None:
    assert is_model_available(model, frozenset({"bge-m3:latest", "gemma4:12b"})) is expected
