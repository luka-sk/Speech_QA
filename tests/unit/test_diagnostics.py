from pathlib import Path

import httpx2
import openai

from speech_qa.application.ingestion import IngestionService
from speech_qa.config import OllamaSettings, Settings, load_settings
from speech_qa.diagnostics import IndexState, run_checks
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.infrastructure.openai_compatible import create_client
from speech_qa.infrastructure.qdrant_index import MANIFEST_FILENAME, QdrantSpeechIndex
from speech_qa.text.sentences import SentenceWindowChunker
from tests.conftest import LONG_SPEECHES, FakeEmbedder


def models_client(*names: str, down: bool = False) -> openai.OpenAI:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if down:
            raise httpx2.ConnectError("connection refused", request=request)
        data = [{"id": name, "object": "model", "created": 0, "owned_by": "x"} for name in names]
        return httpx2.Response(200, json={"object": "list", "data": data})

    settings = OllamaSettings(base_url="http://ollama.test/v1", max_retries=0)
    return create_client(
        settings, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )


def settings_with_index(tmp_path: Path, **sections: str) -> Settings:
    config = tmp_path / "speech_qa.toml"
    extra = "".join(f"\n[{name}]\n{body}\n" for name, body in sections.items())
    config.write_text(f"[index]\npath = 'index'\n{extra}", encoding="utf-8")
    return load_settings(config, environ={})


def ingest_fixtures(settings: Settings, embedding_model: str = "bge-m3") -> None:
    chunking = settings.chunking
    chunker = SentenceWindowChunker(
        chunking.target_chars, chunking.max_chars, chunking.overlap_sentences
    )
    with QdrantSpeechIndex.open_local(settings.index.path, "speech_chunks", create=True) as index:
        IngestionService(index, FakeEmbedder(embedding_model), chunker).ingest(
            read_speeches(LONG_SPEECHES).speeches
        )


ALL_MODELS = ("bge-m3:latest", "gemma4:12b")


def test_everything_ready(tmp_path: Path) -> None:
    settings = settings_with_index(tmp_path)
    ingest_fixtures(settings)

    report = run_checks(settings, tmp_path / "speech_qa.toml", client=models_client(*ALL_MODELS))

    assert report.models_ok
    assert report.index_ok
    assert report.index_state is IndexState.READY
    assert report.manifest is not None
    assert report.manifest.speech_count == 10
    assert report.index_matches_chunking


def test_server_down(tmp_path: Path) -> None:
    report = run_checks(settings_with_index(tmp_path), None, client=models_client(down=True))

    assert report.server_error is not None
    assert not report.models_ok
    assert report.installed_models == frozenset()


def test_missing_model(tmp_path: Path) -> None:
    report = run_checks(settings_with_index(tmp_path), None, client=models_client("gemma4:12b"))

    assert report.chat_model_available
    assert not report.embedding_model_available
    assert not report.models_ok


def test_missing_and_empty_index(tmp_path: Path) -> None:
    settings = settings_with_index(tmp_path)
    missing = run_checks(settings, None, client=models_client(*ALL_MODELS))
    QdrantSpeechIndex.open_local(settings.index.path, "speech_chunks", create=True).close()
    empty = run_checks(settings, None, client=models_client(*ALL_MODELS))

    assert missing.index_state is IndexState.MISSING
    assert empty.index_state is IndexState.EMPTY
    assert not missing.index_ok
    assert not empty.index_ok


def test_locked_index(tmp_path: Path) -> None:
    settings = settings_with_index(tmp_path)
    with QdrantSpeechIndex.open_local(settings.index.path, "speech_chunks", create=True):
        report = run_checks(settings, None, client=models_client(*ALL_MODELS))

    assert report.index_state is IndexState.LOCKED


def test_unreadable_manifest(tmp_path: Path) -> None:
    settings = settings_with_index(tmp_path)
    ingest_fixtures(settings)
    (settings.index.path / MANIFEST_FILENAME).write_text("{", encoding="utf-8")

    report = run_checks(settings, None, client=models_client(*ALL_MODELS))

    assert report.index_state is IndexState.UNREADABLE
    assert "cannot be read" in report.index_detail


def test_index_built_with_other_settings(tmp_path: Path) -> None:
    ingest_fixtures(settings_with_index(tmp_path), embedding_model="other-model")
    settings = settings_with_index(tmp_path, chunking="target_chars = 500\nmax_chars = 800")

    report = run_checks(settings, None, client=models_client(*ALL_MODELS))

    assert not report.index_matches_embedding_model
    assert not report.index_matches_chunking
    assert not report.index_ok


def test_client_is_created_from_the_settings_when_not_given(tmp_path: Path) -> None:
    settings = settings_with_index(
        tmp_path, ollama="base_url = 'http://127.0.0.1:9/v1'\nmax_retries = 0\ntimeout_seconds = 1"
    )

    report = run_checks(settings, None)

    assert report.server_error is not None
