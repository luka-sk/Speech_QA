"""Environment checks behind ``python cli.py check``.

Checks the model server, the two models and the index, so that testers can see
at a glance what is missing. The result is structured; the Czech text is built
in :mod:`speech_qa.messages`.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import openai

from speech_qa.config import Settings
from speech_qa.domain.errors import (
    IndexIncompatibleError,
    IndexLockedError,
    IndexNotFoundError,
    ModelServiceUnavailableError,
)
from speech_qa.domain.models import ChunkingParameters, IndexManifest
from speech_qa.infrastructure.openai_compatible import (
    available_models,
    create_client,
    is_model_available,
)
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex


class IndexState(StrEnum):
    """State of the local index."""

    READY = "ready"
    MISSING = "missing"
    EMPTY = "empty"
    """The directory exists but no ingest has completed."""
    LOCKED = "locked"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class CheckReport:
    """Result of the environment checks.

    Attributes:
        config_file: The configuration file in use, or ``None`` for the defaults.
        settings: The effective settings.
        server_error: Why the model server cannot be reached; ``None`` if it can.
        installed_models: Models installed on the model server.
        index_state: State of the index.
        index_detail: Technical detail for an unreadable index.
        manifest: The index manifest, if the index is ready.
    """

    config_file: Path | None
    settings: Settings
    server_error: str | None
    installed_models: frozenset[str]
    index_state: IndexState
    index_detail: str
    manifest: IndexManifest | None

    @property
    def chat_model_available(self) -> bool:
        """Whether the configured language model is installed."""
        return is_model_available(self.settings.ollama.chat_model, self.installed_models)

    @property
    def embedding_model_available(self) -> bool:
        """Whether the configured embedding model is installed."""
        return is_model_available(self.settings.ollama.embedding_model, self.installed_models)

    @property
    def index_matches_embedding_model(self) -> bool:
        """Whether the index can be queried with the configured embedding model."""
        return (
            self.manifest is not None
            and self.manifest.embedding_model == self.settings.ollama.embedding_model
        )

    @property
    def index_matches_chunking(self) -> bool:
        """Whether the index was chunked with the configured parameters."""
        chunking = self.settings.chunking
        configured = ChunkingParameters(
            chunking.target_chars, chunking.max_chars, chunking.overlap_sentences
        )
        return self.manifest is not None and self.manifest.chunking == configured

    @property
    def models_ok(self) -> bool:
        """Whether the model server is reachable and both models are installed."""
        return (
            self.server_error is None
            and self.chat_model_available
            and self.embedding_model_available
        )

    @property
    def index_ok(self) -> bool:
        """Whether questions can be answered from the index."""
        return self.index_state is IndexState.READY and self.index_matches_embedding_model


def run_checks(
    settings: Settings, config_file: Path | None, *, client: openai.OpenAI | None = None
) -> CheckReport:
    """Check the model server, the models and the index.

    Args:
        settings: The effective settings.
        config_file: The configuration file in use (for the report).
        client: OpenAI client to use (tests); ``None`` creates one from the settings.
    """
    try:
        installed = available_models(client or create_client(settings.ollama))
        server_error = None
    except ModelServiceUnavailableError as error:
        installed, server_error = frozenset(), error.detail

    manifest: IndexManifest | None = None
    detail = ""
    try:
        with QdrantSpeechIndex.open_local(
            settings.index.path, settings.index.collection, create=False
        ) as index:
            manifest = index.read_manifest()
        state = IndexState.READY if manifest else IndexState.EMPTY
    except IndexNotFoundError:
        state = IndexState.MISSING
    except IndexLockedError:
        state = IndexState.LOCKED
    except IndexIncompatibleError as error:
        state, detail = IndexState.UNREADABLE, error.reason

    return CheckReport(
        config_file=config_file,
        settings=settings,
        server_error=server_error,
        installed_models=installed,
        index_state=state,
        index_detail=detail,
        manifest=manifest,
    )
