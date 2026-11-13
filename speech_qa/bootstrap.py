"""Composition root: the only place where concrete adapters are chosen and wired.

Everything else depends on the ports in :mod:`speech_qa.domain.ports`; swapping
Qdrant or Ollama for another implementation means changing this module only.
"""

from speech_qa.application.ingestion import IngestionService
from speech_qa.application.qa import AnswerSettings, SpeechQA
from speech_qa.application.retrieval import RetrievalLimits, Retriever, ensure_queryable
from speech_qa.config import Settings
from speech_qa.domain.errors import IndexNotFoundError
from speech_qa.domain.ports import ChatModel, Embedder, SpeechIndex
from speech_qa.evaluation.judge import LLMJudge
from speech_qa.evaluation.runner import EvaluationRunner
from speech_qa.evaluation.synthetic import SyntheticQuestionGenerator
from speech_qa.infrastructure.openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleEmbedder,
    create_client,
)
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex
from speech_qa.messages import render_answer
from speech_qa.text.query_analysis import RuleBasedQueryAnalyzer
from speech_qa.text.sentences import SentenceWindowChunker


def open_index(settings: Settings, *, create: bool) -> SpeechIndex:
    """Open the configured index.

    Args:
        settings: Application settings.
        create: Create the index directory if it does not exist (ingest);
            otherwise a missing index is an error (queries).
    """
    return QdrantSpeechIndex.open_local(
        settings.index.path, settings.index.collection, create=create
    )


def create_embedder(settings: Settings) -> Embedder:
    """Create the embedder for the configured model server."""
    return OpenAICompatibleEmbedder(
        create_client(settings.ollama),
        settings.ollama.embedding_model,
        batch_size=settings.ingest.embedding_batch_size,
    )


def create_chunker(settings: Settings) -> SentenceWindowChunker:
    """Create the chunker with the configured window sizes."""
    return SentenceWindowChunker(
        target_chars=settings.chunking.target_chars,
        max_chars=settings.chunking.max_chars,
        overlap_sentences=settings.chunking.overlap_sentences,
    )


def create_ingestion_service(settings: Settings, index: SpeechIndex) -> IngestionService:
    """Create the ingestion use case for an open index."""
    return IngestionService(
        index,
        create_embedder(settings),
        create_chunker(settings),
        batch_size=settings.ingest.embedding_batch_size,
    )


def create_retriever(settings: Settings, index: SpeechIndex) -> Retriever:
    """Create the retrieval use case for an open, ingested index.

    The query analyser learns the speakers and parties from the index manifest.

    Raises:
        IndexNotFoundError: If nothing has been ingested yet.
        IndexIncompatibleError: If the index was built with another embedding model.
    """
    manifest = index.read_manifest()
    if manifest is None:
        raise IndexNotFoundError(settings.index.path)
    ensure_queryable(manifest, settings.ollama.embedding_model)
    retrieval = settings.retrieval
    return Retriever(
        RuleBasedQueryAnalyzer(manifest.speakers, manifest.parties),
        create_embedder(settings),
        index,
        RetrievalLimits(
            candidate_chunks=retrieval.candidate_chunks,
            max_speeches=retrieval.max_speeches,
            max_speeches_per_speaker=retrieval.max_speeches_per_speaker,
            max_chunks_per_speech=retrieval.max_chunks_per_speech,
        ),
    )


def create_chat_model(settings: Settings) -> ChatModel:
    """Create the language model client for the configured model server."""
    generation = settings.generation
    return OpenAICompatibleChatModel(
        create_client(settings.ollama),
        settings.ollama.chat_model,
        temperature=generation.temperature,
        reasoning_effort=generation.reasoning_effort,
        default_max_tokens=generation.max_answer_tokens,
    )


def create_qa(settings: Settings, index: SpeechIndex) -> SpeechQA:
    """Create the question-answering use case (the ``PQA`` implementation).

    Raises:
        IndexNotFoundError: If nothing has been ingested yet.
        IndexIncompatibleError: If the index was built with another embedding model.
    """
    return SpeechQA(
        create_retriever(settings, index),
        create_chat_model(settings),
        render_answer,
        AnswerSettings(
            max_query_chars=settings.query.max_chars,
            context_window_tokens=settings.generation.context_window_tokens,
            max_answer_tokens=settings.generation.max_answer_tokens,
        ),
    )


def create_evaluation_runner(
    settings: Settings, index: SpeechIndex, *, judge: bool
) -> EvaluationRunner:
    """Create the evaluation runner for an open, ingested index."""
    chat_model = create_chat_model(settings)
    return EvaluationRunner(
        create_retriever(settings, index),
        create_qa(settings, index),
        LLMJudge(chat_model) if judge else None,
        k=settings.retrieval.max_speeches,
        known_ids=index.indexed_speeches().keys(),
    )


def create_synthetic_generator(
    settings: Settings, index: SpeechIndex
) -> SyntheticQuestionGenerator:
    """Create the generator of synthetic evaluation questions for an ingested index.

    Raises:
        IndexNotFoundError: If nothing has been ingested yet.
    """
    manifest = index.read_manifest()
    if manifest is None:
        raise IndexNotFoundError(settings.index.path)
    return SyntheticQuestionGenerator(
        index,
        create_chat_model(settings),
        RuleBasedQueryAnalyzer(manifest.speakers, manifest.parties),
    )


def describe_configuration(settings: Settings, index: SpeechIndex) -> dict[str, str]:
    """Summarise the settings and the index for the evaluation report."""
    manifest = index.read_manifest()
    chunking = settings.chunking
    retrieval = settings.retrieval
    generation = settings.generation
    description = {
        "Jazykový model": settings.ollama.chat_model,
        "Embedding model": settings.ollama.embedding_model,
        "Úseky": (
            f"cíl {chunking.target_chars} znaků, max. {chunking.max_chars}, "
            f"překryv {chunking.overlap_sentences} věta"
        ),
        "Vyhledávání": (
            f"kandidátní úseky: {retrieval.candidate_chunks}, "
            f"projevy: max. {retrieval.max_speeches}, "
            f"na řečníka: max. {retrieval.max_speeches_per_speaker}, "
            f"úseky na projev: max. {retrieval.max_chunks_per_speech}"
        ),
        "Generování": (
            f"teplota {generation.temperature}, reasoning {generation.reasoning_effort}, "
            f"kontext {generation.context_window_tokens} tokenů, "
            f"odpověď max. {generation.max_answer_tokens} tokenů"
        ),
    }
    if manifest is not None:
        description["Index"] = (
            f"{manifest.speech_count} projevů, {manifest.chunk_count} úseků, "
            f"{len(manifest.speakers)} řečníků"
        )
    return description
