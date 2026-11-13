"""Evaluation tooling with the real Ollama (run with: pytest -m integration --no-cov).

Uses three fixture questions and four synthetic questions to keep the run short.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from speech_qa import bootstrap
from speech_qa.application.ingestion import IngestionService
from speech_qa.config import Settings
from speech_qa.domain.ports import SpeechIndex
from speech_qa.evaluation.dataset import EvalSet, read_eval_set
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex
from tests.conftest import FIXTURES, LONG_SPEECHES

pytestmark = pytest.mark.integration

SETTINGS = Settings()


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SpeechIndex]:
    path: Path = tmp_path_factory.mktemp("evaluation") / "index"
    with QdrantSpeechIndex.open_local(path, SETTINGS.index.collection, create=True) as opened:
        IngestionService(
            opened, bootstrap.create_embedder(SETTINGS), bootstrap.create_chunker(SETTINGS)
        ).ingest(read_speeches(LONG_SPEECHES).speeches)
        yield opened


def test_evaluation_of_fixture_questions(index: SpeechIndex) -> None:
    manual = read_eval_set(FIXTURES / "eval_manual.csv", "manual")
    subset = EvalSet(
        "manual",
        manual.path,
        tuple(q for q in manual.questions if q.question_id in {"q1", "q3", "q7"}),
    )

    result = bootstrap.create_evaluation_runner(SETTINGS, index, judge=True).run([subset])

    summary = result.summaries[0]
    assert summary.answered == 3
    assert summary.hit_at_5 == 1.0
    assert summary.speaker_recall == 1.0
    assert summary.citation_validity == 1.0
    assert summary.judge_failures == 0
    assert summary.faithfulness is not None
    assert summary.faithfulness >= 3


def test_synthetic_questions_are_generated(index: SpeechIndex) -> None:
    result = bootstrap.create_synthetic_generator(SETTINGS, index).generate(4, seed=42)

    assert len(result.questions) == 4
    for question in result.questions:
        assert question.relevant_ids[0].startswith("F-")
        assert question.question.endswith("?")
    assert all(question.expected_speakers for question in result.questions[0::2])
