import datetime as dt
import json
from pathlib import Path

import pytest

from speech_qa.application.ingestion import IngestionService
from speech_qa.application.qa import SpeechQA
from speech_qa.application.retrieval import Retriever
from speech_qa.domain.models import (
    Answer,
    AnswerTimings,
    Chunk,
    Citation,
    Evidence,
    QueryAnalysis,
    RetrievedChunk,
    SpeechInfo,
)
from speech_qa.evaluation.dataset import EvalQuestion, EvalSet, read_eval_set
from speech_qa.evaluation.judge import LLMJudge
from speech_qa.evaluation.report import render_markdown, render_summary, write_reports
from speech_qa.evaluation.runner import (
    EvaluationResult,
    EvaluationRunner,
    sources_as_displayed,
    summarize,
)
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex
from speech_qa.messages import render_answer
from speech_qa.text.query_analysis import RuleBasedQueryAnalyzer
from speech_qa.text.sentences import SentenceWindowChunker
from tests.conftest import FIXTURES, LONG_SPEECHES, FakeChatModel, FakeEmbedder

NOW = dt.datetime(2026, 11, 12, 18, 0, tzinfo=dt.UTC)
CONFIGURATION = {"Jazykový model": "fake-chat"}
JUDGE_REPLY = '{"faithfulness": 4, "relevance": 5, "reason": "fine"}'


@pytest.fixture
def ingested(memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder) -> QdrantSpeechIndex:
    chunker = SentenceWindowChunker(target_chars=1000, max_chars=1500, overlap_sentences=1)
    IngestionService(memory_index, fake_embedder, chunker).ingest(
        read_speeches(LONG_SPEECHES).speeches
    )
    return memory_index


def make_runner(
    index: QdrantSpeechIndex,
    embedder: FakeEmbedder,
    *,
    judge: bool = True,
    answer: str = "Ano [1].",
) -> EvaluationRunner:
    manifest = index.read_manifest()
    assert manifest is not None
    retriever = Retriever(
        RuleBasedQueryAnalyzer(manifest.speakers, manifest.parties), embedder, index
    )
    qa = SpeechQA(retriever, FakeChatModel(answer), render_answer)
    ticks = iter(float(n) for n in range(10_000))
    return EvaluationRunner(
        retriever,
        qa,
        LLMJudge(FakeChatModel(JUDGE_REPLY)) if judge else None,
        k=8,
        known_ids=index.indexed_speeches().keys(),
        clock=lambda: NOW,
        timer=lambda: next(ticks),
    )


def synthetic_set(count: int) -> EvalSet:
    questions = tuple(
        EvalQuestion(
            f"syn-{n}",
            "Co navrhoval Svoboda k dálnicím?",
            ("F-006",),
            ("Martin Svoboda",),
            "synthetic",
        )
        for n in range(count)
    )
    return EvalSet("synthetic", Path("synthetic.csv"), questions)


def test_retrieval_is_measured_for_every_question(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    manual = read_eval_set(FIXTURES / "eval_manual.csv", "manual")

    result = make_runner(ingested, fake_embedder).run([manual], retrieval_only=True)

    assert len(result.results) == 7
    assert all(item.answer is None for item in result.results)
    first = result.results[0]
    assert first.detected_speakers == ("Lukáš Doležal", "Ondřej Doležal")
    assert set(first.retrieved_ids) <= {"F-001", "F-002", "F-003"}
    summary = result.summaries[0]
    assert (summary.questions, summary.answerable, summary.answered) == (7, 6, 0)
    assert summary.hit_at_5 is not None
    assert summary.hit_at_5 > 0.5
    assert summary.speaker_precision == 1.0
    assert summary.speaker_recall == 1.0
    assert summary.faithfulness is None
    assert result.missing_ids == {"manual": ()}
    assert result.k == 8


def test_manual_questions_are_all_answered_and_judged(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    manual = read_eval_set(FIXTURES / "eval_manual.csv", "manual")

    result = make_runner(ingested, fake_embedder).run([manual])

    summary = result.summaries[0]
    assert summary.answered == 7
    assert summary.citation_validity == 1.0
    assert summary.faithfulness == 4.0
    assert summary.relevance == 5.0
    assert summary.judge_failures == 0
    assert summary.false_abstention == 0.0  # every answer cites [1]
    assert summary.correct_abstention == 0.0  # also the unanswerable q7
    assert summary.latency_p50 is not None


def test_only_a_sample_of_synthetic_questions_is_answered(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    result = make_runner(ingested, fake_embedder, judge=False).run(
        [synthetic_set(10)], e2e_sample=3
    )

    answered = [item for item in result.results if item.answer is not None]
    assert len(result.results) == 10
    assert len(answered) == 3
    assert all(item.answer is not None and item.answer.verdict is None for item in answered)
    assert result.summaries[0].judge_failures == 0


def test_abstaining_answers_are_scored(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    manual = read_eval_set(FIXTURES / "eval_manual.csv", "manual")

    result = make_runner(ingested, fake_embedder, answer="Archiv o tom nic neobsahuje.").run(
        [manual]
    )

    summary = result.summaries[0]
    assert summary.correct_abstention == 1.0
    assert summary.false_abstention == 1.0
    assert summary.citation_precision is None


def test_unknown_relevant_ids_and_empty_answers_are_reported(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    questions = (
        EvalQuestion("x1", "Co se říkalo v roce 2030?", ("Q-404",)),
        EvalQuestion("x2", "Co říká Smith?", ("F-010",), ("Eva Smith",)),
    )

    result = make_runner(ingested, fake_embedder).run([EvalSet("manual", Path("m.csv"), questions)])

    assert result.missing_ids == {"manual": ("Q-404",)}
    nothing_found = result.results[0].answer
    assert nothing_found is not None
    assert nothing_found.found_nothing
    assert not nothing_found.judged
    assert result.summaries[0].judge_failures == 0


def test_failed_judge_verdicts_are_counted(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    runner = make_runner(ingested, fake_embedder)
    runner._judge = LLMJudge(FakeChatModel("nonsense"))  # the only way to swap the judge

    result = runner.run(
        [EvalSet("manual", Path("m.csv"), (EvalQuestion("q", "Co říká Smith?", ("F-010",)),))]
    )

    assert result.summaries[0].judge_failures == 1
    assert result.summaries[0].faithfulness is None


def test_summary_of_an_empty_set() -> None:
    summary = summarize("empty", [], 8)

    assert summary.questions == 0
    assert summary.mrr is None
    assert summary.latency_p95 is None


def run_with_report(index: QdrantSpeechIndex, embedder: FakeEmbedder) -> EvaluationResult:
    manual = read_eval_set(FIXTURES / "eval_manual.csv", "manual")
    return make_runner(index, embedder).run([manual, synthetic_set(4)], e2e_sample=2)


def test_markdown_report(ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder) -> None:
    markdown = render_markdown(run_with_report(ingested, fake_embedder), CONFIGURATION)

    assert markdown.startswith("# Evaluace Speech QA – 12. 11. 2026")
    assert "- Jazykový model: fake-chat" in markdown
    assert "| manual | 7 | 6 |" in markdown
    assert "| synthetic | 4 | 4 |" in markdown
    assert "## Odpovědi" in markdown
    assert "## Doba odezvy (s)" in markdown
    assert "## Nejproblematičtější otázky" in markdown
    assert "## Omezení měření" in markdown
    assert "q7" in markdown  # the unanswerable question was answered with a citation


def test_report_files_and_summary(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    result = run_with_report(ingested, fake_embedder)

    markdown, data = write_reports(result, CONFIGURATION, tmp_path / "reports")

    assert markdown.name.startswith("evaluation-")
    assert markdown.suffix == ".md"
    payload = json.loads(data.read_text(encoding="utf-8"))
    assert payload["configuration"] == CONFIGURATION
    assert len(payload["results"]) == 11
    assert payload["started_at"].startswith("2026-11-12")
    summary = render_summary(result, markdown)
    assert summary.startswith("Evaluace dokončena za")
    assert "  manual: 7 otázek" in summary
    assert "odpovědi: 2" in summary  # synthetic sample
    assert str(markdown) in summary


def test_report_without_problems_and_with_missing_ids() -> None:
    result = EvaluationResult(
        started_at=NOW,
        duration_seconds=60.0,
        k=8,
        results=(),
        summaries=(summarize("manual", [], 8),),
        missing_ids={"manual": ("Q-1",)},
    )

    markdown = render_markdown(result, {})

    assert "Žádná otázka nevykazuje problém." in markdown
    assert "- manual: Q-1" in markdown
    assert "| manual | 0 | 0 | – |" in markdown


def test_judge_sees_the_sources_numbered_as_in_the_displayed_answer(
    ingested: QdrantSpeechIndex, fake_embedder: FakeEmbedder
) -> None:
    judge_chat = FakeChatModel(JUDGE_REPLY)
    runner = make_runner(ingested, fake_embedder, answer="Nejdřív druhý [2], pak první [1].")
    runner._judge = LLMJudge(judge_chat)  # the only way to swap the judge
    question = EvalQuestion("q", "Změnil Svoboda názor na dálnice?", ("F-006", "F-007"))

    result = runner.run([EvalSet("manual", Path("m.csv"), (question,))])

    outcome = result.results[0].answer
    assert outcome is not None
    first_cited = outcome.cited_ids[0]
    judge_prompt = judge_chat.calls[0][1].content
    assert outcome.text == "Nejdřív druhý [1], pak první [2]."
    source_one = judge_prompt.split('<source number="1"', 1)[1].split("</source>", 1)[0]
    speeches = {item.speech_id: item for item in read_speeches(LONG_SPEECHES).speeches}
    assert speeches[first_cited].date.isoformat() in source_one


def test_sources_as_displayed_puts_cited_sources_first() -> None:
    def evidence(speech_id: str) -> Evidence:
        info = SpeechInfo(speech_id, "A B", None, dt.date(2020, 1, 1))
        return Evidence(info, (RetrievedChunk(Chunk(speech_id, 0, "t"), info, 0.5),), 0.5)

    first, second, third = evidence("S1"), evidence("S2"), evidence("S3")
    answer = Answer(
        question="?",
        text="[1] [2]",
        citations=(Citation(1, second), Citation(2, first)),
        unknown_citations=(),
        sources=(first, second, third),
        analysis=QueryAnalysis(query="?"),
        found_nothing=False,
        prompt_tokens=None,
        context_may_be_truncated=False,
        timings=AnswerTimings(0.0, 0.0),
    )

    assert sources_as_displayed(answer) == [second, first, third]
