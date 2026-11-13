"""Running an evaluation: every question through retrieval, a selection through answering.

* **Retrieval** is measured for every question: ranking metrics against the
  relevant speeches (on the speeches that would be passed to the model, so
  ``k`` is the configured maximum number of sources) and the recognition of
  speakers in the question.
* **Answering** (the whole pipeline with the language model) is measured for
  every question of a manual set and for a random sample of the synthetic set:
  latency, citation validity and precision, correct abstention and, unless
  disabled, the LLM-as-judge ratings.
"""

import datetime as dt
import random
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

from speech_qa.application.qa import SpeechQA
from speech_qa.application.retrieval import Retriever
from speech_qa.domain.models import Answer, Evidence
from speech_qa.evaluation import metrics
from speech_qa.evaluation.dataset import EvalQuestion, EvalSet
from speech_qa.evaluation.judge import JudgeVerdict, LLMJudge

SYNTHETIC_SET = "synthetic"
"""Name of the set whose answers are measured only on a sample."""


@dataclass(frozen=True, slots=True)
class AnswerOutcome:
    """What happened when a question was answered.

    Attributes:
        text: The answer.
        cited_ids: Speeches cited by the answer.
        unknown_citations: Source numbers cited by the model that it was not given.
        found_nothing: No speech matched; the model was not asked.
        retrieval_seconds: Duration of retrieval.
        generation_seconds: Duration of generation.
        prompt_tokens: Prompt size reported by the model server.
        verdict: Ratings of the judge; ``None`` if not judged or not understood.
        judged: Whether the judge was asked.
    """

    text: str
    cited_ids: tuple[str, ...]
    unknown_citations: tuple[str, ...]
    found_nothing: bool
    retrieval_seconds: float
    generation_seconds: float
    prompt_tokens: int | None
    verdict: JudgeVerdict | None
    judged: bool

    @property
    def abstained(self) -> bool:
        """Whether the answer cites no speech (says the archive has no answer)."""
        return not self.cited_ids

    @property
    def total_seconds(self) -> float:
        """Duration of the whole answer."""
        return self.retrieval_seconds + self.generation_seconds


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Result for one question.

    Attributes:
        set_name: The evaluation set of the question.
        question: The question.
        retrieved_ids: Retrieved speeches, best first.
        detected_speakers: Speakers recognised in the question.
        retrieval_seconds: Duration of retrieval.
        answer: Outcome of answering, if the question was answered.
    """

    set_name: str
    question: EvalQuestion
    retrieved_ids: tuple[str, ...]
    detected_speakers: tuple[str, ...]
    retrieval_seconds: float
    answer: AnswerOutcome | None

    def reciprocal_rank(self) -> float | None:
        """Reciprocal rank of the first relevant speech (``None`` if not answerable)."""
        if not self.question.answerable:
            return None
        return metrics.reciprocal_rank(self.retrieved_ids, self.question.relevant_ids)


@dataclass(frozen=True, slots=True)
class SetSummary:
    """Aggregated metrics of one evaluation set; ``None`` where undefined."""

    name: str
    questions: int
    answerable: int
    answered: int
    hit_at_1: float | None
    hit_at_5: float | None
    recall_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None
    speaker_precision: float | None
    speaker_recall: float | None
    citation_validity: float | None
    citation_precision: float | None
    correct_abstention: float | None
    false_abstention: float | None
    faithfulness: float | None
    relevance: float | None
    judge_failures: int
    latency_p50: float | None
    latency_p95: float | None
    retrieval_p50: float | None
    generation_p50: float | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Everything an evaluation run produced."""

    started_at: dt.datetime
    duration_seconds: float
    k: int
    results: tuple[QuestionResult, ...]
    summaries: tuple[SetSummary, ...]
    missing_ids: dict[str, tuple[str, ...]]


class EvaluationRunner:
    """Runs evaluation sets through the system."""

    def __init__(
        self,
        retriever: Retriever,
        qa: SpeechQA,
        judge: LLMJudge | None,
        *,
        k: int,
        known_ids: Collection[str],
        clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Create the runner.

        Args:
            retriever: Finds the relevant speeches (measured on its own).
            qa: Answers the questions (the whole pipeline).
            judge: Rates the answers; ``None`` skips rating.
            k: Number of retrieved speeches the ranking metrics look at.
            known_ids: Ids of the indexed speeches, to report unknown relevant ids.
            clock: Source of the start time (injectable for tests).
            timer: Source of durations (injectable for tests).
        """
        self._retriever = retriever
        self._qa = qa
        self._judge = judge
        self._k = k
        self._known_ids = set(known_ids)
        self._clock = clock
        self._timer = timer

    def run(
        self,
        sets: Sequence[EvalSet],
        *,
        e2e_sample: int = 20,
        retrieval_only: bool = False,
        seed: int = 42,
    ) -> EvaluationResult:
        """Evaluate the sets.

        Args:
            sets: Evaluation sets.
            e2e_sample: How many synthetic questions are answered by the whole pipeline.
            retrieval_only: Measure retrieval only (no language model).
            seed: Seed for sampling the synthetic questions.
        """
        started_at, started = self._clock(), self._timer()
        results: list[QuestionResult] = []
        for eval_set in sets:
            answered = set() if retrieval_only else _answered_ids(eval_set, e2e_sample, seed)
            for question in eval_set.questions:
                results.append(
                    self._evaluate(eval_set.name, question, question.question_id in answered)
                )
        return EvaluationResult(
            started_at=started_at,
            duration_seconds=self._timer() - started,
            k=self._k,
            results=tuple(results),
            summaries=tuple(
                summarize(
                    eval_set.name, [r for r in results if r.set_name == eval_set.name], self._k
                )
                for eval_set in sets
            ),
            missing_ids={
                eval_set.name: tuple(
                    sorted(
                        {
                            speech_id
                            for question in eval_set.questions
                            for speech_id in question.relevant_ids
                            if speech_id not in self._known_ids
                        }
                    )
                )
                for eval_set in sets
            },
        )

    def _evaluate(self, set_name: str, question: EvalQuestion, answer: bool) -> QuestionResult:
        started = self._timer()
        retrieval = self._retriever.retrieve(question.question)
        retrieval_seconds = self._timer() - started
        return QuestionResult(
            set_name=set_name,
            question=question,
            retrieved_ids=tuple(item.speech.speech_id for item in retrieval.evidence),
            detected_speakers=tuple(sorted(retrieval.analysis.search_filter.speakers)),
            retrieval_seconds=retrieval_seconds,
            answer=self._answer(question) if answer else None,
        )

    def _answer(self, question: EvalQuestion) -> AnswerOutcome:
        answer = self._qa.ask(question.question)
        verdict = None
        judged = self._judge is not None and not answer.found_nothing
        if self._judge is not None and judged:
            verdict = self._judge.judge(answer.question, answer.text, sources_as_displayed(answer))
        return AnswerOutcome(
            text=answer.text,
            cited_ids=tuple(citation.evidence.speech.speech_id for citation in answer.citations),
            unknown_citations=answer.unknown_citations,
            found_nothing=answer.found_nothing,
            retrieval_seconds=answer.timings.retrieval_seconds,
            generation_seconds=answer.timings.generation_seconds,
            prompt_tokens=answer.prompt_tokens,
            verdict=verdict,
            judged=judged,
        )


def sources_as_displayed(answer: Answer) -> list[Evidence]:
    """Sources numbered as in the displayed answer: cited ones first, in citation order.

    The answer text uses the renumbered citations ([1] is the first cited source),
    so a reader of the text, such as the judge, must get the sources in that order.
    Sources that were given to the model but not cited follow.
    """
    cited = [citation.evidence for citation in answer.citations]
    return cited + [source for source in answer.sources if source not in cited]


def _answered_ids(eval_set: EvalSet, sample: int, seed: int) -> set[str]:
    ids = [question.question_id for question in eval_set.questions]
    if eval_set.name != SYNTHETIC_SET or sample >= len(ids):
        return set(ids)
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    return set(rng.sample(ids, max(sample, 0)))


def summarize(name: str, results: Sequence[QuestionResult], k: int) -> SetSummary:
    """Aggregate the results of one set."""
    answerable = [r for r in results if r.question.answerable]
    answered = [r for r in results if r.answer is not None]
    outcomes = [r.answer for r in answered if r.answer is not None]

    def over_answerable(metric: Callable[[QuestionResult], float]) -> float | None:
        return metrics.mean(metric(r) for r in answerable)

    precision, recall = metrics.micro_precision_recall(
        (r.detected_speakers, r.question.expected_speakers) for r in results
    )
    cited_answerable = [
        (r.question, r.answer)
        for r in answered
        if r.answer is not None and r.question.answerable and r.answer.cited_ids
    ]
    verdicts = [outcome.verdict for outcome in outcomes if outcome.verdict is not None]
    totals = [outcome.total_seconds for outcome in outcomes]
    return SetSummary(
        name=name,
        questions=len(results),
        answerable=len(answerable),
        answered=len(answered),
        hit_at_1=over_answerable(
            lambda r: metrics.hit_at_k(r.retrieved_ids, r.question.relevant_ids, 1)
        ),
        hit_at_5=over_answerable(
            lambda r: metrics.hit_at_k(r.retrieved_ids, r.question.relevant_ids, 5)
        ),
        recall_at_k=over_answerable(
            lambda r: metrics.recall_at_k(r.retrieved_ids, r.question.relevant_ids, k)
        ),
        mrr=over_answerable(
            lambda r: metrics.reciprocal_rank(r.retrieved_ids, r.question.relevant_ids)
        ),
        ndcg_at_k=over_answerable(
            lambda r: metrics.ndcg_at_k(r.retrieved_ids, r.question.relevant_ids, k)
        ),
        speaker_precision=precision,
        speaker_recall=recall,
        citation_validity=metrics.mean(
            0.0 if outcome.unknown_citations else 1.0 for outcome in outcomes
        ),
        citation_precision=metrics.mean(
            len(set(outcome.cited_ids) & set(question.relevant_ids)) / len(set(outcome.cited_ids))
            for question, outcome in cited_answerable
        ),
        correct_abstention=metrics.mean(
            1.0 if r.answer.abstained else 0.0
            for r in answered
            if r.answer is not None and not r.question.answerable
        ),
        false_abstention=metrics.mean(
            1.0 if r.answer.abstained else 0.0
            for r in answered
            if r.answer is not None and r.question.answerable
        ),
        faithfulness=metrics.mean(verdict.faithfulness for verdict in verdicts),
        relevance=metrics.mean(verdict.relevance for verdict in verdicts),
        judge_failures=sum(1 for o in outcomes if o.judged and o.verdict is None),
        latency_p50=metrics.percentile(totals, 0.5),
        latency_p95=metrics.percentile(totals, 0.95),
        retrieval_p50=metrics.percentile((o.retrieval_seconds for o in outcomes), 0.5),
        generation_p50=metrics.percentile((o.generation_seconds for o in outcomes), 0.5),
    )
