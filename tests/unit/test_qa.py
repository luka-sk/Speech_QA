import datetime as dt

import pytest

from speech_qa.application.qa import AnswerSettings, SpeechQA, resolve_citations
from speech_qa.application.retrieval import Retriever
from speech_qa.domain.errors import EmptyQueryError, QueryTooLongError
from speech_qa.domain.models import (
    Answer,
    Chunk,
    Evidence,
    QueryAnalysis,
    RetrievalResult,
    RetrievedChunk,
    SpeechInfo,
)
from tests.conftest import FakeChatModel


def evidence(speech_id: str) -> Evidence:
    info = SpeechInfo(speech_id=speech_id, speaker="A B", party=None, date=dt.date(2020, 1, 1))
    chunk = RetrievedChunk(chunk=Chunk(speech_id, 0, f"Text {speech_id}."), speech=info, score=0.5)
    return Evidence(speech=info, passages=(chunk,), score=0.5)


SOURCES = (evidence("S1"), evidence("S2"), evidence("S3"))


class StubRetriever(Retriever):
    """Returns a fixed result and records the queries."""

    def __init__(self, found: tuple[Evidence, ...] = SOURCES) -> None:
        self.found = found
        self.queries: list[str] = []

    def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        return RetrievalResult(analysis=QueryAnalysis(query=query), evidence=self.found)


def make_qa(
    chat: FakeChatModel,
    retriever: StubRetriever | None = None,
    settings: AnswerSettings | None = None,
) -> SpeechQA:
    ticks = iter([0.0, 0.5, 3.0])
    return SpeechQA(
        retriever or StubRetriever(),
        chat,
        lambda answer: f"FORMATTED: {answer.text}",
        settings,
        timer=lambda: next(ticks),
    )


def test_answer_cites_the_given_sources() -> None:
    chat = FakeChatModel("Doležal říká X [2], jinde Y [1][2].")

    answer = make_qa(chat).ask("  Co říká Doležal?  ")

    assert answer.question == "Co říká Doležal?"
    assert answer.text == "Doležal říká X [1], jinde Y [2][1]."
    assert [(c.number, c.evidence.speech.speech_id) for c in answer.citations] == [
        (1, "S2"),
        (2, "S1"),
    ]
    assert answer.unknown_citations == ()
    assert answer.sources == SOURCES
    assert not answer.found_nothing
    assert answer.timings.retrieval_seconds == 0.5
    assert answer.timings.generation_seconds == 2.5


def test_question_is_passed_to_retrieval_and_the_model() -> None:
    chat = FakeChatModel()
    retriever = StubRetriever()

    make_qa(chat, retriever, AnswerSettings(max_answer_tokens=321)).ask("Otázka?")

    assert retriever.queries == ["Otázka?"]
    assert "Question: Otázka?" in chat.calls[0][1].content
    assert chat.max_tokens == [321]


def test_nothing_found_does_not_ask_the_model() -> None:
    chat = FakeChatModel()

    answer = make_qa(chat, StubRetriever(found=())).ask("Co se říkalo v roce 2030?")

    assert answer.found_nothing
    assert answer.text == ""
    assert answer.citations == ()
    assert chat.calls == []
    assert answer.timings.generation_seconds == 0.0


def test_prompt_close_to_the_context_window_is_flagged() -> None:
    settings = AnswerSettings(context_window_tokens=4096)

    near = make_qa(FakeChatModel(prompt_tokens=3900), settings=settings).ask("Otázka?")
    below = make_qa(FakeChatModel(prompt_tokens=3800), settings=settings).ask("Otázka?")
    unknown = make_qa(FakeChatModel(prompt_tokens=None), settings=settings).ask("Otázka?")

    assert near.context_may_be_truncated
    assert near.prompt_tokens == 3900
    assert not below.context_may_be_truncated
    assert not unknown.context_may_be_truncated


def test_sources_are_limited_by_the_prompt_budget() -> None:
    settings = AnswerSettings(context_window_tokens=1400, max_answer_tokens=800, reserve_tokens=100)
    big = tuple(
        Evidence(
            speech=source.speech,
            passages=(
                RetrievedChunk(
                    chunk=Chunk(source.speech.speech_id, 0, "slovo " * 150),
                    speech=source.speech,
                    score=0.5,
                ),
            ),
            score=0.5,
        )
        for source in SOURCES
    )

    answer = make_qa(FakeChatModel(), StubRetriever(big), settings).ask("Otázka?")

    assert settings.prompt_budget_tokens == 500
    assert len(answer.sources) == 1


def test_answer_returns_the_formatted_text() -> None:
    assert make_qa(FakeChatModel("Ano [1].")).answer("Otázka?") == "FORMATTED: Ano [1]."


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_question_is_rejected(query: str) -> None:
    with pytest.raises(EmptyQueryError):
        make_qa(FakeChatModel()).ask(query)


def test_too_long_question_is_rejected() -> None:
    with pytest.raises(QueryTooLongError) as caught:
        make_qa(FakeChatModel(), settings=AnswerSettings(max_query_chars=10)).ask("x" * 11)

    assert (caught.value.length, caught.value.limit) == (11, 10)


@pytest.mark.parametrize(
    ("text", "expected_text", "cited", "unknown"),
    [
        ("Bez citací.", "Bez citací.", [], []),
        ("A [3] B [1, 3] C [2;3]", "A [1] B [2, 1] C [3, 1]", ["S3", "S1", "S2"], []),
        ("A [5] B [1] C [0] D [5]", "A [5] B [1] C [0] D [5]", ["S1"], ["5", "0"]),
        ("Rok [2020] a [sic]", "Rok [2020] a [sic]", [], ["2020"]),
    ],
)
def test_resolve_citations(
    text: str, expected_text: str, cited: list[str], unknown: list[str]
) -> None:
    renumbered, citations, unknown_numbers = resolve_citations(text, SOURCES)

    assert renumbered == expected_text
    assert [citation.evidence.speech.speech_id for citation in citations] == cited
    assert [citation.number for citation in citations] == list(range(1, len(cited) + 1))
    assert list(unknown_numbers) == unknown


def test_answer_is_a_value_object() -> None:
    answer = make_qa(FakeChatModel()).ask("Otázka?")

    assert isinstance(answer, Answer)
    assert answer.analysis.query == "Otázka?"
