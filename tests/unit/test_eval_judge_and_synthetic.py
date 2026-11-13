import datetime as dt
import json
from collections.abc import Sequence

import pytest

from speech_qa.application.ingestion import IngestionService
from speech_qa.domain.models import ChatMessage, Chunk, Evidence, RetrievedChunk, SpeechInfo
from speech_qa.evaluation.judge import LLMJudge
from speech_qa.evaluation.synthetic import SyntheticQuestionGenerator, word_overlap
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex
from speech_qa.text.query_analysis import RuleBasedQueryAnalyzer
from speech_qa.text.sentences import SentenceWindowChunker
from tests.conftest import LONG_SPEECHES, FakeChatModel, FakeEmbedder

CHUNKER = SentenceWindowChunker(target_chars=1000, max_chars=1500, overlap_sentences=1)


def source() -> Evidence:
    info = SpeechInfo("F-001", "Lukáš Doležal", "R7", dt.date(2021, 4, 15))
    chunk = RetrievedChunk(Chunk("F-001", 0, "Kvalita rozhodnutí závisí na datech."), info, 0.9)
    return Evidence(info, (chunk,), 0.9)


# --------------------------------------------------------------------------- #
# judge
# --------------------------------------------------------------------------- #
def test_judge_returns_the_ratings() -> None:
    chat = FakeChatModel('{"faithfulness": 5, "relevance": 4, "reason": "ok", "extra": 1}')

    verdict = LLMJudge(chat).judge("Otázka?", "Odpověď [1].", [source()])

    assert verdict is not None
    assert (verdict.faithfulness, verdict.relevance, verdict.reason) == (5, 4, "ok")
    system, user = chat.calls[0]
    assert "faithfulness" in system.content
    assert "Question: Otázka?" in user.content
    assert '<source number="1" speaker="Lukáš Doležal"' in user.content
    assert user.content.endswith("Odpověď [1].")


def test_judge_without_sources_says_so() -> None:
    chat = FakeChatModel('{"faithfulness": 3, "relevance": 3}')

    LLMJudge(chat).judge("Otázka?", "Nevím.", [])

    assert "(no sources)" in chat.calls[0][1].content


@pytest.mark.parametrize(
    "reply", ["not json", '{"faithfulness": 7, "relevance": 3}', '{"relevance": 3}', "[1, 2]"]
)
def test_unusable_verdict_is_none(reply: str) -> None:
    assert LLMJudge(FakeChatModel(reply)).judge("Otázka?", "Odpověď.", [source()]) is None


# --------------------------------------------------------------------------- #
# synthetic questions
# --------------------------------------------------------------------------- #
@pytest.fixture
def ingested(memory_index: QdrantSpeechIndex, fake_embedder: FakeEmbedder) -> QdrantSpeechIndex:
    IngestionService(memory_index, fake_embedder, CHUNKER).ingest(
        read_speeches(LONG_SPEECHES).speeches
    )
    return memory_index


def generator(index: QdrantSpeechIndex, reply: object) -> SyntheticQuestionGenerator:
    manifest = index.read_manifest()
    assert manifest is not None
    analyzer = RuleBasedQueryAnalyzer(manifest.speakers, manifest.parties)
    return SyntheticQuestionGenerator(index, FakeChatModel(reply), analyzer)  # type: ignore[arg-type]


def question_for(messages: Sequence[ChatMessage]) -> str:
    """A well-behaved model: names the speaker only when asked to."""
    system, user = messages
    speaker = user.content.split("\n", 1)[0].removeprefix("Speaker: ")
    if "Mention the speaker" in system.content:
        surname = speaker.split()[-1]
        return json.dumps({"question": f"Jaký postoj zastává {surname} k tématu projevu?"})
    return json.dumps({"question": "Jaké argumenty zazněly k tomuto tématu?"})


def test_generates_named_and_topic_questions(ingested: QdrantSpeechIndex) -> None:
    result = generator(ingested, question_for).generate(6, seed=1)

    questions = result.questions
    assert [q.question_id for q in questions] == [f"syn-{n:04}" for n in range(1, 7)]
    assert all(q.kind == "synthetic" and len(q.relevant_ids) == 1 for q in questions)
    named, topic = questions[0::2], questions[1::2]
    assert all(len(q.expected_speakers) == 1 for q in named)
    assert all(q.expected_speakers == () for q in topic)
    assert all(q.note.startswith("chunk ") for q in questions)
    assert result.discarded == 0


def test_round_robin_covers_several_speakers(ingested: QdrantSpeechIndex) -> None:
    questions = generator(ingested, question_for).generate(8, seed=3).questions

    speakers = {q.expected_speakers[0] for q in questions[0::2]}
    sources = {q.relevant_ids[0] for q in questions}
    assert len(speakers) == 4
    assert len(sources) >= 7


def test_generation_is_reproducible(ingested: QdrantSpeechIndex) -> None:
    first = generator(ingested, question_for).generate(4, seed=7).questions
    second = generator(ingested, question_for).generate(4, seed=7).questions

    assert first == second


def test_unusable_questions_are_discarded(ingested: QdrantSpeechIndex) -> None:
    replies = iter(
        [
            "not json",  # unusable
            json.dumps({"question": ""}),  # empty
            json.dumps({"question": "Co si myslí o tom?"}),  # named question without a name
            json.dumps({"question": "Co říká Novotný o energetice?"}),  # unknown name
        ]
    )

    def reply(messages: Sequence[ChatMessage]) -> str:
        return next(replies, None) or question_for(messages)

    result = generator(ingested, reply).generate(2, seed=1)

    assert len(result.questions) == 2
    assert result.discarded == 4


def test_topic_question_naming_a_speaker_is_discarded(ingested: QdrantSpeechIndex) -> None:
    def reply(messages: Sequence[ChatMessage]) -> str:
        if "Mention the speaker" in messages[0].content:
            return question_for(messages)
        return json.dumps({"question": "Co si o tom myslí Doležal?"})

    result = generator(ingested, reply).generate(3, seed=1)

    assert len(result.questions) == 1  # every topic question names someone
    assert result.discarded >= 1


def test_questions_copying_the_chunk_are_discarded(ingested: QdrantSpeechIndex) -> None:
    def copying(messages: Sequence[ChatMessage]) -> str:
        excerpt = messages[1].content.split("Excerpt:\n", 1)[1]
        return json.dumps({"question": " ".join(excerpt.split()[:12]) + "?"})

    result = generator(ingested, copying).generate(2, seed=1)

    assert result.questions == ()
    assert result.discarded > 0


def test_word_overlap() -> None:
    overlap = word_overlap("Jak hodnotí kvalitu informací?", "kvalita informací je klíčová")
    assert overlap == pytest.approx(1 / 3)  # "informací" of hodnotí, kvalitu, informací
    assert word_overlap("Co?", "cokoli") == 1.0  # no content word in the question


def test_empty_index_gives_no_questions(memory_index: QdrantSpeechIndex) -> None:
    analyzer = RuleBasedQueryAnalyzer([], [])
    result = SyntheticQuestionGenerator(memory_index, FakeChatModel(), analyzer).generate(5)

    assert result.questions == ()
