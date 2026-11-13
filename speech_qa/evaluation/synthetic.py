"""Synthetic evaluation questions generated from the index by the language model.

For a random chunk the model writes one Czech question that the chunk answers;
the speech of the chunk is the relevant answer. Chunks are drawn in turn from
all speakers (round-robin over a shuffled order, with a fixed seed), so the set
covers the archive evenly and is reproducible.

Half of the questions name the speaker of the chunk, the other half ask only
about the topic. A question is discarded and another chunk is used when:

* the model's reply is not a usable JSON question,
* a "named" question does not mention the speaker, or a "topic" question
  mentions any speaker (checked with the same rule-based analyser the system uses),
* the question copies too many words from the chunk (it would be too easy).

Synthetic questions resemble the source text more than real questions do, so
results on them are optimistic; each question has a single relevant speech even
if other speeches touch the same topic, so recall is pessimistic.
"""

import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from string import Template

from speech_qa.domain.models import ChatMessage, StoredChunk
from speech_qa.domain.ports import ChatModel, QueryAnalyzer, SpeechIndex
from speech_qa.evaluation.dataset import EvalQuestion
from speech_qa.text.normalize import fold

logger = logging.getLogger(__name__)

_PROMPTS = resources.files("speech_qa.prompts")
_SYSTEM_TEMPLATE = Template(
    _PROMPTS.joinpath("synthetic_system.txt").read_text(encoding="utf-8").strip()
)
_USER_TEMPLATE = Template(
    _PROMPTS.joinpath("synthetic_user.txt").read_text(encoding="utf-8").strip()
)
_NAMED_RULE = (
    "Mention the speaker by surname ($surname) in the grammatical case the sentence needs."
)
_TOPIC_RULE = "Do not mention any person or party; ask only about the topic."

MIN_CONTENT_WORD = 4
"""Words shorter than this are ignored when measuring copied words."""


@dataclass(frozen=True, slots=True)
class SyntheticResult:
    """Generated questions and how many attempts were discarded."""

    questions: tuple[EvalQuestion, ...]
    discarded: int


def word_overlap(question: str, text: str) -> float:
    """Share of the question's content words that also occur in the text."""
    question_words = {
        word for word in re.findall(r"\w+", fold(question)) if len(word) >= MIN_CONTENT_WORD
    }
    if not question_words:
        return 1.0
    text_words = set(re.findall(r"\w+", fold(text)))
    return len(question_words & text_words) / len(question_words)


class SyntheticQuestionGenerator:
    """Generates evaluation questions from the chunks of an index."""

    def __init__(
        self,
        index: SpeechIndex,
        chat_model: ChatModel,
        analyzer: QueryAnalyzer,
        *,
        max_overlap: float = 0.6,
    ) -> None:
        """Create the generator.

        Args:
            index: Source of the chunks.
            chat_model: Writes the questions.
            analyzer: Recognises speakers in the generated questions.
            max_overlap: Questions sharing more content words with the chunk are discarded.
        """
        self._index = index
        self._chat_model = chat_model
        self._analyzer = analyzer
        self._max_overlap = max_overlap

    def generate(self, count: int, *, seed: int = 42) -> SyntheticResult:
        """Generate up to ``count`` questions (fewer if the index runs out of chunks)."""
        rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
        questions: list[EvalQuestion] = []
        discarded = 0
        for chunk in _round_robin_by_speaker(list(self._index.iter_chunks()), rng):
            if len(questions) == count:
                break
            named = len(questions) % 2 == 0
            question = self._ask(chunk, named=named)
            if question is None:
                discarded += 1
                continue
            questions.append(
                EvalQuestion(
                    question_id=f"syn-{len(questions) + 1:04}",
                    question=question,
                    relevant_ids=(chunk.speech.speech_id,),
                    expected_speakers=(chunk.speech.speaker,) if named else (),
                    kind="synthetic",
                    note=f"chunk {chunk.chunk.index} of {chunk.speech.speech_id}",
                )
            )
            logger.info("generated %d/%d questions", len(questions), count)
        return SyntheticResult(questions=tuple(questions), discarded=discarded)

    def _ask(self, chunk: StoredChunk, *, named: bool) -> str | None:
        speaker = chunk.speech.speaker
        rule = (
            Template(_NAMED_RULE).substitute(surname=speaker.split()[-1]) if named else _TOPIC_RULE
        )
        messages = (
            ChatMessage("system", _SYSTEM_TEMPLATE.substitute(speaker_rule=rule)),
            ChatMessage(
                "user",
                _USER_TEMPLATE.substitute(
                    speaker=speaker, date=chunk.speech.date.isoformat(), excerpt=chunk.chunk.text
                ),
            ),
        )
        reply = self._chat_model.complete(messages, max_tokens=200, json_output=True).text
        try:
            question = str(json.loads(reply)["question"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.info("discarded an unusable reply: %r", reply[:100])
            return None
        if not question:
            return None
        recognised = self._analyzer.analyze(question).search_filter.speakers
        if (named and speaker not in recognised) or (not named and recognised):
            logger.info("discarded (speaker mention does not match): %r", question)
            return None
        if word_overlap(question, chunk.chunk.text) > self._max_overlap:
            logger.info("discarded (copies the chunk): %r", question)
            return None
        return question


def _round_robin_by_speaker(chunks: list[StoredChunk], rng: random.Random) -> list[StoredChunk]:
    """Order chunks so that speakers take turns; both levels are shuffled."""
    by_speaker: dict[str, list[StoredChunk]] = defaultdict(list)
    for chunk in sorted(chunks, key=lambda item: (item.speech.speech_id, item.chunk.index)):
        by_speaker[chunk.speech.speaker].append(chunk)
    queues = [by_speaker[speaker] for speaker in sorted(by_speaker)]
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)
    ordered: list[StoredChunk] = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop())
    return ordered
