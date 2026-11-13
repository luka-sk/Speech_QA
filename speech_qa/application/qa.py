"""Use case: answer a question from the speech archive (the ``PQA`` interface).

1. Validate the question.
2. Retrieve the relevant speeches (see :mod:`speech_qa.application.retrieval`).
3. Without any speech, return a fixed "nothing found" answer; the language
   model is not asked, so it cannot invent an answer.
4. Otherwise build the prompt with numbered sources and ask the model.
5. Map the cited source numbers back to the speeches, renumber them in the
   order of their first citation and flag numbers the model was not given.
"""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from speech_qa.application.pqa import PQA
from speech_qa.application.prompting import build_prompt
from speech_qa.application.retrieval import Retriever
from speech_qa.domain.errors import EmptyQueryError, QueryTooLongError
from speech_qa.domain.models import Answer, AnswerTimings, Citation, Evidence
from speech_qa.domain.ports import ChatModel

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\s*\d+\s*(?:[,;]\s*\d+\s*)*)\]")
"""A citation such as ``[1]`` or ``[1, 3]``."""

TRUNCATION_WARNING_RATIO = 0.95
"""A prompt above this share of the context window may have been cut."""


@dataclass(frozen=True, slots=True)
class AnswerSettings:
    """Limits of answering.

    Attributes:
        max_query_chars: Longer questions are rejected.
        context_window_tokens: Context window of the language model.
        max_answer_tokens: Upper limit for the length of an answer.
        reserve_tokens: Safety margin kept free in the context window.
    """

    max_query_chars: int = 1000
    context_window_tokens: int = 4096
    max_answer_tokens: int = 800
    reserve_tokens: int = 100

    @property
    def prompt_budget_tokens(self) -> int:
        """Tokens available for the prompt."""
        return self.context_window_tokens - self.max_answer_tokens - self.reserve_tokens


class SpeechQA(PQA):
    """Answers questions with references to the source speeches."""

    def __init__(
        self,
        retriever: Retriever,
        chat_model: ChatModel,
        formatter: Callable[[Answer], str],
        settings: AnswerSettings | None = None,
        *,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Create the question-answering service.

        Args:
            retriever: Finds the relevant speeches.
            chat_model: Writes the answer.
            formatter: Turns an :class:`Answer` into the text returned by :meth:`answer`.
            settings: Limits of answering.
            timer: Clock for the step durations (injectable for tests).
        """
        self._retriever = retriever
        self._chat_model = chat_model
        self._formatter = formatter
        self._settings = settings or AnswerSettings()
        self._timer = timer

    def answer(self, query: str) -> str:
        """Answer a question; the text includes references to the source speeches."""
        return self._formatter(self.ask(query))

    def ask(self, query: str) -> Answer:
        """Answer a question and return the structured result.

        Raises:
            EmptyQueryError: If the question is empty.
            QueryTooLongError: If the question is longer than allowed.
        """
        question = query.strip()
        if not question:
            raise EmptyQueryError
        if len(question) > self._settings.max_query_chars:
            raise QueryTooLongError(len(question), self._settings.max_query_chars)

        started = self._timer()
        retrieval = self._retriever.retrieve(question)
        retrieved = self._timer()
        if not retrieval.evidence:
            return Answer(
                question=question,
                text="",
                citations=(),
                unknown_citations=(),
                sources=(),
                analysis=retrieval.analysis,
                found_nothing=True,
                prompt_tokens=None,
                context_may_be_truncated=False,
                timings=AnswerTimings(retrieved - started, 0.0),
            )

        prompt = build_prompt(
            question,
            retrieval.analysis,
            retrieval.evidence,
            budget_tokens=self._settings.prompt_budget_tokens,
        )
        if prompt.omitted:
            logger.info("%d sources left out to fit the context window", prompt.omitted)
        completion = self._chat_model.complete(
            prompt.messages, max_tokens=self._settings.max_answer_tokens
        )
        generated = self._timer()
        text, citations, unknown = resolve_citations(completion.text, prompt.sources)
        truncated = (
            completion.prompt_tokens is not None
            and completion.prompt_tokens
            >= TRUNCATION_WARNING_RATIO * self._settings.context_window_tokens
        )
        logger.info(
            "answered in %.1f s (retrieval %.1f s, generation %.1f s), prompt tokens %s",
            generated - started,
            retrieved - started,
            generated - retrieved,
            completion.prompt_tokens,
        )
        return Answer(
            question=question,
            text=text,
            citations=citations,
            unknown_citations=unknown,
            sources=prompt.sources,
            analysis=retrieval.analysis,
            found_nothing=False,
            prompt_tokens=completion.prompt_tokens,
            context_may_be_truncated=truncated,
            timings=AnswerTimings(retrieved - started, generated - retrieved),
        )


def resolve_citations(
    text: str, sources: tuple[Evidence, ...]
) -> tuple[str, tuple[Citation, ...], tuple[str, ...]]:
    """Map source numbers in the model's answer to speeches.

    Cited sources are renumbered 1, 2, ... in the order of their first citation,
    so the list of sources under the answer has no gaps. Numbers the model was
    not given are kept as written and reported.

    Returns:
        The answer with renumbered citations, the citations and the unknown numbers.
    """
    display_numbers: dict[int, int] = {}
    unknown: list[str] = []

    def renumber(match: re.Match[str]) -> str:
        numbers = []
        for raw in re.split(r"[,;]", match.group(1)):
            label = int(raw)
            if 1 <= label <= len(sources):
                numbers.append(str(display_numbers.setdefault(label, len(display_numbers) + 1)))
            else:
                numbers.append(str(label))
                if str(label) not in unknown:
                    unknown.append(str(label))
        return "[" + ", ".join(numbers) + "]"

    renumbered = _CITATION.sub(renumber, text)
    citations = tuple(
        Citation(number=display, evidence=sources[label - 1])
        for label, display in sorted(display_numbers.items(), key=lambda item: item[1])
    )
    return renumbered, citations, tuple(unknown)
