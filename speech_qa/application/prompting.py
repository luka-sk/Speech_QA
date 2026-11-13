"""Building the prompt for answering a question from retrieved speeches.

Sources are numbered ``[1]``, ``[2]``, ... instead of being labelled with their
ids: real ids are long (``ParlaMint-CZ_2020-11-13-...``) and a language model
copies short numbers far more reliably. The numbers are mapped back to the
speeches when the answer is processed.

Speech texts are untrusted data. They are put inside ``<source>`` elements with
``<``, ``>`` and ``&`` escaped, so a speech cannot close the element or pose as
an instruction; the system prompt tells the model to treat them as quotations.

The prompt is kept within a token budget (the context window minus the answer
length). Tokens are estimated as characters / 3, a slightly pessimistic value
for Czech (about 3.2 characters per token were measured with gemma4). Sources
are added from the most relevant one until the budget is used up.
"""

import html
import math
from dataclasses import dataclass
from importlib import resources
from string import Template

from speech_qa.domain.models import ChatMessage, Evidence, QueryAnalysis

CHARS_PER_TOKEN = 3.0
"""Pessimistic estimate of characters per token for Czech text."""

PASSAGE_SEPARATOR = "\n[…]\n"
"""Placed between two passages of one speech."""

_PROMPTS = resources.files("speech_qa.prompts")
_SYSTEM_PROMPT = _PROMPTS.joinpath("answer_system.txt").read_text(encoding="utf-8").strip()
_USER_TEMPLATE = Template(_PROMPTS.joinpath("answer_user.txt").read_text(encoding="utf-8").strip())


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens of a text."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class Prompt:
    """A prompt ready to be sent to the language model.

    Attributes:
        messages: System and user message.
        sources: Speeches included as sources; source ``[n]`` is ``sources[n - 1]``.
        omitted: Number of retrieved speeches left out to stay within the budget.
        estimated_tokens: Estimated size of the prompt.
    """

    messages: tuple[ChatMessage, ...]
    sources: tuple[Evidence, ...]
    omitted: int
    estimated_tokens: int


def build_prompt(
    question: str, analysis: QueryAnalysis, evidence: tuple[Evidence, ...], *, budget_tokens: int
) -> Prompt:
    """Build the messages for answering a question.

    Args:
        question: The user's question.
        analysis: How the question was understood (for the search note).
        evidence: Retrieved speeches, best first; must not be empty.
        budget_tokens: Maximum estimated size of the prompt.

    Returns:
        The prompt with as many sources as fit into the budget (at least one,
        shortened if necessary).
    """
    if not evidence:
        msg = "at least one source is required"
        raise ValueError(msg)
    note = search_note(analysis)
    frame_tokens = estimate_tokens(_SYSTEM_PROMPT) + estimate_tokens(
        _USER_TEMPLATE.substitute(
            question=question, search_note=note, source_count=len(evidence), sources=""
        )
    )
    blocks: list[str] = []
    used = frame_tokens
    for number, speech in enumerate(evidence, start=1):
        block = render_source(number, speech)
        cost = estimate_tokens(block) + 1
        if used + cost > budget_tokens:
            if blocks:
                break
            # Even the best source does not fit: include a shortened version of it.
            allowed_chars = max(int((budget_tokens - used) * CHARS_PER_TOKEN), 200)
            block = render_source(number, speech, max_chars=allowed_chars)
            cost = estimate_tokens(block) + 1
        blocks.append(block)
        used += cost
    user_message = _USER_TEMPLATE.substitute(
        question=question,
        search_note=note,
        source_count=len(blocks),
        sources="\n\n".join(blocks),
    )
    return Prompt(
        messages=(
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_message),
        ),
        sources=evidence[: len(blocks)],
        omitted=len(evidence) - len(blocks),
        estimated_tokens=used,
    )


def render_source(number: int, speech: Evidence, *, max_chars: int | None = None) -> str:
    """Render one source as an escaped ``<source>`` element."""
    info = speech.speech
    text = PASSAGE_SEPARATOR.join(passage.chunk.text for passage in speech.passages)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " […]"
    attributes = {
        "number": str(number),
        "speaker": info.speaker,
        "party": info.party or "no party",
        "date": info.date.isoformat(),
    }
    rendered = " ".join(f'{name}="{html.escape(value)}"' for name, value in attributes.items())
    return f"<source {rendered}>\n{html.escape(text, quote=False)}\n</source>"


def search_note(analysis: QueryAnalysis) -> str:
    """Explain to the model which part of the archive was searched."""
    search_filter = analysis.search_filter
    if search_filter.is_empty:
        return "All speeches in the archive were searched."
    notes = []
    for mention in analysis.speaker_mentions:
        if len(mention.speakers) > 1:
            notes.append(
                f'The name "{mention.surface}" in the question matches {len(mention.speakers)} '
                f"different speakers in the archive: {', '.join(mention.speakers)}."
            )
    if search_filter.speakers:
        notes.append(f"Only speeches by {', '.join(sorted(search_filter.speakers))} were searched.")
    if search_filter.parties:
        notes.append(
            f"Only speeches by members of {', '.join(sorted(search_filter.parties))} were searched."
        )
    if search_filter.date_from or search_filter.date_to:
        start = search_filter.date_from.isoformat() if search_filter.date_from else "the beginning"
        end = search_filter.date_to.isoformat() if search_filter.date_to else "today"
        notes.append(f"Only speeches given between {start} and {end} were searched.")
    return " ".join(notes)
