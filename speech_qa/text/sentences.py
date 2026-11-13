"""Sentence splitting and sentence-window chunking for Czech text.

Splitting is rule-based: a sentence ends with ``.``, ``!``, ``?`` or ``…``
followed by whitespace and an upper-case letter or a digit. A full stop after a
common Czech abbreviation (``např.``, ``Ing.``, ``mld.``, ...) or after a single
capital letter (an initial) does not end a sentence. This is good enough for
chunking, where an occasional wrong boundary only shifts a window slightly.
"""

import re
from dataclasses import dataclass

from speech_qa.domain.models import Chunk, ChunkingParameters, Speech

_UPPER = "A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽÄÖÜ"

_BOUNDARY = re.compile(
    rf"(?<=[.!?…])([\"'“”»)\]]*)\s+(?=[\"'„“«(\[]?[{_UPPER}0-9])",
)
"""Whitespace after sentence-final punctuation (and closing quotes or brackets)."""

_LAST_TOKEN = re.compile(r"(\S+)$")

ABBREVIATIONS = frozenset(
    {
        # general
        "např", "tzv", "tj", "tzn", "resp", "mj", "vč", "popř", "příp", "zejm", "srov",
        "viz", "cca", "event", "př",
        # references and units
        "č", "čl", "odst", "písm", "s", "str", "sb", "kap", "obr", "tab", "roč", "r",
        "st", "stol", "tis", "mil", "mld", "min", "max", "hod", "ul", "nám", "tř",
        # titles and forms of address
        "ing", "mgr", "bc", "dr", "mudr", "judr", "phdr", "rndr", "paeddr", "doc", "prof",
        "csc", "p", "pí", "sl", "sv",
    }
)  # fmt: skip
"""Lower-case abbreviations (without the full stop) that do not end a sentence.

Abbreviations that usually end a sentence, such as ``atd.`` or ``apod.``, are
deliberately missing: before a lower-case word they never split anyway.
"""


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; whitespace inside sentences is normalised.

    Args:
        text: Text to split.

    Returns:
        Non-empty sentences in their original order.
    """
    normalized = " ".join(text.split())
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(normalized):
        end = match.end(1)
        if _ends_with_abbreviation(normalized[start:end]):
            continue
        sentences.append(normalized[start:end])
        start = match.end()
    if start < len(normalized):
        sentences.append(normalized[start:])
    return sentences


def _ends_with_abbreviation(candidate: str) -> bool:
    match = _LAST_TOKEN.search(candidate)
    if match is None:  # pragma: no cover - the regex always matches non-empty text
        return False
    token = match.group(1).rstrip("\"'“”»)]")
    if not token.endswith("."):
        return False  # "!", "?" and "…" always end a sentence
    word = token[:-1].lstrip("(\"'„“«[")
    is_initial = len(word) == 1 and word.isupper()
    return is_initial or word.lower() in ABBREVIATIONS


@dataclass(frozen=True, slots=True)
class SentenceWindowChunker:
    """Splits speeches into overlapping windows of whole sentences.

    A window grows sentence by sentence until it reaches ``target_chars``; it
    never exceeds ``max_chars`` unless a single sentence is longer, in which
    case that sentence forms a window on its own (sentences are never cut).
    Consecutive windows share ``overlap_sentences`` sentences, so a statement
    on a window boundary stays in context.
    """

    target_chars: int
    max_chars: int
    overlap_sentences: int

    def __post_init__(self) -> None:
        if not 0 < self.target_chars <= self.max_chars:
            msg = "require 0 < target_chars <= max_chars"
            raise ValueError(msg)
        if self.overlap_sentences < 0:
            msg = "overlap_sentences must not be negative"
            raise ValueError(msg)

    @property
    def parameters(self) -> ChunkingParameters:
        """The parameters, as stored in the index manifest."""
        return ChunkingParameters(
            target_chars=self.target_chars,
            max_chars=self.max_chars,
            overlap_sentences=self.overlap_sentences,
        )

    def split(self, speech: Speech) -> list[Chunk]:
        """Split a speech into chunks.

        Args:
            speech: The speech to split.

        Returns:
            Chunks numbered from zero, covering the whole text.
        """
        sentences = split_sentences(speech.text)
        windows: list[list[str]] = []
        first = 0
        while first < len(sentences):
            window = [sentences[first]]
            length = len(sentences[first])
            after = first + 1
            while after < len(sentences) and length < self.target_chars:
                extended = length + 1 + len(sentences[after])
                if extended > self.max_chars:
                    break
                window.append(sentences[after])
                length = extended
                after += 1
            windows.append(window)
            if after >= len(sentences):
                break
            # Step back by the overlap, but always move forward by at least one sentence.
            first = max(after - self.overlap_sentences, first + 1)
        return [
            Chunk(speech_id=speech.speech_id, index=index, text=" ".join(window))
            for index, window in enumerate(windows)
        ]
