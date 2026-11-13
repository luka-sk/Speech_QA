"""Text normalisation shared by name matching and query analysis."""

import re
import unicodedata
from dataclasses import dataclass

_WORD = re.compile(r"\w+(?:-\w+)*")
"""A word; hyphenated words such as ``Malá-Horáková`` stay together."""


def strip_diacritics(text: str) -> str:
    """Remove diacritics character by character, keeping the length of the text.

    Every character is replaced by its base letter (``ž`` → ``z``); characters
    without a single-character base form are kept, so positions in the result
    correspond to positions in the (NFC-normalised) input.
    """
    return "".join(_base_character(character) for character in unicodedata.normalize("NFC", text))


def _base_character(character: str) -> str:
    base = "".join(
        part for part in unicodedata.normalize("NFD", character) if not unicodedata.combining(part)
    )
    return base if len(base) == 1 else character


def fold(text: str) -> str:
    """Lower-case text without diacritics, used to compare words loosely."""
    return strip_diacritics(text).casefold()


@dataclass(frozen=True, slots=True)
class Token:
    """A word of a text and its position.

    Attributes:
        text: The word as written.
        start: Index of the first character.
        end: Index after the last character.
    """

    text: str
    start: int
    end: int

    @property
    def is_capitalized(self) -> bool:
        """Whether the word starts with an upper-case letter."""
        return self.text[0].isupper()


def tokenize(text: str) -> list[Token]:
    """Split text into words with their positions (punctuation is dropped)."""
    return [Token(match.group(), match.start(), match.end()) for match in _WORD.finditer(text)]
