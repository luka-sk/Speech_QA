"""Rule-based query analysis: speakers, parties and years become a search filter.

* **Speakers** are recognised by the inflected forms of their surnames (see
  :mod:`speech_qa.text.czech_names`), compared without diacritics. Only words
  starting with a capital letter count, so common words such as "svoboda" or
  "král" never become a filter. A neighbouring first name narrows the match
  ("Lukáše Doležala" → only Lukáš); otherwise every speaker with that surname
  is kept, and the answer has to say that the name is ambiguous.
* **Parties** are recognised by their exact code from the data, as a whole
  word, case-sensitively but ignoring diacritics ("ANO" matches, "ano" not).
* **Years** are recognised with a context word or as a range ("v roce 2020",
  "od 2019 do 2021", "2019-2021" with any dash, "po roce 2020"), so that numbers such as
  "2000 bytů" do not become a filter.

Recognised mentions are removed from the text used for the semantic search,
because the filter already takes care of them.
"""

import datetime as dt
import re
import unicodedata
from collections.abc import Iterable

from speech_qa.domain.models import QueryAnalysis, SearchFilter, SpeakerMention
from speech_qa.domain.ports import QueryAnalyzer
from speech_qa.text.czech_names import first_name_stems, matches_first_name, surname_forms
from speech_qa.text.normalize import fold, strip_diacritics, tokenize

type _Span = tuple[int, int]

_YEAR = r"((?:19|20)\d{2})"
_DASH = r"\s*[-\u2013\u2014]\s*"
"""Hyphen, en dash or em dash between two years."""

# Matched against the query without diacritics, so "před" is written "pred".
_RANGE_PATTERNS = (
    re.compile(rf"\bod\s+(?:roku\s+)?{_YEAR}\s+do\s+(?:roku\s+)?{_YEAR}\b", re.IGNORECASE),
    re.compile(rf"\bmezi\s+lety\s+{_YEAR}\s+a\s+{_YEAR}\b", re.IGNORECASE),
    re.compile(rf"\bv\s+letech\s+{_YEAR}(?:{_DASH}|\s+(?:az|a)\s+){_YEAR}\b", re.IGNORECASE),
    re.compile(rf"\b{_YEAR}{_DASH}{_YEAR}\b"),
)
_FROM_PATTERNS = (  # (pattern, years to add to the matched year)
    (re.compile(rf"\bod\s+(?:roku\s+)?{_YEAR}\b", re.IGNORECASE), 0),
    (re.compile(rf"\bpo\s+roce\s+{_YEAR}\b", re.IGNORECASE), 1),
)
_TO_PATTERNS = (
    (re.compile(rf"\bdo\s+(?:roku\s+)?{_YEAR}\b", re.IGNORECASE), 0),
    (re.compile(rf"\bpred\s+rokem\s+{_YEAR}\b", re.IGNORECASE), -1),
)
_SINGLE_YEAR_PATTERN = re.compile(rf"\b(?:v\s+roce|roku|rok|v\s+r\.)\s+{_YEAR}\b", re.IGNORECASE)


class RuleBasedQueryAnalyzer(QueryAnalyzer):
    """Builds a search filter from names, party codes and years in a query."""

    def __init__(self, speakers: Iterable[str], parties: Iterable[str]) -> None:
        """Prepare the lookup tables.

        Args:
            speakers: Full names of all speakers in the archive ("First Last").
            parties: Codes of all parties in the archive.
        """
        self._speakers_by_form: dict[str, set[str]] = {}
        self._first_name_stems: dict[str, frozenset[str]] = {}
        for speaker in speakers:
            names = speaker.split()
            if not names:
                continue
            first_name, surnames = (names[0], names[1:]) if len(names) > 1 else ("", names)
            self._first_name_stems[speaker] = first_name_stems(first_name)
            for surname in surnames:
                for form in surname_forms(surname):
                    self._speakers_by_form.setdefault(form, set()).add(speaker)
        self._party_patterns = [
            (party, re.compile(rf"(?<!\w){re.escape(strip_diacritics(party))}(?!\w)"))
            for party in parties
            if party
        ]

    def analyze(self, query: str) -> QueryAnalysis:
        """Recognise speakers, parties and years in a query."""
        text = unicodedata.normalize("NFC", query)
        plain = strip_diacritics(text)  # same length as text, so spans are shared
        mentions, speaker_spans = self._find_speakers(text)
        parties, party_spans = self._find_parties(plain)
        date_from, date_to, period_spans = _find_period(plain)
        search_filter = SearchFilter(
            speakers=frozenset(name for mention in mentions for name in mention.speakers),
            parties=frozenset(parties),
            date_from=date_from,
            date_to=date_to,
        )
        return QueryAnalysis(
            query=query,
            search_filter=search_filter,
            speaker_mentions=tuple(mentions),
            party_mentions=tuple(parties),
            search_text=_without_spans(text, [*speaker_spans, *party_spans, *period_spans]),
        )

    def _find_speakers(self, text: str) -> tuple[list[SpeakerMention], list[_Span]]:
        tokens = tokenize(text)
        mentions: list[SpeakerMention] = []
        spans: list[_Span] = []
        used: set[int] = set()
        for position, token in enumerate(tokens):
            if position in used or not token.is_capitalized:
                continue
            candidates = self._speakers_by_form.get(fold(token.text))
            if not candidates:
                continue
            start, end = token.start, token.end
            for neighbour in (position - 1, position + 1):
                if neighbour in used or not 0 <= neighbour < len(tokens):
                    continue
                word = tokens[neighbour]
                if not word.is_capitalized:
                    continue
                narrowed = {
                    name
                    for name in candidates
                    if matches_first_name(word.text, self._first_name_stems[name])
                }
                if narrowed:
                    candidates = narrowed
                    used.add(neighbour)
                    start, end = min(start, word.start), max(end, word.end)
                    break
            used.add(position)
            mentions.append(SpeakerMention(text[start:end], tuple(sorted(candidates))))
            spans.append((start, end))
        return mentions, spans

    def _find_parties(self, plain: str) -> tuple[list[str], list[_Span]]:
        parties: list[str] = []
        spans: list[_Span] = []
        for party, pattern in self._party_patterns:
            matches = list(pattern.finditer(plain))
            if matches:
                parties.append(party)
                spans.extend(match.span() for match in matches)
        return parties, spans


def _find_period(plain: str) -> tuple[dt.date | None, dt.date | None, list[_Span]]:
    """Return the date range named in the query and where it is written."""
    starts: list[dt.date] = []
    ends: list[dt.date] = []
    spans: list[_Span] = []

    def unclaimed(match: re.Match[str]) -> bool:
        start, end = match.span()
        return all(end <= taken_start or start >= taken_end for taken_start, taken_end in spans)

    for pattern in _RANGE_PATTERNS:
        for match in filter(unclaimed, pattern.finditer(plain)):
            first, last = sorted((int(match.group(1)), int(match.group(2))))
            starts.append(dt.date(first, 1, 1))
            ends.append(dt.date(last, 12, 31))
            spans.append(match.span())
    for pattern, shift in _FROM_PATTERNS:
        for match in filter(unclaimed, pattern.finditer(plain)):
            starts.append(dt.date(int(match.group(1)) + shift, 1, 1))
            spans.append(match.span())
    for pattern, shift in _TO_PATTERNS:
        for match in filter(unclaimed, pattern.finditer(plain)):
            ends.append(dt.date(int(match.group(1)) + shift, 12, 31))
            spans.append(match.span())
    for match in filter(unclaimed, _SINGLE_YEAR_PATTERN.finditer(plain)):
        year = int(match.group(1))
        starts.append(dt.date(year, 1, 1))
        ends.append(dt.date(year, 12, 31))
        spans.append(match.span())

    date_from = max(starts) if starts else None
    date_to = min(ends) if ends else None
    if date_from and date_to and date_from > date_to:
        return None, None, []  # contradictory periods: better no filter than an empty one
    return date_from, date_to, spans


def _without_spans(text: str, spans: list[_Span]) -> str:
    """Remove the spans from the text; empty if no word would remain."""
    kept = list(text)
    for start, end in spans:
        kept[start:end] = [" "] * (end - start)
    remainder = " ".join("".join(kept).split())
    return remainder if re.search(r"\w", remainder) else ""
