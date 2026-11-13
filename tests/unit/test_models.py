import dataclasses
import datetime as dt

import pytest

from speech_qa.domain.models import QueryAnalysis, SearchFilter, Speech, SpeechInfo


def make_speech(party: str | None = "ABC") -> Speech:
    return Speech(
        speech_id="Q000001",
        speaker="Lukáš Doležal",
        party=party,
        date=dt.date(2024, 3, 27),
        text="Kvalita rozhodnutí závisí na kvalitě informací.",
    )


def test_speech_info_carries_metadata_without_text() -> None:
    speech = make_speech()

    assert speech.info == SpeechInfo(
        speech_id="Q000001", speaker="Lukáš Doležal", party="ABC", date=dt.date(2024, 3, 27)
    )


def test_speech_without_party_keeps_none() -> None:
    assert make_speech(party=None).info.party is None


def test_speech_is_immutable() -> None:
    speech = make_speech()

    with pytest.raises(dataclasses.FrozenInstanceError):
        speech.text = "changed"  # type: ignore[misc]


def test_default_filter_is_empty() -> None:
    assert SearchFilter().is_empty


@pytest.mark.parametrize(
    "search_filter",
    [
        SearchFilter(speakers=frozenset({"Lukáš Doležal"})),
        SearchFilter(parties=frozenset({"ABC"})),
        SearchFilter(date_from=dt.date(2020, 1, 1)),
        SearchFilter(date_to=dt.date(2020, 12, 31)),
    ],
)
def test_filter_with_any_restriction_is_not_empty(search_filter: SearchFilter) -> None:
    assert not search_filter.is_empty


def test_filter_rejects_inverted_date_range() -> None:
    with pytest.raises(ValueError, match="later than"):
        SearchFilter(date_from=dt.date(2021, 1, 1), date_to=dt.date(2020, 1, 1))


def test_filter_accepts_single_day_range() -> None:
    day = dt.date(2020, 5, 5)

    assert SearchFilter(date_from=day, date_to=day).date_to == day


def test_query_analysis_defaults_to_no_filter_and_no_mentions() -> None:
    analysis = QueryAnalysis(query="Co lidé diskutovali ke kvalitě rozhodování?")

    assert analysis.search_filter.is_empty
    assert analysis.speaker_mentions == ()
    assert analysis.party_mentions == ()
