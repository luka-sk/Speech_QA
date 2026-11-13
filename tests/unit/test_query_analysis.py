import datetime as dt
import unicodedata

import pytest

from speech_qa.domain.models import SpeakerMention
from speech_qa.text.query_analysis import RuleBasedQueryAnalyzer

SPEAKERS = [
    "Lukáš Doležal",
    "Ondřej Doležal",
    "Veronika Doležalová",
    "Lenka Doležalová",
    "Tomáš Němec",
    "Petr Němec",
    "Jan Hájek",
    "Klára Hájková",
    "Petr Sedláček",
    "Martin Svoboda",
    "Marie Svobodová",
    "Michal Pokorný",
    "Petra Pokorná",
    "Filip Král",
    "Karel Krejčí",
    "David Müller",
    "Eva Smith",
    "Lucie Malá-Horáková",
    "Madonna",
    "",
]
PARTIES = ["R7", "K12", "VIZE", "NOVA", "LID", "PRO", "KDU-ČSL", "ANO", "TOP 09", ""]
DOLEZALS = {"Lukáš Doležal", "Ondřej Doležal"}

ANALYZER = RuleBasedQueryAnalyzer(SPEAKERS, PARTIES)


def speakers(query: str) -> set[str]:
    return set(ANALYZER.analyze(query).search_filter.speakers)


def test_assignment_query_about_dolezal_matches_both_male_dolezals() -> None:
    query = "Co si myslí Doležal o vlivku kvality rozhodnutí na kvalitě informací?"

    analysis = ANALYZER.analyze(query)

    assert analysis.search_filter.speakers == DOLEZALS
    assert analysis.speaker_mentions == (
        SpeakerMention("Doležal", ("Lukáš Doležal", "Ondřej Doležal")),
    )
    assert analysis.search_text == "Co si myslí o vlivku kvality rozhodnutí na kvalitě informací?"
    assert analysis.embedding_text == analysis.search_text


def test_assignment_query_without_names_has_no_filter() -> None:
    query = "Co lidé diskutovali ke kvalitě rozhodování?"

    analysis = ANALYZER.analyze(query)

    assert analysis.search_filter.is_empty
    assert analysis.embedding_text == query


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Souhlasí kolegové s Doležalem?", DOLEZALS),
        # Known limitation: the male plural "Doležalové" equals a case form of
        # the female surname, so it is understood as the female speakers.
        ("Co navrhují Doležalové?", {"Veronika Doležalová", "Lenka Doležalová"}),
        ("Co řekla Doležalová?", {"Veronika Doležalová", "Lenka Doležalová"}),
        ("Jaký názor má Veronika Doležalová?", {"Veronika Doležalová"}),
        ("Názor Veroniky Doležalové", {"Veronika Doležalová"}),
        ("Podle Lukáše Doležala je to jinak.", {"Lukáš Doležal"}),
        ("Doležal Ondřej to řekl.", {"Ondřej Doležal"}),
        ("Co říkal Jan Doležal?", DOLEZALS),  # unknown first name keeps all
        ("Co navrhoval Němec?", {"Tomáš Němec", "Petr Němec"}),
        ("Názor Tomáše Němce", {"Tomáš Němec"}),
        ("Co říkal Hájek?", {"Jan Hájek"}),
        ("Souhlasíte s Hájkem?", {"Jan Hájek"}),
        ("Co řekla Hájková?", {"Klára Hájková"}),
        ("Podle Sedláčka", {"Petr Sedláček"}),
        ("Názor Pokorného", {"Michal Pokorný"}),
        ("Názor Pokorné", {"Petra Pokorná"}),
        ("S Pokornou nesouhlasím", {"Petra Pokorná"}),
        ("Co říkal Svoboda?", {"Martin Svoboda"}),
        ("Podle Svobody", {"Martin Svoboda"}),
        ("Co říkala Svobodová?", {"Marie Svobodová"}),
        ("Názor Krejčího", {"Karel Krejčí"}),
        ("Co říkal Müller?", {"David Müller"}),
        ("Názor Mullera", {"David Müller"}),
        ("Co navrhovala Eva Smith?", {"Eva Smith"}),
        ("Názor Malé-Horákové", {"Lucie Malá-Horáková"}),
        ("Co říkala Horáková?", {"Lucie Malá-Horáková"}),
        ("Co zpívala Madonna?", {"Madonna"}),
        ("Co rikal Dolezal?", DOLEZALS),
        ("Co říkali Hájek a Němec?", {"Jan Hájek", "Tomáš Němec", "Petr Němec"}),
    ],
)
def test_speakers_are_recognised_in_any_case(query: str, expected: set[str]) -> None:
    assert speakers(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "Jak je důležitá svoboda projevu?",
        "Co se říkalo o svobodě?",
        "Kdo je král debaty?",
        "co rikal dolezal o datech",  # names written in lower case are not recognised
        "Co lidé říkali o kvalitě?",
    ],
)
def test_common_words_and_lower_case_names_are_not_speakers(query: str) -> None:
    assert speakers(query) == set()


def test_decomposed_unicode_query_is_understood() -> None:
    assert speakers(unicodedata.normalize("NFD", "Co říkal Doležal?")) == DOLEZALS


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Co prosazovala VIZE v dopravě?", {"VIZE"}),
        ("Jaká je vize budoucnosti?", set()),
        ("Co říkali poslanci PRO?", {"PRO"}),
        ("Hlasovali pro návrh?", set()),
        ("Co říká LID o energetice?", {"LID"}),
        ("Postoj KDU-ČSL k důchodům", {"KDU-ČSL"}),
        ("Postoj KDU-CSL k důchodům", {"KDU-ČSL"}),
        ("Co navrhuje ANO?", {"ANO"}),
        ("ano, to je otázka", set()),
        ("Program TOP 09", {"TOP 09"}),
        ("Jak se lišily R7 a K12?", {"R7", "K12"}),
        ("Co říká R77?", set()),
    ],
)
def test_parties_are_recognised_by_their_exact_code(query: str, expected: set[str]) -> None:
    analysis = ANALYZER.analyze(query)

    assert set(analysis.search_filter.parties) == expected
    assert set(analysis.party_mentions) == expected


@pytest.mark.parametrize(
    ("query", "date_from", "date_to"),
    [
        ("Co se v roce 2020 říkalo o suchu?", dt.date(2020, 1, 1), dt.date(2020, 12, 31)),
        ("Co zaznělo roku 2019?", dt.date(2019, 1, 1), dt.date(2019, 12, 31)),
        ("Debaty od 2019", dt.date(2019, 1, 1), None),
        ("Debaty od roku 2019", dt.date(2019, 1, 1), None),
        ("Debaty do roku 2021", None, dt.date(2021, 12, 31)),
        ("Co se říkalo po roce 2020?", dt.date(2021, 1, 1), None),
        ("Co se říkalo před rokem 2020?", None, dt.date(2019, 12, 31)),
        ("Energetika 2019\u20132021", dt.date(2019, 1, 1), dt.date(2021, 12, 31)),
        ("Energetika 2019\u2014 2021", dt.date(2019, 1, 1), dt.date(2021, 12, 31)),
        ("Energetika 2021 - 2019", dt.date(2019, 1, 1), dt.date(2021, 12, 31)),
        ("Mezi lety 2019 a 2021", dt.date(2019, 1, 1), dt.date(2021, 12, 31)),
        ("Od roku 2019 do roku 2021", dt.date(2019, 1, 1), dt.date(2021, 12, 31)),
        ("V letech 2018 až 2020", dt.date(2018, 1, 1), dt.date(2020, 12, 31)),
        ("Od 2019 a v roce 2021", dt.date(2021, 1, 1), dt.date(2021, 12, 31)),
        ("Kolik stálo 2000 bytů?", None, None),
        ("Po roce 2022 a před rokem 2020", None, None),
    ],
)
def test_years_become_a_date_range(
    query: str, date_from: dt.date | None, date_to: dt.date | None
) -> None:
    search_filter = ANALYZER.analyze(query).search_filter

    assert (search_filter.date_from, search_filter.date_to) == (date_from, date_to)


def test_recognised_mentions_are_removed_from_the_search_text() -> None:
    analysis = ANALYZER.analyze("Co říkal Martin Svoboda v roce 2023 za PRO o dálnicích?")

    assert analysis.search_filter.speakers == {"Martin Svoboda"}
    assert analysis.search_filter.parties == {"PRO"}
    assert analysis.search_filter.date_from == dt.date(2023, 1, 1)
    assert analysis.search_text == "Co říkal za o dálnicích?"


def test_query_made_only_of_mentions_is_searched_as_a_whole() -> None:
    analysis = ANALYZER.analyze("Doležal")

    assert analysis.search_text == ""
    assert analysis.embedding_text == "Doležal"
