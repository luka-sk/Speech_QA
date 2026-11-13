import pytest

from speech_qa.text.czech_names import first_name_stems, matches_first_name, surname_forms


@pytest.mark.parametrize(
    ("surname", "expected_forms"),
    [
        ("Doležal", {"dolezal", "dolezala", "dolezalovi", "dolezalem", "dolezale"}),
        ("Němec", {"nemec", "nemce", "nemcovi", "nemcem"}),
        ("Hájek", {"hajek", "hajka", "hajkovi", "hajkem"}),
        ("Sedláček", {"sedlacek", "sedlacka", "sedlackovi"}),
        ("Kolář", {"kolar", "kolare", "kolarovi", "kolari", "kolarem"}),
        ("Kučera", {"kucera", "kucery", "kucerovi", "kuceru", "kucerou"}),
        ("Svoboda", {"svoboda", "svobody", "svobodovi", "svobodu", "svobodou"}),
        ("Pokorný", {"pokorny", "pokorneho", "pokornemu", "pokornym", "pokornem"}),
        ("Krejčí", {"krejci", "krejciho", "krejcimu", "krejcim"}),
        ("Doležalová", {"dolezalova", "dolezalove", "dolezalovou"}),
        ("Pokorná", {"pokorna", "pokorne", "pokornou"}),
        ("Müller", {"muller", "mullera", "mullerovi", "mullerem"}),
    ],
)
def test_surname_forms_cover_the_grammatical_cases(surname: str, expected_forms: set[str]) -> None:
    assert expected_forms <= surname_forms(surname)


def test_female_forms_are_exactly_the_three_case_forms() -> None:
    assert surname_forms("Doležalová") == {"dolezalova", "dolezalove", "dolezalovou"}


def test_names_ending_in_a_vowel_other_than_a_are_indeclinable() -> None:
    assert surname_forms("Gaudí") != {"gaudi"}  # -í declines like an adjective
    assert surname_forms("Rousseau") == {"rousseau"}


def test_double_surname_is_declined_as_a_whole_and_by_parts() -> None:
    forms = surname_forms("Malá-Horáková")

    assert {"mala-horakova", "male-horakove", "malou-horakovou"} <= forms
    assert {"horakova", "horakove", "mala"} <= forms
    assert "male-horakovou" not in forms  # parts are combined in the same case


@pytest.mark.parametrize(
    ("male", "female"),
    [
        ("Doležal", "Doležalová"),
        ("Němec", "Němcová"),
        ("Hájek", "Hájková"),
        ("Sedláček", "Sedláčková"),
        ("Pokorný", "Pokorná"),
        ("Černý", "Černá"),
        ("Veselý", "Veselá"),
        ("Svoboda", "Svobodová"),
        ("Kučera", "Kučerová"),
        ("Procházka", "Procházková"),
        ("Růžička", "Růžičková"),
        ("Fiala", "Fialová"),
        ("Král", "Králová"),
        ("Kolář", "Kolářová"),
        ("Beneš", "Benešová"),
        ("Novák", "Nováková"),
        ("Dvořák", "Dvořáková"),
        ("Horák", "Horáková"),
        ("Zeman", "Zemanová"),
        ("Pospíšil", "Pospíšilová"),
    ],
)
def test_male_and_female_surnames_never_share_a_form(male: str, female: str) -> None:
    assert surname_forms(male).isdisjoint(surname_forms(female))


@pytest.mark.parametrize(
    ("first_name", "stems"),
    [
        ("Lukáš", {"lukas"}),
        ("Petra", {"petra", "petr"}),
        ("Pavel", {"pavel", "pavl"}),
        ("Marek", {"marek", "mark"}),
        ("Eva", {"eva"}),
        ("Jan", {"jan"}),
        ("Al", set()),
    ],
)
def test_first_name_stems(first_name: str, stems: set[str]) -> None:
    assert first_name_stems(first_name) == stems


@pytest.mark.parametrize(
    ("word", "first_name", "expected"),
    [
        ("Lukáše", "Lukáš", True),
        ("Lukášovi", "Lukáš", True),
        ("Ondřeje", "Ondřej", True),
        ("Ondreje", "Ondřej", True),
        ("Pavla", "Pavel", True),
        ("Veroniky", "Veronika", True),
        ("Petrovi", "Petr", True),
        ("Lukášovicích", "Lukáš", False),
        ("Tomáše", "Lukáš", False),
    ],
)
def test_matches_first_name(word: str, first_name: str, expected: bool) -> None:
    assert matches_first_name(word, first_name_stems(first_name)) is expected
