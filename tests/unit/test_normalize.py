import unicodedata

from speech_qa.text.normalize import Token, fold, strip_diacritics, tokenize


def test_strip_diacritics_keeps_case_and_length() -> None:
    text = "Příliš žluťoučký kůň úpěl ďábelské ódy, MÜLLER"

    stripped = strip_diacritics(text)

    assert stripped == "Prilis zlutoucky kun upel dabelske ody, MULLER"
    assert len(stripped) == len(text)


def test_strip_diacritics_normalises_decomposed_input() -> None:
    assert strip_diacritics(unicodedata.normalize("NFD", "Doležal")) == "Dolezal"


def test_characters_without_a_simple_base_letter_are_kept() -> None:
    assert strip_diacritics("ß æ ø") == "ß æ ø"


def test_fold_lowercases_and_removes_diacritics() -> None:
    assert fold("Doležalová") == "dolezalova"


def test_tokenize_keeps_positions_and_hyphenated_words() -> None:
    tokens = tokenize("Co řekla Malá-Horáková, a proč?")

    assert tokens == [
        Token("Co", 0, 2),
        Token("řekla", 3, 8),
        Token("Malá-Horáková", 9, 22),
        Token("a", 24, 25),
        Token("proč", 26, 30),
    ]
    assert tokens[2].is_capitalized
    assert not tokens[1].is_capitalized
