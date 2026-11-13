import datetime as dt
import itertools

import pytest

from speech_qa.domain.models import ChunkingParameters, Speech
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.text.sentences import SentenceWindowChunker, split_sentences
from tests.conftest import LONG_SPEECHES


def speech(text: str, speech_id: str = "S1") -> Speech:
    return Speech(
        speech_id=speech_id, speaker="A B", party=None, date=dt.date(2020, 1, 1), text=text
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("První věta. Druhá věta.", ["První věta.", "Druhá věta."]),
        ("Opravdu? Ano! Dobře… Konec.", ["Opravdu?", "Ano!", "Dobře…", "Konec."]),
        ("Rok 2020 byl těžký. 2021 byl lepší.", ["Rok 2020 byl těžký.", "2021 byl lepší."]),
        ("Řekl „ano.“ Pak odešel.", ["Řekl „ano.“", "Pak odešel."]),
        ("Je to např. Praha. Další věta.", ["Je to např. Praha.", "Další věta."]),
        ("Ing. Novák přišel. Mgr. Nováková ne.", ["Ing. Novák přišel.", "Mgr. Nováková ne."]),
        ("Stálo to 5 mld. Kč ročně. Hodně.", ["Stálo to 5 mld. Kč ročně.", "Hodně."]),
        ("Podle J. Nováka to platí. Ano.", ["Podle J. Nováka to platí.", "Ano."]),
        ("Viz čl. 5 a odst. 3. Dále.", ["Viz čl. 5 a odst. 3.", "Dále."]),
        ("Malá písmena. pokračují dál.", ["Malá písmena. pokračují dál."]),
        ("Hlasy, granty atd. Další téma.", ["Hlasy, granty atd.", "Další téma."]),
        ("Bez tečky na konci", ["Bez tečky na konci"]),
        ("  Více   mezer\n a řádků.  Nová věta. ", ["Více mezer a řádků.", "Nová věta."]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_split_sentences_of_empty_text() -> None:
    assert split_sentences("   ") == []


def test_short_speech_is_a_single_chunk() -> None:
    chunker = SentenceWindowChunker(target_chars=1000, max_chars=1500, overlap_sentences=1)

    chunks = chunker.split(speech("Jedna věta. Druhá věta."))

    assert [(chunk.index, chunk.text) for chunk in chunks] == [(0, "Jedna věta. Druhá věta.")]
    assert chunks[0].speech_id == "S1"


def test_windows_reach_the_target_and_overlap_by_one_sentence() -> None:
    sentences = [f"Věta číslo {n} obsahuje několik slov." for n in range(10)]  # 35-36 chars
    chunker = SentenceWindowChunker(target_chars=100, max_chars=150, overlap_sentences=1)

    chunks = chunker.split(speech(" ".join(sentences)))

    windows = [split_sentences(chunk.text) for chunk in chunks]
    assert [len(window) for window in windows] == [3, 3, 3, 3, 2]
    for previous, following in itertools.pairwise(windows):
        assert previous[-1] == following[0]
    assert windows[-1][-1] == sentences[-1]
    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3, 4]


def test_windows_without_overlap_partition_the_sentences() -> None:
    sentences = [f"Věta číslo {n} obsahuje několik slov." for n in range(6)]
    chunker = SentenceWindowChunker(target_chars=70, max_chars=150, overlap_sentences=0)

    chunks = chunker.split(speech(" ".join(sentences)))

    assert [split_sentences(chunk.text) for chunk in chunks] == [
        sentences[0:2],
        sentences[2:4],
        sentences[4:6],
    ]


def test_window_never_exceeds_max_chars_when_sentences_are_shorter() -> None:
    chunker = SentenceWindowChunker(target_chars=1000, max_chars=1500, overlap_sentences=1)
    speeches = read_speeches(LONG_SPEECHES).speeches

    for item in speeches:
        chunks = chunker.split(item)
        assert len(chunks) >= 2, item.speech_id
        assert all(len(chunk.text) <= 1500 for chunk in chunks)
        assert " ".join(item.text.split()).endswith(chunks[-1].text.split(" ")[-1])


def test_sentence_longer_than_max_chars_forms_its_own_window() -> None:
    long_sentence = "Slovo " * 60 + "konec."
    chunker = SentenceWindowChunker(target_chars=100, max_chars=150, overlap_sentences=1)

    chunks = chunker.split(speech(f"Úvod. {long_sentence} Závěr."))

    assert [chunk.text for chunk in chunks] == ["Úvod.", " ".join(long_sentence.split()), "Závěr."]


def test_large_overlap_still_makes_progress() -> None:
    chunker = SentenceWindowChunker(target_chars=10, max_chars=20, overlap_sentences=5)

    chunks = chunker.split(speech("Jedna věta. Druhá věta. Třetí věta."))

    assert [chunk.text for chunk in chunks] == ["Jedna věta.", "Druhá věta.", "Třetí věta."]


def test_parameters_are_exposed_for_the_manifest() -> None:
    chunker = SentenceWindowChunker(target_chars=900, max_chars=1200, overlap_sentences=2)

    assert chunker.parameters == ChunkingParameters(900, 1200, 2)


@pytest.mark.parametrize(
    ("target", "maximum", "overlap"), [(0, 100, 0), (200, 100, 0), (100, 200, -1)]
)
def test_invalid_parameters_are_rejected(target: int, maximum: int, overlap: int) -> None:
    with pytest.raises(ValueError, match=r"must|require"):
        SentenceWindowChunker(target_chars=target, max_chars=maximum, overlap_sentences=overlap)
