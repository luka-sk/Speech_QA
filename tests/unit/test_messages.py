import dataclasses
import datetime as dt
from pathlib import Path

import pytest

from speech_qa.application.ingestion import IngestReport
from speech_qa.config import Settings
from speech_qa.diagnostics import CheckReport, IndexState
from speech_qa.domain.errors import (
    ConfigurationError,
    EmptyQueryError,
    EvaluationDataError,
    IndexIncompatibleError,
    IndexLockedError,
    IndexNotFoundError,
    InputFileError,
    InputFileProblem,
    ModelNotAvailableError,
    ModelRequestError,
    ModelServiceUnavailableError,
    OutputExistsError,
    QueryTooLongError,
    SpeechQAError,
)
from speech_qa.domain.models import (
    Answer,
    AnswerTimings,
    Chunk,
    ChunkingParameters,
    Citation,
    Evidence,
    IndexManifest,
    InputIssue,
    IssueKind,
    ModelPurpose,
    QueryAnalysis,
    RetrievedChunk,
    SearchFilter,
    SpeechInfo,
)
from speech_qa.messages import (
    MAX_LISTED_ISSUES,
    render_answer,
    render_check_report,
    render_error,
    render_ingest_report,
    render_issues,
    render_search,
)

PATH = Path("data/speeches.jsonl")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ModelNotAvailableError("bge-m3", ModelPurpose.EMBEDDING),
            "Embedding model „bge-m3“ není v Ollamě dostupný. Je nutný pro výpočet embeddingů. "
            "Nainstalujte ho příkazem: ollama pull bge-m3",
        ),
        (
            ModelNotAvailableError("gemma4:12b", ModelPurpose.CHAT),
            "Jazykový model „gemma4:12b“ není v Ollamě dostupný. Je nutný pro generování odpovědí. "
            "Nainstalujte ho příkazem: ollama pull gemma4:12b",
        ),
        (
            ModelServiceUnavailableError("http://localhost:11434/v1", "refused"),
            "Nelze se spojit s Ollamou na adrese http://localhost:11434/v1",
        ),
        (ModelRequestError("bge-m3", "HTTP 400"), "Ollama odmítla požadavek pro model „bge-m3“"),
        (IndexNotFoundError(Path("idx")), "python cli.py ingest data/speeches.jsonl"),
        (IndexIncompatibleError("embedding model differs"), "--rebuild"),
        (IndexLockedError(Path("idx")), "právě používá jiný spuštěný příkaz"),
        (
            ConfigurationError("speech_qa.toml", ["a: bad"]),
            "Chybná konfigurace (speech_qa.toml):\n  - a: bad",
        ),
        (
            InputFileError(PATH, InputFileProblem.NOT_FOUND),
            "Soubor data/speeches.jsonl neexistuje.",
        ),
        (InputFileError(PATH, InputFileProblem.NOT_A_FILE), "data/speeches.jsonl není soubor."),
        (
            InputFileError(PATH, InputFileProblem.UNREADABLE, detail="Permission denied"),
            "nelze přečíst.\nDetail: Permission denied",
        ),
        (
            InputFileError(
                PATH, InputFileProblem.INVALID_LINE, (InputIssue(7, IssueKind.INVALID_JSON, "x"),)
            ),
            "Index zůstal beze změny.\n  - řádek 7: neplatný JSON (x)",
        ),
        (InputFileError(PATH, InputFileProblem.NO_VALID_SPEECHES), "žádný platný projev"),
        (SpeechQAError("something odd"), "Chyba: something odd"),
    ],
)
def test_errors_are_rendered_in_czech(error: SpeechQAError, expected: str) -> None:
    assert expected in render_error(error)


def test_every_issue_kind_has_a_text() -> None:
    issues = tuple(InputIssue(n, kind) for n, kind in enumerate(IssueKind, start=1))

    lines = render_issues(issues)

    assert len(lines) == len(IssueKind)
    assert all(line.startswith("  - řádek ") for line in lines)


def test_long_issue_lists_are_shortened() -> None:
    issues = tuple(InputIssue(n, IssueKind.INVALID_JSON) for n in range(MAX_LISTED_ISSUES + 5))

    lines = render_issues(issues)

    assert len(lines) == MAX_LISTED_ISSUES + 1
    assert lines[-1] == "  - … a dalších 5 vadných řádků"


def report(*, rebuilt: bool = False, issues: tuple[InputIssue, ...] = ()) -> IngestReport:
    manifest = IndexManifest(
        schema_version=1,
        embedding_model="bge-m3",
        embedding_dimension=1024,
        chunking=ChunkingParameters(1000, 1500, 1),
        speakers=("A", "B"),
        parties=("R7",),
        speech_count=1000,
        chunk_count=1000,
        updated_at=dt.datetime(2026, 11, 12, tzinfo=dt.UTC),
    )
    return IngestReport(
        speeches_in_input=1000,
        new=990,
        changed=4,
        unchanged=6,
        chunks_written=994,
        issues=issues,
        rebuilt=rebuilt,
        duration_seconds=42.06,
        manifest=manifest,
    )


def test_ingest_report_is_rendered_in_czech() -> None:
    text = render_ingest_report(report(), ".speech_qa/index")

    assert text.splitlines() == [
        "Ingest dokončen za 42,1 s.",
        "  Projevy v souboru: 1000 (nové 990, změněné 4, beze změny 6)",
        "  Vadné řádky (přeskočeny): 0",
        "  Uložené úseky: 994",
        "  Index: .speech_qa/index",
        "    celkem 1000 projevů, 1000 úseků, 2 řečníků, 1 stran",
    ]


def test_ingest_report_mentions_rebuild_and_lists_issues() -> None:
    text = render_ingest_report(
        report(
            rebuilt=True, issues=(InputIssue(3, IssueKind.DUPLICATE_ID, "Q1 (first on line 1)"),)
        ),
        "idx",
    )

    assert text.startswith("Index byl postaven znovu od nuly.\n")
    assert text.endswith("  - řádek 3: duplicitní id (Q1 (first on line 1))")


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #
def make_evidence(speech_id: str, text: str, party: str | None = "R7") -> Evidence:
    info = SpeechInfo(speech_id, "Lukáš Doležal", party, dt.date(2021, 4, 15))
    passages = (
        RetrievedChunk(Chunk(speech_id, 0, "Méně relevantní úsek."), info, 0.4),
        RetrievedChunk(Chunk(speech_id, 1, text), info, 0.8),
    )
    return Evidence(info, passages, 0.8)


def make_answer(**overrides: object) -> Answer:
    values: dict[str, object] = {
        "question": "Co si myslí Doležal?",
        "text": "Doležal tvrdí, že kvalita rozhodnutí závisí na informacích [1].",
        "citations": (Citation(1, make_evidence("ParlaMint-CZ_x.u1", "Kvalita rozhodnutí.")),),
        "unknown_citations": (),
        "sources": (),
        "analysis": QueryAnalysis(query="Co si myslí Doležal?"),
        "found_nothing": False,
        "prompt_tokens": 2000,
        "context_may_be_truncated": False,
        "timings": AnswerTimings(0.1, 2.0),
    }
    values.update(overrides)
    return Answer(**values)  # type: ignore[arg-type]


def test_answer_with_sources_is_rendered() -> None:
    analysis = QueryAnalysis(
        query="Co si myslí Doležal?",
        search_filter=SearchFilter(speakers=frozenset({"Ondřej Doležal", "Lukáš Doležal"})),
    )

    text = render_answer(make_answer(analysis=analysis))

    assert text.splitlines() == [
        "Doležal tvrdí, že kvalita rozhodnutí závisí na informacích [1].",
        "",
        "Hledáno v projevech: Lukáš Doležal, Ondřej Doležal",
        "Zdroje:",
        "  [1] ParlaMint-CZ_x.u1 – Lukáš Doležal (R7), 15. 4. 2021",
        "      „Kvalita rozhodnutí.“",
    ]


def test_speaker_without_party_and_long_excerpt() -> None:
    long_text = "slovo " * 100
    citation = Citation(1, make_evidence("Q1", long_text, party=None))

    lines = render_answer(make_answer(citations=(citation,))).splitlines()

    assert lines[-2] == "  [1] Q1 – Lukáš Doležal (bez strany), 15. 4. 2021"
    assert lines[-1].endswith(" …“")
    assert len(lines[-1]) < 320


def test_answer_without_citations_says_so() -> None:
    text = render_answer(make_answer(text="Archiv k tomu nic neobsahuje.", citations=()))

    assert text.splitlines() == [
        "Archiv k tomu nic neobsahuje.",
        "",
        "(Odpověď neodkazuje na žádný projev.)",
    ]


def test_nothing_found_is_rendered_with_the_filter() -> None:
    analysis = QueryAnalysis(
        query="?",
        search_filter=SearchFilter(
            parties=frozenset({"VIZE"}), date_from=dt.date(2030, 1, 1), date_to=None
        ),
    )

    text = render_answer(make_answer(found_nothing=True, citations=(), text="", analysis=analysis))

    assert text.splitlines() == [
        "V archivu jsem nenašel žádné projevy odpovídající dotazu.",
        "",
        "Strany: VIZE",
        "Období: 1. 1. 2030 – …",
    ]


def test_nothing_found_without_filter_is_a_single_line() -> None:
    text = render_answer(make_answer(found_nothing=True, citations=(), text=""))

    assert text == "V archivu jsem nenašel žádné projevy odpovídající dotazu."


def test_warnings_are_rendered() -> None:
    text = render_answer(
        make_answer(unknown_citations=("5", "7"), context_may_be_truncated=True, prompt_tokens=4000)
    )

    assert (
        "Upozornění: odpověď odkazuje na zdroj [5], [7], který nebyl mezi předanými projevy."
        in text
    )
    assert "Upozornění: prompt měl 4000 tokenů" in text


def test_period_with_open_start() -> None:
    assert render_search(SearchFilter(date_to=dt.date(2019, 12, 31))) == [
        "Období: … – 31. 12. 2019"
    ]


def test_control_characters_are_removed_from_the_answer() -> None:
    text = render_answer(
        make_answer(text="Odpověď\x1b[31m červeně\x07\n\tdalší řádek", citations=())
    )

    assert text.splitlines()[0] == "Odpověď[31m červeně"
    assert text.splitlines()[1] == "\tdalší řádek"


def test_query_errors_are_rendered() -> None:
    assert "Otázka je prázdná." in render_error(EmptyQueryError())
    assert "(1200 znaků, nejvýše 1000)" in render_error(QueryTooLongError(1200, 1000))


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def check_report(**overrides: object) -> CheckReport:
    manifest = report().manifest
    values: dict[str, object] = {
        "config_file": Path("speech_qa.toml"),
        "settings": Settings(),
        "server_error": None,
        "installed_models": frozenset({"bge-m3:latest", "gemma4:12b"}),
        "index_state": IndexState.READY,
        "index_detail": "",
        "manifest": manifest,
    }
    values.update(overrides)
    return CheckReport(**values)  # type: ignore[arg-type]


def test_check_report_when_everything_is_ready() -> None:
    lines = render_check_report(check_report()).splitlines()

    assert lines[0] == "OK       Konfigurace: speech_qa.toml"
    assert lines[1] == "OK       Ollama běží: http://localhost:11434/v1"
    assert lines[2] == "OK       Embedding model „bge-m3“ je nainstalovaný."
    assert lines[3] == "OK       Jazykový model „gemma4:12b“ je nainstalovaný."
    assert lines[4].startswith("OK       Index: ")
    assert "1000 projevů, 1000 úseků, 2 řečníků, 1 stran" in lines[4]
    assert lines[5].startswith("OK       Kontextové okno modelu podle konfigurace: 4096 tokenů")
    assert len(lines) == 6


def test_check_report_with_missing_models_and_server_down() -> None:
    missing = render_check_report(
        check_report(installed_models=frozenset({"gemma4:12b"}), config_file=None)
    )
    down = render_check_report(check_report(server_error="connection refused"))

    assert "výchozí hodnoty" in missing
    assert "CHYBA    Embedding model „bge-m3“ není v Ollamě dostupný." in missing
    assert "ollama pull bge-m3" in missing
    assert "CHYBA    Nelze se spojit s Ollamou" in down
    assert "connection refused" in down


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (IndexState.MISSING, "CHYBA    Index zatím neexistuje"),
        (IndexState.EMPTY, "CHYBA    Index zatím neexistuje"),
        (IndexState.LOCKED, "VAROVÁNÍ Index"),
        (IndexState.UNREADABLE, "CHYBA    Index nelze načíst: broken"),
    ],
)
def test_check_report_index_states(state: IndexState, expected: str) -> None:
    text = render_check_report(
        check_report(index_state=state, manifest=None, index_detail="broken")
    )

    assert expected in text


def test_check_report_incompatible_index() -> None:
    manifest = dataclasses.replace(
        report().manifest, embedding_model="other", chunking=ChunkingParameters(500, 800, 0)
    )

    text = render_check_report(check_report(manifest=manifest))

    assert "CHYBA    Index byl vytvořen embedding modelem „other“" in text
    assert "VAROVÁNÍ Index byl rozdělen s jinými parametry" in text


def test_evaluation_errors_are_rendered() -> None:
    data_error = render_error(
        EvaluationDataError(Path("data/eval/manual.csv"), ["line 3: duplicate id q1"])
    )
    exists = render_error(OutputExistsError(Path("data/eval/synthetic.csv")))

    assert (
        data_error
        == "Evaluační soubor data/eval/manual.csv nelze použít:\n  - line 3: duplicate id q1"
    )
    assert (
        exists == "Soubor data/eval/synthetic.csv už existuje. Přepsat ho lze přepínačem --force."
    )
