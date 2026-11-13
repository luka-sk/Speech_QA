"""User-facing texts of the command-line interface, in Czech.

All fixed texts shown to users live here (apart from the evaluation report in
:mod:`speech_qa.evaluation.report`), so that another language can be added in
one place. Errors arrive as structured exceptions and are rendered here.
"""

import datetime as dt
import unicodedata
from pathlib import Path

from speech_qa.application.ingestion import IngestReport
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
    Evidence,
    InputIssue,
    IssueKind,
    ModelPurpose,
    SearchFilter,
    SpeechInfo,
)

MAX_LISTED_ISSUES = 20
"""How many problematic input lines are listed before the rest is summarised."""

EXCERPT_CHARS = 300
"""Length of the excerpt shown for each cited source."""

NO_PARTY = "bez strany"

_ISSUE_TEXTS = {
    IssueKind.LINE_TOO_LONG: "řádek je delší než 1 MB",
    IssueKind.NOT_UTF8: "řádek není v kódování UTF-8",
    IssueKind.INVALID_JSON: "neplatný JSON",
    IssueKind.NOT_AN_OBJECT: "řádek neobsahuje JSON objekt",
    IssueKind.INVALID_FIELDS: "neplatná nebo chybějící pole",
    IssueKind.DUPLICATE_ID: "duplicitní id",
}

_FILE_PROBLEM_TEXTS = {
    InputFileProblem.NOT_FOUND: "Soubor {path} neexistuje.",
    InputFileProblem.NOT_A_FILE: "{path} není soubor.",
    InputFileProblem.UNREADABLE: "Soubor {path} nelze přečíst.",
    InputFileProblem.NO_VALID_SPEECHES: "Soubor {path} neobsahuje žádný platný projev.",
    InputFileProblem.INVALID_LINE: (
        "Ingest byl zastaven kvůli vadnému řádku v souboru {path} (přepínač --strict). "
        "Index zůstal beze změny."
    ),
}

_MODEL_PURPOSE_TEXTS = {
    ModelPurpose.EMBEDDING: ("Embedding model", "Je nutný pro výpočet embeddingů."),
    ModelPurpose.CHAT: ("Jazykový model", "Je nutný pro generování odpovědí."),
}


def render_error(error: SpeechQAError) -> str:
    """Return the Czech message for an expected error."""
    match error:
        case EmptyQueryError():
            return 'Otázka je prázdná. Zadejte ji v uvozovkách: python cli.py answer "…"'
        case QueryTooLongError():
            return (
                f"Otázka je příliš dlouhá ({error.length} znaků, nejvýše {error.limit}). "
                "Zkraťte ji, nebo změňte limit query.max_chars v konfiguraci."
            )
        case EvaluationDataError():
            lines = [f"Evaluační soubor {error.path} nelze použít:"]
            lines += [f"  - {problem}" for problem in error.problems]
            return "\n".join(lines)
        case OutputExistsError():
            return f"Soubor {error.path} už existuje. Přepsat ho lze přepínačem --force."
        case ConfigurationError():
            lines = [f"Chybná konfigurace ({error.source}):"]
            lines += [f"  - {detail}" for detail in error.details]
            return "\n".join(lines)
        case InputFileError():
            lines = [_FILE_PROBLEM_TEXTS[error.problem].format(path=error.path)]
            if error.detail:
                lines.append(f"Detail: {error.detail}")
            lines += render_issues(error.issues)
            return "\n".join(lines)
        case IndexNotFoundError():
            return (
                f"Index zatím neexistuje ({error.path}). "
                "Nejdřív spusťte: python cli.py ingest data/speeches.jsonl"
            )
        case IndexIncompatibleError():
            return (
                f"Index nelze použít se současným nastavením: {error.reason}. "
                "Postavte ho znovu: python cli.py ingest <soubor> --rebuild"
            )
        case IndexLockedError():
            return (
                f"Index ({error.path}) právě používá jiný spuštěný příkaz. "
                "Počkejte na jeho dokončení a zkuste to znovu."
            )
        case ModelServiceUnavailableError():
            return (
                f"Nelze se spojit s Ollamou na adrese {error.base_url}. "
                f"Ověřte, že Ollama běží. (Detail: {error.detail})"
            )
        case ModelNotAvailableError():
            kind, purpose = _MODEL_PURPOSE_TEXTS[error.purpose]
            return (
                f"{kind} „{error.model}“ není v Ollamě dostupný. {purpose} "
                f"Nainstalujte ho příkazem: ollama pull {error.model}"
            )
        case ModelRequestError():
            return f"Ollama odmítla požadavek pro model „{error.model}“. (Detail: {error.detail})"
        case _:
            return f"Chyba: {error}"


def render_issues(issues: tuple[InputIssue, ...]) -> list[str]:
    """Return one line per problematic input line (at most ``MAX_LISTED_ISSUES``)."""
    lines = []
    for issue in issues[:MAX_LISTED_ISSUES]:
        text = f"  - řádek {issue.line}: {_ISSUE_TEXTS[issue.kind]}"
        lines.append(f"{text} ({issue.detail})" if issue.detail else text)
    if len(issues) > MAX_LISTED_ISSUES:
        lines.append(f"  - … a dalších {len(issues) - MAX_LISTED_ISSUES} vadných řádků")
    return lines


def render_ingest_report(report: IngestReport, index_location: str) -> str:
    """Return the Czech summary printed after an ingest."""
    manifest = report.manifest
    lines = []
    if report.rebuilt:
        lines.append("Index byl postaven znovu od nuly.")
    lines += [
        f"Ingest dokončen za {_seconds(report.duration_seconds)} s.",
        f"  Projevy v souboru: {report.speeches_in_input} "
        f"(nové {report.new}, změněné {report.changed}, beze změny {report.unchanged})",
        f"  Vadné řádky (přeskočeny): {len(report.issues)}",
        f"  Uložené úseky: {report.chunks_written}",
        f"  Index: {index_location}",
        f"    celkem {manifest.speech_count} projevů, {manifest.chunk_count} úseků, "
        f"{len(manifest.speakers)} řečníků, {len(manifest.parties)} stran",
    ]
    lines += render_issues(report.issues)
    return "\n".join(lines)


def _seconds(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def render_answer(answer: Answer) -> str:
    """Return the Czech text printed for an answer: the answer, the search and the sources."""
    lines: list[str] = []
    if answer.found_nothing:
        lines.append("V archivu jsem nenašel žádné projevy odpovídající dotazu.")
    else:
        lines.append(sanitize(answer.text))
    search = render_search(answer.analysis.search_filter)
    if search or not answer.found_nothing:
        lines.append("")
    lines += search
    if answer.citations:
        lines.append("Zdroje:")
        for citation in answer.citations:
            lines += render_source(citation.number, citation.evidence)
    elif not answer.found_nothing:
        lines.append("(Odpověď neodkazuje na žádný projev.)")
    if answer.unknown_citations:
        cited = ", ".join(f"[{number}]" for number in answer.unknown_citations)
        lines.append(
            f"Upozornění: odpověď odkazuje na zdroj {cited}, který nebyl mezi předanými projevy."
        )
    if answer.context_may_be_truncated:
        lines.append(
            f"Upozornění: prompt měl {answer.prompt_tokens} tokenů, tedy téměř celé kontextové "
            "okno modelu, a část zdrojů mohla být oříznuta. Zvětšete kontext v Ollamě "
            "a hodnotu generation.context_window_tokens v konfiguraci."
        )
    return "\n".join(lines)


def render_search(search_filter: SearchFilter) -> list[str]:
    """Describe the applied filter; nothing when the whole archive was searched."""
    lines = []
    if search_filter.speakers:
        lines.append(f"Hledáno v projevech: {', '.join(sorted(search_filter.speakers))}")
    if search_filter.parties:
        lines.append(f"Strany: {', '.join(sorted(search_filter.parties))}")
    if search_filter.date_from or search_filter.date_to:
        start = format_date(search_filter.date_from) if search_filter.date_from else "…"
        end = format_date(search_filter.date_to) if search_filter.date_to else "…"
        lines.append(f"Období: {start} – {end}")
    return lines


def render_source(number: int, evidence: Evidence) -> list[str]:
    """Return the lines describing one cited source."""
    best = max(evidence.passages, key=lambda passage: passage.score)
    return [
        f"  [{number}] {describe_speech(evidence.speech)}",
        f"      „{excerpt(sanitize(best.chunk.text))}“",
    ]


def describe_speech(speech: SpeechInfo) -> str:
    """Return "id – speaker (party), date" for a speech."""
    party = speech.party or NO_PARTY
    return f"{speech.speech_id} – {speech.speaker} ({party}), {format_date(speech.date)}"


def format_date(date: dt.date) -> str:
    """Return a date in the Czech format, e.g. "15. 4. 2021"."""
    return f"{date.day}. {date.month}. {date.year}"


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """Shorten text to at most ``limit`` characters at a word boundary."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


def sanitize(text: str) -> str:
    """Remove control characters (e.g. terminal escape sequences) except line breaks and tabs."""
    return "".join(
        character
        for character in text
        if character in "\n\t" or unicodedata.category(character) != "Cc"
    )


def render_check_report(report: CheckReport) -> str:
    """Return the Czech report of ``python cli.py check``."""
    settings = report.settings
    ok, error, warning = "OK      ", "CHYBA   ", "VAROVÁNÍ"
    config = report.config_file or "výchozí hodnoty (soubor speech_qa.toml nenalezen)"
    lines = [f"{ok} Konfigurace: {config}"]
    if report.server_error is None:
        lines.append(f"{ok} Ollama běží: {settings.ollama.base_url}")
        for model, available, kind, purpose in (
            (
                settings.ollama.embedding_model,
                report.embedding_model_available,
                *_MODEL_PURPOSE_TEXTS[ModelPurpose.EMBEDDING],
            ),
            (
                settings.ollama.chat_model,
                report.chat_model_available,
                *_MODEL_PURPOSE_TEXTS[ModelPurpose.CHAT],
            ),
        ):
            if available:
                lines.append(f"{ok} {kind} „{model}“ je nainstalovaný.")
            else:
                lines.append(
                    f"{error} {kind} „{model}“ není v Ollamě dostupný. {purpose} "
                    f"Nainstalujte ho příkazem: ollama pull {model}"
                )
    else:
        lines.append(
            f"{error} Nelze se spojit s Ollamou na adrese {settings.ollama.base_url}. "
            f"Ověřte, že Ollama běží. (Detail: {report.server_error})"
        )

    index_path = settings.index.path
    manifest = report.manifest
    match report.index_state:
        case IndexState.READY if manifest is not None:
            lines.append(
                f"{ok} Index: {index_path} – {manifest.speech_count} projevů, "
                f"{manifest.chunk_count} úseků, {len(manifest.speakers)} řečníků, "
                f"{len(manifest.parties)} stran; poslední ingest "
                f"{manifest.updated_at.astimezone():%d. %m. %Y %H:%M}"
            )
            if not report.index_matches_embedding_model:
                lines.append(
                    f"{error} Index byl vytvořen embedding modelem „{manifest.embedding_model}“, "
                    f"konfigurace používá „{settings.ollama.embedding_model}“. "
                    "Postavte ho znovu: python cli.py ingest <soubor> --rebuild"
                )
            if not report.index_matches_chunking:
                lines.append(
                    f"{warning} Index byl rozdělen s jinými parametry ({manifest.chunking}). "
                    "Dotazy fungují, ale další ingest vyžaduje --rebuild."
                )
        case IndexState.MISSING | IndexState.EMPTY | IndexState.READY:
            lines.append(
                f"{error} Index zatím neexistuje ({index_path}). "
                "Spusťte: python cli.py ingest data/speeches.jsonl"
            )
        case IndexState.LOCKED:
            lines.append(f"{warning} Index ({index_path}) právě používá jiný spuštěný příkaz.")
        case IndexState.UNREADABLE:
            lines.append(
                f"{error} Index nelze načíst: {report.index_detail}. "
                "Postavte ho znovu: python cli.py ingest <soubor> --rebuild"
            )
    lines.append(
        f"{ok} Kontextové okno modelu podle konfigurace: "
        f"{settings.generation.context_window_tokens} tokenů "
        "(musí odpovídat nastavení Ollamy)"
    )
    return "\n".join(lines)


def render_synthetic_report(
    *, generated: int, requested: int, discarded: int, path: Path, seconds: float
) -> str:
    """Return the Czech summary printed after generating synthetic questions."""
    lines = [
        f"Vygenerováno {generated} otázek (vyřazeno {discarded} nevhodných) → {path}, "
        f"za {_seconds(seconds / 60)} min."
    ]
    if generated < requested:
        lines.append(
            f"Upozornění: požadováno {requested} otázek, index ale neposkytl dost vhodných úseků."
        )
    return "\n".join(lines)
