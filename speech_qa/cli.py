"""Command-line interface, run as ``python cli.py <command>``.

User-facing texts are Czech (see :mod:`speech_qa.messages`); technical logs go
to stderr in English. Exit codes: 0 success, 1 invalid input, configuration or
data, 2 wrong usage, 3 model server unavailable or model missing, 130 interrupted.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from speech_qa import bootstrap
from speech_qa.config import Settings, find_config_file, load_settings
from speech_qa.diagnostics import run_checks
from speech_qa.domain.errors import (
    ModelNotAvailableError,
    ModelRequestError,
    ModelServiceUnavailableError,
    SpeechQAError,
)
from speech_qa.evaluation.loading import load_eval_sets
from speech_qa.evaluation.report import render_summary, write_reports
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.messages import render_check_report, render_error, render_ingest_report

DESCRIPTION = "Otázky a odpovědi nad archivem projevů (proof of concept)."

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_MODEL_SERVICE = 3
EXIT_INTERRUPTED = 130

logger = logging.getLogger(__name__)


class _CzechHelpFormatter(argparse.HelpFormatter):
    """Help formatter with a Czech usage prefix."""

    def add_usage(
        self,
        usage: str | None,
        actions: Iterable[argparse.Action],
        groups: Iterable[argparse._MutuallyExclusiveGroup],
        prefix: str | None = None,
    ) -> None:
        super().add_usage(usage, actions, groups, "použití: " if prefix is None else prefix)


class CzechArgumentParser(argparse.ArgumentParser):
    """Argument parser with Czech help texts; also used for sub-commands."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", _CzechHelpFormatter)
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self._positionals.title = "argumenty"
        self._optionals.title = "volby"
        self.add_argument("-h", "--help", action="help", help="zobrazí tuto nápovědu a skončí")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser of the command-line interface."""
    parser = CzechArgumentParser(prog="python cli.py", description=DESCRIPTION)
    commands = parser.add_subparsers(dest="command", title="příkazy", metavar="<příkaz>")

    ingest = commands.add_parser(
        "ingest",
        help="zaindexuje projevy ze souboru JSONL",
        description=(
            "Zaindexuje projevy ze souboru JSONL. Nové a změněné projevy se přidají, "
            "nezměněné se přeskočí."
        ),
    )
    add_common_options(ingest)
    ingest.add_argument(
        "path",
        type=Path,
        metavar="SOUBOR",
        help="soubor JSONL s projevy, např. data/speeches.jsonl",
    )
    ingest.add_argument(
        "--rebuild", action="store_true", help="smaže index a postaví ho znovu od nuly"
    )
    ingest.add_argument(
        "--strict",
        action="store_true",
        help="při prvním vadném řádku ingest zastaví (index zůstane beze změny)",
    )
    ingest.set_defaults(handler=_run_ingest)

    answer = commands.add_parser(
        "answer",
        help="odpoví na otázku z projevů v indexu",
        description=(
            "Odpoví na otázku v přirozeném jazyce z projevů v indexu a uvede zdroje. "
            "Index musí být vytvořen příkazem ingest."
        ),
    )
    add_common_options(answer)
    answer.add_argument("question", metavar="OTÁZKA", help="otázka, zapsaná v uvozovkách")
    answer.set_defaults(handler=_run_answer)

    check = commands.add_parser(
        "check",
        help="zkontroluje Ollamu, modely a index",
        description="Zkontroluje dostupnost Ollamy, obou modelů a stav indexu.",
    )
    add_common_options(check)
    check.set_defaults(handler=_run_check)

    evaluate = commands.add_parser(
        "evaluate",
        help="změří kvalitu vyhledávání a odpovědí na evaluačních sadách",
        description=(
            "Změří kvalitu vyhledávání a odpovědí na evaluačních sadách (CSV) a uloží report. "
            "Postup je popsán v docs/EVALUATION.md."
        ),
    )
    add_common_options(evaluate)
    evaluate.add_argument(
        "--manual",
        type=Path,
        default=Path("data/eval/manual.csv"),
        metavar="SOUBOR",
        help="ruční sada otázek (výchozí: data/eval/manual.csv)",
    )
    evaluate.add_argument(
        "--synthetic",
        type=Path,
        default=Path("data/eval/synthetic.csv"),
        metavar="SOUBOR",
        help="syntetická sada otázek (výchozí: data/eval/synthetic.csv)",
    )
    evaluate.add_argument(
        "--e2e-sample",
        type=int,
        default=20,
        metavar="N",
        help="kolik syntetických otázek projde celou odpovědí s modelem (výchozí: 20)",
    )
    evaluate.add_argument(
        "--no-judge", action="store_true", help="nehodnotí odpovědi jazykovým modelem"
    )
    evaluate.add_argument(
        "--retrieval-only",
        action="store_true",
        help="měří jen vyhledávání, bez generování odpovědí (rychlé)",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("reports"),
        metavar="SLOŽKA",
        help="kam uložit report (výchozí: reports/)",
    )
    evaluate.set_defaults(handler=_run_evaluate)
    return parser


def add_common_options(command: argparse.ArgumentParser) -> None:
    """Add the options every command accepts (--config, --verbose)."""
    command.add_argument(
        "--config",
        type=Path,
        metavar="SOUBOR",
        help="konfigurační soubor (výchozí: speech_qa.toml v aktuální složce)",
    )
    command.add_argument(
        "-v", "--verbose", action="store_true", help="vypisuje průběh a časy na stderr"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Command-line arguments without the program name; defaults to ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    return execute(args, args.handler)


def execute(
    args: argparse.Namespace, handler: Callable[[argparse.Namespace, Settings], int]
) -> int:
    """Load the settings, configure logging and run a command with error handling.

    Args:
        args: Parsed arguments; must contain ``config`` and ``verbose``.
        handler: The command.

    Returns:
        Process exit code.
    """
    try:
        settings = load_settings(args.config)
        _configure_logging(settings.logging.level, verbose=args.verbose)
        return handler(args, settings)
    except SpeechQAError as error:
        logger.debug("command failed", exc_info=True)
        print(render_error(error), file=sys.stderr)
        return _exit_code(error)
    except KeyboardInterrupt:
        print("Přerušeno uživatelem.", file=sys.stderr)
        return EXIT_INTERRUPTED


def _run_ingest(args: argparse.Namespace, settings: Settings) -> int:
    # The whole file is read and validated before the index is opened, so that
    # --strict never leaves a partially updated index behind.
    result = read_speeches(args.path, strict=args.strict)
    with bootstrap.open_index(settings, create=True) as index:
        service = bootstrap.create_ingestion_service(settings, index)
        report = service.ingest(result.speeches, result.issues, rebuild=args.rebuild)
    print(render_ingest_report(report, _display_path(settings.index.path)))
    return EXIT_OK


def _run_answer(args: argparse.Namespace, settings: Settings) -> int:
    with bootstrap.open_index(settings, create=False) as index:
        print(bootstrap.create_qa(settings, index).answer(args.question))
    return EXIT_OK


def _run_check(args: argparse.Namespace, settings: Settings) -> int:
    report = run_checks(settings, find_config_file(args.config))
    print(render_check_report(report))
    if not report.models_ok:
        return EXIT_MODEL_SERVICE
    return EXIT_OK if report.index_ok else EXIT_ERROR


def _run_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    with bootstrap.open_index(settings, create=False) as index:
        sets = load_eval_sets(args.manual, args.synthetic)
        runner = bootstrap.create_evaluation_runner(settings, index, judge=not args.no_judge)
        result = runner.run(sets, e2e_sample=args.e2e_sample, retrieval_only=args.retrieval_only)
        configuration = bootstrap.describe_configuration(settings, index)
    markdown, _ = write_reports(result, configuration, args.output)
    print(render_summary(result, markdown))
    return EXIT_OK


def _configure_logging(level: str, *, verbose: bool) -> None:
    configured = logging.getLevelNamesMapping()[level]
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    package_logger = logging.getLogger("speech_qa")
    package_logger.handlers[:] = [handler]
    package_logger.setLevel(min(configured, logging.INFO) if verbose else configured)
    package_logger.propagate = False


def _exit_code(error: SpeechQAError) -> int:
    if isinstance(error, ModelServiceUnavailableError | ModelNotAvailableError | ModelRequestError):
        return EXIT_MODEL_SERVICE
    return EXIT_ERROR


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
