"""Preparation of evaluation sets: ``python -m speech_qa.evaluation <command>``.

The measurement itself is ``python cli.py evaluate``; see docs/EVALUATION.md.
"""

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from speech_qa import bootstrap
from speech_qa.cli import EXIT_OK, CzechArgumentParser, add_common_options, execute
from speech_qa.config import Settings
from speech_qa.domain.errors import OutputExistsError
from speech_qa.evaluation.dataset import write_eval_set
from speech_qa.messages import render_synthetic_report


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser of the preparation tool."""
    parser = CzechArgumentParser(
        prog="python -m speech_qa.evaluation",
        description="Příprava evaluačních sad. Postup je popsán v docs/EVALUATION.md.",
    )
    commands = parser.add_subparsers(dest="command", title="příkazy", metavar="<příkaz>")
    synth = commands.add_parser(
        "synth",
        help="vygeneruje syntetickou sadu otázek z indexu",
        description=(
            "Jazykový model vytvoří z náhodně vybraných úseků indexu otázky se známou "
            "správnou odpovědí a uloží je jako CSV."
        ),
    )
    add_common_options(synth)
    synth.add_argument(
        "--count", type=_positive_int, default=150, metavar="N", help="počet otázek (výchozí: 150)"
    )
    synth.add_argument(
        "--seed", type=int, default=42, help="semínko náhodného výběru úseků (výchozí: 42)"
    )
    synth.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/synthetic.csv"),
        metavar="SOUBOR",
        help="výstupní soubor (výchozí: data/eval/synthetic.csv)",
    )
    synth.add_argument("--force", action="store_true", help="přepíše existující výstupní soubor")
    synth.set_defaults(handler=_run_synth)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preparation tool; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    return execute(args, args.handler)


def _run_synth(args: argparse.Namespace, settings: Settings) -> int:
    if args.output.exists() and not args.force:
        raise OutputExistsError(args.output)  # before spending minutes on generation
    started = time.perf_counter()
    with bootstrap.open_index(settings, create=False) as index:
        generator = bootstrap.create_synthetic_generator(settings, index)
        result = generator.generate(args.count, seed=args.seed)
    write_eval_set(args.output, result.questions, force=args.force)
    print(
        render_synthetic_report(
            generated=len(result.questions),
            requested=args.count,
            discarded=result.discarded,
            path=args.output,
            seconds=time.perf_counter() - started,
        )
    )
    return EXIT_OK


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        msg = "musí být kladné celé číslo"
        raise argparse.ArgumentTypeError(msg)
    return number


if __name__ == "__main__":
    raise SystemExit(main())
