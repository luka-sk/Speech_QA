"""Evaluation report: a Czech Markdown document, a JSON file and a terminal summary.

The Markdown report is meant for people (IT department, maintainers); the JSON
file keeps every per-question result for further analysis.
"""

import dataclasses
import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from speech_qa.evaluation.runner import EvaluationResult, QuestionResult

WORST_QUESTIONS = 10
"""How many problematic questions are listed in the report."""


def write_reports(
    result: EvaluationResult, configuration: Mapping[str, str], directory: Path
) -> tuple[Path, Path]:
    """Write the Markdown and JSON reports; returns their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = result.started_at.astimezone().strftime("%Y%m%d-%H%M%S")
    markdown = directory / f"evaluation-{stamp}.md"
    data = directory / f"evaluation-{stamp}.json"
    markdown.write_text(render_markdown(result, configuration), encoding="utf-8")
    payload = {"configuration": dict(configuration), **dataclasses.asdict(result)}
    data.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    return markdown, data


def render_markdown(result: EvaluationResult, configuration: Mapping[str, str]) -> str:
    """Render the Czech Markdown report."""
    started = result.started_at.astimezone()
    lines = [
        f"# Evaluace Speech QA – {started:%d. %m. %Y %H:%M}",
        "",
        f"Doba běhu: {_number(result.duration_seconds / 60, 1)} min. "
        f"Metriky vyhledávání se počítají na prvních k = {result.k} nalezených projevech, "
        "tedy na projevech vybraných pro jazykový model.",
        "",
        "## Konfigurace",
        "",
        *(f"- {name}: {value}" for name, value in configuration.items()),
        "",
        "## Vyhledávání",
        "",
        "| Sada | Otázek | S odpovědí v archivu | Hit@1 | Hit@5 | "
        f"Recall@{result.k} | MRR | nDCG@{result.k} | Řečník – přesnost | Řečník – úplnost |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {s.name} | {s.questions} | {s.answerable} | {_ratio(s.hit_at_1)} | "
        f"{_ratio(s.hit_at_5)} | {_ratio(s.recall_at_k)} | {_number(s.mrr, 2)} | "
        f"{_number(s.ndcg_at_k, 2)} | {_ratio(s.speaker_precision)} | {_ratio(s.speaker_recall)} |"
        for s in result.summaries
    ]
    lines += [
        "",
        "## Odpovědi",
        "",
        "| Sada | Zodpovězeno | Platné citace | Přesnost citací | Správné „nevím“ | "
        "Chybné „nevím“ | Věrnost (1–5) | Relevance (1–5) | Nehodnoceno |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {s.name} | {s.answered} | {_ratio(s.citation_validity)} | "
        f"{_ratio(s.citation_precision)} | {_ratio(s.correct_abstention)} | "
        f"{_ratio(s.false_abstention)} | {_number(s.faithfulness, 1)} | "
        f"{_number(s.relevance, 1)} | {s.judge_failures} |"
        for s in result.summaries
    ]
    lines += [
        "",
        "## Doba odezvy (s)",
        "",
        "| Sada | Celkem p50 | Celkem p95 | Vyhledávání p50 | Generování p50 |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {s.name} | {_number(s.latency_p50, 1)} | {_number(s.latency_p95, 1)} | "
        f"{_number(s.retrieval_p50, 2)} | {_number(s.generation_p50, 1)} |"
        for s in result.summaries
    ]
    lines += ["", "## Nejproblematičtější otázky", ""]
    worst = _worst(result.results)
    lines += [_describe(item) for item in worst] or ["Žádná otázka nevykazuje problém."]
    missing = {name: ids for name, ids in result.missing_ids.items() if ids}
    if missing:
        lines += ["", "## Relevantní projevy, které v indexu chybějí", ""]
        lines += [f"- {name}: {', '.join(ids)}" for name, ids in missing.items()]
    lines += [
        "",
        "## Omezení měření",
        "",
        "- Syntetické otázky jsou formulované blízko zdrojovému textu, výsledky na nich "
        "jsou proto optimistické. Každá má jediný relevantní projev, i když se téma může "
        "objevit i jinde, takže recall je naopak pesimistický.",
        "- Hodnocení věrnosti a relevance dělá stejný jazykový model, který odpovídá. "
        "Čísla jsou orientační.",
        "- U malých sad se metriky mohou mezi běhy výrazně lišit; u každé sady je uveden "
        "počet otázek.",
        "- Správné „nevím“ = odpověď bez citace u otázky, na kterou archiv odpověď nemá; "
        "chybné „nevím“ = odpověď bez citace u otázky, na kterou odpověď existuje.",
    ]
    return "\n".join(lines) + "\n"


def render_summary(result: EvaluationResult, markdown: Path) -> str:
    """Short Czech summary printed to the terminal."""
    lines = [f"Evaluace dokončena za {_number(result.duration_seconds / 60, 1)} min."]
    for s in result.summaries:
        lines.append(
            f"  {s.name}: {s.questions} otázek, Hit@5 {_ratio(s.hit_at_5)}, "
            f"MRR {_number(s.mrr, 2)}, Recall@{result.k} {_ratio(s.recall_at_k)}"
        )
        if s.answered:
            lines.append(
                f"    odpovědi: {s.answered}, přesnost citací {_ratio(s.citation_precision)}, "
                f"věrnost {_number(s.faithfulness, 1)}, relevance {_number(s.relevance, 1)}, "
                f"odezva p50 {_number(s.latency_p50, 1)} s"
            )
    lines.append(f"Report: {markdown} (+ .json)")
    return "\n".join(lines)


def _worst(results: tuple[QuestionResult, ...]) -> list[QuestionResult]:
    def problem(item: QuestionResult) -> float:
        score = 0.0
        rank = item.reciprocal_rank()
        if rank is not None:
            score += 1.0 - rank
        if item.answer is not None:
            if item.answer.abstained == item.question.answerable:
                score += 1.0
            if item.answer.verdict is not None:
                score += (5 - item.answer.verdict.faithfulness) / 4
        return score

    ranked = sorted(results, key=problem, reverse=True)
    return [item for item in ranked[:WORST_QUESTIONS] if problem(item) > 0]


def _describe(item: QuestionResult) -> str:
    question = item.question
    parts = [
        f"- **{item.set_name} / {question.question_id}**: {question.question}",
        f"  - relevantní: {', '.join(question.relevant_ids) or '(žádný)'}; "
        f"nalezeno: {', '.join(item.retrieved_ids[:5]) or '(nic)'}",
    ]
    if item.answer is not None:
        parts.append(f"  - citováno: {', '.join(item.answer.cited_ids) or '(nic)'}")
        if item.answer.verdict is not None:
            verdict = item.answer.verdict
            parts.append(
                f"  - hodnocení: věrnost {verdict.faithfulness}, relevance {verdict.relevance} "
                f"– {verdict.reason}"
            )
    return "\n".join(parts)


def _ratio(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f} %"


def _number(value: float | None, decimals: int) -> str:
    return "–" if value is None else f"{value:.{decimals}f}".replace(".", ",")


def _json_default(value: Any) -> Any:
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, Path | set | frozenset):
        return str(value) if isinstance(value, Path) else sorted(value)
    msg = f"cannot serialise {type(value).__name__}"
    raise TypeError(msg)
