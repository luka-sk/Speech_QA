import datetime as dt
import json
import logging
import os
import runpy
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from speech_qa import bootstrap
from speech_qa.application.ingestion import IngestionService
from speech_qa.application.qa import SpeechQA
from speech_qa.application.retrieval import Retriever
from speech_qa.cli import EXIT_ERROR, EXIT_INTERRUPTED, EXIT_MODEL_SERVICE, EXIT_OK, main
from speech_qa.config import Settings, load_settings
from speech_qa.diagnostics import CheckReport, IndexState
from speech_qa.domain.errors import IndexNotFoundError, ModelNotAvailableError
from speech_qa.domain.models import ChatMessage, ChunkingParameters, IndexManifest, ModelPurpose
from speech_qa.domain.ports import SpeechIndex
from speech_qa.evaluation.__main__ import main as evaluation_main
from speech_qa.evaluation.judge import LLMJudge
from speech_qa.evaluation.runner import EvaluationRunner
from speech_qa.evaluation.synthetic import SyntheticQuestionGenerator
from speech_qa.infrastructure.jsonl_source import read_speeches
from speech_qa.infrastructure.qdrant_index import QdrantSpeechIndex
from speech_qa.messages import render_answer
from speech_qa.text.query_analysis import RuleBasedQueryAnalyzer
from tests.conftest import FIXTURES, LONG_SPEECHES, FakeChatModel, FakeEmbedder

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run in an empty directory with an index there and no environment overrides."""
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.upper().startswith("SPEECH_QA"):
            monkeypatch.delenv(name)
    return tmp_path


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeEmbedder]:
    """Replace the Ollama embedder with a fake one."""
    embedder = FakeEmbedder("bge-m3")

    def create_service(settings: Settings, index: SpeechIndex) -> IngestionService:
        return IngestionService(index, embedder, bootstrap.create_chunker(settings), batch_size=8)

    monkeypatch.setattr(bootstrap, "create_ingestion_service", create_service)
    yield embedder
    logging.getLogger("speech_qa").handlers.clear()


def test_without_arguments_prints_czech_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert output.startswith("použití: python cli.py")
    assert "Otázky a odpovědi nad archivem projevů" in output
    assert "zobrazí tuto nápovědu a skončí" in output
    assert "ingest" in output


def test_help_option_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])

    assert caught.value.code == 0
    assert "použití:" in capsys.readouterr().out


def test_ingest_help_is_czech(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["ingest", "--help"])

    output = capsys.readouterr().out
    assert "argumenty:" in output
    assert "volby:" in output
    assert "--rebuild" in output


def test_root_entry_point_runs_the_cli(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["cli.py"])

    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(REPOSITORY_ROOT / "cli.py"), run_name="__main__")

    assert caught.value.code == 0
    assert "použití: python cli.py" in capsys.readouterr().out


def test_ingest_indexes_the_file_and_prints_a_summary(
    isolated: Path, fake_models: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["ingest", str(LONG_SPEECHES)])

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "Projevy v souboru: 10 (nové 10, změněné 0, beze změny 0)" in output
    assert "Index: .speech_qa/index" in output
    with QdrantSpeechIndex.open_local(
        isolated / ".speech_qa/index", "speech_chunks", create=False
    ) as index:
        assert len(index.indexed_speeches()) == 10


def test_repeated_ingest_skips_unchanged_speeches(
    isolated: Path, fake_models: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["ingest", str(LONG_SPEECHES)])
    capsys.readouterr()

    assert main(["ingest", str(LONG_SPEECHES), "--verbose"]) == EXIT_OK

    captured = capsys.readouterr()
    assert "(nové 0, změněné 0, beze změny 10)" in captured.out


def test_rebuild_is_reported(
    isolated: Path, fake_models: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ingest", str(LONG_SPEECHES), "--rebuild"]) == EXIT_OK

    assert "Index byl postaven znovu od nuly." in capsys.readouterr().out


def test_invalid_lines_are_listed_but_do_not_stop_the_ingest(
    isolated: Path, fake_models: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    data = isolated / "data.jsonl"
    good = {"id": "Q1", "speaker": "A B", "party": "", "date": "2020-01-01", "text": "Věta."}
    data.write_text(json.dumps(good) + "\nnot json\n", encoding="utf-8")

    assert main(["ingest", str(data)]) == EXIT_OK

    output = capsys.readouterr().out
    assert "Vadné řádky (přeskočeny): 1" in output
    assert "  - řádek 2: neplatný JSON" in output


def test_strict_ingest_stops_and_leaves_no_index(
    isolated: Path, fake_models: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    data = isolated / "data.jsonl"
    data.write_text("not json\n", encoding="utf-8")

    assert main(["ingest", str(data), "--strict"]) == EXIT_ERROR

    assert "přepínač --strict" in capsys.readouterr().err
    assert not (isolated / ".speech_qa").exists()


def test_missing_input_file_is_reported(
    isolated: Path, fake_models: FakeEmbedder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ingest", "missing.jsonl"]) == EXIT_ERROR

    assert "Soubor missing.jsonl neexistuje." in capsys.readouterr().err


def test_missing_embedding_model_is_reported_with_exit_code_3(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class MissingModelEmbedder(FakeEmbedder):
        def embed_documents(self, texts: object) -> list[list[float]]:  # type: ignore[override]
            raise ModelNotAvailableError("bge-m3", ModelPurpose.EMBEDDING)

    def create_service(settings: Settings, index: SpeechIndex) -> IngestionService:
        return IngestionService(
            index, MissingModelEmbedder("bge-m3"), bootstrap.create_chunker(settings)
        )

    monkeypatch.setattr(bootstrap, "create_ingestion_service", create_service)

    assert main(["ingest", str(LONG_SPEECHES)]) == EXIT_MODEL_SERVICE

    assert "Je nutný pro výpočet embeddingů." in capsys.readouterr().err


def test_invalid_configuration_is_reported(
    isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (isolated / "speech_qa.toml").write_text("[ollama]\nbse_url = 'x'\n", encoding="utf-8")

    assert main(["ingest", str(LONG_SPEECHES)]) == EXIT_ERROR

    assert "Chybná konfigurace" in capsys.readouterr().err


def test_index_outside_the_working_directory_is_shown_with_its_full_path(
    isolated: Path,
    fake_models: FakeEmbedder,
    tmp_path_factory: pytest.TempPathFactory,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "index"
    monkeypatch.setenv("SPEECH_QA__INDEX__PATH", str(elsewhere))

    assert main(["ingest", str(LONG_SPEECHES)]) == EXIT_OK

    assert f"Index: {elsewhere}" in capsys.readouterr().out


def test_interrupt_is_reported(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("speech_qa.cli.read_speeches", interrupt)

    assert main(["ingest", str(LONG_SPEECHES)]) == EXIT_INTERRUPTED
    assert "Přerušeno" in capsys.readouterr().err


def test_bootstrap_wires_the_real_adapters(isolated: Path) -> None:
    (isolated / "speech_qa.toml").write_text("[index]\npath = 'idx'\n", encoding="utf-8")
    settings = load_settings(environ={}, cwd=isolated)

    with bootstrap.open_index(settings, create=True) as index:
        assert isinstance(index, QdrantSpeechIndex)
        assert isinstance(bootstrap.create_ingestion_service(settings, index), IngestionService)
    assert (isolated / "idx").is_dir()
    assert bootstrap.create_embedder(settings).model_name == "bge-m3"


# --------------------------------------------------------------------------- #
# answer and check
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_answering(monkeypatch: pytest.MonkeyPatch, fake_models: FakeEmbedder) -> FakeChatModel:
    """Answer with a fake language model over the real (temporary) index."""
    chat = FakeChatModel("Doležalové mluvili o datech [1].")

    def create_qa(settings: Settings, index: SpeechIndex) -> SpeechQA:
        manifest = index.read_manifest()
        if manifest is None:
            raise IndexNotFoundError(settings.index.path)
        retriever = Retriever(
            RuleBasedQueryAnalyzer(manifest.speakers, manifest.parties), fake_models, index
        )
        return SpeechQA(retriever, chat, render_answer)

    monkeypatch.setattr(bootstrap, "create_qa", create_qa)
    return chat


def test_answer_prints_the_answer_with_sources(
    isolated: Path, fake_answering: FakeChatModel, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["ingest", str(LONG_SPEECHES)])
    capsys.readouterr()

    exit_code = main(["answer", "Co si myslí Doležal o kvalitě informací?"])

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert output.startswith("Doležalové mluvili o datech [1].")
    assert "Hledáno v projevech: Lukáš Doležal, Ondřej Doležal" in output
    assert "Zdroje:\n  [1] F-00" in output


def test_answer_without_index_tells_to_ingest_first(
    isolated: Path, fake_answering: FakeChatModel, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["answer", "Otázka?"]) == EXIT_ERROR

    assert "Nejdřív spusťte: python cli.py ingest data/speeches.jsonl" in capsys.readouterr().err
    assert not (isolated / ".speech_qa").exists()


def test_empty_question_is_rejected(
    isolated: Path, fake_answering: FakeChatModel, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["ingest", str(LONG_SPEECHES)])

    assert main(["answer", "   "]) == EXIT_ERROR
    assert "Otázka je prázdná." in capsys.readouterr().err
    assert fake_answering.calls == []


def test_bootstrap_creates_the_question_answering_service(isolated: Path) -> None:
    settings = load_settings(environ={}, cwd=isolated)
    with bootstrap.open_index(settings, create=True) as index:
        IngestionService(index, FakeEmbedder("bge-m3"), bootstrap.create_chunker(settings)).ingest(
            read_speeches(LONG_SPEECHES).speeches
        )

        assert isinstance(bootstrap.create_qa(settings, index), SpeechQA)
    assert bootstrap.create_chat_model(settings).model_name == "gemma4:12b"


@pytest.mark.parametrize(
    ("models_ok", "index_ok", "expected"),
    [(True, True, EXIT_OK), (True, False, EXIT_ERROR), (False, True, EXIT_MODEL_SERVICE)],
)
def test_check_exit_code(
    isolated: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    models_ok: bool,
    index_ok: bool,
    expected: int,
) -> None:
    def fake_checks(settings: Settings, config_file: Path | None) -> CheckReport:
        installed = frozenset({"bge-m3:latest", "gemma4:12b"}) if models_ok else frozenset()
        return CheckReport(
            config_file=config_file,
            settings=settings,
            server_error=None,
            installed_models=installed,
            index_state=IndexState.READY if index_ok else IndexState.MISSING,
            index_detail="",
            manifest=_manifest() if index_ok else None,
        )

    monkeypatch.setattr("speech_qa.cli.run_checks", fake_checks)

    assert main(["check"]) == expected
    assert "Konfigurace:" in capsys.readouterr().out


def _manifest() -> IndexManifest:
    return IndexManifest(
        schema_version=1,
        embedding_model="bge-m3",
        embedding_dimension=1024,
        chunking=ChunkingParameters(1000, 1500, 1),
        speakers=("A B",),
        parties=(),
        speech_count=1,
        chunk_count=1,
        updated_at=dt.datetime(2026, 11, 12, tzinfo=dt.UTC),
    )


# --------------------------------------------------------------------------- #
# evaluate and synth
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_evaluation(monkeypatch: pytest.MonkeyPatch, fake_models: FakeEmbedder) -> None:
    """Evaluate and generate questions with fake models over the real (temporary) index."""

    def create_runner(settings: Settings, index: SpeechIndex, *, judge: bool) -> EvaluationRunner:
        manifest = index.read_manifest()
        assert manifest is not None
        retriever = Retriever(
            RuleBasedQueryAnalyzer(manifest.speakers, manifest.parties), fake_models, index
        )
        qa = SpeechQA(retriever, FakeChatModel("Ano [1]."), render_answer)
        return EvaluationRunner(
            retriever,
            qa,
            LLMJudge(FakeChatModel('{"faithfulness": 5, "relevance": 5}')) if judge else None,
            k=8,
            known_ids=index.indexed_speeches().keys(),
        )

    def create_generator(settings: Settings, index: SpeechIndex) -> SyntheticQuestionGenerator:
        manifest = index.read_manifest()
        assert manifest is not None
        chat = FakeChatModel(question_naming_the_speaker_when_asked)
        return SyntheticQuestionGenerator(
            index, chat, RuleBasedQueryAnalyzer(manifest.speakers, manifest.parties)
        )

    monkeypatch.setattr(bootstrap, "create_evaluation_runner", create_runner)
    monkeypatch.setattr(bootstrap, "create_synthetic_generator", create_generator)


def test_evaluate_writes_a_report(
    isolated: Path, fake_evaluation: None, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["ingest", str(LONG_SPEECHES)])
    capsys.readouterr()
    manual = FIXTURES / "eval_manual.csv"

    exit_code = main(["evaluate", "--manual", str(manual), "--output", "out"])

    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "manual: 7 otázek" in output
    assert len(list((isolated / "out").glob("evaluation-*.md"))) == 1
    assert len(list((isolated / "out").glob("evaluation-*.json"))) == 1


def test_evaluate_without_sets_is_an_error(
    isolated: Path, fake_evaluation: None, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["ingest", str(LONG_SPEECHES)])

    assert main(["evaluate"]) == EXIT_ERROR
    assert "Evaluační soubor" in capsys.readouterr().err


def test_synth_generates_questions_with_overwrite_protection(
    isolated: Path, fake_evaluation: None, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["ingest", str(LONG_SPEECHES)])
    capsys.readouterr()

    assert evaluation_main(["synth", "--count", "2", "--output", "syn.csv"]) == EXIT_OK
    assert "Vygenerováno 0 otázek" not in capsys.readouterr().out
    assert evaluation_main(["synth", "--count", "2", "--output", "syn.csv"]) == EXIT_ERROR
    assert "už existuje" in capsys.readouterr().err
    assert evaluation_main(["synth", "--count", "2", "--output", "syn.csv", "--force"]) == EXIT_OK


def test_synth_reports_when_too_few_questions_were_generated(
    isolated: Path, fake_evaluation: None, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["ingest", str(LONG_SPEECHES)])
    capsys.readouterr()

    assert evaluation_main(["synth", "--count", "500", "--output", "syn.csv"]) == EXIT_OK

    assert "Upozornění: požadováno 500 otázek" in capsys.readouterr().out


def test_synth_help_and_invalid_count(capsys: pytest.CaptureFixture[str]) -> None:
    assert evaluation_main([]) == EXIT_OK
    assert "python -m speech_qa.evaluation" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        evaluation_main(["synth", "--count", "0"])
    assert "kladné celé číslo" in capsys.readouterr().err


def test_bootstrap_creates_evaluation_components(isolated: Path) -> None:
    settings = load_settings(environ={}, cwd=isolated)
    with bootstrap.open_index(settings, create=True) as index:
        with pytest.raises(IndexNotFoundError):
            bootstrap.create_synthetic_generator(settings, index)
        IngestionService(index, FakeEmbedder("bge-m3"), bootstrap.create_chunker(settings)).ingest(
            read_speeches(LONG_SPEECHES).speeches
        )

        assert isinstance(
            bootstrap.create_evaluation_runner(settings, index, judge=True), EvaluationRunner
        )
        assert isinstance(
            bootstrap.create_synthetic_generator(settings, index), SyntheticQuestionGenerator
        )
        description = bootstrap.describe_configuration(settings, index)
    assert description["Jazykový model"] == "gemma4:12b"
    assert description["Index"].startswith("10 projevů")


def question_naming_the_speaker_when_asked(messages: Sequence[ChatMessage]) -> str:
    system, user = messages
    if "Mention the speaker" in system.content:
        surname = user.content.split("\n", 1)[0].split()[-1]
        return json.dumps({"question": f"Jaký postoj zastává {surname} k tématu?"})
    return json.dumps({"question": "Jaké argumenty zazněly k tématu?"})
