# Speech QA – proof of concept

Question answering over a speech archive. The tool indexes speeches from a JSON
Lines file and answers natural-language questions (in Czech or English) with
references to the source speeches. Both the language model and the embedding
model run on-prem in [Ollama](https://ollama.com) and are called through the
standard OpenAI client; no API key is used.

```bash
python cli.py ingest data/speeches.jsonl
python cli.py answer "Co si myslí Doležal o vlivu kvality rozhodnutí na kvalitě informací?"
python cli.py answer "Co lidé diskutovali ke kvalitě rozhodování?"
```

Contents: [Quick start](#quick-start) · [Usage](#usage) · [Configuration](#configuration) ·
[How it works](#how-it-works) · [Key decisions and trade-offs](#key-decisions-and-trade-offs) ·
[Evaluation](#evaluation) · [Security](#security) · [Testing and quality](#testing-and-quality) ·
[Known limitations](#known-limitations) · [Next steps](#next-steps)

---

## Quick start

### Prerequisites

* **Python 3.12 or newer** (macOS, Linux or Windows).
* **Ollama** running on the same machine (default `http://localhost:11434`) with two models:

  ```bash
  ollama pull gemma4:12b   # language model that writes the answers
  ollama pull bge-m3       # embedding model (gemma4 cannot produce embeddings)
  ```

### Installation

With [uv](https://docs.astral.sh/uv/) (recommended; installs the exact locked versions):

```bash
uv sync                  # creates .venv with all dependencies, incl. development tools
uv run python cli.py check
```

With plain pip (the `requirements.txt` file pins every runtime dependency with hashes):

```bash
python -m venv .venv                 # Windows: py -3.12 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
                                     # Windows cmd:        .venv\Scripts\activate.bat
pip install -r requirements.txt
python cli.py check
```

All commands below assume an activated virtual environment (or the `uv run` prefix) and
the repository root as the current directory.

### First run

```bash
python cli.py check                       # Ollama, both models and the index
python cli.py ingest data/speeches.jsonl  # build the index (.speech_qa/index)
python cli.py answer "Co lidé diskutovali ke kvalitě rozhodování?"
```

Answering takes tens of seconds or more on a laptop; nearly all of the time is spent by
the language model. Add `--verbose` to see the timings of the individual steps.

### Switching to the real data

The tool always reads one data file, `data/speeches.jsonl` (the whole `data/` folder is
excluded from git). Replace the file with the real data and rebuild the index:

```bash
python cli.py ingest data/speeches.jsonl --rebuild
```

`--rebuild` matters here: an ordinary ingest only adds and updates speeches and never
deletes, so without it the previous test speeches would stay in the index.

---

## Usage

| Command | What it does |
|---|---|
| `python cli.py ingest FILE [--rebuild] [--strict]` | Validates and indexes speeches. New and changed speeches are embedded, unchanged ones are skipped. Invalid lines are skipped and listed; `--strict` stops at the first one and leaves the index untouched. |
| `python cli.py answer "QUESTION"` | Answers a question with references to the speeches. |
| `python cli.py check` | Checks the configuration, Ollama, both models and the index. |
| `python cli.py evaluate [options]` | Measures retrieval and answer quality, see [docs/EVALUATION.md](docs/EVALUATION.md). |
| `python -m speech_qa.evaluation synth [options]` | Generates synthetic evaluation questions, see [docs/EVALUATION.md](docs/EVALUATION.md). |

Every command accepts `--config FILE` (another configuration file) and `-v`/`--verbose`
(progress and timings on stderr); `--help` shows all options. Options are written after
the command name, e.g. `python cli.py ingest data/speeches.jsonl --verbose`.

### Input format

One JSON object per line, UTF-8:

```json
{"id": "Q000001", "speaker": "Tereza Dvořáková", "party": "ABC", "date": "2026-11-09", "text": "…"}
```

`id` must be unique and stable (letters, digits and `_ . : -`, at most 128 characters,
e.g. `ParlaMint-CZ_2020-11-13-ps2017-062-10-007-503.u10`), `date` is an ISO date and an
empty `party` means a speaker without a party. Unknown fields are ignored.

### Output of `answer`

The answer goes to stdout, technical logs and errors to stderr:

```
<answer text with citations [1], [2]>

Hledáno v projevech: <recognised speakers>      (only when a filter was applied)
Strany: <recognised parties>                     (only when a filter was applied)
Období: <d. m. yyyy> – <d. m. yyyy>              (only when a filter was applied)
Zdroje:
  [1] <speech id> – <speaker> (<party> | bez strany), <d. m. yyyy>
      „<excerpt of the best matching passage, at most 300 characters>“
  [2] …
```

* When no speech matches the question (e.g. a year with no speeches), the tool prints
  *„V archivu jsem nenašel žádné projevy odpovídající dotazu.“* without asking the model.
* When the answer cites nothing (the archive has no answer), it says so.
* Warnings are printed when the model cites a source it was not given, or when the prompt
  came close to the model's context window and may have been truncated.

### Exit codes

`0` success · `1` invalid input, configuration, data or index · `2` wrong usage ·
`3` Ollama unreachable or a model missing · `130` interrupted.

---

## Configuration

Settings live in [`speech_qa.toml`](speech_qa.toml); every value is documented there and
equals the built-in default. Any value can be overridden by an environment variable
`SPEECH_QA__<SECTION>__<KEY>` (e.g. `SPEECH_QA__OLLAMA__BASE_URL=http://gpu-server:11434/v1`),
and `SPEECH_QA_CONFIG=<path>` selects another file. Unknown keys and invalid values are
rejected at startup with a message naming the key.

| Section | Most important keys |
|---|---|
| `[ollama]` | `base_url`, `chat_model` (gemma4:12b), `embedding_model` (bge-m3), `timeout_seconds`, `max_retries` |
| `[generation]` | `temperature` (0), `reasoning_effort` ("none"), `max_answer_tokens`, `context_window_tokens` |
| `[index]` | `path` (.speech_qa/index), `collection` |
| `[chunking]` | `target_chars` (1000), `max_chars` (1500), `overlap_sentences` (1) |
| `[retrieval]` | `candidate_chunks` (50), `max_speeches` (8), `max_speeches_per_speaker` (2), `max_passages_per_speech` (2) |
| `[query]` | `max_chars` (1000) |
| `[logging]` | `level` |

**Context window.** `generation.context_window_tokens` (4096 by default) must match the
context length configured in Ollama, which cannot be set per request through the OpenAI
API. The prompt is kept below it; if you need more sources per answer, raise the context
length in Ollama (the context length setting of the Ollama app, or the
`OLLAMA_CONTEXT_LENGTH` environment variable of `ollama serve`) and then this value.
Larger contexts make answers slower.

Changing `chunking.*` or the embedding model requires `ingest --rebuild`; the other
settings take effect immediately.

---

## How it works

### Ingest

```
JSONL ──► validate lines ──► content hash ──► skip unchanged ──► sentence windows ──► bge-m3 ──► Qdrant
          (skip + report)    (id + content)                      (~1000 chars, 1 overlap)        + manifest
```

The manifest (`speech_qa_manifest.json` next to the Qdrant data) records the layout
version, the embedding model, the chunking parameters and the lists of speakers and
parties used by the query analyser.

### Answer

```
question ──► query analysis ──► embed the rest ──► search with filter ──► group by speech ──► prompt ──► gemma4 ──► citations ──► output
             speakers, party,    of the question    (50 chunks)            best 8 speeches,   numbered               checked and
             years → filter                                                 max 2 per speaker  sources [n]            mapped to ids
```

1. **Query analysis** (rules, no model): surnames of known speakers in any grammatical
   case ("Doležala", "s Doležalovou"), party codes and years ("v roce 2020", "od 2019 do 2021")
   become a filter. "Doležal" matches both Lukáš and Ondřej Doležal but never Doležalová.
2. **Retrieval**: the rest of the question is embedded; the most similar chunks that satisfy
   the filter are grouped by speech. Without a named speaker at most two speeches per speaker
   are kept, so a broad question is answered from several people.
3. **Generation**: the sources are given to the model as `[1]`, `[2]`, … with speaker, party
   and date, escaped and marked as quotations. The rules ask for answers only from the sources,
   precise attribution, an explicit note when a name is ambiguous and an honest "no answer".
4. **Citations** are mapped back to the speeches, renumbered in the order of their first use
   and checked; citations of sources the model was not given are reported.

### Architecture

Clean (hexagonal) architecture: dependencies point inwards, the core depends only on
interfaces, and every external system is an adapter.

```
speech_qa/
  domain/           entities and value objects (frozen dataclasses), ports, errors
  application/      use cases: ingestion, retrieval, prompting, SpeechQA (implements PQA)
  text/             language rules without I/O: sentences, Czech name forms, query analysis
  infrastructure/   adapters: JSONL reader, Ollama via the OpenAI client, Qdrant index
  evaluation/       evaluation sets, metrics, LLM judge, synthetic questions, reports
  prompts/          prompt templates (plain text files)
  config.py         TOML + environment configuration, validated with pydantic
  bootstrap.py      composition root: the only place that chooses concrete adapters
  messages.py       all Czech user-facing texts
  diagnostics.py    the `check` command
  cli.py            command-line interface (argparse)
cli.py              entry point required by the assignment
tests/              unit tests (fakes, in-memory Qdrant, simulated HTTP) and integration tests
```

| Port (`domain/ports.py`) | Adapter | Replaceable by |
|---|---|---|
| `Embedder` | `OpenAICompatibleEmbedder` (bge-m3 in Ollama) | any OpenAI-compatible embedding endpoint |
| `ChatModel` | `OpenAICompatibleChatModel` (gemma4:12b in Ollama) | any OpenAI-compatible chat endpoint |
| `SpeechIndex` | `QdrantSpeechIndex` (local mode) | Qdrant server, pgvector, … |
| `QueryAnalyzer` | `RuleBasedQueryAnalyzer` | an LLM-based or hybrid analyser |
| `PQA` (inbound, from the assignment) | `SpeechQA` | – |

* **Design patterns**: ports and adapters, repository (`SpeechIndex`), strategy
  (`QueryAnalyzer`, the model adapters), facade (`SpeechQA` behind `PQA`), composition root
  with constructor injection (`bootstrap.py`), immutable value objects, context managers
  for resources.
* **SOLID**: single responsibility per module (e.g. prompting, citation handling and output
  formatting are separate); open for extension through new adapters; adapters are
  substitutable behind their ports (tested with fakes); small, separate ports instead of one
  "AI service"; use cases depend on abstractions, wired only in the composition root.
* Errors are structured exceptions without user-facing text; `messages.py` turns them into
  Czech messages in one place.

---

## Key decisions and trade-offs

| Area | Decision | Why | Trade-off / alternative |
|---|---|---|---|
| Embeddings | `bge-m3` in Ollama | gemma4 cannot embed; bge-m3 is multilingual and handles Czech well | one more model to install |
| Generation | gemma4:12b, temperature 0, thinking disabled (`reasoning_effort = "none"`) | repeatable answers; thinking mode made answers several times slower | answers are less elaborate on complex questions |
| Vector store | Qdrant in local mode (a directory, no server) | no infrastructure for the PoC; the same API works with a Qdrant server later | local mode allows one process at a time; a server is needed for parallel use |
| Chunking | windows of whole sentences, ~1000 characters, 1 sentence overlap | long speeches mix topics; smaller windows retrieve precisely, the overlap keeps context at boundaries | parameters should be tuned on the real data (see the evaluation guide) |
| Ingest | incremental by id + content hash; `--rebuild` starts over | re-running is cheap and safe; an interrupted run can simply be repeated | nothing is ever deleted without `--rebuild` |
| Invalid input | invalid lines are skipped and listed; `--strict` stops | one bad line should not block a large ingest | strict mode for pipelines that must be clean |
| Query understanding | rules: Czech surname forms, party codes, years | instant, deterministic, testable; filters guarantee that a speaker's own speeches are searched | misses unusual phrasings; an LLM fallback is a next step |
| Name matching | without diacritics, only capitalised words | "Dolezal" is understood; common words ("svoboda", "král") never become filters | names typed in lower case are not recognised |
| No fallback search | a filter without matches returns nothing | searching without the filter would attribute other people's words to the named speaker | the user may need to rephrase |
| Diversity | at most 2 speeches per speaker unless a speaker is named | broad questions are answered from several people | a very active speaker may be under-represented |
| Source labels | the model sees `[1]`, `[2]`; ids are mapped back | real ids are long and models copy short numbers reliably | – |
| Prompt safety | sources escaped and marked as quotations; the model is told to ignore instructions in them | speech texts are untrusted data | no defence is perfect against prompt injection |
| Context budget | sources are added until an estimated budget (characters / 3) is used; a warning when the reported prompt size reaches 95 % of the window | depending on its version and settings, Ollama may silently cut prompts longer than its context length | fewer sources with a small context window |
| Nothing found | a fixed message, the model is not asked | the model cannot invent an answer | – |
| Language | Czech user texts, English code, logs and documentation | users are Czech, code follows common practice | – |
| Configuration | TOML file + environment overrides, validated, unknown keys rejected | readable for testers on every OS, scriptable for deployment | – |
| Packaging | flat package next to `cli.py`; `uv.lock` + `requirements.txt` with hashes; Python ≥ 3.12; argparse | `python cli.py` works exactly as required; reproducible installs; no CLI dependency | – |
| Evaluation | manual set + generated synthetic set, LLM-as-judge, metrics at k = max sources | numbers without labelling hundreds of questions by hand | synthetic questions are optimistic; the judge is the same model |

---

## Evaluation

The evaluation measures retrieval (Hit@k, Recall@k, MRR, nDCG@k, speaker recognition) and
answers (citation validity and precision, correct "no answer", latency, faithfulness and
relevance rated by the language model). The step-by-step guide, including how to create the
manual question set and how to tune the settings, is in
**[docs/EVALUATION.md](docs/EVALUATION.md)**.

```bash
python -m speech_qa.evaluation synth            # synthetic questions → data/eval/synthetic.csv
python cli.py evaluate                          # report → reports/evaluation-<date-time>.md
```

### Results

*To be filled in from `python cli.py evaluate` on the real data.*

| |                                                             |
|---|-------------------------------------------------------------|
| Date | 13. 11. 2026                                                |
| Data | 2430 speeches, 3561 chunks, 412 speakers                    |
| Hardware / Ollama | gemma4:12b, bge-m3                                          |
| Evaluation sets | manual: - questions, synthetic: 150 questions (20 answered) |

**Retrieval** (k = 8 sources passed to the model)

| Set | Hit@1 | Hit@5 | Recall@8 | MRR  | nDCG@8 | Speaker precision | Speaker recall |
|---|------|------|---------|------|--------|-------------------|----------------|
| manual | … | … | … | … | … | … | … |
| synthetic | 80 % | 91 % | 92 % | 0.84 | 0.86 | 89 % | 100 % |

**Answers**

| Set | Valid citations | Citation precision | Correct "no answer" | False "no answer" | Faithfulness (1–5) | Relevance (1–5) |
|---|-----------------|--------------------|---------------------|-------------------|--------------------|-----------------|
| manual | …               | …                  | …                   | …                 | …                  | …               |
| synthetic | 100 % | 65 % | - | 0 % | 4.8 | 5.0 |

**Latency** (seconds)

| Set | Total p50 | Total p95 | Retrieval p50 | Generation p50 |
|---|-----------|-----------|---------------|----------------|
| manual | …         | …         | …             | …              |
| synthetic | 52.7 | 79.3 | 0.07 | 52.6 |

---

## Security

* **On-prem only**: the tool contacts only the configured Ollama endpoint; no API key and
  no secrets in the code or configuration.
* **Input validation**: every input line is validated (types, lengths, id format, 1 MB line
  limit); questions are limited to 1000 characters; the configuration is validated and
  unknown keys are rejected.
* **Prompt injection**: speech texts are escaped, wrapped as quotations and the model is told
  to ignore instructions inside them; citations are checked against the sources actually given.
* **Output**: control characters (e.g. terminal escape sequences) are removed from model
  output before printing.
* **Code and dependencies**: no `eval`, `pickle` or shell calls; ruff security rules (bandit)
  in the lint; `pip-audit` checks dependencies for known vulnerabilities; locked versions with
  hashes; the index manifest is written atomically with owner-only permissions.
* **Data**: the `data/` folder (including evaluation sets derived from the data) and reports
  are excluded from git.

---

## Testing and quality

```bash
uv run pytest                          # unit tests (no Ollama needed), coverage ≥ 85 % enforced
uv run pytest -m integration --no-cov  # integration tests against the running Ollama (minutes)
uv run ruff check . && uv run ruff format --check .
uv run mypy                            # strict typing
uv run pip-audit                       # known vulnerabilities in dependencies
```

Unit tests use a fake embedder and language model, an in-memory Qdrant and simulated HTTP;
small long-text fixtures are in `tests/fixtures/`. Integration tests ingest the fixtures
with the real models and check retrieval, answers (including both assignment questions),
the evaluation tools and `check`. The suite runs on Python 3.12 and 3.13.

---

## Known limitations

* **Speaker recognition**: names written in lower case are not recognised; a capitalised word
  at the start of a sentence can be taken for a surname ("Svoboda je…"); the male plural
  ("Doležalové") equals a female case form and is understood as the female speakers; first
  names whose consonant changes in a case ("Lenka" → "Lence") do not narrow the match; parties
  are recognised only by their exact code, not in inflected or colloquial forms ("Pirátů");
  relative periods ("loni") are not recognised. Functions such as "ministr financí" cannot be
  mapped to people, the data contain no roles.
* **Language model**: gemma4:12b occasionally makes Czech spelling mistakes and may
  misinterpret a passage; it does not always point out a wrong premise in the question.
* **No relevance threshold**: up to 8 sources are always passed to the model, which decides
  whether they answer the question.
* **Local Qdrant**: one process at a time; a second command (e.g. two `answer` runs, or
  `answer` during `evaluate`) reports that the index is in use. qdrant-client in local mode
  leaves an internal SQLite connection open (a warning on Python 3.13, filtered in the tests).
* **Speed**: answers take tens of seconds on a laptop; evaluation with the language model
  takes a long time (see the evaluation guide).
* **Synthetic evaluation questions** resemble the source text and have a single relevant
  speech each; the LLM judge is the model that writes the answers.
* **Limited Number of Sources for Long Speeches**: The retrieval system selects 8 speeches. 
  For long speeches, the content may not fit within the default context window, which has 
  a limited token capacity. This can affect evaluation metrics (Recall@8, nDCG@8).

---

## Next steps

* **Query understanding**: hybrid analysis (rules first, the language model only for
  unrecognised names or relative periods such as "loni"); a register of functions with
  periods ("ministr financí 2019–2021"); Czech lemmatisation.
* **Retrieval**: hybrid search (BM25 + embeddings), a reranker, a relevance threshold
  calibrated on the real data; tuning of chunking and limits on the real data.
* **Operations**: Qdrant server (with payload indexes) for parallel use; a REST API next to
  the CLI (the application layer is ready for it); Docker image; optional query logging with
  a log file and JSON logs; `--json` output for scripts.
* **Evaluation**: a helper suggesting candidate speeches for manual labelling
  (`suggest` / `import-labels`); comparison with the previous run; ablations (chunk size,
  query analysis, diversity); a recommended configuration in the report; a report without
  data for sharing (`--redact`); a judge independent of the answering model; a solution 
  addressing context window token limitations when the 8 retrieved speeches are too long.
* **Development**: pre-commit hooks and CI once the repository location is known; report
  the unclosed SQLite connection to qdrant-client.
