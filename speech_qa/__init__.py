"""Speech QA: question answering over a speech archive.

Package layout (dependencies point inwards only)::

    domain/          entities, value objects, ports (interfaces) and errors
    application/     use cases: ingestion, retrieval, answering
    text/            language utilities: sentence splitting, Czech name forms,
                     rule-based query analysis
    infrastructure/  adapters: JSONL reader, Ollama (OpenAI client), Qdrant
    prompts/         prompt templates
    evaluation/      evaluation tooling (python -m speech_qa.evaluation)
    config.py        configuration loading and validation
    bootstrap.py     composition root, wires adapters to use cases
    cli.py           command-line interface
"""

__version__ = "0.1.0"
