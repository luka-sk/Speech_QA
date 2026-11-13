"""The interfaces cannot be used without implementing every abstract method."""

from abc import ABC

import pytest

from speech_qa.application.pqa import PQA
from speech_qa.domain.ports import ChatModel, Embedder, QueryAnalyzer, SpeechIndex


@pytest.mark.parametrize("interface", [PQA, Embedder, ChatModel, SpeechIndex, QueryAnalyzer])
def test_interfaces_are_abstract(interface: type[ABC]) -> None:
    with pytest.raises(TypeError, match="abstract"):
        interface()


def test_pqa_implementation_answers_queries() -> None:
    class EchoQA(PQA):
        def answer(self, query: str) -> str:
            return f"Odpověď na: {query}"

    assert EchoQA().answer("Co?") == "Odpověď na: Co?"
