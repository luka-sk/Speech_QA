"""The public question-answering interface given by the assignment."""

from abc import ABC, abstractmethod


class PQA(ABC):
    """Answers natural-language questions about the speech archive."""

    @abstractmethod
    def answer(self, query: str) -> str:
        """Answer a natural language query.

        Args:
            query: The user's natural-language question.

        Returns:
            A natural-language answer including references to the source
            speeches where possible.
        """
