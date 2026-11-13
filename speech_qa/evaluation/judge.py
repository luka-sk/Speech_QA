"""LLM-as-judge: the language model rates answers for faithfulness and relevance.

The judge is the same model that writes the answers, so its ratings tend to be
lenient towards its own style; they are indicative, not ground truth.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from string import Template

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from speech_qa.application.prompting import render_source
from speech_qa.domain.models import ChatMessage, Evidence
from speech_qa.domain.ports import ChatModel

logger = logging.getLogger(__name__)

_PROMPTS = resources.files("speech_qa.prompts")
_SYSTEM_PROMPT = _PROMPTS.joinpath("judge_system.txt").read_text(encoding="utf-8").strip()
_USER_TEMPLATE = Template(_PROMPTS.joinpath("judge_user.txt").read_text(encoding="utf-8").strip())


class _VerdictSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Ratings of one answer (1 = worst, 5 = best)."""

    faithfulness: int
    relevance: int
    reason: str


class LLMJudge:
    """Rates answers with a language model."""

    def __init__(self, chat_model: ChatModel) -> None:
        """Create the judge.

        Args:
            chat_model: The model that rates the answers.
        """
        self._chat_model = chat_model

    def judge(self, question: str, answer: str, sources: Sequence[Evidence]) -> JudgeVerdict | None:
        """Rate an answer; ``None`` if the model's verdict cannot be understood."""
        rendered = "\n\n".join(
            render_source(number, source) for number, source in enumerate(sources, start=1)
        )
        user = _USER_TEMPLATE.substitute(
            question=question, sources=rendered or "(no sources)", answer=answer
        )
        completion = self._chat_model.complete(
            (ChatMessage("system", _SYSTEM_PROMPT), ChatMessage("user", user)),
            max_tokens=300,
            json_output=True,
        )
        try:
            verdict = _VerdictSchema.model_validate(json.loads(completion.text))
        except (json.JSONDecodeError, ValidationError):
            logger.warning("the judge returned an unusable verdict: %r", completion.text[:200])
            return None
        return JudgeVerdict(verdict.faithfulness, verdict.relevance, verdict.reason)
