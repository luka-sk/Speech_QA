"""Choosing the evaluation sets for a run."""

import logging
from pathlib import Path

from speech_qa.domain.errors import EvaluationDataError
from speech_qa.evaluation.dataset import EvalSet, read_eval_set
from speech_qa.evaluation.runner import SYNTHETIC_SET

MANUAL_SET = "manual"

logger = logging.getLogger(__name__)


def load_eval_sets(manual: Path, synthetic: Path) -> list[EvalSet]:
    """Read the manual and synthetic sets; a missing file is skipped.

    Raises:
        EvaluationDataError: If neither file exists, or a file is invalid.
    """
    sets = []
    for name, path in ((MANUAL_SET, manual), (SYNTHETIC_SET, synthetic)):
        if path.is_file():
            sets.append(read_eval_set(path, name))
        else:
            logger.warning("evaluation set %s not found at %s, skipped", name, path)
    if not sets:
        raise EvaluationDataError(
            manual, [f"no evaluation set found (looked for {manual} and {synthetic})"]
        )
    return sets
